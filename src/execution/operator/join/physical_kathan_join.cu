//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_kathan_join.cu
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/join/physical_kathan_join.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/operator/logical_join.hpp"

#include <warpcore/single_value_hash_table.cuh>
#include <cuda_runtime.h>
#include <cstdio>

namespace duckdb {

using key_t = uint32_t;
using value_t = uint32_t;
using hash_table_t = warpcore::SingleValueHashTable<key_t, value_t>;

//===--------------------------------------------------------------------===//
// Sink States (build side only)
//===--------------------------------------------------------------------===//
struct PhysicalKathanJoinGlobalSinkState : public GlobalSinkState {
    virtual ~PhysicalKathanJoinGlobalSinkState() {}

    // Build side rows
    std::vector<std::vector<Value>> build_rows;
    idx_t build_size = 0;

    // GPU hash table
    key_t *d_keys = nullptr;
    value_t *d_vals = nullptr;
    unique_ptr<hash_table_t> hash_table;
    bool finalized = false;
};

struct PhysicalKathanJoinLocalSinkState : public LocalSinkState {
    virtual ~PhysicalKathanJoinLocalSinkState() {}

    // Local build rows
    std::vector<std::vector<Value>> local_build_rows;
};

//===--------------------------------------------------------------------===//
// Source States (probe side handled via Source phase)
//===--------------------------------------------------------------------===//
struct PhysicalKathanJoinGlobalSourceState : public GlobalSourceState {
    virtual ~PhysicalKathanJoinGlobalSourceState() {}
    bool done = false; // Indicates if all probe data has been processed
};

struct PhysicalKathanJoinLocalSourceState : public LocalSourceState {
    virtual ~PhysicalKathanJoinLocalSourceState() {}
    key_t *d_probe_keys = nullptr;
    value_t *d_probe_results = nullptr;
};

//===--------------------------------------------------------------------===//
// Constructor
//===--------------------------------------------------------------------===//
PhysicalKathanJoin::PhysicalKathanJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                     unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond,
                                     JoinType join_type, idx_t estimated_cardinality)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::KATHAN_JOIN, std::move(cond), join_type, estimated_cardinality) {

    printf("PhysicalKathanJoin: Constructor called.\n");
    children.push_back(std::move(left));
    children.push_back(std::move(right));

    for (auto &c : conditions) {
        condition_types.push_back(c.left->return_type);
    }

    // Build side (right child)
    auto &build_types = children[1]->GetTypes();
    for (idx_t i = 0; i < build_types.size(); i++) {
        build_payload_col_idx.push_back(i);
        build_payload_types.push_back(build_types[i]);
    }

    // Initialize probe output column indices (probe side columns first)
    for (idx_t i = 0; i < children[0]->GetTypes().size(); i++) {
        probe_output_col_idx.push_back(i);
        probe_output_types.push_back(children[0]->GetTypes()[i]);
    }

    // Add build columns after probe columns for output
    for (auto &btype : build_payload_types) {
        probe_output_types.push_back(btype);
    }

    printf("PhysicalKathanJoin: Initialized successfully.\n");
}

//===--------------------------------------------------------------------===//
// ParamsToString
//===--------------------------------------------------------------------===//
InsertionOrderPreservingMap<string> PhysicalKathanJoin::ParamsToString() const {
    InsertionOrderPreservingMap<string> result;
    result["Join Type"] = EnumUtil::ToString(join_type);
    string condition_info;
    for (idx_t i = 0; i < conditions.size(); i++) {
        if (i > 0) {
            condition_info += "\n";
        }
        auto &cond = conditions[i];
        condition_info += cond.left->GetName() + " " +
                          ExpressionTypeToString(cond.comparison) + " " +
                          cond.right->GetName();
    }
    result["Conditions"] = condition_info;
    SetEstimatedCardinality(result, estimated_cardinality);
    return result;
}

//===--------------------------------------------------------------------===//
// Sink (Build Phase)
//===--------------------------------------------------------------------===//
unique_ptr<GlobalSinkState> PhysicalKathanJoin::GetGlobalSinkState(ClientContext &context) const {
    printf("GetGlobalSinkState called.\n");
    return make_uniq<PhysicalKathanJoinGlobalSinkState>();
}

unique_ptr<LocalSinkState> PhysicalKathanJoin::GetLocalSinkState(ExecutionContext &context) const {
    printf("GetLocalSinkState called.\n");
    return make_uniq<PhysicalKathanJoinLocalSinkState>();
}

SinkResultType PhysicalKathanJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
    auto &gstate = input.global_state.Cast<PhysicalKathanJoinGlobalSinkState>();
    auto &lstate = input.local_state.Cast<PhysicalKathanJoinLocalSinkState>();

    // Treat all incoming chunks as build side data
    chunk.Flatten();
    for (idx_t i = 0; i < chunk.size(); i++) {
        std::vector<Value> row;
        for (auto col_idx : build_payload_col_idx) {
            row.emplace_back(chunk.GetValue(col_idx, i));
        }
        lstate.local_build_rows.emplace_back(std::move(row));
    }
    printf("Sink: Appended %zu build rows locally.\n", (size_t)chunk.size());
    return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalKathanJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
    auto &gstate = input.global_state.Cast<PhysicalKathanJoinGlobalSinkState>();
    auto &lstate = input.local_state.Cast<PhysicalKathanJoinLocalSinkState>();

    // Combine build rows
    for (auto &row : lstate.local_build_rows) {
        gstate.build_rows.emplace_back(std::move(row));
    }
    lstate.local_build_rows.clear();

    printf("Combine: Combined local build rows into global state.\n");
    return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalKathanJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                            OperatorSinkFinalizeInput &input) const {
    auto &gstate = input.global_state.Cast<PhysicalKathanJoinGlobalSinkState>();
    gstate.build_size = gstate.build_rows.size();
    printf("Finalize: Total build side rows = %zu\n", (size_t)gstate.build_size);

    if (gstate.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
        printf("Finalize: Build side empty => empty result.\n");
        return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
    }

    // Build GPU hash table from build_rows
    std::vector<key_t> h_keys;
    std::vector<value_t> h_vals;
    h_keys.reserve(gstate.build_size);
    h_vals.reserve(gstate.build_size);

    // Assume first column is the join key
    for (idx_t i = 0; i < gstate.build_size; i++) {
        auto &key_val = gstate.build_rows[i][0];
        if (!key_val.IsNull()) {
            uint32_t key_extracted = key_val.GetValue<uint32_t>();
            h_keys.emplace_back(static_cast<key_t>(key_extracted));
            h_vals.emplace_back(static_cast<value_t>(i));
        }
    }

    gstate.build_size = h_keys.size();
    if (gstate.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
        return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
    }

    // Allocate GPU memory and copy build keys and values
    cudaMalloc(&gstate.d_keys, sizeof(key_t) * gstate.build_size);
    cudaMalloc(&gstate.d_vals, sizeof(value_t) * gstate.build_size);
    cudaMemcpy(gstate.d_keys, h_keys.data(), sizeof(key_t) * gstate.build_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gstate.d_vals, h_vals.data(), sizeof(value_t) * gstate.build_size, cudaMemcpyHostToDevice);

    // Initialize and build the GPU hash table
    float load_factor = 0.9f;
    uint64_t capacity = static_cast<uint64_t>(gstate.build_size / load_factor);
    gstate.hash_table = make_uniq<hash_table_t>(capacity);
    gstate.hash_table->insert(gstate.d_keys, gstate.d_vals, gstate.build_size);
    cudaDeviceSynchronize();

    gstate.finalized = true;
    printf("Finalize: GPU hash table built with %zu entries.\n", (size_t)gstate.build_size);

    return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source (Probe Phase)
//===--------------------------------------------------------------------===//
unique_ptr<GlobalSourceState> PhysicalKathanJoin::GetGlobalSourceState(ClientContext &context) const {
    return make_uniq<PhysicalKathanJoinGlobalSourceState>();
}

unique_ptr<LocalSourceState> PhysicalKathanJoin::GetLocalSourceState(ExecutionContext &context,
                                                                     GlobalSourceState &gstate) const {
    return make_uniq<PhysicalKathanJoinLocalSourceState>();
}

SourceResultType PhysicalKathanJoin::GetData(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSourceInput &input) const {
    auto &gsource = input.global_state.Cast<PhysicalKathanJoinGlobalSourceState>();
    auto &lsource = input.local_state.Cast<PhysicalKathanJoinLocalSourceState>();
    auto &gsink = sink_state->Cast<PhysicalKathanJoinGlobalSinkState>();

    // Ensure hash table is finalized
    if (!gsink.finalized) {
        // Hash table not ready yet
        return SourceResultType::FINISHED;
    }

    // Handle empty build side if required
    if (gsink.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
        gsource.done = true;
        return SourceResultType::FINISHED;
    }

    // If we've previously finished
    if (gsource.done) {
        return SourceResultType::FINISHED;
    }

    // Fetch a probe chunk from the probe side (left child)
    DataChunk probe_chunk;
    probe_chunk.Initialize(Allocator::DefaultAllocator(), children[0]->GetTypes());

    auto res = children[0]->GetData(context, probe_chunk, input);
    if (res == SourceResultType::FINISHED) {
        // No more probe data
        gsource.done = true;
        return SourceResultType::FINISHED;
    }

    if (probe_chunk.size() == 0) {
        // Got an empty chunk, just return HAVE_MORE_OUTPUT and try again later
        return SourceResultType::HAVE_MORE_OUTPUT;
    }

    // Now we have probe data, we can proceed
    probe_chunk.Flatten();
    idx_t probe_size = probe_chunk.size();
    std::vector<key_t> h_probe_keys(probe_size);

    for (idx_t i = 0; i < probe_size; i++) {
        auto key_val = probe_chunk.GetValue(0, i).GetValue<uint32_t>();
        h_probe_keys[i] = static_cast<key_t>(key_val);
    }

    // Allocate GPU buffers if needed
    if (!lsource.d_probe_keys) {
        cudaMalloc(&lsource.d_probe_keys, sizeof(key_t) * STANDARD_VECTOR_SIZE);
        cudaMalloc(&lsource.d_probe_results, sizeof(value_t) * STANDARD_VECTOR_SIZE);
    }

    // Copy probe keys to GPU
    cudaMemcpy(lsource.d_probe_keys, h_probe_keys.data(), sizeof(key_t) * probe_size, cudaMemcpyHostToDevice);

    // Perform GPU hash table lookup
    gsink.hash_table->retrieve(lsource.d_probe_keys, probe_size, lsource.d_probe_results);
    cudaDeviceSynchronize();

    // Retrieve results from GPU
    std::vector<value_t> h_results(probe_size);
    cudaMemcpy(h_results.data(), lsource.d_probe_results, sizeof(value_t) * probe_size, cudaMemcpyDeviceToHost);

    // Initialize output chunk
    chunk.Initialize(Allocator::DefaultAllocator(), probe_output_types);

    // Join logic: produce matched rows
    idx_t out_idx = 0;
    idx_t probe_col_count = children[0]->GetTypes().size(); // Number of probe columns
    idx_t build_col_count = build_payload_types.size();    // Number of build columns

    for (idx_t i = 0; i < probe_size && out_idx < STANDARD_VECTOR_SIZE; i++) {
        auto match_idx = h_results[i];
        if (match_idx == static_cast<value_t>(-1)) {
            // No match, skip
            continue;
        }

        // Set probe columns
        for (idx_t col = 0; col < probe_col_count; col++) {
            chunk.SetValue(col, out_idx, probe_chunk.GetValue(col, i));
        }

        // Set build columns
        for (idx_t col = 0; col < build_col_count; col++) {
            chunk.SetValue(probe_col_count + col, out_idx, gsink.build_rows[match_idx][build_payload_col_idx[col]]);
        }

        out_idx++;
    }

    chunk.SetCardinality(out_idx);

    if (out_idx == 0) {
        // No matches found in this batch, but maybe next batch has matches
        return SourceResultType::HAVE_MORE_OUTPUT;
    }

    return SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb
