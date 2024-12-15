//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_kathan_join.cu
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/join/physical_kathan_join.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/execution_context.hpp"
#include <warpcore/single_value_hash_table.cuh>
#include <cuda_runtime.h>

namespace duckdb {

using namespace warpcore;

// Define the hash table type
using key_t = uint32_t;
using value_t = uint32_t;
using hash_table_t = SingleValueHashTable<key_t, value_t>;

struct KathanJoinGlobalState : public GlobalSinkState {
    unique_ptr<hash_table_t> hash_table;

    KathanJoinGlobalState(idx_t capacity) {
        hash_table = make_uniq<hash_table_t>(capacity);
    }
};

struct KathanJoinLocalState : public LocalSinkState {
    DataChunk join_keys;
    DataChunk payload;
};

unique_ptr<GlobalSinkState> PhysicalKathanJoin::GetGlobalSinkState(ClientContext &context) const {
    idx_t capacity = 1024; // Define an appropriate capacity based on the workload
    printf("GetGlobalSinkState: Initializing hash table with capacity = %zu\n", capacity);
    return make_uniq<KathanJoinGlobalState>(capacity);
}

unique_ptr<LocalSinkState> PhysicalKathanJoin::GetLocalSinkState(ExecutionContext &context) const {
    auto state = make_uniq<KathanJoinLocalState>();
    state->join_keys.Initialize(Allocator::DefaultAllocator(), condition_types);
    state->payload.Initialize(Allocator::DefaultAllocator(), lhs_output_types);

    // Log initialization details
    printf("GetLocalSinkState: Initialized join_keys with %zu columns.\n", state->join_keys.ColumnCount());
    printf("GetLocalSinkState: Initialized payload with %zu columns.\n", state->payload.ColumnCount());

    return std::move(state);
}

SinkResultType PhysicalKathanJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
    auto &gstate = input.global_state.Cast<KathanJoinGlobalState>();
    auto &lstate = input.local_state.Cast<KathanJoinLocalState>();

    // Debug: Log chunk details
    printf("Sink: Received chunk with %zu rows and %zu columns.\n", chunk.size(), chunk.ColumnCount());

    if (chunk.size() == 0 || chunk.ColumnCount() == 0) {
        throw InternalException("Sink received an empty chunk.");
    }

    // Reset join keys and payload
    lstate.join_keys.Reset();
    lstate.payload.Reset();

    // Validate column count matches join condition
    if (condition_types.size() != chunk.ColumnCount()) {
        throw InternalException("Mismatch between condition types (%zu) and chunk columns (%zu).",
                                condition_types.size(), chunk.ColumnCount());
    }

    // Populate join keys
    for (idx_t col_idx = 0; col_idx < condition_types.size(); col_idx++) {
        for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
            lstate.join_keys.data[col_idx].SetValue(row_idx, chunk.data[col_idx].GetValue(row_idx));
        }
    }

    // Extract raw pointers for keys and payloads
    auto keys_data = FlatVector::GetData<key_t>(lstate.join_keys.data[0]);
    auto payload_data = FlatVector::GetData<value_t>(lstate.payload.data[0]);

    // Insert into the hash table
    try {
        gstate.hash_table->insert(keys_data, payload_data, chunk.size());
    } catch (std::exception &e) {
        throw InternalException("Error inserting into hash table: %s", e.what());
    }

    printf("Sink: Inserted %zu rows into hash table.\n", chunk.size());
    return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalKathanJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
    auto &gstate = input.global_state.Cast<KathanJoinGlobalState>();
    auto &lstate = input.local_state.Cast<KathanJoinLocalState>();

    // No specific combine logic for now
    return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalKathanJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context, OperatorSinkFinalizeInput &input) const {
    auto &gstate = input.global_state.Cast<KathanJoinGlobalState>();

    printf("Finalize: Skipping hash table finalization as no Finalize method is provided.\n");

    // Validate hash table state
    if (!gstate.hash_table) {
        throw InternalException("Hash table is not initialized.");
    }

    printf("Finalize: Hash table is ready for probing.\n");
    return SinkFinalizeType::READY;
}

unique_ptr<OperatorState> PhysicalKathanJoin::GetOperatorState(ExecutionContext &context) const {
    return make_uniq<OperatorState>();
}

OperatorResultType PhysicalKathanJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk, GlobalOperatorState &gstate, OperatorState &state) const {
    auto &global_state = sink_state->Cast<KathanJoinGlobalState>();

    // Debug: Log input details
    printf("ExecuteInternal: Input chunk has %zu rows and %zu columns.\n", input.size(), input.ColumnCount());

    if (input.size() == 0 || input.ColumnCount() == 0) {
        throw InternalException("ExecuteInternal received an empty input chunk.");
    }

    // Extract keys
    auto keys_data = FlatVector::GetData<key_t>(input.data[0]);
    auto chunk_size = input.size();

    // Allocate result buffer
    value_t *result_d = nullptr;
    cudaMalloc(&result_d, sizeof(value_t) * chunk_size);

    // Retrieve from hash table
    try {
        global_state.hash_table->retrieve(keys_data, chunk_size, result_d);
    } catch (std::exception &e) {
        cudaFree(result_d);
        throw InternalException("Error retrieving from hash table: %s", e.what());
    }

    // Copy results back to host
    value_t *result_h = new value_t[chunk_size];
    cudaMemcpy(result_h, result_d, sizeof(value_t) * chunk_size, cudaMemcpyDeviceToHost);

    // Populate output chunk
    chunk.Reset();
    idx_t output_size = 0;
    for (idx_t i = 0; i < chunk_size; ++i) {
        if (result_h[i] != static_cast<value_t>(-1)) {
            for (idx_t col_idx = 0; col_idx < input.ColumnCount(); ++col_idx) {
                chunk.data[col_idx].SetValue(output_size, input.data[col_idx].GetValue(i));
            }
            output_size++;
        }
    }
    chunk.SetCardinality(output_size);

    delete[] result_h;
    cudaFree(result_d);

    return output_size > 0 ? OperatorResultType::HAVE_MORE_OUTPUT : OperatorResultType::FINISHED;
}

} // namespace duckdb
