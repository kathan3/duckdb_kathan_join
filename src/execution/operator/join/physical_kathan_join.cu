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
#include "duckdb/execution/expression_executor.hpp"

#include <warpcore/single_value_hash_table.cuh>
#include <cuda_runtime.h>
#include <cstdio>

namespace duckdb {

using key_t = uint32_t;
using value_t = uint32_t;
using hash_table_t = warpcore::SingleValueHashTable<key_t, value_t>;

//===--------------------------------------------------------------------===//
// Constructor
//===--------------------------------------------------------------------===//
PhysicalKathanJoin::PhysicalKathanJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left, unique_ptr<PhysicalOperator> right,
                                       vector<JoinCondition> cond, JoinType join_type, idx_t estimated_cardinality)
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

	// Probe side (left child)
	auto &probe_types = children[0]->GetTypes();
	for (idx_t i = 0; i < probe_types.size(); i++) {
		probe_output_col_idx.push_back(i);
		probe_output_types.push_back(probe_types[i]);
	}
	// Add build columns after probe columns
	for (auto &btype : build_payload_types) {
		probe_output_types.push_back(btype);
	}

	printf("PhysicalKathanJoin: Initialized successfully.\n");
}

//===--------------------------------------------------------------------===//
// Sink States (build side)
//===--------------------------------------------------------------------===//
struct PhysicalKathanJoinGlobalSinkState : public GlobalSinkState {
	virtual ~PhysicalKathanJoinGlobalSinkState() {}

	std::vector<std::vector<Value>> build_rows;
	idx_t build_size = 0;

	key_t *d_keys = nullptr;
	value_t *d_vals = nullptr;
	unique_ptr<hash_table_t> hash_table;
	bool finalized = false;
};

struct PhysicalKathanJoinLocalSinkState : public LocalSinkState {
	virtual ~PhysicalKathanJoinLocalSinkState() {}
	std::vector<std::vector<Value>> local_build_rows;
};

//===--------------------------------------------------------------------===//
// Source States (probe side)
//===--------------------------------------------------------------------===//
struct PhysicalKathanJoinGlobalSourceState : public GlobalSourceState {
	virtual ~PhysicalKathanJoinGlobalSourceState() {}
	bool done = false; // if we've exhausted probe data or found no more matches
};

struct PhysicalKathanJoinLocalSourceState : public LocalSourceState {
    PhysicalKathanJoinLocalSourceState(ClientContext &context, const vector<JoinCondition> &conditions) 
    : probe_executor(context) {
    vector<LogicalType> condition_types;
        for (auto &cond : conditions) {
            probe_executor.AddExpression(*cond.left);
            condition_types.push_back(cond.left->return_type);
        }
        join_keys.Initialize(Allocator::Get(context), condition_types);
    }
	virtual ~PhysicalKathanJoinLocalSourceState() {}
    ExpressionExecutor probe_executor; // Executor for evaluating probe keys
	DataChunk join_keys;              // Chunk holding evaluated probe keys
	key_t *d_probe_keys = nullptr;
	value_t *d_probe_results = nullptr;
};



//===--------------------------------------------------------------------===//
// ExecuteInternal
//===--------------------------------------------------------------------===//
OperatorResultType PhysicalKathanJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                       DataChunk &chunk, GlobalOperatorState &gstate,
                                                       OperatorState &state) const {
    // This function is not used, but must be implemented to satisfy the base class.
    return OperatorResultType::FINISHED;
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
// We consume build side data here.
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

	chunk.Flatten();
	for (idx_t i = 0; i < chunk.size(); i++) {
		std::vector<Value> row;
		for (auto col_idx : build_payload_col_idx) {
			row.push_back(chunk.GetValue(col_idx, i));
		}
		lstate.local_build_rows.push_back(std::move(row));
	}
	printf("Sink: Appended %zu build rows locally.\n", (size_t)chunk.size());
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalKathanJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<PhysicalKathanJoinGlobalSinkState>();
	auto &lstate = input.local_state.Cast<PhysicalKathanJoinLocalSinkState>();

	for (auto &row : lstate.local_build_rows) {
		gstate.build_rows.push_back(std::move(row));
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

	// Build GPU hash table
	std::vector<key_t> h_keys;
	std::vector<value_t> h_vals;
	h_keys.reserve(gstate.build_size);
	h_vals.reserve(gstate.build_size);

	// Assume first column is the join key
	for (idx_t i = 0; i < gstate.build_size; i++) {
		auto &key_val = gstate.build_rows[i][0];
		if (!key_val.IsNull()) {
			uint32_t key_extracted = key_val.GetValue<uint32_t>();
			h_keys.push_back((key_t)key_extracted);
			h_vals.push_back((value_t)i);
		}
	}

	gstate.build_size = h_keys.size();
	if (gstate.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	cudaMalloc(&gstate.d_keys, sizeof(key_t)*gstate.build_size);
	cudaMalloc(&gstate.d_vals, sizeof(value_t)*gstate.build_size);
	cudaMemcpy(gstate.d_keys, h_keys.data(), sizeof(key_t)*gstate.build_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gstate.d_vals, h_vals.data(), sizeof(value_t)*gstate.build_size, cudaMemcpyHostToDevice);

	float load_factor = 0.9f;
	uint64_t capacity = (uint64_t)(gstate.build_size / load_factor);

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
	return make_uniq<PhysicalKathanJoinLocalSourceState>(context.client, conditions);
}

SourceResultType PhysicalKathanJoin::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	auto &gsource = input.global_state.Cast<PhysicalKathanJoinGlobalSourceState>();
	auto &lsource = input.local_state.Cast<PhysicalKathanJoinLocalSourceState>();
	auto &gsink = sink_state->Cast<PhysicalKathanJoinGlobalSinkState>();
    for (auto &row : gsink.build_rows) {
    printf("Build Key: %u\n", row[0].GetValue<uint32_t>());
    }

    printf("GetData called.\n");
    printf("GetData: Input chunk size = %zu\n", chunk.size());

	if (!gsink.finalized) {
		// hash table not ready yet
		return SourceResultType::FINISHED;
	}

	if (gsink.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
		// empty build side => no output
		gsource.done = true;
		return SourceResultType::FINISHED;
	}

	if (gsource.done) {
		return SourceResultType::FINISHED;
	}
    
    // Simulated input: Add data to the chunk once
	if (!gsource.done) {
		chunk.SetCardinality(4);
		chunk.SetValue(0, 0, Value::INTEGER(2));
		chunk.SetValue(0, 1, Value::INTEGER(3));
		chunk.SetValue(0, 2, Value::INTEGER(4));
        chunk.SetValue(0, 3, Value::INTEGER(5));
		gsource.done = true; // Mark that data is done
	}

    printf("GetData: Getting probe data.\n");
	// Evaluate probe keys using ExpressionExecutor
	lsource.join_keys.Reset();
	lsource.probe_executor.Execute(chunk, lsource.join_keys);
    printf("GetData: Evaluated probe keys.\n");
	idx_t probe_size = lsource.join_keys.size();
    printf("GetData: Probe size = %zu\n", probe_size);
    if (probe_size == 0) {
    return SourceResultType::FINISHED;
    }
    for (idx_t i = 0; i < probe_size; i++) {
    printf("Probe Key[%zu]: %u\n", i, lsource.join_keys.GetValue(0, i).GetValue<uint32_t>());
    }
	std::vector<key_t> h_probe_keys(probe_size);
	for (idx_t i = 0; i < probe_size; i++) {
		h_probe_keys[i] = (key_t)lsource.join_keys.GetValue(0, i).GetValue<uint32_t>();
	}

    // Debug: Print probe keys before sending to the GPU
    for (idx_t i = 0; i < probe_size; i++) {  
        printf("Sending Probe Key[%zu] = %u\n", i, h_probe_keys[i]);
    }
	// allocate GPU buffers if needed
	if (!lsource.d_probe_keys) {
		cudaMalloc(&lsource.d_probe_keys, sizeof(key_t)*STANDARD_VECTOR_SIZE);
		cudaMalloc(&lsource.d_probe_results, sizeof(value_t)*STANDARD_VECTOR_SIZE);
	}

	cudaMemcpy(lsource.d_probe_keys, h_probe_keys.data(), sizeof(key_t)*probe_size, cudaMemcpyHostToDevice);
	gsink.hash_table->retrieve(lsource.d_probe_keys, probe_size, lsource.d_probe_results);
	cudaDeviceSynchronize();

	std::vector<value_t> h_results(probe_size);
	cudaMemcpy(h_results.data(), lsource.d_probe_results, sizeof(value_t)*probe_size, cudaMemcpyDeviceToHost);

	chunk.Initialize(Allocator::DefaultAllocator(), probe_output_types);

	// Join logic:
	// For each probe row, if h_results[i] != -1, we have a build row match.
	idx_t out_idx = 0;
	idx_t probe_col_count = probe_output_col_idx.size();
	idx_t build_col_count = build_payload_types.size();

	for (idx_t i = 0; i < probe_size && out_idx < STANDARD_VECTOR_SIZE; i++) {
		auto match_idx = h_results[i];
		if (match_idx == (value_t)-1) {
			// no match
			continue;
		}
		// Construct output row
		for (idx_t col = 0; col < probe_col_count; col++) {
			chunk.SetValue(col, out_idx,lsource.join_keys.GetValue(col, i));
		}
		auto &build_row = gsink.build_rows[match_idx];
		for (idx_t col = 0; col < build_col_count; col++) {
			chunk.SetValue(probe_col_count + col, out_idx, build_row[col]);
		}

		out_idx++;
	}

	chunk.SetCardinality(out_idx);

	if (out_idx == 0) {
		// no matches in this batch, but maybe next batch
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
    printf("GetData: Output rows = %zu\n", out_idx);

	// Return FINISHED if all data has been output
	return gsource.done ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb
