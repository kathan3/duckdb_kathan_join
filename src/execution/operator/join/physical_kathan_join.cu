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
#include "duckdb/planner/expression/bound_reference_expression.hpp"

// NEW INCLUDES for CachingOperatorState
#include "duckdb/execution/physical_operator.hpp"   // for CachingOperatorState

#include <warpcore/single_value_hash_table.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>

namespace duckdb {

// ------------------------------------------------------------------------
// We mirror the same approach as "PhysicalHashJoin" does in DuckDB
// i.e. we have "lhs_output_columns", "rhs_output_columns", "payload_columns"
// and an "unordered_map" that identifies which RHS columns are join keys.
// ------------------------------------------------------------------------

// A simplistic GPU hash table for single integer keys
using key_t = uint32_t;
using value_t = uint32_t;
using hash_table_t = warpcore::SingleValueHashTable<key_t, value_t>;

//===--------------------------------------------------------------------===//
// Constructor
//===--------------------------------------------------------------------===//

PhysicalKathanJoin::PhysicalKathanJoin(LogicalOperator &op, 
                                       unique_ptr<PhysicalOperator> left,
                                       unique_ptr<PhysicalOperator> right, 
                                       vector<JoinCondition> cond,
                                       JoinType join_type,
                                       const vector<idx_t> &left_projection_map,
                                       const vector<idx_t> &right_projection_map,
                                       idx_t estimated_cardinality)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::KATHAN_JOIN, std::move(cond), join_type, estimated_cardinality) {
	// printf("=== PhysicalKathanJoin: Constructor called ===\n");
	children.push_back(std::move(left));
	children.push_back(std::move(right));

	// Collect condition types, and which conditions are just references
	unordered_map<idx_t, idx_t> build_columns_in_conditions;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
		condition_types.push_back(condition.left->return_type);
		if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			build_columns_in_conditions.emplace(condition.right->Cast<BoundReferenceExpression>().index, cond_idx);
		}
	}

	auto &lhs_input_types = children[0]->GetTypes();

	// Create a projection map for the LHS (if it was empty), for convenience
	lhs_output_columns.col_idxs = left_projection_map;
	if (lhs_output_columns.col_idxs.empty()) {
		lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
		for (idx_t i = 0; i < lhs_input_types.size(); i++) {
			lhs_output_columns.col_idxs.emplace_back(i);
		}
	}

	for (auto &lhs_col : lhs_output_columns.col_idxs) {
		auto &lhs_col_type = lhs_input_types[lhs_col];
		lhs_output_columns.col_types.push_back(lhs_col_type);
	}

	auto &rhs_input_types = children[1]->GetTypes();

	// Create a projection map for the RHS (if it was empty), for convenience
	auto right_projection_map_copy = right_projection_map;
	if (right_projection_map_copy.empty()) {
		right_projection_map_copy.reserve(rhs_input_types.size());
		for (idx_t i = 0; i < rhs_input_types.size(); i++) {
			right_projection_map_copy.emplace_back(i);
		}
	}

	// Now fill payload expressions/types and RHS columns/types
	for (auto &rhs_col : right_projection_map_copy) {
		auto &rhs_col_type = rhs_input_types[rhs_col];
		auto it = build_columns_in_conditions.find(rhs_col);
		if (it == build_columns_in_conditions.end()) {
			// This rhs column is not a join key
			payload_columns.col_idxs.push_back(rhs_col);
			payload_columns.col_types.push_back(rhs_col_type);
			rhs_output_columns.col_idxs.push_back(condition_types.size() + payload_columns.col_types.size() - 1);
		} else {
			// This rhs column is a join key
			rhs_output_columns.col_idxs.push_back(it->second);
		}
		rhs_output_columns.col_types.push_back(rhs_col_type);
	}

	// Print payload columns
	// printf("Payload columns:\n");
	// for (idx_t i = 0; i < payload_columns.col_idxs.size(); i++) {
	// 	printf("  idx: %zu, type: %s\n",
	// 	       payload_columns.col_idxs[i],
	// 	       payload_columns.col_types[i].ToString().c_str());
	// }
	// Print LHS output columns
	// printf("LHS output columns:\n");
	// for (idx_t i = 0; i < lhs_output_columns.col_idxs.size(); i++) {
	// 	printf("  idx: %zu, type: %s\n",
	// 	       lhs_output_columns.col_idxs[i],
	// 	       lhs_output_columns.col_types[i].ToString().c_str());
	// }
	// // Print RHS output columns
	// printf("RHS output columns:\n");
	// for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) {
	// 	printf("  idx: %zu, type: %s\n",
	// 	       rhs_output_columns.col_idxs[i],
	// 	       rhs_output_columns.col_types[i].ToString().c_str());
	// }

	// Overwrite "this->types" with final (LHS + RHS) columns
	this->types.clear();
	for (auto &t : lhs_output_columns.col_types) {
		this->types.push_back(t);
	}
	for (auto &t : rhs_output_columns.col_types) {
		this->types.push_back(t);
	}
	// printf("LHS has %zu columns, RHS has %zu columns.\n",
	//        lhs_input_types.size(), rhs_input_types.size());
	// printf("Output has %zu columns.\n", this->types.size());
}

PhysicalKathanJoin::PhysicalKathanJoin(LogicalOperator &op,
                                       unique_ptr<PhysicalOperator> left,
                                       unique_ptr<PhysicalOperator> right,
                                       vector<JoinCondition> cond,
                                       JoinType join_type,
                                       idx_t estimated_cardinality)
    : PhysicalKathanJoin(op, std::move(left), std::move(right), std::move(cond), join_type, {}, {}, estimated_cardinality) {
}

//===--------------------------------------------------------------------===//
// Global/Local Sink States
//===--------------------------------------------------------------------===//
struct KathanJoinGlobalSinkState : public GlobalSinkState {
	~KathanJoinGlobalSinkState() override {
	}

	// CPU row-wise build side
	vector<vector<Value>> build_rows;
	idx_t build_size = 0;

	// GPU stuff
	key_t *d_keys = nullptr;
	value_t *d_vals = nullptr;
	unique_ptr<hash_table_t> gpu_hash_table;

	bool finalized = false;
};

struct KathanJoinLocalSinkState : public LocalSinkState {
	~KathanJoinLocalSinkState() override {
	}
	vector<vector<Value>> local_build_rows;
};

//===--------------------------------------------------------------------===//
// Operator State (Probe Side)
//===--------------------------------------------------------------------===//

// 1) We inherit from CachingOperatorState to gain access to
//    'initialized', 'can_cache_chunk', and 'cached_chunk'
class KathanJoinOperatorState : public CachingOperatorState {
public:
	// Example constructor, similar to your old OperatorState
	KathanJoinOperatorState(ClientContext &context, const vector<JoinCondition> &cond_p)
	    : probe_executor(context) {

		// Just for single key example
		for (auto &c : cond_p) {
			probe_executor.AddExpression(*c.left);
		}
		vector<LogicalType> key_types;
		key_types.push_back(cond_p[0].left->return_type);
		join_keys.Initialize(Allocator::Get(context), key_types);
	}

	~KathanJoinOperatorState() override {
	}

	ExpressionExecutor probe_executor;
	DataChunk join_keys; // single-col chunk for LHS key

	bool gpu_alloc = false;
	key_t *d_probe_keys = nullptr;
	value_t *d_probe_results = nullptr;

	DataChunk lhs_output;

	void Reset() {
		join_keys.Reset();
		lhs_output.Reset();
	}
	// Optionally override Finalize
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		// e.g. flush profiler
		// context.thread.profiler.Flush(op);
	}
};

//===--------------------------------------------------------------------===//
// Overridden Methods
//===--------------------------------------------------------------------===//

unique_ptr<GlobalSinkState> PhysicalKathanJoin::GetGlobalSinkState(ClientContext &context) const {
	// printf("GetGlobalSinkState called.\n");
	return make_uniq<KathanJoinGlobalSinkState>();
}

unique_ptr<LocalSinkState> PhysicalKathanJoin::GetLocalSinkState(ExecutionContext &context) const {
	// printf("GetLocalSinkState called.\n");
	return make_uniq<KathanJoinLocalSinkState>();
}

// Sink
SinkResultType PhysicalKathanJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<KathanJoinGlobalSinkState>();
	auto &lstate = input.local_state.Cast<KathanJoinLocalSinkState>();
	// chunk.Print();
	chunk.Flatten();

	for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
		vector<Value> row;
		row.reserve(chunk.ColumnCount());
		for (idx_t col_idx = 0; col_idx < chunk.ColumnCount(); col_idx++) {
			row.push_back(chunk.GetValue(col_idx, row_idx));
		}
		lstate.local_build_rows.push_back(std::move(row));
	}
	printf("Sink: appended %zu build rows locally.\n", (size_t)chunk.size());
	return SinkResultType::NEED_MORE_INPUT;
}

// Combine
SinkCombineResultType PhysicalKathanJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<KathanJoinGlobalSinkState>();
	auto &lstate = input.local_state.Cast<KathanJoinLocalSinkState>();

	for (auto &row : lstate.local_build_rows) {
		gstate.build_rows.push_back(std::move(row));
	}
	lstate.local_build_rows.clear();
	printf("Combine: merged local => global.\n");
	return SinkCombineResultType::FINISHED;
}

// Finalize
SinkFinalizeType PhysicalKathanJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                              OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<KathanJoinGlobalSinkState>();
	gstate.build_size = gstate.build_rows.size();
	printf("Finalize: build side = %zu rows\n", (size_t)gstate.build_size);

	if (gstate.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Build GPU HT
	std::vector<key_t> h_keys;
	std::vector<value_t> h_vals;
	h_keys.reserve(gstate.build_size);
	h_vals.reserve(gstate.build_size);

	for (idx_t i = 0; i < gstate.build_size; i++) {
		auto &val = gstate.build_rows[i][0];
		if (!val.IsNull()) {
			uint32_t k = val.GetValue<uint32_t>();
			h_keys.push_back(k);
			h_vals.push_back((value_t)i);
		}
	}

	idx_t final_cnt = h_keys.size();
	printf("GPU build side final count = %zu\n", (size_t)final_cnt);
	if (final_cnt == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// GPU allocations
	cudaMalloc(&gstate.d_keys, sizeof(key_t) * final_cnt);
	cudaMalloc(&gstate.d_vals, sizeof(value_t) * final_cnt);
	cudaMemcpy(gstate.d_keys, h_keys.data(), final_cnt * sizeof(key_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gstate.d_vals, h_vals.data(), final_cnt * sizeof(value_t), cudaMemcpyHostToDevice);

	float load_factor = 0.9f;
	uint64_t capacity = (uint64_t)(final_cnt / load_factor);

	gstate.gpu_hash_table = make_uniq<hash_table_t>(capacity);
	gstate.gpu_hash_table->insert(gstate.d_keys, gstate.d_vals, final_cnt);
	cudaDeviceSynchronize();

	gstate.finalized = true;
	printf("Finalize: GPU HT built with %zu entries.\n", (size_t)final_cnt);
	return SinkFinalizeType::READY;
}

// OperatorState
unique_ptr<OperatorState> PhysicalKathanJoin::GetOperatorState(ExecutionContext &context) const {
	// Create a KathanJoinOperatorState that inherits from CachingOperatorState
	auto state = make_uniq<KathanJoinOperatorState>(context.client, conditions);

	// Initialize "lhs_output" with LHS projection from "lhs_output_columns"
	state->lhs_output.Initialize(Allocator::Get(context.client), lhs_output_columns.col_types);

	// Optionally, set up caching flags
	state->initialized = false;
	state->can_cache_chunk = true; // or true if you want to try caching logic

	return std::move(state);
}

// ExecuteInternal (Probe side)
OperatorResultType PhysicalKathanJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                       GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &sink = sink_state->Cast<KathanJoinGlobalSinkState>();
	auto &op_state = state_p.Cast<KathanJoinOperatorState>();

	if (!sink.finalized) {
		return OperatorResultType::FINISHED;
	}
	if (sink.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
		return OperatorResultType::FINISHED;
	}
	if (input.size() == 0) {
		return OperatorResultType::FINISHED;
	}

	// input.Print();
	// input.Flatten();
	op_state.Reset();

	// 1) reference the LHS columns
	op_state.lhs_output.ReferenceColumns(input, lhs_output_columns.col_idxs);

	// 2) evaluate join keys
	op_state.probe_executor.Execute(input, op_state.join_keys);
	op_state.join_keys.Flatten();

	idx_t size = op_state.join_keys.size();
	if (size == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// 3) GPU probe
	if (!op_state.gpu_alloc) {
		cudaMalloc(&op_state.d_probe_keys, sizeof(key_t) * STANDARD_VECTOR_SIZE);
		cudaMalloc(&op_state.d_probe_results, sizeof(value_t) * STANDARD_VECTOR_SIZE);
		op_state.gpu_alloc = true;
	}

	std::vector<key_t> h_probe_keys(size);
	for (idx_t i = 0; i < size; i++) {
		h_probe_keys[i] = op_state.join_keys.GetValue(0, i).GetValue<uint32_t>();
	}
	cudaMemcpy(op_state.d_probe_keys, h_probe_keys.data(), size * sizeof(key_t), cudaMemcpyHostToDevice);

	sink.gpu_hash_table->retrieve(op_state.d_probe_keys, size, op_state.d_probe_results);
	cudaDeviceSynchronize();

	std::vector<value_t> h_results(size);
	cudaMemcpy(h_results.data(), op_state.d_probe_results, size * sizeof(value_t), cudaMemcpyDeviceToHost);

	// 4) Build the final chunk
	chunk.Destroy(); // ensure no leftover columns
	chunk.Initialize(Allocator::DefaultAllocator(), this->types);

	idx_t out_idx = 0;
	for (idx_t i = 0; i < size && out_idx < STANDARD_VECTOR_SIZE; i++) {
		auto build_row_id = h_results[i];
		if (build_row_id == (value_t)-1) {
			continue; // not found
		}

		// [A] Copy LHS columns
		idx_t col_offset = 0;
		for (idx_t c = 0; c < lhs_output_columns.col_idxs.size(); c++) {
			chunk.SetValue(col_offset, out_idx, op_state.lhs_output.GetValue(c, i));
			col_offset++;
		}

		// [B] Copy from build_rows for RHS columns
		auto &row = sink.build_rows[build_row_id];
		for (idx_t c = 0; c < rhs_output_columns.col_idxs.size(); c++) {
			auto out_col_idx = rhs_output_columns.col_idxs[c];
			chunk.SetValue(col_offset, out_idx, row[out_col_idx]);
			col_offset++;
		}
		out_idx++;
	}
	chunk.SetCardinality(out_idx);

	// SelectionVector sel(chunk.size());
	// for (idx_t i = 0; i < chunk.size(); i++) {
	// 	sel.set_index(i, i); // identity mapping (row i maps to row i)
	// }

	// // For each column of the chunk, create a Dictionary Vector referencing the original
	// for (idx_t col_idx = 0; col_idx < chunk.ColumnCount(); col_idx++) {
	// 	// (a) Grab the flat vector
	// 	auto &src_vec = chunk.data[col_idx];

	// 	// (b) Create a new vector with the same LogicalType
	// 	Vector dict_vec(src_vec.GetType());

	// 	// (c) Slice the source vector using 'sel' -> produce DICTIONARY vector
	// 	dict_vec.Slice(src_vec, sel, chunk.size());

	// 	// (d) Replace the original column with this dictionary vector
	// 	src_vec.Reference(dict_vec);
	// }

	if (out_idx > 0) {
		printf("Final chunk:\n");
		// chunk.Print();
		printf("DEBUG: chunk size = %zu, column count = %zu\n",
       (size_t)chunk.size(),
       (size_t)chunk.ColumnCount());
	}
	
	return (out_idx == 0)? OperatorResultType::FINISHED : OperatorResultType::NEED_MORE_INPUT;
}

// Optional Debug Info
InsertionOrderPreservingMap<string> PhysicalKathanJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Join Type"] = EnumUtil::ToString(join_type);
	string conds;
	for (auto &cond : conditions) {
		if (!conds.empty()) {
			conds += " AND ";
		}
		conds += cond.left->GetName() + " " + ExpressionTypeToString(cond.comparison)
		       + " " + cond.right->GetName();
	}
	result["Conditions"] = conds;
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
