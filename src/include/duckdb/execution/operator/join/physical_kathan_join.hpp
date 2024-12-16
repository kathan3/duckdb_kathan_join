//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_kathan_join.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class PhysicalKathanJoin : public PhysicalComparisonJoin {
public:
    static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::KATHAN_JOIN;

public:
    PhysicalKathanJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                      unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond,
                      JoinType join_type, idx_t estimated_cardinality);

    // The join key types
    vector<LogicalType> condition_types;

    // Build side info (right child)
    vector<idx_t> build_payload_col_idx;
    vector<LogicalType> build_payload_types;

    // Output columns: probe side columns first, then build side columns
    vector<idx_t> probe_output_col_idx;
    vector<LogicalType> probe_output_types;

public:
    // Sink interface (build phase)
    unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
    unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
    SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
    SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
    SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                              OperatorSinkFinalizeInput &input) const override;

    // Source interface (probe phase)
    bool IsSink() const override {
        return true;
    }
    bool ParallelSink() const override {
        return true;
    }
    bool IsSource() const override {
        return true;
    }
    bool ParallelSource() const override {
        return true;
    }

    unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
    unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context, GlobalSourceState &gstate) const override;
    SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;

protected:
    InsertionOrderPreservingMap<string> ParamsToString() const override;
    OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                       GlobalOperatorState &gstate, OperatorState &state) const override {
        // Not used here, but must provide an implementation.
        return OperatorResultType::FINISHED;
    }
};

} // namespace duckdb
