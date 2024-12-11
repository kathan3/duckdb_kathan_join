#pragma once

#include "duckdb/execution/operator/join/physical_comparison_join.hpp"

namespace duckdb {

class PhysicalKathanJoin : public PhysicalComparisonJoin {
public:
    static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::KATHAN_JOIN;

public:
    PhysicalKathanJoin(
        LogicalOperator &op,
        unique_ptr<PhysicalOperator> left,
        unique_ptr<PhysicalOperator> right,
        vector<JoinCondition> cond,
        JoinType join_type,
        idx_t estimated_cardinality
    );

    // Minimal overrides - start small and add more as needed
    unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;

protected:
    OperatorResultType ExecuteInternal(
        ExecutionContext &context,
        DataChunk &input,
        DataChunk &chunk,
        GlobalOperatorState &gstate,
        OperatorState &state
    ) const override;

    // For now, no sink or source logic if you haven't implemented it
    // Add them back gradually once you have minimal compiling code
    
    // If you want sink logic, you can add minimal versions:
    unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
    unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
    SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
    SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
    SinkFinalizeType Finalize(
        Pipeline &pipeline,
        Event &event,
        ClientContext &context,
        OperatorSinkFinalizeInput &input
    ) const override;

};

} // namespace duckdb
