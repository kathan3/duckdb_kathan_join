//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_kathan_join.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include <cstdio> // For printf (if structured logging is not used)

namespace duckdb {

class PhysicalKathanJoin : public PhysicalComparisonJoin {
public:
    static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::KATHAN_JOIN;

   PhysicalKathanJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left, unique_ptr<PhysicalOperator> right,
                   vector<JoinCondition> cond, JoinType join_type, idx_t estimated_cardinality)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::KATHAN_JOIN, std::move(cond), join_type, estimated_cardinality) {
        // Debug message to confirm execution flow
        printf("PhysicalKathanJoin: Constructor called, initializing...\n");

        // Check if join conditions are provided
        if (conditions.empty()) {
            throw InternalException("No join conditions provided in PhysicalKathanJoin.");
        }

        // Populate condition types based on join conditions
        for (auto &condition : conditions) {
            if (!condition.left || !condition.right) {
                throw InternalException("Invalid join condition: missing left or right expression.");
            }
            condition_types.push_back(condition.left->return_type);

            // Debug: Log condition details
            printf("Join condition: %s %s %s\n",
                condition.left->GetName().c_str(),
                ExpressionTypeToOperator(condition.comparison).c_str(),
                condition.right->GetName().c_str());
        }

        // Populate LHS output types from the left input
        if (left) {
            lhs_output_types = left->GetTypes();
            if (lhs_output_types.empty()) {
                throw InternalException("Left input operator has no output types.");
            }
        } else {
            throw InternalException("Left input operator is null in PhysicalKathanJoin.");
        }
        
        printf("LHS Output Types size: %zu\n", lhs_output_types.size());

        printf("PhysicalKathanJoin initialized successfully.\n");
    }
    unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;

    //! The types of the join keys
    vector<LogicalType> condition_types;
    //! The indices/types of the lhs columns that need to be output
    vector<LogicalType> lhs_output_types;

protected:
    OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                       GlobalOperatorState &gstate, OperatorState &state) const override;

public:
    unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
    unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
    SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
    SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
    SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                              OperatorSinkFinalizeInput &input) const override;
};

} // namespace duckdb
//===----------------------------------------------------------------------===//