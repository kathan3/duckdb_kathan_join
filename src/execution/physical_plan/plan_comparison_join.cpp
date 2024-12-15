#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_iejoin.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/join/physical_kathan_join.hpp" //added this
#include "duckdb/execution/operator/join/physical_piecewise_merge_join.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

namespace duckdb {

static void RewriteJoinCondition(Expression &expr, idx_t offset) {
	if (expr.type == ExpressionType::BOUND_REF) {
		auto &ref = expr.Cast<BoundReferenceExpression>();
		ref.index += offset;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { RewriteJoinCondition(child, offset); });
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::PlanComparisonJoin(LogicalComparisonJoin &op) {
	// now visit the children
	D_ASSERT(op.children.size() == 2);
	idx_t lhs_cardinality = op.children[0]->EstimateCardinality(context);
	idx_t rhs_cardinality = op.children[1]->EstimateCardinality(context);
	auto left = CreatePlan(*op.children[0]);
	auto right = CreatePlan(*op.children[1]);
	left->estimated_cardinality = lhs_cardinality;
	right->estimated_cardinality = rhs_cardinality;
	D_ASSERT(left && right);

	if (op.conditions.empty()) {
		// no conditions: insert a cross product
		return make_uniq<PhysicalCrossProduct>(op.types, std::move(left), std::move(right), op.estimated_cardinality);
	}

	idx_t has_range = 0;
	bool has_equality = op.HasEquality(has_range);
	bool can_merge = has_range > 0;
	bool can_iejoin = has_range >= 2 && recursive_cte_tables.empty();
	switch (op.join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::RIGHT_ANTI:
	case JoinType::RIGHT_SEMI:
	case JoinType::MARK:
		can_merge = can_merge && op.conditions.size() == 1;
		can_iejoin = false;
		break;
	default:
		break;
	}
	auto &client_config = ClientConfig::GetConfig(context);

	//	TODO: Extend PWMJ to handle all comparisons and projection maps
	const auto prefer_range_joins = client_config.prefer_range_joins && can_iejoin;

	unique_ptr<PhysicalOperator> plan;

		// At this point, we know which operators are logically feasible:
		// - HashJoin if has_equality && !prefer_range_joins
		// - IEJoin if can_iejoin
		// - Piecewise Merge Join if can_merge
		// - NestedLoop or BlockwiseNL based on conditions otherwise
		// - KathanJoin (our new operator) - Let's assume it's always feasible for demonstration,
		//   or you can add your own feasibility checks.


	    //Define some simple c//ost estimation functions:
		auto EstimateHashJoinCost = [&](idx_t left_card, idx_t right_card) {
			// For demo: hash join cost ~ sum of sizes
			return 1000;
		};
		auto EstimateIEJoinCost = [&](idx_t left_card, idx_t right_card) {
			// For demo: IEJoin cost ~ left_card * log(right_card)
			return (double)left_card * std::log((double)right_card + 1);
		};
		auto EstimatePiecewiseMergeJoinCost = [&](idx_t left_card, idx_t right_card) {
			// For demo: piecewise merge join ~ (left_card + right_card) * log(left_card + right_card)
			double total = (double)left_card + (double)right_card;
			return total * std::log(total + 1);
		};
		auto EstimateNestedLoopCost = [&](idx_t left_card, idx_t right_card) {
			// Nested loop join ~ left_card * right_card
			return (double)left_card * (double)right_card;
		};
		auto EstimateBlockwiseNLCost = [&](idx_t left_card, idx_t right_card) {
			// Blockwise NL slightly better than naive nested loop
			return (double)left_card * (double)right_card * 0.8; // just a demo factor
		};
		auto EstimateKathanJoinCost = [&](idx_t left_card, idx_t right_card) {
			// KathanJoin: for testing, let's say it's 0.
			return 0;
		};

		// Initialize costs to infinity
		double hash_join_cost = std::numeric_limits<double>::infinity();
		double ie_join_cost = std::numeric_limits<double>::infinity();
		double piecewise_merge_cost = std::numeric_limits<double>::infinity();
		double nested_loop_cost = std::numeric_limits<double>::infinity();
		double blockwise_cost = std::numeric_limits<double>::infinity();
		double kathan_join_cost = std::numeric_limits<double>::infinity();

		// Check feasibility and assign costs
		if (has_equality && !prefer_range_joins) {
			hash_join_cost = EstimateHashJoinCost(lhs_cardinality, rhs_cardinality);
			kathan_join_cost = EstimateKathanJoinCost(lhs_cardinality, rhs_cardinality);
		}
		if (can_iejoin) {
			ie_join_cost = EstimateIEJoinCost(lhs_cardinality, rhs_cardinality);
		}
		if (can_merge && !can_iejoin) {
			// If can_merge is true but we didn't pick IEJoin, we can try piecewise merge join
			piecewise_merge_cost = EstimatePiecewiseMergeJoinCost(lhs_cardinality, rhs_cardinality);
		}

		// If none of the above conditions matched for a specialized join,
		// We consider nested loop or blockwise NL join.
		// Check if conditions are supported by nested loop join:
		bool can_nested_loop = PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type);
		if (can_nested_loop) {
			nested_loop_cost = EstimateNestedLoopCost(lhs_cardinality, rhs_cardinality);
		} else {
			// If nested loop is not directly supported, fallback to blockwise NL
			// We need to rewrite conditions for blockwise NL:
			for (auto &cond : op.conditions) {
				RewriteJoinCondition(*cond.right, left->types.size());
			}
			blockwise_cost = EstimateBlockwiseNLCost(lhs_cardinality, rhs_cardinality);
		}


			// Now pick the operator with the lowest cost
		double min_cost = std::min(
			{hash_join_cost, ie_join_cost, piecewise_merge_cost, nested_loop_cost, blockwise_cost, kathan_join_cost});

		// Create the chosen operator
		if (min_cost == kathan_join_cost) {
			// Use PhysicalKathanJoin
			plan = make_uniq<PhysicalKathanJoin>(op, std::move(left), std::move(right), std::move(op.conditions), op.join_type, op.estimated_cardinality);
		} else if (min_cost == hash_join_cost) {
			// Hash Join
			 plan = make_uniq<PhysicalHashJoin>(
				op, std::move(left), std::move(right), std::move(op.conditions), op.join_type, op.left_projection_map,
				op.right_projection_map, std::move(op.mark_types), op.estimated_cardinality, std::move(op.filter_pushdown));
		} else if (min_cost == ie_join_cost) {
			// IE Join
			plan = make_uniq<PhysicalIEJoin>(op, std::move(left), std::move(right), std::move(op.conditions), op.join_type,
											op.estimated_cardinality);
		} else if (min_cost == piecewise_merge_cost) {
			// Piecewise Merge Join
			plan = make_uniq<PhysicalPiecewiseMergeJoin>(op, std::move(left), std::move(right), std::move(op.conditions),
														op.join_type, op.estimated_cardinality);
		} else if (min_cost == nested_loop_cost) {
			// Nested Loop Join
			plan =  make_uniq<PhysicalNestedLoopJoin>(op, std::move(left), std::move(right), std::move(op.conditions),
													op.join_type, op.estimated_cardinality);
		} else {
			// Blockwise NL Join
			auto condition = JoinCondition::CreateExpression(std::move(op.conditions));
			plan =  make_uniq<PhysicalBlockwiseNLJoin>(op, std::move(left), std::move(right), std::move(condition),
													op.join_type, op.estimated_cardinality);
		}
	return plan;
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalComparisonJoin &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		return PlanAsOfJoin(op);
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		return PlanComparisonJoin(op);
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return PlanDelimJoin(op);
	default:
		throw InternalException("Unrecognized operator type for LogicalComparisonJoin");
	}
}

} // namespace duckdb