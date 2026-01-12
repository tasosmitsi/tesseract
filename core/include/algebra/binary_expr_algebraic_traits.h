#pragma once

// Forward declare BinaryExpr
template <typename LHS, typename RHS, template <typename, my_size_t, typename> class Op>
class BinaryExpr;

namespace algebra
{
    // Propagate traits for BinaryExpr
    template <typename LHS, typename RHS,
              template <typename, my_size_t, typename> class Op>
    struct algebraic_traits<BinaryExpr<LHS, RHS, Op>>
    {
        static constexpr bool vector_space = is_vector_space_v<LHS> && is_vector_space_v<RHS>;
        static constexpr bool algebra = is_algebra_v<LHS> && is_algebra_v<RHS>;
        static constexpr bool lie_group = is_lie_group_v<LHS> && is_lie_group_v<RHS>;
        static constexpr bool metric = is_metric_v<LHS> && is_metric_v<RHS>;
        static constexpr bool tensor = is_tensor_v<LHS> && is_tensor_v<RHS>;
    };
} // namespace algebra
