#pragma once

template <typename A, typename B, typename C,
          template <typename, my_size_t, typename> class Op>
class FmaExpr;

template <typename EXPR, typename ScalarT, typename C,
          template <typename, my_size_t, typename> class Op>
class ScalarFmaExpr;

namespace algebra
{
    template <typename A, typename B, typename C,
              template <typename, my_size_t, typename> class Op>
    struct algebraic_traits<FmaExpr<A, B, C, Op>>
    {
        static constexpr bool vector_space =
            is_vector_space_v<A> && is_vector_space_v<B> && is_vector_space_v<C>;
        static constexpr bool algebra =
            is_algebra_v<A> && is_algebra_v<B> && is_algebra_v<C>;
        static constexpr bool lie_group =
            is_lie_group_v<A> && is_lie_group_v<B> && is_lie_group_v<C>;
        static constexpr bool metric =
            is_metric_v<A> && is_metric_v<B> && is_metric_v<C>;
        static constexpr bool tensor =
            is_tensor_v<A> && is_tensor_v<B> && is_tensor_v<C>;
    };

    template <typename EXPR, typename ScalarT, typename C,
              template <typename, my_size_t, typename> class Op>
    struct algebraic_traits<ScalarFmaExpr<EXPR, ScalarT, C, Op>>
    {
        static constexpr bool vector_space = is_vector_space_v<EXPR> && is_vector_space_v<C>;
        static constexpr bool algebra = is_algebra_v<EXPR> && is_algebra_v<C>;
        static constexpr bool lie_group = false;
        static constexpr bool metric = is_metric_v<EXPR> && is_metric_v<C>;
        static constexpr bool tensor = is_tensor_v<EXPR> && is_tensor_v<C>;
    };
} // namespace algebra