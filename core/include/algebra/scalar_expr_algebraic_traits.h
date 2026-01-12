#pragma once

/*
    A scalar expression inherits the tensor-ness of the tensor operand:
        ✔ tensor * scalar → tensor
        ✔ scalar * tensor → tensor
        ✔ vector space preserved
        ✔ algebra preserved only if tensor supports it
 */
template <typename EXPR,
          typename ScalarT,
          template <typename, my_size_t, typename> class Op>
class ScalarExprRHS;

template <typename EXPR,
          typename ScalarT,
          template <typename, my_size_t, typename> class Op>
class ScalarExprLHS;

namespace algebra
{
    template <typename EXPR,
              typename ScalarT,
              template <typename, my_size_t, typename> class Op>
    struct algebraic_traits<ScalarExprRHS<EXPR, ScalarT, Op>>
    {
        static constexpr bool vector_space = is_vector_space_v<EXPR>;
        static constexpr bool algebra = is_algebra_v<EXPR>;
        static constexpr bool lie_group = false;
        static constexpr bool metric = is_metric_v<EXPR>;
        static constexpr bool tensor = is_tensor_v<EXPR>;
    };

    template <typename EXPR,
              typename ScalarT,
              template <typename, my_size_t, typename> class Op>
    struct algebraic_traits<ScalarExprLHS<EXPR, ScalarT, Op>>
    {
        static constexpr bool vector_space = is_vector_space_v<EXPR>;
        static constexpr bool algebra = is_algebra_v<EXPR>;
        static constexpr bool lie_group = false;
        static constexpr bool metric = is_metric_v<EXPR>;
        static constexpr bool tensor = is_tensor_v<EXPR>;
    };
} // namespace algebra
