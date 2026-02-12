#pragma once

template <typename EXPR,
          typename ScalarT,
          template <typename, my_size_t, typename> class Op>
class ScalarExprRHS;

template <typename EXPR,
          typename ScalarT,
          template <typename, my_size_t, typename> class Op>
class ScalarExprLHS;

namespace expression
{
    template <typename EXPR,
              typename ScalarT,
              template <typename, my_size_t, typename> class Op>
    struct traits<ScalarExprRHS<EXPR, ScalarT, Op>>
    {
        static constexpr bool IsPermuted = traits<EXPR>::IsPermuted;
        static constexpr bool IsContiguous = traits<EXPR>::IsContiguous;
    };

    template <typename EXPR,
              typename ScalarT,
              template <typename, my_size_t, typename> class Op>
    struct traits<ScalarExprLHS<EXPR, ScalarT, Op>>
    {
        static constexpr bool IsPermuted = traits<EXPR>::IsPermuted;
        static constexpr bool IsContiguous = traits<EXPR>::IsContiguous;
    };
} // namespace expression
