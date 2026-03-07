#pragma once

template <typename A, typename B, typename C,
          template <typename, my_size_t, typename> class Op>
class FmaExpr;

template <typename EXPR, typename ScalarT, typename C,
          template <typename, my_size_t, typename> class Op>
class ScalarFmaExpr;

namespace expression
{
    template <typename A, typename B, typename C,
              template <typename, my_size_t, typename> class Op>
    struct traits<FmaExpr<A, B, C, Op>>
    {
        static constexpr bool IsPermuted =
            traits<A>::IsPermuted || traits<B>::IsPermuted || traits<C>::IsPermuted;

        static constexpr bool IsContiguous =
            traits<A>::IsContiguous && traits<B>::IsContiguous && traits<C>::IsContiguous;

        static constexpr bool IsPhysical = false;
    };

    template <typename EXPR, typename ScalarT, typename C,
              template <typename, my_size_t, typename> class Op>
    struct traits<ScalarFmaExpr<EXPR, ScalarT, C, Op>>
    {
        static constexpr bool IsPermuted =
            traits<EXPR>::IsPermuted || traits<C>::IsPermuted;

        static constexpr bool IsContiguous =
            traits<EXPR>::IsContiguous && traits<C>::IsContiguous;

        static constexpr bool IsPhysical = false;
    };
} // namespace expression