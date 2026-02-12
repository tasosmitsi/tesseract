#pragma once

// Forward declare BinaryExpr
template <typename LHS, typename RHS, template <typename, my_size_t, typename> class Op>
class BinaryExpr;

namespace expression
{
    // Propagate traits for BinaryExpr
    template <typename LHS, typename RHS,
              template <typename, my_size_t, typename> class Op>
    struct traits<BinaryExpr<LHS, RHS, Op>>
    {
        static constexpr bool IsPermuted =
            traits<LHS>::IsPermuted || traits<RHS>::IsPermuted;

        static constexpr bool IsContiguous =
            traits<LHS>::IsContiguous && traits<RHS>::IsContiguous;
    };
} // namespace expression
