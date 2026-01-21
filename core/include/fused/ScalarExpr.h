#pragma once
#include "fused/BaseExpr.h"
#include "fused/Operations.h"
#include "simple_type_traits.h"

// ===============================
// Scalar Expression Template
// ===============================
template <
    typename EXPR,
    typename ScalarT,
    template <typename, my_size_t, typename> class Op>
class ScalarExprRHS : public BaseExpr<ScalarExprRHS<EXPR, ScalarT, Op>>
{
    // Compile-time check that EXPR value_type and ScalarT are the same
    static_assert(is_same_v<typename EXPR::value_type, ScalarT>,
                  "ScalarExprRHS: EXPR value_type and ScalarT must be the same");

    const EXPR &_expr;
    ScalarT _scalar;

public:
    static constexpr my_size_t NumDims = EXPR::NumDims;
    static constexpr const my_size_t *Dim = EXPR::Dim;
    static constexpr my_size_t TotalSize = EXPR::TotalSize;
    using value_type = typename EXPR::value_type;

    ScalarExprRHS(const EXPR &expr, ScalarT scalar) : _expr(expr), _scalar(scalar) {}

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return _expr.may_alias(output);
    }

    template <my_size_t length>
    inline auto operator()(my_size_t (&indices)[length]) const noexcept
    {
        using T = std::decay_t<decltype(_expr(indices))>;
        return Op<T, 0, GENERICARCH>::apply(
            _expr(indices),
            _scalar);
    }

    template <typename T, my_size_t Bits, typename Arch>
    inline typename Op<T, Bits, Arch>::type evalu(const my_size_t flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(
            _expr.template evalu<T, Bits, Arch>(flat),
            _scalar);
    }

    my_size_t getNumDims() const noexcept
    {
        return _expr.getNumDims();
    }

    my_size_t getDim(my_size_t i) const // TODO: conditionally noexcept
    {
        return _expr.getDim(i);
    }

    my_size_t getTotalSize() const noexcept
    {
        return _expr.getTotalSize();
    }

protected:
    inline auto operator()(const my_size_t *indices) const noexcept
    {
        using T = std::decay_t<decltype(_expr(indices))>;
        return Op<T, 0, GENERICARCH>::apply(
            _expr(indices),
            _scalar);
    }
};

template <
    typename EXPR,
    typename ScalarT,
    template <typename, my_size_t, typename> class Op>
class ScalarExprLHS : public BaseExpr<ScalarExprLHS<EXPR, ScalarT, Op>>
{
    // Compile-time check that EXPR value_type and ScalarT are the same
    static_assert(is_same_v<typename EXPR::value_type, ScalarT>,
                  "ScalarExprRHS: EXPR value_type and ScalarT must be the same");

    const EXPR &_expr;
    ScalarT _scalar;

public:
    static constexpr my_size_t NumDims = EXPR::NumDims;
    static constexpr const my_size_t *Dim = EXPR::Dim;
    static constexpr my_size_t TotalSize = EXPR::TotalSize;
    using value_type = typename EXPR::value_type;

    ScalarExprLHS(const EXPR &expr, ScalarT scalar) : _expr(expr), _scalar(scalar) {}

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return _expr.may_alias(output);
    }

    template <my_size_t length>
    auto operator()(my_size_t (&indices)[length]) const noexcept
    {
        using T = std::decay_t<decltype(_expr(indices))>;
        return Op<T, 0, GENERICARCH>::apply(
            _scalar,
            _expr(indices));
    }

    template <typename T, my_size_t Bits, typename Arch>
    inline typename Op<T, Bits, Arch>::type evalu(const my_size_t flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(
            _scalar,
            _expr.template evalu<T, Bits, Arch>(flat));
    }

    my_size_t getNumDims() const noexcept
    {
        return _expr.getNumDims();
    }

    my_size_t getDim(my_size_t i) const // TODO: conditionally noexcept
    {
        return _expr.getDim(i);
    }

    my_size_t getTotalSize() const noexcept
    {
        return _expr.getTotalSize();
    }

protected:
    inline auto operator()(const my_size_t *indices) const noexcept
    {
        using T = std::decay_t<decltype(_expr(indices))>;
        return Op<T, 0, GENERICARCH>::apply(
            _scalar,
            _expr(indices));
    }
};
