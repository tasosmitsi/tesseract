#pragma once
#include "fused/BaseExpr.h"
#include "fused/Operations.h"

// ===============================
// Scalar Expression Template
// ===============================
template <typename EXPR, template <typename, my_size_t, typename> class Op, typename T, my_size_t Bits, typename Arch>
class ScalarExprRHS : public BaseExpr<ScalarExprRHS<EXPR, Op, T, Bits, Arch>, T>
{
    const EXPR &_expr;
    T _scalar;
    using type = typename Op<T, Bits, Arch>::type; // alias for easier usage

public:
    static constexpr my_size_t NumDims = EXPR::NumDims;
    static constexpr const my_size_t *Dim = EXPR::Dim;
    static constexpr my_size_t TotalSize = EXPR::TotalSize;

    ScalarExprRHS(const EXPR &expr, T scalar) : _expr(expr), _scalar(scalar) {}

    template <my_size_t length>
    T operator()(my_size_t (&indices)[length]) const noexcept
    {
        return Op<T, Bits, GenericArch>::apply(_expr(indices), _scalar); // expr op scalar
    }

    // template <my_size_t length>
    type evalu(const my_size_t flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(_expr.evalu(flat), _scalar);
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
    inline T operator()(const my_size_t *indices) const noexcept
    {
        return Op<T, Bits, GenericArch>::apply(_expr(indices), _scalar);
    }
};

template <typename EXPR, template <typename, my_size_t, typename> class Op, typename T, my_size_t Bits, typename Arch>
class ScalarExprLHS : public BaseExpr<ScalarExprLHS<EXPR, Op, T, Bits, Arch>, T>
{
    const EXPR &_expr;
    T _scalar;
    using type = typename Op<T, Bits, Arch>::type; // alias for easier usage

public:
    static constexpr my_size_t NumDims = EXPR::NumDims;
    static constexpr const my_size_t *Dim = EXPR::Dim;
    static constexpr my_size_t TotalSize = EXPR::TotalSize;

    ScalarExprLHS(const EXPR &expr, T scalar) : _expr(expr), _scalar(scalar) {}

    template <my_size_t length>
    T operator()(my_size_t (&indices)[length]) const noexcept
    {
        return Op<T, Bits, GenericArch>::apply(_scalar, _expr(indices)); // expr op scalar
    }

    // template <my_size_t length>
    type evalu(const my_size_t flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(_scalar, _expr.evalu(flat));
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
    inline T operator()(const my_size_t *indices) const noexcept
    {
        return Op<T, Bits, GenericArch>::apply(_scalar, _expr(indices));
    }
};
