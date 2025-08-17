#pragma once
#include "BaseExpr.h"
#include "Operations.h"
#include "ops/op_traits.h"

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
    ScalarExprRHS(const EXPR &expr, T scalar) : _expr(expr), _scalar(scalar) {}

    template <typename... Indices>
    T operator()(Indices... indices) const
    {
        return Op<T, Bits, GenericArch>::apply(_expr(indices...), _scalar); // expr op scalar
    }

    template <my_size_t length>
    type evalu(my_size_t (&indices)[length]) const
    {
        return Op<T, Bits, Arch>::apply(_expr.evalu(indices), _scalar);
    }

    my_size_t getNumDims() const
    {
        return _expr.getNumDims();
    }

    my_size_t getDim(my_size_t i) const
    {
        return _expr.getDim(i);
    }
};

template <typename EXPR, template <typename, my_size_t, typename> class Op, typename T, my_size_t Bits, typename Arch>
class ScalarExprLHS : public BaseExpr<ScalarExprLHS<EXPR, Op, T, Bits, Arch>, T>
{
    const EXPR &_expr;
    T _scalar;
    using type = typename Op<T, Bits, Arch>::type; // alias for easier usage

public:
    ScalarExprLHS(const EXPR &expr, T scalar) : _expr(expr), _scalar(scalar) {}

    template <typename... Indices>
    T operator()(Indices... indices) const
    {
        return Op<T, Bits, GenericArch>::apply(_scalar, _expr(indices...)); // scalar op expr
    }

    template <my_size_t length>
    type evalu(my_size_t (&indices)[length]) const
    {
        return Op<T, Bits, Arch>::apply(_scalar, _expr.evalu(indices));
    }

    my_size_t getNumDims() const
    {
        return _expr.getNumDims();
    }

    my_size_t getDim(my_size_t i) const
    {
        return _expr.getDim(i);
    }
};
