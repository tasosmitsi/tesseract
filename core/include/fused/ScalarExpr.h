#pragma once
#include "BaseExpr.h"

// ===============================
// Scalar Expression Template
// ===============================
// template <typename EXPR, template <typename> class Op, typename T>
// class ScalarExpr : public BaseExpr<ScalarExpr<EXPR, Op, T>, T>
// {
//     const EXPR &_expr;
//     T _scalar;

// public:
//     ScalarExpr(const EXPR &expr, T scalar) : _expr(expr), _scalar(scalar) {}

//     template <typename... Indices>
//     T operator()(Indices... indices) const
//     {
//         return Op<T>::apply(_expr(indices...), _scalar);
//     }
// };

template <typename EXPR, template <typename> class Op, typename T>
class ScalarExprRHS : public BaseExpr<ScalarExprRHS<EXPR, Op, T>, T>
{
    const EXPR &_expr;
    T _scalar;

public:
    ScalarExprRHS(const EXPR &expr, T scalar) : _expr(expr), _scalar(scalar) {}

    template <typename... Indices>
    T operator()(Indices... indices) const
    {
        return Op<T>::apply(_expr(indices...), _scalar); // expr op scalar
    }

    template <my_size_t length>
    __m256 evalu(my_size_t (&indices)[length]) const
    {
        return Op<__m256>::apply(_expr.evalu(indices), _scalar);
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

template <typename EXPR, template <typename> class Op, typename T>
class ScalarExprLHS : public BaseExpr<ScalarExprLHS<EXPR, Op, T>, T>
{
    const EXPR &_expr;
    T _scalar;

public:
    ScalarExprLHS(const EXPR &expr, T scalar) : _expr(expr), _scalar(scalar) {}

    template <typename... Indices>
    T operator()(Indices... indices) const
    {
        return Op<T>::apply(_scalar, _expr(indices...)); // scalar op expr
    }

    template <my_size_t length>
    __m256 evalu(my_size_t (&indices)[length]) const
    {
        return Op<__m256>::apply(_scalar, _expr.evalu(indices));
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

// ===============================
// Scalar Expression Template (with shared_ptr)
// ===============================
// template<typename EXPR>
// using ExprPtr = std::shared_ptr<const EXPR>;

// template <typename EXPR, template <typename> class Op, typename T>
// class ScalarExpr : public BaseExpr<ScalarExpr<EXPR, Op, T>, T>
// {
//     ExprPtr<EXPR> _expr;
//     T _scalar;

// public:
//     ScalarExpr(ExprPtr<EXPR> expr, T scalar)
//         : _expr(std::move(expr)), _scalar(scalar) {}

//     T operator()(my_size_t idx) const
//     {
//         return Op<T>::apply((*_expr)(idx), _scalar);
//     }

//     T operator()(my_size_t i, my_size_t j) const
//     {
//         return Op<T>::apply((*_expr)(i, j), _scalar);
//     }

//     template <typename... Indices>
//     T operator()(Indices... indices) const
//     {
//         return Op<T>::apply((*_expr)(indices...), _scalar);
//     }
// };
