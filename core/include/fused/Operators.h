#pragma once
#include "BinaryExpr.h"
#include "ScalarExpr.h"
#include "Operations.h"

template <typename Expr1, typename Expr2>
inline void checkDimsMatch(const Expr1 &lhs, const Expr2 &rhs, const std::string &opName)
{
#ifdef RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH
    if (lhs.getNumDims() != rhs.getNumDims())
        MyErrorHandler::error(opName + ": dimension count mismatch");
#endif

#ifdef RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH
    for (my_size_t i = 0; i < lhs.getNumDims(); ++i)
    {
        if (lhs.getDim(i) != rhs.getDim(i))
            MyErrorHandler::error(opName + ": dimension size mismatch at dimension " + std::to_string(i));
    }
#endif
}

// ===============================
// Operator Overloads
// ===============================
template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Add, T, BITS, DefaultArch>
operator+(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs)
{
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    const auto &lhsDerived = lhs.derived();
    const auto &rhsDerived = rhs.derived();

    checkDimsMatch(lhsDerived, rhsDerived, "operator+");

    return BinaryExpr<LHS, RHS, Add, T, BITS, DefaultArch>(lhsDerived, rhsDerived);
#else
    return BinaryExpr<LHS, RHS, Add, T, BITS, DefaultArch>(lhs.derived(), rhs.derived());
#endif
}

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Sub, T, BITS, DefaultArch>
operator-(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs)
{
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)

    const auto &lhsDerived = lhs.derived();
    const auto &rhsDerived = rhs.derived();

    checkDimsMatch(lhsDerived, rhsDerived, "operator-");

    return BinaryExpr<LHS, RHS, Sub, T, BITS, DefaultArch>(lhsDerived, rhsDerived);
#else
    return BinaryExpr<LHS, RHS, Sub, T, BITS, DefaultArch>(lhs.derived(), rhs.derived());
#endif
}

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Mul, T, BITS, DefaultArch>
operator*(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs)
{
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    const auto &lhsDerived = lhs.derived();
    const auto &rhsDerived = rhs.derived();

    checkDimsMatch(lhsDerived, rhsDerived, "operator*");
    return BinaryExpr<LHS, RHS, Mul, T, BITS, DefaultArch>(lhsDerived, rhsDerived);
#else
    return BinaryExpr<LHS, RHS, Mul, T, BITS, DefaultArch>(lhs.derived(), rhs.derived());
#endif
}

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Div, T, BITS, DefaultArch>
operator/(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs)
{
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)

    const auto &lhsDerived = lhs.derived();
    const auto &rhsDerived = rhs.derived();

    checkDimsMatch(lhsDerived, rhsDerived, "operator/");

    return BinaryExpr<LHS, RHS, Div, T, BITS, DefaultArch>(lhsDerived, rhsDerived);
#else
    return BinaryExpr<LHS, RHS, Div, T, BITS, DefaultArch>(lhs.derived(), rhs.derived());
#endif
}

// matrix + scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Add, T, BITS, DefaultArch>
operator+(const BaseExpr<LHS, T> &lhs, T scalar)
{
    return ScalarExprRHS<LHS, Add, T, BITS, DefaultArch>(lhs.derived(), scalar);
}

// scalar + matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprRHS<RHS, Add, T, BITS, DefaultArch>
operator+(T scalar, const BaseExpr<RHS, T> &rhs)
{
    return ScalarExprRHS<RHS, Add, T, BITS, DefaultArch>(rhs.derived(), scalar);
}

// Override operator- to get the negative
template <typename RHS, typename T>
ScalarExprLHS<RHS, Sub, T, BITS, DefaultArch>
operator-(const BaseExpr<RHS, T> &expr)
{
    return ScalarExprLHS<RHS, Sub, T, BITS, DefaultArch>(expr.derived(), T(0)); // Negation is like subtracting from zero
}

// matrix - scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Sub, T, BITS, DefaultArch>
operator-(const BaseExpr<LHS, T> &lhs, T scalar)
{
    return ScalarExprRHS<LHS, Sub, T, BITS, DefaultArch>(lhs.derived(), scalar);
}

// scalar - matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprLHS<RHS, Sub, T, BITS, DefaultArch>
operator-(T scalar, const BaseExpr<RHS, T> &rhs)
{
    return ScalarExprLHS<RHS, Sub, T, BITS, DefaultArch>(rhs.derived(), scalar);
}

// matrix * scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Mul, T, BITS, DefaultArch>
operator*(const BaseExpr<LHS, T> &lhs, T scalar)
{
    return ScalarExprRHS<LHS, Mul, T, BITS, DefaultArch>(lhs.derived(), scalar);
}

// scalar * matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprRHS<RHS, Mul, T, BITS, DefaultArch>
operator*(T scalar, const BaseExpr<RHS, T> &rhs)
{
    return ScalarExprRHS<RHS, Mul, T, BITS, DefaultArch>(rhs.derived(), scalar);
}

// matrix / scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Div, T, BITS, DefaultArch>
operator/(const BaseExpr<LHS, T> &lhs, T scalar)
{
    return ScalarExprRHS<LHS, Div, T, BITS, DefaultArch>(lhs.derived(), scalar);
}

// scalar / matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprLHS<RHS, Div, T, BITS, DefaultArch>
operator/(T scalar, const BaseExpr<RHS, T> &rhs)
{
    return ScalarExprLHS<RHS, Div, T, BITS, DefaultArch>(rhs.derived(), scalar);
}
