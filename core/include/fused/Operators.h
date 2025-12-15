#pragma once
#include "config.h"
#include "fused/BinaryExpr.h"
#include "fused/ScalarExpr.h"
#include "fused/Operations.h"

template <typename Expr1, typename Expr2>
inline void checkDimsMatch(const Expr1 &lhs, const Expr2 &rhs, const std::string &opName) // TODO: conditionally noexcept
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
operator+(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+");
#endif
    return BinaryExpr<LHS, RHS, Add, T, BITS, DefaultArch>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Sub, T, BITS, DefaultArch>
operator-(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator-");
#endif
    return BinaryExpr<LHS, RHS, Sub, T, BITS, DefaultArch>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Mul, T, BITS, DefaultArch>
operator*(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator*");
#endif
    return BinaryExpr<LHS, RHS, Mul, T, BITS, DefaultArch>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Div, T, BITS, DefaultArch>
operator/(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator/");
#endif
    return BinaryExpr<LHS, RHS, Div, T, BITS, DefaultArch>(lhs.derived(), rhs.derived());
}

// matrix + scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Add, T, BITS, DefaultArch>
operator+(const BaseExpr<LHS, T> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, Add, T, BITS, DefaultArch>(lhs.derived(), scalar);
}

// scalar + matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprRHS<RHS, Add, T, BITS, DefaultArch>
operator+(T scalar, const BaseExpr<RHS, T> &rhs) noexcept
{
    return ScalarExprRHS<RHS, Add, T, BITS, DefaultArch>(rhs.derived(), scalar);
}

// Override operator- to get the negative
template <typename RHS, typename T>
ScalarExprLHS<RHS, Sub, T, BITS, DefaultArch>
operator-(const BaseExpr<RHS, T> &expr) noexcept
{
    return ScalarExprLHS<RHS, Sub, T, BITS, DefaultArch>(expr.derived(), T(0)); // Negation is like subtracting from zero
}

// matrix - scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Sub, T, BITS, DefaultArch>
operator-(const BaseExpr<LHS, T> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, Sub, T, BITS, DefaultArch>(lhs.derived(), scalar);
}

// scalar - matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprLHS<RHS, Sub, T, BITS, DefaultArch>
operator-(T scalar, const BaseExpr<RHS, T> &rhs) noexcept
{
    return ScalarExprLHS<RHS, Sub, T, BITS, DefaultArch>(rhs.derived(), scalar);
}

// matrix * scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Mul, T, BITS, DefaultArch>
operator*(const BaseExpr<LHS, T> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, Mul, T, BITS, DefaultArch>(lhs.derived(), scalar);
}

// scalar * matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprRHS<RHS, Mul, T, BITS, DefaultArch>
operator*(T scalar, const BaseExpr<RHS, T> &rhs) noexcept
{
    return ScalarExprRHS<RHS, Mul, T, BITS, DefaultArch>(rhs.derived(), scalar);
}

// matrix / scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Div, T, BITS, DefaultArch>
operator/(const BaseExpr<LHS, T> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, Div, T, BITS, DefaultArch>(lhs.derived(), scalar);
}

// scalar / matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprLHS<RHS, Div, T, BITS, DefaultArch>
operator/(T scalar, const BaseExpr<RHS, T> &rhs) noexcept
{
    return ScalarExprLHS<RHS, Div, T, BITS, DefaultArch>(rhs.derived(), scalar);
}

template <typename LHS, typename RHS, typename T>
bool operator==(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs) // TODO: conditionally noexcept
{
    // Compile-time check that both expressions have the same number of dimensions
    // static_assert(LHS::N == RHS::N, "Cannot compare tensors with different number of dimensions");

    // runtime check that all dimensions are the same
#if defined(RUNTIME_CHECK_DIMENTIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator== or operator!=");
#endif

    // TODO: optimize this by checking using SIMD where possible
    // check the actual elements
    my_size_t indices[MAX_DIMS] = {0};
    for (my_size_t i = 0; i < lhs.derived().getTotalSize(); ++i)
    {
        // increment the indices using for loop
        for (my_size_t j = 0; j < lhs.derived().getNumDims(); ++j)
        {
            if (indices[j] < lhs.derived().getDim(j) - 1)
            {
                indices[j]++;
                break;
            }
            else
            {
                indices[j] = 0;
            }
        }
        // use the () operator to access the elements
        if (std::abs(lhs.derived()(indices) - rhs.derived()(indices)) > T(PRECISION_TOLERANCE))
        {
            return false;
        }
    }
    return true;
}

template <typename LHS, typename RHS, typename T>
bool operator!=(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs) // TODO: conditionally noexcept
{
    return !(lhs == rhs);
}