#pragma once
#include "config.h"
#include "fused/BinaryExpr.h"
#include "fused/ScalarExpr.h"
#include "fused/Operations.h"
#include "fused/operators/operators_common.h"
#include "simple_type_traits.h"
#include "algebra/algebraic_traits.h"

// ===============================
// Operator Overloads
// ===============================

template <typename LHS, typename RHS>
    requires(algebra::is_vector_space_v<LHS> && algebra::is_vector_space_v<RHS>)
BinaryExpr<LHS, RHS, Add>
operator+(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+");
#endif
    return BinaryExpr<LHS, RHS, Add>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS>
    requires(algebra::is_vector_space_v<LHS> && algebra::is_vector_space_v<RHS>)
BinaryExpr<LHS, RHS, Sub>
operator-(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator-");
#endif
    return BinaryExpr<LHS, RHS, Sub>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS>
    requires( // for Hadamard product only it must be tensors, not general algebras
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Mul>
operator*(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator*");
#endif
    return BinaryExpr<LHS, RHS, Mul>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS>
    requires( // for Hadamard product (element-wise division) only it must be tensors, not general algebras
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Div>
operator/(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator/");
#endif
    return BinaryExpr<LHS, RHS, Div>(lhs.derived(), rhs.derived());
}

// matrix + scalar (scalar on RHS)
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Add>
operator+(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Add>(lhs.derived(), scalar);
}

// scalar + matrix (scalar on LHS)
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Add>
operator+(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Add>(rhs.derived(), scalar);
}

// Override operator- to get the negative
template <typename RHS>
    requires(algebra::is_vector_space_v<RHS>)
ScalarExprLHS<RHS, typename RHS::value_type, Sub>
operator-(const BaseExpr<RHS> &expr) noexcept
{
    using T = typename RHS::value_type;
    return ScalarExprLHS<RHS, T, Sub>(expr.derived(), T(0)); // Negation is like subtracting from zero
}

// matrix - scalar (scalar on RHS)
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Sub>
operator-(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Sub>(lhs.derived(), scalar);
}

// scalar - matrix (scalar on LHS)
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprLHS<RHS, T, Sub>
operator-(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprLHS<RHS, T, Sub>(rhs.derived(), scalar);
}

// matrix * scalar (scalar on RHS)
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Mul>
operator*(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Mul>(lhs.derived(), scalar);
}

// scalar * matrix (scalar on LHS)
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Mul>
operator*(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Mul>(rhs.derived(), scalar);
}

// matrix / scalar (scalar on RHS)
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Div>
operator/(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Div>(lhs.derived(), scalar);
}

// scalar / matrix (scalar on LHS)
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprLHS<RHS, T, Div>
operator/(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprLHS<RHS, T, Div>(rhs.derived(), scalar);
}