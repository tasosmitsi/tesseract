#pragma once
#include "config.h"
#include "fused/BinaryExpr.h"
#include "fused/ScalarExpr.h"
#include "fused/Operations.h"
#include "fused/operators/operators_common.h"
#include "simple_type_traits.h"
#include "algebra/algebraic_traits.h"

/**
 * @file min_max.h
 * @brief Element-wise min/max and clamp over tensors.
 *
 * These build expressions rather than computing, so min(max(v, lo), hi)
 * evaluates in a single pass. Constrained to tensors and not general algebras,
 * the same way the Hadamard operator* is.
 *
 * Not to be confused with the reductions of the same name in
 * operators/reductions.h: min(expr) returns a scalar, min(expr, expr) returns
 * an expression.
 */

/**
 * @brief Element-wise minimum of two tensors.
 * @throws if the dimensions do not match and runtime checks are enabled.
 */
template <typename LHS, typename RHS>
    requires(
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Min>
min(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "min");
#endif
    return BinaryExpr<LHS, RHS, Min>(lhs.derived(), rhs.derived());
}

/**
 * @brief Element-wise maximum of two tensors.
 * @throws if the dimensions do not match and runtime checks are enabled.
 */
template <typename LHS, typename RHS>
    requires(
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Max>
max(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "max");
#endif
    return BinaryExpr<LHS, RHS, Max>(lhs.derived(), rhs.derived());
}

/// @brief Cap every element at @p scalar.
template <typename LHS, typename T>
    requires(algebra::is_tensor_v<LHS> &&
             !algebra::is_algebra_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Min>
min(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Min>(lhs.derived(), scalar);
}

/// @brief Cap every element at @p scalar. Min is commutative, so this forwards.
template <typename RHS, typename T>
    requires(algebra::is_tensor_v<RHS> &&
             !algebra::is_algebra_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Min>
min(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Min>(rhs.derived(), scalar);
}

/// @brief Raise every element to at least @p scalar.
template <typename LHS, typename T>
    requires(algebra::is_tensor_v<LHS> &&
             !algebra::is_algebra_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Max>
max(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Max>(lhs.derived(), scalar);
}

/// @brief Raise every element to at least @p scalar. Max is commutative, so this forwards.
template <typename RHS, typename T>
    requires(algebra::is_tensor_v<RHS> &&
             !algebra::is_algebra_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Max>
max(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Max>(rhs.derived(), scalar);
}
