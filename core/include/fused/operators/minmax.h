#pragma once
#include "config.h"
#include "fused/BinaryExpr.h"
#include "fused/ScalarExpr.h"
#include "fused/Operations.h"
#include "fused/operators/operators_common.h"
#include "simple_type_traits.h"
#include "algebra/algebraic_traits.h"

// ===============================
// Min/Max Operators
// ===============================

// tensor min tensor
template <typename LHS, typename RHS>
    requires(
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Min>
min(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "min");
#endif
    return BinaryExpr<LHS, RHS, Min>(lhs.derived(), rhs.derived());
}

// tensor max tensor
template <typename LHS, typename RHS>
    requires(
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Max>
max(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "max");
#endif
    return BinaryExpr<LHS, RHS, Max>(lhs.derived(), rhs.derived());
}

// min(tensor, scalar)
template <typename LHS, typename T>
    requires(algebra::is_tensor_v<LHS> &&
             !algebra::is_algebra_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Min>
min(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Min>(lhs.derived(), scalar);
}

// min(scalar, tensor) — commutative
template <typename RHS, typename T>
    requires(algebra::is_tensor_v<RHS> &&
             !algebra::is_algebra_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Min>
min(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Min>(rhs.derived(), scalar);
}

// max(tensor, scalar)
template <typename LHS, typename T>
    requires(algebra::is_tensor_v<LHS> &&
             !algebra::is_algebra_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Max>
max(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Max>(lhs.derived(), scalar);
}

// max(scalar, tensor) — commutative
template <typename RHS, typename T>
    requires(algebra::is_tensor_v<RHS> &&
             !algebra::is_algebra_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Max>
max(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Max>(rhs.derived(), scalar);
}