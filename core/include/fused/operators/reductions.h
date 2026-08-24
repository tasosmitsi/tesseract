#pragma once
#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/kernel_ops/kernel_ops.h"
#include "algebra/algebraic_traits.h"

/**
 * @file reductions.h
 * @brief Whole-tensor reductions to a scalar.
 *
 * Not to be confused with the element-wise min/max in operators/min_max.h:
 * min(expr) returns a scalar, min(expr, expr) returns an expression.
 */

/// @brief Smallest element of the tensor.
template <typename Expr>
    requires(algebra::is_tensor_v<Expr> &&
             !algebra::is_algebra_v<Expr>)
typename Expr::value_type min(const BaseExpr<Expr> &expr) noexcept
{
    return KernelOps<typename Expr::value_type, BITS, DefaultArch>::reduce_min(expr.derived());
}

/// @brief Largest element of the tensor.
template <typename Expr>
    requires(algebra::is_tensor_v<Expr> &&
             !algebra::is_algebra_v<Expr>)
typename Expr::value_type max(const BaseExpr<Expr> &expr) noexcept
{
    return KernelOps<typename Expr::value_type, BITS, DefaultArch>::reduce_max(expr.derived());
}

/// @brief Sum of every element of the tensor.
template <typename Expr>
    requires(algebra::is_tensor_v<Expr> &&
             !algebra::is_algebra_v<Expr>)
typename Expr::value_type sum(const BaseExpr<Expr> &expr) noexcept
{
    return KernelOps<typename Expr::value_type, BITS, DefaultArch>::reduce_sum(expr.derived());
}
