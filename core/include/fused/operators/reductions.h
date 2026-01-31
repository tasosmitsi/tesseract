#pragma once
#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/microkernels/kernel_ops.h"
#include "algebra/algebraic_traits.h"

// ===============================
// Min/Max/Sum Reduction Functions
// ===============================

template <typename Expr>
    requires(algebra::is_tensor_v<Expr> &&
             !algebra::is_algebra_v<Expr>)
typename Expr::value_type min(const BaseExpr<Expr> &expr)
{
    return KernelOps<Expr, BITS, DefaultArch>::reduce_min(expr.derived());
}

template <typename Expr>
    requires(algebra::is_tensor_v<Expr> &&
             !algebra::is_algebra_v<Expr>)
typename Expr::value_type max(const BaseExpr<Expr> &expr)
{
    return KernelOps<Expr, BITS, DefaultArch>::reduce_max(expr.derived());
}

template <typename Expr>
    requires(algebra::is_tensor_v<Expr> &&
             !algebra::is_algebra_v<Expr>)
typename Expr::value_type sum(const BaseExpr<Expr> &expr)
{
    return KernelOps<Expr, BITS, DefaultArch>::reduce_sum(expr.derived());
}