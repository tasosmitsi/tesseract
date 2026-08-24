#pragma once
#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "fused/kernel_ops/kernel_ops.h"
#include "fused/operators/minmax.h"
#include "algebra/algebraic_traits.h"
#include "helper_traits.h"

/**
 * @file reductions.h
 * @brief Reductions to a scalar, to a column vector, or to a row vector.
 *
 * min(expr) reduces the whole tensor to a scalar. min(expr, expr) is the
 * element-wise operator in minmax.h and returns an expression.
 *
 * Row reductions return [M, 1], column reductions [1, N]. Each the shape its
 * fold produces.
 */

// ===============================
// Whole-tensor reductions to a scalar
// ===============================

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

// ===============================
// Matrix axis reductions
// ===============================
//
// Both fold along the axis being collapsed. For an M x N matrix that is
// N-1 vector operations for the row reductions and M-1 for the column ones.
//
// The result is materialised not an expression.

namespace detail
{
    /// @brief Fold min across every column, leaving a per-row result.
    template <typename T, my_size_t M, my_size_t N, my_size_t... J>
    FORCE_INLINE void fold_min_columns(const FusedMatrix<T, M, N> &A,
                                       FusedVector<T, M> &out,
                                       index_seq<J...>) noexcept
    {
        // J... is 0..N-2, so column J+1 pairs against the running result
        out = A.template column<0>();
        ((out = min(out, A.template column<J + 1>())), ...);
    }

    /// @brief Fold element-wise product across every column.
    template <typename T, my_size_t M, my_size_t N, my_size_t... J>
    FORCE_INLINE void fold_mul_columns(const FusedMatrix<T, M, N> &A,
                                       FusedVector<T, M> &out,
                                       index_seq<J...>) noexcept
    {
        out = A.template column<0>();
        ((out = out * A.template column<J + 1>()), ...);
    }

    /// @brief Fold min across every row, leaving a per-column result.
    template <typename T, my_size_t M, my_size_t N, my_size_t... I>
    FORCE_INLINE void fold_min_rows(const FusedMatrix<T, M, N> &A,
                                    FusedMatrix<T, 1, N> &out,
                                    index_seq<I...>) noexcept
    {
        out = A.template row<0>();
        ((out = min(out, A.template row<I + 1>())), ...);
    }

    /// @brief Fold element-wise product across every row.
    template <typename T, my_size_t M, my_size_t N, my_size_t... I>
    FORCE_INLINE void fold_mul_rows(const FusedMatrix<T, M, N> &A,
                                    FusedMatrix<T, 1, N> &out,
                                    index_seq<I...>) noexcept
    {
        out = A.template row<0>();
        ((out = out * A.template row<I + 1>()), ...);
    }
} // namespace detail

/**
 * @brief Minimum of each row, as an M-element column vector.
 *
 * The Gödel t-norm: with rows holding one rule's antecedent memberships, this
 * is the rule's firing strength under AND-as-min. Rows that should not
 * contribute must be filled with 1.0, the identity, or 0.0. There is no active-row
 * count, since masking would cost a branch in the inner loop to save at most a
 * few SIMD operations. TODO: consider a masked version for large matrices with many inactive rows.
 */
template <typename T, my_size_t M, my_size_t N>
FusedVector<T, M> reduce_min_rows(const FusedMatrix<T, M, N> &A) noexcept
{
    FusedVector<T, M> result;
    detail::fold_min_columns(A, result, typename make_index_seq<N - 1>::type{});
    return result;
}

/**
 * @brief Product of each row, as an M-element column vector.
 *
 * The product t-norm, the other common choice for AND across a rule's
 * antecedents. Same row-filling rule as reduce_min_rows.
 */
template <typename T, my_size_t M, my_size_t N>
FusedVector<T, M> reduce_prod_rows(const FusedMatrix<T, M, N> &A) noexcept
{
    FusedVector<T, M> result;
    detail::fold_mul_columns(A, result, typename make_index_seq<N - 1>::type{});
    return result;
}

/// @brief Minimum of each column, as a [1, N] row vector.
template <typename T, my_size_t M, my_size_t N>
FusedMatrix<T, 1, N> reduce_min_cols(const FusedMatrix<T, M, N> &A) noexcept
{
    FusedMatrix<T, 1, N> result;
    detail::fold_min_rows(A, result, typename make_index_seq<M - 1>::type{});
    return result;
}

/// @brief Product of each column, as a [1, N] row vector.
template <typename T, my_size_t M, my_size_t N>
FusedMatrix<T, 1, N> reduce_prod_cols(const FusedMatrix<T, M, N> &A) noexcept
{
    FusedMatrix<T, 1, N> result;
    detail::fold_mul_rows(A, result, typename make_index_seq<M - 1>::type{});
    return result;
}
