/**
 * @file kernel_ops.h
 * @brief Façade for higher-level kernel operations built on top of microkernels.
 *
 * Delegates to specialized sub-modules:
 *   - kernel_eval.h     — expression evaluation (contiguous / permuted)
 *   - kernel_reduce.h   — reductions (min, max, sum)
 *   - kernel_compare.h  — approximate equality comparisons
 *   - kernel_dot.h      — dot products (contiguous / strided) for einsum
 *   - kernel_helpers.h  — shared SIMD utilities (fmadd_safe)
 *
 * Callers should include only this file.
 */
#ifndef KERNEL_OPS_H
#define KERNEL_OPS_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/kernel_ops/kernel_helpers.h"
#include "fused/kernel_ops/kernel_eval.h"
#include "fused/kernel_ops/kernel_reduce.h"
#include "fused/kernel_ops/kernel_compare.h"
#include "fused/kernel_ops/kernel_dot.h"

template <typename T, my_size_t Bits, typename Arch>
struct KernelOps
{
    using K = Microkernel<T, Bits, Arch>;
    static constexpr my_size_t simdWidth = K::simdWidth;

    // ========================================================================
    // Evaluation
    // ========================================================================

    /**
     * @brief Evaluation: Dispatch: pick contiguous or permuted eval based on expression layout.
     * 
     */
    template <typename Expr>
    FORCE_INLINE static void eval(T *output, const Expr &expr) noexcept
    {
        detail::KernelEval<T, Bits, Arch>::eval(output, expr);
    }

    // ========================================================================
    // Reductions
    // ========================================================================

    template <typename Expr>
    FORCE_INLINE static T reduce_min(const Expr &expr) noexcept
    {
        return detail::KernelReduce<T, Bits, Arch>::reduce_min(expr);
    }

    template <typename Expr>
    FORCE_INLINE static T reduce_max(const Expr &expr) noexcept
    {
        return detail::KernelReduce<T, Bits, Arch>::reduce_max(expr);
    }

    template <typename Expr>
    FORCE_INLINE static T reduce_sum(const Expr &expr) noexcept
    {
        return detail::KernelReduce<T, Bits, Arch>::reduce_sum(expr);
    }

    // ========================================================================
    // Comparisons
    // ========================================================================

    /**
     * @brief Check if all logical elements of two expressions are approximately equal.
     */
    template <typename Expr1, typename Expr2>
    FORCE_INLINE static bool reduce_all_approx_equal(
        const Expr1 &lhs,
        const Expr2 &rhs,
        T tolerance) noexcept
    {
        return detail::KernelCompare<T, Bits, Arch>::reduce_all_approx_equal(lhs, rhs, tolerance);
    }

    // ========================================================================
    // Dot Products
    // ========================================================================

    /**
     * @brief Dispatch dot product based on stride values.
     */
    template <typename Expr1, typename Expr2>
    FORCE_INLINE static T dot(
        const Expr1 &expr1, my_size_t base1, my_size_t stride1,
        const Expr2 &expr2, my_size_t base2, my_size_t stride2,
        my_size_t len) noexcept
    {
        return detail::KernelDot<T, Bits, Arch>::dot(
            expr1, base1, stride1,
            expr2, base2, stride2,
            len);
    }

    /**
     * @brief Naive scalar dot product for testing/validation.
     */
    template <typename Expr1, typename Expr2>
    FORCE_INLINE static T naive_dot_physical(
        const Expr1 &expr1, my_size_t base1, my_size_t stride1,
        const Expr2 &expr2, my_size_t base2, my_size_t stride2,
        my_size_t len) noexcept
    {
        return detail::KernelDot<T, Bits, Arch>::naive_dot_physical(
            expr1, base1, stride1,
            expr2, base2, stride2,
            len);
    }
};

#endif // KERNEL_OPS_H
