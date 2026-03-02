/**
 * @file kernel_compare.h
 * @brief Comparison operations — approximate equality between expressions.
 *
 * Dispatches based on layout compatibility:
 *   - Both contiguous, same padding → fast physical slice iteration
 *   - Otherwise → logical flat iteration (handles any layout combination)
 *
 * Both expressions must have the same logical dimensions.
 */
#ifndef KERNEL_COMPARE_H
#define KERNEL_COMPARE_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "expression_traits/expression_traits.h"

namespace detail
{

    template <typename T, my_size_t Bits, typename Arch>
    struct KernelCompare
    {
        using K = Microkernel<T, Bits, Arch>;
        static constexpr my_size_t simdWidth = K::simdWidth;

        // ========================================================================
        // Public API
        // ========================================================================

        /**
         * @brief Check if all logical elements of two expressions are approximately equal.
         *
         * Dispatches based on layout compatibility:
         *   - Both unpermuted, same padding → fast physical slice iteration
         *   - Otherwise → logical flat iteration (handles any layout combination)
         *
         * Both expressions must have the same logical dimensions.
         * TODO: do we need to check this at compile time, or is it guaranteed by the caller? think...!
         *
         * @param tolerance Approximation tolerance for floating-point comparisons
         * @return true if all corresponding logical elements are approximately equal
         */
        template <typename Expr1, typename Expr2>
        FORCE_INLINE static bool reduce_all_approx_equal(
            const Expr1 &lhs,
            const Expr2 &rhs,
            T tolerance) noexcept
        {
            if constexpr (expression::traits<Expr1>::IsContiguous &&
                          expression::traits<Expr2>::IsContiguous)
            {
                // std::cout << "reduce_all_approx_equal: dispatching to contiguous path" << std::endl;
                return approx_equal_contiguous(lhs, rhs, tolerance);
            }
            else
            {
                // std::cout << "reduce_all_approx_equal: dispatching to logical path" << std::endl;
                return approx_equal_logical(lhs, rhs, tolerance);
            }
        }

    private:
        // ========================================================================
        // Contiguous path
        // ========================================================================

        /**
         * @brief Fast path — both expressions share the same physical layout.
         *
         * Iterates physical slices. evalu receives physical offsets.
         * Same slice geometry for both sides.
         */
        template <typename Expr1, typename Expr2>
        FORCE_INLINE static bool approx_equal_contiguous(
            const Expr1 &lhs,
            const Expr2 &rhs,
            T tolerance) noexcept
        {
            using ExprPadPolicy = typename Expr1::Layout::PadPolicyType;

            static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
            static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
            static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;
            static constexpr my_size_t simdSteps = lastDim / simdWidth;
            static constexpr my_size_t scalarStart = simdSteps * simdWidth;

            if constexpr (simdSteps > 0)
            {
                for (my_size_t slice = 0; slice < numSlices; ++slice)
                {
                    const my_size_t base = slice * paddedLastDim;
                    for (my_size_t i = 0; i < simdSteps; ++i)
                    {
                        auto lhs_vec = lhs.template evalu<T, Bits, Arch>(base + i * simdWidth);
                        auto rhs_vec = rhs.template evalu<T, Bits, Arch>(base + i * simdWidth);
                        if (!K::all_within_tolerance(lhs_vec, rhs_vec, tolerance))
                            return false;
                    }
                }
            }

            if constexpr (scalarStart < lastDim)
            {
                using ScalarK = Microkernel<T, 1, GENERICARCH>;
                for (my_size_t slice = 0; slice < numSlices; ++slice)
                {
                    const my_size_t base = slice * paddedLastDim;
                    for (my_size_t i = scalarStart; i < lastDim; ++i)
                    {
                        T lhs_val = lhs.template evalu<T, 1, GENERICARCH>(base + i);
                        T rhs_val = rhs.template evalu<T, 1, GENERICARCH>(base + i);
                        if (ScalarK::abs(lhs_val - rhs_val) > tolerance)
                            return false;
                    }
                }
            }

            return true;
        }

        // ========================================================================
        // Logical path
        // ========================================================================

        /**
         * @brief General path — one or both expressions are permuted.
         *
         * Iterates logical flat indices. Uses logical_evalu which propagates
         * correct logical-flat semantics through the entire expression tree,
         * handling the physical/logical index convention mismatch at each node.
         *
         *
         * Uses the OUTPUT-side slice geometry to track logical_flat while
         * skipping output padding gaps — same pattern as eval_vectorized_permuted.
         * But since we're comparing (not storing), we use Expr1's padding policy
         * as reference.
         *
         * TODO: Untested
         */
        template <typename Expr1, typename Expr2>
        FORCE_INLINE static bool approx_equal_logical(
            const Expr1 &lhs,
            const Expr2 &rhs,
            T tolerance) noexcept
        {
            // Use Expr1's logical dims to drive iteration
            // (both must have same logical dims)
            // TODO: enforce this at compile time? Or is it guaranteed anyways? think...!
            static constexpr my_size_t logicalSize = Expr1::TotalSize;

            using ScalarK = Microkernel<T, 1, GENERICARCH>;

            for (my_size_t i = 0; i < logicalSize; ++i)
            {
                T lhs_val = lhs.template logical_evalu<T, 1, GENERICARCH>(i);
                T rhs_val = rhs.template logical_evalu<T, 1, GENERICARCH>(i);
                if (ScalarK::abs(lhs_val - rhs_val) > tolerance)
                    return false;
            }

            return true;
        }
    };

} // namespace detail

#endif // KERNEL_COMPARE_H
