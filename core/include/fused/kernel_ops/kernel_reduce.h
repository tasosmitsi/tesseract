/**
 * @file kernel_reduce.h
 * @brief Reduction operations — min, max, sum over expression elements.
 *
 * Parameterized on ReduceOp enum. Dispatches based on expression layout:
 *   - Contiguous: physical slice iteration, SIMD + scalar tail
 *   - Logical:    flat logical index iteration, scalar only (for permuted views)
 *
 * ============================================================================
 * STRATEGY (contiguous path)
 * ============================================================================
 *
 * Physical memory is organized as numSlices × paddedLastDim, where only the
 * first lastDim elements per slice are logical data:
 *
 *   slice 0: [d d d d d P P P]
 *   slice 1: [d d d d d P P P]   d = data, P = padding
 *   slice 2: [d d d d d P P P]
 *          |← lastDim→|
 *          |←  paddedLastDim →|
 *
 * For a 3D tensor [2, 3, 5], numSlices = 2*3 = 6:
 *   slice 0 = [0,0,:]    slice 3 = [1,0,:]
 *   slice 1 = [0,1,:]    slice 4 = [1,1,:]
 *   slice 2 = [0,2,:]    slice 5 = [1,2,:]
 *
 * Per slice, SIMD processes simdWidth-aligned chunks, then a scalar tail
 * handles the remainder. Padding is never read.
 *
 * ============================================================================
 * GENERICARCH (SimdWidth=1): no padding, simdSteps=lastDim, no scalar tail.
 * Microkernel ops inline to plain scalar — same codegen as a manual loop.
 * ============================================================================
 */
#ifndef KERNEL_REDUCE_H
#define KERNEL_REDUCE_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "numeric_limits.h"
#include "expression_traits/expression_traits.h"

namespace detail
{

    // ============================================================================
    // ReduceOp enum — shared by all reduction machinery
    // ============================================================================

    enum class ReduceOp
    {
        Min,
        Max,
        Sum
    };

    template <typename T, my_size_t Bits, typename Arch>
    struct KernelReduce
    {
        using K = Microkernel<T, Bits, Arch>;
        static constexpr my_size_t simdWidth = K::simdWidth;

        // ========================================================================
        // Public API
        // ========================================================================

        template <typename Expr>
        FORCE_INLINE static T reduce_min(const Expr &expr) noexcept
        {
            return reduce<ReduceOp::Min>(expr);
        }

        template <typename Expr>
        FORCE_INLINE static T reduce_max(const Expr &expr) noexcept
        {
            return reduce<ReduceOp::Max>(expr);
        }

        template <typename Expr>
        FORCE_INLINE static T reduce_sum(const Expr &expr) noexcept
        {
            return reduce<ReduceOp::Sum>(expr);
        }

        // ========================================================================
        // Private implementation
        // ========================================================================

    private: // TODO: make private once KernelOps facade is the only caller
        // --- ReduceOp traits ---

        template <ReduceOp Op>
        FORCE_INLINE static T reduce_identity() noexcept
        {
            if constexpr (Op == ReduceOp::Min)
                return NumericLimits<T>::max();
            if constexpr (Op == ReduceOp::Max)
                return NumericLimits<T>::lowest();
            if constexpr (Op == ReduceOp::Sum)
                return T{0};
        }

        template <ReduceOp Op>
        FORCE_INLINE static typename K::VecType reduce_simd_combine(
            typename K::VecType a, typename K::VecType b) noexcept
        {
            if constexpr (Op == ReduceOp::Min)
                return K::min(a, b);
            if constexpr (Op == ReduceOp::Max)
                return K::max(a, b);
            if constexpr (Op == ReduceOp::Sum)
                return K::add(a, b);
        }

        template <ReduceOp Op>
        FORCE_INLINE static T reduce_scalar_combine(T a, T b) noexcept
        {
            using ScalarK = Microkernel<T, 1, GENERICARCH>;
            if constexpr (Op == ReduceOp::Min)
                return ScalarK::min(a, b);
            if constexpr (Op == ReduceOp::Max)
                return ScalarK::max(a, b);
            if constexpr (Op == ReduceOp::Sum)
                return ScalarK::add(a, b);
        }

        // --- Dispatch ---

        template <ReduceOp Op, typename Expr>
        FORCE_INLINE static T reduce(const Expr &expr) noexcept
        {
            if constexpr (!expression::traits<Expr>::IsPermuted)
            {
                // std::cout << "reduce_contiguous" << std::endl;
                return reduce_contiguous<Op>(expr);
            }
            else
            {
                // std::cout << "reduce_logical" << std::endl;
                return reduce_logical<Op>(expr);
            }
        }

        // --- Contiguous path — iterate physical slices, skip padding ---

        template <ReduceOp Op, typename Expr>
        FORCE_INLINE static T reduce_contiguous(const Expr &expr) noexcept
        {
            using ExprPadPolicy = typename Expr::Layout::PadPolicyType;

            static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
            static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
            static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;
            static constexpr my_size_t simdSteps = lastDim / simdWidth;
            static constexpr my_size_t scalarStart = simdSteps * simdWidth;

            T result = reduce_identity<Op>();

            if constexpr (simdSteps > 0)
            {
                typename K::VecType acc = K::set1(reduce_identity<Op>());

                for (my_size_t slice = 0; slice < numSlices; ++slice)
                {
                    const my_size_t base = slice * paddedLastDim;
                    for (my_size_t i = 0; i < simdSteps; ++i)
                        acc = reduce_simd_combine<Op>(
                            acc, expr.template evalu<T, Bits, Arch>(base + i * simdWidth));
                }

                alignas(DATA_ALIGNAS) T tmp[simdWidth];
                K::store(tmp, acc);

                for (my_size_t i = 0; i < simdWidth; ++i)
                    result = reduce_scalar_combine<Op>(result, tmp[i]);
            }

            if constexpr (scalarStart < lastDim)
            {
                for (my_size_t slice = 0; slice < numSlices; ++slice)
                {
                    const my_size_t base = slice * paddedLastDim;
                    for (my_size_t i = scalarStart; i < lastDim; ++i)
                        result = reduce_scalar_combine<Op>(
                            result, expr.template evalu<T, 1, GENERICARCH>(base + i));
                }
            }

            return result;
        }

        // --- Logical path — iterate logical flat indices, scalar only ---
        //
        // For permuted views. Consecutive logical flats are non-contiguous
        // in physical memory, so SIMD load would read wrong data.
        // Scalar evalu handles the remapping internally.

        template <ReduceOp Op, typename Expr>
        FORCE_INLINE static T reduce_logical(const Expr &expr) noexcept
        {
            static constexpr my_size_t totalSize = Expr::TotalSize;

            T result = reduce_identity<Op>();

            for (my_size_t i = 0; i < totalSize; ++i)
                result = reduce_scalar_combine<Op>(
                    result, expr.template logical_evalu<T, 1, GENERICARCH>(i));

            return result;
        }
    };

} // namespace detail

// =============================================================================
// OLD IMPLEMENTATIONS — kept for reference during future refactors
// =============================================================================

// /**
//  * @brief Find the minimum element across all logical elements of an expression.
//  *
//  * Iterates slice-by-slice over the physical buffer, where "slice" means a contiguous
//  * slice along the last dimension. This naturally skips padding without needing
//  * identity-element tricks.
//  *
//  * @tparam Expr Expression type (FusedTensorND, PermutedViewConstExpr, etc.)
//  * @param expr  The expression to reduce
//  * @return T    The minimum logical element
//  *
//  * ============================================================================
//  * STRATEGY
//  * ============================================================================
//  *
//  * Physical memory is organized as numSlices × paddedLastDim, where only the
//  * first lastDim elements per slice are logical data:
//  *
//  *   slice 0: [d d d d d P P P]
//  *   slice 1: [d d d d d P P P]   d = data, P = padding
//  *   slice 2: [d d d d d P P P]
//  *          |← lastDim→|
//  *          |←  paddedLastDim →|
//  *
//  * For a 3D tensor [2, 3, 5], numSlices = 2*3 = 6:
//  *   slice 0 = [0,0,:]    slice 3 = [1,0,:]
//  *   slice 1 = [0,1,:]    slice 4 = [1,1,:]
//  *   slice 2 = [0,2,:]    slice 5 = [1,2,:]
//  *
//  * Per slice, SIMD processes simdWidth-aligned chunks, then a scalar tail
//  * handles the remainder. Padding is never read.
//  *
//  * ============================================================================
//  * EXAMPLE: FusedTensorND<double, 2, 3, 5>, AVX (SimdWidth=4)
//  * ============================================================================
//  *
//  *   lastDim=5, paddedLastDim=8, numSlices=6, simdSteps=1, scalarStart=4
//  *
//  *   Per slice (8 physical slots):
//  *     [d0 d1 d2 d3 | d4 P  P  P ]
//  *      ^^^^^^^^^^^   ^^
//  *      SIMD (i=0)    scalar tail
//  *
//  * ============================================================================
//  * EXAMPLE: FusedTensorND<double, 2, 2>, AVX (SimdWidth=4)
//  * ============================================================================
//  *
//  *   lastDim=2, paddedLastDim=4, numSlices=2, simdSteps=0, scalarStart=0
//  *
//  *   Per slice (4 physical slots):
//  *     [d0 d1 P  P ]
//  *      ^^^^^
//  *      scalar only (SIMD block skipped entirely)
//  *
//  * ============================================================================
//  * GENERICARCH (SimdWidth=1): no padding, simdSteps=lastDim, no scalar tail.
//  * Microkernel ops inline to plain scalar — same codegen as a manual loop.
//  * ============================================================================
//  */
// template <typename Expr>
// FORCE_INLINE static T reduce_min(const Expr &expr) noexcept
// {
//     using ExprLayout = typename Expr::Layout;
//     using ExprPadPolicy = typename ExprLayout::PadPolicyType;
//
//     // Slice geometry from padding policy
//     static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
//     static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
//     static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;
//     static constexpr my_size_t simdSteps = lastDim / simdWidth;
//     static constexpr my_size_t scalarStart = simdSteps * simdWidth;
//
//     T result = NumericLimits<T>::max();
//
//     // --- SIMD path: process simdWidth elements at a time per slice ---
//     if constexpr (simdSteps > 0)
//     {
//         typename K::VecType acc = K::set1(NumericLimits<T>::max());
//
//         for (my_size_t slice = 0; slice < numSlices; ++slice)
//         {
//             const my_size_t base = slice * paddedLastDim;
//             for (my_size_t i = 0; i < simdSteps; ++i)
//                 acc = K::min(acc, expr.template evalu<T, Bits, Arch>(base + i * simdWidth));
//         }
//
//         // Horizontal reduction: collapse SIMD register to scalar
//         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//         K::store(tmp, acc);
//
//         for (my_size_t i = 0; i < simdWidth; ++i)
//         {
//             if (tmp[i] < result)
//                 result = tmp[i];
//         }
//     }
//
//     // --- Scalar tail: elements at [scalarStart, lastDim) per slice ---
//     if constexpr (scalarStart < lastDim)
//     {
//         for (my_size_t slice = 0; slice < numSlices; ++slice)
//         {
//             const my_size_t base = slice * paddedLastDim;
//             for (my_size_t i = scalarStart; i < lastDim; ++i)
//             {
//                 T val = expr.template evalu<T, 1, GENERICARCH>(base + i);
//                 if (val < result)
//                     result = val;
//             }
//         }
//     }
//
//     return result;
// }

// /**
//  * @brief Find the maximum element across all logical elements of an expression.
//  * (Same strategy as reduce_min — see above for full documentation)
//  */
// template <typename Expr>
// FORCE_INLINE static T reduce_max(const Expr &expr) noexcept
// {
//     using ExprLayout = typename Expr::Layout;
//     using ExprPadPolicy = typename ExprLayout::PadPolicyType;
//
//     static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
//     static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
//     static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;
//     static constexpr my_size_t simdSteps = lastDim / simdWidth;
//     static constexpr my_size_t scalarStart = simdSteps * simdWidth;
//
//     T result = NumericLimits<T>::lowest();
//
//     if constexpr (simdSteps > 0)
//     {
//         typename K::VecType acc = K::set1(NumericLimits<T>::lowest());
//
//         for (my_size_t slice = 0; slice < numSlices; ++slice)
//         {
//             const my_size_t base = slice * paddedLastDim;
//             for (my_size_t i = 0; i < simdSteps; ++i)
//                 acc = K::max(acc, expr.template evalu<T, Bits, Arch>(base + i * simdWidth));
//         }
//
//         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//         K::store(tmp, acc);
//
//         for (my_size_t i = 0; i < simdWidth; ++i)
//         {
//             if (tmp[i] > result)
//                 result = tmp[i];
//         }
//     }
//
//     if constexpr (scalarStart < lastDim)
//     {
//         for (my_size_t slice = 0; slice < numSlices; ++slice)
//         {
//             const my_size_t base = slice * paddedLastDim;
//             for (my_size_t i = scalarStart; i < lastDim; ++i)
//             {
//                 T val = expr.template evalu<T, 1, GENERICARCH>(base + i);
//                 if (val > result)
//                     result = val;
//             }
//         }
//     }
//
//     return result;
// }

// /**
//  * @brief Sum all logical elements of an expression.
//  * (Same strategy as reduce_min — see above for full documentation)
//  *
//  * NOTE: Padding is zero, so iterating the full physical buffer would give
//  * a correct sum. But slice-by-slice is used for consistency and because
//  * linear iteration over TotalSize crosses padding boundaries mid-SIMD-load.
//  */
// template <typename Expr>
// FORCE_INLINE static T reduce_sum(const Expr &expr) noexcept
// {
//     using ExprLayout = typename Expr::Layout;
//     using ExprPadPolicy = typename ExprLayout::PadPolicyType;
//
//     static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
//     static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
//     static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;
//     static constexpr my_size_t simdSteps = lastDim / simdWidth;
//     static constexpr my_size_t scalarStart = simdSteps * simdWidth;
//
//     T result = T{0};
//
//     if constexpr (simdSteps > 0)
//     {
//         typename K::VecType acc = K::set1(T{0});
//
//         for (my_size_t slice = 0; slice < numSlices; ++slice)
//         {
//             const my_size_t base = slice * paddedLastDim;
//             for (my_size_t i = 0; i < simdSteps; ++i)
//                 acc = K::add(acc, expr.template evalu<T, Bits, Arch>(base + i * simdWidth));
//         }
//
//         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//         K::store(tmp, acc);
//
//         for (my_size_t i = 0; i < simdWidth; ++i)
//             result += tmp[i];
//     }
//
//     if constexpr (scalarStart < lastDim)
//     {
//         for (my_size_t slice = 0; slice < numSlices; ++slice)
//         {
//             const my_size_t base = slice * paddedLastDim;
//             for (my_size_t i = scalarStart; i < lastDim; ++i)
//                 result += expr.template evalu<T, 1, GENERICARCH>(base + i);
//         }
//     }
//
//     return result;
// }

#endif // KERNEL_REDUCE_H
