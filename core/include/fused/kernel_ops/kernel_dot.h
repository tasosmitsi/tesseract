/**
 * @file kernel_dot.h
 * @brief Dot product operations — contraction primitives for einsum.
 *
 * All functions take PHYSICAL offsets and strides from the layout.
 * The caller (einsum) computes these via Layout::stride() and
 * Layout::logical_coords_to_physical_flat().
 *
 * For C[i,j] = sum_k A[i,k] * B[k,j] with A[M,K] and B[K,N]:
 *
 *   A's fiber along k (last dim):  base=A.stride(0)*i, stride=1       → contiguous
 *   B's fiber along k (first dim): base=j,             stride=B.stride(0) → strided
 *
 *   Physical memory for A[2,3] padded to [2,4]:
 *     [a00 a01 a02  P | a10 a11 a12  P]
 *      ^^^^^^^^^^^      ^^^^^^^^^^^
 *      fiber i=0        fiber i=1      → contiguous, len=3
 *
 *   Physical memory for B[3,2] padded to [3,4]:
 *     [b00 b01  P  P | b10 b11  P  P | b20 b21  P  P]
 *      ^                ^                ^
 *      fiber j=0, stride=4              → strided, len=3
 */
#ifndef KERNEL_DOT_H
#define KERNEL_DOT_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/kernel_ops/kernel_helpers.h"

namespace detail
{

    template <typename T, my_size_t Bits, typename Arch>
    struct KernelDot
    {
        using K = Microkernel<T, Bits, Arch>;
        using Helpers = KernelHelpers<T, Bits, Arch>;
        static constexpr my_size_t simdWidth = K::simdWidth;

        // ========================================================================
        // Public API
        // ========================================================================

        /**
         * @brief Dispatch dot product based on stride values.
         *
         * @param base1   Physical offset of first fiber's start
         * @param stride1 Physical stride along contraction axis (1 = contiguous)
         * @param base2   Physical offset of second fiber's start
         * @param stride2 Physical stride along contraction axis (1 = contiguous)
         * @param len     Number of elements along contraction axis (logical dim)
         */
        template <typename Expr1, typename Expr2>
        FORCE_INLINE static T dot(
            const Expr1 &expr1, my_size_t base1, my_size_t stride1,
            const Expr2 &expr2, my_size_t base2, my_size_t stride2,
            my_size_t len) noexcept
        {
            if (stride1 == 1 && stride2 == 1)
            {
                // std::cout << "dot: dispatching to contiguous impl" << std::endl;
                return dot_contiguous_impl(expr1, expr2, base1, base2, len);
            }
            else
            {
                // std::cout << "dot: dispatching to strided impl" << std::endl;
                return dot_strided_impl(expr1, expr2, base1, base2, stride1, stride2, len);
            }
        }

        /**
         * @brief Naive scalar dot product for testing/validation.
         *
         * Accesses physical memory directly via data_.data().
         * Only used in tests to verify SIMD dot results.
         */
        template <typename Expr1, typename Expr2>
        FORCE_INLINE static T naive_dot_physical(
            const Expr1 &expr1, my_size_t base1, my_size_t stride1,
            const Expr2 &expr2, my_size_t base2, my_size_t stride2,
            my_size_t len) noexcept
        {
            T sum = T{0};
            for (my_size_t i = 0; i < len; ++i)
                sum += expr1.data()[base1 + i * stride1] *
                       expr2.data()[base2 + i * stride2];
            return sum;
        }

    private:
        // ========================================================================
        // Contiguous dot — both fibers have stride 1
        // ========================================================================

        /**
         * @brief Contiguous dot product — both fibers have stride 1.
         *
         * Uses K::load for aligned SIMD access. len may not be a multiple
         * of simdWidth (e.g., logical last dim = 5, simdWidth = 4), so
         * a scalar remainder handles the tail.
         *
         *   fiber1: [v0 v1 v2 v3 | v4]     fiber2: [w0 w1 w2 w3 | w4]
         *            ^^^^^^^^^^^   ^^                ^^^^^^^^^^^   ^^
         *            SIMD          scalar            SIMD          scalar
         */
        template <typename Expr1, typename Expr2>
        FORCE_INLINE static T dot_contiguous_impl(
            const Expr1 &expr1,
            const Expr2 &expr2,
            my_size_t base1,
            my_size_t base2,
            my_size_t len) noexcept
        {
            // std::cout << "dot_contiguous_impl" << std::endl;
            const T *ptr1 = expr1.data() + base1;
            const T *ptr2 = expr2.data() + base2;

            const my_size_t simdSteps = len / simdWidth;
            const my_size_t scalarStart = simdSteps * simdWidth;

            T result = T{0};

            if (simdSteps > 0)
            {
                typename K::VecType acc = K::set1(T{0});

                for (my_size_t i = 0; i < simdSteps; ++i)
                {
                    auto v1 = K::load(ptr1 + i * simdWidth);
                    auto v2 = K::load(ptr2 + i * simdWidth);
                    acc = Helpers::fmadd_safe(v1, v2, acc);
                }

                alignas(DATA_ALIGNAS) T tmp[simdWidth];
                K::store(tmp, acc);

                for (my_size_t i = 0; i < simdWidth; ++i)
                    result += tmp[i];
            }

            for (my_size_t i = scalarStart; i < len; ++i)
                result += ptr1[i] * ptr2[i];

            return result;
        }

        // ========================================================================
        // Strided dot — one or both fibers have stride > 1
        // ========================================================================

        /**
         * @brief Strided dot product — one or both fibers have stride > 1.
         *
         * Builds explicit index lists and uses K::gather for SIMD access.
         * Falls back to scalar for remainder.
         *
         *   fiber along dim 0 of B[3,2] padded to [3,4]:
         *     [b00 _ _ _ | b10 _ _ _ | b20 _ _ _]
         *      ^           ^           ^
         *      idx=0       idx=4       idx=8       stride=4, len=3
         */
        template <typename Expr1, typename Expr2>
        FORCE_INLINE static T dot_strided_impl(
            const Expr1 &expr1,
            const Expr2 &expr2,
            my_size_t idx1,
            my_size_t idx2,
            my_size_t stride1,
            my_size_t stride2,
            my_size_t len) noexcept
        {
            // std::cout << "dot_strided_impl" << std::endl;
            const my_size_t simdSteps = len / simdWidth;
            const my_size_t scalarStart = simdSteps * simdWidth;

            T result = T{0};

            if (simdSteps > 0)
            {
                typename K::VecType acc = K::set1(T{0});

                for (my_size_t i = 0; i < simdSteps; ++i)
                {
                    // Build gather indices for this chunk
                    my_size_t idxList1[simdWidth];
                    my_size_t idxList2[simdWidth];
                    for (my_size_t j = 0; j < simdWidth; ++j)
                    {
                        idxList1[j] = idx1 + j * stride1;
                        idxList2[j] = idx2 + j * stride2;
                    }

                    auto v1 = K::gather(expr1.data(), idxList1);
                    auto v2 = K::gather(expr2.data(), idxList2);
                    acc = Helpers::fmadd_safe(v1, v2, acc);

                    idx1 += simdWidth * stride1;
                    idx2 += simdWidth * stride2;
                }

                alignas(DATA_ALIGNAS) T tmp[simdWidth];
                K::store(tmp, acc);

                for (my_size_t i = 0; i < simdWidth; ++i)
                    result += tmp[i];
            }

            // Scalar tail
            for (my_size_t i = scalarStart; i < len; ++i)
            {
                result += expr1.data()[idx1] * expr2.data()[idx2];
                idx1 += stride1;
                idx2 += stride2;
            }

            return result;
        }
    };

} // namespace detail

#endif // KERNEL_DOT_H
