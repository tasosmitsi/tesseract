// // Higher-level kernel operations built on top of microkernels
// #ifndef KERNEL_OPS_H
// #define KERNEL_OPS_H

// #include "config.h"
// #include "fused/microkernels/microkernel_base.h"
// #include "numeric_limits.h"
// #include "helper_traits.h"
// #include "fused/padding_policies/simd_padding_policy.h"
// #include "expression_traits/expression_traits.h"

// template <typename T, my_size_t Bits, typename Arch>
// struct KernelOps
// {
//     using K = Microkernel<T, Bits, Arch>;
//     static constexpr my_size_t simdWidth = K::simdWidth;

//     // ========================================================================
//     // Evaluation
//     // ========================================================================

//     /**
//      * @brief Dispatch: pick contiguous or permuted eval based on expression layout.
//      *
//      * @tparam Expr
//      * @param output
//      * @param expr
//      */
//     template <typename Expr>
//     FORCE_INLINE static void eval(T *output, const Expr &expr) noexcept
//     {
//         if constexpr (!expression::traits<Expr>::IsPermuted)
//         {
//             // std::cout << "eval_contiguous" << std::endl;
//             eval_vectorized_contiguous(output, expr);
//         }
//         else
//         {
//             // std::cout << "eval_permuted" << std::endl;
//             eval_vectorized_permuted(output, expr);
//         }
//     }

//     // // ========================================================================
//     // // Reductions
//     // // ========================================================================

//     // /**
//     //  * @brief Find the minimum element across all logical elements of an expression.
//     //  *
//     //  * Iterates slice-by-slice over the physical buffer, where "slice" means a contiguous
//     //  * slice along the last dimension. This naturally skips padding without needing
//     //  * identity-element tricks.
//     //  *
//     //  * @tparam Expr Expression type (FusedTensorND, PermutedViewConstExpr, etc.)
//     //  * @param expr  The expression to reduce
//     //  * @return T    The minimum logical element
//     //  *
//     //  * ============================================================================
//     //  * STRATEGY
//     //  * ============================================================================
//     //  *
//     //  * Physical memory is organized as numSlices × paddedLastDim, where only the
//     //  * first lastDim elements per slice are logical data:
//     //  *
//     //  *   slice 0: [d d d d d P P P]
//     //  *   slice 1: [d d d d d P P P]   d = data, P = padding
//     //  *   slice 2: [d d d d d P P P]
//     //  *          |← lastDim→|
//     //  *          |←  paddedLastDim →|
//     //  *
//     //  * For a 3D tensor [2, 3, 5], numSlices = 2*3 = 6:
//     //  *   slice 0 = [0,0,:]    slice 3 = [1,0,:]
//     //  *   slice 1 = [0,1,:]    slice 4 = [1,1,:]
//     //  *   slice 2 = [0,2,:]    slice 5 = [1,2,:]
//     //  *
//     //  * Per slice, SIMD processes simdWidth-aligned chunks, then a scalar tail
//     //  * handles the remainder. Padding is never read.
//     //  *
//     //  * ============================================================================
//     //  * EXAMPLE: FusedTensorND<double, 2, 3, 5>, AVX (SimdWidth=4)
//     //  * ============================================================================
//     //  *
//     //  *   lastDim=5, paddedLastDim=8, numSlices=6, simdSteps=1, scalarStart=4
//     //  *
//     //  *   Per slice (8 physical slots):
//     //  *     [d0 d1 d2 d3 | d4 P  P  P ]
//     //  *      ^^^^^^^^^^^   ^^
//     //  *      SIMD (i=0)    scalar tail
//     //  *
//     //  * ============================================================================
//     //  * EXAMPLE: FusedTensorND<double, 2, 2>, AVX (SimdWidth=4)
//     //  * ============================================================================
//     //  *
//     //  *   lastDim=2, paddedLastDim=4, numSlices=2, simdSteps=0, scalarStart=0
//     //  *
//     //  *   Per slice (4 physical slots):
//     //  *     [d0 d1 P  P ]
//     //  *      ^^^^^
//     //  *      scalar only (SIMD block skipped entirely)
//     //  *
//     //  * ============================================================================
//     //  * GENERICARCH (SimdWidth=1): no padding, simdSteps=lastDim, no scalar tail.
//     //  * Microkernel ops inline to plain scalar — same codegen as a manual loop.
//     //  * ============================================================================
//     //  */
//     // template <typename Expr>
//     // FORCE_INLINE static T reduce_min(const Expr &expr) noexcept
//     // {
//     //     using ExprLayout = typename Expr::Layout;
//     //     using ExprPadPolicy = typename ExprLayout::PadPolicyType;

//     //     // Slice geometry from padding policy
//     //     static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;                        // logical elements per slice
//     //     static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;            // physical stride per slice
//     //     static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim; // total slices across all dims
//     //     static constexpr my_size_t simdSteps = lastDim / simdWidth;                         // full SIMD chunks per slice
//     //     static constexpr my_size_t scalarStart = simdSteps * simdWidth;                     // where scalar tail begins

//     //     T result = NumericLimits<T>::max();

//     //     // --- SIMD path: process simdWidth elements at a time per slice ---
//     //     if constexpr (simdSteps > 0)
//     //     {
//     //         typename K::VecType acc = K::set1(NumericLimits<T>::max());

//     //         for (my_size_t slice = 0; slice < numSlices; ++slice)
//     //         {
//     //             const my_size_t base = slice * paddedLastDim; // physical offset of slice start
//     //             for (my_size_t i = 0; i < simdSteps; ++i)
//     //                 acc = K::min(acc, expr.template evalu<T, Bits, Arch>(base + i * simdWidth));
//     //         }

//     //         // Horizontal reduction: collapse SIMD register to scalar
//     //         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//     //         K::store(tmp, acc);

//     //         for (my_size_t i = 0; i < simdWidth; ++i)
//     //         {
//     //             if (tmp[i] < result)
//     //                 result = tmp[i];
//     //         }
//     //     }

//     //     // --- Scalar tail: elements at [scalarStart, lastDim) per slice ---
//     //     if constexpr (scalarStart < lastDim)
//     //     {
//     //         for (my_size_t slice = 0; slice < numSlices; ++slice)
//     //         {
//     //             const my_size_t base = slice * paddedLastDim;
//     //             for (my_size_t i = scalarStart; i < lastDim; ++i)
//     //             {
//     //                 T val = expr.template evalu<T, 1, GENERICARCH>(base + i);
//     //                 if (val < result)
//     //                     result = val;
//     //             }
//     //         }
//     //     }

//     //     return result;
//     // }

//     // /**
//     //  * @brief Find the maximum element across all logical elements of an expression.
//     //  *
//     //  * Iterates slice-by-slice over the physical buffer, where "slice" means a contiguous
//     //  * slice along the last dimension. This naturally skips padding without needing
//     //  * identity-element tricks.
//     //  *
//     //  * @tparam Expr Expression type (FusedTensorND, PermutedViewConstExpr, etc.)
//     //  * @param expr  The expression to reduce
//     //  * @return T    The maximum logical element
//     //  *
//     //  * ============================================================================
//     //  * STRATEGY
//     //  * ============================================================================
//     //  *
//     //  * Physical memory is organized as numSlices × paddedLastDim, where only the
//     //  * first lastDim elements per slice are logical data:
//     //  *
//     //  *   slice 0: [d d d d d P P P]
//     //  *   slice 1: [d d d d d P P P]   d = data, P = padding
//     //  *   slice 2: [d d d d d P P P]
//     //  *          |← lastDim→|
//     //  *          |←  paddedLastDim →|
//     //  *
//     //  * For a 3D tensor [2, 3, 5], numSlices = 2*3 = 6:
//     //  *   slice 0 = [0,0,:]    slice 3 = [1,0,:]
//     //  *   slice 1 = [0,1,:]    slice 4 = [1,1,:]
//     //  *   slice 2 = [0,2,:]    slice 5 = [1,2,:]
//     //  *
//     //  * Per slice, SIMD processes simdWidth-aligned chunks, then a scalar tail
//     //  * handles the remainder. Padding is never read.
//     //  *
//     //  * ============================================================================
//     //  * GENERICARCH (SimdWidth=1): no padding, simdSteps=lastDim, no scalar tail.
//     //  * Microkernel ops inline to plain scalar — same codegen as a manual loop.
//     //  * ============================================================================
//     //  */
//     // template <typename Expr>
//     // FORCE_INLINE static T reduce_max(const Expr &expr) noexcept
//     // {
//     //     using ExprLayout = typename Expr::Layout;
//     //     using ExprPadPolicy = typename ExprLayout::PadPolicyType;

//     //     // Slice geometry from padding policy
//     //     static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;                        // logical elements per slice
//     //     static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;            // physical stride per slice
//     //     static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim; // total slices across all dims
//     //     static constexpr my_size_t simdSteps = lastDim / simdWidth;                         // full SIMD chunks per slice
//     //     static constexpr my_size_t scalarStart = simdSteps * simdWidth;                     // where scalar tail begins

//     //     T result = NumericLimits<T>::lowest();

//     //     // --- SIMD path: process simdWidth elements at a time per slice ---
//     //     if constexpr (simdSteps > 0)
//     //     {
//     //         typename K::VecType acc = K::set1(NumericLimits<T>::lowest());

//     //         for (my_size_t slice = 0; slice < numSlices; ++slice)
//     //         {
//     //             const my_size_t base = slice * paddedLastDim; // physical offset of slice start
//     //             for (my_size_t i = 0; i < simdSteps; ++i)
//     //                 acc = K::max(acc, expr.template evalu<T, Bits, Arch>(base + i * simdWidth));
//     //         }

//     //         // Horizontal reduction: collapse SIMD register to scalar
//     //         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//     //         K::store(tmp, acc);

//     //         for (my_size_t i = 0; i < simdWidth; ++i)
//     //         {
//     //             if (tmp[i] > result)
//     //                 result = tmp[i];
//     //         }
//     //     }

//     //     // --- Scalar tail: elements at [scalarStart, lastDim) per slice ---
//     //     if constexpr (scalarStart < lastDim)
//     //     {
//     //         for (my_size_t slice = 0; slice < numSlices; ++slice)
//     //         {
//     //             const my_size_t base = slice * paddedLastDim;
//     //             for (my_size_t i = scalarStart; i < lastDim; ++i)
//     //             {
//     //                 T val = expr.template evalu<T, 1, GENERICARCH>(base + i);
//     //                 if (val > result)
//     //                     result = val;
//     //             }
//     //         }
//     //     }

//     //     return result;
//     // }

//     // /**
//     //  * @brief Sum all logical elements of an expression.
//     //  *
//     //  * Iterates slice-by-slice over the physical buffer, skipping padding.
//     //  * Same strategy as reduce_min/reduce_max.
//     //  *
//     //  * ============================================================================
//     //  * NOTE: Padding is zero, so iterating the full physical buffer would give
//     //  * a correct sum. But slice-by-slice is used for consistency and because
//     //  * linear iteration over TotalSize crosses padding boundaries mid-SIMD-load.
//     //  * ============================================================================
//     //  *
//     //  * @tparam Expr Expression type (FusedTensorND, PermutedViewConstExpr, etc.)
//     //  * @param expr  The expression to reduce
//     //  * @return T    The sum of all logical elements
//     //  */
//     // template <typename Expr>
//     // FORCE_INLINE static T reduce_sum(const Expr &expr) noexcept
//     // {
//     //     using ExprLayout = typename Expr::Layout;
//     //     using ExprPadPolicy = typename ExprLayout::PadPolicyType;

//     //     static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
//     //     static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
//     //     static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;
//     //     static constexpr my_size_t simdSteps = lastDim / simdWidth;
//     //     static constexpr my_size_t scalarStart = simdSteps * simdWidth;

//     //     T result = T{0};

//     //     if constexpr (simdSteps > 0)
//     //     {
//     //         typename K::VecType acc = K::set1(T{0});

//     //         for (my_size_t slice = 0; slice < numSlices; ++slice)
//     //         {
//     //             const my_size_t base = slice * paddedLastDim;
//     //             for (my_size_t i = 0; i < simdSteps; ++i)
//     //                 acc = K::add(acc, expr.template evalu<T, Bits, Arch>(base + i * simdWidth));
//     //         }

//     //         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//     //         K::store(tmp, acc);

//     //         for (my_size_t i = 0; i < simdWidth; ++i)
//     //             result += tmp[i];
//     //     }

//     //     if constexpr (scalarStart < lastDim)
//     //     {
//     //         for (my_size_t slice = 0; slice < numSlices; ++slice)
//     //         {
//     //             const my_size_t base = slice * paddedLastDim;
//     //             for (my_size_t i = scalarStart; i < lastDim; ++i)
//     //                 result += expr.template evalu<T, 1, GENERICARCH>(base + i);
//     //         }
//     //     }

//     //     return result;
//     // }

//     template <typename Expr>
//     FORCE_INLINE static T reduce_min(const Expr &expr) noexcept
//     {
//         return reduce<ReduceOp::Min>(expr);
//     }

//     template <typename Expr>
//     FORCE_INLINE static T reduce_max(const Expr &expr) noexcept
//     {
//         return reduce<ReduceOp::Max>(expr);
//     }

//     template <typename Expr>
//     FORCE_INLINE static T reduce_sum(const Expr &expr) noexcept
//     {
//         return reduce<ReduceOp::Sum>(expr);
//     }

//     /**
//      * @brief Check if all logical elements of two expressions are approximately equal.
//      *
//      * Dispatches based on layout compatibility:
//      *   - Both unpermuted, same padding → fast physical slice iteration
//      *   - Otherwise → logical flat iteration (handles any layout combination)
//      *
//      * Both expressions must have the same logical dimensions.
//      * TODO: do we need to check this at compile time, or is it guaranteed by the caller? think...!
//      *
//      * @tparam Expr1 First expression type
//      * @tparam Expr2 Second expression type
//      * @param lhs    Left-hand side expression
//      * @param rhs    Right-hand side expression
//      * @param tolerance Approximation tolerance for floating-point comparisons
//      * @return true if all corresponding logical elements are approximately equal within the given tolerance, false otherwise
//      */
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static bool reduce_all_approx_equal(
//         const Expr1 &lhs,
//         const Expr2 &rhs,
//         T tolerance) noexcept
//     {
//         if constexpr (expression::traits<Expr1>::IsContiguous &&
//                       expression::traits<Expr2>::IsContiguous)
//             return approx_equal_contiguous(lhs, rhs, tolerance);
//         else
//             return approx_equal_logical(lhs, rhs, tolerance);
//     }

//     // ========================================================================
//     // DOT PRODUCTS (used by einsum for contraction along a shared axis)
//     // ========================================================================
//     //
//     // All functions take PHYSICAL offsets and strides from the layout.
//     // The caller (einsum) computes these via Layout::stride() and
//     // Layout::logical_coords_to_physical_flat().
//     //
//     // For C[i,j] = sum_k A[i,k] * B[k,j] with A[M,K] and B[K,N]:
//     //
//     //   A's fiber along k (last dim):  base=A.stride(0)*i, stride=1       → contiguous
//     //   B's fiber along k (first dim): base=j,             stride=B.stride(0) → strided
//     //
//     //   Physical memory for A[2,3] padded to [2,4]:
//     //     [a00 a01 a02  P | a10 a11 a12  P]
//     //      ^^^^^^^^^^^      ^^^^^^^^^^^
//     //      fiber i=0        fiber i=1      → contiguous, len=3
//     //
//     //   Physical memory for B[3,2] padded to [3,4]:
//     //     [b00 b01  P  P | b10 b11  P  P | b20 b21  P  P]
//     //      ^                ^                ^
//     //      fiber j=0, stride=4              → strided, len=3

//     /**
//      * @brief Dispatch dot product based on stride values.
//      *
//      * @param base1   Physical offset of first fiber's start
//      * @param stride1 Physical stride along contraction axis (1 = contiguous)
//      * @param base2   Physical offset of second fiber's start
//      * @param stride2 Physical stride along contraction axis (1 = contiguous)
//      * @param len     Number of elements along contraction axis (logical dim)
//      */
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static T dot(
//         const Expr1 &expr1, my_size_t base1, my_size_t stride1,
//         const Expr2 &expr2, my_size_t base2, my_size_t stride2,
//         my_size_t len) noexcept
//     {
//         if (stride1 == 1 && stride2 == 1)
//             return dot_contiguous_impl(expr1, expr2, base1, base2, len);
//         else
//             return dot_strided_impl(expr1, expr2, base1, base2, stride1, stride2, len);
//     }

//     /**
//      * @brief Contiguous dot product — both fibers have stride 1.
//      *
//      * Uses K::load for aligned SIMD access. len may not be a multiple
//      * of simdWidth (e.g., logical last dim = 5, simdWidth = 4), so
//      * a scalar remainder handles the tail.
//      *
//      *   fiber1: [v0 v1 v2 v3 | v4]     fiber2: [w0 w1 w2 w3 | w4]
//      *            ^^^^^^^^^^^   ^^                ^^^^^^^^^^^   ^^
//      *            SIMD          scalar            SIMD          scalar
//      */
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static T dot_contiguous_impl(
//         const Expr1 &expr1,
//         const Expr2 &expr2,
//         my_size_t base1,
//         my_size_t base2,
//         my_size_t len) noexcept
//     {
//         // std::cout << "dot_contiguous_impl" << std::endl;
//         const my_size_t simdSteps = len / simdWidth;
//         const my_size_t scalarStart = simdSteps * simdWidth;

//         T result = T{0};

//         if (simdSteps > 0)
//         {
//             typename K::VecType acc = K::set1(T{0});

//             for (my_size_t i = 0; i < simdSteps; ++i)
//             {
//                 auto v1 = expr1.template evalu<T, Bits, Arch>(base1 + i * simdWidth);
//                 auto v2 = expr2.template evalu<T, Bits, Arch>(base2 + i * simdWidth);
//                 acc = fmadd_safe(v1, v2, acc);
//             }

//             alignas(DATA_ALIGNAS) T tmp[simdWidth];
//             K::store(tmp, acc);

//             for (my_size_t i = 0; i < simdWidth; ++i)
//                 result += tmp[i];
//         }

//         for (my_size_t i = scalarStart; i < len; ++i)
//         {
//             T v1 = expr1.template evalu<T, 1, GENERICARCH>(base1 + i);
//             T v2 = expr2.template evalu<T, 1, GENERICARCH>(base2 + i);
//             result += v1 * v2;
//         }

//         return result;
//     }

//     /**
//      * @brief Strided dot product — one or both fibers have stride > 1.
//      *
//      * Builds explicit index lists and uses K::gather for SIMD access.
//      * Falls back to scalar for remainder.
//      *
//      *   fiber along dim 0 of B[3,2] padded to [3,4]:
//      *     [b00 _ _ _ | b10 _ _ _ | b20 _ _ _]
//      *      ^           ^           ^
//      *      idx=0       idx=4       idx=8       stride=4, len=3
//      */
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static T dot_strided_impl(
//         const Expr1 &expr1,
//         const Expr2 &expr2,
//         my_size_t idx1,
//         my_size_t idx2,
//         my_size_t stride1,
//         my_size_t stride2,
//         my_size_t len) noexcept
//     {
//         // std::cout << "dot_strided_impl" << std::endl;
//         const my_size_t simdSteps = len / simdWidth;
//         const my_size_t scalarStart = simdSteps * simdWidth;

//         T result = T{0};

//         if (simdSteps > 0)
//         {
//             typename K::VecType acc = K::set1(T{0});

//             for (my_size_t i = 0; i < simdSteps; ++i)
//             {
//                 // Build gather indices for this chunk
//                 my_size_t idxList1[simdWidth];
//                 my_size_t idxList2[simdWidth];
//                 for (my_size_t j = 0; j < simdWidth; ++j)
//                 {
//                     idxList1[j] = idx1 + j * stride1;
//                     idxList2[j] = idx2 + j * stride2;
//                 }

//                 auto v1 = K::gather(expr1.data(), idxList1);
//                 auto v2 = K::gather(expr2.data(), idxList2);
//                 acc = fmadd_safe(v1, v2, acc);

//                 idx1 += simdWidth * stride1;
//                 idx2 += simdWidth * stride2;
//             }

//             alignas(DATA_ALIGNAS) T tmp[simdWidth];
//             K::store(tmp, acc);

//             for (my_size_t i = 0; i < simdWidth; ++i)
//                 result += tmp[i];
//         }

//         // Scalar tail
//         for (my_size_t i = scalarStart; i < len; ++i)
//         {
//             T v1 = expr1.template evalu<T, 1, GENERICARCH>(idx1);
//             T v2 = expr2.template evalu<T, 1, GENERICARCH>(idx2);
//             result += v1 * v2;
//             idx1 += stride1;
//             idx2 += stride2;
//         }

//         return result;
//     }

//     /**
//      * @brief Naive scalar dot product for testing/validation.
//      *
//      * Accesses physical memory directly via data_.data().
//      * Only used in tests to verify SIMD dot results.
//      */
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static T naive_dot_physical(
//         const Expr1 &expr1, my_size_t base1, my_size_t stride1,
//         const Expr2 &expr2, my_size_t base2, my_size_t stride2,
//         my_size_t len) noexcept
//     {
//         T sum = T{0};
//         for (my_size_t i = 0; i < len; ++i)
//             sum += expr1.data_.data()[base1 + i * stride1] *
//                    expr2.data_.data()[base2 + i * stride2];
//         return sum;
//     }

// private:
//     enum class ReduceOp
//     {
//         Min,
//         Max,
//         Sum
//     };

//     template <ReduceOp Op>
//     FORCE_INLINE static T reduce_identity() noexcept
//     {
//         if constexpr (Op == ReduceOp::Min)
//             return NumericLimits<T>::max();
//         if constexpr (Op == ReduceOp::Max)
//             return NumericLimits<T>::lowest();
//         if constexpr (Op == ReduceOp::Sum)
//             return T{0};
//     }

//     template <ReduceOp Op>
//     FORCE_INLINE static typename K::VecType reduce_simd_combine(
//         typename K::VecType a, typename K::VecType b) noexcept
//     {
//         if constexpr (Op == ReduceOp::Min)
//             return K::min(a, b);
//         if constexpr (Op == ReduceOp::Max)
//             return K::max(a, b);
//         if constexpr (Op == ReduceOp::Sum)
//             return K::add(a, b);
//     }

//     template <ReduceOp Op>
//     FORCE_INLINE static T reduce_scalar_combine(T a, T b) noexcept
//     {
//         using ScalarK = Microkernel<T, 1, GENERICARCH>;
//         if constexpr (Op == ReduceOp::Min)
//             return ScalarK::min(a, b);
//         if constexpr (Op == ReduceOp::Max)
//             return ScalarK::max(a, b);
//         if constexpr (Op == ReduceOp::Sum)
//             return ScalarK::add(a, b);
//     }
//     /**
//      * @brief CONTIGUOUS PATH (identity layout) or fast path — no permutation, no remapping.
//      *
//      * Iterates entire physical buffer linearly. PhysicalSize is guaranteed
//      * to be a multiple of simdWidth by the padding policy. Padding
//      * slots contain zeros — harmless for element-wise ops.
//      *
//      * evalu receives physical flat offsets.
//      *
//      * Note: FusedTensorND::evalu works for both paths because with identity layout,
//      * physical flat and logical flat are the same within each slice
//      * (the contiguous path passes physical offsets,
//      * the permuted path passes logical flats,
//      * same numbers when unpermuted).
//      *
//      * @tparam Expr Expression type (FusedTensorND, PermutedViewConstExpr, etc.)
//      * @param output Pointer to output buffer (already allocated, size = PhysicalSize)
//      * @param expr The expression to evaluate and store in output
//      */
//     template <typename Expr>
//     FORCE_INLINE static void eval_vectorized_contiguous(
//         T *output,
//         const Expr &expr) noexcept
//     {
//         using Layout = typename Expr::Layout;
//         static constexpr my_size_t physicalSize = Layout::PhysicalSize;
//         static constexpr my_size_t simdSteps = physicalSize / simdWidth;
//         static constexpr bool hasRemainder = (physicalSize % simdWidth) != 0;

//         // Paranoia check: ensure physical size is a multiple of SIMD width,
//         // so we never read out of bounds
//         static_assert(physicalSize % simdWidth == 0,
//                       "PhysicalSize must be a multiple of SimdWidth");

//         // SIMD loop
//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             auto val = expr.template evalu<T, Bits, Arch>(i * simdWidth);
//             K::store(output + i * simdWidth, val);
//         }

//         // Scalar remainder TODO: The whole point of padding is that PhysicalSize is already
//         // a multiple of SimdWidth — so there's no scalar remainder
//         // Delete this code if confirmed unnecessary
//         if constexpr (hasRemainder)
//         {
//             std::cout << "Warning: Scalar evaluation for remainder elements." << std::endl;
//             // for (my_size_t i = simdSteps * simdWidth; i < physicalSize; ++i)
//             // {
//             //     output[i] = expr.template evalu<T, 1, GENERICARCH>(i);
//             // }
//         }
//     }

//     /**
//      * @brief PERMUTED PATH (any layout) or general path — works with any layout, including permuted.
//      * TODO: untested
//      * Iterates output physical slices for aligned stores. Tracks a running
//      * logical_flat for the expression. Permuted evalu uses K::gather,
//      * unpermuted uses K::load — dispatch happens inside evalu, not here.
//      *
//      * ============================================================================
//      * EXAMPLE: output [3,2] padded to [3,4], source [2,3] transposed
//      * ============================================================================
//      *
//      *   lastDim=2, paddedLastDim=4, numSlices=3, simdSteps=0, scalarStart=0
//      *
//      *   slice 0: out_base=0,  logical_flat 0,1 → store at output[0], output[1]
//      *   slice 1: out_base=4,  logical_flat 2,3 → store at output[4], output[5]
//      *   slice 2: out_base=8,  logical_flat 4,5 → store at output[8], output[9]
//      *
//      *   Padding at output[2,3,6,7,10,11] untouched.
//      * ============================================================================
//      */

//     template <typename Expr, typename Seq>
//     struct OutputPadImpl
//     {
//     };

//     template <typename Expr, my_size_t... Is>
//     struct OutputPadImpl<Expr, index_seq<Is...>>
//     {
//         using type = SimdPaddingPolicy<typename Expr::value_type, Expr::Dim[Is]...>;
//     };

//     template <typename Expr>
//     struct OutputPadPolicy
//     {
//         using type = typename OutputPadImpl<Expr, typename make_index_seq<Expr::NumDims>::type>::type;
//     };

//     template <typename Expr>
//     FORCE_INLINE static void eval_vectorized_permuted(
//         T *output,
//         const Expr &expr) noexcept
//     {
//         // using ExprPadPolicy = typename Expr::Layout::PadPolicyType;

//         // using OutputPad = SimdPaddingPolicy<T, Expr::Layout::PadPolicyType::LogicalDims...>;
//         // using OutputPad = OutputPadPolicy<typename Expr::Layout>;

//         using OutputPad = typename OutputPadPolicy<Expr>::type;

//         // static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
//         // static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
//         // static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;

//         static constexpr my_size_t lastDim = OutputPad::LastDim;
//         static constexpr my_size_t paddedLastDim = OutputPad::PaddedLastDim;
//         static constexpr my_size_t numSlices = OutputPad::PhysicalSize / paddedLastDim;

//         static constexpr my_size_t simdSteps = lastDim / simdWidth;
//         static constexpr my_size_t scalarStart = simdSteps * simdWidth;

//         my_size_t logical_flat = 0;

//         for (my_size_t slice = 0; slice < numSlices; ++slice)
//         {
//             const my_size_t out_base = slice * paddedLastDim;

//             for (my_size_t i = 0; i < simdSteps; ++i)
//             {
//                 auto val = expr.template evalu<T, Bits, Arch>(logical_flat);
//                 K::store(output + out_base + i * simdWidth, val);
//                 logical_flat += simdWidth;
//             }

//             if constexpr (scalarStart < lastDim)
//             {
//                 for (my_size_t i = scalarStart; i < lastDim; ++i)
//                 {
//                     output[out_base + i] = expr.template evalu<T, 1, GENERICARCH>(logical_flat);
//                     ++logical_flat;
//                 }
//             }
//         }
//     }

//     /**
//      * @brief Fast path — both expressions share the same physical layout.
//      *
//      * Iterates physical slices. evalu receives physical offsets.
//      * Same slice geometry for both sides.
//      *
//      * @tparam Expr1 First expression type
//      * @tparam Expr2 Second expression type
//      * @param lhs    Left-hand side expression
//      * @param rhs    Right-hand side expression
//      * @param tolerance Approximation tolerance for floating-point comparisons
//      * @return true if all corresponding logical elements are approximately equal within the given tolerance, false otherwise
//      */
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static bool approx_equal_contiguous(
//         const Expr1 &lhs,
//         const Expr2 &rhs,
//         T tolerance) noexcept
//     {
//         using ExprPadPolicy = typename Expr1::Layout::PadPolicyType;

//         static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
//         static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
//         static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;
//         static constexpr my_size_t simdSteps = lastDim / simdWidth;
//         static constexpr my_size_t scalarStart = simdSteps * simdWidth;

//         if constexpr (simdSteps > 0)
//         {
//             for (my_size_t slice = 0; slice < numSlices; ++slice)
//             {
//                 const my_size_t base = slice * paddedLastDim;
//                 for (my_size_t i = 0; i < simdSteps; ++i)
//                 {
//                     auto lhs_vec = lhs.template evalu<T, Bits, Arch>(base + i * simdWidth);
//                     auto rhs_vec = rhs.template evalu<T, Bits, Arch>(base + i * simdWidth);
//                     if (!K::all_within_tolerance(lhs_vec, rhs_vec, tolerance))
//                         return false;
//                 }
//             }
//         }

//         if constexpr (scalarStart < lastDim)
//         {
//             using ScalarK = Microkernel<T, 1, GENERICARCH>;
//             for (my_size_t slice = 0; slice < numSlices; ++slice)
//             {
//                 const my_size_t base = slice * paddedLastDim;
//                 for (my_size_t i = scalarStart; i < lastDim; ++i)
//                 {
//                     T lhs_val = lhs.template evalu<T, 1, GENERICARCH>(base + i);
//                     T rhs_val = rhs.template evalu<T, 1, GENERICARCH>(base + i);
//                     if (ScalarK::abs(lhs_val - rhs_val) > tolerance)
//                         return false;
//                 }
//             }
//         }

//         return true;
//     }

//     /**
//      * @brief General path — one or both expressions are permuted.
//      *
//      * Iterates logical flat indices. Each expression's evalu handles
//      * its own remapping (gather for permuted, load for unpermuted).
//      *
//      * Uses the OUTPUT-side slice geometry to track logical_flat while
//      * skipping output padding gaps — same pattern as eval_vectorized_permuted.
//      * But since we're comparing (not storing), we use Expr1's padding policy
//      * as reference.
//      *
//      * TODO: Untested
//      *
//      * @tparam Expr1 First expression type
//      * @tparam Expr2 Second expression type
//      * @param lhs    Left-hand side expression
//      * @param rhs    Right-hand side expression
//      * @param tolerance Approximation tolerance for floating-point comparisons
//      * @return true if all corresponding logical elements are approximately equal within the given tolerance, false otherwise
//      */
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static bool approx_equal_logical(
//         const Expr1 &lhs,
//         const Expr2 &rhs,
//         T tolerance) noexcept
//     {
//         // Use Expr1's logical dims to drive iteration
//         // (both must have same logical dims)
//         // TODO: enforce this at compile time? Or is it guaranteed anyways? think...!
//         static constexpr my_size_t logicalSize = Expr1::TotalSize;

//         using ScalarK = Microkernel<T, 1, GENERICARCH>;

//         // Pure scalar — each evalu handles its own layout remapping
//         for (my_size_t i = 0; i < logicalSize; ++i)
//         {
//             T lhs_val = lhs.template evalu<T, 1, GENERICARCH>(i);
//             T rhs_val = rhs.template evalu<T, 1, GENERICARCH>(i);
//             if (ScalarK::abs(lhs_val - rhs_val) > tolerance)
//                 return false;
//         }

//         return true;
//     }

//     // ========================================================================
//     // REDUCTION DISPATCH
//     // ========================================================================

//     template <ReduceOp Op, typename Expr>
//     FORCE_INLINE static T reduce(const Expr &expr) noexcept
//     {
//         if constexpr (!expression::traits<Expr>::IsPermuted)
//         {
//             std::cout << "reduce_contiguous" << std::endl;
//             return reduce_contiguous<Op>(expr);
//         }
//         else
//         {
//             std::cout << "reduce_logical" << std::endl;
//             return reduce_logical<Op>(expr);
//         }
//     }

//     // ========================================================================
//     // CONTIGUOUS PATH — iterate physical slices, skip padding
//     // ========================================================================

//     template <ReduceOp Op, typename Expr>
//     FORCE_INLINE static T reduce_contiguous(const Expr &expr) noexcept
//     {
//         using ExprPadPolicy = typename Expr::Layout::PadPolicyType;

//         static constexpr my_size_t lastDim = ExprPadPolicy::LastDim;
//         static constexpr my_size_t paddedLastDim = ExprPadPolicy::PaddedLastDim;
//         static constexpr my_size_t numSlices = ExprPadPolicy::PhysicalSize / paddedLastDim;
//         static constexpr my_size_t simdSteps = lastDim / simdWidth;
//         static constexpr my_size_t scalarStart = simdSteps * simdWidth;

//         T result = reduce_identity<Op>();

//         if constexpr (simdSteps > 0)
//         {
//             typename K::VecType acc = K::set1(reduce_identity<Op>());

//             for (my_size_t slice = 0; slice < numSlices; ++slice)
//             {
//                 const my_size_t base = slice * paddedLastDim;
//                 for (my_size_t i = 0; i < simdSteps; ++i)
//                     acc = reduce_simd_combine<Op>(
//                         acc, expr.template evalu<T, Bits, Arch>(base + i * simdWidth));
//             }

//             alignas(DATA_ALIGNAS) T tmp[simdWidth];
//             K::store(tmp, acc);

//             for (my_size_t i = 0; i < simdWidth; ++i)
//                 result = reduce_scalar_combine<Op>(result, tmp[i]);
//         }

//         if constexpr (scalarStart < lastDim)
//         {
//             for (my_size_t slice = 0; slice < numSlices; ++slice)
//             {
//                 const my_size_t base = slice * paddedLastDim;
//                 for (my_size_t i = scalarStart; i < lastDim; ++i)
//                     result = reduce_scalar_combine<Op>(
//                         result, expr.template evalu<T, 1, GENERICARCH>(base + i));
//             }
//         }

//         return result;
//     }

//     // ========================================================================
//     // LOGICAL PATH — iterate logical flat indices, scalar only
//     // ========================================================================
//     //
//     // For permuted views. Consecutive logical flats are non-contiguous
//     // in physical memory, so SIMD load would read wrong data.
//     // Scalar evalu handles the remapping internally.

//     template <ReduceOp Op, typename Expr>
//     FORCE_INLINE static T reduce_logical(const Expr &expr) noexcept
//     {
//         static constexpr my_size_t totalSize = Expr::TotalSize;

//         T result = reduce_identity<Op>();

//         for (my_size_t i = 0; i < totalSize; ++i)
//             result = reduce_scalar_combine<Op>(
//                 result, expr.template evalu<T, 1, GENERICARCH>(i));

//         return result;
//     }

//     // ========================================================================
//     // Helper
//     // ========================================================================

//     FORCE_INLINE static typename K::VecType fmadd_safe(
//         typename K::VecType a,
//         typename K::VecType b,
//         typename K::VecType c) noexcept
//     {
//         if constexpr (requires { K::fmadd(a, b, c); })
//         {
//             return K::fmadd(a, b, c);
//         }
//         else
//         {
//             return K::add(K::mul(a, b), c);
//         }
//     }
// };

// #endif // KERNEL_OPS_H