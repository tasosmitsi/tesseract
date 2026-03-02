// // Higher-level kernel operations built on top of microkernels
// #ifndef KERNEL_OPS_H
// #define KERNEL_OPS_H

// #include "config.h"
// #include "fused/microkernels/microkernel_base.h"
// #include "numeric_limits.h"

// // template <typename T, my_size_t Bits, typename Arch, my_size_t... Dims>
// // struct TensorKernels
// // {
// //     using K = Microkernel<T, Bits, Arch>;
// //     static constexpr my_size_t simdWidth = K::simdWidth;
// //     static constexpr my_size_t dimCount = sizeof...(Dims);
// //     static constexpr my_size_t totalSize = (Dims * ...); // product of all dims

// //     // ========================================================================
// //     // ELEMENT-WISE OPERATIONS
// //     // ========================================================================

// //     // Vectorized evaluation with contiguous storage assumption (aka the result tensor is not transposed)
// //     template <typename Expr>
// //     FORCE_INLINE static void eval_vectorized_contiguous(
// //         T *output,
// //         const Expr &expr,
// //         auto &&unravelIndexfn) noexcept
// //     {

// //         const my_size_t simdSteps = totalSize / simdWidth;

// //         // SIMD loop
// //         for (my_size_t i = 0; i < simdSteps; ++i)
// //         {
// //             auto val = expr.template evalu<T, Bits, Arch>(i * simdWidth);
// //             // Contiguous case — fast path
// //             K::store(output + i * simdWidth, val);
// //         }

// //         if constexpr ((totalSize % simdWidth) != 0) // TODO: verify if constexpr works here
// //         // in theory the compiler should be able to optimize out this branch if totalSize is known at compile time to be multiple of simdWidth
// //         // but better be safe than sorry, whith constexpr we ensure no runtime overhead if not needed
// //         {
// //             // Scalar remainder
// //             my_size_t indices[dimCount];
// //             for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
// //             {
// //                 std::forward<decltype(unravelIndexfn)>(unravelIndexfn)(i, indices); // TODO: get rid of std
// //                 // evaluate the remainder
// //                 output[i] = expr(indices);
// //             }
// //         }
// //     }

// //     // Vectorized evaluation with non-contiguous storage (aka the result tensor is transposed)
// //     // In other words this algorithm uses scatter which means the result is saved in non-continuous memmory
// //     // TODO: not tested yet
// //     // TODO: not sure if this is needed at all
// //     // template <typename Expr>
// //     // FORCE_INLINE static void eval_vectorized_non_contiguous(
// //     //     T *output,
// //     //     const Expr &expr,
// //     //     my_size_t (&transposeOrder)[dimCount]) noexcept
// //     // {

// //     //     const my_size_t simdSteps = totalSize / simdWidth;

// //     //     // SIMD loop
// //     //     for (my_size_t i = 0; i < simdSteps; ++i)
// //     //     {
// //     //         my_size_t baseIdx = i * simdWidth;

// //     //         auto val = expr.evalu(i * simdWidth);

// //     //         // Non-contiguous (result tensor is transposed) case
// //     //         my_size_t idxList[simdWidth];
// //     //         for (int j = 0; j < simdWidth; ++j)
// //     //             idxList[j] = remapFlatIndex(baseIdx + j, transposeOrder);
// //     //         K::scatter(output, idxList, val);
// //     //     }

// //     //     // Scalar remainder TODO: this is wrong? — need to remap indices here as well?
// //     //     for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
// //     //     {
// //     //         // std::cout << "Scalar remainder loop" << std::endl;
// //     //         my_size_t indices[dimCount];
// //     //         unravelIndex(i, indices, transposeOrder);
// //     //         output[i] = expr(indices);
// //     //     }
// //     // }

// //     template <typename Expr>
// //     FORCE_INLINE static void eval_scalar(
// //         T *output,
// //         const Expr &expr,
// //         auto &&unravelIndexfn) noexcept
// //     {
// //         // Pure scalar fallback
// //         for (my_size_t i = 0; i < totalSize; ++i)
// //         {
// //             my_size_t indices[dimCount];
// //             std::forward<decltype(unravelIndexfn)>(unravelIndexfn)(i, indices); // TODO: get rid of std
// //             output[i] = expr(indices);
// //         }
// //     }
// // };

// template <typename Expr, my_size_t Bits, typename Arch>
// struct KernelOps
// {
//     using T = typename Expr::value_type;
//     using K = Microkernel<T, Bits, Arch>;

//     static constexpr my_size_t simdWidth = K::simdWidth;
//     static constexpr my_size_t numDims = Expr::NumDims;
//     static constexpr my_size_t totalSize = Expr::TotalSize;
//     static constexpr my_size_t simdSteps = totalSize / simdWidth;
//     static constexpr bool hasRemainder = (totalSize % simdWidth) != 0;

//     FORCE_INLINE static void eval_vectorized_contiguous(
//         T *output,
//         const Expr &expr) noexcept
//     {
//         // SIMD loop
//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             // for GENERICARCH, Bits does not matter, simdWidth=1,
//             // so this works for both scalar and vectorized cases
//             auto val = expr.template evalu<T, Bits, Arch>(i * simdWidth);
//             K::store(output + i * simdWidth, val);
//         }

//         // Scalar remainder
//         if constexpr (hasRemainder)
//         {
//             for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
//             {
//                 // fallback to scalar evaluation using GENERICARCH microkernel
//                 // the 1 here is skipped since in scalar mode simdWidth=1
//                 output[i] = expr.template evalu<T, 1, GENERICARCH>(i);
//             }
//         }
//     }

//     FORCE_INLINE static T reduce_min(
//         const Expr &expr) noexcept
//     {
//         typename K::VecType acc = K::set1(NumericLimits<T>::max());

//         // SIMD loop
//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             acc = K::min(acc, expr.template evalu<T, Bits, Arch>(i * simdWidth));
//         }

//         // Horizontal reduction
//         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//         K::store(tmp, acc);

//         T result = tmp[0];
//         for (my_size_t i = 1; i < simdWidth; ++i)
//         {
//             if (tmp[i] < result)
//                 result = tmp[i];
//         }

//         if constexpr (hasRemainder)
//         {
//             for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
//             {
//                 T val = expr.template evalu<T, 1, GENERICARCH>(i);
//                 if (val < result)
//                     result = val;
//             }
//         }

//         return result;
//     }

//     FORCE_INLINE static T reduce_max(
//         const Expr &expr) noexcept
//     {
//         typename K::VecType acc = K::set1(NumericLimits<T>::lowest());

//         // SIMD loop
//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             acc = K::max(acc, expr.template evalu<T, Bits, Arch>(i * simdWidth));
//         }

//         // Horizontal reduction
//         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//         K::store(tmp, acc);

//         T result = tmp[0];
//         for (my_size_t i = 1; i < simdWidth; ++i)
//         {
//             if (tmp[i] > result)
//                 result = tmp[i];
//         }

//         // Scalar remainder
//         if constexpr (hasRemainder)
//         {
//             for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
//             {
//                 T val = expr.template evalu<T, 1, GENERICARCH>(i);
//                 if (val > result)
//                     result = val;
//             }
//         }

//         return result;
//     }

//     FORCE_INLINE static T reduce_sum(
//         const Expr &expr) noexcept
//     {
//         typename K::VecType acc = K::set1(T{0});

//         // SIMD loop
//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             acc = K::add(acc, expr.template evalu<T, Bits, Arch>(i * simdWidth));
//         }

//         // Horizontal reduction
//         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//         K::store(tmp, acc);

//         T result = tmp[0];
//         for (my_size_t i = 1; i < simdWidth; ++i)
//         {
//             result += tmp[i];
//         }

//         // Scalar remainder
//         if constexpr (hasRemainder)
//         {
//             for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
//             {
//                 result += expr.template evalu<T, 1, GENERICARCH>(i);
//             }
//         }

//         return result;
//     }

//     template <typename ExprLHS, typename ExprRHS>
//     FORCE_INLINE static bool reduce_all_approx_equal(
//         const ExprLHS &lhs,
//         const ExprRHS &rhs,
//         T tolerance) noexcept
//     {
//         // SIMD loop
//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             auto lhs_vec = lhs.template evalu<T, Bits, Arch>(i * simdWidth);
//             auto rhs_vec = rhs.template evalu<T, Bits, Arch>(i * simdWidth);
//             if (!K::all_within_tolerance(lhs_vec, rhs_vec, tolerance))
//             {
//                 return false;
//             }
//         }

//         // Scalar remainder
//         if constexpr (hasRemainder)
//         {
//             using ScalarK = Microkernel<T, 1, GENERICARCH>;
//             for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
//             {
//                 T lhs_val = lhs.template evalu<T, 1, GENERICARCH>(i);
//                 T rhs_val = rhs.template evalu<T, 1, GENERICARCH>(i);
//                 T abs_diff = ScalarK::abs(lhs_val - rhs_val);
//                 if (abs_diff > tolerance)
//                 {
//                     return false;
//                 }
//             }
//         }

//         return true;
//     }
// };

// template <typename T, my_size_t Bits, typename Arch>
// struct EinsumKernel
// {
//     using K = Microkernel<T, Bits, Arch>;
//     static constexpr my_size_t simdWidth = K::simdWidth;

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

//     // Contiguous dot product - both strides along k are 1
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static T dot_contiguous(
//         const Expr1 &expr1,
//         const Expr2 &expr2,
//         my_size_t base1,
//         my_size_t base2,
//         const my_size_t len) noexcept
//     {
//         const my_size_t simdSteps = len / simdWidth;

//         typename K::VecType acc = K::set1(T{0});

//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             auto v1 = expr1.template evalu<T, Bits, Arch>(base1 + i * simdWidth);
//             auto v2 = expr2.template evalu<T, Bits, Arch>(base2 + i * simdWidth);
//             acc = fmadd_safe(v1, v2, acc);
//         }

//         // Horizontal reduction
//         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//         K::store(tmp, acc);

//         T result = tmp[0];
//         for (my_size_t i = 1; i < simdWidth; ++i)
//         {
//             result += tmp[i];
//         }

//         // Remainder
//         for (my_size_t i = simdSteps * simdWidth; i < len; ++i)
//         {
//             T v1 = expr1.template evalu<T, 1, GENERICARCH>(base1 + i);
//             T v2 = expr2.template evalu<T, 1, GENERICARCH>(base2 + i);
//             result += v1 * v2;
//         }

//         return result;
//     }

//     // Strided dot product - scalar fallback
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static T dot_strided_scalar(
//         const Expr1 &expr1,
//         const Expr2 &expr2,
//         my_size_t idx1,
//         my_size_t idx2,
//         const my_size_t stride1,
//         const my_size_t stride2,
//         const my_size_t len) noexcept
//     {
//         T sum = T{0};
//         for (my_size_t k = 0; k < len; ++k)
//         {
//             T v1 = expr1.template evalu<T, 1, GENERICARCH>(idx1);
//             T v2 = expr2.template evalu<T, 1, GENERICARCH>(idx2);
//             sum += v1 * v2;
//             idx1 += stride1;
//             idx2 += stride2;
//         }
//         return sum;
//     }

//     // Strided dot product - SIMD with gather
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static T dot_strided_gather(
//         const Expr1 &expr1,
//         const Expr2 &expr2,
//         my_size_t idx1,
//         my_size_t idx2,
//         const my_size_t stride1,
//         const my_size_t stride2,
//         const my_size_t len) noexcept
//     {
//         const my_size_t simdSteps = len / simdWidth;

//         typename K::VecType acc = K::set1(T{0});

//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             auto v1 = expr1.template evalu_strided<T, Bits, Arch>(idx1, stride1);
//             auto v2 = expr2.template evalu_strided<T, Bits, Arch>(idx2, stride2);
//             acc = fmadd_safe(v1, v2, acc);

//             idx1 += simdWidth * stride1;
//             idx2 += simdWidth * stride2;
//         }

//         // Horizontal reduction
//         alignas(DATA_ALIGNAS) T tmp[simdWidth];
//         K::store(tmp, acc);

//         T result = tmp[0];
//         for (my_size_t i = 1; i < simdWidth; ++i)
//         {
//             result += tmp[i];
//         }

//         // Scalar remainder
//         for (my_size_t i = simdSteps * simdWidth; i < len; ++i)
//         {
//             T v1 = expr1.template evalu<T, 1, GENERICARCH>(idx1);
//             T v2 = expr2.template evalu<T, 1, GENERICARCH>(idx2);
//             result += v1 * v2;
//             idx1 += stride1;
//             idx2 += stride2;
//         }

//         return result;
//     }
// };

// #endif // KERNEL_OPS_H