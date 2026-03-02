// // Higher-level kernel operations built on top of microkernels
// #ifndef KERNEL_OPS_H
// #define KERNEL_OPS_H

// #include "config.h"
// #include "fused/microkernels/microkernel_base.h"
// #include "numeric_limits.h"

// template <typename T, my_size_t Bits, typename Arch>
// struct KernelOps
// {
//     using K = Microkernel<T, Bits, Arch>;
//     static constexpr my_size_t simdWidth = K::simdWidth;

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

//     // ========================================================================
//     // Evaluation
//     // ========================================================================

//     template <typename Expr>
//     FORCE_INLINE static void eval_vectorized_contiguous(
//         T *output,
//         const Expr &expr) noexcept
//     {
//         using Layout = typename Expr::Layout;
//         static constexpr my_size_t physicalSize = Layout::PhysicalSize;
//         // static constexpr my_size_t totalSize = Expr::TotalSize;
//         static constexpr my_size_t simdSteps = physicalSize / simdWidth;
//         static constexpr bool hasRemainder = (physicalSize % simdWidth) != 0;

//         // SIMD loop
//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             auto val = expr.template evalu<T, Bits, Arch>(i * simdWidth);
//             K::store(output + i * simdWidth, val);
//         }

//         // Scalar remainder TODO: The whole point of padding is that PhysicalSize is already a multiple of SimdWidth — so there's no scalar remainder
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

//     // ========================================================================
//     // Reductions
//     // ========================================================================

//     template <typename Expr>
//     FORCE_INLINE static T reduce_min(const Expr &expr) noexcept
//     {
//         static constexpr my_size_t totalSize = Expr::TotalSize;
//         static constexpr my_size_t simdSteps = totalSize / simdWidth;
//         static constexpr bool hasRemainder = (totalSize % simdWidth) != 0;

//         typename K::VecType acc = K::set1(NumericLimits<T>::max());

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

//     template <typename Expr>
//     FORCE_INLINE static T reduce_max(const Expr &expr) noexcept
//     {
//         static constexpr my_size_t totalSize = Expr::TotalSize;
//         static constexpr my_size_t simdSteps = totalSize / simdWidth;
//         static constexpr bool hasRemainder = (totalSize % simdWidth) != 0;

//         typename K::VecType acc = K::set1(NumericLimits<T>::lowest());

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

//     template <typename Expr>
//     FORCE_INLINE static T reduce_sum(const Expr &expr) noexcept
//     {
//         static constexpr my_size_t totalSize = Expr::TotalSize;
//         static constexpr my_size_t simdSteps = totalSize / simdWidth;
//         static constexpr bool hasRemainder = (totalSize % simdWidth) != 0;

//         typename K::VecType acc = K::set1(T{0});

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

//         if constexpr (hasRemainder)
//         {
//             for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
//             {
//                 result += expr.template evalu<T, 1, GENERICARCH>(i);
//             }
//         }

//         return result;
//     }

//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static bool reduce_all_approx_equal(
//         const Expr1 &lhs,
//         const Expr2 &rhs,
//         T tolerance) noexcept
//     {
//         static constexpr my_size_t totalSize = Expr1::TotalSize;
//         static constexpr my_size_t simdSteps = totalSize / simdWidth;
//         static constexpr bool hasRemainder = (totalSize % simdWidth) != 0;

//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             auto lhs_vec = lhs.template evalu<T, Bits, Arch>(i * simdWidth);
//             auto rhs_vec = rhs.template evalu<T, Bits, Arch>(i * simdWidth);
//             if (!K::all_within_tolerance(lhs_vec, rhs_vec, tolerance))
//             {
//                 return false;
//             }
//         }

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

//     // ========================================================================
//     // Dot products (for einsum)
//     // ========================================================================

//     // Contiguous dot product - both strides along k are 1
//     template <typename Expr1, typename Expr2>
//     FORCE_INLINE static T dot_contiguous(
//         const Expr1 &expr1,
//         const Expr2 &expr2,
//         my_size_t base1,
//         my_size_t base2,
//         const my_size_t len) noexcept
//     {
//         const my_size_t steps = len / simdWidth;

//         typename K::VecType acc = K::set1(T{0});

//         for (my_size_t i = 0; i < steps; ++i)
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
//         for (my_size_t i = steps * simdWidth; i < len; ++i)
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
//         const my_size_t steps = len / simdWidth;

//         typename K::VecType acc = K::set1(T{0});

//         for (my_size_t i = 0; i < steps; ++i)
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
//         for (my_size_t i = steps * simdWidth; i < len; ++i)
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