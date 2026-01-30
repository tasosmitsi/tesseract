// Higher-level kernel operations built on top of microkernels
#ifndef KERNEL_OPS_H
#define KERNEL_OPS_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "numeric_limits.h"

// template <typename T, my_size_t Bits, typename Arch, my_size_t... Dims>
// struct TensorKernels
// {
//     using K = Microkernel<T, Bits, Arch>;
//     static constexpr my_size_t simdWidth = K::simdWidth;
//     static constexpr my_size_t dimCount = sizeof...(Dims);
//     static constexpr my_size_t totalSize = (Dims * ...); // product of all dims

//     // ========================================================================
//     // ELEMENT-WISE OPERATIONS
//     // ========================================================================

//     // Vectorized evaluation with contiguous storage assumption (aka the result tensor is not transposed)
//     template <typename Expr>
//     FORCE_INLINE static void eval_vectorized_contiguous(
//         T *output,
//         const Expr &expr,
//         auto &&unravelIndexfn) noexcept
//     {

//         const my_size_t simdSteps = totalSize / simdWidth;

//         // SIMD loop
//         for (my_size_t i = 0; i < simdSteps; ++i)
//         {
//             auto val = expr.template evalu<T, Bits, Arch>(i * simdWidth);
//             // Contiguous case — fast path
//             K::store(output + i * simdWidth, val);
//         }

//         if constexpr ((totalSize % simdWidth) != 0) // TODO: verify if constexpr works here
//         // in theory the compiler should be able to optimize out this branch if totalSize is known at compile time to be multiple of simdWidth
//         // but better be safe than sorry, whith constexpr we ensure no runtime overhead if not needed
//         {
//             // Scalar remainder
//             my_size_t indices[dimCount];
//             for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
//             {
//                 std::forward<decltype(unravelIndexfn)>(unravelIndexfn)(i, indices); // TODO: get rid of std
//                 // evaluate the remainder
//                 output[i] = expr(indices);
//             }
//         }
//     }

//     // Vectorized evaluation with non-contiguous storage (aka the result tensor is transposed)
//     // In other words this algorithm uses scatter which means the result is saved in non-continuous memmory
//     // TODO: not tested yet
//     // TODO: not sure if this is needed at all
//     // template <typename Expr>
//     // FORCE_INLINE static void eval_vectorized_non_contiguous(
//     //     T *output,
//     //     const Expr &expr,
//     //     my_size_t (&transposeOrder)[dimCount]) noexcept
//     // {

//     //     const my_size_t simdSteps = totalSize / simdWidth;

//     //     // SIMD loop
//     //     for (my_size_t i = 0; i < simdSteps; ++i)
//     //     {
//     //         my_size_t baseIdx = i * simdWidth;

//     //         auto val = expr.evalu(i * simdWidth);

//     //         // Non-contiguous (result tensor is transposed) case
//     //         my_size_t idxList[simdWidth];
//     //         for (int j = 0; j < simdWidth; ++j)
//     //             idxList[j] = remapFlatIndex(baseIdx + j, transposeOrder);
//     //         K::scatter(output, idxList, val);
//     //     }

//     //     // Scalar remainder TODO: this is wrong? — need to remap indices here as well?
//     //     for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
//     //     {
//     //         // std::cout << "Scalar remainder loop" << std::endl;
//     //         my_size_t indices[dimCount];
//     //         unravelIndex(i, indices, transposeOrder);
//     //         output[i] = expr(indices);
//     //     }
//     // }

//     template <typename Expr>
//     FORCE_INLINE static void eval_scalar(
//         T *output,
//         const Expr &expr,
//         auto &&unravelIndexfn) noexcept
//     {
//         // Pure scalar fallback
//         for (my_size_t i = 0; i < totalSize; ++i)
//         {
//             my_size_t indices[dimCount];
//             std::forward<decltype(unravelIndexfn)>(unravelIndexfn)(i, indices); // TODO: get rid of std
//             output[i] = expr(indices);
//         }
//     }
// };

template <typename Expr, my_size_t Bits, typename Arch>
struct KernelOps
{
    using T = typename Expr::value_type;
    using K = Microkernel<T, Bits, Arch>;

    static constexpr my_size_t simdWidth = K::simdWidth;
    static constexpr my_size_t numDims = Expr::NumDims;
    static constexpr my_size_t totalSize = Expr::TotalSize;
    static constexpr my_size_t simdSteps = totalSize / simdWidth;
    static constexpr bool hasRemainder = (totalSize % simdWidth) != 0;

    FORCE_INLINE static void eval_vectorized_contiguous(
        T *output,
        const Expr &expr,
        auto &&unravelIndexfn) noexcept
    {
        // SIMD loop
        for (my_size_t i = 0; i < simdSteps; ++i)
        {
            auto val = expr.template evalu<T, Bits, Arch>(i * simdWidth);
            K::store(output + i * simdWidth, val);
        }

        // Scalar remainder
        if constexpr (hasRemainder)
        {
            my_size_t indices[numDims];
            for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
            {
                unravelIndexfn(i, indices);
                output[i] = expr(indices);
            }
        }
    }

    FORCE_INLINE static void eval_scalar(
        T *output,
        const Expr &expr,
        auto &&unravelIndexfn) noexcept
    {
        my_size_t indices[numDims];
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            unravelIndexfn(i, indices);
            output[i] = expr(indices);
        }
    }

    FORCE_INLINE static T reduce_min(
        const Expr &expr,
        auto &&unravelIndexfn) noexcept
    {
        typename K::VecType acc = K::set1(NumericLimits<T>::max());

        // SIMD loop
        for (my_size_t i = 0; i < simdSteps; ++i)
        {
            acc = K::min(acc, expr.template evalu<T, Bits, Arch>(i * simdWidth));
        }

        // Horizontal reduction
        alignas(DATA_ALIGNAS) T tmp[simdWidth];
        K::store(tmp, acc);

        T result = tmp[0];
        for (my_size_t i = 1; i < simdWidth; ++i)
        {
            if (tmp[i] < result)
                result = tmp[i];
        }

        // Scalar remainder
        if constexpr (hasRemainder)
        {
            my_size_t indices[numDims];
            for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
            {
                unravelIndexfn(i, indices);
                T val = expr(indices);
                if (val < result)
                    result = val;
            }
        }

        return result;
    }

    FORCE_INLINE static T reduce_max(
        const Expr &expr,
        auto &&unravelIndexfn) noexcept
    {
        typename K::VecType acc = K::set1(NumericLimits<T>::lowest());

        // SIMD loop
        for (my_size_t i = 0; i < simdSteps; ++i)
        {
            acc = K::max(acc, expr.template evalu<T, Bits, Arch>(i * simdWidth));
        }

        // Horizontal reduction
        alignas(DATA_ALIGNAS) T tmp[simdWidth];
        K::store(tmp, acc);

        T result = tmp[0];
        for (my_size_t i = 1; i < simdWidth; ++i)
        {
            if (tmp[i] > result)
                result = tmp[i];
        }

        // Scalar remainder
        if constexpr (hasRemainder)
        {
            my_size_t indices[numDims];
            for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
            {
                unravelIndexfn(i, indices);
                T val = expr(indices);
                if (val > result)
                    result = val;
            }
        }

        return result;
    }

    FORCE_INLINE static T reduce_sum(
        const Expr &expr,
        auto &&unravelIndexfn) noexcept
    {
        typename K::VecType acc = K::set1(T{0});

        // SIMD loop
        for (my_size_t i = 0; i < simdSteps; ++i)
        {
            acc = K::add(acc, expr.template evalu<T, Bits, Arch>(i * simdWidth));
        }

        // Horizontal reduction
        alignas(DATA_ALIGNAS) T tmp[simdWidth];
        K::store(tmp, acc);

        T result = tmp[0];
        for (my_size_t i = 1; i < simdWidth; ++i)
        {
            result += tmp[i];
        }

        // Scalar remainder
        if constexpr (hasRemainder)
        {
            my_size_t indices[numDims];
            for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
            {
                unravelIndexfn(i, indices);
                result += expr(indices);
            }
        }

        return result;
    }
};

#endif // KERNEL_OPS_H