// Higher-level kernel operations built on top of microkernels
#ifndef KERNEL_OPS_H
#define KERNEL_OPS_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"

template <typename T, my_size_t Bits, typename Arch, my_size_t... Dims>
struct TensorKernels
{
    using K = Microkernel<T, Bits, Arch>;
    static constexpr my_size_t simdWidth = K::simdWidth;
    static constexpr my_size_t dimCount = sizeof...(Dims);
    static constexpr my_size_t totalSize = (Dims * ...); // product of all dims

    // ========================================================================
    // ELEMENT-WISE OPERATIONS
    // ========================================================================

    // Vectorized evaluation with contiguous storage assumption (aka the result tensor is not transposed)
    template <typename Expr>
    FORCE_INLINE static void eval_vectorized_contiguous(
        T *output,
        const Expr &expr,
        auto &&unravelIndexfn) noexcept
    {

        const my_size_t simdSteps = totalSize / simdWidth;

        // SIMD loop
        for (my_size_t i = 0; i < simdSteps; ++i)
        {
            auto val = expr.evalu(i * simdWidth);
            // Contiguous case — fast path
            K::store(output + i * simdWidth, val);
        }

        // Scalar remainder
        my_size_t indices[dimCount];
        for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
        {
            std::forward<decltype(unravelIndexfn)>(unravelIndexfn)(i, indices); // TODO: get rid of std
            // evaluate the remainder
            output[i] = expr(indices);
        }
    }

    // Vectorized evaluation with non-contiguous storage (aka the result tensor is transposed)
    // In other words this algorithm uses scatter which means the result is saved in non-continuous memmory
    // TODO: not tested yet
    // TODO: not sure if this is needed at all
    // template <typename Expr>
    // FORCE_INLINE static void eval_vectorized_non_contiguous(
    //     T *output,
    //     const Expr &expr,
    //     my_size_t (&transposeOrder)[dimCount]) noexcept
    // {

    //     const my_size_t simdSteps = totalSize / simdWidth;

    //     // SIMD loop
    //     for (my_size_t i = 0; i < simdSteps; ++i)
    //     {
    //         my_size_t baseIdx = i * simdWidth;

    //         auto val = expr.evalu(i * simdWidth);

    //         // Non-contiguous (result tensor is transposed) case
    //         my_size_t idxList[simdWidth];
    //         for (int j = 0; j < simdWidth; ++j)
    //             idxList[j] = remapFlatIndex(baseIdx + j, transposeOrder);
    //         K::scatter(output, idxList, val);
    //     }

    //     // Scalar remainder TODO: this is wrong? — need to remap indices here as well?
    //     for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
    //     {
    //         // std::cout << "Scalar remainder loop" << std::endl;
    //         my_size_t indices[dimCount];
    //         unravelIndex(i, indices, transposeOrder);
    //         output[i] = expr(indices);
    //     }
    // }

    template <typename Expr>
    FORCE_INLINE static void eval_scalar(
        T *output,
        const Expr &expr,
        auto &&unravelIndexfn) noexcept
    {
        // Pure scalar fallback
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            my_size_t indices[dimCount];
            std::forward<decltype(unravelIndexfn)>(unravelIndexfn)(i, indices); // TODO: get rid of std
            output[i] = expr(indices);
        }
    }
};

#endif // KERNEL_OPS_H