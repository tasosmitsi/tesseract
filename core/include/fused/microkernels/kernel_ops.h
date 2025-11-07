// Higher-level kernel operations built on top of microkernels
#ifndef KERNEL_OPS_H
#define KERNEL_OPS_H

#include "fused/microkernels/microkernel_base.h"

// TensorKernels: Provides high-level operations using microkernels
template <typename T, typename Arch>
struct TensorKernels
{
    using K = Microkernel<T, Arch>;
    static constexpr my_size_t simdWidth = K::simdWidth;

    // ========================================================================
    // ELEMENT-WISE OPERATIONS
    // ========================================================================

    // Vectorized evaluation of expression templates
    template <typename Expr, my_size_t N>
    static inline void eval_vectorized(
        T *output,
        const Expr &expr,
        my_size_t totalSize,
        void (*unravelIndex)(my_size_t, my_size_t (&)[N]))
    {
        if constexpr (simdWidth > 1)
        {
            const my_size_t simdSteps = totalSize / simdWidth;

            // SIMD loop
            for (my_size_t i = 0; i < simdSteps; ++i)
            {
                my_size_t indices[N];
                unravelIndex(i * simdWidth, indices);

                auto val = expr.evalu(indices);
                K::store(output + i * simdWidth, val);
            }

            // Scalar remainder
            for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
            {
                my_size_t indices[N];
                unravelIndex(i, indices);
                output[i] = expr(indices);
            }
        }
        else
        {
            // Pure scalar fallback
            for (my_size_t i = 0; i < totalSize; ++i)
            {
                my_size_t indices[N];
                unravelIndex(i, indices);
                output[i] = expr(indices);
            }
        }
    }
};
