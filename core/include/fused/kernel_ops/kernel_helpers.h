/**
 * @file kernel_helpers.h
 * @brief Shared SIMD helper utilities for kernel operations.
 *
 * Contains cross-cutting helpers used by multiple kernel sub-modules
 * (e.g., fmadd_safe used by both dot products and potentially reductions).
 */
#ifndef KERNEL_HELPERS_H
#define KERNEL_HELPERS_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"

namespace detail
{

    template <typename T, my_size_t Bits, typename Arch>
    struct KernelHelpers
    {
        using K = Microkernel<T, Bits, Arch>;

        /**
         * @brief Fused multiply-add with fallback for architectures without native FMA.
         *
         * Uses K::fmadd if available, otherwise falls back to K::add(K::mul(a, b), c).
         */
        FORCE_INLINE static typename K::VecType fmadd_safe(
            typename K::VecType a,
            typename K::VecType b,
            typename K::VecType c) noexcept
        {
            if constexpr (requires { K::fmadd(a, b, c); })
            {
                return K::fmadd(a, b, c);
            }
            else
            {
                return K::add(K::mul(a, b), c);
            }
        }
    };

} // namespace detail

#endif // KERNEL_HELPERS_H
