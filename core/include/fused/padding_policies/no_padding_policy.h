#pragma once

#include "config.h"           // for my_size_t, BITS, DefaultArch
#include "containers/array.h" // for Array

/**
 * @brief No-padding policy - for explicit opt-out of padding.
 *
 * @tparam T     Element type
 * @tparam Dims  Logical dimensions
 *
 * Use this when:
 *   - Memory is extremely constrained
 *   - You're willing to use unaligned loads (loadu) everywhere
 *   - Testing/debugging to isolate padding-related issues
 *
 * Note: With GENERICARCH (simdWidth=1), SimdPaddingPolicy already produces
 * no padding. This policy exists for explicit opt-out on SIMD architectures.
 */
template <typename T, my_size_t... Dims>
struct NoPaddingPolicy
{
    static_assert(sizeof...(Dims) > 0, "NoPaddingPolicy: At least one dimension is required");

    static constexpr my_size_t NumDims = sizeof...(Dims);

    static constexpr Array<my_size_t, NumDims> computeLogicalDims()
    {
        return Array<my_size_t, NumDims>{Dims...};
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = computeLogicalDims();
    static constexpr Array<my_size_t, NumDims> PhysicalDims = LogicalDims;

    static constexpr my_size_t LastDim = LogicalDims[NumDims - 1];
    static constexpr my_size_t PaddedLastDim = LastDim; // no padding
    static constexpr my_size_t LogicalSize = (Dims * ...);
    static constexpr my_size_t PhysicalSize = LogicalSize; // no overhead
    static constexpr my_size_t SimdWidth = 1;              // effectively scalar
};