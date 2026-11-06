#pragma once

#include "config.h"
#include "containers/array.h"
#include "fused/layouts/strided_layout_base.h"

/**
 * @brief Compile-time strided layout of a tensor's padded storage.
 *
 * @tparam PadPolicy  Padding policy (e.g., SimdPaddingPolicy<T, Dims...>)
 *
 * All computations happen at compile-time. Zero runtime overhead.
 *
 * Every layout chain starts here. This is the only layout that reads a padding
 * policy: it turns the policy's dims into the LogicalDims and Strides the view
 * layouts in layout_adapters.h build on. Permutation is not part of this layout:
 * PermutedLayout adds it on top.
 *
 * The index functions and the conversions between logical and physical come
 * from LayoutBase and StridedLayoutBase, shared with the view layouts. The only
 * thing that stays here is what only a tensor's own storage can answer: the padded
 * grid, meaning PhysicalSize and physical coordinates, which reach the padding slots.
 * A tensor starts at the beginning of its buffer, so Base is 0.
 */
template <typename PadPolicy>
struct StridedLayout : StridedLayoutBase<StridedLayout<PadPolicy>, PadPolicy::NumDims>
{
    static constexpr my_size_t NumDims = PadPolicy::NumDims;
    static constexpr my_size_t PhysicalSize = PadPolicy::PhysicalSize;

    // A tensor's layout starts at the beginning of its own buffer
    static constexpr my_size_t Base = 0;

    using PadPolicyType = PadPolicy;

    /**
     * @brief Compute strides from the padded dimensions, row-major.
     *
     * @return constexpr Array<my_size_t, NumDims>
     */
    static constexpr Array<my_size_t, NumDims> computeStrides() noexcept
    {
        Array<my_size_t, NumDims> result{};
        result[NumDims - 1] = 1;
        for (my_size_t i = NumDims - 1; i > 0; --i)
        {
            result[i - 1] = result[i] * PadPolicy::PhysicalDims[i];
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = PadPolicy::LogicalDims;
    static constexpr Array<my_size_t, NumDims> Strides = computeStrides();

    // ========================================================================
    // PHYSICAL CONVERSION (padded storage only)
    // ========================================================================

    /**
     * @brief Physical flat index to physical coordinates.
     *
     * Decomposes a flat index into multi-dimensional coordinates
     * using Strides (row-major order in physical memory).
     *
     * Input must be in range [0, PhysicalSize).
     *
     * WARNING: If physical_flat is a padding location, the returned
     * coords will also be in the padding area.
     *
     * Example: 2x3 matrix padded to 2x4, physical flat 7:
     *   Strides = [4, 1]
     *   7 / 4 = 1, 7 - 4 = 3
     *   3 / 1 = 3
     *   → coords (1, 3)  ← column 3 is padding (LogicalDims[1] = 3)
     *
     * @param physical_flat
     * @param indices Output parameter for multi-dimensional coordinates
     * @return constexpr void
     * @throws if physical_flat is out of bounds for PhysicalSize
     */
    FORCE_INLINE static constexpr void physical_flat_to_physical_coords(my_size_t physical_flat, my_size_t (&indices)[NumDims]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        // check physical_flat is in range
        if (physical_flat >= PhysicalSize)
        {
            MyErrorHandler::error("physical_flat_to_physical_coords: physical_flat index out of bounds");
        }

        for (my_size_t i = 0; i < NumDims; ++i)
        {
            const my_size_t s = Strides[i];
            const my_size_t idx = physical_flat / s;
            indices[i] = idx;
            physical_flat -= idx * s;
        }
    }

    /**
     * @brief Physical coordinates to physical flat index (bounds-checked).
     *
     * The inverse of physical_flat_to_physical_coords. Coordinates in the
     * padding area are valid here, since the padded grid includes them.
     *
     * Example: 2x3 matrix padded to 2x4, coords (1, 3):
     *   Strides = [4, 1]
     *   1 * 4 + 3 * 1 = 7
     *
     * @param indices Physical coordinates
     * @return constexpr my_size_t
     * @throws if any index is out of bounds for its physical dimension
     */
    FORCE_INLINE static constexpr my_size_t physical_coords_to_physical_flat(const my_size_t (&indices)[NumDims]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        my_size_t flat = 0;
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            if (indices[i] >= PadPolicy::PhysicalDims[i])
            {
                MyErrorHandler::error("physical_coords_to_physical_flat: index out of bounds for physical dimension");
            }
            flat += indices[i] * Strides[i];
        }
        return flat;
    }
};
