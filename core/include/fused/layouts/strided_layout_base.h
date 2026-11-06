#pragma once

#include "config.h"
#include "containers/array.h"
#include "fused/layouts/layout_base.h"

/**
 * @brief Index functions shared by every strided layout: a tensor's and each view's.
 *
 * A strided layout maps logical coordinates to an offset in the root tensor's
 * buffer:
 *
 *   offset(c) = Base + sum_i c[i] * Strides[i]        0 <= c[i] < LogicalDims[i]
 *
 * The derived layout supplies four things: NumDims (as N), LogicalDims,
 * Strides and Base. Everything else is computed from those four, once, here and
 * in LayoutBase, so StridedLayout and every view layout in layout_adapters.h
 * answer the same questions the same way, in both directions.
 *
 * Base is where the layout starts in the root buffer: 0 for a tensor, the
 * slice's starting offset for a slice, carried through by every other
 * transform. Because a layout adapter reads only these four things from its
 * source, a layout built on another layout composes: a slice of a squeeze of a
 * slice is one layout, in any order.
 *
 * Nothing here knows about padding. Strides are stored, so the padding policy
 * reaches StridedLayout and stops there.
 *
 * @tparam Derived  The layout (CRTP)
 * @tparam N        Its number of dimensions
 */
template <typename Derived, my_size_t N>
struct StridedLayoutBase : LayoutBase<Derived, N>
{
    using LayoutBase<Derived, N>::LogicalSize;
    using LayoutBase<Derived, N>::LogicalStrides;
    using LayoutBase<Derived, N>::is_logical_index_in_bounds;
    using LayoutBase<Derived, N>::logical_coords_to_logical_flat;

    /**
     * @brief Get the physical stride at dimension i: how far one step along it moves in the root buffer.
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= N
     */
    FORCE_INLINE static constexpr my_size_t stride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Derived::Strides.at(i);
    }

    /**
     * @brief True when walking the layout in logical row-major order visits
     * consecutive offsets, so SIMD can load instead of gather.
     *
     * Holds when, for every axis, either the extent is 1 (the axis is never
     * traversed, so its stride is irrelevant) or the logical stride equals the
     * physical one. Examples, double with SimdWidth=4:
     *
     *   Row of [4,6]    → LogicalDims [1,6], LogicalStrides [6,1], Strides [8,1]
     *     axis 0: extent 1, skipped.  axis 1: 1 == 1.  → contiguous
     *
     *   Column of [4,6] → LogicalDims [4,1], LogicalStrides [1,1], Strides [8,1]
     *     axis 0: extent 4 and 1 != 8.  → strided, gather
     */
    FORCE_INLINE static constexpr bool is_contiguous() noexcept
    {
        for (my_size_t i = 0; i < N; ++i)
        {
            if (Derived::LogicalDims[i] == 1)
            {
                continue;
            }
            if (LogicalStrides[i] != Derived::Strides[i])
            {
                return false;
            }
        }
        return true;
    }

    // ========================================================================
    // INDEX CONVERSION: LOGICAL TO PHYSICAL
    // ========================================================================

    /**
     * Convert logical flat index to physical offset in the root buffer.
     *
     * Uses forward iteration + subtraction to minimize divisions.
     * Only N divisions instead of 2N with reverse + modulo.
     */
    FORCE_INLINE static constexpr my_size_t logical_flat_to_physical_flat(my_size_t logical_flat) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if (logical_flat >= LogicalSize)
        {
            MyErrorHandler::error("logical_flat_to_physical_flat: logical_flat index out of bounds");
        }
        my_size_t offset = Derived::Base;
        for (my_size_t i = 0; i < N; ++i)
        {
            const my_size_t s = LogicalStrides[i];
            const my_size_t idx = logical_flat / s;
            logical_flat -= idx * s;
            offset += idx * Derived::Strides[i];
        }
        return offset;
    }

    /**
     * @brief Logical coordinates (Array multi-index) to physical offset in the root buffer (bounds-checked).
     *
     * @return constexpr my_size_t
     * @throws if any index is out of bounds for its logical dimension
     */
    FORCE_INLINE static constexpr my_size_t logical_coords_to_physical_flat(const my_size_t (&indices)[N]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if (!is_logical_index_in_bounds(indices))
        {
            MyErrorHandler::error("logical_coords_to_physical_flat: index out of bounds for logical dimension");
        }

        my_size_t flat = Derived::Base;
        for (my_size_t i = 0; i < N; ++i)
        {
            flat += indices[i] * Derived::Strides[i];
        }
        return flat;
    }

    /**
     * @brief Logical coordinates (variadic multi-index) to physical offset in the root buffer (bounds-checked).
     *
     * @tparam Indices
     * @param indices
     * @return constexpr my_size_t
     * @throws if any index is out of bounds for its logical dimension
     */
    template <typename... Indices>
    FORCE_INLINE static constexpr my_size_t logical_coords_to_physical_flat(Indices... indices) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        static_assert(sizeof...(Indices) == N, "Wrong number of indices");
        my_size_t idx_arr[] = {static_cast<my_size_t>(indices)...};
        return logical_coords_to_physical_flat(idx_arr);
    }

    // ========================================================================
    // INDEX CONVERSION: PHYSICAL TO LOGICAL
    // ========================================================================

    /**
     * @brief True when an element of this layout sits at this offset.
     *
     * False for a padding slot and for any offset a view does not cover. Check
     * this first when you walk a raw buffer: physical_flat_to_logical_coords
     * reports an error for such an offset, and on a target whose error handler
     * halts, that stops the program.
     *
     * Example: column 1 of a 2x3 matrix padded to 2x4 holds offsets 1 and 5.
     *   offset 5 → true
     *   offset 2 → false, it belongs to column 2
     *   offset 3 → false, it is padding
     *
     * @param physical_flat
     * @return constexpr bool
     */
    FORCE_INLINE static constexpr bool is_physical_flat_in_layout(my_size_t physical_flat) noexcept
    {
        my_size_t indices[N]{};
        return decompose(physical_flat, indices);
    }

    /**
     * @brief Physical offset to logical coordinates (bounds-checked).
     *
     * The inverse of logical_coords_to_physical_flat, for every strided layout.
     *
     * Example: transpose of a 2x3 matrix padded to 2x4, physical flat 5:
     *   LogicalDims = [3, 2], Strides = [1, 4], Base = 0
     *   largest stride first: 5 / 4 = 1, 5 - 4 = 1
     *   then:                 1 / 1 = 1
     *   → coords (1, 1)
     *
     * @param physical_flat
     * @param indices Output parameter for logical coordinates
     * @return constexpr void
     * @throws if no element of this layout sits at physical_flat
     */
    FORCE_INLINE static constexpr void physical_flat_to_logical_coords(my_size_t physical_flat, my_size_t (&indices)[N]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if (!decompose(physical_flat, indices))
        {
            MyErrorHandler::error("physical_flat_to_logical_coords: no element of this layout at physical_flat");
        }
    }

    /**
     * @brief Physical offset to logical flat index (bounds-checked).
     *
     * The inverse of logical_flat_to_physical_flat.
     *
     * @param physical_flat
     * @return constexpr my_size_t
     * @throws if no element of this layout sits at physical_flat
     */
    FORCE_INLINE static constexpr my_size_t physical_flat_to_logical_flat(my_size_t physical_flat) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        my_size_t indices[N]{};
        physical_flat_to_logical_coords(physical_flat, indices);
        return logical_coords_to_logical_flat(indices);
    }

private:
    /**
     * @brief The axes from largest stride to smallest.
     *
     * A layout's axes nest: each stride is larger than everything the axes with
     * smaller strides can reach. So dividing an offset in this order recovers
     * one coordinate at a time, whatever order the axes are stored in.
     *
     * @return constexpr Array<my_size_t, N>
     */
    static constexpr Array<my_size_t, N> computeAxisOrder() noexcept
    {
        Array<my_size_t, N> result{};
        for (my_size_t i = 0; i < N; ++i)
        {
            result[i] = i;
        }
        for (my_size_t i = 1; i < N; ++i)
        {
            for (my_size_t j = i; j > 0 && Derived::Strides[result[j - 1]] < Derived::Strides[result[j]]; --j)
            {
                const my_size_t axis = result[j];
                result[j] = result[j - 1];
                result[j - 1] = axis;
            }
        }
        return result;
    }

    static constexpr Array<my_size_t, N> AxisOrder = computeAxisOrder();

    /**
     * @brief Offset to coordinates. False when no element sits at that offset.
     *
     * An axis of extent 1 is skipped: its coordinate is always 0, and its stride
     * can equal another axis's. Anything left over at the end, or a coordinate
     * past its extent, means the offset is padding or outside the layout.
     */
    static constexpr bool decompose(my_size_t physical_flat, my_size_t (&indices)[N]) noexcept
    {
        if (physical_flat < Derived::Base)
        {
            return false;
        }

        my_size_t rest = physical_flat - Derived::Base;
        for (my_size_t k = 0; k < N; ++k)
        {
            const my_size_t axis = AxisOrder[k];
            indices[axis] = 0;
            if (Derived::LogicalDims[axis] == 1)
            {
                continue;
            }

            const my_size_t s = Derived::Strides[axis];
            const my_size_t idx = rest / s;
            if (idx >= Derived::LogicalDims[axis])
            {
                return false;
            }
            indices[axis] = idx;
            rest -= idx * s;
        }
        return rest == 0;
    }
};
