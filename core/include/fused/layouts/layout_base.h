#pragma once

#include "config.h"
#include "containers/array.h"
#include "helper_traits.h"

/**
 * @brief Shape functions shared by every layout.
 *
 * A layout describes a logical shape and where each element of it sits in a
 * buffer. This base answers the shape half, from LogicalDims alone: sizes,
 * bounds, and the conversions between logical coordinates and a logical flat
 * index. It knows nothing about strides, padding or where a layout starts.
 * StridedLayoutBase adds those.
 *
 * The derived layout supplies LogicalDims, and its number of dimensions as N.
 *
 * @tparam Derived  The layout (CRTP)
 * @tparam N        Its number of dimensions
 */
template <typename Derived, my_size_t N>
struct LayoutBase
{
    static constexpr my_size_t computeLogicalSize() noexcept
    {
        my_size_t size = 1;
        for (my_size_t i = 0; i < N; ++i)
        {
            size *= Derived::LogicalDims[i];
        }
        return size;
    }

    /**
     * @brief Number of logical elements.
     *
     */
    static constexpr my_size_t LogicalSize = computeLogicalSize();

    /**
     * @brief Compute logical strides for flat index decomposition.
     *
     * @return constexpr Array<my_size_t, N>
     */
    static constexpr Array<my_size_t, N> computeLogicalStrides() noexcept
    {
        Array<my_size_t, N> result{};
        result[N - 1] = 1;
        for (my_size_t i = N - 1; i > 0; --i)
        {
            result[i - 1] = result[i] * Derived::LogicalDims[i];
        }
        return result;
    }

    /**
     * @brief Logical strides for flat index decomposition: row-major over LogicalDims.
     *
     */
    static constexpr Array<my_size_t, N> LogicalStrides = computeLogicalStrides();

    /**
     * @brief Get number of dimensions.
     *
     * @return constexpr my_size_t
     */
    FORCE_INLINE static constexpr my_size_t num_dims() noexcept { return N; }

    /**
     * @brief Get logical dimension at index i.
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= N
     */
    FORCE_INLINE static constexpr my_size_t logical_dim(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Derived::LogicalDims.at(i);
    }

    /**
     * @brief Get logical stride at dimension i (for flat index decomposition).
     *
     * @param i
     * @return constexpr my_size_t
     * @throws if i >= N
     */
    FORCE_INLINE static constexpr my_size_t logical_stride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return LogicalStrides.at(i);
    }

    /**
     * @brief Check if logical multi-index is in bounds.
     *
     * @return constexpr bool
     */
    FORCE_INLINE static constexpr bool is_logical_index_in_bounds(const my_size_t (&indices)[N]) noexcept
    {
        for (my_size_t i = 0; i < N; ++i)
        {
            if (indices[i] >= Derived::LogicalDims[i])
                return false;
        }
        return true;
    }

    /**
     * @brief Logical flat index to logical coordinates.
     *
     * Decomposes a flat index into multi-dimensional coordinates
     * using LogicalStrides (row-major order in logical space).
     *
     * Input must be in range [0, LogicalSize).
     *
     * Example: 3x4 matrix, logical flat 7:
     *   LogicalStrides = [4, 1]
     *   7 / 4 = 1, 7 - 4 = 3
     *   3 / 1 = 3
     *   → coords (1, 3)
     *
     * @param logical_flat Input flat index in logical space
     * @param indices Output parameter for multi-dimensional coordinates
     * @return constexpr void
     * @throws if logical_flat is out of bounds for LogicalSize
     */
    FORCE_INLINE static constexpr void logical_flat_to_logical_coords(my_size_t logical_flat, my_size_t (&indices)[N]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        // check logical_flat is in range
        if (logical_flat >= LogicalSize)
        {
            MyErrorHandler::error("logical_flat_to_logical_coords: logical_flat index out of bounds");
        }
        for (my_size_t i = 0; i < N; ++i)
        {
            const my_size_t s = LogicalStrides[i];
            const my_size_t idx = logical_flat / s;
            indices[i] = idx;
            logical_flat -= idx * s;
        }
    }

    /**
     * @brief Logical coordinates to logical flat index (bounds-checked).
     *
     * The inverse of logical_flat_to_logical_coords.
     *
     * Example: 3x4 matrix, coords (1, 3):
     *   LogicalStrides = [4, 1]
     *   1 * 4 + 3 * 1 = 7
     *
     * @param indices Multi-dimensional coordinates
     * @return constexpr my_size_t
     * @throws if any index is out of bounds for its logical dimension
     */
    FORCE_INLINE static constexpr my_size_t logical_coords_to_logical_flat(const my_size_t (&indices)[N]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if (!is_logical_index_in_bounds(indices))
        {
            MyErrorHandler::error("logical_coords_to_logical_flat: index out of bounds for logical dimension");
        }

        my_size_t flat = 0;
        for (my_size_t i = 0; i < N; ++i)
        {
            flat += indices[i] * LogicalStrides[i];
        }
        return flat;
    }
};

/**
 * @brief Raw array of a layout's logical dimensions, for expression types.
 *
 * Expression types expose their shape as a raw array, Dim. A view whose shape
 * comes from a layout has no per-dimension pack to expand into one. This
 * helper makes one with index_seq, so the view can bind a reference:
 *
 *   static constexpr const my_size_t (&Dim)[NumDims] = LayoutDims<Layout>::value;
 *
 * The Seq parameter is filled in by its default; callers pass the layout only.
 *
 * @tparam Layout  Any layout: a StridedLayout or a view layout
 * @tparam Seq     Index sequence over the layout's dimensions (defaulted)
 */
template <typename Layout,
          typename Seq = typename make_index_seq<Layout::NumDims>::type>
struct LayoutDims;

/// @cond
template <typename Layout, my_size_t... Is>
struct LayoutDims<Layout, index_seq<Is...>>
{
    static constexpr my_size_t value[] = {Layout::logical_dim(Is)...};
};
/// @endcond
