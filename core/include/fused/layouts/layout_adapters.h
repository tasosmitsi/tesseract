#pragma once

#include "config.h"
#include "containers/array.h"
#include "fused/layouts/strided_layout_base.h"
#include "helper_traits.h"

/**
 * @file layout_adapters.h
 * @brief View layouts: each one computes its layout from the layout under it.
 *
 * Every layout maps logical coordinates to an offset in the root tensor's
 * buffer as Base + sum_i c[i] * Strides[i] (see StridedLayoutBase). Each view
 * kind is a transform of (LogicalDims, Strides, Base):
 *
 *   SlicedLayout     narrows the named axes,    Base += Offset * Strides[Axis]
 *   SqueezedLayout   removes size-1 axes,       Base unchanged
 *   PermutedLayout   reorders axes,             Base unchanged
 *   DiagonalLayout   [N,N] → [N,1] with stride Strides[0] + Strides[1]
 *
 * An adapter reads only NumDims, logical_dim(i), stride(i) and Base from its
 * source, which every layout provides, the tensor's StridedLayout
 * included. So the source can itself be an adapter, and views built on views
 * compose in any order: a slice of a squeeze of a slice is one layout.
 *
 * Strides are stored, not derived from padded dims, so nothing has to be
 * folded or remapped to make them come out right: a squeeze can drop a padded
 * trailing axis, and a squeeze or slice can sit on a transpose.
 *
 * Examples:
 *
 *   SlicedLayout<T, Slice<0,1,1>, Slice<1,2,1>>  [1,1,2,4]  Strides [40,8,4,1]  Base 1*40 + 2*8 = 56
 *   SqueezedLayout<…, 0, 1>                      [2,4]      Strides [4,1]       Base 56
 *   SlicedLayout<…, Slice<0,1,1>>                [1,4]      Strides [4,1]       Base 56 + 1*4 = 60
 */

/**
 * @brief Descriptor for a single axis slice.
 *
 * Groups the three values that describe narrowing one dimension, so that
 * several of them can be passed as a parameter pack.
 *
 *   Slice<0, 1, 1>   pin axis 0 to index 1
 *   Slice<2, 0, 4>   take 4 elements of axis 2 starting at 0
 *
 * @tparam Axis_    Dimension index to slice along
 * @tparam Offset_  Start index along that dimension
 * @tparam Len_     Number of elements along that dimension
 */
template <my_size_t Axis_, my_size_t Offset_, my_size_t Len_>
struct Slice
{
    static constexpr my_size_t Axis = Axis_;
    static constexpr my_size_t Offset = Offset_;
    static constexpr my_size_t Len = Len_;
};

namespace detail
{
    /**
     * @brief An axis that is safe to index with even when its own range check failed.
     *
     * A failed static_assert does not stop compilation, so a check that indexes
     * by an out-of-range axis would add a second, confusing error on top of the
     * real one. Clamping to 0 keeps it to one.
     */
    template <my_size_t NumDims>
    constexpr my_size_t safe_axis(my_size_t axis) noexcept
    {
        return axis < NumDims ? axis : 0;
    }

    /**
     * @brief Per-slice compile-time validation.
     *
     * Instantiated once per Slice so the compiler names the offending
     * Slice<Axis, Offset, Len> in its diagnostic. A fold over the whole pack
     * would report only that some slice was invalid, not which. Axes are the
     * source's own, the ones the caller wrote.
     *
     * @tparam Src  The layout being sliced
     * @tparam S    The Slice descriptor to validate
     */
    template <typename Src, typename S>
    struct ValidateSlice
    {
        static_assert(S::Axis < Src::NumDims,
                      "Slice: Axis must be less than the source's number of dimensions");

        static_assert(S::Len > 0,
                      "Slice: Len must be greater than 0");

        static_assert(S::Axis >= Src::NumDims ||
                          S::Offset + S::Len <= Src::logical_dim(safe_axis<Src::NumDims>(S::Axis)),
                      "Slice: Offset + Len exceeds the source's extent along Axis");

        static constexpr bool ok = true;
    };

    /**
     * @brief Rank after dropping NumDropped axes, clamped.
     *
     * A failed static_assert does not stop compilation, and an underflowed rank
     * would size every Array at 2^64. The clamp keeps a bad pack to its own
     * diagnostic.
     */
    template <my_size_t SourceNumDims, my_size_t NumDropped>
    inline constexpr my_size_t squeezed_rank =
        (NumDropped + 2 <= SourceNumDims) ? SourceNumDims - NumDropped : 2;

    /**
     * @brief all_unique and max_value for packs that may be empty.
     *
     * Both helpers static_assert on an empty pack. An empty pack is already
     * reported by the adapter's own "at least one" check, so these keep it to
     * that one message.
     */
    template <my_size_t... Vals>
    constexpr bool no_duplicates() noexcept
    {
        if constexpr (sizeof...(Vals) < 2)
            return true;
        else
            return all_unique<Vals...>();
    }

    template <my_size_t... Vals>
    constexpr my_size_t max_or_zero() noexcept
    {
        if constexpr (sizeof...(Vals) == 0)
            return 0;
        else
            return max_value<Vals...>();
    }
} // namespace detail

// ============================================================================
// SLICE
// ============================================================================

/**
 * @brief Layout of a slice: narrows the named axes and moves Base to where the slice starts.
 *
 * Rank-preserving: a pinned axis becomes extent 1, it is not dropped (that is
 * SqueezedLayout). Axes not named keep their extent. Strides are the source's,
 * because a slice lives inside the source's memory.
 *
 *   source [2,3,5] double, Strides [24,8,1]
 *   Slice<0,1,1>, Slice<2,0,2>  →  LogicalDims [1,3,2], Strides [24,8,1], Base 1*24 = 24
 *
 * @tparam Src     The layout being sliced
 * @tparam Slices  One Slice<Axis, Offset, Len> per narrowed axis, in the source's axis numbering
 */
template <typename Src, typename... Slices>
struct SlicedLayout : StridedLayoutBase<SlicedLayout<Src, Slices...>, Src::NumDims>
{
    static constexpr my_size_t NumDims = Src::NumDims;

    static_assert(sizeof...(Slices) > 0,
                  "SlicedLayout: at least one Slice is required");

    static_assert(sizeof...(Slices) <= NumDims,
                  "SlicedLayout: more slices than the source has dimensions");

    static_assert(detail::no_duplicates<Slices::Axis...>(),
                  "SlicedLayout: each axis may be sliced at most once");

    // Per-slice bounds and extent checks, each naming its own Slice
    static_assert((detail::ValidateSlice<Src, Slices>::ok && ...));

    static constexpr Array<my_size_t, NumDims> computeLogicalDims() noexcept
    {
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims; ++i)
            result[i] = Src::logical_dim(i);
        ((result[detail::safe_axis<NumDims>(Slices::Axis)] = Slices::Len), ...);
        return result;
    }

    static constexpr Array<my_size_t, NumDims> computeStrides() noexcept
    {
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims; ++i)
            result[i] = Src::stride(i);
        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = computeLogicalDims();
    static constexpr Array<my_size_t, NumDims> Strides = computeStrides();

    // Every coordinate zero except the sliced axes, which start at their offsets
    static constexpr my_size_t Base =
        (Src::Base + ... + (Slices::Offset * Src::stride(detail::safe_axis<NumDims>(Slices::Axis))));
};

// ============================================================================
// SQUEEZE
// ============================================================================

/**
 * @brief Layout of a squeeze: removes size-1 axes.
 *
 * A size-1 axis contributes a coordinate that is always 0, so removing it from
 * LogicalDims and Strides moves nothing: every offset, and Base, is unchanged.
 *
 *   [1,1,2,4], drop 0,1  → [2,4]
 *   [1,3,1],   drop 0    → [3,1]
 *   [M,1,N],   drop 1    → [M,N]
 *
 * Axes are named rather than inferred: [1,3,1] dropping both would give [3],
 * and the library's vectors are [N,1] (rank floor below).
 *
 * @tparam Src   The layout being squeezed
 * @tparam Axes  Axes to drop, each of logical extent 1, in the source's axis numbering
 */
template <typename Src, my_size_t... Axes>
struct SqueezedLayout
    : StridedLayoutBase<SqueezedLayout<Src, Axes...>,
                        detail::squeezed_rank<Src::NumDims, sizeof...(Axes)>>
{
    static constexpr my_size_t SourceNumDims = Src::NumDims;
    static constexpr my_size_t NumDropped = sizeof...(Axes);
    static constexpr my_size_t NumDims = detail::squeezed_rank<SourceNumDims, NumDropped>;

    static_assert(NumDropped > 0,
                  "SqueezedLayout: at least one axis is required");

    static_assert(detail::no_duplicates<Axes...>(),
                  "SqueezedLayout: each axis may be dropped at most once");

    static_assert(detail::max_or_zero<Axes...>() < SourceNumDims,
                  "SqueezedLayout: axis must be less than the source's number of dimensions");

    // Nothing in the library takes a rank-1 shape (FusedVector<T,N> is
    // FusedMatrix<T,N,1>), so a squeeze that produces one has overshot: [1,3,1]
    // wants [3,1]. A convention, not a layout limit. Spelled as an addition so
    // it holds for a rank-1 source, where the subtraction would underflow.
    static_assert(NumDropped + 2 <= SourceNumDims,
                  "SqueezedLayout: a squeeze must leave at least two axes. "
                  "A column is [N,1], not [N]");

    // The range guard on the left short-circuits, so an out-of-range axis does
    // not also index out of bounds and report twice.
    static_assert(((Axes >= SourceNumDims ||
                    Src::logical_dim(detail::safe_axis<SourceNumDims>(Axes)) == 1) &&
                   ...),
                  "SqueezedLayout: only an axis of logical extent 1 can be dropped");

    static constexpr bool is_dropped(my_size_t axis) noexcept
    {
        return ((axis == Axes) || ...);
    }

    static constexpr Array<my_size_t, NumDims> keep(bool strides) noexcept
    {
        Array<my_size_t, NumDims> result{};
        my_size_t k = 0;
        for (my_size_t j = 0; j < SourceNumDims && k < NumDims; ++j)
        {
            if (!is_dropped(j))
                result[k++] = strides ? Src::stride(j) : Src::logical_dim(j);
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = keep(false);
    static constexpr Array<my_size_t, NumDims> Strides = keep(true);
    static constexpr my_size_t Base = Src::Base;
};

// ============================================================================
// PERMUTE
// ============================================================================

/**
 * @brief Layout of a permutation: axis i of the result is axis Perm[i] of the source.
 *
 *   source [2,3] padded to [2,4], Strides [4,1]
 *   Perm 1,0  →  LogicalDims [3,2], Strides [1,4], Base unchanged
 *
 * @tparam Src   The layout being permuted
 * @tparam Perm  A permutation of 0..NumDims-1
 */
template <typename Src, my_size_t... Perm>
struct PermutedLayout : StridedLayoutBase<PermutedLayout<Src, Perm...>, Src::NumDims>
{
    static constexpr my_size_t NumDims = Src::NumDims;

    static_assert(sizeof...(Perm) == NumDims,
                  "PermutedLayout: permutation pack must match the source's number of dimensions");

    static_assert(detail::no_duplicates<Perm...>(),
                  "PermutedLayout: permutation indices must be unique");

    static_assert(detail::max_or_zero<Perm...>() < NumDims,
                  "PermutedLayout: permutation index must be less than the source's number of dimensions");

    static constexpr Array<my_size_t, NumDims> reorder(bool strides) noexcept
    {
        constexpr my_size_t perm[] = {Perm...};
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims && i < sizeof...(Perm); ++i)
        {
            const my_size_t from = detail::safe_axis<NumDims>(perm[i]);
            result[i] = strides ? Src::stride(from) : Src::logical_dim(from);
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = reorder(false);
    static constexpr Array<my_size_t, NumDims> Strides = reorder(true);
    static constexpr my_size_t Base = Src::Base;
};

// ============================================================================
// DIAGONAL
// ============================================================================

/**
 * @brief Layout of the main diagonal of a square 2D layout, as an [N,1] column.
 *
 * Element i is source (i, i), which sits i * (Strides[0] + Strides[1]) from
 * Base. The second axis has extent 1, so its stride is never used.
 *
 *   source [3,3] padded to [3,4], Strides [4,1]
 *   → LogicalDims [3,1], Strides [5,1]: offsets 0, 5, 10
 *
 * @tparam Src  The square 2D layout
 */
template <typename Src>
struct DiagonalLayout : StridedLayoutBase<DiagonalLayout<Src>, 2>
{
    static constexpr my_size_t NumDims = 2;

    static_assert(Src::NumDims == 2,
                  "DiagonalLayout: source must have exactly 2 dimensions");

    static_assert(Src::NumDims != 2 ||
                      Src::logical_dim(0) == Src::logical_dim(detail::safe_axis<Src::NumDims>(1)),
                  "DiagonalLayout: source must be square");

    static constexpr Array<my_size_t, 2> LogicalDims{Src::logical_dim(0), 1};
    static constexpr Array<my_size_t, 2> Strides{
        Src::stride(0) + Src::stride(detail::safe_axis<Src::NumDims>(1)), 1};
    static constexpr my_size_t Base = Src::Base;
};
