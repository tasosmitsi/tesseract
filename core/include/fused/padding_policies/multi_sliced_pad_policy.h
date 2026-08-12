#pragma once

#include "config.h"
#include "containers/array.h"
#include "helper_traits.h"

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

/**
 * @brief Per-slice compile-time validation.
 *
 * Instantiated once per Slice so the compiler names the offending
 * Slice<Axis, Offset, Len> in its diagnostic. A fold over the whole pack
 * would report only that some slice was invalid, not which.
 *
 * All three asserts in a struct are evaluated even after an earlier one fails,
 * so an out-of-range Axis would otherwise index LogicalDims (3rd test)
 * out of bounds in a constant expression and produce
 * a confusing second error on top of the real one.
 *
 * @tparam SourcePadPolicy  The padding policy of the tensor being sliced
 * @tparam S                The Slice descriptor to validate
 */
template <typename SourcePadPolicy, typename S>
struct ValidateSlice
{
    // TODO: axis numbers here are policy-space. Once MultiSliceView remaps
    // view-space axes through the source's permutation, a Slice<0,...> written
    // against a transposed view will be reported against a different axis
    // number than the user wrote. Pass the original axis in for the diagnostic,
    // or validate before remapping.

    static_assert(S::Axis < SourcePadPolicy::NumDims,
                  "Slice: Axis must be less than the source's number of dimensions");

    static_assert(S::Len > 0,
                  "Slice: Len must be greater than 0");

    // we first check that Axis is in range, so this indexing is safe and won't produce
    // a confusing second error.
    static_assert(S::Axis >= SourcePadPolicy::NumDims ||
                      S::Offset + S::Len <= SourcePadPolicy::LogicalDims[S::Axis],
                  "Slice: Offset + Len exceeds the source's extent along Axis");

    static constexpr bool ok = true;
};

/**
 * @brief Padding policy adapter that describes a multi-axis slice.
 *
 * StridedLayoutConstExpr only reads five things from its padding policy:
 * NumDims, LogicalDims, PhysicalDims, LogicalSize, PhysicalSize. It derives
 *
 *   - BaseStrides from PhysicalDims: how far one step along an axis moves
 *     in memory
 *   - LogicalStrides from LogicalDims: how a flat index decomposes into
 *     coordinates
 *   - bounds checks from LogicalDims
 *
 * For an ordinary tensor both arrays come from the same policy and describe
 * the same object. A slice needs them to describe two different things: the
 * view's shape, and the source's memory. Handing the layout a fresh
 * SimdPaddingPolicy over the view's dims would recompute padding from the
 * narrowed shape and produce the wrong strides.
 *
 * For example, a column slice
 * of a [4,6] matrix that gives [4,1] where the source requires [8,1].
 *
 * This adapter passes PhysicalDims and PhysicalSize through untouched and
 * overrides only LogicalDims and LogicalSize, narrowing every axis named in
 * the Slices pack and leaving the rest at their source extent.
 *
 * The slices' starting offsets are NOT part of the layout and thus not part of the
 * padding policy.
 *
 * Worked example, source [2,3,5] double with SimdWidth=4:
 *
 *   Source PadPolicy:  LogicalDims [2,3,5], PhysicalDims [2,3,8]
 *                      (pad(5) = 8, rounded up to the next multiple of 4)
 *
 *   Slice<0,1,1> and Slice<2,0,2>:
 *       LogicalDims: start from the source's [2,3,5], then apply each Len:
 *          axis 0 named by Slice<0,1,1> → 1
 *          axis 1 unnamed               → 3
 *          axis 2 named by Slice<2,0,2> → 2
 *          → [1,3,2]      (unchanged, Len does not care about padding)
 *
 *     So LogicalDims [1,3,2] and
 *     PhysicalDims [2,3,8] (inherited verbatim from the source)
 *
 *     BaseStrides    [24,8,1]  row-major over PhysicalDims [2,3,8]
 *     LogicalStrides [6,2,1]   row-major over LogicalDims  [1,3,2]
 *
 *     PhysicalBase = 1*24 + 0*1 = 24
 *
 *   Only the strides and PhysicalBase differ. That is why PhysicalDims must
 *   be inherited rather than recomputed from the narrowed shape.
 *
 * @tparam SourcePadPolicy  The padding policy of the tensor being sliced
 * @tparam Slices           One Slice descriptor per narrowed axis
 */
template <typename SourcePadPolicy, typename... Slices>
struct MultiSlicedPadPolicy
{
    static constexpr my_size_t NumDims = SourcePadPolicy::NumDims;

    static_assert(sizeof...(Slices) > 0,
                  "MultiSlicedPadPolicy: at least one Slice is required");

    static_assert(sizeof...(Slices) <= NumDims,
                  "MultiSlicedPadPolicy: more slices than the source has dimensions");

    static_assert(all_unique<Slices::Axis...>(),
                  "MultiSlicedPadPolicy: each axis may be sliced at most once");

    // Per-slice bounds and extent checks
    // Each instantiation names its own Slice in the diagnostic TODO:check that
    static_assert((ValidateSlice<SourcePadPolicy, Slices>::ok && ...));

    // Which axes are sliced, and to what extent
    static constexpr Array<my_size_t, sizeof...(Slices)> SlicedAxes{Slices::Axis...};
    static constexpr Array<my_size_t, sizeof...(Slices)> SlicedLens{Slices::Len...};

    static constexpr Array<my_size_t, NumDims> computeLogicalDims() noexcept
    {
        Array<my_size_t, NumDims> result{};

        // Start from the source's shape
        for (my_size_t i = 0; i < NumDims; ++i)
            result[i] = SourcePadPolicy::LogicalDims[i];

        // Narrow each sliced axis
        for (my_size_t s = 0; s < sizeof...(Slices); ++s)
            result[SlicedAxes[s]] = SlicedLens[s];

        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = computeLogicalDims();

    // Physical dims are inherited from the SourcePadPolicy, the slice lives inside the
    // source's buffer and must use the source's strides and thus the source's physical dims
    static constexpr Array<my_size_t, NumDims> PhysicalDims = SourcePadPolicy::PhysicalDims;

    static constexpr my_size_t computeLogicalSize() noexcept
    {
        my_size_t size = 1;
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            size *= LogicalDims[i];
        }
        return size;
    }

    static constexpr my_size_t LogicalSize = computeLogicalSize();

    // Physical size is the same as the source's
    static constexpr my_size_t PhysicalSize = SourcePadPolicy::PhysicalSize;
};
