#pragma once

#include "config.h"
#include "containers/array.h"

/**
 * @brief Padding policy adapter that describes a slice of another tensor.
 *
 * StridedLayoutConstExpr only reads five things from its padding policy:
 * NumDims, LogicalDims, PhysicalDims, LogicalSize, PhysicalSize. A slice
 * needs the layout to compute
 *
 *   - BaseStrides from the SOURCE's physical dims — the slice lives inside
 *     the source's buffer, so stepping one element along an axis moves the
 *     same distance it would in the source
 *   - LogicalStrides from the VIEW's logical dims — flat index decomposition
 *     must walk the slice's own shape, not the source's
 *   - bounds checks against the VIEW's logical dims
 *
 * This adapter supplies exactly that: the source's PhysicalDims verbatim,
 * and LogicalDims with the sliced axis narrowed to Len.
 *
 * The slice's starting offset is NOT part of the layout — it is a single
 * constant added to whatever the layout returns:
 *
 *   PhysicalBase = Offset * Layout::stride(Axis)
 *
 * Worked example, source [4,6] double with SimdWidth=4:
 *
 *   Source PadPolicy:  LogicalDims [4,6], PhysicalDims [4,8]
 *
 *   Row slice (Axis=0, Len=1):
 *     LogicalDims  [1,6]   PhysicalDims [4,8]
 *     BaseStrides  [8,1]   LogicalStrides [6,1]
 *     flat 3 → coords (0,3) → physical 3        (contiguous within a row)
 *
 *   Column slice (Axis=1, Len=1):
 *     LogicalDims  [4,1]   PhysicalDims [4,8]
 *     BaseStrides  [8,1]   LogicalStrides [1,1]
 *     flat 2 → coords (2,0) → physical 16       (one padded row apart)
 *
 * @tparam SourcePadPolicy  The padding policy of the tensor being sliced
 * @tparam Axis             Dimension index being narrowed
 * @tparam Len              New extent along Axis
 */
template <typename SourcePadPolicy, my_size_t Axis, my_size_t Len>
struct SlicedPadPolicy
{
    static constexpr my_size_t NumDims = SourcePadPolicy::NumDims;

    static_assert(Axis < NumDims,
                  "SlicedPadPolicy: Axis must be less than the source's number of dimensions");

    static_assert(Len > 0 && Len <= SourcePadPolicy::LogicalDims[Axis],
                  "SlicedPadPolicy: Len must be in (0, source extent along Axis]");

    // The view's logical shape — source dims with the sliced axis narrowed
    static constexpr Array<my_size_t, NumDims> computeLogicalDims() noexcept
    {
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            result[i] = (i == Axis) ? Len : SourcePadPolicy::LogicalDims[i];
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = computeLogicalDims();

    // Physical dims are inherited verbatim — the slice lives inside the
    // source's buffer and must use the source's strides
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

    // Physical size is the source's — the slice indexes into that buffer
    static constexpr my_size_t PhysicalSize = SourcePadPolicy::PhysicalSize;
};
