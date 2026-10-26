#pragma once

#include "config.h"
#include "containers/array.h"
#include "helper_traits.h"

/**
 * @brief Padding policy adapter that drops size-1 axes.
 *
 * StridedLayoutConstExpr reads five things from its padding policy: NumDims,
 * LogicalDims, PhysicalDims, LogicalSize, PhysicalSize. It derives BaseStrides
 * from PhysicalDims and LogicalStrides from LogicalDims, both by row-major
 * product, both over arrays of length NumDims.
 *
 * Array length is why a squeeze cannot inherit PhysicalDims the way
 * MultiSlicedPadPolicy does. NumDims sizes the arrays, and a squeeze lowers it,
 * so PhysicalDims needs rebuilding with one entry per surviving axis, chosen so
 * the derived strides come back out matching the source's.
 *
 * ============================================================================
 * WHY DROPPING AN AXIS MOVES NOTHING
 * ============================================================================
 *
 * A size-1 axis contributes a coordinate that is always 0, so it adds 0 * stride
 * to every physical offset, and its logical stride multiplies that same 0 in the
 * flat ordering. Both index maps come out unchanged, in both directions.
 *
 *   [1,1,2,4] logical flat 5 → coords (0,0,1,1) → 0 + 0 + 4 + 1 = 5
 *   [2,4]     logical flat 5 → coords (1,1)     →         4 + 1 = 5
 *
 * ============================================================================
 * DERIVING THE SQUEEZED PHYSICAL DIMS
 * ============================================================================
 *
 * Write the surviving axes a[0] < a[1] < ... < a[M-1] and the source's physical
 * dims P. computeBaseStrides sets result[M-1] = 1 and walks down with
 *
 *   result[k-1] = result[k] * PhysicalDims[k]
 *
 * Requiring result[k] to equal the source's base stride s[k] at axis a[k] and
 * solving downward gives
 *
 *   PhysicalDims[k] = s[k-1] / s[k] = product of P[j] for j in (a[k-1], a[k]]
 *
 * Read that as: fold each dropped axis's physical extent into the next surviving
 * axis to its right. A product over a range is an integer, so the division needs
 * no separate divisibility condition. Axes before a[0] fold into PhysicalDims[0].
 * computeBaseStrides never reads that entry, since its loop stops at index 1, but
 * folding into it keeps the product of PhysicalDims equal to the source's
 * PhysicalSize.
 *
 * One condition stays outside the solution. result[M-1] is hardcoded to 1, so the
 * source needs base stride 1 at the last surviving axis already, which means any
 * axes dropped after it multiply to 1. TrailingProduct checks that.
 *
 * Rank 2 is the floor, checked separately. FusedVector<T,N> is
 * FusedMatrix<T,N,1>, since matmul needs both operands at the same rank, so
 * nothing in the library takes a rank-1 shape. Squeezing [1,3,1] to [3] hands
 * the caller something to undo, and [3,1] is the answer either way. Neither
 * check subsumes the other: a rank-3 squeeze clears the floor and can still
 * land a strided axis last.
 *
 * ============================================================================
 * WHY A PER-AXIS "IS IT PHYSICALLY 1" CHECK WOULD BE WRONG
 * ============================================================================
 *
 * Under SimdPaddingPolicy, physical equals logical on every axis but the last, so
 * a dropped axis is physically 1 as well. A MultiSlicedPadPolicy source breaks
 * that: it narrows LogicalDims and inherits PhysicalDims verbatim.
 *
 *   [Na,Nb,2,4] sliced to LogicalDims [1,1,2,4], PhysicalDims [Na,Nb,2,4]
 *
 * Axes 0 and 1 are logically 1 and physically Na and Nb, so a check on physical
 * extent would reject the first caller this adapter exists for. The fold absorbs
 * them into PhysicalDims[0]:
 *
 *   PhysicalDims  = [2*Na*Nb, 4]
 *   BaseStrides   = [4, 1]          matching the source at axes 2 and 3
 *
 * @tparam SourcePadPolicy  Padding policy of the tensor or view being squeezed
 * @tparam DroppedAxes      Axes to drop, each of logical extent 1
 */
template <typename SourcePadPolicy, my_size_t... DroppedAxes>
struct SqueezedPadPolicy
{
    static constexpr my_size_t SourceNumDims = SourcePadPolicy::NumDims;
    static constexpr my_size_t NumDropped = sizeof...(DroppedAxes);

    static_assert(NumDropped > 0,
                  "SqueezedPadPolicy: at least one axis is required");

    static_assert(all_unique<DroppedAxes...>(),
                  "SqueezedPadPolicy: each axis may be dropped at most once");

    static_assert(max_value<DroppedAxes...>() < SourceNumDims,
                  "SqueezedPadPolicy: axis must be less than the source's number of dimensions");

    // Nothing in the library takes a rank-1 shape, so a squeeze that produces one
    // has overshot: [1,3,1] wants [3,1]. Spelled as an addition rather than
    // NumDims >= 2 so it holds for a rank-1 source, where the subtraction below
    // underflows and the comparison would pass on the wrapped value.
    static_assert(NumDropped + 2 <= SourceNumDims,
                  "SqueezedPadPolicy: a squeeze must leave at least two axes. "
                  "A column is [N,1], not [N]");

    // Clamped: a failed static_assert does not stop compilation, and an
    // underflowed NumDims sizes every Array below at 2^64.
    static constexpr my_size_t NumDims =
        (NumDropped + 2 <= SourceNumDims) ? SourceNumDims - NumDropped : 2;

    // The range guard on the left short-circuits, so an out-of-range axis does
    // not also index LogicalDims out of bounds. Without it one mistake reports
    // twice, since a failed static_assert does not stop compilation.
    static_assert(((DroppedAxes >= SourceNumDims ||
                    SourcePadPolicy::LogicalDims[DroppedAxes] == 1) &&
                   ...),
                  "SqueezedPadPolicy: only an axis of logical extent 1 can be dropped");

    static constexpr Array<my_size_t, NumDropped> Dropped{DroppedAxes...};

    static constexpr bool is_dropped(my_size_t axis) noexcept
    {
        for (my_size_t i = 0; i < NumDropped; ++i)
        {
            if (Dropped[i] == axis)
                return true;
        }
        return false;
    }

    // Every axis dropped after the last surviving one, multiplied together.
    // computeBaseStrides forces the last stride to 1, so this has to be 1.
    static constexpr my_size_t computeTrailingProduct() noexcept
    {
        my_size_t acc = 1;
        for (my_size_t j = SourceNumDims; j-- > 0;)
        {
            if (!is_dropped(j))
                break;
            acc *= SourcePadPolicy::PhysicalDims[j];
        }
        return acc;
    }

    static constexpr my_size_t TrailingProduct = computeTrailingProduct();

    static_assert(TrailingProduct == 1,
                  "SqueezedPadPolicy: the surviving trailing axis must have base stride 1. "
                  "Squeezing a strided axis into the last position asks the layout for a "
                  "shape it cannot describe; keep a trailing axis of extent 1 instead");

    static constexpr Array<my_size_t, NumDims> computeLogicalDims() noexcept
    {
        Array<my_size_t, NumDims> result{};

        my_size_t k = 0;
        for (my_size_t j = 0; j < SourceNumDims; ++j)
        {
            if (!is_dropped(j))
                result[k++] = SourcePadPolicy::LogicalDims[j];
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = computeLogicalDims();

    // Fold each dropped axis into the next surviving axis to its right, so the
    // derived base strides match the source's at those axes. Derivation above.
    static constexpr Array<my_size_t, NumDims> computePhysicalDims() noexcept
    {
        Array<my_size_t, NumDims> result{};

        my_size_t k = 0;
        my_size_t acc = 1;
        for (my_size_t j = 0; j < SourceNumDims; ++j)
        {
            acc *= SourcePadPolicy::PhysicalDims[j];
            if (!is_dropped(j))
            {
                result[k++] = acc;
                acc = 1;
            }
        }
        return result;
    }

    static constexpr Array<my_size_t, NumDims> PhysicalDims = computePhysicalDims();

    static constexpr my_size_t computeLogicalSize() noexcept
    {
        my_size_t size = 1;
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            size *= LogicalDims[i];
        }
        return size;
    }

    // Equals the source's LogicalSize, since every dropped axis contributed a
    // factor of 1. Computed rather than inherited so it agrees with LogicalDims
    // by construction.
    static constexpr my_size_t LogicalSize = computeLogicalSize();

    // The fold partitions the source's PhysicalDims, so the product of this
    // policy's PhysicalDims already equals the source's PhysicalSize. Inherited
    // to say that directly.
    static constexpr my_size_t PhysicalSize = SourcePadPolicy::PhysicalSize;
};
