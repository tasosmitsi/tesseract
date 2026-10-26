#include <catch_amalgamated.hpp>

#include "fused/padding_policies/simd_padding_policy.h"
#include "fused/padding_policies/adapters/multi_sliced_pad_policy.h"
#include "fused/padding_policies/adapters/squeezed_pad_policy.h"
#include "fused/layouts/strided_layout_constexpr.h"

// ============================================================================
// The policy is a pure compile-time computation and thus every assertion here
// could be a static_assert. They are written as runtime REQUIREs so that a
// failure names the specific value rather than halting the build, and
// so the whole suite reports at once instead of one error at a time.
//
// The adapter's whole job is PhysicalDims. Dropping an axis lowers NumDims,
// and StridedLayoutConstExpr derives BaseStrides by row-major product over an
// array of that length, so PhysicalDims has to be rebuilt such that the derived
// strides come back out equal to the source's at the surviving axes. Every
// stride test below is checking that equality.
// ============================================================================

// ============================================================================
// SHAPE
// ============================================================================

TEST_CASE("squeezed_pad_policy: leading axis dropped",
          "[squeezed_pad_policy]")
{
    // [1,3,4] double, SimdWidth=4 → pad(4)=4, no padding
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 4>;
    using Policy = SqueezedPadPolicy<Source, 0>;

    REQUIRE(Policy::NumDims == 2);

    REQUIRE(Policy::LogicalDims[0] == 3);
    REQUIRE(Policy::LogicalDims[1] == 4);

    // Axis 0 folds into the next surviving axis: 1*3 = 3
    REQUIRE(Policy::PhysicalDims[0] == 3);
    REQUIRE(Policy::PhysicalDims[1] == 4);

    REQUIRE(Policy::LogicalSize == 12);
}

TEST_CASE("squeezed_pad_policy: two leading axes dropped",
          "[squeezed_pad_policy]")
{
    // [1,1,2,4] double, SimdWidth=4 → pad(4)=4
    using Source = SimdPaddingPolicyBase<double, 4, 1, 1, 2, 4>;
    using Policy = SqueezedPadPolicy<Source, 0, 1>;

    REQUIRE(Policy::NumDims == 2);

    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 4);

    // Both dropped axes fold into PhysicalDims[0]: 1*1*2 = 2
    REQUIRE(Policy::PhysicalDims[0] == 2);
    REQUIRE(Policy::PhysicalDims[1] == 4);

    REQUIRE(Policy::LogicalSize == 8);
}

TEST_CASE("squeezed_pad_policy: middle axis dropped",
          "[squeezed_pad_policy]")
{
    // [2,1,4] double, SimdWidth=4
    using Source = SimdPaddingPolicyBase<double, 4, 2, 1, 4>;
    using Policy = SqueezedPadPolicy<Source, 1>;

    REQUIRE(Policy::NumDims == 2);

    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 4);

    // Axis 1 folds right, into the entry for axis 2: 1*4 = 4
    REQUIRE(Policy::PhysicalDims[0] == 2);
    REQUIRE(Policy::PhysicalDims[1] == 4);
}

TEST_CASE("squeezed_pad_policy: logical size is unchanged",
          "[squeezed_pad_policy]")
{
    // Every dropped axis contributed a factor of 1
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 1, 4>;
    using Policy = SqueezedPadPolicy<Source, 0, 2>;

    REQUIRE(Policy::LogicalSize == Source::LogicalSize);
}

TEST_CASE("squeezed_pad_policy: physical size is unchanged",
          "[squeezed_pad_policy]")
{
    // The fold partitions the source's PhysicalDims, so the product is
    // preserved whatever the grouping
    using Source = SimdPaddingPolicyBase<double, 4, 1, 1, 2, 5>;
    using Policy = SqueezedPadPolicy<Source, 0, 1>;

    REQUIRE(Policy::PhysicalSize == Source::PhysicalSize);

    my_size_t product = 1;
    for (my_size_t i = 0; i < Policy::NumDims; ++i)
    {
        product *= Policy::PhysicalDims[i];
    }
    REQUIRE(product == Source::PhysicalSize);
}

TEST_CASE("squeezed_pad_policy: drop order does not matter",
          "[squeezed_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 1, 4>;
    using Forward = SqueezedPadPolicy<Source, 0, 2>;
    using Reversed = SqueezedPadPolicy<Source, 2, 0>;

    REQUIRE(Forward::NumDims == Reversed::NumDims);
    for (my_size_t i = 0; i < Forward::NumDims; ++i)
    {
        REQUIRE(Forward::LogicalDims[i] == Reversed::LogicalDims[i]);
        REQUIRE(Forward::PhysicalDims[i] == Reversed::PhysicalDims[i]);
    }
}

// ============================================================================
// PADDED SOURCE
// ============================================================================

TEST_CASE("squeezed_pad_policy: padding is carried through the fold",
          "[squeezed_pad_policy]")
{
    // [1,2,5] double, SimdWidth=4 → pad(5)=8 → PhysicalDims [1,2,8]
    using Source = SimdPaddingPolicyBase<double, 4, 1, 2, 5>;
    using Policy = SqueezedPadPolicy<Source, 0>;

    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 5);

    // Axis 0 folds into the entry for axis 1: 1*2 = 2. The padded last axis
    // carries across untouched
    REQUIRE(Policy::PhysicalDims[0] == 2);
    REQUIRE(Policy::PhysicalDims[1] == 8);
}

TEST_CASE("squeezed_pad_policy: scalar SIMD width means no padding",
          "[squeezed_pad_policy]")
{
    // SimdWidth=1 (GENERICARCH) → pad(n)=n
    using Source = SimdPaddingPolicyBase<double, 1, 1, 4, 6>;
    using Policy = SqueezedPadPolicy<Source, 0>;

    REQUIRE(Policy::PhysicalDims[0] == 4);
    REQUIRE(Policy::PhysicalDims[1] == 6);
}

TEST_CASE("squeezed_pad_policy: float source pads to 8",
          "[squeezed_pad_policy]")
{
    // [1,4,6] float, SimdWidth=8 → pad(6)=8
    using Source = SimdPaddingPolicyBase<float, 8, 1, 4, 6>;
    using Policy = SqueezedPadPolicy<Source, 0>;

    REQUIRE(Policy::PhysicalDims[0] == 4);
    REQUIRE(Policy::PhysicalDims[1] == 8);
}

// ============================================================================
// SLICED SOURCE
// ============================================================================
// The case the adapter exists for, and the one that rules out checking each
// dropped axis for physical extent 1. MultiSlicedPadPolicy narrows LogicalDims
// and inherits PhysicalDims verbatim, so its pinned axes are logically 1 and
// physically whatever the source was.

TEST_CASE("squeezed_pad_policy: pinned slice axes are physically large",
          "[squeezed_pad_policy]")
{
    // [3,5,2,4] double, SimdWidth=4 → pad(4)=4 → PhysicalDims [3,5,2,4]
    using Base = SimdPaddingPolicyBase<double, 4, 3, 5, 2, 4>;
    using Sliced = MultiSlicedPadPolicy<Base, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    // The slice is logically [1,1,2,4] and physically still [3,5,2,4]
    REQUIRE(Sliced::LogicalDims[0] == 1);
    REQUIRE(Sliced::LogicalDims[1] == 1);
    REQUIRE(Sliced::PhysicalDims[0] == 3);
    REQUIRE(Sliced::PhysicalDims[1] == 5);

    using Policy = SqueezedPadPolicy<Sliced, 0, 1>;

    REQUIRE(Policy::NumDims == 2);
    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 4);

    // 3*5*2 folded into the first entry, 4 in the second
    REQUIRE(Policy::PhysicalDims[0] == 30);
    REQUIRE(Policy::PhysicalDims[1] == 4);

    REQUIRE(Policy::PhysicalSize == 3 * 5 * 2 * 4);
}

TEST_CASE("squeezed_pad_policy: sliced source with padding",
          "[squeezed_pad_policy]")
{
    // [3,5,2,5] double, SimdWidth=4 → pad(5)=8 → PhysicalDims [3,5,2,8]
    using Base = SimdPaddingPolicyBase<double, 4, 3, 5, 2, 5>;
    using Sliced = MultiSlicedPadPolicy<Base, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using Policy = SqueezedPadPolicy<Sliced, 0, 1>;

    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 5);

    REQUIRE(Policy::PhysicalDims[0] == 30);
    REQUIRE(Policy::PhysicalDims[1] == 8);
}

// ============================================================================
// STRIDES DERIVED BY THE LAYOUT
// ============================================================================

TEST_CASE("squeezed_pad_policy: strides match the source at surviving axes",
          "[squeezed_pad_policy]")
{
    // [1,3,4] → BaseStrides [12,4,1], surviving axes 1 and 2 give [4,1]
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 4>;
    using SourceLayout = StridedLayoutConstExpr<Source>;
    using Policy = SqueezedPadPolicy<Source, 0>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(SourceLayout::stride(1) == 4);
    REQUIRE(SourceLayout::stride(2) == 1);

    REQUIRE(Layout::stride(0) == SourceLayout::stride(1));
    REQUIRE(Layout::stride(1) == SourceLayout::stride(2));
}

TEST_CASE("squeezed_pad_policy: strides survive a padded middle drop",
          "[squeezed_pad_policy]")
{
    // [2,1,5] double, SimdWidth=4 → PhysicalDims [2,1,8]
    // BaseStrides [8,8,1], surviving axes 0 and 2 give [8,1]
    using Source = SimdPaddingPolicyBase<double, 4, 2, 1, 5>;
    using SourceLayout = StridedLayoutConstExpr<Source>;
    using Policy = SqueezedPadPolicy<Source, 1>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::stride(0) == SourceLayout::stride(0));
    REQUIRE(Layout::stride(1) == SourceLayout::stride(2));

    REQUIRE(Layout::stride(0) == 8);
    REQUIRE(Layout::stride(1) == 1);
}

TEST_CASE("squeezed_pad_policy: strides match on a sliced source",
          "[squeezed_pad_policy]")
{
    // The first caller. Source BaseStrides over [3,5,2,4] are [40,8,4,1]
    using Base = SimdPaddingPolicyBase<double, 4, 3, 5, 2, 4>;
    using Sliced = MultiSlicedPadPolicy<Base, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using SlicedLayout = StridedLayoutConstExpr<Sliced>;
    using Policy = SqueezedPadPolicy<Sliced, 0, 1>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(SlicedLayout::stride(2) == 4);
    REQUIRE(SlicedLayout::stride(3) == 1);

    REQUIRE(Layout::stride(0) == SlicedLayout::stride(2));
    REQUIRE(Layout::stride(1) == SlicedLayout::stride(3));
}

TEST_CASE("squeezed_pad_policy: logical strides are row-major over the squeezed shape",
          "[squeezed_pad_policy]")
{
    // Squeezed LogicalDims [3,4] → LogicalStrides [4,1]
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 4>;
    using Policy = SqueezedPadPolicy<Source, 0>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_stride(0) == 4);
    REQUIRE(Layout::logical_stride(1) == 1);
}

// ============================================================================
// INDEX MAPPING
// ============================================================================
// A size-1 axis contributes a coordinate that is always 0, so it adds
// 0 * stride to every physical offset and drops out of the logical flat
// ordering. Both index maps should therefore be unchanged by the squeeze.

TEST_CASE("squeezed_pad_policy: flat index maps identically to the source",
          "[squeezed_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 4>;
    using SourceLayout = StridedLayoutConstExpr<Source>;
    using Policy = SqueezedPadPolicy<Source, 0>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::LogicalSize == SourceLayout::LogicalSize);

    for (my_size_t n = 0; n < Layout::LogicalSize; ++n)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(n) ==
                SourceLayout::logical_flat_to_physical_flat(n));
    }
}

TEST_CASE("squeezed_pad_policy: flat index maps identically with padding",
          "[squeezed_pad_policy]")
{
    // [1,2,5] → PhysicalDims [1,2,8], so offsets skip the padding
    using Source = SimdPaddingPolicyBase<double, 4, 1, 2, 5>;
    using SourceLayout = StridedLayoutConstExpr<Source>;
    using Policy = SqueezedPadPolicy<Source, 0>;
    using Layout = StridedLayoutConstExpr<Policy>;

    for (my_size_t n = 0; n < Layout::LogicalSize; ++n)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(n) ==
                SourceLayout::logical_flat_to_physical_flat(n));
    }

    // flat 5 → squeezed coords (1,0) → 1*8 = 8
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 8);
}

TEST_CASE("squeezed_pad_policy: flat index maps identically on a sliced source",
          "[squeezed_pad_policy]")
{
    using Base = SimdPaddingPolicyBase<double, 4, 3, 5, 2, 4>;
    using Sliced = MultiSlicedPadPolicy<Base, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using SlicedLayout = StridedLayoutConstExpr<Sliced>;
    using Policy = SqueezedPadPolicy<Sliced, 0, 1>;
    using Layout = StridedLayoutConstExpr<Policy>;

    for (my_size_t n = 0; n < Layout::LogicalSize; ++n)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(n) ==
                SlicedLayout::logical_flat_to_physical_flat(n));
    }

    // The slice's own PhysicalBase is added by the view, not the layout,
    // so these are offsets within the corner: 8 consecutive doubles
    for (my_size_t n = 0; n < 8; ++n)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(n) == n);
    }
}

TEST_CASE("squeezed_pad_policy: coords map to the same offsets",
          "[squeezed_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 4>;
    using SourceLayout = StridedLayoutConstExpr<Source>;
    using Policy = SqueezedPadPolicy<Source, 0>;
    using Layout = StridedLayoutConstExpr<Policy>;

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(Layout::logical_coords_to_physical_flat(i, j) ==
                    SourceLayout::logical_coords_to_physical_flat(0, i, j));
        }
    }
}

// ============================================================================
// CONTIGUITY
// ============================================================================

TEST_CASE("squeezed_pad_policy: a contiguous corner stays contiguous",
          "[squeezed_pad_policy]")
{
    using Base = SimdPaddingPolicyBase<double, 4, 3, 5, 2, 4>;
    using Sliced = MultiSlicedPadPolicy<Base, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using Policy = SqueezedPadPolicy<Sliced, 0, 1>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // [2,4] with LogicalStrides [4,1] and Strides [4,1]: one unbroken span,
    // so a view over this loads instead of gathering
    REQUIRE(Layout::logical_stride(0) == Layout::stride(0));
    REQUIRE(Layout::logical_stride(1) == Layout::stride(1));
}

TEST_CASE("squeezed_pad_policy: a padded plane stays strided",
          "[squeezed_pad_policy]")
{
    // [2,4,6] double, SimdWidth=4 → pad(6)=8 → PhysicalDims [2,4,8].
    // Pin axis 0 to one plane, then squeeze it away
    using Base = SimdPaddingPolicyBase<double, 4, 2, 4, 6>;
    using Sliced = MultiSlicedPadPolicy<Base, Slice<0, 1, 1>>;
    using Policy = SqueezedPadPolicy<Sliced, 0>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_dim(0) == 4);
    REQUIRE(Layout::logical_dim(1) == 6);

    // Rows hold 6 logical elements and sit 8 apart, so the plane is not one
    // unbroken span and a view over it gathers
    REQUIRE(Layout::logical_stride(0) == 6);
    REQUIRE(Layout::stride(0) == 8);

    // The last axis still steps by one
    REQUIRE(Layout::logical_stride(1) == Layout::stride(1));
}

// ============================================================================
// THE TRAILING CONSTRAINT
// ============================================================================
// computeBaseStrides hardcodes the last stride to 1, so the source must already
// have base stride 1 at the last surviving axis. Every axis dropped after it
// must therefore multiply to 1.

TEST_CASE("squeezed_pad_policy: trailing product is one when nothing trails",
          "[squeezed_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 4>;
    using Policy = SqueezedPadPolicy<Source, 0>;

    REQUIRE(Policy::TrailingProduct == 1);
}

TEST_CASE("squeezed_pad_policy: the trailing axis can be dropped when unpadded",
          "[squeezed_pad_policy]")
{
    // SimdWidth=1, so [2,3,1] is physically [2,3,1] and axis 2 has extent 1.
    // The same shape under SimdWidth=4 would pad to [2,3,4] and be rejected
    using Source = SimdPaddingPolicyBase<double, 1, 2, 3, 1>;
    using Policy = SqueezedPadPolicy<Source, 2>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Policy::TrailingProduct == 1);
    REQUIRE(Policy::NumDims == 2);
    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 3);
    REQUIRE(Layout::stride(0) == 3);
    REQUIRE(Layout::stride(1) == 1);
}

// ============================================================================
// DROPPED-AXIS LOOKUP
// ============================================================================

TEST_CASE("squeezed_pad_policy: dropped axes array and membership",
          "[squeezed_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 1, 4>;
    using Policy = SqueezedPadPolicy<Source, 0, 2>;

    REQUIRE(Policy::NumDropped == 2);
    REQUIRE(Policy::SourceNumDims == 4);

    REQUIRE(Policy::Dropped[0] == 0);
    REQUIRE(Policy::Dropped[1] == 2);

    REQUIRE(Policy::is_dropped(0));
    REQUIRE_FALSE(Policy::is_dropped(1));
    REQUIRE(Policy::is_dropped(2));
    REQUIRE_FALSE(Policy::is_dropped(3));
}

// ============================================================================
// HIGHER RANK
// ============================================================================

TEST_CASE("squeezed_pad_policy: 5D source down to 3D",
          "[squeezed_pad_policy]")
{
    // [1,2,1,3,5] double, SimdWidth=4 → pad(5)=8 → PhysicalDims [1,2,1,3,8]
    using Source = SimdPaddingPolicyBase<double, 4, 1, 2, 1, 3, 5>;
    using SourceLayout = StridedLayoutConstExpr<Source>;
    using Policy = SqueezedPadPolicy<Source, 0, 2>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Policy::NumDims == 3);

    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 3);
    REQUIRE(Policy::LogicalDims[2] == 5);

    // axis 0 folds into the entry for axis 1: 1*2 = 2
    // axis 2 folds into the entry for axis 3: 1*3 = 3
    REQUIRE(Policy::PhysicalDims[0] == 2);
    REQUIRE(Policy::PhysicalDims[1] == 3);
    REQUIRE(Policy::PhysicalDims[2] == 8);

    // BaseStrides over [1,2,1,3,8] are [48,24,24,8,1]
    REQUIRE(Layout::stride(0) == SourceLayout::stride(1));
    REQUIRE(Layout::stride(1) == SourceLayout::stride(3));
    REQUIRE(Layout::stride(2) == SourceLayout::stride(4));

    for (my_size_t n = 0; n < Layout::LogicalSize; ++n)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(n) ==
                SourceLayout::logical_flat_to_physical_flat(n));
    }
}

// ============================================================================
// VALIDATION
// ============================================================================

// The negative cases are static_asserts and cannot be exercised from a runtime
// test. Instantiating them fails the build. They are listed here so the intent
// is recorded; verify them by temporarily uncommenting one at a time.

// TEST_CASE("squeezed_pad_policy: negative cases are compile-time errors",
//           "[squeezed_pad_policy]")
// {
//     using Source = SimdPaddingPolicyBase<double, 4, 1, 3, 1, 4>;
//     using Padded = SimdPaddingPolicyBase<double, 4, 1, 3, 1>;   // physically [1,3,4]
//     using Column = MultiSlicedPadPolicy<SimdPaddingPolicyBase<double, 4, 4, 6>,
//                                         Slice<1, 3, 1>>;        // [4,1] of a [4,8] buffer

//     // Uncomment one at a time

//     using Policy = SqueezedPadPolicy<Source, 9>;             // axis out of range
//     using Policy = SqueezedPadPolicy<Source, 1>;             // axis is not size 1
//     using Policy = SqueezedPadPolicy<Source, 0, 0>;          // duplicate axis
//     using Policy = SqueezedPadPolicy<Source>;                // empty pack
//     using Policy = SqueezedPadPolicy<Source, 0, 2, 3>;       // leaves one axis, below the rank floor
//     using Policy = SqueezedPadPolicy<Padded, 0, 2>;          // leaves one axis
//     using Policy = SqueezedPadPolicy<Column, 1>;             // column to vector, rank floor and trailing stride
//     (void)Policy::NumDims;
// }

TEST_CASE("squeezed_pad_policy: dropping down to the rank floor is accepted",
          "[squeezed_pad_policy]")
{
    // Two axes survive, which is the minimum. FusedVector<T,5> is
    // FusedMatrix<T,5,1>, so [5,1] is the shape a caller wants anyway
    using Source = SimdPaddingPolicyBase<double, 1, 1, 1, 5, 1>;
    using Policy = SqueezedPadPolicy<Source, 0, 1>;

    REQUIRE(Policy::NumDims == 2);
    REQUIRE(Policy::LogicalDims[0] == 5);
    REQUIRE(Policy::LogicalDims[1] == 1);
    REQUIRE(Policy::LogicalSize == 5);
}

TEST_CASE("squeezed_pad_policy: highest axis index is accepted",
          "[squeezed_pad_policy]")
{
    // SimdWidth=1, so dropping the last axis leaves stride 1 behind
    using Source = SimdPaddingPolicyBase<double, 1, 2, 3, 1>;
    using Policy = SqueezedPadPolicy<Source, 2>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Policy::NumDims == 2);
    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 3);
    REQUIRE(Layout::stride(0) == 3);
    REQUIRE(Layout::stride(1) == 1);
}
