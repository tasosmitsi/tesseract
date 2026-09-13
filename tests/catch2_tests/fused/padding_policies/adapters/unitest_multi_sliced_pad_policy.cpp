#include <catch_amalgamated.hpp>

#include "fused/padding_policies/simd_padding_policy.h"
#include "fused/padding_policies/adapters/multi_sliced_pad_policy.h"
#include "fused/layouts/strided_layout_constexpr.h"

// ============================================================================
// The policy is a pure compile-time computation and thus every assertion here
// could be a static_assert. They are written as runtime REQUIREs so that a
// failure names the specific value rather than halting the build, and
// so the whole suite reports at once instead of one error at a time.
// ============================================================================

// ============================================================================
// UNPADDED SOURCE
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: single axis narrowed",
          "[multi_sliced_pad_policy]")
{
    // [2,3,4] double, SimdWidth=4 → pad(4)=4, no padding
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 0, 2>>;

    REQUIRE(Policy::NumDims == 3);

    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 2); // narrowed from 3
    REQUIRE(Policy::LogicalDims[2] == 4);

    REQUIRE(Policy::LogicalSize == 2 * 2 * 4);
}

TEST_CASE("multi_sliced_pad_policy: two axes narrowed",
          "[multi_sliced_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;

    REQUIRE(Policy::LogicalDims[0] == 1); // pinned
    REQUIRE(Policy::LogicalDims[1] == 3); // untouched
    REQUIRE(Policy::LogicalDims[2] == 2); // narrowed

    REQUIRE(Policy::LogicalSize == 1 * 3 * 2);
}

TEST_CASE("multi_sliced_pad_policy: all axes narrowed",
          "[multi_sliced_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source,
                                        Slice<0, 0, 1>,
                                        Slice<1, 1, 1>,
                                        Slice<2, 2, 1>>;

    REQUIRE(Policy::LogicalDims[0] == 1);
    REQUIRE(Policy::LogicalDims[1] == 1);
    REQUIRE(Policy::LogicalDims[2] == 1);

    REQUIRE(Policy::LogicalSize == 1);
}

TEST_CASE("multi_sliced_pad_policy: full-extent slice is a no-op on shape",
          "[multi_sliced_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 0, 3>>;

    REQUIRE(Policy::LogicalDims[0] == 2);
    REQUIRE(Policy::LogicalDims[1] == 3); // Len == source extent
    REQUIRE(Policy::LogicalDims[2] == 4);

    REQUIRE(Policy::LogicalSize == Source::LogicalSize);
}

TEST_CASE("multi_sliced_pad_policy: offset does not affect shape",
          "[multi_sliced_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using AtZero = MultiSlicedPadPolicy<Source, Slice<2, 0, 2>>;
    using AtTwo = MultiSlicedPadPolicy<Source, Slice<2, 2, 2>>;

    // Only Len determines the shape
    // Offset shifts where the view starts, which lives in the
    // view's PhysicalBase, not in the policy
    REQUIRE(AtZero::LogicalDims[2] == AtTwo::LogicalDims[2]);
    REQUIRE(AtZero::LogicalSize == AtTwo::LogicalSize);
}

TEST_CASE("multi_sliced_pad_policy: physical dims inherited verbatim",
          "[multi_sliced_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(Policy::PhysicalDims[i] == Source::PhysicalDims[i]);
    }

    REQUIRE(Policy::PhysicalSize == Source::PhysicalSize);
}

// ============================================================================
// PADDED SOURCE, SHAPE SHOULD BE UNCHANGED, PHYSICAL SIDE REFLECTS PADDING
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: padded source keeps padded physical dims",
          "[multi_sliced_pad_policy]")
{
    // [2,3,5] double, SimdWidth=4 → pad(5)=8
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 5>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;

    // Logical shape is identical in both padded and unpadded cases
    REQUIRE(Policy::LogicalDims[0] == 1);
    REQUIRE(Policy::LogicalDims[1] == 3);
    REQUIRE(Policy::LogicalDims[2] == 2);

    // Physical side carries the padding through
    REQUIRE(Policy::PhysicalDims[2] == 8);
    REQUIRE(Policy::PhysicalSize == 2 * 3 * 8);
}

TEST_CASE("multi_sliced_pad_policy: slicing the padded axis keeps source padding",
          "[multi_sliced_pad_policy]")
{
    // [4,6] double, SimdWidth=4 → pad(6)=8 → PhysicalDims [4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 2, 3>>;

    REQUIRE(Policy::LogicalDims[0] == 4);
    REQUIRE(Policy::LogicalDims[1] == 3);
    // Narrowing to 3 must NOT re-pad to 4
    // the slicing changes the logical shape,
    // but the physical shape is inherited from the source
    REQUIRE(Policy::PhysicalDims[0] == 4);
    REQUIRE(Policy::PhysicalDims[1] == 8);
}

// ============================================================================
// STRIDES DERIVED BY THE LAYOUT WHICH IS THE POINT OF THE ADAPTER
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: row slice strides and logical strides",
          "[multi_sliced_pad_policy][layout]")
{
    // [4,6] double, SimdWidth=4 → PhysicalDims [4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 2, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [1,6]
    REQUIRE(Layout::logical_dim(0) == 1);
    REQUIRE(Layout::logical_dim(1) == 6);

    // Physical strides inherited from the source's PhysicalDims [4,8]
    REQUIRE(Layout::stride(0) == 8);
    REQUIRE(Layout::stride(1) == 1);

    // Logical strides over the view's own [1,6]
    REQUIRE(Layout::logical_stride(0) == 6);
    REQUIRE(Layout::logical_stride(1) == 1);
}

TEST_CASE("multi_sliced_pad_policy: column slice strides and logical strides",
          "[multi_sliced_pad_policy][layout]")
{
    // [4,6] double, SimdWidth=4 → PhysicalDims [4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 3, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [4,1]
    REQUIRE(Layout::logical_dim(0) == 4);
    REQUIRE(Layout::logical_dim(1) == 1);

    // Physical strides are still the source's. This is what a freshly computed
    // SimdPaddingPolicy<double,4,1> would get wrong (it would give [4,1])
    // because the view depends on the source's buffer and must use the source's strides
    REQUIRE(Layout::stride(0) == 8);
    REQUIRE(Layout::stride(1) == 1);

    REQUIRE(Layout::logical_stride(0) == 1);
    REQUIRE(Layout::logical_stride(1) == 1);
}

TEST_CASE("multi_sliced_pad_policy: 3D two-axis slice strides",
          "[multi_sliced_pad_policy][layout]")
{
    // [2,3,4] unpadded
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_dim(0) == 1);
    REQUIRE(Layout::logical_dim(1) == 3);
    REQUIRE(Layout::logical_dim(2) == 2);

    // LogicalStrides row-major over LogicalDims [1,3,2]
    REQUIRE(Layout::logical_stride(0) == 6);
    REQUIRE(Layout::logical_stride(1) == 2);
    REQUIRE(Layout::logical_stride(2) == 1);

    // BaseStrides row-major over PhysicalDims [2,3,4]
    REQUIRE(Layout::stride(0) == 12);
    REQUIRE(Layout::stride(1) == 4);
    REQUIRE(Layout::stride(2) == 1);
}

TEST_CASE("multi_sliced_pad_policy: same slice on padded source shifts strides only",
          "[multi_sliced_pad_policy][layout]")
{
    // [2,3,5] double, SimdWidth=4 → pad(5)=8, so PhysicalDims [2,3,8]
    using Padded = SimdPaddingPolicyBase<double, 4, 2, 3, 5>;
    using Policy = MultiSlicedPadPolicy<Padded, Slice<0, 1, 1>, Slice<2, 0, 2>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Logical side identical to the unpadded [2,3,4] case above
    REQUIRE(Layout::logical_dim(0) == 1);
    REQUIRE(Layout::logical_dim(1) == 3);
    REQUIRE(Layout::logical_dim(2) == 2);

    // LogicalStrides row-major over LogicalDims [1,3,2]
    REQUIRE(Layout::logical_stride(0) == 6);
    REQUIRE(Layout::logical_stride(1) == 2);
    REQUIRE(Layout::logical_stride(2) == 1);

    // Physical side reflects pad(5)=8
    // BaseStrides row-major over PhysicalDims [2,3,8]
    REQUIRE(Layout::stride(0) == 24);
    REQUIRE(Layout::stride(1) == 8);
    REQUIRE(Layout::stride(2) == 1);
}

// ============================================================================
// INDEX MAPPING THROUGH THE LAYOUT
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: flat index maps to source offset",
          "[multi_sliced_pad_policy][layout]")
{
    // unpadded
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // flat 3 → coords (0,1,1) → 0*12 + 1*4 + 1*1 = 5
    // The view would add PhysicalBase = 1*12 = 12 → 17 → source(1,1,1)
    REQUIRE(Layout::logical_flat_to_physical_flat(3) == 5);

    // flat 0 → coords (0,0,0) → 0
    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);

    // flat 5 → coords (0,2,1) → 2*4 + 1 = 9
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 9);
}

TEST_CASE("multi_sliced_pad_policy: column slice flat indices step by row stride",
          "[multi_sliced_pad_policy][layout]")
{
    // [4,6] double, SimdWidth=4 → PhysicalDims [4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 3, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [4,1]: consecutive flats walk down the column
    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 8);
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 16);
    REQUIRE(Layout::logical_flat_to_physical_flat(3) == 24);
}

TEST_CASE("multi_sliced_pad_policy: row slice flat indices are consecutive",
          "[multi_sliced_pad_policy][layout]")
{
    // [4,6] double, SimdWidth=4 → PhysicalDims [4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 2, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [1,6]: a row of a padded matrix is contiguous
    for (my_size_t i = 0; i < 6; ++i)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(i) == i);
    }
}

TEST_CASE("multi_sliced_pad_policy: coords map to source offsets",
          "[multi_sliced_pad_policy][layout]")
{
    // [2,3,4] unpadded
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [1,3,2]
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1, 1) == 5);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 2, 1) == 9);
}

// ============================================================================
// CONTIGUITY OF THE PHYSICAL BUFFER WHEN SLICED
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: row slice is contiguous",
          "[multi_sliced_pad_policy][layout]")
{
    // [4,6] double, SimdWidth=4 → PhysicalDims [4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 2, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [1,6]: one whole row, physically contiguous
    // axis 0 has size 1 → never traversed, stride irrelevant
    REQUIRE(Layout::logical_dim(0) == 1);
    // axis 1 strides agree → contiguous
    REQUIRE(Layout::logical_stride(1) == Layout::stride(1));
}

TEST_CASE("multi_sliced_pad_policy: column slice is not contiguous",
          "[multi_sliced_pad_policy][layout]")
{
    // [4,6] double, SimdWidth=4 → PhysicalDims [4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 3, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [4,1]: one whole column, physically strided by the row stride
    // axis 0 is traversed and its strides disagree → gather required
    REQUIRE(Layout::logical_dim(0) == 4);
    REQUIRE(Layout::logical_stride(0) != Layout::stride(0));
}

TEST_CASE("multi_sliced_pad_policy: pinned outer axis is contiguous",
          "[multi_sliced_pad_policy][layout]")
{
    // [2,3,4] unpadded
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [1,3,4] : one whole plane, physically contiguous
    REQUIRE(Layout::logical_dim(0) == 1);
    REQUIRE(Layout::logical_stride(1) == Layout::stride(1));
    REQUIRE(Layout::logical_stride(2) == Layout::stride(2));
}

TEST_CASE("multi_sliced_pad_policy: pinned middle axis is not contiguous",
          "[multi_sliced_pad_policy][layout]")
{
    // [2,3,4] unpadded
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 2, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [2,1,4]: two planes 12 apart
    REQUIRE(Layout::logical_dim(0) == 2);
    REQUIRE(Layout::logical_stride(0) == 4);
    REQUIRE(Layout::stride(0) == 12);
    REQUIRE(Layout::logical_stride(0) != Layout::stride(0));
}

// ============================================================================
// 2D VECTOR-LIKE SOURCES
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: column vector subvector",
          "[multi_sliced_pad_policy][layout]")
{
    // [5,1] double, SimdWidth=4 → pad(1)=4 → PhysicalDims [5,4]
    using Source = SimdPaddingPolicyBase<double, 4, 5, 1>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 3>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_dim(0) == 3);
    REQUIRE(Layout::logical_dim(1) == 1);

    // Vector elements sit one per padded row
    REQUIRE(Layout::stride(0) == 4);
    REQUIRE(Layout::logical_stride(0) == 1);

    // Not contiguous (matches the gather observed in SubVectorView)
    REQUIRE(Layout::logical_stride(0) != Layout::stride(0));

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 4);
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 8);
}

TEST_CASE("multi_sliced_pad_policy: row vector subvector is contiguous",
          "[multi_sliced_pad_policy][layout]")
{
    // [1,5] double, SimdWidth=4 → pad(5)=8 → PhysicalDims [1,8]
    using Source = SimdPaddingPolicyBase<double, 4, 1, 5>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 2, 3>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_dim(0) == 1);
    REQUIRE(Layout::logical_dim(1) == 3);

    REQUIRE(Layout::stride(1) == 1);
    REQUIRE(Layout::logical_stride(1) == 1);

    // Contiguous (matches the direct load observed in SubVectorView)
    REQUIRE(Layout::logical_stride(1) == Layout::stride(1));

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 1);
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 2);
}

// ============================================================================
// SINGLE-ELEMENT AND DEGENERATE SOURCES
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: 1x1 source",
          "[multi_sliced_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 1, 1>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 0, 1>>;

    REQUIRE(Policy::LogicalDims[0] == 1);
    REQUIRE(Policy::LogicalDims[1] == 1);
    REQUIRE(Policy::LogicalSize == 1);
}

TEST_CASE("multi_sliced_pad_policy: scalar SIMD width means no padding",
          "[multi_sliced_pad_policy][layout]")
{
    // SimdWidth=1 (GENERICARCH) → pad(n)=n, PhysicalDims == LogicalDims
    using Source = SimdPaddingPolicyBase<double, 1, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 2, 1>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Policy::PhysicalDims[1] == 6);
    REQUIRE(Layout::stride(0) == 6);
    REQUIRE(Layout::stride(1) == 1);
}

// ============================================================================
// FLOAT SOURCE MEANS DIFFERENT SIMD WIDTH
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: float source pads to 8",
          "[multi_sliced_pad_policy][layout]")
{
    // [4,6] float, SimdWidth=8 → pad(6)=8
    using Source = SimdPaddingPolicyBase<float, 8, 4, 6>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 0, 2>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Policy::PhysicalDims[1] == 8);
    REQUIRE(Layout::logical_dim(1) == 2);
    REQUIRE(Layout::stride(0) == 8);
}

// ============================================================================
// SLICE ORDER INDEPENDENCE
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: slice order does not matter",
          "[multi_sliced_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Forward = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;
    using Reversed = MultiSlicedPadPolicy<Source, Slice<2, 0, 2>, Slice<0, 1, 1>>;

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(Forward::LogicalDims[i] == Reversed::LogicalDims[i]);
    }
    REQUIRE(Forward::LogicalSize == Reversed::LogicalSize);
}

// ============================================================================
// PERMUTATION APPLIED TO THE SLICED SHAPE (slice-then-transpose)
// ============================================================================
// This order works today: the slice is already applied, so the perm simply
// relabels the axes of the narrowed shape. TODO: The other order: slicing an
// already-transposed source is broken and tracked in the roadmap.

TEST_CASE("multi_sliced_pad_policy: permutation over a padded slice",
          "[multi_sliced_pad_policy][layout]")
{
    // [2,3,5] double, SimdWidth=4 → pad(5)=8, so PhysicalDims [2,3,8]
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 5>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;

    // Sliced shape [1,3,2], reversed → [2,3,1]
    using Layout = StridedLayoutConstExpr<Policy, 2, 1, 0>;

    SECTION("logical dims are the sliced dims, permuted")
    {
        REQUIRE(Layout::logical_dim(0) == 2);
        REQUIRE(Layout::logical_dim(1) == 3);
        REQUIRE(Layout::logical_dim(2) == 1);
    }

    SECTION("strides are the padded source strides, permuted")
    {
        // BaseStrides over PhysicalDims [2,3,8] are [24,8,1]
        // Permuted by [2,1,0] → [1,8,24]
        REQUIRE(Layout::stride(0) == 1);
        REQUIRE(Layout::stride(1) == 8);
        REQUIRE(Layout::stride(2) == 24);
    }

    SECTION("logical strides are row-major over the permuted shape")
    {
        // Row-major over [2,3,1] → [3,1,1]
        REQUIRE(Layout::logical_stride(0) == 3);
        REQUIRE(Layout::logical_stride(1) == 1);
        REQUIRE(Layout::logical_stride(2) == 1);
    }

    SECTION("flat index maps through both the perm and the padding")
    {
        // flat 4 → coords (1,1,0) → 1*1 + 1*8 + 0*24 = 9
        REQUIRE(Layout::logical_flat_to_physical_flat(4) == 9);

        // flat 0 → coords (0,0,0) → 0
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);

        // flat 3 → coords (1,0,0) → 1*1 = 1
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);
    }
}

TEST_CASE("multi_sliced_pad_policy: transposed padded matrix slice",
          "[multi_sliced_pad_policy][layout]")
{
    // [4,6] double, SimdWidth=4 → pad(6)=8, so PhysicalDims [4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 4, 6>;

    // Take columns 2..4 → sliced shape [4,3]
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 2, 3>>;

    // Transpose the slice → [3,4]
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    REQUIRE(Layout::logical_dim(0) == 3);
    REQUIRE(Layout::logical_dim(1) == 4);

    // BaseStrides [8,1] permuted → [1,8]
    REQUIRE(Layout::stride(0) == 1);
    REQUIRE(Layout::stride(1) == 8);

    // Row-major over [3,4] → [4,1]
    REQUIRE(Layout::logical_stride(0) == 4);
    REQUIRE(Layout::logical_stride(1) == 1);

    // flat 5 → 5/4=1 r1, 1/1=1 → coords (1,1) → 1*1 + 1*8 = 9
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 9);
}

// ============================================================================
// VALIDATION
// ============================================================================

// The negative cases (bad axis, zero Len, out-of-range Offset+Len, duplicate
// axes, too many slices) are static_asserts and cannot be exercised from a
// runtime test. Instantiating them fails the build. They are listed in the
// comment below so the intent is recorded; verify them by temporarily
// uncommenting one at a time.

// TEST_CASE("multi_sliced_pad_policy: negative cases are compile-time errors",
//           "[multi_sliced_pad_policy][validation]")
// {
//     using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;   // 3D
//     using Source2D = SimdPaddingPolicyBase<double, 4, 4, 6>;    // 2D

//     // Uncomment one at a time

//     using Policy = MultiSlicedPadPolicy<Source, Slice<9, 0, 1>>;                 // axis out of range
//     using Policy = MultiSlicedPadPolicy<Source, Slice<0, 0, 0>>;                 // Len == 0
//     using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 5>>;                 // Offset+Len > extent
//     using Policy = MultiSlicedPadPolicy<Source, Slice<0, 0, 1>, Slice<0, 1, 1>>; // duplicate axis
//     using Policy = MultiSlicedPadPolicy<Source>;                                 // empty pack
//     using Policy = MultiSlicedPadPolicy<Source2D, Slice<0,0,1>, Slice<1,0,1>, Slice<0,0,1>>; // more slices than dims
//     (void)Policy::NumDims;
// }

TEST_CASE("multi_sliced_pad_policy: boundary offsets are accepted",
          "[multi_sliced_pad_policy][validation]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;

    // Offset + Len exactly equal to the extent
    using AtEnd = MultiSlicedPadPolicy<Source, Slice<2, 2, 2>>;
    REQUIRE(AtEnd::LogicalDims[2] == 2);

    // Last valid single element
    using LastElement = MultiSlicedPadPolicy<Source, Slice<2, 3, 1>>;
    REQUIRE(LastElement::LogicalDims[2] == 1);

    // Full extent from zero
    using Whole = MultiSlicedPadPolicy<Source, Slice<2, 0, 4>>;
    REQUIRE(Whole::LogicalDims[2] == 4);
}

TEST_CASE("multi_sliced_pad_policy: highest axis index is accepted",
          "[multi_sliced_pad_policy][validation]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<2, 0, 1>>;

    REQUIRE(Policy::LogicalDims[2] == 1);
}

TEST_CASE("multi_sliced_pad_policy: one slice per dimension is accepted",
          "[multi_sliced_pad_policy][validation]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source,
                                        Slice<0, 0, 2>,
                                        Slice<1, 0, 3>,
                                        Slice<2, 0, 4>>;

    REQUIRE(Policy::LogicalSize == Source::LogicalSize);
}

// ============================================================================
// SLICED AXES / LENS LOOKUP ARRAYS
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: sliced axes and lens arrays",
          "[multi_sliced_pad_policy]")
{
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<0, 1, 1>, Slice<2, 0, 2>>;

    REQUIRE(Policy::SlicedAxes[0] == 0);
    REQUIRE(Policy::SlicedAxes[1] == 2);

    REQUIRE(Policy::SlicedLens[0] == 1);
    REQUIRE(Policy::SlicedLens[1] == 2);
}

// ============================================================================
// HIGHER RANK
// ============================================================================

TEST_CASE("multi_sliced_pad_policy: 4D source",
          "[multi_sliced_pad_policy][layout]")
{
    // [2,3,4,5] double, SimdWidth=4 → pad(5)=8 → PhysicalDims [2,3,4,8]
    using Source = SimdPaddingPolicyBase<double, 4, 2, 3, 4, 5>;
    using Policy = MultiSlicedPadPolicy<Source, Slice<1, 1, 1>, Slice<3, 0, 2>>;
    using Layout = StridedLayoutConstExpr<Policy>;

    // Shape [2,1,4,2]
    REQUIRE(Layout::logical_dim(0) == 2);
    REQUIRE(Layout::logical_dim(1) == 1);
    REQUIRE(Layout::logical_dim(2) == 4);
    REQUIRE(Layout::logical_dim(3) == 2);

    // BaseStrides row-major over [2,3,4,8]: [96, 32, 8, 1]
    REQUIRE(Layout::stride(0) == 96);
    REQUIRE(Layout::stride(1) == 32);
    REQUIRE(Layout::stride(2) == 8);
    REQUIRE(Layout::stride(3) == 1);

    // LogicalStrides row-major over [2,1,4,2]: [8, 8, 2, 1]
    REQUIRE(Layout::logical_stride(0) == 8);
    REQUIRE(Layout::logical_stride(1) == 8);
    REQUIRE(Layout::logical_stride(2) == 2);
    REQUIRE(Layout::logical_stride(3) == 1);

    REQUIRE(Policy::LogicalSize == 2 * 1 * 4 * 2);
}
