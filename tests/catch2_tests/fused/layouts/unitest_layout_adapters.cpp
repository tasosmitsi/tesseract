#include <catch_amalgamated.hpp>

#include "fused/padding_policies/no_padding_policy.h"
#include "fused/padding_policies/simd_padding_policy.h"
#include "fused/layouts/strided_layout.h"
#include "fused/layouts/layout_adapters.h"
#include "layout_two_way.h"
#include "simd_widths.h"

// ============================================================================
// The layouts are pure compile-time computations, so every assertion here
// could be a static_assert. They are runtime REQUIREs so that a failure names
// the specific value rather than halting the build, and so the whole suite
// reports at once instead of one error at a time.
//
// Each adapter computes (LogicalDims, Strides, Base) from its source layout.
// The tests check those against hand-computed values, and check element
// offsets against the same element reached through the source. The sources
// use an explicit SIMD width, so the padding, and with it every stride, is the
// same on every machine.
// ============================================================================

// ============================================================================
// SLICE
// ============================================================================

TEST_CASE("sliced_layout: one axis narrowed, strides kept, base moved",
          "[layout_adapters]")
{
    // [4,6] double, SimdWidth=4 → padded to [4,8] → Strides [8,1]
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 4, 6>>;
    using Row2 = SlicedLayout<Root, Slice<0, 2, 1>>;

    REQUIRE(Row2::NumDims == 2);
    REQUIRE(Row2::logical_dim(0) == 1);
    REQUIRE(Row2::logical_dim(1) == 6);
    REQUIRE(Row2::stride(0) == 8);
    REQUIRE(Row2::stride(1) == 1);
    REQUIRE(Row2::Base == 16);
    REQUIRE(Row2::LogicalSize == 6);
}

TEST_CASE("sliced_layout: two axes narrowed",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 4, 6>>;
    using Block = SlicedLayout<Root, Slice<0, 1, 2>, Slice<1, 2, 3>>;

    REQUIRE(Block::logical_dim(0) == 2);
    REQUIRE(Block::logical_dim(1) == 3);
    REQUIRE(Block::Base == 1 * 8 + 2 * 1);
}

TEST_CASE("sliced_layout: 3D slice from the worked example",
          "[layout_adapters]")
{
    // [2,3,5] double, SimdWidth=4 → padded to [2,3,8] → Strides [24,8,1]
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 2, 3, 5>>;
    using View = SlicedLayout<Root, Slice<0, 1, 1>, Slice<2, 0, 2>>;

    REQUIRE(View::logical_dim(0) == 1);
    REQUIRE(View::logical_dim(1) == 3);
    REQUIRE(View::logical_dim(2) == 2);
    REQUIRE(View::stride(0) == 24);
    REQUIRE(View::stride(1) == 8);
    REQUIRE(View::stride(2) == 1);
    REQUIRE(View::Base == 24);
}

TEST_CASE("sliced_layout: column flat indices step by the row stride",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 4, 6>>;
    using Col3 = SlicedLayout<Root, Slice<1, 3, 1>>;

    REQUIRE(Col3::logical_flat_to_physical_flat(0) == 3);
    REQUIRE(Col3::logical_flat_to_physical_flat(1) == 11);
    REQUIRE(Col3::logical_flat_to_physical_flat(2) == 19);
    REQUIRE(Col3::logical_flat_to_physical_flat(3) == 27);
}

TEST_CASE("sliced_layout: coords map to the same source offsets",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 4, 6>>;
    using Block = SlicedLayout<Root, Slice<0, 1, 2>, Slice<1, 2, 3>>;

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(Block::logical_coords_to_physical_flat(i, j) ==
                    Root::logical_coords_to_physical_flat(1 + i, 2 + j));
        }
    }
}

TEST_CASE("sliced_layout: contiguity",
          "[layout_adapters]")
{
    using M = StridedLayout<SimdPaddingPolicyBase<double, 4, 4, 6>>;
    using T3 = StridedLayout<SimdPaddingPolicyBase<double, 4, 2, 3, 4>>;

    REQUIRE(SlicedLayout<M, Slice<0, 2, 1>>::is_contiguous());        // a row of a padded matrix
    REQUIRE_FALSE(SlicedLayout<M, Slice<1, 3, 1>>::is_contiguous());  // a column
    REQUIRE(SlicedLayout<T3, Slice<0, 1, 1>>::is_contiguous());       // a plane on the outer axis
    REQUIRE_FALSE(SlicedLayout<T3, Slice<1, 2, 1>>::is_contiguous()); // a plane on the middle axis
}

TEST_CASE("sliced_layout: slice order does not matter",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 4, 6>>;
    using AB = SlicedLayout<Root, Slice<0, 1, 2>, Slice<1, 2, 3>>;
    using BA = SlicedLayout<Root, Slice<1, 2, 3>, Slice<0, 1, 2>>;

    REQUIRE(AB::Base == BA::Base);
    for (my_size_t i = 0; i < 2; ++i)
    {
        REQUIRE(AB::logical_dim(i) == BA::logical_dim(i));
        REQUIRE(AB::stride(i) == BA::stride(i));
    }
}

TEST_CASE("sliced_layout: float source pads to 8",
          "[layout_adapters]")
{
    // [4,6] float, SimdWidth=8 → padded to [4,8]
    using Root = StridedLayout<SimdPaddingPolicyBase<float, 8, 4, 6>>;
    using Col3 = SlicedLayout<Root, Slice<1, 3, 1>>;

    REQUIRE(Col3::stride(0) == 8);
    REQUIRE(Col3::Base == 3);
}

// ============================================================================
// SQUEEZE
// ============================================================================

TEST_CASE("squeezed_layout: leading axis dropped",
          "[layout_adapters]")
{
    // [1,3,4] double, SimdWidth=4 → Strides [12,4,1]
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 3, 4>>;
    using Sq = SqueezedLayout<Root, 0>;

    REQUIRE(Sq::NumDims == 2);
    REQUIRE(Sq::logical_dim(0) == 3);
    REQUIRE(Sq::logical_dim(1) == 4);
    REQUIRE(Sq::stride(0) == 4);
    REQUIRE(Sq::stride(1) == 1);
    REQUIRE(Sq::Base == 0);
    REQUIRE(Sq::LogicalSize == 12);
}

TEST_CASE("squeezed_layout: middle and leading axes",
          "[layout_adapters]")
{
    using Middle = SqueezedLayout<StridedLayout<SimdPaddingPolicyBase<double, 4, 2, 1, 4>>, 1>;
    REQUIRE(Middle::logical_dim(0) == 2);
    REQUIRE(Middle::logical_dim(1) == 4);
    REQUIRE(Middle::stride(0) == 4);
    REQUIRE(Middle::stride(1) == 1);

    using TwoLeading = SqueezedLayout<StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 1, 2, 4>>, 0, 1>;
    REQUIRE(TwoLeading::NumDims == 2);
    REQUIRE(TwoLeading::logical_dim(0) == 2);
    REQUIRE(TwoLeading::logical_dim(1) == 4);
    REQUIRE(TwoLeading::stride(0) == 4);
}

TEST_CASE("squeezed_layout: flat index maps identically to the source",
          "[layout_adapters]")
{
    // A size-1 axis contributes a coordinate that is always 0, so dropping it
    // changes neither the flat ordering nor any offset. Padded on purpose.
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 2, 5>>;
    using Sq = SqueezedLayout<Root, 0>;

    REQUIRE(Sq::LogicalSize == Root::LogicalSize);
    for (my_size_t n = 0; n < Sq::LogicalSize; ++n)
    {
        REQUIRE(Sq::logical_flat_to_physical_flat(n) == Root::logical_flat_to_physical_flat(n));
    }

    // flat 5 → squeezed coords (1,0) → 1*8 = 8
    REQUIRE(Sq::logical_flat_to_physical_flat(5) == 8);
}

TEST_CASE("squeezed_layout: squeezing a slice keeps the slice's base",
          "[layout_adapters]")
{
    // [3,5,2,4] double, SimdWidth=4 → Strides [40,8,4,1]. The corner at
    // (1,2) starts at 1*40 + 2*8 = 56.
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 5, 2, 4>>;
    using Corner = SlicedLayout<Root, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using Sq = SqueezedLayout<Corner, 0, 1>;

    REQUIRE(Corner::Base == 56);
    REQUIRE(Sq::Base == 56);
    REQUIRE(Sq::logical_dim(0) == 2);
    REQUIRE(Sq::logical_dim(1) == 4);
    REQUIRE(Sq::is_contiguous());

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(Sq::logical_coords_to_physical_flat(i, j) ==
                    Root::logical_coords_to_physical_flat(1, 2, i, j));
        }
    }
}

TEST_CASE("squeezed_layout: a padded trailing axis can be dropped",
          "[layout_adapters]")
{
    // [2,3,1] double, SimdWidth=4 → padded to [2,3,4] → Strides [12,4,1].
    // Dropping axis 2 leaves Strides [12,4]: a strided layout, which explicit
    // strides describe directly.
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 2, 3, 1>>;
    using Sq = SqueezedLayout<Root, 2>;

    REQUIRE(Sq::logical_dim(0) == 2);
    REQUIRE(Sq::logical_dim(1) == 3);
    REQUIRE(Sq::stride(0) == 12);
    REQUIRE(Sq::stride(1) == 4);
    REQUIRE_FALSE(Sq::is_contiguous());

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(Sq::logical_coords_to_physical_flat(i, j) ==
                    Root::logical_coords_to_physical_flat(i, j, 0));
        }
    }
}

TEST_CASE("squeezed_layout: squeezing a transpose",
          "[layout_adapters]")
{
    // [1,3,4] → permuted to [4,3,1] → axis 2 dropped → [4,3], Strides [1,4]
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 3, 4>>;
    using Sq = SqueezedLayout<PermutedLayout<Root, 2, 1, 0>, 2>;

    REQUIRE(Sq::logical_dim(0) == 4);
    REQUIRE(Sq::logical_dim(1) == 3);
    REQUIRE(Sq::stride(0) == 1);
    REQUIRE(Sq::stride(1) == 4);

    for (my_size_t i = 0; i < 4; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(Sq::logical_coords_to_physical_flat(i, j) ==
                    Root::logical_coords_to_physical_flat(0, j, i));
        }
    }
}

TEST_CASE("squeezed_layout: drop order does not matter",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 3, 1, 4>>;
    using Forward = SqueezedLayout<Root, 0, 2>;
    using Reversed = SqueezedLayout<Root, 2, 0>;

    for (my_size_t i = 0; i < 2; ++i)
    {
        REQUIRE(Forward::logical_dim(i) == Reversed::logical_dim(i));
        REQUIRE(Forward::stride(i) == Reversed::stride(i));
    }
}

TEST_CASE("squeezed_layout: dropping down to the rank floor is accepted",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 1, 1, 1, 5, 1>>;
    using Sq = SqueezedLayout<Root, 0, 1>;

    REQUIRE(Sq::NumDims == 2);
    REQUIRE(Sq::logical_dim(0) == 5);
    REQUIRE(Sq::logical_dim(1) == 1);
}

TEST_CASE("squeezed_layout: 5D source down to 3D",
          "[layout_adapters]")
{
    // [1,2,1,3,5] double, SimdWidth=4 → padded to [1,2,1,3,8] → Strides [48,24,24,8,1]
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 2, 1, 3, 5>>;
    using Sq = SqueezedLayout<Root, 0, 2>;

    REQUIRE(Sq::NumDims == 3);
    REQUIRE(Sq::stride(0) == 24);
    REQUIRE(Sq::stride(1) == 8);
    REQUIRE(Sq::stride(2) == 1);

    for (my_size_t n = 0; n < Sq::LogicalSize; ++n)
    {
        REQUIRE(Sq::logical_flat_to_physical_flat(n) == Root::logical_flat_to_physical_flat(n));
    }
}

// ============================================================================
// PERMUTE
// ============================================================================

TEST_CASE("permuted_layout: transpose of a padded matrix",
          "[layout_adapters]")
{
    // [2,3] double, SimdWidth=4 → padded to [2,4] → Strides [4,1]
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 2, 3>>;
    using T = PermutedLayout<Root, 1, 0>;

    REQUIRE(T::logical_dim(0) == 3);
    REQUIRE(T::logical_dim(1) == 2);
    REQUIRE(T::stride(0) == 1);
    REQUIRE(T::stride(1) == 4);
    REQUIRE(T::Base == 0);
    REQUIRE(T::logical_coords_to_physical_flat(0, 1) == 4);
    REQUIRE(T::logical_coords_to_physical_flat(2, 1) == 6);
    REQUIRE_FALSE(T::is_contiguous());
}

TEST_CASE("permuted_layout: 3D permutation",
          "[layout_adapters]")
{
    // [2,3,4] double, SimdWidth=4 → Strides [12,4,1]
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 2, 3, 4>>;
    using P = PermutedLayout<Root, 2, 0, 1>;

    REQUIRE(P::logical_dim(0) == 4);
    REQUIRE(P::logical_dim(1) == 2);
    REQUIRE(P::logical_dim(2) == 3);
    REQUIRE(P::stride(0) == 1);
    REQUIRE(P::stride(1) == 12);
    REQUIRE(P::stride(2) == 4);
}

TEST_CASE("permuted_layout: transposing a block keeps its base",
          "[layout_adapters]")
{
    // [3,3] double, SimdWidth=4 → Strides [4,1]; the block at (1,1) starts at 5
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 3>>;
    using Block = SlicedLayout<Root, Slice<0, 1, 2>, Slice<1, 1, 2>>;
    using T = PermutedLayout<Block, 1, 0>;

    REQUIRE(T::Base == 5);
    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 2; ++j)
        {
            REQUIRE(T::logical_coords_to_physical_flat(i, j) ==
                    Root::logical_coords_to_physical_flat(1 + j, 1 + i));
        }
    }
}

TEST_CASE("permuted_layout: transposing twice gives the source back",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 2, 3>>;
    using TT = PermutedLayout<PermutedLayout<Root, 1, 0>, 1, 0>;

    for (my_size_t i = 0; i < 2; ++i)
    {
        REQUIRE(TT::logical_dim(i) == Root::logical_dim(i));
        REQUIRE(TT::stride(i) == Root::stride(i));
    }
    REQUIRE(TT::Base == Root::Base);
}

TEST_CASE("permuted_layout: LogicalDims with transpose permutation", "[layout][dims][transpose][layout_adapters]")
{
    /*
     * 2x3 matrix transposed via [1,0]
     *
     * Original (2x3):         Transposed view (3x2):
     *   [0,0] [0,1] [0,2]       [0,0] [0,1]
     *   [1,0] [1,1] [1,2]       [1,0] [1,1]
     *                          [2,0] [2,1]
     *
     * Policy::logical_dim = [2, 3]
     * Layout::logical_dim = [3, 2]  (permuted)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    REQUIRE(Layout::logical_dim(0) == 3); // was dim 1
    REQUIRE(Layout::logical_dim(1) == 2); // was dim 0
}

TEST_CASE("permuted_layout: LogicalDims with 3D permutation", "[layout][dims][3d][layout_adapters]")
{
    /*
     * 2x3x4 tensor with permutation [2,0,1]
     *
     * Policy::logical_dim = [2, 3, 4]
     * Permutation [2,0,1] means:
     *   new dim 0 = old dim 2 = 4
     *   new dim 1 = old dim 0 = 2
     *   new dim 2 = old dim 1 = 3
     *
     * Layout::logical_dim = [4, 2, 3]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 2, 0, 1>;

    REQUIRE(Layout::logical_dim(0) == 4);
    REQUIRE(Layout::logical_dim(1) == 2);
    REQUIRE(Layout::logical_dim(2) == 3);
}

TEST_CASE("permuted_layout: PermutedLayout reorders the root's strides", "[layout][strides][permuted][layout_adapters]")
{
    /*
     * 2x3 matrix (no padding) with transpose [1,0]
     *
     * Physical memory (unchanged):
     *   flat:   0    1    2   |   3    4    5
     *         [0,0][0,1][0,2] | [1,0][1,1][1,2]
     *
     * Root strides = [3, 1]
     *
     * Transposed view: logical (i,j) accesses physical (j,i)
     * Strides = [Root[1], Root[0]] = [1, 3]
     *
     * Example: logical (1,0) in transposed view
     *   = 1 * Strides[0] + 0 * Strides[1]
     *   = 1 * 1 + 0 * 3 = 1
     *   = physical element at position 1 = original [0,1] ✓
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    REQUIRE(Root::stride(0) == 3);
    REQUIRE(Root::stride(1) == 1);

    REQUIRE(Layout::stride(0) == 1); // Root::stride(1)
    REQUIRE(Layout::stride(1) == 3); // Root::stride(0)
}

TEST_CASE("permuted_layout: LogicalStrides are row-major from LogicalDims", "[layout][strides][logical][layout_adapters]")
{
    /*
     * 2x3 matrix transposed to 3x2 view
     * LogicalDims = [3, 2]
     *
     * LogicalStrides = [2, 1]
     *   Used for decomposing logical flat indices
     *
     * Example: logical flat 4 in 3x2 view
     *   4 / 2 = 2 (row), 4 % 2 = 0 (col)
     *   logical coords = (2, 0)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    REQUIRE(Layout::logical_dim(0) == 3);
    REQUIRE(Layout::logical_dim(1) == 2);

    REQUIRE(Layout::logical_stride(0) == 2);
    REQUIRE(Layout::logical_stride(1) == 1);
}

TEST_CASE("permuted_layout: logical_dim() returns permuted dimensions", "[layout][query][layout_adapters]")
{
    /*
     * 2x3x4 tensor with permutation [2,0,1]
     * LogicalDims = [4, 2, 3]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 2, 0, 1>;

    REQUIRE(Layout::logical_dim(0) == 4);
    REQUIRE(Layout::logical_dim(1) == 2);
    REQUIRE(Layout::logical_dim(2) == 3);
    CHECK_THROWS(Layout::logical_dim(3)); // out-of-bounds
}

TEST_CASE("permuted_layout: stride() returns permuted strides", "[layout][query][layout_adapters]")
{
    /*
     * 2x3 matrix transposed
     * Root strides = [3, 1]
     * Strides = [1, 3]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    REQUIRE(Layout::stride(0) == 1);
    REQUIRE(Layout::stride(1) == 3);
    CHECK_THROWS(Layout::stride(2)); // out-of-bounds
}

TEST_CASE("permuted_layout: logical_coords_to_physical_flat: transposed", "[layout][index][l2p][transpose][layout_adapters]")
{
    /*
     * 2x3 matrix transposed via [1,0] to 3x2 view, no padding
     *
     * Original physical memory:    Transposed logical view (3x2):
     *   flat: 0  1  2                 [0,0]=0  [0,1]=3
     *         3  4  5                 [1,0]=1  [1,1]=4
     *                                 [2,0]=2  [2,1]=5
     *
     * Strides = [1, 3]
     * logical (i,j) -> physical i*1 + j*3
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    // First column of transposed view = first row of original
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 0) == 2);

    // Second column of transposed view = second row of original
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1) == 3);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 1) == 5);
}

TEST_CASE("permuted_layout: logical_coords_to_physical_flat: transposed with padding", "[layout][index][l2p][transpose][padding][layout_adapters]")
{
    /*
     * 2x3 matrix with AVX padding, transposed to 3x2 view
     *
     * Physical memory:             Transposed logical view (3x2):
     *   flat: 0  1  2  [P]            [0,0]=0  [0,1]=4
     *         4  5  6  [P]            [1,0]=1  [1,1]=5
     *                                 [2,0]=2  [2,1]=6
     *
     * Root strides = [4, 1]
     * Strides = [1, 4]  (permuted)
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 0) == 2);

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1) == 5);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 1) == 6);

    CHECK_THROWS(Layout::logical_coords_to_physical_flat(3, 0)); // out-of-bounds
    CHECK_THROWS(Layout::logical_coords_to_physical_flat(0, 2)); // out-of-bounds
}

TEST_CASE("permuted_layout: logical_coords_to_physical_flat: 3D padding and permutation", "[layout][index][l2p][3d][padding][transpose][layout_adapters]")
{
    /*
     * Original tensor: 2x3x5 with AVX padding (SW=4)
     * Permutation: [2, 0, 1] creates a 5x2x3 logical view
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │ PHYSICAL MEMORY LAYOUT (unchanged by permutation)                   │
     * │                                                                     │
     * │ Policy::logical_dim  = [2, 3, 5]                                    │
     * │ Policy::PhysicalDims = [2, 3, 8]  (last dim padded: 5 → 8)          │
     * │ Root strides = [24, 8, 1]                                           │
     * │                                                                     │
     * │ Slice i=0 (physical coords [0, j, k]):                              │
     * │   j=0: |  0   1   2   3   4  [P] [P] [P] |                          │
     * │   j=1: |  8   9  10  11  12  [P] [P] [P] |                          │
     * │   j=2: | 16  17  18  19  20  [P] [P] [P] |                          │
     * │                                                                     │
     * │ Slice i=1 (physical coords [1, j, k]):                              │
     * │   j=0: | 24  25  26  27  28  [P] [P] [P] |                          │
     * │   j=1: | 32  33  34  35  36  [P] [P] [P] |                          │
     * │   j=2: | 40  41  42  43  44  [P] [P] [P] |                          │
     * │                                                                     │
     * │ [P] = padding element                                               │
     * └─────────────────────────────────────────────────────────────────────┘
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │ PERMUTATION [2, 0, 1] MEANING                                       │
     * │                                                                     │
     * │ Logical dim 0 → Physical dim 2 (size 5)                             │
     * │ Logical dim 1 → Physical dim 0 (size 2)                             │
     * │ Logical dim 2 → Physical dim 1 (size 3)                             │
     * │                                                                     │
     * │ Layout::logical_dim = [5, 2, 3]                                     │
     * │ Strides = [Root[2], Root[0], Root[1]]                               │
     * │         = [1, 24, 8]                                                │
     * └─────────────────────────────────────────────────────────────────────┘
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │ LOGICAL VIEW (5x2x3) - same data, different access pattern          │
     * │                                                                     │
     * │ logical(i, j, k) accesses physical(j, k, i)                         │
     * │                                                                     │
     * │ Examples:                                                           │
     * │   logical(0,0,0) → physical(0,0,0) → flat 0                         │
     * │   logical(1,0,0) → physical(0,0,1) → flat 1                         │
     * │   logical(0,1,0) → physical(1,0,0) → flat 24                        │
     * │   logical(0,0,1) → physical(0,1,0) → flat 8                         │
     * │   logical(0,0,2) → physical(0,2,0) → flat 16                        │
     * │   logical(4,1,2) → physical(1,2,4) → flat 44                        │
     * │                                                                     │
     * │ Formula: flat = i*1 + j*24 + k*8                                    │
     * └─────────────────────────────────────────────────────────────────────┘
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 5>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 2, 0, 1>;

    // Verify dimensions and strides
    REQUIRE(Layout::logical_dim(0) == 5);
    REQUIRE(Layout::logical_dim(1) == 2);
    REQUIRE(Layout::logical_dim(2) == 3);

    REQUIRE(Root::stride(0) == 24);
    REQUIRE(Root::stride(1) == 8);
    REQUIRE(Root::stride(2) == 1);

    REQUIRE(Layout::stride(0) == 1);
    REQUIRE(Layout::stride(1) == 24);
    REQUIRE(Layout::stride(2) == 8);

    // Corner cases
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);  // origin
    REQUIRE(Layout::logical_coords_to_physical_flat(4, 1, 2) == 44); // last valid element

    // Varying dim 0 (stride 1): consecutive in memory
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0, 0) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 0, 0) == 2);
    REQUIRE(Layout::logical_coords_to_physical_flat(3, 0, 0) == 3);
    REQUIRE(Layout::logical_coords_to_physical_flat(4, 0, 0) == 4);

    // Varying dim 1 (stride 24): jumps between physical slices
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1, 0) == 24);

    // Varying dim 2 (stride 8): jumps between rows within a slice
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 1) == 8);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 2) == 16);

    // Mixed coordinates
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 1, 1) == 2 + 24 + 8);  // = 34
    REQUIRE(Layout::logical_coords_to_physical_flat(3, 0, 2) == 3 + 0 + 16);  // = 19
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1, 2) == 1 + 24 + 16); // = 41
}

TEST_CASE("permuted_layout: logical_flat_to_physical_flat: transposed", "[layout][index][lf2pf][transpose][layout_adapters]")
{
    /*
     * Original: 2x3 matrix    Transposed view: 3x2
     * Permutation: [1, 0]
     *
     * Physical memory stores values [A, B, C, D, E, F]:
     *
     *       col0 col1 col2
     *      ┌────┬────┬────┐
     * row0 │ A  │ B  │ C  │     Memory: [A][B][C][D][E][F]
     * row1 │ D  │ E  │ F  │              0  1  2  3  4  5
     *      └────┴────┴────┘
     *
     * Transposed view (3x2) sees the SAME data as:
     *
     *       col0 col1
     *      ┌────┬────┐
     * row0 │ A  │ D  │     (original col 0 becomes row 0)
     * row1 │ B  │ E  │     (original col 1 becomes row 1)
     * row2 │ C  │ F  │     (original col 2 becomes row 2)
     *      └────┴────┘
     *
     * Iterating row-major through transposed view:
     *   A → D → B → E → C → F  (logical flat: 0,1,2,3,4,5)
     *
     * Physical locations of those values:
     *   A=0, D=3, B=1, E=4, C=2, F=5
     *
     * So: logical_flat → physical_flat
     *   0 (A) → 0
     *   1 (D) → 3
     *   2 (B) → 1
     *   3 (E) → 4
     *   4 (C) → 2
     *   5 (F) → 5
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 3);
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 1);
    REQUIRE(Layout::logical_flat_to_physical_flat(3) == 4);
    REQUIRE(Layout::logical_flat_to_physical_flat(4) == 2);
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 5);
    CHECK_THROWS(Layout::logical_flat_to_physical_flat(6)); // out-of-bounds
}

TEST_CASE("permuted_layout: logical_flat_to_physical_flat: transposed with padding", "[layout][index][lf2pf][transpose][padding][layout_adapters]")
{
    /*
     * 2x3 matrix with padding, transposed to 3x2 view
     *
     * Physical memory: 0 1 2 [P] 4 5 6 [P]
     *
     * Transposed logical view (3x2):
     *   [0,0]=0  [0,1]=4
     *   [1,0]=1  [1,1]=5
     *   [2,0]=2  [2,1]=6
     *
     * LogicalStrides = [2, 1]
     * Strides = [1, 4]
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 4);
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 1);
    REQUIRE(Layout::logical_flat_to_physical_flat(3) == 5);
    REQUIRE(Layout::logical_flat_to_physical_flat(4) == 2);
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 6);
    CHECK_THROWS(Layout::logical_flat_to_physical_flat(6));   // out-of-bounds
    CHECK_THROWS(Layout::logical_flat_to_physical_flat(100)); // out-of-bounds
    CHECK_THROWS(Layout::logical_flat_to_physical_flat(-10)); // out-of-bounds
}

TEST_CASE("permuted_layout: logical_flat_to_logical_coords: transposed view", "[layout][index][lf2lc][transpose][layout_adapters]")
{
    /*
     * 2x3 matrix transposed to 3x2 view
     * LogicalDims = [3, 2]
     *
     * Logical flat in transposed view:
     *   0  1          coords: (0,0)(0,1)
     *   2  3                  (1,0)(1,1)
     *   4  5                  (2,0)(2,1)
     *
     * LogicalStrides = [2, 1]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;
    my_size_t coords[2];

    Layout::logical_flat_to_logical_coords(0, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 0);

    Layout::logical_flat_to_logical_coords(1, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 1);

    Layout::logical_flat_to_logical_coords(2, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 0);

    Layout::logical_flat_to_logical_coords(3, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 1);

    Layout::logical_flat_to_logical_coords(4, coords);
    REQUIRE(coords[0] == 2);
    REQUIRE(coords[1] == 0);

    Layout::logical_flat_to_logical_coords(5, coords);
    REQUIRE(coords[0] == 2);
    REQUIRE(coords[1] == 1);

    CHECK_THROWS(Layout::logical_flat_to_logical_coords(6, coords)); // out-of-bounds
}

TEST_CASE("permuted_layout: physical_flat_to_logical_coords: transposed", "[layout][index][pf2lc][transpose][layout_adapters]")
{
    /*
     * 2x3 matrix transposed via [1,0]
     *
     * Physical memory (unchanged):
     *   flat: 0  1  2
     *         3  4  5
     *   physical coords: (0,0)(0,1)(0,2)
     *                    (1,0)(1,1)(1,2)
     *
     * Transposed logical view (3x2):
     *   The element at physical (0,1) appears at logical (1,0)
     *   The element at physical (1,0) appears at logical (0,1)
     *
     * Perm = [1, 0]
     * logical[i] = physical[Perm[i]]
     *
     * physical flat 1 -> physical coords (0,1)
     *   logical[0] = physical[Perm[0]] = physical[1] = 1
     *   logical[1] = physical[Perm[1]] = physical[0] = 0
     *   logical coords = (1, 0)
     *
     * physical flat 3 -> physical coords (1,0)
     *   logical[0] = physical[1] = 0
     *   logical[1] = physical[0] = 1
     *   logical coords = (0, 1)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;
    my_size_t coords[2];

    // physical (0,0) -> logical (0,0)
    Layout::physical_flat_to_logical_coords(0, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 0);

    // physical (0,1) -> logical (1,0)
    Layout::physical_flat_to_logical_coords(1, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 0);

    // physical (0,2) -> logical (2,0)
    Layout::physical_flat_to_logical_coords(2, coords);
    REQUIRE(coords[0] == 2);
    REQUIRE(coords[1] == 0);

    // physical (1,0) -> logical (0,1)
    Layout::physical_flat_to_logical_coords(3, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 1);

    // physical (1,1) -> logical (1,1)
    Layout::physical_flat_to_logical_coords(4, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 1);

    // physical (1,2) -> logical (2,1)
    Layout::physical_flat_to_logical_coords(5, coords);
    REQUIRE(coords[0] == 2);
    REQUIRE(coords[1] == 1);

    CHECK_THROWS(Layout::physical_flat_to_logical_coords(6, coords)); // out-of-bounds
}

TEST_CASE("permuted_layout: physical_flat_to_logical_coords: 3D with permutation", "[layout][index][pf2lc][3d][layout_adapters]")
{
    /*
     * Original: 2x3x4 tensor
     * Permutation: [2, 0, 1] creates a 4x2x3 logical view
     *
     * Physical memory layout (2 slices of 3x4):
     *
     *   Slice i=0:                    Slice i=1:
     *     j\k   0   1   2   3           j\k   0   1   2   3
     *      0 [  0   1   2   3 ]          0 [ 12  13  14  15 ]
     *      1 [  4   5   6   7 ]          1 [ 16  17  18  19 ]
     *      2 [  8   9  10  11 ]          2 [ 20  21  22  23 ]
     *
     *   Root strides = [12, 4, 1]
     *   physical(i,j,k) = i*12 + j*4 + k
     *
     * Permutation [2, 0, 1] means:
     *   logical dim 0 → physical dim 2 (size 4)
     *   logical dim 1 → physical dim 0 (size 2)
     *   logical dim 2 → physical dim 1 (size 3)
     *
     * Reverse mapping (physical coords → logical coords):
     *   logical[0] = physical[Perm[0]] = physical[2]
     *   logical[1] = physical[Perm[1]] = physical[0]
     *   logical[2] = physical[Perm[2]] = physical[1]
     *
     * Examples:
     *   physical flat 0  → physical(0,0,0) → logical(0,0,0)
     *   physical flat 17 → physical(1,1,1) → logical(1,1,1)
     *   physical flat 23 → physical(1,2,3) → logical(3,1,2)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 2, 0, 1>;
    my_size_t coords[3];

    // physical flat 0 -> physical (0,0,0) -> logical (0,0,0)
    Layout::physical_flat_to_logical_coords(0, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 0);
    REQUIRE(coords[2] == 0);

    // physical flat 17 -> physical (1,1,1) -> logical (1,1,1)
    Layout::physical_flat_to_logical_coords(17, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 1);
    REQUIRE(coords[2] == 1);

    // physical flat 23 -> physical (1,2,3) -> logical (3,1,2)
    Layout::physical_flat_to_logical_coords(23, coords);
    REQUIRE(coords[0] == 3);
    REQUIRE(coords[1] == 1);
    REQUIRE(coords[2] == 2);
}

TEST_CASE("permuted_layout: Roundtrip: logical_flat -> logical_coords -> physical_flat -> logical_coords -> physical_flat", "[layout][roundtrip][layout_adapters]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 4>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 2, 0>;

    for (my_size_t lf = 0; lf < Layout::LogicalSize; ++lf)
    {
        my_size_t logical_coords[3];
        Layout::logical_flat_to_logical_coords(lf, logical_coords);

        my_size_t pf = Layout::logical_coords_to_physical_flat(logical_coords);

        my_size_t logical_coords_roundtrip[3];
        Layout::physical_flat_to_logical_coords(pf, logical_coords_roundtrip);

        my_size_t pf_roundtrip = Layout::logical_coords_to_physical_flat(logical_coords_roundtrip);

        REQUIRE(logical_coords[0] == logical_coords_roundtrip[0]);
        REQUIRE(logical_coords[1] == logical_coords_roundtrip[1]);
        REQUIRE(logical_coords[2] == logical_coords_roundtrip[2]);
        REQUIRE(pf == pf_roundtrip);
    }
}

TEST_CASE("permuted_layout: is_logical_index_in_bounds: respects permutation", "[layout][bounds][perm][layout_adapters]")
{
    /*
     * 2x3 transposed to 3x2
     * LogicalDims = [3, 2]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    my_size_t valid[] = {2, 1};   // valid for 3x2
    my_size_t invalid[] = {1, 2}; // col=2 out of bounds for 3x2

    REQUIRE(Layout::is_logical_index_in_bounds(valid));
    REQUIRE_FALSE(Layout::is_logical_index_in_bounds(invalid));
}

TEST_CASE("permuted_layout: All members are constexpr", "[layout][constexpr][layout_adapters]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    // Use in array sizes (compile-time requirement)
    [[maybe_unused]] double arr1[Layout::NumDims];
    [[maybe_unused]] double arr2[Layout::LogicalSize];
    [[maybe_unused]] double arr3[Root::PhysicalSize];

    // Static assertions
    static_assert(Layout::NumDims == 2);
    static_assert(Layout::LogicalSize == 6);
    static_assert(Root::PhysicalSize == 8);

    static_assert(Layout::logical_dim(0) == 3);
    static_assert(Layout::logical_dim(1) == 2);

    static_assert(Root::stride(0) == 4);
    static_assert(Root::stride(1) == 1);

    static_assert(Layout::stride(0) == 1);
    static_assert(Layout::stride(1) == 4);

    SUCCEED("All constexpr usages compiled successfully");
}

TEST_CASE("permuted_layout: All 3D permutations", "[layout][perm][3d][layout_adapters]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 4>;

    SECTION("[0,1,2] identity")
    {
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 0, 1, 2>;
        REQUIRE(Layout::logical_dim(0) == 2);
        REQUIRE(Layout::logical_dim(1) == 3);
        REQUIRE(Layout::logical_dim(2) == 4);
    }

    SECTION("[0,2,1]")
    {
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 0, 2, 1>;
        REQUIRE(Layout::logical_dim(0) == 2);
        REQUIRE(Layout::logical_dim(1) == 4);
        REQUIRE(Layout::logical_dim(2) == 3);
    }

    SECTION("[1,0,2]")
    {
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0, 2>;
        REQUIRE(Layout::logical_dim(0) == 3);
        REQUIRE(Layout::logical_dim(1) == 2);
        REQUIRE(Layout::logical_dim(2) == 4);
    }

    SECTION("[1,2,0]")
    {
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 2, 0>;
        REQUIRE(Layout::logical_dim(0) == 3);
        REQUIRE(Layout::logical_dim(1) == 4);
        REQUIRE(Layout::logical_dim(2) == 2);
    }

    SECTION("[2,0,1]")
    {
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 2, 0, 1>;
        REQUIRE(Layout::logical_dim(0) == 4);
        REQUIRE(Layout::logical_dim(1) == 2);
        REQUIRE(Layout::logical_dim(2) == 3);
    }

    SECTION("[2,1,0] reverse")
    {
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 2, 1, 0>;
        REQUIRE(Layout::logical_dim(0) == 4);
        REQUIRE(Layout::logical_dim(1) == 3);
        REQUIRE(Layout::logical_dim(2) == 2);
    }
}

TEST_CASE("permuted_layout: Layout works with all SIMD widths", "[layout][simd][layout_adapters]")
{
    /*
     * 3x5 matrix transposed to 5x3 view
     * Last dim (5) gets padded based on SIMD width
     *
     * Physical memory (3 rows, padded columns):
     *   row0: [0] [1] [2] [3] [4] [P]...
     *   row1: [S] ...
     *   row2: [2S] ...
     *   where S = padded stride, P = padding
     */

    SECTION("SCALAR (SW=1) - no padding")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 3, 5>;
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Root::PhysicalSize == 15);
        REQUIRE(Layout::logical_dim(0) == 5);
        REQUIRE(Layout::logical_dim(1) == 3);
        REQUIRE(Root::stride(0) == 5);
        REQUIRE(Root::stride(1) == 1);

        // Transposed: logical(i,j) -> physical j*5 + i
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);   // (0,0) -> 0
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 5);   // (0,1) -> 5
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 10);  // (0,2) -> 10
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);   // (1,0) -> 1
        REQUIRE(Layout::logical_flat_to_physical_flat(14) == 14); // (4,2) -> 14
    }

    SECTION("SSE_DOUBLE (SW=2) - padded to 6")
    {
        using Policy = SimdPaddingPolicyBase<double, SSE_DOUBLE, 3, 5>;
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Root::PhysicalSize == 18); // 3 * 6
        REQUIRE(Root::stride(0) == 6);
        REQUIRE(Root::stride(1) == 1);

        // Transposed: logical(i,j) -> physical j*6 + i
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);   // (0,0) -> 0
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 6);   // (0,1) -> 6
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 12);  // (0,2) -> 12
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);   // (1,0) -> 1
        REQUIRE(Layout::logical_flat_to_physical_flat(14) == 16); // (4,2) -> 16
    }

    SECTION("SSE_FLOAT (SW=4) - padded to 8")
    {
        using Policy = SimdPaddingPolicyBase<float, SSE_FLOAT, 3, 5>;
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Root::PhysicalSize == 24); // 3 * 8
        REQUIRE(Root::stride(0) == 8);
        REQUIRE(Root::stride(1) == 1);

        // Transposed: logical(i,j) -> physical j*8 + i
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);   // (0,0) -> 0
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 8);   // (0,1) -> 8
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 16);  // (0,2) -> 16
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);   // (1,0) -> 1
        REQUIRE(Layout::logical_flat_to_physical_flat(14) == 20); // (4,2) -> 20
    }

    SECTION("AVX_DOUBLE (SW=4) - padded to 8")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 3, 5>;
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Root::PhysicalSize == 24); // 3 * 8
        REQUIRE(Root::stride(0) == 8);
        REQUIRE(Root::stride(1) == 1);

        // Transposed: logical(i,j) -> physical j*8 + i
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);   // (0,0) -> 0
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 8);   // (0,1) -> 8
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 16);  // (0,2) -> 16
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);   // (1,0) -> 1
        REQUIRE(Layout::logical_flat_to_physical_flat(14) == 20); // (4,2) -> 20
    }

    SECTION("AVX_FLOAT (SW=8) - padded to 8")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX_FLOAT, 3, 5>;
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Root::PhysicalSize == 24); // 3 * 8
        REQUIRE(Root::stride(0) == 8);
        REQUIRE(Root::stride(1) == 1);

        // Transposed: logical(i,j) -> physical j*8 + i
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);   // (0,0) -> 0
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 8);   // (0,1) -> 8
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 16);  // (0,2) -> 16
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);   // (1,0) -> 1
        REQUIRE(Layout::logical_flat_to_physical_flat(14) == 20); // (4,2) -> 20
    }

    SECTION("AVX512_DOUBLE (SW=8) - padded to 8")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX512_DOUBLE, 3, 5>;
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Root::PhysicalSize == 24); // 3 * 8
        REQUIRE(Root::stride(0) == 8);
        REQUIRE(Root::stride(1) == 1);

        // Transposed: logical(i,j) -> physical j*8 + i
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);   // (0,0) -> 0
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 8);   // (0,1) -> 8
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 16);  // (0,2) -> 16
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);   // (1,0) -> 1
        REQUIRE(Layout::logical_flat_to_physical_flat(14) == 20); // (4,2) -> 20
    }

    SECTION("AVX512_FLOAT (SW=16) - padded to 16")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX512_FLOAT, 3, 5>;
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Root::PhysicalSize == 48); // 3 * 16
        REQUIRE(Root::stride(0) == 16);
        REQUIRE(Root::stride(1) == 1);

        // Transposed: logical(i,j) -> physical j*16 + i
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);   // (0,0) -> 0
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 16);  // (0,1) -> 16
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 32);  // (0,2) -> 32
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);   // (1,0) -> 1
        REQUIRE(Layout::logical_flat_to_physical_flat(14) == 36); // (4,2) -> 36
    }
}

TEST_CASE("permuted_layout: NoPaddingPolicy: 2D transposed", "[layout][nopadding][2d][transpose][layout_adapters]")
{
    /*
     * 2x3 matrix transposed to 3x2 view
     *
     * Physical memory:          Transposed view (3x2):
     *       col0 col1 col2            col0 col1
     *      ┌────┬────┬────┐          ┌────┬────┐
     * row0 │ A  │ B  │ C  │     row0 │ A  │ D  │
     * row1 │ D  │ E  │ F  │     row1 │ B  │ E  │
     *      └────┴────┴────┘     row2 │ C  │ F  │
     *                                └────┴────┘
     *
     * Iterating transposed view row-major: A, D, B, E, C, F
     * Physical locations:                  0, 3, 1, 4, 2, 5
     */
    using Policy = NoPaddingPolicy<double, 2, 3>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 0>;

    REQUIRE(Layout::LogicalSize == 6);
    REQUIRE(Root::PhysicalSize == 6);
    REQUIRE(Layout::logical_dim(0) == 3);
    REQUIRE(Layout::logical_dim(1) == 2);
    REQUIRE(Layout::stride(0) == 1);
    REQUIRE(Layout::stride(1) == 3);

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0); // A
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 3); // D
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 1); // B
    REQUIRE(Layout::logical_flat_to_physical_flat(3) == 4); // E
    REQUIRE(Layout::logical_flat_to_physical_flat(4) == 2); // C
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 5); // F
}

TEST_CASE("permuted_layout: NoPaddingPolicy: 3D with permutation", "[layout][nopadding][3d][transpose][layout_adapters]")
{
    /*
     * 2x3x4 tensor with permutation [2, 0, 1] creates 4x2x3 view
     *
     * Policy::logical_dim = [2, 3, 4]
     * Layout::logical_dim = [4, 2, 3]
     *
     * Root strides = [12, 4, 1]
     * Strides = [1, 12, 4]  (permuted)
     *
     * logical(i, j, k) -> physical(j, k, i)
     * flat = i*1 + j*12 + k*4
     */
    using Policy = NoPaddingPolicy<double, 2, 3, 4>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 2, 0, 1>;

    REQUIRE(Layout::LogicalSize == 24);
    REQUIRE(Root::PhysicalSize == 24);
    REQUIRE(Layout::logical_dim(0) == 4);
    REQUIRE(Layout::logical_dim(1) == 2);
    REQUIRE(Layout::logical_dim(2) == 3);
    REQUIRE(Layout::stride(0) == 1);
    REQUIRE(Layout::stride(1) == 12);
    REQUIRE(Layout::stride(2) == 4);

    // logical(0,0,0) -> 0
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
    // logical(1,0,0) -> 1 (stride 1 in dim 0)
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0, 0) == 1);
    // logical(0,1,0) -> 12 (stride 12 in dim 1)
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1, 0) == 12);
    // logical(0,0,1) -> 4 (stride 4 in dim 2)
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 1) == 4);
    // logical(3,1,2) -> 3 + 12 + 8 = 23
    REQUIRE(Layout::logical_coords_to_physical_flat(3, 1, 2) == 23);
}

TEST_CASE("permuted_layout: NoPaddingPolicy: roundtrip consistency", "[layout][nopadding][roundtrip][layout_adapters]")
{
    /*
     * Verify roundtrip works correctly with NoPaddingPolicy
     * 3x4x5 tensor with permutation [1, 2, 0] creates 4x5x3 view
     */
    using Policy = NoPaddingPolicy<double, 3, 4, 5>;
    using Root = StridedLayout<Policy>;
    using Layout = PermutedLayout<Root, 1, 2, 0>;

    REQUIRE(Layout::logical_dim(0) == 4);
    REQUIRE(Layout::logical_dim(1) == 5);
    REQUIRE(Layout::logical_dim(2) == 3);

    for (my_size_t lf = 0; lf < Layout::LogicalSize; ++lf)
    {
        my_size_t logical_coords[3];
        Layout::logical_flat_to_logical_coords(lf, logical_coords);

        my_size_t pf = Layout::logical_coords_to_physical_flat(logical_coords);

        my_size_t logical_coords_roundtrip[3];
        Layout::physical_flat_to_logical_coords(pf, logical_coords_roundtrip);

        REQUIRE(logical_coords[0] == logical_coords_roundtrip[0]);
        REQUIRE(logical_coords[1] == logical_coords_roundtrip[1]);
        REQUIRE(logical_coords[2] == logical_coords_roundtrip[2]);
    }
}

TEST_CASE("permuted_layout: Roundtrip: logical_coords -> physical_flat -> logical_coords", "[layout][roundtrip][layout_adapters]")
{
    /*
     * For all valid logical coords:
     *   logical_coords -> physical_flat -> logical_coords
     * should return original coords
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;

    SECTION("Transposed")
    {
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        for (my_size_t i = 0; i < Layout::logical_dim(0); ++i)
        {
            for (my_size_t j = 0; j < Layout::logical_dim(1); ++j)
            {
                my_size_t physical = Layout::logical_coords_to_physical_flat(i, j);
                my_size_t result[2];
                Layout::physical_flat_to_logical_coords(physical, result);

                REQUIRE(result[0] == i);
                REQUIRE(result[1] == j);
            }
        }
    }
}

TEST_CASE("permuted_layout: Roundtrip: logical_flat -> physical_flat consistency", "[layout][roundtrip][layout_adapters]")
{
    /*
     * logical_flat_to_physical_flat should equal:
     *   logical_flat -> logical_coords -> physical_flat
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 3, 4>;

    SECTION("Transposed")
    {
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        for (my_size_t lf = 0; lf < Layout::LogicalSize; ++lf)
        {
            my_size_t pf_direct = Layout::logical_flat_to_physical_flat(lf);

            my_size_t logical_coords[2];
            Layout::logical_flat_to_logical_coords(lf, logical_coords);
            my_size_t pf_indirect = Layout::logical_coords_to_physical_flat(logical_coords);

            REQUIRE(pf_direct == pf_indirect);
        }
    }
}

TEST_CASE("permuted_layout: All 2D permutations", "[layout][perm][2d][layout_adapters]")
{
    /*
     * 3x4 matrix with AVX padding (SW=4)
     * PhysicalDims = [3, 4] (4 already aligned)
     *
     * Physical memory:
     *       col0 col1 col2 col3
     *      ┌────┬────┬────┬────┐
     * row0 │ 0  │ 1  │ 2  │ 3  │
     * row1 │ 4  │ 5  │ 6  │ 7  │
     * row2 │ 8  │ 9  │ 10 │ 11 │
     *      └────┴────┴────┴────┘
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 3, 4>;

    SECTION("Transpose [1,0]")
    {
        /*
         * LogicalDims = [4, 3]
         * Transposed view (4x3):
         *
         *       col0 col1 col2
         *      ┌────┬────┬────┐
         * row0 │ 0  │ 4  │ 8  │   logical flat: 0, 1, 2
         * row1 │ 1  │ 5  │ 9  │   logical flat: 3, 4, 5
         * row2 │ 2  │ 6  │ 10 │   logical flat: 6, 7, 8
         * row3 │ 3  │ 7  │ 11 │   logical flat: 9, 10, 11
         *      └────┴────┴────┘
         *
         * Strides = [1, 4]
         */
        using Root = StridedLayout<Policy>;
        using Layout = PermutedLayout<Root, 1, 0>;

        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 4);
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 8);
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);
        REQUIRE(Layout::logical_flat_to_physical_flat(4) == 5);
        REQUIRE(Layout::logical_flat_to_physical_flat(5) == 9);
        REQUIRE(Layout::logical_flat_to_physical_flat(6) == 2);
        REQUIRE(Layout::logical_flat_to_physical_flat(7) == 6);
        REQUIRE(Layout::logical_flat_to_physical_flat(8) == 10);
        REQUIRE(Layout::logical_flat_to_physical_flat(9) == 3);
        REQUIRE(Layout::logical_flat_to_physical_flat(10) == 7);
        REQUIRE(Layout::logical_flat_to_physical_flat(11) == 11);
    }
}

// ============================================================================
// DIAGONAL
// ============================================================================

TEST_CASE("diagonal_layout: diagonal of a padded square matrix",
          "[layout_adapters]")
{
    // [3,3] double, SimdWidth=4 → Strides [4,1] → diagonal stride 5
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 3>>;
    using D = DiagonalLayout<Root>;

    REQUIRE(D::logical_dim(0) == 3);
    REQUIRE(D::logical_dim(1) == 1);
    REQUIRE(D::stride(0) == 5);
    REQUIRE(D::Base == 0);
    REQUIRE_FALSE(D::is_contiguous());

    REQUIRE(D::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(D::logical_flat_to_physical_flat(1) == 5);
    REQUIRE(D::logical_flat_to_physical_flat(2) == 10);
}

TEST_CASE("diagonal_layout: diagonal of a block starts at the block",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 3>>;
    using Block = SlicedLayout<Root, Slice<0, 1, 2>, Slice<1, 1, 2>>;
    using D = DiagonalLayout<Block>;

    REQUIRE(D::Base == 5);
    REQUIRE(D::logical_coords_to_physical_flat(0, 0) == Root::logical_coords_to_physical_flat(1, 1));
    REQUIRE(D::logical_coords_to_physical_flat(1, 0) == Root::logical_coords_to_physical_flat(2, 2));
}

TEST_CASE("diagonal_layout: a transpose has the same diagonal",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 3>>;
    using D = DiagonalLayout<Root>;
    using DT = DiagonalLayout<PermutedLayout<Root, 1, 0>>;

    for (my_size_t n = 0; n < 3; ++n)
    {
        REQUIRE(DT::logical_flat_to_physical_flat(n) == D::logical_flat_to_physical_flat(n));
    }
}

// ============================================================================
// COMPOSITION
// ============================================================================
// Each adapter reads only NumDims, logical_dim, stride and Base from its
// source, so these chains are layouts of layouts, in any order.

TEST_CASE("layout_adapters: row of a squeezed corner",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 5, 2, 4>>;
    using Corner = SlicedLayout<Root, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using Row = SlicedLayout<SqueezedLayout<Corner, 0, 1>, Slice<0, 1, 1>>;

    REQUIRE(Row::Base == 60);
    for (my_size_t j = 0; j < 4; ++j)
    {
        REQUIRE(Row::logical_coords_to_physical_flat(0, j) ==
                Root::logical_coords_to_physical_flat(1, 2, 1, j));
    }
}

TEST_CASE("layout_adapters: slice of a subvector",
          "[layout_adapters]")
{
    // [6,1] double, SimdWidth=4 → padded to [6,4] → Strides [4,1]
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 6, 1>>;
    using Sub = SlicedLayout<Root, Slice<0, 2, 3>>;   // rows 2..4
    using Part = SlicedLayout<Sub, Slice<0, 1, 2>>;   // rows 1..2 of that

    REQUIRE(Sub::logical_dim(0) == 3);
    REQUIRE(Part::logical_dim(0) == 2);
    REQUIRE(Part::logical_coords_to_physical_flat(0, 0) == Root::logical_coords_to_physical_flat(3, 0));
    REQUIRE(Part::logical_coords_to_physical_flat(1, 0) == Root::logical_coords_to_physical_flat(4, 0));
}

TEST_CASE("layout_adapters: slice of a diagonal",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 3>>;
    using Tail = SlicedLayout<DiagonalLayout<Root>, Slice<0, 1, 2>>;

    REQUIRE(Tail::logical_dim(0) == 2);
    REQUIRE(Tail::logical_dim(1) == 1);
    REQUIRE(Tail::logical_coords_to_physical_flat(0, 0) == Root::logical_coords_to_physical_flat(1, 1));
    REQUIRE(Tail::logical_coords_to_physical_flat(1, 0) == Root::logical_coords_to_physical_flat(2, 2));
}

TEST_CASE("layout_adapters: subvector of a column",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 3>>;
    using Col2 = SlicedLayout<Root, Slice<1, 2, 1>>;
    using Last2 = SlicedLayout<Col2, Slice<0, 1, 2>>;

    REQUIRE(Last2::logical_coords_to_physical_flat(0, 0) == Root::logical_coords_to_physical_flat(1, 2));
    REQUIRE(Last2::logical_coords_to_physical_flat(1, 0) == Root::logical_coords_to_physical_flat(2, 2));
}

TEST_CASE("layout_adapters: a slice of a slice equals the flat slice",
          "[layout_adapters]")
{
    using Root = StridedLayout<SimdPaddingPolicyBase<double, 4, 4, 6>>;
    using Nested = SlicedLayout<SlicedLayout<Root, Slice<0, 1, 3>>, Slice<0, 1, 2>>;
    using Flat = SlicedLayout<Root, Slice<0, 2, 2>>;

    REQUIRE(Nested::Base == Flat::Base);
    for (my_size_t i = 0; i < 2; ++i)
    {
        REQUIRE(Nested::logical_dim(i) == Flat::logical_dim(i));
        REQUIRE(Nested::stride(i) == Flat::stride(i));
    }
}

// ============================================================================
// BOTH DIRECTIONS
// ============================================================================
// Every adapter inherits the reverse conversions, so each one has to hold on
// what the adapter computed, with every SIMD width.

template <my_size_t Width>
struct Simd
{
    template <my_size_t... Dims>
    using Policy = SimdPaddingPolicyBase<double, Width, Dims...>;
};

struct NoPadding
{
    template <my_size_t... Dims>
    using Policy = NoPaddingPolicy<double, Dims...>;
};

TEMPLATE_TEST_CASE("layout_adapters: permuted layouts are two-way", "[layout_adapters]",
                   Simd<1>, Simd<2>, Simd<4>, Simd<8>, NoPadding)
{
    using Matrix = StridedLayout<typename TestType::template Policy<4, 6>>;
    using Cube = StridedLayout<typename TestType::template Policy<2, 3, 5>>;
    using Hyper = StridedLayout<typename TestType::template Policy<2, 3, 4, 5>>;
    using Tied = StridedLayout<typename TestType::template Policy<3, 1, 4>>;

    check_two_way<PermutedLayout<Matrix, 1, 0>>(Matrix::PhysicalSize);
    check_two_way<PermutedLayout<Cube, 0, 2, 1>>(Cube::PhysicalSize);
    check_two_way<PermutedLayout<Cube, 1, 2, 0>>(Cube::PhysicalSize);
    check_two_way<PermutedLayout<Cube, 2, 1, 0>>(Cube::PhysicalSize);
    check_two_way<PermutedLayout<Hyper, 3, 1, 0, 2>>(Hyper::PhysicalSize);
    check_two_way<PermutedLayout<Tied, 1, 0, 2>>(Tied::PhysicalSize); // extent 1 first, tied stride
}

TEMPLATE_TEST_CASE("layout_adapters: sliced layouts are two-way", "[layout_adapters]",
                   Simd<1>, Simd<2>, Simd<4>, Simd<8>, NoPadding)
{
    using Matrix = StridedLayout<typename TestType::template Policy<4, 6>>;
    using Cube = StridedLayout<typename TestType::template Policy<2, 3, 5>>;

    check_two_way<SlicedLayout<Matrix, Slice<0, 1, 1>>>(Matrix::PhysicalSize);                 // row
    check_two_way<SlicedLayout<Matrix, Slice<1, 3, 1>>>(Matrix::PhysicalSize);                 // column
    check_two_way<SlicedLayout<Matrix, Slice<0, 1, 2>, Slice<1, 2, 3>>>(Matrix::PhysicalSize); // block
    check_two_way<SlicedLayout<Cube, Slice<0, 1, 1>, Slice<1, 1, 2>>>(Cube::PhysicalSize);     // corner
}

TEMPLATE_TEST_CASE("layout_adapters: squeezed and diagonal layouts are two-way", "[layout_adapters]",
                   Simd<1>, Simd<2>, Simd<4>, Simd<8>, NoPadding)
{
    using Square = StridedLayout<typename TestType::template Policy<4, 4>>;
    using Planes = StridedLayout<typename TestType::template Policy<1, 1, 3, 5>>;
    using Cube = StridedLayout<typename TestType::template Policy<2, 3, 5>>;

    check_two_way<SqueezedLayout<Planes, 0>>(Planes::PhysicalSize);
    check_two_way<SqueezedLayout<Planes, 0, 1>>(Planes::PhysicalSize);
    check_two_way<SqueezedLayout<SlicedLayout<Cube, Slice<0, 1, 1>>, 0>>(Cube::PhysicalSize);
    check_two_way<DiagonalLayout<Square>>(Square::PhysicalSize);
}

TEMPLATE_TEST_CASE("layout_adapters: chains are two-way", "[layout_adapters]",
                   Simd<1>, Simd<2>, Simd<4>, Simd<8>, NoPadding)
{
    using Matrix = StridedLayout<typename TestType::template Policy<4, 6>>;
    using Cube = StridedLayout<typename TestType::template Policy<2, 3, 5>>;

    using Column = SlicedLayout<PermutedLayout<Matrix, 1, 0>, Slice<1, 2, 1>>;
    using Plane = SqueezedLayout<SlicedLayout<Cube, Slice<0, 1, 1>>, 0>;
    using Deep = SlicedLayout<PermutedLayout<Plane, 1, 0>, Slice<0, 1, 2>>;

    check_two_way<Column>(Matrix::PhysicalSize);
    check_two_way<Plane>(Cube::PhysicalSize);
    check_two_way<Deep>(Cube::PhysicalSize);
}

TEST_CASE("layout_adapters: a column holds only its own offsets", "[layout_adapters]")
{
    /*
     * 2x3 matrix padded to 2x4, column 1:
     *
     *   flat: 0 [1] 2  P | 4 [5] 6  P
     *
     * Offsets 1 and 5 are the column. Offset 2 belongs to the matrix but not to
     * the column, offset 3 is padding, and offset 0 sits before Base.
     */
    using Matrix = StridedLayout<SimdPaddingPolicyBase<double, 4, 2, 3>>;
    using Column = SlicedLayout<Matrix, Slice<1, 1, 1>>;
    my_size_t coords[2];

    REQUIRE(Column::is_physical_flat_in_layout(1));
    REQUIRE(Column::is_physical_flat_in_layout(5));
    REQUIRE_FALSE(Column::is_physical_flat_in_layout(0));
    REQUIRE_FALSE(Column::is_physical_flat_in_layout(2));
    REQUIRE_FALSE(Column::is_physical_flat_in_layout(3));
    REQUIRE_FALSE(Column::is_physical_flat_in_layout(9));

    Column::physical_flat_to_logical_coords(5, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 0);
    REQUIRE(Column::physical_flat_to_logical_flat(5) == 1);

    CHECK_THROWS(Column::physical_flat_to_logical_coords(0, coords));
    CHECK_THROWS(Column::physical_flat_to_logical_coords(2, coords));
    CHECK_THROWS(Column::physical_flat_to_logical_flat(3));
}

TEST_CASE("layout_adapters: a diagonal holds every (stride + 1)-th offset", "[layout_adapters]")
{
    using Square = StridedLayout<SimdPaddingPolicyBase<double, 4, 3, 3>>; // padded to [3,4]
    using Diagonal = DiagonalLayout<Square>;

    REQUIRE(Diagonal::is_physical_flat_in_layout(0));
    REQUIRE(Diagonal::is_physical_flat_in_layout(5));
    REQUIRE(Diagonal::is_physical_flat_in_layout(10));
    REQUIRE_FALSE(Diagonal::is_physical_flat_in_layout(4));
    REQUIRE_FALSE(Diagonal::is_physical_flat_in_layout(15)); // one step past the last element
}

// ============================================================================
// VALIDATION
// ============================================================================
// The negative cases are static_asserts and cannot be exercised from a runtime
// test. Instantiating them fails the build. They are listed here so the intent
// is recorded; verify them by uncommenting one at a time.

TEST_CASE("layout_adapters: negative cases are compile-time errors",
          "[layout_adapters]")
{
    using M = StridedLayout<SimdPaddingPolicyBase<double, 4, 4, 6>>;
    using T3 = StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 3, 4>>;
    using T4 = StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 3, 1, 4>>;
    using Column3 = StridedLayout<SimdPaddingPolicyBase<double, 4, 1, 3, 1>>;
    using Sub = SlicedLayout<StridedLayout<SimdPaddingPolicyBase<double, 4, 6, 1>>, Slice<0, 2, 3>>;
    (void)sizeof(M);
    (void)sizeof(T3);
    (void)sizeof(T4);
    (void)sizeof(Column3);
    (void)sizeof(Sub);

    // using L = SlicedLayout<M, Slice<9, 0, 1>>;                  // axis out of range
    // using L = SlicedLayout<M, Slice<0, 0, 0>>;                  // Len == 0
    // using L = SlicedLayout<M, Slice<0, 2, 3>>;                  // Offset + Len > extent
    // using L = SlicedLayout<M, Slice<0, 0, 1>, Slice<0, 1, 1>>;  // duplicate axis
    // using L = SlicedLayout<M>;                                  // empty pack
    // using L = SqueezedLayout<T3>;                               // empty pack
    // using L = SlicedLayout<Sub, Slice<0, 1, 5>>;                // 5 rows of a 3-row view
    // using L = SqueezedLayout<T3, 1>;                            // axis 1 has extent 3
    // using L = SqueezedLayout<T4, 0, 0>;                         // duplicate axis
    // using L = SqueezedLayout<Column3, 0, 2>;                    // leaves [3]: below the rank floor
    // using L = PermutedLayout<M, 0>;                             // one entry for two axes
    // using L = PermutedLayout<M, 0, 0>;                          // duplicate entry
    // using L = DiagonalLayout<M>;                                // not square
    // using L = DiagonalLayout<T3>;                               // not 2D
    // (void)L::Base;
}
