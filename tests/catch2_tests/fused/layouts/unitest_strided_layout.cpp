#include <catch_amalgamated.hpp>
#include "fused/layouts/strided_layout.h"
#include "fused/layouts/layout_adapters.h"
#include "fused/padding_policies/simd_padding_policy.h"
#include "fused/padding_policies/no_padding_policy.h"
#include "layout_two_way.h"
#include "simd_widths.h"

// ############################################################################
//                              TEST CASES
// ############################################################################

// ============================================================================
// SECTION 1: BASIC PROPERTIES (IDENTITY PERMUTATION)
// ============================================================================

TEST_CASE("Layout inherits Policy properties", "[layout][strided_layout]")
{
    /*
     * 2x3 matrix, no padding (SCALAR), identity permutation
     *
     * Logical view:          Physical memory (same):
     *   [0,0] [0,1] [0,2]      0  1  2
     *   [1,0] [1,1] [1,2]      3  4  5
     *
     * LogicalSize = PhysicalSize = 6
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::NumDims == 2);
    REQUIRE(Layout::LogicalSize == 6);
    REQUIRE(Layout::PhysicalSize == 6);
}

TEST_CASE("Layout with padding inherits correct sizes", "[layout][padding][strided_layout]")
{
    /*
     * 2x3 matrix with AVX padding (SW=4)
     *
     * Logical view (2x3):     Physical memory (2x4):
     *   [0,0] [0,1] [0,2]       0  1  2  [P]
     *   [1,0] [1,1] [1,2]       4  5  6  [P]
     *
     * LogicalSize = 6, PhysicalSize = 8
     * [P] = padding element
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::NumDims == 2);
    REQUIRE(Layout::LogicalSize == 6);
    REQUIRE(Layout::PhysicalSize == 8);
}

// ============================================================================
// SECTION 3: LOGICAL DIMENSIONS
// ============================================================================

TEST_CASE("LogicalDims with identity permutation", "[layout][dims][strided_layout]")
{
    /*
     * Identity: LogicalDims = Policy::logical_dim
     */
    SECTION("2D")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 8, 6>;
        using Layout = StridedLayout<Policy>;

        REQUIRE(Layout::logical_dim(0) == 8);
        REQUIRE(Layout::logical_dim(1) == 6);
    }

    SECTION("3D")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
        using Layout = StridedLayout<Policy>;

        REQUIRE(Layout::logical_dim(0) == 2);
        REQUIRE(Layout::logical_dim(1) == 3);
        REQUIRE(Layout::logical_dim(2) == 4);
    }
}

// ============================================================================
// SECTION 4: STRIDE COMPUTATIONS
// ============================================================================

TEST_CASE("Strides are row-major from PhysicalDims", "[layout][strides][base][strided_layout]")
{
    /*
     * 2x3 matrix with AVX padding (SW=4)
     * PhysicalDims = [2, 4]
     *
     * Physical memory:
     *   flat:   0    1    2   3  |   4    5    6   7
     *         [0,0][0,1][0,2][P] | [1,0][1,1][1,2][P]
     *
     * Strides (row-major): [4, 1]
     *   stride[0] = 4 (jump 4 elements to next row)
     *   stride[1] = 1 (jump 1 element to next column)
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::stride(0) == 4);
    REQUIRE(Layout::stride(1) == 1);
}

TEST_CASE("Strides for 3D tensor", "[layout][strides][base][3d][strided_layout]")
{
    /*
     * 2x3x5 tensor with AVX padding (SW=4)
     * PhysicalDims = [2, 3, 8]  (5 padded to 8)
     *
     * Strides: [24, 8, 1]
     *   stride[0] = 3 * 8 = 24
     *   stride[1] = 8
     *   stride[2] = 1
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 5>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::stride(0) == 24);
    REQUIRE(Layout::stride(1) == 8);
    REQUIRE(Layout::stride(2) == 1);
}

TEST_CASE("?Stride invariant: Strides product equals PhysicalSize", "[layout][strides][invariant][strided_layout]")
{
    SECTION("2D no padding")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 8, 6>;
        using Layout = StridedLayout<Policy>;

        REQUIRE(Layout::stride(0) * Policy::PhysicalDims[0] == Layout::PhysicalSize);
    }

    SECTION("2D with padding")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>;
        using Layout = StridedLayout<Policy>;

        REQUIRE(Layout::stride(0) * Policy::PhysicalDims[0] == Layout::PhysicalSize);
    }

    SECTION("3D with padding")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 5>;
        using Layout = StridedLayout<Policy>;

        REQUIRE(Layout::stride(0) * Policy::PhysicalDims[0] == Layout::PhysicalSize);
    }
}

// ============================================================================
// SECTION 5: DIMENSION QUERY FUNCTIONS
// ============================================================================

TEST_CASE("num_dims() returns correct value", "[layout][query][strided_layout]")
{
    REQUIRE(StridedLayout<SimdPaddingPolicyBase<double, SCALAR, 5>>::num_dims() == 1);
    REQUIRE(StridedLayout<SimdPaddingPolicyBase<double, SCALAR, 2, 3>>::num_dims() == 2);
    REQUIRE(StridedLayout<SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>>::num_dims() == 3);
    REQUIRE(StridedLayout<SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4, 5>>::num_dims() == 4);
}

// ============================================================================
// SECTION 6: logical_coords_to_physical_flat
// ============================================================================

TEST_CASE("logical_coords_to_physical_flat: identity permutation", "[layout][index][l2p][strided_layout]")
{
    /*
     * 2x3 matrix, no padding, identity permutation
     *
     * Logical coords:        Physical flat:
     *   [0,0] [0,1] [0,2]      0  1  2
     *   [1,0] [1,1] [1,2]      3  4  5
     *
     * Formula: flat = i * 3 + j
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 2) == 2);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0) == 3);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 2) == 5);
    CHECK_THROWS(Layout::logical_coords_to_physical_flat(2, 0)); // out-of-bounds
    CHECK_THROWS(Layout::logical_coords_to_physical_flat(0, 3)); // out-of-bounds
}

TEST_CASE("logical_coords_to_physical_flat: with padding", "[layout][index][l2p][padding][strided_layout]")
{
    /*
     * 2x3 matrix with AVX padding (SW=4), identity permutation
     *
     * Logical coords:        Physical memory:
     *   [0,0] [0,1] [0,2]      0  1  2  [P]
     *   [1,0] [1,1] [1,2]      4  5  6  [P]
     *
     * Strides = [4, 1]
     * Formula: flat = i * 4 + j
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 2) == 2);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0) == 4); // skips padding
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1) == 5);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 2) == 6);

    CHECK_THROWS(Layout::logical_coords_to_physical_flat(2, 0)); // out-of-bounds
    CHECK_THROWS(Layout::logical_coords_to_physical_flat(0, 3)); // out-of-bounds
}

TEST_CASE("logical_coords_to_physical_flat: 3D tensor", "[layout][index][l2p][3d][strided_layout]")
{
    /*
     * 2x3x4 tensor, no padding, identity permutation
     *
     * Physical layout (slices):
     *   Slice 0 (i=0):           Slice 1 (i=1):
     *     0  1  2  3               12 13 14 15
     *     4  5  6  7               16 17 18 19
     *     8  9 10 11               20 21 22 23
     *
     * Strides = [12, 4, 1]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 3) == 3);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1, 0) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 2, 3) == 11);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0, 0) == 12);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 2, 3) == 23);
}

TEST_CASE("logical_coords_to_physical_flat: array overload", "[layout][index][l2p][array][strided_layout]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayout<Policy>;

    my_size_t indices1[] = {0, 2};
    my_size_t indices2[] = {1, 1};

    REQUIRE(Layout::logical_coords_to_physical_flat(indices1) == 2);
    REQUIRE(Layout::logical_coords_to_physical_flat(indices2) == 4);
}

// ============================================================================
// SECTION 7: logical_flat_to_physical_flat
// ============================================================================

TEST_CASE("logical_flat_to_physical_flat: identity, no padding", "[layout][index][lf2pf][strided_layout]")
{
    /*
     * 2x3 matrix, no padding, identity permutation
     * LogicalSize = PhysicalSize = 6
     *
     * logical flat == physical flat (identity mapping)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayout<Policy>;

    for (my_size_t i = 0; i < Layout::LogicalSize; ++i)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(i) == i);
    }
}

TEST_CASE("logical_flat_to_physical_flat: identity with padding", "[layout][index][lf2pf][padding][strided_layout]")
{
    /*
     * 2x3 matrix with AVX padding (SW=4)
     *
     * Logical flat:  0  1  2  3  4  5
     * Logical coords: (0,0)(0,1)(0,2)(1,0)(1,1)(1,2)
     *
     * Physical memory:
     *   flat: 0  1  2  [3] | 4  5  6  [7]
     *
     * Mapping:
     *   logical 0 -> physical 0
     *   logical 1 -> physical 1
     *   logical 2 -> physical 2
     *   logical 3 -> physical 4  (skips padding at 3)
     *   logical 4 -> physical 5
     *   logical 5 -> physical 6
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 1);
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 2);
    REQUIRE(Layout::logical_flat_to_physical_flat(3) == 4);
    REQUIRE(Layout::logical_flat_to_physical_flat(4) == 5);
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 6);
}

// ============================================================================
// SECTION 8: logical_flat_to_logical_coords
// ============================================================================

TEST_CASE("logical_flat_to_logical_coords: 2D", "[layout][index][lf2lc][strided_layout]")
{
    /*
     * 3x4 matrix
     *
     * Logical flat:    Logical coords:
     *   0  1  2  3       (0,0)(0,1)(0,2)(0,3)
     *   4  5  6  7       (1,0)(1,1)(1,2)(1,3)
     *   8  9 10 11       (2,0)(2,1)(2,2)(2,3)
     *
     * LogicalStrides = [4, 1]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 3, 4>;
    using Layout = StridedLayout<Policy>;
    my_size_t coords[2];

    Layout::logical_flat_to_logical_coords(0, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 0);

    Layout::logical_flat_to_logical_coords(3, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 3);

    Layout::logical_flat_to_logical_coords(4, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 0);

    Layout::logical_flat_to_logical_coords(7, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 3);

    Layout::logical_flat_to_logical_coords(11, coords);
    REQUIRE(coords[0] == 2);
    REQUIRE(coords[1] == 3);
}

TEST_CASE("logical_flat_to_logical_coords: 3D", "[layout][index][lf2lc][3d][strided_layout]")
{
    /*
     * 2x3x4 tensor
     * LogicalStrides = [12, 4, 1]
     *
     * logical flat 17:
     *   17 / 12 = 1, rem 5
     *   5 / 4 = 1, rem 1
     *   1 / 1 = 1
     *   coords = (1, 1, 1)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Layout = StridedLayout<Policy>;
    my_size_t coords[3];

    Layout::logical_flat_to_logical_coords(0, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 0);
    REQUIRE(coords[2] == 0);

    Layout::logical_flat_to_logical_coords(17, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 1);
    REQUIRE(coords[2] == 1);

    Layout::logical_flat_to_logical_coords(23, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 2);
    REQUIRE(coords[2] == 3);

    CHECK_THROWS(Layout::logical_flat_to_logical_coords(24, coords)); // out-of-bounds
}

// ============================================================================
// SECTION 9: physical_flat_to_physical_coords
// ============================================================================

TEST_CASE("physical_flat_to_physical_coords: no padding", "[layout][index][pf2pc][strided_layout]")
{
    /*
     * 2x3 matrix, no padding
     *
     * Physical flat:    Physical coords:
     *   0  1  2           (0,0)(0,1)(0,2)
     *   3  4  5           (1,0)(1,1)(1,2)
     *
     * Strides = [3, 1]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayout<Policy>;
    my_size_t coords[2];

    Layout::physical_flat_to_physical_coords(0, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 0);

    Layout::physical_flat_to_physical_coords(1, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 1);

    Layout::physical_flat_to_physical_coords(2, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 2);

    Layout::physical_flat_to_physical_coords(3, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 0);

    Layout::physical_flat_to_physical_coords(4, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 1);

    Layout::physical_flat_to_physical_coords(5, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 2);

    CHECK_THROWS(Layout::physical_flat_to_physical_coords(6, coords)); // out-of-bounds
}

TEST_CASE("physical_flat_to_physical_coords: with padding", "[layout][index][pf2pc][padding][strided_layout]")
{
    /*
     * 2x3 matrix with AVX padding (SW=4)
     * PhysicalDims = [2, 4]
     *
     * Physical memory:
     *   flat: 0  1  2  3  | 4  5  6  7
     *         ------P---    ------P---
     *
     * Strides = [4, 1]
     *
     * flat 3 -> (0, 3)  <- this is padding!
     * flat 7 -> (1, 3)  <- this is padding!
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayout<Policy>;
    my_size_t coords[2];

    Layout::physical_flat_to_physical_coords(0, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 0);

    Layout::physical_flat_to_physical_coords(2, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 2);

    // Padding location
    Layout::physical_flat_to_physical_coords(3, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 3);

    Layout::physical_flat_to_physical_coords(4, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 0);

    // Padding location
    Layout::physical_flat_to_physical_coords(7, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 3);

    CHECK_THROWS(Layout::physical_flat_to_physical_coords(8, coords)); // out-of-bounds
}

TEST_CASE("physical_coords_to_physical_flat: inverse of physical_flat_to_physical_coords", "[layout][index][pc2pf][padding][strided_layout]")
{
    /*
     * 2x3 matrix with AVX padding
     * PhysicalDims = [2, 4], Strides = [4, 1]
     *
     * Every slot of the padded grid has a position, padding included.
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayout<Policy>;
    my_size_t coords[2];

    for (my_size_t flat = 0; flat < Layout::PhysicalSize; ++flat)
    {
        Layout::physical_flat_to_physical_coords(flat, coords);
        REQUIRE(Layout::physical_coords_to_physical_flat(coords) == flat);
    }

    my_size_t padding[2] = {1, 3};
    REQUIRE(Layout::physical_coords_to_physical_flat(padding) == 7);

    my_size_t past_row[2] = {0, 4};
    CHECK_THROWS(Layout::physical_coords_to_physical_flat(past_row)); // out-of-bounds

    my_size_t past_end[2] = {2, 0};
    CHECK_THROWS(Layout::physical_coords_to_physical_flat(past_end)); // out-of-bounds
}

// ============================================================================
// SECTION 10: physical_flat_to_logical_coords
// ============================================================================

TEST_CASE("physical_flat_to_logical_coords: identity permutation", "[layout][index][pf2lc][strided_layout]")
{
    /*
     * 2x3 matrix, identity permutation
     * physical coords == logical coords
     *
     * Physical flat:    Logical coords:
     *   0  1  2           (0,0)(0,1)(0,2)
     *   3  4  5           (1,0)(1,1)(1,2)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayout<Policy>;
    my_size_t coords[2];

    Layout::physical_flat_to_logical_coords(0, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 0);

    Layout::physical_flat_to_logical_coords(1, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 1);

    Layout::physical_flat_to_logical_coords(2, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 2);

    Layout::physical_flat_to_logical_coords(3, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 0);

    Layout::physical_flat_to_logical_coords(4, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 1);

    Layout::physical_flat_to_logical_coords(5, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 2);

    CHECK_THROWS(Layout::physical_flat_to_logical_coords(6, coords)); // out-of-bounds
}

TEST_CASE("physical_flat_to_logical_coords: padding holds no element", "[layout][index][pf2lc][padding][strided_layout]")
{
    /*
     * 2x3 matrix with AVX padding
     * PhysicalDims = [2, 4], LogicalDims = [2, 3]
     *
     * Physical memory:
     *   flat: 0  1  2  [3]  | 4  5  6  [7]
     *
     * physical flat 3 and 7 are padding, so they have no logical coords.
     * They do have a place in the padded grid: (0, 3) and (1, 3).
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayout<Policy>;
    my_size_t coords[2];

    // The check tells padding from elements without reporting an error
    REQUIRE(Layout::is_physical_flat_in_layout(2));
    REQUIRE_FALSE(Layout::is_physical_flat_in_layout(3));
    REQUIRE(Layout::is_physical_flat_in_layout(4));
    REQUIRE_FALSE(Layout::is_physical_flat_in_layout(7));

    // The conversion reports it
    CHECK_THROWS(Layout::physical_flat_to_logical_coords(3, coords));
    CHECK_THROWS(Layout::physical_flat_to_logical_coords(7, coords));

    // The padded grid still answers
    Layout::physical_flat_to_physical_coords(3, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 3); // >= LogicalDims[1], a padding column
    REQUIRE(coords[1] >= Layout::logical_dim(1));

    Layout::physical_flat_to_physical_coords(7, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 3);
}

// // ============================================================================
// // SECTION 11: ROUNDTRIP INVARIANTS
// // ============================================================================

TEST_CASE("Roundtrip: logical_coords -> physical_flat -> logical_coords", "[layout][roundtrip][strided_layout]")
{
    /*
     * For all valid logical coords:
     *   logical_coords -> physical_flat -> logical_coords
     * should return original coords
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;

    SECTION("Identity permutation")
    {
        using Layout = StridedLayout<Policy>;

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

TEST_CASE("Roundtrip: logical_flat -> physical_flat consistency", "[layout][roundtrip][strided_layout]")
{
    /*
     * logical_flat_to_physical_flat should equal:
     *   logical_flat -> logical_coords -> physical_flat
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 3, 4>;

    SECTION("Identity")
    {
        using Layout = StridedLayout<Policy>;

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

// ============================================================================
// SECTION 12: BOUNDS CHECKING
// ============================================================================

TEST_CASE("is_logical_index_in_bounds", "[layout][bounds][strided_layout]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayout<Policy>;

    SECTION("Valid indices")
    {
        my_size_t valid1[] = {0, 0};
        my_size_t valid2[] = {0, 2};
        my_size_t valid3[] = {1, 0};
        my_size_t valid4[] = {1, 2};

        REQUIRE(Layout::is_logical_index_in_bounds(valid1));
        REQUIRE(Layout::is_logical_index_in_bounds(valid2));
        REQUIRE(Layout::is_logical_index_in_bounds(valid3));
        REQUIRE(Layout::is_logical_index_in_bounds(valid4));
    }

    SECTION("Invalid indices")
    {
        my_size_t invalid1[] = {2, 0}; // row out of bounds
        my_size_t invalid2[] = {0, 3}; // col out of bounds
        my_size_t invalid3[] = {2, 3}; // both out of bounds

        REQUIRE_FALSE(Layout::is_logical_index_in_bounds(invalid1));
        REQUIRE_FALSE(Layout::is_logical_index_in_bounds(invalid2));
        REQUIRE_FALSE(Layout::is_logical_index_in_bounds(invalid3));
    }
}

// ============================================================================
// SECTION 13: EDGE CASES
// ============================================================================

TEST_CASE("1D tensor (vector)", "[layout][edge][1d][strided_layout]")
{
    /*
     * 1D tensor of size 5 with padding to 8
     *
     * Logical: [0] [1] [2] [3] [4]
     * Physical: 0   1   2   3   4  [5] [6] [7]
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 5>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::NumDims == 1);
    REQUIRE(Layout::LogicalSize == 5);
    REQUIRE(Layout::PhysicalSize == 8);

    REQUIRE(Layout::stride(0) == 1);
    REQUIRE(Layout::stride(0) == 1);
    REQUIRE(Layout::logical_stride(0) == 1);

    REQUIRE(Layout::logical_coords_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(4) == 4);

    for (my_size_t i = 0; i < 5; ++i)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(i) == i);
    }

    CHECK_THROWS(Layout::logical_coords_to_physical_flat(5)); // out-of-bounds
    CHECK_THROWS(Layout::logical_flat_to_physical_flat(5));   // out-of-bounds
}

TEST_CASE("Single element 1D tensor", "[layout][edge][single][strided_layout]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 1>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::NumDims == 1);
    REQUIRE(Layout::LogicalSize == 1);
    REQUIRE(Layout::PhysicalSize == 4); // padded

    REQUIRE(Layout::logical_coords_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
}

TEST_CASE("Single element 2D tensor", "[layout][edge][single][strided_layout]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 1, 1>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::NumDims == 2);
    REQUIRE(Layout::LogicalSize == 1);
    REQUIRE(Layout::PhysicalSize == 4); // padded

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
}

TEST_CASE("Large dimensions", "[layout][edge][large][strided_layout]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 100, 100>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::LogicalSize == 10000);
    REQUIRE(Layout::PhysicalSize == 10000); // 100 is already aligned to 4

    REQUIRE(Layout::logical_coords_to_physical_flat(99, 99) == 9999);
    REQUIRE(Layout::logical_flat_to_physical_flat(9999) == 9999);
}

TEST_CASE("High dimensional tensor", "[layout][edge][highdim][strided_layout]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 2, 2, 2, 2>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::NumDims == 5);
    REQUIRE(Layout::LogicalSize == 32);
    REQUIRE(Layout::PhysicalSize == 32);

    // Check strides are powers of 2
    REQUIRE(Layout::stride(0) == 16);
    REQUIRE(Layout::stride(1) == 8);
    REQUIRE(Layout::stride(2) == 4);
    REQUIRE(Layout::stride(3) == 2);
    REQUIRE(Layout::stride(4) == 1);
}

// ============================================================================
// SECTION 14: COMPILE-TIME GUARANTEES
// ============================================================================

TEST_CASE("All members are constexpr", "[layout][constexpr][strided_layout]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayout<Policy>;

    // Use in array sizes (compile-time requirement)
    [[maybe_unused]] double arr1[Layout::NumDims];
    [[maybe_unused]] double arr2[Layout::LogicalSize];
    [[maybe_unused]] double arr3[Layout::PhysicalSize];

    // Static assertions
    static_assert(Layout::NumDims == 2);
    static_assert(Layout::LogicalSize == 6);
    static_assert(Layout::PhysicalSize == 8);

    static_assert(Layout::logical_dim(0) == 2);
    static_assert(Layout::logical_dim(1) == 3);

    static_assert(Layout::stride(0) == 4);
    static_assert(Layout::stride(1) == 1);

    SUCCEED("All constexpr usages compiled successfully");
}

TEST_CASE("Index functions are constexpr", "[layout][constexpr][index][strided_layout]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayout<Policy>;

    constexpr my_size_t flat1 = Layout::logical_coords_to_physical_flat(1, 2);
    static_assert(flat1 == 5);

    constexpr my_size_t flat2 = Layout::logical_flat_to_physical_flat(4);
    static_assert(flat2 == 4);

    SUCCEED("Index functions are constexpr");
}

// ============================================================================
// SECTION 15: IDENTITY PERMUTATION
// ============================================================================

TEST_CASE("All 2D permutations: identity", "[layout][perm][2d][strided_layout]")
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

    SECTION("Identity [0,1]")
    {
        /*
         * LogicalDims = [3, 4]
         * Logical flat iterates row-major: 0,1,2,3,4,5,...
         * Same as physical flat.
         */
        using Layout = StridedLayout<Policy>;

        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 1);
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 2);
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 3);
        REQUIRE(Layout::logical_flat_to_physical_flat(4) == 4);
        REQUIRE(Layout::logical_flat_to_physical_flat(5) == 5);
        REQUIRE(Layout::logical_flat_to_physical_flat(11) == 11);
    }
}

// ============================================================================
// SECTION 17: Tests with NoPaddingPolicy
// ============================================================================

TEST_CASE("NoPaddingPolicy: 2D identity", "[layout][nopadding][2d][strided_layout]")
{
    /*
     * 2x3 matrix, no padding, identity permutation
     *
     * Physical memory:
     *       col0 col1 col2
     *      ┌────┬────┬────┐
     * row0 │ 0  │ 1  │ 2  │
     * row1 │ 3  │ 4  │ 5  │
     *      └────┴────┴────┘
     *
     * LogicalSize == PhysicalSize (no padding)
     * logical flat == physical flat (identity)
     */
    using Policy = NoPaddingPolicy<double, 2, 3>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::LogicalSize == 6);
    REQUIRE(Layout::PhysicalSize == 6);
    REQUIRE(Layout::stride(0) == 3);
    REQUIRE(Layout::stride(1) == 1);
    REQUIRE(Layout::stride(0) == 3);
    REQUIRE(Layout::stride(1) == 1);

    // Identity: logical flat == physical flat
    for (my_size_t lf = 0; lf < Layout::LogicalSize; ++lf)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(lf) == lf);
    }
}

TEST_CASE("NoPaddingPolicy: 3D identity", "[layout][nopadding][3d][strided_layout]")
{
    /*
     * 2x3x4 tensor, no padding, identity permutation
     *
     * Strides = [12, 4, 1]
     * LogicalSize == PhysicalSize == 24
     */
    using Policy = NoPaddingPolicy<double, 2, 3, 4>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::LogicalSize == 24);
    REQUIRE(Layout::PhysicalSize == 24);
    REQUIRE(Layout::stride(0) == 12);
    REQUIRE(Layout::stride(1) == 4);
    REQUIRE(Layout::stride(2) == 1);

    // Identity: logical flat == physical flat
    for (my_size_t lf = 0; lf < Layout::LogicalSize; ++lf)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(lf) == lf);
    }

    // Spot check coords
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 3) == 3);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1, 0) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0, 0) == 12);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 2, 3) == 23);
}

TEST_CASE("NoPaddingPolicy: 1D vector", "[layout][nopadding][1d][strided_layout]")
{
    /*
     * 1D tensor of size 10
     * No permutation possible (only one dimension)
     */
    using Policy = NoPaddingPolicy<double, 10>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::NumDims == 1);
    REQUIRE(Layout::LogicalSize == 10);
    REQUIRE(Layout::PhysicalSize == 10);
    REQUIRE(Layout::stride(0) == 1);

    for (my_size_t lf = 0; lf < Layout::LogicalSize; ++lf)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(lf) == lf);
        REQUIRE(Layout::logical_coords_to_physical_flat(lf) == lf);
    }
}

TEST_CASE("NoPaddingPolicy: single element", "[layout][nopadding][edge][strided_layout]")
{
    using Policy = NoPaddingPolicy<double, 1, 1, 1>;
    using Layout = StridedLayout<Policy>;

    REQUIRE(Layout::NumDims == 3);
    REQUIRE(Layout::LogicalSize == 1);
    REQUIRE(Layout::PhysicalSize == 1);

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
}

// ============================================================================
// SECTION 18: TWO-WAY, EVERY SIMD WIDTH
// ============================================================================

TEMPLATE_TEST_CASE("Two-way with every SIMD width", "[layout][roundtrip][simd][strided_layout]",
                   ScalarWidth, SSEDoubleWidth, AVXDoubleWidth, AVX512DoubleWidth)
{
    using Vector = StridedLayout<SimdPaddingPolicyBase<double, TestType::value, 7>>;
    using Matrix = StridedLayout<SimdPaddingPolicyBase<double, TestType::value, 4, 6>>;
    using Cube = StridedLayout<SimdPaddingPolicyBase<double, TestType::value, 2, 3, 5>>;
    using Hyper = StridedLayout<SimdPaddingPolicyBase<double, TestType::value, 2, 3, 4, 5>>;

    check_two_way<Vector>(Vector::PhysicalSize);
    check_two_way<Matrix>(Matrix::PhysicalSize);
    check_two_way<Cube>(Cube::PhysicalSize);
    check_two_way<Hyper>(Hyper::PhysicalSize);
}

TEMPLATE_TEST_CASE("Two-way with axes of extent 1", "[layout][roundtrip][simd][strided_layout]",
                   ScalarWidth, SSEDoubleWidth, AVXDoubleWidth, AVX512DoubleWidth)
{
    // In [3,1,4] axes 0 and 1 have the same stride
    using Tied = StridedLayout<SimdPaddingPolicyBase<double, TestType::value, 3, 1, 4>>;
    using Ones = StridedLayout<SimdPaddingPolicyBase<double, TestType::value, 1, 1, 3, 1>>;

    check_two_way<Tied>(Tied::PhysicalSize);
    check_two_way<Ones>(Ones::PhysicalSize);
}

TEST_CASE("NoPaddingPolicy: two-way", "[layout][nopadding][roundtrip][strided_layout]")
{
    using Matrix = StridedLayout<NoPaddingPolicy<double, 4, 6>>;
    using Cube = StridedLayout<NoPaddingPolicy<double, 2, 3, 5>>;
    using Tied = StridedLayout<NoPaddingPolicy<double, 3, 1, 4>>;

    check_two_way<Matrix>(Matrix::PhysicalSize);
    check_two_way<Cube>(Cube::PhysicalSize);
    check_two_way<Tied>(Tied::PhysicalSize);
}
