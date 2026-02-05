#include <catch_amalgamated.hpp>
#include "fused/layouts/strided_layout_constexpr.h"
#include "fused/padding_policies/simd_padding_policy.h"

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

enum SimdWidth : my_size_t
{
    SCALAR = 1,
    SSE_DOUBLE = 2,
    SSE_FLOAT = 4,
    AVX_DOUBLE = 4,
    AVX_FLOAT = 8,
    AVX512_DOUBLE = 8,
    AVX512_FLOAT = 16
};

template <my_size_t SW>
struct SimdWidthTag
{
    static constexpr my_size_t value = SW;
};

using ScalarWidth = SimdWidthTag<SCALAR>;
using SSEFloatWidth = SimdWidthTag<SSE_FLOAT>;
using SSEDoubleWidth = SimdWidthTag<SSE_DOUBLE>;
using AVXFloatWidth = SimdWidthTag<AVX_FLOAT>;
using AVXDoubleWidth = SimdWidthTag<AVX_DOUBLE>;
using AVX512FloatWidth = SimdWidthTag<AVX512_FLOAT>;
using AVX512DoubleWidth = SimdWidthTag<AVX512_DOUBLE>;

// ############################################################################
//                              TEST CASES
// ############################################################################

// ============================================================================
// SECTION 1: BASIC PROPERTIES (IDENTITY PERMUTATION)
// ============================================================================

TEST_CASE("Layout inherits Policy properties", "[layout][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::NumDims == 2);
    REQUIRE(Layout::LogicalSize == 6);
    REQUIRE(Layout::PhysicalSize == 6);
}

TEST_CASE("Layout with padding inherits correct sizes", "[layout][padding][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::NumDims == 2);
    REQUIRE(Layout::LogicalSize == 6);
    REQUIRE(Layout::PhysicalSize == 8);
}

TEST_CASE("IsPermProvided flag", "[layout][perm][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;

    SECTION("No permutation provided")
    {
        using Layout = StridedLayoutConstExpr<Policy>;
        REQUIRE(Layout::IsPermProvided == false);
    }

    SECTION("Identity permutation explicitly provided")
    {
        using Layout = StridedLayoutConstExpr<Policy, 0, 1>;
        REQUIRE(Layout::IsPermProvided == true);
    }

    SECTION("Non-identity permutation provided")
    {
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;
        REQUIRE(Layout::IsPermProvided == true);
    }
}

// ============================================================================
// SECTION 2: PERMUTATION ARRAY
// ============================================================================

TEST_CASE("PermArray is identity when no permutation provided", "[layout][perm][strided_layout_constexpr]")
{
    /*
     * Identity permutation: dimension i maps to dimension i
     * PermArray = [0, 1, 2, ...]
     */
    SECTION("2D")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::PermArray[0] == 0);
        REQUIRE(Layout::PermArray[1] == 1);
    }

    SECTION("3D")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::PermArray[0] == 0);
        REQUIRE(Layout::PermArray[1] == 1);
        REQUIRE(Layout::PermArray[2] == 2);
    }

    SECTION("4D")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4, 5>;
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::PermArray[0] == 0);
        REQUIRE(Layout::PermArray[1] == 1);
        REQUIRE(Layout::PermArray[2] == 2);
        REQUIRE(Layout::PermArray[3] == 3);
    }
}

TEST_CASE("PermArray stores provided permutation", "[layout][perm][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;

    SECTION("Transpose first two dims: [1,0,2]")
    {
        using Layout = StridedLayoutConstExpr<Policy, 1, 0, 2>;

        REQUIRE(Layout::PermArray[0] == 1);
        REQUIRE(Layout::PermArray[1] == 0);
        REQUIRE(Layout::PermArray[2] == 2);
    }

    SECTION("Rotate left: [1,2,0]")
    {
        using Layout = StridedLayoutConstExpr<Policy, 1, 2, 0>;

        REQUIRE(Layout::PermArray[0] == 1);
        REQUIRE(Layout::PermArray[1] == 2);
        REQUIRE(Layout::PermArray[2] == 0);
    }

    SECTION("Reverse: [2,1,0]")
    {
        using Layout = StridedLayoutConstExpr<Policy, 2, 1, 0>;

        REQUIRE(Layout::PermArray[0] == 2);
        REQUIRE(Layout::PermArray[1] == 1);
        REQUIRE(Layout::PermArray[2] == 0);
    }
}

TEST_CASE("InversePermArray is correct", "[layout][perm][inverse][strided_layout_constexpr]")
{
    /*
     * If PermArray[i] = j, then InversePermArray[j] = i
     * Used to map physical coords back to logical coords
     */
    SECTION("Identity permutation")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
        using Layout = StridedLayoutConstExpr<Policy>;

        // Identity is its own inverse
        REQUIRE(Layout::InversePermArray[0] == 0);
        REQUIRE(Layout::InversePermArray[1] == 1);
    }

    SECTION("2D transpose: [1,0]")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        // Transpose is its own inverse
        REQUIRE(Layout::InversePermArray[0] == 1);
        REQUIRE(Layout::InversePermArray[1] == 0);
    }

    SECTION("3D rotation: [1,2,0]")
    {
        /*
         * PermArray = [1, 2, 0]
         *   logical dim 0 -> physical dim 1
         *   logical dim 1 -> physical dim 2
         *   logical dim 2 -> physical dim 0
         *
         * InversePermArray = [2, 0, 1]
         *   physical dim 0 -> logical dim 2
         *   physical dim 1 -> logical dim 0
         *   physical dim 2 -> logical dim 1
         */
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
        using Layout = StridedLayoutConstExpr<Policy, 1, 2, 0>;

        REQUIRE(Layout::InversePermArray[0] == 2);
        REQUIRE(Layout::InversePermArray[1] == 0);
        REQUIRE(Layout::InversePermArray[2] == 1);
    }

    SECTION("Inverse property: Perm[InversePerm[i]] == i")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4, 5>;
        using Layout = StridedLayoutConstExpr<Policy, 3, 1, 0, 2>;

        for (my_size_t i = 0; i < Layout::NumDims; ++i)
        {
            REQUIRE(Layout::PermArray[Layout::InversePermArray[i]] == i);
            REQUIRE(Layout::InversePermArray[Layout::PermArray[i]] == i);
        }
    }
}

// ============================================================================
// SECTION 3: LOGICAL DIMENSIONS (PERMUTED)
// ============================================================================

TEST_CASE("LogicalDims with identity permutation", "[layout][dims][strided_layout_constexpr]")
{
    /*
     * Identity: LogicalDims = Policy::LogicalDims
     */
    SECTION("2D")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 8, 6>;
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::LogicalDims[0] == 8);
        REQUIRE(Layout::LogicalDims[1] == 6);
    }

    SECTION("3D")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::LogicalDims[0] == 2);
        REQUIRE(Layout::LogicalDims[1] == 3);
        REQUIRE(Layout::LogicalDims[2] == 4);
    }
}

TEST_CASE("LogicalDims with transpose permutation", "[layout][dims][transpose][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix transposed via [1,0]
     *
     * Original (2x3):         Transposed view (3x2):
     *   [0,0] [0,1] [0,2]       [0,0] [0,1]
     *   [1,0] [1,1] [1,2]       [1,0] [1,1]
     *                          [2,0] [2,1]
     *
     * Policy::LogicalDims = [2, 3]
     * Layout::LogicalDims = [3, 2]  (permuted)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    REQUIRE(Layout::LogicalDims[0] == 3); // was dim 1
    REQUIRE(Layout::LogicalDims[1] == 2); // was dim 0
}

TEST_CASE("LogicalDims with 3D permutation", "[layout][dims][3d][strided_layout_constexpr]")
{
    /*
     * 2x3x4 tensor with permutation [2,0,1]
     *
     * Policy::LogicalDims = [2, 3, 4]
     * Permutation [2,0,1] means:
     *   new dim 0 = old dim 2 = 4
     *   new dim 1 = old dim 0 = 2
     *   new dim 2 = old dim 1 = 3
     *
     * Layout::LogicalDims = [4, 2, 3]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Layout = StridedLayoutConstExpr<Policy, 2, 0, 1>;

    REQUIRE(Layout::LogicalDims[0] == 4);
    REQUIRE(Layout::LogicalDims[1] == 2);
    REQUIRE(Layout::LogicalDims[2] == 3);
}

// ============================================================================
// SECTION 4: STRIDE COMPUTATIONS
// ============================================================================

TEST_CASE("BaseStrides are row-major from PhysicalDims", "[layout][strides][base][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix with AVX padding (SW=4)
     * PhysicalDims = [2, 4]
     *
     * Physical memory:
     *   flat:   0    1    2   3  |   4    5    6   7
     *         [0,0][0,1][0,2][P] | [1,0][1,1][1,2][P]
     *
     * BaseStrides (row-major): [4, 1]
     *   stride[0] = 4 (jump 4 elements to next row)
     *   stride[1] = 1 (jump 1 element to next column)
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::BaseStrides[0] == 4);
    REQUIRE(Layout::BaseStrides[1] == 1);
}

TEST_CASE("BaseStrides for 3D tensor", "[layout][strides][base][3d][strided_layout_constexpr]")
{
    /*
     * 2x3x5 tensor with AVX padding (SW=4)
     * PhysicalDims = [2, 3, 8]  (5 padded to 8)
     *
     * BaseStrides: [24, 8, 1]
     *   stride[0] = 3 * 8 = 24
     *   stride[1] = 8
     *   stride[2] = 1
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 5>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::BaseStrides[0] == 24);
    REQUIRE(Layout::BaseStrides[1] == 8);
    REQUIRE(Layout::BaseStrides[2] == 1);
}

TEST_CASE("Strides are permuted BaseStrides", "[layout][strides][permuted][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix (no padding) with transpose [1,0]
     *
     * Physical memory (unchanged):
     *   flat:   0    1    2   |   3    4    5
     *         [0,0][0,1][0,2] | [1,0][1,1][1,2]
     *
     * BaseStrides = [3, 1]
     *
     * Transposed view: logical (i,j) accesses physical (j,i)
     * Strides = [BaseStrides[1], BaseStrides[0]] = [1, 3]
     *
     * Example: logical (1,0) in transposed view
     *   = 1 * Strides[0] + 0 * Strides[1]
     *   = 1 * 1 + 0 * 3 = 1
     *   = physical element at position 1 = original [0,1] ✓
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    REQUIRE(Layout::BaseStrides[0] == 3);
    REQUIRE(Layout::BaseStrides[1] == 1);

    REQUIRE(Layout::Strides[0] == 1); // BaseStrides[PermArray[0]] = BaseStrides[1]
    REQUIRE(Layout::Strides[1] == 3); // BaseStrides[PermArray[1]] = BaseStrides[0]
}

TEST_CASE("LogicalStrides are row-major from LogicalDims", "[layout][strides][logical][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    REQUIRE(Layout::LogicalDims[0] == 3);
    REQUIRE(Layout::LogicalDims[1] == 2);

    REQUIRE(Layout::LogicalStrides[0] == 2);
    REQUIRE(Layout::LogicalStrides[1] == 1);
}

TEST_CASE("?Stride invariant: BaseStrides product equals PhysicalSize", "[layout][strides][invariant][strided_layout_constexpr]")
{
    SECTION("2D no padding")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 8, 6>;
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::BaseStrides[0] * Policy::PhysicalDims[0] == Layout::PhysicalSize);
    }

    SECTION("2D with padding")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>;
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::BaseStrides[0] * Policy::PhysicalDims[0] == Layout::PhysicalSize);
    }

    SECTION("3D with padding")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 5>;
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::BaseStrides[0] * Policy::PhysicalDims[0] == Layout::PhysicalSize);
    }
}

// ============================================================================
// SECTION 5: DIMENSION QUERY FUNCTIONS
// ============================================================================

TEST_CASE("num_dims() returns correct value", "[layout][query][strided_layout_constexpr]")
{
    REQUIRE(StridedLayoutConstExpr<SimdPaddingPolicyBase<double, SCALAR, 5>>::num_dims() == 1);
    REQUIRE(StridedLayoutConstExpr<SimdPaddingPolicyBase<double, SCALAR, 2, 3>>::num_dims() == 2);
    REQUIRE(StridedLayoutConstExpr<SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>>::num_dims() == 3);
    REQUIRE(StridedLayoutConstExpr<SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4, 5>>::num_dims() == 4);
}

TEST_CASE("logical_dim() returns permuted dimensions", "[layout][query][strided_layout_constexpr]")
{
    /*
     * 2x3x4 tensor with permutation [2,0,1]
     * LogicalDims = [4, 2, 3]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Layout = StridedLayoutConstExpr<Policy, 2, 0, 1>;

    REQUIRE(Layout::logical_dim(0) == 4);
    REQUIRE(Layout::logical_dim(1) == 2);
    REQUIRE(Layout::logical_dim(2) == 3);
    CHECK_THROWS(Layout::logical_dim(3)); // out-of-bounds
}

TEST_CASE("stride() returns permuted strides", "[layout][query][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix transposed
     * BaseStrides = [3, 1]
     * Strides = [1, 3]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    REQUIRE(Layout::stride(0) == 1);
    REQUIRE(Layout::stride(1) == 3);
    CHECK_THROWS(Layout::stride(2)); // out-of-bounds
}

// ============================================================================
// SECTION 6: logical_coords_to_physical_flat
// ============================================================================

TEST_CASE("logical_coords_to_physical_flat: identity permutation", "[layout][index][l2p][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 2) == 2);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0) == 3);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 2) == 5);
    CHECK_THROWS(Layout::logical_coords_to_physical_flat(2, 0)); // out-of-bounds
    CHECK_THROWS(Layout::logical_coords_to_physical_flat(0, 3)); // out-of-bounds
}

TEST_CASE("logical_coords_to_physical_flat: with padding", "[layout][index][l2p][padding][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 2) == 2);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0) == 4); // skips padding
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1) == 5);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 2) == 6);

    CHECK_THROWS(Layout::logical_coords_to_physical_flat(2, 0)); // out-of-bounds
    CHECK_THROWS(Layout::logical_coords_to_physical_flat(0, 3)); // out-of-bounds
}

TEST_CASE("logical_coords_to_physical_flat: transposed", "[layout][index][l2p][transpose][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    // First column of transposed view = first row of original
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 0) == 2);

    // Second column of transposed view = second row of original
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1) == 3);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 1) == 5);
}

TEST_CASE("logical_coords_to_physical_flat: transposed with padding", "[layout][index][l2p][transpose][padding][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix with AVX padding, transposed to 3x2 view
     *
     * Physical memory:             Transposed logical view (3x2):
     *   flat: 0  1  2  [P]            [0,0]=0  [0,1]=4
     *         4  5  6  [P]            [1,0]=1  [1,1]=5
     *                                 [2,0]=2  [2,1]=6
     *
     * BaseStrides = [4, 1]
     * Strides = [1, 4]  (permuted)
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0) == 1);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 0) == 2);

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 1) == 5);
    REQUIRE(Layout::logical_coords_to_physical_flat(2, 1) == 6);

    CHECK_THROWS(Layout::logical_coords_to_physical_flat(3, 0)); // out-of-bounds
    CHECK_THROWS(Layout::logical_coords_to_physical_flat(0, 2)); // out-of-bounds
}

TEST_CASE("logical_coords_to_physical_flat: 3D tensor", "[layout][index][l2p][3d][strided_layout_constexpr]")
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
     * BaseStrides = Strides = [12, 4, 1]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0, 3) == 3);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 1, 0) == 4);
    REQUIRE(Layout::logical_coords_to_physical_flat(0, 2, 3) == 11);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 0, 0) == 12);
    REQUIRE(Layout::logical_coords_to_physical_flat(1, 2, 3) == 23);
}

TEST_CASE("logical_coords_to_physical_flat: 3D padding and permutation", "[layout][index][l2p][3d][padding][transpose][strided_layout_constexpr]")
{
    /*
     * Original tensor: 2x3x5 with AVX padding (SW=4)
     * Permutation: [2, 0, 1] creates a 5x2x3 logical view
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │ PHYSICAL MEMORY LAYOUT (unchanged by permutation)                   │
     * │                                                                     │
     * │ Policy::LogicalDims  = [2, 3, 5]                                    │
     * │ Policy::PhysicalDims = [2, 3, 8]  (last dim padded: 5 → 8)          │
     * │ BaseStrides = [24, 8, 1]                                            │
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
     * │ Layout::LogicalDims = [5, 2, 3]                                     │
     * │ Strides = [BaseStrides[2], BaseStrides[0], BaseStrides[1]]          │
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
    using Layout = StridedLayoutConstExpr<Policy, 2, 0, 1>;

    // Verify dimensions and strides
    REQUIRE(Layout::LogicalDims[0] == 5);
    REQUIRE(Layout::LogicalDims[1] == 2);
    REQUIRE(Layout::LogicalDims[2] == 3);

    REQUIRE(Layout::BaseStrides[0] == 24);
    REQUIRE(Layout::BaseStrides[1] == 8);
    REQUIRE(Layout::BaseStrides[2] == 1);

    REQUIRE(Layout::Strides[0] == 1);
    REQUIRE(Layout::Strides[1] == 24);
    REQUIRE(Layout::Strides[2] == 8);

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

TEST_CASE("logical_coords_to_physical_flat: array overload", "[layout][index][l2p][array][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy>;

    my_size_t indices1[] = {0, 2};
    my_size_t indices2[] = {1, 1};

    REQUIRE(Layout::logical_coords_to_physical_flat(indices1) == 2);
    REQUIRE(Layout::logical_coords_to_physical_flat(indices2) == 4);
}

// ============================================================================
// SECTION 7: logical_flat_to_physical_flat
// ============================================================================

TEST_CASE("logical_flat_to_physical_flat: identity, no padding", "[layout][index][lf2pf][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix, no padding, identity permutation
     * LogicalSize = PhysicalSize = 6
     *
     * logical flat == physical flat (identity mapping)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy>;

    for (my_size_t i = 0; i < Layout::LogicalSize; ++i)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(i) == i);
    }
}

TEST_CASE("logical_flat_to_physical_flat: identity with padding", "[layout][index][lf2pf][padding][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 1);
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 2);
    REQUIRE(Layout::logical_flat_to_physical_flat(3) == 4);
    REQUIRE(Layout::logical_flat_to_physical_flat(4) == 5);
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 6);
}

TEST_CASE("logical_flat_to_physical_flat: transposed", "[layout][index][lf2pf][transpose][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(1) == 3);
    REQUIRE(Layout::logical_flat_to_physical_flat(2) == 1);
    REQUIRE(Layout::logical_flat_to_physical_flat(3) == 4);
    REQUIRE(Layout::logical_flat_to_physical_flat(4) == 2);
    REQUIRE(Layout::logical_flat_to_physical_flat(5) == 5);
    CHECK_THROWS(Layout::logical_flat_to_physical_flat(6)); // out-of-bounds
}

TEST_CASE("logical_flat_to_physical_flat: transposed with padding", "[layout][index][lf2pf][transpose][padding][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

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

// ============================================================================
// SECTION 8: logical_flat_to_logical_coords
// ============================================================================

TEST_CASE("logical_flat_to_logical_coords: 2D", "[layout][index][lf2lc][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy>;
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

TEST_CASE("logical_flat_to_logical_coords: transposed view", "[layout][index][lf2lc][transpose][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;
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

TEST_CASE("logical_flat_to_logical_coords: 3D", "[layout][index][lf2lc][3d][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy>;
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

TEST_CASE("physical_flat_to_physical_coords: no padding", "[layout][index][pf2pc][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix, no padding
     *
     * Physical flat:    Physical coords:
     *   0  1  2           (0,0)(0,1)(0,2)
     *   3  4  5           (1,0)(1,1)(1,2)
     *
     * BaseStrides = [3, 1]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy>;
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

TEST_CASE("physical_flat_to_physical_coords: with padding", "[layout][index][pf2pc][padding][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix with AVX padding (SW=4)
     * PhysicalDims = [2, 4]
     *
     * Physical memory:
     *   flat: 0  1  2  3  | 4  5  6  7
     *         ------P---    ------P---
     *
     * BaseStrides = [4, 1]
     *
     * flat 3 -> (0, 3)  <- this is padding!
     * flat 7 -> (1, 3)  <- this is padding!
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy>;
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

TEST_CASE("physical_flat_to_physical_coords: permutation doesn't affect result", "[layout][index][pf2pc][perm][strided_layout_constexpr]")
{
    /*
     * physical_flat_to_physical_coords uses BaseStrides (unpermuted)
     * so the result is always in physical (unpermuted) coordinate space
     *
     * Same physical flat should give same physical coords regardless of permutation
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using LayoutId = StridedLayoutConstExpr<Policy>;       // identity permutation
    using LayoutTr = StridedLayoutConstExpr<Policy, 1, 0>; // transposed permutation

    my_size_t coords_id[2], coords_tr[2];

    for (my_size_t flat = 0; flat < Policy::PhysicalSize; ++flat)
    {
        LayoutId::physical_flat_to_physical_coords(flat, coords_id);
        LayoutTr::physical_flat_to_physical_coords(flat, coords_tr);

        REQUIRE(coords_id[0] == coords_tr[0]);
        REQUIRE(coords_id[1] == coords_tr[1]);
    }
}

// ============================================================================
// SECTION 10: physical_flat_to_logical_coords
// ============================================================================

TEST_CASE("physical_flat_to_logical_coords: identity permutation", "[layout][index][pf2lc][strided_layout_constexpr]")
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
    using Layout = StridedLayoutConstExpr<Policy>;
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

TEST_CASE("physical_flat_to_logical_coords: transposed", "[layout][index][pf2lc][transpose][strided_layout_constexpr]")
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
     * PermArray = [1, 0]
     * logical[i] = physical[PermArray[i]]
     *
     * physical flat 1 -> physical coords (0,1)
     *   logical[0] = physical[PermArray[0]] = physical[1] = 1
     *   logical[1] = physical[PermArray[1]] = physical[0] = 0
     *   logical coords = (1, 0)
     *
     * physical flat 3 -> physical coords (1,0)
     *   logical[0] = physical[1] = 0
     *   logical[1] = physical[0] = 1
     *   logical coords = (0, 1)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;
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

TEST_CASE("physical_flat_to_logical_coords: padding gives out-of-bounds coords", "[layout][index][pf2lc][padding][strided_layout_constexpr]")
{
    /*
     * 2x3 matrix with AVX padding
     * PhysicalDims = [2, 4], LogicalDims = [2, 3]
     *
     * Physical memory:
     *   flat: 0  1  2  [3]  | 4  5  6  [7]
     *
     * physical flat 3 -> physical coords (0, 3)
     *   logical coords = (0, 3)  <- OUT OF BOUNDS (3 >= LogicalDims[1]=3)
     *
     * physical flat 7 -> physical coords (1, 3)
     *   logical coords = (1, 3)  <- OUT OF BOUNDS
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy>;
    my_size_t coords[2];

    // Padding location
    Layout::physical_flat_to_logical_coords(3, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 3); // >= LogicalDims[1], out of bounds!
    REQUIRE(coords[1] >= Layout::LogicalDims[1]);

    // Another padding location
    Layout::physical_flat_to_logical_coords(7, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 3); // >= LogicalDims[1], out of bounds!
}

TEST_CASE("physical_flat_to_logical_coords: 3D with permutation", "[layout][index][pf2lc][3d][strided_layout_constexpr]")
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
     *   BaseStrides = [12, 4, 1]
     *   physical(i,j,k) = i*12 + j*4 + k
     *
     * Permutation [2, 0, 1] means:
     *   logical dim 0 → physical dim 2 (size 4)
     *   logical dim 1 → physical dim 0 (size 2)
     *   logical dim 2 → physical dim 1 (size 3)
     *
     * Reverse mapping (physical coords → logical coords):
     *   logical[0] = physical[PermArray[0]] = physical[2]
     *   logical[1] = physical[PermArray[1]] = physical[0]
     *   logical[2] = physical[PermArray[2]] = physical[1]
     *
     * Examples:
     *   physical flat 0  → physical(0,0,0) → logical(0,0,0)
     *   physical flat 17 → physical(1,1,1) → logical(1,1,1)
     *   physical flat 23 → physical(1,2,3) → logical(3,1,2)
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3, 4>;
    using Layout = StridedLayoutConstExpr<Policy, 2, 0, 1>;
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

// // ============================================================================
// // SECTION 11: ROUNDTRIP INVARIANTS
// // ============================================================================

TEST_CASE("Roundtrip: logical_coords -> physical_flat -> logical_coords", "[layout][roundtrip][strided_layout_constexpr]")
{
    /*
     * For all valid logical coords:
     *   logical_coords -> physical_flat -> logical_coords
     * should return original coords
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;

    SECTION("Identity permutation")
    {
        using Layout = StridedLayoutConstExpr<Policy>;

        for (my_size_t i = 0; i < Layout::LogicalDims[0]; ++i)
        {
            for (my_size_t j = 0; j < Layout::LogicalDims[1]; ++j)
            {
                my_size_t physical = Layout::logical_coords_to_physical_flat(i, j);
                my_size_t result[2];
                Layout::physical_flat_to_logical_coords(physical, result);

                REQUIRE(result[0] == i);
                REQUIRE(result[1] == j);
            }
        }
    }

    SECTION("Transposed")
    {
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        for (my_size_t i = 0; i < Layout::LogicalDims[0]; ++i)
        {
            for (my_size_t j = 0; j < Layout::LogicalDims[1]; ++j)
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

TEST_CASE("Roundtrip: logical_flat -> logical_coords -> physical_flat -> logical_coords -> physical_flat", "[layout][roundtrip][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 4>;
    using Layout = StridedLayoutConstExpr<Policy, 1, 2, 0>;

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

TEST_CASE("Roundtrip: logical_flat -> physical_flat consistency", "[layout][roundtrip][strided_layout_constexpr]")
{
    /*
     * logical_flat_to_physical_flat should equal:
     *   logical_flat -> logical_coords -> physical_flat
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 3, 4>;

    SECTION("Identity")
    {
        using Layout = StridedLayoutConstExpr<Policy>;

        for (my_size_t lf = 0; lf < Layout::LogicalSize; ++lf)
        {
            my_size_t pf_direct = Layout::logical_flat_to_physical_flat(lf);

            my_size_t logical_coords[2];
            Layout::logical_flat_to_logical_coords(lf, logical_coords);
            my_size_t pf_indirect = Layout::logical_coords_to_physical_flat(logical_coords);

            REQUIRE(pf_direct == pf_indirect);
        }
    }

    SECTION("Transposed")
    {
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

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

TEST_CASE("is_logical_index_in_bounds", "[layout][bounds][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy>;

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

TEST_CASE("is_logical_index_in_bounds: respects permutation", "[layout][bounds][perm][strided_layout_constexpr]")
{
    /*
     * 2x3 transposed to 3x2
     * LogicalDims = [3, 2]
     */
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    my_size_t valid[] = {2, 1};   // valid for 3x2
    my_size_t invalid[] = {1, 2}; // col=2 out of bounds for 3x2

    REQUIRE(Layout::is_logical_index_in_bounds(valid));
    REQUIRE_FALSE(Layout::is_logical_index_in_bounds(invalid));
}

// ============================================================================
// SECTION 13: EDGE CASES
// ============================================================================

TEST_CASE("1D tensor (vector)", "[layout][edge][1d][strided_layout_constexpr]")
{
    /*
     * 1D tensor of size 5 with padding to 8
     *
     * Logical: [0] [1] [2] [3] [4]
     * Physical: 0   1   2   3   4  [5] [6] [7]
     */
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 5>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::NumDims == 1);
    REQUIRE(Layout::LogicalSize == 5);
    REQUIRE(Layout::PhysicalSize == 8);

    REQUIRE(Layout::BaseStrides[0] == 1);
    REQUIRE(Layout::Strides[0] == 1);
    REQUIRE(Layout::LogicalStrides[0] == 1);

    REQUIRE(Layout::logical_coords_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_coords_to_physical_flat(4) == 4);

    for (my_size_t i = 0; i < 5; ++i)
    {
        REQUIRE(Layout::logical_flat_to_physical_flat(i) == i);
    }

    CHECK_THROWS(Layout::logical_coords_to_physical_flat(5)); // out-of-bounds
    CHECK_THROWS(Layout::logical_flat_to_physical_flat(5));   // out-of-bounds
}

TEST_CASE("Single element 1D tensor", "[layout][edge][single][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 1>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::NumDims == 1);
    REQUIRE(Layout::LogicalSize == 1);
    REQUIRE(Layout::PhysicalSize == 4); // padded

    REQUIRE(Layout::logical_coords_to_physical_flat(0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
}

TEST_CASE("Single element 2D tensor", "[layout][edge][single][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 1, 1>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::NumDims == 2);
    REQUIRE(Layout::LogicalSize == 1);
    REQUIRE(Layout::PhysicalSize == 4); // padded

    REQUIRE(Layout::logical_coords_to_physical_flat(0, 0) == 0);
    REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
}

TEST_CASE("Large dimensions", "[layout][edge][large][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 100, 100>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::LogicalSize == 10000);
    REQUIRE(Layout::PhysicalSize == 10000); // 100 is already aligned to 4

    REQUIRE(Layout::logical_coords_to_physical_flat(99, 99) == 9999);
    REQUIRE(Layout::logical_flat_to_physical_flat(9999) == 9999);
}

TEST_CASE("High dimensional tensor", "[layout][edge][highdim][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 2, 2, 2, 2>;
    using Layout = StridedLayoutConstExpr<Policy>;

    REQUIRE(Layout::NumDims == 5);
    REQUIRE(Layout::LogicalSize == 32);
    REQUIRE(Layout::PhysicalSize == 32);

    // Check strides are powers of 2
    REQUIRE(Layout::BaseStrides[0] == 16);
    REQUIRE(Layout::BaseStrides[1] == 8);
    REQUIRE(Layout::BaseStrides[2] == 4);
    REQUIRE(Layout::BaseStrides[3] == 2);
    REQUIRE(Layout::BaseStrides[4] == 1);
}

// ============================================================================
// SECTION 14: COMPILE-TIME GUARANTEES
// ============================================================================

TEST_CASE("All members are constexpr", "[layout][constexpr][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

    // Use in array sizes (compile-time requirement)
    [[maybe_unused]] double arr1[Layout::NumDims];
    [[maybe_unused]] double arr2[Layout::LogicalSize];
    [[maybe_unused]] double arr3[Layout::PhysicalSize];

    // Static assertions
    static_assert(Layout::NumDims == 2);
    static_assert(Layout::LogicalSize == 6);
    static_assert(Layout::PhysicalSize == 8);
    static_assert(Layout::IsPermProvided == true);

    static_assert(Layout::PermArray[0] == 1);
    static_assert(Layout::PermArray[1] == 0);

    static_assert(Layout::LogicalDims[0] == 3);
    static_assert(Layout::LogicalDims[1] == 2);

    static_assert(Layout::BaseStrides[0] == 4);
    static_assert(Layout::BaseStrides[1] == 1);

    static_assert(Layout::Strides[0] == 1);
    static_assert(Layout::Strides[1] == 4);

    SUCCEED("All constexpr usages compiled successfully");
}

TEST_CASE("Index functions are constexpr", "[layout][constexpr][index][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, SCALAR, 2, 3>;
    using Layout = StridedLayoutConstExpr<Policy>;

    constexpr my_size_t flat1 = Layout::logical_coords_to_physical_flat(1, 2);
    static_assert(flat1 == 5);

    constexpr my_size_t flat2 = Layout::logical_flat_to_physical_flat(4);
    static_assert(flat2 == 4);

    SUCCEED("Index functions are constexpr");
}

// ============================================================================
// SECTION 15: COMPREHENSIVE PERMUTATION TESTS
// ============================================================================

TEST_CASE("All 2D permutations", "[layout][perm][2d][strided_layout_constexpr]")
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
        using Layout = StridedLayoutConstExpr<Policy>;

        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 1);
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 2);
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 3);
        REQUIRE(Layout::logical_flat_to_physical_flat(4) == 4);
        REQUIRE(Layout::logical_flat_to_physical_flat(5) == 5);
        REQUIRE(Layout::logical_flat_to_physical_flat(11) == 11);
    }

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
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

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

TEST_CASE("All 3D permutations", "[layout][perm][3d][strided_layout_constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 4>;

    SECTION("[0,1,2] identity")
    {
        using Layout = StridedLayoutConstExpr<Policy, 0, 1, 2>;
        REQUIRE(Layout::LogicalDims[0] == 2);
        REQUIRE(Layout::LogicalDims[1] == 3);
        REQUIRE(Layout::LogicalDims[2] == 4);
    }

    SECTION("[0,2,1]")
    {
        using Layout = StridedLayoutConstExpr<Policy, 0, 2, 1>;
        REQUIRE(Layout::LogicalDims[0] == 2);
        REQUIRE(Layout::LogicalDims[1] == 4);
        REQUIRE(Layout::LogicalDims[2] == 3);
    }

    SECTION("[1,0,2]")
    {
        using Layout = StridedLayoutConstExpr<Policy, 1, 0, 2>;
        REQUIRE(Layout::LogicalDims[0] == 3);
        REQUIRE(Layout::LogicalDims[1] == 2);
        REQUIRE(Layout::LogicalDims[2] == 4);
    }

    SECTION("[1,2,0]")
    {
        using Layout = StridedLayoutConstExpr<Policy, 1, 2, 0>;
        REQUIRE(Layout::LogicalDims[0] == 3);
        REQUIRE(Layout::LogicalDims[1] == 4);
        REQUIRE(Layout::LogicalDims[2] == 2);
    }

    SECTION("[2,0,1]")
    {
        using Layout = StridedLayoutConstExpr<Policy, 2, 0, 1>;
        REQUIRE(Layout::LogicalDims[0] == 4);
        REQUIRE(Layout::LogicalDims[1] == 2);
        REQUIRE(Layout::LogicalDims[2] == 3);
    }

    SECTION("[2,1,0] reverse")
    {
        using Layout = StridedLayoutConstExpr<Policy, 2, 1, 0>;
        REQUIRE(Layout::LogicalDims[0] == 4);
        REQUIRE(Layout::LogicalDims[1] == 3);
        REQUIRE(Layout::LogicalDims[2] == 2);
    }
}

// ============================================================================
// SECTION 16: SIMD WIDTH VARIATIONS
// ============================================================================

TEST_CASE("Layout works with all SIMD widths", "[layout][simd][strided_layout_constexpr]")
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
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Layout::PhysicalSize == 15);
        REQUIRE(Layout::LogicalDims[0] == 5);
        REQUIRE(Layout::LogicalDims[1] == 3);
        REQUIRE(Layout::BaseStrides[0] == 5);
        REQUIRE(Layout::BaseStrides[1] == 1);

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
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Layout::PhysicalSize == 18); // 3 * 6
        REQUIRE(Layout::BaseStrides[0] == 6);
        REQUIRE(Layout::BaseStrides[1] == 1);

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
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Layout::PhysicalSize == 24); // 3 * 8
        REQUIRE(Layout::BaseStrides[0] == 8);
        REQUIRE(Layout::BaseStrides[1] == 1);

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
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Layout::PhysicalSize == 24); // 3 * 8
        REQUIRE(Layout::BaseStrides[0] == 8);
        REQUIRE(Layout::BaseStrides[1] == 1);

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
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Layout::PhysicalSize == 24); // 3 * 8
        REQUIRE(Layout::BaseStrides[0] == 8);
        REQUIRE(Layout::BaseStrides[1] == 1);

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
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Layout::PhysicalSize == 24); // 3 * 8
        REQUIRE(Layout::BaseStrides[0] == 8);
        REQUIRE(Layout::BaseStrides[1] == 1);

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
        using Layout = StridedLayoutConstExpr<Policy, 1, 0>;

        REQUIRE(Layout::LogicalSize == 15);
        REQUIRE(Layout::PhysicalSize == 48); // 3 * 16
        REQUIRE(Layout::BaseStrides[0] == 16);
        REQUIRE(Layout::BaseStrides[1] == 1);

        // Transposed: logical(i,j) -> physical j*16 + i
        REQUIRE(Layout::logical_flat_to_physical_flat(0) == 0);   // (0,0) -> 0
        REQUIRE(Layout::logical_flat_to_physical_flat(1) == 16);  // (0,1) -> 16
        REQUIRE(Layout::logical_flat_to_physical_flat(2) == 32);  // (0,2) -> 32
        REQUIRE(Layout::logical_flat_to_physical_flat(3) == 1);   // (1,0) -> 1
        REQUIRE(Layout::logical_flat_to_physical_flat(14) == 36); // (4,2) -> 36
    }
}