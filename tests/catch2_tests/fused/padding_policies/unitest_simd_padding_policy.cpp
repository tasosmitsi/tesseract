#include <catch_amalgamated.hpp>
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
using SSEDoubleWidth = SimdWidthTag<SSE_DOUBLE>;
using SSEFloatWidth = SimdWidthTag<SSE_FLOAT>;
using AVXDoubleWidth = SimdWidthTag<AVX_DOUBLE>;
using AVXFloatWidth = SimdWidthTag<AVX_FLOAT>;
using AVX512DoubleWidth = SimdWidthTag<AVX512_DOUBLE>;
using AVX512FloatWidth = SimdWidthTag<AVX512_FLOAT>;

// ############################################################################
//                              TEST CASES
// ############################################################################

// ============================================================================
// SECTION 1: BASIC PROPERTIES
// ============================================================================

TEST_CASE("NumDims equals parameter pack size", "[padding][dims]")
{
    REQUIRE(SimdPaddingPolicyBase<double, AVX_DOUBLE, 10>::NumDims == 1);
    REQUIRE(SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>::NumDims == 2);
    REQUIRE(SimdPaddingPolicyBase<float, AVX_FLOAT, 2, 3, 4>::NumDims == 3);
    REQUIRE(SimdPaddingPolicyBase<float, AVX_FLOAT, 2, 3, 4, 5>::NumDims == 4);
    REQUIRE(SimdPaddingPolicyBase<double, SCALAR, 2, 2, 2, 2, 2, 2, 2>::NumDims == 7);
}

TEST_CASE("LogicalDims stores original dimensions in order", "[padding][dims]")
{
    SECTION("2D matrix 8x6")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>;
        REQUIRE(Policy::LogicalDims[0] == 8);
        REQUIRE(Policy::LogicalDims[1] == 6);
    }

    SECTION("3D tensor 2x3x5")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX_FLOAT, 2, 3, 5>;
        REQUIRE(Policy::LogicalDims[0] == 2);
        REQUIRE(Policy::LogicalDims[1] == 3);
        REQUIRE(Policy::LogicalDims[2] == 5);
    }

    SECTION("4D tensor preserves order")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX_FLOAT, 7, 11, 13, 17>;
        REQUIRE(Policy::LogicalDims[0] == 7);
        REQUIRE(Policy::LogicalDims[1] == 11);
        REQUIRE(Policy::LogicalDims[2] == 13);
        REQUIRE(Policy::LogicalDims[3] == 17);
    }
}

TEMPLATE_TEST_CASE("SimdWidth is correctly stored", "[padding][simd]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;
    using Policy = SimdPaddingPolicyBase<double, SW, 8, 6>;
    REQUIRE(Policy::SimdWidth == SW);
}

// ============================================================================
// SECTION 2: PADDING FUNCTION
// ============================================================================

TEST_CASE("pad() rounds up to SimdWidth multiples", "[padding][pad]")
{
    SECTION("SimdWidth=4")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>;

        REQUIRE(Policy::pad(1) == 4);
        REQUIRE(Policy::pad(2) == 4);
        REQUIRE(Policy::pad(3) == 4);
        REQUIRE(Policy::pad(4) == 4);
        REQUIRE(Policy::pad(5) == 8);
        REQUIRE(Policy::pad(6) == 8);
        REQUIRE(Policy::pad(7) == 8);
        REQUIRE(Policy::pad(8) == 8);
        REQUIRE(Policy::pad(9) == 12);
        REQUIRE(Policy::pad(100) == 100);
    }

    SECTION("SimdWidth=8")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX_FLOAT, 8, 6>;

        REQUIRE(Policy::pad(1) == 8);
        REQUIRE(Policy::pad(5) == 8);
        REQUIRE(Policy::pad(7) == 8);
        REQUIRE(Policy::pad(8) == 8);
        REQUIRE(Policy::pad(9) == 16);
        REQUIRE(Policy::pad(15) == 16);
        REQUIRE(Policy::pad(16) == 16);
        REQUIRE(Policy::pad(17) == 24);
    }

    SECTION("SimdWidth=1 (no padding)")
    {
        using Policy = SimdPaddingPolicyBase<double, SCALAR, 8, 6>;

        REQUIRE(Policy::pad(1) == 1);
        REQUIRE(Policy::pad(5) == 5);
        REQUIRE(Policy::pad(6) == 6);
        REQUIRE(Policy::pad(100) == 100);
    }
}

TEST_CASE("pad() edge cases", "[padding][pad][edge]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>;

    REQUIRE(Policy::pad(0) == 0);
    REQUIRE(Policy::pad(1000) == 1000);
    REQUIRE(Policy::pad(1001) == 1004);
    REQUIRE(Policy::pad(1023) == 1024);
    REQUIRE(Policy::pad(1024) == 1024);
}

// ============================================================================
// SECTION 3: LAST DIMENSION PROPERTIES
// ============================================================================

TEST_CASE("LastDim is the final logical dimension", "[padding][lastdim]")
{
    REQUIRE(SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>::LastDim == 6);
    REQUIRE(SimdPaddingPolicyBase<float, AVX_FLOAT, 2, 3, 5>::LastDim == 5);
    REQUIRE(SimdPaddingPolicyBase<double, SCALAR, 42>::LastDim == 42);
    REQUIRE(SimdPaddingPolicyBase<float, SSE_FLOAT, 2, 3, 4, 7>::LastDim == 7);
}

TEMPLATE_TEST_CASE("PaddedLastDim is correctly computed", "[padding][lastdim]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    SECTION("LastDim=6")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 8, 6>;
        constexpr my_size_t expected = ((6 + SW - 1) / SW) * SW;
        REQUIRE(Policy::PaddedLastDim == expected);
    }

    SECTION("LastDim=5")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 2, 3, 5>;
        constexpr my_size_t expected = ((5 + SW - 1) / SW) * SW;
        REQUIRE(Policy::PaddedLastDim == expected);
    }

    SECTION("LastDim already aligned")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 8, SW>;
        REQUIRE(Policy::PaddedLastDim == SW);
    }
}

// ============================================================================
// SECTION 4: SIZE CALCULATIONS
// ============================================================================

TEST_CASE("LogicalSize is product of dimensions", "[padding][size]")
{
    REQUIRE(SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>::LogicalSize == 48);
    REQUIRE(SimdPaddingPolicyBase<float, AVX_FLOAT, 2, 3, 5>::LogicalSize == 30);
    REQUIRE(SimdPaddingPolicyBase<double, SCALAR, 100>::LogicalSize == 100);
    REQUIRE(SimdPaddingPolicyBase<float, SSE_FLOAT, 2, 3, 4, 5>::LogicalSize == 120);
    REQUIRE(SimdPaddingPolicyBase<double, AVX_DOUBLE, 100, 100>::LogicalSize == 10000);
}

TEMPLATE_TEST_CASE("PhysicalSize accounts for padding", "[padding][size]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    SECTION("2D tensor 8x6")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 8, 6>;
        constexpr my_size_t paddedLast = ((6 + SW - 1) / SW) * SW;
        REQUIRE(Policy::PhysicalSize == 8 * paddedLast);
    }

    SECTION("3D tensor 2x3x5")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 2, 3, 5>;
        constexpr my_size_t paddedLast = ((5 + SW - 1) / SW) * SW;
        REQUIRE(Policy::PhysicalSize == 2 * 3 * paddedLast);
    }

    SECTION("1D tensor")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 6>;
        constexpr my_size_t paddedLast = ((6 + SW - 1) / SW) * SW;
        REQUIRE(Policy::PhysicalSize == paddedLast);
    }
}

TEMPLATE_TEST_CASE("PhysicalSize >= LogicalSize invariant", "[padding][size][invariant]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    REQUIRE(SimdPaddingPolicyBase<double, SW, 8, 6>::PhysicalSize >=
            SimdPaddingPolicyBase<double, SW, 8, 6>::LogicalSize);

    REQUIRE(SimdPaddingPolicyBase<double, SW, 5, 10>::PhysicalSize >=
            SimdPaddingPolicyBase<double, SW, 5, 10>::LogicalSize);

    REQUIRE(SimdPaddingPolicyBase<double, SW, 2, 3, 5>::PhysicalSize >=
            SimdPaddingPolicyBase<double, SW, 2, 3, 5>::LogicalSize);

    REQUIRE(SimdPaddingPolicyBase<double, SW, 1, 1, 1>::PhysicalSize >=
            SimdPaddingPolicyBase<double, SW, 1, 1, 1>::LogicalSize);
}

TEMPLATE_TEST_CASE("Zero overhead when already aligned", "[padding][size][overhead]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    using Policy1 = SimdPaddingPolicyBase<double, SW, 8, SW>;
    REQUIRE(Policy1::PhysicalSize == Policy1::LogicalSize);

    using Policy2 = SimdPaddingPolicyBase<double, SW, 8, SW * 2>;
    REQUIRE(Policy2::PhysicalSize == Policy2::LogicalSize);

    using Policy3 = SimdPaddingPolicyBase<double, SW, 10, SW * 10>;
    REQUIRE(Policy3::PhysicalSize == Policy3::LogicalSize);
}

TEST_CASE("SCALAR SimdWidth means zero overhead always", "[padding][size][scalar]")
{
    using Policy1 = SimdPaddingPolicyBase<double, SCALAR, 8, 6>;
    REQUIRE(Policy1::PhysicalSize == Policy1::LogicalSize);

    using Policy2 = SimdPaddingPolicyBase<double, SCALAR, 7, 11, 13>;
    REQUIRE(Policy2::PhysicalSize == Policy2::LogicalSize);

    using Policy3 = SimdPaddingPolicyBase<float, SCALAR, 100, 1>;
    REQUIRE(Policy3::PhysicalSize == Policy3::LogicalSize);
}

// ============================================================================
// SECTION 5: PHYSICAL DIMENSIONS
// ============================================================================

TEST_CASE("PhysicalDims only pads last dimension", "[padding][physdims]")
{
    SECTION("2D")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>;
        REQUIRE(Policy::PhysicalDims[0] == 8);
        REQUIRE(Policy::PhysicalDims[1] == 8); // 6 -> 8
    }

    SECTION("3D")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX_FLOAT, 2, 3, 5>;
        REQUIRE(Policy::PhysicalDims[0] == 2);
        REQUIRE(Policy::PhysicalDims[1] == 3);
        REQUIRE(Policy::PhysicalDims[2] == 8); // 5 -> 8
    }

    SECTION("4D")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 2, 3, 4, 5>;
        REQUIRE(Policy::PhysicalDims[0] == 2);
        REQUIRE(Policy::PhysicalDims[1] == 3);
        REQUIRE(Policy::PhysicalDims[2] == 4);
        REQUIRE(Policy::PhysicalDims[3] == 8); // 5 -> 8
    }

    SECTION("1D")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 6>;
        REQUIRE(Policy::PhysicalDims[0] == 8); // 6 -> 8
    }
}

TEMPLATE_TEST_CASE("PhysicalDims product equals PhysicalSize", "[padding][physdims][invariant]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    SECTION("2D")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 8, 6>;
        constexpr my_size_t product = Policy::PhysicalDims[0] * Policy::PhysicalDims[1];
        REQUIRE(product == Policy::PhysicalSize);
    }

    SECTION("3D")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 2, 3, 5>;
        constexpr my_size_t product =
            Policy::PhysicalDims[0] * Policy::PhysicalDims[1] * Policy::PhysicalDims[2];
        REQUIRE(product == Policy::PhysicalSize);
    }

    SECTION("4D")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 2, 3, 4, 5>;
        constexpr my_size_t product =
            Policy::PhysicalDims[0] * Policy::PhysicalDims[1] *
            Policy::PhysicalDims[2] * Policy::PhysicalDims[3];
        REQUIRE(product == Policy::PhysicalSize);
    }
}

TEMPLATE_TEST_CASE("Non-last PhysicalDims equal LogicalDims", "[padding][physdims]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    SECTION("2D")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 8, 6>;
        REQUIRE(Policy::PhysicalDims[0] == Policy::LogicalDims[0]);
    }

    SECTION("3D")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 2, 3, 5>;
        REQUIRE(Policy::PhysicalDims[0] == Policy::LogicalDims[0]);
        REQUIRE(Policy::PhysicalDims[1] == Policy::LogicalDims[1]);
    }

    SECTION("5D")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 2, 3, 4, 5, 6>;
        REQUIRE(Policy::PhysicalDims[0] == Policy::LogicalDims[0]);
        REQUIRE(Policy::PhysicalDims[1] == Policy::LogicalDims[1]);
        REQUIRE(Policy::PhysicalDims[2] == Policy::LogicalDims[2]);
        REQUIRE(Policy::PhysicalDims[3] == Policy::LogicalDims[3]);
    }
}

// ============================================================================
// SECTION 6: EDGE CASES
// ============================================================================

TEMPLATE_TEST_CASE("1D tensors (vectors)", "[padding][edge][1d]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    SECTION("Small vector needing padding")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 5>;
        REQUIRE(Policy::NumDims == 1);
        REQUIRE(Policy::LogicalSize == 5);
        REQUIRE(Policy::PhysicalSize == Policy::PaddedLastDim);
    }

    SECTION("Single element vector")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 1>;
        REQUIRE(Policy::LogicalSize == 1);
        REQUIRE(Policy::PaddedLastDim == SW);
        REQUIRE(Policy::PhysicalSize == SW);
    }
}

TEMPLATE_TEST_CASE("Minimum dimensions", "[padding][edge][min]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    SECTION("1x1 matrix")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 1, 1>;
        REQUIRE(Policy::LogicalSize == 1);
        REQUIRE(Policy::PaddedLastDim == SW);
        REQUIRE(Policy::PhysicalSize == SW);
    }

    SECTION("1x1x1 3D tensor")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 1, 1, 1>;
        REQUIRE(Policy::LogicalSize == 1);
        REQUIRE(Policy::PhysicalSize == SW);
    }

    SECTION("Tall skinny matrix (100x1)")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 100, 1>;
        REQUIRE(Policy::LogicalSize == 100);
        REQUIRE(Policy::PaddedLastDim == SW);
        REQUIRE(Policy::PhysicalSize == 100 * SW);
    }
}

TEST_CASE("Large dimensions", "[padding][edge][large]")
{
    SECTION("Large aligned (double, SW=4)")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 1000, 1024>;
        REQUIRE(Policy::PaddedLastDim == 1024);
        REQUIRE(Policy::PhysicalSize == Policy::LogicalSize);
    }

    SECTION("Large unaligned (double, SW=4)")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 1000, 1023>;
        REQUIRE(Policy::PaddedLastDim == 1024);
        REQUIRE(Policy::PhysicalSize == 1024000);
    }

    SECTION("Large aligned (float, SW=8)")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX_FLOAT, 1000, 1024>;
        REQUIRE(Policy::PaddedLastDim == 1024);
        REQUIRE(Policy::PhysicalSize == Policy::LogicalSize);
    }
}

TEMPLATE_TEST_CASE("Prime number dimensions", "[padding][edge][prime]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    SECTION("7x11 matrix")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 7, 11>;
        REQUIRE(Policy::LogicalSize == 77);
        REQUIRE(Policy::PaddedLastDim == ((11 + SW - 1) / SW) * SW);
    }

    SECTION("3D tensor with prime dimensions")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 5, 7, 11>;
        REQUIRE(Policy::LogicalSize == 385);
        REQUIRE(Policy::PaddedLastDim == ((11 + SW - 1) / SW) * SW);
    }
}

TEMPLATE_TEST_CASE("Dimensions around SimdWidth boundary", "[padding][edge][boundary]",
                   SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;

    SECTION("Last dim exactly SimdWidth")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 10, SW>;
        REQUIRE(Policy::PaddedLastDim == SW);
        REQUIRE(Policy::PhysicalSize == Policy::LogicalSize);
    }

    SECTION("Last dim one less than SimdWidth")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 10, SW - 1>;
        REQUIRE(Policy::PaddedLastDim == SW);
    }

    SECTION("Last dim one more than SimdWidth")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 10, SW + 1>;
        REQUIRE(Policy::PaddedLastDim == 2 * SW);
    }

    SECTION("Last dim exactly 2x SimdWidth")
    {
        using Policy = SimdPaddingPolicyBase<double, SW, 10, 2 * SW>;
        REQUIRE(Policy::PaddedLastDim == 2 * SW);
        REQUIRE(Policy::PhysicalSize == Policy::LogicalSize);
    }
}

// ============================================================================
// SECTION 7: COMPILE-TIME GUARANTEES
// ============================================================================

TEST_CASE("All members are constexpr", "[padding][constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>;

    [[maybe_unused]] double arr1[Policy::PhysicalSize];
    [[maybe_unused]] double arr2[Policy::LogicalSize];
    [[maybe_unused]] double arr3[Policy::SimdWidth];
    [[maybe_unused]] double arr4[Policy::NumDims];
    [[maybe_unused]] double arr5[Policy::PaddedLastDim];
    [[maybe_unused]] double arr6[Policy::LastDim];

    [[maybe_unused]] Array<double, Policy::PhysicalSize> container;

    static_assert(Policy::PhysicalSize == 64);
    static_assert(Policy::LogicalSize == 48);
    static_assert(Policy::SimdWidth == 4);

    SUCCEED("All constexpr usages compiled successfully");
}

TEST_CASE("pad() is constexpr", "[padding][constexpr]")
{
    using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 6>;

    [[maybe_unused]] Array<double, Policy::pad(5)> arr;

    static_assert(Policy::pad(6) == 8);
    static_assert(Policy::pad(4) == 4);

    constexpr my_size_t padded = Policy::pad(10);
    static_assert(padded == 12);

    SUCCEED("pad() is fully constexpr");
}

// ============================================================================
// SECTION 8: ALIGNMENT VERIFICATION
// ============================================================================

TEMPLATE_TEST_CASE("Row starts are SIMD-aligned (element indices)", "[padding][alignment]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;
    using Policy = SimdPaddingPolicyBase<double, SW, 8, 6>;

    constexpr my_size_t M = Policy::LogicalDims[0];
    constexpr my_size_t stride = Policy::PaddedLastDim;

    REQUIRE(stride % Policy::SimdWidth == 0);

    for (my_size_t row = 0; row < M; ++row)
    {
        my_size_t row_start = row * stride;
        REQUIRE(row_start % Policy::SimdWidth == 0);
    }
}

TEMPLATE_TEST_CASE("3D tensor slice alignment", "[padding][alignment]",
                   ScalarWidth, SSEDoubleWidth, SSEFloatWidth,
                   AVXDoubleWidth, AVXFloatWidth, AVX512FloatWidth,
                   AVX512DoubleWidth)
{
    constexpr my_size_t SW = TestType::value;
    using Policy = SimdPaddingPolicyBase<double, SW, 2, 3, 5>;

    constexpr my_size_t D0 = Policy::LogicalDims[0];
    constexpr my_size_t D1 = Policy::LogicalDims[1];
    constexpr my_size_t PaddedD2 = Policy::PaddedLastDim;

    REQUIRE(PaddedD2 % Policy::SimdWidth == 0);

    for (my_size_t i = 0; i < D0; ++i)
    {
        for (my_size_t j = 0; j < D1; ++j)
        {
            my_size_t row_start = i * D1 * PaddedD2 + j * PaddedD2;
            REQUIRE(row_start % Policy::SimdWidth == 0);
        }
    }
}

// ============================================================================
// SECTION 9: MEMORY OVERHEAD ANALYSIS
// ============================================================================

TEST_CASE("Memory overhead: Worst case (LastDim=1)", "[padding][overhead]")
{
    SECTION("SimdWidth=4: 300% overhead")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 100, 1>;
        REQUIRE(Policy::LogicalSize == 100);
        REQUIRE(Policy::PhysicalSize == 400);
    }

    SECTION("SimdWidth=8: 700% overhead")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX_FLOAT, 100, 1>;
        REQUIRE(Policy::LogicalSize == 100);
        REQUIRE(Policy::PhysicalSize == 800);
    }

    SECTION("SimdWidth=16: 1500% overhead")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX512_FLOAT, 100, 1>;
        REQUIRE(Policy::LogicalSize == 100);
        REQUIRE(Policy::PhysicalSize == 1600);
    }
}

TEST_CASE("Memory overhead: Best case (zero)", "[padding][overhead]")
{
    SECTION("SimdWidth=4, LastDim divisible by 4")
    {
        using Policy = SimdPaddingPolicyBase<double, AVX_DOUBLE, 8, 4>;
        REQUIRE(Policy::PhysicalSize == Policy::LogicalSize);
    }

    SECTION("SimdWidth=8, LastDim divisible by 8")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX_FLOAT, 8, 8>;
        REQUIRE(Policy::PhysicalSize == Policy::LogicalSize);
    }

    SECTION("SimdWidth=16, LastDim divisible by 16")
    {
        using Policy = SimdPaddingPolicyBase<float, AVX512_FLOAT, 10, 64>;
        REQUIRE(Policy::PhysicalSize == Policy::LogicalSize);
    }
}

// ============================================================================
// SECTION 10: PRODUCTION ALIAS TEST
// ============================================================================

TEST_CASE("SimdPaddingPolicy alias works correctly", "[padding][alias]")
{
    using Policy = SimdPaddingPolicy<double, 8, 6>;

    REQUIRE(Policy::NumDims == 2);
    REQUIRE(Policy::LogicalDims[0] == 8);
    REQUIRE(Policy::LogicalDims[1] == 6);
    REQUIRE(Policy::LogicalSize == 48);
    REQUIRE(Policy::SimdWidth >= 1);
    REQUIRE(Policy::PhysicalSize >= Policy::LogicalSize);
    REQUIRE(Policy::PaddedLastDim >= Policy::LastDim);
    REQUIRE(Policy::PaddedLastDim % Policy::SimdWidth == 0);
}