#include <catch_amalgamated.hpp>

#include "fused/layouts/strided_layout_base.h"
#include "layout_two_way.h"

// ============================================================================
// StridedLayoutBase on its own. A strided layout is four values, so these tests
// write them by hand: no padding policy, no StridedLayout, no adapter. A bug in
// a real layout can't fail this file, and if the base starts reading anything
// beyond those four values, this file stops compiling.
//
// The round trips over the real layouts live with them, in
// unitest_strided_layout.cpp and unitest_layout_adapters.cpp.
// ============================================================================

template <my_size_t... Values>
struct DimList
{
};

template <my_size_t... Values>
struct StrideList
{
};

template <typename D, typename S, my_size_t Start = 0>
struct Written;

template <my_size_t... D, my_size_t... S, my_size_t Start>
struct Written<DimList<D...>, StrideList<S...>, Start>
    : StridedLayoutBase<Written<DimList<D...>, StrideList<S...>, Start>, sizeof...(D)>
{
    static constexpr my_size_t NumDims = sizeof...(D);
    static constexpr Array<my_size_t, NumDims> LogicalDims{D...};
    static constexpr Array<my_size_t, NumDims> Strides{S...};
    static constexpr my_size_t Base = Start;
};

/*
 * Most layouts below describe the same buffer, 2 rows of 3 with a gap after each:
 *
 *   flat:  0  1  2  P | 4  5  6  P
 */
using Padded = Written<DimList<2, 3>, StrideList<4, 1>>;
using Column = Written<DimList<2, 1>, StrideList<4, 1>, 1>; // offsets 1, 5
using Row = Written<DimList<1, 3>, StrideList<4, 1>, 4>;    // offsets 4, 5, 6
using Transposed = Written<DimList<3, 2>, StrideList<1, 4>>;

using Dense = Written<DimList<2, 3>, StrideList<3, 1>>;      // no gaps
using Diagonal = Written<DimList<3, 1>, StrideList<5, 1>>;   // offsets 0, 5, 10
using Scrambled = Written<DimList<5, 2, 3>, StrideList<1, 24, 8>>; // largest stride in the middle

// Axes 0 and 1 share a stride. The axis of extent 1 can come first or second.
using TiedFirst = Written<DimList<1, 3, 4>, StrideList<4, 4, 1>>;
using TiedSecond = Written<DimList<3, 1, 4>, StrideList<4, 4, 1>>;

// ============================================================================
// STRIDES AND CONTIGUITY
// ============================================================================

TEST_CASE("strided_layout_base: stride reads the derived strides", "[strided_layout_base]")
{
    REQUIRE(Transposed::stride(0) == 1);
    REQUIRE(Transposed::stride(1) == 4);
    CHECK_THROWS(Transposed::stride(2)); // out-of-bounds
}

TEST_CASE("strided_layout_base: contiguity", "[strided_layout_base]")
{
    STATIC_REQUIRE(Dense::is_contiguous());
    STATIC_REQUIRE(Row::is_contiguous()); // axis 0 has extent 1, so its stride is never used

    STATIC_REQUIRE_FALSE(Padded::is_contiguous());
    STATIC_REQUIRE_FALSE(Column::is_contiguous());
    STATIC_REQUIRE_FALSE(Transposed::is_contiguous());
}

// ============================================================================
// LOGICAL TO PHYSICAL
// ============================================================================

TEST_CASE("strided_layout_base: coords to offset is Base plus coords times strides", "[strided_layout_base]")
{
    REQUIRE(Padded::logical_coords_to_physical_flat(1, 2) == 6);
    REQUIRE(Column::logical_coords_to_physical_flat(1, 0) == 5); // 1 + 1*4
    REQUIRE(Row::logical_coords_to_physical_flat(0, 2) == 6);    // 4 + 2*1
    REQUIRE(Transposed::logical_coords_to_physical_flat(2, 1) == 6);

    const my_size_t coords[2] = {1, 2};
    REQUIRE(Padded::logical_coords_to_physical_flat(coords) == 6);

    CHECK_THROWS(Padded::logical_coords_to_physical_flat(2, 0)); // out-of-bounds
    CHECK_THROWS(Padded::logical_coords_to_physical_flat(0, 3)); // out-of-bounds
    CHECK_THROWS(Column::logical_coords_to_physical_flat(0, 1)); // out-of-bounds
}

TEST_CASE("strided_layout_base: flat to offset walks the logical order", "[strided_layout_base]")
{
    REQUIRE(Column::logical_flat_to_physical_flat(0) == 1);
    REQUIRE(Column::logical_flat_to_physical_flat(1) == 5);

    REQUIRE(Transposed::logical_flat_to_physical_flat(0) == 0);
    REQUIRE(Transposed::logical_flat_to_physical_flat(1) == 4);
    REQUIRE(Transposed::logical_flat_to_physical_flat(2) == 1);
    REQUIRE(Transposed::logical_flat_to_physical_flat(3) == 5);
    REQUIRE(Transposed::logical_flat_to_physical_flat(4) == 2);
    REQUIRE(Transposed::logical_flat_to_physical_flat(5) == 6);

    CHECK_THROWS(Column::logical_flat_to_physical_flat(2)); // out-of-bounds
}

// ============================================================================
// PHYSICAL TO LOGICAL
// ============================================================================

TEST_CASE("strided_layout_base: offset to coords undoes the forward map", "[strided_layout_base]")
{
    my_size_t coords[2];

    Padded::physical_flat_to_logical_coords(6, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 2);

    Column::physical_flat_to_logical_coords(5, coords);
    REQUIRE(coords[0] == 1);
    REQUIRE(coords[1] == 0);

    Transposed::physical_flat_to_logical_coords(6, coords);
    REQUIRE(coords[0] == 2);
    REQUIRE(coords[1] == 1);
}

TEST_CASE("strided_layout_base: strides can be stored in any order", "[strided_layout_base]")
{
    // 44 = 4*1 + 1*24 + 2*8, and the largest stride is the middle axis
    my_size_t coords[3];

    Scrambled::physical_flat_to_logical_coords(44, coords);
    REQUIRE(coords[0] == 4);
    REQUIRE(coords[1] == 1);
    REQUIRE(coords[2] == 2);
}

TEST_CASE("strided_layout_base: an axis of extent 1 can tie on stride", "[strided_layout_base]")
{
    my_size_t coords[3];

    // 9 = 2*4 + 1. Dividing by axis 0's stride first would take the 2 for a
    // coordinate that can only be 0.
    TiedFirst::physical_flat_to_logical_coords(9, coords);
    REQUIRE(coords[0] == 0);
    REQUIRE(coords[1] == 2);
    REQUIRE(coords[2] == 1);

    TiedSecond::physical_flat_to_logical_coords(9, coords);
    REQUIRE(coords[0] == 2);
    REQUIRE(coords[1] == 0);
    REQUIRE(coords[2] == 1);
}

TEST_CASE("strided_layout_base: offset to flat undoes the forward map", "[strided_layout_base]")
{
    REQUIRE(Column::physical_flat_to_logical_flat(5) == 1);
    REQUIRE(Transposed::physical_flat_to_logical_flat(5) == 3);
    REQUIRE(Diagonal::physical_flat_to_logical_flat(10) == 2);
}

// ============================================================================
// OFFSETS THAT HOLD NO ELEMENT
// ============================================================================

TEST_CASE("strided_layout_base: the check tells elements from everything else", "[strided_layout_base]")
{
    SECTION("a gap")
    {
        REQUIRE(Padded::is_physical_flat_in_layout(2));
        REQUIRE_FALSE(Padded::is_physical_flat_in_layout(3));
        REQUIRE(Padded::is_physical_flat_in_layout(4));
        REQUIRE_FALSE(Padded::is_physical_flat_in_layout(7));
    }

    SECTION("past the end")
    {
        REQUIRE_FALSE(Padded::is_physical_flat_in_layout(8));
        REQUIRE_FALSE(Diagonal::is_physical_flat_in_layout(15));
    }

    SECTION("before Base")
    {
        REQUIRE_FALSE(Column::is_physical_flat_in_layout(0));
        REQUIRE_FALSE(Row::is_physical_flat_in_layout(3));
    }

    SECTION("between a view's elements")
    {
        REQUIRE(Column::is_physical_flat_in_layout(1));
        REQUIRE_FALSE(Column::is_physical_flat_in_layout(2));
        REQUIRE(Column::is_physical_flat_in_layout(5));
        REQUIRE_FALSE(Diagonal::is_physical_flat_in_layout(4));
    }
}

TEST_CASE("strided_layout_base: the conversions report an offset with no element", "[strided_layout_base]")
{
    my_size_t coords[2];

    CHECK_THROWS(Padded::physical_flat_to_logical_coords(3, coords));
    CHECK_THROWS(Column::physical_flat_to_logical_coords(0, coords));
    CHECK_THROWS(Column::physical_flat_to_logical_coords(2, coords));

    CHECK_THROWS(Padded::physical_flat_to_logical_flat(7));
    CHECK_THROWS(Column::physical_flat_to_logical_flat(0));
}

// ============================================================================
// EVERY LAYOUT ABOVE, BOTH WAYS
// ============================================================================

TEMPLATE_TEST_CASE("strided_layout_base: round trips", "[strided_layout_base]",
                   Padded, Column, Row, Transposed, Dense, Diagonal, Scrambled, TiedFirst, TiedSecond)
{
    check_two_way<TestType>(64); // larger than every offset above
}

// ============================================================================
// COMPILE TIME
// ============================================================================

TEST_CASE("strided_layout_base: the reverse direction is constexpr", "[strided_layout_base]")
{
    STATIC_REQUIRE(Padded::is_physical_flat_in_layout(6));
    STATIC_REQUIRE_FALSE(Padded::is_physical_flat_in_layout(7));
    STATIC_REQUIRE(Padded::physical_flat_to_logical_flat(6) == 5);
    STATIC_REQUIRE(Transposed::physical_flat_to_logical_flat(5) == 3);
}
