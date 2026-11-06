#include <catch_amalgamated.hpp>

#include "fused/layouts/layout_base.h"

// ============================================================================
// LayoutBase needs nothing but dims. This layout has no strides, no Base and
// no padding policy, so these tests also hold that line: if LayoutBase starts
// reading anything else, this file stops compiling.
// ============================================================================

template <my_size_t... Dims>
struct ShapeOnly : LayoutBase<ShapeOnly<Dims...>, sizeof...(Dims)>
{
    static constexpr my_size_t NumDims = sizeof...(Dims);
    static constexpr Array<my_size_t, NumDims> LogicalDims{Dims...};
};

TEST_CASE("layout_base: sizes and strides come from the dims", "[layout_base]")
{
    using Shape = ShapeOnly<2, 3, 4>;

    STATIC_REQUIRE(Shape::num_dims() == 3);
    STATIC_REQUIRE(Shape::LogicalSize == 24);

    REQUIRE(Shape::logical_dim(0) == 2);
    REQUIRE(Shape::logical_dim(1) == 3);
    REQUIRE(Shape::logical_dim(2) == 4);
    CHECK_THROWS(Shape::logical_dim(3)); // out-of-bounds

    REQUIRE(Shape::logical_stride(0) == 12);
    REQUIRE(Shape::logical_stride(1) == 4);
    REQUIRE(Shape::logical_stride(2) == 1);
    CHECK_THROWS(Shape::logical_stride(3)); // out-of-bounds
}

TEST_CASE("layout_base: bounds", "[layout_base]")
{
    using Shape = ShapeOnly<2, 3>;

    const my_size_t inside[2] = {1, 2};
    const my_size_t past_row[2] = {2, 0};
    const my_size_t past_column[2] = {0, 3};

    REQUIRE(Shape::is_logical_index_in_bounds(inside));
    REQUIRE_FALSE(Shape::is_logical_index_in_bounds(past_row));
    REQUIRE_FALSE(Shape::is_logical_index_in_bounds(past_column));
}

TEST_CASE("layout_base: logical_coords_to_logical_flat", "[layout_base]")
{
    /*
     * 3x4 shape, LogicalStrides = [4, 1]
     *
     *   (0,0) → 0    (0,3) → 3
     *   (1,0) → 4    (1,3) → 7
     *   (2,0) → 8    (2,3) → 11
     */
    using Shape = ShapeOnly<3, 4>;

    const my_size_t first[2] = {0, 0};
    const my_size_t middle[2] = {1, 3};
    const my_size_t last[2] = {2, 3};

    REQUIRE(Shape::logical_coords_to_logical_flat(first) == 0);
    REQUIRE(Shape::logical_coords_to_logical_flat(middle) == 7);
    REQUIRE(Shape::logical_coords_to_logical_flat(last) == 11);

    const my_size_t past_row[2] = {3, 0};
    const my_size_t past_column[2] = {0, 4};
    CHECK_THROWS(Shape::logical_coords_to_logical_flat(past_row));    // out-of-bounds
    CHECK_THROWS(Shape::logical_coords_to_logical_flat(past_column)); // out-of-bounds
}

TEMPLATE_TEST_CASE("layout_base: flat and coords round trip", "[layout_base]",
                   ShapeOnly<7>, (ShapeOnly<3, 4>), (ShapeOnly<2, 3, 4>), (ShapeOnly<1, 5, 1, 2>))
{
    using Shape = TestType;
    my_size_t coords[Shape::NumDims];

    for (my_size_t flat = 0; flat < Shape::LogicalSize; ++flat)
    {
        Shape::logical_flat_to_logical_coords(flat, coords);
        REQUIRE(Shape::is_logical_index_in_bounds(coords));
        REQUIRE(Shape::logical_coords_to_logical_flat(coords) == flat);
    }

    CHECK_THROWS(Shape::logical_flat_to_logical_coords(Shape::LogicalSize, coords)); // out-of-bounds
}

TEST_CASE("layout_base: LayoutDims reads the shape back as a raw array", "[layout_base]")
{
    using Shape = ShapeOnly<2, 3, 4>;

    REQUIRE(LayoutDims<Shape>::value[0] == 2);
    REQUIRE(LayoutDims<Shape>::value[1] == 3);
    REQUIRE(LayoutDims<Shape>::value[2] == 4);
}
