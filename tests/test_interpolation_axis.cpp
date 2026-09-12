#include <catch_amalgamated.hpp>

#include "interpolation/axis.h"

using Catch::Approx;
using namespace interpolation;

// ============================================================================
// UNIFORM AXIS
// ============================================================================

TEMPLATE_TEST_CASE("uniform_axis: locates a query inside the domain",
                   "[axis]", double, float)
{
    using T = TestType;

    // Breakpoints 100, 150, 200, 250
    UniformAxis<T, 4> axis((T)100, (T)50);

    auto loc = axis.locate((T)173);

    CHECK(loc.cell == 1);
    CHECK(loc.frac == Approx((T)0.46));
}

TEMPLATE_TEST_CASE("uniform_axis: exact breakpoints land at fraction zero",
                   "[axis]", double, float)
{
    using T = TestType;

    UniformAxis<T, 4> axis((T)100, (T)50);

    auto l0 = axis.locate((T)100);
    CHECK(l0.cell == 0);
    CHECK(l0.frac == Approx((T)0));

    auto l1 = axis.locate((T)150);
    CHECK(l1.cell == 1);
    CHECK(l1.frac == Approx((T)0));

    auto l2 = axis.locate((T)200);
    CHECK(l2.cell == 2);
    CHECK(l2.frac == Approx((T)0));
}

TEMPLATE_TEST_CASE("uniform_axis: clamps below and above the domain",
                   "[axis]", double, float)
{
    using T = TestType;

    UniformAxis<T, 4> axis((T)100, (T)50);

    // Below: first cell, fraction 0
    auto below = axis.locate((T)50);
    CHECK(below.cell == 0);
    CHECK(below.frac == Approx((T)0));

    // Above: last cell, fraction 1
    auto above = axis.locate((T)900);
    CHECK(above.cell == 2);
    CHECK(above.frac == Approx((T)1));

    // The last breakpoint itself is the top of the last cell
    auto edge = axis.locate((T)250);
    CHECK(edge.cell == 2);
    CHECK(edge.frac == Approx((T)1));
}

TEMPLATE_TEST_CASE("uniform_axis: reports its breakpoints",
                   "[axis]", double, float)
{
    using T = TestType;

    UniformAxis<T, 4> axis((T)100, (T)50);

    CHECK(axis.breakpoint(0) == Approx((T)100));
    CHECK(axis.breakpoint(1) == Approx((T)150));
    CHECK(axis.breakpoint(2) == Approx((T)200));
    CHECK(axis.breakpoint(3) == Approx((T)250));

    CHECK(axis.NumBreakpoints == 4);
    CHECK(axis.NumCells == 3);
}

TEMPLATE_TEST_CASE("uniform_axis: two breakpoints is one cell",
                   "[axis]", double, float)
{
    using T = TestType;

    UniformAxis<T, 2> axis((T)0, (T)1);

    CHECK(axis.NumCells == 1);

    auto mid = axis.locate((T)0.25);
    CHECK(mid.cell == 0);
    CHECK(mid.frac == Approx((T)0.25));

    auto top = axis.locate((T)1);
    CHECK(top.cell == 0);
    CHECK(top.frac == Approx((T)1));
}

// ============================================================================
// NON-UNIFORM AXIS
// ============================================================================

TEMPLATE_TEST_CASE("non_uniform_axis: locates a query inside the domain",
                   "[axis]", double, float)
{
    using T = TestType;

    // Clustered where the plant is sensitive
    NonUniformAxis<T, 5> axis((T)100, (T)120, (T)130, (T)140, (T)200);

    // 173 falls between 140 and 200, 33/60 of the way
    auto loc = axis.locate((T)173);

    CHECK(loc.cell == 3);
    CHECK(loc.frac == Approx((T)33 / (T)60));
}

TEMPLATE_TEST_CASE("non_uniform_axis: exact breakpoints land at fraction zero",
                   "[axis]", double, float)
{
    using T = TestType;

    NonUniformAxis<T, 5> axis((T)100, (T)120, (T)130, (T)140, (T)200);

    auto l1 = axis.locate((T)120);
    CHECK(l1.cell == 1);
    CHECK(l1.frac == Approx((T)0));

    auto l2 = axis.locate((T)130);
    CHECK(l2.cell == 2);
    CHECK(l2.frac == Approx((T)0));
}

TEMPLATE_TEST_CASE("non_uniform_axis: clamps below and above the domain",
                   "[axis]", double, float)
{
    using T = TestType;

    NonUniformAxis<T, 5> axis((T)100, (T)120, (T)130, (T)140, (T)200);

    auto below = axis.locate((T)50);
    CHECK(below.cell == 0);
    CHECK(below.frac == Approx((T)0));

    auto above = axis.locate((T)500);
    CHECK(above.cell == 3);
    CHECK(above.frac == Approx((T)1));

    auto edge = axis.locate((T)200);
    CHECK(edge.cell == 3);
    CHECK(edge.frac == Approx((T)1));
}

TEMPLATE_TEST_CASE("non_uniform_axis: an evenly spaced one agrees with uniform",
                   "[axis]", double, float)
{
    using T = TestType;

    UniformAxis<T, 4> uni((T)100, (T)50);

    NonUniformAxis<T, 4> non((T)100, (T)150, (T)200, (T)250);

    // Same breakpoints by two routes must give the same answers
    for (T q = (T)90; q < (T)270; q += (T)7)
    {
        auto a = uni.locate(q);
        auto b = non.locate(q);

        CHECK(a.cell == b.cell);
        CHECK(a.frac == Approx(b.frac));
    }
}

// ============================================================================
// CURSOR
// ============================================================================

TEMPLATE_TEST_CASE("non_uniform_axis: the cursor is updated in place",
                   "[axis]", double, float)
{
    using T = TestType;

    NonUniformAxis<T, 5> axis((T)100, (T)120, (T)130, (T)140, (T)200);

    my_size_t cursor = 0;

    axis.locate((T)173, cursor);
    CHECK(cursor == 3);

    axis.locate((T)125, cursor);
    CHECK(cursor == 1);
}

TEMPLATE_TEST_CASE("non_uniform_axis: a warm cursor gives the same answer",
                   "[axis]", double, float)
{
    using T = TestType;

    NonUniformAxis<T, 5> axis((T)100, (T)120, (T)130, (T)140, (T)200);

    // Walking a sequence with a carried cursor must match locating each query
    // from scratch, the cursor is just an optimisation, not a change of behaviour
    my_size_t cursor = 0;

    const T queries[] = {(T)105, (T)125, (T)135, (T)190, (T)102, (T)145};

    for (T q : queries)
    {
        auto warm = axis.locate(q, cursor);
        auto cold = axis.locate(q);

        CHECK(warm.cell == cold.cell);
        CHECK(warm.frac == Approx(cold.frac));
    }
}

TEMPLATE_TEST_CASE("non_uniform_axis: a stale cursor still finds the cell and gets updated",
                   "[axis]", double, float)
{
    using T = TestType;

    NonUniformAxis<T, 5> axis((T)100, (T)120, (T)130, (T)140, (T)200);

    // A cursor pointing at the far end of the axis from the query
    // the walk has to cross the whole domain
    my_size_t cursor = 3;
    auto low = axis.locate((T)105, cursor);
    CHECK(low.cell == 0);
    CHECK(cursor == 0);

    cursor = 0;
    auto high = axis.locate((T)190, cursor);
    CHECK(high.cell == 3);
    CHECK(cursor == 3);

    // Out of range entirely
    cursor = 99;
    auto recovered = axis.locate((T)135, cursor);
    CHECK(recovered.cell == 2);
    CHECK(cursor == 2);
}

TEMPLATE_TEST_CASE("uniform_axis: accepts and ignores a cursor",
                   "[axis]", double, float)
{
    using T = TestType;

    UniformAxis<T, 4> axis((T)100, (T)50);

    my_size_t cursor = 99;
    auto loc = axis.locate((T)173, cursor);

    CHECK(loc.cell == 1);
    CHECK(loc.frac == Approx((T)0.46));
    CHECK(cursor == 99); // cursor is ignored, not updated
}

// ============================================================================
// NEGATIVE AND FRACTIONAL DOMAINS
// ============================================================================

TEMPLATE_TEST_CASE("uniform_axis: negative origin",
                   "[axis]", double, float)
{
    using T = TestType;

    // Breakpoints -10, -5, 0, 5
    UniformAxis<T, 4> axis((T)-10, (T)5);

    auto loc = axis.locate((T)-2.5);
    CHECK(loc.cell == 1);
    CHECK(loc.frac == Approx((T)0.5));

    auto below = axis.locate((T)-20);
    CHECK(below.cell == 0);
    CHECK(below.frac == Approx((T)0));
}

TEMPLATE_TEST_CASE("uniform_axis: fractional spacing",
                   "[axis]", double, float)
{
    using T = TestType;

    // Breakpoints 0, 0.25, 0.5, 0.75, 1
    UniformAxis<T, 5> axis((T)0, (T)0.25);

    auto loc = axis.locate((T)0.6);
    CHECK(loc.cell == 2);
    CHECK(loc.frac == Approx((T)0.4));
}
