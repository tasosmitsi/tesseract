#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "fused/operators/minmax.h"

// TODO: the element-wise min/max tests are still in test_fused_tensor.cpp.
// Move them here so the file matches the header.

// The reference the vectorised clamp is checked against.
template <typename Matrix, typename T>
static Matrix clamp_reference(const Matrix &x, T lo, T hi)
{
    Matrix out;
    for (my_size_t i = 0; i < x.getDim(0); ++i)
        for (my_size_t j = 0; j < x.getDim(1); ++j)
            out(i, j) = x(i, j) < lo ? lo : (x(i, j) > hi ? hi : x(i, j));
    return out;
}

template <typename Expr, typename T>
static auto clamp_and_scale(const BaseExpr<Expr> &x, T lo, T hi, T k)
{
    return clamp(x, lo, hi) * k;
}

// ============================================================================
// KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("clamp: scalar bounds",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    T vals[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    FusedMatrix<T, 3, 3> a(vals), out;

    out = clamp(a, (T)2, (T)6);

    T expected[3][3] = {{2, 2, 2}, {3, 4, 5}, {6, 6, 6}};
    CHECK(out == FusedMatrix<T, 3, 3>(expected));
}

TEMPLATE_TEST_CASE("clamp: element-wise bounds",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    // a different window for every column
    T vals[2][3] = {{-5, 5, 50}, {0, 10, 100}};
    T lo_vals[2][3] = {{-1, 6, 0}, {-1, 6, 0}};
    T hi_vals[2][3] = {{1, 8, 60}, {1, 8, 60}};
    FusedMatrix<T, 2, 3> a(vals), lo(lo_vals), hi(hi_vals), out;

    out = clamp(a, lo, hi);

    T expected[2][3] = {{-1, 6, 50}, {0, 8, 60}};
    CHECK(out == FusedMatrix<T, 2, 3>(expected));
}

TEMPLATE_TEST_CASE("clamp: a scalar bound on one side, a tensor on the other",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    T vals[2][3] = {{-5, 5, 50}, {0, 10, 100}};
    T limit_vals[2][3] = {{1, 8, 60}, {1, 8, 60}};
    FusedMatrix<T, 2, 3> a(vals), limits(limit_vals), out;

    SECTION("scalar floor, per-element ceilings")
    {
        out = clamp(a, (T)0, limits);

        T expected[2][3] = {{0, 5, 50}, {0, 8, 60}};
        CHECK(out == FusedMatrix<T, 2, 3>(expected));
    }

    SECTION("per-element floors, scalar ceiling")
    {
        out = clamp(a, limits, (T)70);

        T expected[2][3] = {{1, 8, 60}, {1, 10, 70}};
        CHECK(out == FusedMatrix<T, 2, 3>(expected));
    }
}

TEST_CASE("clamp: bounds must have the tensor's value type",
          "[clamp]")
{
    float vals[2][2] = {{0.0f, 0.5f}, {1.0f, 1.5f}};
    FusedMatrix<float, 2, 2> a(vals), out;

    out = clamp(a, 0.25f, 1.25f);
    CHECK(out(0, 0) == 0.25f);
    CHECK(out(1, 1) == 1.25f);

    // Rejected at compile time, like min, max and the arithmetic operators.
    // Uncomment one at a time to check:
    // out = clamp(a, 0.25, 1.25);   // double bounds on a float tensor
    // out = clamp(a, 0, 1);         // int bounds on a float tensor
    // out = clamp(a, 0.25f, 1);     // bounds of different types

    // Rejected the same way, uncomment both lines:
    // FusedMatrix<int, 2, 2> i(1), io;
    // io = clamp(i, 0.5, 2.5);      // decimal bounds on an int tensor
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEMPLATE_TEST_CASE("clamp: values inside the bounds are unchanged",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    T vals[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    FusedMatrix<T, 3, 3> a(vals), out;

    out = clamp(a, (T)0, (T)8);
    CHECK(out == a);
}

TEMPLATE_TEST_CASE("clamp: equal bounds give a constant",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    T vals[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    FusedMatrix<T, 3, 3> a(vals), out;

    out = clamp(a, (T)4, (T)4);
    CHECK(out == FusedMatrix<T, 3, 3>((T)4));
}

TEMPLATE_TEST_CASE("clamp: lo above hi gives lo",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    T vals[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    FusedMatrix<T, 3, 3> a(vals), out;

    // the max is applied last, so it wins
    out = clamp(a, (T)6, (T)2);
    CHECK(out == FusedMatrix<T, 3, 3>((T)6));
}

TEMPLATE_TEST_CASE("clamp: 1x1",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    FusedMatrix<T, 1, 1> a((T)9), out;

    out = clamp(a, (T)-3, (T)3);
    CHECK(out(0, 0) == (T)3);
}

// ============================================================================
// COMPOSITION AND LIFETIME
// ============================================================================

TEMPLATE_TEST_CASE("clamp: expressions and views as the argument",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    T vals_a[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    T vals_b[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    FusedMatrix<T, 3, 3> a(vals_a), b(vals_b);

    SECTION("an arithmetic expression")
    {
        FusedMatrix<T, 3, 3> out;
        out = clamp(a * (T)2 - b, (T)0, (T)10);

        CHECK(out(0, 0) == (T)0);  // 0*2 - 1 = -1
        CHECK(out(1, 1) == (T)7);  // 4*2 - 1
        CHECK(out(2, 2) == (T)10); // 8*2 - 1 = 15
    }

    SECTION("a transpose view")
    {
        FusedMatrix<T, 3, 3> out;
        out = clamp(a.transpose_view(), (T)2, (T)6);

        CHECK(out(0, 1) == (T)3); // a(1, 0)
        CHECK(out(1, 0) == (T)2); // a(0, 1) = 1
        CHECK(out(2, 0) == (T)2); // a(0, 2)
        CHECK(out(0, 2) == (T)6); // a(2, 0)
    }

    SECTION("a row view")
    {
        FusedMatrix<T, 1, 3> out;
        out = clamp(a.template row<1>(), (T)4, (T)4);

        CHECK(out(0, 0) == (T)4);
        CHECK(out(0, 1) == (T)4);
        CHECK(out(0, 2) == (T)4);
    }

    SECTION("inside a longer expression")
    {
        FusedMatrix<T, 3, 3> out;
        out = clamp(a, (T)2, (T)6) + b;

        CHECK(out(0, 0) == (T)3);
        CHECK(out(1, 1) == (T)5);
        CHECK(out(2, 2) == (T)7);
    }
}

TEMPLATE_TEST_CASE("clamp: can be kept and returned",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    T vals_a[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    T vals_b[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    FusedMatrix<T, 3, 3> a(vals_a), b(vals_b), out;

    SECTION("kept in auto and evaluated later")
    {
        auto c = clamp(a + b, (T)2, (T)6);
        out = c;

        CHECK(out(0, 0) == (T)2);
        CHECK(out(1, 1) == (T)5);
        CHECK(out(2, 2) == (T)6);
    }

    SECTION("returned from a helper")
    {
        auto c = clamp_and_scale(a, (T)2, (T)6, (T)10);
        out = c;

        CHECK(out(0, 0) == (T)20);
        CHECK(out(1, 1) == (T)40);
        CHECK(out(2, 2) == (T)60);
    }
}

// ============================================================================
// GENERIC PATH
// ============================================================================

TEMPLATE_TEST_CASE("clamp: a shape whose width is not a SIMD multiple",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    FusedMatrix<T, 5, 7> a, out;
    a.setSequencial(); // 0 .. 34

    out = clamp(a, (T)5, (T)29);
    CHECK(out == clamp_reference(a, (T)5, (T)29));
}

TEMPLATE_TEST_CASE("clamp: a larger matrix",
                   "[clamp]", double, float, int)
{
    using T = TestType;

    FusedMatrix<T, 20, 20> a, out;
    a.setSequencial(); // 0 .. 399

    out = clamp(a, (T)50, (T)350);
    CHECK(out == clamp_reference(a, (T)50, (T)350));
}
