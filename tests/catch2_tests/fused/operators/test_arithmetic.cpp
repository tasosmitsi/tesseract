#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "fused/operators/arithmetic.h"

// TODO: the arithmetic operators themselves are still tested in
// test_fused_tensor.cpp and test_fma_expressions.cpp. Move them here so the
// file matches the header.

// The reference the vectorised lerp is checked against.
template <typename Matrix, typename T>
static Matrix lerp_reference(const Matrix &a, const Matrix &b, T t)
{
    Matrix out;
    for (my_size_t i = 0; i < a.getDim(0); ++i)
        for (my_size_t j = 0; j < a.getDim(1); ++j)
            out(i, j) = a(i, j) + t * (b(i, j) - a(i, j));
    return out;
}

template <typename A, typename B, typename T>
static auto halfway_then_offset(const BaseExpr<A> &a, const BaseExpr<B> &b, T offset)
{
    return lerp(a, b, (T)0.5) + offset;
}

// ============================================================================
// KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("lerp: endpoints and midpoint",
                   "[lerp]", double, float)
{
    using T = TestType;

    T vals_a[2][3] = {{0, 2, -4}, {10, 1, 6}};
    T vals_b[2][3] = {{8, 2, 4}, {20, -1, 2}};
    FusedMatrix<T, 2, 3> a(vals_a), b(vals_b), out;

    out = lerp(a, b, (T)0);
    CHECK(out == a);

    out = lerp(a, b, (T)1);
    CHECK(out == b);

    out = lerp(a, b, (T)0.5);
    T mid[2][3] = {{4, 2, 0}, {15, 0, 4}};
    CHECK(out == FusedMatrix<T, 2, 3>(mid));
}

TEMPLATE_TEST_CASE("lerp: a quarter of the way",
                   "[lerp]", double, float)
{
    using T = TestType;

    T vals_a[2][2] = {{0, 4}, {-8, 1}};
    T vals_b[2][2] = {{8, 0}, {8, 1}};
    FusedMatrix<T, 2, 2> a(vals_a), b(vals_b), out;

    out = lerp(a, b, (T)0.25);

    T expected[2][2] = {{2, 3}, {-4, 1}};
    CHECK(out == FusedMatrix<T, 2, 2>(expected));
}

TEST_CASE("lerp: t must have the tensor's value type",
          "[lerp]")
{
    FusedMatrix<float, 1, 2> a(0.0f), b(1.0f), out;

    out = lerp(a, b, 0.75f);
    CHECK(out(0, 0) == 0.75f);
    CHECK(out(0, 1) == 0.75f);

    // Rejected at compile time, like the arithmetic operators.
    // Uncomment one at a time to check:
    // out = lerp(a, b, 0.75);       // double t on a float tensor
    // out = lerp(a, b, 1);          // int t on a float tensor

    // Rejected because lerp is floating point only, uncomment both lines:
    // FusedMatrix<int, 1, 2> i(0), j(4);
    // auto e = lerp(i, j, 1);
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEMPLATE_TEST_CASE("lerp: outside [0, 1] it extrapolates",
                   "[lerp]", double, float)
{
    using T = TestType;

    T vals_a[1][2] = {{2, 10}};
    T vals_b[1][2] = {{4, 0}};
    FusedMatrix<T, 1, 2> a(vals_a), b(vals_b), out;

    out = lerp(a, b, (T)2);
    CHECK(out(0, 0) == (T)6);
    CHECK(out(0, 1) == (T)-10);

    out = lerp(a, b, (T)-1);
    CHECK(out(0, 0) == (T)0);
    CHECK(out(0, 1) == (T)20);
}

TEMPLATE_TEST_CASE("lerp: equal endpoints give that value for any t",
                   "[lerp]", double, float)
{
    using T = TestType;

    FusedMatrix<T, 2, 2> a((T)3), b((T)3), out;

    out = lerp(a, b, (T)0.3);
    CHECK(out == a);

    out = lerp(a, b, (T)7);
    CHECK(out == a);
}

// ============================================================================
// PROPERTIES
// ============================================================================

TEMPLATE_TEST_CASE("lerp: swapping the endpoints mirrors t",
                   "[lerp]", double, float)
{
    using T = TestType;

    FusedMatrix<T, 3, 3> a, b, forward, backward;
    a.setSequencial();
    b.setHomogen((T)5);

    forward = lerp(a, b, (T)0.3);
    backward = lerp(b, a, (T)0.7);
    CHECK(forward == backward);
}

TEMPLATE_TEST_CASE("lerp: folds into one fused multiply-add",
                   "[lerp]", double, float)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 3>;
    using Lerp = decltype(lerp(FusedMatrix<T, 3, 3>(), FusedMatrix<T, 3, 3>(), (T)0.5));
    using OneFma = ScalarFmaExpr<BinaryExpr<Tensor, Tensor, Sub>, T, Tensor, Fma>;

    // (b - a) * t + a: one subtraction feeding one FMA node
    CHECK((is_same_v<Lerp, OneFma>));
}

TEMPLATE_TEST_CASE("lerp: repeated toward a target is exponential smoothing",
                   "[lerp]", double, float)
{
    using T = TestType;

    FusedMatrix<T, 2, 2> x((T)0), target((T)16), next;

    // each step keeps 1 - alpha of the remaining distance
    for (int step = 0; step < 4; ++step)
    {
        next = lerp(x, target, (T)0.5);
        x = next;
    }

    // 16 * (1 - 0.5^4) = 15
    CHECK(x == FusedMatrix<T, 2, 2>((T)15));
}

// ============================================================================
// COMPOSITION AND LIFETIME
// ============================================================================

TEMPLATE_TEST_CASE("lerp: expressions and views as the endpoints",
                   "[lerp]", double, float)
{
    using T = TestType;

    T vals_a[3][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    FusedMatrix<T, 3, 3> a(vals_a), b((T)10);

    SECTION("a transpose view")
    {
        FusedMatrix<T, 3, 3> out;
        out = lerp(a.transpose_view(), b, (T)0.5);

        CHECK(out(0, 1) == (T)6.5); // (3 + 10) / 2
        CHECK(out(1, 0) == (T)5.5); // (1 + 10) / 2
    }

    SECTION("two row views")
    {
        FusedMatrix<T, 1, 3> out;
        out = lerp(a.template row<0>(), a.template row<2>(), (T)0.5);

        CHECK(out(0, 0) == (T)3);
        CHECK(out(0, 1) == (T)4);
        CHECK(out(0, 2) == (T)5);
    }

    SECTION("arithmetic expressions")
    {
        FusedMatrix<T, 3, 3> out;
        out = lerp(a + b, a - b, (T)0.5);
        CHECK(out == a);
    }

    SECTION("inside a longer expression")
    {
        FusedMatrix<T, 3, 3> out;
        out = lerp(a, b, (T)0.5) * (T)2 - b;
        CHECK(out == a);
    }
}

TEMPLATE_TEST_CASE("lerp: can be kept and returned",
                   "[lerp]", double, float)
{
    using T = TestType;

    T vals_a[2][2] = {{0, 2}, {4, 6}};
    FusedMatrix<T, 2, 2> a(vals_a), b((T)8), out;

    SECTION("kept in auto and evaluated later")
    {
        auto e = lerp(a + b, b, (T)0.5);
        out = e;

        CHECK(out(0, 0) == (T)8);  // (8 + 8) / 2
        CHECK(out(1, 1) == (T)11); // (14 + 8) / 2
    }

    SECTION("returned from a helper")
    {
        auto e = halfway_then_offset(a, b, (T)1);
        out = e;

        CHECK(out(0, 0) == (T)5);
        CHECK(out(1, 1) == (T)8);
    }
}

// ============================================================================
// GENERIC PATH
// ============================================================================

TEMPLATE_TEST_CASE("lerp: a shape whose width is not a SIMD multiple",
                   "[lerp]", double, float)
{
    using T = TestType;

    FusedMatrix<T, 5, 7> a, b, out;
    a.setSequencial();
    b.setHomogen((T)100);

    out = lerp(a, b, (T)0.25);
    CHECK(out == lerp_reference(a, b, (T)0.25));
}
