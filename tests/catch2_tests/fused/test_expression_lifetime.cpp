#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_tensor.h"
#include "fused/fused_vector.h"
#include "fused/views/multi_slice_view.h"
#include "fused/views/squeeze_view.h"

// Expressions that outlive the statement that built them. Every case below
// evaluates an expression after its sub-expression temporaries have died, so
// it reads dead stack unless the nodes keep their own copies. With nodes that
// hold references, these pass at -O0 by luck, segfault at -O2 and fail under
// AddressSanitizer as stack-use-after-scope.

template <typename V>
concept CopyConstructible = requires(const V &v) { V(v); };

template <typename V>
concept MoveConstructible = requires(V &&v) { V(static_cast<V &&>(v)); };

template <typename V>
concept CopyAssignable = requires(V &lhs, const V &rhs) { lhs = rhs; };

// The helper that got clamp dropped from the library: it returns a composed
// expression whose inner node is a temporary of this function.
template <typename Expr, typename T>
auto clamp_expr(const BaseExpr<Expr> &e, T lo, T hi)
{
    return max(min(e, hi), lo);
}

// ============================================================================
// STORED EXPRESSIONS
// ============================================================================

TEMPLATE_TEST_CASE("expression_lifetime: a composed expression stored in auto evaluates later",
                   "[expression_lifetime]", double, float, int)
{
    using T = TestType;

    T vals_a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    T vals_b[3][3] = {{2, 2, 2}, {1, 1, 1}, {0, 0, 0}};
    FusedMatrix<T, 3, 3> a(vals_a), b(vals_b), out, ref;

    auto e = (a + b) * (a - b);
    out = e;
    ref = (a + b) * (a - b);

    CHECK(out(0, 0) == (T)-3);
    CHECK(out(1, 1) == (T)24);
    CHECK(out(2, 2) == (T)81);
    CHECK(out == ref);

    // evaluating the same stored expression again gives the same result
    FusedMatrix<T, 3, 3> again;
    again = e;
    CHECK(again == ref);

    // tensors are held by reference, so a change shows up on the next
    // evaluation, also through the sums copied into e
    a(0, 0) = (T)10;
    again = e;
    CHECK(again(0, 0) == (T)96); // (10 + 2) * (10 - 2)
    CHECK(again(1, 1) == (T)24); // untouched elements are unchanged
}

TEMPLATE_TEST_CASE("expression_lifetime: a scalar is captured when the expression is built",
                   "[expression_lifetime]", double, float, int)
{
    using T = TestType;

    T vals_a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    FusedMatrix<T, 3, 3> a(vals_a), out;

    // unlike a tensor, the scalar is copied into the node: changing the
    // variable afterwards does not change the expression
    T s = (T)2;
    auto e = a * s;
    s = (T)3;
    out = e;

    CHECK(out(0, 0) == (T)2);
    CHECK(out(2, 2) == (T)18);
}

TEMPLATE_TEST_CASE("expression_lifetime: a helper may return a composed expression",
                   "[expression_lifetime]", double, float, int)
{
    using T = TestType;

    T vals[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    FusedMatrix<T, 3, 3> a(vals), out;

    auto c = clamp_expr(a, (T)3, (T)7);
    out = c;

    CHECK(out(0, 0) == (T)3);
    CHECK(out(0, 2) == (T)3);
    CHECK(out(1, 1) == (T)5);
    CHECK(out(2, 0) == (T)7);
    CHECK(out(2, 2) == (T)7);
}

// ============================================================================
// FMA FUSION
// ============================================================================

TEMPLATE_TEST_CASE("expression_lifetime: fused multiply-add nodes keep their operands",
                   "[expression_lifetime]", double, float, int)
{
    using T = TestType;

    T vals_a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    T vals_b[3][3] = {{2, 2, 2}, {1, 1, 1}, {0, 0, 0}};
    FusedMatrix<T, 3, 3> a(vals_a), b(vals_b), out;

    SECTION("(a + b) * a + b")
    {
        auto f = (a + b) * a + b;
        out = f;

        CHECK(out(0, 0) == (T)5);
        CHECK(out(1, 1) == (T)31);
        CHECK(out(2, 2) == (T)81);
    }

    SECTION("(a + b) * scalar + b")
    {
        auto f = (a + b) * (T)2 + b;
        out = f;

        CHECK(out(0, 0) == (T)8);
        CHECK(out(1, 1) == (T)13);
        CHECK(out(2, 2) == (T)18);
    }

    SECTION("-((a + b) * a) + b")
    {
        auto f = -((a + b) * a) + b;
        out = f;

        CHECK(out(0, 0) == (T)-1);
        CHECK(out(1, 1) == (T)-29);
        CHECK(out(2, 2) == (T)-81);
    }
}

// ============================================================================
// VIEWS
// ============================================================================

TEMPLATE_TEST_CASE("expression_lifetime: view temporaries inside a stored expression",
                   "[expression_lifetime]", double, float, int)
{
    using T = TestType;

    T vals_a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    T vals_b[3][3] = {{2, 2, 2}, {1, 1, 1}, {0, 0, 0}};
    FusedMatrix<T, 3, 3> a(vals_a), b(vals_b);

    SECTION("transpose view")
    {
        FusedMatrix<T, 3, 3> out;
        auto g = a.transpose_view() + b;
        out = g;

        CHECK(out(0, 1) == (T)6);
        CHECK(out(1, 0) == (T)3);
        CHECK(out(2, 2) == (T)9);
    }

    SECTION("two row views")
    {
        FusedMatrix<T, 1, 3> out;
        auto r = a.template row<0>() + a.template row<2>();
        out = r;

        CHECK(out(0, 0) == (T)8);
        CHECK(out(0, 1) == (T)10);
        CHECK(out(0, 2) == (T)12);
    }

    SECTION("diagonal view plus column view")
    {
        FusedVector<T, 3> out;
        auto d = a.diagonal() + a.template column<0>();
        out = d;

        CHECK(out(0) == (T)2);
        CHECK(out(1) == (T)9);
        CHECK(out(2) == (T)16);
    }

    SECTION("head and tail subvector views")
    {
        FusedVector<T, 4> v;
        for (my_size_t i = 0; i < 4; ++i)
            v(i) = (T)(i + 1);

        FusedVector<T, 2> out;
        auto s = v.template head<2>() + v.template tail<2>();
        out = s;

        CHECK(out(0) == (T)4);
        CHECK(out(1) == (T)6);
    }

    SECTION("squeeze view of a slice")
    {
        using Tensor = FusedTensorND<T, 2, 3, 3>;
        using Plane = MultiSliceView<Tensor, Slice<0, 1, 1>>;

        Tensor t;
        t.setSequencial(); // plane 1 holds 9 .. 17

        FusedMatrix<T, 3, 3> out;
        auto s = SqueezeView<Plane, 0>(Plane(t)) + b;
        out = s;

        CHECK(out(0, 0) == (T)11);
        CHECK(out(1, 1) == (T)14);
        CHECK(out(2, 2) == (T)17);
    }
}

TEMPLATE_TEST_CASE("expression_lifetime: a view over a temporary view keeps its own copy",
                   "[expression_lifetime]", double, float, int)
{
    using T = TestType;

    T vals[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    FusedMatrix<T, 3, 3> a(vals);

    using Transposed = decltype(a.transpose_view());
    auto first_row_of_t = MultiSliceView<Transposed, Slice<0, 0, 1>>(a.transpose_view());

    FusedMatrix<T, 1, 3> out;
    out = first_row_of_t;

    // row 0 of the transpose is column 0 of a
    CHECK(out(0, 0) == (T)1);
    CHECK(out(0, 1) == (T)4);
    CHECK(out(0, 2) == (T)7);
}

// ============================================================================
// SEMANTICS
// ============================================================================

TEMPLATE_TEST_CASE("expression_lifetime: stored expressions read their tensors live",
                   "[expression_lifetime]", double, float, int)
{
    using T = TestType;

    T vals_a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    T vals_b[3][3] = {{2, 2, 2}, {1, 1, 1}, {0, 0, 0}};
    FusedMatrix<T, 3, 3> a(vals_a), b(vals_b), out;

    // tensors are held by reference, so a write after building the expression
    // shows up when it is evaluated
    auto e = a + b;
    a(0, 0) = (T)100;
    out = e;

    CHECK(out(0, 0) == (T)102);
    CHECK(out(1, 1) == (T)6);
}

TEMPLATE_TEST_CASE("expression_lifetime: views copy and move but do not assign",
                   "[expression_lifetime]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 4>;

    using Transposed = decltype(Matrix().transpose_view());
    using Row = decltype(Matrix().template row<0>());
    using Diagonal = decltype(Matrix().diagonal());
    using Head = decltype(Vector().template head<2>());
    using Squeeze = SqueezeView<FusedTensorND<T, 1, 3, 3>, 0>;

    CHECK(CopyConstructible<Transposed>);
    CHECK(CopyConstructible<Row>);
    CHECK(CopyConstructible<Diagonal>);
    CHECK(CopyConstructible<Head>);
    CHECK(CopyConstructible<Squeeze>);

    CHECK(MoveConstructible<Transposed>);
    CHECK(MoveConstructible<Row>);
    CHECK(MoveConstructible<Diagonal>);
    CHECK(MoveConstructible<Head>);
    CHECK(MoveConstructible<Squeeze>);

    // a view cannot be re-pointed at another source
    CHECK_FALSE(CopyAssignable<Transposed>);
    CHECK_FALSE(CopyAssignable<Row>);
    CHECK_FALSE(CopyAssignable<Diagonal>);
    CHECK_FALSE(CopyAssignable<Head>);
    CHECK_FALSE(CopyAssignable<Squeeze>);
}
