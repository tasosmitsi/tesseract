#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_tensor.h"
#include "fused/fused_vector.h"
#include "fused/views/diagonal_view.h"
#include "fused/views/multi_slice_view.h"
#include "fused/views/permuted_view_constexpr.h"
#include "fused/views/squeeze_view.h"
#include "fused/views/subvector_view.h"

// ============================================================================
// Views built on views. Each view computes its layout from its source's
// layout, so a chain reads the elements its last view names, in any order.
// Every chain here is checked element by element against the root tensor,
// then through assignment and an expression, which take the SIMD paths.
//
// The first five were wrong before views composed: the outer view dropped
// where the inner one started, or took the inner one's shape from its source.
// ============================================================================

// ============================================================================
// CHAINS THAT USED TO READ THE WRONG ELEMENTS
// ============================================================================

TEMPLATE_TEST_CASE("view_composition: slice of a subvector",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using V = FusedVector<T, 6>;
    using Sub = SubVectorView<V, 2, 3>;               // v(2) v(3) v(4)
    using Part = MultiSliceView<Sub, Slice<0, 1, 2>>; // v(3) v(4)

    V v;
    v.setSequencial();

    auto part = Part(Sub(v));

    REQUIRE(part.getShape() == "(2,1)");
    REQUIRE(part(0, 0) == v(3));
    REQUIRE(part(1, 0) == v(4));

    FusedVector<T, 2> result;
    result = part;
    REQUIRE(result(0) == v(3));
    REQUIRE(result(1) == v(4));
}

TEMPLATE_TEST_CASE("view_composition: slice of a diagonal",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using M = FusedMatrix<T, 3, 3>;
    using Diag = DiagonalView<M>;
    using Tail = MultiSliceView<Diag, Slice<0, 1, 2>>;

    M A;
    A.setSequencial(); // diagonal 0 4 8

    auto tail = Tail(Diag(A));

    REQUIRE(tail.getShape() == "(2,1)");
    REQUIRE(tail(0, 0) == A(1, 1));
    REQUIRE(tail(1, 0) == A(2, 2));

    FusedVector<T, 2> result;
    result = tail;
    REQUIRE(result(0) == A(1, 1));
    REQUIRE(result(1) == A(2, 2));
}

TEMPLATE_TEST_CASE("view_composition: row of a squeezed corner",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using Squeezed = SqueezeView<Corner, 0, 1>;
    using Row = MultiSliceView<Squeezed, Slice<0, 1, 1>>;

    Tensor A;
    A.setSequencial();

    auto row = Row(Squeezed(Corner(A)));

    REQUIRE(row.getShape() == "(1,4)");
    for (my_size_t j = 0; j < 4; ++j)
    {
        REQUIRE(row(0, j) == A(1, 2, 1, j));
    }

    FusedMatrix<T, 1, 4> result;
    result = row;
    for (my_size_t j = 0; j < 4; ++j)
    {
        REQUIRE(result(0, j) == A(1, 2, 1, j));
    }
}

TEMPLATE_TEST_CASE("view_composition: diagonal of a block",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using M = FusedMatrix<T, 3, 3>;
    using Block = MultiSliceView<M, Slice<0, 1, 2>, Slice<1, 1, 2>>;

    M A;
    A.setSequencial();

    auto diag = DiagonalView<Block>(A.template block<1, 1, 2, 2>());

    REQUIRE(diag(0) == A(1, 1));
    REQUIRE(diag(1) == A(2, 2));

    FusedVector<T, 2> result;
    result = diag;
    REQUIRE(result(0) == A(1, 1));
    REQUIRE(result(1) == A(2, 2));
}

TEMPLATE_TEST_CASE("view_composition: subvector of a column",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using M = FusedMatrix<T, 3, 3>;
    using Col = MultiSliceView<M, Slice<1, 2, 1>>;

    M A;
    A.setSequencial();

    auto last2 = SubVectorView<Col, 1, 2>(A.template column<2>());

    REQUIRE(last2(0) == A(1, 2));
    REQUIRE(last2(1) == A(2, 2));

    FusedVector<T, 2> result;
    result = last2;
    REQUIRE(result(0) == A(1, 2));
    REQUIRE(result(1) == A(2, 2));
}

// ============================================================================
// ORDERS THAT USED TO BE REJECTED
// ============================================================================

TEMPLATE_TEST_CASE("view_composition: slice of a slice equals the flat slice",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using M = FusedMatrix<T, 4, 6>;
    using Rows = MultiSliceView<M, Slice<0, 1, 2>>;

    M A;
    A.setSequencial();

    auto nested = MultiSliceView<Rows, Slice<1, 2, 3>>(Rows(A));
    auto flat = A.template block<1, 2, 2, 3>();

    REQUIRE(nested.getShape() == "(2,3)");
    REQUIRE(nested == flat);
}

TEMPLATE_TEST_CASE("view_composition: transpose of a block",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using M = FusedMatrix<T, 3, 3>;
    using Block = MultiSliceView<M, Slice<0, 1, 2>, Slice<1, 1, 2>>;

    M A;
    A.setSequencial();

    auto t = PermutedViewConstExpr<Block, 1, 0>(A.template block<1, 1, 2, 2>());

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 2; ++j)
        {
            REQUIRE(t(i, j) == A(1 + j, 1 + i));
        }
    }

    FusedMatrix<T, 2, 2> result;
    result = t;
    REQUIRE(result(0, 1) == A(2, 1));
    REQUIRE(result(1, 0) == A(1, 2));
}

TEMPLATE_TEST_CASE("view_composition: squeeze of a transpose",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;
    using Transposed = PermutedViewConstExpr<Tensor, 2, 1, 0>; // [4,3,1]

    Tensor A;
    A.setSequencial();

    auto sq = SqueezeView<Transposed, 2>(Transposed(A));

    REQUIRE(sq.getShape() == "(4,3)");
    for (my_size_t i = 0; i < 4; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(sq(i, j) == A(0, j, i));
        }
    }

    FusedMatrix<T, 4, 3> result;
    result = sq;
    for (my_size_t i = 0; i < 4; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(result(i, j) == A(0, j, i));
        }
    }
}

TEMPLATE_TEST_CASE("view_composition: squeeze of a squeeze",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 1, 3, 4>;
    using Inner = SqueezeView<Tensor, 0>; // [1,3,4]

    Tensor A;
    A.setSequencial();

    auto sq = SqueezeView<Inner, 0>(Inner(A));

    REQUIRE(sq.getShape() == "(3,4)");
    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(sq(i, j) == A(0, 0, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("view_composition: three deep, row of the transposed squeezed corner",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using Squeezed = SqueezeView<Corner, 0, 1>;           // [2,4]
    using Transposed = PermutedViewConstExpr<Squeezed, 1, 0>; // [4,2]
    using Row = MultiSliceView<Transposed, Slice<0, 3, 1>>;   // [1,2]: row 3 of the transpose

    Tensor A;
    A.setSequencial();

    auto row = Row(Transposed(Squeezed(Corner(A))));

    REQUIRE(row.getShape() == "(1,2)");
    REQUIRE(row(0, 0) == A(1, 2, 0, 3));
    REQUIRE(row(0, 1) == A(1, 2, 1, 3));
}

// ============================================================================
// CHAINS IN EXPRESSIONS
// ============================================================================

TEMPLATE_TEST_CASE("view_composition: chains in one expression",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using M = FusedMatrix<T, 3, 3>;
    using Block = MultiSliceView<M, Slice<0, 1, 2>, Slice<1, 1, 2>>;

    M A;
    A.setSequencial();

    FusedMatrix<T, 2, 2> result;
    result = PermutedViewConstExpr<Block, 1, 0>(A.template block<1, 1, 2, 2>()) * T(2) +
             A.template block<1, 1, 2, 2>();

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 2; ++j)
        {
            REQUIRE(result(i, j) == A(1 + j, 1 + i) * T(2) + A(1 + i, 1 + j));
        }
    }
}

TEMPLATE_TEST_CASE("view_composition: a chain of temporaries stored in an expression",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using M = FusedMatrix<T, 3, 3>;
    using Block = MultiSliceView<M, Slice<0, 1, 2>, Slice<1, 1, 2>>;

    M A;
    A.setSequencial();

    // the block and the diagonal are both temporaries of this statement; the
    // expression keeps its own copies (fused/expr_storage.h)
    auto e = DiagonalView<Block>(A.template block<1, 1, 2, 2>()) * T(3);

    FusedVector<T, 2> result;
    result = e;
    REQUIRE(result(0) == A(1, 1) * T(3));
    REQUIRE(result(1) == A(2, 2) * T(3));
}

// ============================================================================
// EINSUM / MATMUL OPERANDS
// ============================================================================

TEMPLATE_TEST_CASE("view_composition: only a transpose of a tensor is a matmul operand",
                   "[view_composition]", double, float)
{
    using T = TestType;
    using M = FusedMatrix<T, 3, 3>;
    using Block = MultiSliceView<M, Slice<0, 1, 2>, Slice<1, 1, 2>>;

    // GEMM reads data() through the layout's strides and never sees a Base, so
    // a transpose of a block is rejected like the block itself
    REQUIRE((FusedMatrix<T, 3, 3>::template is_einsum_operand<PermutedViewConstExpr<M, 1, 0>>));
    REQUIRE_FALSE((FusedMatrix<T, 2, 2>::template is_einsum_operand<PermutedViewConstExpr<Block, 1, 0>>));

    // Rejected at compile time with "constraints not satisfied" on
    // FusedTensorND::is_einsum_operand. Uncomment to check:
    // M A;
    // auto C = FusedMatrix<T, 2, 2>::matmul(PermutedViewConstExpr<Block, 1, 0>(A.template block<1, 1, 2, 2>()), FusedMatrix<T, 2, 2>((T)1));
}

// ============================================================================
// NEGATIVE CASES: COMPILE-TIME ERRORS
// ============================================================================
// These are static_asserts. Uncomment one at a time; each must fail the build.
// Reading a member is required, an unused alias never instantiates.

TEST_CASE("view_composition: bounds come from the view underneath",
          "[view_composition]")
{
    using V = FusedVector<double, 6>;
    using Sub = SubVectorView<V, 2, 3>; // 3 elements
    (void)sizeof(Sub);

    // using View = MultiSliceView<Sub, Slice<0, 1, 5>>; // 5 rows of a 3-row view, though the vector has 6
    // (void)View::NumDims;
}

// ============================================================================
// BOTH DIRECTIONS
// ============================================================================

TEMPLATE_TEST_CASE("view_composition: a chain maps offsets back to its own coordinates",
                   "[view_composition]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 3, 5>;
    using Plane = MultiSliceView<Tensor, Slice<0, 1, 1>>;
    using Flat = SqueezeView<Plane, 0>;
    using L = typename Flat::Layout;

    Tensor t;
    t.setSequencial();
    Plane plane(t);
    Flat view(plane);

    my_size_t coords[2];
    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            const my_size_t offset = L::logical_coords_to_physical_flat(i, j);
            REQUIRE(view.data()[offset] == view(i, j));

            L::physical_flat_to_logical_coords(offset, coords);
            REQUIRE(coords[0] == i);
            REQUIRE(coords[1] == j);
        }
    }
}
