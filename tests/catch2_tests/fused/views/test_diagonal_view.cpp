#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "fused/views/diagonal_view.h"
#include "fused/views/subvector_view.h"

using Catch::Approx;

// ============================================================================
// VIEW SHAPE: DIMS AND SHAPE STRING
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: dims and shape",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix3 = FusedMatrix<T, 3, 3>;
    using Matrix7 = FusedMatrix<T, 7, 7>;

    SECTION("3x3 diagonal reports [3, 1]")
    {
        Matrix3 A;
        A.setSequencial();

        auto view = DiagonalView<Matrix3>(A);

        REQUIRE(view.getDim(0) == 3);
        REQUIRE(view.getDim(1) == 1);
        REQUIRE(view.getNumDims() == 2);
        REQUIRE(view.getTotalSize() == 3);
        REQUIRE(view.getShape() == "(3,1)");
    }

    SECTION("7x7 diagonal reports [7, 1]")
    {
        Matrix7 A;
        A.setSequencial();

        auto view = DiagonalView<Matrix7>(A);

        REQUIRE(view.getDim(0) == 7);
        REQUIRE(view.getDim(1) == 1);
        REQUIRE(view.getNumDims() == 2);
        REQUIRE(view.getTotalSize() == 7);
        REQUIRE(view.getShape() == "(7,1)");
    }
}

// ============================================================================
// EXTRACTION: 1×1
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: 1x1",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;
    using Vec1 = FusedVector<T, 1>;

    T A_vals[1][1] = {{42}};
    Matrix A(A_vals);

    Vec1 result;
    result = DiagonalView<Matrix>(A);

    REQUIRE(result(0) == T(42));
}

// ============================================================================
// EXTRACTION: 3×3
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: 3x3 extraction",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    Matrix A(A_vals);

    Vec3 result;
    result = DiagonalView<Matrix>(A);

    REQUIRE(result(0) == T(1));
    REQUIRE(result(1) == T(5));
    REQUIRE(result(2) == T(9));
}

// ============================================================================
// EXTRACTION: 4×4
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: 4x4 extraction",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vec4 = FusedVector<T, 4>;

    Matrix A;
    A.setSequencial();
    // [ 0  1  2  3]
    // [ 4  5  6  7]
    // [ 8  9 10 11]
    // [12 13 14 15]

    Vec4 result;
    result = DiagonalView<Matrix>(A);

    REQUIRE(result(0) == T(0));
    REQUIRE(result(1) == T(5));
    REQUIRE(result(2) == T(10));
    REQUIRE(result(3) == T(15));
}

// ============================================================================
// EXTRACTION: 5×5
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: 5x5 extraction",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 5, 5>;
    using Vec5 = FusedVector<T, 5>;

    Matrix A;
    A.setSequencial();
    // diag(i) = i * 5 + i = i * 6

    Vec5 result;
    result = DiagonalView<Matrix>(A);

    for (my_size_t i = 0; i < 5; ++i)
    {
        REQUIRE(result(i) == T(i * 6));
    }
}

// ============================================================================
// EXTRACTION: 6×6
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: 6x6 extraction",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 6, 6>;
    using Vec6 = FusedVector<T, 6>;

    Matrix A;
    A.setSequencial();
    // diag(i) = i * 6 + i = i * 7

    Vec6 result;
    result = DiagonalView<Matrix>(A);

    for (my_size_t i = 0; i < 6; ++i)
    {
        REQUIRE(result(i) == T(i * 7));
    }
}

// ============================================================================
// EXTRACTION: 7×7
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: 7x7 extraction",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 7, 7>;
    using Vec7 = FusedVector<T, 7>;

    Matrix A;
    A.setSequencial();
    // diag(i) = i * 7 + i = i * 8

    Vec7 result;
    result = DiagonalView<Matrix>(A);

    for (my_size_t i = 0; i < 7; ++i)
    {
        REQUIRE(result(i) == T(i * 8));
    }
}

// ============================================================================
// IDENTITY MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: identity diagonal is all ones",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 5, 5>;
    using Vec5 = FusedVector<T, 5>;

    Matrix I(T(0));
    I.setIdentity();

    Vec5 result;
    result = DiagonalView<Matrix>(I);

    for (my_size_t i = 0; i < 5; ++i)
    {
        REQUIRE(result(i) == T(1));
    }
}

// ============================================================================
// ZERO MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: zero matrix diagonal is zero",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vec4 = FusedVector<T, 4>;

    Matrix A(T(0));

    Vec4 result;
    result = DiagonalView<Matrix>(A);

    Vec4 zero(T(0));
    REQUIRE(result == zero);
}

// ============================================================================
// OFF-DIAGONALS IGNORED
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: reads only the diagonal",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    // Large off-diagonal values that must not leak into the result
    T A_vals[3][3] = {
        {2, 99, 99},
        {99, 4, 99},
        {99, 99, 8}};
    Matrix A(A_vals);

    Vec3 result;
    result = DiagonalView<Matrix>(A);

    REQUIRE(result(0) == T(2));
    REQUIRE(result(1) == T(4));
    REQUIRE(result(2) == T(8));
}

// ============================================================================
// ASYMMETRIC MATRIX: UPPER AND LOWER TRIANGLES DIFFER
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: asymmetric matrix",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vec4 = FusedVector<T, 4>;

    T A_vals[4][4] = {
        {10, 1, 2, 3},
        {90, 20, 4, 5},
        {91, 92, 30, 6},
        {93, 94, 95, 40}};
    Matrix A(A_vals);

    Vec4 result;
    result = DiagonalView<Matrix>(A);

    REQUIRE(result(0) == T(10));
    REQUIRE(result(1) == T(20));
    REQUIRE(result(2) == T(30));
    REQUIRE(result(3) == T(40));
}

// ============================================================================
// ELEMENT ACCESS VIA operator()
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: operator() access",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A;
    A.setSequencial();
    // [0 1 2]
    // [3 4 5]
    // [6 7 8]

    auto view = DiagonalView<Matrix>(A);

    REQUIRE(view(0) == T(0));
    REQUIRE(view(1) == T(4));
    REQUIRE(view(2) == T(8));
}

// ============================================================================
// VIEW DOES NOT COPY & DOES NOT MATERIALIZE JUST READS SOURCE DATA
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: reads live source data",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A(T(0));

    auto view = DiagonalView<Matrix>(A); // it does not call assignment operator.

    // Modify source after view creation
    A(1, 1) = T(7);

    REQUIRE(view(1) == T(7));
}

// ============================================================================
// EXPRESSION TEMPLATE: DIAGONAL + DIAGONAL
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: diagonal + diagonal",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    Matrix A, B;
    A.setSequencial(); // diag = [0, 4, 8]
    B.setSequencial(); // diag = [0, 4, 8]

    Vec3 result;
    result = DiagonalView<Matrix>(A) + DiagonalView<Matrix>(B);

    // [0,4,8] + [0,4,8] = [0,8,16]
    REQUIRE(result(0) == T(0));
    REQUIRE(result(1) == T(8));
    REQUIRE(result(2) == T(16));
}

// ============================================================================
// EXPRESSION TEMPLATE: DIAGONAL + DIAGONAL + DIAGONAL
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: diagonal + diagonal + diagonal",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    Matrix A, B, C;
    A.setSequencial(); // diag = [0, 4, 8]
    B.setSequencial();
    C.setSequencial();

    Vec3 result;
    result = DiagonalView<Matrix>(A) + DiagonalView<Matrix>(B) + DiagonalView<Matrix>(C);

    // [0,4,8] * 3 = [0,12,24]
    REQUIRE(result(0) == T(0));
    REQUIRE(result(1) == T(12));
    REQUIRE(result(2) == T(24));
}

// ============================================================================
// EXPRESSION: DIAGONAL - DIAGONAL
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: diagonal - diagonal",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    Matrix A, B;
    A.setSequencial(); // diag = [0, 4, 8]
    B.setSequencial(); // diag = [0, 4, 8]

    Vec3 result;
    result = DiagonalView<Matrix>(A) - DiagonalView<Matrix>(B);

    Vec3 zero(T(0));
    REQUIRE(result == zero);
}

// ============================================================================
// EXPRESSION TEMPLATE: DIAGONAL * SCALAR
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: diagonal * scalar",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    Matrix A;
    A.setSequencial(); // diag = [0, 4, 8]

    Vec3 result;
    result = DiagonalView<Matrix>(A) * T(3);

    // [0,4,8] * 3 = [0,12,24]
    REQUIRE(result(0) == T(0));
    REQUIRE(result(1) == T(12));
    REQUIRE(result(2) == T(24));
}

// ============================================================================
// EXPRESSION TEMPLATE: DIAGONAL + SUBVECTOR VIEW
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: diagonal + subvector view",
                   "[diagonal_view][subvector_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Matrix A;
    A.setSequencial(); // diag = [0, 4, 8]

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    Vec3 result;
    result = DiagonalView<Matrix>(A) + v.template segment<1, 3>();

    // [0,4,8] + [1,2,3] = [1,6,11]
    REQUIRE(result(0) == T(1));
    REQUIRE(result(1) == T(6));
    REQUIRE(result(2) == T(11));
}

// ============================================================================
// EQUALITY: DIAGONAL VS EXPLICIT VECTOR
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: equality with explicit diagonal",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {2, 0, 0},
        {0, 5, 0},
        {0, 0, 9}};
    Matrix A(A_vals);

    T d_vals[3][1] = {{2}, {5}, {9}};
    Vec3 d(d_vals);

    REQUIRE(d == DiagonalView<Matrix>(A));
}

// ============================================================================
// EQUALITY: MATCHING DIAGONALS
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: matching diagonals are equal",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A, B;
    A.setSequencial();
    B.setSequencial();

    // Same data
    REQUIRE(DiagonalView<Matrix>(A) == DiagonalView<Matrix>(B));
}

// ============================================================================
// DIFFERENT SOURCE SIZES: SAME DIAGONAL PREFIX
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: different source sizes same prefix",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix3 = FusedMatrix<T, 3, 3>;
    using Matrix5 = FusedMatrix<T, 5, 5>;
    using Vec3 = FusedVector<T, 3>;
    using Vec5 = FusedVector<T, 5>;

    Matrix3 A(T(0));
    Matrix5 B(T(0));

    for (my_size_t i = 0; i < 3; ++i)
        A(i, i) = T(i + 1);

    for (my_size_t i = 0; i < 5; ++i)
        B(i, i) = T(i + 1);

    Vec3 ra;
    Vec5 rb;
    ra = DiagonalView<Matrix3>(A);
    rb = DiagonalView<Matrix5>(B);

    // First 3 diagonal entries match
    REQUIRE(ra(0) == rb(0));
    REQUIRE(ra(1) == rb(1));
    REQUIRE(ra(2) == rb(2));
}

// ============================================================================
// NEGATIVE AND MIXED VALUES
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: negative and mixed values",
                   "[diagonal_view]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {-2.5, 1, 1},
        {1, 0, 1},
        {1, 1, 7.25}};
    Matrix A(A_vals);

    Vec3 result;
    result = DiagonalView<Matrix>(A);

    REQUIRE(result(0) == T(-2.5));
    REQUIRE(result(1) == T(0));
    REQUIRE(result(2) == T(7.25));
}

// ============================================================================
// FusedMatrix::diagonal
// ============================================================================

TEMPLATE_TEST_CASE("diagonal_view: FusedMatrix::diagonal",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    Matrix A;
    A.setSequencial();

    auto view = A.diagonal();

    REQUIRE(view.getShape() == "(4,1)");

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(view(i) == A(i, i));
    }
}

TEMPLATE_TEST_CASE("diagonal_view: diagonal matches the explicit view",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 5, 5>;

    Matrix A;
    A.setSequencial();

    REQUIRE(A.diagonal() == DiagonalView<Matrix>(A));
}

TEMPLATE_TEST_CASE("diagonal_view: assign from diagonal",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vec3 = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {2, 9, 9},
        {9, 5, 9},
        {9, 9, 8}};
    Matrix A(A_vals);

    Vec3 result;
    result = A.diagonal();

    REQUIRE(result(0) == T(2));
    REQUIRE(result(1) == T(5));
    REQUIRE(result(2) == T(8));
}

TEMPLATE_TEST_CASE("diagonal_view: diagonal in an expression",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vec4 = FusedVector<T, 4>;

    Matrix A, B;
    A.setSequencial();
    B.setSequencial();

    Vec4 result;
    result = A.diagonal() + B.diagonal();

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(result(i) == A(i, i) + B(i, i));
    }
}

TEMPLATE_TEST_CASE("diagonal_view: diagonal of the identity",
                   "[diagonal_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 5, 5>;
    using Vec5 = FusedVector<T, 5>;

    Matrix I(T(0));
    I.setIdentity();

    Vec5 result;
    result = I.diagonal();

    for (my_size_t i = 0; i < 5; ++i)
    {
        REQUIRE(result(i) == T(1));
    }
}

TEST_CASE("diagonal_view: diagonal plus a column slice",
          "[diagonal_view]")
{
    using Matrix = FusedMatrix<double, 4, 4>;
    using Vec4 = FusedVector<double, 4>;

    Matrix A;
    A.setSequencial();

    // A diagonal view and a slice view in one expression
    Vec4 result;
    result = A.diagonal() + A.column<0>();

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(result(i) == A(i, i) + A(i, 0));
    }
}
