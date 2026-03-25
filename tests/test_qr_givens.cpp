#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/decomposition/qr_givens.h"
#include "algorithms/decomposition/qr.h"
#include "algorithms/operations/determinant.h"

using Catch::Approx;

// ============================================================================
// 3×3 SQUARE — RECONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("qr_givens: 3x3 Q*R = A",
                   "[qr_givens]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    SECTION("Q * R = A")
    {
        auto QR_product = Matrix::matmul(qr.Q, qr.R);
        REQUIRE(QR_product == A);
    }

    SECTION("Q is orthogonal: QᵀQ = I")
    {
        auto QtQ = Matrix::matmul(qr.Q.transpose_view(), qr.Q);
        REQUIRE(QtQ.isIdentity());
    }

    SECTION("R is upper-triangular")
    {
        REQUIRE(qr.R.isUpperTriangular());
    }
}

// ============================================================================
// 4×3 RECTANGULAR — RECONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("qr_givens: 4x3 rectangular Q*R = A",
                   "[qr_givens]", double, float)
{
    using T = TestType;
    using Matrix43 = FusedMatrix<T, 4, 3>;
    using Matrix44 = FusedMatrix<T, 4, 4>;

    T A_vals[4][3] = {
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}};
    Matrix43 A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    SECTION("Q * R = A")
    {
        auto QR_product = Matrix43::matmul(qr.Q, qr.R);
        REQUIRE(QR_product == A);
    }

    SECTION("Q is orthogonal: QᵀQ = I")
    {
        auto QtQ = Matrix44::matmul(qr.Q.transpose_view(), qr.Q);
        REQUIRE(QtQ.isIdentity());
    }

    SECTION("R has zeros below diagonal")
    {
        for (my_size_t i = 1; i < 4; ++i)
        {
            for (my_size_t j = 0; j < i && j < 3; ++j)
            {
                REQUIRE(qr.R(i, j) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
            }
        }
    }
}

// ============================================================================
// IDENTITY MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("qr_givens: identity",
                   "[qr_givens]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I;
    I.setIdentity();

    auto qr = matrix_algorithms::qr_givens(I);

    auto QR_product = Matrix::matmul(qr.Q, qr.R);
    REQUIRE(QR_product == I);
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("qr_givens: 1x1",
                   "[qr_givens]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{7}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    auto QR_product = Matrix::matmul(qr.Q, qr.R);
    REQUIRE(QR_product == A);
}

// ============================================================================
// 2×1 TALL THIN
// ============================================================================

TEMPLATE_TEST_CASE("qr_givens: 2x1 tall thin",
                   "[qr_givens]", double, float)
{
    using T = TestType;
    using Matrix21 = FusedMatrix<T, 2, 1>;

    T A_vals[2][1] = {{3}, {4}};
    Matrix21 A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    auto QR_product = Matrix21::matmul(qr.Q, qr.R);
    REQUIRE(QR_product == A);

    // R(0,0) should be ±norm([3,4]) = ±5
    REQUIRE(math::abs(qr.R(0, 0)) == Approx(T(5)));

    // R(1,0) should be 0
    REQUIRE(qr.R(1, 0) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
}

// ============================================================================
// DIAGONAL MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("qr_givens: diagonal matrix",
                   "[qr_givens]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 0, 0},
        {0, 3, 0},
        {0, 0, 5}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    auto QR_product = Matrix::matmul(qr.Q, qr.R);
    REQUIRE(QR_product == A);
    REQUIRE(qr.R.isUpperTriangular());
}

// ============================================================================
// ZERO MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("qr_givens: zero matrix",
                   "[qr_givens]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A(T(0));

    auto qr = matrix_algorithms::qr_givens(A);

    // Q should be identity (no rotations applied)
    REQUIRE(qr.Q.isIdentity());

    // R should be zero
    Matrix zero(T(0));
    REQUIRE(qr.R == zero);
}

// ============================================================================
// RANK-DEFICIENT
// ============================================================================

TEST_CASE("qr_givens: rank-deficient matrix",
          "[qr_givens]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    // Column 2 = Column 0 + Column 1 → rank 2
    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 9},
        {7, 8, 15}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    SECTION("Q * R = A")
    {
        auto QR_product = Matrix::matmul(qr.Q, qr.R);
        REQUIRE(QR_product == A);
    }

    SECTION("Q is orthogonal")
    {
        auto QtQ = Matrix::matmul(qr.Q.transpose_view(), qr.Q);
        REQUIRE(QtQ.isIdentity());
    }

    SECTION("R has near-zero on last diagonal")
    {
        REQUIRE(math::abs(qr.R(2, 2)) < T(1e-6));
    }
}

// ============================================================================
// PROPERTY: det(Q) = ±1
// ============================================================================

TEST_CASE("qr_givens: det(Q) = +/- 1",
          "[qr_givens]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    T det_Q = matrix_algorithms::determinant(qr.Q);
    REQUIRE(math::abs(det_Q) == Approx(T(1)));
}

// ============================================================================
// PROPERTY: det(Q) = ±1 — RECTANGULAR
// ============================================================================

TEST_CASE("qr_givens: det(Q) = +/- 1 rectangular",
          "[qr_givens]")
{
    using T = double;
    using Matrix43 = FusedMatrix<T, 4, 3>;

    T A_vals[4][3] = {
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}};
    Matrix43 A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    T det_Q = matrix_algorithms::determinant(qr.Q);
    REQUIRE(math::abs(det_Q) == Approx(T(1)));
}

// ============================================================================
// AGREES WITH HOUSEHOLDER
// ============================================================================

TEST_CASE("qr_givens: R diagonal matches Householder R diagonal",
          "[qr_givens]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}};
    Matrix A(A_vals);

    auto givens = matrix_algorithms::qr_givens(A);

    // |R(i,i)| should match between Householder and Givens
    // (signs may differ due to different rotation conventions)
    auto householder = matrix_algorithms::qr_householder(A);
    auto R_h = householder.R();

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(math::abs(givens.R(i, i)) == Approx(math::abs(R_h(i, i))));
    }
}

// ============================================================================
// 5×3 RECTANGULAR
// ============================================================================

TEST_CASE("qr_givens: 5x3 Q is orthogonal",
          "[qr_givens]")
{
    using T = double;
    using Matrix53 = FusedMatrix<T, 5, 3>;
    using Matrix55 = FusedMatrix<T, 5, 5>;

    T A_vals[5][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 0},
        {-1, 3, 2},
        {2, -1, 4}};
    Matrix53 A(A_vals);

    auto qr = matrix_algorithms::qr_givens(A);

    auto QtQ = Matrix55::matmul(qr.Q.transpose_view(), qr.Q);
    REQUIRE(QtQ.isIdentity());

    auto QR_product = Matrix53::matmul(qr.Q, qr.R);
    REQUIRE(QR_product == A);
}

// ============================================================================
// GENERIC PATH — 7×5
// ============================================================================

TEST_CASE("qr_givens: 7x5 reconstruction",
          "[qr_givens]")
{
    using T = double;
    using Matrix75 = FusedMatrix<T, 7, 5>;
    using Matrix77 = FusedMatrix<T, 7, 7>;

    Matrix75 A(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            A(i, j) = T(1) / T(i + j + 1);
        }
        if (i < 5)
            A(i, i) += T(10);
    }

    auto qr = matrix_algorithms::qr_givens(A);

    // Q is orthogonal
    auto QtQ = Matrix77::matmul(qr.Q.transpose_view(), qr.Q);
    REQUIRE(QtQ.isIdentity());

    // Q*R = A
    auto QR_product = Matrix75::matmul(qr.Q, qr.R);
    REQUIRE(QR_product == A);

    // R has zeros below diagonal
    for (my_size_t i = 1; i < 7; ++i)
    {
        for (my_size_t j = 0; j < i && j < 5; ++j)
        {
            REQUIRE(qr.R(i, j) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
        }
    }
}
