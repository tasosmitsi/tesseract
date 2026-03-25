#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/decomposition/qr.h"
#include "algorithms/operations/determinant.h"

using Catch::Approx;

// ============================================================================
// 3×3 SQUARE — RECONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: 3x3 Q*R = A",
                   "[qr]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_householder(A);

    SECTION("Q * R = A")
    {
        auto Q = qr.Q();
        auto R = qr.R();
        auto QR_product = Matrix::matmul(Q, R);

        REQUIRE(QR_product == A);
    }

    SECTION("Q is orthogonal: QᵀQ = I")
    {
        auto Q = qr.Q();
        auto QtQ = FusedMatrix<T, 3, 3>::matmul(Q.transpose_view(), Q);

        REQUIRE(QtQ.isIdentity());
    }

    SECTION("R is upper-triangular")
    {
        auto R = qr.R();

        REQUIRE(R.isUpperTriangular());
    }
}

// ============================================================================
// 4×3 RECTANGULAR — RECONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: 4x3 rectangular Q*R = A",
                   "[qr]", double, float)
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

    auto qr = matrix_algorithms::qr_householder(A);

    SECTION("Q * R = A")
    {
        auto Q = qr.Q(); // 4×4
        auto R = qr.R(); // 4×3
        auto QR_product = Matrix43::matmul(Q, R);

        REQUIRE(QR_product == A);
    }

    SECTION("Q is orthogonal: QᵀQ = I")
    {
        auto Q = qr.Q();
        auto QtQ = Matrix44::matmul(Q.transpose_view(), Q);

        REQUIRE(QtQ.isIdentity());
    }

    SECTION("R has zeros below diagonal")
    {
        auto R = qr.R();

        // Manual check for M×N rectangular upper triangular
        for (my_size_t i = 1; i < 4; ++i)
        {
            for (my_size_t j = 0; j < i && j < 3; ++j)
            {
                REQUIRE(R(i, j) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
            }
        }
    }
}

// ============================================================================
// IDENTITY MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: identity",
                   "[qr]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I;
    I.setIdentity();

    auto qr = matrix_algorithms::qr_householder(I);

    auto Q = qr.Q();
    auto R = qr.R();

    // Q should be identity (or sign-permuted identity)
    // R should be identity
    auto QR_product = Matrix::matmul(Q, R);
    REQUIRE(QR_product == I);
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: 1x1",
                   "[qr]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{7}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_householder(A);

    auto Q = qr.Q();
    auto R = qr.R();

    auto QR_product = Matrix::matmul(Q, R);
    REQUIRE(QR_product == A);
}

// ============================================================================
// 2×1 TALL THIN
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: 2x1 tall thin",
                   "[qr]", double, float)
{
    using T = TestType;
    using Matrix21 = FusedMatrix<T, 2, 1>;

    T A_vals[2][1] = {{3}, {4}};
    Matrix21 A(A_vals);

    auto qr = matrix_algorithms::qr_householder(A);

    auto Q = qr.Q(); // 2×2
    auto R = qr.R(); // 2×1

    auto QR_product = Matrix21::matmul(Q, R);
    REQUIRE(QR_product == A);

    // R(0,0) should be ±norm([3,4]) = ±5
    REQUIRE(math::abs(R(0, 0)) == Approx(T(5)));

    // R(1,0) should be 0
    REQUIRE(R(1, 0) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
}

// ============================================================================
// APPLY_QT — MATCHES Q^T * b
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: apply_Qt matches Q^T * b",
                   "[qr]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}};
    Matrix A(A_vals);

    T b_vals[3][1] = {{1}, {2}, {3}};
    Vector b(b_vals);

    auto qr = matrix_algorithms::qr_householder(A);

    // Method 1: apply_Qt
    auto Qtb_fast = qr.apply_Qt(b);

    // Method 2: form Q explicitly, then multiply
    auto Q = qr.Q();
    Vector Qtb_explicit(T(0));
    for (my_size_t i = 0; i < 3; ++i)
    {
        T sum = T(0);
        for (my_size_t k = 0; k < 3; ++k)
        {
            sum += Q(k, i) * b(k); // Qᵀ(i,k) = Q(k,i)
        }
        Qtb_explicit(i) = sum;
    }

    REQUIRE(Qtb_fast == Qtb_explicit);
}

// ============================================================================
// APPLY_QT — RECTANGULAR (4×3)
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: apply_Qt rectangular",
                   "[qr]", double, float)
{
    using T = TestType;
    using Matrix43 = FusedMatrix<T, 4, 3>;
    using Vector = FusedVector<T, 4>;

    T A_vals[4][3] = {
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}};
    Matrix43 A(A_vals);

    T b_vals[4][1] = {{1}, {2}, {3}, {4}};
    Vector b(b_vals);

    auto qr = matrix_algorithms::qr_householder(A);

    auto Qtb_fast = qr.apply_Qt(b);

    auto Q = qr.Q(); // 4×4
    Vector Qtb_explicit(T(0));
    for (my_size_t i = 0; i < 4; ++i)
    {
        T sum = T(0);
        for (my_size_t k = 0; k < 4; ++k)
        {
            sum += Q(k, i) * b(k);
        }
        Qtb_explicit(i) = sum;
    }

    REQUIRE(Qtb_fast == Qtb_explicit);
}

// ============================================================================
// DIAGONAL MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: diagonal matrix",
                   "[qr]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 0, 0},
        {0, 3, 0},
        {0, 0, 5}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_householder(A);

    auto Q = qr.Q();
    auto R = qr.R();
    auto QR_product = Matrix::matmul(Q, R);

    REQUIRE(QR_product == A);
    REQUIRE(R.isUpperTriangular());
}

// ============================================================================
// PROPERTY: Q IS ORTHOGONAL — 5×3 RECTANGULAR
// ============================================================================

TEST_CASE("qr_householder: 5x3 Q is orthogonal",
          "[qr]")
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

    auto qr = matrix_algorithms::qr_householder(A);

    auto Q = qr.Q();
    auto QtQ = Matrix55::matmul(Q.transpose_view(), Q);

    REQUIRE(QtQ.isIdentity());

    auto R = qr.R();
    auto QR_product = Matrix53::matmul(Q, R);
    REQUIRE(QR_product == A);
}

// ============================================================================
// GENERIC PATH — 7×5
// ============================================================================

TEST_CASE("qr_householder: 7x5 reconstruction",
          "[qr]")
{
    using T = double;
    using Matrix75 = FusedMatrix<T, 7, 5>;
    using Matrix77 = FusedMatrix<T, 7, 7>;

    // Build a well-conditioned rectangular matrix
    Matrix75 A;
    for (my_size_t i = 0; i < 7; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            A(i, j) = T(1) / T(i + j + 1); // Hilbert-like but rectangular
        }
        if (i < 5)
            A(i, i) += T(10); // add diagonal dominance
    }

    auto qr = matrix_algorithms::qr_householder(A);

    auto Q = qr.Q();
    auto R = qr.R();

    // Q is orthogonal
    auto QtQ = Matrix77::matmul(Q.transpose_view(), Q);
    REQUIRE(QtQ.isIdentity());

    // Q*R = A
    auto QR_product = Matrix75::matmul(Q, R);
    REQUIRE(QR_product == A);

    // R has zeros below diagonal
    for (my_size_t i = 1; i < 7; ++i)
    {
        for (my_size_t j = 0; j < i && j < 5; ++j)
        {
            REQUIRE(R(i, j) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
        }
    }
}

// ============================================================================
// ZERO MATRIX — Q=I, R=0
// ============================================================================

TEMPLATE_TEST_CASE("qr_householder: zero matrix",
                   "[qr]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A(T(0));

    auto qr = matrix_algorithms::qr_householder(A);

    auto Q = qr.Q();
    auto R = qr.R();

    // Q should be identity (no reflections applied)
    REQUIRE(Q.isIdentity());

    // R should be zero
    Matrix zero(T(0));
    REQUIRE(R == zero);

    // Q*R = A still holds trivially
    auto QR_product = Matrix::matmul(Q, R);
    REQUIRE(QR_product == A);
}

// ============================================================================
// RANK-DEFICIENT MATRIX — QR still succeeds
// ============================================================================

TEST_CASE("qr_householder: rank-deficient matrix",
          "[qr]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    // Column 2 = Column 0 + Column 1 → rank 2
    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 9},
        {7, 8, 15}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_householder(A);

    SECTION("Q * R = A")
    {
        auto Q = qr.Q();
        auto R = qr.R();
        auto QR_product = Matrix::matmul(Q, R);

        REQUIRE(QR_product == A);
    }

    SECTION("Q is orthogonal")
    {
        auto Q = qr.Q();
        auto QtQ = Matrix::matmul(Q.transpose_view(), Q);

        REQUIRE(QtQ.isIdentity());
    }

    SECTION("R has near-zero on last diagonal (rank deficiency)")
    {
        auto R = qr.R();

        // R(2,2) should be ~0 since column 3 is linearly dependent
        REQUIRE(math::abs(R(2, 2)) < T(1e-6));
    }
}

// ============================================================================
// APPLY_QT — 7×5 GENERIC PATH
// ============================================================================

TEST_CASE("qr_householder: apply_Qt 7x5 generic path",
          "[qr]")
{
    using T = double;
    using Matrix75 = FusedMatrix<T, 7, 5>;
    using Vector = FusedVector<T, 7>;

    Matrix75 A;
    for (my_size_t i = 0; i < 7; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            A(i, j) = T(1) / T(i + j + 1);
        }
        if (i < 5)
            A(i, i) += T(10);
    }

    Vector b;
    for (my_size_t i = 0; i < 7; ++i)
    {
        b(i) = T(i + 1);
    }

    auto qr = matrix_algorithms::qr_householder(A);

    // Method 1: apply_Qt
    auto Qtb_fast = qr.apply_Qt(b);

    // Method 2: explicit Q^T * b
    auto Q = qr.Q();
    Vector Qtb_explicit(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        T sum = T(0);
        for (my_size_t k = 0; k < 7; ++k)
        {
            sum += Q(k, i) * b(k);
        }
        Qtb_explicit(i) = sum;
    }

    REQUIRE(Qtb_fast == Qtb_explicit);
}

// ============================================================================
// PROPERTY: det(Q) = ±1
// ============================================================================

TEST_CASE("qr_householder: det(Q) = +/- 1",
          "[qr]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}};
    Matrix A(A_vals);

    auto qr = matrix_algorithms::qr_householder(A);
    auto Q = qr.Q();

    T det_Q = matrix_algorithms::determinant(Q);

    REQUIRE(math::abs(det_Q) == Approx(T(1)));
}

// ============================================================================
// PROPERTY: det(Q) = ±1 — RECTANGULAR (4×3, Q is 4×4)
// ============================================================================

TEST_CASE("qr_householder: det(Q) = +/- 1 rectangular",
          "[qr]")
{
    using T = double;
    using Matrix43 = FusedMatrix<T, 4, 3>;

    T A_vals[4][3] = {
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}};
    Matrix43 A(A_vals);

    auto qr = matrix_algorithms::qr_householder(A);
    auto Q = qr.Q(); // 4×4

    T det_Q = matrix_algorithms::determinant(Q);

    REQUIRE(math::abs(det_Q) == Approx(T(1)));
}
