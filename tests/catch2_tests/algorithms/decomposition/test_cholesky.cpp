/**
 * @file test_cholesky.cpp
 * @brief Catch2 tests for matrix_algorithms::cholesky and cholesky_or_die.
 *
 * Tests cover:
 *   - Known-answer: hand-computed L for small SPD matrices
 *   - Property: L * Lᵀ ≈ A reconstruction (via FusedMatrix::matmul)
 *   - Property: L is lower-triangular (via FusedMatrix::isLowerTriangular)
 *   - Error: non-symmetric input → MatrixStatus::NotSymmetric
 *   - Error: non-positive-definite input → MatrixStatus::NotPositiveDefinite
 *   - Edge case: 1×1 matrix
 *   - Edge case: identity matrix
 *   - Edge case: diagonal matrix
 *
 * ============================================================================
 * TEST MATRICES
 * ============================================================================
 *
 * 3×3 SPD matrix A:
 *   [  4   12  -16 ]       L:  [  2   0   0 ]
 *   [ 12   37  -43 ]   →       [  6   1   0 ]
 *   [-16  -43   98 ]           [ -8   5   3 ]
 *
 *   Verify: L * Lᵀ = A
 *
 * 2×2 SPD matrix B:
 *   [ 25  15 ]       L:  [ 5  0 ]
 *   [ 15  18 ]   →       [ 3  3 ]
 *
 * ============================================================================
 */

#include <catch_amalgamated.hpp>

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "algorithms/decomposition/cholesky.h"
#include "fused/fused_matrix.h"

using Catch::Approx;
using matrix_traits::MatrixStatus;

// ============================================================================
// KNOWN-ANSWER: 3×3 SPD MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: 3x3 known-answer SPD matrix",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T vals[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};
    Matrix A(vals);

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE(result.has_value());

    auto &L = result.value();

    SECTION("L has correct known values")
    {
        T expected_vals[3][3] = {
            {2, 0, 0},
            {6, 1, 0},
            {-8, 5, 3}};
        Matrix L_expected(expected_vals);
        REQUIRE(L == L_expected);
    }

    SECTION("L is lower-triangular")
    {
        REQUIRE(L.isLowerTriangular());
    }

    SECTION("L * Lᵀ reconstructs A")
    {
        auto LLt = Matrix::matmul(L, L.transpose_view());
        REQUIRE(LLt == A);
    }
}

// ============================================================================
// KNOWN-ANSWER: 2×2 SPD MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: 2x2 known-answer SPD matrix",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T vals[2][2] = {
        {25, 15},
        {15, 18}};
    Matrix A(vals);

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE(result.has_value());

    auto &L = result.value();

    SECTION("L has correct known values")
    {
        T expected_vals[2][2] = {
            {5, 0},
            {3, 3}};
        Matrix L_expected(expected_vals);
        REQUIRE(L == L_expected);
    }

    SECTION("L * Lᵀ reconstructs A")
    {
        auto LLt = Matrix::matmul(L, L.transpose_view());
        REQUIRE(LLt == A);
    }
}

// ============================================================================
// EDGE CASE: 1×1 MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: 1x1 matrix",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    Matrix A(0);
    A(0, 0) = 9;

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value()(0, 0) == Approx(3.0));
}

// ============================================================================
// EDGE CASE: IDENTITY MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: identity matrix returns identity",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I(0);
    I.setIdentity();

    auto result = matrix_algorithms::cholesky(I);

    REQUIRE(result.has_value());

    auto &L = result.value();

    // cholesky(I) = I
    REQUIRE(L.isIdentity());
}

// ============================================================================
// EDGE CASE: DIAGONAL SPD MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: diagonal SPD matrix",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A(0);
    A(0, 0) = 4;
    A(1, 1) = 9;
    A(2, 2) = 16;

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE(result.has_value());

    auto &L = result.value();

    // L is diagonal with sqrt of each diagonal entry
    T expected_vals[3][3] = {
        {2, 0, 0},
        {0, 3, 0},
        {0, 0, 4}};
    Matrix L_expected(expected_vals);
    REQUIRE(L == L_expected);

    REQUIRE(L.isLowerTriangular());
}

// ============================================================================
// ERROR: NON-SYMMETRIC INPUT
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: non-symmetric matrix returns NotSymmetric",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T vals[2][2] = {
        {4, 2},
        {999, 5} // A(1,0) != A(0,1)
    };
    Matrix A(vals);

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::NotSymmetric);
}

// ============================================================================
// ERROR: NOT POSITIVE DEFINITE
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: non-positive-definite matrix returns error",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    // Symmetric but not positive definite: eigenvalues are -1 and 5
    T vals[2][2] = {
        {2, 3},
        {3, 2}};
    Matrix A(vals);

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::NotPositiveDefinite);
}

TEMPLATE_TEST_CASE("cholesky: zero matrix returns NotPositiveDefinite",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    Matrix A(0); // all zeros — symmetric but not positive definite

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::NotPositiveDefinite);
}

TEMPLATE_TEST_CASE("cholesky: negative diagonal returns NotPositiveDefinite",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T vals[2][2] = {
        {-4, 0},
        {0, -4}};
    Matrix A(vals);

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::NotPositiveDefinite);
}

// ============================================================================
// PROPERTY: RECONSTRUCTION FOR LARGER MATRIX (4×4)
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: 4x4 reconstruction property",
                   "[cholesky]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // Build SPD matrix: A = M * Mᵀ (guarantees positive definiteness)
    // Using a simple lower-triangular M:
    //   [ 2  0  0  0 ]
    //   [ 1  3  0  0 ]
    //   [ 0  1  2  0 ]
    //   [ 1  0  1  4 ]
    //
    // A = M * Mᵀ:
    //   [  4   2   0   2 ]
    //   [  2  10   3   1 ]
    //   [  0   3   5   2 ]
    //   [  2   1   2  18 ]

    T vals[4][4] = {
        {4, 2, 0, 2},
        {2, 10, 3, 1},
        {0, 3, 5, 2},
        {2, 1, 2, 18}};
    Matrix A(vals);

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE(result.has_value());

    auto &L = result.value();

    SECTION("L has correct known values")
    {
        // L should equal the M we used to construct A = M * Mᵀ
        T expected_vals[4][4] = {
            {2, 0, 0, 0},
            {1, 3, 0, 0},
            {0, 1, 2, 0},
            {1, 0, 1, 4}};
        Matrix L_expected(expected_vals);
        REQUIRE(L == L_expected);
    }

    SECTION("L is lower-triangular")
    {
        REQUIRE(L.isLowerTriangular());
    }

    SECTION("L * Lᵀ reconstructs A")
    {
        auto LLt = Matrix::matmul(L, L.transpose_view());
        REQUIRE(LLt == A);
    }
}

// ============================================================================
// PROPERTY: RECONSTRUCTION FOR LARGE MATRIX (20×20)
// ============================================================================

TEMPLATE_TEST_CASE("cholesky: 20x20 reconstruction property",
                   "[cholesky]", double)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 20, 20>;

    // Build a known lower-triangular M, then A = M * Mᵀ (guaranteed SPD).
    // M has diagonal dominance to keep the matrix well-conditioned.
    //
    // Strategy:
    //   M(i,i) = i + 2          (diagonal: 2, 3, 4, ..., 21)
    //   M(i,j) = 1/(i-j+1)     (below diagonal, decaying off-diagonal)
    //   M(i,j) = 0              (above diagonal)

    Matrix M(0);

    for (my_size_t i = 0; i < 20; ++i)
    {
        M(i, i) = T(i + 2); // strong diagonal

        for (my_size_t j = 0; j < i; ++j)
        {
            M(i, j) = T(1) / T(i - j + 1); // decaying sub-diagonal
        }
    }

    // A = M * Mᵀ — symmetric positive definite by construction
    Matrix A = Matrix::matmul(M, M.transpose_view());

    auto result = matrix_algorithms::cholesky(A);

    REQUIRE(result.has_value());

    auto &L = result.value();

    REQUIRE(L.isLowerTriangular());

    auto LLt = Matrix::matmul(L, L.transpose_view());
    REQUIRE(LLt == A);
}
