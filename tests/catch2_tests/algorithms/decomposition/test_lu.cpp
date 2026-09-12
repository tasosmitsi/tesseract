#include <catch_amalgamated.hpp>

#include "config.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/decomposition/lu.h"

using matrix_traits::MatrixStatus;

// ============================================================================
// HELPER: Apply permutation to matrix rows (P·A)
// ============================================================================

template <typename T, my_size_t N>
FusedMatrix<T, N, N> apply_permutation(const FusedVector<my_size_t, N> &perm,
                                       const FusedMatrix<T, N, N> &A)
{
    FusedMatrix<T, N, N> PA(T(0));

    for (my_size_t i = 0; i < N; ++i)
    {
        for (my_size_t j = 0; j < N; ++j)
        {
            PA(i, j) = A(perm(i), j);
        }
    }

    return PA;
}

// ============================================================================
// 3×3 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("lu: 3x3 known answer",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 1, 1},
        {4, 3, 3},
        {8, 7, 9}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::lu(A);

    REQUIRE(result.has_value());

    auto &lu = result.value();

    SECTION("L is unit lower-triangular")
    {
        auto L = lu.L();
        REQUIRE(L.isLowerTriangular());

        // Verify unit diagonal
        for (my_size_t i = 0; i < 3; ++i)
        {
            REQUIRE(L(i, i) == T(1));
        }
    }

    SECTION("U is upper-triangular")
    {
        auto U = lu.U();
        REQUIRE(U.isUpperTriangular());
    }

    SECTION("P·A = L·U reconstruction")
    {
        auto L = lu.L();
        auto U = lu.U();
        auto LU_product = Matrix::matmul(L, U);
        auto PA = apply_permutation(lu.perm, A);

        REQUIRE(LU_product == PA);
    }
}

// ============================================================================
// 2×2 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("lu: 2x2 known answer",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T A_vals[2][2] = {
        {1, 2},
        {3, 4}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::lu(A);

    REQUIRE(result.has_value());

    auto &lu = result.value();
    auto L = lu.L();
    auto U = lu.U();
    auto LU_product = Matrix::matmul(L, U);
    auto PA = apply_permutation(lu.perm, A);

    REQUIRE(LU_product == PA);
}

// ============================================================================
// 4×4 RECONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("lu: 4x4 reconstruction property",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    T A_vals[4][4] = {
        {2, -1, 2, -1},
        {4, 5, 2, -17},
        {2, -1, 2, -30},
        {4, 5, 245, -10}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::lu(A);

    REQUIRE(result.has_value());

    auto &lu = result.value();
    auto L = lu.L();
    auto U = lu.U();
    auto LU_product = Matrix::matmul(L, U);
    auto PA = apply_permutation(lu.perm, A);

    REQUIRE(LU_product == PA);
}

// ============================================================================
// IDENTITY MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("lu: identity matrix",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I(0);
    I.setIdentity();

    auto result = matrix_algorithms::lu(I);

    REQUIRE(result.has_value());

    auto &lu = result.value();

    SECTION("L is identity")
    {
        auto L = lu.L();
        REQUIRE(L.isIdentity());
    }

    SECTION("U is identity")
    {
        auto U = lu.U();
        REQUIRE(U.isIdentity());
    }

    SECTION("sign is +1 (no swaps)")
    {
        REQUIRE(lu.sign == 1);
    }
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("lu: 1x1",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{7}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::lu(A);

    REQUIRE(result.has_value());

    auto &lu = result.value();

    // L is identity (1x1 with unit diagonal)
    REQUIRE(lu.L()(0, 0) == T(1));

    // U is just the value
    REQUIRE(lu.U()(0, 0) == T(7));

    // No permutation
    REQUIRE(lu.perm(0) == 0);
    REQUIRE(lu.sign == 1);
}

// ============================================================================
// DIAGONAL MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("lu: diagonal matrix",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {4, 0, 0},
        {0, 9, 0},
        {0, 0, 16}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::lu(A);

    REQUIRE(result.has_value());

    auto &lu = result.value();
    auto L = lu.L();
    auto U = lu.U();

    REQUIRE(L.isIdentity());
    REQUIRE(U == A);
}

// ============================================================================
// PERMUTATION SIGN
// ============================================================================

TEMPLATE_TEST_CASE("lu: permutation sign tracks swaps",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // This matrix requires a row swap (row 0 has small pivot)
    T A_vals[3][3] = {
        {0, 1, 2},
        {1, 0, 3},
        {2, 3, 0}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::lu(A);

    REQUIRE(result.has_value());

    auto &lu = result.value();

    // At least one swap happened
    REQUIRE(lu.sign == -1); // odd number of swaps

    // Still reconstructs
    auto L = lu.L();
    auto U = lu.U();
    auto LU_product = Matrix::matmul(L, U);
    auto PA = apply_permutation(lu.perm, A);

    REQUIRE(LU_product == PA);
}

// ============================================================================
// SINGULAR MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("lu: singular matrix returns Singular",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // Row 2 = Row 0 + Row 1 → singular
    // Use a looser tolerance so float rounding doesn't sneak past
    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {5, 7, 9}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::lu(A, T(1e-4));

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// ZERO MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("lu: zero matrix returns Singular",
                   "[lu]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    Matrix A(T(0));

    auto result = matrix_algorithms::lu(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// GENERIC PATH — 7×7
// ============================================================================

TEST_CASE("lu: 7x7 reconstruction property",
          "[lu]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

    // Build a well-conditioned matrix: diagonally dominant
    Matrix A(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        A(i, i) = T(10 + i);
        for (my_size_t j = 0; j < 7; ++j)
        {
            if (i != j)
            {
                A(i, j) = T(1) / T(i + j + 2);
            }
        }
    }

    auto result = matrix_algorithms::lu(A);

    REQUIRE(result.has_value());

    auto &lu = result.value();
    auto L = lu.L();
    auto U = lu.U();
    auto LU_product = Matrix::matmul(L, U);
    auto PA = apply_permutation(lu.perm, A);

    REQUIRE(LU_product == PA);
}

// ============================================================================
// 20×20 RECONSTRUCTION
// ============================================================================

TEST_CASE("lu: 20x20 reconstruction property",
          "[lu]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 20, 20>;

    // Diagonally dominant matrix
    Matrix A(T(0));
    for (my_size_t i = 0; i < 20; ++i)
    {
        A(i, i) = T(20 + i);
        for (my_size_t j = 0; j < 20; ++j)
        {
            if (i != j)
            {
                A(i, j) = T(1) / T(i + j + 2);
            }
        }
    }

    auto result = matrix_algorithms::lu(A);

    REQUIRE(result.has_value());

    auto &lu = result.value();
    auto L = lu.L();
    auto U = lu.U();
    auto LU_product = Matrix::matmul(L, U);
    auto PA = apply_permutation(lu.perm, A);

    REQUIRE(LU_product == PA);
}
