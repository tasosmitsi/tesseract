#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/solvers/linear_solve.h"

using matrix_traits::MatrixStatus;

// ============================================================================
// LU SOLVE — 3×3 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("lu_solve: 3x3 known answer",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // Non-symmetric matrix — only LU can handle this
    T A_vals[3][3] = {
        {1, 2, 3},
        {0, 1, 4},
        {5, 6, 0}};
    Matrix A(A_vals);

    // x = [1, 2, 3] → b = A*x: row0: 1+4+9=14, row1: 0+2+12=14, row2: 5+12+0=17
    T b_vals[3][1] = {{14}, {14}, {17}};
    Vector b(b_vals);

    T x_expected_vals[3][1] = {{1}, {2}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::lu_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// LU SOLVE — 2×2 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("lu_solve: 2x2 known answer",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    T A_vals[2][2] = {
        {4, 7},
        {2, 6}};
    Matrix A(A_vals);

    // x = [3, -1] → b = A*x: row0: 12-7=5, row1: 6-6=0
    T b_vals[2][1] = {{5}, {0}};
    Vector b(b_vals);

    T x_expected_vals[2][1] = {{3}, {-1}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::lu_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// LU SOLVE — IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("lu_solve: identity gives x = b",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    Matrix I(0);
    I.setIdentity();

    T b_vals[3][1] = {{7}, {-3}, {5}};
    Vector b(b_vals);

    auto result = matrix_algorithms::lu_solve(I, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == b);
}

// ============================================================================
// LU SOLVE — 1×1
// ============================================================================

TEMPLATE_TEST_CASE("lu_solve: 1x1",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;
    using Vector = FusedVector<T, 1>;

    T A_vals[1][1] = {{4}};
    Matrix A(A_vals);

    T b_vals[1][1] = {{8}};
    Vector b(b_vals);

    T x_expected_vals[1][1] = {{2}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::lu_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// LU SOLVE — SINGULAR ERROR
// ============================================================================

TEMPLATE_TEST_CASE("lu_solve: singular matrix returns Singular",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {1, 2, 3}};
    Matrix A(A_vals);

    T b_vals[3][1] = {{1}, {1}, {1}};
    Vector b(b_vals);

    auto result = matrix_algorithms::lu_solve(A, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// LU SOLVE — 4×4 RECONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("lu_solve: 4x4 non-symmetric reconstruction",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vector = FusedVector<T, 4>;

    T A_vals[4][4] = {
        {2, -1, 2, -1},
        {4, 5, 2, -17},
        {2, -1, 2, -30},
        {4, 5, 245, -10}};
    Matrix A(A_vals);

    // x = [1, -1, 2, 0]
    // b = A*x: row0: 2+1+4+0=7, row1: 4-5+4+0=3, row2: 2+1+4+0=7, row3: 4-5+490+0=489
    T b_vals[4][1] = {{7}, {3}, {7}, {489}};
    Vector b(b_vals);

    T x_expected_vals[4][1] = {{1}, {-1}, {2}, {0}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::lu_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// SOLVE DISPATCHER — SPD MATRIX (should take Cholesky path)
// ============================================================================

TEMPLATE_TEST_CASE("solve: SPD matrix",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // Same SPD matrix as cholesky_solve tests
    T A_vals[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};
    Matrix A(A_vals);

    T b_vals[3][1] = {{-20}, {-43}, {192}};
    Vector b(b_vals);

    T x_expected_vals[3][1] = {{1}, {2}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// SOLVE DISPATCHER — NON-SYMMETRIC MATRIX (should fall back to LU)
// ============================================================================

TEMPLATE_TEST_CASE("solve: non-symmetric matrix falls back to LU",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {1, 2, 3},
        {0, 1, 4},
        {5, 6, 0}};
    Matrix A(A_vals);

    T b_vals[3][1] = {{14}, {14}, {17}};
    Vector b(b_vals);

    T x_expected_vals[3][1] = {{1}, {2}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// SOLVE DISPATCHER — SYMMETRIC BUT NOT POSITIVE DEFINITE (falls back to LU)
// ============================================================================

TEMPLATE_TEST_CASE("solve: symmetric non-positive-definite falls back to LU",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    // Symmetric but indefinite: eigenvalues -1 and 5
    T A_vals[2][2] = {
        {2, 3},
        {3, 2}};
    Matrix A(A_vals);

    // x = [1, -1] → b = A*x: row0: 2-3=-1, row1: 3-2=1
    T b_vals[2][1] = {{-1}, {1}};
    Vector b(b_vals);

    T x_expected_vals[2][1] = {{1}, {-1}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// SOLVE DISPATCHER — SINGULAR
// ============================================================================

TEMPLATE_TEST_CASE("solve: singular matrix returns Singular",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    T A_vals[2][2] = {
        {1, 2},
        {2, 4}};
    Matrix A(A_vals);

    T b_vals[2][1] = {{1}, {1}};
    Vector b(b_vals);

    auto result = matrix_algorithms::solve(A, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// SOLVE DISPATCHER — IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("solve: identity gives x = b",
                   "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    Matrix I(0);
    I.setIdentity();

    T b_vals[3][1] = {{7}, {-3}, {5}};
    Vector b(b_vals);

    auto result = matrix_algorithms::solve(I, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == b);
}

// ============================================================================
// GENERIC PATH — 7×7 LU SOLVE
// ============================================================================

TEMPLATE_TEST_CASE("lu_solve: 7x7 reconstruction",
          "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 7, 7>;
    using Vector = FusedVector<T, 7>;

    // Non-symmetric diagonally dominant
    Matrix A(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        A(i, i) = T(10 + i);
        for (my_size_t j = 0; j < 7; ++j)
        {
            if (i != j)
            {
                A(i, j) = T(1) / T(i + j + 2);
                if (i > j)
                    A(i, j) *= T(-1); // make non-symmetric
            }
        }
    }

    // Pick x_true, compute b = A * x_true
    Vector x_true(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        x_true(i) = T(i + 1);
    }

    Vector b(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        T sum = 0;
        for (my_size_t k = 0; k < 7; ++k)
        {
            sum += A(i, k) * x_true(k);
        }
        b(i) = sum;
    }

    auto result = matrix_algorithms::lu_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_true);
}

// ============================================================================
// GENERIC PATH — 7×7 SOLVE DISPATCHER
// ============================================================================

TEMPLATE_TEST_CASE("solve: 7x7 SPD via dispatcher",
          "[linear_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 7, 7>;
    using Vector = FusedVector<T, 7>;

    // Build SPD: A = M * Mᵀ
    Matrix M(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        M(i, i) = T(i + 2);
        for (my_size_t j = 0; j < i; ++j)
        {
            M(i, j) = T(1) / T(i - j + 1);
        }
    }
    Matrix A = Matrix::matmul(M, M.transpose_view());

    Vector x_true(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        x_true(i) = T(i + 1);
    }

    // b = A * x_true
    Vector b(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        T sum = 0;
        for (my_size_t k = 0; k < 7; ++k)
        {
            sum += A(i, k) * x_true(k);
        }
        b(i) = sum;
    }

    auto result = matrix_algorithms::solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_true);
}
