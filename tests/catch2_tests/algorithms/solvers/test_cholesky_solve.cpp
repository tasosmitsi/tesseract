#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/solvers/cholesky_solve.h"

using matrix_traits::MatrixStatus;

// ============================================================================
// CHOLESKY SOLVE — 3×3 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_solve: 3x3 known answer",
                   "[cholesky_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // A*x = b → x = [1, 2, 3]
    // b = A*x: row0: 4+24-48=-20, row1: 12+74-129=-43, row2: -16-86+294=192

    T A_vals[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};
    Matrix A(A_vals);

    T b_vals[3][1] = {{-20}, {-43}, {192}};
    Vector b(b_vals);

    T x_expected_vals[3][1] = {{1}, {2}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::cholesky_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// CHOLESKY SOLVE — IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_solve: identity gives x = b",
                   "[cholesky_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    Matrix I(0);
    I.setIdentity();

    T b_vals[2][1] = {{7}, {-3}};
    Vector b(b_vals);

    auto result = matrix_algorithms::cholesky_solve(I, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == b);
}

// ============================================================================
// CHOLESKY SOLVE — 4×4 RECONSTRUCTION PROPERTY
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_solve: 4x4 reconstruction property",
                   "[cholesky_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vector = FusedVector<T, 4>;

    T A_vals[4][4] = {
        {4, 2, 0, 2},
        {2, 10, 3, 1},
        {0, 3, 5, 2},
        {2, 1, 2, 18}};
    Matrix A(A_vals);

    // x = [1, -1, 2, 0]
    // b = A*x: row0: 4-2+0+0=2, row1: 2-10+6+0=-2, row2: 0-3+10+0=7, row3: 2-1+4+0=5
    T b_vals[4][1] = {{2}, {-2}, {7}, {5}};
    Vector b(b_vals);

    T x_expected_vals[4][1] = {{1}, {-1}, {2}, {0}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::cholesky_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// CHOLESKY SOLVE — 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_solve: 1x1",
                   "[cholesky_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;
    using Vector = FusedVector<T, 1>;

    T A_vals[1][1] = {{9}};
    Matrix A(A_vals);

    T b_vals[1][1] = {{27}};
    Vector b(b_vals);

    T x_expected_vals[1][1] = {{3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::cholesky_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// CHOLESKY SOLVE — ERROR: NOT SYMMETRIC
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_solve: non-symmetric returns NotSymmetric",
                   "[cholesky_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    T A_vals[2][2] = {
        {1, 2},
        {3, 4}};
    Matrix A(A_vals);

    T b_vals[2][1] = {{1}, {1}};
    Vector b(b_vals);

    auto result = matrix_algorithms::cholesky_solve(A, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::NotSymmetric);
}

// ============================================================================
// CHOLESKY SOLVE — ERROR: NOT POSITIVE DEFINITE
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_solve: non-positive-definite returns NotPositiveDefinite",
                   "[cholesky_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    T A_vals[2][2] = {
        {-4, 0},
        {0, -4}};
    Matrix A(A_vals);

    T b_vals[2][1] = {{1}, {1}};
    Vector b(b_vals);

    auto result = matrix_algorithms::cholesky_solve(A, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::NotPositiveDefinite);
}

// ============================================================================
// GENERIC PATH — 7×7 (not unrolled in forward sub or transposed back sub)
// ============================================================================

TEST_CASE("cholesky_solve: 7x7 generic path",
          "[cholesky_solve]")
{
    using T = double;
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

    // Pick x_true, compute b = A * x_true
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

    auto result = matrix_algorithms::cholesky_solve(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_true);
}
