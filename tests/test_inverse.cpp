#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "algorithms/operations/inverse.h"
#include "algorithms/operations/determinant.h"

using Catch::Approx;
using matrix_traits::MatrixStatus;

// ============================================================================
// 1×1 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 1x1",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{4}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value()(0, 0) == Approx(T(0.25)));
}

// ============================================================================
// 2×2 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 2x2 known answer",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    // A = [4 7; 2 6], det = 10
    // A⁻¹ = (1/10) * [6 -7; -2 4] = [0.6 -0.7; -0.2 0.4]
    T A_vals[2][2] = {
        {4, 7},
        {2, 6}};
    Matrix A(A_vals);

    T Ainv_expected_vals[2][2] = {
        {T(0.6), T(-0.7)},
        {T(-0.2), T(0.4)}};
    Matrix Ainv_expected(Ainv_expected_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Ainv_expected);
}

// ============================================================================
// 3×3 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 3x3 known answer",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // A = [1 2 3; 0 1 4; 5 6 0]
    // det = 1, so A⁻¹ = adj(A)
    T A_vals[3][3] = {
        {1, 2, 3},
        {0, 1, 4},
        {5, 6, 0}};
    Matrix A(A_vals);

    T Ainv_expected_vals[3][3] = {
        {-24, 18, 5},
        {20, -15, -4},
        {-5, 4, 1}};
    Matrix Ainv_expected(Ainv_expected_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Ainv_expected);
}

// ============================================================================
// 4×4 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 4x4 known answer",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // Unit upper triangular, det = 1
    T A_vals[4][4] = {
        {1, 2, 3, 4},
        {0, 1, 5, 6},
        {0, 0, 1, 7},
        {0, 0, 0, 1}};
    Matrix A(A_vals);

    T Ainv_expected_vals[4][4] = {
        {1, -2, 7, -41},
        {0, 1, -5, 29},
        {0, 0, 1, -7},
        {0, 0, 0, 1}};
    Matrix Ainv_expected(Ainv_expected_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Ainv_expected);
}

// ============================================================================
// IDENTITY — 3×3
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 3x3 identity inverse is identity",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I;
    I.setIdentity();

    auto result = matrix_algorithms::inverse(I);

    REQUIRE(result.has_value());
    REQUIRE(result.value().isIdentity());
}

// ============================================================================
// IDENTITY — 4×4
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 4x4 identity inverse is identity",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    Matrix I;
    I.setIdentity();

    auto result = matrix_algorithms::inverse(I);

    REQUIRE(result.has_value());
    REQUIRE(result.value().isIdentity());
}

// ============================================================================
// DIAGONAL — 3×3
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 3x3 diagonal",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 0, 0},
        {0, 4, 0},
        {0, 0, 5}};
    Matrix A(A_vals);

    T Ainv_expected_vals[3][3] = {
        {T(0.5), 0, 0},
        {0, T(0.25), 0},
        {0, 0, T(0.2)}};
    Matrix Ainv_expected(Ainv_expected_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Ainv_expected);
}

// ============================================================================
// DIAGONAL — 4×4
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 4x4 diagonal",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    T A_vals[4][4] = {
        {2, 0, 0, 0},
        {0, 4, 0, 0},
        {0, 0, 5, 0},
        {0, 0, 0, 8}};
    Matrix A(A_vals);

    T Ainv_expected_vals[4][4] = {
        {T(0.5), 0, 0, 0},
        {0, T(0.25), 0, 0},
        {0, 0, T(0.2), 0},
        {0, 0, 0, T(0.125)}};
    Matrix Ainv_expected(Ainv_expected_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Ainv_expected);
}

// ============================================================================
// SINGULAR — 2×2 ZERO MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("inverse: zero matrix returns Singular",
                   "[inverse][error]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    Matrix A(T(0));

    auto result = matrix_algorithms::inverse(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// SINGULAR — 3×3
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 3x3 singular returns Singular",
                   "[inverse][error]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // Duplicate rows
    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {1, 2, 3}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// SINGULAR — 4×4
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 4x4 singular returns Singular",
                   "[inverse][error]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // Row 3 = Row 0
    T A_vals[4][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {1, 2, 3, 4}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// PROPERTY: A · A⁻¹ = I — 3×3
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 3x3 A * A_inv = I",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, -1, 3},
        {4, 5, -2},
        {-1, 3, 7}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());

    auto product = Matrix::matmul(A, result.value());

    REQUIRE(product.isIdentity());
}

// ============================================================================
// PROPERTY: A · A⁻¹ = I — 4×4 SPD
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 4x4 A * A_inv = I",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    T A_vals[4][4] = {
        {4, 2, 0, 2},
        {2, 10, 3, 1},
        {0, 3, 5, 2},
        {2, 1, 2, 18}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());

    auto product = Matrix::matmul(A, result.value());

    REQUIRE(product.isIdentity());
}

// ============================================================================
// PROPERTY: A⁻¹ · A = I — 4×4 non-symmetric
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 4x4 A_inv * A = I",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    T A_vals[4][4] = {
        {2, -1, 0, 3},
        {1, 4, -2, 1},
        {0, 3, 5, -1},
        {-1, 2, 1, 6}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());

    auto product = Matrix::matmul(result.value(), A);

    REQUIRE(product.isIdentity());
}

// ============================================================================
// PROPERTY: (A⁻¹)⁻¹ = A — 3×3
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 3x3 (A_inv)_inv = A",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, -1, 3},
        {4, 5, -2},
        {-1, 3, 7}};
    Matrix A(A_vals);

    auto result1 = matrix_algorithms::inverse(A);
    REQUIRE(result1.has_value());

    auto result2 = matrix_algorithms::inverse(result1.value());
    REQUIRE(result2.has_value());

    REQUIRE(result2.value() == A);
}

// ============================================================================
// PROPERTY: (A⁻¹)⁻¹ = A — 4×4
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 4x4 (A_inv)_inv = A",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    T A_vals[4][4] = {
        {2, -1, 0, 3},
        {1, 4, -2, 1},
        {0, 3, 5, -1},
        {-1, 2, 1, 6}};
    Matrix A(A_vals);

    auto result1 = matrix_algorithms::inverse(A);
    REQUIRE(result1.has_value());

    auto result2 = matrix_algorithms::inverse(result1.value());
    REQUIRE(result2.has_value());

    REQUIRE(result2.value() == A);
}

// ============================================================================
// PROPERTY: (A·B)⁻¹ = B⁻¹ · A⁻¹ — 3×3
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 3x3 (AB)_inv = B_inv * A_inv",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 1, 0},
        {1, 3, 1},
        {0, 1, 2}};
    Matrix A(A_vals);

    T B_vals[3][3] = {
        {1, 0, 2},
        {0, 2, 1},
        {3, 1, 0}};
    Matrix B(B_vals);

    auto AB = Matrix::matmul(A, B);

    auto inv_AB = matrix_algorithms::inverse(AB);
    auto inv_A = matrix_algorithms::inverse(A);
    auto inv_B = matrix_algorithms::inverse(B);

    REQUIRE(inv_AB.has_value());
    REQUIRE(inv_A.has_value());
    REQUIRE(inv_B.has_value());

    auto BinvAinv = Matrix::matmul(inv_B.value(), inv_A.value());

    REQUIRE(inv_AB.value() == BinvAinv);
}

// ============================================================================
// PROPERTY: det(A⁻¹) = 1/det(A) — 3×3
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 3x3 det(A_inv) = 1/det(A)",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 1, 0},
        {1, 3, 1},
        {0, 1, 2}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::inverse(A);
    REQUIRE(result.has_value());

    T det_A = matrix_algorithms::determinant(A);
    T det_Ainv = matrix_algorithms::determinant(result.value());

    REQUIRE(det_Ainv == Approx(T(1) / det_A));
}

// ============================================================================
// PROPERTY: det(A⁻¹) = 1/det(A) — 4×4
// ============================================================================

TEST_CASE("inverse: 4x4 det(A_inv) = 1/det(A)",
          "[inverse]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 4, 4>;

    T A_vals[4][4] = {
        {2, -1, 0, 3},
        {1, 4, -2, 1},
        {0, 3, 5, -1},
        {-1, 2, 1, 6}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::inverse(A);
    REQUIRE(result.has_value());

    T det_A = matrix_algorithms::determinant(A);
    T det_Ainv = matrix_algorithms::determinant(result.value());

    REQUIRE(det_Ainv == Approx(T(1) / det_A));
}

// ============================================================================
// GENERIC PATH — 7×7 A · A⁻¹ = I
// ============================================================================

TEMPLATE_TEST_CASE("inverse: 7x7 A * A_inv = I",
                   "[inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 7, 7>;

    // Diagonally dominant → well-conditioned
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

    auto result = matrix_algorithms::inverse(A);

    REQUIRE(result.has_value());

    auto product = Matrix::matmul(A, result.value());

    REQUIRE(product.isIdentity());
}

// ============================================================================
// GENERIC PATH — 7×7 SINGULAR
// ============================================================================

TEST_CASE("inverse: 7x7 singular",
          "[inverse]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

    // Diagonal with row 6 copied from row 0 → linearly dependent
    Matrix A(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        A(i, i) = T(i + 1);
    }
    for (my_size_t j = 0; j < 7; ++j)
    {
        A(6, j) = A(0, j);
    }

    auto result = matrix_algorithms::inverse(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// GENERIC PATH — 7×7 (A⁻¹)⁻¹ = A
// ============================================================================

TEST_CASE("inverse: 7x7 (A_inv)_inv = A",
          "[inverse]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

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

    auto result1 = matrix_algorithms::inverse(A);
    REQUIRE(result1.has_value());

    auto result2 = matrix_algorithms::inverse(result1.value());
    REQUIRE(result2.has_value());

    REQUIRE(result2.value() == A);
}

// ============================================================================
// GENERIC PATH — 7×7 det(A⁻¹) = 1/det(A)
// ============================================================================

TEST_CASE("inverse: 7x7 det(A_inv) = 1/det(A)",
          "[inverse]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

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

    auto result = matrix_algorithms::inverse(A);
    REQUIRE(result.has_value());

    T det_A = matrix_algorithms::determinant(A);
    T det_Ainv = matrix_algorithms::determinant(result.value());

    REQUIRE(det_Ainv == Approx(T(1) / det_A));
}
