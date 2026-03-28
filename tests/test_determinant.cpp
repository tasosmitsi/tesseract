#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "algorithms/operations/determinant.h"

using Catch::Approx;

// ============================================================================
// 1×1 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("determinant: 1x1",
                   "[determinant]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{7}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::determinant(A) == T(7));
}

// ============================================================================
// 2×2 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("determinant: 2x2 known answer",
                   "[determinant]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    // det = ad - bc = 1*4 - 2*3 = -2
    T A_vals[2][2] = {
        {1, 2},
        {3, 4}};
    Matrix A(A_vals);

    T det = matrix_algorithms::determinant(A);

    REQUIRE(det == T(-2));
}

// ============================================================================
// 3×3 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("determinant: 3x3 known answer",
                   "[determinant]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // det = 1(4·0 - 5·6) - 2(0·0 - 5·0) + 3(0·6 - 4·0) = -30
    T A_vals[3][3] = {
        {1, 2, 3},
        {0, 4, 5},
        {0, 6, 0}};
    Matrix A(A_vals);

    T det = matrix_algorithms::determinant(A);

    REQUIRE(det == T(-30));
}

// ============================================================================
// 3×3 KNOWN ANSWER (det = 1)
// ============================================================================

TEMPLATE_TEST_CASE("determinant: 3x3 unit determinant",
                   "[determinant]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {1, 2, 3},
        {0, 1, 4},
        {5, 6, 0}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::determinant(A) == T(1));
}

// ============================================================================
// 4×4 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("determinant: 4x4 known answer",
                   "[determinant]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // SPD matrix from Cholesky tests: A = M * Mᵀ where
    // M = [2 0 0 0; 1 3 0 0; 0 1 2 0; 1 0 1 4]
    // det(A) = det(M)² = (2*3*2*4)² = 48² = 2304
    T A_vals[4][4] = {
        {4, 2, 0, 2},
        {2, 10, 3, 1},
        {0, 3, 5, 2},
        {2, 1, 2, 18}};
    Matrix A(A_vals);

    T det = matrix_algorithms::determinant(A);

    REQUIRE(det == T(2304));
}

// ============================================================================
// IDENTITY MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("determinant: identity is 1",
                   "[determinant]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I(T(0));
    I.setIdentity();

    REQUIRE(matrix_algorithms::determinant(I) == T(1));
}

// ============================================================================
// DIAGONAL MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("determinant: diagonal matrix is product of diagonal",
                   "[determinant]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 0, 0},
        {0, 3, 0},
        {0, 0, 5}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::determinant(A) == T(30));
}

// ============================================================================
// SINGULAR MATRIX RETURNS ZERO
// ============================================================================

TEMPLATE_TEST_CASE("determinant: singular matrix returns 0",
                   "[determinant]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // Duplicate rows
    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {1, 2, 3}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::determinant(A) == T(0));
}

// ============================================================================
// NEGATIVE DETERMINANT
// ============================================================================

TEMPLATE_TEST_CASE("determinant: negative determinant",
                   "[determinant]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    // Swapped rows of identity → det = -1
    T A_vals[2][2] = {
        {0, 1},
        {1, 0}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::determinant(A) == T(-1));
}

// ============================================================================
// PROPERTY: det(A·B) = det(A) · det(B)
// ============================================================================

TEST_CASE("determinant: det(A*B) = det(A) * det(B)",
          "[determinant]")
{
    using T = double;
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

    T det_A = matrix_algorithms::determinant(A);
    T det_B = matrix_algorithms::determinant(B);
    T det_AB = matrix_algorithms::determinant(AB);

    REQUIRE(det_AB == Approx(det_A * det_B));
}

// ============================================================================
// PROPERTY: det(Aᵀ) = det(A)
// ============================================================================

TEST_CASE("determinant: det(A^T) = det(A)",
          "[determinant]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, -1, 3},
        {4, 5, -2},
        {-1, 3, 7}};
    Matrix A(A_vals), At;

    At = A.transpose_view();

    REQUIRE(matrix_algorithms::determinant(A) == matrix_algorithms::determinant(At));
}

// ============================================================================
// PROPERTY: det(cA) = c^N · det(A)
// ============================================================================

TEST_CASE("determinant: det(cA) = c^N * det(A)",
          "[determinant]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 1, 0},
        {1, 3, 1},
        {0, 1, 2}};
    Matrix A(A_vals), cA;

    T c = 3.0;
    cA = A * c;

    T det_A = matrix_algorithms::determinant(A);
    T det_cA = matrix_algorithms::determinant(cA);

    // c^3 * det(A)
    REQUIRE(det_cA == c * c * c * det_A);
}

// ============================================================================
// GENERIC PATH — 5×5 KNOWN ANSWER (upper triangular)
// ============================================================================

TEST_CASE("determinant: 5x5 upper triangular known answer",
          "[determinant]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 5, 5>;

    // Upper triangular — det = product of diagonal = 1*2*3*4*5 = 120
    Matrix A(T(0));
    for (my_size_t i = 0; i < 5; ++i)
    {
        A(i, i) = T(i + 1);
        for (my_size_t j = i + 1; j < 5; ++j)
        {
            A(i, j) = T(1);
        }
    }

    REQUIRE(matrix_algorithms::determinant(A) == Approx(T(120)));
}

// ============================================================================
// GENERIC PATH — 5×5 NEGATIVE DETERMINANT (row swap)
// ============================================================================

TEST_CASE("determinant: 5x5 negative determinant via row swap",
          "[determinant]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 5, 5>;

    // Diagonal matrix with rows 0 and 1 swapped → det negates
    Matrix A(T(0));
    for (my_size_t i = 0; i < 5; ++i)
    {
        A(i, i) = T(i + 1);
    }
    // Swap row 0 and row 1
    for (my_size_t j = 0; j < 5; ++j)
    {
        T tmp = A(0, j);
        A(0, j) = A(1, j);
        A(1, j) = tmp;
    }

    // det = -(1*2*3*4*5) = -120
    REQUIRE(matrix_algorithms::determinant(A) == Approx(T(-120)));
}

// ============================================================================
// GENERIC PATH — 7×7 DIAGONAL
// ============================================================================

TEST_CASE("determinant: 7x7 diagonal",
          "[determinant]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

    // Diagonal matrix: det = product of diagonal
    Matrix A(T(0));
    T expected_det = T(1);

    for (my_size_t i = 0; i < 7; ++i)
    {
        A(i, i) = T(i + 2);
        expected_det *= T(i + 2);
    }

    REQUIRE(matrix_algorithms::determinant(A) == expected_det);
}

// ============================================================================
// GENERIC PATH — 7×7 IDENTITY
// ============================================================================

TEST_CASE("determinant: 7x7 identity",
          "[determinant]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

    Matrix I(T(0));
    I.setIdentity();

    REQUIRE(matrix_algorithms::determinant(I) == T(1));
}

// ============================================================================
// GENERIC PATH — 7×7 SINGULAR
// ============================================================================

TEST_CASE("determinant: 7x7 singular",
          "[determinant]")
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

    REQUIRE(matrix_algorithms::determinant(A) == T(0));
}

// ============================================================================
// GENERIC PATH — 7×7 det(Aᵀ) = det(A)
// ============================================================================

TEST_CASE("determinant: 7x7 det(A^T) = det(A)",
          "[determinant]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

    // Dense non-symmetric matrix
    Matrix A(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        for (my_size_t j = 0; j < 7; ++j)
        {
            A(i, j) = T((i + 1) * 7 + (j + 3)) / T(j + 1);
        }
    }

    Matrix At;
    At = A.transpose_view();

    REQUIRE(matrix_algorithms::determinant(A) ==
            Approx(matrix_algorithms::determinant(At)));
}

// ============================================================================
// GENERIC PATH — 7×7 det(A·B) = det(A) · det(B)
// ============================================================================

TEST_CASE("determinant: 7x7 det(A*B) = det(A)*det(B)",
          "[determinant]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

    Matrix A(T(0)), B(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        for (my_size_t j = 0; j < 7; ++j)
        {
            A(i, j) = T(1) / T(i + j + 1);       // Hilbert-like
            B(i, j) = (i == j) ? T(3) : T(-0.1); // diagonally dominant
        }
    }

    A.print();
    B.print();

    auto AB = Matrix::matmul(A, B);

    T det_A = matrix_algorithms::determinant(A);
    T det_B = matrix_algorithms::determinant(B);
    T det_AB = matrix_algorithms::determinant(AB);

    REQUIRE(det_AB == Approx(det_A * det_B));
}
