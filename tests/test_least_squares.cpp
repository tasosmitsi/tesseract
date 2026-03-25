#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/solvers/least_squares.h"

using Catch::Approx;
using matrix_traits::MatrixStatus;

// ============================================================================
// EXACT SYSTEM (M=N) — RECOVERS EXACT SOLUTION
// ============================================================================

TEMPLATE_TEST_CASE("least_squares: square system recovers exact solution",
                   "[least_squares]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector3 = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {1, 2, 3},
        {0, 1, 4},
        {5, 6, 0}};
    Matrix A(A_vals);

    // x = [1, 2, 3] → b = A*x: row0: 1+4+9=14, row1: 0+2+12=14, row2: 5+12=17
    T b_vals[3][1] = {{14}, {14}, {17}};
    Vector3 b(b_vals);

    T x_expected_vals[3][1] = {{1}, {2}, {3}};
    Vector3 x_expected(x_expected_vals);

    auto result = matrix_algorithms::least_squares(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// OVERDETERMINED — LINEAR FIT (y = a + b*x)
// ============================================================================

TEST_CASE("least_squares: linear regression",
          "[least_squares]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 4, 2>;
    using Vector4 = FusedVector<T, 4>;
    using Vector2 = FusedVector<T, 2>;

    // Data points on the line y = 1 + 2x:
    //   (0,1), (1,3), (2,5), (3,7)
    // A = [1 x; ...], b = [y; ...]
    T A_vals[4][2] = {
        {1, 0},
        {1, 1},
        {1, 2},
        {1, 3}};
    Matrix A(A_vals);

    T b_vals[4][1] = {{1}, {3}, {5}, {7}};
    Vector4 b(b_vals);

    // Exact fit: x = [1, 2] (intercept=1, slope=2)
    T x_expected_vals[2][1] = {{1}, {2}};
    Vector2 x_expected(x_expected_vals);

    auto result = matrix_algorithms::least_squares(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// OVERDETERMINED — NOISY DATA (residual > 0)
// ============================================================================

TEST_CASE("least_squares: noisy linear regression",
          "[least_squares]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 5, 2>;
    using Vector5 = FusedVector<T, 5>;

    // Data near y = 1 + 2x with noise:
    //   (0, 1.1), (1, 2.9), (2, 5.2), (3, 6.8), (4, 9.1)
    T A_vals[5][2] = {
        {1, 0},
        {1, 1},
        {1, 2},
        {1, 3},
        {1, 4}};
    Matrix A(A_vals);

    T b_vals[5][1] = {{T(1.1)}, {T(2.9)}, {T(5.2)}, {T(6.8)}, {T(9.1)}};
    Vector5 b(b_vals);

    auto result = matrix_algorithms::least_squares(A, b);

    REQUIRE(result.has_value());

    auto &x = result.value();

    // Intercept should be near 1, slope near 2
    REQUIRE(x(0) == Approx(T(1)).margin(T(0.5)));
    REQUIRE(x(1) == Approx(T(2)).margin(T(0.2)));
}

// ============================================================================
// OVERDETERMINED — QUADRATIC FIT
// ============================================================================

TEST_CASE("least_squares: quadratic fit",
          "[least_squares]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 5, 3>;
    using Vector5 = FusedVector<T, 5>;
    using Vector3 = FusedVector<T, 3>;

    // Data on y = 1 + 2x + 3x² at x = -1, 0, 1, 2, 3
    // A = [1 x x²; ...], b = [y; ...]
    T A_vals[5][3] = {
        {1, -1, 1},
        {1, 0, 0},
        {1, 1, 1},
        {1, 2, 4},
        {1, 3, 9}};
    Matrix A(A_vals);

    // y = 1 + 2x + 3x²:  x=-1→2, x=0→1, x=1→6, x=2→17, x=3→34
    T b_vals[5][1] = {{2}, {1}, {6}, {17}, {34}};
    Vector5 b(b_vals);

    T x_expected_vals[3][1] = {{1}, {2}, {3}};
    Vector3 x_expected(x_expected_vals);

    auto result = matrix_algorithms::least_squares(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// PROPERTY: Ax ≈ b FOR EXACT SYSTEM
// ============================================================================

TEMPLATE_TEST_CASE("least_squares: Ax = b for exact overdetermined",
                   "[least_squares]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 2>;
    using Vector4 = FusedVector<T, 4>;

    // Exact data on y = 1 + 2x
    T A_vals[4][2] = {
        {1, 0},
        {1, 1},
        {1, 2},
        {1, 3}};
    Matrix A(A_vals);

    T b_vals[4][1] = {{1}, {3}, {5}, {7}};
    Vector4 b(b_vals);

    auto result = matrix_algorithms::least_squares(A, b);
    REQUIRE(result.has_value());

    // Verify A*x = b (residual should be zero for exact data)
    auto &x = result.value();
    auto Ax = FusedMatrix<T, 4, 1>::matmul(A, x);
    REQUIRE(Ax == b);
}

// ============================================================================
// PROPERTY: NORMAL EQUATIONS — AᵀAx = Aᵀb
// ============================================================================

TEST_CASE("least_squares: satisfies normal equations",
          "[least_squares]")
{
    using T = double;
    using Matrix53 = FusedMatrix<T, 5, 3>;
    using Matrix33 = FusedMatrix<T, 3, 3>;
    using Vector5 = FusedVector<T, 5>;

    T A_vals[5][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 0},
        {-1, 3, 2},
        {2, -1, 4}};
    Matrix53 A(A_vals);

    T b_vals[5][1] = {{1}, {2}, {3}, {4}, {5}};
    Vector5 b(b_vals);

    auto result = matrix_algorithms::least_squares(A, b);
    REQUIRE(result.has_value());
    auto &x = result.value();

    // AᵀA
    auto AtA = Matrix33::matmul(A.transpose_view(), A);

    // Aᵀb (3×5 * 5×1 → 3×1)
    auto Atb = FusedMatrix<T, 3, 1>::matmul(A.transpose_view(), b);

    // AᵀAx (3×3 * 3×1 → 3×1)
    auto AtAx = FusedMatrix<T, 3, 1>::matmul(AtA, x);

    REQUIRE(AtAx == Atb);
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("least_squares: 1x1",
                   "[least_squares]", double, float)
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

    auto result = matrix_algorithms::least_squares(A, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// RANK-DEFICIENT — SINGULAR
// ============================================================================

TEST_CASE("least_squares: rank-deficient returns Singular",
          "[least_squares]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 4, 3>;
    using Vector = FusedVector<T, 4>;

    // Col 2 = Col 0 + Col 1 → rank 2, N=3
    T A_vals[4][3] = {
        {1, 2, 3},
        {4, 5, 9},
        {7, 8, 15},
        {0, 1, 1}};
    Matrix A(A_vals);

    T b_vals[4][1] = {{1}, {2}, {3}, {4}};
    Vector b(b_vals);

    auto result = matrix_algorithms::least_squares(A, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// GENERIC PATH — 7×3
// ============================================================================

TEST_CASE("least_squares: 7x3 satisfies normal equations",
          "[least_squares]")
{
    using T = double;
    using Matrix73 = FusedMatrix<T, 7, 3>;
    using Matrix33 = FusedMatrix<T, 3, 3>;
    using Vector7 = FusedVector<T, 7>;

    // Well-conditioned rectangular — use varied structure to ensure full rank
    Matrix73 A(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        A(i, 0) = T(1);                 // constant term
        A(i, 1) = T(i + 1);             // linear term
        A(i, 2) = T((i + 1) * (i + 1)); // quadratic term
    }

    Vector7 b(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        b(i) = T(i * i + 1);
    }

    auto result = matrix_algorithms::least_squares(A, b);
    REQUIRE(result.has_value());
    auto &x = result.value();

    // Verify normal equations: AᵀAx = Aᵀb
    auto AtA = Matrix33::matmul(A.transpose_view(), A);

    // Aᵀb (3×7 * 7×1 → 3×1)
    auto Atb = FusedMatrix<T, 3, 1>::matmul(A.transpose_view(), b);

    // AᵀAx (3×3 * 3×1 → 3×1)
    auto AtAx = FusedMatrix<T, 3, 1>::matmul(AtA, x);

    REQUIRE(AtAx == Atb);
}
