#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/solvers/tridiagonal.h"
#include "algorithms/solvers/linear_solve.h"

using matrix_traits::MatrixStatus;

// ============================================================================
// HELPER: Build tridiagonal matrix and verify Ax = b
// ============================================================================

template <typename T, my_size_t N>
FusedVector<T, N> tridiag_matvec(
    const FusedVector<T, N> &a,
    const FusedVector<T, N> &d,
    const FusedVector<T, N> &c,
    const FusedVector<T, N> &x)
{
    FusedVector<T, N> result(T(0));

    result(0) = d(0) * x(0);
    if (N > 1)
        result(0) += c(0) * x(1);

    for (my_size_t i = 1; i + 1 < N; ++i)
    {
        result(i) = a(i) * x(i - 1) + d(i) * x(i) + c(i) * x(i + 1);
    }

    if (N > 1)
        result(N - 1) = a(N - 1) * x(N - 2) + d(N - 1) * x(N - 1);

    return result;
}

// ============================================================================
// 3×3 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("thomas_solve: 3x3 known answer",
                   "[thomas_solve]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    // [ 2  1  0 ] [x0]   [ 5 ]
    // [ 1  3  1 ] [x1] = [10 ]
    // [ 0  1  2 ] [x2]   [ 7 ]
    T a_vals[3][1] = {{0}, {1}, {1}};
    T d_vals[3][1] = {{2}, {3}, {2}};
    T c_vals[3][1] = {{1}, {1}, {0}};
    T b_vals[3][1] = {{5}, {10}, {7}};

    Vector a(a_vals), d(d_vals), c(c_vals), b(b_vals);

    auto result = matrix_algorithms::thomas_solve(a, d, c, b);

    REQUIRE(result.has_value());

    // Verify Ax = b
    auto Ax = tridiag_matvec(a, d, c, result.value());
    REQUIRE(Ax == b);
}

// ============================================================================
// 4×4 KNOWN ANSWER — DIFFUSION-LIKE
// ============================================================================

TEMPLATE_TEST_CASE("thomas_solve: 4x4 diffusion-like",
                   "[thomas_solve]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 4>;

    // Discretized -u'' = f with Dirichlet BCs
    // [-2  1  0  0] [x0]   [b0]
    // [ 1 -2  1  0] [x1] = [b1]
    // [ 0  1 -2  1] [x2]   [b2]
    // [ 0  0  1 -2] [x3]   [b3]
    T a_vals[4][1] = {{0}, {1}, {1}, {1}};
    T d_vals[4][1] = {{-2}, {-2}, {-2}, {-2}};
    T c_vals[4][1] = {{1}, {1}, {1}, {0}};
    T b_vals[4][1] = {{-1}, {0}, {0}, {-1}};

    Vector a(a_vals), d(d_vals), c(c_vals), b(b_vals);

    auto result = matrix_algorithms::thomas_solve(a, d, c, b);

    REQUIRE(result.has_value());

    auto Ax = tridiag_matvec(a, d, c, result.value());
    REQUIRE(Ax == b);
}

// ============================================================================
// DIAGONAL SYSTEM (a = c = 0)
// ============================================================================

TEMPLATE_TEST_CASE("thomas_solve: diagonal system",
                   "[thomas_solve]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    // Pure diagonal: x(i) = b(i) / d(i)
    T a_vals[3][1] = {{0}, {0}, {0}};
    T d_vals[3][1] = {{2}, {4}, {5}};
    T c_vals[3][1] = {{0}, {0}, {0}};
    T b_vals[3][1] = {{6}, {12}, {15}};

    Vector a(a_vals), d(d_vals), c(c_vals), b(b_vals);

    T x_expected_vals[3][1] = {{3}, {3}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::thomas_solve(a, d, c, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("thomas_solve: 1x1",
                   "[thomas_solve]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 1>;

    T a_vals[1][1] = {{0}};
    T d_vals[1][1] = {{5}};
    T c_vals[1][1] = {{0}};
    T b_vals[1][1] = {{15}};

    Vector a(a_vals), d(d_vals), c(c_vals), b(b_vals);

    T x_expected_vals[1][1] = {{3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::thomas_solve(a, d, c, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// 2×2
// ============================================================================

TEMPLATE_TEST_CASE("thomas_solve: 2x2",
                   "[thomas_solve]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 2>;

    // [4 1] [x0] = [9]
    // [2 3] [x1]   [11]
    T a_vals[2][1] = {{0}, {2}};
    T d_vals[2][1] = {{4}, {3}};
    T c_vals[2][1] = {{1}, {0}};
    T b_vals[2][1] = {{9}, {11}};

    Vector a(a_vals), d(d_vals), c(c_vals), b(b_vals);

    auto result = matrix_algorithms::thomas_solve(a, d, c, b);

    REQUIRE(result.has_value());

    auto Ax = tridiag_matvec(a, d, c, result.value());
    REQUIRE(Ax == b);
}

// ============================================================================
// SINGULAR — ZERO DIAGONAL
// ============================================================================

TEMPLATE_TEST_CASE("thomas_solve: zero diagonal returns Singular",
                   "[thomas_solve]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{0}, {1}, {1}};
    T d_vals[3][1] = {{0}, {3}, {2}}; // d(0) = 0 → singular
    T c_vals[3][1] = {{1}, {1}, {0}};
    T b_vals[3][1] = {{1}, {1}, {1}};

    Vector a(a_vals), d(d_vals), c(c_vals), b(b_vals);

    auto result = matrix_algorithms::thomas_solve(a, d, c, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// SINGULAR — ZERO PIVOT DURING SWEEP
// ============================================================================

TEMPLATE_TEST_CASE("thomas_solve: zero pivot during sweep returns Singular",
                   "[thomas_solve]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    // d'(1) = d(1) - a(1)/d(0) * c(0) = 1 - 1/1 * 1 = 0
    T a_vals[3][1] = {{0}, {1}, {0}};
    T d_vals[3][1] = {{1}, {1}, {1}};
    T c_vals[3][1] = {{1}, {0}, {0}};
    T b_vals[3][1] = {{1}, {1}, {1}};

    Vector a(a_vals), d(d_vals), c(c_vals), b(b_vals);

    auto result = matrix_algorithms::thomas_solve(a, d, c, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// PROPERTY: AGREES WITH LU SOLVE
// ============================================================================

TEST_CASE("thomas_solve: agrees with LU solve",
          "[thomas_solve]")
{
    using T = double;
    using Vector = FusedVector<T, 4>;
    using Matrix = FusedMatrix<T, 4, 4>;

    T a_vals[4][1] = {{0}, {1}, {2}, {1}};
    T d_vals[4][1] = {{4}, {5}, {6}, {4}};
    T c_vals[4][1] = {{1}, {2}, {1}, {0}};
    T b_vals[4][1] = {{7}, {-3}, {10}, {5}};

    Vector a(a_vals), d(d_vals), c(c_vals), b(b_vals);

    // Build full tridiagonal matrix for LU solve
    Matrix A(T(0));
    for (my_size_t i = 0; i < 4; ++i)
    {
        A(i, i) = d(i);
        if (i > 0)
            A(i, i - 1) = a(i);
        if (i + 1 < 4)
            A(i, i + 1) = c(i);
    }

    // Thomas solve
    auto thomas_result = matrix_algorithms::thomas_solve(a, d, c, b);
    REQUIRE(thomas_result.has_value());

    // LU solve
    auto lu_result = matrix_algorithms::lu_solve(A, b);
    REQUIRE(lu_result.has_value());

    REQUIRE(thomas_result.value() == lu_result.value());
}

// ============================================================================
// GENERIC PATH — 10 ELEMENTS
// ============================================================================

TEST_CASE("thomas_solve: 10-element reconstruction",
          "[thomas_solve]")
{
    using T = double;
    using Vector = FusedVector<T, 10>;

    // Diagonally dominant tridiagonal
    Vector a(T(0)), d(T(0)), c(T(0));

    for (my_size_t i = 0; i < 10; ++i)
    {
        d(i) = T(4);
        if (i > 0)
            a(i) = T(-1);
        if (i + 1 < 10)
            c(i) = T(-1);
    }

    // Pick x, compute b = Ax
    Vector x_true(T(0));
    for (my_size_t i = 0; i < 10; ++i)
    {
        x_true(i) = T(i + 1);
    }

    auto b = tridiag_matvec(a, d, c, x_true);

    auto result = matrix_algorithms::thomas_solve(a, d, c, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_true);
}
