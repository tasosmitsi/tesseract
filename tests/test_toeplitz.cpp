#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/solvers/toeplitz.h"
#include "algorithms/solvers/linear_solve.h"

using matrix_traits::MatrixStatus;

// ============================================================================
// HELPER: Build symmetric Toeplitz matrix from first row
// ============================================================================

template <typename T, my_size_t N>
FusedMatrix<T, N, N> build_toeplitz(const FusedVector<T, N> &r)
{
    FusedMatrix<T, N, N> A(T(0));

    for (my_size_t i = 0; i < N; ++i)
    {
        for (my_size_t j = 0; j < N; ++j)
        {
            my_size_t diff = (i > j) ? (i - j) : (j - i);
            A(i, j) = r(diff);
        }
    }

    return A;
}

// ============================================================================
// HELPER: Toeplitz matvec
// ============================================================================

template <typename T, my_size_t N>
FusedVector<T, N> toeplitz_matvec(const FusedVector<T, N> &r, const FusedVector<T, N> &x)
{
    FusedVector<T, N> result(T(0));

    for (my_size_t i = 0; i < N; ++i)
    {
        T sum = T(0);
        for (my_size_t j = 0; j < N; ++j)
        {
            my_size_t diff = (i > j) ? (i - j) : (j - i);
            sum += r(diff) * x(j);
        }
        result(i) = sum;
    }

    return result;
}

// ============================================================================
// 3×3 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("levinson_durbin: 3x3 known answer",
                   "[levinson_durbin][test_toeplitz]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    // T = [4 2 1; 2 4 2; 1 2 4] (SPD Toeplitz)
    T r_vals[3][1] = {{4}, {2}, {1}};
    Vector r(r_vals);

    T b_vals[3][1] = {{7}, {8}, {7}};
    Vector b(b_vals);

    auto result = matrix_algorithms::levinson_durbin(r, b);

    REQUIRE(result.has_value());

    // Verify Tx = b
    auto Tx = toeplitz_matvec(r, result.value());
    REQUIRE(Tx == b);
}

// ============================================================================
// 4×4 RECONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("levinson_durbin: 4x4 reconstruction",
                   "[levinson_durbin][test_toeplitz]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 4>;

    // T = [10 3 1 0.5; 3 10 3 1; 1 3 10 3; 0.5 1 3 10]
    T r_vals[4][1] = {{10}, {3}, {1}, {T(0.5)}};
    Vector r(r_vals);

    T b_vals[4][1] = {{1}, {2}, {3}, {4}};
    Vector b(b_vals);

    auto result = matrix_algorithms::levinson_durbin(r, b);

    REQUIRE(result.has_value());

    auto Tx = toeplitz_matvec(r, result.value());
    REQUIRE(Tx == b);
}

// ============================================================================
// IDENTITY TOEPLITZ (r = [1 0 0 ...])
// ============================================================================

TEMPLATE_TEST_CASE("levinson_durbin: identity gives x = b",
                   "[levinson_durbin][edge][test_toeplitz]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    // r = [1 0 0] → T = I
    T r_vals[3][1] = {{1}, {0}, {0}};
    Vector r(r_vals);

    T b_vals[3][1] = {{5}, {-3}, {7}};
    Vector b(b_vals);

    auto result = matrix_algorithms::levinson_durbin(r, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == b);
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("levinson_durbin: 1x1",
                   "[levinson_durbin][edge][test_toeplitz]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 1>;

    T r_vals[1][1] = {{5}};
    Vector r(r_vals);

    T b_vals[1][1] = {{15}};
    Vector b(b_vals);

    T x_expected_vals[1][1] = {{3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::levinson_durbin(r, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// 2×2
// ============================================================================

TEMPLATE_TEST_CASE("levinson_durbin: 2x2",
                   "[levinson_durbin][test_toeplitz]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 2>;

    // T = [4 1; 1 4]
    T r_vals[2][1] = {{4}, {1}};
    Vector r(r_vals);

    T b_vals[2][1] = {{5}, {5}};
    Vector b(b_vals);

    auto result = matrix_algorithms::levinson_durbin(r, b);

    REQUIRE(result.has_value());

    auto Tx = toeplitz_matvec(r, result.value());
    REQUIRE(Tx == b);
}

// ============================================================================
// SINGULAR — ZERO DIAGONAL
// ============================================================================

TEMPLATE_TEST_CASE("levinson_durbin: zero diagonal returns Singular",
                   "[levinson_durbin][error][test_toeplitz]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T r_vals[3][1] = {{0}, {1}, {2}};
    Vector r(r_vals);

    T b_vals[3][1] = {{1}, {1}, {1}};
    Vector b(b_vals);

    auto result = matrix_algorithms::levinson_durbin(r, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// AGREES WITH LU SOLVE
// ============================================================================

TEST_CASE("levinson_durbin: agrees with LU solve",
          "[levinson_durbin][property][test_toeplitz]")
{
    using T = double;
    using Vector = FusedVector<T, 4>;
    using Matrix = FusedMatrix<T, 4, 4>;

    T r_vals[4][1] = {{10}, {3}, {1}, {T(0.5)}};
    Vector r(r_vals);

    T b_vals[4][1] = {{7}, {-3}, {10}, {5}};
    Vector b(b_vals);

    // Levinson-Durbin
    auto ld_result = matrix_algorithms::levinson_durbin(r, b);
    REQUIRE(ld_result.has_value());

    // LU solve with full Toeplitz matrix
    Matrix A = build_toeplitz(r);
    auto lu_result = matrix_algorithms::lu_solve(A, b);
    REQUIRE(lu_result.has_value());

    REQUIRE(ld_result.value() == lu_result.value());
}

// ============================================================================
// GENERIC PATH — 8 ELEMENTS
// ============================================================================

TEST_CASE("levinson_durbin: 8-element reconstruction",
          "[levinson_durbin][generic][test_toeplitz]")
{
    using T = double;
    using Vector = FusedVector<T, 8>;

    // Strongly diagonally dominant Toeplitz
    Vector r;
    r(0) = T(20);
    for (my_size_t i = 1; i < 8; ++i)
    {
        r(i) = T(1) / T(i + 1);
    }

    // Pick x, compute b = Tx
    Vector x_true;
    for (my_size_t i = 0; i < 8; ++i)
    {
        x_true(i) = T(i + 1);
    }

    auto b = toeplitz_matvec(r, x_true);

    auto result = matrix_algorithms::levinson_durbin(r, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_true);
}
