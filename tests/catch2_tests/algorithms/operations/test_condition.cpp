#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "algorithms/operations/condition.h"

using Catch::Approx;
using matrix_traits::MatrixStatus;

// ============================================================================
// CONDITION — IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("condition: identity has cond = 1",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I(0);
    I.setIdentity();

    auto result = matrix_algorithms::condition(I);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Approx(T(1)));
}

// ============================================================================
// CONDITION — SCALED IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("condition: scaled identity has cond = 1",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    // cI has cond = |c|·(1/|c|) · |c|·(1/|c|) ... wait
    // ‖cI‖₁ = |c|, ‖(cI)⁻¹‖₁ = 1/|c|, cond = 1
    T A_vals[2][2] = {
        {5, 0},
        {0, 5}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::condition(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Approx(T(1)));
}

// ============================================================================
// CONDITION — DIAGONAL MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("condition: diagonal matrix",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // diag(1, 2, 100)
    // ‖A‖₁ = 100, ‖A⁻¹‖₁ = max(1, 0.5, 0.01) = 1
    // cond = 100 * 1 = 100
    T A_vals[3][3] = {
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 100}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::condition(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Approx(T(100)));
}

// ============================================================================
// CONDITION — WELL-CONDITIONED MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("condition: well-conditioned SPD matrix",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // Diagonally dominant SPD → should have small condition number
    T A_vals[3][3] = {
        {10, 1, 0},
        {1, 10, 1},
        {0, 1, 10}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::condition(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() < T(5)); // well-conditioned
}

// ============================================================================
// CONDITION — ILL-CONDITIONED MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("condition: ill-conditioned matrix has large cond",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // Near-singular: third row almost linearly dependent
    T A_vals[3][3] = {
        {1, 0, 0},
        {0, 1000, 0},
        {0, 0, 1e6}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::condition(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() > T(1e4)); // ill-conditioned
}

// ============================================================================
// CONDITION — SINGULAR RETURNS ERROR
// ============================================================================

TEMPLATE_TEST_CASE("condition: singular matrix returns Singular",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T A_vals[2][2] = {
        {1, 2},
        {2, 4}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::condition(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// CONDITION — 1×1
// ============================================================================

TEMPLATE_TEST_CASE("condition: 1x1",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{7}};
    Matrix A(A_vals);

    // cond = |7| * |1/7| = 1
    auto result = matrix_algorithms::condition(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == Approx(T(1)));
}

// ============================================================================
// PROPERTY: cond(A) >= 1
// ============================================================================

TEMPLATE_TEST_CASE("condition: cond(A) >= 1",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, -1, 3},
        {4, 5, -2},
        {-1, 3, 7}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::condition(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() >= T(1));
}

// ============================================================================
// PROPERTY: cond(cA) = cond(A) for scalar c != 0
// ============================================================================

TEMPLATE_TEST_CASE("condition: cond(cA) = cond(A)",
                   "[condition]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, -1, 3},
        {4, 5, -2},
        {-1, 3, 7}};
    Matrix A(A_vals), cA;

    cA = A * T(5);

    auto cond_A = matrix_algorithms::condition(A);
    auto cond_cA = matrix_algorithms::condition(cA);

    REQUIRE(cond_A.has_value());
    REQUIRE(cond_cA.has_value());
    REQUIRE(cond_cA.value() == Approx(cond_A.value()));
}

// ============================================================================
// GENERIC PATH — 7×7
// ============================================================================

TEMPLATE_TEST_CASE("condition: 7x7 diagonally dominant has small cond",
                   "[condition]", double, float)
{
    using T = TestType;
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

    auto result = matrix_algorithms::condition(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value() >= T(1));
    REQUIRE(result.value() < T(10)); // diagonally dominant → well-conditioned
}
