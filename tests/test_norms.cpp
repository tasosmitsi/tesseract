#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "algorithms/operations/norms.h"

// ============================================================================
// NORM1 — KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("norm1: 3x3 known answer",
                   "[norm1]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    // ‖A‖₁ = max column sum of |A|
    // col 0: |1|+|4|+|7| = 12
    // col 1: |2|+|5|+|8| = 15
    // col 2: |3|+|6|+|9| = 18  ← max
    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::norm1(A) == T(18));
}

// ============================================================================
// NORM1 — NEGATIVE ENTRIES
// ============================================================================

TEMPLATE_TEST_CASE("norm1: negative entries",
                   "[norm1]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    // col 0: |-3|+|1| = 4
    // col 1: |2|+|-5| = 7  ← max
    T A_vals[2][2] = {
        {-3, 2},
        {1, -5}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::norm1(A) == T(7));
}

// ============================================================================
// NORM1 — IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("norm1: identity is 1",
                   "[norm1]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I;
    I.setIdentity();

    REQUIRE(matrix_algorithms::norm1(I) == T(1));
}

// ============================================================================
// NORM1 — 1×1
// ============================================================================

TEMPLATE_TEST_CASE("norm1: 1x1",
                   "[norm1]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{-7}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::norm1(A) == T(7));
}

// ============================================================================
// NORM1 — ZERO MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("norm1: zero matrix is 0",
                   "[norm1]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A(T(0));

    REQUIRE(matrix_algorithms::norm1(A) == T(0));
}
