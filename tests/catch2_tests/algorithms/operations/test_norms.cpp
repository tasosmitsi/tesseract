#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/operations/norms.h"
#include "math/math_utils.h"

// ============================================================================
// NORM1 — KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("norm1: 3x3 known answer",
                   "[norm1][norms]", double, float, int)
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
                   "[norm1][norms]", double, float, int)
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
                   "[norm1][norms]", double, float, int)
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
                   "[norm1][norms]", double, float, int)
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
                   "[norm1][norms]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A(T(0));

    REQUIRE(matrix_algorithms::norm1(A) == T(0));
}

// ============================================================================
// NORM2 — 3-VECTOR KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("norm2: 3-vector known answer",
                   "[norm2][norms]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    // ‖[3, 4, 0]‖ = 5
    T v_vals[3][1] = {{3}, {4}, {0}};
    Vector v(v_vals);

    REQUIRE(matrix_algorithms::norm2(v) == T(5));
}

// ============================================================================
// NORM2 — UNIT VECTOR
// ============================================================================

TEMPLATE_TEST_CASE("norm2: unit vector has norm 1",
                   "[norm2][norms]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T v_vals[3][1] = {{0}, {0}, {1}};
    Vector v(v_vals);

    REQUIRE(matrix_algorithms::norm2(v) == T(1));
}

// ============================================================================
// NORM2 — ZERO VECTOR
// ============================================================================

TEMPLATE_TEST_CASE("norm2: zero vector is 0",
                   "[norm2][norms]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    Vector v(T(0));

    REQUIRE(matrix_algorithms::norm2(v) == T(0));
}

// ============================================================================
// NORM2 — 1-ELEMENT
// ============================================================================

TEMPLATE_TEST_CASE("norm2: 1-element",
                   "[norm2][norms]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 1>;

    T v_vals[1][1] = {{-7}};
    Vector v(v_vals);

    REQUIRE(matrix_algorithms::norm2(v) == T(7));
}

// ============================================================================
// NORM2 — NEGATIVE ENTRIES
// ============================================================================

TEMPLATE_TEST_CASE("norm2: negative entries",
                   "[norm2][norms]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 2>;

    // ‖[-3, 4]‖ = 5
    T v_vals[2][1] = {{-3}, {4}};
    Vector v(v_vals);

    REQUIRE(matrix_algorithms::norm2(v) == T(5));
}

// ============================================================================
// NORM2 — PROPERTY: ‖cv‖ = |c| · ‖v‖
// ============================================================================

TEST_CASE("norm2: norm(cv) = |c| * norm(v)",
          "[norm2][norms]")
{
    using T = double;
    using Vector = FusedVector<T, 3>;

    T v_vals[3][1] = {{1}, {2}, {3}};
    Vector v(v_vals);

    T c = -3.0;

    // cv manually
    Vector cv(T(0));
    cv(0) = c * v(0);
    cv(1) = c * v(1);
    cv(2) = c * v(2);

    REQUIRE(matrix_algorithms::norm2(cv) == math::abs(c) * matrix_algorithms::norm2(v));
}
