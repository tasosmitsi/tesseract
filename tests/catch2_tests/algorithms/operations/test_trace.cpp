#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "algorithms/operations/trace.h"

// ============================================================================
// KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("trace: 3x3 known answer",
                   "[trace]", double, float, int32_t)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::trace(A) == T(15));
}

// ============================================================================
// IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("trace: identity is N",
                   "[trace]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    Matrix I;
    I.setIdentity();

    REQUIRE(matrix_algorithms::trace(I) == T(4));
}

// ============================================================================
// 1×1
// ============================================================================

TEMPLATE_TEST_CASE("trace: 1x1",
                   "[trace]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{42}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::trace(A) == T(42));
}

// ============================================================================
// ZERO MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("trace: zero matrix is 0",
                   "[trace]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A(T(0));

    REQUIRE(matrix_algorithms::trace(A) == T(0));
}

// ============================================================================
// DIAGONAL MATRIX
// ============================================================================

TEMPLATE_TEST_CASE("trace: diagonal matrix",
                   "[trace]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, 0, 0},
        {0, 3, 0},
        {0, 0, 5}};
    Matrix A(A_vals);

    REQUIRE(matrix_algorithms::trace(A) == T(10));
}

// ============================================================================
// PROPERTY: tr(A + B) = tr(A) + tr(B)
// ============================================================================

TEST_CASE("trace: tr(A+B) = tr(A) + tr(B)",
          "[trace]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    Matrix A(A_vals), ApB;

    T B_vals[3][3] = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}};
    Matrix B(B_vals);

    ApB = A + B;

    REQUIRE(matrix_algorithms::trace(ApB) ==
            matrix_algorithms::trace(A) + matrix_algorithms::trace(B));
}

// ============================================================================
// PROPERTY: tr(cA) = c · tr(A)
// ============================================================================

TEST_CASE("trace: tr(cA) = c * tr(A)",
          "[trace]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    Matrix A(A_vals), cA;

    T c = 3.0;
    cA = A * c;

    REQUIRE(matrix_algorithms::trace(cA) == c * matrix_algorithms::trace(A));
}

// ============================================================================
// PROPERTY: tr(Aᵀ) = tr(A)
// ============================================================================

TEST_CASE("trace: tr(A^T) = tr(A)",
          "[trace]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {2, -1, 3},
        {4, 5, -2},
        {-1, 3, 7}};
    Matrix A(A_vals), At;

    At = A.transpose_view();

    REQUIRE(matrix_algorithms::trace(A) == matrix_algorithms::trace(At));
}

// ============================================================================
// GENERIC PATH — 7×7
// ============================================================================

TEST_CASE("trace: 7x7",
          "[trace]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

    Matrix A(T(0));
    T expected = 0;

    for (my_size_t i = 0; i < 7; ++i)
    {
        A(i, i) = T(i + 1);
        expected += T(i + 1);
    }

    REQUIRE(matrix_algorithms::trace(A) == expected);
}
