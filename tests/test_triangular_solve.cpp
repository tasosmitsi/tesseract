#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/solvers/triangular_solve.h"

using matrix_traits::MatrixStatus;

// ============================================================================
// FORWARD SUBSTITUTION — KNOWN ANSWER (3×3, hits unrolled path)
// ============================================================================

TEMPLATE_TEST_CASE("forward_substitute: 3x3 known answer",
                   "[forward_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // L*x = b  →  x = [2, 1, 3]
    // L*x: row0: 2*2=4, row1: 6*2+1*1=13, row2: -8*2+5*1+3*3=-2

    T L_vals[3][3] = {
        {2, 0, 0},
        {6, 1, 0},
        {-8, 5, 3}};
    Matrix L(L_vals);

    T b_vals[3][1] = {{4}, {13}, {-2}};
    Vector b(b_vals);

    T x_expected_vals[3][1] = {{2}, {1}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::forward_substitute(L, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// FORWARD SUBSTITUTION — 4×4 (hits unrolled path)
// ============================================================================

TEMPLATE_TEST_CASE("forward_substitute: 4x4 known answer",
                   "[forward_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vector = FusedVector<T, 4>;

    T L_vals[4][4] = {
        {2, 0, 0, 0},
        {1, 3, 0, 0},
        {0, 1, 2, 0},
        {1, 0, 1, 4}};
    Matrix L(L_vals);

    // x = [1, -1, 2, 0]
    // b = L*x: row0: 2, row1: 1-3=-2, row2: -1+4=3, row3: 1+2=3
    T b_vals[4][1] = {{2}, {-2}, {3}, {3}};
    Vector b(b_vals);

    T x_expected_vals[4][1] = {{1}, {-1}, {2}, {0}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::forward_substitute(L, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// FORWARD SUBSTITUTION — UNIT DIAGONAL (3×3)
// ============================================================================

TEMPLATE_TEST_CASE("forward_substitute: 3x3 unit diagonal",
                   "[forward_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // L with unit diagonal (diagonal values ignored)
    T L_vals[3][3] = {
        {999, 0, 0},
        {2, 999, 0},
        {3, 4, 999}};
    Matrix L(L_vals);

    // With UnitDiag: x(0) = b(0), x(1) = b(1) - 2*x(0), x(2) = b(2) - 3*x(0) - 4*x(1)
    // x = [5, 1, 3] → b(0)=5, b(1)=1+2*5=11, b(2)=3+3*5+4*1=22
    T b_vals[3][1] = {{5}, {11}, {22}};
    Vector b(b_vals);

    T x_expected_vals[3][1] = {{5}, {1}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::forward_substitute<true>(L, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// FORWARD SUBSTITUTION — IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("forward_substitute: identity gives x = b",
                   "[forward_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    Matrix I(0);
    I.setIdentity();

    T b_vals[2][1] = {{7}, {-3}};
    Vector b(b_vals);

    auto result = matrix_algorithms::forward_substitute(I, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == b);
}

// ============================================================================
// FORWARD SUBSTITUTION — SINGULAR ERROR
// ============================================================================

TEMPLATE_TEST_CASE("forward_substitute: singular diagonal returns Singular",
                   "[forward_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    T L_vals[2][2] = {
        {1, 0},
        {3, 0} // zero diagonal at (1,1)
    };
    Matrix L(L_vals);

    T b_vals[2][1] = {{1}, {1}};
    Vector b(b_vals);

    auto result = matrix_algorithms::forward_substitute(L, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// FORWARD SUBSTITUTION — 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("forward_substitute: 1x1",
                   "[forward_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;
    using Vector = FusedVector<T, 1>;

    T L_vals[1][1] = {{4}};
    Matrix L(L_vals);

    T b_vals[1][1] = {{8}};
    Vector b(b_vals);

    T x_expected_vals[1][1] = {{2}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::forward_substitute(L, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// BACK SUBSTITUTION — KNOWN ANSWER (3×3, hits unrolled path)
// ============================================================================

TEMPLATE_TEST_CASE("back_substitute: 3x3 known answer",
                   "[back_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // U*x = b → x = [2, 1, 3]
    // row0: 2*2+6*1-8*3=-14, row1: 1+15=16, row2: 3*3=9

    T U_vals[3][3] = {
        {2, 6, -8},
        {0, 1, 5},
        {0, 0, 3}};
    Matrix U(U_vals);

    T b_vals[3][1] = {{-14}, {16}, {9}};
    Vector b(b_vals);

    T x_expected_vals[3][1] = {{2}, {1}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::back_substitute(U, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// BACK SUBSTITUTION — 4×4 (hits unrolled path)
// ============================================================================

TEMPLATE_TEST_CASE("back_substitute: 4x4 known answer",
                   "[back_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vector = FusedVector<T, 4>;

    T U_vals[4][4] = {
        {2, 1, 0, 1},
        {0, 3, 1, 0},
        {0, 0, 2, 1},
        {0, 0, 0, 4}};
    Matrix U(U_vals);

    // x = [1, -1, 2, 0]
    // b = U*x: row3: 0, row2: 4+0=4, row1: -3+2=-1, row0: 2-1+0=1
    T b_vals[4][1] = {{1}, {-1}, {4}, {0}};
    Vector b(b_vals);

    T x_expected_vals[4][1] = {{1}, {-1}, {2}, {0}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::back_substitute(U, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// BACK SUBSTITUTION — UNIT DIAGONAL (3×3)
// ============================================================================

TEMPLATE_TEST_CASE("back_substitute: 3x3 unit diagonal",
                   "[back_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T U_vals[3][3] = {
        {999, 2, 3},
        {0, 999, 4},
        {0, 0, 999}};
    Matrix U(U_vals);

    // With UnitDiag: x(2)=b(2), x(1)=b(1)-4*x(2), x(0)=b(0)-2*x(1)-3*x(2)
    // x = [1, 2, 3] → b(2)=3, b(1)=2+4*3=14, b(0)=1+2*2+3*3=14
    T b_vals[3][1] = {{14}, {14}, {3}};
    Vector b(b_vals);

    T x_expected_vals[3][1] = {{1}, {2}, {3}};
    Vector x_expected(x_expected_vals);

    auto result = matrix_algorithms::back_substitute<true>(U, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_expected);
}

// ============================================================================
// BACK SUBSTITUTION — IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("back_substitute: identity gives x = b",
                   "[back_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    Matrix I(0);
    I.setIdentity();

    T b_vals[2][1] = {{5}, {-2}};
    Vector b(b_vals);

    auto result = matrix_algorithms::back_substitute(I, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == b);
}

// ============================================================================
// BACK SUBSTITUTION — SINGULAR ERROR
// ============================================================================

TEMPLATE_TEST_CASE("back_substitute: singular diagonal returns Singular",
                   "[back_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    T U_vals[2][2] = {
        {0, 3}, // zero diagonal at (0,0)
        {0, 1}};
    Matrix U(U_vals);

    T b_vals[2][1] = {{1}, {1}};
    Vector b(b_vals);

    auto result = matrix_algorithms::back_substitute(U, b);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// MULTI-RHS FORWARD SUBSTITUTION
// ============================================================================

TEMPLATE_TEST_CASE("forward_substitute: multi-RHS 3x3 x 2",
                   "[forward_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using RHS = FusedMatrix<T, 3, 2>;

    T L_vals[3][3] = {
        {2, 0, 0},
        {6, 1, 0},
        {-8, 5, 3}};
    Matrix L(L_vals);

    // Column 0: x=[2,1,3] → b=[4,13,-2] (same as single RHS test)
    // Column 1: x=[1,0,-1] → b=[2, 6, -11]
    //   row0: 2*1=2, row1: 6*1+1*0=6, row2: -8*1+5*0+3*(-1)=-11
    T B_vals[3][2] = {
        {4, 2},
        {13, 6},
        {-2, -11}};
    RHS B(B_vals);

    T X_expected_vals[3][2] = {
        {2, 1},
        {1, 0},
        {3, -1}};
    RHS X_expected(X_expected_vals);

    auto result = matrix_algorithms::forward_substitute(L, B);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == X_expected);
}

// ============================================================================
// MULTI-RHS BACK SUBSTITUTION
// ============================================================================

TEMPLATE_TEST_CASE("back_substitute: multi-RHS 3x3 x 2",
                   "[back_substitute][triangular_solve]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using RHS = FusedMatrix<T, 3, 2>;

    T U_vals[3][3] = {
        {2, 6, -8},
        {0, 1, 5},
        {0, 0, 3}};
    Matrix U(U_vals);

    // Column 0: x=[2,1,3] → b=[-14,16,9] (same as single RHS test)
    // Column 1: x=[1,0,-1] → b=[10,-5,-3]
    //   row2: 3*(-1)=-3, row1: 0+5*(-1)=-5, row0: 2*1+0+(-8)*(-1)=10
    T B_vals[3][2] = {
        {-14, 10},
        {16, -5},
        {9, -3}};
    RHS B(B_vals);

    T X_expected_vals[3][2] = {
        {2, 1},
        {1, 0},
        {3, -1}};
    RHS X_expected(X_expected_vals);

    auto result = matrix_algorithms::back_substitute(U, B);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == X_expected);
}

// ============================================================================
// GENERIC PATH — 7×7 FORWARD SUBSTITUTION (not unrolled)
// ============================================================================

TEST_CASE("forward_substitute: 7x7 generic path",
          "[forward_substitute][triangular_solve]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;
    using Vector = FusedVector<T, 7>;

    // Build L with strong diagonal and mild off-diagonal
    Matrix L(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        L(i, i) = T(i + 2);
        for (my_size_t j = 0; j < i; ++j)
        {
            L(i, j) = T(1) / T(i - j + 1);
        }
    }

    // Pick x, compute b = L*x, then solve
    Vector x_true(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        x_true(i) = T(i + 1);
    }

    // b = L * x_true
    Vector b(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        T sum = 0;
        for (my_size_t k = 0; k <= i; ++k)
        {
            sum += L(i, k) * x_true(k);
        }
        b(i) = sum;
    }

    auto result = matrix_algorithms::forward_substitute(L, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_true);
}

// ============================================================================
// GENERIC PATH — 7×7 BACK SUBSTITUTION (not unrolled)
// ============================================================================

TEST_CASE("back_substitute: 7x7 generic path",
          "[back_substitute][triangular_solve]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;
    using Vector = FusedVector<T, 7>;

    // Build U with strong diagonal and mild off-diagonal
    Matrix U(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        U(i, i) = T(i + 2);
        for (my_size_t j = i + 1; j < 7; ++j)
        {
            U(i, j) = T(1) / T(j - i + 1);
        }
    }

    // Pick x, compute b = U*x, then solve
    Vector x_true(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        x_true(i) = T(i + 1);
    }

    // b = U * x_true
    Vector b(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        T sum = 0;
        for (my_size_t k = i; k < 7; ++k)
        {
            sum += U(i, k) * x_true(k);
        }
        b(i) = sum;
    }

    auto result = matrix_algorithms::back_substitute(U, b);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == x_true);
}

// ============================================================================
// GENERIC PATH — 7×7 MULTI-RHS FORWARD SUBSTITUTION
// ============================================================================

TEST_CASE("forward_substitute: 7x7 x 3 multi-RHS generic path",
          "[forward_substitute][triangular_solve]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;
    using RHS = FusedMatrix<T, 7, 3>;

    Matrix L(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        L(i, i) = T(i + 2);
        for (my_size_t j = 0; j < i; ++j)
        {
            L(i, j) = T(1) / T(i - j + 1);
        }
    }

    // Three different x columns
    RHS X_true(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        X_true(i, 0) = T(i + 1);
        X_true(i, 1) = T(7 - i);
        X_true(i, 2) = (i % 2 == 0) ? T(1) : T(-1);
    }

    // B = L * X_true
    RHS B(T(0));
    for (my_size_t j = 0; j < 3; ++j)
    {
        for (my_size_t i = 0; i < 7; ++i)
        {
            T sum = 0;
            for (my_size_t k = 0; k <= i; ++k)
            {
                sum += L(i, k) * X_true(k, j);
            }
            B(i, j) = sum;
        }
    }

    auto result = matrix_algorithms::forward_substitute(L, B);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == X_true);
}

// ============================================================================
// GENERIC PATH — 7×7 MULTI-RHS BACK SUBSTITUTION
// ============================================================================

TEST_CASE("back_substitute: 7x7 x 3 multi-RHS generic path",
          "[back_substitute][triangular_solve]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;
    using RHS = FusedMatrix<T, 7, 3>;

    Matrix U(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        U(i, i) = T(i + 2);
        for (my_size_t j = i + 1; j < 7; ++j)
        {
            U(i, j) = T(1) / T(j - i + 1);
        }
    }

    RHS X_true(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        X_true(i, 0) = T(i + 1);
        X_true(i, 1) = T(7 - i);
        X_true(i, 2) = (i % 2 == 0) ? T(1) : T(-1);
    }

    // B = U * X_true
    RHS B(T(0));
    for (my_size_t j = 0; j < 3; ++j)
    {
        for (my_size_t i = 0; i < 7; ++i)
        {
            T sum = 0;
            for (my_size_t k = i; k < 7; ++k)
            {
                sum += U(i, k) * X_true(k, j);
            }
            B(i, j) = sum;
        }
    }

    auto result = matrix_algorithms::back_substitute(U, B);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == X_true);
}
