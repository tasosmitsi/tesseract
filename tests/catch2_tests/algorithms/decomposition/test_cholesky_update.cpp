#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/decomposition/cholesky.h"
#include "algorithms/decomposition/cholesky_update.h"

using Catch::Approx;

// ============================================================================
// 2×2 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_rank1_update: 2x2 known answer",
                   "[cholesky_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    // A = [4 2; 2 5], L from Cholesky of A
    T A_vals[2][2] = {
        {4, 2},
        {2, 5}};
    Matrix A(A_vals);

    auto chol_result = matrix_algorithms::cholesky(A);
    REQUIRE(chol_result.has_value());
    auto &L = chol_result.value();

    // v = [1, 0]
    T v_vals[2][1] = {{1}, {0}};
    Vector v(v_vals);

    // A' = A + v*vᵀ = [4+1 2; 2 5] = [5 2; 2 5]
    // Compute expected L' from fresh Cholesky of A'
    T Ap_vals[2][2] = {
        {5, 2},
        {2, 5}};
    Matrix Ap(Ap_vals);

    auto expected_result = matrix_algorithms::cholesky(Ap);
    REQUIRE(expected_result.has_value());
    auto &L_expected = expected_result.value();

    auto Lp = matrix_algorithms::cholesky_rank1_update(L, v);

    REQUIRE(Lp == L_expected);
}

// ============================================================================
// 3×3 RECONSTRUCTION PROPERTY
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_rank1_update: 3x3 reconstruction L'L'T = A + vvT",
                   "[cholesky_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // SPD matrix
    T A_vals[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};
    Matrix A(A_vals);

    auto chol_result = matrix_algorithms::cholesky(A);
    REQUIRE(chol_result.has_value());
    auto &L = chol_result.value();

    T v_vals[3][1] = {{1}, {2}, {3}};
    Vector v(v_vals);

    auto Lp = matrix_algorithms::cholesky_rank1_update(L, v);

    // Verify L'L'ᵀ = A + vvᵀ
    auto LpLpT = Matrix::matmul(Lp, Lp.transpose_view());

    // A + vvᵀ
    Matrix Ap;
    Ap = A + Matrix::matmul(v, v.transpose_view());

    REQUIRE(LpLpT == Ap);
}

// ============================================================================
// RESULT IS LOWER TRIANGULAR
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_rank1_update: result is lower-triangular",
                   "[cholesky_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};
    Matrix A(A_vals);

    auto chol_result = matrix_algorithms::cholesky(A);
    REQUIRE(chol_result.has_value());

    T v_vals[3][1] = {{2}, {-1}, {3}};
    Vector v(v_vals);

    auto Lp = matrix_algorithms::cholesky_rank1_update(chol_result.value(), v);

    REQUIRE(Lp.isLowerTriangular());
}

// ============================================================================
// ZERO VECTOR — NO CHANGE
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_rank1_update: zero vector gives L unchanged",
                   "[cholesky_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    T A_vals[2][2] = {
        {4, 2},
        {2, 5}};
    Matrix A(A_vals);

    auto chol_result = matrix_algorithms::cholesky(A);
    REQUIRE(chol_result.has_value());
    auto &L = chol_result.value();

    Vector v(T(0));

    auto Lp = matrix_algorithms::cholesky_rank1_update(L, v);

    REQUIRE(Lp == L);
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_rank1_update: 1x1",
                   "[cholesky_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;
    using Vector = FusedVector<T, 1>;

    // A = [9], L = [3]
    T L_vals[1][1] = {{3}};
    Matrix L(L_vals);

    // v = [4]
    T v_vals[1][1] = {{4}};
    Vector v(v_vals);

    // A' = 9 + 16 = 25, L' = [5]
    auto Lp = matrix_algorithms::cholesky_rank1_update(L, v);

    REQUIRE(Lp(0, 0) == Approx(T(5)));
}

// ============================================================================
// AGREES WITH FRESH CHOLESKY — 4×4
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_rank1_update: 4x4 agrees with fresh cholesky",
                   "[cholesky_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;
    using Vector = FusedVector<T, 4>;

    T A_vals[4][4] = {
        {4, 2, 0, 2},
        {2, 10, 3, 1},
        {0, 3, 5, 2},
        {2, 1, 2, 18}};
    Matrix A(A_vals);

    auto chol_result = matrix_algorithms::cholesky(A);
    REQUIRE(chol_result.has_value());

    T v_vals[4][1] = {{1}, {-1}, {2}, {0}};
    Vector v(v_vals);

    auto Lp = matrix_algorithms::cholesky_rank1_update(chol_result.value(), v);

    // A + vvᵀ
    Matrix Ap;
    Ap = A + Matrix::matmul(v, v.transpose_view());

    // Fresh Cholesky of A + vvᵀ
    auto fresh_result = matrix_algorithms::cholesky(Ap);
    REQUIRE(fresh_result.has_value());

    REQUIRE(Lp == fresh_result.value());
}

// ============================================================================
// GENERIC PATH — 7×7
// ============================================================================

TEMPLATE_TEST_CASE("cholesky_rank1_update: 7x7 reconstruction",
                   "[cholesky_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 7, 7>;
    using Vector = FusedVector<T, 7>;

    // Build SPD: A = M*Mᵀ
    Matrix M;
    for (my_size_t i = 0; i < 7; ++i)
    {
        M(i, i) = T(i + 2);
        for (my_size_t j = 0; j < i; ++j)
        {
            M(i, j) = T(1) / T(i - j + 1);
        }
    }
    auto A = Matrix::matmul(M, M.transpose_view());

    auto chol_result = matrix_algorithms::cholesky(A);
    REQUIRE(chol_result.has_value());

    // Build v
    Vector v;
    for (my_size_t i = 0; i < 7; ++i)
    {
        v(i) = T(i + 1) * T(0.1);
    }

    auto Lp = matrix_algorithms::cholesky_rank1_update(chol_result.value(), v);

    // Verify L'L'ᵀ = A + vvᵀ
    auto LpLpT = Matrix::matmul(Lp, Lp.transpose_view());

    Matrix Ap;
    Ap = A + Matrix::matmul(v, v.transpose_view());

    REQUIRE(LpLpT == Ap);
}
