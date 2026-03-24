#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "algorithms/operations/rank_update.h"

// ============================================================================
// 2×2 KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("symmetric_rank_k_update: 2x2 known answer",
                   "[rank_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T F_vals[2][2] = {
        {1, 1},
        {0, 1}};
    Matrix F(F_vals);

    T P_vals[2][2] = {
        {1, 0},
        {0, 1}};
    Matrix P(P_vals);

    T Q_vals[2][2] = {
        {0, 0},
        {0, 0}};
    Matrix Q(Q_vals);

    // F*P*Fᵀ + Q = F*I*Fᵀ = F*Fᵀ
    // F*Fᵀ = [1 1; 0 1] * [1 0; 1 1] = [2 1; 1 1]
    T expected_vals[2][2] = {
        {2, 1},
        {1, 1}};
    Matrix expected(expected_vals);

    auto result = matrix_algorithms::symmetric_rank_k_update(F, P, Q);

    REQUIRE(result == expected);
}

// ============================================================================
// WITH PROCESS NOISE
// ============================================================================

TEMPLATE_TEST_CASE("symmetric_rank_k_update: 2x2 with noise",
                   "[rank_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T F_vals[2][2] = {
        {1, 1},
        {0, 1}};
    Matrix F(F_vals);

    T P_vals[2][2] = {
        {1, 0},
        {0, 1}};
    Matrix P(P_vals);

    T Q_vals[2][2] = {
        {T(0.1), 0},
        {0, T(0.1)}};
    Matrix Q(Q_vals);

    // F*P*Fᵀ = [2 1; 1 1], + Q = [2.1 1; 1 1.1]
    T expected_vals[2][2] = {
        {T(2.1), 1},
        {1, T(1.1)}};
    Matrix expected(expected_vals);

    auto result = matrix_algorithms::symmetric_rank_k_update(F, P, Q);

    REQUIRE(result == expected);
}

// ============================================================================
// IDENTITY TRANSITION
// ============================================================================

TEMPLATE_TEST_CASE("symmetric_rank_k_update: identity F gives P + Q",
                   "[rank_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I(0);
    I.setIdentity();

    T P_vals[3][3] = {
        {4, 2, 0},
        {2, 5, 1},
        {0, 1, 3}};
    Matrix P(P_vals);

    T Q_vals[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};
    Matrix Q(Q_vals);

    // I*P*Iᵀ + Q = P + Q
    auto result = matrix_algorithms::symmetric_rank_k_update(I, P, Q);

    REQUIRE(result == P + Q);
}

// ============================================================================
// PROPERTY: RESULT IS SYMMETRIC
// ============================================================================

TEST_CASE("symmetric_rank_k_update: result is symmetric",
          "[rank_update]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T F_vals[3][3] = {
        {1, 2, 0},
        {0, 1, 3},
        {1, 0, 1}};
    Matrix F(F_vals);

    // P is symmetric
    T P_vals[3][3] = {
        {4, 1, 0},
        {1, 3, 2},
        {0, 2, 5}};
    Matrix P(P_vals);

    // Q is symmetric
    T Q_vals[3][3] = {
        {T(0.1), 0, 0},
        {0, T(0.1), 0},
        {0, 0, T(0.1)}};
    Matrix Q(Q_vals);

    auto result = matrix_algorithms::symmetric_rank_k_update(F, P, Q);

    REQUIRE(result.isSymmetric());
}

// ============================================================================
// NO-NOISE OVERLOAD
// ============================================================================

TEMPLATE_TEST_CASE("symmetric_rank_k_update: no-noise overload",
                   "[rank_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T F_vals[2][2] = {
        {1, 1},
        {0, 1}};
    Matrix F(F_vals);

    T P_vals[2][2] = {
        {1, 0},
        {0, 1}};
    Matrix P(P_vals);

    T expected_vals[2][2] = {
        {2, 1},
        {1, 1}};
    Matrix expected(expected_vals);

    auto result = matrix_algorithms::symmetric_rank_k_update(F, P);

    REQUIRE(result == expected);
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("symmetric_rank_k_update: 1x1",
                   "[rank_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T F_vals[1][1] = {{3}};
    Matrix F(F_vals);

    T P_vals[1][1] = {{2}};
    Matrix P(P_vals);

    T Q_vals[1][1] = {{1}};
    Matrix Q(Q_vals);

    // 3*2*3 + 1 = 19
    T expected_vals[1][1] = {{19}};
    Matrix expected(expected_vals);

    auto result = matrix_algorithms::symmetric_rank_k_update(F, P, Q);

    REQUIRE(result == expected);
}

// ============================================================================
// GENERIC PATH — 7×7
// ============================================================================

TEMPLATE_TEST_CASE("symmetric_rank_k_update: 7x7 symmetry preserved",
                   "[rank_update]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 7, 7>;

    // Build non-trivial F
    Matrix F(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        F(i, i) = T(1);
        if (i + 1 < 7)
            F(i, i + 1) = T(0.5);
    }

    // Build SPD P = M*Mᵀ
    Matrix M(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        M(i, i) = T(i + 2);
        for (my_size_t j = 0; j < i; ++j)
        {
            M(i, j) = T(1) / T(i - j + 1);
        }
    }
    Matrix P = Matrix::matmul(M, M.transpose_view());

    // Build diagonal Q
    Matrix Q;
    Q.setDiagonal(T(0.01));

    auto result = matrix_algorithms::symmetric_rank_k_update(F, P, Q);

    REQUIRE(result.isSymmetric());
}
