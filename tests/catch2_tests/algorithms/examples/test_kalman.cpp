#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "algorithms/examples/kalman.h"
#include "algorithms/operations/rank_update.h"

using Catch::Approx;
using matrix_traits::MatrixStatus;

// ============================================================================
// KALMAN GAIN — 2-STATE, 1-MEASUREMENT (position tracking)
// ============================================================================

TEMPLATE_TEST_CASE("kalman_gain: 2-state 1-measurement",
                   "[kalman_gain][test_kalman]", double, float)
{
    using T = TestType;
    // State: [position, velocity], Measurement: [position]
    using StateMatrix = FusedMatrix<T, 2, 2>;
    using ObsMatrix = FusedMatrix<T, 1, 2>;
    using NoiseMatrix = FusedMatrix<T, 1, 1>;
    using GainMatrix = FusedMatrix<T, 2, 1>;

    // P = [1 0; 0 1]
    T P_vals[2][2] = {
        {1, 0},
        {0, 1}};
    StateMatrix P(P_vals);

    // H = [1 0] (observe position only)
    T H_vals[1][2] = {{1, 0}};
    ObsMatrix H(H_vals);

    // R = [0.1] (measurement noise)
    T R_vals[1][1] = {{T(0.1)}};
    NoiseMatrix R(R_vals);

    // S = H*P*Hᵀ + R = [1]*[1 0; 0 1]*[1; 0] + [0.1] = [1] + [0.1] = [1.1]
    // K = P*Hᵀ*S⁻¹ = [1 0; 0 1]*[1; 0]*(1/1.1) = [1/1.1; 0]
    T K_expected_vals[2][1] = {{T(1) / T(1.1)}, {0}};
    GainMatrix K_expected(K_expected_vals);

    auto result = matrix_algorithms::kalman_gain(P, H, R);

    REQUIRE(result.has_value());
    REQUIRE(result.value() == K_expected);
}

// ============================================================================
// KALMAN GAIN — 3-STATE, 2-MEASUREMENT
// ============================================================================

TEST_CASE("kalman_gain: 3-state 2-measurement",
          "[kalman_gain][test_kalman]")
{
    using T = double;
    using StateMatrix = FusedMatrix<T, 3, 3>;
    using ObsMatrix = FusedMatrix<T, 2, 3>;
    using NoiseMatrix = FusedMatrix<T, 2, 2>;
    using GainMatrix = FusedMatrix<T, 3, 2>;

    // P = I
    StateMatrix P(0);
    P.setIdentity();

    // H = [1 0 0; 0 1 0] (observe first two states)
    T H_vals[2][3] = {
        {1, 0, 0},
        {0, 1, 0}};
    ObsMatrix H(H_vals);

    // R = 0.5 * I
    T R_vals[2][2] = {
        {T(0.5), 0},
        {0, T(0.5)}};
    NoiseMatrix R(R_vals);

    auto result = matrix_algorithms::kalman_gain(P, H, R);

    REQUIRE(result.has_value());

    auto &K = result.value();

    // S = H*I*Hᵀ + R = [1 0; 0 1] + [0.5 0; 0 0.5] = [1.5 0; 0 1.5]
    // K = I*Hᵀ*S⁻¹ = [1 0; 0 1; 0 0] * diag(1/1.5, 1/1.5)
    //   = [1/1.5 0; 0 1/1.5; 0 0]
    REQUIRE(K(0, 0) == Approx(T(1) / T(1.5)));
    REQUIRE(K(1, 1) == Approx(T(1) / T(1.5)));
    REQUIRE(K(2, 0) == Approx(T(0)).margin(T(1e-9)));
    REQUIRE(K(2, 1) == Approx(T(0)).margin(T(1e-9)));
}

// ============================================================================
// KALMAN GAIN — SINGULAR RETURNS ERROR
// ============================================================================

TEMPLATE_TEST_CASE("kalman_gain: singular S returns Singular",
                   "[kalman_gain][error][test_kalman]", double, float)
{
    using T = TestType;
    using StateMatrix = FusedMatrix<T, 2, 2>;
    using ObsMatrix = FusedMatrix<T, 2, 2>;
    using NoiseMatrix = FusedMatrix<T, 2, 2>;

    StateMatrix P(T(0)); // zero covariance
    ObsMatrix H(T(0));   // zero observation
    NoiseMatrix R(T(0)); // zero noise → S = 0, singular

    auto result = matrix_algorithms::kalman_gain(P, H, R);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::Singular);
}

// ============================================================================
// JOSEPH UPDATE — 2-STATE, 1-MEASUREMENT
// ============================================================================

TEMPLATE_TEST_CASE("joseph_update: 2-state 1-measurement",
                   "[joseph_update][test_kalman]", double, float)
{
    using T = TestType;
    using StateMatrix = FusedMatrix<T, 2, 2>;
    using ObsMatrix = FusedMatrix<T, 1, 2>;
    using NoiseMatrix = FusedMatrix<T, 1, 1>;
    using GainMatrix = FusedMatrix<T, 2, 1>;

    T P_vals[2][2] = {
        {1, 0},
        {0, 1}};
    StateMatrix P(P_vals);

    T H_vals[1][2] = {{1, 0}};
    ObsMatrix H(H_vals);

    T R_vals[1][1] = {{T(0.1)}};
    NoiseMatrix R(R_vals);

    // Compute gain first
    auto gain_result = matrix_algorithms::kalman_gain(P, H, R);
    REQUIRE(gain_result.has_value());
    auto &K = gain_result.value();

    auto P_updated = matrix_algorithms::joseph_update(K, H, P, R);

    // Result must be symmetric
    REQUIRE(P_updated.isSymmetric());

    // Updated covariance should be smaller than prior (we gained information)
    REQUIRE(P_updated(0, 0) < P(0, 0));
}

// ============================================================================
// JOSEPH UPDATE — RESULT IS SYMMETRIC
// ============================================================================

TEST_CASE("joseph_update: result is symmetric",
          "[joseph_update][property][test_kalman]")
{
    using T = double;
    using StateMatrix = FusedMatrix<T, 3, 3>;
    using ObsMatrix = FusedMatrix<T, 2, 3>;
    using NoiseMatrix = FusedMatrix<T, 2, 2>;
    using GainMatrix = FusedMatrix<T, 3, 2>;

    // Non-diagonal P (still SPD)
    T P_vals[3][3] = {
        {4, 1, 0},
        {1, 3, 2},
        {0, 2, 5}};
    StateMatrix P(P_vals);

    T H_vals[2][3] = {
        {1, 0, 0},
        {0, 1, 0}};
    ObsMatrix H(H_vals);

    T R_vals[2][2] = {
        {T(0.5), T(0.1)},
        {T(0.1), T(0.5)}};
    NoiseMatrix R(R_vals);

    auto gain_result = matrix_algorithms::kalman_gain(P, H, R);
    REQUIRE(gain_result.has_value());
    auto &K = gain_result.value();

    auto P_updated = matrix_algorithms::joseph_update(K, H, P, R);

    REQUIRE(P_updated.isSymmetric());
}

// ============================================================================
// JOSEPH UPDATE — ZERO GAIN GIVES P BACK
// ============================================================================

TEMPLATE_TEST_CASE("joseph_update: zero gain returns P",
                   "[joseph_update][edge][test_kalman]", double, float)
{
    using T = TestType;
    using StateMatrix = FusedMatrix<T, 2, 2>;
    using ObsMatrix = FusedMatrix<T, 1, 2>;
    using NoiseMatrix = FusedMatrix<T, 1, 1>;
    using GainMatrix = FusedMatrix<T, 2, 1>;

    T P_vals[2][2] = {
        {4, 1},
        {1, 3}};
    StateMatrix P(P_vals);

    T H_vals[1][2] = {{1, 0}};
    ObsMatrix H(H_vals);

    T R_vals[1][1] = {{1}};
    NoiseMatrix R(R_vals);

    GainMatrix K(T(0)); // zero gain → no update

    // (I - 0·H)·P·(I - 0·H)ᵀ + 0·R·0ᵀ = I·P·I = P
    auto P_updated = matrix_algorithms::joseph_update(K, H, P, R);

    REQUIRE(P_updated == P);
}

// ============================================================================
// FULL KALMAN STEP — PREDICT + UPDATE CONSISTENCY
// ============================================================================

TEST_CASE("kalman: full predict-update step",
          "[kalman_gain][joseph_update][integration][test_kalman]")
{
    using T = double;
    using StateMatrix = FusedMatrix<T, 2, 2>;
    using ObsMatrix = FusedMatrix<T, 1, 2>;
    using MeasNoise = FusedMatrix<T, 1, 1>;
    using GainMatrix = FusedMatrix<T, 2, 1>;

    // Initial state covariance
    T P_vals[2][2] = {
        {1, 0},
        {0, 1}};
    StateMatrix P(P_vals);

    // State transition (constant velocity model)
    T F_vals[2][2] = {
        {1, 1},
        {0, 1}};
    StateMatrix F(F_vals);

    // Process noise
    T Q_vals[2][2] = {
        {T(0.01), 0},
        {0, T(0.01)}};
    StateMatrix Q(Q_vals);

    // Observation (position only)
    T H_vals[1][2] = {{1, 0}};
    ObsMatrix H(H_vals);

    // Measurement noise
    T R_vals[1][1] = {{T(0.5)}};
    MeasNoise R(R_vals);

    // --- Predict ---
    StateMatrix P_predict = matrix_algorithms::symmetric_rank_k_update(F, P, Q);

    REQUIRE(P_predict.isSymmetric());

    // --- Update ---
    auto gain_result = matrix_algorithms::kalman_gain(P_predict, H, R);
    REQUIRE(gain_result.has_value());
    auto &K = gain_result.value();

    StateMatrix P_update = matrix_algorithms::joseph_update(K, H, P_predict, R);

    REQUIRE(P_update.isSymmetric());

    // Covariance should decrease after update (we gained information)
    REQUIRE(P_update(0, 0) < P_predict(0, 0));

    // Position uncertainty should decrease more than velocity
    // (we directly observed position)
    T pos_reduction = P_predict(0, 0) - P_update(0, 0);
    T vel_reduction = P_predict(1, 1) - P_update(1, 1);
    REQUIRE(pos_reduction > vel_reduction);
}

// ============================================================================
// JOSEPH UPDATE — 1×1
// ============================================================================

TEMPLATE_TEST_CASE("joseph_update: 1x1 edge case",
                   "[joseph_update][edge][test_kalman]", double, float)
{
    using T = TestType;
    using Matrix1 = FusedMatrix<T, 1, 1>;

    T P_vals[1][1] = {{4}};
    Matrix1 P(P_vals);

    T H_vals[1][1] = {{1}};
    Matrix1 H(H_vals);

    T R_vals[1][1] = {{1}};
    Matrix1 R(R_vals);

    // K = P*H'*(H*P*H'+R)⁻¹ = 4*1*(4+1)⁻¹ = 4/5 = 0.8
    auto gain_result = matrix_algorithms::kalman_gain(P, H, R);
    REQUIRE(gain_result.has_value());
    REQUIRE(gain_result.value()(0, 0) == Approx(T(0.8)));

    auto &K = gain_result.value();
    auto P_updated = matrix_algorithms::joseph_update(K, H, P, R);

    // (1-0.8)² * 4 + 0.8² * 1 = 0.04*4 + 0.64 = 0.16 + 0.64 = 0.8
    REQUIRE(P_updated(0, 0) == Approx(T(0.8)));
}
