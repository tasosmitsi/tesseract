#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/operations/skew_symmetric.h"
#include "algorithms/operations/determinant.h"
#include "math/math_utils.h"

using Catch::Approx;

// ============================================================================
// SKEW-SYMMETRIC — KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("skew_symmetric: known answer",
                   "[skew_symmetric]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T v_vals[3][1] = {{1}, {2}, {3}};
    Vector v(v_vals);

    T expected_vals[3][3] = {
        {0, -3, 2},
        {3, 0, -1},
        {-2, 1, 0}};
    Matrix expected(expected_vals);

    auto S = matrix_algorithms::skew_symmetric(v);

    REQUIRE(S == expected);
}

// ============================================================================
// SKEW-SYMMETRIC — IS ANTISYMMETRIC (Sᵀ = −S)
// ============================================================================

TEMPLATE_TEST_CASE("skew_symmetric: S^T = -S",
                   "[skew_symmetric]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T v_vals[3][1] = {{4}, {-2}, {7}};
    Vector v(v_vals);

    auto S = matrix_algorithms::skew_symmetric(v);
    Matrix St;
    St = S.transpose_view();

    REQUIRE(S == -St);
}

// ============================================================================
// SKEW-SYMMETRIC — DIAGONAL IS ZERO
// ============================================================================

TEMPLATE_TEST_CASE("skew_symmetric: zero diagonal",
                   "[skew_symmetric]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T v_vals[3][1] = {{5}, {-1}, {3}};
    Vector v(v_vals);

    auto S = matrix_algorithms::skew_symmetric(v);

    REQUIRE(S(0, 0) == T(0));
    REQUIRE(S(1, 1) == T(0));
    REQUIRE(S(2, 2) == T(0));
}

// ============================================================================
// SKEW-SYMMETRIC — [v]× · u = v × u (cross product)
// ============================================================================

TEMPLATE_TEST_CASE("skew_symmetric: Sv = cross product",
                   "[skew_symmetric]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T v_vals[3][1] = {{1}, {2}, {3}};
    Vector v(v_vals);

    T u_vals[3][1] = {{4}, {5}, {6}};
    Vector u(u_vals);

    auto S = matrix_algorithms::skew_symmetric(v);

    // S * u via matmul
    auto Su = FusedMatrix<T, 3, 1>::matmul(S, u);

    // v × u = [2·6-3·5, 3·4-1·6, 1·5-2·4] = [-3, 6, -3]
    T cross_vals[3][1] = {{-3}, {6}, {-3}};
    Vector cross_expected(cross_vals);

    REQUIRE(Su == cross_expected);
}

// ============================================================================
// SKEW-SYMMETRIC — ZERO VECTOR
// ============================================================================

TEMPLATE_TEST_CASE("skew_symmetric: zero vector gives zero matrix",
                   "[skew_symmetric]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    Vector v(T(0));

    auto S = matrix_algorithms::skew_symmetric(v);

    Matrix zero(T(0));
    REQUIRE(S == zero);
}

// ============================================================================
// RODRIGUES — ZERO ROTATION GIVES IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("rodrigues: zero omega gives identity",
                   "[rodrigues]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    Vector omega(T(0));

    auto R = matrix_algorithms::rodrigues(omega);

    REQUIRE(R.isIdentity());
}

// ============================================================================
// RODRIGUES — 90° ROTATION AROUND Z-AXIS
// ============================================================================

TEST_CASE("rodrigues: 90deg around z-axis",
          "[rodrigues]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // ω = [0, 0, π/2], t = 1 → θ = π/2
    T pi_2 = math::acos(T(0)); // π/2
    T omega_vals[3][1] = {{0}, {0}, {pi_2}};
    Vector omega(omega_vals);

    auto R = matrix_algorithms::rodrigues(omega);

    // Expected: [cos90 -sin90 0; sin90 cos90 0; 0 0 1] = [0 -1 0; 1 0 0; 0 0 1]
    T expected_vals[3][3] = {
        {0, -1, 0},
        {1, 0, 0},
        {0, 0, 1}};
    Matrix expected(expected_vals);

    REQUIRE(R == expected);
}

// ============================================================================
// RODRIGUES — 180° ROTATION AROUND X-AXIS
// ============================================================================

TEST_CASE("rodrigues: 180deg around x-axis",
          "[rodrigues]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // ω = [π, 0, 0], t = 1 → θ = π
    T pi = math::acos(T(0)) * T(2);
    T omega_vals[3][1] = {{pi}, {0}, {0}};
    Vector omega(omega_vals);

    auto R = matrix_algorithms::rodrigues(omega);

    // Expected: [1 0 0; 0 -1 0; 0 0 -1]
    T expected_vals[3][3] = {
        {1, 0, 0},
        {0, -1, 0},
        {0, 0, -1}};
    Matrix expected(expected_vals);

    REQUIRE(R == expected);
}

// ============================================================================
// RODRIGUES — RESULT IS ORTHOGONAL (RᵀR = I)
// ============================================================================

TEST_CASE("rodrigues: R is orthogonal",
          "[rodrigues]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T omega_vals[3][1] = {{T(0.3)}, {T(-0.5)}, {T(0.7)}};
    Vector omega(omega_vals);

    auto R = matrix_algorithms::rodrigues(omega, T(2.0));

    auto RtR = Matrix::matmul(R.transpose_view(), R);
    REQUIRE(RtR.isIdentity());
}

// ============================================================================
// RODRIGUES — det(R) = +1 (proper rotation)
// ============================================================================

TEST_CASE("rodrigues: det(R) = 1",
          "[rodrigues]")
{
    using T = double;
    using Vector = FusedVector<T, 3>;

    T omega_vals[3][1] = {{T(0.3)}, {T(-0.5)}, {T(0.7)}};
    Vector omega(omega_vals);

    auto R = matrix_algorithms::rodrigues(omega, T(1.5));

    REQUIRE(matrix_algorithms::determinant(R) == Approx(T(1)));
}

// ============================================================================
// RODRIGUES — SMALL ANGLE (first-order approximation)
// ============================================================================

TEST_CASE("rodrigues: small angle approximation",
          "[rodrigues]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // Very small ω → R ≈ I + [ω]×
    T omega_vals[3][1] = {{T(1e-12)}, {T(2e-12)}, {T(3e-12)}};
    Vector omega(omega_vals);

    auto R = matrix_algorithms::rodrigues(omega);

    // Should still be orthogonal
    auto RtR = Matrix::matmul(R.transpose_view(), R);
    REQUIRE(RtR.isIdentity());

    // Should be very close to identity
    Matrix I;
    I.setIdentity();
    REQUIRE(R == I);
}

// ============================================================================
// RODRIGUES — OMEGA WITH TIME STEP
// ============================================================================

TEST_CASE("rodrigues: omega with time step",
          "[rodrigues]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // ω = [0, 0, 1] rad/s, t = π/2 → 90° rotation around z
    T pi_2 = math::acos(T(0));
    T omega_vals[3][1] = {{0}, {0}, {1}};
    Vector omega(omega_vals);

    auto R = matrix_algorithms::rodrigues(omega, pi_2);

    // Same as 90° around z
    T expected_vals[3][3] = {
        {0, -1, 0},
        {1, 0, 0},
        {0, 0, 1}};
    Matrix expected(expected_vals);

    REQUIRE(R == expected);
}

// ============================================================================
// RODRIGUES — COMPOSITION: R(ω, 2t) = R(ω, t) · R(ω, t)
// ============================================================================

TEST_CASE("rodrigues: R(omega, 2t) = R(omega, t) * R(omega, t)",
          "[rodrigues]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T omega_vals[3][1] = {{T(0.3)}, {T(-0.5)}, {T(0.7)}};
    Vector omega(omega_vals);

    T t = T(0.8);

    auto R_2t = matrix_algorithms::rodrigues(omega, T(2) * t);
    auto R_t = matrix_algorithms::rodrigues(omega, t);
    auto R_t_sq = Matrix::matmul(R_t, R_t);

    REQUIRE(R_2t == R_t_sq);
}
