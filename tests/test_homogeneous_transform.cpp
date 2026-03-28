#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "algorithms/operations/homogeneous_transform.h"
#include "algorithms/operations/inverse.h"

using Catch::Approx;

// ============================================================================
// KNOWN ANSWER — PURE TRANSLATION
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: pure translation",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // T = [I | (3, -5, 7)]
    // T⁻¹ = [I | (-3, 5, -7)]
    Matrix H;
    H.setIdentity();
    H(0, 3) = T(3);
    H(1, 3) = T(-5);
    H(2, 3) = T(7);

    auto result = matrix_algorithms::homogeneous_inverse(H);

    Matrix expected;
    expected.setIdentity();
    expected(0, 3) = T(-3);
    expected(1, 3) = T(5);
    expected(2, 3) = T(-7);

    REQUIRE(result == expected);
}

// ============================================================================
// KNOWN ANSWER — 90° ROTATION AROUND Z
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: 90 deg rotation around Z",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // R_z(90°) = [0 -1 0; 1 0 0; 0 0 1]
    // R_z⁻¹    = [0 1 0; -1 0 0; 0 0 1]
    T H_vals[4][4] = {
        {0, -1, 0, 0},
        {1, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}};
    Matrix H(H_vals);

    T expected_vals[4][4] = {
        {0, 1, 0, 0},
        {-1, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}};
    Matrix expected(expected_vals);

    auto result = matrix_algorithms::homogeneous_inverse(H);

    REQUIRE(result == expected);
}

// ============================================================================
// KNOWN ANSWER — ROTATION + TRANSLATION
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: rotation + translation",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // 90° around Z, translate (1, 2, 3)
    // H = [0 -1 0 1; 1 0 0 2; 0 0 1 3; 0 0 0 1]
    // Rᵀ = [0 1 0; -1 0 0; 0 0 1]
    // -Rᵀt = -[0*1+1*2+0*3; -1*1+0*2+0*3; 0*1+0*2+1*3] = -[2; -1; 3] = [-2; 1; -3]
    T H_vals[4][4] = {
        {0, -1, 0, 1},
        {1, 0, 0, 2},
        {0, 0, 1, 3},
        {0, 0, 0, 1}};
    Matrix H(H_vals);

    T expected_vals[4][4] = {
        {0, 1, 0, -2},
        {-1, 0, 0, 1},
        {0, 0, 1, -3},
        {0, 0, 0, 1}};
    Matrix expected(expected_vals);

    auto result = matrix_algorithms::homogeneous_inverse(H);

    REQUIRE(result == expected);
}

// ============================================================================
// IDENTITY
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: identity",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    Matrix I;
    I.setIdentity();

    auto result = matrix_algorithms::homogeneous_inverse(I);

    REQUIRE(result.isIdentity());
}

// ============================================================================
// PROPERTY: H · H⁻¹ = I
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: H * H_inv = I",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // 90° around X, translate (4, -2, 6)
    // R_x(90°) = [1 0 0; 0 0 -1; 0 1 0]
    T H_vals[4][4] = {
        {1, 0, 0, 4},
        {0, 0, -1, -2},
        {0, 1, 0, 6},
        {0, 0, 0, 1}};
    Matrix H(H_vals);

    auto H_inv = matrix_algorithms::homogeneous_inverse(H);

    auto product = Matrix::matmul(H, H_inv);

    REQUIRE(product.isIdentity());
}

// ============================================================================
// PROPERTY: H⁻¹ · H = I
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: H_inv * H = I",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // 90° around Y, translate (1, 0, -3)
    // R_y(90°) = [0 0 1; 0 1 0; -1 0 0]
    T H_vals[4][4] = {
        {0, 0, 1, 1},
        {0, 1, 0, 0},
        {-1, 0, 0, -3},
        {0, 0, 0, 1}};
    Matrix H(H_vals);

    auto H_inv = matrix_algorithms::homogeneous_inverse(H);

    auto product = Matrix::matmul(H_inv, H);

    REQUIRE(product.isIdentity());
}

// ============================================================================
// PROPERTY: (H⁻¹)⁻¹ = H
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: double inverse recovers H",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    T H_vals[4][4] = {
        {0, -1, 0, 5},
        {1, 0, 0, -3},
        {0, 0, 1, 8},
        {0, 0, 0, 1}};
    Matrix H(H_vals);

    auto H_inv = matrix_algorithms::homogeneous_inverse(H);
    auto H_double_inv = matrix_algorithms::homogeneous_inverse(H_inv);

    REQUIRE(H_double_inv == H);
}

// ============================================================================
// PROPERTY: (H1·H2)⁻¹ = H2⁻¹ · H1⁻¹
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: (H1*H2)_inv = H2_inv * H1_inv",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // H1: 90° around Z, translate (1, 2, 3)
    T H1_vals[4][4] = {
        {0, -1, 0, 1},
        {1, 0, 0, 2},
        {0, 0, 1, 3},
        {0, 0, 0, 1}};
    Matrix H1(H1_vals);

    // H2: 90° around X, translate (4, -1, 0)
    T H2_vals[4][4] = {
        {1, 0, 0, 4},
        {0, 0, -1, -1},
        {0, 1, 0, 0},
        {0, 0, 0, 1}};
    Matrix H2(H2_vals);

    auto H1H2 = Matrix::matmul(H1, H2);

    auto inv_H1H2 = matrix_algorithms::homogeneous_inverse(H1H2);
    auto inv_H1 = matrix_algorithms::homogeneous_inverse(H1);
    auto inv_H2 = matrix_algorithms::homogeneous_inverse(H2);

    auto H2inv_H1inv = Matrix::matmul(inv_H2, inv_H1);

    REQUIRE(inv_H1H2 == H2inv_H1inv);
}

// ============================================================================
// AGREES WITH GENERIC INVERSE
// ============================================================================

TEST_CASE("homogeneous_inverse: agrees with generic inverse",
          "[homogeneous_inverse]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 4, 4>;

    // 90° around Z + translation
    T H_vals[4][4] = {
        {0, -1, 0, 5},
        {1, 0, 0, -3},
        {0, 0, 1, 8},
        {0, 0, 0, 1}};
    Matrix H(H_vals);

    auto fast_inv = matrix_algorithms::homogeneous_inverse(H);
    auto generic_inv = matrix_algorithms::inverse(H);

    REQUIRE(generic_inv.has_value());
    REQUIRE(fast_inv == generic_inv.value());
}

// ============================================================================
// 180° ROTATION — R² = I, so H_inv has Rᵀ = R
// ============================================================================

TEMPLATE_TEST_CASE("homogeneous_inverse: 180 deg rotation",
                   "[homogeneous_inverse]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 4>;

    // R_z(180°) = [-1 0 0; 0 -1 0; 0 0 1], t = (2, 3, 0)
    // Rᵀ = R (symmetric), -Rᵀt = -[-2; -3; 0] = [2; 3; 0]
    T H_vals[4][4] = {
        {-1, 0, 0, 2},
        {0, -1, 0, 3},
        {0, 0, 1, 0},
        {0, 0, 0, 1}};
    Matrix H(H_vals);

    T expected_vals[4][4] = {
        {-1, 0, 0, 2},
        {0, -1, 0, 3},
        {0, 0, 1, 0},
        {0, 0, 0, 1}};
    Matrix expected(expected_vals);

    auto result = matrix_algorithms::homogeneous_inverse(H);

    REQUIRE(result == expected);
}
