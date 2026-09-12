#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/operations/cross_product.h"
#include "algorithms/operations/determinant.h"
#include "algorithms/operations/skew_symmetric.h"

using Catch::Approx;

// ============================================================================
// KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("cross: known answer",
                   "[cross]", double, float, int)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{1}, {2}, {3}};
    T b_vals[3][1] = {{4}, {5}, {6}};
    Vector a(a_vals), b(b_vals);

    // a × b = [2·6-3·5, 3·4-1·6, 1·5-2·4] = [-3, 6, -3]
    T expected_vals[3][1] = {{-3}, {6}, {-3}};
    Vector expected(expected_vals);

    REQUIRE(matrix_algorithms::cross(a, b) == expected);
}

// ============================================================================
// BASIS VECTORS: x̂ × ŷ = ẑ, ŷ × ẑ = x̂, ẑ × x̂ = ŷ
// ============================================================================

TEMPLATE_TEST_CASE("cross: basis vectors",
                   "[cross]", double, float, int)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T x_vals[3][1] = {{1}, {0}, {0}};
    T y_vals[3][1] = {{0}, {1}, {0}};
    T z_vals[3][1] = {{0}, {0}, {1}};
    Vector x(x_vals), y(y_vals), z(z_vals);

    REQUIRE(matrix_algorithms::cross(x, y) == z);
    REQUIRE(matrix_algorithms::cross(y, z) == x);
    REQUIRE(matrix_algorithms::cross(z, x) == y);
}

// ============================================================================
// ANTICOMMUTATIVE: a × b = −(b × a)
// ============================================================================

TEMPLATE_TEST_CASE("cross: anticommutative",
                   "[cross]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{1}, {2}, {3}};
    T b_vals[3][1] = {{4}, {5}, {6}};
    Vector a(a_vals), b(b_vals);

    auto axb = matrix_algorithms::cross(a, b);
    auto bxa = matrix_algorithms::cross(b, a);

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(axb(i) + bxa(i) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
    }
}

// ============================================================================
// SELF CROSS IS ZERO: a × a = 0
// ============================================================================

TEMPLATE_TEST_CASE("cross: self cross is zero",
                   "[cross]", double, float, int)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{3}, {-1}, {4}};
    Vector a(a_vals);

    auto result = matrix_algorithms::cross(a, a);

    Vector zero(T(0));
    REQUIRE(result == zero);
}

// ============================================================================
// BILINEAR: (αa) × b = α(a × b)
// ============================================================================

TEMPLATE_TEST_CASE("cross: bilinear scalar multiplication",
                   "[cross]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{2}, {-3}, {1}};
    T b_vals[3][1] = {{-1}, {4}, {2}};
    Vector a(a_vals), b(b_vals);

    T alpha = T(3);

    // (αa) × b
    Vector alpha_a;
    alpha_a = a * alpha;
    auto lhs = matrix_algorithms::cross(alpha_a, b);

    // α(a × b)
    auto axb = matrix_algorithms::cross(a, b);
    Vector rhs;
    rhs = axb * alpha;

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(lhs(i) == Approx(rhs(i)));
    }
}

// ============================================================================
// ORTHOGONALITY: (a × b) · a = 0, (a × b) · b = 0
// ============================================================================

TEMPLATE_TEST_CASE("cross: result is orthogonal to both inputs",
                   "[cross]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{4}, {-2}, {7}};
    T b_vals[3][1] = {{-1}, {3}, {5}};
    Vector a(a_vals), b(b_vals);

    auto c = matrix_algorithms::cross(a, b);

    // c · a
    T dot_ca = c(0) * a(0) + c(1) * a(1) + c(2) * a(2);
    // c · b
    T dot_cb = c(0) * b(0) + c(1) * b(1) + c(2) * b(2);

    REQUIRE(dot_ca == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
    REQUIRE(dot_cb == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
}

// ============================================================================
// AGREES WITH SKEW_SYMMETRIC: a × b = [a]× · b
// ============================================================================

TEMPLATE_TEST_CASE("cross: agrees with skew_symmetric",
                   "[cross]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{4}, {-2}, {7}};
    T b_vals[3][1] = {{-1}, {3}, {5}};
    Vector a(a_vals), b(b_vals);

    auto S = matrix_algorithms::skew_symmetric(a);
    auto Sb = FusedMatrix<T, 3, 1>::matmul(S, b);

    auto cross_result = matrix_algorithms::cross(a, b);

    REQUIRE(cross_result == Sb);
}

// ============================================================================
// SCALAR TRIPLE PRODUCT: a · (b × c) = det([a b c]ᵀ)
// ============================================================================

TEST_CASE("cross: scalar triple product equals determinant",
          "[cross]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{1}, {2}, {3}};
    T b_vals[3][1] = {{4}, {5}, {6}};
    T c_vals[3][1] = {{7}, {8}, {0}};
    Vector a(a_vals), b(b_vals), c(c_vals);

    auto bxc = matrix_algorithms::cross(b, c);

    // aᵀ · (b x c)
    auto triple_mat = FusedMatrix<T, 1, 1>::matmul(a.transpose_view(), bxc);
    T triple = triple_mat(0, 0);

    // det([a b c]ᵀ) — rows are a, b, c
    T A_vals[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 0}};
    Matrix A(A_vals);

    REQUIRE(triple == Approx(matrix_algorithms::determinant(A)));
}

// ============================================================================
// ZERO VECTOR: a × 0 = 0
// ============================================================================

TEMPLATE_TEST_CASE("cross: cross with zero vector is zero",
                   "[cross][edge]", double, float, int)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{3}, {-1}, {4}};
    Vector a(a_vals);
    Vector zero(T(0));

    REQUIRE(matrix_algorithms::cross(a, zero) == zero);
    REQUIRE(matrix_algorithms::cross(zero, a) == zero);
}

// ============================================================================
// PARALLEL VECTORS: a × (αa) = 0
// ============================================================================

TEMPLATE_TEST_CASE("cross: parallel vectors give zero",
                   "[cross]", double, float)
{
    using T = TestType;
    using Vector = FusedVector<T, 3>;

    T a_vals[3][1] = {{2}, {-3}, {1}};
    Vector a(a_vals);

    // αa
    Vector scaled(T(0));
    scaled = a * T(5);

    auto result = matrix_algorithms::cross(a, scaled);

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(result(i) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
    }
}
