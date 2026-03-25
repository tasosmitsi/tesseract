#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/decomposition/eigen.h"
#include "algorithms/operations/trace.h"
#include "algorithms/operations/determinant.h"

using Catch::Approx;
using matrix_traits::MatrixStatus;

// ============================================================================
// HELPER: Sort eigenvalues (bubble sort, N is small)
// ============================================================================

template <typename T, my_size_t N>
void sort_eigenvalues(FusedVector<T, N> &vals)
{
    for (my_size_t i = 0; i < N; ++i)
    {
        for (my_size_t j = i + 1; j < N; ++j)
        {
            if (vals(j) < vals(i))
            {
                T tmp = vals(i);
                vals(i) = vals(j);
                vals(j) = tmp;
            }
        }
    }
}

// ============================================================================
// 2×2 KNOWN EIGENVALUES
// ============================================================================

TEMPLATE_TEST_CASE("eigen_jacobi: 2x2 known eigenvalues",
                   "[eigen]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    // A = [2 1; 1 2] → eigenvalues 1 and 3
    T A_vals[2][2] = {
        {2, 1},
        {1, 2}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);

    REQUIRE(result.has_value());

    auto &eig = result.value();

    // Sort eigenvalues for comparison
    Vector vals = eig.eigenvalues;
    sort_eigenvalues(vals);

    REQUIRE(vals(0) == Approx(T(1)));
    REQUIRE(vals(1) == Approx(T(3)));
}

// ============================================================================
// 3×3 KNOWN EIGENVALUES
// ============================================================================

TEMPLATE_TEST_CASE("eigen_jacobi: 3x3 known eigenvalues",
                   "[eigen]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    // eigenvalues of [2 1 0; 1 3 1; 0 1 2] are 1, 2, 4
    // det(A - λI) = (2-λ)(λ-1)(λ-4)
    T A_vals[3][3] = {
        {2, 1, 0},
        {1, 3, 1},
        {0, 1, 2}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);

    REQUIRE(result.has_value());

    auto &eig = result.value();

    Vector vals = eig.eigenvalues;
    sort_eigenvalues(vals);

    REQUIRE(vals(0) == Approx(T(1)));
    REQUIRE(vals(1) == Approx(T(2)));
    REQUIRE(vals(2) == Approx(T(4)));

    // Sum of eigenvalues = trace
    REQUIRE(sum(eig.eigenvalues) == Approx(matrix_algorithms::trace(A)));
}

// ============================================================================
// DIAGONAL MATRIX — EIGENVALUES ARE DIAGONAL ENTRIES
// ============================================================================

TEMPLATE_TEST_CASE("eigen_jacobi: diagonal matrix",
                   "[eigen]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;
    using Vector = FusedVector<T, 3>;

    T A_vals[3][3] = {
        {5, 0, 0},
        {0, 2, 0},
        {0, 0, 8}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);

    REQUIRE(result.has_value());

    auto &eig = result.value();
    Vector vals = eig.eigenvalues;
    sort_eigenvalues(vals);

    REQUIRE(vals(0) == Approx(T(2)));
    REQUIRE(vals(1) == Approx(T(5)));
    REQUIRE(vals(2) == Approx(T(8)));
}

// ============================================================================
// IDENTITY — ALL EIGENVALUES = 1
// ============================================================================

TEMPLATE_TEST_CASE("eigen_jacobi: identity eigenvalues are 1",
                   "[eigen]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix I;
    I.setIdentity();

    auto result = matrix_algorithms::eigen_jacobi(I);

    REQUIRE(result.has_value());

    auto &eig = result.value();

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(eig.eigenvalues(i) == Approx(T(1)));
    }
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("eigen_jacobi: 1x1",
                   "[eigen]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    T A_vals[1][1] = {{7}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);

    REQUIRE(result.has_value());
    REQUIRE(result.value().eigenvalues(0) == Approx(T(7)));
}

// ============================================================================
// NON-SYMMETRIC — ERROR
// ============================================================================

TEMPLATE_TEST_CASE("eigen_jacobi: non-symmetric returns NotSymmetric",
                   "[eigen]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;

    T A_vals[2][2] = {
        {1, 2},
        {3, 4}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == MatrixStatus::NotSymmetric);
}

// ============================================================================
// PROPERTY: V·diag(λ)·Vᵀ = A (reconstruction)
// ============================================================================

TEST_CASE("eigen_jacobi: V * diag(lambda) * V^T = A",
          "[eigen]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {4, -2, 1},
        {-2, 3, -1},
        {1, -1, 5}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);
    REQUIRE(result.has_value());

    auto &eig = result.value();

    // Build D = diag(eigenvalues)
    Matrix D(T(0));
    for (my_size_t i = 0; i < 3; ++i)
    {
        D(i, i) = eig.eigenvalues(i);
    }

    // V * D * Vᵀ
    auto VD = Matrix::matmul(eig.eigenvectors, D);
    auto VDVT = Matrix::matmul(VD, eig.eigenvectors.transpose_view());

    REQUIRE(VDVT == A);
}

// ============================================================================
// PROPERTY: V IS ORTHOGONAL (VᵀV = I)
// ============================================================================

TEST_CASE("eigen_jacobi: V is orthogonal",
          "[eigen]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {4, -2, 1},
        {-2, 3, -1},
        {1, -1, 5}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);
    REQUIRE(result.has_value());

    auto &V = result.value().eigenvectors;
    auto VtV = Matrix::matmul(V.transpose_view(), V);

    REQUIRE(VtV.isIdentity());
}

// ============================================================================
// PROPERTY: SUM OF EIGENVALUES = TRACE
// ============================================================================

TEST_CASE("eigen_jacobi: sum of eigenvalues = trace",
          "[eigen]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 4, 4>;

    T A_vals[4][4] = {
        {4, 2, 0, 2},
        {2, 10, 3, 1},
        {0, 3, 5, 2},
        {2, 1, 2, 18}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);
    REQUIRE(result.has_value());

    REQUIRE(sum(result.value().eigenvalues) == Approx(matrix_algorithms::trace(A)));
}

// ============================================================================
// PROPERTY: PRODUCT OF EIGENVALUES = DETERMINANT
// ============================================================================

TEST_CASE("eigen_jacobi: product of eigenvalues = det",
          "[eigen]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    T A_vals[3][3] = {
        {4, -2, 1},
        {-2, 3, -1},
        {1, -1, 5}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);
    REQUIRE(result.has_value());

    T prod = T(1);
    for (my_size_t i = 0; i < 3; ++i)
    {
        prod *= result.value().eigenvalues(i);
    }

    REQUIRE(prod == Approx(matrix_algorithms::determinant(A)));
}

// ============================================================================
// PROPERTY: SPD MATRIX HAS ALL POSITIVE EIGENVALUES
// ============================================================================

TEST_CASE("eigen_jacobi: SPD matrix has positive eigenvalues",
          "[eigen]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 3, 3>;

    // SPD from Cholesky tests
    T A_vals[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);
    REQUIRE(result.has_value());

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(result.value().eigenvalues(i) > T(0));
    }
}

// ============================================================================
// NEGATIVE EIGENVALUES (indefinite matrix)
// ============================================================================

TEMPLATE_TEST_CASE("eigen_jacobi: indefinite matrix has negative eigenvalue",
                   "[eigen]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 2, 2>;
    using Vector = FusedVector<T, 2>;

    // eigenvalues: -1 and 5
    T A_vals[2][2] = {
        {2, 3},
        {3, 2}};
    Matrix A(A_vals);

    auto result = matrix_algorithms::eigen_jacobi(A);
    REQUIRE(result.has_value());

    Vector vals = result.value().eigenvalues;
    sort_eigenvalues(vals);

    REQUIRE(vals(0) == Approx(T(-1)));
    REQUIRE(vals(1) == Approx(T(5)));
}

// ============================================================================
// ZERO MATRIX — ALL EIGENVALUES = 0
// ============================================================================

TEMPLATE_TEST_CASE("eigen_jacobi: zero matrix",
                   "[eigen]", double, float)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 3, 3>;

    Matrix A(T(0));

    auto result = matrix_algorithms::eigen_jacobi(A);

    REQUIRE(result.has_value());

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(result.value().eigenvalues(i) == Approx(T(0)).margin(T(PRECISION_TOLERANCE)));
    }
}

// ============================================================================
// GENERIC PATH — 7×7
// ============================================================================

TEST_CASE("eigen_jacobi: 7x7 reconstruction and orthogonality",
          "[eigen]")
{
    using T = double;
    using Matrix = FusedMatrix<T, 7, 7>;

    // Build SPD: A = M*Mᵀ
    Matrix M(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        M(i, i) = T(i + 2);
        for (my_size_t j = 0; j < i; ++j)
        {
            M(i, j) = T(1) / T(i - j + 1);
        }
    }
    Matrix A = Matrix::matmul(M, M.transpose_view());

    auto result = matrix_algorithms::eigen_jacobi(A);
    REQUIRE(result.has_value());

    auto &eig = result.value();

    // V is orthogonal
    auto VtV = Matrix::matmul(eig.eigenvectors.transpose_view(), eig.eigenvectors);
    REQUIRE(VtV.isIdentity());

    // V * diag(λ) * Vᵀ = A
    Matrix D(T(0));
    for (my_size_t i = 0; i < 7; ++i)
    {
        D(i, i) = eig.eigenvalues(i);
    }

    auto VD = Matrix::matmul(eig.eigenvectors, D);
    auto VDVT = Matrix::matmul(VD, eig.eigenvectors.transpose_view());

    REQUIRE(VDVT == A);

    // Sum of eigenvalues = trace
    REQUIRE(sum(eig.eigenvalues) == Approx(matrix_algorithms::trace(A)));

    // All eigenvalues positive (SPD)
    for (my_size_t i = 0; i < 7; ++i)
    {
        REQUIRE(eig.eigenvalues(i) > T(0));
    }
}
