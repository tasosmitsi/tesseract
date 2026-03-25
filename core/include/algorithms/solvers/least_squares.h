#ifndef FUSED_ALGORITHMS_LEAST_SQUARES_H
#define FUSED_ALGORITHMS_LEAST_SQUARES_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "algorithms/decomposition/qr.h"
#include "algorithms/solvers/triangular_solve.h"

/**
 * @file least_squares.h
 * @brief Least squares solve: min ‖Ax - b‖² via QR decomposition.
 *
 * For an overdetermined system A (M×N, M≥N) and vector b (M),
 * finds x (N) that minimizes the 2-norm of the residual.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 *   1. Decompose A = Q·R via qr_householder(A)
 *   2. Compute c = Qᵀ·b via apply_Qt (no explicit Q formation)
 *   3. Solve R₁·x = c₁ via back substitution
 *      where R₁ = R(0:N-1, 0:N-1) and c₁ = c(0:N-1)
 *
 * The residual norm is ‖c₂‖ where c₂ = c(N:M-1).
 *
 * Complexity: O(2MN² - 2N³/3) for QR + O(N²) for back substitution.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular — R has a zero diagonal (A is rank-deficient,
 *   least squares solution is not unique)
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Solve the least squares problem min ‖Ax - b‖² via QR.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam M  Number of rows in A (deduced, M ≥ N).
     * @tparam N  Number of columns in A (deduced).
     * @param  A  Input matrix (M×N).
     * @param  b  Right-hand side vector (M).
     * @return Expected containing the least squares solution x (N) on success,
     *         or MatrixStatus::Singular if A is rank-deficient.
     *
     * @par Example:
     * @code
     *   // Fit y = a + b*x to 4 data points
     *   FusedMatrix<double, 4, 2> A;  // [1 x0; 1 x1; 1 x2; 1 x3]
     *   FusedVector<double, 4> b;     // [y0; y1; y2; y3]
     *   auto result = matrix_algorithms::least_squares(A, b);
     *   if (result.has_value()) {
     *       auto& x = result.value();  // x(0)=intercept, x(1)=slope
     *   }
     * @endcode
     */
    template <typename T, my_size_t M, my_size_t N>
    Expected<FusedVector<T, N>, MatrixStatus> least_squares(
        const FusedMatrix<T, M, N> &A,
        const FusedVector<T, M> &b)
    {
        static_assert(is_floating_point_v<T>,
                      "least_squares requires a floating-point scalar type");
        static_assert(M >= N, "least_squares requires M >= N (overdetermined system)");

        // 1. QR decompose A
        auto qr = qr_householder(A);

        // 2. Compute c = Qᵀ·b
        auto c = qr.apply_Qt(b);

        // 3. Extract R₁ (top N×N of R) and c₁ (top N of c)
        FusedMatrix<T, N, N> R1(T(0));

        for (my_size_t i = 0; i < N; ++i)
        {
            for (my_size_t j = i; j < N; ++j)
            {
                R1(i, j) = qr.QR(i, j);
            }
        }

        FusedVector<T, N> c1(T(0));

        for (my_size_t i = 0; i < N; ++i)
        {
            c1(i) = c(i);
        }

        // 4. Back-substitute R₁·x = c₁
        return back_substitute(R1, c1);
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_LEAST_SQUARES_H
