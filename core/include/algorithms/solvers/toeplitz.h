#ifndef FUSED_ALGORITHMS_TOEPLITZ_H
#define FUSED_ALGORITHMS_TOEPLITZ_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h" // math::abs

/**
 * @file toeplitz.h
 * @brief Levinson-Durbin algorithm: O(N²) solver for symmetric Toeplitz systems.
 *
 * A symmetric Toeplitz matrix is fully defined by its first row:
 *   T(i,j) = r(|i−j|)
 *
 *   [ r(0) r(1) r(2) ... r(N-1) ]
 *   [ r(1) r(0) r(1) ... r(N-2) ]
 *   [ r(2) r(1) r(0) ... r(N-3) ]
 *   [  ...  ...  ...  ...  ...   ]
 *   [r(N-1)r(N-2) ...     r(0)  ]
 *
 * The Levinson-Durbin recursion solves Tx = b in O(N²) by exploiting
 * the Toeplitz structure. It also produces the reflection coefficients
 * (partial correlation coefficients) as a byproduct.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * The algorithm proceeds in two stages:
 *
 * Stage 1 — Levinson recursion (solve T·a = −r for prediction coefficients):
 *   Initialize: a(0) = −r(1)/r(0), err = r(0)·(1 − a(0)²)
 *   For m = 1 … N−2:
 *     λ = −(r(m+1) + Σ_{k=0}^{m-1} a(k)·r(m−k)) / err
 *     Update a(k) using λ and reversed previous a
 *     err *= (1 − λ²)
 *
 * Stage 2 — Solve Tx = b using the Levinson solution:
 *   Uses the forward and backward vectors from Stage 1 to
 *   iteratively build the solution.
 *
 * Complexity: O(N²) multiply-adds, O(N) storage.
 *
 * ============================================================================
 * USE CASES
 * ============================================================================
 *
 * - Autocorrelation-based filtering (Wiener filter, linear prediction)
 * - Autoregressive (AR) model estimation
 * - Spectral estimation (Burg's method)
 * - Any system where the matrix has constant diagonals
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular     — r(0) ≈ 0 (zero autocorrelation)
 * - MatrixStatus::NotConverged — prediction error drops to zero or negative
 *   (matrix is not positive definite)
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Solve a symmetric Toeplitz system Tx = b using Levinson-Durbin.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  System size (deduced).
     * @param  r  First row of the Toeplitz matrix (N elements). r(0) is the diagonal.
     * @param  b  Right-hand side vector (N).
     * @return Expected containing solution x on success,
     *         or MatrixStatus error on failure.
     *
     * @par Example:
     * @code
     *   // Solve [4 2 1; 2 4 2; 1 2 4] x = b
     *   FusedVector<double, 3> r, b;
     *   r(0) = 4; r(1) = 2; r(2) = 1;  // first row defines entire matrix
     *   // ... fill b ...
     *   auto result = matrix_algorithms::levinson_durbin(r, b);
     * @endcode
     */
    template <typename T, my_size_t N>
    Expected<FusedVector<T, N>, MatrixStatus> levinson_durbin(
        const FusedVector<T, N> &r,
        const FusedVector<T, N> &b)
    {
        static_assert(is_floating_point_v<T>,
                      "levinson_durbin requires a floating-point scalar type");

        // Check r(0) != 0
        if (math::abs(r(0)) <= T(PRECISION_TOLERANCE))
        {
            return Unexpected{MatrixStatus::Singular};
        }

        // N = 1: trivial case
        if constexpr (N == 1)
        {
            FusedVector<T, 1> x(T(0));
            x(0) = b(0) / r(0);
            return move(x);
        }
        else
        {
            // Forward vector (prediction coefficients)
            FusedVector<T, N> f(T(0));
            // Backward vector
            FusedVector<T, N> f_prev(T(0));
            // Solution vector
            FusedVector<T, N> x(T(0));
            FusedVector<T, N> x_prev(T(0));

            // Initialize order 1
            T err = r(0);
            x(0) = b(0) / err;

            f(0) = -r(1) / r(0);
            err = r(0) * (T(1) - f(0) * f(0));

            if (err <= T(PRECISION_TOLERANCE))
            {
                return Unexpected{MatrixStatus::NotConverged};
            }

            // Update x for order 1
            // x = x_prev + f(0) * reversed(x_prev) + b(1)/err correction
            {
                T x0_prev = x(0);
                T delta = b(1) - r(1) * x0_prev;
                x(0) = x0_prev + (delta / err) * f(0);
                x(1) = delta / err;
            }

            // Iterate for orders 2 … N−1
            for (my_size_t m = 1; m + 1 < N; ++m)
            {
                // Save previous forward vector
                for (my_size_t k = 0; k <= m - 1; ++k)
                {
                    f_prev(k) = f(k);
                }

                // Compute reflection coefficient λ
                T lambda_num = r(m + 1);
                for (my_size_t k = 0; k < m; ++k)
                {
                    lambda_num += f_prev(k) * r(m - k);
                }
                T lambda = -lambda_num / err;

                // Update forward vector: f(k) = f_prev(k) + λ·f_prev(m−1−k)
                for (my_size_t k = 0; k < m; ++k)
                {
                    f(k) = f_prev(k) + lambda * f_prev(m - 1 - k);
                }
                f(m) = lambda;

                // Update error
                err *= (T(1) - lambda * lambda);

                if (err <= T(PRECISION_TOLERANCE))
                {
                    return Unexpected{MatrixStatus::NotConverged};
                }

                // Save previous solution
                for (my_size_t k = 0; k <= m; ++k)
                {
                    x_prev(k) = x(k);
                }

                // Compute delta for the new equation
                T delta = b(m + 1);
                for (my_size_t k = 0; k <= m; ++k)
                {
                    delta -= r(m + 1 - k) * x_prev(k);
                }

                T correction = delta / err;

                // Update solution: x(k) = x_prev(k) + correction·f_reversed
                for (my_size_t k = 0; k <= m; ++k)
                {
                    x(k) = x_prev(k) + correction * f(m - k);
                }
                x(m + 1) = correction;
            }

            return move(x);
        }
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_TOEPLITZ_H
