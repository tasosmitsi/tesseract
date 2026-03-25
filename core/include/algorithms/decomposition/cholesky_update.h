#ifndef FUSED_ALGORITHMS_CHOLESKY_UPDATE_H
#define FUSED_ALGORITHMS_CHOLESKY_UPDATE_H

#include "config.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h" // math::sqrt

/**
 * @file cholesky_update.h
 * @brief Rank-1 Cholesky update: given L where LLᵀ = A, compute L' where L'L'ᵀ = A + vvᵀ.
 *
 * Updates an existing Cholesky factor in O(N²) without recomputing the full
 * O(N³) decomposition. Essential for streaming/online covariance updates,
 * recursive least squares, and square-root Kalman filters.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * Uses Givens rotations to incorporate the rank-1 perturbation:
 *
 *   p = v  (work vector, modified in place)
 *   For k = 0 … N−1:
 *     r = sqrt(L(k,k)² + p(k)²)
 *     c = L(k,k) / r
 *     s = p(k) / r
 *     L'(k,k) = r
 *     For i = k+1 … N−1:
 *       tmp    = L(i,k)
 *       L'(i,k) = c · tmp + s · p(i)
 *       p(i)    = c · p(i) - s · tmp
 *
 * Complexity: O(N²) multiply-adds, O(N) square roots.
 *
 * @note Only the update (A + vvᵀ) is provided. The downdate (A - vvᵀ) requires
 * hyperbolic rotations and can fail if the result is not positive definite.
 * Downdate may be added in a future revision.
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    /**
     * @brief Rank-1 Cholesky update: compute L' where L'L'ᵀ = LLᵀ + vvᵀ.
     *
     * Given a lower-triangular Cholesky factor L and a vector v, computes the
     * updated factor L' in O(N²) via Givens rotations. The input L, v is not modified.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix/vector dimension (deduced).
     * @param  L  Lower-triangular Cholesky factor (N×N).
     * @param  v  Update vector (N).
     * @return Updated Cholesky factor L' such that L'L'ᵀ = LLᵀ + vvᵀ.
     */
    template <typename T, my_size_t N>
    FusedMatrix<T, N, N> cholesky_rank1_update(
        const FusedMatrix<T, N, N> &L,
        const FusedVector<T, N> &v)
    {
        static_assert(is_floating_point_v<T>,
                      "cholesky_rank1_update requires a floating-point scalar type");

        FusedMatrix<T, N, N> Lp = L; // work on a copy
        FusedVector<T, N> p = v;     // work vector

        for (my_size_t k = 0; k < N; ++k)
        {
            T r = math::sqrt(Lp(k, k) * Lp(k, k) + p(k) * p(k));
            T c = Lp(k, k) / r;
            T s = p(k) / r;

            Lp(k, k) = r;

            for (my_size_t i = k + 1; i < N; ++i)
            {
                T tmp = Lp(i, k);
                Lp(i, k) = c * tmp + s * p(i);
                p(i) = c * p(i) - s * tmp;
            }
        }

        return Lp;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_CHOLESKY_UPDATE_H
