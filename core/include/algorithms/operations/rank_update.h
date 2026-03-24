#ifndef FUSED_ALGORITHMS_RANK_UPDATE_H
#define FUSED_ALGORITHMS_RANK_UPDATE_H

#include "config.h"
#include "fused/fused_matrix.h"

/**
 * @file rank_update.h
 * @brief Symmetric rank-k update: P' = F·P·Fᵀ + Q
 *
 * Core building block for Kalman filter prediction step. Propagates a
 * covariance matrix P through a state transition F with process noise Q.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 *   P' = F · P · Fᵀ + Q
 *
 * Computed as two matrix multiplications and one addition:
 *   1. tmp = F · P          (N×N · N×N → N×N)
 *   2. P'  = tmp · Fᵀ + Q  (N×N · N×N + N×N → N×N)
 *
 * Complexity: O(2N³) for the two multiplications, O(N²) for the addition.
 *
 * @note For Kalman filters where F is sparse or structured (e.g. identity
 * plus small perturbation), specialized implementations can exploit that
 * structure. This generic version assumes dense F.
 *
 * @note The result is guaranteed symmetric if P and Q are symmetric, since
 * F·P·Fᵀ preserves symmetry and Q is symmetric by definition (covariance).
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    /**
     * @brief Compute the symmetric rank-k update P' = F·P·Fᵀ + Q.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  F  State transition matrix (N×N).
     * @param  P  Covariance matrix (N×N), symmetric positive (semi-)definite.
     * @param  Q  Process noise matrix (N×N), symmetric positive (semi-)definite.
     * @return P' = F·P·Fᵀ + Q.
     */
    template <typename T, my_size_t N>
    FusedMatrix<T, N, N> symmetric_rank_k_update(
        const FusedMatrix<T, N, N> &F,
        const FusedMatrix<T, N, N> &P,
        const FusedMatrix<T, N, N> &Q)
    {
        FusedMatrix<T, N, N> result;

        auto tmp = FusedMatrix<T, N, N>::matmul(F, P);
        result = FusedMatrix<T, N, N>::matmul(tmp, F.transpose_view()) + Q;
        return result;
    }

    /**
     * @brief Compute the symmetric rank-k update P' = F·P·Fᵀ (no noise term).
     *
     * Useful when process noise is added separately or is zero.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  F  State transition matrix (N×N).
     * @param  P  Covariance matrix (N×N), symmetric.
     * @return P' = F·P·Fᵀ.
     */
    template <typename T, my_size_t N>
    FusedMatrix<T, N, N> symmetric_rank_k_update(
        const FusedMatrix<T, N, N> &F,
        const FusedMatrix<T, N, N> &P)
    {
        FusedMatrix<T, N, N> result;

        auto tmp = FusedMatrix<T, N, N>::matmul(F, P);
        result = FusedMatrix<T, N, N>::matmul(tmp, F.transpose_view());
        return result;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_RANK_UPDATE_H
