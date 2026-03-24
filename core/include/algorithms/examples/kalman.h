#ifndef FUSED_ALGORITHMS_KALMAN_H
#define FUSED_ALGORITHMS_KALMAN_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "fused/fused_matrix.h"
#include "algorithms/operations/inverse.h"

/**
 * @file kalman.h
 * @brief Kalman filter building blocks: gain computation and covariance update.
 *
 * Provides the core update-step operations for a discrete-time Kalman filter.
 * The prediction step uses symmetric_rank_k_update (see rank_update.h).
 *
 * Template parameters:
 *   - N: state dimension
 *   - M: measurement dimension
 *
 * ============================================================================
 * KALMAN GAIN (2b)
 * ============================================================================
 *
 *   S = H·P·Hᵀ + R          (innovation covariance, M×M)
 *   K = P·Hᵀ · S⁻¹          (Kalman gain, N×M)
 *
 * Uses inverse(S) rather than solve to get the full gain matrix.
 * S is typically small (M = 1–3 for most sensor fusion), so the
 * inverse is cheap and straightforward.
 *
 * ============================================================================
 * JOSEPH FORM UPDATE (2c)
 * ============================================================================
 *
 *   P' = (I - K·H) · P · (I - K·H)ᵀ + K·R·Kᵀ
 *
 * Numerically more stable than the standard P' = P - K·S·Kᵀ form.
 * Guarantees symmetry and positive semi-definiteness of the result,
 * even with floating-point rounding.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular — innovation covariance S is not invertible
 *   (degenerate measurement noise or numerical issues)
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Compute the Kalman gain K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  State dimension (deduced).
     * @tparam M  Measurement dimension (deduced).
     * @param  P  State covariance (N×N), symmetric positive definite.
     * @param  H  Observation matrix (M×N).
     * @param  R  Measurement noise covariance (M×M), symmetric positive definite.
     * @return Expected containing the Kalman gain K (N×M) on success,
     *         or MatrixStatus::Singular if the innovation covariance is not invertible.
     */
    template <typename T, my_size_t N, my_size_t M>
    Expected<FusedMatrix<T, N, M>, MatrixStatus> kalman_gain(
        const FusedMatrix<T, N, N> &P,
        const FusedMatrix<T, M, N> &H,
        const FusedMatrix<T, M, M> &R)
    {
        static_assert(is_floating_point_v<T>,
                      "kalman_gain requires a floating-point scalar type");

        // S = H·P·Hᵀ + R  (M×M)
        auto HP = FusedMatrix<T, M, N>::matmul(H, P);

        FusedMatrix<T, M, M> S;
        S = FusedMatrix<T, M, M>::matmul(HP, H.transpose_view()) + R;

        // S⁻¹
        auto S_inv_result = inverse(S);

        if (!S_inv_result.has_value())
        {
            return Unexpected{S_inv_result.error()};
        }

        auto &S_inv = S_inv_result.value();

        // K = P·Hᵀ·S⁻¹  (N×M)
        auto PHt = FusedMatrix<T, N, M>::matmul(P, H.transpose_view());
        auto K = FusedMatrix<T, N, M>::matmul(PHt, S_inv);

        return move(K);
    }

    /**
     * @brief Joseph form covariance update: P' = (I-K·H)·P·(I-K·H)ᵀ + K·R·Kᵀ.
     *
     * Numerically stable covariance update that guarantees symmetry and
     * positive semi-definiteness of the result.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  State dimension (deduced).
     * @tparam M  Measurement dimension (deduced).
     * @param  K  Kalman gain (N×M).
     * @param  H  Observation matrix (M×N).
     * @param  P  Prior state covariance (N×N).
     * @param  R  Measurement noise covariance (M×M).
     * @return Updated covariance P' (N×N).
     */
    template <typename T, my_size_t N, my_size_t M>
    FusedMatrix<T, N, N> joseph_update(
        const FusedMatrix<T, N, M> &K,
        const FusedMatrix<T, M, N> &H,
        const FusedMatrix<T, N, N> &P,
        const FusedMatrix<T, M, M> &R)
    {
        static_assert(is_floating_point_v<T>,
                      "joseph_update requires a floating-point scalar type");

        // IKH = I - K·H  (N×N)
        FusedMatrix<T, N, N> I;
        I.setIdentity();

        auto KH = FusedMatrix<T, N, N>::matmul(K, H);
        FusedMatrix<T, N, N> IKH;
        IKH = I - KH;

        // (I-K·H)·P·(I-K·H)ᵀ
        auto tmp = FusedMatrix<T, N, N>::matmul(IKH, P);
        auto term1 = FusedMatrix<T, N, N>::matmul(tmp, IKH.transpose_view());

        // K·R·Kᵀ
        auto KR = FusedMatrix<T, N, M>::matmul(K, R);
        auto term2 = FusedMatrix<T, N, N>::matmul(KR, K.transpose_view());

        FusedMatrix<T, N, N> result(T(0));
        result = term1 + term2;
        return result;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_KALMAN_H
