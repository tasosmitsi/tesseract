#ifndef FUSED_ALGORITHMS_QR_H
#define FUSED_ALGORITHMS_QR_H

#include "config.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h" // math::sqrt, math::abs

/**
 * @file qr.h
 * @brief Householder QR decomposition for rectangular matrices.
 *
 * Decomposes A (M×N, M≥N) into Q·R where:
 *   - Q is M×M orthogonal (Qᵀ·Q = I)
 *   - R is M×N upper triangular
 *
 * Compact storage stores R in the upper triangle and Householder vectors
 * below the diagonal (with implicit leading 1), plus a tau vector of
 * scaling factors. Q() and R() extract the full matrices when needed.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * For each column j = 0 … N−1:
 *   1. Compute the norm of the sub-column A(j:M−1, j)
 *   2. Choose α = −sign(A(j,j))·‖sub-column‖ (avoids cancellation)
 *   3. Form Householder vector v: v(0) = A(j,j)−α, v(i) = A(i,j) for i>j
 *   4. Normalize: store v(i)/v(0) below diagonal, adjust τ = 2·v(0)²/(vᵀv)
 *   5. Apply reflection H = I − τ·v·vᵀ to remaining columns j+1…N−1
 *   6. Store R(j,j) = α on the diagonal
 *
 * Q is recovered by accumulating the Householder reflections in reverse order.
 *
 * Complexity: O(2MN² − 2N³/3) for the factorization.
 *             O(2M²N − 2MN²/3) additionally to form Q.
 *
 * ============================================================================
 * NOTES
 * ============================================================================
 *
 * - QR is infallible — every matrix has a valid QR decomposition.
 * - The sign convention ensures numerical stability (no catastrophic
 *   cancellation in v(0)).
 * - For least squares, Q is not needed explicitly — apply Qᵀb via the
 *   stored Householder vectors, then back-substitute with R.
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    // ========================================================================
    // QR Result
    // ========================================================================

    /**
     * @brief Result of Householder QR decomposition.
     *
     * Compact storage: R in upper triangle, normalized Householder vectors
     * (leading 1 implicit) below diagonal, tau vector of scaling factors.
     *
     * @tparam T  Scalar type.
     * @tparam M  Number of rows (M ≥ N).
     * @tparam N  Number of columns.
     */
    template <typename T, my_size_t M, my_size_t N>
    struct QRResult
    {
        FusedMatrix<T, M, N> QR; ///< Compact Householder + R storage.
        FusedVector<T, N> tau;   ///< Householder scaling factors.

        /**
         * @brief Extract the full orthogonal factor Q (M×M).
         *
         * Accumulates Householder reflections in reverse order:
         *   Q = H₀ · H₁ · … · H_{N−1}
         * where Hⱼ = I − τⱼ · vⱼ · vⱼᵀ.
         */
        FusedMatrix<T, M, M> Q() const
        {
            FusedMatrix<T, M, M> result(T(0));
            result.setIdentity();

            for (my_size_t jj = N; jj-- > 0;)
            {
                T tj = tau(jj);

                if (tj == T(0))
                    continue;

                // Apply Hⱼ to result from the left: result = Hⱼ · result
                // v = [0..0, 1, QR(jj+1,jj), ..., QR(M-1,jj)]
                for (my_size_t k = 0; k < M; ++k)
                {
                    // dot = vᵀ · result(:, k)
                    T dot = result(jj, k); // v(jj) = 1 (implicit)

                    for (my_size_t i = jj + 1; i < M; ++i)
                    {
                        dot += QR(i, jj) * result(i, k);
                    }

                    // result(:, k) -= τ · v · dot
                    result(jj, k) -= tj * dot;

                    for (my_size_t i = jj + 1; i < M; ++i)
                    {
                        result(i, k) -= tj * QR(i, jj) * dot;
                    }
                }
            }

            return result;
        }

        /**
         * @brief Extract the upper-triangular factor R (M×N).
         *
         * R(i,j) = QR(i,j) for i ≤ j, zero below diagonal.
         */
        FusedMatrix<T, M, N> R() const
        {
            FusedMatrix<T, M, N> result(T(0));

            for (my_size_t i = 0; i < M; ++i)
            {
                for (my_size_t j = i; j < N; ++j)
                {
                    result(i, j) = QR(i, j);
                }
            }

            return result;
        }

        /**
         * @brief Apply Qᵀ to a vector b without forming Q explicitly.
         *
         * Computes Qᵀ·b by applying Householder reflections in forward order.
         * Used for least squares: solve R·x = Qᵀ·b.
         *
         * @param b  Input vector of length M.
         * @return Qᵀ·b.
         */
        FusedVector<T, M> apply_Qt(const FusedVector<T, M> &b) const
        {
            FusedVector<T, M> result = b;

            for (my_size_t j = 0; j < N; ++j)
            {
                T tj = tau(j);

                if (tj == T(0))
                    continue;

                // dot = vᵀ · result
                T dot = result(j); // v(j) = 1 (implicit)

                for (my_size_t i = j + 1; i < M; ++i)
                {
                    dot += QR(i, j) * result(i);
                }

                // result -= τ · v · dot
                result(j) -= tj * dot;

                for (my_size_t i = j + 1; i < M; ++i)
                {
                    result(i) -= tj * QR(i, j) * dot;
                }
            }

            return result;
        }
    };

    // ========================================================================
    // Householder QR Decomposition
    // ========================================================================

    /**
     * @brief Compute the Householder QR decomposition of a rectangular matrix.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam M  Number of rows (deduced, M ≥ N).
     * @tparam N  Number of columns (deduced).
     * @param  A  Input matrix (M×N).
     * @return QRResult containing compact factorization with Q(), R(), and apply_Qt().
     *
     * @par Example:
     * @code
     *   FusedMatrix<double, 4, 3> A;
     *   // ... fill A ...
     *   auto qr = matrix_algorithms::qr_householder(A);
     *   auto Q = qr.Q();     // 4×4 orthogonal
     *   auto R = qr.R();     // 4×3 upper triangular
     *   // Q * R ≈ A
     *
     *   // For least squares (without forming Q):
     *   auto Qtb = qr.apply_Qt(b);  // apply Qᵀ to b
     *   // then back-substitute top N rows of Qtb with R(0:N-1, 0:N-1)
     * @endcode
     */
    template <typename T, my_size_t M, my_size_t N>
    QRResult<T, M, N> qr_householder(const FusedMatrix<T, M, N> &A)
    {
        static_assert(is_floating_point_v<T>,
                      "qr_householder requires a floating-point scalar type");
        static_assert(M >= N, "qr_householder requires M >= N");

        QRResult<T, M, N> result;
        result.QR = A;
        result.tau = FusedVector<T, N>(T(0));

        for (my_size_t j = 0; j < N; ++j)
        {
            // 1. Compute norm of sub-column QR(j:M-1, j)
            T norm_sq = T(0);

            for (my_size_t i = j; i < M; ++i)
            {
                norm_sq += result.QR(i, j) * result.QR(i, j);
            }

            T norm = math::sqrt(norm_sq);

            if (norm <= T(PRECISION_TOLERANCE))
            {
                result.tau(j) = T(0);
                continue;
            }

            // 2. Choose α = -sign(QR(j,j)) · norm
            T alpha = (result.QR(j, j) >= T(0)) ? -norm : norm;

            // 3. Form Householder vector v
            // v(j) = QR(j,j) - α,  v(i) = QR(i,j) for i > j
            T v0 = result.QR(j, j) - alpha;

            // 4. Compute τ and normalize v
            // vᵀv = v0² + Σ QR(i,j)² for i > j
            T vtv = v0 * v0;

            for (my_size_t i = j + 1; i < M; ++i)
            {
                vtv += result.QR(i, j) * result.QR(i, j);
            }

            T tj = T(2) * v0 * v0 / vtv;
            result.tau(j) = tj;

            // Normalize: store v(i)/v(0) below diagonal
            for (my_size_t i = j + 1; i < M; ++i)
            {
                result.QR(i, j) /= v0;
            }

            // 5. Apply reflection to remaining columns k = j+1..N-1
            // col_k -= τ · v_norm · (v_normᵀ · col_k)
            // v_norm(j) = 1, v_norm(i) = QR(i,j) for i > j
            for (my_size_t k = j + 1; k < N; ++k)
            {
                T dot = result.QR(j, k); // v_norm(j) = 1

                for (my_size_t i = j + 1; i < M; ++i)
                {
                    dot += result.QR(i, j) * result.QR(i, k);
                }

                result.QR(j, k) -= tj * dot;

                for (my_size_t i = j + 1; i < M; ++i)
                {
                    result.QR(i, k) -= tj * result.QR(i, j) * dot;
                }
            }

            // 6. Store R(j,j) = α
            result.QR(j, j) = alpha;
        }

        return result;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_QR_H
