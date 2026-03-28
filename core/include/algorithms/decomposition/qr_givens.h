#ifndef FUSED_ALGORITHMS_QR_GIVENS_H
#define FUSED_ALGORITHMS_QR_GIVENS_H

#include "config.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h"

/**
 * @file qr_givens.h
 * @brief Givens rotation-based QR decomposition for rectangular matrices.
 *
 * Decomposes A (M×N, M≥N) into Q·R using plane rotations that zero one
 * sub-diagonal element at a time. Each rotation affects only two rows,
 * making Givens QR ideal for:
 *   - Sparse or banded matrices (minimal fill-in)
 *   - Incremental/streaming updates (add one row at a time)
 *   - Branch-free implementation (rotation parameters computed without branches)
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * For each column j = 0 … N−1:
 *   For each row i = M−1 down to j+1:
 *     1. Compute Givens rotation (c, s) to zero R(i, j) using R(i−1, j):
 *          r = sqrt(R(i−1,j)² + R(i,j)²)
 *          c = R(i−1,j) / r
 *          s = R(i,j) / r
 *     2. Apply rotation to rows i−1 and i of R (columns j … N−1)
 *     3. Accumulate rotation into Q (columns i−1 and i)
 *
 * Complexity: O(3MN² − N³) for the factorization (roughly 50% more FLOPs
 * than Householder for dense matrices, but better for sparse/banded).
 *
 * ============================================================================
 * NOTES
 * ============================================================================
 *
 * - Givens QR is infallible — every matrix has a valid QR decomposition.
 * - Q is built explicitly by accumulating rotations (unlike Householder's
 *   compact storage). This costs O(M²N) extra work but avoids the need
 *   for a separate Q extraction step.
 * - For purely triangularizing (R only, no Q needed), the Q accumulation
 *   can be skipped — a future `qr_givens_r_only()` variant could exploit this.
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    /**
     * @brief Result of Givens QR decomposition.
     *
     * Unlike Householder QRResult which uses compact storage, Givens
     * builds Q explicitly during factorization.
     *
     * @tparam T  Scalar type.
     * @tparam M  Number of rows (M ≥ N).
     * @tparam N  Number of columns.
     */
    template <typename T, my_size_t M, my_size_t N>
    struct GivensQRResult
    {
        FusedMatrix<T, M, M> Q; ///< Orthogonal factor (M×M).
        FusedMatrix<T, M, N> R; ///< Upper-triangular factor (M×N).
    };

    /**
     * @brief Compute the QR decomposition using Givens rotations.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam M  Number of rows (deduced, M ≥ N).
     * @tparam N  Number of columns (deduced).
     * @param  A  Input matrix (M×N).
     * @return GivensQRResult containing Q (M×M) and R (M×N).
     *
     * @par Example:
     * @code
     *   FusedMatrix<double, 4, 3> A;
     *   // ... fill A ...
     *   auto qr = matrix_algorithms::qr_givens(A);
     *   // qr.Q * qr.R ≈ A
     *   // qr.Q is orthogonal: Qᵀ·Q = I
     * @endcode
     */
    template <typename T, my_size_t M, my_size_t N>
    GivensQRResult<T, M, N> qr_givens(const FusedMatrix<T, M, N> &A)
    {
        static_assert(is_floating_point_v<T>,
                      "qr_givens requires a floating-point scalar type");
        static_assert(M >= N, "qr_givens requires M >= N");

        GivensQRResult<T, M, N> result;
        result.R = A;
        result.Q = FusedMatrix<T, M, M>(T(0));
        result.Q.setIdentity();

        for (my_size_t j = 0; j < N; ++j)
        {
            // Zero elements below diagonal, bottom to top
            for (my_size_t i = M - 1; i > j; --i)
            {
                T a = result.R(i - 1, j);
                T b = result.R(i, j);

                // Skip if already zero
                if (math::abs(b) <= T(PRECISION_TOLERANCE))
                    continue;

                // Compute Givens rotation
                T r = math::sqrt(a * a + b * b);
                T c = a / r;
                T s = b / r;

                // Apply rotation to rows i-1 and i of R (columns j..N-1)
                for (my_size_t k = j; k < N; ++k)
                {
                    T r1 = result.R(i - 1, k);
                    T r2 = result.R(i, k);
                    result.R(i - 1, k) = c * r1 + s * r2;
                    result.R(i, k) = -s * r1 + c * r2;
                }

                // Accumulate rotation into Q (columns i-1 and i)
                for (my_size_t k = 0; k < M; ++k)
                {
                    T q1 = result.Q(k, i - 1);
                    T q2 = result.Q(k, i);
                    result.Q(k, i - 1) = c * q1 + s * q2;
                    result.Q(k, i) = -s * q1 + c * q2;
                }
            }
        }

        return result;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_QR_GIVENS_H
