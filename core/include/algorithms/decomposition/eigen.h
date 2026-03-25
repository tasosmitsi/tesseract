#ifndef FUSED_ALGORITHMS_EIGEN_JACOBI_H
#define FUSED_ALGORITHMS_EIGEN_JACOBI_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h" // math::sqrt, math::abs

/**
 * @file eigen.h
 * @brief Jacobi eigenvalue decomposition for small symmetric matrices.
 *
 * Computes all eigenvalues and eigenvectors of a symmetric matrix A by
 * iteratively applying Givens rotations to zero the largest off-diagonal
 * element until convergence. The result is A = V·D·Vᵀ where:
 *   - D is diagonal (eigenvalues)
 *   - V is orthogonal (eigenvectors as columns)
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * Repeat until off-diagonal norm < tolerance:
 *   1. Find the largest off-diagonal element |A(p,q)|
 *   2. Compute Givens rotation angle θ to zero A(p,q):
 *        If A(p,p) == A(q,q): θ = π/4
 *        Else: τ = (A(p,p) − A(q,q)) / (2·A(p,q))
 *        Compute t = sign(τ) / (|τ| + √(1+τ²))  (smaller root for stability)
 *        c = 1/√(1+t²), s = t·c
 *   3. Apply rotation: A' = Jᵀ·A·J (only affects rows/cols p and q)
 *   4. Accumulate: V = V·J
 *
 * Complexity: O(N²) per sweep, typically 5–10 sweeps for convergence.
 * Total: O(5N²) to O(10N²) for small matrices.
 *
 * ============================================================================
 * NOTES
 * ============================================================================
 *
 * - Only for symmetric matrices. Returns NotSymmetric otherwise.
 * - Convergence is guaranteed for symmetric matrices (classical Jacobi).
 * - Best suited for small matrices (N ≤ ~20). For larger matrices,
 *   tridiagonalization + QR iteration (not implemented) is preferred.
 * - Eigenvalues are returned unsorted. Caller can sort if needed.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::NotSymmetric — input fails isSymmetric() check
 * - MatrixStatus::NotConverged — max iterations exceeded
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Result of eigenvalue decomposition.
     *
     * @tparam T  Scalar type.
     * @tparam N  Matrix dimension.
     */
    template <typename T, my_size_t N>
    struct EigenResult
    {
        FusedVector<T, N> eigenvalues;     ///< Diagonal of D (unsorted).
        FusedMatrix<T, N, N> eigenvectors; ///< Columns are eigenvectors (orthogonal).
    };

    /**
     * @brief Compute eigenvalues and eigenvectors of a symmetric matrix via Jacobi.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A          Symmetric input matrix (N×N).
     * @param  max_iters  Maximum number of sweeps (default 100).
     * @param  tol        Convergence tolerance for off-diagonal norm (default PRECISION_TOLERANCE).
     * @return Expected containing EigenResult on success,
     *         or MatrixStatus error on failure.
     *
     * @par Example:
     * @code
     *   FusedMatrix<double, 3, 3> A;
     *   // ... fill A (symmetric) ...
     *   auto result = matrix_algorithms::eigen_jacobi(A);
     *   if (result.has_value()) {
     *       auto& eig = result.value();
     *       // eig.eigenvalues(i) — the i-th eigenvalue
     *       // eig.eigenvectors column i — the i-th eigenvector
     *       // V * diag(λ) * Vᵀ ≈ A
     *   }
     * @endcode
     */
    template <typename T, my_size_t N>
    Expected<EigenResult<T, N>, MatrixStatus> eigen_jacobi(
        const FusedMatrix<T, N, N> &A,
        my_size_t max_iters = 100,
        T tol = T(PRECISION_TOLERANCE))
    {
        static_assert(is_floating_point_v<T>,
                      "eigen_jacobi requires a floating-point scalar type");

        if (!A.isSymmetric())
        {
            return Unexpected{MatrixStatus::NotSymmetric};
        }

        // Work matrix (will be diagonalized in place)
        FusedMatrix<T, N, N> W = A;

        // Eigenvector accumulator (starts as identity)
        FusedMatrix<T, N, N> V(T(0));
        V.setIdentity();

        for (my_size_t iter = 0; iter < max_iters; ++iter)
        {
            // 1. Find largest off-diagonal element
            T max_off = T(0);
            my_size_t p = 0, q = 1;

            for (my_size_t i = 0; i < N; ++i)
            {
                for (my_size_t j = i + 1; j < N; ++j)
                {
                    T val = math::abs(W(i, j));

                    if (val > max_off)
                    {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if (max_off <= tol)
            {
                EigenResult<T, N> result;

                for (my_size_t i = 0; i < N; ++i)
                {
                    result.eigenvalues(i) = W(i, i);
                }

                result.eigenvectors = V;
                return move(result);
            }

            // 2. Compute Givens rotation to zero W(p,q)
            T c, s;

            T diff = W(p, p) - W(q, q);

            if (math::abs(diff) <= tol)
            {
                // A(p,p) ≈ A(q,q) → θ = π/4
                T inv_sqrt2 = T(1) / math::sqrt(T(2));
                c = inv_sqrt2;
                s = (W(p, q) >= T(0)) ? inv_sqrt2 : -inv_sqrt2;
            }
            else
            {
                // τ = (W(p,p) - W(q,q)) / (2·W(p,q))
                T tau = diff / (T(2) * W(p, q));
                // Choose smaller root for numerical stability
                T t;

                if (tau >= T(0))
                {
                    t = T(1) / (tau + math::sqrt(T(1) + tau * tau));
                }
                else
                {
                    t = T(-1) / (-tau + math::sqrt(T(1) + tau * tau));
                }

                c = T(1) / math::sqrt(T(1) + t * t);
                s = t * c;
            }

            // 3. Apply rotation to W: W' = Jᵀ·W·J
            //    Only rows/columns p and q are affected.

            // Update columns (right multiply by J)
            for (my_size_t i = 0; i < N; ++i)
            {
                if (i == p || i == q)
                    continue;

                T wip = W(i, p);
                T wiq = W(i, q);
                W(i, p) = c * wip + s * wiq;
                W(i, q) = -s * wip + c * wiq;
                W(p, i) = W(i, p); // maintain symmetry
                W(q, i) = W(i, q);
            }

            // Update 2×2 block [pp, pq; qp, qq]
            T wpp = W(p, p);
            T wqq = W(q, q);
            T wpq = W(p, q);

            W(p, p) = c * c * wpp + T(2) * c * s * wpq + s * s * wqq;
            W(q, q) = s * s * wpp - T(2) * c * s * wpq + c * c * wqq;
            W(p, q) = T(0); // this is the element we're zeroing
            W(q, p) = T(0);

            // 4. Accumulate eigenvectors: V = V·J
            for (my_size_t i = 0; i < N; ++i)
            {
                T vip = V(i, p);
                T viq = V(i, q);
                V(i, p) = c * vip + s * viq;
                V(i, q) = -s * vip + c * viq;
            }
        }

        // If we get here, we didn't converge
        return Unexpected{MatrixStatus::NotConverged};
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_EIGEN_JACOBI_H
