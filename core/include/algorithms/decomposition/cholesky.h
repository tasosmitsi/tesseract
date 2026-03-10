#ifndef FUSED_ALGORITHMS_CHOLESKY_H
#define FUSED_ALGORITHMS_CHOLESKY_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "math/math_utils.h" // math::sqrt
#include "simple_type_traits.h" // for move

/**
 * @file cholesky.h
 * @brief Cholesky decomposition for symmetric positive-definite matrices.
 *
 * Computes the lower-triangular factor L such that A = L · Lᵀ.
 * Returns the result wrapped in Expected so that callers can
 * distinguish success from well-defined failure modes (not symmetric,
 * not positive definite) without exceptions or dynamic allocation.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * For a symmetric positive-definite matrix A of size N×N, the Cholesky
 * decomposition computes a lower-triangular matrix L such that A = LLᵀ.
 * @code
 * For each row i = 0 … N−1:
 *   For each column j = 0 … i:
 *     If i == j (diagonal):
 *       L(i,i) = sqrt( A(i,i) − Σ_{k=0}^{j-1} L(i,k)² )
 *     Else (below diagonal):
 *       L(i,j) = ( A(i,j) − Σ_{k=0}^{j-1} L(i,k)·L(j,k) ) / L(j,j)
 * @endcode
 * Complexity: O(N³/3) multiplications, O(N) square roots.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::NotSymmetric        — input fails isSymmetric() check
 * - MatrixStatus::NotPositiveDefinite — a diagonal element ≤ PRECISION_TOLERANCE
 *                                       during factorization
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Compute the Cholesky decomposition of a symmetric positive-definite matrix.
     *
     * Decomposes @p A into a lower-triangular matrix L such that A = L · Lᵀ.
     *
     * @tparam MatrixType A square FusedMatrix type exposing:
     *         - `isSymmetric()` — runtime symmetry check
     *         - `getDim(i)` — dimension size along axis i
     *         - `operator()(i, j)` — element access
     *         - `value_type` — scalar type (float, double)
     *
     * @param A Symmetric positive-definite input matrix.
     * @return Expected containing the lower-triangular factor L on success,
     *         or MatrixStatus::NotSymmetric / MatrixStatus::NotPositiveDefinite on failure.
     *
     * @par Example:
     * @code
     *   FusedMatrix<double, 3, 3> A;
     *   // ... fill A as SPD ...
     *   auto result = matrix_algorithms::cholesky(A);
     *   if (!result) {
     *       // handle result.error()
     *       return;
     *   }
     *   auto& L = *result;
     *   // L * L^T ≈ A
     * @endcode
     */
    template <typename MatrixType>
    Expected<MatrixType, MatrixStatus> cholesky(const MatrixType &A)
    {
        if (!A.isSymmetric())
        {
            return Unexpected{MatrixStatus::NotSymmetric};
        }

        MatrixType L(0);

        for (my_size_t i = 0; i < A.getDim(0); ++i)
        {
            for (my_size_t j = 0; j <= i; ++j)
            {
                typename MatrixType::value_type sum = 0;

                for (my_size_t k = 0; k < j; ++k)
                {
                    sum += L(i, k) * L(j, k);
                }

                if (i == j)
                {
                    typename MatrixType::value_type diag = A(i, i) - sum;

                    if (diag <= typename MatrixType::value_type(PRECISION_TOLERANCE))
                    {
                        return Unexpected{MatrixStatus::NotPositiveDefinite};
                    }

                    L(i, j) = math::sqrt(diag);
                }
                else
                {
                    L(i, j) = (A(i, j) - sum) / L(j, j);
                }
            }
        }

        return move(L);
    }

    /**
     * @brief Cholesky decomposition — abort on failure.
     *
     * Convenience wrapper for contexts where failure is unrecoverable
     * (offline computation, test code). Calls MyErrorHandler::error()
     * if the decomposition fails.
     *
     * @tparam MatrixType Same requirements as cholesky().
     * @param A Symmetric positive-definite input matrix.
     * @return The lower-triangular factor L.
     */
    template <typename MatrixType>
    MatrixType cholesky_or_die(const MatrixType &A)
    {
        auto result = cholesky(A);

        if (!result)
        {
            MyErrorHandler::error("cholesky decomposition failed");
        }

        return move(*result);
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_CHOLESKY_H
