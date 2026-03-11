#ifndef FUSED_ALGORITHMS_CHOLESKY_SOLVE_H
#define FUSED_ALGORITHMS_CHOLESKY_SOLVE_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "simple_type_traits.h"
#include "algorithms/decomposition/cholesky.h"
#include "algorithms/solvers/triangular_solve.h"

/**
 * @file cholesky_solve.h
 * @brief Solve Ax = b for symmetric positive-definite A via Cholesky decomposition.
 *
 * Chains cholesky() → forward_substitute() → transposed back substitution
 * to solve Ax = b without forming the inverse. Zero dynamic allocation.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 * Cholesky solve (Ax = b, A symmetric positive-definite):
 * @code
 *   1. Decompose A = LLᵀ via cholesky(A)
 *   2. Solve Ly = b via forward substitution
 *   3. Solve Lᵀx = y via back substitution (accessing L transposed in-place)
 * @endcode
 * Complexity: O(N³/3 + N²) — dominated by Cholesky for large N.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * Forwards all errors from cholesky():
 * - MatrixStatus::NotSymmetric        — input fails isSymmetric() check
 * - MatrixStatus::NotPositiveDefinite — diagonal ≤ tol during factorization
 *
 * From forward_substitute():
 * - MatrixStatus::Singular            — zero diagonal (should not occur if
 *                                       cholesky succeeded)
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Solve Ax = b for symmetric positive-definite A via Cholesky decomposition.
     *
     * Decomposes A = LLᵀ, then solves Ly = b (forward substitution) followed
     * by Lᵀx = y (back substitution using L transposed).
     *
     * The internal back substitution step accesses L(k,i) directly (i.e. Lᵀ(i,k))
     * to avoid creating a separate transpose. Singular checks are skipped for this
     * step since cholesky() already guarantees a valid L with positive diagonal.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix/vector dimension (deduced).
     * @param  A  Symmetric positive-definite matrix (N×N).
     * @param  b  Right-hand side vector (N).
     * @return Expected containing solution x on success,
     *         or MatrixStatus error forwarded from cholesky/substitution on failure.
     *
     * @par Example:
     * @code
     *   FusedMatrix<double, 3, 3> A;
     *   FusedVector<double, 3> b;
     *   // ... fill A (SPD) and b ...
     *   auto result = matrix_algorithms::cholesky_solve(A, b);
     *   if (!result.has_value()) {
     *       // handle result.error()
     *       return;
     *   }
     *   auto& x = result.value();
     *   // A * x ≈ b
     * @endcode
     */
    template <typename T, my_size_t N>
    Expected<FusedVector<T, N>, MatrixStatus> cholesky_solve(
        const FusedMatrix<T, N, N> &A,
        const FusedVector<T, N> &b)
    {
        static_assert(is_floating_point_v<T>,
                      "cholesky_solve requires a floating-point scalar type");

        // 1. Decompose A = LLᵀ
        auto chol_result = cholesky(A);

        if (!chol_result.has_value())
        {
            return Unexpected{chol_result.error()};
        }

        auto &L = chol_result.value();

        // 2. Solve Ly = b (forward substitution)
        //    L comes from cholesky — guaranteed non-singular, but forward_substitute
        //    checks anyway; no overhead for the unrolled paths.
        auto fwd_result = forward_substitute(L, b);

        if (!fwd_result.has_value())
        {
            return Unexpected{fwd_result.error()};
        }

        auto &y = fwd_result.value();

        // 3. Solve Lᵀx = y (back substitution on L transposed)
        //    Access L(k,i) instead of U(i,k) to avoid creating a transpose.
        //    L diagonal is guaranteed positive from cholesky — no singular check needed.
        FusedVector<T, N> x(T(0));

        for (my_size_t i = N; i-- > 0;)
        {
            T sum = y(i);

            for (my_size_t k = i + 1; k < N; ++k)
            {
                sum -= L(k, i) * x(k); // L(k,i) == Lᵀ(i,k)
            }

            x(i) = sum / L(i, i);
        }

        return move(x);
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_CHOLESKY_SOLVE_H
