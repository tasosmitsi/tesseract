#ifndef FUSED_ALGORITHMS_LINEAR_SOLVE_H
#define FUSED_ALGORITHMS_LINEAR_SOLVE_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "algorithms/decomposition/lu.h"
#include "algorithms/solvers/triangular_solve.h"
#include "algorithms/solvers/cholesky_solve.h"

/**
 * @file linear_solve.h
 * @brief Solve Ax = b for general or symmetric positive-definite A.
 *
 * Provides:
 *   - lu_solve(A, b):  always uses LU path — works for any non-singular A
 *   - solve(A, b):     auto-dispatches to Cholesky (if SPD) or LU
 *
 * ============================================================================
 * ALGORITHM — lu_solve
 * ============================================================================
 *
 *   1. Decompose P·A = L·U via lu(A)
 *   2. Permute b: b_perm(i) = b(perm(i))
 *   3. Solve L·y = b_perm via forward substitution (UnitDiag=true)
 *   4. Solve U·x = y via back substitution
 *
 * Complexity: O(2N³/3) for LU + O(N²) for substitutions.
 *
 * ============================================================================
 * ALGORITHM — solve (dispatcher)
 * ============================================================================
 *
 *   1. Attempt cholesky_solve(A, b) — O(N³/3) if A is SPD
 *   2. If it fails (not symmetric or not positive definite), fall back to
 *      lu_solve(A, b) — O(2N³/3) for any non-singular A
 *
 * The Cholesky attempt is not wasted work — if A is SPD, Cholesky is ~2×
 * cheaper than LU. If A is not SPD, the failure is detected quickly
 * (symmetry check is O(N²), positive-definiteness fails at the first
 * non-positive diagonal during factorization).
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular          — from lu() if A is singular
 * - MatrixStatus::DimensionMismatch — if matrix/vector sizes don't match
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Solve Ax = b via LU decomposition with partial pivoting.
     *
     * Works for any non-singular square matrix.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix/vector dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @param  b  Right-hand side vector (N).
     * @return Expected containing solution x on success,
     *         or MatrixStatus error on failure.
     */
    template <typename T, my_size_t N>
    Expected<FusedVector<T, N>, MatrixStatus> lu_solve(
        const FusedMatrix<T, N, N> &A,
        const FusedVector<T, N> &b)
    {
        static_assert(is_floating_point_v<T>,
                      "lu_solve requires a floating-point scalar type");

        // 1. Decompose P·A = L·U
        auto lu_result = lu(A);

        if (!lu_result.has_value())
        {
            return Unexpected{lu_result.error()};
        }

        auto &decomp = lu_result.value();

        // 2. Permute b
        FusedVector<T, N> b_perm(T(0));

        for (my_size_t i = 0; i < N; ++i)
        {
            b_perm(i) = b(decomp.perm(i));
        }

        // 3. Extract L and U
        auto L = decomp.L();
        auto U = decomp.U();

        // 4. Solve L·y = b_perm (forward substitution, unit diagonal)
        auto fwd_result = forward_substitute<true>(L, b_perm);

        if (!fwd_result.has_value())
        {
            return Unexpected{fwd_result.error()};
        }

        auto &y = fwd_result.value();

        // 5. Solve U·x = y (back substitution)
        return back_substitute(U, y);
    }

    /**
     * @brief Solve Ax = b with automatic algorithm selection.
     *
     * Attempts Cholesky path first (cheaper for SPD matrices), falls back
     * to LU path for general non-singular matrices.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix/vector dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @param  b  Right-hand side vector (N).
     * @return Expected containing solution x on success,
     *         or MatrixStatus error on failure.
     *
     * @par Example:
     * @code
     *   FusedMatrix<double, 3, 3> A;
     *   FusedVector<double, 3> b;
     *   // ... fill A and b ...
     *   auto result = matrix_algorithms::solve(A, b);
     *   if (!result.has_value()) {
     *       // handle result.error()
     *       return;
     *   }
     *   auto& x = result.value();
     *   // A * x ≈ b
     * @endcode
     */
    template <typename T, my_size_t N>
    Expected<FusedVector<T, N>, MatrixStatus> solve(
        const FusedMatrix<T, N, N> &A,
        const FusedVector<T, N> &b)
    {
        static_assert(is_floating_point_v<T>,
                      "solve requires a floating-point scalar type");

        // Try Cholesky path first (O(N³/3) — cheaper if SPD)
        auto chol_result = cholesky_solve(A, b);

        if (chol_result.has_value())
        {
            return chol_result;
        }

        // Fall back to LU path (O(2N³/3) — works for any non-singular A)
        return lu_solve(A, b);
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_LINEAR_SOLVE_H
