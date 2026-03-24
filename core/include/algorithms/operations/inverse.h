#ifndef FUSED_ALGORITHMS_INVERSE_H
#define FUSED_ALGORITHMS_INVERSE_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "algorithms/decomposition/lu.h"
#include "algorithms/solvers/triangular_solve.h"

/**
 * @file inverse.h
 * @brief Matrix inverse via LU decomposition.
 *
 * Computes A⁻¹ by solving A·X = I using LU factorization with partial
 * pivoting and multi-RHS triangular substitution. Never forms the adjugate.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 *   1. Decompose P·A = L·U via lu(A)
 *   2. Build P·I (permuted identity from perm vector)
 *   3. Solve L·Y = P·I via forward substitution (UnitDiag=true, multi-RHS)
 *   4. Solve U·X = Y via back substitution (multi-RHS)
 *   5. X = A⁻¹
 *
 * Complexity: O(2N³/3) for LU + O(N³) for the two substitutions = O(5N³/3).
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular — forwarded from lu() if A is singular
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Compute the inverse of a square matrix via LU decomposition.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @return Expected containing A⁻¹ on success,
     *         or MatrixStatus::Singular if A is not invertible.
     *
     * @par Example:
     * @code
     *   FusedMatrix<double, 3, 3> A;
     *   // ... fill A ...
     *   auto result = matrix_algorithms::inverse(A);
     *   if (!result.has_value()) {
     *       // handle singular matrix
     *       return;
     *   }
     *   auto& Ainv = result.value();
     *   // A * Ainv ≈ I
     * @endcode
     */
    template <typename T, my_size_t N>
    Expected<FusedMatrix<T, N, N>, MatrixStatus> inverse(const FusedMatrix<T, N, N> &A)
    {
        static_assert(is_floating_point_v<T>,
                      "inverse requires a floating-point scalar type");

        // 1. Decompose P·A = L·U
        auto lu_result = lu(A);

        if (!lu_result.has_value())
        {
            return Unexpected{lu_result.error()};
        }

        auto &decomp = lu_result.value();

        // 2. Build P·I (permuted identity)
        FusedMatrix<T, N, N> PI(T(0));

        for (my_size_t i = 0; i < N; ++i)
        {
            PI(i, decomp.perm(i)) = T(1);
        }

        // 3. Extract L and U for substitution
        auto L = decomp.L();
        auto U = decomp.U();

        // 4. Solve L·Y = P·I (forward substitution, unit diagonal)
        auto fwd_result = forward_substitute<true>(L, PI);

        if (!fwd_result.has_value())
        {
            return Unexpected{fwd_result.error()};
        }

        auto &Y = fwd_result.value();

        // 5. Solve U·X = Y (back substitution)
        auto back_result = back_substitute(U, Y);

        if (!back_result.has_value())
        {
            return Unexpected{back_result.error()};
        }

        return move(back_result.value());
    }

    /**
     * @brief Matrix inverse — abort on failure.
     *
     * Convenience wrapper for contexts where failure is unrecoverable.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @return A⁻¹.
     */
    template <typename T, my_size_t N>
    FusedMatrix<T, N, N> inverse_or_die(const FusedMatrix<T, N, N> &A)
    {
        auto result = inverse(A);

        if (!result.has_value())
        {
            MyErrorHandler::error("matrix inverse failed: singular matrix");
        }

        return move(result.value());
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_INVERSE_H
