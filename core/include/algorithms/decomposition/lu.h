#ifndef FUSED_ALGORITHMS_LU_H
#define FUSED_ALGORITHMS_LU_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h" // math::abs

/**
 * @file lu.h
 * @brief LU decomposition with partial pivoting for square matrices.
 *
 * Decomposes A into P·A = L·U where:
 *   - L is lower-triangular with unit diagonal (implicit, stored below diagonal)
 *   - U is upper-triangular (stored on and above diagonal)
 *   - P is a row permutation represented as an index vector
 *
 * The compact representation stores L and U in a single N×N matrix (LAPACK-style).
 * Accessor methods L() and U() extract separate matrices when needed.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 * @code
 * For each column j = 0 … N−1:
 *   1. Find pivot: row p ≥ j with max |A(p,j)|
 *   2. Swap rows j and p in LU and record in perm; flip sign
 *   3. Check for singularity: |U(j,j)| ≤ PRECISION_TOLERANCE
 *   4. Eliminate: for each row i > j:
 *        factor = LU(i,j) / LU(j,j)
 *        LU(i,j) = factor                        (stored in L part)
 *        LU(i,k) -= factor * LU(j,k)  for k > j  (update U part)
 * @endcode
 * Complexity: O(2N³/3) multiply-adds, O(N) comparisons for pivoting.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular — pivot magnitude ≤ tol (default PRECISION_TOLERANCE)
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    // ========================================================================
    // LU Result
    // ========================================================================

    /**
     * @brief Result of LU decomposition with partial pivoting.
     *
     * Stores the compact LU factorization (L below diagonal with implicit unit
     * diagonal, U on and above diagonal), the row permutation vector, and the
     * permutation sign for determinant computation.
     *
     * @tparam T  Scalar type.
     * @tparam N  Matrix dimension.
     */
    template <typename T, my_size_t N>
    struct LUResult
    {
        FusedMatrix<T, N, N> LU;        ///< Compact L+U storage.
        FusedVector<my_size_t, N> perm; ///< Row permutation: perm(i) = original row index.
        int sign;                       ///< Permutation sign: +1 (even) or -1 (odd).

        /**
         * @brief Extract the lower-triangular factor L with unit diagonal.
         * @return N×N matrix with L(i,j) from compact storage below diagonal,
         *         ones on diagonal, zeros above.
         */
        FusedMatrix<T, N, N> L() const
        {
            FusedMatrix<T, N, N> result(T(0));

            for (my_size_t i = 0; i < N; ++i)
            {
                result(i, i) = T(1); // unit diagonal

                for (my_size_t j = 0; j < i; ++j)
                {
                    result(i, j) = LU(i, j);
                }
            }

            return result;
        }

        /**
         * @brief Extract the upper-triangular factor U.
         * @return N×N matrix with U(i,j) from compact storage on and above diagonal,
         *         zeros below.
         */
        FusedMatrix<T, N, N> U() const
        {
            FusedMatrix<T, N, N> result(T(0));

            for (my_size_t i = 0; i < N; ++i)
            {
                for (my_size_t j = i; j < N; ++j)
                {
                    result(i, j) = LU(i, j);
                }
            }

            return result;
        }
    };

    // ========================================================================
    // LU Decomposition
    // ========================================================================

    /**
     * @brief Compute the LU decomposition of a square matrix with partial pivoting.
     *
     * Decomposes A into P·A = L·U where L has unit diagonal.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A    Square input matrix (N×N).
     * @param  tol  Pivot tolerance. Pivots with |value| ≤ tol are rejected as singular.
     *              Defaults to PRECISION_TOLERANCE.
     * @return Expected containing LUResult on success,
     *         or MatrixStatus::Singular on zero pivot.
     *
     * @par Example:
     * @code
     *   FusedMatrix<double, 3, 3> A;
     *   // ... fill A ...
     *   auto result = matrix_algorithms::lu(A);
     *   if (!result.has_value()) {
     *       // handle result.error()
     *       return;
     *   }
     *   auto& lu = result.value();
     *   // lu.LU contains compact factorization
     *   // lu.perm contains row permutation
     *   // lu.sign contains permutation sign
     *   auto L = lu.L();  // extract L if needed
     *   auto U = lu.U();  // extract U if needed
     * @endcode
     */
    template <typename T, my_size_t N>
    Expected<LUResult<T, N>, MatrixStatus> lu(
        const FusedMatrix<T, N, N> &A,
        T tol = T(PRECISION_TOLERANCE))
    {
        static_assert(is_floating_point_v<T>,
                      "lu requires a floating-point scalar type");

        LUResult<T, N> result;
        result.LU = A; // work on a copy
        result.sign = 1;

        // Initialize permutation to identity
        for (my_size_t i = 0; i < N; ++i)
        {
            result.perm(i) = i;
        }

        for (my_size_t j = 0; j < N; ++j)
        {
            // 1. Find pivot: row with max |A(p,j)| for p >= j
            my_size_t pivot = j;
            T max_val = math::abs(result.LU(j, j));

            for (my_size_t p = j + 1; p < N; ++p)
            {
                T val = math::abs(result.LU(p, j));

                if (val > max_val)
                {
                    max_val = val;
                    pivot = p;
                }
            }

            // 2. Swap rows j and pivot
            if (pivot != j)
            {
                // Swap entire rows in LU
                for (my_size_t k = 0; k < N; ++k)
                {
                    T tmp = result.LU(j, k);
                    result.LU(j, k) = result.LU(pivot, k);
                    result.LU(pivot, k) = tmp;
                }

                // Swap permutation entries
                my_size_t tmp_perm = result.perm(j);
                result.perm(j) = result.perm(pivot);
                result.perm(pivot) = tmp_perm;

                result.sign = -result.sign;
            }

            // 3. Check for singularity
            T diag = result.LU(j, j);

            if (diag <= tol && diag >= -tol)
            {
                return Unexpected{MatrixStatus::Singular};
            }

            // 4. Eliminate below pivot
            for (my_size_t i = j + 1; i < N; ++i)
            {
                T factor = result.LU(i, j) / diag;
                result.LU(i, j) = factor; // store L factor

                for (my_size_t k = j + 1; k < N; ++k)
                {
                    result.LU(i, k) -= factor * result.LU(j, k);
                }
            }
        }

        return move(result);
    }

    /**
     * @brief LU decomposition — abort on failure.
     *
     * Convenience wrapper for contexts where failure is unrecoverable.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @return LUResult containing factorization.
     */
    template <typename T, my_size_t N>
    LUResult<T, N> lu_or_die(const FusedMatrix<T, N, N> &A)
    {
        auto result = lu(A);

        if (!result.has_value())
        {
            MyErrorHandler::error("LU decomposition failed");
        }

        return move(result.value());
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_LU_H
