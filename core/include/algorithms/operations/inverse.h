#ifndef FUSED_ALGORITHMS_INVERSE_H
#define FUSED_ALGORITHMS_INVERSE_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "algorithms/decomposition/lu.h"
#include "algorithms/solvers/triangular_solve.h"
#include "math/math_utils.h"

/**
 * @file inverse.h
 * @brief Matrix inverse with compile-time dispatch.
 *
 * Small sizes (1×1, 2×2, 3×3, 4×4) use direct adjugate/det formulas — O(1),
 * fully unrolled, no LU overhead.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * Generic path (N>4) uses LU decomposition:
 *   1. Decompose P·A = L·U via lu(A)
 *   2. Solve L·Y = P·I, then U·X = Y
 *
 * Complexity: O(1) for N≤4, O(5N³/3) for N>4.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular — matrix is not invertible (det ≈ 0 or LU fails)
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Compute the inverse of a square matrix.
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

        if constexpr (N == 1)
        {
            if (math::abs(A(0, 0)) <= T(PRECISION_TOLERANCE))
            {
                return Unexpected{MatrixStatus::Singular};
            }

            FusedMatrix<T, 1, 1> result(T(0));
            result(0, 0) = T(1) / A(0, 0);
            return move(result);
        }
        else if constexpr (N == 2)
        {
            T det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

            if (math::abs(det) <= T(PRECISION_TOLERANCE))
            {
                return Unexpected{MatrixStatus::Singular};
            }

            T inv_det = T(1) / det;

            FusedMatrix<T, 2, 2> result(T(0));
            result(0, 0) = A(1, 1) * inv_det;
            result(0, 1) = -A(0, 1) * inv_det;
            result(1, 0) = -A(1, 0) * inv_det;
            result(1, 1) = A(0, 0) * inv_det;
            return move(result);
        }
        else if constexpr (N == 3)
        {
            // Cofactors
            T c00 = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
            T c01 = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
            T c02 = A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0);

            T det = A(0, 0) * c00 + A(0, 1) * c01 + A(0, 2) * c02;

            if (math::abs(det) <= T(PRECISION_TOLERANCE))
            {
                return Unexpected{MatrixStatus::Singular};
            }

            T inv_det = T(1) / det;

            FusedMatrix<T, 3, 3> result(T(0));

            result(0, 0) = c00 * inv_det;
            result(0, 1) = (A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2)) * inv_det;
            result(0, 2) = (A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1)) * inv_det;

            result(1, 0) = c01 * inv_det;
            result(1, 1) = (A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0)) * inv_det;
            result(1, 2) = (A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2)) * inv_det;

            result(2, 0) = c02 * inv_det;
            result(2, 1) = (A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1)) * inv_det;
            result(2, 2) = (A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0)) * inv_det;

            return move(result);
        }
        else if constexpr (N == 4)
        {
            // 2×2 minors from rows 0–1
            T c0 = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
            T c1 = A(0, 0) * A(1, 2) - A(0, 2) * A(1, 0);
            T c2 = A(0, 0) * A(1, 3) - A(0, 3) * A(1, 0);
            T c3 = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
            T c4 = A(0, 1) * A(1, 3) - A(0, 3) * A(1, 1);
            T c5 = A(0, 2) * A(1, 3) - A(0, 3) * A(1, 2);

            // 2×2 minors from rows 2–3
            T s0 = A(2, 0) * A(3, 1) - A(2, 1) * A(3, 0);
            T s1 = A(2, 0) * A(3, 2) - A(2, 2) * A(3, 0);
            T s2 = A(2, 0) * A(3, 3) - A(2, 3) * A(3, 0);
            T s3 = A(2, 1) * A(3, 2) - A(2, 2) * A(3, 1);
            T s4 = A(2, 1) * A(3, 3) - A(2, 3) * A(3, 1);
            T s5 = A(2, 2) * A(3, 3) - A(2, 3) * A(3, 2);

            T det = c0 * s5 - c1 * s4 + c2 * s3 + c3 * s2 - c4 * s1 + c5 * s0;

            if (math::abs(det) <= T(PRECISION_TOLERANCE))
            {
                return Unexpected{MatrixStatus::Singular};
            }

            T inv_det = T(1) / det;

            FusedMatrix<T, 4, 4> result(T(0));

            // Adjugate transposed, row by row
            result(0, 0) = (A(1, 1) * s5 - A(1, 2) * s4 + A(1, 3) * s3) * inv_det;
            result(0, 1) = (-A(0, 1) * s5 + A(0, 2) * s4 - A(0, 3) * s3) * inv_det;
            result(0, 2) = (A(3, 1) * c5 - A(3, 2) * c4 + A(3, 3) * c3) * inv_det;
            result(0, 3) = (-A(2, 1) * c5 + A(2, 2) * c4 - A(2, 3) * c3) * inv_det;

            result(1, 0) = (-A(1, 0) * s5 + A(1, 2) * s2 - A(1, 3) * s1) * inv_det;
            result(1, 1) = (A(0, 0) * s5 - A(0, 2) * s2 + A(0, 3) * s1) * inv_det;
            result(1, 2) = (-A(3, 0) * c5 + A(3, 2) * c2 - A(3, 3) * c1) * inv_det;
            result(1, 3) = (A(2, 0) * c5 - A(2, 2) * c2 + A(2, 3) * c1) * inv_det;

            result(2, 0) = (A(1, 0) * s4 - A(1, 1) * s2 + A(1, 3) * s0) * inv_det;
            result(2, 1) = (-A(0, 0) * s4 + A(0, 1) * s2 - A(0, 3) * s0) * inv_det;
            result(2, 2) = (A(3, 0) * c4 - A(3, 1) * c2 + A(3, 3) * c0) * inv_det;
            result(2, 3) = (-A(2, 0) * c4 + A(2, 1) * c2 - A(2, 3) * c0) * inv_det;

            result(3, 0) = (-A(1, 0) * s3 + A(1, 1) * s1 - A(1, 2) * s0) * inv_det;
            result(3, 1) = (A(0, 0) * s3 - A(0, 1) * s1 + A(0, 2) * s0) * inv_det;
            result(3, 2) = (-A(3, 0) * c3 + A(3, 1) * c1 - A(3, 2) * c0) * inv_det;
            result(3, 3) = (A(2, 0) * c3 - A(2, 1) * c1 + A(2, 2) * c0) * inv_det;

            return move(result);
        }
        else
        {
            // Generic LU path

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
