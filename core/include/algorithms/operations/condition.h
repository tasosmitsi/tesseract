#ifndef FUSED_ALGORITHMS_CONDITION_H
#define FUSED_ALGORITHMS_CONDITION_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "algorithms/operations/norms.h"
#include "algorithms/operations/inverse.h"

/**
 * @file condition.h
 * @brief Condition number estimate via 1-norm: cond₁(A) = ‖A‖₁ · ‖A⁻¹‖₁.
 *
 * A condition number close to 1 means well-conditioned. Large values
 * (>10⁶ for double, >10³ for float) indicate that solve/inverse results
 * may lose significant digits.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 *   1. Compute ‖A‖₁ via norm1(A)
 *   2. Compute A⁻¹ via inverse(A)
 *   3. Compute ‖A⁻¹‖₁ via norm1(A⁻¹)
 *   4. Return ‖A‖₁ · ‖A⁻¹‖₁
 *
 * Complexity: O(N³) for inverse, O(N²) for the two norms.
 *
 * @note This is the exact condition number, not a cheap estimate.
 * A cheaper O(N²) estimate (Hager/Higham algorithm) could be added later
 * using the LU factorization directly without forming the full inverse.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular — A is not invertible (condition = infinity)
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Compute the condition number of A in the 1-norm: cond₁(A) = ‖A‖₁ · ‖A⁻¹‖₁.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @return Expected containing the condition number on success,
     *         or MatrixStatus::Singular if A is not invertible.
     */
    template <typename T, my_size_t N>
    Expected<T, MatrixStatus> condition(const FusedMatrix<T, N, N> &A)
    {
        static_assert(is_floating_point_v<T>,
                      "condition requires a floating-point scalar type");

        T norm_A = norm1(A);

        auto inv_result = inverse(A);

        if (!inv_result.has_value())
        {
            return Unexpected{inv_result.error()};
        }

        T norm_Ainv = norm1(inv_result.value());

        return norm_A * norm_Ainv;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_CONDITION_H
