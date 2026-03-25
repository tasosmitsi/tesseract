#ifndef FUSED_ALGORITHMS_NORMS_H
#define FUSED_ALGORITHMS_NORMS_H

#include "config.h"
#include "fused/fused_matrix.h"
#include "math/math_utils.h" // math::abs

/**
 * @file norms.h
 * @brief Matrix and vector norms.
 *
 * Currently provides:
 *   - norm1: matrix 1-norm (max absolute column sum)
 *
 * Future additions: infinity norm, Frobenius norm, vector norms.
 */

namespace matrix_algorithms
{

    /**
     * @brief Compute the 1-norm of a square matrix: max absolute column sum.
     *
     * Works for any numeric type (not restricted to floating point).
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @return ‖A‖₁ = max_j Σ_i |A(i,j)|.
     */
    template <typename T, my_size_t N>
    T norm1(const FusedMatrix<T, N, N> &A)
    {
        T max_col_sum = T(0);

        for (my_size_t j = 0; j < N; ++j)
        {
            T col_sum = T(0);

            for (my_size_t i = 0; i < N; ++i)
            {
                col_sum += math::abs(A(i, j));
            }

            if (col_sum > max_col_sum)
            {
                max_col_sum = col_sum;
            }
        }

        return max_col_sum;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_NORMS_H
