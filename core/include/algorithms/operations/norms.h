#ifndef FUSED_ALGORITHMS_NORMS_H
#define FUSED_ALGORITHMS_NORMS_H

#include "config.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h" // math::abs, math::sqrt

/**
 * @file norms.h
 * @brief Matrix and vector norms.
 *
 * Currently provides:
 *   - norm1: matrix 1-norm (max absolute column sum)
 *   - norm2: vector Euclidean norm (√(Σ vᵢ²))
 *
 * Future additions: infinity norm, Frobenius norm.
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

    /**
     * @brief Compute the Euclidean (2-norm) of a vector: √(Σ vᵢ²).
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Vector dimension (deduced).
     * @param  v  Input vector (N).
     * @return ‖v‖₂ = √(v(0)² + v(1)² + … + v(N−1)²).
     *
     * @note Future optimization: with a strided diagonal view, this could
     * use a SIMD dot product kernel (dot(v, v) then sqrt).
     */
    template <typename T, my_size_t N>
    T norm2(const FusedVector<T, N> &v)
    {
        static_assert(is_floating_point_v<T>,
                      "norm2 requires a floating-point scalar type");

        T sum_sq = T(0);

        for (my_size_t i = 0; i < N; ++i)
        {
            sum_sq += v(i) * v(i);
        }

        return math::sqrt(sum_sq);
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_NORMS_H
