#ifndef FUSED_ALGORITHMS_TRACE_H
#define FUSED_ALGORITHMS_TRACE_H

#include "config.h"
#include "fused/fused_matrix.h"

/**
 * @file trace.h
 * @brief Trace of a square matrix — sum of diagonal elements.
 *
 * Infallible, O(N). Works for any scalar type (not restricted to floating point).
 *
 * @note Future optimization: if a diagonal view is supported (strided view with
 * stride = N+1 into the flat storage), trace becomes a single `reduce_sum` call
 * on that view, gaining SIMD acceleration for free via the existing reduction kernel.
 */

namespace matrix_algorithms
{

    /**
     * @brief Compute the trace of a square matrix.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @return Sum of diagonal elements: Σ A(i,i) for i = 0 … N−1.
     */
    template <typename T, my_size_t N>
    T trace(const FusedMatrix<T, N, N> &A)
    {
        T sum = T(0);

        for (my_size_t i = 0; i < N; ++i)
        {
            sum += A(i, i);
        }

        return sum;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_TRACE_H
