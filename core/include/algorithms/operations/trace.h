#ifndef FUSED_ALGORITHMS_TRACE_H
#define FUSED_ALGORITHMS_TRACE_H

#include "config.h"
#include "fused/fused_matrix.h"

/**
 * @file trace.h
 * @brief Trace of a square matrix aka sum of diagonal elements.
 *
 * Infallible, O(N). Works for any scalar type (not restricted to floating point).
 *
 * TODO: Worth revisiting on a target whose microkernel lacks a hardware gather.
 * The scalar loop below is deliberate. `sum(A.diagonal())` computes the same
 * thing through the diagonal view and the SIMD reduction kernel, and benchmarks
 * put the two within a few percent of each other at every size from 3x3 to
 * 1000x1000.
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
