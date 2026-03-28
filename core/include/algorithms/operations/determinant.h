#ifndef FUSED_ALGORITHMS_DETERMINANT_H
#define FUSED_ALGORITHMS_DETERMINANT_H

#include "config.h"
#include "algorithms/decomposition/lu.h"

/**
 * @file determinant.h
 * @brief Determinant computation with compile-time dispatch.
 *
 * Small sizes (1×1, 2×2, 3×3) use unrolled cofactor expansion — O(1),
 * works on any scalar type including integers.
 *
 * Generic path (N>3) uses LU decomposition:
 *   det(A) = sign · Π U(i,i)
 * Returns zero for singular matrices. Requires floating-point.
 *
 * Complexity: O(1) for N≤3, O(2N³/3) for N>3.
 */

namespace matrix_algorithms
{

    /**
     * @brief Compute the determinant of a square matrix.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @return Determinant of A. Zero if A is singular (generic path).
     */
    template <typename T, my_size_t N>
    T determinant(const FusedMatrix<T, N, N> &A)
    {
        if constexpr (N == 1)
        {
            return A(0, 0);
        }
        else if constexpr (N == 2)
        {
            return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        }
        else if constexpr (N == 3)
        {
            // clang-format off
            return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
                 - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) 
                 + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
            // clang-format on
        }
        else
        {
            static_assert(is_floating_point_v<T>,
                          "determinant requires a floating-point scalar type for N > 3");

            auto lu_result = lu(A);

            if (!lu_result.has_value())
            {
                return T(0);
            }

            auto &decomp = lu_result.value();

            T det = T(decomp.sign);

            for (my_size_t i = 0; i < N; ++i)
            {
                det *= decomp.LU(i, i);
            }

            return det;
        }
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_DETERMINANT_H
