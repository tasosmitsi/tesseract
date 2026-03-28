#ifndef FUSED_ALGORITHMS_DETERMINANT_H
#define FUSED_ALGORITHMS_DETERMINANT_H

#include "config.h"
#include "algorithms/decomposition/lu.h"

/**
 * @file determinant.h
 * @brief Determinant computation with compile-time dispatch.
 *
 * Small sizes (1×1, 2×2, 3×3, 4×4) use unrolled cofactor expansion — O(1),
 * works on any scalar type including integers.
 *
 * Generic path (N>4) uses LU decomposition:
 *   det(A) = sign · Π U(i,i)
 * Returns zero for singular matrices. Requires floating-point.
 *
 * Complexity: O(1) for N≤4, O(2N³/3) for N>4.
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

            return c0 * s5 - c1 * s4 + c2 * s3 + c3 * s2 - c4 * s1 + c5 * s0;
        }
        else
        {
            static_assert(is_floating_point_v<T>,
                          "determinant requires a floating-point scalar type for N > 4");

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
