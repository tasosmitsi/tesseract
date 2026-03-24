#ifndef FUSED_ALGORITHMS_DETERMINANT_H
#define FUSED_ALGORITHMS_DETERMINANT_H

#include "config.h"
#include "algorithms/decomposition/lu.h"

/**
 * @file determinant.h
 * @brief Determinant computation via LU decomposition.
 *
 * Computes det(A) = sign · Π U(i,i) where sign accounts for row
 * permutations during partial pivoting. Returns zero for singular matrices.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 *   1. Decompose P·A = L·U via lu(A)
 *   2. det(A) = sign · U(0,0) · U(1,1) · … · U(N-1,N-1)
 *
 * L has unit diagonal so det(L) = 1. The permutation contributes ±1.
 *
 * Complexity: O(2N³/3) for LU + O(N) for the product.
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    /**
     * @brief Compute the determinant of a square matrix via LU decomposition.
     *
     * Returns zero if the matrix is singular (LU fails). This makes the
     * function infallible — no Expected needed.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  Matrix dimension (deduced).
     * @param  A  Square input matrix (N×N).
     * @return Determinant of A. Zero if A is singular.
     */
    template <typename T, my_size_t N>
    T determinant(const FusedMatrix<T, N, N> &A)
    {
        static_assert(is_floating_point_v<T>,
                      "determinant requires a floating-point scalar type");

        auto lu_result = lu(A);

        if (!lu_result.has_value())
        {
            return T(0);
        }

        auto &decomp = lu_result.value();

        T det = T(decomp.sign);

        for (my_size_t i = 0; i < N; ++i)
        {
            det *= decomp.LU(i, i); // diagonal of U
        }

        return det;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_DETERMINANT_H
