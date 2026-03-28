#ifndef FUSED_ALGORITHMS_HOMOGENEOUS_TRANSFORM_H
#define FUSED_ALGORITHMS_HOMOGENEOUS_TRANSFORM_H

#include "config.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"

/**
 * @file homogeneous_transform.h
 * @brief Inverse of 4×4 homogeneous transformation matrices.
 *
 * A homogeneous transform encodes a rigid-body transformation in SE(3):
 *
 *       T = [ R | t ]
 *           [ 0 | 1 ]
 *
 * where R ∈ SO(3) is a 3×3 rotation and t ∈ ℝ³ is a translation.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * The inverse exploits the orthogonality of R (R⁻¹ = Rᵀ):
 *
 *       T⁻¹ = [ Rᵀ  | -Rᵀ·t ]
 *             [ 0   |   1    ]
 *
 * This avoids the general 4×4 inverse (which requires cofactor expansion
 * or LU decomposition) and uses only a 3×3 transpose + 3×3·3×1 matmul.
 *
 * Complexity: O(1) — 9 copies (transpose) + 9 multiplies + 6 adds (matmul).
 *
 * ============================================================================
 * PRECONDITIONS
 * ============================================================================
 *
 * The caller is responsible for ensuring T has valid homogeneous form:
 * - R must be orthogonal (Rᵀ·R = I, det(R) = +1)
 * - Bottom row must be [0 0 0 1]
 *
 * No runtime validation is performed. Passing a non-homogeneous matrix
 * produces a mathematically meaningless result.
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    /**
     * @brief Compute the inverse of a 4×4 homogeneous transformation matrix.
     *
     * @tparam T  Scalar type (deduced). Must be floating-point.
     * @param  H  4×4 homogeneous transform [R|t; 0 0 0 1].
     * @return H⁻¹ = [Rᵀ | -Rᵀt; 0 0 0 1].
     */
    template <typename T>
    FusedMatrix<T, 4, 4> homogeneous_inverse(const FusedMatrix<T, 4, 4> &H)
    {
        static_assert(is_floating_point_v<T>,
                      "homogeneous_inverse requires a floating-point scalar type");

        FusedMatrix<T, 4, 4> result(T(0));

        // Rᵀ — transpose the 3×3 rotation block
        for (my_size_t i = 0; i < 3; ++i)
        {
            for (my_size_t j = 0; j < 3; ++j)
            {
                result(i, j) = H(j, i);
            }
        }

        // -Rᵀ · t
        for (my_size_t i = 0; i < 3; ++i)
        {
            // clang-format off
            result(i, 3) =  -(result(i, 0) * H(0, 3) 
                            + result(i, 1) * H(1, 3) 
                            + result(i, 2) * H(2, 3));
            // clang-format on
        }

        // Bottom row: [0 0 0 1]
        result(3, 3) = T(1);

        return result;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_HOMOGENEOUS_TRANSFORM_H
