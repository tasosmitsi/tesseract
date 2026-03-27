#ifndef FUSED_ALGORITHMS_SKEW_SYMMETRIC_H
#define FUSED_ALGORITHMS_SKEW_SYMMETRIC_H

#include "config.h"
#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h"             // math::sqrt, math::sin, math::cos
#include "algorithms/operations/norms.h" // norm2

/**
 * @file skew_symmetric.h
 * @brief Skew-symmetric matrix construction and Rodrigues rotation formula.
 *
 * Provides:
 *   - skew_symmetric(v): 3-vector → 3×3 skew-symmetric matrix [v]×
 *   - rodrigues(omega, t): matrix exponential exp(t·[ω]×) for SO(3) rotations
 *
 * ============================================================================
 * SKEW-SYMMETRIC (5a)
 * ============================================================================
 *
 * Given ω = [ω₁, ω₂, ω₃], the skew-symmetric matrix is:
 *
 *   [ω]× = [  0  -ω₃  ω₂ ]
 *          [  ω₃  0  -ω₁ ]
 *          [ -ω₂  ω₁  0  ]
 *
 * Key property: [ω]× · v = ω × v (cross product).
 *
 * ============================================================================
 * RODRIGUES FORMULA (5b)
 * ============================================================================
 *
 * Given an angular velocity vector ω and time step t, the rotation matrix is:
 *
 *   R = exp(t · [ω]×) = I + sin(θ)/θ · [ω]× + (1 − cos(θ))/θ² · [ω]×²
 *
 * where θ = t · ‖ω‖ is the rotation angle.
 *
 * Special cases:
 *   - θ ≈ 0: R ≈ I + t · [ω]× (first-order Taylor, avoids 0/0)
 *   - ‖ω‖ = 0: R = I (no rotation)
 *
 * Properties of the result:
 *   - R is orthogonal: RᵀR = I
 *   - det(R) = +1 (proper rotation, SO(3))
 *   - R propagates attitude: q(t+dt) = R · q(t)
 *
 * Complexity: O(1) — fixed 3×3 operations, no loops.
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    /**
     * @brief Construct the 3×3 skew-symmetric matrix [v]× from a 3-vector.
     *
     * @tparam T  Scalar type (deduced).
     * @param  v  3-vector [v₁, v₂, v₃].
     * @return 3×3 skew-symmetric matrix such that [v]× · u = v × u.
     */
    template <typename T>
    FusedMatrix<T, 3, 3> skew_symmetric(const FusedVector<T, 3> &v)
    {
        FusedMatrix<T, 3, 3> S(T(0));

        S(0, 1) = -v(2);
        S(0, 2) = v(1);
        S(1, 0) = v(2);
        S(1, 2) = -v(0);
        S(2, 0) = -v(1);
        S(2, 1) = v(0);

        return S;
    }

    /**
     * @brief Compute the rotation matrix R = exp(t · [ω]×) via Rodrigues formula.
     *
     * Computes the SO(3) rotation matrix for angular velocity ω over time step t.
     *
     * @tparam T      Scalar type (deduced).
     * @param  omega  Angular velocity 3-vector (rad/s).
     * @param  t      Time step (seconds). Default 1.0 (omega is then the rotation vector).
     * @return 3×3 rotation matrix R ∈ SO(3).
     */
    template <typename T>
    FusedMatrix<T, 3, 3> rodrigues(const FusedVector<T, 3> &omega, T t = T(1))
    {
        static_assert(is_floating_point_v<T>,
                      "rodrigues requires a floating-point scalar type");

        FusedMatrix<T, 3, 3> I(T(0));
        I.setIdentity();

        // ‖ω‖
        T norm = norm2(omega);

        // θ = t · ‖ω‖
        T theta = t * norm;

        if (theta <= T(PRECISION_TOLERANCE))
        {
            // Small angle: R ≈ I + t · [ω]×
            auto S = skew_symmetric(omega);

            FusedMatrix<T, 3, 3> R(T(0));
            R = I;
            R(0, 1) += t * S(0, 1);
            R(0, 2) += t * S(0, 2);
            R(1, 0) += t * S(1, 0);
            R(1, 2) += t * S(1, 2);
            R(2, 0) += t * S(2, 0);
            R(2, 1) += t * S(2, 1);

            return R;
        }

        // [ω̂]× where ω̂ = ω/‖ω‖ (unit axis)
        FusedVector<T, 3> omega_hat(T(0));
        omega_hat(0) = omega(0) / norm;
        omega_hat(1) = omega(1) / norm;
        omega_hat(2) = omega(2) / norm;

        auto K = skew_symmetric(omega_hat);

        // K² = [ω̂]×²
        auto K2 = FusedMatrix<T, 3, 3>::matmul(K, K);

        // R = I + sin(θ)·K + (1 − cos(θ))·K²
        T s = math::sin(theta);
        T c = math::cos(theta);

        FusedMatrix<T, 3, 3> R(T(0));

        for (my_size_t i = 0; i < 3; ++i)
        {
            for (my_size_t j = 0; j < 3; ++j)
            {
                R(i, j) = I(i, j) + s * K(i, j) + (T(1) - c) * K2(i, j);
            }
        }

        return R;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_SKEW_SYMMETRIC_H
