#ifndef FUSED_ALGORITHMS_CROSS_PRODUCT_H
#define FUSED_ALGORITHMS_CROSS_PRODUCT_H

#include "config.h" // for my_size_t
#include "fused/fused_vector.h"

/**
 * @file cross_product.h
 * @brief Cross product for 3-vectors.
 *
 * Computes a × b where a, b ∈ ℝ³. The result is orthogonal to both
 * inputs with magnitude ‖a‖·‖b‖·sin(θ), following the right-hand rule.
 *
 * ============================================================================
 * FORMULA
 * ============================================================================
 *
 *   (a × b)₀ = a₁·b₂ − a₂·b₁
 *   (a × b)₁ = a₂·b₀ − a₀·b₂
 *   (a × b)₂ = a₀·b₁ − a₁·b₀
 *
 * Fully unrolled — O(1), 6 multiplies + 3 subtracts. Works on any scalar
 * type including integers.
 *
 * The cross product is only defined for N=3. Other dimensions are rejected
 * at compile time.
 *
 * ============================================================================
 * PROPERTIES
 * ============================================================================
 *
 * - Anticommutative: a × b = −(b × a)
 * - Bilinear: (αa) × b = α(a × b)
 * - Self-cross is zero: a × a = 0
 * - Orthogonality: (a × b) · a = 0, (a × b) · b = 0
 * - Scalar triple product: a · (b × c) = det([a b c]ᵀ)
 * - Relation to skew-symmetric: a × b = [a]× · b
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    /**
     * @brief Compute the cross product of two 3-vectors.
     *
     * @tparam T  Scalar type (deduced). Works on float, double, and integers.
     * @tparam N  Vector dimension (deduced). Must be 3.
     * @param  a  First input vector (3×1).
     * @param  b  Second input vector (3×1).
     * @return a × b as a 3-vector.
     */
    template <typename T, my_size_t N>
    FusedVector<T, N> cross(const FusedVector<T, N> &a,
                            const FusedVector<T, N> &b)
    {
        static_assert(N == 3,
                      "cross product is only defined for 3-vectors");

        FusedVector<T, 3> result(T(0));

        result(0) = a(1) * b(2) - a(2) * b(1);
        result(1) = a(2) * b(0) - a(0) * b(2);
        result(2) = a(0) * b(1) - a(1) * b(0);

        return result;
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_CROSS_PRODUCT_H
