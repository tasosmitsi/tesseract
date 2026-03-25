#ifndef FUSED_ALGORITHMS_TRIDIAGONAL_H
#define FUSED_ALGORITHMS_TRIDIAGONAL_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "fused/fused_vector.h"
#include "math/math_utils.h" // math::abs

/**
 * @file tridiagonal.h
 * @brief Thomas algorithm: O(N) solver for tridiagonal systems.
 *
 * Solves Ax = b where A is tridiagonal, represented by three vectors:
 *   - a: sub-diagonal (N−1 elements, a(0) unused)
 *   - d: main diagonal (N elements)
 *   - c: super-diagonal (N−1 elements, c(N−1) unused)
 *
 *   [ d(0) c(0)                    ] [x(0)]   [b(0)]
 *   [ a(1) d(1) c(1)               ] [x(1)]   [b(1)]
 *   [      a(2) d(2) c(2)          ] [x(2)] = [b(2)]
 *   [            ...   ...   ...    ] [ ... ]   [ ... ]
 *   [              a(N-1) d(N-1)    ] [x(N-1)] [b(N-1)]
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 *
 * Forward sweep (eliminate sub-diagonal):
 *   For i = 1 … N−1:
 *     w = a(i) / d'(i−1)
 *     d'(i) = d(i) − w · c(i−1)
 *     b'(i) = b(i) − w · b'(i−1)
 *
 * Back substitution:
 *   x(N−1) = b'(N−1) / d'(N−1)
 *   For i = N−2 … 0:
 *     x(i) = (b'(i) − c(i) · x(i+1)) / d'(i)
 *
 * Complexity: O(N) — optimal for tridiagonal systems.
 *
 * ============================================================================
 * USE CASES
 * ============================================================================
 *
 * - Cubic spline interpolation
 * - Trajectory smoothing
 * - Heat equation / diffusion (implicit methods)
 * - Boundary value problems (finite differences)
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::Singular — zero pivot encountered during forward sweep
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    /**
     * @brief Solve a tridiagonal system Ax = b using the Thomas algorithm.
     *
     * @tparam T  Scalar type (deduced).
     * @tparam N  System size (deduced).
     * @param  a  Sub-diagonal vector (N). a(0) is unused.
     * @param  d  Main diagonal vector (N).
     * @param  c  Super-diagonal vector (N). c(N−1) is unused.
     * @param  b  Right-hand side vector (N).
     * @return Expected containing solution x on success,
     *         or MatrixStatus::Singular on zero pivot.
     *
     * @par Example:
     * @code
     *   // Solve [-2 1 0; 1 -2 1; 0 1 -2] x = b
     *   FusedVector<double, 3> a, d, c, b;
     *   a(0) = 0; a(1) = 1; a(2) = 1;   // sub-diagonal
     *   d(0) = -2; d(1) = -2; d(2) = -2; // main diagonal
     *   c(0) = 1; c(1) = 1; c(2) = 0;   // super-diagonal
     *   // ... fill b ...
     *   auto result = matrix_algorithms::thomas_solve(a, d, c, b);
     * @endcode
     */
    template <typename T, my_size_t N>
    Expected<FusedVector<T, N>, MatrixStatus> thomas_solve(
        const FusedVector<T, N> &a,
        const FusedVector<T, N> &d,
        const FusedVector<T, N> &c,
        const FusedVector<T, N> &b)
    {
        static_assert(is_floating_point_v<T>,
                      "thomas_solve requires a floating-point scalar type");

        // Work on copies (modified during forward sweep)
        FusedVector<T, N> dp = d;
        FusedVector<T, N> bp = b;

        // Forward sweep
        for (my_size_t i = 1; i < N; ++i)
        {
            T diag = dp(i - 1);

            if (diag <= T(PRECISION_TOLERANCE) && diag >= T(-PRECISION_TOLERANCE))
            {
                return Unexpected{MatrixStatus::Singular};
            }

            T w = a(i) / diag;
            dp(i) = dp(i) - w * c(i - 1);
            bp(i) = bp(i) - w * bp(i - 1);
        }

        // Check last pivot
        T last = dp(N - 1);

        if (last <= T(PRECISION_TOLERANCE) && last >= T(-PRECISION_TOLERANCE))
        {
            return Unexpected{MatrixStatus::Singular};
        }

        // Back substitution
        FusedVector<T, N> x(T(0));

        x(N - 1) = bp(N - 1) / dp(N - 1);

        for (my_size_t i = N - 1; i-- > 0;)
        {
            x(i) = (bp(i) - c(i) * x(i + 1)) / dp(i);
        }

        return move(x);
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_TRIDIAGONAL_H
