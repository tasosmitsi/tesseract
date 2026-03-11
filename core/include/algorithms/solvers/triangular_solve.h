#ifndef FUSED_ALGORITHMS_TRIANGULAR_SOLVE_H
#define FUSED_ALGORITHMS_TRIANGULAR_SOLVE_H

#include "config.h"
#include "utilities/expected.h"
#include "matrix_traits.h"
#include "simple_type_traits.h"

/**
 * @file triangular_solve.h
 * @brief Forward/back substitution for triangular systems.
 *
 * Provides solvers for triangular linear systems Lx = b and Ux = b,
 * with compile-time dispatch for fixed-size unrolled paths (N ∈ {3, 4, 6})
 * and a UnitDiag template parameter for LU-style implicit unit diagonals.
 *
 * Also includes multi-RHS overloads (LX = B, UX = B) for matrix
 * right-hand sides, used by matrix inverse and related algorithms.
 *
 * ============================================================================
 * ALGORITHMS
 * ============================================================================
 *
 * Forward substitution (Lx = b, L lower-triangular):
 * @code
 *   For i = 0 … N−1:
 *     x(i) = ( b(i) − Σ_{k=0}^{i-1} L(i,k)·x(k) ) / L(i,i)
 *     (division skipped when UnitDiag = true)
 * @endcode
 * Back substitution (Ux = b, U upper-triangular):
 * @code
 *   For i = N−1 … 0:
 *     x(i) = ( b(i) − Σ_{k=i+1}^{N-1} U(i,k)·x(k) ) / U(i,i)
 *     (division skipped when UnitDiag = true)
 * @endcode
 * Complexity: O(N²/2) per substitution.
 *
 * ============================================================================
 * FAILURE MODES
 * ============================================================================
 *
 * - MatrixStatus::DimensionMismatch — matrix rows != vector size
 * - MatrixStatus::Singular          — zero diagonal encountered during substitution
 *
 * ============================================================================
 */

namespace matrix_algorithms
{

    using matrix_traits::MatrixStatus;

    // ========================================================================
    // Forward substitution: solve Lx = b  (L lower-triangular, single RHS)
    // ========================================================================

    /**
     * @brief Solve the lower-triangular system Lx = b by forward substitution.
     *
     * Compile-time dispatch selects a fully unrolled path for N ∈ {3, 4, 6};
     * all other sizes use a generic scalar loop.
     *
     * @tparam UnitDiag  If true, the diagonal of L is treated as all ones
     *                   (implicit unit diagonal, as produced by LU with partial pivoting).
     *                   If false, the actual diagonal entries are used.
     * @tparam T         Scalar type (deduced).
     * @tparam N         Matrix/vector dimension (deduced).
     * @param  L         Lower-triangular NxN matrix.
     * @param  b         Right-hand side vector of length N.
     * @return Expected containing solution x, or MatrixStatus::Singular on zero diagonal.
     */
    template <bool UnitDiag = false, typename T, my_size_t N>
    Expected<FusedVector<T, N>, MatrixStatus> forward_substitute(
        const FusedMatrix<T, N, N> &L,
        const FusedVector<T, N> &b)
    {
        static_assert(is_floating_point_v<T>,
                      "forward_substitute requires a floating-point scalar type");

        FusedVector<T, N> x(T(0));

        // --- Fixed-size fully unrolled paths ---
        if constexpr (!UnitDiag && N == 3)
        {
            if (L(0, 0) <= T(PRECISION_TOLERANCE) && L(0, 0) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(0) = b(0) / L(0, 0);

            if (L(1, 1) <= T(PRECISION_TOLERANCE) && L(1, 1) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(1) = (b(1) - L(1, 0) * x(0)) / L(1, 1);

            if (L(2, 2) <= T(PRECISION_TOLERANCE) && L(2, 2) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(2) = (b(2) - L(2, 0) * x(0) - L(2, 1) * x(1)) / L(2, 2);
        }
        else if constexpr (UnitDiag && N == 3)
        {
            x(0) = b(0);
            x(1) = b(1) - L(1, 0) * x(0);
            x(2) = b(2) - L(2, 0) * x(0) - L(2, 1) * x(1);
        }
        else if constexpr (!UnitDiag && N == 4)
        {
            if (L(0, 0) <= T(PRECISION_TOLERANCE) && L(0, 0) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(0) = b(0) / L(0, 0);

            if (L(1, 1) <= T(PRECISION_TOLERANCE) && L(1, 1) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(1) = (b(1) - L(1, 0) * x(0)) / L(1, 1);

            if (L(2, 2) <= T(PRECISION_TOLERANCE) && L(2, 2) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(2) = (b(2) - L(2, 0) * x(0) - L(2, 1) * x(1)) / L(2, 2);

            if (L(3, 3) <= T(PRECISION_TOLERANCE) && L(3, 3) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(3) = (b(3) - L(3, 0) * x(0) - L(3, 1) * x(1) - L(3, 2) * x(2)) / L(3, 3);
        }
        else if constexpr (UnitDiag && N == 4)
        {
            x(0) = b(0);
            x(1) = b(1) - L(1, 0) * x(0);
            x(2) = b(2) - L(2, 0) * x(0) - L(2, 1) * x(1);
            x(3) = b(3) - L(3, 0) * x(0) - L(3, 1) * x(1) - L(3, 2) * x(2);
        }
        else if constexpr (!UnitDiag && N == 6)
        {
            if (L(0, 0) <= T(PRECISION_TOLERANCE) && L(0, 0) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(0) = b(0) / L(0, 0);

            if (L(1, 1) <= T(PRECISION_TOLERANCE) && L(1, 1) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(1) = (b(1) - L(1, 0) * x(0)) / L(1, 1);

            if (L(2, 2) <= T(PRECISION_TOLERANCE) && L(2, 2) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(2) = (b(2) - L(2, 0) * x(0) - L(2, 1) * x(1)) / L(2, 2);

            if (L(3, 3) <= T(PRECISION_TOLERANCE) && L(3, 3) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(3) = (b(3) - L(3, 0) * x(0) - L(3, 1) * x(1) - L(3, 2) * x(2)) / L(3, 3);

            if (L(4, 4) <= T(PRECISION_TOLERANCE) && L(4, 4) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(4) = (b(4) - L(4, 0) * x(0) - L(4, 1) * x(1) - L(4, 2) * x(2) - L(4, 3) * x(3)) / L(4, 4);

            if (L(5, 5) <= T(PRECISION_TOLERANCE) && L(5, 5) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(5) = (b(5) - L(5, 0) * x(0) - L(5, 1) * x(1) - L(5, 2) * x(2) - L(5, 3) * x(3) - L(5, 4) * x(4)) / L(5, 5);
        }
        else if constexpr (UnitDiag && N == 6)
        {
            x(0) = b(0);
            x(1) = b(1) - L(1, 0) * x(0);
            x(2) = b(2) - L(2, 0) * x(0) - L(2, 1) * x(1);
            x(3) = b(3) - L(3, 0) * x(0) - L(3, 1) * x(1) - L(3, 2) * x(2);
            x(4) = b(4) - L(4, 0) * x(0) - L(4, 1) * x(1) - L(4, 2) * x(2) - L(4, 3) * x(3);
            x(5) = b(5) - L(5, 0) * x(0) - L(5, 1) * x(1) - L(5, 2) * x(2) - L(5, 3) * x(3) - L(5, 4) * x(4);
        }
        // --- Generic scalar path ---
        else
        {
            for (my_size_t i = 0; i < N; ++i)
            {
                T sum = b(i);

                for (my_size_t k = 0; k < i; ++k)
                {
                    sum -= L(i, k) * x(k);
                }

                if constexpr (UnitDiag)
                {
                    x(i) = sum;
                }
                else
                {
                    T diag = L(i, i);

                    if (diag <= T(PRECISION_TOLERANCE) && diag >= T(-PRECISION_TOLERANCE))
                    {
                        return Unexpected{MatrixStatus::Singular};
                    }

                    x(i) = sum / diag;
                }
            }
        }

        return move(x);
    }

    // ========================================================================
    // Back substitution: solve Ux = b  (U upper-triangular, single RHS)
    // ========================================================================

    /**
     * @brief Solve the upper-triangular system Ux = b by back substitution.
     *
     * Compile-time dispatch selects a fully unrolled path for N ∈ {3, 4, 6};
     * all other sizes use a generic scalar loop.
     *
     * @tparam UnitDiag  If true, the diagonal of U is treated as all ones.
     *                   If false, the actual diagonal entries are used.
     * @tparam T         Scalar type (deduced).
     * @tparam N         Matrix/vector dimension (deduced).
     * @param  U         Upper-triangular NxN matrix.
     * @param  b         Right-hand side vector of length N.
     * @return Expected containing solution x, or MatrixStatus::Singular on zero diagonal.
     */
    template <bool UnitDiag = false, typename T, my_size_t N>
    Expected<FusedVector<T, N>, MatrixStatus> back_substitute(
        const FusedMatrix<T, N, N> &U,
        const FusedVector<T, N> &b)
    {
        static_assert(is_floating_point_v<T>,
                      "back_substitute requires a floating-point scalar type");

        FusedVector<T, N> x(T(0));

        // --- Fixed-size fully unrolled paths ---
        if constexpr (!UnitDiag && N == 3)
        {
            if (U(2, 2) <= T(PRECISION_TOLERANCE) && U(2, 2) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(2) = b(2) / U(2, 2);

            if (U(1, 1) <= T(PRECISION_TOLERANCE) && U(1, 1) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(1) = (b(1) - U(1, 2) * x(2)) / U(1, 1);

            if (U(0, 0) <= T(PRECISION_TOLERANCE) && U(0, 0) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(0) = (b(0) - U(0, 1) * x(1) - U(0, 2) * x(2)) / U(0, 0);
        }
        else if constexpr (UnitDiag && N == 3)
        {
            x(2) = b(2);
            x(1) = b(1) - U(1, 2) * x(2);
            x(0) = b(0) - U(0, 1) * x(1) - U(0, 2) * x(2);
        }
        else if constexpr (!UnitDiag && N == 4)
        {
            if (U(3, 3) <= T(PRECISION_TOLERANCE) && U(3, 3) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(3) = b(3) / U(3, 3);

            if (U(2, 2) <= T(PRECISION_TOLERANCE) && U(2, 2) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(2) = (b(2) - U(2, 3) * x(3)) / U(2, 2);

            if (U(1, 1) <= T(PRECISION_TOLERANCE) && U(1, 1) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(1) = (b(1) - U(1, 2) * x(2) - U(1, 3) * x(3)) / U(1, 1);

            if (U(0, 0) <= T(PRECISION_TOLERANCE) && U(0, 0) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(0) = (b(0) - U(0, 1) * x(1) - U(0, 2) * x(2) - U(0, 3) * x(3)) / U(0, 0);
        }
        else if constexpr (UnitDiag && N == 4)
        {
            x(3) = b(3);
            x(2) = b(2) - U(2, 3) * x(3);
            x(1) = b(1) - U(1, 2) * x(2) - U(1, 3) * x(3);
            x(0) = b(0) - U(0, 1) * x(1) - U(0, 2) * x(2) - U(0, 3) * x(3);
        }
        else if constexpr (!UnitDiag && N == 6)
        {
            if (U(5, 5) <= T(PRECISION_TOLERANCE) && U(5, 5) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(5) = b(5) / U(5, 5);

            if (U(4, 4) <= T(PRECISION_TOLERANCE) && U(4, 4) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(4) = (b(4) - U(4, 5) * x(5)) / U(4, 4);

            if (U(3, 3) <= T(PRECISION_TOLERANCE) && U(3, 3) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(3) = (b(3) - U(3, 4) * x(4) - U(3, 5) * x(5)) / U(3, 3);

            if (U(2, 2) <= T(PRECISION_TOLERANCE) && U(2, 2) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(2) = (b(2) - U(2, 3) * x(3) - U(2, 4) * x(4) - U(2, 5) * x(5)) / U(2, 2);

            if (U(1, 1) <= T(PRECISION_TOLERANCE) && U(1, 1) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(1) = (b(1) - U(1, 2) * x(2) - U(1, 3) * x(3) - U(1, 4) * x(4) - U(1, 5) * x(5)) / U(1, 1);

            if (U(0, 0) <= T(PRECISION_TOLERANCE) && U(0, 0) >= T(-PRECISION_TOLERANCE))
                return Unexpected{MatrixStatus::Singular};
            x(0) = (b(0) - U(0, 1) * x(1) - U(0, 2) * x(2) - U(0, 3) * x(3) - U(0, 4) * x(4) - U(0, 5) * x(5)) / U(0, 0);
        }
        else if constexpr (UnitDiag && N == 6)
        {
            x(5) = b(5);
            x(4) = b(4) - U(4, 5) * x(5);
            x(3) = b(3) - U(3, 4) * x(4) - U(3, 5) * x(5);
            x(2) = b(2) - U(2, 3) * x(3) - U(2, 4) * x(4) - U(2, 5) * x(5);
            x(1) = b(1) - U(1, 2) * x(2) - U(1, 3) * x(3) - U(1, 4) * x(4) - U(1, 5) * x(5);
            x(0) = b(0) - U(0, 1) * x(1) - U(0, 2) * x(2) - U(0, 3) * x(3) - U(0, 4) * x(4) - U(0, 5) * x(5);
        }
        // --- Generic scalar path ---
        else
        {
            for (my_size_t i = N; i-- > 0;)
            {
                T sum = b(i);

                for (my_size_t k = i + 1; k < N; ++k)
                {
                    sum -= U(i, k) * x(k);
                }

                if constexpr (UnitDiag)
                {
                    x(i) = sum;
                }
                else
                {
                    T diag = U(i, i);

                    if (diag <= T(PRECISION_TOLERANCE) && diag >= T(-PRECISION_TOLERANCE))
                    {
                        return Unexpected{MatrixStatus::Singular};
                    }

                    x(i) = sum / diag;
                }
            }
        }

        return move(x);
    }

    // ========================================================================
    // Multi-RHS: solve LX = B  (B is FusedMatrix, column-by-column)
    // ========================================================================

    /**
     * @brief Solve the lower-triangular system LX = B for multiple right-hand sides.
     *
     * Each column of B is solved independently via forward substitution.
     * Uses the generic scalar path (no fixed-size unrolling).
     *
     * @tparam UnitDiag  If true, the diagonal of L is treated as all ones.
     * @tparam T         Scalar type (deduced).
     * @tparam N         System dimension (deduced).
     * @tparam Ncols     Number of right-hand side columns (deduced).
     * @param  L         Lower-triangular NxN matrix.
     * @param  B         Right-hand side matrix of size N × Ncols.
     * @return Expected containing solution matrix X, or MatrixStatus::Singular on zero diagonal.
     */
    template <bool UnitDiag = false, typename T, my_size_t N, my_size_t Ncols>
    Expected<FusedMatrix<T, N, Ncols>, MatrixStatus> forward_substitute(
        const FusedMatrix<T, N, N> &L,
        const FusedMatrix<T, N, Ncols> &B)
    {
        static_assert(is_floating_point_v<T>,
                      "forward_substitute requires a floating-point scalar type");

        FusedMatrix<T, N, Ncols> X(T(0));

        for (my_size_t j = 0; j < Ncols; ++j)
        {
            for (my_size_t i = 0; i < N; ++i)
            {
                T sum = B(i, j);

                for (my_size_t k = 0; k < i; ++k)
                {
                    sum -= L(i, k) * X(k, j);
                }

                if constexpr (UnitDiag)
                {
                    X(i, j) = sum;
                }
                else
                {
                    T diag = L(i, i);

                    if (diag <= T(PRECISION_TOLERANCE) && diag >= T(-PRECISION_TOLERANCE))
                    {
                        return Unexpected{MatrixStatus::Singular};
                    }

                    X(i, j) = sum / diag;
                }
            }
        }

        return move(X);
    }

    // ========================================================================
    // Multi-RHS: solve UX = B  (B is FusedMatrix, column-by-column)
    // ========================================================================

    /**
     * @brief Solve the upper-triangular system UX = B for multiple right-hand sides.
     *
     * Each column of B is solved independently via back substitution.
     * Uses the generic scalar path (no fixed-size unrolling).
     *
     * @tparam UnitDiag  If true, the diagonal of U is treated as all ones.
     * @tparam T         Scalar type (deduced).
     * @tparam N         System dimension (deduced).
     * @tparam Ncols     Number of right-hand side columns (deduced).
     * @param  U         Upper-triangular NxN matrix.
     * @param  B         Right-hand side matrix of size N × Ncols.
     * @return Expected containing solution matrix X, or MatrixStatus::Singular on zero diagonal.
     */
    template <bool UnitDiag = false, typename T, my_size_t N, my_size_t Ncols>
    Expected<FusedMatrix<T, N, Ncols>, MatrixStatus> back_substitute(
        const FusedMatrix<T, N, N> &U,
        const FusedMatrix<T, N, Ncols> &B)
    {
        static_assert(is_floating_point_v<T>,
                      "back_substitute requires a floating-point scalar type");

        FusedMatrix<T, N, Ncols> X(T(0));

        for (my_size_t j = 0; j < Ncols; ++j)
        {
            for (my_size_t i = N; i-- > 0;)
            {
                T sum = B(i, j);

                for (my_size_t k = i + 1; k < N; ++k)
                {
                    sum -= U(i, k) * X(k, j);
                }

                if constexpr (UnitDiag)
                {
                    X(i, j) = sum;
                }
                else
                {
                    T diag = U(i, i);

                    if (diag <= T(PRECISION_TOLERANCE) && diag >= T(-PRECISION_TOLERANCE))
                    {
                        return Unexpected{MatrixStatus::Singular};
                    }

                    X(i, j) = sum / diag;
                }
            }
        }

        return move(X);
    }

} // namespace matrix_algorithms

#endif // FUSED_ALGORITHMS_TRIANGULAR_SOLVE_H
