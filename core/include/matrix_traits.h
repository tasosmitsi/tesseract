#ifndef MATRIXTRAITS_H
#define MATRIXTRAITS_H

/**
 * @file matrix_traits.h
 * @brief Runtime property descriptors and error codes for matrices.
 */

namespace matrix_traits
{

    /**
     * @brief Describes the definiteness of a symmetric matrix.
     *
     * Determined at runtime (e.g. via Cholesky decomposition or eigenvalue
     * inspection). Ordered so that higher values imply
     * a strictly stronger condition.
     */
    enum class Definiteness
    {
        NotPositiveDefinite = 0,  ///< Neither positive definite nor semi-definite.
        PositiveSemiDefinite = 1, ///< All eigenvalues ≥ 0.
        PositiveDefinite = 2,     ///< All eigenvalues > 0.
    };

    /**
     * @brief Error codes for matrix decomposition and solver algorithms.
     *
     * Used as the error type in Expected<T, MatrixStatus> for matrix operations
     * that can fail (Cholesky, LU, solve, inverse, etc.).
     */
    enum class MatrixStatus : unsigned char
    {
        Ok = 0,              ///< Operation succeeded.
        NotPositiveDefinite, ///< Matrix is not positive definite (Cholesky).
        Singular,            ///< Matrix is singular or has a zero pivot (LU, solve, inverse).
        NearSingular,        ///< Condition number exceeds threshold (advisory).
        NotSymmetric,        ///< Matrix is not symmetric (Cholesky, Jacobi eigenvalue).
        NotConverged,        ///< Iterative method did not converge within max iterations.
        DimensionMismatch    ///< Operand dimensions are incompatible.
    };

} // namespace matrix_traits

#endif // MATRIXTRAITS_H
