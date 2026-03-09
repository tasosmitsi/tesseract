#ifndef MATRIXTRAITS_H
#define MATRIXTRAITS_H

/**
 * @file MatrixTraits.h
 * @brief Runtime property descriptors for matrices.
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

} // namespace matrix_traits

#endif // MATRIXTRAITS_H