#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include "math/math_utils.h" // For math::sqrt

/**
 * @file MatrixAlgorithms.h
 * @brief Numerical algorithms operating on FusedMatrix types.
 */

namespace matrix_algorithms
{

    /**
     * @brief Compute the Cholesky decomposition of a symmetric positive-definite matrix.
     *
     * Decomposes @p matrix into a lower-triangular matrix L such that
     * A = L · Lᵀ. Triggers an error via MyErrorHandler if the input
     * is not symmetric or not positive definite.
     *
     * @tparam MatrixType A square FusedMatrix type exposing `isSymmetric()`,
     *                    `getDim()`, `operator()`, and `value_type`.
     * @param matrix Symmetric positive-definite input matrix.
     * @return Lower-triangular factor L.
     */
    template <typename MatrixType>
    MatrixType choleskyDecomposition(const MatrixType &matrix)
    {
        if (!matrix.isSymmetric())
        {
            MyErrorHandler::error("Matrix is not symmetric");
        }

        MatrixType result(0);

        for (my_size_t i = 0; i < matrix.getDim(0); ++i)
        {
            for (my_size_t j = 0; j <= i; ++j)
            {
                typename MatrixType::value_type sum = 0;
                for (my_size_t k = 0; k < j; ++k)
                {
                    sum += result(i, k) * result(j, k);
                }

                if (i == j)
                {
                    typename MatrixType::value_type diag = matrix(i, i) - sum;
                    if (diag <= typename MatrixType::value_type(PRECISION_TOLERANCE))
                    {
                        MyErrorHandler::error("Matrix is not positive definite");
                    }
                    result(i, j) = math::sqrt(diag);
                }
                else
                {
                    result(i, j) = (matrix(i, j) - sum) / result(j, j);
                }
            }
        }

        return result;
    }

} // namespace matrix_algorithms

#endif // MATRIX_ALGORITHMS_H