#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

#include <cmath>

namespace matrix_algorithms
{
    template <typename MatrixType>
    MatrixType choleskyDecomposition(const MatrixType &matrix)
    {
        // Check if the matrix is symmetric, checking if it's square has been done in the isSymmetric function
        if (!matrix.isSymmetric())
        {
            MyErrorHandler::error("Matrix is not symmetric");
        }

        // Create a new zero-initialized matrix to store the result
        MatrixType result(0);

        // Perform the Cholesky decomposition
        for (my_size_t i = 0; i < matrix.getDim(0); ++i)
        {
            for (my_size_t j = 0; j <= i; ++j)
            {
                typename MatrixType::value_type sum = 0;
                for (my_size_t k = 0; k < j; ++k)
                {
                    sum += result(i, k) * result(j, k);
                }

                if (i == j) // Diagonal element
                {
                    typename MatrixType::value_type diag = matrix(i, i) - sum;
                    if (diag <= typename MatrixType::value_type(PRECISION_TOLERANCE))
                    {
                        MyErrorHandler::error("Matrix is not positive definite");
                    }
                    result(i, j) = std::sqrt(diag);
                }
                else // Off-diagonal element
                {
                    // static_assert(std::is_same_v<decltype(matrix(j, j)), double>, "result(j, j) is not double");
                    result(i, j) = (matrix(i, j) - sum) / result(j, j); //<-- this line
                }
            }
        }

        return result; // Result is lower triangular
    }
} // namespace matrix_algorithms

#endif // MATRIX_ALGORITHMS_H
