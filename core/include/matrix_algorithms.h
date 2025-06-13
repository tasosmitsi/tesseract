#ifndef MATRIX_ALGORITHMS_H
#define MATRIX_ALGORITHMS_H

// Forward declaration of Matrix class
template <typename T, my_size_t Rows, my_size_t Cols>
class Matrix;

namespace matrix_algorithms
{
    template <typename T, my_size_t Rows, my_size_t Cols>
    Matrix<T, Rows, Cols> choleskyDecomposition(Matrix<T, Rows, Cols> &matrix) // TODO: the matrix should be const
    {
        // Check if the matrix is symmetric, checking if it's square has been done in the isSymmetric function
        if (!matrix.isSymmetric())
        {
            throw std::runtime_error("Matrix is not symmetric");
        }

        // Create a new zero-initialized matrix to store the result
        Matrix<T, Rows, Cols> result(0);

        // Perform the Cholesky decomposition
        for (my_size_t i = 0; i < matrix.getDim(0); ++i)
        {
            for (my_size_t j = 0; j <= i; ++j)
            {
                T sum = 0;
                for (my_size_t k = 0; k < j; ++k)
                {
                    sum += result(i, k) * result(j, k);
                }

                if (i == j) // Diagonal element
                {
                    T diag = matrix(i, i) - sum;
                    if (diag <= T(PRECISION_TOLERANCE))
                    {
                        throw std::runtime_error("Matrix is not positive definite");
                    }
                    result(i, j) = std::sqrt(diag);
                }
                else // Off-diagonal element
                {
                    result(i, j) = (matrix(i, j) - sum) / result(j, j);
                }
            }
        }

        return result; // Result is lower triangular
    }
} // namespace matrix_algorithms

#endif // MATRIX_ALGORITHMS_H
