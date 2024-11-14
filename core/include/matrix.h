#ifndef MATRIX_H
#define MATRIX_H

#include "tensor.h"
#include <iostream>

// Derived class: Matrix
template <typename T, my_size_t Rows, my_size_t Cols>
class Matrix : public TensorND<T, Rows, Cols>
{
public:
    // Default constructor initializes a 2D matrix with default values
    Matrix() : TensorND<T, Rows, Cols>() {}

    // Constructor to initialize all elements to a specific value
    Matrix(T initValue) : TensorND<T, Rows, Cols>(initValue) {}

    // Copy constructor
    Matrix(const Matrix &other) : TensorND<T, Rows, Cols>(other) {}

    // Move constructor
    Matrix(Matrix &&other) noexcept : TensorND<T, Rows, Cols>(std::move(other)) {}

    Matrix(T (&initList)[Rows][Cols]) : TensorND<T, Rows, Cols>() 
    {
        // loop trough the input and use (i,j) to store them
        for (my_size_t i = 0; i < Rows; ++i)
        {
            for (my_size_t j = 0; j < Cols; ++j)
            {
                (*this)(i, j) = initList[i][j];
            }
        }
    }

    // Override operator= to assign a matrix to another matrix
    Matrix &operator=(const Matrix &other)
    {
        // Call the base class operator= to assign the tensor
        TensorND<T, Rows, Cols>::operator=(other);

        // Return the derived type
        return *this;
    }

    // Override operator= to assign a 2D array to the matrix
    Matrix &operator=(T (&initList)[Rows][Cols])
    {
        // loop trough the input and use (i,j) to store them
        for (my_size_t i = 0; i < Rows; ++i)
        {
            for (my_size_t j = 0; j < Cols; ++j)
            {
                (*this)(i, j) = initList[i][j];
            }
        }
        
        // Return the derived type
        return *this;
    }

    // Override operator+ to return a Matrix
    Matrix operator+(const Matrix &other) const
    {
        // Call the base class operator+ to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator+(other);

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // Override operator+ to add a scalar to the matrix
    Matrix operator+(const T scalar) const
    {
        // Call the base class operator+ to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator+(scalar);

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // overide operator+ to add a matrix to a scalar
    friend Matrix operator+(const T scalar, const Matrix &matrix)
    {
        return matrix + scalar;
    }

    // Override operator- to return a Matrix
    Matrix operator-(const Matrix &other) const
    {
        // Call the base class operator- to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator-(other);

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // Override operator- to subtract a scalar from the matrix
    Matrix operator-(const T scalar) const
    {
        // Call the base class operator- to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator-(scalar);

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // overide operator- to subtract a matrix from a scalar
    friend Matrix operator-(const T scalar, const Matrix &matrix)
    {
        return matrix - scalar;
    }

    // Override operator- to get the negative of the matrix
    Matrix operator-(void) const
    {
        // Call the base class operator- to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator-();

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // Override operator* to return a Matrix
    Matrix operator*(const Matrix &other) const
    {
        // Call the base class operator* to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator*(other);

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // Override operator* to multiply a scalar with the matrix
    Matrix operator*(const T scalar) const
    {
        // Call the base class operator* to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator*(scalar);

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // overide operator* to multiply a matrix with a scalar
    friend Matrix operator*(const T scalar, const Matrix &matrix)
    {
        return matrix * scalar;
    }

    // Override operator/ to return a Matrix
    Matrix operator/(const Matrix &other) const
    {
        // Call the base class operator/ to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator/(other);

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // Override operator/ to divide the matrix by a scalar
    Matrix operator/(const T scalar) const
    {
        // Call the base class operator/ to get a TensorND result
        TensorND<T, Rows, Cols> resultTensor = TensorND<T, Rows, Cols>::operator/(scalar);

        // Cast the result to Matrix and return
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    // overide operator/ to divide a scalar by the matrix
    friend Matrix operator/(const T scalar, const Matrix &matrix)
    {
        return matrix / scalar;
    }

    // Override transpose to return a Matrix
    Matrix& transpose() {
        // Call the base class transpose to perform the transpose operation
        TensorND<T, Rows, Cols>::transpose(); // Modifies transposeOrder_, not data_

        // Cast the base class (TensorND) to Matrix to return the derived type
        return *this;
    }

    // Override setToZero to return a Matrix
    Matrix& setToZero(void)
    {
        // Call the base class setToZero to set all elements to zero
        TensorND<T, Rows, Cols>::setToZero();

        // Cast the base class (TensorND) to Matrix to return the derived type
        return *this;
    }

    // Override setHomogen to return a Matrix
    Matrix& setHomogen(T _val)
    {
        // Call the base class setHomogen to set all elements to a specific value
        TensorND<T, Rows, Cols>::setHomogen(_val);

        // Cast the base class (TensorND) to Matrix to return the derived type
        return *this;
    }

    // Override setRandom to return a Matrix
    Matrix& setRandom(my_size_t _maxRand, my_size_t _minRand)
    {
        // Call the base class setRandom to set all elements to random values
        TensorND<T, Rows, Cols>::setRandom(_maxRand, _minRand);

        // Cast the base class (TensorND) to Matrix to return the derived type
        return *this;
    }

    // Override setDiagonal to return a Matrix
    Matrix& setDiagonal(T _val)
    {
        // Call the base class setDiagonal
        TensorND<T, Rows, Cols>::setDiagonal(_val);

        // Cast the base class (TensorND) to Matrix to return the derived type
        return *this;
    }

    // Override setIdentity to return a Matrix
    Matrix& setIdentity(void)
    {
        // Call the base class setIdentity
        TensorND<T, Rows, Cols>::setIdentity();

        // Cast the base class (TensorND) to Matrix to return the derived type
        return *this;
    }

    // Override setSequencial to return a Matrix
    Matrix& setSequencial(void)
    {
        // Call the base class setSequencial
        TensorND<T, Rows, Cols>::setSequencial();

        // Cast the base class (TensorND) to Matrix to return the derived type
        return *this;
    }

    // matmul using einsum of parent class
    template <my_size_t Common, my_size_t mat1_rows, my_size_t mat2_cols>
    static Matrix<T, Rows, Cols> matmul(const Matrix<T, mat1_rows, Common> &mat1, const Matrix<T, Common, mat2_cols> &mat2)
    {
        auto resultTensor = TensorND<T, Rows, Cols>::einsum(mat1, mat2, 1, 0);
        return static_cast<Matrix<T, Rows, Cols> &>(resultTensor);
    }

    bool isIdentity(void) const
    {
        // Check if the matrix is square
        if (!this->areDimsEqual())
        {
            return false;
        }

        // Check if the diagonal elements are 1 and the rest are 0
        for (my_size_t i = 0; i < this->getDim(0); i++)
        {
            for (my_size_t j = 0; j < this->getDim(1); j++)
            {
                if (i == j)
                {
                    if (std::abs((*this)(i, j) - T(1)) > T(PRECISION_TOLERANCE))
                    {
                        return false;
                    }
                }
                else
                {
                    if (std::abs((*this)(i, j)) > T(PRECISION_TOLERANCE))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    Matrix upperTriangular(bool inplace = false) {
    Matrix upperTriangular(bool inplace = false) 
    {
        // Check if the matrix is square
        if (!this->areDimsEqual())
        {
            throw std::runtime_error("Matrix is not square");
        }

        if (!inplace) {
            // Create a copy and modify it
            // std::cout << "Setting matrix to upper triangular" << std::endl;
            Matrix result = *this;
            for (my_size_t i = 1; i < result.getDim(0); i++)
            {
                my_size_t limit = (i < result.getDim(1)) ? i : result.getDim(1);

                // this assumes that the overloaded operator () 
                // handle memory in a contiguous manner.
                // If not, this will not work.
                std::fill(&result(i, 0), &result(i, limit), T(0));
            }
            return result;
        } else {
            // std::cout << "Setting matrix to upper triangular in place" << std::endl;
            // Modify the matrix in-place
            for (my_size_t i = 1; i < this->getDim(0); i++)
            {
                my_size_t limit = (i < this->getDim(1)) ? i : this->getDim(1);

                // this assumes that the overloaded operator () 
                // handle memory in a contiguous manner.
                // If not, this will not work.
                std::fill(&(*this)(i, 0), &(*this)(i, limit), T(0));
            }
            return *this;  // Returning the modified matrix itself
        }
    }

    /* Invers operation using Gauss-Jordan algorithm */
    Matrix inverse(void) const
    {
        if (!this->areDimsEqual())
        {
            throw std::runtime_error("Matrix is non-invertible cause: not square");
        }

        // check if is identity
        if (this->isIdentity())
        {
            return *this;
        }

        Matrix _outp = *this;
        Matrix _temp = *this;

        my_size_t rows = _temp.getDim(0);
        my_size_t cols = _temp.getDim(1);
        _outp.setIdentity();

        /* Gauss Elimination... */
        for (my_size_t j = 0; j < rows - 1; j++)
        {
            for (my_size_t i = j + 1; i < rows; i++)
            {
                if (std::abs((_temp(j, j)) < T(PRECISION_TOLERANCE)))
                {
                    /* Matrix is non-invertible */
                    throw std::runtime_error("Matrix is non-invertible cause: diagonal element is zero (Gauss Elimination)");
                }

                T tmp = _temp(i, j) / _temp(j, j);

                for (my_size_t k = 0; k < cols; k++)
                {
                    _temp(i, k) -= (_temp(j, k) * tmp);
                    _outp(i, k) -= (_outp(j, k) * tmp);

                    // round _temp(i, k) to zero if it's too small
                    // round _outp(i, k) to zero if it's too small
                }
            }
        }

        /* At this point, the _temp matrix should be an upper triangular matrix.
         * But because of rounding error, it might not.
         * Make it upper triangular by setting the lower triangular to zero.
         */
        _temp.upperTriangular(true);

        /* Jordan... */
        for (my_size_t j = rows - 1; j > 0; j--)
        {
            for (int i = j - 1; i >= 0; i--)
            {
                if (std::abs((_temp(j, j)) < T(PRECISION_TOLERANCE)))
                {
                    /* Matrix is non-invertible */
                    throw std::runtime_error("Matrix is non-invertible cause: diagonal element is zero (Jordan)");
                }

                T tmp = _temp(i, j) / _temp(j, j);
                _temp(i, j) -= (_temp(j, j) * tmp);

                // round _temp(i, j) to zero if it's too small

                for (int k = rows - 1; k >= 0; k--)
                {
                    _outp(i, k) -= (_outp(j, k) * tmp);

                    // round _outp(i, k) to zero if it's too small
                }
            }
        }

        /* Normalize the matrix */
        for (my_size_t i = 0; i < rows; i++)
        {
            if (std::abs((_temp(i, i)) < T(PRECISION_TOLERANCE)))
            {
                /* Matrix is non-invertible */
                throw std::runtime_error("Matrix is non-invertible cause: diagonal element is zero (Normalization)");
            }

            T tmp = _temp(i, i);
            _temp(i, i) = T(1.0);

            for (my_size_t j = 0; j < cols; j++)
            {
                _outp(i, j) /= tmp;
            }
        }
        return _outp;
    }
};

#endif // MATRIX_H