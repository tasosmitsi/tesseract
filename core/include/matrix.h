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

    // Override operator= to assign a matrix to another matrix
    Matrix &operator=(const Matrix &other)
    {
        // Call the base class operator= to assign the tensor
        TensorND<T, Rows, Cols>::operator=(other);

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

};

#endif // MATRIX_H