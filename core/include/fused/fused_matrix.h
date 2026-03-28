#ifndef FUSEDMATRIX_H
#define FUSEDMATRIX_H

#include "simple_type_traits.h"
#include "fused/fused_tensor.h"
#include "algorithms/decomposition/cholesky.h"
#include "matrix_traits.h"
#include "math/math_utils.h"

template <typename T, my_size_t Rows, my_size_t Cols>
class FusedMatrix : public FusedTensorND<T, Rows, Cols>
{
public:
    DEFINE_TYPE_ALIAS(T, value_type);

    // Default constructor initializes a 2D matrix with default values
    FusedMatrix()
        : FusedTensorND<T, Rows, Cols>() {}

    // Constructor to initialize all elements to a specific value
    FusedMatrix(T initValue)
        : FusedTensorND<T, Rows, Cols>(initValue) {}

    // Copy constructor from another FusedMatrix (tested)
    FusedMatrix(const FusedMatrix &other)
        : FusedTensorND<T, Rows, Cols>(other)
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Copy constructor from another FusedMatrix", ErrorLevel::Info);
#endif
    }

    // Copy constructor from base class (tested)
    FusedMatrix(const FusedTensorND<T, Rows, Cols> &baseTensor)
        : FusedTensorND<T, Rows, Cols>(baseTensor)
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Copy constructor from base class FusedTensorND", ErrorLevel::Info);
#endif
    }

    // Move constructor from another FusedMatrix (tested)
    FusedMatrix(FusedMatrix &&other) noexcept
        : FusedTensorND<T, Rows, Cols>(move(other))
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Move constructor from another FusedMatrix", ErrorLevel::Info);
#endif
    }

    // Move constructor from base class (tested)
    FusedMatrix(FusedTensorND<T, Rows, Cols> &&baseTensor) noexcept
        : FusedTensorND<T, Rows, Cols>(move(baseTensor))
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Move constructor from base class FusedTensorND", ErrorLevel::Info);
#endif
    }

    // Constructor to initialize from a 2D array (tested)
    FusedMatrix(T (&initList)[Rows][Cols]) : FusedTensorND<T, Rows, Cols>()
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Constructor to initialize from a 2D array", ErrorLevel::Info);
#endif
        // loop trough the input and use (i,j) to store them
        for (my_size_t i = 0; i < Rows; ++i)
        {
            for (my_size_t j = 0; j < Cols; ++j)
            {
                (*this)(i, j) = initList[i][j];
            }
        }
    }

    /* ------ Transfomation funtions ------ */
    // Static method to create a FusedMatrix from a FusedTensorND (tested)
    static constexpr FusedMatrix moveFromTensor(FusedTensorND<T, Rows, Cols> &&tensor)
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Static method to create a FusedMatrix from a FusedTensorND", ErrorLevel::Info);
#endif
        return FusedMatrix(move(tensor));
    }

    // method to copy this as a FusedTensorND (tested)
    FusedTensorND<T, Rows, Cols> copyToTensor(void) const
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Copy a FusedMatrix to FusedTensorND", ErrorLevel::Info);
#endif
        // Cast to base class to ensure correct type
        return static_cast<const FusedTensorND<T, Rows, Cols> &>(*this);
    }

    // method to move this as a FusedTensorND (tested)
    FusedTensorND<T, Rows, Cols> moveToTensor(void)
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Move a FusedMatrix to FusedTensorND", ErrorLevel::Info);
#endif
        // Cast to base class to ensure correct type
        return FusedTensorND<T, Rows, Cols>(move(*this));
    }
    /*---------------------------------------*/

    // Assignment from expression template (tested)
    template <typename Expr>
    FusedMatrix &operator=(const BaseExpr<Expr> &expr)
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("FusedMatrix assignment from expression", ErrorLevel::Info);
#endif
        FusedTensorND<T, Rows, Cols>::operator=(expr);
        return *this;
    }

    // Copy assingment operators -------------------------------------------
    // Copy assignment from FusedTensorND (tested)
    FusedMatrix &operator=(const FusedTensorND<T, Rows, Cols> &other)
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("FusedMatrix copy assignment from FusedTensorND", ErrorLevel::Info);
#endif
        FusedTensorND<T, Rows, Cols>::operator=(other);
        return *this;
    }

    // Copy assignment from FusedMatrix (tested)
    FusedMatrix &operator=(const FusedMatrix &other)
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("FusedMatrix copy assignment from FusedMatrix", ErrorLevel::Info);
#endif
        // Call the base class operator= to assign the tensor
        FusedTensorND<T, Rows, Cols>::operator=(other);

        // Return the derived type
        return *this;
    }
    // ---------------------------------------------------------------------

    // Move assigment operators --------------------------------------------
    // Move assignment from FusedTensorND (tested)
    FusedMatrix &operator=(FusedTensorND<T, Rows, Cols> &&other) noexcept
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("FusedMatrix move assignment from FusedTensorND", ErrorLevel::Info);
#endif
        FusedTensorND<T, Rows, Cols>::operator=(move(other));
        return *this;
    }

    // Move assignment from FusedMatrix (tested)
    FusedMatrix &operator=(FusedMatrix<T, Rows, Cols> &&other) noexcept
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("FusedMatrix move assignment from FusedMatrix", ErrorLevel::Info);
#endif
        FusedTensorND<T, Rows, Cols>::operator=(move(other));
        return *this;
    }

    // Override operator= to assign a 2D array to the matrix (tested)
    FusedMatrix &operator=(T (&initList)[Rows][Cols])
    {
#ifdef DEBUG_FUSED_MATRIX
        MyErrorHandler::log("Assignment from a 2D array", ErrorLevel::Info);
#endif
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

    T &operator()(my_size_t i, my_size_t j)
    {
        return FusedTensorND<T, Rows, Cols>::operator()(i, j);
    }

    const T &operator()(my_size_t i, my_size_t j) const
    {
        return FusedTensorND<T, Rows, Cols>::operator()(i, j);
    }

    // Override setToZero to return a FusedMatrix
    FusedMatrix &setToZero(void)
    {
        // Call the base class setToZero to set all elements to zero
        FusedTensorND<T, Rows, Cols>::setToZero();

        // Cast the base class (FusedTensorND) to FusedMatrix to return the derived type
        return *this;
    }

    // Override setHomogen to return a FusedMatrix
    FusedMatrix &setHomogen(T _val)
    {
        // Call the base class setHomogen to set all elements to a specific value
        FusedTensorND<T, Rows, Cols>::setHomogen(_val);

        // Cast the base class (FusedTensorND) to FusedMatrix to return the derived type
        return *this;
    }

    // Override setRandom to return a FusedMatrix
    FusedMatrix &setRandom(my_size_t _maxRand, my_size_t _minRand)
    {
        // Call the base class setRandom to set all elements to random values
        FusedTensorND<T, Rows, Cols>::setRandom(_maxRand, _minRand);

        // Cast the base class (FusedTensorND) to FusedMatrix to return the derived type
        return *this;
    }

    // Override setDiagonal to return a FusedMatrix
    FusedMatrix &setDiagonal(T _val)
    {
        // Call the base class setDiagonal
        FusedTensorND<T, Rows, Cols>::setDiagonal(_val);

        // Cast the base class (FusedTensorND) to FusedMatrix to return the derived type
        return *this;
    }

    // Override setIdentity to return a FusedMatrix
    FusedMatrix &setIdentity(void)
    {
        // Call the base class setIdentity
        FusedTensorND<T, Rows, Cols>::setIdentity();

        // Cast the base class (FusedTensorND) to FusedMatrix to return the derived type
        return *this;
    }

    // Override setSequencial to return a FusedMatrix
    FusedMatrix &setSequencial(void)
    {
        // Call the base class setSequencial
        FusedTensorND<T, Rows, Cols>::setSequencial();

        // Cast the base class (FusedTensorND) to FusedMatrix to return the derived type
        return *this;
    }

    // matmul using einsum of parent class
    template <typename LeftExpr, typename RightExpr>
    static FusedMatrix<T, Rows, Cols> matmul(const BaseExpr<LeftExpr> &mat1, const BaseExpr<RightExpr> &mat2)
    {
        return {FusedTensorND<T, Rows, Cols>::einsum(mat1, mat2, 1, 0)};
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
                    if (math::abs((*this)(i, j) - T(1)) > T(PRECISION_TOLERANCE))
                    {
                        return false;
                    }
                }
                else
                {
                    if (math::abs((*this)(i, j)) > T(PRECISION_TOLERANCE))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool isSymmetric(void) const
    {
        // Check if the matrix is square
        if (!this->areDimsEqual())
        {
            MyErrorHandler::error("FusedMatrix is not square");
        }

        // use the fact that A = A^T for symmetric matrices
        // use transpose() to get the transpose of the matrix
        // and see if it's equal to the original matrix

        return (*this == this->transpose_view());
    }

    bool isUpperTriangular(void) const
    {
        my_size_t rows = this->getDim(0);
        my_size_t cols = this->getDim(1);

        for (my_size_t i = 1; i < rows; ++i)
        {
            // Below-diagonal entries: columns 0 to min(i, cols) - 1
            my_size_t jmax = (i < cols) ? i : cols;
            for (my_size_t j = 0; j < jmax; ++j)
            {
                if (math::abs((*this)(i, j)) > T(PRECISION_TOLERANCE))
                {
                    return false;
                }
            }
        }
        return true;
    }

    bool isLowerTriangular(void) const
    {
        my_size_t rows = this->getDim(0);
        my_size_t cols = this->getDim(1);

        for (my_size_t i = 0; i < rows; ++i)
        {
            for (my_size_t j = i + 1; j < cols; ++j)
            {
                if (math::abs((*this)(i, j)) > T(PRECISION_TOLERANCE))
                {
                    return false;
                }
            }
        }
        return true;
    }

    FusedMatrix upperTriangular(bool inplace = false)
    {
        // Check if the matrix is square
        if (!this->areDimsEqual())
        {
            MyErrorHandler::error("FusedMatrix is not square");
        }

        my_size_t matrix_size = this->getDim(0); // Assuming the matrix is square the number of rows and columns are equal

        if (!inplace)
        {
            // Create a copy and modify it
            // std::cout << "Setting matrix to upper triangular" << std::endl;
            FusedMatrix result = *this;
            for (my_size_t i = 1; i < matrix_size; i++)
            {
                for (my_size_t j = 0; j < i; j++)
                {
                    result(i, j) = T(0);
                }
            }
            return result;
        }
        else
        {
            // std::cout << "Setting matrix to upper triangular in place" << std::endl;
            // Modify the matrix in-place
            for (my_size_t i = 1; i < matrix_size; i++)
            {
                for (my_size_t j = 0; j < i; j++)
                {
                    (*this)(i, j) = T(0);
                }
            }
            return *this; // Returning the modified matrix itself
        }
    }

    FusedMatrix lowerTriangular(bool inplace = false)
    {
        // Check if the matrix is square
        if (!this->areDimsEqual())
        {
            MyErrorHandler::error("FusedMatrix is not square");
        }

        my_size_t matrix_size = this->getDim(0); // Assuming the matrix is square the number of rows and columns are equal

        if (!inplace)
        {
            // Create a copy and modify it
            // std::cout << "Setting matrix to lower triangular" << std::endl;
            FusedMatrix result = *this;
            for (my_size_t i = 0; i < matrix_size; i++)
            {
                for (my_size_t j = i + 1; j < matrix_size; j++)
                {
                    result(i, j) = T(0);
                }
            }
            return result;
        }
        else
        {
            // std::cout << "Setting matrix to lower triangular in place" << std::endl;
            // Modify the matrix in-place
            for (my_size_t i = 0; i < matrix_size; i++)
            {
                for (my_size_t j = i + 1; j < matrix_size; j++)
                {
                    (*this)(i, j) = T(0);
                }
            }
            return *this; // Returning the modified matrix itself
        }
    }

    /* Invers operation using Gauss-Jordan algorithm */
    FusedMatrix inverse(void) const
    {
        if (!this->areDimsEqual())
        {
            MyErrorHandler::error("FusedMatrix is non-invertible cause: not square");
        }

        // check if is identity
        if (this->isIdentity())
        {
            return *this;
        }

        FusedMatrix _outp = *this;
        FusedMatrix _temp = *this;

        my_size_t rows = _temp.getDim(0);
        my_size_t cols = _temp.getDim(1);
        _outp.setIdentity();

        /* Gauss Elimination... */
        for (my_size_t j = 0; j < rows - 1; j++)
        {
            for (my_size_t i = j + 1; i < rows; i++)
            {
                if (math::abs(_temp(j, j)) < T(PRECISION_TOLERANCE))
                {
                    /* FusedMatrix is non-invertible */
                    MyErrorHandler::error("FusedMatrix is non-invertible cause: diagonal element is zero (Gauss Elimination)");
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
                if (math::abs(_temp(j, j)) < T(PRECISION_TOLERANCE))
                {
                    /* FusedMatrix is non-invertible */
                    MyErrorHandler::error("FusedMatrix is non-invertible cause: diagonal element is zero (Jordan)");
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
            if (math::abs(_temp(i, i)) < T(PRECISION_TOLERANCE))
            {
                /* FusedMatrix is non-invertible */
                MyErrorHandler::error("FusedMatrix is non-invertible cause: diagonal element is zero (Normalization)");
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

    bool isOrthogonal(void)
    {
        // Check if the matrix is square
        if (!this->areDimsEqual())
        {
            MyErrorHandler::error("FusedMatrix is not square");
        }

        auto ident = FusedMatrix<T, Rows, Cols>::matmul(*this, this->transpose_view());
        if (!ident.isIdentity())
        {
            return false;
        }

        auto ident1 = FusedMatrix<T, Rows, Cols>::matmul(this->transpose_view(), *this);
        if (!ident1.isIdentity())
        {
            return false;
        }
        return true;
    }

    /**
     * @brief Determine if the matrix is positive definite, semi-definite, or neither.
     *
     * Uses a two-pass Cholesky approach:
     *   1. Strict pass (tol = PRECISION_TOLERANCE): if it succeeds, matrix is positive definite.
     *   2. Relaxed pass (tol = -PRECISION_TOLERANCE): if it succeeds, matrix is positive semi-definite
     *      (zero diagonals allowed, truly negative rejected).
     *   3. Both fail: matrix is not positive definite.
     *
     * @return Definiteness::PositiveDefinite, Definiteness::PositiveSemiDefinite,
     *         or Definiteness::NotPositiveDefinite.
     */
    matrix_traits::Definiteness isPositiveDefinite() const
    {
        // Strict: rejects near-zero diagonals
        auto strict = matrix_algorithms::cholesky(*this);
        if (strict.has_value())
        {
            return matrix_traits::Definiteness::PositiveDefinite;
        }

        // Relaxed: allows zero diagonals (semi-definite)
        // Negative tolerance lets exact-zero diagonals pass while
        // still rejecting truly negative ones.
        auto relaxed = matrix_algorithms::cholesky(*this, T(-PRECISION_TOLERANCE));
        if (relaxed.has_value())
        {
            return matrix_traits::Definiteness::PositiveSemiDefinite;
        }

        return matrix_traits::Definiteness::NotPositiveDefinite;
    }
};

#endif // FUSEDMATRIX_H
