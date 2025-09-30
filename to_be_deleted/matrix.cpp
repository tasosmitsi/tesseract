/************************************************************************************
 * Class Matrix
 *  See matrix.h for description
 *
 ***********************************************************************************/

#include "matrix.h"
#include <iostream>


/* ================================================= definition below: ================================================= */
/* ================================================= definition below: ================================================= */

/* ------------------------------------------ Basic Matrix Class functions ------------------------------------------ */
/* ------------------------------------------ Basic Matrix Class functions ------------------------------------------ */

Matrix::Matrix(const int16_t _i16row, const int16_t _i16col, const InitializationType _init)
{
    this->i16row = _i16row;
    this->i16col = _i16col;

    if (_init == InitWithZero)
    {
        this->vSetHomogen(0.0);
    }
}

Matrix::Matrix(const int16_t _i16row, const int16_t _i16col, const float_prec *initData, const InitializationType _init)
{
    this->i16row = _i16row;
    this->i16col = _i16col;

    if (_init == InitWithZero)
    {
        this->vSetHomogen(0.0);
    }
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            (*this)(_i, _j) = *initData;
            initData++;
        }
    }
}

Matrix::Matrix(const Matrix &old_obj)
{
    /* For copy contructor, we only need to copy (_i16row x _i16col) submatrix, there's no need to copy all data */
    this->i16row = old_obj.i16row;
    this->i16col = old_obj.i16col;

    const float_prec *sourc = old_obj.floatData[0];
    float_prec *desti = this->floatData[0];

    for (int16_t _i = 0; _i < i16row; _i++)
    {
        /* Still valid with invalid matrix ((i16row == -1) or (i16col == -1)) */
        memcpy(desti, sourc, sizeof(float_prec) * size_t((this->i16col)));
        sourc += (MATRIX_MAXIMUM_SIZE);
        desti += (MATRIX_MAXIMUM_SIZE);
    }
}

Matrix &Matrix::operator=(const Matrix &obj)
{
    /* For assignment operator, we only need to copy (_i16row x _i16col) submatrix, there's no need to copy all data */
    this->i16row = obj.i16row;
    this->i16col = obj.i16col;

    const float_prec *sourc = obj.floatData[0];
    float_prec *desti = this->floatData[0];

    for (int16_t _i = 0; _i < i16row; _i++)
    {
        /* Still valid with invalid matrix ((i16row == -1) or (i16col == -1)) */
        memcpy(desti, sourc, sizeof(float_prec) * size_t((this->i16col)));
        sourc += (MATRIX_MAXIMUM_SIZE);
        desti += (MATRIX_MAXIMUM_SIZE);
    }

    return *this;
}

Matrix::~Matrix(void)
{
    /* haven't implemented anything here, definition needed for the rule of three */
    /* (´・ω・`) */
}

/* ---------------------------------------- Matrix entry accessing functions ---------------------------------------- */

/* The preferred method to access the matrix data (boring code) */
float_prec &Matrix::operator()(const int16_t _row, const int16_t _col)
{
#if (defined(MATRIX_USE_BOUNDS_CHECKING))
    ASSERT((_row >= 0) && (_row < this->i16row) && (_row < MATRIX_MAXIMUM_SIZE),
           "Matrix index out-of-bounds (at row evaluation)");
    ASSERT((_col >= 0) && (_col < this->i16col) && (_col < MATRIX_MAXIMUM_SIZE),
           "Matrix index out-of-bounds (at column _column)");
#else
#warning("Matrix bounds checking is disabled... good luck >:3");
#endif
    return this->floatData[_row][_col];
}

float_prec Matrix::operator()(const int16_t _row, const int16_t _col) const
{
#if (defined(MATRIX_USE_BOUNDS_CHECKING))
    ASSERT((_row >= 0) && (_row < this->i16row) && (_row < MATRIX_MAXIMUM_SIZE),
           "Matrix index out-of-bounds (at row evaluation)");
    ASSERT((_col >= 0) && (_col < this->i16col) && (_col < MATRIX_MAXIMUM_SIZE),
           "Matrix index out-of-bounds (at column _column)");
#else
#warning("Matrix bounds checking is disabled... good luck >:3");
#endif
    return this->floatData[_row][_col];
}

/* Ref: https://stackoverflow.com/questions/6969881/operator-overload
 * Modified to be lvalue modifiable (I know this is so dirty, but it makes the code so FABULOUS XD)
 */
float_prec &Matrix::Proxy::operator[](const int16_t _col)
{
#if (defined(MATRIX_USE_BOUNDS_CHECKING))
    ASSERT((_col >= 0) && (_col < this->_maxCol) && (_col < MATRIX_MAXIMUM_SIZE),
           "Matrix index out-of-bounds (at column evaluation)");
#else
#warning("Matrix bounds checking is disabled... good luck >:3");
#endif
    return _array.ptr[_col];
}

float_prec Matrix::Proxy::operator[](const int16_t _col) const
{
#if (defined(MATRIX_USE_BOUNDS_CHECKING))
    ASSERT((_col >= 0) && (_col < this->_maxCol) && (_col < MATRIX_MAXIMUM_SIZE),
           "Matrix index out-of-bounds (at column evaluation)");
#else
#warning("Matrix bounds checking is disabled... good luck >:3");
#endif
    return _array.cptr[_col];
}

Matrix::Proxy Matrix::operator[](const int16_t _row)
{
#if (defined(MATRIX_USE_BOUNDS_CHECKING))
    ASSERT((_row >= 0) && (_row < this->i16row) && (_row < MATRIX_MAXIMUM_SIZE),
           "Matrix index out-of-bounds (at row evaluation)");
#else
#warning("Matrix bounds checking is disabled... good luck >:3");
#endif
    return Proxy(floatData[_row], this->i16col); /* Parsing column index for bound checking */
}
const Matrix::Proxy Matrix::operator[](const int16_t _row) const
{
#if (defined(MATRIX_USE_BOUNDS_CHECKING))
    ASSERT((_row >= 0) && (_row < this->i16row) && (_row < MATRIX_MAXIMUM_SIZE),
           "Matrix index out-of-bounds (at row evaluation)");
#else
#warning("Matrix bounds checking is disabled... good luck >:3");
#endif
    return Proxy(floatData[_row], this->i16col); /* Parsing column index for bound checking */
}

/* -------------------------------------- Matrix checking function declaration -------------------------------------- */

bool Matrix::bMatrixIsValid(void)
{
    /* Check whether the matrix is valid or not */
    if ((this->i16row > 0) && (this->i16row <= MATRIX_MAXIMUM_SIZE) &&
        (this->i16col > 0) && (this->i16col <= MATRIX_MAXIMUM_SIZE))
    {
        return true;
    }
    else
    {
        return false;
    }
}

void Matrix::vSetMatrixInvalid(void)
{
    this->i16row = -1;
    this->i16col = -1;
}

bool Matrix::bMatrixIsSquare(void)
{
    return (this->i16row == this->i16col);
}

/* ------------------------------------------ Matrix elementary operations ------------------------------------------ */
/* TODO: We could do loop unrolling here for elementary, simple, and matrix insertion operations. It *might* speed up
 *        the computation time up to 20-30% for processor with FMAC and cached CPU (I still mull on this because
 *        the code will be awful).
 */

bool Matrix::operator==(const Matrix &_compare) const
{
    if ((this->i16row != _compare.i16row) || (this->i16col != _compare.i16col))
    {
        return false;
    }

    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            if (fabs((*this)(_i, _j) - _compare(_i, _j)) > float_prec(float_prec_ZERO_ECO))
            {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix &_compare) const
{
    return (!(*this == _compare));
}

Matrix Matrix::operator-(void) const
{
    Matrix _outp(this->i16row, this->i16col, InitWithoutZero);

    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _outp(_i, _j) = -(*this)(_i, _j);
        }
    }
    return _outp;
}

Matrix Matrix::operator+(const float_prec _scalar) const
{
    Matrix _outp(this->i16row, this->i16col, Matrix::InitWithoutZero);

    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _outp(_i, _j) = (*this)(_i, _j) + _scalar;
        }
    }
    return _outp;
}

Matrix Matrix::operator-(const float_prec _scalar) const
{
    Matrix _outp(this->i16row, this->i16col, Matrix::InitWithoutZero);

    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _outp(_i, _j) = (*this)(_i, _j) - _scalar;
        }
    }
    return _outp;
}

Matrix Matrix::operator*(const float_prec _scalar) const
{
    Matrix _outp(this->i16row, this->i16col, Matrix::InitWithoutZero);

    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _outp(_i, _j) = (*this)(_i, _j) * _scalar;
        }
    }
    return _outp;
}

Matrix Matrix::operator/(const float_prec _scalar) const
{
    Matrix _outp(this->i16row, this->i16col, Matrix::InitWithoutZero);

    if (fabs(_scalar) < float_prec(float_prec_ZERO_ECO))
    {
        _outp.vSetMatrixInvalid();
        return _outp;
    }
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _outp(_i, _j) = (*this)(_i, _j) / _scalar;
        }
    }
    return _outp;
}

Matrix operator+(const float_prec _scalar, const Matrix &_mat)
{
    Matrix _outp(_mat.i16getRow(), _mat.i16getCol(), Matrix::InitWithoutZero);

    for (int16_t _i = 0; _i < _mat.i16getRow(); _i++)
    {
        for (int16_t _j = 0; _j < _mat.i16getCol(); _j++)
        {
            _outp(_i, _j) = _scalar + _mat(_i, _j);
        }
    }
    return _outp;
}

Matrix operator-(const float_prec _scalar, const Matrix &_mat)
{
    Matrix _outp(_mat.i16getRow(), _mat.i16getCol(), Matrix::InitWithoutZero);

    for (int16_t _i = 0; _i < _mat.i16getRow(); _i++)
    {
        for (int16_t _j = 0; _j < _mat.i16getCol(); _j++)
        {
            _outp(_i, _j) = _scalar - _mat(_i, _j);
        }
    }
    return _outp;
}

Matrix operator*(const float_prec _scalar, const Matrix &_mat)
{
    Matrix _outp(_mat.i16getRow(), _mat.i16getCol(), Matrix::InitWithoutZero);

    for (int16_t _i = 0; _i < _mat.i16getRow(); _i++)
    {
        for (int16_t _j = 0; _j < _mat.i16getCol(); _j++)
        {
            _outp(_i, _j) = _scalar * _mat(_i, _j);
        }
    }
    return _outp;
}

Matrix Matrix::operator+(const Matrix &_matAdd) const
{
    Matrix _outp(this->i16row, this->i16col, InitWithoutZero);
    if ((this->i16row != _matAdd.i16row) || (this->i16col != _matAdd.i16col))
    {
        _outp.vSetMatrixInvalid();
        return _outp;
    }

    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _outp(_i, _j) = (*this)(_i, _j) + _matAdd(_i, _j);
        }
    }
    return _outp;
}

Matrix Matrix::operator-(const Matrix &_matSub) const
{
    Matrix _outp(this->i16row, this->i16col, InitWithoutZero);
    if ((this->i16row != _matSub.i16row) || (this->i16col != _matSub.i16col))
    {
        _outp.vSetMatrixInvalid();
        return _outp;
    }

    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _outp(_i, _j) = (*this)(_i, _j) - _matSub(_i, _j);
        }
    }
    return _outp;
}

Matrix Matrix::operator*(const Matrix &_matMul) const
{
    Matrix _outp(this->i16row, _matMul.i16col, InitWithoutZero);
    if ((this->i16col != _matMul.i16row))
    {
        _outp.vSetMatrixInvalid();
        return _outp;
    }

    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < _matMul.i16col; _j++)
        {
            _outp(_i, _j) = 0.0;
            for (int16_t _k = 0; _k < this->i16col; _k++)
            {
                _outp(_i, _j) += ((*this)(_i, _k) * _matMul(_k, _j));
            }
        }
    }
    return _outp;
}

/* -------------------------------------------- Simple Matrix operations -------------------------------------------- */
/* -------------------------------------------- Simple Matrix operations -------------------------------------------- */

void Matrix::vRoundingElementToZero(const int16_t _i, const int16_t _j)
{
    if (fabs((*this)(_i, _j)) < float_prec(float_prec_ZERO_ECO))
    {
        (*this)(_i, _j) = 0.0;
    }
}

Matrix Matrix::RoundingMatrixToZero(void)
{
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            if (fabs((*this)(_i, _j)) < float_prec(float_prec_ZERO_ECO))
            {
                (*this)(_i, _j) = 0.0;
            }
        }
    }
    return (*this);
}

void Matrix::vSetHomogen(const float_prec _val)
{
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            (*this)(_i, _j) = _val;
        }
    }
}

void Matrix::vSetToZero(void)
{
    this->vSetHomogen(0.0);
}

void Matrix::vSetRandom(const int32_t _maxRand, const int32_t _minRand)
{
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            (*this)(_i, _j) = float_prec((rand() % (_maxRand - _minRand + 1)) + _minRand);
        }
    }
}

void Matrix::vSetDiag(const float_prec _val)
{
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            if (_i == _j)
            {
                (*this)(_i, _j) = _val;
            }
            else
            {
                (*this)(_i, _j) = 0.0;
            }
        }
    }
}

void Matrix::vSetIdentity(void)
{
    this->vSetDiag(1.0);
}

Matrix MatIdentity(const int16_t _i16size)
{
    Matrix _outp(_i16size, _i16size, Matrix::InitWithoutZero);
    _outp.vSetDiag(1.0);
    return _outp;
}

/* Return the transpose of the matrix */
Matrix Matrix::Transpose(void)
{
    Matrix _outp(this->i16col, this->i16row, InitWithoutZero);
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _outp(_j, _i) = (*this)(_i, _j);
        }
    }
    return _outp;
}

/* Normalize the vector */
bool Matrix::bNormVector(void)
{
    float_prec _normM = 0.0;
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            _normM = _normM + ((*this)(_i, _j) * (*this)(_i, _j));
        }
    }

    /* Rounding to zero to avoid case where sqrt(0-), and _normM always positive */
    if (_normM < float_prec(float_prec_ZERO))
    {
        return false;
    }
    _normM = sqrt(_normM);
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            (*this)(_i, _j) /= _normM;
        }
    }
    return true;
}

/* --------------------------------------- Matrix/Vector insertion operations --------------------------------------- */
/* --------------------------------------- Matrix/Vector insertion operations --------------------------------------- */

/* Insert vector into matrix at _posCol position
 * Example: A = Matrix 3x3, B = Vector 3x1
 *
 *  C = A.InsertVector(B, 1);
 *
 *  A = [A00  A01  A02]     B = [B00]
 *      [A10  A11  A12]         [B10]
 *      [A20  A21  A22]         [B20]
 *
 *  C = [A00  B00  A02]
 *      [A10  B10  A12]
 *      [A20  B20  A22]
 */
Matrix Matrix::InsertVector(const Matrix &_Vector, const int16_t _posCol)
{
    Matrix _outp(*this);
    if ((_Vector.i16row > this->i16row) || (_Vector.i16col + _posCol > this->i16col))
    {
        /* Return false */
        _outp.vSetMatrixInvalid();
        return _outp;
    }
    for (int16_t _i = 0; _i < _Vector.i16row; _i++)
    {
        _outp(_i, _posCol) = _Vector(_i, 0);
    }
    return _outp;
}

/* Insert submatrix into matrix at _posRow & _posCol position
 * Example: A = Matrix 4x4, B = Matrix 2x3
 *
 *  C = A.InsertSubMatrix(B, 1, 1);
 *
 *  A = [A00  A01  A02  A03]    B = [B00  B01  B02]
 *      [A10  A11  A12  A13]        [B10  B11  B12]
 *      [A20  A21  A22  A23]
 *      [A30  A31  A32  A33]
 *
 *
 *  C = [A00  A01  A02  A03]
 *      [A10  B00  B01  B02]
 *      [A20  B10  B11  B12]
 *      [A30  A31  A32  A33]
 */
Matrix Matrix::InsertSubMatrix(const Matrix &_subMatrix, const int16_t _posRow, const int16_t _posCol)
{
    Matrix _outp(*this);
    if (((_subMatrix.i16row + _posRow) > this->i16row) || ((_subMatrix.i16col + _posCol) > this->i16col))
    {
        /* Return false */
        _outp.vSetMatrixInvalid();
        return _outp;
    }
    for (int16_t _i = 0; _i < _subMatrix.i16row; _i++)
    {
        for (int16_t _j = 0; _j < _subMatrix.i16col; _j++)
        {
            _outp(_i + _posRow, _j + _posCol) = _subMatrix(_i, _j);
        }
    }
    return _outp;
}

/* Insert the first _lenRow-th and first _lenColumn-th submatrix into matrix;
 *  at the matrix's _posRow and _posCol position.
 *
 * Example: A = Matrix 4x4, B = Matrix 2x3
 *
 *  C = A.InsertSubMatrix(B, 1, 1, 2, 2);
 *
 *  A = [A00  A01  A02  A03]    B = [B00  B01  B02]
 *      [A10  A11  A12  A13]        [B10  B11  B12]
 *      [A20  A21  A22  A23]
 *      [A30  A31  A32  A33]
 *
 *
 *  C = [A00  A01  A02  A03]
 *      [A10  B00  B01  A13]
 *      [A20  B10  B11  A23]
 *      [A30  A31  A32  A33]
 */
Matrix Matrix::InsertSubMatrix(const Matrix &_subMatrix, const int16_t _posRow, const int16_t _posCol,
                                      const int16_t _lenRow, const int16_t _lenColumn)
{
    Matrix _outp(*this);
    if (((_lenRow + _posRow) > this->i16row) || ((_lenColumn + _posCol) > this->i16col) ||
        (_lenRow > _subMatrix.i16row) || (_lenColumn > _subMatrix.i16col))
    {
        /* Return false */
        _outp.vSetMatrixInvalid();
        return _outp;
    }
    for (int16_t _i = 0; _i < _lenRow; _i++)
    {
        for (int16_t _j = 0; _j < _lenColumn; _j++)
        {
            _outp(_i + _posRow, _j + _posCol) = _subMatrix(_i, _j);
        }
    }
    return _outp;
}

/* Insert the _lenRow & _lenColumn submatrix, start from _posRowSub & _posColSub submatrix;
 *  into matrix at the matrix's _posRow and _posCol position.
 *
 * Example: A = Matrix 4x4, B = Matrix 2x3
 *
 *  C = A.InsertSubMatrix(B, 1, 1, 0, 1, 1, 2);
 *
 *  A = [A00  A01  A02  A03]    B = [B00  B01  B02]
 *      [A10  A11  A12  A13]        [B10  B11  B12]
 *      [A20  A21  A22  A23]
 *      [A30  A31  A32  A33]
 *
 *  C = [A00  A01  A02  A03]
 *      [A10  B01  B02  A13]
 *      [A20  A21  A22  A23]
 *      [A30  A31  A32  A33]
 */
Matrix Matrix::InsertSubMatrix(const Matrix &_subMatrix, const int16_t _posRow, const int16_t _posCol,
                                      const int16_t _posRowSub, const int16_t _posColSub,
                                      const int16_t _lenRow, const int16_t _lenColumn)
{
    Matrix _outp(*this);
    if (((_lenRow + _posRow) > this->i16row) || ((_lenColumn + _posCol) > this->i16col) ||
        ((_posRowSub + _lenRow) > _subMatrix.i16row) || ((_posColSub + _lenColumn) > _subMatrix.i16col))
    {
        /* Return false */
        _outp.vSetMatrixInvalid();
        return _outp;
    }
    for (int16_t _i = 0; _i < _lenRow; _i++)
    {
        for (int16_t _j = 0; _j < _lenColumn; _j++)
        {
            _outp(_i + _posRow, _j + _posCol) = _subMatrix(_posRowSub + _i, _posColSub + _j);
        }
    }
    return _outp;
}

/* ------------------------------------------------- Big operations ------------------------------------------------- */
/* ------------------------------------------------- Big operations ------------------------------------------------- */

/* Invers operation using Gauss-Jordan algorithm */
Matrix Matrix::Invers(void) const
{
    Matrix _outp(this->i16row, this->i16col, InitWithoutZero);
    Matrix _temp(*this);
    _outp.vSetIdentity();

    /* Gauss Elimination... */
    for (int16_t _j = 0; _j < (_temp.i16row) - 1; _j++)
    {
        for (int16_t _i = _j + 1; _i < _temp.i16row; _i++)
        {
            if (fabs(_temp(_j, _j)) < float_prec(float_prec_ZERO))
            {
                /* Matrix is non-invertible */
                _outp.vSetMatrixInvalid();
                return _outp;
            }

            float_prec _tempfloat = _temp(_i, _j) / _temp(_j, _j);

            for (int16_t _k = 0; _k < _temp.i16col; _k++)
            {
                _temp(_i, _k) -= (_temp(_j, _k) * _tempfloat);
                _outp(_i, _k) -= (_outp(_j, _k) * _tempfloat);

                _temp.vRoundingElementToZero(_i, _k);
                _outp.vRoundingElementToZero(_i, _k);
            }
        }
    }

#if (1)
    /* Here, the _temp matrix should be an upper triangular matrix.
     * But because of rounding error, it might not.
     */
    for (int16_t _i = 1; _i < _temp.i16row; _i++)
    {
        for (int16_t _j = 0; _j < _i; _j++)
        {
            _temp(_i, _j) = 0.0;
        }
    }
#endif

    /* Jordan... */
    for (int16_t _j = (_temp.i16row) - 1; _j > 0; _j--)
    {
        for (int16_t _i = _j - 1; _i >= 0; _i--)
        {
            if (fabs(_temp(_j, _j)) < float_prec(float_prec_ZERO))
            {
                /* Matrix is non-invertible */
                _outp.vSetMatrixInvalid();
                return _outp;
            }

            float_prec _tempfloat = _temp(_i, _j) / _temp(_j, _j);
            _temp(_i, _j) -= (_temp(_j, _j) * _tempfloat);
            _temp.vRoundingElementToZero(_i, _j);

            for (int16_t _k = (_temp.i16row - 1); _k >= 0; _k--)
            {
                _outp(_i, _k) -= (_outp(_j, _k) * _tempfloat);
                _outp.vRoundingElementToZero(_i, _k);
            }
        }
    }

    /* Normalization */
    for (int16_t _i = 0; _i < _temp.i16row; _i++)
    {
        if (fabs(_temp(_i, _i)) < float_prec(float_prec_ZERO))
        {
            /* Matrix is non-invertible */
            _outp.vSetMatrixInvalid();
            return _outp;
        }

        float_prec _tempfloat = _temp(_i, _i);
        _temp(_i, _i) = 1.0;

        for (int16_t _j = 0; _j < _temp.i16row; _j++)
        {
            _outp(_i, _j) /= _tempfloat;
        }
    }
    return _outp;
}

/* Use elemetary row operation to reduce the matrix into upper triangular form
 *  (like in the first phase of gauss-jordan algorithm).
 *
 * Useful if we want to check whether the matrix is positive definite or not
 *  (useful before calling CholeskyDec function).
 */
bool Matrix::bMatrixIsPositiveDefinite(const bool checkPosSemidefinite) const
{
    bool _posDef, _posSemiDef;
    Matrix _temp(*this);

    /* Gauss Elimination... */
    for (int16_t _j = 0; _j < (_temp.i16row) - 1; _j++)
    {
        for (int16_t _i = _j + 1; _i < _temp.i16row; _i++)
        {
            if (fabs(_temp(_j, _j)) < float_prec(float_prec_ZERO))
            {
                /* Q: Do we still need to check this?
                 * A: idk, it's 3 AM. I need sleep :<
                 *
                 * NOTE TO FUTURE SELF: Confirm it!
                 */
                return false;
            }

            float_prec _tempfloat = _temp(_i, _j) / _temp(_j, _j);

            for (int16_t _k = 0; _k < _temp.i16col; _k++)
            {
                _temp(_i, _k) -= (_temp(_j, _k) * _tempfloat);
                _temp.vRoundingElementToZero(_i, _k);
            }
        }
    }

    _posDef = true;
    _posSemiDef = true;
    for (int16_t _i = 0; _i < _temp.i16row; _i++)
    {
        if (_temp(_i, _i) < float_prec(float_prec_ZERO))
        {
            /* false if less than 0+ (zero included) */
            _posDef = false;
        }
        if (_temp(_i, _i) < -float_prec(float_prec_ZERO))
        {
            /* false if less than 0- (zero is not included) */
            _posSemiDef = false;
        }
    }

    if (checkPosSemidefinite)
    {
        return _posSemiDef;
    }
    else
    {
        return _posDef;
    }
}

/* For square matrix 'this' with size MxM, return vector Mx1 with entries
 * correspond with diagonal entries of 'this'.
 *
 *   Example:    this = [a11 a12 a13]
 *                      [a21 a22 a23]
 *                      [a31 a32 a33]
 *
 * out = this.GetDiagonalEntries() = [a11]
 *                                   [a22]
 *                                   [a33]
 */
Matrix Matrix::GetDiagonalEntries(void) const
{
    Matrix _temp(this->i16row, 1, InitWithoutZero);

    if (this->i16row != this->i16col)
    {
        _temp.vSetMatrixInvalid();
        return _temp;
    }
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        _temp(_i, 0) = (*this)(_i, _i);
    }
    return _temp;
}

/* Do the Cholesky Decomposition using Cholesky-Crout algorithm.
 *
 *      A = L*L'     ; A = real, positive definite, and symmetry MxM matrix
 *
 *      L = A.CholeskyDec();
 *
 *      CATATAN! NOTE! The symmetry property is not checked at the beginning to lower
 *          the computation cost. The processing is being done on the lower triangular
 *          component of _A. Then it is assumed the upper triangular is inherently
 *          equal to the lower end.
 *          (as a side note, Scilab & MATLAB is using Lapack routines DPOTRF that process
 *           the upper triangular of _A. The result should be equal mathematically if A
 *           is symmetry).
 */
Matrix Matrix::CholeskyDec(void) const
{
    float_prec _tempFloat;

    /* Note that _outp need to be initialized as zero matrix */
    Matrix _outp(this->i16row, this->i16col, InitWithZero);

    if (this->i16row != this->i16col)
    {
        _outp.vSetMatrixInvalid();
        return _outp;
    }
    for (int16_t _j = 0; _j < this->i16col; _j++)
    {
        for (int16_t _i = _j; _i < this->i16row; _i++)
        {
            _tempFloat = (*this)(_i, _j);
            if (_i == _j)
            {
                for (int16_t _k = 0; _k < _j; _k++)
                {
                    _tempFloat = _tempFloat - (_outp(_i, _k) * _outp(_i, _k));
                }
                if (_tempFloat < -float_prec(float_prec_ZERO))
                {
                    /* Matrix is not positif (semi)definit */
                    _outp.vSetMatrixInvalid();
                    return _outp;
                }
                /* Rounding to zero to avoid case where sqrt(0-) */
                if (fabs(_tempFloat) < float_prec(float_prec_ZERO))
                {
                    _tempFloat = 0.0;
                }
                _outp(_i, _i) = sqrt(_tempFloat);
            }
            else
            {
                for (int16_t _k = 0; _k < _j; _k++)
                {
                    _tempFloat = _tempFloat - (_outp(_i, _k) * _outp(_j, _k));
                }
                if (fabs(_outp(_j, _j)) < float_prec(float_prec_ZERO))
                {
                    /* Matrix is not positif definit */
                    _outp.vSetMatrixInvalid();
                    return _outp;
                }
                _outp(_i, _j) = _tempFloat / _outp(_j, _j);
            }
        }
    }
    return _outp;
}

/* Do the Householder Transformation for QR Decomposition operation.
 *              out = HouseholderTransformQR(A, i, j)
 */
Matrix Matrix::HouseholderTransformQR(const int16_t _rowTransform, const int16_t _colTransform)
{
    float_prec _tempFloat;
    float_prec _xLen;
    float_prec _x1;
    float_prec _u1;
    float_prec _vLen2;

    /* Note that _outp & _vectTemp need to be initialized as zero matrix */
    Matrix _outp(this->i16row, this->i16row, InitWithZero);
    Matrix _vectTemp(this->i16row, 1, InitWithZero);

    if ((_rowTransform >= this->i16row) || (_colTransform >= this->i16col))
    {
        _outp.vSetMatrixInvalid();
        return _outp;
    }

    /* Until here:
     *
     * _xLen    = ||x||            = sqrt(x1^2 + x2^2 + .. + xn^2)
     * _vLen2   = ||u||^2 - [u1^2] = x2^2 + .. + xn^2
     * _vectTemp= [0 0 0 .. x1=0 x2 x3 .. xn]'
     */
    _x1 = (*this)(_rowTransform, _colTransform);
    _xLen = _x1 * _x1;
    _vLen2 = 0.0;
    for (int16_t _i = _rowTransform + 1; _i < this->i16row; _i++)
    {
        _vectTemp(_i, 0) = (*this)(_i, _colTransform);

        _tempFloat = _vectTemp(_i, 0) * _vectTemp(_i, 0);
        _xLen += _tempFloat;
        _vLen2 += _tempFloat;
    }
    _xLen = sqrt(_xLen);

    /* u1    = x1+(-sign(x1))*xLen */
    if (_x1 < 0.0)
    {
        _u1 = _x1 + _xLen;
    }
    else
    {
        _u1 = _x1 - _xLen;
    }

    /* Solve vlen2 & tempHH */
    _vLen2 += (_u1 * _u1);
    _vectTemp(_rowTransform, 0) = _u1;

    if (fabs(_vLen2) < float_prec(float_prec_ZERO_ECO))
    {
        /* x vector is collinear with basis vector e, return result = I */
        _outp.vSetIdentity();
    }
    else
    {
        /* P = -2*(u1*u1')/v_len2 + I */
        /* PR TODO: We need to investigate more on this */
        for (int16_t _i = 0; _i < this->i16row; _i++)
        {
            _tempFloat = _vectTemp(_i, 0);
            if (fabs(_tempFloat) > float_prec(float_prec_ZERO))
            {
                for (int16_t _j = 0; _j < this->i16row; _j++)
                {
                    if (fabs(_vectTemp(_j, 0)) > float_prec(float_prec_ZERO))
                    {
                        _outp(_i, _j) = _vectTemp(_j, 0);
                        _outp(_i, _j) = _outp(_i, _j) * _tempFloat;
                        _outp(_i, _j) = _outp(_i, _j) * (-2.0 / _vLen2);
                    }
                }
            }
            _outp(_i, _i) = _outp(_i, _i) + 1.0;
        }
    }
    return _outp;
}

/* Do the QR Decomposition for matrix using Householder Transformation.
 *                      A = Q*R
 *
 * PERHATIAN! CAUTION! The matrix calculated by this function are Q' and R (Q transpose and R).
 *  Because QR Decomposition usually used to calculate solution for least-squares equation
 *  (that need Q'), we don't do the transpose of Q inside this routine to lower the
 *  computation cost (user need to transpose outside if they want Q).
 *
 * Example of using QRDec to solve least-squares:
 *                      Ax = b
 *                   (QR)x = b
 *                      Rx = Q'b    --> Afterward use back-subtitution to solve x
 */
bool Matrix::QRDec(Matrix &Qt, Matrix &R) const
{
    Matrix Qn(Qt.i16row, Qt.i16col, InitWithoutZero);
    if ((this->i16row < this->i16col) || (!Qt.bMatrixIsSquare()) || (Qt.i16row != this->i16row) ||
        (R.i16row != this->i16row) || (R.i16col != this->i16col))
    {
        Qt.vSetMatrixInvalid();
        R.vSetMatrixInvalid();
        return false;
    }
    R = (*this);
    Qt.vSetIdentity();
    for (int16_t _i = 0; (_i < (this->i16row - 1)) && (_i < this->i16col - 1); _i++)
    {
        Qn = R.HouseholderTransformQR(_i, _i);
        if (!Qn.bMatrixIsValid())
        {
            Qt.vSetMatrixInvalid();
            R.vSetMatrixInvalid();
            return false;
        }
        Qt = Qn * Qt;
        R = Qn * R;
    }
#if (0)
    Qt.RoundingMatrixToZero();
    R.RoundingMatrixToZero();
#else
    Qt.RoundingMatrixToZero();
    for (int16_t _i = 1; ((_i < R.i16row) && (_i < R.i16col)); _i++)
    {
        for (int16_t _j = 0; _j < _i; _j++)
        {
            R(_i, _j) = 0.0;
        }
    }
#endif

    /* Q = Qt.Transpose */
    return true;
}

/* Do the back-subtitution operation for upper triangular matrix A & column matrix B to solve x:
 *                      Ax = B
 *
 * x = BackSubtitution(&A, &B);
 *
 * CATATAN! NOTE! To lower the computation cost, we don't check that A is a upper triangular
 *  matrix (it's assumed that user already make sure of that before calling this routine).
 */
Matrix Matrix::BackSubtitution(const Matrix &A, const Matrix &B) const
{
    Matrix _outp(A.i16row, 1, InitWithoutZero);
    if ((A.i16row != A.i16col) || (A.i16row != B.i16row))
    {
        _outp.vSetMatrixInvalid();
        return _outp;
    }

    for (int16_t _i = A.i16col - 1; _i >= 0; _i--)
    {
        _outp(_i, 0) = B(_i, 0);
        for (int16_t _j = _i + 1; _j < A.i16col; _j++)
        {
            _outp(_i, 0) = _outp(_i, 0) - A(_i, _j) * _outp(_j, 0);
        }
        if (fabs(A(_i, _i)) < float_prec(float_prec_ZERO))
        {
            _outp.vSetMatrixInvalid();
            return _outp;
        }
        _outp(_i, 0) = _outp(_i, 0) / A(_i, _i);
    }

    return _outp;
}

/* Do the forward-subtitution operation for lower triangular matrix A & column matrix B to solve x:
 *                      Ax = B
 *
 * x = ForwardSubtitution(&A, &B);
 *
 * CATATAN! NOTE! To lower the computation cost, we don't check that A is a lower triangular
 *  matrix (it's assumed that user already make sure of that before calling this routine).
 */
Matrix Matrix::ForwardSubtitution(const Matrix &A, const Matrix &B) const
{
    Matrix _outp(A.i16row, 1, InitWithoutZero);
    if ((A.i16row != A.i16col) || (A.i16row != B.i16row))
    {
        _outp.vSetMatrixInvalid();
        return _outp;
    }

    for (int16_t _i = 0; _i < A.i16row; _i++)
    {
        _outp(_i, 0) = B(_i, 0);
        for (int16_t _j = 0; _j < _i; _j++)
        {
            _outp(_i, 0) = _outp(_i, 0) - A(_i, _j) * _outp(_j, 0);
        }
        if (fabs(A(_i, _i)) < float_prec(float_prec_ZERO))
        {
            _outp.vSetMatrixInvalid();
            return _outp;
        }
        _outp(_i, 0) = _outp(_i, 0) / A(_i, _i);
    }
    return _outp;
}

/* -------------------------------------------- Matrix printing function -------------------------------------------- */
/* -------------------------------------------- Matrix printing function -------------------------------------------- */
#if (SYSTEM_IMPLEMENTATION == SYSTEM_IMPLEMENTATION_PC)
void Matrix::vPrint(void)
{
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        cout << "[ ";
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            cout << std::fixed << std::setprecision(3) << (*this)(_i, _j) << " ";
        }
        cout << "]";
        cout << endl;
    }
    cout << endl;
}
void Matrix::vPrintFull(void)
{
    for (int16_t _i = 0; _i < this->i16row; _i++)
    {
        cout << "[ ";
        for (int16_t _j = 0; _j < this->i16col; _j++)
        {
            cout << resetiosflags(ios::fixed | ios::showpoint) << (*this)(_i, _j) << " ";
        }
        cout << "]";
        cout << endl;
    }
    cout << endl;
}
#elif (SYSTEM_IMPLEMENTATION == SYSTEM_IMPLEMENTATION_EMBEDDED_ARDUINO)
// void Matrix::vPrint(void)
// {
//     char _bufSer[10];
//     for (int16_t _i = 0; _i < this->i16row; _i++)
//     {
//         Serial.print("[ ");
//         for (int16_t _j = 0; _j < this->i16col; _j++)
//         {
//             snprintf(_bufSer, sizeof(_bufSer) - 1, "%2.2f ", (*this)(_i, _j));
//             Serial.print(_bufSer);
//         }
//         Serial.println("]");
//     }
//     Serial.println("");
// }
// void Matrix::vPrintFull(void)
// {
//     char _bufSer[32];
//     for (int16_t _i = 0; _i < this->i16row; _i++)
//     {
//         Serial.print("[ ");
//         for (int16_t _j = 0; _j < this->i16col; _j++)
//         {
//             snprintf(_bufSer, sizeof(_bufSer) - 1, "%e ", (*this)(_i, _j));
//             Serial.print(_bufSer);
//         }
//         Serial.println("]");
//     }
//     Serial.println("");
// }
#else
/* User must define the print function somewhere */
/* void vPrint(); */
/* void vPrintFull(); */
#endif


void SPEW_THE_ERROR(char const * str)
{
    #if (SYSTEM_IMPLEMENTATION == SYSTEM_IMPLEMENTATION_PC)
        cout << (str) << endl;
    #elif (SYSTEM_IMPLEMENTATION == SYSTEM_IMPLEMENTATION_EMBEDDED_ARDUINO)
        Serial.println(str);
    #else
        /* Silent function */
    #endif
    while(1);
}