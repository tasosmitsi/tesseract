#ifndef MATRIX_H
#define MATRIX_H

#include "config.h"
#include <cstring>


#if (SYSTEM_IMPLEMENTATION == SYSTEM_IMPLEMENTATION_PC)
#include <iostream>
#include <iomanip> // std::setprecision

using namespace std;
#elif (SYSTEM_IMPLEMENTATION == SYSTEM_IMPLEMENTATION_EMBEDDED_ARDUINO)
// #include <Wire.h>
#endif

class Matrix
{
public:
    typedef enum
    {
        InitWithZero,
        InitWithoutZero
    } InitializationType;

    /* --------------------------------------------- Basic Matrix Class functions --------------------------------------------- */
    /* Init an empty _i16row x _i16col matrix with zeros, */
    Matrix(const int16_t _i16row, const int16_t _i16col, const InitializationType _init = InitWithZero);

    /* Init matrix size _i16row x _i16col with entries initData */
    Matrix(const int16_t _i16row, const int16_t _i16col, const float_prec *initData, const InitializationType _init = InitWithZero);

    /* Copy constructor (for this operation --> A(B)) (copy B into A) */
    Matrix(const Matrix &old_obj);

    /* Assignment operator (for this operation --> A = B) (copy B into A) */
    Matrix &operator=(const Matrix &obj);

    /* Destructor */
    ~Matrix(void);

    /* Get internal state */
    inline int16_t i16getRow(void) const { return this->i16row; }
    inline int16_t i16getCol(void) const { return this->i16col; }

    /* ------------------------------------------- Matrix entry accessing functions ------------------------------------------- */
    /* For example: A(1,2) access the 1st row and 2nd column data of matrix A <--- The preferred way to access the matrix */
    float_prec &operator()(const int16_t _row, const int16_t _col);
    float_prec operator()(const int16_t _row, const int16_t _col) const;

    /* For example: A[1][2] access the 1st row and 2nd column data of matrix A <-- The awesome way */
    class Proxy
    {
    public:
        Proxy(float_prec *_inpArr, const int16_t _maxCol)
        {
            _array.ptr = _inpArr;
            this->_maxCol = _maxCol;
        }
        Proxy(const float_prec *_inpArr, const int16_t _maxCol)
        {
            _array.cptr = _inpArr;
            this->_maxCol = _maxCol;
        }
        float_prec &operator[](const int16_t _col);
        float_prec operator[](const int16_t _col) const;

    private:
        union
        { /* teehee xp */
            const float_prec *cptr;
            float_prec *ptr;
        } _array;
        int16_t _maxCol;
    };
    Proxy operator[](const int16_t _row);
    const Proxy operator[](const int16_t _row) const;

    /* ----------------------------------------- Matrix checking function declaration ----------------------------------------- */
    bool bMatrixIsValid(void);
    void vSetMatrixInvalid(void);
    bool bMatrixIsSquare();
    /* --------------------------------------------- Matrix elementary operations --------------------------------------------- */
    bool operator==(const Matrix &_compare) const;
    bool operator!=(const Matrix &_compare) const;
    Matrix operator-(void) const;
    Matrix operator+(const float_prec _scalar) const;
    Matrix operator-(const float_prec _scalar) const;
    Matrix operator*(const float_prec _scalar) const;
    Matrix operator/(const float_prec _scalar) const;
    Matrix operator+(const Matrix &_matAdd) const;
    Matrix operator-(const Matrix &_matSub) const;
    Matrix operator*(const Matrix &_matMul) const;
    /* Declared outside class below */
    /* inline Matrix operator + (const float_prec _scalar, Matrix _mat); */
    /* inline Matrix operator - (const float_prec _scalar, Matrix _mat); */
    /* inline Matrix operator * (const float_prec _scalar, Matrix _mat); */
    /* ----------------------------------------------- Simple Matrix operations ----------------------------------------------- */
    void vRoundingElementToZero(const int16_t _i, const int16_t _j);
    Matrix RoundingMatrixToZero(void);
    void vSetHomogen(const float_prec _val);
    void vSetToZero(void);
    void vSetRandom(const int32_t _maxRand, const int32_t _minRand);
    void vSetDiag(const float_prec _val);
    void vSetIdentity(void);
    Matrix Transpose(void);
    bool bNormVector(void);
    /* ------------------------------------------ Matrix/Vector insertion operations ------------------------------------------ */
    Matrix InsertVector(const Matrix &_Vector, const int16_t _posCol);
    Matrix InsertSubMatrix(const Matrix &_subMatrix, const int16_t _posRow, const int16_t _posCol);
    Matrix InsertSubMatrix(const Matrix &_subMatrix, const int16_t _posRow, const int16_t _posCol,
                           const int16_t _lenRow, const int16_t _lenColumn);
    Matrix InsertSubMatrix(const Matrix &_subMatrix, const int16_t _posRow, const int16_t _posCol,
                           const int16_t _posRowSub, const int16_t _posColSub,
                           const int16_t _lenRow, const int16_t _lenColumn);
    /* ---------------------------------------------------- Big operations ---------------------------------------------------- */
    /* Matrix invertion using Gauss-Jordan algorithm */
    Matrix Invers(void) const;
    /* Check the definiteness of a matrix */
    bool bMatrixIsPositiveDefinite(const bool checkPosSemidefinite = false) const;
    /* Return the vector (Mx1 matrix) correspond with the diagonal entries of 'this' */
    Matrix GetDiagonalEntries(void) const;
    /* Do the Cholesky Decomposition using Cholesky-Crout algorithm, return 'L' matrix */
    Matrix CholeskyDec(void) const;
    /* Do Householder Transformation for QR Decomposition operation */
    Matrix HouseholderTransformQR(const int16_t _rowTransform, const int16_t _colTransform);
    /* Do QR Decomposition for matrix using Householder Transformation */
    bool QRDec(Matrix &Qt, Matrix &R) const;
    /* Do back-subtitution for upper triangular matrix A & column matrix B:
     * x = BackSubtitution(&A, &B)          ; for Ax = B
     */
    Matrix BackSubtitution(const Matrix &A, const Matrix &B) const;
    /* Do forward-subtitution for lower triangular matrix A & column matrix B:
     * x = ForwardSubtitution(&A, &B)       ; for Ax = B
     */
    Matrix ForwardSubtitution(const Matrix &A, const Matrix &B) const;
    /* ----------------------------------------------- Matrix printing function ----------------------------------------------- */
    void vPrint(void);
    void vPrintFull(void);

private:
    /* Data structure of Matrix class:
     *  0 <= i16row <= MATRIX_MAXIMUM_SIZE      ; i16row is the row of the matrix. i16row is invalid if (i16row == -1)
     *  0 <= i16col <= MATRIX_MAXIMUM_SIZE      ; i16col is the column of the matrix. i16col is invalid if (i16col == -1)
     *
     * Accessing index start from 0 until i16row/i16col, that is:
     *  (0 <= idxRow < i16row)     and     (0 <= idxCol < i16col).
     * There are 3 ways to access the data:
     *  1. A[idxRow][idxCol]          <-- Slow, not recommended, but make a cute code. With bounds checking.
     *  2. A(idxRow, idxCol)          <-- The preferred way. With bounds checking.
     *  3. A._at(idxRow, idxCol)      <-- Just for internal function usage. Without bounds checking.
     *
     * floatData[MATRIX_MAXIMUM_SIZE][MATRIX_MAXIMUM_SIZE] is the memory representation of the matrix. We only use the
     *  first i16row-th and first i16col-th memory for the matrix data. The rest is unused.
     *
     * This configuration might seems wasteful (yes it is). But with this, we can make the matrix library code as cleanly
     *  as possible (like I said in the github page, I've made decision to sacrifice speed & performance to get best code
     *  readability I could get).
     *
     * You could change the data structure of floatData if you want to make the implementation more memory efficient.
     */
    int16_t i16row;
    int16_t i16col;
    float_prec floatData[MATRIX_MAXIMUM_SIZE][MATRIX_MAXIMUM_SIZE];

    /* Private way to access floatData without bound checking.
     *  TODO: For Matrix member function we could do the bound checking once at the beginning of the function, and use this
     *          to access the floatData instead of (i,j) operator. From preliminary experiment doing this only on elementary
     *          operation (experiment @2020-04-27), we can get up to 45% computation boost!!! (MPC benchmark 414 us -> 226 us)!
     */
    float_prec &_at(const int16_t _row, const int16_t _col) { return this->floatData[_row][_col]; }
    float_prec _at(const int16_t _row, const int16_t _col) const { return this->floatData[_row][_col]; }
};

inline Matrix operator+(const float_prec _scalar, const Matrix &_mat);
inline Matrix operator-(const float_prec _scalar, const Matrix &_mat);
inline Matrix operator*(const float_prec _scalar, const Matrix &_mat);
inline Matrix MatIdentity(const int16_t _i16size);


#endif // MATRIX_H