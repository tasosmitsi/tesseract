#include <catch_amalgamated.hpp>
#define CATCH_CONFIG_MAIN

#include "matrix.h"
#include "utilities.h"
#include "matrix_algorithms.h"
#include <chrono>

using namespace std::chrono;

auto start = high_resolution_clock::now();
void tick()
{
    start = high_resolution_clock::now();
}
void tock(std::string message)
{
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << message << ": "
              << duration.count() << " microseconds" << std::endl;
}

void tock()
{
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken: "
              << duration.count() << " microseconds" << std::endl;
}

TEST_CASE("Matrix class", "[matrix]")
{
    Matrix<double, 10, 10> mat1(1), mat2(2), mat4(10);

    SECTION("Matrix accessing elements")
    {
        mat1.setIdentity()(0, 9) = 45.654;

        CHECK(mat1(0, 9) == 45.654);
    }

    SECTION("Matrix total size, number of dimensions, and shape")
    {
        Matrix<double, 2, 2> matrix;
        Matrix<double, 15, 32> matrix1;

        CHECK(matrix.getTotalSize() == 4);
        CHECK(matrix.getNumDims() == 2);
        CHECK(matrix.getShape() == "(2,2)");

        CHECK(matrix1.getTotalSize() == 480);
        CHECK(matrix1.getNumDims() == 2);
        CHECK(matrix1.getShape() == "(15,32)");
    }

    SECTION("Is matrix identity")
    {
        mat1.setIdentity();
        CHECK(mat1.isIdentity());

        mat1(0, 0) = 15;
        CHECK_FALSE(mat1.isIdentity());

        mat1.setIdentity();
        // check if all diagonal elements are 1
        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                if (i == j)
                {
                    CHECK(mat1(i, j) == 1);
                }
            }
        }
    }

    SECTION("Is matriz full of zeros")
    {
        mat1.setToZero();
        // check if all elements are 0
        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                CHECK(mat1(i, j) == 0);
            }
        }
    }

    SECTION("Is matrix homogeneous")
    {
        double value = 13.3;
        mat1.setHomogen(value);
        // check if all elements are 5
        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                CHECK(mat1(i, j) == value);
            }
        }
    }

    SECTION("Is matrix sequential")
    {
        mat1.setSequencial();
        // check if all elements are sequential
        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                CHECK(mat1(i, j) == i * mat1.getDim(1) + j);
            }
        }
    }

    SECTION("Are matrices equal")
    {
        mat1.setIdentity();
        mat2.setIdentity();

        CHECK(mat1 == mat2);

        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                CHECK(mat1(i, j) == mat2(i, j));
            }
        }

        mat1(1, 2) = 3.0;
        CHECK_FALSE(mat1 == mat2);

        // now check in case of transpose
        mat1.inplace_transpose();
        mat2(1, 2) = 3.0;
        CHECK_FALSE(mat1 == mat2);
    }

    SECTION("Check dimensions mismatch and == , != operators")
    {
        // this test should fail when the dimensions of the matrices are not equal
        // and should pass when the dimensions are equal even after transposing one of the matrices
        Matrix<double, 2, 3> matrix1(2);
        Matrix<double, 3, 2> matrix2(2);

        CHECK_THROWS(matrix1 == matrix2);
        CHECK_THROWS(matrix1 != matrix2);

        matrix2.inplace_transpose();
        CHECK_NOTHROW(matrix1 == matrix2);
        CHECK_FALSE(matrix1 != matrix2);
    }

    SECTION("Assign matrix to another matrix")
    {
        mat1.setIdentity();
        mat2 = mat1;

        CHECK(mat1 == mat2);

        mat1(1, 2) = 3.0;

        CHECK_FALSE(mat1 == mat2);
    }

    SECTION("Is matrix diagonal")
    {
        mat1.setDiagonal(1);
        CHECK(mat1.isIdentity());

        mat1(1, 2) = 3.0;
        CHECK_FALSE(mat1.isIdentity());

        mat1.setDiagonal(5);

        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                if (i == j)
                {
                    CHECK(mat1(i, j) == 5);
                }
                else
                {
                    CHECK(mat1(i, j) == 0);
                }
            }
        }
    }

    SECTION("Is matrix symetric")
    {
        mat1.setIdentity();
        CHECK(mat1.isSymmetric());

        mat1(1, 2) = 3.0;
        CHECK_FALSE(mat1.isSymmetric());
        mat1(2, 1) = 3.0;
        CHECK(mat1.isSymmetric());
    }

    SECTION("Is matrix upper triangular")
    {
        mat1.setIdentity();
        CHECK(mat1.isUpperTriangular());

        mat1(1, 2) = 3.0;
        CHECK(mat1.isUpperTriangular());

        mat1(1, 0) = 3.0;
        CHECK_FALSE(mat1.isUpperTriangular());
    }

    SECTION("Is matrix lower triangular")
    {
        mat1.setIdentity();
        CHECK(mat1.isLowerTriangular());

        mat1(2, 1) = 3.0;
        CHECK(mat1.isLowerTriangular());

        mat1(0, 1) = 3.0;
        CHECK_FALSE(mat1.isLowerTriangular());
    }

    SECTION("Make matrix upper triangular")
    {
        mat1.setHomogen(5);

        // check if is not upper triangular
        CHECK_FALSE(mat1.isUpperTriangular());

        // assign the upper triangular matrix to mat2 without
        // inplace modification, check if mat1 is still not upper
        // triangular and mat2 is upper triangular
        mat2 = mat1.upperTriangular();
        CHECK_FALSE(mat1.isUpperTriangular());
        CHECK(mat2.isUpperTriangular());

        // make mat1 upper triangular in place
        mat1.upperTriangular(true);
        CHECK(mat1.isUpperTriangular());

        mat1.setHomogen(5);
        // transpose mat1
        mat1.inplace_transpose();
        // perform upper triangular on the transposed matrix
        mat1.upperTriangular(true);
        // mat1 should stil be upper triangular
        CHECK(mat1.isUpperTriangular());
    }

    SECTION("Make matrix lower triangular")
    {
        mat1.setHomogen(5);

        // check if is not lower triangular
        CHECK_FALSE(mat1.isLowerTriangular());

        // assign the lower triangular matrix to mat2 without
        // inplace modification, check if mat1 is still not lower
        // triangular and mat2 is lower triangular
        mat2 = mat1.lowerTriangular();
        CHECK_FALSE(mat1.isLowerTriangular());
        CHECK(mat2.isLowerTriangular());

        // make mat1 lower triangular in place
        mat1.lowerTriangular(true);
        CHECK(mat1.isLowerTriangular());

        mat1.setHomogen(5);
        // transpose mat1
        mat1.inplace_transpose();
        // perform lower triangular on the transposed matrix
        mat1.lowerTriangular(true);
        // mat1 should stil be lower triangular
        CHECK(mat1.isLowerTriangular());
    }

    SECTION("Matrix elementary operations")
    {
        // TODO: Split this section into smaller sections
        // one for addition, one for subtraction,
        // one for multiplication, one for division

        Matrix<double, 10, 10>
            mat1, mat2, mat4, mat5,
            mat6, mat7, mat8, mat9,
            mat10, mat11, mat12, mat13,
            mat14, mat15, mat16, mat17,
            mat18, mat19, mat20, mat21;

        mat1.setIdentity();
        mat2.setIdentity();

        // additon
        mat4 = mat1 + mat2;
        mat5 = mat2 + mat1;
        mat12 = mat1 + 2;
        mat13 = 2 + mat1;

        // subtraction
        mat6 = mat1 - mat2;
        mat7 = mat2 - mat1;
        mat14 = mat1 - 2;
        mat15 = 2 - mat1;
        mat20 = -mat1;
        mat21 = -mat13;

        // multiplication
        mat8 = mat1 * mat2;
        mat9 = mat2 * mat1;
        mat16 = mat1 * 2;
        mat17 = 2 * mat1;

        // division
        mat1.setHomogen(4);
        mat2.setHomogen(8);

        mat10 = mat1 / mat2;
        mat11 = mat2 / mat1;
        mat18 = mat1 / 2;
        mat19 = 2 / mat1;

        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                if (i == j)
                {
                    // Check only the diagonal elements
                    CHECK(mat4(i, j) == 2);
                    CHECK(mat5(i, j) == 2);

                    CHECK(mat8(i, j) == 1);
                    CHECK(mat9(i, j) == 1);

                    CHECK(mat12(i, j) == 3);
                    CHECK(mat13(i, j) == 3);

                    CHECK(mat14(i, j) == -1);
                    CHECK(mat15(i, j) == 1);

                    CHECK(mat16(i, j) == 2);
                    CHECK(mat17(i, j) == 2);

                    CHECK(mat20(i, j) == -1);
                    CHECK(mat21(i, j) == -3);
                }
                else
                {
                    // check only the non-diagonal elements
                    CHECK(mat4(i, j) == 0);
                    CHECK(mat5(i, j) == 0);

                    CHECK(mat8(i, j) == 0);
                    CHECK(mat9(i, j) == 0);

                    CHECK(mat12(i, j) == 2);
                    CHECK(mat13(i, j) == 2);

                    CHECK(mat14(i, j) == -2);
                    CHECK(mat15(i, j) == 2);

                    CHECK(mat16(i, j) == 0);
                    CHECK(mat17(i, j) == 0);

                    CHECK(mat20(i, j) == 0);
                    CHECK(mat21(i, j) == -2);
                }

                // check all elements
                CHECK(mat6(i, j) == 0);
                CHECK(mat7(i, j) == 0);

                CHECK(mat10(i, j) == 0.5);
                CHECK(mat11(i, j) == 2);

                CHECK(mat18(i, j) == 2);
                CHECK(mat19(i, j) == 0.5);
            }
        }
    }

    SECTION("Check dimensions mismatch on addition, subtraction, multiplication, and division")
    {
        // this test should fail when the dimensions of the matrices are not equal
        // and should pass when the dimensions are equal even after transposing one of the matrices

        Matrix<double, 2, 3> matrix1(2);
        Matrix<double, 3, 2> matrix2(2);

        CHECK_THROWS(matrix1 + matrix2);
        CHECK_THROWS(matrix1 - matrix2);
        CHECK_THROWS(matrix1 * matrix2);
        CHECK_THROWS(matrix1 / matrix2);

        matrix1.inplace_transpose();
        CHECK_NOTHROW(matrix1 + matrix2);
        CHECK_NOTHROW(matrix1 - matrix2);
        CHECK_NOTHROW(matrix1 * matrix2);
        CHECK_NOTHROW(matrix1 / matrix2);
    }

    SECTION("Matrix transpose")
    {
        mat1.setRandom(-10, 10);
        mat2 = mat1;

        // check inplace transpose first
        mat1.inplace_transpose();

        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                CHECK(mat1(i, j) == mat2(j, i));
            }
        }

        // check non-inplace transpose
        mat1.setRandom(-10, 10);
        mat2 = mat1.transposed();

        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                CHECK(mat1(i, j) == mat2(j, i));
            }
        }

        // now check a long oppeartion by adding a zero matrix to the
        // transposed (not in place) matrix. The mat1 should not change.
        mat1.setIdentity();
        mat1(0, 1) = 10;
        mat2.setToZero();

        auto mat3 = mat1.transposed() + mat2;
        auto mat4 = mat1 + mat2;

        // In both cases, mat2 is a zero matrix (should not change the result)
        // mat3 should not be equal to mat1 because of the transpose
        CHECK(mat3 != mat1);
        CHECK(mat3(1, 0) == 10);

        // mat4 should be equal to mat1
        CHECK(mat4 == mat1);
    }

    SECTION("Matrix matmul")
    {
        Matrix<double, 2, 3> matrix1(2);
        Matrix<double, 3, 2> matrix2(2);

        matrix1.setHomogen(10);
        matrix2.setHomogen(33);

        tick();
        auto res = Matrix<double, 2, 2>::matmul(matrix1, matrix2);
        tock("C++ matmul");

        // check the dimensions of the res matrix
        CHECK(res.getDim(0) == 2);
        CHECK(res.getDim(1) == 2);

        // check the values of the res matrix
        for (size_t i = 0; i < res.getDim(0); ++i)
        {
            for (size_t j = 0; j < res.getDim(1); ++j)
            {
                double sum = 0;
                for (size_t k = 0; k < matrix1.getDim(1); ++k)
                {
                    sum += matrix1(i, k) * matrix2(k, j);
                }
                CHECK(res(i, j) == sum);
            }
        }

        // using python numpy
        std::string numpy_string1 = toNumpyArray(matrix1);
        std::string numpy_string2 = toNumpyArray(matrix2);
        std::string python_code = R"(
import numpy as np
import sys
import io
import time
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Redirect output to a string
output = io.StringIO()  # Initialize output before redirection
sys.stdout = output
np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.3f}'})
a = np.array()" + numpy_string1 + R"()
b = np.array()" + numpy_string2 + R"()
# print('Matrix 1:')
# print(a)
# print('Matrix 2:')
# print(b)
# print('Result:')
start = time.time()
result = np.matmul(a, b)
end = time.time()
print(result)
print(',Numpy matmul:', (end - start) * 1000000, 'microseconds')

# Capture the output
sys.stdout = sys.__stdout__
output_string = output.getvalue()
        )";

        // Execute the Python code
        std::string result = executePythonAndGetString(python_code);
        removeNewlines(result);
        std::vector<std::string> results = splitStringByComma(result);

        // Print the result the time taken by numpy
        std::cout << results[1] << std::endl;

        // Check if the output is the same
        CHECK(results[0] == toFormattedNumpyArray(res));
    }

    SECTION("Matrix inverse")
    {
        // init the matrix
        double initValues[4][4] = {
            {2.0, -1, 2.0, -1},
            {4, 5.0, 2.5, -17},
            {2.0, -1, 2.43, -30},
            {4, 5.0, 245, -10}};
        Matrix<double, 4, 4> matrix3 = initValues;

        // using tessaract
        tick();
        auto inv = matrix3.inverse();
        tock("C++ Inverse");

        // using python numpy
        std::string numpy_string = toNumpyArray(matrix3);
        std::string python_code = R"(
import numpy as np
import sys
import io
import time
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Redirect output to a string
output = io.StringIO()  # Initialize output before redirection
sys.stdout = output
np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.3f}'})

a = np.array()" + numpy_string + R"()
# print('Original matrix:')
# print(a)
# print('Inverse matrix:')
start = time.time()
inv = np.linalg.inv(a)
end = time.time()
print(inv)
print(',Numpy inverse:', (end - start) * 1000000, 'microseconds')

# Capture the output
sys.stdout = sys.__stdout__
output_string = output.getvalue()
        )";

        // Execute the Python code
        std::string result = executePythonAndGetString(python_code);
        removeNewlines(result);
        std::vector<std::string> results = splitStringByComma(result);

        // Print the result the time taken by numpy
        std::cout << results[1] << std::endl;

        // Check if the output is the same
        CHECK(results[0] == toFormattedNumpyArray(inv));
    }

    SECTION("Test Cholesky Decomposition")
    {
        // init the matrix
        double initValues[3][3] = {
            {4, 12, -16},
            {12, 37, -43},
            {-16, -43, 98}};
        Matrix<double, 3, 3> matrix3 = initValues;

        double cholesky_values[3][3] = {
            {2, 0, 0},
            {6, 1, 0},
            {-8, 5, 3}};
        Matrix<double, 3, 3> cholesky_matrix = cholesky_values;

        // using tessaract
        tick();
        auto cholesky = matrix_algorithms::choleskyDecomposition(matrix3);
        tock("C++ Cholesky Decomposition");

        CHECK(cholesky == cholesky_matrix);
    }

    SECTION("Is matrix positive definite or semi-definite")
    {
        // init the matrix
        double initValues[3][3] = {
            {4, 12, -16},
            {12, 37, -43},
            {-16, -43, 98}};
        Matrix<double, 3, 3> matrix = initValues;
        auto result = matrix.isPositiveDefinite();
        CHECK(result == matrix_traits::Definiteness::PositiveDefinite);

        // TODO: test semi definite matrix
        // double initValues1[2][2] = {
        //     {3, 4},
        //     {4, 16/3}};

        // Matrix<double, 2, 2> matrix1 = initValues1;
        // auto result = matrix1.isPositiveDefinite(true);
        // CHECK(result == matrix_traits::Definiteness::PositiveSemiDefinite);

        // Not positive definite matrix
        double initValues2[3][3] = {
            {1, 0, 0},
            {0, 0, 1},
            {0, 1, 0}};
        matrix = initValues2;
        result = matrix.isPositiveDefinite();
        CHECK(result == matrix_traits::Definiteness::NotPositiveDefinite);
    }

    SECTION("Is matrix orthogonal")
    {
        // init the matrix
        double initValues[4][4] = {
            {1, 0, 0, 0},
            {0, 0, -1, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 1}};

        Matrix<double, 4, 4> matrix3 = initValues;

        CHECK(matrix3.isOrthogonal());

        matrix3(0, 0) = 2;
        CHECK_FALSE(matrix3.isOrthogonal());
    }
}
