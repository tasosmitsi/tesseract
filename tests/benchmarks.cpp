#include <catch_amalgamated.hpp>
#include "fused/fused_vector.h"
#include <Dense>
#include "utilities.h"

#define EIGEN_NO_MALLOC

TEMPLATE_TEST_CASE("Benchmarks", "[benchmarks]", double, float)
{
    using T = TestType;

    FusedTensorND<T, 100, 100> fmat1, fmat2, fmat3, fmat4, fmat5; // It can work with FusedMatrix amd FusedVector too
    Eigen::Matrix<T, 100, 100> mat1, mat2, mat3, mat4, mat5;

    mat1.setRandom();
    mat2.setRandom();
    mat3.setRandom();
    mat4.setRandom();
    mat5.setRandom();

    fmat1.setRandom(10, -10);
    fmat2.setRandom(10, -10);
    fmat3.setRandom(10, -10);
    fmat4.setRandom(10, -10);
    fmat5.setRandom(10, -10);

    BENCHMARK("FusedMatrix long operations")
    {
        for (int i = 0; i < 100; ++i)
        {
            fmat5 = fmat1 + fmat2 + fmat3 - fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2 + fmat3 - fmat4 + fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2;
        }
        return fmat5;
    };

    BENCHMARK("Eigen long operations")
    {
        for (int i = 0; i < 100; ++i)
        {
            mat5 = mat1 + mat2 + mat3 - mat1 - mat2 + mat3 + mat4 + mat1 - mat2 + mat3 - mat4 + mat1 - mat2 + mat3 + mat4 + mat1 - mat2;
        }
        return mat5;
    };

    BENCHMARK("FusedMatrix matmul")
    {
        FusedMatrix<T, 2, 3> matrix1(2);
        FusedMatrix<T, 3, 2> matrix2(2);
        FusedMatrix<T, 2, 2> res;

        matrix1.setHomogen(10);
        matrix2.setHomogen(33);

        for (int i = 0; i < 100; ++i)
        {
            res = FusedMatrix<T, 2, 2>::matmul(matrix1, matrix2);
        }
        return res;
    };

    BENCHMARK("Eigen matmul")
    {
        Eigen::Matrix<T, 2, 3> matrix1;
        Eigen::Matrix<T, 3, 2> matrix2;
        Eigen::Matrix<T, 2, 2> res;

        matrix1.setConstant(10);
        matrix2.setConstant(33);

        for (int i = 0; i < 100; ++i)
        {
            res = matrix1 * matrix2;
        }
        return res;
    };

    BENCHMARK("FusedMatrix inverse")
    {
        // init the matrix
        T initValues[4][4] = {
            {2.0, -1, 2.0, -1},
            {4, 5.0, 2.5, -17},
            {2.0, -1, 2.43, -30},
            {4, 5.0, 245, -10}};
        FusedMatrix<T, 4, 4> matrix3 = initValues;
        FusedMatrix<T, 4, 4> inv;

        for (int i = 0; i < 100; ++i)
        {
            inv = matrix3.inverse();
        }
        return inv;
    };

    BENCHMARK("Eigen inverse")
    {
        Eigen::Matrix<T, 4, 4> matrix3;
        Eigen::Matrix<T, 4, 4> inv;

        // init the matrix
        T initValues[4][4] = {
            {2.0, -1, 2.0, -1},
            {4, 5.0, 2.5, -17},
            {2.0, -1, 2.43, -30},
            {4, 5.0, 245, -10}};
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                matrix3(i, j) = initValues[i][j];
            }
        }

        for (int i = 0; i < 100; ++i)
        {
            inv = matrix3.inverse();
        }
        return inv;
    };

    BENCHMARK("FusedMatrix Cholesky Decomposition")
    {
        // init the matrix
        T initValues[3][3] = {
            {4, 12, -16},
            {12, 37, -43},
            {-16, -43, 98}};
        FusedMatrix<T, 3, 3> matrix3 = initValues;
        FusedMatrix<T, 3, 3> cholesky;

        // using tessaract
        for (int i = 0; i < 100; ++i)
        {
            cholesky = matrix_algorithms::choleskyDecomposition(matrix3);
        }
        return cholesky;
    };

    BENCHMARK("Eigen Cholesky Decomposition")
    {
        // init the matrix
        T initValues[3][3] = {
            {4, 12, -16},
            {12, 37, -43},
            {-16, -43, 98}};
        Eigen::Matrix<T, 3, 3> matrix3;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                matrix3(i, j) = initValues[i][j];
            }
        }
        Eigen::Matrix<T, 3, 3> cholesky;

        for (int i = 0; i < 100; ++i)
        {
            Eigen::LLT<Eigen::Matrix<T, 3, 3>> lltOfA(matrix3);
            cholesky = lltOfA.matrixL();
        }

        return cholesky;
    };
}
