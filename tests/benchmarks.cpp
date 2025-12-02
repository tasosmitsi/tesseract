#include <catch_amalgamated.hpp>
#include "fused/fused_vector.h"
#include <Dense>
#include "utilities.h"

#define EIGEN_NO_MALLOC

TEMPLATE_TEST_CASE("Benchmarks", "[benchmarks]", double, float)
{
    using T = TestType;

    FusedTensorND<T, 100, 100> fmat1, fmat2, fmat3, fmat4, fmat5, res; // It can work with FusedMatrix amd FusedVector too
    FusedMatrix<T, 100, 100> matrix1, matrix2;
    Eigen::Matrix<T, 100, 100> mat1, mat2, mat3, mat4, mat5;

    T init_inverse_values[4][4] = {
        {2.0, -1, 2.0, -1},
        {4, 5.0, 2.5, -17},
        {2.0, -1, 2.43, -30},
        {4, 5.0, 245, -10}};

    T init_cholesky_values[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}};

    // Init FusedMatrix
    FusedMatrix<T, 4, 4> inv_matrix = init_inverse_values;
    FusedMatrix<T, 4, 4> inv_res;

    FusedMatrix<T, 3, 3> pre_cholesky = init_cholesky_values;
    FusedMatrix<T, 3, 3> cholesky;
    // --------------------------

    // Init Eigen Matrix
    Eigen::Matrix<T, 4, 4> inv_eigen_matrix;
    Eigen::Matrix<T, 4, 4> inv_eigen_res;
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            inv_eigen_matrix(i, j) = init_inverse_values[i][j];
        }
    }

    Eigen::Matrix<T, 3, 3> pre_eigen_cholesky;
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            pre_eigen_cholesky(i, j) = init_cholesky_values[i][j];
        }
    }
    Eigen::Matrix<T, 3, 3> cholesky_eigen;
    // --------------------------

    mat1.setRandom();
    mat2.setRandom();
    mat3.setRandom();
    mat4.setRandom();
    mat5.setRandom();

    matrix1.setRandom(10, -10);
    matrix2.setRandom(10, -10);

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
        res = FusedMatrix<T, 100, 100>::matmul(matrix1, matrix2);
        return res;
    };

    BENCHMARK("Eigen matmul")
    {
        mat3 = mat1 * mat2;
        return mat3;
    };

    BENCHMARK("FusedMatrix inverse")
    {
        for (int i = 0; i < 100; ++i)
        {
            inv_res = inv_matrix.inverse();
        }
        return inv_res;
    };

    BENCHMARK("Eigen inverse")
    {
        for (int i = 0; i < 100; ++i)
        {
            inv_eigen_res = inv_eigen_matrix.inverse();
        }
        return inv_eigen_res;
    };

    BENCHMARK("FusedMatrix Cholesky Decomposition")
    {
        for (int i = 0; i < 100; ++i)
        {
            cholesky = matrix_algorithms::choleskyDecomposition(pre_cholesky);
        }
        return cholesky;
    };

    BENCHMARK("Eigen Cholesky Decomposition")
    {
        for (int i = 0; i < 100; ++i)
        {
            Eigen::LLT<Eigen::Matrix<T, 3, 3>> lltOfA(pre_eigen_cholesky);
            cholesky_eigen = lltOfA.matrixL();
        }
        return cholesky_eigen;
    };
}
