#include <catch_amalgamated.hpp>
#include <Dense>

#include "fused/fused_vector.h"
#include "algorithms/decomposition/cholesky.h"
#include "utilities.h"
#include "utilities/cycle_counter/cycle_counter.h"

#define EIGEN_NO_MALLOC

TEMPLATE_TEST_CASE("Benchmarks", "[benchmarks]", double, float, int32_t, int64_t)
{
    CycleCounter cc;

    using T = TestType;

    FusedTensorND<T, 100, 100> fmat1, fmat2, fmat3, fmat4, fmat5; // It can work with FusedMatrix amd FusedVector too
    FusedMatrix<T, 100, 100> matrix1, matrix2, res;
    Eigen::Matrix<T, 100, 100> mat1, mat2, mat3, mat4, mat5;

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
            cc.start();
            fmat5 = fmat1 + fmat2 + fmat3 - fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2 + fmat3 - fmat4 + fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2;
            cc.stop();
        }
        return fmat5;
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";

    BENCHMARK("Eigen long operations")
    {
        for (int i = 0; i < 100; ++i)
        {
            cc.start();
            mat5 = mat1 + mat2 + mat3 - mat1 - mat2 + mat3 + mat4 + mat1 - mat2 + mat3 - mat4 + mat1 - mat2 + mat3 + mat4 + mat1 - mat2;
            cc.stop();
        }
        return mat5;
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";

    BENCHMARK("FusedMatrix matmul")
    {
        cc.start();
        res = FusedMatrix<T, 100, 100>::matmul(matrix1, matrix2);
        // res = FusedTensorND<T, 100, 100>::einsum(matrix1, matrix2, 0, 0);
        cc.stop();
        return res;
    };

    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";

    BENCHMARK("Eigen matmul")
    {
        cc.start();
        mat3 = mat1.transpose() * mat2;
        cc.stop();
        return mat3;
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";
}

TEMPLATE_TEST_CASE("Benchmarks - floating point", "[benchmarks]", double, float)
{
    CycleCounter cc;

    using T = TestType;

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

    BENCHMARK("FusedMatrix inverse")
    {
        for (int i = 0; i < 100; ++i)
        {
            cc.start();
            inv_res = inv_matrix.inverse();
            cc.stop();
        }
        return inv_res;
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";

    BENCHMARK("Eigen inverse")
    {
        for (int i = 0; i < 100; ++i)
        {
            cc.start();
            inv_eigen_res = inv_eigen_matrix.inverse();
            cc.stop();
        }
        return inv_eigen_res;
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";

    BENCHMARK("FusedMatrix Cholesky Decomposition")
    {
        for (int i = 0; i < 100; ++i)
        {
            cc.start();
            auto result = matrix_algorithms::cholesky(pre_cholesky);
            cc.stop();
        }
        return matrix_algorithms::cholesky(pre_cholesky);
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";

    BENCHMARK("Eigen Cholesky Decomposition")
    {
        for (int i = 0; i < 100; ++i)
        {
            cc.start();
            Eigen::LLT<Eigen::Matrix<T, 3, 3>> lltOfA(pre_eigen_cholesky);
            cholesky_eigen = lltOfA.matrixL();
            cc.stop();
        }
        return cholesky_eigen;
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";

    // Add after the 3x3 Cholesky benchmarks:

    // --- 20x20 Cholesky ---
    FusedMatrix<T, 20, 20> pre_cholesky_20(0);
    Eigen::Matrix<T, 20, 20> pre_eigen_cholesky_20;

    // Build SPD: M * Mᵀ + εI
    {
        FusedMatrix<T, 20, 20> M(0);
        for (my_size_t i = 0; i < 20; ++i)
        {
            M(i, i) = T(i + 2);
            for (my_size_t j = 0; j < i; ++j)
                M(i, j) = T(1) / T(i - j + 1);
        }
        pre_cholesky_20 = FusedMatrix<T, 20, 20>::matmul(M, M.transpose_view());

        for (my_size_t i = 0; i < 20; ++i)
            for (my_size_t j = 0; j < 20; ++j)
                pre_eigen_cholesky_20(i, j) = pre_cholesky_20(i, j);
    }

    BENCHMARK("FusedMatrix Cholesky 20x20")
    {
        for (int i = 0; i < 100; ++i)
        {
            cc.start();
            auto result = matrix_algorithms::cholesky(pre_cholesky_20);
            cc.stop();
        }
        return matrix_algorithms::cholesky(pre_cholesky_20);
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";

    BENCHMARK("Eigen Cholesky 20x20")
    {
        for (int i = 0; i < 100; ++i)
        {
            cc.start();
            Eigen::LLT<Eigen::Matrix<T, 20, 20>> llt(pre_eigen_cholesky_20);
            auto L = llt.matrixL();
            cc.stop();
        }
        return pre_eigen_cholesky_20;
    };
    std::cout << "\n"
              << cc.avg_cycles() << " cycles/call" << std::endl;
    cc.reset();
    std::cout << "------------------------------------------------\n";
}
