#include <catch_amalgamated.hpp>
#include "fused/fused_matrix.h"
#include <Dense>
#include "utilities.h"

#define EIGEN_NO_MALLOC

TEST_CASE("FusedMatrix & Matrix benchmarks", "[fused_eigen_benchmark]")
{
    FusedTensorND<float, 100, 100> fmat1(7.0), fmat2(7.0), fmat3(7.0), fmat4(7.0), fmat5(7.0);
    Eigen::Matrix<float, 100, 100> mat1, mat2, mat3, mat4, mat5;

    SECTION("Long operations benchmark")
    {

        mat1.setRandom();
        mat2.setRandom();
        mat3.setRandom();
        mat4.setRandom();
        mat5.setRandom();

        std::cout << "Benchmarking long operations on FusedMatrix and Matrix" << std::endl;
        tick();
        for (int i = 0; i < 10000; ++i)
        {
            // fmat5 = fmat1 + fmat2 + fmat3 - fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2 + fmat3 - fmat4 + fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2;
            fmat5.eval(fmat1 + fmat2 + fmat3 - fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2 + fmat3 - fmat4 + fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2);
        }

        uint FusedMatrix_time = tock("FusedMatrix long operations");

        tick();
        for (int i = 0; i < 10000; ++i)
        {
            mat5 = mat1 + mat2 + mat3 - mat1 - mat2 + mat3 + mat4 + mat1 - mat2 + mat3 - mat4 + mat1 - mat2 + mat3 + mat4 + mat1 - mat2;
        }
        uint Matrix_time = tock("Matrix long operations");

        // Percentage Increase
        double percentage_increase = (((double)FusedMatrix_time - (double)Matrix_time) / (double)Matrix_time) * 100.0;

        std::cout << "Percentage decrease in time for FusedMatrix compared to Matrix: "
                  << percentage_increase << "%" << std::endl;

        // // fmat4.print();
        // // std::cout << std::endl;
        // // mat4.print();
    }
}
