#include <catch_amalgamated.hpp>
#include "fused/fused_vector.h"
#include <Dense>
#include "utilities.h"

#define EIGEN_NO_MALLOC

TEST_CASE("FusedTensor & Eigen benchmarks", "[fused_eigen_benchmark]")
{
    FusedTensorND<double, 100, 100> fmat1(7.0), fmat2(7.0), fmat3(7.0), fmat4(7.0), fmat5(7.0); // It can work with FusedMatrix amd FusedVector too
    Eigen::Matrix<double, 100, 100> mat1, mat2, mat3, mat4, mat5;

    SECTION("Long operations benchmark")
    {
        // Microkernel<double, 256, X86_AVX>::test();
        size_t order[] = {1,0};
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

        std::cout << "Benchmarking long operations on FusedMatrix and Matrix" << std::endl;
        tick();
        for (int i = 0; i < 10000; ++i)
        {
            mat5 = mat1 + mat2 + mat3 - mat1 - mat2 + mat3 + mat4 + mat1 - mat2 + mat3 - mat4 + mat1 - mat2 + mat3 + mat4 + mat1 - mat2;
        }
        uint Matrix_time = tock("Eigen long operations");

        tick();
        for (int i = 0; i < 10000; ++i)
        {
            fmat5 = fmat1 + fmat2 + fmat3 - fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2 + fmat3 - fmat4 + fmat1 - fmat2 + fmat3 + fmat4 + fmat1 - fmat2;
        }

        uint FusedMatrix_time = tock("FusedMatrix long operations");

        // Percentage Increase
        double percentage_increase = (((double)FusedMatrix_time - (double)Matrix_time) / (double)Matrix_time) * 100.0;

        std::cout << "Percentage decrease in time for FusedMatrix compared to Eigen: "
                  << percentage_increase << "%" << std::endl;

        // // fmat4.print();
        // // std::cout << std::endl;
        // // mat4.print();
    }
}
