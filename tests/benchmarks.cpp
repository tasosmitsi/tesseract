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
}
