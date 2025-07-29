#include <catch_amalgamated.hpp>

#include "fused/fused_vector.h"

TEST_CASE("FusedVector class", "[fused_vector]")
{
    FusedVector<double, 5> vec1(1.1), vec2(2.0), vec3;
    FusedMatrix<double, 1, 5> mat1(10);
    FusedMatrix<double, 5, 5> mat2(10);

    SECTION("FusedVector accessing elements")
    {
        vec1(0) = 3.14;
        CHECK(vec1(0) == 3.14);
    }

    SECTION("FusedVector total size, number of dimensions, and shape")
    {
        FusedVector<double, 2> vector;
        FusedVector<double, 15> vector1;

        CHECK(vector.getTotalSize() == 2);
        CHECK(vector.getNumDims() == 2);
        CHECK(vector.getShape() == "(2,1)");

        CHECK(vector1.getTotalSize() == 15);
        CHECK(vector1.getNumDims() == 2);
        CHECK(vector1.getShape() == "(15,1)");
    }

    SECTION("Is FusedVector identity")
    {
        FusedVector<double, 2> vector(0);
        vector(0) = 1;
        CHECK_FALSE(vec1.isIdentity());
    }

    SECTION("FusedVector total size, number of dimensions, and shape")
    {
        CHECK(vec1.getTotalSize() == 5);
        CHECK(vec1.getNumDims() == 2);
        CHECK(vec1.getShape() == "(5,1)");


        CHECK(vec1.transposed().getShape() == "(1,5)");

        vec1.print();

        vec3 = vec1 + vec2;


        std::cout << "Type: " << typeid(vec3).name() << std::endl;

        FusedVector<double, 5> vec4;

        vec4.setSequencial();
        // vec4.inplace_transpose();
        // std::cout << vec4.getShape() << std::endl;
        // vec4.print();

        // multiply two vectors matmul

        mat1.print();
        auto matmul_res = FusedTensorND<double, 5, 5>::einsum(vec1, mat1, 1, 0);

        matmul_res.print();
        std::cout << "Type: " << typeid(matmul_res).name() << std::endl;

        auto matmul_res1 = FusedMatrix<double, 5, 5>::matmul(vec1, mat1);
        matmul_res1.print();
        std::cout << "Type: " << typeid(matmul_res1).name() << std::endl;

        // TensorND<double, 1,1> diagonalEntries;

        // std::cout << "# dims: " << vec3.getNumDims() << std::endl;

        // std::cout << "Shape: " << vec3.getShape() << std::endl;

        // mat1.getDiagonalEntries(diagonalEntries);

        // diagonalEntries.print();

        // std::cout << "Shape: " << diagonalEntries.getShape() << std::endl;
    }
}
