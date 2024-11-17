#include <catch_amalgamated.hpp>
#define CATCH_CONFIG_MAIN

#include "matrix.h"

TEST_CASE("Matrix class", "[matrix]")
{
    Matrix<double, 10, 10> mat1(1), mat2(2), mat4(10);
    Matrix<double, 2, 3> matrix3(2);

    SECTION("Matrix accessing elements")
    {
        mat1.setIdentity()(0, 9) = 45.654;

        REQUIRE(mat1(0, 9) == 45.654);
    }

    SECTION("Matrix total size, number of dimensions, and shape")
    {
        Matrix<double, 2, 2> matrix;
        Matrix<double, 15, 32> matrix1;

        REQUIRE(matrix.getTotalSize() == 4);
        REQUIRE(matrix.getNumDims() == 2);
        REQUIRE(matrix.getShape() == "(2,2)");

        REQUIRE(matrix1.getTotalSize() == 480);
        REQUIRE(matrix1.getNumDims() == 2);
        REQUIRE(matrix1.getShape() == "(15,32)");
    }

    SECTION("Is matrix identity")
    {
        mat1.setIdentity();
        REQUIRE(mat1.isIdentity());

        mat1(0, 0) = 15;
        REQUIRE_FALSE(mat1.isIdentity());

        mat1.setIdentity();
        // check if all diagonal elements are 1
        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                if (i == j)
                {
                    REQUIRE(mat1(i, j) == 1);
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
                REQUIRE(mat1(i, j) == 0);
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
                REQUIRE(mat1(i, j) == value);
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
                REQUIRE(mat1(i, j) == i * mat1.getDim(1) + j);
            }
        }
    }

    SECTION("Are matrices equal")
    {
        mat1.setIdentity();
        mat2.setIdentity();

        REQUIRE(mat1 == mat2);

        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                REQUIRE(mat1(i, j) == mat2(i, j));
            }
        }

        mat1(1, 2) = 3.0;

        REQUIRE_FALSE(mat1 == mat2);
    }

    SECTION("Is matrix diagonal")
    {
        mat1.setDiagonal(1);
        REQUIRE(mat1.isIdentity());

        mat1(1, 2) = 3.0;
        REQUIRE_FALSE(mat1.isIdentity());

        mat1.setDiagonal(5);

        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                if (i == j)
                {
                    REQUIRE(mat1(i, j) == 5);
                }
                else
                {
                    REQUIRE(mat1(i, j) == 0);
                }
            }
        }
    }

    SECTION("Matrix operations")
    {
        Matrix<double, 10, 10> mat1, mat2, mat4,
            mat5, mat6, mat7, mat8, mat9, mat10,
            mat11, mat12, mat13, mat14, mat15,
            mat16, mat17, mat18, mat19;

        mat1.setIdentity();
        mat2.setIdentity();

        mat4 = mat1 + mat2;
        mat5 = mat2 + mat1;
        mat12 = mat1 + 2;
        mat13 = 2 + mat1;

        mat6 = mat1 - mat2;
        mat7 = mat2 - mat1;
        mat14 = mat1 - 2;
        mat15 = 2 - mat1;

        mat8 = mat1 * mat2;
        mat9 = mat2 * mat1;
        mat16 = mat1 * 2;
        mat17 = 2 * mat1;

        mat1.setHomogen(2);
        mat2.setHomogen(4);

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
                    REQUIRE(mat4(i, j) == 2);
                    REQUIRE(mat5(i, j) == 2);

                    REQUIRE(mat8(i, j) == 1);
                    REQUIRE(mat9(i, j) == 1);

                    REQUIRE(mat12(i, j) == 3);
                    REQUIRE(mat13(i, j) == 3);

                    REQUIRE(mat14(i, j) == -1);
                    REQUIRE(mat15(i, j) == 1);

                    REQUIRE(mat16(i, j) == 2);
                    REQUIRE(mat17(i, j) == 2);
                }
                else
                {
                    // check only the non-diagonal elements
                    REQUIRE(mat4(i, j) == 0);
                    REQUIRE(mat5(i, j) == 0);

                    REQUIRE(mat8(i, j) == 0);
                    REQUIRE(mat9(i, j) == 0);

                    REQUIRE(mat12(i, j) == 2);
                    REQUIRE(mat13(i, j) == 2);

                    REQUIRE(mat14(i, j) == -2);
                    REQUIRE(mat15(i, j) == 2);

                    REQUIRE(mat16(i, j) == 0);
                    REQUIRE(mat17(i, j) == 0);
                }

                // check all elements
                REQUIRE(mat6(i, j) == 0);
                REQUIRE(mat7(i, j) == 0);

                REQUIRE(mat10(i, j) == 0.5);
                REQUIRE(mat11(i, j) == 2);

                REQUIRE(mat18(i, j) == 1);
                REQUIRE(mat19(i, j) == 1);
            }
        }
    }
}