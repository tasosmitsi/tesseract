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

    SECTION("Matrix transpose")
    {
        // TODO: test + benchmark
    }

    SECTION("Matrix matmul")
    {
        // TODO: test + benchmark
    }

    SECTION("Matrix inverse")
    {
        // TODO: test + benchmark
    }
}