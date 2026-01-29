#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "utilities.h"
#include "matrix_algorithms.h"
#include <Dense>

TEMPLATE_TEST_CASE("FusedMatrix class", "[fused_matrix]", double, float)
{
    using T = TestType;

    FusedMatrix<T, 10, 10> mat1(1), mat2(2), mat3, mat4(10);

    SECTION("FusedMatrix elements access")
    {
        mat1.setIdentity()(0, 9) = (T)45.654;

        CHECK(mat1(0, 9) == (T)45.654);
    }

    SECTION("FusedMatrix total size, number of dimensions, and shape")
    {
        FusedMatrix<T, 2, 2> matrix;
        FusedMatrix<T, 15, 32> matrix1;

        CHECK(matrix.getTotalSize() == 4);
        CHECK(matrix.getNumDims() == 2);
        CHECK(matrix.getShape() == "(2,2)");

        CHECK(matrix1.getTotalSize() == 480);
        CHECK(matrix1.getNumDims() == 2);
        CHECK(matrix1.getShape() == "(15,32)");
    }

    SECTION("Is matrix identity")
    {
        // set matrix to identity
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

        // check if mat1.isIdentity() returns true indeed
        CHECK(mat1.isIdentity());

        // change one diagonal element to something other than 1
        mat1(0, 0) = 15;
        CHECK_FALSE(mat1.isIdentity());
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
        T value = (T)13.3;
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
        mat2(1, 2) = 3.0;
        CHECK_FALSE(mat1.transpose_view() == mat2);
    }

    SECTION("Matrix min/max operators with transpose views as part of the expression")
    {
        mat1.setSequencial();

        mat3 = min(max(mat1.transpose_view(), (T)5.0), (T)10.0);

        // check if all elements are between 5 and 10
        for (size_t i = 0; i < mat3.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat3.getDim(1); ++j)
            {
                CHECK(mat3(i, j) >= 5.0);
                CHECK(mat3(i, j) <= 10.0);
            }
        }
    }

    SECTION("Check dimensions mismatch and == , !=, min, max operators")
    {
        // this test should fail when the dimensions of the matrices are not equal
        // and should pass when the dimensions are equal even after transposing one of the matrices
        FusedMatrix<T, 2, 3> matrix1(2);
        FusedMatrix<T, 3, 2> matrix2(2);

        CHECK_THROWS(matrix1 == matrix2);
        CHECK_THROWS(matrix1 != matrix2);
        CHECK_THROWS(min(matrix1, matrix2));
        CHECK_THROWS(max(matrix1, matrix2));

        CHECK_NOTHROW(matrix1 == matrix2.transpose_view());
        CHECK_NOTHROW(min(matrix1, matrix2.transpose_view()));
        CHECK_NOTHROW(max(matrix1, matrix2.transpose_view()));

        CHECK_FALSE(matrix1 != matrix2.transpose_view());
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
        // TODO: transpose mat1
        // mat1.inplace_transpose();
        // // perform upper triangular on the transposed matrix
        // mat1.upperTriangular(true);
        // // mat1 should stil be upper triangular
        // CHECK(mat1.isUpperTriangular());
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

        // // TODO: transpose mat1
        // mat1.inplace_transpose();
        // // perform lower triangular on the transposed matrix
        // mat1.lowerTriangular(true);
        // // mat1 should stil be lower triangular
        // CHECK(mat1.isLowerTriangular());
    }

    SECTION("FusedMatrix elementary operations")
    {
        SECTION("addition")
        {
            FusedMatrix<T, 10, 10>
                mat1, mat2, mat3, mat4,
                mat5, mat6, mat7, mat8;

            mat1.setIdentity();
            mat2.setIdentity();

            mat3 = mat1 + mat2;
            mat4 = mat2 + mat1;
            mat5 = mat1 + (T)2.0;
            mat6 = (T)2.0 + mat1;
            mat7 = mat1 + (T)(-2.0);
            mat8 = (T)-2.0 + mat1;

            for (size_t i = 0; i < mat1.getDim(0); ++i)
            {
                for (size_t j = 0; j < mat1.getDim(1); ++j)
                {
                    CHECK(mat3(i, j) == (mat1(i, j) + mat2(i, j)));
                    CHECK(mat4(i, j) == (mat2(i, j) + mat1(i, j)));
                    CHECK(mat5(i, j) == (mat1(i, j) + (T)2.0));
                    CHECK(mat6(i, j) == (mat1(i, j) + (T)2.0));
                    CHECK(mat7(i, j) == (mat1(i, j) + (T)(-2.0)));
                    CHECK(mat8(i, j) == ((T)(-2.0) + mat1(i, j)));
                }
            }
        }

        SECTION("subtraction")
        {
            FusedMatrix<T, 10, 10>
                mat1, mat2, mat3, mat4,
                mat5, mat6, mat7;

            mat1.setIdentity();
            mat2.setIdentity();

            mat3 = mat1 - mat2;
            mat4 = mat2 - mat1;
            mat5 = mat1 - (T)2.0;
            mat6 = (T)2.0 - mat1;

            mat7 = -mat1;

            for (size_t i = 0; i < mat1.getDim(0); ++i)
            {
                for (size_t j = 0; j < mat1.getDim(1); ++j)
                {
                    CHECK(mat3(i, j) == (mat1(i, j) - mat2(i, j)));
                    CHECK(mat4(i, j) == (mat2(i, j) - mat1(i, j)));
                    CHECK(mat5(i, j) == (mat1(i, j) - (T)2.0));
                    CHECK(mat6(i, j) == ((T)2.0 - mat1(i, j)));
                    CHECK(mat7(i, j) == (-mat1(i, j)));
                }
            }
        }

        SECTION("multiplication")
        {
            FusedMatrix<T, 10, 10>
                mat1, mat2, mat3,
                mat4, mat5, mat6;

            mat1.setIdentity();
            mat2.setIdentity();

            mat3 = mat1 * mat2;
            mat4 = mat2 * mat1;
            mat5 = mat1 * (T)2.0;
            mat6 = (T)2.0 * mat1;

            for (size_t i = 0; i < mat1.getDim(0); ++i)
            {
                for (size_t j = 0; j < mat1.getDim(1); ++j)
                {
                    CHECK(mat3(i, j) == (mat1(i, j) * mat2(i, j)));
                    CHECK(mat4(i, j) == (mat2(i, j) * mat1(i, j)));
                    CHECK(mat5(i, j) == (mat1(i, j) * (T)2.0));
                    CHECK(mat6(i, j) == (mat1(i, j) * (T)2.0));
                }
            }
        }

        SECTION("division")
        {
            FusedMatrix<T, 10, 10>
                mat1, mat2, mat3,
                mat4, mat5, mat6;

            mat1.setHomogen(4);
            mat2.setHomogen(8);

            mat3 = mat1 / mat2;
            mat4 = mat2 / mat1;
            mat5 = mat1 / (T)2.0;
            mat6 = (T)2.0 / mat1;

            for (size_t i = 0; i < mat1.getDim(0); ++i)
            {
                for (size_t j = 0; j < mat1.getDim(1); ++j)
                {
                    CHECK(mat3(i, j) == (mat1(i, j) / mat2(i, j)));
                    CHECK(mat4(i, j) == (mat2(i, j) / mat1(i, j)));
                    CHECK(mat5(i, j) == (mat1(i, j) / (T)2.0));
                    CHECK(mat6(i, j) == ((T)2.0 / mat1(i, j)));
                }
            }
        }
    }

    SECTION("FusedMatrix test fused operations")
    {
        FusedMatrix<T, 10, 10>
            mat1, mat2, mat3, mat4,
            mat5, mat6, mat7;

        mat1.setIdentity();
        mat2.setIdentity();

        mat3 = mat1 + mat2 + (T)2.0;
        mat4 = mat1 + mat2 + mat3;
        mat5 = mat1 + mat2 + mat3 + (T)2.0;
        mat6 = mat1 + mat2 + mat3 + mat4 + (T)2.0;
        mat7 = (T)2.0 - (T)1.0 + mat1 + mat2 * (T)3.0 + mat3 + mat4 + mat5 + (T)2.0;

        // check if the result is correct
        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                CHECK(mat3(i, j) == (mat1(i, j) + mat2(i, j) + (T)2.0));
                CHECK(mat4(i, j) == (mat1(i, j) + mat2(i, j) + mat3(i, j)));
                CHECK(mat5(i, j) == (mat1(i, j) + mat2(i, j) + mat3(i, j) + (T)2.0));
                CHECK(mat6(i, j) == (mat1(i, j) + mat2(i, j) + mat3(i, j) + mat4(i, j) + (T)2.0));
                CHECK(mat7(i, j) == ((T)2.0 - (T)1.0 + mat1(i, j) + mat2(i, j) * (T)3.0 + mat3(i, j) + mat4(i, j) + mat5(i, j) + (T)2.0));
            }
        }
    }

    SECTION("Check dimensions mismatch on addition, subtraction, multiplication, and division")
    {
        // this test should fail when the dimensions of the matrices are not equal
        // and should pass when the dimensions are equal even after transposing one of the matrices

        FusedMatrix<T, 2, 3> matrix1(2);
        FusedMatrix<T, 3, 2> matrix2(2);
        FusedMatrix<T, 3, 2> matrix3;

        CHECK_THROWS(matrix3 = matrix1 + matrix2);
        CHECK_THROWS(matrix3 = matrix1 - matrix2);
        CHECK_THROWS(matrix3 = matrix1 * matrix2);
        CHECK_THROWS(matrix3 = matrix1 / matrix2);

        CHECK_NOTHROW(matrix3 = matrix1.transpose_view() + matrix2);
        CHECK_NOTHROW(matrix3 = matrix1.transpose_view() - matrix2);
        CHECK_NOTHROW(matrix3 = matrix1.transpose_view() * matrix2);
        CHECK_NOTHROW(matrix3 = matrix1.transpose_view() / matrix2);
    }

    SECTION("Check operations after transpose")
    {
        FusedMatrix<T, 4, 4> matrix1, matrix2, res, res1;
        matrix1.setSequencial();
        matrix2.setSequencial();

        res = matrix1.transpose_view() + matrix2;
        res1 = matrix1 + matrix2;

        CHECK(res != res1);
    }

    SECTION("FusedMatrix transpose")
    {
        mat1.setRandom(-10, 10);
        mat2 = mat1;

        // check transpose view first
        for (size_t i = 0; i < mat1.getDim(0); ++i)
        {
            for (size_t j = 0; j < mat1.getDim(1); ++j)
            {
                CHECK(mat1.transpose_view()(i, j) == mat2(j, i));
            }
        }

        // now check a long oppeartion by adding a zero matrix to the
        // transposed (not in place) matrix. The mat1 should not change.
        mat1.setIdentity();
        mat1(0, 1) = (T)10;
        mat2.setToZero();

        mat3 = mat1.transpose_view() + mat2;
        mat4 = mat1 + mat2;

        // In both cases, mat2 is a zero matrix (should not change the result)
        // mat3 should not be equal to mat1 because of the transpose
        CHECK(mat3 != mat1);
        CHECK(mat3(1, 0) == 10);

        // mat4 should be equal to mat1
        CHECK(mat4 == mat1);
    }

    SECTION("FusedMatrix matmul")
    {
        FusedMatrix<T, 2, 3> matrix1(2);
        FusedMatrix<T, 3, 2> matrix2(2);

        matrix1.setHomogen(10);
        matrix2.setHomogen(33);

        auto res = FusedMatrix<T, 2, 2>::matmul(matrix1, matrix2);

        // check the dimensions of the res matrix
        CHECK(res.getDim(0) == 2);
        CHECK(res.getDim(1) == 2);

        // check the values of the res matrix
        for (size_t i = 0; i < res.getDim(0); ++i)
        {
            for (size_t j = 0; j < res.getDim(1); ++j)
            {
                T sum = 0;
                for (size_t k = 0; k < matrix1.getDim(1); ++k)
                {
                    sum += matrix1(i, k) * matrix2(k, j);
                }
                CHECK(res(i, j) == sum);
            }
        }
    }

    SECTION("FusedMatrix inverse")
    {
        // init the matrix
        T initValues[4][4] = {
            {2.0, -1, 2.0, -1},
            {4, 5.0, 2.5, -17},
            {2.0, -1, 2.43, -30},
            {4, 5.0, 245, -10}};
        FusedMatrix<T, 4, 4> matrix3 = initValues;

        // using tessaract
        auto inv = matrix3.inverse();

        // using Eigen for validation
        Eigen::Matrix<T, 4, 4> eigen_matrix;
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                eigen_matrix(i, j) = initValues[i][j];
            }
        }
        Eigen::Matrix<T, 4, 4> eigen_inv = eigen_matrix.inverse();

        // check if the inverse is correct
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                CHECK_THAT(inv(i, j), Catch::Matchers::WithinRel(eigen_inv(i, j), (T)1e-3));
            }
        }
    }

    SECTION("Test Cholesky Decomposition")
    {
        // init the matrix
        T initValues[3][3] = {
            {4, 12, -16},
            {12, 37, -43},
            {-16, -43, 98}};
        FusedMatrix<T, 3, 3> matrix3 = initValues;

        T cholesky_values[3][3] = {
            {2, 0, 0},
            {6, 1, 0},
            {-8, 5, 3}};
        FusedMatrix<T, 3, 3> cholesky_matrix = cholesky_values;
        FusedMatrix<T, 3, 3> cholesky;

        // using tessaract
        cholesky = matrix_algorithms::choleskyDecomposition(matrix3);

        CHECK(cholesky == cholesky_matrix);

        // using Eigen for validation
        Eigen::Matrix<T, 3, 3> eigen_matrix;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                eigen_matrix(i, j) = initValues[i][j];
            }
        }
        Eigen::LLT<Eigen::Matrix<T, 3, 3>> lltOfA(eigen_matrix);
        Eigen::Matrix<T, 3, 3> L = lltOfA.matrixL();
        // check if the cholesky is correct
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CHECK_THAT(cholesky(i, j), Catch::Matchers::WithinRel(L(i, j), (T)1e-3));
            }
        }
    }

    SECTION("Is matrix positive definite or semi-definite")
    {
        // init the matrix
        T initValues[3][3] = {
            {4, 12, -16},
            {12, 37, -43},
            {-16, -43, 98}};
        FusedMatrix<T, 3, 3> matrix = initValues;
        auto result = matrix.isPositiveDefinite();
        CHECK(result == matrix_traits::Definiteness::PositiveDefinite);

        // TODO: test semi definite matrix
        // T initValues1[2][2] = {
        //     {3, 4},
        //     {4, 16/3}};

        // FusedMatrix<T, 2, 2> matrix1 = initValues1;
        // auto result = matrix1.isPositiveDefinite(true);
        // CHECK(result == matrix_traits::Definiteness::PositiveSemiDefinite);

        // Not positive definite matrix
        T initValues2[3][3] = {
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
        T initValues[4][4] = {
            {1, 0, 0, 0},
            {0, 0, -1, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 1}};

        FusedMatrix<T, 4, 4> matrix3 = initValues;

        CHECK(matrix3.isOrthogonal());

        matrix3(0, 0) = 2;
        CHECK_FALSE(matrix3.isOrthogonal());
    }
}
