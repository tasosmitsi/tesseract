#include <catch_amalgamated.hpp>
#include "fused/fused_tensor.h"
#include <cxxabi.h>

TEST_CASE("FusedTensorND class", "[fused_tensor]")
{
    FusedTensorND<double, 10, 10> ten1(1), ten2(2);

    SECTION("FusedTensorND accessing elements")
    {
        ten1.setIdentity()(0, 9) = 45.654;

        CHECK(ten1(0, 9) == 45.654);
    }

    SECTION("FusedTensorND total size, number of dimensions, and shape")
    {
        FusedTensorND<double, 2, 2> tensor;
        FusedTensorND<double, 15, 32> tensor1;

        CHECK(tensor.getTotalSize() == 4);
        CHECK(tensor.getNumDims() == 2);
        CHECK(tensor.getShape() == "(2,2)");

        CHECK(tensor1.getTotalSize() == 480);
        CHECK(tensor1.getNumDims() == 2);
        CHECK(tensor1.getShape() == "(15,32)");
    }

    SECTION("Is FusedTensorND identity")
    {
        ten1.setIdentity();
        CHECK(ten1.isIdentity());

        ten1(0, 0) = 15;
        CHECK_FALSE(ten1.isIdentity());

        ten1.setIdentity();
        // check if all diagonal elements are 1
        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                if (i == j)
                {
                    CHECK(ten1(i, j) == 1);
                }
            }
        }
    }

    SECTION("Is FusedTensorND full of zeros")
    {
        ten1.setToZero();
        // check if all elements are 0
        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                CHECK(ten1(i, j) == 0);
            }
        }
    }

    SECTION("Is FusedTensorND homogeneous")
    {
        double value = 13.3;
        ten1.setHomogen(value);
        // check if all elements are 5
        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                CHECK(ten1(i, j) == value);
            }
        }
    }

    SECTION("Is FusedTensorND sequential")
    {
        ten1.setSequencial();
        // check if all elements are sequential
        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                CHECK(ten1(i, j) == i * ten1.getDim(1) + j);
            }
        }
    }

    SECTION("FusedTensorND check if non inplace transpose yields correct results in fused tensor operations")
    {
        FusedTensorND<double, 2, 3> tensor1(2), result;
        FusedTensorND<double, 3, 2> tensor2(2);

        // non-inplace transpose
        result = 10.0 + tensor1 * tensor2.transposed() + tensor2.transposed() + 10.0;
        // check if the result is correct
        for (size_t i = 0; i < tensor1.getDim(0); ++i)
        {
            for (size_t j = 0; j < tensor2.getDim(1); ++j)
            {
                CHECK(result(i, j) == 26);
            }
        }
    }

    SECTION("Are FusedTensorNDs equal")
    {
        ten1.setIdentity();
        ten2.setIdentity();

        CHECK(ten1 == ten2);

        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                CHECK(ten1(i, j) == ten2(i, j));
            }
        }

        ten1(1, 2) = 3.0;
        CHECK_FALSE(ten1 == ten2);

        // now check in case of transpose
        ten1.inplace_transpose();
        ten2(1, 2) = 3.0;
        CHECK_FALSE(ten1 == ten2);
    }

    SECTION("Check dimensions mismatch and == , != operators")
    {
        // this test should fail when the dimensions of the matrices are not equal
        // and should pass when the dimensions are equal even after transposing one of the matrices
        FusedTensorND<double, 2, 3> tensor1(2);
        FusedTensorND<double, 3, 2> tensor2(2);

        CHECK_THROWS(tensor1 == tensor2);
        CHECK_THROWS(tensor1 != tensor2);

        tensor2.inplace_transpose();
        CHECK_NOTHROW(tensor1 == tensor2);
        CHECK_FALSE(tensor1 != tensor2);
    }

    SECTION("Assign FusedTensorND to another FusedTensorND")
    {
        ten1.setIdentity();
        ten2 = ten1;

        CHECK(ten1 == ten2);

        ten1(1, 2) = 3.0;

        CHECK_FALSE(ten1 == ten2);
    }

    SECTION("Is FusedTensorND diagonal")
    {
        ten1.setDiagonal(1);
        CHECK(ten1.isIdentity());

        ten1(1, 2) = 3.0;
        CHECK_FALSE(ten1.isIdentity());

        ten1.setDiagonal(5);

        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                if (i == j)
                {
                    CHECK(ten1(i, j) == 5);
                }
                else
                {
                    CHECK(ten1(i, j) == 0);
                }
            }
        }
    }

    SECTION("FusedTensorND elementary operations")
    {
        // TODO: Split this section into smaller sections
        // one for addition, one for subtraction,
        // one for multiplication, one for division

        FusedTensorND<double, 10, 10>
            ten1, ten2, ten4, ten5,
            ten6, ten7, ten8, ten9,
            ten10, ten11, ten12, ten13,
            ten14, ten15, ten16, ten17,
            ten18, ten19, ten20, ten21,
            ten22, ten23;

        ten1.setIdentity();
        ten2.setIdentity();

        // additon
        ten4 = ten1 + ten2;
        ten5 = ten2 + ten1;
        ten12 = ten1 + 2.0;
        ten13 = 2.0 + ten1;
        ten22 = ten1 + (-2.0);
        ten23 = -2.0 + ten1;

        // subtraction
        ten6 = ten1 - ten2;
        ten7 = ten2 - ten1;
        ten14 = ten1 - 2.0;
        ten15 = 2.0 - ten1;

        ten20 = -ten1;
        ten21 = -ten13;

        // multiplication
        ten8 = ten1 * ten2;
        ten9 = ten2 * ten1;
        ten16 = ten1 * 2.0;
        ten17 = 2.0 * ten1;

        // division
        ten1.setHomogen(4);
        ten2.setHomogen(8);

        ten10 = ten1 / ten2;
        ten11 = ten2 / ten1;
        ten18 = ten1 / 2.0;
        ten19 = 2.0 / ten1;

        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                if (i == j)
                {
                    // Check only the diagonal elements
                    CHECK(ten4(i, j) == 2);
                    CHECK(ten5(i, j) == 2);

                    CHECK(ten8(i, j) == 1);
                    CHECK(ten9(i, j) == 1);

                    CHECK(ten12(i, j) == 3);
                    CHECK(ten13(i, j) == 3);

                    CHECK(ten14(i, j) == -1);
                    CHECK(ten15(i, j) == 1);

                    CHECK(ten16(i, j) == 2);
                    CHECK(ten17(i, j) == 2);

                    CHECK(ten20(i, j) == -1);
                    CHECK(ten21(i, j) == -3);

                    CHECK(ten22(i, j) == -1);
                    CHECK(ten23(i, j) == -1);
                }
                else
                {
                    // check only the non-diagonal elements
                    CHECK(ten4(i, j) == 0);
                    CHECK(ten5(i, j) == 0);

                    CHECK(ten8(i, j) == 0);
                    CHECK(ten9(i, j) == 0);

                    CHECK(ten12(i, j) == 2);
                    CHECK(ten13(i, j) == 2);

                    CHECK(ten14(i, j) == -2);
                    CHECK(ten15(i, j) == 2);

                    CHECK(ten16(i, j) == 0);
                    CHECK(ten17(i, j) == 0);

                    CHECK(ten20(i, j) == 0);
                    CHECK(ten21(i, j) == -2);

                    CHECK(ten22(i, j) == -2);
                    CHECK(ten23(i, j) == -2);
                }

                // check all elements
                CHECK(ten6(i, j) == 0);
                CHECK(ten7(i, j) == 0);

                CHECK(ten10(i, j) == 0.5);
                CHECK(ten11(i, j) == 2);

                CHECK(ten18(i, j) == 2);
                CHECK(ten19(i, j) == 0.5);
            }
        }
    }

    SECTION("FusedTensorND test fusion opperations")
    {
        FusedTensorND<double, 10, 10> ten1, ten2, ten3, ten4,
            ten5, ten6, ten7, ten8,
            ten9, ten10, ten11, ten12,
            ten13, ten14, ten15, ten16,
            ten17, ten18, ten19, ten20,
            ten21, ten22, ten23;
        ten1.setIdentity();
        ten2.setIdentity();

        ten3 = ten1 + ten2 + 2.0;
        ten4 = ten1 + ten2 + ten3;
        ten5 = ten1 + ten2 + ten3 + 2.0;
        ten6 = ten1 + ten2 + ten3 + ten4 + 2.0;
        ten7 = 2.0 - 1.0 + ten1 + ten2 * 3.0 + ten3 + ten4 + ten5 + 2.0;

        // check if the result is correct
        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                if (i == j)
                {
                    CHECK(ten3(i, j) == 4);
                    CHECK(ten4(i, j) == 6);
                    CHECK(ten5(i, j) == 8);
                    CHECK(ten6(i, j) == 14);
                    CHECK(ten7(i, j) == 25);
                }
                else
                {
                    CHECK(ten3(i, j) == 2);
                    CHECK(ten4(i, j) == 2);
                    CHECK(ten5(i, j) == 4);
                    CHECK(ten6(i, j) == 6);
                    CHECK(ten7(i, j) == 11);
                }
            }
        }
    }

    SECTION("FusedTensorND testing dimentions after transpose")
    {
        FusedTensorND<double, 2, 3> tensor;
        auto transposed = tensor.transposed();

        // Check tensor, it should not be transposed
        CHECK(tensor.getNumDims() == 2);
        CHECK(tensor.getShape() == "(2,3)");
        CHECK(tensor.getDim(0) == 2);
        CHECK(tensor.getDim(1) == 3);

        // Check transposed tensor, it should be transposed
        CHECK(transposed.getNumDims() == 2);
        CHECK(transposed.getShape() == "(3,2)");
        CHECK(transposed.getDim(0) == 3);
        CHECK(transposed.getDim(1) == 2);

        // Now lets transpose the tensor in place
        tensor.inplace_transpose();
        // Check tensor, it should be transposed
        CHECK(tensor.getNumDims() == 2);
        CHECK(tensor.getShape() == "(3,2)");
        CHECK(tensor.getDim(0) == 3);
        CHECK(tensor.getDim(1) == 2);

        // now lets test a higher order tensor
        FusedTensorND<double, 2, 3, 4> tensor1;
        size_t order[] = {2, 1, 0};

        auto transposed1 = tensor1.transposed(order);

        // Check tensor, it should not be transposed
        CHECK(tensor1.getNumDims() == 3);
        CHECK(tensor1.getShape() == "(2,3,4)");
        CHECK(tensor1.getDim(0) == 2);
        CHECK(tensor1.getDim(1) == 3);
        CHECK(tensor1.getDim(2) == 4);

        // Check transposed1 tensor, it should be transposed
        CHECK(transposed1.getNumDims() == 3);
        CHECK(transposed1.getShape() == "(4,3,2)");
        CHECK(transposed1.getDim(0) == 4);
        CHECK(transposed1.getDim(1) == 3);
        CHECK(transposed1.getDim(2) == 2);

        // Now lets transpose the tensor1 in place
        tensor1.inplace_transpose(order);
        // Check tensor1, it should be transposed
        CHECK(tensor1.getNumDims() == 3);
        CHECK(tensor1.getShape() == "(4,3,2)");
        CHECK(tensor1.getDim(0) == 4);
        CHECK(tensor1.getDim(1) == 3);
        CHECK(tensor1.getDim(2) == 2);
    }

    SECTION("Check dimensions mismatch on addition, subtraction, multiplication, and division")
    {
        // this test should fail when the dimensions of the matrices are not equal
        // and should pass when the dimensions are equal even after transposing one of the matrices

        FusedTensorND<double, 2, 3> tensor1(2);
        FusedTensorND<double, 3, 2> tensor2(2);
        FusedTensorND<double, 3, 2> tensor3;

        CHECK_THROWS(tensor3 = tensor1 + tensor2);
        CHECK_THROWS(tensor3 = tensor1 - tensor2);
        CHECK_THROWS(tensor3 = tensor1 * tensor2);
        CHECK_THROWS(tensor3 = tensor1 / tensor2);

        tensor1.inplace_transpose();

        CHECK_NOTHROW(tensor3 = tensor1 + tensor2);
        CHECK_NOTHROW(tensor3 = tensor1 - tensor2);
        CHECK_NOTHROW(tensor3 = tensor1 * tensor2);
        CHECK_NOTHROW(tensor3 = tensor1 / tensor2);
    }

    SECTION("FusedTensorND transpose")
    {
        FusedTensorND<double, 10, 10> ten3, ten4;
        ten1.setRandom(-10, 10);
        ten2 = ten1;

        // check inplace transpose first
        ten1.inplace_transpose();

        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                CHECK(ten1(i, j) == ten2(j, i));
            }
        }

        // check non-inplace transpose
        ten1.setRandom(-10, 10);
        ten2 = ten1.transposed();

        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                CHECK(ten1(i, j) == ten2(j, i));
            }
        }

        // now check a long oppeartion by adding a zero matrix to the
        // transposed (not in place) matrix. The ten1 should not change.
        ten1.setIdentity();
        ten1(0, 1) = 10;
        ten2.setToZero();

        ten3 = ten1.transposed() + ten2;
        ten4 = ten1 + ten2;

        // // In both cases, ten2 is a zero matrix (should not change the result)
        // // ten3 should not be equal to ten1 because of the transpose
        CHECK(ten3 != ten1);
        CHECK(ten3(1, 0) == 10);

        // ten4 should be equal to ten1
        CHECK(ten4 == ten1);
    }

    SECTION("Test FusedTensorND einsum operation")
    {
        FusedTensorND<double, 2, 3> tensor1(2), tensor2(2);
        FusedTensorND<double, 3, 2> tensor3(2);

        auto result = FusedTensorND<double, 2, 2>::einsum(tensor1, tensor2, 1, 1);

        CHECK(result.getNumDims() == 2);
        CHECK(result.getShape() == "(2,2)");
        CHECK(result.getDim(0) == 2);
        CHECK(result.getDim(1) == 2);

        auto result1 = FusedTensorND<double, 3, 3>::einsum(tensor1, tensor3, 0, 1);

        CHECK(result1.getNumDims() == 2);
        CHECK(result1.getShape() == "(3,3)");
        CHECK(result1.getDim(0) == 3);
        CHECK(result1.getDim(1) == 3);
    }
}
