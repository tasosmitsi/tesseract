#include <catch_amalgamated.hpp>
#include "fused/fused_tensor.h"

TEMPLATE_TEST_CASE("FusedTensorND class", "[fused_tensor]", double, float)
{
    using T = TestType;

    FusedTensorND<T, 10, 10> ten1(1), ten2(2);

    SECTION("FusedTensorND elements access")
    {
        ten1.setIdentity()(0, 9) = (T)45.0;
        CHECK(ten1(0, 9) == (T)45.0);
    }

    SECTION("FusedTensorND equality operators with transpose views as part of the expression")
    {
        FusedTensorND<T, 10, 10> transposedTensor;
        FusedTensorND<T, 10, 10> tensor;

        tensor.setHomogen((T)1.0);
        transposedTensor.setHomogen((T)1.0);

        // this sould pass
        CHECK((tensor + (T)1.0) == transposedTensor.transpose_view() + (T)1.0);
        CHECK((transposedTensor.transpose_view() + (T)1.0) == (tensor + (T)1.0));
    }

    SECTION("FusedTensorND min/max operators with transpose views as part of the expression")
    {
        FusedTensorND<T, 5, 6> fmat1, result;
        FusedTensorND<T, 6, 5> result_of_transpose;

        fmat1.setSequencial();

        result = min(max(fmat1, (T)5.0), (T)10.0);
        result_of_transpose = min(max(fmat1.transpose_view(), (T)5.0), (T)10.0);

        // check if all elements are within the range [5.0, 10.0]
        for (size_t i = 0; i < result.getDim(0); ++i)
        {
            for (size_t j = 0; j < result.getDim(1); ++j)
            {
                CHECK(result(i, j) >= (T)5.0);
                CHECK(result(i, j) <= (T)10.0);
                CHECK(result_of_transpose(j, i) >= (T)5.0);
                CHECK(result_of_transpose(j, i) <= (T)10.0);
            }
        }
    }

    SECTION("FusedTensorND min/max reduction operators and as part of expression")
    {
        FusedTensorND<T, 5, 6> fmat1;
        fmat1.setSequencial();

        T min_value = min(fmat1);
        T max_value = max(fmat1);

        CHECK(min_value == (T)0.0);
        CHECK(max_value == (T)(5 * 6 - 1));

        min_value = min(fmat1 + (T)10.0);
        max_value = max(fmat1 + (T)10.0);

        CHECK(min_value == (T)10.0);
        CHECK(max_value == (T)(10.0 + 5 * 6 - 1));
    }

    SECTION("FusedTensorND total size, number of dimensions, and shape")
    {
        FusedTensorND<T, 2, 2> tensor;
        FusedTensorND<T, 15, 32> tensor1;

        CHECK(tensor.getTotalSize() == 4);
        CHECK(tensor.getNumDims() == 2);
        CHECK(tensor.getShape() == "(2,2)");

        CHECK(tensor1.getTotalSize() == 480);
        CHECK(tensor1.getNumDims() == 2);
        CHECK(tensor1.getShape() == "(15,32)");
    }

    SECTION("Is FusedTensorND identity")
    {
        // set tensor to identity
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

        // check if ten1.isIdentity() returns true indeed
        CHECK(ten1.isIdentity());

        // change one diagonal element to something other than 1
        ten1(0, 0) = 15;
        CHECK_FALSE(ten1.isIdentity());
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
        T value = (T)13.3;
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
        FusedTensorND<T, 2, 3> tensor1(2), result;
        FusedTensorND<T, 3, 2> tensor2(2);

        // non-inplace transpose
        result = (T)10.0 + tensor1 * tensor2.transpose_view() + tensor2.transpose_view() + (T)10.0;
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
        ten2(1, 2) = 3.0;
        CHECK_FALSE(ten1.transpose_view() == ten2);
    }

    SECTION("Check dimensions mismatch and == , !=, min, max operators")
    {
        // this test should fail when the dimensions of the matrices are not equal
        // and should pass when the dimensions are equal even after transposing one of the matrices
        FusedTensorND<T, 2, 3> tensor1(2);
        FusedTensorND<T, 3, 2> tensor2(2);

        CHECK_THROWS(tensor1 == tensor2);
        CHECK_THROWS(tensor1 != tensor2);
        CHECK_THROWS(min(tensor1, tensor2));
        CHECK_THROWS(max(tensor1, tensor2));

        CHECK_NOTHROW(tensor1 == tensor2.transpose_view());
        CHECK_NOTHROW(min(tensor1, tensor2.transpose_view()));
        CHECK_NOTHROW(max(tensor1, tensor2.transpose_view()));

        CHECK_FALSE(tensor1 != tensor2.transpose_view());
    }

    SECTION("Assign FusedTensorND to another FusedTensorND")
    {
        ten1.setIdentity();
        ten2 = ten1; // this will yield to a deep copy

        CHECK(ten1 == ten2); // they should be equal

        ten1(1, 2) = 3.0; // change one element of the initial tensor

        CHECK_FALSE(ten1 == ten2); // they should not be equal any more
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
        SECTION("addition")
        {
            FusedTensorND<T, 10, 10>
                ten1, ten2, ten3, ten4,
                ten5, ten6, ten7, ten8;

            ten1.setIdentity();
            ten2.setIdentity();

            ten3 = ten1 + ten2;
            ten4 = ten2 + ten1;
            ten5 = ten1 + (T)2.0;
            ten6 = (T)2.0 + ten1;
            ten7 = ten1 + (T)(-2.0);
            ten8 = (T)-2.0 + ten1;

            for (size_t i = 0; i < ten1.getDim(0); ++i)
            {
                for (size_t j = 0; j < ten1.getDim(1); ++j)
                {
                    CHECK(ten3(i, j) == (ten1(i, j) + ten2(i, j)));
                    CHECK(ten4(i, j) == (ten2(i, j) + ten1(i, j)));
                    CHECK(ten5(i, j) == (ten1(i, j) + (T)2.0));
                    CHECK(ten6(i, j) == (ten1(i, j) + (T)2.0));
                    CHECK(ten7(i, j) == (ten1(i, j) + (T)(-2.0)));
                    CHECK(ten8(i, j) == ((T)(-2.0) + ten1(i, j)));
                }
            }
        }

        SECTION("subtraction")
        {
            FusedTensorND<T, 10, 10>
                ten1, ten2, ten3, ten4,
                ten5, ten6, ten7;

            ten1.setIdentity();
            ten2.setIdentity();

            ten3 = ten1 - ten2;
            ten4 = ten2 - ten1;
            ten5 = ten1 - (T)2.0;
            ten6 = (T)2.0 - ten1;

            ten7 = -ten1;

            for (size_t i = 0; i < ten1.getDim(0); ++i)
            {
                for (size_t j = 0; j < ten1.getDim(1); ++j)
                {
                    CHECK(ten3(i, j) == (ten1(i, j) - ten2(i, j)));
                    CHECK(ten4(i, j) == (ten2(i, j) - ten1(i, j)));
                    CHECK(ten5(i, j) == (ten1(i, j) - (T)2.0));
                    CHECK(ten6(i, j) == ((T)2.0 - ten1(i, j)));
                    CHECK(ten7(i, j) == (-ten1(i, j)));
                }
            }
        }

        SECTION("multiplication")
        {
            FusedTensorND<T, 10, 10>
                ten1, ten2, ten3,
                ten4, ten5, ten6;

            ten1.setIdentity();
            ten2.setIdentity();

            ten3 = ten1 * ten2;
            ten4 = ten2 * ten1;
            ten5 = ten1 * (T)2.0;
            ten6 = (T)2.0 * ten1;

            for (size_t i = 0; i < ten1.getDim(0); ++i)
            {
                for (size_t j = 0; j < ten1.getDim(1); ++j)
                {
                    CHECK(ten3(i, j) == (ten1(i, j) * ten2(i, j)));
                    CHECK(ten4(i, j) == (ten2(i, j) * ten1(i, j)));
                    CHECK(ten5(i, j) == (ten1(i, j) * (T)2.0));
                    CHECK(ten6(i, j) == (ten1(i, j) * (T)2.0));
                }
            }
        }

        SECTION("division")
        {
            FusedTensorND<T, 10, 10>
                ten1, ten2, ten3,
                ten4, ten5, ten6;

            ten1.setHomogen(4);
            ten2.setHomogen(8);

            ten3 = ten1 / ten2;
            ten4 = ten2 / ten1;
            ten5 = ten1 / (T)2.0;
            ten6 = (T)2.0 / ten1;

            for (size_t i = 0; i < ten1.getDim(0); ++i)
            {
                for (size_t j = 0; j < ten1.getDim(1); ++j)
                {
                    CHECK(ten3(i, j) == (ten1(i, j) / ten2(i, j)));
                    CHECK(ten4(i, j) == (ten2(i, j) / ten1(i, j)));
                    CHECK(ten5(i, j) == (ten1(i, j) / (T)2.0));
                    CHECK(ten6(i, j) == ((T)2.0 / ten1(i, j)));
                }
            }
        }
    }

    SECTION("FusedTensorND test fused operations")
    {
        FusedTensorND<T, 10, 10>
            ten1, ten2, ten3, ten4,
            ten5, ten6, ten7;

        ten1.setIdentity();
        ten2.setIdentity();

        ten3 = ten1 + ten2 + (T)2.0;
        ten4 = ten1 + ten2 + ten3;
        ten5 = ten1 + ten2 + ten3 + (T)2.0;
        ten6 = ten1 + ten2 + ten3 + ten4 + (T)2.0;
        ten7 = (T)2.0 - (T)1.0 + ten1 + ten2 * (T)3.0 + ten3 + ten4 + ten5 + (T)2.0;

        // check if the result is correct
        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                CHECK(ten3(i, j) == (ten1(i, j) + ten2(i, j) + (T)2.0));
                CHECK(ten4(i, j) == (ten1(i, j) + ten2(i, j) + ten3(i, j)));
                CHECK(ten5(i, j) == (ten1(i, j) + ten2(i, j) + ten3(i, j) + (T)2.0));
                CHECK(ten6(i, j) == (ten1(i, j) + ten2(i, j) + ten3(i, j) + ten4(i, j) + (T)2.0));
                CHECK(ten7(i, j) == ((T)2.0 - (T)1.0 + ten1(i, j) + ten2(i, j) * (T)3.0 + ten3(i, j) + ten4(i, j) + ten5(i, j) + (T)2.0));
            }
        }
    }

    SECTION("FusedTensorND testing dimension after transpose")
    {
        FusedTensorND<T, 2, 3> tensor;

        // Check tensor, it should not be transposed
        CHECK(tensor.getNumDims() == 2);
        CHECK(tensor.getShape() == "(2,3)");
        CHECK(tensor.getDim(0) == 2);
        CHECK(tensor.getDim(1) == 3);

        // Check transposed tensor, it should be transposed
        CHECK(tensor.transpose_view().getNumDims() == 2);
        CHECK(tensor.transpose_view().getShape() == "(3,2)");
        CHECK(tensor.transpose_view().getDim(0) == 3);
        CHECK(tensor.transpose_view().getDim(1) == 2);

        // now lets test a higher order tensor
        FusedTensorND<T, 2, 3, 4> tensor1;
        // size_t order[] = {2, 1, 0};
        // auto transposed1 = tensor1.transpose_view(order);
        // or the following for constexpr view
        auto transposed1 = tensor1.template transpose_view<2, 1, 0>();

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
    }

    SECTION("Check dimensions mismatch on addition, subtraction, multiplication, and division")
    {
        // this test should fail when the dimensions of the matrices are not equal because
        // invoking an operation should throw an exception due to dimensions mismatch.
        // However, it should pass when the dimensions are equal
        // even after transposing one of the matrices

        FusedTensorND<T, 2, 3> tensor1(2);
        FusedTensorND<T, 3, 2> tensor2(2);
        FusedTensorND<T, 3, 2> tensor3;

        CHECK_THROWS(tensor3 = tensor1 + tensor2);
        CHECK_THROWS(tensor3 = tensor1 - tensor2);
        CHECK_THROWS(tensor3 = tensor1 * tensor2);
        CHECK_THROWS(tensor3 = tensor1 / tensor2);

        CHECK_NOTHROW(tensor3 = tensor1.transpose_view() + tensor2);
        CHECK_NOTHROW(tensor3 = tensor1.transpose_view() - tensor2);
        CHECK_NOTHROW(tensor3 = tensor1.transpose_view() * tensor2);
        CHECK_NOTHROW(tensor3 = tensor1.transpose_view() / tensor2);
    }

    SECTION("FusedTensorND transpose")
    {
        FusedTensorND<T, 10, 10> ten1, ten2, ten3, ten4;
        ten1.setRandom(-10, 10);
        ten2 = ten1;

        for (size_t i = 0; i < ten1.getDim(0); ++i)
        {
            for (size_t j = 0; j < ten1.getDim(1); ++j)
            {
                CHECK(ten1.transpose_view()(i, j) == ten2(j, i));
            }
        }

        // now check a long oppeartion by adding a zero matrix to the
        // transpose view (not in place) matrix. The ten1 should not change.
        ten1.setIdentity();
        ten1(0, 1) = (T)10.0;
        ten2.setToZero();

        ten3 = ten1.transpose_view() + ten2;
        ten4 = ten1 + ten2;

        // In both cases, ten2 is a zero matrix (should not change the result)
        // ten3 should not be equal to ten1 because of the transpose
        CHECK(ten3 != ten1);
        CHECK(ten3(1, 0) == (T)10.0);

        // ten4 should be equal to ten1
        CHECK(ten4 == ten1);
    }

    SECTION("Test FusedTensorND einsum operation")
    {
        // This test checks only if the dimension of the result tensor are
        // correct and not the validity of the operation itself.
        // TODO: check the validity too.

        FusedTensorND<T, 2, 3> tensor1(2), tensor2(2);
        FusedTensorND<T, 3, 2> tensor3(2);
        FusedTensorND<T, 2, 2> result2;

        auto result = FusedTensorND<T, 2, 2>::einsum(tensor1, tensor2, 1, 1);

        CHECK(result.getNumDims() == 2);
        CHECK(result.getShape() == "(2,2)");
        CHECK(result.getDim(0) == 2);
        CHECK(result.getDim(1) == 2);

        auto result1 = FusedTensorND<T, 3, 3>::einsum(tensor1, tensor3, 0, 1);

        CHECK(result1.getNumDims() == 2);
        CHECK(result1.getShape() == "(3,3)");
        CHECK(result1.getDim(0) == 3);
        CHECK(result1.getDim(1) == 3);

        CHECK_NOTHROW(result2 =
                          FusedTensorND<T, 2, 2>::einsum(tensor1, tensor2.template transpose_view<1, 0>(), 1, 0));

        CHECK_NOTHROW(result2 =
                          FusedTensorND<T, 2, 2>::einsum(tensor1, tensor2.template transpose_view<0, 1>(), 1, 1));

        // The following two use the non-constexpr version of PermutedView, because the order is runtime.
        // One should prefere the constexpr version (above) for maximum performance if the permutation
        // in known at compile time. So there are the following versions:
        // transpose_view<param pack with permutations>()
        // transpose_view() -> this assumes 2D -> known permutation and hence constexpr view
        // transpose_vie(order array with permutations)
        size_t order[2] = {1, 0};
        CHECK_NOTHROW(result2 =
                          FusedTensorND<T, 2, 2>::einsum(tensor1, tensor2.template transpose_view(order), 1, 0));

        size_t order1[2] = {0, 1};
        CHECK_NOTHROW(result2 =
                          FusedTensorND<T, 2, 2>::einsum(tensor1, tensor2.template transpose_view(order1), 1, 1));
    }
}
