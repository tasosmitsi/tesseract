#include <catch_amalgamated.hpp>
#include "tensor.h"

TEST_CASE("TensorND class", "[tensor]")
{

    SECTION("TensorND testing dimentions after transpose")
    {
        TensorND<double, 2, 3> tensor;
        tensor.transpose();
        CHECK(tensor.getNumDims() == 2);
        CHECK(tensor.getShape() == "(3,2)");

        CHECK(tensor.getDim(0) == 3);
        CHECK(tensor.getDim(1) == 2);

        // now lets test a higher order tensor
        TensorND<double, 2, 3, 4> tensor1;
        size_t order[] = {2, 1, 0};

        tensor1.transpose(order);

        CHECK(tensor1.getNumDims() == 3);
        CHECK(tensor1.getShape() == "(4,3,2)");

        CHECK(tensor1.getDim(0) == 4);
        CHECK(tensor1.getDim(1) == 3);
        CHECK(tensor1.getDim(2) == 2);
    }

    SECTION("Check dimensions mismatch and == , != operators")
    {
        // this test should fail when the dimensions of the tensors are not equal
        // and should pass when the dimensions are equal even after transposing one of the tensors
        TensorND<double, 2, 3> tensor1(2);
        TensorND<double, 3, 2> tensor2(2);

        CHECK_THROWS(tensor1 == tensor2);
        CHECK_THROWS(tensor1 != tensor2);

        tensor2.transpose();
        CHECK_NOTHROW(tensor1 == tensor2);
        CHECK_FALSE(tensor1 != tensor2);
    }

    SECTION("Check dimensions mismatch on addition, subtraction, multiplication, and division")
    {
        // this test should fail when the dimensions of the tensors are not equal
        // and should pass when the dimensions are equal even after transposing one of the tensors
        TensorND<double, 2, 3> tensor1(2);
        TensorND<double, 3, 2> tensor2(2);

        CHECK_THROWS(tensor1 + tensor2);
        CHECK_THROWS(tensor1 - tensor2);
        CHECK_THROWS(tensor1 * tensor2);
        CHECK_THROWS(tensor1 / tensor2);

        tensor2.transpose();
        CHECK_NOTHROW(tensor1 + tensor2);
        CHECK_NOTHROW(tensor1 - tensor2);
        CHECK_NOTHROW(tensor1 * tensor2);
        CHECK_NOTHROW(tensor1 / tensor2);
    }
}