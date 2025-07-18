#include <catch_amalgamated.hpp>
#include "tensor.h"

TEST_CASE("TensorND class", "[tensor]")
{

    SECTION("TensorND testing dimentions after transpose")
    {
        TensorND<double, 2, 3> tensor;
        auto transposed = tensor.transpose();

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
        tensor.transpose(true);
        // Check tensor, it should be transposed
        CHECK(tensor.getNumDims() == 2);
        CHECK(tensor.getShape() == "(3,2)");
        CHECK(tensor.getDim(0) == 3);
        CHECK(tensor.getDim(1) == 2);

        // now lets test a higher order tensor
        TensorND<double, 2, 3, 4> tensor1;
        size_t order[] = {2, 1, 0};

        auto transposed1 = tensor1.transpose(order);

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
        tensor1.transpose(order, true);
        // Check tensor1, it should be transposed
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

        tensor2.transpose(true);
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

        tensor2.transpose(true);
        CHECK_NOTHROW(tensor1 + tensor2);
        CHECK_NOTHROW(tensor1 - tensor2);
        CHECK_NOTHROW(tensor1 * tensor2);
        CHECK_NOTHROW(tensor1 / tensor2);
    }

    SECTION("Test TensorND einsum operation")
    {
        TensorND<double, 2, 3> tensor1(2), tensor2(2);
        TensorND<double, 3, 2> tensor3(2);

        auto result = TensorND<double, 2, 2>::einsum(tensor1, tensor2, 1, 1);

        CHECK(result.getNumDims() == 2);
        CHECK(result.getShape() == "(2,2)");
        CHECK(result.getDim(0) == 2);
        CHECK(result.getDim(1) == 2);

        auto result1 = TensorND<double, 3, 3>::einsum(tensor1, tensor3, 0, 1);

        CHECK(result1.getNumDims() == 2);
        CHECK(result1.getShape() == "(3,3)");
        CHECK(result1.getDim(0) == 3);
        CHECK(result1.getDim(1) == 3);
    }
}
