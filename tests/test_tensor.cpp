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
}