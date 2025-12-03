#include <catch_amalgamated.hpp>

#include "fused/fused_vector.h"

TEMPLATE_TEST_CASE("FusedVector class", "[fused_vector]", double, float)
{
    using T = TestType;

    FusedVector<T, 5> vec1(1.1), vec2(2.0), vec3;
    FusedMatrix<T, 1, 5> mat1(10);
    FusedMatrix<T, 5, 5> mat2(10);

    SECTION("FusedVector elements access")
    {
        vec1(0) = (T)3.14;
        CHECK(vec1(0) == (T)3.14);
    }

    SECTION("FusedVector total size, number of dimensions, and shape")
    {
        FusedVector<T, 2> vector;
        FusedVector<T, 15> vector1;

        CHECK(vector.getTotalSize() == 2);
        CHECK(vector.getNumDims() == 2);
        CHECK(vector.getShape() == "(2,1)");

        CHECK(vector1.getTotalSize() == 15);
        CHECK(vector1.getNumDims() == 2);
        CHECK(vector1.getShape() == "(15,1)");
        CHECK(vector1.transposed().getShape() == "(1,15)");
    }

    SECTION("Is FusedVector identity")
    {
        FusedVector<T, 2> vector(0);
        CHECK_FALSE(vec1.isIdentity());
    }

    SECTION("FusedVector einsum/matmul with FusedMatrix")
    {
        // This test checks only if the dimension of the result tensor are
        // correct and not the validity of the operation itself.
        // TODO: check the validity too.
        auto einsum_res = FusedTensorND<T, 5, 5>::einsum(vec1, mat1, 1, 0);
        CHECK(einsum_res.getShape() == "(5,5)");

        auto matmul_res1 = FusedMatrix<T, 5, 5>::matmul(vec1, mat1);
        CHECK(matmul_res1.getShape() == "(5,5)");
    }
}
