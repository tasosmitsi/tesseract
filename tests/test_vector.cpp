#include <catch_amalgamated.hpp>

#include "vector.h"

TEST_CASE("Vector class", "[vector]")
{
    Vector<double, 5> vec1(1.0), vec2(2.0);

    SECTION("Vector accessing elements")
    {
        vec1(0) = 3.14;
        CHECK(vec1(0) == 3.14);
    }

    SECTION("Vector total size, number of dimensions, and shape")
    {
        CHECK(vec1.getTotalSize() == 5);
        CHECK(vec1.getNumDims() == 1);
        CHECK(vec1.getShape() == "(5)");

        vec2.print();


    }
}