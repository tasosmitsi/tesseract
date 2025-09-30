#include <catch_amalgamated.hpp>
#include "fused/fused_vector.h"
#include "utilities.h"

TEST_CASE("Test Sparse FusedTensorND", "[sparse]")
{
    FusedTensorND<double, 10, 10> fmat1(7.1), fmat2(8.2), fmat3, fmat4;

    SECTION("General")
    {
        // fmat1(0,0) = 1.0;
        // fmat1(1,1) = 1.0;
        // fmat1 = fmat1 / 0.0;
        // fmat1.print();
        // fmat2.setToZero();
        // fmat3 = fmat1 / fmat2;
        // fmat2.print();
    }
}
