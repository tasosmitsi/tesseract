// #include <catch_amalgamated.hpp>
// #include "utilities.h"
// #include "fused/fused_matrix.h"
// #include "matrix.h"

// TEST_CASE("FusedMatrix1", "[lol]")
// {
//     SECTION("LOL")
//     {
//         FusedTensorND<double, 3, 2> fmat1(1);
//         FusedTensorND<double, 3, 2> fmat2(2);
//         FusedTensorND<double, 2, 3> test(10);

//         fmat1.setRandom(1, 5);
//         fmat1.print();
//         // fmat1.print();
//         // fmat2.print();
//         // std::cout << fmat1.transpose_view().getShape() << std::endl;

//         // fmat2 = fmat1 + test;
//         // std ::cout << demangleTypeName(typeid(lol)) << std::endl;

//         if (fmat1(0, 0) >= 2.5)
//         {
//             size_t order[2] = {1, 0};
//             std::cout << fmat2.transpose_view(order).getShape() << std::endl;
//             // fmat2.transpose_view<0,2>();
//             auto fmat3 = FusedTensorND<double, 3, 3>::einsum(fmat1, fmat2.transpose_view<1, 0>(), 1, 0);
//             fmat3.print();
//         }
//         else
//         {
//             size_t order[2] = {0, 1};
//             std::cout << fmat2.transpose_view(order).getShape() << std::endl;
//             auto fmat3 = FusedTensorND<double, 3, 3>::einsum(fmat1, fmat2.transpose_view<0, 1>(), 1, 1);
//             fmat3.print();
//         }

//         // auto fmat3 = FusedTensorND<double, 2, 2>::einsum(fmat1.transpose_view() + 2.0 + test, fmat2, 1, 0);

//         // fmat3.print();
//     }
// }