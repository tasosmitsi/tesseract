// #include <catch_amalgamated.hpp>
// #include "fused/fused_matrix.h"
// #include "matrix.h"
// #include "utilities.h"

// TEST_CASE("FusedMatrix1", "[fused_bencwfrk]")
// {
//     FusedMatrix<double, 100, 100> fmat1(1), fmat2(2), fmat3(3), fmat4(10), fmat5(10);
//     Matrix<double, 100, 100> mat1(1), mat2(2), mat3(3), mat4(10), mat5(10);

//     SECTION("Long operations benchmark - built in benchmark")
//     {
//         BENCHMARK("FusedMatrix long operations")
//         {
//             fmat4 = fmat1 + fmat2 * fmat3 - fmat1 / fmat2 + fmat3 * fmat4 + fmat1 * fmat2 + fmat3 / fmat4 + fmat1 - fmat2 + fmat3 * fmat4 + fmat1 / fmat2;
//             return fmat4;
//         };
//     }
// }