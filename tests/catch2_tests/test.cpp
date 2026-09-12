// #include <catch_amalgamated.hpp>
// #include <iostream>
// #include "config.h"

// #include "fused/fused_vector.h"
// #include "fused/views/subvector_view.h"

// TEST_CASE("Generic tests", "[test]")
// {
//     FusedVector<double, 3> vec;
//     FusedMatrix<double, 1, 3> mat1;
//     FusedVector<double, 5> vec2, vec3, vec4;
//     FusedMatrix<double, 1, 5> mat;

//     vec.setSequencial();
//     vec2.setSequencial();
//     vec3.setSequencial();
//     vec4.setSequencial();

//     vec2.print(true);
//     vec3.print(true);

//     vec = SubVectorView<FusedVector<double, 5>, 1, 3>(vec2) + SubVectorView<FusedVector<double, 5>, 0, 3>(vec3) + SubVectorView<FusedVector<double, 5>, 2, 3>(vec3);
//     vec.print(true);

//     std::cout << (vec2 == SubVectorView<FusedVector<double, 5>, 0, 5>(vec3)) << std::endl;




//     FusedVector<double, 5> v;
//     v.setSequencial(); // [0, 1, 2, 3, 4]
//     v.print(true);

//     // Column vector → subvector (Axis=0, gather)
//     FusedVector<double, 3> col_slice;
//     col_slice = SubVectorView<FusedVector<double, 5>, 1, 3>(v);
//     col_slice.print(true);
//     // → [1, 2, 3]

//     // Transpose → row vector [1, 5] → subvector (Axis=1, contiguous load)
//     auto tv = v.transpose_view(); // PermutedViewConstExpr, Dim=[1, 5]
//     FusedMatrix<double, 1, 3> row_slice;
//     row_slice.printLayoutInfo();
//     row_slice = SubVectorView<decltype(tv), 1, 3>(tv);
//     row_slice.print(true);
//     // → [1, 2, 3]

//     FusedMatrix<double, 1, 5> row;
//     row = v.transpose_view(); // materializes — elements now contiguous
//     row.print();

//     row_slice = SubVectorView<decltype(row), 2, 3>(row);
//     row_slice.print(true);
//     row_slice.printLayoutInfo();
//     // SliceStride = stride(1) = 1 → direct load

//     vec2.print();
//     std::cout << SubVectorView<FusedVector<double, 5>, 1, 3>(vec2)(0) << std::endl; // should print 1
//     std::cout << SubVectorView<FusedVector<double, 5>, 2, 3>(vec2)(2) << std::endl; // should print 4
// }
