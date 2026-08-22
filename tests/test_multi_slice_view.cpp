#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "fused/fused_tensor.h"
#include "fused/views/multi_slice_view.h"

using Catch::Approx;

// ============================================================================
// SHAPE AND CONSTANTS
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: shape of a row slice",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = MultiSliceView<Matrix, Slice<0, 2, 1>>(A);

    REQUIRE(view.getDim(0) == 1);
    REQUIRE(view.getDim(1) == 6);
    REQUIRE(view.getNumDims() == 2);
    REQUIRE(view.getTotalSize() == 6);
    REQUIRE(view.getShape() == "(1,6)");
}

TEMPLATE_TEST_CASE("multi_slice_view: shape of a column slice",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = MultiSliceView<Matrix, Slice<1, 3, 1>>(A);

    REQUIRE(view.getDim(0) == 4);
    REQUIRE(view.getDim(1) == 1);
    REQUIRE(view.getTotalSize() == 4);
    REQUIRE(view.getShape() == "(4,1)");
}

TEMPLATE_TEST_CASE("multi_slice_view: shape of a block slice",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = MultiSliceView<Matrix, Slice<0, 1, 2>, Slice<1, 2, 3>>(A);

    REQUIRE(view.getDim(0) == 2);
    REQUIRE(view.getDim(1) == 3);
    REQUIRE(view.getTotalSize() == 6);
    REQUIRE(view.getShape() == "(2,3)");
}

TEMPLATE_TEST_CASE("multi_slice_view: unsliced axes keep their extent",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = MultiSliceView<Matrix, Slice<0, 0, 4>>(A);

    // Slicing the full extent of axis 0 leaves the shape unchanged
    REQUIRE(view.getDim(0) == 4);
    REQUIRE(view.getDim(1) == 6);
    REQUIRE(view.getTotalSize() == 24);
}

// ============================================================================
// ELEMENT ACCESS 2D
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: row slice reads the right row",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial(); // A(i,j) = i*6 + j

    auto view = MultiSliceView<Matrix, Slice<0, 2, 1>>(A);

    // Row 2 is [12, 13, 14, 15, 16, 17]
    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(view(0, j) == A(2, j));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: column slice reads the right column",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = MultiSliceView<Matrix, Slice<1, 3, 1>>(A);

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(view(i, 0) == A(i, 3));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: block slice reads the right block",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    // Rows 1..2, columns 2..4
    auto view = MultiSliceView<Matrix, Slice<0, 1, 2>, Slice<1, 2, 3>>(A);

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(view(i, j) == A(1 + i, 2 + j));
        }
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: offset zero reads from the start",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = MultiSliceView<Matrix, Slice<0, 0, 1>>(A);

    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(view(0, j) == A(0, j));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: last valid offset",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = MultiSliceView<Matrix, Slice<0, 3, 1>>(A);

    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(view(0, j) == A(3, j));
    }
}

// ============================================================================
// ELEMENT ACCESS IN VECTOR-LIKE SOURCES
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: subvector of a column vector",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    auto view = MultiSliceView<Vec5, Slice<0, 1, 3>>(v);

    REQUIRE(view.getShape() == "(3,1)");
    REQUIRE(view(0, 0) == T(1));
    REQUIRE(view(1, 0) == T(2));
    REQUIRE(view(2, 0) == T(3));
}

TEMPLATE_TEST_CASE("multi_slice_view: subvector of a row vector",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Row5 = FusedMatrix<T, 1, 5>;

    Row5 row;
    row.setSequencial(); // [0, 1, 2, 3, 4]

    auto view = MultiSliceView<Row5, Slice<1, 2, 3>>(row);

    REQUIRE(view.getShape() == "(1,3)");
    REQUIRE(view(0, 0) == T(2));
    REQUIRE(view(0, 1) == T(3));
    REQUIRE(view(0, 2) == T(4));
}

TEMPLATE_TEST_CASE("multi_slice_view: 1x1 source",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 1, 1>;

    Matrix A;
    A(0, 0) = T(42);

    auto view = MultiSliceView<Matrix, Slice<0, 0, 1>>(A);

    REQUIRE(view.getShape() == "(1,1)");
    REQUIRE(view(0, 0) == T(42));
}

// ============================================================================
// ELEMENT ACCESS 3D AND 4D
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: 3D plane pinned on the outer axis",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 3, 4>;

    Tensor A;
    A.setSequencial();

    auto view = MultiSliceView<Tensor, Slice<0, 1, 1>>(A);

    REQUIRE(view.getShape() == "(1,3,4)");

    for (my_size_t j = 0; j < 3; ++j)
    {
        for (my_size_t k = 0; k < 4; ++k)
        {
            REQUIRE(view(0, j, k) == A(1, j, k));
        }
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: 3D plane pinned on the middle axis",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 3, 4>;

    Tensor A;
    A.setSequencial();

    auto view = MultiSliceView<Tensor, Slice<1, 2, 1>>(A);

    REQUIRE(view.getShape() == "(2,1,4)");

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t k = 0; k < 4; ++k)
        {
            REQUIRE(view(i, 0, k) == A(i, 2, k));
        }
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: 3D two axes pinned gives a vector",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 3, 4>;

    Tensor A;
    A.setSequencial();

    // Pin axis 0 and axis 2, leaving a vector along axis 1
    auto view = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<2, 3, 1>>(A);

    REQUIRE(view.getShape() == "(1,3,1)");

    for (my_size_t j = 0; j < 3; ++j)
    {
        REQUIRE(view(0, j, 0) == A(1, j, 3));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: 3D sub-block",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 3, 4>;

    Tensor A;
    A.setSequencial();

    auto view = MultiSliceView<Tensor, Slice<1, 1, 2>, Slice<2, 1, 2>>(A);

    REQUIRE(view.getShape() == "(2,2,2)");

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 2; ++j)
        {
            for (my_size_t k = 0; k < 2; ++k)
            {
                REQUIRE(view(i, j, k) == A(i, 1 + j, 1 + k));
            }
        }
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: 4D source",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 3, 4, 5>;

    Tensor A;
    A.setSequencial();

    auto view = MultiSliceView<Tensor, Slice<1, 1, 1>, Slice<3, 0, 2>>(A);

    REQUIRE(view.getShape() == "(2,1,4,2)");

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t k = 0; k < 4; ++k)
        {
            for (my_size_t l = 0; l < 2; ++l)
            {
                REQUIRE(view(i, 0, k, l) == A(i, 1, k, l));
            }
        }
    }
}

// ============================================================================
// CONTIGUITY
// ============================================================================

TEST_CASE("multi_slice_view: row slice is contiguous",
          "[multi_slice_view]")
{
    using Matrix = FusedMatrix<double, 4, 6>;
    using View = MultiSliceView<Matrix, Slice<0, 2, 1>>;

    REQUIRE(View::IsPhysicallyContiguous);
}

TEST_CASE("multi_slice_view: column slice is not contiguous",
          "[multi_slice_view]")
{
    using Matrix = FusedMatrix<double, 4, 6>;
    using View = MultiSliceView<Matrix, Slice<1, 3, 1>>;

    REQUIRE_FALSE(View::IsPhysicallyContiguous);
}

TEST_CASE("multi_slice_view: subvector of a column vector is not contiguous",
          "[multi_slice_view]")
{
    // Vector elements sit one per padded row
    using Vec5 = FusedVector<double, 5>;
    using View = MultiSliceView<Vec5, Slice<0, 1, 3>>;

    REQUIRE_FALSE(View::IsPhysicallyContiguous);
}

TEST_CASE("multi_slice_view: 3D outer-axis plane is contiguous",
          "[multi_slice_view]")
{
    using Tensor = FusedTensorND<double, 2, 3, 4>;
    using View = MultiSliceView<Tensor, Slice<0, 1, 1>>;

    REQUIRE(View::IsPhysicallyContiguous);
}

TEST_CASE("multi_slice_view: 3D middle-axis plane is not contiguous",
          "[multi_slice_view]")
{
    using Tensor = FusedTensorND<double, 2, 3, 4>;
    using View = MultiSliceView<Tensor, Slice<1, 2, 1>>;

    REQUIRE_FALSE(View::IsPhysicallyContiguous);
}

// ============================================================================
// PHYSICAL BASE
// ============================================================================

TEST_CASE("multi_slice_view: physical base of a row slice",
          "[multi_slice_view]")
{
    // [4,6] double, SimdWidth=4 → PhysicalDims [4,8], stride(0) = 8
    using Matrix = FusedMatrix<double, 4, 6>;

    REQUIRE(MultiSliceView<Matrix, Slice<0, 0, 1>>::PhysicalBase == 0);
    REQUIRE(MultiSliceView<Matrix, Slice<0, 1, 1>>::PhysicalBase == 8);
    REQUIRE(MultiSliceView<Matrix, Slice<0, 2, 1>>::PhysicalBase == 16);
}

TEST_CASE("multi_slice_view: physical base of a column slice",
          "[multi_slice_view]")
{
    using Matrix = FusedMatrix<double, 4, 6>;

    REQUIRE(MultiSliceView<Matrix, Slice<1, 0, 1>>::PhysicalBase == 0);
    REQUIRE(MultiSliceView<Matrix, Slice<1, 3, 1>>::PhysicalBase == 3);
}

TEST_CASE("multi_slice_view: physical base sums over both slices",
          "[multi_slice_view]")
{
    using Matrix = FusedMatrix<double, 4, 6>;

    // 1*8 + 2*1 = 10
    REQUIRE(MultiSliceView<Matrix, Slice<0, 1, 2>, Slice<1, 2, 3>>::PhysicalBase == 10);
}

// ============================================================================
// ASSIGNMENT / MATERIALIZING A SLICE
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: assign a row slice",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Row6 = FusedMatrix<T, 1, 6>;

    Matrix A;
    A.setSequencial();

    Row6 result;
    result = MultiSliceView<Matrix, Slice<0, 2, 1>>(A);

    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(result(0, j) == A(2, j));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: assign a column slice",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Col4 = FusedMatrix<T, 4, 1>;

    Matrix A;
    A.setSequencial();

    Col4 result;
    result = MultiSliceView<Matrix, Slice<1, 3, 1>>(A);

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(result(i, 0) == A(i, 3));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: assign a block slice",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Block = FusedMatrix<T, 2, 3>;

    Matrix A;
    A.setSequencial();

    Block result;
    result = MultiSliceView<Matrix, Slice<0, 1, 2>, Slice<1, 2, 3>>(A);

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(result(i, j) == A(1 + i, 2 + j));
        }
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: assign a 3D plane",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 3, 4>;
    using Plane = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial();

    Plane result;
    result = MultiSliceView<Tensor, Slice<0, 1, 1>>(A);

    for (my_size_t j = 0; j < 3; ++j)
    {
        for (my_size_t k = 0; k < 4; ++k)
        {
            REQUIRE(result(0, j, k) == A(1, j, k));
        }
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: full-extent slice equals the source",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A, result;
    A.setSequencial();

    result = MultiSliceView<Matrix, Slice<0, 0, 4>>(A);

    REQUIRE(result == A);
}

// ============================================================================
// EXPRESSIONS
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: slice plus slice",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Row6 = FusedMatrix<T, 1, 6>;

    Matrix A;
    A.setSequencial();

    Row6 result;
    result = MultiSliceView<Matrix, Slice<0, 0, 1>>(A) +
             MultiSliceView<Matrix, Slice<0, 1, 1>>(A);

    // Row 0 is [0..5], row 1 is [6..11]
    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(result(0, j) == A(0, j) + A(1, j));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: slice minus itself is zero",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Col4 = FusedMatrix<T, 4, 1>;

    Matrix A;
    A.setSequencial();

    Col4 result;
    result = MultiSliceView<Matrix, Slice<1, 2, 1>>(A) -
             MultiSliceView<Matrix, Slice<1, 2, 1>>(A);

    Col4 zero(T(0));
    REQUIRE(result == zero);
}

TEMPLATE_TEST_CASE("multi_slice_view: slice times scalar",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Row6 = FusedMatrix<T, 1, 6>;

    Matrix A;
    A.setSequencial();

    Row6 result;
    result = MultiSliceView<Matrix, Slice<0, 2, 1>>(A) * T(3);

    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(result(0, j) == A(2, j) * T(3));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: row plus column of the same matrix",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Square = FusedMatrix<T, 4, 4>;
    using Col4 = FusedMatrix<T, 4, 1>;

    Square A;
    A.setSequencial();

    // A column slice and another column slice, one contiguous read pattern
    // and one strided, in the same expression
    Col4 result;
    result = MultiSliceView<Square, Slice<1, 0, 1>>(A) +
             MultiSliceView<Square, Slice<1, 3, 1>>(A);

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(result(i, 0) == A(i, 0) + A(i, 3));
    }
}

// ============================================================================
// EQUALITY
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: equal slices of equal sources",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A, B;
    A.setSequencial();
    B.setSequencial();

    REQUIRE(MultiSliceView<Matrix, Slice<0, 2, 1>>(A) ==
            MultiSliceView<Matrix, Slice<0, 2, 1>>(B));
}

TEMPLATE_TEST_CASE("multi_slice_view: slice equals an explicit tensor",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Row6 = FusedMatrix<T, 1, 6>;

    Matrix A;
    A.setSequencial();

    Row6 expected;
    for (my_size_t j = 0; j < 6; ++j)
        expected(0, j) = T(12 + j); // row 2

    REQUIRE(expected == MultiSliceView<Matrix, Slice<0, 2, 1>>(A));
}

// ============================================================================
// LIVE SOURCE DATA
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: reads live source data",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A(T(0));

    auto view = MultiSliceView<Matrix, Slice<0, 2, 1>>(A);

    // Modify the source after the view exists
    A(2, 3) = T(7);

    REQUIRE(view(0, 3) == T(7));
}

// ============================================================================
// PERMUTED SOURCES
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: row of a transposed matrix",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto tv = A.transpose_view(); // presents [6,4]

    // Row 0 of the transposed view is column 0 of the original
    auto view = MultiSliceView<decltype(tv), Slice<0, 0, 1>>(tv);

    REQUIRE(view.getShape() == "(1,4)");

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(view(0, i) == A(i, 0));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: column of a transposed matrix",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto tv = A.transpose_view(); // presents [6,4]

    // Column 2 of the transposed view is row 2 of the original
    auto view = MultiSliceView<decltype(tv), Slice<1, 2, 1>>(tv);

    REQUIRE(view.getShape() == "(6,1)");

    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(view(j, 0) == A(2, j));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: block of a transposed matrix",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto tv = A.transpose_view(); // presents [6,4]

    auto view = MultiSliceView<decltype(tv), Slice<0, 1, 3>, Slice<1, 1, 2>>(tv);

    REQUIRE(view.getShape() == "(3,2)");

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 2; ++j)
        {
            // view(i,j) is tv(1+i, 1+j) which is A(1+j, 1+i)
            REQUIRE(view(i, j) == A(1 + j, 1 + i));
        }
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: assign a slice of a transposed matrix",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Row4 = FusedMatrix<T, 1, 4>;

    Matrix A;
    A.setSequencial();

    auto tv = A.transpose_view();

    Row4 result;
    result = MultiSliceView<decltype(tv), Slice<0, 0, 1>>(tv);

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(result(0, i) == A(i, 0));
    }
}

TEST_CASE("multi_slice_view: transposed slice agrees with the direct one",
          "[multi_slice_view]")
{
    using Matrix = FusedMatrix<double, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto tv = A.transpose_view();

    // Column 2 of the transposed view and row 2 of the original name the same
    // six elements, in the same order
    auto viaTranspose = MultiSliceView<decltype(tv), Slice<1, 2, 1>>(tv);
    auto direct = MultiSliceView<Matrix, Slice<0, 2, 1>>(A);

    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(viaTranspose(j, 0) == direct(0, j));
    }
}

// ============================================================================
// PADDING BOUNDARIES
// ============================================================================

TEST_CASE("multi_slice_view: slice spanning the padding boundary",
          "[multi_slice_view]")
{
    // [4,6] double, SimdWidth=4 → pad(6)=8. Columns 4..5 are the last real
    // ones; 6..7 are padding. A slice ending at 6 must not read into it.
    using Matrix = FusedMatrix<double, 4, 6>;
    using Block = FusedMatrix<double, 4, 3>;

    Matrix A;
    A.setSequencial();

    Block result;
    result = MultiSliceView<Matrix, Slice<1, 3, 3>>(A);

    for (my_size_t i = 0; i < 4; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(result(i, j) == A(i, 3 + j));
        }
    }
}

TEST_CASE("multi_slice_view: last column of a padded matrix",
          "[multi_slice_view]")
{
    using Matrix = FusedMatrix<double, 4, 6>;
    using Col4 = FusedMatrix<double, 4, 1>;

    Matrix A;
    A.setSequencial();

    Col4 result;
    result = MultiSliceView<Matrix, Slice<1, 5, 1>>(A);

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(result(i, 0) == A(i, 5));
    }
}

// ============================================================================
// FLOAT SOURCE: DIFFERENT SIMD WIDTH
// ============================================================================

TEST_CASE("multi_slice_view: float source",
          "[multi_slice_view]")
{
    // float pads to a different width than double
    using Matrix = FusedMatrix<float, 4, 6>;
    using Row6 = FusedMatrix<float, 1, 6>;

    Matrix A;
    A.setSequencial();

    Row6 result;
    result = MultiSliceView<Matrix, Slice<0, 2, 1>>(A);

    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(result(0, j) == A(2, j));
    }
}

// ============================================================================
// FusedMatrix CONVENIENCE METHODS
// ============================================================================

TEMPLATE_TEST_CASE("multi_slice_view: FusedMatrix::row",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = A.template row<2>();

    REQUIRE(view.getShape() == "(1,6)");

    for (my_size_t j = 0; j < 6; ++j)
    {
        REQUIRE(view(0, j) == A(2, j));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: FusedMatrix::column",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = A.template column<3>();

    REQUIRE(view.getShape() == "(4,1)");

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(view(i, 0) == A(i, 3));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: FusedMatrix::block",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    auto view = A.template block<1, 2, 2, 3>();

    REQUIRE(view.getShape() == "(2,3)");

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            REQUIRE(view(i, j) == A(1 + i, 2 + j));
        }
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: convenience methods match explicit slices",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;

    Matrix A;
    A.setSequencial();

    REQUIRE(A.template row<2>() == MultiSliceView<Matrix, Slice<0, 2, 1>>(A));
    REQUIRE(A.template column<3>() == MultiSliceView<Matrix, Slice<1, 3, 1>>(A));
    REQUIRE((A.template block<1, 2, 2, 3>()) ==
            (MultiSliceView<Matrix, Slice<0, 1, 2>, Slice<1, 2, 3>>(A)));
}

TEMPLATE_TEST_CASE("multi_slice_view: assign from convenience methods",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Matrix = FusedMatrix<T, 4, 6>;
    using Row6 = FusedMatrix<T, 1, 6>;
    using Col4 = FusedMatrix<T, 4, 1>;

    Matrix A;
    A.setSequencial();

    Row6 r;
    r = A.template row<0>();

    Col4 c;
    c = A.template column<5>();

    for (my_size_t j = 0; j < 6; ++j)
        REQUIRE(r(0, j) == A(0, j));

    for (my_size_t i = 0; i < 4; ++i)
        REQUIRE(c(i, 0) == A(i, 5));
}

TEMPLATE_TEST_CASE("multi_slice_view: adding two columns",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Square = FusedMatrix<T, 4, 4>;
    using Col4 = FusedMatrix<T, 4, 1>;

    Square A;
    A.setSequencial();

    Col4 result;
    result = A.template column<0>() + A.template column<3>();

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(result(i, 0) == A(i, 0) + A(i, 3));
    }
}

TEMPLATE_TEST_CASE("multi_slice_view: adding a column and a transposed row",
                   "[multi_slice_view]", double, float, int)
{
    using T = TestType;
    using Square = FusedMatrix<T, 4, 4>;
    using Col4 = FusedMatrix<T, 4, 1>;

    Square A;
    A.setSequencial();

    auto tv = A.transpose_view();

    // Column 1 of the transposed view is row 1 of A, shaped [4,1] so it adds
    // to a column of A directly
    Col4 result;
    result = A.template column<0>() +
             MultiSliceView<decltype(tv), Slice<1, 1, 1>>(tv);

    for (my_size_t i = 0; i < 4; ++i)
    {
        REQUIRE(result(i, 0) == A(i, 0) + A(1, i));
    }
}

TEST_CASE("multi_slice_view: first and last row and column",
          "[multi_slice_view]")
{
    using Matrix = FusedMatrix<double, 4, 6>;

    Matrix A;
    A.setSequencial();

    // the last column also sits next to the padding
    REQUIRE(A.row<0>()(0, 0) == A(0, 0));
    REQUIRE(A.row<3>()(0, 5) == A(3, 5));
    REQUIRE(A.column<0>()(0, 0) == A(0, 0));
    REQUIRE(A.column<5>()(3, 0) == A(3, 5));
}

// ============================================================================
// NEGATIVE CASES: COMPILE-TIME ERRORS
// ============================================================================
// These are static_asserts. Uncomment one at a time; each must fail the build.
// Reading a member is required, an unused alias never instantiates.

TEST_CASE("multi_slice_view: negative cases are compile-time errors",
          "[multi_slice_view]")
{
    using Matrix = FusedMatrix<double, 4, 6>;
    (void)sizeof(Matrix);

    // using View = MultiSliceView<Matrix, Slice<9, 0, 1>>;                  // axis out of range
    // using View = MultiSliceView<Matrix, Slice<0, 0, 0>>;                  // Len == 0
    // using View = MultiSliceView<Matrix, Slice<0, 2, 3>>;                  // Offset+Len > extent
    // using View = MultiSliceView<Matrix, Slice<0, 0, 1>, Slice<0, 1, 1>>;  // duplicate axis
    // using View = MultiSliceView<Matrix>;                                  // empty pack

    // using Inner = MultiSliceView<Matrix, Slice<0, 0, 2>>;
    // using View = MultiSliceView<Inner, Slice<1, 0, 2>>; // nesting

    // (void)View::NumDims;
}
