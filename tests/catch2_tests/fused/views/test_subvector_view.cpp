#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "fused/views/subvector_view.h"

using Catch::Approx;

// ============================================================================
// VIEW SHAPE: DIMS AND SHAPE STRING
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: dims and shape",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Row5 = FusedMatrix<T, 1, 5>;

    SECTION("column slice reports [Len, 1]")
    {
        Vec5 v;
        v.setSequencial();

        auto view = SubVectorView<Vec5, 1, 3>(v);

        REQUIRE(view.getDim(0) == 3);
        REQUIRE(view.getDim(1) == 1);
        REQUIRE(view.getNumDims() == 2);
        REQUIRE(view.getTotalSize() == 3);
        REQUIRE(view.getShape() == "(3,1)");
    }

    SECTION("row slice reports [1, Len]")
    {
        Row5 row;
        row.setSequencial();

        auto view = SubVectorView<Row5, 2, 3>(row);

        REQUIRE(view.getDim(0) == 1);
        REQUIRE(view.getDim(1) == 3);
        REQUIRE(view.getNumDims() == 2);
        REQUIRE(view.getTotalSize() == 3);
        REQUIRE(view.getShape() == "(1,3)");
    }

    SECTION("full view matches source shape")
    {
        Vec5 v;
        v.setSequencial();

        auto view = SubVectorView<Vec5, 0, 5>(v);

        REQUIRE(view.getDim(0) == 5);
        REQUIRE(view.getDim(1) == 1);
        REQUIRE(view.getShape() == "(5,1)");
    }
}

// ============================================================================
// COLUMN VECTOR: HEAD
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: column head",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    Vec3 result;
    result = v.template head<3>();

    REQUIRE(result(0) == T(0));
    REQUIRE(result(1) == T(1));
    REQUIRE(result(2) == T(2));
}

// ============================================================================
// COLUMN VECTOR: TAIL
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: column tail",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    Vec3 result;
    result = v.template tail<3>();

    REQUIRE(result(0) == T(2));
    REQUIRE(result(1) == T(3));
    REQUIRE(result(2) == T(4));
}

// ============================================================================
// COLUMN VECTOR: SEGMENT
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: column segment",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    Vec3 result;
    result = v.template segment<1, 3>();

    REQUIRE(result(0) == T(1));
    REQUIRE(result(1) == T(2));
    REQUIRE(result(2) == T(3));
}

// ============================================================================
// COLUMN VECTOR: FULL VIEW
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: full view equals source",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;

    Vec5 v, result;
    v.setSequencial();

    result = v.template head<5>();

    REQUIRE(result == v);
}

// ============================================================================
// COLUMN VECTOR: SINGLE ELEMENT
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: single element",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec1 = FusedVector<T, 1>;

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    Vec1 result;
    result = v.template segment<3, 1>();

    REQUIRE(result(0) == T(3));
}

// ============================================================================
// COLUMN VECTOR: HEAD AND TAIL PARTITION
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: head and tail partition the vector",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec6 = FusedVector<T, 6>;
    using Vec3 = FusedVector<T, 3>;

    Vec6 v;
    v.setSequencial(); // [0, 1, 2, 3, 4, 5]

    Vec3 first, second;
    first = v.template head<3>();
    second = v.template tail<3>();

    REQUIRE(first(0) == T(0));
    REQUIRE(first(1) == T(1));
    REQUIRE(first(2) == T(2));
    REQUIRE(second(0) == T(3));
    REQUIRE(second(1) == T(4));
    REQUIRE(second(2) == T(5));
}

// ============================================================================
// COLUMN VECTOR: ADJACENT SEGMENTS
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: adjacent segments cover full vector",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec6 = FusedVector<T, 6>;
    using Vec2 = FusedVector<T, 2>;

    Vec6 v;
    v.setSequencial(); // [0, 1, 2, 3, 4, 5]

    Vec2 s0, s1, s2;
    s0 = v.template segment<0, 2>();
    s1 = v.template segment<2, 2>();
    s2 = v.template segment<4, 2>();

    REQUIRE(s0(0) == T(0));
    REQUIRE(s0(1) == T(1));
    REQUIRE(s1(0) == T(2));
    REQUIRE(s1(1) == T(3));
    REQUIRE(s2(0) == T(4));
    REQUIRE(s2(1) == T(5));
}

// ============================================================================
// ROW VECTOR: MATERIALIZED
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: row vector slice",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Row5 = FusedMatrix<T, 1, 5>;
    using Row3 = FusedMatrix<T, 1, 3>;

    Row5 row;
    row.setSequencial(); // [0, 1, 2, 3, 4]

    Row3 result;
    result = SubVectorView<Row5, 1, 3>(row);

    REQUIRE(result(0, 0) == T(1));
    REQUIRE(result(0, 1) == T(2));
    REQUIRE(result(0, 2) == T(3));
}

TEMPLATE_TEST_CASE("subvector_view: row vector head",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Row5 = FusedMatrix<T, 1, 5>;
    using Row3 = FusedMatrix<T, 1, 3>;

    Row5 row;
    row.setSequencial(); // [0, 1, 2, 3, 4]
    row = row * T(10);   // [0, 10, 20, 30, 40]

    Row3 result;
    result = SubVectorView<Row5, 0, 3>(row);

    REQUIRE(result(0, 0) == T(0));
    REQUIRE(result(0, 1) == T(10));
    REQUIRE(result(0, 2) == T(20));
}

TEMPLATE_TEST_CASE("subvector_view: row vector tail",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Row5 = FusedMatrix<T, 1, 5>;
    using Row2 = FusedMatrix<T, 1, 2>;

    Row5 row;
    row.setSequencial(); // [0, 1, 2, 3, 4]

    Row2 result;
    result = SubVectorView<Row5, 3, 2>(row);

    REQUIRE(result(0, 0) == T(3));
    REQUIRE(result(0, 1) == T(4));
}

// ============================================================================
// TRANSPOSED COLUMN → ROW SLICE
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: transposed column vector slice uses gather",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Row3 = FusedMatrix<T, 1, 3>;

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    auto tv = v.transpose_view(); // [1, 5]

    Row3 result;
    result = SubVectorView<decltype(tv), 1, 3>(tv);

    REQUIRE(result(0, 0) == T(1));
    REQUIRE(result(0, 1) == T(2));
    REQUIRE(result(0, 2) == T(3));
}

// ============================================================================
// MATERIALIZED TRANSPOSE → CONTIGUOUS ROW SLICE
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: materialized row slice uses direct load",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Row5 = FusedMatrix<T, 1, 5>;
    using Row3 = FusedMatrix<T, 1, 3>;

    Vec5 v;
    v.setSequencial();

    Row5 row;
    row = v.transpose_view(); // materialize

    Row3 result;
    result = SubVectorView<Row5, 2, 3>(row);

    REQUIRE(result(0, 0) == T(2));
    REQUIRE(result(0, 1) == T(3));
    REQUIRE(result(0, 2) == T(4));
}

// ============================================================================
// EXPRESSION TEMPLATE: VIEW + VIEW
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: view + view",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 a, b;
    a.setSequencial(); // [0, 1, 2, 3, 4]
    b.setSequencial(); // [0, 1, 2, 3, 4]

    Vec3 result;
    result = SubVectorView<Vec5, 1, 3>(a) + SubVectorView<Vec5, 0, 3>(b);

    // [1,2,3] + [0,1,2] = [1,3,5]
    REQUIRE(result(0) == T(1));
    REQUIRE(result(1) == T(3));
    REQUIRE(result(2) == T(5));
}

// ============================================================================
// EXPRESSION TEMPLATE: VIEW + VIEW + VIEW
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: view + view + view",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 a, b;
    a.setSequencial(); // [0, 1, 2, 3, 4]
    b.setSequencial(); // [0, 1, 2, 3, 4]

    Vec3 result;
    result = SubVectorView<Vec5, 1, 3>(a) + SubVectorView<Vec5, 0, 3>(b) + SubVectorView<Vec5, 2, 3>(b);

    // [1,2,3] + [0,1,2] + [2,3,4] = [3,6,9]
    REQUIRE(result(0) == T(3));
    REQUIRE(result(1) == T(6));
    REQUIRE(result(2) == T(9));
}

// ============================================================================
// EXPRESSION TEMPLATE: VIEW + VIEW USING HEAD/TAIL
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: head + tail expression",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec6 = FusedVector<T, 6>;
    using Vec3 = FusedVector<T, 3>;

    Vec6 v;
    v.setSequencial(); // [0, 1, 2, 3, 4, 5]

    Vec3 result;
    result = v.template head<3>() + v.template tail<3>();

    // [0,1,2] + [3,4,5] = [3,5,7]
    REQUIRE(result(0) == T(3));
    REQUIRE(result(1) == T(5));
    REQUIRE(result(2) == T(7));
}

// ============================================================================
// EXPRESSION TEMPLATE: VIEW * SCALAR
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: view * scalar",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    Vec3 result;
    result = v.template segment<1, 3>() * T(2);

    // [1,2,3] * 2 = [2,4,6]
    REQUIRE(result(0) == T(2));
    REQUIRE(result(1) == T(4));
    REQUIRE(result(2) == T(6));
}

// ============================================================================
// EQUALITY: FULL VIEW EQUALS SOURCE
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: equality with full view",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;

    Vec5 a, b;
    a.setSequencial();
    b.setSequencial();

    REQUIRE(a == SubVectorView<Vec5, 0, 5>(b));
}

// ============================================================================
// EQUALITY: MATCHING SEGMENTS
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: matching segments are equal",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;

    Vec5 a, b;
    a.setSequencial();
    b.setSequencial();

    // Same data, same offset
    REQUIRE(SubVectorView<Vec5, 1, 3>(a) == SubVectorView<Vec5, 1, 3>(b));
}

// ============================================================================
// 1×1 EDGE CASE
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: 1x1 vector",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec1 = FusedVector<T, 1>;

    Vec1 v;
    v(0) = T(42);

    Vec1 result;
    result = v.template head<1>();

    REQUIRE(result(0) == T(42));
}

// ============================================================================
// LARGE VECTOR: 7 ELEMENTS
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: 7-element vector segments",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec7 = FusedVector<T, 7>;
    using Vec4 = FusedVector<T, 4>;

    Vec7 v;
    v.setSequencial(); // [0, 1, 2, 3, 4, 5, 6]

    Vec4 result;
    result = v.template segment<3, 4>();

    REQUIRE(result(0) == T(3));
    REQUIRE(result(1) == T(4));
    REQUIRE(result(2) == T(5));
    REQUIRE(result(3) == T(6));
}

// ============================================================================
// NON-SEQUENTIAL VALUES
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: non-sequential values",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 v(T(0));
    v(0) = T(10);
    v(1) = T(-5);
    v(2) = T(3.5);
    v(3) = T(0);
    v(4) = T(99);

    Vec3 result;
    result = v.template segment<1, 3>();

    REQUIRE(result(0) == T(-5));
    REQUIRE(result(1) == T(3.5));
    REQUIRE(result(2) == T(0));
}

// ============================================================================
// VIEW DOES NOT COPY & DOES NOT MATERIALIZE: READS SOURCE DATA
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: reads live source data",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;

    Vec5 v;
    v.setSequencial(); // [0, 1, 2, 3, 4]

    auto view = SubVectorView<Vec5, 1, 3>(v); // it does not call assignment operator.

    REQUIRE(view(0) == T(1));
    REQUIRE(view(1) == T(2));
    REQUIRE(view(2) == T(3));
}

// ============================================================================
// ROW VECTOR: FULL VIEW
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: row full view",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Row5 = FusedMatrix<T, 1, 5>;

    Row5 row, result;
    row.setSequencial(); // [0, 1, 2, 3, 4]

    result = SubVectorView<Row5, 0, 5>(row);

    REQUIRE(result == row);
}

// ============================================================================
// EXPRESSION: VIEW - VIEW
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: view - view",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 a, b;
    a.setSequencial(); // [0, 1, 2, 3, 4]
    b.setSequencial(); // [0, 1, 2, 3, 4]

    Vec3 result;
    result = SubVectorView<Vec5, 2, 3>(a) - SubVectorView<Vec5, 0, 3>(b);

    // [2,3,4] - [0,1,2] = [2,2,2]
    REQUIRE(result(0) == T(2));
    REQUIRE(result(1) == T(2));
    REQUIRE(result(2) == T(2));
}

// ============================================================================
// DIFFERENT SOURCE SIZES: VIEW INTO DIFFERENT VECTORS
// ============================================================================

TEMPLATE_TEST_CASE("subvector_view: different source sizes same result",
                   "[subvector_view]", double, float, int)
{
    using T = TestType;
    using Vec5 = FusedVector<T, 5>;
    using Vec8 = FusedVector<T, 8>;
    using Vec3 = FusedVector<T, 3>;

    Vec5 a;
    a.setSequencial(); // [0, 1, 2, 3, 4]

    Vec8 b;
    b.setSequencial(); // [0, 1, 2, 3, 4, 5, 6, 7]

    Vec3 ra, rb;
    ra = SubVectorView<Vec5, 1, 3>(a);
    rb = SubVectorView<Vec8, 1, 3>(b);

    // Both should give [1, 2, 3]
    REQUIRE(ra == rb);
}
