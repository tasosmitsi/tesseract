#include <catch_amalgamated.hpp>

#include "interpolation/axis.h"
#include "interpolation/lookup_table.h"

using Catch::Approx;
using namespace interpolation;

// ============================================================================
// 1D LINEAR
// ============================================================================

TEMPLATE_TEST_CASE("table1d: interpolates between two breakpoints",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = UniformAxis<T, 4>;

    Axis axis((T)100, (T)50); // 100, 150, 200, 250

    FusedTensorND<T, 4> values;
    values(0) = (T)10;
    values(1) = (T)20;
    values(2) = (T)40;
    values(3) = (T)80;

    Table1D<T, Axis> table(axis, values);

    // 175 is halfway between 150 and 200, so halfway between 20 and 40
    CHECK(table.at((T)175) == Approx((T)30));

    // A quarter of the way from 200 to 250
    CHECK(table.at((T)212.5) == Approx((T)50));
}

TEMPLATE_TEST_CASE("table1d: exact breakpoints return their stored value",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = UniformAxis<T, 4>;

    Axis axis((T)100, (T)50);

    FusedTensorND<T, 4> values;
    values(0) = (T)10;
    values(1) = (T)20;
    values(2) = (T)40;
    values(3) = (T)80;

    Table1D<T, Axis> table(axis, values);

    CHECK(table.at((T)100) == Approx((T)10));
    CHECK(table.at((T)150) == Approx((T)20));
    CHECK(table.at((T)200) == Approx((T)40));
    CHECK(table.at((T)250) == Approx((T)80));
}

TEMPLATE_TEST_CASE("table1d: clamps outside the domain",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = UniformAxis<T, 4>;

    Axis axis((T)100, (T)50);

    FusedTensorND<T, 4> values;
    values(0) = (T)10;
    values(1) = (T)20;
    values(2) = (T)40;
    values(3) = (T)80;

    Table1D<T, Axis> table(axis, values);

    // Holds the boundary value rather than extrapolating
    CHECK(table.at((T)0) == Approx((T)10));
    CHECK(table.at((T)9000) == Approx((T)80));
}

TEMPLATE_TEST_CASE("table1d: a constant table returns that constant",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = UniformAxis<T, 4>;

    Axis axis((T)100, (T)50);

    FusedTensorND<T, 4> values;
    for (my_size_t i = 0; i < 4; ++i)
        values(i) = (T)7;

    Table1D<T, Axis> table(axis, values);

    // Whatever the weights are, they must sum to one
    CHECK(table.at((T)137) == Approx((T)7));
    CHECK(table.at((T)212) == Approx((T)7));
}

TEMPLATE_TEST_CASE("table1d: a linear table reproduces the line",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = UniformAxis<T, 5>;

    // Breakpoints 0..4, values 3x + 1
    Axis axis((T)0, (T)1);

    FusedTensorND<T, 5> values;
    for (my_size_t i = 0; i < 5; ++i)
        values(i) = (T)3 * (T)i + (T)1;

    Table1D<T, Axis> table(axis, values);

    // Linear interpolation of a linear function is exact everywhere
    for (T q = (T)0; q <= (T)4; q += (T)0.3)
    {
        CHECK(table.at(q) == Approx((T)3 * q + (T)1));
    }
}

TEMPLATE_TEST_CASE("table1d: works with a non-uniform axis",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = NonUniformAxis<T, 4>;

    Array<T, 4> bp{(T)0, (T)1, (T)10, (T)100};
    Axis axis(bp);

    FusedTensorND<T, 4> values;
    values(0) = (T)0;
    values(1) = (T)10;
    values(2) = (T)20;
    values(3) = (T)30;

    Table1D<T, Axis> table(axis, values);

    // Halfway across the wide last cell
    CHECK(table.at((T)55) == Approx((T)25));

    // Halfway across the narrow first cell
    CHECK(table.at((T)0.5) == Approx((T)5));
}

TEMPLATE_TEST_CASE("table1d: deduces its type from the axis and the values",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = UniformAxis<T, 4>;

    Axis axis((T)0, (T)1);

    FusedTensorND<T, 4> values;
    values.setSequencial();

    Table1D table(axis, values);

    static_assert(is_same<decltype(table), Table1D<T, Axis>>::value,
                  "Table1D CTAD deduced the wrong type");

    const T expected = (T)0.5 * (values(1) + values(2));
    CHECK(table.at((T)1.5) == Approx(expected));
}

// ============================================================================
// 2D BILINEAR
// ============================================================================

TEMPLATE_TEST_CASE("table2d: interpolates across both axes",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 2>;
    using A1 = UniformAxis<T, 2>;

    A0 a0((T)0, (T)1);
    A1 a1((T)0, (T)1);

    // A unit cell with a known value at each corner
    FusedTensorND<T, 2, 2> values;
    values(0, 0) = (T)0;
    values(1, 0) = (T)10;
    values(0, 1) = (T)20;
    values(1, 1) = (T)30;

    Table2D<T, A0, A1> table(a0, a1, values);

    // Centre is the mean of the four corners
    CHECK(table.at((T)0.5, (T)0.5) == Approx((T)15));

    // Midpoint of each edge is the mean of that edge's two corners
    CHECK(table.at((T)0.5, (T)0) == Approx((T)5));
    CHECK(table.at((T)0.5, (T)1) == Approx((T)25));
    CHECK(table.at((T)0, (T)0.5) == Approx((T)10));
    CHECK(table.at((T)1, (T)0.5) == Approx((T)20));
}

TEMPLATE_TEST_CASE("table2d: corners return their stored value",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);

    FusedTensorND<T, 2, 2> values;
    values(0, 0) = (T)0;
    values(1, 0) = (T)10;
    values(0, 1) = (T)20;
    values(1, 1) = (T)30;

    Table2D<T, A, A> table(a0, a1, values);

    CHECK(table.at((T)0, (T)0) == Approx((T)0));
    CHECK(table.at((T)1, (T)0) == Approx((T)10));
    CHECK(table.at((T)0, (T)1) == Approx((T)20));
    CHECK(table.at((T)1, (T)1) == Approx((T)30));
}

TEMPLATE_TEST_CASE("table2d: a bilinear function is reproduced exactly",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 3>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);

    // f(x, y) = 2x + 3y + 4xy + 1, which bilinear interpolation is exact for
    FusedTensorND<T, 3, 3> values;
    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            const T x = (T)i;
            const T y = (T)j;
            values(i, j) = (T)2 * x + (T)3 * y + (T)4 * x * y + (T)1;
        }
    }

    Table2D<T, A, A> table(a0, a1, values);

    for (T x = (T)0; x <= (T)2; x += (T)0.4)
    {
        for (T y = (T)0; y <= (T)2; y += (T)0.4)
        {
            const T expected = (T)2 * x + (T)3 * y + (T)4 * x * y + (T)1;
            CHECK(table.at(x, y) == Approx(expected).margin((T)1e-4));
        }
    }
}

TEMPLATE_TEST_CASE("table2d: clamps on both axes independently",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);

    FusedTensorND<T, 2, 2> values;
    values(0, 0) = (T)0;
    values(1, 0) = (T)10;
    values(0, 1) = (T)20;
    values(1, 1) = (T)30;

    Table2D<T, A, A> table(a0, a1, values);

    // Off the edge on one axis, mid-range on the other
    CHECK(table.at((T)-5, (T)0.5) == Approx((T)10));
    CHECK(table.at((T)5, (T)0.5) == Approx((T)20));

    // Off both
    CHECK(table.at((T)-5, (T)-5) == Approx((T)0));
    CHECK(table.at((T)5, (T)5) == Approx((T)30));
}

TEMPLATE_TEST_CASE("table2d: mixed axis types",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 2>;
    using A1 = NonUniformAxis<T, 3>;

    A0 a0((T)0, (T)1);

    Array<T, 3> bp{(T)0, (T)1, (T)100};
    A1 a1(bp);

    FusedTensorND<T, 2, 3> values;
    values(0, 0) = (T)0;
    values(1, 0) = (T)0;
    values(0, 1) = (T)10;
    values(1, 1) = (T)10;
    values(0, 2) = (T)20;
    values(1, 2) = (T)20;

    Table2D<T, A0, A1> table(a0, a1, values);

    // The first axis makes no difference here; the second is halfway across
    // its wide upper cell
    CHECK(table.at((T)0.3, (T)50.5) == Approx((T)15));
}

TEMPLATE_TEST_CASE("table2d: deduces its type from the axes and the values",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 2>;
    using A1 = NonUniformAxis<T, 3>;

    A0 a0((T)0, (T)1);

    Array<T, 3> bp{(T)0, (T)1, (T)5};
    A1 a1(bp);

    FusedTensorND<T, 2, 3> values;
    values.setSequencial();

    Table2D table(a0, a1, values);

    // Mixed axis types, so the assert shows each slot deduced on its own
    static_assert(is_same<decltype(table), Table2D<T, A0, A1>>::value,
                  "Table2D CTAD deduced the wrong type");

    const T expected = (T)0.5 * (values(0, 1) + values(1, 1));
    CHECK(table.at((T)0.5, (T)1) == Approx(expected));
}

// ============================================================================
// 3D TRILINEAR
// ============================================================================

TEMPLATE_TEST_CASE("table3d: the centre is the mean of eight corners",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2> values;
    T v = (T)1;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                values(i, j, k) = v++;

    Table3D<T, A, A, A> table(a0, a1, a2, values);

    // Corner values 1..8, so the mean is 4.5
    CHECK(table.at((T)0.5, (T)0.5, (T)0.5) == Approx((T)4.5));
}

TEMPLATE_TEST_CASE("table3d: corners return their stored value",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2> values;
    T v = (T)1;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                values(i, j, k) = v++;

    Table3D<T, A, A, A> table(a0, a1, a2, values);

    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                CHECK(table.at((T)i, (T)j, (T)k) == Approx(values(i, j, k)));
}

TEMPLATE_TEST_CASE("table3d: varies along one axis only",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);

    // Depends on the third axis alone
    FusedTensorND<T, 2, 2, 2> values;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
        {
            values(i, j, 0) = (T)0;
            values(i, j, 1) = (T)100;
        }

    Table3D<T, A, A, A> table(a0, a1, a2, values);

    // The first two queries must not matter
    CHECK(table.at((T)0.2, (T)0.7, (T)0.3) == Approx((T)30));
    CHECK(table.at((T)0.9, (T)0.1, (T)0.3) == Approx((T)30));
}

TEMPLATE_TEST_CASE("table3d: a constant table returns that constant",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2> values;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                values(i, j, k) = (T)7;

    Table3D<T, A, A, A> table(a0, a1, a2, values);

    // The eight weights must sum to one
    CHECK(table.at((T)0.13, (T)0.62, (T)0.87) == Approx((T)7));
}

TEMPLATE_TEST_CASE("table3d: deduces its type from the axes and the values",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 2>;
    using A1 = NonUniformAxis<T, 2>;
    using A2 = UniformAxis<T, 2>;

    A0 a0((T)0, (T)1);

    Array<T, 2> bp{(T)0, (T)1};
    A1 a1(bp);

    A2 a2((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2> values;
    values.setSequencial();

    Table3D table(a0, a1, a2, values);

    static_assert(is_same<decltype(table), Table3D<T, A0, A1, A2>>::value,
                  "Table3D CTAD deduced the wrong type");

    T sum = (T)0;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                sum += values(i, j, k);

    CHECK(table.at((T)0.5, (T)0.5, (T)0.5) == Approx(sum / (T)8));
}

// ============================================================================
// CURSOR
// ============================================================================

TEMPLATE_TEST_CASE("table1d: a carried cursor gives the same answers",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = NonUniformAxis<T, 5>;

    Array<T, 5> bp{(T)100, (T)120, (T)130, (T)140, (T)200};
    Axis axis(bp);

    FusedTensorND<T, 5> values;
    for (my_size_t i = 0; i < 5; ++i)
        values(i) = (T)i * (T)10;

    Table1D<T, Axis> table(axis, values);

    Cursor<1> cursor;

    const T queries[] = {(T)105, (T)125, (T)135, (T)190, (T)102, (T)145};

    for (T q : queries)
    {
        CHECK(table.at(q, cursor) == Approx(table.at(q)));
    }
}

TEMPLATE_TEST_CASE("table2d: a carried cursor gives the same answers",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = NonUniformAxis<T, 4>;
    using A1 = NonUniformAxis<T, 3>;

    Array<T, 4> bp0{(T)0, (T)10, (T)50, (T)100};
    Array<T, 3> bp1{(T)0, (T)1, (T)5};

    A0 a0(bp0);
    A1 a1(bp1);

    FusedTensorND<T, 4, 3> values;
    for (my_size_t i = 0; i < 4; ++i)
        for (my_size_t j = 0; j < 3; ++j)
            values(i, j) = (T)(i * 3 + j);

    Table2D<T, A0, A1> table(a0, a1, values);

    Cursor<2> cursor;

    for (T x = (T)0; x <= (T)100; x += (T)13)
    {
        for (T y = (T)0; y <= (T)5; y += (T)0.7)
        {
            CHECK(table.at(x, y, cursor) == Approx(table.at(x, y)));
        }
    }
}

// ============================================================================
// SHARED AXES
// ============================================================================

TEMPLATE_TEST_CASE("table: several tables share one axis",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = UniformAxis<T, 3>;

    // The case the axis-outside-the-table design exists for
    Axis axis((T)0, (T)1);

    FusedTensorND<T, 3> gains;
    gains(0) = (T)1;
    gains(1) = (T)2;
    gains(2) = (T)3;

    FusedTensorND<T, 3> limits;
    limits(0) = (T)10;
    limits(1) = (T)20;
    limits(2) = (T)30;

    Table1D<T, Axis> gain_table(axis, gains);
    Table1D<T, Axis> limit_table(axis, limits);

    CHECK(gain_table.at((T)0.5) == Approx((T)1.5));
    CHECK(limit_table.at((T)0.5) == Approx((T)15));
}

// ============================================================================
// TableND: AGREEMENT WITH THE EXPLICIT RANKS
// ============================================================================

TEMPLATE_TEST_CASE("tablend: rank 1 agrees with table1d",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using Axis = UniformAxis<T, 4>;

    Axis axis((T)100, (T)50);

    FusedTensorND<T, 4> values;
    values(0) = (T)10;
    values(1) = (T)20;
    values(2) = (T)40;
    values(3) = (T)80;

    Table1D<T, Axis> ref(axis, values);
    TableND<T, Axis> nd(axis, values);

    for (T q = (T)100; q <= (T)250; q += (T)7)
    {
        CHECK(nd.at({q}) == Approx(ref.at(q)).margin((T)1e-4));
    }
}

TEMPLATE_TEST_CASE("tablend: rank 2 agrees with table2d",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 3>;
    using A1 = UniformAxis<T, 3>;

    A0 a0((T)0, (T)1);
    A1 a1((T)0, (T)1);

    FusedTensorND<T, 3, 3> values;
    for (my_size_t i = 0; i < 3; ++i)
        for (my_size_t j = 0; j < 3; ++j)
            values(i, j) = (T)(i * 7 + j * 3 + 1);

    Table2D<T, A0, A1> ref(a0, a1, values);
    TableND<T, A0, A1> nd(a0, a1, values);

    for (T x = (T)0; x <= (T)2; x += (T)0.35)
    {
        for (T y = (T)0; y <= (T)2; y += (T)0.35)
        {
            CHECK(nd.at({x, y}) == Approx(ref.at(x, y)).margin((T)1e-4));
        }
    }
}

TEMPLATE_TEST_CASE("tablend: rank 3 agrees with table3d",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2> values;
    T v = (T)1;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                values(i, j, k) = v++;

    Table3D<T, A, A, A> ref(a0, a1, a2, values);
    TableND<T, A, A, A> nd(a0, a1, a2, values);

    for (T x = (T)0; x <= (T)1; x += (T)0.3)
        for (T y = (T)0; y <= (T)1; y += (T)0.3)
            for (T z = (T)0; z <= (T)1; z += (T)0.3)
                CHECK(nd.at({x, y, z}) == Approx(ref.at(x, y, z)).margin((T)1e-4));
}

// ============================================================================
// TableND: RANK 4
// ============================================================================

TEMPLATE_TEST_CASE("tablend: the centre is the mean of sixteen corners",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);
    A a3((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2, 2> values;
    T v = (T)1;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                for (my_size_t l = 0; l < 2; ++l)
                    values(i, j, k, l) = v++;

    TableND<T, A, A, A, A> table(a0, a1, a2, a3, values);

    // Corner values 1..16, so the mean is 8.5
    CHECK(table.at({(T)0.5, (T)0.5, (T)0.5, (T)0.5}) == Approx((T)8.5));
}

TEMPLATE_TEST_CASE("tablend: each axis indexes its own dimension",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 2>;
    using A1 = UniformAxis<T, 3>;
    using A2 = UniformAxis<T, 4>;
    using A3 = UniformAxis<T, 5>;

    A0 a0((T)0, (T)1);
    A1 a1((T)0, (T)1);
    A2 a2((T)0, (T)1);
    A3 a3((T)0, (T)1);

    // Every index is readable off the stored value, so a swapped axis or a
    // corner bit assigned to the wrong dimension changes the answer
    FusedTensorND<T, 2, 3, 4, 5> values;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 3; ++j)
            for (my_size_t k = 0; k < 4; ++k)
                for (my_size_t l = 0; l < 5; ++l)
                    values(i, j, k, l) = (T)(i * 1000 + j * 100 + k * 10 + l);

    TableND<T, A0, A1, A2, A3> table(a0, a1, a2, a3, values);

    CHECK(table.at({(T)1, (T)2, (T)3, (T)4}) == Approx((T)1234));
    CHECK(table.at({(T)0, (T)1, (T)2, (T)3}) == Approx((T)123));

    // Halfway along the first axis, on grid everywhere else
    CHECK(table.at({(T)0.5, (T)2, (T)3, (T)4}) == Approx((T)734));

    // Halfway along the third axis instead
    CHECK(table.at({(T)1, (T)2, (T)2.5, (T)4}) == Approx((T)1229));
}

// ============================================================================
// TableND: EDGE CASES
// ============================================================================

TEMPLATE_TEST_CASE("tablend: grid points return their stored value",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 3>;
    using A1 = UniformAxis<T, 2>;
    using A2 = UniformAxis<T, 3>;

    A0 a0((T)0, (T)1);
    A1 a1((T)0, (T)1);
    A2 a2((T)0, (T)1);

    FusedTensorND<T, 3, 2, 3> values;
    T v = (T)1;
    for (my_size_t i = 0; i < 3; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 3; ++k)
                values(i, j, k) = v++;

    TableND<T, A0, A1, A2> table(a0, a1, a2, values);

    for (my_size_t i = 0; i < 3; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 3; ++k)
                CHECK(table.at({(T)i, (T)j, (T)k}) == Approx(values(i, j, k)));
}

TEMPLATE_TEST_CASE("tablend: clamps outside the domain on every axis",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2> values;
    T v = (T)1;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                values(i, j, k) = v++;

    TableND<T, A, A, A> table(a0, a1, a2, values);

    // Below every breakpoint holds the low corner, above every breakpoint the high
    CHECK(table.at({(T)-9000, (T)-9000, (T)-9000}) == Approx(values(0, 0, 0)));
    CHECK(table.at({(T)9000, (T)9000, (T)9000}) == Approx(values(1, 1, 1)));

    // Off the edge on two axes, mid-cell on the third
    const T expected = (T)0.5 * (values(1, 0, 0) + values(1, 0, 1));
    CHECK(table.at({(T)9000, (T)-9000, (T)0.5}) == Approx(expected));
}

// ============================================================================
// TableND: PROPERTIES
// ============================================================================

TEMPLATE_TEST_CASE("tablend: a constant table returns that constant",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);
    A a3((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2, 2> values;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                for (my_size_t l = 0; l < 2; ++l)
                    values(i, j, k, l) = (T)7;

    TableND<T, A, A, A, A> table(a0, a1, a2, a3, values);

    // The sixteen weights must sum to one
    CHECK(table.at({(T)0.13, (T)0.62, (T)0.87, (T)0.41}) == Approx((T)7));
    CHECK(table.at({(T)0.99, (T)0.01, (T)0.5, (T)0}) == Approx((T)7));
}

TEMPLATE_TEST_CASE("tablend: axes the values ignore do not affect the result",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);
    A a2((T)0, (T)1);
    A a3((T)0, (T)1);

    // Depends on the second axis alone
    FusedTensorND<T, 2, 2, 2, 2> values;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t k = 0; k < 2; ++k)
            for (my_size_t l = 0; l < 2; ++l)
            {
                values(i, 0, k, l) = (T)0;
                values(i, 1, k, l) = (T)100;
            }

    TableND<T, A, A, A, A> table(a0, a1, a2, a3, values);

    CHECK(table.at({(T)0.2, (T)0.3, (T)0.7, (T)0.9}) == Approx((T)30));
    CHECK(table.at({(T)0.9, (T)0.3, (T)0.1, (T)0.4}) == Approx((T)30));
}

TEMPLATE_TEST_CASE("tablend: a multilinear function is reproduced exactly",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 3>;
    using A1 = UniformAxis<T, 3>;
    using A2 = UniformAxis<T, 2>;
    using A3 = UniformAxis<T, 2>;

    A0 a0((T)0, (T)1);
    A1 a1((T)0, (T)1);
    A2 a2((T)0, (T)1);
    A3 a3((T)0, (T)1);

    // Degree at most one in each variable, which is what multilinear
    // interpolation reproduces
    auto f = [](T x0, T x1, T x2, T x3)
    {
        return (T)1 + (T)2 * x0 + (T)3 * x1 + (T)4 * x2 + (T)5 * x3 +
               x0 * x1 + x2 * x3 + x0 * x1 * x2 * x3;
    };

    FusedTensorND<T, 3, 3, 2, 2> values;
    for (my_size_t i = 0; i < 3; ++i)
        for (my_size_t j = 0; j < 3; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                for (my_size_t l = 0; l < 2; ++l)
                    values(i, j, k, l) = f((T)i, (T)j, (T)k, (T)l);

    TableND<T, A0, A1, A2, A3> table(a0, a1, a2, a3, values);

    for (T x0 = (T)0; x0 <= (T)2; x0 += (T)0.6)
        for (T x1 = (T)0; x1 <= (T)2; x1 += (T)0.6)
            for (T x2 = (T)0; x2 <= (T)1; x2 += (T)0.5)
                for (T x3 = (T)0; x3 <= (T)1; x3 += (T)0.5)
                    CHECK(table.at({x0, x1, x2, x3}) ==
                          Approx(f(x0, x1, x2, x3)).margin((T)1e-3));
}

TEMPLATE_TEST_CASE("tablend: a carried cursor gives the same answers",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = NonUniformAxis<T, 5>;
    using A1 = NonUniformAxis<T, 3>;
    using A2 = UniformAxis<T, 4>;
    using A3 = NonUniformAxis<T, 3>;

    Array<T, 5> bp0{(T)100, (T)120, (T)130, (T)140, (T)200};
    Array<T, 3> bp1{(T)0, (T)1, (T)5};
    Array<T, 3> bp3{(T)-10, (T)0, (T)2};

    A0 a0(bp0);
    A1 a1(bp1);
    A2 a2((T)0, (T)0.25);
    A3 a3(bp3);

    FusedTensorND<T, 5, 3, 4, 3> values;
    for (my_size_t i = 0; i < 5; ++i)
        for (my_size_t j = 0; j < 3; ++j)
            for (my_size_t k = 0; k < 4; ++k)
                for (my_size_t l = 0; l < 3; ++l)
                    values(i, j, k, l) = (T)(i * 37 + j * 11 + k * 5 + l + 1);

    TableND<T, A0, A1, A2, A3> table(a0, a1, a2, a3, values);

    Cursor<4> cursor;

    // A slowly moving query, then a jump backwards, which is the case the
    // cursor's neighbour check has to fall out of
    const T q0[] = {(T)105, (T)125, (T)135, (T)190, (T)102, (T)145};
    const T q1[] = {(T)0.2, (T)0.9, (T)2.0, (T)4.5, (T)0.1, (T)3.0};
    const T q3[] = {(T)-9, (T)-2, (T)0.5, (T)1.9, (T)-8, (T)1.0};

    for (my_size_t n = 0; n < 6; ++n)
    {
        const T q2 = (T)0.1 * (T)n;

        CHECK(table.at({q0[n], q1[n], q2, q3[n]}, cursor) ==
              Approx(table.at({q0[n], q1[n], q2, q3[n]})));
    }
}

// ============================================================================
// TableND: CONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("tablend: both constructor orders build the same table",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A = UniformAxis<T, 2>;

    A a0((T)0, (T)1);
    A a1((T)0, (T)1);

    FusedTensorND<T, 2, 2> values;
    values(0, 0) = (T)0;
    values(1, 0) = (T)10;
    values(0, 1) = (T)20;
    values(1, 1) = (T)30;

    TableND<T, A, A> axes_first(a0, a1, values);
    TableND<T, A, A> values_first(values, a0, a1);

    CHECK(axes_first.at({(T)0.3, (T)0.7}) ==
          Approx(values_first.at({(T)0.3, (T)0.7})));
}

TEMPLATE_TEST_CASE("tablend: deduces its type from the values and the axes",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 2>;
    using A1 = UniformAxis<T, 3>;

    A0 a0((T)0, (T)1);
    A1 a1((T)0, (T)1);

    FusedTensorND<T, 2, 3> values;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 3; ++j)
            values(i, j) = (T)(i * 3 + j);

    // Values first, which is the order the deduction guide needs
    TableND table(values, a0, a1);

    static_assert(is_same<decltype(table), TableND<T, A0, A1>>::value,
                  "TableND CTAD deduced the wrong type");

    CHECK(table.at({(T)0.5, (T)1}) == Approx((T)2.5));
}

TEMPLATE_TEST_CASE("tablend: exposes its axes by index",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 2>;
    using A1 = UniformAxis<T, 3>;

    A0 a0((T)0, (T)1);
    A1 a1((T)0, (T)1);

    FusedTensorND<T, 2, 3> values;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 3; ++j)
            values(i, j) = (T)0;

    TableND<T, A0, A1> table(a0, a1, values);

    // The axes are held, not copied
    CHECK(&table.template axis<0>() == &a0);
    CHECK(&table.template axis<1>() == &a1);
}

// ============================================================================
// TableND: GENERIC PATH
// ============================================================================

TEMPLATE_TEST_CASE("tablend: rank 5 over mixed axis types",
                   "[lookup_table]", double, float)
{
    using T = TestType;
    using A0 = UniformAxis<T, 2>;
    using A1 = NonUniformAxis<T, 2>;
    using A2 = UniformAxis<T, 2>;
    using A3 = NonUniformAxis<T, 2>;
    using A4 = UniformAxis<T, 2>;

    Array<T, 2> bp1{(T)0, (T)1};
    Array<T, 2> bp3{(T)0, (T)1};

    A0 a0((T)0, (T)1);
    A1 a1(bp1);
    A2 a2((T)0, (T)1);
    A3 a3(bp3);
    A4 a4((T)0, (T)1);

    FusedTensorND<T, 2, 2, 2, 2, 2> values;
    T v = (T)1;
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 2; ++j)
            for (my_size_t k = 0; k < 2; ++k)
                for (my_size_t l = 0; l < 2; ++l)
                    for (my_size_t m = 0; m < 2; ++m)
                        values(i, j, k, l, m) = v++;

    TableND<T, A0, A1, A2, A3, A4> table(a0, a1, a2, a3, a4, values);

    // Thirty-two corners holding 1..32, so the centre is 16.5
    CHECK(table.at({(T)0.5, (T)0.5, (T)0.5, (T)0.5, (T)0.5}) == Approx((T)16.5));

    // Grid points still come back exactly
    CHECK(table.at({(T)0, (T)0, (T)0, (T)0, (T)0}) == Approx((T)1));
    CHECK(table.at({(T)1, (T)1, (T)1, (T)1, (T)1}) == Approx((T)32));
}

// ============================================================================
// TableND: REJECTED INSTANTIATIONS
// ============================================================================

TEST_CASE("tablend: rejected instantiations", "[lookup_table]")
{
    using A = UniformAxis<double, 2>;

    // Uncomment one at a time. (void)Table::Rank forces the instantiation;
    // an unused alias never fires the assert.

    // Rank 0. Expect "needs at least one axis", followed by a
    // cascade from FusedTensorND<double> having no dimensions.
    // using Table = TableND<double>;
    // (void)Table::Rank;

    // Rank 9 means 512 corners. Expect "rank above 8 unrolls 2^Rank corners".
    // using Table = TableND<double, A, A, A, A, A, A, A, A, A>;
    // (void)Table::Rank;

    // axis<K> out of range.
    // A a(0.0, 1.0);
    // FusedTensorND<double, 2> v;
    // TableND<double, A> t(a, v);
    // (void)t.axis<1>();

    (void)sizeof(A);
}
