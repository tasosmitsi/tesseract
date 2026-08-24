#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_vector.h"
#include "fused/operators/reductions.h"

// TODO: the whole-tensor reductions (min, max, sum) are still tested in
// test_fused_tensor.cpp. Move them here so the file matches the header.

// ============================================================================
// KNOWN ANSWER
// ============================================================================

TEMPLATE_TEST_CASE("reductions: min and product of each row",
                   "[reductions]", double, float, int)
{
    using T = TestType;

    T vals[3][4] = {
        {4, 2, 7, 3},
        {1, 9, 5, 8},
        {6, 6, 2, 2}};
    FusedMatrix<T, 3, 4> A(vals);

    auto mins = reduce_min_rows(A);
    CHECK(mins(0) == (T)2);
    CHECK(mins(1) == (T)1);
    CHECK(mins(2) == (T)2);

    auto prods = reduce_prod_rows(A);
    CHECK(prods(0) == (T)168);
    CHECK(prods(1) == (T)360);
    CHECK(prods(2) == (T)144);
}

TEMPLATE_TEST_CASE("reductions: min and product of each column",
                   "[reductions]", double, float, int)
{
    using T = TestType;

    T vals[3][4] = {
        {4, 2, 7, 3},
        {1, 9, 5, 8},
        {6, 6, 2, 2}};
    FusedMatrix<T, 3, 4> A(vals);

    // columns: {4,1,6} {2,9,6} {7,5,2} {3,8,2}
    auto mins = reduce_min_cols(A);
    CHECK(mins(0, 0) == (T)1);
    CHECK(mins(0, 1) == (T)2);
    CHECK(mins(0, 2) == (T)2);
    CHECK(mins(0, 3) == (T)2);

    auto prods = reduce_prod_cols(A);
    CHECK(prods(0, 0) == (T)24);
    CHECK(prods(0, 1) == (T)108);
    CHECK(prods(0, 2) == (T)70);
    CHECK(prods(0, 3) == (T)48);
}

// ============================================================================
// RESULT SHAPE
// ============================================================================

TEST_CASE("reductions: axis reductions return the shape of their fold",
          "[reductions]")
{
    FusedMatrix<double, 3, 4> A;
    A.setSequencial();

    // rows collapse to a column vector, columns to a row vector
    auto by_row = reduce_min_rows(A);
    CHECK(by_row.getDim(0) == 3);
    CHECK(by_row.getDim(1) == 1);

    auto by_col = reduce_min_cols(A);
    CHECK(by_col.getDim(0) == 1);
    CHECK(by_col.getDim(1) == 4);
}

// ============================================================================
// DEGENERATE EXTENTS
// ============================================================================

TEMPLATE_TEST_CASE("reductions: a single column leaves the rows unchanged",
                   "[reductions]", double, float, int)
{
    using T = TestType;

    // N == 1 means an empty fold
    T vals[3][1] = {{4}, {1}, {6}};
    FusedMatrix<T, 3, 1> A(vals);

    auto mins = reduce_min_rows(A);
    CHECK(mins(0) == (T)4);
    CHECK(mins(1) == (T)1);
    CHECK(mins(2) == (T)6);

    auto prods = reduce_prod_rows(A);
    CHECK(prods(0) == (T)4);
    CHECK(prods(1) == (T)1);
    CHECK(prods(2) == (T)6);
}

TEMPLATE_TEST_CASE("reductions: a single row leaves the columns unchanged",
                   "[reductions]", double, float, int)
{
    using T = TestType;

    // M == 1 means an empty fold
    T vals[1][3] = {{4, 1, 6}};
    FusedMatrix<T, 1, 3> A(vals);

    auto mins = reduce_min_cols(A);
    CHECK(mins(0, 0) == (T)4);
    CHECK(mins(0, 1) == (T)1);
    CHECK(mins(0, 2) == (T)6);

    auto prods = reduce_prod_cols(A);
    CHECK(prods(0, 0) == (T)4);
    CHECK(prods(0, 1) == (T)1);
    CHECK(prods(0, 2) == (T)6);
}

TEMPLATE_TEST_CASE("reductions: 1x1 matrix",
                   "[reductions]", double, float, int)
{
    using T = TestType;

    T vals[1][1] = {{7}};
    FusedMatrix<T, 1, 1> A(vals);

    CHECK(reduce_min_rows(A)(0) == (T)7);
    CHECK(reduce_min_cols(A)(0, 0) == (T)7);
    CHECK(reduce_prod_rows(A)(0) == (T)7);
    CHECK(reduce_prod_cols(A)(0, 0) == (T)7);
}

// ============================================================================
// AGREEMENT WITH THE WHOLE-TENSOR REDUCTIONS
// ============================================================================

TEMPLATE_TEST_CASE("reductions: reducing both axes matches the scalar reduction",
                   "[reductions]", double, float, int)
{
    using T = TestType;

    T vals[3][4] = {
        {4, 2, 7, 3},
        {1, 9, 5, 8},
        {6, 6, 2, 2}};
    FusedMatrix<T, 3, 4> A(vals);

    // Per-row minima reduced again must equal the minimum of the whole matrix,
    // by either route
    CHECK(min(reduce_min_rows(A)) == min(A));
    CHECK(min(reduce_min_cols(A)) == min(A));
}

TEMPLATE_TEST_CASE("reductions: row and column reductions agree on a transpose",
                   "[reductions]", double, float, int)
{
    using T = TestType;

    T vals[3][4] = {
        {4, 2, 7, 3},
        {1, 9, 5, 8},
        {6, 6, 2, 2}};
    FusedMatrix<T, 3, 4> A(vals);

    FusedMatrix<T, 4, 3> At;
    At = A.transpose_view();

    // Column i of A is row i of At
    auto by_col = reduce_min_cols(A);
    auto by_row = reduce_min_rows(At);

    for (my_size_t i = 0; i < 4; ++i)
    {
        CHECK(by_col(0, i) == by_row(i));
    }
}

// ============================================================================
// VALUE VARIANTS
// ============================================================================

TEMPLATE_TEST_CASE("reductions: negative and mixed values",
                   "[reductions]", double, float)
{
    using T = TestType;

    T vals[2][3] = {
        {-4.5, 2.0, 0.0},  // min -4.5, product 0
        {3.0, -1.0, 2.0}}; // min -1,   product -6
    FusedMatrix<T, 2, 3> A(vals);

    auto mins = reduce_min_rows(A);
    CHECK(mins(0) == (T)-4.5);
    CHECK(mins(1) == (T)-1.0);

    auto prods = reduce_prod_rows(A);
    CHECK(prods(0) == (T)0.0);
    CHECK(prods(1) == (T)-6.0);
}

TEMPLATE_TEST_CASE("reductions: identity rows do not affect the result",
                   "[reductions]", double, float)
{
    using T = TestType;

    // Rows filled with 1.0 are the t-norm identity for both min and product,
    // which is how unused rules are meant to be padded
    T vals[3][3] = {
        {0.4, 0.7, 0.5},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0}};
    FusedMatrix<T, 3, 3> A(vals);

    auto mins = reduce_min_rows(A);
    CHECK(mins(0) == (T)0.4);
    CHECK(mins(1) == (T)1.0);
    CHECK(mins(2) == (T)1.0);

    // The identity rows leave the column reduction reading the first row alone
    auto col_mins = reduce_min_cols(A);
    CHECK(col_mins(0, 0) == (T)0.4);
    CHECK(col_mins(0, 1) == (T)0.7);
    CHECK(col_mins(0, 2) == (T)0.5);
}

// ============================================================================
// SIZES
// ============================================================================

TEMPLATE_TEST_CASE("reductions: a shape whose width is not a SIMD multiple",
                   "[reductions]", double, float, int)
{
    using T = TestType;

    // 7 columns pad to 8, so the eval loop computes one slot per row with no
    // logical element behind it. This checks padding does not leak in.
    FusedMatrix<T, 3, 7> A;
    A.setSequencial(); // 0..20

    auto mins = reduce_min_rows(A);
    CHECK(mins(0) == (T)0);
    CHECK(mins(1) == (T)7);
    CHECK(mins(2) == (T)14);

    auto col_mins = reduce_min_cols(A);
    for (my_size_t j = 0; j < 7; ++j)
    {
        CHECK(col_mins(0, j) == A(0, j));
    }
}

TEST_CASE("reductions: a larger matrix",
          "[reductions]")
{
    using T = double;

    FusedMatrix<T, 16, 20> A;
    A.setSequencial(); // row i runs from i*20 to i*20+19

    auto mins = reduce_min_rows(A);
    for (my_size_t i = 0; i < 16; ++i)
    {
        CHECK(mins(i) == A(i, 0));
    }

    auto col_mins = reduce_min_cols(A);
    for (my_size_t j = 0; j < 20; ++j)
    {
        CHECK(col_mins(0, j) == A(0, j));
    }
}

// ============================================================================
// LIVE SOURCE DATA
// ============================================================================

TEST_CASE("reductions: the result is a copy, not a view",
          "[reductions]")
{
    using T = double;

    FusedMatrix<T, 3, 3> A;
    A.setSequencial();

    auto mins = reduce_min_rows(A);
    T before = mins(0);

    // The reduction materialises, so changing the source afterwards must not
    // change what was already computed
    A(0, 0) = (T)-99.0;

    CHECK(mins(0) == before);
}
