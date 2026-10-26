#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/fused_tensor.h"
#include "fused/fused_vector.h"
#include "fused/views/multi_slice_view.h"
#include "fused/views/permuted_view_constexpr.h"
#include "fused/views/squeeze_view.h"

// The contiguity cases assume a SIMD build, like the other view tests: a last
// axis of 5 doubles is padded (to 8 under AVX2, 6 under SSE2), one of 4 is not.

namespace
{
    template <typename V>
    concept CopyConstructible = requires(const V &v) { V(v); };

    template <typename V>
    concept MoveConstructible = requires(V &&v) { V(static_cast<V &&>(v)); };

    template <typename V>
    concept CopyAssignable = requires(V &lhs, const V &rhs) { lhs = rhs; };
} // namespace

// ============================================================================
// SHAPE AND CONSTANTS
// ============================================================================

TEMPLATE_TEST_CASE("squeeze_view: leading axis dropped",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0>(A);

    REQUIRE(view.getNumDims() == 2);
    REQUIRE(view.getDim(0) == 3);
    REQUIRE(view.getDim(1) == 4);
    REQUIRE(view.getTotalSize() == 12);
    REQUIRE(view.getShape() == "(3,4)");
}

TEMPLATE_TEST_CASE("squeeze_view: middle axis dropped",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 1, 4>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 1>(A);

    REQUIRE(view.getNumDims() == 2);
    REQUIRE(view.getShape() == "(3,4)");
}

TEMPLATE_TEST_CASE("squeeze_view: two pinned axes of a 4D tensor",
                   "[squeeze_view]", double, float, int)
{
    // The case the view exists for: pinning axes 0 and 1 of [3,5,2,4] gives a
    // [1,1,2,4] slice, and the squeeze gives the [2,4] matrix inside it
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Corner, 0, 1>(Corner(A));

    REQUIRE(view.getNumDims() == 2);
    REQUIRE(view.getShape() == "(2,4)");
    REQUIRE(view.getTotalSize() == 8);
}

TEMPLATE_TEST_CASE("squeeze_view: 5D down to 3D",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 2, 1, 3, 5>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0, 2>(A);

    REQUIRE(view.getNumDims() == 3);
    REQUIRE(view.getShape() == "(2,3,5)");
    REQUIRE(view.getTotalSize() == 30);
}

TEST_CASE("squeeze_view: Dim agrees with getDim",
          "[squeeze_view]")
{
    using View = SqueezeView<FusedTensorND<double, 1, 2, 1, 3, 5>, 0, 2>;

    REQUIRE(View::NumDims == 3);
    for (my_size_t i = 0; i < View::NumDims; ++i)
    {
        REQUIRE(View::Dim[i] == View::getDim(i));
    }
}

// ============================================================================
// ELEMENT ACCESS
// ============================================================================

TEMPLATE_TEST_CASE("squeeze_view: leading axis reads the source",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0>(A);

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(view(i, j) == A(0, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: middle axis reads the source",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 1, 4>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 1>(A);

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(view(i, j) == A(i, 0, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: corner of a 4D tensor reads the source",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Corner, 0, 1>(Corner(A));

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(view(i, j) == A(1, 2, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: padded source reads the source",
                   "[squeeze_view]", double, float, int)
{
    // A last axis of 5 is padded, so rows sit further apart than 5
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 5>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0>(A);

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            REQUIRE(view(i, j) == A(0, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: 5D source reads the source",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 2, 1, 3, 5>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0, 2>(A);

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            for (my_size_t k = 0; k < 5; ++k)
            {
                REQUIRE(view(i, j, k) == A(0, i, 0, j, k));
            }
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: array of indices",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0>(A);

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            my_size_t idx[2] = {i, j};
            REQUIRE(view(idx) == A(0, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: drop order does not matter",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 1, 4>;

    Tensor A;
    A.setSequencial();

    auto forward = SqueezeView<Tensor, 0, 2>(A);
    auto reversed = SqueezeView<Tensor, 2, 0>(A);

    REQUIRE(forward.getShape() == reversed.getShape());
    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(forward(i, j) == reversed(i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: reads live source data",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0>(A);

    A(0, 1, 2) = T(100);
    REQUIRE(view(1, 2) == T(100));
}

// ============================================================================
// CONTIGUITY AND PHYSICAL BASE
// ============================================================================

TEST_CASE("squeeze_view: unpadded source is contiguous",
          "[squeeze_view]")
{
    using View = SqueezeView<FusedTensorND<double, 1, 3, 4>, 0>;

    REQUIRE(View::IsPhysicallyContiguous);
}

TEST_CASE("squeeze_view: padded source is not contiguous",
          "[squeeze_view]")
{
    // Rows hold 5 logical elements and sit one padded row apart
    using View = SqueezeView<FusedTensorND<double, 1, 3, 5>, 0>;

    REQUIRE_FALSE(View::IsPhysicallyContiguous);
}

TEST_CASE("squeeze_view: corner of a 4D tensor is contiguous",
          "[squeeze_view]")
{
    using Tensor = FusedTensorND<double, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    REQUIRE(SqueezeView<Corner, 0, 1>::IsPhysicallyContiguous);
}

TEST_CASE("squeeze_view: plane of a padded 3D tensor is not contiguous",
          "[squeeze_view]")
{
    using Tensor = FusedTensorND<double, 2, 4, 5>;
    using Plane = MultiSliceView<Tensor, Slice<0, 1, 1>>;

    REQUIRE_FALSE(SqueezeView<Plane, 0>::IsPhysicallyContiguous);
}

TEST_CASE("squeeze_view: physical base of a tensor source is zero",
          "[squeeze_view]")
{
    REQUIRE(SqueezeView<FusedTensorND<double, 1, 3, 4>, 0>::PhysicalBase == 0);
}

TEST_CASE("squeeze_view: physical base is the slice's",
          "[squeeze_view]")
{
    // Dropping an axis moves nothing, so the squeeze starts where the slice does
    using Tensor = FusedTensorND<double, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    REQUIRE(Corner::PhysicalBase != 0);
    REQUIRE(SqueezeView<Corner, 0, 1>::PhysicalBase == Corner::PhysicalBase);
}

// ============================================================================
// ASSIGNMENT / MATERIALIZING A SQUEEZE
// ============================================================================

TEMPLATE_TEST_CASE("squeeze_view: assign to a matrix",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial();

    FusedMatrix<T, 3, 4> result;
    result = SqueezeView<Tensor, 0>(A);

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(result(i, j) == A(0, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: assign a padded source",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 5>;

    Tensor A;
    A.setSequencial();

    FusedMatrix<T, 3, 5> result;
    result = SqueezeView<Tensor, 0>(A);

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            REQUIRE(result(i, j) == A(0, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: assign the corner of a 4D tensor",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    Tensor A;
    A.setSequencial();

    FusedMatrix<T, 2, 4> result;
    result = SqueezeView<Corner, 0, 1>(Corner(A));

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(result(i, j) == A(1, 2, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: assign the corner of a padded 4D tensor",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 5>;
    using Corner = MultiSliceView<Tensor, Slice<0, 2, 1>, Slice<1, 4, 1>>;

    Tensor A;
    A.setSequencial();

    FusedMatrix<T, 2, 5> result;
    result = SqueezeView<Corner, 0, 1>(Corner(A));

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            REQUIRE(result(i, j) == A(2, 4, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: assign a plane of a padded 3D tensor",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 4, 5>;
    using Plane = MultiSliceView<Tensor, Slice<0, 1, 1>>;

    Tensor A;
    A.setSequencial();

    FusedMatrix<T, 4, 5> result;
    result = SqueezeView<Plane, 0>(Plane(A));

    for (my_size_t i = 0; i < 4; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            REQUIRE(result(i, j) == A(1, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: two pinned axes of a 3D tensor give a vector",
                   "[squeeze_view]", double, float, int)
{
    // Pinning axes 0 and 2 of [2,3,4] gives [1,3,1], which a FusedVector<T,3>
    // ([3,1]) rejects on rank. Squeezing axis 0 gives the [3,1] it accepts
    using T = TestType;
    using Tensor = FusedTensorND<T, 2, 3, 4>;
    using Line = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<2, 2, 1>>;

    Tensor A;
    A.setSequencial();

    FusedVector<T, 3> result;
    result = SqueezeView<Line, 0>(Line(A));

    for (my_size_t i = 0; i < 3; ++i)
    {
        REQUIRE(result(i) == A(1, i, 2));
    }
}

TEMPLATE_TEST_CASE("squeeze_view: assign to a 3D tensor",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 2, 1, 3, 5>;

    Tensor A;
    A.setSequencial();

    FusedTensorND<T, 2, 3, 5> result;
    result = SqueezeView<Tensor, 0, 2>(A);

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 3; ++j)
        {
            for (my_size_t k = 0; k < 5; ++k)
            {
                REQUIRE(result(i, j, k) == A(0, i, 0, j, k));
            }
        }
    }
}

// ============================================================================
// EXPRESSIONS
// ============================================================================

TEMPLATE_TEST_CASE("squeeze_view: squeeze plus squeeze",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 5>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0>(A);

    FusedMatrix<T, 3, 5> result;
    result = view + view;

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 5; ++j)
        {
            REQUIRE(result(i, j) == A(0, i, j) + A(0, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: squeeze times scalar",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    Tensor A;
    A.setSequencial();

    FusedMatrix<T, 2, 4> result;
    result = SqueezeView<Corner, 0, 1>(Corner(A)) * T(3);

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(result(i, j) == A(1, 2, i, j) * T(3));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: squeeze plus a matrix",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial();

    FusedMatrix<T, 3, 4> B;
    B.setSequencial();

    FusedMatrix<T, 3, 4> result;
    result = SqueezeView<Tensor, 0>(A) + B;

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(result(i, j) == A(0, i, j) + B(i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: clamp and min of a squeeze",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial(); // 0 .. 11

    auto view = SqueezeView<Tensor, 0>(A);

    FusedMatrix<T, 3, 4> clamped, lowered;
    clamped = clamp(view, T(2), T(9));
    lowered = min(view, T(5));

    for (my_size_t i = 0; i < 3; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            const T v = A(0, i, j);
            REQUIRE(clamped(i, j) == (v < T(2) ? T(2) : (v > T(9) ? T(9) : v)));
            REQUIRE(lowered(i, j) == (v < T(5) ? v : T(5)));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: equals an explicit matrix",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    Tensor A, B;
    A.setSequencial();
    B.setSequencial();

    FusedMatrix<T, 2, 4> expected;
    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            expected(i, j) = A(1, 2, i, j);
        }
    }

    REQUIRE(expected == SqueezeView<Corner, 0, 1>(Corner(A)));
    REQUIRE(SqueezeView<Corner, 0, 1>(Corner(A)) == SqueezeView<Corner, 0, 1>(Corner(B)));
}

// ============================================================================
// STORAGE AND LIFETIME
// ============================================================================

TEMPLATE_TEST_CASE("squeeze_view: copies and moves but does not assign",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;
    using View = SqueezeView<Corner, 0, 1>;

    CHECK(CopyConstructible<View>);
    CHECK(MoveConstructible<View>);

    // a view cannot be re-pointed at another source
    CHECK_FALSE(CopyAssignable<View>);
}

TEMPLATE_TEST_CASE("squeeze_view: a squeeze of a temporary slice keeps its own copy",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    Tensor A;
    A.setSequencial();

    // Corner(A) is a temporary that dies at the end of this statement
    auto view = SqueezeView<Corner, 0, 1>(Corner(A));

    FusedMatrix<T, 2, 4> result;
    result = view;

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(result(i, j) == A(1, 2, i, j));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: a stored expression over a squeeze evaluates later",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    Tensor A;
    A.setSequencial();

    // the slice and the squeeze are both temporaries of this statement
    auto e = SqueezeView<Corner, 0, 1>(Corner(A)) * T(2);

    FusedMatrix<T, 2, 4> result;
    result = e;

    for (my_size_t i = 0; i < 2; ++i)
    {
        for (my_size_t j = 0; j < 4; ++j)
        {
            REQUIRE(result(i, j) == A(1, 2, i, j) * T(2));
        }
    }
}

TEMPLATE_TEST_CASE("squeeze_view: data() is read-only",
                   "[squeeze_view]", double, float, int)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 3, 4>;

    Tensor A;
    A.setSequencial();

    auto view = SqueezeView<Tensor, 0>(A);

    // even through a non-const view: a view must not hand out write access
    CHECK((is_same_v<decltype(view.data()), const T *>));
}

TEST_CASE("squeeze_view: aliasing is checked against the source",
          "[squeeze_view]")
{
    using Tensor = FusedTensorND<double, 3, 5, 2, 4>;
    using Corner = MultiSliceView<Tensor, Slice<0, 1, 1>, Slice<1, 2, 1>>;

    Tensor A, B;
    auto view = SqueezeView<Corner, 0, 1>(Corner(A));

    REQUIRE(view.may_alias(A));
    REQUIRE_FALSE(view.may_alias(B));
}

TEST_CASE("squeeze_view: expression and algebraic traits",
          "[squeeze_view]")
{
    using View = SqueezeView<FusedTensorND<double, 1, 3, 4>, 0>;

    // Held by value inside expressions, and evaluated through logical indices
    REQUIRE_FALSE(expression::traits<View>::OwnsMemory);
    REQUIRE(expression::traits<View>::IsPhysical);
    REQUIRE(expression::traits<View>::IsPermuted);

    REQUIRE(algebra::is_vector_space_v<View>);
    REQUIRE(algebra::is_tensor_v<View>);
}

TEMPLATE_TEST_CASE("squeeze_view: matmul takes a copy, not the view",
                   "[squeeze_view]", double, float)
{
    using T = TestType;
    using Tensor = FusedTensorND<T, 1, 2, 3>;
    using View = SqueezeView<Tensor, 0>;

    // Like the slice, diagonal and subvector views, a squeeze is not an operand
    REQUIRE_FALSE((FusedMatrix<T, 2, 3>::template is_einsum_operand<View>));

    Tensor A;
    A.setSequencial(); // [[0, 1, 2], [3, 4, 5]]

    FusedMatrix<T, 2, 3> copy;
    copy = View(A);

    FusedMatrix<T, 3, 3> I3;
    I3.setIdentity();

    auto C = FusedMatrix<T, 2, 3>::matmul(copy, I3);
    CHECK(C == copy);

    // Rejected at compile time with "constraints not satisfied" on
    // FusedTensorND::is_einsum_operand. Uncomment to check:
    // auto D = FusedMatrix<T, 2, 3>::matmul(View(A), I3);
}

// ============================================================================
// NEGATIVE CASES: COMPILE-TIME ERRORS
// ============================================================================
// These are static_asserts. Uncomment one at a time; each must fail the build.
// Reading a member is required, an unused alias never instantiates. The
// policy's own cases are in unitest_squeezed_pad_policy.cpp.

TEST_CASE("squeeze_view: negative cases are compile-time errors",
          "[squeeze_view]")
{
    using Tensor = FusedTensorND<double, 1, 3, 4>;
    (void)sizeof(Tensor);

    // using View = SqueezeView<Tensor, 1>;                                    // axis 1 has extent 3
    // using View = SqueezeView<FusedMatrix<double, 1, 4>, 0>;                 // leaves [4]: below the rank floor
    // using View = SqueezeView<FusedTensorND<double, 2, 3, 1>, 2>;            // the 1 is padded: physically [2,3,4]
    // using View = SqueezeView<PermutedViewConstExpr<Tensor, 0, 2, 1>, 0>;    // permuted source

    // using Inner = SqueezeView<FusedTensorND<double, 1, 1, 3, 4>, 0>;
    // using View = SqueezeView<Inner, 0>;                                     // nesting

    // (void)View::NumDims;
}
