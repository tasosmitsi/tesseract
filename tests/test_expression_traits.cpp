#include <catch_amalgamated.hpp>
#include "fused/fused_tensor.h"
#include "fused/fused_matrix.h"
#include "expression_traits/expression_traits.h"

TEST_CASE("Expression traits", "[expression_traits]")
{
    SECTION("Base tensors")
    {
        using Tensor = FusedTensorND<double, 3, 3>;
        using Matrix = FusedMatrix<double, 3, 3>;

        // Base tensors are never permuted
        static_assert(!expression::traits<Tensor>::IsPermuted, "FusedTensorND should NOT be permuted");
        static_assert(expression::traits<Tensor>::IsContiguous, "FusedTensorND should be contiguous");

        static_assert(!expression::traits<Matrix>::IsPermuted, "FusedMatrix should NOT be permuted");
        static_assert(expression::traits<Matrix>::IsContiguous, "FusedMatrix should be contiguous");
        SUCCEED("Base tensors traits checks passed");
    }

    SECTION("Permuted views")
    {
        using Tensor = FusedTensorND<double, 3, 3>;

        using View1 = PermutedViewConstExpr<Tensor, 1, 0>; // simple 2D permute
        using Traits1 = expression::traits<View1>;

        static_assert(Traits1::IsPermuted, "PermutedViewConstExpr should be permuted");
        static_assert(!Traits1::IsContiguous, "Permuted view should not be contiguous");

        // Identity permutation (optional)
        // If Layout::IsPermProvided is false, IsPermuted will be false
        // TODO: fix this ispermprovided from layout is not enough here because
        // we can have a permuted view with identity permutation, which should be considered as not permuted
        static_assert(!expression::traits<PermutedViewConstExpr<Tensor,0,1>>::IsPermuted, "Identity permute not permuted");
        SUCCEED("Permuted views traits checks passed");
    }

    SECTION("Scalar expressions")
    {
        FusedTensorND<double, 3, 3> fmat;

        // tensor + scalar
        auto exprRHS = fmat + 3.0;
        using TraitsRHS = expression::traits<decltype(exprRHS)>;
        static_assert(!TraitsRHS::IsPermuted, "ScalarExprRHS should propagate IsPermuted");
        static_assert(TraitsRHS::IsContiguous, "ScalarExprRHS should propagate IsContiguous");

        // scalar + tensor
        auto exprLHS = 3.0 + fmat;
        using TraitsLHS = expression::traits<decltype(exprLHS)>;
        static_assert(!TraitsLHS::IsPermuted, "ScalarExprLHS should propagate IsPermuted");
        static_assert(TraitsLHS::IsContiguous, "ScalarExprLHS should propagate IsContiguous");

        // scalar + permuted tensor
        auto permuted = fmat.transpose_view();
        auto exprPerm = 3.0 + permuted;
        using TraitsPerm = expression::traits<decltype(exprPerm)>;
        static_assert(TraitsPerm::IsPermuted, "ScalarExpr with permuted child should be permuted");
        static_assert(!TraitsPerm::IsContiguous, "ScalarExpr with permuted child should not be contiguous");
        SUCCEED("Scalar expressions traits checks passed");
    }

    SECTION("Binary expressions")
    {
        FusedTensorND<double, 3, 3> fmat1, fmat2;

        auto expr = fmat1 + fmat2;
        using TraitsBin = expression::traits<decltype(expr)>;

        static_assert(!TraitsBin::IsPermuted, "BinaryExpr should propagate child IsPermuted");
        static_assert(TraitsBin::IsContiguous, "BinaryExpr should propagate child IsContiguous");

        // one child permuted
        auto permuted = fmat2.transpose_view();
        auto expr2 = fmat1 + permuted;
        using TraitsBinPerm = expression::traits<decltype(expr2)>;

        static_assert(TraitsBinPerm::IsPermuted, "BinaryExpr with permuted child should be permuted");
        static_assert(!TraitsBinPerm::IsContiguous, "BinaryExpr with permuted child should not be contiguous");
        SUCCEED("Binary expressions traits checks passed");
    }

    SECTION("Nested scalar + binary expressions")
    {
        FusedTensorND<double, 3, 3> fmat1, fmat2;

        // tensor + scalar + tensor
        auto expr = (fmat1 + 3.0) + fmat2; // scalar expr + tensor -> binary expr
        using Traits = expression::traits<decltype(expr)>;

        static_assert(!Traits::IsPermuted, "Nested scalar+binary should propagate child IsPermuted correctly");
        static_assert(Traits::IsContiguous, "Nested scalar+binary should propagate child IsContiguous correctly");

        // nested with permuted view
        auto permuted = fmat2.transpose_view();
        auto expr2 = (fmat1 + 3.0) + permuted;
        using Traits2 = expression::traits<decltype(expr2)>;

        static_assert(Traits2::IsPermuted, "Nested scalar+binary with permuted child should be permuted");
        static_assert(!Traits2::IsContiguous, "Nested scalar+binary with permuted child should not be contiguous");
        SUCCEED("Nested scalar+binary expressions traits checks passed");
    }
}