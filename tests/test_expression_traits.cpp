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
        static_assert(expression::traits<Tensor>::IsPhysical, "FusedTensorND should be physical");

        static_assert(!expression::traits<Matrix>::IsPermuted, "FusedMatrix should NOT be permuted");
        static_assert(expression::traits<Matrix>::IsContiguous, "FusedMatrix should be contiguous");
        static_assert(expression::traits<Matrix>::IsPhysical, "FusedMatrix should be physical");
        SUCCEED("Base tensors traits checks passed");
    }

    SECTION("Permuted views")
    {
        using Tensor = FusedTensorND<double, 3, 3>;

        using View1 = PermutedViewConstExpr<Tensor, 1, 0>; // simple 2D permute
        using Traits1 = expression::traits<View1>;

        static_assert(Traits1::IsPermuted, "PermutedViewConstExpr should be permuted");
        static_assert(!Traits1::IsContiguous, "Permuted view should not be contiguous");
        static_assert(Traits1::IsPhysical, "Permuted view should still be physical");

        // Identity permutation: is_sequential detects 0,1 at compile time -> IsPermuted = false
        static_assert(!expression::traits<PermutedViewConstExpr<Tensor, 0, 1>>::IsPermuted, "Identity permute should not be permuted");
        static_assert(expression::traits<PermutedViewConstExpr<Tensor, 0, 1>>::IsPhysical, "Identity permuted view should be physical");
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
        static_assert(!TraitsRHS::IsPhysical, "ScalarExprRHS should NOT be physical");

        // scalar + tensor
        auto exprLHS = 3.0 + fmat;
        using TraitsLHS = expression::traits<decltype(exprLHS)>;
        static_assert(!TraitsLHS::IsPermuted, "ScalarExprLHS should propagate IsPermuted");
        static_assert(TraitsLHS::IsContiguous, "ScalarExprLHS should propagate IsContiguous");
        static_assert(!TraitsLHS::IsPhysical, "ScalarExprLHS should NOT be physical");

        // scalar + permuted tensor
        auto permuted = fmat.transpose_view();
        auto exprPerm = 3.0 + permuted;
        using TraitsPerm = expression::traits<decltype(exprPerm)>;
        static_assert(TraitsPerm::IsPermuted, "ScalarExpr with permuted child should be permuted");
        static_assert(!TraitsPerm::IsContiguous, "ScalarExpr with permuted child should not be contiguous");
        static_assert(!TraitsPerm::IsPhysical, "ScalarExpr with permuted child should NOT be physical");
        SUCCEED("Scalar expressions traits checks passed");
    }

    SECTION("Binary expressions")
    {
        FusedTensorND<double, 3, 3> fmat1, fmat2;

        fmat1.setToZero();
        fmat2.setToZero();

        auto expr = fmat1 + fmat2;
        using TraitsBin = expression::traits<decltype(expr)>;

        static_assert(!TraitsBin::IsPermuted, "BinaryExpr should propagate child IsPermuted");
        static_assert(TraitsBin::IsContiguous, "BinaryExpr should propagate child IsContiguous");
        static_assert(!TraitsBin::IsPhysical, "BinaryExpr should NOT be physical");

        // one child permuted
        auto permuted = fmat2.transpose_view();
        auto expr2 = fmat1 + permuted;
        using TraitsBinPerm = expression::traits<decltype(expr2)>;

        static_assert(TraitsBinPerm::IsPermuted, "BinaryExpr with permuted child should be permuted");
        static_assert(!TraitsBinPerm::IsContiguous, "BinaryExpr with permuted child should not be contiguous");
        static_assert(!TraitsBinPerm::IsPhysical, "BinaryExpr with permuted child should NOT be physical");
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
        static_assert(!Traits::IsPhysical, "Nested scalar+binary should NOT be physical");

        // nested with permuted view
        auto permuted = fmat2.transpose_view();
        auto expr2 = (fmat1 + 3.0) + permuted;
        using Traits2 = expression::traits<decltype(expr2)>;

        static_assert(Traits2::IsPermuted, "Nested scalar+binary with permuted child should be permuted");
        static_assert(!Traits2::IsContiguous, "Nested scalar+binary with permuted child should not be contiguous");
        static_assert(!Traits2::IsPhysical, "Nested scalar+binary with permuted child should NOT be physical");
        SUCCEED("Nested scalar+binary expressions traits checks passed");
    }

    SECTION("FMA expressions")
    {
        FusedTensorND<double, 3, 3> A, B, C;

        // FmaExpr with contiguous operands
        auto expr1 = A * B + C;
        using TraitsFma = expression::traits<decltype(expr1)>;
        static_assert(!TraitsFma::IsPermuted, "FmaExpr should propagate IsPermuted");
        static_assert(TraitsFma::IsContiguous, "FmaExpr should propagate IsContiguous");
        static_assert(!TraitsFma::IsPhysical, "FmaExpr should NOT be physical");

        // FmaExpr with one permuted operand
        auto permuted = C.transpose_view();
        auto expr2 = A * B + permuted;
        using TraitsFmaPerm = expression::traits<decltype(expr2)>;
        static_assert(TraitsFmaPerm::IsPermuted, "FmaExpr with permuted child should be permuted");
        static_assert(!TraitsFmaPerm::IsContiguous, "FmaExpr with permuted child should not be contiguous");
        static_assert(!TraitsFmaPerm::IsPhysical, "FmaExpr with permuted child should NOT be physical");

        // FmaExpr with permuted multiply operand
        auto permutedA = A.transpose_view();
        auto expr3 = permutedA * B + C;
        using TraitsFmaPermA = expression::traits<decltype(expr3)>;
        static_assert(TraitsFmaPermA::IsPermuted, "FmaExpr with permuted A should be permuted");
        static_assert(!TraitsFmaPermA::IsContiguous, "FmaExpr with permuted A should not be contiguous");
        static_assert(!TraitsFmaPermA::IsPhysical, "FmaExpr with permuted A should NOT be physical");

        SUCCEED("FMA expression traits checks passed");
    }

    SECTION("ScalarFMA expressions")
    {
        FusedTensorND<double, 3, 3> A, C;

        // ScalarFmaExpr with contiguous operands
        auto expr1 = A * 2.0 + C;
        using TraitsScFma = expression::traits<decltype(expr1)>;
        static_assert(!TraitsScFma::IsPermuted, "ScalarFmaExpr should propagate IsPermuted");
        static_assert(TraitsScFma::IsContiguous, "ScalarFmaExpr should propagate IsContiguous");
        static_assert(!TraitsScFma::IsPhysical, "ScalarFmaExpr should NOT be physical");

        // ScalarFmaExpr with permuted addend
        auto permuted = C.transpose_view();
        auto expr2 = A * 2.0 + permuted;
        using TraitsScFmaPerm = expression::traits<decltype(expr2)>;
        static_assert(TraitsScFmaPerm::IsPermuted, "ScalarFmaExpr with permuted addend should be permuted");
        static_assert(!TraitsScFmaPerm::IsContiguous, "ScalarFmaExpr with permuted addend should not be contiguous");
        static_assert(!TraitsScFmaPerm::IsPhysical, "ScalarFmaExpr with permuted addend should NOT be physical");

        // ScalarFmaExpr with permuted expr
        auto permutedA = A.transpose_view();
        auto expr3 = permutedA * 2.0 + C;
        using TraitsScFmaPermA = expression::traits<decltype(expr3)>;
        static_assert(TraitsScFmaPermA::IsPermuted, "ScalarFmaExpr with permuted expr should be permuted");
        static_assert(!TraitsScFmaPermA::IsContiguous, "ScalarFmaExpr with permuted expr should not be contiguous");
        static_assert(!TraitsScFmaPermA::IsPhysical, "ScalarFmaExpr with permuted expr should NOT be physical");

        SUCCEED("ScalarFMA expression traits checks passed");
    }

    SECTION("Nested FMA expressions")
    {
        FusedTensorND<double, 3, 3> A, B, C, D;

        // (A * B + C) * D + A → FmaExpr nested in FmaExpr
        auto expr1 = (A * B + C) * D + A;
        using Traits1 = expression::traits<decltype(expr1)>;
        static_assert(!Traits1::IsPermuted);
        static_assert(Traits1::IsContiguous);
        static_assert(!Traits1::IsPhysical);

        // (A * 2.0 + C) * B + D → ScalarFmaExpr nested in FmaExpr
        auto expr2 = (A * 2.0 + C) * B + D;
        using Traits2 = expression::traits<decltype(expr2)>;
        static_assert(!Traits2::IsPermuted);
        static_assert(Traits2::IsContiguous);
        static_assert(!Traits2::IsPhysical);

        // A * B + (C * D + A) → FmaExpr as addend
        auto expr3 = A * B + (C * D + A);
        using Traits3 = expression::traits<decltype(expr3)>;
        static_assert(!Traits3::IsPermuted);
        static_assert(Traits3::IsContiguous);
        static_assert(!Traits3::IsPhysical);

        // nested with permuted view
        auto permuted = D.transpose_view();
        auto expr4 = A * B + (C * 2.0 + permuted);
        using Traits4 = expression::traits<decltype(expr4)>;
        static_assert(Traits4::IsPermuted);
        static_assert(!Traits4::IsContiguous);
        static_assert(!Traits4::IsPhysical);

        // chain: (A * B - C) * 2.0 + D
        auto expr5 = (A * B - C) * 2.0 + D;
        using Traits5 = expression::traits<decltype(expr5)>;
        static_assert(!Traits5::IsPermuted);
        static_assert(Traits5::IsContiguous);
        static_assert(!Traits5::IsPhysical);

        // deep nesting: -(A * B + C) * D + A
        auto expr6 = -(A * B + C) * D + A;
        using Traits6 = expression::traits<decltype(expr6)>;
        static_assert(!Traits6::IsPermuted);
        static_assert(Traits6::IsContiguous);
        static_assert(!Traits6::IsPhysical);

        SUCCEED("Nested FMA expression traits checks passed");
    }
}