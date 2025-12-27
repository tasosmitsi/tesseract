#include <catch_amalgamated.hpp>
#include "fused/fused_tensor.h"
#include "fused/fused_matrix.h"
#include "algebra/algebraic_traits.h"
#include "utilities.h"

TEST_CASE("Algebraic traits", "[algebraic_traits]")
{
    SECTION("FusedTensorND")
    {
        FusedTensorND<double, 3, 3> fmat1, fmat2;
        using Tensor = FusedTensorND<double, 3, 3>;
        using Matrix = FusedMatrix<double, 3, 3>;

        static_assert(algebra::is_vector_space_v<Tensor>,
                      "FusedTensorND should be a vector space");

        static_assert(!algebra::is_algebra_v<Tensor>,
                      "FusedTensorND should not be an algebra");

        static_assert(!algebra::is_lie_group_v<Tensor>,
                      "FusedTensorND should not be a Lie group");

        static_assert(!algebra::is_metric_v<Tensor>,
                      "FusedTensorND should not be metric");

        static_assert(algebra::is_tensor_v<Tensor>,
                      "FusedTensorND should be tensor");

        static_assert(!algebra::is_vector_space_v<BaseExpr<Tensor, float>>); // false since we don't have BaseExpr specialization

        auto fmat3 = fmat1 + fmat2 + fmat2;

        std ::cout << demangleTypeName(typeid(fmat3)) << std::endl;
    }
}