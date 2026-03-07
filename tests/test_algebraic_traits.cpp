#include <catch_amalgamated.hpp>
#include "fused/fused_tensor.h"
#include "fused/fused_matrix.h"
#include "algebra/algebraic_traits.h"

TEST_CASE("Algebraic traits", "[algebraic_traits]")
{
    SECTION("FusedTensorND")
    {
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

        static_assert(!algebra::is_vector_space_v<BaseExpr<Tensor>>); // false since we don't have BaseExpr specialization

        static_assert(algebra::is_vector_space_v<Matrix>,
                      "FusedMatrix should be a vector space");

        static_assert(!algebra::is_algebra_v<Matrix>,
                      "FusedMatrix should not be an algebra");

        static_assert(!algebra::is_lie_group_v<Matrix>,
                      "FusedMatrix should not be a Lie group");

        static_assert(!algebra::is_metric_v<Matrix>,
                      "FusedMatrix should not be metric");

        static_assert(algebra::is_tensor_v<Matrix>,
                      "FusedMatrix should be tensor");

        static_assert(!algebra::is_vector_space_v<BaseExpr<Matrix>>); // false since we don't have BaseExpr specializations
    }
}