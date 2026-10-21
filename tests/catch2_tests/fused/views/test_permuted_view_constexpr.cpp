#include <catch_amalgamated.hpp>

#include "fused/fused_matrix.h"
#include "fused/views/permuted_view_constexpr.h"

// TODO: the transpose view is otherwise tested through test_fused_tensor.cpp
// and test_fused_matrix.cpp. Move those tests here so the file matches the header.

// ============================================================================
// READ-ONLY ACCESS
// ============================================================================

TEMPLATE_TEST_CASE("permuted_view: data() is read-only",
                   "[permuted_view]", double, float, int)
{
    using T = TestType;

    FusedMatrix<T, 3, 3> A(T(0));
    auto view = A.transpose_view();

    // even through a non-const view: a view must not hand out write access
    CHECK((is_same_v<decltype(view.data()), const T *>));
}
