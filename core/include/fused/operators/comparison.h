#pragma once
#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/operators/operators_common.h"
#include "simple_type_traits.h"
#include "helper_traits.h"
#include "fused/microkernels/kernel_ops.h"

// ===============================
// Comparison Operators
// ===============================

template <typename LHS, typename RHS>
bool operator==(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
    using lhs_type = typename LHS::value_type;
    using rhs_type = typename RHS::value_type;
    static_assert(is_same_v<lhs_type, rhs_type>,
                  "operator== or operator!= : Cannot compare tensors with different scalar types");

#ifdef COMPILETIME_CHECK_DIMENSIONS_COUNT_MISMATCH
    // Compile-time check that both expressions have the same number of dimensions
    static_assert(LHS::NumDims == RHS::NumDims, "operator== or operator!= : Cannot compare tensors with different number of dimensions");
#endif
#ifdef COMPILETIME_CHECK_DIMENSIONS_SIZE_MISMATCH
    // Compile-time check that both expressions have the same dimensions
    static_assert(dims_match<LHS::NumDims>(LHS::Dim, RHS::Dim),
                  "operator== or operator!= : there is at least one dimension mismatch");
#endif

    // runtime check that all dimensions are the same
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator== or operator!=");
#endif

    return KernelOps<lhs_type, BITS, DefaultArch>::reduce_all_approx_equal(
        lhs.derived(),
        rhs.derived(),
        lhs_type(PRECISION_TOLERANCE));
}

template <typename LHS, typename RHS>
bool operator!=(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
    return !(lhs == rhs);
}