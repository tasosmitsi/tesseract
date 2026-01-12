#pragma once
#include "config.h"
#include "fused/BinaryExpr.h"
#include "fused/ScalarExpr.h"
#include "fused/Operations.h"
#include "simple_type_traits.h"
#include "helper_traits.h"
#include "algebra/algebraic_traits.h"

template <typename Expr1, typename Expr2>
inline void checkDimsMatch(const Expr1 &lhs, const Expr2 &rhs, const std::string &opName) // TODO: conditionally noexcept
{
#ifdef RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH
    if (lhs.getNumDims() != rhs.getNumDims())
        MyErrorHandler::error(opName + ": dimension count mismatch");
#endif

#ifdef RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH
    for (my_size_t i = 0; i < lhs.getNumDims(); ++i)
    {
        if (lhs.getDim(i) != rhs.getDim(i))
            MyErrorHandler::error(opName + ": dimension size mismatch at dimension " + std::to_string(i));
    }
#endif
}

// ===============================
// Operator Overloads
// ===============================
template <typename LHS, typename RHS>
    requires(algebra::is_vector_space_v<LHS> && algebra::is_vector_space_v<RHS>)
BinaryExpr<LHS, RHS, Add>
operator+(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+");
#endif
    return BinaryExpr<LHS, RHS, Add>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS>
    requires(algebra::is_vector_space_v<LHS> && algebra::is_vector_space_v<RHS>)
BinaryExpr<LHS, RHS, Sub>
operator-(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator-");
#endif
    return BinaryExpr<LHS, RHS, Sub>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS>
    requires( // for Hadamard product only it must be tensors, not general algebras
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Mul>
operator*(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator*");
#endif
    return BinaryExpr<LHS, RHS, Mul>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS>
    requires( // for Hadamard product (element-wise division) only it must be tensors, not general algebras
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Div>
operator/(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator/");
#endif
    return BinaryExpr<LHS, RHS, Div>(lhs.derived(), rhs.derived());
}

// matrix + scalar (scalar on RHS)
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Add>
operator+(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Add>(lhs.derived(), scalar);
}

// scalar + matrix (scalar on LHS)
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Add>
operator+(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Add>(rhs.derived(), scalar);
}

// Override operator- to get the negative
template <typename RHS>
    requires(algebra::is_vector_space_v<RHS>)
ScalarExprLHS<RHS, typename RHS::value_type, Sub>
operator-(const BaseExpr<RHS> &expr) noexcept
{
    using T = typename RHS::value_type;
    return ScalarExprLHS<RHS, T, Sub>(expr.derived(), T(0)); // Negation is like subtracting from zero
}

// matrix - scalar (scalar on RHS)
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Sub>
operator-(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Sub>(lhs.derived(), scalar);
}

// scalar - matrix (scalar on LHS)
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprLHS<RHS, T, Sub>
operator-(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprLHS<RHS, T, Sub>(rhs.derived(), scalar);
}

// matrix * scalar (scalar on RHS)
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Mul>
operator*(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Mul>(lhs.derived(), scalar);
}

// scalar * matrix (scalar on LHS)
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Mul>
operator*(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Mul>(rhs.derived(), scalar);
}

// matrix / scalar (scalar on RHS)
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Div>
operator/(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Div>(lhs.derived(), scalar);
}

// scalar / matrix (scalar on LHS)
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprLHS<RHS, T, Div>
operator/(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprLHS<RHS, T, Div>(rhs.derived(), scalar);
}

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

    // TODO: optimize this by checking using SIMD where possible
    // check the actual elements
    my_size_t indices[MAX_DIMS] = {0};
    for (my_size_t i = 0; i < lhs.derived().getTotalSize(); ++i)
    {
        // increment the indices using for loop
        for (my_size_t j = 0; j < lhs.derived().getNumDims(); ++j)
        {
            if (indices[j] < lhs.derived().getDim(j) - 1)
            {
                indices[j]++;
                break;
            }
            else
            {
                indices[j] = 0;
            }
        }
        // use the () operator to access the elements
        if (std::abs(lhs.derived()(indices) - rhs.derived()(indices)) > lhs_type(PRECISION_TOLERANCE))
        {
            return false;
        }
    }
    return true;
}

template <typename LHS, typename RHS>
bool operator!=(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) // TODO: conditionally noexcept
{
    return !(lhs == rhs);
}