#pragma once
#include "config.h"
#include "fused/BinaryExpr.h"
#include "fused/ScalarExpr.h"
#include "fused/FmaExpr.h"
#include "fused/Operations.h"
#include "fused/operators/operators_common.h"
#include "simple_type_traits.h"
#include "algebra/algebraic_traits.h"

/**
 * @file arithmetic.h
 * @brief Arithmetic operators over tensors and scalars.
 *
 * Every operator returns an expression rather than a result, so a chain like
 * a + b * c - d evaluates in one pass with no intermediates.
 *
 * The overloads guarded by TESSERACT_USE_FMAD sit in front of the plain ones
 * and match the shapes that fold into a fused multiply-add. They are chosen by
 * overload resolution on the operand types: an operand that is already a Mul
 * expression is more specialised than a general BaseExpr, so a*b + c binds to
 * the FMA overload and a + c to the plain one.
 *
 * The multiply and divide operators are constrained to tensors and not general
 * algebras, since element-wise product is not what * should mean for a
 * quaternion for example. Addition and subtraction only need vector-space structure.
 */

// ===============================
// FMA detection: operator+
// ===============================

#ifdef TESSERACT_USE_FMAD

/// @brief (A * B) + C, folded into a single fused multiply-add.
template <typename L, typename R, typename C>
    requires(algebra::is_vector_space_v<BinaryExpr<L, R, Mul>> &&
             algebra::is_vector_space_v<C>)
FmaExpr<L, R, C, Fma>
operator+(const BaseExpr<BinaryExpr<L, R, Mul>> &lhs,
          const BaseExpr<C> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+ for FMA pattern (A*B + C)");
#endif
    const auto &mul = lhs.derived();
    return FmaExpr<L, R, C, Fma>(mul.lhs(), mul.rhs(), rhs.derived());
}

/// @brief C + (A * B), folded into a single fused multiply-add.
template <typename C, typename L, typename R>
    requires(algebra::is_vector_space_v<C> &&
             algebra::is_vector_space_v<BinaryExpr<L, R, Mul>>)
FmaExpr<L, R, C, Fma>
operator+(const BaseExpr<C> &lhs,
          const BaseExpr<BinaryExpr<L, R, Mul>> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+ for FMA pattern (C + A*B)");
#endif
    const auto &mul = rhs.derived();
    return FmaExpr<L, R, C, Fma>(mul.lhs(), mul.rhs(), lhs.derived());
}

/// @brief (A * scalar) + C, folded into a single fused multiply-add.
template <typename L, typename T, typename C>
    requires(algebra::is_vector_space_v<ScalarExprRHS<L, T, Mul>> &&
             algebra::is_vector_space_v<C>)
ScalarFmaExpr<L, T, C, Fma>
operator+(const BaseExpr<ScalarExprRHS<L, T, Mul>> &lhs,
          const BaseExpr<C> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+ for FMA pattern (A*scalar + C)");
#endif
    const auto &mul = lhs.derived();
    return ScalarFmaExpr<L, T, C, Fma>(mul.expr(), mul.scalar(), rhs.derived());
}

/// @brief C + (A * scalar), folded into a single fused multiply-add.
template <typename C, typename L, typename T>
    requires(algebra::is_vector_space_v<C> &&
             algebra::is_vector_space_v<ScalarExprRHS<L, T, Mul>>)
ScalarFmaExpr<L, T, C, Fma>
operator+(const BaseExpr<C> &lhs,
          const BaseExpr<ScalarExprRHS<L, T, Mul>> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+ for FMA pattern (C + A*scalar)");
#endif
    const auto &mul = rhs.derived();
    return ScalarFmaExpr<L, T, C, Fma>(mul.expr(), mul.scalar(), lhs.derived());
}

/// @brief -(A * B) + C, folded into a fused negative multiply-add.
template <typename L, typename R, typename T, typename C>
    requires(algebra::is_vector_space_v<ScalarExprLHS<BinaryExpr<L, R, Mul>, T, Sub>> &&
             algebra::is_vector_space_v<C>)
FmaExpr<L, R, C, Fnma>
operator+(const BaseExpr<ScalarExprLHS<BinaryExpr<L, R, Mul>, T, Sub>> &lhs,
          const BaseExpr<C> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+ for FMA pattern (-(A*B) + C)");
#endif
    const auto &neg = lhs.derived();
    const auto &mul = neg.expr(); // the BinaryExpr<L, R, Mul>
    return FmaExpr<L, R, C, Fnma>(mul.lhs(), mul.rhs(), rhs.derived());
}

/// @brief -(A * scalar) + C, folded into a fused negative multiply-add.
template <typename L, typename T1, typename T2, typename C>
    requires(algebra::is_vector_space_v<ScalarExprLHS<ScalarExprRHS<L, T1, Mul>, T2, Sub>> &&
             algebra::is_vector_space_v<C>)
ScalarFmaExpr<L, T1, C, Fnma>
operator+(const BaseExpr<ScalarExprLHS<ScalarExprRHS<L, T1, Mul>, T2, Sub>> &lhs,
          const BaseExpr<C> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+ for FMA pattern (-(A*scalar) + C)");
#endif
    const auto &neg = lhs.derived();
    const auto &mul = neg.expr();
    return ScalarFmaExpr<L, T1, C, Fnma>(mul.expr(), mul.scalar(), rhs.derived());
}

// ===============================
// FMA detection: operator-
// ===============================

/// @brief (A * B) - C, folded into a fused multiply-subtract.
template <typename L, typename R, typename C>
    requires(algebra::is_vector_space_v<BinaryExpr<L, R, Mul>> &&
             algebra::is_vector_space_v<C>)
FmaExpr<L, R, C, Fms>
operator-(const BaseExpr<BinaryExpr<L, R, Mul>> &lhs,
          const BaseExpr<C> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator- for FMA pattern (A*B - C)");
#endif
    const auto &mul = lhs.derived();
    return FmaExpr<L, R, C, Fms>(mul.lhs(), mul.rhs(), rhs.derived());
}

/// @brief C - (A * B), folded into a fused negative multiply-add.
template <typename C, typename L, typename R>
    requires(algebra::is_vector_space_v<C> &&
             algebra::is_vector_space_v<BinaryExpr<L, R, Mul>>)
FmaExpr<L, R, C, Fnma>
operator-(const BaseExpr<C> &lhs,
          const BaseExpr<BinaryExpr<L, R, Mul>> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator- for FMA pattern (C - A*B)");
#endif
    const auto &mul = rhs.derived();
    return FmaExpr<L, R, C, Fnma>(mul.lhs(), mul.rhs(), lhs.derived());
}

/// @brief -(A * B) - C, folded into a fused negative multiply-subtract.
template <typename L, typename R, typename T, typename C>
    requires(algebra::is_vector_space_v<ScalarExprLHS<BinaryExpr<L, R, Mul>, T, Sub>> &&
             algebra::is_vector_space_v<C>)
FmaExpr<L, R, C, Fnms>
operator-(const BaseExpr<ScalarExprLHS<BinaryExpr<L, R, Mul>, T, Sub>> &lhs,
          const BaseExpr<C> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator- for FMA pattern (-(A*B) - C)");
#endif
    const auto &neg = lhs.derived();
    const auto &mul = neg.expr();
    return FmaExpr<L, R, C, Fnms>(mul.lhs(), mul.rhs(), rhs.derived());
}

/// @brief -(A * scalar) - C, folded into a fused negative multiply-subtract.
template <typename L, typename T1, typename T2, typename C>
    requires(algebra::is_vector_space_v<ScalarExprLHS<ScalarExprRHS<L, T1, Mul>, T2, Sub>> &&
             algebra::is_vector_space_v<C>)
ScalarFmaExpr<L, T1, C, Fnms>
operator-(const BaseExpr<ScalarExprLHS<ScalarExprRHS<L, T1, Mul>, T2, Sub>> &lhs,
          const BaseExpr<C> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator- for FMA pattern (-(A*scalar) - C)");
#endif
    const auto &neg = lhs.derived();
    const auto &mul = neg.expr();
    return ScalarFmaExpr<L, T1, C, Fnms>(mul.expr(), mul.scalar(), rhs.derived());
}

/// @brief (A * scalar) - C, folded into a fused multiply-subtract.
template <typename L, typename T, typename C>
    requires(algebra::is_vector_space_v<ScalarExprRHS<L, T, Mul>> &&
             algebra::is_vector_space_v<C>)
ScalarFmaExpr<L, T, C, Fms>
operator-(const BaseExpr<ScalarExprRHS<L, T, Mul>> &lhs,
          const BaseExpr<C> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator- for FMA pattern (A*scalar - C)");
#endif
    const auto &mul = lhs.derived();
    return ScalarFmaExpr<L, T, C, Fms>(mul.expr(), mul.scalar(), rhs.derived());
}

/// @brief C - (A * scalar), folded into a fused negative multiply-add.
template <typename C, typename L, typename T>
    requires(algebra::is_vector_space_v<C> &&
             algebra::is_vector_space_v<ScalarExprRHS<L, T, Mul>>)
ScalarFmaExpr<L, T, C, Fnma>
operator-(const BaseExpr<C> &lhs,
          const BaseExpr<ScalarExprRHS<L, T, Mul>> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator- for FMA pattern (C - A*scalar)");
#endif
    const auto &mul = rhs.derived();
    return ScalarFmaExpr<L, T, C, Fnma>(mul.expr(), mul.scalar(), lhs.derived());
}
#endif // TESSERACT_USE_FMAD

// ===============================
// binary detection: operator+
// ===============================

/**
 * @brief Element-wise sum of two tensors.
 * @throws if the dimensions do not match and runtime checks are enabled.
 */
template <typename LHS, typename RHS>
    requires(algebra::is_vector_space_v<LHS> && algebra::is_vector_space_v<RHS>)
BinaryExpr<LHS, RHS, Add>
operator+(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator+");
#endif
    return BinaryExpr<LHS, RHS, Add>(lhs.derived(), rhs.derived());
}

/**
 * @brief Element-wise difference of two tensors.
 * @throws if the dimensions do not match and runtime checks are enabled.
 */
template <typename LHS, typename RHS>
    requires(algebra::is_vector_space_v<LHS> && algebra::is_vector_space_v<RHS>)
BinaryExpr<LHS, RHS, Sub>
operator-(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator-");
#endif
    return BinaryExpr<LHS, RHS, Sub>(lhs.derived(), rhs.derived());
}

/**
 * @brief Element-wise product (Hadamard), not matrix multiplication.
 * @throws if the dimensions do not match and runtime checks are enabled.
 */
template <typename LHS, typename RHS>
    requires( // for Hadamard product only it must be tensors, not general algebras
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Mul>
operator*(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator*");
#endif
    return BinaryExpr<LHS, RHS, Mul>(lhs.derived(), rhs.derived());
}

/**
 * @brief Element-wise division.
 * @throws if the dimensions do not match and runtime checks are enabled.
 */
template <typename LHS, typename RHS>
    requires( // for Hadamard product (element-wise division) only it must be tensors, not general algebras
        algebra::is_tensor_v<LHS> &&
        algebra::is_tensor_v<RHS> &&
        !algebra::is_algebra_v<LHS> &&
        !algebra::is_algebra_v<RHS>)
BinaryExpr<LHS, RHS, Div>
operator/(const BaseExpr<LHS> &lhs, const BaseExpr<RHS> &rhs) TESSERACT_CONDITIONAL_NOEXCEPT
{
#if defined(RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH) || defined(RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH)
    checkDimsMatch(lhs.derived(), rhs.derived(), "operator/");
#endif
    return BinaryExpr<LHS, RHS, Div>(lhs.derived(), rhs.derived());
}

/// @brief Add @p scalar to every element.
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Add>
operator+(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Add>(lhs.derived(), scalar);
}

/// @brief Add @p scalar to every element. Addition is commutative, so this forwards.
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Add>
operator+(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Add>(rhs.derived(), scalar);
}

/// @brief Negate every element, expressed as subtraction from zero.
template <typename RHS>
    requires(algebra::is_vector_space_v<RHS>)
ScalarExprLHS<RHS, typename RHS::value_type, Sub>
operator-(const BaseExpr<RHS> &expr) noexcept
{
    using T = typename RHS::value_type;
    return ScalarExprLHS<RHS, T, Sub>(expr.derived(), T(0)); // Negation is like subtracting from zero
}

/// @brief Subtract @p scalar from every element.
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Sub>
operator-(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Sub>(lhs.derived(), scalar);
}

/// @brief Subtract every element from @p scalar. Not commutative, hence ScalarExprLHS.
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprLHS<RHS, T, Sub>
operator-(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprLHS<RHS, T, Sub>(rhs.derived(), scalar);
}

/// @brief Scale every element by @p scalar.
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Mul>
operator*(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Mul>(lhs.derived(), scalar);
}

/// @brief Scale every element by @p scalar. Multiplication is commutative, so this forwards.
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<RHS, T, Mul>
operator*(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprRHS<RHS, T, Mul>(rhs.derived(), scalar);
}

/// @brief Divide every element by @p scalar.
template <typename LHS, typename T>
    requires(algebra::is_vector_space_v<LHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprRHS<LHS, T, Div>
operator/(const BaseExpr<LHS> &lhs, T scalar) noexcept
{
    return ScalarExprRHS<LHS, T, Div>(lhs.derived(), scalar);
}

/// @brief Divide @p scalar by every element. Not commutative, hence ScalarExprLHS.
template <typename RHS, typename T>
    requires(algebra::is_vector_space_v<RHS> &&
             !is_base_of_v<detail::BaseExprTag, T>)
ScalarExprLHS<RHS, T, Div>
operator/(T scalar, const BaseExpr<RHS> &rhs) noexcept
{
    return ScalarExprLHS<RHS, T, Div>(rhs.derived(), scalar);
}
