#pragma once
#include "BinaryExpr.h"
#include "ScalarExpr.h"
#include "Operations.h"

// ===============================
// Operator Overloads
// ===============================
template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Add, T> operator+(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs)
{
    return BinaryExpr<LHS, RHS, Add, T>(lhs.derived(), rhs.derived());
}

// // template<typename LHS, typename RHS, typename T>
// // auto operator+(const BaseExpr<LHS, T>& lhs, const BaseExpr<RHS, T>& rhs)
// // {
// //     using LHSExpr = std::decay_t<LHS>;
// //     using RHSExpr = std::decay_t<RHS>;
// //     using ResultExpr = BinaryExpr<LHSExpr, RHSExpr, Add, T>;

// //     // Wrap the sub-expressions in shared_ptr to manage lifetime safely
// //     auto lhs_ptr = std::make_shared<LHSExpr>(lhs.derived());
// //     auto rhs_ptr = std::make_shared<RHSExpr>(rhs.derived());

// //     return ResultExpr(std::move(lhs_ptr), std::move(rhs_ptr));
// // }

// template<typename LHS, typename RHS, typename T>
// auto operator+(const BaseExpr<LHS, T>& lhs, const BaseExpr<RHS, T>& rhs)
// {
//     using LHSExpr = std::decay_t<LHS>;
//     using RHSExpr = std::decay_t<RHS>;
//     using ResultExpr = BinaryExpr<LHSExpr, RHSExpr, Add, T>;

//     // Capture by reference, without taking ownership
//     auto lhs_ptr = std::shared_ptr<const LHSExpr>(&lhs.derived(), [](const LHSExpr*) {});
//     auto rhs_ptr = std::shared_ptr<const RHSExpr>(&rhs.derived(), [](const RHSExpr*) {});

//     return ResultExpr(std::move(lhs_ptr), std::move(rhs_ptr));
// }

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Sub, T> operator-(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs)
{
    return BinaryExpr<LHS, RHS, Sub, T>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Mul, T> operator*(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs)
{
    return BinaryExpr<LHS, RHS, Mul, T>(lhs.derived(), rhs.derived());
}

template <typename LHS, typename RHS, typename T>
BinaryExpr<LHS, RHS, Div, T> operator/(const BaseExpr<LHS, T> &lhs, const BaseExpr<RHS, T> &rhs)
{
    return BinaryExpr<LHS, RHS, Div, T>(lhs.derived(), rhs.derived());
}

// Scalar overloads



// template <typename LHS, typename T>
// auto operator+(const BaseExpr<LHS, T>& lhs, T scalar)
// {
//     using LHSExpr = std::decay_t<LHS>;
//     using ResultExpr = ScalarExpr<LHSExpr, Add, T>;

//     // Create a non-owning shared_ptr to lhs.derived()
//     auto lhs_ptr = std::shared_ptr<const LHSExpr>(&lhs.derived(), [](const LHSExpr*) {});

//     return ResultExpr(std::move(lhs_ptr), scalar);
// }

// matrix + scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Add, T> operator+(const BaseExpr<LHS, T> &lhs, T scalar)
{
    return ScalarExprRHS<LHS, Add, T>(lhs.derived(), scalar);
}

// scalar + matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprRHS<RHS, Add, T> operator+(T scalar, const BaseExpr<RHS, T> &rhs)
{
    return ScalarExprRHS<RHS, Add, T>(rhs.derived(), scalar);
}

// Override operator- to get the negative
template <typename RHS, typename T>
ScalarExprLHS<RHS, Sub, T> operator-(const BaseExpr<RHS, T> &expr)
{
    return ScalarExprLHS<RHS, Sub, T>(expr.derived(), T(0)); // Negation is like subtracting from zero
}

// matrix - scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Sub, T> operator-(const BaseExpr<LHS, T> &lhs, T scalar)
{
    return ScalarExprRHS<LHS, Sub, T>(lhs.derived(), scalar);
}

// scalar - matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprLHS<RHS, Sub, T> operator-(T scalar, const BaseExpr<RHS, T> &rhs)
{
    return ScalarExprLHS<RHS, Sub, T>(rhs.derived(), scalar);
}

// matrix * scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Mul, T> operator*(const BaseExpr<LHS, T> &lhs, T scalar)
{
    return ScalarExprRHS<LHS, Mul, T>(lhs.derived(), scalar);
}

// scalar * matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprRHS<RHS, Mul, T> operator*(T scalar, const BaseExpr<RHS, T> &rhs)
{
    return ScalarExprRHS<RHS, Mul, T>(rhs.derived(), scalar);
}

// matrix / scalar (scalar on RHS)
template <typename LHS, typename T>
ScalarExprRHS<LHS, Div, T> operator/(const BaseExpr<LHS, T> &lhs, T scalar)
{
    return ScalarExprRHS<LHS, Div, T>(lhs.derived(), scalar);
}

// scalar / matrix (scalar on LHS)
template <typename RHS, typename T>
ScalarExprLHS<RHS, Div, T> operator/(T scalar, const BaseExpr<RHS, T> &rhs)
{
    return ScalarExprLHS<RHS, Div, T>(rhs.derived(), scalar);
}
