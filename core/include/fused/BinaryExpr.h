#pragma once
#include "BaseExpr.h"

// ===============================
// Binary Expression Template
// ===============================
template <typename LHS, typename RHS, template <typename> class Op, typename T>
class BinaryExpr : public BaseExpr<BinaryExpr<LHS, RHS, Op, T>, T>
{
    const LHS &_lhs;
    const RHS &_rhs;

public:
    BinaryExpr(const LHS &lhs, const RHS &rhs) : _lhs(lhs), _rhs(rhs) {}

    template <typename... Indices>
    T operator()(Indices... indices) const
    {
        return Op<T>::apply(_lhs(indices...), _rhs(indices...));
    }
};

// template<typename EXPR>
// using ExprPtr = std::shared_ptr<const EXPR>;

// template <typename LHS, typename RHS, template <typename> class Op, typename T>
// class BinaryExpr : public BaseExpr<BinaryExpr<LHS, RHS, Op, T>, T> {
//     ExprPtr<LHS> _lhs;
//     ExprPtr<RHS> _rhs;

// public:
//     BinaryExpr(ExprPtr<LHS> lhs, ExprPtr<RHS> rhs)
//         : _lhs(std::move(lhs)), _rhs(std::move(rhs)) {}

//     // Access lhs_, rhs_ with operator() ...
// T operator()(my_size_t idx) const
// {
//     return Op<T>::apply((*_lhs)(idx), (*_rhs)(idx));
// }

// T operator()(my_size_t i, my_size_t j) const
// {
//     return Op<T>::apply((*_lhs)(i, j), (*_rhs)(i, j));
// }

// template <typename... Indices>
// T operator()(Indices... indices) const
// {
//     return Op<T>::apply((*_lhs)(indices...), (*_rhs)(indices...));
// }
// };
