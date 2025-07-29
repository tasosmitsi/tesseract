#pragma once
#include "BaseExpr.h"
#include <immintrin.h>

// ===============================
// Binary Expression Template
// ===============================

struct SIMD4
{
};

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

    // template <my_size_t length>
    // __m128 evalu(my_size_t (&indices)[length]) const
    // {
    //     return Op<__m128>::apply(_lhs.evalu(indices), _rhs.evalu(indices));
    // }

    template <my_size_t length>
    __m256 evalu(my_size_t (&indices)[length]) const
    {
        return Op<__m256>::apply(_lhs.evalu(indices), _rhs.evalu(indices));
    }

    // SIMD operator() overload, triggered by SIMD4 tag as last argument
    template <typename... Indices>
    __m128 operator()(Indices... indices, SIMD4) const
    {
        // Call the operands with SIMD awareness
        __m128 lhsVec = _lhs(indices..., SIMD4{});
        __m128 rhsVec = _rhs(indices..., SIMD4{});
        return Op<__m128>::apply(lhsVec, rhsVec);
    }

    // Forward getNumDims to _lhs
    inline my_size_t getNumDims() const
    {
        return _lhs.getNumDims();
    }

    // Forward getDim(i) to _lhs
    inline my_size_t getDim(my_size_t i) const
    {
        return _lhs.getDim(i);
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
