#pragma once
#include "BaseExpr.h"
#include "Operations.h"
#include "ops/op_traits.h"

// ===============================
// Binary Expression Template
// ===============================
template <typename LHS, typename RHS, template <typename, my_size_t, typename> class Op, typename T, my_size_t Bits, typename Arch>
class BinaryExpr : public BaseExpr<BinaryExpr<LHS, RHS, Op, T, Bits, Arch>, T>
{
    const LHS &_lhs;
    const RHS &_rhs;
    using type = typename Op<T, Bits, Arch>::type; // alias for easier usage
public:
    BinaryExpr(const LHS &lhs, const RHS &rhs) : _lhs(lhs), _rhs(rhs) {}

    // template <typename... Indices>
    // T operator()(Indices... indices) const
    // {
    //     return Op<T, Bits, GenericArch>::apply(_lhs(indices...), _rhs(indices...));
    // }

    template <my_size_t length>
    T operator()(my_size_t (&indices)[length]) const
    {
        return Op<T, Bits, GenericArch>::apply(_lhs(indices), _rhs(indices));
    }

    template <my_size_t length>
    type evalu(my_size_t (&indices)[length]) const
    {
        return Op<T, Bits, Arch>::apply(_lhs.evalu(indices), _rhs.evalu(indices));
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
