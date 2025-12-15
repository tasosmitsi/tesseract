#pragma once
#include "fused/BaseExpr.h"
#include "fused/Operations.h"

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
    // Expose compile-time shape if LHS provides it
    // static constexpr my_size_t NumDims = LHS::getNumDims();

    BinaryExpr(const LHS &lhs, const RHS &rhs) : _lhs(lhs), _rhs(rhs) {}

    template <my_size_t length>
    inline T operator()(my_size_t (&indices)[length]) const noexcept
    {
        return Op<T, Bits, GenericArch>::apply(_lhs(indices), _rhs(indices));
    }

    // template <my_size_t length>
    inline type evalu(const my_size_t flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(_lhs.evalu(flat), _rhs.evalu(flat));
    }

    // Forward getNumDims to _lhs
    inline my_size_t getNumDims() const noexcept
    {
        return _lhs.getNumDims();
    }

    // Forward getDim(i) to _lhs
    inline my_size_t getDim(my_size_t i) const // TODO: conditionally noexcept
    {
        return _lhs.getDim(i);
    }

    my_size_t getTotalSize() const noexcept
    {
        return _lhs.getTotalSize();
    }

protected:
    inline T operator()(const my_size_t *indices) const noexcept
    {
        return Op<T, Bits, GenericArch>::apply(_lhs(indices), _rhs(indices));
    }
};
