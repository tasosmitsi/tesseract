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

    BinaryExpr(const LHS &lhs, const RHS &rhs) : _lhs(lhs), _rhs(rhs)
    {

        // // Compile-time dimension count check
        // static_assert(LHS::NumDims == RHS::NumDims,
        //               "Dimension count mismatch in BinaryExpr");
    }

    template <my_size_t length>
    T operator()(my_size_t (&indices)[length]) const
    {
        return Op<T, Bits, GenericArch>::apply(_lhs(indices), _rhs(indices));
    }

    // template <my_size_t length>
    type evalu(my_size_t flat) const
    {
        return Op<T, Bits, Arch>::apply(_lhs.evalu(flat), _rhs.evalu(flat));
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

    inline bool getIsTransposed() const
    {
        return _lhs.getIsTransposed();
    }
};
