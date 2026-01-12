#pragma once
#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/Operations.h"
#include "helper_traits.h"
#include "simple_type_traits.h"

// ===============================
// Binary Expression Template
// ===============================
template <
    typename LHS, typename RHS,
    template <typename, my_size_t, typename> class Op>
class BinaryExpr : public BaseExpr<BinaryExpr<LHS, RHS, Op>>
{
    // Compile-time check that LHS and RHS have the same value_type
    static_assert(is_same_v<typename LHS::value_type, typename RHS::value_type>,
                  "BinaryExpr: LHS and RHS must have the same value_type");

#ifdef COMPILETIME_CHECK_DIMENSIONS_COUNT_MISMATCH
    // Compile-time check that both expressions have the same number of dimensions
    static_assert(LHS::NumDims == RHS::NumDims,
                  "BinaryExpr: number of dimensions mismatch");
#endif
#ifdef COMPILETIME_CHECK_DIMENSIONS_SIZE_MISMATCH
    // Compile-time check that both expressions have the same dimensions
    static_assert(dims_match<LHS::NumDims>(LHS::Dim, RHS::Dim),
                  "BinaryExpr: there is at least one dimension mismatch");
#endif
// TODO: runtime checks should be here and not in the operators (in this constructor?)
// Or not, because in the case of the permuted views? We don't want checks there...

    const LHS &_lhs;
    const RHS &_rhs;
    using type = typename Op<T, Bits, Arch>::type; // alias for easier usage
public:
    // Expose compile-time shape constants
    static constexpr my_size_t NumDims = LHS::NumDims;
    static constexpr const my_size_t *Dim = LHS::Dim;
    static constexpr my_size_t TotalSize = LHS::TotalSize;

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
