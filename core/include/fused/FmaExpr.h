#pragma once
#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/Operations.h"
#include "helper_traits.h"
#include "simple_type_traits.h"

// ===============================
// FMA Expression Template (A * B ± C)
// ===============================
template <
    typename A,
    typename B, typename C,
    template <typename, my_size_t, typename> class Op>
class FmaExpr : public BaseExpr<FmaExpr<A, B, C, Op>>
{
    static_assert(is_same_v<typename A::value_type, typename B::value_type>,
                  "FmaExpr: A and B must have the same value_type");
    static_assert(is_same_v<typename A::value_type, typename C::value_type>,
                  "FmaExpr: A and C must have the same value_type");

#ifdef COMPILETIME_CHECK_DIMENSIONS_COUNT_MISMATCH
    static_assert(A::NumDims == B::NumDims && A::NumDims == C::NumDims,
                  "FmaExpr: number of dimensions mismatch");
#endif
#ifdef COMPILETIME_CHECK_DIMENSIONS_SIZE_MISMATCH
    static_assert(dims_match<A::NumDims>(A::Dim, B::Dim),
                  "FmaExpr: dimension mismatch between A and B");
    static_assert(dims_match<A::NumDims>(A::Dim, C::Dim),
                  "FmaExpr: dimension mismatch between A and C");
#endif

    const A &_a;
    const B &_b;
    const C &_c;

public:
    static constexpr my_size_t NumDims = A::NumDims;
    static constexpr const my_size_t *Dim = A::Dim;
    static constexpr my_size_t TotalSize = A::TotalSize;
    using value_type = typename A::value_type;
    using Layout = typename A::Layout;

    FmaExpr(const A &a, const B &b, const C &c) : _a(a), _b(b), _c(c) {}

    const A &lhs() const noexcept { return _a; }
    const B &rhs() const noexcept { return _b; }
    const C &addend() const noexcept { return _c; }

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return _a.may_alias(output) || _b.may_alias(output) || _c.may_alias(output);
    }

    template <my_size_t length>
    inline auto operator()(my_size_t (&indices)[length]) const noexcept
    {
        using T = std::decay_t<decltype(_a(indices))>;
        return Op<T, 0, GENERICARCH>::apply(
            _a(indices), _b(indices), _c(indices));
    }

    template <typename T, my_size_t Bits, typename Arch>
    inline typename Op<T, Bits, Arch>::type evalu(my_size_t flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(
            _a.template evalu<T, Bits, Arch>(flat),
            _b.template evalu<T, Bits, Arch>(flat),
            _c.template evalu<T, Bits, Arch>(flat));
    }

    template <typename T, my_size_t Bits, typename Arch>
    inline typename Op<T, Bits, Arch>::type logical_evalu(my_size_t logical_flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(
            _a.template logical_evalu<T, Bits, Arch>(logical_flat),
            _b.template logical_evalu<T, Bits, Arch>(logical_flat),
            _c.template logical_evalu<T, Bits, Arch>(logical_flat));
    }

    inline my_size_t getNumDims() const noexcept { return _a.getNumDims(); }
    inline my_size_t getDim(my_size_t i) const { return _a.getDim(i); }
    my_size_t getTotalSize() const noexcept { return _a.getTotalSize(); }

protected:
    inline auto operator()(const my_size_t *indices) const noexcept
    {
        using T = std::decay_t<decltype(_a(indices))>;
        return Op<T, 0, GENERICARCH>::apply(
            _a(indices), _b(indices), _c(indices));
    }
};

// ===============================
// Scalar FMA Expression Template (A * scalar ± C)
// ===============================
template <
    typename EXPR,
    typename ScalarT, typename C,
    template <typename, my_size_t, typename> class Op>
class ScalarFmaExpr : public BaseExpr<ScalarFmaExpr<EXPR, ScalarT, C, Op>>
{
    static_assert(is_same_v<typename EXPR::value_type, ScalarT>,
                  "ScalarFmaExpr: EXPR value_type and ScalarT must be the same");
    static_assert(is_same_v<typename EXPR::value_type, typename C::value_type>,
                  "ScalarFmaExpr: EXPR and C must have the same value_type");

#ifdef COMPILETIME_CHECK_DIMENSIONS_COUNT_MISMATCH
    static_assert(EXPR::NumDims == C::NumDims,
                  "ScalarFmaExpr: number of dimensions mismatch");
#endif
#ifdef COMPILETIME_CHECK_DIMENSIONS_SIZE_MISMATCH
    static_assert(dims_match<EXPR::NumDims>(EXPR::Dim, C::Dim),
                  "ScalarFmaExpr: dimension mismatch between EXPR and C");
#endif

    const EXPR &_expr;
    ScalarT _scalar;
    const C &_c;

public:
    static constexpr my_size_t NumDims = EXPR::NumDims;
    static constexpr const my_size_t *Dim = EXPR::Dim;
    static constexpr my_size_t TotalSize = EXPR::TotalSize;
    using value_type = typename EXPR::value_type;
    using Layout = typename EXPR::Layout;

    ScalarFmaExpr(const EXPR &expr, ScalarT scalar, const C &c)
        : _expr(expr), _scalar(scalar), _c(c) {}

    const EXPR &expr() const noexcept { return _expr; }
    ScalarT scalar() const noexcept { return _scalar; }
    const C &addend() const noexcept { return _c; }

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return _expr.may_alias(output) || _c.may_alias(output);
    }

    template <my_size_t length>
    inline auto operator()(my_size_t (&indices)[length]) const noexcept
    {
        using T = std::decay_t<decltype(_expr(indices))>;
        return Op<T, 0, GENERICARCH>::apply(
            _expr(indices), _scalar, _c(indices));
    }

    template <typename T, my_size_t Bits, typename Arch>
    inline typename Op<T, Bits, Arch>::type evalu(my_size_t flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(
            _expr.template evalu<T, Bits, Arch>(flat),
            _scalar,
            _c.template evalu<T, Bits, Arch>(flat));
    }

    template <typename T, my_size_t Bits, typename Arch>
    inline typename Op<T, Bits, Arch>::type logical_evalu(my_size_t logical_flat) const noexcept
    {
        return Op<T, Bits, Arch>::apply(
            _expr.template logical_evalu<T, Bits, Arch>(logical_flat),
            _scalar,
            _c.template logical_evalu<T, Bits, Arch>(logical_flat));
    }

    inline my_size_t getNumDims() const noexcept { return _expr.getNumDims(); }
    inline my_size_t getDim(my_size_t i) const { return _expr.getDim(i); }
    my_size_t getTotalSize() const noexcept { return _expr.getTotalSize(); }

protected:
    inline auto operator()(const my_size_t *indices) const noexcept
    {
        using T = std::decay_t<decltype(_expr(indices))>;
        return Op<T, 0, GENERICARCH>::apply(
            _expr(indices), _scalar, _c(indices));
    }
};