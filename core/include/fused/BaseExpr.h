#pragma once

namespace detail
{
    struct BaseExprTag
    {
    };
}

// ===============================
// Base Expression Interface (CRTP)
// ===============================
template <typename Derived>
class BaseExpr : public detail::BaseExprTag
{
public:
    const Derived &derived() const
    {
        return static_cast<const Derived &>(*this);
    }
};
