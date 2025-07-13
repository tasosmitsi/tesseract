#pragma once

// ===============================
// Base Expression Interface (CRTP)
// ===============================
template <typename Derived, typename T>
class BaseExpr
{
public:
    const Derived &derived() const
    {
        return static_cast<const Derived &>(*this);
    }
};
