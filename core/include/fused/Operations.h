#pragma once
#include <stdexcept>

// ===============================
// Operation Tags
// ===============================
template <typename T>
struct Add
{
    static T apply(T a, T b) { return a + b; }
};

template <typename T>
struct Sub
{
    static T apply(T a, T b) { return a - b; }
};

template <typename T>
struct Mul
{
    static T apply(T a, T b) { return a * b; }
};

template <typename T>
struct Div
{
    static T apply(T a, T b)
    {
        if (b == T(0))
            throw std::runtime_error("Division by zero");
        return a / b;
    }
};
