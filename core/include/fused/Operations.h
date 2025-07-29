#pragma once

// ===============================
// Operation Tags
// ===============================
// template <typename T>
// struct Add
// {
//     static T apply(T a, T b) { return a + b; }
// };

// template <typename T>
// struct Sub
// {
//     static T apply(T a, T b) { return a - b; }
// };

// template <typename T>
// struct Mul
// {
//     static T apply(T a, T b) { return a * b; }
// };

// template <typename T>
// struct Div
// {
//     static T apply(T a, T b)
//     {
//         if (b == T(0))
//             MyErrorHandler::error("Division by zero");
//         return a / b;
//     }
// };

#include "../config.h"
#include "ops/op_traits.h"

template <typename T, typename Arch = DefaultArch>
struct Add {
    static T apply(T a, T b) {
        return OpTraits<T, Arch>::add(a, b);
    }
};

template <typename T, typename Arch = DefaultArch>
struct Sub {
    static T apply(T a, T b) {
        return OpTraits<T, Arch>::sub(a, b);
    }
};

template <typename T, typename Arch = DefaultArch>
struct Mul {
    static T apply(T a, T b) {
        return OpTraits<T, Arch>::mul(a, b);
    }
};

template <typename T, typename Arch = DefaultArch>
struct Div {
    static T apply(T a, T b) {
        return OpTraits<T, Arch>::div(a, b);
    }
};
