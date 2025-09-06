#pragma once

#include "../config.h"
#include "ops/op_traits.h"

// ===============================
// Operation Tags
// ===============================

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Add
{
    using OpTrait = OpTraits<T, Bits, Arch>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static T apply(T a, T b)
    {
        return OpTrait::add(a, b);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Sub
{
    using OpTrait = OpTraits<T, Bits, Arch>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static T apply(T a, T b)
    {
        return OpTrait::sub(a, b);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Mul
{
    using OpTrait = OpTraits<T, Bits, Arch>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static T apply(T a, T b)
    {
        return OpTrait::mul(a, b);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Div
{
    using OpTrait = OpTraits<T, Bits, Arch>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static T apply(T a, T b)
    {
        return OpTrait::div(a, b);
    }
};
