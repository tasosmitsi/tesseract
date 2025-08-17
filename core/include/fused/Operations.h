#pragma once

#include "../config.h"
#include "ops/op_traits.h"

// ===============================
// Operation Tags
// ===============================

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Add
{
    FORCE_INLINE static T apply(T a, T b)
    {
        return OpTraits<T, Bits, Arch>::add(a, b);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Sub
{
    FORCE_INLINE static T apply(T a, T b)
    {
        return OpTraits<T, Bits, Arch>::sub(a, b);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Mul
{
    FORCE_INLINE static T apply(T a, T b)
    {
        return OpTraits<T, Bits, Arch>::mul(a, b);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Div
{
    FORCE_INLINE static T apply(T a, T b)
    {
        return OpTraits<T, Bits, Arch>::div(a, b);
    }
};

// Specialization for BITS and X86_AVX
template <typename T>
struct Add<T, BITS, X86_AVX>
{
    using OpTrait = OpTraits<T, BITS, X86_AVX>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return OpTrait::add(a, b);
    }

    FORCE_INLINE static type apply(type a, T scalar)
    {
        return OpTrait::add(a, scalar);
    }

    FORCE_INLINE static type apply(T scalar, type a)
    {
        return OpTrait::add(a, scalar);
    }
};

template <typename T>
struct Sub<T, BITS, X86_AVX>
{
    using OpTrait = OpTraits<T, BITS, X86_AVX>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return OpTrait::sub(a, b);
    }

    FORCE_INLINE static type apply(type a, T scalar)
    {
        return OpTrait::sub(a, scalar);
    }

    FORCE_INLINE static type apply(T scalar, type a)
    {
        return OpTrait::sub(scalar, a);
    }
};

template <typename T>
struct Mul<T, BITS, X86_AVX>
{
    using OpTrait = OpTraits<T, BITS, X86_AVX>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return OpTrait::mul(a, b);
    }

    FORCE_INLINE static type apply(type a, T scalar)
    {
        return OpTrait::mul(a, scalar);
    }

    FORCE_INLINE static type apply(T scalar, type a)
    {
        return OpTrait::mul(a, scalar);
    }
};

template <typename T>
struct Div<T, BITS, X86_AVX>
{
    using OpTrait = OpTraits<T, BITS, X86_AVX>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return OpTrait::div(a, b);
    }

    FORCE_INLINE static type apply(type a, T scalar)
    {
        return OpTrait::div(a, scalar);
    }

    FORCE_INLINE static type apply(T scalar, type a)
    {
        return OpTrait::div(scalar, a);
    }
};
