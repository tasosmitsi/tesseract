#pragma once

#include "../config.h"
#include "ops/op_traits.h"
#include "simple_type_traits.h"

// ===============================
// Operation Tags
// ===============================

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Add
{
    using OpTrait = OpTraits<T, Bits, Arch>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return OpTrait::add(a, b);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static type apply(Vec a, Scalar scalar)
    {
        return OpTrait::add(a, scalar);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static type apply(Scalar scalar, Vec a)
    {
        return OpTrait::add(a, scalar);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Sub
{
    using OpTrait = OpTraits<T, Bits, Arch>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return OpTrait::sub(a, b);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static type apply(Vec a, Scalar scalar)
    {
        return OpTrait::sub(a, scalar);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static type apply(T scalar, Vec a)
    {
        return OpTrait::sub(scalar, a);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Mul
{
    using OpTrait = OpTraits<T, Bits, Arch>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return OpTrait::mul(a, b);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static type apply(Vec a, Scalar scalar)
    {
        return OpTrait::mul(a, scalar);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static type apply(Scalar scalar, Vec a)
    {
        return OpTrait::mul(a, scalar);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Div
{
    using OpTrait = OpTraits<T, Bits, Arch>;
    using type = typename OpTrait::type; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return OpTrait::div(a, b);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static type apply(Vec a, Scalar scalar)
    {
        return OpTrait::div(a, scalar);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static type apply(Scalar scalar, Vec a)
    {
        return OpTrait::div(scalar, a);
    }
};
