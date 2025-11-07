#pragma once

#include "config.h"
#include "simple_type_traits.h"
#include "fused/microkernels/microkernel_base.h"

// ===============================
// Operation Tags
// ===============================

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Add
{
    using  microkernel = Microkernel<T, Bits, Arch>;
    using type = typename  microkernel::VecType; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return  microkernel::add(a, b);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec apply(Vec a, Scalar scalar)
    {
        return  microkernel::add(a, scalar);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec apply(Scalar scalar, Vec a)
    {
        return  microkernel::add(a, scalar);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Sub
{
    using  microkernel = Microkernel<T, Bits, Arch>;
    using type = typename  microkernel::VecType; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return  microkernel::sub(a, b);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec apply(Vec a, Scalar scalar)
    {
        return  microkernel::sub(a, scalar);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec apply(T scalar, Vec a)
    {
        return  microkernel::sub(scalar, a);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Mul
{
    using  microkernel = Microkernel<T, Bits, Arch>;
    using type = typename  microkernel::VecType; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return  microkernel::mul(a, b);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec apply(Vec a, Scalar scalar)
    {
        return  microkernel::mul(a, scalar);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec apply(Scalar scalar, Vec a)
    {
        return  microkernel::mul(a, scalar);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Div
{
    using  microkernel = Microkernel<T, Bits, Arch>;
    using type = typename  microkernel::VecType; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b)
    {
        return  microkernel::div(a, b);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec apply(Vec a, Scalar scalar)
    {
        return  microkernel::div(a, scalar);
    }

    template <typename Vec = type, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec apply(Scalar scalar, Vec a)
    {
        return  microkernel::div(scalar, a);
    }
};
