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
    using microkernel = Microkernel<T, Bits, Arch>;
    using type = typename microkernel::VecType; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b) noexcept
    {
        // print the types of a and b
        return microkernel::add(a, b);
    }

    template <typename Vec = type>
        requires(!is_same_v<Vec, T>)
    FORCE_INLINE static Vec apply(Vec a, T scalar) noexcept
    {
        return microkernel::add(a, scalar);
    }

    // template <typename Vec = type>
    //     requires(!is_same_v<Vec, T>)
    // FORCE_INLINE static Vec apply(T scalar, Vec a) noexcept
    // {
    //     std::cout << " lol Add::apply Scalar-Vec types: " << typeid(scalar).name() << ", " << typeid(a).name() << std::endl;
    //     return microkernel::add(scalar, a);
    // }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Sub
{
    using microkernel = Microkernel<T, Bits, Arch>;
    using type = typename microkernel::VecType; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b) noexcept
    {
        return microkernel::sub(a, b);
    }

    template <typename Vec = type>
        requires(!is_same_v<Vec, T>)
    FORCE_INLINE static Vec apply(Vec a, T scalar) noexcept
    {
        return microkernel::sub(a, scalar);
    }

    template <typename Vec = type>
        requires(!is_same_v<Vec, T>)
    FORCE_INLINE static Vec apply(T scalar, Vec a) noexcept
    {
        return microkernel::sub(scalar, a);
    }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Mul
{
    using microkernel = Microkernel<T, Bits, Arch>;
    using type = typename microkernel::VecType; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b) noexcept
    {
        return microkernel::mul(a, b);
    }

    template <typename Vec = type>
        requires(!is_same_v<Vec, T>)
    FORCE_INLINE static Vec apply(Vec a, T scalar) noexcept
    {
        return microkernel::mul(a, scalar);
    }

    // template <typename Vec = type>
    //     requires(!is_same_v<Vec, T>)
    // FORCE_INLINE static Vec apply(T scalar, Vec a) noexcept
    // {
    //     std::cout << "Mul::apply Scalar-Vec types: " << typeid(scalar).name() << ", " << typeid(a).name() << std::endl;
    //     return microkernel::mul(a, scalar);
    // }
};

template <typename T, my_size_t Bits, typename Arch = DefaultArch>
struct Div
{
    using microkernel = Microkernel<T, Bits, Arch>;
    using type = typename microkernel::VecType; // alias for easier usage

    FORCE_INLINE static type apply(type a, type b) noexcept
    {
        return microkernel::div(a, b);
    }

    template <typename Vec = type>
        requires(!is_same_v<Vec, T>)
    FORCE_INLINE static Vec apply(Vec a, T scalar) noexcept
    {
        return microkernel::div(a, scalar);
    }

    template <typename Vec = type>
        requires(!is_same_v<Vec, T>)
    FORCE_INLINE static Vec apply(T scalar, Vec a) noexcept
    {
        return microkernel::div(scalar, a);
    }
};
