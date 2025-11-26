#ifndef GENERIC_MICROKERNEL_H
#define GENERIC_MICROKERNEL_H

#include "config.h"

// Generic microkernel for ANY type and ANY bit width (scalar fallback)
// This is a partial specialization that matches any T and any Bits with GenericArch
struct GenericArch
{
}; // Scalar fallback

template <typename T, my_size_t Bits>
struct Microkernel<T, Bits, GenericArch>
{
    static constexpr my_size_t simdWidth = 1;
    using VecType = T;

    FORCE_INLINE static VecType load(const T *ptr) { return *ptr; }
    FORCE_INLINE static void store(T *ptr, VecType val) { *ptr = val; }
    FORCE_INLINE static VecType set1(T scalar) { return scalar; } // In scalar mode, set1 is identity}
    FORCE_INLINE static VecType add(VecType a, VecType b) { return a + b; }
    FORCE_INLINE static VecType mul(VecType a, VecType b) { return a * b; }
    FORCE_INLINE static VecType sub(VecType a, VecType b) { return a - b; }
    FORCE_INLINE static VecType div(VecType a, VecType b) { return a / b; }

    FORCE_INLINE static void test() { std::cout << "Generic Microkernel: scalar fallback for type " << typeid(T).name() << " with bit width " << Bits << "\n"; }
};

#endif // GENERIC_MICROKERNEL_H