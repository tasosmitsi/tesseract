#ifndef GENERIC_MICROKERNEL_H
#define GENERIC_MICROKERNEL_H

#include "config.h"

// Generic microkernel for ANY type and ANY bit width (scalar fallback)
// This is a partial specialization that matches any T and any Bits with GENERICARCH
struct GENERICARCH
{
}; // Scalar fallback

template <typename T, my_size_t Bits>
struct Microkernel<T, Bits, GENERICARCH>
{
    static constexpr my_size_t simdWidth = 1;
    using VecType = T;
    using ScalarType = T; // In scalar mode, VecType is the same as ScalarType

    FORCE_INLINE static VecType load(const T *ptr) noexcept { return *ptr; }
    FORCE_INLINE static VecType loadu(const T *ptr) noexcept { return *ptr; }

    FORCE_INLINE static void store(T *ptr, VecType val) noexcept { *ptr = val; }
    FORCE_INLINE static void storeu(T *ptr, VecType val) noexcept { *ptr = val; }

    FORCE_INLINE static VecType set1(T scalar) noexcept { return scalar; } // In scalar mode, set1 is identity}
    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return a + b; }
    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept { return a * b; }
    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return a - b; }
    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept { return a / b; }
    
    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept { return a < b ? a : b; }
    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept { return a > b ? a : b; }

    FORCE_INLINE static VecType gather(const T *base, const my_size_t *indices) noexcept { return base[indices[0]]; }
    FORCE_INLINE static void scatter(T *base, const my_size_t *indices, VecType val) noexcept { base[indices[0]] = val; }

    FORCE_INLINE static VecType abs(VecType v) noexcept { return v < T{0} ? -v : v; }
    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, T tol) noexcept
    {
        T diff = a - b;
        return abs(diff) <= tol;
    }
};

#endif // GENERIC_MICROKERNEL_H