#ifndef __NEON_MICROKERNEL_H__
#define __NEON_MICROKERNEL_H__

#include "neon_intrinsics.h"

// ============================================================================
// Architecture tags
// ============================================================================
struct ARM_NEON_A55
{
};
struct ARM_NEON_A72
{
};
struct ARM_NEON_A76
{
};

// ============================================================================
// Per-microarch specializations — only tiling differs
// ============================================================================

// --- A55 (in-order, narrow) ---
template <>
struct Microkernel<float, 128, ARM_NEON_A55> : NeonFloatIntrinsics
{
    static constexpr my_size_t MR = 4;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 12
};

template <>
struct Microkernel<double, 128, ARM_NEON_A55> : NeonDoubleIntrinsics
{
    static constexpr my_size_t MR = 4;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 6
};

// --- A72 (RPi4) ---
template <>
struct Microkernel<float, 128, ARM_NEON_A72> : NeonFloatIntrinsics
{
    static constexpr my_size_t MR = 8;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 12
};

template <>
struct Microkernel<double, 128, ARM_NEON_A72> : NeonDoubleIntrinsics
{
    static constexpr my_size_t MR = 8;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 6
};

// --- A76+ (RPi5, Graviton) ---
template <>
struct Microkernel<float, 128, ARM_NEON_A76> : NeonFloatIntrinsics
{
    static constexpr my_size_t MR = 8;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 12
};

template <>
struct Microkernel<double, 128, ARM_NEON_A76> : NeonDoubleIntrinsics
{
    static constexpr my_size_t MR = 8;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 6
};

#endif // __NEON_MICROKERNEL_H__