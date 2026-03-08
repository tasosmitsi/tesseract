// Base interface and architecture tags
#ifndef MICROKERNEL_BASE_H
#define MICROKERNEL_BASE_H

#include "config.h"
#include "simple_type_traits.h"

// Base microkernel interface - all architecture-specific kernels implement this
// Template parameters:
//   T    = scalar type (float, double, int, Complex<float>, etc.)
//   Bits = SIMD register width in bits (128, 256, 512, etc.)
//   Arch = architecture tag (X86_AVX, NEONArch, etc.)

template <typename T, my_size_t Bits, typename Arch>
struct Microkernel
{
    static constexpr my_size_t simdWidth = 1; // Override in specializations
    using VecType = T;                        // Override with architecture-specific vector type

    // Core SIMD operations
    // Memory operations
    FORCE_INLINE static VecType load(const T *ptr) noexcept;
    FORCE_INLINE static void store(T *ptr, VecType val) noexcept;

    // Broadcast scalar to vector (CRITICAL for tensor-scalar ops)
    FORCE_INLINE static VecType set1(T scalar) noexcept;

    // Vector-vector operations
    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept;
    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept;
    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept;
    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept;
    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept;
    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept;

    // // Vector-scalar operations (using set1 internally)
    // template <typename Vec = VecType, typename Scalar = T>
    //     requires(!is_same_v<Vec, Scalar>)
    // FORCE_INLINE static Vec add(Vec a, T scalar) noexcept
    // {
    //     return add(a, set1(scalar));
    // }
    // template <typename Vec = VecType, typename Scalar = T>
    //     requires(!is_same_v<Vec, Scalar>)
    // FORCE_INLINE static Vec mul(Vec a, T scalar) noexcept
    // {
    //     return mul(a, set1(scalar));
    // }
    // template <typename Vec = VecType, typename Scalar = T>
    //     requires(!is_same_v<Vec, Scalar>)
    // FORCE_INLINE static Vec sub(Vec a, T scalar) noexcept
    // {
    //     return sub(a, set1(scalar));
    // }
    // template <typename Vec = VecType, typename Scalar = T>
    //     requires(!is_same_v<Vec, Scalar>)
    // FORCE_INLINE static Vec div(Vec a, T scalar) noexcept
    // {
    //     return div(a, set1(scalar));
    // }

    // // Scalar-vector operations (order matters for sub/div!)
    // template <typename Vec = VecType, typename Scalar = T>
    //     requires(!is_same_v<Vec, Scalar>)
    // FORCE_INLINE static Vec sub(T scalar, Vec a) noexcept
    // {
    //     return sub(set1(scalar), a);
    // }

    // template <typename Vec = VecType, typename Scalar = T>
    //     requires(!is_same_v<Vec, Scalar>)
    // FORCE_INLINE static Vec div(T scalar, Vec a) noexcept
    // {
    //     return div(set1(scalar), a);
    // }
};

// ============================================================================
// Architecture Tags
// ============================================================================
// struct X86_SSE {};       // 128-bit SSE/SSE2
// struct X86_AVX {};       // 256-bit AVX/AVX2
// struct X86_AVX512 {};    // 512-bit AVX-512
// struct ARM_NEON_A76 {};  // 128-bit NEON on Cortex-A76 and newer (RPi5, Graviton)
// struct ARM_NEON_A72 {};  // 128-bit NEON on Cortex-A72 and newer (RPi4)
// struct ARM_NEON_A55 {};  // 128-bit NEON on Cortex-A55 and older (many ARMv8 phones)

// Include all architecture implementations
#include "fused/microkernels/generic/generic_microkernel.h"
// #include "fused/microkernels/generic/generic_complex_microkernel.h"

#if __AVX512F__
#include "fused/microkernels/avx2/avx512_microkernel.h"
// #include "fused/microkernels/avx2/avx512_complex_microkernel.h"
#pragma message "[COMPILE-TIME] Using X86_AVX512F arch"
constexpr my_size_t BITS = 512;
using DefaultArch = X86_AVX512;

#elif __AVX2__
#include "fused/microkernels/avx2/avx2_microkernel.h"
// #include "fused/microkernels/avx2/avx2_complex_microkernel.h"
#pragma message "[COMPILE-TIME] Using X86_AVX arch"
constexpr my_size_t BITS = 256;
using DefaultArch = X86_AVX;

#elif __SSE2__
#include "fused/microkernels/sse2/sse2_microkernel.h"
// #include "fused/microkernels/avx2/sse2_complex_microkernel.h"
#pragma message "[COMPILE-TIME] Using X86_SSE2 arch"
constexpr my_size_t BITS = 128;
using DefaultArch = X86_SSE;

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include "fused/microkernels/neon/neon_microkernel.h"
constexpr my_size_t BITS = 128;

    // User override takes priority
    #if defined(TESSERACT_ARM_UARCH_A76)
        #pragma message "[COMPILE-TIME] Using ARM_NEON_A76 arch (user override)"
        using DefaultArch = ARM_NEON_A76;
    #elif defined(TESSERACT_ARM_UARCH_A72)
        #pragma message "[COMPILE-TIME] Using ARM_NEON_A72 arch (user override)"
        using DefaultArch = ARM_NEON_A72;
    #elif defined(TESSERACT_ARM_UARCH_A55)
        #pragma message "[COMPILE-TIME] Using ARM_NEON_A55 arch (user override)"
        using DefaultArch = ARM_NEON_A55;
    // Auto-detection fallback
    #elif defined(__ARM_FEATURE_DOTPROD)
        #pragma message "[COMPILE-TIME] Using ARM_NEON_A76 arch (auto-detected)"
        using DefaultArch = ARM_NEON_A76;
    #elif defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
        #pragma message "[COMPILE-TIME] Using ARM_NEON_A72 arch (auto-detected)"
        using DefaultArch = ARM_NEON_A72;
    #else
        #pragma message "[COMPILE-TIME] Using ARM_NEON_A55 arch (auto-detected)"
        using DefaultArch = ARM_NEON_A55;
    #endif

#else
#pragma message "[COMPILE-TIME] Using GENERICARCH arch"
constexpr my_size_t BITS = 0;
using DefaultArch = GENERICARCH;
#endif

constexpr my_size_t DATA_ALIGNAS = BITS / 8;

#endif // MICROKERNEL_BASE_H