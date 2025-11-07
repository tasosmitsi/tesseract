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
    using VecType = T; // Override with architecture-specific vector type

    // Core SIMD operations
    // Memory operations
    FORCE_INLINE static VecType load(const T *ptr);
    FORCE_INLINE static void store(T *ptr, VecType val);

    // Broadcast scalar to vector (CRITICAL for tensor-scalar ops)
    FORCE_INLINE static VecType set1(T scalar);

    // Vector-vector operations
    FORCE_INLINE static VecType add(VecType a, VecType b);
    FORCE_INLINE static VecType mul(VecType a, VecType b);
    FORCE_INLINE static VecType sub(VecType a, VecType b);
    FORCE_INLINE static VecType div(VecType a, VecType b);

    // Vector-scalar operations (using set1 internally)
    template <typename Vec = VecType, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec add(Vec a, T scalar) { return add(a, set1(scalar)); }
    template <typename Vec = VecType, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec mul(Vec a, T scalar) { return mul(a, set1(scalar)); }
    template <typename Vec = VecType, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec sub(Vec a, T scalar) { return sub(a, set1(scalar)); }
    template <typename Vec = VecType, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec div(Vec a, T scalar) { return div(a, set1(scalar)); }

    // Scalar-vector operations (order matters for sub/div!)
    template <typename Vec = VecType, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec sub(T scalar, Vec a) { return sub(set1(scalar), a); }

    template <typename Vec = VecType, typename Scalar = T>
        requires(!is_same_v<Vec, Scalar>)
    FORCE_INLINE static Vec div(T scalar, Vec a) { return div(set1(scalar), a); }
};

// ============================================================================
// Architecture Tags
// ============================================================================
// struct X86_SSE {};       // 128-bit SSE/SSE2
// struct X86_AVX {};       // 256-bit AVX/AVX2  
// struct X86_AVX512 {};    // 512-bit AVX-512
// struct NEONArch {};      // 128-bit ARM NEON
// struct SVEArch {};       // Scalable ARM SVE

// Include all architecture implementations
#include "fused/microkernels/generic/generic_microkernel.h"
// #include "fused/microkernels/generic/generic_complex_microkernel.h"

#ifdef __AVX2__
#include "fused/microkernels/avx2/avx2_microkernel.h"
// #include "fused/microkernels/avx2/avx2_complex_microkernel.h"
#pragma message "[COMPILE-TIME] Using X86_AVX arch"
#define BITS 256 // 128 or 256 bits
using DefaultArch = X86_AVX;
#endif

// #ifdef __AVX512F__
// #include "avx512/avx512_microkernel.h"
// // #include "avx512/avx512_complex_microkernel.h"
// #endif

// #ifdef __ARM_NEON
// #include "neon/neon_microkernel.h"
// // #include "neon/neon_complex_microkernel.h"
// #endif


// Add more architecture-specific includes here
// #if defined(__ARM_NEON)
// using DefaultArch = ARM_NEON;
// #elif defined(__AVX__)
// // TODO: use this instead constexpr my_size_t AVX_BITS = 256;
// // #include "op_traits_f_X86_AVX.h"
// #else
// using DefaultArch = GenericArch;
// #endif


#endif // MICROKERNEL_BASE_H