#ifndef __AVX2_MICROKERNEL_H__
#define __AVX2_MICROKERNEL_H__

#include <immintrin.h>
#include "config.h"

// Architecture tag
struct X86_SSE
{
}; // 128-bit SSE/SSE2
struct X86_AVX
{
}; // 256-bit AVX/AVX2

// ============================================================================
// SSE (128-bit) specializations
// ============================================================================
template <>
struct Microkernel<float, 128, X86_SSE>
{
    static constexpr my_size_t simdWidth = 4; // 128 bits / 32 bits per float = 4
    using VecType = __m128;

    FORCE_INLINE static VecType load(const float *ptr) { return _mm_load_ps(ptr); }
    FORCE_INLINE static void store(float *ptr, VecType val) { _mm_store_ps(ptr, val); }
    FORCE_INLINE static VecType set1(float scalar) { return _mm_set1_ps(scalar); }
    FORCE_INLINE static VecType add(VecType a, VecType b) { return _mm_add_ps(a, b); }
    FORCE_INLINE static VecType mul(VecType a, VecType b) { return _mm_mul_ps(a, b); }
    FORCE_INLINE static VecType sub(VecType a, VecType b) { return _mm_sub_ps(a, b); }
    FORCE_INLINE static VecType div(VecType a, VecType b) { return _mm_div_ps(a, b); }
    FORCE_INLINE static void test() { std::cout << "Microkernel: float SSE128\n"; }
};

template <>
struct Microkernel<double, 128, X86_SSE>
{
    static constexpr my_size_t simdWidth = 2; // 128 bits / 64 bits per double = 2
    using VecType = __m128d;

    FORCE_INLINE static VecType load(const double *ptr) { return _mm_load_pd(ptr); }
    FORCE_INLINE static void store(double *ptr, VecType val) { _mm_store_pd(ptr, val); }
    FORCE_INLINE static VecType set1(double scalar) { return _mm_set1_pd(scalar); }
    FORCE_INLINE static VecType add(VecType a, VecType b) { return _mm_add_pd(a, b); }
    FORCE_INLINE static VecType mul(VecType a, VecType b) { return _mm_mul_pd(a, b); }
    FORCE_INLINE static VecType sub(VecType a, VecType b) { return _mm_sub_pd(a, b); }
    FORCE_INLINE static VecType div(VecType a, VecType b) { return _mm_div_pd(a, b); }
    FORCE_INLINE static void test() { std::cout << "Microkernel: double SSE128\n"; }
};

// ============================================================================
// AVX2 (256-bit) specializations
// ============================================================================

template <>
struct Microkernel<float, 256, X86_AVX>
{
    static constexpr my_size_t simdWidth = 8; // 256 bits / 32 bits per float = 8
    using VecType = __m256;

    FORCE_INLINE static VecType load(const float *ptr) { return _mm256_load_ps(ptr); }
    FORCE_INLINE static void store(float *ptr, VecType val) { _mm256_store_ps(ptr, val); }
    FORCE_INLINE static VecType set1(float scalar) { return _mm256_set1_ps(scalar); }
    FORCE_INLINE static VecType add(VecType a, VecType b) { return _mm256_add_ps(a, b); }
    FORCE_INLINE static VecType mul(VecType a, VecType b) { return _mm256_mul_ps(a, b); }
    FORCE_INLINE static VecType sub(VecType a, VecType b) { return _mm256_sub_ps(a, b); }
    FORCE_INLINE static VecType div(VecType a, VecType b) { return _mm256_div_ps(a, b); }
    // FMA operation (bonus!)
    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) { return _mm256_fmadd_ps(a, b, c); }
    FORCE_INLINE static void test() { std::cout << "Microkernel: float AVX256\n"; }
};

template <>
struct Microkernel<double, 256, X86_AVX>
{
    static constexpr my_size_t simdWidth = 4; // 256 bits / 64 bits per double = 4
    using VecType = __m256d;

    FORCE_INLINE static VecType load(const double *ptr) { return _mm256_load_pd(ptr); }
    FORCE_INLINE static void store(double *ptr, VecType val) { _mm256_store_pd(ptr, val); }
    FORCE_INLINE static VecType set1(double scalar) { return _mm256_set1_pd(scalar); }
    FORCE_INLINE static VecType add(VecType a, VecType b) { return _mm256_add_pd(a, b); }
    FORCE_INLINE static VecType mul(VecType a, VecType b) { return _mm256_mul_pd(a, b); }
    FORCE_INLINE static VecType sub(VecType a, VecType b) { return _mm256_sub_pd(a, b); }
    FORCE_INLINE static VecType div(VecType a, VecType b) { return _mm256_div_pd(a, b); }
    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) { return _mm256_fmadd_pd(a, b, c); }
    FORCE_INLINE static void test() { std::cout << "Microkernel: double AVX256\n"; }
};

#endif // __AVX2_MICROKERNEL_H__