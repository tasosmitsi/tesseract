#ifndef __AVX2_MICROKERNEL_H__
#define __AVX2_MICROKERNEL_H__

#include <immintrin.h>
#include "config.h"

// Architecture tag
struct X86_AVX
{
}; // 256-bit AVX/AVX2

// ============================================================================
// AVX2 (256-bit) specializations
// ============================================================================

template <>
struct Microkernel<float, 256, X86_AVX>
{
    static constexpr my_size_t simdWidth = 8; // 256 bits / 32 bits per float = 8
    using VecType = __m256;
    using ScalarType = float;

    FORCE_INLINE static VecType load(const ScalarType *ptr) noexcept { return _mm256_load_ps(ptr); }
    FORCE_INLINE static VecType loadu(const ScalarType *ptr) noexcept { return _mm256_loadu_ps(ptr); }
    FORCE_INLINE static void store(ScalarType *ptr, VecType val) noexcept { _mm256_store_ps(ptr, val); }
    FORCE_INLINE static void storeu(ScalarType *ptr, VecType val) noexcept { _mm256_storeu_ps(ptr, val); }
    FORCE_INLINE static VecType set1(ScalarType scalar) noexcept { return _mm256_set1_ps(scalar); }

    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return _mm256_add_ps(a, b); }
    FORCE_INLINE static VecType add(VecType a, ScalarType b) noexcept { return _mm256_add_ps(a, set1(b)); }

    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept { return _mm256_mul_ps(a, b); }
    FORCE_INLINE static VecType mul(VecType a, ScalarType b) noexcept { return _mm256_mul_ps(a, set1(b)); }

    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return _mm256_sub_ps(a, b); }
    FORCE_INLINE static VecType sub(VecType a, ScalarType b) noexcept { return _mm256_sub_ps(a, set1(b)); }
    FORCE_INLINE static VecType sub(ScalarType a, VecType b) noexcept { return _mm256_sub_ps(set1(a), b); }

    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept { return _mm256_div_ps(a, b); }
    FORCE_INLINE static VecType div(VecType a, ScalarType b) noexcept { return _mm256_div_ps(a, set1(b)); }
    FORCE_INLINE static VecType div(ScalarType a, VecType b) noexcept { return _mm256_div_ps(set1(a), b); }

    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return _mm256_fmadd_ps(a, b, c); }

    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept { return _mm256_min_ps(a, b); }
    FORCE_INLINE static VecType min(VecType a, ScalarType b) noexcept { return _mm256_min_ps(a, set1(b)); }

    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept { return _mm256_max_ps(a, b); }
    FORCE_INLINE static VecType max(VecType a, ScalarType b) noexcept { return _mm256_max_ps(a, set1(b)); }

    // ============================================================================
    // Gather: non-contiguous load using index list
    // ============================================================================
    FORCE_INLINE static VecType gather(const ScalarType *base, const my_size_t *indices) noexcept
    {
        // _mm256_i32gather_ps requires 8 × 32-bit indices.
        // so we convert size_t → int32_t.
        alignas(32) int32_t idx32[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
        {
            idx32[i] = static_cast<int32_t>(indices[i]);
        }

        // loadu (“unaligned load”) is recommended for temporary stack buffers, even when aligned, because:
        // it's just as fast as load on aligned addresses
        // never invokes undefined behavior
        // does not depend on type alignment rules
        // TODO: verify: Intel’s documentation confirms: On aligned addresses, _mm256_loadu_si256 performs identically to _mm256_load_si256.
        __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(idx32));
        return _mm256_i32gather_ps(base, vindex, sizeof(ScalarType));
    }

    FORCE_INLINE static void scatter(ScalarType *base, const my_size_t *indices, VecType val) noexcept
    {
        alignas(32) ScalarType tmp[simdWidth];
        _mm256_storeu_ps(tmp, val);
        for (my_size_t i = 0; i < simdWidth; ++i)
            base[indices[i]] = tmp[i];
    }
};

template <>
struct Microkernel<double, 256, X86_AVX>
{
    static constexpr my_size_t simdWidth = 4; // 256 bits / 64 bits per double = 4
    using VecType = __m256d;
    using ScalarType = double;

    FORCE_INLINE static VecType load(const ScalarType *ptr) noexcept { return _mm256_load_pd(ptr); }
    FORCE_INLINE static VecType loadu(const ScalarType *ptr) noexcept { return _mm256_loadu_pd(ptr); }
    FORCE_INLINE static void store(ScalarType *ptr, VecType val) noexcept { _mm256_store_pd(ptr, val); }
    FORCE_INLINE static void storeu(ScalarType *ptr, VecType val) noexcept { _mm256_storeu_pd(ptr, val); }
    FORCE_INLINE static VecType set1(ScalarType scalar) noexcept { return _mm256_set1_pd(scalar); }

    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return _mm256_add_pd(a, b); }
    FORCE_INLINE static VecType add(VecType a, ScalarType b) noexcept { return _mm256_add_pd(a, set1(b)); }

    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept { return _mm256_mul_pd(a, b); }
    FORCE_INLINE static VecType mul(VecType a, ScalarType b) noexcept { return _mm256_mul_pd(a, set1(b)); }

    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return _mm256_sub_pd(a, b); }
    FORCE_INLINE static VecType sub(VecType a, ScalarType b) noexcept { return _mm256_sub_pd(a, set1(b)); }
    FORCE_INLINE static VecType sub(ScalarType a, VecType b) noexcept { return _mm256_sub_pd(set1(a), b); }

    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept { return _mm256_div_pd(a, b); }
    FORCE_INLINE static VecType div(VecType a, ScalarType b) noexcept { return _mm256_div_pd(a, set1(b)); }
    FORCE_INLINE static VecType div(ScalarType a, VecType b) noexcept { return _mm256_div_pd(set1(a), b); }

    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return _mm256_fmadd_pd(a, b, c); }

    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept { return _mm256_min_pd(a, b); }
    FORCE_INLINE static VecType min(VecType a, ScalarType b) noexcept { return _mm256_min_pd(a, set1(b)); }

    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept { return _mm256_max_pd(a, b); }
    FORCE_INLINE static VecType max(VecType a, ScalarType b) noexcept { return _mm256_max_pd(a, set1(b)); }

    FORCE_INLINE static VecType gather(const ScalarType *base, const my_size_t *indices) noexcept
    {
        __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(indices));
        return _mm256_i64gather_pd(base, vindex, sizeof(ScalarType));
    }

    FORCE_INLINE static void scatter(ScalarType *base, const my_size_t *indices, VecType val) noexcept
    {
        alignas(32) ScalarType tmp[simdWidth];
        _mm256_storeu_pd(tmp, val);
        for (my_size_t i = 0; i < simdWidth; ++i)
            base[indices[i]] = tmp[i];
    }
};

#endif // __AVX2_MICROKERNEL_H__