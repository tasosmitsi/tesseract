#ifndef __SSE2_MICROKERNEL_H__
#define __SSE2_MICROKERNEL_H__

#include <immintrin.h>
#include "config.h"

// Architecture tag
struct X86_SSE
{
}; // 128-bit SSE/SSE2

// ============================================================================
// SSE (128-bit) specializations
// ============================================================================

template <>
struct Microkernel<float, 128, X86_SSE>
{
    static constexpr my_size_t simdWidth = 4; // 128 bits / 32 bits per float = 4
    using VecType = __m128;
    using ScalarType = float;

    FORCE_INLINE static VecType load(const ScalarType *ptr) noexcept { return _mm_load_ps(ptr); }
    FORCE_INLINE static VecType loadu(const ScalarType *ptr) noexcept { return _mm_loadu_ps(ptr); }
    FORCE_INLINE static void store(ScalarType *ptr, VecType val) noexcept { _mm_store_ps(ptr, val); }
    FORCE_INLINE static void storeu(ScalarType *ptr, VecType val) noexcept { _mm_storeu_ps(ptr, val); }
    FORCE_INLINE static VecType set1(ScalarType scalar) noexcept { return _mm_set1_ps(scalar); }

    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return _mm_add_ps(a, b); }
    FORCE_INLINE static VecType add(VecType a, ScalarType b) noexcept { return _mm_add_ps(a, set1(b)); }

    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept { return _mm_mul_ps(a, b); }
    FORCE_INLINE static VecType mul(VecType a, ScalarType b) noexcept { return _mm_mul_ps(a, set1(b)); }

    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return _mm_sub_ps(a, b); }
    FORCE_INLINE static VecType sub(VecType a, ScalarType b) noexcept { return _mm_sub_ps(a, set1(b)); }
    FORCE_INLINE static VecType sub(ScalarType a, VecType b) noexcept { return _mm_sub_ps(set1(a), b); }

    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept { return _mm_div_ps(a, b); }
    FORCE_INLINE static VecType div(VecType a, ScalarType b) noexcept { return _mm_div_ps(a, set1(b)); }
    FORCE_INLINE static VecType div(ScalarType a, VecType b) noexcept { return _mm_div_ps(set1(a), b); }

    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return _mm_fmadd_ps(a, b, c); }

    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept { return _mm_min_ps(a, b); }
    FORCE_INLINE static VecType min(VecType a, ScalarType b) noexcept { return _mm_min_ps(a, set1(b)); }

    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept { return _mm_max_ps(a, b); }
    FORCE_INLINE static VecType max(VecType a, ScalarType b) noexcept { return _mm_max_ps(a, set1(b)); }

    FORCE_INLINE static VecType gather(const ScalarType *base, const my_size_t *indices) noexcept
    {
        // _mm_i32gather_ps requires 4 × 32-bit indices.
        // so we convert size_t → int32_t.
        alignas(16) int32_t idx32[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
        {
            idx32[i] = static_cast<int32_t>(indices[i]);
        }

        // loadu (“unaligned load”) is recommended for temporary stack buffers, even when aligned, because:
        // it's just as fast as load on aligned addresses
        // never invokes undefined behavior
        // does not depend on type alignment rules
        __m128i vindex = _mm_loadu_si128(reinterpret_cast<const __m128i *>(idx32));
        return _mm_i32gather_ps(base, vindex, sizeof(ScalarType));
    }

    FORCE_INLINE static void scatter(ScalarType *base, const my_size_t *indices, VecType val) noexcept
    {
        alignas(16) ScalarType tmp[simdWidth];
        _mm_storeu_ps(tmp, val);
        for (my_size_t i = 0; i < simdWidth; ++i)
            base[indices[i]] = tmp[i];
    }

    FORCE_INLINE static VecType abs(VecType v) noexcept
    {
        __m128 sign_mask = _mm_set1_ps(-0.0f);
        return _mm_andnot_ps(sign_mask, v);
    }

    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, ScalarType tol) noexcept
    {
        __m128 diff = _mm_sub_ps(a, b);
        __m128 abs_diff = abs(diff);
        __m128 tol_vec = _mm_set1_ps(tol);
        __m128 cmp = _mm_cmple_ps(abs_diff, tol_vec);
        int mask = _mm_movemask_ps(cmp);
        return mask == 0xF; // all 4 lanes passed
    }
};

template <>
struct Microkernel<double, 128, X86_SSE>
{
    static constexpr my_size_t simdWidth = 2; // 128 bits / 64 bits per double = 2
    using VecType = __m128d;
    using ScalarType = double;

    FORCE_INLINE static VecType load(const ScalarType *ptr) noexcept { return _mm_load_pd(ptr); }
    FORCE_INLINE static VecType loadu(const ScalarType *ptr) noexcept { return _mm_loadu_pd(ptr); }
    FORCE_INLINE static void store(ScalarType *ptr, VecType val) noexcept { _mm_store_pd(ptr, val); }
    FORCE_INLINE static void storeu(ScalarType *ptr, VecType val) noexcept { _mm_storeu_pd(ptr, val); }
    FORCE_INLINE static VecType set1(ScalarType scalar) noexcept { return _mm_set1_pd(scalar); }

    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return _mm_add_pd(a, b); }
    FORCE_INLINE static VecType add(VecType a, ScalarType b) noexcept { return _mm_add_pd(a, set1(b)); }

    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept { return _mm_mul_pd(a, b); }
    FORCE_INLINE static VecType mul(VecType a, ScalarType b) noexcept { return _mm_mul_pd(a, set1(b)); }

    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return _mm_sub_pd(a, b); }
    FORCE_INLINE static VecType sub(VecType a, ScalarType b) noexcept { return _mm_sub_pd(a, set1(b)); }
    FORCE_INLINE static VecType sub(ScalarType a, VecType b) noexcept { return _mm_sub_pd(set1(a), b); }

    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept { return _mm_div_pd(a, b); }
    FORCE_INLINE static VecType div(VecType a, ScalarType b) noexcept { return _mm_div_pd(a, set1(b)); }
    FORCE_INLINE static VecType div(ScalarType a, VecType b) noexcept { return _mm_div_pd(set1(a), b); }

    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return _mm_fmadd_pd(a, b, c); }

    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept { return _mm_min_pd(a, b); }
    FORCE_INLINE static VecType min(VecType a, ScalarType b) noexcept { return _mm_min_pd(a, set1(b)); }

    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept { return _mm_max_pd(a, b); }
    FORCE_INLINE static VecType max(VecType a, ScalarType b) noexcept { return _mm_max_pd(a, set1(b)); }

    FORCE_INLINE static VecType gather(const ScalarType *base, const my_size_t *indices) noexcept
    {
        __m128i vindex = _mm_loadu_si128(reinterpret_cast<const __m128i *>(indices));
        return _mm_i64gather_pd(base, vindex, sizeof(ScalarType));
    }

    FORCE_INLINE static void scatter(ScalarType *base, const my_size_t *indices, VecType val) noexcept
    {
        alignas(16) ScalarType tmp[simdWidth];
        _mm_storeu_pd(tmp, val);
        for (my_size_t i = 0; i < simdWidth; ++i)
            base[indices[i]] = tmp[i];
    }

    FORCE_INLINE static VecType abs(VecType v) noexcept
    {
        __m128d sign_mask = _mm_set1_pd(-0.0);
        return _mm_andnot_pd(sign_mask, v);
    }

    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, ScalarType tol) noexcept
    {
        __m128d diff = _mm_sub_pd(a, b);
        __m128d abs_diff = abs(diff);
        __m128d tol_vec = _mm_set1_pd(tol);
        __m128d cmp = _mm_cmple_pd(abs_diff, tol_vec);
        int mask = _mm_movemask_pd(cmp);
        return mask == 0x3; // all 2 lanes passed
    }
};

#endif // __SSE2_MICROKERNEL_H__