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
    // GEMM tiling constants (register-blocked)
    static constexpr my_size_t num_registers = 16;
    static constexpr my_size_t MR = 4;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 24
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

    // fmadd: a*b + c
    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return _mm256_fmadd_ps(a, b, c); }
    FORCE_INLINE static VecType fmadd(VecType a, ScalarType b, VecType c) noexcept { return _mm256_fmadd_ps(a, set1(b), c); }

    // fmsub: a*b - c
    FORCE_INLINE static VecType fmsub(VecType a, VecType b, VecType c) noexcept { return _mm256_fmsub_ps(a, b, c); }
    FORCE_INLINE static VecType fmsub(VecType a, ScalarType b, VecType c) noexcept { return _mm256_fmsub_ps(a, set1(b), c); }

    // fnmadd: -(a*b) + c
    FORCE_INLINE static VecType fnmadd(VecType a, VecType b, VecType c) noexcept { return _mm256_fnmadd_ps(a, b, c); }
    FORCE_INLINE static VecType fnmadd(VecType a, ScalarType b, VecType c) noexcept { return _mm256_fnmadd_ps(a, set1(b), c); }

    // fnmsub: -(a*b) - c
    FORCE_INLINE static VecType fnmsub(VecType a, VecType b, VecType c) noexcept { return _mm256_fnmsub_ps(a, b, c); }
    FORCE_INLINE static VecType fnmsub(VecType a, ScalarType b, VecType c) noexcept { return _mm256_fnmsub_ps(a, set1(b), c); }

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

    FORCE_INLINE static VecType abs(VecType v) noexcept
    {
        // Clear sign bit: AND with 0x7FFFFFFF
        __m256 sign_mask = _mm256_set1_ps(-0.0f);
        return _mm256_andnot_ps(sign_mask, v);
    }

    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, ScalarType tol) noexcept
    {
        __m256 diff = _mm256_sub_ps(a, b);
        __m256 abs_diff = abs(diff);
        __m256 tol_vec = _mm256_set1_ps(tol);
        __m256 cmp = _mm256_cmp_ps(abs_diff, tol_vec, _CMP_LE_OQ); // abs_diff <= tol
        int mask = _mm256_movemask_ps(cmp);
        return mask == 0xFF; // all 8 lanes passed
    }
};

template <>
struct Microkernel<double, 256, X86_AVX>
{
    static constexpr my_size_t simdWidth = 4; // 256 bits / 64 bits per double = 4
    // GEMM tiling constants (register-blocked)
    static constexpr my_size_t num_registers = 16;
    static constexpr my_size_t MR = 4;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 12
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

    // fmadd: a*b + c
    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return _mm256_fmadd_pd(a, b, c); }
    FORCE_INLINE static VecType fmadd(VecType a, ScalarType b, VecType c) noexcept { return _mm256_fmadd_pd(a, set1(b), c); }

    // fmsub: a*b - c
    FORCE_INLINE static VecType fmsub(VecType a, VecType b, VecType c) noexcept { return _mm256_fmsub_pd(a, b, c); }
    FORCE_INLINE static VecType fmsub(VecType a, ScalarType b, VecType c) noexcept { return _mm256_fmsub_pd(a, set1(b), c); }

    // fnmadd: -(a*b) + c
    FORCE_INLINE static VecType fnmadd(VecType a, VecType b, VecType c) noexcept { return _mm256_fnmadd_pd(a, b, c); }
    FORCE_INLINE static VecType fnmadd(VecType a, ScalarType b, VecType c) noexcept { return _mm256_fnmadd_pd(a, set1(b), c); }

    // fnmsub: -(a*b) - c
    FORCE_INLINE static VecType fnmsub(VecType a, VecType b, VecType c) noexcept { return _mm256_fnmsub_pd(a, b, c); }
    FORCE_INLINE static VecType fnmsub(VecType a, ScalarType b, VecType c) noexcept { return _mm256_fnmsub_pd(a, set1(b), c); }

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

    FORCE_INLINE static VecType abs(VecType v) noexcept
    {
        __m256d sign_mask = _mm256_set1_pd(-0.0);
        return _mm256_andnot_pd(sign_mask, v);
    }

    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, ScalarType tol) noexcept
    {
        __m256d diff = _mm256_sub_pd(a, b);
        __m256d abs_diff = abs(diff);
        __m256d tol_vec = _mm256_set1_pd(tol);
        __m256d cmp = _mm256_cmp_pd(abs_diff, tol_vec, _CMP_LE_OQ);
        int mask = _mm256_movemask_pd(cmp);
        return mask == 0xF; // all 4 lanes passed
    }
};

// ============================================================================
// AVX2 (256-bit) int32_t specialization
// ============================================================================

template <>
struct Microkernel<int32_t, 256, X86_AVX>
{
    static constexpr my_size_t simdWidth = 8; // 256 bits / 32 bits = 8
    static constexpr my_size_t num_registers = 16;
    static constexpr my_size_t MR = 4;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 24
    using VecType = __m256i;
    using ScalarType = int32_t;

    FORCE_INLINE static VecType load(const ScalarType *ptr) noexcept { return _mm256_load_si256(reinterpret_cast<const __m256i *>(ptr)); }
    FORCE_INLINE static VecType loadu(const ScalarType *ptr) noexcept { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)); }
    FORCE_INLINE static void store(ScalarType *ptr, VecType val) noexcept { _mm256_store_si256(reinterpret_cast<__m256i *>(ptr), val); }
    FORCE_INLINE static void storeu(ScalarType *ptr, VecType val) noexcept { _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val); }
    FORCE_INLINE static VecType set1(ScalarType scalar) noexcept { return _mm256_set1_epi32(scalar); }

    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return _mm256_add_epi32(a, b); }
    FORCE_INLINE static VecType add(VecType a, ScalarType b) noexcept { return _mm256_add_epi32(a, set1(b)); }

    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept { return _mm256_mullo_epi32(a, b); }
    FORCE_INLINE static VecType mul(VecType a, ScalarType b) noexcept { return _mm256_mullo_epi32(a, set1(b)); }

    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return _mm256_sub_epi32(a, b); }
    FORCE_INLINE static VecType sub(VecType a, ScalarType b) noexcept { return _mm256_sub_epi32(a, set1(b)); }
    FORCE_INLINE static VecType sub(ScalarType a, VecType b) noexcept { return _mm256_sub_epi32(set1(a), b); }

    // No SIMD integer divide exists on x86; scalar fallback.
    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept
    {
        alignas(32) ScalarType va[simdWidth], vb[simdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(va), a);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vb), b);
        for (my_size_t i = 0; i < simdWidth; ++i)
            va[i] /= vb[i];
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va));
    }
    FORCE_INLINE static VecType div(VecType a, ScalarType b) noexcept
    {
        alignas(32) ScalarType va[simdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(va), a);
        for (my_size_t i = 0; i < simdWidth; ++i)
            va[i] /= b;
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va));
    }
    FORCE_INLINE static VecType div(ScalarType a, VecType b) noexcept
    {
        alignas(32) ScalarType vb[simdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vb), b);
        alignas(32) ScalarType vr[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
            vr[i] = a / vb[i];
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vr));
    }

    // NOTE: No FMA for integers in AVX2. Emulate as mul + add.
    // fmadd: a*b + c
    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return _mm256_add_epi32(_mm256_mullo_epi32(a, b), c); }
    FORCE_INLINE static VecType fmadd(VecType a, ScalarType b, VecType c) noexcept { return _mm256_add_epi32(_mm256_mullo_epi32(a, set1(b)), c); }

    // fmsub: a*b - c
    FORCE_INLINE static VecType fmsub(VecType a, VecType b, VecType c) noexcept { return _mm256_sub_epi32(_mm256_mullo_epi32(a, b), c); }
    FORCE_INLINE static VecType fmsub(VecType a, ScalarType b, VecType c) noexcept { return _mm256_sub_epi32(_mm256_mullo_epi32(a, set1(b)), c); }

    // fnmadd: -(a*b) + c  =>  c - a*b
    FORCE_INLINE static VecType fnmadd(VecType a, VecType b, VecType c) noexcept { return _mm256_sub_epi32(c, _mm256_mullo_epi32(a, b)); }
    FORCE_INLINE static VecType fnmadd(VecType a, ScalarType b, VecType c) noexcept { return _mm256_sub_epi32(c, _mm256_mullo_epi32(a, set1(b))); }

    // fnmsub: -(a*b) - c
    FORCE_INLINE static VecType fnmsub(VecType a, VecType b, VecType c) noexcept
    {
        // 0 - a*b - c
        __m256i neg_ab = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_mullo_epi32(a, b));
        return _mm256_sub_epi32(neg_ab, c);
    }
    FORCE_INLINE static VecType fnmsub(VecType a, ScalarType b, VecType c) noexcept
    {
        __m256i neg_ab = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_mullo_epi32(a, set1(b)));
        return _mm256_sub_epi32(neg_ab, c);
    }

    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept { return _mm256_min_epi32(a, b); }
    FORCE_INLINE static VecType min(VecType a, ScalarType b) noexcept { return _mm256_min_epi32(a, set1(b)); }

    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept { return _mm256_max_epi32(a, b); }
    FORCE_INLINE static VecType max(VecType a, ScalarType b) noexcept { return _mm256_max_epi32(a, set1(b)); }

    // ============================================================================
    // Gather
    // ============================================================================
    FORCE_INLINE static VecType gather(const ScalarType *base, const my_size_t *indices) noexcept
    {
        alignas(32) int32_t idx32[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
        {
            idx32[i] = static_cast<int32_t>(indices[i]);
        }
        __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(idx32));
        return _mm256_i32gather_epi32(base, vindex, sizeof(ScalarType));
    }

    FORCE_INLINE static void scatter(ScalarType *base, const my_size_t *indices, VecType val) noexcept
    {
        alignas(32) ScalarType tmp[simdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp), val);
        for (my_size_t i = 0; i < simdWidth; ++i)
            base[indices[i]] = tmp[i];
    }

    FORCE_INLINE static VecType abs(VecType v) noexcept
    {
        return _mm256_abs_epi32(v);
    }

    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, ScalarType tol) noexcept
    {
        __m256i diff = _mm256_sub_epi32(a, b);
        __m256i abs_diff = _mm256_abs_epi32(diff);
        __m256i tol_vec = _mm256_set1_epi32(tol);
        // cmpgt: abs_diff > tol  →  we want none to be greater
        __m256i cmp = _mm256_cmpgt_epi32(abs_diff, tol_vec);
        return _mm256_testz_si256(cmp, cmp); // true if cmp is all-zero
    }
};

// ============================================================================
// AVX2 (256-bit) int64_t specialization
// ============================================================================

template <>
struct Microkernel<int64_t, 256, X86_AVX>
{
    static constexpr my_size_t simdWidth = 4; // 256 bits / 64 bits = 4
    static constexpr my_size_t num_registers = 16;
    static constexpr my_size_t MR = 4;
    static constexpr my_size_t NR_VECS = 3;
    static constexpr my_size_t NR = NR_VECS * simdWidth; // 12
    using VecType = __m256i;
    using ScalarType = int64_t;

    FORCE_INLINE static VecType load(const ScalarType *ptr) noexcept { return _mm256_load_si256(reinterpret_cast<const __m256i *>(ptr)); }
    FORCE_INLINE static VecType loadu(const ScalarType *ptr) noexcept { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)); }
    FORCE_INLINE static void store(ScalarType *ptr, VecType val) noexcept { _mm256_store_si256(reinterpret_cast<__m256i *>(ptr), val); }
    FORCE_INLINE static void storeu(ScalarType *ptr, VecType val) noexcept { _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val); }
    FORCE_INLINE static VecType set1(ScalarType scalar) noexcept { return _mm256_set1_epi64x(scalar); }

    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return _mm256_add_epi64(a, b); }
    FORCE_INLINE static VecType add(VecType a, ScalarType b) noexcept { return _mm256_add_epi64(a, set1(b)); }

    // NOTE: AVX2 has NO native 64-bit integer multiply.
    // Emulate via 32-bit partial products.
    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept
    {
        // Low 32 bits of each 64-bit lane: _mm256_mul_epu32 gives 64-bit results
        // For full 64×64→low64, we need:
        //   result = lo(a)*lo(b) + (lo(a)*hi(b) + hi(a)*lo(b)) << 32
        __m256i a_hi = _mm256_srli_epi64(a, 32);
        __m256i b_hi = _mm256_srli_epi64(b, 32);

        __m256i lo_lo = _mm256_mul_epu32(a, b);    // lo(a) * lo(b) → 64-bit
        __m256i lo_hi = _mm256_mul_epu32(a, b_hi); // lo(a) * hi(b) → 64-bit
        __m256i hi_lo = _mm256_mul_epu32(a_hi, b); // hi(a) * lo(b) → 64-bit

        __m256i cross = _mm256_add_epi64(lo_hi, hi_lo);
        __m256i cross_shifted = _mm256_slli_epi64(cross, 32);

        return _mm256_add_epi64(lo_lo, cross_shifted);
    }
    FORCE_INLINE static VecType mul(VecType a, ScalarType b) noexcept { return mul(a, set1(b)); }

    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return _mm256_sub_epi64(a, b); }
    FORCE_INLINE static VecType sub(VecType a, ScalarType b) noexcept { return _mm256_sub_epi64(a, set1(b)); }
    FORCE_INLINE static VecType sub(ScalarType a, VecType b) noexcept { return _mm256_sub_epi64(set1(a), b); }

    // No SIMD integer divide exists on x86; scalar fallback.
    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept
    {
        alignas(32) ScalarType va[simdWidth], vb[simdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(va), a);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vb), b);
        for (my_size_t i = 0; i < simdWidth; ++i)
            va[i] /= vb[i];
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va));
    }
    FORCE_INLINE static VecType div(VecType a, ScalarType b) noexcept
    {
        alignas(32) ScalarType va[simdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(va), a);
        for (my_size_t i = 0; i < simdWidth; ++i)
            va[i] /= b;
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(va));
    }
    FORCE_INLINE static VecType div(ScalarType a, VecType b) noexcept
    {
        alignas(32) ScalarType vb[simdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vb), b);
        alignas(32) ScalarType vr[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
            vr[i] = a / vb[i];
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vr));
    }

    // Emulated FMA
    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return add(mul(a, b), c); }
    FORCE_INLINE static VecType fmadd(VecType a, ScalarType b, VecType c) noexcept { return add(mul(a, b), c); }

    FORCE_INLINE static VecType fmsub(VecType a, VecType b, VecType c) noexcept { return sub(mul(a, b), c); }
    FORCE_INLINE static VecType fmsub(VecType a, ScalarType b, VecType c) noexcept { return sub(mul(a, b), c); }

    FORCE_INLINE static VecType fnmadd(VecType a, VecType b, VecType c) noexcept { return sub(c, mul(a, b)); }
    FORCE_INLINE static VecType fnmadd(VecType a, ScalarType b, VecType c) noexcept { return sub(c, mul(a, b)); }

    FORCE_INLINE static VecType fnmsub(VecType a, VecType b, VecType c) noexcept
    {
        return sub(_mm256_setzero_si256(), add(mul(a, b), c));
    }
    FORCE_INLINE static VecType fnmsub(VecType a, ScalarType b, VecType c) noexcept
    {
        return sub(_mm256_setzero_si256(), add(mul(a, b), c));
    }

    // NOTE: AVX2 has no _mm256_min/max_epi64. Emulate via comparison.
    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept
    {
        // AVX-512 has _mm256_min_epi64, but for AVX2 we emulate:
        __m256i gt = _mm256_cmpgt_epi64(a, b); // a > b ? 0xFFF... : 0
        return _mm256_blendv_epi8(a, b, gt);   // pick b where a > b
    }
    FORCE_INLINE static VecType min(VecType a, ScalarType b) noexcept { return min(a, set1(b)); }

    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept
    {
        __m256i gt = _mm256_cmpgt_epi64(a, b);
        return _mm256_blendv_epi8(b, a, gt); // pick a where a > b
    }
    FORCE_INLINE static VecType max(VecType a, ScalarType b) noexcept { return max(a, set1(b)); }

    // ============================================================================
    // Gather
    // ============================================================================
    FORCE_INLINE static VecType gather(const ScalarType *base, const my_size_t *indices) noexcept
    {
        // _mm256_i64gather_epi64 expects 4 × 64-bit indices — matches my_size_t on 64-bit
        __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(indices));
        return _mm256_i64gather_epi64(reinterpret_cast<const long long *>(base), vindex, sizeof(ScalarType));
    }

    FORCE_INLINE static void scatter(ScalarType *base, const my_size_t *indices, VecType val) noexcept
    {
        alignas(32) ScalarType tmp[simdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp), val);
        for (my_size_t i = 0; i < simdWidth; ++i)
            base[indices[i]] = tmp[i];
    }

    FORCE_INLINE static VecType abs(VecType v) noexcept
    {
        // AVX2 has no _mm256_abs_epi64. Emulate:
        __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v); // sign = (v < 0) ? 0xFFF... : 0
        __m256i neg_v = _mm256_sub_epi64(_mm256_setzero_si256(), v);
        return _mm256_blendv_epi8(v, neg_v, sign); // pick neg_v where v < 0
    }

    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, ScalarType tol) noexcept
    {
        __m256i diff = _mm256_sub_epi64(a, b);
        __m256i abs_diff = abs(diff);
        __m256i tol_vec = _mm256_set1_epi64x(tol);
        // abs_diff > tol ?
        __m256i gt = _mm256_cmpgt_epi64(abs_diff, tol_vec);
        return _mm256_testz_si256(gt, gt);
    }
};

#endif // __AVX2_MICROKERNEL_H__