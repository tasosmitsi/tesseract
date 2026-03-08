#pragma once

#include <arm_neon.h>
#include "config.h"

// ============================================================================
// NEON (128-bit) float intrinsics
// ============================================================================

struct NeonFloatIntrinsics
{
    static constexpr my_size_t simdWidth = 4; // 128 bits / 32 bits per float = 4
    static constexpr my_size_t num_registers = 32;
    using VecType = float32x4_t;
    using ScalarType = float;

    FORCE_INLINE static VecType load(const ScalarType *ptr) noexcept { return vld1q_f32(ptr); }
    FORCE_INLINE static VecType loadu(const ScalarType *ptr) noexcept { return vld1q_f32(ptr); } // NEON has no alignment requirement
    FORCE_INLINE static void store(ScalarType *ptr, VecType val) noexcept { vst1q_f32(ptr, val); }
    FORCE_INLINE static void storeu(ScalarType *ptr, VecType val) noexcept { vst1q_f32(ptr, val); }
    FORCE_INLINE static VecType set1(ScalarType scalar) noexcept { return vdupq_n_f32(scalar); }

    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return vaddq_f32(a, b); }
    FORCE_INLINE static VecType add(VecType a, ScalarType b) noexcept { return vaddq_f32(a, set1(b)); }

    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept { return vmulq_f32(a, b); }
    FORCE_INLINE static VecType mul(VecType a, ScalarType b) noexcept { return vmulq_f32(a, set1(b)); }

    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return vsubq_f32(a, b); }
    FORCE_INLINE static VecType sub(VecType a, ScalarType b) noexcept { return vsubq_f32(a, set1(b)); }
    FORCE_INLINE static VecType sub(ScalarType a, VecType b) noexcept { return vsubq_f32(set1(a), b); }

    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept
    {
        // AArch64 has vdivq_f32; AArch32 needs reciprocal estimate + Newton-Raphson.
#ifdef __aarch64__
        return vdivq_f32(a, b);
#else
        // Two Newton-Raphson iterations on the reciprocal estimate
        float32x4_t recip = vrecpeq_f32(b);
        recip = vmulq_f32(vrecpsq_f32(b, recip), recip);
        recip = vmulq_f32(vrecpsq_f32(b, recip), recip);
        return vmulq_f32(a, recip);
#endif
    }
    FORCE_INLINE static VecType div(VecType a, ScalarType b) noexcept { return div(a, set1(b)); }
    FORCE_INLINE static VecType div(ScalarType a, VecType b) noexcept { return div(set1(a), b); }

    // fmadd: a*b + c   — maps to single VFMA instruction on Cortex-A72
    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return vfmaq_f32(c, a, b); }
    FORCE_INLINE static VecType fmadd(VecType a, ScalarType b, VecType c) noexcept { return vfmaq_f32(c, a, set1(b)); }

    // fmsub: a*b - c
    FORCE_INLINE static VecType fmsub(VecType a, VecType b, VecType c) noexcept { return vsubq_f32(vfmaq_f32(vdupq_n_f32(0.0f), a, b), c); }
    FORCE_INLINE static VecType fmsub(VecType a, ScalarType b, VecType c) noexcept { return fmsub(a, set1(b), c); }

    // fnmadd: -(a*b) + c  — NEON vfmsq_f32 computes c - a*b
    FORCE_INLINE static VecType fnmadd(VecType a, VecType b, VecType c) noexcept { return vfmsq_f32(c, a, b); }
    FORCE_INLINE static VecType fnmadd(VecType a, ScalarType b, VecType c) noexcept { return vfmsq_f32(c, a, set1(b)); }

    // fnmsub: -(a*b) - c
    FORCE_INLINE static VecType fnmsub(VecType a, VecType b, VecType c) noexcept { return vnegq_f32(vfmaq_f32(c, a, b)); }
    FORCE_INLINE static VecType fnmsub(VecType a, ScalarType b, VecType c) noexcept { return fnmsub(a, set1(b), c); }

    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept { return vminq_f32(a, b); }
    FORCE_INLINE static VecType min(VecType a, ScalarType b) noexcept { return vminq_f32(a, set1(b)); }

    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept { return vmaxq_f32(a, b); }
    FORCE_INLINE static VecType max(VecType a, ScalarType b) noexcept { return vmaxq_f32(a, set1(b)); }

    // ============================================================================
    // Gather: NEON has no hardware gather — scalar fallback
    // ============================================================================
    FORCE_INLINE static VecType gather(const ScalarType *base, const my_size_t *indices) noexcept
    {
        alignas(16) ScalarType tmp[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
            tmp[i] = base[indices[i]];
        return vld1q_f32(tmp);
    }

    FORCE_INLINE static void scatter(ScalarType *base, const my_size_t *indices, VecType val) noexcept
    {
        alignas(16) ScalarType tmp[simdWidth];
        vst1q_f32(tmp, val);
        for (my_size_t i = 0; i < simdWidth; ++i)
            base[indices[i]] = tmp[i];
    }

    FORCE_INLINE static VecType abs(VecType v) noexcept
    {
        return vabsq_f32(v);
    }

    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, ScalarType tol) noexcept
    {
        float32x4_t diff = vsubq_f32(a, b);
        float32x4_t abs_diff = vabsq_f32(diff);
        float32x4_t tol_vec = vdupq_n_f32(tol);
        uint32x4_t cmp = vcleq_f32(abs_diff, tol_vec); // abs_diff <= tol
        // All lanes must be 0xFFFFFFFF → min across lanes must be non-zero
        return vminvq_u32(cmp) != 0;
    }
};

// ============================================================================
// NEON (128-bit) double intrinsics
// ============================================================================

struct NeonDoubleIntrinsics
{
    static constexpr my_size_t simdWidth = 2; // 128 bits / 64 bits per double = 2
    static constexpr my_size_t num_registers = 32;
    using VecType = float64x2_t;
    using ScalarType = double;

    FORCE_INLINE static VecType load(const ScalarType *ptr) noexcept { return vld1q_f64(ptr); }
    FORCE_INLINE static VecType loadu(const ScalarType *ptr) noexcept { return vld1q_f64(ptr); }
    FORCE_INLINE static void store(ScalarType *ptr, VecType val) noexcept { vst1q_f64(ptr, val); }
    FORCE_INLINE static void storeu(ScalarType *ptr, VecType val) noexcept { vst1q_f64(ptr, val); }
    FORCE_INLINE static VecType set1(ScalarType scalar) noexcept { return vdupq_n_f64(scalar); }

    FORCE_INLINE static VecType add(VecType a, VecType b) noexcept { return vaddq_f64(a, b); }
    FORCE_INLINE static VecType add(VecType a, ScalarType b) noexcept { return vaddq_f64(a, set1(b)); }

    FORCE_INLINE static VecType mul(VecType a, VecType b) noexcept { return vmulq_f64(a, b); }
    FORCE_INLINE static VecType mul(VecType a, ScalarType b) noexcept { return vmulq_f64(a, set1(b)); }

    FORCE_INLINE static VecType sub(VecType a, VecType b) noexcept { return vsubq_f64(a, b); }
    FORCE_INLINE static VecType sub(VecType a, ScalarType b) noexcept { return vsubq_f64(a, set1(b)); }
    FORCE_INLINE static VecType sub(ScalarType a, VecType b) noexcept { return vsubq_f64(set1(a), b); }

    FORCE_INLINE static VecType div(VecType a, VecType b) noexcept { return vdivq_f64(a, b); }
    FORCE_INLINE static VecType div(VecType a, ScalarType b) noexcept { return vdivq_f64(a, set1(b)); }
    FORCE_INLINE static VecType div(ScalarType a, VecType b) noexcept { return vdivq_f64(set1(a), b); }

    // fmadd: a*b + c
    FORCE_INLINE static VecType fmadd(VecType a, VecType b, VecType c) noexcept { return vfmaq_f64(c, a, b); }
    FORCE_INLINE static VecType fmadd(VecType a, ScalarType b, VecType c) noexcept { return vfmaq_f64(c, a, set1(b)); }

    // fmsub: a*b - c
    FORCE_INLINE static VecType fmsub(VecType a, VecType b, VecType c) noexcept { return vsubq_f64(vfmaq_f64(vdupq_n_f64(0.0), a, b), c); }
    FORCE_INLINE static VecType fmsub(VecType a, ScalarType b, VecType c) noexcept { return fmsub(a, set1(b), c); }

    // fnmadd: -(a*b) + c
    FORCE_INLINE static VecType fnmadd(VecType a, VecType b, VecType c) noexcept { return vfmsq_f64(c, a, b); }
    FORCE_INLINE static VecType fnmadd(VecType a, ScalarType b, VecType c) noexcept { return vfmsq_f64(c, a, set1(b)); }

    // fnmsub: -(a*b) - c
    FORCE_INLINE static VecType fnmsub(VecType a, VecType b, VecType c) noexcept { return vnegq_f64(vfmaq_f64(c, a, b)); }
    FORCE_INLINE static VecType fnmsub(VecType a, ScalarType b, VecType c) noexcept { return fnmsub(a, set1(b), c); }

    FORCE_INLINE static VecType min(VecType a, VecType b) noexcept { return vminq_f64(a, b); }
    FORCE_INLINE static VecType min(VecType a, ScalarType b) noexcept { return vminq_f64(a, set1(b)); }

    FORCE_INLINE static VecType max(VecType a, VecType b) noexcept { return vmaxq_f64(a, b); }
    FORCE_INLINE static VecType max(VecType a, ScalarType b) noexcept { return vmaxq_f64(a, set1(b)); }

    FORCE_INLINE static VecType gather(const ScalarType *base, const my_size_t *indices) noexcept
    {
        alignas(16) ScalarType tmp[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
            tmp[i] = base[indices[i]];
        return vld1q_f64(tmp);
    }

    FORCE_INLINE static void scatter(ScalarType *base, const my_size_t *indices, VecType val) noexcept
    {
        alignas(16) ScalarType tmp[simdWidth];
        vst1q_f64(tmp, val);
        for (my_size_t i = 0; i < simdWidth; ++i)
            base[indices[i]] = tmp[i];
    }

    FORCE_INLINE static VecType abs(VecType v) noexcept
    {
        return vabsq_f64(v);
    }

    FORCE_INLINE static bool all_within_tolerance(VecType a, VecType b, ScalarType tol) noexcept
    {
        float64x2_t diff = vsubq_f64(a, b);
        float64x2_t abs_diff = vabsq_f64(diff);
        float64x2_t tol_vec = vdupq_n_f64(tol);
        uint64x2_t cmp = vcleq_f64(abs_diff, tol_vec);
        // Both lanes must pass
        return (vgetq_lane_u64(cmp, 0) & vgetq_lane_u64(cmp, 1)) != 0;
    }
};