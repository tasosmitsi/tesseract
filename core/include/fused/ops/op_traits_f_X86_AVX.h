#pragma once

#include <immintrin.h>
#include "op_traits_default.h" // to ensure generic is known

struct X86_AVX
{
};

// // Specialization for scalar float on X86_AVX just uses generic
// template <>
// struct OpTraits<float, X86_AVX> : OpTraits<float, GenericArch> {};

template <>
struct OpTraits<__m128, X86_AVX>
{
    FORCE_INLINE static __m128 add(__m128 a, __m128 b)
    {
        // std::cout << "lol";
        return _mm_add_ps(a, b);
    }
    FORCE_INLINE static __m128 sub(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
    FORCE_INLINE static __m128 mul(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
    FORCE_INLINE static __m128 div(__m128 a, __m128 b) { return _mm_div_ps(a, b); }

    static void test() { std::cout << "Specialized __m128 for X86_AVX\n"; }
};

template <>
struct OpTraits<__m256, X86_AVX>
{
    FORCE_INLINE static __m256 add(__m256 a, __m256 b)
    {
        // std::cout << "lol";
        return _mm256_add_ps(a, b);
    }
    FORCE_INLINE static __m256 sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
    FORCE_INLINE static __m256 mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
    FORCE_INLINE static __m256 div(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }

    // Optional : Fused multiply - add(a *b + c)
    FORCE_INLINE static __m256 mul_add(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }

    static void test() { std::cout << "Specialized __m256 for X86_AVX\n"; }
};

// template <>
// struct OpTraits<float, X86_AVX>
// {
//     static float add(float a, float b)
//     {
//         __m128 va = _mm_set_ss(a);
//         __m128 vb = _mm_set_ss(b);
//         __m128 vc = _mm_add_ss(va, vb);
//         return _mm_cvtss_f32(vc);
//     }

//     static float sub(float a, float b)
//     {
//         __m128 va = _mm_set_ss(a);
//         __m128 vb = _mm_set_ss(b);
//         __m128 vc = _mm_sub_ss(va, vb);
//         return _mm_cvtss_f32(vc);
//     }

//     static float mul(float a, float b)
//     {
//         __m128 va = _mm_set_ss(a);
//         __m128 vb = _mm_set_ss(b);
//         __m128 vc = _mm_mul_ss(va, vb);
//         return _mm_cvtss_f32(vc);
//     }

//     static float div(float a, float b)
//     {
//         if (b == 0.0f)
//         {
//             MyErrorHandler::error("Division by zero (AVX)");
//             return 0.0f;
//         }

//         __m128 va = _mm_set_ss(a);
//         __m128 vb = _mm_set_ss(b);
//         __m128 vc = _mm_div_ss(va, vb);
//         return _mm_cvtss_f32(vc);
//     }
// };
