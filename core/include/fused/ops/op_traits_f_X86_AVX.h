#pragma once

#include <immintrin.h>
#include "op_traits_generic.h" // to ensure generic is known

struct X86_AVX
{
};

template <typename T, my_size_t WidthBits>
struct SimdTraits;

// SimdTraits specializations
template <>
struct SimdTraits<int, 256>
{
    using simd_t = __m256i;
    static constexpr my_size_t width = 8; // 256 bits / 32 bits per int = 8

    FORCE_INLINE static simd_t load(const int *ptr)
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    FORCE_INLINE static simd_t loadu(const int *ptr)
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    FORCE_INLINE static void store(int *ptr, simd_t val)
    {
        _mm256_store_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    FORCE_INLINE static simd_t add(simd_t a, simd_t b)
    {
        return _mm256_add_epi32(a, b);
    }

    FORCE_INLINE static simd_t sub(simd_t a, simd_t b)
    {
        return _mm256_sub_epi32(a, b);
    }

    FORCE_INLINE static simd_t mul(simd_t a, simd_t b)
    {
        return _mm256_mullo_epi32(a, b);
    }

    // Integer division (no SIMD support) — fallback to scalar
    FORCE_INLINE static simd_t div(simd_t a, simd_t b)
    {
        alignas(32) int a_vals[8], b_vals[8], res[8];
        _mm256_store_si256(reinterpret_cast<__m256i *>(a_vals), a);
        _mm256_store_si256(reinterpret_cast<__m256i *>(b_vals), b);
        for (int i = 0; i < 8; ++i)
            res[i] = b_vals[i] != 0 ? a_vals[i] / b_vals[i] : 0; // fallback logic
        return _mm256_load_si256(reinterpret_cast<const __m256i *>(res));
    }

    FORCE_INLINE static simd_t set1(int x)
    {
        return _mm256_set1_epi32(x);
    }
};

template <>
struct SimdTraits<int, 128>
{
    using simd_t = __m128i;
    static constexpr my_size_t width = 4; // 128 bits / 32 bits per int = 4

    FORCE_INLINE static simd_t load(const int *ptr) { return _mm_load_si128(reinterpret_cast<const __m128i *>(ptr)); }
    FORCE_INLINE static simd_t loadu(const int *ptr) { return _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)); }
    FORCE_INLINE static void store(int *ptr, simd_t val) { _mm_store_si128(reinterpret_cast<__m128i *>(ptr), val); }

    FORCE_INLINE static simd_t add(simd_t a, simd_t b) { return _mm_add_epi32(a, b); }
    FORCE_INLINE static simd_t sub(simd_t a, simd_t b) { return _mm_sub_epi32(a, b); }
    FORCE_INLINE static simd_t mul(simd_t a, simd_t b)
    {
        // SSE2 doesn't support 32-bit integer multiplication natively.
        // This requires SSE4.1 (_mm_mullo_epi32).
        return _mm_mullo_epi32(a, b);
    }

    // Integer division is not supported in SIMD. You'd need scalar fallback.
    // Here is a dummy to avoid compilation errors.
    FORCE_INLINE static simd_t div(simd_t a, simd_t b)
    {
        // This will not work properly for general use
        // Use scalar fallback or AVX512 if available
        alignas(16) int a_vals[4], b_vals[4], res[4];
        _mm_store_si128(reinterpret_cast<__m128i *>(a_vals), a);
        _mm_store_si128(reinterpret_cast<__m128i *>(b_vals), b);
        for (int i = 0; i < 4; ++i)
            res[i] = b_vals[i] != 0 ? a_vals[i] / b_vals[i] : 0;
        return _mm_load_si128(reinterpret_cast<const __m128i *>(res));
    }

    FORCE_INLINE static simd_t set1(int x) { return _mm_set1_epi32(x); }
};

template <>
struct SimdTraits<float, 128>
{
    using simd_t = __m128;
    static constexpr my_size_t width = 4;

    FORCE_INLINE static simd_t load(const float *ptr) { return _mm_load_ps(ptr); }
    FORCE_INLINE static simd_t loadu(const float *ptr) { return _mm_loadu_ps(ptr); }
    FORCE_INLINE static void store(float *ptr, simd_t val) { _mm_store_ps(ptr, val); }
    FORCE_INLINE static simd_t add(simd_t a, simd_t b) { return _mm_add_ps(a, b); }
    FORCE_INLINE static simd_t sub(simd_t a, simd_t b) { return _mm_sub_ps(a, b); }
    FORCE_INLINE static simd_t mul(simd_t a, simd_t b) { return _mm_mul_ps(a, b); }
    FORCE_INLINE static simd_t div(simd_t a, simd_t b) { return _mm_div_ps(a, b); }
    FORCE_INLINE static simd_t set1(float x) { return _mm_set1_ps(x); }
};

template <>
struct SimdTraits<float, 256>
{
    using simd_t = __m256;
    static constexpr my_size_t width = 8;

    FORCE_INLINE static simd_t load(const float *ptr) { return _mm256_load_ps(ptr); }
    FORCE_INLINE static simd_t loadu(const float *ptr) { return _mm256_loadu_ps(ptr); }
    FORCE_INLINE static void store(float *ptr, simd_t val) { _mm256_store_ps(ptr, val); }
    FORCE_INLINE static simd_t add(simd_t a, simd_t b) { return _mm256_add_ps(a, b); }
    FORCE_INLINE static simd_t sub(simd_t a, simd_t b) { return _mm256_sub_ps(a, b); }
    FORCE_INLINE static simd_t mul(simd_t a, simd_t b) { return _mm256_mul_ps(a, b); }
    FORCE_INLINE static simd_t div(simd_t a, simd_t b) { return _mm256_div_ps(a, b); }
    FORCE_INLINE static simd_t set1(float x) { return _mm256_set1_ps(x); }
    FORCE_INLINE static auto fmadd(simd_t a, simd_t b, simd_t c)
        -> decltype(_mm256_fmadd_ps(a, b, c))
    {
        return _mm256_fmadd_ps(a, b, c);
    }
};

template <>
struct SimdTraits<double, 128>
{
    using simd_t = __m128d;
    static constexpr my_size_t width = 2;

    FORCE_INLINE static simd_t load(const double *ptr) { return _mm_load_pd(ptr); }
    FORCE_INLINE static simd_t loadu(const double *ptr) { return _mm_loadu_pd(ptr); }
    FORCE_INLINE static void store(double *ptr, simd_t val) { _mm_store_pd(ptr, val); }
    FORCE_INLINE static simd_t add(simd_t a, simd_t b) { return _mm_add_pd(a, b); }
    FORCE_INLINE static simd_t sub(simd_t a, simd_t b) { return _mm_sub_pd(a, b); }
    FORCE_INLINE static simd_t mul(simd_t a, simd_t b) { return _mm_mul_pd(a, b); }
    FORCE_INLINE static simd_t div(simd_t a, simd_t b) { return _mm_div_pd(a, b); }
    FORCE_INLINE static simd_t set1(double x) { return _mm_set1_pd(x); }
};

template <>
struct SimdTraits<double, 256>
{
    using simd_t = __m256d;
    static constexpr my_size_t width = 4;

    FORCE_INLINE static simd_t load(const double *ptr) { return _mm256_load_pd(ptr); }
    FORCE_INLINE static simd_t loadu(const double *ptr) { return _mm256_loadu_pd(ptr); }
    FORCE_INLINE static void store(double *ptr, simd_t val) { _mm256_store_pd(ptr, val); }
    FORCE_INLINE static simd_t add(simd_t a, simd_t b) { return _mm256_add_pd(a, b); }
    FORCE_INLINE static simd_t sub(simd_t a, simd_t b) { return _mm256_sub_pd(a, b); }
    FORCE_INLINE static simd_t mul(simd_t a, simd_t b) { return _mm256_mul_pd(a, b); }
    FORCE_INLINE static simd_t div(simd_t a, simd_t b) { return _mm256_div_pd(a, b); }
    FORCE_INLINE static simd_t set1(double x) { return _mm256_set1_pd(x); }
};

// OpTraits specialization for X86_AVX
// OpTraits delegates to SimdTraits<T, WidthBits>
template <typename T, my_size_t WidthBits>
struct OpTraits<T, WidthBits, X86_AVX>
{
    using Simd = SimdTraits<T, WidthBits>;
    using type = typename Simd::simd_t;
    static constexpr my_size_t width = Simd::width;

    FORCE_INLINE static type add(type a, type b) { return Simd::add(a, b); }
    FORCE_INLINE static type sub(type a, type b) { return Simd::sub(a, b); }
    FORCE_INLINE static type mul(type a, type b) { return Simd::mul(a, b); }
    FORCE_INLINE static type div(type a, type b) { return Simd::div(a, b); }

    FORCE_INLINE static type add(type a, T s) { return Simd::add(a, Simd::set1(s)); }
    FORCE_INLINE static type sub(type a, T s) { return Simd::sub(a, Simd::set1(s)); }
    FORCE_INLINE static type mul(type a, T s) { return Simd::mul(a, Simd::set1(s)); }
    FORCE_INLINE static type div(type a, T s) { return Simd::div(a, Simd::set1(s)); }

    FORCE_INLINE static type sub(T s, type a) { return Simd::sub(Simd::set1(s), a); }
    FORCE_INLINE static type div(T s, type a) { return Simd::div(Simd::set1(s), a); }

    FORCE_INLINE static type load(const T *ptr) { return Simd::load(ptr); }
    FORCE_INLINE static type loadu(const T *ptr) { return Simd::loadu(ptr); }
    FORCE_INLINE static void store(T *ptr, type val) { Simd::store(ptr, val); }

    static void test()
    {
        std::cout << "OpTraits: scalar:" << typeid(T).name() << ", width: " << WidthBits << ", width: "
                  << Simd::width << ", bus_type: " << typeid(type).name() << "X86_AVX>\n";
    }
};
