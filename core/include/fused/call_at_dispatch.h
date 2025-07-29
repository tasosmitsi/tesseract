#ifndef CALL_AT_DISPATCH_H
#define CALL_AT_DISPATCH_H

// Check if we have C++14 standard library features
#if defined(__cpp_lib_integer_sequence) || __cplusplus >= 201402L
#include <utility>
#define USE_STD_INDEX_SEQUENCE
#endif

#ifdef USE_STD_INDEX_SEQUENCE
#pragma message "[COMPILE-TIME] Using std::index_sequence in CallAtDispatcher"
#else
#pragma message "[COMPILE-TIME] Using custom index_sequence in CallAtDispatcher"
// Custom index_sequence
template <my_size_t... Is>
struct index_sequence
{
};

template <my_size_t N, my_size_t... Is>
struct make_index_sequence : make_index_sequence<N - 1, N - 1, Is...>
{
};

template <my_size_t... Is>
struct make_index_sequence<0, Is...> : index_sequence<Is...>
{
};
#endif

// Actual dispatcher
template <typename T, my_size_t N>
struct CallAtDispatcher
{
    template <typename Expr, my_size_t... Is>
    static T callAtImpl(const Expr &expr, const my_size_t *indices,
#ifdef USE_STD_INDEX_SEQUENCE
                        std::index_sequence<Is...>
#else
                        index_sequence<Is...>
#endif
    )
    {
        return expr(indices[Is]...);
    }

    template <typename Expr>
    static T callAt(const Expr &expr, const my_size_t *indices)
    {
#ifdef USE_STD_INDEX_SEQUENCE
        return callAtImpl(expr, indices, std::make_index_sequence<N>{});
#else
        return callAtImpl(expr, indices, make_index_sequence<N>{});
#endif
    }
};

template <typename T, size_t N>
struct CallAtDispatcherSIMD
{
    template <typename Expr, size_t... Is>
    inline static __m128 callAtImpl(const Expr &expr, const size_t *indices, std::index_sequence<Is...>)
    {
        // We'll assume last dimension is at N-1 and load 4 consecutive elements there:
        // Build 4 calls where last index = indices[N-1] + offset (0..3)

        // Build an array of 4 __m128 floats by calling expr with adjusted last index
        float vals[4] = {
            expr((Is == N - 1 ? indices[Is] + 0 : indices[Is])..., SIMD4{}),
            expr((Is == N - 1 ? indices[Is] + 1 : indices[Is])..., SIMD4{}),
            expr((Is == N - 1 ? indices[Is] + 2 : indices[Is])..., SIMD4{}),
            expr((Is == N - 1 ? indices[Is] + 3 : indices[Is])..., SIMD4{})};

        // Load into __m128 vector
        return _mm_load_ps(vals);
    }

    template <typename Expr>
    inline static __m128 callAt(const Expr &expr, const size_t *indices)
    {
        return callAtImpl(expr, indices, std::make_index_sequence<N>{});
    }
};

#endif // CALL_AT_DISPATCH_H
