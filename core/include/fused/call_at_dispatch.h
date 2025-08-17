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

#endif // CALL_AT_DISPATCH_H
