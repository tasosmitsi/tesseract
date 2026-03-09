#ifndef HELPERTRAITS_H
#define HELPERTRAITS_H

/**
 * @file HelperTraits.h
 * @brief Compile-time utility functions and types for parameter pack manipulation.
 *
 * Provides consteval helpers for querying and comparing non-type template
 * parameter packs (dimensions, permutations, etc.) used throughout the
 * tensor library's type system.
 */

/**
 * @brief Check if all values in a parameter pack are equal.
 * @tparam First First value in the pack.
 * @tparam Rest  Remaining values.
 * @return True if every value equals @p First.
 */
template <my_size_t First, my_size_t... Rest>
consteval bool all_equal()
{
    return ((Rest == First) && ...);
}

/**
 * @brief Element-wise equality check of two compile-time arrays.
 * @tparam N Array length.
 * @param lhs First array.
 * @param rhs Second array.
 * @return True if all corresponding elements match.
 */
template <my_size_t N>
consteval bool dims_match(const my_size_t lhs[N], const my_size_t rhs[N])
{
    for (my_size_t i = 0; i < N; ++i)
    {
        if (lhs[i] != rhs[i])
            return false;
    }
    return true;
}

/**
 * @brief Compile-time maximum of a non-type parameter pack.
 * @tparam Vals One or more values (static_assert enforced).
 * @return The largest value in the pack.
 */
template <my_size_t... Vals>
consteval my_size_t max_value()
{
    static_assert(sizeof...(Vals) > 0, "max_value requires at least one value");
    my_size_t arr[] = {Vals...};
    my_size_t result = arr[0];
    for (my_size_t i = 1; i < sizeof...(Vals); ++i)
    {
        if (arr[i] > result)
            result = arr[i];
    }
    return result;
}

/**
 * @brief Compile-time minimum of a non-type parameter pack.
 * @tparam Vals One or more values (static_assert enforced).
 * @return The smallest value in the pack.
 */
template <my_size_t... Vals>
consteval my_size_t min_value()
{
    static_assert(sizeof...(Vals) > 0, "min_value requires at least one value");
    my_size_t arr[] = {Vals...};
    my_size_t result = arr[0];
    for (my_size_t i = 1; i < sizeof...(Vals); ++i)
    {
        if (arr[i] < result)
            result = arr[i];
    }
    return result;
}

/**
 * @brief Wrapper struct for carrying a non-type parameter pack.
 * @tparam Dims Values in the pack.
 *
 * Used as a tag type so that pack contents can be passed to
 * and deduced by regular function templates (e.g. packs_are_identical()).
 */
template <my_size_t... Dims>
struct Pack
{
};

/**
 * @brief Element-wise equality comparison of two packs.
 * @tparam A Values in the first pack.
 * @tparam B Values in the second pack.
 * @return True if both packs have the same length and identical elements.
 */
template <my_size_t... A, my_size_t... B>
consteval bool packs_are_identical(Pack<A...>, Pack<B...>)
{
    if constexpr (sizeof...(A) != sizeof...(B))
    {
        return false;
    }
    else
    {
        return ((A == B) && ...);
    }
}

/**
 * @brief Check if two packs have the same min and max values, regardless of order.
 * @tparam A Values in the first pack.
 * @tparam B Values in the second pack.
 * @return True if min and max match, false otherwise.
 */
template <my_size_t... A, my_size_t... B>
consteval bool same_min_max(Pack<A...>, Pack<B...>)
{
    if constexpr (sizeof...(A) != sizeof...(B))
        return false;
    else
        return (max_value<A...>() == max_value<B...>()) && (min_value<A...>() == min_value<B...>());
}

/**
 * @brief Check if all values in a pack are unique.
 * @tparam Vals One or more values (static_assert enforced).
 * @return True if no two values are equal (O(N²) comparison).
 */
template <my_size_t... Vals>
consteval bool all_unique()
{
    static_assert(sizeof...(Vals) > 0, "all_unique requires at least one value");
    my_size_t arr[] = {Vals...};
    for (my_size_t i = 0; i < sizeof...(Vals); ++i)
    {
        for (my_size_t j = i + 1; j < sizeof...(Vals); ++j)
        {
            if (arr[i] == arr[j])
                return false;
        }
    }
    return true;
}

/**
 * @brief Check if a pack forms the identity permutation {0, 1, …, N−1}.
 * @tparam Vals One or more values (static_assert enforced).
 * @return True if `Vals[i] == i` for all i.
 */
template <my_size_t... Vals>
consteval bool is_sequential()
{
    static_assert(sizeof...(Vals) > 0, "is_sequential requires at least one value");
    my_size_t arr[] = {Vals...};
    for (my_size_t i = 0; i < sizeof...(Vals); ++i)
    {
        if (arr[i] != i)
            return false;
    }
    return true;
}

/**
 * @brief Compile-time index sequence (lightweight std::index_sequence alternative).
 * @tparam Is Index values.
 */
template <my_size_t... Is>
struct index_seq
{
};

/**
 * @brief Recursive generator for index_seq<0, 1, …, N−1>.
 * @tparam N Desired sequence length.
 * @tparam Is Accumulated indices (internal).
 *
 * Usage: `typename make_index_seq<N>::type` yields `index_seq<0, …, N−1>`.
 */
template <my_size_t N, my_size_t... Is>
struct make_index_seq : make_index_seq<N - 1, N - 1, Is...>
{
};

/// @cond
template <my_size_t... Is>
struct make_index_seq<0, Is...>
{
    using type = index_seq<Is...>;
};
/// @endcond

#endif // HELPERTRAITS_H