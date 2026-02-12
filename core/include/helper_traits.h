#ifndef HELPERTRAITS_H
#define HELPERTRAITS_H

// Check if all values in a parameter pack are equal
template <my_size_t First, my_size_t... Rest>
consteval bool all_equal()
{
    return ((Rest == First) && ...);
}

// Check if all dimensions in two arrays match
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

// Compute the maximum of a pack at compile time
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

// Compute the minimum of a pack at compile time
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

// Helper wrapper for packs
template <my_size_t... Dims>
struct Pack
{
};

// Compare two `my_size_t` packs at compile time
template <my_size_t... A, my_size_t... B>
consteval bool packs_are_identical(Pack<A...>, Pack<B...>)
{
    if constexpr (sizeof...(A) != sizeof...(B))
    {
        return false; // different lengths
    }
    else
    {
        return ((A == B) && ...); // element-wise comparison
    }
}

// Compare two `my_size_t` packs at compile time
template <my_size_t... A, my_size_t... B>
consteval bool min_max_equal(Pack<A...>, Pack<B...>)
{
    if constexpr (sizeof...(A) != sizeof...(B))
        return false; // different lengths

    // the min and max of both packs do/don't match
    else
        return (max_value<A...>() == max_value<B...>()) && (min_value<A...>() == min_value<B...>());
}

// Check if all values in a pack are unique
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

template <my_size_t... Is>
struct index_seq
{
};

template <my_size_t N, my_size_t... Is>
struct make_index_seq : make_index_seq<N - 1, N - 1, Is...>
{
};

template <my_size_t... Is>
struct make_index_seq<0, Is...>
{
    using type = index_seq<Is...>;
};

#endif // HELPERTRAITS_H
