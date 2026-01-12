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
template <my_size_t First, my_size_t... Rest>
consteval my_size_t max_value()
{
    if constexpr (sizeof...(Rest) == 0)
        return First;
    else
    {
        my_size_t tail_max = max_value<Rest...>();
        return (First > tail_max ? First : tail_max);
    }
}

// Compute the minimum of a pack at compile time
template <my_size_t First, my_size_t... Rest>
consteval my_size_t min_value()
{
    if constexpr (sizeof...(Rest) == 0)
        return First;
    else
    {
        my_size_t tail_min = min_value<Rest...>();
        return (First < tail_min ? First : tail_min);
    }
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

#endif // HELPERTRAITS_H
