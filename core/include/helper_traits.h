#ifndef HELPERTRAITS_H
#define HELPERTRAITS_H

template <my_size_t First, my_size_t... Rest>
constexpr bool all_equal()
{
    return ((Rest == First) && ...);
}

#endif // HELPERTRAITS_H
