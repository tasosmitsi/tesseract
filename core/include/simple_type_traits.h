#ifndef SIMPLE_TYPE_TRAITS_HPP
#define SIMPLE_TYPE_TRAITS_HPP

#include <cstddef> // std::size_t

template <typename T> struct is_pod { static constexpr bool value = false; };

// Specializations — define each type exactly once here:
// Basic fundamental types only once
template <> struct is_pod<char>             { static constexpr bool value = true; };
template <> struct is_pod<unsigned char>    { static constexpr bool value = true; };
template <> struct is_pod<short>            { static constexpr bool value = true; };
template <> struct is_pod<unsigned short>   { static constexpr bool value = true; };
template <> struct is_pod<int>              { static constexpr bool value = true; };
template <> struct is_pod<unsigned int>     { static constexpr bool value = true; };
template <> struct is_pod<long>             { static constexpr bool value = true; };
template <> struct is_pod<unsigned long>    { static constexpr bool value = true; };
template <> struct is_pod<float>            { static constexpr bool value = true; };
template <> struct is_pod<double>           { static constexpr bool value = true; };
#endif // SIMPLE_TYPE_TRAITS_HPP
