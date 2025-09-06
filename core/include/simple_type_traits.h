#ifndef SIMPLE_TYPE_TRAITS_HPP
#define SIMPLE_TYPE_TRAITS_HPP

// ===============================
// POD Type Traits
// ===============================

// Check if a type is POD (Plain Old Data)
// Default case: not POD
template <typename T> struct is_pod { static constexpr bool value = false; };

// // Specializations for fundamental types — define each type exactly once here:
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

// Helper variable for easier usage in if constexpr
template <typename T>
inline constexpr bool is_pod_v = is_pod<T>::value;

// ===============================
// Compile-time Type Comparison
// ===============================

// Default: types are not the same
template <typename A, typename B>
struct is_same { static constexpr bool value = false; };

// Specialization: types are the same
template <typename T>
struct is_same<T, T> { static constexpr bool value = true; };

// Helper variable for easier usage in if constexpr
template <typename A, typename B>
inline constexpr bool is_same_v = is_same<A, B>::value;

/*
Usage examples:

// 1. Check POD at compile-time
static_assert(is_pod<int>::value, "int is POD");
static_assert(!is_pod<void*>::value, "void* is not specialized POD");

// 2. Type comparison
static_assert(is_same<int, int>::value, "int == int");
static_assert(!is_same<int, float>::value, "int != float");

// 3. Using the helper variable
if constexpr (is_same_v<int, float>) {
    // This block won't compile/run
}

// 4. Generic usage in templates
template <typename T>
void foo() {
    if constexpr (is_pod_v<T>) {
        // Only for POD types
    }
}
*/

#endif // SIMPLE_TYPE_TRAITS_HPP
