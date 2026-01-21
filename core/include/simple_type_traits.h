#ifndef SIMPLE_TYPE_TRAITS_HPP
#define SIMPLE_TYPE_TRAITS_HPP

// ===============================
// POD Type Traits
// ===============================

// Check if a type is POD (Plain Old Data)
// Default case: not POD
template <typename T>
struct is_pod
{
    static constexpr bool value = false;
};

// // Specializations for fundamental types — define each type exactly once here:
template <>
struct is_pod<char>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<unsigned char>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<short>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<unsigned short>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<int>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<unsigned int>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<long>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<unsigned long>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<float>
{
    static constexpr bool value = true;
};

template <>
struct is_pod<double>
{
    static constexpr bool value = true;
};

// Helper variable for easier usage in if constexpr
template <typename T>
inline constexpr bool is_pod_v = is_pod<T>::value;

// ===============================
// Compile-time Type Comparison
// ===============================

// Default: types are not the same
template <typename A, typename B>
struct is_same
{
    static constexpr bool value = false;
};

// Specialization: types are the same
template <typename T>
struct is_same<T, T>
{
    static constexpr bool value = true;
};

// Helper variable for easier usage in if constexpr
template <typename A, typename B>
inline constexpr bool is_same_v = is_same<A, B>::value;

// ===============================
// Remove Reference Qualifiers
// ===============================

template <typename T>
struct remove_reference
{
    using type = T;
};

template <typename T>
struct remove_reference<T &>
{
    using type = T;
};

template <typename T>
struct remove_reference<T &&>
{
    using type = T;
};

template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

// ===============================
// Remove CV Qualifiers
// ===============================

template <typename T>
struct remove_cv
{
    using type = T;
};

template <typename T>
struct remove_cv<const T>
{
    using type = T;
};

template <typename T>
struct remove_cv<volatile T>
{
    using type = T;
};

template <typename T>
struct remove_cv<const volatile T>
{
    using type = T;
};

template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

// ===============================
// Remove CV and Reference Qualifiers
// ===============================
template <typename T>
struct remove_cvref
{
    using type = remove_cv_t<remove_reference_t<T>>;
};

// Helper type alias
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

// ===============================
// std::move Replacement
// ===============================

template <typename T>
[[nodiscard]] constexpr remove_reference_t<T> &&move(T &&t) noexcept
{
    return static_cast<remove_reference_t<T> &&>(t);
}

// ===============================
// Compile-time Base Class Check
// ===============================

template <typename Base, typename Derived>
struct is_base_of
{
#if defined(__has_builtin)
#if __has_builtin(__is_base_of)
    static constexpr bool value = __is_base_of(Base, Derived);
#else
    // fallback implementation
#endif
#elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
    static constexpr bool value = __is_base_of(Base, Derived);
#else
    // fallback implementation
#endif
};

template <typename Base, typename Derived>
inline constexpr bool is_base_of_v = is_base_of<Base, Derived>::value;

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

// 5. Remove CV and Reference qualifiers
using CleanType = remove_cvref_t<const volatile int&>; // CleanType is int

// 6. Check base class relationship
static_assert(is_base_of_v<std::exception, std::runtime_error>, "std::exception is base of std::runtime_error");

// 7. Using move
std::string str = "Hello";
std::string newStr = move(str); // Transfers ownership
*/

#endif // SIMPLE_TYPE_TRAITS_HPP
