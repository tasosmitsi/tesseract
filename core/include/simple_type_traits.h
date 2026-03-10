#ifndef SIMPLE_TYPE_TRAITS_HPP
#define SIMPLE_TYPE_TRAITS_HPP

/**
 * @file SimpleTypeTraits.hpp
 * @brief STL-free type traits and utility functions.
 *
 * Provides lightweight, zero-dependency replacements for common
 * `<type_traits>` and `<utility>` facilities, suitable for
 * freestanding / embedded targets where the standard library
 * is unavailable.
 */

/**
 * @brief Compile-time check for Plain Old Data types.
 * @tparam T Type to test.
 *
 * The primary template yields `false`; explicit specializations
 * below whitelist the fundamental arithmetic types.
 */
template <typename T>
struct is_pod
{
    static constexpr bool value = false;
};

// clang-format off
/// @cond
template <> struct is_pod<char>           { static constexpr bool value = true; };
template <> struct is_pod<unsigned char>  { static constexpr bool value = true; };
template <> struct is_pod<short>          { static constexpr bool value = true; };
template <> struct is_pod<unsigned short> { static constexpr bool value = true; };
template <> struct is_pod<int>            { static constexpr bool value = true; };
template <> struct is_pod<unsigned int>   { static constexpr bool value = true; };
template <> struct is_pod<long>           { static constexpr bool value = true; };
template <> struct is_pod<unsigned long>  { static constexpr bool value = true; };
template <> struct is_pod<float>          { static constexpr bool value = true; };
template <> struct is_pod<double>         { static constexpr bool value = true; };
/// @endcond
// clang-format on

/** @brief Helper variable template for is_pod. */
template <typename T>
inline constexpr bool is_pod_v = is_pod<T>::value;

/**
 * @brief Compile-time floating-point type check.
 * @tparam T Type to test.
 *
 * The primary template yields `false`; explicit specializations
 * below whitelist `float` and `double`.
 */
template <typename T>
struct is_floating_point
{
    static constexpr bool value = false;
};

// clang-format off
/// @cond
template <> struct is_floating_point<float>  { static constexpr bool value = true; };
template <> struct is_floating_point<double> { static constexpr bool value = true; };
/// @endcond
// clang-format on

/** @brief Helper variable template for is_floating_point. */
template <typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

/**
 * @brief Compile-time type equality check (replacement for std::is_same).
 * @tparam A First type.
 * @tparam B Second type.
 */
template <typename A, typename B>
struct is_same
{
    static constexpr bool value = false;
};

/// @cond
template <typename T>
struct is_same<T, T>
{
    static constexpr bool value = true;
};
/// @endcond

/** @brief Helper variable template for is_same. */
template <typename A, typename B>
inline constexpr bool is_same_v = is_same<A, B>::value;

/**
 * @brief Strip lvalue/rvalue reference qualifiers from a type.
 * @tparam T Possibly-referenced type.
 */
template <typename T>
struct remove_reference
{
    using type = T;
};

/// @cond
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
/// @endcond

/** @brief Alias template for remove_reference. */
template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

/**
 * @brief Strip const/volatile qualifiers from a type.
 * @tparam T Possibly cv-qualified type.
 */
template <typename T>
struct remove_cv
{
    using type = T;
};

/// @cond
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
/// @endcond

/** @brief Alias template for remove_cv. */
template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

/**
 * @brief Strip cv-qualifiers and references in one step.
 * @tparam T Input type.
 *
 * Equivalent to `remove_cv_t<remove_reference_t<T>>`.
 */
template <typename T>
struct remove_cvref
{
    using type = remove_cv_t<remove_reference_t<T>>;
};

/** @brief Alias template for remove_cvref. */
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

/**
 * @brief Cast to rvalue reference (replacement for std::move).
 * @tparam T Deduced argument type.
 * @param t Value to move from.
 * @return An xvalue reference to @p t.
 */
template <typename T>
[[nodiscard]] constexpr remove_reference_t<T> &&move(T &&t) noexcept
{
    return static_cast<remove_reference_t<T> &&>(t);
}

/**
 * @brief Perfect-forward an lvalue (replacement for std::forward).
 * @tparam T Forwarding reference type.
 * @param t Lvalue reference to forward.
 */
template <typename T>
constexpr T &&forward(remove_reference_t<T> &t) noexcept
{
    return static_cast<T &&>(t);
}

/**
 * @brief Perfect-forward an rvalue (replacement for std::forward).
 * @tparam T Forwarding reference type.
 * @param t Rvalue reference to forward.
 */
template <typename T>
constexpr T &&forward(remove_reference_t<T> &&t) noexcept
{
    return static_cast<T &&>(t);
}

/**
 * @brief Compile-time inheritance check (replacement for std::is_base_of).
 * @tparam Base    Candidate base class.
 * @tparam Derived Candidate derived class.
 *
 * Uses the compiler intrinsic `__is_base_of` when available,
 * otherwise falls back to a pointer-conversion SFINAE check.
 *
 * @note The fallback does not detect private/ambiguous inheritance
 *       (the intrinsic does). This is acceptable for the library's
 *       use cases where inheritance is always public.
 */
template <typename Base, typename Derived>
struct is_base_of
{
#if defined(__has_builtin) && __has_builtin(__is_base_of)
    static constexpr bool value = __is_base_of(Base, Derived);
#elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
    static constexpr bool value = __is_base_of(Base, Derived);
#else
    // Fallback implementation (not fully standards-compliant)
    static constexpr bool value = detail::is_base_of_impl<Base, Derived>::value;
#endif
};

/** @brief Helper variable template for is_base_of. */
template <typename Base, typename Derived>
inline constexpr bool is_base_of_v = is_base_of<Base, Derived>::value;

namespace detail
{

    /**
     * @brief Fallback implementation for is_base_of using pointer conversion.
     * @tparam Base    Candidate base class.
     * @tparam Derived Candidate derived class.
     *
     * Exploits implicit `Derived* → Base*` conversion to detect inheritance.
     * Does not handle private or ambiguous inheritance.
     */
    template <typename Base, typename Derived>
    struct is_base_of_impl
    {
    private:
        static constexpr bool check(Base *) { return true; }
        static constexpr bool check(...) { return false; }

    public:
        static constexpr bool value = check(static_cast<Derived *>(nullptr));
    };

} // namespace detail

/**
 * @brief Compile-time check for trivially destructible types.
 *
 * Uses the compiler intrinsic `__is_trivially_destructible` (available on
 * GCC, Clang, and MSVC) to avoid depending on \<type_traits\>.
 *
 * @tparam T Type to test.
 */
template <typename T>
struct is_trivially_destructible
{
    static constexpr bool value = __has_trivial_destructor(T);
};

/** @brief Helper variable template for is_trivially_destructible. */
template <typename T>
inline constexpr bool is_trivially_destructible_v = is_trivially_destructible<T>::value;

/**
 * @brief Compile-time check for nothrow move constructibility.
 *
 * Uses the compiler intrinsic `__is_nothrow_constructible` (available on
 * GCC, Clang, and MSVC) to avoid depending on \<type_traits\>.
 *
 * @tparam T Type to test.
 */
template <typename T>
struct is_nothrow_move_constructible
{
    static constexpr bool value = __is_nothrow_constructible(T, T &&);
};

/** @brief Helper variable template for is_nothrow_move_constructible. */
template <typename T>
inline constexpr bool is_nothrow_move_constructible_v = is_nothrow_move_constructible<T>::value;

/**
 * @brief Placement new for constructing objects at an existing address.
 *
 * On freestanding targets (where \<new\> is unavailable), this provides
 * the placement new operator directly. On hosted targets, we include
 * \<new\> to avoid redefinition conflicts with the standard library.
 */
#if defined(__STDC_HOSTED__) && __STDC_HOSTED__ == 0
inline void *operator new(decltype(sizeof(0)), void *ptr) noexcept { return ptr; }
#else
#include <new>
#endif

#endif // SIMPLE_TYPE_TRAITS_HPP
