#pragma once

/**
 * @file NumericLimits.hpp
 * @brief STL-free replacement for std::numeric_limits.
 *
 * Provides compile-time min/max/infinity queries for fundamental
 * types using compiler built-in macros. Only the types needed by
 * the library are specialized.
 */

/**
 * @brief Compile-time numeric limits for a given type.
 * @tparam T Arithmetic type (must be explicitly specialized).
 */
template <typename T>
struct NumericLimits;

/// @cond
template <>
struct NumericLimits<float>
{
    static constexpr float max() noexcept { return __FLT_MAX__; }
    static constexpr float lowest() noexcept { return -__FLT_MAX__; }
    static constexpr float infinity() noexcept { return __builtin_huge_valf(); }
};

template <>
struct NumericLimits<double>
{
    static constexpr double max() noexcept { return __DBL_MAX__; }
    static constexpr double lowest() noexcept { return -__DBL_MAX__; }
    static constexpr double infinity() noexcept { return __builtin_huge_val(); }
};

template <>
struct NumericLimits<int>
{
    static constexpr int max() noexcept { return __INT_MAX__; }
    static constexpr int lowest() noexcept { return -__INT_MAX__ - 1; }
};

template <>
struct NumericLimits<long>
{
    static constexpr long max() noexcept { return __LONG_MAX__; }
    static constexpr long lowest() noexcept { return -__LONG_MAX__ - 1L; }
};

template <>
struct NumericLimits<unsigned int>
{
    static constexpr unsigned int max() noexcept { return __INT_MAX__ * 2U + 1U; }
    static constexpr unsigned int lowest() noexcept { return 0; }
};
/// @endcond