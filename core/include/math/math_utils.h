#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "config.h"             // For PRECISION_TOLERANCE and MyErrorHandler
#include "simple_type_traits.h" // For is_floating_point_v and is_same_v

/**
 * @file MathUtils.h
 * @brief STL-free replacements for common math functions.
 *
 * Delegates to compiler builtins which map directly to
 * hardware FP instructions where available.
 */

namespace math
{

    /**
     * @brief Compute the square root of a floating-point value.
     *
     * Maps to a single hardware instruction (`sqrtss`/`sqrtsd`) via
     * compiler builtins. Errors on negative input via MyErrorHandler.
     *
     * @tparam T Floating-point type (float or double).
     * @param x Non-negative input value.
     * @return Square root of @p x.
     */
    template <typename T>
    constexpr T sqrt(T x) noexcept
    {
        static_assert(is_floating_point_v<T>, "sqrt requires a floating-point type");
        if (x < T(0))
            MyErrorHandler::error("sqrt of negative value");
        return is_same_v<T, float> ? __builtin_sqrtf(x) : __builtin_sqrt(x);
    }

    /**
     * @brief Compute the absolute value of a numeric value.
     *
     * Maps to hardware instructions via compiler builtins for
     * float, double, int, and long. Falls back to branch for other types.
     *
     * @tparam T Numeric type.
     * @param x Input value.
     * @return Absolute value of @p x.
     */
    template <typename T>
    constexpr T abs(T x) noexcept
    {
        if constexpr (is_same_v<T, float>)
            return __builtin_fabsf(x);
        else if constexpr (is_same_v<T, double>)
            return __builtin_fabs(x);
        else if constexpr (is_same_v<T, int>)
            return __builtin_abs(x);
        else if constexpr (is_same_v<T, long>)
            return __builtin_labs(x);
        else
            return x < T(0) ? -x : x;
    }

} // namespace math

#endif // MATH_UTILS_H
