#ifndef CONFIG_H
#define CONFIG_H

/**
 * @file config.h
 * @brief Global configuration for the tesseract tensor library.
 *
 * Controls platform detection, error handling backend selection,
 * debug output, compile-time vs runtime safety checks, and
 * numeric precision settings. Intended to be included (directly
 * or indirectly) by every translation unit in the library.
 */

/**
 * @def FORCE_INLINE
 * @brief Hint the compiler to always inline a function.
 *
 * Resolves to `__attribute__((always_inline))` on GCC/Clang,
 * `__forceinline` on MSVC, and plain `inline` elsewhere.
 */
#ifdef __GNUC__
#define FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline
#endif

/**
 * @def DEFINE_TYPE_ALIAS(type, name)
 * @brief Portable type alias macro.
 *
 * Uses `using` on C++11 and later, falls back to `typedef` for
 * pre-C++11 compilers.
 */
#if __cplusplus >= 201103L
#define DEFINE_TYPE_ALIAS(type, name) using name = type
#else
#define DEFINE_TYPE_ALIAS(type, name) typedef type name
#endif

/**
 * @brief Platform-specific error handler selection.
 *
 * On Arduino targets, errors are reported over serial via
 * SerialErrorHandler. On desktop/PC targets, PCErrorHandler
 * is used (typically throws exceptions or prints to stderr).
 */
#include "error_handler/error_handler.h"
#ifdef ARDUINO
#include "error_handler/arduino_serial_error_handler.h"
using MyErrorHandler = ErrorHandler<SerialErrorHandler>;
#else
#include "error_handler/pc_error_handler.h"
using MyErrorHandler = ErrorHandler<PCErrorHandler>;
#define TESSERACT_CONDITIONAL_NOEXCEPT
#endif

/**
 * @def DEBUG_FUSED_MATRIX
 * @brief Enable verbose debug output for FusedMatrix operations.
 *
 * Leave commented out in release builds to allow the compiler
 * to eliminate dead debug branches entirely.
 */
// #define DEBUG_FUSED_MATRIX

/**
 * @def DEBUG_FUSED_TENSOR
 * @brief Enable verbose debug output for FusedTensorND operations.
 */
// #define DEBUG_FUSED_TENSOR

/**
 * @name Compile-Time Checks
 * @brief Static assertions on tensor dimensions.
 *
 * Disabled by default so that test suites can exercise runtime
 * error paths. Enable for maximum safety in production builds.
 * @{
 */

/** @def COMPILETIME_CHECK_DIMENSIONS_COUNT_MISMATCH
 *  @brief Reject mismatched number of indices/dimensions at compile time. */
// #define COMPILETIME_CHECK_DIMENSIONS_COUNT_MISMATCH

/** @def COMPILETIME_CHECK_DIMENSIONS_SIZE_MISMATCH
 *  @brief Reject mismatched dimension sizes at compile time. */
// #define COMPILETIME_CHECK_DIMENSIONS_SIZE_MISMATCH

/** @} */

/**
 * @name Runtime Checks
 * @brief Dynamic bounds and dimension validation.
 * @{
 */

/** @def RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH
 *  @brief Check number of indices/dimensions at runtime. */
#define RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH

/** @def RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH
 *  @brief Check dimension sizes at runtime. */
#define RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH

/** @def RUNTIME_USE_BOUNDS_CHECKING
 *  @brief Enable element-access bounds checking. */
#define RUNTIME_USE_BOUNDS_CHECKING

/** @} */

/**
 * @def PRECISION_TOLERANCE
 * @brief Tolerance for floating-point comparisons (e.g. symmetry checks, Cholesky).
 */
#define PRECISION_TOLERANCE 1e-9

/**
 * @def my_size_t
 * @brief Size/index type used throughout the library.
 *
 * Defaults to `size_t`. Can be changed to `uint32_t` for
 * memory-constrained targets.
 */
#define my_size_t size_t

/**
 * @def TESSERACT_USE_FMAD
 * @brief Enable fused multiply-add pattern detection in expression evaluation.
 */
#define TESSERACT_USE_FMAD

#endif // CONFIG_H