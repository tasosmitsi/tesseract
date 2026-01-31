#ifndef CONFIG_H
#define CONFIG_H

// Configuration file for FusedTensorND

#ifdef __GNUC__
#define FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline
#endif

#if __cplusplus >= 201103L
#define DEFINE_TYPE_ALIAS(type, name) using name = type
#else
#define DEFINE_TYPE_ALIAS(type, name) typedef type name
#endif

#include "error_handler/error_handler.h"
#ifdef ARDUINO
#include "error_handler/arduino_serial_error_handler.h"
using MyErrorHandler = ErrorHandler<SerialErrorHandler>;
#else
#include "error_handler/pc_error_handler.h"
using MyErrorHandler = ErrorHandler<PCErrorHandler>;
#endif

/*  Uncomment this line to enable debug output for FusedMatrix
    Commenting this out will disable debug output for FusedMatrix but
    will help compiler optimize the code */

// #define DEBUG_FUSED_MATRIX

/*  Uncomment the following line to enable debug output for FusedTensorND
    Commenting this out will disable debug output for FusedMatrix but
    will help compiler optimize the code */

// #define DEBUG_FUSED_TENSOR

/*  Comment on the runtime vs compile time checks. In order for the tests to run
    successfully we need exceptions, and for that reason we want the
    code to be able to fail in runtime. Hence, leave runtime
    checks on and compiletime checks off. */

/* Define this to enable compile time number of dimensions/indices checking */
// #define COMPILETIME_CHECK_DIMENSIONS_COUNT_MISMATCH

/* Define this to enable compile time size of dimensions checking */
// #define COMPILETIME_CHECK_DIMENSIONS_SIZE_MISMATCH

/* Define this to enable runtime number of dimensions/indices checking */
#define RUNTIME_CHECK_DIMENSIONS_COUNT_MISMATCH

/* Define this to enable runtime size of dimensions checking */
#define RUNTIME_CHECK_DIMENSIONS_SIZE_MISMATCH

/* Define this to enable matrix bound checking */
#define RUNTIME_USE_BOUNDS_CHECKING

// Define the precision tolerance for floating point comparisons
#define PRECISION_TOLERANCE 1e-9

// Define the type for size_t, can be uint32_t or uint64_t
#define my_size_t size_t

#endif // CONFIG_H
