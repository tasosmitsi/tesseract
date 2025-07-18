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

// Uncomment the following line to enable debug output for FusedTensorND
// #define DEBUG_FUSED_TENSOR

/* Define this to enable matrix number of indices checking */
#define MATRIX_USE_NUMBER_OF_INDICES_CHECKING

/* Define this to enable matrix bound checking */
#define MATRIX_USE_BOUNDS_CHECKING

// Define the precision tolerance for floating point comparisons
#define PRECISION_TOLERANCE 1e-9

// Define the type for size_t, can be uint32_t or uint64_t
#define my_size_t size_t

#endif // CONFIG_H
