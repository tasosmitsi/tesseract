#pragma once

#include "op_traits_generic.h" // The generic template (no default Arch)

// Add more architecture-specific includes here

#if defined(__ARM_NEON)
using DefaultArch = ARM_NEON;
#elif defined(__AVX__)
#pragma message "[COMPILE-TIME] Using X86_AVX arch"
// TODO: use this instead constexpr my_size_t AVX_BITS = 256;
#define BITS 256 // 128 or 256 bits
#include "op_traits_f_X86_AVX.h"
using DefaultArch = X86_AVX;
#else
using DefaultArch = GenericArch;
#endif
