#pragma once

#include "config.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#pragma message "[COMPILE-TIME] Using x86 cycle counter"
#include "cycle_counter_x86.h"
using CycleCounter = detail::CycleCounterX86;

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
#pragma message "[COMPILE-TIME] Using ARM cycle counter"
#include "cycle_counter_arm.h"
using CycleCounter = detail::CycleCounterArm;

#else
#pragma message "[COMPILE-TIME] Using generic cycle counter (fallback)"
#include "cycle_counter_generic.h"
using CycleCounter = detail::CycleCounterGeneric;
#endif