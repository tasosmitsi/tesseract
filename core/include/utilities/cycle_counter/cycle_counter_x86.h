#pragma once

#include <x86intrin.h>

/**
 * @file cycle_counter_x86.h
 * @brief Cycle counter using x86 RDTSC with LFENCE serialization.
 */

namespace detail
{

    /**
     * @brief Hardware cycle counter for x86/x86_64.
     *
     * Uses `__rdtsc()` for timestamp reads, with `_mm_lfence()`
     * to serialize instruction execution and prevent out-of-order
     * measurement artifacts.
     */
    struct CycleCounterX86
    {
        unsigned long long start_cycles;
        unsigned long long total_cycles = 0;
        unsigned long long runs = 0;

        /** @brief Record the starting cycle count (serialized). */
        FORCE_INLINE void start() noexcept
        {
            _mm_lfence();
            start_cycles = __rdtsc();
        }

        /** @brief Record the ending cycle count and accumulate. */
        FORCE_INLINE void stop() noexcept
        {
            unsigned long long end = __rdtsc();
            _mm_lfence();
            total_cycles += (end - start_cycles);
            ++runs;
        }

        /** @brief Reset accumulated cycles and run count to zero. */
        void reset() noexcept
        {
            total_cycles = 0;
            runs = 0;
        }

        /** @brief Return the average cycles per start/stop pair. */
        double avg_cycles() const { return static_cast<double>(total_cycles) / runs; }
    };

} // namespace detail