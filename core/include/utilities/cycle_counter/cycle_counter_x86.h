#pragma once

#include <x86intrin.h>

namespace detail
{
    struct CycleCounterX86
    {
        unsigned long long start_cycles;
        unsigned long long total_cycles = 0;
        unsigned long long runs = 0;

        FORCE_INLINE void start() noexcept
        {
            _mm_lfence();
            start_cycles = __rdtsc();
        }

        FORCE_INLINE void stop() noexcept
        {
            unsigned long long end = __rdtsc();
            _mm_lfence();
            total_cycles += (end - start_cycles);
            ++runs;
        }

        void reset() noexcept
        {
            total_cycles = 0;
            runs = 0;
        }

        double avg_cycles() const { return static_cast<double>(total_cycles) / runs; }
    };
} // namespace detail