#pragma once

namespace detail
{
    struct CycleCounterArm
    {
        unsigned long long start_cycles;
        unsigned long long total_cycles = 0;
        unsigned long long runs = 0;

        FORCE_INLINE void start() noexcept
        {
            unsigned long long val;
            asm volatile("mrs %0, cntvct_el0" : "=r"(val));
            start_cycles = val;
        }

        FORCE_INLINE void stop() noexcept
        {
            unsigned long long end;
            asm volatile("mrs %0, cntvct_el0" : "=r"(end));
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