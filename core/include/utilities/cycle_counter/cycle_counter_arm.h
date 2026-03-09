#pragma once

/**
 * @file cycle_counter_arm.h
 * @brief Cycle counter using the ARM generic timer (CNTVCT_EL0).
 */

namespace detail
{

    /**
     * @brief Hardware cycle counter for AArch64/ARM.
     *
     * Reads the virtual count register (`cntvct_el0`) via inline
     * assembly. Note that this returns timer ticks, not raw CPU
     * cycles — the tick frequency is given by `cntfrq_el0`.
     */
    struct CycleCounterArm
    {
        unsigned long long start_cycles;
        unsigned long long total_cycles = 0;
        unsigned long long runs = 0;

        /** @brief Record the starting tick count. */
        FORCE_INLINE void start() noexcept
        {
            unsigned long long val;
            asm volatile("mrs %0, cntvct_el0" : "=r"(val));
            start_cycles = val;
        }

        /** @brief Record the ending tick count and accumulate. */
        FORCE_INLINE void stop() noexcept
        {
            unsigned long long end;
            asm volatile("mrs %0, cntvct_el0" : "=r"(end));
            total_cycles += (end - start_cycles);
            ++runs;
        }

        /** @brief Reset accumulated ticks and run count to zero. */
        void reset() noexcept
        {
            total_cycles = 0;
            runs = 0;
        }

        /** @brief Return the average ticks per start/stop pair. */
        double avg_cycles() const { return static_cast<double>(total_cycles) / runs; }
    };

} // namespace detail