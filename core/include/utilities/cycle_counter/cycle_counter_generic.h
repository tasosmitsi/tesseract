#pragma once

/**
 * @file cycle_counter_generic.h
 * @brief No-op cycle counter fallback for unsupported platforms.
 */

namespace detail
{

    /**
     * @brief Stub cycle counter for platforms without hardware timer access.
     *
     * All timing methods are no-ops. avg_cycles() always returns 0.
     * Exists so that benchmarking code compiles unconditionally.
     */
    struct CycleCounterGeneric
    {
        unsigned long long total_cycles = 0;
        unsigned long long runs = 0;

        FORCE_INLINE void start() noexcept {}
        FORCE_INLINE void stop() noexcept { ++runs; }

        void reset() noexcept
        {
            total_cycles = 0;
            runs = 0;
        }

        double avg_cycles() const { return 0.0; }
    };

} // namespace detail