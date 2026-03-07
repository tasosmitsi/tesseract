#pragma once

namespace detail
{
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