#pragma once

template <typename T, my_size_t Size>
class FusedVector; // forward declarations

namespace algebra
{
    template <typename T, my_size_t Size>
    struct algebraic_traits<FusedVector<T, Size>>
    {
        static constexpr bool vector_space = true; // q + q, q * scalar
        static constexpr bool algebra = false;     // Hamilton product
        static constexpr bool lie_group = false;   // not unit length
        static constexpr bool metric = false;      // dot, norm
        static constexpr bool tensor = true;       // NOT shape-based
    };

} // namespace algebra
