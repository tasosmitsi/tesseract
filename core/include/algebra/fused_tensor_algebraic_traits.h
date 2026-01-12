#pragma once

template <typename T, my_size_t... Dims>
class FusedTensorND; // forward declarations

namespace algebra
{
    template <typename T, my_size_t... Dims>
    struct algebraic_traits<FusedTensorND<T, Dims...>>
    {
        static constexpr bool vector_space = true; // q + q, q * scalar
        static constexpr bool algebra = false;     // Hamilton product
        static constexpr bool lie_group = false;   // not unit length
        static constexpr bool metric = false;      // dot, norm
        static constexpr bool tensor = true;       // NOT shape-based
    };

} // namespace algebra
