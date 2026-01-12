#pragma once

template <typename Tensor, my_size_t N>
class PermutedView; // forward declarations

namespace algebra
{
    template <typename Tensor, my_size_t N>
    struct algebraic_traits<PermutedView<Tensor, N>>
    {
        static constexpr bool vector_space = true; // q + q, q * scalar
        static constexpr bool algebra = false;     // Hamilton product
        static constexpr bool lie_group = false;   // not unit length
        static constexpr bool metric = false;      // dot, norm
        static constexpr bool tensor = true;       // NOT shape-based
    };

} // namespace algebra
