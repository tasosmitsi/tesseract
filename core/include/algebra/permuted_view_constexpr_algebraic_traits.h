#pragma once

template <typename Tensor, my_size_t... Perm>
class PermutedViewConstExpr; // forward declarations

namespace algebra
{
    template <typename Tensor, my_size_t... Perm>
    struct algebraic_traits<PermutedViewConstExpr<Tensor, Perm...>>
    {
        static constexpr bool vector_space = true; // q + q, q * scalar
        static constexpr bool algebra = false;     // Hamilton product
        static constexpr bool lie_group = false;   // not unit length
        static constexpr bool metric = false;      // dot, norm
        static constexpr bool tensor = true;       // NOT shape-based
    };

} // namespace algebra
