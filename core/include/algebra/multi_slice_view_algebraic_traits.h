#pragma once

template <typename Tensor, typename... Slices>
class MultiSliceView; // forward declarations

namespace algebra
{
    template <typename Tensor, typename... Slices>
    struct algebraic_traits<MultiSliceView<Tensor, Slices...>>
    {
        static constexpr bool vector_space = true;
        static constexpr bool algebra = false;
        static constexpr bool lie_group = false;
        static constexpr bool metric = false;
        static constexpr bool tensor = true;
    };

} // namespace algebra
