#pragma once

template <typename Matrix>
class DiagonalView; // forward declarations

namespace algebra
{
    template <typename Matrix>
    struct algebraic_traits<DiagonalView<Matrix>>
    {
        static constexpr bool vector_space = true;
        static constexpr bool algebra = false;
        static constexpr bool lie_group = false;
        static constexpr bool metric = false;
        static constexpr bool tensor = true;
    };

} // namespace algebra
