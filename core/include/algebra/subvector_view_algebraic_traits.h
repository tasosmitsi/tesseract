#pragma once

template <typename Vector, my_size_t Offset, my_size_t Len>
class SubVectorView; // forward declarations

namespace algebra
{
    template <typename Vector, my_size_t Offset, my_size_t Len>
    struct algebraic_traits<SubVectorView<Vector, Offset, Len>>
    {
        static constexpr bool vector_space = true;
        static constexpr bool algebra = false;
        static constexpr bool lie_group = false;
        static constexpr bool metric = false;
        static constexpr bool tensor = true;
    };

} // namespace algebra
