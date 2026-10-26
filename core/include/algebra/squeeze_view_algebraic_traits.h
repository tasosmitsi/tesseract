#pragma once

#include "config.h"
#include "algebra/basic_algebraic_traits.h"

template <typename Tensor, my_size_t... DroppedAxes>
class SqueezeView; // forward declarations

namespace algebra
{
    template <typename Tensor, my_size_t... DroppedAxes>
    struct algebraic_traits<SqueezeView<Tensor, DroppedAxes...>>
    {
        static constexpr bool vector_space = true;
        static constexpr bool algebra = false;
        static constexpr bool lie_group = false;
        static constexpr bool metric = false;
        static constexpr bool tensor = true;
    };

} // namespace algebra
