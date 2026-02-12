#pragma once

#include "helper_traits.h"

template <typename Tensor, my_size_t... Perm>
class PermutedViewConstExpr; // forward declarations

namespace expression
{
    template <typename Tensor, my_size_t... Perm>
    struct traits<PermutedViewConstExpr<Tensor, Perm...>>
    {
        static constexpr bool IsPermuted = !is_sequential<Perm...>();

        static constexpr bool IsContiguous = !IsPermuted;
    };
} // namespace expression
