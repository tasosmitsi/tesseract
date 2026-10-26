#pragma once

#include "config.h"
#include "expression_traits/basic_expr_traits.h"

template <typename Tensor, my_size_t... DroppedAxes>
class SqueezeView; // forward declarations

namespace expression
{
    template <typename Tensor, my_size_t... DroppedAxes>
    struct traits<SqueezeView<Tensor, DroppedAxes...>>
    {
        static constexpr bool IsPermuted = true; // TODO: Semantically "permuted" is misleading — the data isn't permuted, the output just needs slice-aware stores.
        static constexpr bool IsContiguous = false;
        static constexpr bool IsPhysical = true;
        static constexpr bool OwnsMemory = false;
    };
} // namespace expression
