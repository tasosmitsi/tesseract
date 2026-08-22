#pragma once

#include "helper_traits.h"

template <typename Tensor, typename... Slices>
class MultiSliceView; // forward declarations

namespace expression
{
    template <typename Tensor, typename... Slices>
    struct traits<MultiSliceView<Tensor, Slices...>>
    {
        static constexpr bool IsPermuted = true; // TODO: Semantically "permuted" is misleading — the data isn't permuted, the output just needs slice-aware stores.
        static constexpr bool IsContiguous = false;
        static constexpr bool IsPhysical = true;
    };
} // namespace expression
