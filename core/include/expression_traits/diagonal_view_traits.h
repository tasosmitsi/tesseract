#pragma once

#include "helper_traits.h"

template <typename Matrix>
class DiagonalView; // forward declarations

namespace expression
{
    template <typename Matrix>
    struct traits<DiagonalView<Matrix>>
    {
        static constexpr bool IsPermuted = true; // TODO: Semantically "permuted" is misleading — the data isn't permuted, the output just needs slice-aware stores.
        static constexpr bool IsContiguous = false;
        static constexpr bool IsPhysical = true;
    };
} // namespace expression
