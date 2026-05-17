#pragma once

#include "helper_traits.h"

template <typename Vector, my_size_t Offset, my_size_t Len>
class SubVectorView; // forward declarations

namespace expression
{
    template <typename Vector, my_size_t Offset, my_size_t Len>
    struct traits<SubVectorView<Vector, Offset, Len>>
    {
        static constexpr bool IsPermuted = true; // TODO: Semantically "permuted" is misleading — the data isn't permuted, the output just needs slice-aware stores.
        static constexpr bool IsContiguous = false;
        static constexpr bool IsPhysical = true;
    };
} // namespace expression
