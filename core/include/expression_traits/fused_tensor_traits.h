#pragma once

#include "config.h"
#include "expression_traits/basic_expr_traits.h"

template <typename T, my_size_t... Dims>
class FusedTensorND; // forward declarations

namespace expression
{
    template <typename T, my_size_t... Dims>
    struct traits<FusedTensorND<T, Dims...>>
    {
        static constexpr bool IsPermuted = false;
        static constexpr bool IsContiguous = true;
        static constexpr bool IsPhysical = true;
        static constexpr bool OwnsMemory = true; // held by reference in expressions, see fused/expr_storage.h
    };

} // namespace expression
