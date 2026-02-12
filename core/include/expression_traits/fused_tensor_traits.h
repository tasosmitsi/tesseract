#pragma once

template <typename T, my_size_t... Dims>
class FusedTensorND; // forward declarations

namespace expression
{
    template <typename T, my_size_t... Dims>
    struct traits<FusedTensorND<T, Dims...>>
    {
        static constexpr bool IsPermuted = false;
        static constexpr bool IsContiguous = true;
    };

} // namespace expression
