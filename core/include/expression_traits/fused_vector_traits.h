#pragma once

template <typename T, my_size_t Size>
class FusedVector; // forward declarations

namespace expression
{
    template <typename T, my_size_t Size>
    struct traits<FusedVector<T, Size>>
    {
        static constexpr bool IsPermuted = false;
        static constexpr bool IsContiguous = true;
    };

} // namespace expression
