#pragma once

template <typename T, my_size_t Rows, my_size_t Cols>
class FusedMatrix; // forward declarations

namespace expression
{
    template <typename T, my_size_t Rows, my_size_t Cols>
    struct traits<FusedMatrix<T, Rows, Cols>>
    {
        static constexpr bool IsPermuted = false;
        static constexpr bool IsContiguous = true;
    };

} // namespace expression
