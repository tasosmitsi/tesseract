#ifndef STATIC_STORAGE_H
#define STATIC_STORAGE_H

template <typename T, size_t Size>
class StaticStorage
{
    T _data[Size];

public:
    FORCE_INLINE constexpr T &operator[](size_t idx) noexcept { return _data[idx]; }
    FORCE_INLINE constexpr const T &operator[](size_t idx) const noexcept { return _data[idx]; }

    FORCE_INLINE constexpr T *data() noexcept { return _data; }
    FORCE_INLINE constexpr const T *data() const noexcept { return _data; }

    FORCE_INLINE constexpr T *begin() noexcept { return _data; }
    FORCE_INLINE constexpr const T *begin() const noexcept { return _data; }

    FORCE_INLINE constexpr T *end() noexcept { return _data + Size; }
    FORCE_INLINE constexpr const T *end() const noexcept { return _data + Size; }
};
#endif // STATIC_STORAGE_H
