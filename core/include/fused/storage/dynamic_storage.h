#ifndef DYNAMIC_STORAGE_H
#define DYNAMIC_STORAGE_H

#include <cstdlib>   // for malloc, free
#include <cstring>   // for memcpy
#include <stdexcept> // for exceptions
#include <utility>   // for std::move

template <typename T, my_size_t N>
class DynamicStorage
{
public:
    // Constructor: allocate memory for N elements
    DynamicStorage()
    {
        // _data = static_cast<T *>(std::malloc(N * sizeof(T)));
        _data = static_cast<T *>(std::aligned_alloc(64, N * sizeof(T))); // 64-byte aligned

        if (!_data)
        {
            throw std::bad_alloc();
        }
    }

    // Destructor
    ~DynamicStorage()
    {
        std::free(_data);
    }

    // Copy constructor
    DynamicStorage(const DynamicStorage &other)
    {
        // _data = static_cast<T *>(std::malloc(N * sizeof(T)));
        _data = static_cast<T *>(std::aligned_alloc(64, N * sizeof(T))); // 64-byte aligned
        if (!_data)
            throw std::bad_alloc();
        std::memcpy(_data, other._data, N * sizeof(T));
    }

    // Move constructor
    DynamicStorage(DynamicStorage &&other) noexcept : _data(other._data)
    {
        other._data = nullptr;
    }

    // Copy assignment
    DynamicStorage &operator=(const DynamicStorage &other)
    {
        if (this != &other)
        {
            if (!_data)
            {
                // _data = static_cast<T *>(std::malloc(N * sizeof(T)));
                _data = static_cast<T *>(std::aligned_alloc(64, N * sizeof(T))); // 64-byte aligned
            }
            std::memcpy(_data, other._data, N * sizeof(T));
        }
        return *this;
    }

    // Move assignment
    DynamicStorage &operator=(DynamicStorage &&other) noexcept
    {
        if (this != &other)
        {
            std::free(_data);
            _data = other._data;
            other._data = nullptr;
        }
        return *this;
    }

    // Element access
    FORCE_INLINE constexpr T &operator[](size_t idx) noexcept { return _data[idx]; }
    FORCE_INLINE constexpr const T &operator[](size_t idx) const noexcept { return _data[idx]; }

    FORCE_INLINE constexpr T *data() noexcept { return _data; }
    FORCE_INLINE constexpr const T *data() const noexcept { return _data; }

    FORCE_INLINE constexpr T *begin() noexcept { return _data; }
    FORCE_INLINE constexpr const T *begin() const noexcept { return _data; }

    FORCE_INLINE constexpr T *end() noexcept { return _data + N; }
    FORCE_INLINE constexpr const T *end() const noexcept { return _data + N; }

private:
    T *_data = nullptr;
};

#endif // DYNAMIC_STORAGE_H
