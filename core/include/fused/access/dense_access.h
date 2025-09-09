#ifndef DENSE_ACCESS_H
#define DENSE_ACCESS_H

#include "../storage/static_storage.h"
#include "../../fill_n_optimized.h"

template <typename T, my_size_t Size, template <typename, my_size_t> class StoragePolicy = StaticStorage>

class DenseAccess
{
    StoragePolicy<T, Size> data_; // StoragePolicy for the tensor data

public:
    DenseAccess() = default;

    DenseAccess(T initValue)
    {
        fill_n_optimized(data_.data(), Size, initValue);
    }

    // Copy constructor
    DenseAccess(const DenseAccess &other)
    {
        if (this == &other)
            return; // Handle self-assignment
        copy_n_optimized(other.data_.data(), data_.data(), Size);
    }

    // Move constructor
    DenseAccess(DenseAccess &&other) noexcept
    {
        if (this == &other)
            return; // Handle self-assignment
        std::move(other.data_.data(), other.data_.data() + Size, data_.data());
        // reset other
        fill_n_optimized(other.data_.data(), Size, T{});
    }

    // Copy assignment
    DenseAccess &operator=(const DenseAccess &other)
    {
        if (this != &other)
        {
            copy_n_optimized(other.data_.data(), data_.data(), Size);
        }
        return *this;
    }

    // Move assignment
    DenseAccess &operator=(DenseAccess &&other) noexcept
    {
        if (this != &other)
        {
            std::move(other.data_.data(), other.data_.data() + Size, data_.data());
            // reset other
            fill_n_optimized(other.data_.data(), Size, T{});
        }
        return *this;
    }

    // Forwarding accessors
    FORCE_INLINE constexpr T &operator[](my_size_t idx) noexcept { return data_[idx]; }
    FORCE_INLINE constexpr const T &operator[](my_size_t idx) const noexcept { return data_[idx]; }

    FORCE_INLINE constexpr T *data() noexcept { return data_.data(); }
    FORCE_INLINE constexpr const T *data() const noexcept { return data_.data(); }

    FORCE_INLINE constexpr T *begin() noexcept { return data_.begin(); }
    FORCE_INLINE constexpr const T *begin() const noexcept { return data_.begin(); }

    FORCE_INLINE constexpr T *end() noexcept { return data_.end(); }
    FORCE_INLINE constexpr const T *end() const noexcept { return data_.end(); }
};

#endif // DENSE_ACCESS_H
