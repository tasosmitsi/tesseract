#ifndef DENSE_ACCESS_H
#define DENSE_ACCESS_H

#include "fused/storage/static_storage.h"
#include "fill_n_optimized.h"

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
    DenseAccess(const DenseAccess &other) = default;

    // Move constructor
    DenseAccess(DenseAccess &&other) = default;

    // Copy assignment
    DenseAccess &operator=(const DenseAccess &other) = default;

    // Move assignment
    DenseAccess &operator=(DenseAccess &&other) = default;

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
