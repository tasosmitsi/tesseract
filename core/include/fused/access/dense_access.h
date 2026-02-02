#ifndef DENSE_ACCESS_H
#define DENSE_ACCESS_H

#include "fused/storage/static_storage.h"
#include "fill_n_optimized.h"

/**
 * @brief Dense storage access with padding policy.
 *
 * @tparam T             Element type
 * @tparam PaddingPolicy Padding policy (e.g., SimdPaddingPolicy, NoPaddingPolicy)
 * @tparam StoragePolicy Storage backend (e.g., StaticStorage)
 * @tparam Dims          Logical dimensions of the tensor
 */
template <typename T,
          template <typename, my_size_t...> class PaddingPolicy,
          template <typename, my_size_t> class StoragePolicy,
          my_size_t... Dims>
class DenseAccess
{
public:
    using Policy = PaddingPolicy<T, Dims...>;

    static constexpr my_size_t NumDims = Policy::NumDims;
    static constexpr my_size_t LogicalSize = Policy::LogicalSize;
    static constexpr my_size_t PhysicalSize = Policy::PhysicalSize;
    static constexpr my_size_t SimdWidth = Policy::SimdWidth;

private:
    StoragePolicy<T, PhysicalSize> data_;

public:
    DenseAccess() = default;

    explicit DenseAccess(T initValue)
    {
        fill_n_optimized(data_.data(), PhysicalSize, initValue);
    }

    // Rule of five
    DenseAccess(const DenseAccess &) = default;
    DenseAccess(DenseAccess &&) noexcept = default;
    DenseAccess &operator=(const DenseAccess &) = default;
    DenseAccess &operator=(DenseAccess &&) noexcept = default;
    ~DenseAccess() = default;

    // Element access (physical index)
    FORCE_INLINE constexpr T &operator[](my_size_t idx) noexcept { return data_[idx]; }
    FORCE_INLINE constexpr const T &operator[](my_size_t idx) const noexcept { return data_[idx]; }

    FORCE_INLINE constexpr T &at(my_size_t idx) noexcept { return data_.at(idx); }
    FORCE_INLINE constexpr const T &at(my_size_t idx) const noexcept { return data_.at(idx); }

    // Pointer access
    FORCE_INLINE constexpr T *data() noexcept { return data_.data(); }
    FORCE_INLINE constexpr const T *data() const noexcept { return data_.data(); }

    // Iterators
    FORCE_INLINE constexpr T *begin() noexcept { return data_.begin(); }
    FORCE_INLINE constexpr const T *begin() const noexcept { return data_.begin(); }

    FORCE_INLINE constexpr T *end() noexcept { return data_.end(); }
    FORCE_INLINE constexpr const T *end() const noexcept { return data_.end(); }

    // Dimension queries (compile-time)
    static constexpr my_size_t logical_dim(my_size_t i) { return Policy::LogicalDims.at(i); }
    static constexpr my_size_t physical_dim(my_size_t i) { return Policy::PhysicalDims.at(i); }
};

#endif // DENSE_ACCESS_H