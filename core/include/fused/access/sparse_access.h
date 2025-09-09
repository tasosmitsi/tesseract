#ifndef SPARSE_ACCESS_H
#define SPARSE_ACCESS_H

#include "../storage/static_storage.h"
#include "../../fill_n_optimized.h"

template <
    typename T,
    my_size_t NonZeroCount,
    typename IndexType = my_size_t,
    template <typename, my_size_t> class ValueStoragePolicy = StaticStorage,
    template <typename, my_size_t> class IndexStoragePolicy = StaticStorage>
class SparseAccess
{
    ValueStoragePolicy<T, NonZeroCount> values_;          // Nonzero values
    IndexStoragePolicy<IndexType, NonZeroCount> indices_; // Flat indices

    my_size_t current_size_ = 0;
    mutable my_size_t last_idx_ = 0; // track last successful search

    // Dummy zero for failed reference returns (only used if overflow occurs)
    T dummy_zero_ = T(0);

public:
    SparseAccess()
    {
        fill_n_optimized(indices_.data(), NonZeroCount, IndexType(-1));
    }

    SparseAccess(T initValue)
    {
        for (my_size_t i = 0; i < NonZeroCount; ++i) // assume that
        {
            this->operator[](i) = initValue;
        }
    }

    // Copy constructor
    SparseAccess(const SparseAccess &other)
    {
        if (this == &other)
            return; // Handle self-assignment
        copy_n_optimized(other.values_.data(), values_.data(), NonZeroCount);
        copy_n_optimized(other.indices_.data(), indices_.data(), NonZeroCount);
        current_size_ = other.current_size_;
    }

    // Move constructor
    SparseAccess(SparseAccess &&other) noexcept
    {
        if (this == &other)
            return; // Handle self-assignment
        std::move(other.values_.data(), other.values_.data() + NonZeroCount, values_.data());
        std::move(other.indices_.data(), other.indices_.data() + NonZeroCount, indices_.data());
        current_size_ = other.current_size_;
        // reset other
        fill_n_optimized(other.indices_.data(), NonZeroCount, IndexType(-1));
        fill_n_optimized(other.values_.data(), NonZeroCount, T{});
        other.current_size_ = 0;
    }

    // Copy assignment
    SparseAccess &operator=(const SparseAccess &other)
    {
        if (this != &other)
        {
            copy_n_optimized(other.values_.data(), values_.data(), NonZeroCount);
            copy_n_optimized(other.indices_.data(), indices_.data(), NonZeroCount);
            current_size_ = other.current_size_;
        }
        return *this;
    }

    // Move assignment
    SparseAccess &operator=(SparseAccess &&other) noexcept
    {
        if (this != &other)
        {
            std::move(other.values_.data(), other.values_.data() + NonZeroCount, values_.data());
            std::move(other.indices_.data(), other.indices_.data() + NonZeroCount, indices_.data());
            current_size_ = other.current_size_;

            // reset other
            fill_n_optimized(other.indices_.data(), NonZeroCount, IndexType(-1));
            fill_n_optimized(other.values_.data(), NonZeroCount, T{});
            other.current_size_ = 0;
        }
        return *this;
    }

    // print the sparse representation
    void print() const
    {
        std::cout << "Sparse Representation (NonZeroCount = " << NonZeroCount << "):\n";
        for (my_size_t i = 0; i < current_size_; ++i)
        {
            std::cout << "Index: " << indices_[i] << ", Value: " << values_[i] << "\n";
        }
    }

    // Iterators for values
    FORCE_INLINE constexpr T *data() noexcept { return values_.data(); }
    FORCE_INLINE constexpr const T *data() const noexcept { return values_.data(); }
    FORCE_INLINE constexpr T *values_begin() noexcept { return values_.begin(); }
    FORCE_INLINE constexpr const T *values_begin() const noexcept { return values_.begin(); }
    FORCE_INLINE constexpr T *values_end() noexcept { return values_.end(); }
    FORCE_INLINE constexpr const T *values_end() const noexcept { return values_.end(); }

    // Iterators for indices
    FORCE_INLINE constexpr IndexType *indices_data() noexcept { return indices_.data(); }
    FORCE_INLINE constexpr const IndexType *indices_data() const noexcept { return indices_.data(); }
    FORCE_INLINE constexpr IndexType *indices_begin() noexcept { return indices_.begin(); }
    FORCE_INLINE constexpr const IndexType *indices_begin() const noexcept { return indices_.begin(); }
    FORCE_INLINE constexpr IndexType *indices_end() noexcept { return indices_.end(); }
    FORCE_INLINE constexpr const IndexType *indices_end() const noexcept { return indices_.end(); }

    // Number of nonzero elements
    // static constexpr my_size_t size() noexcept { return NonZeroCount; }

    // Dense-style operator[] returning reference for assignment
    FORCE_INLINE T &operator[](IndexType idx) noexcept
    {
        // Check if index already exists
        for (my_size_t i = 0; i < current_size_; ++i)
        {
            if (indices_[i] == idx)
                return values_[i];
        }

        // Not found: insert if space available
        if (current_size_ < NonZeroCount)
        {
            indices_[current_size_] = idx;
            values_[current_size_] = T{}; // initialize to zero
            ++current_size_;
            return values_[current_size_ - 1];
        }

        // If full, return reference to dummy_zero_
        MyErrorHandler::error("SparseAccess: exceeded NonZeroCount capacity!");
        return dummy_zero_;
    }

    // Const operator[] for read-only access
    FORCE_INLINE const T &operator[](IndexType idx) const noexcept
    {
        // Check the last accessed first
        if (last_idx_ < current_size_ && indices_[last_idx_] == idx)
            return values_[last_idx_];

        for (my_size_t i = 0; i < current_size_; ++i)
        {
            if (indices_[i] == idx)
            {
                last_idx_ = i;
                return values_[i];
            }
        }
        return dummy_zero_;
    }
};

#endif // SPARSE_ACCESS_H
