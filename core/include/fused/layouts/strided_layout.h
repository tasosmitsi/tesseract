#ifndef FUSED_STRIDED_LAYOUT_H
#define FUSED_STRIDED_LAYOUT_H

#include "config.h"
#include "fill_n_optimized.h"
#include "copy_n_optimized.h"

template <my_size_t N>
struct StridedLayout
{
    my_size_t shape[N];
    my_size_t stride[N];

    StridedLayout(const my_size_t dims[N]) noexcept
    {
        for (my_size_t i = 0; i < N; ++i)
            shape[i] = dims[i];

        compute_row_major_strides();
    }

    // copy constructor
    StridedLayout(const StridedLayout &other) noexcept
    {
        if (this == &other)
            return; // Handle self-assignment
        copy_n_optimized(other.shape, shape, N);
        copy_n_optimized(other.stride, stride, N);
    }

    // move constructor
    StridedLayout(StridedLayout &&other) noexcept
    {
        if (this == &other)
            return; // Handle self-assignment
        std::move(other.shape, other.shape + N, shape);
        std::move(other.stride, other.stride + N, stride);
        // reset other
        fill_n_optimized(other.shape, N, my_size_t{0});
        fill_n_optimized(other.stride, N, my_size_t{0});
    }

    // copy assignment
    StridedLayout &operator=(const StridedLayout &other) noexcept
    {
        if (this == &other)
            return *this; // Handle self-assignment
        copy_n_optimized(other.shape, shape, N);
        copy_n_optimized(other.stride, stride, N);
        return *this;
    }

    // move assignment
    StridedLayout &operator=(StridedLayout &&other) noexcept
    {
        if (this == &other)
            return *this; // Handle self-assignment
        std::move(other.shape, other.shape + N, shape);
        std::move(other.stride, other.stride + N, stride);
        return *this;
    }

    FORCE_INLINE constexpr my_size_t getNumDims() const noexcept { return N; }

    FORCE_INLINE my_size_t getDim(my_size_t i) const // TODO: conditionally noexcept
    {
#ifdef RUNTIME_USE_BOUNDS_CHECKING
        if (i >= getNumDims())
        {
            MyErrorHandler::error("In StridedLayout, getDim(): index out of range!");
        }
#endif
        return shape[i];
    }

    FORCE_INLINE my_size_t getStride(my_size_t i) const // TODO: conditionally noexcept
    {
#ifdef RUNTIME_USE_BOUNDS_CHECKING
        if (i >= getNumDims())
        {
            MyErrorHandler::error("In StridedLayout, getStride(): index out of range!");
        }
#endif
        return stride[i];
    }

    void compute_row_major_strides() noexcept
    {
        stride[getNumDims() - 1] = 1;
        for (my_size_t i = getNumDims() - 1; i > 0; --i)
        {
            stride[i - 1] = stride[i] * shape[i];
        }
    }

    FORCE_INLINE void compute_indices_from_flat(my_size_t flatIdx, my_size_t (&indices)[N]) const noexcept
    {
        // We assume: flatIdx = sum(indices[i] * stride_[i])
        // Solve for indices[i] from highest stride to lowest stride.
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            const my_size_t s = stride[i];
            const my_size_t idx = flatIdx / s;
            indices[i] = idx;
            flatIdx -= idx * s;
        }
    }

    FORCE_INLINE my_size_t compute_flat_index(const my_size_t *indices) const // TODO: conditionally noexcept
    {
        my_size_t flatIndex = 0;
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
#ifdef RUNTIME_USE_BOUNDS_CHECKING
            if (indices[i] >= shape[i])
            {
                MyErrorHandler::error("In StridedLayout, compute_flat_index(): index out of range!");
            }
#endif
            flatIndex += indices[i] * stride[i];
        }
        return flatIndex;
    }

    FORCE_INLINE my_size_t computeOffsetFromFlat(my_size_t flat) const noexcept
    {
        my_size_t off = 0;

        for (my_size_t i = N; i-- > 0;)
        {
            my_size_t idx = flat % shape[i];
            flat /= shape[i];
            off += idx * stride[i];
        }
        return off;
    }
};

#endif // FUSED_STRIDED_LAYOUT_H