#ifndef FUSED_STRIDED_LAYOUT_H
#define FUSED_STRIDED_LAYOUT_H

#include "config.h"
#include "fill_n_optimized.h"
#include "copy_n_optimized.h"

template <my_size_t NumberOfDims>
struct StridedLayout
{
    my_size_t shape[NumberOfDims];
    my_size_t stride[NumberOfDims];

    StridedLayout(const my_size_t dims[NumberOfDims]) noexcept
    {
        copy_n_optimized(dims, shape, NumberOfDims);
        compute_row_major_strides();
    }

    // copy constructor
    StridedLayout(const StridedLayout &other) = default;

    // move constructor
    StridedLayout(StridedLayout &&other) = default;

    // copy assignment
    StridedLayout &operator=(const StridedLayout &other) = default;

    // move assignment
    StridedLayout &operator=(StridedLayout &&other) = default;

    FORCE_INLINE constexpr my_size_t getNumDims() const noexcept { return NumberOfDims; }

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

    FORCE_INLINE void compute_indices_from_flat(my_size_t flatIdx, my_size_t (&indices)[NumberOfDims]) const noexcept
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

        for (my_size_t i = getNumDims(); i-- > 0;)
        {
            my_size_t idx = flat % shape[i];
            flat /= shape[i];
            off += idx * stride[i];
        }
        return off;
    }

    // FORCE_INLINE my_size_t remapFlatIndex(my_size_t flatIdx, const my_size_t (&permutations)[sizeof...(Dims)]) const noexcept
    // {
    //     // Step 1: Unravel flat index in **view order** to multi-index
    //     my_size_t idx[numDims]; // idx[i] = index along original axis i
    //     for (my_size_t i = numDims; i-- > 0;)
    //     {
    //         const my_size_t dim = getDim(permutations[i]); // dim of permuted axis
    //         idx[i] = flatIdx % dim;                        // store in original axis position
    //         flatIdx /= dim;
    //     };

    //     // Step 2: Compute flat index in original tensor layout
    //     my_size_t remapedFlatIdx = 0;
    //     my_size_t factor = 1;
    //     for (my_size_t i = numDims; i-- > 0;)
    //     {
    //         remapedFlatIdx += idx[permutations[i]] * factor;
    //         factor *= getDim(i); // multiply by original axis size
    //     }

    //     return remapedFlatIdx;
    // }
};

#endif // FUSED_STRIDED_LAYOUT_H