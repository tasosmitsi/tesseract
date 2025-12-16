#ifndef FUSED_PERMUTED_VIEW_H
#define FUSED_PERMUTED_VIEW_H

#include "config.h"
#include "fused/BaseExpr.h"
#include "copy_n_optimized.h"
#include "fused/layouts/strided_layout.h"

template <typename Tensor, my_size_t NumberOfDims>
class PermutedView : public BaseExpr<PermutedView<Tensor, NumberOfDims>, typename Tensor::value_type>
{
public:
    using VecType = typename Tensor::VecType;
    using T = typename Tensor::value_type;
    static constexpr my_size_t simdWidth = Tensor::simdWidth;
    static constexpr my_size_t N = NumberOfDims;

    explicit PermutedView(const Tensor &t, const my_size_t perm[N])
        : t_(t), layout_(t.layout_) // Bind the reference member t_ to the existing object t
                                    // and copy the layout from the base tensor
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            // then set the permuted shape and stride
            layout_.shape[i] = t_.getDim(perm[i]);
            layout_.stride[i] = t_.getStride(perm[i]);
        }
    }

    // delete copy constructor and copy assignment to avoid accidental copies
    PermutedView(const PermutedView &) = delete;
    PermutedView &operator=(const PermutedView &) = delete;

    // delete move constructor and move assignment to avoid accidental moves
    PermutedView(PermutedView &&) = delete;
    PermutedView &operator=(PermutedView &&) = delete;

    // Const version of the access operator, because this is a view
    template <typename... Indices>
        requires(sizeof...(Indices) == N)
    FORCE_INLINE const T &operator()(Indices... indices) const noexcept
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return t_.rawData().data()[layout_.compute_flat_index(idxArray)];
    }

    // Const version of the access operator with array of indices, because this is a view
    FORCE_INLINE const T &operator()(my_size_t (&indices)[N]) const noexcept
    {
        return t_.rawData().data()[layout_.compute_flat_index(indices)];
    }

    FORCE_INLINE const T &operator()(const my_size_t *indices) const noexcept
    {
        return t_.rawData().data()[layout_.compute_flat_index(indices)];
    }

    FORCE_INLINE VecType evalu(my_size_t flat) const noexcept
    {
        my_size_t idxList[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
            idxList[i] = layout_.computeOffsetFromFlat(flat + i);
        return Tensor::microkernel::gather(t_.rawData().data(), idxList);
    }

    FORCE_INLINE constexpr my_size_t getNumDims() const noexcept { return N; }

    FORCE_INLINE constexpr my_size_t getDim(my_size_t i) const // TODO: conditionally noexcept
    {
        return layout_.getDim(i);
    }

    FORCE_INLINE constexpr my_size_t getTotalSize() const noexcept
    {
        // total size is the same as the base tensor, simply return it
        return t_.getTotalSize();
    }

    // Inverse permutation — restores the base tensor
    FORCE_INLINE const Tensor &transpose() const noexcept { return t_; }

    // Utility function to retrieve the shape of the tensor as (1,5,6) for a 3D tensor use the getNumDims
    std::string getShape() const
    {
        std::string shape = "(";
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            shape += std::to_string(getDim(i));
            if (i < getNumDims() - 1)
                shape += ",";
        }
        shape += ")";
        return shape;
    }

private:
    const Tensor &t_;

    using Layout = StridedLayout<N>;
    Layout layout_;
};

#endif // FUSED_PERMUTED_VIEW_H