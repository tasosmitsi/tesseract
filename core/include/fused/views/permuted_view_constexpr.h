#ifndef FUSED_PERMUTED_VIEW_CONSTEXPR_H
#define FUSED_PERMUTED_VIEW_CONSTEXPR_H

#include "config.h"
#include "fused/BaseExpr.h"
#include "copy_n_optimized.h"
#include "fused/layouts/strided_layout.h"

/*  The reason the PermutedViewConstExpr is because only if permutation is known at compile time
    static checks can be performed. Otherwise if the permutation order is to be decided
    at runtime PermutedView must be usued insted.
 */

template <typename Tensor, my_size_t... Perm>
class PermutedViewConstExpr : public BaseExpr<PermutedViewConstExpr<Tensor, Perm...>>
{

public:
    using value_type = typename Tensor::value_type;
    static constexpr my_size_t Perm_Array[] = {Perm...}; // Fixed array of original dimensions
    static constexpr my_size_t NumDims = Tensor::NumDims;
    static constexpr my_size_t Dim[] = {Tensor::Dim[Perm]...};
    static constexpr my_size_t TotalSize = Tensor::TotalSize;

    explicit PermutedViewConstExpr(const Tensor &t)
        : t_(t), layout_(t.layout_) // Bind the reference member t_ to the existing object t
                                    // and copy the layout from the base tensor
    {
        for (std::size_t i = 0; i < NumDims; ++i)
        {
            // then set the permuted shape and stride
            layout_.shape[i] = t_.getDim(Perm_Array[i]);
            layout_.stride[i] = t_.getStride(Perm_Array[i]);
        }
    }

    // delete copy constructor and copy assignment to avoid accidental copies
    PermutedViewConstExpr(const PermutedViewConstExpr &) = delete;
    PermutedViewConstExpr &operator=(const PermutedViewConstExpr &) = delete;

    // delete move constructor and move assignment to avoid accidental moves
    PermutedViewConstExpr(PermutedViewConstExpr &&) = delete;
    PermutedViewConstExpr &operator=(PermutedViewConstExpr &&) = delete;

    // Const version of the access operator, because this is a view
    template <typename... Indices>
        requires(sizeof...(Indices) == NumDims)
    FORCE_INLINE const value_type &operator()(Indices... indices) const noexcept
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return t_.data_.data()[layout_.compute_flat_index(idxArray)];
    }

    // Const version of the access operator with array of indices, because this is a view
    FORCE_INLINE const value_type &operator()(my_size_t (&indices)[NumDims]) const noexcept
    {
        return t_.data_.data()[layout_.compute_flat_index(indices)];
    }

    FORCE_INLINE const value_type &operator()(const my_size_t *indices) const noexcept
    {
        return t_.data_.data()[layout_.compute_flat_index(indices)];
    }

    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType evalu(my_size_t flat) const noexcept
    {
        using K = Microkernel<T, Bits, Arch>;
        constexpr my_size_t width = K::simdWidth;

        my_size_t idxList[width];
        for (my_size_t i = 0; i < width; ++i)
            idxList[i] = layout_.computeOffsetFromFlat(flat + i);

        return K::gather(t_.data_.data(), idxList);
    }

    FORCE_INLINE constexpr my_size_t getNumDims() const noexcept { return NumDims; }

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

    using Layout = StridedLayout<NumDims>;
    Layout layout_;
};

#endif // FUSED_PERMUTED_VIEW_CONSTEXPR_H