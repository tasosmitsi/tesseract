#ifndef FUSED_PERMUTED_VIEW_H
#define FUSED_PERMUTED_VIEW_H

#include "config.h"
#include "fused/BaseExpr.h"
#include "copy_n_optimized.h"

template <typename Tensor, my_size_t N>
class PermutedView : public BaseExpr<PermutedView<Tensor, N>, typename Tensor::value_type>
{
    const Tensor &t_;
    my_size_t perm_[N];

public:
    using VecType = typename Tensor::VecType;
    using T = typename Tensor::value_type;
    static constexpr my_size_t simdWidth = Tensor::simdWidth;

    explicit PermutedView(const Tensor &t, const my_size_t perm[N])
        : t_(t)
    {
        for (std::size_t i = 0; i < N; ++i)
            perm_[i] = perm[i]; // copy manually
    }

    // delete copy constructor and copy assignment to avoid accidental copies
    PermutedView(const PermutedView &) = delete;
    PermutedView &operator=(const PermutedView &) = delete;

    // delete move constructor and move assignment to avoid accidental moves
    PermutedView(PermutedView &&) = delete;
    PermutedView &operator=(PermutedView &&) = delete;

    // This function copies the data of the transposed view without permuting them
    // In order for the new tensor to be "permuted" the prem order is copied over
    // to the transposeOrder of the new tensor
    void copyToTensor(Tensor &dst) const
    {
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("PermutedView::copyToTensor called", ErrorLevel::Info);
#endif
        // Copy element by element
        const my_size_t totalSize = dst.getTotalSize();
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            dst.data_[i] = t_.rawData().data()[i];
        }

        // Copy the transpose order
        copy_n_optimized(perm_, dst.transposeOrder_, N);
    }

    // Const version of the access operator, because this is a view
    template <typename... Indices>
        requires(sizeof...(Indices) == N)
    FORCE_INLINE const T &operator()(Indices... indices) const noexcept
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return t_.rawData().data()[computeIndex(idxArray)];
    }

    // Const version of the access operator with array of indices, because this is a view
    FORCE_INLINE const T &operator()(my_size_t (&indices)[N]) const noexcept
    {
        return t_.rawData().data()[computeIndex(indices)];
    }

    FORCE_INLINE VecType evalu(my_size_t flat) const noexcept
    {
        my_size_t idxList[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
            idxList[i] = t_.remapFlatIndex(flat + i, perm_);
        return Tensor::microkernel::gather(t_.rawData().data(), idxList);
    }

    FORCE_INLINE constexpr my_size_t getNumDims() const noexcept { return N; }

    FORCE_INLINE constexpr my_size_t getDim(my_size_t i) const noexcept
    {
#ifdef RUNTIME_USE_BOUNDS_CHECKING
        if (i >= getNumDims())
        {
            MyErrorHandler::error("In PermutedView, getDim(): index out of range!");
        }
#endif
        return t_.getDim(perm_[i]);
    }

    // Inverse permutation — restores the base tensor
    FORCE_INLINE const Tensor &transpose() const noexcept { return t_; }

    bool getIsTransposed() const { return t_.getIsTransposed(); }

private:
    my_size_t computeIndex(const my_size_t indices[N]) const
    {
        my_size_t index = 0;
        my_size_t factor = 1;

        for (my_size_t i = N; i-- > 0;)
        {
#ifdef RUNTIME_USE_BOUNDS_CHECKING
            if (indices[perm_[i]] >= t_.dims[i])
            {
                MyErrorHandler::error("In PermutedView, computeIndex(): index out of bounds!");
            }
#endif
            index += indices[perm_[i]] * factor; // Use permuted axis
            factor *= t_.dims[i];                // multiply by original axis size
        }
        return index;
    }
};

#endif // FUSED_PERMUTED_VIEW_H