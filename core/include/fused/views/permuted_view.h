#ifndef FUSED_PERMUTED_VIEW_H
#define FUSED_PERMUTED_VIEW_H

#include "config.h"
#include "fused/BaseExpr.h"

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

    FORCE_INLINE T &operator()(my_size_t (&indices)[N]) noexcept
    {
        return t_.rawData().data()[t_.computeIndex(indices)];
    }

    FORCE_INLINE const T &operator()(my_size_t (&indices)[N]) const noexcept
    {
        return t_.rawData().data()[t_.computeIndex(indices)];
    }

    FORCE_INLINE VecType evalu(my_size_t flat) const noexcept
    {
        my_size_t idxList[simdWidth];
        for (my_size_t i = 0; i < simdWidth; ++i)
            idxList[i] = t_.remapFlatIndex(flat + i, perm_);
        return Tensor::microkernel::gather(t_.rawData().data(), idxList);
    }

    FORCE_INLINE constexpr my_size_t getNumDims() const noexcept { return N; }

    FORCE_INLINE constexpr my_size_t getDim(my_size_t i) const noexcept { return t_.getDim(perm_[i]); }

    // Inverse permutation — restores the base tensor
    FORCE_INLINE const Tensor &transpose() const noexcept { return t_; }
};

#endif // FUSED_PERMUTED_VIEW_H