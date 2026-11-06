#ifndef FUSED_PERMUTED_VIEW_CONSTEXPR_H
#define FUSED_PERMUTED_VIEW_CONSTEXPR_H

#include "config.h"
#include "fused/layouts/layout_adapters.h"
#include "fused/views/strided_view_base.h"

/**
 * @brief Compile-time permuted view over a tensor or another view.
 *
 * This class does not own or copy data, it references the root tensor's physical buffer.
 * Permutation is applied entirely at compile time: the view's layout is a
 * PermutedLayout over the source's layout, which reorders its dims and
 * strides and keeps its Base.
 *
 * The permuted layout reinterprets the same physical memory with permuted
 * logical dimensions and strides:
 *
 *   Source A[2,3] padded to [2,4]:
 *     Physical: [a b c 0 | d e f 0]
 *     Layout:   LogicalDims=[2,3], Strides=[4,1]
 *
 *   Transposed view (Perm = 1,0):
 *     Same physical buffer
 *     Layout:   LogicalDims=[3,2], Strides=[1,4]
 *
 *     view(0,0) → 0*1 + 0*4 = offset 0 → 'a'
 *     view(0,1) → 0*1 + 1*4 = offset 4 → 'd'
 *     view(1,0) → 1*1 + 0*4 = offset 1 → 'b'
 *     view(2,1) → 2*1 + 1*4 = offset 6 → 'f'
 *
 * Over a view the same holds: the transpose of a block starts where the block
 * does. Only a transpose of a tensor is an einsum/matmul operand, though: see
 * FusedTensorND::is_einsum_operand.
 *
 * @tparam Tensor  The source: a tensor or another view
 * @tparam Perm    Compile-time permutation indices
 */
template <typename Tensor, my_size_t... Perm>
class PermutedViewConstExpr
    : public StridedViewBase<PermutedViewConstExpr<Tensor, Perm...>, Tensor,
                             PermutedLayout<typename Tensor::Layout, Perm...>>
{
    // The permutation pack is validated by PermutedLayout

    using Base = StridedViewBase<PermutedViewConstExpr<Tensor, Perm...>, Tensor,
                                 PermutedLayout<typename Tensor::Layout, Perm...>>;

public:
    static constexpr const char *ViewName = "PermutedView";

    using Base::Base;
    using Base::operator();

    // Access by a pointer to NumDims indices
    FORCE_INLINE const typename Base::value_type &operator()(const my_size_t *indices) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        my_size_t idxArray[Base::NumDims];
        for (my_size_t i = 0; i < Base::NumDims; ++i)
            idxArray[i] = indices[i];
        return Base::operator()(idxArray);
    }

    // Inverse permutation that restores the source. For a transpose of a tensor,
    // that is the base tensor, whose physical layout IS the transposed data.
    FORCE_INLINE const Tensor &transpose() const noexcept { return this->source(); }
};

#endif // FUSED_PERMUTED_VIEW_CONSTEXPR_H
