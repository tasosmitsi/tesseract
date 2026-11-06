#ifndef FUSED_MULTI_SLICE_VIEW_H
#define FUSED_MULTI_SLICE_VIEW_H

#include "config.h"
#include "fused/layouts/layout_adapters.h"
#include "fused/views/strided_view_base.h"

/**
 * @brief Compile-time multi-axis slice view over a tensor or another view.
 *
 * Does not own or copy data, it references the root tensor's physical buffer.
 * Slicing happens entirely at compile time: the view's layout is a
 * SlicedLayout over the source's layout, which narrows the named axes and
 * moves Base to where the slice starts.
 *
 * Each Slice<Axis, Offset, Len> narrows one dimension. Axes not named in the
 * pack keep their full extent. **The view is rank-preserving: a pinned axis
 * becomes 1, it is not dropped** (SqueezeView drops it).
 * Examples:
 *
 *   [2,3,4], Slice<1,2,1>                → [2,1,4]  a single plane
 *   [2,3,4], Slice<1,0,2>                → [2,2,4]  first two planes
 *   [2,3,4], Slice<0,1,1>, Slice<2,0,4>  → [1,3,4]  a vector along axis 1
 *   [M,N],   Slice<0,i,1>                → [1,N]    row i
 *   [M,N],   Slice<1,j,1>                → [M,1]    column j
 *   [5,1],   Slice<0,1,3>                → [3,1]    a subvector
 *
 * Slices are written in the source's own axis numbering, whatever the source
 * is. Over a transpose view, Slice<0,0,1> is row 0 of the transpose: the
 * source's layout already presents the permuted axes, so nothing is remapped.
 *
 * Because the layout is computed from the source's layout, the source can be
 * any view, including another slice: a slice of a slice, of a squeeze, of a
 * subvector or of a diagonal reads the right elements, and the bounds are
 * checked against the source view's own shape.
 *
 * TODO: mutable slices. The view is read-only; a mutable variant would let
 * set_row / set_column write through the same layout.
 *
 * @tparam Tensor  The source: a tensor or another view
 * @tparam Slices  One Slice<Axis, Offset, Len> per narrowed dimension
 */
template <typename Tensor, typename... Slices>
class MultiSliceView
    : public StridedViewBase<MultiSliceView<Tensor, Slices...>, Tensor,
                             SlicedLayout<typename Tensor::Layout, Slices...>>
{
    using Base = StridedViewBase<MultiSliceView<Tensor, Slices...>, Tensor,
                                 SlicedLayout<typename Tensor::Layout, Slices...>>;

public:
    static constexpr const char *ViewName = "MultiSliceView";

    using Base::Base;
};

#endif // FUSED_MULTI_SLICE_VIEW_H
