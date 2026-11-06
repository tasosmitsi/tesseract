#ifndef FUSED_SQUEEZE_VIEW_H
#define FUSED_SQUEEZE_VIEW_H

#include "config.h"
#include "fused/layouts/layout_adapters.h"
#include "fused/views/strided_view_base.h"

/**
 * @brief Compile-time view that drops size-1 axes from a tensor or another view.
 *
 * Does not own or copy data, it references the root tensor's physical buffer.
 * Squeezing happens at compile time: the view's layout is a SqueezedLayout
 * over the source's layout, which removes the named axes from its dims and
 * strides.
 *
 * MultiSliceView is rank-preserving, so pinning two axes of a [Na,Nb,2,4]
 * tensor gives [1,1,2,4]. Assigning that to a [2,4] matrix fails on rank:
 * checkDimsMatch compares NumDims first. This view removes the 1s.
 *
 *   [1,1,2,4], drop 0,1  → [2,4]
 *   [1,3,1],   drop 0    → [3,1]
 *   [M,1,N],   drop 1    → [M,N]
 *
 * Axes are named rather than inferred. Dropping every size-1 axis reads better
 * at the call site, but [1,3,1] is a case where dropping both is rejected and
 * dropping one is fine, so the caller has to know which it wants anyway. See
 * the rank floor in SqueezedLayout.
 *
 * ============================================================================
 * COMPOSES WITH ANY SOURCE
 * ============================================================================
 *
 * The source's layout carries where it starts (Base), so a squeeze of a slice
 * reads the right corner, and a slice of a squeeze reads the right row. A
 * squeeze of a transpose or of another squeeze works the same way: the layout
 * is built from the source's dims and strides, whatever produced them.
 *
 * TODO: mutable squeezes, once set_row lands and there is a mutable view to
 * build on.
 *
 * @tparam Tensor       The tensor or view being squeezed
 * @tparam DroppedAxes  Axes to drop, each of logical extent 1
 */
template <typename Tensor, my_size_t... DroppedAxes>
class SqueezeView
    : public StridedViewBase<SqueezeView<Tensor, DroppedAxes...>, Tensor,
                             SqueezedLayout<typename Tensor::Layout, DroppedAxes...>>
{
    using Base = StridedViewBase<SqueezeView<Tensor, DroppedAxes...>, Tensor,
                                 SqueezedLayout<typename Tensor::Layout, DroppedAxes...>>;

public:
    static constexpr const char *ViewName = "SqueezeView";

    using Base::Base;
};

#endif // FUSED_SQUEEZE_VIEW_H
