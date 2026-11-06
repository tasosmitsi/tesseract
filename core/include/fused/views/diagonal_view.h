#ifndef FUSED_DIAGONAL_VIEW_H
#define FUSED_DIAGONAL_VIEW_H

#include "config.h"
#include "fused/layouts/layout_adapters.h"
#include "fused/views/strided_view_base.h"

/**
 * @brief Compile-time diagonal view over a square matrix or square view.
 *
 * This class does not own or copy data, it references the root tensor's physical buffer.
 * The view's layout is a DiagonalLayout over the source's layout: an [N,1]
 * column whose stride is the sum of the source's two strides, starting at the
 * source's Base. So the diagonal of a block reads the block's diagonal.
 *
 * TODO: it can be generalized to rectangular matrices, but the current use case is square matrices only.
 * TODO: it can be generalized to rectangular tensors.
 *
 * Extracts the main diagonal of an N×N matrix as an N-element column vector.
 *
 *   Source A[3,3] with SimdWidth=4:
 *     Physical layout (padded to 3×4):
 *       [a00 a01 a02 _ | a10 a11 a12 _ | a20 a21 a22 _]
 *     Strides: [4, 1]
 *
 *   Diagonal view: Strides [5, 1]
 *     view(0) → A(0,0) → physical 0
 *     view(1) → A(1,1) → physical 5
 *     view(2) → A(2,2) → physical 10
 *
 * Enables `trace = reduce_sum(diagonal(A))` via SIMD.
 *
 * @tparam Matrix  The underlying square matrix or view
 */
template <typename Matrix>
class DiagonalView
    : public StridedViewBase<DiagonalView<Matrix>, Matrix, DiagonalLayout<typename Matrix::Layout>>
{
    // The source must be 2D and square: checked by DiagonalLayout

    using Base = StridedViewBase<DiagonalView<Matrix>, Matrix, DiagonalLayout<typename Matrix::Layout>>;

public:
    static constexpr const char *ViewName = "DiagonalView";

    using Base::Base;
    using Base::operator();

    // Single-index access: view(i) → source(i, i)
    FORCE_INLINE const typename Base::value_type &operator()(my_size_t i) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Base::operator()(i, my_size_t{0});
    }
};

#endif // FUSED_DIAGONAL_VIEW_H
