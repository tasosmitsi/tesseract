#ifndef FUSED_SUBVECTOR_VIEW_H
#define FUSED_SUBVECTOR_VIEW_H

#include "config.h"
#include "fused/layouts/layout_adapters.h"
#include "fused/views/strided_view_base.h"

namespace detail
{
    /**
     * @brief The non-trivial axis of a vector-shaped source: 1 for a row [1,N], 0 for a column [N,1].
     */
    template <typename Vector>
    inline constexpr my_size_t vector_axis = (Vector::Dim[0] == 1) ? 1 : 0;
} // namespace detail

/**
 * @brief Compile-time subvector view over a vector-shaped tensor or view.
 *
 * Does not own or copy data, it references the root tensor's physical buffer.
 * The view's layout is a one-axis SlicedLayout over the source's layout, so
 * access respects the source's padding and strides, and bounds are checked
 * against the subvector itself.
 *
 * Automatically detects whether the source is a column vector (Dim[1]==1)
 * or a row vector (Dim[0]==1) and slices along the non-trivial axis.
 *
 * Works with FusedVector, FusedMatrix<T,N,1>, FusedMatrix<T,1,N>,
 * transposed views of any of these, and other vector-shaped views (a column
 * slice, a diagonal).
 *
 *   Column vector v[5,1] with SimdWidth=4:
 *     Axis=0, view(i) → source(Offset + i, 0)
 *
 *   Row vector (transposed) v[1,5]:
 *     Axis=1, view(i) → source(0, Offset + i)
 *
 * @tparam Vector  The underlying vector-shaped type
 * @tparam Offset  Compile-time start index along the non-trivial axis
 * @tparam Len     Compile-time number of elements in the view
 */
template <typename Vector, my_size_t Offset, my_size_t Len>
class SubVectorView
    : public StridedViewBase<SubVectorView<Vector, Offset, Len>, Vector,
                             SlicedLayout<typename Vector::Layout,
                                          Slice<detail::vector_axis<Vector>, Offset, Len>>>
{
    static_assert(Vector::NumDims == 2,
                  "SubVectorView: source must have exactly 2 dimensions");

    static_assert(Vector::Dim[0] == 1 || Vector::Dim[1] == 1,
                  "SubVectorView: source must be a vector (one dimension must be 1)");

    // Offset, Len and bounds are checked by SlicedLayout, against the source's extent

    using Base = StridedViewBase<SubVectorView<Vector, Offset, Len>, Vector,
                                 SlicedLayout<typename Vector::Layout,
                                              Slice<detail::vector_axis<Vector>, Offset, Len>>>;

public:
    static constexpr const char *ViewName = "SubVectorView";

    // The axis the subvector runs along: 0 for a column, 1 for a row
    static constexpr my_size_t Axis = detail::vector_axis<Vector>;

    using Base::Base;
    using Base::operator();

    // Single-index access: view(i) → element i along the non-trivial axis
    FORCE_INLINE const typename Base::value_type &operator()(my_size_t i) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if constexpr (Axis == 0)
            return Base::operator()(i, my_size_t{0});
        else
            return Base::operator()(my_size_t{0}, i);
    }
};

#endif // FUSED_SUBVECTOR_VIEW_H
