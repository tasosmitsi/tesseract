#pragma once

#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/expr_storage.h"
#include "fused/layouts/layout_base.h"
#include "fused/microkernels/microkernel_base.h"

/**
 * @brief Every view reads its elements through its layout the same way.
 *
 * A view is a source (a tensor or another view) plus a layout computed from
 * the source's layout by one of the adapters in fused/layouts/layout_adapters.h.
 * Every view hands its source's data() through unshifted, so data() is always
 * the root tensor's buffer, and Layout::Base says where the view starts in it:
 *
 *   element c  =  data()[Layout::logical_coords_to_physical_flat(c)]
 *
 * Nothing here depends on what kind of view it is. That is why views chain in
 * any order: a view of a view is just a layout of a layout.
 *
 * The children (MultiSliceView, SqueezeView, SubVectorView, DiagonalView,
 * PermutedViewConstExpr) choose the layout and keep their own extras: one-index
 * access for the vector-shaped views, transpose() for the permuted one. Each
 * names itself for printLayoutInfo through a static ViewName.
 *
 * TODO: mutable views, once set_row lands. The same offsets serve writes.
 *
 * @tparam Derived  The view (CRTP)
 * @tparam Source   What the view is built on: a tensor or another view
 * @tparam L        The view's layout, computed from Source::Layout
 */
template <typename Derived, typename Source, typename L>
class StridedViewBase : public BaseExpr<Derived>
{
public:
    using value_type = typename Source::value_type;
    using Layout = L;

    static constexpr my_size_t NumDims = Layout::NumDims;

    // The view's logical shape, read back out of the layout so it cannot
    // disagree with getDim
    static constexpr const my_size_t (&Dim)[NumDims] = LayoutDims<Layout>::value;

    static constexpr my_size_t TotalSize = Layout::LogicalSize;

    // Offset of the view's first element in the root buffer
    static constexpr my_size_t PhysicalBase = Layout::Base;

    // True when the view is one unbroken physical span, so SIMD can load
    // directly instead of gathering. See StridedLayoutBase::is_contiguous.
    static constexpr bool IsPhysicallyContiguous = Layout::is_contiguous();

    explicit StridedViewBase(const Source &src) noexcept
        : src_(src) {}

    // Copyable and movable, so an expression can keep its own copy of a view
    // that was a temporary (see fused/expr_storage.h). Not assignable: a view
    // cannot be re-pointed.
    StridedViewBase(const StridedViewBase &) = default;
    StridedViewBase &operator=(const StridedViewBase &) = delete;
    StridedViewBase(StridedViewBase &&) = default;
    StridedViewBase &operator=(StridedViewBase &&) = delete;

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return src_.may_alias(output); // recurse to the root tensor
    }

    // Const version of the access operator, because this is a view
    template <typename... Indices>
        requires(sizeof...(Indices) == NumDims)
    FORCE_INLINE const value_type &operator()(Indices... indices) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return src_.data()[Layout::logical_coords_to_physical_flat(idxArray)];
    }

    // Const version of the access operator with array of indices, because this is a view
    FORCE_INLINE const value_type &operator()(my_size_t (&indices)[NumDims]) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return src_.data()[Layout::logical_coords_to_physical_flat(indices)];
    }

    /**
     * @brief SIMD EVALUATION: direct load when contiguous, gather otherwise.
     *
     * Takes a logical flat index: expression::traits marks every view
     * IsPermuted, so evaluation always hands views logical indices. When the
     * view is one span, consecutive logical elements sit at consecutive offsets
     * from Base and a plain load suffices. Otherwise each index is mapped
     * through the layout and the values are gathered.
     *
     * Example: column slice of [4,6] double padded to [4,8], Slice<1,3,1>:
     *   logical_flat 0 → coords(0,0) → offset 3
     *   logical_flat 1 → coords(1,0) → offset 11
     *   logical_flat 2 → coords(2,0) → offset 19
     *   logical_flat 3 → coords(3,0) → offset 27
     *   → gather from offsets [3, 11, 19, 27]
     */
    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType evalu(my_size_t logical_flat) const noexcept
    {
        using K = Microkernel<T, Bits, Arch>;

        if constexpr (IsPhysicallyContiguous)
        {
            return K::load(src_.data() + Layout::Base + logical_flat);
        }
        else
        {
            constexpr my_size_t width = K::simdWidth;

            my_size_t idxList[width];
            for (my_size_t i = 0; i < width; ++i)
                idxList[i] = Layout::logical_flat_to_physical_flat(logical_flat + i);

            return K::gather(src_.data(), idxList);
        }
    }

    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType logical_evalu(my_size_t logical_flat) const noexcept
    {
        // evalu already expects a logical flat index for views
        return evalu<T, Bits, Arch>(logical_flat);
    }

    // The view's logical dimension
    FORCE_INLINE static constexpr my_size_t getDim(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::logical_dim(i);
    }

    // The view's stride in the root buffer: how far one step along axis i moves
    FORCE_INLINE static constexpr my_size_t getStride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::stride(i);
    }

    FORCE_INLINE static constexpr my_size_t getNumDims() noexcept { return NumDims; }

    FORCE_INLINE static constexpr my_size_t getTotalSize() noexcept { return TotalSize; }

    // The root tensor's buffer, unshifted. Layout::Base says where the view starts in it.
    FORCE_INLINE constexpr const value_type *data() const noexcept { return src_.data(); }

    // print layout info
    void printLayoutInfo() const
    {
        MyErrorHandler::log(std::string(Derived::ViewName) + " Layout Info:", ErrorLevel::Info);
        MyErrorHandler::log("Number of Dimensions: " + std::to_string(NumDims), ErrorLevel::Info);
        MyErrorHandler::log("Shape: " + getShape(), ErrorLevel::Info);
        MyErrorHandler::log("Strides: ", ErrorLevel::Info);
        for (my_size_t i = 0; i < NumDims; ++i)
            MyErrorHandler::log(std::to_string(getStride(i)) + " ", ErrorLevel::Info);
        MyErrorHandler::log("\n", ErrorLevel::Info);
    }

    // Utility function to retrieve the shape of the view as (1,5,6) for a 3D view
    std::string getShape() const
    {
        std::string shape = "(";
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            shape += std::to_string(getDim(i));
            if (i < NumDims - 1)
                shape += ",";
        }
        shape += ")";
        return shape;
    }

protected:
    // What the view was built on, for children that hand it back
    FORCE_INLINE const Source &source() const noexcept { return src_; }

private:
    ExprStorage<Source> src_;
};
