#ifndef FUSED_MULTI_SLICE_VIEW_H
#define FUSED_MULTI_SLICE_VIEW_H

#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/layouts/strided_layout_constexpr.h"
#include "fused/padding_policies/multi_sliced_pad_policy.h"
#include "helper_traits.h"

/**
 * @brief Compile-time multi-axis slice view over an N-dimensional tensor.
 *
 * Does not own or copy data, it references the underlying tensor's physical
 * buffer. Slicing happens entirely at compile time through
 * StridedLayoutConstExpr over a MultiSlicedPadPolicy.
 *
 * How it works:
 *  1. Each Slice is translated from the view's axis numbering into the source
 *     policy's, because the source may be permuted and the two numberings then
 *     differ. The translation goes through the source layout's perm_array(),
 *     which is the identity when there is no permutation.
 *  2. The source's PadPolicy is narrowed into a MultiSlicedPadPolicy
 *  3. The source's StridedLayoutConstExpr is rebuilt over the new policy
 *  4. The view's operator() maps logical coordinates through the new layout and
 *     adds PhysicalBase, the offset where the slice starts.
 *
 * Each Slice<Axis, Offset, Len> narrows one dimension. Axes not named in the
 * pack keep their full extent. **The view is rank-preserving: a pinned axis
 * becomes 1, it is not dropped.**
 * Examples:
 *
 *   [2,3,4], Slice<1,2,1>                → [2,1,4]  a single plane
 *   [2,3,4], Slice<1,0,2>                → [2,2,4]  first two planes
 *   [2,3,4], Slice<0,1,1>, Slice<2,0,4>  → [1,3,4]  a vector along axis 1
 *   [M,N],   Slice<0,i,1>                → [1,N]    row i
 *   [M,N],   Slice<1,j,1>                → [M,1]    column j
 *   [5,1],   Slice<0,1,3>                → [3,1]    a subvector
 *
 * The narrowed shape lives in the new layout; the offsets do not. A layout maps
 * coordinates to offsets within a buffer, so where the slice starts is a
 * constant added to whatever the layout returns.
 *
 * ============================================================================
 * TWO AXIS NUMBERINGS
 * ============================================================================
 *
 * The Slice pack is written in the view's numbering. MultiSlicedPadPolicy
 * narrows in the source policy's. For an unpermuted source these coincide; for
 * a permuted one they differ, because the layout defines
 *
 *   logical_dim(i) = PadPolicy::LogicalDims[PermArray[i]]
 *
 * so view axis i is a relabelling of policy axis PermArray[i]. perm_array() is
 * the dictionary, and it is the identity when no permutation was provided.
 * Offset and Len cross unchanged: a permutation relabels axes, it does not move
 * elements within one.
 *
 *   SourceLayout           carries Perm..., presents view space
 *       │ perm_array(a)    ← the dictionary
 *   translated Slices      view axes become policy axes
 *       │
 *   MultiSlicedPadPolicy   narrows LogicalDims in policy space,
 *       │                  inherits PhysicalDims from the source
 *       │ rebind<Policy>   ← the same pack, re-applied
 *   Layout                 logical_dim / stride back in view space
 *       │
 *   Dim[]                  read from Layout::logical_dim
 *
 * The PadPolicy alias below is the boundary: view space above it, including
 * this class's static_asserts so their diagnostics name the caller's axes;
 * policy space from there down.
 *
 * Worked through — A[4,6] padded to [4,8], transposed so it presents as [6,4],
 * asking for row 0 with Slice<0,0,1> and expecting [1,4]:
 *
 *   perm_array(0) = 1               the pack becomes Slice<1,0,1>
 *   policy narrows [4,6] at axis 1  → LogicalDims [4,1]
 *   layout rebuilt with Perm=[1,0]  → logical_dim [1,4], stride [1,8]
 *
 * TODO: mutable slices. The view is read-only; a mutable variant would let
 * set_row / set_column write through the same machinery.
 *
 * @tparam Tensor  The underlying tensor type
 * @tparam Slices  One Slice<Axis, Offset, Len> per narrowed dimension
 */
template <typename Tensor, typename... Slices>
class MultiSliceView;

namespace detail
{
    /**
     * @brief Translate one Slice from the view's axis numbering into the policy's.
     *
     * perm_array() is the identity when the layout carries no permutation, so
     * this is a no-op for a plain tensor. Offset and Len are unaffected: a
     * permutation only relabels axes, it does not move elements within one.
     *
     * TODO: perm_array() returns PermArray.at(i), and the out-of-bounds branch
     * calls a non-constexpr error handler. The view's own static_assert catches 
     * that first, but a failed static_assert does not stop compilation, so RemapSlice 
     * is still instantiated and more errors are thrown. Switching perm_array to 
     * PermArray[i] drops the check and the cascade; the axis is already known in range 
     * by the time this runs.
     */
    template <typename SourceLayout, typename S>
    using RemapSlice = Slice<SourceLayout::perm_array(S::Axis), S::Offset, S::Len>;

    /**
     * @brief Detects MultiSliceView, so nesting can be rejected.
     *
     * Specialized here rather than after the class because the static_assert
     * that uses it lives in the class body. A partial specialization only needs
     * the primary template declared, which the forward declaration above
     * provides.
     */
    template <typename T>
    struct is_multi_slice_view
    {
        static constexpr bool value = false;
    };

    template <typename Tensor, typename... Slices>
    struct is_multi_slice_view<MultiSliceView<Tensor, Slices...>>
    {
        static constexpr bool value = true;
    };
} // namespace detail

template <typename Tensor, typename... Slices>
class MultiSliceView : public BaseExpr<MultiSliceView<Tensor, Slices...>>
{
    // A nested view type-checks, but the inner view's PhysicalBase is not part
    // of its Layout, so the outer one would silently ignore where the inner
    // slice starts. Every nested slice has an equivalent flat one, so reject
    // rather than compose.
    static_assert(!detail::is_multi_slice_view<Tensor>::value,
                  "MultiSliceView: nesting is not supported. Combine the slices into "
                  "one pack: MultiSliceView<T, Slice<0,i,1>, Slice<2,j,1>>");

    // Duplicated from MultiSlicedPadPolicy. The policy is where these
    // constraints actually bite, but repeating them here documents what a Slice
    // pack must satisfy at the point where it is written. Compile-time only, so
    // the duplication costs nothing.
    //
    // These run in view space, before the remap, so their diagnostics name the
    // axes the caller wrote. The extent check is the exception: it needs a
    // clamped axis to avoid indexing out of bounds when the axis is also wrong,
    // so it lives downstream in ValidateSlice and reports a policy axis.
    static_assert(sizeof...(Slices) > 0,
                  "MultiSliceView: at least one Slice is required");

    static_assert(sizeof...(Slices) <= Tensor::NumDims,
                  "MultiSliceView: more slices than the source has dimensions");

    static_assert(all_unique<Slices::Axis...>(),
                  "MultiSliceView: each axis may be sliced at most once");

    static_assert(max_value<Slices::Axis...>() < Tensor::NumDims,
                  "MultiSliceView: Axis must be less than the source's number of dimensions");

    static_assert(((Slices::Len > 0) && ...),
                  "MultiSliceView: Len must be greater than 0");

public:
    using value_type = typename Tensor::value_type;
    using SourceLayout = typename Tensor::Layout;
    using SourcePadPolicy = typename SourceLayout::PadPolicyType;

    // ---- boundary: view space above, policy space below ----

    using PadPolicy = MultiSlicedPadPolicy<SourcePadPolicy,
                                           detail::RemapSlice<SourceLayout, Slices>...>;

    // Re-apply the source's permutation to the narrowed policy, which brings
    // logical_dim() and stride() back out in view space
    using Layout = typename SourceLayout::template rebind<PadPolicy>;

    static constexpr my_size_t NumDims = Layout::NumDims;

    // The view's logical shape, read back out of the layout so it cannot
    // disagree with getDim
    static constexpr const my_size_t (&Dim)[NumDims] = LayoutDims<Layout>::value;

    static constexpr my_size_t TotalSize = Layout::LogicalSize;

    // Physical offset of the view's first element: every coordinate zero except
    // the sliced axes, which start at their offsets. Layout::stride is already
    // back in view space, so Slices::Axis needs no remapping here.
    static constexpr my_size_t PhysicalBase =
        ((Slices::Offset * Layout::stride(Slices::Axis)) + ...);

private:
    // ========================================================================
    // Contiguity detection
    // ========================================================================
    // The view occupies one unbroken physical span iff walking it in logical
    // row-major order produces consecutive physical offsets. That holds when,
    // for every axis i, either:
    //
    //   - the view's extent is 1, so the axis is never traversed and its
    //     stride is irrelevant (only coordinate 0 is ever used), or
    //   - the view's logical stride equals its physical stride, so stepping
    //     one element along that axis moves exactly that far in memory
    //
    // Two examples, double with SimdWidth=4:
    //
    //   Row of [4,6]  → LogicalDims [1,6], LogicalStrides [6,1], Strides [8,1]
    //     axis 0: extent 1, skipped.  axis 1: 1 == 1.  → contiguous
    //     (a row of a padded matrix really is 6 consecutive doubles)
    //
    //   Column of [4,6] → LogicalDims [4,1], LogicalStrides [1,1], Strides [8,1]
    //     axis 0: extent 4 and 1 != 8.  → strided, gather
    //     (consecutive elements sit one padded row apart)
    static constexpr bool computeIsContiguous() noexcept
    {
        for (my_size_t i = 0; i < NumDims; ++i)
        {
            if (Layout::logical_dim(i) == 1)
            {
                continue;
            }
            if (Layout::logical_stride(i) != Layout::stride(i))
            {
                return false;
            }
        }
        return true;
    }

public:
    // True when the view is one unbroken physical span and SIMD can load
    // directly instead of gathering. See computeIsContiguous above.
    static constexpr bool IsPhysicallyContiguous = computeIsContiguous();

    explicit MultiSliceView(const Tensor &t) noexcept
        : t_(t) {}

    // Views are non-copyable, non-movable, they're lightweight references.
    MultiSliceView(const MultiSliceView &) = delete;
    MultiSliceView &operator=(const MultiSliceView &) = delete;
    MultiSliceView(MultiSliceView &&) = delete;
    MultiSliceView &operator=(MultiSliceView &&) = delete;

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return t_.may_alias(output); // recurse to underlying tensor
    }

    // Const version of the access operator, because this is a view
    template <typename... Indices>
        requires(sizeof...(Indices) == NumDims)
    FORCE_INLINE const value_type &operator()(Indices... indices) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return t_.data()[PhysicalBase + Layout::logical_coords_to_physical_flat(idxArray)];
    }

    // Const version of the access operator with array of indices, because this is a view
    FORCE_INLINE const value_type &operator()(my_size_t (&indices)[NumDims]) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return t_.data()[PhysicalBase + Layout::logical_coords_to_physical_flat(indices)];
    }

    /**
     * @brief SIMD EVALUATION — direct load when contiguous, gather otherwise.
     *
     * When IsPhysicallyContiguous holds, consecutive logical elements sit at
     * consecutive physical offsets starting from PhysicalBase, so a plain load
     * suffices. Otherwise each logical flat index is remapped through the
     * sliced layout and the values are gathered.
     *
     * Example: column slice of [4,6], Slice<1,3,1>:
     *   logical_flat 0 → coords(0,0) → physical 0,  +3 → 3
     *   logical_flat 1 → coords(1,0) → physical 8,  +3 → 11
     *   logical_flat 2 → coords(2,0) → physical 16, +3 → 19
     *   logical_flat 3 → coords(3,0) → physical 24, +3 → 27
     *   → gather from offsets [3, 11, 19, 27]
     */
    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType evalu(my_size_t logical_flat) const noexcept
    {
        using K = Microkernel<T, Bits, Arch>;

        if constexpr (IsPhysicallyContiguous)
        {
            return K::load(t_.data() + PhysicalBase + logical_flat);
        }
        else
        {
            constexpr my_size_t width = K::simdWidth;

            my_size_t idxList[width];
            for (my_size_t i = 0; i < width; ++i)
                idxList[i] = PhysicalBase + Layout::logical_flat_to_physical_flat(logical_flat + i);

            return K::gather(t_.data(), idxList);
        }
    }

    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType logical_evalu(my_size_t logical_flat) const noexcept
    {
        // evalu already expects logical flat for sliced views
        return evalu<T, Bits, Arch>(logical_flat);
    }

    // this is the logical dimension of the view, not the source's physical layout
    FORCE_INLINE static constexpr my_size_t getDim(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::logical_dim(i);
    }

    // this is the stride in the source's physical layout, not the view's logical layout
    FORCE_INLINE static constexpr my_size_t getStride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::stride(i);
    }

    FORCE_INLINE static constexpr my_size_t getNumDims() noexcept { return NumDims; }

    FORCE_INLINE static constexpr my_size_t getTotalSize() noexcept { return TotalSize; }

    // print layout info
    void printLayoutInfo() const
    {
        MyErrorHandler::log("MultiSliceView Layout Info:", ErrorLevel::Info);
        MyErrorHandler::log("Number of Dimensions: " + std::to_string(NumDims), ErrorLevel::Info);
        MyErrorHandler::log("Shape: " + getShape(), ErrorLevel::Info);
        MyErrorHandler::log("Strides: ", ErrorLevel::Info);
        for (my_size_t i = 0; i < NumDims; ++i)
            MyErrorHandler::log(std::to_string(getStride(i)) + " ", ErrorLevel::Info);
        MyErrorHandler::log("\n", ErrorLevel::Info);
    }

    // Utility function to retrieve the shape of the tensor as (1,5,6) for a 3D tensor use the getNumDims
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

    FORCE_INLINE constexpr const value_type *data() const noexcept { return t_.data(); }

private:
    const Tensor &t_;
};

#endif // FUSED_MULTI_SLICE_VIEW_H
