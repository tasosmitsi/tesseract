#ifndef FUSED_SQUEEZE_VIEW_H
#define FUSED_SQUEEZE_VIEW_H

#include "config.h"
#include "fused/BaseExpr.h"
#include "fused/expr_storage.h"
#include "fused/layouts/strided_layout_constexpr.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/padding_policies/adapters/squeezed_pad_policy.h"

/**
 * @brief Compile-time view that drops size-1 axes from a tensor or another view.
 *
 * Does not own or copy data, it references the source's physical buffer.
 * Squeezing happens at compile time through StridedLayoutConstExpr over a
 * SqueezedPadPolicy, the same shape MultiSliceView uses.
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
 * the trailing-stride constraint in SqueezedPadPolicy.
 *
 * ============================================================================
 * NO PERMUTED SOURCES
 * ============================================================================
 *
 * A squeezed layout has fewer dimensions than a permutation pack has entries,
 * so the pack would have to be squeezed alongside and renumbered to stay a
 * valid permutation of 0..M-1. Every other translation in the view layer
 * preserves rank; this one does not. SourceLayout::rebind cannot help either,
 * since it re-applies the same Perm... pack and its size check would fire.
 *
 * No caller needs it. The static_assert below rejects it, and the layout is
 * built directly rather than through rebind.
 *
 * TODO: mutable squeezes, once set_row lands and there is a mutable view to
 * build on.
 *
 * @tparam Tensor       The tensor or view being squeezed
 * @tparam DroppedAxes  Axes to drop, each of logical extent 1
 */
template <typename Tensor, my_size_t... DroppedAxes>
class SqueezeView;

namespace detail
{
    /**
     * @brief Detects SqueezeView, so nesting can be rejected.
     *
     * Specialized before the class because the static_assert that uses it lives
     * in the class body. A partial specialization only needs the primary
     * template declared, which the forward declaration above provides.
     */
    template <typename T>
    struct is_squeeze_view
    {
        static constexpr bool value = false;
    };

    template <typename Tensor, my_size_t... DroppedAxes>
    struct is_squeeze_view<SqueezeView<Tensor, DroppedAxes...>>
    {
        static constexpr bool value = true;
    };

    /**
     * @brief The source's PhysicalBase, or 0 for a source that has none.
     *
     * FusedTensorND has no PhysicalBase; MultiSliceView does. Both are valid
     * sources, so the offset is detected rather than required.
     */
    template <typename Tensor>
    static constexpr my_size_t source_physical_base() noexcept
    {
        if constexpr (requires { Tensor::PhysicalBase; })
            return Tensor::PhysicalBase;
        else
            return my_size_t(0);
    }
} // namespace detail

template <typename Tensor, my_size_t... DroppedAxes>
class SqueezeView : public BaseExpr<SqueezeView<Tensor, DroppedAxes...>>
{
    // Two squeezes compose into one pack, so reject rather than compose. This
    // is cheaper to verify than the alternative, and the flat form is always
    // available.
    static_assert(!detail::is_squeeze_view<Tensor>::value,
                  "SqueezeView: nesting is not supported. Combine the axes into one "
                  "pack: SqueezeView<T, 0, 2>");

public:
    using value_type = typename Tensor::value_type;
    using SourceLayout = typename Tensor::Layout;

    // Through the layout rather than Tensor::AccessPolicy::PadPolicy, which is
    // private on FusedTensorND and only reachable through a friend declaration.
    using SourcePadPolicy = typename SourceLayout::PadPolicyType;

private:
    static_assert(!SourceLayout::IsPermProvided,
                  "SqueezeView: permuted sources are not supported. Squeezing a "
                  "permutation needs the pack squeezed and renumbered alongside it");

public:
    using PadPolicy = SqueezedPadPolicy<SourcePadPolicy, DroppedAxes...>;

    // SourceLayout::rebind would expand to this same type, since the assert
    // above leaves Perm empty. Written directly because rebind means "re-apply
    // the source's permutation", which a rank-changing view cannot do: the pack
    // has the source's rank and the policy has fewer dims. It reads as legal
    // only because of an assert several lines away.
    using Layout = StridedLayoutConstExpr<PadPolicy>;

    static constexpr my_size_t NumDims = Layout::NumDims;

    // Read back out of the layout so it cannot disagree with getDim
    static constexpr const my_size_t (&Dim)[NumDims] = LayoutDims<Layout>::value;

    static constexpr my_size_t TotalSize = Layout::LogicalSize;

    // Inherited from the source, since dropping an axis moves nothing. A
    // squeeze of a squeeze would accumulate here correctly; nesting is rejected
    // above for a different reason.
    static constexpr my_size_t PhysicalBase = detail::source_physical_base<Tensor>();

private:
    // The view occupies one unbroken physical span iff walking it in logical
    // row-major order produces consecutive physical offsets. That holds when,
    // for every axis, either the extent is 1 and the axis is never traversed,
    // or the logical stride equals the physical one.
    //
    // The first caller lands here: a [1,1,2,4] slice of a padded [Na,Nb,2,4]
    // squeezes to [2,4] with LogicalStrides [4,1] and Strides [4,1], so both
    // axes match and evalu loads instead of gathering.
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
    static constexpr bool IsPhysicallyContiguous = computeIsContiguous();

    explicit SqueezeView(const Tensor &t) noexcept
        : t_(t) {}

    // Copyable and movable, so an expression can keep its own copy of a view
    // that was a temporary (see fused/expr_storage.h). Not assignable: a view
    // cannot be re-pointed.
    SqueezeView(const SqueezeView &) = default;
    SqueezeView &operator=(const SqueezeView &) = delete;
    SqueezeView(SqueezeView &&) = default;
    SqueezeView &operator=(SqueezeView &&) = delete;

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return t_.may_alias(output); // recurse to the source
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
     * @brief SIMD EVALUATION, direct load when contiguous, gather otherwise.
     *
     * Same two paths as MultiSliceView, for the same reason: consecutive
     * logical elements sit at consecutive physical offsets from PhysicalBase
     * when the view is one span, and need remapping through the layout when it
     * is not.
     *
     * Squeezing a contiguous slice stays contiguous, so the interpolation
     * corners take the load path.
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
        // evalu already expects logical flat for squeezed views
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
        MyErrorHandler::log("SqueezeView Layout Info:", ErrorLevel::Info);
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
    ExprStorage<Tensor> t_;
};

#endif // FUSED_SQUEEZE_VIEW_H
