#ifndef FUSED_SUBVECTOR_VIEW_H
#define FUSED_SUBVECTOR_VIEW_H

#include "config.h"
#include "fused/BaseExpr.h"

/**
 * @brief Compile-time subvector view over a vector-like tensor.
 *
 * Does not own or copy data — references the underlying vector's physical buffer.
 * All access respects the source's layout, padding, and strides via
 * Layout::logical_coords_to_physical_flat for bounds checking and correctness.
 *
 * Automatically detects whether the source is a column vector (Dim[1]==1)
 * or a row vector (Dim[0]==1) and slices along the non-trivial axis.
 *
 * Works with FusedVector, FusedMatrix<T,N,1>, FusedMatrix<T,1,N>,
 * and transposed views of any of these.
 *
 *   Column vector v[5,1] with SimdWidth=4:
 *     Axis=0, view(i) → source(Offset + i, 0)
 *
 *   Row vector (transposed) v[1,5]:
 *     Axis=1, view(i) → source(0, Offset + i)
 *
 * @tparam Vector  The underlying vector-like type
 * @tparam Offset  Compile-time start index along the non-trivial axis
 * @tparam Len     Compile-time number of elements in the view
 */
template <typename Vector, my_size_t Offset, my_size_t Len>
class SubVectorView : public BaseExpr<SubVectorView<Vector, Offset, Len>>
{
    static_assert(Vector::NumDims == 2,
                  "SubVectorView: source must have exactly 2 dimensions");

    static_assert(Vector::Dim[0] == 1 || Vector::Dim[1] == 1,
                  "SubVectorView: source must be a vector (one dimension must be 1)");

    static_assert(Offset + Len <= Vector::TotalSize,
                  "SubVectorView: Offset + Len exceeds source vector length");

    static_assert(Len > 0,
                  "SubVectorView: Len must be greater than 0");

public:
    using value_type = typename Vector::value_type;
    using Layout = typename Vector::Layout;

    // Auto-detect: slice along whichever dimension is non-trivial
    static constexpr my_size_t Axis = (Vector::Dim[0] == 1) ? 1 : 0;

    static constexpr my_size_t NumDims = Layout::NumDims;

    // Output shape mirrors source orientation: column in → column out, row in → row out
    // Note Vector::Dim[Axis] cannot be used here because
    // the view's Dims are dependent on Len
    static constexpr my_size_t Dim[] = {
        (Axis == 0) ? Len : 1,
        (Axis == 1) ? Len : 1};

    // Total size of the view is just the length of the subvector
    static constexpr my_size_t TotalSize = Len;

    // Stride along the sliced axis in the source's physical layout
    static constexpr my_size_t SliceStride = Layout::stride(Axis);

    // Physical offset of the first element in this view
    static constexpr my_size_t PhysicalBase =
        (Axis == 0)
            ? Layout::logical_coords_to_physical_flat(Offset, 0)
            : Layout::logical_coords_to_physical_flat(0, Offset);

    explicit SubVectorView(const Vector &v) noexcept
        : v_(v) {}

    // Non-copyable, non-movable — lightweight reference
    SubVectorView(const SubVectorView &) = delete;
    SubVectorView &operator=(const SubVectorView &) = delete;
    SubVectorView(SubVectorView &&) = delete;
    SubVectorView &operator=(SubVectorView &&) = delete;

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return v_.may_alias(output);
    }

    // Single-index access: view(i) → source along the non-trivial axis
    FORCE_INLINE const value_type &operator()(my_size_t i) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        if constexpr (Axis == 0)
        {
            my_size_t coords[] = {Offset + i, 0};
            return v_.data()[Layout::logical_coords_to_physical_flat(coords)];
        }
        else
        {
            my_size_t coords[] = {0, Offset + i};
            return v_.data()[Layout::logical_coords_to_physical_flat(coords)];
        }
    }

    /**
     * @brief SIMD EVALUATION — stride-aware, layout-based.
     *
     * Elements are spaced by SliceStride in physical memory.
     * If SliceStride == 1 (e.g. materialized row vector with contiguous columns),
     * a direct load suffices. Otherwise, physical offsets are computed through
     * Layout::logical_coords_to_physical_flat and gathered.
     */
    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType evalu(my_size_t logical_flat) const noexcept
    {
        using K = Microkernel<T, Bits, Arch>;

        if constexpr (SliceStride == 1)
        {
            return K::load(v_.data() + PhysicalBase + logical_flat);
        }
        else
        {
            constexpr my_size_t width = K::simdWidth;
            my_size_t idxList[width];
            for (my_size_t i = 0; i < width; ++i)
            {
                if constexpr (Axis == 0)
                {
                    my_size_t coords[] = {Offset + logical_flat + i, 0};
                    idxList[i] = Layout::logical_coords_to_physical_flat(coords);
                }
                else
                {
                    my_size_t coords[] = {0, Offset + logical_flat + i};
                    idxList[i] = Layout::logical_coords_to_physical_flat(coords);
                }
            }
            return K::gather(v_.data(), idxList);
        }
    }

    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType logical_evalu(my_size_t logical_flat) const noexcept
    {
        return evalu<T, Bits, Arch>(logical_flat);
    }

    // this is the logical dimension of the view, not the source's physical layout
    FORCE_INLINE static constexpr my_size_t getDim(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Dim[i];
    }

    // this is the stride in the source's physical layout, not the view's logical layout
    FORCE_INLINE static constexpr my_size_t getStride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::stride(i);
    }

    FORCE_INLINE static constexpr my_size_t getNumDims() noexcept { return NumDims; }
    FORCE_INLINE static constexpr my_size_t getTotalSize() noexcept { return TotalSize; }

    FORCE_INLINE constexpr const value_type *data() const noexcept { return v_.data(); }
    FORCE_INLINE constexpr value_type *data() noexcept { return v_.data(); }

    // print layout info
    void printLayoutInfo() const
    {
        MyErrorHandler::log("PermutedView Layout Info:", ErrorLevel::Info);
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

private:
    const Vector &v_;
};

#endif // FUSED_SUBVECTOR_VIEW_H
