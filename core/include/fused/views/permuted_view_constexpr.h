#ifndef FUSED_PERMUTED_VIEW_CONSTEXPR_H
#define FUSED_PERMUTED_VIEW_CONSTEXPR_H

#include "config.h"
#include "fused/BaseExpr.h"
#include "copy_n_optimized.h"
#include "fused/layouts/strided_layout_constexpr.h"
#include "helper_traits.h"

/**
 * @brief Compile-time permuted view over a tensor.
 *
 * Does not own or copy data — references the underlying tensor's physical buffer.
 * Permutation is applied entirely at compile time through StridedLayoutConstExpr.
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
 * @tparam Tensor  The underlying tensor type (e.g., FusedTensorND<double, 2, 3>)
 * @tparam Perm    Compile-time permutation indices
 */
template <typename Tensor, my_size_t... Perm>
class PermutedViewConstExpr : public BaseExpr<PermutedViewConstExpr<Tensor, Perm...>>
{
    // Static assertions to validate the permutation pack at compile time
    static_assert(sizeof...(Perm) == Tensor::NumDims,
                  "Permutation pack must match tensor's number of dimensions");

    static_assert(all_unique<Perm...>(),
                  "Permutation indices must be unique");

    static_assert(max_value<Perm...>() < Tensor::NumDims,
                  "Max value of permutation pack is greater than the tensor's number of dimensions");

    static_assert(min_value<Perm...>() == 0,
                  "Min value of permutation pack is not equal to 0");

public:
    using value_type = typename Tensor::value_type;
    using PadPolicy = typename Tensor::AccessPolicy::PadPolicy;
    using Layout = StridedLayoutConstExpr<PadPolicy, Perm...>;

    static constexpr my_size_t NumDims = Layout::NumDims;
    static constexpr my_size_t Dim[] = {Tensor::Dim[Perm]...};
    static constexpr my_size_t TotalSize = Tensor::TotalSize;

    explicit PermutedViewConstExpr(const Tensor &t) noexcept
        : t_(t) {}

    // Views are non-copyable, non-movable — they're lightweight references.
    // delete copy constructor and copy assignment to avoid accidental copies
    PermutedViewConstExpr(const PermutedViewConstExpr &) = delete;
    PermutedViewConstExpr &operator=(const PermutedViewConstExpr &) = delete;

    // delete move constructor and move assignment to avoid accidental moves
    PermutedViewConstExpr(PermutedViewConstExpr &&) = delete;
    PermutedViewConstExpr &operator=(PermutedViewConstExpr &&) = delete;

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
        return t_.data_.data()[Layout::logical_coords_to_physical_flat(idxArray)];
    }

    // Const version of the access operator with array of indices, because this is a view
    FORCE_INLINE const value_type &operator()(my_size_t (&indices)[NumDims]) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return t_.data_.data()[Layout::logical_coords_to_physical_flat(indices)];
    }

    FORCE_INLINE const value_type &operator()(const my_size_t *indices) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return t_.data_.data()[Layout::logical_coords_to_physical_flat(indices)];
    }

    // ========================================================================
    // PermutedViewConstExpr::evalu — logical flat, K::gather
    // ========================================================================
    // Only used by permuted path. Each logical flat index is remapped
    // through the permuted layout to a source physical offset.
    // Consecutive logical flats map to non-contiguous source positions → gather.

    /**
     * @brief SIMD EVALUATION — logical flat, K::gather
     *
     * Consecutive logical flat indices map to non-contiguous physical offsets
     * through the permuted layout, so gather is required.
     *
     * Example: transposed [3,2] view of [2,3] source:
     * logical_flat 0 → coords(0,0) → physical 0
     * logical_flat 1 → coords(0,1) → physical 4
     * logical_flat 2 → coords(1,0) → physical 1
     * logical_flat 3 → coords(1,1) → physical 5
     * → gather from offsets [0, 4, 1, 5]
     *
     * @tparam T   The value type for evaluation (e.g., float, double)
     * @tparam Bits Number of bits for the microkernel (e.g., 256 for AVX2)
     * @tparam Arch The target architecture for the microkernel (e.g., AVX2, AVX-512)
     * @param logical_flat The logical flat index to evaluate from
     * @return Microkernel vector type containing the evaluated values for the SIMD width
     */
    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType evalu(my_size_t logical_flat) const noexcept
    {
        using K = Microkernel<T, Bits, Arch>;
        constexpr my_size_t width = K::simdWidth;

        my_size_t idxList[width];
        for (my_size_t i = 0; i < width; ++i)
            idxList[i] = Layout::logical_flat_to_physical_flat(logical_flat + i);

        return K::gather(t_.data_.data(), idxList);
    }

    FORCE_INLINE static constexpr my_size_t getDim(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::logical_dim(i);
    }

    FORCE_INLINE static constexpr my_size_t getStride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::stride(i);
    }

    FORCE_INLINE static constexpr my_size_t getNumDims() noexcept { return NumDims; }

    FORCE_INLINE static constexpr my_size_t getTotalSize() noexcept { return TotalSize; }

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

    // Inverse permutation — restores the base tensor
    FORCE_INLINE const Tensor &transpose() const noexcept { return t_; }

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
    const Tensor &t_;

    FORCE_INLINE constexpr const value_type *data() const noexcept { return t_.data_.data(); }

    template <typename, my_size_t, typename>
    friend struct KernelOps;
};

#endif // FUSED_PERMUTED_VIEW_CONSTEXPR_H