#ifndef FUSED_DIAGONAL_VIEW_H
#define FUSED_DIAGONAL_VIEW_H

#include "config.h"
#include "fused/BaseExpr.h"

/**
 * @brief Compile-time diagonal view over a square matrix.
 *
 * This class does not own or copy data, it just references the underlying matrix's physical buffer.
 * All access respects the source's layout, padding, and strides via
 * Layout::logical_coords_to_physical_flat for bounds checking and correctness.
 * 
 * TODO: it can be generalized to rectangular matrices, but the current use case is square matrices only.
 *
 * Extracts the main diagonal of an N×N matrix as an N-element column vector.
 *
 *   Source A[3,3] with SimdWidth=4:
 *     Physical layout (padded to 3×4):
 *       [a00 a01 a02 _ | a10 a11 a12 _ | a20 a21 a22 _]
 *     Strides: [4, 1]
 *
 *   Diagonal view:
 *     view(0) → A(0,0) → physical 0
 *     view(1) → A(1,1) → physical 5
 *     view(2) → A(2,2) → physical 10
 *
 * Enables `trace = reduce_sum(diagonal(A))` via SIMD.
 *
 * @tparam Matrix  The underlying square matrix type
 */
template <typename Matrix>
class DiagonalView : public BaseExpr<DiagonalView<Matrix>>
{
    static_assert(Matrix::NumDims == 2,
                  "DiagonalView: source must have exactly 2 dimensions");

    static_assert(Matrix::Dim[0] == Matrix::Dim[1],
                  "DiagonalView: source must be a square matrix");

public:
    using value_type = typename Matrix::value_type;
    using Layout = typename Matrix::Layout;

    static constexpr my_size_t NumDims = Layout::NumDims;

    // Output shape: column vector [N, 1]
    // Note Matrix::Dim cannot be used directly for the view's shape because
    // the view collapses the N×N source into an N×1 diagonal
    static constexpr my_size_t Dim[] = {Matrix::Dim[0], 1};

    // Total size of the view is the diagonal length
    static constexpr my_size_t TotalSize = Matrix::Dim[0];

    explicit DiagonalView(const Matrix &m) noexcept
        : m_(m) {}

    // Non-copyable, non-movable
    DiagonalView(const DiagonalView &) = delete;
    DiagonalView &operator=(const DiagonalView &) = delete;
    DiagonalView(DiagonalView &&) = delete;
    DiagonalView &operator=(DiagonalView &&) = delete;

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        return m_.may_alias(output);
    }

    // Single-index access: view(i) → source(i, i)
    FORCE_INLINE const value_type &operator()(my_size_t i) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        my_size_t coords[] = {i, i};
        return m_.data()[Layout::logical_coords_to_physical_flat(coords)];
    }

    /**
     * @brief SIMD EVALUATION: layout-based, always gather.
     *
     * Diagonal elements are spaced by stride(0) + stride(1) in physical
     * memory, which is never 1, so gather is always required.
     * Physical offsets are computed through Layout for correctness.
     */
    template <typename T, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T, Bits, Arch>::VecType evalu(my_size_t logical_flat) const noexcept
    {
        using K = Microkernel<T, Bits, Arch>;
        constexpr my_size_t width = K::simdWidth;

        my_size_t idxList[width];
        for (my_size_t i = 0; i < width; ++i)
        {
            my_size_t idx = logical_flat + i;
            my_size_t coords[] = {idx, idx};
            idxList[i] = Layout::logical_coords_to_physical_flat(coords);
        }
        return K::gather(m_.data(), idxList);
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

    FORCE_INLINE constexpr const value_type *data() const noexcept { return m_.data(); }
    FORCE_INLINE constexpr value_type *data() noexcept { return m_.data(); }

    // print layout info
    void printLayoutInfo() const
    {
        MyErrorHandler::log("DiagonalView Layout Info:", ErrorLevel::Info);
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
    const Matrix &m_;
};

#endif // FUSED_DIAGONAL_VIEW_H
