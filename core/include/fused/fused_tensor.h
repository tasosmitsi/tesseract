#ifndef FUSEDTENSORND_H
#define FUSEDTENSORND_H

#include <random>

#include "copy_n_optimized.h"

#include "config.h"
#include "helper_traits.h"
#include "simple_type_traits.h"

#include "fused/BaseExpr.h"
#include "fused/operators/Operators.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/kernel_ops/kernel_ops.h"
#include "fused/storage/static_storage.h"
#include "fused/storage/dynamic_storage.h"
#include "fused/padding_policies/simd_padding_policy.h"
#include "fused/padding_policies/no_padding_policy.h"
#include "fused/access/dense_access.h"
#include "fused/access/sparse_access.h"
// #include "fused/views/permuted_view.h"
#include "fused/views/permuted_view_constexpr.h"
// #include "fused/layouts/strided_layout.h"
#include "fused/layouts/strided_layout_constexpr.h"
#include "algebra/algebraic_traits.h"

// Base class: FusedTensorND
template <typename T, my_size_t... Dims>
class FusedTensorND : public BaseExpr<FusedTensorND<T, Dims...>>
{
public:
    // Compile time constants
    static constexpr my_size_t NumDims = sizeof...(Dims);
    static constexpr my_size_t Dim[] = {Dims...};
    static constexpr my_size_t TotalSize = (Dims * ...);
    using value_type = T;
    using Self = FusedTensorND<T, Dims...>;

    // Default constructors
    FusedTensorND() noexcept = default;

    // Constructor to initialize all elements to a specific value
    explicit FusedTensorND(T initValue) noexcept
        : data_(initValue) {}

    // Copy constructor
    FusedTensorND(const FusedTensorND &other) noexcept
        : data_(other.data_) // invoke copy constructor of AccessPolicy
    {
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("Copy constructor called", ErrorLevel::Info);
#endif
        if (this == &other)
        {
#ifdef DEBUG_FUSED_TENSOR
            MyErrorHandler::log("Self-assignment detected, skipping copy.", ErrorLevel::Info);
#endif
            return; // Handle self-assignment
        }
    }

    // Move constructor
    FusedTensorND(FusedTensorND &&other) noexcept
        : data_(move(other.data_)) // invoke move constructor of AccessPolicy
    {
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("Move constructor called", ErrorLevel::Info);
#endif
        if (this == &other)
        {
#ifdef DEBUG_FUSED_TENSOR
            MyErrorHandler::log("Self-assignment detected, skipping move.", ErrorLevel::Info);
#endif
            return; // Handle self-assignment
        }
    }

    template <typename Output>
    bool may_alias(const Output &output) const noexcept
    {
        // So the if constexpr is an optimization — when the compiler knows aliasing is impossible,
        // it skips the check. When it can't know (same type), it defers to runtime.
        if constexpr (is_same_v<remove_cvref_t<Output>, FusedTensorND>)
        {
            return this == &output;
        }
        else
        {
            return false;
        }
    }

    template <typename Expr>
    FusedTensorND &operator=(const BaseExpr<Expr> &expr)
    {
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("FusedTensorND assignment operator called", ErrorLevel::Info);
#endif
        const auto &e = expr.derived();

        if (e.may_alias(*this))
        {
            MyErrorHandler::log("Aliasing detected in assignment operator", ErrorLevel::Warning);
        }

        // check if the dimensions match at compile time
        if constexpr (NumDims != Expr::NumDims)
        {
            MyErrorHandler::error("Dimensions count mismatch in assignment operator");
        }
        if constexpr (!dims_match<NumDims>(Dim, Expr::Dim))
        {
            MyErrorHandler::error("Dimensions size mismatch in assignment operator");
        }

        KernelOps<T, BITS, DefaultArch>::eval(
            data_.data(), e);

        return *this;
    }

    // ========================================================================
    // FusedTensorND::evalu — physical flat ONLY, K::load
    // ========================================================================
    // Treats flat as a PHYSICAL offset into the padded buffer.
    // Used by the contiguous kernel path which iterates physical slices.
    //
    // WARNING: Do NOT pass logical flat indices to this function when
    // padding exists (lastDim != paddedLastDim). Use logical_evalu instead.
    template <typename T_, my_size_t Bits, typename Arch>
    typename Microkernel<T_, Bits, Arch>::VecType evalu(my_size_t flat) const noexcept
    {
        using K = Microkernel<T_, Bits, Arch>;
        return K::load(data_.data() + flat);
    }

    /**
     * @brief Evaluate at a LOGICAL flat index.
     *
     * Unlike evalu (which takes physical offsets), this converts
     * logical flat → physical flat via Layout, handling padding gaps.
     * Uses gather for SIMD widths > 1 since consecutive logical flats
     * are not contiguous in physical memory when padding exists.
     */
    template <typename T_, my_size_t Bits, typename Arch>
    FORCE_INLINE typename Microkernel<T_, Bits, Arch>::VecType
    logical_evalu(my_size_t logical_flat) const noexcept
    {
        using K = Microkernel<T_, Bits, Arch>;

        if constexpr (K::simdWidth == 1)
        {
            return K::load(data_.data() +
                           Layout::logical_flat_to_physical_flat(logical_flat));
        }
        else
        {
            my_size_t idxList[K::simdWidth];
            for (my_size_t i = 0; i < K::simdWidth; ++i)
                idxList[i] = Layout::logical_flat_to_physical_flat(logical_flat + i);
            return K::gather(data_.data(), idxList);
        }
    }

    FusedTensorND &operator=(const FusedTensorND &other) noexcept
    {
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("FusedTensorND copy assignment", ErrorLevel::Info);
#endif
        if (this == &other)
        {
#ifdef DEBUG_FUSED_TENSOR
            MyErrorHandler::log("Self-assignment detected, skipping copy.", ErrorLevel::Info);
#endif
            return *this; // Handle self-assignment
        }

        // Copy the data
        data_ = other.data_; // calls the copy assignment of AccessPolicy
        return *this;
    }

    // move assignment operator
    FusedTensorND &operator=(FusedTensorND &&other) noexcept
    {
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("FusedTensorND move assignment", ErrorLevel::Info);
#endif
        if (this == &other)
        {
#ifdef DEBUG_FUSED_TENSOR
            MyErrorHandler::log("Self-assignment detected, skipping move.", ErrorLevel::Info);
#endif
            return *this; // Handle self-assignment
        }

        // Move the data
        data_ = move(other.data_); // calls the move assignment of AccessPolicy
        return *this;
    }

    // Variadic access operator for accessing tensor elements with separate indices
    template <typename... Indices>
        requires(sizeof...(Indices) == NumDims)
    inline T &operator()(Indices... indices) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...}; // Convert indices to an array
        return data_[Layout::logical_coords_to_physical_flat(idxArray)];
    }

    // Const version of the variadic access operator
    template <typename... Indices>
        requires(sizeof...(Indices) == NumDims)
    inline const T &operator()(Indices... indices) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return data_[Layout::logical_coords_to_physical_flat(idxArray)];
    }

    // version of passing a pointer to indices array eg _tensor1(indices1), indices1 is a pointer to an array of known size
    inline T &operator()(const my_size_t *indices) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        // Unsafe — caller must guarantee NumDims elements.
        return data_[Layout::logical_coords_to_physical_flat(indices)];
    }

    inline const T &operator()(const my_size_t *indices) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        // Unsafe — caller must guarantee NumDims elements.
        return data_[Layout::logical_coords_to_physical_flat(indices)];
    }

    // version of passing a array of indices eg _tensor1(indices1), indices1 is an array of known size use template
    inline T &operator()(my_size_t (&indices)[NumDims]) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return data_[Layout::logical_coords_to_physical_flat(indices)];
    }

    inline const T &operator()(my_size_t (&indices)[NumDims]) const TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return data_[Layout::logical_coords_to_physical_flat(indices)];
    }

    // check if all dimensions are the same at compile time
    static constexpr bool areDimsEqual()
    {
        return all_equal<Dims...>();
    }

    bool isIdentity() const
    {
        // Check if the tensor is "square" (hypercube). If the tensor
        // is not square, it cannot be identity -> return false
        if (!areDimsEqual())
        {
            return false;
        }

        // Calculate all indices combinations for all dimensions
        static constexpr my_size_t total_combinations = (1 * ... * Dims); // fold expression to calculate the total number of combinations
        my_size_t combinations[total_combinations][sizeof...(Dims)];      // 2D array to store all combinations
        static constexpr my_size_t max_vals[sizeof...(Dims)] = {Dims...}; // array to store the maximum values for each dimension
        generate_combinations(max_vals, combinations);                    // generate all combinations

        for (my_size_t i = 0; i < total_combinations; ++i)
        {
            // itterate over all dimensions
            // if all indices are the same, then it's a diagonal element
            bool isElementDiagonal = true;
            for (my_size_t j = 0; j < getNumDims(); ++j)
            {
                if (combinations[i][j] != combinations[i][0])
                {
                    isElementDiagonal = false;
                    break;
                }
            }

            if (isElementDiagonal)
            {
                // if the element is diagonal, check if it is equal to 1.
                // element - 1 must be greater than the precision tolerance
                if (std::abs((*this)(combinations[i]) - 1) > PRECISION_TOLERANCE)
                {
                    return false;
                }
            }
            else
            {
                // if the element is not diagonal, check if it is equal to 0.
                // element must be less than the precision tolerance
                if (!(std::abs((*this)(combinations[i])) < PRECISION_TOLERANCE))
                {
                    return false;
                }
            }
        }
        return true;
    }

    // Generic transpose_view by pack
    template <my_size_t... Perm>
    FORCE_INLINE auto transpose_view() const noexcept
    {
        // static_assert to check that Permutation pack is valid are in PermutedViewConstExpr
        return PermutedViewConstExpr<Self, Perm...>(*this);
    }

    FORCE_INLINE auto transpose_view(void) const noexcept
    {
        // since for 2D tenosrs the permutation of axis is known
        // at compile time we can use PermutedViewConstExpr
        static_assert(sizeof...(Dims) == 2, "Transpose is only supported for 2D tensors");
        return PermutedViewConstExpr<Self, 1, 0>(*this);
    }

    // FORCE_INLINE auto transpose_view(const my_size_t perm[NumDims]) const noexcept
    // {
    //     return PermutedView<Self, NumDims>(*this, perm);
    // }

    FORCE_INLINE static constexpr my_size_t getTotalSize() noexcept
    {
        return TotalSize;
    }

    FORCE_INLINE static constexpr my_size_t getNumDims() noexcept
    {
        return NumDims;
    }

    // Utility function to retrieve the shape of the tensor as (1,5,6) for a 3D tensor use the getNumDims
    std::string getShape() const
    {
        std::string shape = "(";
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            shape += std::to_string(getDim(i));
            if (i < getNumDims() - 1)
                shape += ",";
        }
        shape += ")";
        return shape;
    }

    FusedTensorND &setToZero(void) noexcept
    {
        // Safe to fill entire physical buffer — padding stays 0 too
        for (my_size_t i = 0; i < Layout::PhysicalSize; ++i)
            data_[i] = T{};
        return *this;
    }

    FusedTensorND &setHomogen(T _val) noexcept
    {
        /// Safe to fill entire physical buffer with the same value
        for (my_size_t i = 0; i < Layout::PhysicalSize; ++i)
            data_[i] = _val;
        return *this;
    }

    FusedTensorND &setRandom(T _maxRand, T _minRand)
    {
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));

        if constexpr (std::is_floating_point<T>::value)
        {
            std::uniform_real_distribution<T> dist(_minRand, _maxRand);
            for (my_size_t i = 0; i < TotalSize; ++i)
                data_[Layout::logical_flat_to_physical_flat(i)] = dist(rng);
        }
        else
        {
            std::uniform_int_distribution<T> dist(_minRand, _maxRand);
            for (my_size_t i = 0; i < TotalSize; ++i)
                data_[Layout::logical_flat_to_physical_flat(i)] = dist(rng);
        }
        return *this;
    }

    // for all dimensions
    FusedTensorND &setDiagonal(T _val)
    {
        static_assert(sizeof...(Dims) >= 2, "setDiagonal requires at least 2 dimensions.");

        // set the entire matrix to zeros
        setToZero();

        // Calculate the minimum dimension
        constexpr my_size_t minDim = min_value<Dims...>();
        my_size_t indices[NumDims] = {0}; // Initialize all indices to zero

        for (my_size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (my_size_t d = 0; d < getNumDims(); ++d)
            {
                indices[d] = i; // Set the diagonal index, others to zero
            }

            // Calculate the index in the flat array and set the value
            data_[Layout::logical_coords_to_physical_flat(indices)] = _val;
        }
        return *this;
    }

    FusedTensorND &setIdentity(void)
    {
        static_assert(sizeof...(Dims) >= 2, "Identity requires at least 2 dimensions.");
        static_assert(all_equal<Dims...>(), "All dimensions must be equal for an identity tensor");

        this->setDiagonal(1);
        return *this;
    }

    // static FusedTensorND I(void)
    // {
    //     static_assert(sizeof...(Dims) >= 2, "Identity requires at least 2 dimensions.");

    //     static_assert(all_equal<Dims...>(), "All dimensions must be equal for an identity tensor");

    //     FusedTensorND<T, Dims...> _outp;
    //     _outp.setDiagonal(1);
    //     return _outp;
    // }

    FusedTensorND &setSequencial(void)
    {
        // Only set logical elements — padding must stay uninitialized
        for (my_size_t i = 0; i < TotalSize; ++i)
        {
            data_[Layout::logical_flat_to_physical_flat(i)] = static_cast<T>(i);
        }
        return *this;
    }

    template <my_size_t DiagonalSize>
    void getDiagonalEntries(FusedTensorND<T, DiagonalSize, 1> &diagonalEntries) const // TODO: needs to be tested
    {
        static_assert(sizeof...(Dims) >= 2, "Getting diagonal entries requires at least 2 dimensions.");
        // Calculate the minimum dimension
        my_size_t minDim = std::min({Dims...}); // Using initializer list to find the minimum TODO: std::min can be replaced with by helper_trait min_value
        my_size_t indices[getNumDims()] = {0};  // Initialize all indices to zero

        for (my_size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (my_size_t d = 0; d < getNumDims(); ++d)
            {
                indices[d] = i; // Set the diagonal index, others to zero
            }

            // Calculate the index in the flat array and set the value
            diagonalEntries(i, 0) = data_[Layout::logical_coords_to_physical_flat(indices)];
        }
    }

    /**
     * @brief Contract two tensors along specified axes using SIMD dot products.
     *
     * For 2D tensors, always dispatches to register-blocked GEMM by
     * materializing transposed copies when needed. The O(N²) transpose
     * cost is negligible vs O(N³) multiply, and the materialized tensor
     * has proper SIMD-aligned padding for aligned K::load in the micro-kernel.
     *
     * For higher-dimensional tensors, falls back to generic stride-mapped
     * per-element dot products.
     *
     * ============================================================================
     * 2D GEMM — 4 CASES (contract axis a from tensor1, axis b from tensor2)
     * ============================================================================
     *
     *   a=1, b=0: C[M,N] = A[M,K] × B[K,N]   — favorable, no transpose
     *   a=0, b=0: C[K,N] = A^T[K,M] × B[M,N]  — transpose A
     *   a=1, b=1: C[M,K] = A[M,N] × B^T[N,K]  — transpose B
     *   a=0, b=1: C[K,K'] = A^T × B^T           — transpose both
     *
     * ============================================================================
     */
    template <typename LeftExpr, typename RightExpr>
        requires(expression::traits<LeftExpr>::IsPhysical &&
                 expression::traits<RightExpr>::IsPhysical)
    static FusedTensorND einsum(
        const BaseExpr<LeftExpr> &_tensor1,
        const BaseExpr<RightExpr> &_tensor2,
        const my_size_t a,
        const my_size_t b)
    {
        static constexpr my_size_t Dims1 = LeftExpr::NumDims;
        static constexpr my_size_t Dims2 = RightExpr::NumDims;

        static_assert(Dims1 >= 2, "Tensor 1 must have at least 2 dimensions");
        static_assert(Dims2 >= 2, "Tensor 2 must have at least 2 dimensions");

        // Runtime validation
        if (a >= Dims1 || b >= Dims2)
            MyErrorHandler::error("Invalid contraction axis");

        if (_tensor1.derived().getDim(a) != _tensor2.derived().getDim(b))
            MyErrorHandler::error("Contraction dimensions mismatch");

        using Layout1 = typename LeftExpr::Layout;
        using Layout2 = typename RightExpr::Layout;
        using OutputLayout = Layout;
        using Kern = KernelOps<T, BITS, DefaultArch>;

        const my_size_t K_len = _tensor1.derived().getDim(a);
        const my_size_t contract_stride1 = Layout1::stride(a);
        const my_size_t contract_stride2 = Layout2::stride(b);

        // ====================================================================
        // Build and validate output dimensions
        // ====================================================================

        static constexpr my_size_t n_newDims = Dims1 + Dims2 - 2;
        my_size_t out_dims[n_newDims];

        my_size_t d = 0;
        for (my_size_t i = 0; i < Dims1; ++i)
            if (i != a)
                out_dims[d++] = _tensor1.derived().getDim(i);
        for (my_size_t i = 0; i < Dims2; ++i)
            if (i != b)
                out_dims[d++] = _tensor2.derived().getDim(i);

        FusedTensorND _outp;
        for (my_size_t i = 0; i < n_newDims; ++i)
        {
            if (out_dims[i] != _outp.getDim(i))
                MyErrorHandler::error("Output dimensions mismatch");
        }

        // ====================================================================
        // 2D GEMM path — always favorable after optional transpose
        // ====================================================================

        if constexpr (Dims1 == 2 && Dims2 == 2)
        {
            // Lambda: run GEMM given ready-to-go tensors and their axes
            // A_ready has contraction on its last dim (stride 1)
            // B_ready has free dim on its last dim (stride 1)
            auto run_gemm = [&](const auto &A_ready, const auto &B_ready)
            {
                using LayoutA = typename std::remove_cvref_t<decltype(A_ready)>::Layout;
                using LayoutB = typename std::remove_cvref_t<decltype(B_ready)>::Layout;

                const my_size_t M = A_ready.getDim(0);
                const my_size_t N = B_ready.getDim(1);

                detail::KernelGemm<T, BITS, DefaultArch>::gemm(
                    A_ready.data(), M, K_len, LayoutA::stride(0),
                    B_ready.data(), N, LayoutB::stride(0),
                    _outp.data(), OutputLayout::stride(0));
            };

            auto make_transposed = [](const auto &expr)
            {
                using E = remove_cvref_t<decltype(expr)>;
                if constexpr (!requires { expr.transpose(); })
                {
                    // FusedTensorND: materialize transpose_view
                    FusedTensorND<typename E::value_type, E::Dim[1], E::Dim[0]> dst;
                    dst = expr.transpose_view();
                    return dst;
                }
                else if constexpr (expression::traits<E>::IsPermuted)
                {
                    // Real permuted view (e.g. <1,0>): .transpose() returns the
                    // base tensor, whose physical layout IS the transposed data
                    return expr.transpose();
                }
                else
                {
                    // Identity permuted view (e.g. <0,1>): .transpose() returns
                    // the base tensor with SAME dims — need to actually transpose
                    auto &base = expr.transpose();
                    FusedTensorND<typename E::value_type, E::Dim[1], E::Dim[0]> dst;
                    dst = base.transpose_view();
                    return dst;
                }
            };

            auto ensure_materialized = [](const auto &expr)
            {
                using E = remove_cvref_t<decltype(expr)>;
                if constexpr (!requires { expr.transpose(); })
                {
                    // FusedTensorND: physical layout already correct
                    return expr;
                }
                else if constexpr (!expression::traits<E>::IsPermuted)
                {
                    // Identity view: base tensor has matching physical layout
                    return expr.transpose();
                }
                else
                {
                    // Real permuted view: physical layout doesn't match logical dims
                    FusedTensorND<typename E::value_type, E::Dim[0], E::Dim[1]> dst;
                    dst = expr;
                    return dst;
                }
            };

            if (a == 1 && b == 0)
            {
                if constexpr (requires { _tensor1.derived().transpose(); } || requires { _tensor2.derived().transpose(); })
                {
                    // At least one input is a PermutedViewConstExpr —
                    // materialize to fix physical layout
                    run_gemm(ensure_materialized(_tensor1.derived()),
                             ensure_materialized(_tensor2.derived()));
                }
                else
                {
                    // Both are FusedTensorND — physical layout guaranteed favorable
                    run_gemm(_tensor1.derived(), _tensor2.derived());
                }
            }
            else if (a == 0 && b == 0)
            {
                // std::cout << "Running GEMM with A transposed\n";
                auto A_t = make_transposed(_tensor1.derived());
                run_gemm(A_t, _tensor2.derived());
            }
            else if (a == 1 && b == 1)
            {
                // std::cout << "Running GEMM with B transposed\n";
                auto B_t = make_transposed(_tensor2.derived());
                run_gemm(_tensor1.derived(), B_t);
            }
            else
            {
                // std::cout << "Running GEMM with both A and B transposed\n";
                auto A_t = make_transposed(_tensor1.derived());
                auto B_t = make_transposed(_tensor2.derived());
                run_gemm(A_t, B_t);
            }

            return _outp;
        }

        // ====================================================================
        // Generic fallback: stride-mapped per-element dot products
        // ====================================================================

        my_size_t strides1_map[n_newDims];
        my_size_t strides2_map[n_newDims];

        d = 0;
        for (my_size_t i = 0; i < Dims1; ++i)
        {
            if (i != a)
            {
                strides1_map[d] = Layout1::stride(i);
                strides2_map[d] = 0;
                ++d;
            }
        }
        for (my_size_t i = 0; i < Dims2; ++i)
        {
            if (i != b)
            {
                strides1_map[d] = 0;
                strides2_map[d] = Layout2::stride(i);
                ++d;
            }
        }

        my_size_t out_strides[n_newDims];
        for (my_size_t i = 0; i < n_newDims; ++i)
            out_strides[i] = OutputLayout::stride(i);

        T *out_ptr = _outp.data();

        static constexpr my_size_t total_elements = (1 * ... * Dims);

        for (my_size_t flat = 0; flat < total_elements; ++flat)
        {
            my_size_t coords[n_newDims];
            my_size_t tmp = flat;
            for (my_size_t i = n_newDims; i-- > 0;)
            {
                coords[i] = tmp % out_dims[i];
                tmp /= out_dims[i];
            }

            my_size_t base1 = 0;
            my_size_t base2 = 0;
            my_size_t out_phys = 0;
            for (my_size_t i = 0; i < n_newDims; ++i)
            {
                base1 += coords[i] * strides1_map[i];
                base2 += coords[i] * strides2_map[i];
                out_phys += coords[i] * out_strides[i];
            }

            out_ptr[out_phys] = Kern::dot(
                _tensor1.derived(), base1, contract_stride1,
                _tensor2.derived(), base2, contract_stride2,
                K_len);
        }

        return _outp;
    }

    // Function to print the contents of the tensor
    void print(bool with_padding = false) const
    {
        printND(with_padding);
    }

    /**
     * @brief Print tensor of arbitrary dimensions.
     *
     * Convention: last 2 dims are (rows, cols), earlier dims are slice indices.
     * For 1D: prints a single row.
     * For 2D: prints a matrix.
     * For 3D+: prints labeled 2D slices.
     *
     * @param showPadding If true, show padding elements after a '|' separator.
     */
    void printND(bool showPadding = false) const
    {
        static constexpr my_size_t ND = Layout::NumDims;

        my_size_t coords[ND] = {};

        // Number of 2D slices = product of dims 0..ND-3
        my_size_t numSlices = 1;
        for (my_size_t d = 0; d + 2 < ND; ++d)
            numSlices *= getDim(d);

        const my_size_t rowDim = (ND >= 2) ? getDim(ND - 2) : 1;
        const my_size_t colDim = getDim(ND - 1);
        const my_size_t physColDim = Layout::PadPolicyType::PhysicalDims.at(ND - 1);

        for (my_size_t s = 0; s < numSlices; ++s)
        {
            // Print slice header for 3D+
            if constexpr (ND > 2)
            {
                MyErrorHandler::log("Slice [");
                for (my_size_t d = 0; d + 2 < ND; ++d)
                {
                    if (d > 0)
                        MyErrorHandler::log(", ");
                    MyErrorHandler::log(coords[d]);
                }
                MyErrorHandler::log("]:\n");
            }

            // Print 2D matrix
            for (my_size_t i = 0; i < rowDim; ++i)
            {
                if constexpr (ND >= 2)
                    coords[ND - 2] = i;

                // Logical elements
                for (my_size_t j = 0; j < colDim; ++j)
                {
                    coords[ND - 1] = j;
                    my_size_t offset = Layout::logical_coords_to_physical_flat(coords);
                    MyErrorHandler::log(data_[offset]);
                    MyErrorHandler::log(" ");
                }

                // Padding elements
                if (showPadding && physColDim > colDim)
                {
                    MyErrorHandler::log("| ");
                    // Get base offset for this row (col = 0)
                    coords[ND - 1] = 0;
                    my_size_t rowBase = Layout::logical_coords_to_physical_flat(coords);
                    // Last dim has stride 1, so padding is at rowBase + colDim..physColDim-1
                    for (my_size_t j = colDim; j < physColDim; ++j)
                    {
                        MyErrorHandler::log(data_[rowBase + j]);
                        MyErrorHandler::log(" ");
                    }
                }

                MyErrorHandler::log("\n");
            }
            MyErrorHandler::log("\n");

            // Increment outer coordinates (odometer, right-to-left over dims 0..ND-3)
            if constexpr (ND > 2)
            {
                for (my_size_t d = ND - 3;; --d)
                {
                    coords[d]++;
                    if (coords[d] < getDim(d))
                        break;
                    coords[d] = 0;
                    if (d == 0)
                        break;
                }
            }
        }
    }

    FORCE_INLINE static constexpr my_size_t getDim(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::logical_dim(i);
    }

    FORCE_INLINE static constexpr my_size_t getStride(my_size_t i) TESSERACT_CONDITIONAL_NOEXCEPT
    {
        return Layout::stride(i);
    }

    // print layout info
    void printLayoutInfo() const
    {
        MyErrorHandler::log("Tensor Layout Info:", ErrorLevel::Info);
        MyErrorHandler::log("Number of Dimensions: " + std::to_string(NumDims), ErrorLevel::Info);
        MyErrorHandler::log("Shape: " + getShape(), ErrorLevel::Info);
        MyErrorHandler::log("Strides: ", ErrorLevel::Info);
        for (my_size_t i = 0; i < NumDims; ++i)
            MyErrorHandler::log(std::to_string(getStride(i)) + " ", ErrorLevel::Info);
        MyErrorHandler::log("\n", ErrorLevel::Info);
    }

    void print_access_policy_info() const
    {
        MyErrorHandler::log("Access Policy Info:", ErrorLevel::Info);
        MyErrorHandler::log("Physical Size: " + std::to_string(AccessPolicy::PadPolicy::PhysicalSize), ErrorLevel::Info);
        MyErrorHandler::log("Logical Size: " + std::to_string(AccessPolicy::PadPolicy::LogicalSize), ErrorLevel::Info);
        MyErrorHandler::log("SIMD Width: " + std::to_string(AccessPolicy::PadPolicy::SimdWidth), ErrorLevel::Info);
        MyErrorHandler::log("\n", ErrorLevel::Info);
    }

    void print_flat_data() const
    {
        MyErrorHandler::log("Flat Data:", ErrorLevel::Info);
        for (my_size_t i = 0; i < AccessPolicy::PhysicalSize; ++i)
        {
            MyErrorHandler::log(std::to_string(data_[i]) + " ", ErrorLevel::Info);
        }
        MyErrorHandler::log("\n", ErrorLevel::Info);
    }

private:
    // Example of using different access and storage policies
    using AccessPolicy = DenseAccess<T, SimdPaddingPolicy, StaticStorage, Dims...>;
    // using AccessPolicy = DenseAccess<T, NoPaddingPolicy, StaticStorage, Dims...>; // works only for GENERICARCH

    // using AccessPolicy = DenseAccess<T, SimdPaddingPolicy, DynamicStorage, Dims...>; // works
    // using AccessPolicy = DenseAccess<T, NoPaddingPolicy, DynamicStorage, Dims...>; // bad alloc

    // using AccessPolicy = SparseAccess<T, TotalSize, my_size_t, DynamicStorage, DynamicStorage>; // something is wrong here
    // using AccessPolicy = SparseAccess<T, TotalSize, my_size_t, StaticStorage, StaticStorage>; // something is wrong here
    // using AccessPolicy = SparseAccess<T, TotalSize, my_size_t>; // default is static storage // something is wrong here
    AccessPolicy data_;

    template <my_size_t... Dims1>
    FORCE_INLINE void checkDimensionsMismatch(const FusedTensorND<T, Dims1...> &other) const // TODO: conditionally noexcept
    {
        // check if the dimensions of the tensors are the same taking into account the transpose order
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            if (this->getDim(i) != other.getDim(i))
            {
                MyErrorHandler::error("Dimensions mismatch");
            }
        }
    }

    template <my_size_t NumDims, my_size_t M>
    [[deprecated]] static void print_combinations(const my_size_t (&combinations)[M][NumDims])
    {
        for (my_size_t i = 0; i < M; ++i)
        {
            MyErrorHandler::log("{ ");
            for (my_size_t j = 0; j < NumDims; ++j)
            {
                MyErrorHandler::log(combinations[i][j]);
                MyErrorHandler::log(j < NumDims - 1 ? ", " : " ");
            }
            MyErrorHandler::log("}\n");
        }
    }

    // Template function to generate all combinations and store them in a 2D array
    template <my_size_t NumDims, my_size_t M>
    [[deprecated]] static void generate_combinations(const my_size_t (&max_values)[NumDims], my_size_t (&combinations)[M][NumDims])
    {
        my_size_t combination[NumDims] = {0}; // Initialize the first combination with all 0s

        // Fill each row in `combinations` with the next combination
        for (my_size_t row = 0; row < M; ++row)
        {
            for (my_size_t i = 0; i < NumDims; ++i)
            {
                combinations[row][i] = combination[i];
            }

            // print the combination
            // here you can calculate the contraction of the tensor
            // if you don't want to store all the combinations
            // you can calculate the contraction here
            // for now comment this print statement
            // for (my_size_t i = 0; i < NumDims; ++i)
            // {
            //     std::cout << combination[i] << ", ";
            // }
            // std::cout << std::endl;

            // Increment combination like a counter with custom max values
            int position = NumDims - 1; // TODO: do not use int. Make the loop safe -> to not overflow
            while (position >= 0)
            {
                ++combination[position];
                if (combination[position] < max_values[position])
                {
                    break;
                }
                combination[position] = 0;
                --position;
            }
        }
    }

    // 1D print function
    [[deprecated]] void print1D() const
    {
        for (my_size_t i = 0; i < getDim(0); ++i)
        {
            MyErrorHandler::log((*this)(i));
            MyErrorHandler::log(" ");
        }
        MyErrorHandler::log("\n");
    }

    // 2D print function
    [[deprecated]] void print2D(bool with_padding) const
    {
        const my_size_t rows = getDim(0);
        const my_size_t cols = with_padding
                                   ? AccessPolicy::PadPolicy::PhysicalDims[1]
                                   : getDim(1);

        for (my_size_t i = 0; i < rows; ++i)
        {
            for (my_size_t j = 0; j < cols; ++j)
            {
                if (!with_padding)
                {
                    MyErrorHandler::log((*this)(i, j));
                }
                else
                {
                    // Direct physical access: row * physical_stride + col
                    MyErrorHandler::log(data_[Layout::base_stride(0) * i + j]);
                }

                if (with_padding && j == getDim(1) - 1)
                    MyErrorHandler::log(" |"); // visual separator before padding
                MyErrorHandler::log(" ");
            }
            MyErrorHandler::log("\n");
        }
    }

    // 3D print function
    [[deprecated]] void print3D() const
    {
        for (my_size_t s = 0; s < getDim(0); ++s)
        {
            for (my_size_t i = 0; i < getDim(1); ++i)
            {
                for (my_size_t j = 0; j < getDim(2); ++j)
                {
                    MyErrorHandler::log((*this)(s, i, j));
                    MyErrorHandler::log(" ");
                }
                MyErrorHandler::log("\n");
            }
            MyErrorHandler::log("\n");
        }
    }

    [[deprecated]] void print4D() const
    {
        for (my_size_t b = 0; b < getDim(0); ++b)
        {
            MyErrorHandler::log("Batch [");
            MyErrorHandler::log(b);
            MyErrorHandler::log("]:\n");
            for (my_size_t s = 0; s < getDim(1); ++s)
            {
                MyErrorHandler::log("  Slice [");
                MyErrorHandler::log(s);
                MyErrorHandler::log("]:\n");
                for (my_size_t i = 0; i < getDim(2); ++i)
                {
                    MyErrorHandler::log("    [ ");
                    for (my_size_t j = 0; j < getDim(3); ++j)
                    {
                        MyErrorHandler::log(operator()(b, s, i, j));
                        MyErrorHandler::log(" ");
                    }
                    MyErrorHandler::log("]\n");
                }
            }
            MyErrorHandler::log("\n");
        }
    }

    // template <typename, my_size_t>
    // friend class PermutedView;

    template <typename, my_size_t...>
    friend class PermutedViewConstExpr;

public:
    FORCE_INLINE constexpr const T *data() const noexcept { return data_.data(); }
    FORCE_INLINE constexpr T *data() noexcept { return data_.data(); }

    using Layout = StridedLayoutConstExpr<typename AccessPolicy::PadPolicy>;
};

#endif // FUSEDTENSORND_H
