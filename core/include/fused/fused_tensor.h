#ifndef FUSEDTENSORND_H
#define FUSEDTENSORND_H

#include <random>

#include "copy_n_optimized.h"

#include "config.h"
#include "helper_traits.h"
#include "simple_type_traits.h"

#include "fused/BaseExpr.h"
#include "fused/Operators.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/microkernels/kernel_ops.h"
#include "fused/storage/static_storage.h"
#include "fused/storage/dynamic_storage.h"
#include "fused/access/dense_access.h"
#include "fused/access/sparse_access.h"
#include "fused/views/permuted_view.h"
#include "fused/views/permuted_view_constexpr.h"
#include "fused/layouts/strided_layout.h"
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
    // ----------------------
    using Self = FusedTensorND<T, Dims...>;
    static constexpr my_size_t N = sizeof...(Dims); // TODO: use NumDims instead

    // Default constructors
    FusedTensorND() noexcept // TODO make explicit if needed?, use Dim
        : layout_(dims) {}

    // Constructor to initialize all elements to a specific value
    FusedTensorND(T initValue) noexcept // TODO make explicit, use Dim
        : data_(initValue), layout_(dims) {}

    // Copy constructor
    FusedTensorND(const FusedTensorND &other) noexcept
        : data_(other.data_), layout_(other.layout_) // invoke copy constructor of AccessPolicy
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
        : data_(move(other.data_)), layout_(move(other.layout_)) // invoke move constructor of AccessPolicy
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

        // Evaluate using vectorized contiguous if architecture supports it
        if constexpr (!is_same_v<DefaultArch, GENERICARCH>)
        {
            TensorKernels<T, BITS, DefaultArch, Dims...>::eval_vectorized_contiguous(
                data_.data(),
                e,
                [this](my_size_t i, my_size_t(&indices)[sizeof...(Dims)]) constexpr noexcept
                {
                    this->layout_.compute_indices_from_flat(i, indices);
                });
        }
        else
        {
            // Fallback to scalar evaluation if no SIMD support is available
            TensorKernels<T, BITS, DefaultArch, Dims...>::eval_scalar(
                data_.data(),
                e,
                [this](my_size_t i, my_size_t(&indices)[sizeof...(Dims)]) constexpr noexcept
                {
                    this->layout_.compute_indices_from_flat(i, indices);
                });
        }
        return *this;
    }

    template <typename T_, my_size_t Bits, typename Arch>
    typename Microkernel<T_, Bits, Arch>::VecType evalu(my_size_t flat) const noexcept
    {
        using K = Microkernel<T_, Bits, Arch>;
        // TODO: add assert to check alignment if needed
        // assert((flat % K::simdWidth) == 0 && "baseIdx must be multiple of K::simdWidth for aligned load!");
        return K::load(data_.data() + flat);
    }

    // This algorithm means to gather which means that it reads from non-continuous memmory
    // typename Microkernel<T, BITS, DefaultArch>::VecType evaluGather(my_size_t flat) const
    // {
    //     std::cout << "evaluGather" << std::endl;
    //     using K = Microkernel<T, BITS, DefaultArch>;
    //     my_size_t idxList[K::simdWidth];
    //     for (int i = 0; i < K::simdWidth; ++i)
    //         idxList[i] = remapFlatIndex(flat + i, transposeOrder_);
    //     return K::gather(data_.data(), idxList);
    // }

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

        // Copy the layout
        layout_ = other.layout_; // calls the copy assignment of StridedLayout

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

        // Copy the layout
        layout_ = move(other.layout_); // calls the move assignment of StridedLayout

        // Move the data
        data_ = move(other.data_); // calls the move assignment of AccessPolicy
        return *this;
    }

    // Variadic access operator for accessing tensor elements with separate indices
    template <typename... Indices>
        requires(sizeof...(Indices) == N)
    inline T &operator()(Indices... indices) noexcept
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...}; // Convert indices to an array
        return data_[layout_.compute_flat_index(idxArray)];
    }

    // Const version of the variadic access operator
    template <typename... Indices>
        requires(sizeof...(Indices) == N)
    inline const T &operator()(Indices... indices) const noexcept
    {
        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return data_[layout_.compute_flat_index(idxArray)];
    }

    // version of passing a pointer to indices array eg _tensor1(indices1), indices1 is a pointer to an array of known size
    inline T &operator()(const my_size_t *indices) noexcept
    {
        return data_[layout_.compute_flat_index(indices)];
    }

    inline const T &operator()(const my_size_t *indices) const noexcept
    {
        return data_[layout_.compute_flat_index(indices)];
    }

    // version of passing a array of indices eg _tensor1(indices1), indices1 is an array of known size use template
    inline T &operator()(my_size_t (&indices)[N]) noexcept
    {
        return data_[layout_.compute_flat_index(indices)];
    }

    inline const T &operator()(my_size_t (&indices)[N]) const noexcept
    {
        return data_[layout_.compute_flat_index(indices)];
    }

    // check if all dimensions are the same at compile time
    constexpr bool areDimsEqual() const
    {
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            if (dims[i] != dims[0])
            {
                return false;
            }
        }
        return true;
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

    FORCE_INLINE auto transpose_view(const my_size_t perm[NumDims]) const noexcept
    {
        return PermutedView<Self, NumDims>(*this, perm);
    }

    // Utility function to retrieve total number of elements
    FORCE_INLINE constexpr my_size_t getTotalSize() const noexcept
    {
        return TotalSize;
    }

    // Utility function to retrieve the number of dimensions
    FORCE_INLINE constexpr my_size_t getNumDims() const noexcept
    {
        return sizeof...(Dims);
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
        for (my_size_t i = 0; i < TotalSize; ++i)
        {
            data_[i] = T{};
        }
        return *this;
    }

    FusedTensorND &setHomogen(T _val) noexcept
    {
        for (my_size_t i = 0; i < TotalSize; ++i)
        {
            data_[i] = _val;
        }
        return *this;
    }

    FusedTensorND &setRandom(my_size_t _maxRand, my_size_t _minRand)
    {

        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // Mersenne Twister RNG
        std::uniform_real_distribution<T> dist(_minRand, _maxRand);

        for (my_size_t i = 0; i < TotalSize; ++i)
        {
            // TODO: seed the random number generator
            // std::srand(static_cast<unsigned int>(std::time(nullptr)));
            // data_[i] = static_cast<T>((rand()));
            data_[i] = static_cast<T>(dist(rng));
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
        my_size_t minDim = std::min({Dims...}); // Using initializer list to find the minimum
        my_size_t indices[NumDims] = {0};       // Initialize all indices to zero

        for (my_size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (my_size_t d = 0; d < getNumDims(); ++d)
            {
                indices[d] = i; // Set the diagonal index, others to zero
            }

            // Calculate the index in the flat array and set the value
            data_[layout_.compute_flat_index(indices)] = _val;
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

    static FusedTensorND I(void)
    {
        static_assert(sizeof...(Dims) >= 2, "Identity requires at least 2 dimensions.");

        static_assert(all_equal<Dims...>(), "All dimensions must be equal for an identity tensor");

        FusedTensorND<T, Dims...> _outp;
        _outp.setDiagonal(1);
        return _outp;
    }

    FusedTensorND &setSequencial(void)
    {
        for (my_size_t i = 0; i < TotalSize; ++i)
        {
            data_[i] = i;
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
            diagonalEntries(i, 0) = data_[layout_.compute_flat_index(indices)];
        }
    }

    // contract two expression along a specific dimension (axis) and return the result
    template <typename LeftExpr, typename RightExpr>
        requires(
            algebra::is_tensor_v<LeftExpr> &&
            algebra::is_tensor_v<RightExpr>)
    static FusedTensorND einsum(const BaseExpr<LeftExpr> &_tensor1, const BaseExpr<RightExpr> &_tensor2, const my_size_t a, const my_size_t b)
    {
        static const my_size_t Dims1 = LeftExpr::NumDims;
        static const my_size_t Dims2 = RightExpr::NumDims;

        // static_assert(Dims1 >= 2, "Tensor 1 must have at least 2 dimension");
        // static_assert(Dims2 >= 2, "Tensor 2 must have at least 2 dimension");

        if constexpr (Dims1 < 2)
        {
            MyErrorHandler::error("Tensor 1 must have at least 2 dimension");
        }
        if constexpr (Dims2 < 2)
        {
            MyErrorHandler::error("Tensor 2 must have at least 2 dimension");
        }

        // check if a and b are valid dimensions at runtime
        if (a >= Dims1 || b >= Dims2)
        {
            MyErrorHandler::error("Invalid dimensions");
        }

        // check if the a axis of tensor1 is equal to the b axis of tensor2 at runtime
        if (_tensor1.derived().getDim(a) != _tensor2.derived().getDim(b))
        {
            MyErrorHandler::error("Dimensions mismatch between tensors for einsum operation");
        }

        // ------------------------------------------------------
        // TODO: all this inside the ----- can be done at compile time only
        // calculate the new dimensions
        constexpr my_size_t n_newDims = Dims1 + Dims2 - 2;
        my_size_t newDims[n_newDims];
        my_size_t k = 0;
        for (my_size_t i = 0; i < Dims1; ++i)
        {
            if (i != a)
            {
                newDims[k++] = _tensor1.derived().getDim(i);
            }
        }

        for (my_size_t i = 0; i < Dims2; ++i)
        {
            if (i != b)
            {
                newDims[k++] = _tensor2.derived().getDim(i);
            }
        }

        // create a new tensor with the new dimensions
        FusedTensorND<T, Dims...> _outp;

        //  check if the new dimensions one by one are the same as the dimensions of the new tensor
        for (my_size_t i = 0; i < n_newDims; ++i)
        {
            if (newDims[i] != _outp.getDim(i))
            {
                MyErrorHandler::error("Dimensions mismatch in output tensor");
            }
        }
        // ------------------------------------------------------

        // calculate the total number of combinations and create a 2D array to store them
        constexpr my_size_t total_combinations = (1 * ... * Dims);
        my_size_t combinations[total_combinations][n_newDims];

        // generate all the combinations
        generate_combinations(newDims, combinations);

        // calculate the contraction
        for (my_size_t comb = 0; comb < total_combinations; ++comb)
        {
            T sum = 0;
            my_size_t K = _tensor1.derived().getDim(a); // or _tensor2.derived().getDim(b) since they are equal
            for (my_size_t k = 0; k < K; ++k)
            {
                my_size_t indices1[Dims1] = {0};
                my_size_t indices2[Dims2] = {0};

                my_size_t l = 0;
                for (my_size_t i = 0; i < Dims1; ++i)
                {
                    if (i != a)
                    {
                        indices1[i] = combinations[comb][l++];
                    }
                    else
                    {
                        indices1[i] = k;
                    }
                }

                l = Dims1 - 1;
                for (my_size_t i = 0; i < Dims2; ++i)
                {
                    if (i != b)
                    {
                        indices2[i] = combinations[comb][l++];
                    }
                    else
                    {
                        indices2[i] = k;
                    }
                }
                sum += _tensor1.derived()(indices1) * _tensor2.derived()(indices2);
            }
            _outp(combinations[comb]) = sum;
        }
        return _outp;
    }

    // Function to print the contents of the tensor
    void print() const
    {
        // data_.print();
        static_assert(sizeof...(Dims) <= 4, "Printing not supported for tensors with more than 4 dimensions");

        if constexpr (sizeof...(Dims) == 1)
        {
            print1D();
        }
        else if constexpr (sizeof...(Dims) == 2)
        {
            print2D();
        }
        else if constexpr (sizeof...(Dims) == 3)
        {
            print3D();
        }
        else if constexpr (sizeof...(Dims) == 4)
        {
            print4D();
        }
        else
        {
            MyErrorHandler::error("Printing not supported for tensors with more than 4 dimensions");
        }
    }

    // getter for dims
    my_size_t getDim(my_size_t i) const // TODO: conditionally noexcept
    {
        return layout_.getDim(i);
    }

    // getter for strides
    my_size_t getStride(my_size_t i) const // TODO: conditionally noexcept
    {
        return layout_.getStride(i);
    }

private:
    // Calculate total number of elements at compile time
    static constexpr my_size_t dims[] = {Dims...}; // Fixed array of original dimensions TODO: can be replace by Dim[] compile time constant

    // Example of using different access and storage policies
    // using AccessPolicy = DenseAccess<T, TotalSize, StaticStorage>; // tested
    // using AccessPolicy = DenseAccess<T, TotalSize, DynamicStorage>; // tested
    // using AccessPolicy = SparseAccess<T, TotalSize, my_size_t, DynamicStorage, DynamicStorage>; // something is wrong here
    // using AccessPolicy = SparseAccess<T, TotalSize, my_size_t, StaticStorage, StaticStorage>; // something is wrong here
    // using AccessPolicy = SparseAccess<T, TotalSize, my_size_t>; // default is static storage // something is wrong here
    using AccessPolicy = DenseAccess<T, TotalSize>; // default is static storage
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

    template <my_size_t N, my_size_t M>
    static void print_combinations(const my_size_t (&combinations)[M][N])
    {
        for (my_size_t i = 0; i < M; ++i)
        {
            MyErrorHandler::log("{ ");
            for (my_size_t j = 0; j < N; ++j)
            {
                MyErrorHandler::log(combinations[i][j]);
                MyErrorHandler::log(j < N - 1 ? ", " : " ");
            }
            MyErrorHandler::log("}\n");
        }
    }

    // Template function to generate all combinations and store them in a 2D array
    template <my_size_t N, my_size_t M>
    static void generate_combinations(const my_size_t (&max_values)[N], my_size_t (&combinations)[M][N])
    {
        my_size_t combination[N] = {0}; // Initialize the first combination with all 0s

        // Fill each row in `combinations` with the next combination
        for (my_size_t row = 0; row < M; ++row)
        {
            for (my_size_t i = 0; i < N; ++i)
            {
                combinations[row][i] = combination[i];
            }

            // print the combination
            // here you can calculate the contraction of the tensor
            // if you don't want to store all the combinations
            // you can calculate the contraction here
            // for now comment this print statement
            // for (my_size_t i = 0; i < N; ++i)
            // {
            //     std::cout << combination[i] << ", ";
            // }
            // std::cout << std::endl;

            // Increment combination like a counter with custom max values
            int position = N - 1; // TODO: do not use int. Make the loop safe -> to not overflow
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
    void print1D() const
    {
        for (my_size_t i = 0; i < getDim(0); ++i)
        {
            MyErrorHandler::log((*this)(i));
            MyErrorHandler::log(" ");
        }
        MyErrorHandler::log("\n");
    }

    // 2D print function
    void print2D() const
    {
        // account for the trnaspose order as well
        for (my_size_t i = 0; i < getDim(0); ++i)
        {
            for (my_size_t j = 0; j < getDim(1); ++j)
            {
                MyErrorHandler::log((*this)(i, j));
                MyErrorHandler::log(" ");
            }
            MyErrorHandler::log("\n");
        }
    }

    // 3D print function
    void print3D() const
    {
        for (my_size_t k = 0; k < getDim(2); ++k)
        {
            for (my_size_t i = 0; i < getDim(0); ++i)
            {
                for (my_size_t j = 0; j < getDim(1); ++j)
                {
                    MyErrorHandler::log((*this)(i, j, k));
                    MyErrorHandler::log(" ");
                }
                MyErrorHandler::log("\n");
            }
            MyErrorHandler::log("\n");
        }
    }

    void print4D() const
    {
        for (my_size_t l = 0; l < getDim(3); ++l)
        {
            MyErrorHandler::log("Slice [");
            MyErrorHandler::log(l);
            MyErrorHandler::log("]:\n");
            for (my_size_t k = 0; k < getDim(2); ++k)
            {
                MyErrorHandler::log("  Sub-Slice [");
                MyErrorHandler::log(k);
                MyErrorHandler::log("]:\n");
                for (my_size_t i = 0; i < getDim(0); ++i)
                {
                    MyErrorHandler::log("    [ ");
                    for (my_size_t j = 0; j < getDim(1); ++j)
                    {
                        MyErrorHandler::log(operator()(i, j, k, l));
                        MyErrorHandler::log(" ");
                    }
                    MyErrorHandler::log("]");
                }
                MyErrorHandler::log("\n");
            }
            MyErrorHandler::log("\n");
        }
    }

protected:
    using Layout = StridedLayout<N>;
    Layout layout_;

    template <typename, my_size_t>
    friend class PermutedView;

    template <typename, my_size_t...>
    friend class PermutedViewConstExpr;
};

#endif // FUSEDTENSORND_H
