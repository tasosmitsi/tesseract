#ifndef FUSEDTENSORND_H
#define FUSEDTENSORND_H

// #include <algorithm> // for std::fill_n and std::copy
#include <utility> // for std::move

#include "copy_n_optimized.h"

#include "config.h"
#include "helper_traits.h"
#include "simple_type_traits.h"

#include "fused/BaseExpr.h"
#include "fused/Operators.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/storage/static_storage.h"
#include "fused/storage/dynamic_storage.h"
#include "fused/access/dense_access.h"
#include "fused/access/sparse_access.h"

// Base class: FusedTensorND
template <typename T, my_size_t... Dims>
class FusedTensorND : public BaseExpr<FusedTensorND<T, Dims...>, T>
{
public:
    // Default constructor
    FusedTensorND()
    {
        initTransposeOrder();
    }

    // Constructor to initialize all elements to a specific value
    FusedTensorND(T initValue)
        : data_(initValue) // constructor call here
    {
        initTransposeOrder();
    }

    // Copy constructor
    FusedTensorND(const FusedTensorND &other)
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
        // Copy the transpose order
        copy_n_optimized(other.transposeOrder_, transposeOrder_, getNumDims());

        transposeOrderSet_ = true;
        initTransposeOrder();
    }

    // Move constructor
    FusedTensorND(FusedTensorND &&other) noexcept
        : data_(std::move(other.data_)) // invoke move constructor of AccessPolicy
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
        // Copy the transpose order
        copy_n_optimized(other.transposeOrder_, transposeOrder_, getNumDims());

        transposeOrderSet_ = true;
        initTransposeOrder();
    }

    template <typename Expr>
    FusedTensorND &operator=(const BaseExpr<Expr, T> &expr)
    {
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("FusedTensorND assignment operator called", ErrorLevel::Info);
#endif
        const auto &e = expr.derived();

        for (my_size_t idx = 0; idx < totalSize; ++idx)
        {
            my_size_t indices[sizeof...(Dims)];
            unravelIndex(idx, indices);
            data_[idx] = e(indices);
        }
        return *this;
    }

    template <typename Expr>
    FusedTensorND &eval(const BaseExpr<Expr, T> &expr)
    {
        const auto &e = expr.derived();

        static constexpr my_size_t simdWidth = Microkernel<T, BITS, DefaultArch>::simdWidth; // assuming typename OpTraits<T, BITS, DefaultArch>::type for floats

        if constexpr (!is_same_v<DefaultArch, GenericArch>)
        {
            const my_size_t simdSteps = totalSize / simdWidth;
            for (my_size_t i = 0; i < simdSteps; ++i)
            {
                my_size_t indices[sizeof...(Dims)];
                unravelIndex(i * simdWidth, indices); // interpret index as vector chunk index
                typename Microkernel<T, BITS, DefaultArch>::VecType val = e.evalu(indices);
                Microkernel<T, BITS, DefaultArch>::store(data_.data() + i * simdWidth, val); // write simdWidth floats
            }

            // Handle leftover elements scalar-wise
            for (my_size_t i = simdSteps * simdWidth; i < totalSize; ++i)
            {
                my_size_t indices[sizeof...(Dims)];
                unravelIndex(i, indices);
                data_[i] = e(indices);
            }
        }
        else
        {
            // Fallback to scalar evaluation if no SIMD support
            for (my_size_t i = 0; i < totalSize; ++i)
            {
                my_size_t indices[sizeof...(Dims)];
                unravelIndex(i, indices);
                data_[i] = e(indices);
            }
            return *this;
        }

        return *this;
    }

    template <my_size_t length>
    typename Microkernel<T, BITS, DefaultArch>::VecType evalu(my_size_t (&indices)[length]) const
    {
        my_size_t baseIdx = computeIndex(indices);
        assert((baseIdx % Microkernel<T, BITS, DefaultArch>::simdWidth) == 0 && "baseIdx must be multiple of OpTraits<T, BITS, DefaultArch>::width for aligned load!");
        return Microkernel<T, BITS, DefaultArch>::load(data_.data() + baseIdx); // load 4 floats
    }

    FusedTensorND &operator=(const FusedTensorND &other)
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

        // Copy the transpose order
        copy_n_optimized(other.transposeOrder_, transposeOrder_, getNumDims());

        transposeOrderSet_ = true;

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
        // Copy the transpose order
        copy_n_optimized(other.transposeOrder_, transposeOrder_, getNumDims());

        transposeOrderSet_ = true;

        // Move the data
        data_ = std::move(other.data_); // calls the move assignment of AccessPolicy
        return *this;
    }

    // Variadic access operator for accessing tensor elements with separate indices
    template <typename... Indices>
    inline T &operator()(Indices... indices) noexcept
    {
#ifdef STATIC_CHECK_NUMBER_OF_INDICES
        // static_assert(sizeof...(indices) == sizeof...(Dims), "Incorrect number of indices");
        static constexpr bool correct = sizeof...(Indices) == sizeof...(Dims);
        static_assert(correct, "Number of indices must match tensor dimensions");
#endif

        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...}; // Convert indices to an array
        return data_[computeIndex(idxArray)];
    }

    // Const version of the access operator
    template <typename... Indices>
    inline const T &operator()(Indices... indices) const noexcept
    {
#ifdef STATIC_CHECK_NUMBER_OF_INDICES
        // static_assert(sizeof...(indices) == sizeof...(Dims), "Incorrect number of indices");
        static constexpr bool correct = sizeof...(Indices) == sizeof...(Dims);
        static_assert(correct, "Number of indices must match tensor dimensions");
#endif

        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return data_[computeIndex(idxArray)];
    }

    // version of passing a array of indices eg _tensor1(indices1), indices1 is an array of known size use template
    template <my_size_t length>
    inline T &operator()(my_size_t (&indices)[length]) noexcept
    {
#ifdef STATIC_CHECK_NUMBER_OF_INDICES
        static_assert(length == sizeof...(Dims), "Incorrect number of indicessss");
#endif

        return data_[computeIndex(indices)];
    }

    template <my_size_t length>
    inline const T &operator()(my_size_t (&indices)[length]) const noexcept
    {
#ifdef STATIC_CHECK_NUMBER_OF_INDICES
        static_assert(length == sizeof...(Dims), "Incorrect number of indicessss");
#endif

        return data_[computeIndex(indices)];
    }

    // overload == operator to compare two tensors, introduce a tolerance for floating point numbers
    template <my_size_t... Dims1>
    bool operator==(const FusedTensorND<T, Dims1...> &other) const
    {
        // check for dimensions mismatch, we don't check if they are square
        checkDimensionsMismatch(other);

        my_size_t indices[sizeof...(Dims)] = {0};
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            // increment the indices using for loop
            for (my_size_t j = 0; j < sizeof...(Dims); ++j)
            {
                if (indices[j] < getDim(j) - 1)
                {
                    indices[j]++;
                    break;
                }
                else
                {
                    indices[j] = 0;
                }
            }

            // use the () operator to access the elements
            if (std::abs((*this)(indices)-other(indices)) > T(PRECISION_TOLERANCE))
            {
                return false;
            }
        }
        return true;
    }

    // overload != operator to compare two tensors
    template <my_size_t... Dims1>
    bool operator!=(const FusedTensorND<T, Dims1...> &other) const
    {
        return !(*this == other);
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

    FusedTensorND transposed(const my_size_t order[sizeof...(Dims)]) const
    {
        FusedTensorND outp = *this;
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            outp.transposeOrder_[i] = order[i];
        }
        return outp;
    }

    // Non-inplace transpose function
    void inplace_transpose(const my_size_t order[sizeof...(Dims)])
    {
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            this->transposeOrder_[i] = order[i];
        }
    }

    FusedTensorND transposed(void) const
    {
        // check if the tensor is 2D
        static_assert(sizeof...(Dims) == 2, "Transpose is only supported for 2D tensors");
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("Non Inplace transpose called", ErrorLevel::Info);
#endif
        FusedTensorND outp = *this;
        // reverse the transpose order
        if (outp.transposeOrder_[0] == 0)
        {
            outp.transposeOrder_[0] = 1;
            outp.transposeOrder_[1] = 0;
        }
        else
        {
            outp.transposeOrder_[0] = 0;
            outp.transposeOrder_[1] = 1;
        }
        return outp;
    }

    void inplace_transpose(void)
    {
#ifdef DEBUG_FUSED_TENSOR
        MyErrorHandler::log("Inplace transpose called", ErrorLevel::Info);
#endif
        // reverse the transpose order
        if (this->transposeOrder_[0] == 0)
        {
            this->transposeOrder_[0] = 1;
            this->transposeOrder_[1] = 0;
        }
        else
        {
            this->transposeOrder_[0] = 0;
            this->transposeOrder_[1] = 1;
        }
    }

    // Utility function to retrieve total number of elements
    constexpr my_size_t getTotalSize() const
    {
        return totalSize;
    }

    // Utility function to retrieve the number of dimensions
    constexpr my_size_t getNumDims() const
    {
        return sizeof...(Dims);
    }

    // Utility function to retrieve the shape of the tensor as (1,5,6) for a 3D tensor use the getNumDims
    std::string getShape() const
    // account for the trnaspose order as well
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

    FusedTensorND &setToZero(void)
    {
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            data_[i] = T{};
        }
        return *this;
    }

    FusedTensorND &setHomogen(T _val)
    {
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            data_[i] = _val;
        }
        return *this;
    }

    FusedTensorND &setRandom(my_size_t _maxRand, my_size_t _minRand)
    {
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            // TODO: seed the random number generator
            data_[i] = static_cast<T>((rand() % (_maxRand - _minRand + 1)) + _minRand);
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
        my_size_t indices[numDims] = {0};       // Initialize all indices to zero

        for (my_size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (my_size_t d = 0; d < getNumDims(); ++d)
            {
                indices[d] = i; // Set the diagonal index, others to zero
            }

            // Calculate the index in the flat array and set the value
            data_[computeIndex(indices)] = _val;
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
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            data_[i] = i;
        }
        return *this;
    }

    template <my_size_t DiagonalSize>
    void getDiagonalEntries(FusedTensorND<T, DiagonalSize, 1> &diagonalEntries) const
    {
        static_assert(sizeof...(Dims) >= 2, "Getting diagonal entries requires at least 2 dimensions.");
        // Calculate the minimum dimension
        my_size_t minDim = std::min({Dims...}); // Using initializer list to find the minimum
        my_size_t indices[getNumDims()] = {0};  // Initialize all indices to zero

        for (my_size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (my_size_t d = 0; d < getNumDims(); ++d)
            {
                indices[d] = i; // Set the diagonal index, others to zero
            }

            // Calculate the index in the flat array and set the value
            diagonalEntries(i, 0) = data_[computeIndex(indices)];
        }
    }

    // contract two tensors along a specific dimension (axis) and return the result
    template <my_size_t... Dims1, my_size_t... Dims2>
    static FusedTensorND einsum(const FusedTensorND<T, Dims1...> &_tensor1, const FusedTensorND<T, Dims2...> &_tensor2, my_size_t a, my_size_t b)
    {
        static_assert(sizeof...(Dims1) >= 2, "Tensor 1 must have at least 2 dimension");
        static_assert(sizeof...(Dims2) >= 2, "Tensor 2 must have at least 2 dimension");

        // check if a and b are valid dimensions
        if (a >= sizeof...(Dims1) || b >= sizeof...(Dims2))
        {
            MyErrorHandler::error("Invalid dimensions");
        }

        // check if the a axis of tensor1 is equal to the b axis of tensor2
        if (_tensor1.getDim(a) != _tensor2.getDim(b))
        {
            MyErrorHandler::error("Dimensions mismatch");
        }

        // calculate the new dimensions
        constexpr my_size_t n_newDims = sizeof...(Dims1) + sizeof...(Dims2) - 2;
        my_size_t newDims[n_newDims];
        my_size_t k = 0;
        for (my_size_t i = 0; i < sizeof...(Dims1); ++i)
        {
            if (i != a)
            {
                newDims[k++] = _tensor1.getDim(i);
            }
        }

        for (my_size_t i = 0; i < sizeof...(Dims2); ++i)
        {
            if (i != b)
            {
                newDims[k++] = _tensor2.getDim(i);
            }
        }

        // print the new dimensions
        // std::cout << "New dimensions: ";
        // for (my_size_t i = 0; i < n_newDims; ++i)
        // {
        //     std::cout << newDims[i] << " ";
        // }
        // std::cout << std::endl;

        // create a new tensor with the new dimensions
        FusedTensorND<T, Dims...> _outp;

        //  check if the new dimensions one by one are the same as the dimensions of the new tensor
        for (my_size_t i = 0; i < n_newDims; ++i)
        {
            if (newDims[i] != _outp.getDim(i))
            {
                MyErrorHandler::error("Dimensions mismatch");
            }
        }

        // calculate the total number of combinations and create a 2D array to store them
        constexpr my_size_t total_combinations = (1 * ... * Dims);
        my_size_t combinations[total_combinations][n_newDims];

        // generate all the combinations
        generate_combinations(newDims, combinations);

        // print_combinations(combinations);

        // calculate the contraction
        for (my_size_t comb = 0; comb < total_combinations; ++comb)
        {
            T sum = 0;

            // // print the sum with the output tensor
            // std::cout << std::endl << "---------------" << std::endl << "_outp(";
            // for (my_size_t i = 0; i < n_newDims; ++i)
            // {
            //     std::cout << combinations[comb][i] << (i < n_newDims - 1 ? ", " : "");
            // }
            // std::cout << ") = " << "sum" << ";" << std::endl << std::endl;

            my_size_t K = _tensor1.getDim(a); // or _tensor2.getDim(b) since they are equal
            for (my_size_t k = 0; k < K; ++k)
            {
                my_size_t indices1[sizeof...(Dims1)] = {0};
                my_size_t indices2[sizeof...(Dims2)] = {0};

                my_size_t l = 0;
                for (my_size_t i = 0; i < sizeof...(Dims1); ++i)
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

                l = sizeof...(Dims1) - 1;
                for (my_size_t i = 0; i < sizeof...(Dims2); ++i)
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

                // // print the sumation operation with the indices of the tensors
                // std::cout << "Sum += _tensor1(";
                // for (my_size_t i = 0; i < sizeof...(Dims1); ++i)
                // {
                //     std::cout << indices1[i] << (i < sizeof...(Dims1) - 1 ? ", " : "");
                // }
                // std::cout << ") * _tensor2(";
                // for (my_size_t i = 0; i < sizeof...(Dims2); ++i)
                // {
                //     std::cout << indices2[i] << (i < sizeof...(Dims2) - 1 ? ", " : "");
                // }
                // std::cout << ");" << std::endl;

                sum += _tensor1(indices1) * _tensor2(indices2);
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
    my_size_t getDim(my_size_t i) const
    {
        return dims[transposeOrder_[i]];
    }

private:
    // Calculate total number of elements at compile time
    static constexpr my_size_t totalSize = (Dims * ...);
    static constexpr my_size_t dims[] = {Dims...}; // Fixed array of dimensions

    // These vars are being set in runtime
    my_size_t transposeOrder_[sizeof...(Dims)];
    bool transposeOrderSet_ = false;

    // Example of using different access and storage policies
    // using AccessPolicy = DenseAccess<T, totalSize, StaticStorage>;
    // using AccessPolicy = DenseAccess<T, totalSize, DynamicStorage>;
    // using AccessPolicy = SparseAccess<T, totalSize, my_size_t, DynamicStorage, DynamicStorage>;
    // using AccessPolicy = SparseAccess<T, totalSize, my_size_t, StaticStorage, StaticStorage>;
    // using AccessPolicy = SparseAccess<T, totalSize, my_size_t>; // default is static storage
    using AccessPolicy = DenseAccess<T, totalSize>; // default is static storage
    AccessPolicy data_;

    template <my_size_t... Dims1>
    inline void checkDimensionsMismatch(const FusedTensorND<T, Dims1...> &other) const
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

    // init the transpose order
    void initTransposeOrder()
    {
        // check if the transpose order is preset
        if (transposeOrderSet_)
        {
            return;
        }

        // if not set the transpose order to the default order
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            transposeOrder_[i] = i;
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

    // Unravel a flat index into multi-dimensional indices
    // This function fills the indices array with the corresponding indices for each dimension
    void unravelIndex(my_size_t flatIdx, my_size_t *indices) const
    {
        my_size_t dimCount = sizeof...(Dims);
        for (my_size_t i = dimCount; i > 0; --i)
        {
            indices[i - 1] = flatIdx % dims[i - 1];
            flatIdx /= dims[i - 1];
        }
    }

protected:
    AccessPolicy &rawData() { return data_; } // TODO: can be inline or FORCE_INLINE
    const AccessPolicy &rawData() const { return data_; } // TODO: can be inline or FORCE_INLINE
    static constexpr my_size_t numDims = sizeof...(Dims);

    // Compute the flat index from multi-dimensional indices
    my_size_t computeIndex(const my_size_t indices[numDims]) const
    {
        my_size_t index = 0;
        my_size_t factor = 1;

        for (my_size_t i = getNumDims(); i-- > 0;)
        {
            my_size_t dimIndex = transposeOrder_[i]; // Get dimension according to transpose order

#ifdef RUNTIME_USE_BOUNDS_CHECKING
            if (indices[dimIndex] >= dims[i])
            {
                MyErrorHandler::error("Index out of range");
            }
#endif

            index += indices[dimIndex] * factor; // Use the indices in the transpose order
            factor *= dims[i];                   // Update the factor for the next dimension
        }
        return index; // Return the computed flat index
    }
};

#endif // FUSEDTENSORND_H
