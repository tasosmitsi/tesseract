#ifndef TENSORND_H
#define TENSORND_H

#include <stdexcept>
#include <algorithm> // for std::fill_n and std::copy
#include <utility>   // for std::move
#include <iostream>

#include "fused/config.h"

// Base class: TensorND
template <typename T, my_size_t... Dims>
class TensorND
{
public:
    // Default constructor
    TensorND()
    {
        initTransposeOrder();
    }

    // Constructor to initialize all elements to a specific value
    TensorND(T initValue)
    {
        initTransposeOrder();
        std::fill_n(data_, totalSize, initValue);
    }

    // Copy constructor
    TensorND(const TensorND &other)
    {
        std::copy(other.transposeOrder_, other.transposeOrder_ + getNumDims(), transposeOrder_);
        transposeOrderSet_ = true;
        initTransposeOrder();

        std::copy(other.data_, other.data_ + totalSize, data_);
    }

    // Move constructor
    TensorND(TensorND &&other) noexcept
    {
        std::copy(other.transposeOrder_, other.transposeOrder_ + getNumDims(), transposeOrder_);
        transposeOrderSet_ = true;
        initTransposeOrder();

        std::move(other.data_, other.data_ + totalSize, data_);
    }

    // Variadic access operator for accessing tensor elements with separate indices
    template <typename... Indices>
    T &operator()(Indices... indices)
    {
        #ifdef STATIC_CHECK_NUMBER_OF_INDICES
            static_assert(sizeof...(indices) == sizeof...(Dims), "Incorrect number of indices");
        #endif

        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...}; // Convert indices to an array
        return data_[computeIndex(idxArray)];
    }

    // Const version of the access operator
    template <typename... Indices>
    const T &operator()(Indices... indices) const
    {
        #ifdef STATIC_CHECK_NUMBER_OF_INDICES
            static_assert(sizeof...(indices) == sizeof...(Dims), "Incorrect number of indices");
        #endif

        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return data_[computeIndex(idxArray)];
    }

    // version of passing a array of indices eg _tensor1(indices1), indices1 is an array of known size use template
    template<my_size_t length>
    T& operator()(my_size_t (&indices)[length])
    {
        #ifdef STATIC_CHECK_NUMBER_OF_INDICES
            static_assert(length == sizeof...(Dims), "Incorrect number of indicessss");
        #endif

        return data_[computeIndex(indices)];
    }

    template<my_size_t length>
    const T& operator()(my_size_t (&indices)[length]) const
    {
        #ifdef STATIC_CHECK_NUMBER_OF_INDICES
            static_assert(length == sizeof...(Dims), "Incorrect number of indicessss");
        #endif

        return data_[computeIndex(indices)];
    }

    // overload == operator to compare two tensors, introduce a tolerance for floating point numbers
    template <my_size_t... Dims1>
    bool operator==(const TensorND<T, Dims1...> &other) const
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
            if (std::abs((*this)(indices) - other(indices)) > T(PRECISION_TOLERANCE))
            {
                return false;
            }
        }
        return true;
    }

    // overload != operator to compare two tensors
    template <my_size_t... Dims1>
    bool operator!=(const TensorND<T, Dims1...> &other) const
    {
        return !(*this == other);
    }

    // overload = operator to assign a tensor to the tensor
    TensorND &operator=(const TensorND &other)
    {
        // std::cout << "Operator = called" << std::endl;
        if (this == &other)
        {
            return *this;
        }

        // copy the transpose order
        std::copy(other.transposeOrder_, other.transposeOrder_ + getNumDims(), transposeOrder_);
        transposeOrderSet_ = true;

        // copy the data
        std::copy(other.data_, other.data_ + totalSize, data_);
        return *this;
    }

    // overload + operator to add a scalar to the tensor
    TensorND operator+(const T scalar) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            outp.data_[i] += scalar;
        }
        return outp;
    }

    // overload + operator to add the tensor to a scalar
    friend TensorND operator+(const T scalar, const TensorND& tensor)
    {
        return tensor + scalar;
    }

    // overload + operator to add a tensor to the tensor elementwise
    template <my_size_t... Dims1>
    TensorND operator+(const TensorND<T, Dims1...> &other) const
    {
        // check for dimensions mismatch
        checkDimensionsMismatch(other);

        TensorND outp = *this;
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
            outp(indices) += other(indices);
        }
        return outp;
    }

    // overload - operator to subtract a scalar from the tensor
    TensorND operator-(const T scalar) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            outp.data_[i] -= scalar;
        }
        return outp;
    }

    // overload - operator to subtract a scalar from the tensor
    friend TensorND operator-(const T scalar, const TensorND& tensor)
    {
        // return tensor - scalar;
        TensorND outp = tensor;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            outp.data_[i] = scalar - outp.data_[i];
        }
        return outp;
    }

    // overload - operator to get the negative of the tensor
    TensorND operator-(void) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            outp.data_[i] = -outp.data_[i];
        }
        return outp;
    }

    // overload - operator to subtract a tensor from the tensor elementwise
    template <my_size_t... Dims1>
    TensorND operator-(const TensorND<T, Dims1...> &other) const
    {
        // check for dimensions mismatch
        checkDimensionsMismatch(other);

        TensorND outp = *this;
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
            outp(indices) -= other(indices);
            }
        return outp;
    }

    // overload * operator to multiply a scalar with the tensor
    TensorND operator*(const T scalar) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            outp.data_[i] *= scalar;
        }
        return outp;
    }

    // overload * operator to multiply a tensor with a scalar
    friend TensorND operator*(const T scalar, const TensorND& tensor)
    {
        return tensor * scalar;
    }

    // overload an operator to multiply a tensor with a tensor elementwise
    template <my_size_t... Dims1>
    TensorND operator*(const TensorND<T, Dims1...> &other) const
    {
        // check for dimensions mismatch
        checkDimensionsMismatch(other);

        TensorND outp = *this;
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
            outp(indices) *= other(indices);
            }
        return outp;
    }

    // overload / operator to divide the tensor by a scalar, check for division by zero, account floats as well
    TensorND operator/(const T scalar) const
    {
        if (scalar == 0)
        {
            throw std::runtime_error("Division by zero");
        }

        return *this * (1 / scalar);
    }

    // overload / operator to divide a scalar by the tensor
    friend TensorND operator/(const T scalar, const TensorND& tensor)
    {
        TensorND outp = tensor;
        for (my_size_t i = 0; i < tensor.totalSize; ++i)
        {
            if (tensor.data_[i] == 0)
            {
                throw std::runtime_error("Division by zero");
            }
            outp.data_[i] = scalar / tensor.data_[i];
        }
        return outp;
    }

    // overload / operator to divide the tensor by a tensor elementwise, check for division by zero
    template <my_size_t... Dims1>
    TensorND operator/(const TensorND<T, Dims1...> &other) const
    {
        // check for dimensions mismatch
        checkDimensionsMismatch(other);

        TensorND outp = *this;
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
            if (other(indices) == 0)
            {
                throw std::runtime_error("Division by zero");
            }
            outp(indices) /= other(indices);
        }
        return outp;
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
        constexpr my_size_t total_combinations = (1 * ... * Dims); // fold expression to calculate the total number of combinations
        my_size_t combinations[total_combinations][sizeof...(Dims)]; // 2D array to store all combinations
        my_size_t max_vals[sizeof...(Dims)] = {Dims...}; // array to store the maximum values for each dimension
        generate_combinations(max_vals, combinations); // generate all combinations

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

    // non-inplace transpose function
    TensorND transposed(const my_size_t order[sizeof...(Dims)]) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            outp.transposeOrder_[i] = order[i];
        }
        return outp;
    }

    // inplace transpose function
    void inplace_transpose(const my_size_t order[sizeof...(Dims)])
    {
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            this->transposeOrder_[i] = order[i];
        }
    }

    // non-inplace transpose function
    TensorND transposed(void) const
    {
        // check if the tensor is 2D
        static_assert(sizeof...(Dims) == 2, "Transpose is only supported for 2D tensors");

        TensorND outp = *this;
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

    // inplace transpose function
    void inplace_transpose(void)
    {
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

    TensorND& setToZero(void)
    {
        std::fill_n(data_, totalSize, 0);
        return *this;
    }

    TensorND& setHomogen(T _val)
    {
        std::fill_n(data_, totalSize, _val);
        return *this;
    }

    TensorND& setRandom(my_size_t _maxRand, my_size_t _minRand)
    {
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            // TODO: seed the random number generator
            data_[i] = static_cast<T>((rand() % (_maxRand - _minRand + 1)) + _minRand);
        }
        return *this;
    }

    // for all dimensions
    TensorND& setDiagonal(T _val)
    {
        static_assert(sizeof...(Dims) >= 2, "setDiagonal requires at least 2 dimensions.");

        // set the entire matrix to zeros
        setToZero();

        // Calculate the minimum dimension
        my_size_t minDim = std::min({Dims...}); // Using initializer list to find the minimum
        my_size_t indices[getNumDims()] = {0}; // Initialize all indices to zero

        for (my_size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (my_size_t d = 0; d < getNumDims(); ++d) {
                indices[d] = i; // Set the diagonal index, others to zero
            }

            // Calculate the index in the flat array and set the value
            data_[computeIndex(indices)] = _val;
        }
        return *this;
    }

    TensorND& setIdentity(void)
    {
        static_assert(sizeof...(Dims) >= 2, "Identity requires at least 2 dimensions.");
        static_assert(((Dims == dims[0]) && ...), "All dimensions must be equal for an identity tensor");
        this->setDiagonal(1);
        return *this;
    }

    static TensorND I(void)
    {
        static_assert(sizeof...(Dims) >= 2, "Identity requires at least 2 dimensions.");

        static_assert(((Dims == dims[0]) && ...), "All dimensions must be the same for an identity tensor");

        TensorND<T, Dims...> _outp;
        _outp.setDiagonal(1);
        return _outp;
    }

    TensorND& setSequencial(void)
    {
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            data_[i] = i;
        }
        return *this;
    }

    template<my_size_t DiagonalSize>
    void getDiagonalEntries(TensorND<T, DiagonalSize, 1>& diagonalEntries) const
    {
        static_assert(sizeof...(Dims) >= 2, "Getting diagonal entries requires at least 2 dimensions.");
        // Calculate the minimum dimension
        my_size_t minDim = std::min({Dims...}); // Using initializer list to find the minimum
        my_size_t indices[getNumDims()] = {0}; // Initialize all indices to zero

        for (my_size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (my_size_t d = 0; d < getNumDims(); ++d) {
                indices[d] = i; // Set the diagonal index, others to zero
            }

            // Calculate the index in the flat array and set the value
            diagonalEntries(i, 0) = data_[computeIndex(indices)];
        }
    }

    // contract two tensors along a specific dimension (axis) and return the result
    template <my_size_t... Dims1, my_size_t... Dims2>
    static TensorND einsum(const TensorND<T, Dims1...>& _tensor1, const TensorND<T, Dims2...>& _tensor2, my_size_t a, my_size_t b)
    {
        static_assert(sizeof...(Dims1) >= 2 , "Tensor 1 must have at least 2 dimension");
        static_assert(sizeof...(Dims2) >= 2 , "Tensor 2 must have at least 2 dimension");

        // check if a and b are valid dimensions
        if (a >= sizeof...(Dims1) || b >= sizeof...(Dims2))
        {
            throw std::runtime_error("Invalid dimensions");
        }

        // check if the a axis of tensor1 is equal to the b axis of tensor2
        if (_tensor1.getDim(a) != _tensor2.getDim(b))
        {
            throw std::runtime_error("Dimensions mismatch");
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
        TensorND<T, Dims...> _outp;

        //  check if the new dimensions one by one are the same as the dimensions of the new tensor
        for (my_size_t i = 0; i < n_newDims; ++i)
        {
            if (newDims[i] != _outp.getDim(i))
            {
                throw std::runtime_error("Dimensions mismatch");
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

    /* Insert submatrix into matrix at _posRow & _posCol position
    * Example: A = Matrix 4x4, B = Matrix 2x3
    *
    *  C = A.InsertSubMatrix(B, 1, 1);
    *
    *  A = [A00  A01  A02  A03]    B = [B00  B01  B02]
    *      [A10  A11  A12  A13]        [B10  B11  B12]
    *      [A20  A21  A22  A23]
    *      [A30  A31  A32  A33]
    *
    *
    *  C = [A00  A01  A02  A03]
    *      [A10  B00  B01  B02]
    *      [A20  B10  B11  B12]
    *      [A30  A31  A32  A33]
    */
    // template<my_size_t... DimsB>
    // template<typename... insertion_coordinates>
    // TensorND& InsertSubMatrix(const TensorND<T, DimsB...>& _subMatrix, insertion_coordinates... _insertion_coordinates)
    // {
    //     // check if the tensor is 2D
    //     static_assert(sizeof...(Dims) > 3, "InsertSubMatrix is only supported for MAX 3D tensors");
    //     static_assert(sizeof...(DimsB) > 3, "InsertSubMatrix is only supported for MAX 3D tensors");

    //     // check if the submatrix fits into the matrix
    //     if ((_subMatrix.dims[0] + _insertion_coordinates... > dims[0]) || (_subMatrix.dims[1] + _insertion_coordinates... > dims[1]))
    //     {
    //         throw std::runtime_error("Submatrix does not fit into the matrix");
    //     }

    //     for (my_size_t i = 0; i < _subMatrix.dims[0]; ++i)
    //     {
    //         for (my_size_t j = 0; j < _subMatrix.dims[1]; ++j)
    //         {
    //             (*this)(_insertion_coordinates... + i, _insertion_coordinates... + j) = _subMatrix(i, j);
    //         }
    //     }
    //     return *this;
    // }
    // {
    //     // check if the tensor is 2D
    //     static_assert(sizeof...(Dims) == 2, "InsertSubMatrix is only supported for 2D tensors");
    //     static_assert(sizeof...(DimsB) == 2, "InsertSubMatrix is only supported for 2D tensors");

    //     // check if the submatrix fits into the matrix
    //     if ((_subMatrix.dims[0] + _posRow > dims[0]) || (_subMatrix.dims[1] + _posCol > dims[1]))
    //     {
    //         throw std::runtime_error("Submatrix does not fit into the matrix");
    //     }

    //     for (my_size_t i = 0; i < _subMatrix.dims[0]; ++i)
    //     {
    //         for (my_size_t j = 0; j < _subMatrix.dims[1]; ++j)
    //         {
    //             (*this)(_posRow + i, _posCol + j) = _subMatrix(i, j);
    //         }
    //     }
    //     return *this;
    // }

    // Function to print the contents of the tensor
    void print() const {
        static_assert(sizeof...(Dims) <= 4, "Printing not supported for tensors with more than 4 dimensions");

        if constexpr (sizeof...(Dims) == 1) {
            print1D();
        }
        else if constexpr (sizeof...(Dims) == 2) {
            print2D();
        }
        else if constexpr (sizeof...(Dims) == 3) {
            print3D();
        }
        else if constexpr (sizeof...(Dims) == 4) {
            print4D();
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
    T data_[totalSize]; // Contiguous storage of elements in a flat array

    template <my_size_t... Dims1>
    inline void checkDimensionsMismatch(const TensorND<T, Dims1...> &other) const
    {
        // check if the dimensions of the tensors are the same taking into account the transpose order
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            if (this->getDim(i) != other.getDim(i))
            {
                throw std::runtime_error("Dimensions mismatch");
            }
        }
    }

    template <my_size_t N, my_size_t M>
    static void print_combinations(const my_size_t (&combinations)[M][N])
    {
        for (my_size_t i = 0; i < M; ++i)
        {
            std::cout << "{ ";
            for (my_size_t j = 0; j < N; ++j)
            {
                std::cout << combinations[i][j] << (j < N - 1 ? ", " : " ");
            }
            std::cout << "}\n";
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
            my_size_t position = N - 1;
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
    void print1D() const {
        for (my_size_t i = 0; i < getDim(0); ++i)
        {
            std::cout << (*this)(i) << " ";
        }
        std::cout << std::endl;
    }

    // 2D print function
    void print2D() const {
        // account for the trnaspose order as well
        for (my_size_t i = 0; i < getDim(0); ++i)
        {
            for (my_size_t j = 0; j < getDim(1); ++j)
            {
                // std::cout << "(" << i << "," << j << ") ";
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // 3D print function
    void print3D() const {
        for (my_size_t k = 0; k < getDim(2); ++k) {
            // std::cout << "Slice " << i << ":\n";
            for (my_size_t i = 0; i < getDim(0); ++i) {
                for (my_size_t j = 0; j < getDim(1); ++j) {
                    std::cout << (*this)(i, j, k) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void print4D() const {
        for (my_size_t l = 0; l < getDim(3); ++l) {
            std::cout << "Slice [" << l << "]:\n";
            for (my_size_t k = 0; k < getDim(2); ++k) {
                std::cout << "  Sub-Slice [" << k << "]:\n";
                for (my_size_t i = 0; i < getDim(0); ++i) {
                    std::cout << "    [ ";
                    for (my_size_t j = 0; j < getDim(1); ++j) {
                        std::cout << operator()(i, j, k, l) << " ";
                    }
                    std::cout << "]" << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // Compute the flat index from multi-dimensional indices
    my_size_t computeIndex(const my_size_t indices[getNumDims()]) const {
        my_size_t index = 0;
        my_size_t factor = 1;

        // for (my_size_t i = getNumDims() - 1; i >= 0; --i) {
        for (my_size_t i = getNumDims(); i-- > 0; ) {
            my_size_t dimIndex = transposeOrder_[i]; // Get dimension according to transpose order

            #ifdef RUNTIME_USE_BOUNDS_CHECKING
                if (indices[dimIndex] >= dims[i]) {
                    throw std::out_of_range("Index out of range");
                }
            #endif

            index += indices[dimIndex] * factor; // Use the indices in the transpose order
            factor *= dims[i]; // Update the factor for the next dimension
        }
        return index; // Return the computed flat index
    }
};

#endif // TENSORND_H
