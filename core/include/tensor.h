#ifndef TENSORND_H
#define TENSORND_H

#include <stdexcept>
#include <algorithm> // for std::fill_n and std::copy
#include <utility>   // for std::move

/* Define this to enable matrix number of indices checking */
#define MATRIX_USE_NUMBER_OF_INDICES_CHECKING

/* Define this to enable matrix bound checking */
#define MATRIX_USE_BOUNDS_CHECKING

#define PRECISION_TOLERANCE 1e-9

#define my_size_t size_t // can be uint32_t or uint64_t

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
        initTransposeOrder();
        std::copy(other.data_, other.data_ + totalSize, data_);
    }

    // Move constructor
    TensorND(TensorND &&other) noexcept
    {
        initTransposeOrder();
        std::move(other.data_, other.data_ + totalSize, data_);
    }

    // Variadic access operator for accessing tensor elements with separate indices
    template <typename... Indices>
    T &operator()(Indices... indices)
    {
        #ifdef MATRIX_USE_NUMBER_OF_INDICES_CHECKING
            static_assert(sizeof...(indices) == sizeof...(Dims), "Incorrect number of indices");
        #endif

        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...}; // Convert indices to an array
        return data_[computeIndex(idxArray)];
    }

    // Const version of the access operator
    template <typename... Indices>
    const T &operator()(Indices... indices) const
    {
        #ifdef MATRIX_USE_NUMBER_OF_INDICES_CHECKING
            static_assert(sizeof...(indices) == sizeof...(Dims), "Incorrect number of indices");
        #endif

        my_size_t idxArray[] = {static_cast<my_size_t>(indices)...};
        return data_[computeIndex(idxArray)];
    }

    // overload == operator to compare two tensors, introduce a tolerance for floating point numbers
    bool operator==(const TensorND &other) const
    {
        const double tolerance = 1e-9;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            if (std::abs(data_[i] - other.data_[i]) > PRECISION_TOLERANCE)
            {
                return false;
            }
        }
        return true;
    }

    // overload != operator to compare two tensors
    bool operator!=(const TensorND &other) const
    {
        return !(*this == other);
    }

    // overload = operator to assign a tensor to the tensor
    TensorND &operator=(const TensorND &other)
    {
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
    TensorND operator+(const TensorND &other) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            outp.data_[i] += other.data_[i];
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
        return tensor - scalar;
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
    TensorND operator-(const TensorND &other) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            outp.data_[i] -= other.data_[i];
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
    TensorND operator*(const TensorND &other) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            outp.data_[i] *= other.data_[i];
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
    TensorND operator/(const TensorND &other) const
    {
        TensorND outp = *this;
        for (my_size_t i = 0; i < totalSize; ++i)
        {
            if (other.data_[i] == 0)
            {
                throw std::runtime_error("Division by zero");
            }
            outp.data_[i] /= other.data_[i];
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

    TensorND& transpose(const my_size_t order[sizeof...(Dims)])
    {
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            transposeOrder_[i] = order[i];
        }
        return *this;
    }

    TensorND& transpose(void)
    {
        // check if the tensor is 2D
        static_assert(sizeof...(Dims) == 2, "Transpose is only supported for 2D tensors");
        transposeOrder_[0] = 1;
        transposeOrder_[1] = 0;
        return *this;
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
            shape += std::to_string(dims[transposeOrder_[i]]);
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

    // for all dimentions
    TensorND& setDiagonal(T _val)
    {
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
        
        if constexpr (sizeof...(Dims) == 2) {
            print2D();
        }
        else if constexpr (sizeof...(Dims) == 3) {
            print3D();
        }
        else if constexpr (sizeof...(Dims) == 4) {
            print4D();
        }
    }

private:
    // Calculate total number of elements at compile time
    static constexpr my_size_t totalSize = (Dims * ...);
    static constexpr my_size_t dims[] = {Dims...}; // Fixed array of dimensions
    my_size_t transposeOrder_[sizeof...(Dims)];

    
    T data_[totalSize]; // Contiguous storage of elements in a flat array

    // init the transpose order
    void initTransposeOrder()
    {
        for (my_size_t i = 0; i < getNumDims(); ++i)
        {
            transposeOrder_[i] = i;
        }
    }

    // 2D print function
    void print2D() const {
        // account for the trnaspose order as well
        for (my_size_t i = 0; i < dims[transposeOrder_[0]]; ++i)
        {
            for (my_size_t j = 0; j < dims[transposeOrder_[1]]; ++j)
            {
                // std::cout << "(" << i << "," << j << ") ";
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // 3D print function
    void print3D() const {
        for (my_size_t k = 0; k < dims[transposeOrder_[2]]; ++k) {
            // std::cout << "Slice " << i << ":\n";
            for (my_size_t i = 0; i < dims[transposeOrder_[0]]; ++i) {
                for (my_size_t j = 0; j < dims[transposeOrder_[1]]; ++j) {
                    std::cout << (*this)(i, j, k) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void print4D() const {
        for (my_size_t l = 0; l < dims[transposeOrder_[3]]; ++l) {
            std::cout << "Slice [" << l << "]:\n";
            for (my_size_t k = 0; k < dims[transposeOrder_[2]]; ++k) {
                std::cout << "  Sub-Slice [" << k << "]:\n";
                for (my_size_t i = 0; i < dims[transposeOrder_[0]]; ++i) {
                    std::cout << "    [ ";
                    for (my_size_t j = 0; j < dims[transposeOrder_[1]]; ++j) {
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

            #ifdef MATRIX_USE_BOUNDS_CHECKING
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