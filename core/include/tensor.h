#ifndef TENSORND_H
#define TENSORND_H

#include <stdexcept>
#include <algorithm> // for std::fill_n and std::copy
#include <utility>   // for std::move

/* Define this to enable matrix number of indices checking */
#define MATRIX_USE_NUMBER_OF_INDICES_CHECKING

/* Define this to enable matrix bound checking */
#define MATRIX_USE_BOUNDS_CHECKING


// Base class: TensorND
template <typename T, size_t... Dims>
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

        size_t idxArray[] = {static_cast<size_t>(indices)...}; // Convert indices to an array
        return data_[computeIndex(idxArray)];
    }

    // Const version of the access operator
    template <typename... Indices>
    const T &operator()(Indices... indices) const
    {
        #ifdef MATRIX_USE_NUMBER_OF_INDICES_CHECKING
            static_assert(sizeof...(indices) == sizeof...(Dims), "Incorrect number of indices");
        #endif

        size_t idxArray[] = {static_cast<size_t>(indices)...};
        return data_[computeIndex(idxArray)];
    }

    TensorND& transpose(const size_t order[sizeof...(Dims)])
    {
        for (size_t i = 0; i < getNumDims(); ++i)
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
    constexpr size_t getTotalSize() const
    {
        return totalSize;
    }

    // Utility function to retrieve the number of dimensions
    constexpr size_t getNumDims() const
    {
        return sizeof...(Dims);
    }

    // Utility function to retrieve the shape of the tensor as (1,5,6) for a 3D tensor use the getNumDims
    std::string getShape() const
    // account for the trnaspose order as well
    {
        std::string shape = "(";
        for (size_t i = 0; i < getNumDims(); ++i)
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

    TensorND& setRandom(T _maxRand, T _minRand)
    {
        for (size_t i = 0; i < totalSize; ++i)
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
        size_t minDim = std::min({Dims...}); // Using initializer list to find the minimum
        size_t indices[getNumDims()] = {0}; // Initialize all indices to zero
        
        for (size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (size_t d = 0; d < getNumDims(); ++d) {
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
        for (size_t i = 0; i < totalSize; ++i)
        {
            data_[i] = i;
        }
        return *this;
    }

    template<size_t DiagonalSize>
    void getDiagonalEntries(TensorND<T, DiagonalSize, 1>& diagonalEntries) const
    {
        // Calculate the minimum dimension
        size_t minDim = std::min({Dims...}); // Using initializer list to find the minimum
        size_t indices[getNumDims()] = {0}; // Initialize all indices to zero

        for (size_t i = 0; i < minDim; ++i)
        {
            // Set the current diagonal index for all dimensions
            for (size_t d = 0; d < getNumDims(); ++d) {
                indices[d] = i; // Set the diagonal index, others to zero
            }

            // Calculate the index in the flat array and set the value
            diagonalEntries(i, 0) = data_[computeIndex(indices)];
        }
    }

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
    static constexpr size_t totalSize = (Dims * ...);
    static constexpr size_t dims[] = {Dims...}; // Fixed array of dimensions
    size_t transposeOrder_[sizeof...(Dims)];

    
    T data_[totalSize]; // Contiguous storage of elements in a flat array

    // init the transpose order
    void initTransposeOrder()
    {
        for (size_t i = 0; i < getNumDims(); ++i)
        {
            transposeOrder_[i] = i;
        }
    }

    // 2D print function
    void print2D() const {
        // account for the trnaspose order as well
        for (size_t i = 0; i < dims[transposeOrder_[0]]; ++i)
        {
            for (size_t j = 0; j < dims[transposeOrder_[1]]; ++j)
            {
                // std::cout << "(" << i << "," << j << ") ";
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // 3D print function
    void print3D() const {
        for (size_t k = 0; k < dims[transposeOrder_[2]]; ++k) {
            // std::cout << "Slice " << i << ":\n";
            for (size_t i = 0; i < dims[transposeOrder_[0]]; ++i) {
                for (size_t j = 0; j < dims[transposeOrder_[1]]; ++j) {
                    std::cout << (*this)(i, j, k) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void print4D() const {
        for (size_t l = 0; l < dims[transposeOrder_[3]]; ++l) {
            std::cout << "Slice [" << l << "]:\n";
            for (size_t k = 0; k < dims[transposeOrder_[2]]; ++k) {
                std::cout << "  Sub-Slice [" << k << "]:\n";
                for (size_t i = 0; i < dims[transposeOrder_[0]]; ++i) {
                    std::cout << "    [ ";
                    for (size_t j = 0; j < dims[transposeOrder_[1]]; ++j) {
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
    size_t computeIndex(const size_t indices[getNumDims()]) const {
        size_t index = 0;
        size_t factor = 1;

        for (int i = getNumDims() - 1; i >= 0; --i) {
            size_t dimIndex = transposeOrder_[i]; // Get dimension according to transpose order

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