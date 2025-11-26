#ifndef VECTOR_H
#define VECTOR_H

#include "tensor.h"

// Derived class: Vector
template <typename T, my_size_t Size>
class Vector : public TensorND<T, Size, 1>
{
public:
    // Default constructor
    Vector() : TensorND<T, Size, 1>() {}
    // Constructor to initialize all elements to a specific value
    Vector(T initValue) : TensorND<T, Size, 1>(initValue) {}
    // Copy constructor
    Vector(const Vector &other) : TensorND<T, Size, 1>(other) {}
    // Move constructor
    Vector(Vector &&other) noexcept : TensorND<T, Size, 1>(std::move(other)) {}
    // Constructor from an array
    Vector(const T (&array)[Size]) : TensorND<T, Size, 1>()
    {
        for (my_size_t i = 0; i < Size; ++i)
        {
            this->data_[i] = array[i];
        }
    }

    // overload () operator to access elements
    T &operator()(my_size_t index)
    {
        return TensorND<T, Size, 1>::operator()(index, 0);
    }

    const T &operator()(my_size_t index) const
    {
        return TensorND<T, Size, 1>::operator()(index, 0);
    }
};

#endif // VECTOR_H
