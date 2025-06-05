#ifndef VECTOR_H
#define VECTOR_H

#include "tensor.h"
#include <iostream>


// Derived class: Vector
template <typename T, my_size_t Size>
class Vector : public TensorND<T, Size>
{
public:
    // Default constructor initializes a vector with default values
    Vector() : TensorND<T, Size>() {}

    // Constructor to initialize all elements to a specific value
    Vector(T initValue) : TensorND<T, Size>(initValue) {}

    // Copy constructor
    Vector(const Vector &other) : TensorND<T, Size>(other) {}

    // Move constructor
    Vector(Vector &&other) noexcept : TensorND<T, Size>(std::move(other)) {}

    // Constructor from an array
    Vector(T (&initList)[Size]) : TensorND<T, Size>()
    {
        for (my_size_t i = 0; i < Size; ++i)
        {
            (*this)(i) = initList[i];
        }
    }
};

#endif // VECTOR_H