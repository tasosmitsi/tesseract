#ifndef FUSEDVECTOR_H
#define FUSEDVECTOR_H

#include "fused_matrix.h"
#include "views/subvector_view.h"

// Derived class: FusedVector
template <typename T, my_size_t Size>
class FusedVector : public FusedMatrix<T, Size, 1>
{
private:
    using Base = FusedMatrix<T, Size, 1>;

public:
    using Base::Base; // Inherit constructors from FusedMatrix
    using Base::operator=;

    // TODO: Add transfomation funtions

    T &operator()(my_size_t i)
    {
        // No needed at the moment, but we can add it later (as constexpr) if we want
        // to support more general vector shapes (e.g., row vector)
        // if (this->getDim(0) == 1)
        // {
        //     return Base::operator()(0, i);
        // }
        // else
        // {
        return Base::operator()(i, 0);
        // }
    }

    const T &operator()(my_size_t i) const
    {
        // No needed at the moment, but we can add it later (as constexpr) if we want
        // to support more general vector shapes (e.g., row vector)
        // if (this->getDim(0) == 1)
        // {
        //     return Base::operator()(0, i);
        // }
        // else
        // {
        return Base::operator()(i, 0);
        // }
    }

    // View of the first K elements
    template <my_size_t K>
    SubVectorView<FusedVector<T, Size>, 0, K> head() const
    {
        return SubVectorView<FusedVector<T, Size>, 0, K>(*this);
    }

    // View of the last K elements
    template <my_size_t K>
    SubVectorView<FusedVector<T, Size>, Size - K, K> tail() const
    {
        return SubVectorView<FusedVector<T, Size>, Size - K, K>(*this);
    }

    // View of Len elements starting at Start
    template <my_size_t Start, my_size_t Len>
    SubVectorView<FusedVector<T, Size>, Start, Len> segment() const
    {
        return SubVectorView<FusedVector<T, Size>, Start, Len>(*this);
    }
};

#endif // FUSEDVECTOR_H
