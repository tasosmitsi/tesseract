#ifndef FUSEDVECTOR_H
#define FUSEDVECTOR_H

#include "fused_matrix.h"

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
        if (this->getDim(0) == 1)
        {
            return Base::operator()(0, i);
        }
        else
        {
            return Base::operator()(i, 0);
        }
    }

    const T &operator()(my_size_t i) const
    {
        if (this->getDim(0) == 1)
        {
            return Base::operator()(0, i);
        }
        else
        {
            return Base::operator()(i, 0);
        }
    }
};

#endif // FUSEDVECTOR_H
