#ifndef FUSEDVECTOR_H
#define FUSEDVECTOR_H

#include "fused_matrix.h"

// Derived class: FusedVector
template <typename T, my_size_t Size>
class FusedVector : public FusedMatrix<T, Size, 1>
{
public:
    using Base = FusedMatrix<T, Size, 1>;
    using Base::Base; // Inherit constructors from FusedMatrix
    using Base::operator=;

    T &operator()(my_size_t i)
    {
        my_size_t idxArray[2];
        if (this->getDim(0) == 1)
        {
            idxArray[0] = 0;
            idxArray[1] = i;
        }
        else
        {
            idxArray[0] = i;
            idxArray[1] = 0;
        }
        return this->rawData()[this->computeIndex(idxArray)];
    }

    const T &operator()(my_size_t i) const
    {
        my_size_t idxArray[2];
        if (this->getDim(0) == 1)
        {
            idxArray[0] = 0;
            idxArray[1] = i;
        }
        else
        {
            idxArray[0] = i;
            idxArray[1] = 0;
        }
        return this->rawData()[this->computeIndex(idxArray)];
    }
};

#endif // FUSEDVECTOR_H
