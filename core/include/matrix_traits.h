#ifndef MATRIXTRAITS_H
#define MATRIXTRAITS_H

namespace matrix_traits
{
    enum class Definiteness
    {
        NotPositiveDefinite = 0,  // Matrix is neither positive definite nor semi-definite
        PositiveSemiDefinite = 1, // Matrix is positive semi-definite
        PositiveDefinite = 2,     // Matrix is positive definite
    };
}

#endif // MATRIXTRAITS_H
