// #ifndef FUSEDQUATERNION_H
// #define FUSEDQUATERNION_H

// #include "fused/fused_vector.h"
// // Derived class: FusedQuaternion
// template <typename T>
// class FusedQuaternion : public FusedVector<T, 4>
// {
// public:
//     using Base = FusedVector<T, 4>;
//     using Base::Base; // Inherit constructors from FusedVector
//     using Base::operator=;

//     double dot(const FusedQuaternion &q) const; // preferred
//     // OR
//     double operator|(const FusedQuaternion &q) const; // less common, but concise

//     FusedQuaternion conjugate() const;
//     double norm() const;
//     FusedQuaternion inverse() const;

//     // Accessors for quaternion components
//     T &w() { return (*this)(0); }
//     T &x() { return (*this)(1); }
//     T &y() { return (*this)(2); }
//     T &z() { return (*this)(3); }

//     const T &w() const { return (*this)(0); }
//     const T &x() const { return (*this)(1); }
//     const T &y() const { return (*this)(2); }
//     const T &z() const { return (*this)(3); }
// };

// #endif // FUSEDQUATERNION_H
