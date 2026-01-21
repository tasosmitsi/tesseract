// #ifndef FUSEDQUATERNION_H
// #define FUSEDQUATERNION_H

// #include "fused/fused_vector.h"
// #include "fused/BaseExpr.h"
// #include "fused/microkernels/microkernel_base.h"
// #include "config.h"
// #include <cmath>

// /**
//  * @brief Quaternion class using expression templates for SIMD optimization
//  *
//  * Storage order: [w, x, y, z] where w is scalar part, (x,y,z) is vector part
//  * Follows Hamilton convention: i² = j² = k² = ijk = -1
//  */
// template <typename T>
// class FusedQuaternion : public BaseExpr<FusedQuaternion<T>, T>
// {
// public:
//     // Compile-time constants for expression template compatibility
//     static constexpr my_size_t NumDims = 1;
//     static constexpr my_size_t Dim[] = {4};
//     static constexpr my_size_t TotalSize = 4;

//     using Self = FusedQuaternion<T>;
//     using value_type = T;
//     using VecType = typename Microkernel<T, BITS, DefaultArch>::VecType;
//     using microkernel = Microkernel<T, BITS, DefaultArch>;
//     static constexpr my_size_t simdWidth = microkernel::simdWidth;

//     // ========================
//     // Constructors
//     // ========================

//     FusedQuaternion() noexcept
//         : data_()
//     {
//         // Initialize to identity quaternion
//         data_(0) = T{1}; // w
//         data_(1) = T{0}; // x
//         data_(2) = T{0}; // y
//         data_(3) = T{0}; // z
//     }

//     FusedQuaternion(T w, T x, T y, T z) noexcept
//         : data_()
//     {
//         data_(0) = w;
//         data_(1) = x;
//         data_(2) = y;
//         data_(3) = z;
//     }

//     explicit FusedQuaternion(T initValue) noexcept
//         : data_(initValue) {}

//     // Construct from scalar and vector part
//     template <my_size_t N>
//     FusedQuaternion(T scalar, const FusedTensorND<T, N> &vec) noexcept
//     {
//         static_assert(N == 3, "Vector part must be 3D");
//         data_(0) = scalar;
//         data_(1) = vec(0);
//         data_(2) = vec(1);
//         data_(3) = vec(2);
//     }

//     // Copy constructor
//     FusedQuaternion(const FusedQuaternion &other) noexcept
//         : data_(other.data_) {}

//     // Move constructor
//     FusedQuaternion(FusedQuaternion &&other) noexcept
//         : data_(move(other.data_)) {}

//     // ========================
//     // Assignment from expressions (enables SIMD fusion)
//     // ========================

//     template <typename Expr>
//     FusedQuaternion &operator=(const BaseExpr<Expr, T> &expr)
//     {
//         const auto &e = expr.derived();

//         // Compile-time dimension check
//         static_assert(Expr::TotalSize == 4, "Expression must have 4 elements for quaternion assignment");

//         if constexpr (!is_same_v<DefaultArch, GenericArch> && simdWidth >= 4)
//         {
//             // Single SIMD load for all 4 components
//             auto vec = e.evalu(0);
//             microkernel::store(data_.data(), vec);
//         }
//         else
//         {
//             // Scalar fallback
//             my_size_t indices[1];
//             for (my_size_t i = 0; i < 4; ++i)
//             {
//                 indices[0] = i;
//                 data_(i) = e(indices);
//             }
//         }
//         return *this;
//     }

//     FusedQuaternion &operator=(const FusedQuaternion &other) noexcept
//     {
//         if (this != &other)
//         {
//             data_ = other.data_;
//         }
//         return *this;
//     }

//     FusedQuaternion &operator=(FusedQuaternion &&other) noexcept
//     {
//         if (this != &other)
//         {
//             data_ = move(other.data_);
//         }
//         return *this;
//     }

//     // ========================
//     // Expression template interface
//     // ========================

//     // SIMD evaluation - loads all 4 components at once if possible
//     VecType evalu(my_size_t flat) const noexcept
//     {
//         return microkernel::load(data_.data() + flat);
//     }

//     // Scalar access for expression evaluation
//     inline T operator()(const my_size_t *indices) const noexcept
//     {
//         return data_(indices[0]);
//     }

//     inline T &operator()(const my_size_t *indices) noexcept
//     {
//         return data_(indices[0]);
//     }

//     // ========================
//     // Component accessors
//     // ========================

//     FORCE_INLINE T &w() noexcept { return data_(0); }
//     FORCE_INLINE T &x() noexcept { return data_(1); }
//     FORCE_INLINE T &y() noexcept { return data_(2); }
//     FORCE_INLINE T &z() noexcept { return data_(3); }

//     FORCE_INLINE const T &w() const noexcept { return data_(0); }
//     FORCE_INLINE const T &x() const noexcept { return data_(1); }
//     FORCE_INLINE const T &y() const noexcept { return data_(2); }
//     FORCE_INLINE const T &z() const noexcept { return data_(3); }

//     // Index access
//     FORCE_INLINE T &operator[](my_size_t i) noexcept { return data_(i); }
//     FORCE_INLINE const T &operator[](my_size_t i) const noexcept { return data_(i); }

//     // Raw data pointer (for SIMD operations)
//     FORCE_INLINE T *data() noexcept { return data_.data(); }
//     FORCE_INLINE const T *data() const noexcept { return data_.data(); }

//     // ========================
//     // Quaternion-specific operations
//     // ========================

//     // Conjugate: q* = (w, -x, -y, -z)
//     FusedQuaternion conjugate() const noexcept
//     {
//         return FusedQuaternion(w(), -x(), -y(), -z());
//     }

//     // Squared norm: |q|² = w² + x² + y² + z²
//     T normSquared() const noexcept
//     {
//         if constexpr (!is_same_v<DefaultArch, GenericArch> && simdWidth >= 4)
//         {
//             // SIMD dot product
//             VecType v = microkernel::load(data_.data());
//             VecType sq = microkernel::mul(v, v);
//             return microkernel::horizontal_sum(sq);
//         }
//         else
//         {
//             return w() * w() + x() * x() + y() * y() + z() * z();
//         }
//     }

//     // Norm: |q| = sqrt(w² + x² + y² + z²)
//     T norm() const noexcept
//     {
//         return std::sqrt(normSquared());
//     }

//     // Normalize in-place
//     FusedQuaternion &normalize()
//     {
//         T n = norm();
//         if (n > PRECISION_TOLERANCE)
//         {
//             T invNorm = T{1} / n;
//             if constexpr (!is_same_v<DefaultArch, GenericArch> && simdWidth >= 4)
//             {
//                 VecType v = microkernel::load(data_.data());
//                 VecType scale = microkernel::broadcast(invNorm);
//                 microkernel::store(data_.data(), microkernel::mul(v, scale));
//             }
//             else
//             {
//                 data_(0) *= invNorm;
//                 data_(1) *= invNorm;
//                 data_(2) *= invNorm;
//                 data_(3) *= invNorm;
//             }
//         }
//         return *this;
//     }

//     // Return normalized copy
//     FusedQuaternion normalized() const
//     {
//         FusedQuaternion result(*this);
//         result.normalize();
//         return result;
//     }

//     // Inverse: q⁻¹ = q* / |q|²
//     FusedQuaternion inverse() const
//     {
//         T ns = normSquared();
//         if (ns < PRECISION_TOLERANCE)
//         {
//             MyErrorHandler::error("Cannot invert quaternion with zero norm");
//         }
//         T invNormSq = T{1} / ns;
//         return FusedQuaternion(
//             w() * invNormSq,
//             -x() * invNormSq,
//             -y() * invNormSq,
//             -z() * invNormSq);
//     }

//     // ========================
//     // Hamilton product: p * q
//     // ========================

//     FusedQuaternion operator*(const FusedQuaternion &q) const noexcept
//     {
//         // Hamilton product formula:
//         // (p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
//         //  p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
//         //  p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
//         //  p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w)

//         const T pw = w(), px = x(), py = y(), pz = z();
//         const T qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();

//         return FusedQuaternion(
//             pw * qw - px * qx - py * qy - pz * qz, // w
//             pw * qx + px * qw + py * qz - pz * qy, // x
//             pw * qy - px * qz + py * qw + pz * qx, // y
//             pw * qz + px * qy - py * qx + pz * qw  // z
//         );
//     }

//     FusedQuaternion &operator*=(const FusedQuaternion &q) noexcept
//     {
//         *this = *this * q;
//         return *this;
//     }

//     // ========================
//     // Vector rotation: v' = q * v * q⁻¹
//     // ========================

//     template <my_size_t N>
//     FusedTensorND<T, 3> rotate(const FusedTensorND<T, N> &v) const
//     {
//         static_assert(N == 3, "Can only rotate 3D vectors");

//         // Optimized rotation formula (avoiding full quaternion multiplication):
//         // v' = v + 2w(u × v) + 2(u × (u × v))
//         // where u = (x, y, z) is the vector part of the quaternion

//         const T vx = v(0), vy = v(1), vz = v(2);
//         const T ux = x(), uy = y(), uz = z();

//         // t = 2 * (u × v)
//         T tx = T{2} * (uy * vz - uz * vy);
//         T ty = T{2} * (uz * vx - ux * vz);
//         T tz = T{2} * (ux * vy - uy * vx);

//         // v' = v + w*t + (u × t)
//         FusedTensorND<T, 3> result;
//         result(0) = vx + w() * tx + (uy * tz - uz * ty);
//         result(1) = vy + w() * ty + (uz * tx - ux * tz);
//         result(2) = vz + w() * tz + (ux * ty - uy * tx);

//         return result;
//     }

//     // ========================
//     // Static factory methods
//     // ========================

//     static FusedQuaternion identity() noexcept
//     {
//         return FusedQuaternion(T{1}, T{0}, T{0}, T{0});
//     }

//     // Create from axis-angle representation
//     template <my_size_t N>
//     static FusedQuaternion fromAxisAngle(const FusedTensorND<T, N> &axis, T angle)
//     {
//         static_assert(N == 3, "Axis must be 3D");

//         T halfAngle = angle * T{0.5};
//         T s = std::sin(halfAngle);
//         T c = std::cos(halfAngle);

//         // Normalize axis
//         T axisNorm = std::sqrt(axis(0) * axis(0) + axis(1) * axis(1) + axis(2) * axis(2));
//         if (axisNorm < PRECISION_TOLERANCE)
//         {
//             return identity();
//         }
//         T invNorm = T{1} / axisNorm;

//         return FusedQuaternion(
//             c,
//             axis(0) * invNorm * s,
//             axis(1) * invNorm * s,
//             axis(2) * invNorm * s);
//     }

//     // Create from Euler angles (ZYX convention: yaw, pitch, roll)
//     static FusedQuaternion fromEulerAngles(T roll, T pitch, T yaw)
//     {
//         T cr = std::cos(roll * T{0.5});
//         T sr = std::sin(roll * T{0.5});
//         T cp = std::cos(pitch * T{0.5});
//         T sp = std::sin(pitch * T{0.5});
//         T cy = std::cos(yaw * T{0.5});
//         T sy = std::sin(yaw * T{0.5});

//         return FusedQuaternion(
//             cr * cp * cy + sr * sp * sy, // w
//             sr * cp * cy - cr * sp * sy, // x
//             cr * sp * cy + sr * cp * sy, // y
//             cr * cp * sy - sr * sp * cy  // z
//         );
//     }

//     // Convert to Euler angles (returns roll, pitch, yaw)
//     FusedTensorND<T, 3> toEulerAngles() const
//     {
//         FusedTensorND<T, 3> angles;

//         // Roll (x-axis rotation)
//         T sinr_cosp = T{2} * (w() * x() + y() * z());
//         T cosr_cosp = T{1} - T{2} * (x() * x() + y() * y());
//         angles(0) = std::atan2(sinr_cosp, cosr_cosp);

//         // Pitch (y-axis rotation)
//         T sinp = T{2} * (w() * y() - z() * x());
//         if (std::abs(sinp) >= T{1})
//             angles(1) = std::copysign(T{3.14159265358979323846} / T{2}, sinp);
//         else
//             angles(1) = std::asin(sinp);

//         // Yaw (z-axis rotation)
//         T siny_cosp = T{2} * (w() * z() + x() * y());
//         T cosy_cosp = T{1} - T{2} * (y() * y() + z() * z());
//         angles(2) = std::atan2(siny_cosp, cosy_cosp);

//         return angles;
//     }

//     // Spherical linear interpolation (SLERP)
//     static FusedQuaternion slerp(const FusedQuaternion &q0, const FusedQuaternion &q1, T t)
//     {
//         // Compute dot product
//         T dot = q0.w() * q1.w() + q0.x() * q1.x() + q0.y() * q1.y() + q0.z() * q1.z();

//         FusedQuaternion q1_adj = q1;

//         // If dot < 0, negate one quaternion to take shorter path
//         if (dot < T{0})
//         {
//             q1_adj = FusedQuaternion(-q1.w(), -q1.x(), -q1.y(), -q1.z());
//             dot = -dot;
//         }

//         // If quaternions are very close, use linear interpolation
//         if (dot > T{0.9995})
//         {
//             FusedQuaternion result(
//                 q0.w() + t * (q1_adj.w() - q0.w()),
//                 q0.x() + t * (q1_adj.x() - q0.x()),
//                 q0.y() + t * (q1_adj.y() - q0.y()),
//                 q0.z() + t * (q1_adj.z() - q0.z()));
//             return result.normalized();
//         }

//         T theta_0 = std::acos(dot);
//         T theta = theta_0 * t;
//         T sin_theta = std::sin(theta);
//         T sin_theta_0 = std::sin(theta_0);

//         T s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
//         T s1 = sin_theta / sin_theta_0;

//         return FusedQuaternion(
//             s0 * q0.w() + s1 * q1_adj.w(),
//             s0 * q0.x() + s1 * q1_adj.x(),
//             s0 * q0.y() + s1 * q1_adj.y(),
//             s0 * q0.z() + s1 * q1_adj.z());
//     }

//     // ========================
//     // Utility functions
//     // ========================

//     bool isUnit(T tolerance = PRECISION_TOLERANCE) const noexcept
//     {
//         return std::abs(normSquared() - T{1}) < tolerance;
//     }

//     FORCE_INLINE constexpr my_size_t getTotalSize() const noexcept
//     {
//         return TotalSize;
//     }

//     FORCE_INLINE constexpr my_size_t getNumDims() const noexcept
//     {
//         return NumDims;
//     }

//     FORCE_INLINE my_size_t getDim(my_size_t i) const noexcept
//     {
//         return (i == 0) ? 4 : 0;
//     }

//     void print() const
//     {
//         MyErrorHandler::log("Quaternion(w=");
//         MyErrorHandler::log(w());
//         MyErrorHandler::log(", x=");
//         MyErrorHandler::log(x());
//         MyErrorHandler::log(", y=");
//         MyErrorHandler::log(y());
//         MyErrorHandler::log(", z=");
//         MyErrorHandler::log(z());
//         MyErrorHandler::log(")\n");
//     }

// private:
//     // Storage: 4-element vector [w, x, y, z]
//     FusedTensorND<T, 4> data_;
// };

// // ========================
// // Type alias for common types
// // ========================
// using Quaternionf = FusedQuaternion<float>;
// using Quaterniond = FusedQuaternion<double>;

// #endif // FUSEDQUATERNION_H