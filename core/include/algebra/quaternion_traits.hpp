// #pragma once

// #include "algebra/algebraic_traits.h"
// #include "fused/fused_quaternion.h"

// namespace algebra
// {
//     template <typename T>
//     struct algebraic_traits<math::FusedQuaternion<T>>
//     {
//         static constexpr bool vector_space = true; // q + q, q * scalar
//         static constexpr bool algebra = true;      // Hamilton product
//         static constexpr bool lie_group = false;   // not unit length
//         static constexpr bool metric = true;       // dot, norm
//         static constexpr bool tensor = false;      // NOT shape-based
//     };

// } // namespace algebra
