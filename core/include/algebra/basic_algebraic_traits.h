#pragma once

namespace algebra
{
    // ===============================
    // Algebraic trait definition
    // ===============================
    template <typename T>
    struct algebraic_traits
    {
        /*  Linear structure: meaning (math): A type is a vector space over ℝ if it supports:
                Addition: v + w
                Subtraction: v - w
                Zero element
                Scalar multiplication: v * s, s * v
                Distributivity & associativity
            Examples:
                ✔ Vector3
                ✔ TensorND
                ✔ Quaternion
                ✔ so(3)
                ❌ UnitQuaternion
                ❌ RotationMatrix (SO(3))
            If vector_space == true, you allow:
            v1 + v2
            v1 - v2
            v * scalar
            If false → these operations must not compile. Simply use: requires algebraic_traits<T>::vector_space
         */
        static constexpr bool vector_space = false;

        // Closed associative multiplication
        /*  Meaning (math): An algebra is a vector space plus a multiplication: 𝐴 × 𝐴 → 𝐴
            that is:
                closed
                associative (usually)
                bilinear
            Examples:
                ✔ Quaternion (Hamilton product)
                ✔ Matrix
                ✔ DualQuaternion
                ✔ Clifford algebra elements
                ❌ Vector3
                ❌ so(3)
                ❌ UnitQuaternion
            If algebra == true, you allow:
                a * b   // special multiplication (not element-wise)
                This is where Hamilton product lives.
            If false → operator* between two entities is illegal.
         */

        static constexpr bool algebra = false;

        // Lie group structure (composition + inverse)
        /*  Meaning (math): A Lie group is:
                A group (identity, inverse, closure)
                Smooth (continuous)
                NOT a vector space
            Operations:
                Composition
                Inverse
            Examples:
                ✔ UnitQuaternion (SO(3))
                ✔ SE(3)
                ✔ DualQuaternion (rigid transforms)
                ❌ Quaternion
                ❌ TensorND
            Lie groups:
                do NOT support addition
                do NOT support scalar multiplication
            So this flag disables:
                q + q
                q * scalar
            and enables:
                q1 * q2   // composition
                inv(q)
         */
        static constexpr bool lie_group = false;

        // Dot / norm / distance
        /*  Meaning (math): A metric space has:
                a dot product
                a norm / length
                distance
            Examples:
                ✔ Vector3
                ✔ Quaternion
                ✔ so(3)
                ❌ TensorND (general tensors do not define dot)
                ❌ UnitQuaternion (distance is on manifold, not linear)
            If metric == true, you allow:
                dot(a, b)
                norm(a)
            This is NOT an operator — it’s named functions.
         */
        static constexpr bool metric = false;

        // Shape-based tensor semantics
        /*  Meaning (math / semantics): This does not mean “tensor algebra”.
            It means: “This type’s semantics are governed by shape and rank, not algebraic laws.”
            Examples:
                ✔ TensorND
                ✔ Matrix
                ✔ Image / Volume data
                ❌ Quaternion
                ❌ Vector3
            This flag controls:
                dimension checks
                broadcasting rules
                index-based access
                slicing semantics
            It prevents accidental mixing like:
                TensorND<3,3> + Quaternion   // illegal
            even though both are vector spaces.
         */
        static constexpr bool tensor = false;
    };

    // ===============================
    // Convenience helpers
    // ===============================
    template <typename T>
    inline constexpr bool is_vector_space_v = algebraic_traits<T>::vector_space;

    template <typename T>
    inline constexpr bool is_algebra_v = algebraic_traits<T>::algebra;

    template <typename T>
    inline constexpr bool is_lie_group_v = algebraic_traits<T>::lie_group;

    template <typename T>
    inline constexpr bool is_metric_v = algebraic_traits<T>::metric;

    template <typename T>
    inline constexpr bool is_tensor_v = algebraic_traits<T>::tensor;
} // namespace algebra