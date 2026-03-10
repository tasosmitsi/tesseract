#ifndef FUSED_EXPECTED_H
#define FUSED_EXPECTED_H

#include "simple_type_traits.h" // move, forward, is_trivially_destructible, is_nothrow_move_constructible

/**
 * @file expected.h
 * @brief A minimal, STL-free expected/result type for failable operations.
 *
 * Modeled after std::expected (C++23) but with no STL dependencies,
 * no exceptions, and no dynamic allocation. Suitable for freestanding
 * and embedded targets.
 *
 * Uses the library's own simple_type_traits.h for move/forward and
 * a built-in placement new to avoid pulling in any standard headers.
 */

/**
 * @brief Tag type for constructing an Expected in the error state.
 *
 * Usage:
 * @code
 *   return Unexpected{MyErrorEnum::SomeError};
 * @endcode
 *
 * @tparam E Error type.
 */
template <typename E>
struct Unexpected
{
    E error; ///< The error value.
};

/**
 * @brief Deduction guide: allows `Unexpected{MyErrorEnum::SomeError}` without
 *        specifying the template parameter explicitly.
 */
template <typename E>
Unexpected(E) -> Unexpected<E>;

/**
 * @brief A discriminated union holding either a success value or an error.
 *
 * Designed for failable operations in the fused library (decompositions,
 * solvers, etc.). Uses a union internally so that only the active member
 * is ever constructed — on the error path, @p T is never touched.
 *
 * Move-only by design to prevent accidental copies of large results
 * (e.g. matrix types) on embedded targets. With C++17 guaranteed copy
 * elision and NRVO, returning by value from algorithms is zero-cost.
 *
 * Marked @c [[nodiscard]] so the compiler warns if a caller discards
 * the result without checking the error state.
 *
 * @tparam T Success value type.
 * @tparam E Error type (e.g. MatrixStatus, FilterStatus). No default —
 *           callers must always specify the error type explicitly.
 *
 * @par Example — returning success:
 * @code
 *   Expected<Matrix3f, MatrixStatus> cholesky(const Matrix3f& A) {
 *       Matrix3f L;
 *       // ... compute L ...
 *       return move(L);  // or just `return L;` with NRVO
 *   }
 * @endcode
 *
 * @par Example — returning an error:
 * @code
 *   if (!A.isSymmetric())
 *       return Unexpected{MatrixStatus::NotSymmetric};
 * @endcode
 *
 * @par Example — caller side:
 * @code
 *   auto result = cholesky(A);
 *   if (!result) {
 *       handle(result.error());
 *       return;
 *   }
 *   auto& L = *result;
 * @endcode
 */
template <typename T, typename E>
class [[nodiscard]] Expected
{
    union
    {
        T val_;
        E err_;
    };
    bool has_value_;

    /**
     * @brief Destroy the active union member.
     *
     * Only calls the destructor of @p T if it is non-trivially destructible
     * and we are in the success state. @p E is assumed trivially destructible
     * (it is typically an unsigned char enum).
     */
    void destroy() noexcept
    {
        if (has_value_)
        {
            if constexpr (!is_trivially_destructible_v<T>)
            {
                val_.~T();
            }
        }
    }

public:
    using value_type = T;
    using error_type = E;

    // -- Construction -------------------------------------------------------

    /**
     * @brief Construct in the success state by moving a value in.
     * @param v The success value (moved from).
     */
    Expected(T &&v) noexcept(is_nothrow_move_constructible_v<T>)
        : val_(move(v)), has_value_(true)
    {
    }

    /**
     * @brief Construct in the error state from an Unexpected tag.
     * @param u An Unexpected wrapper carrying the error code.
     */
    Expected(Unexpected<E> u) noexcept
        : err_(u.error), has_value_(false)
    {
    }

    // -- Move semantics (no copy) ------------------------------------------

    /**
     * @brief Move constructor.
     *
     * Placement-new constructs the active member from the source.
     * The source is left in a valid but unspecified state.
     */
    Expected(Expected &&other) noexcept(is_nothrow_move_constructible_v<T>)
        : has_value_(other.has_value_)
    {
        if (has_value_)
            new (&val_) T(move(other.val_));
        else
            err_ = other.err_;
    }

    /**
     * @brief Move assignment.
     *
     * Destroys the current active member, then placement-new constructs
     * from the source.
     */
    Expected &operator=(Expected &&other) noexcept(is_nothrow_move_constructible_v<T>)
    {
        if (this != &other)
        {
            destroy();
            has_value_ = other.has_value_;
            if (has_value_)
                new (&val_) T(move(other.val_));
            else
                err_ = other.err_;
        }
        return *this;
    }

    /// @brief Deleted — Expected is move-only to prevent accidental copies.
    Expected(const Expected &) = delete;

    /// @brief Deleted — Expected is move-only to prevent accidental copies.
    Expected &operator=(const Expected &) = delete;

    /**
     * @brief Destructor. Destroys the active union member.
     */
    ~Expected() noexcept { destroy(); }

    // -- Observers ----------------------------------------------------------

    /**
     * @brief Returns true if the Expected holds a success value.
     */
    explicit operator bool() const noexcept { return has_value_; }

    /**
     * @brief Returns true if the Expected holds a success value.
     */
    bool has_value() const noexcept { return has_value_; }

    // -- Value access -------------------------------------------------------

    /**
     * @brief Access the success value by reference.
     * @warning Undefined behavior if in the error state.
     */
    T &value() noexcept { return val_; }

    /**
     * @brief Access the success value by const reference.
     * @warning Undefined behavior if in the error state.
     */
    const T &value() const noexcept { return val_; }

    /**
     * @brief Dereference operator — access the success value.
     * @warning Undefined behavior if in the error state.
     */
    T &operator*() noexcept { return val_; }

    /**
     * @brief Dereference operator (const) — access the success value.
     * @warning Undefined behavior if in the error state.
     */
    const T &operator*() const noexcept { return val_; }

    /**
     * @brief Arrow operator — access members of the success value.
     * @warning Undefined behavior if in the error state.
     */
    T *operator->() noexcept { return &val_; }

    /**
     * @brief Arrow operator (const) — access members of the success value.
     * @warning Undefined behavior if in the error state.
     */
    const T *operator->() const noexcept { return &val_; }

    // -- Error access -------------------------------------------------------

    /**
     * @brief Access the error code.
     * @warning Undefined behavior if in the success state.
     */
    E error() const noexcept { return err_; }

    // -- Monadic operations (future) ----------------------------------------
    // template <typename F> auto and_then(F&& f) -> Expected<...>;
    // template <typename F> auto transform(F&& f) -> Expected<...>;
    // template <typename F> auto or_else(F&& f) -> Expected<...>;
};

#endif // FUSED_EXPECTED_H
