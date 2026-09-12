/**
 * @file test_expected.cpp
 * @brief Catch2 tests for Expected<T, E> and Unexpected<E>.
 *
 * Tests cover:
 *   - Success and error construction
 *   - Move semantics (move construct, move assign)
 *   - Deleted copy operations (compile-time enforced, not runtime-testable)
 *   - Observer methods (operator bool, has_value)
 *   - Value access (value(), operator*, operator->)
 *   - Error access
 *   - Unexpected deduction guide
 *   - Destruction of non-trivial types
 *   - Mixed state transitions via move assignment
 *
 * ============================================================================
 * TYPE SETUP
 * ============================================================================
 *
 * We test with:
 *   - Trivial types (int, float, double) — no destructor concerns
 *   - A non-trivial type (Tracker) — verifies correct destroy() dispatch
 *
 * Error type is a simple enum class to mirror real usage (MatrixStatus).
 *
 * ============================================================================
 */

#include <catch_amalgamated.hpp>

#include "utilities/expected.h"

// ============================================================================
// TEST ERROR ENUM
// ============================================================================

enum class TestError : unsigned char
{
    Ok = 0,
    BadInput,
    Overflow,
    Singular
};

// ============================================================================
// NON-TRIVIAL TYPE — tracks construction / destruction
// ============================================================================

static int g_tracker_alive = 0;

struct Tracker
{
    int val;

    explicit Tracker(int v) : val(v) { ++g_tracker_alive; }
    Tracker(Tracker &&other) noexcept : val(other.val)
    {
        other.val = -1;
        ++g_tracker_alive;
    }
    Tracker &operator=(Tracker &&other) noexcept
    {
        val = other.val;
        other.val = -1;
        return *this;
    }
    ~Tracker() { --g_tracker_alive; }

    Tracker(const Tracker &) = delete;
    Tracker &operator=(const Tracker &) = delete;
};

// ============================================================================
// CONSTRUCTION — SUCCESS
// ============================================================================

TEMPLATE_TEST_CASE("Expected success construction from rvalue",
                   "[expected]", int, float, double)
{
    using T = TestType;

    T val = T(42);
    Expected<T, TestError> result(move(val));

    REQUIRE(result.has_value());
    REQUIRE(static_cast<bool>(result));
    REQUIRE(*result == Catch::Approx(T(42)));
}

TEST_CASE("Expected success construction with non-trivial type",
          "[expected]")
{
    g_tracker_alive = 0;

    {
        Tracker t(99);
        REQUIRE(g_tracker_alive == 1);

        Expected<Tracker, TestError> result(move(t));
        REQUIRE(g_tracker_alive == 2); // original + moved-into
        REQUIRE(result.has_value());
        REQUIRE(result->val == 99);
    }

    REQUIRE(g_tracker_alive == 0);
}

// ============================================================================
// CONSTRUCTION — ERROR
// ============================================================================

TEST_CASE("Expected error construction via Unexpected",
          "[expected]")
{
    Expected<int, TestError> result(Unexpected{TestError::BadInput});

    REQUIRE_FALSE(result.has_value());
    REQUIRE_FALSE(static_cast<bool>(result));
    REQUIRE(result.error() == TestError::BadInput);
}

TEST_CASE("Expected error construction does not construct T",
          "[expected]")
{
    g_tracker_alive = 0;

    {
        Expected<Tracker, TestError> result(Unexpected{TestError::Singular});

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == TestError::Singular);
        REQUIRE(g_tracker_alive == 0); // Tracker never constructed
    }

    REQUIRE(g_tracker_alive == 0);
}

// ============================================================================
// UNEXPECTED DEDUCTION GUIDE
// ============================================================================

TEST_CASE("Unexpected CTAD deduces error type",
          "[unexpected]")
{
    auto u = Unexpected{TestError::Overflow};
    REQUIRE(u.error == TestError::Overflow);

    // Verify it converts to Expected
    Expected<int, TestError> result(u);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == TestError::Overflow);
}

// ============================================================================
// VALUE ACCESS
// ============================================================================

TEST_CASE("Expected value() returns mutable reference",
          "[expected]")
{
    int v = 10;
    Expected<int, TestError> result(move(v));

    result.value() = 20;
    REQUIRE(*result == 20);
}

TEST_CASE("Expected const value() returns const reference",
          "[expected]")
{
    int v = 10;
    const Expected<int, TestError> result(move(v));

    REQUIRE(result.value() == 10);
    REQUIRE(*result == 10);
}

TEST_CASE("Expected arrow operator accesses members",
          "[expected]")
{
    Tracker t(77);
    Expected<Tracker, TestError> result(move(t));

    REQUIRE(result->val == 77);

    // Mutate through arrow
    result->val = 88;
    REQUIRE((*result).val == 88);
}

// ============================================================================
// MOVE CONSTRUCTION
// ============================================================================

TEMPLATE_TEST_CASE("Expected move construction transfers success value",
                   "[expected]", int, float, double)
{
    using T = TestType;

    T val = T(123);
    Expected<T, TestError> a(move(val));
    Expected<T, TestError> b(move(a));

    REQUIRE(b.has_value());
    REQUIRE(*b == Catch::Approx(T(123)));
}

TEST_CASE("Expected move construction transfers error",
          "[expected]")
{
    Expected<int, TestError> a(Unexpected{TestError::Singular});
    Expected<int, TestError> b(move(a));

    REQUIRE_FALSE(b.has_value());
    REQUIRE(b.error() == TestError::Singular);
}

TEST_CASE("Expected move construction with non-trivial type",
          "[expected]")
{
    g_tracker_alive = 0;

    {
        Tracker t(55);
        Expected<Tracker, TestError> a(move(t));
        REQUIRE(g_tracker_alive == 2);

        Expected<Tracker, TestError> b(move(a));
        REQUIRE(g_tracker_alive == 3); // source t is still alive (val=-1)
        REQUIRE(b.has_value());
        REQUIRE(b->val == 55);
    }

    REQUIRE(g_tracker_alive == 0);
}

// ============================================================================
// MOVE ASSIGNMENT
// ============================================================================

TEST_CASE("Expected move assignment: success to success",
          "[expected]")
{
    int v1 = 10, v2 = 20;
    Expected<int, TestError> a(move(v1));
    Expected<int, TestError> b(move(v2));

    b = move(a);

    REQUIRE(b.has_value());
    REQUIRE(*b == 10);
}

TEST_CASE("Expected move assignment: error to success",
          "[expected]")
{
    int v = 10;
    Expected<int, TestError> a(Unexpected{TestError::Overflow});
    Expected<int, TestError> b(move(v));

    b = move(a);

    REQUIRE_FALSE(b.has_value());
    REQUIRE(b.error() == TestError::Overflow);
}

TEST_CASE("Expected move assignment: success to error",
          "[expected]")
{
    int v = 42;
    Expected<int, TestError> a(move(v));
    Expected<int, TestError> b(Unexpected{TestError::BadInput});

    b = move(a);

    REQUIRE(b.has_value());
    REQUIRE(*b == 42);
}

TEST_CASE("Expected move assignment: error to error",
          "[expected]")
{
    Expected<int, TestError> a(Unexpected{TestError::Singular});
    Expected<int, TestError> b(Unexpected{TestError::BadInput});

    b = move(a);

    REQUIRE_FALSE(b.has_value());
    REQUIRE(b.error() == TestError::Singular);
}

// ============================================================================
// DESTRUCTION — NON-TRIVIAL TYPE LIFECYCLE
// ============================================================================

TEST_CASE("Expected destroys T on success path destruction",
          "[expected]")
{
    g_tracker_alive = 0;

    {
        Tracker t(1);
        Expected<Tracker, TestError> result(move(t));
        REQUIRE(g_tracker_alive == 2);
    }

    REQUIRE(g_tracker_alive == 0);
}

TEST_CASE("Expected does not destroy T on error path destruction",
          "[expected]")
{
    g_tracker_alive = 0;

    {
        Expected<Tracker, TestError> result(Unexpected{TestError::Singular});
        REQUIRE(g_tracker_alive == 0);
    }

    REQUIRE(g_tracker_alive == 0);
}

TEST_CASE("Expected move assignment destroys old T before overwriting",
          "[expected]")
{
    g_tracker_alive = 0;

    {
        Tracker t1(1);
        Tracker t2(2);
        Expected<Tracker, TestError> a(move(t1));
        Expected<Tracker, TestError> b(move(t2));
        REQUIRE(g_tracker_alive == 4);

        // b holds Tracker(2), about to be overwritten by a's Tracker(1)
        b = move(a);
        // old b's T destroyed, new T placement-new'd from a
        REQUIRE(b.has_value());
        REQUIRE(b->val == 1);
    }

    REQUIRE(g_tracker_alive == 0);
}

TEST_CASE("Expected move assignment from error destroys old T",
          "[expected]")
{
    g_tracker_alive = 0;

    {
        Tracker t(1);
        Expected<Tracker, TestError> a(Unexpected{TestError::BadInput});
        Expected<Tracker, TestError> b(move(t));
        REQUIRE(g_tracker_alive == 2); // original t + b's Tracker

        b = move(a); // b had a live Tracker, now gets error
        // b's Tracker should be destroyed
        REQUIRE_FALSE(b.has_value());
    }

    REQUIRE(g_tracker_alive == 0);
}

// ============================================================================
// REALISTIC USAGE PATTERN — function returning Expected
// ============================================================================

static Expected<int, TestError> safe_divide(int a, int b)
{
    if (b == 0)
        return Unexpected{TestError::Singular};

    int result = a / b;
    return move(result);
}

TEST_CASE("Expected realistic usage: safe_divide success",
          "[expected]")
{
    auto result = safe_divide(10, 2);

    REQUIRE(result.has_value());
    REQUIRE(*result == 5);
}

TEST_CASE("Expected realistic usage: safe_divide error",
          "[expected]")
{
    auto result = safe_divide(10, 0);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == TestError::Singular);
}

TEST_CASE("Expected realistic usage: error propagation pattern",
          "[expected]")
{
    // Simulate chained operations where first fails
    auto step1 = safe_divide(10, 0);
    if (!step1)
    {
        // Propagate: wrap in new Expected of different value type
        Expected<double, TestError> step2(Unexpected{step1.error()});
        REQUIRE_FALSE(step2.has_value());
        REQUIRE(step2.error() == TestError::Singular);
    }
}

// ============================================================================
// ALL ERROR ENUM VALUES
// ============================================================================

TEST_CASE("Expected can carry all TestError values",
          "[expected]")
{
    auto check = [](TestError e)
    {
        Expected<int, TestError> result(Unexpected{e});
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == e);
    };

    check(TestError::Ok);
    check(TestError::BadInput);
    check(TestError::Overflow);
    check(TestError::Singular);
}
