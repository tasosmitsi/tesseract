#pragma once

struct GenericArch
{
};

template <typename T, my_size_t Bits, typename Arch = GenericArch>
struct OpTraits
{
    static constexpr my_size_t width = 0; // no SIMD support
    using type = T;

    FORCE_INLINE static T add(T a, T b) { return a + b; }
    FORCE_INLINE static T sub(T a, T b) { return a - b; }
    FORCE_INLINE static T mul(T a, T b) { return a * b; }
    FORCE_INLINE static T div(T a, T b)
    {
        if (b == T(0))
            MyErrorHandler::error("Division by zero");
        return a / b;
    }

    static void test() { std::cout << "OpTraits: generic\n"; }
};
