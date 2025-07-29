#pragma once

struct GenericArch
{
};

template <typename T, typename Arch = GenericArch>
struct OpTraits
{
    static T add(T a, T b) { return a + b; }
    static T sub(T a, T b) { return a - b; }
    static T mul(T a, T b) { return a * b; }
    static T div(T a, T b)
    {
        if (b == T(0))
            MyErrorHandler::error("Division by zero");
        return a / b;
    }

    static void test() { std::cout << "Generic\n"; }
};
