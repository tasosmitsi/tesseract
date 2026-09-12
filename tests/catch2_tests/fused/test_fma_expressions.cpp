#include <catch_amalgamated.hpp>
#include "fused/fused_tensor.h"
#include "simple_type_traits.h" // for is_same_v

/*
The preprocessor normally treats every comma as a separator between macro arguments. 
So ScalarFmaExpr<TensorT, T, TensorT, Fma> looks like 4 separate arguments to it.
__VA_ARGS__ is a variadic macro parameter (from the ... in the definition). 
It captures everything after the second argument as a single blob,
commas included. So the macro receives:

name → r3
expr → A * (T)2.0 + C
... / __VA_ARGS__ → ScalarFmaExpr<TensorT, T, TensorT, Fma>

Then it expands to:
auto r3 = A * (T)2.0 + C;
static_assert(is_same_v<decltype(r3), ScalarFmaExpr<TensorT, T, TensorT, Fma>>);
*/

#define CHECK_FMA_TYPE(name, expr, ...) \
    auto name = expr;                   \
    static_assert(is_same_v<decltype(name), __VA_ARGS__>)

TEMPLATE_TEST_CASE("FMA expressions", "[fma_expressions]", double, float)
{
    using T = TestType;

    using TensorT = FusedTensorND<T, 6, 5>;
    TensorT A, B, C;

    CHECK_FMA_TYPE(r1, A * B + C,           FmaExpr<TensorT, TensorT, TensorT, Fma>);
    CHECK_FMA_TYPE(r2, C + A * B,           FmaExpr<TensorT, TensorT, TensorT, Fma>);
    CHECK_FMA_TYPE(r3, A * (T)2.0 + C,      ScalarFmaExpr<TensorT, T, TensorT, Fma>);
    CHECK_FMA_TYPE(r4, (T)2.0 * A + C,      ScalarFmaExpr<TensorT, T, TensorT, Fma>);
    CHECK_FMA_TYPE(r5, C + A * (T)2.0,      ScalarFmaExpr<TensorT, T, TensorT, Fma>);
    CHECK_FMA_TYPE(r6, C + (T)2.0 * A,      ScalarFmaExpr<TensorT, T, TensorT, Fma>);
    CHECK_FMA_TYPE(r7, -(A * B) + C,        FmaExpr<TensorT, TensorT, TensorT, Fnma>);
    CHECK_FMA_TYPE(r8, -(A * (T)2.0) + C,   ScalarFmaExpr<TensorT, T, TensorT, Fnma>);
    CHECK_FMA_TYPE(r9, -((T)2.0 * B) + C,   ScalarFmaExpr<TensorT, T, TensorT, Fnma>);
    CHECK_FMA_TYPE(r10, A * B - C,          FmaExpr<TensorT, TensorT, TensorT, Fms>);
    CHECK_FMA_TYPE(r11, C - A * B,          FmaExpr<TensorT, TensorT, TensorT, Fnma>);
    CHECK_FMA_TYPE(r12, A * (T)2.0 - C,     ScalarFmaExpr<TensorT, T, TensorT, Fms>);
    CHECK_FMA_TYPE(r13, (T)2.0 * A - C,     ScalarFmaExpr<TensorT, T, TensorT, Fms>);
    CHECK_FMA_TYPE(r14, C - A * (T)2.0,     ScalarFmaExpr<TensorT, T, TensorT, Fnma>);
    CHECK_FMA_TYPE(r15, C - (T)2.0 * A,     ScalarFmaExpr<TensorT, T, TensorT, Fnma>);
    CHECK_FMA_TYPE(r16, -(A * B) - C,       FmaExpr<TensorT, TensorT, TensorT, Fnms>);
    CHECK_FMA_TYPE(r17, -(A * (T)2.0) - C,  ScalarFmaExpr<TensorT, T, TensorT, Fnms>);
    CHECK_FMA_TYPE(r18, -((T)2.0 * B) - C,  ScalarFmaExpr<TensorT, T, TensorT, Fnms>);

    SUCCEED("All FMA expression types correctly detected at compile time");
}