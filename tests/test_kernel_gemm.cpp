/**
 * @file test_kernel_gemm.cpp
 * @brief Catch2 tests for KernelGemm and einsum GEMM dispatch path.
 *
 * Tests cover:
 *   - Direct KernelGemm::gemm() with raw pointers and strides
 *   - einsum() GEMM dispatch for all 4 axis combos (a,b ∈ {0,1})
 *   - Remainder handling: M%MR, N%NR, N%simdWidth
 *   - Padding-inducing dimensions (lastDim not multiple of simdWidth)
 *   - Degenerate shapes: single row, single column, 1×1
 *   - Correctness vs naive scalar reference
 *   - Both float and double (different simdWidth → different tiling)
 *
 * ============================================================================
 * PHYSICAL LAYOUT REFERENCE (doubles, simdWidth=4)
 * ============================================================================
 *
 * FusedTensorND<double, 5, 7> padded to [5, 8]:
 *
 *   row 0: [v v v v v v v P]    stride(0) = 8
 *   row 1: [v v v v v v v P]    stride(1) = 1
 *   ...                          ^padding
 *
 *   GEMM tiling with MR=4, NR=12, simdWidth=4:
 *     wide_N   = (7/12)*12 = 0    → no wide tiles
 *     narrow_N = (7/4)*4   = 4    → one narrow tile (cols 0-3)
 *     scalar cols 4,5,6            → three scalar columns
 *     Main body: i=0 (MR=4 rows)
 *     Remainder: i=4 (1 row)
 *
 * ============================================================================
 */

#include <catch_amalgamated.hpp>

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/kernel_ops/kernel_ops.h"
#include "fused/fused_tensor.h"

using Catch::Approx;

// ============================================================================
// HELPERS
// ============================================================================

/// Naive scalar matmul for reference: C[M,N] = A[M,K] * B[K,N]
/// Operates on raw physical memory with strides (like the GEMM kernel does).
template <typename T>
static void naive_gemm(
    const T *A, my_size_t M, my_size_t K, my_size_t strideA,
    const T *B, my_size_t N, my_size_t strideB,
    T *C, my_size_t strideC)
{
    for (my_size_t i = 0; i < M; ++i)
        for (my_size_t j = 0; j < N; ++j)
        {
            T sum = T{0};
            for (my_size_t k = 0; k < K; ++k)
                sum += A[i * strideA + k] * B[k * strideB + j];
            C[i * strideC + j] = sum;
        }
}

/// Naive einsum reference: contracts axis a of tensor1 with axis b of tensor2.
/// Returns the expected output tensor filled via scalar triple loop.
template <typename T, my_size_t... DimsOut, my_size_t... Dims1, my_size_t... Dims2>
static FusedTensorND<T, DimsOut...> naive_einsum(
    const FusedTensorND<T, Dims1...> &A,
    const FusedTensorND<T, Dims2...> &B,
    my_size_t a, my_size_t b)
{
    const my_size_t M = A.getDim(1 - a);
    const my_size_t N = B.getDim(1 - b);
    const my_size_t K = A.getDim(a);

    FusedTensorND<T, DimsOut...> C;

    for (my_size_t i = 0; i < M; ++i)
        for (my_size_t j = 0; j < N; ++j)
        {
            T sum = T{0};
            for (my_size_t k = 0; k < K; ++k)
            {
                // A indices: free dim = i, contract dim = k
                T a_val = (a == 1) ? A(i, k) : A(k, i);
                // B indices: contract dim = k, free dim = j
                T b_val = (b == 0) ? B(k, j) : B(j, k);
                sum += a_val * b_val;
            }
            C(i, j) = sum;
        }

    return C;
}

// ============================================================================
// DIRECT KernelGemm TESTS — raw pointer interface
// ============================================================================
//
// These test the GEMM kernel directly, bypassing einsum dispatch.
// All inputs are in the favorable layout: A[M,K] row-major, B[K,N] row-major.

TEMPLATE_TEST_CASE("gemm direct: 4x3 * 3x4 exact MR fit", "[gemm][direct][test_kernel_gemm]", double, float)
{
    using T = TestType;
    using Gemm = detail::KernelGemm<T, BITS, DefaultArch>;

    // A[4,3] padded to [4,4]:
    //   [1 2 3 P]    B[3,4] padded to [3,4]:
    //   [4 5 6 P]      [1 0 2 1]
    //   [7 8 9 P]      [0 1 1 0]
    //   [2 1 3 P]      [2 1 0 1]
    FusedTensorND<T, 4, 3> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    A(2, 0) = 7;
    A(2, 1) = 8;
    A(2, 2) = 9;
    A(3, 0) = 2;
    A(3, 1) = 1;
    A(3, 2) = 3;

    FusedTensorND<T, 3, 4> B;
    B(0, 0) = 1;
    B(0, 1) = 0;
    B(0, 2) = 2;
    B(0, 3) = 1;
    B(1, 0) = 0;
    B(1, 1) = 1;
    B(1, 2) = 1;
    B(1, 3) = 0;
    B(2, 0) = 2;
    B(2, 1) = 1;
    B(2, 2) = 0;
    B(2, 3) = 1;

    FusedTensorND<T, 4, 4> C;
    C.setToZero();

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;
    using LayoutC = typename decltype(C)::Layout;

    Gemm::gemm(
        A.data(), 4, 3, LayoutA::stride(0),
        B.data(), 4, LayoutB::stride(0),
        C.data(), LayoutC::stride(0));

    // C = A * B (manual computation):
    // C[0,:] = [1*1+2*0+3*2, 1*0+2*1+3*1, 1*2+2*1+3*0, 1*1+2*0+3*1] = [7, 5, 4, 4]
    // C[1,:] = [4*1+5*0+6*2, 4*0+5*1+6*1, 4*2+5*1+6*0, 4*1+5*0+6*1] = [16, 11, 13, 10]
    // C[2,:] = [7*1+8*0+9*2, 7*0+8*1+9*1, 7*2+8*1+9*0, 7*1+8*0+9*1] = [25, 17, 22, 16]
    // C[3,:] = [2*1+1*0+3*2, 2*0+1*1+3*1, 2*2+1*1+3*0, 2*1+1*0+3*1] = [8, 4, 5, 5]
    REQUIRE(C(0, 0) == Approx(7));
    REQUIRE(C(0, 1) == Approx(5));
    REQUIRE(C(0, 2) == Approx(4));
    REQUIRE(C(0, 3) == Approx(4));
    REQUIRE(C(1, 0) == Approx(16));
    REQUIRE(C(1, 1) == Approx(11));
    REQUIRE(C(1, 2) == Approx(13));
    REQUIRE(C(1, 3) == Approx(10));
    REQUIRE(C(2, 0) == Approx(25));
    REQUIRE(C(2, 1) == Approx(17));
    REQUIRE(C(2, 2) == Approx(22));
    REQUIRE(C(2, 3) == Approx(16));
    REQUIRE(C(3, 0) == Approx(8));
    REQUIRE(C(3, 1) == Approx(4));
    REQUIRE(C(3, 2) == Approx(5));
    REQUIRE(C(3, 3) == Approx(5));
}

TEMPLATE_TEST_CASE("gemm direct: 5x7 * 7x5 — remainder rows AND columns", "[gemm][direct][remainder][test_kernel_gemm]", double, float)
{
    // M=5: 4 main + 1 remainder row (MR=4)
    // N=5: padded to 8, but N=5 means:
    //   wide_N=0 (5<NR=12), narrow_N=4 (one simdWidth chunk), scalar cols 4
    //   Exercises: micro_kernel_narrow + scalar_column_MR + single_row_narrow + scalar corner
    using T = TestType;
    using Gemm = detail::KernelGemm<T, BITS, DefaultArch>;

    FusedTensorND<T, 5, 7> A;
    FusedTensorND<T, 7, 5> B;
    FusedTensorND<T, 5, 5> C;
    FusedTensorND<T, 5, 5> C_ref;

    A.setSequencial();
    B.setSequencial();

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;
    using LayoutC = typename decltype(C)::Layout;

    C.setToZero();
    Gemm::gemm(
        A.data(), 5, 7, LayoutA::stride(0),
        B.data(), 5, LayoutB::stride(0),
        C.data(), LayoutC::stride(0));

    // Reference
    naive_gemm(
        A.data(), 5, 7, LayoutA::stride(0),
        B.data(), 5, LayoutB::stride(0),
        C_ref.data(), LayoutC::stride(0));

    for (my_size_t i = 0; i < 5; ++i)
        for (my_size_t j = 0; j < 5; ++j)
            REQUIRE(C(i, j) == Approx(C_ref(i, j)));
}

TEMPLATE_TEST_CASE("gemm direct: 7x13 * 13x11 — exercises all tile paths", "[gemm][direct][remainder][test_kernel_gemm]", double, float)
{
    // M=7: 4 main + 3 remainder rows
    // N=11: For doubles (simdWidth=4, NR=12): wide_N=0, narrow_N=8, scalar=3
    //        For floats (simdWidth=8, NR=24): wide_N=0, narrow_N=8, scalar=3
    //   Exercises multiple narrow tiles + scalar remainder
    using T = TestType;
    using Gemm = detail::KernelGemm<T, BITS, DefaultArch>;

    FusedTensorND<T, 7, 13> A;
    FusedTensorND<T, 13, 11> B;
    FusedTensorND<T, 7, 11> C;
    FusedTensorND<T, 7, 11> C_ref;

    A.setSequencial();
    B.setSequencial();

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;
    using LayoutC = typename decltype(C)::Layout;

    C.setToZero();
    Gemm::gemm(
        A.data(), 7, 13, LayoutA::stride(0),
        B.data(), 11, LayoutB::stride(0),
        C.data(), LayoutC::stride(0));

    naive_gemm(
        A.data(), 7, 13, LayoutA::stride(0),
        B.data(), 11, LayoutB::stride(0),
        C_ref.data(), LayoutC::stride(0));

    for (my_size_t i = 0; i < 7; ++i)
        for (my_size_t j = 0; j < 11; ++j)
            REQUIRE(C(i, j) == Approx(C_ref(i, j)));
}

TEMPLATE_TEST_CASE("gemm direct: 1x8 * 8x1 — single row × single column = scalar", "[gemm][direct][degenerate][test_kernel_gemm]", double, float)
{
    using T = TestType;
    using Gemm = detail::KernelGemm<T, BITS, DefaultArch>;

    FusedTensorND<T, 1, 8> A;
    FusedTensorND<T, 8, 1> B;
    FusedTensorND<T, 1, 1> C;

    A.setSequencial();
    B.setSequencial();

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;
    using LayoutC = typename decltype(C)::Layout;

    C.setToZero();
    Gemm::gemm(
        A.data(), 1, 8, LayoutA::stride(0),
        B.data(), 1, LayoutB::stride(0),
        C.data(), LayoutC::stride(0));

    // [0,1,2,3,4,5,6,7] dot [0,1,2,3,4,5,6,7] = 0+1+4+9+16+25+36+49 = 140
    REQUIRE(C(0, 0) == Approx(140.0));
}

TEMPLATE_TEST_CASE("gemm direct: 1x1 * 1x1 — minimal", "[gemm][direct][degenerate][test_kernel_gemm]", double, float)
{
    using T = TestType;
    using Gemm = detail::KernelGemm<T, BITS, DefaultArch>;

    FusedTensorND<T, 1, 1> A;
    FusedTensorND<T, 1, 1> B;
    FusedTensorND<T, 1, 1> C;

    A(0, 0) = 7.0;
    B(0, 0) = 3.0;

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;
    using LayoutC = typename decltype(C)::Layout;

    C.setToZero();
    Gemm::gemm(
        A.data(), 1, 1, LayoutA::stride(0),
        B.data(), 1, LayoutB::stride(0),
        C.data(), LayoutC::stride(0));

    REQUIRE(C(0, 0) == Approx(21.0));
}

TEMPLATE_TEST_CASE("gemm direct: 8x4 * 4x16 — exact MR fit, wide tiles only", "[gemm][direct][wide][test_kernel_gemm]", double, float)
{
    // M=8: exactly 2 MR blocks, no remainder rows
    // N=16: for doubles NR=12 → wide_N=12 + narrow_N=16, scalar=0
    //        for floats NR=24 → wide_N=0, narrow_N=16, scalar=0
    using T = TestType;
    using Gemm = detail::KernelGemm<T, BITS, DefaultArch>;

    FusedTensorND<T, 8, 4> A;
    FusedTensorND<T, 4, 16> B;
    FusedTensorND<T, 8, 16> C;
    FusedTensorND<T, 8, 16> C_ref;

    A.setSequencial();
    B.setSequencial();

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;
    using LayoutC = typename decltype(C)::Layout;

    C.setToZero();
    Gemm::gemm(
        A.data(), 8, 4, LayoutA::stride(0),
        B.data(), 16, LayoutB::stride(0),
        C.data(), LayoutC::stride(0));

    naive_gemm(
        A.data(), 8, 4, LayoutA::stride(0),
        B.data(), 16, LayoutB::stride(0),
        C_ref.data(), LayoutC::stride(0));

    for (my_size_t i = 0; i < 8; ++i)
        for (my_size_t j = 0; j < 16; ++j)
            REQUIRE(C(i, j) == Approx(C_ref(i, j)));
}

TEMPLATE_TEST_CASE("gemm direct: 100x100 — performance case vs naive", "[gemm][direct][large][test_kernel_gemm]", double, float)
{
    using T = TestType;
    using Gemm = detail::KernelGemm<T, BITS, DefaultArch>;

    FusedTensorND<T, 100, 100> A;
    FusedTensorND<T, 100, 100> B;
    FusedTensorND<T, 100, 100> C;
    FusedTensorND<T, 100, 100> C_ref;

    A.setSequencial();
    B.setSequencial();

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;
    using LayoutC = typename decltype(C)::Layout;

    C.setToZero();
    Gemm::gemm(
        A.data(), 100, 100, LayoutA::stride(0),
        B.data(), 100, LayoutB::stride(0),
        C.data(), LayoutC::stride(0));

    naive_gemm(
        A.data(), 100, 100, LayoutA::stride(0),
        B.data(), 100, LayoutB::stride(0),
        C_ref.data(), LayoutC::stride(0));

    for (my_size_t i = 0; i < 100; ++i)
        for (my_size_t j = 0; j < 100; ++j)
            REQUIRE(C(i, j) == Approx(C_ref(i, j)).epsilon(1e-6));
}

// ============================================================================
// EINSUM GEMM DISPATCH — all 4 axis combinations
// ============================================================================
//
// einsum(A, B, a, b) contracts axis a of A with axis b of B.
// The 2D GEMM path handles all combos via optional transpose.

TEMPLATE_TEST_CASE("einsum gemm: a=1,b=0 — favorable, no transpose", "[gemm][einsum][test_kernel_gemm]", double, float)
{
    // C[M,N] = A[M,K] × B[K,N] — standard matmul
    using T = TestType;

    FusedTensorND<T, 2, 3> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    FusedTensorND<T, 3, 2> B;
    B(0, 0) = 7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = 10;
    B(2, 0) = 11;
    B(2, 1) = 12;

    // C = [[1*7+2*9+3*11, 1*8+2*10+3*12],
    //      [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //   = [[58, 64], [139, 154]]
    auto C = FusedTensorND<T, 2, 2>::einsum(A, B, 1, 0);

    REQUIRE(C(0, 0) == Approx(58));
    REQUIRE(C(0, 1) == Approx(64));
    REQUIRE(C(1, 0) == Approx(139));
    REQUIRE(C(1, 1) == Approx(154));
}

TEMPLATE_TEST_CASE("einsum gemm: a=0,b=0 — transpose A", "[gemm][einsum][test_kernel_gemm]", double, float)
{
    // C[M,N] = A^T[K,M] × B[K,N] → contract first dim of both
    // A[3,2], contract axis 0 (K=3), free axis 1 (M=2)
    // B[3,4], contract axis 0 (K=3), free axis 1 (N=4)
    // C[2,4]
    using T = TestType;

    FusedTensorND<T, 3, 2> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    A(2, 0) = 5;
    A(2, 1) = 6;

    FusedTensorND<T, 3, 4> B;
    B(0, 0) = 1;
    B(0, 1) = 2;
    B(0, 2) = 3;
    B(0, 3) = 4;
    B(1, 0) = 5;
    B(1, 1) = 6;
    B(1, 2) = 7;
    B(1, 3) = 8;
    B(2, 0) = 9;
    B(2, 1) = 10;
    B(2, 2) = 11;
    B(2, 3) = 12;

    auto C = FusedTensorND<T, 2, 4>::einsum(A, B, 0, 0);

    auto C_ref = naive_einsum<T, 2, 4>(A, B, 0, 0);

    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 4; ++j)
            REQUIRE(C(i, j) == Approx(C_ref(i, j)));
}

TEMPLATE_TEST_CASE("einsum gemm: a=1,b=1 — transpose B", "[gemm][einsum][test_kernel_gemm]", double, float)
{
    // A[2,3], contract axis 1 (K=3), free axis 0 (M=2)
    // B[4,3], contract axis 1 (K=3), free axis 0 (N=4)
    // C[2,4]
    using T = TestType;

    FusedTensorND<T, 2, 3> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    FusedTensorND<T, 4, 3> B;
    B(0, 0) = 1;
    B(0, 1) = 2;
    B(0, 2) = 3;
    B(1, 0) = 4;
    B(1, 1) = 5;
    B(1, 2) = 6;
    B(2, 0) = 7;
    B(2, 1) = 8;
    B(2, 2) = 9;
    B(3, 0) = 10;
    B(3, 1) = 11;
    B(3, 2) = 12;

    auto C = FusedTensorND<T, 2, 4>::einsum(A, B, 1, 1);

    auto C_ref = naive_einsum<T, 2, 4>(A, B, 1, 1);

    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 4; ++j)
            REQUIRE(C(i, j) == Approx(C_ref(i, j)));
}

TEMPLATE_TEST_CASE("einsum gemm: a=0,b=1 — transpose both", "[gemm][einsum][test_kernel_gemm]", double, float)
{
    // A[3,2], contract axis 0 (K=3), free axis 1 (M=2)
    // B[4,3], contract axis 1 (K=3), free axis 0 (N=4)
    // C[2,4]
    using T = TestType;

    FusedTensorND<T, 3, 2> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    A(2, 0) = 5;
    A(2, 1) = 6;

    FusedTensorND<T, 4, 3> B;
    B(0, 0) = 1;
    B(0, 1) = 2;
    B(0, 2) = 3;
    B(1, 0) = 4;
    B(1, 1) = 5;
    B(1, 2) = 6;
    B(2, 0) = 7;
    B(2, 1) = 8;
    B(2, 2) = 9;
    B(3, 0) = 10;
    B(3, 1) = 11;
    B(3, 2) = 12;

    auto C = FusedTensorND<T, 2, 4>::einsum(A, B, 0, 1);

    auto C_ref = naive_einsum<T, 2, 4>(A, B, 0, 1);

    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 4; ++j)
            REQUIRE(C(i, j) == Approx(C_ref(i, j)));
}

// ============================================================================
// EINSUM WITH TRANSPOSED INPUTS — PermutedViewConstExpr path
// ============================================================================
//
// When the caller passes a transpose_view(), the einsum's make_transposed
// lambda must handle PermutedViewConstExpr (calls .transpose() to get the
// base tensor back, zero-cost).

TEMPLATE_TEST_CASE("einsum gemm: transposed view input a=1,b=0", "[gemm][einsum][permuted][test_kernel_gemm]", double, float)
{
    using T = TestType;

    // Create A as [3,2], pass transpose_view [2,3] to einsum
    FusedTensorND<T, 3, 2> A_base;
    A_base(0, 0) = 1;
    A_base(0, 1) = 4;
    A_base(1, 0) = 2;
    A_base(1, 1) = 5;
    A_base(2, 0) = 3;
    A_base(2, 1) = 6;
    // transpose_view → logical [2,3]:
    //   row 0: [1, 2, 3]
    //   row 1: [4, 5, 6]

    FusedTensorND<T, 3, 2> B;
    B(0, 0) = 7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = 10;
    B(2, 0) = 11;
    B(2, 1) = 12;

    auto C = FusedTensorND<T, 2, 2>::einsum(
        A_base.template transpose_view<1, 0>(), B, 1, 0);

    // Same as [[1,2,3],[4,5,6]] × [[7,8],[9,10],[11,12]]
    // = [[58, 64], [139, 154]]
    REQUIRE(C(0, 0) == Approx(58));
    REQUIRE(C(0, 1) == Approx(64));
    REQUIRE(C(1, 0) == Approx(139));
    REQUIRE(C(1, 1) == Approx(154));
}

TEMPLATE_TEST_CASE("einsum gemm: transposed view input a=1,b=1", "[gemm][einsum][permuted][test_kernel_gemm]", double, float)
{
    using T = TestType;

    FusedTensorND<T, 2, 3> A;
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    // B_base[3,4], transpose_view → logical [4,3]
    FusedTensorND<T, 3, 4> B_base;
    B_base(0, 0) = 1;
    B_base(0, 1) = 4;
    B_base(0, 2) = 7;
    B_base(0, 3) = 10;
    B_base(1, 0) = 2;
    B_base(1, 1) = 5;
    B_base(1, 2) = 8;
    B_base(1, 3) = 11;
    B_base(2, 0) = 3;
    B_base(2, 1) = 6;
    B_base(2, 2) = 9;
    B_base(2, 3) = 12;
    // transpose_view → logical [4,3]:
    //   row 0: [1,2,3], row 1: [4,5,6], row 2: [7,8,9], row 3: [10,11,12]

    auto C = FusedTensorND<T, 2, 4>::einsum(
        A, B_base.template transpose_view<1, 0>(), 1, 1);

    // A[2,3] × B_t[4,3] with a=1,b=1: contract last dim of both
    // C[i,j] = sum_k A[i,k]*B_t[j,k]
    // C[0,0] = 1*1+2*2+3*3 = 14
    // C[0,1] = 1*4+2*5+3*6 = 32
    // C[0,2] = 1*7+2*8+3*9 = 50
    // C[0,3] = 1*10+2*11+3*12 = 68
    // C[1,0] = 4*1+5*2+6*3 = 32
    // C[1,1] = 4*4+5*5+6*6 = 77
    // C[1,2] = 4*7+5*8+6*9 = 122
    // C[1,3] = 4*10+5*11+6*12 = 167
    REQUIRE(C(0, 0) == Approx(14));
    REQUIRE(C(0, 1) == Approx(32));
    REQUIRE(C(0, 2) == Approx(50));
    REQUIRE(C(0, 3) == Approx(68));
    REQUIRE(C(1, 0) == Approx(32));
    REQUIRE(C(1, 1) == Approx(77));
    REQUIRE(C(1, 2) == Approx(122));
    REQUIRE(C(1, 3) == Approx(167));
}

// ============================================================================
// EINSUM LARGER — remainder + padding + all axis combos
// ============================================================================

TEMPLATE_TEST_CASE("einsum gemm: 5x7 * 7x5, all axis combos vs naive", "[gemm][einsum][remainder][test_kernel_gemm]", double, float)
{
    using T = TestType;

    FusedTensorND<T, 5, 7> A;
    FusedTensorND<T, 7, 5> B;

    A.setSequencial();
    B.setSequencial();

    SECTION("a=1, b=0: C[5,5] = A[5,7] × B[7,5]")
    {
        auto C = FusedTensorND<T, 5, 5>::einsum(A, B, 1, 0);
        auto C_ref = naive_einsum<T, 5, 5>(A, B, 1, 0);
        for (my_size_t i = 0; i < 5; ++i)
            for (my_size_t j = 0; j < 5; ++j)
                REQUIRE(C(i, j) == Approx(C_ref(i, j)));
    }

    SECTION("a=0, b=0: C[7,3] = A^T[7,5] × B2[5,3]")
    {
        // a=0: contract A's dim 0 (size 5), b=0: contract B2's dim 0 (size 5) → K=5
        FusedTensorND<T, 5, 3> B2{};
        B2.setSequencial();

        // C[7,3] = A^T[7,5] × B2[5,3]
        auto C = FusedTensorND<T, 7, 3>::einsum(A, B2, 0, 0);
        auto C_ref = naive_einsum<T, 7, 3>(A, B2, 0, 0);
        for (my_size_t i = 0; i < 7; ++i)
            for (my_size_t j = 0; j < 3; ++j)
                REQUIRE(C(i, j) == Approx(C_ref(i, j)));
    }

    SECTION("a=1, b=1: C[5,3] = A[5,7] × B3^T[3,7]")
    {
        // a=1: contract A's dim 1 (size 7), b=1: contract B's dim 1 (size 5)
        // Need B3[3,7] with b=1 → K=7
        FusedTensorND<T, 3, 7> B3{};
        B3.setSequencial();

        // C[5,3] = A[5,7] × B3^T[3,7] → contract dim 1 of both
        auto C = FusedTensorND<T, 5, 3>::einsum(A, B3, 1, 1);
        auto C_ref = naive_einsum<T, 5, 3>(A, B3, 1, 1);
        for (my_size_t i = 0; i < 5; ++i)
            for (my_size_t j = 0; j < 3; ++j)
                REQUIRE(C(i, j) == Approx(C_ref(i, j)));
    }

    SECTION("a=0, b=1: C[7,3] = A^T[7,5] × B4^T[3,5]")
    {
        // a=0: contract A's dim 0 (size 5), b=1: contract B's dim 1 → need size 5
        FusedTensorND<T, 3, 5> B4{};
        B4.setSequencial();

        // C[7,3] = A^T[7,5] × B4^T[3,5] → contract dim 0 of A, dim 1 of B4
        auto C = FusedTensorND<T, 7, 3>::einsum(A, B4, 0, 1);
        auto C_ref = naive_einsum<T, 7, 3>(A, B4, 0, 1);
        for (my_size_t i = 0; i < 7; ++i)
            for (my_size_t j = 0; j < 3; ++j)
                REQUIRE(C(i, j) == Approx(C_ref(i, j)));
    }
}

// ============================================================================
// IDENTITY AND SPECIAL MATRICES
// ============================================================================

TEMPLATE_TEST_CASE("gemm: multiply by identity", "[gemm][einsum][identity][test_kernel_gemm]", double, float)
{
    using T = TestType;

    FusedTensorND<T, 4, 4> A;
    A.setSequencial();

    FusedTensorND<T, 4, 4> I;
    I.setIdentity();

    SECTION("A × I = A")
    {
        auto C = FusedTensorND<T, 4, 4>::einsum(A, I, 1, 0);
        for (my_size_t i = 0; i < 4; ++i)
            for (my_size_t j = 0; j < 4; ++j)
                REQUIRE(C(i, j) == Approx(A(i, j)));
    }

    SECTION("I × A = A")
    {
        auto C = FusedTensorND<T, 4, 4>::einsum(I, A, 1, 0);
        for (my_size_t i = 0; i < 4; ++i)
            for (my_size_t j = 0; j < 4; ++j)
                REQUIRE(C(i, j) == Approx(A(i, j)));
    }
}

TEMPLATE_TEST_CASE("gemm: zero matrix", "[gemm][einsum][zero][test_kernel_gemm]", double, float)
{
    using T = TestType;

    FusedTensorND<T, 3, 5> A;
    A.setSequencial();

    FusedTensorND<T, 5, 4> Z;
    Z.setToZero();

    auto C = FusedTensorND<T, 3, 4>::einsum(A, Z, 1, 0);

    for (my_size_t i = 0; i < 3; ++i)
        for (my_size_t j = 0; j < 4; ++j)
            REQUIRE(C(i, j) == Approx(0.0));
}

// ============================================================================
// NEGATIVE VALUES AND NUMERICAL PROPERTIES
// ============================================================================

TEMPLATE_TEST_CASE("gemm: negative values", "[gemm][einsum][negative][test_kernel_gemm]", double, float)
{
    using T = TestType;

    FusedTensorND<T, 2, 3> A;
    A(0, 0) = -1;
    A(0, 1) = -2;
    A(0, 2) = -3;
    A(1, 0) = 4;
    A(1, 1) = -5;
    A(1, 2) = 6;

    FusedTensorND<T, 3, 2> B;
    B(0, 0) = -7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = -10;
    B(2, 0) = -11;
    B(2, 1) = 12;

    auto C = FusedTensorND<T, 2, 2>::einsum(A, B, 1, 0);

    // C[0,0] = (-1)(-7)+(-2)(9)+(-3)(-11) = 7-18+33 = 22
    // C[0,1] = (-1)(8)+(-2)(-10)+(-3)(12) = -8+20-36 = -24
    // C[1,0] = (4)(-7)+(-5)(9)+(6)(-11) = -28-45-66 = -139
    // C[1,1] = (4)(8)+(-5)(-10)+(6)(12) = 32+50+72 = 154
    REQUIRE(C(0, 0) == Approx(22));
    REQUIRE(C(0, 1) == Approx(-24));
    REQUIRE(C(1, 0) == Approx(-139));
    REQUIRE(C(1, 1) == Approx(154));
}