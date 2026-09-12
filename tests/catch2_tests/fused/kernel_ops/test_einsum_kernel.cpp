/**
 * @file test_einsum_kernel.cpp
 * @brief Catch2 tests for KernelOps::dot, dot_contiguous_impl, dot_strided_impl
 *
 * All tests go through the public dot() interface. Dispatch is controlled
 * by stride arguments:
 *
 *   stride1=1, stride2=1   → dot_contiguous_impl (K::load)
 *   stride1>1 or stride2>1 → dot_strided_impl    (K::gather)
 *
 * ============================================================================
 * PHYSICAL LAYOUT REFERENCE
 * ============================================================================
 *
 * FusedTensorND<T, 2, 6> with SimdWidth=4, padded to [2, 8]:
 *
 *   Physical buffer (16 slots):
 *     slice 0: [1  2  3  4  5  6  0  0]   offsets 0-7
 *     slice 1: [7  8  9  10 11 12 0  0]   offsets 8-15
 *                                 ^  ^
 *                                padding
 *     BaseStrides = [8, 1]
 *
 *   Fiber along last dim (contiguous, stride=1):
 *     A[0,:] → base=0,  stride=1, len=6  → [1, 2, 3, 4, 5, 6]
 *     A[1,:] → base=8,  stride=1, len=6  → [7, 8, 9, 10, 11, 12]
 *
 *   Fiber along first dim (strided, stride=8):
 *     A[:,0] → base=0,  stride=8, len=2  → [1, 7]
 *     A[:,1] → base=1,  stride=8, len=2  → [2, 8]
 *
 * ============================================================================
 */

#include <catch_amalgamated.hpp>

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/kernel_ops/kernel_ops.h"
#include "fused/fused_tensor.h"

using Catch::Approx;

// // ============================================================================
// // HELPERS
// // ============================================================================

// /// Naive scalar dot for reference
// template <typename T>
// static T naive_dot_physical(
//     const T *data1, my_size_t base1, my_size_t stride1,
//     const T *data2, my_size_t base2, my_size_t stride2,
//     my_size_t len)
// {
//     T sum = T{0};
//     for (my_size_t i = 0; i < len; ++i)
//         sum += data1[base1 + i * stride1] * data2[base2 + i * stride2];
//     return sum;
// }

// ============================================================================
// CONTIGUOUS DOT — both strides = 1 → dispatches to dot_contiguous_impl
// ============================================================================
//
// Both fibers along last dim. K::load path, SIMD + scalar remainder.

TEMPLATE_TEST_CASE("dot contiguous: basic 2x6 row dot product", "[dot][contiguous][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;

    // A[2,6] sequential: [[1,2,3,4,5,6], [7,8,9,10,11,12]]
    // Padded to [2,8]: [[1,2,3,4,5,6,0,0], [7,8,9,10,11,12,0,0]]
    FusedTensorND<T, 2, 6> A;
    A.setSequencial(); // A[i] = i in logical order

    using Layout = typename decltype(A)::Layout;
    static constexpr my_size_t stride0 = Layout::stride(0); // paddedLastDim = 8
    static constexpr my_size_t stride1 = Layout::stride(1); // 1

    SECTION("A[0,:] dot A[0,:] — self dot product of first row")
    {
        // row 0 logical: [0, 1, 2, 3, 4, 5]
        // sum = 0*0 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 0+1+4+9+16+25 = 55
        T result = Kernel::dot(A, 0, stride1, A, 0, stride1, 6);
        REQUIRE(result == Approx(55.0));
    }

    SECTION("A[0,:] dot A[1,:] — dot product of two different rows")
    {
        // row 0: [0, 1, 2, 3, 4, 5], row 1: [6, 7, 8, 9, 10, 11]
        // sum = 0*6 + 1*7 + 2*8 + 3*9 + 4*10 + 5*11 = 0+7+16+27+40+55 = 145
        T result = Kernel::dot(A, 0, stride1, A, stride0, stride1, 6);
        REQUIRE(result == Approx(145.0));
    }

    SECTION("result matches naive scalar computation")
    {
        T simd_result = Kernel::dot(A, 0, stride1, A, stride0, stride1, 6);
        T naive_result = Kernel::naive_dot_physical(A, 0, 1, A, stride0, 1, 6);
        REQUIRE(simd_result == Approx(naive_result));
    }
}

TEMPLATE_TEST_CASE("dot contiguous: len is exact multiple of simdWidth", "[dot][contiguous][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    // 2x8 → padded last dim is 8 (already multiple of 4)
    // No scalar remainder expected
    FusedTensorND<T, 2, 8> A;
    A.setSequencial();

    // row 0: [0,1,2,3,4,5,6,7]
    // self dot = 0+1+4+9+16+25+36+49 = 140
    T result = Kernel::dot(A, 0, 1, A, 0, 1, 8);
    REQUIRE(result == Approx(140.0));
}

TEMPLATE_TEST_CASE("dot contiguous: len smaller than simdWidth", "[dot][contiguous][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    // 2x2 → padded to [2,4], lastDim=2
    // With SimdWidth=4: simdSteps=0, entirely scalar remainder
    FusedTensorND<T, 2, 2> A;
    A.setSequencial();

    using Layout = typename decltype(A)::Layout;
    static constexpr my_size_t stride0 = Layout::stride(0); // paddedLastDim = 4

    SECTION("A[0,:] dot A[0,:] — 2 elements, all scalar")
    {
        // row 0: [0, 1]
        // sum = 0*0 + 1*1 = 1
        T result = Kernel::dot(A, 0, 1, A, 0, 1, 2);
        REQUIRE(result == Approx(1.0));
    }

    SECTION("A[0,:] dot A[1,:] — cross rows, 2 elements")
    {
        // row 0: [0, 1], row 1: [2, 3]
        // sum = 0*2 + 1*3 = 3
        T result = Kernel::dot(A, 0, 1, A, stride0, 1, 2);
        REQUIRE(result == Approx(3.0));
    }
}

TEMPLATE_TEST_CASE("dot contiguous: len = 1", "[dot][contiguous][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    FusedTensorND<T, 1, 1> A;
    A(0, 0) = 5.0;

    FusedTensorND<T, 1, 1> B;
    B(0, 0) = 3.0;

    T result = Kernel::dot(A, 0, 1, B, 0, 1, 1);
    REQUIRE(result == Approx(15.0));
}

// ============================================================================
// STRIDED DOT — one or both strides > 1 → dispatches to dot_strided_impl
// ============================================================================
//
// Fiber along non-last dim. Elements are spaced by paddedLastDim in memory.

TEMPLATE_TEST_CASE("dot strided: fibers along first dim", "[dot][strided][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    // A[3,4] sequential, padded to [3,4] (4 already multiple of 4)
    //   row 0: [ 0,  1,  2,  3]
    //   row 1: [ 4,  5,  6,  7]
    //   row 2: [ 8,  9, 10, 11]
    FusedTensorND<T, 3, 4> A;
    A.setSequencial();

    using Layout = typename decltype(A)::Layout;
    static constexpr my_size_t stride0 = Layout::stride(0); // paddedLastDim = 4

    SECTION("A[:,0] dot A[:,0] — column 0 self-dot")
    {
        // column 0: [0, 4, 8], stride=4
        // sum = 0*0 + 4*4 + 8*8 = 0+16+64 = 80
        T result = Kernel::dot(A, 0, stride0, A, 0, stride0, 3);
        REQUIRE(result == Approx(80.0));
    }

    SECTION("A[:,0] dot A[:,1] — column 0 dot column 1")
    {
        // col 0: [0, 4, 8], col 1: [1, 5, 9], both stride=4
        // sum = 0*1 + 4*5 + 8*9 = 0+20+72 = 92
        T result = Kernel::dot(A, 0, stride0, A, 1, stride0, 3);
        REQUIRE(result == Approx(92.0));
    }

    SECTION("result matches naive scalar computation")
    {
        T simd_result = Kernel::dot(A, 0, stride0, A, 1, stride0, 3);
        T naive_result = Kernel::naive_dot_physical(
            A, 0, stride0,
            A, 1, stride0, 3);
        REQUIRE(simd_result == Approx(naive_result));
    }
}

TEMPLATE_TEST_CASE("dot strided: len larger than simdWidth", "[dot][strided][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    // A[8,2] → padded to [8,4]
    //   row 0: [0, 1, P, P]
    //   row 1: [2, 3, P, P]
    //   ...
    //   row 7: [14, 15, P, P]
    FusedTensorND<T, 8, 2> A;
    A.setSequencial();

    using Layout = typename decltype(A)::Layout;
    static constexpr my_size_t stride0 = Layout::stride(0); // paddedLastDim = 4

    SECTION("A[:,0] dot A[:,1] — 8 elements strided")
    {
        // col 0: [0, 2, 4, 6, 8, 10, 12, 14] at offsets 0, 4, 8, 12, 16, 20, 24, 28
        // col 1: [1, 3, 5, 7, 9, 11, 13, 15] at offsets 1, 5, 9, 13, 17, 21, 25, 29
        // sum = 0*1 + 2*3 + 4*5 + 6*7 + 8*9 + 10*11 + 12*13 + 14*15
        //     = 0 + 6 + 20 + 42 + 72 + 110 + 156 + 210 = 616
        T result = Kernel::dot(A, 0, stride0, A, 1, stride0, 8);
        REQUIRE(result == Approx(616.0));
    }

    SECTION("A[:,0] self-dot — 8 elements strided")
    {
        // col 0: [0, 2, 4, 6, 8, 10, 12, 14]
        // sum = 0+4+16+36+64+100+144+196 = 560
        T result = Kernel::dot(A, 0, stride0, A, 0, stride0, 8);
        REQUIRE(result == Approx(560.0));
    }
}

// ============================================================================
// MIXED DOT — one contiguous, one strided
// ============================================================================
//
// Typical in matmul: A[i,:] (contiguous) dot B[:,j] (strided)

TEMPLATE_TEST_CASE("dot mixed: contiguous x strided (matmul pattern)", "[dot][mixed][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    // A[2,3] sequential, padded to [2,4]:
    //   row 0: [0, 1, 2, P]   offsets 0,1,2,3
    //   row 1: [3, 4, 5, P]   offsets 4,5,6,7
    FusedTensorND<T, 2, 3> A;
    A.setSequencial();

    // B[3,2] with known values, padded to [3,4]:
    //   row 0: [1, 4, P, P]   offsets 0,1,2,3
    //   row 1: [2, 5, P, P]   offsets 4,5,6,7
    //   row 2: [3, 6, P, P]   offsets 8,9,10,11
    FusedTensorND<T, 3, 2> B;
    B(0, 0) = 1;
    B(0, 1) = 4;
    B(1, 0) = 2;
    B(1, 1) = 5;
    B(2, 0) = 3;
    B(2, 1) = 6;

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;

    static constexpr my_size_t A_stride0 = LayoutA::stride(0); // 4
    static constexpr my_size_t B_stride0 = LayoutB::stride(0); // 4

    SECTION("A[0,:] dot B[:,0] — row 0 of A dot column 0 of B")
    {
        // A[0,:] = [0, 1, 2] at base=0, stride=1
        // B[:,0] = [1, 2, 3] at base=0, stride=4
        // sum = 0*1 + 1*2 + 2*3 = 0+2+6 = 8
        T result = Kernel::dot(A, 0, 1, B, 0, B_stride0, 3);
        REQUIRE(result == Approx(8.0));
    }

    SECTION("A[0,:] dot B[:,1] — row 0 of A dot column 1 of B")
    {
        // A[0,:] = [0, 1, 2] at base=0, stride=1
        // B[:,1] = [4, 5, 6] at base=1, stride=4
        // sum = 0*4 + 1*5 + 2*6 = 0+5+12 = 17
        T result = Kernel::dot(A, 0, 1, B, 1, B_stride0, 3);
        REQUIRE(result == Approx(17.0));
    }

    SECTION("A[1,:] dot B[:,0] — row 1 of A dot column 0 of B")
    {
        // A[1,:] = [3, 4, 5] at base=A_stride0, stride=1
        // B[:,0] = [1, 2, 3] at base=0, stride=4
        // sum = 3*1 + 4*2 + 5*3 = 3+8+15 = 26
        T result = Kernel::dot(A, A_stride0, 1, B, 0, B_stride0, 3);
        REQUIRE(result == Approx(26.0));
    }

    SECTION("A[1,:] dot B[:,1] — row 1 of A dot column 1 of B")
    {
        // A[1,:] = [3, 4, 5] at base=A_stride0, stride=1
        // B[:,1] = [4, 5, 6] at base=1, stride=4
        // sum = 3*4 + 4*5 + 5*6 = 12+20+30 = 62
        T result = Kernel::dot(A, A_stride0, 1, B, 1, B_stride0, 3);
        REQUIRE(result == Approx(62.0));
    }

    SECTION("full matmul C = A * B matches expected")
    {
        // C[2,2] = A[2,3] * B[3,2]
        // C = [[0*1+1*2+2*3, 0*4+1*5+2*6],
        //      [3*1+4*2+5*3, 3*4+4*5+5*6]]
        //   = [[8, 17], [26, 62]]
        FusedTensorND<T, 2, 2> C;

        for (my_size_t i = 0; i < 2; ++i)
        {
            for (my_size_t j = 0; j < 2; ++j)
            {
                C(i, j) = Kernel::dot(
                    A, i * A_stride0, 1, // A[i,:] contiguous
                    B, j, B_stride0,     // B[:,j] strided
                    3);                  // contraction length K=3
            }
        }

        REQUIRE(C(0, 0) == Approx(8.0));
        REQUIRE(C(0, 1) == Approx(17.0));
        REQUIRE(C(1, 0) == Approx(26.0));
        REQUIRE(C(1, 1) == Approx(62.0));
    }
}

// ============================================================================
// 3D TENSOR — verify fibers and strides work with higher dimensions
// ============================================================================

TEMPLATE_TEST_CASE("dot with 3D tensor fibers", "[dot][3d][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    // A[2,3,4] sequential, padded to [2,3,4] (4 already aligned)
    //   slice[0,0,:] = [0,  1,  2,  3]    offset 0
    //   slice[0,1,:] = [4,  5,  6,  7]    offset 4
    //   slice[0,2,:] = [8,  9,  10, 11]   offset 8
    //   slice[1,0,:] = [12, 13, 14, 15]   offset 12
    //   slice[1,1,:] = [16, 17, 18, 19]   offset 16
    //   slice[1,2,:] = [20, 21, 22, 23]   offset 20
    FusedTensorND<T, 2, 3, 4> A;
    A.setSequencial();

    using Layout = typename decltype(A)::Layout;
    // BaseStrides for [2,3,4] padded to [2,3,4]: [12, 4, 1]
    static constexpr my_size_t s0 = Layout::stride(0); // 12
    static constexpr my_size_t s1 = Layout::stride(1); // 4
    static constexpr my_size_t s2 = Layout::stride(2); // 1

    SECTION("contiguous fiber along last dim: A[0,0,:] dot A[0,1,:]")
    {
        // A[0,0,:] = [0,1,2,3] at base=0, stride=1
        // A[0,1,:] = [4,5,6,7] at base=4, stride=1
        // sum = 0*4 + 1*5 + 2*6 + 3*7 = 0+5+12+21 = 38
        T result = Kernel::dot(A, 0, s2, A, s1, s2, 4);
        REQUIRE(result == Approx(38.0));
    }

    SECTION("strided fiber along middle dim: A[0,:,0] dot A[1,:,0]")
    {
        // A[0,:,0] = [0, 4, 8] at base=0, stride=4
        // A[1,:,0] = [12, 16, 20] at base=12, stride=4
        // sum = 0*12 + 4*16 + 8*20 = 0+64+160 = 224
        T result = Kernel::dot(A, 0, s1, A, s0, s1, 3);
        REQUIRE(result == Approx(224.0));
    }

    SECTION("strided fiber along first dim: A[:,0,0] dot A[:,1,0]")
    {
        // A[:,0,0] = [0, 12] at base=0, stride=12
        // A[:,1,0] = [4, 16] at base=4, stride=12
        // sum = 0*4 + 12*16 = 0+192 = 192
        T result = Kernel::dot(A, 0, s0, A, s1, s0, 2);
        REQUIRE(result == Approx(192.0));
    }
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEMPLATE_TEST_CASE("dot contiguous: zero vector", "[dot][edge][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    FusedTensorND<T, 1, 4> A;
    A.setToZero();

    FusedTensorND<T, 1, 4> B;
    B.setSequencial();

    T result = Kernel::dot(A, 0, 1, B, 0, 1, 4);
    REQUIRE(result == Approx(0.0));
}

TEMPLATE_TEST_CASE("dot contiguous: ones vector self-dot equals length", "[dot][edge][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    FusedTensorND<T, 1, 6> A;
    A.setHomogen(1.0);

    T result = Kernel::dot(A, 0, 1, A, 0, 1, 6);
    REQUIRE(result == Approx(6.0));
}

TEMPLATE_TEST_CASE("dot: negative values", "[dot][edge][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    FusedTensorND<T, 1, 4> A;
    A(0, 0) = -1.0;
    A(0, 1) = -2.0;
    A(0, 2) = -3.0;
    A(0, 3) = -4.0;

    FusedTensorND<T, 1, 4> B;
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(0, 2) = 3.0;
    B(0, 3) = 4.0;

    // sum = -1*1 + -2*2 + -3*3 + -4*4 = -1-4-9-16 = -30
    T result = Kernel::dot(A, 0, 1, B, 0, 1, 4);
    REQUIRE(result == Approx(-30.0));
}

TEMPLATE_TEST_CASE("dot: partial fiber (len < actual row length)", "[dot][edge][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    // A[1,8] sequential: [0, 1, 2, 3, 4, 5, 6, 7]
    FusedTensorND<T, 1, 8> A;
    A.setSequencial();

    SECTION("dot only first 3 elements")
    {
        // [0,1,2] dot [0,1,2] = 0+1+4 = 5
        T result = Kernel::dot(A, 0, 1, A, 0, 1, 3);
        REQUIRE(result == Approx(5.0));
    }

    SECTION("dot only first 1 element")
    {
        T result = Kernel::dot(A, 0, 1, A, 0, 1, 1);
        REQUIRE(result == Approx(0.0)); // 0*0 = 0
    }

    SECTION("dot starting from offset")
    {
        // Starting at offset 2: [2,3,4] dot [2,3,4] = 4+9+16 = 29
        T result = Kernel::dot(A, 2, 1, A, 2, 1, 3);
        REQUIRE(result == Approx(29.0));
    }
}

TEMPLATE_TEST_CASE("dot strided: stride equals 1 still dispatches contiguous", "[dot][dispatch][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    // Verify that stride1=1, stride2=1 gives same result whether
    // it goes through contiguous or strided path
    FusedTensorND<T, 2, 6> A;
    A.setSequencial();

    // Both stride=1 → contiguous path
    T result_s1 = Kernel::dot(A, 0, 1, A, 0, 1, 6);

    // Sanity: row 0 = [0,1,2,3,4,5], self-dot = 55
    REQUIRE(result_s1 == Approx(55.0));
}

// ============================================================================
// CROSS-TENSOR DOT — two different tensors
// ============================================================================

TEMPLATE_TEST_CASE("dot between two different tensors", "[dot][cross][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    FusedTensorND<T, 3, 5> A;
    FusedTensorND<T, 3, 5> B;

    // Fill with known non-sequential values
    for (my_size_t i = 0; i < 3; ++i)
        for (my_size_t j = 0; j < 5; ++j)
        {
            A(i, j) = static_cast<T>(i * 5 + j + 1);       // 1..15
            B(i, j) = static_cast<T>((i * 5 + j + 1) * 2); // 2..30
        }

    using LayoutA = typename decltype(A)::Layout;
    using LayoutB = typename decltype(B)::Layout;
    static constexpr my_size_t A_s0 = LayoutA::stride(0);
    static constexpr my_size_t B_s0 = LayoutB::stride(0);

    SECTION("contiguous: A[0,:] dot B[0,:] — both row 0")
    {
        // A[0,:] = [1,2,3,4,5], B[0,:] = [2,4,6,8,10]
        // sum = 1*2 + 2*4 + 3*6 + 4*8 + 5*10 = 2+8+18+32+50 = 110
        T result = Kernel::dot(A, 0, 1, B, 0, 1, 5);
        REQUIRE(result == Approx(110.0));
    }

    SECTION("strided: A[:,0] dot B[:,0] — both column 0")
    {
        // A[:,0] = [1, 6, 11] stride=A_s0
        // B[:,0] = [2, 12, 22] stride=B_s0
        // sum = 1*2 + 6*12 + 11*22 = 2+72+242 = 316
        T result = Kernel::dot(A, 0, A_s0, B, 0, B_s0, 3);
        REQUIRE(result == Approx(316.0));
    }

    SECTION("mixed: A[1,:] dot B[:,2] — row x column")
    {
        // A[1,:] = [6,7,8,9,10] at base=A_s0, stride=1
        // B[:,2] = [6, 16, 26] at base=2, stride=B_s0 → only 3 elements
        // But contraction length must match! This tests len=3 on 5-element row
        // A[1,0:3] = [6,7,8], B[:,2] = [6,16,26]
        // sum = 6*6 + 7*16 + 8*26 = 36+112+208 = 356
        T result = Kernel::dot(A, A_s0, 1, B, 2, B_s0, 3);
        REQUIRE(result == Approx(356.0));
    }
}

// ============================================================================
// COMMUTATIVITY
// ============================================================================

TEMPLATE_TEST_CASE("dot is commutative", "[dot][property][test_einsum_kernel]", double, float)
{
    using T = TestType;
    using Kernel = KernelOps<T, BITS, DefaultArch>;
    FusedTensorND<T, 2, 6> A;
    FusedTensorND<T, 2, 6> B;
    A.setSequencial();
    for (my_size_t i = 0; i < 2; ++i)
        for (my_size_t j = 0; j < 6; ++j)
            B(i, j) = static_cast<T>((i * 6 + j) * 3 + 1);

    using Layout = typename decltype(A)::Layout;
    static constexpr my_size_t s0 = Layout::stride(0);

    SECTION("contiguous: dot(A,B) == dot(B,A)")
    {
        T ab = Kernel::dot(A, 0, 1, B, 0, 1, 6);
        T ba = Kernel::dot(B, 0, 1, A, 0, 1, 6);
        REQUIRE(ab == Approx(ba));
    }

    SECTION("strided: dot(A,B) == dot(B,A)")
    {
        T ab = Kernel::dot(A, 0, s0, B, 0, s0, 2);
        T ba = Kernel::dot(B, 0, s0, A, 0, s0, 2);
        REQUIRE(ab == Approx(ba));
    }

    SECTION("mixed: dot(A_row, B_col) == dot(B_col, A_row)")
    {
        T ab = Kernel::dot(A, 0, 1, B, 0, s0, 2);
        T ba = Kernel::dot(B, 0, s0, A, 0, 1, 2);
        REQUIRE(ab == Approx(ba));
    }
}
