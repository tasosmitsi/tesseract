/**
 * @file kernel_gemm.h
 * @brief Register-blocked GEMM for optimized 2D matrix multiplication.
 *
 * Implements C[M,N] = A[M,K] × B[K,N] using an outer-product micro-kernel
 * formulation that maximizes register reuse and SIMD throughput.
 *
 * @section algorithm Algorithm Overview
 *
 * The naive approach computes each C[i,j] as an independent dot product:
 * one row of A dotted with one column of B. For M=N=100, that's 10,000
 * dot products, each reloading the same data — terrible cache utilization.
 *
 * This GEMM instead computes an MR × NR tile of C simultaneously using
 * the outer-product formulation:
 *
 *   For each k in 0..K-1:
 *       Load NR_VECS vectors from B's row k           (NR_VECS SIMD loads)
 *       For each of MR rows of A:
 *           Broadcast A[row, k] to a SIMD register     (1 broadcast)
 *           FMA broadcast × each B vector into accumulators  (NR_VECS FMAs)
 *
 * Key insight: each B vector is loaded once and reused across MR rows of A.
 * Each A broadcast is loaded once and reused across NR_VECS B vectors.
 * This gives MR × NR_VECS × simdWidth multiply-adds per k step from only
 * NR_VECS loads + MR broadcasts.
 *
 * @section register_model Register Pressure Model
 *
 * The micro-kernel holds all accumulators in registers for the entire
 * k-loop, storing to memory only once at the end. The register budget is:
 *
 *   MR × NR_VECS   accumulators (one SIMD register each)
 *   NR_VECS         B vectors (loaded each k step)
 *   1               A broadcast (reused across NR_VECS FMAs)
 *   ─────────────────────────────────────────────────────
 *   MR × NR_VECS + NR_VECS + 1  ≤  num_registers
 *
 * The optimal tile size saturates available registers without spilling.
 * MR, NR_VECS, and num_registers are defined per-architecture in the
 * Microkernel specializations (e.g., AVX2: 16 YMM → MR=4, NR_VECS=3).
 *
 * @section tiling Tiling Strategy
 *
 * The output matrix C is covered by nested loops:
 *
 *   i-loop (rows):    steps by MR, remainder rows handled one at a time
 *   j-loop (columns): three passes per row band:
 *
 *     1. Wide tiles   (j += NR):         MR × NR   via micro_kernel_wide
 *     2. Narrow tiles (j += simdWidth):  MR × simdWidth via micro_kernel_narrow
 *     3. Scalar cols  (j += 1):          MR × 1    via scalar_column_MR
 *
 *   Remainder rows (< MR) use the same three-pass column strategy
 *   with single-row variants.
 *
 *     ┌──────────┬──────────┬───────┬────┐
 *     │  NR      │  NR      │simdW  │scl │  ← j passes
 *     │  wide    │  wide    │narrow │    │
 *   ──┼──────────┼──────────┼───────┼────┤
 *   MR│micro_    │micro_    │micro_ │scl │
 *     │kernel_   │kernel_   │kernel_│col │
 *     │wide      │wide      │narrow │MR  │
 *   ──┼──────────┼──────────┼───────┼────┤
 *   MR│  ...     │  ...     │  ...  │    │
 *   ──┼──────────┼──────────┼───────┼────┤
 *  rem│single_   │single_   │single_│scl │  ← remainder rows
 *     │row_wide  │row_wide  │row_   │    │
 *     │          │          │narrow │    │
 *     └──────────┴──────────┴───────┴────┘
 *
 * @section layout Memory Layout Requirements
 *
 * The micro-kernel assumes the "favorable" layout:
 *   - A's contraction axis is the last dim (stride 1): sequential k access
 *   - B's free axis is the last dim (stride 1): K::load reads contiguous columns
 *
 * The einsum caller is responsible for materializing transposed copies
 * when the original layout is unfavorable. The O(N²) transpose cost is
 * negligible against the O(N³) multiply.
 *
 * @section example Concrete Example (MR=4, simdWidth=4, doubles)
 *
 * Computing a 4×12 tile of C (NR_VECS=3):
 *
 *   k=0:
 *     b_vec[0] = load(B + 0*strideB + 0)   → [b00, b01, b02, b03]
 *     b_vec[1] = load(B + 0*strideB + 4)   → [b04, b05, b06, b07]
 *     b_vec[2] = load(B + 0*strideB + 8)   → [b08, b09, b0A, b0B]
 *
 *     a_bcast = broadcast(A[0, 0]) → [a00, a00, a00, a00]
 *     acc[0][0] += a00 * [b00..b03]   acc[0][1] += a00 * [b04..b07]   acc[0][2] += a00 * [b08..b0B]
 *
 *     a_bcast = broadcast(A[1, 0]) → [a10, a10, a10, a10]
 *     acc[1][0] += a10 * [b00..b03]   acc[1][1] += a10 * [b04..b07]   acc[1][2] += a10 * [b08..b0B]
 *
 *     ... (rows 2, 3)
 *
 *   Total per k step: 3 loads + 4 broadcasts + 12 FMAs = 48 multiply-adds
 *   After K steps: 12 stores write the completed 4×12 tile to C
 */
#ifndef KERNEL_GEMM_H
#define KERNEL_GEMM_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "fused/kernel_ops/kernel_helpers.h"

namespace detail
{

    template <typename T, my_size_t Bits, typename Arch>
    struct KernelGemm
    {
        using K = Microkernel<T, Bits, Arch>;
        using Helpers = KernelHelpers<T, Bits, Arch>;
        static constexpr my_size_t simdWidth = K::simdWidth;

        /// Tile height: rows of C computed per micro-kernel invocation.
        static constexpr my_size_t MR = K::MR;

        /// Number of SIMD vectors per tile column. The tile width is NR = NR_VECS × simdWidth.
        static constexpr my_size_t NR_VECS = K::NR_VECS;

        /// Tile width: columns of C computed per wide micro-kernel invocation.
        static constexpr my_size_t NR = K::NR;

        /**
         * @brief Register-blocked GEMM: C[M,N] = A[M,K] × B[K,N]
         *
         * Top-level dispatcher that tiles the output matrix and routes each
         * tile to the appropriate micro-kernel based on its position.
         *
         * All pointers address raw physical memory with padded row strides.
         * The caller must ensure the favorable layout (see @ref layout).
         *
         * @param A       Pointer to first element of A
         * @param M       Number of rows of A (and C)
         * @param K_len   Contraction length (columns of A, rows of B)
         * @param strideA Physical row stride of A (≥ K_len, includes padding)
         * @param B       Pointer to first element of B
         * @param N       Number of columns of B (and C)
         * @param strideB Physical row stride of B (≥ N, includes padding)
         * @param C       Pointer to first element of C (output, zero-initialized not required)
         * @param strideC Physical row stride of C (≥ N, includes padding)
         */
        static void gemm(
            const T *A, my_size_t M, my_size_t K_len, my_size_t strideA,
            const T *B, my_size_t N, my_size_t strideB,
            T *C, my_size_t strideC) noexcept
        {
            // Column boundaries for the three-pass tiling:
            //   [0, wide_N)    → wide micro-kernel   (steps of NR)
            //   [wide_N, narrow_N) → narrow micro-kernel (steps of simdWidth)
            //   [narrow_N, N)  → scalar column loop   (steps of 1)
            const my_size_t wide_N = (N / NR) * NR;
            const my_size_t narrow_N = (N / simdWidth) * simdWidth;

            // ==============================================================
            // Main body: MR rows at a time
            // ==============================================================
            my_size_t i = 0;
            for (; i + MR <= M; i += MR)
            {
                my_size_t j = 0;

                for (; j < wide_N; j += NR)
                {
                    micro_kernel_wide(
                        A + i * strideA, strideA,
                        B + j, strideB,
                        C + i * strideC + j, strideC,
                        K_len);
                }

                for (; j < narrow_N; j += simdWidth)
                {
                    micro_kernel_narrow(
                        A + i * strideA, strideA,
                        B + j, strideB,
                        C + i * strideC + j, strideC,
                        K_len);
                }

                for (; j < N; ++j)
                {
                    scalar_column_MR(
                        A + i * strideA, strideA,
                        B + j, strideB,
                        C + i * strideC + j, strideC,
                        K_len);
                }
            }

            // ==============================================================
            // Remainder rows (< MR): same three-pass column strategy,
            // but processing one row at a time.
            // ==============================================================
            for (; i < M; ++i)
            {
                my_size_t j = 0;

                for (; j < wide_N; j += NR)
                {
                    single_row_wide(
                        A + i * strideA,
                        B + j, strideB,
                        C + i * strideC + j,
                        K_len);
                }

                for (; j < narrow_N; j += simdWidth)
                {
                    single_row_narrow(
                        A + i * strideA,
                        B + j, strideB,
                        C + i * strideC + j,
                        K_len);
                }

                for (; j < N; ++j)
                {
                    T sum = T{0};
                    for (my_size_t k = 0; k < K_len; ++k)
                        sum += A[i * strideA + k] * B[k * strideB + j];
                    C[i * strideC + j] = sum;
                }
            }
        }

    private:
        /**
         * @brief Wide micro-kernel: computes an MR × NR tile of C.
         *
         * This is the hot inner loop — where the bulk of FLOPs happen.
         *
         * Algorithm (outer-product accumulation):
         *   1. Zero-initialize MR × NR_VECS accumulator registers
         *   2. For each k step:
         *      a. Load NR_VECS consecutive SIMD vectors from B's row k
         *      b. For each of MR rows of A:
         *         - Broadcast A[row, k] into a SIMD register
         *         - FMA broadcast × each B vector into the row's accumulators
         *   3. Store MR × NR_VECS accumulators to C
         *
         * Register allocation (example: AVX2 doubles, MR=4, NR_VECS=3):
         *   12 accumulators (acc[4][3])  + 3 B vectors + 1 A broadcast = 16 YMM
         *
         * @param A       Pointer to A[i, 0] — first row of the MR-row panel
         * @param strideA Row stride of A
         * @param B       Pointer to B[0, j] — start of the NR-column panel
         * @param strideB Row stride of B
         * @param C       Pointer to C[i, j] — output tile origin
         * @param strideC Row stride of C
         * @param K_len   Contraction length
         */
        FORCE_INLINE static void micro_kernel_wide(
            const T *A, my_size_t strideA,
            const T *B, my_size_t strideB,
            T *C, my_size_t strideC,
            my_size_t K_len) noexcept
        {
            // Step 1: zero accumulators
            typename K::VecType acc[MR][NR_VECS];
            for (my_size_t r = 0; r < MR; ++r)
                for (my_size_t v = 0; v < NR_VECS; ++v)
                    acc[r][v] = K::set1(T{0});

            // Step 2: outer-product accumulation over k
            for (my_size_t k = 0; k < K_len; ++k)
            {
                // 2a: load NR_VECS contiguous vectors from B[k, j..j+NR-1]
                typename K::VecType b_vec[NR_VECS];
                for (my_size_t v = 0; v < NR_VECS; ++v)
                    b_vec[v] = K::load(B + k * strideB + v * simdWidth);

                // 2b: broadcast each A element and FMA into accumulators
                for (my_size_t r = 0; r < MR; ++r)
                {
                    auto a_bcast = K::set1(A[r * strideA + k]);
                    for (my_size_t v = 0; v < NR_VECS; ++v)
                        acc[r][v] = Helpers::fmadd_safe(a_bcast, b_vec[v], acc[r][v]);
                }
            }

            // Step 3: store completed tile to C
            for (my_size_t r = 0; r < MR; ++r)
                for (my_size_t v = 0; v < NR_VECS; ++v)
                    K::store(C + r * strideC + v * simdWidth, acc[r][v]);
        }

        /**
         * @brief Narrow micro-kernel: computes an MR × simdWidth tile of C.
         *
         * Used when the remaining columns after wide tiles still contain
         * at least one full SIMD vector's worth (simdWidth ≤ remaining < NR).
         *
         * Same algorithm as micro_kernel_wide but with NR_VECS=1: one B vector
         * loaded per k step, reused across MR rows.
         *
         * @param A       Pointer to A[i, 0]
         * @param strideA Row stride of A
         * @param B       Pointer to B[0, j]
         * @param strideB Row stride of B
         * @param C       Pointer to C[i, j]
         * @param strideC Row stride of C
         * @param K_len   Contraction length
         */
        FORCE_INLINE static void micro_kernel_narrow(
            const T *A, my_size_t strideA,
            const T *B, my_size_t strideB,
            T *C, my_size_t strideC,
            my_size_t K_len) noexcept
        {
            typename K::VecType acc[MR];
            for (my_size_t r = 0; r < MR; ++r)
                acc[r] = K::set1(T{0});

            for (my_size_t k = 0; k < K_len; ++k)
            {
                auto b_vec = K::load(B + k * strideB);

                for (my_size_t r = 0; r < MR; ++r)
                {
                    auto a_bcast = K::set1(A[r * strideA + k]);
                    acc[r] = Helpers::fmadd_safe(a_bcast, b_vec, acc[r]);
                }
            }

            for (my_size_t r = 0; r < MR; ++r)
                K::store(C + r * strideC, acc[r]);
        }

        /**
         * @brief Scalar column kernel: computes MR × 1 output elements.
         *
         * Handles the final columns when N is not a multiple of simdWidth.
         * Still processes MR rows simultaneously to reuse the B scalar across rows.
         *
         * @param A       Pointer to A[i, 0]
         * @param strideA Row stride of A
         * @param B       Pointer to B[0, j] (single column)
         * @param strideB Row stride of B
         * @param C       Pointer to C[i, j] (single column)
         * @param strideC Row stride of C
         * @param K_len   Contraction length
         */
        FORCE_INLINE static void scalar_column_MR(
            const T *A, my_size_t strideA,
            const T *B, my_size_t strideB,
            T *C, my_size_t strideC,
            my_size_t K_len) noexcept
        {
            T acc[MR] = {};

            for (my_size_t k = 0; k < K_len; ++k)
            {
                T b_val = B[k * strideB];
                for (my_size_t r = 0; r < MR; ++r)
                    acc[r] += A[r * strideA + k] * b_val;
            }

            for (my_size_t r = 0; r < MR; ++r)
                C[r * strideC] = acc[r];
        }

        /**
         * @brief Single-row wide kernel: computes 1 × NR output elements.
         *
         * Handles remainder rows (M % MR != 0) across wide column spans.
         * Loads NR_VECS B vectors per k step, each multiplied by one A scalar.
         *
         * @param A       Pointer to A[i, 0] (single row)
         * @param B       Pointer to B[0, j]
         * @param strideB Row stride of B
         * @param C       Pointer to C[i, j]
         * @param K_len   Contraction length
         */
        FORCE_INLINE static void single_row_wide(
            const T *A,
            const T *B, my_size_t strideB,
            T *C,
            my_size_t K_len) noexcept
        {
            typename K::VecType acc[NR_VECS];
            for (my_size_t v = 0; v < NR_VECS; ++v)
                acc[v] = K::set1(T{0});

            for (my_size_t k = 0; k < K_len; ++k)
            {
                auto a_bcast = K::set1(A[k]);
                for (my_size_t v = 0; v < NR_VECS; ++v)
                    acc[v] = Helpers::fmadd_safe(a_bcast, K::load(B + k * strideB + v * simdWidth), acc[v]);
            }

            for (my_size_t v = 0; v < NR_VECS; ++v)
                K::store(C + v * simdWidth, acc[v]);
        }

        /**
         * @brief Single-row narrow kernel: computes 1 × simdWidth output elements.
         *
         * Handles remainder rows across partial column spans (between wide
         * and scalar regions). One B vector per k step, one A scalar.
         *
         * @param A       Pointer to A[i, 0] (single row)
         * @param B       Pointer to B[0, j]
         * @param strideB Row stride of B
         * @param C       Pointer to C[i, j]
         * @param K_len   Contraction length
         */
        FORCE_INLINE static void single_row_narrow(
            const T *A,
            const T *B, my_size_t strideB,
            T *C,
            my_size_t K_len) noexcept
        {
            typename K::VecType acc = K::set1(T{0});

            for (my_size_t k = 0; k < K_len; ++k)
            {
                auto b_vec = K::load(B + k * strideB);
                auto a_bcast = K::set1(A[k]);
                acc = Helpers::fmadd_safe(a_bcast, b_vec, acc);
            }

            K::store(C, acc);
        }
    };

} // namespace detail

#endif // KERNEL_GEMM_H