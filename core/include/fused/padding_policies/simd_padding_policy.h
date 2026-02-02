#pragma once

#include "config.h"                              // for my_size_t, BITS, DefaultArch
#include "fused/microkernels/microkernel_base.h" // for Microkernel, DATA_ALIGNAS
#include "containers/array.h"                    // for Array

/**
 * @brief Padding policy that pads the last dimension for SIMD alignment.
 *
 * @tparam T     Element type (float, double, Complex<float>, etc.)
 * @tparam Dims  Logical dimensions of the tensor (e.g., 8, 6 for an 8x6 matrix)
 *
 * All computations happen at compile-time. No runtime overhead.
 *
 * ============================================================================
 * THE PROBLEM THIS SOLVES
 * ============================================================================
 *
 * SIMD load instructions (e.g., _mm256_load_pd) require memory addresses to be
 * aligned to SimdWidth * sizeof(T) bytes. For AVX with double:
 *   - SimdWidth = 4 (four doubles per register)
 *   - Required alignment = 4 * 8 = 32 bytes
 *
 * For a tensor A[8, 6] stored in row-major order WITHOUT padding:
 *
 *   Memory layout (each cell = 1 double = 8 bytes):
 *
 *   Index:    0  1  2  3  4  5 | 6  7  8  9 10 11 | 12 13 14 15 16 17 | ...
 *   Row:      |---- row 0 ----| |---- row 1 -----| |---- row 2 ------| ...
 *   Address:  0  8 16 24 32 40  48 56 64 72 80 88  96 ...
 *                               ^
 *                               Row 1 starts at byte 48
 *                               48 % 32 = 16 ≠ 0 → NOT ALIGNED!
 *
 *   Strides: [6, 1]
 *   Row bases: 0, 6, 12, 18, 24, 30, 36, 42
 *
 *   Alignment check (need base % SimdWidth == 0):
 *     Row 0: base=0,  0 % 4 = 0 ✓
 *     Row 1: base=6,  6 % 4 = 2 ✗ MISALIGNED → SEGFAULT!
 *     Row 2: base=12, 12 % 4 = 0 ✓
 *     Row 3: base=18, 18 % 4 = 2 ✗ MISALIGNED → SEGFAULT!
 *
 * ============================================================================
 * THE SOLUTION: PAD THE LAST DIMENSION
 * ============================================================================
 *
 * Pad the last dimension to the next multiple of SimdWidth.
 *
 * For A[8, 6] with SimdWidth=4:
 *   - Logical last dim:  6
 *   - Padded last dim:   8 (next multiple of 4)
 *   - Physical storage:  8 rows × 8 cols = 64 elements
 *
 *   Memory layout WITH padding:
 *
 *   Index:    0  1  2  3  4  5  P  P | 8  9 10 11 12 13  P  P | 16 17 ...
 *   Row:      |---- row 0 ----| pad | |---- row 1 -----| pad | ...
 *   Address:  0  8 16 24 32 40 48 56  64 72 80 88 96 104 ...
 *                                     ^
 *                                     Row 1 starts at byte 64
 *                                     64 % 32 = 0 → ALIGNED!
 *
 *   P = padding slots (allocated but unused, zero-initialized)
 *
 *   Strides: [8, 1] ← computed from PADDED dimensions!
 *   Row bases: 0, 8, 16, 24, 32, 40, 48, 56
 *
 *   Alignment check:
 *     Row 0: base=0,  0 % 4 = 0 ✓
 *     Row 1: base=8,  8 % 4 = 0 ✓
 *     Row 2: base=16, 16 % 4 = 0 ✓
 *     Row 3: base=24, 24 % 4 = 0 ✓
 *     ... ALL ALIGNED!
 *
 * ============================================================================
 * EXAMPLES FOR DIFFERENT TYPES AND DIMENSIONS
 * ============================================================================
 *
 * Example 1: FusedTensorND<double, 8, 6> (AVX, SimdWidth=4)
 *   - Logical dims: [8, 6]
 *   - Padded last dim: pad(6) = 8 (next multiple of 4)
 *   - Physical size: 8 * 8 = 64 elements
 *   - Memory overhead: 64 vs 48 = 33%
 *
 * Example 2: FusedTensorND<float, 8, 6> (AVX, SimdWidth=8)
 *   - Logical dims: [8, 6]
 *   - Padded last dim: pad(6) = 8 (next multiple of 8)
 *   - Physical size: 8 * 8 = 64 elements
 *   - Memory overhead: 64 vs 48 = 33%
 *
 * Example 3: FusedTensorND<float, 5, 10> (AVX, SimdWidth=8)
 *   - Logical dims: [5, 10]
 *   - Padded last dim: pad(10) = 16 (next multiple of 8)
 *   - Physical size: 5 * 16 = 80 elements
 *   - Memory overhead: 80 vs 50 = 60%
 *
 * Example 4: FusedTensorND<double, 4, 4> (AVX, SimdWidth=4, already aligned)
 *   - Logical dims: [4, 4]
 *   - Padded last dim: pad(4) = 4 (already multiple of 4, no change)
 *   - Physical size: 4 * 4 = 16 elements
 *   - Memory overhead: 0%
 *
 * Example 5: FusedTensorND<float, 2, 3, 5> (AVX, SimdWidth=8, 3D tensor)
 *   - Logical dims: [2, 3, 5]
 *   - Only LAST dimension is padded: pad(5) = 8
 *   - Physical dims: [2, 3, 8]
 *   - Physical size: 2 * 3 * 8 = 48 elements
 *   - Strides: [24, 8, 1] (computed from physical dims)
 *
 * Example 6: FusedTensorND<Complex<double>, 8, 6> (AVX, SimdWidth=2)
 *   - sizeof(Complex<double>) = 16 bytes
 *   - SimdWidth = 2 (from Microkernel<Complex<double>, 256, X86_AVX>)
 *   - Logical dims: [8, 6]
 *   - Padded last dim: pad(6) = 6 (already multiple of 2)
 *   - Physical size: 8 * 6 = 48 elements
 *   - Memory overhead: 0%
 *
 * Example 7: FusedTensorND<double, 8, 6> (GENERICARCH, SimdWidth=1)
 *   - Logical dims: [8, 6]
 *   - Padded last dim: pad(6) = 6 (everything is multiple of 1)
 *   - Physical size: 8 * 6 = 48 elements
 *   - Memory overhead: 0% (no padding when no SIMD)
 *
 * ============================================================================
 * WHY ONLY PAD THE LAST DIMENSION?
 * ============================================================================
 *
 * In row-major storage, the last dimension is contiguous in memory.
 * When we iterate over the last axis (stride=1), we access consecutive elements.
 *
 * For A[M, N]:
 *   - Iterating row i:    A[i, 0], A[i, 1], A[i, 2], ... (contiguous, stride=1)
 *   - Iterating column j: A[0, j], A[1, j], A[2, j], ... (strided, stride=N)
 *
 * Padding the last dimension ensures that EVERY "contiguous slice" starts at
 * an aligned address:
 *   - Row 0 starts at index 0 * paddedN = 0 → aligned
 *   - Row 1 starts at index 1 * paddedN = paddedN → aligned (if paddedN % SimdWidth == 0)
 *   - Row 2 starts at index 2 * paddedN = 2*paddedN → aligned
 *   - etc.
 *
 * ============================================================================
 * EMBEDDED SYSTEMS CONSIDERATION
 * ============================================================================
 *
 * On modern desktop CPUs (Intel Haswell+, AMD Zen+), unaligned loads (loadu)
 * have essentially zero penalty when data doesn't cross cache line boundaries.
 *
 * However, on embedded systems and older architectures:
 *   - Unaligned loads may not exist at all
 *   - Unaligned loads may be 2-10x slower than aligned loads
 *   - Some DSPs require strict alignment
 *
 * Padding trades memory for guaranteed alignment, which is often the right
 * choice for embedded systems where predictable performance matters.
 *
 * ============================================================================
 * DESIGN: MICROKERNEL AS SINGLE SOURCE OF TRUTH
 * ============================================================================
 *
 * The SimdWidth is obtained from Microkernel<T, BITS, DefaultArch>::SimdWidth.
 *
 * Why not compute it as DATA_ALIGNAS / sizeof(T)?
 *
 *   1. Microkernel is the hardware abstraction layer - it owns SIMD configuration.
 *   2. Avoids duplication: SimdWidth is defined once, in the microkernel.
 *   3. Handles edge cases correctly:
 *      - GENERICARCH: BITS=0, DATA_ALIGNAS=0, but Microkernel::SimdWidth=1
 *      - Complex types: SimdWidth depends on how the microkernel packs them
 *   4. Custom types just need to define their Microkernel specialization.
 *      The padding policy automatically picks up the correct SimdWidth.
 *
 * Dependency chain (no cycles):
 *   Microkernel (defines SimdWidth)
 *       ↓
 *   PaddingPolicy (reads SimdWidth, computes PhysicalSize)
 *       ↓
 *   Storage (allocates PhysicalSize elements)
 *       ↓
 *   Tensor (uses storage)
 */

template <typename T, my_size_t... Dims>
struct SimdPaddingPolicy
{
    // ========================================================================
    // COMPILE-TIME VALIDATION
    // ========================================================================

    static_assert(sizeof...(Dims) > 0, "SimdPaddingPolicy: At least one dimension is required");

    // ========================================================================
    // SIMD CONFIGURATION FROM MICROKERNEL
    // ========================================================================

    /**
     * SimdWidth from the Microkernel - the single source of truth.
     *
     * Examples:
     *   - Microkernel<double, 256, X86_AVX>::SimdWidth = 4
     *   - Microkernel<float, 256, X86_AVX>::SimdWidth = 8
     *   - Microkernel<double, 128, X86_SSE>::SimdWidth = 2
     *   - Microkernel<Complex<double>, 256, X86_AVX>::SimdWidth = 2
     *   - Microkernel<float, 0, GENERICARCH>::SimdWidth = 1
     */
    static constexpr my_size_t SimdWidth = Microkernel<T, BITS, DefaultArch>::simdWidth;

    static_assert(SimdWidth >= 1, "SimdPaddingPolicy: SimdWidth must be at least 1");

    // ========================================================================
    // COMPILE-TIME DIMENSION ANALYSIS
    // ========================================================================

    /** Number of dimensions (e.g., 2 for a matrix, 3 for a 3D tensor) */
    static constexpr my_size_t NumDims = sizeof...(Dims);

    /**
     * Array of logical dimensions.
     *
     * For FusedTensorND<T, 8, 6>: LogicalDims = {8, 6}
     * For FusedTensorND<T, 2, 3, 4>: LogicalDims = {2, 3, 4}
     */
    static constexpr Array<my_size_t, NumDims> computeLogicalDims()
    {
        return Array<my_size_t, NumDims>{Dims...};
    }

    static constexpr Array<my_size_t, NumDims> LogicalDims = computeLogicalDims();

    // ========================================================================
    // PADDING COMPUTATION
    // ========================================================================

    /**
     * Round up n to the next multiple of SimdWidth.
     *
     * Formula: ceil(n / SimdWidth) * SimdWidth
     * Implemented as: ((n + SimdWidth - 1) / SimdWidth) * SimdWidth
     *
     * Examples with SimdWidth=4:
     *   pad(1) = 4
     *   pad(2) = 4
     *   pad(3) = 4
     *   pad(4) = 4 (already aligned)
     *   pad(5) = 8
     *   pad(6) = 8
     *   pad(7) = 8
     *   pad(8) = 8 (already aligned)
     *
     * Examples with SimdWidth=8:
     *   pad(6) = 8
     *   pad(10) = 16
     *
     * With SimdWidth=1 (GENERICARCH):
     *   pad(n) = n (no padding needed)
     */
    static constexpr my_size_t pad(my_size_t n)
    {
        return ((n + SimdWidth - 1) / SimdWidth) * SimdWidth;
    }

    /**
     * Original (logical) last dimension.
     *
     * For FusedTensorND<T, 8, 6>: LastDim = 6
     */
    static constexpr my_size_t LastDim = LogicalDims[NumDims - 1];

    /**
     * Padded last dimension.
     *
     * For FusedTensorND<float, 8, 6> with SimdWidth=8: PaddedLastDim = 8
     * For FusedTensorND<double, 8, 6> with SimdWidth=4: PaddedLastDim = 8
     * For FusedTensorND<double, 8, 6> with SimdWidth=1: PaddedLastDim = 6 (no change)
     */
    static constexpr my_size_t PaddedLastDim = pad(LastDim);

    /**
     * Logical size = product of all logical dimensions.
     *
     * For FusedTensorND<double, 8, 6>: LogicalSize = 48
     */
    static constexpr my_size_t LogicalSize = (Dims * ...);

    /**
     * Compute total physical storage size.
     *
     * Physical size = (product of all LogicalDims except last) × PaddedLastDim
     *
     * For FusedTensorND<double, 8, 6> with SimdWidth=4:
     *   - LogicalDims = {8, 6}
     *   - PaddedLastDim = 8
     *   - PhysicalSize = 8 * 8 = 64
     *
     * For FusedTensorND<float, 2, 3, 5> with SimdWidth=8:
     *   - LogicalDims = {2, 3, 5}
     *   - PaddedLastDim = 8
     *   - PhysicalSize = 2 * 3 * 8 = 48
     *
     * For FusedTensorND<double, 8, 6> with SimdWidth=1 (GENERICARCH):
     *   - LogicalDims = {8, 6}
     *   - PaddedLastDim = 6 (no padding)
     *   - PhysicalSize = 8 * 6 = 48 (no overhead)
     */
    static constexpr my_size_t computePhysicalSize()
    {
        my_size_t size = 1;
        for (my_size_t i = 0; i < NumDims - 1; ++i)
            size *= LogicalDims[i];
        return size * PaddedLastDim;
    }

    /** Total number of elements in physical storage (including padding) */
    static constexpr my_size_t PhysicalSize = computePhysicalSize();

    // ========================================================================
    // PHYSICAL DIMENSIONS (for StridedLayout)
    // ========================================================================

    /**
     * Compile-time array of physical dimensions.
     *
     * Physical dims = [dim0, dim1, ..., dimN-2, PaddedLastDim]
     *
     * For FusedTensorND<double, 8, 6> with SimdWidth=4:
     *   - Logical dims:  [8, 6]
     *   - Physical dims: [8, 8]
     *
     * For FusedTensorND<float, 2, 3, 5> with SimdWidth=8:
     *   - Logical dims:  [2, 3, 5]
     *   - Physical dims: [2, 3, 8]
     */
    static constexpr Array<my_size_t, NumDims> computePhysicalDims()
    {
        Array<my_size_t, NumDims> result{};
        for (my_size_t i = 0; i < NumDims - 1; ++i)
            result[i] = LogicalDims[i];
        result[NumDims - 1] = PaddedLastDim;
        return result;
    }

    /** Physical dimensions array - all computation happens at compile-time */
    static constexpr Array<my_size_t, NumDims> PhysicalDims = computePhysicalDims();
};