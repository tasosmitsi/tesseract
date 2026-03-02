# How the GEMM Micro-Kernel Works

A visual walkthrough of tesseract's register-blocked matrix multiplication.

---

## The 30-Second Version

The old code computed **one C element at a time** — a dot product. The new code computes a **whole tile of C at once** by flipping the loop order.

---

## 1. The Problem: Naive Matrix Multiply

To compute C = A × B, the naive way calculates each output element independently:

```
C[0,0] = A[row0] · B[col0]    → load row0, load col0, dot
C[0,1] = A[row0] · B[col1]    → load row0 AGAIN, load col1, dot
C[0,2] = A[row0] · B[col2]    → load row0 AGAIN, load col2, dot
...
```

Same row of A loaded **100 times** for a 100×100 matrix. Massive waste.

---

## 2. The Trick: Outer Product

Instead of walking **across** A and **down** B for one element, take **one column slice** from A and **one row slice** from B and update the **entire tile** at once:

```
k=0:
                              B row 0 (1 SIMD load)
                         ┌────────────────────────────────┐
                         │ b00      b01      b02      b03 │
                         └────────────────────────────────┘
  A col 0                   ↓        ↓        ↓        ↓
   ┌───┐            ┌──────────────────────────────────────┐
   │a00│ broadcast→ │  a00·b00  a00·b01  a00·b02  a00·b03  │  ← 1 FMA!
   │a10│ broadcast→ │  a10·b00  a10·b01  a10·b02  a10·b03  │  ← 1 FMA!
   │a20│ broadcast→ │  a20·b00  a20·b01  a20·b02  a20·b03  │  ← 1 FMA!
   │a30│ broadcast→ │  a30·b00  a30·b01  a30·b02  a30·b03  │  ← 1 FMA!
   └───┘            └──────────────────────────────────────┘
                              C tile (4×4)
```

**1 load** of B, **4 broadcasts** of A, **4 FMAs** → **16 multiply-adds**.

Then k=1 does the same thing, accumulating onto the **same C tile**. Then k=2, etc. After all K steps, the C tile is complete — store it to memory.

The key: **B is loaded once and reused across all 4 rows of A.** That's 4× fewer loads.

**SIMD makes this fast:** The B row `[b00, b01, b02, b03]` is loaded as *one* SIMD vector. Each A value is *broadcast* (copied to all 4 lanes). Then **FMA** (fused multiply-add) does 4 multiplications + 4 additions in a single CPU instruction.

---

## 3. Making It Wider: NR_VECS

Why stop at 4 columns? Load **three** B vectors instead of one:

```
k=0:
              b_vec[0]            b_vec[1]            b_vec[2]
         ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
         │ b00 b01 b02 b03 │ │ b04 b05 b06 b07 │ │ b08 b09 b0A b0B │
         └─────────────────┘ └─────────────────┘ └─────────────────┘

a00 →  [ FMA  FMA  FMA  FMA | FMA  FMA  FMA  FMA | FMA  FMA  FMA  FMA ]
a10 →  [ FMA  FMA  FMA  FMA | FMA  FMA  FMA  FMA | FMA  FMA  FMA  FMA ]
a20 →  [ FMA  FMA  FMA  FMA | FMA  FMA  FMA  FMA | FMA  FMA  FMA  FMA ]
a30 →  [ FMA  FMA  FMA  FMA | FMA  FMA  FMA  FMA | FMA  FMA  FMA  FMA ]
       └──────────── 4×12 = 48 multiply-adds per k step ──────────────┘
```

Same 4 A broadcasts, but now each one feeds **3 FMAs** instead of 1. Triple the work for just 2 extra loads.

---

## 4. Register Budget: Why MR=4, NR_VECS=3

Everything lives in CPU registers during the k-loop. AVX2 has **16 YMM registers**:

```
┌─────────┬─────────┬─────────┬─────────┐
│acc[0][0]│acc[0][1]│acc[0][2]│ b_vec[0]│
├─────────┼─────────┼─────────┼─────────┤
│acc[1][0]│acc[1][1]│acc[1][2]│ b_vec[1]│
├─────────┼─────────┼─────────┼─────────┤
│acc[2][0]│acc[2][1]│acc[2][2]│ b_vec[2]│
├─────────┼─────────┼─────────┼─────────┤
│acc[3][0]│acc[3][1]│acc[3][2]│ a_bcast │
└─────────┴─────────┴─────────┴─────────┘
  12 accumulators + 3 B vecs + 1 A bcast = 16  ← PERFECT FIT
```

The register pressure formula: `MR × NR_VECS + NR_VECS + 1 ≤ num_registers`

| Config | Registers | Cycles | Status |
|---|---|---|---|
| MR=4, NR_VECS=2 | 11 | 124K | 5 spare |
| MR=6, NR_VECS=2 | 15 | 119K | 1 spare |
| **MR=4, NR_VECS=3** | **16** | **117K** | **perfect fit** |
| MR=6, NR_VECS=3 | 22 | 207K | 6 spilled! |

**MR=6, NR_VECS=3** would need 22 registers → 6 spill to stack → every FMA becomes load-FMA-store → **slower**.

---

## 5. Tiling: Covering the Full Matrix

Real matrices are bigger than one tile. The GEMM covers C with three column passes:

```
         ←── NR=12 ──→ ←── NR ──→ ←simd→ ←s→
        ┌─────────────┬──────────┬──────┬───┐
   MR=4 │   WIDE      │   WIDE   │NARROW│ S │  ← fastest
        │  4×12 tile  │  4×12    │ 4×4  │4×1│
        ├─────────────┼──────────┼──────┼───┤
   MR=4 │   WIDE      │   WIDE   │NARROW│ S │
        │  4×12 tile  │  4×12    │ 4×4  │4×1│
        ├─────────────┼──────────┼──────┼───┤
  rem=2 │ 1-row wide  │ 1-row    │1-row │1×1│  ← remainder rows
        │ 1-row wide  │ 1-row    │1-row │1×1│
        └─────────────┴──────────┴──────┴───┘
```

Wide tiles do the bulk of the work. Narrow and scalar handle the edges.

**Why three passes?** The wide micro-kernel is the fastest — it loads NR_VECS (3) B vectors per k-step. But if N isn't a multiple of NR, there are leftover columns. The **narrow** kernel handles chunks of simdWidth, and the **scalar** path handles the final 1-3 columns. Same logic for remainder rows when M isn't a multiple of MR.

---

## 6. What Happens to the Padding?

The padding is **invisible** to the GEMM. The strides jump over it, and the loops use real dimensions, not padded ones.

Consider A as [4,5] doubles with simdWidth=4:

```
Physical memory (padded to [4,8], strideA=8):

        real data (K=5)            padding
       ┌─────────────────────────┬──────────────┐
row 0: │ a00  a01  a02  a03  a04 │  P    P    P │
row 1: │ a10  a11  a12  a13  a14 │  P    P    P │
row 2: │ a20  a21  a22  a23  a24 │  P    P    P │
row 3: │ a30  a31  a32  a33  a34 │  P    P    P │
       └─────────────────────────┴──────────────┘
        ←────────── strideA = 8 ───────────────→
```

The GEMM reads `A[r * strideA + k]` where **k goes from 0 to K_len−1 = 4**. It never reaches k=5,6,7 (the padding). The stride of 8 just means "jump 8 elements to get to the next row" — the 3 padding elements are skipped over, never touched.

For B as [5,6] padded to [5,8], strideB=8, the j-loop boundaries use **real N=6**, not paddedN=8:

```
NR = 12 (3×simdWidth)    → wide_N   = (6/12)*12 = 0   ← no wide tiles!
simdWidth = 4             → narrow_N = (6/4)*4   = 4   ← one narrow tile

j-loop:
  j=0..3:  K::load(B + k*8 + 0) → reads [b_k0, b_k1, b_k2, b_k3]  ✓ real data
  j=4:     scalar → reads B[k*8 + 4] = b_k4  ✓
  j=5:     scalar → reads B[k*8 + 5] = b_k5  ✓
  j=6,7:   NEVER REACHED — loop stops at N=6
```

The three protections:

1. **k-loop:** 0 to K_len-1 (real K, not paddedK) → A's padding columns skipped
2. **j-loop:** 0 to N-1 (real N, not paddedN) → B/C's padding columns skipped
3. **Strides:** strideA, strideB, strideC = padded widths → correctly hop over padding between rows

The padding exists purely so that `K::load` and `K::store` within the real dimensions are always **SIMD-aligned** (addresses are multiples of 32 bytes for AVX). It's never read from or written to by the GEMM itself.

---

The optimization path in three sentences:

1. Replace per-element dot products with a register-blocked outer-product micro-kernel.
2. Widen the tile to use all available SIMD registers (MR=4, NR_VECS=3 for AVX2).
3. Materialize transposed copies when the layout is unfavorable (O(N²) cost, amortized by O(N³) multiply).
