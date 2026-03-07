// /**
    //  * @brief Contract two tensors along specified axes using SIMD dot products.
    //  *
    //  * Computes C[free_indices] = sum_k A[...,k,...] * B[...,k,...]
    //  * where k runs along axis `a` of tensor1 and axis `b` of tensor2.
    //  *
    //  * For each output element, builds the fiber start (base) for each tensor
    //  * by setting the contraction index to 0, then calls dot() with the
    //  * physical stride along the contraction axis.
    //  *
    //  * ============================================================================
    //  * EXAMPLE: C[2,2] = A[2,3] * B[3,2], contract a=1, b=0
    //  * ============================================================================
    //  *
    //  *   A[2,3] padded to [2,4]:           B[3,2] padded to [3,4]:
    //  *     [a00 a01 a02  P]                  [b00 b01  P  P]
    //  *     [a10 a11 a12  P]                  [b10 b11  P  P]
    //  *     Layout1::stride(1) = 1            [b20 b21  P  P]
    //  *                                       Layout2::stride(0) = 4
    //  *
    //  *   Output C[i,j]:
    //  *     C[0,0]: A[0,:] dot B[:,0]
    //  *       base1=0, stride1=1  →  [a00, a01, a02] contiguous
    //  *       base2=0, stride2=4  →  [b00, b10, b20] strided
    //  *       → dot(A, 0, 1, B, 0, 4, 3)
    //  *
    //  *     C[0,1]: A[0,:] dot B[:,1]
    //  *       base1=0, stride1=1
    //  *       base2=1, stride2=4  →  [b01, b11, b21]
    //  *       → dot(A, 0, 1, B, 1, 4, 3)
    //  *
    //  *     C[1,0]: A[1,:] dot B[:,0]
    //  *       base1=4, stride1=1  →  [a10, a11, a12]
    //  *       base2=0, stride2=4
    //  *       → dot(A, 4, 1, B, 0, 4, 3)
    //  *
    //  * ============================================================================
    //  */
    // template <typename LeftExpr, typename RightExpr>
    //     requires(expression::traits<LeftExpr>::IsPhysical &&
    //              expression::traits<RightExpr>::IsPhysical)
    // static FusedTensorND einsum_new_old(
    //     const BaseExpr<LeftExpr> &_tensor1,
    //     const BaseExpr<RightExpr> &_tensor2,
    //     const my_size_t a,
    //     const my_size_t b)
    // {
    //     static constexpr my_size_t Dims1 = LeftExpr::NumDims;
    //     static constexpr my_size_t Dims2 = RightExpr::NumDims;

    //     static_assert(Dims1 >= 2, "Tensor 1 must have at least 2 dimensions");
    //     static_assert(Dims2 >= 2, "Tensor 2 must have at least 2 dimensions");

    //     // Runtime validation
    //     if (a >= Dims1 || b >= Dims2)
    //         MyErrorHandler::error("Invalid contraction axis");

    //     if (_tensor1.derived().getDim(a) != _tensor2.derived().getDim(b))
    //         MyErrorHandler::error("Contraction dimensions mismatch");

    //     // Contraction length and physical strides along contraction axes
    //     using Layout1 = typename LeftExpr::Layout;
    //     using Layout2 = typename RightExpr::Layout;
    //     using Kern = KernelOps<T, BITS, DefaultArch>;

    //     const my_size_t K = _tensor1.derived().getDim(a);
    //     const my_size_t stride1 = Layout1::stride(a);
    //     const my_size_t stride2 = Layout2::stride(b);

    //     // Build output dimensions (all dims except contracted ones)
    //     static constexpr my_size_t n_newDims = Dims1 + Dims2 - 2;
    //     my_size_t newDims[n_newDims];
    //     my_size_t d = 0;
    //     for (my_size_t i = 0; i < Dims1; ++i)
    //         if (i != a)
    //             newDims[d++] = _tensor1.derived().getDim(i);
    //     for (my_size_t i = 0; i < Dims2; ++i)
    //         if (i != b)
    //             newDims[d++] = _tensor2.derived().getDim(i);

    //     // Validate output dimensions match
    //     FusedTensorND _outp;
    //     for (my_size_t i = 0; i < n_newDims; ++i)
    //     {
    //         if (newDims[i] != _outp.getDim(i))
    //             MyErrorHandler::error("Output dimensions mismatch");
    //     }

    //     // Generate all output index combinations
    //     static constexpr my_size_t total_combinations = (1 * ... * Dims);
    //     my_size_t combinations[total_combinations][n_newDims];
    //     generate_combinations(newDims, combinations);

    //     // For each output element, compute dot product of two fibers
    //     for (my_size_t comb = 0; comb < total_combinations; ++comb)
    //     {
    //         // Build tensor1 indices with contraction axis = 0
    //         my_size_t indices1[Dims1] = {0};
    //         my_size_t l = 0;
    //         for (my_size_t i = 0; i < Dims1; ++i)
    //         {
    //             if (i != a)
    //                 indices1[i] = combinations[comb][l++];
    //             // else indices1[i] = 0 (fiber start)
    //         }

    //         // Build tensor2 indices with contraction axis = 0
    //         my_size_t indices2[Dims2] = {0};
    //         for (my_size_t i = 0; i < Dims2; ++i)
    //         {
    //             if (i != b)
    //                 indices2[i] = combinations[comb][l++];
    //             // else indices2[i] = 0 (fiber start)
    //         }

    //         // Physical base offsets where each fiber starts
    //         const my_size_t base1 = Layout1::logical_coords_to_physical_flat(indices1);
    //         const my_size_t base2 = Layout2::logical_coords_to_physical_flat(indices2);

    //         // Dot product replaces the inner k-loop
    //         _outp(combinations[comb]) = Kern::dot(
    //             _tensor1.derived(), base1, stride1,
    //             _tensor2.derived(), base2, stride2,
    //             K);
    //     }

    //     return _outp;
    // }

    // /**
    //  * @brief Contract two tensors along specified axes using SIMD dot products.
    //  *
    //  * Computes C[free_indices] = sum_k A[...,k,...] * B[...,k,...]
    //  * where k runs along axis `a` of tensor1 and axis `b` of tensor2.
    //  *
    //  * Uses pre-computed stride maps to convert output coordinates directly
    //  * into physical base offsets for each input tensor, avoiding per-element
    //  * index array construction and logical_coords_to_physical_flat calls.
    //  *
    //  * ============================================================================
    //  * STRIDE MAP EXAMPLE: C[2,2] = A[2,3] * B[3,2], contract a=1, b=0
    //  * ============================================================================
    //  *
    //  *   A[2,3] padded to [2,4]:           B[3,2] padded to [3,4]:
    //  *     strides: [4, 1]                   strides: [4, 1]
    //  *     contract dim 1 (stride=1)         contract dim 0 (stride=4)
    //  *
    //  *   Output dims: [2, 2]  (A's dim 0, B's dim 1)
    //  *
    //  *   Stride maps (one entry per output dim):
    //  *     strides1_map = [4, 0]   ← output dim 0 → A's dim 0 (stride 4)
    //  *     strides2_map = [0, 1]   ← output dim 1 → B's dim 1 (stride 1)
    //  *
    //  *   For output coord (1, 1):
    //  *     base1 = 1*4 + 1*0 = 4   (start of A[1,:])
    //  *     base2 = 1*0 + 1*1 = 1   (start of B[:,1])
    //  *     → dot(A, 4, 1, B, 1, 4, 3)
    //  *
    //  * ============================================================================
    //  */
    // template <typename LeftExpr, typename RightExpr>
    //     requires(expression::traits<LeftExpr>::IsPhysical &&
    //              expression::traits<RightExpr>::IsPhysical)
    // static FusedTensorND einsum_new(
    //     const BaseExpr<LeftExpr> &_tensor1,
    //     const BaseExpr<RightExpr> &_tensor2,
    //     const my_size_t a,
    //     const my_size_t b)
    // {
    //     static constexpr my_size_t Dims1 = LeftExpr::NumDims;
    //     static constexpr my_size_t Dims2 = RightExpr::NumDims;

    //     static_assert(Dims1 >= 2, "Tensor 1 must have at least 2 dimensions");
    //     static_assert(Dims2 >= 2, "Tensor 2 must have at least 2 dimensions");

    //     // Runtime validation
    //     if (a >= Dims1 || b >= Dims2)
    //         MyErrorHandler::error("Invalid contraction axis");

    //     if (_tensor1.derived().getDim(a) != _tensor2.derived().getDim(b))
    //         MyErrorHandler::error("Contraction dimensions mismatch");

    //     using Layout1 = typename LeftExpr::Layout;
    //     using Layout2 = typename RightExpr::Layout;
    //     using OutputLayout = Layout;
    //     using Kern = KernelOps<T, BITS, DefaultArch>;

    //     const my_size_t K_len = _tensor1.derived().getDim(a);
    //     const my_size_t contract_stride1 = Layout1::stride(a);
    //     const my_size_t contract_stride2 = Layout2::stride(b);

    //     // ====================================================================
    //     // Build stride maps
    //     // ====================================================================
    //     // For each output dimension d:
    //     //   strides1_map[d] = physical stride in tensor1 (0 if d comes from tensor2)
    //     //   strides2_map[d] = physical stride in tensor2 (0 if d comes from tensor1)
    //     //   out_dims[d]     = size of output dimension d

    //     static constexpr my_size_t n_newDims = Dims1 + Dims2 - 2;
    //     my_size_t strides1_map[n_newDims];
    //     my_size_t strides2_map[n_newDims];
    //     my_size_t out_dims[n_newDims];

    //     my_size_t d = 0;
    //     for (my_size_t i = 0; i < Dims1; ++i)
    //     {
    //         if (i != a)
    //         {
    //             out_dims[d] = _tensor1.derived().getDim(i);
    //             strides1_map[d] = Layout1::stride(i);
    //             strides2_map[d] = 0;
    //             ++d;
    //         }
    //     }
    //     for (my_size_t i = 0; i < Dims2; ++i)
    //     {
    //         if (i != b)
    //         {
    //             out_dims[d] = _tensor2.derived().getDim(i);
    //             strides1_map[d] = 0;
    //             strides2_map[d] = Layout2::stride(i);
    //             ++d;
    //         }
    //     }

    //     // ====================================================================
    //     // Validate output dimensions
    //     // ====================================================================

    //     FusedTensorND _outp;
    //     for (my_size_t i = 0; i < n_newDims; ++i)
    //     {
    //         if (out_dims[i] != _outp.getDim(i))
    //             MyErrorHandler::error("Output dimensions mismatch");
    //     }

    //     // ====================================================================
    //     // Pre-compute output physical strides for direct memory writes
    //     // ====================================================================

    //     my_size_t out_strides[n_newDims];
    //     for (my_size_t i = 0; i < n_newDims; ++i)
    //         out_strides[i] = OutputLayout::stride(i);

    //     T *out_ptr = _outp.data();

    //     // ====================================================================
    //     // Main loop: iterate all output elements
    //     // ====================================================================

    //     static constexpr my_size_t total_elements = (1 * ... * Dims);

    //     for (my_size_t flat = 0; flat < total_elements; ++flat)
    //     {
    //         // Decompose flat index into output coordinates (row-major)
    //         my_size_t coords[n_newDims];
    //         my_size_t tmp = flat;
    //         for (my_size_t i = n_newDims; i-- > 0;)
    //         {
    //             coords[i] = tmp % out_dims[i];
    //             tmp /= out_dims[i];
    //         }

    //         // Compute physical bases via stride maps (dot product of coords × strides)
    //         my_size_t base1 = 0;
    //         my_size_t base2 = 0;
    //         my_size_t out_phys = 0;
    //         for (my_size_t i = 0; i < n_newDims; ++i)
    //         {
    //             base1 += coords[i] * strides1_map[i];
    //             base2 += coords[i] * strides2_map[i];
    //             out_phys += coords[i] * out_strides[i];
    //         }

    //         // Dot product along contraction axis → write directly to output
    //         out_ptr[out_phys] = Kern::dot(
    //             _tensor1.derived(), base1, contract_stride1,
    //             _tensor2.derived(), base2, contract_stride2,
    //             K_len);
    //     }

    //     return _outp;
    // }