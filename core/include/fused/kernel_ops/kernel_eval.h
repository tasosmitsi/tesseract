/**
 * @file kernel_eval.h
 * @brief Expression evaluation — dispatch, contiguous path, and permuted path.
 *
 * Evaluates expression trees into output buffers. Dispatches based on
 * expression::traits<Expr>::IsPermuted:
 *   - Contiguous: linear physical iteration, K::load/K::store
 *   - Permuted:   output-slice iteration with logical_flat tracking, K::gather
 */
#ifndef KERNEL_EVAL_H
#define KERNEL_EVAL_H

#include "config.h"
#include "fused/microkernels/microkernel_base.h"
#include "helper_traits.h"
#include "fused/padding_policies/simd_padding_policy.h"
#include "expression_traits/expression_traits.h"

namespace detail
{

    template <typename T, my_size_t Bits, typename Arch>
    struct KernelEval
    {
        using K = Microkernel<T, Bits, Arch>;
        static constexpr my_size_t simdWidth = K::simdWidth;

        // ========================================================================
        // Public dispatch
        // ========================================================================

        /**
         * @brief Dispatch: pick contiguous or permuted eval based on expression layout.
         */
        template <typename Expr>
        FORCE_INLINE static void eval(T *output, const Expr &expr) noexcept
        {
            if constexpr (!expression::traits<Expr>::IsPermuted)
            {
                // std::cout << "eval_contiguous" << std::endl;
                eval_vectorized_contiguous(output, expr);
            }
            else
            {
                // std::cout << "eval_permuted" << std::endl;
                eval_vectorized_permuted(output, expr);
            }
        }

    private:
        // ========================================================================
        // OutputPadPolicy — derive output padding from permuted expression dims
        // ========================================================================

        template <typename Expr, typename Seq>
        struct OutputPadImpl
        {
        };

        template <typename Expr, my_size_t... Is>
        struct OutputPadImpl<Expr, index_seq<Is...>>
        {
            using type = SimdPaddingPolicy<typename Expr::value_type, Expr::Dim[Is]...>;
        };

        template <typename Expr>
        struct OutputPadPolicy
        {
            using type = typename OutputPadImpl<Expr, typename make_index_seq<Expr::NumDims>::type>::type;
        };

        // ========================================================================
        // Contiguous path
        // ========================================================================

        /**
         * @brief CONTIGUOUS PATH (identity layout) — no permutation, no remapping.
         *
         * Iterates entire physical buffer linearly. PhysicalSize is guaranteed
         * to be a multiple of simdWidth by the padding policy. Padding
         * slots contain zeros — harmless for element-wise ops.
         *
         * evalu receives physical flat offsets.
         *
         * Note: FusedTensorND::evalu works for both paths because with identity layout,
         * physical flat and logical flat are the same within each slice
         * (the contiguous path passes physical offsets,
         * the permuted path passes logical flats,
         * same numbers when unpermuted).
         */
        template <typename Expr>
        FORCE_INLINE static void eval_vectorized_contiguous(
            T *output,
            const Expr &expr) noexcept
        {
            using Layout = typename Expr::Layout;
            static constexpr my_size_t physicalSize = Layout::PhysicalSize;
            static constexpr my_size_t simdSteps = physicalSize / simdWidth;
            static constexpr bool hasRemainder = (physicalSize % simdWidth) != 0;

            // Paranoia check: ensure physical size is a multiple of SIMD width,
            // so we never read out of bounds
            static_assert(physicalSize % simdWidth == 0,
                          "PhysicalSize must be a multiple of SimdWidth");

            // SIMD loop
            for (my_size_t i = 0; i < simdSteps; ++i)
            {
                auto val = expr.template evalu<T, Bits, Arch>(i * simdWidth);
                K::store(output + i * simdWidth, val);
            }

            // Scalar remainder TODO: The whole point of padding is that PhysicalSize is already
            // a multiple of SimdWidth — so there's no scalar remainder
            // Delete this code if confirmed unnecessary
            if constexpr (hasRemainder)
            {
                std::cout << "Warning: Scalar evaluation for remainder elements." << std::endl;
                // for (my_size_t i = simdSteps * simdWidth; i < physicalSize; ++i)
                // {
                //     output[i] = expr.template evalu<T, 1, GENERICARCH>(i);
                // }
            }
        }

        // ========================================================================
        // Permuted path
        // ========================================================================

        /**
         * @brief PERMUTED PATH (any layout) — works with any layout, including permuted.
         * TODO: untested
         * Iterates output physical slices for aligned stores. Tracks a running
         * logical_flat for the expression. Permuted evalu uses K::gather,
         * unpermuted uses K::load — dispatch happens inside evalu, not here.
         *
         * ============================================================================
         * EXAMPLE: output [3,2] padded to [3,4], source [2,3] transposed
         * ============================================================================
         *
         *   lastDim=2, paddedLastDim=4, numSlices=3, simdSteps=0, scalarStart=0
         *
         *   slice 0: out_base=0,  logical_flat 0,1 → store at output[0], output[1]
         *   slice 1: out_base=4,  logical_flat 2,3 → store at output[4], output[5]
         *   slice 2: out_base=8,  logical_flat 4,5 → store at output[8], output[9]
         *
         *   Padding at output[2,3,6,7,10,11] untouched.
         * ============================================================================
         */
        template <typename Expr>
        FORCE_INLINE static void eval_vectorized_permuted(
            T *output,
            const Expr &expr) noexcept
        {
            using OutputPad = typename OutputPadPolicy<Expr>::type;

            static constexpr my_size_t lastDim = OutputPad::LastDim;
            static constexpr my_size_t paddedLastDim = OutputPad::PaddedLastDim;
            static constexpr my_size_t numSlices = OutputPad::PhysicalSize / paddedLastDim;

            static constexpr my_size_t simdSteps = lastDim / simdWidth;
            static constexpr my_size_t scalarStart = simdSteps * simdWidth;

            my_size_t logical_flat = 0;

            for (my_size_t slice = 0; slice < numSlices; ++slice)
            {
                const my_size_t out_base = slice * paddedLastDim;

                for (my_size_t i = 0; i < simdSteps; ++i)
                {
                    auto val = expr.template logical_evalu<T, Bits, Arch>(logical_flat);
                    K::store(output + out_base + i * simdWidth, val);
                    logical_flat += simdWidth;
                }

                if constexpr (scalarStart < lastDim)
                {
                    for (my_size_t i = scalarStart; i < lastDim; ++i)
                    {
                        output[out_base + i] = expr.template logical_evalu<T, 1, GENERICARCH>(logical_flat);
                        ++logical_flat;
                    }
                }
            }
        }
    };

} // namespace detail

#endif // KERNEL_EVAL_H
