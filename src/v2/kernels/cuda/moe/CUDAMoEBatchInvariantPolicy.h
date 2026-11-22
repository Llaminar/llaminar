/**
 * @file CUDAMoEBatchInvariantPolicy.h
 * @brief Fixed arithmetic policy for byte-invariant CUDA MoE execution.
 *
 * CUDA grouped gate/up and down projections use ordered split-K reductions.
 * The partition count changes floating-point parenthesization even when launch
 * order is deterministic. A production-shaped exhaustive comparison found
 * that only the sixteen-partition tree matches the established serial-row
 * contract across grouped M. Partition count is therefore arithmetic, not a
 * tuning control. Geometry-only controls such as block width remain eligible
 * for learned dispatch because they do not alter the reduction tree.
 */

#pragma once

#include <cstdint>

#include "kernels/common/MoEProjectionNumericalContract.h"
#include "tensors/NativeQuantisedFormatInfo.h"

namespace llaminar2
{
    /**
     * @brief Compile-time CUDA MoE arithmetic invariants shared by launchers.
     *
     * Keeping these values outside `DebugEnv` makes a known non-equivalent
     * arithmetic tree unrepresentable in production. Direct down publication
     * assigns exactly one warp to each router slot; additional warps are idle
     * and strictly dominated, while fewer warps serialize independent routes.
     */
    struct CUDAMoEBatchInvariantPolicy final
    {
        /**
         * @brief Capture-time grouped projection engine.
         *
         * Runtime verifier groups contain at most the fifteen drafts plus one
         * target row admitted by the MTP contract. Their tiny row count favors
         * the independently certified ordered-DP4A launch. Larger prefill owns
         * enough row reuse to amortize a compact device directory and uses the
         * integer tensor-core implementation. This is a typed geometry policy,
         * not a fallback: both values are mandatory byte-exact production
         * engines and capture chooses exactly one before replay.
         */
        enum class GroupedProjectionEngine : uint8_t
        {
            OrderedDp4aVerifier = 0,
            TensorCoreImmaPrefill = 1,
        };

        static constexpr int gate_up_k_partitions =
            MoEProjectionNumericalContract::ordered_k_partitions;
        static constexpr int down_k_partitions =
            MoEProjectionNumericalContract::ordered_k_partitions;
        static constexpr int maximum_verifier_rows = 16;
        static constexpr int qfn_gate_up_n = 320; ///< QFN local expert projection width.
        static constexpr int qfn_gate_up_k = 2560; ///< QFN expert projection reduction width.

        /**
         * @brief Identify the packed codebooks measured with QFN's fixed K32 traversal.
         * @param codebook Physical NativeQuantised execution codebook.
         * @return Whether the SM86 verifier has a measured fixed-pitch packed reader.
         */
        static constexpr bool qfnVerifierPackedGateUp(uint8_t codebook) noexcept
        {
            using namespace native_quantised_formats;
            return codebook == Q2_K.codebook_id || codebook == Q3_K.codebook_id ||
                   codebook == Q6_K.codebook_id || codebook == IQ1_S.codebook_id ||
                   codebook == IQ1_M.codebook_id || codebook == IQ2_S.codebook_id ||
                   codebook == IQ2_XS.codebook_id || codebook == IQ2_XXS.codebook_id ||
                   codebook == IQ3_S.codebook_id || codebook == IQ3_XXS.codebook_id;
        }

        /**
         * @brief Select the mandatory grouped engine for a captured row bucket.
         * @param rows Positive grouped token-row count.
         * @return Ordered DP4A for verifier buckets, otherwise grouped IMMA.
         */
        static constexpr GroupedProjectionEngine groupedProjectionEngine(
            int rows) noexcept
        {
            return rows <= maximum_verifier_rows
                       ? GroupedProjectionEngine::OrderedDp4aVerifier
                       : GroupedProjectionEngine::TensorCoreImmaPrefill;
        }

        /** @return the unique non-dominated warp count for a valid top-k. */
        static constexpr int directDownWarps(int top_k) noexcept
        {
            return top_k;
        }
    };

    /** @brief Existing workspace planes for the fused M1 gate/up publication. */
    struct CUDAMoESwiGLUPublication
    {
        std::int8_t *codes = nullptr;
        float *scales = nullptr;
    };

    /** @brief Select the original FP32 input or the fused gate producer's Q8 output. */
    enum class CUDAMoEDecodeDownInput : std::uint8_t
    {
        FloatGateUp,
        PublishedSwiGLU,
    };
}
