/**
 * @file MoEProjectionNumericalContract.h
 * @brief Backend-neutral arithmetic constants for movable MoE projections.
 *
 * ExpertOverlay may execute one expert on CUDA in one residency epoch and on
 * ROCm in the next.  Placement must not silently select a different numerical
 * program.  This contract names the arithmetic values that both backends must
 * consume while publishing route-local Q8 activations and projection rows.
 * Kernel-specific launch geometry remains a backend concern, but it cannot
 * replace these values with an algebraically similar expression whose rounding
 * differs across devices.
 */

#pragma once

#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Immutable numerical policy shared by heterogeneous MoE kernels.
     *
     * `q8_scale_multiplier` is the correctly rounded binary32 value of 1/127.
     * Publishing the scale with one explicit round-to-nearest multiplication
     * avoids relying on vendor division approximations for a value that is
     * subsequently persisted and reused by every projection K block.
     */
    struct MoEProjectionNumericalContract final
    {
        /** Number of source values represented by one NativeVNNI K block. */
        static constexpr int native_vnni_values_per_block = 32;

        static constexpr float q8_scale_multiplier = 0x1.020408p-7f;

        /**
         * Number of ordered K partitions in the heterogeneous projection tree.
         * This is arithmetic identity, not a backend launch-tuning control.
         */
        static constexpr int ordered_k_partitions = 16;

        /**
         * Number of ordered FP32 lanes in every movable floating projection.
         *
         * A CPU tier emulates these logical lanes; CUDA and ROCm map one lane
         * to one thread. The value is arithmetic identity rather than launch
         * tuning because changing it changes every dot-product parenthesis.
         */
        static constexpr int floating_ordered_k_partitions = 256;

        /**
         * @brief Report whether a source policy has a certified heterogeneous tree.
         *
         * Every source format accepted by the movable-expert registry uses the
         * same sixteen-partition tree. The physical decoder remains selected by
         * codebook, but placement cannot also select a backend-trained reduction
         * parenthesization. Reserved codebook ids remain rejected so a newly
         * admitted format must first join the registry and exhaustive witness.
         *
         * @param arithmetic_policy_codebook Canonical source-policy identifier.
         * @return True when this contract owns the policy's K reduction tree.
         */
        static constexpr bool ownsOrderedKPartitionTree(
            uint8_t arithmetic_policy_codebook) noexcept
        {
            return arithmetic_policy_codebook == 0u ||
                   (arithmetic_policy_codebook >= 4u &&
                    arithmetic_policy_codebook <= 17u) ||
                   (arithmetic_policy_codebook >= 19u &&
                    arithmetic_policy_codebook <= 21u);
        }

        /**
         * @brief Resolve the physical Q8 partition count for a reduction width.
         *
         * Small synthetic shapes can contain fewer than sixteen NativeVNNI
         * blocks.  Bounding the certified tree by the number of real blocks
         * preserves its ordering without launching semantically empty planes.
         * Invalid widths return zero and are rejected by the caller.
         *
         * @param reduction_width Number of source values reduced by the row.
         * @return Positive partition count for a valid aligned width, else zero.
         */
        static constexpr int orderedKPartitionsForWidth(
            int reduction_width) noexcept
        {
            if (reduction_width <= 0 ||
                (reduction_width % native_vnni_values_per_block) != 0)
            {
                return 0;
            }
            const int blocks =
                reduction_width / native_vnni_values_per_block;
            return blocks < ordered_k_partitions
                       ? blocks
                       : ordered_k_partitions;
        }
    };
}
