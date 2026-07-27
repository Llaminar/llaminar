/**
 * @file ParitySnapshotSelection.h
 * @brief Topology-aware snapshot selection for tensor-parallel parity tests.
 *
 * Tensor-parallel graph stages expose two different diagnostic products:
 * pre-collective partial tensors and explicit post-collective tensors whose
 * keys end in `_ALLREDUCED`. Local tensor parallelism performs the collective
 * among devices inside one process, so MPI cannot reconstruct the result.
 * Cross-rank tensor parallelism, by contrast, publishes one partial per MPI
 * rank and reconstructs the semantic tensor with an MPI sum in the harness.
 *
 * Keeping this decision in a small pure helper makes the distinction directly
 * unit-testable and prevents a one-rank MPI sum from being mislabeled as a
 * LocalTP allreduce.
 */

#pragma once

#include <string>

namespace llaminar2::test::parity
{
    /**
     * @brief Describes how the selected snapshot becomes a semantic tensor.
     */
    enum class ParitySnapshotReduction
    {
        Direct,       ///< Compare the selected snapshot exactly as published.
        CrossRankSum, ///< Sum one pre-collective partial from every MPI rank.
    };

    /**
     * @brief Snapshot key and reduction policy selected for one parity stage.
     */
    struct ParitySnapshotSelection
    {
        std::string key;
        ParitySnapshotReduction reduction = ParitySnapshotReduction::Direct;
        bool requires_post_collective_key = false;
    };

    /**
     * @brief Select the authoritative snapshot for one semantic parity stage.
     *
     * @param semantic_key Canonical key expected by the reference snapshots,
     *        such as `layer0_MOE_COMBINED_OUTPUT`.
     * @param stage_requires_reduction True when the fixture declares this stage
     *        to be a tensor-parallel partial before its production collective.
     * @param uses_in_process_local_tp True when the production collective spans
     *        multiple devices owned by this process.
     * @return Selection that names either the explicit post-collective LocalTP
     *         slot or the pre-collective cross-rank partial.
     *
     * LocalTP intentionally has no fallback to the semantic partial key. A
     * missing `_ALLREDUCED` slot means the captured production path was not
     * observed and must fail parity rather than silently testing different
     * arithmetic.
     */
    inline ParitySnapshotSelection selectParitySnapshot(
        const std::string &semantic_key,
        bool stage_requires_reduction,
        bool uses_in_process_local_tp)
    {
        if (!stage_requires_reduction)
        {
            return {
                .key = semantic_key,
                .reduction = ParitySnapshotReduction::Direct,
                .requires_post_collective_key = false,
            };
        }

        if (uses_in_process_local_tp)
        {
            return {
                .key = semantic_key + "_ALLREDUCED",
                .reduction = ParitySnapshotReduction::Direct,
                .requires_post_collective_key = true,
            };
        }

        return {
            .key = semantic_key,
            .reduction = ParitySnapshotReduction::CrossRankSum,
            .requires_post_collective_key = false,
        };
    }
} // namespace llaminar2::test::parity
