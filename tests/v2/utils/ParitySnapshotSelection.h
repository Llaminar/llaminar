/**
 * @file ParitySnapshotSelection.h
 * @brief Topology-aware snapshot selection for tensor-parallel parity tests.
 *
 * Tensor-parallel graph stages expose two different diagnostic products:
 * pre-collective partial tensors and explicit post-collective tensors whose
 * keys end in `_ALLREDUCED`. Some production runners expose that completed
 * collective directly, while legacy cross-rank runners expose one partial per
 * MPI rank and require evidence-only reconstruction in the harness. This is an
 * evidence-source distinction, not a topology distinction: a coordinated
 * production NodeTP worker cannot enter unrelated harness MPI collectives while
 * its peer remains inside the live command protocol.
 *
 * Keeping this decision in a small pure helper makes the distinction directly
 * unit-testable and prevents a one-rank MPI sum from being mislabeled as a
 * LocalTP allreduce.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test::parity
{
    /** @brief One exact column interval owned by a TP snapshot participant. */
    struct ParityTPColumnSlice
    {
        size_t start_column = 0;
        size_t column_count = 0;
    };

    /**
     * @brief Build a gap-free reference layout from live TP shard widths.
     *
     * Attention partitions retain whole heads, so participant widths need not
     * be equal when the head count is not divisible by the TP degree. The live
     * typed snapshot geometry is authoritative; this helper validates that it
     * covers the authenticated reference exactly before any tensor comparison.
     *
     * @param participant_widths Column count published by each participant in
     *        rank/device order.
     * @param reference_width Full semantic reference width.
     * @return Cumulative, non-overlapping reference intervals.
     * @throws std::invalid_argument for empty, zero-width, overflowing, or
     *         incomplete layouts.
     */
    inline std::vector<ParityTPColumnSlice> makeParityTPColumnSlices(
        const std::vector<size_t> &participant_widths,
        size_t reference_width)
    {
        if (participant_widths.empty() || reference_width == 0)
        {
            throw std::invalid_argument(
                "TP parity column layout requires participants and a reference width");
        }

        std::vector<ParityTPColumnSlice> slices;
        slices.reserve(participant_widths.size());
        size_t next_column = 0;
        for (const size_t width : participant_widths)
        {
            if (width == 0 ||
                width > std::numeric_limits<size_t>::max() - next_column ||
                next_column + width > reference_width)
            {
                throw std::invalid_argument(
                    "TP parity shard widths do not fit the reference width");
            }
            slices.push_back({
                .start_column = next_column,
                .column_count = width,
            });
            next_column += width;
        }
        if (next_column != reference_width)
        {
            throw std::invalid_argument(
                "TP parity shard widths do not cover the reference exactly");
        }
        return slices;
    }

    /**
     * @brief Derive one checkpoint's semantic width from its authenticated data.
     *
     * Model families may widen a named projection (for example, a fused query
     * gate) without changing the semantic checkpoint name. Reference tensor
     * cardinality is therefore the shape authority; copied model-specific width
     * tables are not.
     *
     * @param reference_elements Total elements in the reference checkpoint.
     * @param rows Authenticated sequence/token row count.
     * @return Exact columns per row.
     * @throws std::invalid_argument when the tensor cannot represent a dense
     *         row-major matrix with the requested row count.
     */
    inline size_t parityTPReferenceColumnCount(
        size_t reference_elements,
        size_t rows)
    {
        if (rows == 0 || reference_elements == 0 ||
            reference_elements % rows != 0)
        {
            throw std::invalid_argument(
                "TP parity reference cardinality is not divisible by its rows");
        }
        return reference_elements / rows;
    }

    /**
     * @brief Describes how the selected snapshot becomes a semantic tensor.
     */
    enum class ParitySnapshotReduction
    {
        Direct,       ///< Compare the selected snapshot exactly as published.
        CrossRankSum, ///< Sum one pre-collective partial from every MPI rank.
    };

    /**
     * @brief Authority from which a reduced semantic checkpoint is obtained.
     */
    enum class ParityCollectiveEvidenceSource
    {
        CrossRankPartials,      ///< Harness sums live pre-collective rank partials.
        PostCollectiveSnapshot, ///< Production graph publishes the completed tensor.
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
     * @param evidence_source Authority used to obtain the completed collective
     *        value for stages that require reduction.
     * @return Selection that names either the explicit post-collective slot or
     *         the pre-collective cross-rank partial.
     *
     * Post-collective evidence intentionally has no fallback to the semantic
     * partial key. A
     * missing `_ALLREDUCED` slot means the captured production path was not
     * observed and must fail parity rather than silently testing different
     * arithmetic.
     */
    inline ParitySnapshotSelection selectParitySnapshot(
        const std::string &semantic_key,
        bool stage_requires_reduction,
        ParityCollectiveEvidenceSource evidence_source)
    {
        if (!stage_requires_reduction)
        {
            return {
                .key = semantic_key,
                .reduction = ParitySnapshotReduction::Direct,
                .requires_post_collective_key = false,
            };
        }

        if (evidence_source ==
            ParityCollectiveEvidenceSource::PostCollectiveSnapshot)
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
