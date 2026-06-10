/**
 * @file MTPRejectionSampler.h
 * @brief Shared vLLM-style speculative rejection-sampling contract.
 *
 * The helpers in this file describe *what* speculative verification means.
 * CPU, CUDA, and ROCm implementations should match these semantics even when
 * they compute the distribution or row decision through backend-specific
 * kernels.
 */

#pragma once

#include "MTPDecodeCatchup.h"
#include "../../utils/Sampler.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Result for one target-verifier row in stochastic MTP.
     *
     * The row compares one draft token against the target distribution. On
     * accept, `token` is exactly the draft token. On reject, `token` is sampled
     * from the residual distribution `max(target - draft, 0)`, falling back to
     * the target distribution if the residual is empty.
     */
    struct MTPRejectionSampleRowResult
    {
        bool ok = false;
        std::string error;

        int32_t draft_token = -1;
        int32_t token = -1;
        bool accepted = false;
        float accept_probability = 0.0f;
        float accept_threshold = 0.0f;
    };

    /**
     * @brief Backend-neutral summary of one stochastic speculative batch.
     *
     * This object is intentionally close to the fixed-size GPU output contract:
     * row kernels decide each verifier row, then a tiny reduction determines how
     * many verifier rows are semantically consumed, which tokens should be
     * emitted, and whether a bonus ready token is available. The CPU helper uses
     * vectors for readability; CUDA/ROCm write the same fields into compact
     * device metadata buffers.
     */
    struct MTPRejectionBatchOutcome
    {
        bool ok = false;
        std::string error;

        std::vector<int32_t> output_tokens;
        std::vector<int32_t> verifier_tokens;
        int consumed_verifier_rows = 0;
        int accepted_speculative_prefix = 0;
        int target_verifier_state_commit_count = 0;
        int32_t ready_token = -1;
        int32_t rejected_verified_token = -1;
        bool stopped_on_output = false;
        bool all_speculative_accepted = true;
        bool sampled_terminal = false;
    };

    /**
     * @brief Sample one stochastic speculative-verifier row from distributions.
     *
     * This is the CPU/reference implementation of the same row-level contract
     * used by GPU speculative verification kernels. Thresholds are supplied by
     * the caller so tests and graph-captured GPU paths can be deterministic and
     * do not need to own a random-number generator.
     */
    MTPRejectionSampleRowResult sampleMTPRejectionRowFromDistributions(
        const std::vector<SamplingDistributionEntry> &target_distribution,
        const std::vector<SamplingDistributionEntry> &draft_distribution,
        int32_t draft_token,
        float accept_threshold,
        float residual_threshold);

    /**
     * @brief Sample a compact distribution with an explicit threshold.
     *
     * Used for the bonus-ready token after all draft rows are accepted. The
     * implementation intentionally matches `sampling_math` so host and device
     * lanes use the same cumulative-threshold rule.
     */
    int32_t sampleMTPDistributionWithThreshold(
        const std::vector<SamplingDistributionEntry> &distribution,
        float threshold);

    /**
     * @brief Summarize verified stochastic rows into the backend-neutral batch contract.
     *
     * `request.draft_tokens[0]` is the first target-model token that has already
     * been sampled. `verified_rows` begins at `request.draft_tokens[1]`. If all
     * verifier rows accept, `bonus_ready_token` must contain the target-model
     * ready token for the next decode step.
     */
    MTPRejectionBatchOutcome summarizeAllPositionMTPRejectionBatch(
        const MTPDecodeCatchupGreedyRequest &request,
        const std::vector<MTPRejectionSampleRowResult> &verified_rows,
        std::optional<int32_t> bonus_ready_token = std::nullopt);

    /**
     * @brief Convert stochastic verifier row decisions into the catch-up result.
     *
     * `request.draft_tokens[0]` is the already-accepted first target token.
     * `verified_rows[0]` verifies `request.draft_tokens[1]`, and so on. The
     * optional `bonus_ready_token` is only consumed when every speculative draft
     * row accepts and generation has not stopped.
     */
    MTPDecodeCatchupGreedyResult buildAllPositionMTPDecodeCatchupStochasticResult(
        const MTPDecodeCatchupGreedyRequest &request,
        const std::vector<MTPRejectionSampleRowResult> &verified_rows,
        std::optional<int32_t> bonus_ready_token = std::nullopt);

} // namespace llaminar2
