/**
 * @file MTPDecodeCatchup.h
 * @brief Shared decode-equivalent MTP catch-up contract.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llaminar2
{
    class IInferenceRunner;

    struct MTPDecodeCatchupGreedyRequest
    {
        std::vector<int32_t> draft_tokens;
        std::vector<int32_t> stop_tokens;
        int base_sidecar_position = 0;
        bool allow_speculative_discard = true;
        std::string verifier_path = "decode_equivalent_catchup";
    };

    struct MTPDecodeCatchupGreedyResult
    {
        bool ok = false;
        std::string error;

        std::vector<int32_t> accepted_tokens;
        std::vector<int32_t> verifier_tokens;

        bool all_speculative_accepted = true;
        bool stopped_on_output = false;
        int accepted_speculative_prefix = 0;
        int32_t rejected_verified_token = -1;
        int32_t ready_token = -1;

        int main_forward_token_count = 0;
        int shifted_commit_count = 0;
    };

    using MTPDecodeCatchupGreedySampler = std::function<int32_t()>;

    /**
     * @brief Run greedy MTP verification through normal one-token decode.
     *
     * This is the canonical shared catch-up implementation for stateful models
     * where batched all-position verifier rows are not known to leave mutable
     * KV/GDN/decode state equal to stepwise decode. CUDA and ROCm optimized
     * catch-up implementations must prove equivalence against this contract
     * before they are promoted.
     */
    MTPDecodeCatchupGreedyResult runSharedStepwiseMTPDecodeCatchupGreedy(
        IInferenceRunner &runner,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedySampler &sample_after_forward);

} // namespace llaminar2
