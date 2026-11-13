/**
 * @file SamplingPenaltyRead.h
 * @brief Read-only candidate transformation shared by CUDA/ROCm Top-K selectors.
 *
 * Raw LM-head rows remain canonical model output. Selection applies the same
 * durable plus visible speculative history law to each loaded candidate, then
 * retains the existing reduction and distribution-normalization order.
 */
#pragma once
#include "SamplingMath.h"
#include "execution/mtp/MTPVerifierOutcomeGraph.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_PENALTY_READ_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_PENALTY_READ_HD inline
#endif

namespace llaminar2::sampling_math
{
    /** @brief Borrowed original device history; never an owner or copied histogram. */
    struct PenaltyLogitRead
    {
        const int *counts = nullptr;
        const MTPGreedyPenaltyPolicy *policy = nullptr;
        const int *tokens = nullptr;
    };

    /** @brief Read one candidate with the original serial penalty operation order.
     * @tparam WithHistory False compiles the raw selector without history accesses.
     * @param value Original unmodified logit.
     * @param token Vocabulary index of this candidate.
     * @param row Logical verifier row; only branch tokens through this row are visible.
     * @param history Original durable counts/policy and optional speculative branch.
     * @return Penalized candidate; caller never writes the source row. */
    template <bool WithHistory>
    LLAMINAR_PENALTY_READ_HD float historyLogit(
        float value, int token, int row, PenaltyLogitRead history)
    {
        if constexpr (WithHistory)
        {
            const auto request = *history.policy;
            if (request.enabled == 0)
                return value;
            const int encoded = history.counts[token];
            int count = TokenPenaltyHistory::generatedCount(encoded);
            if (history.tokens)
            {
                const int begin = request.first_token_already_in_history != 0 ? 1 : 0;
                for (int visible = begin; visible <= row; ++visible)
                    count += history.tokens[visible] == token ? 1 : 0;
            }
            if (count > 0 || (request.repetition_penalty != 1.0f &&
                             TokenPenaltyHistory::inPrompt(encoded)))
            {
                float penalty = 0.0f;
                if (count > 0 && request.presence_penalty != 0.0f)
                    penalty += request.presence_penalty;
                if (request.frequency_penalty != 0.0f)
                    penalty += request.frequency_penalty * static_cast<float>(count);
                value = penalizedLogit(value, request.repetition_penalty, penalty);
            }
        }
        return value;
    }
}

#undef LLAMINAR_PENALTY_READ_HD
