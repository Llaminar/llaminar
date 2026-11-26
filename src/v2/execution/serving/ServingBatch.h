/**
 * @file ServingBatch.h
 * @brief Request and event types of the runner-owned continuous-batching
 *        serving session.
 *
 * `OrchestrationRunner` is the sole lifecycle authority: callers submit
 * immutable prepared requests and drive shared execution steps; the runner
 * owns slot admission, per-row sampling, stop handling, cancellation and
 * retirement. A step advances every resident request together, and a new
 * request enters a free slot while the others keep decoding.
 */
#pragma once

#include "utils/Sampler.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Typed submission outcome (design §5.4 PKV-05). */
    enum class ServingSubmissionOutcome : std::uint8_t
    {
        Direct,  ///< Admitted into a resident slot immediately (free slot).
        Queued,  ///< Accepted into the bounded queue; a slot opens later.
        Refused, ///< Rejected typed (invalid, duplicate id, sealed, full queue);
                 ///< the caller's response incarnation is never minted.
    };

    /** @brief Immutable prepared request submitted to the serving session. */
    struct ServingBatchRequest
    {
        std::uint64_t id = 0;              ///< Caller-unique request identity.
        std::vector<int32_t> prompt;       ///< Non-empty prompt token IDs.
        SamplingParams sampling;           ///< Request-local sampling law.
        int max_new_tokens = 0;            ///< Positive output-token budget.
        std::vector<int32_t> stop_tokens;  ///< Tokens that terminate the request.
        /**
         * @brief Output pacing: when true the request produces tokens only
         *        against explicit grants (grantServingTokens) or forced tokens
         *        (forceServingToken); its prompt still prefills immediately.
         *        When false it decodes until it finishes.
         */
        bool paced = false;
        /**
         * @brief Diagnostic: report a digest of the exact logits bytes behind
         *        every token decision (ServingBatchEvent::logit_digests), to
         *        prove batch composition never changes a request's arithmetic.
         */
        bool record_logit_digests = false;
    };

    /** @brief Why a serving request stopped producing tokens. */
    enum class ServingFinishReason : std::uint8_t
    {
        None,      ///< Still running.
        Stop,      ///< A stop token was produced (it is not emitted).
        Length,    ///< The output budget or context capacity was reached.
        Cancelled, ///< The caller cancelled the request.
        Error      ///< Admission or execution failed for this request.
    };

    /** @brief Output of one request during one shared execution step. */
    /** @brief Serving session occupancy (resident slots in use, waiting requests). */
    struct ServingBatchOccupancy
    {
        int resident = 0;
        int queued = 0;
    };

    struct ServingBatchEvent
    {
        std::uint64_t id = 0;
        /// Tokens committed during this step. A Stop finish ends with the
        /// sampled stop token (as the scalar decodeStep() returns it).
        std::vector<int32_t> tokens;
        ServingFinishReason finish = ServingFinishReason::None;
        int prompt_tokens = 0;        ///< Prompt length (reported once finished).
        int cached_prompt_tokens = 0; ///< Prompt tokens restored from the prefix cache.
        bool prefix_cache_enabled = false; ///< The request was admitted through the prefix cache.
        int prefix_matched_blocks = 0;     ///< Prefix-cache blocks restored into the slot.
        /// FNV-1a digests of the logits bytes of each decision in this step
        /// (only with ServingBatchRequest::record_logit_digests).
        std::vector<std::uint64_t> logit_digests;
        int completion_tokens = 0;    ///< Total committed output tokens so far.
        std::string error;            ///< Diagnostic when finish == Error.

        [[nodiscard]] bool finished() const noexcept
        {
            return finish != ServingFinishReason::None;
        }
    };
} // namespace llaminar2
