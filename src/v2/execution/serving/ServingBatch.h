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
    /** @brief Immutable prepared request submitted to the serving session. */
    struct ServingBatchRequest
    {
        std::uint64_t id = 0;              ///< Caller-unique request identity.
        std::vector<int32_t> prompt;       ///< Non-empty prompt token IDs.
        SamplingParams sampling;           ///< Request-local sampling law.
        int max_new_tokens = 0;            ///< Positive output-token budget.
        std::vector<int32_t> stop_tokens;  ///< Tokens that terminate the request.
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
    struct ServingBatchEvent
    {
        std::uint64_t id = 0;
        std::vector<int32_t> tokens;  ///< Tokens committed during this step.
        ServingFinishReason finish = ServingFinishReason::None;
        int prompt_tokens = 0;        ///< Prompt length (reported once finished).
        int completion_tokens = 0;    ///< Total committed output tokens so far.
        std::string error;            ///< Diagnostic when finish == Error.

        [[nodiscard]] bool finished() const noexcept
        {
            return finish != ServingFinishReason::None;
        }
    };
} // namespace llaminar2
