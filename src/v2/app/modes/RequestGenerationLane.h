/**
 * @file RequestGenerationLane.h
 * @brief One HTTP request's generation interface.
 *
 * The chat handler prepares a request and pulls tokens through a lane. The
 * direct lane drives the runner's scalar request path (one request at a time
 * on the serialized inference worker). The batched lane (ServingBatchDriver)
 * forwards the same operations into the runner-owned continuous-batching
 * session, so concurrent requests share execution while the handler code
 * stays identical. A lane is bound to exactly one request; destroying it
 * retires that request's generation state.
 */
#pragma once

#include "execution/runner/IOrchestrationRunner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Generation operations of one request (see file comment). */
    class IRequestGenerationLane
    {
    public:
        virtual ~IRequestGenerationLane() = default;

        /** @brief Start a fresh request (empty generation state). */
        virtual void beginRequest() = 0;

        /** @brief Install the request's sampling law (may throw on unsupported laws). */
        virtual void setSamplingParams(const SamplingParams &params) = 0;

        /** @brief Install the tokens that terminate the request. */
        virtual void setStopTokens(const std::vector<int32_t> &tokens) = 0;

        /** @brief Admit the prompt; false with lastError() on failure. */
        virtual bool prefill(const std::vector<int32_t> &tokens) = 0;

        /**
         * @brief Next committed tokens, at most @p max_tokens.
         *
         * An empty successful result means the request stopped (stop token).
         */
        virtual GenerationResult decodeStep(int max_tokens) = 0;

        /** @brief Commit @p token as the next output instead of sampling. */
        virtual GenerationResult forceDecodeToken(int32_t token) = 0;

        virtual std::string lastError() const = 0;

        /** @brief Publish diagnostic stage timelines (direct lane only). */
        virtual void flushStageTimeline() {}

        /** @brief Terminal runtime summary of this request, when available. */
        virtual RequestRuntimeSummary requestRuntimeSummary() const { return {}; }
    };

    /** @brief Lane over the runner's scalar request path (serialized worker). */
    class DirectGenerationLane final : public IRequestGenerationLane
    {
    public:
        explicit DirectGenerationLane(IOrchestrationRunner &runner) : runner_(runner) {}

        /** @brief Retires the request's generation state (request boundary). */
        ~DirectGenerationLane() override
        {
            try
            {
                runner_.clearCache();
            }
            catch (...)
            {
                // Cleanup failures surface on the next request's admission.
            }
        }

        DirectGenerationLane(const DirectGenerationLane &) = delete;
        DirectGenerationLane &operator=(const DirectGenerationLane &) = delete;

        void beginRequest() override { runner_.clearCache(); }
        void setSamplingParams(const SamplingParams &params) override { runner_.setSamplingParams(params); }
        void setStopTokens(const std::vector<int32_t> &tokens) override { runner_.setStopTokens(tokens); }
        bool prefill(const std::vector<int32_t> &tokens) override { return runner_.prefill(tokens); }

        GenerationResult decodeStep(int max_tokens) override
        {
            struct BudgetGuard
            {
                IOrchestrationRunner &runner;
                ~BudgetGuard() { runner.setDecodeStepTokenBudget(0); }
            };
            runner_.setDecodeStepTokenBudget(max_tokens);
            BudgetGuard guard{runner_};
            return runner_.decodeStep();
        }

        GenerationResult forceDecodeToken(int32_t token) override { return runner_.forceDecodeToken(token); }
        std::string lastError() const override { return runner_.lastError(); }
        void flushStageTimeline() override { runner_.flushStageTimeline(); }
        RequestRuntimeSummary requestRuntimeSummary() const override { return runner_.requestRuntimeSummary(); }

    private:
        IOrchestrationRunner &runner_;
    };
} // namespace llaminar2
