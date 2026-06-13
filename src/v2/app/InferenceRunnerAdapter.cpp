/**
 * @file InferenceRunnerAdapter.cpp
 * @brief IOrchestrationRunner → IInferenceRunner adapter implementation
 */

#include "app/InferenceRunnerAdapter.h"

#include <utility>

namespace llaminar2
{

    InferenceRunnerAdapter::InferenceRunnerAdapter(IOrchestrationRunner *orch_runner)
        : orch_runner_(orch_runner), position_(0) {}

    bool InferenceRunnerAdapter::forward(const int *tokens, int seq_len)
    {
        std::vector<int32_t> token_vec(tokens, tokens + seq_len);
        bool result = orch_runner_->prefill(token_vec);
        if (result)
            position_ += seq_len;
        return result;
    }

    bool InferenceRunnerAdapter::supportsPrefillBatchForBenchmark(int request_batch) const
    {
        return orch_runner_ != nullptr && orch_runner_->supportsPrefillBatch(request_batch);
    }

    bool InferenceRunnerAdapter::prefillBatchForBenchmark(
        const std::vector<std::vector<int>> &token_batches)
    {
        if (!orch_runner_)
            return false;

        std::vector<std::vector<int32_t>> converted;
        converted.reserve(token_batches.size());
        for (const std::vector<int> &tokens : token_batches)
        {
            converted.emplace_back(tokens.begin(), tokens.end());
        }

        const bool result = orch_runner_->prefillBatch(converted);
        if (result && !converted.empty())
        {
            /*
             * BenchmarkRunner does not use get_position() for request-batched
             * decode accounting, but keep the adapter position meaningful for
             * request 0 diagnostics.
             */
            position_ += static_cast<int>(converted.front().size());
        }
        return result;
    }

    const float *InferenceRunnerAdapter::logits() const
    {
        return orch_runner_->lastLogits();
    }

    int InferenceRunnerAdapter::vocab_size() const
    {
        return orch_runner_->vocabSize();
    }

    void InferenceRunnerAdapter::clear_cache()
    {
        orch_runner_->clearCache();
        position_ = 0;
    }

    int InferenceRunnerAdapter::get_position() const
    {
        return position_;
    }

    ExecutionPath InferenceRunnerAdapter::executionPath() const
    {
        return ExecutionPath::GRAPH;
    }

    const char *InferenceRunnerAdapter::architecture() const
    {
        return "orchestrated";
    }

    const GraphExecutorStats *InferenceRunnerAdapter::executorStats() const
    {
        return orch_runner_->executorStats();
    }

    void InferenceRunnerAdapter::resetExecutorStats()
    {
        orch_runner_->resetExecutorStats();
    }

    int InferenceRunnerAdapter::sampleGreedyOnDevice()
    {
        return orch_runner_->sampleGreedyOnDevice();
    }

    int InferenceRunnerAdapter::sampleOnDevice(const SamplingParams &params)
    {
        return orch_runner_->sampleOnDevice(params);
    }

    bool InferenceRunnerAdapter::supportsDecodeStep() const
    {
        return orch_runner_ != nullptr;
    }

    bool InferenceRunnerAdapter::supportsDecodeStepBatchForBenchmark(int request_batch) const
    {
        return orch_runner_ != nullptr && orch_runner_->supportsDecodeStepBatch(request_batch);
    }

    void InferenceRunnerAdapter::setDecodeSamplingParams(const SamplingParams &params)
    {
        if (orch_runner_)
            orch_runner_->setSamplingParams(params);
    }

    void InferenceRunnerAdapter::setDecodeStepTokenBudget(int max_tokens)
    {
        if (orch_runner_)
            orch_runner_->setDecodeStepTokenBudget(max_tokens);
    }

    DecodeStepOutput InferenceRunnerAdapter::decodeStepForBenchmark()
    {
        if (!orch_runner_)
        {
            return DecodeStepOutput{{}, false, "orchestration runner unavailable"};
        }

        GenerationResult result = orch_runner_->decodeStep();
        return DecodeStepOutput{std::move(result.tokens), result.is_complete, std::move(result.error)};
    }

    DecodeBatchStepOutput InferenceRunnerAdapter::decodeBatchStepForBenchmark(int request_batch)
    {
        if (!orch_runner_)
        {
            return DecodeBatchStepOutput{{}, {}, "orchestration runner unavailable"};
        }

        GenerationBatchResult result = orch_runner_->decodeStepBatch(request_batch);
        DecodeBatchStepOutput output;
        output.error = std::move(result.error);
        output.tokens_by_request.reserve(result.requests.size());
        output.is_complete_by_request.reserve(result.requests.size());
        for (GenerationResult &request_result : result.requests)
        {
            if (output.error.empty() && !request_result.error.empty())
            {
                output.error = request_result.error;
            }
            output.tokens_by_request.push_back(std::move(request_result.tokens));
            output.is_complete_by_request.push_back(request_result.is_complete);
        }
        return output;
    }

    bool InferenceRunnerAdapter::maybeApplyDecodeBoundaryMaintenance()
    {
        return orch_runner_ ? orch_runner_->maybeApplyMoERebalance() : false;
    }

    void InferenceRunnerAdapter::setSkipLogitsGatherDecode(bool skip)
    {
        orch_runner_->setSkipLogitsGatherDecode(skip);
    }

    void InferenceRunnerAdapter::setSkipLogitsGatherPrefill(bool skip)
    {
        orch_runner_->setSkipLogitsGatherPrefill(skip);
    }

    void InferenceRunnerAdapter::setSuppressTimeline(bool suppress)
    {
        orch_runner_->setSuppressTimeline(suppress);
    }

    void InferenceRunnerAdapter::setAccumulatePrefill(bool accumulate)
    {
        orch_runner_->setAccumulatePrefill(accumulate);
    }

    void InferenceRunnerAdapter::flushStageTimeline()
    {
        orch_runner_->flushStageTimeline();
    }

    PrefixRuntimeStateSnapshot InferenceRunnerAdapter::prefixStateProbe() const
    {
        return orch_runner_ ? orch_runner_->prefixStateProbe() : PrefixRuntimeStateSnapshot{};
    }

} // namespace llaminar2
