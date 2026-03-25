/**
 * @file InferenceRunnerAdapter.cpp
 * @brief IOrchestrationRunner → IInferenceRunner adapter implementation
 */

#include "app/InferenceRunnerAdapter.h"

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

    void InferenceRunnerAdapter::flushStageTimeline()
    {
        orch_runner_->flushStageTimeline();
    }

} // namespace llaminar2
