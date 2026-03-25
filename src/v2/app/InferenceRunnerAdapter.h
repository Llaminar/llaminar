/**
 * @file InferenceRunnerAdapter.h
 * @brief IOrchestrationRunner → IInferenceRunner adapter
 *
 * Wraps an IOrchestrationRunner so it can be used by components that
 * expect the IInferenceRunner interface (ChatUI, BenchmarkRunner, etc.).
 */

#pragma once

#include "execution/runner/IOrchestrationRunner.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"

namespace llaminar2
{

    /**
     * @brief Adapts IOrchestrationRunner to IInferenceRunner
     *
     * Replaces the previous duplicate local adapter classes that were
     * defined inside main() (OrchestrationRunnerAdapter, BenchmarkRunnerAdapter).
     * This single adapter is the superset of both.
     */
    class InferenceRunnerAdapter : public IInferenceRunner
    {
    public:
        explicit InferenceRunnerAdapter(IOrchestrationRunner *orch_runner);

        // Core inference
        bool forward(const int *tokens, int seq_len) override;
        const float *logits() const override;
        int vocab_size() const override;
        void clear_cache() override;
        int get_position() const override;
        ExecutionPath executionPath() const override;
        const char *architecture() const override;

        // Profiling
        const GraphExecutorStats *executorStats() const override;
        void resetExecutorStats() override;

        // GPU-side sampling
        int sampleGreedyOnDevice() override;
        int sampleOnDevice(const SamplingParams &params) override;
        void setSkipLogitsGatherDecode(bool skip) override;
        void setSkipLogitsGatherPrefill(bool skip) override;
        void setSuppressTimeline(bool suppress) override;
        void flushStageTimeline() override;

    private:
        IOrchestrationRunner *orch_runner_;
        int position_;
    };

} // namespace llaminar2
