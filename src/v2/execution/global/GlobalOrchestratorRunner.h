/**
 * @file GlobalOrchestratorRunner.h
 * @brief IOrchestrationRunner adapter wrapping GlobalOrchestrator
 *
 * Enables GlobalOrchestrator (which implements IInferenceRunner) to be used
 * as an IOrchestrationRunner in AppContext. Follows the same delegation
 * pattern as OrchestrationRunner → IInferenceRunner, but wraps the
 * cross-machine GlobalOrchestrator instead of a local runner.
 *
 * Phase 4 scope: initialize() expects a pre-built GlobalOrchestrator
 * injected via Config. Full model-loading initialization is Phase 5+ work.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "../runner/IOrchestrationRunner.h"
#include "GlobalOrchestrator.h"
#include "../global_pp/GlobalPPTopology.h"
#include "../../config/OrchestrationConfig.h"
#include "../../interfaces/IMPIContext.h"
#include "../../utils/Sampler.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    class ITokenizer;
    class ModelContext;
    struct GraphExecutorStats;

    /**
     * @brief IOrchestrationRunner adapter wrapping GlobalOrchestrator
     *
     * Provides the IOrchestrationRunner lifecycle (initialize/shutdown) and
     * inference API (prefill/decodeStep/generate) on top of GlobalOrchestrator's
     * lower-level IInferenceRunner interface.
     *
     * The GlobalOrchestrator handles all cross-rank coordination (PP transfers,
     * TP collectives, token broadcast). This adapter just bridges the interface
     * gap and manages sampling state.
     */
    class GlobalOrchestratorRunner : public IOrchestrationRunner
    {
    public:
        // =================================================================
        // Configuration
        // =================================================================

        struct Config
        {
            OrchestrationConfig orchestration_config;
            GlobalPPTopology topology;
            std::shared_ptr<IMPIContext> mpi_ctx;

            // Pre-built GlobalOrchestrator (takes ownership)
            std::unique_ptr<GlobalOrchestrator> global_orchestrator;

            // Tokenizer (shared ownership)
            std::shared_ptr<ITokenizer> tokenizer;
        };

        // =================================================================
        // Construction / Destruction
        // =================================================================

        explicit GlobalOrchestratorRunner(Config config);
        ~GlobalOrchestratorRunner() override;

        // Non-copyable
        GlobalOrchestratorRunner(const GlobalOrchestratorRunner &) = delete;
        GlobalOrchestratorRunner &operator=(const GlobalOrchestratorRunner &) = delete;

        // =================================================================
        // IOrchestrationRunner: Lifecycle
        // =================================================================

        bool initialize() override;
        void shutdown() override;

        // =================================================================
        // IOrchestrationRunner: Inference
        // =================================================================

        bool prefill(const std::vector<int32_t> &tokens) override;
        GenerationResult decodeStep() override;
        GenerationResult generate(
            const std::vector<int32_t> &prompt_tokens,
            int max_new_tokens,
            const SamplingParams &sampling) override;

        // =================================================================
        // IOrchestrationRunner: Configuration
        // =================================================================

        const RankExecutionPlan &executionPlan() const override;
        const OrchestrationConfig &config() const override;

        // =================================================================
        // IOrchestrationRunner: Status
        // =================================================================

        bool isInitialized() const override;
        const std::string &lastError() const override;
        int vocabSize() const override;
        int currentPosition() const override;
        void clearCache() override;

        // =================================================================
        // IOrchestrationRunner: Advanced
        // =================================================================

        const float *lastLogits() const override;
        void setStopTokens(const std::vector<int32_t> &stop_tokens) override;
        std::shared_ptr<ITokenizer> tokenizer() const override;
        const std::string &architecture() const override;

        // =================================================================
        // IOrchestrationRunner: Snapshot API
        // =================================================================

        void enableSnapshotCapture(const std::string &output_dir) override;
        void disableSnapshotCapture() override;
        void clearSnapshots() override;
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;
        std::vector<std::string> getSnapshotKeys() const override;

        // =================================================================
        // IOrchestrationRunner: Profiling
        // =================================================================

        const GraphExecutorStats *executorStats() const override;
        void resetExecutorStats() override;

        // =================================================================
        // IOrchestrationRunner: GPU-side Sampling
        // =================================================================

        int sampleGreedyOnDevice() override;
        int sampleOnDevice(const SamplingParams &params) override;
        void setSkipLogitsGatherDecode(bool skip) override;
        void setSkipLogitsGatherPrefill(bool skip) override;
        void setSuppressTimeline(bool suppress) override;
        void setAccumulatePrefill(bool accumulate) override;
        void flushStageTimeline() override;
        void setSamplingParams(const SamplingParams &params) override;
        SamplingParams getRecommendedSamplingParams() const override;

        // =================================================================
        // IOrchestrationRunner: MPI Worker Loop
        // =================================================================

        void runMPIWorkerLoop() override;
        void shutdownMPIWorkers() override;
        void setMPICoordinatedMode(bool enabled) override;

        // =================================================================
        // Query API
        // =================================================================

        /** @brief Access the underlying GlobalOrchestrator (may be nullptr before init) */
        const GlobalOrchestrator *globalOrchestrator() const { return global_orch_.get(); }

        /** @brief Access the cluster topology */
        const GlobalPPTopology &topology() const { return config_.topology; }

    private:
        bool setError(const std::string &error);

        Config config_;
        std::unique_ptr<GlobalOrchestrator> global_orch_;

        // State
        bool initialized_ = false;
        std::string last_error_;
        std::string architecture_name_;
        RankExecutionPlan empty_plan_; ///< Placeholder — GlobalOrchestrator uses GlobalPPRankPlan instead

        // Sampling state
        Sampler sampler_;
        SamplingParams active_sampling_params_;
        std::vector<int32_t> stop_tokens_;
        int32_t last_token_ = 0;
        bool prefill_logits_ready_ = false;
        bool mpi_coordinated_mode_ = false;
    };

} // namespace llaminar2
