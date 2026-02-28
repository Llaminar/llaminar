/**
 * @file OrchestrationRunner.h
 * @brief Implementation of IOrchestrationRunner for orchestrated inference
 *
 * Concrete implementation that wires together all orchestration components:
 * - OrchestrationConfig (Phase 0) - user configuration
 * - RankExecutionPlan (Phase 1-2) - what this rank should do
 * - PipelineParallelGraphBuilder (Phase 3) - PP stage insertion
 * - ILocalTPContext (Phase 4) - LOCAL TP collective operations
 *
 * This class owns:
 * - The execution plan for this rank
 * - Model weights (partial load for PP, sharded for TP)
 * - Compute graphs (with PP Send/Recv stages)
 * - LOCAL TP context for intra-rank collectives
 * - KV cache state
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "IOrchestrationRunner.h"
#include "../mpi_orchestration/IExecutionPlanBuilder.h"
#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "../mpi_orchestration/DeviceInventory.h"
#include "../local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "../../collective/ILocalTPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../loaders/ModelContext.h"
#include "../../utils/MPIContext.h"
#include "../../utils/Tokenizer.h"
#include <memory>
#include <mutex>
#include <atomic>

namespace llaminar2
{

    /**
     * @brief Concrete implementation of IOrchestrationRunner
     *
     * Manages the full lifecycle of orchestrated inference for a single MPI rank.
     * Each rank has its own OrchestrationRunner with its own execution plan.
     *
     * Initialization flow:
     * 1. Receive OrchestrationConfig (from factory)
     * 2. Build RankExecutionPlan using IExecutionPlanBuilder
     * 3. Set up ILocalTPContext (if LOCAL TP enabled)
     * 4. Partial-load weights for assigned layers
     * 5. Build compute graphs (with PP communication stages)
     *
     * Inference flow:
     * 1. prefill(): Process prompt tokens
     *    - First PP stage: embed + all assigned layers
     *    - Middle PP stages: receive → layers → send
     *    - Last PP stage: layers + LM head
     * 2. decodeStep(): Generate one token
     *    - Same flow as prefill but with seq_len=1
     *    - Last stage samples and broadcasts token to all
     *
     * Pipeline parallel communication:
     * - Send/Recv at PP boundaries using MPI
     * - Only activations are transferred (not weights)
     * - Synchronous (no pipelining in this implementation)
     */
    class OrchestrationRunner : public IOrchestrationRunner
    {
    public:
        /**
         * @brief Construct from configuration
         *
         * Does NOT initialize - call initialize() after construction.
         *
         * @param config Orchestration configuration
         * @param plan_builder Plan builder for creating execution plans
         */
        OrchestrationRunner(
            OrchestrationConfig config,
            std::unique_ptr<IExecutionPlanBuilder> plan_builder);

        /**
         * @brief Construct with pre-built execution plan
         *
         * For testing: allows injecting a known execution plan.
         *
         * @param config Orchestration configuration
         * @param plan Pre-built execution plan for this rank
         */
        OrchestrationRunner(
            OrchestrationConfig config,
            RankExecutionPlan plan);

        ~OrchestrationRunner() override;

        // Disable copy
        OrchestrationRunner(const OrchestrationRunner &) = delete;
        OrchestrationRunner &operator=(const OrchestrationRunner &) = delete;

        // =====================================================================
        // IOrchestrationRunner: Lifecycle
        // =====================================================================

        bool initialize() override;
        void shutdown() override;

        // =====================================================================
        // IOrchestrationRunner: Inference
        // =====================================================================

        bool prefill(const std::vector<int32_t> &tokens) override;
        GenerationResult decodeStep() override;
        GenerationResult generate(
            const std::vector<int32_t> &prompt_tokens,
            int max_new_tokens,
            const SamplingParams &sampling) override;

        // =====================================================================
        // IOrchestrationRunner: Configuration
        // =====================================================================

        const RankExecutionPlan &executionPlan() const override;
        const OrchestrationConfig &config() const override;

        // =====================================================================
        // IOrchestrationRunner: Status
        // =====================================================================

        bool isInitialized() const override;
        const std::string &lastError() const override;
        int vocabSize() const override;
        int currentPosition() const override;
        void clearCache() override;

        // =====================================================================
        // IOrchestrationRunner: Advanced
        // =====================================================================

        const float *lastLogits() const override;
        void setStopTokens(const std::vector<int32_t> &stop_tokens) override;
        std::shared_ptr<ITokenizer> tokenizer() const override;

        // =====================================================================
        // IOrchestrationRunner: Snapshot API
        // =====================================================================

        void enableSnapshotCapture(const std::string &output_dir = "") override;
        void disableSnapshotCapture() override;
        void clearSnapshots() override;
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;
        std::vector<std::string> getSnapshotKeys() const override;

        // =====================================================================
        // IOrchestrationRunner: Profiling
        // =====================================================================

        const GraphExecutorStats *executorStats() const override;
        void resetExecutorStats() override;

        // =====================================================================
        // IOrchestrationRunner: GPU-side Sampling
        // =====================================================================

        int sampleGreedyOnDevice() override;
        int sampleOnDevice(const SamplingParams &params) override;
        void setSkipLogitsGatherDecode(bool skip) override;
        void setSkipLogitsGatherPrefill(bool skip) override;
        void setSamplingParams(const SamplingParams &params) override;

    private:
        // =====================================================================
        // Initialization Helpers
        // =====================================================================

        /**
         * @brief Build execution plan from config
         */
        bool buildExecutionPlan();

        /**
         * @brief Gather cluster inventory for plan building
         */
        ClusterInventory gatherClusterInventory();

        /**
         * @brief Set up LOCAL TP context if enabled
         */
        bool setupLocalTPContext();

        /**
         * @brief Set up LOCAL PP context if enabled
         */
        bool setupLocalPPContext();

        /**
         * @brief Load model weights (partial for PP, sharded for TP)
         */
        bool loadWeights();

        /**
         * @brief Validate TP/PP configuration against loaded model
         *
         * Checks that the requested parallelism configuration is compatible
         * with the model's architecture (head counts, dimensions, layer count).
         *
         * Must be called AFTER loadWeights() (requires model_ctx_).
         *
         * @return true if configuration is valid, false with error message if not
         */
        bool validateTPPPConfiguration();

        /**
         * @brief Build compute graphs
         */
        bool buildComputeGraph();

        /**
         * @brief Initialize MPI context if needed
         */
        bool initializeMPI();

        // =====================================================================
        // PP Communication Helpers
        // =====================================================================

        /**
         * @brief Check if this is the first PP stage
         */
        bool isPipelineHead() const;

        /**
         * @brief Check if this is the last PP stage
         */
        bool isPipelineTail() const;

        /**
         * @brief Send activations to next PP stage
         */
        void sendActivationsToNextStage();

        /**
         * @brief Receive activations from previous PP stage
         */
        void receiveActivationsFromPrevStage();

        // =====================================================================
        // Multi-Device Helpers
        // =====================================================================

        /**
         * @brief Check if LOCAL TP is configured for this rank
         *
         * @return true if plan has multiple LOCAL TP devices
         */
        bool hasLocalTP() const;

        /**
         * @brief Build multi-device configuration from execution plan
         *
         * Converts RankExecutionPlan LOCAL TP settings to MultiDeviceOrchestrator::Config.
         *
         * @return Multi-device configuration
         */
        MultiDeviceOrchestrator::Config buildMultiDeviceConfig() const;

        /**
         * @brief Build compute graph for multi-device (LOCAL TP) execution
         */
        bool buildMultiDeviceComputeGraph();

        /**
         * @brief Build compute graph for single-device execution
         */
        bool buildSingleDeviceComputeGraph();

        // =====================================================================
        // Error Handling
        // =====================================================================

        /**
         * @brief Set error message and return false
         */
        bool setError(const std::string &error);

        // =====================================================================
        // State
        // =====================================================================

        // Configuration
        OrchestrationConfig config_;
        RankExecutionPlan plan_;
        bool plan_built_{false};

        // Dependencies (injected or created)
        std::unique_ptr<IExecutionPlanBuilder> plan_builder_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::unique_ptr<ILocalTPContext> local_tp_ctx_;
        std::unique_ptr<ILocalPPContext> local_pp_ctx_;

        // Execution infrastructure
        std::unique_ptr<IInferenceRunner> runner_;

        // Status
        std::atomic<bool> initialized_{false};
        std::string last_error_;
        mutable std::mutex error_mutex_;

        // Inference state
        std::vector<int32_t> stop_tokens_;
        Sampler sampler_;
        SamplingParams active_sampling_params_; // Current sampling params for decodeStep()
        int32_t last_token_{0};                 // Last token for decode step
        std::shared_ptr<ITokenizer> tokenizer_;
    };

} // namespace llaminar2
