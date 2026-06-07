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
#include "../prefix_cache/PrefixCacheStats.h"
#include "../mtp/MTPDepthController.h"
#include "../local_execution/orchestrators/RankOrchestrator.h"
#include "../../planning/MemoryPlanner.h"
#include "../../collective/ILocalTPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../loaders/ModelContext.h"
#include "../../interfaces/IMPIContext.h"
#include "../../utils/MPIContext.h"
#include "../../utils/Tokenizer.h"
#include "../moe/ExpertWeightTransfer.h"
#include <memory>
#include <mutex>
#include <atomic>
#include <optional>
#include <utility>

namespace llaminar2
{

    class MoERebalanceController;
    class IModelContext;
    struct ExpertReplicaSet;
    struct MoEExpertOverlayExecutionPlan;

    std::shared_ptr<MoEExpertParallelPlan> freezeMoEExpertOverlayPlanForModel(
        IModelContext &model_ctx,
        const std::shared_ptr<MoEExpertParallelPlan> &plan);

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

        /**
         * @brief Construct with injected runner for unit testing
         *
         * Allows injecting a mock IInferenceRunner to test prefill/decode
         * logic without loading a real model.
         *
         * @param config Orchestration configuration
         * @param plan Pre-built execution plan
         * @param runner Pre-built inference runner (takes ownership)
         */
        OrchestrationRunner(
            OrchestrationConfig config,
            RankExecutionPlan plan,
            std::unique_ptr<IInferenceRunner> runner);

        /**
         * @brief Construct with injected runner AND MPI context for unit testing
         *
         * Allows injecting both a mock IInferenceRunner and a mock IMPIContext
         * to test MPI coordination logic without real MPI.
         *
         * @param config Orchestration configuration
         * @param plan Pre-built execution plan
         * @param runner Pre-built inference runner (takes ownership)
         * @param mpi_ctx MPI context (shared ownership)
         */
        OrchestrationRunner(
            OrchestrationConfig config,
            RankExecutionPlan plan,
            std::unique_ptr<IInferenceRunner> runner,
            std::shared_ptr<IMPIContext> mpi_ctx);

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
        void setDecodeStepTokenBudget(int max_tokens) override;
        bool maybeApplyMoERebalance() override;

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
        PrefixRuntimeStateSnapshot prefixStateProbe() const override;

        // =====================================================================
        // IOrchestrationRunner: Advanced
        // =====================================================================

        const float *lastLogits() const override;
        void setStopTokens(const std::vector<int32_t> &stop_tokens) override;
        std::shared_ptr<ITokenizer> tokenizer() const override;
        const std::string &architecture() const override;
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

        /// Get MoE rebalance controller from the underlying DGO (if any)
        MoERebalanceController *moeRebalanceController() const;
        std::vector<MoERebalanceController *> moeRebalanceControllers() const;
        MoERebalanceController *moeRebalanceControllerForDomain(
            const std::string &domain_id) const;

        /// Apply rebalanced expert masks to DGO's cached MoEExpertComputeStages
        void applyMoEExpertMasks(
            const std::vector<std::vector<bool>> &masks,
            const ReceivedWeightsMap &received = {},
            const std::string &domain_id = {});

        /// Apply MoE masks to every local device runner when the underlying
        /// runner is a RankOrchestrator. Returns true if handled.
        bool applyMoEExpertMasksForAllLocalDevices(const MoERebalanceController &controller);

        /// Apply precomputed MoE masks to every local device runner when the
        /// underlying runner is a RankOrchestrator. Returns true if handled.
        bool applyMoEExpertMasksForAllLocalDevices(
            const std::vector<std::vector<std::vector<bool>>> &masks_by_participant,
            const std::string &domain_id = {});

        /// Set expert replica info for per-token dynamic dispatch
        void setExpertReplicaSet(const ExpertReplicaSet &replicas, int participant_id);

        /// Apply one dynamic MoE rebalance cycle, including bounded hot replicas.
        bool applyMoERebalanceWithReplicas(bool log_histogram_summary = false);

        /// Transfer packed weights for migrating experts via MPI.
        ReceivedWeightsMap transferExpertWeights(
            const std::vector<ExpertMigration> &manifest, int num_layers);

        /// Transfer pre-packed weights for replicated experts via MPI (non-destructive).
        ReceivedWeightsMap transferReplicaWeights(
            const ExpertReplicaSet &replicas, int num_layers);

        /// Release raw expert weight data after initial VNNI packing.
        /// @return Total bytes freed/released
        size_t releaseRawExpertWeights();

        // =====================================================================
        // IOrchestrationRunner: GPU-side Sampling
        // =====================================================================

        int sampleGreedyOnDevice() override;
        int sampleOnDevice(const SamplingParams &params) override;
        void setSkipLogitsGatherDecode(bool skip) override;
        void setSkipLogitsGatherPrefill(bool skip) override;
        void setSuppressTimeline(bool suppress) override;
        void setAccumulatePrefill(bool accumulate) override;
        void flushStageTimeline() override;
        void setSamplingParams(const SamplingParams &params) override;
        SamplingParams getSamplingParams() const { return active_sampling_params_; }
        SamplingParams getRecommendedSamplingParams() const override;
        std::string getStopThinkingPrompt() const override;
        ToolCallFormat getToolCallFormat() const override;

        // =====================================================================
        // MPI Worker Loop (for non-root ranks in server mode)
        // =====================================================================

        /**
         * @brief MPI command tags for rank coordination in server mode.
         *
         * Rank 0 broadcasts these tags to tell non-root ranks what to do.
         */
        enum class MPICommand : int32_t
        {
            CLEAR_CACHE = 1,         ///< Clear KV cache
            SET_SAMPLING = 2,        ///< Set sampling parameters (followed by SamplingParams broadcast)
            PREFILL = 3,             ///< Prefill (followed by token count + tokens)
            DECODE_STEP = 4,         ///< Run one decode step
            SKIP_LOGITS_DECODE = 5,  ///< Set skip-logits-gather for decode
            APPLY_MOE_REBALANCE = 6, ///< Apply dynamic MoE rebalance/hot replicas
            SHUTDOWN = 99            ///< Exit the worker loop
        };

        /**
         * @brief Run as MPI worker (non-root rank) in server mode.
         *
         * Blocks in a loop waiting for commands from rank 0. Participates
         * in inference collectives (allreduce) as directed by rank 0.
         * Returns when rank 0 sends SHUTDOWN command.
         */
        void runMPIWorkerLoop() override;

        /**
         * @brief Signal workers to exit their loops.
         */
        void shutdownMPIWorkers() override;
        void setMoEExpertOverlayMPIContext(std::shared_ptr<IMPIContext> mpi_ctx) override
        {
            moe_expert_overlay_mpi_ctx_ = std::move(mpi_ctx);
        }

        /**
         * @brief Enable MPI coordinated mode.
         *
         * When enabled, rank 0 broadcasts commands before inference operations
         * so worker ranks can participate in lockstep. Must be enabled on rank 0
         * before workers enter runMPIWorkerLoop().
         *
         * Only server/interactive modes use this. Modes where all ranks run
         * the same code path (SingleShotChat, Completion) do NOT enable this.
         */
        void setMPICoordinatedMode(bool enabled) override { mpi_coordinated_mode_ = enabled; }

        /**
         * @brief Broadcast an MPI command from rank 0 to all ranks.
         *
         * Only broadcasts when mpi_coordinated_mode_ is enabled.
         * Called internally before operations that require all ranks.
         */
        void broadcastCommand(MPICommand cmd);

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

        bool freezeMoEExpertOverlayPlanForLoadedModel();

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
         * @brief Validate and clamp context length against model capabilities
         *
         * Must be called AFTER loadWeights() (requires model_ctx_).
         * Logs a warning if the user-specified context length exceeds the
         * model's max_position_embeddings and clamps to the model max.
         *
         * @return true always (warnings only, never fails)
         */
        bool validateContextLength();

        /**
         * @brief Validate that the model fits in device memory
         *
         * Uses MemoryPlanner to check weight + KV cache + activation
         * memory against available device memory.
         *
         * @return true if model fits, false with error if not
         */
        bool validateMemoryPlan();

        /**
         * @brief Build compute graphs
         */
        bool buildComputeGraph();

        /**
         * @brief Initialize MPI context if needed
         */
        bool initializeMPI();

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
         * @brief Build compute graph for multi-device (LOCAL TP) execution
         */
        bool buildMultiDeviceComputeGraph();

        /**
         * @brief Build compute graph for local pipeline parallel execution
         *
         * Uses TreeToRunnerCompiler to create a PP pipeline from
         * plan_.local_pp_devices and plan_.local_pp_layer_boundaries.
         */
        bool buildLocalPPComputeGraph();

        /**
         * @brief Build compute graph for single-device execution
         */
        bool buildSingleDeviceComputeGraph();

        /**
         * @brief Print consolidated startup banner (rank 0 only).
         *
         * Called after all validation passes, before buildComputeGraph().
         * Consolidates topology, config, model, and preflight info.
         */
        void printStartupBanner();

        bool shouldUseMTPDecode() const;
        std::string mtpDecodeHardFailureReason() const;
        std::string mtpDecodeBypassReason() const;
        void recordMTPBypass(const std::string &reason);
        bool ensureMTPDepthController(const MTPRuntimeConfig &mtp);
        int effectiveMTPMaxDraftDepth(const MTPRuntimeConfig &mtp) const;
        int currentMTPDraftDepth(const MTPRuntimeConfig &mtp);
        void recordMTPDepthObservation(
            int requested_depth,
            int effective_depth,
            int accepted_speculative_prefix,
            bool budget_limited,
            bool rollback);
        GenerationResult decodeStepMTP();
        bool forwardPrefillTokens(
            const int *tokens,
            int token_count,
            const std::string &failure_message);

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
        ClusterInventory cluster_inventory_;

        // Dependencies (injected or created)
        std::unique_ptr<IExecutionPlanBuilder> plan_builder_;
        std::shared_ptr<IMPIContext> mpi_ctx_;
        std::shared_ptr<IMPIContext> moe_expert_overlay_mpi_ctx_;
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
        SamplingParams active_sampling_params_;                         // Current sampling params for decodeStep()
        SamplingParams recommended_sampling_params_;                    // Model-specific defaults
        std::string stop_thinking_prompt_;                              // Model-specific stop-thinking prompt
        ToolCallFormat tool_call_format_{ToolCallFormat::HERMES_2_PRO}; // Model-specific tool call format
        int32_t last_token_{0};                                         // Last token for decode step
        bool prefill_logits_ready_{false};                              // True when terminal logits already predict the next decode token
        std::optional<int32_t> ready_sampled_token_;                    // Token already sampled from ready terminal logits
        std::optional<SamplingParams> ready_sampled_params_;            // Sampling params used to produce ready_sampled_token_
        int decode_step_token_budget_{0};                               // Optional per-step cap used by generate(); 0 means unlimited
        bool mpi_coordinated_mode_{false};                              // When true, rank 0 broadcasts commands for worker loop
        std::shared_ptr<ITokenizer> tokenizer_;
        MTPStats mtp_stats_;
        std::unique_ptr<MTPDepthController> mtp_depth_controller_;
        PrefillChunkStats prefill_chunk_stats_;
        PrefixCacheRequestSummary prefix_request_summary_;
        bool mtp_bypassed_{false};
        bool mtp_bypass_recorded_for_request_{false};
        std::string mtp_bypass_reason_;
    };

} // namespace llaminar2
