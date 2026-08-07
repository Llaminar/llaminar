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
#include <cstdio>
#include <optional>
#include <unordered_map>
#include <utility>

namespace llaminar2
{

    class MoERebalanceController;
    class IModelContext;
    struct ExpertReplicaSet;
    struct MoEExpertOverlayExecutionPlan;

    std::shared_ptr<MoERoutedExpertPlacementPlan> freezeMoEExpertOverlayPlanForModel(
        IModelContext &model_ctx,
        const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan);

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
        bool initializeForDryRun() override;
        void shutdown() override;

        // =====================================================================
        // IOrchestrationRunner: Inference
        // =====================================================================

        bool prefill(const std::vector<int32_t> &tokens) override;
        bool supportsPrefillBatch(int request_batch) const override;
        bool prefillBatch(
            const std::vector<std::vector<int32_t>> &token_batches) override;
        GenerationResult decodeStep() override;
        GenerationResult forceDecodeToken(int32_t token) override;
        bool supportsDecodeStepBatch(int request_batch) const override;
        GenerationBatchResult decodeStepBatch(int request_batch) override;
        GenerationResult generate(
            const std::vector<int32_t> &prompt_tokens,
            int max_new_tokens,
            const SamplingParams &sampling) override;
        void setDecodeStepTokenBudget(int max_tokens) override;
        bool maybeApplyMoERebalance() override;
        bool usesDeviceSideMoERebalanceController() const override;
        uint64_t moeRuntimeMovementEpoch() const override;

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
        void drainCompletedDecodeBoundaryMaintenanceDiagnostics() override;
        PrefixRuntimeStateSnapshot prefixStateProbe() const override;
        DeviceId primaryDeviceId() const override;

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
        void setSnapshotCaptureFilter(const std::vector<std::string> &keys) override;
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
        bool applyMoEExpertMasksForAllLocalDevices(
            const MoERebalanceController &controller,
            const ExpertReplicaSet *replica_arrivals = nullptr,
            const std::vector<int> *previous_ownership_placement = nullptr);

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
        bool waitForLastForwardCompletionForBenchmark() override;
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
            DECODE_STEP = 4,         ///< Run one decode step (followed by int32 token budget)
            SKIP_LOGITS_DECODE = 5,  ///< Set skip-logits-gather for decode
            APPLY_MOE_REBALANCE = 6, ///< Apply dynamic MoE rebalance/hot replicas
            FORCE_DECODE_TOKEN = 7,  ///< Commit a forced token (followed by token id)
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
        bool loadWeights(bool prepopulate_page_cache = true);

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
        void printStartupBanner(FILE *stream = stderr);

        bool shouldUseMTPDecode() const;
        std::string mtpDecodeHardFailureReason() const;
        std::string mtpDecodeBypassReason() const;
        void recordMTPBypass(const std::string &reason);
        bool ensureMTPDepthController(const MTPRuntimeConfig &mtp);
        int effectiveMTPMaxDraftDepth(const MTPRuntimeConfig &mtp) const;
        int currentMTPDraftDepth(const MTPRuntimeConfig &mtp);
        /**
         * @brief Initialize the scheduler-owned MTP position after prefill.
         *
         * GPU prefill and prefix restore already know exactly how many request
         * tokens they committed.  Recording that scheduler fact here prevents
         * the first MTP transaction from consulting a backend host position
         * mirror before device-resident publication has produced its first
         * logical-state mailbox.
         *
         * @param committed_tokens Logical tokens consumed by the successful
         *        prefill transaction.
         * @param source Diagnostic name for the prefill path.
         * @return true when the position is valid and was initialized, or when
         *         the active lane does not require GPU MTP planning state.
         */
        bool initializeDecodeTransactionPlanningPositionAfterPrefill(
            int committed_tokens,
            const char *source);
        /**
         * @brief Admit the scalar GPU generation ledger exactly once.
         *
         * Every public inference surface ultimately enters MTP through
         * decodeStepMTP(), whereas only the convenience generate() API owns an
         * outer generation loop.  Keeping admission at this shared
         * prefill-to-decode boundary prevents benchmark, HTTP, completion, and
         * chat callers from reaching the first resident verifier without an
         * initialized device controller.
         *
         * The boolean tracked by OrchestrationRunner is control-plane lifecycle
         * state only.  Once admitted, transaction budgets, response progress,
         * stop state, and producer/consumer ordering remain exclusively in the
         * persistent device controller and its event-published handoffs.
         *
         * @param grouped_device_verify Whether this decode transaction uses the
         *        production GPU grouped verifier and resident publisher.
         * @return true when admission is unnecessary or has completed.
         */
        bool admitScalarDeviceResidentGeneration(
            bool grouped_device_verify,
            sampling_math::DeviceGenerationLeadingRowDisposition
                initial_leading_row_disposition);
        /**
         * @brief Admit one complete request-batched GPU generation ledger.
         *
         * The successful batched prefill boundary owns immutable prompt
         * lengths and request seeds, but response progress is never mirrored on
         * the host. This method resolves the common per-request response budget,
         * initializes every persistent controller row on the participant
         * stream, and closes the admission boundary before prefill logits are
         * sampled into resident condition slots.
         *
         * @param request_count Exact number of live request rows.
         * @return true only when every participant published its controller
         *         initialization event for the complete request set.
         */
        bool admitRequestBatchDeviceResidentGeneration(int request_count);
        /**
         * @brief Advance the scheduler position after one committed decode row.
         *
         * A normal decode forward consumes the previous response token and
         * therefore advances the logical position of the logits sampled next.
         * GPU lanes update the scheduler-owned control scalar here instead of
         * observing a backend KV position mirror.
         *
         * @param source Diagnostic name for the forward path.
         * @return true on success; false when a GPU forward has no initialized
         *         decode transaction position.
         */
        bool advanceDecodeTransactionPlanningPositionAfterForward(
            const char *source);
        /**
         * @brief Publish the scheduler position produced by an atomic MTP commit.
         *
         * The validated transaction base and committed-row count are sufficient
         * to advance the scheduler's control-plane coordinate.  Publication must
         * not depend on whether the backend currently exposes a transient
         * resident-outcome mailbox: the first post-prefill direct emit has no
         * outcome mailbox, yet it still commits one real device row.
         *
         * @param transaction_base_tokens Scheduler position at transaction entry.
         * @param committed_rows Number of main-graph rows atomically committed.
         * @param source Diagnostic name for the commit path.
         * @return true when the position was published, or when the active lane
         *         is CPU and therefore owns its position inside the CPU runner.
         */
        bool publishDecodeTransactionPlanningPositionAfterMTPCommit(
            int transaction_base_tokens,
            int committed_rows,
            const char *source);
        /**
         * @brief Run and surface one complete native device-generation parent.
         *
         * The externally orchestrated first transaction has already published
         * its compact outcome and accepted state. This method composes the
         * sampling-specific parent, launches it once, consumes one terminal
         * ledger, and updates host-visible response/statistics only after the
         * request is complete. No per-transaction outcome or controller state
         * crosses this boundary.
         *
         * @param publication_request First committed transaction and exact
         *        controller-owned verifier geometry.
         * @param sampling_mode Greedy or stochastic parent topology.
         * @param transaction_base_cached_tokens Scheduler position at entry.
         * @param requested_draft_depth Device-selected first-transaction depth.
         * @param capture_draft_depth Physical child-family capacity materialized
         *        by the first transaction.
         * @param result Current decode result to complete with terminal tokens.
         * @return Completed result, or an error suitable for atomic rollback.
         */
        GenerationResult completeDeviceResidentGeneration(
            const DeviceSpeculativePublicationRequest &publication_request,
            DeviceGenerationSamplingMode sampling_mode,
            int transaction_base_cached_tokens,
            int requested_draft_depth,
            int capture_draft_depth,
            GenerationResult result);
        /**
         * @brief Complete a request batch from one native device-generation parent.
         *
         * Transaction zero has already committed its controller-owned compact
         * outcome. This method materializes and launches the reusable parent,
         * validates one terminal ledger row per admitted request, and performs
         * the only response D2H boundary in the request lifetime. It never
         * reconstructs a compact transaction or advances live state on the host.
         *
         * @param publication_request Controller-owned first transaction.
         * @param sampling_mode Greedy or stochastic captured topology.
         * @param requested_draft_depth First transaction's selected depth.
         * @param capture_draft_depth Maximum child-family depth materialized.
         * @param result Output object whose request rows receive terminal tokens.
         * @return Complete terminal batch or a fatal lifecycle error.
         */
        GenerationBatchResult completeDeviceResidentBatchGeneration(
            const DeviceSpeculativePublicationRequest &publication_request,
            DeviceGenerationSamplingMode sampling_mode,
            int requested_draft_depth,
            int capture_draft_depth,
            GenerationBatchResult result);
        /**
         * @brief Publish device MoE maintenance before launching a future MTP consumer.
         *
         * Accepted-state publication and the next speculative sidecar use
         * independent GPU streams. Dynamic residency maintenance may flip the
         * active placement bank between those operations, so a sidecar must
         * never be submitted until the complete decode boundary has published
         * its maintenance completion event. This helper is the sole ordering gate
         * for resident sidecar prelaunches: it enqueues maintenance once for the
         * current decode step and records that the ordinary outer-loop boundary
         * must acknowledge, rather than replay, the same transaction.
         *
         * Non-device-controller lanes are unchanged because they do not have a
         * concurrent placement-bank publisher.  A failure is fatal to the
         * current decode step; launching the consumer against ambiguous expert
         * ownership is never an allowed fallback.
         *
         * @param consumer Human-readable consumer name used in diagnostics.
         * @return true when no device maintenance is required or its event has
         *         been published before the future consumer launch.
         */
        bool publishDeviceMoEMaintenanceBeforeMTPConsumer(
            const char *consumer);
        /**
         * @brief Resolve the scalar sidecar base position for MTP planning.
         *
         * GPU lanes must use the orchestration-owned transaction position
         * initialized by prefill and advanced from compact accepted-state
         * publication metadata. They fail closed if that control-plane value
         * is absent; reading `IInferenceRunner::get_position()` is a CPU-only
         * path and can never become a GPU coherence fallback.
         */
        std::optional<int> currentDecodeTransactionPositionForPlanning(
            const char *context,
            std::string *error = nullptr) const;
        void recordMTPDepthZeroBypass();
        void recordMTPDepthObservation(
            int requested_depth,
            int effective_depth,
            int accepted_speculative_prefix,
            bool budget_limited,
            bool rollback);
        GenerationResult decodeStepMTP();
        void clearBatchedDecodeState();
        /**
         * @brief Return true when routed prefill uses current-batch least-loaded assignment.
         *
         * Least-loaded route planning is deliberately a current-window policy:
         * the same token span can produce different route assignments if it is
         * planned inside a larger prefill transaction.  This helper centralizes
         * the typed domain-policy check so cache-enabled and cache-disabled
         * prefill paths share one request-boundary definition for KV, GDN, MTP,
         * and MoE runtime state. Durable residency maintenance is intentionally
         * irrelevant to this decision.
         *
         * @return true when an active routed domain explicitly selects
         *         LeastLoadedResident for ordinary prefill.
         */
        bool leastLoadedCurrentBatchPrefillUsesStableWindows() const;

        /**
         * @brief Resolve the stable current-batch routed-prefill transaction size.
         *
         * A configured @c assignment_window_tokens value is the first-class
         * owner of current-batch assignment state lifetime. Prefix cache may
         * supply an explicit block boundary only when that owner is unset and
         * cache restore is active for the request. Returning zero means one
         * ordinary prefill transaction remains in force.
         *
         * @param prefix_cache_block_size Coordinated prefix-cache block size for
         *        this request, or zero when prefix cache is not active.
         * @param prefix_cache_enabled Whether restored prefix state is active
         *        for this request.
         * @return Positive stable transaction size, or zero for no segmentation.
         */
        int stableRoutedPrefillAssignmentWindowTokens(
            int prefix_cache_block_size,
            bool prefix_cache_enabled) const;

        /**
         * @brief Forward one indivisible prefill transaction through the runner.
         *
         * This method owns the existing activation-graph bucket scheduling
         * contract for a single caller-visible prefill transaction.  Callers that
         * need higher-level semantic boundaries, such as prefix-cache LLEP block
         * boundaries, must split before entering this method.
         *
         * @param tokens Pointer to the first token in this transaction.
         * @param token_count Number of real tokens in this transaction.
         * @param failure_message User-visible context to attach to hard failures.
         * @return true when the runner completed the transaction successfully.
         */
        bool forwardSinglePrefillTransaction(
            const int *tokens,
            int token_count,
            const std::string &failure_message);

        /**
         * @brief Forward prefill tokens, optionally split by a stable transaction size.
         *
         * A positive @p stable_segment_tokens turns the call into a sequence of
         * indivisible prefill transactions of that size, plus one terminal
         * remainder.  The split is used for prefix-cache LLEP prefill so a fresh
         * cache miss, a partial-hit suffix, and an uncached full prompt all cross
         * identical semantic request boundaries.
         *
         * @param tokens Pointer to the first token to prefill.
         * @param token_count Number of tokens to prefill.
         * @param failure_message User-visible context to attach to hard failures.
         * @param stable_segment_tokens Stable transaction size, or zero to keep
         *        the historical single-transaction behavior.
         * @return true when every required prefill transaction succeeds.
         */
        bool forwardPrefillTokens(
            const int *tokens,
            int token_count,
            const std::string &failure_message,
            int stable_segment_tokens = 0);
        void recordMoERebalanceRawExpertRelease(
            const std::string &domain_id,
            const std::string &device);
        bool publishPendingMoERebalanceUpdate();
        bool publishPendingMoERebalanceBeforeForward(const char *reason);
        void drainPendingMoERebalanceBeforeCacheClear();
        void clearUnderlyingRunnerCacheAfterMoEPublish(const char *reason);

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
        /**
         * @brief Whether shutdown may resolve a physical backend for runner_.
         *
         * Production constructors keep this true. Constructors that accept an
         * injected IInferenceRunner are unit-test seams: their CUDA/ROCm-shaped
         * DeviceIds exercise policy only and must never initialize a physical
         * backend during teardown. Injected RankOrchestrator instances still
         * receive synchronizeDevices(); that object independently knows whether
         * its children own physical devices.
         */
        bool physical_runner_backend_access_enabled_ = true;
        /**
         * @brief Owns TP-combined snapshot views returned through the orchestration API.
         *
         * RankOrchestrator keeps per-device snapshots and can assemble a semantic
         * tensor-parallel view on demand.  IOrchestrationRunner callers only ask
         * for snapshot keys, so this cache gives the returned pointer a stable
         * lifetime without leaking local-TP topology details to parity tests.
         */
        mutable std::unordered_map<std::string, std::vector<float>> snapshot_combined_cache_;

        struct PendingMoERebalanceUpdate
        {
            RankOrchestrator *rank = nullptr;
            RankOrchestrator::PreparedMoEExpertMaskUpdate prepared;
            ExpertReplicaSet replica_set;
            int participant_id = 0;
            bool publish_replica_set = false;
            bool release_raw_after_publish = false;
            std::string domain_id;
            std::string device;
        };
        std::optional<PendingMoERebalanceUpdate> pending_moe_rebalance_prepare_;

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
        /**
         * @brief One already-sampled condition at a ready terminal-logits boundary.
         *
         * Ordinary CPU MTP may materialize a sampled token on the host before
         * the next public decode call. GPU resident generation instead ends at
         * an event-published mailbox containing an exact token/position pair.
         * The controller ledger separately records whether that row still belongs
         * to the next response or was already returned as a rejection correction.
         * Those states are deliberately represented by closed enums. In
         * particular, the device form cannot accidentally carry a host token,
         * and the host form cannot omit the concrete response token it owns.
         *
         * The optional resident handle on a host-visible condition is an
         * execution optimization: the served response still owns the concrete
         * token, while the next GPU sidecar consumes the same token and logical
         * position from its authenticated mailbox. A device-only condition is
         * different: its token has not crossed into a host execution authority,
         * so every next-step consumer remains on device until the following
         * terminal ledger materializes newly emitted response tokens.
         */
        struct ReadyMTPCondition
        {
            /** @brief Authority that owns the exact condition token. */
            enum class Authority
            {
                /** A concrete token has already crossed a response boundary. */
                HostVisible,
                /** Only an authenticated device publication owns the token. */
                DeviceResident,
            };

            Authority authority = Authority::HostVisible;
            std::optional<int32_t> host_token;
            SamplingParams sampling_params;
            std::optional<DeviceResidentLogicalSequenceStateHandle>
                resident_state;
            sampling_math::DeviceGenerationLeadingRowDisposition
                response_disposition =
                    sampling_math::DeviceGenerationLeadingRowDisposition::
                        PendingResponse;

            /**
             * @brief Construct a host-visible ready condition.
             * @param token Concrete token already sampled for the response.
             * @param params Sampling policy that produced @p token.
             * @param resident Optional device source for subsequent GPU work.
             */
            [[nodiscard]] static ReadyMTPCondition hostVisible(
                int32_t token,
                const SamplingParams &params,
                std::optional<DeviceResidentLogicalSequenceStateHandle>
                    resident = std::nullopt)
            {
                return ReadyMTPCondition{
                    .authority = Authority::HostVisible,
                    .host_token = token,
                    .sampling_params = params,
                    .resident_state = std::move(resident),
                    .response_disposition =
                        sampling_math::
                            DeviceGenerationLeadingRowDisposition::
                                PendingResponse,
                };
            }

            /**
             * @brief Construct a device-only continuation condition.
             * @param params Sampling policy embedded in the completed loop.
             * @param resident Exact terminal mailbox retained by the GPU.
             * @param disposition Whether the retained row is still owed to the
             *        response or was emitted by the preceding controller.
             */
            [[nodiscard]] static ReadyMTPCondition deviceResident(
                const SamplingParams &params,
                DeviceResidentLogicalSequenceStateHandle resident,
                sampling_math::DeviceGenerationLeadingRowDisposition
                    disposition)
            {
                return ReadyMTPCondition{
                    .authority = Authority::DeviceResident,
                    .host_token = std::nullopt,
                    .sampling_params = params,
                    .resident_state = std::move(resident),
                    .response_disposition = disposition,
                };
            }

            /** @return true when authority and payload form one legal state. */
            [[nodiscard]] bool valid() const noexcept
            {
                const bool resident_valid =
                    !resident_state.has_value() || resident_state->valid();
                switch (authority)
                {
                case Authority::HostVisible:
                    return host_token.has_value() && *host_token >= 0 &&
                           resident_valid &&
                           response_disposition ==
                               sampling_math::
                                   DeviceGenerationLeadingRowDisposition::
                                       PendingResponse;
                case Authority::DeviceResident:
                    return !host_token.has_value() &&
                           resident_state.has_value() &&
                           resident_state->coversRequest(0) &&
                           sampling_math::
                               valid_device_generation_leading_row_disposition(
                                   response_disposition);
                }
                return false;
            }

            /** @return true when no host scalar may source this condition. */
            [[nodiscard]] bool isDeviceResidentOnly() const noexcept
            {
                return authority == Authority::DeviceResident;
            }

            /** @return true when row zero was returned by the prior controller. */
            [[nodiscard]] bool wasAlreadyEmitted() const noexcept
            {
                return response_disposition ==
                       sampling_math::DeviceGenerationLeadingRowDisposition::
                           AlreadyEmitted;
            }
        };

        /**
         * @brief Whether current terminal logits already own the next decode edge.
         *
         * A missing @ref ready_mtp_condition_ means the token still needs to be
         * sampled from these logits, as after ordinary prefill. A populated
         * condition means sampling already happened under the explicitly typed
         * authority above and must never be repeated.
         */
        bool prefill_logits_ready_{false};
        std::optional<ReadyMTPCondition> ready_mtp_condition_;
        /**
         * @brief Already-emitted correction token that still needs a main verifier row.
         *
         * vLLM-style MTP rejection publishes accepted state only through the
         * accepted verifier prefix.  When stochastic verification rejects a
         * draft, the sampled correction is emitted to the caller immediately,
         * but the next speculative transaction can consume that correction as
         * verifier input row zero instead of running a standalone one-token
         * condition forward.  This field owns that handoff.  The token is
         * already in `sampler_` history and must not be emitted or recorded
         * again when the next transaction commits.
         */
        std::optional<int32_t> pending_mtp_condition_token_;
        std::optional<SamplingParams> pending_mtp_condition_params_;
        /**
         * @brief Device-resident source for @ref pending_mtp_condition_token_.
         *
         * The host token above is still required because `decodeStep()` returns
         * concrete token ids to the caller.  It must not, however, become the
         * source of truth for the next GPU sidecar.  When direct MTP
         * publication exposes a logical-state mailbox, this handle lets the
         * next fixed-depth stochastic transaction consume the correction token
         * and logical position from GPU metadata, matching vLLM's
         * device-resident transaction shape.
         */
        std::optional<DeviceResidentLogicalSequenceStateHandle>
            pending_mtp_condition_resident_state_;
        /**
         * @brief First MTP sidecar already queued for the next decode step.
         *
         * vLLM overlaps response copies with target-state postprocessing and
         * the following draft proposal.  In Llaminar's synchronous
         * `decodeStep()` API we still return concrete host tokens, but when a
         * direct GPU publication has a resident continuation token and no stop
         * token can invalidate it, we can enqueue the next first sidecar before
         * waiting for the host response bridge.  The next decode step consumes
         * this handle only if it still matches the pending/ready resident
         * mailbox exactly.
         */
        std::optional<DeviceResidentLogicalSequenceStateHandle>
            prelaunched_mtp_first_sidecar_resident_state_;
        std::optional<SamplingParams>
            prelaunched_mtp_first_sidecar_params_;
        /**
         * @brief Whether this decode step already published device MoE maintenance.
         *
         * The flag bridges the inner MTP transaction, which must order a
         * prelaunched sidecar immediately, and the outer generation loop, which
         * historically owned the decode-boundary callback.  It does not replace
         * a GPU event or carry data-plane state; it only prevents the host
         * scheduler from submitting the same maintenance graph twice.
         */
        bool device_moe_maintenance_published_in_decode_step_ = false;
        /**
         * @brief Scheduler-owned logical position for the current decode transaction.
         *
         * This control-plane scalar is initialized from the request token count
         * after successful prefill. Ordinary decode advances it after each
         * committed condition-token forward; MTP advances it from the same
         * compact outcome metadata that drives device publication. It is never
         * adopted from a backend host mirror. KV, recurrent state, terminal
         * hidden state, and publication metadata remain device-owned.
         */
        std::optional<int> decode_transaction_planning_position_;
        /**
         * @brief Typed scheduler lifecycle for the resident generation controller.
         *
         * A pair of independent `pending` and optional-budget fields previously
         * admitted contradictory states. In particular, a budget-limited public
         * `decodeStep()` could retire the device controller and clear its budget
         * while leaving admission closed; the following step then reached the
         * grouped verifier with no controller transaction to consume. Keeping the
         * phase and its budget in one type makes that state unrepresentable.
         *
         * This object owns control-plane admission metadata only. Tokens, response
         * progress, stop state, dynamic depth, KV state, and recurrent state remain
         * device-owned throughout an active controller transaction.
         */
        struct DeviceGenerationAdmissionLifecycle
        {
            /** @brief Legal scheduler phases for one request's controller. */
            enum class Phase
            {
                /** No live request may admit a generation controller. */
                Inactive,
                /** A live request awaits a positive caller response budget. */
                AwaitingBudget,
                /** One resident controller owns the recorded response budget. */
                ControllerActive,
            };

            /** Current phase; `admitted_token_budget` is positive only when active. */
            Phase phase = Phase::Inactive;
            int admitted_token_budget = 0;

            /**
             * @brief Reset at an explicit request/session boundary.
             *
             * Reset is intentionally unconditional because the underlying runner
             * joins and retires device work before the orchestration request state
             * is cleared. It must not be used as recovery inside a live decode.
             */
            void reset() noexcept
            {
                phase = Phase::Inactive;
                admitted_token_budget = 0;
            }

            /**
             * @brief Open the post-prefill admission boundary.
             * @return true only for the unique Inactive -> AwaitingBudget edge.
             */
            [[nodiscard]] bool openAfterPrefill() noexcept
            {
                if (phase != Phase::Inactive || admitted_token_budget != 0)
                    return false;
                phase = Phase::AwaitingBudget;
                return true;
            }

            /**
             * @brief Bind one positive budget to a newly published controller.
             * @param token_budget Exact terminal response capacity admitted on device.
             * @return true only for AwaitingBudget -> ControllerActive.
             */
            [[nodiscard]] bool admitController(int token_budget) noexcept
            {
                if (phase != Phase::AwaitingBudget ||
                    admitted_token_budget != 0 || token_budget <= 0)
                {
                    return false;
                }
                phase = Phase::ControllerActive;
                admitted_token_budget = token_budget;
                return true;
            }

            /**
             * @brief Retire a validated terminal ledger.
             *
             * A model stop closes the request. Exhausting only the caller's
             * budget leaves the request live and opens a fresh admission edge for
             * the next public `decodeStep()`; no device payload is reconstructed.
             *
             * @param request_complete Whether model stop semantics ended the request.
             * @return true only for ControllerActive -> Inactive/AwaitingBudget.
             */
            [[nodiscard]] bool retireController(bool request_complete) noexcept
            {
                if (phase != Phase::ControllerActive ||
                    admitted_token_budget <= 0)
                {
                    return false;
                }
                phase = request_complete
                            ? Phase::Inactive
                            : Phase::AwaitingBudget;
                admitted_token_budget = 0;
                return true;
            }

            /** @return true when the next grouped decode must admit a controller. */
            [[nodiscard]] bool awaitsAdmission() const noexcept
            {
                return phase == Phase::AwaitingBudget;
            }

            /** @return true while exactly one resident controller is active. */
            [[nodiscard]] bool controllerActive() const noexcept
            {
                return phase == Phase::ControllerActive;
            }

            /** @return Active controller budget, or zero in every other phase. */
            [[nodiscard]] int tokenBudget() const noexcept
            {
                return controllerActive() ? admitted_token_budget : 0;
            }
        } device_generation_admission_;
        /**
         * @brief Whether the terminal GPU ledger owns reported adaptive depth.
         *
         * Native generation does not advance the dormant host depth controller.
         * Once its single terminal bridge succeeds, diagnostics must read the
         * final selector and counters copied from that ledger rather than
         * reviving the host controller as a second authority.
         */
        bool device_generation_terminal_ledger_authoritative_{false};
        /**
         * @brief Per-request state initialized by prefillBatch().
         *
         * Phase 8 request batching must never reuse the scalar last-token or
         * ready-logit fields above. This small state object is the ownership
         * bridge for the future decodeStepBatch() implementation: every
         * request starts with its own terminal prompt token and ready-logit
         * bit, while scalar decode is explicitly disabled until the batch is
         * cleared or consumed by the batched decode API.
         */
        struct BatchedDecodeRequestState
        {
            int32_t last_token = 0;
            int logical_tokens = 0;
            /**
             * @brief Immutable position-keyed stochastic seed for this request.
             *
             * Explicit user seeds remain unchanged so RB=N and RB=1 produce the
             * same draws. A seed of zero is resolved once at request creation;
             * subsequent GPU draws are derived from this seed and resident
             * logical positions instead of advancing host RNG state.
             */
            uint64_t stochastic_position_seed = 0;
            bool prefill_logits_ready = false;
            std::optional<int32_t> ready_sampled_token;
            std::optional<SamplingParams> ready_sampled_params;
            /**
             * @brief Independent sampler history for this logical request.
             *
             * Request batching must match running each prompt as its own scalar
             * decode stream. Sharing `sampler_` across lanes would interleave
             * RNG draws and repetition history, which is harmless for strict
             * greedy decode but wrong for stochastic or penalty-aware MTP.
             */
            Sampler sampler;
            bool is_complete = false;
        };
        std::vector<BatchedDecodeRequestState> batched_request_states_;
        bool batched_decode_active_{false};
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
        uint64_t request_epoch_{0};
    };

} // namespace llaminar2
