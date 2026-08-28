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
#include "RankInitializationLifecycle.h"
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
#include "../local_execution/device/ReusableExecutionWorkspace.h"
#include "../../interfaces/IMPIContext.h"
#include "../../utils/Assertions.h"
#include "../../utils/MPIContext.h"
#include "../../utils/Tokenizer.h"
#include <memory>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaminar2
{

    class MoEOverlayResidencyAuthority;
    class MoEOverlayParticipantResidencyRegistry;
    class MoEOverlayPhysicalResidencyFabric;
    struct MoEOverlayReusableContextSeal;
    class MoEOverlayHostAuthorityDeviceBankPublisher;
    class MoEOverlayParticipantPreparedWaveFactory;
    class MoEOverlayTierMigrationTransport;
    class MoEOverlayMPIRemoteProjectionTransport;
    class MoEOverlayMPIResidencyConsensus;
    class MoEOverlayDistributedResidencyTransport;
    class MoEOverlayMPIResidencyProposalPublisher;
    class MoEOverlayResidencyMaintenanceService;
    class MoEOverlayInferenceInterferenceProbe;
    class MoEOverlayRankBatchTransportRegistry;
    class MoEOverlayNodeLocalDeviceControllerFabric;
    class MoEOverlayDeviceControllerGraphService;
    struct MoEOverlayDeviceControllerTopology;
    class MoEOverlayInferenceTransactionCoordinator;
    class MoEOverlayInferenceTransactionFollower;
    class IModelContext;
    struct MoEExpertOverlayExecutionPlan;
    struct MoEOverlayResidencySnapshot;
    struct InferenceMeasurementReadiness;

    /**
     * @brief Freeze a model-aware ExpertOverlay plan for one runtime graph family.
     *
     * The MTP policy is part of placement identity because a routed NextN
     * sidecar owns an additional prepared expert bank that does not exist in a
     * non-MTP graph family.
     *
     * @param model_ctx Loaded model and tensor inventory authority.
     * @param plan Declarative requested placement plan.
     * @param mtp Exact effective MTP policy selected by orchestration.
     * @return Immutable runtime-active placement plan, or null for no request.
     */
    std::shared_ptr<MoERoutedExpertPlacementPlan> freezeMoEExpertOverlayPlanForModel(
        IModelContext &model_ctx,
        const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan,
        const MTPRuntimeConfig &mtp);

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
         * @brief Construct with an immutable, preloaded real-weight context
         *
         * Runs the ordinary production initialization sequence, including plan
         * construction, graph lowering, capture, and runtime controllers, while
         * reusing one already authenticated ModelContext.  Initialization
         * rejects a context whose path, distribution, or layer ownership does
         * not match the newly built execution plan; it never falls back to
         * loading a different context.
         *
         * @param config Orchestration configuration
         * @param plan_builder Plan builder for creating execution plans
         * @param reuse_contract Immutable model authority and the production
         *        plan that certified its prepared weights
         */
        OrchestrationRunner(
            OrchestrationConfig config,
            std::unique_ptr<IExecutionPlanBuilder> plan_builder,
            std::shared_ptr<ModelContext> model_ctx);

        /**
         * @brief Construct with plan-certified prepared weights from a prior runner.
         *
         * @param config Orchestration configuration.
         * @param plan_builder Builder for the new production plan.
         * @param reuse_contract Context and weight-affecting plan certificate.
         */
        OrchestrationRunner(
            OrchestrationConfig config,
            std::unique_ptr<IExecutionPlanBuilder> plan_builder,
            ModelContextReuseContract reuse_contract);

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
        bool maybeApplyMoERebalance(uint64_t committed_tokens) override;
        uint64_t moeRuntimeMovementEpoch() const override;
        MoEOptimizationStatus moeOptimizationStatus() const override;
        MoEOptimizationMovementLedger
        moeOptimizationMovementLedger() const override;

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
        bool purgePrefixCache() override;
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
        const IModelContext *modelContextForDiagnostics() const override;

        /**
         * @brief Return the immutable live ExpertOverlay map for diagnostics.
         *
         * This read-only view exists so production parity can attribute router
         * checkpoints to the exact published epoch. It exposes no mutation or
         * publication operation and returns null when ExpertOverlay is absent.
         */
        [[nodiscard]] std::shared_ptr<const MoEOverlayResidencySnapshot>
        expertOverlayResidencySnapshotForDiagnostics() const;

        std::optional<ModelContextReuseContract>
        modelContextReuseContract() const override;
        ModelContextReuseStatus modelContextReuseStatus() const override;
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

        // =====================================================================
        // IOrchestrationRunner: GPU-side Sampling
        // =====================================================================

        int sampleGreedyOnDevice() override;
        int sampleOnDevice(const SamplingParams &params) override;
        bool waitForLastForwardCompletionForBenchmark() override;
        InferenceReadiness inferenceReadiness() const override;
        bool prepareForInference() override;
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
            DECODE_STEP = 4,         ///< Run one decode step (followed by typed progress payload)
            SKIP_LOGITS_DECODE = 5,  ///< Set skip-logits-gather for decode
            PURGE_PREFIX_CACHE = 6,  ///< Retire reusable prefix records on every rank
            FORCE_DECODE_TOKEN = 7,  ///< Commit a forced token (followed by token id)
            SET_STOP_TOKENS = 8,     ///< Install request stop policy (followed by count + token IDs)
            SHUTDOWN = 99            ///< Exit the worker loop
        };

        /** Terminal lifecycle of the coordinated worker-command channel. */
        enum class MPIWorkerCommandLifecycle : std::uint8_t
        {
            AcceptingCommands, ///< Worker ranks are blocked in command receive.
            ShutdownPublished, ///< Every worker has observed terminal admission.
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
         * @brief Close command admission and drain topology maintenance.
         *
         * Multi-rank roots publish SHUTDOWN before the collective drain.
         * Single-rank runners perform the identical local drain without a wire
         * command so old residency banks retire before terminal evidence.
         */
        void shutdownMPIWorkers() override;
        int coordinatedRootRank() const override
        {
            return mpi_coordinated_root_rank_;
        }
        void setMoEExpertOverlayMPIContext(std::shared_ptr<IMPIContext> mpi_ctx) override
        {
            moe_expert_overlay_mpi_ctx_ = std::move(mpi_ctx);
        }

        /**
         * @brief Enable MPI coordinated mode.
         *
         * When enabled, the inventory-resolved coordinated root broadcasts
         * commands before inference operations so worker ranks participate in
         * lockstep. The root enables this before workers enter their loops.
         *
         * Only server/interactive modes use this. Modes where all ranks run
         * the same code path (SingleShotChat, Completion) do NOT enable this.
         */
        void setMPICoordinatedMode(bool enabled) override
        {
            /**
             * A worker loop can only be started once for an orchestration
             * runner.  Re-enabling coordinated mode after rank zero has sent
             * SHUTDOWN would make a later command wait for workers that have
             * already deliberately left the communicator protocol.
             */
            if (enabled &&
                mpi_worker_command_lifecycle_ ==
                    MPIWorkerCommandLifecycle::ShutdownPublished)
            {
                LLAMINAR_UNREACHABLE(
                    "Cannot re-enable MPI coordinated mode after worker shutdown; "
                    "construct a new OrchestrationRunner for the next serving session");
            }
            mpi_coordinated_mode_ = enabled;
        }

        /**
         * @brief Broadcast an MPI command from the coordinated root to all ranks.
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

        /**
         * @brief Wait for prior asynchronous host reclaim before runtime allocation.
         *
         * Reused model contexts may still be completing the first-prefill
         * release submitted by their previous runner.  Initialization crosses
         * this model/JIT admission edge collectively before memory planning or
         * graph construction; inference itself never waits on reclamation.
         *
         * @return True after reclaim is complete or was never submitted.
         */
        bool awaitMmapReclaimBeforeRuntimeAllocation();

        bool freezeMoEExpertOverlayPlanForLoadedModel();

        /**
         * @brief Freeze the MPI scope that owns continuation prefix/KV state.
         *
         * Heterogeneous overlays may contain expert-only ranks that follow
         * retained sparse transactions but own no prefix cache.  This setup
         * phase creates a persistent communicator containing only dense
         * continuation ranks, before any request enters the hot path.
         *
         * @return False when the frozen execution plan cannot produce one
         *         valid continuation-state authority set.
         */
        bool initializeMoEContinuationPrefixCoordination();

        /**
         * @brief Release an owned continuation-prefix communicator.
         *
         * This runs after inference and worker-loop admission have closed.
         * World and process-local scopes own no communicator and are inert.
         */
        void releaseMoEContinuationPrefixCoordination() noexcept;

        /**
         * @brief Create the one process-local RCU authority for a tiered overlay.
         *
         * This runs after model-aware placement is frozen and before any root or
         * participant graph is built. Dynamic policies also create the shared
         * double-buffered histogram from the exact owner-map geometry.
         */
        bool initializeMoEExpertOverlayResidencyAuthority();

        /**
         * @brief Return whether the frozen authority is a homogeneous GPU tier.
         *
         * The authority execution enum deliberately also covers homogeneous
         * CPU tiers, where host and participant memory are the same domain.
         * This narrower predicate selects the captured CUDA/ROCm executor and
         * prevents the host maintenance service from becoming a second writer.
         */
        /**
         * @brief Return whether the installed one-domain native GPU controller can own this plan.
         *
         * Every all-GPU plan is logically device-resident. This narrower
         * predicate identifies only the already-composed homogeneous,
         * single-tier NCCL/RCCL implementation; it must never be used to
         * choose host authority for another all-GPU topology.
         */
        bool usesSingleDomainNativeGpuDeviceResidentMoEOverlayAuthority() const;

        /**
         * @brief Materialize the topology-selected authority executor.
         *
         * Called only after graph construction has registered every live
         * initial prepared-engine bank. Homogeneous GPU tiers retain their
         * captured device transaction; CPU, heterogeneous, and multi-tier
         * plans install the process-local physical fabric and background host
         * scheduler. Exactly one branch may publish durable epochs.
         */
        bool initializeMoEExpertOverlayResidencyMaintenance();

        /**
         * @brief Start a fully composed host maintenance service.
         *
         * `initialize()` invokes this as a distinct all-rank phase after every
         * participant has committed maintenance composition. Device-resident
         * and movement-disabled regimes have no host service and succeed as a
         * typed no-op.
         */
        bool startMoEExpertOverlayResidencyMaintenance();

        /**
         * @brief Seal an ordinary GPU runner's admitted serving graph family.
         *
         * Dense and non-overlay GPU execution needs the same setup guarantee as
         * ExpertOverlay: every admitted prefill bucket and the history-bearing
         * decode graph must be captured and instantiated without executing a
         * synthetic request. This leaves KV, prefix, sampler, and response
         * state untouched, so the first admitted request is both a genuine
         * cache miss and an optimized graph replay.
         *
         * ExpertOverlay is excluded because its serving family is composed
         * atomically with transfer-progress epochs by
         * bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily(). CPU
         * runners have no native executable family and are a typed no-op.
         *
         * @return True when no ordinary native family is required or when the
         *         complete admitted family has entered its sealed state.
         */
        bool materializeOrdinaryServingGraphFamilyWithoutLaunch();

        /**
         * @brief Apply the caller-declared diagnostic topology before capture.
         *
         * Snapshot copies are native graph nodes, so enabling them after a
         * serving executable has been materialized changes graph identity.
         * Public snapshot configuration calls made before initialize() are
         * retained in @ref snapshot_capture_setup_ and installed by this
         * initialization phase immediately after the participant graph exists.
         *
         * @return True when diagnostics are disabled or the complete retained
         *         policy was installed on the constructed inference runner.
         */
        bool applyConfiguredSnapshotCaptureSetup();

        /**
         * @brief Bind physical transfer epochs and seal every serving graph.
         *
         * The physical fabric's exact GPU inventory is installed into the
         * continuation or follower runner first. The already-frozen common
         * prefill schedule is then captured without launch, making transfer
         * branch identity and memory-preflight row accounting one atomic setup
         * lifecycle. This must run before any maintenance worker starts.
         */
        bool bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();

        /**
         * @brief Bind the fixed-slot heterogeneous inference control plane.
         *
         * The continuation rank creates one direct publisher per remote expert
         * rank and installs a shared coordinator into its symmetric LocalTP
         * graphs. Each remote rank binds its already-retained expert graph to a
         * follower. No ticket contains token, KV, sampler, or model state.
         */
        bool initializeMoEOverlayInferenceTransactions();

        /** @brief Why one residency-maintenance composition is being drained. */
        enum class MoEOverlayMaintenanceDrainIntent : std::uint8_t
        {
            /**
             * Discard an empty, superseded, or partially built composition.
             *
             * Initialization uses this before rebuilding the maintenance
             * graph. It must never change reusable-model lifecycle state:
             * there is not yet a complete residency authority to certify.
             */
            ResetComposition,
            /**
             * End runner ownership and certify model-owned prepared weights.
             *
             * This is the only intent allowed to restore Dynamic placement and
             * publish the `Sealing -> Reusable` model-context transition.
             */
            TerminalContextSeal,
        };

        /**
         * @brief Stop background proposals and destroy migration ownership safely.
         * @param intent Typed distinction between setup cleanup and terminal seal.
         *
         * The maintenance worker drains exact transfer events while the graph,
         * prepared engines, and devices are still alive. Destruction then
         * proceeds transport, factory, fabric, registry, and authority. Only
         * @ref MoEOverlayMaintenanceDrainIntent::TerminalContextSeal may mutate
         * the reusable model-context lifecycle.
         */
        void shutdownMoEExpertOverlayResidencyMaintenance(
            MoEOverlayMaintenanceDrainIntent intent) noexcept;

        /**
         * @brief Enter the exclusive terminal seal for an exported context.
         * @param error Optional precise lifecycle rejection diagnostic.
         * @return True when no contract exists or this runner owns `Sealing`.
         */
        bool beginModelContextReuseSealIfNeeded(
            std::string *error = nullptr) noexcept;

        /**
         * @brief Restore Dynamic placement and atomically rebind prepared keys.
         * @param error Receives the first protocol or registry failure.
         * @return True when the model context is physically reusable.
         */
        bool sealReusableModelContextPhysicalState(
            std::string *error = nullptr) noexcept;

        /**
         * @brief Seal the exact GPU allocations still owned by the model value.
         * @param retention Receives one unique row per rank-local prepared GPU.
         * @param error Receives missing ownership or overflow diagnostics.
         * @return True after every live model pool and workspace is accounted.
         *
         * This runs only after graph, controller, transport, and local-context
         * teardown. It queries model-owned allocation registries rather than an
         * admission projection or driver free-memory delta, then supplies the
         * value for the atomic `Sealing -> Reusable` publication.
         */
        bool sealModelContextDeviceMemoryRetention(
            std::vector<ModelDeviceMemoryRetention> *retention,
            std::string *error = nullptr) const noexcept;

        /**
         * @brief Rebind the model registry from a canonical physical seal.
         * @param seal Exact loader-allocation aliases produced after reader drain.
         * @param error Receives missing-bank or atomic replacement diagnostics.
         * @return True only after every process-local scope is rebound.
         *
         * Published participant banks may still name temporary shadow slots even
         * after their logical owner map returns to the prepared placement.  Only
         * the fabric's typed terminal seal is therefore a legal reuse source.
         */
        bool rebindModelRegistryFromReusableContextSeal(
            const MoEOverlayReusableContextSeal &seal,
            std::string *error = nullptr) noexcept;

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
         * @brief Refresh live GPU capacity in the rank inventory.
         *
         * Discovery-time free memory is valid only before persistent model
         * allocations begin. A reused ModelContext observes free bytes after
         * its certified weights and workspace are already resident, so final
         * admission must replace the stale discovery value with one exact
         * backend observation before applying retained-allocation credits.
         * Capacity solving and final memory validation both use this method so
         * they cannot drift onto different physical-memory authorities.
         *
         * Every selected context is initialized before any device is sampled;
         * this preserves a common post-runtime-initialization observation
         * boundary on backends with process-wide lazy state.
         *
         * @param devices Rank-local devices participating in the physical plan.
         * @param purpose Stable diagnostic identity for the observation edge.
         * @return True after every unique GPU has updated the rank inventory.
         */
        bool refreshRankLocalGPUCapacityObservations(
            const std::vector<DeviceId> &devices,
            std::string_view purpose);

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
         * @brief Freeze the one bounded prefill schedule used by every overlay rank.
         *
         * After each rank has independently admitted its local graph arena,
         * the continuation root publishes the configured bucket ladder and the
         * overlay communicator reduces the capacity to the smallest safe
         * value.  This runs only during initialization; request execution
         * consumes the resulting typed RuntimeConfig contract without a
         * control-plane collective or a capacity guess.
         *
         * @return True when no distributed overlay is active or a complete
         *         executable schedule contract has been installed.
         */
        bool establishMoEOverlayPrefillScheduleContract();

        /**
         * @brief Bilaterally first-touch every same-node sparse activation channel.
         *
         * All overlay ranks iterate the frozen rank-pair topology in the same
         * order. For each channel its source and target construct the mapping
         * concurrently, so consumer-owned page first-touch completes before
         * either rank begins expensive prepared-weight or graph construction.
         * The resulting process-local transports are published through one
         * immutable registry retained by every graph.
         *
         * @return True when no distributed overlay is active or every required
         *         node-local channel has completed its setup rendezvous.
         */
        bool initializeMoEOverlayRankBatchTransports();

        /**
         * @brief Build compute graphs
         */
        bool buildComputeGraph();

        /**
         * @brief Initialize MPI context if needed
         */
        bool initializeMPI();

        /** @brief Communicator authority used to authenticate one init phase. */
        enum class InitializationConsensusScope : std::uint8_t
        {
            BootstrapWorld,
            ActiveContext,
            ExpertOverlayContext,
        };

        /**
         * @brief Execute one fallible phase and reach the same terminal on all ranks.
         *
         * A returned failure or exception is converted into a typed local
         * outcome before the single phase all-gather. The all-gather also
         * authenticates the phase ordinal and name, preventing reordered or
         * skipped initialization phases from silently matching collectives.
         *
         * @param workflow Operator-facing workflow name for diagnostics.
         * @param identity Ordered phase identity shared by every rank.
         * @param scope Bootstrap world or the established runner communicator.
         * @param step Rank-local work; must not escape its own internal collectives.
         * @return True only when every rank completed this exact phase.
         */
        bool runRankSynchronizedInitializationPhase(
            std::string_view workflow,
            RankInitializationPhaseIdentity identity,
            InitializationConsensusScope scope,
            const std::function<bool()> &step);

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
         * @brief Record that one resident generation step owned its MoE boundaries.
         *
         * Dynamic homogeneous GPU parents contain the first external
         * maintenance publication and every internal transaction tail. This
         * method publishes that ownership as PerfStats evidence and arms one
         * outer-loop acknowledgement; it never launches maintenance itself.
         *
         * @param transaction_count Number of device-ledger transactions whose
         *        graph family contained the typed maintenance boundary.
         * @param execution_policy Exact CUDA/HIP parent execution policy.
         * @return true when no device authority is active or ownership was
         *         recorded; false on an overlapping lifecycle publication.
         */
        bool noteEmbeddedDeviceMoEOverlayMaintenance(
            uint64_t transaction_count,
            DeviceGenerationExecutionPolicy execution_policy);
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
         * @brief Open this rank's ExpertOverlay demand bank at request admission.
         *
         * A coordinated heterogeneous prefill has one logical request boundary
         * but two process-local entry points: the continuation rank enters
         * @ref prefill while expert-only ranks enter the `PREFILL` worker
         * command. Both must perform the same idempotent authority transition
         * before any retained graph ticket executes. Keeping that transition in
         * one helper prevents a follower from remaining in certification
         * quarantine while the coordinator begins proposal publication.
         *
         * @return True when no host ExpertOverlay authority exists or its
         *         request-boundary transition succeeded; false after publishing
         *         a precise runner error.
         */
        bool activateMoEOverlayDemandAtRequestBoundary();
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
        /**
         * @brief Publish the current request generation to graph-native MoE runners.
         *
         * The generation is an overlay-world protocol value rather than a
         * graph-cache or local device epoch. Every rank invokes this at the
         * same initialization/reset boundary before it can enter a sparse
         * dispatch collective.
         *
         * @param reason Stable lifecycle label retained in PerfStats evidence.
         * @return False when a distributed overlay runner rejects the value.
         */
        bool publishMoEOverlayCollectiveRequestGeneration(const char *reason);
        /**
         * @brief Advance and publish a fresh distributed sparse-collective generation.
         *
         * Request reset returns logical token positions to zero. Advancing this
         * separate authority prevents a preserved graph cache from accepting a
         * stale sparse packet key from the previous request.
         */
        bool advanceMoEOverlayCollectiveRequestGeneration(const char *reason);
        /**
         * @brief Reset request-owned runner state after the overlay epoch is quiescent.
         *
         * Expert movement is owned by the ExpertOverlay maintenance service;
         * this lifecycle helper therefore cannot publish masks, replicas, or
         * any other durable placement data.
         */
        void resetUnderlyingRunnerRequestState(const char *reason);

        // =====================================================================
        // Error Handling
        // =====================================================================

        /**
         * @brief Terminate a coordinated MPI worker after an inference command fails.
         *
         * A worker may fail before another rank enters the next sparse MoE or
         * tensor-parallel collective. Returning to the command broadcast loop
         * in that state strands the other rank inside that unmatched collective.
         * A real MPI communicator is therefore aborted as one job. Test
         * contexts deliberately expose `MPI_COMM_NULL`; they receive an
         * exception instead so unit tests can prove that the worker never
         * consumes a later command after failure.
         *
         * @param command Name of the coordinated command that failed.
         * @param reason Command-local failure diagnostic.
         * @note This method never returns.
         */
        [[noreturn]] void terminateFailedMPIWorkerCommand(
            const char *command,
            const std::string &reason);

        /**
         * @brief Read the private ExpertOverlay preparation protocol snapshot.
         *
         * Application modes deliberately cannot call this method. It retains
         * the exact requested workload and progress identity needed by the
         * orchestration-owned preparation driver while the public API exposes
         * only generic readiness.
         */
        [[nodiscard]] InferenceMeasurementReadiness
        internalInferencePreparationReadiness() const;

        /**
         * @brief Set error message and return false
         */
        bool setError(const std::string &error);

        // =====================================================================
        // State
        // =====================================================================

        // Configuration
        OrchestrationConfig config_;
        /** Requested routed-weight identity captured before model-aware freezing. */
        std::string requested_routed_weight_authority_identity_;
        RankExecutionPlan plan_;
        bool plan_built_{false};
        ClusterInventory cluster_inventory_;

        // Dependencies (injected or created)
        std::unique_ptr<IExecutionPlanBuilder> plan_builder_;
        std::shared_ptr<IMPIContext> mpi_ctx_;
        std::shared_ptr<IMPIContext> moe_expert_overlay_mpi_ctx_;
        /**
         * @brief Typed scope for prefix/KV consensus during request admission.
         *
         * `OrchestrationWorld` is the ordinary TP/PP policy.
         * `ProcessLocalContinuation` means one rank already aggregates every
         * continuation device. `ContinuationRankGroup` owns an MPI
         * communicator excluding expert-only followers.
         */
        enum class PrefixCoordinationScope : std::uint8_t
        {
            OrchestrationWorld,
            ProcessLocalContinuation,
            ContinuationRankGroup,
        };
        PrefixCoordinationScope prefix_coordination_scope_{
            PrefixCoordinationScope::OrchestrationWorld};
        /** Persistent subgroup communicator, owned only in group scope. */
        MPI_Comm continuation_prefix_comm_{MPI_COMM_NULL};
        /** Whether this rank owns continuation state in the frozen scope. */
        bool continuation_prefix_participant_{true};
        /** Shared route evidence retained by authority and every child graph. */
        std::shared_ptr<DecodeExpertHistogram>
            moe_expert_overlay_decode_histogram_;
        /** The only live owner-map publication authority in this process. */
        std::shared_ptr<MoEOverlayResidencyAuthority>
            moe_expert_overlay_residency_authority_;
        /** Epoch-indexed prepared banks for participants hosted by this rank. */
        std::shared_ptr<MoEOverlayParticipantResidencyRegistry>
            moe_expert_overlay_participant_residency_;
        /** Preallocated process-local CPU/GPU destination slots and lanes. */
        std::shared_ptr<MoEOverlayPhysicalResidencyFabric>
            moe_expert_overlay_physical_residency_fabric_;
        /** Persistent host-authority publication resources for local GPUs. */
        std::shared_ptr<MoEOverlayHostAuthorityDeviceBankPublisher>
            moe_expert_overlay_device_bank_publisher_;
        /** Private non-blocking MPI data plane for cross-rank projections. */
        std::shared_ptr<MoEOverlayMPIRemoteProjectionTransport>
            moe_expert_overlay_remote_projection_transport_;
        /** Joins physical arrivals into immutable participant candidate banks. */
        std::unique_ptr<MoEOverlayParticipantPreparedWaveFactory>
            moe_expert_overlay_wave_factory_;
        /** Process-local composite event-polled migration transport. */
        std::shared_ptr<MoEOverlayTierMigrationTransport>
            moe_expert_overlay_migration_transport_;
        /** Private all-rank reservation/stage/commit vote authority. */
        std::shared_ptr<MoEOverlayMPIResidencyConsensus>
            moe_expert_overlay_residency_consensus_;
        /** Distributed two-phase wrapper around process-local movement. */
        std::shared_ptr<MoEOverlayDistributedResidencyTransport>
            moe_expert_overlay_distributed_migration_transport_;
        /** Coordinator-to-peer immutable histogram publication lane. */
        std::shared_ptr<MoEOverlayMPIResidencyProposalPublisher>
            moe_expert_overlay_proposal_publisher_;
        /** Sole host background scheduler for durable overlay movement. */
        std::unique_ptr<MoEOverlayResidencyMaintenanceService>
            moe_expert_overlay_maintenance_service_;
        /** Lock-free live inference interval boundary used only when armed. */
        std::shared_ptr<MoEOverlayInferenceInterferenceProbe>
            moe_expert_overlay_interference_probe_;
        /**
         * @brief Typed owner of the root rank's complete prefill interval.
         *
         * Ordinary execution measures at the orchestration call boundary. A
         * distributed heterogeneous graph instead hands only prefill ownership
         * to its transaction coordinator, which can close the interval at the
         * exact aggregate CPU/GPU participant fence. Decode and grouped-MTP
         * measurements remain orchestration-owned in both states.
         */
        enum class MoEOverlayPrefillInterferenceOwner
        {
            OrchestrationScope,   ///< One ordinary runner call owns timing.
            TransactionCoordinator, ///< Exact heterogeneous graph group owns timing.
        };
        MoEOverlayPrefillInterferenceOwner
            moe_overlay_prefill_interference_owner_ =
                MoEOverlayPrefillInterferenceOwner::OrchestrationScope;
        /** Node-local channels whose consumer pages were first-touched pre-graph. */
        std::shared_ptr<MoEOverlayRankBatchTransportRegistry>
            moe_overlay_rank_batch_transport_registry_;
        /** Frozen all-GPU leader/follower authority topology, never host policy state. */
        std::shared_ptr<const MoEOverlayDeviceControllerTopology>
            moe_overlay_device_controller_topology_;
        /**
         * Node-local CUDA/HIP aliases for the sole mapped device authority.
         * Empty for host-resident overlays and native one-group GPU control.
         */
        std::shared_ptr<MoEOverlayNodeLocalDeviceControllerFabric>
            moe_overlay_device_controller_fabric_;
        /** Retained group-root graphs on dedicated controller streams. */
        std::unique_ptr<MoEOverlayDeviceControllerGraphService>
            moe_overlay_device_controller_graph_service_;
        /**
         * Successful terminal optimization state retained after maintenance
         * resources are released. Reset-composition teardown clears it.
         */
        std::optional<MoEOptimizationStatus>
            terminal_moe_optimization_status_;
        /** Complete runner-lifetime movement identities after terminal drain. */
        std::optional<MoEOptimizationMovementLedger>
            terminal_moe_optimization_movement_ledger_;
        /**
         * Continuation-authoritative retired prefill rows awaiting sideband.
         *
         * Remote retained graphs may execute a different capture/retirement
         * schedule, so their local callback count is not an admission authority.
         * The continuation rank carries this exact total in the next existing
         * decode/forced-token command payload and every rank advances one gate
         * from that same fact.
         */
        std::atomic<std::uint64_t>
            moe_overlay_pending_prefill_progress_tokens_{0u};
        /**
         * Continuation-authoritative committed decode steps awaiting sideband.
         *
         * Device-owned Dynamic maintenance never needs a second MPI command.
         * The ordinary post-step hook only queues this progress; the next
         * decode/forced-token command publishes it to every participant before
         * launching inference. A terminal request deliberately leaves its tail
         * unpublished because no later inference can benefit from movement.
         */
        std::atomic<std::uint64_t>
            moe_overlay_pending_decode_progress_tokens_{0u};
        std::shared_ptr<ModelContext> model_ctx_;
        /**
         * Stable backing-only execution storage carried by the prepared model
         * contract. Each live graph owner holds an exclusive typed lease.
         */
        std::shared_ptr<ReusableExecutionWorkspaceRegistry>
            reusable_execution_workspaces_ =
                std::make_shared<ReusableExecutionWorkspaceRegistry>();
        /**
         * @brief Plan that certified a caller-retained PreparedWeightStore.
         *
         * Present only for the explicit reuse constructor.  Initialization
         * compares its weight-affecting topology with the newly resolved plan
         * before memory admission can credit existing device residency.
         */
        std::optional<RankExecutionPlan> retained_prepared_weight_plan_;
        /** Model-frozen physical routed plan retained by the reuse contract. */
        std::shared_ptr<const MoERoutedExpertPlacementPlan>
            retained_prepared_routed_weight_plan_;
        /** Exact requested ExpertOverlay identity carried by the reuse contract. */
        std::string retained_routed_weight_authority_identity_;
        /** True only after the current plan passed the retained-plan comparison. */
        bool retained_prepared_weight_plan_validated_{false};
        /** Model-owned prepared records counted when an imported plan was admitted. */
        std::size_t retained_prepared_entry_count_{0u};
        /** Serializes const contract export with terminal lifecycle publication. */
        mutable std::mutex model_context_reuse_mutex_;
        /** Shared one-runner/sealing authority copied into exported contracts. */
        mutable std::shared_ptr<ModelContextReuseAuthority>
            model_context_reuse_authority_;
        /** Whether this runner acquired an imported reusable contract. */
        bool imported_model_context_reuse_authority_{false};
        /** Prevent repeated physical restoration when shutdown is re-entered. */
        bool model_context_physical_state_sealed_{false};
        /** Sticky terminal result consumed when publishing `Reusable`. */
        bool model_context_reuse_seal_succeeded_{true};
        /** First terminal seal failure retained until contract invalidation. */
        std::string model_context_reuse_seal_error_;
        std::unique_ptr<ILocalTPContext> local_tp_ctx_;
        std::unique_ptr<ILocalPPContext> local_pp_ctx_;

        // Execution infrastructure
        std::unique_ptr<IInferenceRunner> runner_;

        /**
         * @brief Typed setup policy for graph-resident diagnostic copies.
         *
         * A caller is allowed to configure snapshots before initialize(). The
         * policy therefore cannot live only on `runner_`, which is constructed
         * part-way through initialization. Keeping enablement, filter, and
         * destination in one value prevents a filter from being silently lost
         * or applied after serving graphs have already captured another
         * topology. Runtime calls update this same authority before delegating.
         */
        struct SnapshotCaptureSetup
        {
            /** Legal diagnostic topology requested for this runner. */
            enum class Mode
            {
                Disabled, ///< No diagnostic D2D copies belong to graph identity.
                Enabled,  ///< The retained filter belongs to graph identity.
            };

            Mode mode = Mode::Disabled;
            std::string output_directory;
            std::vector<std::string> filter;

            /** @return Whether graph-resident snapshot nodes are requested. */
            [[nodiscard]] bool enabled() const noexcept
            {
                return mode == Mode::Enabled;
            }
        } snapshot_capture_setup_;
        /** Rank-wide source authority; null on remote and non-overlay ranks. */
        std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
            moe_overlay_inference_transaction_coordinator_;
        /** Remote retained-graph scheduler; null on the continuation rank. */
        std::unique_ptr<MoEOverlayInferenceTransactionFollower>
            moe_overlay_inference_transaction_follower_;
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

        // Status
        std::atomic<bool> initialized_{false};
        std::string last_error_;
        mutable std::mutex error_mutex_;
        /** Serializes the idempotent application-startup preparation boundary. */
        mutable std::mutex inference_preparation_mutex_;

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
         * @brief One-shot proof that the completed resident step owned maintenance.
         *
         * `maybeApplyMoERebalance()` consumes this acknowledgement instead of
         * submitting a redundant no-op graph after a native parent has already
         * crossed every committed transaction boundary.
         */
        bool device_generation_embedded_moe_maintenance_pending_ack_{false};
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
        bool mpi_coordinated_mode_{false};
        /**
         * @brief Sole authority for the coordinated command protocol.
         *
         * This is zero for ordinary rank orchestration and the resolved
         * continuation root for heterogeneous ExpertOverlay. It is frozen
         * during plan construction and never changes during inference.
         */
        int mpi_coordinated_root_rank_{0};
        /** Sole typed authority for coordinated command admission. */
        MPIWorkerCommandLifecycle mpi_worker_command_lifecycle_{
            MPIWorkerCommandLifecycle::AcceptingCommands};
        std::shared_ptr<ITokenizer> tokenizer_;
        MTPStats mtp_stats_;
        std::unique_ptr<MTPDepthController> mtp_depth_controller_;
        PrefillChunkStats prefill_chunk_stats_;
        PrefixCacheRequestSummary prefix_request_summary_;
        bool mtp_bypassed_{false};
        bool mtp_bypass_recorded_for_request_{false};
        std::string mtp_bypass_reason_;
        /**
         * @brief Last completed request epoch, owned by the continuation rank.
         *
         * Non-authority ranks retain the exact value published by the
         * continuation rank solely so their next lifecycle broadcast can
         * validate and report it. They never independently advance this
         * counter; sparse wire identity has one authority even when root and
         * expert graph caches have different lifetimes.
         */
        uint64_t request_epoch_{0};
        /** Exact generation most recently installed into every sparse graph. */
        uint64_t moe_overlay_collective_generation_id_{0};
        /** Monotonic outer-command identity owned only by the continuation rank. */
        uint64_t moe_overlay_inference_command_sequence_{0};
    };

} // namespace llaminar2
