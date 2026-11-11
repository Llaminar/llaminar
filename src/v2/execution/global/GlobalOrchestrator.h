/**
 * @file GlobalOrchestrator.h
 * @brief Cross-machine MPI cluster inference orchestrator
 *
 * Top tier of the three-tier orchestration stack:
 *   GlobalOrchestrator       (cross-rank: MPI PP + Global TP)
 *     └─ RankOrchestrator    (per-rank: local devices)
 *          └─ DeviceGraphOrchestrator  (per-device: graph execution)
 *
 * Implements IInferenceRunner, so it is transparent to callers
 * (ChatCompletionHandler, BenchmarkMode, etc.). One instance per MPI rank;
 * each rank consults its own GlobalPPRankPlan.
 *
 * Phase 1-3 scope:
 * - Pure global-TP (all ranks, all layers) — pass-through to RankOrchestrator
 * - Pure global-PP (disjoint layer ranges) — MPI send/recv of activations
 * - Tail-rank sampling with MPI_Bcast of token
 * - Global TP + PP composition (2PP×2TP, mixed PP+TP topologies)
 *
 * @author David Sanftenberg
 * @date April 2026
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 * Stage checkpoint projection borrows the exact registry-owned child without
 * flattening its TP participant snapshots or duplicating payload/event owners.
 */

#pragma once

#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "../local_execution/graph/ReadAheadScheduling.h"
#include "../local_execution/orchestrators/IRankOrchestrator.h"
#include "../global_pp/GlobalPPTopology.h"
#include "OriginalCPUPipelineDomainOwners.h"
#include "../global_pp/GlobalPPRankPlan.h"
#include "../../interfaces/IMPIContext.h"
#include "../../collective/ITPContext.h"
#include "../../collective/ILocalTPContext.h"
#include "../../collective/IGlobalTPContext.h"
#include "../factory/FactoryPPStageConfig.h"
#include "../../utils/Sampler.h"
#include "../moe/MoEOverlayInferenceTransactionService.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    class TensorBase;
    class PreparedWeightStore;
    class MoEOverlayPipelineStageBinding;
    class CapturedTransferRankSession;
    class CapturedTransferChannel;
    class PipelineForwardGraphEdges;
    class PipelineDeviceGeneration;

    /**
     * @brief Stage-local weight ownership record for one pipeline domain.
     *
     * Layer scope is stage-local, but prepared handles belong to the model's
     * one additive store. Stages retain that same authority; constructing a
     * second store would replace or discard previously prepared bindings.
     */
    struct StageWeightContext
    {
        int stage_id = -1;
        std::string domain_name;
        std::optional<FactoryPPStageConfig> pp_stage_config;
        std::shared_ptr<PreparedWeightStore> prepared_store;
    };

    /**
     * @brief Owned runner for one pipeline stage/domain on this MPI rank.
     *
     * Local TP lifetime is owned by the RankOrchestrator itself. Global TP
     * retains its borrowed communicator here until after the runner retires.
     * pp_stage_config records the exact layer range executed by this runner.
     */
    struct StageRunnerEntry
    {
        int stage_id = -1;
        std::string domain_name;
        RankStageAction action;

        /**
         * @return The actual local TP context owned by the live stage runner,
         *         or nullptr for another runner kind. No duplicate context or
         *         independently cached pointer can outlive that runner.
         */
        [[nodiscard]] const ILocalTPContext *localTPContext() const noexcept;

        /// Lifetime owner for global TP context (IGlobalTPContext) if this stage
        /// uses cross-rank TP via GlobalTPContext/DomainCommunicatorRegistry.
        std::shared_ptr<IGlobalTPContext> global_tp_ctx;

        /// PP layer range for this stage runner. Records first_layer/last_layer
        /// (exclusive last_layer) as passed to the factory. Useful for verification.
        std::optional<FactoryPPStageConfig> pp_stage_config;

        /// Stage-local scope retaining the model-owned prepared handle store.
        std::shared_ptr<StageWeightContext> weight_context;

        /// Original stage expert owner, absent for dense stages. Retired after
        /// its runner; never a model-wide placement or a remote source handle.
        std::shared_ptr<const MoEOverlayPipelineStageBinding> moe_runtime;

        /// Actual original MPI owner, including dense stages; raw aliases never manufacture a second namespace.
        std::shared_ptr<IMPIContext> original_mpi_ctx;

        /// Runner is declared after its context owners so destruction tears the
        /// runner down before any raw TP/prepared-store pointers can expire.
        std::unique_ptr<IInferenceRunner> runner;
    };

    /**
     * @brief Owns exactly the explicitly bound local stages for a global PP rank.
     *
     * A remote pipeline head or tail has no local representative. Entry lookup
     * never substitutes an arbitrary local stage, and every prefix/checkpoint
     * transaction walks the same single stage inventory.
     */
    class StageRunnerRegistry
    {
    public:
        StageRunnerRegistry() = default;
        ~StageRunnerRegistry() = default;

        StageRunnerRegistry(const StageRunnerRegistry &) = delete;
        StageRunnerRegistry &operator=(const StageRunnerRegistry &) = delete;
        StageRunnerRegistry(StageRunnerRegistry &&) = default;
        StageRunnerRegistry &operator=(StageRunnerRegistry &&) = default;

        /** @brief Admit a unique, non-null, explicitly identified stage owner. */
        void add(StageRunnerEntry entry);
        /** @brief Destroy all local graph/worker/context borrowers before original CPU domain retirement. */
        void releaseAllAfterTerminalDrain();

        /** @return Whether no local stage has been admitted. */
        bool empty() const;
        /** @return Number of explicitly owned local stage runners. */
        size_t size() const;
        /** @return Whether the exact stage_id is local; never substitutes another stage. */
        bool hasRunnerForStage(int stage_id) const;

        /** @return Borrowed entry for stage_id, or nullptr for a nonlocal stage. */
        StageRunnerEntry *entryForStage(int stage_id);
        /** @copydoc entryForStage(int) */
        const StageRunnerEntry *entryForStage(int stage_id) const;
        /** @return Borrowed entry for domain_name, or nullptr if absent. */
        StageRunnerEntry *entryForDomain(const std::string &domain_name);
        /** @copydoc entryForDomain(const std::string &) */
        const StageRunnerEntry *entryForDomain(const std::string &domain_name) const;

        /** @return Runner owned by the exact stage_id, or nullptr if absent. */
        IInferenceRunner *runnerForStage(int stage_id);
        /** @copydoc runnerForStage(int) */
        const IInferenceRunner *runnerForStage(int stage_id) const;
        /** @return Runner owned by domain_name, or nullptr if absent. */
        IInferenceRunner *runnerForDomain(const std::string &domain_name);
        /** @copydoc runnerForDomain(const std::string &) */
        const IInferenceRunner *runnerForDomain(const std::string &domain_name) const;

        /** @return Local embedding owner, or nullptr when the pipeline head is remote. */
        IInferenceRunner *pipelineHeadRunner();
        /** @copydoc pipelineHeadRunner() */
        const IInferenceRunner *pipelineHeadRunner() const;
        /** @return Local vocabulary owner, or nullptr when the pipeline tail is remote. */
        IInferenceRunner *pipelineTailRunner();
        /** @copydoc pipelineTailRunner() */
        const IInferenceRunner *pipelineTailRunner() const;
        /** @return First local runner for rank-local metadata, not pipeline head authority. */
        IInferenceRunner *defaultRunner();
        /** @copydoc defaultRunner() */
        const IInferenceRunner *defaultRunner() const;
        /** @return Last local runner for local output access, not pipeline tail authority. */
        IInferenceRunner *lastLocalRunner();
        /** @copydoc lastLocalRunner() */
        const IInferenceRunner *lastLocalRunner() const;

        /** @brief Publish a request-boundary reset to every owned stage. */
        void clearCacheAll();
        /** @return True after every local stage retires reusable prefix state. */
        bool purgePrefixCacheAll();
        /** @return Every local stage's metadata publishers, without contacting remote stages. */
        PrefixCacheTelemetrySources prefixCacheTelemetrySourcesAll() const;
        /**
         * @brief Collect terminal prefix owners from every local stage.
         * @param shutdown Cohort retaining original cache owners until terminal persistence.
         * @throws std::exception If retaining source metadata fails.
         */
        void collectPrefixCacheShutdownAll(PrefixCacheShutdown &shutdown) const;
        /**
         * @brief Admit one immutable stop-token policy into every local runner.
         *
         * Global orchestration is a transparent ownership boundary: request
         * controls belong to the leaf runners that own captured execution, not
         * to this registry.  A null child or a rejected policy is therefore a
         * fatal topology error.  The method stops immediately rather than
         * leaving later participants configured under a partially accepted
         * request.
         *
         * @param stop_tokens Complete model-specific stop-token set.
         * @return True after every registered runner accepts the policy.
         * @throws std::logic_error if the registry has no usable runner.
         * @throws std::runtime_error if any runner rejects the policy.
         */
        bool configureMTPRequestStopTokensAll(
            const std::vector<int32_t> &stop_tokens);
        /**
         * @brief Admit one immutable penalty policy into every local runner.
         *
         * The policy is staged at request admission and later published by each
         * leaf on its exact producer stream.  Global orchestration must never
         * retain a wrapper-only copy because captured verifier consumers cannot
         * observe such host state.
         *
         * @param policy Request-constant presence and frequency penalties.
         * @return True after every registered runner accepts the policy.
         * @throws std::logic_error if the registry has no usable runner.
         * @throws std::runtime_error if any runner rejects the policy.
         */
        bool configureMTPRequestPenaltyPolicyAll(
            const MTPRequestPenaltyPolicy &policy);
        /** @brief Forward hosted verifier ordering to every original local stage, including non-tail ranks. */
        void setMTPAllPositionVerifierSyncDeferralAll(bool enabled);
        /** @brief Forward hosted main-forward ordering without borrowing a remote vocabulary owner. */
        void setMTPMainDecodeSyncDeferralAll(bool enabled);
        /** @copydoc IInferenceRunner::setSkipLogitsGatherDecode */
        void setSkipLogitsGatherDecodeAll(bool skip);
        /** @copydoc IInferenceRunner::setSkipLogitsGatherPrefill */
        void setSkipLogitsGatherPrefillAll(bool skip);
        /** @copydoc IInferenceRunner::setSuppressTimeline */
        void setSuppressTimelineAll(bool suppress);
        /** @copydoc IInferenceRunner::setAccumulatePrefill */
        void setAccumulatePrefillAll(bool accumulate);
        /** @brief Flush diagnostics from each owned stage without owning inference state. */
        void flushStageTimelineAll();
        /** @return Whether every local stage supports the existing chained-sidecar contract. */
        bool supportsChainedMTPDraftsAll() const;
        /** @copydoc IInferenceRunner::forwardMTP */
        bool forwardMTPAll(int32_t draft_condition_token);
        /** @copydoc IInferenceRunner::forwardMTPFromLastDraft */
        bool forwardMTPFromLastDraftAll(int32_t draft_condition_token, int position_id);
        /** @copydoc IInferenceRunner::commitMTPShiftedRowsFromLastForward */
        bool commitMTPShiftedRowsFromLastForwardAll(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens);
        /**
         * @brief Commit accepted grouped verifier rows through every local stage.
         *
         * Grouped MTP verification may publish only a suffix of the verifier
         * forward rows into shifted MTP state.  The extra row-count, base
         * offset, and discard arguments preserve the exact single-rank
         * publication contract when GlobalTP wraps the real rank runner.
         */
        bool commitMTPShiftedRowsFromPartialForwardAll(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens,
            int main_forward_token_count,
            bool allow_speculative_discard = false,
            int position_offset_override = -1,
            int already_appended_shifted_kv_tokens = -1);
        /** @copydoc IInferenceRunner::commitMTPShiftedRowFromCurrentTerminalHidden */
        bool commitMTPShiftedRowFromCurrentTerminalHiddenAll(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1);
        /** @copydoc IInferenceRunner::commitMTPShiftedRowFromCheckpointTerminalHidden */
        bool commitMTPShiftedRowFromCheckpointTerminalHiddenAll(
            const PrefixStateSnapshot &checkpoint,
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1);
        /** @copydoc IInferenceRunner::ensureMTPCheckpointTerminalHidden */
        bool ensureMTPCheckpointTerminalHiddenAll();
        /** @copydoc IInferenceRunner::setComputeAllPositionLogits */
        bool setComputeAllPositionLogitsAll(bool enabled);
        /**
         * @brief Toggle compact row-indexed all-position verifier logits.
         *
         * GlobalTP is a transparent IInferenceRunner wrapper.  The grouped
         * verifier planner installs compact verifier rows on the wrapper, so
         * this helper fans the shape toggle out to the actual rank-local
         * runner that owns graph execution and LM-head row packing.
         */
        bool setComputeRowIndexedAllPositionLogitsAll(bool enabled, int row_count);
        /**
         * @brief Install grouped verifier input metadata on every local runner.
         *
         * Row-indexed verifier forwards need the compact row plan and token
         * ordering to live beside the runner that executes the graph.  Without
         * this fan-out, GlobalTP would enable row-indexed logits but leave the
         * child runner without row metadata.
         */
        bool setMTPSpecVerifierInputPlanAll(const MTPSpecDecodeVerifierInputPlan &plan);
        /**
         * @brief Clear stale grouped verifier row metadata from local runners.
         */
        void clearMTPSpecVerifierInputPlanAll();
        /**
         * @brief Publish the accepted grouped verifier prefix through all local runners.
         *
         * The same MTPSpecStepPlanBatch must reach every GlobalTP participant
         * so positions, KV, shifted MTP KV, and recurrent payloads advance in
         * lockstep after the rank-wide verifier decision.
         */
        bool publishGroupedDecodeEquivalentMTPSpecStateBatchAll(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr);
        /** @return Largest immutable placement epoch published by the local stage owners. */
        uint64_t moePlacementEpochAll() const;
        /** @return Largest completed runtime movement epoch across local stages. */
        uint64_t moeRuntimeMovementEpochAll() const;
        /**
         * @brief Coordinate one prefix lookup across every local PP stage.
         *
         * Each stage owns a distinct payload/fingerprint but all stages must
         * restore the same logical token boundary. The registry retains each
         * child result so later population and terminal-state restoration use
         * the exact handles returned by that child.
         *
         * @param tokens Complete request prefix token sequence.
         * @return Common locally restorable prefix and typed terminal requirements.
         */
        /** @brief Install one immutable response policy on every local execution stage. */
        bool admitSparseQueryExecutionProgramAll(const DeviceGenerationAdmissionRequest &request);
        /** @brief Expire host response policy on every stage without retiring retained graphs. */
        void endSparseQueryExecutionProgramAll() noexcept;
        PrefixLookupResult lookupPrefixAll(const std::vector<int32_t> &tokens);
        /** @copydoc IInferenceRunner::prefixRestoreMetadata */
        PrefixRestoreMetadata prefixRestoreMetadataAll(const PrefixLookupResult &hit) const;
        /** @copydoc IInferenceRunner::preparePrefixHarvest */
        bool preparePrefixHarvestAll(
            const PrefixLookupResult &admission,
            const std::vector<int32_t> &tokens,
            const PrefixHarvestSchedule &schedule);
        /**
         * @brief Restore the coordinated prefix into every local PP stage.
         * @param hit Aggregate result returned by @ref lookupPrefixAll.
         * @param seq_idx Sequence slot receiving the restored state.
         * @return True after every child imports its own clamped payload.
         */
        bool populatePrefixAll(const PrefixLookupResult &hit, int seq_idx = 0);
        /**
         * @brief Archive one completed prefix in every local PP stage cache.
         * @param admission Coordinated lookup whose child-owned handles are retained.
         * @param tokens Complete prompt token sequence.
         * @param prompt_token_count Number of live prompt tokens.
         * @return True after every child publishes its stage-owned payload.
         */
        bool harvestPrefixAll(
            const PrefixLookupResult &admission,
            const std::vector<int32_t> &tokens,
            int prompt_token_count);
        /**
         * @brief Restore stage-specific terminal state for a full prefix hit.
         * @param hit Aggregate coordinated hit naming the common token boundary.
         * @return True when every child satisfies its typed terminal contract.
         */
        bool restorePrefixTerminalStateAll(const PrefixLookupResult &hit);
        /**
         * @brief Return the local PP participant runtime-state observation.
         *
         * A single local stage is returned verbatim. Multiple local stages are
         * folded in stable pipeline order so cache inventories and terminal
         * evidence remain complete without inventing a second state authority.
         * @param capture_policy Immutable per-cache ranges forwarded to every stage.
         * @return Combined participant-local runtime evidence.
         */
        PrefixRuntimeStateSnapshot prefixStateProbeAll(
            const PrefixProbeCapturePolicy &capture_policy) const;
        /** @brief Capture seq_idx on all stages; reject mismatched logical frontiers. */
        PrefixStateSnapshot captureLivePrefixStateAll(int seq_idx = 0) const;
        /** @brief Capture the same sealed request cursor on all stages, retaining child leases. */
        PrefixStateSnapshot captureLivePrefixCheckpointAll(
            const PrefixCheckpointCaptureRequest &request) const;

        /**
         * @brief Borrow one exact local stage checkpoint from this registry's capture envelope.
         * @param snapshot Complete aggregate returned by captureLivePrefixCheckpointAll.
         * @param runner Actual owned stage runner, never an inferred terminal index.
         * @return Its valid child, or nullptr for incomplete cardinality, a foreign
         * runner, an invalid child or a stale child frontier.
         * The child remains intact: its nested TP participant snapshots and ready
         * event/source handles are independently authenticated by that stage runner.
         */
        [[nodiscard]] const PrefixStateSnapshot *prefixCheckpointForRunner(
            const PrefixStateSnapshot &snapshot, const IInferenceRunner &runner) const;
        /** @brief Restore each stage from its own child snapshot for seq_idx. */
        bool restoreLivePrefixStateAll(const PrefixStateSnapshot &snapshot, int seq_idx = 0);
        /** @brief Truncate every stage to cached_tokens for the same seq_idx. */
        bool truncateLivePrefixStateAll(int cached_tokens, int seq_idx = 0);
        /** @return First local unsupported reason, or an empty string if every runner accepts. */
        std::string mtpDecodeUnsupportedReasonAll() const;
        /** @brief Enable diagnostic capture on every local stage using output_dir. */
        void enableSnapshotCaptureAll(const std::string &output_dir);
        /** @brief Install one immutable capture filter on every local stage. */
        void setSnapshotCaptureFilterAll(
            const std::vector<std::string> &keys);
        /** @brief Disable diagnostic capture on every local stage. */
        void disableSnapshotCaptureAll();
        /** @brief Discard local diagnostic snapshots, not model or KV state. */
        void clearSnapshotsAll();
        /** @brief Resolve a global-layer key to its owning stage and return its data/count. */
        const float *getSnapshot(const std::string &key, size_t &out_size) const;
        /** @brief Resolve a global-layer key with the owning stage's exact tensor shape. */
        SnapshotInfo getSnapshotWithShape(const std::string &key) const;
        /** @return Deduplicated diagnostic keys projected into global layer coordinates. */
        std::vector<std::string> snapshotKeysAll() const;

    private:
        std::vector<StageRunnerEntry> entries_;
        /** Child-owned lookup handles retained until this prefix transaction ends. */
        std::vector<PrefixLookupResult> last_prefix_hits_;
    };

    /**
     * @brief Moves activations between stage runners and MPI peers.
     */
    class StageActivationRouter
    {
    public:
        /**
         * @brief Complete one declared live hosted activation edge.
         * @param action Original local handoff or MPI send/receive action.
         * @param registry Actual locally owned source and destination runners.
         * @param mpi_ctx Original communicator used by the declared edge.
         * @param rank Original rank owning this action.
         * @param last_seq_len Actual live rows produced by the preceding stage.
         * @param d_model Residual width of each live row.
         * @param activation_buffer Original cold-admitted CPU boundary bank.
         * @param phase Explicit prefill or decode role of this completed range.
         * @return True after the exact host-visible payload completes.
         */
        bool executeTransfer(const RankTransferAction &action,
                             StageRunnerRegistry &registry,
                             IMPIContext &mpi_ctx,
                             int rank,
                             int last_seq_len,
                             int d_model,
                             std::shared_ptr<TensorBase> &activation_buffer,
                             ForwardExecutionPhase phase = ForwardExecutionPhase::Decode) const;

    private:
        static size_t transferElementCount(int last_seq_len, int d_model);
        static void ensureActivationBufferSize(std::shared_ptr<TensorBase> &activation_buffer,
                                               size_t num_elements);
    };

    /**
     * @brief Cross-machine MPI cluster inference orchestrator
     *
     * Coordinates global pipeline parallelism (PP) and/or global tensor
     * parallelism (TP) across MPI ranks, delegating per-rank execution to
     * an IRankOrchestrator (or IInferenceRunner for single-device stages).
     *
     * Execution model:
     * - Each rank owns one GlobalOrchestrator instance
     * - Each rank independently executes its GlobalPPRankPlan
     * - Stages alternate: EXECUTE_STAGE → TRANSFER → EXECUTE_STAGE → ...
     * - Only the pipeline tail rank produces valid logits
     * - The tail rank samples and broadcasts the token to all ranks
     */
    class GlobalOrchestrator : public IInferenceRunner
    {
    public:
        // =================================================================
        // Configuration
        // =================================================================

        struct Config
        {
            // Cluster topology
            GlobalPPTopology topology;       ///< Cluster-wide stage layout
            int rank = 0;                    ///< This MPI rank
            int world_size = 1;              ///< Total MPI ranks

            // MPI context (not owned)
            IMPIContext *mpi_ctx = nullptr;

            // Global TP context (optional, not owned)
            ITPContext *global_tp_ctx = nullptr;

            // Exact per-stage local owners, including a one-stage TP domain.
            std::vector<StageRunnerEntry> stage_runners;

            // Model metadata
            int vocab_size = 0;              ///< Full vocabulary size
            int d_model = 0;                 ///< Hidden state dimension
            std::string architecture_name = "unknown";

            /// Same original shared owner as every local stage; raw mpi_ctx aliases it exactly for native rank transport.
            std::shared_ptr<IMPIContext> original_mpi_ctx;
            /// Sole original admitted CPU expert contexts; retired after actual local stage graphs disappear.
            std::shared_ptr<OriginalCPUPipelineDomainOwners> original_cpu_domains;
            /// Exact original rank PMA and shared stage row envelope for hosted activation storage.
            std::shared_ptr<PhysicalMemoryAuthority> physical_memory;
            int hosted_activation_rows=0;
        };

        // =================================================================
        // Construction / Destruction
        // =================================================================

        /**
         * @brief Construct from configuration
         *
         * Builds the rank's execution plan from the topology, allocates
         * activation transfer buffers (for PP), and takes ownership of
         * the rank/stage runners.
         *
         * @param config Configuration (explicit stage-runner ownership transferred).
         * @throws std::invalid_argument if config is invalid
         */
        explicit GlobalOrchestrator(Config config);

        /** @brief Retire original local graph/cohort owners before transport bindings and context owners. */
        ~GlobalOrchestrator() override;

        /** @return Whether the complete declared multi-stage pipeline consists of actual GPU stages. */
        bool requiresOriginalRankGPUTransport() const noexcept;
        /** @brief Validate the same original rank namespace before any all-rank session collective.
         * @param original Actual shared MPI owner retained by OrchestrationRunner and StageRunnerFactory.
         * @throws std::invalid_argument For foreign rank/topology/context owners. */
        void validateOriginalRankGPUTransportContext(const std::shared_ptr<IMPIContext> &original) const;
        /** @brief One-time cold bind; physical session remains enclosed by original runner shutdown consensus.
         * @param original Same validated original MPI owner. @param session Existing all-rank session.
         * @throws std::invalid_argument For replacement or after graph ports were installed. */
        void bindOriginalRankGPUTransportSession(const std::shared_ptr<IMPIContext> &original,
            std::shared_ptr<CapturedTransferRankSession> session);

        // Non-copyable, non-movable (owns unique_ptrs)
        GlobalOrchestrator(const GlobalOrchestrator &) = delete;
        GlobalOrchestrator &operator=(const GlobalOrchestrator &) = delete;
        GlobalOrchestrator(GlobalOrchestrator &&) = delete;
        GlobalOrchestrator &operator=(GlobalOrchestrator &&) = delete;

        // =================================================================
        // IInferenceRunner — Core Inference API
        // =================================================================

        /** @copydoc IInferenceRunner::admitSparseQueryExecutionProgram */
        bool admitSparseQueryExecutionProgram(const DeviceGenerationAdmissionRequest &request) override;
        /** @copydoc IInferenceRunner::endSparseQueryExecutionProgram */
        void endSparseQueryExecutionProgram() noexcept override;

        bool forward(const int *tokens, int seq_len) override;
        /**
         * @brief Preserve explicit prefill through the original retained serving family.
         * @param tokens Immutable real request tokens beginning at the current cursor.
         * @param seq_len Positive live token count, independent of retained bank capacity.
         * @return True after all original stages complete the shared retained schedule.
         */
        bool forwardPrefill(const int *tokens, int seq_len) override;
        /**
         * @brief Execute a CPU grouped-MTP verifier transaction through GlobalPP/TP.
         *
         * The wrapper must preserve `GroupedMTPVerifier` as a semantic graph
         * role at every local stage. Routing this call through ordinary
         * `forward()` would build a main-inference graph and omit the recurrent
         * verifier checkpoints consumed by accepted-state publication.
         *
         * @param token_batches Logical verifier rows for every request.
         * @return True after every local execute/transfer step succeeds.
         */
        bool forwardGroupedMTPVerifierWithHostTokenIds(
            const std::vector<std::vector<int>> &token_batches) override;
        const float *logits() const override;
        int vocab_size() const override;
        void clear_cache() override;
        /** @copydoc IInferenceRunner::purgePrefixCache */
        bool purgePrefixCache() override;
        /** @copydoc IInferenceRunner::collectPrefixCacheShutdown */
        void collectPrefixCacheShutdown(PrefixCacheShutdown &shutdown) const override;
        /** @copydoc IInferenceRunner::prefixCacheTelemetrySources */
        PrefixCacheTelemetrySources prefixCacheTelemetrySources() const override;
        int get_position() const override;
        ExecutionPath executionPath() const override;
        const char *architecture() const override;
        bool forwardMTP(int32_t draft_condition_token) override;
        /**
         * @brief True when every local participant can execute chained MTP drafts.
         *
         * NodeTP/GlobalTP uses one rank-wide draft token broadcast.  Chained
         * depth-2/3 drafts are safe only when each rank-local participant can
         * consume the previous sidecar hidden state at the same logical shifted
         * MTP position.
         */
        bool supportsChainedMTPDrafts() const override;
        bool forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id) override;
        bool commitMTPShiftedRowsFromLastForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens) override;
        /**
         * @brief Forward grouped verifier shifted-row publication to local stages.
         *
         * IInferenceRunner's default implementation intentionally has a narrow
         * legacy shape.  GlobalTP must override it so grouped publication keeps
         * the verifier row count, base offset, and speculative-discard
         * semantics supplied by OrchestrationRunner.
         */
        bool commitMTPShiftedRowsFromPartialForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens,
            int main_forward_token_count,
            bool allow_speculative_discard = false,
            int position_offset_override = -1,
            int already_appended_shifted_kv_tokens = -1) override;
        bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override;
        bool commitMTPShiftedRowFromCheckpointTerminalHidden(
            const PrefixStateSnapshot &checkpoint,
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override;
        bool ensureMTPCheckpointTerminalHidden() override;
        const float *mtpLogits() const override;
        /**
         * @brief Forward request stop-token admission to every graph owner.
         */
        bool configureMTPRequestStopTokens(
            const std::vector<int32_t> &stop_tokens) override;
        /** @copydoc IInferenceRunner::publishMTPDirectTokenHistory */
        bool publishMTPDirectTokenHistory(int target_sample_slot,
            MTPDirectTokenHistoryDisposition disposition) override;
        /**
         * @brief Forward request penalty admission to every graph owner.
         */
        bool configureMTPRequestPenaltyPolicy(
            const MTPRequestPenaltyPolicy &policy) override;
        bool setComputeAllPositionLogits(bool enabled) override;
        /**
         * @brief Forward compact verifier LM-head row packing to local runners.
         */
        bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override;
        /**
         * @brief Forward grouped verifier input metadata to local runners.
         */
        bool setMTPSpecVerifierInputPlan(
            const MTPSpecDecodeVerifierInputPlan &plan) override;
        /**
         * @brief Clear grouped verifier input metadata from local runners.
         */
        void clearMTPSpecVerifierInputPlan() override;
        const float *getAllPositionLogits() const override;
        bool hasMTPLogitsLocal() const override;
        LogitsLocalInfo getMTPLogitsLocalInfo() const override;
        LogitsLocalInfo consumeMTPLogitsLocalInfoForSampling() override;
        LogitsLocalInfo consumeMTPLogitsLocalInfoForHostGather() override;
        bool hasAllPositionLogitsLocal() const override;
        LogitsLocalInfo getAllPositionLogitsLocalInfo() const override;
        LogitsLocalInfo consumeAllPositionLogitsLocalInfoForSampling() override;
        LogitsLocalInfo consumeAllPositionLogitsLocalInfoForHostGather() override;
        /**
         * @brief Publish accepted grouped verifier rows through GlobalTP children.
         */
        bool publishGroupedDecodeEquivalentMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override;
        std::string mtpDecodeUnsupportedReason() const override;
        bool supportsMTPTokenCoordination() const override;
        uint64_t moePlacementEpoch() const override;
        uint64_t moeRuntimeMovementEpoch() const override;
        /** @brief Coordinate prefix lookup across the local stages on this rank. */
        PrefixLookupResult lookupPrefix(
            const std::vector<int32_t> &tokens) override;
        /** @copydoc IInferenceRunner::prefixRestoreMetadata */
        PrefixRestoreMetadata prefixRestoreMetadata(const PrefixLookupResult &hit) const override;
        /** @brief Populate every local stage from its retained lookup handles. */
        /** @copydoc IInferenceRunner::preparePrefixHarvest */
        bool preparePrefixHarvest(
            const PrefixLookupResult &admission,
            const std::vector<int32_t> &tokens,
            const PrefixHarvestSchedule &schedule) override;

        bool populatePrefix(
            const PrefixLookupResult &hit,
            int seq_idx = 0) override;
        /** @brief Harvest one prefix payload from every local stage. */
        bool harvestPrefix(
            const PrefixLookupResult &admission,
            const std::vector<int32_t> &tokens,
            int prompt_token_count) override;
        /** @brief Restore typed terminal state on every local stage. */
        bool restorePrefixTerminalState(
            const PrefixLookupResult &hit) override;
        int sampleGreedyFromMTPLogitsOnDevice() override;
        int sampleGreedyFromAllPositionLogitsOnDevice(int row) override;
        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override;
        PrefixStateSnapshot captureLivePrefixCheckpoint(
            const PrefixCheckpointCaptureRequest &request) const override;
        bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0) override;
        bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0) override;
        /**
         * @brief Observe the complete rank-local pipeline prefix state.
         * @param capture_policy Exact diagnostic ranges, forwarded unchanged to participants.
         * @return Immutable combined cache and terminal-state evidence.
         */
        PrefixRuntimeStateSnapshot prefixStateProbe(
            const PrefixProbeCapturePolicy &capture_policy = PrefixProbeCapturePolicy::fromEnvironment()) const override;

        // =================================================================
        // IInferenceRunner — GPU-side Sampling
        // =================================================================

        /**
         * @brief Greedy sampling with cross-rank broadcast
         *
         * On the tail rank: delegates to the local tail stage runner.
         * On all ranks: MPI_Bcast the sampled token from tail to all others.
         *
         * @return Token ID on all ranks
         */
        int sampleGreedyOnDevice() override;

        /** @return The resolved vocabulary-head leader for request coordination. */
        std::optional<int> requestAuthorityRank() const override { return tail_rank_; }

        /**
         * @brief Declare participation in the existing greedy token broadcast.
         * @param params Admitted sampler policy.
         * @return True for the collective greedy sampler; CPU stochastic
         *         sampling remains local to the declared request authority.
         */
        bool requiresMPICoordinatedDecodeSampling(const SamplingParams &params) const override;

        /**
         * @brief Full sampling with cross-rank broadcast
         *
         * On the tail rank: delegates to the local tail stage runner.
         * On all ranks: MPI_Bcast the sampled token from tail to all others.
         *
         * @return Token ID on all ranks
         */
        int sampleOnDevice(const SamplingParams &params) override;

        // =================================================================
        // IInferenceRunner — Logits Gather Control
        // =================================================================

        void setSkipLogitsGatherDecode(bool skip) override;
        void setSkipLogitsGatherPrefill(bool skip) override;

        // =================================================================
        // IInferenceRunner — Timeline/Profiling
        // =================================================================

        void setSuppressTimeline(bool suppress) override;
        void setAccumulatePrefill(bool accumulate) override;
        void flushStageTimeline() override;

        // =================================================================
        // IInferenceRunner — Hidden State API (Pipeline Parallelism)
        // =================================================================

        TensorBase *getHiddenState() override;
        const TensorBase *getHiddenState() const override;
        void setHiddenState(TensorBase *hidden_state) override;
        bool hasHiddenStateInput() const override;
        void clearHiddenStateInput() override;

        // =================================================================
        // IInferenceRunner — Snapshot API (delegate to rank runner)
        // =================================================================

        void enableSnapshotCapture(const std::string &output_dir) override;
        /** @brief Forward the graph-identity snapshot filter to local stages. */
        void setSnapshotCaptureFilter(
            const std::vector<std::string> &keys) override;
        void disableSnapshotCapture() override;
        void clearSnapshots() override;
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;
        SnapshotInfo getSnapshotWithShape(const std::string &key) const override;
        std::vector<std::string> getSnapshotKeys() const override;

        // =================================================================
        // IInferenceRunner — Device & Logits Local
        // =================================================================

        DeviceId primaryDeviceId() const override;
        bool hasLogitsLocal() const override;
        LogitsLocalInfo getLogitsLocalInfo() const override;
        LogitsLocalInfo consumeLogitsLocalInfoForSampling() override;
        LogitsLocalInfo consumeLogitsLocalInfoForHostGather() override;

        // =================================================================
        // Query API
        // =================================================================

        /** @brief True if this rank runs the embedding layer */
        bool isPipelineHead() const;

        /** @brief True if this rank runs the LM head */
        bool isPipelineTail() const;

        /** @brief Number of PP stages in the topology */
        int pipelineDepth() const;

        /** @brief This rank's execution plan */
        const GlobalPPRankPlan &rankPlan() const;

        /** @brief The cluster topology */
        const GlobalPPTopology &topology() const;

        /** @brief Number of local stage/domain runners owned by this rank */
        size_t stageRunnerCount() const;
        /** @copydoc IInferenceRunner::requireMoEPipelineRuntimeOwnership */
        void requireMoEPipelineRuntimeOwnership(MoEPipelineOwnershipValidation frontier) const override;
        /** @copydoc IInferenceRunner::setMoEOverlayCollectiveRequestGeneration */
        bool setMoEOverlayCollectiveRequestGeneration(uint64_t generation_id) override;

        /** @brief Stage runner entry for a local stage, or nullptr if absent */
        StageRunnerEntry *stageRunnerEntryForStage(int stage_id);
        const StageRunnerEntry *stageRunnerEntryForStage(int stage_id) const;

        /** @brief Stage runner entry for a local named domain, or nullptr if absent */
        StageRunnerEntry *stageRunnerEntryForDomain(const std::string &domain_name);
        const StageRunnerEntry *stageRunnerEntryForDomain(const std::string &domain_name) const;

        /** @brief Runner for a local stage, or nullptr if absent */
        IInferenceRunner *stageRunnerForStage(int stage_id);
        const IInferenceRunner *stageRunnerForStage(int stage_id) const;

        /** @brief Runner for a local named domain, or nullptr if absent */
        IInferenceRunner *stageRunnerForDomain(const std::string &domain_name);
        const IInferenceRunner *stageRunnerForDomain(const std::string &domain_name) const;

        /** @brief The global TP context (may be nullptr for pure PP) */
        ITPContext *globalTPContext() const;

        /** @brief Get weight shard info for a stage this rank executes */
        const WeightShardInfo *weightShardForStage(int stage_id) const;

    private:
        // =================================================================
        // Internal Execution
        // =================================================================

        /**
         * @brief Execute the existing hosted stage/transfer sequence for one live range.
         * @param tokens Original live request tokens, already offset for this range.
         * @param seq_len Exact live rows crossing every activation boundary.
         * @param phase Original explicit prefill or decode role.
         * @param prefill Original one-chunk prefill policy, or null for ordinary forward.
         * @param pad_token_id Token owned by the admitted prefill capture policy.
         * @param allow_padding Whether that prefill policy admits initialized physical rows.
         * @return True after the original local stages and transfers complete.
         */
        bool forwardHosted(const int *tokens, int seq_len, ForwardExecutionPhase phase,
            const PrefillChunkSchedulerPolicy *prefill = nullptr,
            int pad_token_id = 0, bool allow_padding = false);

        /**
         * @brief Execute one original hosted stage with its typed forward role.
         * @param action Original local stage action.
         * @param tokens Live request tokens, including native nonembedding ingress.
         * @param seq_len Exact live rows.
         * @param phase Original explicit prefill or decode role.
         * @param prefill Original one-chunk prefill policy, or null for ordinary forward.
         * @param pad_token_id Token owned by the admitted prefill capture policy.
         * @param allow_padding Whether that prefill policy admits initialized physical rows.
         * @return True after the stage completes on its existing runner.
         */
        bool executeStage(const RankStageAction &action, const int *tokens, int seq_len,
            ForwardExecutionPhase phase, const PrefillChunkSchedulerPolicy *prefill,
            int pad_token_id, bool allow_padding);

        /**
         * @brief Execute one local stage with the typed grouped-verifier role.
         *
         * Pipeline stages without an embedding still receive the logical row
         * geometry; their graph omits token embedding and consumes the hidden
         * activation installed by the preceding transfer.
         */
        bool executeGroupedMTPVerifierStage(
            const RankStageAction &action,
            const std::vector<std::vector<int>> &token_batches);

        /**
         * @brief Execute a single TRANSFER step (MPI Send or Recv)
         *
         * Uses synchronous MPI_Send / MPI_Recv for activation transfer.
         */
        bool executeTransfer(const RankTransferAction &action,
            ForwardExecutionPhase phase = ForwardExecutionPhase::Decode);

        /**
         * @brief Resolve the unique vocabulary-head domain's request leader.
         * @return Its owner rank, or first participating rank for a TP domain.
         * @throws std::invalid_argument for missing, duplicate or invalid ownership.
         */
        int findTailRank() const;

        // =================================================================
        // State
        // =================================================================

        /** @brief Prepare complete original-rank activation/reverse metadata ports before generation capture.
         * @param plan Exact already-agreed serving bucket family; no request state is executed.
         * @throws std::exception Through original all-rank initialization consensus for incomplete stage ownership. */
        void prepareOriginalRankPipelinePorts(const ServingGraphFamilyMaterializationPlan &plan);

    public:
        /** @brief Freeze every original rank-local graph body and its exact rank ports.
         * @param plan Complete admitted serving family, identical on the original ranks.
         * @return True only after original local cohort capture completes.
         * @throws std::exception For substituted context/roles, stale family, or partial capture.
         */
        /** @copydoc IInferenceRunner::servingGraphPreparationKind */
        ServingGraphPreparationKind servingGraphPreparationKind() const noexcept override;
        /** @copydoc IInferenceRunner::forwardRestoredPrefixMTPDecodeBridge */
        bool forwardRestoredPrefixMTPDecodeBridge(const RestoredPrefixMTPDecodeBridgeRequest &request) override;
        /** @return Whether the exact original pipeline cohort/transport/service was prepared on this rank. */
        bool originalRankPipelineReady() const noexcept { return rank_service_ready_; }
        bool materializeServingGraphFamilyWithoutLaunch(const ServingGraphFamilyMaterializationPlan &plan) override;
        /** @copydoc IInferenceRunner::forwardBatchWithDeviceTokenIds */
        bool forwardBatchWithDeviceTokenIds(const std::vector<std::vector<int>> &token_batches,
            const void *token_ids_device,
            int padded_seq_len) override;
        /** @copydoc IInferenceRunner::prepareMTPVerifierInputTokensOnDevice */
        const void * prepareMTPVerifierInputTokensOnDevice(int32_t first_token,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override;
        /** @copydoc IInferenceRunner::prepareMTPVerifierInputTokenBatchOnDevice */
        const void * prepareMTPVerifierInputTokenBatchOnDevice(const DeviceMTPVerifierInputBatchRequest *requests,
            int request_count,
            int logical_padded_seq_len) override;
        /** @copydoc IInferenceRunner::prepareMTPVerifierInputTokensOnDeviceFromHostRow */
        const void * prepareMTPVerifierInputTokensOnDeviceFromHostRow(const int32_t *verifier_tokens,
            int total_verifier_input_tokens,
            int draft_token_count) override;
        /** @copydoc IInferenceRunner::forwardMTPForDeviceSampling */
        bool forwardMTPForDeviceSampling(int32_t draft_condition_token) override;
        /** @copydoc IInferenceRunner::supportsMTPSidecarSampleFusion */
        bool supportsMTPSidecarSampleFusion() const override;
        /** @copydoc IInferenceRunner::supportsMTPSidecarLogitsStreamHandoff */
        bool supportsMTPSidecarLogitsStreamHandoff() const override;
        /** @copydoc IInferenceRunner::supportsMTPDeviceDraftTokenInput */
        bool supportsMTPDeviceDraftTokenInput() const override;
        /** @copydoc IInferenceRunner::supportsMTPSidecarPreservesMainState */
        bool supportsMTPSidecarPreservesMainState() const override;
        /** @copydoc IInferenceRunner::supportsLogicalMTPVerifierBaseCheckpoint */
        bool supportsLogicalMTPVerifierBaseCheckpoint() const override;
        /** @copydoc IInferenceRunner::supportsMTPShiftedRowReuseFromSidecar */
        bool supportsMTPShiftedRowReuseFromSidecar() const override;
        /** @copydoc IInferenceRunner::requiresMTPDecodeEquivalentVerifierReplay */
        bool requiresMTPDecodeEquivalentVerifierReplay() const override;
        /** @copydoc IInferenceRunner::supportsMTPSpecStatePublication */
        bool supportsMTPSpecStatePublication() const override;
        /** @copydoc IInferenceRunner::supportsDeviceResidentMTPSpecStatePublication */
        bool supportsDeviceResidentMTPSpecStatePublication() const override;
        /** @copydoc IInferenceRunner::publishAcceptedMTPSpecState */
        bool publishAcceptedMTPSpecState(const MTPSpecStepPlan &plan,
            std::string *error = nullptr) override;
        /** @copydoc IInferenceRunner::publishAcceptedMTPSpecStateBatch */
        bool publishAcceptedMTPSpecStateBatch(const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override;
        /** @copydoc IInferenceRunner::forwardMTPFromLastDraftForDeviceSampling */
        bool forwardMTPFromLastDraftForDeviceSampling(int32_t draft_condition_token,
            int position_id) override;
        /** @copydoc IInferenceRunner::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling */
        bool forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(int draft_sample_slot,
            int position_offset) override;
        /** @copydoc IInferenceRunner::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling */
        bool forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(int target_sample_slot) override;
        /** @copydoc IInferenceRunner::forwardMTPFromDeviceTargetAtLivePositionAndSampleGreedyToDeviceDraftSlot */
        bool forwardMTPFromDeviceTargetAtLivePositionAndSampleGreedyToDeviceDraftSlot(int target_sample_slot,
            int draft_sample_slot,
            int32_t *out_token) override;
        /** @copydoc IInferenceRunner::forwardMTPFromDeviceResidentLogicalStateForDeviceSampling */
        bool forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index = 0) override;
        /** @copydoc IInferenceRunner::forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot */
        bool forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int draft_sample_slot,
            int32_t *out_token) override;
        /** @copydoc IInferenceRunner::forwardMTPAndSampleGreedy */
        bool forwardMTPAndSampleGreedy(int32_t draft_condition_token, int32_t *out_token) override;
        /** @copydoc IInferenceRunner::forwardMTPAndSampleGreedyToDeviceDraftSlot */
        bool forwardMTPAndSampleGreedyToDeviceDraftSlot(int32_t draft_condition_token,
            int draft_sample_slot,
            int32_t *out_token) override;
        /** @copydoc IInferenceRunner::forwardMTPBatchAndSampleGreedy */
        bool forwardMTPBatchAndSampleGreedy(const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens) override;
        /** @copydoc IInferenceRunner::advanceMTPRequestBatchConditionOnDevice */
        bool advanceMTPRequestBatchConditionOnDevice(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            const SamplingParams &params,
            const uint64_t *stochastic_position_seeds = nullptr) override;
        /** @copydoc IInferenceRunner::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots */
        bool forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            int first_draft_slot,
            int slot_stride) override;
        /** @copydoc IInferenceRunner::forwardMTPBatchFromLastDraftAndSampleGreedy */
        bool forwardMTPBatchFromLastDraftAndSampleGreedy(const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens) override;
        /** @copydoc IInferenceRunner::forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots */
        bool forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            int first_condition_slot,
            int condition_slot_stride,
            int position_offset,
            int first_draft_slot,
            int draft_slot_stride) override;
        /** @copydoc IInferenceRunner::forwardMTPFromLastDraftAndSampleGreedy */
        bool forwardMTPFromLastDraftAndSampleGreedy(int32_t draft_condition_token,
            int position_id,
            int32_t *out_token) override;
        /** @copydoc IInferenceRunner::forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot */
        bool forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(int32_t draft_condition_token,
            int position_id,
            int draft_sample_slot,
            int32_t *out_token) override;
        /** @copydoc IInferenceRunner::forwardMTPFromDeviceDraftAtLivePositionAndSampleGreedyToDeviceDraftSlot */
        bool forwardMTPFromDeviceDraftAtLivePositionAndSampleGreedyToDeviceDraftSlot(int draft_condition_sample_slot,
            int position_offset,
            int draft_sample_slot,
            int32_t *out_token) override;
        /** @copydoc IInferenceRunner::flushPendingMTPWork */
        bool flushPendingMTPWork() override;
        /** @copydoc IInferenceRunner::setMTPAllPositionVerifierSyncDeferralEnabled */
        void setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled) override;
        /** @copydoc IInferenceRunner::setMTPMainDecodeSyncDeferralEnabled */
        void setMTPMainDecodeSyncDeferralEnabled(bool enabled) override;
        /** @copydoc IInferenceRunner::commitMTPInitialShiftedRowFromDeviceOutcome */
        bool commitMTPInitialShiftedRowFromDeviceOutcome(const PrefixStateSnapshot &checkpoint,
            const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            bool allow_speculative_discard = false) override;
        /** @copydoc IInferenceRunner::commitMTPShiftedRowFromDeviceTargetSample */
        bool commitMTPShiftedRowFromDeviceTargetSample(int target_sample_slot,
            int already_appended_tokens,
            bool allow_speculative_discard = false) override;
        /** @copydoc IInferenceRunner::commitMTPShiftedRowFromDeviceResidentLogicalState */
        bool commitMTPShiftedRowFromDeviceResidentLogicalState(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int already_appended_tokens,
            bool allow_speculative_discard = false) override;
        /** @copydoc IInferenceRunner::commitMTPShiftedRowsFromDeviceOutcome */
        bool commitMTPShiftedRowsFromDeviceOutcome(const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            int already_appended_tokens,
            int max_state_commit_rows,
            int main_forward_token_count,
            bool allow_speculative_discard = false) override;
        /** @copydoc IInferenceRunner::publishesFullVocabularyMTPLogits */
        bool publishesFullVocabularyMTPLogits() const override;
        /** @copydoc IInferenceRunner::sampleGreedyFromMTPLogitsToDeviceDraftSlot */
        bool sampleGreedyFromMTPLogitsToDeviceDraftSlot(int draft_sample_slot,
            int32_t *out_token) override;
        /** @copydoc IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows */
        bool sampleGreedyFromAllPositionLogitsOnDeviceRows(int start_row,
            int row_count,
            int32_t *out_tokens) override;
        /** @copydoc IInferenceRunner::supportsGreedyAllPositionBatchOutcomeOnDevice */
        bool supportsGreedyAllPositionBatchOutcomeOnDevice() const override;
        /** @copydoc IInferenceRunner::verifyGreedyAllPositionBatchOutcomeOnDevice */
        bool verifyGreedyAllPositionBatchOutcomeOnDevice(const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeVerifyBatchOutcome *out) override;
        /** @copydoc IInferenceRunner::verifyGreedyAllPositionBatchOutcomeOnDeviceResident */
        bool verifyGreedyAllPositionBatchOutcomeOnDeviceResident(const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;
        /** @copydoc IInferenceRunner::prepareGreedyAllPositionBatchOutcomeGraph */
        bool prepareGreedyAllPositionBatchOutcomeGraph(int verifier_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            const MTPRequestPenaltyPolicy &penalty_policy =
                MTPRequestPenaltyPolicy{}) override;
        /** @copydoc IInferenceRunner::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident */
        bool verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(const DeviceGreedyBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;
        /** @copydoc IInferenceRunner::deviceResidentLogicalSequenceState */
        DeviceResidentLogicalSequenceStateHandle deviceResidentLogicalSequenceState() const override;
        /** @copydoc IInferenceRunner::observeDeviceResidentNextConditionTokens */
        bool observeDeviceResidentNextConditionTokens(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_count,
            int32_t *out_tokens) override;
        /** @copydoc IInferenceRunner::sampleGreedyFromMainLogitsToDeviceTargetSlot */
        bool sampleGreedyFromMainLogitsToDeviceTargetSlot(int target_sample_slot,
            int32_t *out_token) override;
        /** @copydoc IInferenceRunner::sampleOnDeviceAtLogicalPosition */
        int sampleOnDeviceAtLogicalPosition(const SamplingParams &params,
            int logical_position) override;
        /** @copydoc IInferenceRunner::publishMainLogitsBatchSamplesToDeviceResidentState */
        bool publishMainLogitsBatchSamplesToDeviceResidentState(int request_count,
            const SamplingParams &params,
            const uint64_t *stochastic_position_seeds = nullptr) override;
        /** @copydoc IInferenceRunner::moeOverlayDeviceControllerRuntimeBindings */
        std::vector<MoEOverlayDeviceControllerRuntimeBinding> moeOverlayDeviceControllerRuntimeBindings() const override;
        /** @copydoc IInferenceRunner::applyPenaltiesOnDevice */
        bool applyPenaltiesOnDevice(const LogitPenaltyBatch &penalties,
                                            int vocab_size) override;
        /** @copydoc IInferenceRunner::applyPenaltiesToMTPLogitsOnDevice */
        bool applyPenaltiesToMTPLogitsOnDevice(const LogitPenaltyBatch &penalties,
                                                       int vocab_size) override;
        /** @copydoc IInferenceRunner::applyPenaltiesToAllPositionLogitsOnDeviceRow */
        bool applyPenaltiesToAllPositionLogitsOnDeviceRow(int row,
            const LogitPenaltyBatch &penalties,
            int vocab_size) override;
        /** @copydoc IInferenceRunner::applyDeviceOwnedMTPPenaltiesToLogitRows */
        bool applyDeviceOwnedMTPPenaltiesToLogitRows(DeviceLogitsSource source,
            int row_count,
            const MTPRequestPenaltyPolicy &penalty_policy) override;
        /** @copydoc IInferenceRunner::applyDeviceOwnedMTPBranchPenaltiesToLogits */
        bool applyDeviceOwnedMTPBranchPenaltiesToLogits(int prior_draft_count,
            const MTPRequestPenaltyPolicy &penalty_policy) override;
        /** @copydoc IInferenceRunner::supportsRowLocalAllPositionPenaltyApplication */
        bool supportsRowLocalAllPositionPenaltyApplication() const override;
        /** @copydoc IInferenceRunner::supportsDeviceStochasticMTPVerification */
        bool supportsDeviceStochasticMTPVerification() const override;
        /** @copydoc IInferenceRunner::buildStochasticDistributionOnDevice */
        bool buildStochasticDistributionOnDevice(DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size) override;
        /** @copydoc IInferenceRunner::buildStochasticDistributionsOnDevice */
        bool buildStochasticDistributionsOnDevice(DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override;
        /** @copydoc IInferenceRunner::buildCapturedStochasticVerifierTargetDistributions */
        bool buildCapturedStochasticVerifierTargetDistributions(int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size) override;
        /** @copydoc IInferenceRunner::buildStochasticProbabilityRowsOnDevice */
        bool buildStochasticProbabilityRowsOnDevice(DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override;
        /** @copydoc IInferenceRunner::buildStochasticProcessedLogitRowsOnDevice */
        bool buildStochasticProcessedLogitRowsOnDevice(DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override;
        /** @copydoc IInferenceRunner::sampleStochasticDraftProposalOnDevice */
        int sampleStochasticDraftProposalOnDevice(DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold) override;
        /** @copydoc IInferenceRunner::publishCapturedMTPDraftToken */
        bool publishCapturedMTPDraftToken(int row,
            int slot,
            const MTPRequestPenaltyPolicy &penalty_policy) override;
        /** @copydoc IInferenceRunner::sampleStochasticDraftProposalOnDeviceDeferred */
        bool sampleStochasticDraftProposalOnDeviceDeferred(DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold) override;
        /** @copydoc IInferenceRunner::deviceStochasticTargetSampleProducerSlot */
        DeviceStochasticTargetSampleSlotHandle deviceStochasticTargetSampleProducerSlot(int slot) override;
        /** @copydoc IInferenceRunner::deviceStochasticTargetSampleBroadcastDestinationSlot */
        DeviceStochasticTargetSampleSlotHandle deviceStochasticTargetSampleBroadcastDestinationSlot(int slot) override;
        /** @copydoc IInferenceRunner::recordStochasticTargetSampleSlotReadyFromDevice */
        bool recordStochasticTargetSampleSlotReadyFromDevice(int slot,
            void *producer_stream,
            bool verifier_consumer_pending = true) override;
        /** @copydoc IInferenceRunner::stageStochasticDraftTokensForDeviceVerification */
        bool stageStochasticDraftTokensForDeviceVerification(const int32_t *draft_tokens,
            int draft_token_count,
            int first_draft_slot = 0) override;
        /** @copydoc IInferenceRunner::stageStochasticTargetTokenForDeviceSampling */
        bool stageStochasticTargetTokenForDeviceSampling(int32_t target_token,
            int target_sample_slot = 0) override;
        /** @copydoc IInferenceRunner::publishForcedDeviceResidentConditionToken */
        bool publishForcedDeviceResidentConditionToken(int32_t token,
            int target_sample_slot = 0) override;
        /** @copydoc IInferenceRunner::publishDeviceResidentConditionTokenToTargetSampleSlot */
        bool publishDeviceResidentConditionTokenToTargetSampleSlot(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int target_sample_slot = 0) override;
        /** @copydoc IInferenceRunner::sampleStochasticDistributionOnDevice */
        int sampleStochasticDistributionOnDevice(DeviceDistributionBuffer buffer,
            int slot,
            float threshold) override;
        /** @copydoc IInferenceRunner::sampleStochasticDistributionOnDeviceDeferred */
        bool sampleStochasticDistributionOnDeviceDeferred(DeviceDistributionBuffer buffer,
            int slot,
            float threshold) override;
        /** @copydoc IInferenceRunner::prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken */
        const void * prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(int first_target_sample_slot,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override;
        /** @copydoc IInferenceRunner::verifyStochasticDistributionsOnDevice */
        bool verifyStochasticDistributionsOnDevice(int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            DeviceSpeculativeVerifyResult *out) override;
        /** @copydoc IInferenceRunner::verifyStochasticProbabilityRowOnDevice */
        bool verifyStochasticProbabilityRowOnDevice(int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            uint64_t inverse_sample_seed,
            int inverse_sample_logical_position,
            DeviceSpeculativeVerifyResult *out) override;
        /** @copydoc IInferenceRunner::verifyStochasticDistributionsBatchOnDevice */
        bool verifyStochasticDistributionsBatchOnDevice(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            DeviceSpeculativeVerifyResult *out) override;
        /** @copydoc IInferenceRunner::verifyStochasticDistributionsBatchOutcomeOnDevice */
        bool verifyStochasticDistributionsBatchOutcomeOnDevice(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int32_t first_token,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false) override;
        /** @copydoc IInferenceRunner::verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken */
        bool verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int first_target_sample_slot,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false) override;
        /** @copydoc IInferenceRunner::verifyStochasticDistributionsBatchOutcomeOnDeviceResident */
        bool verifyStochasticDistributionsBatchOutcomeOnDeviceResident(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int32_t first_token,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeOutcomeHandle *out_handle,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false) override;
        /** @copydoc IInferenceRunner::verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident */
        bool verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int first_target_sample_slot,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeOutcomeHandle *out_handle,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false) override;
        /** @copydoc IInferenceRunner::verifyStochasticDistributionsRequestBatchOutcomesOnDevice */
        bool verifyStochasticDistributionsRequestBatchOutcomesOnDevice(const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeVerifyBatchOutcome *outcomes) override;
        /** @copydoc IInferenceRunner::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident */
        bool verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;
        /** @copydoc IInferenceRunner::observeDeviceGenerationDispatchTicket */
        bool observeDeviceGenerationDispatchTicket(sampling_math::DeviceGenerationDispatchTicket *out_ticket) override;
        /** @copydoc IInferenceRunner::beginHostScheduledDeviceGenerationAdvance */
        bool beginHostScheduledDeviceGenerationAdvance(const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t *out_fragment_count) override;
        /** @copydoc IInferenceRunner::submitHostScheduledDeviceGenerationFragment */
        bool submitHostScheduledDeviceGenerationFragment(const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t fragment_index) override;
        /** @copydoc IInferenceRunner::finishHostScheduledDeviceGenerationAdvance */
        bool finishHostScheduledDeviceGenerationAdvance(const sampling_math::DeviceGenerationDispatchTicket &ticket) override;
        /** @copydoc IInferenceRunner::submitHostScheduledDeviceGenerationAdvance */
        bool submitHostScheduledDeviceGenerationAdvance(const sampling_math::DeviceGenerationDispatchTicket &ticket) override;
        /** @copydoc IInferenceRunner::copyDeviceSpeculativeOutcomesToHostForDiagnostics */
        bool copyDeviceSpeculativeOutcomesToHostForDiagnostics(const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes) override;

        /** @copydoc IInferenceRunner::beginDeviceResidentGeneration */
        bool beginDeviceResidentGeneration(const DeviceGenerationAdmissionRequest &request)  override;
        /** @copydoc IInferenceRunner::deviceGenerationExecutionPolicy */
        DeviceGenerationExecutionPolicy deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology topology) const noexcept override;
        /** @copydoc IInferenceRunner::materializeDeviceResidentGeneration */
        bool materializeDeviceResidentGeneration(int request_count, int draft_depth, DeviceGenerationLoopTopology topology, DeviceGenerationSamplingMode sampling_mode)  override;
        /** @copydoc IInferenceRunner::prepareDeviceResidentGenerationContinuation */
        bool prepareDeviceResidentGenerationContinuation(DeviceGenerationSamplingMode sampling_mode)  override;
        /** @copydoc IInferenceRunner::launchDeviceResidentGeneration */
        bool launchDeviceResidentGeneration()  override;
        /** @copydoc IInferenceRunner::finishDeviceResidentGeneration */
        bool finishDeviceResidentGeneration(DeviceGenerationTerminalResult *result)  override;
        /** @copydoc IInferenceRunner::forwardGroupedMTPVerifierWithDeviceTokenIds */
        bool forwardGroupedMTPVerifierWithDeviceTokenIds(const int *token_shadow, const void *token_ids_device, int seq_len)  override;
        /** @copydoc IInferenceRunner::advanceMTPMainConditionFromDeviceResidentLogicalState */
        bool advanceMTPMainConditionFromDeviceResidentLogicalState(int32_t token_shadow, const DeviceResidentLogicalSequenceStateHandle &logical_state, MTPConditionForwardPurpose purpose, int request_index)  override;
        /** @copydoc IInferenceRunner::advanceMTPMainConditionFromDeviceTargetSample */
        bool advanceMTPMainConditionFromDeviceTargetSample(int32_t token_shadow, int target_sample_slot, MTPConditionForwardPurpose purpose)  override;
        /** @copydoc IInferenceRunner::publishAcceptedMTPSpecStateBatchFromDeviceOutcome */
        bool publishAcceptedMTPSpecStateBatchFromDeviceOutcome(const DeviceSpeculativePublicationRequest &request, std::string *error)  override;
        /** @copydoc IInferenceRunner::advanceOrdinaryMainConditionFromDeviceResidentLogicalState */
        bool advanceOrdinaryMainConditionFromDeviceResidentLogicalState(int32_t token_shadow, const DeviceResidentLogicalSequenceStateHandle &logical_state, int request_index)  override;
        /** @copydoc IInferenceRunner::mtpVerifierRowCapability */
        MTPVerifierRowCapability mtpVerifierRowCapability() const override;

        /** @brief Immutable completed original source observation, never scheduler-ticket cadence. */
        struct OriginalRankGenerationTerminal
        {
            uint32_t abi = 1; ///< Fixed terminal metadata contract.
            uint64_t request_generation = 0; ///< Exact original request/reset authority.
            uint64_t command_id = 0; ///< Exact original command, rejecting stale observations.
            int admission_base = -1; ///< Completed request's immutable initial KV provenance.
            int admission_budget = 0; ///< Original response admission, not a live device counter.
            int published_state_commit_count = -1; ///< Actual terminal controller publication count.
            int cached_tokens = -1; ///< Actual completed tail KV observation.
            sampling_math::DeviceGenerationLeadingRowDisposition next_leading_row_disposition =
                sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse;
            bool model_stopped = false; ///< Actual terminal controller outcome.
        };
        /** @return Whether this real rank must service the original tail's generation command. */
        bool originalRankGenerationFollower() const noexcept;
        /** @brief Bind original process-local maintenance notifications before command admission.
         * @param sink Existing OR coalescing notification authority, never a second placement policy.
         * @throws std::logic_error For duplicate binding or an active original command.
         */
        void bindOriginalRankRetiredDecodeProgressSink(MoEOverlayRetiredDecodeProgressSink sink);

        /** @return Whether this real rank owns the admitted original source generation service. */
        bool originalRankGenerationSource() const noexcept { return rank_service_ready_ && is_pipeline_tail_; }

        /** @return Existing original source request/reset epoch, never a service-owned counter. */
        uint64_t originalRankRequestGeneration() const;
        /** @brief Project request and command ordering from the original terminal source authority.
         * @param command_id Source OR's existing monotonic inference command identifier.
         * @return Immutable command identity without observing live device state.
         * @throws std::logic_error For absent source ownership or incomplete placement authority.
         */
        MoEOverlayInferenceCommandIdentity originalRankCommandIdentity(uint64_t command_id) const;
        /** @brief Begin one source-owned command and both directed rank-channel admissions.
         * @param command Existing OR inference-command identity and source placement epoch.
         * @param admission_base Original completed prefill/previous-terminal planning boundary.
         * @return True after immutable command publication, without mutable device observations.
         */
        bool beginOriginalRankGenerationCommand(const MoEOverlayInferenceCommandIdentity &command, int admission_base, DeviceGenerationSamplingMode sampling);
        /** @brief Service immutable admission/tickets, then validate and return actual source terminal accounting.
         * @param terminal Receives completed source publication/KV/leading disposition only.
         * @param error Receives the first protocol/native/independent-KV failure.
         * @return True after exact local native retirement and directed-channel retirement.
         */
        bool serviceOriginalRankGenerationCommand(OriginalRankGenerationTerminal *terminal, std::string *error);
        /** @brief Complete the original source command only from actual terminal accounting.
         * @param model_stopped Source result for a command which needed no resident controller.
         * @return True after all graph slots and native endpoint owners retire.
         */
        bool completeOriginalRankGenerationCommand(bool model_stopped,
            sampling_math::DeviceGenerationLeadingRowDisposition terminal_leading);
        /** @brief Cancel both directed channels before publishing fatal protocol termination. */
        void abortOriginalRankGenerationCommand() noexcept;

        /** @brief Check original local stages against their retained prefill family.
         * @param seq_len Positive logical prompt/suffix length, excluding padding.
         * @return Whether every real local runner accepts that retained schedule.
         */
        bool supportsPrefillChunkSchedule(int seq_len) const override;
        /** @brief Submit only real local participants through their original captured rank edges.
         * @param tokens Immutable prompt/suffix input, shared as request input only.
         * @param seq_len Logical live rows, never transport capacity.
         * @param policy Original root-owned chunk geometry and prefix offset.
         * @param pad_token_id Original admitted graph padding token.
         * @param allow_padded_execution Explicit physical-graph padding admission.
         * @return Whether all local stage owners submitted the complete schedule.
         */
        bool forwardPrefillChunkSchedule(const int *tokens, int seq_len,
            const PrefillChunkSchedulerPolicy &policy, int pad_token_id, bool allow_padded_execution) override;
    private:
        Config config_;
        GlobalPPRankPlan rank_plan_;
        /** @brief Actual directed channel owners, never foreign process contexts/cursors. */
        struct RankBoundaryChannels
        {
            std::shared_ptr<CapturedTransferChannel> activation;
            std::shared_ptr<CapturedTransferChannel> metadata;
        };
        std::shared_ptr<CapturedTransferRankSession> rank_transfer_session_; ///< OR retains the outer retirement owner.
        std::vector<RankBoundaryChannels> rank_boundaries_;
        std::vector<std::unique_ptr<PipelineForwardGraphEdges>> rank_forward_edges_; ///< DGOs retire before these installed pointers.
        /** Original successful hosted family, retained through all request resets. */
        std::optional<ServingGraphFamilyMaterializationPlan> hosted_serving_family_plan_;
        std::optional<ServingGraphFamilyMaterializationPlan> rank_port_family_;

        // The cohort explicitly retires borrowed parents in the destructor body,
        // then survives the original DGO registry so its installed edge owners
        // remain live while those graph/cache owners destroy their recordings.
        /** @brief Freeze source/follower lanes from the exact original topology/session. */
        void prepareOriginalRankGenerationService();
        /** @brief Publish one complete exact-row body to every follower before local enqueue. */
        bool publishOriginalRankTransaction(MoEOverlayInferenceGraphRole role, int rows, int depth);
        /** @brief Join all original follower receipts after authenticated source completion. */
        bool retireOriginalRankTransaction(uint64_t scheduler_frontier);
        /** @brief Admit both existing channel directions under the original command identity. */
        void beginOriginalRankChannelInvocations();
        /** @brief Retire both existing channel directions after original native endpoint joins. */
        bool finishOriginalRankChannelInvocations();
        /** @brief Broadcast one immutable admission before any original follower graph submission. */
        bool publishOriginalRankAdmission(const DeviceGenerationAdmissionRequest &request);
        /** @return Exact local original tail; execution on a follower is a fatal ownership error. */
        IInferenceRunner &generationTail();
        /** @return Read-only original tail authority for capability/state observations. */
        const IInferenceRunner &generationTail() const;
        std::array<uint8_t, 12> rank_tail_capabilities_{}; ///< Source-authenticated static graph capabilities only.
        MTPVerifierRowCapability rank_tail_verifier_capability_; ///< Actual source capability, frozen before request admission.
        DeviceGenerationSamplingMode rank_generation_sampling_mode_ = DeviceGenerationSamplingMode::Greedy;
        // This owner constructs native MPI publishers itself. Retain that exact
        // type so terminal publication reads each protocol's authenticated epoch.
        std::vector<std::shared_ptr<MoEOverlayInferenceTransactionPublisher>> rank_publishers_;
        std::vector<MoEOverlayPublishedInferenceTransaction> rank_pending_transactions_;
        std::vector<std::vector<MoEOverlayPublishedInferenceTransaction>> rank_deferred_transactions_; ///< Bounded real native-completion leases per original follower lane.
        std::size_t rank_transaction_slot_capacity_ = 0; ///< Cold protocol capacity, never a model-state mirror.
        /** @brief Query ready receipts, or join only at bounded ring reuse / outer command terminal. */
        bool progressOriginalRankTransactions(bool terminal);

        std::unique_ptr<MoEOverlayInferenceTransactionFollower> rank_transaction_follower_;
        std::optional<MoEOverlayInferenceCommandIdentity> rank_active_command_;
        std::optional<OriginalRankGenerationTerminal> rank_terminal_observation_;
        int rank_response_admission_base_ = -1; ///< Immutable response-ledger origin after any preceding scalar conditions.
        int rank_admission_base_ = -1;
        int rank_admission_budget_ = 0;
        int rank_graph_capacity_rows_ = 0;
        int rank_draft_capacity_ = 0;
        bool rank_service_ready_ = false;
        bool rank_admission_published_ = false;
        bool rank_channels_admitted_ = false;
        bool rank_transaction_pending_ = false;
        uint64_t rank_retired_decode_scheduler_frontier_ = 0; ///< Authenticated immutable-ticket cadence only; never terminal state accounting.
        MoEOverlayRetiredDecodeProgressSink rank_retired_decode_progress_sink_; ///< Original OR maintenance notification, not a policy/controller owner.

        std::unique_ptr<PipelineDeviceGeneration> rank_generation_;

        // Per-rank local stage execution (owned)
        StageRunnerRegistry stage_runners_;

        // Activation transfer buffer (for PP send/recv)
        std::shared_ptr<TensorBase> activation_buffer_;

        // Cached queries
        int tail_rank_ = 0;
        bool is_pipeline_head_ = false;
        bool is_pipeline_tail_ = false;

        int last_seq_len_ = 0;   ///< Sequence length from last forward() call (for buffer sizing)
    };

} // namespace llaminar2
