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
 */

#pragma once

#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "../local_execution/orchestrators/IRankOrchestrator.h"
#include "../global_pp/GlobalPPTopology.h"
#include "../global_pp/GlobalPPRankPlan.h"
#include "../../interfaces/IMPIContext.h"
#include "../../collective/ITPContext.h"
#include "../../collective/ILocalTPContext.h"
#include "../../collective/IGlobalTPContext.h"
#include "../factory/FactoryPPStageConfig.h"
#include "../../utils/Sampler.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    class TensorBase;
    class PreparedWeightStore;

    /**
     * @brief Stage-local weight ownership record for one pipeline domain.
     *
     * The prepared store is intentionally stage scoped so one runner's prepared
     * handles are not replaced when another runner installs its own store on the
     * shared WeightManager during construction.
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
     * The context shared_ptrs (local_tp_ctx, global_tp_ctx) ensure the TP context
     * outlives the runner that uses it. pp_stage_config records the layer range
     * that this runner executes, enabling callers to verify correct PP slicing.
     */
    struct StageRunnerEntry
    {
        int stage_id = -1;
        std::string domain_name;
        RankStageAction action;

        /// Lifetime owner for local TP context (ILocalTPContext) if this stage
        /// uses local multi-device TP via RankOrchestrator.
        std::shared_ptr<ILocalTPContext> local_tp_ctx;

        /// Lifetime owner for global TP context (IGlobalTPContext) if this stage
        /// uses cross-rank TP via GlobalTPContext/DomainCommunicatorRegistry.
        std::shared_ptr<IGlobalTPContext> global_tp_ctx;

        /// PP layer range for this stage runner. Records first_layer/last_layer
        /// (exclusive last_layer) as passed to the factory. Useful for verification.
        std::optional<FactoryPPStageConfig> pp_stage_config;

        /// Stage-local weight context and prepared handle store lifetime owner.
        std::shared_ptr<StageWeightContext> weight_context;

        /// Runner is declared after its context owners so destruction tears the
        /// runner down before any raw TP/prepared-store pointers can expire.
        std::unique_ptr<IInferenceRunner> runner;
    };

    /**
     * @brief Owns and dispatches the local stage runners for a global PP rank.
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

        void add(StageRunnerEntry entry);
        void setCompatibilityRunner(std::unique_ptr<IInferenceRunner> runner);

        bool empty() const;
        size_t size() const;
        bool hasRunnerForStage(int stage_id) const;

        StageRunnerEntry *entryForStage(int stage_id);
        const StageRunnerEntry *entryForStage(int stage_id) const;
        StageRunnerEntry *entryForDomain(const std::string &domain_name);
        const StageRunnerEntry *entryForDomain(const std::string &domain_name) const;

        IInferenceRunner *runnerForStage(int stage_id);
        const IInferenceRunner *runnerForStage(int stage_id) const;
        IInferenceRunner *runnerForDomain(const std::string &domain_name);
        const IInferenceRunner *runnerForDomain(const std::string &domain_name) const;

        IInferenceRunner *pipelineHeadRunner();
        const IInferenceRunner *pipelineHeadRunner() const;
        IInferenceRunner *pipelineTailRunner();
        const IInferenceRunner *pipelineTailRunner() const;
        IInferenceRunner *defaultRunner();
        const IInferenceRunner *defaultRunner() const;
        IInferenceRunner *lastLocalRunner();
        const IInferenceRunner *lastLocalRunner() const;

        void clearCacheAll();
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
        void setSkipLogitsGatherDecodeAll(bool skip);
        void setSkipLogitsGatherPrefillAll(bool skip);
        void setSuppressTimelineAll(bool suppress);
        void setAccumulatePrefillAll(bool accumulate);
        void flushStageTimelineAll();
        bool supportsChainedMTPDraftsAll() const;
        bool forwardMTPAll(int32_t draft_condition_token);
        bool forwardMTPFromLastDraftAll(int32_t draft_condition_token, int position_id);
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
        bool commitMTPShiftedRowFromCurrentTerminalHiddenAll(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1);
        bool commitMTPShiftedRowFromCheckpointTerminalHiddenAll(
            const PrefixStateSnapshot &checkpoint,
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1);
        bool ensureMTPCheckpointTerminalHiddenAll();
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
        uint64_t moePlacementEpochAll() const;
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
        PrefixLookupResult lookupPrefixAll(const std::vector<int32_t> &tokens);
        /**
         * @brief Restore the coordinated prefix into every local PP stage.
         * @param hit Aggregate result returned by @ref lookupPrefixAll.
         * @param seq_idx Sequence slot receiving the restored state.
         * @return True after every child imports its own clamped payload.
         */
        bool populatePrefixAll(const PrefixLookupResult &hit, int seq_idx = 0);
        /**
         * @brief Archive one completed prefix in every local PP stage cache.
         * @param tokens Complete prompt token sequence.
         * @param prompt_token_count Number of live prompt tokens.
         * @return True after every child publishes its stage-owned payload.
         */
        bool harvestPrefixAll(
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
         */
        PrefixRuntimeStateSnapshot prefixStateProbeAll() const;
        PrefixStateSnapshot captureLivePrefixStateAll(int seq_idx = 0) const;
        PrefixStateSnapshot captureLivePrefixCheckpointAll(
            const PrefixCheckpointCaptureRequest &request) const;
        bool restoreLivePrefixStateAll(const PrefixStateSnapshot &snapshot, int seq_idx = 0);
        bool truncateLivePrefixStateAll(int cached_tokens, int seq_idx = 0);
        std::string mtpDecodeUnsupportedReasonAll() const;
        void enableSnapshotCaptureAll(const std::string &output_dir);
        /** @brief Install one immutable capture filter on every local stage. */
        void setSnapshotCaptureFilterAll(
            const std::vector<std::string> &keys);
        void disableSnapshotCaptureAll();
        void clearSnapshotsAll();
        const float *getSnapshot(const std::string &key, size_t &out_size) const;
        SnapshotInfo getSnapshotWithShape(const std::string &key) const;
        std::vector<std::string> snapshotKeysAll() const;

    private:
        std::vector<StageRunnerEntry> entries_;
        std::unique_ptr<IInferenceRunner> compatibility_runner_;
        /** Child-owned lookup handles retained until this prefix transaction ends. */
        std::vector<PrefixLookupResult> last_prefix_hits_;
        std::optional<PrefixLookupResult> compatibility_prefix_hit_;
    };

    /**
     * @brief Moves activations between stage runners and MPI peers.
     */
    class StageActivationRouter
    {
    public:
        bool executeTransfer(const RankTransferAction &action,
                             StageRunnerRegistry &registry,
                             IMPIContext &mpi_ctx,
                             int rank,
                             int last_seq_len,
                             int d_model,
                             std::shared_ptr<TensorBase> &activation_buffer) const;

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

            // Per-stage local runners (ownership transferred). New multi-domain path.
            std::vector<StageRunnerEntry> stage_runners;

            // Per-rank local runner (ownership transferred). Legacy compatibility path.
            std::unique_ptr<IInferenceRunner> rank_runner;

            // Model metadata
            int vocab_size = 0;              ///< Full vocabulary size
            int d_model = 0;                 ///< Hidden state dimension
            std::string architecture_name = "unknown";
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
         * @param config Configuration (rank_runner ownership transferred)
         * @throws std::invalid_argument if config is invalid
         */
        explicit GlobalOrchestrator(Config config);

        ~GlobalOrchestrator() override = default;

        // Non-copyable, non-movable (owns unique_ptrs)
        GlobalOrchestrator(const GlobalOrchestrator &) = delete;
        GlobalOrchestrator &operator=(const GlobalOrchestrator &) = delete;
        GlobalOrchestrator(GlobalOrchestrator &&) = delete;
        GlobalOrchestrator &operator=(GlobalOrchestrator &&) = delete;

        // =================================================================
        // IInferenceRunner — Core Inference API
        // =================================================================

        bool forward(const int *tokens, int seq_len) override;
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
        /** @brief Populate every local stage from its retained lookup handles. */
        bool populatePrefix(
            const PrefixLookupResult &hit,
            int seq_idx = 0) override;
        /** @brief Harvest one prefix payload from every local stage. */
        bool harvestPrefix(
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
        /** @brief Observe the complete rank-local pipeline prefix state. */
        PrefixRuntimeStateSnapshot prefixStateProbe() const override;

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
         * @brief Execute a single EXECUTE_STAGE step
         *
         * Delegates to the rank runner's forward(), passing tokens only
         * for the pipeline head stage; other stages use hidden state input.
         */
        bool executeStage(const RankStageAction &action,
                          const int *tokens, int seq_len);

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
        bool executeTransfer(const RankTransferAction &action);

        /**
         * @brief Find the tail rank (the rank that has the LM head)
         */
        int findTailRank() const;

        // =================================================================
        // State
        // =================================================================

        Config config_;
        GlobalPPRankPlan rank_plan_;

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
