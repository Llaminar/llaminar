/**
 * @file RankOrchestrator.h
 * @brief Multi-device orchestrator for LOCAL tensor parallelism
 *
 * Coordinates multiple DeviceGraphOrchestrator instances for LOCAL tensor
 * parallelism across multiple devices within a single MPI rank.
 *
 * Key concepts:
 * - LOCAL TP: Multiple devices owned by one MPI rank (decoupled from MPI world_size)
 * - Proportional TP: Devices can have different capacities (weights)
 * - Backend selection: NCCL, RCCL, or HOST based on device types
 *
 * Design philosophy:
 * - Extends IRankOrchestrator (which extends IInferenceRunner)
 * - Drop-in replacement for single-device DeviceGraphOrchestrator
 * - Coordinates collective operations (AllReduce, AllGather) across local devices
 *
 * Execution flow:
 * 1. Distribute tokens to device runners based on sharding strategy
 * 2. Each runner executes forward pass independently
 * 3. AllGather partial logits to combine results
 * 4. Return combined logits from primary device
 *
 * Usage:
 * @code
 * RankOrchestrator::Config config;
 * config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
 * config.weights = {0.73f, 0.27f};  // Optional proportional weights
 * config.backend = CollectiveBackendType::NCCL;
 *
 * auto orchestrator = std::make_unique<RankOrchestrator>(model_ctx, config);
 *
 * // Use as IInferenceRunner (same API as single-device)
 * orchestrator->forward(tokens, seq_len);
 * const float* logits = orchestrator->logits();
 *
 * // Or access multi-device specifics
 * int num_devices = orchestrator->device_count();
 * auto* tp_ctx = orchestrator->localTPContext();
 * @endcode
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "IRankOrchestrator.h"
#include "TPWorkerPool.h"
#include "../../../backends/GlobalDeviceAddress.h"
#include "../../../config/OrchestrationConfig.h"
#include "../../../collective/ILocalPPContext.h"
#include "../../moe/MoERebalanceController.h"
#include "../../moe/ExpertWeightTransfer.h"
#include "../../config/RuntimeConfig.h"
#include "../../debug/TPSnapshot.h"
#include "../../factory/FactoryPPStageConfig.h" // For FactoryPPStageConfig (circular-dependency-safe)
#include "../../mtp/MTPSpecStateContract.h"
#include "../../../kernels/common/SamplingMath.h"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <string>

// Forward declaration for fromPlan() factory method
namespace llaminar2
{
    struct RankExecutionPlan;
}

namespace llaminar2
{

    // Forward declarations
    class IModelContext;
    class ILocalTPContext;
    class IBackend;
    class TensorBase;
    class LogitsGatherer;
    class DeviceSampler;
    class IMPIContext;
    class PreparedWeightStore;
    struct GraphExecutorStats;
    struct MoEExpertParallelPlan;
    struct PlacementPlan;
    struct PPActivationContract;

    namespace rank_orchestrator_detail
    {
        bool sameBackendGpuExpertTransferIsDirectOnly(DeviceId destination, DeviceId source);

        std::vector<std::vector<std::vector<bool>>> buildReplicaArrivalTransferMasks(
            const MoERebalanceController &controller,
            const ExpertReplicaSet &arrivals);

        std::vector<std::vector<std::vector<bool>>> buildOwnershipArrivalTransferMasks(
            const MoERebalanceController &controller,
            const std::vector<int> &previous_placement);
    }

    /**
     * @brief Multi-device orchestrator for LOCAL tensor parallelism
     *
     * Coordinates multiple DeviceGraphOrchestrator instances to enable tensor
     * parallelism across devices within a single MPI rank. This is the primary
     * implementation of IRankOrchestrator.
     *
     * Key features:
     * - Manages per-device inference runners (DeviceGraphOrchestrator instances)
     * - Coordinates collective operations via ILocalTPContext
     * - Supports proportional TP for heterogeneous device configurations
     * - Combines partial logits from column-parallel LM head
     *
     * Thread safety: All public methods are thread-safe. Internal synchronization
     * ensures correct ordering of collective operations.
     */
    class RankOrchestrator : public IRankOrchestrator
    {
    public:
        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Parallelism mode for multi-device orchestration
         */
        enum class ParallelismMode
        {
            AUTO, ///< Auto-detect from configuration
            TP,   ///< Tensor Parallelism only (parallel execution, same layers)
            PP,   ///< Pipeline Parallelism only (sequential execution, different layers)
            TP_PP ///< Combined: PP stages where each stage is a TP domain
        };

        /**
         * @brief Configuration for a single PP stage
         *
         * Specifies layer range and optional TP configuration for this stage.
         */
        struct PPStageConfig
        {
            /// First layer index (inclusive)
            int first_layer = 0;

            /// Last layer index (exclusive)
            int last_layer = 0;

            /// Whether this stage has the embedding layer
            bool has_embedding = false;

            /// Whether this stage has the LM head
            bool has_lm_head = false;

            /// Devices for this stage (single device for pure PP, multiple for TP+PP)
            std::vector<GlobalDeviceAddress> stage_devices;

            /// TP weights for this stage (empty = equal distribution)
            /// Only used when stage_devices.size() > 1
            std::vector<float> tp_weights;

            /// TP backend for this stage (only used when stage_devices.size() > 1)
            CollectiveBackendType tp_backend = CollectiveBackendType::AUTO;

            /// Get the number of layers in this stage
            int numLayers() const { return last_layer - first_layer; }

            /// Check if this stage is a TP domain (multiple devices)
            bool isTPDomain() const { return stage_devices.size() > 1; }

            /// Validate stage configuration
            bool validate() const;
        };

        /**
         * @brief Configuration for multi-device orchestration
         *
         * Specifies devices, weights, backend, and inference parameters.
         * Supports both TP (parallel) and PP (sequential) execution modes.
         */
        struct Config
        {
            // =================================================================
            // Parallelism Mode
            // =================================================================

            /// Parallelism mode (AUTO detects from configuration)
            ParallelismMode mode = ParallelismMode::AUTO;

            // =================================================================
            // TP Configuration (used when mode is TP or AUTO with no PP stages)
            // =================================================================

            /// Devices participating in LOCAL TP
            std::vector<GlobalDeviceAddress> devices;

            /// Proportional weights for work distribution (sum to 1.0)
            /// Empty or unset: equal distribution
            /// Example: {0.73f, 0.27f} for 73%/27% split
            std::vector<float> weights;

            /// Backend for collective operations (AUTO for auto-detection)
            CollectiveBackendType backend = CollectiveBackendType::AUTO;

            // =================================================================
            // PP Configuration (used when mode is PP or TP_PP)
            // =================================================================

            /// PP stage configurations (layer ranges per stage)
            /// If non-empty, enables PP mode
            std::vector<PPStageConfig> pp_stages;

            // =================================================================
            // Common Configuration
            // =================================================================

            /// Maximum sequence length
            size_t max_seq_len = 4096;

            /// Batch size for inference
            int batch_size = 1;

            /// Activation precision for intermediate buffers
            ActivationPrecision activation_precision = ActivationPrecision::FP32;

            /// KV cache scale factors (K and V separate)
            float kv_cache_scale_k = 256.0f;
            float kv_cache_scale_v = 32.0f;

            /// Explicit KV cache precision mode (AUTO preserves legacy behavior)
            KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;

            /// Optional explicit transport precision for TP allreduces.
            std::string tp_allreduce_precision_override;

            /// Prefix-state cache feature gates and storage limits.
            PrefixCacheRuntimeConfig prefix_cache;

            /// Multi-token prediction feature gates and verification mode.
            MTPRuntimeConfig mtp;

            /// Routed MoE expert execution mode for standard Qwen3.5 MoE.
            MoEExpertMode moe_expert_mode = MoEExpertMode::ApportionedExperts;

            /// Bounded hot remote expert cache for dynamic expert-parallel execution.
            MoEHotExpertCacheConfig moe_hot_expert_cache;

            /// Decode histogram / dynamic rebalance settings.
            MoERebalanceRuntimeConfig moe_rebalance;

            /// Use mapped memory for GPU tensors (zero-copy host access)
            /// Required for correct coherence with column-parallel LM head
            bool use_mapped_memory = false;

            // =================================================================
            // Nested TP-in-PP Configuration
            // =================================================================

            /// For TP domains that are part of a PP stage, this holds the PP stage config.
            /// When set, the TP device runners will build partial graphs instead of full graphs.
            /// Set by the parent MDO when creating a nested TP MDO for a PP stage.
            std::optional<FactoryPPStageConfig> nested_pp_stage_config;

            /// Optional stage-local prepared store shared by this RankOrchestrator
            /// and its per-device runners.
            std::shared_ptr<PreparedWeightStore> prepared_weight_store;

            /// Optional same-layer MoE expert overlay plan propagated to child graph runners.
            std::shared_ptr<MoEExpertParallelPlan> moe_expert_parallel_plan;

            /// Optional MPI context used by MoE overlay domain-worker commands.
            std::shared_ptr<IMPIContext> moe_expert_overlay_mpi_ctx;

            // =================================================================
            // Helper Methods
            // =================================================================

            /**
             * @brief Check if PP is enabled
             *
             * @return true if pp_stages is non-empty
             */
            bool hasPP() const { return !pp_stages.empty(); }

            /**
             * @brief Detect parallelism mode from configuration
             *
             * - No devices and no PP stages: Invalid
             * - Devices only: TP
             * - PP stages only: PP
             * - PP stages with TP domains: TP_PP
             *
             * @return Detected parallelism mode
             */
            ParallelismMode detectMode() const;

            /**
             * @brief Get effective mode (resolves AUTO)
             *
             * @return Effective parallelism mode
             */
            ParallelismMode effectiveMode() const
            {
                return mode == ParallelismMode::AUTO ? detectMode() : mode;
            }

            /**
             * @brief Validate configuration
             *
             * Checks:
             * - At least one device specified (for TP) or PP stages defined
             * - Weights sum to ~1.0 (if provided)
             * - Weights count matches device count (if provided)
             * - PP stages have valid layer ranges
             * - PP stages cover all layers without gaps
             *
             * @return true if configuration is valid
             */
            bool validate() const;

            /**
             * @brief Get normalized weights (defaults to equal if unset)
             *
             * If weights are empty or invalid, returns equal distribution.
             *
             * @return Vector of normalized weights summing to 1.0
             */
            std::vector<float> getNormalizedWeights() const;

            /**
             * @brief Build layer boundaries vector from PP stages
             *
             * Returns {0, stage0.last_layer, stage1.last_layer, ...}
             *
             * @return Layer boundary indices
             */
            std::vector<int> buildLayerBoundaries() const;

            /**
             * @brief Canonical factory: build Config from a RankExecutionPlan
             *
             * Handles both TP and PP modes:
             * - TP: Copies devices, weights, backend from plan.local_tp_*
             * - PP: Sets mode=PP, builds PPStageConfig entries from
             *       plan.local_pp_devices + plan.local_pp_layer_boundaries
             *       with cross-vendor detection
             *
             * Runtime fields (max_seq_len, activation_precision, etc.) come
             * from plan.runtime which was pre-parsed in ExecutionPlanBuilder.
             *
             * @param plan The rank execution plan
             * @return Populated Config
             */
            static Config fromPlan(const RankExecutionPlan &plan);
        };

        // =====================================================================
        // Factory Methods
        // =====================================================================

        /**
         * @brief Factory method for unit testing with injected dependencies
         *
         * Allows injection of pre-constructed device runners and TP context
         * for testing without real devices or model files.
         *
         * @param model_ctx Model context (metadata only for testing)
         * @param device_runners Pre-constructed per-device runners
         * @param tp_ctx Pre-constructed LOCAL TP context
         * @param config Configuration
         * @return Unique pointer to RankOrchestrator
         *
         * @code
         * // Test setup with mocks
         * auto model_ctx = std::make_shared<MockModelContext>(preset);
         * std::vector<std::unique_ptr<IInferenceRunner>> runners;
         * runners.push_back(createMockRunner(cuda0));
         * runners.push_back(createMockRunner(cuda1));
         * auto tp_ctx = std::make_unique<MockLocalTPContext>(devices, weights);
         *
         * auto orchestrator = RankOrchestrator::createForTest(
         *     model_ctx, std::move(runners), std::move(tp_ctx), config);
         * @endcode
         */
        static std::unique_ptr<RankOrchestrator> createForTest(
            std::shared_ptr<IModelContext> model_ctx,
            std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
            std::unique_ptr<ILocalTPContext> tp_ctx,
            const Config &config);

        static std::unique_ptr<RankOrchestrator> createForTestWithPipelineStages(
            std::shared_ptr<IModelContext> model_ctx,
            std::vector<std::unique_ptr<IInferenceRunner>> pp_stage_runners,
            const Config &config);

        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Construct with model context and configuration
         *
         * Creates device runners and TP context based on configuration.
         * If a pre-existing TP context is provided, uses it directly (TP mode).
         * Otherwise, auto-detects mode from config and creates TP context if needed.
         *
         * @param model_ctx Model context with weights and metadata
         * @param config Multi-device configuration
         * @param tp_ctx Optional pre-constructed LOCAL TP context (ownership transferred)
         */
        RankOrchestrator(
            std::shared_ptr<IModelContext> model_ctx,
            const Config &config,
            std::unique_ptr<ILocalTPContext> tp_ctx = nullptr);

        /// Destructor
        ~RankOrchestrator() override;

        // Non-copyable, movable
        RankOrchestrator(const RankOrchestrator &) = delete;
        RankOrchestrator &operator=(const RankOrchestrator &) = delete;
        RankOrchestrator(RankOrchestrator &&) noexcept;
        RankOrchestrator &operator=(RankOrchestrator &&) noexcept;

        // =====================================================================
        // IInferenceRunner Interface (from IRankOrchestrator)
        // =====================================================================

        /**
         * @brief Run forward pass across all devices
         *
         * Distributes work to device runners and coordinates collective operations.
         *
         * @param tokens Input token IDs
         * @param seq_len Sequence length
         * @return true if forward pass succeeded on all devices
         */
        bool forward(const int *tokens, int seq_len) override;
        bool forwardPrefill(const int *tokens, int seq_len) override;
        /**
         * @brief Run a LocalTP forward where every child consumes its own
         *        staged device-token row.
         *
         * prepareMTPVerifierInputTokensOnDevice*() returns a rank-owned bundle
         * of child token pointers.  This override fans the bundle out through
         * the TP worker pool so all collective participants enter the verifier
         * graph together.
         */
        bool forwardWithDeviceTokenIds(
            const int *token_shadow,
            const void *token_ids_device,
            int seq_len) override;
        bool forwardBatchWithDeviceTokenIds(
            const std::vector<std::vector<int>> &token_batches,
            const void *token_ids_device,
            int padded_seq_len) override;
        bool supportsPrefillChunkSchedule(int seq_len) const override;
        bool forwardPrefillChunkSchedule(
            const int *tokens,
            int seq_len,
            const PrefillChunkSchedulerPolicy &policy,
            int pad_token_id,
            bool allow_padded_execution) override;
        DeviceId primaryDeviceId() const override;

        /**
         * @brief Get combined logits from last forward pass
         *
         * Returns logits gathered from all devices via AllGather.
         *
         * @return Pointer to combined logits [vocab_size], or nullptr if unavailable
         */
        const float *logits() const override;
        bool forwardMTP(int32_t draft_condition_token) override;
        bool forwardMTPForDeviceSampling(int32_t draft_condition_token) override;
        /**
         * @brief True when every LocalTP participant can consume a previous
         *        MTP sidecar hidden row as the next draft input.
         *
         * Depth-2/3 MTP is only valid for a rank when all child runners can
         * keep their shifted MTP KV and sidecar hidden state in lockstep.
         */
        bool supportsChainedMTPDrafts() const override;

        /**
         * @brief Run one chained MTP sidecar step on every LocalTP participant.
         *
         * The token and logical shifted-cache position are rank-wide scalar
         * decisions.  Every child receives the same values and must complete
         * before the rank reports success, preserving the vLLM-style
         * participant-symmetric graph sequence.
         */
        bool forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id) override;
        bool forwardMTPFromLastDraftForDeviceSampling(
            int32_t draft_condition_token,
            int position_id) override;
        bool forwardMTPFromDeviceDraftForDeviceSampling(
            int draft_sample_slot,
            int position_id) override;
        bool forwardMTPFromDeviceTargetForDeviceSampling(
            int target_sample_slot,
            int position_id) override;
        bool forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index = 0) override;
        /**
         * @brief Return a rank-owned aggregate of child resident logical-state
         *        mailboxes after grouped compact publication.
         *
         * The returned handle is an identity token for RankOrchestrator; callers
         * must pass it back to rank-level resident-state methods rather than
         * dereferencing its device-row pointers directly.  Rank then dispatches
         * the matching child-owned mailbox handle to every LocalTP participant.
         */
        DeviceResidentLogicalSequenceStateHandle
        deviceResidentLogicalSequenceState() const override;
        bool commitMTPShiftedRowsFromLastForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens) override;
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
        bool commitMTPInitialShiftedRowFromDeviceOutcome(
            const PrefixStateSnapshot &checkpoint,
            const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            int main_forward_token_count,
            bool allow_speculative_discard = false) override;
        bool commitMTPShiftedRowFromDeviceTargetSample(
            int target_sample_slot,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override;
        bool commitMTPShiftedRowFromDeviceResidentLogicalState(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int already_appended_tokens,
            bool allow_speculative_discard = false) override;
        bool commitMTPShiftedRowsFromDeviceOutcome(
            const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            int already_appended_tokens,
            int max_state_commit_rows,
            int main_forward_token_count,
            bool allow_speculative_discard = false) override;
        bool flushPendingMTPWork() override;
        bool ensureMTPCheckpointTerminalHidden() override;
        const float *mtpLogits() const override;
        bool setComputeAllPositionLogits(bool enabled) override;
        bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override;
        bool setMTPSpecVerifierInputPlan(
            const MTPSpecDecodeVerifierInputPlan &plan) override;
        void clearMTPSpecVerifierInputPlan() override;
        bool supportsLogicalMTPVerifierBaseCheckpoint() const override;
        /**
         * @brief True when every active LocalTP participant can publish accepted
         *        verifier state from the current MTP target verifier graph.
         *
         * Rank-level publication is an all-participant contract.  The rank must
         * not advertise this capability unless each child runner can restore its
         * own KV/recurrent/terminal-hidden slice from the same speculative step.
         */
        bool supportsMTPSpecStatePublication() const override;
        MTPVerifierRowCapability mtpVerifierRowCapability() const override;

        /**
         * @brief Publish accepted MTP verifier state on every LocalTP child.
         *
         * The same logical step plan is coordinated through the shared
         * common-prefix contract before fan-out.  Today LocalTP receives one
         * accepted-count decision from the rank-level verifier; this hook keeps
         * the publication path symmetric so future per-participant plans can be
         * clamped at the same boundary instead of growing a special case.
         */
        bool publishAcceptedMTPSpecState(
            const MTPSpecStepPlan &plan,
            std::string *error = nullptr) override;
        /**
         * @brief Publish accepted MTP verifier state for a request batch on
         *        every LocalTP or LocalPP participant.
         *
         * The batch form is the canonical vLLM-style publication contract.  A
         * rank-level runner must clamp every request to a common accepted
         * prefix across the topology before any child mutates live KV,
         * recurrent state, or terminal hidden buffers.
         */
        bool publishAcceptedMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override;
        /**
         * @brief Publish a grouped decode-equivalent MTP transaction on every
         *        LocalTP or LocalPP participant.
         *
         * The rank first clamps accepted counts through the same common-prefix
         * coordination used by direct batch publication.  It then calls each
         * child runner's grouped decode-equivalent publisher, not the ordinary
         * direct publisher, so LocalTP can use grouped verifier rows without
         * accidentally advertising the stronger all-position contract.
         *
         * @param plans Host-visible accepted-row plan for every logical request.
         * @param error Optional destination for the first participant failure.
         * @return true when every participant published the common accepted
         *         prefix and no child observed a divergent publication plan.
         */
        bool publishGroupedDecodeEquivalentMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override;
        /**
         * @brief True when LocalTP has child-resident verifier publication.
         *
         * Multi-child GPU LocalTP only advertises this when every child owns a
         * mirrored full-vocabulary MTP verifier head and can publish the compact
         * outcome it produced itself.  Rank-owned compact metadata uploads are not
         * a production path.
         */
        bool supportsDeviceResidentMTPSpecStatePublication() const override;
        /**
         * @brief Publish LocalTP accepted state from child-resident outcomes.
         *
         * The outcome handle must have been produced by
         * verifyGreedyAllPositionBatchOutcomeOnDeviceResident() on this same
         * RankOrchestrator instance or by the stochastic resident request-batch
         * verifier.  Publication feeds every child the exact handle produced by
         * that child; it must not expand the outcome into host step plans, upload
         * rank compact metadata, or call serial row replay.
         */
        bool publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
            const DeviceSpeculativePublicationRequest &request,
            std::string *error = nullptr) override;
        const float *getAllPositionLogits() const override;
        std::string mtpDecodeUnsupportedReason() const override;
        bool supportsMTPSidecarLogitsStreamHandoff() const override;
        bool supportsMTPDeviceDraftTokenInput() const override;
        bool supportsMTPSidecarPreservesMainState() const override;
        bool supportsMTPShiftedRowReuseFromSidecar() const override;
        bool usesMirroredLocalTPMTPHeadForVerifier() const override;
        bool supportsGreedyAllPositionBatchOutcomeOnDevice() const override;
        bool applyPenaltiesOnDevice(
            const std::vector<LogitPenalty> &penalties,
            int vocab_size) override;
        bool applyPenaltiesToMTPLogitsOnDevice(
            const std::vector<LogitPenalty> &penalties,
            int vocab_size) override;
        bool applyPenaltiesToAllPositionLogitsOnDeviceRow(
            int row,
            const std::vector<LogitPenalty> &penalties,
            int vocab_size) override;
        bool supportsRowLocalAllPositionPenaltyApplication() const override;
        int sampleGreedyFromMTPLogitsOnDevice() override;
        bool sampleGreedyFromMTPLogitsToDeviceDraftSlot(
            int draft_sample_slot,
            int32_t *out_token) override;
        bool sampleGreedyFromMainLogitsToDeviceTargetSlot(
            int target_sample_slot,
            int32_t *out_token) override;
        int sampleGreedyFromAllPositionLogitsOnDevice(int row) override;
        bool sampleGreedyFromAllPositionLogitsOnDeviceRows(
            int start_row,
            int row_count,
            int32_t *out_tokens) override;
        bool verifyGreedyAllPositionBatchOutcomeOnDevice(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeVerifyBatchOutcome *out) override;
        bool verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;
        bool verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
            const DeviceGreedyBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;
        /**
         * @brief Materialize a rank-owned compact verifier handle for response
         *        bookkeeping.
         *
         * This is a compatibility bridge.  It copies from small rank-owned
         * compact arrays, not from row replay, and the publication path consumes
         * the same compact summary before this bridge is needed for response
         * tokens.
         */
        bool copyDeviceSpeculativeOutcomesToHost(
            const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes) override;
        bool supportsDeviceStochasticMTPVerification() const override;
        bool buildStochasticDistributionOnDevice(
            DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size) override;
        bool buildStochasticDistributionsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override;
        bool buildStochasticProcessedLogitRowsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override;
        int sampleStochasticDraftProposalOnDevice(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold) override;
        bool sampleStochasticDraftProposalOnDeviceDeferred(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold) override;
        int sampleStochasticDistributionOnDevice(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold) override;
        bool sampleStochasticDistributionOnDeviceDeferred(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold) override;
        bool stageStochasticDraftTokensForDeviceVerification(
            const int32_t *draft_tokens,
            int draft_token_count,
            int first_draft_slot = 0) override;
        bool stageStochasticTargetTokenForDeviceSampling(
            int32_t target_token,
            int target_sample_slot = 0) override;
        const void *prepareMTPVerifierInputTokensOnDevice(
            int32_t first_token,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override;
        const void *prepareMTPVerifierInputTokenBatchOnDevice(
            const DeviceMTPVerifierInputBatchRequest *requests,
            int request_count,
            int padded_seq_len) override;
        const void *prepareMTPVerifierInputTokensOnDeviceFromHostRow(
            const int32_t *verifier_tokens,
            int total_verifier_input_tokens,
            int draft_token_count) override;
        const void *prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
            int first_target_sample_slot,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override;
        bool verifyStochasticDistributionsOnDevice(
            int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            DeviceSpeculativeVerifyResult *out) override;
        bool verifyStochasticDistributionsBatchOnDevice(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            DeviceSpeculativeVerifyResult *out) override;
        bool verifyStochasticDistributionsBatchOutcomeOnDevice(
            int first_target_slot,
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
        bool verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
            int first_target_slot,
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
        bool verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override;
        void setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled) override;
        void setMTPMainDecodeSyncDeferralEnabled(bool enabled) override;

        /**
         * @brief GPU-side greedy sampling for decode
         *
         * Runs argmax on each device's local logits, D2H only the result pair (8 bytes per device).
         * Avoids D2H-ing the entire combined logits tensor (~600 KB for 152K vocab).
         *
         * @return Token ID (>= 0) if on-device sampling succeeded, -1 if not supported
         */
        int sampleGreedyOnDevice() override;

        /**
         * @brief GPU-side sampling with top-k/top-p support
         *
         * For greedy, uses per-device argmax. For non-greedy, runs GPU top-k
         * per device, then host-side merge + softmax + top-p + sample.
         */
        int sampleOnDevice(const SamplingParams &params) override;
        int sampleOnDeviceAtLogicalPosition(
            const SamplingParams &params,
            int logical_position) override;
        bool requiresMPICoordinatedDecodeSampling(const SamplingParams &params) const override;

        /**
         * @brief Enable GPU-side decode sampling (skip D2H gatherLogits for seq_len=1)
         *
         * When enabled, forwardTP() skips gatherLogits for decode tokens.
         * Caller MUST use sampleGreedyOnDevice() instead of logits() for decode.
         */
        void setSkipLogitsGatherDecode(bool skip) override;

        /**
         * @brief Skip logits gather after prefill (seq_len > 1)
         *
         * In the standard generation flow, prefill logits are never consumed —
         * the first generated token comes from a decode step. Skipping the
         * D2H gather eliminates massive PCIe traffic for multi-token forwards.
         */
        void setSkipLogitsGatherPrefill(bool skip) override;

        void setSuppressTimeline(bool suppress) override;
        void setAccumulatePrefill(bool accumulate) override;
        void flushStageTimeline() override;

        /**
         * @brief Batched forward pass
         *
         * @param token_batches Vector of token sequences
         * @return true if forward pass succeeded
         */
        bool forward_batch(const std::vector<std::vector<int>> &token_batches) override;

        /**
         * @brief Get logits for a specific sequence in batch
         *
         * @param seq_idx Sequence index in batch
         * @return Pointer to logits [padded_seq_len, vocab_size], or nullptr
         */
        const float *getLogits(int seq_idx = 0) const override;

        /**
         * @brief Get current batch size
         */
        int batch_size() const override;

        /**
         * @brief Get padded sequence length for current batch
         */
        int padded_seq_len() const override;

        /**
         * @brief Get sequence lengths for current batch
         */
        const std::vector<int> &sequence_lengths() const override;

        /**
         * @brief Get vocabulary size
         */
        int vocab_size() const override;

        /**
         * @brief Get current parallelism mode (PP, TP, or TP_PP)
         *
         * Returns the effective mode after AUTO resolution.
         * Useful for testing and diagnostics.
         */
        ParallelismMode effectiveMode() const { return mode_; }

        /**
         * @brief Reset request-scoped live inference state on every participant.
         *
         * This is the multi-device request boundary corresponding to
         * IInferenceRunner::resetInferenceState(). It must clear KV/GDN,
         * logical positions, MTP sidecars/handoffs, and request-local metadata
         * across all child runners in lockstep while preserving child graph
         * topology, prepared weights, workspaces, and device contexts.
         */
        void resetInferenceState(const InferenceStateResetRequest &request) override;
        void clear_cache() override;
        void drainCompletedDecodeBoundaryMaintenanceDiagnostics() override;

        /**
         * @brief Get current position in cache
         */
        int get_position() const override;

        /**
         * @brief Get execution path type
         */
        ExecutionPath executionPath() const override;

        /**
         * @brief Get architecture name
         */
        const char *architecture() const override;
        uint64_t moePlacementEpoch() const override;
        uint64_t moeRuntimeMovementEpoch() const override;

        /**
         * @brief Aggregate per-runner runtime state for prefix-cache/MTP probes.
         */
        PrefixRuntimeStateSnapshot prefixStateProbe() const override;

        PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) override;
        bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0) override;
        bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count) override;
        bool restorePrefixTerminalState(const PrefixLookupResult &hit) override;
        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override;
        PrefixStateSnapshot captureLivePrefixCheckpoint(int seq_idx = 0) const override;
        bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0) override;
        bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0) override;

        // =====================================================================
        // Hidden State API (for Pipeline Parallelism nesting)
        // =====================================================================

        /**
         * @brief Get final hidden state from last forward pass
         *
         * In TP mode: Delegates to primary device runner (device_runners_[0])
         * In PP/TP_PP mode: Delegates to last stage runner (has final hidden state)
         *
         * @return Pointer to hidden state tensor, or nullptr if unavailable
         */
        TensorBase *getHiddenState() override;
        const TensorBase *getHiddenState() const override;

        /**
         * @brief Set initial hidden state for forward pass
         *
         * In TP mode: Sets on ALL device runners (all need same input)
         * In PP/TP_PP mode: Sets on first stage runner (stage 0 receives input)
         *
         * @param hidden_state Tensor containing hidden state [seq_len, d_model]
         */
        void setHiddenState(TensorBase *hidden_state) override;

        /**
         * @brief Check if this orchestrator has hidden state set for next forward
         *
         * @return true if setHiddenState was called and not yet consumed/cleared
         */
        bool hasHiddenStateInput() const override;

        /**
         * @brief Clear hidden state input (reset to normal embedding mode)
         *
         * In TP mode: Clears on all device runners
         * In PP/TP_PP mode: Clears on first stage runner
         */
        void clearHiddenStateInput() override;

        // =====================================================================
        // Snapshot API (from IInferenceRunner)
        // =====================================================================

        /**
         * @brief Enable snapshot capture on all device runners
         *
         * @param output_dir Directory for snapshot output
         */
        void enableSnapshotCapture(const std::string &output_dir = "") override;
        void setSnapshotCaptureFilter(const std::vector<std::string> &keys) override;

        /**
         * @brief Disable snapshot capture on all device runners
         */
        void disableSnapshotCapture() override;

        /**
         * @brief Clear snapshots on all device runners
         */
        void clearSnapshots() override;

        /**
         * @brief Get snapshot from primary device runner
         *
         * @param key Snapshot identifier
         * @param out_size Output size in bytes
         * @return Pointer to snapshot data, or nullptr if not found
         */
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;

        /**
         * @brief Get snapshot with 2D shape metadata
         *
         * Delegates to device runners/PP stages, preserving shape info.
         */
        SnapshotInfo getSnapshotWithShape(const std::string &key) const override;

        /**
         * @brief Get all snapshot keys from primary device runner
         */
        std::vector<std::string> getSnapshotKeys() const override;

        /**
         * @brief Get tensor-parallel aware snapshot for a stage
         *
         * Retrieves snapshots from ALL device runners and combines them
         * according to the stage's sharding mode.
         *
         * @param key Snapshot identifier (e.g., "layer0_ATTENTION_CONTEXT")
         * @return TPSnapshot with per-device data and combined view
         */
        TPSnapshot getTPSnapshot(const std::string &key) const;

        /**
         * @brief Get all snapshot keys with their sharding modes
         *
         * @return Vector of (key, sharding_mode) pairs
         */
        std::vector<std::pair<std::string, SnapshotShardingMode>> getSnapshotKeysWithSharding() const;

        // =====================================================================
        // Profiling API (from IInferenceRunner)
        // =====================================================================

        /**
         * @brief Get aggregated executor statistics
         *
         * Returns combined stats from all device runners.
         *
         * @return Pointer to aggregated stats, or nullptr if unavailable
         */
        const GraphExecutorStats *executorStats() const override;

        /**
         * @brief Reset statistics on all device runners
         */
        void resetExecutorStats() override;

        // =====================================================================
        // Orchestration API (from IInferenceRunner)
        // =====================================================================

        /**
         * @brief Check if a PlacementPlan is configured
         */
        bool hasPlacementPlan() const override;

        /**
         * @brief Get the PlacementPlan (from primary device runner)
         */
        const PlacementPlan &getPlacementPlan() const override;

        // =====================================================================
        // IRankOrchestrator Interface
        // =====================================================================

        /**
         * @brief Get number of devices in LOCAL TP
         *
         * @return Device count (>= 1)
         */
        int device_count() const override;

        /**
         * @brief Get inference runner for a specific device
         *
         * @param device_idx 0-based device index
         * @return Pointer to device's inference runner
         * @throws std::out_of_range if device_idx >= device_count()
         */
        IInferenceRunner *deviceRunner(int device_idx) override;

        /**
         * @brief Get inference runner for a specific device (const)
         *
         * @param device_idx 0-based device index
         * @return Const pointer to device's inference runner
         * @throws std::out_of_range if device_idx >= device_count()
         */
        const IInferenceRunner *deviceRunner(int device_idx) const override;

        /**
         * @brief Get LOCAL TP context
         *
         * @return Pointer to TP context (may be nullptr for single device)
         */
        ILocalTPContext *localTPContext() override;

        /**
         * @brief Get LOCAL TP context (const)
         *
         * @return Const pointer to TP context
         */
        const ILocalTPContext *localTPContext() const override;

        /**
         * @brief Check if all devices are ready
         *
         * @return true if all device runners are initialized and ready
         */
        bool allDevicesReady() const override;

        /**
         * @brief Synchronize all devices
         *
         * Ensures all pending operations have completed.
         */
        void synchronizeDevices() override;

        MoERebalanceController *moeRebalanceController() const;
        std::vector<MoERebalanceController *> moeRebalanceControllers() const override;
        MoERebalanceController *moeRebalanceControllerForDomain(
            const std::string &domain_id) const override;
        bool usesDeviceSideMoERebalanceController() const override;

        struct MoEExpertMaskSnapshot
        {
            std::string domain_id;
            std::vector<std::vector<std::vector<bool>>> masks_by_participant;
            std::optional<std::vector<std::vector<std::vector<bool>>>> transfer_masks_by_participant;

            bool empty() const
            {
                return masks_by_participant.empty();
            }

            const std::vector<std::vector<std::vector<bool>>> *transferMasks() const
            {
                return transfer_masks_by_participant.has_value()
                           ? &transfer_masks_by_participant.value()
                           : nullptr;
            }
        };

        struct PreparedMoEExpertMaskUpdate
        {
            std::string domain_id;
            std::vector<std::vector<std::vector<bool>>> masks_by_participant;
            std::vector<ReceivedWeightsMap> received_by_device;
            std::vector<bool> gpu_direct_prepare_ok_by_device;

            bool empty() const
            {
                return masks_by_participant.empty();
            }
        };

        /**
         * Snapshot active and transfer masks from a controller on the caller thread.
         *
         * Async prepare must not read controller state while decode continues to
         * update runtime histograms, so callers freeze the masks first and pass
         * the snapshot to prepareMoEExpertMaskTransfersForAllDevices().
         */
        MoEExpertMaskSnapshot snapshotMoEExpertMasksForAllDevices(
            const MoERebalanceController &controller,
            const ExpertReplicaSet *replica_arrivals = nullptr,
            const std::vector<int> *previous_ownership_placement = nullptr) const;

        /**
         * Prepare local expert arrivals for a future mask publication.
         *
         * This snapshots the publish masks and gathers/stages received weights.
         * Same-backend GPU arrivals are staged into transfer slots only; active
         * GEMM tables and masks are mutated later by publishPreparedMoEExpertMasksForAllDevices().
         */
        PreparedMoEExpertMaskUpdate prepareMoEExpertMaskTransfersForAllDevices(
            const std::vector<std::vector<std::vector<bool>>> &masks_by_participant,
            const std::string &domain_id = {},
            const std::vector<std::vector<std::vector<bool>>> *transfer_masks_by_participant = nullptr);

        /// Publish a previously-prepared mask update. This is the only phase
        /// that should mutate active MoE masks once async staging is enabled.
        bool publishPreparedMoEExpertMasksForAllDevices(
            const PreparedMoEExpertMaskUpdate &prepared);
        void clearPendingGpuDirectExpertTransfersForAllDevices();
        bool applyMoEExpertMasksForAllDevices(
            const MoERebalanceController &controller,
            const ExpertReplicaSet *replica_arrivals = nullptr,
            const std::vector<int> *previous_ownership_placement = nullptr);
        bool applyMoEExpertMasksForAllDevices(
            const std::vector<std::vector<std::vector<bool>>> &masks_by_participant,
            const std::string &domain_id = {},
            const std::vector<std::vector<std::vector<bool>>> *transfer_masks_by_participant = nullptr);
        void setExpertReplicaSetForAllDevices(const ExpertReplicaSet &replicas);

    private:
        // =====================================================================
        // Private Constructor (for createForTest)
        // =====================================================================

        /**
         * @brief Private constructor for factory method
         */
        RankOrchestrator(
            std::shared_ptr<IModelContext> model_ctx,
            std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
            std::unique_ptr<ILocalTPContext> tp_ctx,
            const Config &config);

        // =====================================================================
        // Private Methods
        // =====================================================================

        /**
         * @brief Initialize device runners from configuration
         */
        void initializeDeviceRunners();

        /**
         * @brief Initialize device runners for PP mode
         *
         * Creates one DeviceGraphOrchestrator per PP stage (or one
         * RankOrchestrator per stage for TP+PP mode).
         */
        void initializePPDeviceRunners();

        /**
         * @brief Initialize PP context for inter-stage transfers
         *
         * Creates HierarchicalPPContext with appropriate stage types
         * (single device or TP domain) based on configuration.
         */
        void initializePPContext();

        /**
         * @brief Execute forward pass in TP mode (parallel)
         *
         * All devices execute the same layers in parallel with sharded weights.
         * AllReduce after row-parallel ops, AllGather for logits.
         *
         * @param tokens Input token IDs
         * @param seq_len Sequence length
         * @return true if forward pass succeeded on all devices
         */
        bool forwardTP(
            const int *tokens,
            int seq_len,
            bool force_prefill_phase = false,
            const PrefillChunkSchedulerPolicy *chunk_schedule_policy = nullptr,
            int chunk_schedule_pad_token_id = 0,
            bool chunk_schedule_allow_padded_execution = false);

        /**
         * @brief Execute forward pass in PP mode (sequential)
         *
         * Stages execute sequentially with hidden state transfers between stages.
         * Each stage processes a subset of layers.
         *
         * @param tokens Input token IDs
         * @param seq_len Sequence length
         * @return true if forward pass succeeded
         */
        bool forwardPP(const int *tokens, int seq_len, bool force_prefill_phase = false);

        /**
         * @brief Worker join timeout for TP runner operations.
         *
         * Collective timeouts are enforced inside LocalTP/NCCL/RCCL rendezvous
         * calls. Worker joins wait for the per-device runner operation to finish
         * so snapshot-heavy parity/debug forwards are not mistaken for a stuck
         * single collective.
         */
        int effectiveTPWorkerJoinTimeoutMs() const;

        /**
         * @brief Return the PP stage that owns MTP sidecar execution.
         *
         * In pipeline-parallel decode the normal verifier/replay path still
         * runs through every PP stage via forwardPP().  The Qwen3.6 MTP
         * sidecar, however, consumes the terminal hidden row and produces
         * sidecar logits, so it belongs to the final PP stage: the same stage
         * that owns output norm and the LM head.  Keeping this helper explicit
         * avoids accidentally treating PP stages like TP participants.
         */
        IInferenceRunner *finalPPSidecarRunner();

        /**
         * @brief Const overload of finalPPSidecarRunner().
         */
        const IInferenceRunner *finalPPSidecarRunner() const;

        /**
         * @brief Refresh rank-owned public sequence metadata from the primary child runner.
         *
         * DeviceGraphOrchestrator instances own the real KV, recurrent, short-conv,
         * terminal-hidden, and logits payloads.  RankOrchestrator still exposes
         * aggregate `position`, `padded_seq_len`, and `sequence_lengths` through
         * the same IInferenceRunner API, and debug/replay code reads those values
         * when comparing committed MTP state against serial replay.
         *
         * MTP spec-state publication is unusual because a verifier graph may first
         * advance rank-level bookkeeping for all verifier rows, then each child
         * runner publishes only the accepted prefix.  After publication succeeds,
         * the rank wrapper must adopt the child-visible accepted boundary so a
         * rejected correction token remains pending instead of looking as though
         * the rank already consumed it.
         */
        void refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();

        /**
         * @brief Aggregate stats from all device runners
         */
        void aggregateStats() const;

        /**
         * @brief Apply runtime-specific TP snapshot sharding overrides.
         *
         * Static schemas describe architectural defaults, but some snapshot
         * lifetimes depend on the concrete model dimensions and TP degree.  GQA
         * K/V streams are the important example: when n_kv_heads is smaller
         * than the TP degree, every participant owns the complete K/V view
         * rather than a column slice.  This helper keeps K/V projection, RoPE,
         * append-source, cache, and effective-attention snapshots under one
         * first-class runtime owner instead of scattering ad hoc overrides.
         */
        void applyRuntimeSnapshotShardingOverrides();

        SnapshotShardingMode resolveSnapshotShardingMode(const std::string &key) const;
        bool phaseSplitReplicatedDecodeSnapshotsActive() const;
        static bool isPhaseSplitReplicatedDecodeSnapshotKey(const std::string &key);

        /**
         * @brief Replay rank-level logits gather policy into gatherers/children.
         *
         * The skip flags are part of the rank runner contract: callers that use
         * GPU-side sampling expect every nested runner to avoid publishing
         * full logits to host during decode.  Keep this centralized so late
         * gatherer construction and nested TP/PP runners inherit the same
         * policy.
         */
        void applyLogitsGatherSkipFlags();

        /**
         * @brief Resolve the backend used by rank-owned GPU logits operations.
         *
         * Production ranks have no injected resolver and therefore use the
         * process-wide BackendManager.  The createForTest() constructor installs
         * a resolver that deliberately returns nullptr, allowing unit tests to
         * exercise CUDA/ROCm-shaped placement metadata without initializing a
         * physical GPU runtime.
         *
         * @param device GPU device that owns the logits payload.
         * @return Backend for the device, or nullptr when backend access is
         *         intentionally unavailable.
         */
        IBackend *resolveLogitsBackend(DeviceId device) const;

        /**
         * @brief Forward same-domain LocalTP child runtime histograms into the
         *        selected rebalance controller.
         *
         * Each DeviceGraphOrchestrator owns its runtime MoE table and registers
         * histogram sync callbacks against its local controller.  Rank-level
         * rebalancing, however, runs through a single selected controller per
         * domain.  This bridge keeps that selected controller's host view
         * representative of every LocalTP participant.
         */
        void wireLocalTPMoERuntimeHistogramSyncs();

        // =====================================================================
        // Member Variables
        // =====================================================================

        /// Model context (shared across device runners)
        std::shared_ptr<IModelContext> model_ctx_;

        /// PP activation transfer contract (built during initializePPDeviceRunners)
        std::unique_ptr<PPActivationContract> pp_activation_contract_;

        /// LOCAL TP context for collective operations (TP mode)
        std::unique_ptr<ILocalTPContext> tp_ctx_;

        /// LOCAL PP context for inter-stage transfers (PP mode)
        std::unique_ptr<ILocalPPContext> pp_ctx_;

        /// Effective parallelism mode (resolved from config)
        ParallelismMode mode_ = ParallelismMode::TP;

        /// Per-device inference runners
        /// In TP mode: one runner per device
        /// In PP mode: one runner per stage
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners_;

        /// PP stage runners (when stages are TP domains, these are RankOrchestrator)
        /// Only used in TP+PP mode - in pure PP mode, device_runners_ holds stage runners
        std::vector<std::unique_ptr<IInferenceRunner>> pp_stage_runners_;

        /// Per-child prefix hits captured during the last rank-level lookup.
        std::vector<PrefixLookupResult> last_device_prefix_hits_;
        std::vector<PrefixLookupResult> last_pp_prefix_hits_;

        /// Configuration
        Config config_;

        /// Logits buffer management and D2H gather operations (extracted helper)
        std::unique_ptr<LogitsGatherer> logits_gatherer_;
        mutable std::unique_ptr<LogitsGatherer> mtp_logits_gatherer_;
        mutable std::unique_ptr<LogitsGatherer> all_position_logits_gatherer_;
        IBackend *(*logits_backend_resolver_)(DeviceId) = nullptr;
        bool skip_logits_gather_decode_ = false;
        bool skip_logits_gather_prefill_ = false;

        /// Aggregated executor stats (mutable for lazy computation)
        mutable std::unique_ptr<GraphExecutorStats> aggregated_stats_;

        /// Current position in KV cache
        int current_position_ = 0;

        /// Current batch size
        int current_batch_size_ = 1;

        /// Padded sequence length for current batch
        int current_padded_seq_len_ = 0;

        /// Compact all-position verifier rows requested by the current MTP path.
        /// A value of zero means gather every row from current_padded_seq_len_.
        int current_all_position_logit_rows_ = 0;

        /// Sequence lengths for current batch
        std::vector<int> current_sequence_lengths_;

        /// Flag indicating if stats need re-aggregation
        mutable bool stats_dirty_ = true;
        bool host_resident_released_ = false; ///< Whether host-resident weight data has been released after first prefill
        bool mmap_dontneed_advised_ = false;  ///< Whether mmap pages were advised away after first prefill

        /// Stage type → sharding mode map from the model's schema factory.
        /// Initialized at construction from SchemaFactoryRegistry::getStageShardingConfig().
        /// Replaces hardcoded getStageShardingMode() lookups for snapshot reassembly.
        StageShardingConfig stage_sharding_map_;

        /// Hidden state input for PP nesting (when this orchestrator is a PP stage)
        /// Set via setHiddenState(), cleared after forward or via clearHiddenStateInput()
        TensorBase *hidden_state_input_ = nullptr;

        /// Persistent worker pool for TP device forwarding.
        /// Eliminates per-decode thread creation/destruction overhead (~100-150µs).
        /// Lazy-initialized on first TP forward call.
        std::unique_ptr<TPWorkerPool> tp_worker_pool_;
        bool tp_first_forward_completed_ = false;

        /**
         * @brief Permit this rank to initialize or synchronize physical GPU backends.
         *
         * Production constructors leave this enabled. `createForTest()` injects
         * mock child runners that may advertise CUDA/ROCm-shaped device IDs to
         * exercise placement policy, but unit tests must never initialize those
         * physical backends. The private test constructor therefore disables
         * backend pinning and mmap-release synchronization explicitly.
         */
        bool external_device_backend_access_enabled_ = true;
        /**
         * @brief Scoped switch that routes rank fan-out to grouped child APIs.
         *
         * publishAcceptedMTPSpecStateBatch() owns the common-prefix clamping and
         * parallel worker-pool dispatch code.  Reusing it avoids a second
         * publication state machine, but this flag makes the child calls use
         * publishGroupedDecodeEquivalentMTPSpecStateBatch() while the grouped
         * wrapper is active.
         */
        bool grouped_decode_equivalent_spec_publication_scope_ = false;

        /**
         * @brief Rank-local compact verifier scratch for diagnostics.
         *
         * GPU LocalTP production MTP uses child-owned mirrored verifier outcomes.
         * These arrays remain available for response materialization in rank-level
         * diagnostic reducers and CPU-local coordination paths; they must not be
         * uploaded into GPU child runners for live-state publication.
         */
        std::array<int32_t, sampling_math::kSpeculativeBatchMaxOutputTokens>
            rank_compact_output_tokens_{};
        std::array<int, sampling_math::kSpeculativeBatchMetaCount>
            rank_compact_output_meta_{};
        enum class RankCompactOutcomeKind : uint8_t
        {
            None,
            Greedy,
            Stochastic,
            MirroredGreedy,
            MirroredStochastic
        };
        RankCompactOutcomeKind rank_compact_outcome_kind_ =
            RankCompactOutcomeKind::None;
        std::vector<int32_t> rank_compact_last_draft_tokens_;
        std::vector<int32_t> rank_compact_last_stop_tokens_;
        std::vector<const void *> rank_mtp_verifier_child_token_inputs_;
        int rank_mtp_verifier_child_token_count_ = 0;

        /**
         * @brief LocalTP stochastic distribution slot capacity.
         *
         * The grouped MTP verifier needs one slot per speculative comparison
         * row plus one bonus/first-token slot.  Keep the rank scratch shape
         * identical to the backend compact-summary ABI so the rank-level
         * reducer and single-device GPU reducers share the same bounds.
         */
        static constexpr int kRankStochasticMaxSlots =
            sampling_math::kSpeculativeBatchMaxOutputTokens;

        /**
         * @brief Rank-owned compact stochastic target tables for LocalTP.
         *
         * Multi-child LocalTP cannot ask one child to build a full-vocab
         * stochastic distribution because every child owns only one LM-head
         * shard.  RankOrchestrator therefore gathers only each shard's top-k
         * candidates, merges those small candidate lists, and stores the final
         * compact tables here for the later verifier reducer.  These arrays are
         * small: at most `(max verifier rows + bonus) * 256` entries.
         */
        std::array<int,
                   kRankStochasticMaxSlots * sampling_math::kMaxTopK>
            rank_stochastic_target_token_ids_{};
        std::array<float,
                   kRankStochasticMaxSlots * sampling_math::kMaxTopK>
            rank_stochastic_target_probs_{};
        std::array<int, kRankStochasticMaxSlots>
            rank_stochastic_target_top_k_{};
        std::array<int32_t, kRankStochasticMaxSlots>
            rank_stochastic_target_sample_tokens_{};
        std::array<int32_t, kRankStochasticMaxSlots>
            rank_stochastic_draft_sample_tokens_{};
        std::vector<int32_t> rank_stochastic_staged_draft_tokens_;

        /**
         * @brief Build one rank-owned stochastic distribution from TP shards.
         *
         * @param source Which logits surface to read from each child runner.
         * @param row Logical row within that surface.
         * @param buffer Target or draft compact table namespace.
         * @param slot Compact distribution slot to write.
         * @param params Sampling parameters; `top_k` must be in [1, 256].
         * @param vocab_size Full vocabulary size across all shards.
         * @return true when the rank compact table was populated.
         */
        bool buildRankStochasticDistributionFromLocalTP(
            DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size);

        /**
         * @brief Build several contiguous target rows from TP shard logits.
         *
         * This is the LocalTP counterpart to backend batched target-table
         * construction.  It consumes one local-logits view per child and reuses
         * it for every requested row so graph-produced all-position logits are
         * ordered behind the same producer streams.
         */
        bool buildRankStochasticTargetDistributionsFromLocalTPRows(
            DeviceLogitsSource source,
            int first_row,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size);

        /**
         * @brief Sample a rank-owned compact distribution with a caller draw.
         *
         * The sampled token is recorded in the rank target/draft sample slot so
         * later device-token verifier input composition can resolve deferred
         * sentinels without reading full logits or replaying verifier rows.
         */
        bool sampleRankStochasticDistribution(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold,
            int32_t *out_token);

        /**
         * @brief Resolve a LocalTP greedy MTP-head sample into a child draft slot.
         *
         * The MTP sidecar LM head is vocab-sharded under LocalTP, so no child can
         * independently choose the full-vocab greedy draft token.  The rank
         * consumes each child's compact local-logits metadata, performs the same
         * cross-shard argmax used by immediate greedy sampling, and then stages
         * the single winning global token into every child draft slot.  This is
         * the grouped verifier production path: it avoids a full-logit host
         * gather, avoids row replay, and preserves the device-slot contract used
         * by greedy and stochastic verifier input materialization.
         */
        bool sampleRankGreedyMTPLogitsToLocalTPDraftSlot(
            int draft_sample_slot,
            int32_t *out_token);

        /**
         * @brief Resolve a LocalTP greedy main-logits sample into a child target slot.
         *
         * Greedy MTP first-token deferral uses the same child-owned target-token
         * arena as stochastic verification.  In LocalTP no child owns full logits,
         * so the rank performs the economical cross-shard argmax from each
         * child's local logits metadata, records the compact winning token, and
         * fans that one token into every child device target slot.  This keeps the
         * verifier path grouped and device-slot based without falling back to a
         * full-logit host gather.
         */
        bool sampleRankGreedyMainLogitsToLocalTPTargetSlot(
            int target_sample_slot,
            int32_t *out_token);

        /**
         * @brief Install one rank-resolved target token into every child slot.
         *
         * The LocalTP rank reducer owns the compact full-vocab sample decision,
         * but each child graph owns the device buffer consumed by sidecar and
         * verifier token-row materialization.  This helper fans the small token
         * upload through the child runner API so production verifier setup can
         * use child device-slot plans instead of row-wide host staging.
         */
        bool stageRankStochasticTargetTokenForLocalTP(int slot, int32_t token);

        /**
         * @brief Install rank-resolved draft tokens into every child slot.
         *
         * Draft proposals are sampled once at rank scope from TP-sharded MTP
         * logits.  Every participant then receives the same compact token in
         * its runner-owned draft slot, preserving the participant-symmetric
         * sidecar sequence while avoiding full-logit gather or verifier row
         * replay.
         */
        bool stageRankStochasticDraftTokensForLocalTP(
            const int32_t *tokens,
            int token_count,
            int first_slot);

        /**
         * @brief Broadcast the primary mirrored child draft slot to every child.
         *
         * Deferred stochastic LocalTP keeps MTP proposals device-resident.  With a
         * mirrored MTP head, child 0 samples the full-vocabulary proposal once and
         * the rank broadcasts that one INT32 mailbox slot to all other children.
         * Each child then records the broadcast stream as the slot's producer, so
         * chained sidecars and verifier reducers consume one common rank-owned
         * draft token without reading it back to the host.
         */
        bool broadcastPrimaryMirroredLocalTPDraftSampleSlotToChildren(int slot);

        /**
         * @brief Ask every child to materialize a verifier row from device slots.
         *
         * Entry zero may be a host scalar or a child target sample slot; draft
         * entries always come from staged child draft sample slots.  The return
         * value is the same rank-owned child-pointer bundle consumed by
         * forwardWithDeviceTokenIds().
         */
        const void *prepareRankVerifierTokenSlotsForLocalTP(
            bool first_token_from_device,
            int32_t first_token,
            int first_target_sample_slot,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens);

        /**
         * @brief Compose and stage the verifier token row for every TP child.
         *
         * LocalTP stochastic sampling resolves full-vocab tokens at the rank
         * level.  Child graphs still need a device-resident token row for the
         * verifier forward, so RankOrchestrator builds the tiny logical row and
         * asks each child to stage it with its existing host-row staging API.
         */
        const void *stageRankVerifierTokenRowForLocalTP(
            const int32_t *tokens,
            int total_verifier_input_tokens,
            int draft_token_count);

        /**
         * @brief Resolve deferred LocalTP greedy outcome tokens from rank slots.
         *
         * Device-resident greedy MTP can intentionally pass compact shadow
         * sentinels through OrchestrationRunner while the actual first token and
         * draft tokens live in rank-owned target/draft sample slots.  The
         * rank-level outcome reducer must summarize the real tokens that the
         * verifier graph consumed, not the negative shadows.  This helper maps a
         * negative first entry to target slot zero and negative draft entries to
         * the corresponding draft slots, returning a compact host row suitable
         * for SamplingMath metadata generation and publication bookkeeping.
         */
        bool resolveRankGreedyOutcomeTokensForLocalTP(
            const int32_t *draft_tokens,
            int draft_token_count,
            std::vector<int32_t> *out_tokens) const;

        /**
         * @brief Run greedy verifier reduction on mirrored LocalTP children.
         *
         * Every child owns a replicated MTP verifier head and therefore has the
         * local state needed to publish accepted rows.  The compact accept/reject
         * decision, however, is a single rank-level fact: all children must publish
         * the same accepted prefix and next-condition token.  After child-local
         * reducers produce resident mailboxes, the rank broadcasts the primary
         * child's compact outcome into every peer mailbox on device before any
         * live-state publication occurs.
         */
        bool verifyGreedyMirroredLocalTPBatchOutcomeOnDeviceResident(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeOutcomeHandle *out_handle);

        /**
         * @brief Run stochastic verifier reduction on mirrored children.
         *
         * Stochastic MTP must be batch-invariant across the LocalTP group.  The
         * primary child computes the authoritative compact stochastic outcome,
         * then a device-side LocalTP broadcast copies that compact token/meta row
         * into every child-owned resident outcome buffer.  Publication can then
         * remain child-local for KV/GDN/terminal-hidden state while consuming one
         * common accept/reject decision.
         */
        bool verifyStochasticMirroredLocalTPRequestBatchOutcomesOnDeviceResident(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle);

        /**
         * @brief Broadcast the primary mirrored outcome into all child mailboxes.
         *
         * The child handles in @p child_outcomes own device-resident compact
         * output-token and metadata buffers.  This helper validates that every
         * handle is current, then enqueues two INT32 LocalTP broadcast sidebands
         * on each participant stream: one for output tokens and one for metadata.
         * No host copy or row replay is involved; NCCL/RCCL provide the device
         * transport for homogeneous GPU LocalTP domains.
         *
         * @param child_outcomes Per-child resident compact outcome handles.
         * @param context_name Short diagnostic name attached to perfstats/errors.
         * @return true when every child mailbox now contains the primary compact
         *         outcome and remains ordered on its own stream.
         */
        bool broadcastPrimaryMirroredLocalTPDeviceOutcomeToChildren(
            std::vector<DeviceSpeculativeOutcomeHandle> &child_outcomes,
            const char *context_name);

        /**
         * @brief Publish child-resident outcomes from mirrored LocalTP verification.
         *
         * The request outcome must be the primary child handle returned by the
         * most recent mirrored greedy or stochastic verifier reduction. Each
         * participant receives its own stored handle after the rank has broadcast
         * the primary compact outcome into all child mailboxes, preserving stream
         * ownership while guaranteeing one common accepted count.
         */
        bool publishMirroredLocalTPDeviceResidentMTPSpecStateBatch(
            const DeviceSpeculativePublicationRequest &request,
            std::string *error);

        /**
         * @brief Return a previously sampled rank draft token slot.
         */
        int32_t rankStochasticDraftSampleToken(int slot) const;

        /**
         * @brief Return a previously sampled rank target token slot.
         */
        int32_t rankStochasticTargetSampleToken(int slot) const;
        mutable std::vector<DeviceResidentLogicalSequenceStateHandle>
            rank_resident_child_logical_state_handles_;
        mutable std::array<int32_t, 1> rank_resident_logical_state_marker_{};
        mutable int rank_resident_logical_state_stream_token_ = 0;
        mutable int rank_resident_logical_state_ready_event_token_ = 0;
        mutable uint64_t rank_resident_logical_state_epoch_ = 1;
        std::vector<DeviceSpeculativeOutcomeHandle>
            rank_mirrored_child_outcomes_;
        DeviceSpeculativeOutcomeHandle rank_mirrored_primary_outcome_;
        bool rank_mirrored_child_outcomes_valid_ = false;
        int rank_compact_outcome_stream_token_ = 0;
        int rank_compact_outcome_ready_event_token_ = 0;
        bool rank_compact_outcome_valid_ = false;

        /// Guard against registering duplicate sibling histogram callbacks.
        bool local_tp_moe_histogram_syncs_wired_ = false;

        // =====================================================================
        // TP Decode Profiling
        //
        // Lightweight wall-clock accumulation of the orchestrator-level decode
        // lifecycle. Measured at the forwardTP() level — above all per-device
        // GPU work — revealing dispatch/collect/gather overhead invisible to
        // per-device StageTimeline.  Gated on LLAMINAR_PROFILING=1.
        // Printed by flushStageTimeline() at benchmark end.
        // =====================================================================
        struct TPDecodeStats
        {
            double total_wall_ms = 0;     ///< Total forwardTP wall time (decode only)
            double total_dispatch_ms = 0; ///< Time to dispatch workers (condition_variable notify)
            double total_wait_ms = 0;     ///< Time blocked on collectAll (waiting for slowest device)
            double total_gather_ms = 0;   ///< Time in gatherLogits
            size_t iterations = 0;        ///< Number of decode steps

            void record(double wall_ms, double dispatch_ms, double wait_ms, double gather_ms)
            {
                total_wall_ms += wall_ms;
                total_dispatch_ms += dispatch_ms;
                total_wait_ms += wait_ms;
                total_gather_ms += gather_ms;
                iterations++;
            }

            void reset()
            {
                total_wall_ms = 0;
                total_dispatch_ms = 0;
                total_wait_ms = 0;
                total_gather_ms = 0;
                iterations = 0;
            }
        };

        TPDecodeStats tp_decode_stats_;
    };

} // namespace llaminar2
