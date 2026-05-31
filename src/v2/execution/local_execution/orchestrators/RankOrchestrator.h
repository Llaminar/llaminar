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
#include "../../config/RuntimeConfig.h"
#include "../../debug/TPSnapshot.h"
#include "../../factory/FactoryPPStageConfig.h" // For FactoryPPStageConfig (circular-dependency-safe)
#include <memory>
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
    class TensorBase;
    class LogitsGatherer;
    class DeviceSampler;
    class IMPIContext;
    class PreparedWeightStore;
    struct GraphExecutorStats;
    struct MoEExpertParallelPlan;
    struct PlacementPlan;
    struct PPActivationContract;

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

            /// Prefix-state cache feature gates and storage limits.
            PrefixCacheRuntimeConfig prefix_cache;

            /// Multi-token prediction feature gates and verification mode.
            MTPRuntimeConfig mtp;

            /// Routed MoE expert execution mode for standard Qwen3.5 MoE.
            MoEExpertMode moe_expert_mode = MoEExpertMode::ExpertParallel;

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

        /**
         * @brief Get combined logits from last forward pass
         *
         * Returns logits gathered from all devices via AllGather.
         *
         * @return Pointer to combined logits [vocab_size], or nullptr if unavailable
         */
        const float *logits() const override;
        bool forwardMTP(int32_t draft_condition_token) override;
        const float *mtpLogits() const override;
        bool setComputeAllPositionLogits(bool enabled) override;
        const float *getAllPositionLogits() const override;
        std::string mtpDecodeUnsupportedReason() const override;

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
         * @brief Clear KV cache on all devices
         */
        void clear_cache() override;

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

        /**
         * @brief Aggregate per-runner runtime state for prefix-cache/MTP probes.
         */
        PrefixRuntimeStateSnapshot prefixStateProbe() const override;

        PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) override;
        bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0) override;
        bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count) override;
        bool restorePrefixTerminalState(const PrefixLookupResult &hit) override;
        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override;
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
        void applyMoEExpertMasksForAllDevices(const MoERebalanceController &controller);
        void applyMoEExpertMasksForAllDevices(
            const std::vector<std::vector<std::vector<bool>>> &masks_by_socket);
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
        bool forwardTP(const int *tokens, int seq_len);

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
        bool forwardPP(const int *tokens, int seq_len);

        /**
        /**
         * @brief Aggregate stats from all device runners
         */
        void aggregateStats() const;

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

        /// Aggregated executor stats (mutable for lazy computation)
        mutable std::unique_ptr<GraphExecutorStats> aggregated_stats_;

        /// Current position in KV cache
        int current_position_ = 0;

        /// Current batch size
        int current_batch_size_ = 1;

        /// Padded sequence length for current batch
        int current_padded_seq_len_ = 0;

        /// Sequence lengths for current batch
        std::vector<int> current_sequence_lengths_;

        /// Flag indicating if stats need re-aggregation
        mutable bool stats_dirty_ = true;
        bool host_resident_released_ = false; ///< Whether host-resident weight data has been released after first prefill

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
