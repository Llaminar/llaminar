/**
 * @file MultiDeviceOrchestrator.h
 * @brief Multi-device orchestrator for LOCAL tensor parallelism
 *
 * Coordinates multiple DeviceGraphOrchestrator instances for LOCAL tensor
 * parallelism across multiple devices within a single MPI rank.
 *
 * Key concepts:
 * - LOCAL TP: Multiple devices owned by one MPI rank (decoupled from MPI world_size)
 * - Proportional TP: Devices can have different capacities (weights)
 * - Backend selection: NCCL, RCCL, PCIeBAR, or HOST based on device types
 *
 * Design philosophy:
 * - Extends IMultiDeviceOrchestrator (which extends IInferenceRunner)
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
 * MultiDeviceOrchestrator::Config config;
 * config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
 * config.weights = {0.73f, 0.27f};  // Optional proportional weights
 * config.backend = CollectiveBackendType::NCCL;
 *
 * auto orchestrator = std::make_unique<MultiDeviceOrchestrator>(model_ctx, config);
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

#include "IMultiDeviceOrchestrator.h"
#include "../../../backends/GlobalDeviceAddress.h"
#include "../../../config/OrchestrationConfig.h"
#include "../../config/RuntimeConfig.h"
#include "../../debug/TPSnapshot.h"
#include <memory>
#include <vector>
#include <string>

namespace llaminar2
{

    // Forward declarations
    class DeviceGraphOrchestrator;
    class IModelContext;
    class ILocalTPContext;
    class TensorBase;
    struct GraphExecutorStats;
    struct PlacementPlan;

    /**
     * @brief Multi-device orchestrator for LOCAL tensor parallelism
     *
     * Coordinates multiple DeviceGraphOrchestrator instances to enable tensor
     * parallelism across devices within a single MPI rank. This is the primary
     * implementation of IMultiDeviceOrchestrator.
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
    class MultiDeviceOrchestrator : public IMultiDeviceOrchestrator
    {
    public:
        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Configuration for multi-device orchestration
         *
         * Specifies devices, weights, backend, and inference parameters.
         */
        struct Config
        {
            /// Devices participating in LOCAL TP
            std::vector<GlobalDeviceAddress> devices;

            /// Proportional weights for work distribution (sum to 1.0)
            /// Empty or unset: equal distribution
            /// Example: {0.73f, 0.27f} for 73%/27% split
            std::vector<float> weights;

            /// Backend for collective operations (AUTO for auto-detection)
            CollectiveBackendType backend = CollectiveBackendType::AUTO;

            /// Maximum sequence length
            size_t max_seq_len = 4096;

            /// Batch size for inference
            int batch_size = 1;

            /// Activation precision for intermediate buffers
            ActivationPrecision activation_precision = ActivationPrecision::FP32;

            /// KV cache scale factor
            float kv_cache_scale = 1.0f;

            /// Use mapped memory for GPU tensors (zero-copy host access)
            /// Required for correct coherence with column-parallel LM head
            bool use_mapped_memory = false;

            /**
             * @brief Validate configuration
             *
             * Checks:
             * - At least one device specified
             * - Weights sum to ~1.0 (if provided)
             * - Weights count matches device count (if provided)
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
         * @return Unique pointer to MultiDeviceOrchestrator
         *
         * @code
         * // Test setup with mocks
         * auto model_ctx = std::make_shared<MockModelContext>(preset);
         * std::vector<std::unique_ptr<DeviceGraphOrchestrator>> runners;
         * runners.push_back(createMockRunner(cuda0));
         * runners.push_back(createMockRunner(cuda1));
         * auto tp_ctx = std::make_unique<MockLocalTPContext>(devices, weights);
         *
         * auto orchestrator = MultiDeviceOrchestrator::createForTest(
         *     model_ctx, std::move(runners), std::move(tp_ctx), config);
         * @endcode
         */
        static std::unique_ptr<MultiDeviceOrchestrator> createForTest(
            std::shared_ptr<IModelContext> model_ctx,
            std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
            std::unique_ptr<ILocalTPContext> tp_ctx,
            const Config &config);

        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Construct with model context and configuration
         *
         * Creates device runners and TP context based on configuration.
         *
         * @param model_ctx Model context with weights and metadata
         * @param config Multi-device configuration
         */
        MultiDeviceOrchestrator(
            std::shared_ptr<IModelContext> model_ctx,
            const Config &config);

        /**
         * @brief Construct with pre-existing TP context
         *
         * Uses provided TP context instead of creating one from config.
         * Useful when sharing TP context across multiple orchestrators.
         *
         * @param model_ctx Model context with weights and metadata
         * @param tp_ctx Pre-constructed LOCAL TP context (ownership transferred)
         * @param config Multi-device configuration (devices/weights from tp_ctx take precedence)
         */
        MultiDeviceOrchestrator(
            std::shared_ptr<IModelContext> model_ctx,
            std::unique_ptr<ILocalTPContext> tp_ctx,
            const Config &config);

        /// Destructor
        ~MultiDeviceOrchestrator() override;

        // Non-copyable, movable
        MultiDeviceOrchestrator(const MultiDeviceOrchestrator &) = delete;
        MultiDeviceOrchestrator &operator=(const MultiDeviceOrchestrator &) = delete;
        MultiDeviceOrchestrator(MultiDeviceOrchestrator &&) noexcept;
        MultiDeviceOrchestrator &operator=(MultiDeviceOrchestrator &&) noexcept;

        // =====================================================================
        // IInferenceRunner Interface (from IMultiDeviceOrchestrator)
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
        // IMultiDeviceOrchestrator Interface
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

    private:
        // =====================================================================
        // Private Constructor (for createForTest)
        // =====================================================================

        /**
         * @brief Private constructor for factory method
         */
        MultiDeviceOrchestrator(
            std::shared_ptr<IModelContext> model_ctx,
            std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
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
         * @brief Gather partial logits from all devices
         *
         * Performs AllGather of local logits shards into combined buffer.
         *
         * @param seq_len The actual sequence length (number of tokens) for this forward pass.
         *                Required because logits_local buffer is pre-allocated for max_seq_len,
         *                but for decode we only want to gather 1 row.
         * @return true if gather succeeded
         */
        bool gatherLogits(size_t seq_len);

        /**
         * @brief Aggregate stats from all device runners
         */
        void aggregateStats() const;

        // =====================================================================
        // Member Variables
        // =====================================================================

        /// Model context (shared across device runners)
        std::shared_ptr<IModelContext> model_ctx_;

        /// LOCAL TP context for collective operations
        std::unique_ptr<ILocalTPContext> tp_ctx_;

        /// Per-device inference runners
        std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners_;

        /// Configuration
        Config config_;

        /// Combined logits buffer after AllGather [vocab_size]
        std::unique_ptr<TensorBase> combined_logits_;

        /// Actual size of gathered logits from last gatherLogits() call
        /// This may be smaller than combined_logits_->numel() for decode (1 token vs max_seq_len)
        size_t last_gathered_logits_size_ = 0;

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
    };

} // namespace llaminar2
