/**
 * @file PipelineBase.h
 * @brief Base class for transformer pipelines
 *
 * Provides common infrastructure for all model architectures:
 * - MPI context management
 * - Device placement
 * - Weight and activation management
 * - Common pipeline operations
 *
 * Derived classes implement architecture-specific details:
 * - Qwen2Pipeline, Qwen3Pipeline, Qwen3MoEPipeline, etc.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../inference/IInferenceRunner.h"
#include "../utils/MPIContext.h"
#include "../backends/ComputeBackend.h"
#include "../loaders/ModelContext.h"
#include "../loaders/WeightPlacementMap.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/TensorKernels.h"
#include "../tensors/UnifiedKVCache.h" // Unified KV cache for single-sequence and batched decode
#include "PipelineConfig.h"            // Runtime configuration
#include "MPIStrategy.h"               // MPI parallelization strategies
#include "PipelineExecutor.h"          // Multi-device executor framework
#include "ops/Ops.h"                   // Self-validating operations
#include <vector>
#include <memory>
#include <string>
#include <map>

/**
 * @brief Validate pointer is non-null, log error and return false if null
 *
 * Usage: VALIDATE_POINTER(ptr, "description")
 *
 * Expands to:
 *   if (!ptr) {
 *       LOG_ERROR("Failed to load description");
 *       return false;
 *   }
 *
 * @param ptr Pointer to validate
 * @param desc Description for error message (will be prefixed with "Failed to load ")
 */
#define VALIDATE_POINTER(ptr, desc)                 \
    do                                              \
    {                                               \
        if (!(ptr))                                 \
        {                                           \
            LOG_ERROR("Failed to load " << (desc)); \
            return false;                           \
        }                                           \
    } while (0)

/**
 * @brief Validate pointer and create kernel, log error and return false if fails
 *
 * Usage: VALIDATE_KERNEL(kernel_var, tensor_ptr->createKernel(), "kernel type")
 *
 * Expands to:
 *   auto kernel_var = tensor_ptr->createKernel();
 *   if (!kernel_var) {
 *       LOG_ERROR("Failed to create kernel type");
 *       return false;
 *   }
 *
 * @param var Variable name to assign kernel to
 * @param expr Expression that creates the kernel
 * @param desc Description for error message (will be prefixed with "Failed to create ")
 */
#define VALIDATE_KERNEL(var, expr, desc)              \
    auto var = (expr);                                \
    do                                                \
    {                                                 \
        if (!(var))                                   \
        {                                             \
            LOG_ERROR("Failed to create " << (desc)); \
            return false;                             \
        }                                             \
    } while (0)

/**
 * @brief Validate operation succeeded, log error and return false if failed
 *
 * Usage: VALIDATE_OP(some_operation(), "operation description")
 *
 * Expands to:
 *   if (!some_operation()) {
 *       LOG_ERROR("operation description failed");
 *       return false;
 *   }
 *
 * @param expr Boolean expression or function call to validate
 * @param desc Description for error message (will be suffixed with " failed")
 */
#define VALIDATE_OP(expr, desc)             \
    do                                      \
    {                                       \
        if (!(expr))                        \
        {                                   \
            LOG_ERROR((desc) << " failed"); \
            return false;                   \
        }                                   \
    } while (0)

/**
 * @brief Check numerical health of activation tensor (Debug builds only)
 *
 * Detects numerical instability issues:
 * - Exploding activations (max > 1e3)
 * - Collapsing distributions (mean < 1e-5)
 * - Poor dynamic range (max/mean ratio)
 *
 * Compiles to no-op in Release builds for zero overhead.
 * Uses vectorized AVX512/AVX2 for minimal Debug overhead.
 *
 * Usage: CHECK_NUMERICAL_HEALTH("layer5_FFN_output", tensor_ptr->data(), 9 * 896)
 *
 * @param stage_name Human-readable stage identifier for logging
 * @param data Pointer to FP32 activation buffer
 * @param len Number of elements in buffer
 */
#ifndef NDEBUG
#define CHECK_NUMERICAL_HEALTH(stage_name, data, len) \
    check_numerical_health_impl((stage_name), (data), (len))
#else
#define CHECK_NUMERICAL_HEALTH(stage_name, data, len) \
    do                                                \
    {                                                 \
        (void)(stage_name);                           \
        (void)(data);                                 \
        (void)(len);                                  \
    } while (0)
#endif

/**
 * @brief Capture tensor snapshot for E2E parity testing
 *
 * Captures the full tensor contents for later comparison with PyTorch reference.
 * Only active when ENABLE_PIPELINE_SNAPSHOTS is defined (E2E/Debug builds).
 * Compiles to no-op in Release builds for zero overhead.
 *
 * Also triggers tensor dump to disk if LLAMINAR_SNAPSHOT_TENSOR_DUMP is enabled.
 *
 * Usage: CAPTURE_SNAPSHOT("layer5_ATTENTION_CONTEXT", buffers.attn_output.get())
 *
 * @param key Human-readable snapshot identifier
 * @param tensor_ptr Pointer to TensorBase with fp32_data() and shape() methods
 */
#ifdef ENABLE_PIPELINE_SNAPSHOTS
#define CAPTURE_SNAPSHOT(key, tensor_ptr)                                                         \
    do                                                                                            \
    {                                                                                             \
        const auto &_shape = (tensor_ptr)->shape();                                               \
        size_t _numel = 1;                                                                        \
        for (auto _dim : _shape)                                                                  \
            _numel *= _dim;                                                                       \
        captureSnapshot((key), (tensor_ptr)->fp32_data(), _numel);                                \
        maybeDumpTensor((key), (tensor_ptr), static_cast<int>(_shape.size() > 0 ? _shape[0] : 1), \
                        static_cast<int>(_shape.size() > 1 ? _shape[1] : _numel));                \
    } while (0)

/**
 * @brief Capture tensor snapshot with view for buffers larger than actual data
 *
 * Creates a view of the specified dimensions before capturing, useful when
 * activation buffers are pre-allocated larger than the current sequence length.
 * Only active when ENABLE_PIPELINE_SNAPSHOTS is defined (E2E/Debug builds).
 *
 * Also triggers tensor dump to disk if LLAMINAR_SNAPSHOT_TENSOR_DUMP is enabled.
 *
 * Usage: CAPTURE_SNAPSHOT_VIEW("layer0_Q_PROJECTION", buffers.Q, seq_len, n_heads * head_dim)
 *
 * @param key Human-readable snapshot identifier
 * @param tensor_ptr Pointer to TensorBase with create_view() method
 * @param rows Number of rows (typically effective_seq_len)
 * @param cols Number of columns (typically feature dimension)
 */
#define CAPTURE_SNAPSHOT_VIEW(key, tensor_ptr, rows, cols)                                              \
    do                                                                                                  \
    {                                                                                                   \
        auto _view = (tensor_ptr)->create_view({static_cast<size_t>(rows), static_cast<size_t>(cols)}); \
        if (_view)                                                                                      \
        {                                                                                               \
            const auto &_shape = _view->shape();                                                        \
            size_t _numel = 1;                                                                          \
            for (auto _dim : _shape)                                                                    \
                _numel *= _dim;                                                                         \
            captureSnapshot((key), _view->fp32_data(), _numel);                                         \
        }                                                                                               \
        maybeDumpTensor((key), (tensor_ptr), (rows), (cols));                                           \
    } while (0)
#else
// Without ENABLE_PIPELINE_SNAPSHOTS, we still support tensor dumps for production debugging
#define CAPTURE_SNAPSHOT(key, tensor_ptr)                                     \
    do                                                                        \
    {                                                                         \
        const auto &_shape = (tensor_ptr)->shape();                           \
        maybeDumpTensor((key), (tensor_ptr),                                  \
                        static_cast<int>(_shape.size() > 0 ? _shape[0] : 1),  \
                        static_cast<int>(_shape.size() > 1 ? _shape[1] : 1)); \
    } while (0)
#define CAPTURE_SNAPSHOT_VIEW(key, tensor_ptr, rows, cols)    \
    do                                                        \
    {                                                         \
        maybeDumpTensor((key), (tensor_ptr), (rows), (cols)); \
    } while (0)
#endif

namespace llaminar2
{
    // Forward declarations
#ifndef NDEBUG
    /**
     * @brief Vectorized numerical health check implementation (Debug only)
     *
     * Detects numerical drift in activation tensors.
     * See CHECK_NUMERICAL_HEALTH macro for usage.
     *
     * @param stage_name Human-readable identifier
     * @param data FP32 activation buffer
     * @param len Number of elements
     * @return Dynamic range (max / mean), or 0 if unhealthy
     */
    float check_numerical_health_impl(const char *stage_name,
                                      const float *data,
                                      size_t len);
#endif

    /**
     * @brief Pre-allocated activation buffers for inference
     *
     * Generic buffer structure used by all transformer pipelines.
     * Dimensions are architecture-specific (set by derived class).
     * Reused across all layers and forward passes to avoid allocations in hot path.
     */
    struct ActivationBuffers
    {
        // Residual connections (precision set by config_.activation_precision)
        std::shared_ptr<TensorBase> residual;

        // Normalization buffer (reused by attention and FFN blocks)
        std::shared_ptr<TensorBase> normalized;

        // Attention buffers
        std::shared_ptr<TensorBase> Q;
        std::shared_ptr<TensorBase> K;
        std::shared_ptr<TensorBase> V;
        std::shared_ptr<TensorBase> attn_output;
        std::shared_ptr<TensorBase> attn_proj;

        // FFN buffers
        std::shared_ptr<TensorBase> gate;
        std::shared_ptr<TensorBase> up;
        std::shared_ptr<TensorBase> ffn_output;

        int max_seq_len = 0; // Maximum sequence length these buffers support
    };

    /**
     * @brief Base class for transformer pipelines
     *
     * Provides common infrastructure for model execution.
     * Derived classes implement architecture-specific logic.
     *
     * Implements IInferenceRunner interface for unified execution.
     */
    class PipelineBase : public IInferenceRunner
    {
    public:
        /**
         * @brief Construct pipeline with MPI context and device placement
         *
         * @param model_ctx Model context with GGUF metadata and weights
         * @param mpi_ctx MPI context for distributed execution (nullptr = single-rank)
         * @param device_idx Default device index (-1 = CPU, ≥0 = GPU)
         * @param placement_map Optional weight placement strategy (nullptr = single device)
         * @param config Runtime configuration (max_seq_len, threading, etc.)
         */
        PipelineBase(std::shared_ptr<ModelContext> model_ctx,
                     std::shared_ptr<MPIContext> mpi_ctx,
                     int device_idx,
                     std::shared_ptr<WeightPlacementMap> placement_map,
                     const PipelineConfig &config = PipelineConfig{});
        virtual ~PipelineBase() = default;

        /**
         * @brief Forward pass (prefill or decode)
         *
         * @param tokens Token IDs [seq_len]
         * @param seq_len Number of tokens
         * @return true on success, false on error
         */
        bool forward(const int *tokens, int seq_len) override = 0;

        /**
         * @brief Get output logits (FP32)
         *
         * @return Logits tensor [seq_len, vocab_size], or nullptr if not available
         */
        const float *logits() const override;

        // ===== Snapshot Capture API (for parity testing / debugging) =====
        // Only available when ENABLE_PIPELINE_SNAPSHOTS is defined (test builds)
        // In release builds, the default no-op implementations from IInferenceRunner are used

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        /**
         * @brief Enable snapshot capture of intermediate activations
         *
         * When enabled, pipeline stores copies of intermediate tensors at key points
         * (e.g., after each attention block, FFN block, normalization).
         * Used for granular debugging and parity testing against reference implementations.
         *
         * NOTE: Only available in test builds (ENABLE_PIPELINE_SNAPSHOTS defined).
         * In release builds, uses IInferenceRunner's default no-op.
         *
         * @param output_dir Optional directory to save snapshots (empty = memory only)
         */
        void enableSnapshotCapture(const std::string &output_dir = "") override;

        /**
         * @brief Disable snapshot capture and clear stored snapshots
         *
         * NOTE: Only available in test builds (ENABLE_PIPELINE_SNAPSHOTS defined).
         */
        void disableSnapshotCapture() override;

        /**
         * @brief Clear stored snapshots but keep capture enabled
         *
         * Useful for incremental decode testing where you want to capture
         * each decode step separately without re-enabling capture.
         *
         * NOTE: Only available in test builds (ENABLE_PIPELINE_SNAPSHOTS defined).
         */
        void clearSnapshots() override;

        /**
         * @brief Retrieve a captured snapshot by key
         *
         * NOTE: Only available in test builds (ENABLE_PIPELINE_SNAPSHOTS defined).
         *
         * @param key Snapshot identifier (e.g., "layer_0_q_projection", "embedding")
         * @param out_size Output parameter for tensor size (number of float elements)
         * @return Pointer to snapshot data, or nullptr if key doesn't exist
         */
        const float *getSnapshot(const std::string &key, size_t &out_size) const override;

        /**
         * @brief Get list of all captured snapshot keys
         *
         * NOTE: Only available in test builds (ENABLE_PIPELINE_SNAPSHOTS defined).
         *
         * @return Vector of snapshot identifiers
         */
        std::vector<std::string> getSnapshotKeys() const override;

        /**
         * @brief Dump tensor data to disk for debugging
         *
         * Writes tensor data to files in the configured dump directory.
         * Supports both FP32 (dequantized) and native Q8_1 block formats.
         * Controlled by LLAMINAR_SNAPSHOT_TENSOR_DUMP and related env vars.
         *
         * Output files created:
         *   - <key>_fp32.bin: FP32 dequantized data
         *   - <key>_q8_1_blocks.bin: Raw Q8_1 blocks (if tensor is Q8_1)
         *   - <key>_metadata.txt: Shape, type, and scale information
         *
         * @param key Snapshot key (e.g., "layer21_FFN_RESIDUAL")
         * @param tensor Tensor to dump
         * @param rows Effective rows (sequence length)
         * @param cols Effective columns (hidden dimension)
         */
        void dumpTensorToDisk(const std::string &key, TensorBase *tensor, int rows, int cols);
#endif // ENABLE_PIPELINE_SNAPSHOTS

        // ===== Tensor Dump API (always available, controlled by debugEnv) =====
        // Unlike snapshots which require ENABLE_PIPELINE_SNAPSHOTS, tensor dumps
        // can be triggered in any build via environment variables.
        // This allows debugging production builds without recompilation.

        /**
         * @brief Check if tensor dump should occur for this key and dump if so
         *
         * Uses debugEnv().snapshot configuration to determine if this tensor
         * should be dumped. If tensor dump is enabled and the key matches
         * the configured layers/stages, dumps the tensor to disk.
         *
         * This is the primary entry point for the tensor dump feature.
         * Called automatically by capture_snapshot() when appropriate.
         *
         * @param key Snapshot key (e.g., "layer21_FFN_RESIDUAL")
         * @param tensor Tensor to potentially dump
         * @param rows Effective rows (sequence length)
         * @param cols Effective columns (hidden dimension)
         */
        void maybeDumpTensor(const std::string &key, TensorBase *tensor, int rows, int cols);

        /**
         * @brief Get model architecture name
         *
         * @return Architecture string (e.g., "qwen2", "qwen3", "qwen3-moe")
         */
        virtual const char *architecture() const override = 0;

        /**
         * @brief Get model context
         *
         * @return Model context with metadata and loader
         */
        std::shared_ptr<ModelContext> model_context() const { return model_ctx_; }

        /**
         * @brief Get MPI context
         *
         * @return MPI context pointer, or nullptr if not using MPI
         */
        std::shared_ptr<MPIContext> mpi_context() const { return mpi_ctx_; }

        /**
         * @brief Get default device index
         *
         * @return Device index (-1 = CPU, ≥0 = GPU device)
         */
        int device_index() const { return device_idx_; }

        /**
         * @brief Clear KV cache and reset position counters
         *
         * Call this to reset the pipeline state for a new inference run.
         * Used by benchmark mode to ensure consistent timing across iterations.
         *
         * Default implementation is a no-op. Derived classes should override
         * if they maintain cached state (e.g., KV cache for attention).
         */
        void clear_cache() override {}

        /**
         * @brief Get vocabulary size
         *
         * @return Number of tokens in vocabulary
         */
        int vocab_size() const override { return vocab_size_; }

        /**
         * @brief Get current cache position
         *
         * For autoregressive generation, this returns the position offset
         * for the next token (i.e., how many tokens have been processed).
         *
         * @return Current position in the sequence
         */
        int get_position() const override { return 0; }

        /**
         * @brief Get execution path (always PIPELINE for PipelineBase)
         */
        ExecutionPath executionPath() const override { return ExecutionPath::PIPELINE; }

    protected:
        // Context management
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        int device_idx_;        // Primary device (-1 = CPU, ≥0 = GPU index)
        PipelineConfig config_; // Runtime configuration (max_seq_len, etc.)

        /**
         * @brief Tensor factory for creating activation and residual tensors
         *
         * Provides NUMA-aware allocation with proper device placement.
         * All tensor creation in pipelines should go through this factory.
         * Initialized in PipelineBase constructor.
         */
        std::unique_ptr<TensorFactory> tensor_factory_;

        /**
         * @brief Default MPI context for single-rank (non-MPI) operation
         *
         * Created when mpi_ctx_ is null, allows TensorFactory to work without
         * explicit MPI initialization. Uses MPI_COMM_SELF for isolation.
         */
        std::unique_ptr<MPIContext> default_mpi_ctx_;

        // Model path for convenience (from model_ctx_)
        std::string model_path_;

        // Common model parameters (set by derived classes)
        int n_layers_ = 0;
        int d_model_ = 0;
        int n_heads_ = 0;    // Number of attention heads (needed for MPI validation)
        int n_kv_heads_ = 0; // Number of KV heads (GQA: ≤ n_heads)
        int head_dim_ = 0;   // Dimension per attention head
        int vocab_size_ = 0;

        // Common activations (used by all pipelines)
        std::shared_ptr<FP32Tensor> logits_; // [seq_len, vocab_size] output logits

        // ===== MPI Parallelization Infrastructure =====

        /**
         * @brief MPI configuration for distributed execution
         */
        MPIConfig mpi_config_;

        /**
         * @brief Active MPI strategy (selected in constructor)
         */
        MPIStrategy mpi_strategy_ = MPIStrategy::None;

        // ===== Multi-Device Infrastructure (Phase 4) =====

        /**
         * @brief Weight placement map (Phase 4.1)
         *
         * Maps weight names to device IDs. Enables heterogeneous execution
         * (e.g., layer 0-11 on GPU:0, layer 12-23 on CPU).
         */
        std::shared_ptr<WeightPlacementMap> placement_map_;

        /**
         * @brief Active devices for this rank (Phase 4.1)
         *
         * List of devices that have at least one weight placed on them.
         * Discovered during initialization by scanning placement_map_.
         */
        std::vector<int> active_devices_;

        /**
         * @brief Per-device activation buffer pools (Phase 4.1)
         *
         * Each device has separate Q/K/V/FFN buffers to enable parallel execution.
         * Buffers are lazily allocated on first use.
         */
        std::map<int, ActivationBuffers> buffers_per_device_;

        /**
         * @brief Attention workspace buffers (Phase 4.2 - zero-allocation hot path)
         *
         * Pre-allocated reusable buffers for attention computation.
         * Eliminates per-call allocations in MpiAttentionOrchestrator hot path.
         * Sized for max_seq_len during initializeDeviceInfrastructure().
         */
        std::shared_ptr<TensorBase> attention_workspace_scores_;     // [max_heads * max_seq, max_seq]
        std::shared_ptr<TensorBase> attention_workspace_qkv_buffer_; // [max_threads * max_seq * head_dim * 3]
        std::shared_ptr<TensorBase> attention_workspace_context_;    // [max_threads * max_seq * head_dim]
        std::shared_ptr<TensorBase> attention_workspace_mask_;       // [max_seq * max_seq]

        /**
         * @brief Legacy activation buffers (single-device mode)
         *
         * Deprecated: Use buffers_per_device_ for multi-device support.
         * Points to buffers_per_device_[device_idx_] for backward compat.
         * Initialized by initializeDeviceInfrastructure().
         */
        ActivationBuffers activation_buffers_;

        /**
         * @brief Unified KV cache for autoregressive decode
         *
         * Stores past key/value projections for incremental generation.
         * Supports both single-sequence (batch_size=1) and batched modes.
         * Initialized by initializeKVCache() with per-layer device placement.
         *
         * December 2025: Unified cache replaces separate IKVCache and BatchedKVCache.
         * Cache precision matches pipeline's ActivationPrecision by default.
         */
        std::unique_ptr<IUnifiedKVCache> kv_cache_;

        /**
         * @brief Current position per sequence (for incremental decode, batch-aware)
         *
         * Tracks how many tokens have been processed for each sequence in batch.
         * Size equals batch_size (resized in forward_batch()).
         * Reset to 0 on clear(), incremented per-sequence by forward().
         */
        std::vector<int> current_positions_;

        /**
         * @brief Get all weight names for device discovery (architecture-specific)
         *
         * Derived classes return all weight names in their architecture.
         * Used by discoverActiveDevices() to scan placement map.
         *
         * Example (Qwen2):
         *   ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight", ...]
         *
         * @return Vector of all weight names in GGUF format
         */
        virtual std::vector<std::string> getAllWeightNames() const = 0;

        /**
         * @brief Create activation buffers for a device (architecture-specific)
         *
         * Derived classes allocate buffers with architecture-specific dimensions.
         * Qwen2: uses n_heads_, n_kv_heads_, head_dim_, d_ff_
         * LLaMA: may use different dimension formulas
         *
         * @param device_idx Device to allocate buffers on (-1=CPU, >=0=GPU)
         * @param max_seq_len Maximum sequence length to support
         * @return Allocated buffer structure
         */
        virtual ActivationBuffers createBuffersForDevice(int device_idx, int max_seq_len) = 0;

        /**
         * @brief Discover which devices have weights placed on them (Phase 4.1)
         *
         * Scans all weight names (from getAllWeightNames()) and queries placement_map_.
         * Returns sorted list of unique device IDs.
         *
         * @return Vector of device IDs that have at least one weight
         */
        std::vector<int> discoverActiveDevices();

        /**
         * @brief Get activation buffers for a specific device (Phase 4.1)
         *
         * Returns buffer pool for the given device. If buffers don't exist yet,
         * they are lazily allocated via createBuffersForDevice().
         *
         * @param device_idx Device ID (-1=CPU, >=0=GPU)
         * @return Reference to buffer pool for device
         * @throws std::runtime_error if device is not in active_devices_
         */
        ActivationBuffers &getBuffersForDevice(int device_idx);

        /**
         * @brief Get device ID for a weight (Phase 4.1)
         *
         * Queries placement_map_ for the device holding this weight.
         *
         * @param weight_name Weight name in GGUF format (e.g., "blk.5.attn_q.weight")
         * @param layer_idx Layer index for layer-based lookup (-1 for non-layer weights)
         * @return Device ID (-1=CPU, >=0=GPU)
         */
        int getWeightDevice(const std::string &weight_name, int layer_idx = -1) const;

        // ===== Phase 3: MoE Device Placement Helpers =====

        /**
         * @brief Detect attention device for each layer from placement map (Phase 3)
         *
         * Queries placement_map_->getAttentionDevice() for each layer to build
         * a vector suitable for KVCache initialization.
         *
         * Uses block-level methods from Phase 2 (WeightPlacementMap enhancement).
         * KV cache resides where attention computation happens (Q/K/V weights).
         *
         * @param n_layers Number of transformer layers
         * @return Vector of device IDs, one per layer [n_layers]
         *
         * Example:
         *   auto attn_devices = detectAttentionDevices(24);
         *   kv_cache = make_shared<KVCache>(24, max_seq_len, ..., attn_devices);
         */
        std::vector<int> detectAttentionDevices(int n_layers) const;

        /**
         * @brief Detect FFN device for each layer from placement map (Phase 3)
         *
         * Queries placement_map_->getFFNDevice() for each layer.
         * Useful for heterogeneous execution where FFN may be on different device than attention.
         *
         * @param n_layers Number of transformer layers
         * @return Vector of device IDs, one per layer [n_layers]
         */
        std::vector<int> detectFFNDevices(int n_layers) const;

        // ===== Generic Initialization (extracted from Qwen2Pipeline) =====

        /**
         * @brief Initialize device infrastructure (Phase 1)
         *
         * Generic device initialization extracted from derived class constructors:
         * 1. Discover active devices via placement map
         * 2. Allocate activation buffers for each device
         * 3. Handle single-device vs multi-device modes
         *
         * Must be called AFTER architecture parameters (n_layers_, d_model_, etc.) are set.
         * Derived class must implement createBuffersForDevice() for buffer allocation.
         *
         * @param max_seq_len Maximum sequence length to support (e.g., 2048)
         * @param batch_size Maximum batch size to support (default: 1)
         */
        void initializeDeviceInfrastructure(int max_seq_len, int batch_size = 1);

        /**
         * @brief Configure MPI strategy (Phase 2)
         *
         * Generic MPI strategy configuration extracted from derived class constructors:
         * 1. Get default MPI config
         * 2. Auto-select or validate user strategy
         * 3. Log strategy and distribution info
         *
         * Must be called AFTER architecture parameters are set (for validation).
         * Derived class can override logMPIStrategyInfo() for custom logging.
         */
        void configureMPIStrategy();

        /**
         * @brief Initialize KV cache (Phase 3)
         *
         * Generic KV cache initialization extracted from derived class constructors:
         * 1. Detect attention devices per layer
         * 2. Create KVCache with appropriate device placement
         *
         * Must be called AFTER n_layers_, n_kv_heads_, head_dim_ are set.
         *
         * @param max_seq_len Maximum sequence length for KV cache (e.g., 2048)
         * @param batch_size Number of sequences to support (default: 1 for single-sequence)
         */
        void initializeKVCache(int max_seq_len, int batch_size = 1);

        /**
         * @brief Complete pipeline infrastructure initialization
         *
         * Performs all generic initialization steps in sequence:
         * 1. Device infrastructure (discovery, buffer allocation)
         * 2. MPI strategy configuration
         * 3. KV cache initialization
         *
         * Derived classes should call this as the LAST step in their constructor,
         * after setting all architecture parameters (n_layers_, d_model_, n_heads_,
         * n_kv_heads_, head_dim_, etc.).
         *
         * Example usage in derived constructor:
         *   Qwen2Pipeline(...) : PipelineBase(...) {
         *       // 1. Read architecture from GGUF metadata
         *       n_layers_ = model.block_count;
         *       n_heads_ = model.head_count;
         *       // ... etc
         *
         *       // 2. Derived-specific setup (weight vectors, etc)
         *       layers_.resize(n_layers_);
         *
         *       // 3. Call generic initialization (LAST step)
         *       initializeInfrastructure();
         *   }
         */
        void initializeInfrastructure();

        /**
         * @brief Log MPI strategy info (override for custom logging)
         *
         * Called by configureMPIStrategy() to log strategy-specific info.
         * Default implementation logs tensor-parallel head distribution.
         * Derived classes can override for architecture-specific logging.
         */
        virtual void logMPIStrategyInfo();

        /**
         * @brief Prepare activation for execution on target device (Phase 4.3)
         *
         * Smart transfer logic:
         * - If activation already on target device → return as-is (fast path)
         * - Otherwise → transfer via device-specific residual buffer (staging area)
         *
         * Enables heterogeneous execution (e.g., attention on GPU, FFN on CPU).
         *
         * @param activation Input activation tensor
         * @param target_device Device where operation will execute (-1=CPU, >=0=GPU)
         * @param context Description for logging (e.g., "attention_L5")
         * @return Pointer to activation on target device (may be same or different from input)
         */
        TensorBase *prepareActivationForDevice(TensorBase *activation, int target_device, const std::string &context);

        /**
         * @brief Process a single transformer layer
         *
         * To be implemented by derived classes.
         *
         * @param layer_idx Layer index (0-indexed)
         * @param seq_len Sequence length
         * @return true on success, false on error
         */
        virtual bool transformer_layer(int layer_idx, int seq_len) = 0;

        /**
         * @brief Standard GQA (Grouped Query Attention) orchestration
         *
         * Convenience wrapper around MpiAttentionOrchestrator::compute() for use in pipelines.
         *
         * Handles ~95% of production models (Qwen, Llama, Mistral, Gemma, etc.).
         * Pipelines with custom attention (e.g., DeepSeek MLA) override attention_block().
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim]
         * @param K Key tensor [seq_len, n_kv_heads * head_dim]
         * @param V Value tensor [seq_len, n_kv_heads * head_dim]
         * @param output Output tensor [seq_len, n_heads * head_dim] (pre-allocated)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (GQA: ≤ n_heads)
         * @param head_dim Dimension per head
         * @param causal Apply causal masking for autoregressive generation
         * @param window_size Sliding window size (-1 = full attention, ≥0 = local window)
         * @param batch_size Number of sequences in batch (default=1 for single sequence)
         * @param sequence_lengths Actual lengths per sequence for padding mask (nullptr = no padding)
         * @return true on success, false on error
         *
         * @note Delegates to MpiAttentionOrchestrator::compute() (see pipelines/attention/MpiAttentionOrchestrator.h)
         * @note For MPI parallelization, use attention_gqa_mpi()
         * @deprecated Use AttentionWithKVCacheStage or CPUAttentionKernelTyped via KernelFactory for new code
         */
        [[deprecated("Use AttentionWithKVCacheStage or KernelFactory::createAttention() for new code")]]
        virtual bool attention_gqa(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal = true, int window_size = -1,
            int batch_size = 1, const std::vector<int> *sequence_lengths = nullptr);

        /**
         * @brief Batched grouped-query attention (GQA) with padding support
         *
         * Convenience wrapper around MpiAttentionOrchestrator::compute_batch() for use in pipelines.
         *
         * @param Q Query tensor for all batches (flattened)
         * @param K Key tensor for all batches (flattened)
         * @param V Value tensor for all batches (flattened)
         * @param output Output tensor for all batches (pre-allocated, flattened)
         * @param actual_lengths Actual sequence lengths (before padding) [batch_size]
         * @param batch_size Number of sequences
         * @param seq_len Maximum sequence length (after padding)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         * @param causal Apply causal masking
         * @param window_size Sliding window size
         * @return true on success, false on error
         *
         * @note Delegates to MpiAttentionOrchestrator::compute_batch() (see pipelines/attention/MpiAttentionOrchestrator.h)
         * @deprecated Use AttentionWithKVCacheStage or CPUAttentionKernelTyped via KernelFactory for new code
         */
        [[deprecated("Use AttentionWithKVCacheStage or KernelFactory::createAttention() for new code")]]
        virtual bool attention_gqa_batch(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            const std::vector<int> &actual_lengths,
            int batch_size, int seq_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal = true, int window_size = -1);

        // ===== MPI-Aware Attention Methods =====

        /**
         * @brief MPI-aware attention dispatcher
         *
         * Convenience wrapper around MpiAttentionOrchestrator::compute_mpi() for use in pipelines.
         * Dispatches to appropriate implementation based on MPI strategy.
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim]
         * @param K Key tensor [seq_len, n_kv_heads * head_dim]
         * @param V Value tensor [seq_len, n_kv_heads * head_dim]
         * @param output Output tensor [seq_len, n_heads * head_dim] (pre-allocated)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         * @param causal Apply causal masking
         * @param window_size Sliding window size
         * @param batch_size Number of sequences in batch (default=1 for single sequence)
         * @param sequence_lengths Actual lengths per sequence for padding mask (nullptr = no padding)
         * @return true on success, false on error
         *
         * @note Delegates to MpiAttentionOrchestrator::compute_mpi() (see pipelines/attention/MpiAttentionOrchestrator.h)
         * @deprecated Use AttentionWithKVCacheStage or CPUAttentionKernelTyped via KernelFactory for new code
         */
        [[deprecated("Use AttentionWithKVCacheStage or KernelFactory::createAttention() for new code")]]
        bool attention_gqa_mpi(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal = true, int window_size = -1,
            int batch_size = 1, const std::vector<int> *sequence_lengths = nullptr);

        /**
         * @brief Tensor-parallel attention implementation
         *
         * Convenience wrapper around MpiAttentionOrchestrator::compute_tensor_parallel() for use in pipelines.
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim]
         * @param K Key tensor [seq_len, n_kv_heads * head_dim]
         * @param V Value tensor [seq_len, n_kv_heads * head_dim]
         * @param output Output tensor [seq_len, n_heads * head_dim] (pre-allocated)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         * @param causal Apply causal masking
         * @param window_size Sliding window size
         * @param batch_size Number of sequences in batch (default=1)
         * @param sequence_lengths Actual lengths per sequence for padding mask (nullptr = no padding)
         * @return true on success, false on error
         *
         * @note Delegates to MpiAttentionOrchestrator::compute_tensor_parallel() (see pipelines/attention/MpiAttentionOrchestrator.h)
         * @deprecated Use AttentionWithKVCacheStage or CPUAttentionKernelTyped via KernelFactory for new code
         */
        [[deprecated("Use AttentionWithKVCacheStage or KernelFactory::createAttention() for new code")]]
        bool attention_gqa_tensor_parallel(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size,
            int batch_size = 1, const std::vector<int> *sequence_lengths = nullptr);

        // ===== MPI Strategy Management =====

        /**
         * @brief Select optimal MPI strategy based on model architecture
         *
         * Heuristic:
         * 1. Try TensorParallel if n_heads % world_size == 0
         * 2. Try PipelineParallel if n_layers % world_size == 0
         * 3. Fallback to None (warn user)
         *
         * Called during pipeline construction when auto_select=true.
         *
         * @return Selected strategy
         */
        MPIStrategy selectOptimalStrategy();

        /**
         * @brief Validate that strategy is compatible with model
         *
         * Checks:
         * - TensorParallel: n_heads % world_size == 0, d_ff % world_size == 0
         * - PipelineParallel: n_layers % world_size == 0
         * - SequenceParallel: Always valid (can split any sequence)
         *
         * @param strategy Strategy to validate
         * @return true if valid, false otherwise
         */
        bool validateStrategy(MPIStrategy strategy);

        // ===== MPI Distribution Helpers =====

        /**
         * @brief Get attention head distribution for this rank
         *
         * Divides n_heads across world_size ranks as evenly as possible.
         *
         * @param n_heads Total number of attention heads
         * @return {start_head, local_n_heads} for this rank
         */
        std::pair<size_t, size_t> getHeadDistribution(int n_heads);

        /**
         * @brief Get layer distribution for pipeline parallelism
         *
         * Divides n_layers across world_size ranks.
         *
         * @param n_layers Total number of layers
         * @return {start_layer, local_n_layers} for this rank
         */
        std::pair<size_t, size_t> getLayerDistribution(int n_layers);

        /**
         * @brief Get token distribution for sequence parallelism
         *
         * Divides seq_len across world_size ranks.
         *
         * @param seq_len Total sequence length
         * @return {start_token, local_seq_len} for this rank
         */
        std::pair<size_t, size_t> getTokenDistribution(int seq_len);

        // ===== Weight Device Orchestration (Phase 5) =====
        // Generic lazy GPU transfer helpers usable by all pipeline implementations

        /**
         * @brief Ensure a collection of weights are on the target device (lazy GPU transfer)
         *
         * Triggers lazy transfer of weights to GPU if target_device >= 0.
         * No-op if weights are already on target device or target is CPU (-1).
         * This is the generic building block for architecture-specific orchestration.
         *
         * @param weights Vector of weight tensors to transfer (nullptrs are skipped)
         * @param target_device Target device index (-1 = CPU, >=0 = GPU)
         * @param context Description for error logging (e.g., "attention_L5", "ffn_L3")
         * @return true if all transfers succeeded (or no transfers needed)
         */
        bool ensureWeightsOnDevice(
            const std::vector<std::shared_ptr<TensorBase>> &weights,
            int target_device,
            const std::string &context = "");

        /**
         * @brief Ensure a single weight tensor is on the target device
         *
         * Convenience overload for single tensor transfer.
         *
         * @param weight Weight tensor to transfer (nullptr = no-op)
         * @param target_device Target device index (-1 = CPU, >=0 = GPU)
         * @param weight_name Name for error logging
         * @return true if transfer succeeded (or no transfer needed)
         */
        bool ensureWeightOnDevice(
            const std::shared_ptr<TensorBase> &weight,
            int target_device,
            const std::string &weight_name = "");

        // ===== Snapshot Capture Helper (for derived classes) =====
        // Compiles to NOOP in release builds (ENABLE_PIPELINE_SNAPSHOTS not defined)

        /**
         * @brief Capture a snapshot of an intermediate activation
         *
         * Derived classes call this at instrumentation points (e.g., after Q projection).
         * If ENABLE_PIPELINE_SNAPSHOTS is defined AND capture_enabled_ is true, stores a copy.
         * Otherwise, compiles away to a NOOP (zero overhead in release builds).
         *
         * @param key Snapshot identifier (e.g., "layer_5_attention_scores")
         * @param data Pointer to tensor data (float*)
         * @param size Number of elements in tensor
         */
        inline void captureSnapshot([[maybe_unused]] const std::string &key,
                                    [[maybe_unused]] const float *data,
                                    [[maybe_unused]] size_t size)
        {
#ifdef ENABLE_PIPELINE_SNAPSHOTS
            if (snapshot_capture_enabled_)
            {
                snapshots_[key].assign(data, data + size);
            }
#endif
            // In release builds (no ENABLE_PIPELINE_SNAPSHOTS), this is a NOOP
            // Compiler optimizes away the entire function call
        }

        // =============================================================================
        // Declarative Compute Graph Operations
        // =============================================================================
        //
        // These high-level operations encapsulate the common patterns:
        //   - Device placement (from placement_map_)
        //   - MPI context (from mpi_ctx_)
        //   - Snapshot capture (CAPTURE_SNAPSHOT macros)
        //   - Validation (VALIDATE_TENSOR macros)
        //
        // Child pipelines chain these to form a declarative compute graph.
        // All complexity is pushed up to PipelineBase.
        //
        // Example usage in Qwen2Pipeline::attention_block():
        //   TRY_OP(rms_norm(input, gamma, output, seq_len, d_model, eps, "layer0_ATTENTION_NORM"));
        //   TRY_OP(project(normalized, W_q, Q, seq_len, q_dim, d_model, "layer0_Q_PROJECTION"));
        //   TRY_OP(add_residual(residual, attn_out, hidden, "layer0_ATTENTION_RESIDUAL"));

        /**
         * @brief RMSNorm with snapshot capture and validation
         *
         * @param input Input tensor
         * @param gamma Normalization weights [hidden_dim]
         * @param output Output tensor (can be same as input for in-place)
         * @param rows Number of rows (sequence length)
         * @param cols Number of columns (hidden dimension)
         * @param eps Epsilon for numerical stability
         * @param snapshot_key Snapshot identifier for parity testing
         * @param device Target device (-1 to use current device)
         * @return true on success
         */
        bool rms_norm(
            TensorBase *input, const TensorBase *gamma, TensorBase *output,
            int rows, int cols, float eps,
            const std::string &snapshot_key, int device = -1);

        /**
         * @brief Weight projection (GEMM) with snapshot capture and validation
         *
         * Computes: output = input @ weight^T
         *
         * @param input Activation tensor [m, k]
         * @param weight Weight tensor [n, k]
         * @param output Output tensor [m, n]
         * @param m Number of rows (sequence length)
         * @param n Output dimension
         * @param k Input dimension
         * @param snapshot_key Snapshot identifier for parity testing
         * @param device Target device (-1 to use current device)
         * @return true on success
         */
        bool project(
            const TensorBase *input, TensorBase *weight, TensorBase *output,
            int m, int n, int k,
            const std::string &snapshot_key, int device = -1);

        /**
         * @brief Row-parallel projection with MPI allreduce
         *
         * For tensor parallelism: In row-parallel projections (Wo, FFN Down),
         * each rank processes a partition of the input and produces a partial
         * contribution to the output. This method performs the local GEMM and
         * then allreduces to sum contributions across all ranks.
         *
         * NOTE: Currently, all ranks compute the full GEMM (no compute savings).
         * The allreduce ensures correctness for when we implement true row-parallel
         * GEMM where each rank only computes its partition.
         *
         * @param input Input tensor [m, k]
         * @param weight Weight tensor [n, k]
         * @param output Output tensor [m, n]
         * @param m Number of rows (sequence length)
         * @param n Output dimension
         * @param k Input dimension
         * @param snapshot_key Snapshot identifier for parity testing
         * @param device Target device (-1 to use current device)
         * @return true on success
         */
        bool project_row_parallel(
            const TensorBase *input, TensorBase *weight, TensorBase *output,
            int m, int n, int k,
            const std::string &snapshot_key, int device = -1);

        /**
         * @brief Column-parallel projection (for cascaded tensor parallelism)
         *
         * Used when input is from a column-parallel layer (e.g., FFN Gate/Up).
         * Each rank has input [m, k_local] and weight [n, k_local].
         * Computes local GEMM: [m, k_local] @ [n, k_local]^T = [m, n]
         * Then allreduce-sum to combine partial results.
         *
         * @param input Input tensor [m, k_local] (local slice)
         * @param weight Weight tensor [n, k_local] (column-parallel slice)
         * @param output Output tensor [m, n] (full, allreduced)
         * @param m Number of rows (sequence length)
         * @param n Output dimension
         * @param k_local Local input dimension (k_full / world_size)
         * @param snapshot_key Snapshot identifier for parity testing
         * @param device Target device (-1 to use current device)
         * @return true on success
         */
        bool project_column_parallel(
            const TensorBase *input, TensorBase *weight, TensorBase *output,
            int m, int n, int k_local,
            const std::string &snapshot_key, int device = -1);

        /**
         * @brief Residual connection with snapshot capture (batch-aware, padding-safe)
         *
         * Computes: output = residual + input (with padding zeroing)
         *
         * @param residual Saved residual tensor
         * @param input Current activation to add
         * @param output Output tensor
         * @param batch_size Number of sequences in batch
         * @param seq_len Padded sequence length
         * @param hidden_dim Hidden dimension
         * @param sequence_lengths Actual lengths per sequence (for padding)
         * @param snapshot_key Snapshot identifier for parity testing
         * @return true on success
         */
        bool add_residual(
            const TensorBase *residual, const TensorBase *input, TensorBase *output,
            int batch_size, int seq_len, int hidden_dim,
            const std::vector<int> &sequence_lengths,
            const std::string &snapshot_key);

        /**
         * @brief SwiGLU activation with snapshot capture
         *
         * Computes: output = gate * silu(up)
         *
         * @param gate Gate projection output
         * @param up Up projection output (silu applied to this)
         * @param output Output tensor (can reuse up buffer)
         * @param rows Number of rows
         * @param cols Number of columns (d_ff)
         * @param snapshot_key Snapshot identifier for parity testing
         * @param device Target device
         * @return true on success
         */
        bool swiglu(
            TensorBase *gate, TensorBase *up, TensorBase *output,
            int rows, int cols,
            const std::string &snapshot_key, int device = -1);

        /**
         * @brief RoPE (Rotary Position Embedding) with snapshot capture
         *
         * Applies RoPE to Q and K tensors in-place.
         *
         * @param Q Query tensor (modified in-place)
         * @param K Key tensor (modified in-place)
         * @param position_ids Position IDs for each token
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param theta RoPE theta parameter
         * @param snapshot_prefix Prefix for snapshot keys (e.g., "layer0")
         * @param device Target device
         * @return true on success
         */
        bool apply_rope(
            TensorBase *Q, TensorBase *K,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float theta,
            const std::string &snapshot_prefix, int device = -1);

        /**
         * @brief Copy tensor data (for saving residual)
         *
         * @param src Source tensor
         * @param dst Destination tensor
         * @param num_elements Number of elements to copy
         * @return true on success
         */
        bool copy_tensor(const TensorBase *src, TensorBase *dst, size_t num_elements);

        /**
         * @brief Save residual for later addition
         *
         * Convenience wrapper for copy_tensor with common pattern.
         *
         * @param input Input tensor to copy from
         * @param residual_buffer Buffer to copy to
         * @param seq_len Sequence length
         * @param hidden_dim Hidden dimension
         * @return true on success
         */
        bool save_residual(const TensorBase *input, TensorBase *residual_buffer, int seq_len, int hidden_dim);

        /**
         * @brief GQA attention with view creation and snapshot capture
         *
         * Handles view creation, attention computation, and snapshot capture.
         * This is the high-level attention interface for pipelines.
         *
         * @param Q Query buffer [seq_len, n_heads * head_dim]
         * @param K Key buffer [seq_len, n_kv_heads * head_dim]
         * @param V Value buffer [seq_len, n_kv_heads * head_dim]
         * @param output Output buffer [seq_len, n_heads * head_dim]
         * @param seq_len Effective sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param batch_size Batch size
         * @param sequence_lengths Actual sequence lengths (for padding mask)
         * @param padded_seq_len Padded sequence length
         * @param causal Whether to use causal masking
         * @param snapshot_key Snapshot key for attention context output
         * @return true on success
         */
        bool compute_attention(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            int batch_size, const std::vector<int> &sequence_lengths, int padded_seq_len,
            bool causal, const std::string &snapshot_key);

        /**
         * @brief Compute attention with KV cache support (asymmetric Q/K/V lengths)
         *
         * Used for incremental decode where Q has length 1 (current token)
         * but K/V have full cached context length.
         *
         * @param Q Query tensor [q_seq_len, n_heads * head_dim]
         * @param K Key tensor [kv_seq_len, n_kv_heads * head_dim] (from KV cache)
         * @param V Value tensor [kv_seq_len, n_kv_heads * head_dim] (from KV cache)
         * @param output Output tensor [q_seq_len, n_heads * head_dim]
         * @param q_seq_len Query sequence length (1 for decode, prompt_len for prefill)
         * @param kv_seq_len Key/Value sequence length (total cached tokens)
         * @param n_heads Number of attention heads
         * @param n_kv_heads Number of KV heads (for GQA)
         * @param head_dim Head dimension
         * @param batch_size Batch size
         * @param sequence_lengths Actual sequence lengths (for padding mask)
         * @param causal Whether to use causal masking
         * @param snapshot_key Snapshot key for attention context output
         * @return true on success
         */
        bool compute_attention_with_kv_cache(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int q_seq_len, int kv_seq_len, int n_heads, int n_kv_heads, int head_dim,
            int batch_size, const std::vector<int> &sequence_lengths,
            bool causal, const std::string &snapshot_key);

        /**
         * @brief Capture a snapshot of a tensor view
         *
         * Helper for manual snapshot capture when needed.
         *
         * @param key Snapshot key
         * @param tensor Tensor to capture
         * @param rows Number of rows in view
         * @param cols Number of columns in view
         */
        void capture_snapshot(const std::string &key, TensorBase *tensor, int rows, int cols);

    private:
        // ===== Snapshot Storage (for parity testing / debugging) =====
        // Only exists in test builds (ENABLE_PIPELINE_SNAPSHOTS defined)

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        bool snapshot_capture_enabled_ = false;               // Whether to capture snapshots
        std::string snapshot_output_dir_;                     // Optional directory for saving
        std::map<std::string, std::vector<float>> snapshots_; // In-memory snapshot storage
#endif

        // ===== Typed Operations (zero-overhead precision dispatch) =====
        // Created once at initialization based on config_.activation_precision
        // These are used for Q8_1/BF16/FP16 native paths without runtime dispatch
        std::unique_ptr<IGemmOp> typed_gemm_op_;
        std::unique_ptr<IRMSNormOp> typed_rmsnorm_op_;
        std::unique_ptr<IResidualOp> typed_residual_op_;
        std::unique_ptr<ISwiGLUOp> typed_swiglu_op_;
        std::unique_ptr<IRoPEOp> typed_rope_op_;

        // ===== Multi-Device Execution Framework =====
        /**
         * @brief Optional PipelineExecutor for multi-device work distribution
         *
         * When config_.executor_* flags are enabled, operations use this
         * executor for hierarchical work distribution across devices.
         * Nullptr when all executor flags are disabled (default).
         */
        std::unique_ptr<PipelineExecutor> pipeline_executor_;

    protected:
        /**
         * @brief Initialize typed ops based on activation precision
         *
         * Called during pipeline construction after config_ is set.
         * Creates typed ops that match the configured activation precision.
         */
        void initializeTypedOps()
        {
            typed_gemm_op_ = createGemmOp(config_.activation_precision);
            typed_rmsnorm_op_ = createRMSNormOp(config_.activation_precision);
            typed_residual_op_ = createResidualOp(config_.activation_precision);
            typed_swiglu_op_ = createSwiGLUOp(config_.activation_precision);
            typed_rope_op_ = createRoPEOp(config_.activation_precision);

            LOG_DEBUG("[PipelineBase] Initialized typed ops for precision: "
                      << static_cast<int>(config_.activation_precision));

            // Initialize PipelineExecutor if any executor feature flag is enabled
            initializePipelineExecutor();
        }

        /**
         * @brief Initialize PipelineExecutor if any feature flags are enabled
         *
         * Creates PipelineExecutor only when needed (at least one executor_* flag set).
         * This lazy approach avoids overhead when using traditional execution path.
         */
        void initializePipelineExecutor()
        {
            // Check if any executor feature is enabled
            bool any_executor_enabled =
                config_.executor_ffn_norm ||
                config_.executor_ffn_swiglu ||
                config_.executor_ffn_residual ||
                config_.executor_attn_norm ||
                config_.executor_attn_residual ||
                config_.executor_rope;

            if (!any_executor_enabled)
            {
                LOG_DEBUG("[PipelineBase] No executor features enabled, skipping PipelineExecutor init");
                return;
            }

            // Create PipelineExecutor configuration mirroring pipeline flags
            PipelineExecutorConfig exec_config;
            exec_config.executor_ffn_norm = config_.executor_ffn_norm;
            exec_config.executor_ffn_swiglu = config_.executor_ffn_swiglu;
            exec_config.executor_ffn_residual = config_.executor_ffn_residual;
            exec_config.executor_attn_norm = config_.executor_attn_norm;
            exec_config.executor_attn_residual = config_.executor_attn_residual;
            exec_config.executor_rope = config_.executor_rope;

            // Create executor with MPI context
            // Use the raw pointer constructor when we have unique_ptr
            MPIContext *mpi_raw = mpi_ctx_ ? mpi_ctx_.get() : (default_mpi_ctx_ ? default_mpi_ctx_.get() : nullptr);
            pipeline_executor_ = std::make_unique<PipelineExecutor>(exec_config, mpi_raw);

            // Add primary device context
            pipeline_executor_->setDeviceContext(device_idx_);

            LOG_INFO("[PipelineBase] Initialized PipelineExecutor with device " << device_idx_);
            LOG_DEBUG("  executor_ffn_norm=" << config_.executor_ffn_norm);
            LOG_DEBUG("  executor_ffn_swiglu=" << config_.executor_ffn_swiglu);
            LOG_DEBUG("  executor_ffn_residual=" << config_.executor_ffn_residual);
            LOG_DEBUG("  executor_attn_norm=" << config_.executor_attn_norm);
            LOG_DEBUG("  executor_attn_residual=" << config_.executor_attn_residual);
            LOG_DEBUG("  executor_rope=" << config_.executor_rope);
        }

        /**
         * @brief Get PipelineExecutor (may be nullptr if no flags enabled)
         */
        PipelineExecutor *pipelineExecutor() const { return pipeline_executor_.get(); }
    };

} // namespace llaminar2
