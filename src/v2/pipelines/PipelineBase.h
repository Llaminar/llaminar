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

#include "../utils/MPIContext.h"
#include "../backends/ComputeBackend.h"
#include "../loaders/ModelContext.h"
#include "../loaders/WeightPlacementMap.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorKernels.h"
#include "../tensors/KVCache.h" // KV cache for autoregressive decode
#include "PipelineConfig.h"     // Runtime configuration
#include "MPIStrategy.h"        // MPI parallelization strategies
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

namespace llaminar2
{
    /**
     * @brief Pre-allocated activation buffers for inference
     *
     * Generic buffer structure used by all transformer pipelines.
     * Dimensions are architecture-specific (set by derived class).
     * Reused across all layers and forward passes to avoid allocations in hot path.
     */
    struct ActivationBuffers
    {
        // Residual connections
        std::shared_ptr<FP32Tensor> residual;

        // Normalization buffer (reused by attention and FFN blocks)
        std::shared_ptr<FP32Tensor> normalized;

        // Attention buffers
        std::shared_ptr<FP32Tensor> Q;
        std::shared_ptr<FP32Tensor> K;
        std::shared_ptr<FP32Tensor> V;
        std::shared_ptr<FP32Tensor> attn_output;
        std::shared_ptr<FP32Tensor> attn_proj;

        // FFN buffers
        std::shared_ptr<FP32Tensor> gate;
        std::shared_ptr<FP32Tensor> up;
        std::shared_ptr<FP32Tensor> ffn_output;

        int max_seq_len = 0; // Maximum sequence length these buffers support
    };

    /**
     * @brief Base class for transformer pipelines
     *
     * Provides common infrastructure for model execution.
     * Derived classes implement architecture-specific logic.
     */
    class PipelineBase
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
        virtual bool forward(const int *tokens, int seq_len) = 0;

        /**
         * @brief Get output logits (FP32)
         *
         * @return Logits tensor [seq_len, vocab_size], or nullptr if not available
         */
        virtual const float *logits() const;

        /**
         * @brief Get model architecture name
         *
         * @return Architecture string (e.g., "qwen2", "qwen3", "qwen3-moe")
         */
        virtual const char *architecture() const = 0;

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

    protected:
        // Context management
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        int device_idx_;        // Primary device (-1 = CPU, ≥0 = GPU index)
        PipelineConfig config_; // Runtime configuration (max_seq_len, etc.)

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
         * @brief Legacy activation buffers (single-device mode)
         *
         * Deprecated: Use buffers_per_device_ for multi-device support.
         * Points to buffers_per_device_[device_idx_] for backward compat.
         * Initialized by initializeDeviceInfrastructure().
         */
        ActivationBuffers activation_buffers_;

        /**
         * @brief KV cache for autoregressive decode (Phase 3)
         *
         * Stores past key/value projections for incremental generation.
         * Initialized by initializeKVCache() with per-layer device placement.
         * Per-layer device affinity enables heterogeneous execution.
         */
        std::shared_ptr<KVCache> kv_cache_;

        /**
         * @brief Current position in sequence (for incremental decode)
         *
         * Tracks how many tokens have been processed.
         * Reset to 0 on clear(), incremented by forward().
         */
        int current_position_ = 0;

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
         * @brief Initialize device infrastructure (Phase 4.1)
         *
         * Generic initialization logic extracted from derived class constructors:
         * 1. Discover active devices via placement map
         * 2. Allocate activation buffers for each device
         * 3. Handle single-device vs multi-device modes
         *
         * Must be called AFTER architecture parameters (n_layers_, d_model_, etc.) are set.
         * Derived class must implement createBuffersForDevice() for buffer allocation.
         *
         * @param max_seq_len Maximum sequence length to support (e.g., 2048)
         */
        void initializeDeviceInfrastructure(int max_seq_len);

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
         */
        void initializeKVCache(int max_seq_len);

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
         * Default attention implementation supporting:
         * - GQA: n_heads > n_kv_heads (broadcast K/V heads)
         * - MHA: n_heads == n_kv_heads (no broadcasting)
         * - MQA: n_kv_heads == 1 (broadcast single K/V to all Q heads)
         * - Sliding window: Optional local attention window
         *
         * Handles ~95% of production models (Qwen, Llama, Mistral, Gemma, etc.).
         * Pipelines with custom attention (e.g., DeepSeek MLA) override attention_block().
         *
         * Algorithm:
         * 1. Broadcast K/V heads to match Q heads (if n_kv_heads < n_heads)
         * 2. Compute attention scores: Q @ K^T (per-head batched GEMM)
         * 3. Scale by 1/sqrt(head_dim)
         * 4. Apply causal mask (optional sliding window)
         * 5. Softmax over scores
         * 6. Compute context: scores @ V (per-head batched GEMM)
         * 7. Concatenate heads back to [seq_len, n_heads * head_dim]
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim]
         * @param K Key tensor [seq_len, n_kv_heads * head_dim]
         * @param V Value tensor [seq_len, n_kv_heads * head_dim]
         * @param output Output tensor [seq_len, n_heads * head_dim] (pre-allocated)
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (GQA: ≤ n_heads)
         * @param head_dim Dimension per head
         * @param causal Apply causal masking for autoregressive generation
         * @param window_size Sliding window size (-1 = full attention, ≥0 = local window)
         * @return true on success, false on error
         *
         * @note Single-rank implementation (no MPI coordination)
         * @note For MPI parallelization, use attention_gqa_mpi()
         * @note Uses primitive kernels: ITensorGemm, ITensorSoftmax
         */
        virtual bool attention_gqa(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal = true, int window_size = -1);

        // ===== MPI-Aware Attention Methods =====

        /**
         * @brief MPI-aware attention dispatcher
         *
         * Dispatches to appropriate attention implementation based on MPI strategy:
         * - MPIStrategy::None → attention_gqa() (single-rank fast path)
         * - MPIStrategy::TensorParallel → attention_gqa_tensor_parallel()
         * - MPIStrategy::SequenceParallel → attention_gqa_sequence_parallel()
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
         * @return true on success, false on error
         */
        bool attention_gqa_mpi(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal = true, int window_size = -1);

        /**
         * @brief Tensor-parallel attention implementation
         *
         * Distributes attention heads across MPI ranks:
         * - Rank i computes heads [start_head, start_head + local_n_heads)
         * - Allreduce to sum outputs from all ranks
         *
         * Memory: O(seq_len * local_n_heads * head_dim) per rank
         * Communication: 1× allreduce (seq_len * n_heads * head_dim elements)
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
         * @return true on success, false on error
         *
         * @note Requires n_heads % world_size == 0 (validated in constructor)
         */
        bool attention_gqa_tensor_parallel(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size);

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
    };

} // namespace llaminar2
