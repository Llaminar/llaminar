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
#include <vector>
#include <memory>
#include <string>
#include <map>

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
         * @brief Construct pipeline base
         *
         * @param model_ctx Model context with GGUF metadata and loader
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Default device for tensors (-1 = CPU, ≥0 = GPU device)
         * @param placement_map Weight placement map (nullptr = create default with all on device_idx)
         */
        PipelineBase(std::shared_ptr<ModelContext> model_ctx,
                     std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                     int device_idx = -1,
                     std::shared_ptr<WeightPlacementMap> placement_map = nullptr);

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
        int device_idx_; // -1 = CPU, ≥0 = GPU device

        // Model path for convenience (from model_ctx_)
        std::string model_path_;

        // Common model parameters (set by derived classes)
        int n_layers_ = 0;
        int d_model_ = 0;
        int vocab_size_ = 0;

        // Common activations (used by all pipelines)
        std::shared_ptr<FP32Tensor> logits_; // [seq_len, vocab_size] output logits

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
         * @note Uses primitive kernels: ITensorGemm, ITensorSoftmax
         * @note Pipelines can override attention_block() for custom attention types
         */
        virtual bool attention_gqa(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal = true, int window_size = -1);
    };

} // namespace llaminar2
