/**
 * @file PipelineExecutor.h
 * @brief Adapter to integrate LayerExecutor with existing pipeline infrastructure
 * @author David Sanftenberg
 * @date December 2025
 *
 * PipelineExecutor bridges the gap between the new execution framework
 * (LayerExecutor, ComputeStage, DeviceContext) and the existing pipeline
 * infrastructure (FusedGEMM, Q8_1 activations, KV cache, etc.).
 *
 * Design goals:
 * 1. Incremental adoption - can be enabled per-layer or per-block
 * 2. Minimal disruption - uses existing weight/buffer structures
 * 3. Performance parity - same kernels, just different orchestration
 * 4. Observability - integrates with existing snapshot/profiling
 */

#pragma once

#include "../execution/LayerExecutor.h"
#include "../execution/DeviceContext.h"
#include "../execution/WorkDistributor.h"
#include "../tensors/Tensors.h"
#include "../utils/MPIContext.h"
#include <memory>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class PipelineBase;
    class FusedGEMM;
    struct ActivationBuffers;

    /**
     * @brief Configuration for PipelineExecutor
     */
    struct PipelineExecutorConfig
    {
        bool use_layer_executor = false; ///< Enable LayerExecutor for supported ops
        bool use_compute_graph = false;  ///< Use ComputeGraph for dependency tracking
        ExecutionMode execution_mode = ExecutionMode::SEQUENTIAL;
        bool enable_profiling = false;

        // Feature flags for incremental rollout
        bool executor_ffn_norm = false;      ///< Use executor for FFN norm
        bool executor_ffn_swiglu = false;    ///< Use executor for SwiGLU
        bool executor_ffn_residual = false;  ///< Use executor for FFN residual
        bool executor_attn_norm = false;     ///< Use executor for attention norm
        bool executor_attn_residual = false; ///< Use executor for attention residual
        bool executor_rope = false;          ///< Use executor for RoPE operations
    };

    /**
     * @brief Bridges LayerExecutor with existing pipeline infrastructure
     *
     * This adapter allows gradual migration from direct kernel calls to
     * the LayerExecutor-based execution model without disrupting existing
     * functionality.
     *
     * Usage in pipeline:
     * @code
     * // During pipeline initialization
     * pipeline_executor_ = std::make_unique<PipelineExecutor>(config, mpi_ctx_);
     * pipeline_executor_->setDeviceContext(device_idx_, num_threads);
     *
     * // In ffn_block():
     * if (pipeline_executor_->config().executor_ffn_norm) {
     *     pipeline_executor_->executeRMSNorm(input, gamma, output, seq_len, d_model, eps);
     * } else {
     *     // Existing direct kernel path
     * }
     * @endcode
     */
    class PipelineExecutor
    {
    public:
        /**
         * @brief Construct pipeline executor with shared MPI context
         * @param config Configuration
         * @param mpi_ctx MPI context (may be nullptr for single-rank)
         */
        PipelineExecutor(const PipelineExecutorConfig &config,
                         std::shared_ptr<MPIContext> mpi_ctx);

        /**
         * @brief Construct pipeline executor with raw MPI context pointer
         *
         * This constructor is useful when the caller has a unique_ptr to MPIContext
         * (e.g., default_mpi_ctx_ in PipelineBase). The executor does not take
         * ownership of the context - caller must ensure it outlives the executor.
         *
         * @param config Configuration
         * @param mpi_ctx Raw MPI context pointer (may be nullptr)
         */
        PipelineExecutor(const PipelineExecutorConfig &config,
                         MPIContext *mpi_ctx);

        ~PipelineExecutor();

        // Non-copyable
        PipelineExecutor(const PipelineExecutor &) = delete;
        PipelineExecutor &operator=(const PipelineExecutor &) = delete;

        // =========================================================================
        // Configuration
        // =========================================================================

        /**
         * @brief Get current configuration
         */
        const PipelineExecutorConfig &config() const { return config_; }

        /**
         * @brief Update configuration
         */
        void setConfig(const PipelineExecutorConfig &config) { config_ = config; }

        /**
         * @brief Set device context for a device
         * @param device_idx Device index
         * @param num_threads Thread count for CPU devices (ignored for GPU)
         */
        void setDeviceContext(int device_idx, int num_threads = 0);

        /**
         * @brief Get device context for a device
         */
        IDeviceContext *getDeviceContext(int device_idx) const;

        // =========================================================================
        // Individual Operation Execution
        // =========================================================================

        /**
         * @brief Execute RMSNorm using LayerExecutor
         *
         * @param input Input tensor (in-place modification)
         * @param gamma Norm weights
         * @param output Output tensor (can be same as input)
         * @param seq_len Sequence length
         * @param hidden_dim Hidden dimension
         * @param eps Epsilon for numerical stability
         * @param device_idx Target device
         * @return true on success
         */
        bool executeRMSNorm(TensorBase *input,
                            const TensorBase *gamma,
                            TensorBase *output,
                            int seq_len,
                            int hidden_dim,
                            float eps,
                            int device_idx = 0);

        /**
         * @brief Execute SwiGLU using LayerExecutor
         *
         * @param gate Gate tensor (silu applied to this)
         * @param up Up tensor
         * @param output Output tensor (can be same as up)
         * @param seq_len Sequence length
         * @param intermediate_dim FFN intermediate dimension
         * @param device_idx Target device
         * @return true on success
         */
        bool executeSwiGLU(TensorBase *gate,
                           TensorBase *up,
                           TensorBase *output,
                           int seq_len,
                           int intermediate_dim,
                           int device_idx = 0);

        /**
         * @brief Execute residual add using LayerExecutor
         *
         * output = input + residual
         *
         * @param input Input tensor
         * @param residual Residual tensor
         * @param output Output tensor
         * @param num_elements Total elements
         * @param device_idx Target device
         * @return true on success
         */
        bool executeResidualAdd(const TensorBase *input,
                                const TensorBase *residual,
                                TensorBase *output,
                                size_t num_elements,
                                int device_idx = 0);

        /**
         * @brief Execute RoPE using LayerExecutor
         *
         * @param Q Query tensor to modify
         * @param K Key tensor to modify
         * @param position_ids Position IDs for each token
         * @param seq_len Sequence length
         * @param n_heads Number of attention heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param theta RoPE theta base
         * @param device_idx Target device
         * @return true on success
         */
        bool executeRoPE(TensorBase *Q,
                         TensorBase *K,
                         const int *position_ids,
                         int seq_len,
                         int n_heads,
                         int n_kv_heads,
                         int head_dim,
                         float theta,
                         int device_idx = 0);

        // =========================================================================
        // Block-Level Execution (Future)
        // =========================================================================

        /**
         * @brief Execute complete FFN block using ComputeGraph
         *
         * This builds and executes a full FFN graph:
         * RMSNorm → Gate/Up → SwiGLU → Down → Residual
         *
         * Currently stubbed - requires GEMM integration with KernelFactory
         */
        // bool executeFFNBlock(const FFNBlockParams& params);

        /**
         * @brief Execute complete attention block using ComputeGraph
         *
         * Currently stubbed - requires attention kernel integration
         */
        // bool executeAttentionBlock(const AttentionBlockParams& params);

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Get LayerExecutor statistics
         */
        const LayerExecutorStats &stats() const;

        /**
         * @brief Reset statistics
         */
        void resetStats();

    private:
        PipelineExecutorConfig config_;
        std::shared_ptr<MPIContext> mpi_ctx_; ///< Shared ownership (may be nullptr)
        MPIContext *mpi_ctx_raw_ = nullptr;   ///< Raw pointer for operations (always valid if either ctor arg was non-null)
        std::unique_ptr<LayerExecutor> layer_executor_;
        std::unordered_map<int, std::unique_ptr<IDeviceContext>> device_contexts_;

        // Get or create context for device
        IDeviceContext *ensureContext(int device_idx);

        // Initialize LayerExecutor (shared between constructors)
        void initializeLayerExecutor();
    };

} // namespace llaminar2
