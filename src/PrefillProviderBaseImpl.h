/**
 * @file PrefillProviderBaseImpl.h
 * @brief Shared implementation logic for prefill providers
 * @author David Sanftenberg
 *
 * This file extracts common execution patterns from OpenBLASPrefillProvider and
 * COSMAPrefillProvider to eliminate ~200-250 lines of code duplication.
 *
 * Architecture:
 * - Template Method Pattern: Base class defines execution flow, derived classes
 *   implement backend-specific operations (matmul strategy, attention path)
 * - Strategy Pattern: Virtual methods for projections allow OpenBLAS vs COSMA
 *
 * Shared Logic (~70-80% of provider code):
 * - Main execution scaffolding (embedding → layers → final norm → LM head)
 * - FFN block structure (norm → gate/up → swiglu → down → residual)
 * - Snapshot capture at all 387 standardized stages
 * - Timing/metrics collection
 * - Error handling patterns
 *
 * Backend-Specific Logic (delegated to derived classes):
 * - executeLinearProjection(): MPILinearOperator vs adaptiveMatMul
 * - executeAttentionBlock(): MPIAttentionOperator vs COSMA fused path
 * - executeEmbedding(): MPIEmbeddingOperator vs manual memcpy
 *
 * Benefits:
 * - DRY principle: Single source of truth for execution flow
 * - Consistency: Identical snapshot capture across backends
 * - Maintainability: Fix bugs once, applies to all backends
 * - Extensibility: New backends (e.g., cuBLAS) only implement 3 virtual methods
 *
 * Usage:
 * @code
 *   class MyPrefillProvider : public PrefillProviderBaseImpl {
 *   protected:
 *       bool executeEmbedding(...) override { ... }
 *       bool executeLinearProjection(...) override { ... }
 *       bool executeAttentionBlock(...) override { ... }
 *   };
 * @endcode
 */

#pragma once

#include "PrefillProvider.h"
#include "QwenPipeline.h"
#include "PipelineBase.h"
#include "Logger.h"
#include "PerformanceTimer.h"
#include <memory>
#include <vector>
#include <chrono>

namespace llaminar
{
    /**
     * @brief Base implementation class providing shared prefill execution logic
     *
     * This class implements the Template Method pattern, defining the overall
     * execution flow while delegating backend-specific operations to derived classes.
     *
     * Execution Flow (shared across all backends):
     * 1. Token embedding lookup
     * 2. N transformer layers (attention + FFN)
     * 3. Final RMSNorm
     * 4. LM head projection
     *
     * Each derived class must implement:
     * - executeEmbedding(): How to map token IDs to embeddings
     * - executeLinearProjection(): How to perform Y = XW (backend-specific matmul)
     * - executeAttentionBlock(): How to compute attention (kernel vs COSMA vs etc.)
     */
    class PrefillProviderBaseImpl : public PrefillProvider
    {
    public:
        /**
         * @brief Construct base implementation provider
         *
         * @param config Model configuration
         * @param mpi_ctx MPI context for distributed execution
         */
        PrefillProviderBaseImpl(const ModelConfig &config, const MPIContext &mpi_ctx)
            : PrefillProvider(config, mpi_ctx), n_past_(0)
        {
        }

        virtual ~PrefillProviderBaseImpl() = default;

        /**
         * @brief Execute prefill for input tokens (shared implementation)
         *
         * This method implements the complete prefill pipeline using the template
         * method pattern. It calls virtual methods for backend-specific operations.
         *
         * @param tokens Input token IDs
         * @param weights Model weights (must be QwenPipeline::ModelWeights)
         * @param output Output tensor (final hidden states or logits)
         * @param ctx Stage context (sequence length, KV cache state)
         * @param metrics Output metrics (timing, FLOPs)
         * @param cache_provider Optional KV cache provider
         *
         * @return true if execution succeeded, false on error
         */
        bool execute(
            const std::vector<int> &tokens,
            const IModelWeights &weights,
            std::shared_ptr<TensorBase> &output,
            StageContext &ctx,
            PrefillMetrics &metrics,
            KVCacheProvider *cache_provider = nullptr) override;

    protected:
        // ========================================================================
        // Virtual methods for backend-specific operations (must be implemented)
        // ========================================================================

        /**
         * @brief Execute token embedding lookup (backend-specific)
         *
         * @param tokens Input token IDs
         * @param embedding_weight Embedding weight tensor [vocab_size, d_model]
         * @param output Output embeddings [seq_len, d_model]
         * @param vocab_size Vocabulary size
         *
         * @return true if successful, false on error
         *
         * Implementations:
         * - OpenBLAS: Uses MPIEmbeddingOperator
         * - COSMA: Manual memcpy loop (simple lookup, no matmul needed)
         */
        virtual bool executeEmbedding(
            const std::vector<int> &tokens,
            std::shared_ptr<TensorBase> embedding_weight,
            std::shared_ptr<TensorBase> &output,
            int vocab_size) = 0;

        /**
         * @brief Execute linear projection Y = XW^T (backend-specific matmul)
         *
         * @param input Input tensor X [m, k]
         * @param weight Weight tensor W [n, k] (will be transposed to [k, n])
         * @param output Output tensor Y [m, n]
         * @param m Number of rows (sequence length)
         * @param n Number of output features
         * @param k Number of input features
         * @param is_prefill Whether this is a prefill operation (affects backend selection)
         * @param operation_name Human-readable name for logging/debugging
         *
         * @return true if successful, false on error
         *
         * Implementations:
         * - OpenBLAS: Uses MPILinearOperator (wraps cblas_sgemm)
         * - COSMA: Uses adaptiveMatMul (may route to COSMA for large ops)
         *
         * Note: Weight is assumed to be stored as [n, k] and needs transpose flag.
         */
        virtual bool executeLinearProjection(
            std::shared_ptr<TensorBase> input,
            std::shared_ptr<TensorBase> weight,
            std::shared_ptr<TensorBase> &output,
            int m, int n, int k,
            bool is_prefill,
            const std::string &operation_name) = 0;

        /**
         * @brief Execute complete attention block (backend-specific)
         *
         * This encompasses:
         * 1. Attention normalization (RMSNorm)
         * 2. Q/K/V projections
         * 3. RoPE (rotary position encoding)
         * 4. Attention scores and softmax
         * 5. Context computation
         * 6. Output projection
         *
         * @param layer_idx Layer index (0-based)
         * @param input Input tensor [seq_len, d_model]
         * @param weights Model weights for this layer
         * @param attn_norm_out Output: Normalized input [seq_len, d_model]
         * @param attn_out Output: Final attention output [seq_len, d_model]
         * @param metrics Metrics to update (timing)
         * @param cache_provider Optional KV cache provider to populate during execution
         *
         * @return true if successful, false on error
         *
         * Implementations:
         * - OpenBLAS: Uses MPIAttentionOperator (all-in-one kernel)
         * - COSMA: Uses CosmaPrefillManager (fused RMSNorm+QKV) + attention primitives
         *
         * CRITICAL: Must populate attn_norm_out for snapshot capture consistency!
         * CRITICAL: If cache_provider is non-null, MUST populate with K/V cache for this layer!
         */
        virtual bool executeAttentionBlock(
            int layer_idx,
            std::shared_ptr<TensorBase> &input,
            const QwenPipeline::ModelWeights &weights,
            std::shared_ptr<TensorBase> &attn_norm_out,
            std::shared_ptr<TensorBase> &attn_out,
            PrefillMetrics &metrics,
            KVCacheProvider *cache_provider) = 0;

        // ========================================================================
        // Shared implementation methods (used by execute())
        // ========================================================================

        /**
         * @brief Execute complete transformer layer (shared implementation)
         *
         * This method implements the full transformer layer pipeline:
         * - Attention block (delegated to executeAttentionBlock)
         * - Attention residual connection
         * - FFN block (shared implementation below)
         * - FFN residual connection
         *
         * All snapshot captures are performed at standardized locations.
         *
         * @param layer_idx Layer index (0-based)
         * @param input Layer input [seq_len, d_model]
         * @param weights Model weights
         * @param output Layer output [seq_len, d_model]
         * @param metrics Metrics to update
         *
         * @return true if successful, false on error
         */
        bool executeTransformerLayer(
            int layer_idx,
            std::shared_ptr<TensorBase> &input,
            const QwenPipeline::ModelWeights &weights,
            std::shared_ptr<TensorBase> &output,
            PrefillMetrics &metrics,
            KVCacheProvider *cache_provider);

        /**
         * @brief Execute FFN block (shared implementation)
         *
         * Pipeline:
         * 1. FFN normalization (RMSNorm)
         * 2. Gate projection: gate = norm(x) @ W_gate^T
         * 3. Up projection: up = norm(x) @ W_up^T
         * 4. SwiGLU activation: swiglu = silu(gate) * up
         * 5. Down projection: out = swiglu @ W_down^T
         *
         * All projections delegate to executeLinearProjection().
         * All snapshots captured at standardized FFN stages.
         *
         * @param layer_idx Layer index (0-based)
         * @param input FFN input (after attention residual) [seq_len, d_model]
         * @param weights Model weights for this layer
         * @param output FFN output (before final residual) [seq_len, d_model]
         * @param metrics Metrics to update
         *
         * @return true if successful, false on error
         */
        bool executeFfnBlock(
            int layer_idx,
            std::shared_ptr<TensorBase> &input,
            const QwenPipeline::ModelWeights &weights,
            std::shared_ptr<TensorBase> &output,
            PrefillMetrics &metrics);

        /**
         * @brief Create local (non-distributed) tensor
         *
         * Utility for allocating intermediate tensors. Derived classes can override
         * if they need different allocation strategies (e.g., COSMA-aligned buffers).
         *
         * @param shape Tensor shape
         * @return Allocated tensor
         */
        virtual std::shared_ptr<TensorBase> createLocalTensor(const std::vector<int> &shape);

        /**
         * @brief Execute a registered kernel
         *
         * @param kernel_name Kernel name (e.g., "rmsnorm", "swiglu", "residual")
         * @param inputs Input tensors
         * @param outputs Output tensors
         * @return true if successful, false on error
         */
        bool executeKernel(
            const std::string &kernel_name,
            const std::vector<std::shared_ptr<TensorBase>> &inputs,
            std::vector<std::shared_ptr<TensorBase>> &outputs);

        /**
         * @brief Get a registered kernel by name
         *
         * @param name Kernel name
         * @return Pointer to kernel, or nullptr if not found
         */
        MPIOperatorBase *getKernel(const std::string &name);

        /**
         * @brief Register a kernel
         *
         * @param name Kernel name
         * @param kernel Kernel instance (ownership transferred)
         * @return true if successful, false if already registered
         */
        bool registerOperator(const std::string &name, std::unique_ptr<MPIOperatorBase> kernel);

        // ========================================================================
        // Member variables
        // ========================================================================

        /// Registered kernels (shared: rmsnorm, swiglu, residual; backend-specific: others)
        std::unordered_map<std::string, std::unique_ptr<MPIOperatorBase>> kernels_;

        /// Number of tokens processed so far (for incremental decode position tracking)
        int n_past_;
    };

} // namespace llaminar
