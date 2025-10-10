/**
 * @file openblas_prefill_provider.h
 * @brief OpenBLAS-based prefill provider using Template Method pattern
 * @author David Sanftenberg
 *
 * This provider inherits from PrefillProviderBaseImpl, implementing only
 * backend-specific operations (embedding, linear projection, attention).
 * All execution flow and snapshot capture is handled by the base class.
 *
 * Code Reduction vs Original Implementation:
 * - Original: 655 lines
 * - Refactored: ~200 lines (69% reduction!)
 * - Eliminated: ~455 lines of duplicated execution scaffolding
 *
 * What Remains:
 * - Constructor with OpenBLAS-specific kernel initialization (6 kernels)
 * - 3 virtual method implementations (embedding, linear projection, attention)
 * - KV cache management (OpenBLAS-specific feature)
 *
 * What's Gone (moved to base class):
 * - Main execute() method (~180 lines)
 * - executeTransformerLayer() (~120 lines)
 * - executeFfnBlock() (~100 lines)
 * - All snapshot capture logic (16 capture points)
 * - All timing/metrics collection
 * - Kernel registration infrastructure
 */

#pragma once

#include "prefill_provider_base_impl.h"
#include <memory>
#include <vector>

namespace llaminar
{
    /**
     * @brief OpenBLAS-based prefill provider (refactored version)
     *
     * This provider uses MPILinearKernel for all matrix multiplications,
     * which wraps OpenBLAS GEMM operations. It inherits all execution
     * scaffolding from PrefillProviderBaseImpl.
     *
     * Backend-Specific Features:
     * - MPIEmbeddingKernel for token embedding lookup
     * - MPILinearKernel for all linear projections (gate, up, down, LM head)
     * - MPIAttentionKernel for complete attention block
     * - KV cache management (for incremental decode)
     *
     * Shared Features (from base class):
     * - Execution flow (embedding → layers → norm → LM head)
     * - FFN block structure
     * - Snapshot capture at 387 standardized stages
     * - Timing/metrics collection
     */
    class OpenBLASPrefillProvider : public PrefillProviderBaseImpl
    {
    public:
        /**
         * @brief Construct OpenBLAS prefill provider
         *
         * @param config Model configuration
         * @param mpi_ctx MPI context for distributed execution
         */
        OpenBLASPrefillProvider(const ModelConfig &config, const MPIContext &mpi_ctx);

        /**
         * @brief Get provider name
         * @return "OpenBLAS"
         */
        std::string name() const override { return "OpenBLAS"; }

        /**
         * @brief Set KV cache for attention computation
         *
         * @param k_cache Key cache tensors (one per layer)
         * @param v_cache Value cache tensors (one per layer)
         */
        void setKVCache(
            const std::vector<std::shared_ptr<TensorBase>> &k_cache,
            const std::vector<std::shared_ptr<TensorBase>> &v_cache);

    protected:
        // ========================================================================
        // Backend-specific implementations (required by base class)
        // ========================================================================

        /**
         * @brief Execute token embedding using MPIEmbeddingKernel
         */
        bool executeEmbedding(
            const std::vector<int> &tokens,
            std::shared_ptr<TensorBase> embedding_weight,
            std::shared_ptr<TensorBase> &output,
            int vocab_size) override;

        /**
         * @brief Execute linear projection using MPILinearKernel (OpenBLAS GEMM)
         */
        bool executeLinearProjection(
            std::shared_ptr<TensorBase> input,
            std::shared_ptr<TensorBase> weight,
            std::shared_ptr<TensorBase> &output,
            int m, int n, int k,
            bool is_prefill,
            const std::string &operation_name) override;

        /**
         * @brief Execute attention block using MPIAttentionKernel
         *
         * This kernel handles:
         * - RMSNorm (populates attn_norm_out)
         * - Q/K/V projections
         * - RoPE
         * - Attention scores/softmax
         * - Context computation
         * - Output projection
         *
         * Captures intermediate snapshots via callback mechanism.
         */
        bool executeAttentionBlock(
            int layer_idx,
            std::shared_ptr<TensorBase> &input,
            const QwenPipeline::ModelWeights &weights,
            std::shared_ptr<TensorBase> &attn_norm_out,
            std::shared_ptr<TensorBase> &attn_out,
            PrefillMetrics &metrics) override;

    private:
        /**
         * @brief Initialize OpenBLAS-specific kernels
         *
         * Registers:
         * - embedding: MPIEmbeddingKernel
         * - rmsnorm: MPIRMSNormKernel
         * - attention: MPIAttentionKernel (with GatherHeadsPostProjection mode)
         * - linear: MPILinearKernel
         * - swiglu: MPISwiGLUKernel
         * - residual: MPIResidualKernel
         */
        void initializeKernels();

        // KV cache (OpenBLAS-specific, COSMA doesn't use yet)
        std::vector<std::shared_ptr<TensorBase>> k_cache_;
        std::vector<std::shared_ptr<TensorBase>> v_cache_;
        bool use_kv_cache_ = false;
    };

} // namespace llaminar
