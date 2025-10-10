/**
 * @file cosma_prefill_provider.h
 * @brief COSMA-based prefill provider using Template Method pattern
 * @author David Sanftenberg
 *
 * This provider inherits from PrefillProviderBaseImpl, implementing only
 * backend-specific operations (embedding, linear projection, attention).
 * COSMA is used for distributed matrix multiplication when beneficial.
 *
 * Code Reduction vs Original Implementation:ile cosma_prefill_provider_refactored.h
 * @brief Refactored COSMA provider using shared base implementation
 * @author David Sanftenberg
 *
 * This is a SKETCH showing how COSMAPrefillProvider would look after
 * refactoring to use PrefillProviderBaseImpl. This demonstrates:
 *
 * Code Reduction:
 * - Original: 608 lines
 * - Refactored: ~180 lines (70% reduction!)
 * - Eliminated: ~428 lines of duplicated execution scaffolding
 *
 * What Remains:
 * - Constructor with COSMA-specific kernel initialization (4 kernels)
 * - 3 virtual method implementations (embedding, adaptive matmul, attention)
 * - COSMA-specific attention path (fused RMSNorm+QKV via CosmaPrefillManager)
 *
 * What's Gone (moved to base class):
 * - Main execute() method (~180 lines)
 * - executeTransformerLayer() (~120 lines)
 * - executeFfnBlock() (~100 lines)
 * - All snapshot capture logic (16 capture points)
 * - All timing/metrics collection
 * - Kernel registration infrastructure
 *
 * Key Difference from OpenBLAS:
 * - Uses adaptiveMatMul() instead of MPILinearKernel
 * - Uses CosmaPrefillManager for fused attention operations
 * - Manual embedding (simple memcpy, no kernel needed)
 */

#pragma once

#include "prefill_provider_base_impl.h"
#include "cosma_prefill_manager.h"
#include "large_matmul_plan.h"
#include <memory>
#include <vector>

namespace llaminar
{
    /**
     * @brief COSMA-based prefill provider (refactored version)
     *
     * This provider uses COSMA distributed matrix multiplication for large
     * operations and adaptiveMatMul for others. It inherits all execution
     * scaffolding from PrefillProviderBaseImpl.
     *
     * Backend-Specific Features:
     * - Manual embedding (memcpy, no kernel overhead)
     * - adaptiveMatMul for all linear projections (routes to COSMA when beneficial)
     * - CosmaPrefillManager for fused RMSNorm+QKV attention
     * - LargeMatmulPlan for distributed execution planning
     *
     * Shared Features (from base class):
     * - Execution flow (embedding → layers → norm → LM head)
     * - FFN block structure
     * - Snapshot capture at 387 standardized stages
     * - Timing/metrics collection
     */
    class COSMAPrefillProvider : public PrefillProviderBaseImpl
    {
    public:
        /**
         * @brief Construct COSMA prefill provider
         *
         * @param config Model configuration
         * @param mpi_ctx MPI context for distributed execution
         */
        COSMAPrefillProvider(const ModelConfig &config, const MPIContext &mpi_ctx);

        /**
         * @brief Get provider name
         * @return "COSMA"
         */
        std::string name() const override { return "COSMA"; }

    protected:
        // ========================================================================
        // Backend-specific implementations (required by base class)
        // ========================================================================

        /**
         * @brief Execute token embedding using manual memcpy
         *
         * No kernel overhead - simple lookup is faster for embedding.
         */
        bool executeEmbedding(
            const std::vector<int> &tokens,
            std::shared_ptr<TensorBase> embedding_weight,
            std::shared_ptr<TensorBase> &output,
            int vocab_size) override;

        /**
         * @brief Execute linear projection using adaptiveMatMul
         *
         * adaptiveMatMul automatically selects backend based on operation size:
         * - Small ops (<8K elements): Single-threaded OpenBLAS
         * - Medium ops: Multi-threaded OpenBLAS
         * - Large ops (≥8M elements, prefill): May use COSMA distributed
         */
        bool executeLinearProjection(
            std::shared_ptr<TensorBase> input,
            std::shared_ptr<TensorBase> weight,
            std::shared_ptr<TensorBase> &output,
            int m, int n, int k,
            bool is_prefill,
            const std::string &operation_name) override;

        /**
         * @brief Execute attention block using COSMA fused path
         *
         * Uses CosmaPrefillManager for:
         * - Fused RMSNorm + QKV projection (single distributed operation)
         * - Attention primitives (RoPE, scores, softmax, context)
         * - Output projection (via adaptiveMatMul)
         *
         * Populates attn_norm_out from fused operation for snapshot consistency.
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
         * @brief Initialize COSMA-specific kernels
         *
         * Registers:
         * - attention: MPIAttentionKernel (for snapshot callback mechanism)
         * - rmsnorm: MPIRMSNormKernel (fallback if COSMA fused path unavailable)
         * - swiglu: MPISwiGLUKernel
         * - residual: MPIResidualKernel
         *
         * Note: No linear kernel - uses adaptiveMatMul directly.
         * Note: No embedding kernel - manual memcpy is faster.
         */
        void initializeKernels();

        /**
         * @brief Timing breakdown for COSMA attention
         */
        struct PrefillAttentionTiming
        {
            double norm_ms = 0.0;      ///< RMSNorm + QKV projection time
            double attention_ms = 0.0; ///< RoPE + attention computation time
            double linear_ms = 0.0;    ///< Output projection time
        };
    };

} // namespace llaminar
