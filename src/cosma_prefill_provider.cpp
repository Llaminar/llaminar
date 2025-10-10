/**
 * @file cosma_prefill_provider.cpp
 * @brief COSMA-based prefill provider implementation
 * @author David Sanftenberg
 *
 * This implementation uses the Template Method pattern via PrefillProviderBaseImpl,
 * achieving significant code reduction:
 *
 * Original implementation: 608 lines
 * Refactored version: ~180 lines (70% reduction!)
 *
 * Eliminated code (now in base class):
 * - execute() main loop: ~180 lines
 * - executeTransformerLayer(): ~120 lines
 * - executeFfnBlock(): ~100 lines
 * - Utility methods: ~28 lines
 *
 * Remaining code (COSMA-specific):
 * - Constructor + initializeKernels(): ~100 lines
 * - executeEmbedding(): ~20 lines (manual memcpy, no kernel)
 * - executeLinearProjection(): ~10 lines (delegates to adaptiveMatMul)
 * - executeAttentionBlock(): ~50 lines (uses MPIAttentionKernel with COSMA backend)
 */

#include "cosma_prefill_provider.h"
#include "qwen_pipeline_adapter.h"
#include "kernels/MPIRMSNormKernel.h"
#include "kernels/MPIAttentionKernel.h"
#include "kernels/MPISwiGLUKernel.h"
#include "kernels/MPIResidualKernel.h"
#include "kernels/common/attention_primitives.h"
#include "tensors/tensor_factory.h"
#include "adaptive_matmul.h"
#include "backend_selector.h"
#include "logger.h"
#include "performance_timer.h"
#include <chrono>
#include <cstring>
#include <cmath>

namespace llaminar
{
    COSMAPrefillProvider::COSMAPrefillProvider(
        const ModelConfig &config, const MPIContext &mpi_ctx)
        : PrefillProviderBaseImpl(config, mpi_ctx)
    {
        initializeKernels();
    }

    void COSMAPrefillProvider::initializeKernels()
    {
        const auto &layer_cfg = config().getLayerConfig();

        // Attention kernel (used for snapshot callback mechanism)
        {
            auto attention_kernel = std::make_unique<MPIAttentionKernel>(
                layer_cfg.n_head,
                layer_cfg.n_head_kv,
                layer_cfg.head_dim,
                layer_cfg.rope_freq_base);

            // Configure output mode
            attention_kernel->setOutputMode(MPIAttentionKernel::AttentionOutputMode::GatherHeadsPostProjection);

            // Wire up snapshot callback for intermediate stages
            attention_kernel->setSnapshotCallback(
                [this](PipelineStage stage, int layer_idx, const float *data, int seq_len, int feature_dim)
                {
                    captureSnapshot(stage, layer_idx, data, seq_len, feature_dim);
                });

            if (!registerKernel("attention", std::move(attention_kernel)))
            {
                throw std::runtime_error("COSMAPrefillProvider: Failed to register attention kernel");
            }
        }

        // SwiGLU activation kernel
        {
            auto swiglu_kernel = std::make_unique<MPISwiGLUKernel>(
                MPISwiGLUKernel::DistributionStrategy::SEQUENCE_WISE);
            if (!registerKernel("swiglu", std::move(swiglu_kernel)))
            {
                throw std::runtime_error("COSMAPrefillProvider: Failed to register swiglu kernel");
            }
        }

        // Residual connection kernel
        {
            auto residual_kernel = std::make_unique<MPIResidualKernel>(
                MPIResidualKernel::DistributionStrategy::SEQUENCE_WISE);
            if (!registerKernel("residual", std::move(residual_kernel)))
            {
                throw std::runtime_error("COSMAPrefillProvider: Failed to register residual kernel");
            }
        }

        // RMSNorm kernel (fallback if COSMA fused path fails)
        {
            auto rmsnorm_kernel = std::make_unique<MPIRMSNormKernel>(
                MPIRMSNormKernel::DistributionStrategy::SEQUENCE_WISE);
            rmsnorm_kernel->setEpsilon(layer_cfg.eps);
            if (!registerKernel("rmsnorm", std::move(rmsnorm_kernel)))
            {
                throw std::runtime_error("COSMAPrefillProvider: Failed to register rmsnorm kernel");
            }
        }

        if (mpiContext().rank == 0)
        {
            LOG_DEBUG("COSMAPrefillProvider: Initialized " << kernels_.size() << " kernels");
        }
    }

    // ========================================================================
    // Backend-specific implementations
    // ========================================================================

    bool COSMAPrefillProvider::executeEmbedding(
        const std::vector<int> &tokens,
        std::shared_ptr<TensorBase> embedding_weight,
        std::shared_ptr<TensorBase> &output,
        int vocab_size)
    {
        // Simple embedding lookup (host-side, no COSMA needed)
        const auto &layer_cfg = config().getLayerConfig();
        int seq_len = static_cast<int>(tokens.size());
        int d_model = layer_cfg.d_model;

        const float *embed_weight = embedding_weight->data();
        float *embed_out = output->data();

        for (int i = 0; i < seq_len; ++i)
        {
            int token_id = tokens[i];
            if (token_id < 0 || token_id >= vocab_size)
            {
                LOG_ERROR("COSMAPrefillProvider: Invalid token ID " << token_id);
                return false;
            }
            std::memcpy(embed_out + (size_t)i * d_model,
                        embed_weight + (size_t)token_id * d_model,
                        d_model * sizeof(float));
        }

        return true;
    }

    bool COSMAPrefillProvider::executeLinearProjection(
        std::shared_ptr<TensorBase> input,
        std::shared_ptr<TensorBase> weight,
        std::shared_ptr<TensorBase> &output,
        int m, int n, int k,
        bool is_prefill,
        const std::string &operation_name)
    {
        // Use adaptiveMatMul (may route to COSMA for large ops)
        // Weight is [n, k], needs transpose to [k, n] for matmul
        bool ok = adaptiveMatMul(
            input->data(),
            weight->data(),
            output->data(),
            m,
            n,
            k,
            /*is_prefill=*/is_prefill,
            /*is_decode=*/false,
            /*transposed_a=*/false,
            /*transposed_b=*/true, // Weight stored as [n, k], transpose for matmul
            1.0f,
            0.0f);

        if (!ok)
        {
            LOG_ERROR("COSMAPrefillProvider: " << operation_name << " failed");
            return false;
        }

        return true;
    }

    bool COSMAPrefillProvider::executeAttentionBlock(
        int layer_idx,
        std::shared_ptr<TensorBase> &input,
        const QwenPipeline::ModelWeights &weights,
        std::shared_ptr<TensorBase> &attn_norm_out,
        std::shared_ptr<TensorBase> &attn_out,
        PrefillMetrics &metrics)
    {
        PERF_SCOPED_TIMER("COSMAPrefillProvider::executeAttentionBlock");

        const auto &layer_cfg = config().getLayerConfig();
        int seq_len = input->shape()[0];
        int d_model = layer_cfg.d_model;

        // Create plan for COSMA attention
        auto plan = plan_attention_prefill(seq_len, config(), mpiContext().size, mpiContext().rank);

        if (!plan.is_valid())
        {
            LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " COSMA plan invalid: " << plan.rationale);
            return false;
        }

        PrefillAttentionTiming timing;

        // --- Stage 1: RMSNorm (separate kernel, like OpenBLAS path) ---
        auto norm_start = std::chrono::high_resolution_clock::now();
        {
            std::vector<std::shared_ptr<TensorBase>> norm_inputs = {
                input,
                weights.attn_norm_weight[layer_idx]};
            std::vector<std::shared_ptr<TensorBase>> norm_outputs = {attn_norm_out};

            if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " attention norm failed");
                return false;
            }
        }
        auto norm_end = std::chrono::high_resolution_clock::now();
        timing.norm_ms = std::chrono::duration_cast<std::chrono::microseconds>(norm_end - norm_start).count() / 1000.0;

        // --- Stage 2: Attention via MPIAttentionKernel with COSMA backend ---
        auto attn_start = std::chrono::high_resolution_clock::now();

        // Configure attention kernel to use COSMA
        CosmaPrefillManager &manager = CosmaPrefillManager::instance();
        auto attention_kernel = dynamic_cast<MPIAttentionKernel *>(getKernel("attention"));
        if (attention_kernel)
        {
            attention_kernel->setSequencePosition(0); // Prefill always starts at position 0
            attention_kernel->setLayerIndex(layer_idx);
            attention_kernel->setCosmaManager(&manager); // INJECT COSMA BACKEND!
        }

        // Prepare temporary KV cache tensors for this layer
        auto k_cache = createLocalTensor({seq_len, layer_cfg.n_head_kv * layer_cfg.head_dim});
        auto v_cache = createLocalTensor({seq_len, layer_cfg.n_head_kv * layer_cfg.head_dim});

        // Call attention kernel (handles Q/K/V projections, RoPE, scores, output projection)
        std::vector<std::shared_ptr<TensorBase>> attn_inputs = {
            attn_norm_out,
            weights.wq[layer_idx],
            weights.wk[layer_idx],
            weights.wv[layer_idx],
            weights.wo[layer_idx],
            weights.bq[layer_idx],
            weights.bk[layer_idx],
            weights.bv[layer_idx],
            k_cache,
            v_cache};
        std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out};

        if (!executeKernel("attention", attn_inputs, attn_outputs))
        {
            LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " attention failed");
            return false;
        }

        auto attn_end = std::chrono::high_resolution_clock::now();
        timing.attention_ms = std::chrono::duration_cast<std::chrono::microseconds>(attn_end - attn_start).count() / 1000.0;

        // Update metrics
        metrics.norm_ms += timing.norm_ms;
        metrics.attention_ms += timing.attention_ms;

        return true;
    }

} // namespace llaminar
