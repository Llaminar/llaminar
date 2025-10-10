/**
 * @file openblas_prefill_provider.cpp
 * @brief OpenBLAS-based prefill provider implementation
 * @author David Sanftenberg
 *
 * This implementation uses the Template Method pattern via PrefillProviderBaseImpl,
 * achieving significant code reduction:
 *
 * Original implementation: 655 lines
 * Refactored version: ~200 lines (69% reduction!)
 *
 * Eliminated code (now in base class):
 * - execute() main loop: ~180 lines
 * - executeTransformerLayer(): ~120 lines
 * - executeFfnBlock(): ~100 lines
 * - Utility methods: ~55 lines
 *
 * Remaining code (OpenBLAS-specific):
 * - Constructor + initializeKernels(): ~140 lines
 * - executeEmbedding(): ~25 lines
 * - executeLinearProjection(): ~15 lines
 * - executeAttentionBlock(): ~80 lines
 * - setKVCache(): ~10 lines
 */

#include "openblas_prefill_provider.h"
#include "qwen_pipeline_adapter.h"
#include "kernels/MPILinearKernel.h"
#include "kernels/MPIRMSNormKernel.h"
#include "kernels/MPIAttentionKernel.h"
#include "kernels/MPIEmbeddingKernel.h"
#include "kernels/MPISwiGLUKernel.h"
#include "kernels/MPIResidualKernel.h"
#include "tensors/tensor_factory.h"
#include "logger.h"
#include "performance_timer.h"
#include <chrono>
#include <cstring>

namespace llaminar
{
    OpenBLASPrefillProvider::OpenBLASPrefillProvider(
        const ModelConfig &config, const MPIContext &mpi_ctx)
        : PrefillProviderBaseImpl(config, mpi_ctx)
    {
        initializeKernels();

        // Pre-allocate KV cache vectors (will be populated later if needed)
        int n_layers = config.getLayerConfig().n_layers;
        k_cache_.resize(n_layers);
        v_cache_.resize(n_layers);
    }

    void OpenBLASPrefillProvider::initializeKernels()
    {
        const auto &layer_cfg = config().getLayerConfig();

        // Embedding kernel
        {
            auto embedding_kernel = std::make_unique<MPIEmbeddingKernel>(
                layer_cfg.vocab_size,
                layer_cfg.d_model);
            if (!registerKernel("embedding", std::move(embedding_kernel)))
            {
                throw std::runtime_error("OpenBLASPrefillProvider: Failed to register embedding kernel");
            }
        }

        // RMSNorm kernel (sequence-wise distribution)
        {
            auto rmsnorm_kernel = std::make_unique<MPIRMSNormKernel>(
                MPIRMSNormKernel::DistributionStrategy::SEQUENCE_WISE);
            rmsnorm_kernel->setEpsilon(layer_cfg.eps);
            if (!registerKernel("rmsnorm", std::move(rmsnorm_kernel)))
            {
                throw std::runtime_error("OpenBLASPrefillProvider: Failed to register rmsnorm kernel");
            }
        }

        // Attention kernel (handles Q/K/V/O projections internally)
        {
            auto attention_kernel = std::make_unique<MPIAttentionKernel>(
                layer_cfg.n_head, layer_cfg.n_head_kv,
                layer_cfg.head_dim, layer_cfg.rope_freq_base);

            // CRITICAL: Must use GatherHeadsPostProjection mode for multi-rank execution
            attention_kernel->setOutputMode(MPIAttentionKernel::AttentionOutputMode::GatherHeadsPostProjection);

            if (!registerKernel("attention", std::move(attention_kernel)))
            {
                throw std::runtime_error("OpenBLASPrefillProvider: Failed to register attention kernel");
            }
        }

        // Linear kernel (for FFN projections)
        {
            auto linear_kernel = std::make_unique<MPILinearKernel>();
            if (!registerKernel("linear", std::move(linear_kernel)))
            {
                throw std::runtime_error("OpenBLASPrefillProvider: Failed to register linear kernel");
            }
        }

        // SwiGLU activation kernel
        {
            auto swiglu_kernel = std::make_unique<MPISwiGLUKernel>(
                MPISwiGLUKernel::DistributionStrategy::SEQUENCE_WISE);
            if (!registerKernel("swiglu", std::move(swiglu_kernel)))
            {
                throw std::runtime_error("OpenBLASPrefillProvider: Failed to register swiglu kernel");
            }
        }

        // Residual connection kernel
        {
            auto residual_kernel = std::make_unique<MPIResidualKernel>(
                MPIResidualKernel::DistributionStrategy::SEQUENCE_WISE);
            if (!registerKernel("residual", std::move(residual_kernel)))
            {
                throw std::runtime_error("OpenBLASPrefillProvider: Failed to register residual kernel");
            }
        }

        if (mpiContext().rank == 0)
        {
            LOG_DEBUG("OpenBLASPrefillProvider: Initialized " << kernels_.size() << " kernels");
        }
    }

    void OpenBLASPrefillProvider::setKVCache(
        const std::vector<std::shared_ptr<TensorBase>> &k_cache,
        const std::vector<std::shared_ptr<TensorBase>> &v_cache)
    {
        k_cache_ = k_cache;
        v_cache_ = v_cache;
        use_kv_cache_ = true;
    }

    // ========================================================================
    // Backend-specific implementations
    // ========================================================================

    bool OpenBLASPrefillProvider::executeEmbedding(
        const std::vector<int> &tokens,
        std::shared_ptr<TensorBase> embedding_weight,
        std::shared_ptr<TensorBase> &output,
        int vocab_size)
    {
        // Use MPIEmbeddingKernel
        std::vector<std::shared_ptr<TensorBase>> embed_inputs = {embedding_weight};
        std::vector<std::shared_ptr<TensorBase>> embed_outputs = {output};

        // Create temporary 1D token tensor for embedding kernel
        auto token_tensor = createLocalTensor({static_cast<int>(tokens.size())});
        float *token_data = token_tensor->data();
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            token_data[i] = static_cast<float>(tokens[i]);
        }
        embed_inputs.insert(embed_inputs.begin(), token_tensor);

        return executeKernel("embedding", embed_inputs, embed_outputs);
    }

    bool OpenBLASPrefillProvider::executeLinearProjection(
        std::shared_ptr<TensorBase> input,
        std::shared_ptr<TensorBase> weight,
        std::shared_ptr<TensorBase> &output,
        int m, int n, int k,
        bool is_prefill,
        const std::string &operation_name)
    {
        // Use MPILinearKernel (wraps OpenBLAS GEMM)
        std::vector<std::shared_ptr<TensorBase>> linear_inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> linear_outputs = {output};

        return executeKernel("linear", linear_inputs, linear_outputs);
    }

    bool OpenBLASPrefillProvider::executeAttentionBlock(
        int layer_idx,
        std::shared_ptr<TensorBase> &input,
        const QwenPipeline::ModelWeights &weights,
        std::shared_ptr<TensorBase> &attn_norm_out,
        std::shared_ptr<TensorBase> &attn_out,
        PrefillMetrics &metrics)
    {
        PERF_SCOPED_TIMER("OpenBLASPrefillProvider::executeAttentionBlock");

        const auto &layer_cfg = config().getLayerConfig();
        int seq_len = input->shape()[0];
        int d_model = layer_cfg.d_model;

        auto t_attn_start = std::chrono::high_resolution_clock::now();

        // Perform RMSNorm first (separate operation)
        {
            auto t_norm_start = std::chrono::high_resolution_clock::now();

            std::vector<std::shared_ptr<TensorBase>> norm_inputs = {
                input,
                weights.attn_norm_weight[layer_idx]};
            std::vector<std::shared_ptr<TensorBase>> norm_outputs = {attn_norm_out};

            if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " attention norm failed");
                return false;
            }

            auto t_norm_end = std::chrono::high_resolution_clock::now();
            metrics.norm_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                                   t_norm_end - t_norm_start)
                                   .count() /
                               1000.0;
        }

        // Configure attention kernel
        auto attention_kernel = dynamic_cast<MPIAttentionKernel *>(getKernel("attention"));
        if (attention_kernel)
        {
            attention_kernel->setSequencePosition(n_past_);
            attention_kernel->setLayerIndex(layer_idx);

            // Set snapshot callback for intermediate attention stages
            attention_kernel->setSnapshotCallback(
                [this, layer_idx](PipelineStage stage, int layer, const float *data, int seq_len, int feature_dim)
                {
                    this->captureSnapshot(stage, layer, data, seq_len, feature_dim);
                });
        }

        // Prepare KV cache tensors
        auto k_cache = use_kv_cache_ && k_cache_[layer_idx]
                           ? k_cache_[layer_idx]
                           : createLocalTensor({seq_len, layer_cfg.n_head_kv * layer_cfg.head_dim});
        auto v_cache = use_kv_cache_ && v_cache_[layer_idx]
                           ? v_cache_[layer_idx]
                           : createLocalTensor({seq_len, layer_cfg.n_head_kv * layer_cfg.head_dim});

        // Execute attention kernel
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
            LOG_ERROR("Layer " << layer_idx << " attention failed");
            return false;
        }

        auto t_attn_end = std::chrono::high_resolution_clock::now();
        metrics.attention_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                                    t_attn_end - t_attn_start)
                                    .count() /
                                1000.0;

        return true;
    }

} // namespace llaminar
