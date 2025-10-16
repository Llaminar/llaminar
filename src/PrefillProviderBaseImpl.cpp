/**
 * @file PrefillProviderBaseImpl.cpp
 * @brief Implementation of shared prefill execution logic
 * @author David Sanftenberg
 */

#include "PrefillProviderBaseImpl.h"
#include "QwenPipelineAdapter.h"
#include "operators/MPIRMSNormOperator.h"
#include "operators/MPISwiGLUOperator.h"
#include "operators/MPIResidualOperator.h"
#include "tensors/TensorFactory.h"
#include "Logger.h"
#include "PerformanceTimer.h"
#include <chrono>
#include <cstring>

namespace llaminar
{
    // ========================================================================
    // Main execution method (shared across all backends)
    // ========================================================================

    bool PrefillProviderBaseImpl::execute(
        const std::vector<int> &tokens,
        const IModelWeights &weights,
        std::shared_ptr<TensorBase> &output,
        StageContext &ctx,
        PrefillMetrics &metrics,
        KVCacheProvider *cache_provider)
    {
        PERF_SCOPED_TIMER("PrefillProviderBaseImpl::execute");

        // Reset metrics (backend name set by derived class)
        metrics.reset();

        // Cast weights to concrete type
        const auto *qwen_model_weights = dynamic_cast<const QwenModelWeights *>(&weights);
        if (!qwen_model_weights)
        {
            LOG_ERROR("PrefillProviderBaseImpl: Invalid weights type (expected QwenModelWeights)");
            return false;
        }
        const auto *qwen_weights = &qwen_model_weights->inner;

        const auto &layer_cfg = config().getLayerConfig();
        int seq_len = static_cast<int>(tokens.size());
        int d_model = layer_cfg.d_model;
        int n_layers = layer_cfg.n_layers;
        int vocab_size = layer_cfg.vocab_size;

        // Update context
        ctx.seq_len = seq_len;

        // === Stage 1: Token Embedding ===
        auto t_embed_start = std::chrono::high_resolution_clock::now();

        auto embedded = createLocalTensor({seq_len, d_model});
        if (!executeEmbedding(tokens, qwen_weights->token_embedding, embedded, vocab_size))
        {
            LOG_ERROR("PrefillProviderBaseImpl: Embedding failed");
            return false;
        }

        auto t_embed_end = std::chrono::high_resolution_clock::now();
        metrics.embedding_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                   t_embed_end - t_embed_start)
                                   .count() /
                               1000.0;

        // Capture embedding snapshot
        captureSnapshot(PipelineStage::EMBEDDING, -1, embedded->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // === Stage 2: Transformer Layers ===
        auto layer_input = embedded;
        auto layer_output = createLocalTensor({seq_len, d_model});

        for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
        {
            if (!executeTransformerLayer(layer_idx, layer_input, *qwen_weights, layer_output, metrics, cache_provider))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " failed");
                return false;
            }

            // Swap buffers for next layer
            std::swap(layer_input, layer_output);
            metrics.layers_executed++;
        }

        // === Stage 3: Final Normalization ===
        auto t_norm_start = std::chrono::high_resolution_clock::now();

        auto final_norm_out = createLocalTensor({seq_len, d_model});
        {
            std::vector<std::shared_ptr<TensorBase>> norm_inputs = {
                layer_input,
                qwen_weights->output_norm_weight};
            std::vector<std::shared_ptr<TensorBase>> norm_outputs = {final_norm_out};

            if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Final norm failed");
                return false;
            }
        }

        auto t_norm_end = std::chrono::high_resolution_clock::now();
        metrics.norm_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                               t_norm_end - t_norm_start)
                               .count() /
                           1000.0;

        // Capture final norm snapshot
        captureSnapshot(PipelineStage::FINAL_NORM, -1, final_norm_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // === Stage 4: LM Head Projection ===
        auto t_lm_start = std::chrono::high_resolution_clock::now();

        auto logits = createLocalTensor({seq_len, vocab_size});
        if (!executeLinearProjection(
                final_norm_out,
                qwen_weights->lm_head,
                logits,
                seq_len,
                vocab_size,
                d_model,
                /*is_prefill=*/true,
                "lm_head"))
        {
            LOG_ERROR("PrefillProviderBaseImpl: LM head failed");
            return false;
        }

        auto t_lm_end = std::chrono::high_resolution_clock::now();
        metrics.lm_head_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                 t_lm_end - t_lm_start)
                                 .count() /
                             1000.0;

        // Capture LM head snapshot
        captureSnapshot(PipelineStage::LM_HEAD, -1, logits->data(), seq_len, vocab_size);
        incrementSnapshotCounter(metrics);

        // Set output
        output = logits;

        // Log summary
        if (mpiContext().rank == 0)
        {
            LOG_INFO("PrefillProvider[" << metrics.backend_name << "]: "
                                        << seq_len << " tokens, "
                                        << n_layers << " layers, "
                                        << metrics.total_ms() << "ms total, "
                                        << metrics.snapshots_captured << " snapshots");
        }

        return true;
    }

    // ========================================================================
    // Transformer layer execution (shared)
    // ========================================================================

    bool PrefillProviderBaseImpl::executeTransformerLayer(
        int layer_idx,
        std::shared_ptr<TensorBase> &input,
        const QwenPipeline::ModelWeights &weights,
        std::shared_ptr<TensorBase> &output,
        PrefillMetrics &metrics,
        KVCacheProvider *cache_provider)
    {
        PERF_SCOPED_TIMER("PrefillProviderBaseImpl::executeTransformerLayer");

        const auto &layer_cfg = config().getLayerConfig();
        int seq_len = input->shape()[0];
        int d_model = layer_cfg.d_model;

        // Allocate intermediate tensors
        auto attn_norm_out = createLocalTensor({seq_len, d_model});
        auto attn_out = createLocalTensor({seq_len, d_model});
        auto residual_tmp = createLocalTensor({seq_len, d_model});

        // === Attention Block (delegated to backend) ===
        if (!executeAttentionBlock(layer_idx, input, weights, attn_norm_out, attn_out, metrics, cache_provider))
        {
            LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " attention failed");
            return false;
        }

        // Capture attention norm snapshot (populated by executeAttentionBlock)
        captureSnapshot(PipelineStage::ATTENTION_NORM, layer_idx, attn_norm_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // Capture attention output snapshot
        captureSnapshot(PipelineStage::ATTENTION_OUTPUT, layer_idx, attn_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // Attention residual connection
        {
            std::vector<std::shared_ptr<TensorBase>> residual_inputs = {input, attn_out};
            std::vector<std::shared_ptr<TensorBase>> residual_outputs = {residual_tmp};

            if (!executeKernel("residual", residual_inputs, residual_outputs))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " attention residual failed");
                return false;
            }
        }

        // Capture attention residual snapshot
        captureSnapshot(PipelineStage::ATTENTION_RESIDUAL, layer_idx, residual_tmp->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // === FFN Block (shared implementation) ===
        auto ffn_out = createLocalTensor({seq_len, d_model});
        if (!executeFfnBlock(layer_idx, residual_tmp, weights, ffn_out, metrics))
        {
            LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " FFN failed");
            return false;
        }

        // FFN residual connection
        {
            std::vector<std::shared_ptr<TensorBase>> final_residual_inputs = {residual_tmp, ffn_out};
            std::vector<std::shared_ptr<TensorBase>> final_residual_outputs = {output};

            if (!executeKernel("residual", final_residual_inputs, final_residual_outputs))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " FFN residual failed");
                return false;
            }
        }

        // Capture FFN residual snapshot
        captureSnapshot(PipelineStage::FFN_RESIDUAL, layer_idx, output->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        return true;
    }

    // ========================================================================
    // FFN block execution (shared - 95% identical across backends)
    // ========================================================================

    bool PrefillProviderBaseImpl::executeFfnBlock(
        int layer_idx,
        std::shared_ptr<TensorBase> &input,
        const QwenPipeline::ModelWeights &weights,
        std::shared_ptr<TensorBase> &output,
        PrefillMetrics &metrics)
    {
        PERF_SCOPED_TIMER("PrefillProviderBaseImpl::executeFfnBlock");

        const auto &layer_cfg = config().getLayerConfig();
        int seq_len = input->shape()[0];
        int d_model = layer_cfg.d_model;
        int d_ff = layer_cfg.d_ff;

        auto t_ffn_start = std::chrono::high_resolution_clock::now();

        // === FFN Normalization ===
        auto ffn_norm_out = createLocalTensor({seq_len, d_model});
        {
            auto t_norm_start = std::chrono::high_resolution_clock::now();

            std::vector<std::shared_ptr<TensorBase>> norm_inputs = {
                input,
                weights.ffn_norm_weight[layer_idx]};
            std::vector<std::shared_ptr<TensorBase>> norm_outputs = {ffn_norm_out};

            if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " FFN norm failed");
                return false;
            }

            auto t_norm_end = std::chrono::high_resolution_clock::now();
            metrics.norm_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                                   t_norm_end - t_norm_start)
                                   .count() /
                               1000.0;
        }

        // Capture FFN norm snapshot
        captureSnapshot(PipelineStage::FFN_NORM, layer_idx, ffn_norm_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        auto gate_out = createLocalTensor({seq_len, d_ff});
        auto up_out = createLocalTensor({seq_len, d_ff});

        // FFN fusion path: combine gate+up projections into single matmul
        const bool use_fusion = debugEnv().pipeline.ffn_fusion_enabled && weights.w_gate_up_fused[layer_idx];

        if (use_fusion)
        {
            // Fused gate+up projection
            auto fused_out = createLocalTensor({seq_len, 2 * d_ff});
            if (!executeLinearProjection(
                    ffn_norm_out,
                    weights.w_gate_up_fused[layer_idx],
                    fused_out,
                    seq_len,
                    2 * d_ff,
                    d_model,
                    /*is_prefill=*/true,
                    "ffn_fused_gate_up"))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " fused gate+up projection failed");
                return false;
            }

            // Split fused output into gate and up tensors
            const float *fused_data = fused_out->data();
            float *gate_data = const_cast<float *>(gate_out->data());
            float *up_data = const_cast<float *>(up_out->data());

#pragma omp parallel for
            for (int i = 0; i < seq_len; ++i)
            {
                const float *fused_row = fused_data + i * (2 * d_ff);
                float *gate_row = gate_data + i * d_ff;
                float *up_row = up_data + i * d_ff;

                std::memcpy(gate_row, fused_row, d_ff * sizeof(float));
                std::memcpy(up_row, fused_row + d_ff, d_ff * sizeof(float));
            }
        }
        else
        {
            // Original separate gate and up projections
            // === Gate Projection ===
            if (!executeLinearProjection(
                    ffn_norm_out,
                    weights.w_gate[layer_idx],
                    gate_out,
                    seq_len,
                    d_ff,
                    d_model,
                    /*is_prefill=*/true,
                    "ffn_gate"))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " gate projection failed");
                return false;
            }

            // === Up Projection ===
            if (!executeLinearProjection(
                    ffn_norm_out,
                    weights.w_up[layer_idx],
                    up_out,
                    seq_len,
                    d_ff,
                    d_model,
                    /*is_prefill=*/true,
                    "ffn_up"))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " up projection failed");
                return false;
            }
        }

        // Capture FFN gate snapshot
        captureSnapshot(PipelineStage::FFN_GATE, layer_idx, gate_out->data(), seq_len, d_ff);
        incrementSnapshotCounter(metrics);

        // Capture FFN up snapshot
        captureSnapshot(PipelineStage::FFN_UP, layer_idx, up_out->data(), seq_len, d_ff);
        incrementSnapshotCounter(metrics);

        // === SwiGLU Activation ===
        auto swiglu_out = createLocalTensor({seq_len, d_ff});
        {
            std::vector<std::shared_ptr<TensorBase>> swiglu_inputs = {gate_out, up_out};
            std::vector<std::shared_ptr<TensorBase>> swiglu_outputs = {swiglu_out};

            if (!executeKernel("swiglu", swiglu_inputs, swiglu_outputs))
            {
                LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " SwiGLU failed");
                return false;
            }
        }

        // Capture FFN SwiGLU snapshot
        captureSnapshot(PipelineStage::FFN_SWIGLU, layer_idx, swiglu_out->data(), seq_len, d_ff);
        incrementSnapshotCounter(metrics);

        // === Down Projection ===
        if (!executeLinearProjection(
                swiglu_out,
                weights.w_down[layer_idx],
                output,
                seq_len,
                d_model,
                d_ff,
                /*is_prefill=*/true,
                "ffn_down"))
        {
            LOG_ERROR("PrefillProviderBaseImpl: Layer " << layer_idx << " down projection failed");
            return false;
        }

        auto t_ffn_end = std::chrono::high_resolution_clock::now();
        metrics.ffn_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                              t_ffn_end - t_ffn_start)
                              .count() /
                          1000.0;

        // Capture FFN down snapshot
        captureSnapshot(PipelineStage::FFN_DOWN, layer_idx, output->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        return true;
    }

    // ========================================================================
    // Utility methods
    // ========================================================================

    std::shared_ptr<TensorBase> PrefillProviderBaseImpl::createLocalTensor(const std::vector<int> &shape)
    {
        return TensorFactory::create_simple(shape);
    }

    bool PrefillProviderBaseImpl::executeKernel(
        const std::string &kernel_name,
        const std::vector<std::shared_ptr<TensorBase>> &inputs,
        std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        auto *kernel = getKernel(kernel_name);
        if (!kernel)
        {
            LOG_ERROR("PrefillProviderBaseImpl: Kernel '" << kernel_name << "' not found");
            return false;
        }

        return kernel->execute(inputs, outputs);
    }

    MPIOperatorBase *PrefillProviderBaseImpl::getKernel(const std::string &name)
    {
        auto it = kernels_.find(name);
        if (it == kernels_.end())
        {
            return nullptr;
        }
        return it->second.get();
    }

    bool PrefillProviderBaseImpl::registerOperator(
        const std::string &name,
        std::unique_ptr<MPIOperatorBase> kernel)
    {
        if (kernels_.find(name) != kernels_.end())
        {
            LOG_WARN("PrefillProviderBaseImpl: Kernel '" << name << "' already registered");
            return false;
        }

        kernels_[name] = std::move(kernel);
        return true;
    }

} // namespace llaminar
