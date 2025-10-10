/**
 * @file cosma_prefill_provider.cpp
 * @brief Implementation of COSMA-based distributed prefill provider
 * @author David Sanftenberg
 */

#include "cosma_prefill_provider.h"
#include "qwen_pipeline_adapter.h"
#include "kernels/MPIRMSNormKernel.h"
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
    COSMAPrefillProvider::COSMAPrefillProvider(const ModelConfig &config, const MPIContext &mpi_ctx)
        : PrefillProvider(config, mpi_ctx)
    {
        initializeKernels();
    }

    void COSMAPrefillProvider::initializeKernels()
    {
        const auto &layer_cfg = config().getLayerConfig();

        // Attention kernel (used by COSMA path)
        {
            auto attention_kernel = std::make_unique<MPIAttentionKernel>(
                layer_cfg.n_head,
                layer_cfg.n_head_kv,
                layer_cfg.head_dim,
                layer_cfg.rope_freq_base);

            // Configure output mode
            attention_kernel->setOutputMode(MPIAttentionKernel::AttentionOutputMode::GatherHeadsPostProjection);

            // Wire up snapshot callback for intermediate stages
            attention_kernel->setSnapshotCallback([this](PipelineStage stage, int layer_idx, const float *data, int seq_len, int feature_dim)
                                                  { captureSnapshot(stage, layer_idx, data, seq_len, feature_dim); });

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

    bool COSMAPrefillProvider::execute(
        const std::vector<int> &tokens,
        const IModelWeights &weights,
        std::shared_ptr<TensorBase> &output,
        StageContext &ctx,
        PrefillMetrics &metrics)
    {
        PERF_SCOPED_TIMER("COSMAPrefillProvider::execute");

        // Reset metrics
        metrics.reset();
        metrics.backend_name = "COSMA";

        // Cast weights to concrete type
        const auto *qwen_model_weights = dynamic_cast<const QwenModelWeights *>(&weights);
        if (!qwen_model_weights)
        {
            LOG_ERROR("COSMAPrefillProvider: Invalid weights type (expected QwenModelWeights)");
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

        // Simple embedding lookup (host-side, no COSMA needed)
        auto embedded = createLocalTensor({seq_len, d_model});
        {
            const float *embed_weight = qwen_weights->token_embedding->data();
            float *embed_out = embedded->data();

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
            if (!executeTransformerLayer(layer_idx, layer_input, *qwen_weights, layer_output, metrics))
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " failed");
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
            // Use RMSNorm kernel for final norm
            std::vector<std::shared_ptr<TensorBase>> norm_inputs = {
                layer_input,
                qwen_weights->output_norm_weight};
            std::vector<std::shared_ptr<TensorBase>> norm_outputs = {final_norm_out};

            if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
            {
                LOG_ERROR("COSMAPrefillProvider: Final norm failed");
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
        {
            // Use adaptiveMatMul for LM head (may use COSMA for large ops)
            // Weight is [vocab_size, d_model], needs transpose to [d_model, vocab_size]
            bool ok = adaptiveMatMul(
                final_norm_out->data(),
                qwen_weights->lm_head->data(),
                logits->data(),
                seq_len,
                vocab_size,
                d_model,
                /*is_prefill=*/true,
                /*is_decode=*/false,
                /*transposed_a=*/false,
                /*transposed_b=*/true, // Weight stored as [vocab_size, d_model], transpose for matmul
                1.0f,
                0.0f);

            if (!ok)
            {
                LOG_ERROR("COSMAPrefillProvider: LM head failed");
                return false;
            }
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
            LOG_INFO("COSMAPrefillProvider: "
                     << seq_len << " tokens, "
                     << n_layers << " layers, "
                     << metrics.total_ms() << "ms total, "
                     << metrics.snapshots_captured << " snapshots");
        }

        return true;
    }

    bool COSMAPrefillProvider::executeTransformerLayer(
        int layer_idx,
        std::shared_ptr<TensorBase> &input,
        const QwenPipeline::ModelWeights &weights,
        std::shared_ptr<TensorBase> &output,
        PrefillMetrics &metrics)
    {
        PERF_SCOPED_TIMER("COSMAPrefillProvider::executeTransformerLayer");

        const auto &layer_cfg = config().getLayerConfig();
        int seq_len = input->shape()[0];
        int d_model = layer_cfg.d_model;
        int d_ff = layer_cfg.d_ff;

        // Allocate intermediate tensors
        auto attn_norm_out = createLocalTensor({seq_len, d_model});
        auto attn_out = createLocalTensor({seq_len, d_model});
        auto ffn_norm_out = createLocalTensor({seq_len, d_model});
        auto ffn_out = createLocalTensor({seq_len, d_model});
        auto residual_tmp = createLocalTensor({seq_len, d_model});

        // === Attention Block (COSMA path) ===
        {
            // Create plan for COSMA attention
            auto plan = plan_attention_prefill(seq_len, config(), mpiContext().size, mpiContext().rank);

            if (!plan.is_valid())
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " COSMA plan invalid: " << plan.rationale);
                return false;
            }

            PrefillAttentionTiming timing;
            if (!executeAttentionCosma(layer_idx, plan, input, weights, attn_norm_out, attn_out, timing))
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " COSMA attention failed");
                return false;
            }

            // Update metrics
            metrics.norm_ms += timing.norm_ms;
            metrics.attention_ms += timing.attention_ms + timing.linear_ms;
        }

        // Capture attention norm snapshot (already populated by executeAttentionCosma)
        captureSnapshot(PipelineStage::ATTENTION_NORM, layer_idx, attn_norm_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // Capture attention output snapshot
        captureSnapshot(PipelineStage::ATTENTION_OUTPUT, layer_idx, attn_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // Attention residual
        {
            std::vector<std::shared_ptr<TensorBase>> residual_inputs = {input, attn_out};
            std::vector<std::shared_ptr<TensorBase>> residual_outputs = {residual_tmp};

            if (!executeKernel("residual", residual_inputs, residual_outputs))
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " attention residual failed");
                return false;
            }
        }

        // Capture attention residual snapshot
        captureSnapshot(PipelineStage::ATTENTION_RESIDUAL, layer_idx, residual_tmp->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // === FFN Block ===
        auto t_ffn_start = std::chrono::high_resolution_clock::now();

        // FFN normalization
        {
            auto t_norm_start = std::chrono::high_resolution_clock::now();

            std::vector<std::shared_ptr<TensorBase>> norm_inputs = {
                residual_tmp,
                weights.ffn_norm_weight[layer_idx]};
            std::vector<std::shared_ptr<TensorBase>> norm_outputs = {ffn_norm_out};

            if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " FFN norm failed");
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

        // Gate and Up projections (via adaptiveMatMul - may use COSMA)
        auto gate_out = createLocalTensor({seq_len, d_ff});
        auto up_out = createLocalTensor({seq_len, d_ff});

        {
            bool ok = adaptiveMatMul(
                ffn_norm_out->data(),
                weights.w_gate[layer_idx]->data(),
                gate_out->data(),
                seq_len,
                d_ff,
                d_model,
                /*is_prefill=*/true,
                /*is_decode=*/false,
                /*transposed_a=*/false,
                /*transposed_b=*/true, // Weight is [d_ff, d_model], needs transpose to [d_model, d_ff]
                1.0f,
                0.0f);

            if (!ok)
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " gate projection failed");
                return false;
            }
        }

        // Capture FFN gate projection snapshot
        captureSnapshot(PipelineStage::FFN_GATE, layer_idx, gate_out->data(), seq_len, d_ff);
        incrementSnapshotCounter(metrics);

        {
            bool ok = adaptiveMatMul(
                ffn_norm_out->data(),
                weights.w_up[layer_idx]->data(),
                up_out->data(),
                seq_len,
                d_ff,
                d_model,
                /*is_prefill=*/true,
                /*is_decode=*/false,
                /*transposed_a=*/false,
                /*transposed_b=*/true, // Weight is [d_ff, d_model], needs transpose to [d_model, d_ff]
                1.0f,
                0.0f);

            if (!ok)
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " up projection failed");
                return false;
            }
        }

        // Capture FFN up projection snapshot
        captureSnapshot(PipelineStage::FFN_UP, layer_idx, up_out->data(), seq_len, d_ff);
        incrementSnapshotCounter(metrics);

        // SwiGLU activation
        auto swiglu_out = createLocalTensor({seq_len, d_ff});
        {
            std::vector<std::shared_ptr<TensorBase>> swiglu_inputs = {gate_out, up_out};
            std::vector<std::shared_ptr<TensorBase>> swiglu_outputs = {swiglu_out};

            if (!executeKernel("swiglu", swiglu_inputs, swiglu_outputs))
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " SwiGLU failed");
                return false;
            }
        }

        // Capture FFN SwiGLU activation snapshot
        captureSnapshot(PipelineStage::FFN_SWIGLU, layer_idx, swiglu_out->data(), seq_len, d_ff);
        incrementSnapshotCounter(metrics);

        // Down projection (via adaptiveMatMul - may use COSMA)
        {
            bool ok = adaptiveMatMul(
                swiglu_out->data(),
                weights.w_down[layer_idx]->data(),
                ffn_out->data(),
                seq_len,
                d_model,
                d_ff,
                /*is_prefill=*/true,
                /*is_decode=*/false,
                /*transposed_a=*/false,
                /*transposed_b=*/true, // Weight is [d_model, d_ff], needs transpose to [d_ff, d_model]
                1.0f,
                0.0f);

            if (!ok)
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " down projection failed");
                return false;
            }
        }

        auto t_ffn_end = std::chrono::high_resolution_clock::now();
        metrics.ffn_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                              t_ffn_end - t_ffn_start)
                              .count() /
                          1000.0;

        // Capture FFN down projection snapshot
        captureSnapshot(PipelineStage::FFN_DOWN, layer_idx, ffn_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // FFN residual
        {
            std::vector<std::shared_ptr<TensorBase>> final_residual_inputs = {residual_tmp, ffn_out};
            std::vector<std::shared_ptr<TensorBase>> final_residual_outputs = {output};

            if (!executeKernel("residual", final_residual_inputs, final_residual_outputs))
            {
                LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " final residual failed");
                return false;
            }
        }

        // Capture FFN residual snapshot
        captureSnapshot(PipelineStage::FFN_RESIDUAL, layer_idx, output->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        return true;
    }

    bool COSMAPrefillProvider::executeAttentionCosma(
        int layer_idx,
        const LargeMatmulPlan &plan,
        std::shared_ptr<TensorBase> &input,
        const QwenPipeline::ModelWeights &weights,
        std::shared_ptr<TensorBase> &attn_norm_out,
        std::shared_ptr<TensorBase> &attn_out,
        PrefillAttentionTiming &timing)
    {
        PERF_SCOPED_TIMER("COSMAPrefillProvider::executeAttentionCosma");

        // Validate plan
        if (!plan.is_valid())
        {
            LOG_ERROR("COSMAPrefillProvider::executeAttentionCosma called with invalid plan: " << plan.rationale);
            return false;
        }

        CosmaPrefillManager &manager = CosmaPrefillManager::instance();
        const int seq_len = plan.seq_len;
        const int hidden_size = plan.d_model;

        // --- Stage 1: RMSNorm (separate kernel, like OpenBLAS path) ---
        auto norm_start = std::chrono::high_resolution_clock::now();

        std::vector<std::shared_ptr<TensorBase>> norm_inputs = {
            input,
            weights.attn_norm_weight[layer_idx]};
        std::vector<std::shared_ptr<TensorBase>> norm_outputs = {attn_norm_out};

        if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
        {
            LOG_ERROR("COSMAPrefillProvider: Layer " << layer_idx << " attention norm failed");
            return false;
        }

        auto norm_end = std::chrono::high_resolution_clock::now();
        timing.norm_ms = std::chrono::duration<double, std::milli>(norm_end - norm_start).count();

        // --- Stage 2: Attention via MPIAttentionKernel with COSMA backend ---
        auto attention_start = std::chrono::high_resolution_clock::now();

        // Configure attention kernel
        auto attention_kernel = dynamic_cast<MPIAttentionKernel *>(getKernel("attention"));
        if (attention_kernel)
        {
            attention_kernel->setSequencePosition(n_past_);
            attention_kernel->setLayerIndex(layer_idx);
            attention_kernel->setCosmaManager(&manager); // INJECT COSMA BACKEND!
        }

        // Prepare KV cache tensors (use placeholder tensors since we don't have kv_cache in provider)
        const auto &layer_cfg = config().getLayerConfig();
        auto k_cache = createLocalTensor({seq_len, layer_cfg.n_head_kv * layer_cfg.head_dim});
        auto v_cache = createLocalTensor({seq_len, layer_cfg.n_head_kv * layer_cfg.head_dim});

        // Call attention kernel (SAME as OpenBLAS path!)
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

        auto attention_end = std::chrono::high_resolution_clock::now();
        timing.attention_ms = std::chrono::duration<double, std::milli>(attention_end - attention_start).count();

        // Note: Output projection is now handled inside MPIAttentionKernel
        // No need for separate adaptiveMatMul call
        timing.linear_ms = 0.0; // Included in attention_ms

        return true;
    }

    bool COSMAPrefillProvider::executeKernel(
        const std::string &kernel_name,
        const std::vector<std::shared_ptr<TensorBase>> &inputs,
        std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        auto *kernel = getKernel(kernel_name);
        if (!kernel)
        {
            LOG_ERROR("COSMAPrefillProvider: Kernel '" << kernel_name << "' not found");
            return false;
        }

        return kernel->execute(inputs, outputs);
    }

    MPIKernelBase *COSMAPrefillProvider::getKernel(const std::string &name)
    {
        auto it = kernels_.find(name);
        if (it == kernels_.end())
        {
            return nullptr;
        }
        return it->second.get();
    }

    bool COSMAPrefillProvider::registerKernel(
        const std::string &name,
        std::unique_ptr<MPIKernelBase> kernel)
    {
        if (kernels_.find(name) != kernels_.end())
        {
            LOG_WARN("COSMAPrefillProvider: Kernel '" << name << "' already registered");
            return false;
        }

        kernels_[name] = std::move(kernel);
        return true;
    }

    std::shared_ptr<TensorBase> COSMAPrefillProvider::createLocalTensor(const std::vector<int> &shape)
    {
        return TensorFactory::create_simple(shape);
    }

} // namespace llaminar
