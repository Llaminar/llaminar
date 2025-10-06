/**
 * @file openblas_prefill_provider.cpp
 * @brief Implementation of OpenBLAS-based prefill provider
 * @author David Sanftenberg
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
    OpenBLASPrefillProvider::OpenBLASPrefillProvider(const ModelConfig &config, const MPIContext &mpi_ctx)
        : PrefillProvider(config, mpi_ctx)
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

            // CRITICAL FIX: Must set GatherHeadsPostProjection mode for multi-rank execution!
            //
            // Background: MPIAttentionKernel defaults to LocalHeads mode, which was designed for
            // future tensor-parallel sharding where each rank owns a subset of heads. In that mode,
            // the kernel returns only the local rank's head contributions WITHOUT summing across ranks.
            //
            // Problem: For our current row-partitioned W_o implementation, each rank computes PARTIAL
            // contributions to ALL output dimensions. These must be summed via MPI_Allreduce.
            //
            // Without this setOutputMode() call:
            //   - Each rank returns only its partial W_o @ heads contribution
            //   - Missing other ranks' contributions → systematic negative bias
            //   - Cascading errors through all downstream layers
            //   - 98.6% parity test failure rate (145/147 checks failing)
            //
            // With GatherHeadsPostProjection mode:
            //   - Triggers MPI_Allreduce(MPI_SUM) to sum all ranks' contributions
            //   - Produces correct full attention output
            //   - All parity tests pass
            //
            // See: Investigation in docs/OPENBLAS_PREFILL_ROOT_CAUSE_ANALYSIS.md
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

    bool OpenBLASPrefillProvider::execute(
        const std::vector<int> &tokens,
        const IModelWeights &weights,
        std::shared_ptr<TensorBase> &output,
        StageContext &ctx,
        PrefillMetrics &metrics)
    {
        PERF_SCOPED_TIMER("OpenBLASPrefillProvider::execute");

        // Reset metrics
        metrics.reset();
        metrics.backend_name = "OpenBLAS";

        // Cast weights to concrete type
        const auto *qwen_model_weights = dynamic_cast<const QwenModelWeights *>(&weights);
        if (!qwen_model_weights)
        {
            LOG_ERROR("OpenBLASPrefillProvider: Invalid weights type (expected QwenModelWeights)");
            return false;
        }
        const auto *qwen_weights = &qwen_model_weights->inner;

        const auto &layer_cfg = config().getLayerConfig();
        int seq_len = static_cast<int>(tokens.size());
        int d_model = layer_cfg.d_model;
        int n_layers = layer_cfg.n_layers;

        // Update context
        ctx.seq_len = seq_len;

        // === Stage 1: Token Embedding ===
        auto t_embed_start = std::chrono::high_resolution_clock::now();

        auto embedded = createLocalTensor({seq_len, d_model});
        {
            std::vector<std::shared_ptr<TensorBase>> embed_inputs = {
                qwen_weights->token_embedding};
            std::vector<std::shared_ptr<TensorBase>> embed_outputs = {embedded};

            // Create temporary token tensor for embedding kernel
            auto token_tensor = createLocalTensor({static_cast<int>(tokens.size()), 1});
            float *token_data = token_tensor->data();
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                token_data[i] = static_cast<float>(tokens[i]);
            }
            embed_inputs.insert(embed_inputs.begin(), token_tensor);

            if (!executeKernel("embedding", embed_inputs, embed_outputs))
            {
                LOG_ERROR("OpenBLASPrefillProvider: Embedding failed");
                return false;
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
                LOG_ERROR("OpenBLASPrefillProvider: Layer " << layer_idx << " failed");
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
                LOG_ERROR("OpenBLASPrefillProvider: Final norm failed");
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

        int vocab_size = layer_cfg.vocab_size;
        auto logits = createLocalTensor({seq_len, vocab_size});
        {
            std::vector<std::shared_ptr<TensorBase>> lm_inputs = {
                final_norm_out,
                qwen_weights->lm_head};
            std::vector<std::shared_ptr<TensorBase>> lm_outputs = {logits};

            if (!executeKernel("linear", lm_inputs, lm_outputs))
            {
                LOG_ERROR("OpenBLASPrefillProvider: LM head failed");
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
            LOG_INFO("OpenBLASPrefillProvider: "
                     << seq_len << " tokens, "
                     << n_layers << " layers, "
                     << metrics.total_ms() << "ms total, "
                     << metrics.snapshots_captured << " snapshots");
        }

        return true;
    }

    bool OpenBLASPrefillProvider::executeTransformerLayer(
        int layer_idx,
        std::shared_ptr<TensorBase> &input,
        const QwenPipeline::ModelWeights &weights,
        std::shared_ptr<TensorBase> &output,
        PrefillMetrics &metrics)
    {
        PERF_SCOPED_TIMER("OpenBLASPrefillProvider::executeTransformerLayer");

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

        // === Attention Block ===
        auto t_attn_start = std::chrono::high_resolution_clock::now();

        // Attention normalization
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

        // Capture attention norm snapshot
        captureSnapshot(PipelineStage::ATTENTION_NORM, layer_idx, attn_norm_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // Attention computation
        {
            // Set sequence position for attention kernel (for RoPE)
            auto attention_kernel = dynamic_cast<MPIAttentionKernel *>(getKernel("attention"));
            if (attention_kernel)
            {
                attention_kernel->setSequencePosition(n_past_);
                attention_kernel->setLayerIndex(layer_idx);

                // Set snapshot callback to capture intermediate attention states
                // Hook snapshot callback for intermediate attention stages
                attention_kernel->setSnapshotCallback(
                    [this, layer_idx](PipelineStage stage, int layer, const float *data, int seq_len, int feature_dim)
                    {
                        // Log first few values for debugging (only layer 0)
                        if (layer == 0)
                        {
                            LOG_INFO("Layer " << layer << " stage=" << static_cast<int>(stage) << ": "
                                              << "shape=[" << seq_len << "," << feature_dim << "] "
                                              << "first_vals=[" << data[0] << "," << data[1] << "," << data[2] << "]");
                        }
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

            std::vector<std::shared_ptr<TensorBase>> attn_inputs = {
                attn_norm_out,
                weights.wq[layer_idx],
                weights.wk[layer_idx],
                weights.wv[layer_idx],
                weights.wo[layer_idx],
                k_cache,
                v_cache};
            std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out};

            if (!executeKernel("attention", attn_inputs, attn_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " attention failed");
                return false;
            }
        }

        auto t_attn_end = std::chrono::high_resolution_clock::now();
        metrics.attention_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                                    t_attn_end - t_attn_start)
                                    .count() /
                                1000.0;

        // Capture attention output snapshot
        captureSnapshot(PipelineStage::ATTENTION_OUTPUT, layer_idx, attn_out->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        // Attention residual
        {
            std::vector<std::shared_ptr<TensorBase>> residual_inputs = {input, attn_out};
            std::vector<std::shared_ptr<TensorBase>> residual_outputs = {residual_tmp};

            if (!executeKernel("residual", residual_inputs, residual_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " attention residual failed");
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
                LOG_ERROR("Layer " << layer_idx << " FFN norm failed");
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

        // Gate and Up projections
        auto gate_out = createLocalTensor({seq_len, d_ff});
        auto up_out = createLocalTensor({seq_len, d_ff});

        {
            std::vector<std::shared_ptr<TensorBase>> gate_inputs = {
                ffn_norm_out,
                weights.w_gate[layer_idx]};
            std::vector<std::shared_ptr<TensorBase>> gate_outputs = {gate_out};

            if (!executeKernel("linear", gate_inputs, gate_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " gate projection failed");
                return false;
            }
        }

        {
            std::vector<std::shared_ptr<TensorBase>> up_inputs = {
                ffn_norm_out,
                weights.w_up[layer_idx]};
            std::vector<std::shared_ptr<TensorBase>> up_outputs = {up_out};

            if (!executeKernel("linear", up_inputs, up_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " up projection failed");
                return false;
            }
        }

        // SwiGLU activation
        auto swiglu_out = createLocalTensor({seq_len, d_ff});
        {
            std::vector<std::shared_ptr<TensorBase>> swiglu_inputs = {gate_out, up_out};
            std::vector<std::shared_ptr<TensorBase>> swiglu_outputs = {swiglu_out};

            if (!executeKernel("swiglu", swiglu_inputs, swiglu_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " SwiGLU failed");
                return false;
            }
        }

        // Down projection
        {
            std::vector<std::shared_ptr<TensorBase>> down_inputs = {
                swiglu_out,
                weights.w_down[layer_idx]};
            std::vector<std::shared_ptr<TensorBase>> down_outputs = {ffn_out};

            if (!executeKernel("linear", down_inputs, down_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " down projection failed");
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
                LOG_ERROR("Layer " << layer_idx << " final residual failed");
                return false;
            }
        }

        // Capture FFN residual snapshot
        captureSnapshot(PipelineStage::FFN_RESIDUAL, layer_idx, output->data(), seq_len, d_model);
        incrementSnapshotCounter(metrics);

        return true;
    }

    bool OpenBLASPrefillProvider::executeKernel(
        const std::string &kernel_name,
        const std::vector<std::shared_ptr<TensorBase>> &inputs,
        std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        auto *kernel = getKernel(kernel_name);
        if (!kernel)
        {
            LOG_ERROR("OpenBLASPrefillProvider: Kernel '" << kernel_name << "' not found");
            return false;
        }

        return kernel->execute(inputs, outputs);
    }

    MPIKernelBase *OpenBLASPrefillProvider::getKernel(const std::string &name)
    {
        auto it = kernels_.find(name);
        if (it == kernels_.end())
        {
            return nullptr;
        }
        return it->second.get();
    }

    bool OpenBLASPrefillProvider::registerKernel(
        const std::string &name,
        std::unique_ptr<MPIKernelBase> kernel)
    {
        if (kernels_.find(name) != kernels_.end())
        {
            LOG_WARN("OpenBLASPrefillProvider: Kernel '" << name << "' already registered");
            return false;
        }

        kernels_[name] = std::move(kernel);
        return true;
    }

    std::shared_ptr<TensorBase> OpenBLASPrefillProvider::createLocalTensor(const std::vector<int> &shape)
    {
        return TensorFactory::create_simple(shape);
    }

} // namespace llaminar
