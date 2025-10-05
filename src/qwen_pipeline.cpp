/**
 * @file qwen_pipeline.cpp
 * @brief Canonical implementation of the QwenPipeline class.
 * @author David Sanftenberg
 *
 * This is the primary implementation translation unit for the Qwen-specific transformer pipeline.
 * It was migrated from the legacy distributed_transformer_pipeline.cpp.
 * Historical implementation details remain in the legacy file for reference.
 *
 * @section architecture Architecture Overview
 *
 * The QwenPipeline implements a multi-stage transformer inference engine
 * with support for:
 * - Multi-node MPI-based distribution
 * - NUMA-aware tensor placement
 * - Hybrid execution backends (OpenBLAS, COSMA)
 * - KV cache management with dynamic growth
 * - Incremental decode with optional parity validation
 *
 * @section parity_replay Parity Replay Design (Option A)
 *
 * The incremental decode path supports optional parity diagnostics through full prefix replay.
 * This design choice was made to address RoPE position encoding divergence issues.
 *
 * @subsection problem Problem Statement
 * Earlier implementations used minimal single-token replay (n_past=0), which produced large
 * RoPE divergences. The incremental path correctly rotated tokens at their absolute positions,
 * but the replay path incorrectly treated each token as position 0.
 *
 * @subsection solution Solution
 * Full prefix reconstruction ensures:
 * 1. **Identical RoPE positions** - Both paths see the same absolute token indices
 * 2. **Authentic KV context** - Accumulated key/value history matches exactly
 * 3. **Simplified validation** - No synthetic self-pairing logic needed for missing rows
 *
 * @subsection mechanics Implementation Mechanics
 *
 * The replay comparison proceeds as follows:
 * 1. Build `replay_seq` of size `n_past_` containing all prior tokens plus the new token
 * 2. Execute a fresh pipeline instance with `layer_token_diff` enabled to capture stage outputs
 * 3. Filter replay rows where `seq_len == replay_seq.size()` to isolate final token stages
 * 4. Pair incremental token rows with filtered replay rows (matched by stage + layer)
 * 5. Record first divergence exceeding threshold (rel_l2 > 1e-5) in global atomic flag
 *
 * @subsection tradeoffs Trade-offs
 *
 * **Cost**: O(prefix_length) computational overhead for each incremental token
 *
 * **Benefit**: Precise parity validation for regression testing and debugging
 *
 * **Mitigation**: This diagnostic path is gated behind debug environment flags
 * (`LLAMINAR_LAYER_TOKEN_DIFF` and `LLAMINAR_LAYER_REPLAY_COMPARE`).
 * Production incremental decode remains unchanged and incurs no overhead.
 */

#include "qwen_pipeline.h"
#include "qwen_pipeline_adapter.h" // For QwenModelWeights
#include "prefill_diagnostics.h"   // For baseline comparison and FFN tracing
#include "model_loader.h"
#include "kernels/MPISwiGLUKernel.h"
#include "kernels/MPIRoPEKernel.h"
#include "kernels/MPIResidualKernel.h"
#include "kernels/MPIEmbeddingKernel.h"
#include "kernels/common/rmsnorm_core.h"
#include "kernels/common/attention_primitives.h"
#include "tensors/tensor_factory.h"
#include "tensors/simple_tensor.h"
#include "debug_utils.h"
#include "performance_timer.h"
#include "cosma_prefill_manager.h"
#include "adaptive_matmul.h"
#include "backend_selector.h"
#include "matmul_backend_selection.h"
using llaminar::BackendContext;
using llaminar::BackendDecision;
using llaminar::MatMulBackendDecision;
using llaminar::MatMulBackendSelector;
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <algorithm>
#include "utils/debug_env.h"
#include <cblas.h>
#include <omp.h>
#include <sstream>
#include <filesystem>
#include <tuple>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace
{
    // (BufferStats, computeBufferStats, DiffSummary, computeDiffSummary moved to prefill_diagnostics.h/.cpp)

    // (parity sentinel accessors defined later after the atomic is declared)
    static std::atomic<bool> g_replay_first_exceed{false};
}
bool getReplayFirstExceedFlag() { return g_replay_first_exceed.load(); }
void resetReplayFirstExceedFlag() { g_replay_first_exceed.store(false); }

// === Section 1: Factory registration, statics, constructor, kernel registration, FFN shard diagnostics ===
namespace llaminar
{
    // Forward declaration of internal bridge (defined later in this TU)
    QwenPipeline::ModelWeights loadModelWeights_impl_bridge(
        ModelLoader &loader,
        const QwenPipeline::LayerConfig &config);
    // Factory helper implementation (migrated from legacy)
    std::unique_ptr<QwenPipeline> createQwenPipeline(const ModelConfig &config)
    {
        return std::make_unique<QwenPipeline>(config);
    }

    // (logFFNRowPreviewIfEnabled and isFFNShardTracingEnabledFor moved to prefill_diagnostics.h/.cpp)

    bool QwenPipeline::executeTransformerLayer(int layer_idx,
                                               std::shared_ptr<TensorBase> &input,
                                               const ModelWeights &weights,
                                               std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("QwenPipeline::executeTransformerLayer");
        int seq_len = input->shape()[0];

        // Initialize thread-local attention instrumentation context (consumed inside MPIAttentionKernel)
        // Only active when both global layer_token_diff (for outer diff collection) AND
        // attention.internal_diff are enabled. We snapshot minimal metadata to avoid extra lookups in kernel.
        struct AttnInternalCaptureContext
        {
            bool active = false; // gating
            int layer = -1;
            int seq_len = 0;
            int n_past = 0;
            QwenPipeline *pipeline = nullptr;
        };
        static thread_local AttnInternalCaptureContext g_attn_ctx; // thread-local to remain safe under possible OMP parallelism later
        g_attn_ctx.active = debugEnv().pipeline.layer_token_diff && debugEnv().attention.internal_diff && getRank() == 0;
        g_attn_ctx.layer = layer_idx;
        g_attn_ctx.seq_len = seq_len;
        g_attn_ctx.n_past = n_past_;
        g_attn_ctx.pipeline = this;
        // Expose lightweight accessor for kernel (forward declare below outside namespace for single TU linkage)
        auto set_attn_kernel_context = [&]()
        {
            // no-op placeholder; attention kernel fetches via weak external symbol
        };
        set_attn_kernel_context();
        auto attn_norm_out = createLocalTensor({seq_len, config_.getLayerConfig().d_model});
        auto attn_out = createLocalTensor({seq_len, config_.getLayerConfig().d_model});
        auto ffn_norm_out = createLocalTensor({seq_len, config_.getLayerConfig().d_model});
        auto ffn_out = createLocalTensor({seq_len, config_.getLayerConfig().d_model});
        auto residual_tmp = createLocalTensor({seq_len, config_.getLayerConfig().d_model});

        // Helper lambda for stage capture (last token row) to reduce duplication.
        auto capture_stage = [&](const std::shared_ptr<TensorBase> &tensor, const std::string &stage_label)
        {
            if (!tensor)
                return;
            if (!debugEnv().pipeline.layer_token_diff)
                return;
            if (getRank() != 0)
                return;
            if (!tensor->data())
                return;
            int rows = tensor->shape()[0];
            if (rows <= 0)
                return;
            int hidden = tensor->shape()[1];
            if (hidden <= 0)
                return;
            LayerTokenDiffRow row;
            row.layer = layer_idx;
            row.seq_len = rows;
            row.incremental = in_incremental_pass_;
            row.pipeline = this;
            row.stage = stage_label;
            row.values.assign(tensor->data() + (size_t)(rows - 1) * hidden, tensor->data() + (size_t)rows * hidden);
            last_layer_token_rows_.push_back(std::move(row));
            if (debugEnv().pipeline.layer_token_diff_verbose)
            {
                LOG_INFO("[LayerTokenCapture] pipe=" << this << " layer=" << layer_idx << " stage=" << stage_label << " rows=" << rows << " hidden=" << hidden << " total_rows=" << last_layer_token_rows_.size());
            }
        };

        const auto &abl = debugEnv().ablation;
        const auto &cap = debugEnv().layer_capture;
        static bool ablation_logged = false;
        if (!ablation_logged && getRank() == 0)
        {
            ablation_logged = true;
            LOG_INFO("[AblationConfig] attention=" << (abl.ablate_attention ? "ON" : "OFF") << " ffn=" << (abl.ablate_ffn ? "ON" : "OFF") << " capture=" << (cap.capture ? "ON" : "OFF"));
        }
        // Attention path
        if (!abl.ablate_attention)
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.getLayerConfig().d_model;
            bctx.n_layers = config_.getLayerConfig().n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                // Create plan for COSMA prefill attention
                auto plan = plan_attention_prefill(seq_len, config_, getSize(), getRank());
                if (!plan.is_valid())
                {
                    LOG_ERROR("Layer " << layer_idx << " COSMA plan validation failed: " << plan.rationale);
                    return false;
                }

                PrefillAttentionTiming timing;
                if (!executePrefillAttentionCosma(layer_idx, plan, input, weights, attn_norm_out, attn_out, timing))
                {
                    LOG_ERROR("Layer " << layer_idx << " COSMA attention failed");
                    return false;
                }
                total_norm_time_ += timing.norm_ms;
                total_attention_time_ += timing.attention_ms;
                total_linear_time_ += timing.linear_ms;
            }
            else
            {
                // Norm
                std::vector<std::shared_ptr<TensorBase>> norm_inputs = {input, weights.attn_norm_weight[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> norm_outputs = {attn_norm_out};
                if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
                {
                    LOG_ERROR("Layer " << layer_idx << " attention norm failed");
                    return false;
                }
                // Stage capture: post-attention norm (QKV input)
                capture_stage(attn_norm_out, "attn_qkv_in");
                // Parity capture: attention norm output (input to QKV)
                captureIfEnabled(PipelineStage::ATTENTION_NORM, layer_idx, attn_norm_out);
                // Attention kernel
                auto attention_kernel = dynamic_cast<MPIAttentionKernel *>(getKernel("attention"));
                if (attention_kernel)
                {
                    attention_kernel->setSequencePosition(n_past_);
                    attention_kernel->setLayerIndex(layer_idx);
                    if (debugEnv().pipeline.layer_token_diff_verbose && getRank() == 0)
                    {
                        LOG_INFO("[AttnKernelLayerSet] layer=" << layer_idx << " n_past=" << n_past_);
                    }
                }
                std::vector<std::shared_ptr<TensorBase>> attn_inputs = {attn_norm_out, weights.wq[layer_idx], weights.wk[layer_idx], weights.wv[layer_idx], weights.wo[layer_idx], use_kv_cache_ ? k_cache_[layer_idx] : createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim}), use_kv_cache_ ? v_cache_[layer_idx] : createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim})};
                std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out};
                if (!executeKernel("attention", attn_inputs, attn_outputs))
                {
                    LOG_ERROR("Layer " << layer_idx << " attention failed");
                    return false;
                }
                total_attention_time_ += 0; // (timing omitted for non-COSMA path placeholder)
            }
            // Stage capture: attention output
            capture_stage(attn_out, "attn_out");
            // Parity capture: attention output
            captureIfEnabled(PipelineStage::ATTENTION_OUTPUT, layer_idx, attn_out);
            // Residual add
            std::vector<std::shared_ptr<TensorBase>> residual_inputs = {input, attn_out};
            std::vector<std::shared_ptr<TensorBase>> residual_outputs = {residual_tmp};
            if (!executeKernel("residual", residual_inputs, residual_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " attention residual failed");
                return false;
            }
            // Stage capture: post-attention residual
            capture_stage(residual_tmp, "attn_residual");
            // Parity capture: post-attention residual
            captureIfEnabled(PipelineStage::ATTENTION_RESIDUAL, layer_idx, residual_tmp);
        }
        else
        {
            std::memcpy(residual_tmp->data(), input->data(), sizeof(float) * (size_t)seq_len * config_.getLayerConfig().d_model);
            std::memcpy(attn_out->data(), input->data(), sizeof(float) * (size_t)seq_len * config_.getLayerConfig().d_model);
            std::memset(attn_norm_out->data(), 0, sizeof(float) * (size_t)seq_len * config_.getLayerConfig().d_model);
            if (getRank() == 0)
                LOG_WARN("[Ablation] Layer " << layer_idx << " attention skipped");
            // Even in ablation, capture a synthetic attention path state for consistency
            capture_stage(attn_out, "attn_out");
            capture_stage(residual_tmp, "attn_residual");
        }
        // FFN
        if (abl.ablate_ffn)
        {
            std::memcpy(output->data(), residual_tmp->data(), sizeof(float) * (size_t)seq_len * config_.getLayerConfig().d_model);
            if (getRank() == 0)
                LOG_WARN("[Ablation] Layer " << layer_idx << " FFN skipped");
            return true;
        }
        std::vector<std::shared_ptr<TensorBase>> norm_inputs = {residual_tmp, weights.ffn_norm_weight[layer_idx]};
        std::vector<std::shared_ptr<TensorBase>> norm_outputs = {ffn_norm_out};
        if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " FFN norm failed");
            return false;
        }
        // Stage capture: FFN norm output
        capture_stage(ffn_norm_out, "ffn_norm");
        // Parity capture: FFN norm output
        captureIfEnabled(PipelineStage::FFN_NORM, layer_idx, ffn_norm_out);
        auto gate_out = createLocalTensor({seq_len, config_.getLayerConfig().d_ff});
        auto up_out = createLocalTensor({seq_len, config_.getLayerConfig().d_ff});
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.getLayerConfig().d_model;
            bctx.n_layers = config_.getLayerConfig().n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                if (!adaptiveMatMul(ffn_norm_out->data(), weights.w_gate[layer_idx]->data(), gate_out->data(), seq_len, config_.getLayerConfig().d_ff, config_.getLayerConfig().d_model, is_prefill_stage_, false, false, false, 1.0f, 0.0f))
                {
                    LOG_ERROR("gate projection failed cosma");
                    return false;
                }
            }
            else
            {
                std::vector<std::shared_ptr<TensorBase>> gate_inputs = {ffn_norm_out, weights.w_gate[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> gate_outputs = {gate_out};
                if (!executeKernel("linear", gate_inputs, gate_outputs))
                {
                    LOG_ERROR("gate projection failed");
                    return false;
                }
            }
        }
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.getLayerConfig().d_model;
            bctx.n_layers = config_.getLayerConfig().n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                if (!adaptiveMatMul(ffn_norm_out->data(), weights.w_up[layer_idx]->data(), up_out->data(), seq_len, config_.getLayerConfig().d_ff, config_.getLayerConfig().d_model, is_prefill_stage_, false, false, false, 1.0f, 0.0f))
                {
                    LOG_ERROR("up projection failed cosma");
                    return false;
                }
            }
            else
            {
                std::vector<std::shared_ptr<TensorBase>> up_inputs = {ffn_norm_out, weights.w_up[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> up_outputs = {up_out};
                if (!executeKernel("linear", up_inputs, up_outputs))
                {
                    LOG_ERROR("up projection failed");
                    return false;
                }
            }
        }
        auto swiglu_out = createLocalTensor({seq_len, config_.getLayerConfig().d_ff});
        {
            std::vector<std::shared_ptr<TensorBase>> sw_in = {gate_out, up_out};
            std::vector<std::shared_ptr<TensorBase>> sw_out = {swiglu_out};
            if (!executeKernel("swiglu", sw_in, sw_out))
            {
                LOG_ERROR("SwiGLU failed layer=" << layer_idx);
                return false;
            }
        }
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.getLayerConfig().d_model;
            bctx.n_layers = config_.getLayerConfig().n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                if (!adaptiveMatMul(swiglu_out->data(), weights.w_down[layer_idx]->data(), ffn_out->data(), seq_len, config_.getLayerConfig().d_model, config_.getLayerConfig().d_ff, is_prefill_stage_, false, false, false, 1.0f, 0.0f))
                {
                    LOG_ERROR("down projection failed cosma");
                    return false;
                }
            }
            else
            {
                std::vector<std::shared_ptr<TensorBase>> down_inputs = {swiglu_out, weights.w_down[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> down_outputs = {ffn_out};
                if (!executeKernel("linear", down_inputs, down_outputs))
                {
                    LOG_ERROR("down projection failed");
                    return false;
                }
            }
        }
        // Stage capture: FFN output before residual
        capture_stage(ffn_out, "ffn_out");
        // Parity capture: FFN down projection output (before final residual)
        captureIfEnabled(PipelineStage::FFN_DOWN, layer_idx, ffn_out);
        std::vector<std::shared_ptr<TensorBase>> final_residual_inputs = {residual_tmp, ffn_out};
        std::vector<std::shared_ptr<TensorBase>> final_residual_outputs = {output};
        if (!executeKernel("residual", final_residual_inputs, final_residual_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " final residual failed");
            return false;
        }
        // Diagnostics: capture last token row if enabled
        capture_stage(output, "layer_output");
        // Parity capture: final layer output (after FFN residual)
        captureIfEnabled(PipelineStage::FFN_RESIDUAL, layer_idx, output);
        return true;
    }

    // --- Helpers migrated from legacy TU (minimal subset) ---
    // Local minimal layout enforcement used during migration. Avoids depending on legacy TU symbol.
    static void enforce_matrix_layout_compat(std::shared_ptr<llaminar::TensorBase> &tensor,
                                             int expected_rows,
                                             int expected_cols,
                                             const std::string &label)
    {
        if (!tensor)
            throw std::runtime_error(label + " tensor null");
        const auto &shape = tensor->shape();
        if (shape.size() != 2)
            throw std::runtime_error(label + " dim=" + std::to_string(shape.size()));
        // Accept if matches
        if (shape[0] == expected_rows && shape[1] == expected_cols)
            return;
        // Transpose if flipped
        if (shape[0] == expected_cols && shape[1] == expected_rows)
        {
            const float *src = tensor->data();
            std::vector<float> transposed((size_t)expected_rows * expected_cols);
            for (int r = 0; r < expected_rows; ++r)
                for (int c = 0; c < expected_cols; ++c)
                    transposed[(size_t)r * expected_cols + c] = src[(size_t)c * expected_rows + r];
            tensor = std::make_shared<llaminar::SimpleTensor>(std::vector<int>{expected_rows, expected_cols}, transposed);
            LOG_INFO(label << " transposed to " << expected_rows << "x" << expected_cols);
            return;
        }
        LOG_WARN(label << " unexpected shape [" << shape[0] << "," << shape[1] << "] expected [" << expected_rows << "," << expected_cols << "] or transpose");
    }
    // (isFFNShardTracingEnabledFor and PrefillBaselineRegistry moved to prefill_diagnostics.h/.cpp)

    // DEPRECATED: Old registration function - use qwen_pipeline_adapter.cpp version instead
    // static std::once_flag qwen_register_flag;
    // void registerQwenPipeline()
    // {
    //     std::call_once(qwen_register_flag, []()
    //                    { PipelineFactory::instance().registerCreator("qwen", [](const ModelConfig &cfg) -> std::unique_ptr<AbstractPipeline>
    //                                                                  {
    // 			auto impl = std::make_unique<QwenPipeline>(cfg);
    // 			return std::unique_ptr<AbstractPipeline>(impl.release()); }); });
    // }

    std::atomic<size_t> QwenPipeline::small_seq_fast_path_calls_{0};
    std::vector<float> QwenPipeline::last_pre_lm_hidden_;
    std::vector<QwenPipeline::LayerActivationStat> QwenPipeline::last_layer_stats_;
    std::vector<QwenPipeline::LayerTokenDiffRow> QwenPipeline::last_layer_token_rows_;
    // (parity sentinel atomic declared globally below outside namespace for accessor simplicity)

    QwenPipeline::QwenPipeline(const ModelConfig &config)
        : PipelineBase(), config_(config), use_kv_cache_(true), n_past_(0),
          total_embedding_time_(0.0), total_attention_time_(0.0), total_linear_time_(0.0),
          total_norm_time_(0.0), total_activation_time_(0.0), total_communication_time_(0.0)
    {
        initializeKernels();
        kv_cache_dynamic_init_ = debugEnv().kv_cache.dynamic_init;
        if (getRank() == 0)
        {
            LOG_INFO("[KVCacheInitMode] constructing pipeline this=" << (const void *)this << " dynamic_init=" << (kv_cache_dynamic_init_ ? "on" : "off"));
        }

        // (accessor implementations defined at TU end)

        if (use_kv_cache_)
        {
            int initial_capacity = config_.getLayerConfig().max_seq_len;
            if (kv_cache_dynamic_init_)
                initial_capacity = 1; // defer sizing
            initializeKVCache(initial_capacity);
            if (getRank() == 0)
            {
                LOG_INFO("[KVCacheInitMode] post-initial-capacity this=" << (const void *)this << " capacity_tokens=" << kv_cache_state_.capacity_tokens);
            }
        }
        LOG_INFO("QwenPipeline initialized on rank " << getRank() << "/" << getSize()
                                                     << " with arch='" << config_.architecture << "', " << config_.getLayerConfig().n_layers
                                                     << " layers, " << config_.getLayerConfig().n_head << " heads");
    }

    QwenPipeline::QwenPipeline(const ModelConfig &config, const MPIContext &ctx)
        : PipelineBase(ctx), config_(config), use_kv_cache_(true), n_past_(0),
          total_embedding_time_(0.0), total_attention_time_(0.0), total_linear_time_(0.0),
          total_norm_time_(0.0), total_activation_time_(0.0), total_communication_time_(0.0)
    {
        initializeKernels();
        kv_cache_dynamic_init_ = debugEnv().kv_cache.dynamic_init;
        if (getRank() == 0)
        {
            LOG_INFO("[KVCacheInitMode] constructing pipeline with MPIContext this=" << (const void *)this << " dynamic_init=" << (kv_cache_dynamic_init_ ? "on" : "off"));
        }

        if (use_kv_cache_)
        {
            int initial_capacity = config_.getLayerConfig().max_seq_len;
            if (kv_cache_dynamic_init_)
                initial_capacity = 1; // defer sizing
            initializeKVCache(initial_capacity);
            if (getRank() == 0)
            {
                LOG_INFO("[KVCacheInitMode] post-initial-capacity this=" << (const void *)this << " capacity_tokens=" << kv_cache_state_.capacity_tokens);
            }
        }
        LOG_INFO("QwenPipeline initialized with " << mpi_ctx_.toString()
                                                  << " arch='" << config_.architecture << "', " << config_.getLayerConfig().n_layers
                                                  << " layers, " << config_.getLayerConfig().n_head << " heads");
    }

    QwenPipeline::~QwenPipeline() = default;

    void QwenPipeline::initializeKernels()
    {
        {
            auto embedding_kernel = std::make_unique<MPIEmbeddingKernel>(config_.getLayerConfig().vocab_size, config_.getLayerConfig().d_model);
            if (!registerKernel("embedding", std::move(embedding_kernel)))
                throw std::runtime_error("Failed to register Embedding kernel");
        }
        auto rmsnorm_kernel = std::make_unique<MPIRMSNormKernel>(MPIRMSNormKernel::DistributionStrategy::SEQUENCE_WISE);
        rmsnorm_kernel->setEpsilon(config_.getLayerConfig().eps);
        if (!registerKernel("rmsnorm", std::move(rmsnorm_kernel)))
            throw std::runtime_error("Failed to register RMSNorm kernel");

        auto attention_kernel = std::make_unique<MPIAttentionKernel>(config_.getLayerConfig().n_head, config_.getLayerConfig().n_head_kv, config_.getLayerConfig().head_dim, config_.getLayerConfig().rope_freq_base);
        if (!registerKernel("attention", std::move(attention_kernel)))
            throw std::runtime_error("Failed to register Attention kernel");

        auto linear_kernel = std::make_unique<MPILinearKernel>();
        if (!registerKernel("linear", std::move(linear_kernel)))
            throw std::runtime_error("Failed to register Linear kernel");

        auto swiglu_kernel = std::make_unique<MPISwiGLUKernel>(MPISwiGLUKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("swiglu", std::move(swiglu_kernel)))
            throw std::runtime_error("Failed to register SwiGLU kernel");

        auto rope_kernel = std::make_unique<MPIRoPEKernel>(config_.getLayerConfig().max_seq_len, config_.getLayerConfig().head_dim, config_.getLayerConfig().rope_freq_base, MPIRoPEKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("rope", std::move(rope_kernel)))
            throw std::runtime_error("Failed to register RoPE kernel");

        auto residual_kernel = std::make_unique<MPIResidualKernel>(MPIResidualKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("residual", std::move(residual_kernel)))
            throw std::runtime_error("Failed to register Residual kernel");

        LOG_DEBUG("QwenPipeline: Registered " << getKernelNames().size() << " kernels on rank " << getRank());
    }

    /**
     * @brief Trace diagnostics for an FFN shard buffer (stats + optional samples + baseline compare).
     *
     * Readability refactor: expanded spacing, explicit braces, grouped logical blocks.
     *
     * Flow:
     *  1. Guard conditions (env enabled, data valid, dims > 0, label allow‑list)
     *  2. Compute stats for current shard
     *  3. (Rank 0) Optional baseline fetch & diff stats
     *  4. Emit summary header line
     *  5. Optionally emit sampled rows (bounded by cfg.limit)
     */
    void QwenPipeline::traceFFNShardDiagnostics(const std::string &label,
                                                const float *data,
                                                int seq_len,
                                                int feature_dim)
    {
        // --- Stage 1: Guards ---
        const auto &cfg = debugEnv().ffn_shard_trace;

        if (!cfg.enabled || !data)
        {
            return;
        }

        if (seq_len <= 0 || feature_dim <= 0)
        {
            return;
        }

        if (!isFFNShardTracingEnabledFor(label))
        {
            return;
        }

        // --- Stage 2: Basic stats ---
        const int sample_limit = (cfg.limit > 0) ? cfg.limit : 1;

        const int rank = getRank(); // Use cached rank from MPIContext

        const size_t total_elements = static_cast<size_t>(seq_len) * static_cast<size_t>(feature_dim);
        auto stats = computeBufferStats(data, total_elements);

        // --- Stage 3: Baseline diff (rank 0 only) ---
        const bool baseline_enabled = debugEnv().baseline.compare;
        std::vector<float> baseline_buffer;
        const float *baseline_ptr = nullptr;
        BufferStats baseline_stats{}; // (optional – retained for future extended diff logging)

        if (rank == 0 && baseline_enabled)
        {
            auto &registry = PrefillBaselineRegistry::instance();
            if (registry.fetch(label, baseline_buffer) && baseline_buffer.size() == total_elements)
            {
                baseline_ptr = baseline_buffer.data();
                baseline_stats = computeBufferStats(baseline_ptr, total_elements);
            }
            else
            {
                LOG_DEBUG("[PrefillFFNTrace] baseline unavailable for shard '" << label << "'");
            }
        }

        // --- Stage 4: Summary header ---
        std::ostringstream header;
        header << "[PrefillFFNTrace] rank=" << rank
               << " shard=" << label
               << " shape=(" << seq_len << "," << feature_dim << ")"
               << " min=" << stats.min
               << " max=" << stats.max
               << " mean=" << stats.mean
               << " rms=" << stats.rms
               << " stddev=" << stats.stddev;

        if (baseline_ptr)
        {
            auto diff = computeDiffSummary(data, baseline_ptr, total_elements);
            header << " rel_l2=" << diff.rel_l2
                   << " mean_abs=" << diff.mean_abs
                   << " max_abs=" << diff.max_abs;
        }

        if (rank == 0)
        {
            LOG_INFO(header.str());
        }

        // --- Stage 5: Sample rows (rank 0 only) ---
        if (cfg.limit > 0 && rank == 0)
        { // legacy: using limit as sample trigger
            const int rows = std::min(seq_len, sample_limit);
            const int cols = std::min(feature_dim, sample_limit);

            for (int r = 0; r < rows; ++r)
            {
                std::ostringstream row_ss;
                row_ss << "[PrefillFFNSample] shard=" << label
                       << " row=" << r
                       << " vals=";

                const float *row_ptr = data + (size_t)r * feature_dim;
                for (int c = 0; c < cols; ++c)
                {
                    row_ss << row_ptr[c];
                    if (c + 1 < cols)
                    {
                        row_ss << ',';
                    }
                }
                LOG_INFO(row_ss.str());
            }
        }
    }
} // namespace llaminar

// Out-of-line full weight loader implementation.
namespace llaminar
{
    QwenPipeline::ModelWeights loadModelWeights_impl_bridge(
        ModelLoader &loader,
        const QwenPipeline::LayerConfig &config)
    {
        QwenPipeline::ModelWeights weights;
        LOG_INFO("[WeightLoad] begin vocab=" << config.vocab_size << " d_model=" << config.d_model << " layers=" << config.n_layers);

        // === Token Embedding ===
        // Load vocabulary embedding matrix (vocab_size x d_model)
        weights.token_embedding = loader.loadTensor("token_embd.weight");

        // Log embedding shape for verification
        const auto emb_shape = weights.token_embedding->shape();
        if (emb_shape.size() == 2)
        {
            LOG_INFO("[WeightLoad] token_embd.weight shape=" << emb_shape[0] << "x" << emb_shape[1]);
        }

        // Lightweight anomaly detection on sample of embedding values
        {
            const float *ptr = weights.token_embedding->data();
            size_t total = (size_t)weights.token_embedding->size();
            size_t sample = std::min<size_t>(total, (size_t)512);

            // Count NaN and Inf values in sample
            size_t nan_ct = 0, inf_ct = 0;
            for (size_t i = 0; i < sample; ++i)
            {
                float v = ptr[i];
                if (std::isnan(v))
                    ++nan_ct;
                if (std::isinf(v))
                    ++inf_ct;
            }

            // Warn if anomalies detected
            if (nan_ct || inf_ct)
                LOG_WARN("[WeightLoad] token_embd anomalies nan=" << nan_ct << " inf=" << inf_ct);
        }

        // === Output Normalization Weight ===
        // Load final RMSNorm gamma parameters
        weights.output_norm_weight = loader.loadTensor("output_norm.weight");
        if (!weights.output_norm_weight)
            throw std::runtime_error("Failed to load output_norm.weight");

        // Validate dimensions
        if (weights.output_norm_weight->size() != config.d_model)
        {
            LOG_WARN("[WeightLoad] output_norm.weight size=" << weights.output_norm_weight->size()
                                                             << " != d_model=" << config.d_model);
        }

        // Debug override: force unit gamma (disable normalization scaling)
        if (debugEnv().output_norm.force_unit || debugEnv().output_norm.force_unit_all)
        {
            float *g = const_cast<float *>(weights.output_norm_weight->data());
            for (int i = 0; i < weights.output_norm_weight->size(); ++i)
                g[i] = 1.0f;
            LOG_WARN("[WeightLoad] Forced output_norm to unit gamma");
        }
        // Debug override: clamp gamma to reasonable range
        else if (debugEnv().output_norm.clamp)
        {
            float *g = const_cast<float *>(weights.output_norm_weight->data());
            for (int i = 0; i < weights.output_norm_weight->size(); ++i)
                g[i] = std::clamp(g[i], 0.0f, 4.0f);
            LOG_WARN("[WeightLoad] Clamped output_norm gamma range [0,4]");
        }

        // === LM Head (Language Model Output Projection) ===
        // Attempt to load dedicated output projection weight
        std::shared_ptr<TensorBase> lm_head;
        try
        {
            lm_head = loader.loadTensor("output.weight");
        }
        catch (const std::exception &e)
        {
            LOG_WARN("[WeightLoad] output.weight load exc: " << e.what());
        }

        if (lm_head)
        {
            // Successfully loaded dedicated LM head
            try
            {
                // Check if debug override bypasses orientation enforcement
                bool raw = debugEnv().lm_head.raw_orientation;

                // Enforce expected layout (d_model x vocab_size) unless raw mode
                if (!raw)
                    enforce_matrix_layout_compat(lm_head, config.d_model, config.vocab_size, "output.weight");

                weights.lm_head = lm_head;
            }
            catch (const std::exception &e)
            {
                // Orientation correction failed, fall back to tied embeddings
                LOG_WARN("[WeightLoad] lm_head orientation issue: " << e.what()
                                                                    << " using tied embedding");
                weights.lm_head = weights.token_embedding;
            }
        }
        else
        {
            // No dedicated LM head found, use weight tying (common in many models)
            LOG_WARN("[WeightLoad] output.weight missing; using tied embeddings as LM head");
            weights.lm_head = weights.token_embedding;
        }

        // === Per-Layer Weight Vectors ===
        // Pre-allocate storage for all layer weights to avoid reallocation
        weights.attn_norm_weight.reserve(config.n_layers);
        weights.wq.reserve(config.n_layers);
        weights.wk.reserve(config.n_layers);
        weights.wv.reserve(config.n_layers);
        weights.wo.reserve(config.n_layers);
        weights.ffn_norm_weight.reserve(config.n_layers);
        weights.w_gate.reserve(config.n_layers);
        weights.w_up.reserve(config.n_layers);
        weights.w_down.reserve(config.n_layers);

        LOG_INFO("[WeightLoad] per-layer loading start layers=" << config.n_layers);
        for (int layer = 0; layer < config.n_layers; ++layer)
        {
            // Construct tensor name prefix for this layer
            std::string prefix = "blk." + std::to_string(layer) + ".";

            // --- Attention Normalization ---
            // Load pre-attention RMSNorm gamma
            auto attn_norm = loader.loadTensor(prefix + "attn_norm.weight");
            if (!attn_norm)
                throw std::runtime_error("Failed to load " + prefix + "attn_norm.weight");

            // Debug override: force unit gamma if requested
            if (debugEnv().output_norm.force_unit_all)
            {
                float *g = const_cast<float *>(attn_norm->data());
                for (int i = 0; i < attn_norm->size(); ++i)
                    g[i] = 1.0f;
            }
            weights.attn_norm_weight.push_back(attn_norm);

            // --- Attention Projection Matrices ---
            // Load Q, K, V, and output projection weights
            auto wq = loader.loadTensor(prefix + "attn_q.weight");      // Query projection
            auto wk = loader.loadTensor(prefix + "attn_k.weight");      // Key projection
            auto wv = loader.loadTensor(prefix + "attn_v.weight");      // Value projection
            auto wo = loader.loadTensor(prefix + "attn_output.weight"); // Output projection

            // Validate all attention weights loaded successfully
            if (!wq || !wk || !wv || !wo)
                throw std::runtime_error("Failed to load attention weights for layer " + std::to_string(layer));

            // Store attention weights for this layer
            weights.wq.push_back(wq);
            weights.wk.push_back(wk);
            weights.wv.push_back(wv);
            weights.wo.push_back(wo);

            // --- FFN Normalization ---
            // Load pre-FFN RMSNorm gamma
            auto ffn_norm = loader.loadTensor(prefix + "ffn_norm.weight");
            if (!ffn_norm)
                throw std::runtime_error("Failed to load " + prefix + "ffn_norm.weight");

            // Debug override: force unit gamma if requested
            if (debugEnv().output_norm.force_unit_all)
            {
                float *g = const_cast<float *>(ffn_norm->data());
                for (int i = 0; i < ffn_norm->size(); ++i)
                    g[i] = 1.0f;
            }
            weights.ffn_norm_weight.push_back(ffn_norm);

            // --- FFN Projection Matrices (SwiGLU) ---
            // Load gate, up, and down projection weights for SwiGLU activation
            auto w_gate = loader.loadTensor(prefix + "ffn_gate.weight"); // Gate projection
            auto w_up = loader.loadTensor(prefix + "ffn_up.weight");     // Up projection
            auto w_down = loader.loadTensor(prefix + "ffn_down.weight"); // Down projection

            // Validate all FFN weights loaded successfully
            if (!w_gate || !w_up || !w_down)
                throw std::runtime_error("Failed to load FFN weights for layer " + std::to_string(layer));

            // Store FFN weights for this layer
            weights.w_gate.push_back(w_gate);
            weights.w_up.push_back(w_up);
            weights.w_down.push_back(w_down);
        }
        LOG_INFO("[WeightLoad] per-layer loading complete");
        return weights;

        // (Helper forward declarations below weight loader section)
    }

    // (handle_prefill_stage_snapshot moved to prefill_diagnostics.h/.cpp)

    // Minimal row-spec parser (range/list) used for embedding traces if central util not yet migrated.
    static std::vector<int> parseSimpleRowSpec(const std::string &spec, int max_rows)
    {
        std::vector<int> rows;
        if (spec.empty() || max_rows <= 0)
            return rows;
        std::stringstream ss(spec);
        std::string tok;
        std::unordered_set<int> seen;
        while (std::getline(ss, tok, ','))
        {
            if (tok.empty())
                continue;
            size_t dash = tok.find('-');
            auto toInt = [&](const std::string &s) -> int
            { try { return std::stoi(s); } catch(...) { return -1; } };
            if (dash == std::string::npos)
            {
                int v = toInt(tok);
                if (v >= 0 && v < max_rows && !seen.count(v))
                {
                    rows.push_back(v);
                    seen.insert(v);
                }
            }
            else
            {
                int a = toInt(tok.substr(0, dash));
                int b = toInt(tok.substr(dash + 1));
                if (a > b)
                    std::swap(a, b);
                if (a < 0)
                    a = 0;
                if (b >= max_rows)
                    b = max_rows - 1;
                for (int v = a; v <= b; ++v)
                    if (!seen.count(v))
                    {
                        rows.push_back(v);
                        seen.insert(v);
                    }
            }
        }
        std::sort(rows.begin(), rows.end());
        return rows;
    }

    bool QwenPipeline::executeEmbedding(const std::vector<int> &token_ids,
                                        const std::shared_ptr<TensorBase> &embedding_weight,
                                        std::shared_ptr<TensorBase> &embedded_output)
    {
        PERF_SCOPED_TIMER("QwenPipeline::executeEmbedding");
        int seq_len = (int)token_ids.size();
        if (seq_len <= 0)
        {
            LOG_ERROR("executeEmbedding: empty token_ids");
            return false;
        }
        if (!embedded_output || embedded_output->shape().size() != 2 || embedded_output->shape()[0] != seq_len || embedded_output->shape()[1] != config_.getLayerConfig().d_model)
        {
            embedded_output = createLocalTensor({seq_len, config_.getLayerConfig().d_model});
        }
        // Rank 0 performs embedding lookup using registered kernel
        if (getRank() == 0)
        {
            auto token_ids_tensor = createLocalTensor({seq_len});
            for (int i = 0; i < seq_len; ++i)
                token_ids_tensor->data()[i] = (float)token_ids[i];
            std::vector<std::shared_ptr<TensorBase>> inputs = {token_ids_tensor, embedding_weight};
            std::vector<std::shared_ptr<TensorBase>> outputs = {embedded_output};
            ASSERT_TENSOR_VALID(token_ids_tensor, "Embedding token_ids");
            ASSERT_TENSOR_VALID(embedding_weight, "Embedding weight");
            TensorLogger::logTensorStats(token_ids_tensor, "token_ids", "EMBEDDING_INPUT");
            TensorLogger::logTensorStats(embedding_weight, "embedding_weight", "EMBEDDING_INPUT");
            if (!executeKernel("embedding", inputs, outputs))
            {
                LOG_ERROR("Embedding kernel execution failed");
                return false;
            }
            ASSERT_TENSOR_NOT_NAN(embedded_output, "Embedding output");
            TensorLogger::logTensorStats(embedded_output, "embedded_output", "EMBEDDING_OUTPUT");
        }
        // Broadcast embedded_output
        if (!broadcastTensor(embedded_output, 0))
        {
            LOG_ERROR("Embedding broadcast failed");
            return false;
        }

        // Optional row trace
        if (!debugEnv().embedding.trace_rows_spec.empty())
        {
            auto rows = parseSimpleRowSpec(debugEnv().embedding.trace_rows_spec, embedded_output->shape()[0]);
            if (!rows.empty())
                logTensorRowPreview(embedded_output, "embedding_output", rows, 8, "EMBEDDING_TRACE");
        }
        // Baseline snapshot
        const bool capture_baseline = debugEnv().baseline.capture;
        const bool compare_baseline = debugEnv().baseline.compare;
        if (capture_baseline || compare_baseline)
        {
            handle_prefill_stage_snapshot(getRank(), "embedding_output", embedded_output->data(), (size_t)embedded_output->size(), config_.getLayerConfig().d_model, 5e-4, capture_baseline, compare_baseline);
        }

        // Parity capture: embedding output
        captureIfEnabled(PipelineStage::EMBEDDING, -1, embedded_output);

        return true;
    }

    // Simple intermediate tensor factory
    std::vector<std::shared_ptr<TensorBase>> QwenPipeline::createIntermediateTensors(int seq_len)
    {
        return {createLocalTensor({seq_len, config_.getLayerConfig().d_model}), createLocalTensor({seq_len, config_.getLayerConfig().d_model})};
    }

    bool QwenPipeline::executeOutputProjection(std::shared_ptr<TensorBase> &input,
                                               const ModelWeights &weights,
                                               std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("QwenPipeline::executeOutputProjection");
        int seq_len = input->shape()[0];

        // Debug instrumentation: trace entry for layer 0 when layer token diff enabled to diagnose
        // missing replay capture rows in incremental parity test. This is temporary and can be
        // removed once root cause is identified.
        if (getRank() == 0 && debugEnv().pipeline.layer_token_diff)
        {
            LOG_INFO("[LayerTokenDiffTrace] output_projection_enter pipeline=" << this
                                                                               << " is_prefill=" << (is_prefill_stage_ ? 1 : 0)
                                                                               << " use_kv=" << (use_kv_cache_ ? 1 : 0)
                                                                               << " existing_rows=" << last_layer_token_rows_.size());
        }
        // Final norm
        auto normed = createLocalTensor({seq_len, config_.getLayerConfig().d_model});
        std::vector<std::shared_ptr<TensorBase>> norm_inputs = {input, weights.output_norm_weight};
        std::vector<std::shared_ptr<TensorBase>> norm_outputs = {normed};
        if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
        {
            LOG_ERROR("Output projection norm failed");
            return false;
        }

        // Parity capture: final norm output
        captureIfEnabled(PipelineStage::FINAL_NORM, -1, normed);

        // Keep last hidden (rank0) for diagnostics
        if (getRank() == 0)
        {
            size_t elems = (size_t)seq_len * (size_t)config_.getLayerConfig().d_model;
            last_pre_lm_hidden_.assign(normed->data(), normed->data() + elems);
        }
        // Allocate output logits
        if (!output || output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != config_.getLayerConfig().vocab_size)
        {
            output = createLocalTensor({seq_len, config_.getLayerConfig().vocab_size});
        }
        // Use linear kernel: (seq_len,d_model) x (d_model,vocab)
        std::vector<std::shared_ptr<TensorBase>> lin_inputs = {normed, weights.lm_head};
        std::vector<std::shared_ptr<TensorBase>> lin_outputs = {output};
        if (!executeKernel("linear", lin_inputs, lin_outputs))
        {
            LOG_ERROR("LM head projection failed");
            return false;
        }
        last_logits_ = output; // cache

        // Parity capture: LM head output (final logits)
        captureIfEnabled(PipelineStage::LM_HEAD, -1, output);

        // Optional baseline snapshot of final logits (rank 0 only)
        const bool capture_baseline = debugEnv().baseline.capture;
        const bool compare_baseline = debugEnv().baseline.compare;
        if ((capture_baseline || compare_baseline) && getRank() == 0)
        {
            handle_prefill_stage_snapshot(0, "final_logits", output->data(), (size_t)output->size(), config_.getLayerConfig().vocab_size, 5e-4, capture_baseline, compare_baseline);
        }
        return true;
    }

    bool QwenPipeline::validate(const ModelWeights &w) const
    {
        // Helper: validate a vector of per-layer tensors
        auto check_vec = [&](const std::vector<std::shared_ptr<TensorBase>> &v, const char *name) -> bool
        {
            if ((int)v.size() != config_.getLayerConfig().n_layers)
            {
                LOG_ERROR("Weights vector size mismatch for " << name << " expected=" << config_.getLayerConfig().n_layers << " got=" << v.size());
                return false;
            }
            for (size_t i = 0; i < v.size(); ++i)
            {
                if (!v[i])
                {
                    LOG_ERROR("Null weight in " << name << " index=" << i);
                    return false;
                }
            }
            return true;
        };

        // Core singleton tensors
        if (!w.token_embedding || !w.output_norm_weight || !w.lm_head)
        {
            LOG_ERROR("Missing core weights (embedding/output_norm/lm_head)");
            return false;
        }

        // Per-layer collections
        if (!check_vec(w.attn_norm_weight, "attn_norm_weight"))
            return false;
        if (!check_vec(w.wq, "wq"))
            return false;
        if (!check_vec(w.wk, "wk"))
            return false;
        if (!check_vec(w.wv, "wv"))
            return false;
        if (!check_vec(w.wo, "wo"))
            return false;
        if (!check_vec(w.ffn_norm_weight, "ffn_norm_weight"))
            return false;
        if (!check_vec(w.w_gate, "w_gate"))
            return false;
        if (!check_vec(w.w_up, "w_up"))
            return false;
        if (!check_vec(w.w_down, "w_down"))
            return false;

        return true;
    }

    // KV cache initialization (simple replicated per-layer key/value buffers)
    void QwenPipeline::initializeKVCache(int seq_len)
    {
        if (!use_kv_cache_)
            return;
        if (seq_len <= 0)
            seq_len = 1;
        k_cache_.resize(config_.getLayerConfig().n_layers);
        v_cache_.resize(config_.getLayerConfig().n_layers);
        for (int l = 0; l < config_.getLayerConfig().n_layers; ++l)
        {
            if (!k_cache_[l] || k_cache_[l]->shape()[0] < seq_len)
                k_cache_[l] = createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim});
            if (!v_cache_[l] || v_cache_[l]->shape()[0] < seq_len)
                v_cache_[l] = createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim});
        }
        kv_cache_state_.capacity_tokens = seq_len;
        kv_cache_state_.used_tokens = 0;
        kv_cache_state_.growth_events = 0;
    }

    bool QwenPipeline::ensureKVCapacityInternal(int required_tokens)
    {
        if (!use_kv_cache_)
            return true;
        if (required_tokens <= kv_cache_state_.capacity_tokens)
            return true;
        int new_cap = std::max(required_tokens, kv_cache_state_.capacity_tokens * 2);
        if (new_cap > config_.getLayerConfig().max_seq_len)
            new_cap = config_.getLayerConfig().max_seq_len;
        initializeKVCache(new_cap);
        kv_cache_state_.growth_events++;
        if (getRank() == 0)
            LOG_INFO("[KVCacheGrow] new_capacity=" << new_cap);
        return required_tokens <= kv_cache_state_.capacity_tokens;
    }

    bool QwenPipeline::ensureKVCapacity(int required_tokens) { return ensureKVCapacityInternal(required_tokens); }

    std::unique_ptr<IModelWeights> QwenPipeline::loadWeights(const std::string &path)
    {
        // Use ModelLoader directly instead of deprecated free function
        ModelLoader loader;
        if (!loader.loadModel(path))
        {
            throw std::runtime_error("ModelLoader loadModel failed: " + path);
        }
        auto loaded = llaminar::loadModelWeights_impl_bridge(loader, config_.getLayerConfig());
        auto weights = std::make_unique<QwenModelWeights>();
        weights->inner = std::move(loaded);
        return weights;
    }

    bool QwenPipeline::execute(const std::vector<int> &token_ids,
                               const ModelWeights &weights,
                               std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("QwenPipeline::execute");
        start_time_ = std::chrono::high_resolution_clock::now();
        if (!validate(weights))
        {
            LOG_ERROR("QwenPipeline: Weight validation failed");
            return false;
        }
        int seq_len = (int)token_ids.size();
        if (seq_len <= 0 || seq_len > config_.getLayerConfig().max_seq_len)
        {
            LOG_ERROR("Invalid sequence length " << seq_len);
            return false;
        }

        // When doing layer replay compare diagnostics we must force the full distributed path
        // (the small_seq_fast_path bypasses executeTransformerLayer and thus emits no stage captures).
        bool force_full_layer_capture = debugEnv().pipeline.layer_token_diff && debugEnv().pipeline.layer_replay_compare;

        const bool capture_baseline = debugEnv().baseline.capture;
        const bool compare_baseline = debugEnv().baseline.compare;
        if (capture_baseline && getRank() == 0)
        {
            PrefillBaselineRegistry::instance().clear();
            LOG_DEBUG("[PrefillBaseline] cleared registry at execute start");
        }

        // Dynamic KV cache initial sizing
        if (use_kv_cache_ && kv_cache_dynamic_init_)
        {
            if (kv_cache_state_.capacity_tokens < seq_len)
            {
                if (getRank() == 0)
                    LOG_INFO("[KVCacheInit] dynamic resize to " << seq_len);
                initializeKVCache(seq_len);
            }
        }

        // Small sequence fast path (replicated) if seq_len < world_size unless diagnostics need full layer captures
        int world_size = getSize();
        if (seq_len < world_size && !force_full_layer_capture)
        {
            // Increment fast path counter for diagnostics
            small_seq_fast_path_calls_.fetch_add(1, std::memory_order_relaxed);

            // Only rank 0 executes the replicated forward pass
            if (getRank() == 0)
            {
                // Log diagnostic bypass if needed
                if (force_full_layer_capture)
                {
                    LOG_INFO("[LayerTokenDiffDiag] bypass small_seq_fast_path due to replay_compare instrumentation");
                }

                // Allocate output tensor if needed
                if (!output || output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != config_.getLayerConfig().vocab_size)
                    output = createLocalTensor({seq_len, config_.getLayerConfig().vocab_size});

                // === Naive local forward pass (legacy simplified path) ===

                // Get embedding table pointer
                auto embed_data = weights.token_embedding->data();

                // Allocate working buffers for hidden states
                std::vector<float> hidden(seq_len * config_.getLayerConfig().d_model, 0.f);
                std::vector<float> tmp(seq_len * config_.getLayerConfig().d_model, 0.f);

                // Define inline helper lambdas for basic operations

                // RMSNorm: normalize rows and scale by gamma
                auto rmsnorm = [&](std::vector<float> &mat, const float *wn)
                {
                    kernels::RMSNormExecOptions opts;
                    kernels::rmsnorm_row_major_fused(mat.data(), wn, mat.data(),
                                                     (size_t)seq_len, (size_t)config_.getLayerConfig().d_model,
                                                     config_.getLayerConfig().eps, kernels::GammaMode::REPLICATED, 0, opts);
                };

                // Matrix multiplication: C = A * B (naive implementation)
                auto matmul = [&](const std::vector<float> &A, const float *B, int k, int n, std::vector<float> &C)
                {
                    C.assign(seq_len * n, 0.f);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        const float *a = &A[i * k];
                        float *crow = &C[i * n];
                        for (int kk = 0; kk < k; ++kk)
                        {
                            float aval = a[kk];
                            const float *bcol = &B[kk * n];
                            for (int j = 0; j < n; ++j)
                                crow[j] += aval * bcol[j];
                        }
                    }
                };

                // Element-wise addition: A += B
                auto elementwise_add = [&](std::vector<float> &A, const std::vector<float> &B)
                {
                    for (size_t i = 0; i < A.size(); ++i)
                        A[i] += B[i];
                };

                // Sigmoid activation
                auto sigmoid = [](float x)
                {
                    return 1.f / (1.f + std::exp(-x));
                };

                // SwiGLU activation: out = up * sigmoid(gate)
                auto swiglu = [&](const std::vector<float> &up, const std::vector<float> &gate, std::vector<float> &out)
                {
                    out.resize(up.size());
                    for (size_t i = 0; i < up.size(); ++i)
                        out[i] = up[i] * sigmoid(gate[i]);
                };

                // --- Embedding lookup ---
                // Copy token embeddings into hidden state buffer
                for (int t = 0; t < seq_len; ++t)
                {
                    int tok = token_ids[t];
                    std::memcpy(&hidden[t * config_.getLayerConfig().d_model],
                                &embed_data[tok * config_.getLayerConfig().d_model],
                                sizeof(float) * config_.getLayerConfig().d_model);
                }
                // --- Per-layer transformer blocks ---
                for (int layer = 0; layer < config_.getLayerConfig().n_layers; ++layer)
                {
                    // === Attention Block ===

                    // Pre-attention normalization
                    rmsnorm(hidden, weights.attn_norm_weight[layer]->data());

                    // Project to Q, V (simplified: no K, using Q approximation)
                    std::vector<float> Q, V, context;
                    matmul(hidden, weights.wq[layer]->data(), config_.getLayerConfig().d_model,
                           config_.getLayerConfig().n_head * config_.getLayerConfig().head_dim, tmp);
                    Q = tmp;

                    // Project to V (value)
                    matmul(hidden, weights.wv[layer]->data(), config_.getLayerConfig().d_model,
                           config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim, tmp);
                    V = tmp;

                    // Simplified attention: use Q as initial context
                    context = Q;

                    // Compute mean pooling of V across sequence
                    int ctx_dim = config_.getLayerConfig().n_head * config_.getLayerConfig().head_dim;
                    std::vector<float> v_mean(ctx_dim, 0.f);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        const float *vrow = &V[i * ctx_dim];
                        for (int j = 0; j < ctx_dim; ++j)
                            v_mean[j] += vrow[j];
                    }
                    for (int j = 0; j < ctx_dim; ++j)
                        v_mean[j] /= std::max(1, seq_len);

                    // Blend context with V mean (simplified attention)
                    for (int i = 0; i < seq_len; ++i)
                    {
                        float *crow = &context[i * ctx_dim];
                        for (int j = 0; j < ctx_dim; ++j)
                            crow[j] = 0.5f * (crow[j] + v_mean[j]);
                    }

                    // Output projection
                    matmul(context, weights.wo[layer]->data(), ctx_dim, config_.getLayerConfig().d_model, tmp);

                    // Residual connection
                    elementwise_add(tmp, hidden);
                    hidden = tmp;

                    // === FFN Block ===

                    // Pre-FFN normalization
                    rmsnorm(hidden, weights.ffn_norm_weight[layer]->data());

                    // Gate projection
                    std::vector<float> gate, up, swiglu_out;
                    matmul(hidden, weights.w_gate[layer]->data(), config_.getLayerConfig().d_model, config_.getLayerConfig().d_ff, gate);

                    // Up projection
                    matmul(hidden, weights.w_up[layer]->data(), config_.getLayerConfig().d_model, config_.getLayerConfig().d_ff, up);

                    // SwiGLU activation
                    swiglu(up, gate, swiglu_out);

                    // Down projection
                    matmul(swiglu_out, weights.w_down[layer]->data(), config_.getLayerConfig().d_ff, config_.getLayerConfig().d_model, tmp);

                    // Residual connection
                    elementwise_add(tmp, hidden);
                    hidden = tmp;
                }
                // --- Final Output ---

                // Final normalization before LM head
                rmsnorm(hidden, weights.output_norm_weight->data());

                // Project to vocabulary logits
                std::vector<float> logits;
                matmul(hidden, weights.lm_head->data(), config_.getLayerConfig().d_model, config_.getLayerConfig().vocab_size, logits);

                // Copy logits to output tensor
                std::memcpy(const_cast<float *>(output->data()), logits.data(),
                            sizeof(float) * logits.size());

                // Clean up baseline registry if comparison was enabled
                if (compare_baseline)
                {
                    PrefillBaselineRegistry::instance().clear();
                }

                // Update KV cache state (though not used in fast path)
                if (use_kv_cache_)
                {
                    kv_cache_state_.used_tokens = seq_len;
                    n_past_ = seq_len;
                }
            }
            // --- Broadcast output to all ranks ---
            // Other ranks need to allocate output buffer to receive broadcast
            if (getRank() != 0)
            {
                if (!output || output->shape().size() != 2 ||
                    output->shape()[0] != seq_len || output->shape()[1] != config_.getLayerConfig().vocab_size)
                {
                    output = createLocalTensor({seq_len, config_.getLayerConfig().vocab_size});
                }
            }

            // Broadcast result from rank 0 to all other ranks
            checkMPIError(MPI_Bcast(const_cast<float *>(output->data()),
                                    seq_len * config_.getLayerConfig().vocab_size, MPI_FLOAT, 0, getComm()),
                          "MPI_Bcast small-seq output");

            return true;
        }

        // Standard distributed path
        auto tensors = createIntermediateTensors(seq_len);
        auto current_input = tensors[0];
        auto layer_output = tensors[1];
        if (!executeEmbedding(token_ids, weights.token_embedding, current_input))
            return false;
        for (int layer = 0; layer < config_.getLayerConfig().n_layers; ++layer)
        {
            // Ensure attention kernel records correct layer index during standard prefill execution
            if (auto attn = dynamic_cast<MPIAttentionKernel *>(getKernel("attention")))
            {
                attn->setLayerIndex(layer);
            }
            if (!executeTransformerLayer(layer, current_input, weights, layer_output))
            {
                LOG_ERROR("Layer " << layer << " execution failed");
                return false;
            }
            std::swap(current_input, layer_output);
        }
        // Output projection
        if (!executeOutputProjection(current_input, weights, output))
            return false;

        if (compare_baseline && getRank() == 0)
        {
            PrefillBaselineRegistry::instance().clear();
            LOG_DEBUG("[PrefillBaseline] cleared registry after comparison run");
        }
        if (use_kv_cache_)
        {
            kv_cache_state_.used_tokens = seq_len;
            n_past_ = seq_len;
        }
        return true;
    }

    // Override variant from AbstractPipeline not yet supported here
    bool QwenPipeline::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                               std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        LOG_ERROR("QwenPipeline::execute(vector) not supported; use execute(token_ids, weights, output) overload");
        return false;
    }

    // Minimal logits() implementation – returns cached last logits tensor
    bool QwenPipeline::logits(std::shared_ptr<TensorBase> &out_logits)
    {
        if (!last_logits_)
        {
            LOG_WARN("logits() requested but no cached logits available");
            return false;
        }
        out_logits = last_logits_;
        return true;
    }

    const KVCacheState *QwenPipeline::kvCacheState() const
    {
        kv_snapshot_.capacity_tokens = kv_cache_state_.capacity_tokens;
        kv_snapshot_.used_tokens = kv_cache_state_.used_tokens;
        kv_snapshot_.growth_events = kv_cache_state_.growth_events;
        return &kv_snapshot_;
    }

    // (handle_prefill_stage_snapshot moved to prefill_diagnostics.h/.cpp)
} // namespace llaminar

// === Section 4: AbstractPipeline interface (prefill/decode/logits) partial migration ===
namespace llaminar
{
    bool QwenPipeline::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.empty() || outputs.empty())
            return false;
        return true;
    }

    bool QwenPipeline::prefill(const std::vector<int> &tokens,
                               const IModelWeights &weights_iface,
                               StageContext &ctx)
    {
        setStagePrefill();
        current_tokens_ = tokens;
        const auto *w = dynamic_cast<const QwenModelWeights *>(&weights_iface);
        if (!w)
        {
            LOG_ERROR("prefill: invalid weights type");
            return false;
        }
        auto output = TensorFactory::create_simple({(int)tokens.size(), config_.getLayerConfig().vocab_size});
        if (!execute(tokens, w->inner, output))
            return false;
        last_logits_ = output;
        kv_snapshot_.capacity_tokens = getKVCacheCapacity();
        kv_snapshot_.used_tokens = getKVCacheUsed();
        kv_snapshot_.growth_events = getKVCacheGrowthEvents();
        ctx.stage = InferenceStage::Prefill;
        ctx.seq_len = (int)tokens.size();
        ctx.generated = 0;
        ctx.kv_capacity = kv_snapshot_.capacity_tokens;
        ctx.kv_used = kv_snapshot_.used_tokens;
        return true;
    }

    bool QwenPipeline::decode(int next_token,
                              const IModelWeights &weights_iface,
                              StageContext &ctx)
    {
        const auto *w = dynamic_cast<const QwenModelWeights *>(&weights_iface);
        if (!w)
        {
            LOG_ERROR("decode: invalid weights type");
            return false;
        }
        setStageDecode();
        std::shared_ptr<TensorBase> one_logits;
        bool used_incremental = incrementalDecodeToken(next_token, w->inner, one_logits);
        if (!used_incremental)
        {
            current_tokens_.push_back(next_token);
            auto replay = TensorFactory::create_simple({(int)current_tokens_.size(), config_.getLayerConfig().vocab_size});
            if (!execute(current_tokens_, w->inner, replay))
                return false;
            last_logits_ = replay;
        }
        else
        {
            current_tokens_.push_back(next_token);
            if (!last_logits_)
                last_logits_ = one_logits;
        }
        ctx.stage = InferenceStage::Decode;
        ctx.seq_len = (int)current_tokens_.size();
        ctx.generated += 1;
        kv_snapshot_.capacity_tokens = getKVCacheCapacity();
        kv_snapshot_.used_tokens = getKVCacheUsed();
        kv_snapshot_.growth_events = getKVCacheGrowthEvents();
        ctx.kv_capacity = kv_snapshot_.capacity_tokens;
        ctx.kv_used = kv_snapshot_.used_tokens;
        return true;
    }

    /**
     * @brief Incrementally decode a single token using the existing KV cache (fast path).
     *
     * High-level stages:
     *  1. Environment + guard checks (disable flags, kv cache availability, capacity growth)
     *  2. Single-token embedding allocation (shape 1 x d_model)
     *  3. Per-layer forward (attention + FFN) with stage capture for diagnostics
     *  4. Output projection -> logits (1 x vocab)
     *  5. Optional incremental hidden / cache tracing
     *  6. Optional replay parity comparison (rebuild full prefix and diff captured stages)
     *
     * Replay Parity Mode (layer_replay_compare):
     *  When enabled, after producing incremental rows for the new token we reconstruct a full
     *  prefix sequence (all previous tokens + new token) in a fresh pipeline instance and capture
     *  only the final token's stages. We then diff stage-by-stage to identify the first divergence.
     *
     * Return semantics:
     *  true  -> incremental path executed (logits written to output_logits, parity check optional)
     *  false -> caller should fall back to full prefill+decode (e.g. guards, capacity failure)
     */
    bool QwenPipeline::incrementalDecodeToken(int token_id,
                                              const ModelWeights &weights,
                                              std::shared_ptr<TensorBase> &output_logits)
    {
        // === Stage 1: Environment + guard checks ===
        const auto &env = debugEnv();
        if (getRank() == 0)
        {
            LOG_INFO("[IncrDecodeEnv] n_past=" << n_past_
                                               << " use_kv=" << (use_kv_cache_ ? 1 : 0)
                                               << " ablate_attention=" << (env.ablation.ablate_attention ? 1 : 0)
                                               << " ablate_ffn=" << (env.ablation.ablate_ffn ? 1 : 0)
                                               << " layer_token_diff=" << (env.pipeline.layer_token_diff ? 1 : 0)
                                               << " replay_compare=" << (env.pipeline.layer_replay_compare ? 1 : 0)
                                               << " disable_incr=" << (env.pipeline.disable_incremental_decode ? 1 : 0));
        }
        if (env.pipeline.disable_incremental_decode)
        {
            LOG_INFO("[IncrEarlyReturn] rank=" << getRank() << " reason=disabled_incremental");
            return false;
        }
        // If doing layer replay compare we must still run full layers (ablation ignored locally)
        const bool diag_force_full_layers = env.pipeline.layer_token_diff && env.pipeline.layer_replay_compare;
        (void)diag_force_full_layers; // (currently only informational; ablation checks occur in executeTransformerLayer)

        if (!use_kv_cache_)
        {
            LOG_INFO("[IncrEarlyReturn] rank=" << getRank() << " reason=kv_cache_disabled");
            return false;
        }
        if (k_cache_.empty() || v_cache_.empty() || (int)k_cache_.size() != config_.getLayerConfig().n_layers || (int)v_cache_.size() != config_.getLayerConfig().n_layers)
        {
            LOG_INFO("[IncrEarlyReturn] rank=" << getRank() << " reason=kv_cache_uninit_or_size_mismatch k_size=" << k_cache_.size() << " v_size=" << v_cache_.size() << " expected_layers=" << config_.getLayerConfig().n_layers);
            return false;
        }
        if (!weights.token_embedding)
        {
            LOG_ERROR("incrementalDecodeToken: missing token embedding");
            return false;
        }
        // Capacity (position == n_past_)
        if (!ensureKVCapacity(n_past_ + 1))
        {
            LOG_WARN("[IncrEarlyReturn] rank=" << getRank() << " reason=ensureKVCapacity_failed requested=" << (n_past_ + 1));
            return false;
        }

        // === Stage 2: Single-token embedding ===
        auto current = embedSingleToken(token_id, weights.token_embedding);
        if (!current)
            return false;
        size_t layer_rows_offset_before = last_layer_token_rows_.size();
        if (env.pipeline.incr_trace && getRank() == 0)
        {
            LOG_INFO("[IncrTrace] start token=" << token_id << " n_past=" << n_past_ << " use_kv=1");
        }
        setStageDecode();
        const int position = n_past_;
        setSequencePosition(position);

        // === Stage 3: Layer forward (seq_len == 1) ===
        const int seq_len = 1;
        auto layer_output = createLocalTensor({seq_len, config_.getLayerConfig().d_model});
        if (!layer_output)
        {
            LOG_ERROR("incrementalDecodeToken: failed layer_output alloc");
            return false;
        }
        auto logits = createLocalTensor({seq_len, config_.getLayerConfig().vocab_size});
        if (!logits)
        {
            LOG_ERROR("incrementalDecodeToken: failed logits alloc");
            return false;
        }
        for (int layer = 0; layer < config_.getLayerConfig().n_layers; ++layer)
        {
            // Update attention kernel context (expected window = committed + 1)
            if (auto attn = dynamic_cast<MPIAttentionKernel *>(getKernel("attention")))
            {
                attn->setSequencePosition(n_past_);
                attn->setLayerIndex(layer);
                attn->setExpectedTotalWindow((size_t)n_past_ + 1);
            }
            if (!executeTransformerLayer(layer, current, weights, layer_output))
            {
                LOG_ERROR("incrementalDecodeToken: layer " << layer << " failed");
                return false;
            }
            current = layer_output;
            if (env.pipeline.incr_cache_trace && getRank() == 0 && use_kv_cache_)
            {
                // Minimal KV preview (first 4 values of first head at this position)
                if (layer < (int)k_cache_.size() && k_cache_[layer])
                {
                    const int head_dim = config_.getLayerConfig().head_dim;
                    size_t base = (size_t)position * head_dim;
                    float k_prev[4] = {0}, v_prev[4] = {0};
                    auto &K = k_cache_[layer];
                    auto &V = v_cache_[layer];
                    for (int i = 0; i < 4 && i < head_dim; ++i)
                    {
                        k_prev[i] = K->data()[base + i];
                        v_prev[i] = V->data()[base + i];
                    }
                    LOG_INFO("[IncrCacheTrace] layer=" << layer << " pos=" << position << " k0=[" << k_prev[0] << "," << k_prev[1] << "," << k_prev[2] << "," << k_prev[3]
                                                       << "] v0=[" << v_prev[0] << "," << v_prev[1] << "," << v_prev[2] << "," << v_prev[3] << "]");
                }
            }
        }

        // === Stage 4: Output projection ===
        if (!executeOutputProjection(current, weights, logits))
        {
            LOG_ERROR("incrementalDecodeToken: output projection failed");
            return false;
        }

        // === Stage 5: Optional hidden preview ===
        if (env.pipeline.incr_hidden_trace && getRank() == 0)
        {
            int d = config_.getLayerConfig().d_model;
            int dump = std::min(d, 16);
            std::ostringstream oss;
            oss << "[IncrHiddenTrace] pos=" << n_past_ << " dims=" << d << " preview=";
            const float *row = current->data();
            for (int i = 0; i < dump; ++i)
            {
                oss << row[i];
                if (i + 1 < dump)
                    oss << ',';
            }
            LOG_INFO(oss.str());
        }
        output_logits = logits;
        n_past_ += 1;
        if (use_kv_cache_)
            kv_cache_state_.used_tokens = n_past_;

        // === Stage 6: Optional replay parity comparison ===
        const auto &penv = debugEnv().pipeline;
        if (penv.layer_token_diff && penv.layer_replay_compare)
        {
            // Symmetry: ensure all ranks participate
            int local_flag = 1;
            int global_sum = 0;
            MPI_Allreduce(&local_flag, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (global_sum != getSize())
            {
                if (getRank() == 0)
                    LOG_WARN("[LayerReplayIsoSkip] rank quorum mismatch global_sum=" << global_sum << " world=" << getSize());
            }
            else
            {
                if (getRank() == 0)
                    LOG_INFO("[LayerReplayIsoEnter] token_pos=" << n_past_ << " rows_size=" << last_layer_token_rows_.size());
                try
                {
                    // Build replay prefix: prior tokens + new token (n_past_ total length)
                    std::vector<int> replay_seq;
                    replay_seq.reserve((size_t)n_past_);
                    int prior_count = std::min((int)current_tokens_.size(), n_past_ - 1);
                    for (int i = 0; i < prior_count; ++i)
                        replay_seq.push_back(current_tokens_[i]);
                    replay_seq.push_back(token_id);
                    if ((int)replay_seq.size() != n_past_ && getRank() == 0)
                        LOG_WARN("[LayerReplayIsoGuard] replay_seq.size()=" << replay_seq.size() << " expected=" << n_past_);

                    // Collect incremental rows for THIS token
                    std::vector<LayerTokenDiffRow> inc_rows;
                    if (getRank() == 0)
                    {
                        for (size_t i = layer_rows_offset_before; i < last_layer_token_rows_.size(); ++i)
                        {
                            const auto &r = last_layer_token_rows_[i];
                            if (!r.pipeline || r.pipeline == this)
                                inc_rows.push_back(r);
                        }
                        // Retro add internal attention rows earlier than offset with matching seq_len
                        for (size_t i = 0; i < layer_rows_offset_before; ++i)
                        {
                            const auto &r = last_layer_token_rows_[i];
                            if (r.incremental && r.seq_len == (int)replay_seq.size() && r.stage.rfind("attn_int_", 0) == 0)
                                inc_rows.push_back(r);
                        }
                    }
                    if (getRank() == 0)
                        last_layer_token_rows_.clear(); // prepare for replay capture

                    // Execute replay in fresh pipeline
                    auto replay_pipe = createQwenPipeline(config_);
                    auto replay_logits = TensorFactory::create_simple({(int)replay_seq.size(), config_.getLayerConfig().vocab_size});
                    bool replay_ok = replay_pipe && replay_pipe->execute(replay_seq, weights, replay_logits);
                    std::vector<LayerTokenDiffRow> rep_rows;
                    if (getRank() == 0)
                    {
                        for (auto &r_all : last_layer_token_rows_)
                            if (r_all.seq_len == (int)replay_seq.size())
                                rep_rows.push_back(r_all);
                        if (rep_rows.empty())
                            rep_rows = last_layer_token_rows_; // fallback
                        // Restore only incremental rows for future tokens
                        last_layer_token_rows_.clear();
                        last_layer_token_rows_.insert(last_layer_token_rows_.end(), inc_rows.begin(), inc_rows.end());
                        if (!replay_ok)
                            LOG_WARN("[LayerReplayIso] replay execute failed");
                    }

                    // Pairing + diff (rank 0)
                    if (getRank() == 0 && !rep_rows.empty())
                    {
                        auto order = [](const std::string &v)
                        {
                            if (v == "attn_qkv_in") return 0;
                            if (v == "attn_int_q_proj") return 1;
                            if (v == "attn_int_k_proj") return 2;
                            if (v == "attn_int_q_rope") return 3;
                            if (v == "attn_int_k_rope") return 4;
                            if (v == "attn_int_context") return 5;
                            if (v == "attn_int_context_full") return 6;
                            if (v == "attn_int_out_partial") return 7;
                            if (v == "attn_out") return 8;
                            if (v == "attn_residual") return 9;
                            if (v == "ffn_norm") return 10;
                            if (v == "ffn_out") return 11;
                            if (v == "layer_output") return 12;
                            if (v == "attn_norm") return 100; // legacy
                            return 500; };
                        struct StagePair
                        {
                            const LayerTokenDiffRow *inc;
                            const LayerTokenDiffRow *rep;
                        };
                        std::map<int, std::vector<StagePair>> layer_pairs;
                        for (auto &ir : inc_rows)
                            layer_pairs[ir.layer];
                        for (auto &ir : inc_rows)
                        {
                            const LayerTokenDiffRow *rep_match = nullptr;
                            for (auto &rr : rep_rows)
                                if (rr.layer == ir.layer && rr.stage == ir.stage)
                                {
                                    rep_match = &rr;
                                    break;
                                }
                            if (!rep_match)
                                for (auto &rr : rep_rows)
                                    if (rr.stage == ir.stage)
                                    {
                                        rep_match = &rr;
                                        break;
                                    }
                            layer_pairs[ir.layer].push_back(StagePair{&ir, rep_match});
                        }
                        const double rel_l2_warn = 1e-5;
                        bool first_exceed_recorded = false;
                        g_replay_first_exceed.store(false);
                        DiffSummary first_exceed_ds{};
                        int first_exceed_layer = 0;
                        std::string first_exceed_stage;
                        for (auto &kv : layer_pairs)
                        {
                            int layer = kv.first;
                            auto &pairs = kv.second;
                            std::stable_sort(pairs.begin(), pairs.end(), [&](const StagePair &a, const StagePair &b)
                                             { return order(a.inc->stage) < order(b.inc->stage); });
                            for (auto &sp : pairs)
                            {
                                if (!sp.rep)
                                    continue; // missing replay stage
                                if (sp.inc->values.empty() || sp.inc->values.size() != sp.rep->values.size())
                                    continue;
                                DiffSummary ds = computeDiffSummary(sp.inc->values.data(), sp.rep->values.data(), sp.inc->values.size());
                                // Log policy: internal attention + boundaries always; layer0 any attn_*; others only forced types
                                bool force_log = false;
                                bool internal_attn = (sp.inc->stage.rfind("attn_int_", 0) == 0);
                                bool attn_boundary = (sp.inc->stage == "attn_qkv_in" || sp.inc->stage == "attn_out");
                                if (internal_attn || attn_boundary)
                                    force_log = true;
                                if (!force_log && layer == 0 && sp.inc->stage.rfind("attn_", 0) == 0)
                                    force_log = true;
                                if (force_log)
                                    LOG_INFO("[LayerReplayIsoStage] token_pos=" << (replay_seq.size() - 1) << " layer=" << layer << " stage=" << sp.inc->stage << " rel_l2=" << ds.rel_l2 << " max_abs=" << ds.max_abs << " mean_abs=" << ds.mean_abs);
                                if (ds.rel_l2 > rel_l2_warn && !first_exceed_recorded)
                                {
                                    first_exceed_recorded = true;
                                    first_exceed_ds = ds;
                                    first_exceed_layer = layer;
                                    first_exceed_stage = sp.inc->stage;
                                    g_replay_first_exceed.store(true);
                                    LOG_WARN("[LayerReplayIsoFirstExceed] token_pos=" << (replay_seq.size() - 1) << " layer=" << layer << " stage=" << sp.inc->stage << " rel_l2=" << ds.rel_l2 << " max_abs=" << ds.max_abs << " worst_index=" << ds.worst_index);
                                }
                            }
                        }
                        if (first_exceed_recorded)
                        {
                            LOG_WARN("[LayerReplayIso] token_pos=" << (replay_seq.size() - 1) << " first_exceed_layer=" << first_exceed_layer << " stage=" << first_exceed_stage << " rel_l2=" << first_exceed_ds.rel_l2 << " max_abs=" << first_exceed_ds.max_abs << " worst_index=" << first_exceed_ds.worst_index << " inc=" << first_exceed_ds.value_a << " rep=" << first_exceed_ds.value_b);
                        }
                        else
                        {
                            LOG_INFO("[LayerReplayIso] token_pos=" << (replay_seq.size() - 1) << " all_layers_all_stages_rel_l2<=1e-5");
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    if (getRank() == 0)
                        LOG_ERROR("[LayerReplayIso] exception: " << e.what());
                }
            }
        }
        return true;
    }

} // namespace llaminar

// === Section 5: COSMA fused prefill attention implementation (standalone) ===
namespace llaminar
{
    /**
     * @brief Execute fused RMSNorm + QKV + scaled masked attention using COSMA backend for prefill.
     *
     * Stages:
     *  1. Descriptor setup & fused rmsnorm+qkv generation (CosmaPrefillManager)
     *  2. Materialize normalized + Q/K/V into row-major temporary buffers
     *  3. Apply RoPE and compute attention using optimized primitives (attention_primitives.h)
     *  4. Output projection via adaptiveMatMul (may dispatch COSMA/local backend internally)
     *  5. Timing capture for norm/attention/linear phases
     *
     * This implementation uses the optimized attention primitives from attention_primitives.cpp
     * which provide vectorized RoPE, efficient QK score computation, and numerically stable softmax.
     */
    bool QwenPipeline::executePrefillAttentionCosma(int layer_idx,
                                                    const LargeMatmulPlan &plan,
                                                    std::shared_ptr<TensorBase> &input,
                                                    const ModelWeights &weights,
                                                    std::shared_ptr<TensorBase> &attn_norm_out,
                                                    std::shared_ptr<TensorBase> &attn_out,
                                                    PrefillAttentionTiming &timing)
    {
        PERF_SCOPED_TIMER("QwenPipeline::executePrefillAttentionCosma");

        // Validate plan
        if (!plan.is_valid())
        {
            LOG_ERROR("executePrefillAttentionCosma called with invalid plan: " << plan.rationale);
            return false;
        }

        CosmaPrefillManager &manager = CosmaPrefillManager::instance();

        // Use plan configuration instead of recalculating
        const int seq_len = plan.seq_len;
        const int hidden_size = plan.d_model;
        const int head_dim = plan.head_dim;
        const int n_heads = plan.n_heads;
        const int n_kv_heads = plan.n_kv_heads;
        const int total_head_dim = plan.total_head_dim();
        const int kv_head_dim = plan.kv_head_dim();

        // --- Stage 1: weight descriptors for fused op ---
        auto make_desc = [&](const std::shared_ptr<TensorBase> &tensor, const std::string &id) -> WeightDescriptor
        {
            const auto &shape = tensor->shape();
            return WeightDescriptor{id, shape.size() > 0 ? shape[0] : 0, shape.size() > 1 ? shape[1] : 0, (int64_t)(shape.size() > 1 ? shape[1] : 0), 1, 0, tensor->data(), 0};
        };
        WeightDescriptor wq_desc = make_desc(weights.wq[layer_idx], "layer" + std::to_string(layer_idx) + "_wq");
        WeightDescriptor wk_desc = make_desc(weights.wk[layer_idx], "layer" + std::to_string(layer_idx) + "_wk");
        WeightDescriptor wv_desc = make_desc(weights.wv[layer_idx], "layer" + std::to_string(layer_idx) + "_wv");
        const float scale = 1.0f / std::sqrt((float)head_dim);
        auto fused_start = std::chrono::high_resolution_clock::now();
        auto fused = manager.fused_rmsnorm_qkv(input->data(),
                                               weights.attn_norm_weight[layer_idx]->data(),
                                               wq_desc, wk_desc, wv_desc,
                                               seq_len, hidden_size, config_.getLayerConfig().eps, scale, false);
        auto fused_end = std::chrono::high_resolution_clock::now();
        if ((!fused.normalized.mat && !fused.normalized.host_owned) || (!fused.q.mat && !fused.q.host_owned) || (!fused.k.mat && !fused.k.host_owned) || (!fused.v.mat && !fused.v.host_owned))
        {
            LOG_ERROR("COSMA fused_rmsnorm_qkv incomplete for layer " << layer_idx);
            return false;
        }
        // --- Stage 2: materialize row-major normalized + Q K V ---
        std::vector<float> norm_buf((size_t)seq_len * hidden_size, 0.f);
        manager.to_row_major(fused.normalized, norm_buf.data());
        std::memcpy(attn_norm_out->data(), norm_buf.data(), norm_buf.size() * sizeof(float)); // output rmsnorm
        std::vector<float> q_buf((size_t)seq_len * total_head_dim, 0.f),
            k_buf((size_t)seq_len * kv_head_dim, 0.f),
            v_buf((size_t)seq_len * kv_head_dim, 0.f);
        manager.to_row_major(fused.q, q_buf.data(), true);
        manager.to_row_major(fused.k, k_buf.data(), true);
        manager.to_row_major(fused.v, v_buf.data(), true);
        timing.norm_ms = std::chrono::duration<double, std::milli>(fused_end - fused_start).count();

        // --- Stage 3: Apply RoPE and compute attention using optimized primitives ---
        auto attention_start = std::chrono::high_resolution_clock::now();

        // Output buffer for multi-head attention context
        std::vector<float> context_concat((size_t)seq_len * total_head_dim, 0.f);

        // Handle GQA (Grouped Query Attention) by replicating K/V heads if needed
        if (n_kv_heads != n_heads)
        {
            // Expand K and V to match number of query heads
            std::vector<float> k_expanded((size_t)seq_len * total_head_dim, 0.f);
            std::vector<float> v_expanded((size_t)seq_len * total_head_dim, 0.f);

            for (int row = 0; row < seq_len; ++row)
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    int kv_h = h % n_kv_heads; // Map query head to KV head

                    // Copy this KV head's data to the expanded buffer
                    const float *k_src = k_buf.data() + (size_t)row * kv_head_dim + kv_h * head_dim;
                    const float *v_src = v_buf.data() + (size_t)row * kv_head_dim + kv_h * head_dim;

                    float *k_dst = k_expanded.data() + (size_t)row * total_head_dim + h * head_dim;
                    float *v_dst = v_expanded.data() + (size_t)row * total_head_dim + h * head_dim;

                    std::memcpy(k_dst, k_src, head_dim * sizeof(float));
                    std::memcpy(v_dst, v_src, head_dim * sizeof(float));
                }
            }

            // Use the attention primitives with expanded K/V
            llaminar::attn::apply_rope(q_buf.data(), k_expanded.data(), seq_len, head_dim, n_heads,
                                       n_past_, config_.getLayerConfig().rope_freq_base);
            llaminar::attn::fused_attention(q_buf.data(), k_expanded.data(), v_expanded.data(),
                                            context_concat.data(), seq_len, head_dim, n_heads,
                                            /*causal=*/true);
        }
        else
        {
            // Standard MHA: same number of query and KV heads
            // Note: primitives expect Q/K/V with layout [seq_len, n_heads * head_dim]
            // but our buffers have different sizes, so we need temporary expanded buffers
            std::vector<float> k_expanded((size_t)seq_len * total_head_dim, 0.f);
            std::vector<float> v_expanded((size_t)seq_len * total_head_dim, 0.f);

            // Copy K/V to match Q layout (this is a no-op when n_kv_heads == n_heads)
            for (int row = 0; row < seq_len; ++row)
            {
                std::memcpy(k_expanded.data() + (size_t)row * total_head_dim,
                            k_buf.data() + (size_t)row * kv_head_dim,
                            kv_head_dim * sizeof(float));
                std::memcpy(v_expanded.data() + (size_t)row * total_head_dim,
                            v_buf.data() + (size_t)row * kv_head_dim,
                            kv_head_dim * sizeof(float));
            }

            // Apply RoPE to Q and K in-place
            llaminar::attn::apply_rope(q_buf.data(), k_expanded.data(), seq_len, head_dim, n_heads,
                                       n_past_, config_.getLayerConfig().rope_freq_base);

            // Compute full attention: QK^T scores + softmax + apply to V
            llaminar::attn::fused_attention(q_buf.data(), k_expanded.data(), v_expanded.data(),
                                            context_concat.data(), seq_len, head_dim, n_heads,
                                            /*causal=*/true);
        }

        auto attention_end = std::chrono::high_resolution_clock::now();
        timing.attention_ms = std::chrono::duration<double, std::milli>(attention_end - attention_start).count();

        // --- Stage 4: output projection ---
        auto proj_start = std::chrono::high_resolution_clock::now();
        bool ok = adaptiveMatMul(context_concat.data(), weights.wo[layer_idx]->data(), attn_out->data(),
                                 seq_len, hidden_size, total_head_dim, /*is_prefill=*/true,
                                 /*is_decode=*/false, /*transposed_a=*/false, /*transposed_b=*/false,
                                 1.0f, 0.0f);
        if (!ok)
        {
            LOG_ERROR("COSMA attention output projection failed layer=" << layer_idx);
            return false;
        }
        auto proj_end = std::chrono::high_resolution_clock::now();
        timing.linear_ms = std::chrono::duration<double, std::milli>(proj_end - proj_start).count();
        return true;
    }
} // namespace llaminar

// === Section 6: Weight Loading Bridge removed (handled fully inline via bridge decl in header) ===
