/**
 * @file QwenPipeline.cpp
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

#include "QwenPipeline.h"
#include "QwenPipelineAdapter.h" // For QwenModelWeights
#include "PrefillDiagnostics.h"  // For baseline comparison and FFN tracing
#include "PrefillProvider.h"     // For PrefillProviderFactory
#include "KvCacheProvider.h"     // For SimpleKVCacheProvider
#include "ModelLoader.h"
#include "WeightContracts.h" // For contract-driven weight loading
#include "BiasContracts.h"   // For bias dimension validation
#include "operators/MPISwiGLUOperator.h"
#include "operators/MPIRoPEOperator.h"
#include "operators/MPIResidualOperator.h"
#include "operators/MPIEmbeddingOperator.h"
#include "operators/common/RmsnormCore.h"
#include "operators/common/AttentionPrimitives.h"
#include "tensors/TensorFactory.h"
#include "tensors/SimpleTensor.h"
#include "DebugUtils.h"
#include "PerformanceTimer.h"
#include "CosmaPrefillManager.h"
#include "AdaptiveMatmul.h"
#include "BackendSelector.h"
#include "MatmulBackendSelection.h"
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
#include "utils/DebugEnv.h"
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
    // (BufferStats, computeBufferStats, DiffSummary, computeDiffSummary moved to PrefillDiagnostics.h/.cpp)

    // (parity sentinel accessors defined later after the atomic is declared)
    static std::atomic<bool> g_replay_first_exceed{false};

    /**
     * @brief Transpose a 2D tensor (swap rows and columns).
     *
     * This utility is used to convert GGUF projection weights from PyTorch format
     * [out_features, in_features] to C++ matmul format [in_features, out_features].
     *
     * PyTorch convention: output = input @ weight.T (explicit transpose)
     * C++ convention: output = input @ weight (no transpose, weight already correct)
     *
     * @param tensor Input 2D tensor to transpose
     * @return New tensor with dimensions swapped
     * @throws std::runtime_error if tensor is null or not 2D
     */
    std::shared_ptr<llaminar::TensorBase> transpose2D(
        const std::shared_ptr<llaminar::TensorBase> &tensor)
    {
        if (!tensor)
        {
            throw std::runtime_error("transpose2D: tensor is null");
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("transpose2D: expected 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        const int rows = shape[0];
        const int cols = shape[1];
        const float *src = tensor->data();

        // Allocate transposed storage: [rows, cols] → [cols, rows]
        std::vector<float> transposed_data((size_t)rows * cols);

        // Transpose: dst[c, r] = src[r, c]
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                transposed_data[(size_t)c * rows + r] = src[(size_t)r * cols + c];
            }
        }

        // Create new tensor with swapped dimensions
        return std::make_shared<llaminar::SimpleTensor>(
            std::vector<int>{cols, rows}, transposed_data);
    }
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

    // (logFFNRowPreviewIfEnabled and isFFNShardTracingEnabledFor moved to PrefillDiagnostics.h/.cpp)

    /**
     * @brief Helper to capture pipeline stage snapshots for parity testing
     *
     * This inline helper bridges existing capture call sites to the new PipelineSnapshotManager
     * infrastructure. Only active in debug builds when LLAMINAR_PARITY_CAPTURE=1.
     *
     * @param stage Pipeline stage being captured (e.g., EMBEDDING, ATTENTION_NORM)
     * @param layer_idx Layer index (-1 for non-layer stages like embedding/final_norm)
     * @param tensor Tensor to capture (must be valid with data allocated)
     *
     * @note Compile-time no-op in release builds (NDEBUG defined)
     * @note Only rank 0 performs captures to avoid redundant storage
     */
    inline void QwenPipeline::captureIfEnabled(
        PipelineStage stage,
        int layer_idx,
        const std::shared_ptr<TensorBase> &tensor)
    {
        // Early exit if parity capture not enabled (no-op in release builds)
        if (!AbstractPipeline::isParityEnabled())
            return;

        // Only rank 0 captures to avoid redundant multi-rank snapshots
        if (getRank() != 0)
            return;

        // Validate tensor before accessing
        if (!tensor || !tensor->data())
        {
            LOG_WARN("captureIfEnabled: invalid tensor for stage " << static_cast<int>(stage));
            return;
        }

        // Validate tensor has at least 2D shape (seq_len, feature_dim)
        if (tensor->shape().size() < 2)
        {
            LOG_WARN("captureIfEnabled: tensor has insufficient dimensions for stage "
                     << static_cast<int>(stage));
            return;
        }

        int seq_len = tensor->shape()[0];
        int feature_dim = tensor->shape()[1];

        // Delegate to AbstractPipeline base class method which calls PipelineSnapshotManager
        // Use snapshot_source_ from PipelineBase to differentiate batch vs sequential
        AbstractPipeline::captureStageSnapshot(stage, layer_idx, tensor->data(), seq_len, feature_dim, snapshot_source_);
    }

    bool QwenPipeline::executeTransformerLayer(int layer_idx,
                                               std::shared_ptr<TensorBase> &input,
                                               const ModelWeights &weights,
                                               std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("QwenPipeline::executeTransformerLayer");
        int seq_len = input->shape()[0];

        // Initialize thread-local attention instrumentation context (consumed inside MPIAttentionOperator)
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
            if (getRank() == 0)
            {
                LOG_DEBUG("[BACKEND_DECISION_PREFILL] layer=" << layer_idx
                                                              << " use_cosma=" << dec.use_cosma()
                                                              << " seq_len=" << seq_len);
            }
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
                // Attention operator
                auto attention_kernel = dynamic_cast<MPIAttentionOperator *>(getKernel("attention"));
                if (attention_kernel)
                {
                    attention_kernel->setSequencePosition(n_past_);
                    attention_kernel->setLayerIndex(layer_idx);
                    if (debugEnv().pipeline.layer_token_diff_verbose && getRank() == 0)
                    {
                        LOG_INFO("[AttnKernelLayerSet] layer=" << layer_idx << " n_past=" << n_past_);
                    }
                }
                std::vector<std::shared_ptr<TensorBase>> attn_inputs = {attn_norm_out, weights.wq[layer_idx], weights.wk[layer_idx], weights.wv[layer_idx], weights.wo[layer_idx], weights.bq[layer_idx], weights.bk[layer_idx], weights.bv[layer_idx], use_kv_cache_ ? k_cache_[layer_idx] : createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim}), use_kv_cache_ ? v_cache_[layer_idx] : createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim})};
                // Expect 3 outputs: [0]=attention_output, [1]=updated_k_cache, [2]=updated_v_cache
                std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out, nullptr, nullptr};
                if (!executeKernel("attention", attn_inputs, attn_outputs))
                {
                    LOG_ERROR("Layer " << layer_idx << " attention failed");
                    return false;
                }
                // Update KV cache with kernel outputs
                if (use_kv_cache_ && attn_outputs.size() >= 3)
                {
                    if (getRank() == 0 && layer_idx == 0)
                    {
                        LOG_INFO("[CACHE_UPDATE_CONDITION] use_kv_cache=" << use_kv_cache_
                                                                          << " attn_outputs.size()=" << attn_outputs.size()
                                                                          << " attn_outputs[1]=" << (void *)attn_outputs[1].get()
                                                                          << " attn_outputs[2]=" << (void *)attn_outputs[2].get()
                                                                          << " bool(attn_outputs[1])=" << (bool)(attn_outputs[1])
                                                                          << " bool(attn_outputs[2])=" << (bool)(attn_outputs[2]));
                    }

                    if (attn_outputs[1] && attn_outputs[2])
                    {
                        if (getRank() == 0 && layer_idx == 0)
                        {
                            LOG_INFO("[CACHE_UPDATE_DEBUG] BEFORE update:");
                            LOG_INFO("  k_cache_[0] pointer: " << (void *)k_cache_[layer_idx].get());
                            if (k_cache_[layer_idx])
                            {
                                LOG_INFO("  k_cache_[0] shape: [" << k_cache_[layer_idx]->shape()[0] << ", " << k_cache_[layer_idx]->shape()[1] << "]");
                                LOG_INFO("  k_cache_[0] first 10: "
                                         << k_cache_[layer_idx]->data()[0] << " " << k_cache_[layer_idx]->data()[1] << " "
                                         << k_cache_[layer_idx]->data()[2] << " " << k_cache_[layer_idx]->data()[3] << " "
                                         << k_cache_[layer_idx]->data()[4] << " " << k_cache_[layer_idx]->data()[5] << " "
                                         << k_cache_[layer_idx]->data()[6] << " " << k_cache_[layer_idx]->data()[7] << " "
                                         << k_cache_[layer_idx]->data()[8] << " " << k_cache_[layer_idx]->data()[9]);
                            }
                            LOG_INFO("  attn_outputs[1] pointer: " << (void *)attn_outputs[1].get());
                            LOG_INFO("  attn_outputs[1] shape: [" << attn_outputs[1]->shape()[0] << ", " << attn_outputs[1]->shape()[1] << "]");
                            LOG_INFO("  attn_outputs[1] first 10: "
                                     << attn_outputs[1]->data()[0] << " " << attn_outputs[1]->data()[1] << " "
                                     << attn_outputs[1]->data()[2] << " " << attn_outputs[1]->data()[3] << " "
                                     << attn_outputs[1]->data()[4] << " " << attn_outputs[1]->data()[5] << " "
                                     << attn_outputs[1]->data()[6] << " " << attn_outputs[1]->data()[7] << " "
                                     << attn_outputs[1]->data()[8] << " " << attn_outputs[1]->data()[9]);
                        }

                        k_cache_[layer_idx] = attn_outputs[1];
                        v_cache_[layer_idx] = attn_outputs[2];

                        if (getRank() == 0 && layer_idx == 0)
                        {
                            LOG_INFO("[CACHE_UPDATE_DEBUG] AFTER update:");
                            LOG_INFO("  k_cache_[0] pointer: " << (void *)k_cache_[layer_idx].get());
                            LOG_INFO("  k_cache_[0] shape: [" << k_cache_[layer_idx]->shape()[0] << ", " << k_cache_[layer_idx]->shape()[1] << "]");
                            LOG_INFO("  k_cache_[0] first 10: "
                                     << k_cache_[layer_idx]->data()[0] << " " << k_cache_[layer_idx]->data()[1] << " "
                                     << k_cache_[layer_idx]->data()[2] << " " << k_cache_[layer_idx]->data()[3] << " "
                                     << k_cache_[layer_idx]->data()[4] << " " << k_cache_[layer_idx]->data()[5] << " "
                                     << k_cache_[layer_idx]->data()[6] << " " << k_cache_[layer_idx]->data()[7] << " "
                                     << k_cache_[layer_idx]->data()[8] << " " << k_cache_[layer_idx]->data()[9]);
                        }

                        if (debugEnv().pipeline.layer_token_diff_verbose && getRank() == 0)
                        {
                            LOG_DEBUG("[CacheUpdate] layer=" << layer_idx
                                                             << " k_cache_seq_len=" << k_cache_[layer_idx]->shape()[0]
                                                             << " v_cache_seq_len=" << v_cache_[layer_idx]->shape()[0]);
                        }
                    }
                    else
                    {
                        if (getRank() == 0 && layer_idx == 0)
                        {
                            LOG_WARN("[CACHE_UPDATE_SKIPPED] Condition FALSE! attn_outputs[1]=" << (bool)(attn_outputs[1])
                                                                                                << " attn_outputs[2]=" << (bool)(attn_outputs[2]));
                        }
                    }
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

        // FFN fusion path: combine gate+up projections into single matmul
        const bool use_fusion = debugEnv().pipeline.ffn_fusion_enabled && weights.w_gate_up_fused[layer_idx];

        if (use_fusion)
        {
            // Fused gate+up projection: single matmul with 2*d_ff output dimension
            auto fused_out = createLocalTensor({seq_len, 2 * config_.getLayerConfig().d_ff});

            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.getLayerConfig().d_model;
            bctx.n_layers = config_.getLayerConfig().n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);

            if (dec.use_cosma())
            {
                // Fused weight is [2*d_ff, d_model], needs transpose_B
                if (!adaptiveMatMul(ffn_norm_out->data(), weights.w_gate_up_fused[layer_idx]->data(), fused_out->data(),
                                    seq_len, 2 * config_.getLayerConfig().d_ff, config_.getLayerConfig().d_model,
                                    is_prefill_stage_, false, false, true, 1.0f, 0.0f))
                {
                    LOG_ERROR("fused gate+up projection failed cosma");
                    return false;
                }
            }
            else
            {
                std::vector<std::shared_ptr<TensorBase>> fused_inputs = {ffn_norm_out, weights.w_gate_up_fused[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> fused_outputs = {fused_out};
                if (!executeKernel("linear", fused_inputs, fused_outputs))
                {
                    LOG_ERROR("fused gate+up projection failed");
                    return false;
                }
            }

            // Split fused output into gate and up tensors
            const float *fused_data = fused_out->data();
            float *gate_data = const_cast<float *>(gate_out->data());
            float *up_data = const_cast<float *>(up_out->data());
            size_t d_ff = config_.getLayerConfig().d_ff;

// Copy first half to gate_out, second half to up_out
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
                    // w_gate stored as [d_ff(out), d_model(in)] -> need transpose_B
                    if (!adaptiveMatMul(ffn_norm_out->data(), weights.w_gate[layer_idx]->data(), gate_out->data(), seq_len, config_.getLayerConfig().d_ff, config_.getLayerConfig().d_model, is_prefill_stage_, false, false, true, 1.0f, 0.0f))
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
                    // w_up stored as [d_ff(out), d_model(in)] -> need transpose_B
                    if (!adaptiveMatMul(ffn_norm_out->data(), weights.w_up[layer_idx]->data(), up_out->data(), seq_len, config_.getLayerConfig().d_ff, config_.getLayerConfig().d_model, is_prefill_stage_, false, false, true, 1.0f, 0.0f))
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
        }

        // Parity capture: FFN gate projection output
        captureIfEnabled(PipelineStage::FFN_GATE, layer_idx, gate_out);

        // Parity capture: FFN up projection output
        captureIfEnabled(PipelineStage::FFN_UP, layer_idx, up_out);

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
        // Parity capture: FFN SwiGLU activation output (gate * silu(up))
        captureIfEnabled(PipelineStage::FFN_SWIGLU, layer_idx, swiglu_out);
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
                // w_down stored as [d_model(out), d_ff(in)] -> need transpose_B
                if (!adaptiveMatMul(swiglu_out->data(), weights.w_down[layer_idx]->data(), ffn_out->data(), seq_len, config_.getLayerConfig().d_model, config_.getLayerConfig().d_ff, is_prefill_stage_, false, false, true, 1.0f, 0.0f))
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
    // (isFFNShardTracingEnabledFor and PrefillBaselineRegistry moved to PrefillDiagnostics.h/.cpp)

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
            auto embedding_kernel = std::make_unique<MPIEmbeddingOperator>(config_.getLayerConfig().vocab_size, config_.getLayerConfig().d_model);
            if (!registerOperator("embedding", std::move(embedding_kernel)))
                throw std::runtime_error("Failed to register Embedding kernel");
        }
        auto rmsnorm_kernel = std::make_unique<MPIRMSNormOperator>(MPIRMSNormOperator::DistributionStrategy::SEQUENCE_WISE);
        rmsnorm_kernel->setEpsilon(config_.getLayerConfig().eps);
        if (!registerOperator("rmsnorm", std::move(rmsnorm_kernel)))
            throw std::runtime_error("Failed to register RMSNorm operator");

        auto attention_kernel = std::make_unique<MPIAttentionOperator>(config_.getLayerConfig().n_head, config_.getLayerConfig().n_head_kv, config_.getLayerConfig().head_dim, config_.getLayerConfig().rope_freq_base);

        // CRITICAL FIX: Must set GatherHeadsPostProjection mode for multi-rank execution!
        //
        // Background: MPIAttentionOperator defaults to LocalHeads mode, which was designed for
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
        attention_kernel->setOutputMode(MPIAttentionOperator::AttentionOutputMode::GatherHeadsPostProjection);

        // Wire up snapshot callback for intermediate attention stages (Q/K/V proj, RoPE, scores, etc.)
        attention_kernel->setSnapshotCallback([this](PipelineStage stage, int layer_idx, const float *data, int seq_len, int feature_dim)
                                              { AbstractPipeline::captureStageSnapshot(stage, layer_idx, data, seq_len, feature_dim, snapshot_source_); });

        if (!registerOperator("attention", std::move(attention_kernel)))
            throw std::runtime_error("Failed to register Attention operator");

        auto linear_kernel = std::make_unique<MPILinearOperator>();
        if (!registerOperator("linear", std::move(linear_kernel)))
            throw std::runtime_error("Failed to register Linear kernel");

        auto swiglu_kernel = std::make_unique<MPISwiGLUOperator>(MPISwiGLUOperator::DistributionStrategy::SEQUENCE_WISE);
        if (!registerOperator("swiglu", std::move(swiglu_kernel)))
            throw std::runtime_error("Failed to register SwiGLU kernel");

        auto rope_kernel = std::make_unique<MPIRoPEOperator>(config_.getLayerConfig().max_seq_len, config_.getLayerConfig().head_dim, config_.getLayerConfig().rope_freq_base, MPIRoPEOperator::DistributionStrategy::SEQUENCE_WISE);
        if (!registerOperator("rope", std::move(rope_kernel)))
            throw std::runtime_error("Failed to register RoPE kernel");

        auto residual_kernel = std::make_unique<MPIResidualOperator>(MPIResidualOperator::DistributionStrategy::SEQUENCE_WISE);
        if (!registerOperator("residual", std::move(residual_kernel)))
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
    QwenPipeline::ModelWeights loadModelWeights_batch_bridge(
        ModelLoader &loader,
        const QwenPipeline::LayerConfig &config)
    {
        QwenPipeline::ModelWeights weights;
        LOG_INFO("[WeightLoad:BATCH] begin vocab=" << config.vocab_size << " d_model=" << config.d_model << " layers=" << config.n_layers);

        // Load token embedding (REPLICATED)
        weights.token_embedding = loader.loadTensor("token_embd.weight");
        if (!weights.token_embedding)
            throw std::runtime_error("Failed to load token_embd.weight");

        // Load output norm (REPLICATED)
        weights.output_norm_weight = loader.loadTensor("output_norm.weight");
        if (!weights.output_norm_weight)
            throw std::runtime_error("Failed to load output_norm.weight");

        // Load lm_head (REPLICATED)
        weights.lm_head = loader.loadTensor("output.weight");
        if (!weights.lm_head)
        {
            LOG_WARN("[WeightLoad:BATCH] output.weight not found, using tied embeddings");
            weights.lm_head = weights.token_embedding;
        }

        // Pre-allocate layer weights
        weights.attn_norm_weight.reserve(config.n_layers);
        weights.wq.reserve(config.n_layers);
        weights.wk.reserve(config.n_layers);
        weights.wv.reserve(config.n_layers);
        weights.wo.reserve(config.n_layers);
        weights.bq.reserve(config.n_layers);
        weights.bk.reserve(config.n_layers);
        weights.bv.reserve(config.n_layers);
        weights.ffn_norm_weight.reserve(config.n_layers);
        weights.w_gate.reserve(config.n_layers);
        weights.w_up.reserve(config.n_layers);
        weights.w_down.reserve(config.n_layers);

        LOG_INFO("[WeightLoad:BATCH] Loading " << config.n_layers << " layers with REPLICATED weights (batch mode)");

        // Get batch-mode contracts (all REPLICATED)
        auto contracts = llaminar::getBatchQwenWeightContracts();

        int mpi_rank = 0, mpi_size = 1;
#ifdef LLAMINAR_HAVE_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        }
#endif

        // Contract indices
        const int IDX_ATTN_NORM = 0;
        const int IDX_Q = 1;
        const int IDX_K = 2;
        const int IDX_V = 3;
        const int IDX_O = 4;
        const int IDX_FFN_NORM = 5;
        const int IDX_W_GATE = 6;
        const int IDX_W_UP = 7;
        const int IDX_W_DOWN = 8;

        for (int layer = 0; layer < config.n_layers; ++layer)
        {
            std::string prefix = "blk." + std::to_string(layer) + ".";

            // Load all tensors as REPLICATED (full copy on every rank)
            auto attn_norm = contracts.layer_weights[IDX_ATTN_NORM].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto wq = contracts.layer_weights[IDX_Q].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto wk = contracts.layer_weights[IDX_K].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto wv = contracts.layer_weights[IDX_V].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto wo = contracts.layer_weights[IDX_O].load(
                loader, config, mpi_rank, mpi_size, layer);

            // Load biases (REPLICATED - batch operators handle distribution)
            auto bq = loader.loadTensor(prefix + "attn_q.bias");
            auto bk = loader.loadTensor(prefix + "attn_k.bias");
            auto bv = loader.loadTensor(prefix + "attn_v.bias");

            auto ffn_norm = contracts.layer_weights[IDX_FFN_NORM].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto w_gate = contracts.layer_weights[IDX_W_GATE].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto w_up = contracts.layer_weights[IDX_W_UP].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto w_down = contracts.layer_weights[IDX_W_DOWN].load(
                loader, config, mpi_rank, mpi_size, layer);

            // Store weights
            weights.attn_norm_weight.push_back(attn_norm);
            weights.wq.push_back(wq);
            weights.wk.push_back(wk);
            weights.wv.push_back(wv);
            weights.wo.push_back(wo);
            weights.bq.push_back(bq ? bq : std::make_shared<SimpleTensor>(std::vector<int>{1}));
            weights.bk.push_back(bk ? bk : std::make_shared<SimpleTensor>(std::vector<int>{1}));
            weights.bv.push_back(bv ? bv : std::make_shared<SimpleTensor>(std::vector<int>{1}));
            weights.ffn_norm_weight.push_back(ffn_norm);
            weights.w_gate.push_back(w_gate);
            weights.w_up.push_back(w_up);
            weights.w_down.push_back(w_down);
            weights.w_gate_up_fused.push_back(nullptr); // Not used in batch mode

            if (mpi_rank == 0 && layer == 0)
            {
                LOG_INFO("[WeightLoad:BATCH] Layer 0 weight shapes:");
                LOG_INFO("  wq: [" << wq->shape()[0] << ", " << wq->shape()[1] << "]");
                LOG_INFO("  wk: [" << wk->shape()[0] << ", " << wk->shape()[1] << "]");
                LOG_INFO("  wv: [" << wv->shape()[0] << ", " << wv->shape()[1] << "]");
                LOG_INFO("  wo: [" << wo->shape()[0] << ", " << wo->shape()[1] << "]");
            }
        }

        if (mpi_rank == 0)
        {
            LOG_INFO("[WeightLoad:BATCH] Successfully loaded all " << config.n_layers << " layers");
        }

        return weights;
    }

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

        // Log first few embedding values to verify consistency across ranks
        {
            // Note: This is a free function, so we query MPI directly
            int mpi_rank = 0;
#ifdef LLAMINAR_HAVE_MPI
            int mpi_initialized = 0;
            MPI_Initialized(&mpi_initialized);
            if (mpi_initialized)
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            }
#endif
            const float *emb_ptr = weights.token_embedding->data();
            LOG_INFO("[WeightLoad] rank=" << mpi_rank
                                          << " token_embd first 10 values: ["
                                          << emb_ptr[0] << ", " << emb_ptr[1] << ", " << emb_ptr[2] << ", "
                                          << emb_ptr[3] << ", " << emb_ptr[4] << ", " << emb_ptr[5] << ", "
                                          << emb_ptr[6] << ", " << emb_ptr[7] << ", " << emb_ptr[8] << ", "
                                          << emb_ptr[9] << "]");
            // Also log some values from token 1 embedding (offset 896)
            const float *token1_emb = emb_ptr + 896;
            LOG_INFO("[WeightLoad] rank=" << mpi_rank
                                          << " token_embd[1] first 10 values: ["
                                          << token1_emb[0] << ", " << token1_emb[1] << ", " << token1_emb[2] << ", "
                                          << token1_emb[3] << ", " << token1_emb[4] << ", " << token1_emb[5] << ", "
                                          << token1_emb[6] << ", " << token1_emb[7] << ", " << token1_emb[8] << ", "
                                          << token1_emb[9] << "]");
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

                // Enforce expected layout [vocab_size, d_model] = [out, in] per new convention
                if (!raw)
                    enforce_matrix_layout_compat(lm_head, config.vocab_size, config.d_model, "output.weight");

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

            // With our new convention, token_embedding is already [vocab_size, d_model] = [out, in]
            // This is exactly what we need for lm_head! No transpose needed.
            auto emb_shape = weights.token_embedding->shape();
            if (emb_shape.size() == 2 && emb_shape[0] == config.vocab_size && emb_shape[1] == config.d_model)
            {
                // Use tied embedding directly - already in correct orientation
                weights.lm_head = weights.token_embedding;
                LOG_INFO("[WeightLoad] Using tied embeddings for lm_head (no transpose needed): ["
                         << emb_shape[0] << ", " << emb_shape[1] << "]");
            }
            else
            {
                LOG_ERROR("[WeightLoad] Token embedding has unexpected shape for tied lm_head: ["
                          << emb_shape[0] << "," << emb_shape[1] << "] - expected ["
                          << config.vocab_size << ", " << config.d_model << "]");
                weights.lm_head = weights.token_embedding; // fallback
            }
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
        weights.w_gate_up_fused.reserve(config.n_layers);

        LOG_INFO("[WeightLoad] per-layer loading start layers=" << config.n_layers);

        // Get MPI context for weight slicing
        int mpi_rank = 0, mpi_size = 1;
#ifdef LLAMINAR_HAVE_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        }
#endif

        // Get Qwen weight contracts for contract-driven loading
        auto contracts = llaminar::getQwenWeightContracts();

        // Weight indices in contracts.layer_weights
        const int IDX_ATTN_NORM = 0;
        const int IDX_Q = 1;
        const int IDX_K = 2;
        const int IDX_V = 3;
        const int IDX_O = 4;
        const int IDX_FFN_NORM = 5;
        const int IDX_W_GATE = 6;
        const int IDX_W_UP = 7;
        const int IDX_W_DOWN = 8;

        if (mpi_rank == 0)
        {
            LOG_INFO("[WeightLoad] Using contract-driven loading with MPI world_size=" << mpi_size);
        }

        for (int layer = 0; layer < config.n_layers; ++layer)
        {
            // Construct tensor name prefix for this layer
            std::string prefix = "blk." + std::to_string(layer) + ".";

            // --- Attention Normalization ---
            // Contract-driven loading handles everything automatically
            auto attn_norm = contracts.layer_weights[IDX_ATTN_NORM].load(
                loader, config, mpi_rank, mpi_size, layer);

            // Debug override: force unit gamma if requested
            if (debugEnv().output_norm.force_unit_all)
            {
                float *g = const_cast<float *>(attn_norm->data());
                for (int i = 0; i < attn_norm->size(); ++i)
                    g[i] = 1.0f;
            }
            weights.attn_norm_weight.push_back(attn_norm);

            // --- Attention Projection Matrices ---
            // Contract-driven loading automatically handles:
            // - GGUF vs PyTorch dimension conventions
            // - Row/column slicing based on MPI context
            // - Data transposition when needed
            // - Shape validation

            auto wq = contracts.layer_weights[IDX_Q].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto wk = contracts.layer_weights[IDX_K].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto wv = contracts.layer_weights[IDX_V].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto wo = contracts.layer_weights[IDX_O].load(
                loader, config, mpi_rank, mpi_size, layer);

            // Load Q, K, V projection biases as REPLICATED, then slice them for this rank
            // Biases are 1D vectors aligned with the row dimension of their respective weights
            auto bq_full = loader.loadTensor(prefix + "attn_q.bias");
            auto bk_full = loader.loadTensor(prefix + "attn_k.bias");
            auto bv_full = loader.loadTensor(prefix + "attn_v.bias");

            if (!bq_full || !bk_full || !bv_full)
                throw std::runtime_error("Failed to load attention biases for layer " + std::to_string(layer));

            // Calculate head distribution for this rank (same logic as MPIAttentionOperator)
            auto calc_head_distribution = [](int total_heads, int rank, int world_size) -> std::pair<int, int>
            {
                int heads_per_rank = total_heads / world_size;
                int remainder = total_heads % world_size;
                int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
                int head_offset = rank * heads_per_rank + std::min(rank, remainder);
                return {local_heads, head_offset};
            };

            auto [local_q_heads, q_head_offset] = calc_head_distribution(config.n_head, mpi_rank, mpi_size);
            auto [local_kv_heads, kv_head_offset] = calc_head_distribution(config.n_head_kv, mpi_rank, mpi_size);

            const int head_dim = config.d_model / config.n_head;
            const int local_q_dim = local_q_heads * head_dim;
            const int local_kv_dim = local_kv_heads * head_dim;

            // Create bias contracts for validation
            const int full_q_dim = config.n_head * head_dim;
            const int full_kv_dim = config.n_head_kv * head_dim;

            BiasContract bq_contract("blk." + std::to_string(layer) + ".attn_q.bias",
                                     "Q projection bias (head-sliced)",
                                     full_q_dim, local_q_dim, mpi_rank, mpi_size);
            BiasContract bk_contract("blk." + std::to_string(layer) + ".attn_k.bias",
                                     "K projection bias (head-sliced)",
                                     full_kv_dim, local_kv_dim, mpi_rank, mpi_size);
            BiasContract bv_contract("blk." + std::to_string(layer) + ".attn_v.bias",
                                     "V projection bias (head-sliced)",
                                     full_kv_dim, local_kv_dim, mpi_rank, mpi_size);

            // Validate full bias dimensions BEFORE slicing
            if (!bq_contract.validate_full(bq_full, layer, prefix + "attn_q.bias") ||
                !bk_contract.validate_full(bk_full, layer, prefix + "attn_k.bias") ||
                !bv_contract.validate_full(bv_full, layer, prefix + "attn_v.bias"))
            {
                throw std::runtime_error("Bias dimension validation failed at layer " + std::to_string(layer));
            }

            // Pre-slice biases to match local head dimensions
            // This happens ONCE at load time, not every forward pass
            std::shared_ptr<TensorBase> bq, bk, bv;

            if (mpi_size > 1)
            {
                // Multi-rank: slice biases
                const int bq_offset = q_head_offset * head_dim;
                const int bk_offset = kv_head_offset * head_dim;
                const int bv_offset = kv_head_offset * head_dim;

                bq = TensorFactory::create_simple({local_q_dim});
                bk = TensorFactory::create_simple({local_kv_dim});
                bv = TensorFactory::create_simple({local_kv_dim});

                memcpy(bq->data(), bq_full->data() + bq_offset, local_q_dim * sizeof(float));
                memcpy(bk->data(), bk_full->data() + bk_offset, local_kv_dim * sizeof(float));
                memcpy(bv->data(), bv_full->data() + bv_offset, local_kv_dim * sizeof(float));

                if (layer == 0)
                {
                    LOG_INFO("[BIAS_SLICE] Layer " << layer << " Rank " << mpi_rank);
                    LOG_INFO("  Q bias: full[" << bq_full->size() << "] -> local[" << bq->size()
                                               << "] offset=" << bq_offset);
                    LOG_INFO("  K bias: full[" << bk_full->size() << "] -> local[" << bk->size()
                                               << "] offset=" << bk_offset);
                    LOG_INFO("  V bias: full[" << bv_full->size() << "] -> local[" << bv->size()
                                               << "] offset=" << bv_offset);
                    LOG_INFO("  bq first 3: [" << bq->data()[0] << ", " << bq->data()[1] << ", "
                                               << bq->data()[2] << "]");
                }
            }
            else
            {
                // Single rank: use full biases
                bq = bq_full;
                bk = bk_full;
                bv = bv_full;
            }

            // Validate sliced bias dimensions AFTER slicing
            if (!bq_contract.validate(bq, layer, prefix + "attn_q.bias") ||
                !bk_contract.validate(bk, layer, prefix + "attn_k.bias") ||
                !bv_contract.validate(bv, layer, prefix + "attn_v.bias"))
            {
                throw std::runtime_error("Sliced bias dimension validation failed at layer " + std::to_string(layer));
            }

            // Store attention weights for this layer
            weights.wq.push_back(wq);
            weights.wk.push_back(wk);
            weights.wv.push_back(wv);
            weights.wo.push_back(wo);

            // Store PRE-SLICED attention biases
            weights.bq.push_back(bq);
            weights.bk.push_back(bk);
            weights.bv.push_back(bv);

            // --- FFN Normalization ---
            // Contract-driven loading
            auto ffn_norm = contracts.layer_weights[IDX_FFN_NORM].load(
                loader, config, mpi_rank, mpi_size, layer);

            // Debug override: force unit gamma if requested
            if (debugEnv().output_norm.force_unit_all)
            {
                float *g = const_cast<float *>(ffn_norm->data());
                for (int i = 0; i < ffn_norm->size(); ++i)
                    g[i] = 1.0f;
            }
            weights.ffn_norm_weight.push_back(ffn_norm);

            // --- FFN Projection Matrices (SwiGLU) ---
            // Contract-driven loading automatically handles GGUF->PyTorch conversion and transposition
            auto w_gate = contracts.layer_weights[IDX_W_GATE].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto w_up = contracts.layer_weights[IDX_W_UP].load(
                loader, config, mpi_rank, mpi_size, layer);
            auto w_down = contracts.layer_weights[IDX_W_DOWN].load(
                loader, config, mpi_rank, mpi_size, layer);

            // Store FFN weights for this layer
            weights.w_gate.push_back(w_gate);
            weights.w_up.push_back(w_up);
            weights.w_down.push_back(w_down);

            // Create fused gate+up weight for FFN optimization (if fusion enabled)
            if (debugEnv().pipeline.ffn_fusion_enabled)
            {
                // Get dimensions: w_gate and w_up are both [local_d_ff, d_model]
                const auto &gate_shape = w_gate->shape();
                const auto &up_shape = w_up->shape();

                if (gate_shape.size() != 2 || up_shape.size() != 2)
                {
                    LOG_ERROR("[FFN_FUSION] Unexpected weight shape: gate=" << gate_shape.size() << "D, up=" << up_shape.size() << "D");
                    throw std::runtime_error("FFN fusion requires 2D weights");
                }

                size_t local_d_ff = gate_shape[0];
                size_t d_model = gate_shape[1];

                // Verify up weight has same dimensions
                if (up_shape[0] != local_d_ff || up_shape[1] != d_model)
                {
                    LOG_ERROR("[FFN_FUSION] Shape mismatch: gate=[" << local_d_ff << "," << d_model
                                                                    << "], up=[" << up_shape[0] << "," << up_shape[1] << "]");
                    throw std::runtime_error("FFN fusion requires matching gate/up shapes");
                }

                // Allocate fused weight: [2*local_d_ff, d_model]
                auto w_fused = TensorFactory::create_simple({static_cast<int>(2 * local_d_ff), static_cast<int>(d_model)});
                float *fused_data = const_cast<float *>(w_fused->data());
                const float *gate_data = w_gate->data();
                const float *up_data = w_up->data();

                // Copy gate weights to first half
                std::memcpy(fused_data, gate_data, local_d_ff * d_model * sizeof(float));

                // Copy up weights to second half
                std::memcpy(fused_data + (local_d_ff * d_model), up_data, local_d_ff * d_model * sizeof(float));

                weights.w_gate_up_fused.push_back(w_fused);

                if (mpi_rank == 0 && layer == 0)
                {
                    LOG_INFO("[FFN_FUSION] Enabled: fused_shape=[" << (2 * local_d_ff) << "," << d_model
                                                                   << "], memory_savings=" << (local_d_ff * d_model * sizeof(float) / 1024 / 1024) << "MB per matmul");
                }
            }
            else
            {
                // Fusion disabled: push empty placeholder to maintain vector alignment
                weights.w_gate_up_fused.push_back(nullptr);
            }
        }
        LOG_INFO("[WeightLoad] per-layer loading complete");
        return weights;

        // (Helper forward declarations below weight loader section)
    }

    // (handle_prefill_stage_snapshot moved to PrefillDiagnostics.h/.cpp)

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

        // Parity capture: embedding output
        captureIfEnabled(PipelineStage::EMBEDDING, -1, embedded_output);

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

        if (getRank() == 0)
        {
            float sum_sq = 0.0f;
            for (size_t i = 0; i < output->size(); ++i)
            {
                sum_sq += output->data()[i] * output->data()[i];
            }
            float l2_norm = std::sqrt(sum_sq / output->size());
            LOG_DEBUG("[MAGNITUDE_TRACE_SEQ] Rank0 FINAL LOGITS (sequential): L2_norm=" << l2_norm << " size=" << output->size()
                                                                                        << " first_5=[" << output->data()[0] << "," << output->data()[1] << ","
                                                                                        << output->data()[2] << "," << output->data()[3] << "," << output->data()[4] << "]");
        }

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

        int rank = getRank();
        if (rank == 0)
        {
            LOG_INFO("[CACHE_INIT_DEBUG] initializeKVCache called with seq_len=" << seq_len
                                                                                 << " current_capacity=" << kv_cache_state_.capacity_tokens
                                                                                 << " current_used=" << kv_cache_state_.used_tokens);
        }

        k_cache_.resize(config_.getLayerConfig().n_layers);
        v_cache_.resize(config_.getLayerConfig().n_layers);
        for (int l = 0; l < config_.getLayerConfig().n_layers; ++l)
        {
            bool recreated_k = false;
            bool recreated_v = false;

            if (!k_cache_[l] || k_cache_[l]->shape()[0] < seq_len)
            {
                k_cache_[l] = createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim});
                recreated_k = true;
            }
            if (!v_cache_[l] || v_cache_[l]->shape()[0] < seq_len)
            {
                v_cache_[l] = createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim});
                recreated_v = true;
            }

            if (rank == 0 && l == 0 && (recreated_k || recreated_v))
            {
                LOG_DEBUG("[CACHE_INIT_DEBUG] Layer 0: Recreated cache tensors! This will WIPE existing cache data!");
                LOG_DEBUG("  k_cache recreated: " << (recreated_k ? "YES" : "no"));
                LOG_DEBUG("  v_cache recreated: " << (recreated_v ? "YES" : "no"));
            }
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
            if (auto attn = dynamic_cast<MPIAttentionOperator *>(getKernel("attention")))
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

    // (handle_prefill_stage_snapshot moved to PrefillDiagnostics.h/.cpp)
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

        // Use PrefillProviderFactory for optimal backend selection
        auto provider = PrefillProviderFactory::create(
            config_,
            mpi_ctx_,
            static_cast<int>(tokens.size()));

        if (!provider)
        {
            LOG_ERROR("prefill: Failed to create PrefillProvider");
            return false;
        }

        // Create cache provider if KV cache enabled
        SimpleKVCacheProvider cache_provider;
        const int n_layers = config_.getLayerConfig().n_layers;
        if (use_kv_cache_)
        {
            const int kv_head_dim = (config_.getLayerConfig().n_head_kv / mpi_ctx_.size) *
                                    config_.getLayerConfig().head_dim;
            cache_provider.reserve(n_layers, tokens.size(), kv_head_dim);
        }

        // Execute prefill via provider (with cache capture if enabled)
        std::shared_ptr<TensorBase> output;
        PrefillMetrics metrics;

        // DEBUG: Uncomment for detailed provider execution tracing
        // std::cerr << "!!!!! [QwenPipeline] ABOUT TO CALL provider->execute, seq_len=" << tokens.size() << std::endl << std::flush;

        bool success = provider->execute(tokens, weights_iface, output, ctx, metrics,
                                         use_kv_cache_ ? &cache_provider : nullptr);

        // DEBUG: Uncomment for detailed provider execution tracing
        // std::cerr << "!!!!! [QwenPipeline] RETURNED FROM provider->execute, success=" << success << std::endl << std::flush;

        if (!success)
        {
            LOG_ERROR("prefill: Provider execution failed");
            return false;
        }

        // Transfer cache from provider to pipeline storage
        if (use_kv_cache_)
        {
            const auto &k_caches = cache_provider.getKCache();
            const auto &v_caches = cache_provider.getVCache();

            // Initialize cache vectors if needed
            if (k_cache_.empty())
            {
                k_cache_.resize(n_layers);
                v_cache_.resize(n_layers);
            }

            // Copy cache references (shared_ptr assignment, zero-copy)
            for (int i = 0; i < n_layers; ++i)
            {
                if (cache_provider.hasCache(i))
                {
                    k_cache_[i] = k_caches[i];
                    v_cache_[i] = v_caches[i];
                }
            }
        }

        // Log metrics
        if (getRank() == 0)
        {
            LOG_INFO("Prefill metrics [" << provider->name() << "]: "
                                         << "total=" << metrics.total_ms() << "ms, "
                                         << "embed=" << metrics.embedding_ms << "ms, "
                                         << "norm=" << metrics.norm_ms << "ms, "
                                         << "attn=" << metrics.attention_ms << "ms, "
                                         << "ffn=" << metrics.ffn_ms << "ms, "
                                         << "lm_head=" << metrics.lm_head_ms << "ms, "
                                         << "snapshots=" << metrics.snapshots_captured);
        }

        // Store output logits
        last_logits_ = output;

        // Update KV cache state
        kv_snapshot_.capacity_tokens = getKVCacheCapacity();
        kv_snapshot_.used_tokens = getKVCacheUsed();
        kv_snapshot_.growth_events = getKVCacheGrowthEvents();

        // CRITICAL: Update n_past_ for incremental decode to work correctly
        // The incremental decode relies on n_past_ to know the current KV cache position
        if (use_kv_cache_)
        {
            n_past_ = (int)tokens.size();
        }

        // Update stage context
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

        // Conditional debug logging for embedding details (parity debugging)
        if (env.pipeline.debug_decode_embed && getRank() == 0)
        {
            LOG_INFO("[DECODE_EMBED_DEBUG] token_id=" << token_id
                                                      << " embedding_shape=[1," << current->shape()[1] << "]"
                                                      << " first_10=[" << current->data()[0] << "," << current->data()[1] << "," << current->data()[2]
                                                      << "," << current->data()[3] << "," << current->data()[4] << "," << current->data()[5]
                                                      << "," << current->data()[6] << "," << current->data()[7] << "," << current->data()[8]
                                                      << "," << current->data()[9] << "]");
        }

        // Capture embedding snapshot for parity testing
        captureIfEnabled(PipelineStage::EMBEDDING, -1, current);

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
            if (auto attn = dynamic_cast<MPIAttentionOperator *>(getKernel("attention")))
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
     *  3. Apply RoPE and compute attention using optimized primitives (AttentionPrimitives.h)
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
            LOG_ERROR("Layer " << layer_idx << " attention norm failed");
            return false;
        }

        auto norm_end = std::chrono::high_resolution_clock::now();
        timing.norm_ms = std::chrono::duration<double, std::milli>(norm_end - norm_start).count();

        // Parity capture: attention norm output (input to QKV)
        captureIfEnabled(PipelineStage::ATTENTION_NORM, layer_idx, attn_norm_out);

        // --- Stage 2: Attention via MPIAttentionOperator with COSMA backend ---
        auto attention_start = std::chrono::high_resolution_clock::now();

        // Configure attention kernel
        auto attention_kernel = dynamic_cast<MPIAttentionOperator *>(getKernel("attention"));
        if (attention_kernel)
        {
            attention_kernel->setSequencePosition(n_past_);
            attention_kernel->setLayerIndex(layer_idx);
            attention_kernel->setCosmaManager(&manager); // INJECT COSMA BACKEND!
        }

        // Prepare KV cache tensors (same as OpenBLAS path)
        auto k_cache = use_kv_cache_ && k_cache_[layer_idx]
                           ? k_cache_[layer_idx]
                           : createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim});
        auto v_cache = use_kv_cache_ && v_cache_[layer_idx]
                           ? v_cache_[layer_idx]
                           : createLocalTensor({seq_len, config_.getLayerConfig().n_head_kv * config_.getLayerConfig().head_dim});

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
        std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out, nullptr, nullptr};

        if (!executeKernel("attention", attn_inputs, attn_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " attention failed");
            return false;
        }

        // Update KV cache with kernel outputs (same as OpenBLAS path!)
        if (use_kv_cache_ && attn_outputs.size() >= 3)
        {
            if (attn_outputs[1] && attn_outputs[2])
            {
                k_cache_[layer_idx] = attn_outputs[1];
                v_cache_[layer_idx] = attn_outputs[2];

                if (getRank() == 0 && debugEnv().pipeline.layer_token_diff_verbose)
                {
                    LOG_DEBUG("[CacheUpdate] COSMA path: layer=" << layer_idx
                                                                 << " k_cache_seq_len=" << k_cache_[layer_idx]->shape()[0]
                                                                 << " v_cache_seq_len=" << v_cache_[layer_idx]->shape()[0]);
                }
            }
        }

        auto attention_end = std::chrono::high_resolution_clock::now();
        timing.attention_ms = std::chrono::duration<double, std::milli>(attention_end - attention_start).count();

        // Note: Output projection is now handled inside MPIAttentionOperator
        // No need for separate adaptiveMatMul call
        timing.linear_ms = 0.0; // Included in attention_ms

        return true;
    }
} // namespace llaminar

// === Batch Interface Stubs (redirect to sequential) ===
namespace llaminar
{
    bool QwenPipeline::prefillBatch(const std::vector<std::vector<int>> &token_batches,
                                    const IModelWeights &weights,
                                    StageContext &ctx,
                                    std::shared_ptr<TensorBase> &out_logits)
    {
        // Sequential fallback: process each sequence independently
        if (token_batches.empty())
            return false;

        // For now, just process first sequence (full implementation would loop)
        StageContext temp_ctx;
        bool success = prefill(token_batches[0], weights, temp_ctx);
        if (success)
        {
            return logits(out_logits);
        }
        return false;
    }

    bool QwenPipeline::decodeBatch(const std::vector<int> &next_tokens,
                                   const IModelWeights &weights,
                                   StageContext &ctx,
                                   std::shared_ptr<TensorBase> &out_logits)
    {
        // Sequential fallback: process only first token
        if (next_tokens.empty())
            return false;

        StageContext temp_ctx;
        bool success = decode(next_tokens[0], weights, temp_ctx);
        if (success)
        {
            return logits(out_logits);
        }
        return false;
    }

    void QwenPipeline::ensureBatchState(int batch_size)
    {
        // Stub for now - sequential pipeline doesn't need batch state
    }

    void QwenPipeline::clearBatchState()
    {
        // Stub
    }
}

// === Section 6: Weight Loading Bridge removed (handled fully inline via bridge decl in header) ===
