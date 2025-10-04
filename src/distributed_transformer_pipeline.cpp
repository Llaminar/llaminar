// Canonical implementation translation unit for DistributedTransformerPipeline.
// Formerly implemented in mpi_transformer_pipeline.cpp (now a deprecation shim).
// NOTE: Original history remains in that file; future refactors will operate here.

#include "distributed_transformer_pipeline.h"
#include "model_loader.h"
#include "kernels/MPISwiGLUKernel.h"
#include "kernels/MPIRoPEKernel.h"
#include "kernels/MPIResidualKernel.h"
#include "kernels/MPIEmbeddingKernel.h"
#include "kernels/common/rmsnorm_core.h"
#include "tensors/tensor_factory.h"
#include "tensors/simple_tensor.h"
#include "debug_utils.h"
#include "performance_timer.h"
#include "cosma_prefill_manager.h"
#include "adaptive_matmul.h"
#include "backend_selector.h"
using llaminar::BackendContext;
using llaminar::BackendDecision;
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
    struct BufferStats
    {
        double min = 0.0;
        double max = 0.0;
        double mean = 0.0;
        double l2 = 0.0;
        double rms = 0.0;
        double stddev = 0.0;
    };

    BufferStats computeBufferStats(const float *data, size_t size)
    {
        BufferStats stats;
        if (!data || size == 0)
            return stats;
        double sum = 0.0;
        double sumsq = 0.0;
        double min_v = static_cast<double>(data[0]);
        double max_v = static_cast<double>(data[0]);
        for (size_t i = 0; i < size; ++i)
        {
            double v = static_cast<double>(data[i]);
            sum += v;
            sumsq += v * v;
            if (v < min_v)
                min_v = v;
            if (v > max_v)
                max_v = v;
        }
        double mean = sum / static_cast<double>(size);
        double variance = std::max(0.0, sumsq / static_cast<double>(size) - mean * mean);
        stats.min = min_v;
        stats.max = max_v;
        stats.mean = mean;
        stats.rms = std::sqrt(sumsq / static_cast<double>(size));
        stats.l2 = std::sqrt(sumsq);
        stats.stddev = std::sqrt(variance);
        return stats;
    }

    struct DiffSummary
    {
        double max_abs = 0.0;
        double mean_abs = 0.0;
        double rel_l2 = 0.0;
        size_t worst_index = 0;
        float value_a = 0.0f;
        float value_b = 0.0f;
    };

    DiffSummary computeDiffSummary(const float *a, const float *b, size_t size)
    {
        DiffSummary summary;
        if (!a || !b || size == 0)
            return summary;
        double max_abs = 0.0;
        size_t worst = 0;
        float worst_a = 0.0f;
        float worst_b = 0.0f;
        double sum_abs = 0.0;
        long double sum_sq = 0.0L;
        long double denom_sq = 0.0L;
        for (size_t i = 0; i < size; ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            double abs_diff = std::fabs(diff);
            sum_abs += abs_diff;
            sum_sq += diff * diff;
            double base = static_cast<double>(b[i]);
            denom_sq += base * base;
            if (abs_diff > max_abs)
            {
                max_abs = abs_diff;
                worst = i;
                worst_a = a[i];
                worst_b = b[i];
            }
        }
        summary.max_abs = max_abs;
        summary.mean_abs = sum_abs / static_cast<double>(size);
        summary.rel_l2 = std::sqrt(static_cast<double>(sum_sq)) / (std::sqrt(static_cast<double>(denom_sq)) + 1e-30);
        summary.worst_index = worst;
        summary.value_a = worst_a;
        summary.value_b = worst_b;
        return summary;
    }
}

// === Section 1: Factory registration, statics, constructor, kernel registration, FFN shard diagnostics ===
namespace llaminar
{
    // Forward declaration of internal bridge (defined later in this TU)
    DistributedTransformerPipeline::ModelWeights loadModelWeights_impl_bridge(
        ModelLoader &loader,
        const DistributedTransformerPipeline::LayerConfig &config);
    // Factory helper implementation (migrated from legacy)
    std::unique_ptr<DistributedTransformerPipeline> createDistributedTransformerPipeline(const DistributedTransformerPipeline::LayerConfig &config)
    {
        return std::make_unique<DistributedTransformerPipeline>(config);
    }

    // Helper from legacy TU: FFN row preview tied to shard tracing env (simplified migration of utility)
    bool isFFNShardTracingEnabledFor(const std::string &label); // forward decl
    static void logFFNRowPreviewIfEnabled(const std::string &label,
                                          const std::shared_ptr<TensorBase> &tensor,
                                          size_t default_preview_cols = 8)
    {
        if (!tensor)
            return;
        const auto &cfg = debugEnv().ffn_shard_trace;
        if (!cfg.enabled || !isFFNShardTracingEnabledFor(label))
            return;
        const auto &shape = tensor->shape();
        if (shape.size() < 2)
            return;
        int total_rows = shape[0], total_cols = shape[1];
        if (total_rows <= 0 || total_cols <= 0)
            return;
        std::vector<int> rows;
        if (!cfg.rows_spec.empty())
        {
            std::stringstream ss(cfg.rows_spec);
            std::string tok;
            while (std::getline(ss, tok, ','))
            {
                if (tok.empty())
                    continue;
                try
                {
                    int v = std::stoi(tok);
                    if (v >= 0 && v < total_rows)
                        rows.push_back(v);
                }
                catch (...)
                {
                }
            }
        }
        else
        {
            rows.push_back(0);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        if (rows.empty())
            return;
        size_t preview_cols = default_preview_cols;
        if (!cfg.cols.empty())
        {
            preview_cols = std::min<size_t>(cfg.cols.size(), (size_t)total_cols);
        }
        preview_cols = std::max<size_t>(1, std::min(preview_cols, (size_t)total_cols));
        logTensorRowPreview(tensor, label, rows, preview_cols, "PrefillFFNPreview");
    }

    bool DistributedTransformerPipeline::executeTransformerLayer(int layer_idx,
                                                                 std::shared_ptr<TensorBase> &input,
                                                                 const ModelWeights &weights,
                                                                 std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("DistributedTransformerPipeline::executeTransformerLayer");
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
            DistributedTransformerPipeline *pipeline = nullptr;
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
        auto attn_norm_out = createLocalTensor({seq_len, config_.d_model});
        auto attn_out = createLocalTensor({seq_len, config_.d_model});
        auto ffn_norm_out = createLocalTensor({seq_len, config_.d_model});
        auto ffn_out = createLocalTensor({seq_len, config_.d_model});
        auto residual_tmp = createLocalTensor({seq_len, config_.d_model});

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
            bctx.d_model = config_.d_model;
            bctx.n_layers = config_.n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                PrefillAttentionTiming timing;
                if (!executePrefillAttentionCosma(layer_idx, input, weights, attn_norm_out, attn_out, timing))
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
                std::vector<std::shared_ptr<TensorBase>> attn_inputs = {attn_norm_out, weights.wq[layer_idx], weights.wk[layer_idx], weights.wv[layer_idx], weights.wo[layer_idx], use_kv_cache_ ? k_cache_[layer_idx] : createLocalTensor({seq_len, config_.n_head_kv * config_.head_dim}), use_kv_cache_ ? v_cache_[layer_idx] : createLocalTensor({seq_len, config_.n_head_kv * config_.head_dim})};
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
        }
        else
        {
            std::memcpy(residual_tmp->data(), input->data(), sizeof(float) * (size_t)seq_len * config_.d_model);
            std::memcpy(attn_out->data(), input->data(), sizeof(float) * (size_t)seq_len * config_.d_model);
            std::memset(attn_norm_out->data(), 0, sizeof(float) * (size_t)seq_len * config_.d_model);
            if (getRank() == 0)
                LOG_WARN("[Ablation] Layer " << layer_idx << " attention skipped");
            // Even in ablation, capture a synthetic attention path state for consistency
            capture_stage(attn_out, "attn_out");
            capture_stage(residual_tmp, "attn_residual");
        }
        // FFN
        if (abl.ablate_ffn)
        {
            std::memcpy(output->data(), residual_tmp->data(), sizeof(float) * (size_t)seq_len * config_.d_model);
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
        auto gate_out = createLocalTensor({seq_len, config_.d_ff});
        auto up_out = createLocalTensor({seq_len, config_.d_ff});
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.d_model;
            bctx.n_layers = config_.n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                if (!adaptiveMatMul(ffn_norm_out->data(), weights.w_gate[layer_idx]->data(), gate_out->data(), seq_len, config_.d_ff, config_.d_model, is_prefill_stage_, false, false, false, 1.0f, 0.0f))
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
            bctx.d_model = config_.d_model;
            bctx.n_layers = config_.n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                if (!adaptiveMatMul(ffn_norm_out->data(), weights.w_up[layer_idx]->data(), up_out->data(), seq_len, config_.d_ff, config_.d_model, is_prefill_stage_, false, false, false, 1.0f, 0.0f))
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
        auto swiglu_out = createLocalTensor({seq_len, config_.d_ff});
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
            bctx.d_model = config_.d_model;
            bctx.n_layers = config_.n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                if (!adaptiveMatMul(swiglu_out->data(), weights.w_down[layer_idx]->data(), ffn_out->data(), seq_len, config_.d_model, config_.d_ff, is_prefill_stage_, false, false, false, 1.0f, 0.0f))
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
        std::vector<std::shared_ptr<TensorBase>> final_residual_inputs = {residual_tmp, ffn_out};
        std::vector<std::shared_ptr<TensorBase>> final_residual_outputs = {output};
        if (!executeKernel("residual", final_residual_inputs, final_residual_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " final residual failed");
            return false;
        }
        // Diagnostics: capture last token row if enabled
        capture_stage(output, "layer_output");
        return true;
    }

    // Public weight loading wrappers calling internal bridge
    DistributedTransformerPipeline::ModelWeights loadModelWeights(ModelLoader &loader, const DistributedTransformerPipeline::LayerConfig &config)
    {
        return loadModelWeights_impl_bridge(loader, config);
    }

    DistributedTransformerPipeline::ModelWeights loadModelWeights(const std::string &model_path, const DistributedTransformerPipeline::LayerConfig &config)
    {
        ModelLoader loader;
        if (!loader.loadModel(model_path))
        {
            throw std::runtime_error("ModelLoader loadModel failed: " + model_path);
        }
        return loadModelWeights_impl_bridge(loader, config);
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
    bool isFFNShardTracingEnabledFor(const std::string &label)
    {
        const auto &cfg = llaminar::debugEnv().ffn_shard_trace;
        if (!cfg.enabled)
            return false;
        if (cfg.match_all)
            return true;
        if (cfg.shards_spec.empty())
            return true;
        std::stringstream ss(cfg.shards_spec);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            // lightweight trim
            size_t b = tok.find_first_not_of(" \t\n\r");
            if (b == std::string::npos)
                continue;
            size_t e = tok.find_last_not_of(" \t\n\r");
            std::string t = tok.substr(b, e - b + 1);
            if (t == label)
                return true;
        }
        return false;
    }

    class PrefillBaselineRegistry
    {
    public:
        static PrefillBaselineRegistry &instance()
        {
            static PrefillBaselineRegistry inst;
            return inst;
        }
        void clear()
        {
            std::lock_guard<std::mutex> g(mutex_);
            storage_.clear();
        }
        bool ensure(const std::string &name, const float *data, size_t count)
        {
            if (!data || count == 0)
                return false;
            std::lock_guard<std::mutex> g(mutex_);
            auto it = storage_.find(name);
            if (it == storage_.end() || it->second.size() != count)
            {
                storage_[name] = std::vector<float>(data, data + count);
                return true;
            }
            return false;
        }
        bool fetch(const std::string &name, std::vector<float> &out) const
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = storage_.find(name);
            if (it == storage_.end())
                return false;
            out = it->second;
            return true;
        }

    private:
        PrefillBaselineRegistry() = default;
        PrefillBaselineRegistry(const PrefillBaselineRegistry &) = delete;
        PrefillBaselineRegistry &operator=(const PrefillBaselineRegistry &) = delete;
        mutable std::mutex mutex_;
        std::unordered_map<std::string, std::vector<float>> storage_;
    };
    static std::once_flag qwen_register_flag;
    void registerQwenPipeline()
    {
        std::call_once(qwen_register_flag, []()
                       { PipelineFactory::instance().registerCreator("qwen", [](const TransformerLayerConfig &cfg) -> std::unique_ptr<AbstractPipeline>
                                                                     {
				auto impl = std::make_unique<DistributedTransformerPipeline>(cfg);
				return std::unique_ptr<AbstractPipeline>(impl.release()); }); });
    }

    std::atomic<size_t> DistributedTransformerPipeline::small_seq_fast_path_calls_{0};
    std::vector<float> DistributedTransformerPipeline::last_pre_lm_hidden_;
    std::vector<DistributedTransformerPipeline::LayerActivationStat> DistributedTransformerPipeline::last_layer_stats_;
    std::vector<DistributedTransformerPipeline::LayerTokenDiffRow> DistributedTransformerPipeline::last_layer_token_rows_;

    DistributedTransformerPipeline::DistributedTransformerPipeline(const LayerConfig &config)
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
        if (use_kv_cache_)
        {
            int initial_capacity = config_.max_seq_len;
            if (kv_cache_dynamic_init_)
                initial_capacity = 1; // defer sizing
            initializeKVCache(initial_capacity);
            if (getRank() == 0)
            {
                LOG_INFO("[KVCacheInitMode] post-initial-capacity this=" << (const void *)this << " capacity_tokens=" << kv_cache_state_.capacity_tokens);
            }
        }
        LOG_INFO("DistributedTransformerPipeline initialized on rank " << getRank() << "/" << getSize() << " with " << config_.n_layers << " layers, " << config_.n_head << " heads");
    }

    DistributedTransformerPipeline::~DistributedTransformerPipeline() = default;

    void DistributedTransformerPipeline::initializeKernels()
    {
        {
            auto embedding_kernel = std::make_unique<MPIEmbeddingKernel>(config_.vocab_size, config_.d_model);
            if (!registerKernel("embedding", std::move(embedding_kernel)))
                throw std::runtime_error("Failed to register Embedding kernel");
        }
        auto rmsnorm_kernel = std::make_unique<MPIRMSNormKernel>(MPIRMSNormKernel::DistributionStrategy::SEQUENCE_WISE);
        rmsnorm_kernel->setEpsilon(config_.eps);
        if (!registerKernel("rmsnorm", std::move(rmsnorm_kernel)))
            throw std::runtime_error("Failed to register RMSNorm kernel");

        auto attention_kernel = std::make_unique<MPIAttentionKernel>(config_.n_head, config_.n_head_kv, config_.head_dim, config_.rope_freq_base);
        if (!registerKernel("attention", std::move(attention_kernel)))
            throw std::runtime_error("Failed to register Attention kernel");

        auto linear_kernel = std::make_unique<MPILinearKernel>();
        if (!registerKernel("linear", std::move(linear_kernel)))
            throw std::runtime_error("Failed to register Linear kernel");

        auto swiglu_kernel = std::make_unique<MPISwiGLUKernel>(MPISwiGLUKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("swiglu", std::move(swiglu_kernel)))
            throw std::runtime_error("Failed to register SwiGLU kernel");

        auto rope_kernel = std::make_unique<MPIRoPEKernel>(config_.max_seq_len, config_.head_dim, config_.rope_freq_base, MPIRoPEKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("rope", std::move(rope_kernel)))
            throw std::runtime_error("Failed to register RoPE kernel");

        auto residual_kernel = std::make_unique<MPIResidualKernel>(MPIResidualKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("residual", std::move(residual_kernel)))
            throw std::runtime_error("Failed to register Residual kernel");

        LOG_DEBUG("DistributedTransformerPipeline: Registered " << getKernelNames().size() << " kernels on rank " << getRank());
    }

    void DistributedTransformerPipeline::traceFFNShardDiagnostics(const std::string &label, const float *data, int seq_len, int feature_dim)
    {
        const auto &cfg = debugEnv().ffn_shard_trace;
        if (!cfg.enabled || !data || seq_len <= 0 || feature_dim <= 0)
            return;
        if (!isFFNShardTracingEnabledFor(label))
            return;
        const int sample_limit = cfg.limit > 0 ? cfg.limit : 1;
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const size_t total_elements = static_cast<size_t>(seq_len) * static_cast<size_t>(feature_dim);
        auto stats = computeBufferStats(data, total_elements);
        const bool baseline_enabled = llaminar::debugEnv().baseline.compare;
        std::vector<float> baseline_buffer;
        const float *baseline_ptr = nullptr;
        BufferStats baseline_stats{};
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
        std::ostringstream header;
        header << "[PrefillFFNTrace] rank=" << rank << " shard=" << label
               << " shape=(" << seq_len << "," << feature_dim << ")"
               << " min=" << stats.min << " max=" << stats.max
               << " mean=" << stats.mean << " rms=" << stats.rms
               << " stddev=" << stats.stddev;
        if (baseline_ptr)
        {
            auto diff = computeDiffSummary(data, baseline_ptr, total_elements);
            header << " rel_l2=" << diff.rel_l2 << " mean_abs=" << diff.mean_abs << " max_abs=" << diff.max_abs;
        }
        if (rank == 0)
            LOG_INFO(header.str());
        if (cfg.limit > 0 && rank == 0)
        { // using limit as proxy for sample request (legacy field sample_prefix not yet migrated)
            int rows = std::min(seq_len, sample_limit);
            int cols = std::min(feature_dim, sample_limit);
            for (int r = 0; r < rows; ++r)
            {
                std::ostringstream row_ss;
                row_ss << "[PrefillFFNSample] shard=" << label << " row=" << r << " vals=";
                const float *row_ptr = data + r * feature_dim;
                for (int c = 0; c < cols; ++c)
                    row_ss << row_ptr[c] << (c + 1 < cols ? ',' : '\0');
                LOG_INFO(row_ss.str());
            }
        }
    }
} // namespace llaminar

// Out-of-line full weight loader implementation.
namespace llaminar
{
    DistributedTransformerPipeline::ModelWeights loadModelWeights_impl_bridge(
        ModelLoader &loader,
        const DistributedTransformerPipeline::LayerConfig &config)
    {
        DistributedTransformerPipeline::ModelWeights weights;
        LOG_INFO("[WeightLoad] begin vocab=" << config.vocab_size << " d_model=" << config.d_model << " layers=" << config.n_layers);

        // Token embedding
        weights.token_embedding = loader.loadTensor("token_embd.weight");
        // (Removed stray incremental decode guard accidentally injected during patch)
        const auto emb_shape = weights.token_embedding->shape();
        if (emb_shape.size() == 2)
        {
            LOG_INFO("[WeightLoad] token_embd.weight shape=" << emb_shape[0] << "x" << emb_shape[1]);
        }
        // Sample stats (lightweight)
        {
            const float *ptr = weights.token_embedding->data();
            size_t total = (size_t)weights.token_embedding->size();
            size_t sample = std::min<size_t>(total, (size_t)512);
            size_t nan_ct = 0, inf_ct = 0;
            for (size_t i = 0; i < sample; ++i)
            {
                float v = ptr[i];
                if (std::isnan(v))
                    ++nan_ct;
                if (std::isinf(v))
                    ++inf_ct;
            }
            if (nan_ct || inf_ct)
                LOG_WARN("[WeightLoad] token_embd anomalies nan=" << nan_ct << " inf=" << inf_ct);
        }

        // Output norm
        weights.output_norm_weight = loader.loadTensor("output_norm.weight");
        if (!weights.output_norm_weight)
            throw std::runtime_error("Failed to load output_norm.weight");
        if (weights.output_norm_weight->size() != config.d_model)
        {
            LOG_WARN("[WeightLoad] output_norm.weight size=" << weights.output_norm_weight->size() << " != d_model=" << config.d_model);
        }
        if (debugEnv().output_norm.force_unit || debugEnv().output_norm.force_unit_all)
        {
            float *g = const_cast<float *>(weights.output_norm_weight->data());
            for (int i = 0; i < weights.output_norm_weight->size(); ++i)
                g[i] = 1.0f;
            LOG_WARN("[WeightLoad] Forced output_norm to unit gamma");
        }
        else if (debugEnv().output_norm.clamp)
        {
            float *g = const_cast<float *>(weights.output_norm_weight->data());
            for (int i = 0; i < weights.output_norm_weight->size(); ++i)
                g[i] = std::clamp(g[i], 0.0f, 4.0f);
            LOG_WARN("[WeightLoad] Clamped output_norm gamma range [0,4]");
        }

        // LM head (optional)
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
            try
            {
                bool raw = debugEnv().lm_head.raw_orientation;
                if (!raw)
                    enforce_matrix_layout_compat(lm_head, config.d_model, config.vocab_size, "output.weight");
                weights.lm_head = lm_head;
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[WeightLoad] lm_head orientation issue: " << e.what() << " using tied embedding");
                weights.lm_head = weights.token_embedding;
            }
        }
        else
        {
            LOG_WARN("[WeightLoad] output.weight missing; using tied embeddings as LM head");
            weights.lm_head = weights.token_embedding;
        }

        // Reserve vectors
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
            std::string prefix = "blk." + std::to_string(layer) + ".";
            auto attn_norm = loader.loadTensor(prefix + "attn_norm.weight");
            if (!attn_norm)
                throw std::runtime_error("Failed to load " + prefix + "attn_norm.weight");
            if (debugEnv().output_norm.force_unit_all)
            {
                float *g = const_cast<float *>(attn_norm->data());
                for (int i = 0; i < attn_norm->size(); ++i)
                    g[i] = 1.0f;
            }
            weights.attn_norm_weight.push_back(attn_norm);

            auto wq = loader.loadTensor(prefix + "attn_q.weight");
            auto wk = loader.loadTensor(prefix + "attn_k.weight");
            auto wv = loader.loadTensor(prefix + "attn_v.weight");
            auto wo = loader.loadTensor(prefix + "attn_output.weight");
            if (!wq || !wk || !wv || !wo)
                throw std::runtime_error("Failed to load attention weights for layer " + std::to_string(layer));
            weights.wq.push_back(wq);
            weights.wk.push_back(wk);
            weights.wv.push_back(wv);
            weights.wo.push_back(wo);

            auto ffn_norm = loader.loadTensor(prefix + "ffn_norm.weight");
            if (!ffn_norm)
                throw std::runtime_error("Failed to load " + prefix + "ffn_norm.weight");
            if (debugEnv().output_norm.force_unit_all)
            {
                float *g = const_cast<float *>(ffn_norm->data());
                for (int i = 0; i < ffn_norm->size(); ++i)
                    g[i] = 1.0f;
            }
            weights.ffn_norm_weight.push_back(ffn_norm);

            auto w_gate = loader.loadTensor(prefix + "ffn_gate.weight");
            auto w_up = loader.loadTensor(prefix + "ffn_up.weight");
            auto w_down = loader.loadTensor(prefix + "ffn_down.weight");
            if (!w_gate || !w_up || !w_down)
                throw std::runtime_error("Failed to load FFN weights for layer " + std::to_string(layer));
            weights.w_gate.push_back(w_gate);
            weights.w_up.push_back(w_up);
            weights.w_down.push_back(w_down);
        }
        LOG_INFO("[WeightLoad] per-layer loading complete");
        return weights;

        // (Helper forward declarations below weight loader section)
    }

    // Forward declare helper again (implemented below this section)
    void handle_prefill_stage_snapshot(int rank, const std::string &name, const float *data, size_t count,
                                       int cols, double warn_threshold, bool capture_enabled, bool compare_enabled);

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

    bool DistributedTransformerPipeline::executeEmbedding(const std::vector<int> &token_ids,
                                                          const std::shared_ptr<TensorBase> &embedding_weight,
                                                          std::shared_ptr<TensorBase> &embedded_output)
    {
        PERF_SCOPED_TIMER("DistributedTransformerPipeline::executeEmbedding");
        int seq_len = (int)token_ids.size();
        if (seq_len <= 0)
        {
            LOG_ERROR("executeEmbedding: empty token_ids");
            return false;
        }
        if (!embedded_output || embedded_output->shape().size() != 2 || embedded_output->shape()[0] != seq_len || embedded_output->shape()[1] != config_.d_model)
        {
            embedded_output = createLocalTensor({seq_len, config_.d_model});
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
            handle_prefill_stage_snapshot(getRank(), "embedding_output", embedded_output->data(), (size_t)embedded_output->size(), config_.d_model, 5e-4, capture_baseline, compare_baseline);
        }
        return true;
    }

    // Simple intermediate tensor factory
    std::vector<std::shared_ptr<TensorBase>> DistributedTransformerPipeline::createIntermediateTensors(int seq_len)
    {
        return {createLocalTensor({seq_len, config_.d_model}), createLocalTensor({seq_len, config_.d_model})};
    }

    bool DistributedTransformerPipeline::executeOutputProjection(std::shared_ptr<TensorBase> &input,
                                                                 const ModelWeights &weights,
                                                                 std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("DistributedTransformerPipeline::executeOutputProjection");
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
        auto normed = createLocalTensor({seq_len, config_.d_model});
        std::vector<std::shared_ptr<TensorBase>> norm_inputs = {input, weights.output_norm_weight};
        std::vector<std::shared_ptr<TensorBase>> norm_outputs = {normed};
        if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
        {
            LOG_ERROR("Output projection norm failed");
            return false;
        }
        // Keep last hidden (rank0) for diagnostics
        if (getRank() == 0)
        {
            size_t elems = (size_t)seq_len * (size_t)config_.d_model;
            last_pre_lm_hidden_.assign(normed->data(), normed->data() + elems);
        }
        // Allocate output logits
        if (!output || output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != config_.vocab_size)
        {
            output = createLocalTensor({seq_len, config_.vocab_size});
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
        // Optional baseline snapshot of final logits (rank 0 only)
        const bool capture_baseline = debugEnv().baseline.capture;
        const bool compare_baseline = debugEnv().baseline.compare;
        if ((capture_baseline || compare_baseline) && getRank() == 0)
        {
            handle_prefill_stage_snapshot(0, "final_logits", output->data(), (size_t)output->size(), config_.vocab_size, 5e-4, capture_baseline, compare_baseline);
        }
        return true;
    }

    bool DistributedTransformerPipeline::validate(const ModelWeights &w) const
    {
        auto check_vec = [&](const std::vector<std::shared_ptr<TensorBase>> &v, const char *name)
        { if((int)v.size()!=config_.n_layers){ LOG_ERROR("Weights vector size mismatch for "<<name); return false;} for(size_t i=0;i<v.size();++i) if(!v[i]) { LOG_ERROR("Null weight in "<<name<<" index="<<i); return false; } return true; };
        if (!w.token_embedding || !w.output_norm_weight || !w.lm_head)
        {
            LOG_ERROR("Missing core weights (embedding/output_norm/lm_head)");
            return false;
        }
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
    void DistributedTransformerPipeline::initializeKVCache(int seq_len)
    {
        if (!use_kv_cache_)
            return;
        if (seq_len <= 0)
            seq_len = 1;
        k_cache_.resize(config_.n_layers);
        v_cache_.resize(config_.n_layers);
        for (int l = 0; l < config_.n_layers; ++l)
        {
            if (!k_cache_[l] || k_cache_[l]->shape()[0] < seq_len)
                k_cache_[l] = createLocalTensor({seq_len, config_.n_head_kv * config_.head_dim});
            if (!v_cache_[l] || v_cache_[l]->shape()[0] < seq_len)
                v_cache_[l] = createLocalTensor({seq_len, config_.n_head_kv * config_.head_dim});
        }
        kv_cache_state_.capacity_tokens = seq_len;
        kv_cache_state_.used_tokens = 0;
        kv_cache_state_.growth_events = 0;
    }

    bool DistributedTransformerPipeline::ensureKVCapacityInternal(int required_tokens)
    {
        if (!use_kv_cache_)
            return true;
        if (required_tokens <= kv_cache_state_.capacity_tokens)
            return true;
        int new_cap = std::max(required_tokens, kv_cache_state_.capacity_tokens * 2);
        if (new_cap > config_.max_seq_len)
            new_cap = config_.max_seq_len;
        initializeKVCache(new_cap);
        kv_cache_state_.growth_events++;
        if (getRank() == 0)
            LOG_INFO("[KVCacheGrow] new_capacity=" << new_cap);
        return required_tokens <= kv_cache_state_.capacity_tokens;
    }

    bool DistributedTransformerPipeline::ensureKVCapacity(int required_tokens) { return ensureKVCapacityInternal(required_tokens); }

    bool DistributedTransformerPipeline::execute(const std::vector<int> &token_ids,
                                                 const ModelWeights &weights,
                                                 std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("DistributedTransformerPipeline::execute");
        start_time_ = std::chrono::high_resolution_clock::now();
        if (!validate(weights))
        {
            LOG_ERROR("DistributedTransformerPipeline: Weight validation failed");
            return false;
        }
        int seq_len = (int)token_ids.size();
        if (seq_len <= 0 || seq_len > config_.max_seq_len)
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
            small_seq_fast_path_calls_.fetch_add(1, std::memory_order_relaxed);
            if (getRank() == 0)
            {
                if (force_full_layer_capture)
                {
                    LOG_INFO("[LayerTokenDiffDiag] bypass small_seq_fast_path due to replay_compare instrumentation");
                }
                if (!output || output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != config_.vocab_size)
                    output = createLocalTensor({seq_len, config_.vocab_size});
                // Naive local forward (embedding + simplified transformer) from legacy path (trimmed diagnostics)
                auto embed_data = weights.token_embedding->data();
                std::vector<float> hidden(seq_len * config_.d_model, 0.f), tmp(seq_len * config_.d_model, 0.f);
                auto rmsnorm = [&](std::vector<float> &mat, const float *wn)
                { kernels::RMSNormExecOptions opts; kernels::rmsnorm_row_major_fused(mat.data(), wn, mat.data(), (size_t)seq_len, (size_t)config_.d_model, config_.eps, kernels::GammaMode::REPLICATED, 0, opts); };
                auto matmul = [&](const std::vector<float> &A, const float *B, int k, int n, std::vector<float> &C)
                { C.assign(seq_len*n,0.f); for(int i=0;i<seq_len;++i){ const float *a=&A[i*k]; float *crow=&C[i*n]; for(int kk=0;kk<k;++kk){ float aval=a[kk]; const float *bcol=&B[kk*n]; for(int j=0;j<n;++j) crow[j]+=aval*bcol[j]; } } };
                auto elementwise_add = [&](std::vector<float> &A, const std::vector<float> &B)
                { for(size_t i=0;i<A.size();++i) A[i]+=B[i]; };
                auto sigmoid = [](float x)
                { return 1.f / (1.f + std::exp(-x)); };
                auto swiglu = [&](const std::vector<float> &up, const std::vector<float> &gate, std::vector<float> &out)
                { out.resize(up.size()); for(size_t i=0;i<up.size();++i) out[i]=up[i]*sigmoid(gate[i]); };
                for (int t = 0; t < seq_len; ++t)
                {
                    int tok = token_ids[t];
                    std::memcpy(&hidden[t * config_.d_model], &embed_data[tok * config_.d_model], sizeof(float) * config_.d_model);
                }
                for (int layer = 0; layer < config_.n_layers; ++layer)
                {
                    rmsnorm(hidden, weights.attn_norm_weight[layer]->data());
                    std::vector<float> Q, V, context;
                    matmul(hidden, weights.wq[layer]->data(), config_.d_model, config_.n_head * config_.head_dim, tmp);
                    Q = tmp;
                    matmul(hidden, weights.wv[layer]->data(), config_.d_model, config_.n_head_kv * config_.head_dim, tmp);
                    V = tmp;
                    context = Q;
                    int ctx_dim = config_.n_head * config_.head_dim;
                    std::vector<float> v_mean(ctx_dim, 0.f);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        const float *vrow = &V[i * ctx_dim];
                        for (int j = 0; j < ctx_dim; ++j)
                            v_mean[j] += vrow[j];
                    }
                    for (int j = 0; j < ctx_dim; ++j)
                        v_mean[j] /= std::max(1, seq_len);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        float *crow = &context[i * ctx_dim];
                        for (int j = 0; j < ctx_dim; ++j)
                            crow[j] = 0.5f * (crow[j] + v_mean[j]);
                    }
                    matmul(context, weights.wo[layer]->data(), ctx_dim, config_.d_model, tmp);
                    elementwise_add(tmp, hidden);
                    hidden = tmp;
                    rmsnorm(hidden, weights.ffn_norm_weight[layer]->data());
                    std::vector<float> gate, up, swiglu_out;
                    matmul(hidden, weights.w_gate[layer]->data(), config_.d_model, config_.d_ff, gate);
                    matmul(hidden, weights.w_up[layer]->data(), config_.d_model, config_.d_ff, up);
                    swiglu(up, gate, swiglu_out);
                    matmul(swiglu_out, weights.w_down[layer]->data(), config_.d_ff, config_.d_model, tmp);
                    elementwise_add(tmp, hidden);
                    hidden = tmp;
                }
                rmsnorm(hidden, weights.output_norm_weight->data());
                std::vector<float> logits;
                matmul(hidden, weights.lm_head->data(), config_.d_model, config_.vocab_size, logits);
                std::memcpy(const_cast<float *>(output->data()), logits.data(), sizeof(float) * logits.size());
                if (compare_baseline)
                {
                    PrefillBaselineRegistry::instance().clear();
                }
                if (use_kv_cache_)
                {
                    kv_cache_state_.used_tokens = seq_len;
                    n_past_ = seq_len;
                }
            }
            // Broadcast output to all ranks
            if (getRank() != 0)
            {
                if (!output || output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != config_.vocab_size)
                    output = createLocalTensor({seq_len, config_.vocab_size});
            }
            checkMPIError(MPI_Bcast(const_cast<float *>(output->data()), seq_len * config_.vocab_size, MPI_FLOAT, 0, getComm()), "MPI_Bcast small-seq output");
            return true;
        }

        // Standard distributed path
        auto tensors = createIntermediateTensors(seq_len);
        auto current_input = tensors[0];
        auto layer_output = tensors[1];
        if (!executeEmbedding(token_ids, weights.token_embedding, current_input))
            return false;
        for (int layer = 0; layer < config_.n_layers; ++layer)
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
    bool DistributedTransformerPipeline::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                                 std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        LOG_ERROR("DistributedTransformerPipeline::execute(vector) not supported; use execute(token_ids, weights, output) overload");
        return false;
    }

    // Minimal logits() implementation – returns cached last logits tensor
    bool DistributedTransformerPipeline::logits(std::shared_ptr<TensorBase> &out_logits)
    {
        if (!last_logits_)
        {
            LOG_WARN("logits() requested but no cached logits available");
            return false;
        }
        out_logits = last_logits_;
        return true;
    }

    const KVCacheState *DistributedTransformerPipeline::kvCacheState() const
    {
        kv_snapshot_.capacity_tokens = kv_cache_state_.capacity_tokens;
        kv_snapshot_.used_tokens = kv_cache_state_.used_tokens;
        kv_snapshot_.growth_events = kv_cache_state_.growth_events;
        return &kv_snapshot_;
    }

    // Baseline snapshot handler (minimal implementation)
    void handle_prefill_stage_snapshot(int rank, const std::string &name, const float *data, size_t count,
                                       int cols, double warn_threshold, bool capture_enabled, bool compare_enabled)
    {
        if (rank != 0 || !data || count == 0)
            return;
        auto &reg = PrefillBaselineRegistry::instance();
        if (capture_enabled)
        {
            reg.ensure(name, data, count);
            LOG_TRACE("[PrefillBaseline] captured " << name << " count=" << count);
        }
        if (compare_enabled)
        {
            std::vector<float> baseline;
            if (!reg.fetch(name, baseline) || baseline.size() != count)
            {
                LOG_WARN("[PrefillBaseline] missing baseline for compare name=" << name);
                return;
            }
            auto diff = computeDiffSummary(data, baseline.data(), count);
            if (diff.rel_l2 > warn_threshold)
            {
                LOG_WARN("[PrefillBaseline] rel_l2=" << diff.rel_l2 << " max_abs=" << diff.max_abs << " mean_abs=" << diff.mean_abs << " name=" << name);
            }
            else
            {
                LOG_DEBUG("[PrefillBaseline] OK name=" << name << " rel_l2=" << diff.rel_l2);
            }
        }
    }
} // namespace llaminar

// === Section 4: AbstractPipeline interface (prefill/decode/logits) partial migration ===
namespace llaminar
{
    bool DistributedTransformerPipeline::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                                  const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.empty() || outputs.empty())
            return false;
        return true;
    }

    bool DistributedTransformerPipeline::prefill(const std::vector<int> &tokens,
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
        auto output = TensorFactory::create_simple({(int)tokens.size(), config_.vocab_size});
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

    bool DistributedTransformerPipeline::decode(int next_token,
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
            auto replay = TensorFactory::create_simple({(int)current_tokens_.size(), config_.vocab_size});
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

    bool DistributedTransformerPipeline::incrementalDecodeToken(int token_id,
                                                                const ModelWeights &weights,
                                                                std::shared_ptr<TensorBase> &output_logits)
    {
        // Incremental decode ported from legacy pipeline (single-token fast path)
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
        // Diagnostic override: ensure attention/ffn not ablated when doing layer replay compare, so we can capture full stage set.
        bool diag_force_full_layers = env.pipeline.layer_token_diff && env.pipeline.layer_replay_compare;
        bool saved_ablate_attention = false, saved_ablate_ffn = false;
        if (diag_force_full_layers)
        {
            // We cannot mutate global snapshot; instead we branch locally by guarding ablation checks later.
            // We'll signal via a local flag used below where ablation is checked.
        }
        if (!use_kv_cache_)
        {
            LOG_INFO("[IncrEarlyReturn] rank=" << getRank() << " reason=kv_cache_disabled");
            return false;
        }
        if (k_cache_.empty() || v_cache_.empty() || (int)k_cache_.size() != config_.n_layers || (int)v_cache_.size() != config_.n_layers)
        {
            LOG_INFO("[IncrEarlyReturn] rank=" << getRank() << " reason=kv_cache_uninit_or_size_mismatch k_size=" << k_cache_.size() << " v_size=" << v_cache_.size() << " expected_layers=" << config_.n_layers);
            return false;
        }
        if (!weights.token_embedding)
        {
            LOG_ERROR("incrementalDecodeToken: missing token embedding");
            return false;
        }
        // Ensure capacity for new token write (position = n_past_)
        if (!ensureKVCapacity(n_past_ + 1))
        {
            LOG_WARN("[IncrEarlyReturn] rank=" << getRank() << " reason=ensureKVCapacity_failed requested=" << (n_past_ + 1));
            return false;
        }
        // Embed single token (1 x d_model)
        auto current = embedSingleToken(token_id, weights.token_embedding);
        if (!current)
            return false;
        // Record starting offset for layer token diff rows (so we can isolate incremental rows for this token)
        size_t layer_rows_offset_before = last_layer_token_rows_.size();
        if (env.pipeline.incr_trace && getRank() == 0)
        {
            LOG_INFO("[IncrTrace] start token=" << token_id << " n_past=" << n_past_ << " use_kv=1");
        }
        setStageDecode();
        int position = n_past_;
        setSequencePosition(position);
        const int seq_len = 1;
        auto layer_output = createLocalTensor({seq_len, config_.d_model});
        if (!layer_output)
        {
            LOG_ERROR("incrementalDecodeToken: failed layer_output alloc");
            return false;
        }
        auto logits = createLocalTensor({seq_len, config_.vocab_size});
        if (!logits)
        {
            LOG_ERROR("incrementalDecodeToken: failed logits alloc");
            return false;
        }
        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            // Force per-layer attention kernel sequence position and expected window (prefill len + decoded count)
            if (auto attn = dynamic_cast<MPIAttentionKernel *>(getKernel("attention")))
            {
                attn->setSequencePosition(n_past_);
                attn->setLayerIndex(layer);
                // expected window = current committed tokens (n_past_) + 1 (this token)
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
                // Minimal KV preview (first 4 floats of first head) assuming contiguous per-token layout
                if (layer < (int)k_cache_.size() && k_cache_[layer])
                {
                    int head_dim = config_.head_dim;
                    size_t base = (size_t)position * head_dim;
                    float k_prev[4] = {0}, v_prev[4] = {0};
                    auto &K = k_cache_[layer];
                    auto &V = v_cache_[layer];
                    for (int i = 0; i < 4 && i < head_dim; ++i)
                    {
                        k_prev[i] = K->data()[base + i];
                        v_prev[i] = V->data()[base + i];
                    }
                    LOG_INFO("[IncrCacheTrace] layer=" << layer << " pos=" << position << " k0=[" << k_prev[0] << "," << k_prev[1] << "," << k_prev[2] << "," << k_prev[3] << "] v0=[" << v_prev[0] << "," << v_prev[1] << "," << v_prev[2] << "," << v_prev[3] << "]");
                }
            }
        }
        if (!executeOutputProjection(current, weights, logits))
        {
            LOG_ERROR("incrementalDecodeToken: output projection failed");
            return false;
        }
        if (env.pipeline.incr_hidden_trace && getRank() == 0)
        {
            int d = config_.d_model;
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

        // Optional replay compare: run full replay for sequence (previous tokens + new token) and diff per-layer last-row.
        // Isolation model: copy incremental rows, clear global vector, run replay to capture only replay rows,
        // then restore incremental rows before diffing.
        const auto &penv = debugEnv().pipeline;
        if (penv.layer_token_diff && penv.layer_replay_compare)
        {
            // Symmetry enforcement: all ranks must be here; verify via Allreduce.
            int local_flag = 1;
            int global_sum = 0;
            MPI_Allreduce(&local_flag, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            int world_size = getSize();
            if (global_sum != world_size)
            {
                if (getRank() == 0)
                    LOG_WARN("[LayerReplayIsoSkip] not all ranks in incremental path (global_sum=" << global_sum << " world=" << world_size << ") skipping isolation token_pos=" << n_past_);
            }
            else
            {
                if (getRank() == 0)
                {
                    LOG_INFO("[LayerReplayIsoEnter] triggering isolation replay compare token_pos=" << n_past_ << " rows_size=" << last_layer_token_rows_.size());
                }
                try
                {
                    // Full prefix reconstruction (Option A): rebuild all prior tokens plus this new token.
                    // At this point n_past_ was just incremented above (now equals total committed tokens).
                    // current_tokens_ has NOT yet been updated with token_id (push_back happens later only if incremental path succeeded).
                    // So we build a replay sequence consisting of: prior committed tokens (n_past_-1) + this new token.
                    std::vector<int> replay_seq;
                    replay_seq.reserve((size_t)n_past_);
                    if ((int)current_tokens_.size() != n_past_ - 1 && getRank() == 0)
                        LOG_WARN("[LayerReplayIsoGuard] current_tokens_.size()=" << current_tokens_.size() << " expected=" << (n_past_ - 1));
                    // Copy prior tokens
                    int prior_count = std::min((int)current_tokens_.size(), n_past_ - 1);
                    for (int i = 0; i < prior_count; ++i)
                        replay_seq.push_back(current_tokens_[i]);
                    // Append the new token we are decoding now
                    replay_seq.push_back(token_id);
                    if ((int)replay_seq.size() != n_past_ && getRank() == 0)
                        LOG_WARN("[LayerReplayIsoGuard] replay_seq.size()=" << replay_seq.size() << " n_past_=" << n_past_);
                    if (getRank() == 0)
                        LOG_INFO("[LayerReplayIsoTrace] full_replay_prefix len=" << replay_seq.size());
                    // Collect incremental rows for THIS token & pipeline (rank 0 for decision)
                    std::vector<LayerTokenDiffRow> inc_rows;
                    inc_rows.reserve(16);
                    if (getRank() == 0)
                    {
                        LOG_INFO("[LayerReplayIsoTrace] collecting_inc_rows start offset=" << layer_rows_offset_before << " total_rows=" << last_layer_token_rows_.size());
                        for (size_t i = layer_rows_offset_before; i < last_layer_token_rows_.size(); ++i)
                        {
                            const auto &r = last_layer_token_rows_[i];
                            if (!r.pipeline || r.pipeline == this)
                                inc_rows.push_back(r);
                        }
                        // Retro scan: include any internal attention rows for current token (seq_len==replay_seq.size())
                        // that may have been captured before offset (defensive; should normally be none, but adds robustness).
                        int retro_added = 0;
                        for (size_t i = 0; i < layer_rows_offset_before; ++i)
                        {
                            const auto &r = last_layer_token_rows_[i];
                            if (!(r.stage.rfind("attn_int_", 0) == 0))
                                continue;
                            if (!r.incremental)
                                continue;
                            if (r.seq_len != (int)replay_seq.size())
                                continue;
                            bool exists = false;
                            for (auto &e : inc_rows)
                                if (&e == &r)
                                {
                                    exists = true;
                                    break;
                                }
                            if (exists)
                                continue;
                            inc_rows.push_back(r);
                            ++retro_added;
                        }
                        if (retro_added > 0)
                            LOG_INFO("[LayerReplayIsoTrace] retro_added_internal_rows=" << retro_added);
                        // List stages gathered
                        if (penv.layer_token_diff_verbose)
                        {
                            std::ostringstream oss;
                            oss << "[LayerReplayIsoTrace] inc_rows_list=";
                            for (size_t i = 0; i < inc_rows.size(); ++i)
                            {
                                oss << inc_rows[i].stage;
                                if (i + 1 < inc_rows.size())
                                    oss << ",";
                            }
                            LOG_INFO(oss.str());
                        }
                        LOG_INFO("[LayerReplayIsoTrace] collected_inc_rows count=" << inc_rows.size());
                    }
                    bool run_replay = true; // Force replay; we want rep rows regardless of inc row capture
                    if (getRank() == 0)
                        LOG_INFO("[LayerReplayIsoDecision] forced run_replay=1 inc_rows=" << inc_rows.size());

                    if (penv.layer_token_diff_verbose && getRank() == 0)
                    {
                        LOG_INFO("[LayerReplayIsoDiag] inc_rows=" << inc_rows.size() << " total_rows_before=" << last_layer_token_rows_.size());
                    }
                    // Only rank 0 manipulates diagnostic row buffer; others just execute replay
                    if (getRank() == 0)
                        last_layer_token_rows_.clear();
                    if (getRank() == 0)
                        LOG_INFO("[LayerReplayIsoExecPrep] creating replay pipeline token_pos=" << (replay_seq.size() - 1));
                    auto replay_pipe = createDistributedTransformerPipeline(config_);
                    if (getRank() == 0)
                        LOG_INFO("[LayerReplayIsoExecPrep] created replay pipeline ptr=" << (void *)replay_pipe.get());
                    auto replay_logits = TensorFactory::create_simple({(int)replay_seq.size(), config_.vocab_size});
                    if (getRank() == 0)
                        LOG_INFO("[LayerReplayIsoExecCall] about to execute replay pipeline seq_len=" << replay_seq.size());
                    bool replay_ok = replay_pipe && replay_pipe->execute(replay_seq, weights, replay_logits);
                    if (getRank() == 0)
                        LOG_INFO("[LayerReplayIsoExecCall] returned from replay pipeline");
                    std::vector<LayerTokenDiffRow> rep_rows;
                    if (getRank() == 0)
                    {
                        LOG_INFO("[LayerReplayIsoExec] replay_ok=" << (replay_ok ? 1 : 0) << " rep_rows_after_execute=" << last_layer_token_rows_.size());
                        // Filter replay rows to only the LAST token (absolute pos = n_past_-1) by selecting rows whose seq_len == replay_seq.size()
                        // Assumption: internal capture sets seq_len to total sequence length seen by kernel at capture time.
                        for (auto &r_all : last_layer_token_rows_)
                        {
                            if (r_all.seq_len == (int)replay_seq.size())
                                rep_rows.push_back(r_all);
                        }
                        if (rep_rows.empty())
                            LOG_WARN("[LayerReplayIsoFilter] no final-token replay rows found (expected seq_len=" << replay_seq.size() << ") using unfiltered set");
                        if (rep_rows.empty())
                            rep_rows = last_layer_token_rows_; // fallback to prior behavior
                        // Restore incremental rows for future tokens (preserve only inc_rows)
                        last_layer_token_rows_.clear();
                        last_layer_token_rows_.insert(last_layer_token_rows_.end(), inc_rows.begin(), inc_rows.end());
                    }
                    // No barrier here; rely on mirrored execution order.
                    if (!replay_ok)
                    {
                        if (getRank() == 0)
                            LOG_WARN("[LayerReplayIso] replay execute failed");
                    }
                    else if (getRank() == 0 && rep_rows.empty())
                    {
                        LOG_WARN("[LayerReplayIso] replay produced no rows");
                    }
                    // (Removed duplicate second incremental collection block)
                    // Build maps layer->stage rows
                    auto order = [](const std::string &v)
                    {
							// Fine-grained internal attention stages inserted between qkv_in and attn_out
							if(v=="attn_qkv_in") return 0;
							if(v=="attn_int_q_proj") return 1;
							if(v=="attn_int_k_proj") return 2;
							if(v=="attn_int_q_rope") return 3;
							if(v=="attn_int_k_rope") return 4;
							if(v=="attn_int_context") return 5;
							if(v=="attn_int_context_full") return 6; // incremental reconstructed window
							if(v=="attn_int_out_partial") return 7;
							if(v=="attn_out") return 8;
							if(v=="attn_residual") return 9;
							if(v=="ffn_norm") return 10;
							if(v=="ffn_out") return 11;
							if(v=="layer_output") return 12;
							// Legacy / unused placeholder retained for compatibility
							if(v=="attn_norm") return 100;
							return 500; };
                    struct StagePair
                    {
                        const LayerTokenDiffRow *inc;
                        const LayerTokenDiffRow *rep;
                    };
                    std::map<int, std::vector<StagePair>> layer_pairs;
                    if (getRank() == 0)
                        LOG_INFO("[LayerReplayIsoSentinel2] layer_pairs_container_initialized layers_inc=" << inc_rows.size());
                    for (auto &ir : inc_rows)
                        layer_pairs[ir.layer];
                    // DEBUG: count internal incremental rows before pairing
                    if (penv.layer_token_diff_verbose && getRank() == 0)
                    {
                        int inc_int_ct = 0;
                        for (auto &ir : inc_rows)
                            if (ir.stage.rfind("attn_int_", 0) == 0)
                                ++inc_int_ct;
                        LOG_INFO("[LayerReplayIsoPrePairDiag] inc_internal_rows=" << inc_int_ct);
                    }
                    for (auto &ir : inc_rows)
                    {
                        if (getRank() == 0 && (&ir == &inc_rows.front()))
                        {
                            LOG_INFO("[LayerReplayIsoSentinel3] first_inc_row stage=" << ir.stage << " layer=" << ir.layer);
                        }
                        const LayerTokenDiffRow *rep_match = nullptr;
                        // With filtered rep_rows containing only last token captures, we just need direct match.
                        for (auto &rr : rep_rows)
                        {
                            if (rr.layer == ir.layer && rr.stage == ir.stage)
                            {
                                rep_match = &rr;
                                break;
                            }
                        }
                        if (!rep_match)
                        {
                            // Stage-only fallback
                            for (auto &rr : rep_rows)
                                if (rr.stage == ir.stage)
                                {
                                    rep_match = &rr;
                                    break;
                                }
                        }
                        layer_pairs[ir.layer].push_back(StagePair{&ir, rep_match});
                    }
                    // Diagnostic: enumerate unmatched stages (verbose mode only)
                    if (penv.layer_token_diff_verbose && getRank() == 0)
                    {
                        for (auto &kv_diag : layer_pairs)
                        {
                            int layer = kv_diag.first;
                            auto &pairs = kv_diag.second;
                            std::vector<std::string> inc_only;
                            std::vector<std::string> rep_only;
                            // Build rep stage set for quick lookup
                            std::vector<std::string> rep_stages;
                            rep_stages.reserve(rep_rows.size());
                            for (auto &rr : rep_rows)
                                if (rr.layer == layer)
                                    rep_stages.push_back(rr.stage);
                            auto has_rep = [&](const std::string &s)
                            { return std::find(rep_stages.begin(), rep_stages.end(), s) != rep_stages.end(); };
                            for (auto &p : pairs)
                            {
                                if (!p.rep)
                                    inc_only.push_back(p.inc->stage);
                            }
                            // Find rep-only (stages captured in replay but not in incremental set)
                            for (auto &rr : rep_rows)
                                if (rr.layer == layer)
                                {
                                    bool found = false;
                                    for (auto &p : pairs)
                                        if (p.inc->stage == rr.stage)
                                        {
                                            found = true;
                                            break;
                                        }
                                    if (!found)
                                        rep_only.push_back(rr.stage);
                                }
                            if (!inc_only.empty() || !rep_only.empty())
                            {
                                std::ostringstream oss;
                                oss << "[LayerReplayIsoStageDiag] layer=" << layer;
                                if (!inc_only.empty())
                                {
                                    oss << " inc_only=";
                                    for (size_t i = 0; i < inc_only.size(); ++i)
                                    {
                                        oss << inc_only[i];
                                        if (i + 1 < inc_only.size())
                                            oss << ",";
                                    }
                                }
                                if (!rep_only.empty())
                                {
                                    oss << " rep_only=";
                                    for (size_t i = 0; i < rep_only.size(); ++i)
                                    {
                                        oss << rep_only[i];
                                        if (i + 1 < rep_only.size())
                                            oss << ",";
                                    }
                                }
                                LOG_INFO(oss.str());
                            }
                        }
                    }
                    const double rel_l2_warn = 1e-5;
                    bool exceed = false; // legacy flag preserved (not used for early break now)
                    bool first_exceed_recorded = false;
                    DiffSummary first_exceed_ds{};
                    int first_exceed_layer = 0;
                    std::string first_exceed_stage;
                    // Synthetic pairing pass: attach replay rows to internal incremental rows with layer=-1
                    // that lack a rep match but have a stage starting with attn_int_. This enables diff logging
                    // for kernel-level captures executed outside a fully indexed pipeline layer context.
                    auto synth_pair_internal = [&]()
                    {
                        auto it = layer_pairs.find(-1);
                        if (it == layer_pairs.end())
                            return; // no synthetic bucket
                        for (auto &sp : it->second)
                        {
                            if (sp.rep)
                                continue; // already matched (unlikely for layer -1)
                            if (!sp.inc || sp.inc->stage.rfind("attn_int_", 0) != 0)
                                continue;
                            // find first replay row with same stage (any layer)
                            for (auto &rr : rep_rows)
                            {
                                if (rr.stage == sp.inc->stage)
                                {
                                    sp.rep = &rr;
                                    break;
                                }
                            }
                        }
                    };
                    synth_pair_internal();
                    for (auto &kv : layer_pairs)
                    {
                        int layer = kv.first;
                        auto &pairs = kv.second;
                        std::stable_sort(pairs.begin(), pairs.end(), [&](const StagePair &a, const StagePair &b)
                                         { return order(a.inc->stage) < order(b.inc->stage); });
                        for (auto &sp : pairs)
                        {
                            if (penv.layer_token_diff_verbose && getRank() == 0 && sp.inc)
                            {
                                // Iteration diagnostic before any filtering logic; helps identify ordering, missing reps, or size mismatches
                                int ord = order(sp.inc->stage);
                                size_t rep_sz = sp.rep ? sp.rep->values.size() : 0;
                                LOG_INFO("[LayerReplayIsoStageIterDiag] layer=" << layer
                                                                                << " stage=" << sp.inc->stage
                                                                                << " order=" << ord
                                                                                << " has_rep=" << (sp.rep ? 1 : 0)
                                                                                << " inc_size=" << sp.inc->values.size()
                                                                                << " rep_size=" << rep_sz);
                            }
                            bool synth_self_pair = false;
                            if (!sp.rep)
                            {
                                // Now that we filter to last token, missing rep indicates genuine capture mismatch; skip.
                                if (penv.layer_token_diff_verbose && getRank() == 0 && sp.inc)
                                    LOG_WARN("[LayerReplayIsoStageSkip] layer=" << layer << " stage=" << sp.inc->stage << " reason=no_final_replay_row");
                                continue;
                            }
                            if (sp.inc->values.empty())
                            {
                                if (penv.layer_token_diff_verbose && getRank() == 0)
                                {
                                    LOG_INFO("[LayerReplayIsoSizeMismatch] layer=" << layer << " stage=" << sp.inc->stage << " reason=empty_inc_values");
                                }
                                continue;
                            }
                            if (!synth_self_pair && sp.rep && sp.inc->values.size() != sp.rep->values.size())
                            {
                                if (penv.layer_token_diff_verbose && getRank() == 0)
                                {
                                    LOG_INFO("[LayerReplayIsoSizeMismatch] layer=" << layer
                                                                                   << " stage=" << sp.inc->stage
                                                                                   << " inc_size=" << sp.inc->values.size()
                                                                                   << " rep_size=" << (sp.rep ? sp.rep->values.size() : 0));
                                }
                                continue;
                            }
                            DiffSummary ds{};
                            // Normal diff (no more synthetic self-pairs for internal stages once full prefix replay is used)
                            ds = computeDiffSummary(sp.inc->values.data(), sp.rep->values.data(), sp.inc->values.size());
                            // Always log internal stages and attn_out for layer 0; previously only attn_qkv_in was forced.
                            if (getRank() == 0)
                            {
                                // New policy: Always emit all internal attention stages (attn_int_*) for every layer
                                // plus the macro attention boundary stages (attn_qkv_in, attn_out) so we can localize
                                // the first divergence regardless of layer index. Original behavior restricted most
                                // stages to layer==0 which hid per-layer internal drift.
                                bool force_log = false;
                                const bool is_internal_attn = (sp.inc->stage.rfind("attn_int_", 0) == 0);
                                const bool is_attn_boundary = (sp.inc->stage == "attn_qkv_in" || sp.inc->stage == "attn_out");
                                if (is_internal_attn || is_attn_boundary)
                                    force_log = true;
                                // Preserve previous special cases (synthetic bucket layer -1 or self-pair)
                                if (layer == -1 && is_internal_attn)
                                    force_log = true;
                                if (synth_self_pair && is_internal_attn)
                                    force_log = true;
                                // Backward compatibility: still log any other attention-prefixed stage for layer 0
                                if (!force_log && layer == 0 && sp.inc->stage.rfind("attn_", 0) == 0)
                                    force_log = true;
                                if (force_log)
                                {
                                    LOG_INFO("[LayerReplayIsoStage] token_pos=" << (replay_seq.size() - 1)
                                                                                << " layer=" << layer
                                                                                << " stage=" << sp.inc->stage
                                                                                << " rel_l2=" << ds.rel_l2
                                                                                << " max_abs=" << ds.max_abs
                                                                                << " mean_abs=" << ds.mean_abs);
                                }
                            }
                            // Optional detailed per-stage diff logging (verbose mode)
                            if (getRank() == 0 && penv.layer_token_diff_verbose)
                            {
                                // Compute simple checksum diagnostics to distinguish scale vs orientation drift
                                long double sum_inc = 0.0L, sum_rep = 0.0L, sumsq_inc = 0.0L, sumsq_rep = 0.0L;
                                for (size_t ci = 0; ci < sp.inc->values.size(); ++ci)
                                {
                                    long double ai = sp.inc->values[ci];
                                    long double bi = sp.rep->values[ci];
                                    sum_inc += ai;
                                    sum_rep += bi;
                                    sumsq_inc += ai * ai;
                                    sumsq_rep += bi * bi;
                                }
                                long double l2_inc = std::sqrt((double)sumsq_inc);
                                long double l2_rep = std::sqrt((double)sumsq_rep);
                                LOG_INFO("[LayerReplayIsoStageVerbose] token_pos=" << (replay_seq.size() - 1)
                                                                                   << " layer=" << layer
                                                                                   << " stage=" << sp.inc->stage
                                                                                   << " rel_l2=" << ds.rel_l2
                                                                                   << " max_abs=" << ds.max_abs
                                                                                   << " mean_abs=" << ds.mean_abs
                                                                                   << " size=" << sp.inc->values.size()
                                                                                   << " sum_inc=" << (double)sum_inc
                                                                                   << " sum_rep=" << (double)sum_rep
                                                                                   << " l2_inc=" << (double)l2_inc
                                                                                   << " l2_rep=" << (double)l2_rep
                                                                                   << " sum_diff=" << (double)(sum_inc - sum_rep)
                                                                                   << " l2_ratio=" << ((l2_rep > 0) ? (double)(l2_inc / l2_rep) : 0.0));
                            }
                            if (ds.rel_l2 > rel_l2_warn && !first_exceed_recorded)
                            {
                                first_exceed_recorded = true;
                                first_exceed_ds = ds;
                                first_exceed_layer = layer;
                                first_exceed_stage = sp.inc->stage;
                                if (getRank() == 0)
                                    LOG_WARN("[LayerReplayIsoFirstExceed] token_pos=" << (replay_seq.size() - 1)
                                                                                      << " layer=" << layer
                                                                                      << " stage=" << sp.inc->stage
                                                                                      << " rel_l2=" << ds.rel_l2
                                                                                      << " max_abs=" << ds.max_abs
                                                                                      << " worst_index=" << ds.worst_index);
                                if (getRank() == 0 && layer == 0 && sp.inc->stage == "attn_out")
                                {
                                    const LayerTokenDiffRow *inc_qkv = nullptr;
                                    const LayerTokenDiffRow *rep_qkv = nullptr;
                                    for (auto &sp2 : pairs)
                                        if (sp2.rep && sp2.inc->stage == "attn_qkv_in")
                                        {
                                            inc_qkv = sp2.inc;
                                            rep_qkv = sp2.rep;
                                            break;
                                        }
                                    if (inc_qkv && rep_qkv && layer < (int)weights.wq.size() && weights.wq[layer])
                                    {
                                        auto wqT = weights.wq[layer];
                                        const auto &wshape = wqT->shape();
                                        if (wshape.size() == 2)
                                        {
                                            int R = wshape[0], C = wshape[1];
                                            int d_model = config_.d_model;
                                            const float *wptr = wqT->data();
                                            std::vector<float> q_inc, q_rep;
                                            if (R == d_model)
                                            {
                                                q_inc.assign(C, 0.f);
                                                q_rep.assign(C, 0.f);
                                                for (int i = 0; i < d_model; ++i)
                                                {
                                                    float vi = inc_qkv->values[i];
                                                    float vr = rep_qkv->values[i];
                                                    const float *wrow = wptr + (size_t)i * C;
                                                    for (int j = 0; j < C; ++j)
                                                    {
                                                        q_inc[j] += vi * wrow[j];
                                                        q_rep[j] += vr * wrow[j];
                                                    }
                                                }
                                            }
                                            else if (C == d_model)
                                            {
                                                int proj = R;
                                                q_inc.assign(proj, 0.f);
                                                q_rep.assign(proj, 0.f);
                                                for (int j = 0; j < proj; ++j)
                                                {
                                                    const float *wrow = wptr + (size_t)j * C;
                                                    float ai = 0.f, ar = 0.f;
                                                    for (int c = 0; c < C; ++c)
                                                    {
                                                        ai += inc_qkv->values[c] * wrow[c];
                                                        ar += rep_qkv->values[c] * wrow[c];
                                                    }
                                                    q_inc[j] = ai;
                                                    q_rep[j] = ar;
                                                }
                                            }
                                            if (!q_inc.empty() && q_inc.size() == q_rep.size())
                                            {
                                                DiffSummary qds = computeDiffSummary(q_inc.data(), q_rep.data(), q_inc.size());
                                                if (getRank() == 0)
                                                    LOG_WARN("[LayerReplayIsoQProj] token_pos=" << (replay_seq.size() - 1) << " layer=0 rel_l2=" << qds.rel_l2 << " max_abs=" << qds.max_abs << " mean_abs=" << qds.mean_abs << " size=" << q_inc.size());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (first_exceed_recorded && getRank() == 0)
                    {
                        LOG_WARN("[LayerReplayIso] token_pos=" << (replay_seq.size() - 1)
                                                               << " first_exceed_layer=" << first_exceed_layer
                                                               << " stage=" << first_exceed_stage
                                                               << " rel_l2=" << first_exceed_ds.rel_l2
                                                               << " max_abs=" << first_exceed_ds.max_abs
                                                               << " worst_index=" << first_exceed_ds.worst_index
                                                               << " inc=" << first_exceed_ds.value_a
                                                               << " rep=" << first_exceed_ds.value_b);
                    }
                    if (!first_exceed_recorded && getRank() == 0)
                    {
                        LOG_INFO("[LayerReplayIso] token_pos=" << (replay_seq.size() - 1)
                                                               << " all_layers_all_stages_rel_l2<=" << rel_l2_warn);
                    }
                    // No final barrier: avoid interference with outer harness collectives.
                }
                catch (const std::exception &e)
                {
                    if (getRank() == 0)
                        LOG_ERROR("[LayerReplayIso] exception: " << e.what());
                }
            } // end symmetry else
        }
        return true;
    }

} // namespace llaminar

// === Section 5: COSMA fused prefill attention implementation (standalone) ===
namespace llaminar
{
    bool DistributedTransformerPipeline::executePrefillAttentionCosma(int layer_idx,
                                                                      std::shared_ptr<TensorBase> &input,
                                                                      const ModelWeights &weights,
                                                                      std::shared_ptr<TensorBase> &attn_norm_out,
                                                                      std::shared_ptr<TensorBase> &attn_out,
                                                                      PrefillAttentionTiming &timing)
    {
        PERF_SCOPED_TIMER("DistributedTransformerPipeline::executePrefillAttentionCosma");
        CosmaPrefillManager &manager = CosmaPrefillManager::instance();
        const int seq_len = input->shape()[0];
        const int hidden_size = config_.d_model;
        const int head_dim = config_.head_dim;
        const int n_heads = config_.n_head;
        const int n_kv_heads = config_.n_head_kv;
        const int total_head_dim = n_heads * head_dim;
        const int kv_head_dim = n_kv_heads * head_dim;
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
                                               seq_len, hidden_size, config_.eps, scale, false);
        auto fused_end = std::chrono::high_resolution_clock::now();
        if ((!fused.normalized.mat && !fused.normalized.host_owned) || (!fused.q.mat && !fused.q.host_owned) || (!fused.k.mat && !fused.k.host_owned) || (!fused.v.mat && !fused.v.host_owned))
        {
            LOG_ERROR("COSMA fused_rmsnorm_qkv incomplete for layer " << layer_idx);
            return false;
        }
        std::vector<float> norm_buf((size_t)seq_len * hidden_size, 0.f);
        manager.to_row_major(fused.normalized, norm_buf.data());
        std::memcpy(attn_norm_out->data(), norm_buf.data(), norm_buf.size() * sizeof(float));
        std::vector<float> q_buf((size_t)seq_len * total_head_dim, 0.f), k_buf((size_t)seq_len * kv_head_dim, 0.f), v_buf((size_t)seq_len * kv_head_dim, 0.f);
        manager.to_row_major(fused.q, q_buf.data(), true);
        manager.to_row_major(fused.k, k_buf.data(), true);
        manager.to_row_major(fused.v, v_buf.data(), true);
        timing.norm_ms = std::chrono::duration<double, std::milli>(fused_end - fused_start).count();
        std::vector<float> context_concat((size_t)seq_len * total_head_dim, 0.f);
        std::vector<float> q_head((size_t)seq_len * head_dim, 0.f), k_head((size_t)seq_len * head_dim, 0.f), v_head((size_t)seq_len * head_dim, 0.f);
        std::vector<float> scores((size_t)seq_len * seq_len, 0.f), context_head((size_t)seq_len * head_dim, 0.f);
        constexpr float theta_base = 10000.0f;
        auto attention_start = std::chrono::high_resolution_clock::now();
        for (int head = 0; head < n_heads; ++head)
        {
            int kv_head = head % n_kv_heads;
            for (int row = 0; row < seq_len; ++row)
            {
                const float *q_src = q_buf.data() + (size_t)row * total_head_dim + head * head_dim;
                const float *k_src = k_buf.data() + (size_t)row * kv_head_dim + kv_head * head_dim;
                const float *v_src = v_buf.data() + (size_t)row * kv_head_dim + kv_head * head_dim;
                std::memcpy(q_head.data() + (size_t)row * head_dim, q_src, head_dim * sizeof(float));
                std::memcpy(k_head.data() + (size_t)row * head_dim, k_src, head_dim * sizeof(float));
                std::memcpy(v_head.data() + (size_t)row * head_dim, v_src, head_dim * sizeof(float));
            }
            for (int row = 0; row < seq_len; ++row)
            {
                float *q_row = q_head.data() + (size_t)row * head_dim;
                float *k_row = k_head.data() + (size_t)row * head_dim;
                float position = (float)(n_past_ + row);
                for (int dpair = 0; dpair < head_dim / 2; ++dpair)
                {
                    float theta = 1.0f / std::pow(theta_base, (2.0f * (float)dpair) / (float)head_dim);
                    float c = std::cos(position * theta);
                    float s = std::sin(position * theta);
                    float q0 = q_row[2 * dpair], q1 = q_row[2 * dpair + 1];
                    q_row[2 * dpair] = q0 * c - q1 * s;
                    q_row[2 * dpair + 1] = q0 * s + q1 * c;
                    float k0 = k_row[2 * dpair], k1 = k_row[2 * dpair + 1];
                    k_row[2 * dpair] = k0 * c - k1 * s;
                    k_row[2 * dpair + 1] = k0 * s + k1 * c;
                }
            }
            for (int row = 0; row < seq_len; ++row)
            {
                const float *q_row = q_head.data() + (size_t)row * head_dim;
                float *score_row = scores.data() + (size_t)row * seq_len;
                float row_max = -std::numeric_limits<float>::infinity();
                for (int col = 0; col <= row; ++col)
                {
                    const float *k_row = k_head.data() + (size_t)col * head_dim;
                    float dot = 0.f;
                    for (int d = 0; d < head_dim; ++d)
                        dot += q_row[d] * k_row[d];
                    float scaled = dot * scale;
                    score_row[col] = scaled;
                    if (scaled > row_max)
                        row_max = scaled;
                }
                for (int col = row + 1; col < seq_len; ++col)
                    score_row[col] = -std::numeric_limits<float>::infinity();
                double denom = 0.0;
                for (int col = 0; col <= row; ++col)
                {
                    float v = std::exp(score_row[col] - row_max);
                    score_row[col] = v;
                    denom += v;
                }
                float inv = denom > 0.0 ? (float)(1.0 / denom) : 1.0f;
                for (int col = 0; col <= row; ++col)
                    score_row[col] *= inv;
                for (int col = row + 1; col < seq_len; ++col)
                    score_row[col] = 0.f;
                float *ctx_row = context_head.data() + (size_t)row * head_dim;
                std::fill(ctx_row, ctx_row + head_dim, 0.f);
                for (int col = 0; col <= row; ++col)
                {
                    const float *v_row = v_head.data() + (size_t)col * head_dim;
                    float w = scores[(size_t)row * seq_len + col];
                    for (int d = 0; d < head_dim; ++d)
                        ctx_row[d] += w * v_row[d];
                }
            }
            for (int row = 0; row < seq_len; ++row)
            {
                float *dst = context_concat.data() + (size_t)row * total_head_dim + head * head_dim;
                const float *src = context_head.data() + (size_t)row * head_dim;
                std::memcpy(dst, src, head_dim * sizeof(float));
            }
        }
        auto attention_end = std::chrono::high_resolution_clock::now();
        timing.attention_ms = std::chrono::duration<double, std::milli>(attention_end - attention_start).count();
        auto proj_start = std::chrono::high_resolution_clock::now();
        bool ok = adaptiveMatMul(context_concat.data(), weights.wo[layer_idx]->data(), attn_out->data(), seq_len, hidden_size, total_head_dim, true, false, false, false, 1.0f, 0.0f);
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
