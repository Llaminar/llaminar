// Canonical header for Qwen-specific pipeline implementation (renamed from distributed_transformer_pipeline.h).
// NOTE: During deprecation window, distributed_transformer_pipeline.h is a shim that includes this file.
// New code must include qwen_pipeline.h directly.
#pragma once

#include "pipeline_base.h"
#include "abstract_pipeline.h"
#include "mpi_context.h"
#include "large_matmul_plan.h"

// Forward declarations
class ModelLoader;
#include "kernels/MPILinearKernel.h"
#include "kernels/MPIRMSNormKernel.h"
#include "kernels/MPIAttentionKernel.h"
#include "kernels/MPISwiGLUKernel.h"
#include "kernels/MPIRoPEKernel.h"
#include "kernels/MPIResidualKernel.h"
#include "transformer_config.h"
#include "model_loader.h"
#include "logger.h"
#include <memory>
#include <vector>
#include <mpi.h>
#include <atomic>

namespace llaminar
{

    class QwenPipeline : public PipelineBase, public AbstractPipeline
    {
    public:
        using LayerConfig = TransformerLayerConfig;

        struct ModelWeights
        {
            std::shared_ptr<TensorBase> token_embedding;
            std::vector<std::shared_ptr<TensorBase>> attn_norm_weight;
            std::vector<std::shared_ptr<TensorBase>> wq;
            std::vector<std::shared_ptr<TensorBase>> wk;
            std::vector<std::shared_ptr<TensorBase>> wv;
            std::vector<std::shared_ptr<TensorBase>> wo;
            std::vector<std::shared_ptr<TensorBase>> ffn_norm_weight;
            std::vector<std::shared_ptr<TensorBase>> w_gate;
            std::vector<std::shared_ptr<TensorBase>> w_up;
            std::vector<std::shared_ptr<TensorBase>> w_down;
            std::shared_ptr<TensorBase> output_norm_weight;
            std::shared_ptr<TensorBase> lm_head;
        };

        explicit QwenPipeline(const ModelConfig &config);

        /**
         * @brief Construct with explicit MPI context (preferred for new code)
         * @param config Model configuration including architecture and layer settings
         * @param ctx MPI context with rank, size, and communicator
         */
        explicit QwenPipeline(const ModelConfig &config, const MPIContext &ctx);

        ~QwenPipeline() override;

        bool execute(const std::vector<int> &token_ids,
                     const ModelWeights &weights,
                     std::shared_ptr<TensorBase> &output);

        bool validate(const ModelWeights &weights) const;

        const TransformerLayerConfig &getConfig() const { return config_.getLayerConfig(); }
        const ModelConfig &config() const override { return config_; }

        void enableKVCache(bool enable) { use_kv_cache_ = enable; }
        void setSequencePosition(int pos) { n_past_ = pos; }
        void setStagePrefill() { is_prefill_stage_ = true; }
        void setStageDecode() { is_prefill_stage_ = false; }
        bool isPrefillStage() const { return is_prefill_stage_; }

        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;
        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getName() const { return "QwenPipeline"; }
        std::string name() const override { return "QwenPipeline"; }
        std::string getKernelType() const override { return "QwenPipeline"; }
        size_t getExpectedInputCount() const override { return 1; }
        size_t getExpectedOutputCount() const override { return 1; }

        // AbstractPipeline interface
        bool prefill(const std::vector<int> &tokens,
                     const IModelWeights &weights,
                     StageContext &ctx) override;
        bool decode(int next_token,
                    const IModelWeights &weights,
                    StageContext &ctx) override;
        bool logits(std::shared_ptr<TensorBase> &out_logits) override;
        const KVCacheState *kvCacheState() const override;
        bool ensureKVCapacity(int required_tokens) override;

        /**
         * @brief Load model weights from file.
         *
         * @param path Path to the model file (e.g., GGUF file)
         * @return Loaded weights wrapped in IModelWeights implementation
         *
         * This implementation wraps the legacy loadModelWeights() free function.
         */
        std::unique_ptr<IModelWeights> loadWeights(const std::string &path) override;

        bool incrementalDecodeToken(int token_id,
                                    const ModelWeights &weights,
                                    std::shared_ptr<TensorBase> &output_logits);

        // Inline small helper to avoid link-order issues when referenced across TUs
        inline std::shared_ptr<TensorBase> embedSingleToken(int token_id,
                                                            const std::shared_ptr<TensorBase> &embedding_weight)
        {
            if (!embedding_weight || embedding_weight->shape().size() != 2)
            {
                LOG_ERROR("embedSingleToken: invalid embedding weight");
                return nullptr;
            }
            int vocab = embedding_weight->shape()[0];
            int dim = embedding_weight->shape()[1];
            if (token_id < 0 || token_id >= vocab)
            {
                LOG_ERROR("embedSingleToken: token id out of range: " << token_id << "/" << vocab);
                return nullptr;
            }
            auto out = createLocalTensor({1, dim});
            const float *src = embedding_weight->data() + (size_t)token_id * dim;
            float *dst = out->data();
            std::memcpy(dst, src, sizeof(float) * dim);
            if (debugEnv().embedding.trace && getRank() == 0)
            {
                int show = std::min(dim, debugEnv().embedding.trace_dims);
                std::ostringstream oss;
                oss << "[EmbedTrace] token=" << token_id << " dims=" << dim << " preview=";
                for (int i = 0; i < show; ++i)
                {
                    oss << dst[i];
                    if (i + 1 < show)
                        oss << ',';
                }
                LOG_INFO(oss.str());
            }
            return out;
        }

        static size_t getSmallSeqFastPathCount() { return small_seq_fast_path_calls_.load(); }
        static void resetSmallSeqFastPathCount() { small_seq_fast_path_calls_.store(0); }
        static const std::vector<float> &getLastPreLMHidden() { return last_pre_lm_hidden_; }

        struct LayerActivationStat
        {
            double rms;
            double max_abs;
            double mean;
            int layer;
        };
        static const std::vector<LayerActivationStat> &getLastLayerActivationStats() { return last_layer_stats_; }

        struct LayerTokenDiffRow
        {
            int layer = -1;
            int seq_len = 0;
            // token_pos: 0-based position of the token row within the captured window.
            // For current instrumentation we approximate token_pos = seq_len - 1 (the "last" token
            // in the window) because we only capture final-row projections (and internal attention
            // stages) for the active pass. Full prefix replay (Option A) filters to the final token
            // anyway, so this provides stable tagging for future multi-token analysis without
            // requiring historical accumulation.
            int token_pos = -1;
            bool incremental = false;
            const void *pipeline = nullptr;
            // Stage label within the layer lifecycle. For backward compatibility existing
            // captures (prior to this patch) default to "layer_output" when not explicitly set.
            // Stages (ordered) currently emitted when layer_token_diff enabled:
            //  attn_norm, attn_out, attn_residual, ffn_norm, ffn_out, layer_output
            std::string stage = "layer_output";
            std::vector<float> values;
        };
        static const std::vector<LayerTokenDiffRow> &getLastLayerTokenRows() { return last_layer_token_rows_; }
        static void resetLayerTokenRows() { last_layer_token_rows_.clear(); }
        static void appendLayerTokenDiffRow(LayerTokenDiffRow row) { last_layer_token_rows_.push_back(std::move(row)); }
        // Accessors for replay parity exceed sentinel
        friend bool getReplayFirstExceedFlag();
        friend void resetReplayFirstExceedFlag();
        // Centralized internal attention capture helper: ensures pipeline pointer + verbose logging
        static void appendInternalAttnRow(QwenPipeline *pipeline,
                                          int layer,
                                          int seq_len,
                                          bool incremental,
                                          const std::string &stage,
                                          const float *data,
                                          size_t size)
        {
            const auto &env = debugEnv();
            if (!env.pipeline.layer_token_diff || !env.attention.internal_diff)
                return;
            if (!pipeline)
                return;                           // Safety check - pipeline pointer required for rank access
            const int rank = pipeline->getRank(); // Use pipeline's cached MPIContext rank
            if (rank != 0)
                return;
            if (!data || size == 0)
                return;
            LayerTokenDiffRow row;
            row.layer = layer;
            row.seq_len = seq_len;
            row.token_pos = seq_len > 0 ? (seq_len - 1) : -1;
            row.incremental = incremental;
            row.pipeline = pipeline;
            row.stage = stage;
            row.values.assign(data, data + size);
            last_layer_token_rows_.push_back(std::move(row));
            if (env.pipeline.layer_token_diff_verbose)
            {
                LOG_INFO("[LayerTokenInternalCapture] pipe=" << pipeline
                                                             << " layer=" << layer
                                                             << " stage=" << stage
                                                             << " seq_len=" << seq_len
                                                             << " size=" << size
                                                             << " total_rows=" << last_layer_token_rows_.size());
                if (layer == -1)
                {
                    LOG_INFO("[LayerTokenInternalCaptureDiag] warning=unset_layer_index stage=" << stage << " seq_len=" << seq_len << " suggest=setLayerIndex in executeTransformerLayer path");
                }
            }
        }
        static void resetDiagnostics()
        {
            last_pre_lm_hidden_.clear();
            last_layer_stats_.clear();
        }

        // Introspection helpers
        int getKVCacheCapacity() const noexcept { return kv_cache_state_.capacity_tokens; }
        int getKVCacheUsed() const noexcept { return kv_cache_state_.used_tokens; }
        int getKVCacheGrowthEvents() const noexcept { return kv_cache_state_.growth_events; }
        bool isKVDynamicInit() const noexcept { return kv_cache_dynamic_init_; }
        bool ensureKVCapacityPublic(int required_tokens) { return ensureKVCapacityInternal(required_tokens); }
        std::shared_ptr<TensorBase> allocateTestLocalTensor(const std::vector<int> &shape) { return createLocalTensor(shape); }

    private:
        void initializeKernels();
        bool executeEmbedding(const std::vector<int> &token_ids,
                              const std::shared_ptr<TensorBase> &embedding_weight,
                              std::shared_ptr<TensorBase> &embedded_output);
        bool executeTransformerLayer(int layer_idx,
                                     std::shared_ptr<TensorBase> &input,
                                     const ModelWeights &weights,
                                     std::shared_ptr<TensorBase> &output);
        bool executeOutputProjection(std::shared_ptr<TensorBase> &input,
                                     const ModelWeights &weights,
                                     std::shared_ptr<TensorBase> &output);
        bool decodeToken(int token_id,
                         const ModelWeights &weights,
                         std::shared_ptr<TensorBase> &output_logits);
        void traceFFNShardDiagnostics(const std::string &label,
                                      const float *data,
                                      int seq_len,
                                      int feature_dim);

        /**
         * @brief Helper to capture pipeline stage snapshots for parity testing
         *
         * This inline helper bridges existing capture call sites to the PipelineSnapshotManager.
         * Only active in debug builds when LLAMINAR_PARITY_CAPTURE=1.
         *
         * @param stage Pipeline stage being captured
         * @param layer_idx Layer index (-1 for non-layer stages)
         * @param tensor Tensor to capture
         */
        inline void captureIfEnabled(PipelineStage stage,
                                     int layer_idx,
                                     const std::shared_ptr<TensorBase> &tensor);

        struct PrefillAttentionTiming
        {
            double norm_ms{0.0};
            double attention_ms{0.0};
            double linear_ms{0.0};
        };
        // shouldUseCosmaPrefill() removed - use MatMulBackendSelector::selectBackend() instead
        bool executePrefillAttentionCosma(int layer_idx,
                                          const LargeMatmulPlan &plan,
                                          std::shared_ptr<TensorBase> &input,
                                          const ModelWeights &weights,
                                          std::shared_ptr<TensorBase> &attn_norm_out,
                                          std::shared_ptr<TensorBase> &attn_out,
                                          PrefillAttentionTiming &timing);
        void initializeKVCache(int seq_len);
        std::vector<std::shared_ptr<TensorBase>> createIntermediateTensors(int seq_len);
        bool ensureKVCapacityInternal(int required_tokens);

    private:
        ModelConfig config_;
        bool use_kv_cache_;
        int n_past_;
        bool is_prefill_stage_{true};
        bool kv_cache_dynamic_init_{false};
        struct InternalKVState
        {
            int capacity_tokens = 0;
            int used_tokens = 0;
            int growth_events = 0;
        } kv_cache_state_;
        bool in_incremental_pass_ = false;
        std::shared_ptr<TensorBase> last_logits_{};
        std::vector<int> current_tokens_{};
        mutable KVCacheState kv_snapshot_{};

    public: // KV cache tensors
        std::vector<std::shared_ptr<TensorBase>> k_cache_;
        std::vector<std::shared_ptr<TensorBase>> v_cache_;

    private:
        mutable std::chrono::high_resolution_clock::time_point start_time_;
        mutable double total_embedding_time_{};
        mutable double total_attention_time_{};
        mutable double total_linear_time_{};
        mutable double total_norm_time_{};
        mutable double total_activation_time_{};
        mutable double total_communication_time_{};
        static std::atomic<size_t> small_seq_fast_path_calls_;
        static std::vector<float> last_pre_lm_hidden_;
        static std::vector<LayerActivationStat> last_layer_stats_;
        static std::vector<LayerTokenDiffRow> last_layer_token_rows_;
    };

    // DEPRECATED: Use qwen_pipeline_adapter.h version instead
    // void registerQwenPipeline();

    // --- Inline migration shims (temporary until link-order cleanup) ---
    // Forward declaration for the full out-of-line loader implementation bridge
    // Factory & weight loading (implemented out-of-line in distributed_transformer_pipeline.cpp)
    std::unique_ptr<QwenPipeline> createQwenPipeline(
        const ModelConfig &config);

    /**
     * @brief Internal bridge function for weight loading
     * @note This is an implementation detail used by pipeline adapters.
     *       External code should use AbstractPipeline::loadWeights() instead.
     */
    QwenPipeline::ModelWeights loadModelWeights_impl_bridge(
        ModelLoader &loader,
        const QwenPipeline::LayerConfig &config);

} // namespace llaminar
