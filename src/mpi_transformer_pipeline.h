#pragma once

#include "pipeline_base.h"
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

    /**
     * @brief MPI-enabled transformer pipeline for distributed LLM inference using kernel composition
     *
     * This class coordinates the execution of a complete transformer forward pass
     * using composed MPI-distributed kernels across multiple processes. It manages:
     * - Kernel composition and orchestration through PipelineBase
     * - Distributed embedding lookup
     * - MPI-parallel RMS normalization
     * - MPI-parallel multi-head attention with head-wise distribution
     * - MPI-parallel linear transformations with COSMA integration
     * - MPI-parallel SwiGLU activation
     * - MPI-parallel rotary position embedding (RoPE)
     * - MPI-parallel residual connections
     * - Inter-kernel communication and data synchronization
     */
    class MPITransformerPipeline : public PipelineBase
    {
    public:
        // Type alias for compatibility
        using LayerConfig = TransformerLayerConfig;

        /**
         * @brief Model weights for transformer layers
         */
        struct ModelWeights
        {
            // Embedding weights
            std::shared_ptr<TensorBase> token_embedding;

            // Per-layer weights (indexed by layer number)
            std::vector<std::shared_ptr<TensorBase>> attn_norm_weight;
            std::vector<std::shared_ptr<TensorBase>> wq;
            std::vector<std::shared_ptr<TensorBase>> wk;
            std::vector<std::shared_ptr<TensorBase>> wv;
            std::vector<std::shared_ptr<TensorBase>> wo;
            std::vector<std::shared_ptr<TensorBase>> ffn_norm_weight;
            std::vector<std::shared_ptr<TensorBase>> w_gate;
            std::vector<std::shared_ptr<TensorBase>> w_up;
            std::vector<std::shared_ptr<TensorBase>> w_down;

            // Output layer
            std::shared_ptr<TensorBase> output_norm_weight;
            std::shared_ptr<TensorBase> lm_head;
        };

        /**
         * @brief Constructor for MPI transformer pipeline
         * @param config Transformer layer configuration
         */
        explicit MPITransformerPipeline(const LayerConfig &config);

        ~MPITransformerPipeline() override; // defined out-of-line to ensure vtable emission

        /**
         * @brief Execute complete transformer forward pass
         * @param token_ids Input token sequence
         * @param weights Model weights
         * @param output Output logits tensor
         * @return Success status
         */
        bool execute(const std::vector<int> &token_ids,
                     const ModelWeights &weights,
                     std::shared_ptr<TensorBase> &output);

        /**
         * @brief Validate pipeline configuration and weights
         * @param weights Model weights to validate
         * @return Validation success
         */
        bool validate(const ModelWeights &weights) const;

        /**
         * @brief Get pipeline configuration
         * @return Layer configuration
         */
        const LayerConfig &getConfig() const { return config_; }

        /**
         * @brief Enable/disable attention caching for inference
         * @param enable Cache enable flag
         */
        void enableKVCache(bool enable) { use_kv_cache_ = enable; }

        /**
         * @brief Set current position for attention computation
         * @param pos Current sequence position
         */
        void setSequencePosition(int pos) { n_past_ = pos; }

        /**
         * @brief Explicitly mark upcoming execution as prefill stage.
         * This avoids relying on implicit (n_past_==0) heuristics so the adapter
         * can replay full sequences during decode without confusing backend selection.
         */
        void setStagePrefill() { is_prefill_stage_ = true; }

        /**
         * @brief Mark subsequent execution as decode stage.
         */
        void setStageDecode() { is_prefill_stage_ = false; }

        /**
         * @brief Query whether current stage is prefill.
         */
        bool isPrefillStage() const { return is_prefill_stage_; }

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        /**
         * @brief Get pipeline name for debugging and profiling
         * @return Pipeline name string
         */
        std::string getName() const { return "MPITransformerPipeline"; }

        /**
         * @brief Get the kernel type name for debugging/logging
         * @return String identifying the kernel type
         */
        std::string getKernelType() const override { return "MPITransformerPipeline"; }

        /**
         * @brief Get expected number of input tensors
         * @return Number of input tensors this kernel expects
         */
        size_t getExpectedInputCount() const override { return 1; } // input_embeddings

        /**
         * @brief Get expected number of output tensors
         * @return Number of output tensors this kernel produces
         */
        size_t getExpectedOutputCount() const override { return 1; } // output_logits

        /**
         * @brief Attempt incremental single-token decode using KV cache and return logits for that token.
         * Falls back to false when incremental path is disabled (env flag), KV cache disabled, or any
         * internal failure occurs. On success, advances internal n_past_ and updates stage to decode.
         * @param token_id Next token id to decode
         * @param weights Model weights
         * @param output_logits Output logits tensor (1 x vocab)
         * @return true if incremental path executed successfully; false if caller should replay full sequence
         */
        bool incrementalDecodeToken(int token_id,
                                    const ModelWeights &weights,
                                    std::shared_ptr<TensorBase> &output_logits);

        /**
         * @brief Get number of times the replicated small-sequence fast path executed (process-local count).
         * Used by tests to verify activation behavior for seq_len < world_size.
         */
        static size_t getSmallSeqFastPathCount() { return small_seq_fast_path_calls_.load(); }

        /**
         * @brief Reset the small sequence fast path counter (primarily for test isolation).
         */
        static void resetSmallSeqFastPathCount() { small_seq_fast_path_calls_.store(0); }

        /**
         * @brief Access the last captured pre-LM head hidden state (post-final RMSNorm) if capture was enabled.
         * Capture is enabled when the environment variable LLAMINAR_PIPELINE_CAPTURE_PRE_LM is set before execute().
         * @return const reference to internal buffer (empty if not captured or pipeline not yet executed).
         */
        static const std::vector<float> &getLastPreLMHidden() { return last_pre_lm_hidden_; }

        /**
         * @brief Layer activation statistics (post-layer output) recorded when LLAMINAR_PIPELINE_LAYERWISE_STATS=1.
         */
        struct LayerActivationStat
        {
            double rms;
            double max_abs;
            double mean;
            int layer;
        };

        /**
         * @brief Get last recorded layer activation stats sequence.
         */
        static const std::vector<LayerActivationStat> &getLastLayerActivationStats() { return last_layer_stats_; }

        // Layer token diff capture (stores per-layer last-token hidden row when enabled)
        struct LayerTokenDiffRow
        {
            int layer = -1;                 // layer index
            int seq_len = 0;                // sequence length at time of capture
            bool incremental = false;       // captured during incrementalDecodeToken path
            const void *pipeline = nullptr; // originating pipeline instance (this)
            std::vector<float> values;      // size = d_model
        };
        static const std::vector<LayerTokenDiffRow> &getLastLayerTokenRows() { return last_layer_token_rows_; }
        static void resetLayerTokenRows() { last_layer_token_rows_.clear(); }

        /**
         * @brief Reset captured diagnostic buffers (mainly for tests).
         */
        static void resetDiagnostics()
        {
            last_pre_lm_hidden_.clear();
            last_layer_stats_.clear();
        }

    private:
        // Flag set while inside incrementalDecodeToken execution to tag diagnostics.
        bool in_incremental_pass_ = false;
        /**
         * @brief Initialize all kernels for transformer pipeline
         */
        void initializeKernels();

        /**
         * @brief Execute embedding lookup using composition
         * @param token_ids Input token sequence
         * @param embedding_weight Embedding weight matrix
         * @param embedded_output Output embedded sequence
         * @return Success status
         */
        bool executeEmbedding(const std::vector<int> &token_ids,
                              const std::shared_ptr<TensorBase> &embedding_weight,
                              std::shared_ptr<TensorBase> &embedded_output);

        /**
         * @brief Execute single transformer layer using kernel composition
         * @param layer_idx Layer index
         * @param input Input tensor from previous layer
         * @param weights Layer-specific weights
         * @param output Output tensor for this layer
         * @return Success status
         */
        bool executeTransformerLayer(int layer_idx,
                                     std::shared_ptr<TensorBase> &input,
                                     const ModelWeights &weights,
                                     std::shared_ptr<TensorBase> &output);

        /**
         * @brief Execute final output projection using kernel composition
         * @param input Final layer output
         * @param weights Model weights (norm + lm_head)
         * @param output Final logits
         * @return Success status
         */
        bool executeOutputProjection(std::shared_ptr<TensorBase> &input,
                                     const ModelWeights &weights,
                                     std::shared_ptr<TensorBase> &output);

        /**
         * @brief Incrementally decode a single next token using existing KV cache without replay.
         * Falls back to full execute path if KV cache disabled or env flag disables incremental.
         * @param token_id Next token id
         * @param weights Model weights
         * @param output_logits Output logits tensor (1 x vocab)
         */
        bool decodeToken(int token_id,
                         const ModelWeights &weights,
                         std::shared_ptr<TensorBase> &output_logits);

        // Helper: embed a single token into a 1 x d_model tensor
        std::shared_ptr<TensorBase> embedSingleToken(int token_id,
                                                     const std::shared_ptr<TensorBase> &embedding_weight);

        void traceFFNShardDiagnostics(const std::string &label,
                                      const float *data,
                                      int seq_len,
                                      int feature_dim);

        struct PrefillAttentionTiming
        {
            double norm_ms{0.0};
            double attention_ms{0.0};
            double linear_ms{0.0};
        };

        /**
         * @deprecated Legacy COSMA prefill decision helper.
         * Replaced by centralized backend selector (selectAttentionBackend).
         * TODO(dsanftenberg): Remove once all call sites migrate and tests updated.
         */
        bool shouldUseCosmaPrefill(int seq_len) const;
        bool executePrefillAttentionCosma(int layer_idx,
                                          std::shared_ptr<TensorBase> &input,
                                          const ModelWeights &weights,
                                          std::shared_ptr<TensorBase> &attn_norm_out,
                                          std::shared_ptr<TensorBase> &attn_out,
                                          PrefillAttentionTiming &timing);

        /**
         * @brief Initialize KV cache tensors for attention
         * @param seq_len Maximum sequence length
         */
        void initializeKVCache(int seq_len);

        /**
         * @brief Create intermediate tensors for transformer computation
         * @param seq_len Sequence length for tensor creation
         * @return Vector of intermediate tensors
         */
        std::vector<std::shared_ptr<TensorBase>> createIntermediateTensors(int seq_len);

    private:
        LayerConfig config_;                ///< Transformer configuration
        bool use_kv_cache_;                 ///< KV cache enabled flag
        int n_past_;                        ///< Current sequence position
        bool is_prefill_stage_{true};       ///< Explicit stage flag (prefill until first decode)
        bool kv_cache_dynamic_init_{false}; ///< Capture dynamic init intent at construction (stable across env mutations)

        // KV cache state (dynamic capacity management)
        struct KVCacheState
        {
            int capacity_tokens = 0; // total allocated token slots per layer (rows in k/v cache tensors)
            int used_tokens = 0;     // tokens populated (prefill length + decoded)
            int growth_events = 0;   // number of reallocations performed
        } kv_cache_state_;

        // Ensure KV cache has space for at least required_tokens. Returns true on success.
        bool ensureKVCapacity(int required_tokens);

    public:
        // Introspection helpers (used in tests / diagnostics)
        int getKVCacheCapacity() const noexcept { return kv_cache_state_.capacity_tokens; }
        int getKVCacheUsed() const noexcept { return kv_cache_state_.used_tokens; }
        int getKVCacheGrowthEvents() const noexcept { return kv_cache_state_.growth_events; }
        bool isKVDynamicInit() const noexcept { return kv_cache_dynamic_init_; }
        // Public wrapper for adapter access (maintains encapsulation of growth policy internals)
        bool ensureKVCapacityPublic(int required_tokens) { return ensureKVCapacity(required_tokens); }

        // Test utility: allocate a local tensor (exposed for lightweight unit tests that need deterministic weight fabrication)
        std::shared_ptr<TensorBase> allocateTestLocalTensor(const std::vector<int> &shape) { return createLocalTensor(shape); }

        // KV cache tensors (if enabled)
        std::vector<std::shared_ptr<TensorBase>> k_cache_; ///< Key cache per layer
        std::vector<std::shared_ptr<TensorBase>> v_cache_; ///< Value cache per layer

        // Performance monitoring
        mutable std::chrono::high_resolution_clock::time_point start_time_;
        mutable double total_embedding_time_;
        mutable double total_attention_time_;
        mutable double total_linear_time_;
        mutable double total_norm_time_;
        mutable double total_activation_time_;
        mutable double total_communication_time_;

        // Instrumentation: replicated small-sequence fast path invocation counter
        static std::atomic<size_t> small_seq_fast_path_calls_;

        // Diagnostics (static so tests can fetch after execute())
        static std::vector<float> last_pre_lm_hidden_;
        static std::vector<LayerActivationStat> last_layer_stats_;
        static std::vector<LayerTokenDiffRow> last_layer_token_rows_;
    };

    /**
     * @brief Factory function for creating MPI transformer pipeline
     * @param config Transformer configuration
     * @return Unique pointer to MPI transformer pipeline
     */
    std::unique_ptr<MPITransformerPipeline> createMPITransformerPipeline(
        const MPITransformerPipeline::LayerConfig &config);

    /**
     * @brief Utility function to load model weights from file/memory
     * @param model_path Path to model weights
     * @param config Transformer configuration
     * @return Model weights structure
     */
    MPITransformerPipeline::ModelWeights loadModelWeights(
        const std::string &model_path,
        const MPITransformerPipeline::LayerConfig &config);

    /**
     * @brief Utility function to load model weights using existing ModelLoader
     * @param loader Existing ModelLoader instance (already loaded)
     * @param config Transformer configuration
     * @return Model weights structure
     */
    MPITransformerPipeline::ModelWeights loadModelWeights(
        ModelLoader &loader,
        const MPITransformerPipeline::LayerConfig &config);

} // namespace llaminar