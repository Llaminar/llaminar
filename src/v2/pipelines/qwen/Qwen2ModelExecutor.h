/**
 * @file Qwen2ModelExecutor.h
 * @brief Qwen2-specific model executor for declarative forward pass execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * Qwen2ModelExecutor implements the model-level compute graph building
 * for Qwen2 architecture models. This class bridges Qwen2Pipeline's
 * weight and buffer management with the declarative execution framework.
 *
 * Key Responsibilities:
 * - Build compute graphs for embedding, transformer layers, and LM head
 * - Manage Qwen2-specific architecture parameters
 * - Integrate with Qwen2LayerExecutor for layer-level details
 *
 * Migration Path:
 * 1. Qwen2Pipeline creates Qwen2ModelExecutor
 * 2. Qwen2Pipeline::forward() can delegate to executeForward()
 * 3. Eventually, all forward logic moves to executor
 */

#pragma once

#include "../../execution/ModelExecutor.h"
#include "../../execution/ComputeStage.h"
#include "Qwen2LayerExecutor.h"
#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class Qwen2Pipeline;

    /**
     * @brief Configuration specific to Qwen2 model execution
     */
    struct Qwen2ModelExecutorConfig
    {
        // Model architecture
        int n_layers = 0;
        int d_model = 0;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int d_ff = 0;
        int vocab_size = 0;

        // FFN sharding (for tensor parallelism)
        int d_ff_local = 0; ///< Local FFN dim per rank
        bool ffn_column_parallel = false;

        // Precision and execution
        float rms_norm_eps = 1e-6f;
        float rope_theta = 10000.0f;
        ActivationPrecision activation_precision = ActivationPrecision::FP32;

        // Base executor config
        ModelExecutorConfig base_config;
    };

    /**
     * @brief Weight accessors for Qwen2 model
     *
     * Provides access to model weights without owning them.
     * The pipeline owns the weights; executor just references them.
     */
    struct Qwen2ModelWeights
    {
        // Global weights
        TensorBase *embedding_table = nullptr; ///< [vocab_size, d_model]
        TensorBase *final_norm = nullptr;      ///< [d_model]
        TensorBase *lm_head = nullptr;         ///< [vocab_size, d_model]

        // Per-layer weight accessor
        std::function<Qwen2LayerWeights(int layer_idx)> get_layer_weights;
    };

    /**
     * @brief Activation buffer set for model execution
     *
     * Pre-allocated buffers reused across forward passes.
     * The pipeline allocates these; executor uses them.
     */
    struct Qwen2ModelBuffers
    {
        // Main hidden state (shared across layers)
        TensorBase *current_hidden = nullptr; ///< [batch_size * seq_len, d_model]

        // Output buffer
        TensorBase *logits = nullptr; ///< [batch_size * seq_len, vocab_size]

        // Per-layer activation buffers (from PipelineBase::ActivationBuffers)
        Qwen2ActivationBuffers layer_buffers;
    };

    /**
     * @brief Qwen2-specific model executor
     *
     * Implements buildFullForwardGraph() and related methods for Qwen2 architecture.
     */
    class Qwen2ModelExecutor : public ModelExecutor
    {
    public:
        /**
         * @brief Construct Qwen2ModelExecutor
         *
         * @param model_ctx Model context with GGUF metadata
         * @param mpi_ctx MPI context (nullptr for single-rank)
         * @param config Qwen2-specific configuration
         */
        Qwen2ModelExecutor(std::shared_ptr<ModelContext> model_ctx,
                           std::shared_ptr<MPIContext> mpi_ctx,
                           const Qwen2ModelExecutorConfig &config);

        ~Qwen2ModelExecutor() override = default;

        /**
         * @brief Set weight accessors (called by pipeline)
         *
         * The pipeline owns weights; this provides read access.
         */
        void setWeights(const Qwen2ModelWeights &weights) { weights_ = weights; }

        /**
         * @brief Set activation buffers (called by pipeline)
         *
         * The pipeline allocates buffers; this provides access for execution.
         */
        void setBuffers(const Qwen2ModelBuffers &buffers) { buffers_ = buffers; }

        /**
         * @brief Get Qwen2-specific configuration
         */
        const Qwen2ModelExecutorConfig &qwen2Config() const { return qwen2_config_; }

        // =========================================================================
        // Graph Building (ModelExecutor interface)
        // =========================================================================

        ComputeGraph buildFullForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override;

        ComputeGraph buildEmbeddingGraph(
            const ForwardInput &input,
            TensorBase *output_hidden) override;

        ComputeGraph buildTransformerLayersGraph(
            TensorBase *input_hidden,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx) override;

        ComputeGraph buildLayerGraph(
            int layer_idx,
            TensorBase *input_hidden,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx) override;

        ComputeGraph buildLMHeadGraph(
            TensorBase *hidden_states,
            TensorBase *output_logits,
            int device_idx) override;

        // =========================================================================
        // State Management
        // =========================================================================

        void clearCache() override;

    private:
        Qwen2ModelExecutorConfig qwen2_config_;
        Qwen2ModelWeights weights_;
        Qwen2ModelBuffers buffers_;

        // Layer executor for per-layer graph building
        std::unique_ptr<Qwen2LayerExecutor> layer_executor_;

        // Current execution state
        int current_batch_size_ = 0;
        int current_seq_len_ = 0;
        std::vector<int> position_ids_buffer_;

        /**
         * @brief Build position IDs for RoPE
         */
        std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset);

        /**
         * @brief Add final normalization to graph
         */
        void addFinalNormToGraph(
            ComputeGraph &graph,
            TensorBase *hidden,
            const std::string &prev_node,
            int seq_len,
            int device_idx);
    };

} // namespace llaminar2
