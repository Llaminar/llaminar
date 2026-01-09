/**
 * @file GraphOrchestrator.h
 * @brief Generic orchestrator for compute graph execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements the execution layer for transformer models, separating
 * graph execution concerns from graph definition (IGraphBuilder implementations).
 *
 * Design Philosophy:
 * - Graph Builders (Qwen2Graph, etc.): Declarative, stateless, build ComputeGraph DAGs
 * - GraphOrchestrator: Imperative executor (manages state, caching, device contexts)
 *
 * The orchestrator owns:
 * - GraphExecutor (for DAG execution)
 * - Device context cache (lazy initialization)
 * - Graph cache (decode optimization)
 * - Execution state (position offset tracking)
 *
 * Usage:
 * @code
 * auto graph_builder = std::make_shared<Qwen2Graph>(config, mpi_ctx);
 * GraphOrchestrator orchestrator(graph_builder, mpi_ctx);
 *
 * // Execute a layer
 * orchestrator.executeLayer(weights, buffers, layer_idx, seq_len, kv_cache, pos_ids, device_idx);
 * @endcode
 */

#pragma once

#include "../models/qwen/Qwen2Graph.h"
#include "../backends/DeviceId.h"
#include "IInferenceRunner.h"
#include "GraphExecutor.h"
#include "GraphBufferManager.h"
#include "DeviceContext.h"
#include "compute_stages/ComputeStages.h" // For StageDumpInfo
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>

namespace llaminar2
{

    // Forward declarations
    class MPIContext;
    class ICPUKVCache;

    /**
     * @brief Configuration for graph caching behavior
     */
    struct GraphCacheConfig
    {
        bool enabled = true;         ///< Enable graph caching (Phase 10)
        int decode_seq_len = 1;      ///< Sequence length that triggers decode caching
        bool cache_attention = true; ///< Cache attention graphs
        bool cache_ffn = true;       ///< Cache FFN graphs
    };

    /**
     * @brief Cached graphs for a single transformer layer
     *
     * Stores pre-built attention and FFN graphs for decode mode (seq_len=1).
     * These graphs have stable buffer pointers and only need parameter updates
     * (position offset) between executions.
     */
    struct LayerGraphCache
    {
        std::unique_ptr<ComputeGraph> attention_decode; ///< Cached attention graph for decode
        std::unique_ptr<ComputeGraph> ffn_decode;       ///< Cached FFN graph for decode
        int cached_seq_len = 0;                         ///< Sequence length when cached
        bool valid = false;                             ///< Whether cache entries are valid

        void invalidate()
        {
            attention_decode.reset();
            ffn_decode.reset();
            cached_seq_len = 0;
            valid = false;
        }
    };

    /**
     * @brief Inference state owned by GraphOrchestrator (Phase 5)
     *
     * This struct encapsulates all mutable inference state, allowing the
     * orchestrator to manage state internally rather than requiring the
     * pipeline to pass buffers for each forward call.
     *
     * State includes:
     * - Hidden state buffer (current layer activations)
     * - Logits buffer (output vocabulary scores)
     * - KV cache (attention key/value history)
     * - Position tracking (per-sequence position offsets)
     * - Sequence lengths (for variable-length batches)
     * - Activation buffers (intermediate tensors for attention/FFN)
     */
    struct InferenceState
    {
        // === Core Buffers ===
        std::shared_ptr<TensorBase> hidden; ///< [batch_size * seq_len, d_model]
        std::shared_ptr<TensorBase> logits; ///< [batch_size * seq_len, vocab_size]

        /// Local logits for column-parallel LM head [batch_size * seq_len, vocab_local]
        /// Only allocated when lm_head_column_parallel is enabled
        std::shared_ptr<TensorBase> logits_local;

        // === KV Cache ===
        std::unique_ptr<ICPUKVCache> kv_cache; ///< Attention KV history

        // === Position Tracking ===
        std::vector<int> positions;        ///< Per-sequence position offset
        std::vector<int> sequence_lengths; ///< Per-sequence length (for padding)

        // === Activation Buffers (shared with Qwen2ActivationBuffers) ===
        std::shared_ptr<TensorBase> normalized;
        std::shared_ptr<TensorBase> residual;
        std::shared_ptr<TensorBase> Q;
        std::shared_ptr<TensorBase> K;
        std::shared_ptr<TensorBase> V;
        std::shared_ptr<TensorBase> attn_output;
        std::shared_ptr<TensorBase> attn_proj;
        std::shared_ptr<TensorBase> gate;
        std::shared_ptr<TensorBase> up;
        std::shared_ptr<TensorBase> ffn_output;

        // === Hybrid Mode Buffers ===
        /// FP32 Q after RoPE (Hybrid mode only - avoids requantization)
        std::shared_ptr<TensorBase> Q_rope;
        /// FP32 K after RoPE (Hybrid mode only - avoids requantization)
        std::shared_ptr<TensorBase> K_rope;
        /// FP32 V dequantized (Hybrid mode only - for KV cache append when V is Q8_1)
        std::shared_ptr<TensorBase> V_dequant;

        // === HybridQ16 K Precision Fix Buffers ===
        /// Per-head dynamic scales for K vectors from RoPE Q16→Q16 path
        /// Shape: [batch_size * max_seq_len * n_kv_heads]
        /// Only used when GEMM outputs K as Q16_1 (K precision fix mode)
        std::vector<float> K_head_scales;

        // === Attention Workspace ===
        std::shared_ptr<TensorBase> workspace_scores;
        std::shared_ptr<TensorBase> workspace_context;
        std::shared_ptr<TensorBase> workspace_mask;

        // === Snapshot Buffers (for E2E debugging) ===
        /// Optional buffer to capture attention context before Wo projection
        /// Allocated when ENABLE_PIPELINE_SNAPSHOTS is defined
        std::shared_ptr<TensorBase> context_snapshot;

        /// Optional buffer to capture attention output (Wo projection, before residual)
        /// Shape: [batch_size * max_seq_len, d_model] - corresponds to ATTENTION_OUTPUT
        std::shared_ptr<TensorBase> attention_output_snapshot;

        /// Optional buffer to capture attention residual (after residual add)
        /// Shape: [batch_size * max_seq_len, d_model] - corresponds to ATTENTION_RESIDUAL
        std::shared_ptr<TensorBase> attention_residual_snapshot;

        // === Configuration ===
        int batch_size = 0;
        int max_seq_len = 0;
        int d_model = 0;
        int vocab_size = 0;
        DeviceId device_id = DeviceId::cpu();

        /**
         * @brief Check if state is initialized
         */
        bool isInitialized() const
        {
            return hidden != nullptr && logits != nullptr && batch_size > 0;
        }

        /**
         * @brief Clear state (reset positions, clear KV cache)
         */
        void clear()
        {
            if (kv_cache)
                kv_cache->clear();
            std::fill(positions.begin(), positions.end(), 0);
            std::fill(sequence_lengths.begin(), sequence_lengths.end(), 0);
        }
    };

    /**
     * @brief Generic orchestrator for compute graph execution
     *
     * Separates execution concerns from graph definition, implementing:
     * - Graph execution via GraphExecutor
     * - Device context management with lazy initialization
     * - Graph caching for decode mode
     * - Execution state tracking
     *
     * This class is the imperative counterpart to declarative graph builders.
     * Currently supports Qwen2Graph, designed for extension to other architectures.
     *
     * Implements IInferenceRunner for unified inference API.
     */
    class GraphOrchestrator : public IInferenceRunner
    {
    public:
        /**
         * @brief Construct orchestrator with graph builder
         *
         * @param graph_builder Shared pointer to Qwen2Graph (graph definition)
         * @param mpi_ctx MPI context for distributed execution
         * @param cache_config Graph caching configuration
         */
        GraphOrchestrator(
            std::shared_ptr<Qwen2Graph> graph_builder,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            const GraphCacheConfig &cache_config = {});

        /**
         * @brief Construct orchestrator with graph config (creates internal graph builder)
         *
         * @param graph_config Configuration for Qwen2Graph
         * @param mpi_ctx MPI context for distributed execution
         * @param cache_config Graph caching configuration
         */
        GraphOrchestrator(
            const Qwen2GraphConfig &graph_config,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            const GraphCacheConfig &cache_config = {});

        ~GraphOrchestrator() = default;

        // Non-copyable, movable
        GraphOrchestrator(const GraphOrchestrator &) = delete;
        GraphOrchestrator &operator=(const GraphOrchestrator &) = delete;
        GraphOrchestrator(GraphOrchestrator &&) = default;
        GraphOrchestrator &operator=(GraphOrchestrator &&) = default;

        // =========================================================================
        // Execution Methods (moved from Qwen2Graph)
        // =========================================================================

        /**
         * @brief Execute full forward pass
         *
         * Builds and executes the complete forward graph including:
         * - Embedding lookup
         * - All transformer layers
         * - Final normalization
         * - LM head projection
         *
         * @param input Forward pass input (tokens, sequence info)
         * @param output Forward pass output (logits buffer)
         * @return true if execution succeeded
         */
        bool executeForward(
            const Qwen2ForwardInput &input,
            Qwen2ForwardOutput &output);

        /**
         * @brief Execute attention block for a single layer
         *
         * Builds and executes attention graph:
         * - Pre-attention RMSNorm
         * - Q/K/V projections
         * - RoPE application
         * - Attention computation with KV cache
         * - Output projection
         * - Residual connection
         *
         * Uses cached graph for decode mode (seq_len=1) when enabled.
         *
         * @param layer Layer weights
         * @param buffers Activation buffers
         * @param layer_idx Layer index
         * @param seq_len Sequence length
         * @param kv_cache KV cache for attention
         * @param position_ids Position IDs for RoPE
         * @param device_idx Target device
         * @return true if execution succeeded
         */
        bool executeAttention(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            ICPUKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        /**
         * @brief Execute FFN block for a single layer
         *
         * Builds and executes FFN graph:
         * - Pre-FFN RMSNorm
         * - Gate and Up projections
         * - SwiGLU activation
         * - Down projection
         * - Residual connection
         *
         * Uses cached graph for decode mode (seq_len=1) when enabled.
         *
         * @param layer Layer weights
         * @param buffers Activation buffers
         * @param layer_idx Layer index
         * @param seq_len Sequence length
         * @param device Target device
         * @return true if execution succeeded
         */
        bool executeFFN(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            DeviceId device);

        /**
         * @brief Execute complete transformer layer (attention + FFN)
         *
         * Convenience method that executes both attention and FFN blocks.
         *
         * @param layer Layer weights
         * @param buffers Activation buffers
         * @param layer_idx Layer index
         * @param seq_len Sequence length
         * @param kv_cache KV cache for attention
         * @param position_ids Position IDs for RoPE
         * @param device Target device
         * @return true if execution succeeded
         */
        bool executeLayer(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            ICPUKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        /**
         * @brief Execute a pre-built compute graph
         *
         * Low-level method for executing arbitrary graphs.
         *
         * @param graph ComputeGraph to execute
         * @param ctx Device context for execution
         * @return true if execution succeeded
         */
        bool execute(ComputeGraph &graph, IDeviceContext *ctx);

        // =========================================================================
        // Cache Management
        // =========================================================================

        /**
         * @brief Clear all caches (graphs, device contexts, state)
         */
        void clearCache();

        /**
         * @brief Invalidate graph cache for a specific layer
         *
         * @param layer_idx Layer index to invalidate (-1 for all)
         */
        void invalidateGraphCache(int layer_idx = -1);

        /**
         * @brief Check if a cached graph exists for a layer
         *
         * @param layer_idx Layer index
         * @param is_attention true for attention, false for FFN
         * @return true if valid cached graph exists
         */
        bool hasValidCachedGraph(int layer_idx, bool is_attention) const;

        /**
         * @brief Enable or disable graph caching
         *
         * @param enabled Whether caching should be enabled
         */
        void setGraphCachingEnabled(bool enabled);

        /**
         * @brief Check if graph caching is enabled
         */
        bool isGraphCachingEnabled() const { return cache_config_.enabled; }

        /**
         * @brief Initialize graph cache for n_layers
         *
         * Must be called before caching can be used.
         *
         * @param n_layers Number of transformer layers
         */
        void initializeGraphCache(int n_layers);

        // =========================================================================
        // Weight and Buffer Configuration
        // =========================================================================

        /**
         * @brief Set model weights for full forward pass
         *
         * Must be called before executeForward() to enable embedding lookup,
         * final normalization, and LM head projection.
         *
         * @param weights Model weights including embedding_table, final_norm, lm_head
         */
        void setWeights(const Qwen2ModelWeights &weights);

        /**
         * @brief Set activation buffers for full forward pass
         *
         * @param buffers Model buffers including current_hidden, logits, layer_buffers
         */
        void setBuffers(const Qwen2ModelBuffers &buffers);

        /**
         * @brief Check if weights are configured for full forward
         */
        bool hasGlobalWeights() const;

        // =========================================================================
        // Graph Buffer Management (Phase 3 - moved from Qwen2Graph)
        // =========================================================================

        /**
         * @brief Set TensorFactory for graph-managed buffer allocation
         * @param factory TensorFactory pointer (not owned)
         */
        void setTensorFactory(TensorFactory *factory) { tensor_factory_ = factory; }

        /**
         * @brief Get TensorFactory
         * @return TensorFactory pointer (nullptr if not set)
         */
        TensorFactory *tensorFactory() const { return tensor_factory_; }

        /**
         * @brief Initialize activation buffers using GraphBufferManager
         *
         * Allocates all activation buffers with automatic aliasing optimization
         * for SCRATCH buffers. This is an alternative to manual buffer allocation.
         *
         * @param seq_len Maximum sequence length for buffer allocation
         * @return true if allocation successful
         */
        bool initializeBuffers(int seq_len);

        /**
         * @brief Release all graph-managed buffers
         *
         * Call this when buffers are no longer needed to free memory.
         */
        void releaseBuffers();

        /**
         * @brief Check if graph buffer management is active
         */
        bool hasGraphManagedBuffers() const { return buffer_manager_ != nullptr; }

        /**
         * @brief Get internal activation buffers (for graph-managed mode)
         *
         * When using graph-managed buffers, the pipeline should use these
         * instead of creating its own buffer mappings.
         *
         * @return Reference to internal activation buffers
         */
        Qwen2ActivationBuffers &getInternalBuffers();
        const Qwen2ActivationBuffers &getInternalBuffers() const;

        /**
         * @brief Get model-level buffers (current_hidden, logits)
         *
         * When using graph-managed buffers, these are allocated by the orchestrator.
         *
         * @return Reference to model buffers
         */
        const Qwen2ModelBuffers &getModelBuffers() const;

        /**
         * @brief Get buffer manager statistics
         *
         * @return BufferAllocationStats or nullptr if not using graph buffer management
         */
        const BufferAllocationStats *bufferStats() const;

        // =========================================================================
        // Inference State Management (Phase 5)
        // =========================================================================

        /**
         * @brief Initialize inference state owned by orchestrator
         *
         * Allocates all buffers needed for inference. After calling this,
         * the simplified forward() API can be used without passing buffers.
         *
         * @param config Configuration specifying dimensions
         * @param batch_size Maximum batch size
         * @param max_seq_len Maximum sequence length
         * @param device_idx Device for buffer allocation
         * @return true if initialization succeeded
         */
        bool initializeInferenceState(
            int batch_size,
            int max_seq_len,
            DeviceId device_id = DeviceId::cpu());

        /**
         * @brief Check if inference state is initialized
         */
        bool hasInferenceState() const { return state_.isInitialized(); }

        /**
         * @brief Get inference state (read-only)
         */
        const InferenceState &inferenceState() const { return state_; }

        /**
         * @brief Simplified forward pass using orchestrator-owned state
         *
         * This is the high-level API for inference. The orchestrator manages
         * all buffers and state internally.
         *
         * @param tokens Token IDs [batch_size * seq_len]
         * @param seq_len Sequence length per batch item
         * @param batch_size Number of sequences (default 1)
         * @return Pointer to logits buffer, or nullptr on failure
         */
        const float *forward(
            const int *tokens,
            int seq_len,
            int batch_size = 1);

        /**
         * @brief Get logits from last forward pass
         *
         * @return Pointer to logits buffer, or nullptr if not available
         */
        const float *logits() const override;

        /**
         * @brief Get current position offset for a sequence
         *
         * @param seq_idx Sequence index (default 0)
         * @return Current position offset
         */
        int getPosition(int seq_idx = 0) const;

        /**
         * @brief Clear inference state (reset positions, clear KV cache)
         */
        void clearInferenceState();

        // =========================================================================
        // Accessors
        // =========================================================================

        /**
         * @brief Get the underlying graph builder
         */
        Qwen2Graph *graphBuilder() { return graph_builder_.get(); }
        const Qwen2Graph *graphBuilder() const { return graph_builder_.get(); }

        /**
         * @brief Get the underlying executor
         */
        GraphExecutor &executor() { return executor_; }
        const GraphExecutor &executor() const { return executor_; }

        /**
         * @brief Get device context for a device (creates if needed)
         *
         * @param device Device identifier
         * @return Device context pointer (owned by orchestrator)
         */
        IDeviceContext *getDeviceContext(DeviceId device);

        // =========================================================================
        // IInferenceRunner Interface Implementation
        // =========================================================================

        /**
         * @brief Run forward pass (IInferenceRunner override)
         */
        bool forward(const int *tokens, int seq_len) override
        {
            return forward(tokens, seq_len, 1) != nullptr;
        }

        /**
         * @brief Get logits (IInferenceRunner override - already declared above)
         */
        // const float *logits() const; - declared above

        /**
         * @brief Get vocabulary size (IInferenceRunner override)
         */
        int vocab_size() const override { return graph_builder_ ? graph_builder_->config().vocab_size : 0; }

        /**
         * @brief Clear KV cache (IInferenceRunner override)
         */
        void clear_cache() override { clearInferenceState(); }

        /**
         * @brief Get current position (IInferenceRunner override)
         */
        int get_position() const override { return getPosition(0); }

        /**
         * @brief Get execution path (always GRAPH)
         */
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }

        /**
         * @brief Get architecture name
         */
        const char *architecture() const override { return "qwen2"; }

        // =========================================================================
        // Batch Interface (IInferenceRunner overrides)
        // =========================================================================

        /**
         * @brief Batched forward pass with variable-length sequences
         *
         * @param token_batches Vector of token sequences
         * @return true if forward pass succeeded
         */
        bool forward_batch(const std::vector<std::vector<int>> &token_batches) override;

        /**
         * @brief Get logits for a specific sequence in batch
         *
         * @param seq_idx Sequence index in batch (default=0)
         * @return Pointer to logits [padded_seq_len, vocab_size], or nullptr
         */
        const float *getLogits(int seq_idx = 0) const override;

        /**
         * @brief Get current batch size
         */
        int batch_size() const override { return state_.batch_size; }

        /**
         * @brief Get padded sequence length for current batch
         */
        int padded_seq_len() const override { return padded_seq_len_; }

        /**
         * @brief Get sequence lengths for current batch
         */
        const std::vector<int> &sequence_lengths() const override { return state_.sequence_lengths; }

        // =========================================================================
        // Snapshot Capture API (Pipeline-compatible for E2E testing)
        // =========================================================================

        /**
         * @brief Enable snapshot capture of intermediate activations
         *
         * When enabled, orchestrator stores copies of intermediate tensors from
         * each stage execution. Used for parity testing against reference implementations.
         *
         * @param output_dir Optional directory to save snapshots (currently unused)
         */
        void enableSnapshotCapture(const std::string &output_dir = "") override
        {
            (void)output_dir; // Reserved for future disk output
            snapshots_.clear();
            snapshot_enabled_ = true;

            executor_.setSnapshotCallback(
                [this](const std::string &name, const StageDumpInfo &dump)
                {
                    LOG_TRACE("[Snapshot] Callback invoked for stage: " << name << " outputs.size=" << dump.outputs.size());
                    // Handle fused QKV stage specially - split into separate Q, K, V snapshots
                    // The fused stage outputs are ordered: Q (output_q), K (output_k), V (output_v)
                    if (name.find("_qkv_proj") != std::string::npos)
                    {
                        // Extract layer prefix (e.g., "layer0" from "layer0_qkv_proj")
                        size_t qkv_pos = name.find("_qkv_proj");
                        std::string prefix = name.substr(0, qkv_pos);

                        // Store Q, K, V separately with pipeline-compatible keys
                        if (dump.outputs.size() >= 3)
                        {
                            // Output 0 = Q
                            if (dump.outputs[0].data)
                            {
                                const auto &out = dump.outputs[0];
                                size_t count = out.rows * out.cols;
                                std::vector<float> data(count);
                                std::memcpy(data.data(), out.data, count * sizeof(float));
                                snapshots_[prefix + "_Q_PROJECTION"] = std::move(data);
                            }
                            // Output 1 = K
                            if (dump.outputs[1].data)
                            {
                                const auto &out = dump.outputs[1];
                                size_t count = out.rows * out.cols;
                                std::vector<float> data(count);
                                std::memcpy(data.data(), out.data, count * sizeof(float));
                                snapshots_[prefix + "_K_PROJECTION"] = std::move(data);
                            }
                            // Output 2 = V
                            if (dump.outputs[2].data)
                            {
                                const auto &out = dump.outputs[2];
                                size_t count = out.rows * out.cols;
                                std::vector<float> data(count);
                                std::memcpy(data.data(), out.data, count * sizeof(float));
                                snapshots_[prefix + "_V_PROJECTION"] = std::move(data);
                            }
                        }
                        return;
                    }

                    // Handle fused Gate/Up stage specially - split into separate GATE and UP snapshots
                    if (name.find("_gate_up") != std::string::npos)
                    {
                        size_t pos = name.find("_gate_up");
                        std::string prefix = name.substr(0, pos);

                        if (dump.outputs.size() >= 2)
                        {
                            // Output 0 = gate
                            if (dump.outputs[0].data)
                            {
                                const auto &out = dump.outputs[0];
                                size_t count = out.rows * out.cols;
                                std::vector<float> data(count);
                                std::memcpy(data.data(), out.data, count * sizeof(float));
                                snapshots_[prefix + "_FFN_GATE"] = std::move(data);
                            }
                            // Output 1 = up
                            if (dump.outputs[1].data)
                            {
                                const auto &out = dump.outputs[1];
                                size_t count = out.rows * out.cols;
                                std::vector<float> data(count);
                                std::memcpy(data.data(), out.data, count * sizeof(float));
                                snapshots_[prefix + "_FFN_UP"] = std::move(data);
                            }
                        }
                        return;
                    }

                    // Handle RoPE stage - captures Q_ROPE and K_ROPE
                    if (name.find("_rope") != std::string::npos && name.find("_q_rope") == std::string::npos && name.find("_k_rope") == std::string::npos)
                    {
                        // Generic "_rope" stage (fused Q/K RoPE)
                        size_t pos = name.find("_rope");
                        std::string prefix = name.substr(0, pos);

                        // RoPE stage outputs are Q and K after RoPE application
                        if (dump.outputs.size() >= 2)
                        {
                            if (dump.outputs[0].data)
                            {
                                const auto &out = dump.outputs[0];
                                size_t count = out.rows * out.cols;
                                std::vector<float> data(count);
                                std::memcpy(data.data(), out.data, count * sizeof(float));
                                snapshots_[prefix + "_Q_ROPE"] = std::move(data);
                            }
                            if (dump.outputs[1].data)
                            {
                                const auto &out = dump.outputs[1];
                                size_t count = out.rows * out.cols;
                                std::vector<float> data(count);
                                std::memcpy(data.data(), out.data, count * sizeof(float));
                                snapshots_[prefix + "_K_ROPE"] = std::move(data);
                            }
                        }
                        return;
                    }

                    // Handle fused attention+Wo stage - captures ATTENTION_CONTEXT, ATTENTION_OUTPUT, and ATTENTION_RESIDUAL
                    // FusedAttentionWoStage::getDumpInfo() outputs are named (in order):
                    //   "context" (ATTENTION_CONTEXT) - pre-Wo attention output
                    //   "attention_output" (ATTENTION_OUTPUT) - post-Wo projection, before residual add
                    //   "attention_residual" (ATTENTION_RESIDUAL) - after residual add
                    //   "output" - primary output tensor (may be Q16_1 for HybridQ16)
                    // The outputs vector preserves this order for non-null snapshots.
                    if (name.find("_fused_attn_wo") != std::string::npos)
                    {
                        size_t pos = name.find("_fused_attn_wo");
                        std::string prefix = name.substr(0, pos);

                        LOG_TRACE("[Snapshot] fused_attn_wo handler: prefix=" << prefix
                                                                              << " dump.outputs.size()=" << dump.outputs.size());

                        // Iterate through outputs and map by name to snapshot keys
                        for (size_t i = 0; i < dump.outputs.size(); ++i)
                        {
                            const auto &out = dump.outputs[i];
                            if (!out.data || !out.name)
                                continue;

                            std::string key;
                            std::string out_name(out.name);

                            // Match by name from StageDumpInfo::OutputBuffer
                            if (out_name == "context")
                            {
                                key = prefix + "_ATTENTION_CONTEXT";
                            }
                            else if (out_name == "attention_output")
                            {
                                key = prefix + "_ATTENTION_OUTPUT";
                            }
                            else if (out_name == "attention_residual")
                            {
                                key = prefix + "_ATTENTION_RESIDUAL";
                            }
                            else if (out_name == "output")
                            {
                                // Skip primary output - it's handled by residual stage or is Q16_1
                                continue;
                            }
                            else
                            {
                                LOG_WARN("[Snapshot] fused_attn_wo unknown output name: " << out_name);
                                continue;
                            }

                            size_t count = out.rows * out.cols;
                            LOG_TRACE("[Snapshot] Storing " << key << ": rows=" << out.rows
                                                            << " cols=" << out.cols << " count=" << count);
                            std::vector<float> data(count);
                            std::memcpy(data.data(), out.data, count * sizeof(float));
                            snapshots_[key] = std::move(data);
                        }
                        return;
                    }

                    // Standard single-output stages
                    LOG_DEBUG("[Snapshot] Standard path: stage=" << name
                                                                 << " outputs.size=" << dump.outputs.size()
                                                                 << " out[0].data=" << (dump.outputs.empty() ? nullptr : dump.outputs[0].data));
                    if (!dump.outputs.empty() && dump.outputs[0].data)
                    {
                        const auto &out = dump.outputs[0];
                        size_t count = out.rows * out.cols;
                        std::vector<float> data(count);
                        std::memcpy(data.data(), out.data, count * sizeof(float));

                        // Convert graph stage name to pipeline-style key
                        std::string key = convertStageNameToSnapshotKey(name);
                        LOG_DEBUG("[Snapshot] Storing key=" << key << " count=" << count);
                        snapshots_[key] = std::move(data);
                    }
                });
        }

        /**
         * @brief Disable snapshot capture and clear stored snapshots
         */
        void disableSnapshotCapture() override
        {
            snapshot_enabled_ = false;
            snapshots_.clear();
            executor_.setSnapshotCallback(nullptr);
        }

        /**
         * @brief Clear stored snapshots but keep capture enabled
         */
        void clearSnapshots() override
        {
            snapshots_.clear();
        }

        /**
         * @brief Retrieve a captured snapshot by key
         *
         * @param key Snapshot identifier (e.g., "layer0_Q_PROJECTION", "EMBEDDING")
         * @param out_size Output parameter for tensor size (number of float elements)
         * @return Pointer to snapshot data, or nullptr if key doesn't exist
         */
        const float *getSnapshot(const std::string &key, size_t &out_size) const override
        {
            auto it = snapshots_.find(key);
            if (it == snapshots_.end())
            {
                LOG_DEBUG("[GraphOrchestrator::getSnapshot] Key NOT FOUND: " << key
                                                                             << " (have " << snapshots_.size() << " snapshots)");
                out_size = 0;
                return nullptr;
            }
            out_size = it->second.size();
            LOG_TRACE("[GraphOrchestrator::getSnapshot] Key found: " << key << " size=" << out_size);
            return it->second.data();
        }

        /**
         * @brief Get list of all captured snapshot keys
         *
         * @return Vector of snapshot identifiers
         */
        std::vector<std::string> getSnapshotKeys() const override
        {
            std::vector<std::string> keys;
            keys.reserve(snapshots_.size());
            for (const auto &p : snapshots_)
            {
                keys.push_back(p.first);
            }
            return keys;
        }

        /**
         * @brief Check if snapshot capture is enabled
         */
        bool isSnapshotCaptureEnabled() const { return snapshot_enabled_; }

        /**
         * @brief Convert graph stage name to pipeline-style snapshot key
         *
         * Graph stages use snake_case (e.g., "layer0_q_proj")
         * Pipeline snapshots use SCREAMING_CASE (e.g., "layer0_Q_PROJECTION")
         *
         * This is public to allow unit testing of the conversion logic.
         *
         * @param stage_name Graph stage name
         * @return Pipeline-compatible snapshot key
         */
        static std::string convertStageNameToSnapshotKey(const std::string &stage_name)
        {
            // Stage name mappings (graph → pipeline)
            static const std::unordered_map<std::string, std::string> suffix_map = {
                {"_attn_norm", "_ATTENTION_NORM"},
                {"_q_proj", "_Q_PROJECTION"},
                {"_k_proj", "_K_PROJECTION"},
                {"_v_proj", "_V_PROJECTION"},
                {"_q_rope", "_Q_ROPE"},
                {"_k_rope", "_K_ROPE"},
                {"_attention", "_ATTENTION_CONTEXT"}, // Graph uses "attention" not "attn_compute"
                {"_wo_proj", "_ATTENTION_OUTPUT"},    // Graph uses "wo_proj" not "attn_proj"
                {"_attn_residual", "_ATTENTION_RESIDUAL"},
                {"_ffn_norm", "_FFN_NORM"},
                {"_ffn_gate", "_FFN_GATE"},
                {"_ffn_up", "_FFN_UP"},
                {"_swiglu", "_FFN_SWIGLU"},
                {"_down_proj", "_FFN_DOWN"}, // Graph uses "down_proj" not "ffn_down"
                {"_ffn_residual", "_FFN_RESIDUAL"},
            };

            // Global stages
            if (stage_name == "embedding")
                return "EMBEDDING";
            if (stage_name == "final_norm")
                return "FINAL_NORM";
            if (stage_name == "lm_head")
                return "LM_HEAD";

            // Layer-specific stages: extract layer prefix and convert suffix
            for (const auto &[suffix, replacement] : suffix_map)
            {
                size_t pos = stage_name.find(suffix);
                if (pos != std::string::npos)
                {
                    // Extract "layerN" prefix
                    std::string prefix = stage_name.substr(0, pos);
                    return prefix + replacement;
                }
            }

            // Fallback: return original name (uppercase)
            std::string result = stage_name;
            for (char &c : result)
            {
                if (c >= 'a' && c <= 'z')
                    c = c - 'a' + 'A';
                if (c == '_')
                    c = '_'; // Keep underscores
            }
            return result;
        }

        // =========================================================================
        // Model Metadata Accessors (Convenience)
        // =========================================================================

        /**
         * @brief Get model hidden dimension
         */
        int d_model() const { return graph_builder_ ? graph_builder_->config().d_model : 0; }

        /**
         * @brief Get number of transformer layers
         */
        int n_layers() const { return graph_builder_ ? graph_builder_->config().n_layers : 0; }

        /**
         * @brief Get maximum sequence length (from config)
         */
        int max_seq_len() const { return graph_builder_ ? graph_builder_->config().max_seq_len : 0; }

        /**
         * @brief Get number of attention heads
         */
        int n_heads() const { return graph_builder_ ? graph_builder_->config().n_heads : 0; }

        /**
         * @brief Get number of KV heads (GQA)
         */
        int n_kv_heads() const { return graph_builder_ ? graph_builder_->config().n_kv_heads : 0; }

        /**
         * @brief Get cache statistics
         */
        struct CacheStats
        {
            size_t attention_cache_hits = 0;
            size_t attention_cache_misses = 0;
            size_t ffn_cache_hits = 0;
            size_t ffn_cache_misses = 0;
            size_t cached_layers = 0;
        };

        CacheStats getCacheStats() const { return cache_stats_; }

    private:
        // =========================================================================
        // Private Helpers
        // =========================================================================

        /**
         * @brief Update dynamic parameters in a cached graph
         *
         * Updates position offset and sequence length in all stages
         * that have dynamic parameters.
         *
         * @param graph Graph to update
         * @param pos_offset New position offset
         * @param seq_len New sequence length
         */
        void updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len);

        /**
         * @brief Check if we can use cached graph for current execution
         *
         * @param layer_idx Layer index
         * @param seq_len Current sequence length
         * @return true if cached graph can be reused
         */
        bool canUseCachedGraph(int layer_idx, int seq_len) const;

        // =========================================================================
        // Members
        // =========================================================================

        /// Graph builder (declarative layer)
        std::shared_ptr<Qwen2Graph> graph_builder_;

        /// Graph executor
        GraphExecutor executor_;

        /// MPI context for distributed execution
        std::shared_ptr<MPIContext> mpi_ctx_;

        /// Graph caching configuration
        GraphCacheConfig cache_config_;

        /// Per-layer graph cache
        std::vector<LayerGraphCache> layer_graph_cache_;

        /// Device context cache (lazy initialization)
        std::unordered_map<int, std::unique_ptr<IDeviceContext>> device_contexts_;

        /// Cache statistics
        mutable CacheStats cache_stats_;

        /// Last position offset (for cache validation)
        int last_pos_offset_ = -1;

        /// Inference state (Phase 5 - owned buffers)
        InferenceState state_;

        /// Padded sequence length from last forward_batch() call
        int padded_seq_len_ = 0;

        // =========================================================================
        // Snapshot Capture Members
        // =========================================================================

        /// Whether snapshot capture is enabled
        bool snapshot_enabled_ = false;

        /// Captured snapshots (key -> FP32 data)
        std::unordered_map<std::string, std::vector<float>> snapshots_;

        // =========================================================================
        // Graph Buffer Management Members (Phase 3 - moved from Qwen2Graph)
        // =========================================================================

        /// TensorFactory for buffer allocation (not owned)
        TensorFactory *tensor_factory_ = nullptr;

        /// Buffer manager for graph-managed allocation (nullptr if using manual buffers)
        std::unique_ptr<GraphBufferManager> buffer_manager_;

        /// Owned tensors when using graph-managed allocation
        std::vector<std::unique_ptr<TensorBase>> owned_buffers_;

        /// Model-level buffers (when using graph-managed allocation)
        Qwen2ModelBuffers managed_buffers_;

        /**
         * @brief Populate managed_buffers_ from graph-managed allocations
         * @param seq_len Sequence length for buffer sizing
         */
        void bindGraphManagedBuffers(int seq_len);
    };

} // namespace llaminar2
