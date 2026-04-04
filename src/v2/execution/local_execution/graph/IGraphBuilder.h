/**
 * @file IGraphBuilder.h
 * @brief Interface for declarative compute graph builders
 * @author David Sanftenberg
 * @date December 19, 2025
 *
 * This interface defines the contract for building compute graphs in a
 * declarative, stateless manner. Model-specific implementations (Qwen2Graph,
 * Qwen3Graph, LlamaGraph, etc.) derive from this interface.
 *
 * Design Principles:
 * - Stateless: Graph builders should not hold mutable state
 * - Declarative: Methods return ComputeGraph objects, not execute them
 * - Testable: Interface enables mock implementations for unit testing
 */

#pragma once

#include "DeviceGraphExecutor.h"
#include "GraphSchema.h"
#include "GraphResolver.h"
#include "../../../models/GraphTypes.h"
#include "../../../backends/DeviceId.h"

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class ICPUKVCache;
    struct LayerWeights;
    struct ActivationBuffers;

    // Forward declarations for IGraphBuilder interface
    struct PipelineConfig;
    class ILocalPPContext;
    class ILocalTPContext;

    // =========================================================================
    // Generic Input/Output Structures
    // =========================================================================

    /**
     * @brief Generic forward pass input
     *
     * Contains all fields needed for forward pass execution including
     * pipeline parallelism and variable-length batching support.
     */
    struct ForwardInput
    {
        const int *token_ids = nullptr;    ///< Token IDs [batch_size * seq_len]
        const int *position_ids = nullptr; ///< Position IDs [batch_size * seq_len]
        int batch_size = 1;                ///< Number of sequences
        int seq_len = 0;                   ///< Sequence length per batch
        int position_offset = 0;           ///< KV cache position offset (legacy fallback)
        DeviceId device = DeviceId::cpu(); ///< Target device
        IKVCache *kv_cache = nullptr;      ///< KV cache (optional)

        // ----- Pipeline Parallelism -----

        /// Per-device KV caches for Pipeline Parallelism (PP).
        /// When set (non-empty), each PP stage uses the KV cache for its device.
        /// Takes precedence over kv_cache when device is found in this map.
        const std::unordered_map<DeviceId, IKVCache *> *pp_kv_caches = nullptr;

        /// External hidden state input (for PP middle stages that don't have embedding).
        /// When set, embedding is skipped and this tensor is used as initial hidden state.
        TensorBase *external_hidden_state = nullptr;

        // ----- Variable-Length Batching -----

        /// Sequence lengths for variable-length batching (nullptr = all equal to seq_len).
        /// When set, enables proper batch-separating attention masks that
        /// prevent cross-sequence attention in batched execution.
        const std::vector<int> *sequence_lengths = nullptr;

        /// Batched input (alternative to token_ids)
        struct Batch
        {
            const int *tokens;
            int len;
            int offset;
        };
        const Batch *batches = nullptr;
        int num_batches = 0;

        // ----- Helpers -----

        /// Get the KV cache for a specific device (PP) or the default (non-PP)
        IKVCache *getKVCacheForDevice(const DeviceId &dev) const
        {
            if (pp_kv_caches && !pp_kv_caches->empty())
            {
                auto it = pp_kv_caches->find(dev);
                if (it != pp_kv_caches->end())
                    return it->second;
            }
            return kv_cache;
        }

        virtual ~ForwardInput() = default;
    };

    /**
     * @brief Generic forward pass output
     */
    struct ForwardOutput
    {
        TensorBase *logits = nullptr; ///< Output logits [batch_size * seq_len, vocab_size]
        TensorBase *hidden = nullptr; ///< Optional: final hidden states

        virtual ~ForwardOutput() = default;
    };

    /**
     * @brief Context for layer-level graph building
     */
    struct LayerContext
    {
        int layer_idx = 0;                 ///< Layer index
        int seq_len = 0;                   ///< Sequence length
        int batch_size = 1;                ///< Batch size (number of sequences)
        DeviceId device = DeviceId::cpu(); ///< Target device
        const int *position_ids = nullptr; ///< Position IDs for RoPE
        IKVCache *kv_cache = nullptr;      ///< KV cache
        /// Sequence lengths for variable-length batching (nullptr = all equal)
        const std::vector<int> *sequence_lengths = nullptr;
    };

    // =========================================================================
    // IGraphBuilder Interface
    // =========================================================================

    /**
     * @brief Interface for declarative compute graph builders
     *
     * This interface defines the contract that all model graph builders must
     * implement. It enables:
     * - Polymorphic graph building across different model architectures
     * - Mock implementations for unit testing
     * - Clear separation between graph building and execution
     *
     * Example usage:
     * @code
     * std::unique_ptr<IGraphBuilder> builder = std::make_unique<Qwen2Graph>(...);
     * ForwardInput input{...};
     * ForwardOutput output{...};
     * ComputeGraph graph = builder->buildForwardGraph(input, output);
     * executor.execute(graph, ctx);
     * @endcode
     */
    class IGraphBuilder
    {
    public:
        virtual ~IGraphBuilder() = default;

        // =====================================================================
        // Core Graph Building Methods
        // =====================================================================

        /**
         * @brief Build complete forward graph
         *
         * Constructs a ComputeGraph representing the full forward pass:
         * embedding → transformer layers → output projection (LM head).
         *
         * @param input Forward pass input parameters
         * @param output Forward pass output tensors (logits, optional hidden)
         * @return Complete forward compute graph
         */
        virtual ComputeGraph buildForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) = 0;

        /**
         * @brief Build single transformer layer graph
         *
         * Constructs a ComputeGraph for one transformer layer (attention + FFN).
         *
         * @param ctx Layer context with index, seq_len, device, etc.
         * @return Single layer compute graph
         */
        virtual ComputeGraph buildLayerGraph(const LayerContext &ctx) = 0;

        // =====================================================================
        // Optional Methods (with default implementations)
        // =====================================================================

        /**
         * @brief Get the number of transformer layers
         *
         * @return Number of layers in the model
         */
        virtual int numLayers() const { return 0; }

        /**
         * @brief Get model hidden dimension
         *
         * @return Hidden dimension (d_model)
         */
        virtual int hiddenDim() const { return 0; }

        /**
         * @brief Check if the builder is properly initialized
         *
         * @return true if weights and buffers are set
         */
        virtual bool isInitialized() const { return false; }

        // =====================================================================
        // Configuration Access
        // =====================================================================

        /// Get model configuration (dimensions, layer count, etc.)
        virtual const GraphConfig &config() const = 0;

        /// Get the architecture name (e.g. "qwen2", "qwen3", "llama")
        virtual std::string architectureName() const { return "unknown"; }

        // =====================================================================
        // Weight / Buffer Management
        // =====================================================================

        /// Set model weights
        virtual void setWeights(const ModelWeights &weights) = 0;

        /// Set activation buffers (for manual buffer management)
        virtual void setBuffers(const ModelBuffers &buffers) = 0;

        /// Set buffer arena for arena-managed allocation
        virtual void setArena(BufferArena *arena) { (void)arena; }

        /// Get current activation buffers
        virtual const ModelBuffers &buffers() const = 0;

        // =====================================================================
        // Schema / Resolver
        // =====================================================================

        /// Get the declarative schema for this architecture
        virtual GraphSchema getSchema() const { return {}; }

        /// Get resolver config for buffer allocation
        virtual GraphResolverConfig getResolverConfig(int seq_len) const
        {
            (void)seq_len;
            return {};
        }

        // =====================================================================
        // Pipeline / Parallelism Configuration
        // =====================================================================

        /// Set pipeline configuration for PP graph building
        virtual void setPipelineConfig(std::shared_ptr<PipelineConfig> config)
        {
            (void)config;
        }

        /// Register a PP context for inter-stage transfers
        virtual void setPPContext(int from_stage, int to_stage, ILocalPPContext *pp_ctx)
        {
            (void)from_stage;
            (void)to_stage;
            (void)pp_ctx;
        }

        /// Register a TP context for a named domain
        virtual void setTPContext(const std::string &domain_name, ILocalTPContext *tp_ctx)
        {
            (void)domain_name;
            (void)tp_ctx;
        }

        // =====================================================================
        // Snapshot
        // =====================================================================

        /// Set snapshot callback for debugging
        virtual void setSnapshotCallback(StageSnapshotCallback callback)
        {
            (void)callback;
        }

        // =====================================================================
        // PP Graph Building Variants
        // =====================================================================

        /// Build forward graph for a PP subset of layers
        virtual ComputeGraph buildPartialForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output,
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head)
        {
            (void)input;
            (void)output;
            (void)first_layer;
            (void)last_layer;
            (void)has_embedding;
            (void)has_lm_head;
            return {};
        }

        /// Build unified PP+TP pipeline graph
        virtual ComputeGraph buildUnifiedPipelineGraph(
            const ForwardInput &input,
            ForwardOutput &output)
        {
            (void)input;
            (void)output;
            return {};
        }

        // =====================================================================
        // Full Forward & Layer-Level Graph Building
        // =====================================================================

        /// Build the complete forward graph (embedding → all layers → LM head)
        virtual ComputeGraph buildFullForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output)
        {
            (void)input;
            (void)output;
            return {};
        }

        /// Build attention block sub-graph for a single layer
        virtual ComputeGraph buildAttentionGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device,
            const std::vector<int> *sequence_lengths = nullptr)
        {
            (void)layer;
            (void)buffers;
            (void)layer_idx;
            (void)seq_len;
            (void)batch_size;
            (void)kv_cache;
            (void)position_ids;
            (void)device;
            (void)sequence_lengths;
            return {};
        }

        /// Build FFN block sub-graph for a single layer
        virtual ComputeGraph buildFFNGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            DeviceId device)
        {
            (void)layer;
            (void)buffers;
            (void)layer_idx;
            (void)seq_len;
            (void)batch_size;
            (void)device;
            return {};
        }

        // =====================================================================
        // Utility Methods
        // =====================================================================

        /**
         * @brief Build position IDs array for RoPE
         *
         * Static utility function that can be used by any graph builder.
         *
         * @param seq_len Sequence length
         * @param batch_size Number of sequences
         * @param offset Position offset (for KV cache continuation)
         * @return Vector of position IDs [batch_size * seq_len]
         */
        static std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset)
        {
            std::vector<int> pos_ids(batch_size * seq_len);
            for (int b = 0; b < batch_size; ++b)
            {
                for (int s = 0; s < seq_len; ++s)
                {
                    pos_ids[b * seq_len + s] = offset + s;
                }
            }
            return pos_ids;
        }
    };

    // =========================================================================
    // MockGraphBuilder for Testing
    // =========================================================================

    /**
     * @brief Mock graph builder for unit testing
     *
     * Provides a controllable mock implementation of IGraphBuilder that:
     * - Records method calls for verification
     * - Returns configurable mock graphs
     * - Enables testing of DeviceGraphOrchestrator without real model weights
     *
     * Example usage:
     * @code
     * auto mock = std::make_shared<MockGraphBuilder>();
     * mock->setMockForwardGraph(some_graph);
     * mock->setNumLayers(24);
     *
     * DeviceGraphOrchestrator orchestrator(mock);
     * orchestrator.executeForward(input, output);
     *
     * EXPECT_EQ(mock->buildForwardGraphCallCount(), 1);
     * @endcode
     */
    class MockGraphBuilder : public IGraphBuilder
    {
    public:
        MockGraphBuilder() = default;
        ~MockGraphBuilder() override = default;

        // =====================================================================
        // IGraphBuilder Implementation
        // =====================================================================

        ComputeGraph buildForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override
        {
            ++build_forward_calls_;
            last_forward_input_ = &input;
            last_forward_output_ = &output;

            if (forward_graph_factory_)
            {
                return forward_graph_factory_(input, output);
            }
            // Return empty graph by default (ComputeGraph is move-only)
            return ComputeGraph{};
        }

        ComputeGraph buildLayerGraph(const LayerContext &ctx) override
        {
            ++build_layer_calls_;
            last_layer_ctx_ = ctx;

            if (layer_graph_factory_)
            {
                return layer_graph_factory_(ctx);
            }

            // Check for layer-specific factory
            if (ctx.layer_idx < static_cast<int>(layer_graph_factories_.size()) &&
                layer_graph_factories_[ctx.layer_idx])
            {
                return layer_graph_factories_[ctx.layer_idx](ctx);
            }

            // Return empty graph by default
            return ComputeGraph{};
        }

        int numLayers() const override { return num_layers_; }
        int hiddenDim() const override { return hidden_dim_; }
        bool isInitialized() const override { return initialized_; }

        // New IGraphBuilder overrides
        const GraphConfig &config() const override { return config_; }
        void setWeights(const ModelWeights &weights) override { weights_ = weights; }
        void setBuffers(const ModelBuffers &buffers) override { buffers_ = buffers; }
        const ModelBuffers &buffers() const override { return buffers_; }

        // =====================================================================
        // Mock Configuration
        // =====================================================================

        /**
         * @brief Set factory function for forward graph
         *
         * The factory will be called each time buildForwardGraph is invoked,
         * allowing dynamic graph creation based on input parameters.
         *
         * @param factory Function that creates a ComputeGraph from ForwardInput
         */
        using ForwardGraphFactory = std::function<ComputeGraph(const ForwardInput &, ForwardOutput &)>;
        void setForwardGraphFactory(ForwardGraphFactory factory)
        {
            forward_graph_factory_ = std::move(factory);
        }

        /**
         * @brief Set factory function for layer graph (default for all layers)
         *
         * @param factory Function that creates a ComputeGraph from LayerContext
         */
        using LayerGraphFactory = std::function<ComputeGraph(const LayerContext &)>;
        void setLayerGraphFactory(LayerGraphFactory factory)
        {
            layer_graph_factory_ = std::move(factory);
        }

        /**
         * @brief Set factory function for a specific layer
         *
         * @param layer_idx Layer index
         * @param factory Function that creates a ComputeGraph for this layer
         */
        void setLayerGraphFactory(int layer_idx, LayerGraphFactory factory)
        {
            if (layer_idx >= static_cast<int>(layer_graph_factories_.size()))
            {
                layer_graph_factories_.resize(layer_idx + 1);
            }
            layer_graph_factories_[layer_idx] = std::move(factory);
        }

        /// Configure mock model properties
        void setNumLayers(int n)
        {
            num_layers_ = n;
            config_.n_layers = n;
        }
        void setHiddenDim(int d)
        {
            hidden_dim_ = d;
            config_.d_model = d;
        }
        void setInitialized(bool init) { initialized_ = init; }
        void setConfig(const GraphConfig &cfg)
        {
            config_ = cfg;
            num_layers_ = cfg.n_layers;
            hidden_dim_ = cfg.d_model;
        }

        // =====================================================================
        // Call Tracking (for test assertions)
        // =====================================================================

        /// Get number of buildForwardGraph calls
        int buildForwardGraphCallCount() const { return build_forward_calls_; }

        /// Get number of buildLayerGraph calls
        int buildLayerGraphCallCount() const { return build_layer_calls_; }

        /// Get last forward input (for inspection)
        const ForwardInput *lastForwardInput() const { return last_forward_input_; }

        /// Get last forward output (for inspection)
        const ForwardOutput *lastForwardOutput() const { return last_forward_output_; }

        /// Get last layer context (for inspection)
        const LayerContext &lastLayerContext() const { return last_layer_ctx_; }

        /// Reset all call counters
        void resetCallCounts()
        {
            build_forward_calls_ = 0;
            build_layer_calls_ = 0;
            last_forward_input_ = nullptr;
            last_forward_output_ = nullptr;
            last_layer_ctx_ = {};
        }

        // =====================================================================
        // Schema / Resolver Support (for buffer management testing)
        // =====================================================================

        /// Override getSchema() to return a custom schema
        GraphSchema getSchema() const override
        {
            if (schema_)
                return *schema_;
            return {};
        }

        /// Override getResolverConfig() to return a custom resolver config
        GraphResolverConfig getResolverConfig(int seq_len) const override
        {
            if (resolver_config_factory_)
                return resolver_config_factory_(seq_len);
            return {};
        }

        /// Set the schema returned by getSchema()
        void setSchema(GraphSchema schema)
        {
            schema_ = std::make_unique<GraphSchema>(std::move(schema));
        }

        /// Set a factory for resolver config (receives seq_len)
        using ResolverConfigFactory = std::function<GraphResolverConfig(int)>;
        void setResolverConfigFactory(ResolverConfigFactory factory)
        {
            resolver_config_factory_ = std::move(factory);
        }

    private:
        // Factory functions for dynamic graph creation
        ForwardGraphFactory forward_graph_factory_;
        LayerGraphFactory layer_graph_factory_;
        std::vector<LayerGraphFactory> layer_graph_factories_;

        // Schema / Resolver support
        std::unique_ptr<GraphSchema> schema_;
        ResolverConfigFactory resolver_config_factory_;

        // Model properties
        GraphConfig config_{};
        ModelWeights weights_{};
        ModelBuffers buffers_{};
        int num_layers_ = 24;
        int hidden_dim_ = 896;
        bool initialized_ = true;

        // Call tracking
        int build_forward_calls_ = 0;
        int build_layer_calls_ = 0;
        const ForwardInput *last_forward_input_ = nullptr;
        const ForwardOutput *last_forward_output_ = nullptr;
        LayerContext last_layer_ctx_;
    };

} // namespace llaminar2
