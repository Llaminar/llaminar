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

#include "GraphExecutor.h"
#include "../backends/DeviceId.h"

namespace llaminar2
{

    // Forward declarations
    class CPUTensorBase;
    using TensorBase = CPUTensorBase; // Backward compatibility alias
    class ICPUKVCache;
    struct Qwen2LayerWeights;
    struct Qwen2ActivationBuffers;

    // =========================================================================
    // Generic Input/Output Structures
    // =========================================================================

    /**
     * @brief Generic forward pass input
     *
     * Base structure for forward pass inputs. Model-specific implementations
     * may extend this with additional fields.
     */
    struct ForwardInput
    {
        const int *token_ids = nullptr;    ///< Token IDs [batch_size * seq_len]
        const int *position_ids = nullptr; ///< Position IDs [batch_size * seq_len]
        int batch_size = 1;                ///< Number of sequences
        int seq_len = 0;                   ///< Sequence length per batch
        int position_offset = 0;           ///< KV cache position offset (legacy fallback)
        DeviceId device = DeviceId::cpu(); ///< Target device
        ICPUKVCache *kv_cache = nullptr;   ///< KV cache (optional)

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
        ICPUKVCache *kv_cache = nullptr;   ///< KV cache
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
     * - Enables testing of GraphOrchestrator without real model weights
     *
     * Example usage:
     * @code
     * auto mock = std::make_shared<MockGraphBuilder>();
     * mock->setMockForwardGraph(some_graph);
     * mock->setNumLayers(24);
     *
     * GraphOrchestrator orchestrator(mock);
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
        void setNumLayers(int n) { num_layers_ = n; }
        void setHiddenDim(int d) { hidden_dim_ = d; }
        void setInitialized(bool init) { initialized_ = init; }

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

    private:
        // Factory functions for dynamic graph creation
        ForwardGraphFactory forward_graph_factory_;
        LayerGraphFactory layer_graph_factory_;
        std::vector<LayerGraphFactory> layer_graph_factories_;

        // Model properties
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
