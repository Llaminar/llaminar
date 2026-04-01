/**
 * @file Qwen2Graph.h
 * @brief Qwen2 compute graph builder for declarative forward pass execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * Qwen2Graph is the graph builder for Qwen2 architecture models.
 *
 * Key Design:
 * - This class BUILDS compute graphs (it's a graph builder, not executor)
 * - The actual execution is delegated to DeviceGraphExecutor
 * - Supports both model-level and layer-level graph construction
 *
 * Graph Building Methods:
 * - buildFullForwardGraph(): Complete embedding → layers → LM head
 * - buildEmbeddingGraph(): Token embedding lookup
 * - buildTransformerLayersGraph(): All transformer layers
 * - buildLayerGraph(): Single transformer layer
 * - buildAttentionGraph(): Attention block within a layer
 * - buildFFNGraph(): FFN block within a layer
 * - buildLMHeadGraph(): Final projection to vocabulary
 */

#pragma once

#include "../GraphTypes.h"
#include "../../execution/local_execution/graph/DeviceGraphExecutor.h"
#include "../../execution/compute_stages/ComputeStages.h"
#include "../../execution/local_execution/device/DeviceContext.h"
#include "../../execution/config/ExecutionPolicy.h"
#include "../../memory/BufferArena.h"
#include "../../execution/local_execution/graph/IGraphBuilder.h"
#include "../../execution/config/RuntimeConfig.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../backends/DeviceId.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/TensorFactory.h"
#include "../../kernels/cpu/CPUKVCache.h"
#include "../../loaders/ModelContext.h"
#include "../../utils/MPIContext.h"
#include "../../config/TensorParallelConfig.h"
#include "../../config/TPDomain.h"
#include "Qwen2Schema.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

namespace llaminar2
{

    // Forward declarations
    class Qwen2Pipeline;

    // Type definitions (GraphConfig, LayerWeights, ModelWeights, ActivationBuffers,
    // ModelBuffers) and backward-compatible aliases (GraphConfig, LayerWeights,
    // etc.) are provided by models/GraphTypes.h, included above.

    // =========================================================================
    // Qwen2Graph Class
    // =========================================================================

    /**
     * @brief Qwen2 compute graph builder
     *
     * Builds ComputeGraph instances for Qwen2 architecture,
     * delegating execution to DeviceGraphExecutor.
     *
     * Implements IGraphBuilder interface for polymorphic graph building
     * and testability via MockGraphBuilder.
     */
    class Qwen2Graph : public IGraphBuilder
    {
    public:
        /**
         * @brief Construct Qwen2Graph with full context
         *
         * @param model_ctx Model context with GGUF metadata
         * @param mpi_ctx MPI context (nullptr for single-rank)
         * @param config Qwen2-specific configuration
         */
        Qwen2Graph(std::shared_ptr<ModelContext> model_ctx,
                   std::shared_ptr<MPIContext> mpi_ctx,
                   const GraphConfig &config);

        /**
         * @brief Construct Qwen2Graph for layer-level operations only
         *
         * This constructor is used when only layer-level graph building is needed,
         * without model-level operations like embedding or LM head.
         * Backward compatible with Qwen2LayerExecutor(config, mpi_ctx).
         *
         * @param config Qwen2-specific configuration
         * @param mpi_ctx MPI context (nullptr for single-rank)
         */
        Qwen2Graph(const GraphConfig &config,
                   std::shared_ptr<MPIContext> mpi_ctx = nullptr);

        ~Qwen2Graph() = default;

        // Non-copyable
        Qwen2Graph(const Qwen2Graph &) = delete;
        Qwen2Graph &operator=(const Qwen2Graph &) = delete;

        // =====================================================================
        // Configuration
        // =====================================================================

        const GraphConfig &config() const override { return config_; }

        std::string architectureName() const override { return "qwen2"; }

        // =====================================================================
        // Unified PP Configuration Mutators (Phase 6)
        // =====================================================================

        /**
         * @brief Set pipeline configuration for unified PP graph
         *
         * Enables buildUnifiedPipelineGraph() to create multi-stage graphs.
         *
         * @param pipeline_config Shared ownership of pipeline configuration
         */
        void setPipelineConfig(std::shared_ptr<PipelineConfig> pipeline_config) override
        {
            config_.pipeline_config = std::move(pipeline_config);
        }

        /**
         * @brief Register a PP context for inter-stage activation transfers
         *
         * Called by DeviceGraphOrchestrator after initializing LocalPPContexts.
         *
         * @param from_stage Source PP stage ID
         * @param to_stage Destination PP stage ID
         * @param pp_ctx Pointer to ILocalPPContext (must remain valid during graph execution)
         */
        void setPPContext(int from_stage, int to_stage, ILocalPPContext *pp_ctx) override
        {
            config_.pp_contexts[{from_stage, to_stage}] = pp_ctx;
        }

        /**
         * @brief Register a TP context for a named domain
         *
         * Called by DeviceGraphOrchestrator after initializing LocalTPContexts.
         *
         * @param domain_name Name of the TP domain
         * @param tp_ctx Pointer to ILocalTPContext (must remain valid during graph execution)
         */
        void setTPContext(const std::string &domain_name, ILocalTPContext *tp_ctx) override
        {
            config_.domain_tp_contexts[domain_name] = tp_ctx;
        }

        /**
         * @brief Set weight accessors (called by pipeline)
         */
        void setWeights(const ModelWeights &weights) override { weights_ = weights; }

        /**
         * @brief Set activation buffers (called by pipeline)
         *
         * Use this for manual buffer management (tests, PP path).
         * Alternative: Use setArena() for arena-managed allocation.
         */
        void setBuffers(const ModelBuffers &buffers) override { buffers_ = buffers; }

        /**
         * @brief Set the BufferArena for arena-managed buffer resolution
         *
         * When set, populates buffers_ from the arena, allowing all graph-building
         * methods to use the arena-allocated tensors via the existing buffers_ paths.
         * This eliminates the bindArenaToManagedBuffers() shim in the orchestrator.
         *
         * The arena pointer is non-owning; the caller (DeviceGraphOrchestrator)
         * owns the arena lifetime.
         *
         * @param arena BufferArena pointer (not owned, must outlive graph builder)
         */
        void setArena(BufferArena *arena) override;

        /// Get the arena (nullptr if not set)
        BufferArena *arena() const { return arena_; }

        /// Get the current buffers (for cache-replay PP copy).
        /// When arena is set, populates a snapshot from arena on demand.
        const ModelBuffers &buffers() const override;

        /**
         * @brief Set TensorFactory for graph-managed buffer allocation
         * @param factory TensorFactory pointer (not owned)
         */
        void setTensorFactory(TensorFactory *factory) { tensor_factory_ = factory; }

        /**
         * @brief Get the declarative schema for this architecture
         *
         * Returns the Qwen2 GraphSchema which defines all buffers, stages,
         * and their relationships declaratively.
         *
         * @return GraphSchema for Qwen2 architecture
         */
        GraphSchema getSchema() const override;

        /**
         * @brief Get resolver config for buffer allocation
         *
         * Creates a GraphResolverConfig populated with this graph's
         * configuration, including tensor-parallel local dimensions.
         * Used by BufferAllocator to resolve buffer shapes.
         *
         * @param seq_len Sequence length for buffer sizing
         * @return GraphResolverConfig with model dimensions
         */
        GraphResolverConfig getResolverConfig(int seq_len) const override;

        // =====================================================================
        // Buffer Management
        // =====================================================================
        // NOTE: Buffer lifecycle management (initializeBuffers, releaseBuffers, etc.)
        // has been moved to DeviceGraphOrchestrator as part of the declarative refactor.
        // Use DeviceGraphOrchestrator::initializeBuffers() for graph-managed allocation.
        // =====================================================================

        /**
         * @brief Set snapshot callback for debugging
         *
         * Note: The callback is stored for use by DeviceGraphOrchestrator when it executes
         * graphs built by this Qwen2Graph.
         */
        void setSnapshotCallback(StageSnapshotCallback callback) override
        {
            snapshot_callback_ = std::move(callback);
            // Note: DeviceGraphOrchestrator will call its own executor_.setSnapshotCallback()
        }

        /**
         * @brief Get the current snapshot callback
         */
        const StageSnapshotCallback &getSnapshotCallback() const { return snapshot_callback_; }

        // =====================================================================
        // IGraphBuilder Interface Implementation
        // =====================================================================

        /**
         * @brief Build complete forward graph (IGraphBuilder interface)
         */
        ComputeGraph buildForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override;

        /**
         * @brief Build single transformer layer graph (IGraphBuilder interface)
         *
         * Uses layer context to build attention + FFN graph.
         */
        ComputeGraph buildLayerGraph(const LayerContext &ctx) override;

        /**
         * @brief Get number of transformer layers
         */
        int numLayers() const override { return config_.n_layers; }

        /**
         * @brief Get model hidden dimension
         */
        int hiddenDim() const override { return config_.d_model; }

        /**
         * @brief Check if the builder is properly initialized
         */
        bool isInitialized() const override
        {
            return weights_.get_layer_weights != nullptr;
        }

        // =====================================================================
        // Model-Level Graph Building
        // =====================================================================

        /**
         * @brief Build complete forward graph (embedding → layers → LM head)
         *
         * LEGACY: This method uses imperative graph building.
         * For new code, prefer buildForwardGraphFromSchema().
         */
        ComputeGraph buildFullForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output) override;

        /**
         * @brief Build partial forward graph for Pipeline Parallelism
         *
         * Builds a compute graph for a subset of the model, enabling PP execution
         * where each stage runs only its assigned components:
         *
         * Stage 0: [embedding] → layers[0, mid) → hidden output
         * Stage 1: hidden input → layers[mid, n) → [LM head] → logits
         *
         * @param input Forward pass input (tokens for embedding stage, hidden for others)
         * @param output Forward pass output (hidden for non-final stages, logits for final)
         * @param first_layer First layer to include (inclusive)
         * @param last_layer Last layer to include (exclusive)
         * @param has_embedding Include embedding lookup stage
         * @param has_lm_head Include final_norm and LM head projection
         * @return ComputeGraph for partial forward pass
         */
        ComputeGraph buildPartialForwardGraph(
            const ForwardInput &input,
            ForwardOutput &output,
            int first_layer,
            int last_layer,
            bool has_embedding,
            bool has_lm_head) override;

        /**
         * @brief Build unified forward graph with PP + TP composition
         *
         * Creates a single ComputeGraph spanning all PP stages with:
         * - Per-layer device assignment based on PP stage → TP domain mapping
         * - TPAllreduceStage nodes within each TP domain
         * - LocalPPTransferStage nodes at PP stage boundaries
         *
         * Requires: config_.pipeline_config must be set and valid
         *
         * @param input Forward pass input
         * @param output Forward pass output (populated)
         * @return ComputeGraph spanning all PP stages
         */
        ComputeGraph buildUnifiedPipelineGraph(
            const ForwardInput &input,
            ForwardOutput &output) override;

        /**
         * @brief Build forward graph using declarative schema
         *
         * This method uses the new three-layer architecture:
         * 1. Qwen2Schema - Declarative graph structure
         * 2. GraphResolver - Evaluates runtime conditionals
         * 3. GraphBuilder - Constructs the ComputeGraph
         *
         * Benefits:
         * - All TP/MPI logic shared with other models
         * - All debugEnv toggling handled uniformly
         * - Graph structure defined declaratively (could be YAML)
         *
         * @param input Forward pass input
         * @param output Forward pass output (modified)
         * @return Constructed ComputeGraph
         */
        ComputeGraph buildForwardGraphFromSchema(
            const ForwardInput &input,
            ForwardOutput &output);

        /**
         * @brief Build embedding lookup graph
         */
        ComputeGraph buildEmbeddingGraph(
            const ForwardInput &input,
            TensorBase *output_hidden);

        /**
         * @brief Build all transformer layers graph
         */
        ComputeGraph buildTransformerLayersGraph(
            TensorBase *input_hidden,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        /**
         * @brief Build single transformer layer graph
         */
        ComputeGraph buildLayerGraph(
            int layer_idx,
            TensorBase *input_hidden,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device);

        /**
         * @brief Build LM head projection graph
         * @param hidden_states Final hidden states from transformer layers
         * @param output_logits Output tensor for full logits [seq_len, vocab_size]
         * @param total_tokens Number of tokens (batch_size * seq_len)
         * @param device_idx Target device
         * @param logits_local Optional local logits buffer for column-parallel LM head
         *                     [seq_len, vocab_local]. When lm_head_column_parallel is
         *                     enabled and this is non-null, the LM head writes to
         *                     logits_local, then AllGather collects to output_logits.
         * @return LM head compute graph
         */
        ComputeGraph buildLMHeadGraph(
            TensorBase *hidden_states,
            TensorBase *output_logits,
            int total_tokens,
            DeviceId device,
            TensorBase *logits_local = nullptr);

        // =====================================================================
        // Layer-Level Graph Building
        // =====================================================================

        /**
         * @brief Build attention block graph
         * @param batch_size Number of sequences in batch (1 for single-sequence)
         * @param sequence_lengths Actual lengths per sequence (nullptr = all equal to seq_len)
         */
        ComputeGraph buildAttentionGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device,
            const std::vector<int> *sequence_lengths = nullptr) override;

        /**
         * @brief Build FFN block graph
         */
        ComputeGraph buildFFNGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            DeviceId device) override;

    private:
        // =====================================================================
        // Configuration
        // =====================================================================
        GraphConfig config_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<MPIContext> mpi_ctx_;

        // TensorFactory for buffer allocation (not owned)
        TensorFactory *tensor_factory_ = nullptr;

        // BufferArena for arena-managed buffer resolution (not owned)
        BufferArena *arena_ = nullptr;

        // Weights and buffers (not owned)
        ModelWeights weights_;
        ModelBuffers buffers_;

        // Snapshot callback
        StageSnapshotCallback snapshot_callback_;

        // =====================================================================
        // Helper: TP Allreduce Stage Creation
        // =====================================================================
        /**
         * @brief Check if TP (LOCAL or GLOBAL) requires allreduce collective
         * @return True if either LOCAL TP (degree > 1) or GLOBAL TP (world_size > 1) is active
         */
        bool needsTPAllreduce() const;

        /**
         * @brief Create an allreduce stage appropriate for the active TP mode
         *
         * Creates TPAllreduceStage for LOCAL TP (single rank, multiple devices)
         * or AllreduceStage for GLOBAL TP (multiple MPI ranks).
         *
         * @param buffer Tensor to reduce
         * @param count Number of elements
         * @param device Target device
         * @param layer_idx Layer index for domain routing
         * @param is_attention True for attention, false for FFN
         * @param stage_name Name for the stage (used for BAR-backed tensor lookup)
         * @return Unique pointer to the allreduce stage
         */
        std::unique_ptr<IComputeStage> createTPAllreduceStage(
            TensorBase *buffer,
            size_t count,
            DeviceId device,
            int layer_idx,
            bool is_attention,
            const std::string &stage_name = "",
            std::optional<BufferId> tensor_buffer_id = std::nullopt) const;

    public:
        /**
         * @brief Build position IDs array for RoPE
         * @param seq_len Sequence length
         * @param batch_size Number of sequences
         * @param offset Position offset (for KV cache continuation)
         * @return Vector of position IDs [batch_size * seq_len]
         *
         * Static utility function that can be used by callers to build position IDs.
         * For batch_size=1: [offset, offset+1, ..., offset+seq_len-1]
         * For batch_size>1: Repeated pattern per batch
         */
        static std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset);

        void addFinalNormToGraph(
            ComputeGraph &graph,
            TensorBase *hidden,
            TensorBase *normalized_out,
            const std::string &prev_node,
            int seq_len,
            DeviceId device);

        /**
         * @brief Get TPDomain for collective operations in a specific layer
         *
         * Queries the MultiDomainTPConfig (if set) to determine which domain
         * should handle collective operations for the given layer.
         *
         * @param layer_idx Layer index (0 to n_layers-1)
         * @param is_attention True for attention Wo allreduce, false for FFN down allreduce
         * @return Pointer to TPDomain, or nullptr if no domain config (legacy MPI path)
         */
        const TPDomain *getDomainForLayer(int layer_idx, bool is_attention) const;
    };

} // namespace llaminar2
