/**
 * @file GraphOrchestrator.cpp
 * @brief Implementation of Qwen2 compute graph orchestrator
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements the execution layer for Qwen2 models, managing
 * graph execution, device contexts, and caching.
 */

#include "GraphOrchestrator.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "../utils/MPIContext.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/UnifiedKVCache.h"
#include <chrono>

namespace llaminar2
{

    // =========================================================================
    // Constructors
    // =========================================================================

    GraphOrchestrator::GraphOrchestrator(
        std::shared_ptr<Qwen2Graph> graph_builder,
        std::shared_ptr<MPIContext> mpi_ctx,
        const GraphCacheConfig &cache_config)
        : graph_builder_(std::move(graph_builder)),
          mpi_ctx_(std::move(mpi_ctx)),
          cache_config_(cache_config)
    {
        if (!graph_builder_)
        {
            throw std::invalid_argument("GraphOrchestrator requires a valid graph builder");
        }

        // Configure executor from graph builder's config
        GraphExecutorConfig exec_config;
        exec_config.default_device = graph_builder_->config().default_device;
        exec_config.enable_profiling = graph_builder_->config().enable_profiling;
        exec_config.enable_validation = graph_builder_->config().enable_validation;

        // Parse execution mode from environment
        const auto &env = debugEnv();
        if (env.execution.execution_mode == "parallel")
        {
            exec_config.mode = ExecutionMode::PARALLEL;
        }
        else if (env.execution.execution_mode == "pipelined")
        {
            exec_config.mode = ExecutionMode::PIPELINED;
        }
        else
        {
            exec_config.mode = ExecutionMode::SEQUENTIAL;
        }

        executor_ = GraphExecutor(exec_config);

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_INFO("[GraphOrchestrator] Initialized with graph builder, caching="
                 << (cache_config_.enabled ? "enabled" : "disabled"));
    }

    GraphOrchestrator::GraphOrchestrator(
        const Qwen2GraphConfig &graph_config,
        std::shared_ptr<MPIContext> mpi_ctx,
        const GraphCacheConfig &cache_config)
        // Create Qwen2Graph with MPI context FIRST (before any move)
        : graph_builder_(std::make_shared<Qwen2Graph>(graph_config, mpi_ctx)),
          mpi_ctx_(std::move(mpi_ctx)),
          cache_config_(cache_config)
    {
        // Duplicate initialization logic from the other constructor
        if (!graph_builder_)
        {
            throw std::invalid_argument("GraphOrchestrator requires a valid graph builder");
        }

        // Configure executor from graph builder's config
        GraphExecutorConfig exec_config;
        exec_config.default_device = graph_builder_->config().default_device;
        exec_config.enable_profiling = graph_builder_->config().enable_profiling;
        exec_config.enable_validation = graph_builder_->config().enable_validation;

        // Parse execution mode from environment
        const auto &env = debugEnv();
        if (env.execution.execution_mode == "parallel")
        {
            exec_config.mode = ExecutionMode::PARALLEL;
        }
        else if (env.execution.execution_mode == "pipelined")
        {
            exec_config.mode = ExecutionMode::PIPELINED;
        }
        else
        {
            exec_config.mode = ExecutionMode::SEQUENTIAL;
        }

        executor_ = GraphExecutor(exec_config);

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_INFO("[GraphOrchestrator] Initialized with graph builder, caching="
                 << (cache_config_.enabled ? "enabled" : "disabled"));
    }

    // =========================================================================
    // Device Context Management
    // =========================================================================

    IDeviceContext *GraphOrchestrator::getDeviceContext(int device_idx)
    {
        auto it = device_contexts_.find(device_idx);
        if (it != device_contexts_.end())
        {
            return it->second.get();
        }

        // Create new context
        auto ctx = IDeviceContext::create(device_idx);
        if (!ctx)
        {
            LOG_ERROR("[GraphOrchestrator] Failed to create device context for device " << device_idx);
            return nullptr;
        }

        IDeviceContext *raw_ptr = ctx.get();
        device_contexts_[device_idx] = std::move(ctx);

        LOG_DEBUG("[GraphOrchestrator] Created device context for device " << device_idx);
        return raw_ptr;
    }

    // =========================================================================
    // Weight and Buffer Configuration
    // =========================================================================

    void GraphOrchestrator::setWeights(const Qwen2ModelWeights &weights)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[GraphOrchestrator] Cannot set weights: graph builder not initialized");
            return;
        }
        graph_builder_->setWeights(weights);
        LOG_DEBUG("[GraphOrchestrator] Model weights configured for full forward pass");
    }

    void GraphOrchestrator::setBuffers(const Qwen2ModelBuffers &buffers)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[GraphOrchestrator] Cannot set buffers: graph builder not initialized");
            return;
        }
        graph_builder_->setBuffers(buffers);
        LOG_DEBUG("[GraphOrchestrator] Model buffers configured for full forward pass");
    }

    bool GraphOrchestrator::hasGlobalWeights() const
    {
        if (!graph_builder_)
        {
            return false;
        }
        // Check if the builder's isInitialized returns true AND we have global weights
        // isInitialized() checks get_layer_weights, but we also need embedding_table etc.
        // For now, rely on the graph builder's internal state
        return graph_builder_->isInitialized();
    }

    // =========================================================================
    // Graph Buffer Management (Phase 3 - moved from Qwen2Graph)
    // =========================================================================

    bool GraphOrchestrator::initializeBuffers(int seq_len)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[GraphOrchestrator] initializeBuffers called but graph_builder not set");
            return false;
        }

        const auto &config = graph_builder_->config();
        if (!config.use_graph_buffer_management)
        {
            LOG_WARN("[GraphOrchestrator] initializeBuffers called but use_graph_buffer_management=false");
            return false;
        }

        LOG_INFO("[GraphOrchestrator] Initializing buffers with graph management, seq_len=" << seq_len);

        // Get schema and resolver config from graph builder
        GraphSchema schema = graph_builder_->getSchema();
        GraphResolverConfig resolver_config = graph_builder_->getResolverConfig(seq_len);

        // Verify TensorFactory is set
        if (!tensor_factory_)
        {
            LOG_ERROR("[GraphOrchestrator] TensorFactory not set. Call setTensorFactory() before initializeBuffers()");
            return false;
        }

        // Create buffer manager with TensorFactory
        buffer_manager_ = std::make_unique<GraphBufferManager>(
            tensor_factory_, mpi_ctx_.get());

        // Build layer buffer specifications using BufferAllocator
        auto layer_reqs = BufferAllocator::resolveLayerBuffers(schema, resolver_config);

        // Allocate layer buffers (iterate over vector)
        for (const auto &desc : layer_reqs.buffers)
        {
            if (!buffer_manager_->allocateBuffer("layer", desc))
            {
                LOG_ERROR("[GraphOrchestrator] Failed to allocate layer buffer: " << desc.name);
                return false;
            }
        }

        // Build model-level buffer specifications (current_hidden, logits)
        auto model_reqs = BufferAllocator::resolveModelBuffers(schema, resolver_config);

        for (const auto &desc : model_reqs.buffers)
        {
            if (!buffer_manager_->allocateBuffer("model", desc))
            {
                LOG_ERROR("[GraphOrchestrator] Failed to allocate model buffer: " << desc.name);
                return false;
            }
        }

        // Bind allocated buffers to managed_buffers_ struct
        bindGraphManagedBuffers(seq_len);

        auto &stats = buffer_manager_->stats();
        LOG_INFO("[GraphOrchestrator] Buffer initialization complete: "
                 << "total=" << (stats.total_bytes / (1024.0 * 1024.0)) << " MB, "
                 << "scratch=" << (stats.scratch_bytes / (1024.0 * 1024.0)) << " MB, "
                 << "output=" << (stats.output_bytes / (1024.0 * 1024.0)) << " MB");

        // Log theoretical aliasing savings
        auto [original, optimized] = BufferAllocator::estimateMemorySavings(schema, resolver_config);
        double savings = (original > 0) ? 100.0 * (original - optimized) / original : 0.0;
        LOG_INFO("[GraphOrchestrator] Theoretical aliasing savings: "
                 << (original / 1024.0) << " KB -> " << (optimized / 1024.0) << " KB"
                 << " (" << savings << "% reduction)");

        // Also update the graph builder's buffers reference
        graph_builder_->setBuffers(managed_buffers_);

        return true;
    }

    void GraphOrchestrator::releaseBuffers()
    {
        if (buffer_manager_)
        {
            buffer_manager_->releaseAll();
            buffer_manager_.reset();
            LOG_INFO("[GraphOrchestrator] Buffers released");
        }

        owned_buffers_.clear();

        // Clear buffer pointers
        managed_buffers_ = Qwen2ModelBuffers{};
    }

    Qwen2ActivationBuffers &GraphOrchestrator::getInternalBuffers()
    {
        return managed_buffers_.layer_buffers;
    }

    const Qwen2ActivationBuffers &GraphOrchestrator::getInternalBuffers() const
    {
        return managed_buffers_.layer_buffers;
    }

    const Qwen2ModelBuffers &GraphOrchestrator::getModelBuffers() const
    {
        return managed_buffers_;
    }

    const BufferAllocationStats *GraphOrchestrator::bufferStats() const
    {
        if (!buffer_manager_)
        {
            return nullptr;
        }
        return &buffer_manager_->stats();
    }

    void GraphOrchestrator::bindGraphManagedBuffers(int seq_len)
    {
        (void)seq_len; // May be used for validation in the future

        if (!buffer_manager_)
        {
            LOG_ERROR("[GraphOrchestrator] bindGraphManagedBuffers: buffer_manager_ is null");
            return;
        }

        // Bind layer buffers
        auto &lb = managed_buffers_.layer_buffers;

        lb.residual = buffer_manager_->getBuffer("layer", BufferNames::RESIDUAL);
        lb.normalized = buffer_manager_->getBuffer("layer", BufferNames::NORMALIZED);

        // Attention buffers
        lb.Q = buffer_manager_->getBuffer("layer", BufferNames::Q);
        lb.K = buffer_manager_->getBuffer("layer", BufferNames::K);
        lb.V = buffer_manager_->getBuffer("layer", BufferNames::V);
        lb.attn_output = buffer_manager_->getBuffer("layer", BufferNames::ATTN_OUTPUT);
        lb.attn_proj = buffer_manager_->getBuffer("layer", BufferNames::ATTN_PROJ);
        lb.workspace_scores = buffer_manager_->getBuffer("layer", BufferNames::WORKSPACE_SCORES);
        lb.workspace_context = buffer_manager_->getBuffer("layer", BufferNames::WORKSPACE_CONTEXT);
        lb.workspace_mask = buffer_manager_->getBuffer("layer", BufferNames::WORKSPACE_MASK);

        // FFN buffers
        lb.gate = buffer_manager_->getBuffer("layer", BufferNames::GATE);
        lb.up = buffer_manager_->getBuffer("layer", BufferNames::UP);
        lb.ffn_output = buffer_manager_->getBuffer("layer", BufferNames::FFN_OUTPUT);

        // Model-level buffers
        managed_buffers_.current_hidden = buffer_manager_->getBuffer("model", BufferNames::CURRENT_HIDDEN);
        managed_buffers_.logits = buffer_manager_->getBuffer("model", BufferNames::LOGITS);

        LOG_DEBUG("[GraphOrchestrator] Bound graph-managed buffers: "
                  << "residual=" << lb.residual
                  << " Q=" << lb.Q
                  << " gate=" << lb.gate
                  << " current_hidden=" << managed_buffers_.current_hidden
                  << " logits=" << managed_buffers_.logits);
    }

    // =========================================================================
    // Execution Methods
    // =========================================================================

    bool GraphOrchestrator::executeForward(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!input.token_ids && !input.batches)
        {
            LOG_ERROR("[GraphOrchestrator] No token input provided");
            return false;
        }

        if (input.seq_len <= 0)
        {
            LOG_ERROR("[GraphOrchestrator] Invalid sequence length: " << input.seq_len);
            return false;
        }

        LOG_TRACE("[GraphOrchestrator] executeForward: batch_size=" << input.batch_size
                                                                    << ", seq_len=" << input.seq_len
                                                                    << ", device=" << input.device_idx);

        // Build position IDs if not provided externally
        std::vector<int> position_ids_storage;
        Qwen2ForwardInput effective_input = input;

        if (!input.position_ids)
        {
            position_ids_storage = Qwen2Graph::buildPositionIds(
                input.seq_len, input.batch_size, input.position_offset);
            effective_input.position_ids = position_ids_storage.data();
        }

        // Build forward graph via declarative builder
        ComputeGraph graph = graph_builder_->buildFullForwardGraph(effective_input, output);

        if (graph.size() == 0)
        {
            LOG_ERROR("[GraphOrchestrator] Empty forward graph");
            return false;
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(input.device_idx);
        if (!ctx)
        {
            LOG_ERROR("[GraphOrchestrator] Failed to get device context");
            return false;
        }

        // Execute
        bool success = executor_.execute(graph, ctx);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        LOG_DEBUG("[GraphOrchestrator] Forward completed in " << ms << "ms, success=" << success);

        return success;
    }

    bool GraphOrchestrator::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        return executor_.execute(graph, ctx);
    }

    bool GraphOrchestrator::executeAttention(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        // Debug: dump input to attention (for layer 0 only)
        if (layer_idx == 0)
        {
            const float *input = buffers.current_hidden->fp32_data();
            LOG_INFO("[ORCH_ATTN_INPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                << " input[0:4]=" << input[0] << "," << input[1]
                                                << "," << input[2] << "," << input[3]);
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(device_idx);
        if (!ctx)
        {
            return false;
        }

        int pos_offset = position_ids ? position_ids[0] : 0;

        // =============================================================================
        // Graph Caching for Decode Mode (Phase 10)
        // =============================================================================
        if (cache_config_.enabled && cache_config_.cache_attention &&
            seq_len == cache_config_.decode_seq_len &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached graph
            if (cache.attention_decode && cache.cached_seq_len == seq_len && cache.valid)
            {
                LOG_DEBUG("[GraphOrchestrator] Reusing cached attention graph for layer "
                          << layer_idx << " (pos_offset=" << pos_offset << ")");

                // Update dynamic parameters (position offset)
                updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);

                // Execute cached graph
                bool success = executor_.execute(*cache.attention_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[GraphOrchestrator] Cached attention graph failed at layer " << layer_idx);
                }

                cache.attention_decode->reset();
                cache_stats_.attention_cache_hits++;
                return success;
            }

            // Build and cache the graph
            LOG_DEBUG("[GraphOrchestrator] Building and caching attention graph for layer "
                      << layer_idx << " (decode mode)");

            cache.attention_decode = std::make_unique<ComputeGraph>(
                graph_builder_->buildAttentionGraph(layer, buffers, layer_idx, seq_len,
                                                    1, kv_cache, position_ids, device_idx, nullptr));
            cache.cached_seq_len = seq_len;
            cache.valid = true;
            cache_stats_.attention_cache_misses++;

            // Execute the newly built graph
            bool success = executor_.execute(*cache.attention_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[GraphOrchestrator] Attention block failed at layer " << layer_idx);
            }

            cache.attention_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled)
        // =============================================================================
        cache_stats_.attention_cache_misses++;

        ComputeGraph graph = graph_builder_->buildAttentionGraph(
            layer, buffers, layer_idx, seq_len, 1, kv_cache, position_ids, device_idx, nullptr);

        // Debug: log graph structure
        if (layer_idx == 0)
        {
            auto order = graph.getExecutionOrder();
            LOG_INFO("[ORCH_ATTN] Graph has " << graph.size() << " nodes, execution order:");
            for (const auto &name : order)
            {
                LOG_INFO("[ORCH_ATTN]   - " << name);
            }
        }

        bool success = executor_.execute(graph, ctx);

        // Debug: dump intermediate buffers (for layer 0 only)
        if (layer_idx == 0 && buffers.normalized && buffers.Q &&
            buffers.attn_output && buffers.attn_proj)
        {
            const float *output = buffers.current_hidden->fp32_data();
            LOG_INFO("[ORCH_ATTN_OUTPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                 << " output[0:4]=" << output[0] << "," << output[1]
                                                 << "," << output[2] << "," << output[3]);
        }

        if (!success)
        {
            LOG_ERROR("[GraphOrchestrator] Attention block failed at layer " << layer_idx);
        }

        return success;
    }

    bool GraphOrchestrator::executeFFN(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int device_idx)
    {
        // Get device context
        IDeviceContext *ctx = getDeviceContext(device_idx);
        if (!ctx)
        {
            return false;
        }

        // =============================================================================
        // Graph Caching for Decode Mode (Phase 10)
        // =============================================================================
        if (cache_config_.enabled && cache_config_.cache_ffn &&
            seq_len == cache_config_.decode_seq_len &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached FFN graph
            if (cache.ffn_decode && cache.valid)
            {
                LOG_DEBUG("[GraphOrchestrator] Reusing cached FFN graph for layer " << layer_idx);

                // Execute cached graph (no params to update for FFN)
                bool success = executor_.execute(*cache.ffn_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[GraphOrchestrator] Cached FFN graph failed at layer " << layer_idx);
                }

                cache.ffn_decode->reset();
                cache_stats_.ffn_cache_hits++;
                return success;
            }

            // Build and cache the graph
            LOG_DEBUG("[GraphOrchestrator] Building and caching FFN graph for layer "
                      << layer_idx << " (decode mode)");

            cache.ffn_decode = std::make_unique<ComputeGraph>(
                graph_builder_->buildFFNGraph(layer, buffers, layer_idx, seq_len, 1, device_idx)); // batch_size=1 for decode
            cache_stats_.ffn_cache_misses++;

            // Execute the newly built graph
            bool success = executor_.execute(*cache.ffn_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[GraphOrchestrator] FFN block failed at layer " << layer_idx);
            }

            cache.ffn_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled)
        // =============================================================================
        cache_stats_.ffn_cache_misses++;

        ComputeGraph graph = graph_builder_->buildFFNGraph(layer, buffers, layer_idx, seq_len, 1, device_idx); // batch_size=1 for deprecated path

        bool success = executor_.execute(graph, ctx);

        if (!success)
        {
            LOG_ERROR("[GraphOrchestrator] FFN block failed at layer " << layer_idx);
        }

        return success;
    }

    bool GraphOrchestrator::executeLayer(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        LOG_INFO("[GraphOrchestrator::executeLayer] LAYER_EXEC_ENTERED layer_idx="
                 << layer_idx << " seq_len=" << seq_len);

        // Execute attention block
        if (!executeAttention(layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device_idx))
        {
            return false;
        }

        // Execute FFN block
        if (!executeFFN(layer, buffers, layer_idx, seq_len, device_idx))
        {
            return false;
        }

        return true;
    }

    // =========================================================================
    // Cache Management
    // =========================================================================

    void GraphOrchestrator::clearCache()
    {
        // Clear graph caches
        for (auto &cache : layer_graph_cache_)
        {
            cache.invalidate();
        }

        // Clear device contexts
        device_contexts_.clear();

        // Reset state
        last_pos_offset_ = -1;

        // Reset stats
        cache_stats_ = CacheStats{};

        LOG_DEBUG("[GraphOrchestrator] All caches cleared");
    }

    void GraphOrchestrator::invalidateGraphCache(int layer_idx)
    {
        if (layer_idx < 0)
        {
            // Invalidate all layers
            for (auto &cache : layer_graph_cache_)
            {
                cache.invalidate();
            }
            cache_stats_.cached_layers = 0;
            LOG_DEBUG("[GraphOrchestrator] All layer graph caches invalidated");
        }
        else if (static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            layer_graph_cache_[layer_idx].invalidate();
            LOG_DEBUG("[GraphOrchestrator] Layer " << layer_idx << " graph cache invalidated");
        }
    }

    bool GraphOrchestrator::hasValidCachedGraph(int layer_idx, bool is_attention) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        if (!cache.valid)
            return false;

        return is_attention ? (cache.attention_decode != nullptr) : (cache.ffn_decode != nullptr);
    }

    void GraphOrchestrator::setGraphCachingEnabled(bool enabled)
    {
        if (cache_config_.enabled != enabled)
        {
            cache_config_.enabled = enabled;
            if (!enabled)
            {
                invalidateGraphCache(-1);
            }
            LOG_INFO("[GraphOrchestrator] Graph caching "
                     << (enabled ? "enabled" : "disabled"));
        }
    }

    void GraphOrchestrator::initializeGraphCache(int n_layers)
    {
        layer_graph_cache_.resize(n_layers);
        cache_stats_.cached_layers = n_layers;
        LOG_INFO("[GraphOrchestrator] Graph cache initialized for " << n_layers << " layers");
    }

    // =========================================================================
    // Inference State Management (Phase 5)
    // =========================================================================

    bool GraphOrchestrator::initializeInferenceState(
        int batch_size,
        int max_seq_len,
        int device_idx)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[GraphOrchestrator] Cannot initialize state: no graph builder");
            return false;
        }

        const auto &config = graph_builder_->config();
        int d_model = config.d_model;
        int vocab_size = config.vocab_size;
        int n_layers = config.n_layers;
        int n_heads = config.n_heads;
        int n_kv_heads = config.n_kv_heads;
        int head_dim = config.head_dim;
        int d_ff = config.d_ff;

        // For tensor parallelism, use local head counts for buffer allocation
        // When qkv_column_parallel is enabled, each rank processes only its subset of heads
        int buffer_n_heads = config.qkv_column_parallel ? config.local_n_heads : n_heads;
        int buffer_n_kv_heads = config.qkv_column_parallel ? config.local_n_kv_heads : n_kv_heads;
        int buffer_d_ff = config.ffn_column_parallel ? config.d_ff_local : d_ff;

        LOG_INFO("[GraphOrchestrator] Initializing inference state: batch_size=" << batch_size
                                                                                 << " max_seq_len=" << max_seq_len
                                                                                 << " d_model=" << d_model
                                                                                 << " vocab_size=" << vocab_size);

        if (config.qkv_column_parallel)
        {
            LOG_DEBUG("[GraphOrchestrator] Using local buffer sizes for TP: n_heads=" << buffer_n_heads
                                                                                      << "/" << n_heads << " n_kv_heads=" << buffer_n_kv_heads << "/" << n_kv_heads
                                                                                      << " d_ff=" << buffer_d_ff << "/" << d_ff);
        }

        // Create a default MPI context if none was provided
        std::shared_ptr<MPIContext> local_mpi_ctx = mpi_ctx_;
        if (!local_mpi_ctx)
        {
            // Create a single-rank MPI context for non-MPI usage
            local_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            LOG_DEBUG("[GraphOrchestrator] Created default single-rank MPI context");
        }

        // Create tensor factory
        TensorFactory factory(*local_mpi_ctx);

        // Get activation precision from config
        ActivationPrecision act_prec = config.activation_precision;
        const char *prec_name = (act_prec == ActivationPrecision::Q8_1) ? "Q8_1" : (act_prec == ActivationPrecision::FP16) ? "FP16"
                                                                               : (act_prec == ActivationPrecision::BF16)   ? "BF16"
                                                                                                                           : "FP32";
        LOG_INFO("[GraphOrchestrator] Activation buffer precision: " << prec_name);

        // Allocate core buffers (always FP32 - interface with embeddings/softmax)
        state_.hidden = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);
        state_.logits = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(vocab_size)},
            device_idx);

        // Phase 5: Allocate local logits buffer for column-parallel LM head
        if (config.lm_head_column_parallel && config.vocab_local > 0)
        {
            state_.logits_local = factory.createFP32(
                std::vector<size_t>{static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(config.vocab_local)},
                device_idx);
            LOG_DEBUG("[GraphOrchestrator] Allocated logits_local buffer: ["
                      << batch_size * max_seq_len << ", " << config.vocab_local << "]");
        }

        // Allocate norm/residual buffers (always FP32 - high precision needed for residual stream)
        state_.normalized = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);
        state_.residual = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);

        // QKV buffers - use configured activation precision
        // These are the primary activation tensors that benefit from quantization
        state_.Q = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_heads * head_dim)},
            act_prec, device_idx);
        state_.K = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_kv_heads * head_dim)},
            act_prec, device_idx);
        state_.V = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_kv_heads * head_dim)},
            act_prec, device_idx);
        state_.attn_output = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_heads * head_dim)},
            act_prec, device_idx);
        // attn_proj is the output of Wo projection which feeds into the residual stream
        // Keep as FP32 for numerical stability in residual connections
        state_.attn_proj = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);

        // FFN buffers - gate and up use configured activation precision
        state_.gate = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_d_ff)},
            act_prec, device_idx);
        state_.up = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_d_ff)},
            act_prec, device_idx);
        // ffn_output is the output of FFN Down projection which feeds into the residual stream
        // Keep as FP32 for numerical stability in residual connections
        state_.ffn_output = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);

        // Attention workspace - use local head counts for tensor parallelism
        // scores: [batch_size * buffer_n_heads, max_seq_len, max_seq_len]
        state_.workspace_scores = factory.createFP32(
            {static_cast<size_t>(batch_size * buffer_n_heads * max_seq_len), static_cast<size_t>(max_seq_len)},
            device_idx);
        // context: [batch_size * buffer_n_heads, max_seq_len, head_dim]
        state_.workspace_context = factory.createFP32(
            {static_cast<size_t>(batch_size * buffer_n_heads * max_seq_len), static_cast<size_t>(head_dim)},
            device_idx);
        // mask: [batch_size, max_seq_len, max_seq_len]
        state_.workspace_mask = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(max_seq_len)},
            device_idx);

        // Create KV cache - use sharded cache for tensor parallelism
        // Check if tensor parallelism is enabled (indicated by local_n_kv_heads set)
        // KV cache uses the same activation precision as other buffers
        bool use_sharded_cache = (config.local_n_kv_heads > 0 && config.local_n_kv_heads < n_kv_heads);
        if (use_sharded_cache && mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            // Compute KV head distribution from local_n_kv_heads
            // kv_head_start = rank * local_n_kv_heads (assumes even distribution)
            int kv_head_start = mpi_ctx_->rank() * config.local_n_kv_heads;

            LOG_DEBUG("[GraphOrchestrator] Creating sharded KV cache: "
                      << n_kv_heads << " total KV heads, "
                      << config.local_n_kv_heads << " local KV heads (start=" << kv_head_start << ")"
                      << " precision=" << prec_name);

            state_.kv_cache = createShardedKVCache(
                act_prec,
                *local_mpi_ctx,
                n_layers,
                batch_size,
                max_seq_len,
                n_kv_heads,
                config.local_n_kv_heads,
                kv_head_start,
                head_dim,
                device_idx);
        }
        else
        {
            // Non-sharded (replicated) KV cache for single-rank or non-tensor-parallel
            state_.kv_cache = createUnifiedKVCache(
                act_prec,
                *local_mpi_ctx,
                n_layers,
                batch_size,
                max_seq_len,
                n_kv_heads,
                head_dim,
                device_idx);
        }

        // Initialize position tracking
        state_.positions.assign(batch_size, 0);
        state_.sequence_lengths.assign(batch_size, 0);

        // Store config
        state_.batch_size = batch_size;
        state_.max_seq_len = max_seq_len;
        state_.d_model = d_model;
        state_.vocab_size = vocab_size;
        state_.device_idx = device_idx;

        LOG_INFO("[GraphOrchestrator] Inference state initialized successfully");
        return true;
    }

    const float *GraphOrchestrator::forward(
        const int *tokens,
        int seq_len,
        int batch_size)
    {
        if (!state_.isInitialized())
        {
            LOG_ERROR("[GraphOrchestrator] forward() called without initialized state");
            return nullptr;
        }

        if (!hasGlobalWeights())
        {
            LOG_ERROR("[GraphOrchestrator] forward() called without global weights set");
            return nullptr;
        }

        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[GraphOrchestrator] Batch size " << batch_size
                                                        << " exceeds initialized batch size " << state_.batch_size);
            return nullptr;
        }

        int total_tokens = batch_size * seq_len;
        if (total_tokens > state_.batch_size * state_.max_seq_len)
        {
            LOG_ERROR("[GraphOrchestrator] Total tokens " << total_tokens
                                                          << " exceeds buffer capacity "
                                                          << state_.batch_size * state_.max_seq_len);
            return nullptr;
        }

        // Build position IDs
        std::vector<int> position_ids;
        position_ids.reserve(total_tokens);
        for (int b = 0; b < batch_size; ++b)
        {
            int pos_offset = state_.positions[b];
            for (int s = 0; s < seq_len; ++s)
            {
                position_ids.push_back(pos_offset + s);
            }
        }

        // Prepare model buffers from state
        Qwen2ModelBuffers model_buffers;
        model_buffers.current_hidden = state_.hidden.get();
        model_buffers.logits = state_.logits.get();

        // Phase 5: Set local logits buffer for column-parallel LM head
        if (state_.logits_local)
        {
            model_buffers.logits_local = state_.logits_local.get();
        }

        // Populate layer buffers
        model_buffers.layer_buffers.current_hidden = state_.hidden.get();
        model_buffers.layer_buffers.normalized = state_.normalized.get();
        model_buffers.layer_buffers.residual = state_.residual.get();
        model_buffers.layer_buffers.Q = state_.Q.get();
        model_buffers.layer_buffers.K = state_.K.get();
        model_buffers.layer_buffers.V = state_.V.get();
        model_buffers.layer_buffers.attn_output = state_.attn_output.get();
        model_buffers.layer_buffers.attn_proj = state_.attn_proj.get();
        model_buffers.layer_buffers.gate = state_.gate.get();
        model_buffers.layer_buffers.up = state_.up.get();
        model_buffers.layer_buffers.ffn_output = state_.ffn_output.get();
        model_buffers.layer_buffers.workspace_scores = state_.workspace_scores.get();
        model_buffers.layer_buffers.workspace_context = state_.workspace_context.get();
        model_buffers.layer_buffers.workspace_mask = state_.workspace_mask.get();

        setBuffers(model_buffers);

        // Build forward input
        Qwen2ForwardInput input;
        input.token_ids = tokens;
        input.position_ids = position_ids.data();
        input.batch_size = batch_size;
        input.seq_len = seq_len;
        input.position_offset = state_.positions[0]; // Legacy compat
        input.device_idx = state_.device_idx;
        input.kv_cache = state_.kv_cache.get();
        // Pass sequence_lengths for batch-aware attention masking
        // This enables proper separation of sequences in batched execution
        input.sequence_lengths = (batch_size > 1 && !state_.sequence_lengths.empty())
                                     ? &state_.sequence_lengths
                                     : nullptr;

        // Build forward output
        Qwen2ForwardOutput output;
        output.logits = state_.logits.get();
        output.hidden = state_.hidden.get();

        // Execute forward pass
        bool success = executeForward(input, output);

        if (!success)
        {
            LOG_ERROR("[GraphOrchestrator] forward() execution failed");
            return nullptr;
        }

        // Update positions
        for (int b = 0; b < batch_size; ++b)
        {
            state_.positions[b] += seq_len;
            state_.sequence_lengths[b] += seq_len;
        }

        // Return logits pointer
        return state_.logits->fp32_data();
    }

    const float *GraphOrchestrator::logits() const
    {
        if (!state_.logits)
        {
            return nullptr;
        }
        return state_.logits->fp32_data();
    }

    // =========================================================================
    // Batch Interface Implementation
    // =========================================================================

    bool GraphOrchestrator::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        if (token_batches.empty())
        {
            LOG_ERROR("[GraphOrchestrator] forward_batch() called with empty batch");
            return false;
        }

        int batch_size = static_cast<int>(token_batches.size());
        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[GraphOrchestrator] Batch size " << batch_size
                                                        << " exceeds initialized batch size " << state_.batch_size);
            return false;
        }

        // Find max sequence length (for padding)
        int max_len = 0;
        for (const auto &seq : token_batches)
        {
            max_len = std::max(max_len, static_cast<int>(seq.size()));
        }
        padded_seq_len_ = max_len;

        // Update sequence lengths
        state_.sequence_lengths.resize(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            state_.sequence_lengths[i] = static_cast<int>(token_batches[i].size());
        }

        // Create flattened, padded token array [batch_size * padded_seq_len]
        std::vector<int> flat_tokens(batch_size * padded_seq_len_, 0); // pad with 0
        for (int b = 0; b < batch_size; ++b)
        {
            const auto &seq = token_batches[b];
            for (size_t s = 0; s < seq.size(); ++s)
            {
                flat_tokens[b * padded_seq_len_ + s] = seq[s];
            }
        }

        // Call the 3-parameter forward() with padded tokens
        const float *result = forward(flat_tokens.data(), padded_seq_len_, batch_size);
        return result != nullptr;
    }

    const float *GraphOrchestrator::getLogits(int seq_idx) const
    {
        if (!state_.logits)
        {
            return nullptr;
        }

        if (seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            LOG_ERROR("[GraphOrchestrator] Invalid sequence index " << seq_idx
                                                                    << " (batch_size=" << state_.batch_size << ")");
            return nullptr;
        }

        // Return pointer to logits for requested sequence
        // Layout: [batch_size * padded_seq_len, vocab_size]
        // For sequence seq_idx, logits start at row (seq_idx * padded_seq_len_)
        const float *base = state_.logits->fp32_data();
        if (!base)
        {
            return nullptr;
        }

        return base + (seq_idx * padded_seq_len_ * state_.vocab_size);
    }

    int GraphOrchestrator::getPosition(int seq_idx) const
    {
        if (seq_idx < 0 || static_cast<size_t>(seq_idx) >= state_.positions.size())
        {
            return 0;
        }
        return state_.positions[seq_idx];
    }

    void GraphOrchestrator::clearInferenceState()
    {
        state_.clear();
        LOG_DEBUG("[GraphOrchestrator] Inference state cleared");
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    void GraphOrchestrator::updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len)
    {
        // Update all stages in the graph that have dynamic parameters
        auto order = graph.getExecutionOrder();

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
                continue;

            // Update dynamic params (pos_offset, seq_len)
            // Only stages that override updateDynamicParams will actually do anything
            node->stage->updateDynamicParams(pos_offset, seq_len);
        }

        LOG_TRACE("[GraphOrchestrator] Updated cached graph params: pos_offset="
                  << pos_offset << " seq_len=" << seq_len);
    }

    bool GraphOrchestrator::canUseCachedGraph(int layer_idx, int seq_len) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        return cache.valid && cache.cached_seq_len == seq_len;
    }

} // namespace llaminar2
