/**
 * @file DeviceGraphOrchestrator.cpp
 * @brief Implementation of Qwen2 compute graph orchestrator
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements the execution layer for Qwen2 models, managing
 * graph execution, device contexts, and caching.
 */

#include "DeviceGraphOrchestrator.h"
#include "../../config/HybridPrecisionConfig.h"
#include "../../../loaders/WeightManager.h"
#include "../../../loaders/WeightPlacementMap.h"
#include "../../../config/TensorParallelConfig.h"
#include "../../../config/PipelineConfig.h"
#include "../../../collective/ILocalTPContext.h" // createLocalTPContext()
#include "../../../collective/ILocalPPContext.h" // createLocalPPContext(), HierarchicalPPConfig
#include "../../../collective/PPStage.h"         // PPStage variant type
#include "../../../backends/p2p/DirectP2P.h"     // DirectP2PEngine for BAR-backed allocation
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/MPIContext.h"
#include "../../../tensors/TensorFactory.h"
#include "../../../tensors/TensorClasses.h" // For FP32Tensor::createMapped()
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include <chrono>
#include <algorithm>

namespace llaminar2
{

    // =========================================================================
    // Constructors
    // =========================================================================

    DeviceGraphOrchestrator::DeviceGraphOrchestrator(
        Dependencies deps,
        const Qwen2GraphConfig &graph_config,
        const GraphCacheConfig &cache_config)
        : graph_builder_(std::make_shared<Qwen2Graph>(graph_config, nullptr)), // Create graph without MPI (uses topology)
          mpi_ctx_(nullptr),                                                   // No direct MPI context - use injected topology
          cache_config_(cache_config),
          injected_model_ctx_(std::move(deps.model_ctx)),
          injected_topology_(std::move(deps.topology)),
          injected_collective_ctx_(std::move(deps.collective_ctx))
    {
        if (!injected_model_ctx_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator Dependencies requires a valid model_ctx");
        }

        if (!graph_builder_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator failed to create graph builder");
        }

        // Configure executor from graph builder's config
        GraphExecutorConfig exec_config;
        exec_config.default_device = graph_builder_->config().default_device;

        // Parse execution mode and profiling from environment
        const auto &env = debugEnv();

        // Enable profiling from either graph config or env variable
        exec_config.enable_profiling = graph_builder_->config().enable_profiling || env.execution.executor_profiling;
        exec_config.enable_validation = graph_builder_->config().enable_validation;

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

        executor_ = DeviceGraphExecutor(exec_config);

        // Propagate MPI rank to executor for stage dumping (from injected topology)
        if (injected_topology_)
        {
            executor_.setMPIRank(injected_topology_->rank());
        }

        // Wire CollectiveContext to executor for GPU-native collectives (RCCL/NCCL/PCIeBAR)
        if (injected_collective_ctx_)
        {
            executor_.setCollectiveContext(injected_collective_ctx_.get());
            LOG_INFO("[DeviceGraphOrchestrator] Wired CollectiveContext to DeviceGraphExecutor");
        }

        LOG_INFO("[DeviceGraphOrchestrator] Initialized with injected dependencies, caching="
                 << (cache_config_.enabled ? "enabled" : "disabled")
                 << ", topology=" << (injected_topology_ ? "provided" : "none")
                 << ", collective=" << (injected_collective_ctx_ ? "provided" : "none"));
    }

    DeviceGraphOrchestrator::DeviceGraphOrchestrator(
        std::shared_ptr<Qwen2Graph> graph_builder,
        std::shared_ptr<MPIContext> mpi_ctx,
        const GraphCacheConfig &cache_config)
        : graph_builder_(std::move(graph_builder)),
          mpi_ctx_(std::move(mpi_ctx)),
          cache_config_(cache_config)
    {
        if (!graph_builder_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator requires a valid graph builder");
        }

        // Configure executor from graph builder's config
        GraphExecutorConfig exec_config;
        exec_config.default_device = graph_builder_->config().default_device;

        // Parse execution mode and profiling from environment
        const auto &env = debugEnv();

        // Enable profiling from either graph config or env variable
        exec_config.enable_profiling = graph_builder_->config().enable_profiling || env.execution.executor_profiling;
        exec_config.enable_validation = graph_builder_->config().enable_validation;

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

        executor_ = DeviceGraphExecutor(exec_config);

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_INFO("[DeviceGraphOrchestrator] Initialized with graph builder, caching="
                 << (cache_config_.enabled ? "enabled" : "disabled"));
    }

    DeviceGraphOrchestrator::DeviceGraphOrchestrator(
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
            throw std::invalid_argument("DeviceGraphOrchestrator requires a valid graph builder");
        }

        // Configure executor from graph builder's config
        GraphExecutorConfig exec_config;
        exec_config.default_device = graph_builder_->config().default_device;

        // Parse execution mode and profiling from environment
        const auto &env = debugEnv();

        // Enable profiling from either graph config or env variable
        exec_config.enable_profiling = graph_builder_->config().enable_profiling || env.execution.executor_profiling;
        exec_config.enable_validation = graph_builder_->config().enable_validation;
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

        executor_ = DeviceGraphExecutor(exec_config);

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_INFO("[DeviceGraphOrchestrator] Initialized with graph builder, caching="
                 << (cache_config_.enabled ? "enabled" : "disabled"));
    }

    // =========================================================================
    // Device Context Management
    // =========================================================================

    IDeviceContext *DeviceGraphOrchestrator::getDeviceContext(DeviceId device)
    {
        auto it = device_contexts_.find(device);
        if (it != device_contexts_.end())
        {
            return it->second.get();
        }

        // Create new context using DeviceId
        auto ctx = IDeviceContext::create(device);
        if (!ctx)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to create device context for device " << device.toString());
            return nullptr;
        }

        IDeviceContext *raw_ptr = ctx.get();
        device_contexts_[device] = std::move(ctx);

        LOG_DEBUG("[DeviceGraphOrchestrator] Created device context for device " << device.to_string());
        return raw_ptr;
    }

    // =========================================================================
    // Weight and Buffer Configuration
    // =========================================================================

    void DeviceGraphOrchestrator::setWeights(const Qwen2ModelWeights &weights)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set weights: graph builder not initialized");
            return;
        }
        graph_builder_->setWeights(weights);
        LOG_DEBUG("[DeviceGraphOrchestrator] Model weights configured for full forward pass");
    }

    void DeviceGraphOrchestrator::setBuffers(const Qwen2ModelBuffers &buffers)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set buffers: graph builder not initialized");
            return;
        }
        graph_builder_->setBuffers(buffers);
        LOG_DEBUG("[DeviceGraphOrchestrator] Model buffers configured for full forward pass");
    }

    bool DeviceGraphOrchestrator::hasGlobalWeights() const
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

    bool DeviceGraphOrchestrator::initializeBuffers(int seq_len)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] initializeBuffers called but graph_builder not set");
            return false;
        }

        const auto &config = graph_builder_->config();
        if (!config.use_graph_buffer_management)
        {
            LOG_WARN("[DeviceGraphOrchestrator] initializeBuffers called but use_graph_buffer_management=false");
            return false;
        }

        LOG_INFO("[DeviceGraphOrchestrator] Initializing buffers with graph management, seq_len=" << seq_len);

        // Get schema and resolver config from graph builder
        GraphSchema schema = graph_builder_->getSchema();
        GraphResolverConfig resolver_config = graph_builder_->getResolverConfig(seq_len);

        // Verify TensorFactory is set
        if (!tensor_factory_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] TensorFactory not set. Call setTensorFactory() before initializeBuffers()");
            return false;
        }

        // Configure buffer manager with mapped memory for GPU + snapshot scenarios
        // Mapped memory enables zero-copy host access (avoids slow hipMemcpy/cudaMemcpy syncs)
        // This benefits both CUDA and ROCm backends equally
        GraphBufferManagerConfig buffer_config;
        bool use_mapped = state_.device_id.is_gpu() && snapshot_enabled_;
        if (use_mapped)
        {
            buffer_config.use_mapped_memory = true;
            LOG_INFO("[DeviceGraphOrchestrator] Enabling mapped memory for GPU + snapshot mode (zero-copy host access)");
        }

        // =====================================================================
        // Configure BAR-backed allocation for LOCAL TP with PCIeBAR backend
        // =====================================================================
        // When LOCAL TP is active with PCIeBAR backend, row-parallel output
        // buffers (attn_proj, ffn_output) need to be allocated in BAR memory
        // for efficient cross-vendor allreduce. Each device needs its own buffer:
        // - CUDA device: standard FP32 tensor
        // - ROCm device: BAR-backed FP32 tensor accessible by both devices
        //
        // The DeviceGraphBufferManager checks Qwen2BufferSpec::requiresBARBacked() to
        // identify which buffers need BAR allocation.
        // =====================================================================
        if (config.local_tp_ctx && config.local_tp_ctx->degree() > 1)
        {
            buffer_config.tp_degree = config.local_tp_ctx->degree();
            buffer_config.collective_backend = config.local_tp_ctx->backend();
            buffer_config.local_tp_ctx = config.local_tp_ctx;

            // For PCIeBAR backend, find CUDA and ROCm devices for BAR allocation
            if (config.local_tp_ctx->backend() == CollectiveBackendType::PCIE_BAR)
            {
                const auto &devices = config.local_tp_ctx->devices();
                for (const auto &device_addr : devices)
                {
                    DeviceId device_id = device_addr.toLocalDeviceId();
                    if (device_id.is_cuda() && !buffer_config.cuda_device.is_cuda())
                    {
                        buffer_config.cuda_device = device_id;
                    }
                    else if (device_id.is_rocm() && !buffer_config.rocm_device.is_rocm())
                    {
                        buffer_config.rocm_device = device_id;
                    }
                }

                if (buffer_config.cuda_device.is_cuda() && buffer_config.rocm_device.is_rocm())
                {
                    LOG_INFO("[DeviceGraphOrchestrator] Enabled BAR-backed allocation for LOCAL TP PCIeBAR: "
                             << "CUDA=" << buffer_config.cuda_device.toString()
                             << ", ROCm=" << buffer_config.rocm_device.toString());
                }
                else
                {
                    LOG_WARN("[DeviceGraphOrchestrator] PCIeBAR backend but missing CUDA/ROCm pair: "
                             << "CUDA=" << (buffer_config.cuda_device.is_cuda() ? buffer_config.cuda_device.toString() : "(none)")
                             << ", ROCm=" << (buffer_config.rocm_device.is_rocm() ? buffer_config.rocm_device.toString() : "(none)"));
                }
            }
        }

        // Create buffer manager with TensorFactory
        buffer_manager_ = std::make_unique<DeviceGraphBufferManager>(
            tensor_factory_, mpi_ctx_.get(), buffer_config);

        // Build layer buffer specifications using BufferAllocator
        auto layer_reqs = BufferAllocator::resolveLayerBuffers(schema, resolver_config);

        // Allocate layer buffers (iterate over vector)
        for (const auto &desc : layer_reqs.buffers)
        {
            if (!buffer_manager_->allocateBuffer("layer", desc))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate layer buffer: " << desc.name);
                return false;
            }
        }

        // Build model-level buffer specifications (current_hidden, logits)
        auto model_reqs = BufferAllocator::resolveModelBuffers(schema, resolver_config);

        for (const auto &desc : model_reqs.buffers)
        {
            if (!buffer_manager_->allocateBuffer("model", desc))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate model buffer: " << desc.name);
                return false;
            }
        }

        // Bind allocated buffers to managed_buffers_ struct
        bindGraphManagedBuffers(seq_len);

        auto &stats = buffer_manager_->stats();
        LOG_INFO("[DeviceGraphOrchestrator] Buffer initialization complete: "
                 << "total=" << (stats.total_bytes / (1024.0 * 1024.0)) << " MB, "
                 << "scratch=" << (stats.scratch_bytes / (1024.0 * 1024.0)) << " MB, "
                 << "output=" << (stats.output_bytes / (1024.0 * 1024.0)) << " MB");

        // Log theoretical aliasing savings
        auto [original, optimized] = BufferAllocator::estimateMemorySavings(schema, resolver_config);
        double savings = (original > 0) ? 100.0 * (original - optimized) / original : 0.0;
        LOG_INFO("[DeviceGraphOrchestrator] Theoretical aliasing savings: "
                 << (original / 1024.0) << " KB -> " << (optimized / 1024.0) << " KB"
                 << " (" << savings << "% reduction)");

        // Also update the graph builder's buffers reference
        graph_builder_->setBuffers(managed_buffers_);

        return true;
    }

    void DeviceGraphOrchestrator::releaseBuffers()
    {
        if (buffer_manager_)
        {
            buffer_manager_->releaseAll();
            buffer_manager_.reset();
            LOG_INFO("[DeviceGraphOrchestrator] Buffers released");
        }

        owned_buffers_.clear();

        // Clear buffer pointers
        managed_buffers_ = Qwen2ModelBuffers{};
    }

    Qwen2ActivationBuffers &DeviceGraphOrchestrator::getInternalBuffers()
    {
        return managed_buffers_.layer_buffers;
    }

    const Qwen2ActivationBuffers &DeviceGraphOrchestrator::getInternalBuffers() const
    {
        return managed_buffers_.layer_buffers;
    }

    const Qwen2ModelBuffers &DeviceGraphOrchestrator::getModelBuffers() const
    {
        return managed_buffers_;
    }

    const BufferAllocationStats *DeviceGraphOrchestrator::bufferStats() const
    {
        if (!buffer_manager_)
        {
            return nullptr;
        }
        return &buffer_manager_->stats();
    }

    void DeviceGraphOrchestrator::bindGraphManagedBuffers(int seq_len)
    {
        (void)seq_len; // May be used for validation in the future

        if (!buffer_manager_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] bindGraphManagedBuffers: buffer_manager_ is null");
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

        LOG_DEBUG("[DeviceGraphOrchestrator] Bound graph-managed buffers: "
                  << "residual=" << lb.residual
                  << " Q=" << lb.Q
                  << " gate=" << lb.gate
                  << " current_hidden=" << managed_buffers_.current_hidden
                  << " logits=" << managed_buffers_.logits);
    }

    // =========================================================================
    // Execution Methods
    // =========================================================================

    bool DeviceGraphOrchestrator::executeForward(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        // Enable device-scoped logging if not already set by caller (e.g., from forward())
        // This ensures executeForward() can be called directly with proper log attribution
        ScopedDeviceLog device_log(input.device);

        auto start = std::chrono::high_resolution_clock::now();

        // Token input OR activation input is required
        bool has_token_input = input.token_ids || input.batches;
        bool has_activation_input = external_hidden_state_input_ != nullptr;

        // For PP stages without embedding, activation input is required instead of tokens
        bool is_pp_middle_stage = pp_stage_config_.has_value() &&
                                  !pp_stage_config_.value().has_embedding;

        if (!has_token_input && !has_activation_input)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No token or activation input provided");
            return false;
        }

        if (is_pp_middle_stage && !has_activation_input)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] PP middle stage requires activation input "
                      "via setHiddenState()");
            return false;
        }

        if (input.seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid sequence length: " << input.seq_len);
            return false;
        }

        LOG_TRACE("[DeviceGraphOrchestrator] executeForward: batch_size=" << input.batch_size
                                                                          << ", seq_len=" << input.seq_len
                                                                          << ", device=" << input.device);

        // Build position IDs if not provided externally
        std::vector<int> position_ids_storage;
        Qwen2ForwardInput effective_input = input;

        if (!input.position_ids)
        {
            position_ids_storage = Qwen2Graph::buildPositionIds(
                input.seq_len, input.batch_size, input.position_offset);
            effective_input.position_ids = position_ids_storage.data();
        }

        // Pass external hidden state to graph builder input for PP middle/final stages
        if (external_hidden_state_input_)
        {
            effective_input.external_hidden_state = external_hidden_state_input_;
            LOG_DEBUG("[DeviceGraphOrchestrator] Using external hidden state input: "
                      << external_hidden_state_input_->numel() << " elements");

            // Clear for next invocation (single-use semantics)
            external_hidden_state_input_ = nullptr;
        }

        // =====================================================================
        // Decode Graph Cache: Reuse cached graph for decode mode (seq_len=1)
        // =====================================================================
        // During decode, the graph structure is identical between steps —
        // only token_ids, position_ids, and position_offset change.
        // Instead of rebuilding hundreds of stage objects every forward() call,
        // we cache the graph after the first decode step and reuse it.
        //
        // Benefits:
        // - Eliminates stage object construction/destruction (~100s of allocs)
        // - Preserves kernel caches in stages (JIT attention, RoPE inv_freq)
        // - Avoids workspace re-binding (bindWorkspace → inv_freq reset)
        // - Skips graph traversal in ensureDeviceWorkspaceAllocated()
        // =====================================================================

        bool is_decode = (effective_input.seq_len == 1 && effective_input.batch_size <= 1);
        bool is_standard_path = !pipeline_config_ && !pp_stage_config_.has_value();
        bool use_cached_forward = is_decode && is_standard_path && forward_cache_.valid && cache_config_.enabled;

        if (use_cached_forward)
        {
            // ===== CACHE HIT: Reuse cached decode graph =====

            // Update stable buffers — stages hold pointers to these, so the
            // pointed-to values change but the pointers remain valid
            forward_cache_.token_ids[0] = effective_input.token_ids[0];
            forward_cache_.position_ids[0] = effective_input.position_ids[0];

            // For GPU graph replay: set the capture stream on all stages ONCE.
            // The capture_stream never changes between decode steps, so after the
            // first pass we skip this 339-stage loop entirely.
            void *replay_stream = forward_cache_.segment_cache.capture_stream;
            if (replay_stream && !forward_cache_.gpu_stream_applied)
            {
                const auto &order = forward_cache_.graph->getExecutionOrder();
                for (const auto &node_name : order)
                {
                    ComputeNode *node = forward_cache_.graph->getNode(node_name);
                    if (node && node->stage)
                        node->stage->setGPUStream(replay_stream);
                }
                forward_cache_.gpu_stream_applied = true;
            }

            // Update position-dependent params using cached stage pointers.
            // Only ~4 stages override updateDynamicParams() — avoids iterating
            // all ~339 stages with hash lookups on every decode step.
            if (!forward_cache_.dynamic_param_stages_cached)
            {
                forward_cache_.dynamic_param_stages.clear();
                const auto &order = forward_cache_.graph->getExecutionOrder();
                for (const auto &node_name : order)
                {
                    ComputeNode *node = forward_cache_.graph->getNode(node_name);
                    if (node && node->stage && node->stage->hasDynamicParams())
                        forward_cache_.dynamic_param_stages.push_back(node->stage.get());
                }
                forward_cache_.dynamic_param_stages_cached = true;
            }
            for (auto *stage : forward_cache_.dynamic_param_stages)
            {
                stage->updateDynamicParams(effective_input.position_offset,
                                           effective_input.seq_len);
            }

            // Skip graph reset when Phase 3 replay is active — Phase 3 doesn't
            // call markCompleted(), so all flags are already false from last reset.
            if (!forward_cache_.phase3_active)
            {
                forward_cache_.graph->reset();
            }

            output = forward_cache_.output;

            // Execute with single device context (standard path, no PP)
            IDeviceContext *ctx = getDeviceContext(input.device);
            if (!ctx)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to get device context");
                return false;
            }

            bool success;
            const bool has_collective_nodes = !forward_cache_.collective_nodes.empty();
            const auto capture_policy = buildDecodeCapturePolicy(has_collective_nodes, ctx);
            if (capture_policy.collective_segmented_enabled)
            {
                LOG_INFO("[DeviceGraphOrchestrator] Experimental collective segmented GPU-graph replay enabled");
            }

            if (capture_policy.allow_segmented_capture && !forward_cache_.gpu_stream)
            {
                // Ensure backend factories are registered
                DeviceId dev_id = ctx->deviceId();
#ifdef HAVE_ROCM
                if (dev_id.is_rocm())
                    ensureAMDFactoryRegistered();
#endif
#ifdef HAVE_CUDA
                if (dev_id.is_cuda())
                    ensureNvidiaFactoryRegistered();
#endif

                auto &pool = GPUDeviceContextPool::instance();
                IWorkerGPUContext *gpu_ctx = nullptr;
                if (dev_id.is_rocm())
                {
                    gpu_ctx = &pool.getAMDContext(dev_id.rocm_ordinal());
                }
                else if (dev_id.is_cuda())
                {
                    gpu_ctx = &pool.getNvidiaContext(dev_id.cuda_ordinal());
                }
                if (gpu_ctx)
                {
                    forward_cache_.gpu_stream = gpu_ctx->defaultStream();
                    forward_cache_.gpu_ctx = gpu_ctx;
                }
            }

            bool used_segmented_capture = false;
            success = executor_.executeDecodeWithCapturePolicy(
                *forward_cache_.graph,
                ctx,
                &forward_cache_.segment_cache,
                forward_cache_.gpu_stream,
                forward_cache_.gpu_ctx,
                &forward_cache_.collective_nodes,
                capture_policy,
                &used_segmented_capture);

            if (success && used_segmented_capture &&
                forward_cache_.segment_cache.initialized &&
                !forward_cache_.segment_cache.needs_capture)
            {
                // Phase 3 replay doesn't call markCompleted(), so we can
                // skip graph.reset() on subsequent steps.
                forward_cache_.phase3_active = true;
            }
            else
            {
                forward_cache_.phase3_active = false;
            }

            // Sync the stream at the forward pass boundary so logits are
            // immediately available to the caller without per-access event waits.
            // This moves the synchronization point from the lazy data() call
            // (inside ensureOnHost) to here, eliminating coherence overhead
            // when the sampler reads logits.
            if (success)
            {
                syncLogitsAtBoundary(ctx);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

            LOG_TRACE("[DeviceGraphOrchestrator] Forward (cached decode) completed in "
                      << ms << "ms, success=" << success);

            return success;
        }

        // ===== CACHE MISS: Build new graph =====

        // Invalidate decode cache on phase transition (e.g., prefill → decode → prefill)
        if (forward_cache_.valid && (!is_decode || !is_standard_path))
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Invalidating forward graph cache "
                      << "(seq_len=" << effective_input.seq_len
                      << ", batch_size=" << effective_input.batch_size << ")");
            forward_cache_.invalidate();
        }

        // For the first decode step on the standard path: redirect token_ids and
        // position_ids to stable buffers so that cached stages' pointers survive.
        bool should_cache_after_build = is_decode && is_standard_path && cache_config_.enabled && !forward_cache_.valid;
        if (should_cache_after_build)
        {
            int total_tokens = effective_input.batch_size * effective_input.seq_len;
            forward_cache_.token_ids.assign(
                effective_input.token_ids,
                effective_input.token_ids + total_tokens);
            forward_cache_.position_ids.assign(
                effective_input.position_ids,
                effective_input.position_ids + total_tokens);

            // Redirect input to stable buffers before graph build
            effective_input.token_ids = forward_cache_.token_ids.data();
            effective_input.position_ids = forward_cache_.position_ids.data();
        }

        // Build forward graph via fluent builder API
        // Priority is auto-selected: unified PP > partial PP stage > full forward
        GraphBuildResult build_result = [&]() -> GraphBuildResult
        {
            // Start building the graph with input
            auto session = buildGraph()
                               .forInput(effective_input);

            // Add external hidden state for PP middle/final stages (if set above)
            // Note: external_hidden_state is already in effective_input.external_hidden_state
            // but the fluent API uses it via prepareInput()

            if (pipeline_config_ && pipeline_config_->hasPP())
            {
                // Unified PP+TP path: single graph spanning all PP stages
                LOG_DEBUG("[DeviceGraphOrchestrator] Building UNIFIED PIPELINE graph: "
                          << pipeline_config_->numStages() << " PP stages, "
                          << pipeline_config_->total_layers << " layers");

                // Initialize PP and TP contexts if needed
                if (!pp_contexts_initialized_ && !initializePPContexts())
                {
                    return GraphBuildResult("Failed to initialize PP contexts");
                }
                if (!tp_contexts_initialized_ && !initializeTPContexts())
                {
                    return GraphBuildResult("Failed to initialize TP contexts");
                }

                // Wire PP contexts to the session
                for (const auto &[key, ctx] : pp_contexts_)
                {
                    session.withPPContext(key.first, key.second, ctx.get());
                }

                // Wire TP contexts to the session
                for (const auto &[name, ctx] : domain_tp_contexts_)
                {
                    session.withTPContext(name, ctx.get());
                }

                return session
                    .withPipelineConfig(pipeline_config_)
                    .buildUnified();
            }
            else if (pp_stage_config_.has_value())
            {
                // Legacy single-stage PP path
                const auto &pp = pp_stage_config_.value();
                LOG_DEBUG("[DeviceGraphOrchestrator] Building PARTIAL forward graph: "
                          << "layers=[" << pp.first_layer << ", " << pp.last_layer << ") "
                          << "has_embedding=" << pp.has_embedding
                          << " has_lm_head=" << pp.has_lm_head);

                return session
                    .forPPStage(pp.first_layer, pp.last_layer, pp.has_embedding, pp.has_lm_head)
                    .buildPartial();
            }
            else
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Building FULL forward graph...");
                return session.buildForward();
            }
        }();

        if (!build_result)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Graph build failed: " << build_result.error());
            return false;
        }

        output = build_result.output();
        ComputeGraph graph = build_result.takeGraph();

        LOG_DEBUG("[DeviceGraphOrchestrator] Forward graph built with " << graph.size() << " stages");

        if (graph.size() == 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Empty forward graph");
            return false;
        }

        // Ensure GPU workspace is allocated for GEMM kernels (lazy initialization)
        ensureDeviceWorkspaceAllocated(graph);

        bool success = false;

        // Execution path depends on configuration:
        // - Unified PP: multi-device execution with all PP stage devices
        // - Single-device: standard single-context execution
        if (pipeline_config_ && pipeline_config_->hasPP())
        {
            // Build device contexts map for all devices in the pipeline
            std::unordered_map<DeviceId, IDeviceContext *> contexts;
            for (const auto &device : pipeline_config_->getAllDevices())
            {
                IDeviceContext *ctx = getDeviceContext(device);
                if (!ctx)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to get device context for " << device);
                    return false;
                }
                contexts[device] = ctx;
            }

            LOG_DEBUG("[DeviceGraphOrchestrator] Executing unified PP graph with "
                      << contexts.size() << " device contexts...");

            success = executor_.executeMultiDevice(graph, contexts);
        }
        else
        {
            // Get single device context
            LOG_DEBUG("[DeviceGraphOrchestrator] Getting device context for " << input.device << "...");
            IDeviceContext *ctx = getDeviceContext(input.device);
            if (!ctx)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to get device context");
                return false;
            }
            LOG_DEBUG("[DeviceGraphOrchestrator] Got device context, starting execution...");

            success = executor_.execute(graph, ctx);
        }

        // Sync the stream at the forward pass boundary (same as cached path above)
        if (success)
        {
            IDeviceContext *sync_ctx = getDeviceContext(input.device);
            if (sync_ctx)
            {
                syncLogitsAtBoundary(sync_ctx);
            }
        }

        // Cache the graph for future decode steps (first decode step only)
        if (should_cache_after_build && success)
        {
            forward_cache_.graph = std::make_unique<ComputeGraph>(std::move(graph));
            forward_cache_.output = output;
            // Pre-compute collective node set for fast decode intercept
            forward_cache_.collective_nodes.clear();
            for (const auto &n : forward_cache_.graph->getExecutionOrder())
            {
                auto *nd = forward_cache_.graph->getNode(n);
                if (nd && nd->stage)
                {
                    auto t = nd->stage->type();
                    if (t == ComputeStageType::ALLREDUCE ||
                        t == ComputeStageType::ALLGATHER ||
                        t == ComputeStageType::ALLGATHER_V)
                    {
                        forward_cache_.collective_nodes.insert(n);
                    }
                }
            }

            forward_cache_.valid = true;
            LOG_INFO("[DeviceGraphOrchestrator] Cached forward graph for decode mode ("
                     << forward_cache_.graph->size() << " stages)");
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        LOG_DEBUG("[DeviceGraphOrchestrator] Forward completed in " << ms << "ms, success=" << success);

        return success;
    }

    void DeviceGraphOrchestrator::syncLogitsAtBoundary(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            return; // CPU execution is synchronous — nothing to sync
        }

        // Single stream sync ensures ALL GPU work (all stages) is complete.
        // This replaces the lazy per-tensor hipEventSynchronize that would
        // otherwise fire inside ensureOnHost() when logits->data() is called.
        ctx->synchronize();

        // Clear the mapped sync flag so data()/fp32_data() return the mapped
        // pointer immediately without any further synchronization.
        if (state_.logits && state_.logits->isMapped())
        {
            state_.logits->markMappedSynced();
        }
    }

    DeviceGraphExecutor::DecodeCapturePolicy DeviceGraphOrchestrator::buildDecodeCapturePolicy(
        bool has_collective_nodes,
        IDeviceContext *ctx) const
    {
        DeviceGraphExecutor::DecodeCapturePolicy policy;

        const auto &env = debugEnv();
        policy.allow_fast_decode =
            env.execution.fast_decode &&
            !env.execution.executor_profiling &&
            !executor_.config().snapshot_callback;

        if (!policy.allow_fast_decode)
        {
            return policy;
        }

        const bool allow_collective_segmented = env.execution.gpu_graph_collective_segmented;
        bool collective_segmented_backend_supported = true;
        if (has_collective_nodes && allow_collective_segmented)
        {
            collective_segmented_backend_supported = collectivesSupportSegmentedReplay();
        }

        policy.collective_segmented_enabled =
            has_collective_nodes &&
            allow_collective_segmented &&
            collective_segmented_backend_supported;

        const bool can_use_segmented_graph =
            !has_collective_nodes ||
            policy.collective_segmented_enabled;

        policy.allow_segmented_capture =
            env.execution.gpu_graphs &&
            ctx && ctx->isGPU() &&
            can_use_segmented_graph &&
            forward_cache_.segment_cache.consecutive_failures < DeviceGraphExecutor::GraphSegmentCache::kMaxFailures;

        policy.max_segment_failures = DeviceGraphExecutor::GraphSegmentCache::kMaxFailures;
        return policy;
    }

    bool DeviceGraphOrchestrator::collectivesSupportSegmentedReplay() const
    {
        const auto &graph_cfg = graph_builder_->config();
        const bool has_local_tp = graph_cfg.local_tp_ctx && graph_cfg.local_tp_ctx->degree() > 1;
        const bool single_rank_collectives =
            injected_collective_ctx_ && injected_collective_ctx_->worldSize() == 1;

        if (!(has_local_tp && single_rank_collectives))
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Disabling collective segmented GPU-graph replay for cross-rank or non-local-TP collectives");
            return false;
        }

        const auto backend = graph_cfg.local_tp_ctx->backend();
        const bool supported =
            (backend == CollectiveBackendType::NCCL ||
             backend == CollectiveBackendType::RCCL);

        if (!supported)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Disabling collective segmented GPU-graph replay for non-stream backend");
        }

        return supported;
    }

    bool DeviceGraphOrchestrator::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        // Ensure GPU workspace is allocated for GEMM kernels
        ensureDeviceWorkspaceAllocated(graph);
        return executor_.execute(graph, ctx);
    }

    bool DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated(const ComputeGraph &graph)
    {
        const auto &config = graph_builder_->config();
        if (!buffer_manager_)
        {
            buffer_manager_ = std::make_unique<DeviceGraphBufferManager>(
                tensor_factory_, mpi_ctx_.get(), GraphBufferManagerConfig{});
        }

        WorkspaceSizingHints hints;
        hints.max_seq_len = config.max_seq_len > 0 ? config.max_seq_len : 4096;
        hints.n_heads = config.n_heads > 0 ? config.n_heads : 128;
        hints.head_dim = config.d_model > 0 && config.n_heads > 0
                             ? config.d_model / config.n_heads
                             : 128;
        hints.d_model = config.d_model > 0 ? config.d_model : 896;
        hints.batch_size = state_.batch_size > 0 ? state_.batch_size : 1;
        hints.vocab_size = config.vocab_size > 0 ? config.vocab_size : 151936;

        std::vector<WorkspaceConsumerRequest> extras;
        if (state_.kv_cache)
        {
            auto *kv_consumer = dynamic_cast<IWorkspaceConsumer *>(state_.kv_cache.get());
            if (kv_consumer && state_.device_id.is_gpu())
            {
                extras.push_back(WorkspaceConsumerRequest{
                    kv_consumer,
                    state_.device_id,
                    std::max(1, hints.batch_size),
                    0,
                    0,
                });
            }
        }

        WorkspaceBudgetConfig workspace_budget;
        return buffer_manager_->allocateDeviceWorkspaceForGraph(graph, hints, extras, workspace_budget);
    }

    bool DeviceGraphOrchestrator::executeAttention(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
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
        IDeviceContext *ctx = getDeviceContext(device);
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
                LOG_DEBUG("[DeviceGraphOrchestrator] Reusing cached attention graph for layer "
                          << layer_idx << " (pos_offset=" << pos_offset << ")");

                // Update dynamic parameters (position offset)
                updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);

                // Execute cached graph
                bool success = executor_.execute(*cache.attention_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Cached attention graph failed at layer " << layer_idx);
                }

                cache.attention_decode->reset();
                cache_stats_.attention_cache_hits++;
                return success;
            }

            // Build and cache the graph using fluent API
            LOG_DEBUG("[DeviceGraphOrchestrator] Building and caching attention graph for layer "
                      << layer_idx << " (decode mode)");

            auto result = buildAttentionGraph()
                              .forLayer(layer, layer_idx)
                              .withBuffers(buffers)
                              .withSequence(seq_len, 1)
                              .onDevice(device)
                              .withKVCache(kv_cache)
                              .withPositionIds(position_ids)
                              .build();

            if (!result)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Attention graph build failed: " << result.error());
                return false;
            }

            cache.attention_decode = std::make_unique<ComputeGraph>(result.takeGraph());
            cache.cached_seq_len = seq_len;
            cache.valid = true;
            cache_stats_.attention_cache_misses++;

            // Execute the newly built graph
            bool success = executor_.execute(*cache.attention_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Attention block failed at layer " << layer_idx);
            }

            cache.attention_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled) using fluent API
        // =============================================================================
        cache_stats_.attention_cache_misses++;

        auto result = buildAttentionGraph()
                          .forLayer(layer, layer_idx)
                          .withBuffers(buffers)
                          .withSequence(seq_len, 1)
                          .onDevice(device)
                          .withKVCache(kv_cache)
                          .withPositionIds(position_ids)
                          .build();

        if (!result)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Attention graph build failed: " << result.error());
            return false;
        }

        ComputeGraph graph = result.takeGraph();

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
            LOG_ERROR("[DeviceGraphOrchestrator] Attention block failed at layer " << layer_idx);
        }

        return success;
    }

    bool DeviceGraphOrchestrator::executeFFN(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        DeviceId device)
    {
        // Get device context
        IDeviceContext *ctx = getDeviceContext(device);
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
                LOG_DEBUG("[DeviceGraphOrchestrator] Reusing cached FFN graph for layer " << layer_idx);

                // Execute cached graph (no params to update for FFN)
                bool success = executor_.execute(*cache.ffn_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Cached FFN graph failed at layer " << layer_idx);
                }

                cache.ffn_decode->reset();
                cache_stats_.ffn_cache_hits++;
                return success;
            }

            // Build and cache the graph using fluent API
            LOG_DEBUG("[DeviceGraphOrchestrator] Building and caching FFN graph for layer "
                      << layer_idx << " (decode mode)");

            auto result = buildFFNGraph()
                              .forLayer(layer, layer_idx)
                              .withBuffers(buffers)
                              .withSequence(seq_len, 1)
                              .onDevice(device)
                              .build();

            if (!result)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] FFN graph build failed: " << result.error());
                return false;
            }

            cache.ffn_decode = std::make_unique<ComputeGraph>(result.takeGraph());
            cache_stats_.ffn_cache_misses++;

            // Execute the newly built graph
            bool success = executor_.execute(*cache.ffn_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] FFN block failed at layer " << layer_idx);
            }

            cache.ffn_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled) using fluent API
        // =============================================================================
        cache_stats_.ffn_cache_misses++;

        auto result = buildFFNGraph()
                          .forLayer(layer, layer_idx)
                          .withBuffers(buffers)
                          .withSequence(seq_len, 1)
                          .onDevice(device)
                          .build();

        if (!result)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] FFN graph build failed: " << result.error());
            return false;
        }

        ComputeGraph graph = result.takeGraph();

        bool success = executor_.execute(graph, ctx);

        if (!success)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] FFN block failed at layer " << layer_idx);
        }

        return success;
    }

    bool DeviceGraphOrchestrator::executeLayer(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        LOG_INFO("[DeviceGraphOrchestrator::executeLayer] LAYER_EXEC_ENTERED layer_idx="
                 << layer_idx << " seq_len=" << seq_len);

        // =====================================================================
        // Weight Streaming Hooks (Option B)
        // =====================================================================
        // Before layer execution: Ensure weights are on device
        // After layer execution: Release layer and prefetch next
        // =====================================================================
        if (weight_streamer_)
        {
            // Ensure this layer's weights are on the target device
            if (!weight_streamer_->ensureLayerOnDevice(layer_idx, device))
            {
                LOG_ERROR("[DeviceGraphOrchestrator::executeLayer] Failed to stream layer "
                          << layer_idx << " to device " << device.toString());
                return false;
            }

            // Prefetch next layer(s) asynchronously to overlap with compute
            int n_layers = graph_builder_ ? graph_builder_->config().n_layers : 0;
            if (layer_idx + 1 < n_layers)
            {
                weight_streamer_->prefetchLayer(layer_idx + 1, device);
                LOG_TRACE("[DeviceGraphOrchestrator::executeLayer] Prefetching layer " << (layer_idx + 1));
            }
        }

        // Execute attention block
        if (!executeAttention(layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device))
        {
            // Release layer on failure if streaming
            if (weight_streamer_)
            {
                weight_streamer_->releaseLayer(layer_idx);
            }
            return false;
        }

        // Execute FFN block
        if (!executeFFN(layer, buffers, layer_idx, seq_len, device))
        {
            // Release layer on failure if streaming
            if (weight_streamer_)
            {
                weight_streamer_->releaseLayer(layer_idx);
            }
            return false;
        }

        // After layer execution: Release layer (marks as eligible for eviction)
        if (weight_streamer_)
        {
            weight_streamer_->releaseLayer(layer_idx);
            LOG_TRACE("[DeviceGraphOrchestrator::executeLayer] Released layer " << layer_idx);
        }

        return true;
    }

    // =========================================================================
    // Cache Management
    // =========================================================================

    void DeviceGraphOrchestrator::clearCache()
    {
        // Clear graph caches
        for (auto &cache : layer_graph_cache_)
        {
            cache.invalidate();
        }

        // Clear forward graph cache
        forward_cache_.invalidate();

        // Clear device contexts
        device_contexts_.clear();

        // Reset state
        last_pos_offset_ = -1;

        // Reset stats
        cache_stats_ = CacheStats{};

        LOG_DEBUG("[DeviceGraphOrchestrator] All caches cleared");
    }

    void DeviceGraphOrchestrator::invalidateGraphCache(int layer_idx)
    {
        if (layer_idx < 0)
        {
            // Invalidate all layers
            for (auto &cache : layer_graph_cache_)
            {
                cache.invalidate();
            }
            cache_stats_.cached_layers = 0;
            LOG_DEBUG("[DeviceGraphOrchestrator] All layer graph caches invalidated");
        }
        else if (static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            layer_graph_cache_[layer_idx].invalidate();
            LOG_DEBUG("[DeviceGraphOrchestrator] Layer " << layer_idx << " graph cache invalidated");
        }
    }

    bool DeviceGraphOrchestrator::hasValidCachedGraph(int layer_idx, bool is_attention) const
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

    void DeviceGraphOrchestrator::setGraphCachingEnabled(bool enabled)
    {
        if (cache_config_.enabled != enabled)
        {
            cache_config_.enabled = enabled;
            if (!enabled)
            {
                invalidateGraphCache(-1);
            }
            LOG_INFO("[DeviceGraphOrchestrator] Graph caching "
                     << (enabled ? "enabled" : "disabled"));
        }
    }

    void DeviceGraphOrchestrator::initializeGraphCache(int n_layers)
    {
        layer_graph_cache_.resize(n_layers);
        cache_stats_.cached_layers = n_layers;
        LOG_DEBUG("[DeviceGraphOrchestrator] Graph cache initialized for " << n_layers << " layers");
    }

    // =========================================================================
    // Inference State Management (Phase 5)
    // =========================================================================

    bool DeviceGraphOrchestrator::initializeInferenceState(
        int batch_size,
        int max_seq_len,
        DeviceId device,
        const InferenceStateInitConfig &init_config)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot initialize state: no graph builder");
            return false;
        }

        const auto &config = graph_builder_->config();
        int d_model = config.d_model;
        int vocab_size = config.vocab_size;
        // For PP stages, use the stage's layer count (not full model) for KV cache allocation
        int n_layers = pp_stage_config_.has_value()
                           ? pp_stage_config_.value().layerCount()
                           : config.n_layers;
        int n_heads = config.n_heads;
        int n_kv_heads = config.n_kv_heads;
        int head_dim = config.head_dim;
        int d_ff = config.d_ff;

        // For tensor parallelism, use local head counts for buffer allocation
        // When qkv_column_parallel is enabled, each rank processes only its subset of heads
        int buffer_n_heads = config.qkv_column_parallel ? config.local_n_heads : n_heads;
        int buffer_n_kv_heads = config.qkv_column_parallel ? config.local_n_kv_heads : n_kv_heads;
        int buffer_d_ff = config.ffn_column_parallel ? config.d_ff_local : d_ff;

        LOG_DEBUG("[DeviceGraphOrchestrator] Initializing inference state: batch_size=" << batch_size
                                                                                        << " max_seq_len=" << max_seq_len
                                                                                        << " d_model=" << d_model
                                                                                        << " vocab_size=" << vocab_size);

        if (config.qkv_column_parallel)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Using local buffer sizes for TP: n_heads=" << buffer_n_heads
                                                                                            << "/" << n_heads << " n_kv_heads=" << buffer_n_kv_heads << "/" << n_kv_heads
                                                                                            << " d_ff=" << buffer_d_ff << "/" << d_ff);
        }

        // Create a default MPI context if none was provided
        std::shared_ptr<MPIContext> local_mpi_ctx = mpi_ctx_;
        if (!local_mpi_ctx)
        {
            // Create a single-rank MPI context for non-MPI usage
            local_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            LOG_DEBUG("[DeviceGraphOrchestrator] Created default single-rank MPI context");
        }

        // Create tensor factory
        TensorFactory factory(*local_mpi_ctx);

        // Enable mapped memory allocation for GPU tensors when requested
        // This enables zero-copy host access for all FP32 GPU tensors, avoiding slow memcpy syncs
        // Works for both CUDA and ROCm backends (cudaHostAllocMapped / hipHostMallocMapped)
        if (init_config.use_mapped_memory && device.is_gpu())
        {
            factory.setUseMappedMemoryForGPU(true);
            LOG_DEBUG("[DeviceGraphOrchestrator] Enabling mapped memory for ALL GPU tensors (zero-copy host access)");
        }

        // =====================================================================
        // BAR-Backed Allocation for LOCAL TP with PCIeBAR Backend
        // =====================================================================
        // When using LOCAL TP with PCIeBAR backend, ROCm devices need BAR-backed
        // tensors for row-parallel outputs (attn_proj, ffn_output). This enables
        // zero-copy allreduce where CUDA reads directly from ROCm's BAR region.
        //
        // Conditions for BAR-backed allocation:
        // 1. LOCAL TP is active (local_tp_ctx != nullptr && degree > 1)
        // 2. Backend is PCIeBAR
        // 3. Current device is ROCm
        // 4. DirectP2PEngine is available from LocalTPContext
        // =====================================================================
        bool use_bar_allocation = false;
        DeviceId cuda_device_for_bar; // CUDA device that will read from BAR
        DeviceId rocm_device_for_bar; // ROCm device that will own BAR memory

        if (config.local_tp_ctx &&
            config.local_tp_ctx->degree() > 1 &&
            config.local_tp_ctx->backend() == CollectiveBackendType::PCIE_BAR &&
            device.is_rocm())
        {
            // Get DirectP2PEngine from LocalTPContext
            auto p2p_engine = config.local_tp_ctx->getDirectP2PEngine();
            if (p2p_engine)
            {
                factory.setDirectP2P(p2p_engine);

                // Find the CUDA device in the LOCAL TP group
                for (const auto &dev : config.local_tp_ctx->devices())
                {
                    if (dev.toLocalDeviceId().is_cuda())
                    {
                        cuda_device_for_bar = dev.toLocalDeviceId();
                        break;
                    }
                }

                rocm_device_for_bar = device; // Current device is ROCm
                use_bar_allocation = cuda_device_for_bar.is_cuda();

                if (use_bar_allocation)
                {
                    LOG_INFO("[DeviceGraphOrchestrator] Enabled BAR-backed allocation for LOCAL TP PCIeBAR: "
                             << "ROCm=" << rocm_device_for_bar.toString()
                             << ", CUDA=" << cuda_device_for_bar.toString());
                }
                else
                {
                    LOG_WARN("[DeviceGraphOrchestrator] Cannot enable BAR allocation: no CUDA device in LOCAL TP group");
                }
            }
            else
            {
                LOG_WARN("[DeviceGraphOrchestrator] PCIeBAR backend but DirectP2PEngine not available - "
                         << "allreduce will fall back to staged transfers");
            }
        }

        // Get activation precision from config
        ActivationPrecision act_prec = config.activation_precision;
        LOG_DEBUG("[DeviceGraphOrchestrator] Activation buffer precision: " << activationPrecisionToString(act_prec));

        // Allocate core buffers (always FP32 - interface with embeddings/softmax)
        // =========================================================================
        // Hidden State: BAR-backed allocation for cross-vendor PP transfers
        // =========================================================================
        // When use_bar_backed_hidden is true and the device is ROCm, allocate hidden
        // state in PCIe BAR memory. This enables zero-copy reads from CUDA devices
        // during PP activation transfer (ROCm stage → CUDA stage).
        // =========================================================================
        if (init_config.use_bar_backed_hidden && device.is_rocm())
        {
            // Get DirectP2PEngine for BAR allocation (may be shared singleton)
            auto p2p = DirectP2PEngine::getSharedInstance();
            if (p2p && p2p->isPCIeBarActive())
            {
                // Set P2P on factory so it can create BAR-backed tensors
                factory.setDirectP2P(p2p);

                // Use cuda:0 as the default CUDA device for BAR allocation
                // In typical cross-vendor PP setups, there's usually one CUDA device
                DeviceId cuda_device_for_bar = DeviceId::cuda(0);

                try
                {
                    auto bar_backed_hidden = factory.createFP32BARBacked(
                        {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                        device, // ROCm device owns the BAR memory
                        cuda_device_for_bar);

                    if (bar_backed_hidden)
                    {
                        state_.hidden = std::move(bar_backed_hidden);
                        LOG_INFO("[DeviceGraphOrchestrator] Allocated BAR-backed hidden state for cross-vendor PP: "
                                 << "ROCm=" << device.toString() << ", CUDA=" << cuda_device_for_bar.toString());
                    }
                    else
                    {
                        LOG_WARN("[DeviceGraphOrchestrator] BAR-backed hidden allocation returned nullptr, "
                                 << "falling back to standard allocation");
                    }
                }
                catch (const std::exception &e)
                {
                    LOG_WARN("[DeviceGraphOrchestrator] BAR-backed hidden allocation failed: " << e.what()
                                                                                               << " - falling back to standard allocation");
                }
            }
            else
            {
                LOG_WARN("[DeviceGraphOrchestrator] use_bar_backed_hidden requested but PCIe BAR not active, "
                         << "falling back to standard allocation");
            }
        }

        // Standard allocation if BAR-backed not used or failed
        if (!state_.hidden)
        {
            state_.hidden = factory.createFP32(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                device);
        }

        // Logits ALWAYS use mapped memory on GPU devices (regardless of init_config)
        // because logits are returned to the caller for sampling - they must be host-accessible.
        // Mapped memory avoids the slow synchronous hipMemcpy/cudaMemcpy in ensureOnHost().
        if (device.is_gpu())
        {
            state_.logits = FP32Tensor::createMapped(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(vocab_size)},
                device);
            LOG_INFO("[DeviceGraphOrchestrator] Allocated logits with mapped memory (zero-copy for sampling)");
        }
        else
        {
            state_.logits = factory.createFP32(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(vocab_size)},
                device);
        }

        // Phase 5: Allocate local logits buffer for column-parallel LM head
        if (config.lm_head_column_parallel && config.vocab_local > 0)
        {
            state_.logits_local = factory.createFP32(
                std::vector<size_t>{static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(config.vocab_local)},
                device);
            LOG_DEBUG("[DeviceGraphOrchestrator] Allocated logits_local buffer: ["
                      << batch_size * max_seq_len << ", " << config.vocab_local << "]");
        }

        // Allocate norm buffer (FP32 - output of RMSNorm for GEMM input)
        state_.normalized = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device);

        // Allocate residual buffer based on activation precision mode
        // HybridQ16 uses Q16_1 for 266× better precision than Q8_1 in the residual stream
        if (act_prec == ActivationPrecision::HybridQ16)
        {
            LOG_INFO("[DeviceGraphOrchestrator] Using Q16_1 residual stream for HybridQ16 mode");
            state_.residual = factory.createQ16_1(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                device);
        }
        else
        {
            state_.residual = factory.createFP32(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                device);
        }

        // QKV buffers - use per-projection precision for HybridQ16 mode
        // HybridQ16: K is Q16_1 (256× better precision), Q and V are Q8_1
        // This fixes the K precision loss where Q8_1 zeros out small values
        ActivationPrecision q_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::Q_GEMM_Output, nullptr);
        ActivationPrecision k_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::K_GEMM_Output, nullptr);
        ActivationPrecision v_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::V_GEMM_Output, nullptr);

        LOG_DEBUG("[DeviceGraphOrchestrator] QKV GEMM output precision: Q=" << activationPrecisionToString(q_prec)
                                                                            << " K=" << activationPrecisionToString(k_prec)
                                                                            << " V=" << activationPrecisionToString(v_prec));

        state_.Q = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_heads * head_dim)},
            q_prec, device);
        // K buffer: Q16_1 for HybridQ16 mode to preserve small values
        // Pass head_dim for optimal Q16 block size (1 block per head)
        state_.K = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_kv_heads * head_dim)},
            k_prec, head_dim, device);
        state_.V = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_kv_heads * head_dim)},
            v_prec, device);

        // Hybrid/HybridQ16 mode: Allocate separate FP32 buffers for Q/K after RoPE
        // This eliminates the requantization step in RoPE
        // Also allocate V_dequant buffer for KV cache append (V doesn't go through RoPE
        // but needs to be converted to KV cache precision for storage)
        if (act_prec == ActivationPrecision::Hybrid || act_prec == ActivationPrecision::HybridQ16)
        {
            // Use resolveBufferPrecision to get the correct precision for Q/K after RoPE
            // Hybrid: FP32, HybridQ16: Q16_1
            ActivationPrecision q_rope_prec = resolveBufferPrecision(
                act_prec, HybridBufferType::Q_After_RoPE, nullptr);
            ActivationPrecision k_rope_prec = resolveBufferPrecision(
                act_prec, HybridBufferType::K_After_RoPE, nullptr);
            LOG_DEBUG("[DeviceGraphOrchestrator] " << activationPrecisionToString(act_prec)
                                                   << " mode: allocating Q_rope buffer ("
                                                   << activationPrecisionToString(q_rope_prec) << ")");
            // Pass head_dim for Q16 block size selection (must match KV cache block size)
            state_.Q_rope = factory.createActivation(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_heads * head_dim)},
                q_rope_prec, head_dim, device);
            LOG_DEBUG("[DeviceGraphOrchestrator] " << activationPrecisionToString(act_prec)
                                                   << " mode: allocating K_rope buffer ("
                                                   << activationPrecisionToString(k_rope_prec) << ")");
            state_.K_rope = factory.createActivation(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_kv_heads * head_dim)},
                k_rope_prec, head_dim, device);

            // V_dequant: buffer for V before KV cache append
            // V is Q8_1 from GEMM, needs to match KV cache precision (FP32 for Hybrid, Q16_1 for HybridQ16)
            ActivationPrecision kv_cache_prec = resolveBufferPrecision(
                act_prec, HybridBufferType::KV_Cache, nullptr);
            kv_cache_prec = resolveKVCacheStoragePrecision(config.kv_cache_precision, kv_cache_prec);
            state_.V_dequant = factory.createActivation(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_kv_heads * head_dim)},
                kv_cache_prec, head_dim, device);
            LOG_DEBUG("[DeviceGraphOrchestrator] " << activationPrecisionToString(act_prec)
                                                   << " mode: allocating V_dequant buffer ("
                                                   << activationPrecisionToString(kv_cache_prec) << ")");

            // HybridQ16 K precision fix: allocate per-head K scales buffer
            // This stores dynamic scales from RoPE Q16→Q16 path for attention kernel
            if (act_prec == ActivationPrecision::HybridQ16)
            {
                const size_t k_head_scales_size = static_cast<size_t>(batch_size * max_seq_len * buffer_n_kv_heads);
                state_.K_head_scales.resize(k_head_scales_size, 1.0f);
                LOG_DEBUG("[DeviceGraphOrchestrator] HybridQ16 K precision fix: allocating K_head_scales buffer ("
                          << k_head_scales_size << " floats, "
                          << k_head_scales_size * sizeof(float) / 1024 << " KB)");
            }
        }

        // Attention output buffer
        // For Hybrid mode: attention context is FP32 (from softmax × V)
        ActivationPrecision attn_ctx_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::Attention_Context, nullptr);
        state_.attn_output = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_heads * head_dim)},
            attn_ctx_prec, device);

        // attn_proj is the output of Wo projection which feeds into the residual stream
        // HybridQ16 mode: Q8_1 (native add to Q16_1 residual via q16_1_add_q8_1)
        // Other modes: FP32 for numerical stability in residual connections
        //
        // For LOCAL TP with PCIeBAR backend: ROCm device uses BAR-backed FP32 tensor
        // to enable zero-copy allreduce (CUDA reads directly from ROCm BAR region)
        if (act_prec == ActivationPrecision::HybridQ16)
        {
            // Q8_1 cannot be BAR-backed (BAR allocation is FP32 only)
            state_.attn_proj = factory.createQ8_1(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                device);
        }
        else if (use_bar_allocation)
        {
            // BAR-backed FP32 for PCIeBAR zero-copy allreduce
            state_.attn_proj = factory.createFP32BARBacked(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                rocm_device_for_bar, cuda_device_for_bar);
            LOG_INFO("[DeviceGraphOrchestrator] Allocated attn_proj as BAR-backed FP32 for PCIeBAR allreduce");
        }
        else
        {
            state_.attn_proj = factory.createFP32(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                device);
        }

        // FFN buffers - gate and up are kept FP32 to avoid triple quantization in SwiGLU
        ActivationPrecision gate_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::FFN_Gate, nullptr);
        state_.gate = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_d_ff)},
            gate_prec, device);
        ActivationPrecision up_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::FFN_Up, nullptr);
        state_.up = factory.createActivation(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_d_ff)},
            up_prec, device);
        // ffn_output is the output of FFN Down projection which feeds into the residual stream
        // HybridQ16 mode: Q8_1 (native add to Q16_1 residual via q16_1_add_q8_1)
        // Other modes: FP32 for numerical stability in residual connections
        //
        // For LOCAL TP with PCIeBAR backend: ROCm device uses BAR-backed FP32 tensor
        // to enable zero-copy allreduce (CUDA reads directly from ROCm BAR region)
        if (act_prec == ActivationPrecision::HybridQ16)
        {
            // Q8_1 cannot be BAR-backed (BAR allocation is FP32 only)
            state_.ffn_output = factory.createQ8_1(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                device);
        }
        else if (use_bar_allocation)
        {
            // BAR-backed FP32 for PCIeBAR zero-copy allreduce
            state_.ffn_output = factory.createFP32BARBacked(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                rocm_device_for_bar, cuda_device_for_bar);
            LOG_INFO("[DeviceGraphOrchestrator] Allocated ffn_output as BAR-backed FP32 for PCIeBAR allreduce");
        }
        else
        {
            state_.ffn_output = factory.createFP32(
                {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
                device);
        }

        // Attention workspace - use local head counts for tensor parallelism
        // scores: [batch_size * buffer_n_heads, max_seq_len, max_seq_len]
        state_.workspace_scores = factory.createFP32(
            {static_cast<size_t>(batch_size * buffer_n_heads * max_seq_len), static_cast<size_t>(max_seq_len)},
            device);
        // context: [batch_size * buffer_n_heads, max_seq_len, head_dim]
        state_.workspace_context = factory.createFP32(
            {static_cast<size_t>(batch_size * buffer_n_heads * max_seq_len), static_cast<size_t>(head_dim)},
            device);
        // mask: [batch_size, max_seq_len, max_seq_len]
        state_.workspace_mask = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(max_seq_len)},
            device);

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        // Allocate context snapshot buffer for debugging attention
        // Shape: [batch_size * max_seq_len, buffer_n_heads * head_dim]
        state_.context_snapshot = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(buffer_n_heads * head_dim)},
            device);
        LOG_DEBUG("[DeviceGraphOrchestrator] Allocated context_snapshot buffer: ["
                  << batch_size * max_seq_len << ", " << buffer_n_heads * head_dim << "]");

        // Allocate attention output snapshot buffer (Wo projection result, before residual)
        // Shape: [batch_size * max_seq_len, d_model]
        state_.attention_output_snapshot = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device);
        LOG_DEBUG("[DeviceGraphOrchestrator] Allocated attention_output_snapshot buffer: ["
                  << batch_size * max_seq_len << ", " << d_model << "]");

        // Allocate attention residual snapshot buffer (after residual add)
        // Shape: [batch_size * max_seq_len, d_model]
        state_.attention_residual_snapshot = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device);
        LOG_DEBUG("[DeviceGraphOrchestrator] Allocated attention_residual_snapshot buffer: ["
                  << batch_size * max_seq_len << ", " << d_model << "]");
#endif

        // Create KV cache - use sharded cache for tensor parallelism
        // Check if tensor parallelism is enabled (indicated by local_n_kv_heads set)
        // For Hybrid mode: KV cache uses BF16 (better than Q8_1, 2x compression)
        ActivationPrecision kv_cache_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::KV_Cache, nullptr);
        kv_cache_prec = resolveKVCacheStoragePrecision(config.kv_cache_precision, kv_cache_prec);
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache precision: " << activationPrecisionToString(kv_cache_prec));
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache precision mode: "
                  << kvCachePrecisionToString(config.kv_cache_precision));

        // Determine KV cache layout mode:
        // - Q16_1 precision requires HEAD_MAJOR layout for Q16IntegerAttention kernel
        // - Other precisions use POSITION_MAJOR (legacy layout)
        KVCacheLayoutMode kv_layout_mode = (kv_cache_prec == ActivationPrecision::Q16_1)
                                               ? KVCacheLayoutMode::HEAD_MAJOR
                                               : KVCacheLayoutMode::POSITION_MAJOR;
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache layout mode: "
                  << (kv_layout_mode == KVCacheLayoutMode::HEAD_MAJOR ? "HEAD_MAJOR" : "POSITION_MAJOR"));

        // Set sharding parameters if needed (tensor parallelism)
        // Sharding is needed when local_n_kv_heads < n_kv_heads, AND tensor parallelism is active.
        // TP can be:
        // - GLOBAL TP: Multiple MPI ranks (mpi_ctx_->world_size() > 1)
        // - LOCAL TP: Multiple devices within single rank (local_tp_ctx->degree() > 1)
        bool use_sharded_cache = (config.local_n_kv_heads > 0 && config.local_n_kv_heads < n_kv_heads);
        bool is_global_tp = mpi_ctx_ && mpi_ctx_->world_size() > 1;
        bool is_local_tp = config.local_tp_ctx && config.local_tp_ctx->degree() > 1;

        // =====================================================================
        // KV Cache Creation: Per-stage for PP, single for non-PP
        // =====================================================================
        if (pipeline_config_ && pipeline_config_->hasPP())
        {
            // Pipeline Parallelism: Create a KV cache for each PP stage's device
            // Each cache only stores the layers processed by that stage
            LOG_DEBUG("[DeviceGraphOrchestrator] Creating per-stage KV caches for PP ("
                      << pipeline_config_->numStages() << " stages)");

            for (const auto &pp_stage : pipeline_config_->pp_stages)
            {
                // Get device for this stage
                const TPDomainConfig *domain = pipeline_config_->getDomainForStage(pp_stage.stage_id);
                if (!domain)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] No domain for PP stage " << pp_stage.stage_id);
                    return false;
                }
                DeviceId stage_device = domain->primaryDevice();

                // Skip if we already have a cache for this device
                // (multiple stages on same device share one cache)
                if (state_.pp_kv_caches.find(stage_device) != state_.pp_kv_caches.end())
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Reusing existing KV cache for device "
                              << stage_device.to_string());
                    continue;
                }

                // Count layers on this device
                int layers_on_device = 0;
                int first_layer_on_device = -1;
                for (const auto &stage : pipeline_config_->pp_stages)
                {
                    const TPDomainConfig *stage_domain = pipeline_config_->getDomainForStage(stage.stage_id);
                    if (stage_domain && stage_domain->primaryDevice() == stage_device)
                    {
                        int stage_layers = stage.last_layer - stage.first_layer;
                        if (first_layer_on_device < 0)
                            first_layer_on_device = stage.first_layer;
                        layers_on_device += stage_layers;
                    }
                }

                // Build KVCacheConfig for this stage
                llaminar::v2::kernels::KVCacheConfig kv_config;
                kv_config.precision = kv_cache_prec;
                kv_config.device = stage_device;
                kv_config.num_layers = layers_on_device;
                kv_config.first_layer_index = first_layer_on_device; // Layer index offset
                kv_config.batch_size = batch_size;
                kv_config.max_seq_len = max_seq_len;
                kv_config.n_kv_heads = n_kv_heads;
                kv_config.head_dim = head_dim;
                kv_config.layout_mode = kv_layout_mode;
                kv_config.mpi_ctx = local_mpi_ctx.get();

                if (use_sharded_cache && (is_global_tp || is_local_tp))
                {
                    kv_config.local_n_kv_heads = config.local_n_kv_heads;
                    if (is_local_tp)
                    {
                        kv_config.kv_head_start = config.local_tp_device_idx * config.local_n_kv_heads;
                    }
                    else
                    {
                        kv_config.kv_head_start = mpi_ctx_->rank() * config.local_n_kv_heads;
                    }
                }

                LOG_DEBUG("[DeviceGraphOrchestrator] Creating KV cache for PP stage device "
                          << stage_device.to_string() << ": layers [" << first_layer_on_device
                          << ", " << (first_layer_on_device + layers_on_device) << "), "
                          << layers_on_device << " layers, precision="
                          << activationPrecisionToString(kv_cache_prec));

                state_.pp_kv_caches[stage_device] =
                    llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);

                if (!state_.pp_kv_caches[stage_device])
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to create KV cache for device "
                              << stage_device.to_string());
                    return false;
                }
            }

            LOG_DEBUG("[DeviceGraphOrchestrator] Created " << state_.pp_kv_caches.size()
                                                           << " per-device KV caches for PP");
        }
        else
        {
            // Non-PP: Single KV cache for all layers
            llaminar::v2::kernels::KVCacheConfig kv_config;
            kv_config.precision = kv_cache_prec;
            kv_config.device = device;
            kv_config.num_layers = n_layers;
            kv_config.batch_size = batch_size;
            kv_config.max_seq_len = max_seq_len;
            kv_config.n_kv_heads = n_kv_heads;
            kv_config.head_dim = head_dim;
            kv_config.layout_mode = kv_layout_mode;
            kv_config.mpi_ctx = local_mpi_ctx.get();

            if (use_sharded_cache && (is_global_tp || is_local_tp))
            {
                kv_config.local_n_kv_heads = config.local_n_kv_heads;

                // Calculate kv_head_start based on TP mode:
                // - LOCAL TP: Use device index within the LOCAL TP context
                // - GLOBAL TP: Use MPI rank
                if (is_local_tp)
                {
                    kv_config.kv_head_start = config.local_tp_device_idx * config.local_n_kv_heads;
                    LOG_DEBUG("[DeviceGraphOrchestrator] Creating sharded KV cache (LOCAL TP): "
                              << n_kv_heads << " total KV heads, "
                              << config.local_n_kv_heads << " local KV heads (device_idx="
                              << config.local_tp_device_idx << ", start=" << kv_config.kv_head_start << ")"
                              << " precision=" << activationPrecisionToString(kv_cache_prec));
                }
                else
                {
                    kv_config.kv_head_start = mpi_ctx_->rank() * config.local_n_kv_heads;
                    LOG_DEBUG("[DeviceGraphOrchestrator] Creating sharded KV cache (GLOBAL TP): "
                              << n_kv_heads << " total KV heads, "
                              << config.local_n_kv_heads << " local KV heads (rank="
                              << mpi_ctx_->rank() << ", start=" << kv_config.kv_head_start << ")"
                              << " precision=" << activationPrecisionToString(kv_cache_prec));
                }
            }

            // Create cache via factory (handles sharded vs non-sharded automatically)
            state_.kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);
        }

        // Initialize position tracking
        state_.positions.assign(batch_size, 0);
        state_.sequence_lengths.assign(batch_size, 0);

        // Store config
        state_.batch_size = batch_size;
        state_.max_seq_len = max_seq_len;
        state_.d_model = d_model;
        state_.vocab_size = vocab_size;
        state_.device_id = device;

        LOG_DEBUG("[DeviceGraphOrchestrator] Inference state initialized successfully");
        return true;
    }

    const float *DeviceGraphOrchestrator::forward(
        const int *tokens,
        int seq_len,
        int batch_size)
    {
        // Enable device-scoped logging for this execution
        // All LOG_* calls from this thread will include the device ID
        ScopedDeviceLog device_log(state_.device_id);

        if (!state_.isInitialized())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward() called without initialized state");
            return nullptr;
        }

        if (!hasGlobalWeights())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward() called without global weights set");
            return nullptr;
        }

        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batch size " << batch_size
                                                              << " exceeds initialized batch size " << state_.batch_size);
            return nullptr;
        }

        int total_tokens = batch_size * seq_len;
        if (total_tokens > state_.batch_size * state_.max_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Total tokens " << total_tokens
                                                                << " exceeds buffer capacity "
                                                                << state_.batch_size * state_.max_seq_len);
            return nullptr;
        }

        // =====================================================================
        // Gap 4: Automatic phase transition based on sequence length
        // =====================================================================
        // - seq_len > 1: PREFILL phase (processing prompt, compute-bound)
        // - seq_len == 1: DECODE phase (generating tokens, bandwidth-bound)
        //
        // This affects weight selection via getPhaseAwareWeight():
        // - PREFILL: Full weights on GPU (compute-bound - all weights needed)
        // - DECODE: May use CPU decode shards for parallel execution
        // =====================================================================
        InferencePhase new_phase = (seq_len > 1) ? InferencePhase::PREFILL : InferencePhase::DECODE;
        transitionToPhase(new_phase);

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

        // Hybrid mode buffers: FP32 Q/K after RoPE, V_dequant for KV cache
        model_buffers.layer_buffers.Q_rope = state_.Q_rope.get();
        model_buffers.layer_buffers.K_rope = state_.K_rope.get();
        model_buffers.layer_buffers.V_dequant = state_.V_dequant.get();

        // HybridQ16 K precision fix: per-head K scales from RoPE Q16→Q16
        if (!state_.K_head_scales.empty())
        {
            model_buffers.layer_buffers.K_head_scales = state_.K_head_scales.data();
            model_buffers.layer_buffers.K_head_scales_capacity = state_.K_head_scales.size();
        }

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        // Pass context snapshot buffer for attention debugging
        model_buffers.layer_buffers.context_snapshot = state_.context_snapshot.get();
        // Pass attention output snapshot buffer (Wo projection, before residual)
        model_buffers.layer_buffers.attention_output_snapshot = state_.attention_output_snapshot.get();
        // Pass attention residual snapshot buffer (after residual add)
        model_buffers.layer_buffers.attention_residual_snapshot = state_.attention_residual_snapshot.get();
#endif

        setBuffers(model_buffers);

        // Build forward input
        Qwen2ForwardInput input;
        input.token_ids = tokens;
        input.position_ids = position_ids.data();
        input.batch_size = batch_size;
        input.seq_len = seq_len;
        input.position_offset = state_.positions[0]; // Legacy compat
        input.device = state_.device_id;
        input.kv_cache = state_.kv_cache.get();

        // For PP mode: build raw pointer map from per-device KV caches
        std::unordered_map<DeviceId, IKVCache *> pp_kv_cache_ptrs;
        if (!state_.pp_kv_caches.empty())
        {
            for (const auto &[device, cache] : state_.pp_kv_caches)
            {
                pp_kv_cache_ptrs[device] = cache.get();
            }
            input.pp_kv_caches = &pp_kv_cache_ptrs;
            LOG_DEBUG("[DeviceGraphOrchestrator] Set " << pp_kv_cache_ptrs.size()
                                                       << " per-device KV caches for PP forward");
        }

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
            LOG_ERROR("[DeviceGraphOrchestrator] forward() execution failed");
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

    const float *DeviceGraphOrchestrator::logits() const
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

    bool DeviceGraphOrchestrator::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        // Enable device-scoped logging for this execution
        ScopedDeviceLog device_log(state_.device_id);

        if (token_batches.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward_batch() called with empty batch");
            return false;
        }

        int batch_size = static_cast<int>(token_batches.size());
        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batch size " << batch_size
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

        // Store actual lengths BEFORE calling forward (forward() will overwrite with padded len)
        std::vector<int> actual_lengths(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            actual_lengths[i] = static_cast<int>(token_batches[i].size());
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
        // Note: forward() will set sequence_lengths[b] = padded_seq_len for all b
        const float *result = forward(flat_tokens.data(), padded_seq_len_, batch_size);

        // Restore actual sequence lengths (forward() set them to padded_seq_len)
        // This is important for:
        // 1. Proper logits extraction (only extract non-padded logits)
        // 2. Snapshot comparison (shapes should match actual token count)
        // 3. KV cache position tracking (only actual tokens contribute to cache)
        state_.sequence_lengths.resize(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            state_.sequence_lengths[i] = actual_lengths[i];
        }

        return result != nullptr;
    }

    const float *DeviceGraphOrchestrator::getLogits(int seq_idx) const
    {
        if (!state_.logits)
        {
            return nullptr;
        }

        if (seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid sequence index " << seq_idx
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

    int DeviceGraphOrchestrator::getPosition(int seq_idx) const
    {
        if (seq_idx < 0 || static_cast<size_t>(seq_idx) >= state_.positions.size())
        {
            return 0;
        }
        return state_.positions[seq_idx];
    }

    void DeviceGraphOrchestrator::clearInferenceState()
    {
        state_.clear();
        forward_cache_.invalidate();
        LOG_DEBUG("[DeviceGraphOrchestrator] Inference state cleared");
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    void DeviceGraphOrchestrator::updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len)
    {
        // Update all stages in the graph that have dynamic parameters
        const auto &order = graph.getExecutionOrder();

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
                continue;

            // Update dynamic params (pos_offset, seq_len)
            // Only stages that override updateDynamicParams will actually do anything
            node->stage->updateDynamicParams(pos_offset, seq_len);
        }

        LOG_TRACE("[DeviceGraphOrchestrator] Updated cached graph params: pos_offset="
                  << pos_offset << " seq_len=" << seq_len);
    }

    bool DeviceGraphOrchestrator::canUseCachedGraph(int layer_idx, int seq_len) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        return cache.valid && cache.cached_seq_len == seq_len;
    }

    // =========================================================================
    // Phase-Aware Weight Access (Gap 3 - CPU Decode Participation)
    // =========================================================================

    void DeviceGraphOrchestrator::setWeightManager(std::shared_ptr<WeightManager> weight_manager)
    {
        weight_manager_ = std::move(weight_manager);
        LOG_DEBUG("[DeviceGraphOrchestrator] WeightManager set");
    }

    void DeviceGraphOrchestrator::setWeightPlacementMap(std::shared_ptr<WeightPlacementMap> placement_map)
    {
        weight_placement_map_ = std::move(placement_map);
        LOG_DEBUG("[DeviceGraphOrchestrator] WeightPlacementMap set");
    }

    // =========================================================================
    // Tensor Parallel Configuration (Phase 1c: Proportional TP)
    // =========================================================================

    void DeviceGraphOrchestrator::setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config)
    {
        tp_config_ = std::move(config);

        if (tp_config_)
        {
            LOG_INFO("[DeviceGraphOrchestrator] TensorParallelConfig set: "
                     << "world_size=" << tp_config_->worldSize()
                     << ", proportional=" << (tp_config_->isProportional() ? "yes" : "no"));

            // If we have a graph builder, propagate the config to it
            if (graph_builder_)
            {
                // Note: The graph builder's config is read-only after construction,
                // but we store the tp_config for use in buffer allocation and KV cache creation
                LOG_DEBUG("[DeviceGraphOrchestrator] TensorParallelConfig will be used for buffer sizing");
            }
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] TensorParallelConfig cleared");
        }
    }

    // =========================================================================
    // Multi-Domain Tensor Parallel Configuration (Phase 6.3: Heterogeneous TP)
    // =========================================================================

    void DeviceGraphOrchestrator::setDomainConfig(std::shared_ptr<MultiDomainTPConfig> config)
    {
        domain_config_ = std::move(config);

        if (domain_config_)
        {
            LOG_INFO("[DeviceGraphOrchestrator] MultiDomainTPConfig set: "
                     << "domains=" << domain_config_->domains().size()
                     << ", has_gpu=" << (domain_config_->gpuDomain() ? "yes" : "no")
                     << ", has_cpu=" << (domain_config_->cpuDomain() ? "yes" : "no")
                     << ", cross_rank=" << (domain_config_->hasCrossRankTP() ? "yes" : "no"));

            // Propagate domain config to graph builder if present
            if (graph_builder_)
            {
                // Graph builder can access domain config through config_.multi_domain_tp_config
                // Note: The graph builder uses getDomainForLayer() which delegates to this config
                LOG_DEBUG("[DeviceGraphOrchestrator] Domain config available for AllreduceStage routing");
            }
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] MultiDomainTPConfig cleared (legacy MPI path)");
        }
    }

    // =========================================================================
    // Pipeline Parallelism Configuration
    // =========================================================================

    void DeviceGraphOrchestrator::setPPStageConfig(const FactoryPPStageConfig &config)
    {
        if (!config.isValid())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid FactoryPPStageConfig: "
                      << "first_layer=" << config.first_layer
                      << ", last_layer=" << config.last_layer);
            throw std::invalid_argument("Invalid FactoryPPStageConfig");
        }

        pp_stage_config_ = config;

        LOG_INFO("[DeviceGraphOrchestrator] PP stage configured: "
                 << "layers=[" << config.first_layer << ", " << config.last_layer << ") "
                 << "has_embedding=" << (config.has_embedding ? "yes" : "no")
                 << " has_lm_head=" << (config.has_lm_head ? "yes" : "no"));
    }

    // =========================================================================
    // Hidden State API (for Pipeline Parallelism)
    // =========================================================================

    TensorBase *DeviceGraphOrchestrator::getHiddenState()
    {
        if (!state_.hidden)
        {
            LOG_WARN("[DeviceGraphOrchestrator] getHiddenState: no hidden state available");
            return nullptr;
        }
        return state_.hidden.get();
    }

    const TensorBase *DeviceGraphOrchestrator::getHiddenState() const
    {
        return state_.hidden.get();
    }

    void DeviceGraphOrchestrator::setHiddenState(TensorBase *hidden_state)
    {
        external_hidden_state_input_ = hidden_state;
        LOG_DEBUG("[DeviceGraphOrchestrator] setHiddenState: "
                  << (hidden_state ? "set" : "cleared")
                  << " external hidden state input");
    }

    bool DeviceGraphOrchestrator::hasHiddenStateInput() const
    {
        return external_hidden_state_input_ != nullptr;
    }

    void DeviceGraphOrchestrator::clearHiddenStateInput()
    {
        external_hidden_state_input_ = nullptr;
    }

    const TPDomain *DeviceGraphOrchestrator::getDomainForLayer(int layer_idx, bool is_attention) const
    {
        if (!domain_config_)
        {
            return nullptr; // No domain config - use legacy MPI path
        }

        const TPDomain *domain = domain_config_->domainForLayer(layer_idx, is_attention);

        LOG_DEBUG("[DeviceGraphOrchestrator] getDomainForLayer: layer=" << layer_idx
                                                                        << ", is_attention=" << (is_attention ? "true" : "false")
                                                                        << " -> domain=" << (domain ? domain->name : "nullptr"));

        return domain;
    }

    void DeviceGraphOrchestrator::transitionToPhase(InferencePhase phase)
    {
        if (current_phase_ != phase)
        {
            InferencePhase old_phase = current_phase_;
            current_phase_ = phase;

            LOG_DEBUG("[DeviceGraphOrchestrator] Phase transition: " << toString(old_phase)
                                                                     << " -> " << toString(phase));

            // Notify weight streamer of phase transition (if streaming is enabled)
            if (weight_streamer_)
            {
                weight_streamer_->onPhaseTransition(old_phase, phase);
                LOG_DEBUG("[DeviceGraphOrchestrator] Weight streamer notified of phase transition");
            }
        }
    }

    // =========================================================================
    // Weight Streaming (Option B)
    // =========================================================================

    void DeviceGraphOrchestrator::setWeightStreamer(std::shared_ptr<IWeightStreamer> streamer)
    {
        weight_streamer_ = std::move(streamer);
        if (weight_streamer_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Weight streaming enabled");
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Weight streaming disabled");
        }
    }

    bool DeviceGraphOrchestrator::isWeightStreamingEnabled() const
    {
        return weight_streamer_ != nullptr;
    }

    void DeviceGraphOrchestrator::setCollectiveContext(std::shared_ptr<ICollectiveContext> collective_ctx)
    {
        injected_collective_ctx_ = std::move(collective_ctx);
        if (injected_collective_ctx_)
        {
            // Wire to executor for GPU-native collective interception
            executor_.setCollectiveContext(injected_collective_ctx_.get());
            LOG_INFO("[DeviceGraphOrchestrator] GPU-native collectives enabled via CollectiveContext");
        }
        else
        {
            executor_.setCollectiveContext(nullptr);
            LOG_DEBUG("[DeviceGraphOrchestrator] CollectiveContext cleared - using CPU MPI fallback");
        }
    }

    std::shared_ptr<TensorBase> DeviceGraphOrchestrator::getPhaseAwareWeight(
        const std::string &name,
        int layer_idx,
        InferencePhase phase) const
    {
        if (!weight_manager_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator::getPhaseAwareWeight] WeightManager not set");
            return nullptr;
        }

        // PREFILL phase: Always use full weight (GPU is primary, compute-bound)
        if (phase == InferencePhase::PREFILL)
        {
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] PREFILL phase - returning full weight for " << name);
            auto weight = weight_manager_->getWeight(name);
            if (!weight)
            {
                LOG_ERROR("[DeviceGraphOrchestrator::getPhaseAwareWeight] Failed to load weight: " << name);
            }
            return weight;
        }

        // DECODE phase: Check if CPU should participate
        if (!shouldUseCPUDecodeWeight(name, layer_idx))
        {
            // No CPU participation - use full weight (GPU handles it)
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, no CPU participation - returning full weight for " << name);
            return weight_manager_->getWeight(name);
        }

        // CPU decode participation enabled - get decode shard
        if (!weight_placement_map_)
        {
            // No placement map - fall back to full weight
            LOG_WARN("[DeviceGraphOrchestrator::getPhaseAwareWeight] CPU decode participation but no placement map - using full weight for " << name);
            return weight_manager_->getWeight(name);
        }

        // Get device info from placement map
        WeightDeviceInfo device_info = weight_placement_map_->getDeviceInfoForWeight(name, layer_idx);

        if (!device_info.cpu_decode_participation)
        {
            // This weight doesn't have CPU decode participation
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, weight " << name << " has no CPU participation - returning full weight");
            return weight_manager_->getWeight(name);
        }

        // Find the CPU device in decode_devices and get its fraction
        for (size_t i = 0; i < device_info.decode_devices.size(); ++i)
        {
            if (device_info.decode_devices[i].is_cpu())
            {
                float fraction = device_info.decode_fractions[i];
                LOG_DEBUG("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase - returning CPU decode shard for "
                          << name << " (fraction=" << fraction << ")");
                return weight_manager_->getDecodeWeight(name, DeviceId::cpu(), fraction, layer_idx);
            }
        }

        // No CPU in decode devices - use full weight
        LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, CPU not in decode devices - returning full weight for " << name);
        return weight_manager_->getWeight(name);
    }

    bool DeviceGraphOrchestrator::shouldUseCPUDecodeWeight(const std::string &name, int layer_idx) const
    {
        // Check phase constraint:
        // - Default (cpu_prefill_participate=false): CPU only participates in DECODE phase
        // - With cpu_prefill_participate=true: CPU participates in BOTH phases (Option C fallback)
        if (current_phase_ == InferencePhase::PREFILL)
        {
            // Check if CPU prefill participation is enabled (Option C: memory-constrained systems)
            if (!debugEnv().execution.cpu_prefill_participate)
            {
                return false;
            }
            // If cpu_prefill_participate is true, continue with the rest of the checks
        }
        // DECODE phase always allows CPU participation if configured

        // Must have placement map
        if (!weight_placement_map_)
        {
            return false;
        }

        // Get device info
        WeightDeviceInfo device_info = weight_placement_map_->getDeviceInfoForWeight(name, layer_idx);

        // Check if CPU decode participation is enabled for this weight
        if (!device_info.cpu_decode_participation)
        {
            return false;
        }

        // Check if this MPI rank should handle CPU decode
        // For now, rank 0 is the designated CPU decode participant
        // This could be made configurable in the future
        int my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;

        // The CPU decode participant is typically the first rank (rank 0)
        // In future, could look at device_info.decode_devices to find which
        // ranks have CPUs and distribute work accordingly
        bool is_cpu_decode_rank = (my_rank == 0);

        if (is_cpu_decode_rank)
        {
            // Verify that CPU is actually in the decode devices
            for (const auto &dev : device_info.decode_devices)
            {
                if (dev.is_cpu())
                {
                    return true;
                }
            }
        }

        return false;
    }

    // =========================================================================
    // Unified Pipeline Configuration (Phase 6)
    // =========================================================================

    void DeviceGraphOrchestrator::setPipelineConfig(std::shared_ptr<PipelineConfig> config)
    {
        if (config)
        {
            std::string validation_error;
            if (!config->validate(&validation_error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Invalid PipelineConfig: " << validation_error);
                throw std::invalid_argument("Invalid PipelineConfig: " + validation_error);
            }
        }

        pipeline_config_ = std::move(config);

        // Reset initialization flags - contexts need to be recreated
        pp_contexts_initialized_ = false;
        tp_contexts_initialized_ = false;
        pp_contexts_.clear();
        domain_tp_contexts_.clear();

        if (pipeline_config_)
        {
            LOG_INFO("[DeviceGraphOrchestrator] Unified pipeline configured: "
                     << pipeline_config_->numStages() << " PP stages, "
                     << pipeline_config_->tp_domains.size() << " TP domains, "
                     << pipeline_config_->total_layers << " total layers");

            // Propagate to graph builder via setter
            graph_builder_->setPipelineConfig(pipeline_config_);
        }
        else
        {
            LOG_INFO("[DeviceGraphOrchestrator] Pipeline configuration cleared (single-device mode)");
            graph_builder_->setPipelineConfig(nullptr);
        }
    }

    bool DeviceGraphOrchestrator::initializePPContexts()
    {
        if (pp_contexts_initialized_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] PP contexts already initialized");
            return true;
        }

        if (!pipeline_config_ || !pipeline_config_->hasPP())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] No PP configuration - skipping PP context initialization");
            pp_contexts_initialized_ = true;
            return true;
        }

        // Ensure TP contexts are initialized first (needed for PPStage::fromTPContext)
        if (!initializeTPContexts())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize TP contexts - cannot create PP contexts");
            return false;
        }

        // Check if any domain has internal TP (degree > 1)
        // If so, we need to use HierarchicalPPConfig to properly handle TP domains
        const bool has_internal_tp = pipeline_config_->hasTP();

        LOG_INFO("[DeviceGraphOrchestrator] Initializing PP contexts for "
                 << (pipeline_config_->numStages() - 1) << " inter-stage transfers"
                 << (has_internal_tp ? " (with TP domains)" : "") << "...");

        // Create PP context for each adjacent pair of stages
        for (int stage = 0; stage < pipeline_config_->numStages() - 1; ++stage)
        {
            int next_stage = stage + 1;
            auto key = std::make_pair(stage, next_stage);

            // Get domains for the two stages
            const auto *domain_from = pipeline_config_->getDomainForStage(stage);
            const auto *domain_to = pipeline_config_->getDomainForStage(next_stage);

            if (!domain_from || !domain_to)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to get domains for PP transfer "
                          << stage << " -> " << next_stage);
                return false;
            }

            // Build layer boundaries for the PP context
            // We need to include all stages up to and including next_stage
            std::vector<int> layer_boundaries;
            for (int s = 0; s <= next_stage; ++s)
            {
                const auto &pp_stage = pipeline_config_->pp_stages[s];
                if (s == 0)
                {
                    layer_boundaries.push_back(pp_stage.first_layer);
                }
                layer_boundaries.push_back(pp_stage.last_layer);
            }

            std::unique_ptr<ILocalPPContext> pp_ctx;

            if (has_internal_tp)
            {
                // Use HierarchicalPPConfig with PPStage variant type
                // This allows PP transfers to understand TP domains
                HierarchicalPPConfig pp_config;
                pp_config.layer_boundaries = layer_boundaries;

                // Build PPStage for each stage 0..next_stage
                for (int s = 0; s <= next_stage; ++s)
                {
                    const auto *domain = pipeline_config_->getDomainForStage(s);
                    if (!domain)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Missing domain for stage " << s);
                        return false;
                    }

                    if (domain->degree() > 1)
                    {
                        // This stage has internal TP - look up the TP context
                        auto it = domain_tp_contexts_.find(domain->name);
                        if (it == domain_tp_contexts_.end())
                        {
                            LOG_ERROR("[DeviceGraphOrchestrator] TP context not found for domain '"
                                      << domain->name << "' (stage " << s << ")");
                            return false;
                        }
                        pp_config.stages.push_back(PPStage::fromTPContext(it->second));
                        LOG_DEBUG("[DeviceGraphOrchestrator] Stage " << s << " → TP domain '"
                                                                     << domain->name << "' (" << domain->degree() << " devices)");
                    }
                    else
                    {
                        // Single device stage
                        auto device = GlobalDeviceAddress::fromLocalDeviceId(domain->primaryDevice());
                        pp_config.stages.push_back(PPStage::fromDevice(device));
                        LOG_DEBUG("[DeviceGraphOrchestrator] Stage " << s << " → single device "
                                                                     << device.toString());
                    }
                }

                if (!pp_config.isValid())
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Invalid HierarchicalPPConfig for stages "
                              << stage << " -> " << next_stage);
                    return false;
                }

                pp_ctx = createLocalPPContext(pp_config);
            }
            else
            {
                // Use flat LocalPPConfig (simpler, no TP domain awareness needed)
                std::vector<GlobalDeviceAddress> stage_devices;
                for (int s = 0; s <= next_stage; ++s)
                {
                    const auto *domain = pipeline_config_->getDomainForStage(s);
                    if (domain && !domain->devices.empty())
                    {
                        stage_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(domain->primaryDevice()));
                    }
                }

                LocalPPConfig pp_ctx_config{
                    .stage_devices = stage_devices,
                    .layer_boundaries = layer_boundaries};

                pp_ctx = createLocalPPContext(pp_ctx_config);
            }

            if (!pp_ctx)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create PP context for transfer "
                          << stage << " -> " << next_stage);
                return false;
            }

            pp_contexts_[key] = std::move(pp_ctx);

            LOG_DEBUG("[DeviceGraphOrchestrator] Created PP context: stage " << stage
                                                                             << " (" << domain_from->name << ") -> stage " << next_stage
                                                                             << " (" << domain_to->name << ")");
        }

        // Wire PP contexts to graph builder via setters
        for (auto &[key, ctx] : pp_contexts_)
        {
            graph_builder_->setPPContext(key.first, key.second, ctx.get());
        }

        pp_contexts_initialized_ = true;
        LOG_INFO("[DeviceGraphOrchestrator] Initialized " << pp_contexts_.size() << " PP contexts"
                                                          << (has_internal_tp ? " (hierarchical)" : " (flat)"));
        return true;
    }

    bool DeviceGraphOrchestrator::initializeTPContexts()
    {
        if (tp_contexts_initialized_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] TP contexts already initialized");
            return true;
        }

        if (!pipeline_config_ || pipeline_config_->tp_domains.empty())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] No TP domains - skipping TP context initialization");
            tp_contexts_initialized_ = true;
            return true;
        }

        LOG_INFO("[DeviceGraphOrchestrator] Initializing TP contexts for "
                 << pipeline_config_->tp_domains.size() << " domains...");

        // Create TP context for each domain that has degree > 1
        for (const auto &domain : pipeline_config_->tp_domains)
        {
            if (domain.devices.size() <= 1)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Domain '" << domain.name
                                                               << "' has degree " << domain.devices.size() << " - no TP context needed");
                continue;
            }

            // Convert DeviceId to GlobalDeviceAddress
            std::vector<GlobalDeviceAddress> addresses;
            for (const auto &dev : domain.devices)
            {
                addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(dev));
            }

            // Equal weights for now (TODO: support proportional TP)
            std::vector<float> weights(domain.devices.size(), 1.0f / domain.devices.size());

            // Create TP context (convert unique_ptr to shared_ptr so PPStage can reference it)
            auto tp_ctx = createLocalTPContext(addresses, weights, domain.tp_backend);
            if (!tp_ctx)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create TP context for domain '"
                          << domain.name << "'");
                return false;
            }

            domain_tp_contexts_[domain.name] = std::shared_ptr<ILocalTPContext>(std::move(tp_ctx));

            LOG_DEBUG("[DeviceGraphOrchestrator] Created TP context for domain '" << domain.name
                                                                                  << "': " << domain.devices.size() << " devices, backend="
                                                                                  << static_cast<int>(domain.tp_backend));
        }

        // Wire TP contexts to graph builder via setters
        for (auto &[name, ctx] : domain_tp_contexts_)
        {
            graph_builder_->setTPContext(name, ctx.get());
        }

        tp_contexts_initialized_ = true;
        LOG_INFO("[DeviceGraphOrchestrator] Initialized " << domain_tp_contexts_.size() << " TP contexts");
        return true;
    }

    bool DeviceGraphOrchestrator::initializeBufferPool(const PPStageBufferSpec &spec, const MPIContext *mpi_ctx)
    {
        if (!pipeline_config_)
        {
            LOG_WARN("[DeviceGraphOrchestrator] Cannot initialize buffer pool without PipelineConfig");
            return false;
        }

        if (!pipeline_config_->hasPP())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] No PP stages - buffer pool not needed");
            return true;
        }

        LOG_INFO("[DeviceGraphOrchestrator] Initializing per-stage buffer pool for "
                 << pipeline_config_->numStages() << " PP stages...");

        // Initialize the buffer pool
        buffer_pool_.emplace();
        if (!buffer_pool_->initialize(*pipeline_config_, spec, mpi_ctx))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize per-stage buffer pool");
            buffer_pool_.reset();
            return false;
        }

        LOG_INFO("[DeviceGraphOrchestrator] Per-stage buffer pool initialized: "
                 << buffer_pool_->stats().total_bytes() << " bytes across "
                 << buffer_pool_->numStages() << " stages");
        return true;
    }

    // =========================================================================
    // GraphBuildSession Implementation (nested class)
    // =========================================================================

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::forInput(const Qwen2ForwardInput &input)
    {
        input_ = input;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withPositionIds(const int *position_ids)
    {
        explicit_position_ids_ = position_ids;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withExternalHiddenState(TensorBase *hidden_state)
    {
        external_hidden_state_ = hidden_state;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withPipelineConfig(std::shared_ptr<PipelineConfig> config)
    {
        pipeline_config_ = std::move(config);
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::forPPStage(int first_layer, int last_layer,
                                                           bool has_embedding, bool has_lm_head)
    {
        pp_stage_ = PPStageSpec{first_layer, last_layer, has_embedding, has_lm_head};
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withPPContext(int from_stage, int to_stage, ILocalPPContext *context)
    {
        pp_contexts_[{from_stage, to_stage}] = context;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withTPContext(const std::string &domain_name, ILocalTPContext *context)
    {
        tp_contexts_[domain_name] = context;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withWeights(const Qwen2ModelWeights &weights)
    {
        weights_ = weights;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withBuffers(const Qwen2ModelBuffers &buffers)
    {
        buffers_ = buffers;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withKVCache(IKVCache *kv_cache)
    {
        kv_cache_ = kv_cache;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withBufferPool(PerStageBufferPool *pool)
    {
        buffer_pool_ = pool;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildResult
    DeviceGraphOrchestrator::GraphBuildSession::buildForward()
    {
        if (!isValid())
        {
            return GraphBuildResult(validationError());
        }

        applyConfiguration();
        auto prepared_input = prepareInput();

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return GraphBuildResult("No graph builder available");
        }

        Qwen2ForwardOutput output;
        ComputeGraph graph = graph_builder->buildFullForwardGraph(prepared_input, output);

        if (graph.size() == 0)
        {
            return GraphBuildResult("buildFullForwardGraph returned empty graph");
        }

        LOG_DEBUG("[GraphBuildSession] Built full forward graph with " << graph.size() << " nodes");
        return GraphBuildResult(std::move(graph), output);
    }

    DeviceGraphOrchestrator::GraphBuildResult
    DeviceGraphOrchestrator::GraphBuildSession::buildPartial()
    {
        if (!isValid())
        {
            return GraphBuildResult(validationError());
        }

        if (!pp_stage_.has_value())
        {
            return GraphBuildResult("buildPartial() requires forPPStage() configuration");
        }

        applyConfiguration();
        auto prepared_input = prepareInput();

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return GraphBuildResult("No graph builder available");
        }

        const auto &stage = pp_stage_.value();
        Qwen2ForwardOutput output;
        ComputeGraph graph = graph_builder->buildPartialForwardGraph(
            prepared_input, output,
            stage.first_layer, stage.last_layer,
            stage.has_embedding, stage.has_lm_head);

        if (graph.size() == 0)
        {
            return GraphBuildResult("buildPartialForwardGraph returned empty graph");
        }

        LOG_DEBUG("[GraphBuildSession] Built partial forward graph: layers=["
                  << stage.first_layer << ", " << stage.last_layer << ") with "
                  << graph.size() << " nodes");
        return GraphBuildResult(std::move(graph), output);
    }

    DeviceGraphOrchestrator::GraphBuildResult
    DeviceGraphOrchestrator::GraphBuildSession::buildUnified()
    {
        if (!isValid())
        {
            return GraphBuildResult(validationError());
        }

        if (!pipeline_config_ || !pipeline_config_->hasPP())
        {
            return GraphBuildResult("buildUnified() requires withPipelineConfig() with hasPP()");
        }

        applyConfiguration();
        auto prepared_input = prepareInput();

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return GraphBuildResult("No graph builder available");
        }

        // Set pipeline config on graph builder
        graph_builder->setPipelineConfig(pipeline_config_);

        // Wire PP contexts
        for (const auto &[key, ctx] : pp_contexts_)
        {
            graph_builder->setPPContext(key.first, key.second, ctx);
        }

        // Wire TP contexts
        for (const auto &[name, ctx] : tp_contexts_)
        {
            graph_builder->setTPContext(name, ctx);
        }

        Qwen2ForwardOutput output;
        ComputeGraph graph = graph_builder->buildUnifiedPipelineGraph(prepared_input, output);

        if (graph.size() == 0)
        {
            return GraphBuildResult("buildUnifiedPipelineGraph returned empty graph");
        }

        LOG_DEBUG("[GraphBuildSession] Built unified PP graph: "
                  << pipeline_config_->numStages() << " stages, "
                  << graph.size() << " nodes");
        return GraphBuildResult(std::move(graph), output);
    }

    DeviceGraphOrchestrator::GraphBuildResult
    DeviceGraphOrchestrator::GraphBuildSession::build()
    {
        // Auto-select based on configuration
        if (pipeline_config_ && pipeline_config_->hasPP())
        {
            return buildUnified();
        }
        else if (pp_stage_.has_value())
        {
            return buildPartial();
        }
        else
        {
            return buildForward();
        }
    }

    bool DeviceGraphOrchestrator::GraphBuildSession::isValid() const
    {
        return validationError().empty();
    }

    std::string DeviceGraphOrchestrator::GraphBuildSession::validationError() const
    {
        if (!input_.has_value())
        {
            return "No input configured (call forInput())";
        }

        const auto &input = input_.value();

        // Check if this is a PP middle/final stage (no embedding, uses external hidden state)
        bool is_pp_non_embedding_stage = pp_stage_.has_value() && !pp_stage_->has_embedding;
        // Check both session-level and input-level external hidden state
        bool has_external_hidden = external_hidden_state_ != nullptr ||
                                   input.external_hidden_state != nullptr;

        if (input.token_ids == nullptr)
        {
            // PP middle/final stages don't need token_ids if they have external hidden state
            if (is_pp_non_embedding_stage && has_external_hidden)
            {
                // Valid: PP stage with external hidden state input
            }
            else if (is_pp_non_embedding_stage)
            {
                return "PP stage without embedding requires external hidden state (call withExternalHiddenState())";
            }
            else
            {
                return "Input token_ids are null";
            }
        }

        if (input.seq_len <= 0)
        {
            return "Invalid sequence length: " + std::to_string(input.seq_len);
        }

        if (input.batch_size <= 0)
        {
            return "Invalid batch size: " + std::to_string(input.batch_size);
        }

        // Unified PP requires pipeline config
        if (pipeline_config_ && pipeline_config_->hasPP())
        {
            // PP contexts should be registered for all stage pairs
            int num_stages = pipeline_config_->numStages();
            for (int s = 0; s < num_stages - 1; ++s)
            {
                auto key = std::make_pair(s, s + 1);
                if (pp_contexts_.find(key) == pp_contexts_.end())
                {
                    return "Missing PP context for stage transfer " + std::to_string(s) +
                           " -> " + std::to_string(s + 1);
                }
            }
        }

        return "";
    }

    Qwen2ForwardInput DeviceGraphOrchestrator::GraphBuildSession::prepareInput() const
    {
        Qwen2ForwardInput prepared = input_.value();

        // Override position IDs if explicitly set
        if (explicit_position_ids_)
        {
            prepared.position_ids = explicit_position_ids_;
        }

        // Set external hidden state for PP middle/final stages
        if (external_hidden_state_)
        {
            prepared.external_hidden_state = external_hidden_state_;
        }

        // Set KV cache
        if (kv_cache_)
        {
            prepared.kv_cache = kv_cache_;
        }

        return prepared;
    }

    void DeviceGraphOrchestrator::GraphBuildSession::applyConfiguration()
    {
        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return;
        }

        // Apply weights if provided
        if (weights_.has_value())
        {
            graph_builder->setWeights(weights_.value());
        }

        // Apply buffers if provided
        if (buffers_.has_value())
        {
            graph_builder->setBuffers(buffers_.value());
        }
    }

    // =========================================================================
    // AttentionGraphSession Implementation (nested class)
    // =========================================================================

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::forLayer(const Qwen2LayerWeights &layer, int layer_idx)
    {
        layer_ = &layer;
        layer_idx_ = layer_idx;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withBuffers(Qwen2ActivationBuffers &buffers)
    {
        buffers_ = &buffers;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withSequence(int seq_len, int batch_size)
    {
        seq_len_ = seq_len;
        batch_size_ = batch_size;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::onDevice(DeviceId device)
    {
        device_ = device;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withKVCache(IKVCache *kv_cache)
    {
        kv_cache_ = kv_cache;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withPositionIds(const int *position_ids)
    {
        position_ids_ = position_ids;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withSequenceLengths(const std::vector<int> *lengths)
    {
        sequence_lengths_ = lengths;
        return *this;
    }

    bool DeviceGraphOrchestrator::AttentionGraphSession::isValid() const
    {
        return validationError().empty();
    }

    std::string DeviceGraphOrchestrator::AttentionGraphSession::validationError() const
    {
        if (!layer_)
        {
            return "No layer weights configured (call forLayer())";
        }
        if (!buffers_)
        {
            return "No buffers configured (call withBuffers())";
        }
        if (layer_idx_ < 0)
        {
            return "Invalid layer index: " + std::to_string(layer_idx_);
        }
        if (seq_len_ <= 0)
        {
            return "Invalid sequence length (call withSequence())";
        }
        if (batch_size_ <= 0)
        {
            return "Invalid batch size: " + std::to_string(batch_size_);
        }
        if (!device_.has_value())
        {
            return "No device configured (call onDevice())";
        }
        return "";
    }

    DeviceGraphOrchestrator::SubGraphBuildResult
    DeviceGraphOrchestrator::AttentionGraphSession::build()
    {
        if (!isValid())
        {
            return SubGraphBuildResult(validationError());
        }

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return SubGraphBuildResult("No graph builder available");
        }

        ComputeGraph graph = graph_builder->buildAttentionGraph(
            *layer_, *buffers_, layer_idx_, seq_len_, batch_size_,
            kv_cache_, position_ids_, device_.value(), sequence_lengths_);

        if (graph.size() == 0)
        {
            return SubGraphBuildResult("buildAttentionGraph returned empty graph");
        }

        LOG_DEBUG("[AttentionGraphSession] Built attention graph for layer "
                  << layer_idx_ << " with " << graph.size() << " nodes");

        return SubGraphBuildResult(std::move(graph));
    }

    // =========================================================================
    // FFNGraphSession Implementation (nested class)
    // =========================================================================

    DeviceGraphOrchestrator::FFNGraphSession &
    DeviceGraphOrchestrator::FFNGraphSession::forLayer(const Qwen2LayerWeights &layer, int layer_idx)
    {
        layer_ = &layer;
        layer_idx_ = layer_idx;
        return *this;
    }

    DeviceGraphOrchestrator::FFNGraphSession &
    DeviceGraphOrchestrator::FFNGraphSession::withBuffers(Qwen2ActivationBuffers &buffers)
    {
        buffers_ = &buffers;
        return *this;
    }

    DeviceGraphOrchestrator::FFNGraphSession &
    DeviceGraphOrchestrator::FFNGraphSession::withSequence(int seq_len, int batch_size)
    {
        seq_len_ = seq_len;
        batch_size_ = batch_size;
        return *this;
    }

    DeviceGraphOrchestrator::FFNGraphSession &
    DeviceGraphOrchestrator::FFNGraphSession::onDevice(DeviceId device)
    {
        device_ = device;
        return *this;
    }

    bool DeviceGraphOrchestrator::FFNGraphSession::isValid() const
    {
        return validationError().empty();
    }

    std::string DeviceGraphOrchestrator::FFNGraphSession::validationError() const
    {
        if (!layer_)
        {
            return "No layer weights configured (call forLayer())";
        }
        if (!buffers_)
        {
            return "No buffers configured (call withBuffers())";
        }
        if (layer_idx_ < 0)
        {
            return "Invalid layer index: " + std::to_string(layer_idx_);
        }
        if (seq_len_ <= 0)
        {
            return "Invalid sequence length (call withSequence())";
        }
        if (batch_size_ <= 0)
        {
            return "Invalid batch size: " + std::to_string(batch_size_);
        }
        if (!device_.has_value())
        {
            return "No device configured (call onDevice())";
        }
        return "";
    }

    DeviceGraphOrchestrator::SubGraphBuildResult
    DeviceGraphOrchestrator::FFNGraphSession::build()
    {
        if (!isValid())
        {
            return SubGraphBuildResult(validationError());
        }

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return SubGraphBuildResult("No graph builder available");
        }

        ComputeGraph graph = graph_builder->buildFFNGraph(
            *layer_, *buffers_, layer_idx_, seq_len_, batch_size_, device_.value());

        if (graph.size() == 0)
        {
            return SubGraphBuildResult("buildFFNGraph returned empty graph");
        }

        LOG_DEBUG("[FFNGraphSession] Built FFN graph for layer "
                  << layer_idx_ << " with " << graph.size() << " nodes");

        return SubGraphBuildResult(std::move(graph));
    }

} // namespace llaminar2
