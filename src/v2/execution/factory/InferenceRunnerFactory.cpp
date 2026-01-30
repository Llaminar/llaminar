/**
 * @file InferenceRunnerFactory.cpp
 * @brief Factory implementation for creating IInferenceRunner instances
 * @author David Sanftenberg
 * @date December 2025
 */

#include "InferenceRunnerFactory.h"
#include "../../backends/DeviceId.h"
#include "../../backends/BackendManager.h"
#include "../local_execution/collective/CollectiveContext.h"
#include "../mpi_orchestration/DeviceInventory.h"
#include "../../models/qwen/Qwen2Graph.h"
#include "../../models/qwen/Qwen2Schema.h"
#include "../local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "../../loaders/ModelContext.h"
#include "../../loaders/ModelLoader.h"
#include "../../loaders/WeightManager.h"
#include "../../loaders/WeightStreamerFactory.h"
#include "../../collective/ILocalTPContext.h"
#include "../local_execution/orchestrators/IMultiDeviceOrchestrator.h"
#include "../local_execution/orchestrators/MultiDeviceOrchestrator.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    // Forward declarations of factory helpers
    static std::unique_ptr<IInferenceRunner> createDeviceGraphOrchestratorImpl(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config,
        const std::string &architecture);

    static bool configureOrchestratorWeightsImpl(
        DeviceGraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx,
        DeviceId device);

    // =========================================================================
    // Helper: Build ClusterInventory for GPU Collective Context
    // =========================================================================

    /**
     * @brief Build a ClusterInventory from available GPU backends
     *
     * Detects local CUDA and ROCm GPUs via backend APIs and builds a
     * single-rank ClusterInventory suitable for CollectiveContextFactory.
     *
     * For multi-node MPI, each rank builds its local inventory, and
     * the full cluster inventory would be built via MPI_Allgather.
     * For now, we support single-node with multiple GPUs.
     *
     * @param mpi_ctx MPI context (for rank info)
     * @return ClusterInventory with detected devices
     */
    static ClusterInventory buildLocalClusterInventory(
        const std::shared_ptr<MPIContext> &mpi_ctx)
    {
        ClusterInventory inventory;
        RankInventory rank_inv;

        rank_inv.rank = mpi_ctx ? mpi_ctx->rank() : 0;
        rank_inv.node_id = 0; // Single-node for now
        rank_inv.local_rank = rank_inv.rank;
        rank_inv.hostname = "localhost";

#ifdef HAVE_CUDA
        IBackend *cuda_backend = getCUDABackend();
        if (cuda_backend)
        {
            int cuda_count = cuda_backend->deviceCount();
            for (int i = 0; i < cuda_count; ++i)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::CUDA;
                gpu.local_device_id = i;
                gpu.memory_bytes = cuda_backend->deviceMemoryTotal(i);
                gpu.name = cuda_backend->deviceName(i);
                gpu.supports_p2p = true;
                rank_inv.gpus.push_back(gpu);
            }
            LOG_DEBUG("[InferenceRunner] Detected " << cuda_count << " CUDA GPU(s)");
        }
#endif

#ifdef HAVE_ROCM
        IBackend *rocm_backend = getROCmBackend();
        if (rocm_backend)
        {
            int rocm_count = rocm_backend->deviceCount();
            for (int i = 0; i < rocm_count; ++i)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::ROCm;
                gpu.local_device_id = i;
                gpu.memory_bytes = rocm_backend->deviceMemoryTotal(i);
                gpu.name = rocm_backend->deviceName(i);
                rank_inv.gpus.push_back(gpu);
            }
            LOG_DEBUG("[InferenceRunner] Detected " << rocm_count << " ROCm GPU(s)");
        }
#endif

        inventory.ranks.push_back(rank_inv);
        inventory.world_size = mpi_ctx ? mpi_ctx->world_size() : 1;
        inventory.buildNodeAggregations();

        return inventory;
    }

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<IInferenceRunner> createInferenceRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config)
    {
        LOG_DEBUG("[InferenceRunner] createInferenceRunner called with mpi_ctx="
                  << (mpi_ctx ? "valid" : "nullptr")
                  << " world_size=" << (mpi_ctx ? mpi_ctx->world_size() : -1));

        if (!model_ctx)
        {
            LOG_ERROR("[InferenceRunner] model_ctx is null");
            return nullptr;
        }

        // Validate device
        if (!device.is_valid())
        {
            LOG_ERROR("[InferenceRunner] Invalid device " << device
                                                          << ". Use DeviceId::cpu() for CPU.");
            return nullptr;
        }
        LOG_DEBUG("[InferenceRunner] Using device " << device);

        // Graph is the only execution path (as of January 2025 cleanup)
        std::string architecture = model_ctx->architecture();
        LOG_DEBUG("[InferenceRunner] Using GRAPH path");
        return createDeviceGraphOrchestratorImpl(model_ctx, mpi_ctx, device, config, architecture);
    }

    // =========================================================================
    // Factory Helper Implementations
    // =========================================================================

    static std::unique_ptr<IInferenceRunner> createDeviceGraphOrchestratorImpl(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config,
        const std::string &architecture)
    {
        // Currently only Qwen2 is supported
        if (architecture != "qwen2")
        {
            LOG_ERROR("[InferenceRunner] Only qwen2 architecture is supported, got: " << architecture);
            return nullptr;
        }

        // Configure weight sharding from Qwen2 schema
        auto weight_mgr = model_ctx->concreteWeightManager();
        if (weight_mgr)
        {
            Qwen2SchemaFactory schema_factory;
            weight_mgr->setWeightShardingConfig(schema_factory.getWeightShardingConfig());
            LOG_DEBUG("[InferenceRunner] Applied Qwen2 sharding config to WeightManager");
        }

        // Get model metadata
        ModelLoader &loader = model_ctx->concreteLoader();
        const auto &model = loader.getModel();

        // Build Qwen2GraphConfig from model metadata
        Qwen2GraphConfig graph_config;
        graph_config.vocab_size = static_cast<int>(model.vocab_size);
        graph_config.d_model = static_cast<int>(model.embedding_length);
        graph_config.n_layers = static_cast<int>(model.block_count);
        graph_config.n_heads = static_cast<int>(model.head_count);
        graph_config.n_kv_heads = static_cast<int>(model.head_count_kv);
        graph_config.head_dim = graph_config.d_model / graph_config.n_heads;
        graph_config.d_ff = 0; // Will need to compute from intermediate_size metadata
        graph_config.max_seq_len = config.max_seq_len;
        graph_config.rope_theta = model.rope_theta;
        graph_config.rms_norm_eps = model.rms_norm_eps;

        // CRITICAL: Set default device for kernel dispatch
        // This determines which kernels (CPU vs CUDA) are selected for execution
        graph_config.default_device = device;
        LOG_DEBUG("[InferenceRunner] Default device: " << graph_config.default_device.to_string());

        // Propagate activation precision from runtime config
        // This determines buffer types (FP32/Q8_1) and kernel selection
        graph_config.activation_precision = config.activation_precision;
        LOG_DEBUG("[InferenceRunner] Activation precision: " << activationPrecisionToString(config.activation_precision));

        // Propagate fused attention backend selection
        // This determines which kernel implementation to use for fused attention
        // For HybridQ16 mode, automatically use Q16_INTEGER backend (JIT doesn't support Q16_1)
        FusedAttentionBackend effective_backend = config.fused_attention_backend;
        if (config.activation_precision == ActivationPrecision::HybridQ16 &&
            config.fused_attention_backend == FusedAttentionBackend::JIT)
        {
            effective_backend = FusedAttentionBackend::Q16_INTEGER;
            LOG_DEBUG("[InferenceRunner] HybridQ16 mode: auto-selecting Q16_INTEGER backend (JIT doesn't support Q16_1)");
        }
        graph_config.fused_attention_backend = effective_backend;
        LOG_DEBUG("[InferenceRunner] Fused attention backend: " << fusedAttentionBackendToString(effective_backend));

        // Propagate kv_cache_scale for Q16_1 KV cache quantization
        // This fixed scale determines the FP32 range that maps to INT16 [-32767, +32767]
        graph_config.kv_cache_scale = config.kv_cache_scale;
        LOG_DEBUG("[InferenceRunner] KV cache scale: " << config.kv_cache_scale
                                                       << " (±" << config.kv_cache_scale << " FP32 range)");

        // Try to get d_ff from metadata (intermediate_size)
        if (model.hasMetadata("llama.feed_forward_length"))
        {
            auto it = model.metadata.find("llama.feed_forward_length");
            if (it != model.metadata.end())
            {
                if (it->second.type == GGUFValueType::UINT64)
                {
                    graph_config.d_ff = static_cast<int>(it->second.asUInt64());
                }
                else if (it->second.type == GGUFValueType::UINT32)
                {
                    graph_config.d_ff = static_cast<int>(it->second.asUInt32());
                }
            }
        }
        else if (model.hasMetadata("qwen2.feed_forward_length"))
        {
            auto it = model.metadata.find("qwen2.feed_forward_length");
            if (it != model.metadata.end())
            {
                if (it->second.type == GGUFValueType::UINT64)
                {
                    graph_config.d_ff = static_cast<int>(it->second.asUInt64());
                }
                else if (it->second.type == GGUFValueType::UINT32)
                {
                    graph_config.d_ff = static_cast<int>(it->second.asUInt32());
                }
            }
        }

        // Fallback: estimate d_ff as ~4x d_model (common SwiGLU ratio)
        if (graph_config.d_ff == 0)
        {
            graph_config.d_ff = graph_config.d_model * 4;
            LOG_WARN("[InferenceRunner] Could not find feed_forward_length, using estimate: " << graph_config.d_ff);
        }

        // =====================================================================
        // Phase 3+: Tensor-Parallel Configuration
        // =====================================================================
        // Three modes are supported (in order of precedence):
        //
        // A) LOCAL TP (config.local_tp_ctx): Single MPI rank, multiple devices
        //    - Configured via ILocalTPContext in InferenceRunnerConfig
        //    - Collectives via NCCL/RCCL/PCIeBAR (high bandwidth, no host staging)
        //    - Uses ILocalTPContext for proportional head/FFN/vocab assignment
        //
        // B) TensorParallelConfig (Phase 1c): Proportional GLOBAL TP
        //    - Used for heterogeneous GLOBAL TP (e.g., NVIDIA 73% + AMD 27%)
        //    - Assignment comes from DeviceShardingAssignment per MPI rank
        //
        // C) Equal split (legacy): 1/world_size equal GLOBAL TP
        //    - Used for homogeneous GLOBAL setups
        //
        // IMPORTANT: Only enable tensor parallelism if weights are actually sharded.
        // If weights are REPLICATED, each rank has the full weight and should use
        // the full head counts to avoid buffer/weight dimension mismatch.
        // =====================================================================
        const bool weights_sharded = weight_mgr &&
                                     (weight_mgr->strategy() == WeightDistributionStrategy::SHARDED);

        // Check if LOCAL TP context is provided (takes precedence over GLOBAL TP)
        ILocalTPContext *local_tp_ctx = config.local_tp_ctx;
        const int local_tp_device_idx = config.local_tp_device_index;

        // Check if TensorParallelConfig is available from WeightManager (for GLOBAL TP)
        const TensorParallelConfig *tp_config = weight_mgr ? weight_mgr->tensorParallelConfig() : nullptr;
        const int current_rank = mpi_ctx ? mpi_ctx->rank() : 0;

        if (local_tp_ctx && local_tp_ctx->degree() > 1 && weights_sharded)
        {
            // =====================================================================
            // LOCAL TP: Single rank with multiple devices via ILocalTPContext
            // =====================================================================
            const int tp_degree = local_tp_ctx->degree();
            const auto &devices = local_tp_ctx->devices();
            const auto &weights = local_tp_ctx->weights();

            if (local_tp_device_idx < 0 || local_tp_device_idx >= tp_degree)
            {
                LOG_ERROR("[InferenceRunner] Invalid local_tp_device_index: " << local_tp_device_idx
                                                                              << " (degree=" << tp_degree << ")");
                return nullptr;
            }

            const GlobalDeviceAddress &my_device = devices[local_tp_device_idx];
            const float my_weight = weights.empty() ? (1.0f / tp_degree) : weights[local_tp_device_idx];

            // Compute head assignment based on weight/index
            int head_start = 0;
            int kv_head_start = 0;
            int d_ff_start = 0;
            int vocab_start = 0;

            // Accumulate starts from devices before this one
            for (int i = 0; i < local_tp_device_idx; ++i)
            {
                float w = weights.empty() ? (1.0f / tp_degree) : weights[i];
                head_start += static_cast<int>(std::round(w * graph_config.n_heads));
                kv_head_start += static_cast<int>(std::round(w * graph_config.n_kv_heads));
                d_ff_start += static_cast<int>(std::round(w * graph_config.d_ff));
                vocab_start += static_cast<int>(std::round(w * graph_config.vocab_size));
            }

            // Compute counts for this device
            int local_n_heads = static_cast<int>(std::round(my_weight * graph_config.n_heads));
            int local_n_kv_heads = static_cast<int>(std::round(my_weight * graph_config.n_kv_heads));
            int d_ff_local = static_cast<int>(std::round(my_weight * graph_config.d_ff));
            int vocab_local = static_cast<int>(std::round(my_weight * graph_config.vocab_size));

            // Handle rounding: last device gets remainder
            if (local_tp_device_idx == tp_degree - 1)
            {
                local_n_heads = graph_config.n_heads - head_start;
                local_n_kv_heads = graph_config.n_kv_heads - kv_head_start;
                d_ff_local = graph_config.d_ff - d_ff_start;
                vocab_local = graph_config.vocab_size - vocab_start;
            }

            graph_config.head_start = head_start;
            graph_config.local_n_heads = local_n_heads;
            graph_config.local_n_kv_heads = local_n_kv_heads;
            graph_config.qkv_column_parallel = true;
            graph_config.local_rank = local_tp_device_idx;

            graph_config.d_ff_local = d_ff_local;
            graph_config.ffn_column_parallel = true;

            graph_config.vocab_local = vocab_local;
            graph_config.lm_head_column_parallel = true;

            // Store LOCAL TP context for collective operations
            graph_config.local_tp_ctx = local_tp_ctx;
            graph_config.local_tp_device_idx = local_tp_device_idx;

            LOG_INFO("[InferenceRunner] LOCAL TP enabled: degree=" << tp_degree
                                                                   << " device_idx=" << local_tp_device_idx
                                                                   << " device=" << my_device.toString()
                                                                   << " weight=" << (my_weight * 100.0f) << "%"
                                                                   << " backend=" << static_cast<int>(local_tp_ctx->backend()));
            LOG_DEBUG("[InferenceRunner] LOCAL TP QKV: head_start=" << head_start
                                                                    << " local_n_heads=" << local_n_heads << "/" << graph_config.n_heads
                                                                    << " local_n_kv_heads=" << local_n_kv_heads << "/" << graph_config.n_kv_heads);
            LOG_DEBUG("[InferenceRunner] LOCAL TP FFN: d_ff_local=" << d_ff_local << "/" << graph_config.d_ff);
            LOG_DEBUG("[InferenceRunner] LOCAL TP LMHead: vocab_local=" << vocab_local << "/" << graph_config.vocab_size);
        }
        else if (tp_config && weights_sharded)
        {
            // =====================================================================
            // Phase 1c: Use TensorParallelConfig for proportional assignment
            // =====================================================================
            const DeviceShardingAssignment &assignment = tp_config->forRank(current_rank);

            // Store config and rank in graph_config for downstream use
            graph_config.tp_config = std::make_shared<TensorParallelConfig>(*tp_config);
            graph_config.local_rank = current_rank;

            // QKV head assignment
            graph_config.head_start = assignment.head_start;
            graph_config.local_n_heads = assignment.head_count;
            graph_config.local_n_kv_heads = assignment.kv_head_count;
            graph_config.qkv_column_parallel = true;

            // FFN dimension assignment
            graph_config.d_ff_local = assignment.d_ff_count;
            graph_config.ffn_column_parallel = true;

            // Vocab/LM head assignment
            graph_config.vocab_local = assignment.vocab_count;
            graph_config.lm_head_column_parallel = true;

            LOG_INFO("[InferenceRunner] Using TensorParallelConfig (proportional split): "
                     << "rank=" << current_rank << "/" << tp_config->worldSize()
                     << " device=" << assignment.device.to_string()
                     << " work_fraction=" << (assignment.work_fraction * 100.0f) << "%");
            LOG_DEBUG("[InferenceRunner] QKV: head_start=" << graph_config.head_start
                                                           << " local_n_heads=" << graph_config.local_n_heads << "/" << graph_config.n_heads
                                                           << " local_n_kv_heads=" << graph_config.local_n_kv_heads << "/" << graph_config.n_kv_heads);
            LOG_DEBUG("[InferenceRunner] FFN: d_ff_local=" << graph_config.d_ff_local << "/" << graph_config.d_ff);
            LOG_DEBUG("[InferenceRunner] LMHead: vocab_local=" << graph_config.vocab_local << "/" << graph_config.vocab_size);
        }
        else if (mpi_ctx && mpi_ctx->world_size() > 1 && weights_sharded)
        {
            // =====================================================================
            // Legacy: Equal 1/world_size split
            // =====================================================================
            // Compute local head distribution
            auto [q_head_start, local_n_q_heads] = mpi_ctx->get_local_slice(
                static_cast<size_t>(graph_config.n_heads));
            auto [kv_head_start, local_n_kv_h] = mpi_ctx->get_local_slice(
                static_cast<size_t>(graph_config.n_kv_heads));

            graph_config.head_start = static_cast<int>(q_head_start);
            graph_config.local_n_heads = static_cast<int>(local_n_q_heads);
            graph_config.local_n_kv_heads = static_cast<int>(local_n_kv_h);
            graph_config.qkv_column_parallel = true;
            graph_config.local_rank = current_rank;

            LOG_DEBUG("[InferenceRunner] QKV Column-Parallel enabled (equal split): "
                      << "head_start=" << graph_config.head_start
                      << ", local_n_heads=" << graph_config.local_n_heads << "/" << graph_config.n_heads
                      << ", local_n_kv_heads=" << graph_config.local_n_kv_heads << "/" << graph_config.n_kv_heads
                      << " (rank " << mpi_ctx->rank() << "/" << mpi_ctx->world_size() << ")");

            // FFN dimension (equal split)
            int world_size = mpi_ctx->world_size();
            if (graph_config.d_ff % world_size != 0)
            {
                LOG_ERROR("[InferenceRunner] d_ff (" << graph_config.d_ff
                                                     << ") not divisible by world_size (" << world_size << ")");
                throw std::runtime_error("FFN dimension not divisible by world_size for tensor parallelism");
            }
            graph_config.d_ff_local = graph_config.d_ff / world_size;
            graph_config.ffn_column_parallel = true;

            LOG_DEBUG("[InferenceRunner] FFN Column-Parallel enabled (equal split): "
                      << "d_ff_local=" << graph_config.d_ff_local << "/" << graph_config.d_ff
                      << " (rank " << mpi_ctx->rank() << "/" << world_size << ")");

            // Vocab/LM head (equal split)
            if (graph_config.vocab_size % world_size != 0)
            {
                LOG_ERROR("[InferenceRunner] vocab_size (" << graph_config.vocab_size
                                                           << ") not divisible by world_size (" << world_size << ")");
                throw std::runtime_error("Vocab size not divisible by world_size for tensor parallelism");
            }
            graph_config.vocab_local = graph_config.vocab_size / world_size;
            graph_config.lm_head_column_parallel = true;

            LOG_DEBUG("[InferenceRunner] LM Head Column-Parallel enabled (equal split): "
                      << "vocab_local=" << graph_config.vocab_local << "/" << graph_config.vocab_size
                      << " (rank " << mpi_ctx->rank() << "/" << world_size << ")");
        }
        else
        {
            // Single rank OR weights not sharded: use full dimensions (no sharding)
            graph_config.head_start = 0;
            graph_config.local_n_heads = graph_config.n_heads;
            graph_config.local_n_kv_heads = graph_config.n_kv_heads;
            graph_config.qkv_column_parallel = false;
            graph_config.local_rank = 0;

            graph_config.d_ff_local = graph_config.d_ff;
            graph_config.ffn_column_parallel = false;

            graph_config.vocab_local = graph_config.vocab_size;
            graph_config.lm_head_column_parallel = false;

            if (mpi_ctx && mpi_ctx->world_size() > 1 && !weights_sharded)
            {
                LOG_WARN("[InferenceRunner] MPI world_size > 1 but weights are REPLICATED, "
                         << "not SHARDED. Using full buffer sizes (no tensor parallelism). "
                         << "Pass WeightDistributionStrategy::SHARDED to ModelContext::create() "
                         << "to enable tensor parallelism.");
            }
        }

        LOG_DEBUG("[InferenceRunner] GraphConfig: "
                  << "vocab=" << graph_config.vocab_size
                  << ", d_model=" << graph_config.d_model
                  << ", n_layers=" << graph_config.n_layers
                  << ", n_heads=" << graph_config.n_heads
                  << ", n_kv_heads=" << graph_config.n_kv_heads
                  << ", d_ff=" << graph_config.d_ff);

        // Create DeviceGraphOrchestrator with config
        GraphCacheConfig cache_config;
        cache_config.enabled = true;
        cache_config.decode_seq_len = 1;

        LOG_DEBUG("[InferenceRunner] About to create DeviceGraphOrchestrator with mpi_ctx="
                  << (mpi_ctx ? "valid" : "nullptr")
                  << " world_size=" << (mpi_ctx ? mpi_ctx->world_size() : -1));

        auto orchestrator = std::make_unique<DeviceGraphOrchestrator>(
            graph_config, mpi_ctx, cache_config);

        // Initialize graph cache
        orchestrator->initializeGraphCache(graph_config.n_layers);

        // Initialize inference state (allocates buffers)
        // Pass mapped memory config for GPU zero-copy access
        InferenceStateInitConfig init_config;
        init_config.use_mapped_memory = config.use_mapped_memory;

        if (!orchestrator->initializeInferenceState(
                config.batch_size, config.max_seq_len, device, init_config))
        {
            LOG_ERROR("[InferenceRunner] Failed to initialize inference state");
            return nullptr;
        }

        // Load weights and configure orchestrator
        if (!configureOrchestratorWeightsImpl(orchestrator.get(), model_ctx, device))
        {
            LOG_ERROR("[InferenceRunner] Failed to configure orchestrator weights");
            return nullptr;
        }

        // =====================================================================
        // Weight Streaming (Option B) - Create streamer from environment
        // =====================================================================
        // If LLAMINAR_WEIGHT_STREAMING=1, create a LayerWeightStreamer.
        // The streamer manages GPU-side weight caching and on-demand transfers.
        // Note: weight_mgr is already declared above (line 91)
        // =====================================================================
        if (weight_mgr)
        {
            auto weight_streamer = WeightStreamerFactory::createFromEnv(
                weight_mgr, graph_config.n_layers);
            if (weight_streamer)
            {
                orchestrator->setWeightStreamer(std::move(weight_streamer));
            }
        }

        // =====================================================================
        // GPU-Native Collectives (NCCL/RCCL/PCIeBAR)
        // =====================================================================
        // Create CollectiveContext for GPU-native collective operations.
        // This eliminates GPU→CPU→GPU transfers during tensor-parallel inference:
        // - NCCL for CUDA devices
        // - RCCL for ROCm devices
        // - MPI fallback for CPU-only or heterogeneous setups
        // =====================================================================
        if (mpi_ctx && mpi_ctx->world_size() > 1)
        {
            // Build local cluster inventory (detects CUDA/ROCm GPUs)
            ClusterInventory cluster_inventory = buildLocalClusterInventory(mpi_ctx);

            // Only enable GPU collectives if we have GPUs
            if (cluster_inventory.hasAnyGPU())
            {
                // Create intra-node context which automatically selects:
                // - NCCL for all-CUDA groups
                // - RCCL for all-ROCm groups
                // - MPI fallback for mixed or CPU-only groups
                auto collective_ctx = CollectiveContextFactory::createIntraNode(
                    cluster_inventory, mpi_ctx);
                if (collective_ctx)
                {
                    orchestrator->setCollectiveContext(std::move(collective_ctx));
                    LOG_INFO("[InferenceRunner] GPU-native collectives enabled (NCCL/RCCL)");
                }
                else
                {
                    LOG_WARN("[InferenceRunner] Failed to create CollectiveContext - using CPU MPI fallback");
                }
            }
            else
            {
                LOG_DEBUG("[InferenceRunner] No GPUs detected - using CPU MPI for collectives");
            }
        }

        LOG_DEBUG("[InferenceRunner] DeviceGraphOrchestrator created successfully");

        // DeviceGraphOrchestrator implements IInferenceRunner directly
        return orchestrator;
    }

    static bool configureOrchestratorWeightsImpl(
        DeviceGraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx,
        DeviceId device)
    {
        if (!orchestrator || !model_ctx)
        {
            return false;
        }

        auto weight_mgr = model_ctx->concreteWeightManager();
        if (!weight_mgr)
        {
            LOG_ERROR("[InferenceRunner] No weight manager in model context");
            return false;
        }

        // =====================================================================
        // Set WeightManager and PlacementMap for phase-aware weight access
        // (Gap 3: CPU Decode Participation)
        // =====================================================================
        orchestrator->setWeightManager(weight_mgr);
        if (auto placement_map = model_ctx->placementMap())
        {
            orchestrator->setWeightPlacementMap(placement_map);
            LOG_DEBUG("[InferenceRunner] Phase-aware weight access configured with placement map");
        }

        // =====================================================================
        // Preload weights for target device BEFORE accessing them
        // =====================================================================
        // WeightManager packs GEMM weights and uploads non-GEMM weights to GPU.
        // This MUST happen before weights are retrieved, as GEMM kernel creation
        // requires the raw tensor data to still be available for packing.
        // =====================================================================
        DeviceType target_device = DeviceType::CPU;
        if (device.is_gpu())
        {
            // Determine if it's CUDA or ROCm based on the device
            if (device.is_rocm())
            {
                target_device = DeviceType::ROCm;
            }
            else
            {
                target_device = DeviceType::CUDA;
            }
        }

        const char *device_name = (target_device == DeviceType::CPU) ? "CPU" : (target_device == DeviceType::CUDA) ? "CUDA"
                                                                                                                   : "ROCm";

        // Build Qwen2ModelWeights
        Qwen2ModelWeights weights;

        // Get global weights
        auto embedding = weight_mgr->getWeight("token_embd.weight");
        auto final_norm = weight_mgr->getWeight("output_norm.weight");
        auto lm_head = weight_mgr->getWeight("output.weight");

        if (!embedding || !final_norm || !lm_head)
        {
            LOG_ERROR("[InferenceRunner] Missing global weights");
            return false;
        }

        weights.embedding_table = embedding.get();
        weights.final_norm = final_norm.get();
        weights.lm_head = lm_head.get();

        // =====================================================================
        // Eager load ALL layer weights into cache BEFORE preloading
        // =====================================================================
        // Use schema factory to determine which weights are required vs optional.
        // This ensures consistent handling: required weights fail if missing,
        // optional weights (like QKV biases) silently skip.
        Qwen2SchemaFactory schema_factory;

        int n_layers = model_ctx->blockCount();
        LOG_DEBUG("[InferenceRunner] Eagerly loading " << n_layers << " layers of weights...");
        for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
        {
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // All possible layer weights (schema determines which are required)
            const std::vector<std::string> layer_weights = {
                // Attention weights (required)
                prefix + "attn_q.weight",
                prefix + "attn_k.weight",
                prefix + "attn_v.weight",
                prefix + "attn_output.weight",
                prefix + "attn_norm.weight",
                // Attention biases (optional per schema)
                prefix + "attn_q.bias",
                prefix + "attn_k.bias",
                prefix + "attn_v.bias",
                // FFN weights (required)
                prefix + "ffn_gate.weight",
                prefix + "ffn_up.weight",
                prefix + "ffn_down.weight",
                prefix + "ffn_norm.weight"};

            for (const auto &weight_name : layer_weights)
            {
                auto weight = weight_mgr->getWeight(weight_name);
                bool is_optional = schema_factory.isWeightOptional(weight_name);

                if (!weight)
                {
                    if (is_optional)
                    {
                        // Optional weight missing - this is fine (e.g., model without QKV biases)
                        LOG_TRACE("[InferenceRunner] Optional weight not present: " << weight_name);
                    }
                    else
                    {
                        // Required weight missing - this is an error
                        LOG_ERROR("[InferenceRunner] Failed to load required weight: " << weight_name);
                        return false;
                    }
                }
            }
        }
        LOG_DEBUG("[InferenceRunner] All layer weights loaded into cache");

        // =====================================================================
        // NOW preload weights for target device (GPU packing/upload)
        // =====================================================================
        // Pack GEMM weights (creates kernels and uploads to GPU for GPU targets)
        if (!weight_mgr->packGemmWeights(device))
        {
            LOG_WARN("[InferenceRunner] Weight packing failed for device "
                     << device_name << ", will use lazy kernel creation");
            // Not fatal - kernels will be created lazily on first use
        }
        else
        {
            LOG_DEBUG("[InferenceRunner] Packed GEMM weights for " << device_name);
        }

        // Upload non-GEMM weights (norms, embeddings) to GPU
        if (!weight_mgr->uploadNonGemmWeights(device))
        {
            LOG_WARN("[InferenceRunner] Non-GEMM weight upload failed for device "
                     << device_name);
        }
        else
        {
            LOG_DEBUG("[InferenceRunner] Uploaded non-GEMM weights for " << device_name);
        }

        // Layer weight accessor - capture weight_mgr by value (shared_ptr copy)
        // Weights are now in cache, so these calls will be fast
        weights.get_layer_weights = [weight_mgr](int layer_idx) -> Qwen2LayerWeights
        {
            Qwen2LayerWeights layer;
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // Attention weights - all required (should be in cache from eager load)
            auto wq = weight_mgr->getWeight(prefix + "attn_q.weight");
            auto wk = weight_mgr->getWeight(prefix + "attn_k.weight");
            auto wv = weight_mgr->getWeight(prefix + "attn_v.weight");
            auto wo = weight_mgr->getWeight(prefix + "attn_output.weight");
            auto attn_norm = weight_mgr->getWeight(prefix + "attn_norm.weight");

            // Validate required attention weights
            if (!wq || !wk || !wv || !wo || !attn_norm)
            {
                LOG_ERROR("[InferenceRunner] Missing required attention weight for layer " << layer_idx);
                // Return empty layer - caller should check for nullptr members
                return layer;
            }

            layer.wq = wq.get();
            layer.wk = wk.get();
            layer.wv = wv.get();
            layer.wo = wo.get();
            layer.attn_norm = attn_norm.get();

            // Attention biases (may be null)
            auto q_bias = weight_mgr->getWeight(prefix + "attn_q.bias");
            auto k_bias = weight_mgr->getWeight(prefix + "attn_k.bias");
            auto v_bias = weight_mgr->getWeight(prefix + "attn_v.bias");
            layer.q_bias = q_bias ? q_bias.get() : nullptr;
            layer.k_bias = k_bias ? k_bias.get() : nullptr;
            layer.v_bias = v_bias ? v_bias.get() : nullptr;

            // FFN weights - all required
            auto gate_proj = weight_mgr->getWeight(prefix + "ffn_gate.weight");
            auto up_proj = weight_mgr->getWeight(prefix + "ffn_up.weight");
            auto down_proj = weight_mgr->getWeight(prefix + "ffn_down.weight");
            auto ffn_norm = weight_mgr->getWeight(prefix + "ffn_norm.weight");

            if (!gate_proj || !up_proj || !down_proj || !ffn_norm)
            {
                LOG_ERROR("[InferenceRunner] Missing required FFN weight for layer " << layer_idx);
                // Return partially filled layer - caller should check for nullptr members
                return layer;
            }

            layer.gate_proj = gate_proj.get();
            layer.up_proj = up_proj.get();
            layer.down_proj = down_proj.get();
            layer.ffn_norm = ffn_norm.get();

            return layer;
        };

        orchestrator->setWeights(weights);
        LOG_DEBUG("[InferenceRunner] Weights configured on orchestrator");
        return true;
    }

    // =========================================================================
    // Testable Factory Function (Interface-Based)
    // =========================================================================

    std::unique_ptr<IInferenceRunner> createTestableInferenceRunner(
        std::shared_ptr<IModelContext> model_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config)
    {
        LOG_DEBUG("[InferenceRunner] createTestableInferenceRunner called");

        if (!model_ctx)
        {
            LOG_ERROR("[InferenceRunner] model_ctx is null");
            return nullptr;
        }

        // Validate device
        if (!device.is_valid())
        {
            LOG_ERROR("[InferenceRunner] Invalid device " << device
                                                          << ". Use DeviceId::cpu() for CPU.");
            return nullptr;
        }
        LOG_DEBUG("[InferenceRunner] Using device " << device);

        // Currently only Qwen2 is supported
        std::string architecture = model_ctx->architecture();
        if (architecture != "qwen2")
        {
            LOG_ERROR("[InferenceRunner] Only qwen2 architecture is supported, got: " << architecture);
            return nullptr;
        }

        // Build Qwen2GraphConfig from IModelContext
        Qwen2GraphConfig graph_config;
        graph_config.vocab_size = model_ctx->vocabSize();
        graph_config.d_model = model_ctx->embeddingLength();
        graph_config.n_layers = model_ctx->blockCount();
        graph_config.n_heads = model_ctx->headCount();
        graph_config.n_kv_heads = model_ctx->headCountKV();
        graph_config.head_dim = graph_config.d_model / graph_config.n_heads;
        graph_config.max_seq_len = config.max_seq_len;
        graph_config.default_device = device;
        graph_config.activation_precision = config.activation_precision;
        graph_config.fused_attention_backend = config.fused_attention_backend;
        graph_config.kv_cache_scale = config.kv_cache_scale;

        // Get d_ff from model context if available, otherwise estimate as ~4x d_model
        int d_ff = model_ctx->feedForwardLength();
        if (d_ff > 0)
        {
            graph_config.d_ff = d_ff;
        }
        else
        {
            graph_config.d_ff = graph_config.d_model * 4;
            LOG_WARN("[InferenceRunner] feedForwardLength() returned 0, using estimate: " << graph_config.d_ff);
        }

        // Check for LOCAL TP configuration
        ILocalTPContext *local_tp_ctx = config.local_tp_ctx;
        const int local_tp_device_idx = config.local_tp_device_index;

        if (local_tp_ctx && local_tp_ctx->degree() > 1)
        {
            // =====================================================================
            // LOCAL TP: Single rank with multiple devices via ILocalTPContext
            // Mirrors the logic in createDeviceGraphOrchestratorImpl (lines 294-369)
            // =====================================================================
            const int tp_degree = local_tp_ctx->degree();
            const auto &devices = local_tp_ctx->devices();
            const auto &weights = local_tp_ctx->weights();

            if (local_tp_device_idx < 0 || local_tp_device_idx >= tp_degree)
            {
                LOG_ERROR("[InferenceRunner] Invalid local_tp_device_index: " << local_tp_device_idx
                                                                              << " (degree=" << tp_degree << ")");
                return nullptr;
            }

            const GlobalDeviceAddress &my_device = devices[local_tp_device_idx];
            const float my_weight = weights.empty() ? (1.0f / tp_degree) : weights[local_tp_device_idx];

            // Compute head assignment based on weight/index
            int head_start = 0;
            int kv_head_start = 0;
            int d_ff_start = 0;
            int vocab_start = 0;

            // Accumulate starts from devices before this one
            for (int i = 0; i < local_tp_device_idx; ++i)
            {
                float w = weights.empty() ? (1.0f / tp_degree) : weights[i];
                head_start += static_cast<int>(std::round(w * graph_config.n_heads));
                kv_head_start += static_cast<int>(std::round(w * graph_config.n_kv_heads));
                d_ff_start += static_cast<int>(std::round(w * graph_config.d_ff));
                vocab_start += static_cast<int>(std::round(w * graph_config.vocab_size));
            }

            // Compute counts for this device
            int local_n_heads = static_cast<int>(std::round(my_weight * graph_config.n_heads));
            int local_n_kv_heads = static_cast<int>(std::round(my_weight * graph_config.n_kv_heads));
            int d_ff_local = static_cast<int>(std::round(my_weight * graph_config.d_ff));
            int vocab_local = static_cast<int>(std::round(my_weight * graph_config.vocab_size));

            // Handle rounding: last device gets remainder
            if (local_tp_device_idx == tp_degree - 1)
            {
                local_n_heads = graph_config.n_heads - head_start;
                local_n_kv_heads = graph_config.n_kv_heads - kv_head_start;
                d_ff_local = graph_config.d_ff - d_ff_start;
                vocab_local = graph_config.vocab_size - vocab_start;
            }

            graph_config.head_start = head_start;
            graph_config.local_n_heads = local_n_heads;
            graph_config.local_n_kv_heads = local_n_kv_heads;
            graph_config.qkv_column_parallel = true;
            graph_config.local_rank = local_tp_device_idx;

            graph_config.d_ff_local = d_ff_local;
            graph_config.ffn_column_parallel = true;

            graph_config.vocab_local = vocab_local;
            graph_config.lm_head_column_parallel = true;

            // Store LOCAL TP context for collective operations
            graph_config.local_tp_ctx = local_tp_ctx;
            graph_config.local_tp_device_idx = local_tp_device_idx;

            LOG_INFO("[InferenceRunner] Testable LOCAL TP enabled: degree=" << tp_degree
                                                                            << " device_idx=" << local_tp_device_idx
                                                                            << " device=" << my_device.toString()
                                                                            << " weight=" << (my_weight * 100.0f) << "%"
                                                                            << " backend=" << static_cast<int>(local_tp_ctx->backend()));
            LOG_DEBUG("[InferenceRunner] Testable LOCAL TP QKV: head_start=" << head_start
                                                                             << " local_n_heads=" << local_n_heads << "/" << graph_config.n_heads
                                                                             << " local_n_kv_heads=" << local_n_kv_heads << "/" << graph_config.n_kv_heads);
            LOG_DEBUG("[InferenceRunner] Testable LOCAL TP FFN: d_ff_local=" << d_ff_local << "/" << graph_config.d_ff);
            LOG_DEBUG("[InferenceRunner] Testable LOCAL TP LMHead: vocab_local=" << vocab_local << "/" << graph_config.vocab_size);
        }
        else
        {
            // Single rank configuration (testable runner doesn't use MPI by default)
            graph_config.head_start = 0;
            graph_config.local_n_heads = graph_config.n_heads;
            graph_config.local_n_kv_heads = graph_config.n_kv_heads;
            graph_config.qkv_column_parallel = false;
            graph_config.d_ff_local = graph_config.d_ff;
            graph_config.ffn_column_parallel = false;
            graph_config.vocab_local = graph_config.vocab_size;
            graph_config.lm_head_column_parallel = false;
        }

        LOG_DEBUG("[InferenceRunner] TestableGraphConfig: "
                  << "vocab=" << graph_config.vocab_size
                  << ", d_model=" << graph_config.d_model
                  << ", n_layers=" << graph_config.n_layers
                  << ", n_heads=" << graph_config.n_heads
                  << ", n_kv_heads=" << graph_config.n_kv_heads
                  << ", d_ff=" << graph_config.d_ff);

        // Create Dependencies struct
        DeviceGraphOrchestrator::Dependencies deps;
        deps.model_ctx = model_ctx;
        // topology and collective_ctx left as nullptr for single-rank testing

        // Create DeviceGraphOrchestrator with injected dependencies
        GraphCacheConfig cache_config;
        cache_config.enabled = true;
        cache_config.decode_seq_len = 1;

        auto orchestrator = std::make_unique<DeviceGraphOrchestrator>(
            std::move(deps), graph_config, cache_config);

        // Initialize graph cache
        orchestrator->initializeGraphCache(graph_config.n_layers);

        // Initialize inference state (allocates buffers)
        // Pass mapped memory config for GPU zero-copy access
        InferenceStateInitConfig init_config;
        init_config.use_mapped_memory = config.use_mapped_memory;

        if (!orchestrator->initializeInferenceState(
                config.batch_size, config.max_seq_len, device, init_config))
        {
            LOG_ERROR("[InferenceRunner] Failed to initialize inference state");
            return nullptr;
        }

        // Configure weights from IModelContext
        // Build Qwen2ModelWeights using IModelContext::getWeightForDevice
        //
        // IMPORTANT: Use getWeightForDevice() instead of getWeight() to get
        // device-specific tensor instances. This is critical for multi-device
        // scenarios where each device needs its own tensor for coherence tracking.
        // The WeightManager handles cloning automatically when called from
        // different devices.
        Qwen2ModelWeights weights;

        // Get global weights via interface - use device-specific instances
        auto embedding = model_ctx->getWeightForDevice("token_embd.weight", device);
        auto final_norm = model_ctx->getWeightForDevice("output_norm.weight", device);
        auto lm_head = model_ctx->getWeightForDevice("output.weight", device);

        if (!embedding || !final_norm || !lm_head)
        {
            LOG_ERROR("[InferenceRunner] Missing global weights from IModelContext");
            return nullptr;
        }

        weights.embedding_table = embedding.get();
        weights.final_norm = final_norm.get();
        weights.lm_head = lm_head.get();

        // Layer weight accessor using IModelContext interface
        // Capture model_ctx AND device by value for lambda lifetime
        weights.get_layer_weights = [model_ctx, device](int layer_idx) -> Qwen2LayerWeights
        {
            Qwen2LayerWeights layer;
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // Attention weights - device-specific instances
            layer.wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", device).get();
            layer.wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", device).get();
            layer.wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", device).get();
            layer.wo = model_ctx->getWeightForDevice(prefix + "attn_output.weight", device).get();
            layer.attn_norm = model_ctx->getWeightForDevice(prefix + "attn_norm.weight", device).get();

            // Attention biases (may be null)
            auto q_bias = model_ctx->getWeightForDevice(prefix + "attn_q.bias", device);
            auto k_bias = model_ctx->getWeightForDevice(prefix + "attn_k.bias", device);
            auto v_bias = model_ctx->getWeightForDevice(prefix + "attn_v.bias", device);
            layer.q_bias = q_bias ? q_bias.get() : nullptr;
            layer.k_bias = k_bias ? k_bias.get() : nullptr;
            layer.v_bias = v_bias ? v_bias.get() : nullptr;

            // FFN weights
            layer.gate_proj = model_ctx->getWeightForDevice(prefix + "ffn_gate.weight", device).get();
            layer.up_proj = model_ctx->getWeightForDevice(prefix + "ffn_up.weight", device).get();
            layer.down_proj = model_ctx->getWeightForDevice(prefix + "ffn_down.weight", device).get();
            layer.ffn_norm = model_ctx->getWeightForDevice(prefix + "ffn_norm.weight", device).get();

            return layer;
        };

        orchestrator->setWeights(weights);
        LOG_INFO("[InferenceRunner] Testable DeviceGraphOrchestrator created successfully");

        return orchestrator;
    }

    // =========================================================================
    // Multi-Device Orchestrator Factory Functions
    // =========================================================================

    std::unique_ptr<IMultiDeviceOrchestrator> createMultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const MultiDeviceOrchestrator::Config &config)
    {
        if (!model_ctx)
        {
            LOG_ERROR("[InferenceRunner] model_ctx is null for createMultiDeviceOrchestrator");
            return nullptr;
        }

        if (!tp_ctx)
        {
            LOG_ERROR("[InferenceRunner] tp_ctx is null for createMultiDeviceOrchestrator");
            return nullptr;
        }

        if (!config.validate())
        {
            LOG_ERROR("[InferenceRunner] Invalid MultiDeviceOrchestrator config");
            return nullptr;
        }

        LOG_INFO("[InferenceRunner] Creating MultiDeviceOrchestrator with "
                 << config.devices.size() << " devices, backend="
                 << static_cast<int>(config.backend));

        try
        {
            return std::make_unique<MultiDeviceOrchestrator>(
                model_ctx, std::move(tp_ctx), config);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[InferenceRunner] Failed to create MultiDeviceOrchestrator: " << e.what());
            return nullptr;
        }
    }

    std::unique_ptr<IMultiDeviceOrchestrator> createTestableMultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const MultiDeviceOrchestrator::Config &config)
    {
        if (!model_ctx)
        {
            LOG_ERROR("[InferenceRunner] model_ctx is null for createTestableMultiDeviceOrchestrator");
            return nullptr;
        }

        if (device_runners.empty())
        {
            LOG_ERROR("[InferenceRunner] device_runners is empty");
            return nullptr;
        }

        LOG_DEBUG("[InferenceRunner] Creating testable MultiDeviceOrchestrator with "
                  << device_runners.size() << " injected runners");

        try
        {
            return MultiDeviceOrchestrator::createForTest(
                model_ctx, std::move(device_runners), std::move(tp_ctx), config);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[InferenceRunner] Failed to create testable MultiDeviceOrchestrator: " << e.what());
            return nullptr;
        }
    }

} // namespace llaminar2
