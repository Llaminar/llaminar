/**
 * @file InferenceRunnerFactory.cpp
 * @brief Factory implementation for creating IInferenceRunner instances
 * @author David Sanftenberg
 * @date December 2025
 */

#include "InferenceRunnerFactory.h"
#include "../backends/DeviceId.h"
#include "../models/qwen/Qwen2Graph.h"
#include "../models/qwen/Qwen2Schema.h"
#include "GraphOrchestrator.h"
#include "../loaders/ModelContext.h"
#include "../loaders/ModelLoader.h"
#include "../loaders/WeightManager.h"
#include "../loaders/WeightPreloader.h"
#include "../loaders/WeightStreamerFactory.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    // Forward declarations of factory helpers
    static std::unique_ptr<IInferenceRunner> createGraphOrchestratorImpl(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        DeviceId device,
        const InferenceRunnerConfig &config,
        const std::string &architecture);

    static bool configureOrchestratorWeightsImpl(
        GraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx,
        DeviceId device);

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
        LOG_INFO("[InferenceRunner] Using GRAPH path");
        return createGraphOrchestratorImpl(model_ctx, mpi_ctx, device, config, architecture);
    }

    // =========================================================================
    // Factory Helper Implementations
    // =========================================================================

    static std::unique_ptr<IInferenceRunner> createGraphOrchestratorImpl(
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
        // Phase 3: Tensor-Parallel Configuration for Column-Parallel QKV
        // =====================================================================
        // When running with multiple MPI ranks AND weights are sharded, compute head distribution:
        // - Each rank handles local_n_heads = n_heads / world_size
        // - Each rank handles local_n_kv_heads = n_kv_heads / world_size (for GQA)
        // - head_start identifies which head range this rank owns
        //
        // Weight sharding in WeightManager uses COLUMN_PARALLEL for Q/K/V:
        // - Q: [n_heads * head_dim, d_model] → [local_n_heads * head_dim, d_model]
        // - K/V: [n_kv_heads * head_dim, d_model] → [local_n_kv_heads * head_dim, d_model]
        //
        // IMPORTANT: Only enable tensor parallelism if weights are actually sharded.
        // If weights are REPLICATED, each rank has the full weight and should use
        // the full head counts to avoid buffer/weight dimension mismatch.
        // =====================================================================
        const bool weights_sharded = weight_mgr &&
                                     (weight_mgr->strategy() == WeightDistributionStrategy::SHARDED);
        if (mpi_ctx && mpi_ctx->world_size() > 1 && weights_sharded)
        {
            // Compute local head distribution
            auto [q_head_start, local_n_q_heads] = mpi_ctx->get_local_slice(
                static_cast<size_t>(graph_config.n_heads));
            auto [kv_head_start, local_n_kv_h] = mpi_ctx->get_local_slice(
                static_cast<size_t>(graph_config.n_kv_heads));

            graph_config.head_start = static_cast<int>(q_head_start);
            graph_config.local_n_heads = static_cast<int>(local_n_q_heads);
            graph_config.local_n_kv_heads = static_cast<int>(local_n_kv_h);
            graph_config.qkv_column_parallel = true;

            LOG_INFO("[InferenceRunner] QKV Column-Parallel enabled: "
                     << "head_start=" << graph_config.head_start
                     << ", local_n_heads=" << graph_config.local_n_heads << "/" << graph_config.n_heads
                     << ", local_n_kv_heads=" << graph_config.local_n_kv_heads << "/" << graph_config.n_kv_heads
                     << " (rank " << mpi_ctx->rank() << "/" << mpi_ctx->world_size() << ")");
        }
        else
        {
            // Single rank OR weights not sharded: use full head counts
            graph_config.head_start = 0;
            graph_config.local_n_heads = graph_config.n_heads;
            graph_config.local_n_kv_heads = graph_config.n_kv_heads;
            graph_config.qkv_column_parallel = false;

            if (mpi_ctx && mpi_ctx->world_size() > 1 && !weights_sharded)
            {
                LOG_WARN("[InferenceRunner] MPI world_size > 1 but weights are REPLICATED, "
                         << "not SHARDED. Using full buffer sizes (no tensor parallelism). "
                         << "Pass WeightDistributionStrategy::SHARDED to ModelContext::create() "
                         << "to enable tensor parallelism.");
            }
        }

        // =====================================================================
        // Phase 4: Tensor-Parallel Configuration for Column-Parallel FFN
        // =====================================================================
        // When running with multiple MPI ranks AND weights are sharded, compute local FFN dimension:
        // - Each rank handles d_ff_local = d_ff / world_size
        // - Gate/Up weights: [d_ff, d_model] → [d_ff_local, d_model]
        // - Down weight remains row-parallel: [d_model, d_ff_local] per rank
        // =====================================================================
        if (mpi_ctx && mpi_ctx->world_size() > 1 && weights_sharded)
        {
            int world_size = mpi_ctx->world_size();
            if (graph_config.d_ff % world_size != 0)
            {
                LOG_ERROR("[InferenceRunner] d_ff (" << graph_config.d_ff
                                                     << ") not divisible by world_size (" << world_size << ")");
                throw std::runtime_error("FFN dimension not divisible by world_size for tensor parallelism");
            }
            graph_config.d_ff_local = graph_config.d_ff / world_size;
            graph_config.ffn_column_parallel = true;

            LOG_INFO("[InferenceRunner] FFN Column-Parallel enabled: "
                     << "d_ff_local=" << graph_config.d_ff_local << "/" << graph_config.d_ff
                     << " (rank " << mpi_ctx->rank() << "/" << world_size << ")");
        }
        else
        {
            // Single rank: use full FFN dimension (no sharding)
            graph_config.d_ff_local = graph_config.d_ff;
            graph_config.ffn_column_parallel = false;
        }

        // =====================================================================
        // Phase 5: Tensor-Parallel Configuration for Column-Parallel LM Head
        // =====================================================================
        // When running with multiple MPI ranks AND weights are sharded, compute local vocab dimension:
        // - Each rank handles vocab_local = vocab_size / world_size
        // - LM head weight: [vocab_size, d_model] → [vocab_local, d_model]
        // - Output: [seq, vocab_local] per rank, then AllGather to [seq, vocab_size]
        // =====================================================================
        if (mpi_ctx && mpi_ctx->world_size() > 1 && weights_sharded)
        {
            int world_size = mpi_ctx->world_size();
            if (graph_config.vocab_size % world_size != 0)
            {
                LOG_ERROR("[InferenceRunner] vocab_size (" << graph_config.vocab_size
                                                           << ") not divisible by world_size (" << world_size << ")");
                throw std::runtime_error("Vocab size not divisible by world_size for tensor parallelism");
            }
            graph_config.vocab_local = graph_config.vocab_size / world_size;
            graph_config.lm_head_column_parallel = true;

            LOG_INFO("[InferenceRunner] LM Head Column-Parallel enabled: "
                     << "vocab_local=" << graph_config.vocab_local << "/" << graph_config.vocab_size
                     << " (rank " << mpi_ctx->rank() << "/" << world_size << ")");
        }
        else
        {
            // Single rank: use full vocab size (no sharding)
            graph_config.vocab_local = graph_config.vocab_size;
            graph_config.lm_head_column_parallel = false;
        }

        LOG_DEBUG("[InferenceRunner] GraphConfig: "
                  << "vocab=" << graph_config.vocab_size
                  << ", d_model=" << graph_config.d_model
                  << ", n_layers=" << graph_config.n_layers
                  << ", n_heads=" << graph_config.n_heads
                  << ", n_kv_heads=" << graph_config.n_kv_heads
                  << ", d_ff=" << graph_config.d_ff);

        // Create GraphOrchestrator with config
        GraphCacheConfig cache_config;
        cache_config.enabled = true;
        cache_config.decode_seq_len = 1;

        LOG_DEBUG("[InferenceRunner] About to create GraphOrchestrator with mpi_ctx="
                  << (mpi_ctx ? "valid" : "nullptr")
                  << " world_size=" << (mpi_ctx ? mpi_ctx->world_size() : -1));

        auto orchestrator = std::make_unique<GraphOrchestrator>(
            graph_config, mpi_ctx, cache_config);

        // Initialize graph cache
        orchestrator->initializeGraphCache(graph_config.n_layers);

        // Initialize inference state (allocates buffers)
        if (!orchestrator->initializeInferenceState(
                config.batch_size, config.max_seq_len, device))
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

        LOG_INFO("[InferenceRunner] GraphOrchestrator created successfully");

        // GraphOrchestrator implements IInferenceRunner directly
        return orchestrator;
    }

    static bool configureOrchestratorWeightsImpl(
        GraphOrchestrator *orchestrator,
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
        // WeightPreloader creates device-targeted GEMM kernels (CPU, CUDA, or ROCm) and
        // optionally releases raw tensor data to save memory.
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
        // This ensures weights are in cache for WeightPreloader to iterate
        int n_layers = model_ctx->blockCount();
        LOG_INFO("[InferenceRunner] Eagerly loading " << n_layers << " layers of weights...");
        for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx)
        {
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";
            // Load all layer weights into cache
            weight_mgr->getWeight(prefix + "attn_q.weight");
            weight_mgr->getWeight(prefix + "attn_k.weight");
            weight_mgr->getWeight(prefix + "attn_v.weight");
            weight_mgr->getWeight(prefix + "attn_output.weight");
            weight_mgr->getWeight(prefix + "attn_norm.weight");
            weight_mgr->getWeight(prefix + "ffn_gate.weight");
            weight_mgr->getWeight(prefix + "ffn_up.weight");
            weight_mgr->getWeight(prefix + "ffn_down.weight");
            weight_mgr->getWeight(prefix + "ffn_norm.weight");
            // Biases (may not exist)
            weight_mgr->getWeight(prefix + "attn_q.bias");
            weight_mgr->getWeight(prefix + "attn_k.bias");
            weight_mgr->getWeight(prefix + "attn_v.bias");
        }
        LOG_INFO("[InferenceRunner] All layer weights loaded into cache");

        // =====================================================================
        // NOW preload weights for target device (GPU packing/upload)
        // =====================================================================
        // Use full DeviceId to preserve ordinal for multi-GPU setups
        WeightPreloader preloader(weight_mgr);
        if (!preloader.preloadForDevice(device))
        {
            LOG_WARN("[InferenceRunner] Weight preloading failed for device "
                     << device_name << ", will use lazy kernel creation");
            // Not fatal - kernels will be created lazily on first use
        }
        else
        {
            LOG_INFO("[InferenceRunner] Preloaded weights for " << device_name);
        }

        // Layer weight accessor - capture weight_mgr by value (shared_ptr copy)
        // Weights are now in cache, so these calls will be fast
        weights.get_layer_weights = [weight_mgr](int layer_idx) -> Qwen2LayerWeights
        {
            Qwen2LayerWeights layer;
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // Attention weights
            layer.wq = weight_mgr->getWeight(prefix + "attn_q.weight").get();
            layer.wk = weight_mgr->getWeight(prefix + "attn_k.weight").get();
            layer.wv = weight_mgr->getWeight(prefix + "attn_v.weight").get();
            layer.wo = weight_mgr->getWeight(prefix + "attn_output.weight").get();
            layer.attn_norm = weight_mgr->getWeight(prefix + "attn_norm.weight").get();

            // Attention biases (may be null)
            auto q_bias = weight_mgr->getWeight(prefix + "attn_q.bias");
            auto k_bias = weight_mgr->getWeight(prefix + "attn_k.bias");
            auto v_bias = weight_mgr->getWeight(prefix + "attn_v.bias");
            layer.q_bias = q_bias ? q_bias.get() : nullptr;
            layer.k_bias = k_bias ? k_bias.get() : nullptr;
            layer.v_bias = v_bias ? v_bias.get() : nullptr;

            // FFN weights
            layer.gate_proj = weight_mgr->getWeight(prefix + "ffn_gate.weight").get();
            layer.up_proj = weight_mgr->getWeight(prefix + "ffn_up.weight").get();
            layer.down_proj = weight_mgr->getWeight(prefix + "ffn_down.weight").get();
            layer.ffn_norm = weight_mgr->getWeight(prefix + "ffn_norm.weight").get();

            return layer;
        };

        orchestrator->setWeights(weights);
        LOG_INFO("[InferenceRunner] Weights configured on orchestrator");
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

        // Estimate d_ff as ~4x d_model (common SwiGLU ratio) - tests can override
        // In real usage, IModelContext implementations can provide accurate d_ff
        graph_config.d_ff = graph_config.d_model * 4;

        // Single rank configuration (testable runner doesn't use MPI by default)
        graph_config.head_start = 0;
        graph_config.local_n_heads = graph_config.n_heads;
        graph_config.local_n_kv_heads = graph_config.n_kv_heads;
        graph_config.qkv_column_parallel = false;
        graph_config.d_ff_local = graph_config.d_ff;
        graph_config.ffn_column_parallel = false;
        graph_config.vocab_local = graph_config.vocab_size;
        graph_config.lm_head_column_parallel = false;

        LOG_DEBUG("[InferenceRunner] TestableGraphConfig: "
                  << "vocab=" << graph_config.vocab_size
                  << ", d_model=" << graph_config.d_model
                  << ", n_layers=" << graph_config.n_layers
                  << ", n_heads=" << graph_config.n_heads
                  << ", n_kv_heads=" << graph_config.n_kv_heads
                  << ", d_ff=" << graph_config.d_ff);

        // Create Dependencies struct
        GraphOrchestrator::Dependencies deps;
        deps.model_ctx = model_ctx;
        // topology and collective_ctx left as nullptr for single-rank testing

        // Create GraphOrchestrator with injected dependencies
        GraphCacheConfig cache_config;
        cache_config.enabled = true;
        cache_config.decode_seq_len = 1;

        auto orchestrator = std::make_unique<GraphOrchestrator>(
            std::move(deps), graph_config, cache_config);

        // Initialize graph cache
        orchestrator->initializeGraphCache(graph_config.n_layers);

        // Initialize inference state (allocates buffers)
        if (!orchestrator->initializeInferenceState(
                config.batch_size, config.max_seq_len, device))
        {
            LOG_ERROR("[InferenceRunner] Failed to initialize inference state");
            return nullptr;
        }

        // Configure weights from IModelContext
        // Build Qwen2ModelWeights using IModelContext::getWeight
        Qwen2ModelWeights weights;

        // Get global weights via interface
        auto embedding = model_ctx->getWeight("token_embd.weight");
        auto final_norm = model_ctx->getWeight("output_norm.weight");
        auto lm_head = model_ctx->getWeight("output.weight");

        if (!embedding || !final_norm || !lm_head)
        {
            LOG_ERROR("[InferenceRunner] Missing global weights from IModelContext");
            return nullptr;
        }

        weights.embedding_table = embedding.get();
        weights.final_norm = final_norm.get();
        weights.lm_head = lm_head.get();

        // Layer weight accessor using IModelContext interface
        // Capture model_ctx by value (shared_ptr copy) for lambda lifetime
        weights.get_layer_weights = [model_ctx](int layer_idx) -> Qwen2LayerWeights
        {
            Qwen2LayerWeights layer;
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // Attention weights
            layer.wq = model_ctx->getWeight(prefix + "attn_q.weight").get();
            layer.wk = model_ctx->getWeight(prefix + "attn_k.weight").get();
            layer.wv = model_ctx->getWeight(prefix + "attn_v.weight").get();
            layer.wo = model_ctx->getWeight(prefix + "attn_output.weight").get();
            layer.attn_norm = model_ctx->getWeight(prefix + "attn_norm.weight").get();

            // Attention biases (may be null)
            auto q_bias = model_ctx->getWeight(prefix + "attn_q.bias");
            auto k_bias = model_ctx->getWeight(prefix + "attn_k.bias");
            auto v_bias = model_ctx->getWeight(prefix + "attn_v.bias");
            layer.q_bias = q_bias ? q_bias.get() : nullptr;
            layer.k_bias = k_bias ? k_bias.get() : nullptr;
            layer.v_bias = v_bias ? v_bias.get() : nullptr;

            // FFN weights
            layer.gate_proj = model_ctx->getWeight(prefix + "ffn_gate.weight").get();
            layer.up_proj = model_ctx->getWeight(prefix + "ffn_up.weight").get();
            layer.down_proj = model_ctx->getWeight(prefix + "ffn_down.weight").get();
            layer.ffn_norm = model_ctx->getWeight(prefix + "ffn_norm.weight").get();

            return layer;
        };

        orchestrator->setWeights(weights);
        LOG_INFO("[InferenceRunner] Testable GraphOrchestrator created successfully");

        return orchestrator;
    }

} // namespace llaminar2
