/**
 * @file InferenceRunnerFactory.cpp
 * @brief Factory implementation for creating IInferenceRunner instances
 * @author David Sanftenberg
 * @date December 2025
 */

#include "InferenceRunnerFactory.h"
#include "../models/qwen/Qwen2Graph.h"
#include "../models/qwen/Qwen2Schema.h"
#include "GraphOrchestrator.h"
#include "../loaders/ModelContext.h"
#include "../loaders/ModelLoader.h"
#include "../loaders/WeightManager.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    // Forward declarations of factory helpers
    static std::unique_ptr<IInferenceRunner> createGraphOrchestratorImpl(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
        const InferenceRunnerConfig &config,
        const std::string &architecture);

    static bool configureOrchestratorWeightsImpl(
        GraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx);

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<IInferenceRunner> createInferenceRunner(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
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

        // Validate device index - no magic values allowed
        auto &dm = DeviceManager::instance();
        if (!dm.isValidDeviceIndex(device_idx))
        {
            LOG_ERROR("[InferenceRunner] Invalid device_idx " << device_idx
                                                              << ". Use DeviceManager::cpuDeviceIndex() for CPU, not -1.");
            return nullptr;
        }
        LOG_DEBUG("[InferenceRunner] Using device index " << device_idx
                                                          << " (type: " << static_cast<int>(dm.devices()[device_idx].type) << ")");

        // Graph is the only execution path (as of January 2025 cleanup)
        std::string architecture = model_ctx->architecture();
        LOG_INFO("[InferenceRunner] Using GRAPH path");
        return createGraphOrchestratorImpl(model_ctx, mpi_ctx, device_idx, config, architecture);
    }

    // =========================================================================
    // Factory Helper Implementations
    // =========================================================================

    static std::unique_ptr<IInferenceRunner> createGraphOrchestratorImpl(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
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
        auto weight_mgr = model_ctx->weightManager();
        if (weight_mgr)
        {
            Qwen2SchemaFactory schema_factory;
            weight_mgr->setWeightShardingConfig(schema_factory.getWeightShardingConfig());
            LOG_DEBUG("[InferenceRunner] Applied Qwen2 sharding config to WeightManager");
        }

        // Get model metadata
        auto &loader = model_ctx->loader();
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
        // When running with multiple MPI ranks, compute head distribution:
        // - Each rank handles local_n_heads = n_heads / world_size
        // - Each rank handles local_n_kv_heads = n_kv_heads / world_size (for GQA)
        // - head_start identifies which head range this rank owns
        //
        // Weight sharding in WeightManager uses COLUMN_PARALLEL for Q/K/V:
        // - Q: [n_heads * head_dim, d_model] → [local_n_heads * head_dim, d_model]
        // - K/V: [n_kv_heads * head_dim, d_model] → [local_n_kv_heads * head_dim, d_model]
        // =====================================================================
        if (mpi_ctx && mpi_ctx->world_size() > 1)
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
            // Single rank: use full head counts (no sharding)
            graph_config.head_start = 0;
            graph_config.local_n_heads = graph_config.n_heads;
            graph_config.local_n_kv_heads = graph_config.n_kv_heads;
            graph_config.qkv_column_parallel = false;
        }

        // =====================================================================
        // Phase 4: Tensor-Parallel Configuration for Column-Parallel FFN
        // =====================================================================
        // When running with multiple MPI ranks, compute local FFN dimension:
        // - Each rank handles d_ff_local = d_ff / world_size
        // - Gate/Up weights: [d_ff, d_model] → [d_ff_local, d_model]
        // - Down weight remains row-parallel: [d_model, d_ff_local] per rank
        // =====================================================================
        if (mpi_ctx && mpi_ctx->world_size() > 1)
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
        // When running with multiple MPI ranks, compute local vocab dimension:
        // - Each rank handles vocab_local = vocab_size / world_size
        // - LM head weight: [vocab_size, d_model] → [vocab_local, d_model]
        // - Output: [seq, vocab_local] per rank, then AllGather to [seq, vocab_size]
        // =====================================================================
        if (mpi_ctx && mpi_ctx->world_size() > 1)
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
                config.batch_size, config.max_seq_len, device_idx))
        {
            LOG_ERROR("[InferenceRunner] Failed to initialize inference state");
            return nullptr;
        }

        // Load weights and configure orchestrator
        if (!configureOrchestratorWeightsImpl(orchestrator.get(), model_ctx))
        {
            LOG_ERROR("[InferenceRunner] Failed to configure orchestrator weights");
            return nullptr;
        }

        LOG_INFO("[InferenceRunner] GraphOrchestrator created successfully");

        // GraphOrchestrator implements IInferenceRunner directly
        return orchestrator;
    }

    static bool configureOrchestratorWeightsImpl(
        GraphOrchestrator *orchestrator,
        std::shared_ptr<ModelContext> model_ctx)
    {
        if (!orchestrator || !model_ctx)
        {
            return false;
        }

        auto weight_mgr = model_ctx->weightManager();
        if (!weight_mgr)
        {
            LOG_ERROR("[InferenceRunner] No weight manager in model context");
            return false;
        }

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

        // Layer weight accessor - capture weight_mgr by value (shared_ptr copy)
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

} // namespace llaminar2
