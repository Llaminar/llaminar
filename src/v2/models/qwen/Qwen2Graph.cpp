/**
 * @file Qwen2Graph.cpp
 * @brief Implementation of Qwen2 compute graph builder
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements Qwen2Graph, the unified graph builder for Qwen2
 * architecture models. It combines the functionality previously split
 * across Qwen2LayerExecutor and Qwen2ModelExecutor.
 */

#include "Qwen2Graph.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../backends/ComputeBackend.h"
#include "../../execution/config/InferenceMode.h"
#include "../../tensors/TensorSlice.h"
#include "../../tensors/Tensors.h"
#include "../../utils/MPIContext.h"
#include "../../utils/MPIStrategy.h"
#include "../../execution/local_execution/graph/GraphBuildUtils.h"
#include "../../execution/config/RuntimeConfig.h"
#include "../../collective/ILocalTPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../collective/ITPContext.h"
#include "../../collective/BackendRouter.h"
#include "../../execution/compute_stages/stages/TPAllreduceStage.h"
#include "../../execution/compute_stages/stages/LocalPPTransferStage.h"
#include "../../execution/compute_stages/stages/QKNormStage.h"
#include "../../execution/compute_stages/stages/FusedResidualNormStage.h"
#include "../../config/PipelineConfig.h"
#include "../../memory/BufferId.h" // Phase 2: contract BufferIds
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace llaminar2
{

    // Import graph_utils for cleaner code
    using namespace graph_utils;

    // =============================================================================
    // Constructors
    // =============================================================================

    Qwen2Graph::Qwen2Graph(std::shared_ptr<ModelContext> model_ctx,
                           std::shared_ptr<MPIContext> mpi_ctx,
                           const GraphConfig &config)
        : config_(config),
          model_ctx_(std::move(model_ctx)),
          mpi_ctx_(std::move(mpi_ctx))
    {
        LOG_DEBUG("[Qwen2Graph] Initializing (full): n_layers=" << config_.n_layers
                                                                << " d_model=" << config_.d_model
                                                                << " vocab_size=" << config_.vocab_size
                                                                << " d_ff=" << config_.d_ff
                                                                << " ffn_column_parallel=" << config_.ffn_column_parallel
                                                                << " n_heads=" << config_.n_heads
                                                                << " n_kv_heads=" << config_.n_kv_heads);

        // Populate per-layer allreduce precision from schema if not already set
        if (config_.tp_allreduce_precision.empty() && config_.n_layers > 0)
        {
            Qwen2SchemaFactory factory;
            auto schema = factory.createSchema();
            config_.populateAllreducePrecision(
                schema.tp_allreduce_default_precision,
                schema.tp_allreduce_fp32_layer_count);
            LOG_DEBUG("[Qwen2Graph] Populated per-layer allreduce precision: "
                      << "fp32_layers=" << schema.tp_allreduce_fp32_layer_count
                      << " default=" << schema.tp_allreduce_default_precision);
        }

        LOG_DEBUG("[Qwen2Graph] Initialized (full) with " << config_.n_layers
                                                          << " layers, precision="
                                                          << activationPrecisionToString(config_.activation_precision));
    }

    Qwen2Graph::Qwen2Graph(const GraphConfig &config,
                           std::shared_ptr<MPIContext> mpi_ctx)
        : config_(config),
          model_ctx_(nullptr),
          mpi_ctx_(std::move(mpi_ctx))
    {
        // This constructor is for layer-level operations only (no model context needed)
        LOG_DEBUG("[Qwen2Graph] Initializing (layer-only): d_model=" << config_.d_model
                                                                     << " d_ff=" << config_.d_ff
                                                                     << " ffn_column_parallel=" << config_.ffn_column_parallel
                                                                     << " n_heads=" << config_.n_heads
                                                                     << " n_kv_heads=" << config_.n_kv_heads
                                                                     << " mpi_ctx=" << (mpi_ctx_ ? "valid" : "nullptr")
                                                                     << " world_size=" << (mpi_ctx_ ? mpi_ctx_->world_size() : -1)
                                                                     << " default_device=" << config_.default_device.to_string());

        // Populate per-layer allreduce precision from schema if not already set
        if (config_.tp_allreduce_precision.empty() && config_.n_layers > 0)
        {
            Qwen2SchemaFactory factory;
            auto schema = factory.createSchema();
            config_.populateAllreducePrecision(
                schema.tp_allreduce_default_precision,
                schema.tp_allreduce_fp32_layer_count);
        }

        LOG_DEBUG("[Qwen2Graph] Initialized (layer-only)");
    }

    // =============================================================================
    // GraphConfig Helper Methods
    // =============================================================================

    bool GraphConfig::hasUnifiedPP() const
    {
        return pipeline_config != nullptr && pipeline_config->hasPP();
    }

    DeviceId GraphConfig::getDeviceForLayer(int layer_idx) const
    {
        if (pipeline_config)
        {
            return pipeline_config->getDeviceForLayer(layer_idx);
        }
        return default_device;
    }

    // =============================================================================
    // IGraphBuilder Interface Implementation
    // =============================================================================

    ComputeGraph Qwen2Graph::buildForwardGraph(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        // Adapt generic ForwardInput to Qwen2ForwardInput
        Qwen2ForwardInput qwen_input;
        qwen_input.token_ids = input.token_ids;
        qwen_input.position_ids = input.position_ids;
        qwen_input.batch_size = input.batch_size;
        qwen_input.seq_len = input.seq_len;
        qwen_input.position_offset = input.position_offset;
        qwen_input.device = input.device;
        qwen_input.kv_cache = input.kv_cache;

        // Adapt generic ForwardOutput to Qwen2ForwardOutput
        Qwen2ForwardOutput qwen_output;
        qwen_output.logits = output.logits;
        qwen_output.hidden = output.hidden;

        // Build the graph
        ComputeGraph graph = buildFullForwardGraph(qwen_input, qwen_output);

        // Copy back output pointers (in case they were set by the builder)
        output.logits = qwen_output.logits;
        output.hidden = qwen_output.hidden;

        return graph;
    }

    ComputeGraph Qwen2Graph::buildLayerGraph(const LayerContext &ctx)
    {
        // Validate layer index against total model layers (supports absolute PP indices)
        int max_layers = config_.total_n_layers > 0 ? config_.total_n_layers : config_.n_layers;
        if (ctx.layer_idx < 0 || ctx.layer_idx >= max_layers)
        {
            LOG_ERROR("[Qwen2Graph::buildLayerGraph] Invalid layer index: " << ctx.layer_idx);
            return ComputeGraph{};
        }

        // Get layer weights
        if (!weights_.get_layer_weights)
        {
            LOG_ERROR("[Qwen2Graph::buildLayerGraph] Layer weight accessor not set");
            return ComputeGraph{};
        }

        LayerWeights layer_weights = weights_.get_layer_weights(ctx.layer_idx);

        // Build attention graph
        ComputeGraph attn_graph = buildAttentionGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.kv_cache, ctx.position_ids, ctx.device,
            ctx.sequence_lengths);

        // Build FFN graph
        ComputeGraph ffn_graph = buildFFNGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.device);

        // Merge: attention -> FFN
        std::string attn_last = "layer" + std::to_string(ctx.layer_idx) + "_attn_residual";
        attn_graph.merge(std::move(ffn_graph), attn_last);

        return attn_graph;
    }

    // =============================================================================
    // Model-Level Graph Building
    // =============================================================================

    ComputeGraph Qwen2Graph::buildFullForwardGraph(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        LOG_DEBUG("[Qwen2Graph] Building full forward graph: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        if (!weights_.embedding_table || !weights_.final_norm || !weights_.lm_head)
        {
            LOG_ERROR("[Qwen2Graph] Weights not set! Call setWeights() first.");
            throw std::runtime_error("Qwen2Graph weights not initialized");
        }

        if (!buffers_.current_hidden || !buffers_.logits)
        {
            LOG_ERROR("[Qwen2Graph] Buffers not set! Call setBuffers() first.");
            throw std::runtime_error("Qwen2Graph buffers not initialized");
        }

        DeviceId device = config_.default_device;
        int total_tokens = input.batch_size * input.seq_len;

        ComputeGraph graph;

        // -------------------------------------------------------------------------
        // Stage 1: Embedding Lookup
        // -------------------------------------------------------------------------
        // For HybridQ16 mode: output to Q16_1 residual buffer (the residual stream)
        // For other modes: output to FP32 current_hidden
        InferenceMode full_pass_inference_mode(config_.activation_precision);
        TensorBase *embed_output = buffers_.layer_buffers.residual &&
                                           full_pass_inference_mode.isHybridQ16()
                                       ? buffers_.layer_buffers.residual
                                       : buffers_.current_hidden;

        EmbeddingStage::Params embed_params;
        embed_params.embed_table = weights_.embedding_table;
        embed_params.token_ids = input.token_ids;
        embed_params.output = embed_output;
        embed_params.num_tokens = total_tokens;
        embed_params.d_model = config_.d_model;
        embed_params.vocab_size = config_.vocab_size;
        embed_params.device_id = config_.default_device;
        embed_params.output_buffer_id = BufferId::HIDDEN_STATE;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(embed_params),
                      device);

        // -------------------------------------------------------------------------
        // Stage 2: Transformer Layers (complete graphs, not placeholders)
        // -------------------------------------------------------------------------
        std::string prev_node = "embedding";

        // Position IDs must be provided externally (or use fallback for backward compat)
        const int *position_ids = input.position_ids;
        std::vector<int> local_position_ids;
        if (!position_ids)
        {
            // Fallback: build position IDs internally (deprecated path)
            local_position_ids = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
            position_ids = local_position_ids.data();
            LOG_DEBUG("[Qwen2Graph] Position IDs built internally (deprecated - prefer external input)");
        }

        // Check if we have layer weight accessor
        if (!weights_.get_layer_weights)
        {
            LOG_ERROR("[Qwen2Graph] Layer weight accessor not set!");
            throw std::runtime_error("Qwen2Graph layer weight accessor not initialized");
        }

        // Build complete graphs for each layer
        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            // Get layer weights
            LayerWeights layer_weights = weights_.get_layer_weights(layer);

            // Build attention graph for this layer
            // Pass sequence_lengths for proper batch-aware attention masking
            ComputeGraph attn_graph = buildAttentionGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, input.kv_cache, position_ids, device,
                input.sequence_lengths);

            // Get the leaf node(s) of attention graph before merging
            auto attn_leaves = attn_graph.getLeafNodes();
            std::string attn_last;
            if (!attn_leaves.empty())
            {
                std::string prefix = "layer" + std::to_string(layer) + "_";
                attn_last = prefix + "attn_residual";
                if (std::find(attn_leaves.begin(), attn_leaves.end(), attn_last) == attn_leaves.end())
                {
                    attn_last = attn_leaves.front();
                }
            }
            else
            {
                // Fallback: use previous node if attention graph is empty
                attn_last = prev_node;
            }

            // Merge attention graph, connecting to previous node
            graph.merge(std::move(attn_graph), prev_node);

            // Build FFN graph for this layer
            ComputeGraph ffn_graph = buildFFNGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, device);

            // Get the leaf node(s) of FFN graph before merging (we need the last node)
            auto ffn_leaves = ffn_graph.getLeafNodes();
            std::string ffn_last;
            if (!ffn_leaves.empty())
            {
                // Prefer ffn_residual if it exists, otherwise use first leaf
                std::string prefix = "layer" + std::to_string(layer) + "_";
                ffn_last = prefix + "ffn_residual";
                if (std::find(ffn_leaves.begin(), ffn_leaves.end(), ffn_last) == ffn_leaves.end())
                {
                    ffn_last = ffn_leaves.front(); // Fall back to first leaf
                }
            }
            else
            {
                // Fallback: use attention residual if FFN graph is empty
                ffn_last = attn_last;
            }

            // Merge FFN graph, connecting to attention residual
            graph.merge(std::move(ffn_graph), attn_last);

            // Use the determined FFN leaf as the prev node for next layer
            prev_node = ffn_last;
        }

        // -------------------------------------------------------------------------
        // Stage 3: Final RMSNorm
        // -------------------------------------------------------------------------
        // IMPORTANT:
        // - In HybridQ16, the live activation stream is Q16_1 in buffers_.layer_buffers.residual.
        // - Our Q16_1 RMSNorm kernel is Q16_1 input -> FP32 output.
        // Therefore final_norm must:
        //   1) read from the correct activation stream (residual for HybridQ16)
        //   2) write out-of-place into the FP32 normalized buffer
        InferenceMode final_norm_mode(config_.activation_precision);
        TensorBase *final_norm_input = (final_norm_mode.isHybridQ16() && buffers_.layer_buffers.residual)
                                           ? buffers_.layer_buffers.residual
                                           : buffers_.current_hidden;

        addFinalNormToGraph(graph, final_norm_input, buffers_.layer_buffers.normalized,
                            prev_node, total_tokens, device);
        prev_node = "final_norm";

        // -------------------------------------------------------------------------
        // Stage 4: LM Head (with optional Column-Parallel + AllGather)
        // -------------------------------------------------------------------------
        // When lm_head_column_parallel is enabled:
        // - LM head weight is sharded by vocab: [vocab_local, d_model]
        // - LM head outputs to buffers_.logits_local: [seq_len, vocab_local]
        // - AllGather collects to buffers_.logits: [seq_len, vocab_size]
        // -------------------------------------------------------------------------
        bool use_column_parallel = config_.lm_head_column_parallel && buffers_.logits_local != nullptr;

        // Determine output buffer and vocab size for LM head stage
        TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
        int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

        LOG_DEBUG("[Qwen2Graph] LM head in buildFullForwardGraph: use_column_parallel="
                  << use_column_parallel << " lm_head_vocab_size=" << lm_head_vocab_size);

        LMHeadStage::Params lm_params;
        // Feed LM head from the final RMSNorm output (FP32).
        lm_params.hidden_states = buffers_.layer_buffers.normalized;
        lm_params.lm_head_weight = weights_.lm_head;
        lm_params.logits = lm_head_output;
        lm_params.seq_len = total_tokens;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = lm_head_vocab_size;
        lm_params.bias_tensor = nullptr; // Qwen2 has no LM head bias
        lm_params.device_id = config_.default_device;
        lm_params.input_buffer_id = BufferId::NORMALIZED;
        lm_params.output_buffer_id = use_column_parallel ? BufferId::LOGITS_LOCAL : BufferId::LOGITS;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device);
        graph.addDependency("lm_head", prev_node);
        prev_node = "lm_head";

        // Phase 5: AllGather stage for column-parallel LM head
        if (use_column_parallel && mpi_ctx_)
        {
            LOG_DEBUG("[Qwen2Graph] Adding lm_head_allgather in buildFullForwardGraph: world_size="
                      << mpi_ctx_->world_size() << " total_tokens=" << total_tokens);

            AllGatherStage::Params allgather_params;
            allgather_params.local_input = buffers_.logits_local;
            allgather_params.full_output = buffers_.logits;
            allgather_params.mpi_ctx = mpi_ctx_.get();
            allgather_params.actual_seq_len = static_cast<size_t>(total_tokens);
            // LM head is not layer-specific; use nullptr for domain (legacy MPI path)
            // Multi-domain TP typically doesn't route LM head to a specific domain
            allgather_params.domain = nullptr;
            allgather_params.input_buffer_id = BufferId::LOGITS_LOCAL;
            allgather_params.output_buffer_id = BufferId::LOGITS;

            graph.addNode("lm_head_allgather",
                          ComputeStageFactory::createAllGather(allgather_params),
                          device);
            graph.addDependency("lm_head_allgather", prev_node);
        }

        // Set output
        output.logits = buffers_.logits;

        LOG_DEBUG("[Qwen2Graph] Built full forward graph with "
                  << graph.size() << " nodes");

        return graph;
    }

    // =========================================================================
    // Partial Forward Graph Building (Pipeline Parallelism)
    // =========================================================================

    ComputeGraph Qwen2Graph::buildPartialForwardGraph(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output,
        int first_layer,
        int last_layer,
        bool has_embedding,
        bool has_lm_head)
    {
        LOG_DEBUG("[Qwen2Graph] Building partial forward graph: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len
                  << ", layers=[" << first_layer << ", " << last_layer << ")"
                  << ", has_embedding=" << has_embedding
                  << ", has_lm_head=" << has_lm_head);

        // Validate layer range against total model layers (not local PP stage count)
        int max_layers = config_.total_n_layers > 0 ? config_.total_n_layers : config_.n_layers;
        if (first_layer < 0 || last_layer > max_layers || first_layer >= last_layer)
        {
            LOG_ERROR("[Qwen2Graph] Invalid layer range: [" << first_layer << ", " << last_layer
                                                            << ") for model with " << max_layers << " total layers");
            throw std::invalid_argument("Invalid layer range for partial forward graph");
        }

        // Validate required weights based on configuration
        if (has_embedding && !weights_.embedding_table)
        {
            LOG_ERROR("[Qwen2Graph] Embedding weights required but not set");
            throw std::runtime_error("Qwen2Graph embedding weights not initialized");
        }
        if (has_lm_head && (!weights_.final_norm || !weights_.lm_head))
        {
            LOG_ERROR("[Qwen2Graph] LM head weights required but not set");
            throw std::runtime_error("Qwen2Graph LM head weights not initialized");
        }
        if (!weights_.get_layer_weights)
        {
            LOG_ERROR("[Qwen2Graph] Layer weight accessor not set");
            throw std::runtime_error("Qwen2Graph layer weight accessor not initialized");
        }

        // Validate buffers
        if (has_embedding && !buffers_.current_hidden && !buffers_.layer_buffers.residual)
        {
            LOG_ERROR("[Qwen2Graph] Hidden buffer required for embedding output");
            throw std::runtime_error("Qwen2Graph hidden buffer not initialized");
        }
        if (has_lm_head && !buffers_.logits)
        {
            LOG_ERROR("[Qwen2Graph] Logits buffer required for LM head output");
            throw std::runtime_error("Qwen2Graph logits buffer not initialized");
        }

        DeviceId device = config_.default_device;
        int total_tokens = input.batch_size * input.seq_len;

        ComputeGraph graph;
        std::string prev_node;

        // -------------------------------------------------------------------------
        // Stage 1: Embedding Lookup (optional - only for first PP stage)
        // -------------------------------------------------------------------------
        if (has_embedding)
        {
            // For HybridQ16 mode: output to Q16_1 residual buffer (the residual stream)
            // For other modes: output to FP32 current_hidden
            InferenceMode full_pass_inference_mode(config_.activation_precision);
            TensorBase *embed_output = buffers_.layer_buffers.residual &&
                                               full_pass_inference_mode.isHybridQ16()
                                           ? buffers_.layer_buffers.residual
                                           : buffers_.current_hidden;

            EmbeddingStage::Params embed_params;
            embed_params.embed_table = weights_.embedding_table;
            embed_params.token_ids = input.token_ids;
            embed_params.output = embed_output;
            embed_params.num_tokens = total_tokens;
            embed_params.d_model = config_.d_model;
            embed_params.vocab_size = config_.vocab_size;
            embed_params.device_id = config_.default_device;
            embed_params.output_buffer_id = BufferId::HIDDEN_STATE;

            graph.addNode("embedding",
                          ComputeStageFactory::createEmbedding(embed_params),
                          device);
            prev_node = "embedding";
        }
        else if (input.external_hidden_state)
        {
            // -----------------------------------------------------------------
            // PP middle/final stage: Use external hidden state as starting point
            // The external_hidden_state tensor contains activations from the
            // previous PP stage. We need to copy it to our working buffer if
            // they're different.
            // -----------------------------------------------------------------
            InferenceMode full_pass_inference_mode(config_.activation_precision);
            TensorBase *working_buffer = buffers_.layer_buffers.residual &&
                                                 full_pass_inference_mode.isHybridQ16()
                                             ? buffers_.layer_buffers.residual
                                             : buffers_.current_hidden;

            // Check if external buffer differs from working buffer
            if (input.external_hidden_state != working_buffer)
            {
                LOG_DEBUG("[Qwen2Graph] PP middle stage: copying external hidden state to working buffer");

                // NOTE: TensorCopyStage is not yet implemented. For now, we perform
                // the copy inline. In Phase 3, we should add TensorCopyStage for
                // proper graph-based memory transfer with device awareness.

                size_t copy_bytes = static_cast<size_t>(total_tokens * config_.d_model);
                if (full_pass_inference_mode.isHybridQ16())
                {
                    // Q16_1: copy the raw Q16_1 blocks
                    // Block size = 32 elements, so num_blocks = copy_elements / 32
                    size_t num_blocks = (copy_bytes + 31) / 32;
                    copy_bytes = num_blocks * sizeof(Q16_1Block);
                }
                else
                {
                    // FP32: copy floats
                    copy_bytes *= sizeof(float);
                }

                // Device-aware copy: handle GPU coherence for PP decode
                // The external_hidden_state may be GPU-authoritative from previous PP stage
                if (config_.default_device.is_gpu())
                {
                    // GPU execution path: handle coherence properly
                    // For PP transfers, the external_hidden_state may be:
                    // 1. BAR-backed (transferred via PCIe BAR) - already accessible from target
                    // 2. Regular GPU tensor - need ensureOnDevice()

                    DeviceId target_device = config_.default_device;
                    const void *src_ptr = nullptr;

                    // Check if source is BAR-backed (from PCIe BAR transfer)
                    if (input.external_hidden_state->isBARBacked())
                    {
                        // BAR-backed tensor: get the appropriate pointer for the target device
                        if (target_device.is_rocm())
                        {
                            // For ROCm target, use the BAR address directly (for D2D copy)
                            // The rocm_data_ptr() might not be available (HIP staging not allocated)
                            // but bar_address() gives us the BAR mmap address usable with hipMemcpy
                            src_ptr = input.external_hidden_state->bar_address();
                            if (!src_ptr)
                            {
                                // Fallback to gpu_data_ptr() which might be the CUDA-accessible pointer
                                src_ptr = input.external_hidden_state->gpu_data_ptr();
                            }
                        }
                        else
                        {
                            // For CUDA target, use the regular gpu_data_ptr()
                            src_ptr = input.external_hidden_state->gpu_data_ptr();
                        }
                        LOG_DEBUG("[Qwen2Graph] PP copy: source is BAR-backed, src_ptr=" << src_ptr);
                    }
                    else
                    {
                        // Regular GPU tensor: ensure it's on the target device
                        if (!input.external_hidden_state->ensureOnDevice(target_device))
                        {
                            LOG_ERROR("[Qwen2Graph] Failed to ensure external_hidden_state on "
                                      << target_device.toString());
                            throw std::runtime_error("Failed to upload external_hidden_state to device");
                        }
                        src_ptr = input.external_hidden_state->gpu_data_ptr();
                    }

                    if (!src_ptr)
                    {
                        LOG_ERROR("[Qwen2Graph] Source pointer null for PP copy");
                        throw std::runtime_error("Source pointer not available for PP copy");
                    }

                    // Allocate destination on device (don't upload stale host data)
                    if (!working_buffer->allocateOnDevice(target_device))
                    {
                        LOG_ERROR("[Qwen2Graph] Failed to allocate working_buffer on "
                                  << target_device.toString());
                        throw std::runtime_error("Failed to allocate working_buffer on device");
                    }

                    void *dst_ptr = working_buffer->gpu_data_ptr();
                    if (!dst_ptr)
                    {
                        LOG_ERROR("[Qwen2Graph] Destination pointer null for PP copy");
                        throw std::runtime_error("Destination pointer not available for PP copy");
                    }

                    // Use GlobalBackendRouter for device-agnostic copy
                    auto *router = GlobalBackendRouter::get();
                    if (!router)
                    {
                        LOG_ERROR("[Qwen2Graph] GlobalBackendRouter not initialized");
                        throw std::runtime_error("Backend router not available for PP copy");
                    }

                    // For BAR-backed source, we need to use the PCIeBAR backend or same-device copy
                    // The source device is effectively the target device since BAR provides cross-device access
                    ICollectiveBackend *backend = router->getBackendForCopy(target_device, target_device);
                    if (!backend)
                    {
                        LOG_ERROR("[Qwen2Graph] No backend for " << target_device.toString());
                        throw std::runtime_error("No backend available for PP copy");
                    }

                    // Perform device copy using backend API
                    if (!backend->copy(dst_ptr, target_device, src_ptr, target_device, copy_bytes))
                    {
                        LOG_ERROR("[Qwen2Graph] Backend copy failed for PP hidden state");
                        throw std::runtime_error("Backend copy failed for PP hidden state");
                    }

                    // Mark destination as device-authoritative (GPU has latest data)
                    working_buffer->mark_device_dirty();

                    LOG_DEBUG("[Qwen2Graph] PP GPU copy: " << copy_bytes << " bytes on "
                                                           << target_device.toString());
                }
                else
                {
                    // CPU execution path: direct memcpy between host buffers
                    std::memcpy(working_buffer->mutable_data(),
                                input.external_hidden_state->data(),
                                copy_bytes);
                    LOG_DEBUG("[Qwen2Graph] PP CPU copy: " << copy_bytes << " bytes");
                }
            }
            else
            {
                LOG_DEBUG("[Qwen2Graph] PP middle stage: external hidden state IS working buffer, no copy needed");
            }
            // No prev_node set - first layer will have no dependencies
        }
        else if (!has_embedding)
        {
            LOG_ERROR("[Qwen2Graph] PP stage without embedding requires external_hidden_state input");
            throw std::runtime_error("PP stage without embedding requires external_hidden_state");
        }

        // -------------------------------------------------------------------------
        // Stage 2: Transformer Layers (subset for this PP stage)
        // -------------------------------------------------------------------------
        // Position IDs must be provided externally (or use fallback for backward compat)
        const int *position_ids = input.position_ids;
        std::vector<int> local_position_ids;
        if (!position_ids)
        {
            // Fallback: build position IDs internally (deprecated path)
            local_position_ids = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
            position_ids = local_position_ids.data();
            LOG_DEBUG("[Qwen2Graph] Position IDs built internally (deprecated - prefer external input)");
        }

        // Build graphs for assigned layer range [first_layer, last_layer)
        for (int layer = first_layer; layer < last_layer; ++layer)
        {
            // Get layer weights
            LayerWeights layer_weights = weights_.get_layer_weights(layer);

            // Build attention graph for this layer
            ComputeGraph attn_graph = buildAttentionGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, input.kv_cache, position_ids, device,
                input.sequence_lengths);

            // Get the leaf node(s) of attention graph before merging
            auto attn_leaves = attn_graph.getLeafNodes();
            std::string attn_last;
            if (!attn_leaves.empty())
            {
                std::string prefix = "layer" + std::to_string(layer) + "_";
                attn_last = prefix + "attn_residual";
                if (std::find(attn_leaves.begin(), attn_leaves.end(), attn_last) == attn_leaves.end())
                {
                    attn_last = attn_leaves.front();
                }
            }
            else
            {
                // Fallback: use previous node if attention graph is empty
                attn_last = prev_node;
            }

            // Merge attention graph, connecting to previous node
            if (!prev_node.empty())
            {
                graph.merge(std::move(attn_graph), prev_node);
            }
            else
            {
                // First layer without embedding: merge directly
                // The attention graph root nodes have no dependencies initially
                for (const auto &root : attn_graph.getRootNodes())
                {
                    // Move all nodes from attn_graph to main graph
                }
                graph.merge(std::move(attn_graph), "");
            }

            // Build FFN graph for this layer
            ComputeGraph ffn_graph = buildFFNGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, device);

            // Get the leaf node(s) of FFN graph before merging
            auto ffn_leaves = ffn_graph.getLeafNodes();
            std::string ffn_last;
            if (!ffn_leaves.empty())
            {
                std::string prefix = "layer" + std::to_string(layer) + "_";
                ffn_last = prefix + "ffn_residual";
                if (std::find(ffn_leaves.begin(), ffn_leaves.end(), ffn_last) == ffn_leaves.end())
                {
                    ffn_last = ffn_leaves.front();
                }
            }
            else
            {
                // Fallback: use attention residual if FFN graph is empty
                ffn_last = attn_last;
            }

            // Merge FFN graph, connecting to attention residual
            graph.merge(std::move(ffn_graph), attn_last);

            // Use the determined FFN leaf as the prev node for next layer
            prev_node = ffn_last;
        }

        // -------------------------------------------------------------------------
        // Stage 3: Final RMSNorm + LM Head (optional - only for final PP stage)
        // -------------------------------------------------------------------------
        if (has_lm_head)
        {
            // Final RMSNorm
            InferenceMode final_norm_mode(config_.activation_precision);
            TensorBase *final_norm_input = (final_norm_mode.isHybridQ16() && buffers_.layer_buffers.residual)
                                               ? buffers_.layer_buffers.residual
                                               : buffers_.current_hidden;

            addFinalNormToGraph(graph, final_norm_input, buffers_.layer_buffers.normalized,
                                prev_node, total_tokens, device);
            prev_node = "final_norm";

            // LM Head (with optional Column-Parallel + AllGather)
            bool use_column_parallel = config_.lm_head_column_parallel && buffers_.logits_local != nullptr;

            TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
            int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

            LOG_DEBUG("[Qwen2Graph] LM head in buildPartialForwardGraph: use_column_parallel="
                      << use_column_parallel << " lm_head_vocab_size=" << lm_head_vocab_size);

            LMHeadStage::Params lm_params;
            lm_params.hidden_states = buffers_.layer_buffers.normalized;
            lm_params.lm_head_weight = weights_.lm_head;
            lm_params.logits = lm_head_output;
            lm_params.seq_len = total_tokens;
            lm_params.d_model = config_.d_model;
            lm_params.vocab_size = lm_head_vocab_size;
            lm_params.bias_tensor = nullptr;
            lm_params.device_id = config_.default_device;
            lm_params.input_buffer_id = BufferId::NORMALIZED;
            lm_params.output_buffer_id = use_column_parallel ? BufferId::LOGITS_LOCAL : BufferId::LOGITS;

            graph.addNode("lm_head",
                          ComputeStageFactory::createLMHead(lm_params),
                          device);
            graph.addDependency("lm_head", prev_node);
            prev_node = "lm_head";

            // AllGather stage for column-parallel LM head
            if (use_column_parallel && mpi_ctx_)
            {
                LOG_DEBUG("[Qwen2Graph] Adding lm_head_allgather in buildPartialForwardGraph: world_size="
                          << mpi_ctx_->world_size() << " total_tokens=" << total_tokens);

                AllGatherStage::Params allgather_params;
                allgather_params.local_input = buffers_.logits_local;
                allgather_params.full_output = buffers_.logits;
                allgather_params.mpi_ctx = mpi_ctx_.get();
                allgather_params.actual_seq_len = static_cast<size_t>(total_tokens);
                allgather_params.domain = nullptr;
                allgather_params.input_buffer_id = BufferId::LOGITS_LOCAL;
                allgather_params.output_buffer_id = BufferId::LOGITS;

                graph.addNode("lm_head_allgather",
                              ComputeStageFactory::createAllGather(allgather_params),
                              device);
                graph.addDependency("lm_head_allgather", prev_node);
            }

            // Set output logits
            output.logits = buffers_.logits;
        }
        else
        {
            // Non-final PP stage: output hidden states directly
            // The residual buffer contains the hidden states after the last layer
            InferenceMode mode(config_.activation_precision);
            output.hidden = (mode.isHybridQ16() && buffers_.layer_buffers.residual)
                                ? buffers_.layer_buffers.residual
                                : buffers_.current_hidden;
            output.logits = nullptr;
        }

        LOG_DEBUG("[Qwen2Graph] Built partial forward graph with "
                  << graph.size() << " nodes for layers [" << first_layer << ", " << last_layer << ")");

        return graph;
    }

    // =========================================================================
    // Unified Pipeline Graph Building (PP + TP Composition)
    // =========================================================================

    ComputeGraph Qwen2Graph::buildUnifiedPipelineGraph(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        // =====================================================================
        // 1. Validate prerequisites
        // =====================================================================
        if (!config_.pipeline_config)
        {
            LOG_ERROR("[Qwen2Graph] buildUnifiedPipelineGraph called without pipeline_config");
            throw std::runtime_error("pipeline_config required for unified PP graph");
        }

        std::string validation_error;
        if (!config_.pipeline_config->validate(&validation_error))
        {
            LOG_ERROR("[Qwen2Graph] Invalid pipeline_config: " << validation_error);
            throw std::runtime_error("Invalid pipeline_config: " + validation_error);
        }

        LOG_DEBUG("[Qwen2Graph] Building unified pipeline graph: "
                  << config_.pipeline_config->numStages() << " PP stages, "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        // Validate weights
        if (!weights_.get_layer_weights)
        {
            LOG_ERROR("[Qwen2Graph] Layer weight accessor not set");
            throw std::runtime_error("Qwen2Graph layer weight accessor not initialized");
        }

        // =====================================================================
        // 2. Prepare execution context
        // =====================================================================
        ComputeGraph graph;
        std::string prev_node;
        int total_tokens = input.batch_size * input.seq_len;

        // Position IDs
        const int *position_ids = input.position_ids;
        std::vector<int> local_position_ids;
        if (!position_ids)
        {
            local_position_ids = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
            position_ids = local_position_ids.data();
            LOG_DEBUG("[Qwen2Graph] Position IDs built internally for unified PP graph");
        }

        // Inference mode for buffer selection
        InferenceMode inference_mode(config_.activation_precision);

        // =====================================================================
        // 3. Iterate over PP stages
        // =====================================================================
        for (const auto &pp_stage : config_.pipeline_config->pp_stages)
        {
            LOG_DEBUG("[Qwen2Graph] Building PP stage " << pp_stage.stage_id
                                                        << " (" << pp_stage.domain_name << "): layers ["
                                                        << pp_stage.first_layer << ", " << pp_stage.last_layer << ")"
                                                        << " has_embedding=" << pp_stage.has_embedding
                                                        << " has_lm_head=" << pp_stage.has_lm_head);

            // -----------------------------------------------------------------
            // 3a. Get domain config for this stage
            // -----------------------------------------------------------------
            const TPDomainConfig *domain = config_.pipeline_config->getDomainForStage(pp_stage.stage_id);
            if (!domain)
            {
                LOG_ERROR("[Qwen2Graph] Domain not found for stage " << pp_stage.stage_id
                                                                     << " (domain_name=" << pp_stage.domain_name << ")");
                throw std::runtime_error("Domain not found for PP stage " +
                                         std::to_string(pp_stage.stage_id));
            }

            // Get device and TP context for this stage
            DeviceId stage_device = domain->primaryDevice();
            ILocalTPContext *stage_tp_ctx = nullptr;
            auto tp_it = config_.domain_tp_contexts.find(pp_stage.domain_name);
            if (tp_it != config_.domain_tp_contexts.end())
            {
                stage_tp_ctx = tp_it->second;
            }

            LOG_DEBUG("[Qwen2Graph] Stage " << pp_stage.stage_id
                                            << " device=" << stage_device.to_string()
                                            << " has_tp_ctx=" << (stage_tp_ctx != nullptr));

            // -----------------------------------------------------------------
            // 3b. Build embedding if this is the first stage
            // -----------------------------------------------------------------
            if (pp_stage.has_embedding)
            {
                if (!weights_.embedding_table)
                {
                    LOG_ERROR("[Qwen2Graph] Embedding weights required but not set");
                    throw std::runtime_error("Embedding weights not initialized for unified PP graph");
                }

                // For HybridQ16 mode: output to Q16_1 residual buffer
                // For other modes: output to FP32 current_hidden
                TensorBase *embed_output = buffers_.layer_buffers.residual &&
                                                   inference_mode.isHybridQ16()
                                               ? buffers_.layer_buffers.residual
                                               : buffers_.current_hidden;

                EmbeddingStage::Params embed_params;
                embed_params.embed_table = weights_.embedding_table;
                embed_params.token_ids = input.token_ids;
                embed_params.output = embed_output;
                embed_params.num_tokens = total_tokens;
                embed_params.d_model = config_.d_model;
                embed_params.vocab_size = config_.vocab_size;
                embed_params.device_id = stage_device;

                graph.addNode("embedding",
                              ComputeStageFactory::createEmbedding(embed_params),
                              stage_device);
                prev_node = "embedding";

                LOG_DEBUG("[Qwen2Graph] Added embedding stage on device " << stage_device.to_string());
            }

            // -----------------------------------------------------------------
            // 3c. Build layers for this PP stage
            // -----------------------------------------------------------------
            // Temporarily override config_.default_device for layer building
            // This ensures all stages in the layer use the correct device
            DeviceId original_device = config_.default_device;
            ILocalTPContext *original_tp_ctx = config_.local_tp_ctx;

            config_.default_device = stage_device;
            if (stage_tp_ctx)
            {
                config_.local_tp_ctx = stage_tp_ctx;
            }

            for (int layer = pp_stage.first_layer; layer < pp_stage.last_layer; ++layer)
            {
                // Get layer weights
                LayerWeights layer_weights = weights_.get_layer_weights(layer);

                // Get KV cache for this stage's device (PP-aware)
                IKVCache *layer_kv_cache = input.getKVCacheForDevice(stage_device);

                // Build attention graph with stage_device
                ComputeGraph attn_graph = buildAttentionGraph(
                    layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                    input.batch_size, layer_kv_cache, position_ids, stage_device,
                    input.sequence_lengths);

                // Get the leaf node of attention graph
                auto attn_leaves = attn_graph.getLeafNodes();
                std::string attn_last;
                if (!attn_leaves.empty())
                {
                    std::string prefix = "layer" + std::to_string(layer) + "_";
                    attn_last = prefix + "attn_residual";
                    if (std::find(attn_leaves.begin(), attn_leaves.end(), attn_last) == attn_leaves.end())
                    {
                        attn_last = attn_leaves.front();
                    }
                }
                else
                {
                    attn_last = prev_node;
                }

                // Merge attention graph
                if (!prev_node.empty())
                {
                    graph.merge(std::move(attn_graph), prev_node);
                }
                else
                {
                    graph.merge(std::move(attn_graph), "");
                }

                // Build FFN graph
                ComputeGraph ffn_graph = buildFFNGraph(
                    layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                    input.batch_size, stage_device);

                // Get the leaf node of FFN graph
                auto ffn_leaves = ffn_graph.getLeafNodes();
                std::string ffn_last;
                if (!ffn_leaves.empty())
                {
                    std::string prefix = "layer" + std::to_string(layer) + "_";
                    ffn_last = prefix + "ffn_residual";
                    if (std::find(ffn_leaves.begin(), ffn_leaves.end(), ffn_last) == ffn_leaves.end())
                    {
                        ffn_last = ffn_leaves.front();
                    }
                }
                else
                {
                    ffn_last = attn_last;
                }

                // Merge FFN graph
                graph.merge(std::move(ffn_graph), attn_last);

                prev_node = ffn_last;
            }

            // Restore original config
            config_.default_device = original_device;
            config_.local_tp_ctx = original_tp_ctx;

            // -----------------------------------------------------------------
            // 3d. Insert PP transfer stage if not the last PP stage
            // -----------------------------------------------------------------
            int next_stage_id = pp_stage.stage_id + 1;
            if (next_stage_id < config_.pipeline_config->numStages())
            {
                // Get PP context for this transfer
                auto pp_key = std::make_pair(pp_stage.stage_id, next_stage_id);
                ILocalPPContext *pp_ctx = nullptr;
                auto pp_it = config_.pp_contexts.find(pp_key);
                if (pp_it != config_.pp_contexts.end())
                {
                    pp_ctx = pp_it->second;
                }

                if (!pp_ctx)
                {
                    LOG_ERROR("[Qwen2Graph] PP context not found for transfer "
                              << pp_stage.stage_id << " -> " << next_stage_id);
                    throw std::runtime_error("PP context not found for stage transfer " +
                                             std::to_string(pp_stage.stage_id) + " -> " +
                                             std::to_string(next_stage_id));
                }

                // Get the hidden state buffer for transfer
                TensorBase *hidden_state = (inference_mode.isHybridQ16() && buffers_.layer_buffers.residual)
                                               ? buffers_.layer_buffers.residual
                                               : buffers_.current_hidden;

                // Create LocalPPTransferStage
                LocalPPTransferStage::Params transfer_params;
                transfer_params.pp_ctx = pp_ctx;
                transfer_params.tensor = hidden_state;
                transfer_params.stage_from = pp_stage.stage_id;
                transfer_params.stage_to = next_stage_id;
                transfer_params.stage_name = "pp_transfer_" +
                                             std::to_string(pp_stage.stage_id) + "_to_" +
                                             std::to_string(next_stage_id);
                transfer_params.device_id = stage_device;
                transfer_params.mpi_ctx = mpi_ctx_.get();

                std::string transfer_name = transfer_params.stage_name;

                graph.addNode(transfer_name,
                              std::make_unique<LocalPPTransferStage>(transfer_params),
                              stage_device); // Transfer stage starts on source device
                graph.addDependency(transfer_name, prev_node);

                prev_node = transfer_name;

                LOG_DEBUG("[Qwen2Graph] Added PP transfer: " << transfer_name
                                                             << " (source device=" << stage_device.to_string() << ")");
            }

            // -----------------------------------------------------------------
            // 3e. Build LM head if this is the last stage
            // -----------------------------------------------------------------
            if (pp_stage.has_lm_head)
            {
                if (!weights_.final_norm || !weights_.lm_head)
                {
                    LOG_ERROR("[Qwen2Graph] LM head weights required but not set");
                    throw std::runtime_error("LM head weights not initialized for unified PP graph");
                }

                // Final RMSNorm
                TensorBase *final_norm_input = (inference_mode.isHybridQ16() && buffers_.layer_buffers.residual)
                                                   ? buffers_.layer_buffers.residual
                                                   : buffers_.current_hidden;

                addFinalNormToGraph(graph, final_norm_input, buffers_.layer_buffers.normalized,
                                    prev_node, total_tokens, stage_device);
                prev_node = "final_norm";

                // LM Head
                bool use_column_parallel = config_.lm_head_column_parallel && buffers_.logits_local != nullptr;
                TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
                int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

                LOG_DEBUG("[Qwen2Graph] LM head in unified PP: use_column_parallel="
                          << use_column_parallel << " vocab_size=" << lm_head_vocab_size);

                LMHeadStage::Params lm_params;
                lm_params.hidden_states = buffers_.layer_buffers.normalized;
                lm_params.lm_head_weight = weights_.lm_head;
                lm_params.logits = lm_head_output;
                lm_params.seq_len = total_tokens;
                lm_params.d_model = config_.d_model;
                lm_params.vocab_size = lm_head_vocab_size;
                lm_params.bias_tensor = nullptr;
                lm_params.device_id = stage_device;

                graph.addNode("lm_head",
                              ComputeStageFactory::createLMHead(lm_params),
                              stage_device);
                graph.addDependency("lm_head", prev_node);
                prev_node = "lm_head";

                // AllGather stage for column-parallel LM head
                if (use_column_parallel && mpi_ctx_)
                {
                    LOG_DEBUG("[Qwen2Graph] Adding lm_head_allgather in unified PP: world_size="
                              << mpi_ctx_->world_size());

                    AllGatherStage::Params allgather_params;
                    allgather_params.local_input = buffers_.logits_local;
                    allgather_params.full_output = buffers_.logits;
                    allgather_params.mpi_ctx = mpi_ctx_.get();
                    allgather_params.actual_seq_len = static_cast<size_t>(total_tokens);
                    allgather_params.domain = nullptr;
                    allgather_params.input_buffer_id = BufferId::LOGITS_LOCAL;
                    allgather_params.output_buffer_id = BufferId::LOGITS;

                    graph.addNode("lm_head_allgather",
                                  ComputeStageFactory::createAllGather(allgather_params),
                                  stage_device);
                    graph.addDependency("lm_head_allgather", prev_node);
                }

                output.logits = buffers_.logits;

                LOG_DEBUG("[Qwen2Graph] Added LM head stage on device " << stage_device.to_string());
            }
        }

        LOG_DEBUG("[Qwen2Graph] Built unified pipeline graph with " << graph.size()
                                                                    << " nodes across " << config_.pipeline_config->numStages() << " PP stages");

        return graph;
    }

    // =========================================================================
    // Schema-Based Graph Building (Phase 4d)
    // =========================================================================

    ComputeGraph Qwen2Graph::buildForwardGraphFromSchema(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        LOG_DEBUG("[Qwen2Graph] Building forward graph from schema: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        // =====================================================================
        // Step 1: Create the declarative schema
        // =====================================================================
        Qwen2SchemaFactory factory;
        GraphSchema schema = factory.createSchema();

        // =====================================================================
        // Step 2: Build runtime configuration
        // =====================================================================
        GraphResolverConfig config;

        // MPI / Tensor Parallelism
        config.world_size = mpi_ctx_ ? mpi_ctx_->world_size() : 1;
        config.rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        config.mpi_ctx = mpi_ctx_.get();

        // Execution dimensions
        config.batch_size = input.batch_size;
        config.seq_len = input.seq_len;
        config.default_device = config_.default_device;

        // Model dimensions
        config.n_layers = config_.n_layers;
        config.d_model = config_.d_model;
        config.n_heads = config_.n_heads;
        config.n_kv_heads = config_.n_kv_heads;
        config.head_dim = config_.head_dim;
        config.d_ff = config_.d_ff;
        config.vocab_size = config_.vocab_size;

        // TP-adjusted dimensions
        config.local_n_heads = config_.qkv_column_parallel ? config_.local_n_heads : config_.n_heads;
        config.local_n_kv_heads = config_.qkv_column_parallel ? config_.local_n_kv_heads : config_.n_kv_heads;
        config.local_d_ff = config_.ffn_column_parallel ? config_.d_ff_local : config_.d_ff;
        config.local_vocab = config_.lm_head_column_parallel ? config_.vocab_local : config_.vocab_size;

        // Model parameters
        config.rms_norm_eps = config_.rms_norm_eps;
        config.rope_theta = config_.rope_theta;

        // Phase 5.4: VNNI-safe Q16 KV cache scale
        config.kv_cache_scale = config_.kv_cache_scale;

        // KV cache state
        config.has_kv_cache = (input.kv_cache != nullptr);
        if (config.has_kv_cache)
        {
            config.cached_tokens = input.kv_cache->get_cached_tokens(0, 0);
        }

        // Execution policy from debugEnv
        config.exec_policy = ExecutionPolicyFlags::fromDebugEnv();

        // =====================================================================
        // Step 3: Build TensorContext for name → tensor resolution
        // =====================================================================
        TensorContext tensors;

        // Activation buffers
        tensors.buffers["hidden"] = buffers_.current_hidden;
        tensors.buffers["normalized"] = buffers_.layer_buffers.normalized;
        tensors.buffers["Q"] = buffers_.layer_buffers.Q;
        tensors.buffers["K"] = buffers_.layer_buffers.K;
        tensors.buffers["V"] = buffers_.layer_buffers.V;
        tensors.buffers["attn_output"] = buffers_.layer_buffers.attn_output;
        tensors.buffers["attn_proj"] = buffers_.layer_buffers.attn_proj;
        tensors.buffers["gate"] = buffers_.layer_buffers.gate;
        tensors.buffers["up"] = buffers_.layer_buffers.up;
        tensors.buffers["logits"] = buffers_.logits;
        tensors.buffers["logits_local"] = buffers_.logits_local;

        // Model-level weights
        tensors.model_weights["embedding_table"] = weights_.embedding_table;
        tensors.model_weights["final_norm"] = weights_.final_norm;
        tensors.model_weights["lm_head"] = weights_.lm_head;

        // Layer weight accessor
        tensors.get_layer_weight = [this](int layer_idx, const std::string &name) -> TensorBase *
        {
            if (!weights_.get_layer_weights)
                return nullptr;

            LayerWeights layer = weights_.get_layer_weights(layer_idx);

            if (name == "wq")
                return layer.wq;
            if (name == "wk")
                return layer.wk;
            if (name == "wv")
                return layer.wv;
            if (name == "wo")
                return layer.wo;
            if (name == "attn_norm")
                return layer.attn_norm;
            if (name == "q_bias")
                return layer.q_bias;
            if (name == "k_bias")
                return layer.k_bias;
            if (name == "v_bias")
                return layer.v_bias;
            if (name == "gate_proj")
                return layer.gate_proj;
            if (name == "up_proj")
                return layer.up_proj;
            if (name == "down_proj")
                return layer.down_proj;
            if (name == "ffn_norm")
                return layer.ffn_norm;

            LOG_WARN("[TensorContext] Unknown layer weight: " << name);
            return nullptr;
        };

        // =====================================================================
        // Step 4: Resolve schema to concrete graph spec
        // =====================================================================
        GraphResolver resolver;
        ResolvedGraphSpec resolved = resolver.resolve(schema, config, tensors);

        LOG_DEBUG("[Qwen2Graph] Schema resolved: " << resolved.stages.size() << " stages"
                                                   << " (emitted=" << resolved.stats.stages_emitted
                                                   << ", skipped=" << resolved.stats.stages_skipped
                                                   << ", allreduce=" << resolved.stats.allreduce_inserted
                                                   << ", allgather=" << resolved.stats.allgather_inserted << ")");

        // =====================================================================
        // Step 5: Build the ComputeGraph
        // =====================================================================
        ComputeGraph graph = GraphBuilder::build(resolved);

        // Set output
        output.logits = buffers_.logits;

        LOG_DEBUG("[Qwen2Graph] Built schema-based forward graph with "
                  << graph.size() << " nodes");

        return graph;
    }

    ComputeGraph Qwen2Graph::buildEmbeddingGraph(
        const Qwen2ForwardInput &input,
        TensorBase *output_hidden)
    {
        LOG_DEBUG("[Qwen2Graph] Building embedding graph for "
                  << (input.batch_size * input.seq_len) << " tokens");

        ComputeGraph graph;

        EmbeddingStage::Params params;
        params.embed_table = weights_.embedding_table;
        params.token_ids = input.token_ids;
        params.output = output_hidden;
        params.num_tokens = input.batch_size * input.seq_len;
        params.d_model = config_.d_model;
        params.vocab_size = config_.vocab_size;
        params.device_id = config_.default_device;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(params),
                      config_.default_device);

        return graph;
    }

    ComputeGraph Qwen2Graph::buildTransformerLayersGraph(
        TensorBase *input_hidden,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        LOG_DEBUG("[Qwen2Graph] Building transformer layers graph: "
                  << config_.n_layers << " layers");

        ComputeGraph graph;
        std::string prev_node;

        // Build placeholder nodes for layer sequencing
        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            std::string layer_name = "layer_" + std::to_string(layer);

            // Add node with nullptr stage - placeholder for sequencing
            graph.addNode(layer_name, nullptr, device);

            if (!prev_node.empty())
            {
                graph.addDependency(layer_name, prev_node);
            }

            prev_node = layer_name;
        }

        return graph;
    }

    ComputeGraph Qwen2Graph::buildLayerGraph(
        int layer_idx,
        TensorBase *input_hidden,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        LOG_DEBUG("[Qwen2Graph] Building layer " << layer_idx << " graph");

        ComputeGraph graph;
        graph.addNode("layer_" + std::to_string(layer_idx), nullptr, device);

        return graph;
    }

    ComputeGraph Qwen2Graph::buildLMHeadGraph(
        TensorBase *hidden_states,
        TensorBase *output_logits,
        int total_tokens,
        DeviceId device,
        TensorBase *logits_local)
    {
        LOG_DEBUG("[Qwen2Graph] Building LM head graph for " << total_tokens << " tokens"
                                                             << " lm_head_column_parallel=" << config_.lm_head_column_parallel
                                                             << " vocab_local=" << config_.vocab_local);

        ComputeGraph graph;

        // Final RMSNorm
        RMSNormStage::Params norm_params;
        norm_params.input = hidden_states;
        norm_params.output = hidden_states; // In-place norm
        norm_params.gamma = weights_.final_norm;
        norm_params.eps = config_.rms_norm_eps;
        norm_params.seq_len = total_tokens;
        norm_params.device_id = device;

        graph.addNode("final_norm",
                      ComputeStageFactory::createRMSNorm(norm_params),
                      device);

        // =================================================================
        // LM Head Projection - Column-Parallel or Full
        // =================================================================
        // When lm_head_column_parallel is enabled:
        // - LM head weight is sharded by vocab: [vocab_local, d_model]
        // - LM head outputs to logits_local: [seq_len, vocab_local]
        // - AllGather collects to output_logits: [seq_len, vocab_size]
        // =================================================================

        bool use_column_parallel = config_.lm_head_column_parallel && logits_local != nullptr;

        // Determine output buffer and vocab size for LM head stage
        TensorBase *lm_head_output = use_column_parallel ? logits_local : output_logits;
        int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

        LOG_DEBUG("[Qwen2Graph] LM head: use_column_parallel=" << use_column_parallel
                                                               << " lm_head_vocab_size=" << lm_head_vocab_size
                                                               << " lm_head_output=" << lm_head_output);

        // LM Head projection
        LMHeadStage::Params lm_params;
        lm_params.hidden_states = hidden_states;
        lm_params.lm_head_weight = weights_.lm_head;
        lm_params.logits = lm_head_output;
        lm_params.seq_len = total_tokens;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = lm_head_vocab_size;
        lm_params.bias_tensor = nullptr;
        lm_params.device_id = device;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device);
        graph.addDependency("lm_head", "final_norm");

        // =================================================================
        // AllGather stage for column-parallel LM head
        // =================================================================
        if (use_column_parallel && mpi_ctx_)
        {
            LOG_DEBUG("[Qwen2Graph] Adding lm_head_allgather: world_size=" << mpi_ctx_->world_size()
                                                                           << " total_tokens=" << total_tokens);

            AllGatherStage::Params allgather_params;
            allgather_params.local_input = logits_local;
            allgather_params.full_output = output_logits;
            allgather_params.mpi_ctx = mpi_ctx_.get();
            allgather_params.actual_seq_len = static_cast<size_t>(total_tokens);
            // LM head is not layer-specific; use nullptr for domain (legacy MPI path)
            allgather_params.domain = nullptr;
            allgather_params.input_buffer_id = BufferId::LOGITS_LOCAL;
            allgather_params.output_buffer_id = BufferId::LOGITS;

            graph.addNode("lm_head_allgather",
                          ComputeStageFactory::createAllGather(allgather_params),
                          device);
            graph.addDependency("lm_head_allgather", "lm_head");
        }

        return graph;
    }

    // =============================================================================
    // Layer-Level Graph Building
    // =============================================================================

    ComputeGraph Qwen2Graph::buildAttentionGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device,
        const std::vector<int> *sequence_lengths)
    {
        ComputeGraph graph;
        const auto &env = debugEnv();
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        // Total tokens for GEMM m dimension = batch_size * seq_len
        int total_tokens = batch_size * seq_len;

        LOG_DEBUG("[buildAttentionGraph] layer_idx=" << layer_idx << " seq_len=" << seq_len
                                                     << " batch_size=" << batch_size
                                                     << " total_tokens=" << total_tokens
                                                     << " layer.wq=" << static_cast<const void *>(layer.wq)
                                                     << " layer.wo=" << layer.wo << " world_size="
                                                     << (mpi_ctx_ ? mpi_ctx_->world_size() : 1)
                                                     << " device=" << device.to_string()
                                                     << " sequence_lengths=" << (sequence_lengths ? "valid" : "nullptr")
                                                     << (sequence_lengths ? " size=" + std::to_string(sequence_lengths->size()) : ""));

        // Determine backend type for stage creation from DeviceId
        ComputeBackendType backend = ComputeBackendType::CPU;
        if (device.is_cuda())
        {
            backend = ComputeBackendType::GPU_CUDA;
        }
        else if (device.is_rocm())
        {
            backend = ComputeBackendType::GPU_ROCM;
        }

        // Stage 1: Pre-attention RMSNorm
        // Non-first layers: Fused ResidualAdd + RMSNorm (previous layer's ffn output + this norm)
        // First layer: standalone RMSNorm (no prior residual to fuse)
        if (env.execution.exec_rmsnorm)
        {
            InferenceMode inference_mode(config_.activation_precision);

            if (!inference_mode.isHybridQ16() && layer_idx > config_.pp_layer_offset)
            {
                // Non-first layers: Fused ResidualAdd + RMSNorm
                // Combines previous layer's ffn_residual with this layer's attn_norm
                // On GPU: single fused kernel. On CPU: sequential ResidualAdd + RMSNorm via KernelFactory.
                FusedResidualNormStage::Params fused_params;
                fused_params.device_id = device;
                fused_params.input = buffers.attn_proj;         // Previous layer's down_proj output
                fused_params.residual = buffers.current_hidden; // Hidden state (in-place update)
                fused_params.gamma = layer.attn_norm;
                fused_params.norm_output = buffers.normalized;
                fused_params.eps = config_.rms_norm_eps;
                fused_params.seq_len = total_tokens;
                fused_params.hidden_dim = config_.d_model;
                fused_params.input_buffer_id = BufferId::ATTN_PROJ;
                fused_params.residual_buffer_id = BufferId::HIDDEN_STATE;
                fused_params.norm_output_buffer_id = BufferId::NORMALIZED;

                graph.addNode(prefix + "attn_norm",
                              ComputeStageFactory::createFusedResidualNorm(fused_params),
                              device);
            }
            else
            {
                // First layer: standalone RMSNorm (no prior residual to fuse)
                RMSNormStage::Params attn_norm_params;
                attn_norm_params.input = buffers.current_hidden;
                attn_norm_params.output = buffers.normalized;
                attn_norm_params.gamma = layer.attn_norm;
                attn_norm_params.eps = config_.rms_norm_eps;
                attn_norm_params.seq_len = total_tokens;
                attn_norm_params.device_id = device;
                attn_norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
                attn_norm_params.output_buffer_id = BufferId::NORMALIZED;

                graph.addNode(prefix + "attn_norm",
                              ComputeStageFactory::createRMSNorm(attn_norm_params),
                              device);
            }
        }

        // Stage 2: Q/K/V projections using FusedQKVGEMMStage
        if (env.execution.exec_gemm && layer.wq && layer.wk && layer.wv)
        {
            LOG_DEBUG("[Qwen2Graph] Using FusedQKVGEMMStage");

            int k = config_.d_model;
            int q_n = static_cast<int>(layer.wq->shape()[0]);
            int k_n = static_cast<int>(layer.wk->shape()[0]);
            int v_n = static_cast<int>(layer.wv->shape()[0]);

            LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " QKV dims: q_n=" << q_n
                                            << " k_n=" << k_n << " v_n=" << v_n
                                            << " wq_shape=[" << layer.wq->shape()[0] << "," << layer.wq->shape()[1] << "]"
                                            << " wk_shape=[" << layer.wk->shape()[0] << "," << layer.wk->shape()[1] << "]");

            FusedQKVGEMMStage::Params qkv_params;
            qkv_params.input = buffers.normalized;
            qkv_params.m = total_tokens; // Use total_tokens = batch_size * seq_len
            qkv_params.k = k;
            qkv_params.wq = layer.wq;
            qkv_params.output_q = buffers.Q;
            qkv_params.n_q = q_n;
            qkv_params.bias_q = layer.q_bias; // TensorBase* for tensor-aware GPU path
            qkv_params.wk = layer.wk;
            qkv_params.output_k = buffers.K;
            qkv_params.n_k = k_n;
            qkv_params.bias_k = layer.k_bias; // TensorBase* for tensor-aware GPU path
            qkv_params.wv = layer.wv;
            qkv_params.output_v = buffers.V;
            qkv_params.n_v = v_n;
            qkv_params.bias_v = layer.v_bias; // TensorBase* for tensor-aware GPU path
            qkv_params.device_id = device;
            qkv_params.input_buffer_id = BufferId::NORMALIZED;
            qkv_params.output_q_buffer_id = BufferId::Q_PROJ;
            qkv_params.output_k_buffer_id = BufferId::K_PROJ;
            qkv_params.output_v_buffer_id = BufferId::V_PROJ;
            LOG_DEBUG("[Qwen2Graph] Creating FusedQKVGEMM with device_id=" << device.to_string());

            graph.addNode(prefix + "qkv_proj",
                          ComputeStageFactory::createFusedQKVGEMM(qkv_params),
                          device);

            if (env.execution.exec_rmsnorm)
            {
                graph.addDependency(prefix + "qkv_proj", prefix + "attn_norm");
            }
        }

        // =================================================================
        // Resolve local head counts for tensor-parallel attention
        // =================================================================
        // When qkv_column_parallel is enabled, each rank processes a subset of heads.
        // The weight shapes from QKV projection already reflect local dimensions,
        // and RoPE/Attention stages must use matching local head counts.
        // =================================================================
        int local_n_heads = config_.qkv_column_parallel
                                ? config_.local_n_heads
                                : config_.n_heads;
        int local_n_kv_heads = config_.qkv_column_parallel
                                   ? config_.local_n_kv_heads
                                   : config_.n_kv_heads;

        // Validate local head counts (safety check)
        if (local_n_heads <= 0)
            local_n_heads = config_.n_heads;
        if (local_n_kv_heads <= 0)
            local_n_kv_heads = config_.n_kv_heads;

        // Create InferenceMode for centralized mode logic
        InferenceMode inference_mode(config_.activation_precision);

        // Stage 2.5: Per-head QK RMSNorm (Qwen3)
        // Applied to Q and K projections independently before RoPE.
        // Each head is normalized separately with gamma of shape [head_dim].
        if (layer.q_norm && layer.k_norm)
        {
            LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " using QK norm (Qwen3)");

            // Q norm: normalize each Q head independently
            QKNormStage::Params q_norm_params;
            q_norm_params.input = buffers.Q;
            q_norm_params.output = buffers.Q; // In-place
            q_norm_params.gamma = layer.q_norm;
            q_norm_params.n_heads = local_n_heads;
            q_norm_params.head_dim = config_.head_dim;
            q_norm_params.eps = config_.rms_norm_eps;
            q_norm_params.seq_len = total_tokens;
            q_norm_params.device_id = device;
            q_norm_params.input_buffer_id = BufferId::Q_PROJ;
            q_norm_params.output_buffer_id = BufferId::Q_PROJ; // In-place

            graph.addNode(prefix + "q_norm",
                          ComputeStageFactory::createQKNorm(q_norm_params),
                          device);

            if (env.execution.exec_gemm && layer.wq)
            {
                graph.addDependency(prefix + "q_norm", prefix + "qkv_proj");
            }

            // K norm: normalize each K head independently
            QKNormStage::Params k_norm_params;
            k_norm_params.input = buffers.K;
            k_norm_params.output = buffers.K; // In-place
            k_norm_params.gamma = layer.k_norm;
            k_norm_params.n_heads = local_n_kv_heads;
            k_norm_params.head_dim = config_.head_dim;
            k_norm_params.eps = config_.rms_norm_eps;
            k_norm_params.seq_len = total_tokens;
            k_norm_params.device_id = device;
            k_norm_params.input_buffer_id = BufferId::K_PROJ;
            k_norm_params.output_buffer_id = BufferId::K_PROJ; // In-place

            graph.addNode(prefix + "k_norm",
                          ComputeStageFactory::createQKNorm(k_norm_params),
                          device);

            if (env.execution.exec_gemm && layer.wk)
            {
                graph.addDependency(prefix + "k_norm", prefix + "qkv_proj");
            }
        }

        // Stage 3: RoPE on Q and K
        if (env.execution.exec_rope)
        {
            // For batched execution, pass the full position_ids array
            // This enables correct per-token position encoding for variable-length sequences
            // Fallback pos_offset is still used for compatibility with single-sequence execution
            int pos_offset = position_ids ? position_ids[0] : 0;

            RoPEStage::Params rope_params;
            rope_params.device_id = device; // Use graph's target device for kernel dispatch
            rope_params.Q = buffers.Q;
            rope_params.K = buffers.K;
            rope_params.n_heads = local_n_heads;       // Use local head count for TP
            rope_params.n_kv_heads = local_n_kv_heads; // Use local KV head count for TP
            rope_params.head_dim = config_.head_dim;
            rope_params.pos_offset = pos_offset;
            rope_params.position_ids = position_ids; // Pass full array for batched execution
            rope_params.theta_base = config_.rope_theta;
            rope_params.seq_len = total_tokens; // Use total_tokens = batch_size * seq_len
            rope_params.q_buffer_id = BufferId::Q_PROJ;
            rope_params.k_buffer_id = BufferId::K_PROJ;

            graph.addNode(prefix + "rope",
                          ComputeStageFactory::createRoPE(rope_params),
                          device);

            if (env.execution.exec_gemm)
            {
                // If QK norms are present, RoPE depends on norms (which depend on qkv_proj)
                if (layer.q_norm && layer.k_norm)
                {
                    graph.addDependency(prefix + "rope", prefix + "q_norm");
                    graph.addDependency(prefix + "rope", prefix + "k_norm");
                }
                else
                {
                    graph.addDependency(prefix + "rope", prefix + "qkv_proj");
                }
            }
        }

        // Stage 4: Attention computation with KV cache integration
        // NOTE: Decomposed attention (Phase 9) is now the ONLY supported path.
        // Legacy AttentionWithKVCacheStage has been removed (Phase 7 cleanup).
        std::string wo_producer_node;
        if (env.execution.exec_attention)
        {
            // Phase 9 Decomposed Path: KVCacheAppendStage + AttentionComputeStage
            if (kv_cache)
            {
                // For batched execution, K/V are [batch_size * seq_len, kv_dim]
                // Each sequence's K/V is appended to its own seq_idx in the cache
                int total_tokens = batch_size * seq_len;

                KVCacheAppendStage::Params kv_append_params;
                kv_append_params.device_id = device;
                kv_append_params.K = buffers.K;
                kv_append_params.k_buffer_id = BufferId::K_PROJ;

                kv_append_params.V = buffers.V;
                kv_append_params.v_buffer_id = BufferId::V_PROJ;
                kv_append_params.kv_cache = kv_cache;
                // For PP stages: map global layer index to local KV cache index
                kv_append_params.layer_idx = layer_idx - config_.pp_layer_offset;
                kv_append_params.seq_idx = 0; // Starting seq_idx
                kv_append_params.num_tokens = total_tokens;
                kv_append_params.batch_size = batch_size; // Phase 3: Per-sequence append
                kv_append_params.seq_len = seq_len;       // Phase 3: Tokens per sequence

                // Phase 5.4: VNNI-safe Q16 KV cache quantization parameters
                kv_append_params.kv_cache_scale = config_.kv_cache_scale;
                kv_append_params.head_dim = config_.head_dim;

                graph.addNode(prefix + "kv_append",
                              ComputeStageFactory::createKVCacheAppend(kv_append_params),
                              device);

                if (env.execution.exec_rope)
                {
                    graph.addDependency(prefix + "kv_append", prefix + "rope");
                }
                else if (env.execution.exec_gemm)
                {
                    graph.addDependency(prefix + "kv_append", prefix + "qkv_proj");
                }
            }

            // For Hybrid mode attention path selection:
            // - Decomposed attention: Use FP32 K_rope/V_dequant for best precision
            // - Fused attention: Use Q8_1 K/V (kernel only supports Q8_1 K/V currently)
            // KV cache always gets FP32 K_rope (stored separately from attention path)
            ITensor *K_for_attn = buffers.K;
            ITensor *V_for_attn = buffers.V;

            int total_query_tokens = batch_size * seq_len;
            int kv_len = total_query_tokens; // Static hint for mode detection (actual queried at runtime)
            bool use_gather_stage = false;

            // For attention K/V source:
            // - Prefill (cached_tokens == 0): Use projected K/V directly
            // - Decode (cached_tokens > 0 and batch_size == 1): Use cache (single-sequence only)
            // - Batched decode (cached_tokens > 0 and batch_size > 1): Gather K/V from multiple cache slots
            // Map global layer index to local KV cache index for PP stages
            int kv_local_layer = layer_idx - config_.pp_layer_offset;

            if (kv_cache)
            {
                int cached_tokens = kv_cache->get_cached_tokens(kv_local_layer, 0);
                if (cached_tokens > 0 && batch_size == 1)
                {
                    // Single-sequence decode: read K/V from cache
                    // Use ITensor* directly (works for both CPU and GPU caches)
                    K_for_attn = kv_cache->get_k(kv_local_layer, 0);
                    V_for_attn = kv_cache->get_v(kv_local_layer, 0);
                    kv_len = cached_tokens;
                    LOG_TRACE("[Qwen2Graph] Layer " << layer_idx << " (local=" << kv_local_layer << ") using cached K/V (decode mode)");
                }
                else if (cached_tokens > 0 && batch_size > 1)
                {
                    // Batched decode: gather K/V from multiple cache slots
                    if (buffers.gathered_K && buffers.gathered_V)
                    {
                        use_gather_stage = true;
                        K_for_attn = buffers.gathered_K;
                        V_for_attn = buffers.gathered_V;
                        // kv_len will be updated by gather stage; use cache max for now
                        kv_len = cached_tokens; // Approximate - actual max determined at gather
                        LOG_TRACE("[Qwen2Graph] Layer " << layer_idx << " using gathered K/V (batched decode mode)");
                    }
                    else
                    {
                        // Fallback: use projected K/V if gather buffers not provided
                        LOG_WARN("[Qwen2Graph] Layer " << layer_idx
                                                       << " batched decode but no gather buffers - using projected K/V");
                    }
                }
                else
                {
                    // Prefill or batched prefill: use projected K/V directly
                    // KV cache will be populated but attention uses fresh projections
                    LOG_TRACE("[Qwen2Graph] Layer " << layer_idx << " using projected K/V (prefill/batch mode)");
                }
            }

            // Add KVCacheGatherStage if batched decode
            if (use_gather_stage)
            {
                KVCacheGatherStage::Params gather_params;
                gather_params.kv_cache = kv_cache;
                // For PP stages: map global layer index to local KV cache index
                gather_params.layer_idx = layer_idx - config_.pp_layer_offset;
                gather_params.batch_size = batch_size;
                gather_params.out_K = buffers.gathered_K;
                gather_params.out_V = buffers.gathered_V;
                // Note: out_max_kv_len and out_per_seq_kv_lens can be retrieved from stage after execute

                graph.addNode(prefix + "kv_gather",
                              ComputeStageFactory::createKVCacheGather(gather_params),
                              device);

                // Gather depends on append (must append new tokens before gathering full history)
                graph.addDependency(prefix + "kv_gather", prefix + "kv_append");
            }

            // Decomposed Path: Attention -> Wo projection
            if (env.execution.exec_attention)
            {
                AttentionMode mode = detect_attention_mode(batch_size, seq_len, kv_len);
                LOG_TRACE("[Qwen2Graph] Layer " << layer_idx
                                                << " attention mode: " << attention_mode_name(mode)
                                                << " (batch_size=" << batch_size << ", seq_len=" << seq_len << ", kv_len=" << kv_len << ")");

                AttentionComputeStage::Params attn_params;
                attn_params.Q = buffers.Q;
                attn_params.K = K_for_attn;
                attn_params.V = V_for_attn;
                attn_params.output = buffers.attn_output;
                attn_params.batch_size = batch_size;
                attn_params.seq_len = seq_len;
                attn_params.kv_len = kv_len;
                attn_params.n_heads = local_n_heads;
                attn_params.n_kv_heads = local_n_kv_heads;
                attn_params.head_dim = config_.head_dim;
                attn_params.causal = true;
                attn_params.window_size = -1;
                attn_params.attention_mode = mode;
                attn_params.auto_detect_mode = true;
                attn_params.workspace_scores = buffers.workspace_scores;
                attn_params.workspace_context = buffers.workspace_context;
                attn_params.workspace_mask = buffers.workspace_mask;
                attn_params.kv_cache = kv_cache;
                attn_params.layer_idx = layer_idx - config_.pp_layer_offset;
                attn_params.read_kv_from_cache = device.is_gpu() &&
                                                 (!kv_cache || kv_cache->precision() != ActivationPrecision::Q8_1);
                attn_params.position_offset = position_ids ? position_ids[0] : 0;
                attn_params.mpi_ctx = mpi_ctx_.get();
                attn_params.device_id = device;
                attn_params.q_buffer_id = BufferId::Q_PROJ;
                attn_params.output_buffer_id = BufferId::ATTN_OUTPUT;
                attn_params.workspace_scores_buffer_id = BufferId::ATTN_SCORES_WORKSPACE;
                attn_params.workspace_context_buffer_id = BufferId::ATTN_CONTEXT_WORKSPACE;

                graph.addNode(prefix + "attention",
                              ComputeStageFactory::createAttentionCompute(attn_params),
                              device);

                if (use_gather_stage)
                    graph.addDependency(prefix + "attention", prefix + "kv_gather");
                else if (kv_cache)
                    graph.addDependency(prefix + "attention", prefix + "kv_append");
                else if (env.execution.exec_rope)
                    graph.addDependency(prefix + "attention", prefix + "rope");
                else if (env.execution.exec_gemm)
                    graph.addDependency(prefix + "attention", prefix + "qkv_proj");

                LOG_DEBUG("[Qwen2Graph] Using decomposed attention path");
            }

            // Stage 5: Output projection (Wo)
            if (env.execution.exec_gemm && layer.wo)
            {
                int wo_n = static_cast<int>(layer.wo->shape()[0]);
                int wo_k = static_cast<int>(layer.wo->shape()[1]);

                LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " Wo dims: wo_n=" << wo_n
                                                << " wo_k=" << wo_k
                                                << " wo_shape=[" << layer.wo->shape()[0] << "," << layer.wo->shape()[1] << "]"
                                                << " attn_output_shape=" << buffers.attn_output->shape()[0] << "x" << buffers.attn_output->shape()[1]);

                wo_producer_node = prefix + "wo_proj";
                graph.addNode(wo_producer_node,
                              ComputeStageFactory::createGEMM(
                                  GEMMStage::Params{
                                      .device_id = device,
                                      .A = buffers.attn_output,
                                      .B = layer.wo,
                                      .C = buffers.attn_proj,
                                      .m = total_tokens,
                                      .n = wo_n,
                                      .k = wo_k,
                                      .alpha = 1.0f,
                                      .beta = 0.0f,
                                      .transpose_B = false,
                                      .gemm_context = GemmContext::ATTN,
                                      .a_buffer_id = BufferId::ATTN_OUTPUT,
                                      .c_buffer_id = BufferId::ATTN_PROJ}),
                              device);

                if (env.execution.exec_attention)
                {
                    graph.addDependency(wo_producer_node, prefix + "attention");
                }
            }

            // Common AllReduce for Wo
            if (env.execution.exec_gemm && layer.wo && !wo_producer_node.empty())
            {
                bool wo_is_sharded = isRowParallelSharded(layer.wo);

                if (wo_is_sharded && needsTPAllreduce())
                {
                    size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);

                    TensorBase *allreduce_buffer = buffers.attn_proj;
                    BufferId wo_allreduce_bid = BufferId::ATTN_PROJ;

                    std::string stage_name = prefix + "wo_allreduce";
                    auto allreduce_stage = createTPAllreduceStage(
                        allreduce_buffer, allreduce_count, device, layer_idx, /*is_attention=*/true, stage_name,
                        wo_allreduce_bid);

                    if (allreduce_stage)
                    {
                        graph.addNode(stage_name, std::move(allreduce_stage), device);
                        graph.addDependency(stage_name, wo_producer_node);
                        wo_producer_node = stage_name;

                        LOG_TRACE("[Qwen2Graph] Layer " << layer_idx
                                                        << " Wo: row-parallel sharded, adding allreduce");
                    }
                }
            }
        }

        // Attention residual is now fused into FusedResidualNormStage in buildFFNGraph
        // (for non-first layers) or handled by the first layer's standalone RMSNorm path.

        return graph;
    }

    ComputeGraph Qwen2Graph::buildFFNGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        DeviceId device)
    {
        ComputeGraph graph;
        const auto &env = debugEnv();
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        // Compute total tokens for GEMM m parameter
        int total_tokens = batch_size * seq_len;

        // Determine backend type from DeviceId
        ComputeBackendType backend = ComputeBackendType::CPU;
        if (device.is_cuda())
        {
            backend = ComputeBackendType::GPU_CUDA;
        }
        else if (device.is_rocm())
        {
            backend = ComputeBackendType::GPU_ROCM;
        }

        // Stage 1: Pre-FFN RMSNorm (fused with attention residual add)
        // Combines the attention output residual add with FFN normalization.
        // On GPU: single fused kernel. On CPU: sequential ResidualAdd + RMSNorm via KernelFactory.
        if (env.execution.exec_rmsnorm)
        {
            FusedResidualNormStage::Params fused_params;
            fused_params.device_id = device;
            fused_params.input = buffers.attn_proj;         // Wo output (to be added)
            fused_params.residual = buffers.current_hidden; // Hidden state (in-place update)
            fused_params.gamma = layer.ffn_norm;            // FFN norm gamma
            fused_params.norm_output = buffers.normalized;
            fused_params.eps = config_.rms_norm_eps;
            fused_params.seq_len = total_tokens;
            fused_params.hidden_dim = config_.d_model;
            fused_params.input_buffer_id = BufferId::ATTN_PROJ;
            fused_params.residual_buffer_id = BufferId::HIDDEN_STATE;
            fused_params.norm_output_buffer_id = BufferId::NORMALIZED;

            graph.addNode(prefix + "ffn_norm",
                          ComputeStageFactory::createFusedResidualNorm(fused_params),
                          device);
        }

        // Stage 2: Gate and Up projections using FusedGateUpGEMMStage
        if (env.execution.exec_gemm && layer.gate_proj && layer.up_proj)
        {
            LOG_DEBUG("[Qwen2Graph] FFN using FusedGateUpGEMMStage");

            int k = config_.d_model;
            int gate_n = static_cast<int>(layer.gate_proj->shape()[0]);
            int up_n = static_cast<int>(layer.up_proj->shape()[0]);

            FusedGateUpGEMMStage::Params gate_up_params;
            gate_up_params.input = buffers.normalized;
            gate_up_params.m = total_tokens; // Use total_tokens = batch_size * seq_len
            gate_up_params.k = k;
            gate_up_params.w_gate = layer.gate_proj;
            gate_up_params.output_gate = buffers.gate;
            gate_up_params.n_gate = gate_n;
            gate_up_params.w_up = layer.up_proj;
            gate_up_params.output_up = buffers.up;
            gate_up_params.n_up = up_n;
            gate_up_params.mpi_ctx = mpi_ctx_.get();
            gate_up_params.device_id = device;
            gate_up_params.input_buffer_id = BufferId::NORMALIZED;
            gate_up_params.output_gate_buffer_id = BufferId::GATE_PROJ;
            gate_up_params.output_up_buffer_id = BufferId::UP_PROJ;

            graph.addNode(prefix + "gate_up_proj",
                          ComputeStageFactory::createFusedGateUpGEMM(gate_up_params),
                          device);

            if (env.execution.exec_rmsnorm)
            {
                graph.addDependency(prefix + "gate_up_proj", prefix + "ffn_norm");
            }
        }

        // Stage 3: SwiGLU activation
        // Always fused into down_proj GEMM: silu(gate)*up + GEMM in single dispatch.
        // On GPU: kernel-level fusion (quantize + SwiGLU + GEMM).
        // On CPU: multiply_tensor_with_fused_swiglu() applies SwiGLU then GEMM.
        const bool swiglu_fusion = true;

        // Stage 4: Down projection
        if (env.execution.exec_gemm && layer.down_proj)
        {
            int down_n = static_cast<int>(layer.down_proj->shape()[0]);
            int down_k = static_cast<int>(layer.down_proj->shape()[1]);

            GEMMStage::Params down_params{
                .device_id = device,
                .A = buffers.up,
                .B = layer.down_proj,
                .C = buffers.attn_proj,
                .m = total_tokens,
                .n = down_n,
                .k = down_k,
                .alpha = 1.0f,
                .beta = 0.0f,
                .transpose_B = false,
                .gemm_context = GemmContext::FFN,
                .a_buffer_id = BufferId::UP_PROJ,
                .c_buffer_id = BufferId::ATTN_PROJ};

            // SwiGLU fusion: pass gate buffer to GEMM for fused silu(gate)*up + GEMM
            if (swiglu_fusion && env.execution.exec_swiglu)
            {
                down_params.gate_input = buffers.gate;
                down_params.do_swiglu = true;
            }

            graph.addNode(prefix + "down_proj",
                          ComputeStageFactory::createGEMM(down_params),
                          device);

            if (swiglu_fusion || !env.execution.exec_swiglu)
            {
                // SwiGLU fused into GEMM or disabled: down_proj depends directly on gate_up_proj
                if (env.execution.exec_gemm)
                {
                    graph.addDependency(prefix + "down_proj", prefix + "gate_up_proj");
                }
            }
            else if (env.execution.exec_swiglu)
            {
                graph.addDependency(prefix + "down_proj", prefix + "swiglu");
            }

            bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
            bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);

            if (needs_allreduce && needsTPAllreduce())
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * down_n;
                LOG_DEBUG("[buildFFNGraph] Adding down_allreduce: ffn_column_parallel="
                          << config_.ffn_column_parallel << " down_is_row_sharded=" << down_is_row_sharded
                          << " count=" << allreduce_count);

                std::string stage_name = prefix + "down_allreduce";
                auto allreduce_stage = createTPAllreduceStage(
                    buffers.attn_proj, allreduce_count, device, layer_idx, /*is_attention=*/false, stage_name,
                    BufferId::ATTN_PROJ);

                if (allreduce_stage)
                {
                    graph.addNode(stage_name, std::move(allreduce_stage), device);
                    graph.addDependency(stage_name, prefix + "down_proj");
                }
            }
        }

        // Stage 5: Residual connection
        // Non-last layers: Skip - fused into next layer's FusedResidualNormStage attn_norm
        // Last layer must keep ffn_residual since final_norm doesn't include residual add
        const bool skip_ffn_residual = (layer_idx < config_.pp_layer_offset + config_.n_layers - 1);
        if (env.execution.exec_residual && !skip_ffn_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.device_id = device;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
            res_params.input_buffer_id = BufferId::ATTN_PROJ;
            res_params.residual_buffer_id = BufferId::HIDDEN_STATE;
            res_params.output_buffer_id = BufferId::HIDDEN_STATE; // In-place with residual

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device);

            if (env.execution.exec_gemm && layer.down_proj)
            {
                bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
                bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);

                if (needs_allreduce && needsTPAllreduce())
                {
                    graph.addDependency(prefix + "ffn_residual", prefix + "down_allreduce");
                }
                else
                {
                    graph.addDependency(prefix + "ffn_residual", prefix + "down_proj");
                }
            }
        }

        return graph;
    }

    // =============================================================================
    // Helper Methods
    // =============================================================================

    std::vector<int> Qwen2Graph::buildPositionIds(int seq_len, int batch_size, int offset)
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

    GraphSchema Qwen2Graph::getSchema() const
    {
        Qwen2SchemaFactory factory;
        return factory.createSchema();
    }

    GraphResolverConfig Qwen2Graph::getResolverConfig(int seq_len) const
    {
        GraphResolverConfig config;

        // MPI context
        if (mpi_ctx_)
        {
            config.world_size = mpi_ctx_->world_size();
            config.rank = mpi_ctx_->rank();
            config.mpi_ctx = mpi_ctx_.get();
        }

        // Sequence configuration
        config.seq_len = seq_len;
        config.batch_size = 1;

        // Device configuration
        config.default_device = config_.default_device;

        // Model architecture
        config.n_layers = config_.n_layers;
        config.d_model = config_.d_model;
        config.n_heads = config_.n_heads;
        config.n_kv_heads = config_.n_kv_heads;
        config.head_dim = config_.head_dim;
        config.d_ff = config_.d_ff;
        config.vocab_size = config_.vocab_size;
        config.rms_norm_eps = config_.rms_norm_eps;
        config.rope_theta = config_.rope_theta;

        // Phase 5.4: VNNI-safe Q16 KV cache scale
        config.kv_cache_scale = config_.kv_cache_scale;

        // TP-adjusted local dimensions
        // Use local head counts when QKV is column-parallel
        config.local_n_heads = config_.qkv_column_parallel
                                   ? config_.local_n_heads
                                   : config_.n_heads;
        config.local_n_kv_heads = config_.qkv_column_parallel
                                      ? config_.local_n_kv_heads
                                      : config_.n_kv_heads;

        // Validate (safety check for uninitialized config)
        if (config.local_n_heads <= 0)
            config.local_n_heads = config_.n_heads;
        if (config.local_n_kv_heads <= 0)
            config.local_n_kv_heads = config_.n_kv_heads;

        // Use local FFN dimension when FFN is column-parallel
        config.local_d_ff = config_.ffn_column_parallel
                                ? config_.d_ff_local
                                : config_.d_ff;
        if (config.local_d_ff <= 0)
            config.local_d_ff = config_.d_ff;

        // Use local vocab when LM head is column-parallel
        config.local_vocab = config_.lm_head_column_parallel
                                 ? config_.vocab_local
                                 : config_.vocab_size;
        if (config.local_vocab <= 0)
            config.local_vocab = config_.vocab_size;

        LOG_DEBUG("[Qwen2Graph::getResolverConfig] Created config: "
                  << "seq_len=" << config.seq_len << ", "
                  << "local_n_heads=" << config.local_n_heads << " (n_heads=" << config.n_heads << "), "
                  << "local_n_kv_heads=" << config.local_n_kv_heads << " (n_kv_heads=" << config.n_kv_heads << "), "
                  << "local_d_ff=" << config.local_d_ff << " (d_ff=" << config.d_ff << "), "
                  << "local_vocab=" << config.local_vocab << " (vocab_size=" << config.vocab_size << ")");

        return config;
    }

    void Qwen2Graph::addFinalNormToGraph(
        ComputeGraph &graph,
        TensorBase *hidden,
        TensorBase *normalized_out,
        const std::string &prev_node,
        int n_tokens,
        DeviceId device)
    {
        RMSNormStage::Params norm_params;
        norm_params.input = hidden;
        norm_params.output = normalized_out;
        norm_params.gamma = weights_.final_norm;
        norm_params.eps = config_.rms_norm_eps;
        norm_params.seq_len = n_tokens;
        norm_params.device_id = device;
        norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
        norm_params.output_buffer_id = BufferId::NORMALIZED;

        graph.addNode("final_norm",
                      ComputeStageFactory::createRMSNorm(norm_params),
                      device);

        if (!prev_node.empty())
        {
            graph.addDependency("final_norm", prev_node);
        }
    }

    const TPDomain *Qwen2Graph::getDomainForLayer(int layer_idx, bool is_attention) const
    {
        if (!config_.multi_domain_tp_config)
        {
            return nullptr; // No domain config - use legacy MPI path
        }
        return config_.multi_domain_tp_config->domainForLayer(layer_idx, is_attention);
    }

    // =========================================================================
    // TP Allreduce Helpers
    // =========================================================================

    bool Qwen2Graph::needsTPAllreduce() const
    {
        // Check LOCAL TP first (takes precedence)
        if (config_.local_tp_ctx && config_.local_tp_ctx->degree() > 1)
        {
            return true;
        }
        // Check GLOBAL TP
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            return true;
        }
        return false;
    }

    std::unique_ptr<IComputeStage> Qwen2Graph::createTPAllreduceStage(
        TensorBase *buffer,
        size_t count,
        DeviceId device,
        int layer_idx,
        bool is_attention,
        const std::string &stage_name,
        std::optional<BufferId> tensor_buffer_id) const
    {
        // LOCAL TP takes precedence if configured
        if (config_.local_tp_ctx && config_.local_tp_ctx->degree() > 1)
        {
            LOG_DEBUG("[Qwen2Graph] Creating TPAllreduceStage (LOCAL): degree="
                      << config_.local_tp_ctx->degree()
                      << " device_idx=" << config_.local_tp_device_idx
                      << " count=" << count
                      << " backend=" << static_cast<int>(config_.local_tp_ctx->backend())
                      << " stage_name=" << stage_name);

            // =========================================================
            // Register BAR-backed tensor for PCIeBAR allreduce
            // =========================================================
            // For PCIeBAR backend, row-parallel output tensors need to be
            // registered with LocalTPContext so executePCIeBarAllreduce()
            // can find them by stage name. The tensor was allocated as
            // BAR-backed by DeviceGraphBufferManager if conditions were met.
            //
            // This registration is called for EACH device orchestrator,
            // so each device registers its own tensor for this stage.
            // =========================================================
            if (config_.local_tp_ctx->backend() == CollectiveBackendType::PCIE_BAR &&
                config_.local_tp_device_idx >= 0 &&
                static_cast<size_t>(config_.local_tp_device_idx) < config_.local_tp_ctx->devices().size())
            {
                const GlobalDeviceAddress &device_addr =
                    config_.local_tp_ctx->devices()[config_.local_tp_device_idx];

                // Register the tensor with the stage name
                // registerBARBackedOutput checks if tensor is BAR-backed and handles accordingly
                config_.local_tp_ctx->registerBARBackedOutput(stage_name, device_addr, buffer);
            }

            TPAllreduceStage::Params params;
            params.device_id = device;
            params.tp_ctx = config_.local_tp_ctx; // ILocalTPContext* -> ITPContext* (inheritance)
            params.tensor = buffer;
            params.count = count;
            params.stage_name = stage_name;
            params.precision = config_.getAllreducePrecisionForLayer(layer_idx);
            params.tensor_buffer_id = tensor_buffer_id;

            return std::make_unique<TPAllreduceStage>(params);
        }

        // Fall back to GLOBAL TP (MPI-based)
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            LOG_DEBUG("[Qwen2Graph] Creating AllreduceStage (MPI): world_size="
                      << mpi_ctx_->world_size()
                      << " rank=" << mpi_ctx_->rank()
                      << " count=" << count);

            AllreduceStage::Params params;
            params.device_id = device;
            params.mpi_ctx = mpi_ctx_.get();
            params.buffer = buffer;
            params.count = count;
            params.collective_ctx = nullptr;
            params.domain = getDomainForLayer(layer_idx, is_attention);

            return ComputeStageFactory::createAllreduce(params);
        }

        // No TP active - should not reach here (caller should check needsTPAllreduce())
        LOG_WARN("[Qwen2Graph] createTPAllreduceStage called but no TP active");
        return nullptr;
    }

} // namespace llaminar2
