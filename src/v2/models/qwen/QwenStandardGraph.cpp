/**
 * @file QwenStandardGraph.cpp
 * @brief Qwen2-specific graph builder implementation
 * @author David Sanftenberg
 * @date December 2025
 *
 * Implements the Qwen2-specific attention pattern and schema.
 * All shared infrastructure is in QwenGraphBase.cpp.
 */

#include "QwenStandardGraph.h"
#include "Qwen2Schema.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../tensors/TensorSlice.h"
#include "../../tensors/Tensors.h"
#include "../../utils/MPIContext.h"
#include "../../execution/local_execution/graph/GraphBuildUtils.h"
#include "../../execution/config/RuntimeConfig.h"
#include "../../collective/ILocalTPContext.h"
#include "../../memory/BufferId.h"
#include <stdexcept>

namespace llaminar2
{

    // Import graph_utils for cleaner code
    using namespace graph_utils;

    // =============================================================================
    // Constructors (delegate to QwenGraphBase, populate Qwen2 schema)
    // =============================================================================

    QwenStandardGraph::QwenStandardGraph(std::shared_ptr<ModelContext> model_ctx,
                           std::shared_ptr<IMPIContext> mpi_ctx,
                           const GraphConfig &config)
        : QwenGraphBase(std::move(model_ctx), std::move(mpi_ctx), config)
    {
        // Qwen2-specific: populate per-layer allreduce precision from schema
        if (config_.tp_allreduce_precision.empty() && config_.n_layers > 0)
        {
            Qwen2SchemaFactory factory;
            auto schema = factory.createSchema();
            config_.populateAllreducePrecision(
                schema.tp_allreduce_default_precision,
                schema.tp_allreduce_fp32_layer_count);
            LOG_DEBUG("[QwenStandardGraph] Populated per-layer allreduce precision: "
                      << "fp32_layers=" << schema.tp_allreduce_fp32_layer_count
                      << " default=" << schema.tp_allreduce_default_precision);
        }
    }

    QwenStandardGraph::QwenStandardGraph(const GraphConfig &config,
                           std::shared_ptr<IMPIContext> mpi_ctx)
        : QwenGraphBase(config, std::move(mpi_ctx))
    {
        // Qwen2-specific: populate per-layer allreduce precision from schema
        if (config_.tp_allreduce_precision.empty() && config_.n_layers > 0)
        {
            Qwen2SchemaFactory factory;
            auto schema = factory.createSchema();
            config_.populateAllreducePrecision(
                schema.tp_allreduce_default_precision,
                schema.tp_allreduce_fp32_layer_count);
        }
    }

    // =============================================================================
    // Schema
    // =============================================================================

    GraphSchema QwenStandardGraph::getSchema() const
    {
        Qwen2SchemaFactory factory;
        return factory.createSchema();
    }

    // =============================================================================
    // Attention Graph Building (Qwen2-specific)
    // =============================================================================

    ComputeGraph QwenStandardGraph::buildAttentionGraph(
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
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";
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

        // Stage 1: Pre-attention RMSNorm
        addPreAttentionNorm(graph, prefix, buffers, layer.attn_norm,
                            total_tokens, layer_idx, device);

        // Stage 2: Q/K/V projections
        const bool has_qkv_proj = (layer.wq && layer.wk && layer.wv);
        if (has_qkv_proj)
        {
            LOG_DEBUG("[QwenStandardGraph] Using FusedQKVGEMMStage");

            int k = config_.d_model;
            int q_n = static_cast<int>(layer.wq->shape()[0]);
            int k_n = static_cast<int>(layer.wk->shape()[0]);
            int v_n = static_cast<int>(layer.wv->shape()[0]);

            LOG_DEBUG("[QwenStandardGraph] Layer " << layer_idx << " QKV dims: q_n=" << q_n
                                            << " k_n=" << k_n << " v_n=" << v_n
                                            << " wq_shape=[" << layer.wq->shape()[0] << "," << layer.wq->shape()[1] << "]"
                                            << " wk_shape=[" << layer.wk->shape()[0] << "," << layer.wk->shape()[1] << "]");

            graph.addNode(prefix + "qkv_proj",
                          ComputeStageFactory::createFusedQKVGEMM({
                              .device_id = device,
                              .input = buffers.normalized,
                              .m = total_tokens,
                              .k = k,
                              .wq = layer.wq,
                              .output_q = buffers.Q,
                              .n_q = q_n,
                              .bias_q = layer.q_bias,
                              .wk = layer.wk,
                              .output_k = buffers.K,
                              .n_k = k_n,
                              .bias_k = layer.k_bias,
                              .wv = layer.wv,
                              .output_v = buffers.V,
                              .n_v = v_n,
                              .bias_v = layer.v_bias,
                              .input_buffer_id = BufferId::NORMALIZED,
                              .output_q_buffer_id = BufferId::Q_PROJ,
                              .output_k_buffer_id = BufferId::K_PROJ,
                              .output_v_buffer_id = BufferId::V_PROJ,
                              .prepared_store = prepared_weight_store_,
                          }),
                          device);
            graph.addDependency(prefix + "qkv_proj", prefix + "attn_norm");
        }

        // Resolve local head counts for TP
        auto [local_n_heads, local_n_kv_heads] = resolveLocalHeadCounts();

        // Stage 2.5: Per-head QK RMSNorm (Qwen3)
        bool has_qk_norms = addQKNorms(
            graph, prefix, buffers, layer,
            local_n_heads, local_n_kv_heads, total_tokens, device,
            has_qkv_proj ? prefix + "qkv_proj" : prefix + "attn_norm",
            has_qkv_proj ? prefix + "qkv_proj" : prefix + "attn_norm");

        // Stage 3: RoPE on Q and K
        std::string rope_node = addRoPE(
            graph, prefix, buffers,
            local_n_heads, local_n_kv_heads, total_tokens,
            position_ids, device);

        // Wire RoPE dependencies
        if (has_qk_norms)
        {
            graph.addDependency(rope_node, prefix + "q_norm");
            graph.addDependency(rope_node, prefix + "k_norm");
        }
        else if (has_qkv_proj)
        {
            graph.addDependency(rope_node, prefix + "qkv_proj");
        }

        // Stage 4: KV cache + Attention
        std::string attn_node = addKVCacheAndAttention(
            graph, prefix, buffers, layer_idx,
            seq_len, batch_size, local_n_heads, local_n_kv_heads,
            kv_cache, position_ids, device, has_qkv_proj, rope_node);

        // Stage 5: Wo projection + optional TP allreduce
        std::string terminal = addWoProjectionAndAllreduce(
            graph, prefix, buffers, layer.wo,
            total_tokens, layer_idx, device, attn_node);

        graph.setTerminalNode(terminal);
        return graph;
    }

} // namespace llaminar2
