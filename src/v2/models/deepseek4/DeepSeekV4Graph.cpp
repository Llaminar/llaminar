/**
 * @file DeepSeekV4Graph.cpp
 * @brief Bind DeepSeek-V4 model operands to participant-local graph stages.
 *
 * Attention dataflow for layer l (rows M, all buffers arena-owned):
 *   NORMALIZED = HCRead(trunk)                                      [M, d]
 *   DSV4_QR    = RMS(NORMALIZED . q_a) * q_a_norm                   [M, q_rank]
 *   DSV4_Q     = RoPE_tail(RMS_head(DSV4_QR . q_b))                 [M, H*hd]
 *   K_PROJ     = RoPE_tail(RMS(NORMALIZED . kv) * kv_norm)          [M, hd]  (V_PROJ duplicate for the cache)
 *   raw ring  <- K_PROJ at position % ring
 *   ratio>0:  compressor ring <- (kv, gate + ape); pool <- gated pooling of completed blocks
 *   ratio==4: indexer queries/weights/compressor -> top-k compressed blocks
 *   DSV4_ATTN  = inverse-RoPE(sink softmax attention over window + compressed)
 *   DSV4_OA    = grouped wo_a(DSV4_ATTN);  ATTN_PROJ = DSV4_OA . wo_b
 *   trunk      = HCWrite(trunk, ATTN_PROJ)
 * Ordering: every consumer depends on its producers' node names; persistent
 * banks are written before any reader of the same graph and read positions are
 * functions of the row's absolute position only.
 */
#include "DeepSeekV4Graph.h"
#include "DeepSeekV4Config.h"
#include "../../execution/compute_stages/stages/deepseek4/DeepSeekV4PrimitiveStage.h"
#include "../../execution/compute_stages/ComputeStageFactory.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../memory/BufferArena.h"
#include "../../tensors/Tensors.h"

#include <cmath>
#include <stdexcept>

namespace llaminar2
{
DeepSeekV4Graph::DeepSeekV4Graph(const GraphConfig &config, std::shared_ptr<IMPIContext> mpi)
    : Qwen35MoEGraph(config, std::move(mpi))
{
    if (!config_.deepseek_v4)
        throw std::invalid_argument("DeepSeek-V4 graph requires an authenticated deepseek4 GGUF configuration");
    if (config_.default_device.is_rocm())
        throw std::invalid_argument(deepseek4::kROCmUnimplemented);
}

GraphSchema DeepSeekV4Graph::getSchema() const
{
    auto schema = Qwen35MoEGraph::getSchema();
    schema.name = architectureName();
    applyDeepSeekV4BufferLayout(schema);
    return schema;
}

GraphResolverConfig DeepSeekV4Graph::getResolverConfig(int seq_len) const
{
    auto resolver = Qwen35MoEGraph::getResolverConfig(seq_len);
    applyDeepSeekV4BufferGeometry(resolver, DeepSeekV4MemoryGeometry::fromConfig(*config_.deepseek_v4),
                                  std::max(resolver.seq_len, config_.max_activation_rows), config_.max_seq_len);
    return resolver;
}

void DeepSeekV4Graph::setArena(BufferArena *arena)
{
    Qwen35MoEGraph::setArena(arena);
    for (const auto &entry : deepSeekV4BufferDeclarations())
    {
        buffers_.layer_buffers.extensions.erase(entry.id);
        if (arena)
            if (auto *view = dynamic_cast<TensorBase *>(arena->getTensor(entry.id)))
                buffers_.layer_buffers.extensions.emplace(entry.id, view);
    }
}

DeepSeekV4StateLayout DeepSeekV4Graph::stateLayout() const
{
    const auto &m = *config_.deepseek_v4;
    const auto *ring = buffers_.layer_buffers.get(BufferId::DSV4_RAW_RING);
    if (!ring)
        throw std::invalid_argument("DeepSeek-V4 graph requires its admitted raw-ring bank");
    const std::size_t per_layer = static_cast<std::size_t>(m.layers()) * m.headDim();
    if (ring->numel() % per_layer != 0)
        throw std::logic_error("DeepSeek-V4 raw-ring bank is not a whole number of layer rings");
    const int ring_rows = static_cast<int>(ring->numel() / per_layer);
    return deepSeekV4StateLayout(m, ring_rows - m.window(), config_.max_seq_len);
}

ComputeGraph DeepSeekV4Graph::buildAttentionGraph(
    const LayerWeights &layer, ActivationBuffers &buffers, int layer_idx, int seq_len, int batch_size,
    IKVCache *kv_cache, const int *position_ids, DeviceId device, const std::vector<int> *sequence_lengths,
    const void *position_ids_device, const int32_t *sequence_lengths_device)
{
    (void)sequence_lengths;
    if (batch_size != 1)
        throw std::invalid_argument("DeepSeek-V4 graphs currently admit exactly one request per graph");
    if (device.is_rocm())
        throw std::invalid_argument(deepseek4::kROCmUnimplemented);
    const auto &m = *config_.deepseek_v4;
    const auto &policy = m.layer(layer_idx);
    const int rows = seq_len * batch_size;
    const std::string prefix = "layer" + std::to_string(layer_idx) + "_";
    const auto bindings = layerWeightBindingsForGraph(layer_idx);
    const auto &w = bindings.deepseek_v4;
    const auto layout = stateLayout();
    ComputeGraph graph;

    auto read = addPreAttentionNorm(graph, prefix, buffers, layer.attn_norm, rows, layer_idx, device);
    if (!read)
        throw std::logic_error("DeepSeek-V4 attention requires a residual-trunk read");
    const std::string normalized = read->terminal();
    const auto stage_rows = modelExtensionRows(rows, device);

    const auto *positions_data = device.is_gpu() ? static_cast<const std::int32_t *>(position_ids_device)
                                                 : reinterpret_cast<const std::int32_t *>(position_ids);
    if (!positions_data)
        throw std::invalid_argument("DeepSeek-V4 attention requires row positions on its execution device");
    const Qwen4ExpIntInput positions(Qwen4ExpNativeIntInput{device, positions_data, static_cast<std::size_t>(rows)});

    const auto tensor = [&](BufferId role) -> TensorBase & {
        // Core activation roles are typed fields; DeepSeek-V4 roles are extensions.
        TensorBase *value = role == BufferId::NORMALIZED ? buffers.normalized
                          : role == BufferId::K_PROJ ? buffers.K
                          : role == BufferId::V_PROJ ? buffers.V
                          : role == BufferId::ATTN_PROJ ? buffers.attn_proj
                          : role == BufferId::HIDDEN_STATE ? buffers.current_hidden
                          : buffers.get(buffers.idFor(role));
        if (!value)
            throw std::invalid_argument("DeepSeek-V4 is missing admitted arena buffer " + std::string(bufferIdName(role)));
        return *value;
    };
    const auto f32 = [&](BufferId role, std::size_t element_offset = 0) {
        return Qwen4ExpFloatBinding{tensor(role), buffers.idFor(role), element_offset * sizeof(float)};
    };
    const auto i32 = [&](BufferId role) { return Qwen4ExpIntBinding{tensor(role), buffers.idFor(role)}; };
    const auto gain = [&](const WeightBinding *binding, const char *name) {
        if (!binding || !binding->tensor)
            throw std::invalid_argument(std::string("DeepSeek-V4 layer is missing ") + name);
        return Qwen4ExpImmutableWeightBinding(*binding->tensor);
    };
    const auto gemm = [&](const std::string &node, const WeightBinding *weight, BufferId input, BufferId output,
                          const std::string &dependency) {
        if (!weight || !weight->tensor)
            throw std::invalid_argument("DeepSeek-V4 projection " + node + " lacks its frozen weight");
        GEMMStage::Params params{
            .device_id = device,
            .A = &tensor(input),
            .B = weight->tensor,
            .C = &tensor(output),
            .m = rows,
            .n = static_cast<int>(weight->tensor->shape()[0]),
            .k = static_cast<int>(weight->tensor->shape()[1]),
            .a_buffer_id = buffers.idFor(input),
            .c_buffer_id = buffers.idFor(output),
            .prepared_ref = preparedRefForGraphWeight(weight, device),
            .prepared_store = prepared_weight_store_,
        };
        graph.addNode(node, ComputeStageFactory::createGEMM(params), device);
        graph.addDependency(node, dependency);
        return node;
    };
    using Stage = DeepSeekV4PrimitiveStage;
    const auto prim = [&](const std::string &node, Stage::Operation operation,
                          std::initializer_list<std::string> dependencies) {
        graph.addNode(node, std::make_unique<Stage>(Stage::Params{
            .device_id = device, .mpi_ctx = mpi_ctx_.get(), .rows = stage_rows, .operation = std::move(operation)}),
            device);
        for (const auto &dependency : dependencies) graph.addDependency(node, dependency);
        return node;
    };
    deepseek4::RopePolicy no_rope;
    no_rope.ropeDim = 0;
    const float eps = m.normEps();
    const int hd = m.headDim();

    // ---------------------------------------------------------------- query
    const auto q_a = gemm(prefix + "q_a", w.q_a, BufferId::NORMALIZED, BufferId::DSV4_QR, normalized);
    const auto q_a_norm = prim(prefix + "q_a_norm", Stage::HeadNormRope{1, m.qLoraRank(), true, eps,
        gain(w.q_a_norm, "attn_q_a_norm"), no_rope, false, std::nullopt, f32(BufferId::DSV4_QR)}, {q_a});
    const auto q_b = gemm(prefix + "q_b", w.q_b, BufferId::DSV4_QR, BufferId::DSV4_Q, q_a_norm);
    const auto q_rope = prim(prefix + "q_norm_rope", Stage::HeadNormRope{m.heads(), hd, true, eps, std::nullopt,
        policy.rope, false, positions, f32(BufferId::DSV4_Q)}, {q_b});

    // ---------------------------------------------------------------- raw kv (+ canonical cache append)
    const auto kv = gemm(prefix + "kv", w.kv, BufferId::NORMALIZED, BufferId::K_PROJ, normalized);
    const auto kv_v = gemm(prefix + "kv_v", w.kv, BufferId::NORMALIZED, BufferId::V_PROJ, normalized);
    const auto kv_rope = prim(prefix + "kv_norm_rope", Stage::HeadNormRope{1, hd, true, eps,
        gain(w.kv_norm, "attn_kv_a_norm"), policy.rope, false, positions, f32(BufferId::K_PROJ)}, {kv});
    const auto kv_rope_v = prim(prefix + "kv_norm_rope_v", Stage::HeadNormRope{1, hd, true, eps,
        gain(w.kv_norm, "attn_kv_a_norm"), policy.rope, false, positions, f32(BufferId::V_PROJ)}, {kv_v});
    std::vector<std::string> attention_dependencies{q_rope};
    if (kv_cache)
    {
        const auto execution_policy = resolveAttentionExecutionPolicy(device, true);
        const auto append = addKVCacheAppend(graph, prefix, buffers, layer_idx, seq_len, batch_size, kv_cache,
                                             sequence_lengths_device, device, execution_policy, kv_rope, {kv_rope_v});
        attention_dependencies.push_back(append);
    }
    const auto raw_store = prim(prefix + "raw_store", Stage::RawStore{hd, layout.rawRingRows, positions,
        f32(BufferId::K_PROJ), f32(BufferId::DSV4_RAW_RING, static_cast<std::size_t>(layer_idx) * layout.rawRingRows * hd)},
        {kv_rope});
    attention_dependencies.push_back(raw_store);

    // ---------------------------------------------------------------- compressor + indexer
    std::optional<Qwen4ExpFloatBinding> pool;
    std::optional<Qwen4ExpIntBinding> selected;
    int pool_rows = 1;
    if (policy.ratio != 0)
    {
        const deepseek4::CompressorGeometry geometry{policy.ratio, hd,
            policy.ratio == 4 ? layout.csaRingRows : layout.hcaRingRows,
            policy.ratio == 4 ? layout.csaPoolRows : layout.hcaPoolRows};
        pool_rows = geometry.poolRows;
        const auto ring_offset = layout.compRingElementOffset(m, layer_idx);
        const auto pool_offset = layout.compPoolElementOffset(m, layer_idx);
        const auto c_kv = gemm(prefix + "comp_kv", w.compressor.kv, BufferId::NORMALIZED, BufferId::DSV4_COMP_KV, normalized);
        const auto c_sc = gemm(prefix + "comp_gate", w.compressor.gate, BufferId::NORMALIZED, BufferId::DSV4_COMP_SCORE, normalized);
        const auto c_store = prim(prefix + "comp_store", Stage::CompressorStore{geometry, positions,
            f32(BufferId::DSV4_COMP_KV), f32(BufferId::DSV4_COMP_SCORE), gain(w.compressor.ape, "attn_compressor_ape"),
            f32(BufferId::DSV4_COMP_RING_KV, ring_offset), f32(BufferId::DSV4_COMP_RING_SCORE, ring_offset)}, {c_kv, c_sc});
        const auto c_pool = prim(prefix + "comp_pool", Stage::CompressorPool{geometry, positions,
            f32(BufferId::DSV4_COMP_RING_KV, ring_offset), f32(BufferId::DSV4_COMP_RING_SCORE, ring_offset),
            gain(w.compressor.norm, "attn_compressor_norm"), eps, policy.rope,
            f32(BufferId::DSV4_COMP_POOL, pool_offset)}, {c_store});
        attention_dependencies.push_back(c_pool);
        pool = f32(BufferId::DSV4_COMP_POOL, pool_offset);
        if (policy.ratio == 4)
        {
            const int ihd = m.indexerHeadDim();
            const deepseek4::CompressorGeometry indexer_geometry{4, ihd, layout.csaRingRows, layout.csaPoolRows};
            const auto iring = layout.indexerRingElementOffset(m, layer_idx);
            const auto ipool = layout.indexerPoolElementOffset(m, layer_idx);
            const auto i_q = gemm(prefix + "idx_q", w.indexer_q_b, BufferId::DSV4_QR, BufferId::DSV4_IDX_Q, q_a_norm);
            const auto i_q_rope = prim(prefix + "idx_q_rope", Stage::HeadNormRope{m.indexerHeads(), ihd, false, eps,
                std::nullopt, policy.rope, false, positions, f32(BufferId::DSV4_IDX_Q)}, {i_q});
            const auto i_w = gemm(prefix + "idx_w", w.indexer_proj, BufferId::NORMALIZED, BufferId::DSV4_IDX_W, normalized);
            const auto i_kv = gemm(prefix + "idx_kv", w.indexer_compressor.kv, BufferId::NORMALIZED, BufferId::DSV4_IDX_KV, normalized);
            const auto i_sc = gemm(prefix + "idx_gate", w.indexer_compressor.gate, BufferId::NORMALIZED, BufferId::DSV4_IDX_SCORE, normalized);
            const auto i_store = prim(prefix + "idx_store", Stage::CompressorStore{indexer_geometry, positions,
                f32(BufferId::DSV4_IDX_KV), f32(BufferId::DSV4_IDX_SCORE), gain(w.indexer_compressor.ape, "indexer_compressor_ape"),
                f32(BufferId::DSV4_IDX_RING_KV, iring), f32(BufferId::DSV4_IDX_RING_SCORE, iring)}, {i_kv, i_sc});
            const auto i_pool = prim(prefix + "idx_pool", Stage::CompressorPool{indexer_geometry, positions,
                f32(BufferId::DSV4_IDX_RING_KV, iring), f32(BufferId::DSV4_IDX_RING_SCORE, iring),
                gain(w.indexer_compressor.norm, "indexer_compressor_norm"), eps, policy.rope,
                f32(BufferId::DSV4_IDX_POOL, ipool)}, {i_store});
            const float weight_scale = 1.0f / std::sqrt(static_cast<float>(ihd) * static_cast<float>(m.indexerHeads()));
            const auto topk = prim(prefix + "idx_topk", Stage::IndexerTopK{m.indexerHeads(), ihd, 4, m.indexerTopK(),
                layout.csaPoolRows, weight_scale, positions, f32(BufferId::DSV4_IDX_Q), f32(BufferId::DSV4_IDX_W),
                f32(BufferId::DSV4_IDX_POOL, ipool), f32(BufferId::DSV4_IDX_SCORES), i32(BufferId::DSV4_IDX_SELECTED)},
                {i_q_rope, i_w, i_pool});
            attention_dependencies.push_back(topk);
            selected = i32(BufferId::DSV4_IDX_SELECTED);
        }
    }

    // ---------------------------------------------------------------- sparse attention + output
    const auto access = policy.ratio == 0 ? deepseek4::CompressedAccess::None
                      : policy.ratio == 4 ? deepseek4::CompressedAccess::Selected
                                          : deepseek4::CompressedAccess::All;
    graph.addNode(prefix + "attention", std::make_unique<Stage>(Stage::Params{
        .device_id = device, .mpi_ctx = mpi_ctx_.get(), .rows = stage_rows,
        .operation = Stage::SparseAttention{m.heads(), hd, m.window(), layout.rawRingRows, policy.ratio,
            m.indexerTopK(), pool_rows, access, 1.0f / std::sqrt(static_cast<float>(hd)), policy.rope, positions,
            f32(BufferId::DSV4_Q), f32(BufferId::DSV4_RAW_RING, static_cast<std::size_t>(layer_idx) * layout.rawRingRows * hd),
            pool, selected, gain(w.sinks, "attn_sinks"), f32(BufferId::DSV4_ATTN)}}), device);
    for (const auto &dependency : attention_dependencies) graph.addDependency(prefix + "attention", dependency);

    if (!w.output_a || !w.output_a->tensor)
        throw std::invalid_argument("DeepSeek-V4 layer is missing attn_output_a");
    const auto wo_a = prim(prefix + "wo_a", Stage::GroupedProjection{m.outputGroups(), m.outputLoraRank(),
        m.heads() * hd / m.outputGroups(), DeepSeekV4NativeWeightBinding(*w.output_a->tensor),
        f32(BufferId::DSV4_ATTN), f32(BufferId::DSV4_OA)}, {prefix + "attention"});
    const auto wo_b = gemm(prefix + "wo_b", w.output_b, BufferId::DSV4_OA, BufferId::ATTN_PROJ, wo_a);

    const auto terminal = addBranchBlockWrite(graph, std::move(*read), buffers, wo_b);
    graph.setTerminalNode(terminal);
    return graph;
}
} // namespace llaminar2
