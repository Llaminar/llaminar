/**
 * @file DeepSeekV4Graph.cpp
 * @brief Bind DeepSeek-V4 model operands to participant-local graph stages.
 *
 * Attention dataflow for layer l (rows M, all buffers arena-owned):
 *   NORMALIZED = HCRead(trunk)                                      [M, d]
 *   DSV4_QR    = RMS(NORMALIZED . q_a) * q_a_norm                   [M, q_rank]
 *   DSV4_Q     = RoPE_tail(RMS_head(DSV4_QR . q_b))                 [M, H*hd]
 *   K_PROJ     = QAT_fp8(RoPE_tail(RMS(NORMALIZED . kv) * kv_norm)) [M, hd]
 *   ratio>0:  pool <- QAT-packed gated pooling of completed blocks, reading
 *             this chunk's projections from scratch and earlier positions
 *             from the FP32 compressor state; then state <- last 2*ratio rows
 *   ratio==4: indexer q = QAT_fp4(Hadamard(RoPE(qr . idx_q))), weights and its
 *             own compressor (MXFP4 pool) -> top-k compressed blocks
 *   DSV4_ATTN  = inverse-RoPE(sink softmax attention over window + compressed),
 *                window keys from K_PROJ (this chunk) or the packed raw ring
 *   raw ring  <- packed last `window` rows of K_PROJ (after attention)
 *   DSV4_OA    = grouped wo_a(DSV4_ATTN);  ATTN_PROJ = DSV4_OA . wo_b
 *   trunk      = HCWrite(trunk, ATTN_PROJ)
 * The canonical KV cache owns only the sequence frontier (its DeepSeek-V4
 * layers are model_state layers); on GPU the terminal layer advances it once.
 * Ordering: every consumer depends on its producers' node names. Persistent
 * window/state banks are read before they are overwritten (raw_store after
 * attention, comp_store after comp_pool), and wo_b waits for every bank writer
 * so the next layer cannot reuse the chunk scratch they read.
 */
#include "DeepSeekV4Graph.h"
#include "DeepSeekV4Config.h"
#include "../../execution/compute_stages/stages/deepseek4/DeepSeekV4PrimitiveStage.h"
#include "../../execution/compute_stages/ComputeStageFactory.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../memory/BufferArena.h"
#include "../../backends/ExplicitGPUStream.h"
#include "../../kernels/IHybridKVCache.h"
#include "../../kernels/KernelFactory.h"
#include "../../transfer/TransferEngine.h"
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

void DeepSeekV4Graph::initializeCacheAuxiliaryState(IKVCache &cache, void *publicationStream)
{
    const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
    if (config_.attention_state_owner != AttentionStateOwner::ModelBanks || !hybrid ||
        hybrid->kvLayerCount() != 0 || hybrid->gdnLayerCount() != 0 || cache.n_layers() != config_.n_layers ||
        cache.first_layer_index() != config_.pp_layer_offset || cache.auxiliaryState())
        throw std::invalid_argument("DeepSeek-V4 requires its frontier-only cache of model_state layers");
    const DeviceId device = config_.default_device;
    if (!device.is_gpu())
    {
        if (publicationStream)
            throw std::invalid_argument("DeepSeek-V4 CPU state banks have no GPU publication stream");
        return;
    }
    (void)ExplicitGPUStream(publicationStream);
    const auto kernels = llaminar::v2::kernels::KernelFactory::createDeepSeekV4(device);
    for (const BufferId role : {BufferId::DSV4_RAW_RING, BufferId::DSV4_COMP_RING_KV, BufferId::DSV4_COMP_RING_SCORE,
                                BufferId::DSV4_IDX_RING_KV, BufferId::DSV4_IDX_RING_SCORE})
    {
        auto *bank = buffers_.layer_buffers.get(role);
        if (!bank)
            throw std::invalid_argument(std::string("DeepSeek-V4 is missing admitted state bank ") + bufferIdName(role));
        TransferEngine::allocateDeviceStorage(bank, device);
        kernels->zeroBank(bank->gpu_data_ptr(), bank->size_bytes(), publicationStream);
        TransferEngine::publishDeviceWrite(bank, device, publicationStream);
    }
}

DeepSeekV4StateLayout DeepSeekV4Graph::stateLayout() const
{
    const auto &m = *config_.deepseek_v4;
    auto layout = deepSeekV4StateLayout(m, 1, config_.max_seq_len);
    // The admitted score scratch fixes the bounded indexer pass height.
    const auto *scores = buffers_.layer_buffers.get(BufferId::DSV4_IDX_SCORES);
    if (!scores)
        throw std::invalid_argument("DeepSeek-V4 graph requires its admitted indexer score scratch");
    if (scores->numel() % static_cast<std::size_t>(layout.csaPoolRows) != 0)
        throw std::logic_error("DeepSeek-V4 indexer score scratch is not a whole number of pool rows");
    layout.indexerScoreRows = static_cast<int>(scores->numel() / static_cast<std::size_t>(layout.csaPoolRows));
    return layout;
}

ComputeGraph DeepSeekV4Graph::buildAttentionGraph(
    const LayerWeights &layer, ActivationBuffers &buffers, int layer_idx, int seq_len, int batch_size,
    IKVCache *kv_cache, const int *position_ids, DeviceId device, const std::vector<int> *sequence_lengths,
    const void *position_ids_device, const int32_t *sequence_lengths_device)
{
    (void)sequence_lengths;
    (void)sequence_lengths_device; // Row counts come from modelExtensionRows (one row authority).
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
        gain(w.q_a_norm, "attn_q_a_norm"), no_rope, false, deepseek4::ActivationQuant::None, std::nullopt,
        f32(BufferId::DSV4_QR)}, {q_a});
    const auto q_b = gemm(prefix + "q_b", w.q_b, BufferId::DSV4_QR, BufferId::DSV4_Q, q_a_norm);
    const auto q_rope = prim(prefix + "q_norm_rope", Stage::HeadNormRope{m.heads(), hd, true, eps, std::nullopt,
        policy.rope, false, deepseek4::ActivationQuant::None, positions, f32(BufferId::DSV4_Q)}, {q_b});

    // ---------------------------------------------------------------- raw kv (QAT FP8 nope + BF16 rope)
    const auto kv = gemm(prefix + "kv", w.kv, BufferId::NORMALIZED, BufferId::K_PROJ, normalized);
    const auto kv_rope = prim(prefix + "kv_norm_rope", Stage::HeadNormRope{1, hd, true, eps,
        gain(w.kv_norm, "attn_kv_a_norm"), policy.rope, false, deepseek4::ActivationQuant::FP8NopeBF16Rope, positions,
        f32(BufferId::K_PROJ)}, {kv});
    std::vector<std::string> attention_dependencies{q_rope, kv_rope};
    std::vector<std::string> bank_writers;
    const auto i32at = [&](BufferId role, std::size_t words) {
        return Qwen4ExpIntBinding{tensor(role), buffers.idFor(role), words * sizeof(std::int32_t)};
    };

    // ---------------------------------------------------------------- compressor + indexer
    std::optional<Qwen4ExpIntBinding> pool;
    std::optional<Qwen4ExpIntBinding> selected;
    int pool_rows = 1;
    if (policy.ratio != 0)
    {
        const deepseek4::CompressorGeometry geometry{policy.ratio, hd,
            policy.ratio == 4 ? layout.csaPoolRows : layout.hcaPoolRows, deepseek4::CompressedStream::Attention,
            policy.rope.ropeDim};
        pool_rows = geometry.poolRows;
        const auto state_offset = layout.compStateElementOffset(m, layer_idx);
        const auto pool_offset = layout.compPoolWordOffset(m, layer_idx);
        const auto c_kv = gemm(prefix + "comp_kv", w.compressor.kv, BufferId::NORMALIZED, BufferId::DSV4_COMP_KV, normalized);
        const auto c_sc = gemm(prefix + "comp_gate", w.compressor.gate, BufferId::NORMALIZED, BufferId::DSV4_COMP_SCORE, normalized);
        const auto ape = gain(w.compressor.ape, "attn_compressor_ape");
        const auto c_pool = prim(prefix + "comp_pool", Stage::CompressorPool{geometry, positions,
            f32(BufferId::DSV4_COMP_KV), f32(BufferId::DSV4_COMP_SCORE), ape,
            f32(BufferId::DSV4_COMP_RING_KV, state_offset), f32(BufferId::DSV4_COMP_RING_SCORE, state_offset),
            gain(w.compressor.norm, "attn_compressor_norm"), eps, policy.rope,
            i32at(BufferId::DSV4_COMP_POOL, pool_offset)}, {c_kv, c_sc});
        // The carried state is overwritten only after the pool has read it.
        const auto c_store = prim(prefix + "comp_store", Stage::CompressorStore{geometry, positions,
            f32(BufferId::DSV4_COMP_KV), f32(BufferId::DSV4_COMP_SCORE), ape,
            f32(BufferId::DSV4_COMP_RING_KV, state_offset), f32(BufferId::DSV4_COMP_RING_SCORE, state_offset)},
            {c_pool});
        attention_dependencies.push_back(c_pool);
        bank_writers.push_back(c_store);
        pool = i32at(BufferId::DSV4_COMP_POOL, pool_offset);
        if (policy.ratio == 4)
        {
            const int ihd = m.indexerHeadDim();
            const deepseek4::CompressorGeometry indexer_geometry{4, ihd, layout.csaPoolRows,
                deepseek4::CompressedStream::Indexer, 0};
            const auto istate = layout.indexerStateElementOffset(m, layer_idx);
            const auto ipool = layout.indexerPoolWordOffset(m, layer_idx);
            const auto i_q = gemm(prefix + "idx_q", w.indexer_q_b, BufferId::DSV4_QR, BufferId::DSV4_IDX_Q, q_a_norm);
            const auto i_q_rope = prim(prefix + "idx_q_rope", Stage::HeadNormRope{m.indexerHeads(), ihd, false, eps,
                std::nullopt, policy.rope, false, deepseek4::ActivationQuant::HadamardFP4, positions,
                f32(BufferId::DSV4_IDX_Q)}, {i_q});
            const auto i_w = gemm(prefix + "idx_w", w.indexer_proj, BufferId::NORMALIZED, BufferId::DSV4_IDX_W, normalized);
            const auto i_kv = gemm(prefix + "idx_kv", w.indexer_compressor.kv, BufferId::NORMALIZED, BufferId::DSV4_IDX_KV, normalized);
            const auto i_sc = gemm(prefix + "idx_gate", w.indexer_compressor.gate, BufferId::NORMALIZED, BufferId::DSV4_IDX_SCORE, normalized);
            const auto i_ape = gain(w.indexer_compressor.ape, "indexer_compressor_ape");
            const auto i_pool = prim(prefix + "idx_pool", Stage::CompressorPool{indexer_geometry, positions,
                f32(BufferId::DSV4_IDX_KV), f32(BufferId::DSV4_IDX_SCORE), i_ape,
                f32(BufferId::DSV4_IDX_RING_KV, istate), f32(BufferId::DSV4_IDX_RING_SCORE, istate),
                gain(w.indexer_compressor.norm, "indexer_compressor_norm"), eps, policy.rope,
                i32at(BufferId::DSV4_IDX_POOL, ipool)}, {i_kv, i_sc});
            const auto i_store = prim(prefix + "idx_store", Stage::CompressorStore{indexer_geometry, positions,
                f32(BufferId::DSV4_IDX_KV), f32(BufferId::DSV4_IDX_SCORE), i_ape,
                f32(BufferId::DSV4_IDX_RING_KV, istate), f32(BufferId::DSV4_IDX_RING_SCORE, istate)}, {i_pool});
            bank_writers.push_back(i_store);
            const float weight_scale = 1.0f / std::sqrt(static_cast<float>(ihd) * static_cast<float>(m.indexerHeads()));
            std::optional<Qwen4ExpFloatBinding> scores;
            if (device.is_gpu()) scores = f32(BufferId::DSV4_IDX_SCORES);
            const auto topk = prim(prefix + "idx_topk", Stage::IndexerTopK{m.indexerHeads(), layout.indexer, 4,
                m.indexerTopK(), layout.csaPoolRows, layout.indexerScoreRows, weight_scale, positions,
                f32(BufferId::DSV4_IDX_Q), f32(BufferId::DSV4_IDX_W), i32at(BufferId::DSV4_IDX_POOL, ipool), scores,
                i32(BufferId::DSV4_IDX_SELECTED)}, {i_q_rope, i_w, i_pool});
            attention_dependencies.push_back(topk);
            selected = i32(BufferId::DSV4_IDX_SELECTED);
        }
    }

    // ---------------------------------------------------------------- sparse attention + output
    const auto access = policy.ratio == 0 ? deepseek4::CompressedAccess::None
                      : policy.ratio == 4 ? deepseek4::CompressedAccess::Selected
                                          : deepseek4::CompressedAccess::All;
    const auto raw_ring = i32at(BufferId::DSV4_RAW_RING, layout.rawRingWordOffset(layer_idx));
    graph.addNode(prefix + "attention", std::make_unique<Stage>(Stage::Params{
        .device_id = device, .mpi_ctx = mpi_ctx_.get(), .rows = stage_rows,
        .operation = Stage::SparseAttention{m.heads(), layout.kv, m.window(), policy.ratio, m.indexerTopK(), pool_rows,
            access, 1.0f / std::sqrt(static_cast<float>(hd)), policy.rope, positions, f32(BufferId::DSV4_Q),
            f32(BufferId::K_PROJ), raw_ring, pool, selected, gain(w.sinks, "attn_sinks"), f32(BufferId::DSV4_ATTN)}}),
        device);
    for (const auto &dependency : attention_dependencies) graph.addDependency(prefix + "attention", dependency);
    // The window ring is overwritten only after attention read the previous window.
    bank_writers.push_back(prim(prefix + "raw_store", Stage::RawStore{layout.kv, m.window(), positions,
        f32(BufferId::K_PROJ), raw_ring}, {prefix + "attention"}));

    // The cache owns no DeepSeek-V4 payload; the terminal layer publishes the
    // forward's real rows to its device frontier exactly once (GPU). CPU
    // execution keeps positions in the orchestrator's host state.
    const int terminal_layer = config_.pp_layer_offset + config_.n_layers - 1;
    if (kv_cache && device.is_gpu() && layer_idx == terminal_layer)
        bank_writers.push_back(prim(prefix + "sequence_frontier", Stage::AdvanceSequenceFrontier{kv_cache, layer_idx,
            positions}, {prefix + "attention"}));

    if (!w.output_a || !w.output_a->tensor)
        throw std::invalid_argument("DeepSeek-V4 layer is missing attn_output_a");
    const auto wo_a = prim(prefix + "wo_a", Stage::GroupedProjection{m.outputGroups(), m.outputLoraRank(),
        m.heads() * hd / m.outputGroups(), DeepSeekV4NativeWeightBinding(*w.output_a->tensor),
        f32(BufferId::DSV4_ATTN), f32(BufferId::DSV4_OA)}, {prefix + "attention"});
    const auto wo_b = gemm(prefix + "wo_b", w.output_b, BufferId::DSV4_OA, BufferId::ATTN_PROJ, wo_a);
    // Bank writers read this layer's K_PROJ / compressor scratch: the next
    // layer (ordered after this terminal) must not overwrite it earlier.
    for (const auto &writer : bank_writers) graph.addDependency(wo_b, writer);

    const auto terminal = addBranchBlockWrite(graph, std::move(*read), buffers, wo_b);
    graph.setTerminalNode(terminal);
    return graph;
}
} // namespace llaminar2
