/**
 * @file DeepSeekV4BufferLayout.cpp
 * @brief Overflow-checked DeepSeek-V4 arena declarations and formulas.
 *
 * Bank layout (element offsets are in units of FP32):
 *   DSV4_RAW_RING          [layers, rawRingRows, head_dim]
 *   DSV4_COMP_RING_KV/SCORE  concatenation over compressed layers of
 *                          [ringRows(ratio), projectionWidth(ratio)]
 *   DSV4_COMP_POOL         concatenation over compressed layers of [poolRows(ratio), head_dim]
 *   DSV4_IDX_RING_KV/SCORE concatenation over ratio-4 layers of [csaRingRows, 2*index_head_dim]
 *   DSV4_IDX_POOL          concatenation over ratio-4 layers of [csaPoolRows, index_head_dim]
 * Banks are flattened 1-D roles so mixed-ratio layers share one allocation.
 */
#include "DeepSeekV4BufferLayout.h"
#include "DeepSeekV4Config.h"
#include "../GraphTypes.h"
#include "../../execution/local_execution/graph/GraphResolver.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
namespace
{
std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("DeepSeek-V4 arena geometry exceeds addressable capacity");
    return a * b;
}

const std::vector<DeepSeekV4BufferDeclaration> declarations{
    {BufferId::QFN_EMBEDDING_ROWS, {"qfn_embedding_rows", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_HC_MIXES, {"dsv4_hc_mixes", {"seq_len", "dsv4_mix_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_HC_POST, {"dsv4_hc_post", {"seq_len", "dsv4_hc"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_HC_COMB, {"dsv4_hc_comb", {"seq_len", "dsv4_hc_square"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_QR, {"dsv4_qr", {"seq_len", "dsv4_q_rank"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_Q, {"dsv4_q", {"seq_len", "dsv4_q_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_KV, {"dsv4_kv", {"seq_len", "dsv4_head_dim"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_COMP_KV, {"dsv4_comp_kv", {"seq_len", "dsv4_comp_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_COMP_SCORE, {"dsv4_comp_score", {"seq_len", "dsv4_comp_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_Q, {"dsv4_idx_q", {"seq_len", "dsv4_idx_q_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_W, {"dsv4_idx_w", {"seq_len", "dsv4_idx_heads"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_KV, {"dsv4_idx_kv", {"seq_len", "dsv4_idx_comp_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_SCORE, {"dsv4_idx_score", {"seq_len", "dsv4_idx_comp_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_SCORES, {"dsv4_idx_scores", {"seq_len", "dsv4_csa_pool_rows"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_SELECTED, {"dsv4_idx_selected", {"seq_len", "dsv4_idx_topk"}, "int32", BufferSemantic::Scratch}},
    {BufferId::DSV4_ATTN, {"dsv4_attn", {"seq_len", "dsv4_q_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_OA, {"dsv4_oa", {"seq_len", "dsv4_oa_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_ROUTER_LOGITS, {"dsv4_router_logits", {"seq_len", "dsv4_experts"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_ROUTE_IDS, {"dsv4_route_ids", {"seq_len", "dsv4_experts_used"}, "int32", BufferSemantic::Scratch}},
    {BufferId::DSV4_ROUTE_WEIGHTS, {"dsv4_route_weights", {"seq_len", "dsv4_experts_used"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_RAW_RING, {"dsv4_raw_ring", {"dsv4_raw_ring_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_COMP_RING_KV, {"dsv4_comp_ring_kv", {"dsv4_comp_ring_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_COMP_RING_SCORE, {"dsv4_comp_ring_score", {"dsv4_comp_ring_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_COMP_POOL, {"dsv4_comp_pool", {"dsv4_comp_pool_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_IDX_RING_KV, {"dsv4_idx_ring_kv", {"dsv4_idx_ring_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_IDX_RING_SCORE, {"dsv4_idx_ring_score", {"dsv4_idx_ring_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_IDX_POOL, {"dsv4_idx_pool", {"dsv4_idx_pool_elements"}, "fp32", BufferSemantic::InOut}},
};

int ceilDiv(int a, int b) { return (a + b - 1) / b; }
} // namespace

DeepSeekV4StateLayout deepSeekV4StateLayout(const DeepSeekV4Config &model, int maxRows, int contextTokens)
{
    if (maxRows <= 0 || contextTokens <= 0)
        throw std::invalid_argument("DeepSeek-V4 state layout requires positive row and context envelopes");
    DeepSeekV4StateLayout layout;
    layout.rawRingRows = maxRows + model.window();
    layout.csaRingRows = maxRows + 2 * 4;
    layout.hcaRingRows = maxRows + 2 * 128;
    layout.csaPoolRows = ceilDiv(contextTokens, 4);
    layout.hcaPoolRows = ceilDiv(contextTokens, 128);
    return layout;
}

std::size_t DeepSeekV4StateLayout::compRingElementOffset(const DeepSeekV4Config &model, int layer) const
{
    std::size_t offset = 0;
    for (int il = 0; il < layer; ++il)
    {
        const int ratio = model.layer(il).ratio;
        if (ratio == 4) offset += product(csaRingRows, 2ull * model.headDim());
        else if (ratio == 128) offset += product(hcaRingRows, model.headDim());
    }
    return offset;
}

std::size_t DeepSeekV4StateLayout::compPoolElementOffset(const DeepSeekV4Config &model, int layer) const
{
    std::size_t offset = 0;
    for (int il = 0; il < layer; ++il)
    {
        const int ratio = model.layer(il).ratio;
        if (ratio == 4) offset += product(csaPoolRows, model.headDim());
        else if (ratio == 128) offset += product(hcaPoolRows, model.headDim());
    }
    return offset;
}

std::size_t DeepSeekV4StateLayout::indexerRingElementOffset(const DeepSeekV4Config &model, int layer) const
{
    std::size_t offset = 0;
    for (int il = 0; il < layer; ++il)
        if (model.layer(il).ratio == 4) offset += product(csaRingRows, 2ull * model.indexerHeadDim());
    return offset;
}

std::size_t DeepSeekV4StateLayout::indexerPoolElementOffset(const DeepSeekV4Config &model, int layer) const
{
    std::size_t offset = 0;
    for (int il = 0; il < layer; ++il)
        if (model.layer(il).ratio == 4) offset += product(csaPoolRows, model.indexerHeadDim());
    return offset;
}

std::span<const DeepSeekV4BufferDeclaration> deepSeekV4BufferDeclarations() { return declarations; }

DeepSeekV4MemoryGeometry DeepSeekV4MemoryGeometry::fromConfig(const DeepSeekV4Config &m)
{
    const auto u = [](int v) { return static_cast<uint32_t>(v); };
    return {u(m.layers()), u(m.hidden()), u(m.hcMult()), u(m.heads()), u(m.headDim()), u(m.qLoraRank()),
            u(m.outputGroups()), u(m.outputLoraRank()), u(m.window()), u(m.indexerHeads()), u(m.indexerHeadDim()),
            u(m.indexerTopK()), u(m.experts()), u(m.expertsUsed()),
            u(m.countLayers(DeepSeekV4AttentionKind::Compressed)), u(m.countLayers(DeepSeekV4AttentionKind::Heavy))};
}

std::size_t deepSeekV4ArenaExtensionBytes(const DeepSeekV4MemoryGeometry &g, int rows, int contextTokens)
{
    if (rows <= 0 || contextTokens <= 0 || !g.layers || !g.hidden || !g.hc)
        throw std::invalid_argument("DeepSeek-V4 arena admission requires positive rows, context and geometry");
    const std::size_t r = static_cast<std::size_t>(rows);
    const std::size_t csa_pool = (static_cast<std::size_t>(contextTokens) + 3) / 4;
    const std::size_t hca_pool = (static_cast<std::size_t>(contextTokens) + 127) / 128;
    const std::size_t csa_ring = r + 8, hca_ring = r + 256, raw_ring = r + g.window;
    const std::size_t mix = (2ull + g.hc) * g.hc;
    // Per-row scratch widths, in declaration order of deepSeekV4BufferDeclarations().
    const std::size_t per_row = g.hidden /*embedding rows*/ + mix + g.hc + std::size_t{g.hc} * g.hc + g.q_rank +
        std::size_t{g.heads} * g.head_dim /*q*/ + g.head_dim + 2ull * g.head_dim * 2 /*comp kv/score*/ +
        std::size_t{g.index_heads} * g.index_head_dim + g.index_heads + 2ull * g.index_head_dim * 2 +
        csa_pool /*indexer scores*/ + g.index_topk + std::size_t{g.heads} * g.head_dim /*attn*/ +
        std::size_t{g.o_groups} * g.o_rank + g.experts + 2ull * g.experts_used;
    std::size_t elements = product(r, per_row);
    elements += product(product(g.layers, raw_ring), g.head_dim);
    elements += 2 * std::max<std::size_t>(1, product(g.csa_layers, product(csa_ring, 2ull * g.head_dim)) +
                                              product(g.hca_layers, product(hca_ring, g.head_dim)));
    elements += std::max<std::size_t>(1, product(g.csa_layers, product(csa_pool, g.head_dim)) +
                                         product(g.hca_layers, product(hca_pool, g.head_dim)));
    elements += 2 * std::max<std::size_t>(1, product(g.csa_layers, product(csa_ring, 2ull * g.index_head_dim)));
    elements += std::max<std::size_t>(1, product(g.csa_layers, product(csa_pool, g.index_head_dim)));
    // HIDDEN_STATE and RESIDUAL widen from hidden to hc * hidden.
    elements += 2 * product(r, product(g.hc - 1, g.hidden));
    return product(elements, sizeof(float));
}

void applyDeepSeekV4BufferLayout(GraphSchema &schema)
{
    const auto widen = [](BufferSpec &spec) {
        if (spec.name == "hidden" || spec.name == "residual") spec.shape.back() = "dsv4_trunk";
    };
    for (auto &spec : schema.model_buffers) widen(spec);
    for (auto &spec : schema.layer_buffers) widen(spec);
    // DeepSeek-V4 has no Gated DeltaNet layers: drop the inherited GDN scratch
    // roles, whose geometry formulas resolve to zero for this architecture.
    const auto gdn = [](const BufferSpec &spec) { return spec.name.rfind("gdn_", 0) == 0; };
    schema.layer_buffers.erase(std::remove_if(schema.layer_buffers.begin(), schema.layer_buffers.end(), gdn),
                               schema.layer_buffers.end());
    for (const auto &entry : declarations)
    {
        const auto duplicate = [&entry](const BufferSpec &spec) { return spec.name == entry.spec.name; };
        if (std::any_of(schema.layer_buffers.begin(), schema.layer_buffers.end(), duplicate) ||
            std::any_of(schema.model_buffers.begin(), schema.model_buffers.end(), duplicate))
            throw std::invalid_argument("DeepSeek-V4 buffer layout was applied twice: " + entry.spec.name);
        schema.layer_buffers.push_back(entry.spec);
    }
}

void applyDeepSeekV4BufferGeometry(GraphResolverConfig &resolver, const DeepSeekV4MemoryGeometry &g,
                                   int rows, int contextTokens)
{
    if (rows <= 0 || contextTokens <= 0 || !g.layers || !g.hidden || !g.hc)
        throw std::invalid_argument("DeepSeek-V4 resolver requires positive rows, context and geometry");
    const std::size_t r = static_cast<std::size_t>(rows);
    const std::size_t csa_pool = (static_cast<std::size_t>(contextTokens) + 3) / 4;
    const std::size_t hca_pool = (static_cast<std::size_t>(contextTokens) + 127) / 128;
    const std::size_t csa_ring = r + 8, hca_ring = r + 256, raw_ring = r + g.window;
    auto &f = resolver.custom_formulas;
    f["dsv4_trunk"] = product(g.hc, g.hidden);
    f["dsv4_hc"] = g.hc;
    f["dsv4_hc_square"] = std::size_t{g.hc} * g.hc;
    f["dsv4_mix_width"] = (2ull + g.hc) * g.hc;
    f["dsv4_q_rank"] = g.q_rank;
    f["dsv4_q_width"] = product(g.heads, g.head_dim);
    f["dsv4_head_dim"] = g.head_dim;
    f["dsv4_comp_width"] = 2ull * g.head_dim;
    f["dsv4_idx_q_width"] = product(g.index_heads, g.index_head_dim);
    f["dsv4_idx_heads"] = g.index_heads;
    f["dsv4_idx_comp_width"] = 2ull * g.index_head_dim;
    f["dsv4_idx_topk"] = g.index_topk;
    f["dsv4_csa_pool_rows"] = csa_pool;
    f["dsv4_oa_width"] = product(g.o_groups, g.o_rank);
    f["dsv4_experts"] = g.experts;
    f["dsv4_experts_used"] = g.experts_used;
    f["dsv4_raw_ring_elements"] = product(product(g.layers, raw_ring), g.head_dim);
    f["dsv4_comp_ring_elements"] = std::max<std::size_t>(1,
        product(g.csa_layers, product(csa_ring, 2ull * g.head_dim)) + product(g.hca_layers, product(hca_ring, g.head_dim)));
    f["dsv4_comp_pool_elements"] = std::max<std::size_t>(1,
        product(g.csa_layers, product(csa_pool, g.head_dim)) + product(g.hca_layers, product(hca_pool, g.head_dim)));
    f["dsv4_idx_ring_elements"] = std::max<std::size_t>(1, product(g.csa_layers, product(csa_ring, 2ull * g.index_head_dim)));
    f["dsv4_idx_pool_elements"] = std::max<std::size_t>(1, product(g.csa_layers, product(csa_pool, g.index_head_dim)));
    for (const auto &entry : declarations)
    {
        const auto [found, inserted] = resolver.buffer_name_to_id.emplace(entry.spec.name, entry.id);
        if (!inserted && found->second != entry.id)
            throw std::invalid_argument("DeepSeek-V4 buffer name is already bound to another identity: " + entry.spec.name);
    }
}
} // namespace llaminar2
