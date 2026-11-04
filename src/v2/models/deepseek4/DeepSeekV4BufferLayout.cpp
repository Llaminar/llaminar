/**
 * @file DeepSeekV4BufferLayout.cpp
 * @brief Overflow-checked DeepSeek-V4 arena declarations, formulas and state offsets.
 *
 * Every size is produced once, by deepSeekV4Formulas(): the resolver receives
 * those formulas for allocation and deepSeekV4ArenaExtensionBytes() resolves
 * the same declarations through them for admission. Per-layer bank regions are
 * concatenated in model layer order; inside a region the sequence axis is
 * outermost, then rows, then the row payload (FP32 elements or packed words).
 */
#include "DeepSeekV4BufferLayout.h"
#include "DeepSeekV4Config.h"
#include "../GraphTypes.h"
#include "../../execution/local_execution/graph/GraphResolver.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("DeepSeek-V4 arena geometry exceeds addressable capacity");
    return a + b;
}

const std::vector<DeepSeekV4BufferDeclaration> declarations{
    {BufferId::QFN_EMBEDDING_ROWS, {"qfn_embedding_rows", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_HC_MIXES, {"dsv4_hc_mixes", {"seq_len", "dsv4_mix_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_HC_POST, {"dsv4_hc_post", {"seq_len", "dsv4_hc"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_HC_COMB, {"dsv4_hc_comb", {"seq_len", "dsv4_hc_square"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_QR, {"dsv4_qr", {"seq_len", "dsv4_q_rank"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_Q, {"dsv4_q", {"seq_len", "dsv4_q_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_COMP_KV, {"dsv4_comp_kv", {"seq_len", "dsv4_comp_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_COMP_SCORE, {"dsv4_comp_score", {"seq_len", "dsv4_comp_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_Q, {"dsv4_idx_q", {"seq_len", "dsv4_idx_q_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_W, {"dsv4_idx_w", {"seq_len", "dsv4_idx_heads"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_KV, {"dsv4_idx_kv", {"seq_len", "dsv4_idx_comp_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_SCORE, {"dsv4_idx_score", {"seq_len", "dsv4_idx_comp_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_SCORES, {"dsv4_idx_scores", {"dsv4_idx_score_rows", "dsv4_csa_pool_rows"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_IDX_SELECTED, {"dsv4_idx_selected", {"seq_len", "dsv4_idx_topk"}, "int32", BufferSemantic::Scratch}},
    {BufferId::DSV4_ATTN, {"dsv4_attn", {"seq_len", "dsv4_q_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_OA, {"dsv4_oa", {"seq_len", "dsv4_oa_width"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_ROUTER_LOGITS, {"dsv4_router_logits", {"seq_len", "dsv4_experts"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_ROUTE_IDS, {"dsv4_route_ids", {"seq_len", "dsv4_experts_used"}, "int32", BufferSemantic::Scratch}},
    {BufferId::DSV4_ROUTE_WEIGHTS, {"dsv4_route_weights", {"seq_len", "dsv4_experts_used"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::DSV4_RAW_RING, {"dsv4_raw_ring", {"dsv4_raw_ring_words"}, "int32", BufferSemantic::InOut}},
    {BufferId::DSV4_COMP_RING_KV, {"dsv4_comp_ring_kv", {"dsv4_comp_state_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_COMP_RING_SCORE, {"dsv4_comp_ring_score", {"dsv4_comp_state_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_COMP_POOL, {"dsv4_comp_pool", {"dsv4_comp_pool_words"}, "int32", BufferSemantic::InOut}},
    {BufferId::DSV4_IDX_RING_KV, {"dsv4_idx_ring_kv", {"dsv4_idx_state_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_IDX_RING_SCORE, {"dsv4_idx_ring_score", {"dsv4_idx_state_elements"}, "fp32", BufferSemantic::InOut}},
    {BufferId::DSV4_IDX_POOL, {"dsv4_idx_pool", {"dsv4_idx_pool_words"}, "int32", BufferSemantic::InOut}},
};

/** @brief Sequence slots of every persistent bank (one request per graph today). */
constexpr std::size_t kSequences = 1;

std::size_t ceilDivSize(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

void requireGeometry(const DeepSeekV4MemoryGeometry &g, int rows, int contextTokens)
{
    if (rows <= 0 || contextTokens <= 0 || !g.layers || !g.hidden || !g.hc || !g.window)
        throw std::invalid_argument("DeepSeek-V4 arena admission requires positive rows, context and geometry");
    if (!g.kvFormat().valid() || !g.indexerFormat().valid())
        throw std::invalid_argument("DeepSeek-V4 packed KV record geometry is not representable");
}

/**
 * @brief The one formula table: every custom dimension of every declaration.
 * @details "seq_len" is the admitted row count and is resolved by the caller.
 */
std::vector<std::pair<std::string, std::size_t>> deepSeekV4Formulas(const DeepSeekV4MemoryGeometry &g, int rows,
                                                                    int contextTokens)
{
    requireGeometry(g, rows, contextTokens);
    const std::size_t context = static_cast<std::size_t>(contextTokens);
    const std::size_t csa_pool = ceilDivSize(context, 4), hca_pool = ceilDivSize(context, 128);
    const std::size_t kv_words = static_cast<std::size_t>(g.kvFormat().entryWords());
    const std::size_t idx_words = static_cast<std::size_t>(g.indexerFormat().entryWords());
    // Carried compressor state is 2 * ratio rows of the projection width.
    const std::size_t csa_state = product(2 * 4, 2ull * g.head_dim), hca_state = product(2 * 128, g.head_dim);
    const std::size_t idx_state = product(2 * 4, 2ull * g.index_head_dim);
    const std::size_t per_seq = kSequences;
    return {
        {"dsv4_trunk", product(g.hc, g.hidden)},
        {"dsv4_hc", g.hc},
        {"dsv4_hc_square", std::size_t{g.hc} * g.hc},
        {"dsv4_mix_width", (2ull + g.hc) * g.hc},
        {"dsv4_q_rank", g.q_rank},
        {"dsv4_q_width", product(g.heads, g.head_dim)},
        {"dsv4_comp_width", 2ull * g.head_dim},
        {"dsv4_idx_q_width", product(g.index_heads, g.index_head_dim)},
        {"dsv4_idx_heads", g.index_heads},
        {"dsv4_idx_comp_width", 2ull * g.index_head_dim},
        {"dsv4_idx_topk", g.index_topk},
        {"dsv4_idx_score_rows", static_cast<std::size_t>(deepSeekV4IndexerScoreRows(rows, contextTokens))},
        {"dsv4_csa_pool_rows", csa_pool},
        {"dsv4_oa_width", product(g.o_groups, g.o_rank)},
        {"dsv4_experts", g.experts},
        {"dsv4_experts_used", g.experts_used},
        {"dsv4_raw_ring_words", product(product(product(g.layers, per_seq), g.window), kv_words)},
        {"dsv4_comp_state_elements", std::max<std::size_t>(1,
            sum(product(product(g.csa_layers, per_seq), csa_state), product(product(g.hca_layers, per_seq), hca_state)))},
        {"dsv4_comp_pool_words", std::max<std::size_t>(1,
            sum(product(product(g.csa_layers, per_seq), product(csa_pool, kv_words)),
                product(product(g.hca_layers, per_seq), product(hca_pool, kv_words))))},
        {"dsv4_idx_state_elements", std::max<std::size_t>(1, product(product(g.csa_layers, per_seq), idx_state))},
        {"dsv4_idx_pool_words", std::max<std::size_t>(1,
            product(product(g.csa_layers, per_seq), product(csa_pool, idx_words)))},
    };
}

int ceilDiv(int a, int b) { return (a + b - 1) / b; }
} // namespace

int deepSeekV4IndexerScoreRows(int rows, int contextTokens)
{
    if (rows <= 0 || contextTokens <= 0)
        throw std::invalid_argument("DeepSeek-V4 indexer scratch requires positive rows and context");
    const std::size_t row_bytes = ceilDivSize(static_cast<std::size_t>(contextTokens), 4) * sizeof(float);
    const std::size_t budget_rows = std::max<std::size_t>(1, kDeepSeekV4IndexerScoreBudgetBytes / row_bytes);
    return static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(rows), budget_rows));
}

DeepSeekV4StateLayout deepSeekV4StateLayout(const DeepSeekV4Config &model, int maxRows, int contextTokens)
{
    if (maxRows <= 0 || contextTokens <= 0)
        throw std::invalid_argument("DeepSeek-V4 state layout requires positive row and context envelopes");
    DeepSeekV4StateLayout layout;
    layout.sequences = static_cast<int>(kSequences);
    layout.window = model.window();
    layout.csaPoolRows = ceilDiv(contextTokens, 4);
    layout.hcaPoolRows = ceilDiv(contextTokens, 128);
    layout.indexerScoreRows = deepSeekV4IndexerScoreRows(maxRows, contextTokens);
    layout.kv = {model.headDim(), model.ropeDim()};
    layout.indexer = {model.indexerHeadDim()};
    if (!layout.kv.valid() || !layout.indexer.valid() || layout.window <= 0)
        throw std::invalid_argument("DeepSeek-V4 packed KV record geometry is not representable");
    return layout;
}

std::size_t DeepSeekV4StateLayout::rawRingWordOffset(int layer, int sequence) const
{
    const std::size_t region = product(product(static_cast<std::size_t>(sequences), window), kv.entryWords());
    return sum(product(static_cast<std::size_t>(layer), region),
               product(product(static_cast<std::size_t>(sequence), window), kv.entryWords()));
}

std::size_t DeepSeekV4StateLayout::compStateElementOffset(const DeepSeekV4Config &model, int layer, int sequence) const
{
    const auto region = [&](int ratio) {
        return product(2ull * ratio, ratio == 4 ? 2ull * model.headDim() : static_cast<std::size_t>(model.headDim()));
    };
    std::size_t offset = 0;
    for (int il = 0; il < layer; ++il)
        if (const int ratio = model.layer(il).ratio) offset = sum(offset, product(sequences, region(ratio)));
    return sum(offset, product(static_cast<std::size_t>(sequence), region(model.layer(layer).ratio)));
}

std::size_t DeepSeekV4StateLayout::compPoolWordOffset(const DeepSeekV4Config &model, int layer, int sequence) const
{
    const auto region = [&](int ratio) {
        return product(static_cast<std::size_t>(ratio == 4 ? csaPoolRows : hcaPoolRows), kv.entryWords());
    };
    std::size_t offset = 0;
    for (int il = 0; il < layer; ++il)
        if (const int ratio = model.layer(il).ratio) offset = sum(offset, product(sequences, region(ratio)));
    return sum(offset, product(static_cast<std::size_t>(sequence), region(model.layer(layer).ratio)));
}

std::size_t DeepSeekV4StateLayout::indexerStateElementOffset(const DeepSeekV4Config &model, int layer,
                                                             int sequence) const
{
    const std::size_t region = product(8, 2ull * model.indexerHeadDim());
    std::size_t offset = 0;
    for (int il = 0; il < layer; ++il)
        if (model.layer(il).ratio == 4) offset = sum(offset, product(sequences, region));
    return sum(offset, product(static_cast<std::size_t>(sequence), region));
}

std::size_t DeepSeekV4StateLayout::indexerPoolWordOffset(const DeepSeekV4Config &model, int layer, int sequence) const
{
    const std::size_t region = product(static_cast<std::size_t>(csaPoolRows), indexer.entryWords());
    std::size_t offset = 0;
    for (int il = 0; il < layer; ++il)
        if (model.layer(il).ratio == 4) offset = sum(offset, product(sequences, region));
    return sum(offset, product(static_cast<std::size_t>(sequence), region));
}

std::span<const DeepSeekV4BufferDeclaration> deepSeekV4BufferDeclarations() { return declarations; }

DeepSeekV4MemoryGeometry DeepSeekV4MemoryGeometry::fromConfig(const DeepSeekV4Config &m)
{
    const auto u = [](int v) { return static_cast<uint32_t>(v); };
    return {u(m.layers()), u(m.hidden()), u(m.hcMult()), u(m.heads()), u(m.headDim()), u(m.qLoraRank()),
            u(m.outputGroups()), u(m.outputLoraRank()), u(m.window()), u(m.indexerHeads()), u(m.indexerHeadDim()),
            u(m.indexerTopK()), u(m.experts()), u(m.expertsUsed()),
            u(m.countLayers(DeepSeekV4AttentionKind::Compressed)), u(m.countLayers(DeepSeekV4AttentionKind::Heavy)),
            u(m.ropeDim())};
}

std::size_t deepSeekV4ArenaExtensionBytes(const DeepSeekV4MemoryGeometry &g, int rows, int contextTokens)
{
    const auto formulas = deepSeekV4Formulas(g, rows, contextTokens);
    const auto dimension = [&](const std::string &name) -> std::size_t {
        if (name == "seq_len") return static_cast<std::size_t>(rows);
        if (name == "d_model") return g.hidden;
        for (const auto &[key, value] : formulas)
            if (key == name) return value;
        throw std::logic_error("DeepSeek-V4 declaration uses an unpublished dimension: " + name);
    };
    std::size_t bytes = 0;
    for (const auto &entry : declarations)
    {
        if (entry.spec.dtype != "fp32" && entry.spec.dtype != "int32")
            throw std::logic_error("DeepSeek-V4 declarations are 4-byte FP32/INT32 roots");
        std::size_t elements = 1;
        for (const auto &axis : entry.spec.shape) elements = product(elements, dimension(axis));
        bytes = sum(bytes, product(elements, 4));
    }
    // HIDDEN_STATE and RESIDUAL widen from hidden to hc * hidden.
    return sum(bytes, product(2 * static_cast<std::size_t>(rows), product(g.hc - 1, g.hidden) * sizeof(float)));
}

std::size_t deepSeekV4PersistentPoolBytes(const DeepSeekV4MemoryGeometry &g, int contextTokens)
{
    std::size_t bytes = 0;
    for (const auto &[key, value] : deepSeekV4Formulas(g, 1, contextTokens))
        if (key == "dsv4_comp_pool_words" || key == "dsv4_idx_pool_words") bytes = sum(bytes, product(value, 4));
    return bytes;
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
    for (const auto &[key, value] : deepSeekV4Formulas(g, rows, contextTokens)) resolver.custom_formulas[key] = value;
    for (const auto &entry : declarations)
    {
        const auto [found, inserted] = resolver.buffer_name_to_id.emplace(entry.spec.name, entry.id);
        if (!inserted && found->second != entry.id)
            throw std::invalid_argument("DeepSeek-V4 buffer name is already bound to another identity: " + entry.spec.name);
    }
}
} // namespace llaminar2
