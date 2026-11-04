/**
 * @file DeepSeekV4BufferLayout.h
 * @brief One declaration of DeepSeek-V4 arena identities, geometry, lifetimes and persistent-state sizes.
 *
 * Schema allocation, resolver formulas, graph lookup and memory admission
 * consume the same typed inventory (the Qwen4Exp buffer-layout discipline):
 * deepSeekV4ArenaExtensionBytes() is the sum of exactly the declarations below
 * resolved through the formulas applyDeepSeekV4BufferGeometry() publishes, so
 * the planner can neither double count nor omit a DeepSeek-V4 bank.
 *
 * DeepSeek-V4 attention state is owned here, not by the canonical KV cache
 * (whose DeepSeek-V4 layers are `model_state` layers without any payload):
 *
 *   DSV4_RAW_RING          packed [layer][sequence][window] KV records (584 B)
 *   DSV4_COMP_RING_KV/SCORE FP32 [compressed layer][sequence][2*ratio][projection width]
 *   DSV4_COMP_POOL         packed [compressed layer][sequence][context/ratio] KV records
 *   DSV4_IDX_RING_KV/SCORE FP32 [ratio-4 layer][sequence][8][2*index_head_dim]
 *   DSV4_IDX_POOL          packed [ratio-4 layer][sequence][context/4] MXFP4 keys (68 B)
 *
 * Packed banks are INT32 arena words carrying the records described by
 * DeepSeekV4KVFormat.h. Only the pools grow with context; the window ring and
 * compressor state are O(window) / O(ratio) per sequence because each chunk
 * reads its own rows from projection scratch. Banks are laid out with an
 * explicit sequence axis (currently one sequence per graph).
 */
#pragma once

#include "../../memory/BufferId.h"
#include "../../execution/local_execution/graph/GraphSchema.h"
#include "../../kernels/deepseek4/DeepSeekV4KVFormat.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace llaminar2
{
struct GraphConfig;
struct GraphResolverConfig;
class DeepSeekV4Config;

/** @brief Indexer score scratch budget (vLLM VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 equivalent). */
inline constexpr std::size_t kDeepSeekV4IndexerScoreBudgetBytes = std::size_t{64} << 20;

/**
 * @brief Metadata-only DeepSeek-V4 geometry consumed by activation admission.
 * @details Carried in ModelMemoryProfile (and across MPI) so planning prices
 * exactly the arena roles graph construction allocates.
 */
struct DeepSeekV4MemoryGeometry
{
    uint32_t layers = 0, hidden = 0, hc = 0, heads = 0, head_dim = 0, q_rank = 0, o_groups = 0, o_rank = 0;
    uint32_t window = 0, index_heads = 0, index_head_dim = 0, index_topk = 0, experts = 0, experts_used = 0;
    uint32_t csa_layers = 0, hca_layers = 0, rope_dim = 0;
    /** @brief Project an authenticated config. */
    static DeepSeekV4MemoryGeometry fromConfig(const DeepSeekV4Config &model);
    /** @brief Packed attention record of the raw ring and compressed pools. */
    deepseek4::PackedKVEntryFormat kvFormat() const
    {
        return {static_cast<int>(head_dim), static_cast<int>(rope_dim)};
    }
    /** @brief Packed indexer key record. */
    deepseek4::PackedIndexerKeyFormat indexerFormat() const { return {static_cast<int>(index_head_dim)}; }
};

/**
 * @brief Rows of the bounded indexer score scratch for one admitted envelope.
 * @return min(rows, max(1, kDeepSeekV4IndexerScoreBudgetBytes / (csaPoolRows * 4))).
 */
int deepSeekV4IndexerScoreRows(int rows, int contextTokens);

/**
 * @brief Additional arena bytes of the DeepSeek-V4 extension over the shared MoE schema.
 * @param rows Admitted flattened activation rows.
 * @param contextTokens Admitted per-request context.
 * @return Bytes of every DeepSeek-V4 declaration (resolved through the same
 *         formulas as allocation) plus the widened trunk roles.
 */
std::size_t deepSeekV4ArenaExtensionBytes(const DeepSeekV4MemoryGeometry &geometry, int rows, int contextTokens);

/** @brief Bytes of the context-proportional packed pools alone (diagnostic BOM line). */
std::size_t deepSeekV4PersistentPoolBytes(const DeepSeekV4MemoryGeometry &geometry, int contextTokens);

/** @brief One registered buffer and its schema allocation contract. */
struct DeepSeekV4BufferDeclaration
{
    BufferId id;
    BufferSpec spec;
};

/**
 * @brief Per-layer, per-sequence offsets into the concatenated state banks.
 * @details FP32 banks are addressed in elements, packed banks in INT32 words.
 */
struct DeepSeekV4StateLayout
{
    int sequences = 1;            ///< Sequence slots of every bank.
    int window = 0;               ///< Raw ring rows per sequence.
    int csaPoolRows = 0;          ///< Compressed pool rows of a ratio-4 layer.
    int hcaPoolRows = 0;          ///< Compressed pool rows of a ratio-128 layer.
    int indexerScoreRows = 0;     ///< Rows of one bounded indexer scoring pass.
    deepseek4::PackedKVEntryFormat kv;           ///< Raw ring / pool record.
    deepseek4::PackedIndexerKeyFormat indexer;   ///< Indexer pool record.

    /** @brief Word offset of @p layer's raw ring for @p sequence. */
    std::size_t rawRingWordOffset(int layer, int sequence = 0) const;
    /** @brief Element offset of @p layer's compressor state for @p sequence. */
    std::size_t compStateElementOffset(const DeepSeekV4Config &model, int layer, int sequence = 0) const;
    /** @brief Word offset of @p layer's compressed pool for @p sequence. */
    std::size_t compPoolWordOffset(const DeepSeekV4Config &model, int layer, int sequence = 0) const;
    /** @brief Element offset of a ratio-4 @p layer's indexer compressor state. */
    std::size_t indexerStateElementOffset(const DeepSeekV4Config &model, int layer, int sequence = 0) const;
    /** @brief Word offset of a ratio-4 @p layer's indexer pool. */
    std::size_t indexerPoolWordOffset(const DeepSeekV4Config &model, int layer, int sequence = 0) const;
};

/** @brief Resolve the state-bank geometry from the admitted row and context envelope. */
DeepSeekV4StateLayout deepSeekV4StateLayout(const DeepSeekV4Config &model, int maxRows, int contextTokens);

/** @brief The single inventory used by schema, resolver, arena binding and admission. */
std::span<const DeepSeekV4BufferDeclaration> deepSeekV4BufferDeclarations();

/** @brief Widen the trunk roles to hc*H, drop GDN roles and add every DeepSeek-V4 declaration. */
void applyDeepSeekV4BufferLayout(GraphSchema &schema);

/**
 * @brief Publish exact formulas and buffer identities for one admitted envelope.
 * @param rows Flattened activation rows used for scratch and the indexer pass bound.
 * @param contextTokens Per-request context used for compressed pool capacities.
 */
void applyDeepSeekV4BufferGeometry(GraphResolverConfig &resolver, const DeepSeekV4MemoryGeometry &geometry,
                                   int rows, int contextTokens);
} // namespace llaminar2
