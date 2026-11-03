/**
 * @file DeepSeekV4BufferLayout.h
 * @brief One declaration of DeepSeek-V4 arena identities, geometry and lifetimes.
 *
 * Schema allocation, resolver formulas and graph lookup consume the same typed
 * inventory (the Qwen4Exp buffer-layout discipline). Scratch roles are shared
 * by sequential layers. Position-indexed state banks (raw window rings,
 * compressor rings and compressed pools) concatenate every owned layer; each
 * layer binds a byte-offset subview. Ring capacities are sized from the admitted
 * activation-row envelope so a whole prefill chunk plus the window (or two
 * compression blocks) is resident at once.
 */
#pragma once

#include "../../memory/BufferId.h"
#include "../../execution/local_execution/graph/GraphSchema.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace llaminar2
{
struct GraphConfig;
struct GraphResolverConfig;
class DeepSeekV4Config;

/**
 * @brief Metadata-only DeepSeek-V4 geometry consumed by activation admission.
 * @details Carried in ModelMemoryProfile (and across MPI) so planning prices
 * exactly the arena roles graph construction allocates.
 */
struct DeepSeekV4MemoryGeometry
{
    uint32_t layers = 0, hidden = 0, hc = 0, heads = 0, head_dim = 0, q_rank = 0, o_groups = 0, o_rank = 0;
    uint32_t window = 0, index_heads = 0, index_head_dim = 0, index_topk = 0, experts = 0, experts_used = 0;
    uint32_t csa_layers = 0, hca_layers = 0;
    /** @brief Project an authenticated config. */
    static DeepSeekV4MemoryGeometry fromConfig(const DeepSeekV4Config &model);
};

/**
 * @brief Additional arena bytes of the DeepSeek-V4 extension over the shared MoE schema.
 * @param rows Admitted flattened activation rows.
 * @param contextTokens Admitted per-request context.
 * @return Bytes of every DeepSeek-V4 declaration plus the widened trunk roles.
 */
std::size_t deepSeekV4ArenaExtensionBytes(const DeepSeekV4MemoryGeometry &geometry, int rows, int contextTokens);

/** @brief One registered buffer and its schema allocation contract. */
struct DeepSeekV4BufferDeclaration
{
    BufferId id;
    BufferSpec spec;
};

/** @brief Per-layer offsets into the concatenated state banks (in rows). */
struct DeepSeekV4StateLayout
{
    int rawRingRows = 0;          ///< Raw ring rows per layer.
    int csaRingRows = 0;          ///< Compressor ring rows of a ratio-4 layer.
    int hcaRingRows = 0;          ///< Compressor ring rows of a ratio-128 layer.
    int csaPoolRows = 0;          ///< Compressed pool rows of a ratio-4 layer.
    int hcaPoolRows = 0;          ///< Compressed pool rows of a ratio-128 layer.
    /** @brief Element offset of @p layer's compressor ring in the shared kv/score banks. */
    std::size_t compRingElementOffset(const DeepSeekV4Config &model, int layer) const;
    /** @brief Element offset of @p layer's compressed pool. */
    std::size_t compPoolElementOffset(const DeepSeekV4Config &model, int layer) const;
    /** @brief Element offset of a ratio-4 @p layer's indexer ring. */
    std::size_t indexerRingElementOffset(const DeepSeekV4Config &model, int layer) const;
    /** @brief Element offset of a ratio-4 @p layer's indexer pool. */
    std::size_t indexerPoolElementOffset(const DeepSeekV4Config &model, int layer) const;
};

/** @brief Resolve the state-bank geometry from the admitted row and context envelope. */
DeepSeekV4StateLayout deepSeekV4StateLayout(const DeepSeekV4Config &model, int maxRows, int contextTokens);

/** @brief The single inventory used by schema, resolver and arena binding. */
std::span<const DeepSeekV4BufferDeclaration> deepSeekV4BufferDeclarations();

/** @brief Widen the trunk roles to hc*H, drop GDN roles and add every DeepSeek-V4 declaration. */
void applyDeepSeekV4BufferLayout(GraphSchema &schema);

/**
 * @brief Publish exact formulas and buffer identities for one admitted envelope.
 * @param rows Flattened activation rows used for scratch and ring capacities.
 * @param contextTokens Per-request context used for compressed pool capacities.
 */
void applyDeepSeekV4BufferGeometry(GraphResolverConfig &resolver, const DeepSeekV4MemoryGeometry &geometry,
                                   int rows, int contextTokens);
} // namespace llaminar2
