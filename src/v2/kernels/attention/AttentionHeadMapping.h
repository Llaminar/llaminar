/**
 * @file AttentionHeadMapping.h
 * @brief One checked local-query to visible-KV mapping for every attention backend.
 *
 * Sharded KV uses an integral local grouping. Fully replicated KV instead uses
 * the original global query-head offset and GQA ratio; an odd local query count
 * need not divide the complete KV-head count. This is immutable launch geometry,
 * not mutable inference state or a second tensor-parallel assignment authority.
 */
#pragma once
#include <cstdint>

namespace llaminar2::attention
{
/** @brief Participant-visible query ownership and original global GQA grouping. */
struct HeadMappingGeometry
{
    int local_query_heads = 0;    ///< Positive number of local query heads.
    int visible_kv_heads = 0;     ///< Positive number of addressable local KV heads.
    int head_start = 0;           ///< Original global Q offset when KV is replicated.
    int replicated_gqa_n_rep = 0; ///< Original global Q/KV ratio; zero selects local grouping.
};
/**
 * @brief Check every local query head maps into the retained visible KV bank.
 * @param geometry Exact participant query interval and local/global grouping.
 * @return True for integral local grouping or bounded original global mapping.
 *
 * Widened arithmetic checks the complete half-open global interval without
 * overflow. A missing global ratio never licenses a fractional local grouping.
 */
[[nodiscard]] inline constexpr bool isValidHeadMapping(const HeadMappingGeometry &geometry)
{
    if (geometry.local_query_heads <= 0 || geometry.visible_kv_heads <= 0 ||
        geometry.head_start < 0 || geometry.replicated_gqa_n_rep < 0)
        return false;
    if (geometry.replicated_gqa_n_rep == 0)
        return geometry.local_query_heads % geometry.visible_kv_heads == 0;
    const std::int64_t final_query_head =
        static_cast<std::int64_t>(geometry.head_start) + geometry.local_query_heads;
    const std::int64_t represented_query_heads =
        static_cast<std::int64_t>(geometry.visible_kv_heads) * geometry.replicated_gqa_n_rep;
    return final_query_head <= represented_query_heads;
}
} // namespace llaminar2::attention
