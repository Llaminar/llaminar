/**
 * @file KVHeadCoverage.h
 * @brief Original-source bounds and pure consumer coverage for contiguous GQA heads.
 *
 * This opt-in declaration changes no default placement. A query interval may
 * need one or several original KV heads, and several consumers may need the
 * same head. The source's global bounds, never a union maximum, own identity.
 * Planner/cache/arithmetic/consumer propagation is required before adoption.
 */
#pragma once
#include <stdexcept>

namespace llaminar2
{
/** @brief Closed KV assignment policy; legacy retains current full-bank/disjoint rules. */
enum class KVHeadCoveragePolicy
{
    Legacy,
    QueryInterval,
};
/** @brief Exact original global geometry, retained by an explicitly covered assignment. */
struct KVHeadCoverageDeclaration
{
    int query_heads = 0;
    int kv_heads = 0;
};
/** @brief Half-open interval of original KV heads; no storage or replica ledger. */
struct KVHeadInterval
{
    int first = 0;
    int count = 0;
    /** @return Exclusive original end; resolver guarantees representable bounded values. */
    [[nodiscard]] int end() const noexcept { return first + count; }
    bool operator==(const KVHeadInterval &) const = default;
};
/**
 * @brief Validate original integral GQA geometry before division or local arithmetic.
 * @param source Original source head counts, not derived participant totals.
 * @throws std::invalid_argument For absent/nonintegral/foreign source geometry.
 */
inline void validateKVHeadCoverageDeclaration(KVHeadCoverageDeclaration source)
{
    if (source.query_heads <= 0 || source.kv_heads <= 0 ||
        source.query_heads % source.kv_heads != 0)
        throw std::invalid_argument("KV query coverage requires positive integral original GQA geometry");
}
/**
 * @brief Resolve the smallest contiguous original KV interval consumed by an actual Q interval.
 * @param source Authenticated original Q/KV counts, with integral positive GQA ratio.
 * @param query_first Original Q origin; participant/device ordinal is never an origin.
 * @param query_count Positive original Q width; empty assignments remain unsupported.
 * @return Exactly the original heads touched by the first and last consumed Q head.
 * @throws std::invalid_argument Before overflow/division for invalid or empty intervals.
 */
[[nodiscard]] inline KVHeadInterval resolveQueryKVHeadCoverage(
    KVHeadCoverageDeclaration source, int query_first, int query_count)
{
    validateKVHeadCoverageDeclaration(source);
    if (query_first < 0 || query_count <= 0 || query_first >= source.query_heads ||
        query_count > source.query_heads - query_first)
        throw std::invalid_argument("KV query coverage interval is empty or outside original Q heads");
    const int ratio = source.query_heads / source.kv_heads;
    const int first = query_first / ratio;
    const int last = (query_first + query_count - 1) / ratio;
    return {first, last - first + 1};
}
} // namespace llaminar2
