/**
 * @file GraphSnapshotSourceOutput.h
 * @brief Frozen source-owned diagnostic selections, independent of live counts.
 *
 * A semantic key selects one actual QSA producer output on a main decoder
 * layer. This identity prices native state capacity rather than the smaller
 * live observation in an NPY. It owns no storage, count, allocator or ledger.
 */
#pragma once
#include <charconv>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2
{
/** @brief Closed main-QSA producer roles with authored native extent metadata. */
enum class GraphSnapshotSourceRole { QwenSparseQuery, QwenSparsePooledKeys, QwenSparseSelectedIndices, QwenSparseRawQuery };
/** @brief Sealed source role/layer identity; the planner authenticates residency and exact model geometry. */
class GraphSnapshotSourceOutput final
{
public:
    /** @brief Admit a known QSA producer role and nonnegative main-model layer. */
    static GraphSnapshotSourceOutput qwenSparse(int layer, GraphSnapshotSourceRole role)
    {
        if (layer < 0) throw std::invalid_argument("QSA snapshot source requires a nonnegative main layer");
        switch (role) {
        case GraphSnapshotSourceRole::QwenSparseRawQuery:
        case GraphSnapshotSourceRole::QwenSparseQuery:
        case GraphSnapshotSourceRole::QwenSparsePooledKeys:
        case GraphSnapshotSourceRole::QwenSparseSelectedIndices: return {layer, role};
        }
        throw std::invalid_argument("Unknown source-owned graph snapshot role");
    }
    /** @return Exact decoder source layer, never a predictor/cache layer identity. */
    int layer() const noexcept { return layer_; }
    /** @return Immutable producer role. */
    GraphSnapshotSourceRole role() const noexcept { return role_; }
    /** @return Whether native full-state payload replaces live reference-file payload. */
    bool replacesReferencePayload() const noexcept { return role_ == GraphSnapshotSourceRole::QwenSparsePooledKeys; }
    /** @return Canonical semantic key emitted by the real source producer. */
    std::string semanticKey() const
    {
        const char *suffix = role_ == GraphSnapshotSourceRole::QwenSparseRawQuery ? "_QSA_INDEXER_RAW_QUERY" :
            role_ == GraphSnapshotSourceRole::QwenSparseQuery ? "_QSA_INDEXER_QUERY" :
            role_ == GraphSnapshotSourceRole::QwenSparsePooledKeys ? "_QSA_INDEXER_POOLED_KEYS" : "_QSA_SELECTED_INDICES";
        return "layer" + std::to_string(layer_) + suffix;
    }
    /**
     * @brief Parse only the source's exact canonical main-layer key grammar.
     * @return A source selection, or no value for unrelated/child/alias keys.
     *
     * This does not infer model membership. The metadata planner must reject a
     * layer absent from the admitted sparse-main manifest before reservation.
     */
    static std::optional<GraphSnapshotSourceOutput> fromQwenSparseSemanticKey(std::string_view key)
    {
        if (!key.starts_with("layer")) return std::nullopt;
        const auto separator = key.find('_', 5);
        if (separator == std::string_view::npos) return std::nullopt;
        const auto digits = key.substr(5, separator - 5);
        if (digits.empty() || (digits.size() > 1 && digits.front() == '0') ||
            !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) return std::nullopt;
        int layer = -1;
        const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
        if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || layer < 0) return std::nullopt;
        const auto suffix = key.substr(separator);
        if (suffix == "_QSA_INDEXER_RAW_QUERY") return qwenSparse(layer, GraphSnapshotSourceRole::QwenSparseRawQuery);
        if (suffix == "_QSA_INDEXER_QUERY") return qwenSparse(layer, GraphSnapshotSourceRole::QwenSparseQuery);
        if (suffix == "_QSA_INDEXER_POOLED_KEYS") return qwenSparse(layer, GraphSnapshotSourceRole::QwenSparsePooledKeys);
        if (suffix == "_QSA_SELECTED_INDICES") return qwenSparse(layer, GraphSnapshotSourceRole::QwenSparseSelectedIndices);
        return std::nullopt;
    }
    friend bool operator==(const GraphSnapshotSourceOutput &, const GraphSnapshotSourceOutput &) = default;
private:
    /** @brief Only checked factories construct immutable identities. */
    GraphSnapshotSourceOutput(int layer, GraphSnapshotSourceRole role) : layer_(layer), role_(role) {}
    int layer_;
    GraphSnapshotSourceRole role_;
};
} // namespace llaminar2
