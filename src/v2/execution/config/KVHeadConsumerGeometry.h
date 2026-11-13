/**
 * @file KVHeadConsumerGeometry.h
 * @brief Checked cache and diagnostic consumers of an explicitly retained query KV assignment.
 *
 * The original assignment owns head origin/count. Full-model retained families
 * are a separate typed demand, never a failed subset lookup. These pure helpers
 * allocate no inference storage, inspect no payload and do not opt in a model.
 * Cache/PMA and prefix identities continue to use the actual cache geometry.
 */
#pragma once
#include "../../models/GraphTypes.h"
#include "../../collective/ILocalTPContext.h"
#include "../debug/KVHeadSnapshotColumnGroups.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>
#include <optional>
#include <span>
#include <string_view>

namespace llaminar2
{
/** @brief Exact retained family demand; full-model replication is an explicit policy. */
enum class KVHeadConsumerDemand { AssignedQueryInterval, FullModel };
/**
 * @brief Recognize an explicitly authenticated query-coverage declaration.
 * @param config Actual participant graph configuration.
 * @return Whether this consumer must use original head origins rather than index arithmetic.
 */
inline bool hasQueryIntervalKVDeclaration(const GraphConfig &config) noexcept
{
    return config.tp_config && config.tp_config->kvHeadCoveragePolicy() == KVHeadCoveragePolicy::QueryInterval;
}
/**
 * @brief Resolve a retained cache's exact source interval without creating or observing a cache.
 * @param config Actual participant configuration retaining the original assignment.
 * @param device Actual owning device, not its vector position or hardware ordinal arithmetic.
 * @param demand Explicit assigned interval or declared complete-family ownership.
 * @return No value only for the unchanged legacy policy; otherwise the checked original interval.
 * @throws std::invalid_argument For stale source, assignment, device or consumer geometry.
 */
inline std::optional<KVHeadInterval> resolveQueryIntervalCacheHeads(
    const GraphConfig &config, DeviceId device, KVHeadConsumerDemand demand)
{
    if (demand != KVHeadConsumerDemand::AssignedQueryInterval && demand != KVHeadConsumerDemand::FullModel)
        throw std::invalid_argument("unknown retained KV consumer demand");
    if (!hasQueryIntervalKVDeclaration(config)) return std::nullopt;
    const auto &tp = *config.tp_config;
    if (!tp.validate() || !tp.kvHeadCoverageDeclaration() || config.local_rank < 0 ||
        config.local_rank >= tp.worldSize() || config.tp_device_idx != config.local_rank)
        throw std::invalid_argument("query KV consumer lacks a valid original assignment");
    if (config.tp_ctx)
    {
        if (config.tp_ctx->degree() != tp.worldSize())
            throw std::invalid_argument("query KV consumer context degree differs from its assignment");
        if (config.tp_ctx->isLocal())
        {
            const auto *local = dynamic_cast<const ILocalTPContext *>(config.tp_ctx);
            if (!local || local->devices().size() != tp.assignments().size())
                throw std::invalid_argument("query KV consumer lacks its declared local membership");
            for (int rank = 0; rank < tp.worldSize(); ++rank)
                if (local->devices()[rank].toLocalDeviceId() != tp.forRank(rank).device)
                    throw std::invalid_argument("query KV consumer context membership differs from its assignment");
        }
    }
    const auto source = *tp.kvHeadCoverageDeclaration();
    const auto &a = tp.forRank(config.local_rank);
    if (source.query_heads != config.n_heads || source.kv_heads != config.n_kv_heads ||
        a.local_rank != config.local_rank || a.device != device ||
        a.head_start != config.head_start || a.head_count != config.local_n_heads ||
        a.kv_head_count != config.local_n_kv_heads)
        throw std::invalid_argument("query KV consumer disagrees with original geometry/device/assignment");
    const auto exact = resolveQueryKVHeadCoverage(source, a.head_start, a.head_count);
    if (exact.first != a.kv_head_start || exact.count != a.kv_head_count)
        throw std::invalid_argument("query KV consumer assignment no longer covers its actual queries");
    return demand == KVHeadConsumerDemand::FullModel ? KVHeadInterval{0, source.kv_heads} : exact;
}
/**
 * @brief Recognize only real K/V semantic outputs, including their explicit per-row diagnostic aliases.
 * @param stage Schema-facing stage name after transaction/layer scope normalization.
 * @return Whether original KV head groups, rather than generic concatenation, own this output.
 */
inline bool isKVHeadSnapshotStage(std::string_view stage) noexcept
{
    const auto row = stage.rfind("_ROW");
    if (row != std::string_view::npos && row + 4 < stage.size())
    {
        for (size_t i = row + 4; i < stage.size(); ++i)
            if (stage[i] < '0' || stage[i] > '9') return false;
        stage = stage.substr(0, row);
    }
    constexpr std::array<std::string_view, 10> names = {
        "K_PROJECTION", "V_PROJECTION", "K_NORM", "K_ROPE", "KV_APPEND_SOURCE_K", "KV_APPEND_SOURCE_V",
        "KV_CACHE_K", "KV_CACHE_V", "ATTENTION_EFFECTIVE_K", "ATTENTION_EFFECTIVE_V"};
    return std::find(names.begin(), names.end(), stage) != names.end();
}
/**
 * @brief Remove only known transaction scopes while retaining the original producer layer identity.
 * @param key Original retained checkpoint key.
 * @return View naming the actual main/depth producer, never the current transaction.
 */
inline std::string_view kvHeadSnapshotProducerKey(std::string_view key)
{
    for (const auto scope : {std::string_view("MTP_REQUEST_BATCH_CONDITION_"), std::string_view("MTP_SCALAR_CONDITION_")})
        if (key.starts_with(scope)) { key.remove_prefix(scope.size()); break; }
    const auto offset = prefillChunkSnapshotSemanticStageOffset(key);
    if (offset != std::string_view::npos) key.remove_prefix(offset);
    return key;
}
/** @brief Concrete original layer geometry; no default lookup can substitute a foreign source layer. */
struct KVHeadSnapshotLayerGeometry
{
    int columns_per_head = 0; ///< Original absolute layer's head dimension.
    int original_kv_heads = 0; ///< Original absolute layer's KV count.
};
/**
 * @brief Authenticate original layer ownership before selecting any per-layer geometry.
 * @param config Actual graph declaration with original global count and owned PP interval.
 * @param key Retained publication key, including any historical/condition scope.
 * @return Actual source layer head geometry; depth-qualified predictors keep their separate declaration.
 * @throws std::invalid_argument For missing/foreign source layers, incomplete global arrays or stale PP bounds.
 *
 * GraphConfig's general getLayer accessors intentionally return defaults beyond
 * their arrays. That behavior cannot authenticate a retained original source.
 * PP keys are absolute; production config builders retain global arrays after
 * reducing n_layers to the local PP count, so no subtraction is applied here.
 */
inline KVHeadSnapshotLayerGeometry resolveQueryKVSnapshotLayerGeometry(
    const GraphConfig &config, std::string_view key)
{
    const auto scoped = kvHeadSnapshotProducerKey(key);
    if (isMTPDepthQualifiedSnapshot(scoped))
    {
        if (config.head_dim <= 0 || config.n_kv_heads <= 0)
            throw std::invalid_argument("query KV predictor lacks its declared source geometry");
        return {config.head_dim, config.n_kv_heads};
    }
    if (!scoped.starts_with("layer") && !scoped.starts_with("LAYER"))
        throw std::invalid_argument("query KV snapshot lacks an original source layer");
    size_t cursor = 5;
    if (cursor == scoped.size() || scoped[cursor] < '0' || scoped[cursor] > '9')
        throw std::invalid_argument("query KV snapshot has malformed original layer identity");
    int layer = 0;
    while (cursor < scoped.size() && scoped[cursor] >= '0' && scoped[cursor] <= '9')
    {
        if (layer > (std::numeric_limits<int>::max() - (scoped[cursor] - '0')) / 10)
            throw std::invalid_argument("query KV snapshot layer identity overflows");
        layer = layer * 10 + (scoped[cursor++] - '0');
    }
    if (cursor == scoped.size() || scoped[cursor] != '_')
        throw std::invalid_argument("query KV snapshot has malformed original layer identity");
    const int global = config.total_n_layers;
    if (global <= 0 || config.n_layers <= 0 || config.pp_layer_offset < 0 ||
        config.pp_layer_offset >= global || config.n_layers > global - config.pp_layer_offset ||
        layer >= global || layer < config.pp_layer_offset || layer - config.pp_layer_offset >= config.n_layers)
        throw std::invalid_argument("query KV snapshot layer is outside its declared original/owned PP interval");
    const auto select = [global, layer](const std::vector<int> &original, int homogeneous) {
        if (!original.empty() && original.size() != static_cast<size_t>(global))
            throw std::invalid_argument("query KV snapshot original layer geometry is incomplete");
        const int value = original.empty() ? homogeneous : original[static_cast<size_t>(layer)];
        if (value <= 0) throw std::invalid_argument("query KV snapshot original layer geometry is invalid");
        return value;
    };
    if (select(config.layer_n_heads, config.n_heads) != config.n_heads)
        throw std::invalid_argument("query KV snapshot original query geometry differs from its assignment");
    return {select(config.layer_head_dim, config.head_dim), select(config.layer_n_kv_heads, config.n_kv_heads)};
}

/** @brief Passive geometry of one actual captured producer in canonical participant order. */
struct KVHeadSnapshotProducer
{
    const GraphConfig *config = nullptr; ///< Actual retained producer configuration; no source shadow.
    DeviceId device = DeviceId::cpu(); ///< Actual physical owner, independent of participant position.
    size_t row_columns = 0; ///< Width reported by the actual immutable snapshot publication.
    int columns_per_head = 0; ///< Concrete producing layer's original head dimension.
    int original_kv_heads = 0; ///< Concrete producing layer's original KV head count.
};
/**
 * @brief Resolve original-head groups from every real producer's configuration and reported row width.
 * @param producers All participant producers in assignment order; no missing replica is ignored.
 * @param demand Exact retained family policy used by the producing graph.
 * @return No value only when no producer opted in; otherwise complete checked original-head groups.
 * @throws std::invalid_argument For mixed/missing/foreign membership, geometry or producer widths.
 */
inline std::optional<std::vector<SnapshotColumnGroup>> resolveQueryIntervalKVSnapshotGroups(
    std::span<const KVHeadSnapshotProducer> producers, KVHeadConsumerDemand demand)
{
    if (demand != KVHeadConsumerDemand::AssignedQueryInterval && demand != KVHeadConsumerDemand::FullModel)
        throw std::invalid_argument("unknown retained KV snapshot demand");
    const GraphConfig *first = nullptr;
    for (const auto &p : producers)
        if (p.config && hasQueryIntervalKVDeclaration(*p.config)) { first = p.config; break; }
    if (!first) return std::nullopt;
    const auto &tp = *first->tp_config;
    if (producers.size() != static_cast<size_t>(tp.worldSize()))
        throw std::invalid_argument("query KV snapshot is missing a participant");
    const int width = producers.front().columns_per_head;
    if (width <= 0 || first->n_kv_heads <= 0 || static_cast<size_t>(width) >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(first->n_kv_heads)) throw std::invalid_argument("query KV snapshot lacks actual producer head width");
    std::vector<size_t> row_widths;
    for (size_t rank = 0; rank < producers.size(); ++rank)
    {
        const auto &p = producers[rank];
        if (!p.config || !hasQueryIntervalKVDeclaration(*p.config) || p.config->local_rank != static_cast<int>(rank) ||
            p.columns_per_head != width || p.original_kv_heads != first->n_kv_heads ||
            p.config->tp_config->worldSize() != tp.worldSize())
            throw std::invalid_argument("query KV snapshot has mixed or stale producer geometry");
        const auto interval = resolveQueryIntervalCacheHeads(*p.config, p.device, demand);
        for (int member = 0; member < tp.worldSize(); ++member)
        {
            const auto &expected = tp.forRank(member), &actual = p.config->tp_config->forRank(member);
            if (expected.device != actual.device || expected.local_rank != actual.local_rank ||
                expected.head_start != actual.head_start || expected.head_count != actual.head_count ||
                expected.kv_head_start != actual.kv_head_start || expected.kv_head_count != actual.kv_head_count)
                throw std::invalid_argument("query KV snapshot producers retain different memberships/assignments");
        }
        if (p.row_columns != static_cast<size_t>(interval->count) * static_cast<size_t>(width))
            throw std::invalid_argument("query KV snapshot width differs from its actual producer interval");
        row_widths.push_back(p.row_columns);
    }
    if (demand == KVHeadConsumerDemand::AssignedQueryInterval)
        return resolveKVHeadSnapshotColumnGroups(tp, row_widths, static_cast<size_t>(width));
    // A declared complete retained family has every head on every participant.
    std::vector<SnapshotColumnGroup> groups;
    for (int head = 0; head < first->n_kv_heads; ++head)
        groups.push_back({"KV_HEAD_" + std::to_string(head), static_cast<size_t>(width),
            SnapshotColumnGroupMode::REPLICATED, std::vector<size_t>(producers.size(), static_cast<size_t>(width))});
    return groups;
}
} // namespace llaminar2
