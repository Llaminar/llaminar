/**
 * @file Qwen4ExpWeights.h
 * @brief Typed graph bindings for Flash Next's branch mixers, indexer and PLE.
 *
 * The same declarations carry frozen weight handles and execution tensor views.
 * Name resolution happens once while constructing a graph; execution does not
 * probe names or substitute ordinary transformer norms for branch mixers. The
 * enormous immutable n-gram table deliberately has no graph weight binding:
 * its native rows belong to the explicit CPU row-storage service.
 */
#pragma once
#include <string>
#include <type_traits>

namespace llaminar2
{
/** @brief One learned read/write mixer; terminal mixers have no injection matrix. */
template <typename Handle> struct HyperConnectionWeights
{
    Handle norm{}; ///< Distinct checkpoint gains [branches,H].
    Handle down{}; ///< [rank,branches*H], followed by /branches and SiLU.
    Handle up{}; ///< [branches*H,rank], followed by sigmoid on the read path.
    Handle inject{}; ///< [branches,branches*H], followed by /branches on the write path.
};

/** @brief Replicated compressed-indexer projections and per-head norm gains. */
template <typename Handle> struct QwenSparseIndexerWeights
{
    Handle query{}, key{}, query_norm{}, key_norm{};
};

/** @brief PLE projection, branch-normalization and dilated-convolution weights. */
template <typename Handle> struct QwenPLEWeights
{
    Handle key{}, value{}, key_norm{}, query_norm{}, convolution_norm{}, convolution{};
};

/** @brief Complete layer-local Flash Next extension, separate from compute-width blocks. */
template <typename Handle> struct Qwen4ExpLayerWeights
{
    HyperConnectionWeights<Handle> attention, ffn;
    QwenSparseIndexerWeights<Handle> indexer;
    QwenPLEWeights<Handle> ple;
};

/**
 * @brief Resolve a mixer from an explicitly scoped canonical-name accessor.
 * @param get Frozen binding or tensor accessor, with caller-owned layer prefix.
 * @param prefix Mixer suffix prefix, including its terminating underscore.
 * @return All four mixer roles; the absent terminal injection remains empty.
 */
template <typename Get>
auto loadHyperConnectionWeights(Get get, const std::string &prefix)
{
    using Handle = std::decay_t<decltype(get(prefix + "norm.weight"))>;
    return HyperConnectionWeights<Handle>{get(prefix + "norm.weight"),
        get(prefix + "down.weight"), get(prefix + "up.weight"),
        get(prefix + "inject.weight")};
}

/** @brief Resolve all layer extension roles without loading the immutable row table. */
template <typename Get> auto loadQwen4ExpLayerWeights(Get get)
{
    using Handle = std::decay_t<decltype(get(std::string{}))>;
    return Qwen4ExpLayerWeights<Handle>{
        .attention = loadHyperConnectionWeights(get, "hc_attn_"),
        .ffn = loadHyperConnectionWeights(get, "hc_ffn_"),
        .indexer = {get("indexer.q_proj.weight"), get("indexer.k_proj.weight"),
                    get("indexer.q_norm.weight"), get("indexer.k_norm.weight")},
        .ple = {get("ple_key.weight"), get("ple_value.weight"), get("ple_norm_key.weight"),
                get("ple_norm_query.weight"), get("ple_norm_conv.weight"), get("ple_conv1d.weight")}};
}

/** @brief Translate handles while retaining semantic roles and null optional members. */
template <typename Handle, typename Map>
auto mapHyperConnectionWeights(const HyperConnectionWeights<Handle> &source, Map map)
{
    using Result = std::decay_t<decltype(map(source.norm))>;
    return HyperConnectionWeights<Result>{map(source.norm), map(source.down),
                                          map(source.up), map(source.inject)};
}

/** @brief Build execution views from frozen layer bindings without duplicating names. */
template <typename Handle, typename Map>
auto mapQwen4ExpLayerWeights(const Qwen4ExpLayerWeights<Handle> &source, Map map)
{
    using Result = std::decay_t<decltype(map(source.attention.norm))>;
    return Qwen4ExpLayerWeights<Result>{
        .attention = mapHyperConnectionWeights(source.attention, map),
        .ffn = mapHyperConnectionWeights(source.ffn, map),
        .indexer = {map(source.indexer.query), map(source.indexer.key),
                    map(source.indexer.query_norm), map(source.indexer.key_norm)},
        .ple = {map(source.ple.key), map(source.ple.value), map(source.ple.key_norm),
                map(source.ple.query_norm), map(source.ple.convolution_norm), map(source.ple.convolution)}};
}
} // namespace llaminar2
