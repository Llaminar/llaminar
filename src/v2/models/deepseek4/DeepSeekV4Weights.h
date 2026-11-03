/**
 * @file DeepSeekV4Weights.h
 * @brief Typed DeepSeek-V4 layer weight roles over any handle (tensor or binding).
 *
 * One declaration names every model-specific canonical GGUF tensor so frozen
 * bindings, legacy tensor views and graph construction cannot drift. Ordinary
 * roles (attn_norm, ffn_norm, routed/shared experts, router gate) stay in the
 * shared LayerWeights fields; this struct carries only DeepSeek-V4 extensions.
 */
#pragma once

#include <string>
#include <type_traits>

namespace llaminar2
{
/** @brief Sinkhorn hyper-connection site: fn [mix, hc*H], base [mix], scale [3] (or [1] for the head). */
template <typename Handle> struct DeepSeekV4HyperConnectionWeights
{
    Handle fn{}, base{}, scale{};
};

/** @brief Gated-pooling compressor: kv/gate projections, position bias and output norm. */
template <typename Handle> struct DeepSeekV4CompressorWeights
{
    Handle kv{}, gate{}, ape{}, norm{};
};

/** @brief Complete DeepSeek-V4 layer extension. */
template <typename Handle> struct DeepSeekV4LayerWeights
{
    DeepSeekV4HyperConnectionWeights<Handle> hc_attn, hc_ffn;
    Handle q_a{}, q_a_norm{}, q_b{};          ///< Low-rank query.
    Handle kv{}, kv_norm{};                   ///< Single latent K == V.
    Handle output_a{}, output_b{};            ///< Grouped low-rank output.
    Handle sinks{};                           ///< Per-head attention sinks.
    DeepSeekV4CompressorWeights<Handle> compressor;           ///< Ratio 4/128 layers.
    DeepSeekV4CompressorWeights<Handle> indexer_compressor;   ///< Ratio 4 layers.
    Handle indexer_q_b{}, indexer_proj{};                     ///< Ratio 4 layers.
    Handle router_bias{}, router_bias_vl{}, hash_table{};     ///< MoE routing policy.
};

/** @brief Resolve every layer role from a layer-scoped accessor (missing roles stay empty). */
template <typename Get> auto loadDeepSeekV4LayerWeights(Get get)
{
    using Handle = std::decay_t<decltype(get(std::string{}))>;
    DeepSeekV4LayerWeights<Handle> w;
    w.hc_attn = {get("hc_attn_fn.weight"), get("hc_attn_base.weight"), get("hc_attn_scale.weight")};
    w.hc_ffn = {get("hc_ffn_fn.weight"), get("hc_ffn_base.weight"), get("hc_ffn_scale.weight")};
    w.q_a = get("attn_q_a.weight");
    w.q_a_norm = get("attn_q_a_norm.weight");
    w.q_b = get("attn_q_b.weight");
    w.kv = get("attn_kv.weight");
    w.kv_norm = get("attn_kv_a_norm.weight");
    w.output_a = get("attn_output_a.weight");
    w.output_b = get("attn_output_b.weight");
    w.sinks = get("attn_sinks.weight");
    w.compressor = {get("attn_compressor_kv.weight"), get("attn_compressor_gate.weight"),
                    get("attn_compressor_ape.weight"), get("attn_compressor_norm.weight")};
    w.indexer_compressor = {get("indexer_compressor_kv.weight"), get("indexer_compressor_gate.weight"),
                            get("indexer_compressor_ape.weight"), get("indexer_compressor_norm.weight")};
    w.indexer_q_b = get("indexer.attn_q_b.weight");
    w.indexer_proj = get("indexer.proj.weight");
    w.router_bias = get("exp_probs_b.bias");
    w.router_bias_vl = get("exp_probs_b_vl.bias");
    w.hash_table = get("ffn_gate_tid2eid.weight");
    return w;
}

/** @brief Translate handles while retaining semantic roles and null optional members. */
template <typename Handle, typename Map>
auto mapDeepSeekV4LayerWeights(const DeepSeekV4LayerWeights<Handle> &s, Map map)
{
    using Result = std::decay_t<decltype(map(s.q_a))>;
    DeepSeekV4LayerWeights<Result> w;
    w.hc_attn = {map(s.hc_attn.fn), map(s.hc_attn.base), map(s.hc_attn.scale)};
    w.hc_ffn = {map(s.hc_ffn.fn), map(s.hc_ffn.base), map(s.hc_ffn.scale)};
    w.q_a = map(s.q_a);
    w.q_a_norm = map(s.q_a_norm);
    w.q_b = map(s.q_b);
    w.kv = map(s.kv);
    w.kv_norm = map(s.kv_norm);
    w.output_a = map(s.output_a);
    w.output_b = map(s.output_b);
    w.sinks = map(s.sinks);
    w.compressor = {map(s.compressor.kv), map(s.compressor.gate), map(s.compressor.ape), map(s.compressor.norm)};
    w.indexer_compressor = {map(s.indexer_compressor.kv), map(s.indexer_compressor.gate),
                            map(s.indexer_compressor.ape), map(s.indexer_compressor.norm)};
    w.indexer_q_b = map(s.indexer_q_b);
    w.indexer_proj = map(s.indexer_proj);
    w.router_bias = map(s.router_bias);
    w.router_bias_vl = map(s.router_bias_vl);
    w.hash_table = map(s.hash_table);
    return w;
}
} // namespace llaminar2
