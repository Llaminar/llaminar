/**
 * @file LayerGraphIdentity.h
 * @brief Checked source/cache/workload identities for participant-local Qwen fragments.
 *
 * Graph construction carries immutable identities as explicit inputs. A shared
 * Dense0 predictor names physical source48 and its own cache0 without changing
 * the target's main interval, selecting sparse state, or borrowing ambient input.
 */
#pragma once
#include "../qwen4exp/Qwen4ExpPredictorPolicy.h"
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
namespace llaminar2
{
/** @brief Cache public API index space; physical compression remains cache-owned. */
enum class GraphCacheLayerIndex { ModelGlobal, CacheLocal };
/** @brief Native full-attention reader policy, independent of source tensor availability. */
enum class GraphAttentionConsumer { Dense, Selected };
/** @brief Semantic graph workload; a main layer never carries a predictor depth. */
enum class GraphLayerWorkload { MainDecoder, MTPDepth };
/** @brief Authenticated source convention, separate from its runtime cache ordinal. */
enum class GraphLayerSourcePolicy { MainDecoder, InlinePredictor, Qwen4ExpSharedPredictor };

/** @brief Sealed immutable source/cache/workload identity; factories reject contradictory tags. */
class LayerGraphSourceIdentity final
{
public:
    /**
     * @brief Name an ordinary layer in the exact target decoder interval.
     * @param source Physical global model layer.
     * @param main_layers Complete target layer count, excluding predictors.
     * @param pp_first First layer admitted by this participant.
     * @param cache_index Explicit public cache index space.
     * @return A main identity with no predictor depth.
     * @throws std::invalid_argument For an invalid interval or index tag.
     */
    static LayerGraphSourceIdentity main(int source, int main_layers, int pp_first,
        GraphCacheLayerIndex cache_index)
    {
        if (main_layers <= 0 || pp_first < 0 || source < pp_first || source >= main_layers ||
            (cache_index != GraphCacheLayerIndex::ModelGlobal && cache_index != GraphCacheLayerIndex::CacheLocal))
            throw std::invalid_argument("Main graph layer must belong to its admitted decoder interval");
        return {source, cache_index == GraphCacheLayerIndex::ModelGlobal ? source : source - pp_first,
            cache_index, GraphLayerSourcePolicy::MainDecoder, std::nullopt};
    }
    /**
     * @brief Name an existing inline sidecar from its frozen manifest source.
     * @param source Exact frozen source layer, never a guessed main-layer replacement.
     * @param depth Manifest depth served by this separate one-layer cache.
     * @return Explicit inline predictor identity with cache-local layer0.
     * @throws std::invalid_argument For a missing source or negative depth.
     */
    static LayerGraphSourceIdentity inlinePredictor(int source, int depth)
    {
        if (source < 0 || depth < 0)
            throw std::invalid_argument("Inline predictor requires its exact source and depth");
        return {source, 0, GraphCacheLayerIndex::CacheLocal,
            GraphLayerSourcePolicy::InlinePredictor, depth};
    }
    /**
     * @brief Authenticate the separate published QFN Dense0/PerBranch source pair.
     * @param main_layers Exact unchanged target layer count.
     * @param metadata Loader-authenticated source policy.
     * @param frozen_source Source named by the frozen MTP depth binding.
     * @param depth Only learned depth0 exists; recursive execution reuses it.
     * @param cache_layer Only this predictor's cache-local layer0 is admitted.
     * @return Shared predictor identity; sparse or mismatched policies are rejected.
     * @throws std::invalid_argument For any inconsistent source/cache/policy tag.
     */
    static LayerGraphSourceIdentity qwen4ExpSharedPredictor(int main_layers,
        const Qwen4ExpPredictorMetadata &metadata, int frozen_source, int depth, int cache_layer)
    {
        if (main_layers <= 0 || !metadata.validForMainLayers(static_cast<uint32_t>(main_layers)) ||
            metadata.source != Qwen4ExpPredictorSource::SharedGGUF ||
            metadata.attention != Qwen4ExpPredictorAttention::Dense ||
            metadata.hidden_norm != Qwen4ExpPredictorHiddenNorm::PerBranch ||
            frozen_source != static_cast<int>(metadata.source_layer) || depth != 0 || cache_layer != 0)
            throw std::invalid_argument("QFN predictor requires paired Dense0/PerBranch sourceN/cache0/depth0");
        return {frozen_source, cache_layer, GraphCacheLayerIndex::CacheLocal,
            GraphLayerSourcePolicy::Qwen4ExpSharedPredictor, depth};
    }
    /** @return Exact immutable physical source layer. */
    int sourceLayer() const noexcept { return source_; }
    /** @return Public cache-layer ordinal in the declared index space. */
    int cacheLayer() const noexcept { return cache_; }
    /** @return Explicit public cache index space. */
    GraphCacheLayerIndex cacheIndex() const noexcept { return cache_index_; }
    /** @return Main versus predictor workload, derived from the sealed source tag. */
    GraphLayerWorkload workload() const noexcept
    { return depth_ ? GraphLayerWorkload::MTPDepth : GraphLayerWorkload::MainDecoder; }
    /** @return Predictor depth only for predictor workloads; main has no value. */
    std::optional<int> predictorDepth() const noexcept { return depth_; }
    /** @return Immutable artifact convention authenticated at construction. */
    GraphLayerSourcePolicy sourcePolicy() const noexcept { return source_policy_; }
    /** @return Existing main/MTP snapshot prefix, independent of source/cache equality. */
    std::string prefix() const
    { return depth_ ? "MTP" + std::to_string(*depth_) + "_" : "layer" + std::to_string(source_) + "_"; }
    /** @brief Reject a frozen binding from a different physical source before graph mutation. */
    void requireFrozenSource(int source) const
    {
        if (source != source_) throw std::invalid_argument("Layer graph identity disagrees with frozen source weights");
    }
private:
    /** @brief Factories alone construct internally consistent tagged identities. */
    LayerGraphSourceIdentity(int source, int cache, GraphCacheLayerIndex index,
        GraphLayerSourcePolicy policy, std::optional<int> depth)
        : source_(source), cache_(cache), cache_index_(index), source_policy_(policy), depth_(depth) {}
    int source_, cache_;
    GraphCacheLayerIndex cache_index_;
    GraphLayerSourcePolicy source_policy_;
    std::optional<int> depth_;
};

/** @brief Reader selection bound to a checked source; Dense0 cannot carry a QSA consumer. */
class FullAttentionLayerIdentity final
{
public:
    /**
     * @brief Bind an ordinary main FA layer to its declared dense/selected reader.
     * @param source Exact main source/cache identity, with no predictor depth.
     * @param consumer Declared main architecture consumer policy.
     * @return Immutable FA identity; predictor sources are rejected here.
     */
    static FullAttentionLayerIdentity main(LayerGraphSourceIdentity source, GraphAttentionConsumer consumer)
    {
        if (source.workload() != GraphLayerWorkload::MainDecoder ||
            (consumer != GraphAttentionConsumer::Dense && consumer != GraphAttentionConsumer::Selected))
            throw std::invalid_argument("Main FA reader requires a main source and explicit native consumer");
        return {std::move(source), consumer};
    }
    /** @brief Bind an existing inline predictor's canonical native dense reader. */
    static FullAttentionLayerIdentity inlinePredictor(LayerGraphSourceIdentity source)
    {
        if (source.sourcePolicy() != GraphLayerSourcePolicy::InlinePredictor)
            throw std::invalid_argument("Inline FA reader requires an inline predictor source");
        return {std::move(source), GraphAttentionConsumer::Dense};
    }
    /** @brief Bind only the authenticated shared Dense0 source, with no selected-state option. */
    static FullAttentionLayerIdentity qwen4ExpSharedPredictor(LayerGraphSourceIdentity source)
    {
        if (source.sourcePolicy() != GraphLayerSourcePolicy::Qwen4ExpSharedPredictor)
            throw std::invalid_argument("Dense0 FA reader requires its paired shared predictor source");
        return {std::move(source), GraphAttentionConsumer::Dense};
    }
    /** @return Checked source/cache/workload identity consumed by projections and KV stages. */
    const LayerGraphSourceIdentity &source() const noexcept { return source_; }
    /** @return The only admitted native reader for this fragment. */
    GraphAttentionConsumer consumer() const noexcept { return consumer_; }
    /** @brief Reject attaching QSA state/indexer operations to a native dense consumer. */
    void requireSelectedConsumer() const
    {
        if (consumer_ != GraphAttentionConsumer::Selected)
            throw std::invalid_argument("Native dense FA cannot bind selected-index/QSA state");
    }
private:
    /** @brief Reader policy is sealed by the source-specific factories. */
    FullAttentionLayerIdentity(LayerGraphSourceIdentity source, GraphAttentionConsumer consumer)
        : source_(std::move(source)), consumer_(consumer) {}
    LayerGraphSourceIdentity source_;
    GraphAttentionConsumer consumer_;
};
} // namespace llaminar2
