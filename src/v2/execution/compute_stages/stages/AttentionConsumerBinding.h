/**
 * @file AttentionConsumerBinding.h
 * @brief Exact participant-local attention inputs after native cache publication.
 *
 * Model-specific consumers receive prepared Q and the original projection
 * input before ordinary attention chooses any gathered/converted KV view. The
 * actual layer weights travel with this binding, including shifted MTP layers;
 * cache-local layer indices are never used to guess a main-model weight set.
 */
#pragma once
#include "../../../backends/DeviceId.h"
#include "../../../memory/BufferId.h"
#include "../../../kernels/attention/AttentionExecutionPolicy.h"
#include "../../../models/qwen/LayerGraphIdentity.h"
#include <string>

namespace llaminar2
{
class TensorBase;
class IKVCache;
struct LayerWeights;
/** @brief Exact existing activation identity with no stage-owned storage. */
struct AttentionArenaTensor
{
    TensorBase &tensor;
    BufferId buffer;
};
/** @brief Sealed core inputs and explicit producer edges for an attention consumer. */
struct AttentionConsumerBinding
{
    FullAttentionLayerIdentity layerIdentity; ///< Sealed source/cache/reader policy, independent of tensor availability.
    const LayerWeights &weights; ///< Actual model/MTP FA layer passed to the graph builder.
    std::string prefix; ///< Existing unique layer node prefix.
    DeviceId device; ///< Exact participant; movements remain graph collectives.
    AttentionArenaTensor projectionInput; ///< Input shared by ordinary Q/K/V and extra model indexer projections.
    AttentionArenaTensor preparedQuery; ///< Ordinary per-head normalized and rotary Q.
    AttentionArenaTensor output; ///< Ordinary attention result consumed by gate/Wo.
    IKVCache *cache; ///< Native cache authority; specialized cache-backed consumers require non-null.
    int modelLayer, cacheLayer; ///< Explicit weight identity and resolved native cache slot.
    int queryRows, requests, queryHeads, kvHeads, queryHeadStart, kvHeadStart;
    attention::AttentionKeyCachePolicy keyPolicy; ///< Same immutable encoding used by native cache append.
    std::string projectionInputReady, preparedQueryReady, nativeCacheReady; ///< Exact graph producers.
    /** @brief Authenticate the public cache ordinal before any append or auxiliary attachment.
     * @throws std::invalid_argument For a conflicting source or cache-layer projection.
     */
    void requireLayerIdentity() const
    {
        if (modelLayer != layerIdentity.source().sourceLayer() ||
            cacheLayer != layerIdentity.source().cacheLayer())
            throw std::invalid_argument("Attention handoff contradicts its sealed source/cache identity");
    }
};
} // namespace llaminar2
