/**
 * @file PrefixAuxiliaryStateIO.cpp
 * @brief Transfer economical model pool blocks and endpoint archives without dense bank gathering.
 *
 * Each FA ordinal resolves through the canonical cache's global model map.
 * A host archive is an intentional prefix-tier transfer boundary; device-hot
 * archives remain native D2D. No allocation, count readback or synchronization
 * occurs here, and owner-specific codecs/lifecycle stay behind the typed API.
 */
#include "PrefixAuxiliaryStateIO.h"
#include "PrefixPayloadLayout.h"
#include <stdexcept>
namespace llaminar2
{
namespace
{
/** @brief Immutable range authentication shared by export and import. */
struct Geometry
{
    bool mtp,device;
    int first,count,end,layers;
    const PrefixAuxiliaryPayloadLayout &layout;
    Geometry(const PrefixBlockHandle &handle,PrefixAuxiliaryLane lane)
        :mtp(lane==PrefixAuxiliaryLane::ShiftedMTP),device(handle.tier==PrefixStorageTier::DeviceHot),
        first(mtp ? mtpShiftedBlockRange(handle.key.token_start,handle.key.token_count,handle.layout.mtp_origin).first : handle.key.token_start),
        count(mtp ? mtpShiftedBlockRange(handle.key.token_start,handle.key.token_count,handle.layout.mtp_origin).count : handle.key.token_count),
        end(first+count),layers(mtp ? handle.layout.mtp_layers : handle.layout.fa_layers),
        layout(mtp ? handle.layout.mtp_auxiliary : handle.layout.main_auxiliary) {}
};
/** @brief Reject archive identity mismatches before any native payload mutation. */
void validate(const ICacheAuxiliaryState *owner,const Geometry &g)
{
    if(bool(owner)!=g.layout.kind.has_value() || (owner && owner->kind()!=*g.layout.kind))
        throw std::invalid_argument("prefix auxiliary archive kind differs from canonical cache state");
}
}
void exportPrefixAuxiliaryState(const IKVCache &cache,int sequence,PrefixAuxiliaryLane lane,
    PrefixBlockHandle &handle,void *stream)
{
    if(lane==PrefixAuxiliaryLane::ShiftedMTP)
        requireMTPShiftedCacheOrigin(cache.stateOwnership().shifted_origin,handle.layout.mtp_origin);
    const Geometry g(handle,lane);const auto *owner=cache.auxiliaryState();validate(owner,g);if(!owner)return;
    const auto domain=g.device ? CacheAuxiliaryMemoryDomain::Device : CacheAuxiliaryMemoryDomain::Host;
    for(int ordinal=0;ordinal<g.layers;++ordinal) {
        const CacheAuxiliaryBlockRange range{prefixFALayerForIndex(cache,ordinal),sequence,g.first,g.count};
        const auto bytes=owner->blockArchiveBytes(range);
        if(bytes>g.layout.bytes_per_attention_layer)throw std::invalid_argument("prefix auxiliary block exceeds declared live stride");
        if(!bytes)continue;
        auto *data=g.mtp ? (g.device ? handle.deviceMTPAuxiliaryBlockData(ordinal) : handle.mtpAuxiliaryBlockData(ordinal)) :
            (g.device ? handle.deviceMainAuxiliaryBlockData(ordinal) : handle.mainAuxiliaryBlockData(ordinal));
        owner->exportBlock(range,{data,bytes,domain},stream);
    }
    if(g.layout.includes_endpoint) {
        const auto bytes=owner->endpointArchiveBytes(sequence,g.end);
        if(bytes!=g.layout.endpoint_bytes)throw std::invalid_argument("prefix auxiliary endpoint differs from immutable prefix geometry");
        if(bytes) {
            auto *data=g.mtp ? (g.device ? handle.deviceMTPAuxiliaryEndpointData() : handle.mtpAuxiliaryEndpointData()) :
                (g.device ? handle.deviceMainAuxiliaryEndpointData() : handle.mainAuxiliaryEndpointData());
            owner->exportEndpoint(sequence,g.end,{data,bytes,domain},stream);
        }
    }
}
void importPrefixAuxiliaryState(IKVCache &cache,int sequence,PrefixAuxiliaryLane lane,
    const PrefixBlockHandle &handle,void *stream)
{
    if(lane==PrefixAuxiliaryLane::ShiftedMTP)
        requireMTPShiftedCacheOrigin(cache.stateOwnership().shifted_origin,handle.layout.mtp_origin);
    const Geometry g(handle,lane);auto *owner=cache.auxiliaryState();validate(owner,g);if(!owner)return;
    const auto domain=g.device ? CacheAuxiliaryMemoryDomain::Device : CacheAuxiliaryMemoryDomain::Host;
    for(int ordinal=0;ordinal<g.layers;++ordinal) {
        const CacheAuxiliaryBlockRange range{prefixFALayerForIndex(cache,ordinal),sequence,g.first,g.count};
        const auto bytes=owner->blockArchiveBytes(range);
        if(bytes>g.layout.bytes_per_attention_layer)throw std::invalid_argument("prefix auxiliary import exceeds declared live stride");
        if(!bytes)continue;
        const auto *data=g.mtp ? (g.device ? handle.deviceMTPAuxiliaryBlockData(ordinal) : handle.mtpAuxiliaryBlockData(ordinal)) :
            (g.device ? handle.deviceMainAuxiliaryBlockData(ordinal) : handle.mainAuxiliaryBlockData(ordinal));
        owner->importBlock(range,{const_cast<std::uint8_t *>(data),bytes,domain},stream);
    }
    if(g.layout.includes_endpoint) {
        const auto bytes=owner->endpointArchiveBytes(sequence,g.end);
        if(bytes!=g.layout.endpoint_bytes)throw std::invalid_argument("prefix auxiliary imported endpoint differs from immutable prefix geometry");
        if(bytes) {
            const auto *data=g.mtp ? (g.device ? handle.deviceMTPAuxiliaryEndpointData() : handle.mtpAuxiliaryEndpointData()) :
                (g.device ? handle.deviceMainAuxiliaryEndpointData() : handle.mainAuxiliaryEndpointData());
            owner->importEndpoint(sequence,g.end,{const_cast<std::uint8_t *>(data),bytes,domain},stream);
        }
    }
}
}
