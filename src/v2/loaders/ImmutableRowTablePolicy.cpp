/**
 * @file ImmutableRowTablePolicy.cpp
 * @brief Validate requested policy without consulting capacity, payload or runtime allocation.
 *
 * Automatic is a planning preference. This validator cannot resolve a mode or
 * touch the table; PMA owns that decision after higher-priority obligations.
 */
#include "ImmutableRowTablePolicy.h"
#include <limits>
#include <algorithm>
#include <charconv>
#include <string_view>
#include <stdexcept>
#include <unistd.h>
namespace llaminar2
{
void AdaptiveGPUCachePlacementRequest::validate() const
{
    if(mode==AdaptiveGPUCachePlacementMode::ConsumerLocal)
    {
        if(!owners.empty() || coordinator)
            throw std::invalid_argument("ConsumerLocal GPU cache placement cannot carry declared owners/coordinator");
        return;
    }
    if(mode!=AdaptiveGPUCachePlacementMode::DeclaredRankLocal || owners.empty() || !coordinator)
        throw std::invalid_argument("DeclaredRankLocal GPU cache placement needs ordered owners and exact coordinator");
    if(std::none_of(owners.begin(),owners.end(),[&](const auto &o){return o.participant==*coordinator;}))
        throw std::invalid_argument("Declared GPU cache coordinator requires its own explicit owner ceiling, including C0");
    const auto valid=[](const AdaptiveGPUCacheParticipant &p) { return p.world_rank>=0 && p.device().is_gpu(); };
    if(!valid(*coordinator))throw std::invalid_argument("GPU cache coordinator must be an exact rank-local CUDA/ROCm endpoint");
    for(std::size_t i=0;i<owners.size();++i)
    {
        const auto &o=owners[i];
        if(!valid(o.participant) || o.participant.world_rank!=coordinator->world_rank || o.maximum_cache_rows>0x7fffffffULL)
            throw std::invalid_argument("GPU cache owner requires bounded slots on the coordinator rank; cross-rank materialization is unimplemented");
        for(std::size_t j=0;j<i;++j)if(owners[j].participant==o.participant)
            throw std::invalid_argument("GPU cache owner declaration duplicates a physical endpoint");
        if(o.participant!=*coordinator && o.participant.backend==coordinator->backend)
            throw std::invalid_argument("Same-vendor remote GPU cache materialization is unimplemented; no native P2P substitute is admitted");
    }
}
std::vector<AdaptiveGPUCacheParticipant> AdaptiveGPUCachePlacementRequest::declaredParticipants() const
{
    validate();std::vector<AdaptiveGPUCacheParticipant> result;
    for(const auto &owner:owners)result.push_back(owner.participant);
    if(coordinator && std::find(result.begin(),result.end(),*coordinator)==result.end())result.push_back(*coordinator);
    return result;
}
AdaptiveGPUCachePlacementRequest parseAdaptiveGPUCachePlacement(const std::string &text)
{
    if(text=="consumer-local")return {};
    const auto number=[](std::string_view text) -> std::uint64_t {
        std::uint64_t value=0;
        const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
        if(text.empty() || error!=std::errc{} || end!=text.data()+text.size())
            throw std::invalid_argument("GPU cache placement requires exact nonnegative decimal fields");
        return value;
    };
    const auto endpoint=[&](std::string_view text) {
        const auto a=text.find(':'),b=a==text.npos?text.npos:text.find(':',a+1);
        if(a==text.npos || b==text.npos || text.find(':',b+1)!=text.npos)
            throw std::invalid_argument("GPU cache participant requires rank:cuda|rocm:ordinal");
        const auto rank=number(text.substr(0,a)),ordinal=number(text.substr(b+1));
        const auto backend=text.substr(a+1,b-a-1);
        if(rank>static_cast<std::uint64_t>(std::numeric_limits<int>::max()) || ordinal>static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
           (backend!="cuda" && backend!="rocm"))throw std::invalid_argument("GPU cache participant rank/backend/ordinal is invalid");
        return AdaptiveGPUCacheParticipant{static_cast<int>(rank),backend=="cuda"?DeviceType::CUDA:DeviceType::ROCm,static_cast<int>(ordinal)};
    };
    const std::string_view input=text;const auto at=input.find('@');
    if(at==input.npos || input.find('@',at+1)!=input.npos)
        throw std::invalid_argument("Declared GPU cache placement requires ordered owners@coordinator");
    AdaptiveGPUCachePlacementRequest result;result.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal;
    result.coordinator=endpoint(input.substr(at+1));
    auto remaining=input.substr(0,at);
    do
    {
        const auto comma=remaining.find(',');const auto owner=remaining.substr(0,comma);
        const auto a=owner.find('/'),b=a==owner.npos?owner.npos:owner.find('/',a+1);
        if(a==owner.npos || b==owner.npos || owner.find('/',b+1)!=owner.npos)
            throw std::invalid_argument("GPU cache owner requires rank:backend:ordinal/rows/bytes");
        const auto bytes=number(owner.substr(b+1));
        if(bytes>std::numeric_limits<std::size_t>::max())throw std::invalid_argument("GPU cache owner byte ceiling overflows size_t");
        result.owners.push_back({endpoint(owner.substr(0,a)),number(owner.substr(a+1,b-a-1)),static_cast<std::size_t>(bytes)});
        if(comma==remaining.npos)break;
        remaining.remove_prefix(comma+1);
    }while(true);
    result.validate();return result;
}
/** @brief Concrete mode construction validates the shared lossless cache geometry once. */
ImmutableRowTableStoragePolicy ImmutableRowTableStoragePolicy::forStorage(
    ImmutableRowStorage storage, std::size_t window_bytes, std::size_t window_count)
{
    ImmutableRowStoragePreference preference;
    switch (storage)
    {
    case ImmutableRowStorage::VRAM: preference = ImmutableRowStoragePreference::VRAM; break;
    case ImmutableRowStorage::RAM: preference = ImmutableRowStoragePreference::RAM; break;
    case ImmutableRowStorage::MemoryMapped: preference = ImmutableRowStoragePreference::MemoryMapped; break;
    default: throw std::invalid_argument("unknown concrete immutable row storage");
    }
    ImmutableRowTablePolicy{preference, window_bytes, window_count}.validate();
    return ImmutableRowTableStoragePolicy(storage, window_bytes, window_count);
}
bool immutableRowLookupUsesNativeDeviceStorage(ImmutableRowLookupExecution execution)
{
    switch (execution)
    {
    case ImmutableRowLookupExecution::HostProvider: return false;
    case ImmutableRowLookupExecution::ResidentGPU:
    case ImmutableRowLookupExecution::FixedHeadShardGPU: return true;
    case ImmutableRowLookupExecution::AdaptiveGPUCache:
        throw std::invalid_argument("GPU adaptive lookup is a mixed owner contract, not whole/range native storage");
    }
    throw std::invalid_argument("Unknown ngram lookup execution in native owner classification");
}
ImmutableRowLookupExecution immutableRowLookupExecution(ImmutableRowStorage storage)
{
    switch (storage)
    {
    case ImmutableRowStorage::RAM:
    case ImmutableRowStorage::MemoryMapped: return ImmutableRowLookupExecution::HostProvider;
    case ImmutableRowStorage::VRAM: return ImmutableRowLookupExecution::ResidentGPU;
    }
    throw std::invalid_argument("Unknown concrete ngram tier in lookup execution projection");
}
/** @brief Validate every lossless requested policy field before planning. */
void ImmutableRowTablePolicy::validate() const
{
    if(distribution!=ImmutableRowTableDistribution::Replicated &&
       distribution!=ImmutableRowTableDistribution::FixedHeadSharded)
        throw std::invalid_argument("Unknown immutable row distribution");
    if(isFixedHeadSharded() && storage!=ImmutableRowStoragePreference::VRAM)
        throw std::invalid_argument("FixedHeadSharded requires explicit VRAM; Automatic and hosted tiers are unsupported");
    switch (storage)
    {
    case ImmutableRowStoragePreference::Automatic:
    case ImmutableRowStoragePreference::VRAM:
    case ImmutableRowStoragePreference::RAM:
    case ImmutableRowStoragePreference::MemoryMapped:
    case ImmutableRowStoragePreference::HostAdaptiveRAMSSD:
    case ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD:
        break;
    default:
        throw std::invalid_argument("unknown immutable row storage preference");
    }
    if (!isHostAdaptive() && host_adaptive != HostAdaptiveRAMSSDRequest{})
        throw std::invalid_argument("Whole-tier ngrams cannot carry adaptive cache request fields");
    if (isHostAdaptive() &&
        (host_adaptive.maximum_cache_rows >= std::numeric_limits<std::uint32_t>::max() ||
         !host_adaptive.policy.aging_accesses || !host_adaptive.policy.protected_denominator ||
         host_adaptive.policy.protected_numerator > host_adaptive.policy.protected_denominator))
        throw std::invalid_argument("HostAdaptiveRAMSSD requires bounded uint32 slots and a valid frozen cache policy");
    if (!isGPUAdaptive() && gpu_adaptive != AdaptiveVRAMRAMSSDRequest{})
        throw std::invalid_argument("Other ngram modes cannot carry GPU adaptive request fields");
    if (isGPUAdaptive())
    {
        gpu_adaptive.placement.validate();
        if(gpu_adaptive.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal &&
            (gpu_adaptive.maximum_gpu_cache_rows || gpu_adaptive.gpu_cache_physical_byte_limit))
            throw std::invalid_argument("Declared GPU cache owners cannot mix ConsumerLocal scalar GPU ceilings");
        const auto validPolicy=[](const CpuImmutableRowCachePolicy &p) {
            return p.aging_accesses && p.protected_denominator && p.protected_numerator<=p.protected_denominator;
        };
        if(gpu_adaptive.maximum_gpu_cache_rows>0x7fffffffULL ||
            gpu_adaptive.host.maximum_cache_rows>=std::numeric_limits<std::uint32_t>::max() ||
            !validPolicy(gpu_adaptive.gpu_policy) || !validPolicy(gpu_adaptive.host.policy))
            throw std::invalid_argument("AdaptiveVRAMRAMSSD needs bounded GPU/host slots and explicit valid policies");
    }
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || window_bytes == 0 ||
        window_bytes % static_cast<std::size_t>(page_size) != 0)
        throw std::invalid_argument("immutable row mmap window must be page-aligned and nonzero");
    if (window_count == 0 || window_count > std::numeric_limits<std::size_t>::max() / window_bytes)
        throw std::invalid_argument("immutable row mmap window count must be positive without budget overflow");
}
}
