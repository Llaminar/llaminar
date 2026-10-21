/**
 * @file ImmutableRowTablePolicy.cpp
 * @brief Validate requested policy without consulting capacity, payload or runtime allocation.
 *
 * Automatic is a planning preference. This validator cannot resolve a mode or
 * touch the table; PMA owns that decision after higher-priority obligations.
 */
#include "ImmutableRowTablePolicy.h"
#include <limits>
#include <stdexcept>
#include <unistd.h>
namespace llaminar2
{
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
