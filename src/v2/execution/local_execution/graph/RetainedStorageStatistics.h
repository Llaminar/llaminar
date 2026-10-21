/**
 * @file RetainedStorageStatistics.h
 * @brief Immutable terminal observations of the retained CPU storage protocol.
 *
 * Counters are produced by the I/O authority and sampled only after its worker
 * joins. Interval subtraction prevents request reuse and measurement resets from
 * recounting a channel's lifetime totals. These values contain no model state.
 */
#pragma once
#include "../../../loaders/HostImmutableRowProvider.h"
#include <limits>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
/** @brief Exact completed I/O work and wire bytes, independent of allocation capacity. */
struct RetainedStorageCounters
{
    std::uint64_t requests=0,ids=0,idBytes=0,nativeBytes=0,metadataBytes=0,failures=0;
    std::uint64_t acknowledgementBytes=0,terminalBytes=0;
    // Successful gathers only, matching ids/nativeBytes. A failed partial
    // gather can still alter the table's lifetime mapping counters; these
    // observations never claim that failure performed zero cache/OS work.
    std::uint64_t windowHits=0,windowMisses=0,windowEvictions=0;
    std::uint64_t adaptiveBatches=0,ramRows=0,sourceRows=0,duplicateRows=0,cacheAdmissions=0,cacheEvictions=0;
    std::uint64_t cacheAdmissionRejections=0,allPinnedResponses=0;
};
/**
 * @brief Add one successful provider observation to its actual caller's counters.
 * @param totals Caller/service-owned observations; never shared provider lifetime totals.
 * @param mode Actual immutable provider alternative.
 * @param work Successful native-byte observation issued inside the provider operation.
 * @param rows Exact completed requested rows.
 * @param bytes Exact completed native bytes, including duplicate response positions.
 * @throws std::exception For mode/extent drift or observation overflow.
 */
inline void accumulateHostImmutableRowWork(RetainedStorageCounters &totals,HostImmutableRowMode mode,
    const HostImmutableRowGatherObservation &work,std::uint64_t rows,std::uint64_t bytes)
{
    (void)hostImmutableRowModeName(mode);
    RetainedStorageCounters next=totals;
    const auto add=[](std::uint64_t &a,std::uint64_t b) {
        if(b>std::numeric_limits<std::uint64_t>::max()-a)throw std::overflow_error("Host native row observation overflow");
        a+=b;
    };
    ImmutableRowGatherObservation windows;
    if(const auto *whole=std::get_if<WholeImmutableRowGatherObservation>(&work))
    {
        const auto expected=whole->storage==ImmutableRowStorage::RAM?HostImmutableRowMode::WholeRAM:
            whole->storage==ImmutableRowStorage::MemoryMapped?HostImmutableRowMode::WholeMemoryMapped:
            throw std::invalid_argument("Host whole observation has a non-host concrete tier");
        if(mode!=expected || whole->rows!=rows || whole->bytes!=bytes)
            throw std::invalid_argument("Host whole native row observation mode/extent changed");
        windows=whole->windows;
    }
    else
    {
        const auto &cache=std::get<CpuImmutableRowGatherObservation>(work);
        if(mode!=HostImmutableRowMode::AdaptiveRAMSSD || cache.rows!=rows || cache.bytes!=bytes)
            throw std::invalid_argument("Adaptive native row observation mode/extent changed");
        if(cache.ram_rows>rows || cache.source_rows>rows-cache.ram_rows ||
            cache.duplicate_rows!=rows-cache.ram_rows-cache.source_rows || cache.all_pinned_responses>rows ||
            cache.admissions>rows || cache.admission_rejections>rows || cache.evictions>cache.admissions)
            throw std::invalid_argument("Adaptive row work does not partition the completed caller response");
        windows=cache.windows;add(next.adaptiveBatches,1);
        add(next.ramRows,cache.ram_rows);add(next.sourceRows,cache.source_rows);add(next.duplicateRows,cache.duplicate_rows);
        add(next.cacheAdmissions,cache.admissions);add(next.cacheEvictions,cache.evictions);
        add(next.cacheAdmissionRejections,cache.admission_rejections);add(next.allPinnedResponses,cache.all_pinned_responses);
    }
    if(mode==HostImmutableRowMode::WholeRAM && (windows.windowHits || windows.windowMisses || windows.windowEvictions))
        throw std::invalid_argument("Whole RAM observation cannot claim mapping work");
    add(next.windowHits,windows.windowHits);add(next.windowMisses,windows.windowMisses);add(next.windowEvictions,windows.windowEvictions);
    totals=next;
}
/** @brief Actual source-channel protocol; native cache decisions remain
 * separately device-owned. */
enum class RetainedStorageTransport : std::uint8_t
{
    MappedChannel,      ///< Ordinary host-provider requests with no GPU cache.
    AdaptiveSubordinate ///< CPU protocol2 source work beneath one GPU-owned
                        ///< native cache.
};
/** @brief One completed storage interval, safe to publish without GPU observation. */
struct RetainedStorageTerminalObservation
{
    std::uint64_t generation=0; ///< Monotonic channel I/O interval, never a model loop decision.
    HostImmutableRowMode mode=HostImmutableRowMode::WholeRAM;
    RetainedStorageCounters counters;
    RetainedStorageTransport transport =
        RetainedStorageTransport::MappedChannel; ///< Exact executed transport,
                                                 ///< not requested policy.

    /** @brief Authenticate monotonic cumulative counters and retain only this interval.
     * @param generation Positive channel interval that has completed its worker join.
     * @param mode Actual immutable owner implementation, without automatic mode changes.
     * @param before I/O authority snapshot before this interval was armed.
     * @param after Same authority's snapshot after its terminal worker join.
     * @return Exact per-interval observation, including zero-work terminal closes.
     */
    static RetainedStorageTerminalObservation between(std::uint64_t generation,HostImmutableRowMode mode,
        const RetainedStorageCounters &before,const RetainedStorageCounters &after)
    {
        (void)hostImmutableRowModeName(mode);
        if(!generation)
            throw std::invalid_argument("Retained storage statistics require a positive interval and explicit storage policy");
        RetainedStorageTerminalObservation result{generation,mode,{}};
        constexpr std::array members{&RetainedStorageCounters::requests,&RetainedStorageCounters::ids,
            &RetainedStorageCounters::idBytes,&RetainedStorageCounters::nativeBytes,
            &RetainedStorageCounters::metadataBytes,&RetainedStorageCounters::failures,
            &RetainedStorageCounters::acknowledgementBytes,&RetainedStorageCounters::terminalBytes,
            &RetainedStorageCounters::windowHits,&RetainedStorageCounters::windowMisses,
            &RetainedStorageCounters::windowEvictions,&RetainedStorageCounters::adaptiveBatches,
            &RetainedStorageCounters::ramRows,&RetainedStorageCounters::sourceRows,&RetainedStorageCounters::duplicateRows,
            &RetainedStorageCounters::cacheAdmissions,&RetainedStorageCounters::cacheEvictions,
            &RetainedStorageCounters::cacheAdmissionRejections,&RetainedStorageCounters::allPinnedResponses};
        for(auto member:members)
        {
            if(after.*member<before.*member)throw std::logic_error("Retained storage cumulative counter regressed");
            result.counters.*member=after.*member-before.*member;
        }
        return result;
    }
};
} // namespace llaminar2
