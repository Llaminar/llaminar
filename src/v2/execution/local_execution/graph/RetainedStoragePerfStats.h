/**
 * @file RetainedStoragePerfStats.h
 * @brief Bounded PerfStats publication of already joined immutable storage intervals.
 *
 * Each interval is consumed exactly once even when diagnostics are disabled.
 * Generation identities enter ordered numeric evidence, never map keys. A reset
 * of PerfStats therefore cannot replay an earlier interval's cumulative work.
 */
#pragma once
#include "RetainedStorageStatistics.h"
#include "../../../loaders/ImmutableRowTable.h"
#include "../../../backends/DeviceId.h"
#include "utils/PerfStatsCollector.h"
#include <string>

namespace llaminar2
{
/** @brief Direct CPU gather/decode completed or failed before publishing usable rows. */
enum class RetainedStorageCPUOutcome { Completed,Failed };
/** @brief Publish synchronous CPU-native I/O work after the stage completes.
 * @param mode Actual retained host implementation, without erasing adaptive work to mmap.
 * @param device Exact CPU execution participant label.
 * @param ids Successfully gathered and decoded native row count; zero on failure.
 * @param nativeBytes Exact successful native gather bytes; zero on failure.
 * @param outcome Whether this completed CPU attempt produced usable rows.
 * @param work This call's successful gather observation, never a shared
 *        table-lifetime delta. Failure carries no completed observation, even
 *        when the table performed partial cache work before throwing.
 * @details CPU work has no mapped-channel ID, metadata or acknowledgement
 * transport. Each completed call contributes its own work, so collector reset
 * cannot cause a later call to republish an authority's lifetime totals.
 */
inline void recordCPUImmutableLookup(HostImmutableRowMode mode,DeviceId device,
    std::uint64_t ids,std::uint64_t nativeBytes,RetainedStorageCPUOutcome outcome,
    std::optional<HostImmutableRowGatherObservation> work=std::nullopt)
{
    (void)hostImmutableRowModeName(mode);
    if(outcome!=RetainedStorageCPUOutcome::Completed&&outcome!=RetainedStorageCPUOutcome::Failed)
        throw std::invalid_argument("CPU native row evidence has an unknown outcome");
    if((outcome==RetainedStorageCPUOutcome::Failed&&(ids||nativeBytes||work)) ||
       (outcome==RetainedStorageCPUOutcome::Completed&&!work))
        throw std::invalid_argument("CPU native row evidence requires its exact successful observation or a failed zero-work result");
    RetainedStorageCounters observed;
    if(work)accumulateHostImmutableRowWork(observed,mode,*work,ids,nativeBytes);
    if(!PerfStatsCollector::isDomainEnabled("immutable_lookup"))return;
    const PerfStatsCollector::Tags tags{{"storage",std::string(hostImmutableRowModeName(mode))},{"transport","cpu_direct"}};
    const auto record=[&](const char *name,std::uint64_t value)
    {PerfStatsCollector::addCounter("immutable_lookup",name,static_cast<double>(value),"lookup_completion",device.toString(),tags);};
    record("requests",1);record("ids",ids);record("native_bytes",nativeBytes);record("failures",outcome==RetainedStorageCPUOutcome::Failed?1:0);
    record("window_hits",observed.windowHits);record("window_misses",observed.windowMisses);record("window_evictions",observed.windowEvictions);
    if(mode==HostImmutableRowMode::AdaptiveRAMSSD)
    {
    record("adaptive_batches",observed.adaptiveBatches);record("ram_rows",observed.ramRows);record("source_rows",observed.sourceRows);
    record("duplicate_rows",observed.duplicateRows);record("cache_admissions",observed.cacheAdmissions);
    record("cache_evictions",observed.cacheEvictions);record("cache_admission_rejections",observed.cacheAdmissionRejections);
    record("all_pinned_responses",observed.allPinnedResponses);
    }
}
/** @brief Participant-owned freshness guard for terminal I/O evidence only. */
class RetainedStoragePerfStats final
{
public:
    /** @brief Publish one newer immutable observation at the existing terminal boundary.
     * @param observation Per-interval counters obtained after the I/O worker joined.
     * @param device Stable owning GPU/participant label.
     * @throws std::logic_error If an interval is stale or already projected.
     */
    void publish(const RetainedStorageTerminalObservation &observation,const std::string &device)
    {
        if(!observation.generation||observation.generation<=generation_)
            throw std::logic_error("Retained storage terminal evidence is stale or already published");
        (void)hostImmutableRowModeName(observation.mode);
        if(observation.mode==HostImmutableRowMode::WholeRAM&&(observation.counters.windowHits||
            observation.counters.windowMisses||observation.counters.windowEvictions))
            throw std::invalid_argument("RAM lookup evidence cannot claim mapping-window work");
        if (observation.transport != RetainedStorageTransport::MappedChannel &&
            observation.transport != RetainedStorageTransport::AdaptiveSubordinate &&
            observation.transport != RetainedStorageTransport::PooledSubordinate)
            throw std::invalid_argument("Retained storage evidence has an unknown executed transport");
        if ((observation.transport == RetainedStorageTransport::AdaptiveSubordinate ||
             observation.transport == RetainedStorageTransport::PooledSubordinate) &&
            observation.mode != HostImmutableRowMode::AdaptiveRAMSSD)
            throw std::invalid_argument("Subordinate source work requires its "
                                        "actual adaptive host owner");
        generation_=observation.generation;
        if(!PerfStatsCollector::isDomainEnabled("immutable_lookup"))return;
        const PerfStatsCollector::Tags tags{
            {"storage", std::string(hostImmutableRowModeName(observation.mode))},
            {"transport", observation.transport == RetainedStorageTransport::AdaptiveSubordinate?"adaptive_subordinate":
                observation.transport == RetainedStorageTransport::PooledSubordinate?"pooled_subordinate":"mapped_channel"}};
        const auto record=[&](const char *name,std::uint64_t value)
        {
            PerfStatsCollector::addCounterWithSequence("immutable_lookup",name,static_cast<double>(value),
                {observation.generation,value},"generation_terminal",device,tags);
        };
        const auto &c=observation.counters;
        record("intervals",1);record("requests",c.requests);record("ids",c.ids);
        record("id_bytes",c.idBytes);record("native_bytes",c.nativeBytes);
        record("metadata_bytes",c.metadataBytes);record("acknowledgement_bytes",c.acknowledgementBytes);
        record("terminal_bytes",c.terminalBytes);record("failures",c.failures);
        record("window_hits",c.windowHits);record("window_misses",c.windowMisses);
        record("window_evictions",c.windowEvictions);
        if(observation.mode==HostImmutableRowMode::AdaptiveRAMSSD)
        {
        record("adaptive_batches",c.adaptiveBatches);record("ram_rows",c.ramRows);record("source_rows",c.sourceRows);
        record("duplicate_rows",c.duplicateRows);record("cache_admissions",c.cacheAdmissions);record("cache_evictions",c.cacheEvictions);
        record("cache_admission_rejections",c.cacheAdmissionRejections);record("all_pinned_responses",c.allPinnedResponses);
        }
    }
private:
    std::uint64_t generation_=0; ///< Last consumed observation; not live protocol/model authority.
};
} // namespace llaminar2
