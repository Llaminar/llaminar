/**
 * @file Qwen4ExpGPUAdaptiveLookupEvidence.h
 * @brief Closed terminal GPU-cache and subordinate-source proof for canonical model diagnostics.
 *
 * GPU probes/promotions come only from the actual canonical device-controller
 * publisher. CPU observations independently prove compact protocol2 source
 * traffic. Neither owner can substitute for missing work of the other. The
 * caller exports raw CSV first; malformed records are retained, never filtered
 * or relabeled as whole mmap evidence.
 */
#pragma once
#include "Qwen4ExpLookupRecordEvidence.h"
#include "execution/qwen4exp/ImmutableLookupProtocol.h"
#include "loaders/CpuSubordinateImmutableRowRead.h"
#include "kernels/qwen4exp/QwenAdaptiveTerminalCounters.h"
#include "utils/PerfStatsCollector.h"
#include <bitset>
#include <cstdint>
#include <map>

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Exact producer metric order, checked against the actual device terminal tuple. */
inline constexpr std::array<std::string_view,9> kGPUAdaptiveLookupCounters{
    "adaptive_lookup_calls","adaptive_completed_live_ids","adaptive_cache_hit_rows",
    "adaptive_cache_miss_rows","adaptive_source_rows","adaptive_source_native_bytes",
    "adaptive_promotions","adaptive_evictions","adaptive_failed_frames"};
/** @brief Complete protocol2 source publisher fields, including zero observations. */
inline constexpr std::array<std::string_view,20> kGPUAdaptiveSourceCounters{
    "intervals","requests","ids","id_bytes","native_bytes","metadata_bytes","acknowledgement_bytes",
    "terminal_bytes","failures","window_hits","window_misses","window_evictions","adaptive_batches",
    "ram_rows","source_rows","duplicate_rows","cache_admissions","cache_evictions",
    "cache_admission_rejections","all_pinned_responses"};
/** @brief Only these three independently authenticated publishers may appear in this mode. */
enum class GPUAdaptiveLookupCounterRole { DeviceTerminal, SubordinateSource, RequestResetLifecycle };

/**
 * @brief Separate completed forward diagnostics from one real generation's source/device reporting window.
 * @return Unmodified immutable-lookup records from the completed diagnostic requests, for a separate raw CSV.
 * @details The fixture first retires the public request and authenticates the
 * idle original owner. Only collection storage is reset here; the service's
 * monotonic publication guard, warmed cache, native controller and graphs are
 * untouched. Every other observed domain is retained exactly, including
 * numerical-path, prefix, placement and graph-capture evidence.
 */
inline std::vector<PerfStatRecord> beginQwen4ExpGPUAdaptiveGenerationEvidenceWindow()
{
    const auto original = PerfStatsCollector::snapshot();
    std::vector<std::string> preserved;
    std::vector<PerfStatRecord> forward;
    for (const auto &record : original)
    {
        if (record.domain == "immutable_lookup")
            forward.push_back(record);
        else if (std::find(preserved.begin(), preserved.end(), record.domain) == preserved.end())
            preserved.push_back(record.domain);
    }
    PerfStatsCollector::resetPreservingDomains(preserved);
    return forward;
}
/**
 * @brief Authenticate one actual terminal record without altering its tags or byte values.
 * @param record Original PerfStats counter. @param consumer Sole resolved GPU PLE consumer.
 * @return Exact closed producer class.
 * @throws std::invalid_argument For unknown/foreign, malformed, wrong-phase or relabeled records.
 */
inline GPUAdaptiveLookupCounterRole classifyQwen4ExpGPUAdaptiveLookupCounter(
    const PerfStatRecord &record,DeviceId consumer)
{
    if(record.name=="reset_terminal_closes" || record.name=="reset_terminal_waits")
    {
        const std::array owners{consumer};
        (void)classifyQwen4ExpLookupCounter(record,owners,std::nullopt);
        return GPUAdaptiveLookupCounterRole::RequestResetLifecycle;
    }
    if(!consumer.is_valid() || !consumer.is_gpu() || record.device!=consumer.toString() ||
        record.domain!="immutable_lookup" || record.kind!=PerfStatRecord::Kind::Counter ||
        record.phase!="generation_terminal" || !record.count || record.count>9007199254740991ULL ||
        !std::isfinite(record.value) || record.value<0 || std::floor(record.value)!=record.value ||
        record.value>9007199254740991.0)
        throw std::invalid_argument("GPU adaptive lookup requires an exact completed counter from its sole native consumer");
    const auto storage=record.tags.find("storage"),transport=record.tags.find("transport");
    if(storage==record.tags.end() || transport==record.tags.end() || transport->second!="pooled_subordinate")
        throw std::invalid_argument("GPU adaptive counter lost its actual subordinate transport identity");
    if(storage->second=="vram")
    {
        const auto authority=record.tags.find("authority"),request=record.tags.find("request");
        if(record.tags.size()!=4 || authority==record.tags.end() || authority->second!="terminal_device_generation_control" ||
            request==record.tags.end() || request->second.empty() ||
            (request->second.size()>1 && request->second.front()=='0') ||
            !std::all_of(request->second.begin(),request->second.end(),[](char c){return c>='0' && c<='9';}) ||
            std::find(kGPUAdaptiveLookupCounters.begin(),kGPUAdaptiveLookupCounters.end(),record.name)==kGPUAdaptiveLookupCounters.end())
            throw std::invalid_argument("GPU adaptive device work requires its original controller/request publisher and known name");
        return GPUAdaptiveLookupCounterRole::DeviceTerminal;
    }
    if(storage->second!=hostImmutableRowModeName(HostImmutableRowMode::AdaptiveRAMSSD) || record.tags.size()!=2 ||
        std::find(kGPUAdaptiveSourceCounters.begin(),kGPUAdaptiveSourceCounters.end(),record.name)==kGPUAdaptiveSourceCounters.end())
        throw std::invalid_argument("GPU adaptive source work requires its actual closed CPU provider and known name");
    return GPUAdaptiveLookupCounterRole::SubordinateSource;
}
/** @brief Immutable completed evidence; source fields remain independent of the GPU reporting tuple. */
struct Qwen4ExpGPUAdaptiveLookupEvidence final
{
    ::llaminar2::qwen4exp::QwenAdaptiveCounterTotals device;
    std::array<std::uint64_t,20> source{};
};
/**
 * @brief Require complete actual work and exact compact native/protocol traffic.
 * @param records Raw collected immutable-domain observations.
 * @param consumer Sole resolved GPU requester. @param row_bytes Exact original native stride.
 * @param maximum_ids Original admitted per-frame ID bound, never a communication extent.
 * @param gpu_slots Actual admitted GPU slots. @param ram_slots Actual admitted CPU slots.
 * @return Checked independent device/source totals, without mutating records.
 * @throws std::invalid_argument For missing tuple fields, invalid identities, errors, byte equations or overflow.
 * @note Positive cache churn is not required. C0 must never report cache hits/promotions/evictions.
 */
inline Qwen4ExpGPUAdaptiveLookupEvidence requireQwen4ExpGPUAdaptiveLookupEvidence(
    std::span<const PerfStatRecord> records,DeviceId consumer,std::size_t row_bytes,
    std::uint64_t maximum_ids,std::uint64_t gpu_slots,std::uint64_t ram_slots)
{
    struct DeviceWindow { std::array<std::uint64_t,9> values{};std::array<std::uint64_t,9> counts{};std::bitset<9> seen; };
    std::map<std::string,DeviceWindow> windows;
    std::array<std::uint64_t,20> source{},source_counts{};std::bitset<20> seen;
    const auto add=[](std::uint64_t &out,std::uint64_t value) {
        if(value>UINT64_MAX-out)throw std::invalid_argument("GPU adaptive evidence sum overflows");out+=value;
    };
    const auto multiply=[](std::uint64_t a,std::uint64_t b) {
        if(b && a>UINT64_MAX/b)throw std::invalid_argument("GPU adaptive evidence byte product overflows");return a*b;
    };
    for(const auto &record:records)
    {
        if(record.domain!="immutable_lookup")continue;
        const auto role=classifyQwen4ExpGPUAdaptiveLookupCounter(record,consumer);
        if(role==GPUAdaptiveLookupCounterRole::RequestResetLifecycle)continue;
        const auto value=static_cast<std::uint64_t>(record.value);
        if(role==GPUAdaptiveLookupCounterRole::DeviceTerminal)
        {
            const auto index=static_cast<std::size_t>(std::find(kGPUAdaptiveLookupCounters.begin(),kGPUAdaptiveLookupCounters.end(),record.name)-kGPUAdaptiveLookupCounters.begin());
            auto &window=windows[record.tags.at("request")];add(window.values[index],value);add(window.counts[index],record.count);window.seen.set(index);
        }
        else
        {
            const auto index=static_cast<std::size_t>(std::find(kGPUAdaptiveSourceCounters.begin(),kGPUAdaptiveSourceCounters.end(),record.name)-kGPUAdaptiveSourceCounters.begin());
            add(source[index],value);add(source_counts[index],record.count);seen.set(index);
        }
    }
    std::array<std::uint64_t,9> total{};
    if(windows.empty() || !seen.all())throw std::invalid_argument("GPU adaptive evidence is missing an actual device/source publisher");
    for(const auto &[request,window]:windows)
    {
        if(!window.seen.all())throw std::invalid_argument("GPU adaptive request has an incomplete terminal tuple: "+request);
        const auto &v=window.values;
        const ::llaminar2::qwen4exp::QwenAdaptiveCounterTotals actual{
            v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]};
        if(!::llaminar2::qwen4exp::validQwenAdaptiveCounters(actual,row_bytes,maximum_ids) || actual.failed_frames)
            throw std::invalid_argument("GPU adaptive request has an invalid/error terminal tuple: "+request);
        for(std::size_t i=0;i<9;++i)
        {
            if(window.counts[i]!=window.counts[0])throw std::invalid_argument("GPU adaptive terminal tuple intervals disagree");
            add(total[i],window.values[i]);
        }
    }
    const ::llaminar2::qwen4exp::QwenAdaptiveCounterTotals device{
        total[0],total[1],total[2],total[3],total[4],total[5],total[6],total[7],total[8]};
    if(!::llaminar2::qwen4exp::validQwenAdaptiveCounters(device,row_bytes,maximum_ids) ||
        !device.calls || !device.completed_live_ids || device.failed_frames)
        throw std::invalid_argument("GPU adaptive device terminal lacks complete successful original-ID work");
    for(const auto count:source_counts)
        if(count!=source_counts.front())throw std::invalid_argument("GPU adaptive source tuple intervals disagree");
    const auto &[intervals,requests,ids,id_bytes,native,metadata,acks,terminal,failures,window_hits,window_misses,
        window_evictions,batches,ram_rows,source_rows,duplicates,admissions,evictions,rejections,all_pinned]=source;
    (void)window_hits;(void)window_misses;(void)window_evictions;
    const auto headers=multiply(requests,::llaminar2::qwen4exp::kLookupRequestLiveBytes+::llaminar2::qwen4exp::kLookupCompletionLiveBytes);
    const auto intents=multiply(ids,sizeof(ImmutableRowSubordinateIntent)+sizeof(ImmutableRowInvalidationAcknowledgement));
    auto expected_metadata=headers;add(expected_metadata,intents);
    auto partition=ram_rows;add(partition,source_rows);add(partition,duplicates);
    if(!intervals || intervals!=source_counts.front() || requests>device.calls || failures || batches!=requests ||
        ids!=device.source_rows || native!=device.source_bytes || native!=multiply(ids,row_bytes) ||
        id_bytes!=multiply(ids,sizeof(std::int64_t)) || metadata!=expected_metadata ||
        acks!=multiply(requests,sizeof(std::uint64_t)) || terminal!=multiply(intervals,sizeof(std::uint64_t)) ||
        // Same per-live-row bounds enforced by accumulateHostImmutableRowWork;
        // source cache events cannot be justified by admitted frame capacity.
        partition!=ids || duplicates || admissions>ids || rejections>ids || all_pinned>ids || evictions>admissions ||
        (!gpu_slots && (device.hit_rows || device.promotions || device.evictions)) ||
        (!ram_slots && (ram_rows || admissions || evictions)))
        throw std::invalid_argument("GPU adaptive compact source/cache evidence disagrees with exact live native work");
    return {device,source};
}
} // namespace llaminar2::test::parity::qwen4exp
