/**
 * @file Qwen4ExpGPUAdaptiveLookupEvidence.h
 * @brief Closed terminal GPU-cache and subordinate-source proof for canonical model diagnostics.
 *
 * Shared replicated readers retain one coordinator/source reporting tuple;
 * their original result/ACK role cannot be relabeled as another CPU source.
 * GPU probes/promotions come only from the actual canonical device-controller
 * publisher. CPU observations independently prove compact protocol2 source
 * traffic. Neither owner can substitute for missing work of the other. The
 * caller exports raw CSV first; malformed records are retained, never filtered
 * or relabeled as whole mmap evidence.
 */
#pragma once
#include "Qwen4ExpLookupRecordEvidence.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "execution/qwen4exp/ImmutableLookupProtocol.h"
#include "loaders/CpuSubordinateImmutableRowRead.h"
#include "kernels/qwen4exp/QwenAdaptiveTerminalCounters.h"
#include "utils/PerfStatsCollector.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include <bitset>
#include <charconv>
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
    if((storage->second!=hostImmutableRowModeName(HostImmutableRowMode::AdaptiveRAMSSD) &&
        storage->second!=hostImmutableRowModeName(HostImmutableRowMode::AdaptiveFullRAM)) || record.tags.size()!=2 ||
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
 * @param backing Exact original host backing; full RAM cannot claim bounded-cache/mmap work.
 * @return Checked independent device/source totals, without mutating records.
 * @throws std::invalid_argument For missing tuple fields, invalid identities, errors, byte equations or overflow.
 * @note Positive cache churn is not required. C0 must never report cache hits/promotions/evictions.
 */
inline Qwen4ExpGPUAdaptiveLookupEvidence requireQwen4ExpGPUAdaptiveLookupEvidence(
    std::span<const PerfStatRecord> records,DeviceId consumer,std::size_t row_bytes,
    std::uint64_t maximum_ids,std::uint64_t gpu_slots,std::uint64_t ram_slots,
    AdaptiveImmutableRowBacking backing=AdaptiveImmutableRowBacking::SSD)
{
    if(backing!=AdaptiveImmutableRowBacking::SSD && backing!=AdaptiveImmutableRowBacking::RAM)
        throw std::invalid_argument("GPU adaptive evidence has unknown original backing");
    if(backing==AdaptiveImmutableRowBacking::RAM && ram_slots)
        throw std::invalid_argument("Full-RAM source evidence cannot claim redundant RAM cache slots");
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
            const auto expected=backing==AdaptiveImmutableRowBacking::RAM?HostImmutableRowMode::AdaptiveFullRAM:HostImmutableRowMode::AdaptiveRAMSSD;
            if(record.tags.at("storage")!=hostImmutableRowModeName(expected))throw std::invalid_argument("GPU source observation differs from original admitted backing");
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
        // Successful service completion publishes the GPU close generation and
        // the CPU worker's retired generation/invocation pair. Direct processor
        // close-only observations do not certify a retired serving interval.
        acks!=multiply(requests,sizeof(std::uint64_t)) || terminal!=multiply(intervals,3*sizeof(std::uint64_t)) ||
        // Same per-live-row bounds enforced by accumulateHostImmutableRowWork;
        // source cache events cannot be justified by admitted frame capacity.
        partition!=ids || duplicates || admissions>ids || rejections>ids || all_pinned>ids || evictions>admissions ||
        (!gpu_slots && (device.hit_rows || device.promotions || device.evictions)) ||
        (!ram_slots && (ram_rows || admissions || evictions)) ||
        (backing==AdaptiveImmutableRowBacking::RAM && (window_hits || window_misses || window_evictions || rejections || all_pinned)))
        throw std::invalid_argument("GPU adaptive compact source/cache evidence disagrees with exact live native work");
    return {device,source};
}
/** @brief Actual lookup role; a replicated query reader never owns another source service. */
enum class Qwen4ExpAdaptiveExecutionRole { SourceBoundary, QueryReader };
/**
 * @brief Resolve the original source/reader role without creating any execution authority.
 * @param layout Original admitted physical geometry. @param request Original policy.
 * @param consumer Exact original rank/device consumer, not an enumeration ordinal.
 * @return Source boundary for independent consumers/coordinator; reader for shared noncoordinator.
 * @throws std::invalid_argument For absent/foreign consumer or coordinator.
 */
inline Qwen4ExpAdaptiveExecutionRole qwen4ExpAdaptiveExecutionRole(
    const ImmutableRowTableLayoutGeometry& layout,const AdaptiveVRAMRAMSSDRequest& request,
    PhysicalMemoryAllocatorIdentity consumer)
{
    if(std::find(layout.consumers().begin(),layout.consumers().end(),consumer)==layout.consumers().end())
        throw std::invalid_argument("Adaptive runtime evidence has a foreign reader");
    if(request.placement.mode!=AdaptiveGPUCachePlacementMode::DeclaredRankLocal)
        return Qwen4ExpAdaptiveExecutionRole::SourceBoundary;
    if(!request.placement.coordinator)throw std::invalid_argument("Shared adaptive evidence has no original coordinator");
    const PhysicalMemoryAllocatorIdentity coordinator{request.placement.coordinator->world_rank,request.placement.coordinator->device()};
    if(std::find(layout.consumers().begin(),layout.consumers().end(),coordinator)==layout.consumers().end())
        throw std::invalid_argument("Shared adaptive evidence coordinator is not an original reader");
    return consumer==coordinator?Qwen4ExpAdaptiveExecutionRole::SourceBoundary:Qwen4ExpAdaptiveExecutionRole::QueryReader;
}
/** @brief Actual original participant disposition observed cold from retained participants, never inferred from cache capacity. */
struct Qwen4ExpAdaptiveOwnerExecution {
    PhysicalMemoryAllocatorIdentity allocator;
    bool embedded; ///< Original hasEmbeddedParticipant result; includes inline readers and expert parents.
};
/**
 * @brief Validate real successful source submissions against original standalone versus embedded ownership.
 * @param records Existing immutable PerfStats snapshot, not synthetic request counts.
 * @param coordinator Original source device. @param owners Actual retained physical owner dispositions.
 * @throws std::invalid_argument For absent, foreign, malformed or incorrectly attributed submissions.
 * @note Embedded ownership alone proves no work; expert parents additionally require completed expert evidence.
 */
inline void requireQwen4ExpAdaptiveNativeSubmissions(std::span<const PerfStatRecord> records,
    PhysicalMemoryAllocatorIdentity coordinator,std::span<const Qwen4ExpAdaptiveOwnerExecution> owners)
{
    std::uint64_t standalone=0,source=0,remote=0,source_count=0,remote_count=0;bool source_seen=false,remote_seen=false;
    if(owners.empty())throw std::invalid_argument("Adaptive submission evidence lost its original owner inventory");
    for(size_t i=0;i<owners.size();++i){
        if(owners[i].allocator.world_rank!=coordinator.world_rank ||
           std::find_if(owners.begin(),owners.begin()+i,[&](const auto& value){return value.allocator==owners[i].allocator;})!=owners.begin()+i)
            throw std::invalid_argument("Adaptive submission evidence changed original owner identity");
        if(owners[i].allocator.device!=coordinator.device && !owners[i].embedded)++standalone;
    }
    for(const auto& record:records){
        if(record.domain!="ngram_lookup" || (record.name!="retained_source_native_invocations" && record.name!="retained_remote_native_enqueues"))continue;
        if(record.kind!=PerfStatRecord::Kind::Counter || record.device!=coordinator.device.toString() ||
           record.phase!="hosted_transaction" || record.tags!=PerfStatsCollector::Tags{{"policy","retained_hosted"}} ||
           !record.count || !std::isfinite(record.value) || record.value<0 || std::floor(record.value)!=record.value || record.value>9007199254740991.0)
            throw std::invalid_argument("Adaptive submissions lost their exact original native publisher");
        const auto value=static_cast<std::uint64_t>(record.value);
        auto& total=record.name=="retained_source_native_invocations"?source:remote;
        if(value>UINT64_MAX-total)throw std::invalid_argument("Adaptive submission total overflows");total+=value;
        auto& count=record.name=="retained_source_native_invocations"?source_count:remote_count;
        if(record.count>UINT64_MAX-count)throw std::invalid_argument("Adaptive submission count overflows");count+=record.count;
        if(record.name=="retained_source_native_invocations")source_seen=true;else remote_seen=true;
    }
    if(!source_seen || !remote_seen || !source || source_count!=source || remote_count!=source_count || (standalone && source>UINT64_MAX/standalone) || remote!=source*standalone)
        throw std::invalid_argument("Adaptive source/standalone enqueue evidence differs from actual embedded roles");
}
/**
 * @brief Require useful completed native expert work on every original embedded expert owner.
 * @param records Existing completed campaign snapshot; capture declarations/attachments do not count.
 * @param plan SAME original routed topology. @param owners Exact expert lookup-owner rank/device set.
 * @throws std::invalid_argument For unknown/foreign ownership, missing work or invalid completion records.
 */
inline void requireQwen4ExpEmbeddedExpertWork(std::span<const PerfStatRecord> records,
    const MoERoutedExpertPlacementPlan& plan,std::span<const PhysicalMemoryAllocatorIdentity> owners)
{
    const auto participants=MoEExpertOwnerMap::declaredParticipants(plan,{.reject_tensor_sharded_domains=false});
    for(size_t coordinate=0;coordinate<owners.size();++coordinate){
        const auto owner=owners[coordinate];
        if(std::find(owners.begin(),owners.begin()+coordinate,owner)!=owners.begin()+coordinate)
            throw std::invalid_argument("Expert useful-work evidence repeats an original owner");
        const auto participant=std::find_if(participants.begin(),participants.end(),[&](const auto& value){
            return value.world_rank_known && value.world_rank==owner.world_rank && value.device==owner.device &&
                value.domain_name!=plan.base_model_domain && value.domain_name!=plan.continuation_domain;});
        if(participant==participants.end())throw std::invalid_argument("Expert useful-work owner is foreign or unresolved");
        std::uint64_t routes=0,rows=0;
        for(const auto& record:records){
            const bool route=record.domain=="forward_graph" && record.name=="moe_overlay_local_expert_active_routes";
            const bool returned=record.domain=="moe_overlay" && record.name=="device_epoch_gpu_rows";
            if(!route && !returned)continue;
            const auto p=record.tags.find("participant");
            if(p==record.tags.end() || p->second!=std::to_string(participant->participant_id))continue;
            const auto tag=[&](const char* name,const std::string& expected){const auto found=record.tags.find(name);return found!=record.tags.end() && found->second==expected;};
            const auto positive_rows=[&](const char* name){
                const auto found=record.tags.find(name);if(found==record.tags.end() || found->second.empty() || found->second.front()=='0')return false;
                std::uint64_t value=0;const auto& text=found->second;
                const auto result=std::from_chars(text.data(),text.data()+text.size(),value);
                return result.ec==std::errc{} && result.ptr==text.data()+text.size() && value>0;
            };
            if(record.kind!=PerfStatRecord::Kind::Counter || record.device!=owner.device.toString() || !record.count ||
               !std::isfinite(record.value) || record.value<=0 || std::floor(record.value)!=record.value || record.value>9007199254740991.0 ||
               !tag("completion","device_owned_epoch_complete") || !tag("tier",std::to_string(participant->tier_idx)) ||
               (route && (record.phase!="moe_overlay" || !positive_rows("input_rows") || !positive_rows("output_rows") || !tag("identity_source","device_owned_activation_epoch") ||
                   !tag("device_kind",owner.device.is_cuda()?"CUDA":"ROCm") ||
                   !(tag("service_source","prefill") || tag("service_source","decode") || tag("service_source","grouped_verifier")))) ||
               (returned && (record.phase!="gn_local_expert" || !tag("domain_kind","GPU") || !tag("transport","local"))))
                throw std::invalid_argument("Expert useful-work evidence is not an original completed native endpoint");
            auto& total=route?routes:rows;const auto value=static_cast<std::uint64_t>(record.value);
            if(value>UINT64_MAX-total)throw std::invalid_argument("Expert useful-work total overflows");total+=value;
        }
        if(!routes || !rows || routes<rows)throw std::invalid_argument("Original embedded expert owner has no useful completed work");
    }
}
/**
 * @brief Validate each original requester exactly once without duplicating shared source observations.
 * @param records Unmodified terminal records, including every actual channel.
 * @param layout Original resolved plural or singleton placement.
 * @param source Original native table metadata, without payload reads.
 * @param request Original GPU/shared RAM policy.
 * @return Independent channel totals in consumer order, or ONE shared coordinator/source tuple.
 * @throws std::invalid_argument For foreign/duplicate consumers, missing channels or invalid compact byte equations.
 *
 * CPU source work is published by its original processor channel and tagged
 * with that requester. Partition those records once; never copy the shared
 * provider's cumulative statistics into each device's evidence. Shared reader
 * completion is authenticated separately through the original result/ACK roots
 * and coordinator retirement; optional reader reporting words claim no cache
 * work. This record validator never manufactures another source observation.
 */
inline std::vector<Qwen4ExpGPUAdaptiveLookupEvidence> requireQwen4ExpGPUAdaptiveCohortEvidence(
    std::span<const PerfStatRecord> records, const ImmutableRowTableLayoutGeometry& layout,
    const ImmutableRowTableSource& source, const AdaptiveVRAMRAMSSDRequest& request)
{
    const auto& consumers = layout.consumers();
    if (consumers.empty()) throw std::invalid_argument("GPU adaptive evidence has no original consumers");
    const bool shared=request.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal && consumers.size()>1;
    if(shared){
        if(!request.placement.coordinator)throw std::invalid_argument("Shared adaptive evidence has no coordinator");
        const PhysicalMemoryAllocatorIdentity coordinator{request.placement.coordinator->world_rank,request.placement.coordinator->device()};
        (void)qwen4ExpAdaptiveExecutionRole(layout,request,coordinator);
        std::vector<PerfStatRecord> source_records;
        // Query readers may project zero-valued reporting words, but they may
        // never publish another CPU source tuple or claim coordinator work.
        struct ReaderWindow {std::bitset<9> seen;std::array<std::uint64_t,9> counts{};};
        std::map<std::pair<std::string,std::string>,ReaderWindow> reader_windows;
        std::map<std::string,std::uint64_t> coordinator_windows;
        for(const auto& record:records){
            if(record.domain!="immutable_lookup")continue;
            if(record.device==coordinator.device.toString()){
                source_records.push_back(record);
                if(record.name=="adaptive_lookup_calls"){
                    const auto request_tag=record.tags.find("request");
                    if(request_tag!=record.tags.end()){
                        auto& count=coordinator_windows[request_tag->second];
                        if(record.count>UINT64_MAX-count)throw std::invalid_argument("Coordinator reporting count overflows");count+=record.count;
                    }
                }
                continue;
            }
            const auto consumer=std::find_if(consumers.begin(),consumers.end(),[&](const auto& value){return value.device.toString()==record.device;});
            if(consumer==consumers.end())throw std::invalid_argument("Shared adaptive evidence has a foreign reader");
            const auto role=classifyQwen4ExpGPUAdaptiveLookupCounter(record,consumer->device);
            if(role==GPUAdaptiveLookupCounterRole::RequestResetLifecycle)continue;
            if(role!=GPUAdaptiveLookupCounterRole::DeviceTerminal || record.value!=0)
                throw std::invalid_argument("Query-only reader claimed duplicate shared source/cache work");
            const auto metric=std::find(kGPUAdaptiveLookupCounters.begin(),kGPUAdaptiveLookupCounters.end(),record.name)-kGPUAdaptiveLookupCounters.begin();
            auto& window=reader_windows[{record.device,record.tags.at("request")}];
            window.seen.set(static_cast<size_t>(metric));
            auto& count=window.counts[static_cast<size_t>(metric)];
            if(record.count>UINT64_MAX-count)throw std::invalid_argument("Query reader reporting count overflows");count+=record.count;
        }
        for(const auto& [key,window]:reader_windows){
            if(!window.seen.all())throw std::invalid_argument("Query reader terminal tuple is incomplete");
            const auto coordinator_count=coordinator_windows.find(key.second);
            if(coordinator_count==coordinator_windows.end())throw std::invalid_argument("Query reader reported a foreign request window");
            for(const auto count:window.counts)
                if(count!=coordinator_count->second)throw std::invalid_argument("Query reader/coordinator terminal windows disagree");
        }
        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(layout,source,request);
        std::uint64_t slots=0;
        for(const auto& owner:geometry.physical().owners()){
            if(owner.slots>UINT64_MAX-slots)throw std::invalid_argument("Shared admitted slot observation overflows");slots+=owner.slots;
        }
        // ONE source tuple. Original result/ACK roots and coordinator retirement
        // authenticate readers in the caller; no per-reader source is invented.
        return {requireQwen4ExpGPUAdaptiveLookupEvidence(source_records,coordinator.device,
            layout.extent().rowBytes(),layout.maximumInflightRows(),slots,geometry.host().slots(),request.source_backing)};
    }
    std::vector<std::vector<PerfStatRecord>> channels(consumers.size());
    for (std::size_t i = 0; i < consumers.size(); ++i) {
        if (!consumers[i].device.is_gpu() ||
            std::find(consumers.begin(), consumers.begin() + i, consumers[i]) != consumers.begin() + i)
            throw std::invalid_argument("GPU adaptive evidence repeats or changes its original consumer");
    }
    for (const auto& record : records) {
        if (record.domain != "immutable_lookup") continue;
        const auto consumer = std::find_if(consumers.begin(), consumers.end(), [&](const auto& candidate) {
            return record.device == candidate.device.toString();
        });
        if (consumer == consumers.end()) throw std::invalid_argument("GPU adaptive evidence contains a foreign channel");
        channels[static_cast<std::size_t>(consumer - consumers.begin())].push_back(record);
    }
    std::vector<Qwen4ExpGPUAdaptiveLookupEvidence> result;
    for (std::size_t i = 0; i < consumers.size(); ++i) {
        const auto geometry = layout.adaptiveGPUDomain() == ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal
            ? AdaptiveImmutableLookupGeometry::forRequester(layout, source, request, consumers[i])
            : AdaptiveImmutableLookupGeometry::fromLayout(layout, source, request);
        result.push_back(requireQwen4ExpGPUAdaptiveLookupEvidence(channels[i], consumers[i].device,
            layout.extent().rowBytes(), layout.maximumInflightRows(),
            geometry.physical().owners().front().slots, geometry.host().slots(),request.source_backing));
    }
    return result;
}
} // namespace llaminar2::test::parity::qwen4exp
