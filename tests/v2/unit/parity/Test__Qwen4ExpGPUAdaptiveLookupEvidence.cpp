/**
 * @file Test__Qwen4ExpGPUAdaptiveLookupEvidence.cpp
 * @brief Missing, foreign or padded adaptive observations cannot certify a model cell.
 *
 * These are independent exact producer-schema examples, not native execution
 * or a replacement cache implementation. The production parity fixture consumes
 * the very same validator after preserving the original raw counter CSV.
 */
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveLookupEvidence.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <cstdlib>
#include <optional>

namespace llaminar2::test::parity::qwen4exp
{
namespace
{
/** @brief Isolate this device-free collector check without entering a backend or changing runtime policy. */
struct EvidenceCollectionScope final
{
    std::array<const char *, 2> names{"LLAMINAR_PERF_STATS_JSON", "LLAMINAR_PERF_STATS_FILTER"};
    std::array<std::optional<std::string>, 2> previous;
    /** @brief Enable all actual collection domains before authoring bounded schema observations. */
    EvidenceCollectionScope()
    {
        for (std::size_t i = 0; i < names.size(); ++i)
            if (const auto *value = std::getenv(names[i]))
                previous[i] = value;
        ::setenv(names[0], "1", 1);
        ::unsetenv(names[1]);
        PerfStatsCollector::reloadConfigurationFromEnvironment();
        PerfStatsCollector::reset();
    }
    /** @brief Retire only diagnostic collection and restore the caller's original collection policy. */
    ~EvidenceCollectionScope()
    {
        PerfStatsCollector::reset();
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            if (previous[i])
                ::setenv(names[i], previous[i]->c_str(), 1);
            else
                ::unsetenv(names[i]);
        }
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
};

/** @brief Insert schema examples through the real collector; these values do not claim native execution. */
void collectSchemaRecords(std::span<const PerfStatRecord> records)
{
    for (const auto &record : records)
        PerfStatsCollector::addCounter(record.domain, record.name, record.value,
                                       record.phase, record.device, record.tags);
}

/** @return One independently authored successful controller/source window with live unique rows and original duplicates. */
std::vector<PerfStatRecord> example(DeviceId owner=DeviceId::cuda(0),bool zero=false,bool hitsOnly=false)
{
    const std::array<std::uint64_t,9> gpu=hitsOnly?std::array<std::uint64_t,9>{3,48,4,0,0,0,0,0,0}:
        zero?std::array<std::uint64_t,9>{3,48,0,6,6,540,0,0,0}:
             std::array<std::uint64_t,9>{3,48,2,4,4,360,2,1,0};
    const std::uint64_t requests=hitsOnly?0:zero?3:2,ids=gpu[4];
    const auto metadata=requests*(::llaminar2::qwen4exp::kLookupRequestLiveBytes+::llaminar2::qwen4exp::kLookupCompletionLiveBytes)+
        ids*(sizeof(ImmutableRowSubordinateIntent)+sizeof(ImmutableRowInvalidationAcknowledgement));
    const std::array<std::uint64_t,20> source{1,requests,ids,ids*8,ids*90,metadata,requests*8,24,0,
        hitsOnly?0u:1u,hitsOnly?0u:2u,0,requests,hitsOnly||zero?0u:1u,hitsOnly?0u:zero?6u:3u,0,
        hitsOnly||zero?0u:2u,hitsOnly||zero?0u:1u,0,0};
    std::vector<PerfStatRecord> records;
    for(std::size_t i=0;i<gpu.size();++i)
        records.push_back({.kind=PerfStatRecord::Kind::Counter,.domain="immutable_lookup",
            .name=std::string(kGPUAdaptiveLookupCounters[i]),.phase="generation_terminal",.device=owner.toString(),
            .tags={{"request","0"},{"storage","vram"},{"transport","pooled_subordinate"},
                {"authority","terminal_device_generation_control"}},.count=1,.value=static_cast<double>(gpu[i])});
    for(std::size_t i=0;i<source.size();++i)
        records.push_back({.kind=PerfStatRecord::Kind::Counter,.domain="immutable_lookup",
            .name=std::string(kGPUAdaptiveSourceCounters[i]),.phase="generation_terminal",.device=owner.toString(),
            .tags={{"storage",std::string(hostImmutableRowModeName(HostImmutableRowMode::AdaptiveRAMSSD))},
                {"transport","pooled_subordinate"}},.count=1,.value=static_cast<double>(source[i])});
    return records;
}
/** @return The actual validator's complete evidence, without fabricating a provider or native source. */
Qwen4ExpGPUAdaptiveLookupEvidence validate(const std::vector<PerfStatRecord> &records,DeviceId owner=DeviceId::cuda(0),bool zero=false)
{
    return requireQwen4ExpGPUAdaptiveLookupEvidence(records,owner,90,8192,zero?0:64,zero?0:64);
}
}

/** @test Plural source projection validates every native channel once, preserving shared host ownership. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, PluralConsumerProjectionPreservesEveryChannelAndExactTotals)
{
    for (const auto type : {DeviceType::CUDA,DeviceType::ROCm})
        for (const bool zero : {false,true}) {
            const DeviceId first{type,0},second{type,1};
            const PhysicalMemoryResource host{0,DeviceId::cpu(),1ull<<34,1ull<<34};
            const std::array resources{PhysicalMemoryResource{0,second,1ull<<34,1ull<<34},
                PhysicalMemoryResource{0,first,1ull<<34,1ull<<34}};
            ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
            policy.gpu_adaptive.maximum_gpu_cache_rows=zero?0:64;
            policy.gpu_adaptive.host.maximum_cache_rows=zero?0:64;
            const auto input=ImmutableRowTablePlacementInput::forConsumers(
                ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,128,90),host,resources);
            const auto layout=input.gpuAdaptiveLayout(policy,zero?0:64,zero?0:64,8192);
            const ImmutableRowTableSource source{.rows=128,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
            auto records=example(first,zero);
            const auto peer=example(second,zero);
            records.insert(records.end(),peer.begin(),peer.end());
            const auto actual=requireQwen4ExpGPUAdaptiveCohortEvidence(records,layout,source,policy.gpu_adaptive);
            ASSERT_EQ(layout.sourceServices().size(),1u);
            ASSERT_EQ(actual.size(),2u);
            std::uint64_t total_source=0,total_bytes=0,total_intervals=0;
            for (std::size_t i=0;i<actual.size();++i) {
                const auto requester=layout.consumers()[i];
                const auto geometry=AdaptiveImmutableLookupGeometry::forRequester(layout,source,policy.gpu_adaptive,requester);
                EXPECT_EQ(geometry.physical().owners().size(),1u);
                EXPECT_EQ(geometry.physical().owners().front().allocator,requester);
                EXPECT_EQ(actual[i].source[2],zero?6u:4u);
                total_source+=actual[i].source[2];total_bytes+=actual[i].source[4];total_intervals+=actual[i].source[0];
            }
            EXPECT_EQ(total_source,zero?12u:8u);
            EXPECT_EQ(total_bytes,total_source*90u);
            EXPECT_EQ(total_intervals,2u); // Two original channel closures, not duplicated global provider stats.
            EXPECT_THROW(AdaptiveImmutableLookupGeometry::fromLayout(layout,source,policy.gpu_adaptive),std::invalid_argument);
            EXPECT_THROW(requireQwen4ExpGPUAdaptiveCohortEvidence(peer,layout,source,policy.gpu_adaptive),std::invalid_argument);
            auto foreign=records;foreign.front().device=DeviceId{type,2}.toString();
            EXPECT_THROW(requireQwen4ExpGPUAdaptiveCohortEvidence(foreign,layout,source,policy.gpu_adaptive),std::invalid_argument);
            auto doubled=records;
            for (const auto& record:peer)
                if (record.tags.at("storage")!="vram") doubled.push_back(record);
            EXPECT_THROW(requireQwen4ExpGPUAdaptiveCohortEvidence(doubled,layout,source,policy.gpu_adaptive),std::invalid_argument);
            auto padded=records;
            for (auto& record:padded) if(record.name=="metadata_bytes") {record.value+=1;break;}
            EXPECT_THROW(requireQwen4ExpGPUAdaptiveCohortEvidence(padded,layout,source,policy.gpu_adaptive),std::invalid_argument);
        }
}

/** @test A fresh real-generation reporting window preserves all unrelated evidence and the original raw forward rows. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, GenerationWindowPreservesOtherDomainsAndRawForwardEvidence)
{
    EvidenceCollectionScope collection;
    for (const auto owner : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SCOPED_TRACE(owner.toString());
        PerfStatsCollector::reset();
        auto original = example(owner);
        original.erase(original.begin(), original.begin() + 9); // Completed forward source, no generation publisher.
        collectSchemaRecords(original);
        const auto beforeRaw = PerfStatsCollector::snapshot({"immutable_lookup"});
        PerfStatsCollector::addCounterWithSequence("forward_graph", "actual_capture", 2, {7, 11}, "setup", owner.toString());
        PerfStatsCollector::recordTimingNs("forward_graph", "retained_phase_time", 31, "setup", owner.toString());
        PerfStatsCollector::recordTimingNs("forward_graph", "retained_phase_time", 47, "setup", owner.toString());
        PerfStatsCollector::addCounter("mtp", "actual_ordinary_path", 3, "decode", owner.toString());
        PerfStatsCollector::addCounter("previously_unlisted_owner", "actual_claim", 5, "setup", owner.toString());
        const std::vector<std::string> preserved{"forward_graph", "mtp", "previously_unlisted_owner"};
        const auto unrelated = PerfStatsCollector::csvString(preserved);
        const auto forward = beginQwen4ExpGPUAdaptiveGenerationEvidenceWindow();
        EXPECT_EQ(PerfStatsCollector::csvString(preserved), unrelated);
        ASSERT_EQ(forward.size(), original.size());
        std::ostringstream actualRaw, expectedRaw;
        writeQwen4ExpLookupCounterCSV(actualRaw, forward);
        writeQwen4ExpLookupCounterCSV(expectedRaw, beforeRaw);
        EXPECT_EQ(actualRaw.str(), expectedRaw.str());
        EXPECT_TRUE(PerfStatsCollector::snapshot({"immutable_lookup"}).empty());
        EXPECT_THROW((void)validate(PerfStatsCollector::snapshot({"immutable_lookup"}), owner), std::invalid_argument);

        // Independently authored metadata proves accounting isolation only;
        // the explicit native lifecycle cases own real source/GPU execution.
        const auto generation = example(owner);
        collectSchemaRecords(generation);
        const auto actual = validate(PerfStatsCollector::snapshot({"immutable_lookup"}), owner);
        EXPECT_EQ(actual.source[2], 4u);
        EXPECT_EQ(actual.device.source_rows, 4u);
        EXPECT_EQ(PerfStatsCollector::csvString(preserved), unrelated);
        auto contaminated = generation;
        contaminated.insert(contaminated.end(), original.begin(), original.end());
        EXPECT_THROW((void)validate(contaminated, owner), std::invalid_argument);
    }
}

/** @test A source-only numerical/prefix interval cannot substitute for either publisher in the following generation. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, GenerationWindowRequiresNewCompleteDeviceAndSourcePublishers)
{
    EvidenceCollectionScope collection;
    auto sourceOnly = example();
    sourceOnly.erase(sourceOnly.begin(), sourceOnly.begin() + 9);
    collectSchemaRecords(sourceOnly);
    const auto forward = beginQwen4ExpGPUAdaptiveGenerationEvidenceWindow();
    EXPECT_THROW((void)validate(forward), std::invalid_argument);
    auto deviceOnly = example();
    deviceOnly.resize(9);
    collectSchemaRecords(deviceOnly);
    EXPECT_THROW((void)validate(PerfStatsCollector::snapshot({"immutable_lookup"})), std::invalid_argument);
}
/** @test Both native backends require independent actual GPU and source tuples, including explicit C0 and all-hit U0. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, CompletePositiveZeroAndAllHitWindowsRetainExactNativeTraffic)
{
    for(const auto owner:{DeviceId::cuda(0),DeviceId::rocm(0)})for(const bool zero:{false,true})
    {
        const auto records=example(owner,zero);const auto actual=validate(records,owner,zero);
        EXPECT_EQ(actual.device.completed_live_ids,48u);EXPECT_EQ(actual.device.source_bytes,actual.source[4]);
        EXPECT_EQ(actual.device.source_rows,actual.source[2]);EXPECT_EQ(actual.source[3],actual.source[2]*8);
        EXPECT_EQ(actual.source[6],actual.source[1]*8);EXPECT_EQ(actual.source[7],24u);
        if(zero) { EXPECT_EQ(actual.device.promotions,0u); }
        else { EXPECT_EQ(actual.device.promotions,2u); }
        const auto hits=validate(example(owner,false,true),owner);
        EXPECT_EQ(hits.device.hit_rows,4u);EXPECT_EQ(hits.device.source_rows,0u);EXPECT_EQ(hits.source[1],0u);
    }
}
/** @test No absent field, including a zero failure observation, can become implicit successful work. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, EveryGPUAndSourceFieldIsMandatoryAndResetIsNotWork)
{
    const auto original=example();ASSERT_EQ(original.size(),29u);
    for(std::size_t missing=0;missing<original.size();++missing)
    {
        auto records=original;records.erase(records.begin()+static_cast<std::ptrdiff_t>(missing));
        EXPECT_THROW((void)validate(records),std::invalid_argument) << missing;
    }
    const std::vector<PerfStatRecord> reset{{.kind=PerfStatRecord::Kind::Counter,.domain="immutable_lookup",
        .name="reset_terminal_closes",.phase="request_reset",.device=DeviceId::cuda(0).toString(),.count=1,.value=1}};
    EXPECT_THROW((void)validate(reset),std::invalid_argument);
    auto withReset=original;withReset.push_back(reset.front());
    EXPECT_NO_THROW((void)validate(withReset));
}
/** @test Exact producer identity/name/phase is required; unfamiliar records never become a broad exception. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, WrongModeOwnerAuthorityPhaseAndNumericRecordsReject)
{
    for(int mutation=0;mutation<19;++mutation)
    {
        auto records=example();auto &record=records.front();
        switch(mutation)
        {
        case 0:record.tags.erase("storage");break;
        case 1:record.tags["storage"]="mmap";break;
        case 2:record.tags["transport"]="mapped_channel";break;
        case 3:record.tags.erase("authority");break;
        case 4:record.tags["authority"]="host_guess";break;
        case 5:record.tags.erase("request");break;
        case 6:record.tags["request"]="01";break;
        case 7:record.tags["request"]="-1";break;
        case 8:record.device=DeviceId::cpu().toString();break;
        case 9:record.device=DeviceId::cuda(1).toString();break;
        case 10:record.phase="lookup_completion";break;
        case 11:record.kind=PerfStatRecord::Kind::Timer;break;
        case 12:record.name="future_unknown_lookup";break;
        case 13:record.count=0;break;
        case 14:record.value=-1;break;
        case 15:record.value=1.5;break;
        case 16:record.value=std::numeric_limits<double>::quiet_NaN();break;
        case 17:record.value=std::numeric_limits<double>::infinity();break;
        case 18:record.value=9007199254740992.0;break;
        }
        EXPECT_THROW((void)validate(records),std::invalid_argument) << mutation;
    }
    for(const auto index:{9u,10u})
    {
        auto records=example();records[index].tags["transport"]="mapped_channel";
        EXPECT_THROW((void)validate(records),std::invalid_argument);
    }
    auto records=example();records.push_back(records.front());records.back().name="unexpected_native_work";
    EXPECT_THROW((void)validate(records),std::invalid_argument);
}
/** @test Compact useful traffic, source partitions and actually granted C0 semantics remain independent obligations. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, CapacityPaddingMissingWorkAndCacheSubstitutionReject)
{
    for(int mutation=0;mutation<14;++mutation)
    {
        auto records=example();
        switch(mutation)
        {
        case 0:records[8].value=1;break; // GPU failure is present, never suppressed.
        case 1:records[17].value=1;break; // CPU failure is independent.
        case 2:++records[11].value;break; // CPU source IDs disagree with validated GPU rows.
        case 3:records[12].value=8192*8;break; // Admitted ID capacity is not sent bytes.
        case 4:records[13].value=8192*90;break;
        case 5:++records[14].value;break;
        case 6:++records[15].value;break;
        case 7:records[16].value=0;break;
        case 8:++records[22].value;break; // RAM/source/duplicate partition.
        case 9:records[24].value=1;--records[23].value;break; // GPU miss compaction forbids CPU duplicate traffic.
        case 10:records[6].value=5;break; // More promotions than real source rows.
        case 11:records[1].value=47;break; // Whole token/head coordinate extent required.
        case 12:++records[3].count;break; // Same original controller observation windows.
        case 13:++records[10].count;break; // Same original CPU interval observations.
        }
        EXPECT_THROW((void)validate(records),std::invalid_argument) << mutation;
    }
    EXPECT_THROW((void)validate(example(),DeviceId::cuda(0),true),std::invalid_argument);
    EXPECT_THROW((void)requireQwen4ExpGPUAdaptiveLookupEvidence(example(),DeviceId::cuda(0),0,8192,64,64),std::invalid_argument);
    auto foreignWindow=example();foreignWindow.front().tags["request"]="1";
    EXPECT_THROW((void)validate(foreignWindow),std::invalid_argument);
}
/** @test Equal aggregate bytes cannot conceal malformed source work in different controller request windows. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, PerRequestTuplesCannotBorrowEachOthersMissingSourceBytes)
{
    auto records=example(),second=example();
    for(std::size_t i=0;i<9;++i)second[i].tags["request"]="1";
    records.insert(records.end(),second.begin(),second.end());
    EXPECT_NO_THROW((void)validate(records));
    ++records[4].value;--records[29+4].value;
    EXPECT_THROW((void)validate(records),std::invalid_argument);
}
/** @test CPU cache events cannot exceed actual source IDs, including zero-source all-hit windows. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, SourceCacheEventBoundsUseLiveIdsAndPreserveEveryRawObservation)
{
    for(const bool hitsOnly:{false,true})
        for(const auto name:{"cache_admissions","cache_admission_rejections","all_pinned_responses"})
        {
            auto records=example(DeviceId::cuda(0),false,hitsOnly);
            const auto selected=std::find_if(records.begin(),records.end(),[&](const auto &record) {
                return record.name==name;
            });
            ASSERT_NE(selected,records.end());
            const auto ids=std::find_if(records.begin(),records.end(),[](const auto &record) {
                return record.name=="ids";
            });
            ASSERT_NE(ids,records.end());
            selected->value=ids->value;
            EXPECT_NO_THROW((void)validate(records)); // Exact live bound is legal, including zero.
            selected->value=ids->value+1;
            std::ostringstream before;writeQwen4ExpLookupCounterCSV(before,records);
            EXPECT_THROW((void)validate(records),std::invalid_argument) << name;
            std::ostringstream after;writeQwen4ExpLookupCounterCSV(after,records);
            EXPECT_EQ(after.str(),before.str());
            EXPECT_EQ(selected->value,ids->value+1);EXPECT_EQ(records.size(),29u);
        }
}
/** @test The raw diagnostic CSV retains malformed producer evidence unchanged before fatal validation. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, RawCSVAndInputsRemainUntouchedOnFailure)
{
    auto records=example();records[4].value=8192;const auto value=records[4].value;
    std::ostringstream csv;writeQwen4ExpLookupCounterCSV(csv,records);
    EXPECT_NE(csv.str().find("vram,pooled_subordinate,generation_terminal,CUDA:0,adaptive_source_rows,8192\n"),std::string::npos);
    EXPECT_THROW((void)validate(records),std::invalid_argument);
    EXPECT_EQ(records[4].value,value);EXPECT_EQ(records.size(),29u);
}

/** @test Replay the completed native source tuple, including its exact worker-retirement words. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, CompletedWorkerRetirementRetainsExactTerminalBytes)
{
    // Recorded CUDA C0 Fixed1 generation witness: one source request, 32 native
    // rows at 90 bytes each, one GPU close and one CPU worker retirement pair.
    // Replaying the same schema with either consumer identity is device-free;
    // native integration remains responsible for executing each backend.
    constexpr std::array<std::uint64_t,9> device{1,32,0,32,32,2880,0,0,0};
    constexpr std::array<std::uint64_t,20> source{
        1,1,32,256,2880,992,8,24,0,32,0,0,1,0,32,0,0,0,0,0};
    for(const auto owner:{DeviceId::cuda(0),DeviceId::rocm(0)})
    {
        auto records=example(owner,true);
        for(std::size_t i=0;i<device.size();++i)records[i].value=device[i];
        for(std::size_t i=0;i<source.size();++i)records[device.size()+i].value=source[i];
        const auto actual=validate(records,owner,true);
        EXPECT_EQ(actual.source[7],24u);
        EXPECT_EQ(actual.device.source_rows,32u);
        // Missing the close or either retirement word, or including unrelated
        // cancellation/padding, must still fail the exact successful protocol.
        for(const auto bytes:{0u,8u,16u,23u,25u,32u,40u})
        {
            auto invalid=records;invalid[device.size()+7].value=bytes;
            EXPECT_THROW((void)validate(invalid,owner,true),std::invalid_argument)<<bytes;
        }
        auto second=records;
        for(std::size_t i=0;i<device.size();++i)second[i].tags["request"]="1";
        records.insert(records.end(),second.begin(),second.end());
        EXPECT_EQ(validate(records,owner,true).source[7],48u);
    }
}
} // namespace llaminar2::test::parity::qwen4exp

namespace llaminar2::test::parity::qwen4exp {
/** @test Full-RAM source observations bind actual original backing, never redundant cache/mapping work. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, FullRAMSourceRequiresExactModeAndZeroWindowWork){
    auto records=example(DeviceId::cuda(0),true);
    for(auto& record:records)if(record.tags.at("storage")!="vram"){
        record.tags["storage"]=std::string(hostImmutableRowModeName(HostImmutableRowMode::AdaptiveFullRAM));
        if(record.name.starts_with("window_"))record.value=0;
    }
    EXPECT_NO_THROW(requireQwen4ExpGPUAdaptiveLookupEvidence(records,DeviceId::cuda(0),90,8192,0,0,AdaptiveImmutableRowBacking::RAM));
    EXPECT_THROW(requireQwen4ExpGPUAdaptiveLookupEvidence(records,DeviceId::cuda(0),90,8192,0,0),std::invalid_argument);
    EXPECT_THROW(requireQwen4ExpGPUAdaptiveLookupEvidence(records,DeviceId::cuda(0),90,8192,0,1,AdaptiveImmutableRowBacking::RAM),std::invalid_argument);
    for(const auto name:{"cache_admission_rejections","all_pinned_responses"}){
        auto invalid=records;for(auto& record:invalid)if(record.name==name)record.value=1;
        EXPECT_THROW(requireQwen4ExpGPUAdaptiveLookupEvidence(invalid,DeviceId::cuda(0),90,8192,0,0,AdaptiveImmutableRowBacking::RAM),std::invalid_argument)<<name;
    }
    for(auto& record:records)if(record.name=="window_misses")record.value=1;
    EXPECT_THROW(requireQwen4ExpGPUAdaptiveLookupEvidence(records,DeviceId::cuda(0),90,8192,0,0,AdaptiveImmutableRowBacking::RAM),std::invalid_argument);
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Shared query readers have result/ACK roles, not a duplicate source/cache publisher. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, SharedCoordinatorRetainsSoleSourceAndExactReaderRoles) {
    const auto resource=[](DeviceId device){return PhysicalMemoryResource{0,device,1ull<<34,1ull<<34};};
    const std::array consumers{resource(DeviceId::cuda(0)),resource(DeviceId::cuda(1))};
    std::vector<PhysicalMemoryResource> devices(consumers.begin(),consumers.end());
    ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
    policy.gpu_adaptive.placement.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal;
    policy.gpu_adaptive.placement.coordinator=AdaptiveGPUCacheParticipant{0,DeviceType::CUDA,0};
    for(int ordinal=0;ordinal<4;++ordinal){
        devices.push_back(resource(DeviceId::rocm(ordinal)));
        policy.gpu_adaptive.placement.owners.push_back({{0,DeviceType::ROCm,ordinal},0,0});
    }
    const std::array edges{ImmutableRowTableReaderPeerAccess{{0,DeviceId::cuda(0)},{0,DeviceId::cuda(1)},PeerAccessCoverage::None}};
    const auto input=ImmutableRowTablePlacementInput::forConsumers(
        ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,128,90),resource(DeviceId::cpu()),consumers)
        .withGPUCacheResources(policy,devices).withReaderPeerAccess(edges);
    const std::vector<size_t> slots(4,0);
    const auto layout=input.gpuAdaptiveLayout(policy,slots,0,8192);
    const ImmutableRowTableSource source{.rows=128,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
    EXPECT_EQ(qwen4ExpAdaptiveExecutionRole(layout,policy.gpu_adaptive,{0,DeviceId::cuda(0)}),Qwen4ExpAdaptiveExecutionRole::SourceBoundary);
    EXPECT_EQ(qwen4ExpAdaptiveExecutionRole(layout,policy.gpu_adaptive,{0,DeviceId::cuda(1)}),Qwen4ExpAdaptiveExecutionRole::QueryReader);
    EXPECT_THROW(qwen4ExpAdaptiveExecutionRole(layout,policy.gpu_adaptive,{1,DeviceId::cuda(1)}),std::invalid_argument);
    auto missing=policy.gpu_adaptive;missing.placement.coordinator.reset();
    EXPECT_THROW(qwen4ExpAdaptiveExecutionRole(layout,missing,{0,DeviceId::cuda(0)}),std::invalid_argument);
    const auto check=[&](const auto& records){return requireQwen4ExpGPUAdaptiveCohortEvidence(records,layout,source,policy.gpu_adaptive);};
    const auto records=example(DeviceId::cuda(0),true);
    const auto actual=check(records);ASSERT_EQ(actual.size(),1u);EXPECT_EQ(actual.front().device.completed_live_ids,48u);
    auto zero_reader=records;
    for(const auto& original:records)if(original.tags.at("storage")=="vram"){
        auto record=original;record.device=DeviceId::cuda(1).toString();record.value=0;zero_reader.push_back(std::move(record));
    }
    EXPECT_NO_THROW(check(zero_reader));
    auto forged=zero_reader;forged.back().value=1;EXPECT_THROW(check(forged),std::invalid_argument);
    auto partial=zero_reader;partial.pop_back();EXPECT_THROW(check(partial),std::invalid_argument);
    auto wrong_count=zero_reader;wrong_count.back().count=2;EXPECT_THROW(check(wrong_count),std::invalid_argument);
    auto wrong_request=zero_reader;for(auto& record:wrong_request)if(record.device==DeviceId::cuda(1).toString())record.tags["request"]="1";
    EXPECT_THROW(check(wrong_request),std::invalid_argument);
    auto duplicate_source=records;
    for(const auto& original:records)if(original.tags.at("storage")!="vram"){
        auto record=original;record.device=DeviceId::cuda(1).toString();duplicate_source.push_back(std::move(record));
    }
    EXPECT_THROW(check(duplicate_source),std::invalid_argument);
    auto foreign=zero_reader;foreign.back().device=DeviceId::cuda(2).toString();EXPECT_THROW(check(foreign),std::invalid_argument);
    auto no_coordinator=zero_reader;std::erase_if(no_coordinator,[](const auto& record){return record.device==DeviceId::cuda(0).toString();});
    EXPECT_THROW(check(no_coordinator),std::invalid_argument);
    auto ram_policy=policy;ram_policy.gpu_adaptive.source_backing=AdaptiveImmutableRowBacking::RAM;
    const auto ram_input=ImmutableRowTablePlacementInput::forConsumers(
        ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,128,90),resource(DeviceId::cpu()),consumers)
        .withGPUCacheResources(ram_policy,devices).withReaderPeerAccess(edges);
    const auto ram_layout=ram_input.gpuAdaptiveLayout(ram_policy,slots,0,8192);
    auto ram_records=records;
    for(auto& record:ram_records){
        if(record.tags.at("storage")!="vram")record.tags["storage"]=std::string(hostImmutableRowModeName(HostImmutableRowMode::AdaptiveFullRAM));
        if(record.name=="window_hits" || record.name=="window_misses" || record.name=="window_evictions")record.value=0;
    }
    EXPECT_NO_THROW(requireQwen4ExpGPUAdaptiveCohortEvidence(ram_records,ram_layout,source,ram_policy.gpu_adaptive));
    ram_records.back().tags["storage"]=std::string(hostImmutableRowModeName(HostImmutableRowMode::AdaptiveRAMSSD));
    EXPECT_THROW(requireQwen4ExpGPUAdaptiveCohortEvidence(ram_records,ram_layout,source,ram_policy.gpu_adaptive),std::invalid_argument);
}

/** @test Actual embedded expert/inline-reader owners never count as independent native enqueues. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, NativeSubmissionsExcludeActualEmbeddedOwners) {
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(0)};
    std::vector<Qwen4ExpAdaptiveOwnerExecution> owners{{coordinator,true},{{0,DeviceId::cuda(1)},true}};
    for(int ordinal=0;ordinal<4;++ordinal)owners.push_back({{0,DeviceId::rocm(ordinal)},true});
    std::vector<PerfStatRecord> records;
    for(const auto name:{"retained_source_native_invocations","retained_remote_native_enqueues"})
        records.push_back({.kind=PerfStatRecord::Kind::Counter,.domain="ngram_lookup",.name=name,
            .phase="hosted_transaction",.device=coordinator.device.toString(),.tags={{"policy","retained_hosted"}},
            .count=3,.value=name==std::string_view("retained_source_native_invocations")?3.:0.});
    EXPECT_NO_THROW(requireQwen4ExpAdaptiveNativeSubmissions(records,coordinator,owners));
    auto forged=records;forged.back().value=15;EXPECT_THROW(requireQwen4ExpAdaptiveNativeSubmissions(forged,coordinator,owners),std::invalid_argument);
    auto missing=records;missing.erase(missing.begin());EXPECT_THROW(requireQwen4ExpAdaptiveNativeSubmissions(missing,coordinator,owners),std::invalid_argument);
    auto wrong_count=records;wrong_count.back().count=2;EXPECT_THROW(requireQwen4ExpAdaptiveNativeSubmissions(wrong_count,coordinator,owners),std::invalid_argument);
    auto foreign=records;foreign.front().device=DeviceId::cuda(1).toString();EXPECT_THROW(requireQwen4ExpAdaptiveNativeSubmissions(foreign,coordinator,owners),std::invalid_argument);
    owners.back().embedded=false;records.back().value=3;
    EXPECT_NO_THROW(requireQwen4ExpAdaptiveNativeSubmissions(records,coordinator,owners));
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Cold attachment or a foreign/fractional route cannot substitute for every follower's completed useful work. */
TEST(Qwen4ExpGPUAdaptiveLookupEvidence, EveryEmbeddedExpertRequiresOriginalCompletedUsefulWork) {
    const auto definition=qwen4ExpHardwareTargetDefinition(Qwen4ExpHardwareTarget::CUDA2ROCm4,Qwen4ExpHardwareNgramOwners::Follower);
    const auto& plan=*definition.topology.expert_overlay_plan;
    const auto participants=MoEExpertOwnerMap::declaredParticipants(plan,{.reject_tensor_sharded_domains=false});
    std::vector<PhysicalMemoryAllocatorIdentity> owners;
    std::vector<PerfStatRecord> records;
    for(const auto& participant:participants)if(participant.device.is_rocm()){
        owners.push_back({participant.world_rank,participant.device});
        records.push_back({.kind=PerfStatRecord::Kind::Counter,.domain="forward_graph",.name="moe_overlay_local_expert_active_routes",
            .phase="moe_overlay",.device=participant.device.toString(),.tags={{"completion","device_owned_epoch_complete"},
            {"identity_source","device_owned_activation_epoch"},{"device_kind","ROCm"},{"input_rows","1"},{"output_rows","1"},
            {"participant",std::to_string(participant.participant_id)},{"tier",std::to_string(participant.tier_idx)},
            {"service_source","prefill"}},.count=1,.value=1});
        records.push_back({.kind=PerfStatRecord::Kind::Counter,.domain="moe_overlay",.name="device_epoch_gpu_rows",
            .phase="gn_local_expert",.device=participant.device.toString(),.tags={{"completion","device_owned_epoch_complete"},
            {"domain_kind","GPU"},{"participant",std::to_string(participant.participant_id)},
            {"tier",std::to_string(participant.tier_idx)},{"transport","local"}},.count=1,.value=1});
    }
    ASSERT_EQ(owners.size(),4u);
    const auto check=[&](const auto& rows){requireQwen4ExpEmbeddedExpertWork(rows,plan,owners);};
    EXPECT_NO_THROW(check(records));
    auto missing=records;missing.pop_back();EXPECT_THROW(check(missing),std::invalid_argument);
    auto cold=records;cold.front().tags["completion"]="attached_only";EXPECT_THROW(check(cold),std::invalid_argument);
    auto fractional=records;fractional.front().value=.5;EXPECT_THROW(check(fractional),std::invalid_argument);
    auto foreign=records;foreign.front().device=DeviceId::cuda(0).toString();EXPECT_THROW(check(foreign),std::invalid_argument);
    auto wrong_tier=records;wrong_tier.front().tags["tier"]="99";EXPECT_THROW(check(wrong_tier),std::invalid_argument);
    auto empty=records;empty.front().value=0;EXPECT_THROW(check(empty),std::invalid_argument);
    auto no_rows=records;no_rows.front().tags["input_rows"]="0";EXPECT_THROW(check(no_rows),std::invalid_argument);
    auto foreign_owners=owners;foreign_owners.front().world_rank=1;
    EXPECT_THROW(requireQwen4ExpEmbeddedExpertWork(records,plan,foreign_owners),std::invalid_argument);
}
}
