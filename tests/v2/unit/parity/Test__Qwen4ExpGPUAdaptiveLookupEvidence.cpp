/**
 * @file Test__Qwen4ExpGPUAdaptiveLookupEvidence.cpp
 * @brief Missing, foreign or padded adaptive observations cannot certify a model cell.
 *
 * These are independent exact producer-schema examples, not native execution
 * or a replacement cache implementation. The production parity fixture consumes
 * the very same validator after preserving the original raw counter CSV.
 */
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveLookupEvidence.h"
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
    const std::array<std::uint64_t,20> source{1,requests,ids,ids*8,ids*90,metadata,requests*8,8,0,
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
        EXPECT_EQ(actual.source[6],actual.source[1]*8);EXPECT_EQ(actual.source[7],8u);
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
} // namespace llaminar2::test::parity::qwen4exp
