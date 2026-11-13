/**
 * @file Test__Qwen4ExpLookupRecordEvidence.cpp
 * @brief Request-reset lifecycle observations cannot masquerade as lookup work.
 *
 * Records use the real PerfStats schema and the exact DGO/reset and retained
 * storage producer contracts. No backend, model, reference pack or producer
 * capability is fabricated. These tests exercise the same classifier/export
 * used by the mathematical campaign after it records all original counters.
 */
#include "integration/parity/qwen4exp/Qwen4ExpLookupRecordEvidence.h"
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <vector>

namespace llaminar2::test::parity::qwen4exp
{
namespace
{
/** @return Exact metadata produced by two real reset-terminal close publications. */
PerfStatRecord resetRecord(std::string name = "reset_terminal_closes")
{
    return {.kind = PerfStatRecord::Kind::Counter, .domain = "immutable_lookup", .name = std::move(name),
        .phase = "request_reset", .device = DeviceId::cuda(0).toString(), .count = 2, .value = 2};
}
/** @return One tagged native row count, with explicit completed provider/transport identity. */
PerfStatRecord workRecord(DeviceId owner, HostImmutableRowMode mode, std::string name = "ids")
{
    return {.kind = PerfStatRecord::Kind::Counter, .domain = "immutable_lookup", .name = std::move(name),
        .phase = owner.is_cpu() ? "lookup_completion" : "generation_terminal", .device = owner.toString(),
        .tags = {{"storage", std::string(hostImmutableRowModeName(mode))},
            {"transport", owner.is_cpu() ? "cpu_direct" : "mapped_channel"}}, .count = 1, .value = 16};
}
}
/** @test The two actual untagged producers are lifecycle only, for either supported GPU backend. */
TEST(Qwen4ExpLookupRecordEvidence, NamedRequestResetRecordsAreNotLookupWork)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(1)})
        for (const auto name : {"reset_terminal_closes", "reset_terminal_waits"})
        {
            const std::array owners{device};
            auto record = resetRecord(name); record.device = device.toString();
            for (const auto mode : {HostImmutableRowMode::WholeRAM, HostImmutableRowMode::WholeMemoryMapped,
                                    HostImmutableRowMode::AdaptiveRAMSSD})
                EXPECT_EQ(classifyQwen4ExpLookupCounter(record, owners, mode), Qwen4ExpLookupCounterRole::RequestResetLifecycle);
            EXPECT_EQ(classifyQwen4ExpLookupCounter(record, owners, std::nullopt), Qwen4ExpLookupCounterRole::RequestResetLifecycle);
        }
}
/** @test Names alone cannot waive the exact reset phase, GPU owner, no-tag and unit-add contract. */
TEST(Qwen4ExpLookupRecordEvidence, MalformedOrForeignResetRecordsRemainFatal)
{
    const std::array owners{DeviceId::cuda(0), DeviceId::cpu()};
    for (int change = 0; change < 13; ++change)
    {
        SCOPED_TRACE(change);
        auto record = resetRecord();
        switch (change)
        {
        case 0: record.phase = "generation_terminal"; break;
        case 1: record.device = DeviceId::cuda(1).toString(); break;
        case 2: record.device = DeviceId::cpu().toString(); break;
        case 3: record.tags["storage"] = "mmap"; break;
        case 4: record.tags["transport"] = "mapped_channel"; break;
        case 5: record.count = 0; break;
        case 6: record.value = 0; break;
        case 7: record.value = 1; break;
        case 8: record.value = 2.5; break;
        case 9: record.value = std::numeric_limits<double>::infinity(); break;
        case 10: record.kind = PerfStatRecord::Kind::Timer; break;
        case 11: record.domain = "forward_graph"; break;
        case 12: record.name = "reset_terminal_unknown"; break;
        }
        EXPECT_THROW(classifyQwen4ExpLookupCounter(record, owners, HostImmutableRowMode::WholeMemoryMapped), std::invalid_argument);
    }
}
/** @test Real whole/adaptive CPU and hosted-GPU names retain their concrete producer identities. */
TEST(Qwen4ExpLookupRecordEvidence, HostedModesRetainKnownNamesPhaseAndExactOwner)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(1)})
        for (const auto mode : {HostImmutableRowMode::WholeRAM, HostImmutableRowMode::WholeMemoryMapped,
                               HostImmutableRowMode::AdaptiveRAMSSD})
        {
            const std::array owners{device};
            for (const auto name : {"requests", "ids", "native_bytes", "failures", "window_hits", "window_misses", "window_evictions"})
                EXPECT_EQ(classifyQwen4ExpLookupCounter(workRecord(device, mode, name), owners, mode), Qwen4ExpLookupCounterRole::LookupWork);
            if (device.is_gpu())
            {
                for (const auto name : {"intervals", "id_bytes", "metadata_bytes", "acknowledgement_bytes", "terminal_bytes"})
                    EXPECT_EQ(classifyQwen4ExpLookupCounter(workRecord(device, mode, name), owners, mode), Qwen4ExpLookupCounterRole::LookupWork);
            }
            for (const auto name : {"adaptive_batches", "ram_rows", "source_rows", "duplicate_rows", "cache_admissions", "cache_evictions", "cache_admission_rejections", "all_pinned_responses"})
            {
                const auto record = workRecord(device, mode, name);
                if (mode == HostImmutableRowMode::AdaptiveRAMSSD)
                {
                    EXPECT_EQ(classifyQwen4ExpLookupCounter(record, owners, mode), Qwen4ExpLookupCounterRole::LookupWork);
                }
                else
                {
                    EXPECT_THROW(classifyQwen4ExpLookupCounter(record, owners, mode), std::invalid_argument);
                }
            }
        }
}
/** @test Untagged, unknown and cross-mode lookup records never acquire lifecycle permission. */
TEST(Qwen4ExpLookupRecordEvidence, UnexpectedLookupRecordsCannotBeSilentlySkipped)
{
    const std::array owners{DeviceId::cuda(0)};
    for (int change = 0; change < 11; ++change)
    {
        SCOPED_TRACE(change);
        auto record = workRecord(owners[0], HostImmutableRowMode::WholeMemoryMapped);
        switch (change)
        {
        case 0: record.tags.erase("storage"); break;
        case 1: record.tags.erase("transport"); break;
        case 2: record.tags["storage"] = "ram"; break;
        case 3: record.tags["transport"] = "cpu_direct"; break;
        case 4: record.phase = "request_reset"; break;
        case 5: record.device = DeviceId::rocm(0).toString(); break;
        case 6: record.name = "future_unregistered_lookup"; break;
        case 7: record.value = -1; break;
        case 8: record.value = 0.5; break;
        case 9: record.count = 0; break;
        case 10: record.value = std::numeric_limits<double>::quiet_NaN(); break;
        }
        EXPECT_THROW(classifyQwen4ExpLookupCounter(record, owners, HostImmutableRowMode::WholeMemoryMapped), std::invalid_argument);
    }
    const std::array cpu{DeviceId::cpu()};
    EXPECT_THROW(classifyQwen4ExpLookupCounter(workRecord(cpu[0], HostImmutableRowMode::WholeRAM, "id_bytes"), cpu,
        HostImmutableRowMode::WholeRAM), std::invalid_argument);
}
/** @test Reset cannot satisfy missing byte evidence, and native equations remain strict with reset present. */
TEST(Qwen4ExpLookupRecordEvidence, LifecycleNeverSuppliesMissingNativeLookupEvidence)
{
    const std::array owners{DeviceId::cuda(0), DeviceId::cuda(1)};
    std::vector<PerfStatRecord> records{resetRecord(), resetRecord("reset_terminal_waits")};
    for (const auto &record : records)
        EXPECT_EQ(classifyQwen4ExpLookupCounter(record, owners, std::nullopt), Qwen4ExpLookupCounterRole::RequestResetLifecycle);
    EXPECT_THROW(requireQwen4ExpFixedHeadLookupEvidence(records, owners, 90), std::invalid_argument);
    constexpr std::array<double, 6> values{2, 24, 2160, 0, 16, 2160};
    for (const auto owner : owners)
        for (std::size_t field = 0; field < values.size(); ++field)
            records.push_back({.kind = PerfStatRecord::Kind::Counter, .domain = "immutable_lookup",
                .name = std::string(kFixedHeadLookupCounters[field]), .phase = "generation_terminal", .device = owner.toString(),
                .tags = {{"storage", "vram"}, {"transport", "native_fixed_head"}}, .count = 1, .value = values[field]});
    for (const auto &record : records)
        EXPECT_NO_THROW(classifyQwen4ExpLookupCounter(record, owners, std::nullopt));
    EXPECT_NO_THROW(requireQwen4ExpFixedHeadLookupEvidence(records, owners, 90));
    records.back().value += 90;
    EXPECT_THROW(requireQwen4ExpFixedHeadLookupEvidence(records, owners, 90), std::invalid_argument);
}
/** @test Export preserves named reset and malformed evidence before the validator can reject it. */
TEST(Qwen4ExpLookupRecordEvidence, RawCSVRetainsLifecycleAndMalformedRecords)
{
    auto malformed = workRecord(DeviceId::cuda(0), HostImmutableRowMode::WholeRAM);
    malformed.tags.erase("storage");
    const std::array records{resetRecord(), malformed};
    std::ostringstream csv;
    writeQwen4ExpLookupCounterCSV(csv, records);
    EXPECT_NE(csv.str().find(",,request_reset,CUDA:0,reset_terminal_closes,2\n"), std::string::npos);
    EXPECT_NE(csv.str().find(",mapped_channel,generation_terminal,CUDA:0,ids,16\n"), std::string::npos);
    const std::array owners{DeviceId::cuda(0)};
    EXPECT_THROW(classifyQwen4ExpLookupCounter(malformed, owners, HostImmutableRowMode::WholeRAM), std::invalid_argument);
}
} // namespace llaminar2::test::parity::qwen4exp
