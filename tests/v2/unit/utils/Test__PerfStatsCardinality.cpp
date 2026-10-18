/**
 * @file Test__PerfStatsCardinality.cpp
 * @brief Long-session counter evidence stays bounded by participants and roles.
 *
 * Epochs and token positions vary through independent simulated coding sessions.
 * The real collector must retain exact counter totals and ordered witnesses
 * without retaining one map node per transaction. Production physical-fabric
 * and idle-controller publishers are exercised directly with changing lifecycle
 * values. No devices or models are used.
 */
#include "utils/PerfStatsCollector.h"
#include "execution/moe/DeviceMoERebalancePerfStats.h"
#include "execution/moe/MoEPhysicalResidencyPerfStats.h"
#include "execution/moe/MoEOverlayControllerPerfStats.h"
#include "execution/moe/NativeMoEMovementPerfStats.h"
#include "utils/ControllerMovementEvidence.h"
#include "utils/ControllerMovementFixture.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdlib>
#include <optional>
#include <thread>

namespace llaminar2
{
namespace
{
    /** @brief Own the process-wide collection switch for one serial native test. */
    class PerfStatsCardinality : public ::testing::Test
    {
    protected:
        /** @brief Enable all domains explicitly, preserving the caller's environment. */
        void SetUp() override
        {
            for (const char *key : {"LLAMINAR_PERF_STATS_JSON", "LLAMINAR_PERF_STATS_FILTER"})
            {
                const char *value = std::getenv(key);
                environment_[key] = value ? std::optional<std::string>(value) : std::nullopt;
            }
            ::setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
            ::unsetenv("LLAMINAR_PERF_STATS_FILTER");
            PerfStatsCollector::reset();
        }
        /** @brief Retire retained counters and restore the immutable environment policy. */
        void TearDown() override
        {
            for (const auto &[key, value] : environment_)
                if (value) ::setenv(key.c_str(), value->c_str(), 1);
                else ::unsetenv(key.c_str());
            PerfStatsCollector::reset();
        }
        std::map<std::string, std::optional<std::string>> environment_;
    };
}

TEST_F(PerfStatsCardinality, LongSessionsKeepCountersAndWitnessesInConstantRows)
{
    constexpr uint64_t sessions = 20000;
    constexpr uint64_t initial_epoch = (uint64_t{1} << 54); // Preserve integers beyond exact double range.
    for (uint64_t session = 0; session < sessions; ++session)
        for (const char *device : {"CPU:0", "CUDA:0", "ROCm:0"})
            for (const char *role : {"producer", "consumer"})
                PerfStatsCollector::addCounterWithSequence(
                    "mtp", "device_resident_mtp_transaction_fences", 3.0,
                    {initial_epoch + session, session * 15, session % 4096},
                    "decode", device, {{"role", role}});
    const auto records = PerfStatsCollector::snapshot();
    ASSERT_EQ(records.size(), 6u);
    for (const auto &record : records)
    {
        EXPECT_EQ(record.kind, PerfStatRecord::Kind::Counter);
        EXPECT_EQ(record.count, sessions);
        EXPECT_EQ(record.value, sessions * 3.0);
        EXPECT_EQ(record.sequence_word_count, sessions * 3);
        EXPECT_EQ(record.sequence_digest_lo, records.front().sequence_digest_lo);
        EXPECT_EQ(record.sequence_digest_hi, records.front().sequence_digest_hi);
        EXPECT_EQ(record.tags.size(), 1u);
        EXPECT_EQ(record.sequence_minimum_words, (std::vector<uint64_t>{initial_epoch, 0, 0}));
        EXPECT_EQ(record.sequence_maximum_words, (std::vector<uint64_t>{initial_epoch + sessions - 1, (sessions - 1) * 15, 4095}));
    }
    const auto document = nlohmann::json::parse(PerfStatsCollector::jsonString());
    ASSERT_EQ(document["records"].size(), 6u);
    EXPECT_EQ(document["records"][0]["sequence_word_count"], sessions * 3);
    EXPECT_EQ(document["records"][0]["sequence_minimum_words"].get<std::vector<uint64_t>>(),
        records.front().sequence_minimum_words);
    EXPECT_EQ(document["records"][0]["sequence_maximum_words"].get<std::vector<uint64_t>>(),
        records.front().sequence_maximum_words);
    const auto csv = PerfStatsCollector::csvString();
    EXPECT_NE(csv.find(",\"" + nlohmann::json(records.front().sequence_minimum_words).dump() + "\""),
        std::string::npos);
    EXPECT_LT(document.dump().size(), 6000u);
}

TEST_F(PerfStatsCardinality, ControllerMovementKeepsExactHistoryInConstantRows)
{
    constexpr std::uint64_t first = std::uint64_t{1} << 54;
    MoEOptimizationMovementLedger ledger;
    for (std::uint64_t wave = 0; wave < 512; ++wave)
    {
        auto completed = test::controllerMovementFixture(first + wave, first + wave, wave % 3);
        ledger.edges.insert(ledger.edges.end(), completed.edges.begin(), completed.edges.end());
        ledger.economy.push_back(completed.economy.front());
        ledger.device_publications.push_back(completed.device_publications.front());
        for (const char *device : {"CUDA:0", "ROCm:0", "overlay"})
            recordMoEControllerCompletedMovement(completed.device_publications.front(), completed.edges, device);
    }
    const auto all = PerfStatsCollector::snapshot();
    ASSERT_EQ(all.size(), 36u);
    for (const char *device : {"CUDA:0", "ROCm:0", "overlay"})
    {
        std::vector<PerfStatRecord> rows;
        std::copy_if(all.begin(), all.end(), std::back_inserter(rows),
            [&](const auto &record) { return record.device == device; });
        ASSERT_EQ(rows.size(), 12u);
        EXPECT_TRUE(test::validateControllerMovementPerfStats(rows, ledger, 1).empty());
        auto mutated = ledger;
        // Preserve endpoints, record count, byte totals and extrema while changing
        // one interior exact transaction identity beyond double's integer range.
        ++mutated.device_publications[201].transaction;
        for (auto &edge : mutated.edges)
            if (edge.transaction == first + 201) ++edge.transaction;
        EXPECT_FALSE(test::validateControllerMovementPerfStats(rows, mutated, 1).empty());
        EXPECT_FALSE(test::validateControllerMovementPerfStats(rows, ledger, 0).empty());
        auto missing = rows;
        missing.erase(missing.begin() + 4);
        EXPECT_FALSE(test::validateControllerMovementPerfStats(missing, ledger, 1).empty());
        rows.push_back(rows.back());
        EXPECT_FALSE(test::validateControllerMovementPerfStats(rows, ledger, 1).empty());
    }
}

TEST_F(PerfStatsCardinality, ControllerMovementKeepsPolicyAndPhysicalCycleCountsDistinct)
{
    const auto merged = test::controllerMovementFixture(8, 9, 3);
    const auto &publication = merged.device_publications.front();
    ASSERT_EQ(publication.controller->accepted_cycles, 2u);
    ASSERT_EQ(publication.controller->physical_cycles, 1u);
    ASSERT_EQ(merged.economy.front().cycle_count, 1u);
    recordMoEControllerCompletedMovement(publication, merged.edges, "overlay");
    const auto records = PerfStatsCollector::snapshot();
    EXPECT_TRUE(test::validateControllerMovementPerfStats(records, merged, 2).empty());
    EXPECT_FALSE(test::validateControllerMovementPerfStats(records, merged, 1).empty());
}

TEST_F(PerfStatsCardinality, ControllerMovementRejectsInvalidWaveBeforeCounterMutation)
{
    const auto good = test::controllerMovementFixture(8, 9, 2);
    recordMoEControllerCompletedMovement(good.device_publications.front(), good.edges, "overlay");
    const auto before = PerfStatsCollector::jsonString();
    for (int defect = 0; defect < 9; ++defect)
    {
        auto bad = good;
        auto &publication = bad.device_publications.front();
        auto &receipt = *publication.controller;
        switch (defect)
        {
        case 0: ++receipt.participant_flow_violations; break;
        case 1: ++receipt.economy.projected_net_benefit_ns; break;
        case 2: ++receipt.edges_checked; break;
        case 3: --receipt.cross_backend_moves; break;
        case 4: bad.edges.front().direction = MoEOptimizationMovementDirection::SamePriority; break;
        case 5: bad.edges.back().destination_participant = 17; break;
        case 6: bad.edges.back().source_world_rank_known = false; break;
        case 7: bad.edges.back().cycle_size = 4; break;
        case 8: publication.controller.reset(); break;
        }
        EXPECT_THROW(recordMoEControllerCompletedMovement(publication, bad.edges, "overlay"), std::invalid_argument);
        EXPECT_EQ(PerfStatsCollector::jsonString(), before);
    }
}

TEST_F(PerfStatsCardinality, NativeMovementWavesRetainConstantRowsAndExactTransportIdentity)
{
    constexpr std::uint64_t waves = 512, first = std::uint64_t{1} << 54;
    std::array<MoEOptimizationMovementEdge, 2> edges;
    for (std::size_t i = 0; i < edges.size(); ++i)
        edges[i] = {.authority = MoEOptimizationAuthority::Device,
            .transaction = first, .candidate_epoch = first, .layer = 1, .expert = static_cast<int>(i),
            .cycle_index = 0, .cycle_size = 2, .direction = MoEOptimizationMovementDirection::SamePriority,
            .axis = MoEOptimizationMovementAxis::ParticipantPlacement,
            .source_participant = static_cast<int>(i), .destination_participant = static_cast<int>(1 - i),
            .source_priority = -7, .destination_priority = -7,
            .source_device = DeviceId::rocm(static_cast<int>(i)),
            .destination_device = DeviceId::rocm(static_cast<int>(1 - i)),
            .source_world_rank = 3, .destination_world_rank = 3,
            .source_world_rank_known = true, .destination_world_rank_known = true,
            .estimated_weight_bytes = 8192, .activation_count = 17};
    std::uint64_t bytes = 0;
    for (std::uint64_t wave = 0; wave < waves; ++wave)
    {
        const MoEOptimizationDeviceMovementPublication publication{first + wave, first + wave, 2, 6000 + wave};
        bytes += publication.physical_payload_bytes;
        for (auto &edge : edges)
            edge.transaction = edge.candidate_epoch = publication.transaction;
        for (const char *device : {"CUDA:0", "ROCm:0", "ROCm:1"})
            recordNativeMoECompletedMovement(publication, edges, device, 0, 2);
    }
    const auto records = PerfStatsCollector::snapshot({"moe_overlay_controller"});
    ASSERT_EQ(records.size(), 12u);
    for (const auto &record : records)
    {
        EXPECT_EQ(record.tags.size(), 5u);
        EXPECT_EQ(record.sequence_minimum_words.front(), first);
        EXPECT_EQ(record.sequence_maximum_words.front(), first + waves - 1);
        const bool identity = record.name == "dynamic_migration_edge_identities";
        EXPECT_EQ(record.count, identity ? 2 * waves : waves);
        EXPECT_EQ(record.sequence_word_count, identity ? 23 * 2 * waves : 4 * waves);
        if (record.name == "dynamic_physical_bytes") EXPECT_EQ(record.value, bytes);
        else if (record.name == "dynamic_movement_transactions") EXPECT_EQ(record.value, waves);
        else EXPECT_EQ(record.value, 2 * waves);
        if (!identity)
            for (const auto &other : records)
                if (other.device == record.device && other.name != "dynamic_migration_edge_identities")
                {
                    EXPECT_EQ(other.sequence_digest_lo, record.sequence_digest_lo);
                    EXPECT_EQ(other.sequence_digest_hi, record.sequence_digest_hi);
                }
    }
}

TEST_F(PerfStatsCardinality, NativeMovementRejectsMismatchedEvidenceBeforeAnyPublication)
{
    const MoEOptimizationDeviceMovementPublication publication{9, 10, 2, 6000};
    MoEOptimizationMovementEdge edge{.authority = MoEOptimizationAuthority::Device,
        .transaction = 9, .candidate_epoch = 10, .layer = 0, .expert = 3,
        .cycle_index = 0, .cycle_size = 2, .direction = MoEOptimizationMovementDirection::SamePriority,
        .axis = MoEOptimizationMovementAxis::ParticipantPlacement,
        .source_participant = 0, .destination_participant = 1,
        .source_device = DeviceId::cuda(0), .destination_device = DeviceId::cuda(1),
        .estimated_weight_bytes = 8192};
    for (int mutation = 0; mutation < 8; ++mutation)
    {
        auto changed = publication;
        std::array edges{edge, edge};
        edges[1].expert = 4;
        switch (mutation)
        {
        case 0: changed.physical_payload_bytes = 0; break;
        case 1: changed.command_count = 1; break;
        case 2: ++edges[1].transaction; break;
        case 3: edges[1].source_device = DeviceId::invalid(); break;
        case 4: edges[1].destination_device = DeviceId::rocm(1); break;
        case 5: edges[1].estimated_weight_bytes = 0; break;
        case 6: edges[1].blocking_inference = true; break;
        case 7: edges[1].destination_priority = 1; break;
        }
        EXPECT_THROW(recordNativeMoECompletedMovement(changed, edges, "CUDA:0", 0, 2), std::invalid_argument) << mutation;
        EXPECT_TRUE(PerfStatsCollector::snapshot().empty());
    }
}

TEST_F(PerfStatsCardinality, NativeMovementPipelineStagesOnOneDeviceKeepSeparateBoundedWitnesses)
{
    constexpr std::uint64_t waves = 512;
    for (std::uint64_t wave = 1; wave <= waves; ++wave)
        for (int first_layer : {0, 8})
        {
            std::array<MoEOptimizationMovementEdge, 2> edges;
            for (int i = 0; i < 2; ++i)
                edges[i] = {.authority = MoEOptimizationAuthority::Device,
                    .transaction = wave, .candidate_epoch = wave, .layer = first_layer + 1, .expert = i,
                    .cycle_index = 0, .cycle_size = 2, .direction = MoEOptimizationMovementDirection::SamePriority,
                    .axis = MoEOptimizationMovementAxis::ParticipantPlacement,
                    .source_participant = i, .destination_participant = 1 - i,
                    .source_device = DeviceId::cuda(i), .destination_device = DeviceId::cuda(1 - i),
                    .estimated_weight_bytes = 8192};
            const MoEOptimizationDeviceMovementPublication publication{wave, wave, 2, static_cast<std::uint64_t>(6000 + first_layer)};
            recordNativeMoECompletedMovement(publication, edges, "CUDA:0", first_layer, 8);
        }
    const auto records = PerfStatsCollector::snapshot({"moe_overlay_controller"});
    ASSERT_EQ(records.size(), 8);
    for (const auto &record : records)
    {
        const int first_layer = std::stoi(record.tags.at("first_model_layer"));
        EXPECT_TRUE(first_layer == 0 || first_layer == 8);
        EXPECT_EQ(record.tags.at("layer_count"), "8");
        if (record.name == "dynamic_physical_bytes")
            EXPECT_EQ(record.value, waves * (6000 + first_layer));
        else if (record.name == "dynamic_movement_transactions")
            EXPECT_EQ(record.value, waves);
        else
            EXPECT_EQ(record.value, 2 * waves);
    }
}

TEST_F(PerfStatsCardinality, PhysicalResidencyLifetimesRetainBoundedRowsAndExactWork)
{
    constexpr std::uint64_t waves = 512;
    constexpr std::uint64_t first = std::uint64_t{1} << 54;
    std::uint64_t staged = 0, retired = 0, recycled = 0;
    for (std::uint64_t wave = 0; wave < waves; ++wave)
    {
        staged += 2 + wave % 7;
        retired += wave % 7; // A rank may have no local source in a global wave.
        recycled += 1 + wave % 4;
        for (const char *device : {"CPU:0", "CUDA:0", "ROCm:0"})
        {
            recordMoEPhysicalWave(MoEPhysicalWaveEvent::DestinationsStaged,
                first + 2 * wave, first + wave, first + wave + 1, 2 + wave % 7, device);
            recordMoEPhysicalWave(MoEPhysicalWaveEvent::EpochPublished,
                first + 2 * wave, first + wave, first + wave + 1, 0, device);
            recordMoEPhysicalWave(MoEPhysicalWaveEvent::SourcesRetired,
                first + 2 * wave, first + wave, first + wave + 1, wave % 7, device);
            recordMoEPhysicalWave(MoEPhysicalWaveEvent::WaveAborted,
                first + 2 * wave + 1, first + wave + 1, first + wave + 2, 0, device);
            recordMoEBootstrapSlotsRecycled(first + wave, 1 + wave % 4, device);
            recordMoEReusableContextSeal(first + wave + 1, 32 + wave % 16, wave % 8, device);
        }
    }
    const auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 18u) << "Lifetime identities must never allocate a statistics key";
    std::map<std::string, PerfStatRecord> first_device;
    for (const auto &row : rows)
    {
        EXPECT_TRUE(row.tags.empty());
        EXPECT_EQ(row.count, waves);
        if (row.name == "device_physical_destinations_staged") EXPECT_EQ(row.value, staged);
        else if (row.name == "device_physical_sources_retired") EXPECT_EQ(row.value, retired);
        else if (row.name == "bootstrap_live_slots_recycled") EXPECT_EQ(row.value, recycled);
        else EXPECT_EQ(row.value, waves);
        const auto width = row.name == "bootstrap_live_slots_recycled" ? 2u :
            row.name == "reusable_context_physical_seals" ? 3u : 4u;
        EXPECT_EQ(row.sequence_word_count, waves * width);
        ASSERT_EQ(row.sequence_minimum_words.size(), width);
        ASSERT_EQ(row.sequence_maximum_words.size(), width);
        const auto [reference, inserted] = first_device.emplace(row.name, row);
        if (!inserted)
        {
            EXPECT_EQ(row.sequence_digest_lo, reference->second.sequence_digest_lo);
            EXPECT_EQ(row.sequence_digest_hi, reference->second.sequence_digest_hi);
            EXPECT_EQ(row.sequence_minimum_words, reference->second.sequence_minimum_words);
            EXPECT_EQ(row.sequence_maximum_words, reference->second.sequence_maximum_words);
        }
        if (row.name == "device_physical_sources_retired")
        {
            EXPECT_EQ(row.sequence_minimum_words, (std::vector<std::uint64_t>{first, first, first + 1, 0}));
            EXPECT_EQ(row.sequence_maximum_words,
                (std::vector<std::uint64_t>{first + 2 * (waves - 1), first + waves - 1, first + waves, 6}));
        }
    }
    EXPECT_LT(PerfStatsCollector::jsonString().size(), 25000u);
}

TEST_F(PerfStatsCardinality, PhysicalResidencyResetAndRejectedEventCannotPublishOldWork)
{
    recordMoEPhysicalWave(MoEPhysicalWaveEvent::SourcesRetired, 1, 1, 2, 4, "ROCm:1");
    recordMoEBootstrapSlotsRecycled(1, 2, "ROCm:1");
    recordMoEReusableContextSeal(2, 8, 0, "ROCm:1");
    PerfStatsCollector::reset();
    EXPECT_THROW(recordMoEPhysicalWave(static_cast<MoEPhysicalWaveEvent>(99), 2, 2, 3, 4, "ROCm:1"),
                 std::invalid_argument);
    EXPECT_TRUE(PerfStatsCollector::snapshot().empty());
    constexpr std::uint64_t first = (std::uint64_t{1} << 54) + 17;
    recordMoEPhysicalWave(MoEPhysicalWaveEvent::SourcesRetired, first, first + 1, first + 2, 0, "ROCm:1");
    const auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().value, 0);
    EXPECT_EQ(rows.front().count, 1u);
    EXPECT_EQ(rows.front().sequence_minimum_words, (std::vector<std::uint64_t>{first, first + 1, first + 2, 0}));
    EXPECT_EQ(rows.front().sequence_minimum_words, rows.front().sequence_maximum_words);
}

TEST_F(PerfStatsCardinality, ZeroMovementDecisionsRetainBoundedRowsAndExactIdentity)
{
    constexpr std::uint64_t decisions = 2048;
    constexpr std::uint64_t first = (std::uint64_t{1} << 54) + 1;
    for (std::uint64_t decision = 0; decision < decisions; ++decision)
    {
        MoEOverlayDeviceControllerCommandHeader command;
        command.kind = static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::DynamicPlacement);
        command.transaction_id = first + decision;
        command.base_epoch = command.candidate_epoch = first;
        command.snapshot_observations = 19 * decision;
        command.priority_cost_before = command.priority_cost_after = first + decision;
        command.same_priority_makespan_before = command.same_priority_makespan_after = decision;
        command.rejected_cycles = decision % 13;
        command.phase_tradeoff_candidates = decision % 17;
        command.improvement_floor_rejected_cycles = decision % 19;
        command.payoff_rejected_cycles = decision % 23;
        command.residency_rejected_cycles = decision % 29;
        command.layer_scan_start = decision % 48;
        command.layer_scan_next = (decision + 1) % 48;
        for (const char *device : {"CPU:0", "CUDA:0", "ROCm:0"})
            recordMoEZeroMovementCompletion(command, device);
        command.kind = static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore);
        for (const char *device : {"CPU:0", "CUDA:0", "ROCm:0"})
            recordMoEZeroMovementCompletion(command, device);
    }
    const auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 6u);
    for (const auto &row : rows)
    {
        EXPECT_EQ(row.value, decisions);
        EXPECT_EQ(row.count, decisions);
        EXPECT_FALSE(row.tags.contains("transaction"));
        EXPECT_FALSE(row.tags.contains("snapshot_observations"));
        const auto width = row.name == "dynamic_zero_movement_transactions" ? 20u : 2u;
        EXPECT_EQ(row.sequence_word_count, decisions * width);
        ASSERT_EQ(row.sequence_minimum_words.size(), width);
        ASSERT_EQ(row.sequence_maximum_words.size(), width);
        EXPECT_EQ(row.sequence_minimum_words[0], first);
        EXPECT_EQ(row.sequence_maximum_words[0], first + decisions - 1);
        EXPECT_EQ(row.sequence_minimum_words[1], first);
        EXPECT_EQ(row.sequence_maximum_words[1], first);
        if (width == 20u)
        {
            EXPECT_EQ(row.sequence_minimum_words[3], 0u);
            EXPECT_EQ(row.sequence_maximum_words[3], 19u * (decisions - 1));
            EXPECT_EQ(row.sequence_minimum_words[4], first);
            EXPECT_EQ(row.sequence_maximum_words[4], first + decisions - 1);
        }
    }
}

TEST_F(PerfStatsCardinality, ZeroMovementPublisherRejectsPhysicalWorkAndInvalidKinds)
{
    MoEOverlayDeviceControllerCommandHeader good;
    good.kind = static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::DynamicPlacement);
    good.transaction_id = 91;
    good.base_epoch = good.candidate_epoch = 6;
    for (unsigned mutation = 0; mutation < 6; ++mutation)
    {
        auto bad = good;
        switch (mutation)
        {
        case 0: bad.transaction_id = 0; break;
        case 1: bad.base_epoch = bad.candidate_epoch = 0; break;
        case 2: ++bad.candidate_epoch; break;
        case 3: bad.command_count = 1; break;
        case 4: bad.packed_weight_bytes = 64; break;
        case 5: bad.kind = static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::StaticCheck); break;
        }
        EXPECT_THROW(recordMoEZeroMovementCompletion(bad, "CUDA:0"), std::invalid_argument);
    }
    EXPECT_TRUE(PerfStatsCollector::snapshot().empty());
    recordMoEZeroMovementCompletion(good, "CUDA:0");
    EXPECT_EQ(PerfStatsCollector::snapshot().size(), 1u);
    PerfStatsCollector::reset();
    ++good.transaction_id;
    recordMoEZeroMovementCompletion(good, "CUDA:0");
    const auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().count, 1u);
    ASSERT_EQ(rows.front().sequence_minimum_words.size(), 20u);
    EXPECT_EQ(rows.front().sequence_minimum_words[0], good.transaction_id);
}

TEST_F(PerfStatsCardinality, MaintenanceDiagnosticsStayBoundedAcrossChangingWindowsAndSnapshots)
{
    constexpr std::uint64_t waves = 512;
    constexpr std::uint64_t first = std::uint64_t{1} << 54;
    std::uint64_t total_operations = 0, total_polls = 0, total_rows = 0;
    for (std::uint64_t wave = 0; wave < waves; ++wave)
    {
        const char *phase = wave % 2 == 0 ? "prefill" : "decode";
        const auto migrations = 1 + wave % 5;
        const auto operations = migrations * 3;
        const auto polls = operations + wave % 13;
        const auto rows = wave % 11;
        total_operations += operations;
        total_polls += polls;
        total_rows += rows;
        MoEOverlayDeviceControllerCommandHeader command;
        command.kind = static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore);
        command.transaction_id = first + wave;
        command.base_epoch = wave + 1;
        command.candidate_epoch = wave + 2;
        command.command_count = migrations;
        command.packed_weight_bytes = migrations * 4096;
        for (const char *device : {"CPU:0", "CUDA:0", "ROCm:0"})
        {
            recordMoEMaintenanceNotification(first + wave, 1 + wave, phase, device);
            recordMoEMaintenanceWindowGrowth(1 + wave, 2 + wave, 65536, 1.25, device);
            recordMoEServicePhaseObservation(wave % 3, phase, wave, 3 * wave, device);
            recordMoEServiceSnapshotPublication(wave % 3, first + wave, device);
            recordMoEServiceSnapshotRows(rows, 17 * wave, device);
            recordMoEPhysicalWaveProgress({first + wave, migrations, operations, polls,
                1 + polls / 4, 4 + wave % 4}, device);
            recordMoEPreparedContextMovement(command, device);
        }
    }
    const auto records = PerfStatsCollector::snapshot();
    ASSERT_EQ(records.size(), 48u);
    std::map<std::string, PerfStatRecord> progress_by_device;
    for (const auto &record : records)
    {
        for (const char *tag : {"transaction", "base_epoch", "candidate_epoch", "generation",
             "coalesced_tokens", "previous_tokens", "next_tokens", "samples", "activations",
             "operations", "migrations", "poll_quanta", "physical_bytes"})
            EXPECT_FALSE(record.tags.contains(tag)) << record.name << '/' << tag;
        if (record.name == "physical_wave_parallel_operations_started"
            || record.name == "physical_wave_bounded_progress_polls")
        {
            EXPECT_EQ(record.count, waves);
            EXPECT_EQ(record.sequence_word_count, waves * 6);
            EXPECT_EQ(record.value, record.name == "physical_wave_bounded_progress_polls"
                ? total_polls : total_operations);
            const auto [other, inserted] = progress_by_device.emplace(record.device, record);
            if (!inserted)
            {
                EXPECT_EQ(record.sequence_digest_lo, other->second.sequence_digest_lo);
                EXPECT_EQ(record.sequence_digest_hi, other->second.sequence_digest_hi);
                EXPECT_EQ(record.sequence_minimum_words, other->second.sequence_minimum_words);
                EXPECT_EQ(record.sequence_maximum_words, other->second.sequence_maximum_words);
            }
        }
        else if (record.name == "device_service_snapshot_rows")
        {
            EXPECT_EQ(record.count, waves);
            EXPECT_EQ(record.value, total_rows);
            EXPECT_EQ(record.sequence_word_count, waves * 2);
        }
        else if (record.name == "prepared_context_restore_movement_waves")
        {
            EXPECT_EQ(record.count, waves);
            EXPECT_EQ(record.value, waves);
            EXPECT_EQ(record.phase, "model_teardown");
            EXPECT_EQ(record.sequence_word_count, waves * 5);
            ASSERT_EQ(record.sequence_minimum_words.size(), 5u);
            EXPECT_EQ(record.sequence_minimum_words[0], first);
            EXPECT_EQ(record.sequence_maximum_words[0], first + waves - 1);
            EXPECT_EQ(record.tags.at("excluded_from_optimization_ledger"), "true");
        }
    }
    EXPECT_LT(PerfStatsCollector::jsonString().size(), 50000u);
}

TEST_F(PerfStatsCardinality, MaintenanceRestorationCannotCertifyAnOptimizationOrWrappedEpoch)
{
    MoEOverlayDeviceControllerCommandHeader command;
    command.kind = static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore);
    command.transaction_id = 19;
    command.base_epoch = 3;
    command.candidate_epoch = 4;
    command.command_count = 2;
    command.packed_weight_bytes = 8192;
    for (unsigned mutation = 0; mutation < 7; ++mutation)
    {
        auto bad = command;
        switch (mutation)
        {
        case 0: bad.kind = static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::DynamicPlacement); break;
        case 1: bad.transaction_id = 0; break;
        case 2: bad.base_epoch = 0; break;
        case 3: bad.candidate_epoch = bad.base_epoch; break;
        case 4: bad.command_count = 0; break;
        case 5: bad.packed_weight_bytes = 0; break;
        case 6: bad.base_epoch = UINT64_MAX; bad.candidate_epoch = 0; break;
        }
        EXPECT_THROW(recordMoEPreparedContextMovement(bad, "ROCm:0"), std::invalid_argument);
    }
    EXPECT_TRUE(PerfStatsCollector::snapshot().empty());
}

TEST_F(PerfStatsCardinality, ReorderedOrChangedLargeGenerationsProduceDifferentWitnesses)
{
    const auto publish = [](const char *role, uint64_t first, uint64_t second)
    {
        PerfStatsCollector::addCounterWithSequence("mtp", "publication", 1.0, {first}, "decode", "CPU:0", {{"role", role}});
        PerfStatsCollector::addCounterWithSequence("mtp", "publication", 1.0, {second}, "decode", "CPU:0", {{"role", role}});
    };
    constexpr uint64_t large = uint64_t{1} << 54;
    publish("ordered", large, large + 1);
    publish("reordered", large + 1, large);
    publish("changed", large, large + 2);
    const auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 3u);
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = i + 1; j < rows.size(); ++j)
        {
            EXPECT_EQ(rows[i].value, rows[j].value);
            EXPECT_NE(rows[i].sequence_digest_lo, rows[j].sequence_digest_lo);
            EXPECT_NE(rows[i].sequence_digest_hi, rows[j].sequence_digest_hi);
        }
}

TEST_F(PerfStatsCardinality, MissingOrChangedEvidenceSchemaFailsWithoutChangingCounters)
{
    EXPECT_THROW(PerfStatsCollector::addCounterWithSequence("mtp", "empty", 1, {}), std::invalid_argument);
    EXPECT_TRUE(PerfStatsCollector::snapshot().empty());
    PerfStatsCollector::addCounterWithSequence("mtp", "evidenced", 4, {1, 2});
    EXPECT_THROW(PerfStatsCollector::addCounter("mtp", "evidenced"), std::logic_error);
    EXPECT_THROW(PerfStatsCollector::addCounterWithSequence("mtp", "evidenced", 1, {3}), std::logic_error);
    auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].count, 1u);
    EXPECT_EQ(rows[0].value, 4);
    EXPECT_EQ(rows[0].sequence_word_count, 2u);
    PerfStatsCollector::addCounter("mtp", "ordinary", 7);
    EXPECT_THROW(PerfStatsCollector::addCounterWithSequence("mtp", "ordinary", 1, {1}), std::logic_error);
    rows = PerfStatsCollector::snapshot();
    EXPECT_EQ(rows[1].value, 7);
}

TEST_F(PerfStatsCardinality, ConcurrentOwnersStayIndependentAndResetRetainsOnlyRequestedDomains)
{
    std::vector<std::jthread> workers;
    for (unsigned owner = 0; owner < 4; ++owner)
        workers.emplace_back([owner]
        {
            for (uint64_t step = 0; step < 2000; ++step)
                PerfStatsCollector::addCounterWithSequence("mtp", "events", 2, {step}, "decode", std::to_string(owner));
        });
    workers.clear();
    const auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 4u);
    for (const auto &row : rows)
    {
        EXPECT_EQ(row.count, 2000u);
        EXPECT_EQ(row.value, 4000);
        EXPECT_EQ(row.sequence_digest_lo, rows.front().sequence_digest_lo);
    }
    PerfStatsCollector::addCounterWithSequence("physical_memory", "proof", 1, {42});
    PerfStatsCollector::resetPreservingDomains({"physical_memory"});
    const auto retained = PerfStatsCollector::snapshot();
    ASSERT_EQ(retained.size(), 1u);
    EXPECT_EQ(retained[0].domain, "physical_memory");
    EXPECT_EQ(retained[0].sequence_word_count, 1u);
}
TEST_F(PerfStatsCardinality, TimingsRetainExtremaAndExactTotalsAcrossChangingPositions)
{
    uint64_t expected_ns = 0;
    for (uint64_t step = 0; step < 8192; ++step)
    {
        const uint64_t duration = 100 + step % 7;
        expected_ns += duration;
        PerfStatsCollector::recordTimingNsWithSequence("forward_graph", "prefill_graph_launch",
            duration, {step, step * 64, step * 64 + 64}, "prefill", "ROCm:0",
            {{"bucket_seq_len", "64"}, {"launch_kind", "replay"}});
    }
    const auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].kind, PerfStatRecord::Kind::Timer);
    EXPECT_EQ(rows[0].count, 8192u);
    EXPECT_EQ(rows[0].sequence_word_count, 8192u * 3);
    EXPECT_EQ(rows[0].total_ns, expected_ns);
    EXPECT_EQ(rows[0].min_ns, 100u);
    EXPECT_EQ(rows[0].max_ns, 106u);
    EXPECT_EQ(rows[0].sequence_minimum_words, (std::vector<uint64_t>{0, 0, 64}));
    EXPECT_EQ(rows[0].sequence_maximum_words, (std::vector<uint64_t>{8191, 8191 * 64, 8192 * 64}));
    EXPECT_THROW(PerfStatsCollector::recordTimingNsWithSequence("t", "empty", 1, {}), std::invalid_argument);
    PerfStatsCollector::recordTimingNsWithSequence("t", "sampled", 1, {1});
    EXPECT_THROW(PerfStatsCollector::recordTimingNs("t", "sampled", 1), std::logic_error);
    EXPECT_THROW(PerfStatsCollector::recordTimingNsWithSequence("t", "sampled", 1, {1, 2}), std::logic_error);
    PerfStatsCollector::recordTimingNs("t", "ordinary", 1);
    EXPECT_THROW(PerfStatsCollector::recordTimingNsWithSequence("t", "ordinary", 1, {1}), std::logic_error);
}

TEST_F(PerfStatsCardinality, IncompleteMovementObservationsCannotCombineIntoCompletion)
{
    for (unsigned request = 0; request < 10000; ++request)
    {
        recordCompletedDeviceMoEMovement({.copied_payload_lower_bound = 2},
            "decode", "ROCm:0", "maintenance_status");
        recordCompletedDeviceMoEMovement({.applied_payload_lower_bound = 2},
            "decode", "ROCm:0", "maintenance_status");
    }
    EXPECT_TRUE(PerfStatsCollector::snapshot().empty());
    for (unsigned request = 0; request < 10000; ++request)
        recordCompletedDeviceMoEMovement({.copied_payload_lower_bound = 3,
            .applied_payload_lower_bound = 2, .completed_payload_lower_bound = 2,
            .useful_payload_bytes_lower_bound = 4096}, "decode", "ROCm:0", "maintenance_status");
    const auto rows = PerfStatsCollector::snapshot();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].count, 10000u);
    EXPECT_EQ(rows[0].sequence_word_count, 40000u);
    EXPECT_EQ(rows[0].sequence_minimum_words, (std::vector<uint64_t>{3, 2, 2, 4096}));
    EXPECT_EQ(rows[0].sequence_minimum_words, rows[0].sequence_maximum_words);
    EXPECT_THROW(recordCompletedDeviceMoEMovement({.copied_payload_lower_bound = 1,
        .applied_payload_lower_bound = 2, .completed_payload_lower_bound = 2,
        .useful_payload_bytes_lower_bound = 4096}, "decode", "ROCm:0", "maintenance_status"), std::logic_error);
    EXPECT_EQ(PerfStatsCollector::snapshot()[0].count, 10000u);
}

}
