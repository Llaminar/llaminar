/**
 * @file Test__SnapshotRowPartition.cpp
 * @brief Device-free row ownership, bit-preserving assembly and chunk lifecycle proofs.
 *
 * Empty owners, signed zeros, NaN payloads, missing rows and overlaps are
 * deliberate adversaries. Diagnostics must not silently reduce partial rows
 * or reconstruct ownership from their values.
 */
#include "snapshots/SnapshotCapture.h"
#include "execution/debug/TPSnapshot.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstring>
#include <numeric>
using namespace llaminar2;

/** @test Every count through a partial bucket assembles unchanged bits in arbitrary producer order. */
TEST(SnapshotRowPartition, AllCountsAndDegreesPreserveBits)
{
    constexpr int capacity = 65, cols = 7;
    std::vector<float> oracle(capacity * cols);
    for (std::size_t i = 0; i < oracle.size(); ++i)
        oracle[i] = std::bit_cast<float>(std::uint32_t(0x80000000u + i));
    oracle[3] = std::bit_cast<float>(0x7fc12345u);
    for (int degree : {2, 3, 4, 8})
        for (int live = 1; live <= capacity; ++live)
        {
            std::vector<std::vector<float>> projected(degree);
            std::vector<SnapshotOwnedRows> ownership(degree);
            std::vector<SnapshotRowSource> sources;
            for (int p = 0; p < degree; ++p)
            {
                const auto partition = DeviceRowPartition::balanced(p, degree);
                const auto range = partition.resolveFor(capacity, live);
                std::vector<float> compact(partition.capacityFor(capacity) * cols, 123.0f);
                std::memcpy(compact.data(), oracle.data() + range.first * cols, range.count * cols * sizeof(float));
                ownership[p] = projectSnapshotOwnedRows(compact, {partition, capacity}, live, cols, projected[p]);
            }
            for (int p = degree - 1; p >= 0; --p)
                sources.push_back({projected[p], std::size_t(live), cols, &ownership[p]});
            const auto result = assembleSnapshotOwnedRows(sources);
            ASSERT_EQ(result.size(), std::size_t(live * cols));
            EXPECT_EQ(std::memcmp(result.data(), oracle.data(), result.size() * sizeof(float)), 0);
        }
}

/** @test Malformed or incomplete metadata fails before pretending to certify an output. */
TEST(SnapshotRowPartition, RejectsMissingOverlappingOrUnobservedRows)
{
    std::vector<float> data(4, 1.f);
    SnapshotOwnedRows first{{{0, 2}}}, rest{{{2, 2}}}, overlap{{{1, 3}}};
    std::array<SnapshotRowSource, 2> sources{{{data, 4, 1, &first}, {data, 4, 1, &rest}}};
    EXPECT_NO_THROW((void)assembleSnapshotOwnedRows(sources));
    sources[1].owned = &overlap;
    EXPECT_THROW((void)assembleSnapshotOwnedRows(sources), std::invalid_argument);
    sources[1].owned = nullptr;
    EXPECT_THROW((void)assembleSnapshotOwnedRows(sources), std::invalid_argument);
    EXPECT_THROW((void)assembleSnapshotOwnedRows(std::span(sources).first(1)), std::invalid_argument);
    StageDumpInfo dump;
    dump.addOutput("router_logits", data.data(), 2, 2);
    dump.outputs.back().row_layout = SnapshotCompactRows{DeviceRowPartition::balanced(0, 2), 4};
    SnapshotCapture capture;
    EXPECT_THROW(capture.captureStage("layer0_moe_routing_owned", dump), std::logic_error);
}

/** @test Per-device chunk concatenation retains discontiguous global row ownership. */
TEST(SnapshotRowPartition, ChunkAssemblyKeepsProducerCoordinates)
{
    std::array<SnapshotCapture, 2> captures;
    const std::vector<SnapshotChunkSequencePart> chunks{{"PrefillChunk0", 3}, {"PrefillChunk1", 5}};
    for (int p = 0; p < 2; ++p)
    {
        for (int chunk = 0; chunk < 2; ++chunk)
        {
            const int live = chunk ? 5 : 3;
            const auto partition = DeviceRowPartition::balanced(p, 2);
            const auto range = partition.resolveFor(8, live);
            const auto storage = chunk ? SnapshotRowStorage::CompleteReplicated : SnapshotRowStorage::CompactOwned;
            std::vector<float> compact(chunk ? 8 : 4);
            for (int i = 0; i < (chunk ? live : range.count); ++i)
                compact[i] = (chunk ? 3 : range.first) + i + 1;
            std::vector<float> projected;
            auto owned = projectSnapshotOwnedRows(compact, {partition, 8, storage}, live, 1, projected);
            StageDumpInfo dump; dump.addOutput("router_logits", projected.data(), live, 1);
            dump.outputs.back().row_layout = std::move(owned);
            captures[p].captureStage(chunks[chunk].context + "::layer0_moe_routing_owned", dump);
        }
        const auto result = captures[p].aggregateSequentialChunkSnapshots(chunks);
        ASSERT_TRUE(result.ok) << result.error;
    }
    std::vector<SnapshotRowSource> sources;
    for (auto &capture : captures)
    {
        const auto snap = capture.getShared("layer0_MOE_ROUTER_OUTPUT");
        ASSERT_TRUE(snap);
        EXPECT_EQ(snap->publication, SnapshotPublication::RowPartition);
        EXPECT_EQ(snap->row_ownership.intervals.size(), 2u);
        sources.push_back({snap->data, snap->rows, snap->cols, &snap->row_ownership});
    }
    EXPECT_EQ(assembleSnapshotOwnedRows(sources), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}));
}

/** @test The public TP snapshot surface preserves participants and rejects duplicate/missing ownership. */
TEST(SnapshotRowPartition, TPSnapshotRetainsEveryProducer)
{
    TPSnapshot snapshot;
    snapshot.mode = SnapshotShardingMode::TOKEN_ROW_PARTITION;
    snapshot.tp_degree = 3;
    for (int p = 0; p < 3; ++p)
    {
        DeviceSnapshotData data;
        data.device_index = p; data.rows = 2; data.cols = 1; data.data = {-0.0f, 3.f};
        if (p < 2) data.row_ownership.intervals.push_back({std::size_t(p), 1});
        snapshot.device_data.push_back(std::move(data));
    }
    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.device_data.size(), 3u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(snapshot.combined_data[0]), 0x80000000u);
    EXPECT_EQ(snapshot.combined_data[1], 3.f);
    snapshot.device_data[2].device_index = 1;
    EXPECT_FALSE(snapshot.computeCombined());
    snapshot.device_data[2].device_index = 2;
    snapshot.device_data[2].row_ownership.intervals.push_back({0, 1});
    EXPECT_FALSE(snapshot.computeCombined());
    snapshot.device_data.pop_back();
    EXPECT_FALSE(snapshot.computeCombined());
}
