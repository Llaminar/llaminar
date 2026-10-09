/**
 * @file Test__MoEOverlayServiceTelemetryPublication.cpp
 * @brief CPU-only protocol tests for mapped GPU service snapshots.
 *
 * Adversarial publication headers prove that stage identity and bounded row
 * geometry are checked before any cumulative service data is imported.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEOverlayServiceTelemetryPublication.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr std::uint32_t kLayers = 2u;
        constexpr int kParticipant = 7;
        constexpr std::size_t kPublicationBytes =
            deviceMoEOverlayServiceTelemetryPublicationBytes(kLayers);

        /** @brief Build one complete even-generation publication in host RAM. */
        struct PublicationFixture
        {
            alignas(64) std::array<std::byte, kPublicationBytes> bytes{};

            PublicationFixture()
            {
                auto *const header = publication();
                *header = MoEOverlayDeviceServiceTelemetryPublicationHeader{
                    .participant_id = kParticipant,
                    .layer_count = kLayers,
                    .generation = 2u,
                };
                auto *const cells =
                    deviceMoEOverlayServiceTelemetryCells(header);
                for (std::size_t index = 0u;
                     index < deviceMoEOverlayServiceTelemetryCellCount(
                                 kLayers);
                     ++index)
                {
                    cells[index].total_nanoseconds = 100u + index;
                    cells[index].activation_count = 10u + index;
                    cells[index].sample_count = 1u + index;
                }
            }

            /** @return Properly aligned ABI header stored at byte zero. */
            MoEOverlayDeviceServiceTelemetryPublicationHeader *publication()
                noexcept
            {
                return reinterpret_cast<
                    MoEOverlayDeviceServiceTelemetryPublicationHeader *>(
                    bytes.data());
            }
        };
    } // namespace

    /** Prove exact layer/phase mapping from one coherent even generation. */
    TEST(MoEOverlayServiceTelemetryPublication, AcquiresCompleteEvenGeneration)
    {
        PublicationFixture fixture;
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
        std::uint64_t generation = 0u;

        ASSERT_TRUE(trySnapshotMoEOverlayServiceTelemetryPublication(
            fixture.publication(),
            kParticipant,
            0,
            kLayers,
            &rows,
            &generation));
        ASSERT_EQ(generation, 2u);
        ASSERT_EQ(rows.size(), kLayers);
        for (std::size_t layer = 0u; layer < rows.size(); ++layer)
        {
            EXPECT_EQ(rows[layer].participant_id, kParticipant);
            EXPECT_EQ(rows[layer].layer, static_cast<int>(layer));
            for (std::size_t phase = 0u;
                 phase < kDeviceMoEOverlayServicePhaseCount;
                 ++phase)
            {
                const std::size_t index =
                    layer * kDeviceMoEOverlayServicePhaseCount + phase;
                EXPECT_EQ(
                    rows[layer].total_nanoseconds[phase],
                    100u + index);
                EXPECT_EQ(
                    rows[layer].activation_count[phase],
                    10u + index);
                EXPECT_EQ(rows[layer].sample_count[phase], 1u + index);
            }
        }
    }

    /** An odd generation means the device is changing cells and must defer. */
    TEST(MoEOverlayServiceTelemetryPublication, RejectsWriterOwnedGeneration)
    {
        PublicationFixture fixture;
        fixture.publication()->generation = 3u;
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows(1u);
        std::uint64_t generation = 99u;

        EXPECT_FALSE(trySnapshotMoEOverlayServiceTelemetryPublication(
            fixture.publication(),
            kParticipant,
            0,
            kLayers,
            &rows,
            &generation));
        EXPECT_TRUE(rows.empty());
        EXPECT_EQ(generation, 0u);
    }

    /** A mapped page cannot impersonate another topology participant. */
    TEST(MoEOverlayServiceTelemetryPublication, RejectsIdentityMismatch)
    {
        PublicationFixture fixture;
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows;

        EXPECT_FALSE(trySnapshotMoEOverlayServiceTelemetryPublication(
            fixture.publication(),
            kParticipant + 1,
            0,
            kLayers,
            &rows));
        EXPECT_TRUE(rows.empty());
    }

    /** Compact mapped rows retain exact global IDs, including the namespace boundary. */
    TEST(MoEOverlayServiceTelemetryPublication, PipelineStageDecodesGlobalRows)
    {
        for (const int first : {20, 40, std::numeric_limits<int>::max() - 2})
        {
            PublicationFixture fixture;
            fixture.publication()->first_model_layer = first;
            std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
            std::uint64_t generation = 0;
            ASSERT_TRUE(trySnapshotMoEOverlayServiceTelemetryPublication(
                fixture.publication(), kParticipant, first, kLayers, &rows, &generation));
            ASSERT_EQ(rows.size(), kLayers);
            EXPECT_EQ(generation, 2u);
            for (std::size_t row = 0; row < rows.size(); ++row)
            {
                EXPECT_EQ(rows[row].layer, first + row);
                for (std::size_t phase = 0; phase < kDeviceMoEOverlayServicePhaseCount; ++phase)
                    EXPECT_EQ(rows[row].total_nanoseconds[phase],
                              100 + row * kDeviceMoEOverlayServicePhaseCount + phase);
            }
            EXPECT_EQ(fixture.bytes.size(), kPublicationBytes)
                << "Global origin must not introduce preceding padding rows";
        }
    }

    /** An equal-shaped foreign publication cannot poison another stage's service costs. */
    TEST(MoEOverlayServiceTelemetryPublication, PipelineStageRejectsForeignOriginAndOldABI)
    {
        PublicationFixture fixture;
        fixture.publication()->first_model_layer = 20;
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows(2);
        std::uint64_t generation = 100;
        EXPECT_FALSE(trySnapshotMoEOverlayServiceTelemetryPublication(
            fixture.publication(), kParticipant, 0, kLayers, &rows, &generation));
        EXPECT_TRUE(rows.empty());
        EXPECT_EQ(generation, 0u);
        fixture.publication()->version = kDeviceMoEOverlayServiceTelemetryVersion - 1;
        EXPECT_FALSE(trySnapshotMoEOverlayServiceTelemetryPublication(
            fixture.publication(), kParticipant, 20, kLayers, &rows, &generation));
        EXPECT_TRUE(rows.empty());
        EXPECT_EQ(generation, 0u);
    }

    /** Reject malformed intervals before resizing or interpreting mapped cells. */
    TEST(MoEOverlayServiceTelemetryPublication, PipelineStageRejectsInvalidIntervals)
    {
        for (const auto [first, count] : std::vector<std::pair<int, std::uint32_t>>{
                 {-1, 2}, {20, 0}, {std::numeric_limits<int>::max(), 2}, {20, UINT32_MAX}})
        {
            PublicationFixture fixture;
            fixture.publication()->first_model_layer = first;
            fixture.publication()->layer_count = count;
            std::vector<MoEOverlayParticipantLayerServiceTotals> rows(2);
            std::uint64_t generation = 100;
            EXPECT_FALSE(trySnapshotMoEOverlayServiceTelemetryPublication(
                fixture.publication(), kParticipant, first, count, &rows, &generation));
            EXPECT_TRUE(rows.empty());
            EXPECT_EQ(generation, 0u);
        }
    }
} // namespace llaminar2::test
