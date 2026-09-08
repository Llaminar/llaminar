/**
 * @file Test__MoEOverlayServiceTelemetryPublication.cpp
 * @brief CPU-only protocol tests for mapped GPU service snapshots.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEOverlayServiceTelemetryPublication.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
            kLayers,
            &rows));
        EXPECT_TRUE(rows.empty());
    }
} // namespace llaminar2::test
