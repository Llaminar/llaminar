/**
 * @file Test__MoECompletedMovementPublication.cpp
 * @brief Preserve actual per-wave transport receipts across request boundaries.
 *
 * Device-free tests use the production native journal publisher and archive.
 * Actual copied bytes deliberately disagree with padded expert descriptors,
 * and adjacent uint64 identities exceed the exact range of a double. Missing
 * evidence remains a fatal diagnostic condition after journal exhaustion.
 */
#include "execution/moe/NativeMoEMovementArchive.h"
#include "app/modes/MoEMovementTransportJson.h"
#include "utils/ControllerMovementFixture.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>

namespace llaminar2::test
{
    namespace
    {
        /** @return One two-device domain with room for exactly two complete waves. */
        NativeMoEMovementArchiveConfig publicationGeometry(bool rocm)
        {
            return {.workspace_generation = 7, .layers = 2, .experts = 8,
                .wave_capacity = 2, .edge_capacity = 4,
                .participants = {{rocm ? DeviceId::rocm(1) : DeviceId::cuda(1), 3, 17},
                                 {rocm ? DeviceId::rocm(0) : DeviceId::cuda(0), 3, 17}}};
        }

        /** Own the same bounded POD storage written by the captured finalizer. */
        struct PublicationSnapshot
        {
            DeviceMoERebalanceMovementJournalState state{};
            std::array<DeviceMoERebalanceMovementWave, 2> waves{};
            std::array<DeviceMoERebalanceMovementEdge, 4> edges{};

            /**
             * @brief Publish a reciprocal ownership cycle through the production protocol.
             * @param epoch Durable model epoch; command-local ordinals may reset.
             * @param actual_bytes Useful physical bytes reported by the completed copy.
             */
            void commit(std::uint64_t epoch, std::uint64_t actual_bytes)
            {
                const DeviceMoERebalanceLoadSpreadProof proof{
                    .accepted_spread_improvement = 40, .pre_wave_spread = 100, .post_wave_spread = 60,
                    .pre_wave_total = 200, .post_wave_total = 200,
                    .pre_participant_spread = 0, .post_participant_spread = 20,
                    .pre_participant_total = 200, .post_participant_total = 200,
                    .requested_payload_slots = 1, .minimum_improvement_per_slot = 40,
                    .maximum_post_spread_per_mille = 300, .ownership_swap_accepts = 1};
                DeviceMoERebalanceMovementJournalView view{&state, waves.data(), edges.data(), 2, 4};
                auto append = prepareDeviceMoEMovementJournalAppend(view, epoch, 1, 2, proof, actual_bytes);
                ASSERT_NE(append.disposition, DeviceMoEMovementJournalDisposition::Invalid);
                if (append.disposition == DeviceMoEMovementJournalDisposition::Record)
                {
                    edges[append.first_edge] = {0, 3, 0, 1, 90, 8192};
                    edges[append.first_edge + 1] = {0, 4, 1, 0, 10, 8192};
                }
                ASSERT_TRUE(commitDeviceMoEMovementJournalAppend(view, append));
            }

            /**
             * @brief Observe only the live journal extent with the admitted workspace identity.
             * @param archive Existing model-lifetime authority.
             * @param request Monotonic request identity, independent of model epoch.
             */
            void observe(NativeMoEMovementArchive &archive, std::uint64_t request) const
            {
                archive.observe(request, 7, state,
                    std::span(waves).first(state.committed_waves),
                    std::span(edges).first(state.committed_edges));
            }
        };

        /** @brief Own one private metadata-only export directory for an I/O regression. */
        struct PublicationDirectory
        {
            std::filesystem::path path;
            /** @brief Create an exclusive temporary directory without shared filenames. */
            PublicationDirectory()
            {
                auto pattern = (std::filesystem::temp_directory_path() / "llaminar-movement-XXXXXX").string();
                const auto created = ::mkdtemp(pattern.data());
                if (!created)
                    throw std::runtime_error("Cannot create private movement export test directory");
                path = created;
            }
            /** @brief Retire only this fixture's metadata files. */
            ~PublicationDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path, ignored);
            }
            PublicationDirectory(const PublicationDirectory &) = delete;
            PublicationDirectory &operator=(const PublicationDirectory &) = delete;
        };
    }

    TEST(MoECompletedMovementPublication, ControllerTransportRetainsPhysicalProofAndLeaderOwnership)
    {
        for (unsigned geometry = 0; geometry < 4; ++geometry)
        {
            auto ledger = controllerMovementFixture((std::uint64_t{1} << 54) + 1, 9, geometry);
            const auto leader = moeMovementTransportJson(ledger);
            ASSERT_TRUE(leader["device_publications"][0].contains("controller"));
            EXPECT_EQ(leader["device_publications"][0]["controller"]["edges_checked"], ledger.edges.size());
            auto follower = ledger;
            follower.economy.clear();
            const auto mirrored = moeMovementTransportJson(follower);
            EXPECT_EQ(mirrored["device_publications"], leader["device_publications"]);
            EXPECT_TRUE(mirrored["movement"]["economy"].empty());
            ++ledger.device_publications[0].controller->economy.projected_service_gain_ns;
            ++ledger.device_publications[0].controller->economy.projected_net_benefit_ns;
            EXPECT_THROW(moeMovementTransportJson(ledger), std::invalid_argument);
            follower.device_publications[0].controller->participant_coordinates_checked = 0;
            EXPECT_THROW(moeMovementTransportJson(follower), std::invalid_argument);
        }
    }

    TEST(MoECompletedMovementPublication, ActualBytesAndExactIdentitiesSurviveRequestReset)
    {
        constexpr auto first = std::uint64_t{1} << 54;
        for (const bool rocm : {false, true})
        {
            NativeMoEMovementArchive archive(publicationGeometry(rocm));
            PublicationSnapshot before;
            before.commit(first, first + 1);
            before.observe(archive, 1);
            const auto accepted = archive.ledger().device_publications;
            before.observe(archive, 1);
            EXPECT_EQ(archive.ledger().device_publications, accepted);

            PublicationSnapshot after;
            after.observe(archive, 2); // A restored request can retire no new wave.
            EXPECT_EQ(archive.ledger().device_publications, accepted);
            after.commit(first + 1, 6000);
            after.observe(archive, 2);

            const auto &records = archive.ledger().device_publications;
            ASSERT_EQ(records.size(), 2u);
            EXPECT_EQ(records[0], (MoEOptimizationDeviceMovementPublication{first, first, 2, first + 1}));
            EXPECT_EQ(records[1], (MoEOptimizationDeviceMovementPublication{first + 1, first + 1, 2, 6000}));
            EXPECT_EQ(records[0].physical_payload_bytes + records[1].physical_payload_bytes,
                archive.totals().physical_bytes);
            EXPECT_NE(records[1].physical_payload_bytes,
                archive.ledger().edges[2].estimated_weight_bytes + archive.ledger().edges[3].estimated_weight_bytes);
        }
    }

    TEST(MoECompletedMovementPublication, RejectedMutationCannotRewriteCompletedTransport)
    {
        NativeMoEMovementArchive archive(publicationGeometry(false));
        PublicationSnapshot before;
        before.commit(9, 6000);
        before.observe(archive, 1);
        const auto accepted = archive.ledger().device_publications;
        ++before.waves[0].physical_payload_bytes;
        EXPECT_THROW(before.observe(archive, 1), std::invalid_argument);
        EXPECT_EQ(archive.ledger().device_publications, accepted);
        EXPECT_EQ(archive.totals().physical_bytes, 6000u);
        EXPECT_TRUE(archive.ledger().complete());
    }

    TEST(MoECompletedMovementPublication, ExhaustionRetainsOldReceiptsAndReportsEveryLostWave)
    {
        NativeMoEMovementArchive archive(publicationGeometry(true));
        PublicationSnapshot first;
        first.commit(9, 6000);
        first.commit(10, 7000);
        first.observe(archive, 1);
        const auto accepted = archive.ledger().device_publications;
        first.commit(11, 8000); // Device storage is exhausted, so no receipt can be recovered.
        first.observe(archive, 1);
        EXPECT_EQ(archive.ledger().discarded_device_publications, 1u);
        first.observe(archive, 1);
        EXPECT_EQ(archive.ledger().discarded_device_publications, 1u);

        PublicationSnapshot next;
        next.commit(12, 9000); // A fresh request cannot overwrite the full model archive.
        next.observe(archive, 2);
        EXPECT_EQ(archive.ledger().device_publications, accepted);
        EXPECT_EQ(archive.ledger().discarded_device_publications, 2u);
        EXPECT_EQ(archive.ledger().discarded_economy_records, 2u);
        EXPECT_FALSE(archive.ledger().complete());
        EXPECT_EQ(archive.totals().physical_bytes, 22000u); // Only retained device receipts count.

        MoEOptimizationMovementLedger lost;
        lost.discarded_device_publications = 1;
        EXPECT_FALSE(lost.complete());
    }

    TEST(MoECompletedMovementPublication, TerminalExportPreservesActualBytesAndFollowerScope)
    {
        constexpr auto epoch = (std::uint64_t{1} << 54) + 1;
        NativeMoEMovementArchive archive(publicationGeometry(false));
        PublicationSnapshot snapshot;
        snapshot.commit(epoch, epoch + 2);
        snapshot.observe(archive, 1);
        const auto terminal = nlohmann::json::parse(moeMovementTransportJson(archive.ledger()).dump());
        ASSERT_EQ(terminal["device_publications"].size(), 1u);
        EXPECT_EQ(terminal["device_publications"][0]["transaction"].get<std::uint64_t>(), epoch);
        EXPECT_EQ(terminal["device_publications"][0]["physical_payload_bytes"].get<std::uint64_t>(), epoch + 2);
        EXPECT_EQ(terminal["movement"]["edges"].size(), 2u);
        EXPECT_EQ(terminal["movement"]["economy"].size(), 1u);

        auto follower = archive.ledger();
        follower.economy.clear();
        const auto follower_terminal = moeMovementTransportJson(follower);
        EXPECT_TRUE(follower_terminal["movement"]["economy"].empty());
        EXPECT_EQ(follower_terminal["device_publications"], terminal["device_publications"]);
        EXPECT_EQ(follower_terminal["movement"]["edges"], terminal["movement"]["edges"]);
    }

    TEST(MoECompletedMovementPublication, TerminalExportRejectsIncompleteOrReorderedPhysicalHistory)
    {
        NativeMoEMovementArchive archive(publicationGeometry(true));
        PublicationSnapshot snapshot;
        snapshot.commit(9, 6000);
        snapshot.commit(10, 7000);
        snapshot.observe(archive, 1);
        for (int mutation = 0; mutation < 8; ++mutation)
        {
            auto changed = archive.ledger();
            switch (mutation)
            {
            case 0: changed.device_publications.erase(changed.device_publications.begin()); break;
            case 1: changed.device_publications[0] = changed.device_publications[1]; break;
            case 2: std::swap(changed.device_publications[0], changed.device_publications[1]); break;
            case 3: changed.device_publications[0].command_count = 1; break;
            case 4: changed.device_publications[0].physical_payload_bytes = 0; break;
            case 5: changed.discarded_device_publications = 1; break;
            case 6: std::swap(changed.edges[1], changed.edges[2]); break;
            case 7: ++changed.device_publications[0].candidate_epoch; break;
            }
            EXPECT_THROW(moeMovementTransportJson(changed), std::invalid_argument) << mutation;
        }
        EXPECT_NO_THROW(moeMovementTransportJson(archive.ledger()));
        EXPECT_TRUE(moeMovementTransportJson({})["device_publications"].empty());
    }

    TEST(MoECompletedMovementPublication, TerminalFilePublicationPreservesPriorEvidenceOnFailure)
    {
        PublicationDirectory directory;
        const auto perf = directory.path / "rank-7.json";
        const auto sidecar = std::filesystem::path(perf.string() + ".movement.json");
        writeMoEMovementTransportJson(perf, 7, {});
        nlohmann::json original;
        {
            std::ifstream input(sidecar);
            input >> original;
        }
        EXPECT_EQ(original["rank"], 7);
        EXPECT_TRUE(original["device_publications"].empty());
        EXPECT_FALSE(std::filesystem::exists(perf));
        EXPECT_FALSE(std::filesystem::exists(sidecar.string() + ".tmp"));

        MoEOptimizationMovementLedger truncated;
        truncated.discarded_device_publications = 1;
        EXPECT_THROW(writeMoEMovementTransportJson(perf, 7, truncated), std::invalid_argument);
        nlohmann::json retained;
        {
            std::ifstream input(sidecar);
            input >> retained;
        }
        EXPECT_EQ(retained, original);
        EXPECT_FALSE(std::filesystem::exists(sidecar.string() + ".tmp"));

        // A destination directory makes rename fail even when the test runs as
        // root; permissions alone would not be a reliable negative control.
        const auto blocked = directory.path / "blocked.json";
        std::filesystem::create_directory(blocked.string() + ".movement.json");
        EXPECT_THROW(writeMoEMovementTransportJson(blocked, 7, {}), std::filesystem::filesystem_error);
        EXPECT_FALSE(std::filesystem::exists(blocked.string() + ".movement.json.tmp"));
        EXPECT_TRUE(std::filesystem::is_directory(blocked.string() + ".movement.json"));
    }
}
