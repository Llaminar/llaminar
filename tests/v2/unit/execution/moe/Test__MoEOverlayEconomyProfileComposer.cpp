/**
 * @file Test__MoEOverlayEconomyProfileComposer.cpp
 * @brief Device-free tests for measured ExpertOverlay economy certification.
 *
 * These regressions lock down the setup protocol independently of CUDA, ROCm,
 * and CPU timing implementations.  They prove total measurement coverage,
 * conservative participant-to-tier reduction, order-independent identities,
 * strict integer-priority consistency, and complete directed migration costs.
 */

#include "execution/moe/MoEOverlayEconomyProfileComposer.h"
#include "execution/moe/MoEOverlayEconomyCalibrationPlanner.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Build one whole-expert domain with explicit logical endpoints. */
        RoutedExpertDomain domain(
            std::string name,
            std::vector<GlobalDeviceAddress> participants,
            CollectiveBackendType backend)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = participants.size() == 1
                               ? ExecutionDomainScope::SINGLE
                               : ExecutionDomainScope::RANK_LOCAL;
            result.backend = backend;
            result.participants = std::move(participants);
            /* LOCAL participants share this rank; owner_rank is authoritative. */
            result.owner_rank = 0;
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /**
         * @brief Two tiers whose declaration indices, names, and priorities disagree.
         *
         * Tier zero is deliberately the numerically less-preferred coverage
         * tier.  This catches code that accidentally treats declaration order
         * or a thermal-looking label as placement policy.
         */
        MoERoutedExpertPlacementPlan arbitraryPriorityPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "continuation_alpha";
            plan.shared_expert_domain = "continuation_alpha";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.domains = {
                domain(
                    "coverage_zeta",
                    {GlobalDeviceAddress::cpu(1)},
                    CollectiveBackendType::MPI),
                domain(
                    "continuation_alpha",
                    {GlobalDeviceAddress::cuda(0, 0),
                     GlobalDeviceAddress::cuda(1, 0)},
                    CollectiveBackendType::NCCL),
            };
            plan.routed_tiers = {
                RoutedExpertTier{
                    .name = "opaque-zeta",
                    .domain = "coverage_zeta",
                    .priority = 41,
                    .fallback = true,
                },
                RoutedExpertTier{
                    .name = "opaque-alpha",
                    .domain = "continuation_alpha",
                    .priority = -7,
                    .max_experts_per_layer = 4,
                },
            };
            plan.placements = {
                {.layer = 0,
                 .routed_expert_tier = {1, 1, 1, 1, 0, 0}},
                {.layer = 1,
                 .routed_expert_tier = {0, 1, 1, 1, 1, 0}},
            };
            return plan;
        }

        /** @brief Exact model geometry used by the pure certification tests. */
        MoERoutedExpertModelMetadata metadata()
        {
            return {
                .num_layers = 2,
                .num_experts = 6,
                .d_model = 64,
                .routed_intermediate_size = 32,
                .routed_quant_type = "Q8_0",
            };
        }

        /** @brief Build one projection contract for service-equivalence tests. */
        MoEOverlayProjectionWeightManifest serviceProjection(
            ExpertTierWeightProjection projection,
            int N,
            int K)
        {
            return {
                .projection = projection,
                .N = N,
                .K = K,
                .format = ExpertWeightFormat::nativeVnni({
                    .codebook_id = native_vnni_formats::Q4_0.codebook_id,
                    .is_superblock =
                        native_vnni_formats::Q4_0.is_superblock,
                    .present = true,
                }),
            };
        }

        /** @brief Build one complete exact layer contract with selectable width. */
        MoEOverlayLayerWeightManifest serviceLayer(int layer, int width)
        {
            return {
                .layer_idx = layer,
                .projections = {{
                    serviceProjection(
                        ExpertTierWeightProjection::Gate, width, 64),
                    serviceProjection(
                        ExpertTierWeightProjection::Up, width, 64),
                    serviceProjection(
                        ExpertTierWeightProjection::Down, 64, width),
                }},
            };
        }

        /** @brief Append all directed endpoint/layer migration measurements. */
        std::vector<MoEOverlayParticipantLayerMigrationCost>
        completeMigrationMeasurements(int participants, int layers)
        {
            std::vector<MoEOverlayParticipantLayerMigrationCost> result;
            for (int source = 0; source < participants; ++source)
            {
                for (int destination = 0;
                     destination < participants;
                     ++destination)
                {
                    if (source == destination)
                        continue;
                    for (int layer = 0; layer < layers; ++layer)
                    {
                        result.push_back({
                            .source_participant = source,
                            .destination_participant = destination,
                            .layer = layer,
                            .transfer_and_repack_ns =
                                static_cast<uint64_t>(
                                    1000 + source * 100 +
                                    destination * 10 + layer),
                            .inference_interference_ns =
                                static_cast<uint64_t>(
                                    source * 17 + destination * 3 + layer),
                        });
                    }
                }
            }
            return result;
        }

        /** @brief Complete participant service rows with a slower coverage tier. */
        MoEOverlayEconomyMeasurements completeMeasurements()
        {
            MoEOverlayEconomyMeasurements result;
            result.service_measurement_identity =
                "prepared-generation-17/service-events";
            result.migration_measurement_identity =
                "prepared-generation-17/transfer-events";
            /* Participant zero belongs to tier zero (priority 41). */
            result.participant_service = {
                {.participant_id = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {100, 200, 300},
                 .sample_count = {9, 7, 5}},
                {.participant_id = 0,
                 .layer = 1,
                 .nanoseconds_per_activation = {110, 210, 310},
                 .sample_count = {8, 6, 4}},
                /* Participants one and two share tier one (priority -7). */
                {.participant_id = 1,
                 .layer = 0,
                 .nanoseconds_per_activation = {10, 20, 30},
                 .sample_count = {10, 10, 10}},
                {.participant_id = 1,
                 .layer = 1,
                 .nanoseconds_per_activation = {11, 21, 31},
                 .sample_count = {10, 10, 10}},
                {.participant_id = 2,
                 .layer = 0,
                 .nanoseconds_per_activation = {12, 18, 25},
                 .sample_count = {12, 12, 12}},
                {.participant_id = 2,
                 .layer = 1,
                 .nanoseconds_per_activation = {13, 19, 29},
                 .sample_count = {12, 12, 12}},
            };
            result.directed_migration =
                completeMigrationMeasurements(3, 2);
            return result;
        }

        /** @brief Locate one composed tier/layer service row. */
        const MoERoutedTierLayerPhaseServiceCost &serviceRow(
            const MoERoutedTierServiceProfile &profile,
            int tier,
            int layer)
        {
            const auto found = std::find_if(
                profile.costs.begin(),
                profile.costs.end(),
                [&](const auto &row)
                {
                    return row.tier_index == tier && row.layer == layer;
                });
            if (found == profile.costs.end())
                throw std::logic_error("Composed profile omitted a test row");
            return *found;
        }

        /** @brief Locate one retained participant/layer service row. */
        const MoERoutedParticipantLayerPhaseServiceCost &participantServiceRow(
            const MoERoutedTierServiceProfile &profile,
            int participant,
            int layer)
        {
            const auto found = std::find_if(
                profile.participant_costs.begin(),
                profile.participant_costs.end(),
                [&](const auto &row)
                {
                    return row.participant_id == participant &&
                           row.layer == layer;
                });
            if (found == profile.participant_costs.end())
            {
                throw std::logic_error(
                    "Composed profile omitted a participant service row");
            }
            return *found;
        }

        /** @brief Produce one retained projection observation for pure tests. */
        ExpertTierProjectionTransferMeasurement projectionMeasurement(
            std::uint64_t sequence,
            std::uint64_t wall_nanoseconds,
            std::uint64_t device_nanoseconds,
            std::uint64_t host_nanoseconds = 0)
        {
            return {
                .sequence = sequence,
                .bytes = 4096 + sequence,
                .wall_nanoseconds = wall_nanoseconds,
                .device_nanoseconds = device_nanoseconds,
                .host_nanoseconds = host_nanoseconds,
            };
        }
    } // namespace

    TEST(
        MoEOverlayEconomyProfileComposer,
        ConservativeTierCostsAndIdentityIgnoreInputRowOrder)
    {
        const auto plan = arbitraryPriorityPlan();
        const auto model = metadata();
        const auto owner_map = MoEExpertOwnerMap::build(plan);

        auto first_measurements = completeMeasurements();
        auto second_measurements = first_measurements;
        std::reverse(
            second_measurements.participant_service.begin(),
            second_measurements.participant_service.end());
        std::reverse(
            second_measurements.directed_migration.begin(),
            second_measurements.directed_migration.end());

        const auto first = MoEOverlayEconomyProfileComposer::compose(
            plan,
            model,
            owner_map,
            std::move(first_measurements),
            MoEOverlayMigrationEconomyPolicy{});
        const auto second = MoEOverlayEconomyProfileComposer::compose(
            plan,
            model,
            owner_map,
            std::move(second_measurements),
            MoEOverlayMigrationEconomyPolicy{});

        ASSERT_TRUE(first.valid());
        ASSERT_TRUE(second.valid());
        EXPECT_EQ(first.service->identity, second.service->identity);
        EXPECT_EQ(first.migration->identity, second.migration->identity);
        ASSERT_EQ(first.service->costs.size(), second.service->costs.size());
        for (std::size_t index = 0;
             index < first.service->costs.size();
             ++index)
        {
            EXPECT_EQ(
                first.service->costs[index].tier_index,
                second.service->costs[index].tier_index);
            EXPECT_EQ(
                first.service->costs[index].layer,
                second.service->costs[index].layer);
            EXPECT_EQ(
                first.service->costs[index].nanoseconds_per_activation,
                second.service->costs[index].nanoseconds_per_activation);
        }
        ASSERT_EQ(
            first.service->participant_costs.size(),
            second.service->participant_costs.size());
        for (std::size_t index = 0;
             index < first.service->participant_costs.size();
             ++index)
        {
            EXPECT_EQ(
                first.service->participant_costs[index].participant_id,
                second.service->participant_costs[index].participant_id);
            EXPECT_EQ(
                first.service->participant_costs[index].layer,
                second.service->participant_costs[index].layer);
            EXPECT_EQ(
                first.service->participant_costs[index]
                    .nanoseconds_per_activation,
                second.service->participant_costs[index]
                    .nanoseconds_per_activation);
        }
        ASSERT_EQ(
            first.migration->costs.size(),
            second.migration->costs.size());
        for (std::size_t index = 0;
             index < first.migration->costs.size();
             ++index)
        {
            const auto &lhs = first.migration->costs[index];
            const auto &rhs = second.migration->costs[index];
            EXPECT_EQ(lhs.source_participant, rhs.source_participant);
            EXPECT_EQ(
                lhs.destination_participant,
                rhs.destination_participant);
            EXPECT_EQ(lhs.layer, rhs.layer);
            EXPECT_EQ(
                lhs.transfer_and_repack_ns,
                rhs.transfer_and_repack_ns);
            EXPECT_EQ(
                lhs.inference_interference_ns,
                rhs.inference_interference_ns);
        }

        /* Tier one is reduced with max(participant one, participant two). */
        EXPECT_EQ(
            serviceRow(*first.service, 1, 0)
                .nanoseconds_per_activation,
            (std::array<uint64_t, kExpertHistogramProductionSourceCount>{
                12, 20, 30}));
        EXPECT_EQ(
            serviceRow(*first.service, 1, 1)
                .nanoseconds_per_activation,
            (std::array<uint64_t, kExpertHistogramProductionSourceCount>{
                13, 21, 31}));
        EXPECT_EQ(
            serviceRow(*first.service, 0, 0)
                .nanoseconds_per_activation,
            (std::array<uint64_t, kExpertHistogramProductionSourceCount>{
                100, 200, 300}));
        EXPECT_EQ(
            participantServiceRow(*first.service, 1, 0)
                .nanoseconds_per_activation,
            (std::array<uint64_t, kExpertHistogramProductionSourceCount>{
                10, 20, 30}));
        EXPECT_EQ(
            participantServiceRow(*first.service, 2, 0)
                .nanoseconds_per_activation,
            (std::array<uint64_t, kExpertHistogramProductionSourceCount>{
                12, 18, 25}));
    }

    TEST(
        MoEOverlayEconomyProfileComposer,
        RawSiblingEvidenceParticipatesInIdentityEvenWhenTierMaximumDoesNot)
    {
        const auto plan = arbitraryPriorityPlan();
        const auto model = metadata();
        const auto owner_map = MoEExpertOwnerMap::build(plan);
        auto baseline_measurements = completeMeasurements();
        auto changed_measurements = baseline_measurements;
        /* This remains below participant two's tier maximum in phase zero. */
        changed_measurements.participant_service[2]
            .nanoseconds_per_activation[0] = 11;

        const auto baseline = MoEOverlayEconomyProfileComposer::compose(
            plan,
            model,
            owner_map,
            std::move(baseline_measurements),
            {});
        const auto changed = MoEOverlayEconomyProfileComposer::compose(
            plan,
            model,
            owner_map,
            std::move(changed_measurements),
            {});

        ASSERT_EQ(
            baseline.service->costs.size(),
            changed.service->costs.size());
        for (std::size_t index = 0;
             index < baseline.service->costs.size();
             ++index)
        {
            EXPECT_EQ(
                baseline.service->costs[index]
                    .nanoseconds_per_activation,
                changed.service->costs[index]
                    .nanoseconds_per_activation);
        }
        EXPECT_NE(baseline.service->identity, changed.service->identity);
    }

    TEST(
        MoEOverlayEconomyProfileComposer,
        MTPDisabledProfilesKeepGroupedCostZeroAndRejectGroupedDemand)
    {
        const auto plan = arbitraryPriorityPlan();
        const auto model = metadata();
        const auto owner_map = MoEExpertOwnerMap::build(plan);
        auto measurements = completeMeasurements();
        measurements.active_sources = {true, true, false};
        for (auto &row : measurements.participant_service)
        {
            row.nanoseconds_per_activation[2] = 0;
            row.sample_count[2] = 0;
        }
        const auto profiles = MoEOverlayEconomyProfileComposer::compose(
            plan,
            model,
            owner_map,
            std::move(measurements),
            {});
        ASSERT_TRUE(profiles.valid());
        EXPECT_EQ(
            profiles.service->active_sources,
            (ExpertHistogramProductionSourceMask{true, true, false}));
        for (const auto &row : profiles.service->costs)
            EXPECT_EQ(row.nanoseconds_per_activation[2], 0u);

        const std::size_t entries =
            static_cast<std::size_t>(model.num_layers * model.num_experts);
        DecodeExpertHistogramWindow grouped_demand;
        grouped_demand.generation = 1;
        grouped_demand.token_count = 1;
        grouped_demand.source_token_counts[2] = 1;
        grouped_demand.num_layers = model.num_layers;
        grouped_demand.num_experts = model.num_experts;
        grouped_demand.expert_counts.assign(entries, 0);
        grouped_demand.source_expert_counts.assign(
            entries * kExpertHistogramProductionSourceCount,
            0);
        grouped_demand.expert_counts[0] = 1;
        grouped_demand.source_expert_counts[2 * entries] = 1;
        ASSERT_TRUE(grouped_demand.valid());

        MoERoutedExpertPlacementPlannerOptions options;
        options.decode_histogram_window = &grouped_demand;
        options.phase_service_profile = profiles.service.get();
        options.rebalancer.enabled = true;
        options.rebalancer.previous_placements = plan.placements;
        EXPECT_THROW(
            (void)MoERoutedExpertPlacementPlanner::plan(
                plan, model, options),
            std::logic_error);

        auto inconsistent = completeMeasurements();
        inconsistent.active_sources = {true, true, false};
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::compose(
                plan,
                model,
                owner_map,
                std::move(inconsistent),
                {}),
            std::invalid_argument);
    }

    TEST(
        MoEOverlayEconomyProfileComposer,
        RejectsUnsampledDuplicateAndMissingEvidenceButAllowsPhaseCrossovers)
    {
        const auto plan = arbitraryPriorityPlan();
        const auto model = metadata();
        const auto owner_map = MoEExpertOwnerMap::build(plan);

        auto unsampled = completeMeasurements();
        unsampled.participant_service.front().sample_count[1] = 0;
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::compose(
                plan, model, owner_map, std::move(unsampled), {}),
            std::invalid_argument);

        auto duplicated = completeMeasurements();
        duplicated.participant_service.back() =
            duplicated.participant_service.front();
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::compose(
                plan, model, owner_map, std::move(duplicated), {}),
            std::invalid_argument);

        auto missing_transfer = completeMeasurements();
        missing_transfer.directed_migration.pop_back();
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::compose(
                plan,
                model,
                owner_map,
                std::move(missing_transfer),
                {}),
            std::invalid_argument);

        auto contradictory = completeMeasurements();
        /* A lower-priority backend may win one measured phase/geometry. */
        contradictory.participant_service[0]
            .nanoseconds_per_activation = {1, 1, 1};
        contradictory.participant_service[1]
            .nanoseconds_per_activation = {1, 1, 1};
        const auto crossover = MoEOverlayEconomyProfileComposer::compose(
            plan,
            model,
            owner_map,
            std::move(contradictory),
            {});
        EXPECT_TRUE(crossover.valid());
    }

    TEST(
        MoEOverlayEconomyProfileComposer,
        StaticInitialPlacementCanBeCertifiedForDynamicMaintenance)
    {
        auto plan = arbitraryPriorityPlan();
        plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
        const auto model = metadata();
        const auto owner_map = MoEExpertOwnerMap::build(plan);
        const auto profiles = MoEOverlayEconomyProfileComposer::compose(
            plan,
            model,
            owner_map,
            completeMeasurements(),
            {});
        EXPECT_TRUE(profiles.valid());
    }

    TEST(
        MoEOverlayEconomyProfileComposer,
        RawServiceTotalsNormalizeWithConservativeIntegerRounding)
    {
        std::vector<MoEOverlayParticipantLayerServiceTotals> totals{
            {
                .participant_id = 2,
                .layer = 1,
                .total_nanoseconds = {101, 200, 301},
                .activation_count = {2, 4, 3},
                .sample_count = {3, 4, 5},
            },
            {
                .participant_id = 0,
                .layer = 0,
                .total_nanoseconds = {1, 2, 3},
                .activation_count = {8, 2, 2},
                .sample_count = {1, 1, 1},
            },
        };
        const auto normalized =
            MoEOverlayEconomyProfileComposer::normalizeServiceTotals(
                std::move(totals));
        ASSERT_EQ(normalized.size(), 2u);
        EXPECT_EQ(normalized[0].participant_id, 0);
        EXPECT_EQ(normalized[0].layer, 0);
        EXPECT_EQ(
            normalized[0].nanoseconds_per_activation,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{1, 1, 2}));
        EXPECT_EQ(normalized[1].participant_id, 2);
        EXPECT_EQ(normalized[1].layer, 1);
        EXPECT_EQ(
            normalized[1].nanoseconds_per_activation,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{51, 50, 101}));
        EXPECT_EQ(
            normalized[1].sample_count,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{3, 4, 5}));

        MoEOverlayParticipantLayerServiceTotals missing_phase{
            .participant_id = 0,
            .layer = 0,
            .total_nanoseconds = {1, 0, 1},
            .activation_count = {1, 0, 1},
            .sample_count = {1, 0, 1},
        };
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::normalizeServiceTotals(
                {missing_phase}),
            std::invalid_argument);

        auto overflowed = missing_phase;
        overflowed.total_nanoseconds = {1, 1, 1};
        overflowed.activation_count = {1, 1, 1};
        overflowed.sample_count = {1, 1, 1};
        overflowed.overflowed[2] = true;
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::normalizeServiceTotals(
                {overflowed}),
            std::invalid_argument);
    }

    TEST(
        MoEOverlayEconomyProfileComposer,
        EquivalentLayersPoolIntegerEvidenceBeforeNormalization)
    {
        const MoEOverlayEconomyCalibrationLayerCatalog catalog({
            serviceLayer(0, 96),
            serviceLayer(1, 96),
            serviceLayer(2, 128),
        });
        std::vector<MoEOverlayParticipantLayerServiceTotals> totals{
            {.participant_id = 1,
             .layer = 2,
             .total_nanoseconds = {303, 603, 903},
             .activation_count = {3, 3, 3},
             .sample_count = {3, 3, 3}},
            {.participant_id = 0,
             .layer = 1,
             .total_nanoseconds = {100, 200, 300},
             .activation_count = {9, 9, 9},
             .sample_count = {2, 2, 2}},
            {.participant_id = 1,
             .layer = 0,
             .total_nanoseconds = {400, 800, 1200},
             .activation_count = {4, 4, 4},
             .sample_count = {1, 1, 1}},
            {.participant_id = 0,
             .layer = 2,
             .total_nanoseconds = {51, 101, 151},
             .activation_count = {2, 2, 2},
             .sample_count = {1, 1, 1}},
            {.participant_id = 1,
             .layer = 1,
             .total_nanoseconds = {50, 100, 150},
             .activation_count = {6, 6, 6},
             .sample_count = {2, 2, 2}},
            {.participant_id = 0,
             .layer = 0,
             .total_nanoseconds = {100, 200, 300},
             .activation_count = {1, 1, 1},
             .sample_count = {1, 1, 1}},
        };

        const auto normalized = MoEOverlayEconomyProfileComposer::
            normalizeEquivalentServiceTotals(
                std::move(totals),
                kAllExpertHistogramProductionSources,
                catalog);
        ASSERT_EQ(normalized.size(), 6u);

        /* Participant zero's class sums before division: 200 / 10. */
        EXPECT_EQ(
            normalized[0].nanoseconds_per_activation,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{20, 40, 60}));
        EXPECT_EQ(
            normalized[1].nanoseconds_per_activation,
            normalized[0].nanoseconds_per_activation);
        EXPECT_EQ(
            normalized[0].sample_count,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{3, 3, 3}));
        EXPECT_EQ(normalized[2].layer, 2);
        EXPECT_EQ(
            normalized[2].nanoseconds_per_activation,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{26, 51, 76}));

        /* Participant one is pooled independently over the same class. */
        EXPECT_EQ(normalized[3].participant_id, 1);
        EXPECT_EQ(
            normalized[3].nanoseconds_per_activation,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{45, 90, 135}));
        EXPECT_EQ(
            normalized[4].nanoseconds_per_activation,
            normalized[3].nanoseconds_per_activation);
        EXPECT_EQ(
            normalized[5].nanoseconds_per_activation,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{101, 201, 301}));
    }

    TEST(
        MoEOverlayEconomyProfileComposer,
        EquivalentLayersPermitSparseMembersButNeverAnUnmeasuredClass)
    {
        const MoEOverlayEconomyCalibrationLayerCatalog catalog({
            serviceLayer(0, 96),
            serviceLayer(1, 96),
        });
        std::vector<MoEOverlayParticipantLayerServiceTotals> sparse{
            {.participant_id = 0,
             .layer = 0,
             .total_nanoseconds = {101, 201, 301},
             .activation_count = {2, 4, 5},
             .sample_count = {3, 3, 3}},
            {.participant_id = 0, .layer = 1},
        };

        const auto normalized = MoEOverlayEconomyProfileComposer::
            normalizeEquivalentServiceTotals(
                sparse,
                kAllExpertHistogramProductionSources,
                catalog);
        ASSERT_EQ(normalized.size(), 2u);
        EXPECT_EQ(
            normalized[0].nanoseconds_per_activation,
            (std::array<uint64_t,
                        kExpertHistogramProductionSourceCount>{51, 51, 61}));
        EXPECT_EQ(
            normalized[1].nanoseconds_per_activation,
            normalized[0].nanoseconds_per_activation);

        sparse[0] = {.participant_id = 0, .layer = 0};
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::
                normalizeEquivalentServiceTotals(
                    sparse,
                    kAllExpertHistogramProductionSources,
                    catalog),
            std::invalid_argument);
    }

    TEST(
        MoEOverlayEconomyProfileComposer,
        MigrationNormalizationUsesMeasuredWaveCriticalPathNotProjectionSum)
    {
        MoEOverlayParticipantLayerMigrationMeasurement later{
            .source_participant = 2,
            .destination_participant = 0,
            .layer = 1,
            .projections = {
                projectionMeasurement(1, 90, 70, 4),
                projectionMeasurement(2, 120, 95, 6),
                projectionMeasurement(3, 100, 80, 5),
            },
            /* A projection median can conservatively exceed the wave median. */
            .wave_wall_nanoseconds = 110,
            .wave_sample_count = 5,
            .inference_interference_nanoseconds = 0,
            .interference_sample_count = 0,
        };
        MoEOverlayParticipantLayerMigrationMeasurement earlier{
            .source_participant = 0,
            .destination_participant = 1,
            .layer = 0,
            .projections = {
                /* CPU-only movement has host work and no device component. */
                projectionMeasurement(1, 40, 0, 35),
                projectionMeasurement(2, 50, 0, 42),
                projectionMeasurement(3, 45, 0, 39),
            },
            .wave_wall_nanoseconds = 70,
            .wave_sample_count = 3,
            // Transfer profiling is finite setup evidence. Runtime service
            // interference is observed independently by ordinary inference
            // telemetry and must not be folded into this sealed row.
            .inference_interference_nanoseconds = 0,
            .interference_sample_count = 0,
        };

        const auto normalized =
            MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(
                {later, earlier});
        ASSERT_EQ(normalized.size(), 2u);
        EXPECT_EQ(normalized[0].source_participant, 0);
        EXPECT_EQ(normalized[0].destination_participant, 1);
        EXPECT_EQ(normalized[0].transfer_and_repack_ns, 70u);
        EXPECT_EQ(normalized[0].inference_interference_ns, 0u);
        EXPECT_EQ(normalized[1].source_participant, 2);
        EXPECT_EQ(normalized[1].destination_participant, 0);
        /* max(whole wave 110, longest lane 120), never 90+120+100. */
        EXPECT_EQ(normalized[1].transfer_and_repack_ns, 120u);
        EXPECT_EQ(normalized[1].inference_interference_ns, 0u);

        auto undersampled = later;
        undersampled.wave_sample_count = 2;
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::
                normalizeMigrationMeasurements({undersampled}),
            std::invalid_argument);

        auto no_component_work = later;
        no_component_work.projections[0].device_nanoseconds = 0;
        no_component_work.projections[0].host_nanoseconds = 0;
        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::
                normalizeMigrationMeasurements({no_component_work}),
            std::invalid_argument);

        EXPECT_THROW(
            (void)MoEOverlayEconomyProfileComposer::
                normalizeMigrationMeasurements({later, later}),
            std::invalid_argument);
    }

} // namespace llaminar2::test
