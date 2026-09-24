/**
 * @file NodeExpertOverlayParityMovementEvidence.cpp
 * @brief MovementEvidence implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Prove all-GPU placement with the sole device-resident authority.
     *
     * The heterogeneous host authority publishes `committed_waves` and its
     * migration ledger under `moe_overlay_residency`.  An all-GPU topology has
     * no such second authority: the authenticated device-controller command is
     * the placement decision and its completed physical transaction is the
     * evidence. This fold checks numeric-priority promotion/demotion in every
     * Dynamic topology and additionally requires same-priority skew movement
     * whenever one declared priority owns two or more physical participants.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertDeviceResidentMovementEvidence() const -> void
    {
        enum Evidence : size_t
        {
            DomainEnabled,
            StaticChecks,
            EconomyReady,
            CertificationComplete,
            RuntimeEpochParticipants,
            Transactions,
            Commands,
            PhysicalBytes,
            Promotions,
            Demotions,
            SamePriorityMoves,
            TypedLedgerComplete,
            TypedLedgerMalformed,
            TypedLedgerEdges,
            TypedTierResidencyEdges,
            TypedParticipantPlacementEdges,
            TypedCombinedEdges,
            CrossDomainMoves,
            CrossRankMoves,
            CrossBackendMoves,
            MigrationEdges,
            BackgroundNotifications,
            PhysicalOperations,
            ConfiguredCycleCapRecords,
            AdoptedInitialSlots,
            BootstrapSlotsRecycled,
            UniqueImprovingEpochs,
            EconomyAuthorityLedgers,
            EconomyAuthorityPublishedWaves,
            CapacityConservationCertifications,
            TaggedTransactions,
            TaggedCommands,
            TaggedPhysicalBytes,
            TaggedPromotions,
            TaggedDemotions,
            TaggedSamePriorityMoves,
            TaggedCrossDomainMoves,
            TaggedCrossRankMoves,
            TaggedCrossBackendMoves,
            EvidenceViolations,
            EvidenceCount,
        };

        std::array<uint64_t, EvidenceCount> local{};
        local[DomainEnabled] =
            PerfStatsCollector::isDomainEnabled("moe_overlay_controller")
                ? 1u
                : 0u;
        local[RuntimeEpochParticipants] =
            orch_runner_ && orch_runner_->moeRuntimeMovementEpoch() > 0u
                ? 1u
                : 0u;
        ASSERT_NE(orch_runner_, nullptr);
        const auto movement_ledger =
            orch_runner_->moeOptimizationMovementLedger();
        local[TypedLedgerComplete] = movement_ledger.complete() ? 1u : 0u;
        std::map<std::uint64_t, std::uint64_t>
            typed_edges_by_transaction;
        std::map<std::uint64_t, std::set<std::size_t>>
            typed_cycles_by_transaction;
        for (const auto &edge : movement_ledger.edges)
        {
            if (!edge.valid() ||
                edge.authority != MoEOptimizationAuthority::Device ||
                edge.blocking_inference)
            {
                ++local[TypedLedgerMalformed];
                continue;
            }
            ++local[TypedLedgerEdges];
            ++typed_edges_by_transaction[edge.transaction];
            typed_cycles_by_transaction[edge.transaction].insert(
                edge.cycle_index);
            switch (edge.axis)
            {
            case MoEOptimizationMovementAxis::TierResidency:
                ++local[TypedTierResidencyEdges];
                break;
            case MoEOptimizationMovementAxis::ParticipantPlacement:
                ++local[TypedParticipantPlacementEdges];
                break;
            case MoEOptimizationMovementAxis::Combined:
                ++local[TypedCombinedEdges];
                break;
            }
        }
        if (!movement_ledger.economy.empty())
        {
            ++local[EconomyAuthorityLedgers];
            local[EconomyAuthorityPublishedWaves] =
                optimizationStatus().published_movement_waves;
        }
        std::set<std::uint64_t> improving_epochs;
        for (const auto &economy : movement_ledger.economy)
        {
            const auto edge_count =
                typed_edges_by_transaction.find(economy.transaction);
            const auto cycle_count =
                typed_cycles_by_transaction.find(economy.transaction);
            const bool identity_valid =
                economy.valid() &&
                economy.authority == MoEOptimizationAuthority::Device &&
                edge_count != typed_edges_by_transaction.end() &&
                edge_count->second == economy.command_count &&
                cycle_count != typed_cycles_by_transaction.end() &&
                cycle_count->second.size() == economy.cycle_count &&
                improving_epochs.insert(economy.candidate_epoch).second;
            if (!identity_valid)
                ++local[EvidenceViolations];
        }
        const auto add = [&local](Evidence evidence, double value)
        {
            if (value > 0.0 && std::isfinite(value) &&
                value <= static_cast<double>(
                    std::numeric_limits<uint64_t>::max()))
            {
                local[evidence] += static_cast<uint64_t>(value);
            }
        };
        const auto parse_u64 = [](const std::string &text,
                                  uint64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto parse_i64 = [](const std::string &text,
                                  std::int64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto tag_u64 = [&](const PerfStatRecord &record,
                                 const char *name,
                                 uint64_t &value)
        {
            const auto found = record.tags.find(name);
            return found != record.tags.end() &&
                   parse_u64(found->second, value);
        };

        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_controller", "moe_overlay_residency"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter)
                continue;

            if (record.domain == "moe_overlay_residency")
            {
                if (record.name == "economy_certification_complete")
                    add(CertificationComplete, record.value);
                else if (record.name == "physical_fabrics_materialized")
                {
                    add(ConfiguredCycleCapRecords, record.value);
                    uint64_t cycle_cap = 0u;
                    uint64_t stream_cap = 0u;
                    uint64_t adopted_slots = 0u;
                    if (!tag_u64(
                            record,
                            "maximum_concurrent_cycles",
                            cycle_cap) ||
                        cycle_cap !=
                            convergence_migration_transfer_slots_ ||
                        !tag_u64(
                            record,
                            "maximum_execution_streams",
                            stream_cap) ||
                        stream_cap !=
                            convergence_migration_execution_streams_ ||
                        !tag_u64(
                            record,
                            "adopted_initial_slots",
                            adopted_slots))
                    {
                        ++local[EvidenceViolations];
                    }
                    else
                    {
                        local[AdoptedInitialSlots] += adopted_slots;
                    }
                }
                else if (record.name == "bootstrap_live_slots_recycled")
                {
                    add(BootstrapSlotsRecycled, record.value);
                }
                continue;
            }
            if (record.domain != "moe_overlay_controller")
                continue;

            if (record.name == "static_no_movement_transactions")
            {
                uint64_t commands = 1u;
                uint64_t bytes = 1u;
                const auto waits = record.tags.find("inference_stream_waits");
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "movement_commands", commands) &&
                    commands == 0u &&
                    tag_u64(record, "packed_weight_bytes", bytes) &&
                    bytes == 0u && waits != record.tags.end() &&
                    waits->second == "0";
                if (valid)
                    add(StaticChecks, record.value);
                else
                    ++local[EvidenceViolations];
            }
            else if (record.name == "device_economy_ready")
            {
                add(EconomyReady, record.value);
            }
            else if (record.name == "dynamic_movement_transactions")
            {
                uint64_t transaction = 0u;
                uint64_t base_epoch = 0u;
                uint64_t candidate_epoch = 0u;
                uint64_t commands = 0u;
                uint64_t bytes = 0u;
                uint64_t promotions = 0u;
                uint64_t demotions = 0u;
                uint64_t same_priority = 0u;
                uint64_t cross_domain = 0u;
                uint64_t cross_rank = 0u;
                uint64_t cross_backend = 0u;
                uint64_t accepted_cycles = 0u;
                uint64_t service_gain = 0u;
                uint64_t net_benefit = 0u;
                const auto policy_owner = record.tags.find("policy_owner");
                const auto blocking = record.tags.find("blocking_inference");
                const bool valid =
                    record.phase == "maintenance" && record.value > 0.0 &&
                    record.count > 0u &&
                    tag_u64(record, "transaction", transaction) &&
                    transaction > 0u &&
                    tag_u64(record, "base_epoch", base_epoch) &&
                    tag_u64(record, "candidate_epoch", candidate_epoch) &&
                    candidate_epoch == base_epoch + 1u &&
                    tag_u64(record, "movement_commands", commands) &&
                    commands > 0u &&
                    tag_u64(record, "physical_bytes", bytes) && bytes > 0u &&
                    tag_u64(record, "promotions", promotions) &&
                    tag_u64(record, "demotions", demotions) &&
                    tag_u64(record, "same_priority_moves", same_priority) &&
                    tag_u64(record, "cross_domain_moves", cross_domain) &&
                    tag_u64(record, "cross_rank_moves", cross_rank) &&
                    tag_u64(record, "cross_backend_moves", cross_backend) &&
                    tag_u64(record, "accepted_cycles", accepted_cycles) &&
                    accepted_cycles > 0u &&
                    accepted_cycles <=
                        convergence_migration_cycles_per_wave_ &&
                    tag_u64(
                        record,
                        "projected_service_gain_ns",
                        service_gain) &&
                    service_gain > 0u &&
                    tag_u64(
                        record,
                        "projected_net_benefit_ns",
                        net_benefit) &&
                    net_benefit > 0u &&
                    policy_owner != record.tags.end() &&
                    policy_owner->second == "device" &&
                    blocking != record.tags.end() &&
                    blocking->second == "false";
                if (!valid)
                {
                    ++local[EvidenceViolations];
                    continue;
                }
                add(Transactions, record.value);
                add(TaggedTransactions, record.value);
                local[TaggedCommands] += commands;
                local[TaggedPhysicalBytes] += bytes;
                local[TaggedPromotions] += promotions;
                local[TaggedDemotions] += demotions;
                local[TaggedSamePriorityMoves] += same_priority;
                local[TaggedCrossDomainMoves] += cross_domain;
                local[TaggedCrossRankMoves] += cross_rank;
                local[TaggedCrossBackendMoves] += cross_backend;
            }
            else if (record.name == "dynamic_movement_commands")
                add(Commands, record.value);
            else if (record.name == "dynamic_physical_bytes")
                add(PhysicalBytes, record.value);
            else if (record.name == "dynamic_promotions")
                add(Promotions, record.value);
            else if (record.name == "dynamic_demotions")
                add(Demotions, record.value);
            else if (record.name == "dynamic_same_priority_moves")
                add(SamePriorityMoves, record.value);
            else if (record.name == "dynamic_cross_domain_moves")
                add(CrossDomainMoves, record.value);
            else if (record.name == "dynamic_cross_rank_moves")
                add(CrossRankMoves, record.value);
            else if (record.name == "dynamic_cross_backend_moves")
                add(CrossBackendMoves, record.value);
            else if (record.name ==
                     "dynamic_capacity_conservation_certifications")
            {
                uint64_t edges = 0u;
                uint64_t participant_coordinates = 0u;
                uint64_t tier_coordinates = 0u;
                uint64_t malformed = 0u;
                uint64_t participant_violations = 0u;
                uint64_t tier_violations = 0u;
                const auto direction_proxy = record.tags.find(
                    "direction_counts_are_capacity_proof");
                const bool valid =
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "edges_checked", edges) && edges > 0u &&
                    tag_u64(
                        record,
                        "participant_coordinates_checked",
                        participant_coordinates) &&
                    participant_coordinates > 0u &&
                    tag_u64(
                        record,
                        "tier_coordinates_checked",
                        tier_coordinates) &&
                    tier_coordinates > 0u &&
                    tag_u64(record, "malformed_edges", malformed) &&
                    malformed == 0u &&
                    tag_u64(
                        record,
                        "participant_flow_violations",
                        participant_violations) &&
                    participant_violations == 0u &&
                    tag_u64(
                        record,
                        "tier_flow_violations",
                        tier_violations) &&
                    tier_violations == 0u &&
                    direction_proxy != record.tags.end() &&
                    direction_proxy->second == "false";
                if (valid)
                    add(
                        CapacityConservationCertifications,
                        record.value);
                else
                    ++local[EvidenceViolations];
            }
            else if (record.name == "background_notification_batches")
                add(BackgroundNotifications, record.value);
            else if (record.name ==
                     "physical_wave_parallel_operations_started")
                add(PhysicalOperations, record.value);
            else if (record.name == "dynamic_migration_edges")
            {
                std::int64_t source_priority = 0;
                std::int64_t destination_priority = 0;
                uint64_t estimated_bytes = 0u;
                const auto direction = record.tags.find("direction");
                const auto source = record.tags.find("source_priority");
                const auto destination =
                    record.tags.find("destination_priority");
                const auto blocking = record.tags.find("blocking_inference");
                const bool parsed =
                    direction != record.tags.end() &&
                    source != record.tags.end() &&
                    destination != record.tags.end() &&
                    parse_i64(source->second, source_priority) &&
                    parse_i64(destination->second, destination_priority) &&
                    tag_u64(
                        record,
                        "estimated_weight_bytes",
                        estimated_bytes) &&
                    estimated_bytes > 0u;
                const bool direction_valid = parsed &&
                    ((direction->second == "promotion" &&
                      destination_priority < source_priority) ||
                     (direction->second == "demotion" &&
                      destination_priority > source_priority) ||
                     (direction->second == "same_priority" &&
                      destination_priority == source_priority));
                if (!direction_valid || blocking == record.tags.end() ||
                    blocking->second != "false")
                {
                    ++local[EvidenceViolations];
                }
                else
                {
                    add(MigrationEdges, record.value);
                }
            }
        }
        local[UniqueImprovingEpochs] = improving_epochs.size();

        std::array<uint64_t, EvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        const uint64_t ranks = static_cast<uint64_t>(mpiWorldSize());
        ASSERT_EQ(global[DomainEnabled], ranks)
            << "Every all-GPU participant rank must retain device-controller evidence";
        EXPECT_EQ(global[TypedLedgerComplete], ranks)
            << "Every device follower must retain the complete controller-authored movement ledger";
        EXPECT_EQ(global[TypedLedgerMalformed], 0u);
        EXPECT_EQ(global[EvidenceViolations], 0u);
        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[StaticChecks], ranks)
                << "Every Static device follower must certify zero movement";
            EXPECT_EQ(global[Transactions], 0u);
            EXPECT_EQ(global[Commands], 0u);
            EXPECT_EQ(global[PhysicalBytes], 0u);
            EXPECT_EQ(global[Promotions], 0u);
            EXPECT_EQ(global[Demotions], 0u);
            EXPECT_EQ(global[SamePriorityMoves], 0u);
            EXPECT_EQ(global[MigrationEdges], 0u);
            EXPECT_EQ(global[TypedLedgerEdges], 0u);
            EXPECT_EQ(global[UniqueImprovingEpochs], 0u);
            EXPECT_EQ(global[EconomyAuthorityLedgers], 0u);
            EXPECT_EQ(global[EconomyAuthorityPublishedWaves], 0u);
            return;
        }

        EXPECT_GE(global[EconomyReady], ranks)
            << "Every device follower must acquire the certified economy profile";
        EXPECT_GE(global[CertificationComplete], ranks);
        EXPECT_EQ(global[RuntimeEpochParticipants], ranks)
            << "Every rank must observe the completed durable movement epoch";
        EXPECT_GE(global[ConfiguredCycleCapRecords], ranks);
        EXPECT_GT(global[AdoptedInitialSlots], 0u);
        EXPECT_GT(global[BootstrapSlotsRecycled], 0u);
        EXPECT_GE(global[Transactions], ranks);
        EXPECT_EQ(global[EconomyAuthorityLedgers], 1u)
            << "Exactly one device-resident policy leader must own the "
               "admitting economics";
        EXPECT_GE(global[UniqueImprovingEpochs], 1u)
            << "The device policy leader retained no profitable movement epoch";
        EXPECT_EQ(
            global[UniqueImprovingEpochs],
            global[EconomyAuthorityPublishedWaves])
            << "Every device-authority publication must retain exactly one "
               "typed economy proof";
        EXPECT_EQ(
            global[Transactions],
            global[EconomyAuthorityPublishedWaves] * ranks)
            << "Follower telemetry must mirror each device-authority "
               "transaction; it is not an independent economy proof";
        EXPECT_GT(global[Commands], 0u);
        EXPECT_EQ(global[TypedLedgerEdges], global[Commands])
            << "Device-authored movement ledger and authenticated command accounting diverged";
        EXPECT_GT(global[PhysicalBytes], 0u);
        EXPECT_GT(global[BackgroundNotifications], 0u);
        EXPECT_GT(global[PhysicalOperations], 0u);
        EXPECT_GT(global[Promotions], 0u)
            << "Dynamic device policy never promoted a histogram-hot expert";
        EXPECT_GT(global[Demotions], 0u)
            << "Dynamic device policy never demoted an expert to release capacity";
        EXPECT_GT(
            global[TypedTierResidencyEdges] + global[TypedCombinedEdges],
            0u)
            << "Dynamic device policy never completed its tier-residency objective";
        EXPECT_EQ(
            global[CapacityConservationCertifications],
            global[Transactions])
            << "Every device-owned transaction must explicitly conserve participant and tier slots";
        if (dynamicMovementAxisContract(resolvedOverlayPlan()) ==
            DynamicMovementAxisContract::
                PriorityMigrationAndParticipantBalance)
        {
            EXPECT_GT(
                global[TypedParticipantPlacementEdges] +
                    global[TypedCombinedEdges],
                0u)
                << "Dynamic device policy never completed its participant-placement objective";
        }
        else
        {
            EXPECT_EQ(
                global[TypedParticipantPlacementEdges] +
                    global[TypedCombinedEdges],
                0u)
                << "A layout without participant-balancing freedom reported that objective";
        }
        EXPECT_EQ(
            global[CrossDomainMoves],
            global[Promotions] + global[Demotions]);
        EXPECT_EQ(
            global[CrossBackendMoves],
            global[Promotions] + global[Demotions]);
        EXPECT_LE(global[CrossRankMoves], global[Commands]);
        EXPECT_EQ(global[MigrationEdges], global[Commands]);

        EXPECT_EQ(global[TaggedTransactions], global[Transactions]);
        EXPECT_EQ(global[TaggedCommands], global[Commands]);
        EXPECT_EQ(global[TaggedPhysicalBytes], global[PhysicalBytes]);
        EXPECT_EQ(global[TaggedPromotions], global[Promotions]);
        EXPECT_EQ(global[TaggedDemotions], global[Demotions]);
        EXPECT_EQ(
            global[TaggedSamePriorityMoves],
            global[SamePriorityMoves]);
        EXPECT_EQ(global[TaggedCrossDomainMoves], global[CrossDomainMoves]);
        EXPECT_EQ(global[TaggedCrossRankMoves], global[CrossRankMoves]);
        EXPECT_EQ(global[TaggedCrossBackendMoves], global[CrossBackendMoves]);
    }

    /**
     * @brief Prove Dynamic movement used the bounded physical GPU stream pool.
     *
     * The typed handles and real-device integration tests are the stream-
     * ownership authority. This fold is deliberately observability-only: it
     * proves that the real model constructed that path, without treating a
     * counter as evidence that a transfer or movement objective completed.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertPersistentTransferExecutionPoolEvidence() const -> void
    {
        if (!isDynamicResidencyProductionTest())
            return;

        enum Evidence : size_t
        {
            DomainEnabled,
            PoolRecords,
            BoundExecutionStreams,
            EvidenceViolations,
            EvidenceCount,
        };
        std::array<uint64_t, EvidenceCount> local{};
        local[DomainEnabled] =
            PerfStatsCollector::isDomainEnabled("transfer") ? 1u : 0u;

        const auto parse_u64 = [](const std::string &text, uint64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        for (const auto &record : PerfStatsCollector::snapshot({"transfer"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.name !=
                    "persistent_execution_stream_pools_materialized")
            {
                continue;
            }

            uint64_t lane_count = 0u;
            uint64_t new_stream_count = 0u;
            const auto pool = record.tags.find("pool");
            const auto lanes = record.tags.find("lane_count");
            const auto created = record.tags.find("new_stream_count");
            const auto stream_class = record.tags.find("stream_class");
            const bool valid =
                record.phase == "model_setup" &&
                std::isfinite(record.value) && record.value >= 1.0 &&
                record.value <= static_cast<double>(
                                    std::numeric_limits<uint64_t>::max()) &&
                std::floor(record.value) == record.value &&
                pool != record.tags.end() &&
                pool->second == "moe_overlay_physical_fabric" &&
                lanes != record.tags.end() &&
                parse_u64(lanes->second, lane_count) &&
                lane_count >= convergence_migration_execution_streams_ &&
                lane_count % convergence_migration_execution_streams_ == 0u &&
                created != record.tags.end() &&
                parse_u64(created->second, new_stream_count) &&
                new_stream_count <= lane_count &&
                stream_class != record.tags.end() &&
                stream_class->second == "background_maintenance";
            if (!valid)
            {
                ++local[EvidenceViolations];
                continue;
            }

            const uint64_t materializations =
                static_cast<uint64_t>(record.value);
            local[PoolRecords] += materializations;
            if (lane_count <=
                std::numeric_limits<uint64_t>::max() /
                    materializations)
            {
                local[BoundExecutionStreams] +=
                    lane_count * materializations;
            }
            else
            {
                ++local[EvidenceViolations];
            }
        }

        std::array<uint64_t, EvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        EXPECT_EQ(
            global[DomainEnabled],
            static_cast<uint64_t>(mpiWorldSize()))
            << "Every rank must retain transfer-pool path evidence";
        EXPECT_EQ(global[EvidenceViolations], 0u)
            << "The physical transfer stream pool did not preserve its typed cycle bound";
        EXPECT_GT(global[PoolRecords], 0u)
            << "Dynamic ExpertOverlay created no physical GPU execution-stream pool";
        EXPECT_GE(
            global[BoundExecutionStreams],
            static_cast<uint64_t>(convergence_migration_execution_streams_))
            << "The production pool cannot materialize its configured physical migration streams";
    }

    /**
     * @brief Fold and validate static immobility or dynamic movement evidence.
     *
     * Dynamic movement must be a capacity-preserving promotion/demotion cycle
     * across the two distinct domains, MPI ranks, and GPU backends. Static cells
     * must publish their typed immobility check and no positive movement edge.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertResidencyMovementEvidence() const -> void
    {
        assertPersistentTransferExecutionPoolEvidence();
        if (!topologyUsesCpu())
        {
            assertDeviceResidentMovementEvidence();
            return;
        }

        enum Evidence : size_t
        {
            DomainEnabled,
            StaticChecks,
            TransportProfileComplete,
            CertificationComplete,
            AuthorityCertifications,
            MaintenanceCertifications,
            CommittedWaves,
            CommittedMigrations,
            Promotions,
            Demotions,
            SamePriority,
            TypedLedgerComplete,
            TypedLedgerMalformed,
            TypedLedgerEdges,
            TypedTierResidencyEdges,
            TypedParticipantPlacementEdges,
            TypedCombinedEdges,
            CrossDomain,
            CrossRank,
            CrossBackend,
            EstimatedBytes,
            BackgroundNotifications,
            StageFailures,
            CommitFailures,
            FatalFailures,
            BlockingInferenceViolations,
            ImprovingEpochs,
            EconomyAuthorityLedgers,
            EconomyAuthorityPublishedWaves,
            ServiceGainEvidenceViolations,
            NetBenefitEvidenceViolations,
            PriorityDirectionViolations,
            ConfiguredCycleCapRecords,
            CycleCapViolations,
            HostAdmissionAuthorityLedgers,
            HostAdmissionRecords,
            HostPolicyEligibleParticipantCycles,
            HostAdmissionViolations,
            MultiCycleAuthorityWaves,
            AdoptedInitialSlots,
            BootstrapSlotsRecycled,
            PhysicalWavesPrepared,
            PhysicalEvidenceViolations,
            CapacityConservationCertifications,
            CapacityConservationViolations,
            EvidenceCount,
        };

        std::array<uint64_t, EvidenceCount> local{};
        std::map<std::uint64_t, std::uint64_t>
            typed_edges_by_transaction;
        std::map<std::uint64_t, std::set<std::size_t>>
            typed_cycles_by_transaction;
        std::set<std::uint64_t> host_admission_epochs;
        local[DomainEnabled] =
            PerfStatsCollector::isDomainEnabled("moe_overlay_residency")
                ? 1u
                : 0u;
        ASSERT_NE(orch_runner_, nullptr);
        const auto movement_ledger =
            orch_runner_->moeOptimizationMovementLedger();
        local[TypedLedgerComplete] = movement_ledger.complete() ? 1u : 0u;
        for (const auto &edge : movement_ledger.edges)
        {
            if (!edge.valid() ||
                edge.authority != MoEOptimizationAuthority::Host)
            {
                ++local[TypedLedgerMalformed];
                continue;
            }
            ++local[TypedLedgerEdges];
            ++typed_edges_by_transaction[edge.transaction];
            typed_cycles_by_transaction[edge.transaction].insert(
                edge.cycle_index);
            switch (edge.axis)
            {
            case MoEOptimizationMovementAxis::TierResidency:
                ++local[TypedTierResidencyEdges];
                break;
            case MoEOptimizationMovementAxis::ParticipantPlacement:
                ++local[TypedParticipantPlacementEdges];
                break;
            case MoEOptimizationMovementAxis::Combined:
                ++local[TypedCombinedEdges];
                break;
            }
            if (edge.blocking_inference)
                ++local[BlockingInferenceViolations];
        }
        if (!movement_ledger.host_admissions.empty())
            ++local[HostAdmissionAuthorityLedgers];
        for (const auto &admission : movement_ledger.host_admissions)
        {
            const auto physical_cycles =
                typed_cycles_by_transaction.find(admission.transaction);
            const bool identity_valid =
                admission.valid() &&
                admission.authority == MoEOptimizationAuthority::Host &&
                admission.cycle_capacity_kind ==
                    MoEOptimizationCycleCapacityKind::Bounded &&
                admission.maximum_concurrent_cycles ==
                    convergence_migration_cycles_per_wave_ &&
                physical_cycles != typed_cycles_by_transaction.end() &&
                physical_cycles->second.size() ==
                    admission.admitted_physical_cycles &&
                host_admission_epochs.insert(
                    admission.candidate_epoch).second;
            if (!identity_valid)
            {
                ++local[HostAdmissionViolations];
                continue;
            }
            ++local[HostAdmissionRecords];
            local[HostPolicyEligibleParticipantCycles] +=
                admission.policy_eligible_axes
                    .participant_placement +
                admission.policy_eligible_axes.combined;
            if (admission.admitted_physical_cycles > 1u)
                ++local[MultiCycleAuthorityWaves];
        }
        if (!movement_ledger.economy.empty())
        {
            ++local[EconomyAuthorityLedgers];
            local[EconomyAuthorityPublishedWaves] =
                optimizationStatus().published_movement_waves;
        }
        std::set<std::uint64_t> economy_epochs;
        for (const auto &economy : movement_ledger.economy)
        {
            const auto edge_count =
                typed_edges_by_transaction.find(economy.transaction);
            const auto cycle_count =
                typed_cycles_by_transaction.find(economy.transaction);
            const bool identity_valid =
                economy.valid() &&
                economy.authority == MoEOptimizationAuthority::Host &&
                economy.transaction == economy.candidate_epoch &&
                edge_count != typed_edges_by_transaction.end() &&
                edge_count->second == economy.command_count &&
                cycle_count != typed_cycles_by_transaction.end() &&
                cycle_count->second.size() == economy.cycle_count &&
                host_admission_epochs.contains(economy.candidate_epoch) &&
                economy_epochs.insert(economy.candidate_epoch).second;
            if (!identity_valid)
            {
                ++local[ServiceGainEvidenceViolations];
                ++local[NetBenefitEvidenceViolations];
                continue;
            }
            ++local[ImprovingEpochs];
        }
        for (const std::uint64_t candidate_epoch : host_admission_epochs)
        {
            if (!economy_epochs.contains(candidate_epoch))
                ++local[HostAdmissionViolations];
        }
        const auto addValue = [&local](Evidence index, double value)
        {
            if (value > 0.0)
                local[index] += static_cast<uint64_t>(value);
        };
        const auto parse_u64 = [](const std::string &text,
                                  uint64_t &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto parse_int = [](const std::string &text, int &value)
        {
            const char *const begin = text.data();
            const char *const end = begin + text.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end;
        };
        const auto tag_u64 = [&parse_u64](
                                 const PerfStatRecord &record,
                                 const char *name,
                                 uint64_t &value)
        {
            const auto found = record.tags.find(name);
            return found != record.tags.end() &&
                   parse_u64(found->second, value);
        };
        for (const auto &record :
             PerfStatsCollector::snapshot(
                 {"moe_overlay_residency", "moe_overlay_controller"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }

            if (record.domain == "moe_overlay_controller" &&
                record.name == "static_no_movement_transactions")
            {
                const auto movement_commands =
                    record.tags.find("movement_commands");
                const auto packed_weight_bytes =
                    record.tags.find("packed_weight_bytes");
                const auto inference_stream_waits =
                    record.tags.find("inference_stream_waits");
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    movement_commands != record.tags.end() &&
                    movement_commands->second == "0" &&
                    packed_weight_bytes != record.tags.end() &&
                    packed_weight_bytes->second == "0" &&
                    inference_stream_waits != record.tags.end() &&
                    inference_stream_waits->second == "0";
                if (valid)
                    addValue(StaticChecks, record.value);
                else
                    ++local[PhysicalEvidenceViolations];
                continue;
            }
            if (record.domain != "moe_overlay_residency")
                continue;

            if (record.name == "static_no_movement_checks")
                addValue(StaticChecks, record.value);
            else if (record.name == "production_maintenance_composed")
            {
                addValue(ConfiguredCycleCapRecords, record.value);
                const auto cap_tag =
                    record.tags.find("migration_transfer_slots");
                uint64_t cap = 0;
                const uint64_t expected_cap =
                    isDynamicResidencyProductionTest()
                        ? convergence_migration_transfer_slots_
                        : 1u;
                if (cap_tag == record.tags.end() ||
                    !parse_u64(cap_tag->second, cap) ||
                    cap != expected_cap)
                {
                    ++local[CycleCapViolations];
                }
            }
            else if (record.name == "physical_fabrics_materialized")
            {
                const auto adopted_tag =
                    record.tags.find("adopted_initial_slots");
                uint64_t adopted = 0;
                if (adopted_tag == record.tags.end() ||
                    !parse_u64(adopted_tag->second, adopted))
                {
                    ++local[PhysicalEvidenceViolations];
                }
                else
                {
                    local[AdoptedInitialSlots] += adopted;
                }
            }
            else if (record.name == "economy_transport_profile_complete")
            {
                const auto synthetic =
                    record.tags.find("synthetic_inference");
                const auto publishes_residency =
                    record.tags.find("publish_residency");
                const auto waves_tag = record.tags.find("waves");
                const auto elapsed_tag =
                    record.tags.find("elapsed_nanoseconds");
                const auto origin_tag = record.tags.find("origin");
                const auto identity_tag =
                    record.tags.find("measurement_identity");
                const auto coordinate_count_tag =
                    record.tags.find("coordinate_count");
                uint64_t waves = 0u;
                uint64_t elapsed = 0u;
                uint64_t coordinate_count = 0u;
                const bool origin_valid =
                    origin_tag != record.tags.end() &&
                    (origin_tag->second == "measured" ||
                     origin_tag->second == "reused");
                const bool valid =
                    record.phase == "model_setup" &&
                    record.value == 1.0 && record.count == 1u &&
                    synthetic != record.tags.end() &&
                    synthetic->second == "false" &&
                    publishes_residency != record.tags.end() &&
                    publishes_residency->second == "false" &&
                    waves_tag != record.tags.end() &&
                    parse_u64(waves_tag->second, waves) && waves > 0u &&
                    elapsed_tag != record.tags.end() &&
                    parse_u64(elapsed_tag->second, elapsed) &&
                    origin_valid &&
                    identity_tag != record.tags.end() &&
                    !identity_tag->second.empty() &&
                    coordinate_count_tag != record.tags.end() &&
                    parse_u64(
                        coordinate_count_tag->second, coordinate_count) &&
                    coordinate_count > 0u &&
                    ((origin_tag->second == "measured" && elapsed > 0u) ||
                     (origin_tag->second == "reused" && elapsed == 0u));
                if (valid)
                    addValue(TransportProfileComplete, record.value);
                else
                    ++local[PhysicalEvidenceViolations];
            }
            else if (record.name == "economy_certification_complete")
                addValue(CertificationComplete, record.value);
            else if (record.name == "economy_certifications")
                addValue(AuthorityCertifications, record.value);
            else if (record.name == "maintenance_economy_certifications")
                addValue(MaintenanceCertifications, record.value);
            else if (record.name == "committed_waves")
                addValue(CommittedWaves, record.value);
            else if (record.name == "committed_expert_migrations")
                addValue(CommittedMigrations, record.value);
            else if (record.name == "committed_migration_cycles")
            {
                if (record.value >
                        static_cast<double>(
                            convergence_migration_cycles_per_wave_))
                    ++local[CycleCapViolations];
            }
            else if (record.name == "bootstrap_live_slots_recycled")
                addValue(BootstrapSlotsRecycled, record.value);
            else if (record.name == "physical_waves_prepared")
                addValue(PhysicalWavesPrepared, record.value);
            else if (record.name ==
                     "capacity_conservation_certifications")
            {
                uint64_t edges = 0u;
                uint64_t cycles = 0u;
                uint64_t participant_coordinates = 0u;
                uint64_t tier_coordinates = 0u;
                uint64_t malformed = 0u;
                uint64_t participant_violations = 0u;
                uint64_t tier_violations = 0u;
                const auto direction_proxy = record.tags.find(
                    "direction_counts_are_capacity_proof");
                const bool valid =
                    record.value == 1.0 && record.count == 1u &&
                    tag_u64(record, "edges_checked", edges) && edges > 0u &&
                    tag_u64(record, "closed_cycles", cycles) && cycles > 0u &&
                    tag_u64(
                        record,
                        "participant_coordinates_checked",
                        participant_coordinates) &&
                    participant_coordinates > 0u &&
                    tag_u64(
                        record,
                        "tier_coordinates_checked",
                        tier_coordinates) &&
                    tier_coordinates > 0u &&
                    tag_u64(record, "malformed_edges", malformed) &&
                    tag_u64(
                        record,
                        "participant_flow_violations",
                        participant_violations) &&
                    tag_u64(
                        record,
                        "tier_flow_violations",
                        tier_violations) &&
                    direction_proxy != record.tags.end() &&
                    direction_proxy->second == "false";
                if (valid)
                    addValue(
                        CapacityConservationCertifications,
                        record.value);
                if (!valid || malformed != 0u ||
                    participant_violations != 0u || tier_violations != 0u)
                {
                    ++local[CapacityConservationViolations];
                }
            }
            else if (record.name == "promotions")
                addValue(Promotions, record.value);
            else if (record.name == "demotions")
                addValue(Demotions, record.value);
            else if (record.name == "same_priority_moves")
                addValue(SamePriority, record.value);
            else if (record.name == "cross_domain_migrations")
                addValue(CrossDomain, record.value);
            else if (record.name == "cross_rank_migrations")
                addValue(CrossRank, record.value);
            else if (record.name == "cross_backend_migrations")
                addValue(CrossBackend, record.value);
            else if (record.name == "estimated_weight_bytes")
                addValue(EstimatedBytes, record.value);
            else if (record.name == "decode_boundary_background_notifications")
                addValue(BackgroundNotifications, record.value);
            else if (record.name == "migration_stage_failures")
                addValue(StageFailures, record.value);
            else if (record.name == "migration_commit_failures")
                addValue(CommitFailures, record.value);
            else if (record.name == "maintenance_fatal_failures")
                addValue(FatalFailures, record.value);
            else if (record.name == "expert_migration_edges")
            {
                const auto direction = record.tags.find("direction");
                const auto source = record.tags.find("source_priority");
                const auto destination =
                    record.tags.find("destination_priority");
                int source_priority = 0;
                int destination_priority = 0;
                uint64_t cycle_index = 0u;
                uint64_t cycle_size = 0u;
                bool direction_valid =
                    direction != record.tags.end() &&
                    source != record.tags.end() &&
                    destination != record.tags.end() &&
                    parse_int(source->second, source_priority) &&
                    parse_int(destination->second, destination_priority) &&
                    tag_u64(record, "cycle_index", cycle_index) &&
                    tag_u64(record, "cycle_size", cycle_size) &&
                    cycle_size > 0u;
                if (direction_valid)
                {
                    direction_valid =
                        (direction->second == "promotion" &&
                         destination_priority < source_priority) ||
                        (direction->second == "demotion" &&
                         destination_priority > source_priority) ||
                        (direction->second == "same_priority" &&
                         destination_priority == source_priority);
                }
                if (!direction_valid)
                    ++local[PriorityDirectionViolations];
            }

            const auto blocking = record.tags.find("blocking");
            const auto blocking_inference =
                record.tags.find("blocking_inference");
            const auto inference_path = record.tags.find("inference_path");
            const bool setup_only_blocking =
                blocking != record.tags.end() && blocking->second == "true" &&
                record.phase == "model_setup" &&
                inference_path != record.tags.end() &&
                inference_path->second == "false";
            if ((blocking != record.tags.end() && blocking->second == "true" &&
                 !setup_only_blocking) ||
                (blocking_inference != record.tags.end() &&
                 blocking_inference->second != "false"))
            {
                ++local[BlockingInferenceViolations];
            }
        }

        std::array<uint64_t, EvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        ASSERT_EQ(global[DomainEnabled], static_cast<uint64_t>(mpiWorldSize()))
            << "Every rank must retain ExpertOverlay residency PerfStats";
        EXPECT_EQ(
            global[TypedLedgerComplete],
            static_cast<uint64_t>(mpiWorldSize()))
            << "Every rank must retain its complete authority-owned movement ledger";
        EXPECT_EQ(global[TypedLedgerMalformed], 0u)
            << "The authority-owned movement ledger contained a malformed or wrongly owned edge";
        EXPECT_EQ(global[StageFailures], 0u);
        EXPECT_EQ(global[CommitFailures], 0u);
        EXPECT_EQ(global[FatalFailures], 0u);
        EXPECT_EQ(global[BlockingInferenceViolations], 0u)
            << "Expert movement exposed a blocking inference-path tag";
        EXPECT_EQ(global[PriorityDirectionViolations], 0u)
            << "Migration direction did not match numeric tier priority";

        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[StaticChecks], static_cast<uint64_t>(mpiWorldSize()))
                << "Every static authority must prove immobility exactly once";
            EXPECT_EQ(global[CommittedWaves], 0u);
            EXPECT_EQ(global[CommittedMigrations], 0u)
                << "Static ExpertOverlay placement moved an expert";
            EXPECT_EQ(global[Promotions], 0u);
            EXPECT_EQ(global[Demotions], 0u);
            EXPECT_EQ(global[SamePriority], 0u);
            EXPECT_EQ(global[TypedLedgerEdges], 0u)
                << "Static ExpertOverlay published an authoritative movement edge";
            EXPECT_EQ(global[ImprovingEpochs], 0u);
            EXPECT_EQ(global[EconomyAuthorityLedgers], 0u)
                << "Static ExpertOverlay published a movement-economy ledger";
            EXPECT_EQ(global[EconomyAuthorityPublishedWaves], 0u);
            EXPECT_EQ(global[HostAdmissionAuthorityLedgers], 0u)
                << "Static ExpertOverlay published a host-policy admission ledger";
            EXPECT_EQ(global[HostAdmissionRecords], 0u);
            EXPECT_EQ(global[HostAdmissionViolations], 0u);
            EXPECT_EQ(global[BootstrapSlotsRecycled], 0u)
                << "Static ExpertOverlay recycled no bootstrap assignment";
            return;
        }

        const uint64_t ranks = static_cast<uint64_t>(mpiWorldSize());
        EXPECT_GE(global[ConfiguredCycleCapRecords], ranks)
            << "Every rank must publish its production cycle cap";
        EXPECT_EQ(global[CycleCapViolations], 0u)
            << "The configured migration transfer-slot policy was not preserved through production composition";
        EXPECT_EQ(global[HostAdmissionAuthorityLedgers], 1u)
            << "Exactly one host policy authority must own candidate, policy, and capacity admission accounting";
        EXPECT_EQ(global[HostAdmissionViolations], 0u)
            << "Migration admission left a cycle unclassified or mislabeled an economic rejection as capacity pressure";
        EXPECT_EQ(global[PhysicalEvidenceViolations], 0u)
            << "The physical fabric published malformed slot-adoption evidence";
        if (activeOverlayTierCount() >= 3u)
        {
            EXPECT_GT(global[MultiCycleAuthorityWaves], 0u)
                << "The three-tier real-model campaign never used concurrent "
                   "migration transfer slots";
        }
        EXPECT_GT(global[AdoptedInitialSlots], 0u)
            << "The physical fabric did not adopt loader-owned expert slots";
        EXPECT_GT(global[BootstrapSlotsRecycled], 0u)
            << "Movement never recycled a retired loader-owned expert slot";
        EXPECT_GT(global[PhysicalWavesPrepared], 0u)
            << "No physical migration wave reached background preparation";
        EXPECT_GE(global[TransportProfileComplete], ranks)
            << "Every dynamic authority must finish bounded transport profiling without synthetic inference";
        EXPECT_GE(global[CertificationComplete], ranks);
        EXPECT_GE(global[AuthorityCertifications], ranks);
        EXPECT_GE(global[MaintenanceCertifications], ranks);
        const uint64_t minimum_epochs_per_rank =
            dynamicResidencyConvergenceTarget().minimum_published_waves;
        EXPECT_EQ(global[EconomyAuthorityLedgers], 1u)
            << "Exactly one host policy authority must own the admitting "
               "economics for a distributed movement protocol";
        EXPECT_GE(
            global[ImprovingEpochs],
            minimum_epochs_per_rank)
            << "The host policy authority did not retain the required distinct "
               "profitable publication epochs";
        EXPECT_EQ(
            global[ImprovingEpochs],
            global[EconomyAuthorityPublishedWaves])
            << "Every authority-published wave must retain exactly one typed "
               "economy proof";
        EXPECT_EQ(
            global[HostAdmissionRecords],
            global[EconomyAuthorityPublishedWaves])
            << "Every authority-published wave must retain exactly one typed "
               "host admission proof; followers must not duplicate it";
        EXPECT_EQ(
            global[CommittedWaves],
            global[EconomyAuthorityPublishedWaves] * ranks)
            << "Follower telemetry must mirror each authority-published wave; "
               "it is not an independent policy proof";
        EXPECT_EQ(global[ServiceGainEvidenceViolations], 0u)
            << "Every authority-owned economy record must improve measured "
               "service cost and match its completed movement transaction";
        EXPECT_EQ(global[NetBenefitEvidenceViolations], 0u)
            << "Every authority-owned economy record must remain profitable "
               "after movement cost and inference interference";
        EXPECT_GT(global[CommittedMigrations], 0u);
        EXPECT_EQ(global[TypedLedgerEdges], global[CommittedMigrations])
            << "Typed authority movement and physical commit accounting diverged";
        EXPECT_GT(global[Promotions], 0u);
        EXPECT_GT(global[Demotions], 0u);
        EXPECT_GT(
            global[TypedTierResidencyEdges] + global[TypedCombinedEdges],
            0u)
            << "Dynamic residency never completed its tier-residency objective";
        EXPECT_EQ(global[CapacityConservationViolations], 0u)
            << "A committed wave violated typed per-tier or per-participant slot flow";
        EXPECT_EQ(
            global[CapacityConservationCertifications],
            global[CommittedWaves])
            << "Every committed wave must carry one explicit flow-conservation proof";
        if (dynamicMovementAxisContract(resolvedOverlayPlan()) ==
            DynamicMovementAxisContract::
                PriorityMigrationAndParticipantBalance)
        {
            const std::uint64_t completed_participant_edges =
                global[TypedParticipantPlacementEdges] +
                global[TypedCombinedEdges];
            if (global[HostPolicyEligibleParticipantCycles] > 0u)
            {
                EXPECT_GT(completed_participant_edges, 0u)
                    << "A host-policy participant cycle survived measured "
                       "economics but never completed physical movement";
            }
            else
            {
                EXPECT_EQ(completed_participant_edges, 0u)
                    << "The movement ledger reported a participant objective "
                       "that no typed host admission classified as economical";
                EXPECT_GE(
                    global[HostAdmissionRecords],
                    minimum_epochs_per_rank)
                    << "Tier-only convergence requires a complete typed "
                       "policy-admission proof for every required wave";
            }
        }
        else
        {
            EXPECT_EQ(
                global[TypedParticipantPlacementEdges] +
                    global[TypedCombinedEdges],
                0u)
                << "A layout without a participant-balancing degree of freedom reported that objective";
        }
        EXPECT_EQ(
            global[CommittedMigrations],
            global[Promotions] + global[Demotions] +
                global[SamePriority]);
        EXPECT_EQ(
            global[CrossDomain],
            global[Promotions] + global[Demotions]);
        EXPECT_EQ(
            global[CrossBackend],
            global[Promotions] + global[Demotions]);
        const auto &test_case = activeModelParityCaseOrThrow();
        if (test_case.topology.mpi_ranks == 1)
        {
            EXPECT_EQ(global[CrossRank], 0u)
                << "A rank-local overlay reported cross-rank movement";
        }
        else
        {
            if (topologyUsesCpu() &&
                activeTypedParticipantCount(
                    [](const GlobalDeviceAddress &participant)
                    { return participant.isCPU(); }) > 1u)
            {
                EXPECT_GT(global[CrossRank], 0u)
                    << "The two-rank CPU tier never exercised distributed movement";
            }
            EXPECT_LE(global[CrossRank], global[CommittedMigrations]);
        }
        EXPECT_GT(global[EstimatedBytes], 0u);
        EXPECT_GT(global[BackgroundNotifications], 0u)
            << "Inference never exercised the wake-only maintenance boundary";
    }

    /**
     * @brief Prove request-local LLEP movement or static immobility.
     *
     * Durable tier migration is certified by moe_overlay_residency above.
     * LLEP additionally owns a request-scoped assignment/copy/apply graph, so
     * its proof must come from the independent moe_rebalance counters. Static
     * cells check the same counters at zero; setup-time initial placement is
     * intentionally outside this request-movement surface.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertRequestMovementPolicyEvidence() const -> void
    {
        enum Counter : size_t
        {
            LLEPAssignments,
            LLEPMovementLayers,
            CopiedArrivals,
            AppliedArrivals,
            UsefulPayloadBytes,
            CounterCount,
        };
        std::array<uint64_t, CounterCount> local{};
        const auto add = [&local](Counter counter, double value)
        {
            if (value > 0.0)
                local[counter] += static_cast<uint64_t>(value);
        };
        for (const auto &record :
             PerfStatsCollector::snapshot({"moe_rebalance"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_rebalance")
            {
                continue;
            }
            if (record.name ==
                "device_rebalance_llep_resident_assignment_calls")
                add(LLEPAssignments, record.value);
            else if (record.name ==
                     "device_rebalance_prefill_current_batch_movement_layers")
                add(LLEPMovementLayers, record.value);
            else if (record.name ==
                         "device_rebalance_copy_copied_arrivals" ||
                     record.name ==
                         "device_rebalance_transfer_current_copied_arrivals" ||
                     record.name ==
                         "device_rebalance_wave_copied_arrivals_total")
                add(CopiedArrivals, record.value);
            else if (record.name ==
                         "device_rebalance_apply_applied_arrivals" ||
                     record.name ==
                         "device_rebalance_transfer_current_applied_arrivals" ||
                     record.name ==
                         "device_rebalance_wave_applied_arrivals_total")
                add(AppliedArrivals, record.value);
            else if (record.name ==
                         "device_rebalance_transfer_useful_payload_bytes" ||
                     record.name ==
                         "device_rebalance_request_useful_payload_bytes_lower_bound")
                add(UsefulPayloadBytes, record.value);
        }

        std::array<uint64_t, CounterCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank() || !isQwen122ProductionTest())
            return;

        if (isLLEPProductionTest())
        {
            EXPECT_GT(global[LLEPAssignments], 0u)
                << "LLEP ran no least-loaded resident assignment";
            EXPECT_GT(global[LLEPMovementLayers], 0u)
                << "LLEP moved no expert payload for the current prefill";
            EXPECT_GT(global[CopiedArrivals], 0u);
            EXPECT_GT(global[AppliedArrivals], 0u);
            EXPECT_GT(global[UsefulPayloadBytes], 0u);
            return;
        }

        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(global[LLEPAssignments], 0u);
            EXPECT_EQ(global[LLEPMovementLayers], 0u);
            EXPECT_EQ(global[CopiedArrivals], 0u)
                << "Static placement copied a request-time expert payload";
            EXPECT_EQ(global[AppliedArrivals], 0u)
                << "Static placement applied a request-time expert payload";
            EXPECT_EQ(global[UsefulPayloadBytes], 0u)
                << "Static placement transferred request-time expert bytes";
        }
    }

}
