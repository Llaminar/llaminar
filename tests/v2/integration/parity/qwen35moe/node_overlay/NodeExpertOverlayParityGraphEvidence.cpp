/**
 * @file NodeExpertOverlayParityGraphEvidence.cpp
 * @brief GraphEvidence implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 * Logical ticket proof is distinct from the retained parent's compiler-shard
 * count; the shared evidence interpreter authenticates that distinction.
 */
#include "NodeExpertOverlayParityFixture.h"
#include "../../../../utils/HeterogeneousTicketParityEvidence.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Prove each expert-only endpoint owns one immutable compact arena.
     *
     * Every process retains its own PerfStats records, while the graph-native
     * overlay spreads its accelerator tier and the two CPU-NUMA endpoints
     * across MPI instances. The dense continuation uses its model graph's
     * activation arena and must not duplicate the follower runner's storage.
     * Aggregate setup evidence only after the worker loop closes; this proves
     * the real follower graph did not allocate one packet per layer or alias
     * independent participants.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertParticipantCompactBufferArenaEvidence() const -> void
    {
        constexpr size_t kAllocationValue = 0;
        constexpr size_t kAllocationRecordCount = 1;
        constexpr size_t kAllocationBytes = 2;
        constexpr size_t kByteRecordCount = 3;
        constexpr size_t kMalformedRecords = 4;
        constexpr size_t kParticipantStart = 5;
        const size_t participant_count = activeOverlayParticipantCount();
        ASSERT_GT(participant_count, 1u);
        /*
         * A continuation domain may itself be tensor-parallel. Every one of
         * those participants reuses its dense model arena; only participants
         * outside that domain own the compact follower arena. Derive the exact
         * set from the published owner-map topology instead of assuming one
         * continuation participant at global id zero.
         */
        std::vector<uint64_t> expected_compact_participant(
            participant_count, 0u);
        int topology_valid = 1;
        if (isRootParityRank())
        {
            const auto *const concrete =
                dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
            const auto snapshot = concrete
                                      ? concrete
                                            ->expertOverlayResidencySnapshotForDiagnostics()
                                      : nullptr;
            if (!snapshot || !snapshot->valid() || !overlay_plan_)
            {
                topology_valid = 0;
            }
            else
            {
                for (const auto &participant :
                     snapshot->owner_map.participants())
                {
                    if (participant.participant_id < 0 ||
                        static_cast<size_t>(participant.participant_id) >=
                            participant_count)
                    {
                        topology_valid = 0;
                        break;
                    }
                    if (participant.domain_name !=
                        overlay_plan_->continuation_domain)
                    {
                        expected_compact_participant[
                            static_cast<size_t>(participant.participant_id)] =
                            1u;
                    }
                }
            }
        }
        MPI_Bcast(
            &topology_valid,
            1,
            MPI_INT,
            parityArtifactAuthorityRank(),
            parityCoordinationCommunicator());
        MPI_Bcast(
            expected_compact_participant.data(),
            static_cast<int>(expected_compact_participant.size()),
            MPI_UINT64_T,
            parityArtifactAuthorityRank(),
            parityCoordinationCommunicator());
        ASSERT_EQ(topology_valid, 1)
            << "Published ExpertOverlay topology could not classify compact followers";
        const size_t follower_participant_count =
            static_cast<size_t>(std::accumulate(
                expected_compact_participant.begin(),
                expected_compact_participant.end(),
                uint64_t{0}));
        ASSERT_GT(follower_participant_count, 0u);
        const size_t evidence_count = kParticipantStart + participant_count;

        const int local_memory_domain_enabled =
            PerfStatsCollector::isDomainEnabled("memory") ? 1 : 0;
        int all_memory_domains_enabled = 0;
        MPI_Allreduce(
            &local_memory_domain_enabled,
            &all_memory_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            parityCoordinationCommunicator());
        if (all_memory_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the PerfStats "
                       "memory domain to prove serial compact-arena ownership";
            }
            return;
        }

        std::vector<uint64_t> local(evidence_count, 0u);
        for (const auto &record : PerfStatsCollector::snapshot({"memory"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "memory")
            {
                continue;
            }

            const bool is_allocation =
                record.name == "moe_serial_local_expert_buffer_arena_allocations";
            const bool is_bytes =
                record.name == "moe_serial_local_expert_buffer_arena_bytes";
            if (!is_allocation && !is_bytes)
                continue;

            const auto ownership = record.tags.find("ownership");
            const auto immutable = record.tags.find("immutable");
            const auto participant = record.tags.find("participant");
            const bool tag_contract_ok =
                record.phase == "model_setup" &&
                ownership != record.tags.end() &&
                ownership->second == "per_device_participant_serial_graph_family" &&
                immutable != record.tags.end() && immutable->second == "true" &&
                participant != record.tags.end();
            if (!tag_contract_ok || record.count != 1u || record.value <= 0.0)
            {
                ++local[kMalformedRecords];
                continue;
            }

            int participant_id = -1;
            for (int candidate = 0;
                 candidate < static_cast<int>(participant_count);
                 ++candidate)
            {
                if (participant->second == std::to_string(candidate))
                {
                    participant_id = candidate;
                    break;
                }
            }
            if (participant_id < 0)
            {
                ++local[kMalformedRecords];
                continue;
            }

            if (is_allocation)
            {
                if (record.value != 1.0)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                ++local[kAllocationValue];
                ++local[kAllocationRecordCount];
                ++local[kParticipantStart + static_cast<size_t>(participant_id)];
                continue;
            }

            local[kAllocationBytes] += static_cast<uint64_t>(record.value);
            ++local[kByteRecordCount];
        }

        std::vector<uint64_t> global(evidence_count, 0u);
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());

        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Serial compact-arena PerfStats tags or aggregation were malformed";
        EXPECT_EQ(global[kAllocationValue], follower_participant_count)
            << "Each expert-only follower must allocate one compact graph-family arena";
        EXPECT_EQ(global[kAllocationRecordCount], follower_participant_count)
            << "Each expert-only follower must publish one unaggregated arena allocation record";
        EXPECT_EQ(global[kByteRecordCount], follower_participant_count)
            << "Each expert-only follower must publish one arena byte-accounting record";
        EXPECT_GT(global[kAllocationBytes], 0u)
            << "Follower compact arenas must account for their fixed graph-family storage";
        for (size_t participant = 0;
             participant < participant_count;
             ++participant)
        {
            const uint64_t expected =
                expected_compact_participant[participant];
            EXPECT_EQ(global[kParticipantStart + participant], expected)
                << (expected != 0u
                        ? "Expected exactly one follower compact arena for participant p"
                        : "Dense continuation participant must not duplicate a compact arena for p")
                << participant;
        }
    }

    /**
     * @brief Prove a one-rank GPU+CPU graph used canonical ticket boundaries.
     *
     * A rank-local topology has no auxiliary MPI runner and must not advertise
     * one merely to satisfy the mapped-follower evidence used by node-wide
     * cases. Instead, every MoE layer must materialize and capture its exact
     * canonical-ticket consumer, while the full graph must lower to typed
     * captured/manual/captured transactions with authenticated successors.
     * Compiler shards and the combined CPU service are not logical boundaries.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertRankLocalCanonicalTicketGraphEvidence() const -> void
    {
        constexpr size_t kMalformedRecords = 0u;
        constexpr size_t kLifecycleTransactions = 1u;
        constexpr size_t kMappedFollowerRecords = 2u;
        constexpr size_t kMaterializations = 3u;
        constexpr size_t kCaptureLaunches = 4u;
        constexpr size_t kFixedEvidenceCount = 5u;
        /*
         * Setup capacity and live execution policy are deliberately distinct.
         * Every generated 122B cell retains the maximum MTP graph-family
         * envelope so adjacent cells can reuse one ModelContext, including the
         * MTPOff control cell. Setup must therefore materialize sidecar ticket
         * consumers whenever that envelope is retained, while capture may
         * include them only when this cell actually enables MTP. Keeping two
         * exact inventories makes a disabled sidecar launch a hard failure
         * without misclassifying its setup-owned immutable stage as malformed.
         */
        std::vector<int> materialized_layer_ids;
        std::vector<int> captured_layer_ids;
        const int main_layer_count = parityLayerCount();
        ASSERT_GT(main_layer_count, 0);
        materialized_layer_ids.reserve(
            static_cast<size_t>(main_layer_count) + 1u);
        captured_layer_ids.reserve(
            static_cast<size_t>(main_layer_count) + 1u);
        for (int layer = 0; layer < main_layer_count; ++layer)
        {
            materialized_layer_ids.push_back(layer);
            captured_layer_ids.push_back(layer);
        }

        const int retained_mtp_capacity = activeMTPRetainedDraftCapacity();
        const int active_mtp_depth = activeMTPDraftDepth();
        ASSERT_GE(retained_mtp_capacity, 0);
        ASSERT_GE(active_mtp_depth, 0);
        ASSERT_LE(active_mtp_depth, retained_mtp_capacity)
            << "A live MTP policy cannot exceed its retained graph-family envelope";
        if (retained_mtp_capacity > 0)
        {
            const ModelContext *const active_model =
                activeModelContextForDiagnostics();
            ASSERT_NE(active_model, nullptr);
            const int raw_layer_count = std::max(
                active_model->totalBlockCount(),
                active_model->blockCount());
            const MTPWeightManifest manifest = discoverMTPWeightManifest(
                active_model->concreteLoader(),
                active_model->architecture(),
                raw_layer_count,
                /*explicit_mtp=*/true);
            ASSERT_TRUE(manifest.available) << manifest.diagnostic;
            for (const auto &depth : manifest.depths)
            {
                if (depth.moe_ffn_layout)
                {
                    materialized_layer_ids.push_back(
                        depth.source_layer_index);
                    if (active_mtp_depth > 0)
                    {
                        captured_layer_ids.push_back(
                            depth.source_layer_index);
                    }
                }
            }
        }
        const auto normalize_layer_ids = [](std::vector<int> &layer_ids)
        {
            std::sort(layer_ids.begin(), layer_ids.end());
            layer_ids.erase(
                std::unique(layer_ids.begin(), layer_ids.end()),
                layer_ids.end());
        };
        normalize_layer_ids(materialized_layer_ids);
        normalize_layer_ids(captured_layer_ids);
        const size_t materialized_layer_count =
            materialized_layer_ids.size();
        const size_t captured_layer_count = captured_layer_ids.size();
        ASSERT_GT(materialized_layer_count, 0u);
        ASSERT_GT(captured_layer_count, 0u);
        const size_t materialized_layers_start = kFixedEvidenceCount;
        const size_t captured_layers_start =
            materialized_layers_start + materialized_layer_count;
        std::vector<uint64_t> local(
            captured_layers_start + captured_layer_count,
            0u);

        const int participant_domain_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_overlay_participant_graph")
                ? 1
                : 0;
        const int lifecycle_domain_enabled =
            PerfStatsCollector::isDomainEnabled("forward_graph") ? 1 : 0;
        int all_domains_enabled = 0;
        const int local_domains_enabled =
            participant_domain_enabled && lifecycle_domain_enabled;
        MPI_Allreduce(
            &local_domains_enabled,
            &all_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            parityCoordinationCommunicator());
        if (all_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Rank-local canonical-ticket parity requires both "
                       "forward_graph and moe_overlay_participant_graph PerfStats";
            }
            return;
        }

        const auto parse_nonnegative = [](
                                           const PerfStatRecord &record,
                                           const char *name) -> int
        {
            const auto found = record.tags.find(name);
            if (found == record.tags.end())
                return -1;
            int value = -1;
            const char *const begin = found->second.data();
            const char *const end = begin + found->second.size();
            const auto parsed = std::from_chars(begin, end, value);
            return parsed.ec == std::errc{} && parsed.ptr == end &&
                           value >= 0
                       ? value
                       : -1;
        };

        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_participant_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_participant_graph")
            {
                continue;
            }
            if (record.name == "materialized_mapped_follower_families" ||
                record.name == "ticket_selected_graphs")
            {
                local[kMappedFollowerRecords] += record.count;
                continue;
            }

            const bool materialization =
                record.name ==
                "materialized_rank_local_canonical_ticket_consumers";
            const bool capture_launch =
                record.name ==
                "rank_local_canonical_ticket_consumer_launches";
            if (!materialization && !capture_launch)
                continue;

            const auto &expected_layer_ids =
                materialization ? materialized_layer_ids : captured_layer_ids;
            const size_t layer_count = expected_layer_ids.size();
            const int layer = parse_nonnegative(record, "layer");
            const auto layer_position = std::lower_bound(
                expected_layer_ids.begin(),
                expected_layer_ids.end(),
                layer);
            const bool expected_layer =
                layer_position != expected_layer_ids.end() &&
                *layer_position == layer;
            const size_t layer_slot = expected_layer
                                          ? static_cast<size_t>(
                                                std::distance(
                                                    expected_layer_ids.begin(),
                                                    layer_position))
                                          : layer_count;
            const auto ticket_kind = record.tags.find("ticket_kind");
            bool valid = record.count > 0u && record.value > 0.0 &&
                         expected_layer && layer_slot < layer_count &&
                         ticket_kind != record.tags.end() &&
                         ticket_kind->second == "canonical_route";
            if (materialization)
            {
                valid = valid && record.phase == "model_setup" &&
                        parse_nonnegative(record, "bucket_rows") > 0 &&
                        parse_nonnegative(record, "d_model") > 0 &&
                        parse_nonnegative(record, "route_capacity") > 0;
            }
            else
            {
                valid = valid && record.phase == "graph_capture";
            }
            if (!valid)
            {
                ++local[kMalformedRecords];
                continue;
            }

            if (materialization)
            {
                local[kMaterializations] += record.count;
                local[materialized_layers_start +
                      layer_slot] += record.count;
            }
            else
            {
                local[kCaptureLaunches] += record.count;
                local[captured_layers_start +
                      layer_slot] += record.count;
            }
        }

        for (const auto &record :
             PerfStatsCollector::snapshot({"forward_graph"}))
        {
            const auto disposition =
                classifyHeterogeneousTicketLifecycleEvidence(record);
            if (disposition == HeterogeneousTicketEvidenceDisposition::Unrelated)
                continue;
            if (disposition != HeterogeneousTicketEvidenceDisposition::Authority)
            {
                ++local[kMalformedRecords];
                // Preserve the actual rejected contract instead of reporting
                // only a global malformed count after MPI aggregation.
                std::ostringstream diagnostic;
                diagnostic << "Invalid rank-local ticket lifecycle evidence: phase="
                           << record.phase << " count=" << record.count
                           << " value=" << record.value;
                for (const auto &[key, value] : record.tags)
                    diagnostic << ' ' << key << '=' << value;
                LOG_ERROR(diagnostic.str());
            }
            else
                local[kLifecycleTransactions] += record.count;
        }

        std::vector<uint64_t> global(local.size(), 0u);
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Rank-local canonical-ticket graph evidence was malformed";
        EXPECT_EQ(global[kMappedFollowerRecords], 0u)
            << "A one-rank topology materialized an auxiliary mapped follower";
        EXPECT_GT(global[kLifecycleTransactions], 0u)
            << "No typed heterogeneous ticket transaction reached capture";
        EXPECT_GT(global[kMaterializations], 0u);
        EXPECT_GT(global[kCaptureLaunches], 0u);
        for (size_t layer = 0u; layer < materialized_layer_count; ++layer)
        {
            EXPECT_GT(global[materialized_layers_start + layer], 0u)
                << "Retained routed graph layer "
                << materialized_layer_ids[layer]
                << " has no setup-owned canonical ticket consumer";
        }
        for (size_t layer = 0u; layer < captured_layer_count; ++layer)
        {
            EXPECT_GT(global[captured_layers_start + layer], 0u)
                << "Active routed graph layer " << captured_layer_ids[layer]
                << " never recorded its canonical ticket consumer in a native graph";
        }
    }

    /**
     * @brief Prove tickets selected the topology's production specialization.
     *
     * Device-owned epochs replaced the old host-scheduled fixed-capacity
     * participant runner. Setup materializes a bounded row-shape family once;
     * each authenticated ticket selects one member without mutating token or
     * position state on the follower. Native parent capture/replay is asserted
     * independently by the shared production-path evidence gate.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertMappedParticipantGraphEvidence() const -> void
    {
        if (!topologySpansMultipleMPIRanks())
        {
            assertRankLocalCanonicalTicketGraphEvidence();
            return;
        }

        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kFamilyMaterializations = 1;
        constexpr size_t kMainFamilyMaterializations = 2;
        constexpr size_t kMTPFamilyCapacity = 3;
        constexpr size_t kSelectedGraphs = 4;
        constexpr size_t kLargestPrefillSelections = 5;
        constexpr size_t kOneRowPrefillSelections = 6;
        constexpr size_t kOneRowDecodeSelections = 7;
        constexpr size_t kSelectedMTPPhysicalRows = 8;
        constexpr size_t kRetiredHostRunnerRecords = 9;
        constexpr size_t kMaterializedGpuTransactions = 10;
        constexpr size_t kMaterializedCpuEndpoints = 11;
        constexpr size_t kFollowerGpuRuntimeTables = 12;
        constexpr size_t kHostAuthorityGpuRuntimes = 13;
        constexpr size_t kDeviceAuthorityGpuRuntimes = 14;
        constexpr size_t kMappedControllerGpuRuntimes = 15;
        constexpr size_t kEvidenceCount = 16;

        int expected_largest_prefill_rows = 0;
        if (isRootParityRank())
        {
            expected_largest_prefill_rows =
                isSegmentedPrefillProductionTest()
                    ? activeSegmentedPrefillCaptureRows()
                    : static_cast<int>(config_.token_ids.size());
        }
        MPI_Bcast(
            &expected_largest_prefill_rows,
            1,
            MPI_INT,
            parityArtifactAuthorityRank(),
            parityCoordinationCommunicator());
        ASSERT_GT(expected_largest_prefill_rows, 0);

        const int local_domain_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_overlay_participant_graph")
                ? 1
                : 0;
        int all_domains_enabled = 0;
        MPI_Allreduce(
            &local_domain_enabled,
            &all_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            parityCoordinationCommunicator());
        if (all_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the "
                       "moe_overlay_participant_graph PerfStats domain";
            }
            return;
        }

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_participant_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_participant_graph")
            {
                continue;
            }
            if (record.name == "materialized_graphs" ||
                record.name == "fixed_capacity_graph_reuses")
            {
                local[kRetiredHostRunnerRecords] += record.count;
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto parse_positive = [&](const char *name) -> int
            {
                const auto value = tag(name);
                if (!value)
                    return 0;
                int parsed = 0;
                const char *const begin = value->data();
                const char *const end = begin + value->size();
                const auto result = std::from_chars(begin, end, parsed);
                return result.ec == std::errc{} && result.ptr == end &&
                               parsed > 0
                           ? parsed
                           : 0;
            };
            const auto parse_nonnegative = [&](const char *name) -> int
            {
                const auto value = tag(name);
                if (!value)
                    return -1;
                int parsed = -1;
                const char *const begin = value->data();
                const char *const end = begin + value->size();
                const auto result = std::from_chars(begin, end, parsed);
                return result.ec == std::errc{} && result.ptr == end &&
                               parsed >= 0
                           ? parsed
                           : -1;
            };

            if (record.name == "materialized_mapped_follower_families")
            {
                const auto graph_family = tag("graph_family");
                const auto standalone_progress =
                    tag("standalone_progress_launch");
                const int graph_family_ordinal =
                    parse_nonnegative("graph_family");
                const int row_capacity = parse_positive("row_capacity");
                const int gpu_transactions = parse_nonnegative(
                    "setup_materialized_gpu_transactions");
                const int cpu_endpoints = parse_nonnegative(
                    "setup_materialized_cpu_endpoints");
                const bool valid =
                    record.phase == "model_setup" && record.value == 1.0 &&
                    record.count == 1u && graph_family &&
                    graph_family_ordinal >= 0 &&
                    row_capacity > 0 && gpu_transactions >= 0 &&
                    cpu_endpoints >= 0 &&
                    (gpu_transactions > 0 || cpu_endpoints > 0) &&
                    parse_nonnegative("captured_transfer_branches") >= 0 &&
                    standalone_progress &&
                    *standalone_progress == "false";
                if (!valid)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                ++local[kFamilyMaterializations];
                local[kMaterializedGpuTransactions] +=
                    static_cast<uint64_t>(gpu_transactions);
                local[kMaterializedCpuEndpoints] +=
                    static_cast<uint64_t>(cpu_endpoints);
                if (graph_family_ordinal == 0)
                    ++local[kMainFamilyMaterializations];
                if (graph_family_ordinal == 1 &&
                    row_capacity == activeMTPGraphCapacityVerifierRows())
                {
                    ++local[kMTPFamilyCapacity];
                }
                continue;
            }
            if (record.name != "ticket_selected_graphs")
                continue;

            const int logical_rows = parse_positive("logical_rows");
            const int physical_rows = parse_positive("physical_rows");
            const auto transport_path = tag("transport_path");
            const auto position_mutated = tag("position_mutated");
            const auto gpu_host_dispatches = tag("gpu_host_layer_dispatches");
            const bool phase_valid =
                record.phase == "main_prefill" ||
                record.phase == "main_decode" ||
                record.phase == "mtp_grouped_verifier" ||
                record.phase == "mtp_draft";
            const bool valid =
                record.value > 0.0 && record.count > 0u && phase_valid &&
                logical_rows > 0 && physical_rows >= logical_rows &&
                transport_path &&
                transport_path->rfind("node_local_mapped_", 0) == 0 &&
                position_mutated && *position_mutated == "false" &&
                gpu_host_dispatches && *gpu_host_dispatches == "0";
            if (!valid)
            {
                ++local[kMalformedRecords];
                continue;
            }

            local[kSelectedGraphs] += record.count;
            if (record.phase == "main_prefill" &&
                logical_rows == expected_largest_prefill_rows)
            {
                local[kLargestPrefillSelections] += record.count;
            }
            if (record.phase == "main_prefill" && logical_rows == 1)
                local[kOneRowPrefillSelections] += record.count;
            if (record.phase == "main_decode" && logical_rows == 1)
                local[kOneRowDecodeSelections] += record.count;
            if (record.phase == "mtp_grouped_verifier" &&
                physical_rows == activeMTPPhysicalVerifierRows())
            {
                local[kSelectedMTPPhysicalRows] += record.count;
            }
        }

        /*
         * Policy location and follower execution state are orthogonal. Every
         * remote GPU endpoint needs a device runtime table even when a CPU
         * participant makes the sole policy authority host-resident. This is
         * the focused production-path regression for the former manual
         * dispatch-consume segment in an otherwise device-owned envelope.
         */
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"moe_overlay_controller"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay_controller" ||
                record.name != "follower_runtime_tables_materialized")
            {
                continue;
            }
            const auto tag = [&record](const char *name)
                -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto authority = tag("authority_execution");
            const auto mapped = tag("mapped_device_controller");
            const auto blocking = tag("blocking_hot_path");
            const auto layers = tag("layers");
            int parsed_layers = 0;
            if (layers)
            {
                const char *const begin = layers->data();
                const char *const end = begin + layers->size();
                const auto parsed = std::from_chars(
                    begin, end, parsed_layers);
                if (parsed.ec != std::errc{} || parsed.ptr != end)
                    parsed_layers = 0;
            }
            const bool valid =
                record.phase == "model_setup" && record.value == 1.0 &&
                record.count == 1u && authority && mapped && blocking &&
                *blocking == "false" && parsed_layers > 0 &&
                (*authority == "host-resident" ||
                 *authority == "device-resident") &&
                (*mapped == "true" || *mapped == "false") &&
                ((*authority == "host-resident" && *mapped == "false") ||
                 (*authority == "device-resident" && *mapped == "true"));
            if (!valid)
            {
                ++local[kMalformedRecords];
                continue;
            }
            ++local[kFollowerGpuRuntimeTables];
            if (*authority == "host-resident")
                ++local[kHostAuthorityGpuRuntimes];
            else
                ++local[kDeviceAuthorityGpuRuntimes];
            if (*mapped == "true")
                ++local[kMappedControllerGpuRuntimes];
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Mapped participant graph evidence was malformed";
        /*
         * Qwen3.5 owns one set of MTP sidecar weights and recursively replays
         * that same retained graph for every speculative slot. Draft depth is
         * therefore transaction geometry, not graph-family identity. A cell
         * with retained MTP setup capacity materializes one dormant recursive
         * sidecar even when request-time MTP execution is off; it must not
         * execute that family. A model with no retained capacity owns only the
         * main family.
         */
        const uint64_t expected_family_materializations =
            activeMTPRetainedDraftCapacity() > 0 ? 2u : 1u;
        EXPECT_EQ(
            global[kFamilyMaterializations],
            expected_family_materializations)
            << "The auxiliary MPI runner materialized a graph family outside the production request";
        EXPECT_EQ(global[kMainFamilyMaterializations], 1u)
            << "The one auxiliary MPI runner must materialize its main mapped graph family exactly once";
        const bool has_remote_gpu_endpoint =
            topologyHasSecondaryGpuDomain();
        if (has_remote_gpu_endpoint)
        {
            EXPECT_GT(global[kMaterializedGpuTransactions], 0u)
                << "The mapped follower family materialized no native GPU transaction";
            EXPECT_GT(global[kFollowerGpuRuntimeTables], 0u)
                << "A remote GPU follower executed without a device-resident placement table";
            if (topologyUsesCpu())
            {
                EXPECT_EQ(
                    global[kHostAuthorityGpuRuntimes],
                    global[kFollowerGpuRuntimeTables])
                    << "CPU-participating topology must retain one host policy authority while every GPU keeps local execution state";
                EXPECT_EQ(global[kMappedControllerGpuRuntimes], 0u)
                    << "Host-authority GPU followers must not acquire a competing mapped policy controller";
            }
            else
            {
                EXPECT_EQ(
                    global[kDeviceAuthorityGpuRuntimes],
                    global[kFollowerGpuRuntimeTables]);
                EXPECT_EQ(
                    global[kMappedControllerGpuRuntimes],
                    global[kFollowerGpuRuntimeTables])
                    << "All-GPU followers must bind the sole mapped device policy controller";
            }
        }
        if (topologyUsesCpu())
        {
            EXPECT_GT(global[kMaterializedCpuEndpoints], 0u)
                << "The mapped follower family retained no typed CPU boundary endpoint";
        }
        if (activeMTPRetainedDraftCapacity() > 0)
        {
            EXPECT_EQ(global[kMTPFamilyCapacity], 1u)
                << "The recursive MTP follower family did not retain the shared "
                << activeMTPGraphCapacityVerifierRows()
                << "-row campaign capacity";
        }
        else
        {
            EXPECT_EQ(global[kMTPFamilyCapacity], 0u)
                << "A campaign without retained MTP capacity materialized a recursive sidecar family";
        }
        EXPECT_GT(global[kSelectedGraphs], 0u)
            << "No authenticated ticket selected a mapped participant graph";
        EXPECT_GT(global[kLargestPrefillSelections], 0u)
            << "The mapped family never served the largest root-published live prefill chunk";
        EXPECT_GT(global[kOneRowDecodeSelections], 0u)
            << "Decode never selected its setup-owned one-row retained parent";
        if (activeMTPEnabled())
        {
            EXPECT_GT(global[kSelectedMTPPhysicalRows], 0u)
                << "No mapped grouped-verifier graph selected the admitted "
                << activeMTPPhysicalVerifierRows()
                << "-row physical transaction bucket";
        }
        else
        {
            EXPECT_EQ(global[kSelectedMTPPhysicalRows], 0u)
                << "The execution-off control selected its dormant retained MTP family";
        }
        EXPECT_EQ(global[kRetiredHostRunnerRecords], 0u)
            << "The retired host-scheduled participant graph path executed";
        if (isSegmentedPrefillProductionTest())
        {
            EXPECT_GT(global[kOneRowPrefillSelections], 0u)
                << "The short prefill tail never selected its bounded mapped graph";
        }
    }

    /**
     * @brief Prove the real sparse transport moved compact packets between tiers.
     *
     * Local-route completion proves that every participant ran an expert, but
     * it does not independently prove that the production sparse collective
     * carried compact request and result packets. The graph-native transport
     * publishes those byte counts through PerfStats. Folding them across both
     * MPI instances makes a missing dispatch, missing return, or silently
     * bypassed CPU cold tier a fatal parity failure without paying for a second
     * model setup in a profiler-only smoke test.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertSparseTransportPerfStatsEvidence() const -> void
    {
        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kCompactDispatchBytes = 1;
        constexpr size_t kCompactReturnBytes = 2;
        constexpr size_t kCpuRows = 3;
        constexpr size_t kGpuRows = 4;
        constexpr size_t kRankLocalDispatchBytes = 5;
        constexpr size_t kRankLocalReturnBytes = 6;
        constexpr size_t kRankLocalCpuRows = 7;
        constexpr size_t kEvidenceCount = 8;

        const int local_domain_enabled =
            PerfStatsCollector::isDomainEnabled("moe_overlay") ? 1 : 0;
        int all_domains_enabled = 0;
        MPI_Allreduce(
            &local_domain_enabled,
            &all_domains_enabled,
            1,
            MPI_INT,
            MPI_MIN,
            parityCoordinationCommunicator());
        if (all_domains_enabled == 0)
        {
            if (isRootParityRank())
            {
                ADD_FAILURE()
                    << "Production graph-native parity must retain the "
                       "moe_overlay PerfStats domain";
            }
            return;
        }

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot({"moe_overlay"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "moe_overlay")
            {
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto found = record.tags.find(name);
                return found == record.tags.end() ? nullptr : &found->second;
            };
            const auto transport = tag("transport");
            const auto domain_kind = tag("domain_kind");
            const auto participant = tag("participant");
            const bool positive = record.count > 0u && record.value > 0.0;

            const bool rank_local_dispatch =
                record.name ==
                "rank_local_canonical_ticket_dispatch_bytes";
            const bool rank_local_return =
                record.name ==
                "rank_local_canonical_ticket_return_bytes";
            if (rank_local_dispatch || rank_local_return)
            {
                const auto completion = tag("completion");
                const auto identity_source = tag("identity_source");
                if (!positive || !transport || *transport != "compact" ||
                    !participant || !completion ||
                    *completion !=
                        "rank_local_canonical_ticket_complete" ||
                    !identity_source ||
                    *identity_source != "sparse_collective_key" ||
                    record.phase !=
                        (rank_local_dispatch
                             ? "gn_sparse_dispatch"
                             : "gn_return_reduce"))
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                const uint64_t bytes =
                    static_cast<uint64_t>(record.value);
                local[rank_local_dispatch
                          ? kRankLocalDispatchBytes
                          : kRankLocalReturnBytes] += bytes;
                local[rank_local_dispatch
                          ? kCompactDispatchBytes
                          : kCompactReturnBytes] += bytes;
                continue;
            }
            if (record.name ==
                "rank_local_canonical_ticket_cpu_rows")
            {
                const auto completion = tag("completion");
                const auto identity_source = tag("identity_source");
                if (record.phase != "gn_local_expert" || !positive ||
                    !transport || *transport != "local" || !domain_kind ||
                    *domain_kind != "CPU" || !participant || !completion ||
                    *completion !=
                        "rank_local_canonical_ticket_complete" ||
                    !identity_source ||
                    *identity_source != "sparse_collective_key")
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                const uint64_t rows =
                    static_cast<uint64_t>(record.value);
                local[kRankLocalCpuRows] += rows;
                local[kCpuRows] += rows;
                continue;
            }

            if (record.name == "compact_dispatch_bytes")
            {
                if (record.phase != "gn_sparse_dispatch" || !positive ||
                    !transport || *transport != "compact")
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                local[kCompactDispatchBytes] +=
                    static_cast<uint64_t>(record.value);
                continue;
            }
            if (record.name == "compact_return_bytes")
            {
                if (record.phase != "gn_return_reduce" || !positive ||
                    !transport || *transport != "compact")
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                local[kCompactReturnBytes] +=
                    static_cast<uint64_t>(record.value);
                continue;
            }
            if (record.name != "device_epoch_cpu_rows" &&
                record.name != "device_epoch_gpu_rows")
                continue;

            int participant_id = -1;
            try
            {
                participant_id = participant ? std::stoi(*participant) : -1;
            }
            catch (const std::exception &)
            {
                participant_id = -1;
            }
            const bool cpu_record =
                record.name == "device_epoch_cpu_rows";
            const char *expected_kind = cpu_record ? "CPU" : "GPU";
            if (record.phase != "gn_local_expert" || !positive ||
                !transport || *transport != "local" ||
                !domain_kind || *domain_kind != expected_kind ||
                participant_id < 0)
            {
                ++local[kMalformedRecords];
                continue;
            }
            local[cpu_record ? kCpuRows : kGpuRows] +=
                static_cast<uint64_t>(record.value);
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());
        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Graph-native sparse transport PerfStats evidence was malformed";
        if (parity_residency_by_participant_.size() !=
            activeOverlayParticipantCount())
        {
            ADD_FAILURE()
                << "Published ExpertOverlay participation was not retained for sparse transport evidence";
            return;
        }
        PublishedSparseTierParticipation active_tiers;
        try
        {
            active_tiers = summarizePublishedParticipation(
                resolvedOverlayPlan(),
                parity_residency_by_participant_);
        }
        catch (const std::invalid_argument &error)
        {
            ADD_FAILURE()
                << "Published ExpertOverlay participation is invalid: "
                << error.what();
            return;
        }
        if (active_tiers.sparse_follower)
        {
            EXPECT_GT(global[kCompactDispatchBytes], 0u)
                << "A published sparse follower received no compact dispatch bytes";
            EXPECT_GT(global[kCompactReturnBytes], 0u)
                << "A published sparse follower emitted no compact return bytes";
        }
        else
        {
            /*
             * Automatic capacity may fit the entire model in the continuation
             * domain. Configured lower-priority endpoints then remain valid,
             * explicit idle participants: manufacturing transport solely to
             * satisfy a counter would no longer exercise production routing.
             */
            EXPECT_EQ(global[kCompactDispatchBytes], 0u)
                << "An idle sparse topology emitted compact dispatch traffic";
            EXPECT_EQ(global[kCompactReturnBytes], 0u)
                << "An idle sparse topology emitted compact return traffic";
        }
        if (active_tiers.secondary_gpu)
        {
            EXPECT_GT(global[kGpuRows], 0u)
                << "No remote GPU tier published completed device-owned expert rows";
        }
        else
        {
            EXPECT_EQ(global[kGpuRows], 0u)
                << "A topology without a remote GPU endpoint published remote GPU rows";
        }
        if (active_tiers.cpu)
        {
            EXPECT_GT(global[kCpuRows], 0u)
                << "A CPU tier owning published experts completed no expert rows";
            if (!topologySpansMultipleMPIRanks())
            {
                EXPECT_GT(global[kRankLocalDispatchBytes], 0u)
                    << "The rank-local CPU tier consumed no compact dispatch payload";
                EXPECT_GT(global[kRankLocalReturnBytes], 0u)
                    << "The rank-local CPU tier published no canonical return payload";
                EXPECT_GT(global[kRankLocalCpuRows], 0u)
                    << "The rank-local canonical ticket carried no completed CPU routes";
            }
        }
        else
        {
            EXPECT_EQ(global[kCpuRows], 0u)
                << "A capacity-resolved idle or absent CPU tier unexpectedly executed expert rows";
        }
    }

    /**
     * @brief Prove the full request lifecycle honored the shared segmented graph.
     *
     * PrefixRuntimeStateSnapshot proves the public OrchestrationRunner admitted
     * a chunked request instead of treating the test as three unrelated
     * forwards. Per-rank PerfStats then prove the distributed schedule contract
     * was published, an expert-only participant consumed the same ordered
     * transactions, and the CUDA continuation retained complete prompt-wide
     * checkpoints for the existing Hugging Face/CSV comparator. Dynamic cells
     * deliberately execute calibration and migration-training requests before
     * parity, so PerfStats and the prefix probe are cumulative by design. The
     * assertion consequently proves the whole lifecycle rather than inventing
     * a test-only reset edge: all schedules succeed, every captured row is
     * accounted for, both roles retain identical ordered digests, and the one
     * request-scoped parity evidence independently proves `[4,4,1]`. Training
     * and prefix seeding may also publish valid snapshot aggregations. Mandatory
     * prefix restore then contributes exactly one serial suffix-decode record.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertSegmentedPrefillEvidence() const -> void
    {
        constexpr uint64_t kExpectedChunks = 3;
        constexpr uint64_t kExpectedRealTokens = 9;
        constexpr uint64_t kExpectedPaddedTokens = 3;
        constexpr size_t kContractRecords = 0;
        constexpr size_t kSnapshotAggregationRecords = 1;
        constexpr size_t kContinuationTransactionRecords = 2;
        constexpr size_t kParticipantTransactionRecords = 3;
        constexpr size_t kContinuationFourRowTransactions = 4;
        constexpr size_t kContinuationOneRowTransactions = 5;
        constexpr size_t kParticipantFourRowTransactions = 6;
        constexpr size_t kParticipantOneRowTransactions = 7;
        constexpr size_t kContinuationSequenceRecords = 8;
        constexpr size_t kParticipantSequenceRecords = 9;
        constexpr size_t kContinuationSequenceSteps = 10;
        constexpr size_t kParticipantSequenceSteps = 11;
        constexpr size_t kContinuationSequenceWords = 12;
        constexpr size_t kParticipantSequenceWords = 13;
        constexpr size_t kContinuationSequenceDigestLo = 14;
        constexpr size_t kParticipantSequenceDigestLo = 15;
        constexpr size_t kContinuationSequenceDigestHi = 16;
        constexpr size_t kParticipantSequenceDigestHi = 17;
        constexpr size_t kMalformedRecords = 18;
        constexpr size_t kRestoredPrefixSuffixDecodeTransactions = 19;
        constexpr size_t kEvidenceCount = 20;
        constexpr uint64_t kWordsPerSequenceStep = 4u;

        std::array<uint64_t, kEvidenceCount> local{};
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"forward_graph", "moe_overlay_participant_graph"}))
        {
            const auto tagEquals = [&record](
                                       const char *name,
                                       const std::string &expected)
            {
                const auto it = record.tags.find(name);
                return it != record.tags.end() && it->second == expected;
            };
            const auto unsignedTag = [&record](const char *name)
                -> std::optional<uint64_t>
            {
                const auto it = record.tags.find(name);
                if (it == record.tags.end() || it->second.empty())
                    return std::nullopt;
                uint64_t value = 0;
                const char *const begin = it->second.data();
                const char *const end = begin + it->second.size();
                const auto parsed = std::from_chars(begin, end, value);
                if (parsed.ec != std::errc{} || parsed.ptr != end)
                    return std::nullopt;
                return value;
            };

            if (record.kind == PerfStatRecord::Kind::OrderedSequence &&
                record.domain == "forward_graph" &&
                record.name ==
                    "moe_overlay_collective_transaction_sequence")
            {
                if (record.phase != "prefill")
                    continue;

                const bool common_contract =
                    record.count > 0u &&
                    record.value == static_cast<double>(record.count) &&
                    record.sequence_word_count ==
                        record.count * kWordsPerSequenceStep &&
                    record.sequence_digest_lo != 0u &&
                    record.sequence_digest_hi != 0u &&
                    tagEquals(
                        "identity_source",
                        "orchestration_request_and_chunk") &&
                    tagEquals(
                        "logical_step_semantics",
                        "monotonic_transaction");
                const bool continuation =
                    tagEquals("role", "continuation_graph");
                const bool participant =
                    tagEquals("role", "expert_participant_graph");
                if (!common_contract || continuation == participant)
                {
                    ++local[kMalformedRecords];
                    continue;
                }

                const size_t record_slot =
                    continuation ? kContinuationSequenceRecords
                                 : kParticipantSequenceRecords;
                const size_t step_slot =
                    continuation ? kContinuationSequenceSteps
                                 : kParticipantSequenceSteps;
                const size_t word_slot =
                    continuation ? kContinuationSequenceWords
                                 : kParticipantSequenceWords;
                const size_t digest_lo_slot =
                    continuation ? kContinuationSequenceDigestLo
                                 : kParticipantSequenceDigestLo;
                const size_t digest_hi_slot =
                    continuation ? kContinuationSequenceDigestHi
                                 : kParticipantSequenceDigestHi;
                if (local[record_slot] != 0u)
                {
                    ++local[kMalformedRecords];
                    continue;
                }
                local[record_slot] = 1u;
                local[step_slot] = record.count;
                local[word_slot] = record.sequence_word_count;
                local[digest_lo_slot] = record.sequence_digest_lo;
                local[digest_hi_slot] = record.sequence_digest_hi;
                continue;
            }

            if (record.kind != PerfStatRecord::Kind::Counter)
                continue;

            if (record.domain == "forward_graph" &&
                record.name ==
                    "restored_prefix_suffix_decode_transactions")
            {
                const bool valid =
                    record.phase == "decode" &&
                    record.value == static_cast<double>(record.count) &&
                    record.count == 1u &&
                    tagEquals("outer_command", "prefill") &&
                    tagEquals("mathematical_phase", "decode") &&
                    tagEquals("logical_rows", "1") &&
                    tagEquals("state_transition", "main_only");
                if (!valid)
                {
                    std::ostringstream detail;
                    detail << "Malformed restored-prefix suffix evidence: "
                           << "phase=" << record.phase
                           << ",value=" << record.value
                           << ",count=" << record.count;
                    for (const auto &[name, value] : record.tags)
                        detail << ',' << name << '=' << value;
                    ADD_FAILURE() << detail.str();
                    ++local[kMalformedRecords];
                }
                else
                    local[kRestoredPrefixSuffixDecodeTransactions] +=
                        record.count;
                continue;
            }

            if (record.domain == "forward_graph" &&
                record.name == "moe_overlay_prefill_schedule_contract_rows")
            {
                if (record.phase != "model_setup" ||
                    record.value != static_cast<double>(
                                        activeSegmentedPrefillCaptureRows()) ||
                    record.count != 1u ||
                    !tagEquals("authority", "continuation_root") ||
                    !tagEquals("bucket_count", "1") ||
                    !tagEquals("immutable", "true"))
                {
                    ++local[kMalformedRecords];
                }
                else
                {
                    ++local[kContractRecords];
                }
                continue;
            }

            if (record.domain == "forward_graph" &&
                record.name == "prefill_chunk_snapshot_sequence_keys")
            {
                try
                {
                    const auto evidence = ParityPrefillSnapshotEvidence::capture(
                        std::span<const PerfStatRecord>(&record, 1u), kExpectedChunks);
                    local[kSnapshotAggregationRecords] += evidence.transactions();
                }
                catch (const std::logic_error &error)
                {
                    std::ostringstream detail;
                    detail << error.what() << ": "
                           << "phase=" << record.phase
                           << ",value=" << record.value
                           << ",count=" << record.count;
                    for (const auto &[name, value] : record.tags)
                        detail << ',' << name << '=' << value;
                    ADD_FAILURE() << detail.str();
                    ++local[kMalformedRecords];
                }
                continue;
            }

            if (record.domain == "forward_graph" &&
                record.name == "moe_overlay_collective_transaction")
            {
                /*
                 * Decode parity performs its own prefill initialization, then
                 * emits serial decode transactions. Those records are covered
                 * by the decode comparator; they are not malformed prefill
                 * evidence merely because they share the sparse metric name.
                 */
                if (record.phase != "prefill")
                    continue;

                const auto logical_rows = unsignedTag("logical_rows");
                const auto physical_rows = unsignedTag("physical_rows");
                const bool bounded_identity =
                    record.tags.count("generation") == 0u &&
                    record.tags.count("logical_step") == 0u &&
                    record.tags.count("prefill_chunk_index") == 0u &&
                    record.tags.count("token_offset") == 0u;
                const bool chunk_geometry_ok = logical_rows && physical_rows &&
                    *physical_rows == activeSegmentedPrefillCaptureRows() &&
                    *logical_rows > 0u &&
                    *logical_rows <= *physical_rows;
                const bool transaction_contract_ok =
                    record.count > 0u &&
                    record.value == static_cast<double>(record.count) &&
                    tagEquals("role", "continuation_graph") &&
                    tagEquals("identity_source", "orchestration_request_and_chunk") &&
                    tagEquals("logical_step_semantics", "monotonic_transaction") &&
                    bounded_identity &&
                    chunk_geometry_ok;
                if (!transaction_contract_ok)
                {
                    ++local[kMalformedRecords];
                    continue;
                }

                local[kContinuationTransactionRecords] += record.count;
                if (*logical_rows == 4u)
                    local[kContinuationFourRowTransactions] += record.count;
                else if (*logical_rows == 1u)
                    local[kContinuationOneRowTransactions] += record.count;
                continue;
            }

            if (record.domain == "moe_overlay_participant_graph" &&
                record.name == "ticket_selected_graphs" &&
                record.phase == "main_prefill")
            {
                const auto logical_rows = unsignedTag("logical_rows");
                const auto physical_rows = unsignedTag("physical_rows");
                const bool bounded_identity =
                    record.tags.count("request_generation") == 0u &&
                    record.tags.count("command_id") == 0u &&
                    record.tags.count("transaction_ordinal") == 0u &&
                    record.tags.count("logical_step") == 0u &&
                    record.tags.count("prefill_schedule_fingerprint") == 0u;
                const bool transaction_geometry_ok = logical_rows &&
                    *logical_rows > 0u && physical_rows &&
                    *logical_rows <= *physical_rows;
                const bool valid =
                    record.count > 0u &&
                    record.value == static_cast<double>(record.count) &&
                    tagEquals("logical_step_semantics", "monotonic_transaction") &&
                    bounded_identity &&
                    transaction_geometry_ok &&
                    *physical_rows == activeSegmentedPrefillCaptureRows();
                if (!valid)
                {
                    std::ostringstream detail;
                    detail << "Malformed segmented-prefill follower ticket: "
                           << "phase=" << record.phase
                           << ",value=" << record.value
                           << ",count=" << record.count;
                    for (const auto &[name, value] : record.tags)
                        detail << ',' << name << '=' << value;
                    ADD_FAILURE() << detail.str();
                    ++local[kMalformedRecords];
                    continue;
                }

                local[kParticipantTransactionRecords] += record.count;
                if (*logical_rows == 4u)
                    local[kParticipantFourRowTransactions] += record.count;
                else if (*logical_rows == 1u)
                    local[kParticipantOneRowTransactions] += record.count;
            }
        }

        std::array<uint64_t, kEvidenceCount> global{};
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());

        if (!isRootParityRank())
            return;

        const PrefixRuntimeStateSnapshot probe = activePrefixStateProbe();
        EXPECT_GT(probe.prefill_chunk_schedules, 0u)
            << "The production lifecycle never entered the segmented scheduler";
        EXPECT_EQ(
            probe.prefill_chunk_successful_schedules,
            probe.prefill_chunk_schedules)
            << "Every admitted production schedule must complete";
        EXPECT_GE(probe.prefill_chunks, kExpectedChunks)
            << "The authenticated parity prefill must contribute [4,4,1]";
        EXPECT_GE(probe.prefill_chunk_real_tokens, kExpectedRealTokens)
            << "The authenticated parity prompt's real rows were not counted";
        EXPECT_GE(probe.prefill_chunk_padded_tokens, kExpectedPaddedTokens)
            << "The authenticated parity prompt's final bucket was not padded";
        EXPECT_EQ(
            probe.prefill_chunk_real_tokens +
                probe.prefill_chunk_padded_tokens,
            probe.prefill_chunks * activeSegmentedPrefillCaptureRows())
            << "Every captured row must be classified as real or padding";
        EXPECT_EQ(probe.prefill_chunk_failures, 0u)
            << "A graph-native heterogeneous schedule may not recover through "
               "an eager or uncaptured fallback";

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Segmented prefill PerfStats tags or geometry were malformed";
        EXPECT_EQ(
            global[kContractRecords],
            static_cast<uint64_t>(mpiWorldSize()))
            << "Every MPI participant must install the immutable shared bucket contract";
        EXPECT_GE(global[kSnapshotAggregationRecords], 1u)
            << "The continuation graph must publish validated snapshot aggregations; "
               "the exact parity request delta is checked with its checkpoints";
        EXPECT_EQ(
            global[kParticipantTransactionRecords],
            global[kContinuationTransactionRecords])
            << "The remote expert graph must consume every lifecycle prefill ticket";
        EXPECT_GE(global[kContinuationFourRowTransactions], 2u);
        EXPECT_GE(global[kContinuationOneRowTransactions], 1u);
        EXPECT_EQ(
            global[kParticipantFourRowTransactions],
            global[kContinuationFourRowTransactions]);
        EXPECT_EQ(
            global[kParticipantOneRowTransactions],
            global[kContinuationOneRowTransactions]);
        EXPECT_EQ(global[kContinuationSequenceRecords], 1u)
            << "The continuation must retain one bounded prefill sequence row";
        EXPECT_EQ(global[kParticipantSequenceRecords], 1u)
            << "The follower must retain one bounded prefill sequence row";
        EXPECT_EQ(
            global[kParticipantSequenceSteps],
            global[kContinuationSequenceSteps]);
        EXPECT_EQ(
            global[kContinuationSequenceSteps],
            global[kContinuationTransactionRecords]);
        EXPECT_EQ(
            global[kParticipantSequenceWords],
            global[kContinuationSequenceWords]);
        EXPECT_EQ(
            global[kContinuationSequenceWords],
            global[kContinuationSequenceSteps] * kWordsPerSequenceStep);
        EXPECT_EQ(
            global[kRestoredPrefixSuffixDecodeTransactions],
            1u)
            << "The partial prefix proof must consume its uncached row through serial decode math";
        EXPECT_NE(global[kContinuationSequenceDigestLo], 0u);
        EXPECT_NE(global[kContinuationSequenceDigestHi], 0u);
        EXPECT_EQ(
            global[kContinuationSequenceDigestLo],
            global[kParticipantSequenceDigestLo])
            << "Continuation and follower observed different transaction order or geometry";
        EXPECT_EQ(
            global[kContinuationSequenceDigestHi],
            global[kParticipantSequenceDigestHi])
            << "Continuation and follower observed different transaction order or geometry";
    }

    /**
     * @brief Verify every chunk-scoped Hugging Face checkpoint became full-prompt data.
     *
     * SnapshotCapture retains context-qualified copies such as
     * `PREFILL_CHUNK_0_layer0_...` for diagnosis and rewrites the bare semantic
     * key to the ordered aggregate. This check prevents a short final chunk
     * from passing merely because a comparison used the shorter tensor length.
     * @param before Immutable counters immediately before the parity request;
     *               its delta excludes earlier training and later prefix seeds.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertSegmentedPrefillCheckpointCoverage(
        const ParityPrefillSnapshotEvidence &before) -> void
    {
        const uint64_t chunks =
            (config_.token_ids.size() + activeSegmentedPrefillCaptureRows() - 1u) /
            activeSegmentedPrefillCaptureRows();
        EXPECT_NO_THROW(ParityPrefillSnapshotEvidence::capture(
                            PerfStatsCollector::snapshot({"forward_graph"}), chunks)
                            .requireSingleRequestSince(before))
            << "The authenticated prefill must aggregate its complete prompt exactly once";
        constexpr const char *kFirstChunkPrefix = "PREFILL_CHUNK_0_";
        const auto keys = activeSnapshotKeys();
        size_t checked = 0;
        for (const std::string &scoped_key : keys)
        {
            if (scoped_key.rfind(kFirstChunkPrefix, 0) != 0)
                continue;

            const std::string semantic_key =
                scoped_key.substr(std::char_traits<char>::length(kFirstChunkPrefix));
            if (semantic_key.empty() || semantic_key == "LM_HEAD")
                continue;

            const std::vector<float> reference = loadPyTorchSnapshot(semantic_key);
            if (reference.empty())
                continue;

            size_t aggregate_elements = 0;
            const float *aggregate = activeSnapshot(
                semantic_key,
                aggregate_elements);
            ASSERT_NE(aggregate, nullptr)
                << "Segmented checkpoint '" << semantic_key
                << "' lost its bare parity key after aggregation";
            EXPECT_EQ(aggregate_elements, reference.size())
                << "Segmented checkpoint '" << semantic_key
                << "' must contain every real prompt row";
            ++checked;
        }

        EXPECT_GT(checked, 0u)
            << "Segmented production prefill published no Hugging Face-backed "
               "chunk checkpoints";
    }


    auto Qwen35MoENodeExpertOverlayParityTest::collectivelyCheckHardwareAndModel() const -> bool
    {
        bool available = false;
        if (isRootParityRank())
            available =
                !acceleratorHardwareBlocker(cluster_inventory_).has_value() &&
                modelAvailable();
        return broadcastRootFlag(available);
    }

}
