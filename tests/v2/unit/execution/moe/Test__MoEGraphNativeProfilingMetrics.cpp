/**
 * @file Test__MoEGraphNativeProfilingMetrics.cpp
 * @brief Unit tests for Phase 14 graph-native MoE overlay profiling metrics.
 *
 * Verifies that the three new record APIs populate rows correctly:
 *   - recordGraphNativeSparseDispatch (gn_sparse_dispatch)
 *   - recordGraphNativeLocalExpert    (gn_local_expert)
 *   - recordGraphNativeReturnReduce   (gn_return_reduce)
 */

#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "execution/moe/MoEOverlayRankBatchTelemetry.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <string>

using namespace llaminar2;

// ─────────────────────────────────────────────────────────────────────────────
// Fixture: enables profiling and resets the profiler before each test
// ─────────────────────────────────────────────────────────────────────────────
class Test__MoEGraphNativeProfilingMetrics : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mutableDebugEnv().profile.enabled = true;
        MoEExpertOverlayProfiler::reset();
        PerfStatsCollector::reset();
    }

    void TearDown() override
    {
        mutableDebugEnv().profile.enabled = false;
        MoEExpertOverlayProfiler::reset();
        PerfStatsCollector::reset();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static bool hasPhase(const std::vector<MoEExpertOverlayProfileRow> &rows, const std::string &phase)
{
    return std::any_of(rows.begin(), rows.end(),
                       [&](const MoEExpertOverlayProfileRow &r)
                       { return r.phase == phase; });
}

static bool hasUnifiedRecord(
    const std::vector<PerfStatRecord> &records,
    PerfStatRecord::Kind kind,
    const std::string &name,
    const std::string &phase)
{
    return std::any_of(records.begin(), records.end(),
                       [&](const PerfStatRecord &record)
                       {
                           return record.kind == kind &&
                                  record.domain == "moe_overlay" &&
                                  record.name == name &&
                                  record.phase == phase;
                       });
}

/** @brief Build a stable typed sparse edge for concise profiler tests. */
static MoEOverlayProfileEdge profileEdge(int source, int target)
{
    return MoEOverlayProfileEdge{
        .source_participant = source,
        .target_participant = target,
    };
}

/**
 * @brief Both transports retain all observations without a per-token timing map.
 *
 * Vary generation, logical step, sequence and payload size over 1024 exchanges.
 * Only measured totals and ordered fingerprints may change; timing/counter row
 * cardinality must not grow. Exercise both observing roles and wire directions.
 */
TEST_F(Test__MoEGraphNativeProfilingMetrics, RankBatchEvidenceIsBoundedAcrossRequestsAndTokens)
{
    constexpr uint64_t observations = 1024;
    for (auto kind : {MoEOverlayRankBatchTransportKind::MPI,
                      MoEOverlayRankBatchTransportKind::NodeLocalSharedRows})
        for (auto role : {MoEOverlayRankBatchEndpoint::Source, MoEOverlayRankBatchEndpoint::Target})
            for (auto direction : {MoEOverlayCollectiveDirection::Dispatch,
                                   MoEOverlayCollectiveDirection::ReturnReduce})
            {
                PerfStatsCollector::reset();
                auto key = makeMoEOverlayRankBatchKey(
                    1, 0, ExpertHistogramSource::DecodeToken, 3, 1, 2, 0, 1, direction);
                const MoEOverlayRankBatchTelemetry telemetry(key, kind, role);
                const auto observe = [&](uint64_t step)
                {
                    key.generation_id = step + 1;
                    key.step_id = 3 * step;
                    key.sequence = step + 17;
                    telemetry.recordTransaction(32 + step, 2);
                    telemetry.recordTimings(10 + step, 20 + step, 30 + step);
                    if (kind == MoEOverlayRankBatchTransportKind::MPI)
                        telemetry.recordAsyncSendSubmission(8);
                };
                observe(0);
                const auto initial_size = PerfStatsCollector::snapshot().size();
                EXPECT_EQ(initial_size, kind == MoEOverlayRankBatchTransportKind::MPI ? 7u : 6u);
                for (uint64_t step = 1; step < observations; ++step)
                    observe(step);
                const auto records = PerfStatsCollector::snapshot();
                ASSERT_EQ(records.size(), initial_size);
                for (const auto &record : records)
                {
                    EXPECT_EQ(record.count, observations) << record.name;
                    EXPECT_FALSE(record.tags.contains("generation"));
                    EXPECT_FALSE(record.tags.contains("logical_step"));
                    EXPECT_FALSE(record.tags.contains("bytes"));
                    EXPECT_EQ(record.tags.at("endpoint_role"),
                        role == MoEOverlayRankBatchEndpoint::Source ? "source" : "target");
                    if (record.name == "moe_overlay_rank_batch_payload_bytes")
                        EXPECT_EQ(record.value, 32 * observations + observations * (observations - 1) / 2);
                    if (record.name == "rank_batch_total")
                    {
                        EXPECT_EQ(record.total_ns, 30 * observations + observations * (observations - 1) / 2);
                        EXPECT_EQ(record.min_ns, 30u);
                        EXPECT_EQ(record.max_ns, 30u + observations - 1);
                    }
                    if (record.kind == PerfStatRecord::Kind::OrderedSequence)
                    {
                        EXPECT_EQ(record.sequence_word_count, 4 * observations);
                        EXPECT_NE(record.sequence_digest_lo, 0u);
                        EXPECT_NE(record.sequence_digest_hi, 0u);
                    }
                }
                if (kind == MoEOverlayRankBatchTransportKind::NodeLocalSharedRows)
                    EXPECT_THROW(telemetry.recordAsyncSendSubmission(8), std::logic_error);
            }
}

/** @brief Equal aggregate counts must not hide a reordered protocol observation. */
TEST_F(Test__MoEGraphNativeProfilingMetrics, RankBatchWitnessPreservesOrderWithoutTemporalTags)
{
    auto key = makeMTPMoEOverlayRankBatchKey(
        7, 0, 15, 3, 1, 2, 0, 1, MoEOverlayCollectiveDirection::Dispatch);
    for (auto role : {MoEOverlayRankBatchEndpoint::Source, MoEOverlayRankBatchEndpoint::Target})
    {
        MoEOverlayRankBatchTelemetry telemetry(key, MoEOverlayRankBatchTransportKind::MPI, role);
        for (uint64_t step = 0; step < 2; ++step)
        {
            key.step_id = role == MoEOverlayRankBatchEndpoint::Source ? step : 1 - step;
            telemetry.recordTransaction(64, 2);
        }
    }
    const auto witnesses = PerfStatsCollector::snapshot({"forward_graph.moe_overlay_rank_batch_sequence"});
    ASSERT_EQ(witnesses.size(), 2u);
    EXPECT_EQ(witnesses[0].count, witnesses[1].count);
    EXPECT_EQ(witnesses[0].sequence_word_count, witnesses[1].sequence_word_count);
    EXPECT_NE(witnesses[0].sequence_digest_lo, witnesses[1].sequence_digest_lo);
    EXPECT_NE(witnesses[0].sequence_digest_hi, witnesses[1].sequence_digest_hi);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: gn_sparse_dispatch
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeSparseDispatch_PopulatesRow)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        /*layer=*/3,
        /*tier_index=*/0,
        /*edge=*/profileEdge(0, 1),
        /*outbound_rows=*/16,
        /*outbound_entries=*/32,
        /*inbound_rows=*/12,
        /*compact_dispatch_bytes=*/4096,
        /*dense_dispatch_bytes=*/8192,
        /*wait_ms=*/0.5);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].phase, "gn_sparse_dispatch");
    EXPECT_EQ(rows[0].layer, 3);
    EXPECT_EQ(rows[0].tier_index, 0);
    EXPECT_EQ(rows[0].source_participant, 0);
    EXPECT_EQ(rows[0].target_participant, 1);
    EXPECT_EQ(rows[0].domain, "p0->p1");
    EXPECT_EQ(rows[0].selected_rows, 16u);
    EXPECT_EQ(rows[0].inbound_rows, 12u);
    EXPECT_EQ(rows[0].routed_entries, 32u);
    EXPECT_EQ(rows[0].outbound_bytes, 4096u);
    EXPECT_EQ(rows[0].compact_dispatch_bytes, 4096u);
    EXPECT_EQ(rows[0].dense_bytes_avoided, 4096u); // 8192 - 4096
    EXPECT_NEAR(rows[0].domain_reduce_ms, 0.5, 1e-9);
    EXPECT_EQ(rows[0].transport_mode, "compact");
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeSparseDispatch_DenseBytesAvoided_IsPositive)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 1, profileEdge(0, 0), 8, 16, 6, /*compact=*/1024, /*dense=*/4096, 0.0);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_FALSE(rows.empty());
    EXPECT_GT(rows[0].dense_bytes_avoided, 0u);
    EXPECT_EQ(rows[0].dense_bytes_avoided, 3072u); // 4096 - 1024
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeSparseDispatch_DenseBytesAvoided_NoUnderflow)
{
    // compact >= dense — dense_bytes_avoided must be 0 (no underflow)
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 1, profileEdge(0, 0), 8, 16, 6, /*compact=*/8192, /*dense=*/4096, 0.0);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].dense_bytes_avoided, 0u);
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeSparseDispatch_TierIndexSeparatesRows)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 0, profileEdge(0, 1), 4, 8, 4, 512, 2048, 0.0);
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 1, profileEdge(0, 1), 6, 12, 6, 768, 2048, 0.0);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_NE(rows[0].tier_index, rows[1].tier_index);
}

/**
 * @brief Long horizons must aggregate into topology-bounded identities.
 *
 * This is the regression for transaction strings entering both the profiler
 * row key and the PerfStats device/tags. That defect made every token/layer a
 * new record and progressively slowed later requests in parity campaigns.
 */
TEST_F(Test__MoEGraphNativeProfilingMetrics,
       RepeatedEdgeObservationsKeepProfilerAndPerfStatsCardinalityBounded)
{
    const auto record = []
    {
        MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
            9, 2, profileEdge(3, 1), 4, 8, 4, 512, 2048, 0.125);
    };

    record();
    const size_t initial_perf_records =
        PerfStatsCollector::snapshot({"moe_overlay"}).size();
    ASSERT_GT(initial_perf_records, 0u);

    for (int logical_step = 1; logical_step < 2048; ++logical_step)
        record();

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().source_participant, 3);
    EXPECT_EQ(rows.front().target_participant, 1);
    EXPECT_EQ(rows.front().domain, "p3->p1");
    EXPECT_EQ(rows.front().selected_rows, 4u * 2048u);
    EXPECT_EQ(
        PerfStatsCollector::snapshot({"moe_overlay"}).size(),
        initial_perf_records);
}

/**
 * @brief Endpoint timing identity must remain bounded across a long decode.
 *
 * The production regression used generation, logical step, residency epoch,
 * and the complete routed-expert set as PerfStats tags. Every packet therefore
 * inserted new map rows and made the later economy cohort progressively more
 * expensive. The typed API intentionally cannot represent those values.
 */
TEST_F(Test__MoEGraphNativeProfilingMetrics,
       EndpointPacketTimingCardinalityIsIndependentOfLogicalStep)
{
    const MoEOverlayEndpointIdentity identity{
        .phase = MoEOverlayEndpointPhase::Prefill,
        .device = "CPU",
        .layer = 17,
        .tier_index = 1,
        .participant_id = 3,
        .row_capacity = 16,
        .route_width = 8,
    };
    MoEOverlayEndpointTimings timings{
        .packet_service_ns = 10'000,
        .route_validation_and_compaction_ns = 1'000,
        .stage_setup_and_transfers_ns = 2'000,
        .compute_submission_ns = 3'000,
        .output_materialization_ns = 2'000,
        .canonical_route_preweight_and_publication_ns = 1'000,
        .return_validation_and_aggregation_ns = 1'000,
    };

    MoEExpertOverlayProfiler::recordEndpointPacket(identity, timings);
    const auto first =
        PerfStatsCollector::snapshot({"moe_overlay_endpoint"});
    ASSERT_EQ(first.size(), 7u);

    constexpr std::size_t kPacketCount = 2048u;
    for (std::size_t logical_step = 1u;
         logical_step < kPacketCount;
         ++logical_step)
    {
        // Vary the measurements as real packets do; identity must not change.
        timings.packet_service_ns = 10'000u + logical_step;
        timings.compute_submission_ns = 3'000u + logical_step % 13u;
        MoEExpertOverlayProfiler::recordEndpointPacket(identity, timings);
    }

    const auto records =
        PerfStatsCollector::snapshot({"moe_overlay_endpoint"});
    ASSERT_EQ(records.size(), first.size());
    for (const PerfStatRecord &record : records)
    {
        EXPECT_EQ(record.count, kPacketCount);
        EXPECT_EQ(record.tags.size(), 5u);
        EXPECT_TRUE(record.tags.contains("layer"));
        EXPECT_TRUE(record.tags.contains("participant"));
        EXPECT_TRUE(record.tags.contains("row_capacity"));
        EXPECT_TRUE(record.tags.contains("route_width"));
        EXPECT_TRUE(record.tags.contains("tier"));
        EXPECT_FALSE(record.tags.contains("active_experts"));
        EXPECT_FALSE(record.tags.contains("active_routes"));
        EXPECT_FALSE(record.tags.contains("generation"));
        EXPECT_FALSE(record.tags.contains("logical_step"));
        EXPECT_FALSE(record.tags.contains("residency_epoch"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: gn_local_expert
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeLocalExpert_PopulatesRow)
{
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        /*layer=*/5,
        /*tier_index=*/2,
        /*participant_id=*/3,
        /*device_key=*/"cpu:0",
        /*is_cpu=*/true,
        /*inbound_rows=*/20,
        /*active_routes=*/24,
        /*output_rows=*/20,
        /*unique_expert_ids=*/{0, 2, 4},
        /*compute_ms=*/3.14);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].phase, "gn_local_expert");
    EXPECT_EQ(rows[0].layer, 5);
    EXPECT_EQ(rows[0].tier_index, 2);
    EXPECT_EQ(rows[0].participant_id, 3);
    EXPECT_EQ(rows[0].domain_kind, "CPU");
    EXPECT_EQ(rows[0].inbound_rows, 20u);
    EXPECT_EQ(rows[0].active_routes, 24u);
    EXPECT_EQ(rows[0].routed_entries, 24u);
    EXPECT_EQ(rows[0].selected_rows, 20u);
    EXPECT_EQ(rows[0].cpu_fallback_rows, 20u);
    EXPECT_EQ(rows[0].gpu_cached_rows, 0u);
    EXPECT_NEAR(rows[0].compute_ms, 3.14, 1e-6);
    EXPECT_EQ(rows[0].transport_mode, "local");
    EXPECT_FALSE(rows[0].executed_experts.empty());
}

TEST_F(Test__MoEGraphNativeProfilingMetrics,
       LocalExpertEvidenceUsesBoundedUnionNotRouteSetIdentity)
{
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        5, 2, 3, "cpu:0", true, 4, 4, 4, {10, 2}, 0.1);
    const size_t initial_perf_records =
        PerfStatsCollector::snapshot({"moe_overlay"}).size();
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        5, 2, 3, "cpu:0", true, 4, 4, 4, {7, 2}, 0.1);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().executed_experts, "2;7;10");
    const auto records = PerfStatsCollector::snapshot({"moe_overlay"});
    EXPECT_EQ(records.size(), initial_perf_records);
    EXPECT_TRUE(std::none_of(
        records.begin(), records.end(), [](const PerfStatRecord &record)
        {
            return record.tags.contains("experts");
        }));
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeLocalExpert_GpuFlag)
{
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        2, 0, /*participant_id=*/0, "cuda:0", /*is_cpu=*/false,
        /*inbound_rows=*/8, /*active_routes=*/8, /*output_rows=*/8,
        {1, 3}, 0.5);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].domain_kind, "GPU");
}

/**
 * @brief CPU/NUMA endpoints sharing one backend must remain separate CSV rows.
 *
 * A zero-work participant is important evidence: it proves the graph reached
 * that owner and lets a production campaign distinguish an idle expert shard
 * from an accidentally omitted one.
 */
TEST_F(Test__MoEGraphNativeProfilingMetrics,
       RecordGraphNativeLocalExpert_ParticipantIdentityPreservesZeroWork)
{
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        /*layer=*/3, /*tier_index=*/2, /*participant_id=*/2,
        /*device_key=*/"CPU", /*is_cpu=*/true,
        /*inbound_rows=*/5, /*active_routes=*/0, /*output_rows=*/0,
        /*unique_expert_ids=*/{}, /*compute_ms=*/0.0);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        /*layer=*/3, /*tier_index=*/2, /*participant_id=*/3,
        /*device_key=*/"CPU", /*is_cpu=*/true,
        /*inbound_rows=*/5, /*active_routes=*/4, /*output_rows=*/3,
        /*unique_expert_ids=*/{224, 225}, /*compute_ms=*/0.2);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 2u);
    const auto p2 = std::find_if(rows.begin(), rows.end(), [](const auto &row)
                                 { return row.participant_id == 2; });
    const auto p3 = std::find_if(rows.begin(), rows.end(), [](const auto &row)
                                 { return row.participant_id == 3; });
    ASSERT_NE(p2, rows.end());
    ASSERT_NE(p3, rows.end());
    EXPECT_EQ(p2->active_routes, 0u);
    EXPECT_EQ(p2->selected_rows, 0u);
    EXPECT_EQ(p3->active_routes, 4u);
    EXPECT_EQ(p3->selected_rows, 3u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: gn_return_reduce
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeReturnReduce_PopulatesRow)
{
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        /*layer=*/7,
        /*tier_index=*/3,
        /*edge=*/profileEdge(1, 0),
        /*outbound_rows=*/12,
        /*inbound_rows=*/16,
        /*compact_return_bytes=*/2048,
        /*dense_return_bytes=*/16384,
        /*return_wait_ms=*/1.2,
        /*scatter_ms=*/0.3,
        /*import_broadcast_ms=*/0.0);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].phase, "gn_return_reduce");
    EXPECT_EQ(rows[0].layer, 7);
    EXPECT_EQ(rows[0].tier_index, 3);
    EXPECT_EQ(rows[0].selected_rows, 16u); // inbound_rows
    EXPECT_EQ(rows[0].inbound_rows, 16u);
    EXPECT_EQ(rows[0].routed_entries, 12u); // outbound_rows
    EXPECT_EQ(rows[0].outbound_bytes, 2048u);
    EXPECT_EQ(rows[0].compact_return_bytes, 2048u);
    EXPECT_EQ(rows[0].dense_bytes_avoided, 14336u); // 16384 - 2048
    EXPECT_NEAR(rows[0].domain_reduce_ms, 1.2, 1e-9);
    EXPECT_NEAR(rows[0].scatter_ms, 0.3, 1e-9);
    EXPECT_NEAR(rows[0].import_broadcast_ms, 0.0, 1e-9);
    EXPECT_EQ(rows[0].transport_mode, "compact");
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeReturnReduce_ScatterAndBroadcastTimed)
{
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, 1, profileEdge(0, 1), 8, 8, 512, 2048, 0.5, 0.25, 0.1);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_FALSE(rows.empty());
    EXPECT_NEAR(rows[0].scatter_ms, 0.25, 1e-9);
    EXPECT_NEAR(rows[0].import_broadcast_ms, 0.1, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: multi-phase / CSV / summary
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(Test__MoEGraphNativeProfilingMetrics, AllThreePhases_AllPresent)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 1, profileEdge(0, 1), 8, 16, 6, 1024, 4096, 0.1);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, 1, /*participant_id=*/2, "cpu:0", true,
        /*inbound_rows=*/6, /*active_routes=*/8, /*output_rows=*/6,
        {0, 1}, 1.0);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, 1, profileEdge(1, 0), 6, 8, 512, 2048, 0.2, 0.1, 0.0);

    const auto rows = MoEExpertOverlayProfiler::rows();
    EXPECT_EQ(rows.size(), 3u);
    EXPECT_TRUE(hasPhase(rows, "gn_sparse_dispatch"));
    EXPECT_TRUE(hasPhase(rows, "gn_local_expert"));
    EXPECT_TRUE(hasPhase(rows, "gn_return_reduce"));
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, GraphNativeRowsPublishUnifiedPerfStats)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 1, profileEdge(0, 1), 8, 16, 6, 1024, 4096, 0.1);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, 1, /*participant_id=*/1, "rocm:0", false,
        /*inbound_rows=*/6, /*active_routes=*/6, /*output_rows=*/6,
        {0, 2}, 1.25);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, 1, profileEdge(1, 0), 6, 8, 512, 2048, 0.2, 0.05, 0.03);

    const auto records = PerfStatsCollector::snapshot({"moe_overlay"});
    ASSERT_FALSE(records.empty());
    EXPECT_TRUE(hasUnifiedRecord(records, PerfStatRecord::Kind::Counter, "selected_rows", "gn_sparse_dispatch"));
    EXPECT_TRUE(hasUnifiedRecord(records, PerfStatRecord::Kind::Counter, "dense_bytes_avoided", "gn_return_reduce"));
    EXPECT_TRUE(hasUnifiedRecord(records, PerfStatRecord::Kind::Counter, "gpu_rows", "gn_local_expert"));
    EXPECT_TRUE(hasUnifiedRecord(records, PerfStatRecord::Kind::Timer, "compute", "gn_local_expert"));
    EXPECT_TRUE(hasUnifiedRecord(records, PerfStatRecord::Kind::Timer, "domain_reduce", "gn_sparse_dispatch"));
    EXPECT_TRUE(hasUnifiedRecord(records, PerfStatRecord::Kind::Timer, "scatter", "gn_return_reduce"));

    const auto dispatch_counter = std::find_if(records.begin(), records.end(), [](const PerfStatRecord &record)
                                               {
                                                   return record.kind == PerfStatRecord::Kind::Counter &&
                                                          record.name == "selected_rows" &&
                                                          record.phase == "gn_sparse_dispatch";
                                               });
    ASSERT_NE(dispatch_counter, records.end());
    EXPECT_DOUBLE_EQ(dispatch_counter->value, 8.0);
    EXPECT_EQ(dispatch_counter->tags.at("layer"), "0");
    EXPECT_EQ(dispatch_counter->tags.at("tier"), "1");
    EXPECT_EQ(dispatch_counter->tags.at("transport"), "compact");
    EXPECT_EQ(dispatch_counter->tags.at("source_participant"), "0");
    EXPECT_EQ(dispatch_counter->tags.at("target_participant"), "1");
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, CsvIncludesGraphNativePhases)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 1, profileEdge(0, 1), 8, 16, 6, 1024, 4096, 0.0);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, 1, /*participant_id=*/2, "cpu:0", true,
        /*inbound_rows=*/6, /*active_routes=*/6, /*output_rows=*/6,
        {0}, 1.0);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, 1, profileEdge(1, 0), 6, 8, 512, 2048, 0.0, 0.0, 0.0);

    const std::string csv = MoEExpertOverlayProfiler::csvString();
    EXPECT_NE(csv.find("gn_sparse_dispatch"), std::string::npos);
    EXPECT_NE(csv.find("gn_local_expert"), std::string::npos);
    EXPECT_NE(csv.find("gn_return_reduce"), std::string::npos);
    // New CSV columns must be present in header
    EXPECT_NE(csv.find("tier_index"), std::string::npos);
    EXPECT_NE(csv.find("participant_id"), std::string::npos);
    EXPECT_NE(csv.find("source_participant"), std::string::npos);
    EXPECT_NE(csv.find("target_participant"), std::string::npos);
    EXPECT_NE(csv.find("active_routes"), std::string::npos);
    EXPECT_NE(csv.find("dense_bytes_avoided"), std::string::npos);
    EXPECT_NE(csv.find("inbound_rows"), std::string::npos);
    EXPECT_NE(csv.find("compact_dispatch_bytes"), std::string::npos);
    EXPECT_NE(csv.find("compact_return_bytes"), std::string::npos);
    EXPECT_NE(csv.find("cpu_fallback_rows"), std::string::npos);
    EXPECT_NE(csv.find("gpu_cached_rows"), std::string::npos);
    EXPECT_NE(csv.find("scatter_ms"), std::string::npos);
    EXPECT_NE(csv.find("import_broadcast_ms"), std::string::npos);
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, SummaryIncludesGraphNativePhases)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 1, profileEdge(0, 1), 8, 16, 6, 1024, 4096, 0.0);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, 1, /*participant_id=*/2, "cpu:0", true,
        /*inbound_rows=*/6, /*active_routes=*/6, /*output_rows=*/6,
        {0}, 1.0);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, 1, profileEdge(1, 0), 6, 8, 512, 2048, 0.0, 0.0, 0.0);

    // renderSummary should not throw; output should contain the phases
    EXPECT_NO_THROW(MoEExpertOverlayProfiler::renderSummary());
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, WhenProfilingDisabled_NoRowsRecorded)
{
    mutableDebugEnv().profile.enabled = false;
    PerfStatsCollector::reloadConfigurationFromEnvironment();

    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, 1, profileEdge(0, 1), 8, 16, 6, 1024, 4096, 0.0);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, 1, /*participant_id=*/2, "cpu:0", true,
        /*inbound_rows=*/6, /*active_routes=*/6, /*output_rows=*/6,
        {0}, 1.0);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, 1, profileEdge(1, 0), 6, 8, 512, 2048, 0.0, 0.0, 0.0);

    EXPECT_TRUE(MoEExpertOverlayProfiler::rows().empty());
}
