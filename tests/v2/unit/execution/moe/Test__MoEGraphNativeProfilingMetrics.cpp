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
#include "utils/DebugEnv.h"

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
    }

    void TearDown() override
    {
        MoEExpertOverlayProfiler::reset();
        mutableDebugEnv().profile.enabled = false;
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests: gn_sparse_dispatch
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeSparseDispatch_PopulatesRow)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        /*layer=*/3,
        /*domain_key=*/"layer3/tier0/dispatch",
        /*source_participant=*/0,
        /*target_participant=*/1,
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
        0, "key", 0, 0, 8, 16, 6, /*compact=*/1024, /*dense=*/4096, 0.0);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_FALSE(rows.empty());
    EXPECT_GT(rows[0].dense_bytes_avoided, 0u);
    EXPECT_EQ(rows[0].dense_bytes_avoided, 3072u); // 4096 - 1024
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeSparseDispatch_DenseBytesAvoided_NoUnderflow)
{
    // compact >= dense — dense_bytes_avoided must be 0 (no underflow)
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, "key", 0, 0, 8, 16, 6, /*compact=*/8192, /*dense=*/4096, 0.0);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].dense_bytes_avoided, 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: gn_local_expert
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeLocalExpert_PopulatesRow)
{
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        /*layer=*/5,
        /*device_key=*/"cpu:0",
        /*is_cpu=*/true,
        /*input_rows=*/20,
        /*output_rows=*/20,
        /*unique_expert_ids=*/{0, 2, 4},
        /*compute_ms=*/3.14);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].phase, "gn_local_expert");
    EXPECT_EQ(rows[0].layer, 5);
    EXPECT_EQ(rows[0].domain_kind, "CPU");
    EXPECT_EQ(rows[0].selected_rows, 20u);
    EXPECT_EQ(rows[0].cpu_fallback_rows, 20u);
    EXPECT_EQ(rows[0].gpu_cached_rows, 0u);
    EXPECT_NEAR(rows[0].compute_ms, 3.14, 1e-6);
    EXPECT_EQ(rows[0].transport_mode, "local");
    EXPECT_FALSE(rows[0].executed_experts.empty());
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeLocalExpert_GpuFlag)
{
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        2, "cuda:0", /*is_cpu=*/false, 8, 8, {1, 3}, 0.5);

    const auto rows = MoEExpertOverlayProfiler::rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].domain_kind, "GPU");
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests: gn_return_reduce
// ─────────────────────────────────────────────────────────────────────────────
TEST_F(Test__MoEGraphNativeProfilingMetrics, RecordGraphNativeReturnReduce_PopulatesRow)
{
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        /*layer=*/7,
        /*domain_key=*/"layer7/tier0/return",
        /*source_participant=*/1,
        /*target_participant=*/0,
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
        0, "k", 0, 1, 8, 8, 512, 2048, 0.5, 0.25, 0.1);

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
        0, "k", 0, 1, 8, 16, 6, 1024, 4096, 0.1);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, "cpu:0", true, 6, 6, {0, 1}, 1.0);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, "k", 1, 0, 6, 8, 512, 2048, 0.2, 0.1, 0.0);

    const auto rows = MoEExpertOverlayProfiler::rows();
    EXPECT_EQ(rows.size(), 3u);
    EXPECT_TRUE(hasPhase(rows, "gn_sparse_dispatch"));
    EXPECT_TRUE(hasPhase(rows, "gn_local_expert"));
    EXPECT_TRUE(hasPhase(rows, "gn_return_reduce"));
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, CsvIncludesGraphNativePhases)
{
    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, "k", 0, 1, 8, 16, 6, 1024, 4096, 0.0);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, "cpu:0", true, 6, 6, {0}, 1.0);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, "k", 1, 0, 6, 8, 512, 2048, 0.0, 0.0, 0.0);

    const std::string csv = MoEExpertOverlayProfiler::csvString();
    EXPECT_NE(csv.find("gn_sparse_dispatch"), std::string::npos);
    EXPECT_NE(csv.find("gn_local_expert"), std::string::npos);
    EXPECT_NE(csv.find("gn_return_reduce"), std::string::npos);
    // New CSV columns must be present in header
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
        0, "k", 0, 1, 8, 16, 6, 1024, 4096, 0.0);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, "cpu:0", true, 6, 6, {0}, 1.0);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, "k", 1, 0, 6, 8, 512, 2048, 0.0, 0.0, 0.0);

    // renderSummary should not throw; output should contain the phases
    EXPECT_NO_THROW(MoEExpertOverlayProfiler::renderSummary());
}

TEST_F(Test__MoEGraphNativeProfilingMetrics, WhenProfilingDisabled_NoRowsRecorded)
{
    mutableDebugEnv().profile.enabled = false;

    MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
        0, "k", 0, 1, 8, 16, 6, 1024, 4096, 0.0);
    MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        0, "cpu:0", true, 6, 6, {0}, 1.0);
    MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
        0, "k", 1, 0, 6, 8, 512, 2048, 0.0, 0.0, 0.0);

    EXPECT_TRUE(MoEExpertOverlayProfiler::rows().empty());
}
