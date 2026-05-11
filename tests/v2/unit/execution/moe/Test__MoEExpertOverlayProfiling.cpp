/**
 * @file Test__MoEExpertOverlayProfiling.cpp
 * @brief Unit coverage for Phase 9A MoE expert overlay profiling plumbing.
 */

#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoEExpertParallelReduceStage.h"
#include "execution/moe/MoEExpertOverlayCPUFallback.h"
#include "execution/moe/MoEExpertOverlayLocalTPExecutor.h"
#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "utils/DebugEnv.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr const char *kMoEProfilerEnvVars[] = {
            "LLAMINAR_MOE_EP_DENSE_TRANSFER",
            "LLAMINAR_MOE_EP_TRACE",
            "LLAMINAR_MOE_EP_DUMP_PLACEMENT",
            "LLAMINAR_MOE_EP_TRANSFER_TRACE",
            "LLAMINAR_MOE_EP_PROFILE_CSV",
        };

        void clearMoEProfilerEnv()
        {
            for (const char *name : kMoEProfilerEnvVars)
                unsetenv(name);
            mutableDebugEnv().reload();
            MoEExpertOverlayProfiler::reset();
        }

        struct MoEProfilerEnvGuard
        {
            MoEProfilerEnvGuard() { clearMoEProfilerEnv(); }
            ~MoEProfilerEnvGuard() { clearMoEProfilerEnv(); }
        };

        MoEOverlayRuntimeDomain makeLocalTPDomain()
        {
            MoEOverlayRuntimeDomain domain;
            domain.name = "gpu_fast";
            domain.kind = ExpertDomainKind::LocalTP;
            domain.backend = CollectiveBackendType::NCCL;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            domain.participants.resize(2);
            domain.participants[0].participant_index = 0;
            domain.participants[0].local_device = DeviceId::cuda(0);
            domain.participants[1].participant_index = 1;
            domain.participants[1].local_device = DeviceId::cuda(1);
            return domain;
        }

        ExpertComputeDomain makeCPUFallbackDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cpu_safety";
            domain.kind = ExpertDomainKind::NodeLocalTP;
            domain.backend = CollectiveBackendType::MPI;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            domain.participants.resize(2);
            return domain;
        }
    } // namespace

    TEST(Test__MoEExpertOverlayProfiling, DebugEnvParsesOverlayProfilerControls)
    {
        MoEProfilerEnvGuard guard;

        setenv("LLAMINAR_MOE_EP_DENSE_TRANSFER", "1", 1);
        setenv("LLAMINAR_MOE_EP_TRACE", "true", 1);
        setenv("LLAMINAR_MOE_EP_DUMP_PLACEMENT", "on", 1);
        setenv("LLAMINAR_MOE_EP_TRANSFER_TRACE", "yes", 1);
        setenv("LLAMINAR_MOE_EP_PROFILE_CSV", "moe_overlay_profile.csv", 1);
        mutableDebugEnv().reload();

        const auto &env = debugEnv().moe_expert_overlay;
        EXPECT_TRUE(env.dense_transfer);
        EXPECT_TRUE(env.trace);
        EXPECT_TRUE(env.dump_placement);
        EXPECT_TRUE(env.transfer_trace);
        EXPECT_TRUE(env.profile_csv_enabled);
        EXPECT_EQ(env.profile_csv_path, "moe_overlay_profile.csv");

        setenv("LLAMINAR_MOE_EP_PROFILE_CSV", "1", 1);
        mutableDebugEnv().reload();
        EXPECT_TRUE(debugEnv().moe_expert_overlay.profile_csv_enabled);
        EXPECT_FALSE(debugEnv().moe_expert_overlay.profile_csv_path.empty());

        setenv("LLAMINAR_MOE_EP_PROFILE_CSV", "off", 1);
        mutableDebugEnv().reload();
        EXPECT_FALSE(debugEnv().moe_expert_overlay.profile_csv_enabled);
    }

    TEST(Test__MoEExpertOverlayProfiling, AggregatesRowsWithStableCsvShape)
    {
        MoEProfilerEnvGuard guard;

        MoEExpertOverlayProfileRow row;
        row.phase = "localtp";
        row.layer = 3;
        row.domain = "gpu_fast";
        row.domain_kind = "LocalTP";
        row.backend = "nccl";
        row.routed_entries = 7;
        row.selected_rows = 4;
        row.transfer_bytes = 1024;
        row.compute_ms = 1.5;
        row.domain_reduce_ms = 0.25;
        row.participant_count = 2;
        row.executed_experts = "1;4";
        row.transport_mode = "SparseTokenRows";
        MoEExpertOverlayProfiler::recordRow(row);

        row.routed_entries = 5;
        row.selected_rows = 2;
        row.transfer_bytes = 512;
        row.compute_ms = 0.5;
        row.executed_experts = "4;7";
        MoEExpertOverlayProfiler::recordRow(row);

        const auto rows = MoEExpertOverlayProfiler::rows();
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].routed_entries, 12u);
        EXPECT_EQ(rows[0].selected_rows, 6u);
        EXPECT_EQ(rows[0].transfer_bytes, 1536u);
        EXPECT_DOUBLE_EQ(rows[0].compute_ms, 2.0);
        EXPECT_EQ(rows[0].participant_count, 2);
        EXPECT_NE(rows[0].executed_experts.find("7"), std::string::npos);

        const std::string csv = MoEExpertOverlayProfiler::csvString();
        EXPECT_NE(csv.find("phase,layer,domain,domain_kind,backend"), std::string::npos);
        EXPECT_NE(csv.find("cross_domain_reduce_ms"), std::string::npos);
        EXPECT_NE(csv.find("gpu_fast"), std::string::npos);

        const std::string summary = MoEExpertOverlayProfiler::renderSummary();
        EXPECT_NE(summary.find("MOE EXPERT OVERLAY PROFILING SUMMARY"), std::string::npos);
        EXPECT_NE(summary.find("gpu_fast"), std::string::npos);
    }

    TEST(Test__MoEExpertOverlayProfiling, RecordsDispatchLocalFallbackAndFinalReduceDiagnostics)
    {
        MoEProfilerEnvGuard guard;
        setenv("LLAMINAR_MOE_EP_TRACE", "1", 1);
        mutableDebugEnv().reload();

        ExpertLayerPlacement placement;
        placement.layer = 5;
        placement.routed_expert_tier = {0, 1, 1, 0};

        std::vector<ExpertRoutedTier> routed_tiers(2);
        routed_tiers[0].name = "fast";
        routed_tiers[0].domain = "gpu_fast";
        routed_tiers[0].max_experts_per_layer = 2;
        routed_tiers[1].name = "safety";
        routed_tiers[1].domain = "cpu_safety";
        routed_tiers[1].fallback = true;

        MoEExpertDispatchOutput output;
        output.seq_len = 2;
        output.top_k = 2;
        output.d_model = 8;
        output.continuation_domain = "gpu_fast";
        output.tiers.resize(2);
        output.tiers[0].tier_index = 0;
        output.tiers[0].domain = "gpu_fast";
        output.tiers[0].transfer_mode = MoEExpertTransferMode::SparseTokenRows;
        output.tiers[0].entries.resize(3);
        output.tiers[0].token_rows = {0, 1};
        output.tiers[0].transfer_volume.outbound_bytes = 64;
        output.tiers[0].transfer_volume.return_bytes = 32;
        output.tiers[1].tier_index = 1;
        output.tiers[1].domain = "cpu_safety";
        output.tiers[1].transfer_mode = MoEExpertTransferMode::DenseFullSequence;
        output.tiers[1].entries.resize(1);
        output.tiers[1].token_rows = {1};
        output.tiers[1].transfer_volume.outbound_bytes = 128;
        output.tiers[1].transfer_volume.return_bytes = 128;
        MoEExpertOverlayProfiler::recordDispatch(5, output, placement, routed_tiers);

        auto local_domain = makeLocalTPDomain();
        MoEExpertOverlayLocalTPDiagnostics local_diag;
        local_diag.domain_name = "gpu_fast";
        local_diag.backend = CollectiveBackendType::NCCL;
        local_diag.degree = 2;
        local_diag.total_routed_entries = 3;
        local_diag.transfer_mode = MoEExpertTransferMode::SparseTokenRows;
        local_diag.transfer_volume.outbound_bytes = 64;
        local_diag.transfer_volume.return_bytes = 32;
        local_diag.selected_token_rows = {0, 1};
        local_diag.compute_ms = 2.25;
        local_diag.domain_reduce_ms = 0.75;
        local_diag.participants.resize(2);
        local_diag.participants[0].executed_expert_ids = {0, 3};
        local_diag.participants[1].executed_expert_ids = {0, 3};
        MoEExpertOverlayProfiler::recordLocalTP(5, local_domain, 2, local_diag);

        auto fallback_domain = makeCPUFallbackDomain();
        MoECPUFallbackTransferStats transfer_stats;
        transfer_stats.mode = MoEExpertTransferMode::DenseFullSequence;
        transfer_stats.token_rows = {1};
        transfer_stats.volume.outbound_bytes = 128;
        transfer_stats.volume.return_bytes = 128;
        MoECPUFallbackTensorParallelStats tp_stats;
        tp_stats.domain_degree = 2;
        tp_stats.processed_expert_ids = {1, 2};
        MoEExpertOverlayProfiler::recordCPUFallback(
            5, fallback_domain, 2, 1, 1, &transfer_stats, &tp_stats, 3.5);

        MoEExpertParallelReduceDiagnostics reduce_diag;
        reduce_diag.mode = MoEExpertParallelReduceMode::ContinuationDeviceOptimized;
        reduce_diag.continuation_domain = "gpu_fast";
        reduce_diag.host_staged = false;
        reduce_diag.output_resident_on_continuation = true;
        reduce_diag.partial_count = 2;
        reduce_diag.device_to_host_bytes = 16;
        reduce_diag.host_to_device_bytes = 32;
        reduce_diag.total_transfer_bytes = 48;
        reduce_diag.reduce_ms = 0.42;
        reduce_diag.partials.resize(2);
        reduce_diag.partials[0].accumulation_path = MoEExpertParallelReducePartialAccumulationPath::ContinuationDeviceAccumulated;
        reduce_diag.partials[1].accumulation_path = MoEExpertParallelReducePartialAccumulationPath::HostStagedThenDeviceAccumulated;
        MoEExpertOverlayProfiler::recordFinalReduce(5, reduce_diag);

        const auto rows = MoEExpertOverlayProfiler::rows();
        ASSERT_EQ(rows.size(), 5u);

        const std::string csv = MoEExpertOverlayProfiler::csvString();
        EXPECT_NE(csv.find("dispatch"), std::string::npos);
        EXPECT_NE(csv.find("localtp"), std::string::npos);
        EXPECT_NE(csv.find("cpu_fallback"), std::string::npos);
        EXPECT_NE(csv.find("final_reduce"), std::string::npos);
        EXPECT_NE(csv.find("ContinuationDeviceOptimized"), std::string::npos);
        EXPECT_NE(csv.find("HostStagedThenDeviceAccumulated"), std::string::npos);
    }

} // namespace llaminar2::test
