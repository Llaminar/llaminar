#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <algorithm>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    MoEPrefixRestoreParityCase cudaSingleDeviceCase()
    {
        auto test_case = qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE CUDA SingleDevice parity",
            MoEPrefixParityTopology::SingleDevice);
        test_case.devices = {GlobalDeviceAddress::cuda(0)};
        test_case.required_cuda_devices = 1;
        test_case.required_rocm_devices = 0;
        // The metadata fixture's third greedy token is a near-tie on CUDA MoE
        // (760 vs 71093) across fresh model loads. Keep exact prefix/MTP
        // restore checks on the stable first two tokens; the dedicated CUDA
        // math parity suite remains responsible for layer/logit tolerances.
        test_case.decode_steps = 2;
        return test_case;
    }

    MoEPrefixRestoreParityCase cudaSingleDeviceBenchmarkPromptCase()
    {
        auto test_case = cudaSingleDeviceCase();
        test_case.name = "Qwen3.6 MoE CUDA SingleDevice benchmark-prompt MTP diagnostic";
        test_case.prompt = qwen36MoEBenchmarkPrompt();
        test_case.metadata_envs = {"LLAMINAR_QWEN36_MOE_CUDA_MTP_DIAGNOSTIC_METADATA"};
        test_case.default_metadata_path =
            "pytorch_qwen36_moe_cuda_mtp_diagnostic_snapshots/metadata.txt";
        test_case.decode_steps = 4;
        test_case.max_seq_len = 768;
        return test_case;
    }

    MoEPrefixRestoreParityCase cudaSingleDeviceDepth3Case()
    {
        auto test_case = cudaSingleDeviceBenchmarkPromptCase();
        test_case.name = "Qwen3.6 MoE CUDA SingleDevice depth-3 MTP parity";
        test_case.decode_steps = 4;
        return test_case;
    }

    void expectCudaMoESharedExpertGroupedDecodePath()
    {
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.cuda_moe_grouped_decode_gateup_calls",
             "kernel.cuda_moe_grouped_decode_down_calls"});
        auto tag_equals = [](const PerfStatRecord &record,
                             const char *key,
                             const char *value) -> bool
        {
            const auto it = record.tags.find(key);
            return it != record.tags.end() && it->second == value;
        };
        auto has_shared_decode_record = [&](const char *name) -> bool
        {
            return std::find_if(
                       records.begin(),
                       records.end(),
                       [&](const PerfStatRecord &record)
                       {
                           return record.name == name &&
                                  tag_equals(record, "source", "table") &&
                                  (tag_equals(record, "route", "serial") ||
                                   tag_equals(record, "route", "kpart")) &&
                                  tag_equals(record, "active_slots", "1") &&
                                  tag_equals(record, "d_model", "2048") &&
                                  tag_equals(record, "intermediate", "512");
                       }) != records.end();
        };

        ASSERT_TRUE(has_shared_decode_record("cuda_moe_grouped_decode_gateup_calls"))
            << "CUDA shared expert decode must use the grouped table gate/up path.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_grouped_decode_gateup_calls",
                    "kernel.cuda_moe_grouped_decode_down_calls"});
        ASSERT_TRUE(has_shared_decode_record("cuda_moe_grouped_decode_down_calls"))
            << "CUDA shared expert decode must use the grouped table down path.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cuda_moe_grouped_decode_gateup_calls",
                    "kernel.cuda_moe_grouped_decode_down_calls"});
    }

} // namespace

#define QWEN36_MOE_PREFIX_MTP_SUITE Qwen36MoECUDASingleDevicePrefixMTPParity
#define QWEN36_MOE_PREFIX_MTP_CASE cudaSingleDeviceCase
#define QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE cudaSingleDeviceBenchmarkPromptCase
#define QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE cudaSingleDeviceDepth3Case
#include "Qwen36MoESingleDevicePrefixMTPParityTests.inc"

TEST(Qwen36MoECUDASingleDevicePrefixMTPPathGuards, MTPBenchmarkStyleUsesFusedVerifierPrefillPath)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMTPBenchmarkStyleSkipGatherParity(
        cudaSingleDeviceBenchmarkPromptCase(),
        4);
    expectCudaMoEMTPVerifierGDNProjectionFusedPath();
    expectCudaMoEMTPVerifierFusedPrefillPath();
    expectCudaMoEMTPVerifierSharedExpertFusedPrefillPath();
    PerfStatsCollector::reset();
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPPathGuards, Depth1CorrectionReplayResetsCapturedStateBoundary)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMTPBenchmarkStyleSkipGatherParity(
        cudaSingleDeviceBenchmarkPromptCase(),
        16,
        1,
        {},
        true);

    const auto records = PerfStatsCollector::snapshot({"mtp"});
    auto tag_equals = [](const PerfStatRecord &record,
                         const char *key,
                         const char *value) -> bool
    {
        const auto it = record.tags.find(key);
        return it != record.tags.end() && it->second == value;
    };

    const auto reset_boundary = std::find_if(
        records.begin(),
        records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.kind == PerfStatRecord::Kind::Counter &&
                   record.domain == "mtp" &&
                   record.name == "live_prefix_replay_state_after_mutation" &&
                   tag_equals(record, "operation", "mtp_spec_state_publication") &&
                   tag_equals(record, "replay_state", "reset") &&
                   tag_equals(record, "kernel_dynamic_state", "reset");
        });
    ASSERT_NE(reset_boundary, records.end())
        << "CUDA MoE depth-1 MTP correction replay must reset captured replay "
        << "and kernel dynamic state after accepted-state publication. "
        << "Preserving verifier replay state across this rejected-token boundary "
        << "can leave the following main decode graph with stale captured state.\n"
        << PerfStatsCollector::summaryString({"mtp"});

    const auto correction_forward = std::find_if(
        records.begin(),
        records.end(),
        [&](const PerfStatRecord &record)
        {
            return record.domain == "mtp" &&
                   record.name == "all_position_correction_forward";
        });
    ASSERT_NE(correction_forward, records.end())
        << "Regression case did not exercise all-position correction replay.\n"
        << PerfStatsCollector::summaryString({"mtp"});
    PerfStatsCollector::reset();
}

TEST(Qwen36MoECUDASingleDevicePrefixMTPPathGuards, NoMTPBenchmarkStyleUsesGroupedDecodePath)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
    });
    PerfStatsCollector::reset();
    runMoENoMTPBenchmarkStyleSkipGatherArgmaxParity(
        cudaSingleDeviceBenchmarkPromptCase(),
        16);
    expectCudaMoESharedExpertGroupedDecodePath();
    PerfStatsCollector::reset();
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
