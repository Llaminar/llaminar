#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <algorithm>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    MoEPrefixRestoreParityCase rocmSingleDeviceCase()
    {
        auto test_case = qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE ROCm SingleDevice parity",
            MoEPrefixParityTopology::SingleDevice);
        test_case.devices = {GlobalDeviceAddress::rocm(0)};
        test_case.required_cuda_devices = 0;
        test_case.required_rocm_devices = 1;
        test_case.decode_steps = 2;
        return test_case;
    }

    MoEPrefixRestoreParityCase rocmSingleDeviceBenchmarkPromptCase()
    {
        auto test_case = rocmSingleDeviceCase();
        test_case.name = "Qwen3.6 MoE ROCm SingleDevice benchmark-prompt MTP diagnostic";
        test_case.prompt = qwen36MoEBenchmarkPrompt();
        test_case.metadata_envs = {"LLAMINAR_QWEN36_MOE_ROCM_MTP_DIAGNOSTIC_METADATA"};
        test_case.default_metadata_path =
            "pytorch_qwen36_moe_rocm_mtp_diagnostic_snapshots/metadata.txt";
        test_case.decode_steps = 4;
        test_case.max_seq_len = 768;
        return test_case;
    }

    MoEPrefixRestoreParityCase rocmSingleDeviceDepth3Case()
    {
        auto test_case = rocmSingleDeviceBenchmarkPromptCase();
        test_case.name = "Qwen3.6 MoE ROCm SingleDevice depth-3 MTP parity";
        test_case.decode_steps = 4;
        return test_case;
    }

    void expectRocmMoEMTPVerifierUsesCombinedKPartPrefillPath(int expected_seq_len = 2)
    {
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.rocm_moe_combined_shared_prefill_group_calls",
             "kernel.rocm_moe_grouped_prefill_active_expert_grid_calls",
             "kernel.rocm_moe_shared_expert_prefill_group_calls"});
        auto tag_equals = [](const PerfStatRecord &record,
                             const char *key,
                             const char *value) -> bool
        {
            const auto it = record.tags.find(key);
            return it != record.tags.end() && it->second == value;
        };
        const int expected_combined_top_k = 9;
        const int expected_combined_experts = 257;
        const int expected_total_slots = expected_seq_len * expected_combined_top_k;
        const int expected_active_slots = std::min(expected_total_slots, expected_combined_experts);
        // ROCm keeps Qwen3.6 MoE verifier buckets, including M=3/4, on the
        // compact tile-M=2 grouped-prefill lane. CUDA uses tile-M=4 for M=3/4,
        // but the ROCm MI50 evidence favored tile-M=2; the production contract
        // here is the combined K-part route, not identical CUDA tile geometry.
        const int expected_tile_m = 2;
        const std::string seq_len_tag = std::to_string(expected_seq_len);
        const std::string total_slots_tag = std::to_string(expected_total_slots);
        const std::string active_slots_tag = std::to_string(expected_active_slots);
        const std::string tile_m_tag = std::to_string(expected_tile_m);

        const auto combined_group = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "rocm_moe_combined_shared_prefill_group_calls" &&
                       tag_equals(record, "seq_len", seq_len_tag.c_str()) &&
                       tag_equals(record, "routed_top_k", "8") &&
                       tag_equals(record, "combined_top_k", "9") &&
                       tag_equals(record, "active_expert_slots", active_slots_tag.c_str());
            });
        ASSERT_NE(combined_group, records.end())
            << "ROCm Qwen3.6 MoE MTP verifier must keep the vLLM-style "
            << "combined routed+shared expert grouping path. Splitting routed "
            << "and shared experts here reintroduces extra graph work in the "
            << "hot verifier lane.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.rocm_moe_combined_shared_prefill_group_calls"});

        const auto routed_kpart = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "rocm_moe_grouped_prefill_active_expert_grid_calls" &&
                       tag_equals(record, "seq_len", seq_len_tag.c_str()) &&
                       tag_equals(record, "top_k", "9") &&
                       tag_equals(record, "total_slots", total_slots_tag.c_str()) &&
                       tag_equals(record, "active_expert_slots", active_slots_tag.c_str()) &&
                       tag_equals(record, "tile_m", tile_m_tag.c_str()) &&
                       tag_equals(record, "gateup_route", "kpart_prefill");
            });
        ASSERT_NE(routed_kpart, records.end())
            << "ROCm Qwen3.6 MoE MTP verifier should stay on the current "
            << "graph-capturable K-part active-expert grouped prefill path for "
            << "verifier rows. A focused A/B showed this route is faster than the "
            << "non-K-part fused route on the MI50 lane, so falling off it is a "
            << "Phase 10 performance regression.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.rocm_moe_grouped_prefill_active_expert_grid_calls"});

        const auto split_shared_group = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "rocm_moe_shared_expert_prefill_group_calls" &&
                       tag_equals(record, "seq_len", seq_len_tag.c_str());
            });
        ASSERT_EQ(split_shared_group, records.end())
            << "ROCm Qwen3.6 MoE MTP verifier unexpectedly used the standalone "
            << "shared-expert grouping path. The benchmark-style verifier should "
            << "consume routed and shared experts as one combined group.\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.rocm_moe_shared_expert_prefill_group_calls"});
    }
} // namespace

#define QWEN36_MOE_PREFIX_MTP_SUITE Qwen36MoEROCmSingleDevicePrefixMTPParity
#define QWEN36_MOE_PREFIX_MTP_CASE rocmSingleDeviceCase
#define QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE rocmSingleDeviceBenchmarkPromptCase
#define QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE rocmSingleDeviceDepth3Case
#define QWEN36_MOE_PREFIX_MTP_EXPECTS_PERSISTENT_SIDECAR_METADATA 1
#include "Qwen36MoESingleDevicePrefixMTPParityTests.inc"

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
