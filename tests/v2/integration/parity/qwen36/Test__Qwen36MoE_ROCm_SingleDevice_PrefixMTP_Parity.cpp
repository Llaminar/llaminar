#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <algorithm>
#include <cstdlib>
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

    void expectRocmMoEMTPVerifierUsesSafeCompositeGroupedPrefillPath(int expected_seq_len = 2)
    {
        const auto records = PerfStatsCollector::snapshot({"kernel", "mtp"});
        auto tag_equals = [](const PerfStatRecord &record,
                             const char *key,
                             const char *value) -> bool
        {
            const auto it = record.tags.find(key);
            return it != record.tags.end() && it->second == value;
        };
        auto tag_int_in_range = [](const PerfStatRecord &record,
                                   const char *key,
                                   int lower_bound,
                                   int upper_bound) -> bool
        {
            const auto it = record.tags.find(key);
            if (it == record.tags.end())
            {
                return false;
            }
            char *end = nullptr;
            const long value = std::strtol(it->second.c_str(), &end, 10);
            return end != it->second.c_str() && end != nullptr && *end == '\0' &&
                   value >= lower_bound && value <= upper_bound;
        };
        const int expected_routed_top_k = 8;
        const int expected_routed_experts = 256;
        const int expected_total_slots = expected_seq_len * expected_routed_top_k;
        // ROCm owns each routed gate/up row and each final token/down column in
        // fixed grouped launches. This byte-exact contract admits no
        // batch-shaped fused or K-part publication branch.
        const std::string seq_len_tag = std::to_string(expected_seq_len);
        const std::string total_slots_tag = std::to_string(expected_total_slots);

        const auto routed_grouped = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name == "rocm_moe_grouped_prefill_batch_invariant_calls" &&
                       tag_equals(record, "seq_len", seq_len_tag.c_str()) &&
                       tag_equals(record, "top_k", "8") &&
                       tag_equals(record, "total_slots", total_slots_tag.c_str()) &&
                       tag_int_in_range(record,
                                        "active_expert_slots",
                                        1,
                                        expected_total_slots) &&
                       tag_equals(record, "num_experts", "256") &&
                       tag_equals(record,
                                  "gateup_route",
                                  "route_owned_router_q8") &&
                       tag_equals(record,
                                  "down_route",
                                  "direct_ordered_publish") &&
                       tag_equals(record, "row_tile", "1");
            });
        ASSERT_NE(routed_grouped, records.end())
            << "ROCm Qwen3.6 MoE MTP verifier should stay on the current "
            << "graph-capturable route-owned grouped prefill path for verifier "
            << "rows while shared-expert work is owned by the standalone "
            << "decode-equivalent grouped table-prefill verifier stage.\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});

        const auto shared_grouped_table_prefill = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.domain == "mtp" &&
                       record.name == "moe_shared_grouped_decode_equivalent_verifier_prefill_rows" &&
                       tag_equals(record, "route", "grouped_table_prefill") &&
                       tag_equals(record, "stage", "shared_expert");
            });
        ASSERT_NE(shared_grouped_table_prefill, records.end())
            << "ROCm Qwen3.6 MoE MTP verifier did not run the standalone "
            << "shared-expert grouped table-prefill verifier path.\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});

        const auto combined = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.domain == "mtp" &&
                       record.name == "moe_combined_decode_equivalent_verifier_prefill_rows";
            });
        ASSERT_EQ(combined, records.end())
            << "ROCm Qwen3.6 MoE MTP verifier unexpectedly ran the unaccepted "
            << "combined routed+shared owner.\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});

    }
} // namespace

#define QWEN36_MOE_PREFIX_MTP_SUITE Qwen36MoEROCmSingleDevicePrefixMTPParity
#define QWEN36_MOE_PREFIX_MTP_CASE rocmSingleDeviceCase
#define QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE rocmSingleDeviceBenchmarkPromptCase
#define QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE rocmSingleDeviceDepth3Case
#define QWEN36_MOE_PREFIX_MTP_EXPECTS_DIRECT_PUBLICATION 0
#define QWEN36_MOE_PREFIX_MTP_EXPECTS_PERSISTENT_SIDECAR_METADATA 1
#define QWEN36_MOE_PREFIX_MTP_TESTS_DEVICE_RESIDENT_PUBLICATION 1
#include "Qwen36MoESingleDevicePrefixMTPParityTests.inc"

TEST(Qwen36MoEROCmSingleDevicePrefixMTPPathGuards, GroupedVerifierUsesRoutedPrefillPath)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
    });
    PerfStatsCollector::reset();
    runMoEMainVerifierGroupedRowsMatchSerialDecode(
        rocmSingleDeviceBenchmarkPromptCase(),
        2);
    expectRocmMoEMTPVerifierUsesSafeCompositeGroupedPrefillPath(2);
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
