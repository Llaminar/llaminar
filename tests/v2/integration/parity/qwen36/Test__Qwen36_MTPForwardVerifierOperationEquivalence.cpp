/**
 * @file Test__Qwen36_MTPForwardVerifierOperationEquivalence.cpp
 * @brief Focused Qwen3.6 MTP verifier-forward operation equivalence tests.
 *
 * The full PyTorch parity harness is intentionally broad: it proves end-to-end
 * model behavior, but a failing token or final logit does not immediately say
 * which verifier operation drifted.  This focused suite uses serial Llaminar
 * decode as the oracle for the target verifier pass itself.  For each backend
 * and model family, it runs grouped verifier rows for M=1,2,3,4, captures every
 * stage snapshot exposed by the graph, and compares each verifier row against
 * the corresponding serial decode prefix.
 *
 * The contract being protected here is stricter than ordinary "same sampled
 * token" parity.  The grouped verifier path must produce the same per-operation
 * tensors, the same row logits, the same greedy samples, and, for GPU MoE, the
 * expected grouped verifier kernel counters.  That makes this suite a regression
 * net for the CUDA, ROCm, and CPU verifier-forward paths without forcing every
 * iteration through the heavier PyTorch snapshot matrix.
 */

#include "Qwen36MoEParityTestBase.h"

#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen36;

namespace
{
    /**
     * @brief Backends covered by the focused operation-equivalence matrix.
     *
     * The enum keeps the call sites explicit: CPU proves the pure serial/device
     * graph behavior, while CUDA and ROCm also validate that the optimized GPU
     * MoE verifier kernels are actually exercised by the grouped pass.
     */
    enum class VerifierBackend
    {
        CPU,
        CUDA,
        ROCm,
    };

    /**
     * @brief Assert that multi-row verifier logits used the grouped LM-head path.
     *
     * The logits byte-equality check proves correctness after the fact.  This
     * perfstats guard proves the graph reached the intended production owner:
     * the decode-equivalent grouped LM-head projection, not an ordinary M>1
     * GEMM whose reduction order can drift by a few ULPs.
     */
    void expectLMHeadGroupedDecodeEquivalentVerifierPrefillPath(
        int expected_seq_len)
    {
        if (expected_seq_len <= 1)
            return;

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        const auto grouped_lm_head = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                const auto route_it = record.tags.find("route");
                return record.domain == "mtp" &&
                       record.name ==
                           "lm_head_grouped_decode_equivalent_verifier_prefill_rows" &&
                       route_it != record.tags.end() &&
                       route_it->second == "grouped" &&
                       record.value >= static_cast<double>(expected_seq_len);
            });
        ASSERT_NE(grouped_lm_head, records.end())
            << "M=" << expected_seq_len
            << " verifier logits must exercise the grouped decode-equivalent "
               "LM-head path.\n"
            << PerfStatsCollector::summaryString({"mtp"});
    }

    /**
     * @brief Print generated CUDA grouped-GEMV decisions for a failed depth.
     *
     * A full model run records thousands of kernel counters. Dumping all of
     * them obscures the first divergent projection, so this diagnostic keeps
     * only NativeVNNI dispatches for serial M=1 and the failing runtime M.
     * Printing both is important because eager and graph-captured policy
     * surfaces may select different reduction parenthesizations. Geometry,
     * codebook, execution mode, route, K partitioning, and grouped row reuse
     * remain visible together, which is enough to distinguish a grouped-kernel
     * arithmetic defect from a mismatched execution-mode contract.
     */
    void printCudaGroupedGemvFailureDiagnostics(int verifier_rows)
    {
        const std::string expected_m = std::to_string(verifier_rows);
        const auto records = PerfStatsCollector::snapshot({"kernel"});
        std::cerr << "CUDA grouped verifier dispatches for M="
                  << verifier_rows << ":\n";
        for (const auto &record : records)
        {
            if (record.name != "cuda_native_vnni_gemv_dispatch")
                continue;

            const auto m = record.tags.find("m");
            if (m == record.tags.end() ||
                (m->second != expected_m && m->second != "1"))
            {
                continue;
            }

            auto tag = [&](const char *name) -> std::string
            {
                const auto it = record.tags.find(name);
                return it == record.tags.end() ? "<missing>" : it->second;
            };
            std::cerr << "  codebook=" << tag("codebook")
                      << " m=" << tag("m")
                      << " n=" << tag("n")
                      << " k=" << tag("k")
                      << " mode=" << tag("execution_mode")
                      << " policy_mode=" << tag("policy_execution_mode")
                      << " contract=" << tag("semantic_contract")
                      << " candidate=" << tag("effective_candidate_id")
                      << " route=" << tag("route")
                      << " tile_n=" << tag("tile_n")
                      << " cpt=" << tag("cpt")
                      << " effective_kb=" << tag("effective_kb")
                      << '\n';
        }
    }

    /**
     * @brief Build a dense SingleDevice parity case for the requested backend.
     *
     * The shared dense helper owns model loading, prefix setup, row-plan
     * construction, grouped verifier execution, serial replay, and operation
     * snapshot comparison.  This case function only selects the device and the
     * minimum device-count guard needed for a loud skip on machines without that
     * backend.
     */
    DensePrefixRestoreParityCase denseSingleDeviceCase(VerifierBackend backend)
    {
        auto test_case = qwen36DensePrefixParityCase(
            "Qwen3.6 dense verifier-forward operation equivalence",
            DensePrefixParityTopology::SingleDevice);

        switch (backend)
        {
        case VerifierBackend::CPU:
            test_case.name += " CPU";
            test_case.devices = {GlobalDeviceAddress::cpu()};
            test_case.required_cuda_devices = 0;
            test_case.required_rocm_devices = 0;
            break;
        case VerifierBackend::CUDA:
            test_case.name += " CUDA";
            test_case.devices = {GlobalDeviceAddress::cuda(0)};
            test_case.required_cuda_devices = 1;
            test_case.required_rocm_devices = 0;
            break;
        case VerifierBackend::ROCm:
            test_case.name += " ROCm";
            test_case.devices = {GlobalDeviceAddress::rocm(0)};
            test_case.required_cuda_devices = 0;
            test_case.required_rocm_devices = 1;
            break;
        }

        return test_case;
    }

    /**
     * @brief Build a MoE benchmark-prompt case for verifier-forward coverage.
     *
     * MoE verifier bugs tend to depend on top-k routing and shared-expert work,
     * so this suite intentionally uses the same benchmark-prompt metadata as
     * the existing MoE MTP diagnostic cells.  The longer prompt gives routed
     * experts and recurrent state enough context to expose row-order, gather, and
     * publication mistakes while still avoiding a full long-context parity run.
     */
    MoEPrefixRestoreParityCase moeBenchmarkPromptCase(VerifierBackend backend)
    {
        auto test_case = qwen36MoEPrefixParityCase(
            "Qwen3.6 MoE verifier-forward operation equivalence",
            MoEPrefixParityTopology::SingleDevice);

        test_case.prompt = qwen36MoEBenchmarkPrompt();
        test_case.decode_steps = 4;
        test_case.max_seq_len = 768;

        switch (backend)
        {
        case VerifierBackend::CPU:
            test_case.name += " CPU";
            test_case.devices = {GlobalDeviceAddress::cpu()};
            test_case.required_cuda_devices = 0;
            test_case.required_rocm_devices = 0;
            test_case.metadata_envs = {
                "LLAMINAR_QWEN36_MOE_CPU_MTP_DIAGNOSTIC_METADATA"};
            test_case.default_metadata_path =
                "pytorch_qwen36_moe_cpu_mtp_diagnostic_snapshots/metadata.txt";
            break;
        case VerifierBackend::CUDA:
            test_case.name += " CUDA";
            test_case.devices = {GlobalDeviceAddress::cuda(0)};
            test_case.required_cuda_devices = 1;
            test_case.required_rocm_devices = 0;
            test_case.metadata_envs = {
                "LLAMINAR_QWEN36_MOE_CUDA_MTP_DIAGNOSTIC_METADATA"};
            test_case.default_metadata_path =
                "pytorch_qwen36_moe_cuda_mtp_diagnostic_snapshots/metadata.txt";
            break;
        case VerifierBackend::ROCm:
            test_case.name += " ROCm";
            test_case.devices = {GlobalDeviceAddress::rocm(0)};
            test_case.required_cuda_devices = 0;
            test_case.required_rocm_devices = 1;
            test_case.metadata_envs = {
                "LLAMINAR_QWEN36_MOE_ROCM_MTP_DIAGNOSTIC_METADATA"};
            test_case.default_metadata_path =
                "pytorch_qwen36_moe_rocm_mtp_diagnostic_snapshots/metadata.txt";
            break;
        }

        return test_case;
    }

    /**
     * @brief Assert that the ROCm MoE grouped verifier used the target kernels.
     *
     * Output equality alone can hide a rowwise or partially serialized fallback.
     * The ROCm target contract for this path is active-expert grouped prefill for
     * the routed branch plus the standalone grouped table-prefill owner for the
     * shared expert.  This check mirrors the CUDA guard in the shared parity base
     * and deliberately rejects the older combined routed+shared owner.
     */
    void expectRocmMoEVerifierGroupedPrefillPath(int expected_seq_len)
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
        const int expected_total_slots =
            expected_seq_len * expected_routed_top_k;
        const std::string seq_len_tag = std::to_string(expected_seq_len);
        const std::string total_slots_tag =
            std::to_string(expected_total_slots);

        if (expected_seq_len == 1)
        {
            const auto decode_equivalent = std::find_if(
                records.begin(),
                records.end(),
                [](const PerfStatRecord &record)
                {
                    return record.domain == "mtp" &&
                           record.name == "moe_decode_equivalent_verifier_prefill_runs";
                });
            ASSERT_EQ(decode_equivalent, records.end())
                << "ROCm MoE M=1 verifier must use ordinary production decode, "
                   "not a decode-equivalent oracle path.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});

            const auto combined = std::find_if(
                records.begin(),
                records.end(),
                [](const PerfStatRecord &record)
                {
                    return record.domain == "mtp" &&
                           record.name ==
                               "moe_combined_decode_equivalent_verifier_prefill_rows";
                });
            ASSERT_EQ(combined, records.end())
                << "ROCm MoE verifier unexpectedly used the retired "
                   "combined routed+shared owner for M=1.\n"
                << PerfStatsCollector::summaryString({"kernel", "mtp"});
            return;
        }

        const auto routed_grouped = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name ==
                           "rocm_moe_grouped_prefill_batch_invariant_calls" &&
                       tag_equals(record, "seq_len", seq_len_tag.c_str()) &&
                       tag_equals(record, "top_k", "8") &&
                       tag_equals(record,
                                  "total_slots",
                                  total_slots_tag.c_str()) &&
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
            << "ROCm MoE grouped verifier did not exercise the batch-invariant "
               "direct verifier path for M="
            << expected_seq_len << ".\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});

        const auto router_q8_reuse = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.name ==
                           "rocm_moe_grouped_prefill_router_q8_reuse_calls" &&
                       tag_equals(record, "seq_len", seq_len_tag.c_str()) &&
                       tag_equals(record, "top_k", "8") &&
                       tag_equals(record,
                                  "descriptor_source",
                                  "static_table");
            });
        ASSERT_NE(router_q8_reuse, records.end())
            << "ROCm MoE grouped verifier did not reuse the router-owned Q8 "
               "hidden rows for M="
            << expected_seq_len << ".\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});

        const auto shared_grouped_table_prefill = std::find_if(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                return record.domain == "mtp" &&
                       record.name ==
                           "moe_shared_grouped_decode_equivalent_verifier_prefill_rows" &&
                       tag_equals(record, "route", "grouped_table_prefill") &&
                       tag_equals(record, "stage", "shared_expert");
            });
        ASSERT_NE(shared_grouped_table_prefill, records.end())
            << "ROCm MoE grouped verifier did not exercise the standalone "
               "shared-expert grouped table-prefill verifier path for M="
            << expected_seq_len << ".\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});

        const auto combined = std::find_if(
            records.begin(),
            records.end(),
            [](const PerfStatRecord &record)
            {
                return record.domain == "mtp" &&
                       record.name ==
                           "moe_combined_decode_equivalent_verifier_prefill_rows";
            });
        ASSERT_EQ(combined, records.end())
            << "ROCm MoE grouped verifier unexpectedly used the retired "
               "combined routed+shared owner for M="
            << expected_seq_len << ".\n"
            << PerfStatsCollector::summaryString({"kernel", "mtp"});
    }

    /**
     * @brief Run dense grouped verifier rows M=1..4 against serial decode.
     *
     * The diagnostic environment flag is scoped here so the test itself always
     * compares captured operation snapshots.  Developers running this binary by
     * hand do not need to remember a separate environment incantation.
     */
    void runDenseVerifierOperationCase(VerifierBackend backend, int verifier_rows)
    {
        ScopedEnvironmentValues operation_diagnostics({
            {"LLAMINAR_DENSE_VERIFIER_SNAPSHOT_DIAGNOSTIC", "1"},
            {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });
        const auto test_case = denseSingleDeviceCase(backend);
        SCOPED_TRACE("dense verifier_rows=" + std::to_string(verifier_rows));
        PerfStatsCollector::reset();
        runDenseMainVerifierGroupedRowsMatchSerialDecode(
            test_case,
            verifier_rows);
        if (::testing::Test::HasFailure())
        {
            if (backend == VerifierBackend::CUDA)
                printCudaGroupedGemvFailureDiagnostics(verifier_rows);
            PerfStatsCollector::reset();
            return;
        }
        expectLMHeadGroupedDecodeEquivalentVerifierPrefillPath(verifier_rows);
        PerfStatsCollector::reset();
    }

    /**
     * @brief Run one MoE grouped verifier bucket against serial decode.
     *
     * In addition to enabling operation snapshots, this helper enables
     * perfstats for the GPU cells and checks that the grouped verifier really
     * used the target CUDA/ROCm MoE paths.  CPU has no GPU kernel path to prove,
     * but still gets the same row-logit and per-operation snapshot comparisons.
     */
    void runMoEVerifierOperationCase(VerifierBackend backend, int verifier_rows)
    {
        ScopedEnvironmentValues operation_diagnostics({
            {"LLAMINAR_MOE_GROUPED_VERIFIER_SNAPSHOT_DIAGNOSTIC", "1"},
            {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
            {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
        });
        const auto test_case = moeBenchmarkPromptCase(backend);
        SCOPED_TRACE("MoE verifier_rows=" + std::to_string(verifier_rows));
        PerfStatsCollector::reset();
        runMoEMainVerifierGroupedRowsMatchSerialDecode(
            test_case,
            verifier_rows);
        if (::testing::Test::HasFailure())
        {
            if (backend == VerifierBackend::CUDA)
                printCudaGroupedGemvFailureDiagnostics(verifier_rows);
            PerfStatsCollector::reset();
            return;
        }
        expectLMHeadGroupedDecodeEquivalentVerifierPrefillPath(verifier_rows);

        if (backend == VerifierBackend::CUDA)
        {
            expectCudaMoEMTPVerifierFusedPrefillPath(verifier_rows);
        }
        else if (backend == VerifierBackend::ROCm)
        {
            expectRocmMoEVerifierGroupedPrefillPath(verifier_rows);
        }
        PerfStatsCollector::reset();
    }
} // namespace

#define LLAMINAR_QWEN36_VERIFIER_OPERATION_TESTS(BACKEND_LABEL, BACKEND_ENUM) \
    TEST(Qwen36MTPForwardVerifierOperationEquivalence, Dense##BACKEND_LABEL##_M1) \
    { \
        runDenseVerifierOperationCase(VerifierBackend::BACKEND_ENUM, 1); \
    } \
    TEST(Qwen36MTPForwardVerifierOperationEquivalence, Dense##BACKEND_LABEL##_M2) \
    { \
        runDenseVerifierOperationCase(VerifierBackend::BACKEND_ENUM, 2); \
    } \
    TEST(Qwen36MTPForwardVerifierOperationEquivalence, Dense##BACKEND_LABEL##_M3) \
    { \
        runDenseVerifierOperationCase(VerifierBackend::BACKEND_ENUM, 3); \
    } \
    TEST(Qwen36MTPForwardVerifierOperationEquivalence, Dense##BACKEND_LABEL##_M4) \
    { \
        runDenseVerifierOperationCase(VerifierBackend::BACKEND_ENUM, 4); \
    } \
    TEST(Qwen36MTPForwardVerifierOperationEquivalence, MoE##BACKEND_LABEL##_M1) \
    { \
        runMoEVerifierOperationCase(VerifierBackend::BACKEND_ENUM, 1); \
    } \
    TEST(Qwen36MTPForwardVerifierOperationEquivalence, MoE##BACKEND_LABEL##_M2) \
    { \
        runMoEVerifierOperationCase(VerifierBackend::BACKEND_ENUM, 2); \
    } \
    TEST(Qwen36MTPForwardVerifierOperationEquivalence, MoE##BACKEND_LABEL##_M3) \
    { \
        runMoEVerifierOperationCase(VerifierBackend::BACKEND_ENUM, 3); \
    } \
    TEST(Qwen36MTPForwardVerifierOperationEquivalence, MoE##BACKEND_LABEL##_M4) \
    { \
        runMoEVerifierOperationCase(VerifierBackend::BACKEND_ENUM, 4); \
    }

LLAMINAR_QWEN36_VERIFIER_OPERATION_TESTS(CPU, CPU)
LLAMINAR_QWEN36_VERIFIER_OPERATION_TESTS(CUDA, CUDA)
LLAMINAR_QWEN36_VERIFIER_OPERATION_TESTS(ROCm, ROCm)

#undef LLAMINAR_QWEN36_VERIFIER_OPERATION_TESTS

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
