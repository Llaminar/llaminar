/**
 * @file Test__IQ4XSSourceEquations.cpp
 * @brief Focused captured IQ4 source-equation regressions on CUDA and ROCm.
 *
 * Keep the source-preservation fixtures in a dedicated translation unit so
 * changes do not recompile the large general GEMM/MoE suites. Every registered
 * case drives production preparation, descriptors, streams and captured kernels;
 * the separately named verifier diagnostic explicitly selects DIRECT/ROWPAR. Mixed-pair cases prove both scale-program orders and
 * alternating large/partial runtime-planned prefill, in addition to decode.
 */
#include "utils/IQ4XSFusedMoEGpuSourceWitness.h"
#include "utils/IQ4XSMixedPrefillSourceWitness.h"
#include "utils/DebugEnv.h"
#include <cstdlib>
#include <gtest/gtest.h>

namespace witness=llaminar2::test::iq4xs_fused_source_witness;

namespace
{
/**
 * @brief Own the kernel-counter policy for one isolated source-equation test.
 *
 * Enable collection before any worker constructs kernels or graphs. Explicit
 * kernel filtering keeps inherited caller filters from hiding route evidence;
 * restoring both variables and clearing records prevents cross-test evidence.
 */
class ScopedIQ4KernelStats final
{
public:
    /** @brief Save the caller policy and begin a fresh kernel evidence scope. */
    ScopedIQ4KernelStats()
    {
        for (size_t i = 0; i < names_.size(); ++i)
        {
            if (const char *value = std::getenv(names_[i]))
            {
                present_[i] = true;
                previous_[i] = value;
            }
        }
        setenv(names_[0], "1", 1);
        setenv(names_[1], "kernel", 1);
        llaminar2::mutableDebugEnv().reload();
        llaminar2::PerfStatsCollector::reset();
    }

    /** @brief Restore caller policy after all worker/graph owners have retired. */
    ~ScopedIQ4KernelStats()
    {
        for (size_t i = 0; i < names_.size(); ++i)
        {
            if (present_[i]) setenv(names_[i], previous_[i].c_str(), 1);
            else unsetenv(names_[i]);
        }
        llaminar2::mutableDebugEnv().reload();
        llaminar2::PerfStatsCollector::reset();
    }

    ScopedIQ4KernelStats(const ScopedIQ4KernelStats &) = delete;
    ScopedIQ4KernelStats &operator=(const ScopedIQ4KernelStats &) = delete;

private:
    static constexpr std::array<const char *, 2> names_{
        "LLAMINAR_PERF_STATS_SUMMARY", "LLAMINAR_PERF_STATS_FILTER"};
    std::array<bool, 2> present_{};
    std::array<std::string, 2> previous_{};
};
} // namespace

#ifdef HAVE_CUDA
/** @brief All original source factors and the IQ4_NL control through dense capture. */
TEST(IQ4XSSourceCUDA,DenseOriginalEquation)
{
    ScopedIQ4KernelStats stats;
    witness::runDense(llaminar2::DeviceId::cuda(0));
}
/** @brief Source gate/up/down equations through the real fused decode transaction. */
TEST(IQ4XSSourceCUDA,FusedOriginalEquation)
{
    ScopedIQ4KernelStats stats;
    witness::runGpu(llaminar2::DeviceId::cuda(0));
}
/** @brief Both mixed source orders through M1 and retained M512/M3 prefill graphs. */
TEST(IQ4XSSourceCUDA,MixedSourcesCaptured)
{
    ScopedIQ4KernelStats stats;
    witness::runGpu(llaminar2::DeviceId::cuda(0),witness::mixedRoles);
    witness::runMixedPrefill(llaminar2::DeviceId::cuda(0));
}
#endif

#ifdef HAVE_ROCM
/** @brief All original source factors and the IQ4_NL control through dense capture. */
TEST(IQ4XSSourceROCm,DenseOriginalEquation)
{
    ScopedIQ4KernelStats stats;
    witness::runDense(llaminar2::DeviceId::rocm(0));
}
/** @brief Source gate/up/down equations through the real fused decode transaction. */
TEST(IQ4XSSourceROCm,FusedOriginalEquation)
{
    ScopedIQ4KernelStats stats;
    witness::runGpu(llaminar2::DeviceId::rocm(0));
}
/** @brief Both mixed source orders through M1 and retained M512/M3 prefill graphs. */
TEST(IQ4XSSourceROCm,MixedSourcesCaptured)
{
    ScopedIQ4KernelStats stats;
    witness::runGpu(llaminar2::DeviceId::rocm(0),witness::mixedRoles);
    witness::runMixedPrefill(llaminar2::DeviceId::rocm(0));
}
#endif

#ifdef HAVE_CUDA
extern "C" {
void cudaNativeVNNIGemvSweep_setConfig(int,int,int,int,int,int,int,int);
void cudaNativeVNNIGemvSweep_setGroupedRows(int);
void cudaNativeVNNIGemvSweep_clearConfig();
int cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config();
void cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(int);
}
namespace llaminar2::test {
/**
 * @brief Exercise IQ4 source factors through actual DIRECT and ROWPAR bypasses.
 *
 * The ordinary all-format source-equation test remains separate. This explicit
 * verifier diagnostic uses the same original blocks, exact observed Q8 input,
 * fixed single-live-block oracle and retained twenty changed-input replays.
 * Dispatch records must prove both M1 and M8 physical24/source4 launches and
 * unchanged IQ4_NL physical4; setup-only records cannot satisfy capture proof.
 */
TEST(IQ4XSSourceCUDA, DirectAndRowparVerifierCaptured)
{
    const DeviceId gpu_device_ = DeviceId::cuda(0);
    namespace witness = llaminar2::test::iq4xs_fused_source_witness;
    constexpr std::array<int,2> rows{1,8};
    ScopedIQ4KernelStats stats;
    for (int family : {2,3})
    {
        SCOPED_TRACE("family=" + std::to_string(family));
        PerfStatsCollector::reset();
        /** @brief Restore diagnostic family state even after a fatal assertion. */
        struct Selection
        {
            int previous; ///< Existing serial equivalence policy.
            /** @brief Select exactly one already implemented dispatch family. */
            explicit Selection(int selected)
                : previous(cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config())
            {
                cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(1);
                cudaNativeVNNIGemvSweep_setConfig(selected,
                    selected == 3 ? 2 : 64,1,1,1,1,0,0);
                cudaNativeVNNIGemvSweep_setGroupedRows(8);
            }
            /** @brief Clear the family and restore the previous scoped policy. */
            ~Selection()
            {
                cudaNativeVNNIGemvSweep_clearConfig();
                cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(previous);
            }
        } selection(family);
        ASSERT_NO_FATAL_FAILURE(witness::runDense(gpu_device_, rows,
            witness::DenseExecutionScope::VerifierDecodeEquivalent));
        const auto records = PerfStatsCollector::snapshot(
            {"kernel.cuda_native_vnni_gemv_dispatch"});
        for (int physical : {4,24}) for (int m : rows)
        {
            uint64_t captured = 0;
            for (const auto &record : records)
            {
                if (record.domain != "kernel" || record.name != "cuda_native_vnni_gemv_dispatch" ||
                    record.device != "cuda:" + std::to_string(gpu_device_.ordinal)) continue;
                if (record.tags.at("codebook") != std::to_string(physical) ||
                    record.tags.at("m") != std::to_string(m) ||
                    record.tags.at("execution_mode") != "graph_captured") continue;
                ASSERT_EQ(record.kind, PerfStatRecord::Kind::Counter);
                ASSERT_EQ(record.tags.at("arithmetic_policy_codebook"), "4");
                ASSERT_EQ(record.tags.at("route"), family == 3 ? "rowpar" : "direct");
                if (family == 3) ASSERT_EQ(record.tags.at("rowmajor_available"), "true");
                captured += record.count;
            }
            EXPECT_GT(captured, 0u) << "physical=" << physical << " M=" << m;
        }
    }
}

} // namespace llaminar2::test
#endif
