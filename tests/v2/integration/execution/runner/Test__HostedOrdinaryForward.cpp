/**
 * @file Test__HostedOrdinaryForward.cpp
 * @brief Complete host-serviced ExpertOverlay setup and retained ordinary replay.
 *
 * A tiny generated GGUF exercises production ownership and native graphs without
 * model downloads. GPU continuation plus CPU-resident routed experts must keep
 * its complete hosted body; it cannot be exported as one native child capture.
 * The control explicitly selects ordinary single-device GPU execution, with all
 * experts local and no ExpertOverlay authority. HIP's outer ticket scheduler
 * therefore retains a native forward independently of host-serviced ownership.
 * Successful submissions are observed from each actual production publisher;
 * optimized hosted retained replay does not emit the generic decode-phase key.
 */
#include "PipelineGenerationTestSupport.h"
#include "../../../utils/PlanningGGUFFixture.h"
#include "../../../utils/ServingGGUFFixturePayload.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include <gtest/gtest.h>
namespace llaminar2::test
{
namespace
{
/** @return Only compiled backend cells; broad execution skips absent hardware. */
std::vector<std::string> hostedBackends()
{
    std::vector<std::string> result;
#ifdef HAVE_CUDA
    result.emplace_back("CUDA_Hosted");
    result.emplace_back("CUDA_Native");
#endif
#ifdef HAVE_ROCM
    result.emplace_back("ROCm_Hosted");
    result.emplace_back("ROCm_Native");
#endif
    return result;
}
/**
 * @brief Count successful forwards from the producer owning this retained body.
 * @param retention Explicit hosted-body versus native-child ownership.
 * @param device Exact GPU participant; other devices cannot satisfy evidence.
 * @return Submitted hosted MainInference replays, or terminal-authenticated
 *         native composed invocations, after the public request has completed.
 * @throws std::invalid_argument If a fixture asks for unknown retention.
 *
 * Hosted retained replay emits its own counter only after the complete body was
 * submitted and its exact terminal ordered back to the transaction stream. The
 * public decode return retires that transaction. Native parents instead publish
 * their actual completed child invocations at the authenticated terminal. Never
 * synthesize either count from tokens or accept the other execution publisher.
 */
double completedForwardCount(OrdinaryForwardRetentionKind retention, DeviceId device)
{
    if (retention != OrdinaryForwardRetentionKind::HostedExpertOverlay &&
        retention != OrdinaryForwardRetentionKind::NativeComposable)
        throw std::invalid_argument("Unknown fixture ordinary retention kind");
    double count = 0;
    for (const auto &record : PerfStatsCollector::snapshot())
    {
        if (record.domain != "forward_graph" || record.device != device.toString())
            continue;
        if (retention == OrdinaryForwardRetentionKind::HostedExpertOverlay)
        {
            if (record.name == "hosted_retained_decode_replays" &&
                record.tags.contains("role") && record.tags.at("role") ==
                    std::to_string(static_cast<int>(ForwardExecutionRole::MainInference)))
                count += record.value;
        }
        else if (record.name == "decode_graph_phase" &&
                 record.tags.contains("context") && record.tags.at("context") == "main_decode" &&
                 record.tags.contains("phase") && record.tags.at("phase") == "replay" &&
                 record.tags.contains("source") && record.tags.at("source") == "captured_generation_parent" &&
                 record.tags.contains("observation") && record.tags.at("observation") == "authenticated_terminal")
            count += record.value;
    }
    return count;
}
/** @brief One native GPU control or GPU continuation with an actual host expert tier. */
class HostedOrdinaryForward : public ::testing::TestWithParam<std::string> {};
TEST_P(HostedOrdinaryForward, MaterializesCompleteHostedBodyAndReusesAcrossResets)
{
    const bool cuda = GetParam().starts_with("CUDA");
    const bool hosted = GetParam().ends_with("Hosted");
    const auto retention = hosted ? OrdinaryForwardRetentionKind::HostedExpertOverlay
                                  : OrdinaryForwardRetentionKind::NativeComposable;
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    if (!backend || backend->deviceCount() < 1) GTEST_SKIP() << "Requires " << GetParam() << " device zero";
    initCPUBackend(-1);
    const auto device = cuda ? DeviceId::cuda(0) : DeviceId::rocm(0);
    PlanningGGUFFixture fixture(true, false, GGUFTensorType::Q4_0, 256,
        std::nullopt, 8, PlanningGGUFFixture::Geometry::FullAttentionGraph);
    populateServingGGUFFixture(fixture);
    const std::string gpu = cuda ? "cuda:0" : "rocm:0";
    std::vector<std::string> arguments{"hosted-ordinary-forward", "-m", fixture.path(),
        "--mpi-procs", "1", "--context-length", "512", "--no-mtp",
        "--activation-precision", "fp32", "--kv-cache-precision", "fp16",
        "--moe-residency-maintenance", "off"};
    if (hosted)
        arguments.insert(arguments.end(), {"--backend", "heterogeneous",
            "--expert-tier", "native=" + gpu + ";priority=0;max-experts-per-layer=4",
            "--expert-tier", "host=cpu:0;priority=10;max-experts-per-layer=4",
            "--moe-routed-expert-continuation-domain", "native",
            "--moe-routed-expert-shared-domain", "native"});
    else
        // A single-tier ExpertOverlay requests the device controller, whose
        // valid topology requires at least two routed participants. This local
        // control owns all experts through ordinary single-device execution.
        arguments.insert(arguments.end(), {"--device", gpu});
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
    if (hosted)
    {
        ASSERT_TRUE(config.moe_routed_expert_plan && config.moe_routed_expert_plan->usesExpertOverlayAuthority());
    }
    else
    {
        ASSERT_FALSE(config.moe_routed_expert_plan && config.moe_routed_expert_plan->usesExpertOverlayAuthority());
    }
    config.prefill_max_bucket_size = 64;
    config.prefix_cache.enabled = false;
    publishOrchestrationStartupPolicy(config);
    OrdinaryGraphEvidence evidence;
    OrchestrationRunner runner(config, createExecutionPlanBuilder());
    ASSERT_TRUE(runner.initialize()) << runner.lastError();
    ASSERT_EQ(runner.primaryDeviceId(), device);
    const auto hosted_exports = [] {
        double count = 0;
        for (const auto &record : PerfStatsCollector::snapshot())
            if (record.domain == "forward_graph" && record.name == "ordinary_hosted_forward_exports")
                count += record.value;
        return count;
    };
    // Old unconditional native export fails initialize before this point.
    if (hosted) EXPECT_GT(hosted_exports(), 0.0);
    else EXPECT_EQ(hosted_exports(), 0.0) << "Local ROCm tickets retain a native-composable forward";
    SamplingParams sampling;
    sampling.temperature = 0.0F;
    sampling.seed = 123;
    runner.setSamplingParams(sampling);
    runner.setStopTokens({});
    double first_materializations = 0;
    for (int request = 0; request < 20; ++request)
    {
        SCOPED_TRACE(request);
        runner.clearCache();
        ASSERT_TRUE(runner.prefill({11, 12, 13})) << runner.lastError();
        const auto completed_before = completedForwardCount(retention, device);
        runner.setDecodeStepTokenBudget(3);
        const auto result = runner.decodeStep();
        ASSERT_TRUE(result.success()) << result.error;
        ASSERT_EQ(result.tokens.size(), 3u);
        // The first response samples existing prefill logits; the remaining
        // two responses each require one real complete retained forward.
        EXPECT_EQ(completedForwardCount(retention, device) - completed_before, 2.0);
        const auto materializations = evidence.materializations();
        ASSERT_GT(materializations, 0.0);
        if (request == 0) first_materializations = materializations;
        else EXPECT_EQ(materializations, first_materializations) << "Reset must reuse the same complete hosted family";
    }
    if (hosted) EXPECT_GT(hosted_exports(), 1.0) << "Both setup and generation must request the explicit hosted export";
    else EXPECT_EQ(hosted_exports(), 0.0);
    runner.shutdown();
}
INSTANTIATE_TEST_SUITE_P(Backends, HostedOrdinaryForward, ::testing::ValuesIn(hostedBackends()),
    [](const auto &info) { return info.param; });
}
}
