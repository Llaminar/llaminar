/**
 * @file Test__MoEPipelineWeightPreparation.cpp
 * @brief Real GGUF preparation through the PP factory and its admitted memory authority.
 *
 * A tiny generated model reaches the production loading, packing and graph
 * initialization paths. Sparse zero payloads test ownership, never model
 * quality. Planning alone cannot detect generic preparation incorrectly
 * charging routed experts to PrimaryModelWeights. Every participant must load
 * only its own PP layers and logical expert slices, including terminal MTP.
 * Initialization must also materialize the captured maintenance family before
 * checking readiness; a prepared weight registry alone is not a serving graph.
 * A reusable context must publish completion and exact retirement ownership
 * for every inner TP participant, including the non-primary pipeline stages.
 * Explicitly unsupported singleton-MoE and homogeneous nested-TP domains are
 * negative controls: their fail-closed diagnostics are not inference coverage.
 * Request regressions additionally enter real sparse generation after setup,
 * reset and prefix reuse. Every stage must receive the root's request identity;
 * successful weight preparation alone cannot prove that lifecycle publication.
 */
#include "../../../utils/PlanningGGUFFixture.h"
#include "PipelineGenerationTestSupport.h"
#include "backends/BackendManager.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEPipelinePreparedPlan.h"
#include "execution/runner/OrchestrationRunner.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/PerfStatsCollector.h"
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace llaminar2::test
{
namespace
{
    /** @brief Explicit serving policy, independent of the generated test name. */
    enum class MTPLane { Disabled, Fixed, Automatic };

    /** @brief Serving support is explicit, never inferred from a test result. */
    enum class PreparationOutcome { NativeServing, RejectSingletonController, RejectHomogeneousDomains };

    /** @brief Two disjoint native TP domains and one selected MTP policy. */
    struct PreparationCase
    {
        bool first_cuda;
        bool last_cuda;
        int width;
        MTPLane lane;
        PreparationOutcome outcome;
        /** @return Stable CTest/GTest identity for the exact physical topology. */
        std::string name() const
        {
            const auto domain = [this](bool cuda) { return std::string(cuda ? "CUDA" : "ROCm") + std::to_string(width); };
            return domain(first_cuda) + "_" + domain(last_cuda) + "_" +
                (lane == MTPLane::Disabled ? "DisabledMTP" : lane == MTPLane::Fixed ? "FixedMTP" : "DynamicMTP");
        }
    };

    /** @return Only topologies whose native backends were compiled. */
    std::vector<PreparationCase> cases()
    {
        std::vector<PreparationCase> result;
        for (const auto lane : {MTPLane::Disabled, MTPLane::Fixed, MTPLane::Automatic})
        {
#ifdef HAVE_CUDA
            result.push_back({true, true, 1, lane, PreparationOutcome::RejectSingletonController});
#endif
#ifdef HAVE_ROCM
            result.push_back({false, false, 1, lane, PreparationOutcome::RejectSingletonController});
            result.push_back({false, false, 2, lane, PreparationOutcome::RejectHomogeneousDomains});
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            for (const int width : {2, 4})
            {
                result.push_back({false, true, width, lane, PreparationOutcome::NativeServing});
                result.push_back({true, false, width, lane, PreparationOutcome::NativeServing});
            }
#endif
        }
        return result;
    }

    /** @return The exact two-stage configuration shared by loading and request tests. */
    OrchestrationConfig configuration(const PreparationCase &test, const std::string &model)
    {
        std::vector<std::string> arguments{"moe-pipeline-preparation", "-m", model,
            "--mpi-procs", "1", "--tp-scope", "rank_local", "--context-length", "512"};
        for (int stage = 0; stage < 2; ++stage)
        {
            const bool cuda = stage ? test.last_cuda : test.first_cuda;
            std::string domain = "stage" + std::to_string(stage) + "=";
            for (int participant = 0; participant < test.width; ++participant)
            {
                const int ordinal = participant + (stage && test.first_cuda == test.last_cuda ? test.width : 0);
                domain += (participant ? "," : "") + std::string(cuda ? "cuda:" : "rocm:") + std::to_string(ordinal);
            }
            domain += ";scope=rank_local;backend=" + std::string(cuda ? "nccl" : "rccl") + ";owner=0";
            arguments.insert(arguments.end(), {"--define-domain", domain, "--pp-stage",
                std::to_string(stage) + "=stage" + std::to_string(stage) + ":" + std::to_string(stage) + "-" + std::to_string(stage)});
        }
        if (test.lane == MTPLane::Disabled) arguments.push_back("--no-mtp");
        if (test.lane == MTPLane::Fixed)
            arguments.insert(arguments.end(), {"--mtp-depth-policy", "fixed", "--mtp-draft-tokens", "3"});
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
        config.prefill_max_bucket_size = 64;
        return config;
    }

    /** @brief Complete model initialization, not an admission-only substitute. */
    class MoEPipelineWeightPreparation : public ::testing::TestWithParam<PreparationCase>
    {
    protected:
        /** @brief Require actual devices; only the explicit wider cases may skip. */
        void SetUp() override
        {
            const auto test = GetParam();
            const int cuda_count = (int(test.first_cuda) + int(test.last_cuda)) * test.width;
            const int rocm_count = (int(!test.first_cuda) + int(!test.last_cuda)) * test.width;
            for (const auto [cuda, required] : {std::pair{true, cuda_count}, std::pair{false, rocm_count}})
            {
                if (!required) continue;
                auto *backend = cuda ? getCUDABackend() : getROCmBackend();
                ASSERT_NE(backend, nullptr);
                if (required >= 4 && backend->deviceCount() < required)
                    GTEST_SKIP() << "Requires " << required << (cuda ? " CUDA" : " ROCm") << " devices";
                ASSERT_GE(backend->deviceCount(), required);
            }
        }
    };

    TEST_P(MoEPipelineWeightPreparation, NativePreparationContract)
    {
        const auto test = GetParam();
        PlanningGGUFFixture fixture(true, true, GGUFTensorType::Q4_0, 256,
            std::nullopt, 8, PlanningGGUFFixture::Geometry::FullAttentionGraph);
        auto config = configuration(test, fixture.path());
        std::vector<std::vector<DeviceId>> devices(2);
        for (int stage = 0; stage < 2; ++stage)
        {
            const bool cuda = stage ? test.last_cuda : test.first_cuda;
            for (int participant = 0; participant < test.width; ++participant)
            {
                const int ordinal = participant + (stage && test.first_cuda == test.last_cuda ? test.width : 0);
                devices[stage].push_back(cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal));
            }
        }
        publishOrchestrationStartupPolicy(config);
        OrchestrationRunner runner(config, createExecutionPlanBuilder());
        const bool initialized = runner.initialize();
        if (test.outcome != PreparationOutcome::NativeServing)
        {
            ASSERT_FALSE(initialized) << "Unsupported topology must never select a different execution path";
            const std::string expected = test.outcome == PreparationOutcome::RejectSingletonController
                ? "Local MoE pipeline stages require native GPU controller domains"
                : "ExpertOverlay maintenance initialization step threw: 'materializePipelineServingGraphs': "
                  "Nested homogeneous pipeline domains require native domain-to-domain composition, not a heterogeneous channel";
            EXPECT_EQ(runner.lastError(), expected);
            EXPECT_FALSE(runner.isInitialized());
            EXPECT_FALSE(runner.modelContextReuseContract().has_value());
            runner.shutdown();
            return;
        }
        ASSERT_TRUE(initialized) << runner.lastError();
        ASSERT_TRUE(runner.isInitialized());
        EXPECT_EQ(runner.executionPlan().runtime.mtp.enabled, test.lane != MTPLane::Disabled);
        if (test.lane != MTPLane::Disabled)
            EXPECT_EQ(runner.executionPlan().runtime.mtp.depth_policy.mode,
                test.lane == MTPLane::Fixed ? MTPDepthPolicyMode::Fixed : MTPDepthPolicyMode::Dynamic);
        const auto contract = runner.modelContextReuseContract();
        ASSERT_TRUE(contract.has_value());
        ASSERT_NE(contract->pipeline_prepared_plan, nullptr);
        ASSERT_NE(contract->physical_memory_authority, nullptr);
        const auto manager = contract->context->concreteWeightManager();
        ASSERT_NE(manager, nullptr);
        EXPECT_TRUE(manager->lifecycleGates().graph_materialization_complete);
        EXPECT_TRUE(manager->lifecycleGates().canReleaseHostData());
        EXPECT_EQ(manager->preparedWeightStore()->physicalMemoryAuthority(), contract->physical_memory_authority);
        const auto &registry = manager->expertGemmRegistry();
        const auto &stages = contract->pipeline_prepared_plan->stages();
        ASSERT_EQ(stages.size(), 2u);
        for (size_t stage = 0; stage < stages.size(); ++stage)
        {
            const auto runtime = resolveMoEExpertOverlayRuntimePlan(stages[stage].placement);
            ASSERT_NE(runtime, nullptr);
            const auto preparation = MoEExpertOverlayPreparationPlan::build(*runtime, contract->context->concreteLoader());
            std::set<int> prepared_layers;
            std::map<std::tuple<std::string, DeviceId, int>, size_t> expected_entries;
            for (const auto &request : preparation.requests())
            {
                prepared_layers.insert(request.layer);
                ASSERT_TRUE(request.participant_world_rank_known);
                ASSERT_EQ(request.participant_world_rank, 0);
                // A participant key and its domain alias retain the same
                // prepared engine. Neither is a second physical allocation.
                expected_entries[{request.domain_name, request.device, request.layer}] += 2;
                EXPECT_NE(registry.getEngineForParticipant(request.domain_name, request.device,
                    request.participant_world_rank, request.participant_index, request.layer,
                    request.expert_id, request.role, request.projection_ownership), nullptr)
                    << request.domain_name << " layer=" << request.layer << " expert=" << request.expert_id;
            }
            const std::set<int> expected_layers = stage == 0 ? std::set<int>{0}
                : test.lane == MTPLane::Disabled ? std::set<int>{1} : std::set<int>{1, 2};
            EXPECT_EQ(prepared_layers, expected_layers);
            for (const auto &[key, count] : expected_entries)
            {
                const auto &[domain, device, layer] = key;
                EXPECT_EQ(registry.countEnginesForLayerInDomain(domain, device, layer), count);
            }
            for (const auto device : devices[stage])
            {
                const auto &authority = *contract->physical_memory_authority;
                const auto live = authority.claimedBytes(device, PhysicalMemoryOwner::RoutedExpertWeights,
                    PhysicalMemoryMaterializationKind::NewAllocation);
                EXPECT_GT(live, 0u) << device.toString();
                EXPECT_LE(live, authority.plannedBytes(device, PhysicalMemoryOwner::RoutedExpertWeights));
                for (int layer = 0; layer < 3; ++layer)
                    if (!expected_layers.contains(layer))
                        for (const auto &domain : runtime->domains())
                            EXPECT_EQ(registry.countEnginesForLayerInDomain(domain.name, device, layer), 0u);
                // The overlay is the only routed owner. A legacy generic
                // preparation must not publish another full expert bank.
                EXPECT_EQ(registry.countEnginesForDevice(device), 0u);
            }
        }
        runner.shutdown();
        EXPECT_FALSE(runner.isInitialized());
        ASSERT_NE(contract->reuse_authority, nullptr);
        std::string retention_error;
        const auto retention = contract->reuse_authority->sealedDeviceMemoryRetention(&retention_error);
        ASSERT_TRUE(retention.has_value()) << retention_error;
        std::set<DeviceId> expected_devices, retained_devices;
        for (const auto &stage : devices)
            expected_devices.insert(stage.begin(), stage.end());
        for (const auto &row : *retention)
        {
            EXPECT_TRUE(retained_devices.insert(row.device).second);
            EXPECT_GT(row.prepared_weight_bytes, 0u);
            EXPECT_EQ(row.prepared_weight_bytes, manager->retainedPreparedDeviceBytes(row.device));
        }
        EXPECT_EQ(retained_devices, expected_devices);
    }

    INSTANTIATE_TEST_SUITE_P(Topologies, MoEPipelineWeightPreparation,
        ::testing::ValuesIn(cases()), [](const auto &info) { return info.param.name(); });

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
    /** @brief Public sparse request lifecycle on each supported pipeline placement. */
    class MoEPipelineRequestLifecycle : public MoEPipelineWeightPreparation {};

    TEST_P(MoEPipelineRequestLifecycle, FirstDecodeResetAndPrefixReuse)
    {
        const auto test = GetParam();
        PlanningGGUFFixture fixture(true, true, GGUFTensorType::Q4_0, 256,
            std::nullopt, 8, PlanningGGUFFixture::Geometry::FullAttentionGraph);
        auto config = configuration(test, fixture.path());
        config.prefix_cache.ram_budget_bytes = 64ull * 1024 * 1024;
        config.prefix_cache.disk_budget_bytes = 0;
        publishOrchestrationStartupPolicy(config);
        OrdinaryGraphEvidence evidence("generation,forward_graph,mtp,prefix_cache");
        OrchestrationRunner runner(config, createExecutionPlanBuilder());
        ASSERT_TRUE(runner.initialize()) << runner.lastError();

        const auto publications = []
        {
            double count = 0;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "forward_graph" &&
                    record.name == "moe_overlay_collective_request_generation")
                    count += record.value;
            return count;
        };
        // Check the lifecycle boundary before any collective launch. The old
        // single-plan-only publisher returned success while every PP child
        // retained generation zero; its first hosted draft then failed fatally.
        ASSERT_EQ(publications(), 1.0);
        SamplingParams sampling;
        sampling.temperature = 0.8F;
        sampling.top_k = 20;
        sampling.seed = 123;
        runner.setSamplingParams(sampling);
        runner.setStopTokens({});
        std::vector<int32_t> prompt(96);
        for (size_t i = 0; i < prompt.size(); ++i)
            prompt[i] = 11 + static_cast<int32_t>(i);
        for (int request = 0; request < 3; ++request)
        {
            SCOPED_TRACE(::testing::Message() << "request=" << request);
            if (request != 0)
            {
                const double before = publications();
                runner.clearCache();
                ASSERT_EQ(publications(), before + 1);
            }
            if (request == 2) prompt.push_back(137);
            ASSERT_TRUE(runner.prefill(prompt)) << runner.lastError();
            for (const int budget : {1, 17, 1})
            {
                SCOPED_TRACE(::testing::Message() << "budget=" << budget);
                runner.setDecodeStepTokenBudget(budget);
                const auto result = runner.decodeStep();
                ASSERT_TRUE(result.success()) << result.error;
                ASSERT_EQ(result.tokens.size(), static_cast<size_t>(budget));
                for (const int32_t token : result.tokens)
                    EXPECT_GE(token, 0);
                // The serving frontend retires each completed response into
                // maintenance policy before admitting the next outer step.
                ASSERT_TRUE(runner.maybeApplyMoERebalance(result.tokens.size())) << runner.lastError();
            }
            const auto summary = runner.requestRuntimeSummary();
            ASSERT_TRUE(summary.prefix_request.enabled);
            ASSERT_FALSE(summary.prefix_request.bypassed) << summary.prefix_request.bypass_reason;
            if (request != 0)
                EXPECT_GT(summary.prefix_request.matched_tokens, 0);
            EXPECT_EQ(summary.mtp_request.enabled, test.lane != MTPLane::Disabled);
            if (test.lane != MTPLane::Disabled)
            {
                EXPECT_GT(summary.mtp_verifier_runs, 0u);
                EXPECT_GT(summary.mtp_request.draft_steps, 0u);
                EXPECT_FALSE(summary.mtp_request.bypassed) << summary.mtp_request.bypass_reason;
                EXPECT_EQ(summary.mtp_request.depth_policy_mode,
                    test.lane == MTPLane::Fixed ? "fixed" : "dynamic");
            }
        }
        const auto records = PerfStatsCollector::snapshot();
        double bridges = 0;
        for (const auto &record : records)
            if (record.domain == "forward_graph" && record.name == "pipeline_scalar_input_transactions")
                bridges += record.value;
        EXPECT_EQ(bridges, 1) << "The one-token suffix must traverse the captured pipeline in every MTP mode";
        if (test.lane != MTPLane::Disabled)
        {
            // Every upstream TP participant must execute a complete retained
            // transaction after the first externally scheduled verifier. A
            // terminal-only lease leaves these routers with an empty ticket.
            // Reset/prefix restore changes data, never this graph topology.
            for (int member = 0; member < test.width; ++member)
            {
                const auto device = (test.first_cuda ? DeviceId::cuda(member) : DeviceId::rocm(member)).toString();
                const auto count = [&](const char *name)
                {
                    double value = 0;
                    for (const auto &record : records)
                        if (record.domain == "generation" && record.name == name && record.device == device)
                            value += record.value;
                    return value;
                };
                EXPECT_GT(count("pipeline_mtp_follower_transaction_submissions"), 0) << device;
                EXPECT_GE(count("pipeline_mtp_follower_graph_reuses"), 2) << device;
                EXPECT_EQ(count("pipeline_mtp_follower_graph_materializations"), 1) << device;
            }
        }
        runner.shutdown();
        EXPECT_FALSE(runner.isInitialized());
    }

    /** @return Supported four/eight-device cells only; rejection cases remain above. */
    std::vector<PreparationCase> requestCases()
    {
        auto result = cases();
        std::erase_if(result, [](const auto &value)
            { return value.outcome != PreparationOutcome::NativeServing; });
        return result;
    }

    INSTANTIATE_TEST_SUITE_P(Topologies, MoEPipelineRequestLifecycle,
        ::testing::ValuesIn(requestCases()), [](const auto &info) { return info.param.name(); });
#endif
}
}
