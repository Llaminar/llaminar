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
 * Singleton stages retain their exact owner without migration work. Unsupported
 * homogeneous nested-TP domains remain fail-closed transport negative controls.
 * Request regressions additionally enter real sparse generation after setup,
 * reset and prefix reuse. Every stage must receive the root's request identity;
 * successful weight preparation alone cannot prove that lifecycle publication.
 */
#include "../../../utils/PlanningGGUFFixture.h"
#include "../../../utils/ServingGGUFFixturePayload.h"
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
#include "utils/KernelProfiler.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "execution/moe/MoEOverlayNoPeerResidency.h"
#include <gtest/gtest.h>
#include <charconv>
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
    enum class PreparationOutcome { NativeServing, RejectHomogeneousDomains };

    /**
     * @brief Authenticate an actual speculative native-parent launch observation.
     * @param record Existing DGO counter recorded after successful launch and terminal publication.
     * @param participant Exact upstream device whose optimized execution must be proved.
     * @return False for ordinary depth0, foreign devices/backends, hosted or malformed records.
     */
    bool isNativeSpeculativeParentLaunch(const PerfStatRecord &record,DeviceId participant)
    {
        if(!participant.is_gpu() || record.kind!=PerfStatRecord::Kind::Counter ||
            record.domain!="mtp" || record.name!="device_generation_loop_graph_launches" ||
            record.device!=participant.toString())return false;
        const auto backend=record.tags.find("backend"),depth=record.tags.find("draft_depth"),
            execution=record.tags.find("execution");
        if(backend==record.tags.end() || depth==record.tags.end() || execution==record.tags.end() ||
            backend->second!=(participant.is_cuda()?"CUDA":"HIP") ||
            (execution->second!="single_async_native_while_launch" &&
             execution->second!="single_async_native_selector_while_launch"))return false;
        int value=0;const auto *end=depth->second.data()+depth->second.size();
        const auto parsed=std::from_chars(depth->second.data(),end,value);
        return parsed.ec==std::errc{} && parsed.ptr==end && value>0;
    }

    /** @test Ordinary, foreign and hosted work cannot certify a missing speculative native parent. */
    TEST(MoEPipelineGenerationEvidence, NativeSpeculativeLaunchRejectsOrdinaryForeignOrHostedRecords)
    {
        const auto device=DeviceId::cuda(0);
        const PerfStatRecord valid{.domain="mtp",.name="device_generation_loop_graph_launches",.phase="decode",
            .device=device.toString(),.tags={{"backend","CUDA"},{"draft_depth","3"},
                {"execution","single_async_native_while_launch"}},.value=1};
        EXPECT_TRUE(isNativeSpeculativeParentLaunch(valid,device));
        auto selector=valid;selector.tags["execution"]="single_async_native_selector_while_launch";
        EXPECT_TRUE(isNativeSpeculativeParentLaunch(selector,device));
        for(const auto *depth:{"0","-1","3junk","","999999999999999999999999"})
        {auto invalid=valid;invalid.tags["draft_depth"]=depth;EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));}
        auto invalid=valid;invalid.tags.erase("draft_depth");EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.tags.erase("backend");EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.tags.erase("execution");EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.device=DeviceId::cuda(1).toString();EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.tags["backend"]="HIP";EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.tags["execution"]="hosted_captured_transactions_with_ticket_only_dispatch";
        EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.domain="generation";EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.kind=PerfStatRecord::Kind::Timer;EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        EXPECT_FALSE(isNativeSpeculativeParentLaunch(valid,DeviceId::cpu()));
    }

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
            result.push_back({true, true, 1, lane, PreparationOutcome::NativeServing});
#endif
#ifdef HAVE_ROCM
            result.push_back({false, false, 1, lane, PreparationOutcome::NativeServing});
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

    /** @brief Exact native stage order followed by a real host compute tail. */
    struct NoPeerHostTailCase
    {
        std::vector<DeviceType> stages;
        std::string name;
    };

    /** @brief Backend-conditional cells; every native member is physical ordinal zero. */
    std::vector<NoPeerHostTailCase> noPeerHostTailCases()
    {
        std::vector<NoPeerHostTailCase> result;
#ifdef HAVE_CUDA
        result.push_back({{DeviceType::CUDA, DeviceType::CPU}, "CUDA1_CPU1"});
#endif
#ifdef HAVE_ROCM
        result.push_back({{DeviceType::ROCm, DeviceType::CPU}, "ROCm1_CPU1"});
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        result.push_back({{DeviceType::CUDA, DeviceType::ROCm, DeviceType::CPU}, "CUDA1_ROCm1_CPU1"});
        result.push_back({{DeviceType::ROCm, DeviceType::CUDA, DeviceType::CPU}, "ROCm1_CUDA1_CPU1"});
#endif
        return result;
    }

    /** @brief Actual OR loading/admission and Rank host/native family composition. */
    class MoENoPeerHostTail : public ::testing::TestWithParam<NoPeerHostTailCase> {};

    TEST_P(MoENoPeerHostTail, DynamicOwnershipAndLiveActivationExtents)
    {
        const auto test = GetParam();
        // This single-rank fixture enters the runner below RuntimeInitPhase.
        // Publish its host allocation owner before any CPU workspace is bound,
        // just as production startup and the canonical parity fixture do.
        initCPUBackend(-1);
        for (const auto type : test.stages)
        {
            if (type == DeviceType::CPU) continue;
            auto *backend = type == DeviceType::CUDA ? getCUDABackend() : getROCmBackend();
            ASSERT_NE(backend, nullptr);
            ASSERT_GE(backend->deviceCount(), 1);
        }
        PlanningGGUFFixture fixture(true, false, GGUFTensorType::Q4_0, 256,
            std::nullopt, 8, PlanningGGUFFixture::Geometry::FullAttentionGraph, test.stages.size());
        // Unlike setup-only sparse files, this fixture executes verified CPU
        // layers. Publish real nonzero source values before model admission.
        populateServingGGUFFixture(fixture);
        std::vector<std::string> arguments{"no-peer-host-tail", "-m", fixture.path(), "--mpi-procs", "1",
            "--backend", "heterogeneous", "--context-length", "512", "--no-mtp",
            "--activation-precision", "fp32", "--kv-cache-precision", "fp16",
            "--moe-residency-maintenance", "dynamic"};
        for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
        {
            const auto kind = test.stages[stage] == DeviceType::CUDA ? "cuda" :
                test.stages[stage] == DeviceType::ROCm ? "rocm" : "cpu";
            const auto domain = "s" + std::to_string(stage);
            arguments.insert(arguments.end(), {"--define-domain",
                domain + "=" + kind + ":0;scope=rank_local;backend=auto;owner=0", "--pp-stage",
                std::to_string(stage) + "=" + domain + ":" + std::to_string(stage) + "-" + std::to_string(stage)});
        }
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
        config.prefill_max_bucket_size = 64;
        // Separate prefix archival traffic from activation-economy proof.
        // Prefix/reset ownership remains covered by the retained pipeline family.
        config.prefix_cache.enabled = false;
        publishOrchestrationStartupPolicy(config);
        OrdinaryGraphEvidence evidence("generation,forward_graph,moe_overlay_residency,transfer,request_admission");
        OrchestrationRunner runner(config, createExecutionPlanBuilder());
        ASSERT_TRUE(runner.initialize()) << runner.lastError();
        ASSERT_EQ(runner.primaryDeviceId(), DeviceId::cpu());
        EXPECT_EQ(runner.executionPlan().runtime.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
        const auto contract = runner.modelContextReuseContract();
        ASSERT_TRUE(contract);
        ASSERT_TRUE(contract->pipeline_prepared_plan);
        ASSERT_EQ(contract->pipeline_prepared_plan->stages().size(), test.stages.size());
        const auto topology = runner.moeOptimizationMovementTopology();
        ASSERT_EQ(topology.authority, MoEOptimizationAuthority::Pipeline);
        EXPECT_EQ(topology.axes, MoEOptimizationMovementAxes::None);
        ASSERT_EQ(topology.stages.entries().size(), test.stages.size());
        const auto check_owners = [&]
        {
            const auto status = runner.moeOptimizationStatus();
            ASSERT_EQ(status.authority, MoEOptimizationAuthority::Pipeline);
            ASSERT_EQ(status.stages.entries().size(), test.stages.size());
            for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
            {
                const auto &prepared = contract->pipeline_prepared_plan->stages()[stage];
                const auto owners = MoEExpertOwnerMap::build(*prepared.placement);
                const auto proof = MoEOverlayNoPeerResidency::seal(*prepared.placement, owners, prepared.metadata);
                ASSERT_TRUE(proof);
                const auto device = test.stages[stage] == DeviceType::CPU ? DeviceId::cpu() :
                    test.stages[stage] == DeviceType::CUDA ? DeviceId::cuda(0) : DeviceId::rocm(0);
                const auto &scope = topology.stages.entries()[stage].identity;
                ASSERT_EQ(scope.participants().size(), 1u);
                EXPECT_EQ(scope.participants().front().toLocalDeviceId(), device);
                EXPECT_EQ(scope.firstLayer(), stage);
                EXPECT_EQ(scope.mainLastLayer(), stage + 1u);
                EXPECT_EQ(scope.terminal(), stage + 1u == test.stages.size());
                EXPECT_EQ(proof->device(), device);
                EXPECT_EQ(prepared.placement->first_model_layer, stage);
                EXPECT_EQ(MoEOverlayCapacityAdmission::selectMigrationStorage(*prepared.placement,
                    MoERebalanceRuntimeMode::Dynamic), MoEOverlayMigrationStorageKind::NoPeer);
                const auto &value = status.stages.entries()[stage].value;
                EXPECT_EQ(value.authority, device.is_gpu() ? MoEOptimizationAuthority::Device : MoEOptimizationAuthority::Host);
                EXPECT_EQ(value.state, MoEOptimizationLifecycleState::MovementDisabled);
                EXPECT_EQ(value.completed_movement.physical_bytes, 0u);
                EXPECT_EQ(value.completed_movement.commands, 0u);
                EXPECT_EQ(value.completed_decision_windows, 0u);
                EXPECT_EQ(contract->physical_memory_authority->plannedBytes(device,
                    PhysicalMemoryOwner::ExpertMigrationStaging), 0u);
            }
        };
        check_owners();
        SamplingParams sampling;
        sampling.temperature = 0.0F;
        sampling.seed = 123;
        runner.setSamplingParams(sampling);
        runner.setStopTokens({});
        const auto capture_count = [] {
            double value = 0;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "forward_graph" && record.name == "decode_graph_phase" &&
                    record.tags.contains("phase") && record.tags.at("phase") == "capture")
                    value += record.value;
            return value;
        };
        // These terminal observations prove participant-local ingress; no
        // execution decision reads them and no device state is mirrored here.
        const auto decode_ingress_bytes = [](DeviceId device, const char *name) {
            double value = 0;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "request_admission" && record.name == name &&
                    record.phase == "decode" && record.device == device.toString())
                    value += record.value;
            return value;
        };
        const auto original_captures = capture_count();
        EXPECT_GT(original_captures, 0.0) << "Initialization must materialize actual native stages";
        const auto gpu_stages = test.stages.size() - 1u;
        for (const int rows : {65, 3, 1})
        {
            SCOPED_TRACE(rows);
            runner.clearCache();
            std::vector<int32_t> tokens(static_cast<std::size_t>(rows), 11);
            TransferProfiler::reset();
            ASSERT_TRUE(runner.prefill(tokens)) << runner.lastError();
            // This model has H256. CPU owns the final sampler, so every D2H
            // byte in this diagnostic-free/no-prefix call is a real boundary
            // activation. Each GPU boundary must copy only root live rows,
            // including the final partial64+1 and large-to-small request reset.
            const auto expected = static_cast<std::uint64_t>(gpu_stages) * rows * 256u * sizeof(float);
            EXPECT_EQ(TransferProfiler::getD2HStats().total_bytes.load(), expected);
            std::vector<std::tuple<DeviceId, double, double, double>> ingress_before;
            for (const auto type : test.stages)
                if (type != DeviceType::CPU)
                {
                    const auto device = type == DeviceType::CUDA ? DeviceId::cuda(0) : DeviceId::rocm(0);
                    ingress_before.emplace_back(device,
                        decode_ingress_bytes(device, "device_input_token_bytes"),
                        decode_ingress_bytes(device, "device_input_position_bytes"),
                        decode_ingress_bytes(device, "device_input_padding_bytes"));
                }
            // The CPU-tail serial API emits one token per decodeStep; its
            // budget is an upper bound, not a request to batch scalar calls.
            // Retire each actual sample before admitting the next command,
            // matching generate(). The first call consumes prefill logits;
            // only the second consumes one true history-bearing pipeline row.
            for (const int step : {0, 1})
            {
                SCOPED_TRACE(step);
                runner.setDecodeStepTokenBudget(1);
                const auto result = runner.decodeStep();
                ASSERT_TRUE(result.success()) << result.error;
                ASSERT_EQ(result.tokens.size(), 1u);
                ASSERT_FALSE(result.is_complete);
                const auto history_bytes = static_cast<std::uint64_t>(step) *
                    gpu_stages * 256u * sizeof(float);
                EXPECT_EQ(TransferProfiler::getD2HStats().total_bytes.load(),
                          expected + history_bytes)
                    << "Only the history call transfers one live activation row per GPU boundary";
                // Every GPU, including a non-embedding middle participant,
                // receives exactly the live scalar token on the history call.
                // Its captured KV root authors positions; no position shadow
                // or padded token capacity crosses host admission.
                for (const auto &[device, token_bytes, position_bytes, padding_bytes] : ingress_before)
                {
                    EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_token_bytes") - token_bytes,
                                     static_cast<double>(step * sizeof(int32_t)));
                    EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_position_bytes"), position_bytes);
                    EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_padding_bytes"), padding_bytes);
                }
                ASSERT_TRUE(runner.maybeApplyMoERebalance(result.tokens.size())) << runner.lastError();
            }
            for (const auto type : test.stages)
                if (type != DeviceType::CPU)
                    EXPECT_GT(evidence.completedForwards(type == DeviceType::CUDA ?
                        DeviceId::cuda(0) : DeviceId::rocm(0)), 0.0);
            EXPECT_EQ(capture_count(), original_captures) << "Reset must preserve the retained native graph family";
            check_owners();
        }
        runner.clearCache();
        TransferProfiler::reset();
        EXPECT_FALSE(runner.prefill({})) << "Empty input is rejected before a PP activation transfer";
        EXPECT_EQ(TransferProfiler::getD2HStats().total_bytes.load(), 0u);
        EXPECT_EQ(TransferProfiler::getH2DStats().total_bytes.load(), 0u);
        check_owners();
        runner.shutdown();
    }

    INSTANTIATE_TEST_SUITE_P(Topologies, MoENoPeerHostTail,
        ::testing::ValuesIn(noPeerHostTailCases()), [](const auto &info) { return info.param.name; });

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
            const std::string expected =
                "ExpertOverlay maintenance initialization step threw: 'materializePipelineServingGraphs': "
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
        if (test.lane == MTPLane::Disabled && test.width == 1 && test.first_cuda == test.last_cuda)
        {
            double native_tail_builds = 0;
            for (const auto &record : records)
                if (record.domain == "generation" &&
                    record.name == "ordinary_native_pipeline_nopeer_tail_materializations")
                    native_tail_builds += record.value;
            EXPECT_GT(native_tail_builds, 0)
                << "The real native PP parent must import the singleton overlay tail's sampler and forward lease";
        }
        if (test.lane != MTPLane::Disabled)
        {
            // Every upstream TP participant must execute its complete admitted
            // native parent or hosted transaction after the first verifier. A
            // terminal-only lease omits required follower verification/commit.
            // Reset/prefix restore changes data, never this graph topology.
            for (int member = 0; member < test.width; ++member)
            {
                const auto participant=test.first_cuda?DeviceId::cuda(member):DeviceId::rocm(member);
                const auto device=participant.toString();
                // Only the supported homogeneous singleton CUDA topology owns
                // one native conditional PP parent. HIP and explicit mixed
                // boundaries retain their authenticated hosted transactions.
                const auto policy=test.first_cuda && test.last_cuda && test.width==1
                    ? DeviceGenerationExecutionPolicy::NativeConditionalGraph
                    : DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
                const auto count = [&](const char *name,const char *domain="generation")
                {
                    double value = 0;
                    for (const auto &record : records)
                        if (record.domain == domain && record.name == name && record.device == device)
                            value += record.value;
                    return value;
                };
                double native_launches=0,materializations_for_policy=0;
                for(const auto &record:records)
                {
                    if(isNativeSpeculativeParentLaunch(record,participant))native_launches+=record.value;
                    if(record.domain=="generation" && record.name=="pipeline_mtp_follower_graph_materializations" &&
                        record.device==device)
                        if(const auto execution=record.tags.find("execution");execution!=record.tags.end() &&
                            execution->second==deviceGenerationExecutionPolicyName(policy))
                            materializations_for_policy+=record.value;
                }
                if(policy==DeviceGenerationExecutionPolicy::NativeConditionalGraph)
                {
                    EXPECT_GT(native_launches,0) << device;
                    EXPECT_EQ(count("pipeline_mtp_follower_transaction_submissions"),0) << device;
                    EXPECT_EQ(count("device_generation_dispatch_tickets_observed","mtp"),0) << device;
                }
                else
                {
                    EXPECT_GT(count("pipeline_mtp_follower_transaction_submissions"),0) << device;
                    EXPECT_EQ(native_launches,0) << device;
                }
                EXPECT_GE(count("pipeline_mtp_follower_graph_reuses"), 2) << device;
                EXPECT_EQ(count("pipeline_mtp_follower_graph_materializations"), 1) << device;
                EXPECT_EQ(materializations_for_policy,1) << device;
            }
        }
        runner.shutdown();
        EXPECT_FALSE(runner.isInitialized());
    }

    /** @return Supported singleton/four/eight-device cells; rejection cases remain above. */
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
