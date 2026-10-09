/**
 * @file Test__PipelineDomainGeneration.cpp
 * @brief Real captured inter-vendor generation with native TP inside each domain.
 *
 * The tiny numerical graph uses production embedding, KV and sampling kernels.
 * Its independent four-token distribution detects bad metadata/activation
 * handoffs without loading a model. Public request APIs exercise first admission,
 * continuation and reset. Main forwards must be complete before serving;
 * request-specific helper captures must survive reset. This is model-free preflight.
 */
#include "PipelineGenerationTestSupport.h"
#include "config/OrchestrationStartupPolicy.h"
#include <map>
#include <tuple>

namespace llaminar2::test
{
/** @brief Each serving policy is selected explicitly and retains its own evidence. */
enum class PipelineGenerationLane { Ordinary, FixedMTP, DynamicMTP };

/** @brief Explicit physical topology and generation lane, never inferred from a name. */
struct PipelineDomainCase
{
    bool reverse;
    int width;
    PipelineGenerationLane lane;
    /** @return Whether this case executes speculative verification. */
    bool usesMTP() const noexcept { return lane != PipelineGenerationLane::Ordinary; }
    int admitted_rows = 256; ///< Exact PMA-selected arena capacity, independent of startup ceiling.
};

/** @brief Shared numerical fixture with explicit one-, two- and four-member TP domains. */
class PipelineDomainGeneration : public ::testing::TestWithParam<PipelineDomainCase>
{
protected:
    /** @brief Separate model forwards from the request-specific helper family. */
    enum class CaptureScope { MainForward, CompleteGeneration };

    /**
     * @brief Count nodes at the native forward-capture boundary for each device.
     * @param scope Main-model serving family or all generation helpers as well.
     * @return Monotonic capture totals, including retained child templates.
     *
     * Setup's semantic family counter alone cannot expose a request-time cache
     * miss. These counters are emitted by the native capture controller after
     * recording, so any later recording changes the observed inventory.
     */
    static std::map<std::string, double> capturedForwardNodes(
        CaptureScope scope = CaptureScope::MainForward)
    {
        std::map<std::string, double> result;
        for (const auto &record : PerfStatsCollector::snapshot())
        {
            const auto context = record.tags.find("context");
            if (scope == CaptureScope::MainForward &&
                (context == record.tags.end() ||
                 (!context->second.starts_with("main_") && !context->second.starts_with("prefill"))))
                continue;
            if (record.domain == "forward_graph" &&
                (record.name == "full_graph_capture_executable_nodes" ||
                 record.name == "segmented_graph_capture_executable_nodes" ||
                 record.name == "retained_parent_child_graph_nodes"))
                result[record.device] += record.value;
        }
        return result;
    }

    /**
     * @brief Require both verifier outcomes at every admitted request width.
     * @param participants Actual GPU owners whose setup has completed.
     * @param enabled Whether this case retains the depth-15 MTP family.
     */
    static void expectRetainedVerifierSetup(
        const std::vector<DeviceGraphOrchestrator *> &participants, bool enabled)
    {
        using Key = std::tuple<std::string, std::string, std::string>;
        std::map<Key, double> expected, actual;
        if (enabled)
            for (const auto *participant : participants)
                for (const int rows : {2, 4, 8, 16})
                    for (const auto outcome : {MTPVerifierOutcomeGraphMode::Greedy,
                                               MTPVerifierOutcomeGraphMode::Disabled})
                        expected[{participant->primaryDeviceId().toString(), std::to_string(rows),
                            std::to_string(static_cast<int>(outcome))}] = 1;
        for (const auto &record : PerfStatsCollector::snapshot())
        {
            if (record.domain != "forward_graph" || record.name != "serving_graph_family_materializations")
                continue;
            const auto role = record.tags.find("role");
            if (role == record.tags.end() || role->second != "mtp_grouped_verifier") continue;
            EXPECT_EQ(record.phase, "setup");
            EXPECT_EQ(record.tags.at("submission"), "materialized_unlaunched");
            actual[{record.device, record.tags.at("physical_rows"), record.tags.at("outcome_mode")}] += record.value;
        }
        EXPECT_EQ(actual, expected) << "All request widths must exist before the first prompt is admitted.";
    }

    /** @brief Execute exact public requests after admitting all original physical owners. */
    void run()
    {
        const auto test = GetParam();
        initCPUBackend(-1);
        auto *cuda = getCUDABackend();
        auto *rocm = getROCmBackend();
        ASSERT_NE(cuda, nullptr);
        ASSERT_NE(rocm, nullptr);
        // Four total GPUs are the required hybrid preflight baseline. The
        // explicitly registered eight-GPU extension reports missing hardware
        // as skipped; it must never masquerade as a successful execution.
        if (test.width == 4 && (cuda->deviceCount() < test.width || rocm->deviceCount() < test.width))
            GTEST_SKIP() << "Requires " << test.width << " CUDA and ROCm devices";
        ASSERT_GE(cuda->deviceCount(), test.width) << "Required hybrid preflight CUDA participants";
        ASSERT_GE(rocm->deviceCount(), test.width) << "Required hybrid preflight ROCm participants";
        struct Startup
        {
            std::optional<std::string> previous;
            /** @brief Publish the same bucket policy as the frontend. */
            Startup()
            {
                if (const char *value = std::getenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES")) previous = value;
                OrchestrationConfig config;
                config.prefill_max_bucket_size = 256;
                publishOrchestrationStartupPolicy(config);
            }
            /** @brief Restore policy after every graph and worker has retired. */
            ~Startup()
            {
                if (previous) setenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", previous->c_str(), 1);
                else unsetenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES");
                mutableDebugEnv().reload();
            }
        } startup;
        MTPRuntimeConfig mtp;
        mtp.enabled = test.usesMTP();
        mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
        if (test.usesMTP())
        {
            mtp.graph_capacity_draft_tokens = 15;
            // Fixed execution deliberately retains the same depth-15 graph
            // capacity as dynamic execution; seven is its active depth only.
            mtp.draft_tokens = 7;
            mtp.depth_policy.mode = test.lane == PipelineGenerationLane::DynamicMTP
                ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed;
            mtp.depth_policy.initial_depth = test.lane == PipelineGenerationLane::DynamicMTP ? 15 : 7;
            mtp.depth_policy.max_depth = 15;
            mtp.depth_policy.window_size = 1;
            mtp.depth_policy.min_samples = 1;
            mtp.depth_policy.cooldown_steps = 0;
        }
        OrdinaryGraphEvidence evidence("generation,forward_graph,mtp");
        std::vector<std::vector<GlobalDeviceAddress>> members(2);
        std::vector<std::unique_ptr<LocalTPContext>> contexts(2);
        std::vector<DevicePlanConfig> inputs;
        for (int domain = 0; domain < 2; ++domain)
        {
            const bool is_cuda = bool(domain) == test.reverse;
            for (int member = 0; member < test.width; ++member)
            {
                const int ordinal = test.width - 1 - member;
                const auto device = is_cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
                members[domain].push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
                DevicePlanConfig input;
                input.world_rank = 0;
                input.device = device;
                input.first_layer = input.last_layer = domain;
                auto *backend = getBackendFor(device);
                input.device_total_bytes = backend->deviceMemoryTotal(ordinal);
                input.device_free_bytes = backend->deviceMemoryFree(ordinal);
                input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
                    .total_bytes = 1ull << 30, .admission_available_bytes = 1ull << 30};
                input.device_compute_units = 128;
                input.max_seq_len = 512;
                input.activation_seq_len = test.admitted_rows;
                input.mtp_enabled = test.usesMTP();
                input.mtp_target_query_rows = test.usesMTP() ? 16 : 1;
                input.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory({test.admitted_rows}, mtp);
                if (test.width > 1) input.local_tp_backend = is_cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                if (!member) input.captured_pipeline_boundaries = {domain
                    ? PipelineBoundarySide::LaterDomain : PipelineBoundarySide::EarlierDomain};
                inputs.push_back(input);
            }
            if (test.width > 1)
                contexts[domain] = std::make_unique<LocalTPContext>(members[domain], std::vector<float>{}, CollectiveBackendType::AUTO);
        }
        ModelMemoryProfile profile;
        profile.architecture = "qwen35";
        profile.n_layers = test.usesMTP() ? 3 : 2;
        profile.mtp_layer_count = test.usesMTP() ? 1 : 0;
        profile.d_model = profile.head_dim = profile.vocab_size = 32;
        profile.d_ff = 64;
        profile.n_heads = profile.n_kv_heads = 1;
        profile.max_seq_len = 512;
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(MemoryPlanner::plan(profile, inputs).physicalPlan()), 0);
        std::vector<std::shared_ptr<OrdinaryDGOGraph>> graphs;
        std::vector<DeviceGraphOrchestrator *> participants;
        std::vector<std::unique_ptr<IInferenceRunner>> domains;
        for (int domain = 0; domain < 2; ++domain)
        {
            if (contexts[domain]) ASSERT_TRUE(contexts[domain]->reserveGraphCaptureBoundaryResources(memory));
            std::vector<std::unique_ptr<IInferenceRunner>> children;
            for (int member = 0; member < test.width; ++member)
            {
                const auto device = members[domain][member].toLocalDeviceId();
                GraphConfig config;
                config.d_model = config.head_dim = config.vocab_size = config.vocab_local = 32;
                config.d_ff = config.d_ff_local = 64;
                config.n_heads = config.local_n_heads = config.n_kv_heads = config.local_n_kv_heads = 1;
                config.n_layers = 1;
                config.total_n_layers = profile.n_layers;
                config.pp_layer_offset = domain;
                config.layer_types.assign(profile.n_layers, "full_attention");
                // The shared hybrid schema declares both attention families,
                // even when this numerical probe only executes full attention.
                config.gdn.conv_kernel_size = 4;
                config.gdn.state_size = config.head_dim;
                config.gdn.inner_size = config.d_model;
                config.gdn.group_count = config.n_kv_heads;
                config.gdn.time_step_rank = config.n_heads;
                config.default_device = device;
                config.max_seq_len = 512;
                config.kv_cache_precision = KVCachePrecision::FP16;
                config.mtp = mtp;
                config.mtp_request_terminal_hidden_publication = MTPRequestTerminalHiddenPublicationPolicy::GraphCapturedDeviceGeometry;
                config.mtp_shifted_prefill_hidden_publication = MTPShiftedPrefillHiddenPublicationPolicy::GraphIntegratedKVTransaction;
                config.mtp_verifier_outcome_ownership = MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal;
                config.tp_ctx = contexts[domain].get();
                config.tp_device_idx = config.local_rank = member;
                config.lm_head_column_parallel = test.width > 1 && domain == 1;
                config.prefix_cache.enabled = false;
                config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
                auto graph = std::make_shared<OrdinaryDGOGraph>(config, domain == 0, domain == 1);
                auto owner = DeviceGraphOrchestrator::createForTest({.model_ctx = MockModelContext::createMinimal(),
                    .graph_builder = graph, .physical_memory_authority = memory});
                owner->setPPStageConfig({.first_layer = domain, .last_layer = domain + 1,
                    .has_embedding = domain == 0, .has_lm_head = domain == 1});
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                context.submitAndWait([&] {
                    ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
                    TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
                    ASSERT_TRUE(owner->initializeInferenceStateFromArena(1, 512, device,
                        {.activation_seq_len = test.admitted_rows}));
                });
                ASSERT_FALSE(HasFatalFailure());
                if (test.usesMTP() && domain == 1) owner->setFrozenWeightSet(makeOrdinaryMTPFrozenWeightSet(*graph, device));
                participants.push_back(owner.get());
                graphs.push_back(graph);
                children.push_back(std::move(owner));
            }
            if (test.width == 1) domains.push_back(std::move(children.front()));
            else
            {
                RankOrchestrator::Config config;
                config.mode = RankOrchestrator::ParallelismMode::TP;
                config.devices = members[domain];
                config.max_seq_len = 512;
                config.mtp = mtp;
                config.nested_pp_stage_config = FactoryPPStageConfig{.first_layer = domain, .last_layer = domain + 1,
                    .has_embedding = domain == 0, .has_lm_head = domain == 1};
                auto domain_model = MockModelContext::createMinimal();
                domain_model->setVocabSize(32);
                domains.push_back(RankOrchestrator::createForTest(std::move(domain_model),
                    std::move(children), std::move(contexts[domain]), config));
            }
        }
        RankOrchestrator::Config rank_config;
        rank_config.mode = RankOrchestrator::ParallelismMode::PP;
        rank_config.max_seq_len = 512;
        rank_config.mtp = mtp;
        for (int domain = 0; domain < 2; ++domain)
            rank_config.pp_stages.push_back({.first_layer = domain, .last_layer = domain + 1,
                .has_embedding = domain == 0, .has_lm_head = domain == 1, .stage_devices = members[domain]});
        auto model = MockModelContext::createMinimal();
        model->setVocabSize(32);
        auto rank = RankOrchestrator::createForTestWithPipelineStages(model, std::move(domains), rank_config);
        ASSERT_TRUE(rank->materializeServingGraphFamilyWithoutLaunch({.prefill_bucket_rows = {test.admitted_rows},
            .prefill_pad_token_id = 0, .main_decode_graph = ServingMainDecodeGraphKind::HistoryBearingSerial}));
        expectRetainedVerifierSetup(participants, test.usesMTP());
        const auto captured_before_requests = capturedForwardNodes();
        ASSERT_EQ(captured_before_requests.size(), participants.size());
        for (const auto *participant : participants)
            ASSERT_GT(captured_before_requests.at(participant->primaryDeviceId().toString()), 0);
        const auto geometry = PipelineTransferMemory::forRows(32, test.admitted_rows, test.usesMTP() ? 16 : 1);
        for (size_t i = 0; i < participants.size(); ++i)
            EXPECT_EQ(memory->claimedBytes(participants[i]->primaryDeviceId(), PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), i % test.width ? 0 : geometry.device_bytes_per_boundary);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), geometry.host_bytes_per_boundary);

        OrchestrationConfig config;
        config.max_seq_len = 512;
        config.prefill_max_bucket_size = test.admitted_rows;
        config.mtp = mtp;
        config.prefix_cache.enabled = false;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        RankExecutionPlan plan;
        plan.primary_device = members.back().front();
        plan.runtime.max_seq_len = 512;
        // The real admission path publishes this capacity into the root plan
        // as well as participant arenas. Keep the request scheduler on that
        // same authority while the startup bucket ceiling remains larger.
        plan.runtime.resident_graph_rows = test.admitted_rows;
        plan.runtime.mtp = mtp;
        plan.runtime.prefix_cache = config.prefix_cache;
        auto *pipeline = rank.get();
        OrchestrationRunner runner(config, plan, std::move(rank));
        int consumed_main_rows = 0;
        std::optional<std::map<std::string, double>> retained_generation_captures;
        for (int repetition = 0; repetition < 2; ++repetition)
        {
            runner.clearCache();
            SamplingParams law;
            law.temperature = 1.0F;
            law.top_k = 4;
            law.seed = 0xFE000001u + repetition;
            runner.setSamplingParams(law);
            runner.setStopTokens({});
            int previous = repetition * 7;
            std::vector<int> prompt(256, previous);
            ASSERT_TRUE(runner.prefill(prompt)) << runner.lastError();
            EXPECT_EQ(capturedForwardNodes(), captured_before_requests)
                << "Prefill must reuse the admitted native forward family.";
            if (retained_generation_captures)
                EXPECT_EQ(capturedForwardNodes(CaptureScope::CompleteGeneration), *retained_generation_captures)
                    << "Request reset must preserve every prior generation helper.";
            int emitted = 0;
            // Exercise the one-token boundary before a verifier has ever run,
            // as well as between speculative chunks. Otherwise a prior graph
            // output can accidentally mask missing scalar-state publication.
            for (const int budget : {1, 65, 1, 17})
            {
                runner.setDecodeStepTokenBudget(budget);
                const auto result = runner.decodeStep();
                ASSERT_TRUE(result.success()) << result.error;
                ASSERT_EQ(result.tokens.size(), size_t(budget));
                EXPECT_EQ(capturedForwardNodes(), captured_before_requests)
                    << "Decode must reuse the admitted native forward family: repetition="
                    << repetition << " budget=" << budget;
                if (retained_generation_captures)
                    EXPECT_EQ(capturedForwardNodes(CaptureScope::CompleteGeneration), *retained_generation_captures)
                        << "Repeated generation must reuse every retained helper.";
                for (const auto token : result.tokens)
                {
                    std::array<int, 4> support;
                    for (int i = 0; i < 4; ++i) support[i] = (previous + i + 1) % 32;
                    std::sort(support.begin(), support.end());
                    const auto draw = sampling_math::mtp_spec_threshold_from_seed(law.seed, 256 + emitted, 0);
                    previous = support[std::min(3, int(draw * 4))];
                    ASSERT_EQ(token, previous) << "repetition=" << repetition << " output_row=" << emitted;
                    ++emitted;
                }
            }
            // A token-only oracle cannot prove the follower wrote its own KV.
            // Inspect original cache payloads only after terminal completion,
            // so a diagnostic wait cannot hide an ingress ordering defect.
            int32_t cached = 0;
            auto &first = *participants.front();
            auto &context = GPUDeviceContextPool::instance().getContext(first.primaryDeviceId());
            context.submitAndWait([&] {
                ASSERT_TRUE(getBackendFor(first.primaryDeviceId())->deviceToHost(&cached,
                    first.inferenceState().kv_cache->deviceSequenceCachedTokenCountPtr(0), sizeof(cached),
                    first.primaryDeviceId().gpu_ordinal(), context.defaultStream()));
            });
            ASSERT_FALSE(HasFatalFailure());
            EXPECT_GE(cached, int(prompt.size()) + emitted - 1);
            expectOrdinaryPipelinePromptKV(*pipeline, prompt, cached, participants.size());
            consumed_main_rows += cached - int(prompt.size());
            retained_generation_captures = capturedForwardNodes(CaptureScope::CompleteGeneration);
        }
        const auto sum = [&](const char *domain, const char *name, const char *phase = nullptr) {
            double value = 0;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == domain && record.name == name && (!phase || record.phase == phase))
                    value += record.value;
            return value;
        };
        EXPECT_EQ(sum("forward_graph", "segmented_plan_segments"), 2);
        EXPECT_EQ(sum("forward_graph", "segmented_graph_capture_segments"), 2);
        EXPECT_GT(sum("forward_graph", "segmented_replay_segments", "prefill"), 0);
        EXPECT_GT(sum("forward_graph", "segmented_replay_segments", "decode"), 0);
        if (test.usesMTP())
        {
            EXPECT_GT(sum("mtp", "budget_limited_direct_emits"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_transactions"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_attempted_draft_tokens"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_verifier_tokens"), 0);
            if (test.lane == PipelineGenerationLane::DynamicMTP)
                EXPECT_GT(sum("mtp", "device_generation_terminal_depth_evaluated_windows"), 0);
            else
                EXPECT_EQ(sum("mtp", "device_generation_terminal_depth_evaluated_windows"), 0);
        }
        else
        {
            // Ordinary sampling uses the same controller/terminal ledger.
            // Its transactions are real, but none may perform speculative work.
            EXPECT_GT(sum("mtp", "device_generation_terminal_transactions"), 0);
            // Each independently observed cache row requires one traversal of
            // both layer domains, not one traversal per sampled response token.
            EXPECT_EQ(sum("forward_graph", "segmented_replay_segments", "decode"), 2 * consumed_main_rows);
            for (const char *counter : {"device_generation_terminal_attempted_draft_tokens",
                     "device_generation_terminal_verifier_tokens",
                     "device_generation_terminal_compact_outcome_reductions",
                     "device_generation_terminal_depth_evaluated_windows"})
                EXPECT_EQ(sum("mtp", counter), 0) << counter;
        }
    }
};

TEST_P(PipelineDomainGeneration, ExactCapturedRequestsAndContinuation) { run(); }

INSTANTIATE_TEST_SUITE_P(CrossBackend, PipelineDomainGeneration,
    ::testing::ValuesIn([] {
        std::vector<PipelineDomainCase> cases;
        for (const bool reverse : {false, true})
            for (const int width : {1, 2, 4})
                for (const auto lane : {PipelineGenerationLane::Ordinary, PipelineGenerationLane::FixedMTP,
                                       PipelineGenerationLane::DynamicMTP})
                    for (const int rows : {128, 256})
                        cases.push_back({reverse, width, lane, rows});
        return cases;
    }()),
    [](const auto &info) {
        const auto &p = info.param;
        return std::string(p.reverse ? "ROCm" : "CUDA") + std::to_string(p.width) + "_" +
            (p.reverse ? "CUDA" : "ROCm") + std::to_string(p.width) + (p.lane == PipelineGenerationLane::Ordinary ? "_Ordinary" :
                p.lane == PipelineGenerationLane::FixedMTP ? "_FixedMTP" : "_DynamicMTP") +
            (p.admitted_rows == 256 ? "" : "_Admitted" + std::to_string(p.admitted_rows));
    });
}
