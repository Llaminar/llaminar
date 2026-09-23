/**
 * @file Test__PipelineDomainGeneration.cpp
 * @brief Real captured inter-vendor generation with native TP inside each domain.
 *
 * The tiny numerical graph uses production embedding, KV and sampling kernels.
 * Its independent four-token distribution detects bad metadata/activation
 * handoffs without loading a model. Public request APIs exercise first admission,
 * continuation and reset; this is preflight, not a real-model HTTP certificate.
 */
#include "PipelineGenerationTestSupport.h"
#include "config/OrchestrationStartupPolicy.h"

namespace llaminar2::test
{
/** @brief Explicit physical topology and generation lane, never inferred from a name. */
struct PipelineDomainCase { bool reverse; int width; bool mtp; };

/** @brief Shared numerical fixture with a bounded one-/two-member domain sweep. */
class PipelineDomainGeneration : public ::testing::TestWithParam<PipelineDomainCase>
{
protected:
    /** @brief Execute exact public requests after admitting all original physical owners. */
    void run()
    {
        const auto test = GetParam();
        initCPUBackend(-1);
        auto *cuda = getCUDABackend();
        auto *rocm = getROCmBackend();
        ASSERT_NE(cuda, nullptr);
        ASSERT_NE(rocm, nullptr);
        if (cuda->deviceCount() < test.width || rocm->deviceCount() < test.width)
            GTEST_SKIP() << "Requires " << test.width << " CUDA and ROCm devices";
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
        mtp.enabled = test.mtp;
        mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
        if (test.mtp)
        {
            mtp.graph_capacity_draft_tokens = 15;
            mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
            mtp.depth_policy.initial_depth = 15;
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
                auto *backend = getBackendFor(device);
                input.device_total_bytes = backend->deviceMemoryTotal(ordinal);
                input.device_free_bytes = backend->deviceMemoryFree(ordinal);
                input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
                    .total_bytes = 1ull << 30, .admission_available_bytes = 1ull << 30};
                input.device_compute_units = 128;
                input.max_seq_len = 512;
                input.activation_seq_len = 256;
                input.mtp_enabled = test.mtp;
                input.mtp_target_query_rows = test.mtp ? 16 : 1;
                input.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory({256}, mtp);
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
        profile.n_layers = test.mtp ? 2 : 1;
        profile.mtp_layer_count = test.mtp ? 1 : 0;
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
                owner->setPPStageConfig({.first_layer = 0, .last_layer = 1, .has_embedding = domain == 0, .has_lm_head = domain == 1});
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                context.submitAndWait([&] {
                    ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
                    TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
                    ASSERT_TRUE(owner->initializeInferenceStateFromArena(1, 512, device,
                        {.activation_seq_len = 256}));
                });
                ASSERT_FALSE(HasFatalFailure());
                if (test.mtp && domain == 1) owner->setFrozenWeightSet(makeOrdinaryMTPFrozenWeightSet(*graph, device));
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
                config.nested_pp_stage_config = FactoryPPStageConfig{.first_layer = 0, .last_layer = 1,
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
        auto model = MockModelContext::createMinimal();
        model->setVocabSize(32);
        auto rank = RankOrchestrator::createForTestWithPipelineStages(model, std::move(domains), rank_config);
        ASSERT_TRUE(rank->materializeServingGraphFamilyWithoutLaunch({.prefill_bucket_rows = {256},
            .prefill_pad_token_id = 0, .main_decode_graph = ServingMainDecodeGraphKind::HistoryBearingSerial}));
        const auto geometry = PipelineTransferMemory::forRows(32, 256, test.mtp ? 16 : 1);
        for (size_t i = 0; i < participants.size(); ++i)
            EXPECT_EQ(memory->claimedBytes(participants[i]->primaryDeviceId(), PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), i % test.width ? 0 : geometry.device_bytes_per_boundary);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), geometry.host_bytes_per_boundary);

        OrchestrationConfig config;
        config.max_seq_len = 512;
        config.prefill_max_bucket_size = 256;
        config.mtp = mtp;
        config.prefix_cache.enabled = false;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        RankExecutionPlan plan;
        plan.primary_device = members.back().front();
        plan.runtime.max_seq_len = 512;
        plan.runtime.mtp = mtp;
        plan.runtime.prefix_cache = config.prefix_cache;
        auto *pipeline = rank.get();
        OrchestrationRunner runner(config, plan, std::move(rank));
        int consumed_main_rows = 0;
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
        if (test.mtp)
        {
            EXPECT_GT(sum("mtp", "budget_limited_direct_emits"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_transactions"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_attempted_draft_tokens"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_verifier_tokens"), 0);
            EXPECT_GT(sum("mtp", "device_generation_terminal_depth_evaluated_windows"), 0);
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
    ::testing::Values(PipelineDomainCase{false, 1, false}, PipelineDomainCase{true, 1, false},
        PipelineDomainCase{false, 2, false}, PipelineDomainCase{true, 2, false},
        PipelineDomainCase{false, 1, true}, PipelineDomainCase{true, 1, true},
        PipelineDomainCase{false, 2, true}, PipelineDomainCase{true, 2, true}),
    [](const auto &info) {
        const auto &p = info.param;
        return std::string(p.reverse ? "ROCm" : "CUDA") + std::to_string(p.width) + "_" +
            (p.reverse ? "CUDA" : "ROCm") + std::to_string(p.width) + (p.mtp ? "_DynamicMTP" : "_Ordinary");
    });
}
