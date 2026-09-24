/**
 * @file Test__PipelineMTPMainForward.inc
 * @brief Exercise rank MTP condition fanout through real DGO forward execution.
 *
 * A tiny numerical model isolates orchestration: production embedding, native
 * row/activation collectives and KV kernels run on every GPU. The tail alone
 * publishes a forced target sample. Followers must consume its captured row
 * receive, advance independently and reuse serving-time graphs. This is a
 * model-free lifecycle proof, not real-model MTP accuracy certification.
 */

/** @test Both condition purposes traverse every layer stage, without new captures. */
TEST_P(PipelineMTPStateOwnership, RankMTPConditionAdvancesEveryCapturedParticipant)
{
    if (GetParam() == "CPU") return;
    initCPUBackend(-1);
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2) GTEST_SKIP() << "Requires two native GPU participants";
    for (const bool reverse : {false, true})
    {
        constexpr int participants = 2;
        ModelMemoryProfile profile;
        profile.architecture = "qwen35";
        // The retained main-forward transaction has one main block and one
        // learned MTP predictor block.  The tiny graph below executes only
        // the main block, but the terminal pipeline participant still
        // materializes the predictor's shifted KV cache.  Keep the planner's
        // physical model geometry identical to that live allocation: saying
        // that this model has only one total layer admits the main cache but
        // leaves the predictor cache without an authority-owned budget.
        profile.n_layers = 2;
        profile.mtp_layer_count = 1;
        profile.d_model = profile.head_dim = profile.vocab_size = 32;
        profile.d_ff = 64;
        profile.n_heads = profile.n_kv_heads = 1;
        profile.max_seq_len = 64;
        MTPRuntimeConfig mtp;
        mtp.enabled = true;
        mtp.graph_capacity_draft_tokens = 15;
        std::vector<DevicePlanConfig> memory_inputs;
        for (int ordinal = 0; ordinal < participants; ++ordinal)
        {
            DevicePlanConfig input;
            input.world_rank = 0;
            input.device = cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
            // Declare real admission capacity: the retained depth-15 family
            // includes opaque native graph/workspace bounds beyond this toy's
            // tiny tensor payloads. No reserve or secondary ledger is added.
            input.device_total_bytes = backend->deviceMemoryTotal(ordinal);
            input.device_free_bytes = backend->deviceMemoryFree(ordinal);
            input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
                .device = DeviceId::cpu(), .total_bytes = 1ull << 30, .admission_available_bytes = 1ull << 30};
            input.device_compute_units = 128;
            input.max_seq_len = input.activation_seq_len = 64;
            input.mtp_enabled = true;
            input.mtp_target_query_rows = 16;
            input.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory({32}, mtp);
            input.local_pipeline_backend = cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
            memory_inputs.push_back(input);
        }
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(MemoryPlanner::plan(profile, memory_inputs).physicalPlan()), 0);
        std::vector<std::shared_ptr<OrdinaryDGOGraph>> graphs;
        std::vector<std::unique_ptr<IInferenceRunner>> owners;
        std::vector<DeviceGraphOrchestrator *> stages;
        for (int participant = 0; participant < participants; ++participant)
        {
            const int ordinal = reverse ? participants - 1 - participant : participant;
            const auto device = cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
            auto config = stateConfig(device);
            config.n_layers = 1;
            config.total_n_layers = 2;
            config.layer_types.assign(2, "full_attention");
            config.kv_cache_precision = KVCachePrecision::FP16;
            config.mtp = mtp;
            config.prefix_cache.enabled = false;
            config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
            auto graph = std::make_shared<OrdinaryDGOGraph>(config, participant == 0, participant + 1 == participants);
            auto owner = DeviceGraphOrchestrator::createForTest({
                .model_ctx = MockModelContext::createMinimal(), .graph_builder = graph, .physical_memory_authority = memory});
            owner->setPPStageConfig({.first_layer = 0, .last_layer = 1,
                .has_embedding = participant == 0, .has_lm_head = participant + 1 == participants});
            auto &context = GPUDeviceContextPool::instance().getContext(device);
            context.submitAndWait([&] {
                ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
                TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
                ASSERT_TRUE(owner->initializeInferenceStateFromArena(1, 64, device));
            });
            ASSERT_FALSE(HasFatalFailure());
            if (participant + 1 == participants)
            {
                // Only the PP terminal owns a learned predictor.  Installing
                // its frozen marker before family materialization verifies the
                // same typed authority production uses for a retained sidecar;
                // followers deliberately remain predictor-free.
                owner->setFrozenWeightSet(
                    makeOrdinaryMTPFrozenWeightSet(*graph, device));
            }
            stages.push_back(owner.get());
            graphs.push_back(std::move(graph));
            owners.push_back(std::move(owner));
        }
        RankOrchestrator::Config rank_config;
        rank_config.mode = RankOrchestrator::ParallelismMode::PP;
        rank_config.max_seq_len = 64;
        auto rank = RankOrchestrator::createForTestWithPipelineStages(
            MockModelContext::createMinimal(), std::move(owners), rank_config);
        ASSERT_TRUE(rank->materializeServingGraphFamilyWithoutLaunch({
            .prefill_bucket_rows = {32}, .prefill_pad_token_id = 0,
            .main_decode_graph = ServingMainDecodeGraphKind::HistoryBearingSerial}));
        const std::array builds{graphs[0]->builds, graphs[1]->builds};
        // Use the exact captured prefill geometry.  A one-token prompt would
        // be an inadmissible padded replay below the production bucket's
        // minimum, whereas this full bucket proves the same graph that a
        // serving request actually executes.
        std::array<int, 32> prompt{};
        prompt.fill(2);
        // The injected pipeline deliberately has no legacy LocalPPContext:
        // production graph-native PP prefill crosses participants through the
        // retained bucket transaction installed above.  Going through the old
        // sequential forwardPP() entrypoint would test an absent transport
        // object rather than the captured pipeline being proved here.
        ASSERT_TRUE(rank->supportsPrefillChunkSchedule(
            static_cast<int>(prompt.size())));
        ASSERT_TRUE(rank->forwardPrefillChunkSchedule(
            prompt.data(),
            static_cast<int>(prompt.size()),
            {.bucket_sizes = {32}, .fixed_chunk_real_tokens = 32,
             .real_token_start = 0,
             .real_token_count = static_cast<int>(prompt.size())},
            /*restored_prefix_tokens=*/0,
            /*allow_padded_execution=*/true));
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            const int token = (iteration * 7 + 3) % 32;
            const auto purpose = kMTPConditionForwardPurposes[iteration % kMTPConditionForwardPurposes.size()];
            ASSERT_TRUE(rank->stageStochasticTargetTokenForDeviceSampling(token, 0));
            ASSERT_TRUE(rank->advanceMTPMainConditionFromDeviceTargetSample(-1, 0, purpose));
            for (int participant = 0; participant < participants; ++participant)
            {
                SCOPED_TRACE(::testing::Message() << "reverse=" << reverse << " iteration=" << iteration
                    << " participant=" << participant);
                auto &stage = *stages[participant];
                const auto device = stage.primaryDeviceId();
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                context.submitAndWait([&] {
                    void *const producer = DeviceGraphOrchestratorLiveStateTestAccess::mainForwardProducerStream(stage);
                    ASSERT_NE(producer, nullptr);
                    int32_t cached = -1;
                    ASSERT_TRUE(backend->deviceToHost(&cached,
                        stage.inferenceState().kv_cache->deviceSequenceCachedTokenCountPtr(0), sizeof(cached),
                        device.gpu_ordinal(), producer));
                    EXPECT_EQ(cached, iteration + 33);
                    const auto row = DeviceGraphOrchestratorLiveStateTestAccess::followerInput(stage, 1);
                    int32_t received = -1;
                    ASSERT_TRUE(backend->deviceToHost(&received, row.token_ids_device, sizeof(received),
                        device.gpu_ordinal(), producer));
                    EXPECT_EQ(received, token);
                    std::array<float, 32> hidden{};
                    ASSERT_TRUE(backend->deviceToHost(hidden.data(), stage.inferenceState().hidden->gpu_data_ptr(),
                        sizeof(hidden), device.gpu_ordinal(), producer));
                    for (int column = 0; column < 32; ++column)
                        EXPECT_EQ(hidden[column], (column - token - 1 + 32) % 32 < 4 ? 4.0F : -16.0F);
                });
                EXPECT_EQ(graphs[participant]->builds, builds[participant]) << "Live execution must reuse serving captures";
            }
        }
    }
}

/** @brief Publish the same bucket policy as the frontend, restoring it on exit. */
class PipelineMTPTestStartup final
{
public:
    /** @brief Freeze the tiny model's admitted serving ladder before construction. */
    PipelineMTPTestStartup()
    {
        if (const char *value = std::getenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES"))
            previous_ = value;
        OrchestrationConfig config;
        config.prefill_max_bucket_size = 256;
        publishOrchestrationStartupPolicy(config);
    }
    /** @brief Restore the process environment after workers and captures retire. */
    ~PipelineMTPTestStartup()
    {
        if (previous_) setenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES", previous_->c_str(), 1);
        else unsetenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES");
        mutableDebugEnv().reload();
    }
private:
    std::optional<std::string> previous_;
};

/**
 * @test Complete dynamic MTP generation has one sampler and independent PP state.
 *
 * The public request runner drives real prefill, proposals, grouped verification,
 * publication and the backend's complete generation policy. Tokens are checked
 * against an independent seeded serial oracle, not against another GPU run.
 */
TEST_P(PipelineMTPStateOwnership, DynamicPipelineMTPRunsThroughPublicRequestBoundary)
{
    if (GetParam() == "CPU") return;
    PipelineMTPTestStartup startup;
    OrdinaryGraphEvidence evidence("generation,forward_graph,mtp");
    initCPUBackend(-1);
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2) GTEST_SKIP() << "Requires two native GPU participants";
    for (const bool reverse : {false, true})
    {
    SCOPED_TRACE(::testing::Message() << "reverse=" << reverse);
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
    mtp.graph_capacity_draft_tokens = 15;
    mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    mtp.depth_policy.min_depth = 1;
    mtp.depth_policy.max_depth = 15;
    // Match the adversarial canonical E2E policy: start at the full envelope
    // and let every committed transaction change the selected draft branch.
    // A shallow default can pass while never exercising a depth transition.
    mtp.depth_policy.initial_depth = 15;
    mtp.depth_policy.window_size = 1;
    mtp.depth_policy.min_samples = 1;
    mtp.depth_policy.cooldown_steps = 0;
    mtp.depth_policy.promote_consecutive_windows = 1;
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_layers = 2;
    profile.mtp_layer_count = 1;
    profile.d_model = profile.head_dim = profile.vocab_size = 32;
    profile.d_ff = 64;
    profile.n_heads = profile.n_kv_heads = 1;
    profile.max_seq_len = 512;
    std::vector<DevicePlanConfig> memory_inputs;
    for (int ordinal = 0; ordinal < 2; ++ordinal)
    {
        DevicePlanConfig input;
        input.world_rank = 0;
        input.device = cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
        input.device_total_bytes = backend->deviceMemoryTotal(ordinal);
        input.device_free_bytes = backend->deviceMemoryFree(ordinal);
        input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
            .device = DeviceId::cpu(), .total_bytes = 1ull << 30, .admission_available_bytes = 1ull << 30};
        input.device_compute_units = 128;
        input.max_seq_len = input.activation_seq_len = 512;
        input.mtp_enabled = true;
        input.mtp_target_query_rows = 16;
        input.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory({256}, mtp);
        input.local_pipeline_backend = cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        memory_inputs.push_back(input);
    }
    auto memory = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(MemoryPlanner::plan(profile, memory_inputs).physicalPlan()), 0);
    std::vector<std::shared_ptr<OrdinaryDGOGraph>> graphs;
    std::vector<std::unique_ptr<IInferenceRunner>> owners;
    std::vector<DeviceGraphOrchestrator *> stages;
    for (int participant = 0; participant < 2; ++participant)
    {
        const auto device = memory_inputs[reverse ? 1 - participant : participant].device;
        auto config = stateConfig(device);
        config.n_layers = 1;
        config.total_n_layers = 2;
        config.layer_types.assign(2, "full_attention");
        config.kv_cache_precision = KVCachePrecision::FP16;
        config.max_seq_len = 512;
        config.mtp = mtp;
        config.prefix_cache.enabled = false;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        auto graph = std::make_shared<OrdinaryDGOGraph>(config, participant == 0, participant == 1);
        auto owner = DeviceGraphOrchestrator::createForTest({
            .model_ctx = MockModelContext::createMinimal(), .graph_builder = graph, .physical_memory_authority = memory});
        owner->setPPStageConfig({.first_layer = 0, .last_layer = 1,
            .has_embedding = participant == 0, .has_lm_head = participant == 1});
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
            TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
            ASSERT_TRUE(owner->initializeInferenceStateFromArena(1, 512, device));
        });
        ASSERT_FALSE(HasFatalFailure());
        if (participant == 1)
            owner->setFrozenWeightSet(makeOrdinaryMTPFrozenWeightSet(*graph, device));
        stages.push_back(owner.get());
        graphs.push_back(std::move(graph));
        owners.push_back(std::move(owner));
    }
    RankOrchestrator::Config rank_config;
    rank_config.mode = RankOrchestrator::ParallelismMode::PP;
    rank_config.max_seq_len = 512;
    auto model = MockModelContext::createMinimal();
    model->setVocabSize(32);
    auto rank = RankOrchestrator::createForTestWithPipelineStages(
        std::move(model), std::move(owners), rank_config);
    ASSERT_TRUE(rank->materializeServingGraphFamilyWithoutLaunch({
        .prefill_bucket_rows = {256}, .prefill_pad_token_id = 0,
        .main_decode_graph = ServingMainDecodeGraphKind::HistoryBearingSerial}));
    OrchestrationConfig config;
    config.max_seq_len = 512;
    config.prefill_max_bucket_size = 256;
    config.mtp = mtp;
    config.prefix_cache.enabled = false;
    config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
    RankExecutionPlan plan;
    plan.primary_device = GlobalDeviceAddress::fromLocalDeviceId(stages.back()->primaryDeviceId());
    plan.runtime.max_seq_len = 512;
    plan.runtime.mtp = mtp;
    plan.runtime.prefix_cache = config.prefix_cache;
    auto *const pipeline = rank.get();
    OrchestrationRunner runner(config, plan, std::move(rank));
    for (const bool greedy : {false, true})
    for (int repeat = 0; repeat < 20; ++repeat)
    {
        SCOPED_TRACE(::testing::Message() << "greedy=" << greedy << " repeat=" << repeat);
        runner.clearCache();
        SamplingParams law;
        law.temperature = greedy ? 0.0F : 1.0F;
        law.top_k = 4;
        law.seed = 0xFE000001u + repeat;
        runner.setSamplingParams(law);
        runner.setStopTokens({});
        int previous = (repeat * 7) % 32;
        std::vector<int> prompt(256, previous);
        ASSERT_TRUE(runner.prefill(prompt)) << runner.lastError();
        // HTTP thinking budgets inject a multi-token closing sequence before
        // speculative generation resumes. Submit the entire policy-token burst
        // without host observations: a per-row KV readback would hide missing
        // event edges between the shifted sidecar and pipeline main forwards.
        for (int forced = 0; forced < 24; ++forced)
        {
            previous = (previous + 7) % 32;
            const auto result = runner.forceDecodeToken(previous);
            ASSERT_TRUE(result.success()) << result.error;
            prompt.push_back(previous);
        }
        int emitted = 0;
        // The second admission must continue the existing device frontier,
        // not resample prefill or leave a follower in its previous transaction.
        for (const int budget : {65, 17})
        {
            runner.setDecodeStepTokenBudget(budget);
            const auto result = runner.decodeStep();
            ASSERT_TRUE(result.success()) << result.error;
            ASSERT_EQ(result.tokens.size(), static_cast<size_t>(budget));
            for (const auto token : result.tokens)
            {
                std::array<int, 4> support;
                for (int i = 0; i < 4; ++i) support[i] = (previous + i + 1) % 32;
                std::sort(support.begin(), support.end());
                // Sampling coordinates are absolute sequence positions; this
                // prompt occupies 256 rows, unlike the one-row ordinary fixture.
                const auto draw = sampling_math::mtp_spec_threshold_from_seed(
                    law.seed, static_cast<int>(prompt.size()) + emitted, 0);
                previous = support[greedy ? 0 : std::min(3, static_cast<int>(draw * 4))];
                ASSERT_EQ(token, previous) << "position=" << emitted;
                ++emitted;
            }
        }
        // Terminal-only inspection proves real independent layer execution.
        // No host-observed count is used to decide or repair a transaction.
        std::array<int32_t, 2> cached{};
        for (size_t participant = 0; participant < stages.size(); ++participant)
        {
            auto &stage = *stages[participant];
            auto &context = GPUDeviceContextPool::instance().getContext(stage.primaryDeviceId());
            context.submitAndWait([&] {
                ASSERT_TRUE(backend->deviceToHost(&cached[participant],
                    stage.inferenceState().kv_cache->deviceSequenceCachedTokenCountPtr(0),
                    sizeof(int32_t), stage.primaryDeviceId().gpu_ordinal(), context.defaultStream()));
            });
        }
        EXPECT_EQ(cached[0], cached[1]);
        EXPECT_GE(cached[0], static_cast<int>(prompt.size()) + emitted - 1);
        expectOrdinaryPipelinePromptKV(*pipeline, prompt, cached[0], 2);
    }
    double committed_transactions = 0;
    for (const auto &record : PerfStatsCollector::snapshot())
        if (record.domain == "mtp" && record.name == "device_generation_terminal_transactions")
            committed_transactions += record.value;
    EXPECT_GT(committed_transactions, 80)
        << "Every request/continuation must execute complete captured MTP transactions";
    }
}
