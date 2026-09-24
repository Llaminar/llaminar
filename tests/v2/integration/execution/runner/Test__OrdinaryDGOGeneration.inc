/**
 * @file Test__OrdinaryDGOGeneration.inc
 * @brief Exercise ordinary generation through DGO admission, graph and terminal APIs.
 *
 * The small graph uses production embedding and KV-append kernels. It isolates
 * lifecycle from transformer arithmetic; expected tokens are independently
 * generated from an exact four-token uniform distribution. This is model-free
 * preflight evidence, not HTTP or real-model numerical certification.
 */

/** @test Real rank composition produces exact seeded responses with independent stage KV.
 * Each request resets data but reuses the complete captured pipeline policy.
 * A one-token budget executes only the tail sampler; followers must still
 * retire without consuming a row. Continued responses consume their previous
 * frontier exactly once. Physical ordinals are reversible on CUDA and ROCm.
 * Three/four-stage cohorts additionally prove that an intermediate participant
 * can both receive and send, and survive capture phases in which it is idle. */
TEST_P(PipelineMTPStateOwnership, OrdinaryPipelineHasOneTailResponseOwner)
{
    enum class PrefixEntry { Fresh, Full, Partial };
    if (GetParam() == "CPU") return; // GPU graph lifecycle; CPU owns host execution.
    initCPUBackend(-1);
    const auto first_device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto *backend = getBackendFor(first_device);
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2) GTEST_SKIP() << "Requires two " << GetParam() << " participants";
    // The injected rank owns real DGOs but bypasses frontend transport setup.
    // Install the same canonical router used by production prefill handoffs.
    struct RouterLifetime
    {
        /** @brief Preserve any transport owner installed by the surrounding fixture. */
        RouterLifetime() : owned(GlobalBackendRouter::get() == nullptr)
        { if (owned) GlobalBackendRouter::initForTests(); }
        /** @brief Release test-owned communicators only after all runner graphs. */
        ~RouterLifetime() { if (owned) GlobalBackendRouter::shutdown(); }
        bool owned;
    } router;
    OrdinaryGraphEvidence evidence;
    using namespace sampling_math;
    // Two stages do not exercise an intermediate owner or an idle native
    // capture participant. Admit each available topology rather than assuming
    // that a head/tail proof also certifies the middle of a longer pipeline.
    for (const int participant_count : {2, 3, 4})
    {
        if (backend->deviceCount() < participant_count) continue;
        for (const bool reverse : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "participants=" << participant_count << " reverse=" << reverse);
            const double prior_materializations = evidence.materializations("pipeline_graph_materializations");
            // Even this model-free transport proof uses the production memory BOM
            // and one rank-wide ledger. No independent collective allocator may
            // materialize fence words outside admission.
            ModelMemoryProfile profile;
            profile.architecture = "qwen35";
            profile.n_layers = 1;
            profile.d_model = profile.head_dim = profile.vocab_size = 32;
            profile.d_ff = 64;
            profile.n_heads = profile.n_kv_heads = 1;
            profile.max_seq_len = 1024;
            std::vector<DevicePlanConfig> memory_inputs;
            for (int ordinal = 0; ordinal < participant_count; ++ordinal)
            {
                DevicePlanConfig input;
                input.world_rank = 0;
                input.device = GetParam() == "CUDA" ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
                input.device_total_bytes = input.device_free_bytes = 1ull << 30;
                input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
                    .device = DeviceId::cpu(), .total_bytes = 1ull << 30, .admission_available_bytes = 1ull << 30};
                input.device_compute_units = 128;
                input.max_seq_len = input.activation_seq_len = 1024;
                input.local_pipeline_backend = GetParam() == "CUDA" ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                input.prefix_cache.enabled = true;
                input.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
                input.prefix_cache.ram_budget_bytes = 4u << 20;
                input.prefix_cache.device_budget_bytes = input.prefix_cache.disk_budget_bytes = 0;
                input.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory({256, 512}, {});
                memory_inputs.push_back(input);
            }
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(MemoryPlanner::plan(profile, memory_inputs).physicalPlan()), 0);
            std::vector<std::shared_ptr<OrdinaryDGOGraph>> graphs;
            std::vector<std::unique_ptr<IInferenceRunner>> owners;
            std::vector<DeviceGraphOrchestrator *> stages(participant_count);
            for (int i = 0; i < participant_count; ++i)
            {
                const int ordinal = reverse ? participant_count - 1 - i : i;
                const auto device = GetParam() == "CUDA" ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
                auto config = stateConfig(device);
                config.n_layers = config.total_n_layers = 1;
                config.layer_types.assign(1, "full_attention");
                config.max_seq_len = 1024;
                config.kv_cache_precision = KVCachePrecision::FP16;
                // This explicit model-free constructor has no file-backed model
                // admission. Exercise the real RAM prefix tier; device-hot/disk
                // admission belongs to the production-factory model tests.
                config.prefix_cache.enabled = true;
                config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
                config.prefix_cache.ram_budget_bytes = 4u << 20;
                config.prefix_cache.device_budget_bytes = 0;
                config.prefix_cache.disk_budget_bytes = 0;
                auto graph = std::make_shared<OrdinaryDGOGraph>(config, i == 0, i + 1 == participant_count);
                auto owner = DeviceGraphOrchestrator::createForTest({
                    .model_ctx = MockModelContext::createMinimal(), .graph_builder = graph,
                    .physical_memory_authority = memory});
                owner->setPPStageConfig({.first_layer = 0, .last_layer = 1,
                    .has_embedding = i == 0, .has_lm_head = i + 1 == participant_count});
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                context.submitAndWait([&] {
                    ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
                    TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
                    ASSERT_TRUE(owner->initializeInferenceStateFromArena(1, 1024, device));
                });
                ASSERT_FALSE(HasFatalFailure());
                stages[i] = owner.get();
                graphs.push_back(std::move(graph));
                owners.push_back(std::move(owner));
            }
            RankOrchestrator::Config rank_config;
            rank_config.mode = RankOrchestrator::ParallelismMode::PP;
            rank_config.max_seq_len = 1024;
            auto rank = RankOrchestrator::createForTestWithPipelineStages(
                MockModelContext::createMinimal(), std::move(owners), rank_config);
            ASSERT_TRUE(rank->materializeServingGraphFamilyWithoutLaunch({
                .prefill_bucket_rows = {256, 512}, .prefill_pad_token_id = 0,
                .main_decode_graph = ServingMainDecodeGraphKind::HistoryBearingSerial}));
            for (auto *stage : stages)
                EXPECT_EQ(memory->claimedBytes(stage->primaryDeviceId(), PhysicalMemoryOwner::LocalCollective,
                    PhysicalMemoryMaterializationKind::NewAllocation), sizeof(int32_t));
            std::optional<std::vector<int>> retained_builds;
            for (int repetition = 0; repetition < 20; ++repetition)
            {
                const int budget = std::array{1, 2, 17, 384}[repetition % 4];
                const int prompt_count = budget == 384 ? 1 : std::array{1, 17, 257, 509}[(repetition / 4) % 4];
                // Long prompts cross archive and graph-bucket boundaries. Reuse
                // their captured parents for a full hit and a divergent suffix;
                // archive lookup/import/terminal restoration remain production APIs.
                const bool exercise_prefix = repetition == 10 || repetition == 14;
                for (const auto entry : {PrefixEntry::Fresh, PrefixEntry::Full, PrefixEntry::Partial})
                {
                    if (entry != PrefixEntry::Fresh && !exercise_prefix) continue;
                    std::vector<int> prompt(prompt_count);
                    for (int row = 0; row < prompt_count; ++row) prompt[row] = (repetition + row) % 32;
                    if (entry == PrefixEntry::Partial) prompt.back() = (prompt.back() + 7) % 32;
                    const uint64_t seed = (uint64_t{1} << 40) + repetition;
                    SCOPED_TRACE(::testing::Message() << "reverse=" << reverse << " request=" << repetition
                        << " budget=" << budget << " prefix=" << static_cast<int>(entry));
                    rank->resetInferenceState(InferenceStateResetRequest::requestBoundary("pipeline-native-replay"));
                    const int bucket = repetition % 2 ? 512 : 256;
                    ASSERT_TRUE(rank->supportsPrefillChunkSchedule(prompt_count));
                    if (entry != PrefixEntry::Fresh)
                    {
                        // Reset normally retires positions, not all physical KV
                        // bytes. Overwrite those bytes with a different real
                        // prefill so a missing import cannot pass accidentally.
                        // Do not synchronize: reset/import must join that producer.
                        std::vector<int> unrelated = prompt;
                        for (auto &token : unrelated) token = (token + 16) % 32;
                        ASSERT_TRUE(rank->forwardPrefillChunkSchedule(unrelated.data(), prompt_count,
                            {.bucket_sizes = {bucket}, .fixed_chunk_real_tokens = bucket,
                             .real_token_start = 0, .real_token_count = prompt_count}, 0, true));
                        rank->resetInferenceState(InferenceStateResetRequest::requestBoundary("pipeline-prefix-replacement"));
                    }
                    const auto admission = rank->lookupPrefix(prompt);
                    ASSERT_TRUE(admission.cache_enabled);
                    ASSERT_TRUE(admission.supported) << admission.bypass_reason;
                    ASSERT_GT(admission.block_size, 0);
                    if (entry == PrefixEntry::Fresh) ASSERT_EQ(admission.cached_tokens, 0);
                    else
                    {
                        const int expected_cached = entry == PrefixEntry::Full ? prompt_count
                            : ((prompt_count - 1) / admission.block_size) * admission.block_size;
                        ASSERT_EQ(admission.cached_tokens, expected_cached);
                        ASSERT_GT(admission.cached_tokens, 0);
                        ASSERT_TRUE(rank->populatePrefix(admission));
                    }
                    if (entry == PrefixEntry::Full)
                    {
                        ASSERT_TRUE(admission.has_terminal_logits);
                        ASSERT_TRUE(rank->restorePrefixTerminalState(admission));
                    }
                    else
                    {
                        ASSERT_TRUE(rank->forwardPrefillChunkSchedule(prompt.data() + admission.cached_tokens,
                            prompt_count - admission.cached_tokens,
                            {.bucket_sizes = {bucket}, .fixed_chunk_real_tokens = bucket,
                             .real_token_start = admission.cached_tokens,
                             .real_token_count = prompt_count - admission.cached_tokens}, 0, true));
                        if (exercise_prefix && entry == PrefixEntry::Fresh)
                            ASSERT_TRUE(rank->harvestPrefix(admission, prompt, prompt_count));
                    }
                    SamplingParams law;
                    law.temperature = 1.0F;
                    law.top_k = 4;
                    int previous = prompt.back();
                    int emitted = 0;
                    for (const bool continuation : {false, true})
                    {
                        const int segment_budget = continuation ? 17 : budget;
                        SCOPED_TRACE(::testing::Message() << "continuation=" << continuation);
                        std::vector<double> prior_forwards;
                        for (auto *stage : stages)
                            prior_forwards.push_back(evidence.completedForwards(stage->primaryDeviceId()));
                        ASSERT_TRUE(rank->beginDeviceResidentGeneration({.request_count = 1, .max_new_tokens = segment_budget,
                            .depth_policy = DeviceGenerationPolicy::ordinary(),
                            .initial_leading_row_disposition = continuation ? DeviceGenerationLeadingRowDisposition::AlreadyEmitted
                                                                          : DeviceGenerationLeadingRowDisposition::PendingResponse,
                            .sampling_seeds = GenerationRequestSeeds(std::vector<uint64_t>{seed}), .ordinary_sampling = law}));
                        EXPECT_FALSE(rank->launchDeviceResidentGeneration()) << "Admission alone cannot submit an incomplete pipeline";
                        ASSERT_TRUE(rank->materializeDeviceResidentGeneration(1, 0, DeviceGenerationLoopTopology::FixedDepth,
                            DeviceGenerationSamplingMode::Stochastic));
                        EXPECT_EQ(evidence.materializations("pipeline_graph_materializations"), prior_materializations + participant_count)
                            << "Each participant must retain its graph across request resets and continuations";
                        std::vector<int> builds;
                        for (const auto &graph : graphs) builds.push_back(graph->builds);
                        if (retained_builds) EXPECT_EQ(builds, *retained_builds);
                        retained_builds = builds;
                        ASSERT_TRUE(rank->launchDeviceResidentGeneration());
                        EXPECT_FALSE(rank->launchDeviceResidentGeneration()) << "A distributed request is submitted exactly once";
                        DeviceGenerationTerminalResult result;
                        ASSERT_TRUE(rank->finishDeviceResidentGeneration(&result));
                        ASSERT_TRUE(result.valid());
                        for (size_t i = 0; i < stages.size(); ++i)
                            EXPECT_EQ(evidence.completedForwards(stages[i]->primaryDeviceId()),
                                prior_forwards[i] + segment_budget - (continuation ? 0 : 1));
                        EXPECT_EQ(result.device, stages.back()->primaryDeviceId());
                        ASSERT_EQ(result.requests.size(), 1u);
                        ASSERT_EQ(result.requests[0].tokens.size(), static_cast<size_t>(segment_budget));
                        for (int row = 0; row < segment_budget; ++row)
                        {
                            std::array<int, 4> support;
                            for (int j = 0; j < 4; ++j) support[j] = (previous + j + 1) % 32;
                            std::sort(support.begin(), support.end());
                            previous = support[std::min(3, static_cast<int>(mtp_spec_threshold_from_seed(seed, prompt_count + emitted, 0) * 4))];
                            ASSERT_EQ(result.requests[0].tokens[row], previous) << "row=" << emitted;
                            ++emitted;
                        }
                        // This read is test-only terminal evidence, not a dispatch input.
                        for (auto *stage : stages)
                        {
                            int cached = -1;
                            const auto device = stage->primaryDeviceId();
                            ASSERT_TRUE(backend->deviceToHost(&cached,
                                stage->inferenceState().kv_cache->deviceSequenceCachedTokenCountPtr(0), sizeof(cached),
                                device.gpu_ordinal(), GPUDeviceContextPool::instance().getContext(device).defaultStream()));
                            EXPECT_EQ(cached, prompt_count + emitted - 1);
                        }
                    }
                    if (exercise_prefix)
                        ASSERT_NO_FATAL_FAILURE(expectOrdinaryPipelinePromptKV(*rank, prompt,
                            prompt_count + emitted - 1, stages.size()));
                }
            }
            rank.reset(); // Retire borrowing graphs before the communicator's fence leases.
            for (const auto &input : memory_inputs)
                EXPECT_EQ(memory->claimedBytes(input.device, PhysicalMemoryOwner::LocalCollective,
                    PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        }
    }
}

/** @test DGO returns exact resident-generated tokens across twenty seeded request resets. */
TEST_P(PipelineMTPStateOwnership, OrdinaryGenerationRunsThroughProductionOwner)
{
    if (GetParam() == "CPU") return; // CPU owns a host generation loop, not GPU graphs.
    using namespace sampling_math;
    initCPUBackend(-1);
    const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    ASSERT_NE(getBackendFor(device), nullptr);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        auto config = stateConfig(device);
        config.n_layers = config.total_n_layers = 1;
        config.layer_types.assign(1, "full_attention");
        config.max_seq_len = 512;
        auto graph = std::make_shared<OrdinaryDGOGraph>(config);
        ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
        TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
        DeviceGraphOrchestrator runner(graph, nullptr);
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 512, device));
        auto *const backend = getBackendFor(device);
        std::optional<int> retained_forward_builds;
        for (const bool greedy : {true, false})
            for (int repetition = 0; repetition < 20; ++repetition)
            {
                const int budget = std::array{1, 2, 17, 384}[repetition % 4];
                const int prompt = (repetition * 7) % 32, position = 0;
                const uint64_t seed = uint64_t{1} << 40 | (19 + repetition);
                SCOPED_TRACE(::testing::Message() << "greedy=" << greedy << " repeat=" << repetition << " budget=" << budget);
                runner.resetInferenceState(InferenceStateResetRequest::requestBoundary("ordinary-dgo-replay"));
                // Produce genuine initial logits and KV state. Explicit decode
                // geometry avoids padded prompt work in this single-row probe.
                ForwardInput input;
                input.token_ids = &prompt;
                input.position_ids = &position;
                input.batch_size = input.seq_len = input.real_seq_len = 1;
                input.execution_phase = ForwardExecutionPhase::Decode;
                input.device = device;
                input.kv_cache = runner.inferenceState().kv_cache.get();
                input.device_state_publication_stream = context.defaultStream();
                ForwardOutput output;
                runner.setBuffers(runner.inferenceState().toModelBuffers());
                ASSERT_TRUE(runner.executeForward(input, output));
                SamplingParams law;
                law.temperature = greedy ? 0.0F : 1.0F;
                law.top_k = 4;
                const DeviceGenerationAdmissionRequest admission{
                    .request_count = 1, .max_new_tokens = budget,
                    .depth_policy = DeviceGenerationPolicy::ordinary(),
                    .sampling_seeds = GenerationRequestSeeds(std::vector<uint64_t>{seed}),
                    .ordinary_sampling = law};
                ASSERT_TRUE(runner.beginDeviceResidentGeneration(admission));
                ASSERT_TRUE(runner.materializeDeviceResidentGeneration(1, 0,
                    DeviceGenerationLoopTopology::FixedDepth,
                    greedy ? DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic));
                const int builds = graph->builds;
                if (retained_forward_builds)
                    EXPECT_EQ(builds, *retained_forward_builds) << "Request reset must preserve retained forward topology";
                retained_forward_builds = builds;
                ASSERT_TRUE(runner.launchDeviceResidentGeneration());
                DeviceGenerationTerminalResult result;
                ASSERT_TRUE(runner.finishDeviceResidentGeneration(&result));
                ASSERT_TRUE(result.valid());
                ASSERT_EQ(result.requests.size(), 1u);
                ASSERT_EQ(result.requests.front().tokens.size(), budget);
                int previous = prompt;
                for (int row = 0; row < budget; ++row)
                {
                    std::array<int, 4> support;
                    for (int i = 0; i < 4; ++i) support[i] = (previous + i + 1) % 32;
                    std::sort(support.begin(), support.end());
                    const auto draw = mtp_spec_threshold_from_seed(seed, row + 1, 0);
                    previous = support[greedy ? 0 : std::min(3, static_cast<int>(draw * 4))];
                    ASSERT_EQ(result.requests.front().tokens[row], previous) << "row=" << row;
                }
                EXPECT_EQ(graph->builds, builds);
                int32_t cached = -1;
                ASSERT_TRUE(backend->deviceToHost(&cached,
                    runner.inferenceState().kv_cache->deviceSequenceCachedTokenCountPtr(0), sizeof(cached),
                    device.gpu_ordinal(), context.defaultStream()));
                EXPECT_EQ(cached, budget); // One prompt + budget-1 consumed outputs.
                EXPECT_EQ(result.requests.front().published_state_commit_count, budget - 1);
            }
    });
}

/**
 * @test Public request admission composes the real device owner, including continuations.
 *
 * One-token prompts may use a padded prefill bucket, but this model-free graph
 * has exactly one real row. The public runner, DGO, GPU sampler, graph compiler
 * and KV owner are real. Expected stochastic outputs use the independent
 * uniform-distribution oracle; no saved answer or host sampler drives execution.
 */
TEST_P(PipelineMTPStateOwnership, OrdinaryGenerationRunsThroughPublicRequestBoundary)
{
    if (GetParam() == "CPU") return;
    OrdinaryGraphEvidence evidence;
    initCPUBackend(-1);
    const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        auto graph_config = stateConfig(device);
        graph_config.n_layers = graph_config.total_n_layers = 1;
        graph_config.layer_types.assign(1, "full_attention");
        graph_config.max_seq_len = 512;
        graph_config.prefix_cache.enabled = false;
        graph_config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        auto graph = std::make_shared<OrdinaryDGOGraph>(graph_config);
        ASSERT_TRUE(graph->table.ensureOnDevice(device, context.defaultStream()));
        TransferEngine::publishDeviceWrite(&graph->table, device, context.defaultStream());
        auto child = std::make_unique<DeviceGraphOrchestrator>(graph, nullptr);
        ASSERT_TRUE(child->initializeInferenceStateFromArena(1, 512, device));
        ASSERT_TRUE(child->materializeServingGraphFamilyWithoutLaunch({
            .prefill_bucket_rows = {256, 512},
            .prefill_pad_token_id = 0,
            .main_decode_graph = ServingMainDecodeGraphKind::HistoryBearingSerial}));
        auto *const state_owner = child.get();
        OrchestrationConfig config;
        config.max_seq_len = 512;
        config.mtp.enabled = false;
        config.prefix_cache.enabled = false;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        RankExecutionPlan plan;
        plan.primary_device = device.is_cuda() ? GlobalDeviceAddress::cuda(0) : GlobalDeviceAddress::rocm(0);
        plan.runtime.max_seq_len = 512;
        plan.runtime.mtp.enabled = false;
        plan.runtime.prefix_cache = config.prefix_cache;
        OrchestrationRunner runner(config, plan, std::move(child));
        for (const bool greedy : {true, false})
            for (int repeat = 0; repeat < 20; ++repeat)
            {
                SCOPED_TRACE(::testing::Message() << "greedy=" << greedy << " repeat=" << repeat);
                runner.clearCache();
                SamplingParams law;
                law.temperature = greedy ? 0.0F : 1.0F;
                law.top_k = 4;
                law.seed = 0xFE000001u + repeat;
                law.presence_penalty = greedy ? 0.5F : 0.0F;
                law.frequency_penalty = greedy ? 0.125F : 0.0F;
                runner.setSamplingParams(law);
                runner.setStopTokens({});
                int previous = (repeat * 7) % 32;
                ASSERT_TRUE(runner.prefill({previous})) << runner.lastError();
                int emitted = 0;
                std::array<int, 32> history{};
                // Each call continues the same request without refeeding a
                // returned host token. The first budget-one call owns no row.
                for (const int budget : {1, 2, 17, 384})
                {
                    const double prior_forwards = evidence.completedForwards(device);
                    runner.setDecodeStepTokenBudget(budget);
                    const auto result = runner.decodeStep();
                    ASSERT_TRUE(result.success()) << result.error;
                    EXPECT_EQ(evidence.completedForwards(device),
                        prior_forwards + budget - (emitted == 0 ? 1 : 0));
                    ASSERT_EQ(result.tokens.size(), budget);
                    EXPECT_FALSE(result.is_complete);
                    for (const auto token : result.tokens)
                    {
                        std::array<int, 4> support;
                        for (int i = 0; i < 4; ++i) support[i] = (previous + i + 1) % 32;
                        std::sort(support.begin(), support.end());
                        const auto draw = sampling_math::mtp_spec_threshold_from_seed(law.seed, emitted + 1, 0);
                        previous = greedy ? *std::min_element(support.begin(), support.end(),
                            [&](int left, int right) { return history[left] < history[right]; })
                            : support[std::min(3, static_cast<int>(draw * 4))];
                        ASSERT_EQ(token, previous) << "row=" << emitted;
                        ++history[token];
                        ++emitted;
                    }
                    int32_t cached = -1;
                    ASSERT_TRUE(getBackendFor(device)->deviceToHost(&cached,
                        state_owner->inferenceState().kv_cache->deviceSequenceCachedTokenCountPtr(0),
                        sizeof(cached), device.gpu_ordinal(), context.defaultStream()));
                    EXPECT_EQ(cached, emitted);
                    EXPECT_EQ(evidence.materializations(), greedy ? 1 : 2)
                        << "Budget/seed/reset changes must not rebuild the complete generation graph";
                }
            }
        // A stop sampled from prefill consumes no row; a later stop consumes
        // only its preceding conditions. Neither permits another admission.
        for (const int stop : {1, 3})
        {
            runner.clearCache();
            SamplingParams law;
            law.temperature = 0.0F;
            runner.setSamplingParams(law);
            runner.setStopTokens({stop});
            ASSERT_TRUE(runner.prefill({0})) << runner.lastError();
            runner.setDecodeStepTokenBudget(384);
            const auto result = runner.decodeStep();
            ASSERT_TRUE(result.success()) << result.error;
            ASSERT_EQ(result.tokens.size(), stop);
            EXPECT_EQ(result.tokens.back(), stop);
            EXPECT_TRUE(result.is_complete);
            int32_t cached = -1;
            ASSERT_TRUE(getBackendFor(device)->deviceToHost(&cached,
                state_owner->inferenceState().kv_cache->deviceSequenceCachedTokenCountPtr(0),
                sizeof(cached), device.gpu_ordinal(), context.defaultStream()));
            EXPECT_EQ(cached, stop);
            EXPECT_FALSE(runner.decodeStep().success());
            EXPECT_EQ(evidence.materializations(), 3)
                << "Changing stop tokens updates request data, not graph topology";
        }
    });
}
