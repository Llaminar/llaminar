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
