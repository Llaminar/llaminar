/**
 * @file Test__PipelineMTPPublicationBindings.inc
 * @brief Prove DGO publication uses canonical follower storage and capture identity.
 *
 * The real model arena and checkpoint pool provide every metadata address. A
 * nonterminal participant must never need a fabricated outcome or predictor
 * mailbox. This setup proof complements native captured transport: it tests
 * the production materializer instead of reconstructing its Params in a test.
 */

/** @test A follower installs and reuses the exact local publication family without a sampler. */
TEST_P(PipelineMTPStateOwnership, FollowerPublicationUsesCanonicalLocalOwnership)
{
    if (GetParam() == "CPU") return;
    initCPUBackend(-1);
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2) GTEST_SKIP() << "Requires a native pipeline cohort";
    const auto device = cuda ? DeviceId::cuda(0) : DeviceId::rocm(0);
    const auto tail = cuda ? DeviceId::cuda(1) : DeviceId::rocm(1);
    LocalTPContext collective({GlobalDeviceAddress::fromLocalDeviceId(device),
        GlobalDeviceAddress::fromLocalDeviceId(tail)}, {}, CollectiveBackendType::AUTO);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
        auto config = stateConfig(device);
        config.mtp.enabled = true;
        config.mtp.graph_capacity_draft_tokens = 15;
        auto model = std::make_shared<Qwen35Graph>(config, nullptr);
        DeviceGraphOrchestrator runner(model, nullptr);
        runner.setPPStageConfig({.first_layer = 0, .last_layer = 3,
            .has_embedding = true, .has_lm_head = false});
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 64, device));
        IForwardExecutionHost &host = runner;
        ASSERT_TRUE(host.ensureDeviceWorkspaceAllocated(ComputeGraph{}, 16));
        const auto checkpoint = Peer::pipelineFollowerState(runner);
        PipelineForwardGraphEdges edges(collective, 0, *runner.inferenceState().hidden, config.d_model, checkpoint);
        // Both the frozen contributor and local verifier identities outlive
        // the publication cache. Failure paths use this same RAII retirement.
        ComputeGraph verifier;
        HiddenStateRowSelectStage::Params observation;
        observation.device_id = device;
        observation.input = runner.inferenceState().hidden.get();
        observation.output = runner.inferenceState().hidden.get();
        observation.seq_len = 1;
        observation.d_model = config.d_model;
        verifier.addNode("local_verifier_state", std::make_unique<HiddenStateRowSelectStage>(observation), device);
        const auto detach = [&](void *) {
            Peer::retirePublication(runner);
            Peer::bindPipelineEdges(runner, nullptr);
        };
        std::unique_ptr<void, decltype(detach)> lifetime(&edges, detach);
        std::string error;
        EXPECT_FALSE(Peer::prepareFollowerPublication(runner, verifier, 2, &error));
        EXPECT_EQ(Peer::publicationGraph(runner), nullptr);
        Peer::bindPipelineEdges(runner, &edges);
        for (const int rows : {2, 3, 4, 16})
        {
            SCOPED_TRACE(rows);
            ASSERT_TRUE(Peer::prepareFollowerPublication(runner, verifier, rows, &error)) << error;
            auto *const installed = Peer::publicationGraph(runner);
            ASSERT_NE(installed, nullptr);
            EXPECT_EQ(installed->getExecutionOrder(), (std::vector<std::string>{
                "mtp_speculative_state_publication"}));
            const auto params = Peer::publicationParams(runner);
            EXPECT_EQ(params.authority, MTPSpeculativeStatePublicationStage::Authority::PipelineFollower);
            ASSERT_EQ(params.main_kv_bindings.size(), 1u);
            EXPECT_EQ(params.main_kv_bindings.front().cache, checkpoint.checkpoint.cache);
            EXPECT_EQ(params.main_kv_bindings.front().base_checkpoint_device, checkpoint.checkpoint.checkpoint_device);
            EXPECT_EQ(params.main_kv_bindings.front().base_checkpoint_bytes, checkpoint.checkpoint.checkpoint_bytes);
            EXPECT_EQ(params.verifier_rows_per_request, rows);
            EXPECT_EQ(params.verifier_state_stages.size(), 1u);
            EXPECT_NE(params.accepted_restore_rows_device, nullptr);
            EXPECT_NE(params.target_cached_tokens_device, nullptr);
            EXPECT_NE(params.accepted_state_counts_device, nullptr);
            EXPECT_NE(params.publication_ok_flags_device, nullptr);
            EXPECT_EQ(params.outcome_tokens_device, nullptr);
            EXPECT_EQ(params.generation_control_device, nullptr);
            EXPECT_EQ(params.next_sidecar_condition_tokens_device, nullptr);
            EXPECT_EQ(params.penalty_policy_device, nullptr);
            ASSERT_TRUE(Peer::prepareFollowerPublication(runner, verifier, rows, &error)) << error;
            EXPECT_EQ(Peer::publicationGraph(runner), installed) << "Unchanged identity must preserve the graph";
            for (const int invalid : {0, 1, 17, std::numeric_limits<int>::max()})
            {
                EXPECT_FALSE(Peer::prepareFollowerPublication(runner, verifier, invalid, &error));
                EXPECT_EQ(Peer::publicationGraph(runner), installed);
            }
            EXPECT_FALSE(Peer::executePublication(runner, nullptr, &error));
            Peer::bindPipelineEdges(runner, nullptr);
            EXPECT_FALSE(Peer::executePublication(runner, context.defaultStream(), &error));
            EXPECT_EQ(Peer::publicationGraph(runner), installed);
            Peer::bindPipelineEdges(runner, &edges);
        }
    });
}
