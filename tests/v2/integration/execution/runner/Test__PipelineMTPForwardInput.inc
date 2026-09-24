/**
 * @file Test__PipelineMTPForwardInput.inc
 * @brief Reject split input authority before launching a pipeline MTP forward.
 *
 * Real DGO arenas provide the local row, mailbox and rollback bindings. The
 * production prelude must accept a follower only when its frozen graph owns
 * the native row receive, and must reject foreign/null/partial bindings before
 * any device work. Native transfer/checkpoint execution has a separate captured
 * test in this same preflight group; this test covers its actual DGO admission.
 */

/** @test Followers consume one exact local row bank, never a fictitious sampler plan. */
TEST_P(PipelineMTPStateOwnership, FollowerForwardRequiresExactCapturedInputAuthority)
{
    if (GetParam() == "CPU") return; // CPU owns host rows rather than native GPU collective input.
    initCPUBackend(-1);
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2) GTEST_SKIP() << "Requires two native GPU participants";
    const int participants = std::min(3, backend->deviceCount());
    std::vector<GlobalDeviceAddress> devices;
    for (int index = 0; index < participants; ++index)
        devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(
            cuda ? DeviceId::cuda(index) : DeviceId::rocm(index)));
    LocalTPContext collective(devices, {}, CollectiveBackendType::AUTO);
    for (int participant = 0; participant + 1 < participants; ++participant)
    {
        const auto device = devices[participant].toLocalDeviceId();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
            auto config = stateConfig(device);
            config.mtp.enabled = true;
            config.mtp.graph_capacity_draft_tokens = 15;
            auto model = std::make_shared<Qwen35Graph>(config, nullptr);
            DeviceGraphOrchestrator runner(model, nullptr);
            runner.setPPStageConfig({.first_layer = participant, .last_layer = participant + 1,
                .has_embedding = participant == 0, .has_lm_head = false});
            ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 64, device));
            IForwardExecutionHost &host = runner;
            void *const stream = context.defaultStream();
            PipelineForwardGraphEdges edges(collective, participant, *runner.inferenceState().hidden,
                config.d_model, Peer::pipelineFollowerState(runner));
            // The borrowed contributor is removed before it goes out of scope.
            auto detach = [&](void *) { Peer::bindPipelineEdges(runner, nullptr); };
            std::unique_ptr<void, decltype(detach)> edge_lifetime(&edges, detach);
            for (const int rows : {1, 2, 3, 4, 16})
            {
                SCOPED_TRACE(std::to_string(participant) + ":rows=" + std::to_string(rows));
                const auto input = Peer::followerInput(runner, rows);
                ASSERT_NE(input.token_ids_device, nullptr);
                ASSERT_NE(input.position_ids_device, nullptr);
                ASSERT_NE(input.sequence_lengths_device, nullptr);
                Peer::bindPipelineEdges(runner, nullptr);
                EXPECT_FALSE(host.prepareDeviceTokenInputsForForwardGraphExecution(input, stream, device));
                EXPECT_FALSE(host.prepareAllPositionVerifierGraphMetadata(input, stream, device));
                Peer::bindPipelineEdges(runner, &edges);
                EXPECT_TRUE(host.prepareDeviceTokenInputsForForwardGraphExecution(input, stream, device));
                EXPECT_TRUE(host.prepareAllPositionVerifierGraphMetadata(input, stream, device));
                auto reject = [&](const ForwardInput &bad, void *bad_stream, DeviceId bad_device) {
                    EXPECT_FALSE(host.prepareDeviceTokenInputsForForwardGraphExecution(bad, bad_stream, bad_device));
                    EXPECT_FALSE(host.prepareAllPositionVerifierGraphMetadata(bad, bad_stream, bad_device));
                };
                reject(input, nullptr, device);
                reject(input, stream, DeviceId::cpu());
                auto bad = input;
                bad.token_ids_device = nullptr;
                reject(bad, stream, device);
                bad = input;
                bad.position_ids_device = nullptr;
                reject(bad, stream, device);
                bad = input;
                bad.sequence_lengths_device = nullptr;
                reject(bad, stream, device);
                bad = input;
                bad.token_ids_device = static_cast<const int32_t *>(input.token_ids_device) + 1;
                reject(bad, stream, device);
                bad = input;
                bad.position_ids_device = static_cast<const int32_t *>(input.position_ids_device) + 1;
                reject(bad, stream, device);
                bad = input;
                bad.sequence_lengths_device = input.sequence_lengths_device + 1;
                reject(bad, stream, device);
                bad = input;
                bad.kv_cache = nullptr;
                reject(bad, stream, device);
                bad = input;
                bad.batch_size = 2;
                reject(bad, stream, device);
                bad = input;
                bad.seq_len = std::numeric_limits<int>::max();
                reject(bad, stream, device);
                bad = input;
                bad.external_hidden_state = participant ? nullptr : runner.inferenceState().hidden.get();
                reject(bad, stream, device);
                // Validation itself cannot consume or manufacture a token plan;
                // an unchanged legal transaction remains admissible afterward.
                EXPECT_TRUE(host.prepareDeviceTokenInputsForForwardGraphExecution(input, stream, device));
            }
        });
    }
}
