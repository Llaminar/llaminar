/**
 * @file Test__PipelineMTPPublication.inc
 * @brief Captured publication of tail-committed MTP state into a pipeline follower.
 *
 * A real participant-local KV cache appends speculative rows and then consumes
 * committed metadata. There is deliberately no outcome reducer, vocabulary,
 * response ledger, penalty history or shifted sidecar on this participant.
 * Both checkpoint preparation and acceptance use their production stage roles.
 * This isolates publication ownership from token mathematics; it does not
 * certify the complete multi-device MTP command/activation protocol.
 */

/**
 * @test Committed metadata alone advances each owned cache after grouped append.
 *
 * The complete checkpoint/append/publication sequence is captured once per
 * physical verifier width. Vary base positions and acceptance across twenty
 * resets without recapture. Zero accepted rows and unhealthy commands must
 * not manufacture a response or independently derive an acceptance decision.
 */
TEST_P(PipelineMTPStateOwnership, CapturedFollowerPublishesOnlyTailCommittedState)
{
    if (GetParam() == "CPU") return; // CPU publication is host-owned, not this GPU stage.
    initCPUBackend(-1);
    const auto device = GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend, nullptr);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        void *const stream = context.defaultStream();
        auto config = stateConfig(device);
        config.kv_cache_precision = KVCachePrecision::FP16;
        auto execution = IDeviceContext::create(device, 1);
        ASSERT_NE(execution, nullptr);
        auto model = std::make_shared<Qwen35Graph>(config, nullptr);
        DeviceGraphOrchestrator runner(model, nullptr);
        runner.setPPStageConfig({.first_layer = 0, .last_layer = 3,
            .has_embedding = true, .has_lm_head = false});
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 64, device));
        auto *const cache = runner.inferenceState().kv_cache.get();
        ASSERT_NE(cache, nullptr);
        auto rows = TestTensorFactory::createFP32Ones({16, 32});
        ASSERT_TRUE(rows->ensureOnDevice(device, stream));
        TransferEngine::requireDeviceInput(rows.get(), device, stream);

        // Device scratch owners precede every graph which embeds their addresses.
        const auto release = [backend, device](void *p) {
            if (p) backend->free(p, device.gpu_ordinal());
        };
        const size_t checkpoint_bytes = cache->deviceSequenceStateCheckpointBytes();
        ASSERT_GT(checkpoint_bytes, 0u);
        std::unique_ptr<void, decltype(release)> checkpoint(
            backend->allocate(checkpoint_bytes, device.gpu_ordinal()), release);
        std::unique_ptr<void, decltype(release)> metadata(
            backend->allocate(4 * sizeof(int32_t), device.gpu_ordinal()), release);
        ASSERT_TRUE(checkpoint && metadata);
        auto *const words = static_cast<int32_t *>(metadata.get());
        for (const int width : {2, 3, 4, 16})
        {
            SCOPED_TRACE(width);
            using Preparation = MTPVerifierPreparationStage;
            const std::array checkpoints{Preparation::MainKVCheckpointBinding{
                .cache = cache, .sequence_index = 0,
                .checkpoint_device = checkpoint.get(), .checkpoint_bytes = checkpoint_bytes}};
            Preparation::Params preparation_params;
            preparation_params.device_id = device;
            preparation_params.backend = backend;
            preparation_params.authority = Preparation::Authority::PipelineFollower;
            preparation_params.request_count = 1;
            preparation_params.padded_seq_len = width;
            preparation_params.main_kv_checkpoints = checkpoints;
            // Poison the checkpoint once. Every invalid descriptor must fail
            // before enqueueing a cache copy, not merely return false afterwards.
            std::vector<unsigned char> sentinel(checkpoint_bytes, 0xA5);
            ASSERT_TRUE(backend->hostToDevice(checkpoint.get(), sentinel.data(), checkpoint_bytes,
                device.gpu_ordinal(), stream));
            const auto reject_preparation = [&](const auto &change) {
                auto malformed = preparation_params;
                change(malformed);
                Preparation candidate(malformed);
                candidate.setGPUStream(stream);
                EXPECT_FALSE(candidate.execute(execution.get()));
            };
            for (const auto authority : {Preparation::Authority::Unbound,
                    Preparation::Authority::VerifierInputOwner, static_cast<Preparation::Authority>(255)})
                reject_preparation([&](auto &bad) { bad.authority = authority; });
            const std::array tokens{Preparation::TokenRowBinding{
                .first_token_device = words, .draft_tokens_device = words,
                .destination_device = words, .draft_token_count = 1}};
            reject_preparation([&](auto &bad) { bad.token_rows = tokens; });
            reject_preparation([&](auto &bad) { bad.base_cached_tokens_device = words; });
            reject_preparation([&](auto &bad) { bad.valid_graph_rows_device = words; });
            reject_preparation([](auto &bad) { bad.valid_graph_row_count = 1; });
            reject_preparation([&](auto &bad) { bad.generation_control_device = words; });
            reject_preparation([](auto &bad) { bad.generation_control_stride = -1; });
            reject_preparation([&](auto &bad) { bad.maintenance_rows_remaining_device =
                reinterpret_cast<const uint32_t *>(words); });
            reject_preparation([&](auto &bad) { bad.maintenance_due_device =
                reinterpret_cast<const uint32_t *>(words); });
            reject_preparation([&](auto &bad) { bad.decode_boundary_advanced_device =
                reinterpret_cast<uint32_t *>(words); });
            reject_preparation([&](auto &bad) { bad.position_ids_device = words; });
            reject_preparation([&](auto &bad) { bad.request_lengths_device = words; });
            reject_preparation([&](auto &bad) { bad.base_cached_tokens_snapshot_device = words; });
            reject_preparation([](auto &bad) { bad.main_kv_checkpoints = {}; });
            reject_preparation([](auto &bad) { bad.request_count = 0; });
            reject_preparation([](auto &bad) { bad.request_count = std::numeric_limits<int>::max(); });
            reject_preparation([](auto &bad) { bad.padded_seq_len = 0; });
            std::vector<unsigned char> checkpoint_after(checkpoint_bytes);
            ASSERT_TRUE(backend->deviceToHost(checkpoint_after.data(), checkpoint.get(), checkpoint_bytes,
                device.gpu_ordinal(), stream));
            EXPECT_EQ(checkpoint_after, sentinel);
            Preparation preparation(preparation_params);
            preparation.setGPUStream(stream);
            MTPSpeculativeStatePublicationStage::Params params;
            params.device_id = device;
            params.backend = backend;
            params.authority = MTPSpeculativeStatePublicationStage::Authority::PipelineFollower;
            params.request_count = 1;
            params.verifier_rows_per_request = width;
            params.accepted_restore_rows_device = words;
            params.target_cached_tokens_device = words + 1;
            params.accepted_state_counts_device = words + 2;
            params.publication_ok_flags_device = words + 3;
            params.main_kv_bindings = {{.cache = cache, .first_sequence_index = 0,
                .base_checkpoint_device = checkpoint.get(), .base_checkpoint_bytes = checkpoint_bytes}};
            // Invalid role combinations must fail before enqueueing anything.
            // Every pointer below is an allocated address, but no rejected
            // descriptor is allowed to consume its bytes as another authority.
            const auto reject = [&](const auto &change) {
                auto malformed = params;
                change(malformed);
                MTPSpeculativeStatePublicationStage candidate(malformed);
                ASSERT_TRUE(candidate.prepareGraphLaunch(execution.get(), stream));
                EXPECT_FALSE(candidate.execute(execution.get()));
            };
            using Authority = MTPSpeculativeStatePublicationStage::Authority;
            for (const auto authority : {Authority::Unbound, Authority::CompactOutcome,
                                         Authority::BoundedGeneration, static_cast<Authority>(255)})
                reject([&](auto &bad) { bad.authority = authority; });
            reject([&](auto &bad) { bad.outcome_tokens_device = words; });
            reject([&](auto &bad) { bad.generation_control_device = words; });
            reject([&](auto &bad) { bad.next_sidecar_condition_tokens_device = words; });
            reject([&](auto &bad) { bad.penalty_policy_device = words; });
            reject([&](auto &bad) { bad.accepted_state_counts_device = nullptr; });
            reject([&](auto &bad) { bad.main_kv_bindings.clear(); });
            MTPSpeculativeStatePublicationStage publication(params);
            ASSERT_TRUE(publication.prepareGraphLaunch(execution.get(), stream));
            auto graph = context.createGraphCapture(stream);
            ASSERT_NE(graph, nullptr);
            ScopedBackendGraphCapture recording(context, *graph, "pipeline follower publication");
            ASSERT_TRUE(recording.begin());
            const bool prepared = preparation.execute(execution.get());
            for (int layer = 0; layer < 3; ++layer)
                ASSERT_TRUE(cache->appendWithStream(layer, 0, rows.get(), rows.get(), width, stream));
            const bool enqueued = publication.execute(execution.get());
            recording.finish(); // Retire capture even when the ownership regression is red.
            ASSERT_TRUE(prepared) << "Follower checkpointing must not require draft or controller state";
            ASSERT_TRUE(enqueued) << "A follower must not require local sampling or outcome authority";
            ASSERT_TRUE(graph->instantiate());
            for (int repeat = 0; repeat < 20; ++repeat)
            {
                SCOPED_TRACE(repeat);
                const int base = 1 + repeat % 7;
                const int accepted = repeat % (width + 1);
                const bool healthy = repeat != 19;
                const std::array<int32_t, 4> committed{
                    accepted - 1, base + accepted, accepted, healthy ? 1 : 0};
                ASSERT_TRUE(cache->resetRequestState(
                    IKVCache::StateResetContext::testReinitialization(stream)));
                for (int layer = 0; layer < 3; ++layer)
                    ASSERT_TRUE(cache->appendWithStream(layer, 0, rows.get(), rows.get(), base, stream));
                ASSERT_TRUE(backend->hostToDevice(words, committed.data(), sizeof(committed),
                    device.gpu_ordinal(), stream));
                ASSERT_TRUE(graph->launch());

                // Only terminal diagnostics observe device state. There is no
                // wait between append and publication that could hide an edge.
                std::array<int32_t, 4> observed{};
                ASSERT_TRUE(backend->deviceToHost(observed.data(), words, sizeof(observed),
                    device.gpu_ordinal(), stream));
                EXPECT_EQ(observed, committed) << "Follower rewrote its tail-owned command";
                for (int layer = 0; layer < 3; ++layer)
                {
                    int head = -1, count = -1;
                    ASSERT_TRUE(backend->deviceToHost(&head, cache->deviceRingHeadPtr(layer, 0),
                        sizeof(head), device.gpu_ordinal(), stream));
                    ASSERT_TRUE(backend->deviceToHost(&count, cache->deviceCachedTokenCountPtr(layer, 0),
                        sizeof(count), device.gpu_ordinal(), stream));
                    const int expected = base + (healthy ? accepted : width);
                    EXPECT_EQ(head, expected) << "layer=" << layer;
                    EXPECT_EQ(count, expected) << "layer=" << layer;
                }
            }
        }
    });
}
