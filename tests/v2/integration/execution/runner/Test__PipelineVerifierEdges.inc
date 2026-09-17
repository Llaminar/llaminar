/**
 * @file Test__PipelineVerifierEdges.inc
 * @brief Native GPU proof of scalar/grouped pipeline activation transport and bank reuse.
 *
 * Real row-selection kernels consume tail-broadcast lengths and independently
 * owned activation banks. The production graph contributor inserts the row
 * broadcasts, follower-local checkpoint and native activation nodes. Capture
 * contains no prefill materializer or host activation transfer. Test-only
 * uploads precede replay and diagnostics follow the complete captured graph.
 * This proves the transport primitive, not the complete speculative controller.
 */

/**
 * @test Exact condition/verifier rows cross native edges without unused capacity.
 *
 * Reverse physical order and reuse each retained graph twenty times. A known
 * row selected by a real kernel proves receive-before-compute ordering; guard
 * rows prove that the native collective does not copy the entire arena bank.
 */
TEST_P(PipelineMTPStateOwnership, CapturedVerifierEdgesTransferOnlyPhysicalRows)
{
    if (GetParam() == "CPU") return; // CPU pipeline transport is a separate host-owned protocol.
    initCPUBackend(-1);
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2) GTEST_SKIP() << "Requires two native GPU participants";
    for (const int participants : {2, 3})
    {
    if (participants > backend->deviceCount()) continue;
    SCOPED_TRACE(participants);
    for (const bool reverse : {false, true})
    {
        SCOPED_TRACE(reverse);
        std::vector<GlobalDeviceAddress> devices;
        std::vector<DevicePlanConfig> memory_inputs;
        ModelMemoryProfile profile;
        profile.architecture = "qwen35";
        profile.n_layers = 1;
        profile.d_model = profile.head_dim = profile.vocab_size = 32;
        profile.d_ff = 64;
        profile.n_heads = profile.n_kv_heads = 1;
        profile.max_seq_len = 64;
        for (int participant = 0; participant < participants; ++participant)
        {
            const int ordinal = reverse ? participants - 1 - participant : participant;
            const auto device = cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
            devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
            DevicePlanConfig input;
            input.world_rank = 0;
            input.device = device;
            input.device_total_bytes = input.device_free_bytes = 1ull << 30;
            input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
                .device = DeviceId::cpu(), .total_bytes = 1ull << 30, .admission_available_bytes = 1ull << 30};
            input.device_compute_units = 128;
            input.max_seq_len = input.activation_seq_len = 64;
            input.local_pipeline_backend = cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
            memory_inputs.push_back(input);
        }
        // Native capture fences use the same canonical admission as real PP.
        // The fixture does not allocate communicator scratch behind its ledger.
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(
                MemoryPlanner::plan(profile, memory_inputs).physicalPlan()), 0);
        LocalTPContext collective(devices, {}, CollectiveBackendType::AUTO);
        ASSERT_TRUE(collective.reserveGraphCaptureBoundaryResources(memory));
        TPWorkerPool workers(participants);
        workers.setFailureCallback([&] { collective.requestAbort(); });
        workers.dispatch([&](size_t participant) {
            const auto device = devices[participant].toLocalDeviceId();
            auto &context = GPUDeviceContextPool::instance().getContext(device);
            bool passed = false;
            context.submitAndWait([&] {
                void *const stream = context.defaultStream();
                auto execution = IDeviceContext::create(device, 1);
                ASSERT_NE(execution, nullptr);
                auto hidden = TestTensorFactory::createFP32Ones({32, 32});
                auto observation = TestTensorFactory::createFP32Ones({1, 32});
                ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
                ASSERT_TRUE(observation->ensureOnDevice(device, stream));
                TransferEngine::requireDeviceInput(hidden.get(), device, stream);
                TransferEngine::requireDeviceInput(observation.get(), device, stream);
                auto state_config = stateConfig(device);
                state_config.kv_cache_precision = KVCachePrecision::FP16;
                auto model = std::make_shared<Qwen35Graph>(state_config, nullptr);
                DeviceGraphOrchestrator state_owner(model, nullptr);
                ASSERT_TRUE(state_owner.initializeInferenceStateFromArena(1, 64, device));
                auto *cache = state_owner.inferenceState().kv_cache.get();
                ASSERT_NE(cache, nullptr);
                const size_t checkpoint_bytes = cache->deviceSequenceStateCheckpointBytes();
                const auto release = [backend, device](void *p) { if (p) backend->free(p, device.gpu_ordinal()); };
                std::unique_ptr<void, decltype(release)> checkpoint(
                    backend->allocate(checkpoint_bytes, device.gpu_ordinal()), release);
                ASSERT_TRUE(checkpoint);
                const PipelineForwardGraphEdges::FollowerState follower{
                    .backend = backend, .checkpoint = {.cache = cache, .sequence_index = 0,
                        .checkpoint_device = checkpoint.get(), .checkpoint_bytes = checkpoint_bytes}};
                PipelineForwardGraphEdges edges(collective, static_cast<int>(participant), *hidden, 32,
                    participant + 1 < static_cast<size_t>(participants) ? std::optional(follower) : std::nullopt);
                // Independent banks deliberately defeat a whole-struct memcpy
                // assumption. Guard words must survive every physical width.
                auto tokens = TestTensorFactory::createINT32({18});
                auto positions = TestTensorFactory::createINT32({18});
                auto lengths = TestTensorFactory::createINT32({3});
                for (auto *bank : {tokens.get(), positions.get(), lengths.get()})
                {
                    ASSERT_TRUE(bank->ensureOnDevice(device, stream));
                    TransferEngine::requireDeviceInput(bank, device, stream);
                }
                auto *const token_rows = static_cast<const int32_t *>(tokens->gpu_data_ptr());
                auto *const position_rows = static_cast<const int32_t *>(positions->gpu_data_ptr());
                auto *const real_rows = static_cast<const int32_t *>(lengths->gpu_data_ptr());
                for (const int width : {1, 2, 3, 4, 16})
                {
                    SCOPED_TRACE(width);
                    ForwardInput input;
                    input.device = device;
                    input.seq_len = width;
                    input.execution_role = width == 1 ? ForwardExecutionRole::MTPCondition
                                                      : ForwardExecutionRole::GroupedMTPVerifier;
                    input.execution_phase = ForwardExecutionPhase::Decode;
                    input.token_ids_device = token_rows;
                    input.position_ids_device = position_rows;
                    input.sequence_lengths_device = real_rows;
                    input.kv_cache = cache;
                    input.external_hidden_state = participant ? hidden.get() : nullptr;
                    HiddenStateRowSelectStage::Params select;
                    select.device_id = device;
                    select.input = hidden.get();
                    select.output = observation.get();
                    select.seq_len = width;
                    select.d_model = 32;
                    select.selected_row_idx = width - 1;
                    select.selection_policy = HiddenStateRowSelectStage::SelectionPolicy::DeviceResidentRequestLength;
                    select.request_sequence_length_device = real_rows;
                    select.input_buffer_id = BufferId::HIDDEN_STATE;
                    select.output_buffer_id = BufferId::LOGITS;
                    ComputeGraph graph;
                    graph.addNode("verifier_row_observation", std::make_unique<HiddenStateRowSelectStage>(select), device);
                    edges.append(graph, input);
                    const auto order = graph.getExecutionOrder();
                    for (const auto &name : order)
                    {
                        // The production executor binds the exact stream before
                        // calling optional launch preparation. A fixed-row
                        // compute stage legitimately has no preparation work.
                        graph.getNode(name)->stage->setGPUStream(stream);
                        ASSERT_TRUE(graph.getNode(name)->stage->prepareGraphLaunch(execution.get(), stream));
                    }
                    auto capture = context.createGraphCapture(stream);
                    ASSERT_NE(capture, nullptr);
                    ASSERT_TRUE(edges.captureBoundary("verifier_fixture_begin_" + std::to_string(width), stream));
                    ScopedBackendGraphCapture recording(context, *capture, "pipeline verifier activation edges");
                    ASSERT_TRUE(recording.begin());
                    bool recorded = true;
                    for (const auto &name : order)
                        recorded = graph.getNode(name)->stage->execute(execution.get()) && recorded;
                    recording.finish();
                    ASSERT_TRUE(recorded);
                    ASSERT_TRUE(edges.captureBoundary("verifier_fixture_end_" + std::to_string(width), stream));
                    ASSERT_TRUE(capture->instantiate());
                    for (int repeat = 0; repeat < 20; ++repeat)
                    {
                        SCOPED_TRACE(repeat);
                        const int active = repeat % width + 1;
                        std::array<int32_t, 18> token_values, position_values;
                        std::array<int32_t, 3> length_values;
                        token_values.fill(-11);
                        position_values.fill(-12);
                        length_values.fill(-13);
                        if (participant + 1 == static_cast<size_t>(participants))
                        {
                            for (int row = 0; row < width; ++row)
                            {
                                token_values[row] = repeat * 64 + row;
                                position_values[row] = repeat + row;
                            }
                            length_values[0] = active;
                        }
                        ASSERT_TRUE(backend->hostToDevice(tokens->gpu_data_ptr(), token_values.data(),
                            sizeof(token_values), device.gpu_ordinal(), stream));
                        ASSERT_TRUE(backend->hostToDevice(positions->gpu_data_ptr(), position_values.data(),
                            sizeof(position_values), device.gpu_ordinal(), stream));
                        ASSERT_TRUE(backend->hostToDevice(lengths->gpu_data_ptr(), length_values.data(),
                            sizeof(length_values), device.gpu_ordinal(), stream));
                        std::vector<float> source(32 * 32, -1000.0f - static_cast<float>(participant));
                        if (!participant)
                            for (size_t i = 0; i < source.size(); ++i)
                                source[i] = static_cast<float>(repeat * 1024 + i);
                        ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(), source.data(),
                            source.size() * sizeof(float), device.gpu_ordinal(), stream));
                        ASSERT_TRUE(cache->resetRequestState(IKVCache::StateResetContext::testReinitialization(stream)));
                        for (int layer = 0; layer < 3; ++layer)
                            ASSERT_TRUE(cache->appendWithStream(layer, 0, hidden.get(), hidden.get(),
                                1 + layer + repeat % 3 + static_cast<int>(participant), stream));
                        ASSERT_TRUE(capture->launch());
                        // No diagnostic wait separates the receive from its
                        // consumer. Observe only after the complete local DAG.
                        std::array<float, 32> selected{};
                        ASSERT_TRUE(backend->deviceToHost(selected.data(), observation->gpu_data_ptr(),
                            sizeof(selected), device.gpu_ordinal(), stream));
                        for (int column = 0; column < 32; ++column)
                            EXPECT_EQ(selected[column], static_cast<float>(repeat * 1024 + (active - 1) * 32 + column));
                        std::array<int32_t, 18> received_tokens, received_positions;
                        std::array<int32_t, 3> received_lengths;
                        ASSERT_TRUE(backend->deviceToHost(received_tokens.data(), tokens->gpu_data_ptr(),
                            sizeof(received_tokens), device.gpu_ordinal(), stream));
                        ASSERT_TRUE(backend->deviceToHost(received_positions.data(), positions->gpu_data_ptr(),
                            sizeof(received_positions), device.gpu_ordinal(), stream));
                        ASSERT_TRUE(backend->deviceToHost(received_lengths.data(), lengths->gpu_data_ptr(),
                            sizeof(received_lengths), device.gpu_ordinal(), stream));
                        for (int row = 0; row < 18; ++row)
                        {
                            EXPECT_EQ(received_tokens[row], row < width ? repeat * 64 + row : -11);
                            EXPECT_EQ(received_positions[row], row < width ? repeat + row : -12);
                        }
                        EXPECT_EQ(received_lengths, (std::array<int32_t, 3>{active, -13, -13}));
                        if (participant + 1 < static_cast<size_t>(participants) && width > 1)
                        {
                            // Restore from the checkpoint made inside the graph,
                            // after disturbing the cache. Every layer must recover
                            // its own prior frontier, not a replicated layer zero.
                            ASSERT_TRUE(cache->resetRequestState(IKVCache::StateResetContext::testReinitialization(stream)));
                            ASSERT_TRUE(cache->restoreDeviceSequenceStateCheckpoint(
                                0, checkpoint.get(), checkpoint_bytes, stream));
                            for (int layer = 0; layer < 3; ++layer)
                            {
                                int32_t count = -1;
                                ASSERT_TRUE(backend->deviceToHost(&count, cache->deviceCachedTokenCountPtr(layer, 0),
                                    sizeof(count), device.gpu_ordinal(), stream));
                                EXPECT_EQ(count, 1 + layer + repeat % 3 + static_cast<int>(participant));
                            }
                        }
                        std::vector<float> received(source.size());
                        ASSERT_TRUE(backend->deviceToHost(received.data(), hidden->gpu_data_ptr(),
                            received.size() * sizeof(float), device.gpu_ordinal(), stream));
                        for (size_t i = 0; i < received.size(); ++i)
                        {
                            const float expected = !participant || i < static_cast<size_t>(width * 32)
                                ? static_cast<float>(repeat * 1024 + i) : source[i];
                            EXPECT_EQ(received[i], expected) << "element=" << i;
                        }
                    }
                }
                passed = !::testing::Test::HasFailure();
            });
            return passed;
        });
        for (const auto &result : workers.collectAll(120000))
        {
            ASSERT_TRUE(result.completed);
            if (result.exception) std::rethrow_exception(result.exception);
            ASSERT_TRUE(result.success);
        }
    }
    }
}
