/**
 * @file Test__PipelineMTPPublicationTransport.inc
 * @brief Real native collective proof of tail-authoritative accepted-state publication.
 *
 * The terminal's production publication kernel derives the committed fields;
 * NCCL/RCCL carries four independent words into each follower's captured local
 * restore. Guard words defeat full-buffer copies and overlapping-bank shortcuts.
 * The fixed transport is captured once; only the tail's local commit identity
 * changes between clipped and full publication. Followers must reuse their
 * native graph while the tail composes a new local parent, without rendezvous.
 * No host observation or event is inserted between producer, transfer and use.
 * This proves the publication transaction, not the remaining full MTP loop.
 */

/** @test Tail commit, native broadcast and local restore replay as one captured DAG. */
TEST_P(PipelineMTPStateOwnership, CapturedAcceptanceCrossesNativeParticipants)
{
    if (GetParam() == "CPU") return; // CPU authority does not use a GPU collective.
    using namespace sampling_math;
    using Publication = MTPSpeculativeStatePublicationStage;
    using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
    initCPUBackend(-1);
    const bool cuda = GetParam() == "CUDA";
    auto *backend = cuda ? getCUDABackend() : getROCmBackend();
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2) GTEST_SKIP() << "Requires two native GPU participants";
    for (const int participants : {2, 3})
    {
        if (participants > backend->deviceCount()) continue;
        for (const bool reverse : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "participants=" << participants << " reverse=" << reverse);
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
                const int ordinal = reverse ? participants - participant - 1 : participant;
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
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(
                    MemoryPlanner::plan(profile, memory_inputs).physicalPlan()), 0);
            LocalTPContext collective(devices, {}, CollectiveBackendType::AUTO);
            ASSERT_TRUE(collective.reserveGraphCaptureBoundaryResources(memory));
            TPWorkerPool workers(participants);
            workers.setFailureCallback([&] { collective.requestAbort(); });
            workers.dispatch([&](size_t participant) {
                const auto device = devices[participant].toLocalDeviceId();
                const bool tail = participant + 1 == static_cast<size_t>(participants);
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                bool passed = false;
                context.submitAndWait([&] {
                    void *const stream = context.defaultStream();
                    auto execution = IDeviceContext::create(device, 1);
                    ASSERT_NE(execution, nullptr);
                    auto config = stateConfig(device);
                    config.kv_cache_precision = KVCachePrecision::FP16;
                    config.mtp.enabled = true;
                    config.mtp.graph_capacity_draft_tokens = 15;
                    auto model = std::make_shared<Qwen35Graph>(config, nullptr);
                    auto owner_storage = std::make_unique<DeviceGraphOrchestrator>(model, nullptr);
                    auto &owner = *owner_storage;
                    owner.setPPStageConfig({.first_layer = 0, .last_layer = 3,
                        .has_embedding = participant == 0, .has_lm_head = tail});
                    ASSERT_TRUE(owner.initializeInferenceStateFromArena(1, 64, device));
                    IForwardExecutionHost &host = owner;
                    ASSERT_TRUE(host.ensureDeviceWorkspaceAllocated(ComputeGraph{}, 16));
                    auto *const cache = owner.inferenceState().kv_cache.get();
                    ASSERT_NE(cache, nullptr);
                    auto hidden = TestTensorFactory::createFP32Ones({16, 32});
                    ASSERT_TRUE(hidden->ensureOnDevice(device, stream));
                    TransferEngine::requireDeviceInput(hidden.get(), device, stream);
                    const auto follower = Peer::pipelineFollowerState(owner);
                    // Owners outlive all graphs and use unrelated allocations.
                    // Metadata goes in the middle word, leaving two guard words.
                    std::array<std::shared_ptr<INT32Tensor>, 8> banks;
                    for (auto &bank : banks)
                        bank = TestTensorFactory::createINT32({3});
                    auto outcome = TestTensorFactory::createINT32({32});
                    auto meta = TestTensorFactory::createINT32({kSpeculativeBatchMetaCount});
                    auto policy = TestTensorFactory::createINT32({sizeof(MTPGreedyPenaltyPolicy) / sizeof(int32_t)});
                    auto history = TestTensorFactory::createINT32({32});
                    std::vector<INT32Tensor *> storage;
                    for (const auto &bank : banks) storage.push_back(bank.get());
                    for (auto *bank : {outcome.get(), meta.get(), policy.get(), history.get()}) storage.push_back(bank);
                    for (auto *bank : storage)
                    {
                        ASSERT_TRUE(bank->ensureOnDevice(device, stream));
                        TransferEngine::requireDeviceInput(bank, device, stream);
                        ASSERT_TRUE(backend->memset(bank->gpu_data_ptr(), 0, bank->numel() * sizeof(int32_t),
                            device.gpu_ordinal(), stream));
                    }
                    const auto word = [&](size_t i) { return static_cast<int32_t *>(banks[i]->gpu_data_ptr()) + 1; };
                    PipelineForwardGraphEdges edges(collective, static_cast<int>(participant), *hidden, 32,
                        tail ? std::nullopt : std::optional(follower));
                    ComputeGraph verifier;
                    const auto retire = [&](void *) {
                        if (!owner_storage) return;
                        Peer::retirePublication(owner);
                        Peer::retirePublicationTransport(owner);
                        Peer::bindPipelineEdges(owner, nullptr);
                    };
                    std::unique_ptr<void, decltype(retire)> publication_lifetime(&edges, retire);
                    Peer::bindPipelineEdges(owner, &edges);
                    const PipelineForwardGraphEdges::PublicationBanks wire{word(0), word(1), word(2), word(3)};
                    const std::array<int32_t, 3> setup_guard{-301, -400 - static_cast<int>(participant), -303};
                    for (size_t field = 0; field < 4; ++field)
                        ASSERT_TRUE(backend->hostToDevice(banks[field]->gpu_data_ptr(), setup_guard.data(),
                            sizeof(setup_guard), device.gpu_ordinal(), stream));
                    ASSERT_TRUE(Peer::capturePublicationTransport(owner, edges, wire, *execution, context));
                    const auto *const retained_wire = edges.publicationTransport(wire);
                    ASSERT_NE(retained_wire, nullptr);
                    // A second preparation is local reuse, not another collective wave.
                    ASSERT_TRUE(Peer::capturePublicationTransport(owner, edges, wire, *execution, context));
                    EXPECT_EQ(edges.publicationTransport(wire), retained_wire);
                    for (size_t field = 0; field < 4; ++field)
                    {
                        std::array<int32_t, 3> observed{};
                        ASSERT_TRUE(backend->deviceToHost(observed.data(), banks[field]->gpu_data_ptr(),
                            sizeof(observed), device.gpu_ordinal(), stream));
                        EXPECT_EQ(observed, setup_guard) << "Serving setup must not launch its retained broadcast";
                    }
                    auto foreign_wire = wire;
                    foreign_wire.healthy = word(7);
                    EXPECT_THROW(Peer::capturePublicationTransport(owner, edges, foreign_wire, *execution, context), std::logic_error);
                    const IGPUGraphCapture *previous_publication = nullptr;
                    for (const int width : {2, 3, 4, 16})
                    for (const int limit : {width, width - 1})
                    {
                        SCOPED_TRACE(::testing::Message() << "width=" << width << " limit=" << limit);
                        MTPVerifierPreparationStage::Params preparation;
                        preparation.device_id = device;
                        preparation.backend = backend;
                        preparation.authority = MTPVerifierPreparationStage::Authority::PipelineFollower;
                        preparation.request_count = 1;
                        preparation.padded_seq_len = width;
                        preparation.main_kv_checkpoints = std::span(&follower.checkpoint, 1);
                        MTPVerifierPreparationStage prepare(preparation);
                        prepare.setGPUStream(stream);
                        Publication::Params params;
                        params.device_id = device;
                        params.backend = backend;
                        params.authority = tail ? Publication::Authority::CompactOutcome : Publication::Authority::PipelineFollower;
                        params.request_count = 1;
                        params.verifier_rows_per_request = width;
                        params.accepted_restore_rows_device = word(0);
                        params.target_cached_tokens_device = word(1);
                        params.accepted_state_counts_device = word(2);
                        params.publication_ok_flags_device = word(3);
                        if (tail)
                        {
                            params.outcome_tokens_device = static_cast<int32_t *>(outcome->gpu_data_ptr());
                            params.outcome_meta_device = static_cast<int *>(meta->gpu_data_ptr());
                            params.outcome_token_stride = 32;
                            params.outcome_meta_stride = kSpeculativeBatchMetaCount;
                            params.base_cached_tokens_device = word(4);
                            params.next_condition_tokens_device = word(5);
                            params.all_drafts_accepted_flags_device = word(6);
                            params.stopped_flags_device = word(7);
                            params.next_verifier_condition_tokens_device = static_cast<int32_t *>(outcome->gpu_data_ptr()) + 31;
                            params.max_state_commit_rows = limit;
                            params.penalty_policy_device = policy->gpu_data_ptr();
                            params.generated_token_counts_device = static_cast<int32_t *>(history->gpu_data_ptr());
                            params.vocab_size = 32;
                        }
                        std::string error;
                        ASSERT_TRUE(Peer::installPublication(owner, params, verifier, &error)) << error;
                        ASSERT_TRUE(Peer::capturePublication(owner, stream, &error)) << error;
                        const auto *publication = Peer::publicationCapture(owner);
                        ASSERT_NE(publication, nullptr);
                        if (!tail && limit == width - 1)
                            EXPECT_EQ(publication, previous_publication) << "A tail-only limit must not recapture followers";
                        previous_publication = publication;
                        EXPECT_EQ(edges.publicationTransport(wire), retained_wire);
                        // Record synthetic verifier writes, then compose the real
                        // production publication parent after them. The collective
                        // is never recorded inside these policy-specific captures.
                        auto verifier_capture = context.createGraphCapture(stream);
                        ASSERT_NE(verifier_capture, nullptr);
                        ScopedBackendGraphCapture recording(context, *verifier_capture, "pipeline verifier state writes");
                        ASSERT_TRUE(recording.begin());
                        bool recorded = prepare.execute(execution.get());
                        for (int layer = 0; layer < 3; ++layer)
                            recorded = cache->appendWithStream(layer, 0, hidden.get(), hidden.get(), width, stream) && recorded;
                        recording.finish();
                        ASSERT_TRUE(recorded);
                        auto capture = context.createGraphCapture(stream);
                        ASSERT_NE(capture, nullptr);
                        const std::array steps{
                            GPUOrderedTimelineStep{.name = "verifier writes", .capture = verifier_capture.get()},
                            GPUOrderedTimelineStep{.name = "production publication", .capture = publication}};
                        ASSERT_TRUE(capture->buildOrderedTimelineTransaction(steps));
                        ASSERT_TRUE(capture->instantiate());
                        for (int repeat = 0; repeat < 20; ++repeat)
                        {
                            SCOPED_TRACE(repeat);
                            const int base = 1 + repeat % 7;
                            const int proposed = repeat % (width + 1);
                            const bool healthy = repeat != 19;
                            const int accepted = healthy ? std::min(proposed, limit) : 0;
                            ASSERT_TRUE(cache->resetRequestState(IKVCache::StateResetContext::testReinitialization(stream)));
                            for (int layer = 0; layer < 3; ++layer)
                                ASSERT_TRUE(cache->appendWithStream(layer, 0, hidden.get(), hidden.get(), base, stream));
                            for (size_t i = 0; i < banks.size(); ++i)
                            {
                                // Distinct participant guards expose an oversized
                                // broadcast even if the copied metadata is correct.
                                const int32_t guard = 10 * static_cast<int32_t>(participant) + static_cast<int32_t>(i);
                                const std::array<int32_t, 3> values{-101 - guard, tail && i == 4 ? base : -999, -103 - guard};
                                ASSERT_TRUE(backend->hostToDevice(banks[i]->gpu_data_ptr(), values.data(), sizeof(values),
                                    device.gpu_ordinal(), stream));
                            }
                            if (tail)
                            {
                                std::array<int32_t, kSpeculativeBatchMetaCount> values{};
                                values[kSpecBatchMetaOk] = healthy ? 1 : 0;
                                values[kSpecBatchMetaTargetVerifierStateCommitCount] = proposed;
                                values[kSpecBatchMetaOutputCount] = proposed;
                                values[kSpecBatchMetaStoppedOnOutput] = repeat % 3 == 0;
                                values[kSpecBatchMetaAllSpeculativeAccepted] = proposed == width;
                                ASSERT_TRUE(backend->hostToDevice(meta->gpu_data_ptr(), values.data(), sizeof(values),
                                    device.gpu_ordinal(), stream));
                            }
                            ASSERT_TRUE(capture->launch());
                            // Terminal diagnostics only: a wrong dependency must
                            // be visible as poisoned metadata or the wrong KV frontier.
                            const std::array<int32_t, 4> expected{
                                accepted - 1, base + accepted, accepted, healthy ? 1 : 0};
                            for (size_t i = 0; i < 4; ++i)
                            {
                                std::array<int32_t, 3> observed{};
                                ASSERT_TRUE(backend->deviceToHost(observed.data(), banks[i]->gpu_data_ptr(), sizeof(observed),
                                    device.gpu_ordinal(), stream));
                                const int32_t guard = 10 * static_cast<int32_t>(participant) + static_cast<int32_t>(i);
                                EXPECT_EQ(observed, (std::array<int32_t, 3>{-101 - guard, expected[i], -103 - guard}));
                            }
                            for (int layer = 0; layer < 3; ++layer)
                            {
                                int32_t count = -1;
                                ASSERT_TRUE(backend->deviceToHost(&count, cache->deviceCachedTokenCountPtr(layer, 0),
                                    sizeof(count), device.gpu_ordinal(), stream));
                                EXPECT_EQ(count, base + (healthy ? accepted : width)) << "layer=" << layer;
                            }
                        }
                    }
                    // Real DGO teardown owns both parents and the wire lease;
                    // the surviving rank topology must not retain arena borrowers.
                    owner_storage.reset();
                    EXPECT_EQ(edges.publicationTransport(wire), nullptr);
                    DeviceGraphExecutor unused_executor;
                    EXPECT_THROW(edges.materializePublicationTransport(wire, unused_executor, *execution, context), std::logic_error)
                        << "A retired participant cannot resurrect capture against an old arena";
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
