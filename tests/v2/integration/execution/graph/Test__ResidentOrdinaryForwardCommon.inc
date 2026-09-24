/**
 * @file Test__ResidentOrdinaryForwardCommon.inc
 * @brief Token-to-token integration of retained forward, KV state and resident sampling.
 *
 * Included after the shared graph-cache fixture inside its anonymous namespace.
 * Real production embedding, residual-add, KV-append and sampling kernels run
 * through the production forward engine and graph compiler. This is deliberately
 * a small model-free lifecycle oracle, not a real-model or HTTP certificate.
 * Every expected token comes from an independent exact uniform distribution;
 * the production seed-to-threshold law is shared, not the device sampler.
 */

/**
 * @test A returned token is the next embedding input, without a host token round trip.
 *
 * CUDA owns one native initialization/WHILE parent. HIP owns complete captured
 * forward-plus-sampler transactions, submitted only after authenticated immutable
 * tickets. Request-time changes retain addresses and graph identity. The final
 * output token, including EOS, must remain unconsumed by the canonical KV cache.
 */
TEST_F(PrefillGraphCacheExecutionTest, ResidentOrdinaryGenerationConsumesItsOwnTokensAndKV)
{
    using namespace sampling_math;
    using Sampler = OrdinaryGenerationSamplingStage;
    ScopedDebugEnv env({{"LLAMINAR_GPU_GRAPHS", "1"},
                        {"LLAMINAR_GPU_STAGE_TIMING", "0"},
                        {"LLAMINAR_GPU_STAGE_TIMING_DETAIL", "0"}});
    constexpr int vocab = kHiddenDim, capacity = 384, bank_words = 4096;
    embedding_table_ = std::make_unique<FP32Tensor>(std::vector<size_t>{vocab, vocab});
    auto *weights = embedding_table_->mutable_data();
    for (int input = 0; input < vocab; ++input)
        for (int token = 0; token < vocab; ++token)
        {
            // Each input selects a different four-token uniform distribution.
            // Binary fractions cancel the real residual producer exactly.
            const int relative = (token - (input + 1) + vocab) % vocab;
            const float logit = relative < 4 ? 4.0F : -16.0F;
            weights[input * vocab + token] = logit -
                (0.25F + static_cast<float>(token % 13) * 0.0625F);
        }
    ASSERT_TRUE(host_->bindEmbeddingTable(*embedding_table_));
    host_->enableDeferredDecodeProbe();
    host_->setUseKVAppendProbe(true);
    // The arena outlives the engine, including assertion-failure unwinding.
    // Views share one fixture allocation; each writable region is disjoint.
    auto *bank = graph_arena_.createPersistentTensor<INT32Tensor>(
        BufferId::MTP_GENERATION_CONTROL, std::vector<size_t>{bank_words},
        std::vector<int32_t>(bank_words, 0));
    auto *position_row = graph_arena_.createPersistentTensor<INT32Tensor>(
        BufferId::REQUEST_POSITION_IDS, std::vector<size_t>{1}, std::vector<int32_t>{-77});
    auto *worker = host_->getWorkerGPUContext(device_);
    ASSERT_NE(worker, nullptr);
    worker->submitAndWait([&] {
        void *const stream = worker->defaultStream();
        auto *const backend = getBackendFor(device_);
        ASSERT_NE(backend, nullptr);
        ASSERT_TRUE(host_->initializeKVCacheProbeForTesting());
        ASSERT_TRUE(host_->admitProbeResidual(stream));
        ASSERT_TRUE(host_->admitProbeKVInputs(stream));
        ASSERT_TRUE(embedding_table_->ensureOnDevice(device_, stream));
        TransferEngine::publishDeviceWrite(embedding_table_.get(), device_, stream);
        ASSERT_TRUE(bank->ensureOnDevice(device_, stream));
        TransferEngine::publishDeviceWrite(bank, device_, stream);
        ASSERT_TRUE(position_row->ensureOnDevice(device_, stream));
        TransferEngine::publishDeviceWrite(position_row, device_, stream);
        auto *const words = static_cast<int32_t *>(bank->gpu_data_ptr());
        auto *const seed = reinterpret_cast<uint64_t *>(words);
        auto *const control = words + 2;
        auto *const logical = control + kDeviceGenerationControlCount;
        auto *const condition = logical + 3;
        auto *const position = static_cast<int32_t *>(position_row->gpu_data_ptr());
        auto *const sample = logical + 7;
        auto *const response = sample + 1;
        auto *const history = response + capacity;
        auto *const stop = history + vocab;
        auto *const ticket = reinterpret_cast<DeviceGenerationDispatchTicket *>(stop + 1);
        auto *const values = reinterpret_cast<float *>(ticket + 1);
        auto *const indices = reinterpret_cast<int32_t *>(values + 256);
        auto *const partial_values = reinterpret_cast<float *>(indices + 256);
        auto *const partial_indices = reinterpret_cast<int32_t *>(partial_values + 1024);
        ASSERT_LE(partial_indices + 1024, words + bank_words);

        ForwardInput input;
        input.token_ids_device = condition;
        input.position_ids_device = position;
        input.position_offset = 1; // Graph geometry, never a live position shadow.
        input.batch_size = input.seq_len = input.real_seq_len = 1;
        input.device = device_;
        input.execution_phase = ForwardExecutionPhase::Decode;
        input.device_decode_position = {backend,
            host_->kvSequenceCachedTokensDeviceForTesting(), position};
        input.graph_submission_intent = ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;
        ForwardOutput output;
        std::optional<ForwardGraphSignature> signature;
        ASSERT_TRUE(engine_->execute(input, output, *host_, &signature));
        ASSERT_TRUE(signature.has_value());
        const auto first_signature = signature;
        // Setup hits and misses publish the same exact identity, without
        // reporting a forward execution or querying diagnostic observations.
        ASSERT_TRUE(engine_->execute(input, output, *host_, &signature));
        ASSERT_EQ(signature, first_signature);
        EXPECT_FALSE(engine_->lastExecutedForwardGraph().has_value());
        std::string error;
        const auto child = engine_->deviceLoopGraphTemplate(*signature, &error);
        ASSERT_TRUE(child.has_value()) << error;
        // Prompt construction is outside generation. It uses exactly the same
        // retained forward producer, rather than injecting fixture logits.
        auto prompt_forward = worker->createGraphCapture(stream);
        const GPUOrderedTimelineStep prompt_steps[] = {{.name = "real prompt producer", .capture = child->capture}};
        ASSERT_TRUE(prompt_forward && prompt_forward->buildOrderedTimelineTransaction(prompt_steps));
        ASSERT_TRUE(prompt_forward->instantiate());

        const GenerationLogicalStateRows rows{
            logical, logical + 1, logical + 2, condition, logical + 4, logical + 5, logical + 6};
        OrdinaryGenerationPublication publication{
            .request_count = 1, .sampled_tokens = sample, .stop_tokens = {stop, 1, 0},
            .source = OrdinaryGenerationSampleSource::PrefillLogits,
            .response_tokens = response, .response_token_stride = capacity,
            .control = control, .control_stride = kDeviceGenerationControlCount,
            .frontier = {logical + 1, condition, logical + 5, logical + 6, 1, 512},
            .history = {history, vocab, vocab, 1}};
        Sampler::Params params;
        params.device_id = device_;
        params.backend = backend;
        params.logits = static_cast<float *>(output.logits->gpu_data_ptr());
        params.vocab_size = params.logits_row_stride = vocab;
        params.workspace = {values, indices, 256, partial_values, partial_indices, 1024};
        params.publication = publication;
        const DeviceGenerationGraphControl graph_control{
            DeviceGenerationPolicy::ordinary(), control, kDeviceGenerationControlCount, 1};

        for (const bool stochastic : {false, true})
        {
            params.policy = stochastic ? std::variant<Sampler::Greedy, Sampler::Stochastic>(
                Sampler::Stochastic{4, 1.0F, 1.0F, seed}) : Sampler::Greedy{};
            params.publication.source = OrdinaryGenerationSampleSource::PrefillLogits;
            Sampler prefill(params);
            params.publication.source = OrdinaryGenerationSampleSource::DecodeLogits;
            Sampler decode(params);
            prefill.setGPUStream(stream);
            decode.setGPUStream(stream);
            auto prefill_graph = worker->createGraphCapture(stream);
            auto decode_graph = worker->createGraphCapture(stream);
            ASSERT_TRUE(prefill_graph && prefill_graph->beginCapture());
            ASSERT_TRUE(prefill.execute(device_ctx_.get()));
            ASSERT_TRUE(prefill_graph->endCapture());
            ASSERT_TRUE(decode_graph && decode_graph->beginCapture());
            ASSERT_TRUE(decode.execute(device_ctx_.get()));
            ASSERT_TRUE(decode_graph->endCapture());
            auto parent = worker->createGraphCapture(stream);
            auto continuation = worker->createGraphCapture(stream);
            auto ticket_publisher = worker->createGraphCapture(stream);
            ASSERT_TRUE(parent && continuation && ticket_publisher);
            if (device_.is_cuda())
            {
                const DeviceControlledLoopFragment initialization[] = {{
                    .name = "sample existing prefill logits", .capture = prefill_graph.get(),
                    .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordZero,
                    .condition_word_device = reinterpret_cast<const uint32_t *>(
                        control + kDeviceGenerationControlNextLeadingCommittedOutputCount)}};
                const DeviceControlledLoopFragment iteration[] = {
                    {.name = "consume resident condition", .capture = child->capture},
                    {.name = "sample new logits and publish", .capture = decode_graph.get()}};
                ASSERT_TRUE(DeviceGenerationGraphProgram::native(*parent, graph_control,
                    {.initialization = initialization, .iteration = iteration}, error)) << error;
            }
            else
            {
                // A complete retained transaction, not per-stage host dispatch.
                const GPUOrderedTimelineStep steps[] = {
                    {.name = "consume resident condition", .capture = child->capture},
                    {.name = "sample new logits and publish", .capture = decode_graph.get()}};
                ASSERT_TRUE(continuation->buildOrderedTimelineTransaction(steps));
                ASSERT_TRUE(continuation->instantiate());
                ASSERT_TRUE(prefill_graph->instantiate());
                ASSERT_TRUE(DeviceGenerationGraphProgram::ticketPublisher(*ticket_publisher,
                    *worker, *backend, device_, graph_control, nullptr, ticket, error)) << error;
            }

            for (int reset = 0; reset < 20; ++reset)
            {
                const int budget = std::array{1, 2, 17, capacity}[reset % 4];
                const int prefix = std::array{1, 7, 31}[reset % 3];
                const bool already_emitted = (reset / 4) % 2 != 0;
                const auto leading = already_emitted ? DeviceGenerationLeadingRowDisposition::AlreadyEmitted
                    : DeviceGenerationLeadingRowDisposition::PendingResponse;
                const uint64_t request_seed = 0x100000000ULL + reset + 19;
                const int32_t prompt_token = (reset * 11 + 5) % vocab;
                SCOPED_TRACE(::testing::Message() << "stochastic=" << stochastic << " reset=" << reset
                    << " budget=" << budget << " prefix=" << prefix << " leading=" << int(leading));
                ASSERT_TRUE(host_->resetKVCacheProbeForTesting(stream));
                ASSERT_TRUE(backend->memset(words, 0, bank_words * sizeof(int32_t), device_.gpu_ordinal(), stream));
                ASSERT_TRUE(backend->hostToDevice(condition, &prompt_token, sizeof(prompt_token), device_.gpu_ordinal(), stream));
                for (int row = 0; row < prefix; ++row)
                    ASSERT_TRUE(prompt_forward->launch());
                ASSERT_TRUE(backend->hostToDevice(seed, &request_seed, sizeof(request_seed), device_.gpu_ordinal(), stream));
                ASSERT_TRUE(backend->enqueueInitializeGenerationLogicalState({
                    .source = already_emitted ? GenerationInitialFrontier::SampledCondition
                                             : GenerationInitialFrontier::UnsampledLogits,
                    .positions = host_->kvSequenceCachedTokensDeviceForTesting(),
                    .sampled_tokens = already_emitted ? condition : nullptr,
                    .request_count = 1, .output = rows}, device_.gpu_ordinal(), stream));

                std::vector<int32_t> expected;
                int previous = prompt_token;
                for (int row = 0; row < budget; ++row)
                {
                    std::array<int, 4> support;
                    for (int i = 0; i < 4; ++i) support[i] = (previous + 1 + i) % vocab;
                    std::sort(support.begin(), support.end());
                    const float threshold = mtp_spec_threshold_from_seed(request_seed,
                        prefix + row + (already_emitted ? 1 : 0), 0);
                    const int index = stochastic ? std::min(3, static_cast<int>(threshold * 4)) : 0;
                    expected.push_back(support[index]);
                    previous = expected.back();
                }
                // Alternate no EOS, first-token EOS, and a later EOS. An
                // earlier recurrence must stop at its first occurrence.
                const int32_t stop_token = reset % 5 == 0 ? expected.front()
                    : reset % 5 == 1 ? expected[std::min<int>(5, expected.size() - 1)] : -1;
                if (const auto at = std::find(expected.begin(), expected.end(), stop_token); at != expected.end())
                    expected.erase(at + 1, expected.end());
                ASSERT_TRUE(backend->hostToDevice(stop, &stop_token, sizeof(stop_token), device_.gpu_ordinal(), stream));
                ASSERT_TRUE(backend->memset(response, 0xff, capacity * sizeof(int32_t), device_.gpu_ordinal(), stream));
                ASSERT_TRUE(backend->enqueueInitializeDeviceGeneration(1, budget,
                    DeviceGenerationPolicy::ordinary(), leading, capacity, response,
                    kDeviceGenerationControlCount, control, device_.gpu_ordinal(), stream));
                if (device_.is_cuda())
                {
                    ASSERT_TRUE(parent->launch());
                    ASSERT_TRUE(parent->launch()) << "Terminal replay must not consume the last output token";
                }
                else
                {
                    ASSERT_TRUE(backend->enqueueInitializeDeviceGenerationDispatchTicket(
                        reset + 1, 1, control, kDeviceGenerationControlCount, 1,
                        ticket, device_.gpu_ordinal(), stream));
                    ASSERT_TRUE(already_emitted ? continuation->launch() : prefill_graph->launch());
                    HostedDeviceGenerationCursor cursor;
                    ASSERT_TRUE(cursor.startScheduler());
                    for (int observation = 0; observation < budget; ++observation)
                    {
                        ASSERT_TRUE(cursor.markTicketObservationSubmitted());
                        ASSERT_TRUE(ticket_publisher->launch());
                        DeviceGenerationDispatchTicket observed;
                        ASSERT_TRUE(backend->deviceToHost(&observed, ticket, sizeof(observed), device_.gpu_ordinal(), stream));
                        ASSERT_TRUE(observed.matchesLifecycle(reset + 1, 1));
                        ASSERT_EQ(observed.healthy, 1);
                        ASSERT_EQ(observed.error_code, 0);
                        ASSERT_EQ(observed.next_draft_depth, 0);
                        ASSERT_TRUE(cursor.acceptTicket(observed.transaction_count));
                        ASSERT_TRUE(cursor.beginAdvance(observed.transaction_count));
                        if (!observed.complete) ASSERT_TRUE(continuation->launch());
                        ASSERT_TRUE(cursor.finishAdvance(observed.transaction_count, observed.complete != 0));
                        if (observed.complete) break;
                    }
                    ASSERT_TRUE(cursor.terminalSubmitted());
                    ASSERT_TRUE(cursor.retireCompleted());
                }
                // Only terminal evidence observes mutable model/controller data.
                std::array<int32_t, capacity> actual;
                std::array<int, kDeviceGenerationControlCount> terminal;
                std::array<int, vocab> actual_history{}, expected_history{};
                ASSERT_TRUE(backend->deviceToHost(actual.data(), response, sizeof(actual), device_.gpu_ordinal(), stream));
                ASSERT_TRUE(backend->deviceToHost(terminal.data(), control, sizeof(terminal), device_.gpu_ordinal(), stream));
                ASSERT_TRUE(backend->deviceToHost(actual_history.data(), history, sizeof(actual_history), device_.gpu_ordinal(), stream));
                for (size_t i = 0; i < expected.size(); ++i) {
                    ASSERT_EQ(actual[i], expected[i]) << "token=" << i;
                    ++expected_history[expected[i]];
                }
                for (size_t i = expected.size(); i < actual.size(); ++i) EXPECT_EQ(actual[i], -1);
                EXPECT_EQ(actual_history, expected_history);
                EXPECT_EQ(terminal[kDeviceGenerationControlResponseTokenCount], expected.size());
                EXPECT_EQ(validateDeviceGenerationTerminal(terminal,
                    {.request_count = 1, .max_new_tokens = budget,
                     .depth_policy = DeviceGenerationPolicy::ordinary(), .initial_leading_row_disposition = leading},
                    capacity, 0, 0), DeviceGenerationTerminalError::None);
                EXPECT_EQ(host_->kvDeviceCachedTokensForTesting(stream),
                    prefix + static_cast<int>(expected.size()) - (already_emitted ? 0 : 1));
                engine_->resetSessionReplayState(true);
                const auto retained = engine_->deviceLoopGraphTemplate(*signature, &error);
                ASSERT_TRUE(retained.has_value()) << error;
                EXPECT_EQ(retained->capture, child->capture);
                EXPECT_EQ(host_->build_calls, 1);
                EXPECT_EQ(host_->committed_forward_output_calls, 0);
            }
        }
    });
}
