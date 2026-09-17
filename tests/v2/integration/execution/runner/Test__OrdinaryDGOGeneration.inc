/**
 * @file Test__OrdinaryDGOGeneration.inc
 * @brief Exercise ordinary generation through DGO admission, graph and terminal APIs.
 *
 * The small graph uses production embedding and KV-append kernels. It isolates
 * lifecycle from transformer arithmetic; expected tokens are independently
 * generated from an exact four-token uniform distribution. This is model-free
 * preflight evidence, not HTTP or real-model numerical certification.
 */

/** @brief Observe retained graph construction without enabling stage profiling. */
class OrdinaryGraphEvidence final
{
public:
    /** @brief Save and install the narrowly scoped diagnostic collection policy. */
    OrdinaryGraphEvidence()
    {
        for (const char *key : {"LLAMINAR_PERF_STATS_SUMMARY", "LLAMINAR_PERF_STATS_FILTER"})
            previous.emplace_back(key, std::getenv(key) ? std::optional<std::string>(std::getenv(key)) : std::nullopt);
        setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
        setenv("LLAMINAR_PERF_STATS_FILTER", "generation,forward_graph", 1);
        PerfStatsCollector::reset();
    }
    /** @brief Restore the process policy after every assertion/exception exit. */
    ~OrdinaryGraphEvidence()
    {
        for (const auto &[key, value] : previous)
            if (value) setenv(key.c_str(), value->c_str(), 1); else unsetenv(key.c_str());
        PerfStatsCollector::reset();
    }
    /** @return Number of parent constructions, never an inference decision input. */
    double materializations(const char *name = "ordinary_graph_materializations") const
    {
        double count = 0;
        for (const auto &record : PerfStatsCollector::snapshot())
            if (record.domain == "generation" && record.name == name)
                count += record.value;
        return count;
    }
    /** @return Terminal-observed forwards executed inside a complete generation parent.
     * The first prefill sample consumes no model row. Evidence must count real
     * model executions, not sampler transactions or unlaunched setup graphs. */
    double completedForwards(DeviceId device) const
    {
        double count = 0;
        for (const auto &record : PerfStatsCollector::snapshot())
            if (record.domain == "forward_graph" && record.name == "decode_graph_phase" &&
                record.device == device.toString() &&
                record.tags.contains("context") && record.tags.at("context") == "main_decode" &&
                record.tags.contains("phase") && record.tags.at("phase") == "replay")
                count += record.value;
        return count;
    }
private:
    std::vector<std::pair<std::string, std::optional<std::string>>> previous;
};

/** @brief A token-dependent numerical graph with actual canonical KV advancement. */
class OrdinaryDGOGraph final : public Qwen35Graph
{
public:
    /** @brief Own immutable weight bytes before any graph borrows their address. */
    explicit OrdinaryDGOGraph(const GraphConfig &config, bool embedding = true, bool head = true)
        : Qwen35Graph(config, nullptr), table({32, 32}), owns_embedding(embedding), owns_head(head)
    {
        auto *data = table.mutable_data();
        for (int input = 0; input < 32; ++input)
            for (int token = 0; token < 32; ++token)
                data[input * 32 + token] = (token - input - 1 + 32) % 32 < 4 ? 4.0F : -16.0F;
    }
    /** @return This model-free graph owns its only immutable weight. */
    bool isInitialized() const override { return true; }
    /** @brief Bind actual resident token input, logits and KV append in one DAG.
     * @param input Exact production invocation, never a host token shadow.
     * @param output Receives the arena-owned terminal logits.
     * @return Participant-local graph with no mocked kernels or callbacks. */
    ComputeGraph buildForwardGraph(const ForwardInput &input, ForwardOutput &output) override
    {
        if (input.seq_len <= 0 || input.batch_size != 1 || !input.kv_cache)
            throw std::invalid_argument("Ordinary DGO probe requires one canonical KV row");
        ++builds;
        EmbeddingStage::Params embedding;
        embedding.device_id = input.device;
        embedding.embed_table = &table;
        embedding.token_ids = input.token_ids;
        embedding.token_ids_device = input.token_ids_device;
        embedding.output = buffers_.current_hidden;
        embedding.num_tokens = input.seq_len;
        embedding.d_model = embedding.vocab_size = 32;
        embedding.output_buffer_id = BufferId::HIDDEN_STATE;
        KVCacheAppendStage::Params append;
        append.device_id = input.device;
        append.K = append.V = buffers_.current_hidden;
        append.kv_cache = input.kv_cache;
        append.layer_idx = append.seq_idx = 0;
        append.num_tokens = append.seq_len = input.seq_len;
        append.batch_size = 1;
        // A scalar condition's length bank contains its absolute KV position,
        // not an append extent. Grouped verification/prefill use real row counts.
        append.request_sequence_lengths_device = input.execution_role == ForwardExecutionRole::MTPCondition
            ? nullptr : input.sequence_lengths_device;
        if (input.execution_role == ForwardExecutionRole::GroupedMTPVerifier)
            append.append_semantics = KVCacheAppendSemantics::DecodeEquivalentVerifier;
        append.head_dim = 32;
        append.k_buffer_id = append.v_buffer_id = BufferId::HIDDEN_STATE;
        // Use the real bucketed LM-head selection policy: padded rows may
        // execute embedding, but only the resident real length owns KV and
        // the terminal logits row. No fixed row or host length replaces it.
        HiddenStateRowSelectStage::Params select;
        select.device_id = input.device;
        select.input = buffers_.current_hidden;
        select.output = buffers_.logits;
        select.seq_len = input.seq_len;
        select.d_model = 32;
        select.input_buffer_id = BufferId::HIDDEN_STATE;
        select.output_buffer_id = BufferId::LOGITS;
        select.selected_row_idx = input.seq_len - 1;
        select.selection_policy = input.sequence_lengths_device && input.execution_role != ForwardExecutionRole::MTPCondition
            ? HiddenStateRowSelectStage::SelectionPolicy::DeviceResidentRequestLength
            : HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow;
        select.request_sequence_length_device = input.sequence_lengths_device;
        ComputeGraph graph;
        if (owns_embedding)
            graph.addNode("ordinary_embedding", std::make_unique<EmbeddingStage>(embedding), input.device);
        graph.addNode("ordinary_kv_append", std::make_unique<KVCacheAppendStage>(append), input.device);
        if (owns_embedding)
            graph.addDependency("ordinary_kv_append", "ordinary_embedding");
        if (owns_head)
        {
            graph.addNode("ordinary_terminal_row", std::make_unique<HiddenStateRowSelectStage>(select), input.device);
            graph.addDependency("ordinary_terminal_row", "ordinary_kv_append");
        }
        output.logits = owns_head ? buffers_.logits : nullptr;
        output.hidden = buffers_.current_hidden;
        return graph;
    }
    /** @brief Full graph setup has the same exact probe, not an alternate implementation. */
    ComputeGraph buildFullForwardGraph(const ForwardInput &input, ForwardOutput &output) override
    {
        return buildForwardGraph(input, output);
    }
    /**
     * @brief Declare this fixture's real shifted-KV sidecar transaction.
     *
     * The pipeline MTP-condition regression does not execute a learned draft;
     * it does, however, retain the production terminal predictor family while
     * it captures the main-condition graph.  A normal Qwen sidecar would need
     * a complete transformer weight set, which would obscure this lifecycle
     * proof with model arithmetic.  The probe therefore declares the exact
     * stateful primitive relevant here: append the terminal hidden rows to the
     * terminal-owned shifted KV cache.  The sidecar remains a real compute
     * graph and an actual KV stage, so the workspace/capture contract cannot
     * be satisfied by an empty mock graph.
     *
     * @param depth_idx Predictor depth requested by the retained family.
     * @param bindings Frozen marker proving that the terminal owns a sidecar
     *        binding; this tiny numerical graph has no learned projection.
     * @param input Production MTP sidecar input and shifted cache owner.
     * @param output Unused because this declaration only owns shifted KV rows.
     * @return One stateful production KV-append stage, or an empty graph for
     *         an invalid sidecar contract.
     */
    ComputeGraph buildMTPGraph(
        int depth_idx,
        const MTPDepthWeightBindings &bindings,
        const MTPForwardInput &input,
        MTPForwardOutput &output) override
    {
        (void)depth_idx;
        (void)bindings;
        (void)output;
        ComputeGraph graph;
        if (!input.kv_cache || !input.terminal_hidden ||
            input.batch_size <= 0 || input.seq_len <= 0)
            return graph;

        KVCacheAppendStage::Params append;
        append.device_id = input.device;
        append.K = append.V = input.terminal_hidden;
        append.kv_cache = input.kv_cache;
        append.layer_idx = append.seq_idx = 0;
        append.num_tokens = input.batch_size * input.seq_len;
        append.seq_len = input.seq_len;
        append.batch_size = input.batch_size;
        append.head_dim = 32;
        append.k_buffer_id = append.v_buffer_id = BufferId::MTP_HIDDEN;
        graph.addNode("ordinary_mtp_shifted_kv_append",
            std::make_unique<KVCacheAppendStage>(append), input.device);
        return graph;
    }
    /** @brief Honor the production PP entrypoint with this fixture's owned stages.
     * @param input Exact invocation, including the participant-local hidden bank.
     * @param output Receives this stage's hidden/logit publications.
     * @param first_layer First fixture layer (one local KV layer).
     * @param last_layer Exclusive fixture layer bound.
     * @param embedding Must match the entry stage selected at construction.
     * @param head Must match the vocabulary owner selected at construction.
     * @return The same real embedding/KV/row-selection kernels, scoped to this stage. */
    ComputeGraph buildPartialForwardGraph(const ForwardInput &input, ForwardOutput &output,
        int first_layer, int last_layer, bool embedding, bool head) override
    {
        if (first_layer != 0 || last_layer != 1 || embedding != owns_embedding || head != owns_head)
            throw std::invalid_argument("Ordinary pipeline fixture role differs from the runner plan");
        return buildForwardGraph(input, output);
    }
    FP32Tensor table; ///< Outlives the runner's retained embedding graph.
    int builds = 0; ///< Setup-only evidence of cache reuse, not inference authority.
    bool owns_embedding; ///< Only the pipeline entry embeds the tail's token.
    bool owns_head; ///< Only the tail produces vocabulary logits for sampling.
};

/**
 * @brief Freeze the terminal-only sidecar binding required by the MTP family.
 *
 * The ordinary graph overrides sidecar arithmetic above, so one immutable
 * marker tensor is sufficient.  Routing the marker through the same frozen
 * authority used by production is intentional: a terminal graph family must
 * never infer sidecar ownership from an incidental cache pointer.
 *
 * @param graph Fixture that owns the tensor for every retained graph lifetime.
 * @param resident_device Participant on which the terminal graph is captured.
 * @return Frozen graph-time authority with one depth-zero sidecar binding.
 */
std::unique_ptr<FrozenModelWeightSet> makeOrdinaryMTPFrozenWeightSet(
    OrdinaryDGOGraph &graph, DeviceId resident_device)
{
    InferenceStrategy strategy;
    strategy.mode = WeightInferenceMode::SingleDevice;
    strategy.devices.push_back(resident_device);

    ModelWeightSetBuilder builder(strategy);
    WeightBinding binding;
    binding.identity.canonical_name = "mtp.fc.weight";
    binding.identity.role = WeightRole::Other;
    binding.identity.logical_id = stableWeightLogicalId(
        binding.identity.canonical_name);
    binding.residency.home_device = DeviceId::cpu();
    binding.residency.resident_device = resident_device;
    binding.tensor = &graph.table;
    binding.immutable = true;
    builder.addBinding(std::move(binding));

    auto frozen = std::make_unique<FrozenModelWeightSet>(
        strategy, builder.freezeBindings());
    frozen->validateForGraph();
    return frozen;
}

/**
 * @brief Independently verify every archived prompt KV element at terminal observation.
 * @param rank Production pipeline owner after generation and continuation finish.
 * @param prompt Exact request tokens, including a changed partial-hit suffix.
 * @param cached_tokens Expected live prompt plus consumed generated history.
 * @param participants Number of independently owned pipeline stage caches.
 *
 * This toy model appends its known embedding to K and V. Checking only tokens
 * would miss a broken cache import because its logits do not read old KV.
 * Inspect only after generation: an earlier diagnostic wait could hide a
 * missing restore-to-generation event edge.
 */
void expectOrdinaryPipelinePromptKV(IInferenceRunner &rank, const std::vector<int> &prompt,
    int cached_tokens, size_t participants)
{
    std::vector<uint16_t> expected(prompt.size() * 32);
    for (size_t row = 0; row < prompt.size(); ++row)
        for (int column = 0; column < 32; ++column)
            expected[row * 32 + column] = fp32_to_fp16(
                (column - prompt[row] - 1 + 32) % 32 < 4 ? 4.0F : -16.0F);
    const auto snapshot = rank.captureLivePrefixState();
    ASSERT_TRUE(snapshot.valid);
    ASSERT_EQ(snapshot.cached_tokens, cached_tokens);
    ASSERT_EQ(snapshot.participant_snapshots.size(), participants);
    for (const auto &participant : snapshot.participant_snapshots)
    {
        ASSERT_EQ(participant.blocks.size(), 1u);
        const auto &block = participant.blocks.front();
        ASSERT_EQ(block.layout.fa_layers, 1);
        ASSERT_EQ(block.layout.local_kv_heads, 1);
        ASSERT_EQ(block.layout.head_dim, 32);
        ASSERT_EQ(block.layout.k_precision, ActivationPrecision::FP16);
        ASSERT_EQ(block.layout.v_precision, ActivationPrecision::FP16);
        const size_t bytes = expected.size() * sizeof(uint16_t);
        ASSERT_GE(block.layout.bytes_per_fa_layer_k, bytes);
        ASSERT_GE(block.layout.bytes_per_fa_layer_v, bytes);
        ASSERT_NE(block.kvKData(), nullptr);
        ASSERT_NE(block.kvVData(), nullptr);
        EXPECT_EQ(std::memcmp(block.kvKData(), expected.data(), bytes), 0);
        EXPECT_EQ(std::memcmp(block.kvVData(), expected.data(), bytes), 0);
    }
}

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
