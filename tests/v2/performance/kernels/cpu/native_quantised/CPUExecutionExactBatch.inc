/**
 * @file CPUExecutionExactBatch.inc
 * @brief Compatible exact requests through one prepared fused CPU fixture.
 *
 * Weights, expert-local source rows, serial oracles and maximum admitted tensor
 * storage are prepared once. Request boundaries bind only tight live prefixes;
 * no capacity tail enters the production kernel. A timing team spans requests.
 * Profiling reuses the native worker pool through the existing counter owner.
 */

/** @brief One immutable request; source/geometry/worker contract belongs to the batch. */
struct CPUExecutionExactBatchRequest
{
    std::string id, purpose_name, candidate_name, evidence_path;
    std::string profile_request_id, profile_output_path;
    int m = 0, target_descriptor = 0;
    std::vector<int> expert_rows;
    ProjectionRowsPurpose purpose = ProjectionRowsPurpose::Decode;
    VerifierRowsPolicy candidate = VerifierRowsPolicy::Auto;
    bool holdout = false;
};

/**
 * @brief Run the actual fused owner for compatible, independently retained requests.
 * @param format Canonical source/encoding registry entry selected by GTest.
 * @param request_path Immutable request manifest; no per-request environment mutation.
 */
void runCPUExecutionExactBatch(const QuantizedVerifierFormatCase &format, const char *request_path)
{
    std::ifstream request_stream(request_path);
    ASSERT_TRUE(request_stream.is_open());
    const auto manifest = nlohmann::json::parse(request_stream);
    const std::string tournament_mode = std::getenv("LLAMINAR_NATIVE_QUANTISED_TOURNAMENT")
        ? std::getenv("LLAMINAR_NATIVE_QUANTISED_TOURNAMENT") : "";
    ASSERT_TRUE(tournament_mode.empty() || tournament_mode == "adaptive_v1");
    const bool tournament = tournament_mode == "adaptive_v1";
    ASSERT_EQ(manifest.at("schema").get<std::string>(),
        tournament ? "cpu_execution_exact_batch_v2" : "cpu_execution_exact_batch_v1");
    ASSERT_TRUE(manifest.at("requests").is_array()); ASSERT_FALSE(manifest.at("requests").empty());
    const int n = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_A", 320);
    const int k = dimension("LLAMINAR_CPU_FUSED_VERIFIER_K", 2560);
    const int policy_n = dimension("LLAMINAR_CPU_FUSED_VERIFIER_POLICY_N", n);
    const int warmup = dimension("LLAMINAR_CPU_FUSED_VERIFIER_WARMUP", 4);
    const int samples = dimension("LLAMINAR_CPU_FUSED_VERIFIER_SAMPLES", 12);
    const int repeats = dimension("LLAMINAR_CPU_FUSED_VERIFIER_REPEATS", 8);
    const std::string role = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_OUTPUT_ROLE")
        ? std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_OUTPUT_ROLE") : "projection";
    ASSERT_TRUE(role == "gate_up" || role == "down" || role == "projection");
    if (const char *numerical = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_NUMERICAL_POLICY"))
        ASSERT_STREQ(numerical, "GPUAlignedExpert");
    const bool paired = role == "gate_up";
    const size_t roots = paired ? 2u : 1u;
    const std::string team = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_TEAM_LIFETIME")
        ? std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_TEAM_LIFETIME") : "retained";
    ASSERT_TRUE(team == "retained" || team == "standalone");
    ASSERT_NO_THROW(requireQuantizedVerifierSourceShape(format, n, k));
    const std::string shape = manifest.value("shape", std::string(std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_SHAPE")
        ? std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_EXACT_SHAPE") : ""));
    ASSERT_FALSE(shape.empty());
    std::vector<CPUExecutionExactBatchRequest> requests;
    std::vector<int> maximum_expert_rows;
    std::set<std::string> request_ids, output_paths;
    std::optional<bool> profile_batch;
    for (const auto &item : manifest.at("requests"))
    {
        CPUExecutionExactBatchRequest request;
        request.id = item.at("id").get<std::string>();
        request.m = item.at("m").get<int>();
        request.purpose_name = item.at("purpose").get<std::string>();
        request.candidate_name = item.at("candidate").get<std::string>();
        request.target_descriptor = item.at("target_descriptor").get<int>();
        request.expert_rows = item.at("expert_rows").get<std::vector<int>>();
        request.evidence_path = item.at("evidence_path").get<std::string>();
        request.holdout = item.value("holdout", false);
        request.profile_request_id = item.value("profile_request_id", std::string{});
        request.profile_output_path = item.value("profile_output_path", std::string{});
        ASSERT_FALSE(request.id.empty()); ASSERT_TRUE(request_ids.insert(request.id).second);
        ASSERT_FALSE(request.evidence_path.empty()); ASSERT_TRUE(output_paths.insert(request.evidence_path).second);
        ASSERT_GT(request.m, 1); ASSERT_FALSE(request.expert_rows.empty());
        ASSERT_LE(request.expert_rows.size() * roots, NativeQuantisedFusedRowsPlan::capacity);
        ASSERT_FALSE(std::ifstream(request.evidence_path).good()) << "Exact evidence path already exists";
        ASSERT_GE(request.target_descriptor, 0);
        ASSERT_LT(static_cast<size_t>(request.target_descriptor), request.expert_rows.size() * roots);
        ASSERT_EQ(request.expert_rows[static_cast<size_t>(request.target_descriptor) / roots], request.m);
        if (request.purpose_name == "decode") request.purpose = ProjectionRowsPurpose::Decode;
        else if (request.purpose_name == "mtp_verifier") request.purpose = ProjectionRowsPurpose::MTPVerifier;
        else { ASSERT_EQ(request.purpose_name, "prefill"); request.purpose = ProjectionRowsPurpose::Prefill; }
        if (request.candidate_name == "Auto") request.candidate = VerifierRowsPolicy::Auto;
        else if (request.candidate_name == "Pairwise") request.candidate = VerifierRowsPolicy::Pairwise;
        else { ASSERT_EQ(request.candidate_name, "WideRows"); request.candidate = VerifierRowsPolicy::WideRows; }
        const bool profiling = !request.profile_request_id.empty();
        ASSERT_EQ(profiling, !request.profile_output_path.empty());
        if (profile_batch) ASSERT_EQ(*profile_batch, profiling);
        else profile_batch = profiling;
        maximum_expert_rows.resize(std::max(maximum_expert_rows.size(), request.expert_rows.size()), 0);
        for (size_t expert = 0; expert < request.expert_rows.size(); ++expert)
        {
            ASSERT_GT(request.expert_rows[expert], 0);
            maximum_expert_rows[expert] = std::max(maximum_expert_rows[expert], request.expert_rows[expert]);
        }
        requests.push_back(std::move(request));
    }
    if (tournament)
    {
        ASSERT_FALSE(*profile_batch) << "Isolated profiler/Auto proof uses the original single-call protocol";
        ASSERT_EQ(team, "retained") << "Adaptive expert tournament requires its declared persistent team";
        ASSERT_GE(warmup, 5) << "Adaptive finalists require the original canonical warmup floor";
        ASSERT_GE(samples, 30) << "Adaptive finalists require the original canonical sample floor";
        for (const auto& request : requests)
            ASSERT_NE(request.candidate, VerifierRowsPolicy::Auto) << "Auto is observed baseline, never a tuning candidate";
        ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"))
            << "Tournament needs its actual unforced Auto route observation";
    }
    const size_t maximum_projections = maximum_expert_rows.size() * roots;
    ASSERT_LE(maximum_projections, NativeQuantisedFusedRowsPlan::capacity);
    int workers = 0; std::set<int> allowed_physical_cores; bool disjoint_masks = false;
    captureCPUFusedPlacement(workers, allowed_physical_cores, disjoint_masks);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    std::vector<size_t> oracle_offsets(maximum_expert_rows.size());
    size_t maximum_total_rows = 0;
    for (size_t expert = 0; expert < maximum_expert_rows.size(); ++expert)
    {
        oracle_offsets[expert] = maximum_total_rows;
        maximum_total_rows += static_cast<size_t>(maximum_expert_rows[expert]);
    }
    ASSERT_LE(maximum_total_rows, static_cast<size_t>(std::numeric_limits<int>::max()));
    ASSERT_LE(maximum_total_rows, std::numeric_limits<size_t>::max() / static_cast<size_t>(std::max(n, k)));
    constexpr float guard = -12345.0f;
    ASSERT_LE(n, std::numeric_limits<int>::max() - 7);
    const int blocks = k / 32, stride = n + 7;
    std::vector<float> original_inputs(maximum_total_rows * static_cast<size_t>(k));
    for (size_t index = 0; index < original_inputs.size(); ++index)
        original_inputs[index] = float(int((index * 37 + 13) % 257) - 128) * 0.003125f;
    AlignedVector<Q8_1Block> oracle_inputs(maximum_total_rows * blocks), live_inputs(maximum_total_rows * blocks);
    quantize_activations_to_q8_1(original_inputs.data(), oracle_inputs.data(), static_cast<int>(maximum_total_rows),
        k, blocks, CPUProjectionNumericalPolicy::GPUAlignedExpert);
    std::vector<std::unique_ptr<TensorBase>> sources;
    std::vector<std::unique_ptr<CPUNativeQuantisedGemmKernel>> kernels;
    std::vector<AlignedVector<float>> outputs;
    std::vector<std::vector<float>> guarded(maximum_projections), serial(maximum_projections);
    std::uint64_t weight_bytes = 0;
    sources.reserve(maximum_projections); kernels.reserve(maximum_projections); outputs.reserve(roots);
    for (size_t root = 0; root < roots; ++root) outputs.emplace_back(maximum_total_rows * n, guard);
    for (size_t projection = 0; projection < maximum_projections; ++projection)
    {
        sources.push_back(format.create({static_cast<size_t>(n), static_cast<size_t>(k)}, 4242 + projection));
        ASSERT_NE(sources.back(), nullptr);
        kernels.push_back(std::make_unique<CPUNativeQuantisedGemmKernel>(sources.back().get(), 0, -1,
            CPUProjectionNumericalPolicy::GPUAlignedExpert));
        ASSERT_TRUE(kernels.back()->isValid());
        const auto &packed = kernels.back()->packedWeights();
        ASSERT_EQ(packed.codebook_id, format.source_codebook_id);
        ASSERT_EQ(packed.is_superblock, format.source_is_superblock);
        weight_bytes += packed.native_interleaved.size() + packed.payload.size() + packed.int8_flat.size();
        const size_t expert = projection / roots;
        guarded[projection].assign(static_cast<size_t>(maximum_expert_rows[expert]) * stride + 64u, guard);
        serial[projection].resize(static_cast<size_t>(maximum_expert_rows[expert]) * n);
        if (!*profile_batch)
        {
            NativeQuantisedTestPartialStorage partial(packed, 1);
            for (int row = 0; row < maximum_expert_rows[expert]; ++row)
                gemv_native_quantised_preq(packed, oracle_inputs.data() + (oracle_offsets[expert] + row) * blocks,
                    serial[projection].data() + static_cast<size_t>(row) * n, partial.span());
        }
    }
    // All tensor/vector capacity is owned before the persistent team. Requests
    // publish only live prefixes and cannot grow any of these buffers.
    std::vector<FusedVerifierRowsDesc> descriptors(maximum_projections), guard_descriptors(maximum_projections);
    std::vector<CPUExecutionExactKey> keys(maximum_projections);
    std::vector<uint32_t> baseline_policies(maximum_projections), selected_policies(maximum_projections);
    std::vector<size_t> live_offsets(maximum_projections);
    std::vector<double> timings(static_cast<size_t>(samples));
    std::vector<float> flat_actual, flat_serial;
    flat_actual.reserve(maximum_total_rows * n * roots); flat_serial.reserve(maximum_total_rows * n * roots);
    NativeQuantisedFusedRowsPlan plan;
    CPUExecutionExactKey target_key;
    size_t live_projections = 0, live_rows = 0;
    int maximum_rows = 1;
    bool key_verified = false, stop = false;
    std::uint64_t sample_observations = 0;
    std::array<size_t, 2> live_root_floats{};
    const CPUExecutionExactBatchRequest *current = nullptr;
    const auto policy_id = [](const FusedVerifierRowsPlan &member) -> uint32_t
    {
        if (member.rows == 1) return static_cast<uint32_t>(member.effective_decode_schedule);
        if (member.effective_verifier_schedule == VerifierRowsPolicy::Pairwise) return 0;
        if (member.effective_verifier_schedule == VerifierRowsPolicy::WideRows) return 1;
        throw std::runtime_error("Exact filled K-part class cannot emit a foreign physical row policy");
    };
    const auto bind_request = [&](bool refresh_inputs = true)
    {
        if (refresh_inputs)
        {
        live_projections = current->expert_rows.size() * roots;
        live_rows = 0; maximum_rows = 1; live_root_floats = {};
        for (auto &root : outputs) std::fill(root.begin(), root.end(), guard);
        for (size_t expert = 0; expert < current->expert_rows.size(); ++expert)
        {
            const int rows = current->expert_rows[expert];
            maximum_rows = std::max(maximum_rows, rows);
            std::memcpy(live_inputs.data() + live_rows * blocks, oracle_inputs.data() + oracle_offsets[expert] * blocks,
                static_cast<size_t>(rows) * blocks * sizeof(Q8_1Block));
            for (size_t root = 0; root < roots; ++root)
            {
                const size_t projection = expert * roots + root;
                live_offsets[projection] = live_rows;
                descriptors[projection] = {.packed = &kernels[projection]->packedWeights(),
                    .output = outputs[root].data() + live_root_floats[root], .N = n, .ldc = n, .rows = rows,
                    .input = live_inputs.data() + live_rows * blocks, .purpose = current->purpose, .policy_n = policy_n};
                live_root_floats[root] += static_cast<size_t>(rows) * n;
                std::fill(guarded[projection].begin(), guarded[projection].end(), guard);
                guard_descriptors[projection] = descriptors[projection];
                guard_descriptors[projection].output = guarded[projection].data();
                guard_descriptors[projection].ldc = stride;
            }
            live_rows += static_cast<size_t>(rows);
        }
        }
        for (size_t member = 0; member < live_projections; ++member)
        {
            descriptors[member].verifier_schedule = VerifierRowsPolicy::Auto;
            guard_descriptors[member].verifier_schedule = VerifierRowsPolicy::Auto;
        }
        const auto live = std::span<const FusedVerifierRowsDesc>(descriptors.data(), live_projections);
        const auto baseline = planNativeQuantisedFusedRows(live, maximum_rows, workers);
        ASSERT_EQ(baseline.execution_exact_class, CPUExecutionExactBundleClass::FilledKPartOutputTileLocal);
        for (size_t member = 0; member < live_projections; ++member)
        {
            keys[member] = cpuExecutionExactKey(descriptors[member], baseline.projections[member], workers,
                baseline.use_avx512, baseline.execution_exact_class);
            baseline_policies[member] = policy_id(baseline.projections[member]);
        }
        target_key = keys[current->target_descriptor];
        for (size_t member = 0; member < live_projections; ++member)
        {
            if (keys[member] == target_key && current->candidate != VerifierRowsPolicy::Auto)
                descriptors[member].verifier_schedule = current->candidate;
            guard_descriptors[member].verifier_schedule = descriptors[member].verifier_schedule;
        }
        plan = planNativeQuantisedFusedRows(live, maximum_rows, workers);
        key_verified = plan.execution_exact_class == CPUExecutionExactBundleClass::FilledKPartOutputTileLocal &&
            plan.partial_route == NativeQuantisedFusedPartialRoute::OutputTileLocal;
        ASSERT_EQ(plan.sharedPartialFloats(), 0u);
        for (size_t member = 0; member < live_projections; ++member)
        {
            key_verified = key_verified && keys[member] == cpuExecutionExactKey(descriptors[member],
                plan.projections[member], workers, plan.use_avx512, plan.execution_exact_class);
            selected_policies[member] = policy_id(plan.projections[member]);
            if (keys[member] != target_key) ASSERT_EQ(selected_policies[member], baseline_policies[member]);
        }
        ASSERT_TRUE(key_verified);
        PerfStatsCollector::reset();
    };
    const auto run = [&]()
    {
        if (!gemm_native_quantised_fused_verifier_rows_preq(live_inputs.data(), descriptors.data(), {},
                static_cast<int>(live_projections), maximum_rows, blocks))
            throw std::runtime_error("Exact batch rejected a prepared live fused request");
    };
    const auto run_guarded = [&]()
    {
        if (!gemm_native_quantised_fused_verifier_rows_preq(live_inputs.data(), guard_descriptors.data(), {},
                static_cast<int>(live_projections), maximum_rows, blocks))
            throw std::runtime_error("Exact batch rejected guarded fused replay");
    };
    const auto poison_guards = [&]()
    {
        for (size_t member = 0; member < live_projections; ++member)
            std::fill(guarded[member].begin(), guarded[member].end(), guard);
    };
    const auto check_bytes = [&]()
    {
        for (size_t member = 0; member < live_projections; ++member)
        {
            const auto &descriptor = descriptors[member];
            for (int row = 0; row < descriptor.rows; ++row)
            {
                ASSERT_EQ(std::memcmp(descriptor.output + static_cast<size_t>(row) * n,
                    serial[member].data() + static_cast<size_t>(row) * n, static_cast<size_t>(n) * sizeof(float)), 0);
                ASSERT_EQ(std::memcmp(guarded[member].data() + static_cast<size_t>(row) * stride,
                    serial[member].data() + static_cast<size_t>(row) * n, static_cast<size_t>(n) * sizeof(float)), 0);
            }
            for (size_t index = 0; index < guarded[member].size(); ++index)
                if (index >= static_cast<size_t>(descriptor.rows) * stride || index % stride >= static_cast<size_t>(n))
                    ASSERT_EQ(guarded[member][index], guard);
        }
        for (size_t root = 0; root < roots; ++root)
            for (size_t index = live_root_floats[root]; index < outputs[root].size(); ++index)
                ASSERT_EQ(outputs[root][index], guard);
    };
    const auto sample_count = [&]()
    {
        sample_observations = 0;
        for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_native_quantised_fused_verifier_rows_projection_launch"}))
            sample_observations += record.count;
        if (PerfStatsCollector::isDomainEnabled("kernel"))
            ASSERT_EQ(sample_observations, (2u + static_cast<uint64_t>(warmup) + static_cast<uint64_t>(samples) * repeats) * live_projections);
    };
    nlohmann::json tournament_receipt;
    std::vector<double> current_screen_samples;
    std::string current_timing_stage, current_finalist_reason;
    bool current_promotion_grade = false;
    std::uint64_t current_timed_projection_count = 0;
    const auto emit = [&](bool profiling)
    {
        nlohmann::json descriptor_keys = nlohmann::json::array();
        std::vector<uint32_t> baseline(baseline_policies.begin(), baseline_policies.begin() + live_projections);
        std::vector<uint32_t> selected(selected_policies.begin(), selected_policies.begin() + live_projections);
        for (size_t member = 0; member < live_projections; ++member) descriptor_keys.push_back(keys[member].scalars());
        uint64_t live_weight_bytes = 0;
        for (size_t member = 0; member < live_projections; ++member)
        {
            const auto &packed = kernels[member]->packedWeights();
            live_weight_bytes += packed.native_interleaved.size() + packed.payload.size() + packed.int8_flat.size();
        }
        std::vector<int> member_rows;
        member_rows.reserve(live_projections);
        for (size_t member = 0; member < live_projections; ++member) member_rows.push_back(descriptors[member].rows);
        nlohmann::json evidence{{"schema", profiling ? "cpu_execution_exact_profile_v1" : tournament ? "cpu_execution_exact_native_v2" : "cpu_execution_exact_native_v1"},
            {"request_id", current->id}, {"shape", shape}, {"exact_key", target_key.scalars()},
            {"descriptor_keys", descriptor_keys}, {"descriptor_policies", selected}, {"baseline_descriptor_policies", baseline},
            {"target_descriptor", current->target_descriptor}, {"candidate", current->candidate_name},
            {"exact_key_verified", key_verified}, {"holdout", current->holdout}, {"source_format", format.label},
            {"source_superblock", format.source_is_superblock}, {"prepared_codebook", kernels[0]->packedWeights().codebook_id},
            {"prepared_encoding", static_cast<int>(kernels[0]->packedWeights().encoding)},
            {"numerical_policy", "GPUAlignedExpert"}, {"purpose", current->purpose_name}, {"weight_bytes", live_weight_bytes},
            {"retained_prepared_weight_bytes", weight_bytes}, {"partial_route", "OutputTileLocal"},
            {"partial_route_id", static_cast<int>(plan.partial_route)}, {"bundle_class", static_cast<uint32_t>(plan.execution_exact_class)},
            {"team_lifetime", team}, {"output_layout", "aligned_expert_major_v1"},
            {"activation_layout", "aligned_tightly_packed_expert_rows_v1"}, {"expert_rows", current->expert_rows},
            {"member_rows", member_rows},
            {"activation_row_offsets", std::vector<size_t>(live_offsets.begin(), live_offsets.begin() + live_projections)},
            {"output_role", role}, {"placement", {{"workers", workers}, {"admitted_physical_cores", allowed_physical_cores.size()},
                {"disjoint_single_core_masks", disjoint_masks}}}};
        if (profiling)
        {
            evidence["profile_request_id"] = current->profile_request_id;
            evidence["profiled_calls"] = 1; evidence["preconditioning_calls"] = 2;
            nlohmann::json routes = nlohmann::json::array(); uint64_t observations = 0;
            for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_native_quantised_fused_verifier_rows_projection_launch"}))
            {
                observations += record.count;
                routes.push_back({{"tags", record.tags}, {"count", record.count}, {"value", record.value}});
            }
            if (PerfStatsCollector::isDomainEnabled("kernel")) ASSERT_EQ(observations, live_projections);
            evidence["counters"] = {{"enabled", PerfStatsCollector::isDomainEnabled("kernel")},
                {"projection_observations", observations}, {"expected_projection_observations", live_projections},
                {"route_records", routes}};
        }
        else
        {
            flat_actual.clear(); flat_serial.clear();
            for (size_t member = 0; member < live_projections; ++member)
                for (int row = 0; row < descriptors[member].rows; ++row)
                {
                    const float *actual = descriptors[member].output + static_cast<size_t>(row) * n;
                    const float *expected = serial[member].data() + static_cast<size_t>(row) * n;
                    flat_actual.insert(flat_actual.end(), actual, actual + n);
                    flat_serial.insert(flat_serial.end(), expected, expected + n);
                }
            const auto bytes = trainer::compareFP32(flat_actual, flat_serial, n); ASSERT_TRUE(bytes.bitwiseEqual());
            auto sorted = timings; std::sort(sorted.begin(), sorted.end());
            const auto aggregate = trainer::summarizeSortedTimingSamples(sorted);
            evidence["raw_samples_us"] = timings; evidence["sorted_samples_us"] = sorted;
            evidence["timing"] = {{"sample_count", timings.size()},
                {"repeat_count", tournament && !current_promotion_grade ? 1 : repeats},
                {"warmup_count", tournament && !current_promotion_grade ? 0 : warmup},
                {"min_us", aggregate.min}, {"median_us", aggregate.median}, {"p95_us", aggregate.p95},
                {"mad_us", aggregate.mad}, {"cv", aggregate.cv}, {"sorted_native_digest", aggregate.digest}};
            evidence["correctness"] = {{"bitwise_equal", bytes.bitwiseEqual()}, {"mismatch_count", bytes.mismatch_count},
                {"nonfinite_count", bytes.nonfinite_count}, {"actual_digest", bytes.actual_digest},
                {"expected_digest", bytes.expected_digest}, {"max_abs", bytes.max_abs}, {"relative_l2", bytes.relative_l2},
                {"cosine", bytes.cosine}, {"symmetric_kld", bytes.symmetric_kld}, {"guard_replays", tournament ? 4 : 2}, {"repeated_replay", true}};
            nlohmann::json routes = nlohmann::json::array(); uint64_t authenticated = 0, hits = 0;
            for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_native_quantised_fused_verifier_rows_projection_launch"}))
            {
                authenticated += record.count; routes.push_back({{"tags", record.tags}, {"count", record.count}, {"value", record.value}});
            }
            for (const auto &record : PerfStatsCollector::snapshot({"kernel.cpu_native_quantised_execution_exact_policy_hits"})) hits += record.count;
            if (PerfStatsCollector::isDomainEnabled("kernel")) ASSERT_EQ(authenticated, live_projections);
            evidence["counters"] = {{"enabled", PerfStatsCollector::isDomainEnabled("kernel")},
                {"projection_observations", sample_observations}, {"expected_projection_observations",
                    tournament ? current_timed_projection_count :
                    (2u + static_cast<uint64_t>(warmup) + static_cast<uint64_t>(samples) * repeats) * live_projections},
                {"authenticated_projection_observations", authenticated}, {"execution_exact_hits", hits},
                {"route_records", routes}, {"authentication_calls", 1}, {"authentication_outside_timing", true}};
        }
        if (tournament)
        {
            evidence["timing_stage"] = current_timing_stage;
            evidence["promotion_grade"] = current_promotion_grade;
            evidence["finalist_reason"] = current_finalist_reason;
            evidence["screen_samples_us"] = current_screen_samples;
            evidence["screen_repeat_count"] = 1;
            evidence["configured_full_sample_count"] = samples;
            evidence["configured_full_repeat_count"] = repeats;
            evidence["tournament"] = tournament_receipt;
            ASSERT_EQ(sample_observations, current_timed_projection_count);
            evidence["counters"]["sample_observation_basis"] = "screen_and_final_timed_calls_only";
        }
        std::ofstream output(current->evidence_path); ASSERT_TRUE(output.is_open());
        output << evidence.dump(2) << '\n'; output.flush(); ASSERT_TRUE(output.good());
        std::cout << "execution_exact_request_complete,id=" << current->id << ",profile=" << profiling << '\n';
    };
    if (*profile_batch)
    {
        // Existing perf control owns concurrent per-TID arm/disarm. Calling it
        // inside an active team would incorrectly capture a disabled nested team.
        std::optional<native_quantised_dispatch::LinuxPerfControl> control;
        for (const auto &request : requests)
        {
            current = &request;
            const auto launch = [&]()
            {
                if (team == "retained")
                {
#pragma omp parallel
                    { run(); }
                }
                else run();
            };
            if (team == "retained")
            {
#pragma omp parallel
                {
#pragma omp master
                    { bind_request(); }
#pragma omp barrier
                    if (!::testing::Test::HasFatalFailure()) for (int iteration = 0; iteration < 2; ++iteration) run();
                }
            }
            else { bind_request(); for (int iteration = 0; iteration < 2; ++iteration) run(); }
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            if (!control) control.emplace(request.profile_request_id, request.profile_output_path);
            PerfStatsCollector::reset();
            native_quantised_dispatch::profileExactOpenMPRegion(*control, request.profile_request_id, request.profile_output_path, launch);
            emit(true); ASSERT_FALSE(::testing::Test::HasFatalFailure());
        }
        return;
    }
    if (tournament)
    {
        // Only candidate policy varies inside a contest; all live tensor
        // addresses and per-expert row ranges are rebound once before screening.
        std::vector<std::vector<size_t>> contests;
        std::map<std::string, size_t> contest_indices;
        for (size_t index = 0; index < requests.size(); ++index)
        {
            const auto& request = requests[index];
            const auto identity = nlohmann::json{{"m", request.m}, {"purpose", request.purpose_name},
                {"expert_rows", request.expert_rows}, {"target", request.target_descriptor},
                {"holdout", request.holdout}}.dump();
            auto [found, inserted] = contest_indices.emplace(identity, contests.size());
            if (inserted) contests.emplace_back();
            contests[found->second].push_back(index);
        }
        std::stable_sort(contests.begin(), contests.end(), [&](const auto& left, const auto& right) {
            const auto& a = requests[left.front()]; const auto& b = requests[right.front()];
            return std::tie(a.purpose_name, a.m, a.holdout, a.expert_rows, a.target_descriptor) <
                   std::tie(b.purpose_name, b.m, b.holdout, b.expert_rows, b.target_descriptor);
        });
        std::vector<std::vector<double>> screens(requests.size(), std::vector<double>(3));
        std::vector<std::vector<double>> finals(requests.size(), std::vector<double>(samples));
        std::vector<uint64_t> timed_counts(requests.size(), 0);
        std::map<std::pair<std::array<uint32_t, 18>, int>, std::string> anchor_winners;
        const std::string row_policy = std::getenv("LLAMINAR_NATIVE_QUANTISED_PREFILL_ROW_POLICY")
            ? std::getenv("LLAMINAR_NATIVE_QUANTISED_PREFILL_ROW_POLICY") : "";
        ASSERT_TRUE(row_policy.empty() || row_policy == "two_anchor_v1");
        trainer::TournamentSelection selection;
        size_t baseline_index = 0;
        std::chrono::steady_clock::time_point start;
        std::vector<nlohmann::json> baseline_routes(contests.size());
        const auto verify_live_bytes = [&]() {
            for (size_t member = 0; member < live_projections; ++member)
                for (int row = 0; row < descriptors[member].rows; ++row)
                    ASSERT_EQ(std::memcmp(descriptors[member].output + static_cast<size_t>(row) * n,
                        serial[member].data() + static_cast<size_t>(row) * n, static_cast<size_t>(n) * sizeof(float)), 0);
        };
        const auto observe_timed_calls = [&](size_t index, int count) {
            uint64_t observed = 0;
            for (const auto& record : PerfStatsCollector::snapshot({"kernel.cpu_native_quantised_fused_verifier_rows_projection_launch"}))
                observed += record.count;
            ASSERT_EQ(observed, static_cast<uint64_t>(count) * live_projections);
            timed_counts[index] += observed;
        };
        // One team spans preparation, every candidate's byte checks, rotating
        // screening, full finalist timing and separately authenticated output.
#pragma omp parallel
        {
            for (size_t contest = 0; contest < contests.size(); ++contest)
            {
                const auto& members = contests[contest];
#pragma omp master
                {
                    current = &requests[members.front()]; bind_request();
                    for (size_t member = 0; member < live_projections; ++member)
                        descriptors[member].verifier_schedule = VerifierRowsPolicy::Auto;
                    stop = ::testing::Test::HasFatalFailure();
                }
#pragma omp barrier
                if (stop) break;
                run(); // Actual unforced Auto, outside correctness/timing clocks.
#pragma omp master
                {
                    const auto authenticate_baseline = [&]() {
                        verify_live_bytes();
                        const auto baseline_name = baseline_policies[current->target_descriptor] == 0 ? "Pairwise" : "WideRows";
                        uint64_t observed = 0; baseline_routes[contest] = nlohmann::json::array();
                        for (const auto& record : PerfStatsCollector::snapshot({"kernel.cpu_native_quantised_fused_verifier_rows_projection_launch"}))
                        {
                            observed += record.count;
                            baseline_routes[contest].push_back({{"tags", record.tags}, {"count", record.count}, {"value", record.value}});
                            if (record.tags.at("m") == std::to_string(current->m))
                            {
                                ASSERT_EQ(record.tags.at("requested_verifier_policy"), "Auto");
                                ASSERT_EQ(record.tags.at("effective_verifier_policy"), baseline_name);
                            }
                        }
                        ASSERT_EQ(observed, live_projections);
                        baseline_index = members.size();
                        std::set<std::string> roster;
                        for (size_t position = 0; position < members.size(); ++position)
                        {
                            const auto& candidate = requests[members[position]].candidate_name;
                            ASSERT_TRUE(roster.insert(candidate).second);
                            if (candidate == baseline_name) baseline_index = position;
                        }
                        ASSERT_LT(baseline_index, members.size()) << "Actual Auto baseline missing from forceable roster";
                        ASSERT_TRUE(roster.count("Pairwise"));
                        ASSERT_EQ(roster.count("WideRows"), static_cast<size_t>(plan.use_avx512 && current->m >= 3));
                    };
                    authenticate_baseline(); stop = ::testing::Test::HasFatalFailure();
                }
#pragma omp barrier
                if (stop) break;
                // Full byte/guard/repeat admission for every supported candidate.
                for (const auto index : members)
                {
#pragma omp master
                    { current = &requests[index]; bind_request(false); poison_guards(); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                    if (stop) break;
                    run_guarded(); run();
#pragma omp master
                    { check_bytes(); poison_guards(); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                    if (stop) break;
                    run_guarded(); run();
#pragma omp master
                    { check_bytes(); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                    if (stop) break;
                }
                if (stop) break;
                for (size_t round = 0; round < 3 && !stop; ++round)
                    for (size_t offset = 0; offset < members.size(); ++offset)
                    {
                        const size_t index = members[(round + offset) % members.size()];
#pragma omp master
                        { current = &requests[index]; bind_request(false); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                        if (stop) break;
#pragma omp master
                        { start = std::chrono::steady_clock::now(); }
#pragma omp barrier
                        run();
#pragma omp barrier
#pragma omp master
                        {
                            screens[index][round] = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - start).count();
                            observe_timed_calls(index, 1); stop = ::testing::Test::HasFatalFailure();
                        }
#pragma omp barrier
                        if (stop) break;
                    }
                if (stop) break;
#pragma omp master
                {
                    const auto select = [&]() {
                        std::vector<trainer::TournamentCandidateScreen> observations;
                        for (const auto index : members)
                            observations.push_back({requests[index].candidate_name, true, screens[index]});
                        selection = trainer::selectTournamentFinalists(observations, baseline_index);
                        if (row_policy == "two_anchor_v1" && current->purpose == ProjectionRowsPurpose::Prefill &&
                            (current->m == 256 || current->m == 1024))
                        {
                            auto domain = target_key.scalars(); domain[9] = 0;
                            const auto found = anchor_winners.find({domain, current->m == 256 ? 128 : 512});
                            ASSERT_NE(found, anchor_winners.end()) << "Prefill spot lacks its actually timed same-class anchor";
                            size_t anchor = members.size();
                            for (size_t position = 0; position < members.size(); ++position)
                                if (requests[members[position]].candidate_name == found->second) anchor = position;
                            ASSERT_LT(anchor, members.size());
                            selection.finalists[anchor] = true;
                            if (selection.reasons[anchor] == "screened_out") selection.reasons[anchor].clear();
                            if (!selection.reasons[anchor].empty()) selection.reasons[anchor] += '|';
                            selection.reasons[anchor] += "row_regime_anchor";
                        }
                    };
                    select(); stop = ::testing::Test::HasFatalFailure();
                }
#pragma omp barrier
                if (stop) break;
                for (int64_t round = 0; round < static_cast<int64_t>(warmup) + samples && !stop; ++round)
                    for (size_t offset = 0; offset < members.size(); ++offset)
                    {
                        const size_t position = (static_cast<size_t>(round) + offset) % members.size();
                        if (!selection.finalists[position]) continue;
                        const size_t index = members[position];
#pragma omp master
                        { current = &requests[index]; bind_request(false); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                        if (stop) break;
#pragma omp master
                        { start = std::chrono::steady_clock::now(); }
#pragma omp barrier
                        for (int repeat = 0; repeat < (round < warmup ? 1 : repeats); ++repeat) run();
#pragma omp barrier
#pragma omp master
                        {
                            const double elapsed = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - start).count();
                            if (round >= warmup)
                            {
                                finals[index][round - warmup] = elapsed / repeats;
                                observe_timed_calls(index, repeats);
                            }
                            stop = ::testing::Test::HasFatalFailure();
                        }
#pragma omp barrier
                        if (stop) break;
                    }
                if (stop) break;
#pragma omp master
                {
                    const auto finish_receipt = [&]() {
                        size_t winner = members.size(); double best = std::numeric_limits<double>::infinity();
                        tournament_receipt = {{"schema", "native_quantised_tournament_v1"},
                            {"backend", "cpu"}, {"source_format", format.label}, {"shape", shape},
                            {"purpose", current->purpose_name}, {"m", current->m}, {"n", n}, {"k", k},
                            {"threads", workers}, {"build_isa", static_cast<uint32_t>(target_key.build_isa)},
                            {"runtime_isa", static_cast<uint32_t>(target_key.runtime_isa)}, {"holdout", current->holdout},
                            {"expert_rows", current->expert_rows}, {"target_descriptor", current->target_descriptor},
                            {"candidate_ids", nlohmann::json::array()}, {"finalist_candidate_ids", nlohmann::json::array()},
                            {"baseline_candidate_id", requests[members[baseline_index]].candidate_name},
                            {"baseline_auto_calls", 1}, {"baseline_auto_route_records", baseline_routes[contest]},
                            {"exact_key", target_key.scalars()}, {"candidates", nlohmann::json::array()}};
                        for (size_t position = 0; position < members.size(); ++position)
                        {
                            const size_t index = members[position];
                            auto sorted_screen = screens[index]; std::sort(sorted_screen.begin(), sorted_screen.end());
                            tournament_receipt["candidate_ids"].push_back(requests[index].candidate_name);
                            if (selection.finalists[position])
                            {
                                tournament_receipt["finalist_candidate_ids"].push_back(requests[index].candidate_name);
                                auto sorted = finals[index]; std::sort(sorted.begin(), sorted.end());
                                const auto aggregate = trainer::summarizeSortedTimingSamples(sorted);
                                if (aggregate.median < best) { best = aggregate.median; winner = position; }
                            }
                            tournament_receipt["candidates"].push_back({{"candidate_id", requests[index].candidate_name},
                                {"correctness_pass", true}, {"screen_samples_us", screens[index]},
                                {"screen_sorted_native_digest", trainer::summarizeSortedTimingSamples(sorted_screen).digest},
                                {"timing_stage", selection.finalists[position] ? "final" : "screening"},
                                {"promotion_grade", static_cast<bool>(selection.finalists[position])}, {"finalist_reason", selection.reasons[position]},
                                {"final_sample_count", selection.finalists[position] ? samples : 0}});
                        }
                        ASSERT_LT(winner, members.size());
                        tournament_receipt["winner_candidate_id"] = requests[members[winner]].candidate_name;
                        if (row_policy == "two_anchor_v1" && !current->holdout &&
                            current->purpose == ProjectionRowsPurpose::Prefill && (current->m == 128 || current->m == 512))
                        {
                            auto domain = target_key.scalars(); domain[9] = 0;
                            ASSERT_TRUE(anchor_winners.emplace(std::make_pair(domain, current->m),
                                requests[members[winner]].candidate_name).second)
                                << "Duplicate training anchor contests cannot define one row-regime control";
                        }
                    };
                    finish_receipt(); stop = ::testing::Test::HasFatalFailure();
                }
#pragma omp barrier
                if (stop) break;
                // Timing never weakens the all-candidate byte/repeat/dirty-tail gate.
                for (size_t position = 0; position < members.size(); ++position)
                {
                    const auto index = members[position];
#pragma omp master
                    { current = &requests[index]; bind_request(false); poison_guards(); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                    if (stop) break;
                    run_guarded(); run();
#pragma omp master
                    { check_bytes(); poison_guards(); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                    if (stop) break;
                    run_guarded(); run();
#pragma omp master
                    { check_bytes(); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                    if (stop) break;
                }
                if (stop) break;
                // Publish only after every candidate passed its post-timing checks.
                for (size_t position = 0; position < members.size(); ++position)
                {
                    const auto index = members[position];
#pragma omp master
                    { current = &requests[index]; bind_request(false); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                    if (stop) break;
                    run(); // One fresh per-candidate route authentication outside clocks.
#pragma omp master
                    {
                        check_bytes();
                        current_screen_samples = screens[index];
                        current_promotion_grade = selection.finalists[position];
                        current_timing_stage = current_promotion_grade ? "final" : "screening";
                        current_finalist_reason = selection.reasons[position];
                        timings = current_promotion_grade ? finals[index] : screens[index];
                        sample_observations = timed_counts[index];
                        current_timed_projection_count = (3u + (current_promotion_grade
                            ? static_cast<uint64_t>(samples) * repeats : 0u)) * live_projections;
                        emit(false); stop = ::testing::Test::HasFatalFailure();
                    }
#pragma omp barrier
                    if (stop) break;
                }
                if (stop) break;
            }
        }
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        return;
    }
    if (team == "retained")
    {
        std::chrono::steady_clock::time_point start;
#pragma omp parallel
        {
            for (size_t request = 0; request < requests.size(); ++request)
            {
#pragma omp master
                { current = &requests[request]; bind_request(); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                if (stop) break;
                run_guarded();
                for (int iteration = 0; iteration < warmup; ++iteration) run();
                for (int sample = 0; sample < samples; ++sample)
                {
#pragma omp barrier
#pragma omp master
                    { start = std::chrono::steady_clock::now(); }
#pragma omp barrier
                    for (int repeat = 0; repeat < repeats; ++repeat) run();
#pragma omp barrier
#pragma omp master
                    { timings[sample] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats; }
                }
#pragma omp master
                { poison_guards(); }
#pragma omp barrier
                run_guarded();
#pragma omp master
                { check_bytes(); sample_count(); PerfStatsCollector::reset(); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                if (stop) break;
                run();
#pragma omp master
                { check_bytes(); emit(false); stop = ::testing::Test::HasFatalFailure(); }
#pragma omp barrier
                if (stop) break;
            }
        }
    }
    else for (const auto &request : requests)
    {
        current = &request; bind_request(); ASSERT_FALSE(::testing::Test::HasFatalFailure());
        run_guarded(); for (int iteration = 0; iteration < warmup; ++iteration) run();
        for (int sample = 0; sample < samples; ++sample)
        {
            const auto start = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < repeats; ++repeat) run();
            timings[sample] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
        }
        poison_guards(); run_guarded(); check_bytes(); sample_count(); ASSERT_FALSE(::testing::Test::HasFatalFailure());
        PerfStatsCollector::reset(); run(); check_bytes(); emit(false); ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
}
