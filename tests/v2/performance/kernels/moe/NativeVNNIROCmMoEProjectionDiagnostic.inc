/**
 * @file NativeVNNIROCmMoEProjectionDiagnostic.inc
 * @brief M1 owner/column diagnostic branch of the existing canonical MoE trainer.
 *
 * This shard is included only by Perf__ROCmMoEVerifierPrefill.cpp. It reuses its
 * streams, capture transaction, event timer, tensor helpers and timing summary.
 * Prepared ownership follows MoEProjectionPipeline: sparse initial gate/up
 * payloads in the ordinary runtime and every original-source fixed down slice.
 * Extra full-H oracle engines cover only the actually routed source experts.
 * Runtime-mutable format reachability is certified from prepared/setup metadata;
 * each actual normalized physical leaf retains its own resources and receipt.
 * No expert descriptor aliases, replacement arithmetic, transport, scheduler,
 * installed policy or complete-operation performance provenance is introduced.
 */
namespace route87
{
using namespace llaminar2;
using Diagnostic = test::native_vnni_dispatch::MoEProjectionDiagnostic;
using Json = nlohmann::json;

/** @brief Fail one diagnostic cell without dropping a candidate or timing row. */
inline void require(bool ok, const char *message)
{
    if (!ok) throw std::runtime_error(message);
}

/** @brief Actual descriptor metadata independent of pointer identity or source aliases. */
inline Json descriptorIdentity(const DeviceNativeVNNIMatrixDesc &d)
{
    return {{"n", d.n}, {"k", d.k}, {"execution_codebook", d.codebook_id},
        {"source_codebook", d.source_codebook_id}, {"source_is_superblock", bool(d.source_is_superblock)},
        {"source_identity_present", bool(d.source_identity_present)}, {"blocks_per_row", d.blocks_per_row},
        {"descriptor_abi_bytes", sizeof(d)}, {"payload", reinterpret_cast<std::uintptr_t>(d.payload)},
        {"scales", reinterpret_cast<std::uintptr_t>(d.scales)},
        {"mins", reinterpret_cast<std::uintptr_t>(d.mins)}, {"emins", reinterpret_cast<std::uintptr_t>(d.emins)}};
}

/** @brief Build the existing ordinal owner authority without creating a communicator. */
inline MoEExpertOwnerMap makeOwners(const Diagnostic &spec)
{
    MoERoutedExpertPlacementPlan plan;
    plan.enabled = true;
    plan.topology = RoutedExpertPlacementTopology::SingleDomain;
    plan.continuation_domain = "route87";
    plan.shared_expert_domain = "route87";
    plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
    plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
    RoutedExpertDomain domain;
    domain.name = "route87";
    domain.scope = ExecutionDomainScope::RANK_LOCAL;
    domain.owner_rank = 0;
    domain.backend = CollectiveBackendType::RCCL;
    domain.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
    for (int ordinal : spec.device_ordinals)
        domain.participants.push_back(GlobalDeviceAddress::rocm(ordinal));
    plan.domains.push_back(std::move(domain));
    plan.routed_tiers.push_back({.name = "route87", .domain = "route87", .priority = 0});
    plan.placements.push_back({.layer = 0, .routed_expert_tier = std::vector<int>(spec.source.expert_count, 0)});
    return MoEExpertOwnerMap::build(plan);
}

/**
 * @brief Immutable source/prepared owners shared by every candidate graph.
 *
 * Only local gate/up owners and the small routed-oracle set are synthesized.
 * Every expert has its real full-K down slice; no missing descriptor aliases
 * another expert. Full-H GPU down engines exist only for the serial oracle's
 * actual routed IDs. Parents, views, pools, stores and fixed bank outlive graphs.
 */
struct PreparedOwner final
{
    struct Original { std::shared_ptr<TensorBase> gate, up, down, slice; };
    DeviceId device;
    MoEExpertProjectionOwnership ownership;
    MoEExpertOwnerMap owners;
    std::vector<Original> originals;
    std::vector<std::shared_ptr<test::GpuPreparedGemm>> engines;
    std::vector<MoEOverlayPreparedExpertPayload> local_gate_up;
    std::shared_ptr<const MoEOverlayFixedDownProjectionBank> fixed;
    std::vector<DeviceNativeVNNIMatrixDesc> oracle_gate, oracle_up, oracle_down;
    std::map<int, int> oracle_ids;
    Json inventory = Json::array();

    /**
     * @brief Prepare exactly one participant and the independent serial oracle.
     * @param spec Explicit original geometry/source, physical member and interval.
     * @param routing Actual authored synthetic row, unchanged by the candidates.
     * @note Source synthesis uses the existing physical/OpenMP worker authority;
     *       only setup is parallel. GPU packing uses the production batch helper.
     */
    PreparedOwner(const Diagnostic &spec, const std::vector<float> &routing)
        : device(DeviceId::rocm(spec.device_ordinals.at(spec.participant))),
          ownership(spec.ownership()), owners(makeOwners(spec)),
          originals(spec.source.expert_count), local_gate_up(spec.source.expert_count)
    {
        const auto &source = spec.source;
        const auto &gate_format = test::quantizedMoEVerifierFormat(source.routed.gate);
        const auto &down_format = test::quantizedMoEVerifierFormat(source.routed.down);
        std::set<int> routed;
        for (float id : routing)
        {
            const int expert = static_cast<int>(id);
            require(id == static_cast<float>(expert) && expert >= 0 && expert < source.expert_count,
                "diagnostic routing must name exact original expert IDs");
            routed.insert(expert);
        }
        for (int expert : routed) oracle_ids.emplace(expert, static_cast<int>(oracle_ids.size()));
        const auto mpi = MPIContextFactory::global();
        require(bool(mpi->clusterInventory()), "route87 requires the original physical rank inventory");
        const auto *hardware = DeviceManager::instance().hardware();
        require(hardware && hardware->total_physical_cores() > 0 && omp_get_max_threads() > 0,
            "route87 synthesis requires the canonical physical worker budget");
        const int workers = std::min({source.expert_count, hardware->total_physical_cores(), omp_get_max_threads()});
        std::vector<std::exception_ptr> failures(source.expert_count);
        const auto down = ownership.projection(WeightRole::MoEExpertDown);
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
        for (int expert = 0; expert < source.expert_count; ++expert)
        {
            try
            {
                auto &original = originals[expert];
                if (owners.ownerFor(0, expert)->owner_participant == spec.participant || routed.contains(expert))
                {
                    original.gate = gate_format.create({std::size_t(source.routed_expert_width), std::size_t(source.hidden_size)}, 51800 + expert);
                    original.up = gate_format.create({std::size_t(source.routed_expert_width), std::size_t(source.hidden_size)}, 51900 + expert);
                }
                original.down = down_format.create({std::size_t(source.hidden_size), std::size_t(source.routed_expert_width)}, 52000 + expert);
                require(bool(original.down), "route87 original down factory returned null");
                original.slice = original.down->create_view({std::size_t(down.rows), std::size_t(source.routed_expert_width)},
                    std::size_t(down.first_row) * source.routed_expert_width);
                require(original.down && original.slice, "route87 original down/source slice creation failed");
            }
            catch (...) { failures[expert] = std::current_exception(); }
        }
        for (const auto &error : failures) if (error) std::rethrow_exception(error);
        std::vector<TensorBase *> jobs;
        constexpr auto absent = std::numeric_limits<std::size_t>::max();
        std::vector<std::array<std::size_t, 4>> indices(source.expert_count);
        for (int expert = 0; expert < source.expert_count; ++expert)
        {
            indices[expert].fill(absent);
            const auto add = [&](int role, const std::shared_ptr<TensorBase> &tensor) {
                require(bool(tensor), "route87 prepared source role is missing");
                indices[expert][role] = jobs.size(); jobs.push_back(tensor.get());
            };
            if (originals[expert].gate) { add(0, originals[expert].gate); add(1, originals[expert].up); }
            add(2, originals[expert].slice);
            if (routed.contains(expert)) add(3, originals[expert].down);
        }
        auto prepared = test::makeGpuPreparedGemmBatch(jobs, device, "moe.route87.original");
        for (auto &engine : prepared) engines.push_back(std::make_shared<test::GpuPreparedGemm>(std::move(engine)));
        const auto handle = [&](std::size_t index) {
            require(index != absent, "route87 prepared role index is absent");
            return std::shared_ptr<ITensorGemm>(engines.at(index), engines.at(index)->kernel);
        };
        const auto descriptor = [&](std::size_t index, const test::QuantizedVerifierFormatCase &format, int n, int k) {
            DeviceNativeVNNIMatrixDesc d{};
            require(handle(index)->exportNativeVNNIMatrixDesc(d) && d.valid() && d.n == n && d.k == k &&
                d.codebook_id == format.device_execution_codebook_id && d.source_identity_present &&
                d.source_codebook_id == format.source_codebook_id && bool(d.source_is_superblock) == format.source_is_superblock,
                "route87 prepared descriptor contradicts its original family or geometry");
            return d;
        };
        ExpertGemmRegistry registry;
        const auto *endpoint = owners.participantForId(spec.participant);
        require(endpoint && endpoint->device == device && endpoint->domain_participant_index == spec.participant,
            "route87 ordinal participant mapping changed");
        for (int expert = 0; expert < source.expert_count; ++expert)
        {
            auto slice = handle(indices[expert][2]);
            registry.registerEngineForParticipant("route87", device, 0, spec.participant, 0, expert,
                ExpertGemmRegistry::WeightRole::DOWN, slice.get(), slice, ownership);
            const auto sd = descriptor(indices[expert][2], down_format, down.rows, source.routed_expert_width);
            Json entry{{"expert", expert}, {"down_slice", descriptorIdentity(sd)}};
            if (owners.ownerFor(0, expert)->owner_participant == spec.participant)
            {
                local_gate_up[expert] = MoEOverlayPreparedExpertPayload::gateUp(handle(indices[expert][0]), handle(indices[expert][1]));
                entry["gate"] = descriptorIdentity(descriptor(indices[expert][0], gate_format, source.routed_expert_width, source.hidden_size));
                entry["up"] = descriptorIdentity(descriptor(indices[expert][1], gate_format, source.routed_expert_width, source.hidden_size));
            }
            inventory.push_back(std::move(entry));
        }
        fixed = MoEOverlayFixedDownProjectionBank::resolve(registry, *endpoint, 0, ownership);
        // Compact only the independent oracle's descriptor table, preserving
        // original route order/weights through an explicit ID map outside timing.
        for (const auto &[expert, ordinal] : oracle_ids)
        {
            (void)ordinal;
            oracle_gate.push_back(descriptor(indices[expert][0], gate_format, source.routed_expert_width, source.hidden_size));
            oracle_up.push_back(descriptor(indices[expert][1], gate_format, source.routed_expert_width, source.hidden_size));
            oracle_down.push_back(descriptor(indices[expert][3], down_format, source.hidden_size, source.routed_expert_width));
        }
    }
};

/** @brief Exact resource receipt for this loaded leaf, not expert-tile proxies. */
inline Json resources(const rocm::MoERouteOwnedKernelResources &r)
{
    return {{"function", reinterpret_cast<std::uintptr_t>(r.function)},
        {"specialization", r.specialization ? r.specialization : ""}, {"block_threads", r.block_threads},
        {"registers", r.registers_per_thread}, {"local_bytes", r.local_memory_bytes_per_thread},
        {"static_shared_bytes", r.static_shared_memory_bytes}, {"dynamic_shared_bytes", r.dynamic_shared_memory_bytes},
        {"max_threads_per_block", r.max_threads_per_block}, {"active_blocks_per_sm", r.max_active_blocks_per_sm}};
}

/** @brief One retained contender and its chronological balanced samples. */
struct Trial
{
    std::string candidate, boundary;
    std::unique_ptr<ScopedHipPerfGraph> graph;
    Json physical;
    std::vector<double> samples;
    struct Sample { int round, position; double interval_us, latency_us; };
    std::vector<Sample> raw;
};

/**
 * @brief Own mutable component buffers independently of read-only prepared weights.
 *
 * Destruction joins the exact diagnostic stream before graphs and any embedded
 * pointers can retire, including exceptions after partial setup/capture. The
 * outer canonical process owner still owns watchdog, driver window and 30-second
 * retirement; this is only the existing harness's local diagnostic boundary.
 */
struct State final
{
    const Diagnostic &spec;
    ScopedMoEPrefillHipStream stream_owner;
    hipStream_t stream;
    PreparedOwner prepared;
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<ROCmMoEKernel> kernel, oracle;
    std::unique_ptr<DeviceMoERuntimeTable> runtime;
    std::shared_ptr<FP32Tensor> hidden, ids, weights, oracle_ids, oracle_routes, oracle_output;
    std::shared_ptr<FP32Tensor> compact_packet, fixed_packets, fixed_packet, routes, output;
    std::shared_ptr<DeviceTransferBuffer> count_owner;
    std::uint64_t *packet_bytes = nullptr;
    MoEGroupedIntermediateLayout layout;
    int gate_table = -1, down_table = -1, oracle_gate = -1, oracle_down = -1;
    std::vector<Trial> trials;
    NativeVnniExecutionFormatEnvelope gateup_envelope;
    Json gateup_setup_identity;

    /** @brief Retain immutable weights once; mutable native preparation follows in prepare(). */
    State(const Diagnostic &request, const std::vector<float> &routing)
        : spec(request), stream(stream_owner.get()), prepared(request, routing) {}

    /** @brief Join submitted work before every graph/tensor/workspace owner retires. */
    ~State()
    {
        if (hipStreamSynchronize(stream) != hipSuccess)
            ADD_FAILURE() << "route87 diagnostic stream retirement failed";
        trials.clear();
        if (kernel) { kernel->unbindWorkspace(); kernel->clearGPUStreamBinding(); }
        if (oracle) { oracle->unbindWorkspace(); oracle->clearGPUStreamBinding(); }
    }

    /** @brief Allocate through the existing tensor/transfer owner outside capture. */
    std::shared_ptr<FP32Tensor> tensor(std::size_t elements)
    {
        auto result = makeZeros({elements});
        require(result->ensureOnDevice(prepared.device, stream), "route87 tensor admission failed");
        return result;
    }

    /** @brief Publish only authored diagnostic inputs, outside timing/capture. */
    void publish(const std::shared_ptr<FP32Tensor> &tensor, const std::vector<float> &values)
    {
        require(tensor && tensor->numel() == values.size(), "route87 input publication shape changed");
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        require(tensor->ensureOnDevice(prepared.device, stream), "route87 input publication failed");
        TransferEngine::requireDeviceInput(tensor.get(), prepared.device, stream);
    }

    /** @return Diagnostic output bytes after an explicit untimed completion boundary. */
    template <class T> std::vector<T> download(const void *pointer, std::size_t count)
    {
        std::vector<T> result(count);
        require(hipMemcpyAsync(result.data(), pointer, count * sizeof(T), hipMemcpyDeviceToHost, stream) == hipSuccess &&
            hipStreamSynchronize(stream) == hipSuccess, "route87 diagnostic readback failed");
        return result;
    }

    /**
     * @brief Authenticate the initial authored runtime family before capture.
     * @note This reads the setup recipe just published from immutable owners,
     *       never a readback or host mirror of mutable device execution state.
     *       The canonical format authority expands runtime migration reachability;
     *       currently prepared bytes, reachable decoders and source trees remain
     *       separate facts. Every local pair contributes both gate and up.
     */
    void authenticateInitialGateUpEnvelope()
    {
        const auto &s = spec.source;
        const auto &initial = runtime->hostLayerState(0);
        require(runtime->hasInitialLayerRuntimeState(0) && initial.active_bank < 2 &&
            initial.expert_count == static_cast<unsigned>(s.expert_count) &&
            initial.top_k == static_cast<unsigned>(s.experts_per_token) &&
            initial.participant_id == static_cast<unsigned>(spec.participant) &&
            initial.participant_count == spec.device_ordinals.size(),
            "route87 initial runtime recipe has different participant geometry");
        const auto &bank = initial.banks[initial.active_bank];
        NativeVnniExecutionFormatEnvelope prepared_envelope, runtime_envelope;
        std::uint32_t prepared_mask = 0;
        int local_pairs = 0;
        const auto add = [](NativeVnniExecutionFormatEnvelope &envelope, const DeviceNativeVNNIMatrixDesc &d) {
            return d.valid() && addNativeVnniExecutionFormat(envelope, d.codebook_id,
                NativeVnniSourceIdentity{.codebook_id = d.source_codebook_id,
                    .is_superblock = d.source_is_superblock != 0, .present = d.source_identity_present != 0},
                true); // Same RuntimePlacementTable contract as the real upload.
        };
        for (int expert = 0; expert < s.expert_count; ++expert)
        {
            const auto *owner = prepared.owners.ownerFor(0, expert);
            const auto &actual = bank.experts[expert];
            require(owner && actual.logical_expert_id == expert &&
                actual.owner_participant == owner->owner_participant &&
                actual.projection_set == DeviceMoEProjectionSet::GateUp &&
                actual.weight_format == DeviceMoEWeightFormat::NativeVNNI,
                "route87 authored runtime expert identity differs from immutable ownership");
            const bool local = owner->owner_participant == spec.participant;
            require(bool(bank.local_compute_mask[expert]) == local,
                "route87 authored runtime local mask differs from immutable ownership");
            if (!local)
            {
                require(prepared.local_gate_up[expert].empty() && !actual.gate.valid() && !actual.up.valid(),
                    "route87 nonresident initial runtime expert unexpectedly has a payload");
                continue;
            }
            DeviceMoEExpertDescriptor expected{};
            require(exportDeviceMoEPreparedPayload(prepared.local_gate_up[expert], expected) &&
                expected.projection_set == DeviceMoEProjectionSet::GateUp && actual.movableWeightsReady() &&
                actual.gate.n == s.routed_expert_width && actual.gate.k == s.hidden_size &&
                actual.up.n == s.routed_expert_width && actual.up.k == s.hidden_size &&
                std::memcmp(&actual.gate, &expected.gate, sizeof(actual.gate)) == 0 &&
                std::memcmp(&actual.up, &expected.up, sizeof(actual.up)) == 0,
                "route87 initial runtime descriptor differs from its original prepared owner");
            require(add(prepared_envelope, expected.gate) && add(prepared_envelope, expected.up) &&
                add(runtime_envelope, actual.gate) && add(runtime_envelope, actual.up),
                "route87 prepared/runtime source has no exact reachable format envelope");
            prepared_mask |= nativeVnniCodebookMaskBit(expected.gate.codebook_id) |
                nativeVnniCodebookMaskBit(expected.up.codebook_id);
            ++local_pairs;
        }
        require(local_pairs > 0 && prepared_envelope.valid() && runtime_envelope.valid() &&
            prepared_envelope.execution_codebook_mask == runtime_envelope.execution_codebook_mask &&
            prepared_envelope.policy_codebook_mask == runtime_envelope.policy_codebook_mask,
            "route87 prepared and initial runtime format envelopes disagree");
        gateup_envelope = prepared_envelope;
        gateup_setup_identity = {{"authority", "immutable-prepared-and-authored-initial-runtime-recipe"},
            {"descriptor_source", "RuntimePlacementTable"}, {"local_prepared_pairs", local_pairs},
            {"prepared_execution_mask", prepared_mask},
            {"reachable_execution_mask", gateup_envelope.execution_codebook_mask},
            {"source_policy_mask", gateup_envelope.policy_codebook_mask},
            {"initial_bank", initial.active_bank}, {"initial_epoch", initial.active_epoch}};
    }

    /** @brief Bind exact workspace, runtime placement and immutable descriptor tables once. */
    void prepare()
    {
        const auto &s = spec.source;
        const auto requirements = MoEWorkspaceBuffers::rocmMoE(1, s.hidden_size, s.routed_expert_width, s.expert_count, s.experts_per_token);
        workspace = std::make_unique<DeviceWorkspaceManager>(prepared.device, requirements.total_bytes_with_alignment());
        require(workspace->allocate(requirements), "route87 exact named workspace admission failed");
        kernel = std::make_unique<ROCmMoEKernel>(prepared.device.ordinal);
        oracle = std::make_unique<ROCmMoEKernel>(prepared.device.ordinal);
        for (auto *moe : {kernel.get(), oracle.get()}) { moe->setGPUStream(stream); moe->bindWorkspace(workspace.get()); }
        std::vector<DeviceNativeVNNIMatrixDesc> gates(s.expert_count), ups(s.expert_count);
        for (int expert = 0; expert < s.expert_count; ++expert)
            if (!prepared.local_gate_up[expert].empty())
            {
                DeviceMoEExpertDescriptor descriptor{};
                require(exportDeviceMoEPreparedPayload(prepared.local_gate_up[expert], descriptor), "route87 sparse gate/up export failed");
                gates[expert] = descriptor.gate; ups[expert] = descriptor.up;
            }
        gate_table = kernel->uploadGroupedExpertGateUpDescriptorTables(gates.data(), ups.data(), s.expert_count,
            s.hidden_size, s.routed_expert_width, MoEDecodeDescriptorSource::RuntimePlacementTable);
        const auto fixed = std::get<MoEOverlayFixedDownProjectionBank::NativeDescriptorTable>(prepared.fixed->exportDescriptorTable());
        const auto down = prepared.ownership.projection(WeightRole::MoEExpertDown);
        down_table = kernel->uploadGroupedExpertDownDescriptorTable(fixed.experts.data(), s.expert_count, down.rows, s.routed_expert_width);
        const int oracle_experts = static_cast<int>(prepared.oracle_gate.size());
        oracle_gate = oracle->uploadGroupedExpertGateUpDescriptorTables(prepared.oracle_gate.data(), prepared.oracle_up.data(),
            oracle_experts, s.hidden_size, s.routed_expert_width);
        oracle_down = oracle->uploadGroupedExpertDownDescriptorTable(prepared.oracle_down.data(), oracle_experts,
            s.hidden_size, s.routed_expert_width);
        require(gate_table >= 0 && down_table >= 0 && oracle_gate >= 0 && oracle_down >= 0, "route87 descriptor table publication failed");
        DeviceMoERuntimeTable::Config config;
        config.device_id = prepared.device; config.num_layers = 1; config.num_experts = s.expert_count;
        config.top_k = s.experts_per_token; config.prefill_token_capacity = 1; config.mirror_to_device = true;
        config.fixed_down_banks = {prepared.fixed};
        runtime = std::make_unique<DeviceMoERuntimeTable>(config);
        publishInitialMoEProjectionRuntimeLayer(*runtime, prepared.owners, 0, prepared.local_gate_up, stream);
        authenticateInitialGateUpEnvelope();
        layout = {MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, static_cast<std::uint32_t>(s.routed_expert_width),
            static_cast<std::uint32_t>(s.experts_per_token), static_cast<std::uint32_t>(spec.device_ordinals.size())};
        require(layout.compactValid(), "route87 intermediate layout is invalid");
        hidden = tensor(s.hidden_size); ids = tensor(s.experts_per_token); weights = tensor(s.experts_per_token);
        oracle_ids = tensor(s.experts_per_token); oracle_routes = tensor(std::size_t(s.experts_per_token) * s.hidden_size);
        oracle_output = tensor(s.hidden_size);
        compact_packet = tensor(layout.compactCapacityBytes() / sizeof(float) + 16);
        fixed_packets = tensor(layout.packetBytes() * layout.participants / sizeof(float));
        fixed_packet = tensor(layout.packetBytes() / sizeof(float));
        routes = tensor(std::size_t(s.experts_per_token) * down.rows + 16); output = tensor(down.rows + 16);
        count_owner = TransferEngine::instance().allocateDeviceTransferBuffer(sizeof(std::uint64_t), prepared.device);
        require(count_owner && count_owner->isBound(), "route87 device compact-count owner is missing");
        packet_bytes = static_cast<std::uint64_t *>(count_owner->mutableDeviceData());
        require(hipStreamSynchronize(stream) == hipSuccess, "route87 setup publication did not complete");
    }

    /** @brief Enqueue the production owner-local grouping, gate/up and compact pack. */
    void ownerPhase(rocm::MoERouteOwnedGateUpLaunchSet *receipt = nullptr)
    {
        const auto &s = spec.source;
        const auto &host = runtime->hostLayerState(0);
        require(kernel->publishCompleteGroupedPrefillPlanFromRouter(runtime->deviceLayerState(0), ids.get(), weights.get(),
            1, 1, s.expert_count, s.experts_per_token, gate_table, down_table, true,
            MoEGroupedPlanDemand::None, DeviceMoEProjectionSet::GateUp), "route87 runtime owner group publication failed");
        require(kernel->executeGroupedPrefillProjectionFromPublishedRuntimePlan(runtime->deviceLayerState(0), host,
            hidden.get(), nullptr, gate_table, down_table, 1, s.hidden_size, s.routed_expert_width,
            s.expert_count, s.experts_per_token, MoEPrefillProjectionExecution::gateUp(s.hidden_size)), "route87 owner gate/up failed");
        if (receipt)
        {
            const auto launch = rocm::lastMoERouteOwnedGateUpLaunchSet();
            require(launch.has_value(), "route87 actual gate/up launch receipt missing");
            *receipt = *launch; // Before any other projection can invalidate it.
        }
        require(kernel->exportCompactGroupedPrefillIntermediates(layout, host, compact_packet.get(), packet_bytes),
            "route87 original compact intermediate publication failed");
    }

    /** @brief Regroup then import valid dependencies under that new map, outside isolated down timing. */
    void importForDown()
    {
        const auto &s = spec.source;
        const auto *owners = runtime->hostLayerState(0).route_participant_ids;
        require(kernel->prepareExpertGroupsAsync(ids.get(), weights.get(), 1, s.expert_count, s.experts_per_token) &&
            kernel->importGroupedPrefillIntermediates(layout, owners, fixed_packets.get()) &&
            kernel->importCompactGroupedPrefillIntermediates(layout, owners, compact_packet.get(), 0, packet_bytes, spec.participant),
            "route87 regroup/import did not bind current producer rows");
    }

    /** @brief Enqueue the unchanged physical-N down64 producer and canonical fold. */
    void downPhase()
    {
        const auto &s = spec.source; const auto down = prepared.ownership.projection(WeightRole::MoEExpertDown);
        require(kernel->executeGroupedPrefillProjection(nullptr, nullptr, gate_table, down_table, 1,
            s.hidden_size, s.routed_expert_width, s.expert_count, s.experts_per_token,
            MoEPrefillProjectionExecution::down(s.hidden_size, down.first_row, down.rows), routes.get()) &&
            kernel->reduceCanonicalRouteContributions(routes.get(), output.get(), 1, s.experts_per_token, down.rows),
            "route87 fixed down64/canonical fold failed");
    }
};

/**
 * @brief Populate transport-free peer dependencies and the public serial oracle.
 * @param state Retained original/source/runtime owner.
 * @param hidden Current authored hidden bits.
 * @param ids Current original expert IDs; -1 is an explicitly inactive route.
 * @param weights Current original router-order weights.
 * @note Peer payload setup uses the public grouped gate/up representation but
 *       is not called an independent numerical oracle. The public serial M1
 *       complete projection below independently certifies every canonical
 *       addend and folded output; candidate-owned compact records replace their
 *       corresponding prepopulated peer data before down validation.
 */
inline void prepareReference(State &state, const std::vector<float> &hidden,
    const std::vector<float> &ids, const std::vector<float> &weights)
{
    const auto &s = state.spec.source;
    state.publish(state.hidden, hidden); state.publish(state.ids, ids); state.publish(state.weights, weights);
    std::vector<float> remapped;
    for (float id : ids) remapped.push_back(id < 0 ? -1.f : float(state.prepared.oracle_ids.at(static_cast<int>(id))));
    state.publish(state.oracle_ids, remapped);
    require(state.oracle->groupedExpertDecodeFromRouting(state.hidden.get(), state.oracle_ids.get(), state.weights.get(),
        state.oracle_gate, state.oracle_down, s.experts_per_token, state.oracle_output.get(),
        s.hidden_size, s.routed_expert_width,
        nullptr, state.oracle_routes.get()), "route87 independent public serial M1 oracle failed");
    require(state.oracle->reduceCanonicalRouteContributions(state.oracle_routes.get(), state.oracle_output.get(),
        1, s.experts_per_token, s.hidden_size), "route87 independent public serial M1 fold failed");
    state.ownerPhase(); // Establish authoritative current route participants; untimed control.
    std::vector<std::uint32_t> peer_words;
    for (std::uint32_t peer = 0; peer < state.layout.participants; ++peer)
    {
        // Each packet is a single producer's publication. Author that peer's
        // synthetic input with inactive slots preserved, so its inverse map
        // contains -1 for every unowned route. Reusing an all-route map would
        // violate the packer's ownership contract even when its values match.
        // This is untimed reference construction, never runtime route planning.
        auto peer_ids = remapped;
        for (std::size_t slot = 0; slot < ids.size(); ++slot)
            if (ids[slot] < 0 || state.prepared.owners.ownerFor(0, static_cast<int>(ids[slot]))->owner_participant != static_cast<int>(peer))
                peer_ids[slot] = -1.f;
        state.publish(state.oracle_ids, peer_ids);
        require(state.oracle->prepareExpertGroupsAsync(state.oracle_ids.get(), state.weights.get(), 1,
            static_cast<int>(state.prepared.oracle_ids.size()), s.experts_per_token) &&
            state.oracle->executeGroupedPrefillProjection(state.hidden.get(), nullptr, state.oracle_gate, state.oracle_down,
                1, s.hidden_size, s.routed_expert_width, static_cast<int>(state.prepared.oracle_ids.size()), s.experts_per_token,
                MoEPrefillProjectionExecution::gateUp(s.hidden_size)), "route87 peer dependency production failed");
        require(state.oracle->exportGroupedPrefillIntermediates(state.layout,
            state.runtime->hostLayerState(0).route_participant_ids, state.fixed_packet.get(), static_cast<int>(peer)),
            "route87 peer dependency publication failed");
        const auto packet = state.download<std::uint32_t>(state.fixed_packet->gpu_data_ptr(), state.layout.packetBytes() / 4);
        peer_words.insert(peer_words.end(), packet.begin(), packet.end());
    }
    state.publish(state.oracle_ids, remapped);
    require(peer_words.size() * 4 == state.fixed_packets->size_bytes(), "route87 prepopulated peer bank extent changed");
    std::memcpy(state.fixed_packets->mutable_data(), peer_words.data(), peer_words.size() * 4);
    require(state.fixed_packets->ensureOnDevice(state.prepared.device, state.stream), "route87 peer dependency admission failed");
    for (const auto &tensor : {state.hidden, state.ids, state.weights, state.fixed_packets})
        TransferEngine::requireDeviceInput(tensor.get(), state.prepared.device, state.stream);
    require(hipStreamSynchronize(state.stream) == hipSuccess, "route87 reference preparation did not retire");
}

/** @brief Serialize native node geometry without assuming a named kernel is present. */
inline Json nodes(const std::vector<GPUGraphKernelNodeInfo> &inventory)
{
    Json result = Json::array();
    for (const auto &node : inventory)
        result.push_back({{"function", node.function_identity}, {"name", node.name},
            {"name_resolved", node.name_resolved}, {"graph_path", node.graph_path},
            {"grid", {node.grid_x, node.grid_y, node.grid_z}},
            {"block", {node.block_x, node.block_y, node.block_z}},
            {"dynamic_shared_bytes", node.dynamic_shared_memory_bytes}});
    return result;
}

/**
 * @brief Capture one real owner transaction under an admitted lexical geometry.
 * @param state Shared immutable preparation and exact construction stream.
 * @param admission Every normalized leaf in the exact runtime envelope, inspected before capture.
 * @param scoped False only for the untouched functional control, never a timed fourth alias.
 * @return Retained original graph and exact receipt/node identities.
 */
inline Trial captureOwner(State &state, const rocm::MoERouteOwnedGateUpAdmission &admission, bool scoped = true)
{
    const int columns = rocm::moeRouteOwnedGateUpColumns(admission.geometry());
    Trial result;
    result.candidate = scoped ? "rocm.moe.route_owned.gate_up.tn" + std::to_string(columns) + ".k1"
                              : "rocm.moe.route_owned.gate_up.untouched-functional-control";
    result.boundary = "owner-gate-up-pack-m1";
    std::unique_ptr<rocm::ScopedMoERouteOwnedGateUpGeometry> selected;
    if (scoped) selected = std::make_unique<rocm::ScopedMoERouteOwnedGateUpGeometry>(admission);
    std::vector<rocm::MoERouteOwnedGateUpLaunchSet> receipts;
    // Reserve before recording so retaining complete per-operation observations
    // never grows a host container inside the capture transaction.
    receipts.reserve(state.spec.operations_per_sample);
    auto graph = std::make_unique<ScopedHipPerfGraph>(state.stream, state.prepared.device.ordinal, "ROCm M1 owner gate/up diagnostic");
    for (int operation = 0; operation < state.spec.operations_per_sample; ++operation)
    {
        rocm::MoERouteOwnedGateUpLaunchSet receipt;
        state.ownerPhase(scoped ? &receipt : nullptr);
        if (scoped) receipts.push_back(receipt);
    }
    require(graph->finishAndInstantiate(), "route87 owner graph instantiation failed");
    selected.reset(); // Replay retains bound geometry after the scope has retired.
    std::vector<GPUGraphKernelNodeInfo> inventory; std::string error;
    require(graph->inspectKernelNodes(inventory, &error), "route87 retained native node inspection failed");
    if (scoped)
    {
        const auto &envelope = admission.executionEnvelope();
        require(receipts.size() == static_cast<std::size_t>(state.spec.operations_per_sample) &&
            envelope.execution_codebook_mask == state.gateup_envelope.execution_codebook_mask &&
            envelope.policy_codebook_mask == state.gateup_envelope.policy_codebook_mask,
            "route87 resource admission differs from prepared/runtime envelope");
        Json operations = Json::array(), ordered_codebooks = Json::array(), arithmetic = Json::array();
        for (std::size_t operation = 0; operation < receipts.size(); ++operation)
        {
            const auto &set = receipts[operation];
            require(set.execution_envelope.execution_codebook_mask == envelope.execution_codebook_mask &&
                set.execution_envelope.policy_codebook_mask == envelope.policy_codebook_mask &&
                set.leaf_count == admission.leafCount() && set.leaf_count > 0 && set.leaf_count <= set.leaves.size(),
                "route87 actual launch set omitted or changed the admitted envelope");
            Json leaves = Json::array(), order = Json::array(), trees = Json::array();
            std::uint32_t observed_mask = 0;
            for (std::size_t index = 0; index < set.leaf_count; ++index)
            {
                const auto &leaf = set.leaves[index];
                const auto bit = nativeVnniCodebookMaskBit(leaf.execution_codebook);
                require(bit && !(observed_mask & bit) && (admission.launchCodebookMask() & bit),
                    "route87 actual leaf is duplicated or outside normalized admission");
                observed_mask |= bit;
                const auto &admitted = admission.leafResources(leaf.execution_codebook);
                require(leaf.geometry == admission.geometry() &&
                    leaf.device_ordinal == state.prepared.device.ordinal && leaf.stream == state.stream &&
                    leaf.n == state.spec.source.routed_expert_width && leaf.k == state.spec.source.hidden_size &&
                    leaf.route_capacity == state.spec.source.experts_per_token && leaf.block_x == columns &&
                    leaf.block_y == 1 && leaf.block_z == 1 && resources(leaf.resources) == resources(admitted),
                    "route87 effective leaf differs from admitted resources/source identity");
                order.push_back(leaf.execution_codebook);
                trees.push_back({{"execution_codebook", leaf.execution_codebook},
                    {"arithmetic_partitions", leaf.arithmetic_partitions}});
                leaves.push_back({{"execution_codebook", leaf.execution_codebook},
                    {"n", leaf.n}, {"k", leaf.k}, {"route_capacity", leaf.route_capacity},
                    {"stream", reinterpret_cast<std::uintptr_t>(leaf.stream)},
                    {"grid", {leaf.grid_x, leaf.grid_y, leaf.grid_z}},
                    {"block", {leaf.block_x, leaf.block_y, leaf.block_z}},
                    {"arithmetic_partitions", leaf.arithmetic_partitions}, {"resources", resources(leaf.resources)}});
                if (operation == 0)
                {
                    int matches = 0;
                    for (const auto &node : inventory)
                        if (node.function_identity == reinterpret_cast<std::uintptr_t>(leaf.resources.function))
                        {
                            require(node.grid_x == static_cast<unsigned>(leaf.grid_x) &&
                                node.grid_y == static_cast<unsigned>(leaf.grid_y) && node.grid_z == static_cast<unsigned>(leaf.grid_z) &&
                                node.block_x == static_cast<unsigned>(leaf.block_x) && node.block_y == 1 && node.block_z == 1 &&
                                node.dynamic_shared_memory_bytes == leaf.resources.dynamic_shared_memory_bytes,
                                "route87 captured node disagrees with its actual producer leaf");
                            ++matches;
                        }
                    require(matches == state.spec.operations_per_sample,
                        "route87 graph did not retain every admitted physical producer leaf");
                }
                else
                {
                    const auto &first = receipts.front().leaves[index];
                    require(leaf.execution_codebook == first.execution_codebook && leaf.grid_x == first.grid_x &&
                        leaf.grid_y == first.grid_y && leaf.grid_z == first.grid_z &&
                        leaf.arithmetic_partitions == first.arithmetic_partitions,
                        "route87 repeated captured operation changed its actual ordered leaf contract");
                }
            }
            require(observed_mask == admission.launchCodebookMask(),
                "route87 actual launch set differs from normalized production launch mask");
            if (operation == 0) { ordered_codebooks = order; arithmetic = trees; }
            else require(order == ordered_codebooks && trees == arithmetic,
                "route87 captured operation changed source arithmetic or launch order");
            operations.push_back({{"operation", operation}, {"leaf_count", set.leaf_count}, {"leaves", leaves}});
        }
        result.physical = {{"gate_tn", columns}, {"reachable_execution_mask", envelope.execution_codebook_mask},
            {"source_policy_mask", envelope.policy_codebook_mask}, {"normalized_launch_mask", admission.launchCodebookMask()},
            {"leaf_count_per_operation", admission.leafCount()}, {"ordered_leaf_codebooks", ordered_codebooks},
            {"arithmetic_identity", arithmetic}, {"recorded_operations", operations}};
    }
    result.physical["captured_kernels"] = nodes(inventory);
    result.graph = std::move(graph);
    return result;
}

/** @brief Compare every canonical/folded byte and suffix after an accepted down replay. */
inline void proveDownOutput(State &state)
{
    constexpr std::uint32_t poison = 0xa5a5a5a5u;
    const auto down = state.prepared.ownership.projection(WeightRole::MoEExpertDown);
    const auto actual = state.download<std::uint32_t>(state.routes->gpu_data_ptr(), state.routes->numel());
    const auto expected = state.download<std::uint32_t>(state.oracle_routes->gpu_data_ptr(), state.oracle_routes->numel());
    for (int route = 0; route < state.spec.source.experts_per_token; ++route)
        require(std::memcmp(actual.data() + std::size_t(route) * down.rows,
            expected.data() + std::size_t(route) * state.spec.source.hidden_size + down.first_row, down.rows * 4) == 0,
            "route87 actual compact down contribution differs from public serial M1 bytes");
    for (std::size_t index = std::size_t(state.spec.source.experts_per_token) * down.rows; index < actual.size(); ++index)
        require(actual[index] == poison, "route87 down producer overwrote suffix guard");
    const auto output = state.download<std::uint32_t>(state.output->gpu_data_ptr(), state.output->numel());
    const auto oracle = state.download<std::uint32_t>(state.oracle_output->gpu_data_ptr(), state.oracle_output->numel());
    require(std::memcmp(output.data(), oracle.data() + down.first_row, down.rows * 4) == 0,
        "route87 compact folded columns differ from public serial M1 bytes");
    for (std::size_t index = down.rows; index < output.size(); ++index)
        require(output[index] == poison, "route87 fold overwrote suffix guard");
}

/**
 * @brief Erase every named candidate activation/intermediate bank before proof.
 * @param state Original workspace owner; exact named extents come from its plan.
 * @note Reference preparation shares these banks and can leave current valid
 *       data. Poisoning after that preparation ensures a captured producer that
 *       skips its writes cannot borrow the eager reference's Q8 values/scales.
 *       Prepared descriptors, router inputs and fixed peer packets are separate
 *       owners and remain intact. This untimed diagnostic clear is not a transfer
 *       extent and adds no operation to the retained production graph.
 */
inline void poisonOwnerScratch(State &state)
{
    for (const char *name : {MoEWorkspaceBuffers::PREFILL_A_INT8,
             MoEWorkspaceBuffers::PREFILL_A_SCALES, MoEWorkspaceBuffers::PREFILL_GATE,
             MoEWorkspaceBuffers::PREFILL_UP, MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,
             MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES})
    {
        void *pointer = state.workspace->getBuffer(name);
        const auto bytes = state.workspace->getBufferSize(name);
        require(pointer && bytes > 0, "route87 candidate scratch bank is not admitted");
        require(hipMemsetAsync(pointer, 0xa5, bytes, state.stream) == hipSuccess,
            "route87 candidate scratch poison failed");
    }
}

/**
 * @brief Prove compact live rows and every consumed canonical/output byte.
 * @param ids Authored diagnostic route row; expected counts are test inputs only.
 * @note This launches regroup/import/down after the candidate owner replay.
 *       It does not time setup transport, fabricate device counts or use a
 *       grouped TN64 candidate as an independent serial numerical oracle.
 */
inline void proveOwner(State &state, Trial &trial, const std::vector<float> &ids)
{
    constexpr std::uint32_t poison = 0xa5a5a5a5u;
    poisonOwnerScratch(state);
    require(hipMemsetAsync(state.packet_bytes, 0xa5, sizeof(*state.packet_bytes), state.stream) == hipSuccess,
        "route87 compact-count poison failed");
    require(hipMemsetAsync(state.compact_packet->gpu_data_ptr(), 0xa5, state.compact_packet->size_bytes(), state.stream) == hipSuccess &&
        hipMemsetAsync(state.routes->gpu_data_ptr(), 0xa5, state.routes->size_bytes(), state.stream) == hipSuccess &&
        hipMemsetAsync(state.output->gpu_data_ptr(), 0xa5, state.output->size_bytes(), state.stream) == hipSuccess,
        "route87 diagnostic poison failed");
    require(trial.graph->launch(), "route87 retained owner replay failed");
    const auto count = state.download<std::uint64_t>(state.packet_bytes, 1).front();
    std::set<int> expected_slots;
    for (int slot = 0; slot < state.spec.source.experts_per_token; ++slot)
        if (ids[slot] >= 0 && state.prepared.owners.ownerFor(0, static_cast<int>(ids[slot]))->owner_participant == state.spec.participant)
            expected_slots.insert(slot);
    require(count == expected_slots.size() * state.layout.compactRecordWords() * 4,
        "route87 compact producer published capacity or wrong owner rows");
    const auto packet = state.download<std::uint32_t>(state.compact_packet->gpu_data_ptr(), state.compact_packet->size_bytes() / 4);
    const auto fixed = state.download<std::uint32_t>(state.fixed_packets->gpu_data_ptr(), state.fixed_packets->size_bytes() / 4);
    std::set<int> observed_slots;
    for (std::size_t record = 0; record < expected_slots.size(); ++record)
    {
        const auto offset = record * state.layout.compactRecordWords();
        const int slot = static_cast<int>(packet[offset]);
        require(expected_slots.contains(slot) && observed_slots.insert(slot).second, "route87 compact route IDs changed ownership/order");
        const auto fixed_offset = (std::size_t(state.spec.participant) * state.layout.route_capacity + slot) * state.layout.routeWords();
        require(std::memcmp(packet.data() + offset + 1, fixed.data() + fixed_offset, state.layout.routeWords() * 4) == 0,
            "route87 compact Q8/scales differ from production peer dependency bits");
    }
    for (std::size_t index = state.layout.compactCapacityBytes() / 4; index < packet.size(); ++index)
        require(packet[index] == poison, "route87 compact producer overwrote suffix guard");
    state.importForDown(); state.downPhase();
    proveDownOutput(state);
}

/** @brief Preserve completed paid samples on success or a later cohort failure. */
inline void writeRawEvidence(std::FILE *raw, const Diagnostic &spec, const Trial &trial)
{
    const bool gate_up = trial.boundary == "owner-gate-up-pack-m1";
    for (const auto &sample : trial.raw)
        std::fprintf(raw, "rocm,%s,%s,%s,%d,%d,%d,%.9f,%a,%.9f,%a,%d,%s,rocm-moe-owner-compact-diagnostic-v2\n",
            trial.boundary.c_str(), spec.evidenceId().c_str(), trial.candidate.c_str(), sample.round, sample.position,
            spec.operations_per_sample, sample.interval_us, sample.interval_us, sample.latency_us, sample.latency_us, 1, gate_up ? "balanced-rotation-paired-v1" : "singleton-fixed-control-v1");
    require(std::fflush(raw) == 0, "route87 raw evidence flush failed");
}

/** @brief Time all three gate/up contenders once per balanced chronological round. */
inline void timeOwners(State &state, std::FILE *raw)
{
    using test::native_vnni_dispatch::moeRouteGeometryTimingOrder;
    for (int round = 0; round < state.spec.warmups; ++round)
        for (auto index : moeRouteGeometryTimingOrder(round))
            require(state.trials[1 + index].graph->launch(), "route87 owner warmup replay failed");
    require(hipStreamSynchronize(state.stream) == hipSuccess, "route87 owner warmup did not complete");
    MoEPrefillEventTimer timer(state.stream);
    const auto publish_raw = [&] {
        for (std::size_t index = 1; index <= 3; ++index)
            writeRawEvidence(raw, state.spec, state.trials[index]);
    };
    try
    {
        for (int round = 0; round < state.spec.samples; ++round)
        {
            const auto &order = moeRouteGeometryTimingOrder(round);
            for (std::size_t position = 0; position < order.size(); ++position)
            {
                auto &trial = state.trials[1 + order[position]];
                const double interval = timer.sample(1, [&] { return trial.graph->launch(); }) * 1000.;
                require(std::isfinite(interval) && interval > 0, "route87 event interval must be finite and positive");
                const double latency = interval / state.spec.operations_per_sample;
                trial.raw.push_back({round, static_cast<int>(position), interval, latency}); trial.samples.push_back(latency);
            }
        }
    }
    catch (...)
    {
        publish_raw(); // Preserve every completed sample before State retires.
        throw;
    }
    publish_raw();
}

/** @brief Keep all raw timing rounds; sorting is only for the existing summary. */
inline void writeEvidence(std::FILE *aggregate, const Diagnostic &spec, const Trial &trial)
{
    auto sorted = trial.samples; std::sort(sorted.begin(), sorted.end());
    const auto summary = test::trainer::summarizeSortedTimingSamples(sorted);
    const bool gate_up = trial.boundary == "owner-gate-up-pack-m1";
    const auto down = spec.ownership().projection(WeightRole::MoEExpertDown);
    std::fprintf(aggregate,
        "rocm,%s,%s,%s,%s,%s,%s,1,%d,%d,%d,%d,%d,%zu,%d,%d,%d,%s,%s,%s,%d,1,64,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%s,1,1,0,0,rocm-moe-owner-compact-diagnostic-v2\n",
        trial.boundary.c_str(), spec.evidenceId().c_str(),
        std::string(test::native_vnni_dispatch::moeRoutingProfileName(spec.route_profile)).c_str(),
        spec.source.routed.gate.c_str(), spec.source.routed.up.c_str(), spec.source.routed.down.c_str(),
        spec.source.hidden_size, spec.source.routed_expert_width, spec.source.expert_count, spec.source.experts_per_token,
        spec.participant, spec.device_ordinals.size(), down.first_row, down.rows, spec.source.hidden_size,
        gate_up ? "SelectedRoutes" : "AllRoutes", gate_up ? "IntermediateRows" : "OriginalRouteRowsThenTokenRows",
        trial.candidate.c_str(), gate_up ? trial.physical.at("gate_tn").get<int>() : 0,
        spec.operations_per_sample, summary.min, summary.median, summary.p95, summary.mad, summary.cv,
        trial.samples.size(), summary.digest.c_str());
    require(std::fflush(aggregate) == 0, "route87 aggregate evidence flush failed");
}

/**
 * @brief Run the bounded branch in the existing canonical ROCm MoE trainer.
 * @param spec Fully validated diagnostic declaration; no installed-source tier.
 * @note The gate/up contest includes local runtime grouping and compact pack.
 *       The separate down64 control is isolated after dependencies are imported;
 *       it times canonical down plus fold, not a complete peer exchange.
 */
inline void run(const Diagnostic &spec)
{
    require(test::native_vnni_dispatch::profilerRequestId().empty(),
        "route87 profiling requires the separate explicit MoE collector route; old decode/pair requests are forbidden");
    const char *aggregate_path = std::getenv("LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_CSV");
    const char *raw_path = std::getenv("LLAMINAR_ROCM_MOE_PRODUCTION_SWEEP_TIMING_CSV");
    require(aggregate_path && *aggregate_path && raw_path && *raw_path,
        "route87 requires explicit fresh aggregate and raw timing destinations");
    using File = std::unique_ptr<std::FILE, int (*)(std::FILE *)>;
    File aggregate(std::fopen(aggregate_path, "wx"), std::fclose), raw(std::fopen(raw_path, "wx"), std::fclose);
    require(bool(aggregate) && bool(raw), "route87 evidence destinations must be fresh and writable");
    const std::string bundle_path = std::string(aggregate_path) + ".operation.json";
    File bundle(std::fopen(bundle_path.c_str(), "wx"), std::fclose);
    require(bool(bundle), "route87 operation bundle destination must be fresh");
    std::fputs("backend,phase,case_id,route_profile,source_gate,source_up,source_down,m,model_hidden,expert_width,logical_expert_count,top_k,participant,degree,first_column,physical_down_n,arithmetic_policy_down_n,route_scope,publication,candidate_id,gate_tn,k_lanes,down_tn,operations_per_sample,min_us,median_us,p95_us,mad_us,cv,samples,sample_digest,serial_byte_equal,compact_extent_equal,installed_policy_eligible,winner_installable,evidence_generation\n", aggregate.get());
    std::fputs("backend,phase,case_id,candidate_id,round_index,order_position,complete_operations_per_sample,interval_us,interval_us_hex,latency_us,latency_us_hex,graph_submissions,ordering_protocol,evidence_generation\n", raw.get());
    Json evidence{{"schema", "native-vnni-moe-projection-diagnostic-evidence-v2"}, {"complete", false}, {"passed", false},
        {"installable", false}, {"case_id", spec.evidenceId()}, {"descriptor_document", spec.descriptor_document},
        {"descriptor_evidence", spec.descriptor_evidence}, {"weight_authority", "synthetic-independent-seeds-51800-51900-52000"},
        {"owner_map", "ordinal"}, {"native_membership", spec.device_ordinals}, {"participant", spec.participant},
        {"original_layer", spec.layer}, {"physical_m", 1}, {"top_k_authority", "explicit-diagnostic-declaration"},
        {"operations_per_sample", spec.operations_per_sample}, {"warmups", spec.warmups}, {"samples", spec.samples},
        {"timing_protocol", {{"owner_contest", "balanced-rotation-paired-v1"}, {"fixed_down_control", "singleton-fixed-control-v1"}}}, {"scope", "transport-free-components-not-full-TP4-economy"},
        {"router_q8_reuse", false}, {"capture_input_quantization", "existing-production-path"},
        {"candidate_dispositions", Json::array()}};
    const auto finish = [&] {
        const auto text = evidence.dump(2);
        require(std::fprintf(bundle.get(), "%s\n", text.c_str()) >= 0 && std::fflush(bundle.get()) == 0,
            "route87 operation bundle publication failed");
    };
    try
    {
        int device_count = 0;
        require(hipGetDeviceCount(&device_count) == hipSuccess, "route87 native ROCm discovery failed");
        for (int ordinal : spec.device_ordinals) require(ordinal < device_count, "route87 physical native member unavailable");
        const int device = spec.device_ordinals.at(spec.participant);
        require(hipSetDevice(device) == hipSuccess, "route87 device selection failed");
        const auto &down_format = test::quantizedMoEVerifierFormat(spec.source.routed.down);
        const auto base_ids = test::native_vnni_dispatch::makeMoERoutingIndices(spec.route_profile, 1,
            spec.source.experts_per_token, spec.source.expert_count);
        const auto base_weights = makeRoutingWeights(1, spec.source.experts_per_token);
        const auto base_hidden = makeHiddenValues(1, spec.source.hidden_size);
        State state(spec, base_ids); state.prepare(); state.trials.reserve(5);
        evidence["gateup_setup_identity"] = state.gateup_setup_identity;
        std::vector<rocm::MoERouteOwnedGateUpAdmission> admissions;
        bool resource_rejection = false;
        for (auto geometry : rocm::kMoERouteOwnedGateUpGeometries)
        {
            const auto id = "rocm.moe.route_owned.gate_up.tn" + std::to_string(rocm::moeRouteOwnedGateUpColumns(geometry)) + ".k1";
            try
            {
                admissions.push_back(rocm::admitMoERouteOwnedGateUpGeometry(geometry, state.gateup_envelope, device));
                const auto &admitted = admissions.back();
                Json leaves = Json::array();
                for (std::uint8_t codebook = 0; codebook < 32; ++codebook)
                    if (admitted.launchCodebookMask() & nativeVnniCodebookMaskBit(codebook))
                        leaves.push_back({{"execution_codebook", codebook},
                            {"resources", resources(admitted.leafResources(codebook))}});
                require(leaves.size() == admitted.leafCount(), "route87 admission leaf inventory is incomplete");
                evidence["candidate_dispositions"].push_back({{"candidate", id}, {"status", "resource-admitted"},
                    {"reachable_execution_mask", admitted.executionEnvelope().execution_codebook_mask},
                    {"source_policy_mask", admitted.executionEnvelope().policy_codebook_mask},
                    {"normalized_launch_mask", admitted.launchCodebookMask()}, {"leaves", leaves}});
            }
            catch (const std::exception &error)
            {
                resource_rejection = true;
                evidence["candidate_dispositions"].push_back({{"candidate", id}, {"status", "resource-rejected"}, {"error", error.what()}});
            }
        }
        require(!resource_rejection && admissions.size() == 3, "route87 complete three-geometry resource admission failed");
        const auto fixed_down_resources = rocm::inspectMoERouteOwnedDown64(
            down_format.device_execution_codebook_id, MoEPrefillOutputLayout::OriginalRouteRows, device);
        evidence["fixed_down64_resources"] = resources(fixed_down_resources);
        const auto physical_down = spec.ownership().projection(WeightRole::MoEExpertDown);
        evidence["operation_geometry"] = {{"original_h", spec.source.hidden_size}, {"full_f", spec.source.routed_expert_width},
            {"logical_experts", spec.source.expert_count}, {"top_k", spec.source.experts_per_token},
            {"down_first_column", physical_down.first_row}, {"down_physical_n", physical_down.rows},
            {"down_arithmetic_policy_n", spec.source.hidden_size}, {"down_k", spec.source.routed_expert_width}};
        evidence["prepared_inventory"] = state.prepared.inventory;
        evidence["source_profile_original_route_ids"] = base_ids;
        evidence["source_profile_original_weights"] = base_weights;
        evidence["oracle_original_expert_to_compact_id"] = state.prepared.oracle_ids;
        evidence["oracle_kind"] = "public-groupedExpertDecodeFromRouting-M1-original-H-canonical-addends";
        evidence["peer_setup"] = "untimed-public-gateUp-fixed-packets-no-transport-not-an-independent-oracle";
        prepareReference(state, base_hidden, base_ids, base_weights);
        state.trials.push_back(captureOwner(state, admissions.front(), false)); // Untimed untouched physical control.
        for (const auto &admission : admissions) state.trials.push_back(captureOwner(state, admission));
        evidence["untouched_control"] = state.trials.front().physical;
        evidence["owner_contest_operations"] = Json::array();
        for (std::size_t index = 1; index <= 3; ++index)
            evidence["owner_contest_operations"].push_back({{"candidate", state.trials[index].candidate},
                {"physical", state.trials[index].physical}, {"installed_policy_eligible", false}});
        const auto partitions = state.trials[1].physical.at("arithmetic_identity");
        for (std::size_t contender = 1; contender <= 3; ++contender)
            require(state.trials[contender].physical.at("arithmetic_identity") == partitions &&
                state.trials[contender].physical.at("ordered_leaf_codebooks") == state.trials[1].physical.at("ordered_leaf_codebooks"),
                "route87 TN candidate changed original serial-M1 arithmetic partitions");
        for (int replay = 0; replay < 20; ++replay)
        {
            auto hidden = base_hidden;
            for (std::size_t index = 0; index < hidden.size(); ++index)
                hidden[index] += float((int(index) + replay * 7) % 17 - 8) * .00031f;
            auto ids = base_ids; std::rotate(ids.begin(), ids.begin() + replay % ids.size(), ids.end());
            auto weights = base_weights;
            const int live = replay % 4 == 1 ? 0 : replay % 4 == 2 ? spec.source.experts_per_token / 2 : spec.source.experts_per_token;
            for (int route = live; route < spec.source.experts_per_token; ++route) { ids[route] = -1.f; weights[route] = 0.f; }
            prepareReference(state, hidden, ids, weights);
            for (auto &trial : state.trials) proveOwner(state, trial, ids);
        }
        evidence["owner_contest_replays_per_candidate"] = 20;
        evidence["owner_scratch_poison"] = "all-six-named-activation-and-intermediate-banks-before-each-replay";
        evidence["functional_history"] = "full-empty-partial-full; hidden-generation-and-route-order-changes; immutable-prepared-weights";
        prepareReference(state, base_hidden, base_ids, base_weights);
        const auto timed_bytes = state.download<std::uint64_t>(state.packet_bytes, 1).front();
        const auto record_bytes = state.layout.compactRecordWords() * sizeof(std::uint32_t);
        require(timed_bytes % record_bytes == 0 && timed_bytes <= state.layout.compactCapacityBytes(),
            "route87 timed owner publication has an invalid live extent");
        evidence["timed_device_compact_bytes"] = timed_bytes;
        evidence["timed_owner_live_routes"] = timed_bytes / record_bytes;
        evidence["timed_owner_has_work"] = timed_bytes > 0;
        timeOwners(state, raw.get());
        for (std::size_t index = 1; index <= 3; ++index)
            writeEvidence(aggregate.get(), spec, state.trials[index]);
        // Owner captures mutate grouping/scratch. Restore all routes and import
        // before capturing the separate isolated fixed-down control, then run
        // it alone. No stale owner grouping enters the down timing interval.
        prepareReference(state, base_hidden, base_ids, base_weights); state.importForDown();
        Trial fixed; fixed.candidate = "rocm.moe.route_owned.down.canonical.tn64.fixed";
        fixed.boundary = "down-columns-fold-prepopulated-m1";
        fixed.graph = std::make_unique<ScopedHipPerfGraph>(state.stream, device, "ROCm M1 prepopulated compact down64 control");
        for (int operation = 0; operation < spec.operations_per_sample; ++operation) state.downPhase();
        require(fixed.graph->finishAndInstantiate(), "route87 fixed down64 graph instantiation failed");
        std::vector<GPUGraphKernelNodeInfo> fixed_nodes; std::string inspect_error;
        require(fixed.graph->inspectKernelNodes(fixed_nodes, &inspect_error), "route87 fixed down64 native node inspection failed");
        int fixed_matches = 0;
        for (const auto &node : fixed_nodes)
            if (node.function_identity == reinterpret_cast<std::uintptr_t>(fixed_down_resources.function))
            {
                require(node.block_x == 64 && node.block_y == 1 && node.block_z == 1 &&
                    node.grid_x == static_cast<unsigned>((physical_down.rows + 63) / 64) &&
                    node.grid_y == static_cast<unsigned>(spec.source.experts_per_token) && node.grid_z == 1 &&
                    node.dynamic_shared_memory_bytes == fixed_down_resources.dynamic_shared_memory_bytes,
                    "route87 fixed down64 control launched different geometry"); ++fixed_matches;
            }
        require(fixed_matches == spec.operations_per_sample, "route87 fixed down64 control symbol was not captured");
        fixed.physical = {{"resources", resources(fixed_down_resources)}, {"captured_kernels", nodes(fixed_nodes)}};
        state.trials.push_back(std::move(fixed));
        auto &control = state.trials.back();
        require(hipMemsetAsync(state.routes->gpu_data_ptr(), 0xa5, state.routes->size_bytes(), state.stream) == hipSuccess &&
            hipMemsetAsync(state.output->gpu_data_ptr(), 0xa5, state.output->size_bytes(), state.stream) == hipSuccess &&
            control.graph->launch(), "route87 captured down64 poison replay failed");
        proveDownOutput(state);
        for (int round = 0; round < spec.warmups; ++round) require(control.graph->launch(), "route87 down64 warmup failed");
        evidence["fixed_down_capture_proof_replays"] = 1;
        MoEPrefillEventTimer down_timer(state.stream);
        try
        {
            for (int round = 0; round < spec.samples; ++round)
            {
                const double interval = down_timer.sample(1, [&] { return control.graph->launch(); }) * 1000.;
                require(std::isfinite(interval) && interval > 0, "route87 down64 interval must be positive");
                control.raw.push_back({round, 0, interval, interval / spec.operations_per_sample});
                control.samples.push_back(interval / spec.operations_per_sample);
            }
        }
        catch (...)
        {
            writeRawEvidence(raw.get(), spec, control);
            throw;
        }
        writeRawEvidence(raw.get(), spec, control);
        writeEvidence(aggregate.get(), spec, control);
        evidence["serial_byte_equal"] = true; evidence["compact_record_bits_equal"] = true;
        evidence["compact_live_extent_equal"] = true;
        evidence["timed_operations"] = Json::array();
        for (std::size_t index = 1; index < state.trials.size(); ++index)
        {
            const auto &trial = state.trials[index];
            evidence["timed_operations"].push_back({{"candidate", trial.candidate}, {"boundary", trial.boundary},
                {"physical", trial.physical}, {"sample_count", trial.samples.size()}, {"installed_policy_eligible", false}});
        }
        evidence["complete"] = true; evidence["passed"] = true;
        finish();
    }
    catch (const std::exception &error)
    {
        evidence["error"] = error.what(); evidence["complete"] = true; evidence["passed"] = false; finish(); throw;
    }
}
} // namespace route87
