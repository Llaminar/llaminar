/**
 * @file MoEIntermediateShardROCmMappedRetainedStage.inc
 * @brief Binds the ROCm two-kernel mapped-overlay HC+Router+FFN program to the decode graph.
 *
 * Included inside MoEIntermediateShardPipeline.cpp's anonymous namespace so it shares
 * Binding and descriptor-declaration authority with CUDARetainedStage and ResidentStage.
 *
 * Kernel 1 (`fastMappedRoutePrefixKernel`) fuses previous residual write, BranchRMSNorm,
 * optional HC injection, HC Down/Up/Mix, FP16 Router, and TopK/softmax publication, then
 * publishes its HIP capture frontier to `HIPEarlyPacketPublication` so CPU/remote dispatch
 * packets launch concurrently on reserved co-residency CUs while Kernel 2
 * (`fastMappedFFNSuffixKernel`) computes compact local routed + shared FFN on the
 * remaining CUs across all 18 quantization codebooks with zero spills.
 */

#ifdef HAVE_ROCM

/** @brief Preparation and publication of one ROCm participant's mapped local HC/FFN region. */
class ROCmMappedRetainedStage final : public IComputeStage, public IWorkspaceConsumer
{
public:
    /**
     * @brief Authenticate original mapped M1 graph operands before replacing the closed region.
     * @param binding Original routed intermediate-shard binding (`LocalResult::RouteSlots`).
     * @param router Original router stage parameters.
     * @param shared Original prepared shared-expert FFN parameters.
     * @param prefix Original staged post-attention HC phase owner.
     * @param early_readers Early outbound packet/ticket consumers (non-null on continuation root).
     * @param name Exact original local route-slot producer node name.
     */
    ROCmMappedRetainedStage(
        std::shared_ptr<Binding> binding,
        const MoERoutingStage::Params &router,
        const SharedExpertFFNStage::Params &shared,
        std::shared_ptr<Qwen4ExpHCReadPhaseOwner> prefix,
        CUDARetainedEarlyRouteReaders early_readers,
        std::string name)
        : IComputeStage(binding->device()),
          binding_(std::move(binding)),
          router_(router),
          shared_(shared),
          prefix_(std::move(prefix)),
          early_readers_(std::move(early_readers)),
          name_(std::move(name))
    {
        const auto &p = binding_->params;
        const auto &original = binding_->original;
        if (!device().is_rocm() || binding_->local_result != LocalResult::RouteSlots ||
            binding_->route_authority != RouteAuthority::OriginalRouter || binding_->prepared ||
            p.rows != 1 || p.phase != MoEOverlayServicePhaseHint::Decode ||
            binding_->format != DeviceMoEWeightFormat::NativeQuantised ||
            !router_.gate_weights || router_.gate_weights->native_type() != TensorType::FP16 ||
            router_.decode_route_publication != MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable ||
            shared_.device_id != device() || shared_.input != p.hidden.tensor ||
            shared_.input_buffer_id != p.hidden.id || !shared_.output ||
            shared_.seq_len != p.rows || shared_.d_model != int(original.modelColumns()) ||
            shared_.active_row_count_device != router_.active_row_count_device ||
            shared_.required_router_q8_publication || shared_.force_decode_equivalent_verifier_prefill ||
            shared_.force_grouped_verifier_prefill_for_decode || shared_.disable_grouped_decode_shortcut ||
            !shared_.prepared_store || !shared_.prepared_ref_gate || !shared_.prepared_ref_up ||
            !shared_.prepared_ref_down || !prefix_)
        {
            throw std::invalid_argument("ROCm mapped retained QFN requires original native mapped M1 graph operands");
        }
        const auto prepared_shared = sharedDescriptor();
        execution_ = resolveExecutionMasks(prepared_shared);
        if (!rocm::fastMappedFFNSuffixKernelFunction(execution_))
        {
            throw std::invalid_argument(
                "ROCm mapped retained QFN has no fast suffix kernel for layer " +
                std::to_string(binding_->layer) + " on " + device().toString());
        }
        prefix_->sealFullOutputDemand(output_demand_);
        prefix_inputs_ = prefix_->phaseInputs();
        prefix_outputs_ = prefix_->phaseOutputs();
        prefix_contract_ = prefix_->phaseDeclaration().bufferContract();
    }

    ComputeStageType type() const override { return ComputeStageType::MOE_PROJECTION_PHASE; }
    std::string name() const override { return name_; }
    bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::GPU_ROCM; }
    bool isGraphCapturable() const override { return program_.has_value(); }
    bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
    bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
    bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
    {
        return GraphLaunchPreparationPolicy::CaptureOnly;
    }
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }

    StageBufferContract bufferContract() const override
    {
        auto contract = prefix_contract_;
        contract.addWeight(router_.gate_weights);
        for (const auto &output : outputs())
            contract.addOutput(output.id);
        contract.addPreparedWeight(shared_.gate_w, shared_.prepared_store, *shared_.prepared_ref_gate);
        contract.addPreparedWeight(shared_.up_w, shared_.prepared_store, *shared_.prepared_ref_up);
        return contract.addPreparedWeight(shared_.down_w, shared_.prepared_store, *shared_.prepared_ref_down);
    }

    StageBufferRequirements getBufferRequirements() const override
    {
        auto requirements = prefix_->phaseDeclaration().getBufferRequirements();
        requirements.addOutput("routing_indices", router_.output_indices->shape(), BufferTensorType::FP32);
        requirements.addOutput("routing_weights", router_.output_weights->shape(), BufferTensorType::FP32);
        requirements.addOutput("local_routed_slots", binding_->params.route_partials.tensor->shape(), BufferTensorType::FP32);
        requirements.addOutput("local_shared", shared_.output->shape(), BufferTensorType::FP32);
        return requirements;
    }

    StageDumpInfo buildDumpInfoImpl() const override
    {
        auto info = prefix_->phaseDeclaration().getDumpInfoSnapshot();
        info.addWeight("router_gate", router_.gate_weights);
        info.addOutput("routing_indices", router_.output_indices, 1, router_.top_k);
        info.addOutput("routing_weights", router_.output_weights, 1, router_.top_k);
        info.addOutput("local_routed_slots", binding_->params.route_partials.tensor,
                       router_.top_k, router_.d_model);
        info.addOutput("local_shared", shared_.output, 1, router_.d_model);
        return info;
    }

    WorkspaceRequirements getWorkspaceRequirements(int, int = 0, int = 0) const override
    {
        auto requirements = prefix_->phaseWorkspaceRequirements();
        MoEWorkspaceBuffers::add(
            requirements, MoEWorkspaceBuffers::ROUTE_LOGITS,
            std::size_t(router_.num_experts) * sizeof(float));
        const std::size_t gate_values =
            std::size_t(router_.top_k) * binding_->original.localIntermediateColumns() +
            static_cast<std::size_t>(shared_.intermediate);
        MoEWorkspaceBuffers::add(
            requirements, MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,
            gate_values * sizeof(float));
        MoEWorkspaceBuffers::add(
            requirements, MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,
            FullNativeHCFFNBarrierLayout::bytes * MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
            WorkspaceContentLifetime::SerialGraphFamily);
        return requirements;
    }

    void bindWorkspace(DeviceWorkspaceManager *workspace) override
    {
        if (program_ && workspace != workspace_)
            throw std::logic_error("ROCm mapped retained QFN cannot rebind a prepared workspace");
        prefix_->bindPhaseWorkspace(workspace);
        workspace_ = workspace;
    }

    void unbindWorkspace() override { bindWorkspace(nullptr); }
    bool hasWorkspace() const override { return workspace_ != nullptr; }
    DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

    bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
    {
        if (!ctx || ctx->deviceId() != device() || !stream || !workspace_ || isGraphCaptureActive())
            throw std::logic_error("ROCm mapped retained QFN requires cold preparation on its original stream and workspace");
        if (program_)
        {
            validateStorage(stream);
            return true;
        }
        if (router_.collect_device_runtime_histogram)
            binding_->params.runtime->prepareDecodeHistogramProducerStream(stream);
        prefix_->preparePhaseBindings(ctx, stream);

        const auto addresses = tensorAddresses();
        const auto prepared_shared = sharedDescriptor();
        const auto &p = binding_->params;
        auto *runtime = p.runtime->deviceLayerState(binding_->layer);
        if (!runtime)
            throw std::logic_error("ROCm mapped retained QFN lost its original device runtime table");

        const std::size_t route_logits_bytes = std::size_t(router_.num_experts) * sizeof(float);
        const std::size_t gate_partials_bytes =
            (std::size_t(router_.top_k) * binding_->original.localIntermediateColumns() +
             static_cast<std::size_t>(shared_.intermediate)) * sizeof(float);
        auto *route_logits = static_cast<float *>(workspace_->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS));
        auto *gate_partials = static_cast<float *>(workspace_->getBuffer(MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS));
        if (!route_logits || workspace_->getBufferSize(MoEWorkspaceBuffers::ROUTE_LOGITS) < route_logits_bytes ||
            !gate_partials || workspace_->getBufferSize(MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS) < gate_partials_bytes)
        {
            throw std::logic_error("ROCm mapped retained QFN lacks admitted router/gate scratch workspace");
        }

        auto barrier_slot = workspace_->acquirePersistentSlot(
            MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,
            MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
        if (!barrier_slot)
            throw std::logic_error("ROCm mapped retained QFN could not acquire a persistent barrier slot");
        constexpr std::size_t kBarrierBytes = FullNativeHCFFNBarrierLayout::bytes;
        auto *barrier_base = static_cast<std::uint32_t *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,
            MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
            barrier_slot->slot(), kBarrierBytes));
        if (!barrier_base)
            throw std::logic_error("ROCm mapped retained QFN barrier slot address is unavailable");
        const auto barrier_offset = reinterpret_cast<const std::byte *>(barrier_base) -
            static_cast<const std::byte *>(workspace_->getBuffer(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER));
        auto barrier_lease = workspace_->retainBuffer(
            MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER, barrier_offset, kBarrierBytes);
        TransferEngine::instance().initializeWorkspaceMetadata(
            *barrier_lease, std::vector<std::byte>(kBarrierBytes, std::byte{0}), stream);

        rocm::FullNativeHCFFNPrefixArgs hc_prefix = prefix_->fullPrefixArguments();
        hc_prefix.barrier_state = barrier_base;
        hc_prefix.output_demand = output_demand_;
        hc_prefix.selected_descriptor_demand = MoESelectedDescriptorDemand::AcquiredBankOnly;

        const DeviceRowRange rows = p.live_rows.value_or(DeviceRowRange::fullyActive(1));
        rocm::MappedRoutePrefixArgs prefix_args{
            .prefix = hc_prefix,
            .runtime = runtime,
            .original = binding_->original,
            .router_weights = static_cast<const std::uint16_t *>(addresses[0]),
            .router_logits = route_logits,
            .router_indices = static_cast<float *>(addresses[1]),
            .router_weights_output = static_cast<float *>(addresses[2]),
            .rows = rows,
            .experts = router_.num_experts,
            .top_k = router_.top_k,
            .normalize_router_weights = router_.norm_topk_prob,
            .observe_histogram = router_.collect_device_runtime_histogram,
        };

        rocm::MappedFFNSuffixArgs suffix_args{
            .barrier_state = barrier_base,
            .runtime = runtime,
            .original = binding_->original,
            .hidden = hc_prefix.hc.mixed,
            .rows = rows,
            .hidden_columns = static_cast<int>(binding_->original.modelColumns()),
            .routed_columns = static_cast<int>(binding_->original.localIntermediateColumns()),
            .shared_columns = shared_.intermediate,
            .experts = router_.num_experts,
            .top_k = router_.top_k,
            .shared_gate = prepared_shared.gate,
            .shared_up = prepared_shared.up,
            .shared_down = prepared_shared.down,
            .execution = execution_,
            .gate_partials = gate_partials,
            .route_contributions = static_cast<float *>(addresses[3]),
            .shared_output = static_cast<float *>(addresses[4]),
        };

        auto program = rocm::prepareMappedRetainedHCFFN(device(), prefix_args, suffix_args, workspace_->id());
        if (early_readers_.mapped_dispatch)
            early_readers_.mapped_dispatch->prepare(stream);
        if (early_readers_.cpu_ticket)
            early_readers_.cpu_ticket->prepare(stream);

        prepared_stream_ = stream;
        prepared_workspace_id_ = workspace_->id();
        prepared_tensors_ = addresses;
        prepared_route_logits_ = route_logits;
        prepared_gate_partials_ = gate_partials;
        prepared_barrier_ = barrier_base;
        prepared_runtime_ = runtime;
        prepared_shared_ = prepared_shared;
        prefix_args_.emplace(prefix_args);
        suffix_args_.emplace(suffix_args);
        barrier_slot_ = std::move(barrier_slot);
        barrier_lease_ = std::move(barrier_lease);
        program_.emplace(std::move(program));
        setGPUStream(stream);

        PerfStatsCollector::addCounter(
            "moe_resident_ffn", "rocm_mapped_prepared_program", 1., "setup", device().toString(),
            {{"layer", std::to_string(binding_->layer)},
             {"prefix_grid", std::to_string(program_->prefix_grid)},
             {"suffix_grid", std::to_string(program_->suffix_grid)},
             {"prefix_vgprs", std::to_string(program_->prefix_resources.registers_per_thread)},
             {"suffix_vgprs", std::to_string(program_->suffix_resources.registers_per_thread)}});
        return true;
    }

    bool execute(IDeviceContext *ctx) override
    {
        if (!ctx || ctx->deviceId() != device() || !program_)
            throw std::logic_error("ROCm mapped retained QFN has no prepared original program");
        const auto execution = gpuExecution();
        validateStorage(execution.nativeStream());
        for (const auto &input : prefix_inputs_)
            execution.requirePreparedInput(input.tensor);
        for (auto *weight : prefix_contract_.weight_tensors)
            execution.requirePreparedInput(weight);
        execution.requirePreparedInput(router_.gate_weights);
        for (const auto &output : prefix_outputs_)
            execution.requirePreparedOutput(output.tensor);
        for (const auto &output : outputs())
            execution.requirePreparedOutput(output.tensor);

        if (early_readers_.mapped_dispatch || early_readers_.cpu_ticket)
        {
            rocm::launchMappedRetainedHCFFN(
                *program_, *prefix_args_, *suffix_args_, execution.nativeStream(), prepared_workspace_id_,
                [&] {
                    if (early_readers_.mapped_dispatch)
                        early_readers_.mapped_dispatch->recordCompletedProducer(execution.nativeStream());
                    if (early_readers_.cpu_ticket)
                        early_readers_.cpu_ticket->recordCompletedProducer(execution.nativeStream());
                });
        }
        else
        {
            rocm::launchMappedRetainedHCFFN(
                *program_, *prefix_args_, *suffix_args_, execution.nativeStream(), prepared_workspace_id_);
        }

        for (const auto &output : prefix_outputs_)
            execution.publish(output.tensor);
        for (const auto &output : outputs())
            execution.publish(output.tensor);

        PerfStatsCollector::addCounter(
            "moe_resident_ffn", "rocm_mapped_local_program_enqueued", 1., "graph_capture",
            device().toString(),
            {{"layer", std::to_string(binding_->layer)},
             {"early_cpu_packet", early_readers_.mapped_dispatch ? "1" : "0"},
             {"early_cpu_ticket", early_readers_.cpu_ticket ? "1" : "0"}});
        return true;
    }

    void invalidateKernelDynamicState() override
    {
        program_.reset();
        prefix_args_.reset();
        suffix_args_.reset();
        barrier_lease_.reset();
        barrier_slot_.reset();
        prefix_->invalidatePhaseBindings();
    }

    std::uint64_t graphCaptureVariantSignature() const override
    {
        if (program_)
            validateStorage(prepared_stream_);
        auto signature = binding_->signature;
        const auto mix = [&](std::uint64_t value) { signature = (signature ^ value) * kFNV1a64Prime; };
        mix(prefix_->phaseDeclaration().graphCaptureVariantSignature());
        mix(execution_.routed_gate);
        mix(execution_.routed_down);
        mix(execution_.shared_gate);
        mix(execution_.shared_down);
        mix(output_demand_.bits());
        mix(router_.norm_topk_prob);
        mix(router_.collect_device_runtime_histogram);
        mix(reinterpret_cast<std::uintptr_t>(router_.gate_weights));
        mix(reinterpret_cast<std::uintptr_t>(early_readers_.mapped_dispatch.get()));
        mix(reinterpret_cast<std::uintptr_t>(early_readers_.cpu_ticket.get()));
        mix(reinterpret_cast<std::uintptr_t>(shared_.prepared_store));
        mix(shared_.prepared_ref_gate->binding_id);
        mix(shared_.prepared_ref_up->binding_id);
        mix(shared_.prepared_ref_down->binding_id);
        mix(shared_.intermediate);
        mix(reinterpret_cast<std::uintptr_t>(shared_.output));
        mix(static_cast<unsigned>(shared_.output_buffer_id));
        return signature ? signature : 1;
    }

private:
    using Addresses = std::array<void *, 5>;

    std::array<MoEIntermediateShardTensorBinding, 4> outputs() const
    {
        return {{{router_.output_indices, router_.output_indices_buffer_id},
                 {router_.output_weights, router_.output_weights_buffer_id},
                 binding_->params.route_partials,
                 {shared_.output, shared_.output_buffer_id}}};
    }

    Addresses tensorAddresses() const
    {
        Addresses addresses{};
        addresses[0] = router_.gate_weights->gpu_data_ptr();
        const auto publications = outputs();
        for (std::size_t i = 0; i < publications.size(); ++i)
            addresses[i + 1] = publications[i].tensor->gpu_data_ptr();
        if (std::find(addresses.begin(), addresses.end(), nullptr) != addresses.end())
            throw std::logic_error("ROCm mapped retained QFN original tensors are not materialized");
        return addresses;
    }

    DeviceMoEExpertDescriptor sharedDescriptor() const
    {
        auto &store = *shared_.prepared_store;
        DeviceMoEExpertDescriptor descriptor{};
        if (!exportDeviceMoEExpertWeightDescriptors(
                store.gemmKernel(*shared_.prepared_ref_gate),
                store.gemmKernel(*shared_.prepared_ref_up),
                store.gemmKernel(*shared_.prepared_ref_down),
                shared_.d_model, shared_.intermediate, descriptor) ||
            descriptor.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            !descriptor.gate.valid() || !descriptor.up.valid() || !descriptor.down.valid() ||
            descriptor.gate.codebook_id != descriptor.up.codebook_id ||
            descriptor.gate.codebook_id != descriptor.down.codebook_id)
        {
            throw std::invalid_argument("ROCm mapped retained QFN requires valid NativeQuantised shared descriptors");
        }
        return descriptor;
    }

    rocm::ResidentMoEFFNExecutionMasks resolveExecutionMasks(const DeviceMoEExpertDescriptor &shared_desc) const
    {
        const auto gate = binding_->descriptor_declaration.nativeGateUpEnvelope();
        const auto down = binding_->descriptor_declaration.nativeDownEnvelope();
        const std::uint32_t shared_mask = nativeQuantisedCodebookMaskBit(shared_desc.gate.codebook_id);
        rocm::ResidentMoEFFNExecutionMasks masks{};
        masks.routed_gate = gate.execution_codebook_mask;
        masks.routed_down = down.execution_codebook_mask;
        masks.shared_gate = shared_mask;
        masks.shared_down = shared_mask;
        return masks;
    }

    void validateStorage(void *stream) const
    {
        const auto descriptor = sharedDescriptor();
        const auto same = [](const auto &a, const auto &b) { return std::memcmp(&a, &b, sizeof(a)) == 0; };
        if (!workspace_ || stream != prepared_stream_ || workspace_->id() != prepared_workspace_id_ ||
            tensorAddresses() != prepared_tensors_ ||
            workspace_->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS) != prepared_route_logits_ ||
            workspace_->getBuffer(MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS) != prepared_gate_partials_ ||
            !barrier_slot_ ||
            workspace_->getPersistentSlotBuffer(
                MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                barrier_slot_->slot(), FullNativeHCFFNBarrierLayout::bytes) != prepared_barrier_ ||
            binding_->params.runtime->deviceLayerState(binding_->layer) != prepared_runtime_ ||
            !same(descriptor.gate, prepared_shared_.gate) ||
            !same(descriptor.up, prepared_shared_.up) ||
            !same(descriptor.down, prepared_shared_.down))
        {
            throw std::logic_error("ROCm mapped retained QFN capture storage changed");
        }
        prefix_->validatePhaseBindings(stream);
    }

    const std::shared_ptr<Binding> binding_;
    const MoERoutingStage::Params router_;
    const SharedExpertFFNStage::Params shared_;
    const std::shared_ptr<Qwen4ExpHCReadPhaseOwner> prefix_;
    const CUDARetainedEarlyRouteReaders early_readers_;
    const std::string name_;
    const FullNativeHCFFNOutputDemand output_demand_ = FullNativeHCFFNOutputDemand::none();
    rocm::ResidentMoEFFNExecutionMasks execution_{};
    std::vector<Qwen4ExpFloatBinding> prefix_inputs_;
    std::vector<Qwen4ExpFloatBinding> prefix_outputs_;
    StageBufferContract prefix_contract_{};
    DeviceWorkspaceManager *workspace_ = nullptr;
    void *prepared_stream_ = nullptr;
    std::uint64_t prepared_workspace_id_ = 0u;
    Addresses prepared_tensors_{};
    void *prepared_route_logits_ = nullptr;
    void *prepared_gate_partials_ = nullptr;
    void *prepared_barrier_ = nullptr;
    DeviceMoELayerRuntime *prepared_runtime_ = nullptr;
    DeviceMoEExpertDescriptor prepared_shared_{};
    std::optional<rocm::MappedRoutePrefixArgs> prefix_args_;
    std::optional<rocm::MappedFFNSuffixArgs> suffix_args_;
    std::shared_ptr<PersistentWorkspaceSlotLease> barrier_slot_;
    std::shared_ptr<const WorkspaceBufferLease> barrier_lease_;
    std::optional<rocm::PreparedMappedRetainedHCFFN> program_;
};

#endif // HAVE_ROCM
