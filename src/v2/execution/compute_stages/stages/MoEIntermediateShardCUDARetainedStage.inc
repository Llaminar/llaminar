/**
 * @file MoEIntermediateShardCUDARetainedStage.inc
 * @brief Binds the CUDA cooperative QFN program to the original mixed decode graph.
 *
 * The HC owner retains its original inputs, weights and workspace. The live
 * routed weights belong to the acquired runtime bank. The stage publishes raw
 * route slots and the shared FFN output for the existing ticket and TP graph.
 */

/** @brief Immutable routed-expert binding abstracted across TensorSharded and Apportioned CUDA graphs. */
struct CUDARetainedRoutedContract
{
    DeviceId device = DeviceId::invalid();
    int layer = -1;
    int d_model = 0;
    int routed_width = 0;
    int top_k = 0;
    IMoERuntimeTable* runtime = nullptr;
    RoutedExpertRowExecutionPolicy row_policy = RoutedExpertRowExecutionPolicy::ParticipantAssigned;
    MoEIntermediateShardTensorBinding hidden{};
    MoEIntermediateShardTensorBinding route_partials{};
    cuda::moe::RetainedM1RoutePublication route_publication = cuda::moe::RetainedM1RoutePublication::Raw;
    uint8_t gate_codebook = 0;
    uint8_t down_codebook = 0;
    uint8_t gate_promoted_codebook = 0xffu;
    uint8_t down_promoted_codebook = 0xffu;
    uint8_t gate_source_superblock = 0;
    uint8_t down_source_superblock = 0;
    std::uint64_t signature = 0;
};

/** @brief Preparation and publication of one participant's complete local CUDA region. */
class CUDARetainedStage final : public IComputeStage, public IWorkspaceConsumer
{
public:
    /**
     * @brief Authenticate the original TensorSharded graph rather than deriving a second FFN placement.
     */
    CUDARetainedStage(std::shared_ptr<Binding> binding, const MoERoutingStage::Params& router,
        const SharedExpertFFNStage::Params& shared, std::shared_ptr<Qwen4ExpHCReadPhaseOwner> prefix,
        CUDARetainedEarlyRouteReaders early_readers, std::string name)
        : IComputeStage(binding->device()), binding_(std::move(binding)), router_(router), shared_(shared),
          prefix_(std::move(prefix)), early_readers_(std::move(early_readers)), name_(std::move(name))
    {
        const auto& p = binding_->params;
        const auto& original = binding_->original;
        if (!device().is_cuda() || binding_->local_result != LocalResult::RouteSlots ||
            binding_->route_authority != RouteAuthority::OriginalRouter || binding_->prepared ||
            p.rows != 1 || p.phase != MoEOverlayServicePhaseHint::Decode ||
            binding_->format != DeviceMoEWeightFormat::NativeQuantised)
            throw std::invalid_argument("CUDA retained QFN requires original native mapped M1 graph operands");
        routed_ = CUDARetainedRoutedContract{
            .device = binding_->device(),
            .layer = binding_->layer,
            .d_model = static_cast<int>(original.modelColumns()),
            .routed_width = static_cast<int>(original.localIntermediateColumns()),
            .top_k = p.top_k,
            .runtime = p.runtime,
            .row_policy = p.row_policy,
            .hidden = p.hidden,
            .route_partials = p.route_partials,
            .route_publication = cuda::moe::RetainedM1RoutePublication::Raw,
            .signature = binding_->signature,
        };
        resolveCodebooksFromIntermediateShard();
        finalizeConstruction();
    }

    /**
     * @brief Authenticate an original Apportioned TieredOverlay CUDA graph (e.g. 2xCUDA+4xROCm).
     */
    CUDARetainedStage(const MoEExpertComputeStage::Params& compute, const MoERoutingStage::Params& router,
        const SharedExpertFFNStage::Params& shared, std::shared_ptr<Qwen4ExpHCReadPhaseOwner> prefix,
        CUDARetainedEarlyRouteReaders early_readers, std::string name)
        : IComputeStage(compute.device_id), router_(router), shared_(shared),
          prefix_(std::move(prefix)), early_readers_(std::move(early_readers)), name_(std::move(name))
    {
        if (!device().is_cuda() || compute.seq_len != 1 || !compute.moe_runtime_table ||
            !compute.canonical_route_contributions ||
            compute.canonical_route_layout != MoECanonicalRoutePublicationLayout::DenseOriginalRouteSlots ||
            (compute.weight_descriptor_source != MoEDecodeDescriptorSource::RuntimePlacementTable &&
             compute.weight_descriptor_source != MoEDecodeDescriptorSource::StaticDescriptorTable) ||
            compute.d_model <= 0 || compute.expert_intermediate <= 0 ||
            (compute.expert_intermediate % 32) != 0 ||
            compute.input != router_.input)
            throw std::invalid_argument("CUDA retained QFN Apportioned requires original native runtime-table M1 operands");
        const auto route_pub = compute.canonical_route_arithmetic ==
                MoECanonicalRouteArithmeticPolicy::PreweightedContributionThenOrderedAdd
            ? cuda::moe::RetainedM1RoutePublication::Preweighted
            : cuda::moe::RetainedM1RoutePublication::Raw;
        std::uint64_t sig = kFNV1a64OffsetBasis;
        const auto mix = [&](std::uint64_t v) { sig = (sig ^ v) * kFNV1a64Prime; };
        mix(static_cast<std::uint64_t>(compute.device_id.ordinal));
        mix(static_cast<std::uint64_t>(compute.layer_idx));
        mix(static_cast<std::uint64_t>(compute.d_model));
        mix(static_cast<std::uint64_t>(compute.expert_intermediate));
        mix(static_cast<std::uint64_t>(compute.top_k));
        mix(static_cast<std::uint64_t>(compute.routed_row_execution_policy));
        mix(static_cast<std::uint64_t>(route_pub));
        mix(reinterpret_cast<std::uintptr_t>(compute.moe_runtime_table));
        mix(reinterpret_cast<std::uintptr_t>(compute.canonical_route_contributions));
        routed_ = CUDARetainedRoutedContract{
            .device = compute.device_id,
            .layer = compute.layer_idx,
            .d_model = compute.d_model,
            .routed_width = compute.expert_intermediate,
            .top_k = compute.top_k,
            .runtime = compute.moe_runtime_table,
            .row_policy = compute.routed_row_execution_policy,
            .hidden = {compute.input, compute.input_buffer_id},
            .route_partials = {compute.canonical_route_contributions, compute.canonical_route_contributions_buffer_id},
            .route_publication = route_pub,
            .signature = sig,
        };
        resolveCodebooksFromPreparedEngines(compute);
        finalizeConstruction();
    }

    /** @return Classification of the original local routed computation. */
    ComputeStageType type() const override { return ComputeStageType::MOE_PROJECTION_PHASE; }
    /** @return Exact original route-slot producer name. */
    std::string name() const override { return name_; }
    /** @return CUDA support for this explicit native program. */
    bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::GPU_CUDA; }
    /** @return Whether the native entry and its original workspace are sealed. */
    bool isGraphCapturable() const override { return program_.has_value(); }
    /** @return Cold preparation never runs a forward. */
    bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
    /** @return Live single-row launch eligibility is consumed on-device. */
    bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
    /** @return Cold graph setup supplies all launch operands. */
    bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
    /** @return Native setup precedes capture. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
    { return GraphLaunchPreparationPolicy::CaptureOnly; }
    /** @return The exact producer stream controls publication of the original outputs. */
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }

    /** @return Original public tensor and store-owned source declarations. */
    StageBufferContract bufferContract() const override
    {
        auto contract = prefix_contract_;
        contract.addWeight(router_.gate_weights);
        for (const auto& output : outputs())
            contract.addOutput(output.id);
        contract.addPreparedWeight(shared_.gate_w, shared_.prepared_store, *shared_.prepared_ref_gate);
        contract.addPreparedWeight(shared_.up_w, shared_.prepared_store, *shared_.prepared_ref_up);
        return contract.addPreparedWeight(shared_.down_w, shared_.prepared_store, *shared_.prepared_ref_down);
    }

    /** @return Tensor shapes for the original public inputs, weights and outputs. */
    StageBufferRequirements getBufferRequirements() const override
    {
        auto requirements = prefix_->phaseDeclaration().getBufferRequirements();
        requirements.addOutput("routing_indices", router_.output_indices->shape(), BufferTensorType::FP32);
        requirements.addOutput("routing_weights", router_.output_weights->shape(), BufferTensorType::FP32);
        requirements.addOutput("local_routed_slots", routed_.route_partials.tensor->shape(), BufferTensorType::FP32);
        requirements.addOutput("local_shared", shared_.output->shape(), BufferTensorType::FP32);
        return requirements;
    }

    /** @return Original public HC and MoE tensors for production capture snapshots. */
    StageDumpInfo buildDumpInfoImpl() const override
    {
        auto info = prefix_->phaseDeclaration().getDumpInfoSnapshot();
        info.addWeight("router_gate", router_.gate_weights);
        info.addOutput("routing_indices", router_.output_indices, 1, router_.top_k);
        info.addOutput("routing_weights", router_.output_weights, 1, router_.top_k);
        info.addOutput("local_routed_slots", routed_.route_partials.tensor,
            router_.top_k, router_.d_model);
        info.addOutput("local_shared", shared_.output, 1, router_.d_model);
        return info;
    }

    /**
     * @brief Declare disposable banks for this participant's exact retained row.
     * @return Original HC banks, original router logits and compact FFN work.
     */
    WorkspaceRequirements getWorkspaceRequirements(int, int = 0, int = 0) const override
    {
        auto requirements = prefix_->phaseWorkspaceRequirements();
        for (const auto& bank : banks())
            MoEWorkspaceBuffers::add(requirements, bank.name, bank.bytes);
        return requirements;
    }

    /** @brief Borrow the executor's admitted workspace. @param workspace Original owner. */
    void bindWorkspace(DeviceWorkspaceManager* workspace) override
    {
        if (program_ && workspace != workspace_)
            throw std::logic_error("CUDA retained QFN cannot rebind a prepared workspace");
        prefix_->bindPhaseWorkspace(workspace);
        workspace_ = workspace;
    }
    /** @brief Release the workspace after native graph retirement. */
    void unbindWorkspace() override { bindWorkspace(nullptr); }
    /** @return Whether the original workspace was admitted. */
    bool hasWorkspace() const override { return workspace_ != nullptr; }
    /** @return The current physical workspace owner. */
    DeviceWorkspaceManager* getWorkspace() const override { return workspace_; }

    /**
     * @brief Resolve published addresses and check exact native admission.
     * @param ctx Current participant's executor context.
     * @param stream Original non-default compute stream.
     * @return True once the complete program has been prepared.
     */
    bool prepareGraphLaunch(IDeviceContext* ctx, void* stream) override
    {
        if (!ctx || ctx->deviceId() != device() || !stream || !workspace_ || isGraphCaptureActive())
            throw std::logic_error("CUDA retained QFN requires cold preparation on its original stream and workspace");
        if (program_)
        {
            validateStorage(stream);
            return true;
        }
        if (router_.collect_device_runtime_histogram)
            routed_.runtime->prepareDecodeHistogramProducerStream(stream);
        prefix_->preparePhaseBindings(ctx, stream);
        const auto pointers = bankAddresses();
        const auto addresses = tensorAddresses();
        const auto prepared_shared = sharedDescriptor();
        auto* runtime = routed_.runtime->deviceLayerState(routed_.layer);
        if (!runtime)
            throw std::logic_error("CUDA retained QFN lost its original device runtime table");
        auto* codes = static_cast<int8_t*>(pointers[DownCodes]);
        auto* scales = static_cast<float*>(pointers[DownScales]);
        const int width = routed_.routed_width;
        const int shared_offset = width * routed_.top_k;
        const cuda::moe::RetainedM1HCFFN arguments{
            .hc = prefix_->cudaPrefixArguments(),
            .router = {
                .runtime = runtime, .weights = static_cast<const uint16_t*>(addresses[0]),
                .logits = static_cast<float*>(pointers[RouterLogits]),
                .route_indices = static_cast<float*>(addresses[1]), .route_weights = static_cast<float*>(addresses[2]),
                .position = router_.absolute_position_ids_device, .experts = router_.num_experts, .top_k = router_.top_k,
                .local_policy = routed_.row_policy, .normalize = router_.norm_topk_prob,
            },
            .shared_gate_input = nullptr,
            .shared = {prepared_shared.gate, prepared_shared.up, prepared_shared.down},
            .selected = static_cast<cuda::moe::RetainedM1Expert*>(pointers[Descriptors]),
            .route_order = static_cast<int*>(pointers[RouteOrder]),
            .route_plan_ready = static_cast<int*>(pointers[RoutePlanReady]), .routed_width = width,
            .routed_workspace = {codes, scales},
            .shared_workspace = {codes + shared_offset, scales + shared_offset / 32},
            .hidden_codes = static_cast<int8_t*>(pointers[HiddenCodes]),
            .hidden_scales = static_cast<float*>(pointers[HiddenScales]),
            .shared_gate_factor = nullptr,
            .routed_contributions = static_cast<float*>(addresses[3]),
            .shared_contribution = static_cast<float*>(addresses[4]),
            .suffix_partials = static_cast<float*>(pointers[SuffixPartials]),
            .suffix_done = static_cast<uint32_t*>(pointers[SuffixDone]),
            .suffix_gate_up_node = nullptr,
            .suffix_down_node = nullptr,
            .route_publication = routed_.route_publication,
            .shared_publication = cuda::moe::RetainedM1SharedPublication::PartialBeforeCollective,
            .gate_codebook = routed_.gate_codebook,
            .down_codebook = routed_.down_codebook,
            .gate_promoted_codebook = routed_.gate_promoted_codebook,
            .down_promoted_codebook = routed_.down_promoted_codebook,
            .gate_source_superblock = routed_.gate_source_superblock,
            .down_source_superblock = routed_.down_source_superblock,
        };
        using Ordering = cuda::moe::RetainedM1PacketOrdering;
        const auto ordering = (!arguments.hc.norm.extent.liveRows ||
            early_readers_.mapped_dispatch || early_readers_.cpu_ticket)
            ? Ordering::NativeCompletion : Ordering::CompleteProgram;
        auto program = cuda::moe::RetainedM1Program::prepare(device(), arguments, routed_.gate_codebook, routed_.down_codebook,
            output_demand_, router_.collect_device_runtime_histogram, workspace_->id(), ordering);
        if (program.admission().resources().private_bytes)
            throw std::logic_error("CUDA retained QFN requires a spill-free resident entry");
        if (early_readers_.mapped_dispatch)
            early_readers_.mapped_dispatch->prepare(stream);
        if (early_readers_.cpu_ticket)
            early_readers_.cpu_ticket->prepare(stream);
        prepared_stream_ = stream;
        prepared_workspace_id_ = workspace_->id();
        prepared_banks_ = pointers;
        prepared_tensors_ = addresses;
        prepared_runtime_ = runtime;
        prepared_shared_ = prepared_shared;
        validated_store_epoch_ = shared_.prepared_store->mutationEpoch();
        program_.emplace(std::move(program));
        setGPUStream(stream);
        const auto& native = program_->admission();
        PerfStatsCollector::addCounter("moe_resident_ffn", "cuda_mapped_prepared_program", 1., "setup",
            device().toString(), {{"layer", std::to_string(routed_.layer)},
                {"routed_width", std::to_string(routed_.routed_width)},
                {"shared_width", std::to_string(shared_.intermediate)},
                {"threads", std::to_string(native.threads())},
                {"registers_per_thread", std::to_string(native.resources().registers_per_thread)}});
        return true;
    }

    /**
     * @brief Record the cooperative entry and publish its original visible graph buffers.
     * @param ctx Current participant's executor context.
     * @return True after successful exact-stream recording.
     */
    bool execute(IDeviceContext* ctx) override
    {
        if (!ctx || ctx->deviceId() != device() || !program_)
            throw std::logic_error("CUDA retained QFN has no prepared original program");
        const auto execution = gpuExecution();
        validateStorage(execution.nativeStream());
        for (const auto& input : prefix_inputs_)
            execution.requirePreparedInput(input.tensor);
        for (auto* weight : prefix_contract_.weight_tensors)
            execution.requirePreparedInput(weight);
        execution.requirePreparedInput(router_.gate_weights);
        for (const auto& output : prefix_outputs_)
            execution.requirePreparedOutput(output.tensor);
        for (const auto& output : outputs())
            execution.requirePreparedOutput(output.tensor);
        if (early_readers_.mapped_dispatch || early_readers_.cpu_ticket)
        {
            program_->launch(execution.nativeStream(), prepared_workspace_id_, [&] {
                if (early_readers_.mapped_dispatch)
                    early_readers_.mapped_dispatch->recordCompletedProducer(execution.nativeStream());
                if (early_readers_.cpu_ticket)
                    early_readers_.cpu_ticket->recordCompletedProducer(execution.nativeStream());
            });
        }
        else
        {
            program_->launch(execution.nativeStream(), prepared_workspace_id_);
        }
        for (const auto& output : prefix_outputs_)
            execution.publish(output.tensor);
        for (const auto& output : outputs())
            execution.publish(output.tensor);
        PerfStatsCollector::addCounter("moe_resident_ffn", "cuda_mapped_local_program_enqueued", 1.,
            "graph_capture", device().toString(), {{"layer", std::to_string(routed_.layer)},
                {"early_cpu_packet", early_readers_.mapped_dispatch ? "1" : "0"},
                {"early_cpu_ticket", early_readers_.cpu_ticket ? "1" : "0"}});
        return true;
    }

    /** @brief Retire the enclosing program before releasing original HC leases. */
    void invalidateKernelDynamicState() override
    {
        program_.reset();
        validated_store_epoch_ = 0;
        prefix_->invalidatePhaseBindings();
    }

    /** @return Frozen source, geometry and policy identity, validated against physical capture leases. */
    std::uint64_t graphCaptureVariantSignature() const override
    {
        if (program_)
            validateStorage(prepared_stream_, /*validate_prefix=*/false);
        auto signature = routed_.signature;
        const auto mix = [&](std::uint64_t value) { signature = (signature ^ value) * kFNV1a64Prime; };
        mix(prefix_->phaseDeclaration().graphCaptureVariantSignature());
        mix(routed_.gate_codebook); mix(routed_.down_codebook);
        mix(routed_.gate_promoted_codebook); mix(routed_.down_promoted_codebook);
        mix(output_demand_.bits());
        mix(router_.norm_topk_prob); mix(router_.collect_device_runtime_histogram);
        mix(reinterpret_cast<std::uintptr_t>(router_.gate_weights));
        mix(reinterpret_cast<std::uintptr_t>(router_.absolute_position_ids_device));
        mix(reinterpret_cast<std::uintptr_t>(early_readers_.mapped_dispatch.get()));
        mix(reinterpret_cast<std::uintptr_t>(early_readers_.cpu_ticket.get()));
        mix(reinterpret_cast<std::uintptr_t>(shared_.prepared_store));
        mix(shared_.prepared_ref_gate->binding_id); mix(shared_.prepared_ref_up->binding_id);
        mix(shared_.prepared_ref_down->binding_id); mix(shared_.intermediate);
        mix(reinterpret_cast<std::uintptr_t>(shared_.output)); mix(static_cast<unsigned>(shared_.output_buffer_id));
        return signature ? signature : 1;
    }

private:
    /** @brief Unambiguous index of every admitted scratch plane in the native operands. */
    enum Bank : std::size_t { RouterLogits, Descriptors, RouteOrder, RoutePlanReady, HiddenCodes, HiddenScales, DownCodes, DownScales, SuffixPartials, SuffixDone, BankCount };
    /** @brief Size of one named workspace plane. */
    struct BankRequirement { const char* name; std::size_t bytes; };
    using Addresses = std::array<void*, 5>;

    /** @brief Common post-contract constructor checks and HC prefix demand seal. */
    void finalizeConstruction()
    {
        if (!router_.gate_weights || router_.gate_weights->native_type() != TensorType::FP16 ||
            router_.decode_route_publication != MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable ||
            shared_.device_id != device() || shared_.input != routed_.hidden.tensor ||
            shared_.input_buffer_id != routed_.hidden.id || !shared_.output ||
            shared_.seq_len != 1 || shared_.d_model != routed_.d_model ||
            shared_.active_row_count_device != router_.active_row_count_device ||
            shared_.required_router_q8_publication || shared_.force_decode_equivalent_verifier_prefill ||
            shared_.force_grouped_verifier_prefill_for_decode || shared_.disable_grouped_decode_shortcut ||
            !shared_.prepared_store || !shared_.prepared_ref_gate || !shared_.prepared_ref_up ||
            !shared_.prepared_ref_down || !prefix_)
            throw std::invalid_argument("CUDA retained QFN requires original native mapped M1 graph operands");
        (void)sharedDescriptor();
        prefix_->sealFullOutputDemand(output_demand_);
        prefix_inputs_ = prefix_->phaseInputs();
        prefix_outputs_ = prefix_->phaseOutputs();
        prefix_contract_ = prefix_->phaseDeclaration().bufferContract();
    }

    /** @return Exact execution scratch by original source width and selected route capacity. */
    std::array<BankRequirement, BankCount> banks() const
    {
        using namespace MoEWorkspaceBuffers;
        const std::size_t top_k = router_.top_k;
        const std::size_t hidden = router_.d_model;
        const std::size_t channels = top_k * static_cast<std::size_t>(routed_.routed_width) +
                                     static_cast<std::size_t>(shared_.intermediate);
        const std::size_t suffix_packets =
            top_k * (static_cast<std::size_t>(routed_.routed_width) / 32u) +
            (static_cast<std::size_t>(shared_.intermediate) / 32u);
        return {{{ROUTE_LOGITS, std::size_t(router_.num_experts) * sizeof(float)},
            {"cuda_retained_qfn_selected_descriptors", top_k * sizeof(cuda::moe::RetainedM1Expert)},
            {"cuda_retained_qfn_route_order", (2 + top_k) * sizeof(int)},
            {"cuda_retained_qfn_route_plan_ready", sizeof(int)},
            {DECODE_HIDDEN_INT8, hidden * sizeof(int8_t)}, {DECODE_HIDDEN_SCALES, hidden / 32 * sizeof(float)},
            {DECODE_SWIGLU_INT8, channels * sizeof(int8_t)}, {DECODE_SWIGLU_SCALES, channels / 32 * sizeof(float)},
            {"cuda_retained_qfn_suffix_partials", suffix_packets * 16 * 2 * 32 * sizeof(float)},
            {"cuda_retained_qfn_suffix_done", suffix_packets * sizeof(uint32_t)}}};
    }

    /** @return Existing graph publications with their original arena identities. */
    std::array<MoEIntermediateShardTensorBinding, 4> outputs() const
    {
        return {{{router_.output_indices, router_.output_indices_buffer_id},
            {router_.output_weights, router_.output_weights_buffer_id}, routed_.route_partials,
            {shared_.output, shared_.output_buffer_id}}};
    }

    /** @return Exact addresses from arena tensors. Capture may not silently change any address. */
    Addresses tensorAddresses() const
    {
        Addresses addresses{};
        addresses.front() = router_.gate_weights->gpu_data_ptr();
        const auto publications = outputs();
        for (std::size_t i = 0; i < publications.size(); ++i)
            addresses[i + 1] = publications[i].tensor->gpu_data_ptr();
        if (std::find(addresses.begin(), addresses.end(), nullptr) != addresses.end())
            throw std::logic_error("CUDA retained QFN original tensors are not materialized");
        return addresses;
    }

    /** @return All materialized named-bank addresses from the admitted workspace. */
    std::array<void*, BankCount> bankAddresses() const
    {
        const auto requirements = banks();
        std::array<void*, BankCount> addresses{};
        for (std::size_t i = 0; i < requirements.size(); ++i)
        {
            const auto& bank = requirements[i];
            addresses[i] = workspace_->getBuffer(bank.name);
            if (!addresses[i] || workspace_->getBufferSize(bank.name) < bank.bytes)
                throw std::logic_error(std::string("CUDA retained QFN lacks admitted workspace ") + bank.name);
        }
        return addresses;
    }

    /** @return Prepared shared source triple from the original store for this participant. */
    DeviceMoEExpertDescriptor sharedDescriptor() const
    {
        auto& store = *shared_.prepared_store;
        DeviceMoEExpertDescriptor descriptor{};
        const auto valid_native = [](const DeviceNativeQuantisedMatrixDesc& matrix) {
            return matrix.valid() && matrix.source_identity_present == 1 &&
                (cuda::moe::retainedQFNM1SuffixGateUpShard0(matrix.codebook_id) != nullptr ||
                 cuda::moe::retainedQFNM1SuffixGateUpShard1(matrix.codebook_id) != nullptr);
        };
        if (!exportDeviceMoEExpertWeightDescriptors(store.gemmKernel(*shared_.prepared_ref_gate),
                store.gemmKernel(*shared_.prepared_ref_up), store.gemmKernel(*shared_.prepared_ref_down),
                shared_.d_model, shared_.intermediate, descriptor) ||
            descriptor.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            !valid_native(descriptor.gate) || !valid_native(descriptor.up) || !valid_native(descriptor.down))
            throw std::invalid_argument("CUDA retained QFN requires a valid NativeQuantised shared-expert triple");
        return descriptor;
    }

    /** @brief Admit all 18 compiled codebooks + dynamic promotion for TensorSharded overlays. */
    void resolveCodebooksFromIntermediateShard()
    {
        const auto gate = binding_->descriptor_declaration.nativeGateUpEnvelope();
        const auto down = binding_->descriptor_declaration.nativeDownEnvelope();
        const auto& sources = binding_->descriptor_declaration.sourceFormats();
        if (!sources[0].isNativeQuantised() || !sources[1].isNativeQuantised() || !sources[2].isNativeQuantised() ||
            !sources[0].native_quantised.present || !sources[1].native_quantised.present || !sources[2].native_quantised.present ||
            sources[0].native_quantised.codebook_id != sources[1].native_quantised.codebook_id)
            throw std::invalid_argument("CUDA retained QFN requires matching NativeQuantised Gate/Up/Down sources");

        const uint8_t g = sources[0].native_quantised.codebook_id;
        const uint8_t d = sources[2].native_quantised.codebook_id;
        const uint8_t g_phys = (g == native_quantised_formats::IQ4_XS.codebook_id && sources[0].native_quantised.is_superblock)
            ? kNativeQuantisedIQ4XSSourceScaleCodebook : g;
        const uint8_t d_phys = (d == native_quantised_formats::IQ4_XS.codebook_id && sources[2].native_quantised.is_superblock)
            ? kNativeQuantisedIQ4XSSourceScaleCodebook : d;
        const bool can_use_native = !router_.active_row_count_device &&
            cuda::moe::retainedQFNM1NativeCompletionSupported(g_phys, d_phys);
        const bool can_use_complete = cuda::moe::retainedQFNM1Function(
            g_phys, d_phys, output_demand_, router_.collect_device_runtime_histogram, routed_.row_policy) != nullptr;
        if (!can_use_native && !can_use_complete)
        {
            throw std::invalid_argument("CUDA retained QFN has no entry for layer " + std::to_string(routed_.layer) +
                " on " + device().toString() + " with gate/down codebooks " +
                std::to_string(g_phys) + "/" + std::to_string(d_phys));
        }

        routed_.gate_codebook = g_phys;
        routed_.down_codebook = d_phys;
        routed_.gate_source_superblock = sources[0].native_quantised.is_superblock ? 1u : 0u;
        routed_.down_source_superblock = sources[2].native_quantised.is_superblock ? 1u : 0u;
        const auto exp_bit = nativeQuantisedCodebookMaskBit(kNativeQuantisedExpandedInt8MinCodebook);
        if ((gate.execution_codebook_mask & exp_bit) != 0u)
            routed_.gate_promoted_codebook = kNativeQuantisedExpandedInt8MinCodebook;
        if ((down.execution_codebook_mask & exp_bit) != 0u)
            routed_.down_promoted_codebook = kNativeQuantisedExpandedInt8MinCodebook;
    }

    /** @brief Resolve canonical and dynamic-promoted codebooks from Apportioned prepared experts. */
    void resolveCodebooksFromPreparedEngines(const MoEExpertComputeStage::Params& compute)
    {
        bool found = false;
        uint8_t g_cb = 0, d_cb = 0;
        uint8_t g_sb = 0, d_sb = 0;
        for (std::size_t e = 0; e < compute.prepared_gate_gemm.size(); ++e)
        {
            if (!compute.prepared_gate_gemm[e] || !compute.prepared_up_gemm[e] || !compute.prepared_down_gemm[e])
                continue;
            DeviceMoEExpertDescriptor desc{};
            if (!exportDeviceMoEExpertWeightDescriptors(
                    compute.prepared_gate_gemm[e], compute.prepared_up_gemm[e], compute.prepared_down_gemm[e],
                    compute.d_model, compute.expert_intermediate, desc) ||
                desc.weight_format != DeviceMoEWeightFormat::NativeQuantised)
                throw std::invalid_argument("CUDA retained QFN Apportioned requires NativeQuantised prepared experts");
            if (!found)
            {
                g_cb = desc.gate.codebook_id;
                d_cb = desc.down.codebook_id;
                g_sb = desc.gate.source_is_superblock;
                d_sb = desc.down.source_is_superblock;
                found = true;
            }
            else if (!areNativeQuantisedGateUpCodebooksCompatible(desc.gate.codebook_id, g_cb) ||
                     !areNativeQuantisedGateUpCodebooksCompatible(desc.up.codebook_id, g_cb) ||
                     !areNativeQuantisedGateUpCodebooksCompatible(desc.down.codebook_id, d_cb))
            {
                throw std::invalid_argument("CUDA retained QFN Apportioned layer mixes incompatible routed codebooks");
            }
        }
        if (!found || !cuda::moe::retainedQFNM1NativeCompletionSupported(g_cb, d_cb))
            throw std::invalid_argument("CUDA retained QFN Apportioned lacks supported prepared routed experts");
        routed_.gate_codebook = g_cb;
        routed_.down_codebook = d_cb;
        routed_.gate_source_superblock = g_sb;
        routed_.down_source_superblock = d_sb;
    }

    /** @brief Reject any rebound prepared source, workspace, native tensor or stream. */
    void validateStorage(void* stream, bool validate_prefix = true) const
    {
        if (!workspace_ || !workspace_->isAllocated() || stream != prepared_stream_ ||
            workspace_->id() != prepared_workspace_id_ ||
            tensorAddresses() != prepared_tensors_ ||
            routed_.runtime->deviceLayerState(routed_.layer) != prepared_runtime_)
            throw std::logic_error("CUDA retained QFN capture storage changed");
        const bool store_unchanged = validated_store_epoch_ != 0 &&
            shared_.prepared_store->mutationEpoch() == validated_store_epoch_;
        if (!store_unchanged)
        {
            const auto descriptor = sharedDescriptor();
            const auto same = [](const auto& a, const auto& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; };
            if (bankAddresses() != prepared_banks_ ||
                !same(descriptor.gate, prepared_shared_.gate) || !same(descriptor.up, prepared_shared_.up) ||
                !same(descriptor.down, prepared_shared_.down))
                throw std::logic_error("CUDA retained QFN capture storage changed");
            validated_store_epoch_ = shared_.prepared_store->mutationEpoch();
        }
        if (validate_prefix)
            prefix_->validatePhaseBindings(stream);
    }

    const std::shared_ptr<Binding> binding_;
    CUDARetainedRoutedContract routed_{};
    const MoERoutingStage::Params router_;
    const SharedExpertFFNStage::Params shared_;
    const std::shared_ptr<Qwen4ExpHCReadPhaseOwner> prefix_;
    const CUDARetainedEarlyRouteReaders early_readers_;
    const std::string name_;
    const FullNativeHCFFNOutputDemand output_demand_ = FullNativeHCFFNOutputDemand::all();
    std::vector<Qwen4ExpFloatBinding> prefix_inputs_, prefix_outputs_;
    StageBufferContract prefix_contract_;
    DeviceWorkspaceManager* workspace_ = nullptr;
    std::optional<cuda::moe::RetainedM1Program> program_;
    std::array<void*, BankCount> prepared_banks_{};
    Addresses prepared_tensors_{};
    DeviceMoELayerRuntime* prepared_runtime_ = nullptr;
    DeviceMoEExpertDescriptor prepared_shared_{};
    std::uint64_t prepared_workspace_id_ = 0;
    mutable std::uint64_t validated_store_epoch_ = 0;
    void* prepared_stream_ = nullptr;
};
