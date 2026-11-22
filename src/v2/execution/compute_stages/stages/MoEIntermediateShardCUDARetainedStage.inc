/**
 * @file MoEIntermediateShardCUDARetainedStage.inc
 * @brief Binds the CUDA cooperative QFN program to the original mixed decode graph.
 *
 * The HC owner retains its original inputs, weights and workspace. The live
 * routed weights belong to the acquired runtime bank. The stage publishes raw
 * route slots and the shared FFN output for the existing ticket and TP graph.
 */

/** @brief Preparation and publication of one participant's complete local CUDA region. */
class CUDARetainedStage final : public IComputeStage, public IWorkspaceConsumer
{
public:
    /**
     * @brief Authenticate the original graph rather than deriving a second FFN placement.
     * @param binding Original routed stage's placement and descriptor declaration.
     * @param router Original router, authenticated by the route-slot builder.
     * @param shared Original prepared shared projections and Down publication.
     * @param prefix Original staged post-attention HC owner.
     * @param packet Original explicit CPU packet stages' optional early native edge.
     * @param name Existing terminal of the local routed math.
     */
    CUDARetainedStage(std::shared_ptr<Binding> binding, const MoERoutingStage::Params& router,
        const SharedExpertFFNStage::Params& shared, std::shared_ptr<Qwen4ExpHCReadPhaseOwner> prefix,
        std::shared_ptr<CUDAEarlyPacketPublication> packet, std::string name)
        : IComputeStage(binding->device()), binding_(std::move(binding)), router_(router), shared_(shared),
          prefix_(std::move(prefix)), packet_(std::move(packet)), name_(std::move(name))
    {
        const auto& p = binding_->params;
        const auto& original = binding_->original;
        if (!device().is_cuda() || binding_->local_result != LocalResult::RouteSlots ||
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
            throw std::invalid_argument("CUDA retained QFN requires original native mapped M1 graph operands");
        resolveCodebooks();
        (void)sharedDescriptor();
        prefix_->sealFullOutputDemand(output_demand_);
        prefix_inputs_ = prefix_->phaseInputs();
        prefix_outputs_ = prefix_->phaseOutputs();
        prefix_contract_ = prefix_->phaseDeclaration().bufferContract();
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
        requirements.addOutput("local_routed_slots", binding_->params.route_partials.tensor->shape(), BufferTensorType::FP32);
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
        info.addOutput("local_routed_slots", binding_->params.route_partials.tensor,
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
        if (packet_)
            MoEWorkspaceBuffers::add(requirements, packet_word_name, sizeof(std::uint64_t));
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
            binding_->params.runtime->prepareDecodeHistogramProducerStream(stream);
        prefix_->preparePhaseBindings(ctx, stream);
        const auto& p = binding_->params;
        const auto pointers = bankAddresses();
        const auto addresses = tensorAddresses();
        const auto prepared_shared = sharedDescriptor();
        auto* runtime = p.runtime->deviceLayerState(binding_->layer);
        if (!runtime)
            throw std::logic_error("CUDA retained QFN lost its original device runtime table");
        auto* codes = static_cast<int8_t*>(pointers[DownCodes]);
        auto* scales = static_cast<float*>(pointers[DownScales]);
        const int width = binding_->original.localIntermediateColumns();
        const int shared_offset = width * p.top_k;
        const cuda::moe::RetainedM1HCFFN arguments{
            .hc = prefix_->cudaPrefixArguments(),
            .router = {
                .runtime = runtime, .weights = static_cast<const uint16_t*>(addresses[0]),
                .logits = static_cast<float*>(pointers[RouterLogits]),
                .route_indices = static_cast<float*>(addresses[1]), .route_weights = static_cast<float*>(addresses[2]),
                .position = router_.absolute_position_ids_device, .experts = router_.num_experts, .top_k = router_.top_k,
                .local_policy = p.row_policy, .normalize = router_.norm_topk_prob,
            },
            .shared_gate_input = nullptr,
            .shared = {prepared_shared.gate, prepared_shared.up, prepared_shared.down},
            .selected = static_cast<cuda::moe::RetainedM1Expert*>(pointers[Descriptors]),
            .route_order = static_cast<int*>(pointers[RouteOrder]), .routed_width = width,
            .routed_workspace = {codes, scales},
            .shared_workspace = {codes + shared_offset, scales + shared_offset / 32},
            .hidden_codes = static_cast<int8_t*>(pointers[HiddenCodes]),
            .hidden_scales = static_cast<float*>(pointers[HiddenScales]),
            .shared_gate_factor = nullptr,
            .routed_contributions = static_cast<float*>(addresses[3]),
            .shared_contribution = static_cast<float*>(addresses[4]),
            .route_publication = cuda::moe::RetainedM1RoutePublication::Raw,
            .shared_publication = cuda::moe::RetainedM1SharedPublication::PartialBeforeCollective,
            .early_packet_ready = earlyPacketWord(),
        };
        auto program = cuda::moe::RetainedM1Program::prepare(device(), arguments, gate_, down_,
            output_demand_, router_.collect_device_runtime_histogram, workspace_->id());
        if (program.admission().resources().private_bytes)
            throw std::logic_error("CUDA retained QFN requires a spill-free resident entry");
        if (packet_)
            packet_->prepare(stream, arguments.early_packet_ready);
        prepared_stream_ = stream;
        prepared_workspace_id_ = workspace_->id();
        prepared_banks_ = pointers;
        prepared_tensors_ = addresses;
        prepared_runtime_ = runtime;
        prepared_shared_ = prepared_shared;
        prepared_packet_word_ = arguments.early_packet_ready;
        program_.emplace(std::move(program));
        setGPUStream(stream);
        const auto& native = program_->admission();
        PerfStatsCollector::addCounter("moe_resident_ffn", "cuda_mapped_prepared_program", 1., "setup",
            device().toString(), {{"layer", std::to_string(binding_->layer)},
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
        if (packet_)
            packet_->recordProducer(execution.nativeStream(), prepared_packet_word_);
        program_->launch(execution.nativeStream(), prepared_workspace_id_);
        for (const auto& output : prefix_outputs_)
            execution.publish(output.tensor);
        for (const auto& output : outputs())
            execution.publish(output.tensor);
        PerfStatsCollector::addCounter("moe_resident_ffn", "cuda_mapped_local_program_enqueued", 1.,
            "graph_capture", device().toString(), {{"layer", std::to_string(binding_->layer)},
                {"early_cpu_packet", packet_ ? "1" : "0"}});
        return true;
    }

    /** @brief Retire the enclosing program before releasing original HC leases. */
    void invalidateKernelDynamicState() override
    {
        program_.reset();
        prefix_->invalidatePhaseBindings();
    }

    /** @return Frozen source, geometry and policy identity, validated against physical capture leases. */
    std::uint64_t graphCaptureVariantSignature() const override
    {
        if (program_)
            validateStorage(prepared_stream_);
        auto signature = binding_->signature;
        const auto mix = [&](std::uint64_t value) { signature = (signature ^ value) * kFNV1a64Prime; };
        mix(prefix_->phaseDeclaration().graphCaptureVariantSignature());
        mix(gate_); mix(down_); mix(output_demand_.bits());
        mix(router_.norm_topk_prob); mix(router_.collect_device_runtime_histogram);
        mix(reinterpret_cast<std::uintptr_t>(router_.gate_weights));
        mix(reinterpret_cast<std::uintptr_t>(router_.absolute_position_ids_device));
        mix(reinterpret_cast<std::uintptr_t>(packet_.get()));
        mix(reinterpret_cast<std::uintptr_t>(shared_.prepared_store));
        mix(shared_.prepared_ref_gate->binding_id); mix(shared_.prepared_ref_up->binding_id);
        mix(shared_.prepared_ref_down->binding_id); mix(shared_.intermediate);
        mix(reinterpret_cast<std::uintptr_t>(shared_.output)); mix(static_cast<unsigned>(shared_.output_buffer_id));
        return signature ? signature : 1;
    }

private:
    static constexpr const char* packet_word_name = "cuda_retained_qfn_cpu_packet_ready";

    /** @return The named arena word only for a participant with original early packet readers. */
    std::uint64_t* earlyPacketWord() const
    {
        if (!packet_)
            return nullptr;
        auto* word = static_cast<std::uint64_t*>(workspace_->getBuffer(packet_word_name));
        if (!word || workspace_->getBufferSize(packet_word_name) < sizeof(*word))
            throw std::logic_error("CUDA retained QFN has no admitted early packet signal word");
        return word;
    }

    /** @brief Unambiguous index of every admitted scratch plane in the native operands. */
    enum Bank : std::size_t { RouterLogits, Descriptors, RouteOrder, HiddenCodes, HiddenScales, DownCodes, DownScales, BankCount };
    /** @brief Size of one named workspace plane. */
    struct BankRequirement { const char* name; std::size_t bytes; };
    using Addresses = std::array<void*, 5>;

    /** @return Exact execution scratch by original source width and selected route capacity. */
    std::array<BankRequirement, BankCount> banks() const
    {
        using namespace MoEWorkspaceBuffers;
        const std::size_t top_k = router_.top_k;
        const std::size_t hidden = router_.d_model;
        const std::size_t channels = top_k * binding_->original.localIntermediateColumns() + shared_.intermediate;
        return {{{ROUTE_LOGITS, std::size_t(router_.num_experts) * sizeof(float)},
            {"cuda_retained_qfn_selected_descriptors", top_k * sizeof(cuda::moe::RetainedM1Expert)},
            {"cuda_retained_qfn_route_order", (2 + top_k) * sizeof(int)},
            {DECODE_HIDDEN_INT8, hidden * sizeof(int8_t)}, {DECODE_HIDDEN_SCALES, hidden / 32 * sizeof(float)},
            {DECODE_SWIGLU_INT8, channels * sizeof(int8_t)}, {DECODE_SWIGLU_SCALES, channels / 32 * sizeof(float)}}};
    }

    /** @return Existing graph publications with their original arena identities. */
    std::array<MoEIntermediateShardTensorBinding, 4> outputs() const
    {
        return {{{router_.output_indices, router_.output_indices_buffer_id},
            {router_.output_weights, router_.output_weights_buffer_id}, binding_->params.route_partials,
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
        const auto q8 = [](const DeviceNativeQuantisedMatrixDesc& matrix) {
            return matrix.codebook_id == native_quantised_formats::Q8_0.codebook_id &&
                matrix.source_identity_present == 1 && matrix.source_codebook_id == matrix.codebook_id &&
                matrix.source_is_superblock == 0;
        };
        if (!exportDeviceMoEExpertWeightDescriptors(store.gemmKernel(*shared_.prepared_ref_gate),
                store.gemmKernel(*shared_.prepared_ref_up), store.gemmKernel(*shared_.prepared_ref_down),
                shared_.d_model, shared_.intermediate, descriptor) ||
            descriptor.weight_format != DeviceMoEWeightFormat::NativeQuantised ||
            !q8(descriptor.gate) || !q8(descriptor.up) || !q8(descriptor.down))
            throw std::invalid_argument("CUDA retained QFN requires the original prepared Q8_0 shared-expert triple");
        return descriptor;
    }

    /** @brief Admit only compiled policy pairs covering every legal execution decoder, including movement. */
    void resolveCodebooks()
    {
        namespace formats = native_quantised_formats;
        const auto gate = binding_->descriptor_declaration.nativeGateUpEnvelope();
        const auto down = binding_->descriptor_declaration.nativeDownEnvelope();
        const auto& sources = binding_->descriptor_declaration.sourceFormats();
        const auto matches = [](const auto& source, uint8_t codebook, bool superblock) {
            return source.isNativeQuantised() && source.native_quantised.present &&
                source.native_quantised.codebook_id == codebook && source.native_quantised.is_superblock == superblock;
        };
        constexpr std::array gates{formats::Q4_K.codebook_id, formats::Q5_K.codebook_id};
        constexpr std::array downs{formats::Q5_1.codebook_id, formats::Q8_0.codebook_id};
        for (const uint8_t g : gates)
        {
            const auto gate_bit = nativeQuantisedCodebookMaskBit(g);
            const auto compiled_gate = gate_bit | (g == formats::Q5_K.codebook_id
                ? nativeQuantisedCodebookMaskBit(kNativeQuantisedExpandedInt8MinCodebook) : 0u);
            if (!matches(sources[0], g, true) || !matches(sources[1], g, true) ||
                gate.policy_codebook_mask != gate_bit || (gate.execution_codebook_mask & ~compiled_gate) != 0u)
                continue;
            for (const uint8_t d : downs)
            {
                const auto down_bit = nativeQuantisedCodebookMaskBit(d);
                const auto compiled = down_bit | (d == formats::Q5_1.codebook_id
                    ? nativeQuantisedCodebookMaskBit(kNativeQuantisedExpandedInt8MinCodebook) : 0u);
                if (matches(sources[2], d, false) && down.policy_codebook_mask == down_bit &&
                    (down.execution_codebook_mask & ~compiled) == 0u &&
                    cuda::moe::retainedQFNM1Function(g, d, output_demand_, router_.collect_device_runtime_histogram,
                        binding_->params.row_policy))
                {
                    gate_ = g;
                    down_ = d;
                    return;
                }
            }
        }
        throw std::invalid_argument("CUDA retained QFN has no entry for layer " + std::to_string(binding_->layer) +
            " on " + device().toString() + " with gate/up execution/policy masks " +
            std::to_string(gate.execution_codebook_mask) + "/" + std::to_string(gate.policy_codebook_mask) +
            " and down execution/policy masks " + std::to_string(down.execution_codebook_mask) +
            "/" + std::to_string(down.policy_codebook_mask));
    }

    /** @brief Reject any rebound prepared source, workspace, native tensor or stream. */
    void validateStorage(void* stream) const
    {
        const auto descriptor = sharedDescriptor();
        const auto same = [](const auto& a, const auto& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; };
        if (!workspace_ || stream != prepared_stream_ || workspace_->id() != prepared_workspace_id_ ||
            tensorAddresses() != prepared_tensors_ || bankAddresses() != prepared_banks_ ||
            earlyPacketWord() != prepared_packet_word_ ||
            binding_->params.runtime->deviceLayerState(binding_->layer) != prepared_runtime_ ||
            !same(descriptor.gate, prepared_shared_.gate) || !same(descriptor.up, prepared_shared_.up) ||
            !same(descriptor.down, prepared_shared_.down))
            throw std::logic_error("CUDA retained QFN capture storage changed");
        prefix_->validatePhaseBindings(stream);
    }

    const std::shared_ptr<Binding> binding_;
    const MoERoutingStage::Params router_;
    const SharedExpertFFNStage::Params shared_;
    const std::shared_ptr<Qwen4ExpHCReadPhaseOwner> prefix_;
    const std::shared_ptr<CUDAEarlyPacketPublication> packet_;
    const std::string name_;
    const FullNativeHCFFNOutputDemand output_demand_ = FullNativeHCFFNOutputDemand::all();
    std::vector<Qwen4ExpFloatBinding> prefix_inputs_, prefix_outputs_;
    StageBufferContract prefix_contract_;
    uint8_t gate_ = 0, down_ = 0;
    DeviceWorkspaceManager* workspace_ = nullptr;
    std::optional<cuda::moe::RetainedM1Program> program_;
    std::array<void*, BankCount> prepared_banks_{};
    Addresses prepared_tensors_{};
    DeviceMoELayerRuntime* prepared_runtime_ = nullptr;
    std::uint64_t* prepared_packet_word_ = nullptr;
    DeviceMoEExpertDescriptor prepared_shared_{};
    std::uint64_t prepared_workspace_id_ = 0;
    void* prepared_stream_ = nullptr;
};
