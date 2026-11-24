/**
 * @file MoEIntermediateShardGroupedFFNStage.inc
 * @brief Closed grouped IntermediateShard local region with original mutable plan authority.
 * Included next to the normal staged binding so only the local graph shape is
 * replaced. Device-owned routing, movement, acceptance, and rank sum stay on
 * the same original producers.
 */
namespace frontier = rocm::grouped_ffn;

/** @brief Freeze the measured implementation and its original route/bank producer. */
struct GroupedFFNFrontierSelection
{
    frontier::Program program; ///< Exact physical numerical and workspace program.
    bool direct_bank; ///< Borrow the router packet and its request-selected bank directly.
};

/**
 * @brief Choose the closed counted verifier program from already admitted plan and source.
 * @param binding Original participant owner, complete row count and source declaration.
 * @param shared Exact shared-stage source and scratch declaration.
 * @param source Original prepared shared gate, up and down descriptors.
 * @param original_fp16_router Whether the finalizer authenticated the original counted FP16 router.
 * @return Local program and bank authority, or no replacement for the declared staged program.
 */
std::optional<GroupedFFNFrontierSelection> groupedFFNFrontierProgram(const Binding& binding,
    const SharedExpertFFNStage::Params& shared,const DeviceMoEExpertDescriptor& source,bool original_fp16_router)
{
    const auto& p=binding.params;
    if(!binding.device().is_rocm() || p.phase!=MoEOverlayServicePhaseHint::GroupedVerifier ||
       p.row_mode!=MoEIntermediateShardRowMode::Grouped || !frontier::supportsFormatRows(p.rows) ||
       p.top_k!=frontier::topK || !p.live_rows || !p.live_rows->countOwner() ||
       binding.route_authority==RouteAuthority::PreparedService ||
       binding.original.modelColumns()!=frontier::hidden ||
       binding.descriptors.size()!=frontier::experts ||
       binding.format!=DeviceMoEWeightFormat::NativeQuantised ||
       source.weight_format!=DeviceMoEWeightFormat::NativeQuantised ||
       !shared.gate_scratch || !shared.up_scratch || !shared.required_router_q8_publication)
        return std::nullopt;
    const bool formats=qfnGroupedFFNAllFormatsOptIn();
    if(p.rows!=frontier::capacity && !formats)return std::nullopt;
    const int columns=binding.original.localIntermediateColumns();
    const bool equal=columns==frontier::routedColumns(frontier::RoutedGeometry::F160);
    const auto count=p.live_rows->captureIdentity();
    const bool measured_default=p.verifier_kernel_policy==LocalGPUExpertKernelPolicy::QFNMI50TP4 &&
        equal && shared.intermediate==frontier::sharedWidth && p.rows==frontier::capacity &&
        count.count_capacity==p.rows && count.first_physical_row==0 && count.repeats_per_row==1;
    if(!qfnGroupedFFNFrontierOptIn(measured_default))return std::nullopt;
    const bool direct=qfnGroupedFFNDirectBankExperiment(measured_default);
    const auto& declaration=binding.descriptor_declaration;
    // An explicit all-format diagnostic keeps its portable source-aware program.
    // The automatic F160 M4 lane uses the independently timed fixed schedules,
    // so it can qualify the exact affine-sum producer/consumer jointly.
    if(formats) {
        if(!direct)
            throw std::invalid_argument("All-format grouped FFN requires the counted direct-bank router");
        const auto& original=declaration.sourceFormats();
        return GroupedFFNFrontierSelection{frontier::SourceFormatProgram::prepare(
            frontier::declareSourceFormats(columns,shared.intermediate,
                declaration.nativeGateUpEnvelope(),declaration.nativeDownEnvelope(),
                {original[0].native_quantised,original[1].native_quantised,original[2].native_quantised},
                {source.gate,source.up,source.down})),true};
    }
    const auto hardware=rocm::nativeQuantisedHardwareForDevice(binding.device().ordinal);
    if(native_quantised::selectVerifierDispatchTable(hardware)!=native_quantised::VerifierDispatchTable::MI50)
        return std::nullopt;
    const auto exactShared=[](const DeviceNativeQuantisedMatrixDesc& m,int n,int k) {
        return m.valid() && m.n==n && m.k==k && m.codebook_id==19 &&
            m.source_identity_present && m.source_codebook_id==19;
    };
    if((columns!=frontier::routedWidth && !(equal && direct && qfnGroupedFFNEqualFrontierOptIn(measured_default))) ||
       shared.intermediate!=frontier::sharedWidth ||
       !exactShared(source.gate,frontier::sharedWidth,frontier::hidden) ||
       !exactShared(source.up,frontier::sharedWidth,frontier::hidden) ||
       !exactShared(source.down,frontier::hidden,frontier::sharedWidth))return std::nullopt;
    if(const auto decoder=frontier::measuredDecoder(columns,declaration.nativeGateUpEnvelope(),
            declaration.nativeDownEnvelope(),declaration.sourceFormats())) {
        const auto geometry=static_cast<frontier::RoutedGeometry>(columns);
        const bool sums=direct && frontier::exactK32SumsOptIn(measured_default && original_fp16_router) &&
            frontier::hasMeasuredSumLayout(geometry,*decoder,declaration.sourceFormats(),
                {source.gate,source.up,source.down});
        return GroupedFFNFrontierSelection{frontier::MeasuredProgram{geometry,*decoder,
            sums?frontier::AffineSumSource::RouterPublication:frontier::AffineSumSource::Inline},direct};
    }
    return std::nullopt;
}

/** @brief Replace only the participant-local routed/shared contraction, excluding router and TP. */
class GroupedFFNFrontierStage final : public IComputeStage, public IWorkspaceConsumer
{
    std::shared_ptr<Binding> binding_;
    SharedExpertFFNStage::Params shared_;
    frontier::LocalPublication publication_;
    const bool direct_bank_;
    BufferId combined_id_;
    std::string name_;
    DeviceWorkspaceManager* workspace_=nullptr;
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>,6> operands_;
    struct ScratchRequirement
    {
        const char* name;
        std::size_t minimum_bytes;
    };
    /** @return Exact live shared Q8 extent or the explicitly selected sum layout in the same input bank. */
    std::array<ScratchRequirement,4> scratchRequirements()const
    {
        const auto rows=std::size_t(publication_.rows.physicalRows());
        const auto codes=rows*frontier::sharedColumns(publication_.program);
        const bool sums=frontier::usesExactK32Sums(publication_.program);
        return {{{MoEWorkspaceBuffers::PREFILL_A_INT8,sums?frontier::k32SumWorkspaceBytes:codes},
            {MoEWorkspaceBuffers::PREFILL_A_SCALES,codes/32*sizeof(float)},
            {MoEWorkspaceBuffers::ROCM_SHARED_GATE,rows*sizeof(float)},
            {MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY,frontier::bankIdentityBytes}}};
    }
    std::array<std::shared_ptr<const WorkspaceBufferLease>,4> scratch_leases_;
    IBackend* backend_=nullptr;
    std::uint64_t runtime_generation_=0;

    /** @return Original visible local accumulators and the exact shared scratch debug roles. */
    std::array<MoEIntermediateShardTensorBinding,6> outputs()const
    {
        return {binding_->params.route_partials,binding_->params.output,
            MoEIntermediateShardTensorBinding{shared_.output,shared_.output_buffer_id},
            MoEIntermediateShardTensorBinding{shared_.gate_scratch,shared_.gate_scratch_buffer_id},
            MoEIntermediateShardTensorBinding{shared_.up_scratch,shared_.up_scratch_buffer_id},
            MoEIntermediateShardTensorBinding{publication_.combined,combined_id_}};
    }

    /** @return Physical tensor roots of former shared/gate stages removed by this contraction. */
    std::array<TensorBase*,6> extraOperands()const
    {
        return {publication_.sharedInputGate,publication_.sharedGateScratch,publication_.sharedUpScratch,
            publication_.routedOutput,publication_.sharedOutput,publication_.combined};
    }

    /** @brief Validate cold workspace storage and backend lifetime without observing request state. */
    void validateSeal()const
    {
        binding_->validatePreparedStorage();
        if(!scratch_leases_[0])return;
        if(!workspace_ || !binding_->prepared || workspace_!=binding_->prepared->workspace ||
           workspace_->id()!=binding_->prepared->workspace_id ||
           getBackendFor(device())!=backend_ || backend_->deviceRuntimeGeneration(device().ordinal)!=runtime_generation_)
            throw std::logic_error("Grouped FFN frontier changed its captured workspace or backend generation");
        const auto scratch_requirements=scratchRequirements();
        for(std::size_t i=0;i<scratch_requirements.size();++i) {
            const auto* name=scratch_requirements[i].name;
            if(workspace_->getBuffer(name)!=scratch_leases_[i]->data() ||
               workspace_->getBufferSize(name)!=scratch_leases_[i]->sizeBytes())
                throw std::logic_error("Grouped FFN frontier changed its original existing scratch banks");
        }
        const auto tensors=extraOperands();
        for(std::size_t i=0;i<tensors.size();++i)
            TransferEngine{}.requireCapturedDeviceStorage(*operands_[i],*tensors[i],device(),tensors[i]->size_bytes());
    }

public:
    /** @brief Capture the original exact shared bindings and the original local plan owner. */
    GroupedFFNFrontierStage(std::shared_ptr<Binding> binding,const SharedExpertFFNStage::Params& shared,
        const DeviceMoEExpertDescriptor& descriptor,GroupedFFNFrontierSelection selection,TensorBase* gate,
        MoEIntermediateShardTensorBinding combined,std::string name)
        : IComputeStage(binding->device()),binding_(std::move(binding)),shared_(shared),
        publication_{*binding_->params.live_rows,{descriptor.gate,descriptor.up,descriptor.down},gate,
            shared.gate_scratch,shared.up_scratch,dynamic_cast<TensorBase*>(binding_->params.output.tensor),
            shared.output,dynamic_cast<TensorBase*>(combined.tensor),{},std::move(selection.program),{},
            binding_->params.swiglu_activation,shared.swiglu_activation},
            direct_bank_(selection.direct_bank),combined_id_(combined.id),name_(std::move(name))
    {
        for(auto* tensor:extraOperands())
            if(!tensor || tensor->native_type()!=TensorType::FP32)
                throw std::invalid_argument("Grouped FFN frontier requires original FP32 source and arena outputs");
    }

    /** @return Original local grouped FFN schedule role. */
    ComputeStageType type()const override {return ComputeStageType::MOE_PROJECTION_PHASE;}
    /** @return Preserved public graph sink. */
    std::string name()const override {return name_;}
    /** @return The original qualified ROCm participant. */
    bool supportsBackend(ComputeBackendType backend)const override {return backend==ComputeBackendType::GPU_ROCM;}
    /** @return The stage may submit only after its real workspace and capture generation are sealed. */
    bool isGraphCapturable()const override {return workspace_ && scratch_leases_[0];}
    /** @return Cold preparation uses no eager execution. */
    bool supportsGraphCaptureAfterLaunchPreparation()const override {return true;}
    /** @return An original device count independently owns each retained live prefix. */
    bool supportsPaddedPrefillGraphCapturePreflight()const override {return true;}
    /** @return Cold native graph setup owns the workspace. */
    bool supportsLazyPrefillGraphCapturePreflight()const override {return true;}
    /** @return No setup operation may first bind GPU storage during capture. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override
    {return GraphLaunchPreparationPolicy::CaptureOnly;}

    /** @return The unchanged original grouped quote, including the router-disjoint reusable scratch. */
    WorkspaceRequirements getWorkspaceRequirements(int m,int=0,int=0)const override
    {
        auto requirements=MoEWorkspaceBuffers::rocmMoE(std::max(m,publication_.rows.physicalRows()),frontier::hidden,
            frontier::routedColumns(publication_.program),frontier::experts,frontier::topK);
        // The routed and shared branches reuse the same named banks at disjoint times.
        WorkspaceRequirements shared;
        for(const auto& scratch:scratchRequirements())
            MoEWorkspaceBuffers::add(shared,scratch.name,scratch.minimum_bytes);
        requirements.merge(shared);
        return requirements;
    }
    /** @brief Borrow the executor's exact admitted workspace. */
    void bindWorkspace(DeviceWorkspaceManager* workspace) override
    {
        if(scratch_leases_[0] && workspace_!=workspace)
            throw std::logic_error("Grouped FFN frontier must retire its graph before rebinding the workspace");
        workspace_=workspace;
    }
    /** @brief Retire the borrowed workspace only through native graph invalidation. */
    void unbindWorkspace() override {bindWorkspace(nullptr);}
    /** @return Whether the real workspace owner has been supplied. */
    bool hasWorkspace()const override {return workspace_!=nullptr;}
    /** @return Exact admitted workspace. */
    DeviceWorkspaceManager* getWorkspace()const override {return workspace_;}
    /** @return The original sorted runtime publication and every participant-local output. */
    StageBufferContract bufferContract()const override
    {
        auto contract=StageBufferContract::build();
        for(const auto& input:{binding_->params.hidden,binding_->params.routing_indices,binding_->params.routing_weights})
            contract.addInput(input.id);
        for(const auto& output:outputs())contract.addOutput(output.id);
        contract.addWeight(publication_.sharedInputGate);
        contract.addPreparedWeight(shared_.gate_w,shared_.prepared_store,*shared_.prepared_ref_gate);
        contract.addPreparedWeight(shared_.up_w,shared_.prepared_store,*shared_.prepared_ref_up);
        contract.addPreparedWeight(shared_.down_w,shared_.prepared_store,*shared_.prepared_ref_down);
        return contract;
    }
    /** @return Complete original external input and output requests. */
    StageBufferRequirements getBufferRequirements()const override
    {
        StageBufferRequirements out;
        for(const auto& input:{binding_->params.hidden,binding_->params.routing_indices,binding_->params.routing_weights})
            out.addInput(std::string(bufferIdName(input.id)),input.tensor->shape(),toBufferTensorType(input.tensor->native_type()));
        for(const auto& output:outputs())
            out.addOutput(std::string(bufferIdName(output.id)),output.tensor->shape(),toBufferTensorType(output.tensor->native_type()));
        return out;
    }
    /** @return Capture key includes both fixed shared source and the original movable/count-binding identity. */
    std::uint64_t graphCaptureVariantSignature()const override
    {
        validateSeal();
        auto value=binding_->signature^0x46464e47524f5550ULL;
        const auto mix=[&](std::uint64_t part){value=(value^part)*kFNV1a64Prime;};
        mix(publication_.sharedActivation.captureWord());
        mix(direct_bank_);
        mix(frontier::usesExactK32Sums(publication_.program));
        mix(publication_.program.index());
        mix(publication_.rows.physicalRows());
        mix(frontier::routedColumns(publication_.program));
        mix(frontier::sharedColumns(publication_.program));
        const auto mixFormat=[&](const NativeQuantisedExecutionFormatEnvelope& format) {
            mix(format.execution_codebook_mask);
            mix(format.policy_codebook_mask);
        };
        for(const auto& format:frontier::routedFormats(publication_.program))mixFormat(format);
        if(const auto* program=std::get_if<frontier::SourceFormatProgram>(&publication_.program)) {
            mix(program->family());
            for(const int parts:program->routedPartitions())mix(parts);
            for(const int parts:program->sharedPartitions())mix(parts);
            // A union loses role order and block-family identity. Both are captured here.
            for(const auto& source:program->routedSources()) {
                mix(source.codebook_id);
                mix(source.is_superblock);
            }
            for(const auto& format:program->formats().shared)mixFormat(format);
        }
        for(auto* tensor:extraOperands())mix(reinterpret_cast<std::uintptr_t>(tensor));
        for(const auto& matrix:publication_.sharedWeights) {
            mix(reinterpret_cast<std::uintptr_t>(matrix.payload));
            mix(reinterpret_cast<std::uintptr_t>(matrix.scales));
            mix(reinterpret_cast<std::uintptr_t>(matrix.mins));
            mix(reinterpret_cast<std::uintptr_t>(matrix.emins));
            mix(matrix.n);mix(matrix.k);mix(matrix.blocks_per_row);mix(matrix.codebook_id);
            mix(matrix.allocation_payload_bytes_per_block);mix(matrix.allocation_has_mins);mix(matrix.allocation_has_emins);
            mix(matrix.source_codebook_id);mix(matrix.source_is_superblock);mix(matrix.source_identity_present);
        }
        for(const auto& output:outputs())mix(static_cast<std::uint64_t>(output.id));
        return value?value:1;
    }

    /** @brief Materialize original descriptor owners and retain, rather than allocate, physical operands. */
    bool prepareGraphLaunch(IDeviceContext* ctx,void* stream)override
    {
        if(!ctx || ctx->deviceId()!=device() || !workspace_ || !stream || isGraphCaptureActive())return false;
        if(const auto* program=std::get_if<frontier::SourceFormatProgram>(&publication_.program);
           program && !frontier::prepareFormatTables(program->formats(),device().ordinal))
            throw std::logic_error("Grouped FFN could not prepare the original IQ grid tables");
        binding_->prepare(workspace_,stream);
        setGPUStream(stream);
        if(scratch_leases_[0]){validateSeal();return true;}
        backend_=getBackendFor(device());
        runtime_generation_=backend_?backend_->deviceRuntimeGeneration(device().ordinal):0;
        if(!runtime_generation_)
            throw std::logic_error("Grouped FFN frontier has no admitted backend generation");
        if(binding_->params.demand==MoEGroupedPlanDemand::DeferredAcceptedRows) {
            const auto& runtime=binding_->params.runtime->hostLayerState(binding_->layer);
            const auto slots=std::uint32_t(publication_.rows.physicalRows()*frontier::topK);
            if(runtime.deferred_verifier_route_capacity<slots ||
               !runtime.deferred_verifier_route_expert_ids || !runtime.deferred_verifier_route_participant_ids)
                throw std::logic_error("Grouped FFN frontier requires the original full-capacity deferred route planes");
        }
        if(direct_bank_)
            publication_.boundSourceProgram=frontier::BoundSourceFormatProgram::prepare(
                *binding_->params.runtime,binding_->layer,publication_.program,publication_.sharedWeights);
        const auto tensors=extraOperands();
        for(std::size_t i=0;i<tensors.size();++i)
            operands_[i]=TransferEngine{}.retainCapturedDeviceStorage(*tensors[i],device(),tensors[i]->size_bytes());
        const auto scratch_requirements=scratchRequirements();
        for(std::size_t i=0;i<scratch_requirements.size();++i) {
            const auto& required=scratch_requirements[i];
            const auto extent=workspace_->getBufferSize(required.name);
            const auto minimum=required.minimum_bytes;
            if(extent<minimum)
                throw std::logic_error("Grouped FFN frontier scratch is undersized: "+
                    std::string(required.name)+" expected "+std::to_string(minimum)+
                    " bytes, got "+std::to_string(extent));
            scratch_leases_[i]=workspace_->retainBuffer(required.name,0,extent);
        }
        validateSeal();
        return true;
    }

    /** @brief Retire captured runtime state before allowing any new recording. */
    void invalidateKernelDynamicState()override
    {
        scratch_leases_={};
        operands_={};
        publication_.boundSourceProgram.reset();
        binding_->invalidate();
    }

    /** @brief Record original plan publication and both exact measured consumers. */
    bool execute(IDeviceContext* ctx)override
    {
        if(!ctx || ctx->deviceId()!=device() || !scratch_leases_[0])return false;
        validateSeal();
        const auto execution=gpuExecution();
        for(const auto& input:{binding_->params.hidden,binding_->params.routing_indices,binding_->params.routing_weights})
            execution.requirePreparedInput(input.tensor);
        execution.requirePreparedInput(publication_.sharedInputGate);
        for(const auto& output:outputs())execution.requirePreparedOutput(output.tensor);
        if(!binding_->executeFrontier(publication_,direct_bank_,execution.nativeStream()))return false;
        for(const auto& output:outputs())execution.publish(output.tensor);
        return true;
    }

    /** @return The original live input bindings are prepared; this stage publishes the output roles. */
    CoherencePolicy coherencePolicy()const override {return CoherencePolicy::OUTPUT;}

    /** @return Exact original local publications for optional diagnostic observation. */
    StageDumpInfo buildDumpInfoImpl()const override
    {
        StageDumpInfo result;
        constexpr std::array names{"local_route_partials","local_routed_partial","shared_output",
            "shared_gate_scratch","shared_up_scratch","local_combined"};
        const auto tensors=outputs();
        for(std::size_t i=0;i<tensors.size();++i) {
            auto* tensor=tensors[i].tensor;
            result.addOutput(names[i],tensor,tensor->shape().front(),tensor->shape().back());
        }
        return result;
    }
};
