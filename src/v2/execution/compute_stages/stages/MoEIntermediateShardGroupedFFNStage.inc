/**
 * @file MoEIntermediateShardGroupedFFNStage.inc
 * @brief Closed grouped IntermediateShard local region with original mutable plan authority.
 * Included next to the normal staged binding so only the local graph shape is
 * replaced. Device-owned routing, movement, acceptance, and rank sum stay on
 * the same original producers.
 */
namespace frontier = rocm::grouped_ffn;

/** @brief Admit the original width or the independently selected F160 program. */
bool groupedFFNFrontierWidthEligible(int columns)
{
    return columns==frontier::routedWidth ||
        (columns==frontier::routedColumns(frontier::RoutedGeometry::F160) &&
         qfnGroupedFFNEqualFrontierOptIn() && qfnGroupedFFNDirectBankExperiment());
}

/** @brief Select an exact candidate without weakening the original narrow binding. */
std::optional<frontier::RoutedDecoder> groupedFFNFrontierDecoder(const Binding& binding)
{
    const auto gate=binding.descriptor_declaration.nativeGateUpEnvelope();
    const auto down=binding.descriptor_declaration.nativeDownEnvelope();
    const auto matches=[&](unsigned g,unsigned d) {
        return gate.execution_codebook_mask==(1u<<g) && gate.policy_codebook_mask==(1u<<g) &&
            down.execution_codebook_mask==(1u<<d) && down.policy_codebook_mask==(1u<<d);
    };
    using Program=frontier::RoutedDecoder;
    if(matches(5,7))return Program::Q4K_Q51;
    if(binding.original.localIntermediateColumns()==160) {
        if(matches(5,19))return Program::Q4K_Q80;
        if(matches(7,19))return Program::Q5K_Q80;
    }
    return std::nullopt;
}

/** @brief Select exactly the measured MI50 M4 source tuple. */
bool groupedFFNFrontierEligible(const Binding& binding,const SharedExpertFFNStage::Params& shared,
    const DeviceMoEExpertDescriptor& source)
{
    const auto& p=binding.params;
    if(!binding.device().is_rocm() || p.phase!=MoEOverlayServicePhaseHint::GroupedVerifier ||
       p.row_mode!=MoEIntermediateShardRowMode::Grouped || p.rows!=frontier::capacity ||
       p.top_k!=frontier::topK || !p.live_rows || !p.live_rows->countOwner() ||
       binding.route_authority==RouteAuthority::PreparedService ||
       binding.original.modelColumns()!=frontier::hidden ||
       !groupedFFNFrontierWidthEligible(binding.original.localIntermediateColumns()) ||
       binding.descriptors.size()!=frontier::experts ||
       binding.format!=DeviceMoEWeightFormat::NativeQuantised ||
       source.weight_format!=DeviceMoEWeightFormat::NativeQuantised ||
       shared.intermediate!=frontier::sharedWidth || !shared.gate_scratch || !shared.up_scratch ||
       !shared.required_router_q8_publication)return false;
    const auto& hardware=rocmHardwareMetadata().at(binding.device().ordinal);
    if(hardware.compute_units!=60 || hardware.native_wave_width!=64 ||
       (hardware.isa_target!="gfx906" && !hardware.isa_target.starts_with("gfx906:")))return false;
    const auto exactShared=[](const DeviceNativeQuantisedMatrixDesc& m,int n,int k) {
        return m.valid() && m.n==n && m.k==k && m.codebook_id==19 &&
            m.source_identity_present && m.source_codebook_id==19;
    };
    return groupedFFNFrontierDecoder(binding).has_value() &&
        exactShared(source.gate,frontier::sharedWidth,frontier::hidden) &&
        exactShared(source.up,frontier::sharedWidth,frontier::hidden) &&
        exactShared(source.down,frontier::hidden,frontier::sharedWidth);
}

/** @brief Replace only the participant-local routed/shared contraction, excluding router and TP. */
class GroupedFFNFrontierStage final : public IComputeStage, public IWorkspaceConsumer
{
    std::shared_ptr<Binding> binding_;
    SharedExpertFFNStage::Params shared_;
    frontier::LocalPublication publication_;
    BufferId combined_id_;
    std::string name_;
    DeviceWorkspaceManager* workspace_=nullptr;
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>,6> operands_;
    struct ScratchRequirement
    {
        const char* name;
        std::size_t minimum_bytes;
    };
    static constexpr std::array scratch_requirements{
        ScratchRequirement{MoEWorkspaceBuffers::PREFILL_A_INT8,frontier::sharedCodesBytes},
        ScratchRequirement{MoEWorkspaceBuffers::PREFILL_A_SCALES,frontier::sharedScalesBytes},
        ScratchRequirement{MoEWorkspaceBuffers::ROCM_SHARED_GATE,frontier::sharedGateBytes},
        ScratchRequirement{MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY,frontier::bankIdentityBytes}};
    std::array<std::shared_ptr<const WorkspaceBufferLease>,scratch_requirements.size()> scratch_leases_;
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
        const DeviceMoEExpertDescriptor& descriptor,TensorBase* gate,
        MoEIntermediateShardTensorBinding combined,std::string name)
        : IComputeStage(binding->device()),binding_(std::move(binding)),shared_(shared),
        publication_{*binding_->params.live_rows,{descriptor.gate,descriptor.up,descriptor.down},gate,
            shared.gate_scratch,shared.up_scratch,dynamic_cast<TensorBase*>(binding_->params.output.tensor),
            shared.output,dynamic_cast<TensorBase*>(combined.tensor),{},
            static_cast<frontier::RoutedGeometry>(binding_->original.localIntermediateColumns()),
            groupedFFNFrontierDecoder(*binding_).value()},combined_id_(combined.id),name_(std::move(name))
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
        return MoEWorkspaceBuffers::rocmMoE(std::max(m,frontier::capacity),frontier::hidden,
            frontier::routedColumns(publication_.routedGeometry),frontier::experts,frontier::topK);
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
        mix(qfnGroupedFFNDirectBankExperiment());
        mix(frontier::routedColumns(publication_.routedGeometry));
        mix(static_cast<unsigned>(publication_.routedDecoder));
        for(auto* tensor:extraOperands())mix(reinterpret_cast<std::uintptr_t>(tensor));
        for(const auto& matrix:publication_.sharedWeights) {
            mix(reinterpret_cast<std::uintptr_t>(matrix.payload));
            mix(reinterpret_cast<std::uintptr_t>(matrix.scales));
        }
        for(const auto& output:outputs())mix(static_cast<std::uint64_t>(output.id));
        return value?value:1;
    }

    /** @brief Materialize original descriptor owners and retain, rather than allocate, physical operands. */
    bool prepareGraphLaunch(IDeviceContext* ctx,void* stream)override
    {
        if(!ctx || ctx->deviceId()!=device() || !workspace_ || !stream || isGraphCaptureActive())return false;
        binding_->prepare(workspace_,stream);
        setGPUStream(stream);
        if(scratch_leases_[0]){validateSeal();return true;}
        backend_=getBackendFor(device());
        runtime_generation_=backend_?backend_->deviceRuntimeGeneration(device().ordinal):0;
        if(!runtime_generation_)
            throw std::logic_error("Grouped FFN frontier has no admitted backend generation");
        const auto tensors=extraOperands();
        for(std::size_t i=0;i<tensors.size();++i)
            operands_[i]=TransferEngine{}.retainCapturedDeviceStorage(*tensors[i],device(),tensors[i]->size_bytes());
        for(std::size_t i=0;i<scratch_requirements.size();++i) {
            const auto& required=scratch_requirements[i];
            const auto extent=workspace_->getBufferSize(required.name);
            if(extent<required.minimum_bytes)
                throw std::logic_error("Grouped FFN frontier scratch is undersized: "+
                    std::string(required.name)+" expected "+std::to_string(required.minimum_bytes)+
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
        if(!binding_->executeFrontier(publication_,execution.nativeStream()))return false;
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
