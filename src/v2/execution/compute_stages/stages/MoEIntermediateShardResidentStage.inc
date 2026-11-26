/**
 * @file MoEIntermediateShardResidentStage.inc
 * @brief Cold lowering of an authenticated local FFN region to one retained native program.
 *
 * Included inside the original intermediate-shard builder's private namespace
 * so it consumes the same Binding and table preparation authority. No model,
 * routing, placement, communication or allocation authority is introduced.
 * A full region seals actual external and diagnostic demand before materializing
 * optional former-stage outputs. Internal numerical storage and live routing
 * state retain their original owners and publication ordering.
 */

/** @brief Distinguish model-owned decode from an explicitly borrowed local graph transaction. */
enum class ResidentFFNGraphRole { LocalRegion, MainInference };

/** @brief One participant's complete routed/shared local FFN, excluding router and final collective. */
class ResidentStage final : public IComputeStage, public IWorkspaceConsumer, public IGraphOutputDemand
{
public:
    /** @brief Freeze the original finalizer's already-authenticated source and arena declarations. */
    ResidentStage(std::shared_ptr<Binding> binding,const SharedExpertFFNStage::Params &shared,
        DeviceMoEExpertDescriptor descriptor,TensorBase *input_gate,
        MoEIntermediateShardTensorBinding combined,std::string name,unsigned shared_source_first_column)
        :IComputeStage(binding->device()),binding_(std::move(binding)),shared_(shared),descriptor_(descriptor),
         input_gate_(input_gate),combined_(combined),name_(std::move(name)),
         shared_source_first_column_(shared_source_first_column)
    {
        const auto &p=binding_->params;
        if(p.rows!=1 || p.row_mode!=MoEIntermediateShardRowMode::Decode ||
            p.phase!=MoEOverlayServicePhaseHint::Decode || p.demand!=MoEGroupedPlanDemand::None ||
            binding_->format!=DeviceMoEWeightFormat::NativeQuantised ||
            descriptor_.weight_format!=DeviceMoEWeightFormat::NativeQuantised ||
            !binding_->device().is_rocm() || !shared_.gate_scratch || !shared_.up_scratch ||
            !input_gate_ || !combined_.tensor)
            throw std::invalid_argument("Resident FFN candidate requires original ROCm native-quantised ordinary M1 operands");
    }
    /** @return Original graph-owned router dependency. */
    const std::string& routerName() const noexcept { return binding_->params.router_node; }
    /** @brief Retain the same declarations while admitting the original router inside this node. */
    std::unique_ptr<ResidentStage> withNativeRouter(const MoERoutingStage::Params& router,
        ResidentFFNGraphRole role) const
    {
        if(program_||binding_->prepared||binding_->route_authority!=RouteAuthority::OriginalRouter||
            (role!=ResidentFFNGraphRole::MainInference&&!router.collect_device_runtime_histogram)||
            router.routed_pipeline_kernel_owner!=binding_->params.kernel_owner||
            router.moe_runtime_table!=binding_->params.runtime||router.input!=binding_->params.hidden.tensor||
            router.output_indices!=binding_->params.routing_indices.tensor||router.output_weights!=binding_->params.routing_weights.tensor||
            router.device_id!=device()||router.seq_len!=binding_->params.rows||
            router.d_model!=int(binding_->original.modelColumns())||router.num_experts!=int(binding_->descriptors.size())||
            router.top_k!=binding_->params.top_k||router.routed_row_execution_policy!=binding_->params.row_policy||
            router.active_row_count_device!=(binding_->params.live_rows?binding_->params.live_rows->countOwner():nullptr)||
            router.force_grouped_verifier_prefill_for_decode||router.force_decode_equivalent_verifier_prefill||router.owned_rows||
            router.decode_route_publication!=MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable)
            throw std::logic_error("Native-layout FFN requires cold original router histogram authority");
        auto result=std::make_unique<ResidentStage>(binding_,shared_,descriptor_,input_gate_,combined_,name_,
            shared_source_first_column_);
        result->native_router_=router;
        return result;
    }
    /**
     * @brief Retain the same original FFN owner with one original preparation-only HC prefix.
     * @param prefix Original write/read/source/workspace lifetime authority.
     * @param mixed Original read publication, exactly this router/FFN input.
     * @return Cold complete local-region stage; no standalone HC program is prepared.
     */
    std::unique_ptr<ResidentStage> withFullNativePrefix(std::shared_ptr<Qwen4ExpHCReadPhaseOwner> prefix,
        Qwen4ExpFloatBinding mixed,rocm::FullNativeHCFFNImplementation implementation,ResidentFFNGraphRole role) const {
        if(program_||binding_->prepared||!native_router_||!prefix||full_prefix_||
            mixed.tensor!=binding_->params.hidden.tensor||mixed.buffer!=binding_->params.hidden.id||mixed.byteOffset!=0)
            throw std::logic_error("Full native HC requires cold original router/FFN and exact mixed publication");
        auto result=std::make_unique<ResidentStage>(binding_,shared_,descriptor_,input_gate_,combined_,name_,
            shared_source_first_column_);
        result->graph_role_=role;
        result->native_router_=native_router_;result->full_prefix_=std::move(prefix);result->implementation_=implementation;
        result->all_phase_outputs_=result->full_prefix_->phaseOutputs();
        const auto hc=result->full_prefix_->optionalPhaseOutputs();
        const auto add=[&](FullNativeHCFFNOutputRole role,ITensor* tensor,BufferId id,std::size_t bytes) {
            auto* original=dynamic_cast<TensorBase*>(tensor);
            if(!original)throw std::logic_error("Full-region output demand requires an original tensor owner");
            result->optional_outputs_.push_back({role,name_,original,id,0,bytes});
        };
        add(FullNativeHCFFNOutputRole::HCActivated,hc[0].tensor,hc[0].buffer,hc[0].tensor->size_bytes());
        add(FullNativeHCFFNOutputRole::HCUp,hc[1].tensor,hc[1].buffer,hc[1].tensor->size_bytes());
        const auto& p=binding_->params;const auto h=binding_->original.modelColumns();
        add(FullNativeHCFFNOutputRole::RouteContributions,p.route_partials.tensor,p.route_partials.id,std::size_t(p.top_k)*h*sizeof(float));
        add(FullNativeHCFFNOutputRole::Routed,p.output.tensor,p.output.id,std::size_t(h)*sizeof(float));
        add(FullNativeHCFFNOutputRole::Shared,shared_.output,shared_.output_buffer_id,std::size_t(h)*sizeof(float));
        add(FullNativeHCFFNOutputRole::SharedGate,shared_.gate_scratch,shared_.gate_scratch_buffer_id,std::size_t(shared_.intermediate)*sizeof(float));
        add(FullNativeHCFFNOutputRole::SharedUp,shared_.up_scratch,shared_.up_scratch_buffer_id,std::size_t(shared_.intermediate)*sizeof(float));
        return result;
    }
    /** @return Complete possible observations before native pointers exist; only full regions opt in. */
    std::optional<StageDumpInfo> coldSnapshotPublicationDeclaration() const override
    {return full_prefix_?std::optional<StageDumpInfo>{dumpPublications(true)}:std::nullopt;}
    /** @brief Resolve one immutable output policy from final graph readers and existing snapshot selection.
     * @param graph Final participant graph. @param arena Canonical tensor registry.
     * @param selected Observations selected by the executor's existing filter.
     * @param snapshot_epoch Existing observation-configuration identity.
     */
    void sealGraphOutputDemand(const ComputeGraph& graph,const BufferArena& arena,
        std::span<const std::string> selected,std::uint64_t snapshot_epoch) override {
        if(!full_prefix_)return;
        if(output_seal_) {
            if(output_seal_->topology!=graph.topologyGeneration()||output_seal_->snapshot_epoch!=snapshot_epoch)
                throw std::logic_error("Prepared full-region output demand changed graph or observation generation");
        }
        else if(program_||binding_->prepared)throw std::logic_error("Full-region output demand must precede native preparation");
        const auto declaration=dumpPublications(true);
        auto observations=FullNativeHCFFNOutputDemand::none();
        bool probabilities=false;
        for(const auto& name:selected) {
            const auto found=std::find_if(declaration.outputs.begin(),declaration.outputs.end(),
                [&](const auto& value){return value.name&&name==value.name;});
            if(found==declaration.outputs.end())throw std::invalid_argument("Unknown full-region snapshot publication");
            if(name=="router_logits"){probabilities=true;continue;}
            for(const auto& value:optional_outputs_)if(found->tensor==value.tensor)observations=observations.with(value.role);
        }
        auto demand=resolveFullNativeHCFFNOutputDemand(graph,arena,name_,optional_outputs_,observations);
        if(probabilities)demand=demand.with(FullNativeHCFFNOutputRole::RouterProbabilities);
        // This is a cold lifecycle boundary, not initialized replay. Revalidate
        // canonical bindings and actual observation names even when a caller
        // presents an unchanged epoch; an epoch cannot authenticate substituted
        // tensors or a different output selection on the same prepared owner.
        auto descriptors=MoESelectedDescriptorDemand::ExternalTables;
#ifdef HAVE_ROCM
        descriptors=MoESelectedDescriptorDemand::AcquiredBankOnly;
        for(const auto& name:graph.getExecutionOrder()) {
            const auto* node=graph.getNode(name);
            const auto* consumer=dynamic_cast<const IMoESelectedDescriptorConsumer*>(node->stage.get());
            if(!consumer)continue;
            for(const auto& read:consumer->selectedDescriptorReads()) {
                if(read.producer!=name_)continue;
                if(read.runtime!=binding_->params.runtime||read.model_layer!=binding_->layer||
                   node->stage->device()!=device()||
                   std::find(node->dependencies.begin(),node->dependencies.end(),name_)==node->dependencies.end())
                    throw std::invalid_argument("Selected MoE descriptor read lost its producer, participant, or layer binding");
                descriptors=MoESelectedDescriptorDemand::ExternalTables;
            }
        }
#endif
        if(output_seal_) {
            if(output_seal_->demand!=demand||output_seal_->descriptors!=descriptors)
                throw std::logic_error("Prepared full-region output demand changed within one observation generation");
            return;
        }
        full_prefix_->sealFullOutputDemand(demand);
        output_seal_=OutputSeal{demand,descriptors,graph.topologyGeneration(),snapshot_epoch};
        invalidateDumpInfoCache();
    }
    /** @return The original participant-local projection classification, never a collective. */
    ComputeStageType type() const override { return ComputeStageType::MOE_PROJECTION_PHASE; }
    /** @return Original final local-publication node identity. */
    std::string name() const override { return name_; }
    /** @return This implemented candidate's native backend; no alternate execution is selected. */
    bool supportsBackend(ComputeBackendType backend) const override { return backend==ComputeBackendType::GPU_ROCM; }
    /** @return True only after actual tables, workspace and native occupancy are prepared. */
    bool isGraphCapturable() const override { return bool(program_) && workspace_ && binding_->prepared.has_value(); }
    /** @return All cold preparation is explicit and requires no eager model forward. */
    bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
    /** @return Live M1 emptiness is resolved from the original device-owned row contract. */
    bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
    /** @return Ordinary candidate preparation has no warmup forward. */
    bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
    /** @return Preparation belongs before native recording, never inside enqueue. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return GraphLaunchPreparationPolicy::CaptureOnly; }

    /** @brief Materialize original tables then seal the one native program on the exact caller stream. */
    bool prepareGraphLaunch(IDeviceContext *ctx,void *stream) override
    {
        if(!ctx || ctx->deviceId()!=device() || !stream || !workspace_ || isGraphCaptureActive())
            throw std::logic_error("Resident FFN requires its exact cold participant, workspace and stream");
        if(full_prefix_&&!output_seal_)throw std::logic_error("Full-region output demand was not sealed before native preparation");
#ifdef HAVE_ROCM
        if(rocm::currentResidentMoEFFNProgram()==rocm::ResidentMoEFFNProgram::OrdinaryPhases&&
            binding_->route_authority!=RouteAuthority::OriginalRouter)
            throw std::invalid_argument("Ordinary FFN requires its original paired router, without rebalance or prepared-service publication");
#endif
        if(native_router_)native_router_->moe_runtime_table->prepareDecodeHistogramProducerStream(stream);
        if(full_prefix_) {full_prefix_->bindPhaseWorkspace(workspace_);full_prefix_->preparePhaseBindings(ctx,stream);}
        binding_->prepare(workspace_,stream,outputDemand());
        if(program_){validateStorage();return true;}
        auto *preparer=dynamic_cast<IResidentMoEFFNPreparation *>(binding_->params.kernel_owner->kernel.get());
        if(!preparer)throw std::runtime_error("Selected backend has no resident FFN implementation");
        const auto &p=binding_->params;
        ResidentMoEFFNRequest request;
        request.original=binding_->descriptor_declaration;
        request.runtime=p.runtime->deviceLayerState(binding_->layer);
        request.hidden=dynamic_cast<TensorBase *>(p.hidden.tensor);
        request.rows=p.live_rows.value_or(DeviceRowRange::fullyActive(1));
        request.gate_up_table=binding_->prepared->gate;request.down_table=binding_->prepared->down;
        request.top_k=p.top_k;request.shared=descriptor_;request.shared_input_gate=input_gate_;
        request.route_contributions=p.route_partials.tensor;request.routed_output=p.output.tensor;
        request.shared_output=shared_.output;request.combined_output=combined_.tensor;
        request.shared_gate_scratch=shared_.gate_scratch;request.shared_up_scratch=shared_.up_scratch;
        if(graph_role_==ResidentFFNGraphRole::MainInference)
            request.model_main_inference=ResidentMoEFFNModelCoordinates{binding_->layer,shared_source_first_column_,
                native_router_->collect_device_runtime_histogram?MoERuntimeHistogramProducerRole::ProductionDecode:
                    MoERuntimeHistogramProducerRole::NotProducer};
        std::unique_ptr<PreparedResidentMoEFFN> admitted;
        if(native_router_)
        {
#ifdef HAVE_ROCM
            auto* owner=dynamic_cast<ROCmMoEKernel*>(binding_->params.kernel_owner->kernel.get());
            if(!owner)throw std::logic_error("Native-layout router FFN lost original backend owner");
            if(full_prefix_) {
                auto prefix=full_prefix_->fullPrefixArguments();
                prefix.implementation=implementation_;
                prefix.selected_descriptor_demand=output_seal_->descriptors;
                admitted=owner->prepareFullNativeHCFFN(request,native_router_->gate_weights,
                    native_router_->output_indices,native_router_->output_weights,native_router_->norm_topk_prob,
                    prefix,stream);
            }
            else admitted=owner->prepareNativeLayoutRouterFFN(request,native_router_->gate_weights,
                native_router_->output_indices,native_router_->output_weights,native_router_->norm_topk_prob,stream);
#else
            throw std::logic_error("Native-layout router FFN requires ROCm");
#endif
        }
        else admitted=preparer->prepareResidentMoEFFN(request,stream);
        if(!admitted)throw std::runtime_error("Resident FFN preparation returned no native program");
        const auto addresses=externalAddresses();
        const auto program_identity=admitted->captureSignature();
        if(!program_identity)throw std::logic_error("Resident FFN native preparation has no complete capture identity");
        program_=std::move(admitted);program_identity_=program_identity;addresses_=addresses;
        if(full_prefix_&&outputDemand().contains(FullNativeHCFFNOutputRole::RouterProbabilities)) {
            auto* pointer=workspace_->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS);
            const auto experts=std::size_t(native_router_->num_experts);
            if(!pointer||workspace_->getBufferSize(MoEWorkspaceBuffers::ROUTE_LOGITS)<experts*sizeof(float))
                throw std::logic_error("Full-region probability observation lacks its original workspace");
            router_probability_view_=std::make_unique<GpuTensorView>(pointer,1,experts,TensorType::FP32,device());
        }
        invalidateDumpInfoCache();
        prepared_kernel_owner_=binding_->params.kernel_owner->kernel.get();setGPUStream(stream);
        const auto &native=program_->programEvidence();
        PerfStatsCollector::addCounter("moe_resident_ffn","prepared_programs",1.,"setup",device().toString(),
            {{"layer",std::to_string(binding_->layer)}});
        for(std::size_t index=0;index<native.nativeEntries().size();++index)
        {
            const auto& entry=native.nativeEntries()[index];
            PerfStatsCollector::addCounter("moe_resident_ffn","prepared_native_entries",1.,"setup",device().toString(),
                {{"layer",std::to_string(binding_->layer)},{"entry",std::to_string(index)},
                 {"launch_kind",native.kind==ResidentMoEFFNLaunchKind::Cooperative?"cooperative":"ordinary_phases"},
                 {"threads",std::to_string(entry.threads)},{"grid",std::to_string(entry.grid)},
                 {"registers_per_thread",std::to_string(entry.resources.registers_per_thread)}});
        }
        return true;
    }
    /** @brief Submit the exact ordered native program and publish every original output on its exact stream. */
    bool execute(IDeviceContext *ctx) override
    {
        if(!ctx || ctx->deviceId()!=device() || !program_)throw std::logic_error("Resident FFN is not prepared");
        validateStorage();
        const auto execution=gpuExecution();
        if(full_prefix_) {
            full_prefix_->validatePhaseBindings(execution.nativeStream());
            for(const auto& input:full_prefix_->phaseInputs())execution.requirePreparedInput(input.tensor);
            for(auto* weight:full_prefix_->phaseDeclaration().bufferContract().weight_tensors) {
                auto* tensor=dynamic_cast<TensorBase*>(weight);
                if(!tensor)throw std::logic_error("Full HC original direct weight lost its tensor owner");
                execution.requirePreparedInput(tensor);
            }
            for(const auto& output:full_prefix_->phaseOutputs())execution.requirePreparedOutput(output.tensor);
        } else for(const auto &input:inputs())if(!native_router_||input.tensor==binding_->params.hidden.tensor)
            execution.requirePreparedInput(input.tensor);
        if(native_router_)
        {
            execution.requirePreparedInput(native_router_->gate_weights);
            execution.requirePreparedOutput(native_router_->output_indices);
            execution.requirePreparedOutput(native_router_->output_weights);
        }
        execution.requirePreparedInput(input_gate_);
        for(const auto &output:outputs())execution.requirePreparedOutput(output.tensor);
        program_->enqueue(execution.nativeStream());
        for(const auto &output:outputs())execution.publish(output.tensor);
        if(full_prefix_)for(const auto& output:full_prefix_->phaseOutputs())execution.publish(output.tensor);
        if(native_router_){execution.publish(native_router_->output_indices);execution.publish(native_router_->output_weights);}
        PerfStatsCollector::addCounter("moe_resident_ffn","local_program_enqueued",1.,"graph_capture",device().toString(),
            {{"layer",std::to_string(binding_->layer)},{"publication","local_partial"}});
        return true;
    }
    /** @brief Retire the native program before releasing its original descriptor/workspace preparation. */
    void invalidateKernelDynamicState() override { program_.reset();program_identity_=0;prepared_kernel_owner_=nullptr;
        router_probability_view_.reset();output_seal_.reset();
        if(full_prefix_)full_prefix_->invalidatePhaseBindings();binding_->invalidate();invalidateDumpInfoCache(); }
    /** @return Complete policy/source/arena identity; no live route or source-byte hashing. */
    std::uint64_t graphCaptureVariantSignature() const override
    {
        binding_->validatePreparedStorage();if(program_)validateStorage();
        auto signature=binding_->signature;
        const auto mix=[&](std::uint64_t value){signature=(signature^value)*kFNV1a64Prime;};
        mix(static_cast<unsigned>(MoEIntermediateShardFFNExecution::ResidentCandidate));
        mix(bool(full_prefix_));
        mix(outputDemand().bits());
        if(full_prefix_) {
            const auto descriptors=output_seal_?output_seal_->descriptors:MoESelectedDescriptorDemand::ExternalTables;
            mix(static_cast<std::uint32_t>(descriptors));
        }
        if(full_prefix_)mix(full_prefix_->phaseDeclaration().graphCaptureVariantSignature());
        mix(bool(native_router_));
        if(native_router_){mix(reinterpret_cast<std::uintptr_t>(native_router_->gate_weights));mix(native_router_->norm_topk_prob);}
        for(const auto &output:outputs()){mix(reinterpret_cast<std::uintptr_t>(output.tensor));mix(static_cast<unsigned>(output.id));}
        for(const auto *weight:{input_gate_,shared_.gate_w,shared_.up_w,shared_.down_w})mix(reinterpret_cast<std::uintptr_t>(weight));
        mix(shared_.intermediate);mix(reinterpret_cast<std::uintptr_t>(shared_.prepared_store));
        // The declarative signature must not change merely because the arena
        // became physical. Native preparation separately seals its complete
        // address/function/generation identity and validateStorage rejects any
        // change to that seal before identity queries or submission.
        mix(static_cast<unsigned>(descriptor_.weight_format));
        for(const auto &matrix:{descriptor_.gate,descriptor_.up,descriptor_.down})
        {
            mix(reinterpret_cast<std::uintptr_t>(matrix.payload));mix(reinterpret_cast<std::uintptr_t>(matrix.scales));
            mix(reinterpret_cast<std::uintptr_t>(matrix.mins));mix(reinterpret_cast<std::uintptr_t>(matrix.emins));
            mix(matrix.n);mix(matrix.k);mix(matrix.blocks_per_row);mix(matrix.codebook_id);
            mix(matrix.allocation_payload_bytes_per_block);mix(matrix.allocation_has_mins);mix(matrix.allocation_has_emins);
            mix(matrix.source_codebook_id);mix(matrix.source_is_superblock);mix(matrix.source_identity_present);
        }
        return signature ? signature : 1;
    }
    /** @return Original public output/weight dependencies; scratch remains physically arena-owned. */
    StageBufferContract bufferContract() const override
    {
        auto contract=StageBufferContract::build();
        for(const auto &input:inputs())if(!full_prefix_&&(!native_router_||input.tensor==binding_->params.hidden.tensor))contract.addInput(input.id);
        if(native_router_)
        {
            contract.addWeight(native_router_->gate_weights);
            contract.addOutput(native_router_->output_indices_buffer_id);
            contract.addOutput(native_router_->output_weights_buffer_id);
        }
        for(const auto &output:outputs())contract.addOutput(output.id);
        if(full_prefix_) {
            const auto original=full_prefix_->phaseDeclaration().bufferContract();
            contract.inputs.insert(contract.inputs.end(),original.inputs.begin(),original.inputs.end());
            contract.outputs.insert(contract.outputs.end(),original.outputs.begin(),original.outputs.end());
            contract.weight_tensors.insert(contract.weight_tensors.end(),original.weight_tensors.begin(),original.weight_tensors.end());
            contract.prepared_weights.insert(contract.prepared_weights.end(),original.prepared_weights.begin(),original.prepared_weights.end());
            contract.inouts.insert(contract.inouts.end(),original.inouts.begin(),original.inouts.end());
        }
        contract.addWeight(input_gate_);
        contract.addPreparedWeight(shared_.gate_w,shared_.prepared_store,*shared_.prepared_ref_gate);
        contract.addPreparedWeight(shared_.up_w,shared_.prepared_store,*shared_.prepared_ref_up);
        return contract.addPreparedWeight(shared_.down_w,shared_.prepared_store,*shared_.prepared_ref_down);
    }
    /** @return Existing tensor geometries without allocating or estimating physical storage. */
    StageBufferRequirements getBufferRequirements() const override
    {
        StageBufferRequirements result;std::size_t index=0;
        for(const auto &input:inputs())if(!full_prefix_&&(!native_router_||input.tensor==binding_->params.hidden.tensor))
            result.addInput("resident_input_"+std::to_string(index++),input.tensor->shape(),toBufferTensorType(input.tensor->native_type()));
        if(native_router_)
        {
            result.addOutput("router_indices",native_router_->output_indices->shape(),toBufferTensorType(native_router_->output_indices->native_type()));
            result.addOutput("router_weights",native_router_->output_weights->shape(),toBufferTensorType(native_router_->output_weights->native_type()));
        }
        if(full_prefix_) {
            const auto original=full_prefix_->phaseDeclaration().getBufferRequirements();
            result.buffers.insert(result.buffers.end(),original.buffers.begin(),original.buffers.end());
        }
        index=0;
        for(const auto &output:outputs())result.addOutput("resident_output_"+std::to_string(index++),output.tensor->shape(),toBufferTensorType(output.tensor->native_type()));
        return result;
    }
    /** @return Original MoE scratch sized for both source-local intermediate widths. */
    WorkspaceRequirements getWorkspaceRequirements(int m,int=0,int=0) const override
    {
        auto requirements=MoEWorkspaceBuffers::rocmMoE(std::max(m,1),binding_->original.modelColumns(),
            std::max(int(binding_->original.localIntermediateColumns()),shared_.intermediate),
            int(binding_->descriptors.size()),binding_->params.top_k);
        if(full_prefix_)requirements.merge(full_prefix_->phaseWorkspaceRequirements());
#ifdef HAVE_ROCM
        if(native_router_)
        {
            MoEWorkspaceBuffers::add(requirements,MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER,
                (full_prefix_?rocm::fullNativeHCFFNSlotBytes(implementation_):NativeLayoutBarrierLayout::bytes)*
                    MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,WorkspaceContentLifetime::SerialGraphFamily);
        }
#else
        if(native_router_)throw std::logic_error("Native-layout router FFN workspace requires ROCm");
#endif
        return requirements;
    }
    /** @brief Borrow only the executor's admitted workspace. */
    void bindWorkspace(DeviceWorkspaceManager *workspace) override
    { if(program_ && workspace_!=workspace)throw std::logic_error("Resident FFN cannot rebind a prepared workspace");
      if(full_prefix_)full_prefix_->bindPhaseWorkspace(workspace);workspace_=workspace; }
    /** @brief Withdraw the borrow only after prepared/native graph retirement. */
    void unbindWorkspace() override
    { if(program_)throw std::logic_error("Resident FFN must retire its prepared program before workspace release");
      if(full_prefix_)full_prefix_->bindPhaseWorkspace(nullptr);workspace_=nullptr; }
    /** @return Whether the original executor has supplied admitted storage. */
    bool hasWorkspace() const override { return workspace_!=nullptr; }
    /** @return Exact executor-owned workspace, without a second allocator. */
    DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }
    /** @return Inputs are immutable for this operation; all output planes are published. */
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
    /** @return Only materialized values are observable in this exact prepared variant. */
    StageDumpInfo buildDumpInfoImpl() const override {return dumpPublications(false);}
private:
    /** @brief The existing graph and snapshot generations authenticate one immutable native policy. */
    struct OutputSeal {FullNativeHCFFNOutputDemand demand;MoESelectedDescriptorDemand descriptors;std::uint64_t topology,snapshot_epoch;};
    /** @return All standalone outputs or the full graph's cold-resolved demand. */
    FullNativeHCFFNOutputDemand outputDemand() const noexcept
    {return output_seal_?output_seal_->demand:FullNativeHCFFNOutputDemand::all();}
    /** @brief Author stable semantic output names before and after native preparation.
     * @param cold Whether to describe all possible outputs for existing snapshot selection.
     * @return Metadata only; no tensor payload or live count is read.
     */
    StageDumpInfo dumpPublications(bool cold) const {
        StageDumpInfo info;const auto& p=binding_->params;const auto h=binding_->original.modelColumns();
        info.addScalarInt("resident_ffn",1);
        const auto add=[&](FullNativeHCFFNOutputRole role,const char* name,const ITensor* tensor,std::size_t rows,std::size_t columns) {
            if(cold||!full_prefix_||outputDemand().contains(role))info.addOutput(name,tensor,rows,columns);
        };
        add(FullNativeHCFFNOutputRole::RouteContributions,"local_route_partials",p.route_partials.tensor,p.top_k,h);
        add(FullNativeHCFFNOutputRole::Routed,"local_routed_partial",p.output.tensor,1,h);
        add(FullNativeHCFFNOutputRole::Shared,"shared_local_partial",shared_.output,1,h);
        info.addOutput("local_combined",combined_.tensor,1,h);
        if(full_prefix_) {
            add(FullNativeHCFFNOutputRole::SharedGate,"shared_gate",shared_.gate_scratch,1,shared_.intermediate);
            add(FullNativeHCFFNOutputRole::SharedUp,"shared_up",shared_.up_scratch,1,shared_.intermediate);
            static constexpr std::array<const char*,7> names{"hc_public_0","hc_public_1","hc_public_2",
                "hc_public_3","hc_public_4","hc_public_5","hc_public_6"};
            for(std::size_t index=0;index<all_phase_outputs_.size();++index) {
                if(index>=names.size())throw std::logic_error("Full HC public dump extent exceeds its original declarations");
                const auto& output=all_phase_outputs_[index];
                const auto optional=std::find_if(optional_outputs_.begin(),optional_outputs_.end(),
                    [&](const auto& value){return value.tensor==output.tensor;});
                if(cold||optional==optional_outputs_.end()||outputDemand().contains(optional->role))
                    info.addOutput(names[index],output.tensor,1,output.tensor->numel());
            }
            if(cold)info.addOutput("router_logits",static_cast<const float*>(nullptr),1,native_router_->num_experts);
            else if(router_probability_view_)info.addOutput("router_logits",router_probability_view_.get(),1,native_router_->num_experts);
        }
        return info;
    }
    /** @return Original immutable activation declarations in stable order. */
    std::array<MoEIntermediateShardTensorBinding,3> inputs() const
    { return {binding_->params.hidden,binding_->params.routing_indices,binding_->params.routing_weights}; }
    /** @return Actual demanded publications; internal numerical work does not depend on these mirrors. */
    std::vector<MoEIntermediateShardTensorBinding> outputs() const {
        std::vector<MoEIntermediateShardTensorBinding> result;
        const auto add=[&](FullNativeHCFFNOutputRole role,MoEIntermediateShardTensorBinding binding) {
            if(outputDemand().contains(role))result.push_back(binding);
        };
        add(FullNativeHCFFNOutputRole::RouteContributions,binding_->params.route_partials);
        add(FullNativeHCFFNOutputRole::Routed,binding_->params.output);
        add(FullNativeHCFFNOutputRole::Shared,{shared_.output,shared_.output_buffer_id});result.push_back(combined_);
        add(FullNativeHCFFNOutputRole::SharedGate,{shared_.gate_scratch,shared_.gate_scratch_buffer_id});
        add(FullNativeHCFFNOutputRole::SharedUp,{shared_.up_scratch,shared_.up_scratch_buffer_id});return result;
    }
    /** @return Physical addresses sealed after materialization, never a mutable execution-state mirror. */
    std::vector<const void*> externalAddresses() const
    {
        std::vector<const void*> result;
        for(const auto &output:outputs())result.push_back(output.tensor->gpu_data_ptr());
        result.push_back(input_gate_->gpu_data_ptr());
        for(const auto *address:result)if(!address)throw std::logic_error("Resident FFN has an unmaterialized original output/weight");
        return result;
    }
    /** @brief Reject stale workspace, stream or arena storage before retained submission or identity use. */
    void validateStorage() const
    {
        // The backend program retains its original descriptor owner. Zero
        // staged-publication signatures cannot authenticate a substitute owner.
        if(!prepared_kernel_owner_||binding_->params.kernel_owner->kernel.get()!=prepared_kernel_owner_)
            throw std::logic_error("Resident FFN original prepared kernel owner changed");
        binding_->validatePreparedStorage();
        if(!program_ || !binding_->prepared || binding_->prepared->workspace!=workspace_ ||
            binding_->prepared->workspace_id!=workspace_->id() || addresses_!=externalAddresses() ||
            program_->captureSignature()!=program_identity_)
            throw std::logic_error("Resident FFN lost its original prepared storage generation");
        if(full_prefix_)full_prefix_->validatePhaseBindings(requireGPUStream());
        program_->validateBinding();
    }
    std::shared_ptr<Qwen4ExpHCReadPhaseOwner> full_prefix_; ///< Original retained binding owner, no nested execution.
    rocm::FullNativeHCFFNImplementation implementation_=rocm::FullNativeHCFFNImplementation::Canonical; ///< Cold exact full-region native body.
    std::optional<MoERoutingStage::Params> native_router_;
    const std::shared_ptr<Binding> binding_;
    const SharedExpertFFNStage::Params shared_;
    const DeviceMoEExpertDescriptor descriptor_;
    TensorBase *const input_gate_;
    const MoEIntermediateShardTensorBinding combined_;
    const std::string name_;
    const unsigned shared_source_first_column_; ///< Original authenticated shared down K origin.
    ResidentFFNGraphRole graph_role_=ResidentFFNGraphRole::LocalRegion;
    DeviceWorkspaceManager *workspace_=nullptr;
    std::unique_ptr<PreparedResidentMoEFFN> program_;
    std::uint64_t program_identity_=0;
    const IMoEKernel* prepared_kernel_owner_=nullptr; ///< Exact backend instance borrowed by the sealed native program.
    std::vector<const void*> addresses_;
    std::vector<FullNativeHCFFNOutputBinding> optional_outputs_; ///< Original logical roles; no second storage owner.
    std::vector<Qwen4ExpFloatBinding> all_phase_outputs_; ///< Stable original diagnostic names across materialization choices.
    std::optional<OutputSeal> output_seal_; ///< The single immutable cold publication decision.
    std::unique_ptr<ITensor> router_probability_view_; ///< View of original ROUTE_LOGITS workspace only when observed.
};
