/**
 * @file Qwen4ExpPersistentHCReadStage.inc
 * @brief Exact HC read/injection graph owner over original prepared sources and arena bytes.
 *
 * Included in the HC builder's private namespace. This owner replaces only
 * local math, retains every original public publication, and borrows the
 * original prepared engines and named workspace. One admitted cooperative
 * node publishes all values. Full composition borrows the same binding-only
 * owner with its explicit previous write; every collective remains outside.
 */
class PersistentHCReadStage final : public IComputeStage, public IWorkspaceConsumer, public Qwen4ExpHCReadPhaseOwner
{
    /** @brief One original tensor allocation and its complete constructor/storage identity. */
    struct TensorSeal {
        TensorBase* tensor; std::vector<std::size_t> shape; std::size_t bytes;
        TensorType type; void* pointer=nullptr;
        std::shared_ptr<const CapturedDeviceStorageLease> lease;
    };
    /** @brief One canonical named workspace borrow; no independent capacity ledger. */
    struct BankSeal {
        std::string name; void* pointer; std::size_t bytes;
        std::shared_ptr<const WorkspaceBufferLease> lease;
    };
    /** @brief Original prepared source declaration; its engine/store retains packed weight allocations. */
    struct SourceSeal {const ITensor* tensor;std::vector<std::size_t> shape;std::size_t bytes;TensorType type;ITensorGemm* engine;};
    /** @brief Raw original row-control span borrows its existing allocator record, never a new allocation/PMA claim. */
    struct CountSeal {const std::int32_t* pointer;NativeAllocationCaptureLease lease;};
public:
    /**
     * @brief Retain original graph declarations before their individual nodes retire.
     * @param context Original participant, rows, arithmetic and mixer geometry.
     * @param norm Original branch norm, including its actual immutable gain.
     * @param down Original prepared down declaration and canonical workspace quote.
     * @param up Original two-term up declaration and query-row authority.
     * @param inject Original FP32 injection declaration, independently live until write.
     * @param buffers Original public arena views, retaining exact permitted down alias.
     * @param injection Original four-logit publication consumed by the later HC write.
     * @param terminal Original read-node identity retained for every outside consumer.
     * @param workspace_requirements Exact original ordered workspace quote captured before graph replacement.
     * @param previous_write Optional original write immediately publishing the full prefix input.
     */
    PersistentHCReadStage(Qwen4ExpHyperconnectionContext context,Qwen4ExpPrimitiveStage::BranchRMSNorm norm,
        GEMMStage::Params down,GEMMStage::Params up,std::optional<GEMMStage::Params> inject,
        Qwen4ExpHyperconnectionReadBuffers buffers,std::optional<Qwen4ExpFloatBinding> injection,std::string terminal,
        WorkspaceRequirements workspace_requirements,
        std::optional<Qwen4ExpPrimitiveStage::ResidualWrite> previous_write=std::nullopt,
        Qwen4ExpHCTrunkSource trunk_source=Qwen4ExpHCTrunkSource::PreviousWrite)
        :IComputeStage(context.device),context_(std::move(context)),norm_(std::move(norm)),
         down_(std::move(down)),up_(std::move(up)),inject_(std::move(inject)),buffers_(buffers),
         injection_(injection),terminal_(std::move(terminal)),previous_write_(std::move(previous_write)),trunk_source_(trunk_source),workspace_requirements_(std::move(workspace_requirements))
    {
        check(supportsBackend(device().is_cuda() ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM)&&
            device().is_gpu()&&context_.rows.capacity==1&&context_.arithmetic==Qwen4ExpProjectionArithmetic::Ordinary&&
            up_.quantized_operand_precision==QuantizedActivationOperandPrecision::TwoTermQ8,
            "Persistent HC requires the explicit original ordinary GPU M1 two-term operation");
        if (device().is_cuda())
            check((previous_write_ || trunk_source_ == Qwen4ExpHCTrunkSource::PublishedTrunk) &&
                  context_.readExecution == Qwen4ExpHCReadExecution::Staged,
                "CUDA retained HC requires staged ordinary previous-write or published-trunk bindings");
        if(trunk_source_==Qwen4ExpHCTrunkSource::PublishedTrunk)
            check(!previous_write_&&!inject_&&!ctaOwned(),"Published HC trunk requires its original standalone input and injection");
        check(inject_.has_value()==injection_.has_value(),"Original HC injection source and output must be present together");
        check(norm_.hidden==context_.hidden&&std::bit_cast<std::uint32_t>(norm_.epsilon)==std::bit_cast<std::uint32_t>(context_.epsilon)&&
            norm_.input.tensor==buffers_.trunk.tensor&&norm_.input.buffer==buffers_.trunk.buffer&&
            norm_.output.tensor==buffers_.normalized.tensor&&norm_.output.buffer==buffers_.normalized.buffer,
            "Persistent HC original norm declaration differs from its mixer context");
        check(norm_.diagnostic_input_publication==Qwen4ExpPrimitiveStage::DiagnosticInputPublication::None&&
            norm_.input_role==Qwen4ExpPrimitiveStage::BranchNormInputRole::Ordinary,
            "Persistent HC does not substitute a diagnostic or predictor norm");
        check(context_.persistentGrid==Qwen4ExpHCPersistentGrid::OnePerProcessor||context_.persistentGrid==Qwen4ExpHCPersistentGrid::MaximumResident,
            "Persistent HC has an unknown resident grid policy");
        if(ctaOwned())check(context_.downProducerProgram==HCDownProducerProgram::Original&&
            context_.branchNormExecution==Qwen4ExpHCBranchNormExecution::Staged&&!context_.projectionRows,
            "CTA-owned HC requires unchanged original fully-active down producer and separate norm");
        check(up_.hc_residual_down&&up_.hc_residual_down->activated==buffers_.activatedDown.tensor,
            "Persistent HC requires the original /4-SiLU activated output");
        for(const auto* projection:projections()) {
            check(projection->prepared_store&&projection->prepared_ref&&projection->alpha==1.f&&projection->beta==0.f&&
                !projection->transpose_B&&!projection->bias&&!projection->bias_tensor,
                "Persistent HC requires exact unbiased prepared original projections");
        }
        check(down_.A==buffers_.normalized.tensor&&down_.C==buffers_.down.tensor&&
            up_.A==buffers_.down.tensor&&up_.C==buffers_.readLogits.tensor&&
            (!inject_||(inject_->A==buffers_.normalized.tensor&&inject_->C==injection_->tensor))&&
            down_.n==context_.lowRank&&down_.k==4*context_.hidden&&up_.n==4*context_.hidden&&up_.k==context_.lowRank&&
            (!inject_||(inject_->n==4&&inject_->k==4*context_.hidden)),
            "Persistent HC lost its original projection operands or dimensions");
        for(const auto binding:publicRoots()) {
            check(binding.tensor&&binding.byteOffset==0,"Persistent HC requires original zero-offset public tensor views");
            roots_.push_back({binding.tensor,binding.tensor->shape(),binding.tensor->size_bytes(),binding.tensor->native_type()});
        }
        auto* gain=const_cast<TensorBase*>(norm_.gain.tensor);
        roots_.push_back({gain,gain->shape(),gain->size_bytes(),gain->native_type()});
        for(const auto* p:projections()) {
            check(p->B!=nullptr,"Persistent HC original prepared source is absent");
            sources_.push_back({p->B,p->B->shape(),p->B->size_bytes(),p->B->native_type(),&engine(*p)});
        }
        if(previous_write_) {
            const auto& write=*previous_write_;
            check(!ctaOwned()&&write.hidden==context_.hidden&&write.output.tensor==buffers_.trunk.tensor&&
                write.output.buffer==buffers_.trunk.buffer&&write.output.byteOffset==0,
                "Full HC write must publish the exact original norm input");
            for(const auto& input:std::array{write.residual,write.block,write.gateLogits}) {
                check(input.tensor&&input.byteOffset==0,"Full HC write requires original zero-offset roots");
                const bool exists=std::any_of(roots_.begin(),roots_.end(),[&](const auto& item){return item.tensor==input.tensor;});
                // The original model reuses its four-logit bank: previous
                // write consumes it before the same region's norm/injection
                // path overwrites it. The factory proved write→norm→injection
                // edges before issuing this immutable phase owner. A different
                // tensor/view/buffer or a partial overlap is never this alias.
                const bool gate_injection_reuse=inject_&&injection_&&
                    input.tensor==write.gateLogits.tensor&&input.buffer==write.gateLogits.buffer&&
                    input.tensor==injection_->tensor&&input.buffer==injection_->buffer&&
                    input.byteOffset==injection_->byteOffset&&input.byteOffset==0&&
                    input.tensor->native_type()==TensorType::FP32&&input.tensor->cols()==4&&
                    input.tensor->rows()>=1&&input.tensor->size_bytes()>=4*sizeof(float);
                check(!exists||(input.tensor==write.residual.tensor&&input.tensor==buffers_.trunk.tensor&&input.buffer==buffers_.trunk.buffer)||
                    gate_injection_reuse,
                    "Full HC write input aliases an unrelated original public value or gain");
                if(!exists)roots_.push_back({input.tensor,input.tensor->shape(),input.tensor->size_bytes(),input.tensor->native_type()});
            }
        }
        // Only the exact raw/activated down alias is permitted; every other
        // original public arena allocation must remain independently live.
        validateDeclaredRoots();
        (void)normRows();
    }
    /** @return This original declaration without any standalone execution admission. */
    const IComputeStage& phaseDeclaration() const override {return *this;}
    /** @brief Retain original workspace. @param workspace Existing executor owner. */
    void bindPhaseWorkspace(DeviceWorkspaceManager* workspace) override {bindWorkspace(workspace);}
    /** @brief Resolve original bindings only. @param ctx Exact worker. @param stream Cold exact stream. */
    void preparePhaseBindings(IDeviceContext* ctx,void* stream) override {prepareBindings(ctx,stream);}
    /** @brief Reject stale original borrows. @param stream Prepared exact stream. */
    void validatePhaseBindings(void* stream) const override {validateStorage(stream);}
    /** @return Original named-bank quote, without admitting any kernel. */
    WorkspaceRequirements phaseWorkspaceRequirements() const override {return workspace_requirements_;}
    /** @return Original phase/write arguments without a barrier or separate admission. */
    rocm::FullNativeHCFFNPrefixArgs fullPrefixArguments() const override {
        check(args_.has_value(),"Full HC arguments require original prepared read bindings");
        if(trunk_source_==Qwen4ExpHCTrunkSource::PublishedTrunk)
            return {*args_,{},nullptr,output_demand_};
        check(previous_write_.has_value(),"Full HC arguments require the original previous write");
        const auto& w=*previous_write_;
        return {*args_,{static_cast<const float*>(w.residual.tensor->gpu_data_ptr()),
            static_cast<const float*>(w.block.tensor->gpu_data_ptr()),static_cast<const float*>(w.gateLogits.tensor->gpu_data_ptr()),
            static_cast<float*>(w.output.tensor->gpu_data_ptr())},nullptr,output_demand_};
    }
#ifdef HAVE_CUDA
    /** @copydoc Qwen4ExpHCReadPhaseOwner::cudaPrefixArguments */
    const cuda::moe::RetainedHCPrefix& cudaPrefixArguments() const override {
        check(cuda_args_.has_value(),"Full CUDA HC arguments require original prepared read bindings");
        return *cuda_args_;
    }
#endif
    /** @copydoc Qwen4ExpHCReadPhaseOwner::trunkSource */
    Qwen4ExpHCTrunkSource trunkSource() const noexcept override {return trunk_source_;}
    /** @return Original inputs for this declaration's numerical boundary. */
    std::vector<Qwen4ExpFloatBinding> phaseInputs() const override {
        if(previous_write_)return {previous_write_->residual,previous_write_->block,previous_write_->gateLogits};
        return {ctaOwned()?buffers_.normalized:buffers_.trunk};
    }
    /** @return Distinct original public outputs including write and injection. */
    std::vector<Qwen4ExpFloatBinding> phaseOutputs() const override {
        std::vector<Qwen4ExpFloatBinding> result;
        if(previous_write_)result.push_back(previous_write_->output);
        forEachOutput([&](const auto& output){result.push_back(output);});return result;
    }
    /** @copydoc Qwen4ExpHCReadPhaseOwner::sealFullOutputDemand */
    void sealFullOutputDemand(FullNativeHCFFNOutputDemand demand) override {
        check(previous_write_.has_value()||trunk_source_==Qwen4ExpHCTrunkSource::PublishedTrunk,
            "Only a complete full-region owner may narrow HC publications");
        check(!bound()||demand==output_demand_,"Prepared HC output demand cannot change");
        output_demand_=demand;
        cached_signature_=0;
    }
    /** @copydoc Qwen4ExpHCReadPhaseOwner::optionalPhaseOutputs */
    std::array<Qwen4ExpFloatBinding,2> optionalPhaseOutputs() const override
    {return {buffers_.activatedDown,buffers_.readLogits};}
    /** @brief Retire the original phase leases after enclosing execution retires. */
    void invalidatePhaseBindings() override {invalidateKernelDynamicState();}
    /** @return Original local read classification, never communication. */
    ComputeStageType type() const override {return ComputeStageType::QWEN4EXP_RESIDUAL_READ;}
    /** @return The retained original read frontier name. */
    std::string name() const override {return terminal_;}
    /** @return Only the explicitly implemented native participant. */
    bool supportsBackend(ComputeBackendType backend) const override {
        switch (backend)
        {
#ifdef HAVE_CUDA
            case ComputeBackendType::GPU_CUDA: return device().is_cuda();
#endif
#ifdef HAVE_ROCM
            case ComputeBackendType::GPU_ROCM: return device().is_rocm();
#endif
            default: return false;
        }
    }
    /** @return Capture requires completed exact native program preparation. */
    bool isGraphCapturable() const override {return prepared();}
    /** @return Cold preparation needs no eager forward. */
    bool supportsGraphCaptureAfterLaunchPreparation() const override {return true;}
    /** @return Lazy graph preflight may rely on explicit cold preparation without an eager forward. */
    bool supportsLazyPrefillGraphCapturePreflight() const override {return true;}
    /** @return The native program retains original row authorities. */
    bool supportsPaddedPrefillGraphCapturePreflight() const override {return true;}
    /** @return All physical binding is cold and explicit. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override {return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Original public input, output and prepared-weight ownership. */
    StageBufferContract bufferContract() const override {
        auto result=StageBufferContract::build();
        for(const auto& input:phaseInputs())result.addInput(input.buffer);
        if(previous_write_)result.addOutput(previous_write_->output.buffer);
        forEachOutput([&](const auto& output){result.addOutput(output.buffer);});
        if(!ctaOwned())result.addWeight(const_cast<TensorBase*>(norm_.gain.tensor));
        for(const auto* projection:projections())
            if(!ctaOwned()||!inject_||projection!=&*inject_)result.addPreparedWeight(const_cast<ITensor*>(projection->B),projection->prepared_store,*projection->prepared_ref);
        return result;
    }
    /** @return Original arena geometries; this stage declares no new activation allocation. */
    StageBufferRequirements getBufferRequirements() const override {
        StageBufferRequirements result;
        unsigned input_index=0;
        for(const auto& input:phaseInputs())result.addInput("persistent_hc_input_"+std::to_string(input_index++),input.tensor->shape(),toBufferTensorType(input.tensor->native_type()));
        if(previous_write_)result.addOutput("persistent_hc_write",previous_write_->output.tensor->shape(),BufferTensorType::FP32);
        std::size_t index=0;
        forEachOutput([&](const auto& output){result.addOutput("persistent_hc_output_"+std::to_string(index++),
            output.tensor->shape(),toBufferTensorType(output.tensor->native_type()));});
        return result;
    }
    /** @return The exact original ordered canonical BOM, without rebuilding stages or repacking its physical tail. */
    WorkspaceRequirements getWorkspaceRequirements(int,int=0,int=0) const override {
        return workspace_requirements_;
    }
    /** @brief Borrow only the executor's already-admitted workspace. @param workspace Original named-bank owner. */
    void bindWorkspace(DeviceWorkspaceManager* workspace) override {
        if(bound()&&workspace_!=workspace)throw std::logic_error("Prepared persistent HC cannot rebind workspace");
        workspace_=workspace;
    }
    /** @brief Withdraw the borrow only after all native program borrowers retire. */
    void unbindWorkspace() override {
        if(bound())throw std::logic_error("Persistent HC program must retire before workspace release");workspace_=nullptr;
    }
    /** @return Whether canonical workspace ownership has been supplied. */
    bool hasWorkspace() const override {return workspace_!=nullptr;}
    /** @return The original executor-owned workspace. */
    DeviceWorkspaceManager* getWorkspace() const override {return workspace_;}
    /**
     * @brief Resolve exact immutable sources, lease existing storage and admit the real native entry.
     * @param ctx Original participant worker. @param stream Exact nondefault preparation/capture stream.
     * @return True after complete native admission; no warmup or alternate program is executed.
     */
    bool prepareGraphLaunch(IDeviceContext* ctx,void* stream) override {
#ifdef HAVE_ROCM
        check(device().is_rocm(), "CUDA HC bindings require their enclosing resident program");
        prepareBindings(ctx,stream);
        if(prepared())return true;
        if(ctaOwned())ordinary_=rocm::prepareCTAOwnedHCRead(device(),*args_,workspace_id_);
        else admission_=rocm::preparePersistentHCRead(device(),*args_,workspace_id_);
        validateStorage(stream);return true;
#else
        (void)ctx;
        (void)stream;
        throw std::logic_error("CUDA HC bindings require their enclosing resident program");
#endif
    }
    /**
     * @brief Prepare the one original binding owner without function admission.
     * @param ctx Exact original worker. @param stream Nondefault cold stream.
     * @return True after all tensor/source/count/workspace borrows are sealed.
     */
    bool prepareBindings(IDeviceContext* ctx,void* stream) {
        if(!ctx||ctx->deviceId()!=device()||!workspace_||workspace_->device()!=device()||
            !workspace_->isAllocated()||reinterpret_cast<std::uintptr_t>(stream)<=2||isGraphCaptureActive())
            throw std::logic_error("Persistent HC requires exact cold context/workspace/stream");
        if(bound()){validateStorage(stream);return true;}
        validateDeclaredRoots();
        const auto requirements=getWorkspaceRequirements(1);
        std::vector<BankSeal> banks;
        for(const auto& requirement:requirements.buffers) {
            auto* pointer=workspace_->getBuffer(requirement.name);
            const auto bytes=workspace_->getBufferSize(requirement.name);
            check(pointer&&bytes>=requirement.size_bytes,"Persistent HC canonical workspace quote was not materialized");
            banks.push_back({requirement.name,pointer,bytes,workspace_->retainBuffer(requirement.name,0,bytes)});
        }
        for(auto& root:roots_) {
            if(!requiresStorage(root.tensor)){root.pointer=nullptr;root.lease.reset();continue;}
            check(root.tensor->current_device()==device()&&root.tensor->gpu_data_ptr(),"Persistent HC original tensors are not prebound");
            root.pointer=root.tensor->gpu_data_ptr();
            root.lease=TransferEngine{}.retainCapturedDeviceStorage(*root.tensor,device(),root.bytes);
        }
        if(previous_write_&&injection_&&previous_write_->gateLogits.tensor==injection_->tensor) {
            const auto& gate=previous_write_->gateLogits;
            check(gate.buffer==injection_->buffer&&gate.byteOffset==0&&injection_->byteOffset==0&&
                gate.tensor->current_device()==device()&&gate.tensor->gpu_data_ptr()==injection_->tensor->gpu_data_ptr()&&
                gate.tensor->native_type()==TensorType::FP32&&gate.tensor->cols()==4&&
                gate.tensor->rows()>=1&&gate.tensor->size_bytes()>=4*sizeof(float),
                "Full HC gate/injection reuse lost its exact original native allocation/range");
        }
        validatePhysicalIntervals(banks);
        const auto bank=[&](const std::string& name){return workspace_->getBuffer(name);};
        const qwen4exp::BranchRMSNormArgs norm{.extent={1,normRows()},.hidden=context_.hidden,.epsilon=context_.epsilon,
            .input=static_cast<const float*>(buffers_.trunk.tensor->gpu_data_ptr()),
            .gain=static_cast<const float*>(norm_.gain.tensor->gpu_data_ptr()),
            .output=static_cast<float*>(buffers_.normalized.tensor->gpu_data_ptr())};
        const GPUHCDownOutputView publication{upRows(),context_.lowRank,
            output_demand_.contains(FullNativeHCFFNOutputRole::HCActivated)?static_cast<float*>(buffers_.activatedDown.tensor->gpu_data_ptr()):nullptr,
            static_cast<signed char*>(bank(kGPUProjectionPrimaryQ8)),static_cast<signed char*>(bank(kGPUProjectionResidualQ8)),
            static_cast<float*>(bank(kGPUProjectionPrimaryQ8Scales)),static_cast<float*>(bank(kGPUProjectionResidualQ8Scales))};
        auto* codes=static_cast<std::int8_t*>(bank(GemmWorkspaceBuffers::QUANT_A));
        auto* scales=static_cast<float*>(bank(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
        auto* raw_down=static_cast<float*>(buffers_.down.tensor->gpu_data_ptr());
        auto* mixed=static_cast<float*>(buffers_.mixed.tensor->gpu_data_ptr());
        auto* logits=output_demand_.contains(FullNativeHCFFNOutputRole::HCUp)
            ? static_cast<float*>(buffers_.readLogits.tensor->gpu_data_ptr()) : nullptr;
        const auto generation=workspace_->id();
        auto* backend=getBackendFor(device());const auto runtime=backend?backend->deviceRuntimeGeneration(device().ordinal):0;
        check(runtime!=0,"Persistent HC has no original native runtime generation");
        std::vector<CountSeal> counts;
        // Qwen4ExpNativeIntInput deliberately carries a raw device control span,
        // not a Tensor facade. This cold infrastructure boundary borrows the
        // SAME existing backend allocation record used by TransferEngine leases.
        for(const auto* count:std::array{normRows(),upRows().countOwner(),projectionRows().countOwner()}) {
            if(!count||std::any_of(counts.begin(),counts.end(),[&](const auto& item){return item.pointer==count;}))continue;
            counts.push_back({count,backend->retainNativeAllocation(count,sizeof(std::int32_t),device().ordinal)});
        }
        if (device().is_cuda())
        {
#ifdef HAVE_CUDA
            cuda_args_=bindCudaPrefix(norm,publication,codes,scales,raw_down,mixed,logits);
#endif
        }
        else
        {
#ifdef HAVE_ROCM
            rocm::PersistentHCReadArgs args{
                .norm=norm,.down_weights={},.input_codes=codes,.input_scales=scales,
                .down_partials=static_cast<float*>(bank(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL)),
                .raw_down=raw_down,.publication=publication,
                .up={{},publication.rows,Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
                    publication.activated,logits,publication.primary,publication.residual,
                    publication.primaryScales,publication.residualScales},
                .mixed=mixed,
                .injection={nullptr,inject_?static_cast<float*>(injection_->tensor->gpu_data_ptr()):nullptr,projectionRows(),8},
                .partitions=0,.grid_policy=context_.persistentGrid==Qwen4ExpHCPersistentGrid::OnePerProcessor?
                    rocm::PersistentHCGridPolicy::OnePerProcessor:rocm::PersistentHCGridPolicy::MaximumResident};
            rocm::resolvePersistentHCReadSources(args,engine(down_),engine(up_),inject_?&engine(*inject_):nullptr);
            args_=args;
            barrier_state_=context_.readExecution==Qwen4ExpHCReadExecution::PersistentOriginalPhases
                ? static_cast<std::uint32_t*>(bank(GemmWorkspaceBuffers::ROCM_PERSISTENT_HC_BARRIER))
                : nullptr;
#endif
        }
        counts_=std::move(counts);
        banks_=std::move(banks);workspace_id_=generation;runtime_generation_=runtime;backend_=backend;
        same_prepared_store_=(up_.prepared_store==down_.prepared_store && (!inject_ || inject_->prepared_store==down_.prepared_store));
        validated_store_epoch_=0;
        setGPUStream(stream);prepared_stream_=stream;
        validateStorage(stream);
        if(same_prepared_store_)validated_store_epoch_=down_.prepared_store->mutationEpoch();
        return true;
    }
    /** @brief Return whether this standalone ROCm persistent read can absorb an immediately preceding write. */
    bool canAbsorbPreviousWrite(const Qwen4ExpPrimitiveStage::ResidualWrite& write) const noexcept {
        return device().is_rocm() && !ctaOwned() && !previous_write_.has_value() && !bound() &&
            write.hidden == context_.hidden &&
            write.output.tensor == buffers_.trunk.tensor &&
            write.output.buffer == buffers_.trunk.buffer &&
            write.output.byteOffset == 0;
    }
    /** @brief Build a cold replacement stage that fuses @p write into this persistent HC read. */
    std::unique_ptr<PersistentHCReadStage> cloneWithPreviousWrite(
        const Qwen4ExpPrimitiveStage::ResidualWrite& write) const {
        return std::make_unique<PersistentHCReadStage>(
            context_, norm_, down_, up_, inject_, buffers_, injection_, terminal_,
            workspace_requirements_, write, Qwen4ExpHCTrunkSource::PreviousWrite);
    }
    /** @brief Submit only the sealed native program and publish every original public output. @param ctx Exact worker. */
    bool execute(IDeviceContext* ctx) override {
#ifdef HAVE_ROCM
        if(!ctx||ctx->deviceId()!=device())throw std::logic_error("Persistent HC has a foreign worker");
        check(device().is_rocm() && prepared(),"CUDA or unprepared HC requires its enclosing resident program");
        const auto execution=gpuExecution();validateStorage(execution.nativeStream());
        for(const auto& input:phaseInputs())execution.requirePreparedInput(input.tensor);
        if(!ctaOwned())execution.requirePreparedInput(const_cast<TensorBase*>(norm_.gain.tensor));
        for(const auto& output:phaseOutputs())execution.requirePreparedOutput(output.tensor);
        rocm::PersistentHCWriteInput write_input{};
        if(previous_write_) {
            write_input={
                static_cast<const float*>(previous_write_->residual.tensor->gpu_data_ptr()),
                static_cast<const float*>(previous_write_->block.tensor->gpu_data_ptr()),
                static_cast<const float*>(previous_write_->gateLogits.tensor->gpu_data_ptr()),
                static_cast<float*>(previous_write_->output.tensor->gpu_data_ptr())};
        }
        if(ctaOwned())rocm::launchCTAOwnedHCRead(*ordinary_,*args_,execution.nativeStream(),workspace_id_);
        else rocm::launchPersistentHCRead(*admission_,*args_,barrier_state_,execution.nativeStream(),workspace_id_,write_input);
        for(const auto& output:phaseOutputs())execution.publish(output.tensor);
        return true;
#else
        (void)ctx;
        throw std::logic_error("CUDA HC bindings require their enclosing resident program");
#endif
    }
    /** @return Complete declaration identity, stable across cold physical materialization. */
    std::uint64_t graphCaptureVariantSignature() const override {
        if(bound())validateStorage(prepared_stream_);
        else validateDeclaredRoots(true);
        if(cached_signature_!=0)return cached_signature_;
        auto hash=kFNV1a64OffsetBasis;const auto mix=[&](std::uint64_t value){hash=(hash^value)*kFNV1a64Prime;};
        mix(bool(inject_));mix(bool(previous_write_));mix(output_demand_.bits());mix(static_cast<unsigned>(trunk_source_));
        if(previous_write_)for(const auto& v:std::array{previous_write_->residual,previous_write_->block,previous_write_->gateLogits,previous_write_->output}) {mix(reinterpret_cast<std::uintptr_t>(v.tensor));mix(static_cast<unsigned>(v.buffer));mix(v.byteOffset);}
        mix(static_cast<unsigned>(context_.readExecution));mix(context_.hidden);mix(context_.lowRank);
        if(ctaOwned()){mix(1024);mix(48);mix(context_.lowRank/32);}
        mix(std::bit_cast<std::uint32_t>(context_.epsilon));mix(reinterpret_cast<std::uintptr_t>(normRows()));
        const auto row=[&](DeviceRowRange rows){const auto identity=rows.captureIdentity();
            mix(reinterpret_cast<std::uintptr_t>(identity.count));mix(identity.count_capacity);mix(identity.first_physical_row);
            mix(identity.physical_rows);mix(identity.repeats_per_row);mix(identity.schema_version);};
        row(upRows());row(projectionRows());mix(static_cast<unsigned>(context_.persistentGrid));
        mix(static_cast<unsigned>(context_.downPublication));mix(static_cast<unsigned>(context_.downProducerProgram));
        mix(static_cast<unsigned>(context_.branchNormExecution));mix(device().ordinal);
        for(const auto binding:publicRoots())mix(static_cast<unsigned>(binding.buffer));
        for(const auto& root:roots_){mix(reinterpret_cast<std::uintptr_t>(root.tensor));mix(root.bytes);mix(static_cast<unsigned>(root.type));for(auto size:root.shape)mix(size);}
        for(const auto* p:projections()){mix(reinterpret_cast<std::uintptr_t>(p->prepared_store));mix(p->prepared_ref->binding_id);mix(reinterpret_cast<std::uintptr_t>(p->B));}
        for(const auto& source:sources_){mix(reinterpret_cast<std::uintptr_t>(source.tensor));mix(source.bytes);mix(static_cast<unsigned>(source.type));for(auto size:source.shape)mix(size);}
        cached_signature_=hash?hash:1;
        return cached_signature_;
    }
    /** @brief Retire the program before physical leases and workspace borrows. */
    void invalidateKernelDynamicState() override {
        admission_.reset();ordinary_.reset();args_.reset();barrier_state_=nullptr;
#ifdef HAVE_CUDA
        cuda_args_.reset();
#endif
        validated_store_epoch_=0;
        counts_.clear();banks_.clear();for(auto& root:roots_)root.lease.reset();
    }
    /** @return Inputs remain unchanged; the exact submitted stream authors public outputs. */
    CoherencePolicy coherencePolicy() const override {return CoherencePolicy::OUTPUT;}
protected:
    /**
     * @return Tensor-backed descriptors for the retained original publications.
     *
     * This is diagnostic metadata only: no live count or GPU payload is read.
     * An exactly aliased raw-down view has already been overwritten by the
     * original activation publication, so only that final value is described.
     */
    StageDumpInfo buildDumpInfoImpl() const override {
        StageDumpInfo dump;
        if(ctaOwned())dump.addInput("normalized",buffers_.normalized.tensor,1,4*context_.hidden);
        else {dump.addInput("trunk",buffers_.trunk.tensor,1,4*context_.hidden);
            dump.addOutput("normalized",buffers_.normalized.tensor,1,4*context_.hidden);}
        if(buffers_.down.tensor!=buffers_.activatedDown.tensor)
            dump.addOutput("raw_down",buffers_.down.tensor,1,context_.lowRank);
        dump.addOutput("activated_down",buffers_.activatedDown.tensor,1,context_.lowRank);
        dump.addOutput("read_logits",buffers_.readLogits.tensor,1,4*context_.hidden);
        dump.addOutput("mixed",buffers_.mixed.tensor,1,context_.hidden);
        if(!ctaOwned()){if(inject_)dump.addOutput("injection_logits",injection_->tensor,1,4);
            dump.addWeight("norm_gain",norm_.gain.tensor,1,4*context_.hidden,"FP32");}
        dump.addWeight("down_weights",down_.B);
        dump.addWeight("up_weights",up_.B);
        if(!ctaOwned()&&inject_)dump.addWeight("injection_weights",inject_->B);
        dump.addScalarInt("admitted_rows",1);
        dump.addScalarBool("device_counted_rows",context_.rows.liveRows.has_value());
        return dump;
    }
private:
    /** @return Whether an enclosing program already owns the phase's physical bindings. */
    bool bound() const noexcept
    {
#ifdef HAVE_CUDA
        if (cuda_args_) return true;
#endif
        return args_.has_value();
    }
#ifdef HAVE_CUDA
    /**
     * @brief Seal the original CUDA HC projections and serial reduction before capture.
     * @param norm Existing branch normalization arguments and row owner.
     * @param publication Original activated output and both Q8 terms.
     * @param codes Existing single-term Down activation bytes.
     * @param scales Existing single-term Down activation scales.
     * @param raw_down Original raw Down output.
     * @param mixed Original public HC read output.
     * @param logits Original Up output when observed.
     * @return A view of the same prepared sources and named arena banks.
     */
    cuda::moe::RetainedHCPrefix bindCudaPrefix(const qwen4exp::BranchRMSNormArgs& norm,
        const GPUHCDownOutputView& publication, std::int8_t* codes, float* scales,
        float* raw_down, float* mixed, float* logits) const
    {
        DeviceNativeQuantisedMatrixDesc down{}, up{};
        check(engine(down_).inspectPreparedNativeQuantisedMatrixDesc(down) &&
            engine(up_).inspectPreparedNativeQuantisedMatrixDesc(up),
            "CUDA retained HC requires its originally prepared native projections");
        (void)requireQ8ResidualNativeMatrix(down,1,context_.lowRank,4*context_.hidden);
        (void)requireQ8ResidualNativeMatrix(up,1,4*context_.hidden,context_.lowRank);
        int ordered = 0, partitions = 0;
        const auto& hardware = cudaHardwareMetadata().at(device().ordinal);
        check(cudaNativeQuantisedGemvTuned_queryCanonicalM1Schedule(down.source_codebook_id,
                down.n, down.k, hardware.sm_count, &ordered, &partitions) &&
                partitions > 0 && partitions <= down.k / 32 && (ordered || partitions == 1),
            "CUDA retained HC has no original serial Down schedule");
        auto* partials = ordered ? static_cast<float*>(workspace_->getBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS)) : nullptr;
        if (ordered)
            check(partials && workspace_->getBufferSize(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS) >=
                static_cast<std::size_t>(partitions) * down.n * sizeof(float),
                "CUDA retained HC requires its original serial GEMV partition bank");
        check(codes && scales && raw_down && mixed && publication.primary && publication.residual &&
            publication.primaryScales && publication.residualScales &&
            publication.rows.countOwner() == norm.extent.liveRows,
            "CUDA retained HC requires matching row authority and original activation banks");
        const float* injection_weights = nullptr;
        float* injection_output = nullptr;
        if (inject_)
        {
            ContiguousFloatingPointWeightDescriptor injection{};
            check(engine(*inject_).exportContiguousFloatingPointWeights(injection) &&
                  injection.valid() && injection.type == TensorType::FP32 &&
                  injection.n == 4 && injection.k == 4 * context_.hidden,
                "CUDA retained HC injection requires original contiguous FP32 [4, 4H] weights");
            injection_weights = static_cast<const float*>(injection.data);
            injection_output = static_cast<float*>(injection_->tensor->gpu_data_ptr());
        }
        qwen4exp::ResidualWriteArgs write_args{.extent=norm.extent,.hidden=norm.hidden};
        if (previous_write_)
        {
            const auto& w = *previous_write_;
            write_args = {norm.extent, norm.hidden,
                static_cast<const float*>(w.residual.tensor->gpu_data_ptr()),
                static_cast<const float*>(w.block.tensor->gpu_data_ptr()),
                static_cast<const float*>(w.gateLogits.tensor->gpu_data_ptr()),
                static_cast<float*>(w.output.tensor->gpu_data_ptr())};
        }
        return {
            .write=write_args,
            .norm=norm,.down=down,.down_partitions={partitions,(down.k/32+partitions-1)/partitions},
            .down_fold=ordered ? cuda::moe::HCDownFold::OrderedPartitions : cuda::moe::HCDownFold::Direct,
            .norm_codes=codes,.norm_scales=scales,.down_partials=partials,.raw_down=raw_down,
            .down_publication=publication,
            .up={up,publication.rows,Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal,raw_down,logits,
                publication.primary,publication.residual,publication.primaryScales,publication.residualScales},
            .mixed=mixed,
            .injection_weights=injection_weights,
            .injection_output=injection_output,
            .has_previous_write=previous_write_.has_value()};
    }
#endif
    /** @return Whether this declaration retains original norm/injection graph nodes. */
    bool ctaOwned() const {return context_.readExecution==Qwen4ExpHCReadExecution::CTAOwnedOriginalTail;}
    /** @return Exactly one admitted native implementation, never an automatic alternative. */
    bool prepared() const {return ctaOwned()?ordinary_.has_value():admission_.has_value();}
    /** @brief Reject invalid declarations without inspecting device payloads. @param value Required condition. @param message Precise failure. */
    static void check(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
    /** @return Original public views in stable ownership order, including the input trunk. */
    std::vector<Qwen4ExpFloatBinding> publicRoots() const {
        std::vector<Qwen4ExpFloatBinding> result{buffers_.trunk,buffers_.normalized,buffers_.down,buffers_.activatedDown,buffers_.readLogits,buffers_.mixed};
        if(inject_)result.push_back(*injection_);return result;
    }
    /** @return Original numerical storage plus demanded optional publications; exact down aliases remain shared. */
    std::vector<Qwen4ExpFloatBinding> outputs() const {
        std::vector<Qwen4ExpFloatBinding> result{buffers_.normalized,buffers_.down};
        if(output_demand_.contains(FullNativeHCFFNOutputRole::HCActivated))result.push_back(buffers_.activatedDown);
        if(output_demand_.contains(FullNativeHCFFNOutputRole::HCUp))result.push_back(buffers_.readLogits);
        result.push_back(buffers_.mixed);
        if(inject_)result.push_back(*injection_);return result;
    }
    /** @brief Visit each distinct original public output exactly once, including its exact down alias.
     * @tparam Function Nonallocating declaration/publication callback. @param function Receives the canonical original view.
     */
    template<class Function> void forEachOutput(Function&& function) const {
        const auto views=outputs();
        for(std::size_t i=0;i<views.size();++i) {
            if(ctaOwned()&&(views[i].tensor==buffers_.normalized.tensor||
                (injection_&&views[i].tensor==injection_->tensor)))continue;
            if(std::any_of(views.begin(),views.begin()+i,[&](const auto& prior){return prior.tensor==views[i].tensor;}))continue;
            function(views[i]);
        }
    }
    /** @brief Distinguish omitted copies from still-required aliased raw storage.
     * @param tensor Original tensor owner. @return Whether this exact full/standalone operation uses its storage.
     */
    bool requiresStorage(const TensorBase* tensor) const {
        if(tensor==buffers_.readLogits.tensor)return output_demand_.contains(FullNativeHCFFNOutputRole::HCUp);
        if(tensor==buffers_.activatedDown.tensor&&tensor!=buffers_.down.tensor)
            return output_demand_.contains(FullNativeHCFFNOutputRole::HCActivated);
        return true;
    }
    /** @return The original native live scalar; no host read of its value occurs. */
    const int32_t* normRows() const {
        if(!context_.rows.liveRows)return nullptr;
        const auto* native=std::get_if<Qwen4ExpNativeIntInput>(&context_.rows.liveRows->value);
        check(native&&native->owner==device()&&native->data&&native->capacity==1,"Persistent HC requires its exact native M1 norm count");return native->data;
    }
    /** @return Original down/injection projection row owner, independently declared. */
    DeviceRowRange projectionRows() const {return context_.projectionRows.value_or(DeviceRowRange::fullyActive(1));}
    /** @return Original two-term publication/up query owner, never inferred from a live count value. */
    DeviceRowRange upRows() const {return context_.upProjectionRows.value_or(projectionRows());}
    /** @return Only original present projection declarations, preserving execution/weight order. */
    std::vector<const GEMMStage::Params*> projections() const {
        std::vector<const GEMMStage::Params*> result{&down_,&up_};
        if(inject_)result.push_back(&*inject_);return result;
    }
    /** @return Exact still-owned immutable prepared engine. @param projection Original source reference. */
    ITensorGemm& engine(const GEMMStage::Params& projection) const {
        check(projection.prepared_store->contains(*projection.prepared_ref),"Persistent HC prepared source retired");
        auto* result=projection.prepared_store->gemmKernel(*projection.prepared_ref);
        check(result!=nullptr,"Persistent HC source has no original prepared engine");return *result;
    }
    /** @brief Reject same-byte shape/type/view substitution before signature queries and launch. */
    void validateDeclaredRoots(bool check_store = true) const {
        for(const auto& root:roots_)check(root.tensor&&root.tensor->shape()==root.shape&&root.tensor->size_bytes()==root.bytes&&root.tensor->native_type()==root.type,
            "Persistent HC original tensor declaration changed");
        if(check_store) {
            std::size_t source=0;
            for(const auto* p:projections()) {
                const auto& saved=sources_[source++];
                check(p->B==saved.tensor&&p->B->shape()==saved.shape&&p->B->size_bytes()==saved.bytes&&
                    p->B->native_type()==saved.type && &engine(*p)==saved.engine,
                    "Persistent HC original prepared source declaration or owner changed");
            }
        }
        for(std::size_t i=0;i<roots_.size();++i)for(std::size_t j=0;j<i;++j)
            if(roots_[i].tensor==roots_[j].tensor)
                check(i==3&&j==2,"Persistent HC has an undeclared public tensor alias");
    }
    /** @brief Prove existing public roots/banks cannot overwrite another live allocation. @param banks Original canonical named leases. */
    void validatePhysicalIntervals(const std::vector<BankSeal>& banks) const {
        const auto overlaps=[](const void* a,std::size_t an,const void* b,std::size_t bn){
            const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
            check(an<=std::numeric_limits<std::uintptr_t>::max()-x&&bn<=std::numeric_limits<std::uintptr_t>::max()-y,"Persistent HC physical interval overflows");
            return an&&bn&&x<y+bn&&y<x+an;
        };
        for(std::size_t i=0;i<roots_.size();++i)for(std::size_t j=0;j<i;++j)
            if(requiresStorage(roots_[i].tensor)&&requiresStorage(roots_[j].tensor)&&
                overlaps(roots_[i].pointer,roots_[i].bytes,roots_[j].pointer,roots_[j].bytes))
                check((i==3&&j==2)&&roots_[i].pointer==roots_[j].pointer&&roots_[i].bytes==roots_[j].bytes,
                    "Persistent HC public roots overlap outside the exact original down alias");
        for(const auto* count:std::array{normRows(),upRows().countOwner(),projectionRows().countOwner()})if(count) {
            for(const auto& root:roots_)if(requiresStorage(root.tensor))
                check(!overlaps(count,sizeof(std::int32_t),root.pointer,root.bytes),"Persistent HC row control aliases a public value");
            for(const auto& bank:banks)check(!overlaps(count,sizeof(std::int32_t),bank.pointer,bank.bytes),"Persistent HC row control aliases phase scratch");
        }
        for(std::size_t i=0;i<banks.size();++i){
            for(const auto& root:roots_)if(requiresStorage(root.tensor))
                check(!overlaps(banks[i].pointer,banks[i].bytes,root.pointer,root.bytes),"Persistent HC scratch aliases a public root");
            for(std::size_t j=0;j<i;++j)
                if(overlaps(banks[i].pointer,banks[i].bytes,banks[j].pointer,banks[j].bytes))
                    throw std::logic_error("Persistent HC named scratch banks overlap: "+banks[i].name+" and "+banks[j].name);
        }
    }
    /** @brief Validate exact original storage/runtime/workspace/source identity without device queries or data transfers. @param stream Original prepared stream. */
    void validateStorage(void* stream) const {
        const bool store_unchanged = same_prepared_store_ && validated_store_epoch_ != 0 &&
            down_.prepared_store->mutationEpoch() == validated_store_epoch_;
        validateDeclaredRoots(!store_unchanged);
        if(!bound()||!workspace_||workspace_->id()!=workspace_id_||!workspace_->isAllocated()||
            stream!=prepared_stream_||requireGPUStream()!=prepared_stream_||reinterpret_cast<std::uintptr_t>(stream)<=2||getBackendFor(device())!=backend_||
            backend_->deviceRuntimeGeneration(device().ordinal)!=runtime_generation_)
            throw std::logic_error("Persistent HC lost prepared workspace/runtime/stream identity");
#ifdef HAVE_ROCM
        if(ordinary_)ordinary_->requireBinding(device(),*args_,workspace_id_);
#endif
        for(const auto& root:roots_){
            if(!requiresStorage(root.tensor)) {
                check(!root.pointer&&!root.lease,"Omitted HC materialization retained an unexpected storage lease");
                continue;
            }
            if(root.tensor->gpu_data_ptr()!=root.pointer)throw std::logic_error("Persistent HC public pointer changed");
            TransferEngine{}.requireCapturedDeviceStorage(*root.lease,*root.tensor,device(),root.bytes);
        }
        for(const auto& bank:banks_)if(!bank.lease||bank.lease->data()!=bank.pointer||bank.lease->sizeBytes()!=bank.bytes)
            throw std::logic_error("Persistent HC named scratch binding changed");
        // Each retained NativeAllocationCaptureLease pins its allocator record
        // under the backend lifecycle lock; re-calling retainNativeAllocation()
        // on every replay repeats a contended global registry scan per layer.
        for(const auto& count:counts_)if(!count.lease.valid())
            throw std::logic_error("Persistent HC original row control allocation changed");
        if(store_unchanged)return;
        DeviceNativeQuantisedMatrixDesc down{},up{};
        if(!engine(down_).inspectPreparedNativeQuantisedMatrixDesc(down)||!engine(up_).inspectPreparedNativeQuantisedMatrixDesc(up))
            throw std::logic_error("Persistent HC original prepared descriptors changed");
        const auto descriptorsMatch = [&](const auto& originalDown,const auto& originalUp) {
            return std::memcmp(&down,&originalDown,sizeof(down)) == 0 && std::memcmp(&up,&originalUp,sizeof(up)) == 0;
        };
#ifdef HAVE_CUDA
        if (cuda_args_) {
            check(descriptorsMatch(cuda_args_->down,cuda_args_->up.weights),"Persistent CUDA HC original prepared descriptors changed");
            if (inject_) {
                ContiguousFloatingPointWeightDescriptor injection{};
                if (!engine(*inject_).exportContiguousFloatingPointWeights(injection) || !injection.valid() ||
                    injection.type != TensorType::FP32 || injection.n != 4 || injection.k != 4 * context_.hidden ||
                    injection.data != cuda_args_->injection_weights)
                    throw std::logic_error("Persistent CUDA HC original injection source changed");
            }
            if(same_prepared_store_)validated_store_epoch_=down_.prepared_store->mutationEpoch();
            return;
        }
#endif
        check(descriptorsMatch(args_->down_weights,args_->up.weights),"Persistent ROCm HC original prepared descriptors changed");
        if(inject_) {
            ContiguousFloatingPointWeightDescriptor injection{};
            if(!engine(*inject_).exportContiguousFloatingPointWeights(injection)||!injection.valid()||
                injection.type!=TensorType::FP32||injection.n!=4||injection.k!=4*context_.hidden||injection.data!=args_->injection.weights)
                throw std::logic_error("Persistent HC original injection source changed");
        }
        if(same_prepared_store_)validated_store_epoch_=down_.prepared_store->mutationEpoch();
    }
    const Qwen4ExpHyperconnectionContext context_;
    const Qwen4ExpPrimitiveStage::BranchRMSNorm norm_;
    const GEMMStage::Params down_,up_;
    const std::optional<GEMMStage::Params> inject_; ///< Absent only for an original final read, never a dummy projection.
    const Qwen4ExpHyperconnectionReadBuffers buffers_;
    const std::optional<Qwen4ExpFloatBinding> injection_; ///< Present exactly with the original injection projection.
    const std::string terminal_;
    const std::optional<Qwen4ExpPrimitiveStage::ResidualWrite> previous_write_; ///< Original full-prefix write declaration, never a second owner.
    Qwen4ExpHCTrunkSource trunk_source_; ///< Exact original owner of the HC norm input.
    const WorkspaceRequirements workspace_requirements_; ///< Original named-bank order determines the same exact packed PMA extent.
    FullNativeHCFFNOutputDemand output_demand_=FullNativeHCFFNOutputDemand::all(); ///< Frozen before storage/native preparation.
    std::vector<TensorSeal> roots_;
    std::vector<SourceSeal> sources_;
    std::vector<CountSeal> counts_;
    std::vector<BankSeal> banks_;
    DeviceWorkspaceManager* workspace_=nullptr;
#ifdef HAVE_CUDA
    std::optional<cuda::moe::RetainedHCPrefix> cuda_args_;
#endif
    std::optional<rocm::PersistentHCReadArgs> args_;
    std::uint32_t* barrier_state_=nullptr;
    std::optional<NativeCooperativeKernelAdmission> admission_;
    std::optional<rocm::CTAOwnedHCReadProgram> ordinary_;
    std::uint64_t workspace_id_=0,runtime_generation_=0;
    bool same_prepared_store_=false;
    mutable std::uint64_t validated_store_epoch_=0;
    mutable std::uint64_t cached_signature_=0;
    IBackend* backend_=nullptr;
    void* prepared_stream_=nullptr;
};
