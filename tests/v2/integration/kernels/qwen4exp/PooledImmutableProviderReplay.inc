/**
 * @file PooledImmutableProviderReplay.inc
 * @brief Shared actual local pooled/provider fixture for correctness and Release timing.
 *
 * Included inside the original fixture namespace after its original production
 * headers. Capture, kernels, service, oracle and retirement are shared literally;
 * measurement only splits diagnostic preparation/observation from execution.
 * A grouped program may retain the same original provider/service across exact
 * participant-local roots. Complete-group admission remains parent-owned.
 */
/** @brief Terminal diagnostics copied only after complete native result retirement. */
struct PooledNgramReplayObservation {
    std::uint64_t live_ids=0,unique_ids=0,gpu_hits=0,source_rows=0,promotions=0,evictions=0;
    std::uint64_t source_native_bytes=0,hit_wire_bytes=0;
};
#include "PooledImmutableProviderFile.inc"
/** @brief Explicit recorded setup fault, never a production selection or retry policy. */
enum class OwnerAdmission {Original,RejectForeignAuthority};
/** @brief Immutable diagnostic capture edit; never a runtime serving selector. */
enum class ACKInterleaving {None,Cancel900BeforeReady,OriginalUnchecked900};
/** @brief Recorded activation lifecycle; both modes execute the same production stages and native code. */
enum class ActivationPreparation {BeforeGraph,ColdGraph};
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
inline constexpr bool ProviderReplayRawDiagnostic=true;
/** @brief Explicit raw actual-program schedule; it never selects ordinary qualified execution. */
enum class ProviderReplayDiagnosticProgram {OrdinaryLookup,ReadAheadLookup,AuthoredRemoteLookup};
#else
inline constexpr bool ProviderReplayRawDiagnostic=false;
#endif
#ifdef LLAMINAR_POOLED_IDENTITY_REJECTION_TESTS
namespace identity_fixture=pooled_source_fixture;
/** @brief Closed captured rejection purpose, independent of ordinary source/timing programs. */
struct IdentityRejectionCase final {
    n::OwnerAction action;
    identity_fixture::IdentityFault fault;
    std::uint32_t position;
};
/** @brief Test-only invalid input producer immediately before the real selected native phase. */
class IdentityFaultStage final:public IComputeStage {
public:
    /** @brief Retain only original declared roots; no execution or alternate validation is performed.
     * @param owner Original prepared physical owner.
     * @param coordinator Original prepared source coordinator.
     * @param fault Closed fault and original live-prefix coordinate.
     */
    IdentityFaultStage(std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner,
        std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator,IdentityRejectionCase fault)
        :IComputeStage(owner->plan().parameters().device),owner_(std::move(owner)),coordinator_(std::move(coordinator)),fault_(fault){}
    /** @brief Seal the same original native addresses outside capture.
     * @param context Original participant context.
     * @param stream Exact original capture stream.
     * @return True after immutable original binding.
     */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override{
        requireProvider(context && context->deviceId()==device() && stream && reinterpret_cast<std::uintptr_t>(stream)>2,"Identity fault requires original participant/stream");
        setGPUStream(stream);return true;
    }
    /** @brief Author only the selected invalid input; the next stage is the original production operation.
     * @param context Original participant context.
     * @return True after exact original-stream submission.
     */
    bool execute(IDeviceContext* context)override{
        requireProvider(context && context->deviceId()==device(),"Identity fault has foreign context");
        auto execution=gpuExecution();
        const auto own=owner_->roots(),coord=coordinator_->roots();
        execution.requirePreparedInput(own.frame.get());execution.requirePreparedInput(coord.native.get());
        auto* changed=fault_.action==n::OwnerAction::Pin?coord.native.get():own.frame.get();
        execution.requirePreparedOutput(changed);
        const bool changesSlot=fault_.fault==identity_fixture::IdentityFault::PinCommitFlags;
        if(changesSlot){execution.requirePreparedInput(own.cache.get());execution.requirePreparedOutput(own.cache.get());}
        identity_fixture::Args args{owner_->plan().parameters(),coordinator_->plan().parameters(),
            changesSlot?static_cast<std::byte*>(own.cache->gpu_data_ptr()):nullptr,
            static_cast<std::byte*>(own.frame->gpu_data_ptr()),nullptr,nullptr,static_cast<std::byte*>(coord.native->gpu_data_ptr()),
            nullptr,nullptr,nullptr,nullptr,{},0,0,fault_.fault,fault_.position,fault_.action!=n::OwnerAction::Pin};
#ifdef HAVE_CUDA
        if(device().is_cuda())identity_fixture::launchCUDA(identity_fixture::Action::CorruptIdentity,args,execution.nativeStream());
#endif
#ifdef HAVE_ROCM
        if(device().is_rocm())identity_fixture::launchROCm(identity_fixture::Action::CorruptIdentity,args,execution.nativeStream());
#endif
        execution.publish(changed);
        if(changesSlot)execution.publish(own.cache.get());
        return true;
    }
    /** @return Only original preallocated root reads/writes. */
    StageBufferContract bufferContract()const override{
        auto contract=StageBufferContract::build();
        if(fault_.action==n::OwnerAction::Pin)contract.addInput(BufferId::QFN_POOLED_OWNER_FRAME,"INT8").addPreallocatedInOut(BufferId::QFN_POOLED_COORDINATOR_NATIVE,"INT8");
        else contract.addInput(BufferId::QFN_POOLED_COORDINATOR_NATIVE,"INT8").addPreallocatedInOut(BufferId::QFN_POOLED_OWNER_FRAME,"INT8");
        if(fault_.fault==identity_fixture::IdentityFault::PinCommitFlags)
            contract.addPreallocatedInOut(BufferId::QFN_POOLED_OWNER_CACHE,"INT8");
        return contract;
    }
    /** @return Complete immutable original address/geometry/fault identity. */
    std::uint64_t graphCaptureVariantSignature()const override{
        std::uint64_t h=owner_->captureIdentity()^coordinator_->captureIdentity();
        for(auto v:{static_cast<unsigned>(fault_.action),static_cast<unsigned>(fault_.fault),fault_.position})h=(h^v)*1099511628211ULL;
        return h?h:1;
    }
    /** @return Capture-only declared invalid producer, never serving code. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Original participant-local native semantic kind. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_NATIVE_PHASE;}
    /** @param backend Candidate actual execution backend.
     * @return Only this original backend.
     */
    bool supportsBackend(ComputeBackendType backend)const override{return (device().is_cuda()&&backend==ComputeBackendType::GPU_CUDA)||(device().is_rocm()&&backend==ComputeBackendType::GPU_ROCM);}
    /** @return Every inactive original coordinate stays untouched on partial/empty replay. */
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    /** @return No new allocation or workspace. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Static purpose only, no live device observation. */
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
private:
    std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner_;
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator_;
    IdentityRejectionCase fault_;
};
#endif
/** @brief Complete immutable actual provider case; original byte authoring survives all captures. */
struct ProviderReplaySpec final {
    std::shared_ptr<ProviderFile> file;
    std::uint32_t maximum,slots,hostSlots;
    n::OriginalIdWidth width;
    bool tokenQueries=false;
    OwnerAdmission ownership=OwnerAdmission::Original;
    ACKInterleaving interleaving=ACKInterleaving::None;
    ActivationPreparation activationPreparation=ActivationPreparation::BeforeGraph;
    std::uint32_t uniqueCapacity=16;
    bool measurement=false;
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    ProviderReplayDiagnosticProgram diagnosticProgram=ProviderReplayDiagnosticProgram::OrdinaryLookup;
#endif
#ifdef LLAMINAR_POOLED_IDENTITY_REJECTION_TESTS
    std::optional<IdentityRejectionCase> identityRejection;
#endif
    /** @brief Validate complete immutable case syntax before selected source/graph creation. */
    void requireComplete()const{
        requireProvider(file && maximum>=16 && uniqueCapacity && uniqueCapacity<=maximum && uniqueCapacity<=file->rows && slots<=file->rows && hostSlots<=file->rows,
            "Provider program has incomplete original geometry");
        requireProvider(width==n::OriginalIdWidth::UInt32 || width==n::OriginalIdWidth::UInt64,"Provider program has unknown original ID width");
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        requireProvider(diagnosticProgram==ProviderReplayDiagnosticProgram::OrdinaryLookup ||
            (diagnosticProgram==ProviderReplayDiagnosticProgram::ReadAheadLookup && tokenQueries &&
             ownership==OwnerAdmission::Original && interleaving==ACKInterleaving::None && activationPreparation==ActivationPreparation::BeforeGraph),
            "Raw read-ahead requires its actual ordinary token-query program; semantic fault edits are separate programs");
#endif
#ifdef LLAMINAR_POOLED_IDENTITY_REJECTION_TESTS
        if(identityRejection){
            const auto role=identityRejection->action;const auto fault=identityRejection->fault;
            const bool pin=role==n::OwnerAction::Pin;
            const bool common=fault==identity_fixture::IdentityFault::DuplicateRow || fault==identity_fixture::IdentityFault::DuplicateUnique;
            requireProvider(!tokenQueries && ownership==OwnerAdmission::Original && interleaving==ACKInterleaving::None &&
                activationPreparation==ActivationPreparation::BeforeGraph && maximum>identityRejection->position && slots>=maximum &&
                (pin || role==n::OwnerAction::AcceptFree || role==n::OwnerAction::AcceptVictim) &&
                (common || (pin?(fault==identity_fixture::IdentityFault::DuplicateSlot || fault==identity_fixture::IdentityFault::PinSlotBounds ||
                    fault==identity_fixture::IdentityFault::PinOrdinal || fault==identity_fixture::IdentityFault::PinCommitFlags):
                    fault==identity_fixture::IdentityFault::ForeignOwner || fault==identity_fixture::IdentityFault::ForeignInterval)),
                "Invalid captured identity rejection declaration");
        }
#endif
        requireProvider(!tokenQueries || (!(maximum%16) && width==n::OriginalIdWidth::UInt64),"Production PLE program requires complete original INT64 head geometry");
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        switch(diagnosticProgram){
        case ProviderReplayDiagnosticProgram::OrdinaryLookup:break;
        case ProviderReplayDiagnosticProgram::ReadAheadLookup:requireProvider(tokenQueries,"Raw read-ahead requires the actual production PLE semantic gather");break;
        default:throw std::invalid_argument("Unknown raw provider program declaration");
        }
#endif
    }
};
/** @brief Actual graphs live outside their original worker; all embedded roots die only after these handles. */
class ProviderReplayGraphs final {
public:
    ProviderReplayGraphs(const ProviderReplayGraphs&)=delete;
    ProviderReplayGraphs& operator=(const ProviderReplayGraphs&)=delete;
    /** @brief Bind every original graph slot before its first original creator, without new native allowances. */
    ProviderReplayGraphs(const PreparedGPUContextUsePermit& original,unsigned count
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        ,ProviderReplayDiagnosticProgram selected=ProviderReplayDiagnosticProgram::OrdinaryLookup,
        std::optional<PooledRemoteGraphProgram> remoteProgram={},std::shared_ptr<PooledRemoteGraphCalibration> receipt={}
#endif
        ):original_(original){
        auto* stream=ExplicitGPUStream(original.worker()->defaultStream()).get();
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(bool(remoteProgram)!=bool(receipt) || (remoteProgram && (count!=1 || selected!=ProviderReplayDiagnosticProgram::OrdinaryLookup)))
            throw std::invalid_argument("Authored remote diagnostic requires exactly one original descriptor/receipt pair");
        if(receipt)receipt->checkpoint("before_remote_create");
#endif
        for(unsigned i=0;i<count;++i){
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
            owners_.push_back(remoteProgram?PooledRemoteGraphCalibration::create(original,*remoteProgram):
                RetainedParticipantGraphExecutableFixtureAccess::rawDiagnostic(original,stream));
#else
            owners_.push_back(RetainedParticipantGraphExecutableFixtureAccess::original(original,stream));
#endif
            captures_.push_back(&RetainedParticipantGraphExecutableFixtureAccess::capture(owners_.back()));
        }
        ordinals_.assign(owners_.size(),0);
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(receipt)receipt->checkpoint("after_remote_create");
        if(selected==ProviderReplayDiagnosticProgram::ReadAheadLookup){
            parent_=PooledNativeProgramDiagnosticAccess::parent(original);
            side_=PooledNativeProgramDiagnosticAccess::side(original,stream);
            side_->bindTerminal(parent_->eventOwner());
            std::cout<<"POOLED_PROVIDER_AUXILIARY "<<nlohmann::json{{"qualified",false},{"same_original_source",true},{"same_pma",true},
                {"side_streams",1},{"side_ordering_events",2},{"separate_parent_terminal_events",1},
                {"family_identity",side_->captureIdentity()},{"primary_stream",reinterpret_cast<std::uintptr_t>(stream)},
                {"side_stream",reinterpret_cast<std::uintptr_t>(side_->stream())},
                {"fork_event",reinterpret_cast<std::uintptr_t>(side_->forkEvent())},{"join_event",reinterpret_cast<std::uintptr_t>(side_->joinEvent())},
                {"parent_terminal_event",reinterpret_cast<std::uintptr_t>(parent_->eventOwner().get())}}.dump()<<'\n';
        }else if(selected!=ProviderReplayDiagnosticProgram::OrdinaryLookup)throw std::invalid_argument("Unknown raw provider native family");
#endif

    }
    /** @return Original slot's capture object; no graph replacement is possible. */
    IGPUGraphCapture* capture(unsigned index)const{return captures_.at(index);}
    /** @return The sole original definition for a checked participant-local cohort, never a raw SDK capture. */
    const std::shared_ptr<RetainedParticipantGraphExecutable>& definition(unsigned index)const{return owners_.at(index);}
    /** @brief Mark the real owner before forwarding exactly the original launch. */
    bool launch(IGPUGraphCapture* graph){for(std::size_t i=0;i<owners_.size();++i)if(captures_[i]==graph){
        if(ordinals_[i]==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Provider graph invocation ordinal exhausted");
        return RetainedParticipantGraphExecutableFixtureAccess::submit(owners_[i],++ordinals_[i]);
    }throw std::logic_error("Provider graph is foreign to its original declared program");}
    /** @brief Query only after original fixture completion; never poll or wait in a captured stage. */
    void retire(){
        for(const auto& owner:owners_)if(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owner))throw std::logic_error("Original provider graph has outstanding native readers");
        owners_.clear();captures_.clear();ordinals_.clear();
    }
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    /** @return Sole original raw lane, absent for the explicitly ordinary diagnostic. */
    const std::shared_ptr<ReadAheadNativeResourceFamily>& side()const noexcept{return side_;}
    /** @brief Begin one original admitted request, allowing all its FIFO lookup invocations. */
    void beginRequest(){if(side_)side_->beginSubmission();}
#endif
    /** @brief Queue the exact marker AFTER the original terminal graph and BEFORE its existing native join. */
    void publishRequestTerminal(){
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(side_)side_->recordTerminal();
#endif
    }
    /** @brief Observe only after that same original join proves the newly enqueued marker complete. */
    void observeRequestTerminal(){
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(side_ && !side_->observeTerminal())throw std::logic_error("Joined actual provider retained an incomplete original side terminal");
#endif
    }
private:
    PreparedGPUContextUsePermit original_;
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    std::shared_ptr<ReadAheadNativeParentTerminal> parent_; ///< Independent original population survives all graph/root readers.
    std::shared_ptr<ReadAheadNativeResourceFamily> side_; ///< Same actual source; destroyed before its retained parent alias.
#endif
    std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>> owners_;
    std::vector<IGPUGraphCapture*> captures_; ///< Immutable construction views owned by the original graph holders.
    std::vector<std::uint64_t> ordinals_; ///< Actual SDK program submission order, never model/controller state.
};
/** @brief Exact serialized payload plan, plus separately retained native group/graph obligations. */
struct ProviderReplayGeometry final {
    ImmutableRowTableLayoutGeometry layout;
    q::PooledImmutablePhysicalOwnerGeometry physical;
    q::PooledImmutableCoordinatorGeometry policy;
    CpuAdaptiveImmutableRowCacheGeometry cpu;
    q::ImmutableLookupLayout channel;
    std::array<CPUTensorStorageDeclaration,4> declarations;
    std::size_t activation;
    unsigned graphCount;
    PhysicalMemoryPlan payload,retained;
    RetainedHostServiceProgram ioProgram; ///< Same original declaration used for both BOM and actual service birth.
    /** @return Same original geometry and arithmetic used by actual materialization, never synthetic capacities. */
    static ProviderReplayGeometry create(const ProviderReplaySpec& spec,DeviceId device,const PreparedNativeLookupFixture& session,const RetainedHostServiceProgram& ioProgram,
        const NativeOrderingEventBudgetProjection& ordering=NativeOrderingEventBudgetProjection::installed()){
        spec.requireComplete();
        const auto& file=*spec.file;const auto source=file.descriptor();const PhysicalMemoryAllocatorIdentity gpuId{0,device},hostId{0,DeviceId::cpu()};
        const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
        auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
            ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
            {{{gpuId,ImmutableRowStorage::VRAM},0,spec.slots,{}},{{hostId,ImmutableRowStorage::RAM},0,spec.hostSlots,{}}},{gpuId},{},
            {{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},spec.slots?std::optional(gpuId):std::nullopt,spec.uniqueCapacity);
        auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{gpuId},gpuId);
        auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},spec.maximum);
        auto cpu=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,gpuId);
        const auto gpu=session.resource(device),host=session.resource(DeviceId::cpu());
        auto channel=q::ImmutableLookupLayout::createSubordinate(spec.uniqueCapacity,source.rowBytes());
        PhysicalMemoryPlanBuilder payload;
        for(const auto& bom:{cpu.bom(host),physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),policy.policyBOM(gpu),policy.mandatoryBOM(gpu),
            q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,0,gpu),q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,gpu),q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,source,gpu)})
            for(const auto& row:bom.resources())payload.add(row);
        payload.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
        const auto io=ioProgram.bom(host);
        for(const auto& row:io.resources())payload.add(row.resource(),PhysicalMemoryOwner::RetainedHostServiceStorage,row.bytes(PhysicalMemoryOwner::RetainedHostServiceStorage));
        const std::array declarations{
            CPUTensorStorageDeclaration(1,std::size_t{spec.maximum}*(spec.width==n::OriginalIdWidth::UInt32?1:2),BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(spec.tokenQueries?spec.maximum/16:1,spec.tokenQueries?file.columns*16:std::size_t{spec.maximum}*file.columns,BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
        const auto activation=std::size_t{spec.maximum}*(spec.width==n::OriginalIdWidth::UInt32?4:8)+sizeof(int)+sampling_math::kDeviceGenerationControlCount*sizeof(int)+std::size_t{spec.maximum}*file.columns*sizeof(float);
        payload.add(gpu,PhysicalMemoryOwner::ActivationArena,activation);
        for(const auto& declaration:declarations)payload.add(host,PhysicalMemoryOwner::ActivationArena,declaration.physicalBytes());
        unsigned graphCount=spec.tokenQueries?4:3;
#ifdef LLAMINAR_POOLED_IDENTITY_REJECTION_TESTS
        if(spec.identityRejection)graphCount=spec.identityRejection->action==n::OwnerAction::Pin?4:5;
#endif
        PhysicalMemoryPlanBuilder retained;
        for(const auto& row:io.resources())retained.add(row.resource(),PhysicalMemoryOwner::RetainedHostServiceRuntime,row.bytes(PhysicalMemoryOwner::RetainedHostServiceRuntime));
        #ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        retained.add(gpu,PhysicalMemoryOwner::NativeGraphExecutable,graphCount*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General));
#else
        const auto general=RetainedParticipantGraphExecutable::generalBOM(host,gpu,graphCount);
        for(const auto& row:general.resources())retained.add(row);
#endif
        for(const auto program:{q::PooledImmutableOwnerNativePlan::orderingProgram(physical,policy,0),q::PooledImmutableCoordinatorNativePlan::orderingProgram(physical,policy),
            q::PooledImmutableSourceGeometry::orderingProgram(physical,policy,source),NativeOrderingEventProgram::lookupSubmissionRetirement()}){
            const auto bom=ordering.bom(program,host,gpu);for(const auto& row:bom.resources())retained.add(row);
        }
        return {std::move(layout),std::move(physical),std::move(policy),std::move(cpu),std::move(channel),declarations,activation,graphCount,payload.build(),retained.build(),ioProgram};
    }
};
/** @brief Immutable program token ties each declared case to its one original session and geometry. */
struct PreparedProviderReplayProgram final {
    ProviderReplaySpec spec;
    ProviderReplayGeometry geometry;
    std::shared_ptr<PreparedNativeLookupFixture> session;
    DeviceId device;
    std::shared_ptr<RetainedHostServiceBinding> ioBinding;
    RetainedHostServiceInstance ioInstance;
    std::shared_ptr<HostImmutableRowProvider> sharedProvider; ///< Optional complete-group owner, never one replica per consumer.
    std::shared_ptr<q::ImmutableLookupService> sharedService; ///< Same admitted mapped frames and original I/O worker for every group member.
};

/** @brief One actual admitted service and retained fixed graph; no mutable device state is mirrored on the host. */
class ProviderReplay final {
public:
    ProviderReplay(const PreparedProviderReplayProgram& program,ProviderReplayGraphs& captures)
        :file(*program.spec.file),device(program.device),worker(*program.session->permit(device).worker()),backend(getBackendFor(device)),
         stream(ExplicitGPUStream(worker.defaultStream()).get()),maximum(program.spec.maximum),slots(program.spec.slots),width(program.spec.width),
         tokenQueries(program.spec.tokenQueries),interleaving(program.spec.interleaving),activationPreparation(program.spec.activationPreparation),measurement(program.spec.measurement),graphs(captures),originalProgram(program){
        requireProvider(backend && maximum>=16,"Real provider fixture requires its exact native participant");
        program.spec.requireComplete();
        requireProvider(!tokenQueries || (!(maximum%16) && width==n::OriginalIdWidth::UInt64),"Production PLE replay requires the complete original INT64 sixteen-head token envelope");
        memory=program.session->memory();
        requireProvider(memory->phase()==PhysicalMemoryAuthorityPhase::Admitted,"Provider program requires original final admission before payload creation");
        {const auto actual=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);actual.requireWorker(program.session->permit(device).worker());}
        struct Construction {ProviderReplay& self;~Construction(){self.join();}} construction{*this};
        const auto& physical=program.geometry.physical;const auto& policy=program.geometry.policy;const auto& cpuGeometry=program.geometry.cpu;
        const auto& channel=program.geometry.channel;const auto& declarations=program.geometry.declarations;const auto source=file.descriptor();
        activationLease=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,program.geometry.activation);
        requireProvider(bool(program.sharedProvider)==bool(program.sharedService),"Grouped provider/service ownership is incomplete");
        provider=program.sharedProvider?program.sharedProvider:
            HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,cpuGeometry,memory));
        service=program.sharedService?program.sharedService:
            q::ImmutableLookupService::create(provider,channel,std::array{device},memory,DeviceId::cpu(),program.ioBinding->take(program.ioInstance));
        const auto sourceBinding=service->bindSubordinate(device);
        auto ownerPlan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,sourceBinding.deviceView().base.table,sourceBinding.storageGeneration());
        auto coordinatorPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,sourceBinding.deviceView().base.table,sourceBinding.storageGeneration());
        const auto sourceGeometry=q::PooledImmutableSourceGeometry::fromPlans(ownerPlan,coordinatorPlan,source);
        if(program.spec.ownership==OwnerAdmission::RejectForeignAuthority){
            auto foreign=std::make_shared<PhysicalMemoryAuthority>(memory->admission(),0);
            EXPECT_THROW(q::PooledImmutableOwnerPreparedStorage::create(ownerPlan,foreign,stream),std::logic_error);
            EXPECT_EQ(foreign->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);
        }
        owner=q::PooledImmutableOwnerPreparedStorage::create(std::move(ownerPlan),memory,stream);
        coordinator=q::PooledImmutableCoordinatorPreparedStorage::create(std::move(coordinatorPlan),memory,stream);
#ifdef HAVE_CUDA
        if(device.is_cuda())code=q::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())code=q::createROCmPooledImmutableNativeKernels();
#endif
        requireProvider(bool(code),"Real provider native phase bundle unavailable");code->initialize(*owner,stream);code->initialize(*coordinator,stream);
        boundary=q::ImmutableLookupExecutionBoundary::createPooled(service,sourceGeometry,owner,coordinator,memory,stream);
        arena=std::make_unique<BufferArena>();
        const std::array roles{BufferId::QFN_NGRAM_ROW_IDS,BufferId::PREFILL_CHUNK_GEOMETRY,BufferId::MTP_GENERATION_CONTROL,BufferId::QFN_PLE_EMBEDDING};
        for(std::size_t index=0;index<declarations.size();++index){
            std::shared_ptr<TensorBase> root(declarations[index].materialize(memory));
            if(index==3)std::fill_n(static_cast<float*>(root->raw_mutable_data()),std::size_t{maximum}*file.columns,-12345.f);
            if(activationPreparation==ActivationPreparation::BeforeGraph || (index!=0 && index!=3)){
                TransferEngine::prepareDeviceInput(root.get(),device,stream);TransferEngine::requireDeviceInput(root.get(),device,stream);
            }
            requireProvider(arena->registerExternalBuffer(roles[index],root),"Real provider original activation registration failed");
        }
        bindPooledNativeRoots(*arena,*owner);bindPooledNativeRoots(*arena,*coordinator);bindPooledSourceRoots(*arena,*boundary);
        control.emplace(DeviceGenerationControlBinding::admit(*arena,device,0));boundary->bindPooledGenerationControl(*control);
        expected.assign(std::size_t{maximum}*file.columns,-12345.f);
        // Poison only setup payload banks. No production copy may clear padding
        // or inactive rows merely because those bytes were admitted.
        for(const auto& root:{owner->roots().cache,boundary->pooledStorage()->roots().frame})if(root){
            const auto offset=root==owner->roots().cache?owner->plan().parameters().cache.payload:0;
            std::vector<std::byte> poison(root->size_bytes()-offset,std::byte{0xa5});
            requireProvider(backend->hostToDeviceOnStream(static_cast<std::byte*>(root->gpu_data_ptr())+offset,poison.data(),poison.size(),device.ordinal,stream),"Real provider payload poison failed");join();
            TransferEngine::publishDeviceWrite(root.get(),device,stream);
        }
        context=IDeviceContext::create(device,1);build();if(tokenQueries)buildSmallProductionGraph();identity=boundary->captureIdentity();

    }
    /** @brief Retire actual submissions and worker before any graph/source/backing owner releases. */
    ~ProviderReplay(){
        join();
        if(boundary && boundary->hasActiveInterval()){
            try{boundary->recordTerminalClose(stream);join();boundary->awaitTerminal();}
            catch(...){/* The original failed service remains retained through joined graph teardown. */}
        }
    }
    /** @brief Setup/terminal diagnostic wait only, never called by a production stage. */
    void join()const{if(!backend || !backend->synchronizeStream(stream,device.ordinal))std::terminate();}
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    /** @brief Cold semantic extension point; a requested unbound raw lane fails instead of changing graph execution. */
    void bindDiagnosticProgram(ComputeGraph& graph){
        switch(originalProgram.spec.diagnosticProgram){
        case ProviderReplayDiagnosticProgram::OrdinaryLookup:
            if(graph.readAheadBranch())throw std::logic_error("Ordinary raw provider program has a foreign side-lane declaration");
            return;
        case ProviderReplayDiagnosticProgram::ReadAheadLookup:
            {
            if(!graphs.side() || !tokenQueries)throw std::logic_error("Actual raw lookup has no original side/parent family");
            const std::string producer=(&graph==&smallCompute)?"ple_gather_small":"ple_gather";
            const auto* reader=graph.getNode(producer);
            if(!reader || reader->dependencies.size()!=1)throw std::logic_error("Actual side reader lost its sole original producer");
            std::string consumer;
            for(const auto& name:graph.getExecutionOrder()){
                const auto* node=graph.getNode(name);
                const auto* stage=dynamic_cast<const PooledImmutableSourcePhaseStage*>(node->stage.get());
                if(stage && stage->matchesOriginal(boundary.get(),q::PooledSourceAction::ReadersRetired,q::adaptive_lookup::None,*control)){
                    if(!consumer.empty())throw std::logic_error("Actual side reader duplicated its original retirement anchor");
                    consumer=name;
                }
            }
            if(consumer.empty())throw std::logic_error("Actual side reader lacks its original ReadersRetired operation");
            graph.setReadAheadPhasePolicy(ForwardExecutionPhase::Decode,ImmutableLookupInvocationSchedule::ResidentSideLane);
            graph.setReadAheadBranch({reader->dependencies.front(),producer,consumer,graphs.side()});
            graph.planReadAheadRecordingStream(stream);
            return;
        }
        }
        throw std::invalid_argument("Unknown raw provider diagnostic program");
    }
#endif
    /** @brief Build once through actual reusable stages and capture dependency planning. */
    void build(){
        q::PooledNativeOperandSpans begin{};const auto ids=arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arena->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
        const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        begin[static_cast<unsigned>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,std::size_t{maximum}*(width==n::OriginalIdWidth::UInt32?4:8)};
        begin[static_cast<unsigned>(q::PooledNativeOperand::OriginalCount)]=q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,4};
        begin[static_cast<unsigned>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{boundary->pooledStorage()->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,
            offsets.protocol+offsetof(q::ImmutableLookupDeviceState,admissionGeneration),8};
        const auto output=arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        const auto tokens=DeviceRowRange::deviceCounted(tokenQueries?maximum/16:maximum,
            static_cast<const std::int32_t*>(count->gpu_data_ptr()));
        const auto original=tokenQueries?tokens.repeatEachRow(16):tokens;
        overboundInputs.emplace(n::CoordinatorAction::Begin,coordinator->plan(),n::OriginalRows{original,width},begin);
        if(tokenQueries)
            appendPooledImmutablePLELookupGraph(compute,*arena,boundary,code,tokens,
                {ids,BufferId::QFN_NGRAM_ROW_IDS,0,std::size_t{maximum}*sizeof(std::uint64_t)},
                q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,sizeof(std::int32_t)},*control,
                {output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()},"production_ple","ple_gather");
        else appendPooledImmutableLocalLookupGraph(compute,*arena,boundary,code,{original,width},std::move(begin),*control,
            {output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()},"real_provider");
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        bindDiagnosticProgram(compute);
#endif
        executor.setArena(arena.get());
        std::vector<std::uint64_t> declarations;
        for(const auto& name:compute.getExecutionOrder()){
            const auto& stage=*compute.getNode(name)->stage;
            declarations.push_back(stage.graphCaptureVariantSignature());
            (void)stage.bufferContract();
        }
        if(activationPreparation==ActivationPreparation::ColdGraph){
            // The old production constructor seal fails here before the graph
            // exists. Exact original declarations must survive without native addresses.
            EXPECT_EQ(ids->gpu_data_ptr(),nullptr);EXPECT_EQ(output->gpu_data_ptr(),nullptr);
            auto* begin=compute.getNode(compute.getExecutionOrder().front())->stage.get();
            auto* reader=compute.getNode("ple_gather")->stage.get();
            EXPECT_THROW(begin->execute(context.get()),std::logic_error);
            EXPECT_THROW(reader->execute(context.get()),std::logic_error);
            EXPECT_THROW(begin->prepareGraphLaunch(context.get(),stream),std::logic_error);
            EXPECT_THROW(reader->prepareGraphLaunch(context.get(),stream),std::logic_error);
            EXPECT_EQ(ids->gpu_data_ptr(),nullptr);EXPECT_EQ(output->gpu_data_ptr(),nullptr);
            requireProvider(executor.allocateGraphStorageForCapture(compute,context.get(),"real provider cold declarations"),
                "Real provider cold arena materialization failed");
            EXPECT_NE(ids->gpu_data_ptr(),nullptr);EXPECT_NE(output->gpu_data_ptr(),nullptr);
            // Initial fixture I/O supplies the same poison and input bytes used
            // by warm setup, after allocation and before the captured transaction.
            for(const auto& root:{ids,output}){
                TransferEngine::prepareDeviceInput(root.get(),device,stream);
                TransferEngine::requireDeviceInput(root.get(),device,stream);
            }
        }
        for(const auto& name:compute.getExecutionOrder()){
            auto& stage=*compute.getNode(name)->stage;
            requireProvider(stage.prepareGraphLaunch(context.get(),compute.executionStreamForNode(name,stream)),"Real provider capture-only stage preparation failed");
            requireProvider(stage.prepareGraphLaunch(context.get(),compute.executionStreamForNode(name,stream)),"Real provider repeated capture-only preparation failed");
        }
        for(std::size_t index=0;const auto& name:compute.getExecutionOrder())
            EXPECT_EQ(compute.getNode(name)->stage->graphCaptureVariantSignature(),declarations[index++]);
        requireProvider(executor.prepareGraphStorageForCapture(compute,context.get(),stream,"real provider complete native lookup"),"Real provider graph preparation failed");
        graph=graphs.capture(0);terminal=graphs.capture(1);overbound=graphs.capture(2);
        requireProvider(graph && terminal && overbound,"Real provider retained graph creation failed");
        compute.sealReadAheadRecordingStream(graph->executionStream());
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(compute.readAheadBranch()){
            const auto& lane=*compute.readAheadBranch();
            requireProvider(compute.getNode(lane.producer_node)->stage->requireGPUStream()==lane.resources->stream(),"Actual optimized PLE reader did not consume its retained side lane");
            std::cout<<"POOLED_PROVIDER_SIDE_GRAPH "<<nlohmann::json{{"qualified",false},{"producer",lane.producer_node},
                {"fork",lane.fork_node},{"join",lane.join_node},{"capture_identity",compute.readAheadCaptureIdentity()},
                {"recording_stream",reinterpret_cast<std::uintptr_t>(graph->executionStream())},
                {"source_storage_identity",boundary->pooledStorage()->captureIdentity()},
                {"arithmetic_program","original_ple_gather"}}.dump()<<'\n';
        }
#endif
        const auto& order=compute.getExecutionOrder();auto dependencies=executor.planGraphCaptureDependencies(compute,order,device,stream,"real provider source/owner/control edges");
        requireProvider(bool(dependencies),"Real provider capture dependency plan missing");
        overboundInputs->seal();
        {
            ScopedBackendGraphCapture capture(worker,*graph,"real provider complete local lookup",dependencies.get());requireProvider(capture.begin(),"Real provider capture begin failed");
            requireProvider(executor.recordSetupGraphMaterialization(compute,context.get()),"Real provider actual stage recording failed");capture.finish();
        }
        {
            ScopedBackendGraphCapture capture(worker,*terminal,"real provider complete parent terminal");requireProvider(capture.begin(),"Real provider terminal capture failed");
            boundary->recordTerminalClose(stream,control->healthy(),control->row()+sampling_math::kDeviceGenerationControlRequestComplete);capture.finish();
        }
#ifdef LLAMINAR_POOLED_IDENTITY_REJECTION_TESTS
        if(originalProgram.spec.identityRejection)buildIdentityRejection();
        else
#endif
        {
            ScopedBackendGraphCapture capture(worker,*overbound,"real provider unique-bound rejection");requireProvider(capture.begin(),"Real provider over-bound capture failed");
            code->enqueue(n::CoordinatorAction::Begin,*coordinator,*overboundInputs,stream);capture.finish();
        }
        std::string detail;requireProvider(graph->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
        requireProvider(terminal->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
        requireProvider(overbound->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
        if(interleaving!=ACKInterleaving::None){
            const auto original=sourceArguments();
            const auto replay=interleaving==ACKInterleaving::OriginalUnchecked900?
                pooled_ack::AuthorizationReplay::OriginalUnchecked:pooled_ack::AuthorizationReplay::Production;
            RetainedParticipantGraphExecutableFixtureAccess::spliceRecordedACK(
                graphs.definition(0),[&](IGPUGraphCapture& native){
#ifdef HAVE_CUDA
                    if(device.is_cuda())pooled_ack::spliceCUDA(native,original,stream,replay);
#endif
#ifdef HAVE_ROCM
                    if(device.is_rocm())pooled_ack::spliceROCm(native,original,stream,replay);
#endif
                });
        }
        requireProvider(graph->instantiate() && terminal->instantiate() && overbound->instantiate(),"Real provider retained graph instantiation failed");
        if(interleaving!=ACKInterleaving::None){
            bool mutated=false;
            EXPECT_THROW(RetainedParticipantGraphExecutableFixtureAccess::spliceRecordedACK(
                graphs.definition(0),[&](IGPUGraphCapture&){mutated=true;}),std::logic_error);
            EXPECT_FALSE(mutated);
        }
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        nlohmann::json stages=nlohmann::json::array();
        for(const auto& name:compute.getExecutionOrder()){
            const auto* node=compute.getNode(name);stages.push_back({{"name",name},{"device",node->device.toString()},
                {"signature",node->stage->graphCaptureVariantSignature()},{"dependencies",node->dependencies}});
        }
        std::cout<<"POOLED_PROVIDER_GRAPH "<<nlohmann::json{{"qualified",false},{"program_role","actual_pooled_local_lookup"},
            {"recorded_graphs",{{"lookup",graph->nodeCount()},{"terminal",terminal->nodeCount()},{"bound_rejection",overbound->nodeCount()}}},
            {"original_capture_slots",{0,1,2}},{"table_rows",file.rows},{"columns",file.columns},{"row_bytes",file.row_bytes},
            {"maximum_ids",maximum},{"unique_capacity",originalProgram.spec.uniqueCapacity},{"cache_slots",slots},
            {"read_ahead_identity",compute.readAheadCaptureIdentity()},{"stages",std::move(stages)},
            {"general_host_certificate_scope","none; exact actual program observation only"}}.dump()<<'\n';
#endif

        std::vector<GPUGraphKernelNodeInfo> nodes;requireProvider(graph->inspectKernelNodes(nodes,&detail),detail.c_str());
        for(const auto& node:nodes){requireProvider(node.valid(),"Real provider native resource inventory incomplete");
            const auto scope=node.name.find("pooled_source")!=std::string::npos?"source":node.name.find("coordinator")!=std::string::npos?"coordinator":"owner";
            std::cout<<"POOLED_PROVIDER_NODE scope="<<scope<<" name="<<node.name<<" threads="<<node.block_x<<" registers="<<node.registers_per_thread
                <<" local_bytes="<<node.local_memory_bytes_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes
                <<" active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';
        }
    }
    /** @return Exact original source launch operands for setup-only graph authentication. */
    q::pooled_source::DeviceArgs sourceArguments()const{
        const auto storage=boundary->pooledStorage();const auto own=owner->roots(),coord=coordinator->roots(),root=storage->roots();
        const auto address=[](const std::shared_ptr<TensorBase>& value){return value?static_cast<std::byte*>(value->gpu_data_ptr()):nullptr;};
        const auto& p=storage->geometry().parameters();
        return {p,{p.owner,address(own.cache),address(own.frame),address(own.native),{}},
            {p.coordinator,address(coord.cache),address(coord.frame),address(coord.native),{}},
            address(root.frame),address(root.native),boundary->pooledKernels().source().deviceView(),control->healthy(),control->error(),control->row()};
    }
    /** @brief Capture a genuine smaller PLE variant against the same original source roots and token count. */
    void buildSmallProductionGraph(){
        const auto ids=arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arena->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
        const auto output=arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        const auto token=DeviceRowRange::deviceCounted(1,static_cast<const std::int32_t*>(count->gpu_data_ptr()));
        appendPooledImmutablePLELookupGraph(smallCompute,*arena,boundary,code,token,
            {ids,BufferId::QFN_NGRAM_ROW_IDS,0,16*sizeof(std::uint64_t)},
            q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,sizeof(std::int32_t)},*control,
            {output,BufferId::QFN_PLE_EMBEDDING,0,16*file.columns*sizeof(float)},"production_ple_small","ple_gather_small");
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        bindDiagnosticProgram(smallCompute);
#endif
        smallExecutor.setArena(arena.get());
        for(const auto& name:smallCompute.getExecutionOrder())
            requireProvider(smallCompute.getNode(name)->stage->prepareGraphLaunch(context.get(),smallCompute.executionStreamForNode(name,stream)),
                "Small production PLE capture-only stage preparation failed");
        requireProvider(smallExecutor.prepareGraphStorageForCapture(smallCompute,context.get(),stream,"production PLE smaller variant"),"Small production PLE preparation failed");
        const auto& order=smallCompute.getExecutionOrder();
        smallGraph=graphs.capture(3);requireProvider(bool(smallGraph),"Small production PLE retained graph missing");
        smallCompute.sealReadAheadRecordingStream(smallGraph->executionStream());
        auto dependencies=smallExecutor.planGraphCaptureDependencies(smallCompute,order,device,stream,"production PLE smaller native frame");
        requireProvider(bool(dependencies),"Small production PLE capture dependency plan missing");
        {ScopedBackendGraphCapture capture(worker,*smallGraph,"production PLE smaller variant",dependencies.get());
            requireProvider(capture.begin(),"Small PLE capture begin failed");
            requireProvider(smallExecutor.recordSetupGraphMaterialization(smallCompute,context.get()),"Small PLE recording failed");capture.finish();}
        std::string error;requireProvider(smallGraph->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&error),error.c_str());
        requireProvider(smallGraph->instantiate(),"Small production PLE retained graph instantiation failed");
        const auto* reader=smallCompute.getNode("ple_gather_small");requireProvider(bool(reader),"Small PLE semantic reader absent");
        const auto& dump=reader->stage->getDumpInfo();requireProvider(dump.outputs.size()==1 && dump.outputs.front().rows==1,
            "Small production PLE output span does not preserve actual graph geometry");
    }
#ifdef LLAMINAR_POOLED_IDENTITY_REJECTION_TESTS
#include "PooledIdentityRejectionMethods.inc"
#endif
    /** @brief Initial input/control publication; native output/cache state is never downloaded and re-uploaded.
     * @param original Original fixed live input IDs.
     * @param error Optional existing canonical first-error input, authored before the original upload.
     */
    void input(std::span<const std::uint64_t> original,sampling_math::DeviceGenerationError error=sampling_math::DeviceGenerationError::None){
        requireProvider(original.size()<=maximum && (!tokenQueries || !(original.size()%16)),"Real provider original input exceeds its exact token/ID bound");auto ids=arena->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS);
        std::memset(ids->raw_mutable_data(),0xff,ids->size_bytes());
        if(width==n::OriginalIdWidth::UInt64){if(!original.empty())std::memcpy(ids->raw_mutable_data(),original.data(),original.size_bytes());}
        else for(std::size_t i=0;i<original.size();++i){requireProvider(original[i]<=~std::uint32_t{0},"Real UInt32 producer overflow");static_cast<std::uint32_t*>(ids->raw_mutable_data())[i]=original[i];}
        auto count=arena->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);*static_cast<int*>(count->raw_mutable_data())=tokenQueries?original.size()/16:original.size();
        auto controller=arena->getSharedTensor(BufferId::MTP_GENERATION_CONTROL);std::memset(controller->raw_mutable_data(),0,controller->size_bytes());static_cast<int*>(controller->raw_mutable_data())[sampling_math::kDeviceGenerationControlOk]=1;
        if(error!=sampling_math::DeviceGenerationError::None)
            (void)sampling_math::fail_device_generation_control(static_cast<int*>(controller->raw_mutable_data()),error);
        for(const auto& root:{ids,count,controller}){TransferEngine::prepareDeviceInput(root.get(),device,stream);TransferEngine::requireDeviceInput(root.get(),device,stream);}join();
    }
    /** @brief Read terminal evidence only; no observation below can select a production branch or repair native state. */
    template<class T>std::vector<T> read(const void* source,std::size_t count){
        std::vector<T> result(count);if(count)requireProvider(backend->deviceToHostOnStream(result.data(),source,count*sizeof(T),device.ordinal,stream),"Real provider terminal read failed");join();return result;
    }
    /** @brief Arm the actual existing boundary once per request while retaining proven captured topology. */
    void admit(std::uint64_t generation){boundary->beginAdmission(generation,stream);join();
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        graphs.beginRequest();
#endif
    }
    /** @brief Replay actual provider delivery and compare unchanged native rows, original permutation and final pin retirement. */
    void replay(std::span<const std::uint64_t> original){
        input(original);void* start=backend->createTimingEvent(device.ordinal);void* stop=backend->createTimingEvent(device.ordinal);
        requireProvider(start && stop,"Real provider timing events missing");requireProvider(backend->recordEvent(start,device.ordinal,stream),"Real provider timing start failed");
        requireProvider(graphs.launch(tokenQueries && original.size()<=16?smallGraph:graph),"Real provider captured replay failed");
        for(const auto& storage:{owner->roots(),coordinator->roots(),boundary->pooledStorage()->roots()})
            for(const auto& root:{storage.cache,storage.frame,storage.native})if(root)TransferEngine::publishDeviceWrite(root.get(),device,stream);
        TransferEngine::publishDeviceWrite(control->owner(),device,stream);TransferEngine::publishDeviceWrite(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING).get(),device,stream);
        requireProvider(backend->recordEvent(stop,device.ordinal,stream) && backend->waitForEvent(stop,device.ordinal),"Real provider replay did not retire");
        float elapsed=0;requireProvider(backend->eventElapsedTimeMs(start,stop,device.ordinal,&elapsed),"Real provider timing read failed");
        backend->destroyEvent(stop,device.ordinal);backend->destroyEvent(start,device.ordinal);
        std::cout<<"POOLED_PROVIDER_TIMING original="<<original.size()<<" maximum="<<maximum<<" slots="<<slots<<" format="<<static_cast<unsigned>(file.type)<<" columns="<<file.columns<<" row_bytes="<<file.row_bytes<<" captured_us="<<elapsed*1000<<" uploads_terminal_observations_timed=0 profile=unqualified\n";
        validateReplay(original);
    }
    /** @brief Upload only fixed test inputs outside measured execution. @param original Declared input trace. */
    void prepareTiming(std::span<const std::uint64_t> original){input(original);}
    /** @brief Submit the original complete maximum-geometry graph; no oracle/readback is performed. */
    void launchTiming(){requireProvider(graphs.launch(graph),"Timed original pooled graph launch failed");}
    /** @brief Publish and validate the completed original graph outside timing. @param original Fixed input trace. */
    void validateTiming(std::span<const std::uint64_t> original){
        for(const auto& storage:{owner->roots(),coordinator->roots(),boundary->pooledStorage()->roots()})
            for(const auto& root:{storage.cache,storage.frame,storage.native})if(root)TransferEngine::publishDeviceWrite(root.get(),device,stream);
        TransferEngine::publishDeviceWrite(control->owner(),device,stream);TransferEngine::publishDeviceWrite(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING).get(),device,stream);
        validateReplay(original);
    }
    /** @brief Shared byte/counter oracle, never part of measured execution. @param original Fixed input trace. */
    void validateReplay(std::span<const std::uint64_t> original){
        const auto own=owner->roots(),coord=coordinator->roots(),source=boundary->pooledStorage()->roots();
        const auto& op=owner->plan().parameters();const auto& cp=coordinator->plan().parameters();const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        const auto frame=read<p::LookupFrame>(static_cast<const std::byte*>(coord.frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_EQ(frame.phase,p::LookupPhase::Idle);EXPECT_EQ(frame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(frame.original_ids,original.size());
        const auto physical=read<p::OwnerFrame>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.control,1).front();
        EXPECT_EQ(physical.phase,p::OwnerPhase::Idle);EXPECT_EQ(physical.held_pins,0u);EXPECT_EQ(physical.reservations,0u);
        const auto controller=read<int>(control->row(),sampling_math::kDeviceGenerationControlCount);EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlOk],1);EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlErrorCode],0);
        q::QwenAdaptiveCounterTotals observed;ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),observed));
        EXPECT_EQ(observed.calls,1u);EXPECT_EQ(observed.completed_live_ids,original.size());EXPECT_EQ(observed.failed_frames,0u);
        EXPECT_EQ(observed.source_bytes,observed.source_rows*file.row_bytes);
        if(tokenQueries)EXPECT_TRUE(q::validQwenAdaptiveCounters(observed,file.row_bytes,maximum));
        if(!slots){EXPECT_EQ(observed.hit_rows,0u);EXPECT_EQ(observed.promotions,0u);EXPECT_EQ(observed.evictions,0u);}
        if(!original.empty()){
            std::vector<std::byte> selected(original.size()*file.row_bytes);
            for(std::size_t position=0;position<original.size();++position){const auto row=file.original(original[position]);std::memcpy(selected.data()+position*file.row_bytes,row.data(),row.size());}
            q::decodeCPUImmutableLookupRows(file.type,file.columns,file.row_bytes,selected.data(),original.size(),expected.data());
        }
        const auto actual=read<float>(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected.size());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);
        const auto io=read<q::ImmutableLookupDeviceState>(source.native->gpu_data_ptr(),1).front();
        const auto outcome=read<n::SourceFrontier>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.frontier,1).front();
        EXPECT_EQ(outcome.disposition,io.idCount?n::SourceDisposition::Consumed:n::SourceDisposition::Unpublished);
        EXPECT_EQ(outcome.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(io.issuedSequence,io.consumedSequence);
        const auto recordBytes=read<std::uint64_t>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.record_bytes,1).front();
        EXPECT_EQ(recordBytes%sizeof(n::Fill),0u);
        const auto delivered=read<n::Fill>(static_cast<const std::byte*>(source.frame->gpu_data_ptr())+offsets.fills,recordBytes/sizeof(n::Fill));
        for(const auto& fill:delivered){EXPECT_EQ(fill.acknowledgement.destination_token,fill.original.destination_token);EXPECT_EQ(fill.acknowledgement.source.sequence,outcome.source.sequence);}
        if(slots){if(op.owner.rowWidth()==q::adaptive_lookup::RowKeyWidth::UInt32)verifySlots<q::adaptive_lookup::Slot32>();else verifySlots<q::adaptive_lookup::Slot64>();}
        if(measurement){
        const auto unique=read<p::CoordinatorUnique>(static_cast<const std::byte*>(coord.frame->gpu_data_ptr())+cp.frame.unique,frame.unique_rows);
        std::uint64_t hits=0;for(const auto& row:unique)if(row.phase==p::UniquePhase::Read)++hits;
        lastObservation={original.size(),frame.unique_rows,hits,io.idCount,observed.promotions,observed.evictions,io.nativeBytes,0};
        }
        expectedRequests+=io.idCount!=0;expectedIds+=io.idCount;EXPECT_EQ(boundary->captureIdentity(),identity);
    }
    /** @brief Check unchanged original native cache bytes and padding with the actual source-row metadata width. */
    template<class Slot>void verifySlots(){
        const auto roots=owner->roots();const auto& op=owner->plan().parameters();
        const auto metadata=read<Slot>(static_cast<const std::byte*>(roots.cache->gpu_data_ptr())+op.cache.slots,slots);
        const auto payload=read<std::byte>(static_cast<const std::byte*>(roots.cache->gpu_data_ptr())+op.cache.payload,slots*op.slot_stride);
        for(std::size_t slot=0;slot<slots;++slot){
            for(std::size_t byte=file.row_bytes;byte<op.slot_stride;++byte)EXPECT_EQ(payload[slot*op.slot_stride+byte],std::byte{0xa5});
            if(q::adaptive_lookup::PhysicalSlot<Slot>::state(metadata[slot])==q::adaptive_lookup::SlotState::Ready){const auto row=file.original(metadata[slot].row);EXPECT_EQ(std::memcmp(row.data(),payload.data()+slot*op.slot_stride,row.size()),0);}
        }
    }
    /** @brief Prove acknowledged cancellation cannot promote its reservation, while warm pins and consumed witness retire exactly. */
    void cancelledACK(std::span<const std::uint64_t> original){
        requireProvider(interleaving!=ACKInterleaving::None && !original.empty() && original.back()==900,
            "ACK trace requires the original retained cancellation graph and target row");
        const bool originalUnchecked=interleaving==ACKInterleaving::OriginalUnchecked900;
        const auto own=owner->roots(),coord=coordinator->roots(),source=boundary->pooledStorage()->roots();
        const auto& op=owner->plan().parameters();const auto& cp=coordinator->plan().parameters();
        const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        std::uint64_t warmGeneration=0;
        if(slots){const auto before=read<q::adaptive_lookup::Slot64>(static_cast<const std::byte*>(own.cache->gpu_data_ptr())+op.cache.slots,slots);
            for(const auto& row:before)if(row.row==7 && q::adaptive_lookup::PhysicalSlot<q::adaptive_lookup::Slot64>::state(row)==q::adaptive_lookup::SlotState::Ready)warmGeneration=row.generation;
            requireProvider(bool(warmGeneration),"ACK trace requires its actual warmed original row7");}
        input(original);requireProvider(graphs.launch(graph),"Pooled ACK cancellation replay failed");join();
        const auto controller=read<int>(control->row(),sampling_math::kDeviceGenerationControlCount);
        EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlOk],0);
        EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlErrorCode],static_cast<int>(sampling_math::DeviceGenerationError::InvalidController));
        q::QwenAdaptiveCounterTotals observation;ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),observation));
        EXPECT_EQ(observation.calls,1u);EXPECT_EQ(observation.failed_frames,1u);EXPECT_EQ(observation.completed_live_ids,0u);EXPECT_EQ(observation.promotions,originalUnchecked?1u:0u);
        const auto frame=read<p::OwnerFrame>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.control,1).front();
        EXPECT_EQ(frame.phase,p::OwnerPhase::Idle);EXPECT_EQ(frame.held_pins,0u);EXPECT_EQ(frame.reservations,0u);
        const auto lookup=read<p::LookupFrame>(static_cast<const std::byte*>(coord.frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_EQ(lookup.phase,p::LookupPhase::Idle);
        if(slots){const auto after=read<q::adaptive_lookup::Slot64>(static_cast<const std::byte*>(own.cache->gpu_data_ptr())+op.cache.slots,slots);
            bool preserved=false,promoted=false;for(const auto& row:after){const auto state=q::adaptive_lookup::PhysicalSlot<q::adaptive_lookup::Slot64>::state(row);
                promoted|=row.row==900 && state==q::adaptive_lookup::SlotState::Ready;
                if(row.row==7 && state==q::adaptive_lookup::SlotState::Ready){preserved=true;EXPECT_EQ(row.generation,warmGeneration);}}
            EXPECT_TRUE(preserved);EXPECT_EQ(promoted,originalUnchecked);}
        const auto io=read<q::ImmutableLookupDeviceState>(source.native->gpu_data_ptr(),1).front();
        const auto frontier=read<n::SourceFrontier>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.frontier,1).front();
        EXPECT_EQ(frontier.disposition,n::SourceDisposition::Consumed);EXPECT_EQ(frontier.status,q::adaptive_lookup::Status::Cancelled);
        EXPECT_EQ(io.issuedSequence,io.consumedSequence);EXPECT_GT(io.nativeBytes,0u);EXPECT_EQ(io.nativeBytes,io.idCount*file.row_bytes);
        EXPECT_EQ(observation.source_bytes,io.nativeBytes);EXPECT_EQ(observation.source_rows,io.idCount);
        const auto actual=read<float>(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected.size());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);
        expectedRequests+=io.idCount!=0;expectedIds+=io.idCount;
        EXPECT_EQ(boundary->captureIdentity(),identity);
    }
    /** @brief Exercise the entire actual source capacity while free slots still guarantee mixed reservations/source-only. */
    void fullSourceCapacity(std::uint64_t first=2000){
        std::array<std::uint64_t,16> full;for(std::size_t i=0;i<full.size();++i)full[i]=first+i;replay(full);
        const auto roots=boundary->pooledStorage()->roots();const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        const auto io=read<q::ImmutableLookupDeviceState>(roots.native->gpu_data_ptr(),1).front();
        EXPECT_EQ(io.idCount,16u);EXPECT_EQ(io.nativeBytes,16*file.row_bytes);
        const auto fills=read<std::uint64_t>(static_cast<const std::byte*>(roots.native->gpu_data_ptr())+offsets.record_bytes,1).front()/sizeof(n::Fill);
        if(slots){EXPECT_GE(fills,slots-2u);EXPECT_LE(fills,slots);EXPECT_LT(fills,16u);}else EXPECT_EQ(fills,0u);
    }
    /** @brief Replay the exact cooperative production Begin alone; overflow must reject before any owner/source scope opens. */
    void rejectUniqueOverflow(){
        std::array<std::uint64_t,17> original;for(std::size_t i=0;i<original.size();++i)original[i]=3000+i;input(original);
        requireProvider(!boundary->hasActiveInterval(),"Bound negative must not leave a worker waiting on an invalid transaction");
        const auto beforeIO=service->stats();const auto roots=owner->roots();const std::array ownerRoots{roots.cache,roots.frame,roots.native};
        std::array<std::vector<std::byte>,3> originalOwner;
        for(std::size_t i=0;i<ownerRoots.size();++i)if(ownerRoots[i])originalOwner[i]=read<std::byte>(ownerRoots[i]->gpu_data_ptr(),ownerRoots[i]->size_bytes());
        const auto sourceNative=boundary->pooledStorage()->roots().native;
        const auto originalSource=read<std::byte>(sourceNative->gpu_data_ptr(),sourceNative->size_bytes());
        requireProvider(graphs.launch(overbound),"Real provider bound-negative Begin launch failed");join();
        for(std::size_t i=0;i<ownerRoots.size();++i)if(ownerRoots[i]){
            const auto after=read<std::byte>(ownerRoots[i]->gpu_data_ptr(),ownerRoots[i]->size_bytes());EXPECT_EQ(after,originalOwner[i]);
        }
        EXPECT_EQ(read<std::byte>(sourceNative->gpu_data_ptr(),sourceNative->size_bytes()),originalSource);
        const auto delta=RetainedStorageTerminalObservation::between(1,service->mode(),beforeIO,service->stats()).counters;
        static_assert(std::has_unique_object_representations_v<RetainedStorageCounters>);
        const RetainedStorageCounters zero{};EXPECT_EQ(std::memcmp(&delta,&zero,sizeof(delta)),0);EXPECT_FALSE(boundary->hasActiveInterval());
        const auto& cp=coordinator->plan().parameters();const auto& op=owner->plan().parameters();
        const auto frame=read<p::LookupFrame>(static_cast<const std::byte*>(coordinator->roots().frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_NE(frame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(frame.unique_rows,17u);
        const auto physical=read<p::OwnerFrame>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr())+op.frame.control,1).front();
        EXPECT_EQ(physical.phase,p::OwnerPhase::Idle);EXPECT_EQ(physical.held_pins,0u);EXPECT_EQ(physical.reservations,0u);
        const auto actual=read<float>(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected.size());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);
    }
    /** @brief Fail the actual retained source parent before request publication or through its original provider.
     * @param original Original live producer IDs that still create the real two allocation waves.
     * @param abortBeforePublication Whether canonical request failure bars I/O before any ACK; false truncates the real original source.
     * @note No source bytes/completion/ACK are authored by this test. The existing worker owns every source outcome.
     */
    void replayFailure(std::span<const std::uint64_t> original,bool abortBeforePublication){
        // Healthy0/error0 is a valid pre-response controller state. Author a
        // genuine first error through the original controller helper before its
        // existing publication; never use health alone as an abort authority.
        input(original,abortBeforePublication?sampling_math::DeviceGenerationError::InvalidController:sampling_math::DeviceGenerationError::None);
        if(!abortBeforePublication)requireProvider(::ftruncate(file.fd,0)==0,"Cannot fail the original admitted provider source");
        void* start=backend->createTimingEvent(device.ordinal);void* stop=backend->createTimingEvent(device.ordinal);
        requireProvider(start && stop,"Real provider failure timing events missing");
        requireProvider(backend->recordEvent(start,device.ordinal,stream),"Real provider failure timing start failed");
        requireProvider(graphs.launch(graph),"Real provider failed parent launch failed");
        for(const auto& storage:{owner->roots(),coordinator->roots(),boundary->pooledStorage()->roots()})
            for(const auto& root:{storage.cache,storage.frame,storage.native})if(root)TransferEngine::publishDeviceWrite(root.get(),device,stream);
        TransferEngine::publishDeviceWrite(control->owner(),device,stream);
        TransferEngine::publishDeviceWrite(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING).get(),device,stream);
        requireProvider(backend->recordEvent(stop,device.ordinal,stream) && backend->waitForEvent(stop,device.ordinal),"Real provider failure did not retire");
        float elapsed=0;requireProvider(backend->eventElapsedTimeMs(start,stop,device.ordinal,&elapsed),"Real provider failure timing read failed");
        backend->destroyEvent(stop,device.ordinal);backend->destroyEvent(start,device.ordinal);
        std::cout<<"POOLED_PROVIDER_FAILURE_TIMING original="<<original.size()<<" maximum="<<maximum<<" slots="<<slots
            <<" path="<<(abortBeforePublication?"before_publication_abort":"provider_storage_failure")<<" captured_us="<<elapsed*1000
            <<" uploads_terminal_observations_timed=0 profile=unqualified\n";
        const auto own=owner->roots(),coord=coordinator->roots(),source=boundary->pooledStorage()->roots();
        const auto& op=owner->plan().parameters();const auto& cp=coordinator->plan().parameters();const auto& offsets=boundary->pooledStorage()->geometry().parameters().offsets;
        const auto frame=read<p::LookupFrame>(static_cast<const std::byte*>(coord.frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_EQ(frame.phase,p::LookupPhase::Idle);EXPECT_EQ(frame.status,q::adaptive_lookup::Status::Success); // Retired frame, not a successful source outcome.
        const auto physical=read<p::OwnerFrame>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.control,1).front();
        EXPECT_EQ(physical.phase,p::OwnerPhase::Idle);EXPECT_EQ(physical.held_pins,0u);EXPECT_EQ(physical.reservations,0u);
        const auto allocation=read<p::AllocationFrame>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.allocation_control,1).front();
        const auto expectedGrantCount=slots?3u:0u; // Independent fixture oracle: two free grants plus one victim.
        const auto grants=read<p::Grant>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.replies,expectedGrantCount);
        const auto accepted=read<p::AllocationDecision>(static_cast<const std::byte*>(own.frame->gpu_data_ptr())+op.frame.accepted_plan,expectedGrantCount);
        EXPECT_EQ(allocation.count,0u);EXPECT_EQ(allocation.phase,p::AllocationPhase::Idle);
        for(std::size_t index=0;index<grants.size();++index){
            EXPECT_EQ(grants[index].row,accepted[index].row);EXPECT_EQ(grants[index].unique,accepted[index].unique);
            EXPECT_EQ(grants[index].destination_token,std::uint64_t{grants[index].unique}+1);
            EXPECT_NE(grants[index].generation,0u);EXPECT_EQ(grants[index].kind,p::GrantKind::Reservation);
        }

        const auto controller=read<int>(control->row(),sampling_math::kDeviceGenerationControlCount);
        EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlOk],0);
        EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlErrorCode],static_cast<int>(abortBeforePublication?
            sampling_math::DeviceGenerationError::InvalidController:sampling_math::DeviceGenerationError::ImmutableStorageFailure));
        q::QwenAdaptiveCounterTotals observed;ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),observed));
        EXPECT_EQ(observed.calls,1u);EXPECT_EQ(observed.failed_frames,1u);EXPECT_EQ(observed.completed_live_ids,0u);
        EXPECT_EQ(observed.source_rows,0u);EXPECT_EQ(observed.source_bytes,0u);EXPECT_EQ(observed.promotions,0u);
        EXPECT_EQ(observed.evictions,slots?1u:0u);
        const auto actual=read<float>(arena->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected.size());
        EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0); // No failed row/output escapes.
        const auto outcome=read<n::SourceFrontier>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.frontier,1).front();
        EXPECT_EQ(outcome.disposition,abortBeforePublication?n::SourceDisposition::Aborted:n::SourceDisposition::Consumed);
        EXPECT_NE(outcome.status,q::adaptive_lookup::Status::Success);
        EXPECT_EQ(read<std::uint64_t>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.record_bytes,1).front(),0u);
        EXPECT_EQ(read<std::uint64_t>(static_cast<const std::byte*>(source.native->gpu_data_ptr())+offsets.payload_bytes,1).front(),0u);
        requireProvider(graphs.launch(terminal),"Real provider failed complete-parent terminal launch failed");join();
        EXPECT_THROW(boundary->awaitTerminal(),std::runtime_error);
        const auto stats=boundary->terminalObservation();EXPECT_EQ(stats.counters.requests,abortBeforePublication?0u:1u);
        EXPECT_EQ(stats.counters.nativeBytes,0u);EXPECT_EQ(stats.counters.ids,0u);EXPECT_EQ(stats.counters.idBytes,0u);
        EXPECT_EQ(stats.counters.metadataBytes,abortBeforePublication?0u:q::kLookupRequestLiveBytes+q::kLookupCompletionLiveBytes+3*sizeof(ImmutableRowSubordinateIntent));
        // Both negative workers retire before complete-parent close. Count the
        // original retired pair, plus the exact source-abort pair on a genuine
        // pre-publication abort; an already exited worker cannot observe close.
        constexpr auto workerRetirementBytes=sizeof(decltype(q::ImmutableLookupAcknowledgement::retiredGeneration))+
            sizeof(decltype(q::ImmutableLookupAcknowledgement::retiredInvocation));
        constexpr auto sourceAbortBytes=sizeof(decltype(q::ImmutableLookupAcknowledgement::abortGeneration))+
            sizeof(decltype(q::ImmutableLookupAcknowledgement::abortInvocation));
        EXPECT_EQ(stats.counters.terminalBytes,workerRetirementBytes+(abortBeforePublication?sourceAbortBytes:0));
        EXPECT_EQ(boundary->captureIdentity(),identity);
    }
    /** @brief Publish only canonical terminal completion and join the actual worker through its existing boundary. */
    void close(){
        publishClose();
        retireOriginalProviderRequest(*this,graphs);
        validateClose();
    }
    /** @brief Publish the original terminal on this channel; a grouped parent publishes every peer before awaiting any. */
    void publishClose(){
        const int complete=1;requireProvider(backend->hostToDeviceOnStream(control->row()+sampling_math::kDeviceGenerationControlRequestComplete,&complete,sizeof(complete),device.ordinal,stream),"Real provider terminal input failed");join();
        requireProvider(graphs.launch(terminal),"Real provider retained terminal launch failed");
    }
    /** @brief Validate the same immutable terminal counters after the parent joined and consumed this channel's await right. */
    void validateClose(){
        const auto stats=boundary->terminalObservation();EXPECT_EQ(stats.transport,RetainedStorageTransport::PooledSubordinate);
        EXPECT_EQ(stats.counters.requests,expectedRequests);EXPECT_EQ(stats.counters.ids,expectedIds);EXPECT_EQ(stats.counters.idBytes,expectedIds*sizeof(std::int64_t));
        EXPECT_EQ(stats.counters.nativeBytes,expectedIds*file.row_bytes);EXPECT_EQ(stats.counters.terminalBytes,24u);
        expectedRequests=expectedIds=0;
    }
    PooledNgramReplayObservation lastObservation{}; ///< Final copied diagnostics, never execution input.
    ProviderFile& file;DeviceId device;IWorkerGPUContext& worker;IBackend* backend;void* stream;std::uint32_t maximum,slots;n::OriginalIdWidth width;bool tokenQueries;ACKInterleaving interleaving;ActivationPreparation activationPreparation;bool measurement=false; ///< Fixture-only terminal observation; original defaults keep their I/O.
    std::shared_ptr<PhysicalMemoryAuthority> memory;PhysicalMemoryAllocationLease activationLease;
    ProviderReplayGraphs& graphs; ///< Original graph owners remain on the parent until handle retirement.
    const PreparedProviderReplayProgram& originalProgram; ///< Immutable source/session program declaration for cold semantic extensions.
    std::shared_ptr<HostImmutableRowProvider> provider;std::shared_ptr<q::ImmutableLookupService> service;
    std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner;std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code;std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary;
    std::unique_ptr<BufferArena> arena;std::optional<DeviceGenerationControlBinding> control;std::unique_ptr<IDeviceContext> context;
    std::vector<float> expected;ComputeGraph compute,smallCompute;DeviceGraphExecutor executor,smallExecutor;
    std::optional<q::PooledImmutableCoordinatorOperands> overboundInputs;IGPUGraphCapture *graph=nullptr,*terminal=nullptr,*overbound=nullptr,*smallGraph=nullptr;
#ifdef LLAMINAR_POOLED_IDENTITY_REJECTION_TESTS
    ComputeGraph identityBeforeCompute,identityFaultCompute,identityDrainCompute;
    DeviceGraphExecutor identityBeforeExecutor,identityFaultExecutor,identityDrainExecutor;
    IGPUGraphCapture* identityFaultGraph=nullptr;
    IGPUGraphCapture* identityDrainGraph=nullptr;
    const PooledImmutableNativePhaseStage::Owner* identitySelected=nullptr; ///< Borrowed immutable phase owned by original compute.
#endif
    std::uint64_t identity=0,expectedRequests=0,expectedIds=0;
};
/** @brief One original prepared population and complete declared serialized provider program set. */
class ProviderReplayPrograms final {
public:
    ProviderReplayPrograms(const ProviderReplayPrograms&)=delete;
    ProviderReplayPrograms& operator=(const ProviderReplayPrograms&)=delete;
    ProviderReplayPrograms(ProviderReplayPrograms&&)noexcept=default;
    /** @brief Preserve actual reset receipts on partial pre-submission construction failure as well as success. */
    ~ProviderReplayPrograms()noexcept{try{retire();}catch(const std::exception& failure){std::cerr<<"FATAL original provider retirement: "<<failure.what()<<'\n';std::terminate();}}
    /** @brief Declare actual immutable files/geometries before any native selected worker is born. */
    explicit ProviderReplayPrograms(std::vector<ProviderReplaySpec> specs):specs_(std::move(specs)){
        requireProvider(!specs_.empty(),"Provider program set cannot omit its real cases");for(const auto& spec:specs_)spec.requireComplete();
    }
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    /** @brief Select an explicit raw measurement program before any selected worker or authority is created. */
    void setDiagnosticProgram(ProviderReplayDiagnosticProgram selected){
        if(session_)throw std::logic_error("Raw provider program cannot change after original preparation");
        for(auto& spec:specs_){spec.diagnosticProgram=selected;spec.requireComplete();}
    }
#endif
    /** @brief Freeze and admit the payload envelope plus all separately retained native populations once. */
    void prepare(DeviceId device){
        if(session_)throw std::logic_error("Provider program set preparation is one-shot");
        const std::array selected{device};session_=std::shared_ptr<PreparedNativeLookupFixture>(PreparedNativeLookupFixture::prepare(selected));
        std::vector<ProviderReplayGeometry> declarations;
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        const auto io=llaminar2::test::RetainedHostServiceCalibrationAccess::program(RetainedHostServiceRole::ImmutableLookupIO);
#else
        const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
#endif
        for(const auto& spec:specs_){auto geometry=ProviderReplayGeometry::create(spec,device,*session_,io);declarations.push_back(geometry);
            const RetainedHostServiceInstance instance{0,device,RetainedHostServiceRole::ImmutableLookupIO,programs_.size()};
            programs_.push_back({spec,std::move(geometry),session_,device,{},instance});}
        session_->admit(programBOM(declarations));consumed_.assign(programs_.size(),false);
        // Each run is an explicit serialized original program. Independent
        // bindings claim only that instance before construction; all opaque
        // obligations were already included additively in the same final BOM.
        for(auto& program:programs_){RetainedHostServiceDeclarations::Builder declared(0);declared.add(program.ioInstance,program.geometry.ioProgram);
            const auto immutable=std::move(declared).freeze();program.ioBinding=immutable->bind(session_->memory(),std::span{&program.ioInstance,std::size_t{1}});}
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        const auto& original=session_->permit(device);const auto originalKey=original.provenance().originalBirthKey();
        if(!originalKey)throw std::logic_error("Actual raw program lost its original prepared source class");
        std::cout<<"POOLED_PROVIDER_SOURCE "<<nlohmann::json{{"qualified",false},{"birth_key",nativeSourceBirthKeyJson(*originalKey)},
            {"auxiliary_program_family",NativeAuxiliaryResourceCatalog::resourceSourceIdentity()},{"original_generation",original.generation()},
            {"original_worker",reinterpret_cast<std::uintptr_t>(original.worker().get())},
            {"original_pma",reinterpret_cast<std::uintptr_t>(original.memory().get())}}.dump()<<'\n';
        std::size_t graphCount=0;for(const auto& program:programs_)graphCount+=program.geometry.graphCount;
        std::cout<<"POOLED_PROVIDER_ADMISSION "<<nlohmann::json{{"original_source",true},{"same_pma",true},{"source_and_known_payload_admitted",true},
            {"whole_native_program_qualified",false},{"cpu_io_exact_stack_object_admitted",true},{"cpu_io_creator_policy",static_cast<unsigned>(io.creatorPolicy())},{"cpu_io_resource_class",io.resourceIdentity()},{"execution_role","raw_actual_program_diagnostic"},{"declared_cases",programs_.size()},
            {"general_graph_slots",graphCount},{"activation_root_declarations_per_case",4},{"unknown_native_populations",{"general_graph_host","activation_publication_and_capture_dependencies","diagnostic_timing_events","cpu_io_service_libc_tls"}},{"whole_process_byte_completeness",false}}.dump()<<'\n';
#endif
    }
    /** @return Serialized payload envelope plus every independently retained native obligation. */
    static PhysicalMemoryPlan programBOM(std::span<const ProviderReplayGeometry> declarations){
        if(declarations.empty())throw std::invalid_argument("Provider plan requires its complete recorded program set");
        std::vector<PhysicalMemoryPlan> alternatives;PhysicalMemoryPlanBuilder retained;
        for(const auto& geometry:declarations){alternatives.push_back(geometry.payload);for(const auto& row:geometry.retained.resources())retained.add(row);}
        PhysicalMemoryPlanBuilder full;full.addMutuallyExclusive(alternatives);const auto native=retained.build();for(const auto& row:native.resources())full.add(row);return full.build();
    }
    /** @return Exact original source worker; this accessor never creates a context. */
    IWorkerGPUContext& worker()const{if(!session_ || programs_.empty())throw std::logic_error("Provider worker requires its original prepared program set");return *session_->permit(programs_.front().device).worker();}
    /** @return Immutable admitted case token. */
    const PreparedProviderReplayProgram& program(std::size_t index)const{return programs_.at(index);}
    /** @brief Run exactly one actual fixture body; graph handles retire on parent before embedded payload roots. */
    template<class Body>void run(std::size_t index,Body&& body){
        if(consumed_.at(index))throw std::logic_error("Provider program native population is one-shot; request replay reuses its existing captures");
        consumed_[index]=true;const auto& current=program(index);ProviderReplayGraphs graphs(session_->permit(current.device),current.geometry.graphCount
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
            ,current.spec.diagnosticProgram
#endif
        );
        std::unique_ptr<ProviderReplay> replay;std::exception_ptr primary;
        try{worker().submitAndWait([&]{replay=std::make_unique<ProviderReplay>(current,graphs);body(*replay);replay->join();});}
        catch(...){primary=std::current_exception();}
        if(replay)worker().submitAndWait([&]{replay->join();});
        graphs.retire();worker().submitAndWait([&]{replay.reset();});
        session_->requirePayloadRetired();
        if(primary)std::rethrow_exception(primary);
    }
    /** @brief Finish the actual original population; process-native reservations deliberately remain retained. */
    void retire(){if(session_){session_->retire();
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(!retirementReported_){
            for(const auto& receipt:session_->retirementReceipts()){
                requireProvider(receipt.runtime_reset_invoked && receipt.runtime_post_reset_state==DeviceRuntimePostResetState::Quiescent &&
                    receipt.retired_context_generation && receipt.retired_runtime_generation &&
                    receipt.successor_runtime_generation==receipt.retired_runtime_generation+1 &&
                    !receipt.canonical_allocations_before_runtime_reset && !receipt.canonical_allocation_bytes_before_runtime_reset,
                    "Actual native program omitted its original validated reset receipt");
                std::cout<<"POOLED_PROVIDER_RETIRED "<<nlohmann::json{{"qualified",false},{"device",receipt.device.toString()},
                    {"runtime_reset_invoked",receipt.runtime_reset_invoked},{"runtime_post_reset_state","quiescent"},
                    {"retired_context_generation",receipt.retired_context_generation},{"retired_runtime_generation",receipt.retired_runtime_generation},
                    {"successor_runtime_generation",receipt.successor_runtime_generation},
                    {"canonical_allocations_before_runtime_reset",receipt.canonical_allocations_before_runtime_reset},
                    {"canonical_bytes_before_runtime_reset",receipt.canonical_allocation_bytes_before_runtime_reset},
                    {"general_device_tokens_after_reset",session_->memory()->reservedBytes(receipt.device,PhysicalMemoryOwner::NativeGraphExecutable)},
                    {"general_host_certificate",nullptr},{"physical_release_claimed",false}}.dump()<<'\n';
            }
            retirementReported_=true;
        }
#endif
    }}
private:
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    bool retirementReported_=false; ///< Exactly-once diagnostic publication of immutable original reset receipts.
#endif
    std::vector<ProviderReplaySpec> specs_;
    std::shared_ptr<PreparedNativeLookupFixture> session_;
    std::vector<PreparedProviderReplayProgram> programs_;
    std::vector<bool> consumed_; ///< CPU-owned cold program-instance lifecycle, never GPU cache/model state.
};

/** @return Existing loader discriminator, derived from its supported code range rather than a backend-local format list.
 * @note A newly inventoried format outside the current GGUF enum range fails the sweep; it cannot disappear silently. */
std::optional<GGUFTensorType> providerFormat(TensorType type){
    for(std::uint32_t code=0;code<=static_cast<std::uint32_t>(GGUFTensorType::BF16);++code){
        try{const auto format=static_cast<GGUFTensorType>(code);if(ggufToTensorType(format)==type)return format;}
        catch(const std::invalid_argument&){}
    }
    return std::nullopt;
}
