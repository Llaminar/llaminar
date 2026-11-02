/**
 * @file PooledCrossVendorReadReplay.inc
 * @brief Actual two-participant source→publication→hit-read→retirement capture fixture.
 *
 * Included inside the canonical ProviderSourceDevice fixture namespace to reuse
 * its original ProviderFile/format inventory and diagnostic boundaries. Both
 * endpoints use the supported original cross-vendor channel holders. This test
 * has no alternate source, local fake ingress, copied count bank, direct remote
 * compute access or native-peer mapped substitution. Input uploads and final
 * observations are diagnostic boundaries; every protocol phase stays captured.
 */
#include "PooledCrossVendorReadReplayGeometry.inc"
/** @brief Immutable execution token holds the actual single original source/PMA session. */
struct PreparedCrossVendorReadReplayProgram final {
    CrossVendorReadReplaySpec spec;CrossVendorReadReplayGeometry geometry;
    std::shared_ptr<PreparedNativeLookupFixture> session;
    DeviceId source,owner;
    std::shared_ptr<RetainedHostServiceBinding> ioBinding;
    RetainedHostServiceInstance ioInstance;
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    std::shared_ptr<PooledRemoteGraphCalibration> calibration;
#endif
};
/** @brief Native definitions live on the parent until original captured operands retire. */
class CrossVendorReadReplayGraphs final {
public:
    /** @brief Materialize the same original3source+1owner General slots after their exact declaration/admission. */
    explicit CrossVendorReadReplayGraphs(const PreparedCrossVendorReadReplayProgram& program)
        :original_(program),source(program.session->permit(program.source),3),owner(program.session->permit(program.owner),1
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
            ,ProviderReplayDiagnosticProgram::OrdinaryLookup,program.geometry.remoteProgram,program.calibration
#endif
        ){}
private:
    const PreparedCrossVendorReadReplayProgram& original_; ///< Frozen declaration outlives all native definitions.
public:
    ProviderReplayGraphs source,owner;
    /** @brief Query only after the existing complete native join, then retire every sole original definition. */
    void retire(){source.retire();
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(original_.calibration)original_.calibration->checkpoint("before_remote_metadata_release");
#endif
        owner.retire();
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(original_.calibration)original_.calibration->checkpoint("after_remote_metadata_release");
#endif
    }
};
/** @brief Real provider/root/graph owners for one independent source and one remote storage participant. */
class CrossVendorReadReplay final {
public:
    /** @brief Price original typed roots/channel/executables before their actual cold materialization. */
    CrossVendorReadReplay(const PreparedCrossVendorReadReplayProgram& program,CrossVendorReadReplayGraphs& definitions)
        :Maximum(program.spec.maximum),Unique(program.spec.unique),file_(*program.spec.file),devices_{program.source,program.owner},
         workers_{program.session->permit(program.source).worker(),program.session->permit(program.owner).worker()},slots_(program.spec.slots),
         definitions_(definitions),originalProgram_(program){
        program.spec.requireComplete();
        requireProvider(bool(program.ioBinding) && program.ioInstance.worldRank==0 && program.ioInstance.consumer==program.source &&
            program.ioInstance.role==RetainedHostServiceRole::ImmutableLookupIO,"Cross-vendor program omitted its original declared CPU I/O instance");
        const auto sourceDevice=program.source,ownerDevice=program.owner;
        const auto& physical=program.geometry.physical;const auto& policy=program.geometry.policy;const auto& cpuGeometry=program.geometry.cpuGeometry;
        const auto& channel=program.geometry.channel;const auto& declarations=program.geometry.declarations;const auto activation=program.geometry.activation;
        const auto source=file_.descriptor();memory_=program.session->memory();
        requireProvider(memory_->phase()==PhysicalMemoryAuthorityPhase::Admitted,"Cross-vendor program requires its original final admission");
        requireProvider(Maximum>=16 && Maximum%16==0 && Unique && Unique<=Maximum,"Read replay requires its exact retained token/unique geometry");
        requireProvider(sourceDevice.type!=ownerDevice.type && sourceDevice.is_gpu() && ownerDevice.is_gpu(),
            "Read component requires its real supported cross-vendor endpoints");
        for(unsigned i=0;i<2;++i){const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(devices_[i],memory_);original.requireWorker(workers_[i]);}
        for(unsigned i=0;i<2;++i){arenas_[i]=std::make_shared<BufferArena>();graphs_[i]=std::make_shared<ComputeGraph>();backends_[i]=getBackendFor(devices_[i]);requireProvider(backends_[i],"Read component backend unavailable");
            workers_[i]->submitAndWait([&,i]{streams_[i]=ExplicitGPUStream(workers_[i]->defaultStream()).get();contexts_[i]=IDeviceContext::create(devices_[i],1);});}
        struct ConstructionRetirement {CrossVendorReadReplay& original;~ConstructionRetirement(){original.join();}} retirement{*this};
        activation_=memory_->claimNewAllocation(sourceDevice,PhysicalMemoryOwner::ActivationArena,activation);
        provider_=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,cpuGeometry,memory_));
        service_=q::ImmutableLookupService::create(provider_,channel,std::array{sourceDevice},memory_,DeviceId::cpu(),program.ioBinding->take(program.ioInstance));
        const auto issued=service_->bindSubordinate(sourceDevice);const auto table=issued.deviceView().base.table,generation=issued.storageGeneration();
        const auto cp=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,table,generation);
        const auto op=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,table,generation);
        workers_[0]->submitAndWait([&]{coordinator_=q::PooledImmutableCoordinatorPreparedStorage::create(cp,memory_,streams_[0]);code_[0]=bundle(sourceDevice);code_[0]->initialize(*coordinator_,streams_[0]);});
        workers_[1]->submitAndWait([&]{owner_=q::PooledImmutableOwnerPreparedStorage::create(op,memory_,streams_[1]);code_[1]=bundle(ownerDevice);code_[1]->initialize(*owner_,streams_[1]);});
        workers_[0]->submitAndWait([&]{
            boundary_=q::ImmutableLookupExecutionBoundary::createPooled(service_,q::PooledImmutableSourceGeometry::fromPlans(op,cp,source),owner_,coordinator_,memory_,streams_[0]);
            const std::array roles{BufferId::QFN_NGRAM_ROW_IDS,BufferId::PREFILL_CHUNK_GEOMETRY,BufferId::MTP_GENERATION_CONTROL,BufferId::QFN_PLE_EMBEDDING,BufferId::MTP_GENERATION_RESPONSE_TOKENS};
            for(size_t i=0;i<roles.size();++i){std::shared_ptr<TensorBase> root(declarations[i].materialize(memory_));if(i==3)std::fill_n(static_cast<float*>(root->raw_mutable_data()),Maximum*file_.columns,-12345.f);
                TransferEngine::prepareDeviceInput(root.get(),devices_[0],streams_[0]);TransferEngine::requireDeviceInput(root.get(),devices_[0],streams_[0]);requireProvider(arenas_[0]->registerExternalBuffer(roles[i],root),"Read component original activation role replaced");}
            bindPooledNativeRoots(*arenas_[0],*coordinator_);bindPooledSourceRoots(*arenas_[0],*boundary_);control_.emplace(DeviceGenerationControlBinding::admit(*arenas_[0],devices_[0],0));boundary_->bindPooledGenerationControl(*control_);
            forward_=q::PooledImmutableCapturedPinEdge::bind(coordinator_,owner_,memory_,streams_[0],streams_[1]);returned_=q::PooledImmutableCapturedPinReturn::bind(forward_,memory_,streams_[1],streams_[0]);
            free_=q::PooledImmutableCapturedFreeWave::bind(returned_,memory_);victim_=q::PooledImmutableCapturedVictimWave::bind(free_,memory_);source_=q::PooledImmutableCapturedSourceWave::bind(victim_,boundary_,memory_);
            publication_=q::PooledImmutableCapturedPublicationWave::bind(source_,memory_);read_=q::PooledImmutableCapturedReadWave::bind(publication_,memory_);
        });
        workers_[1]->submitAndWait([&]{bindPooledNativeRoots(*arenas_[1],*owner_);});
        expected_.assign(Maximum*file_.columns,-12345.f);
        expectedResponses_.assign(Unique*file_.row_bytes,std::byte{0xa5});expectedPacket_.assign(slots_*file_.row_bytes,std::byte{0xa5});expectedOwnerPacket_.assign(slots_*file_.row_bytes,std::byte{0x5a});
        workers_[0]->submitAndWait([&]{const auto root=boundary_->pooledStorage()->roots().frame;std::vector<std::byte> poison(root->size_bytes(),std::byte{0xa5});requireProvider(backends_[0]->hostToDeviceOnStream(root->gpu_data_ptr(),poison.data(),poison.size(),devices_[0].ordinal,streams_[0]),"Read response poison failed");TransferEngine::publishDeviceWrite(root.get(),devices_[0],streams_[0]);requireProvider(backends_[0]->synchronizeStream(streams_[0],devices_[0].ordinal),"Cold source poison upload did not retire while its original host bytes were alive");});
        if(slots_)workers_[1]->submitAndWait([&]{const auto roots=owner_->roots();const auto& plan=owner_->plan().parameters();
            std::vector<std::byte> poison(slots_*plan.slot_stride,std::byte{0xa5});requireProvider(backends_[1]->hostToDeviceOnStream(static_cast<std::byte*>(roots.cache->gpu_data_ptr())+plan.cache.payload,poison.data(),poison.size(),devices_[1].ordinal,streams_[1]),"Read cache poison failed");
            requireProvider(backends_[1]->hostToDeviceOnStream(static_cast<std::byte*>(roots.native->gpu_data_ptr())+plan.native.source_payload,expectedOwnerPacket_.data(),expectedOwnerPacket_.size(),devices_[1].ordinal,streams_[1]),"Read owner payload poison failed");
            TransferEngine::publishDeviceWrite(roots.cache.get(),devices_[1],streams_[1]);TransferEngine::publishDeviceWrite(roots.native.get(),devices_[1],streams_[1]);requireProvider(backends_[1]->synchronizeStream(streams_[1],devices_[1].ordinal),"Cold owner poison upload did not retire while its original host bytes were alive");});
        build();identity_=read_->captureIdentity();
    }
    /** @brief Retire every real submission before any captured pointer/root/worker owner can unwind. */
    ~CrossVendorReadReplay(){join();if(boundary_ && boundary_->hasActiveInterval())std::terminate();}
    /** @return Original authoritative interval state, never a fixture-owned shadow. */
    bool hasActiveInterval()const{return boundary_ && boundary_->hasActiveInterval();}
    /** @brief Exceptional fixture cleanup uses the same captured reset and terminal service/native join. */
    void finishOutstandingRequest(){if(hasActiveInterval())close(true);}
    /** @brief Drop the cohort after its existing terminal join; actual definitions close before root teardown even under strong aliases. */
    void retireCohort(){join();if(boundary_ && boundary_->hasActiveInterval())throw std::logic_error("Cannot retire an active original cohort");
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(originalProgram_.calibration)originalProgram_.calibration->checkpoint("before_remote_handles_destroyed");
#endif
        cohort_.reset();
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(originalProgram_.calibration)originalProgram_.calibration->checkpoint("after_remote_handles_destroyed");
#endif
    }
    /** @return Original sole component authority for post-destruction claim assertions. */
    const std::shared_ptr<PhysicalMemoryAuthority>& authority()const noexcept{return memory_;}
    /** @brief Admit only the real existing provider ticket, outside retained capture. */
    void admit(std::uint64_t generation){workers_[0]->submitAndWait([&]{boundary_->beginAdmission(generation,streams_[0]);});join();}
    /** @brief Replay exact source/owner captures; observations never choose a protocol branch or repair state. */
    void replay(std::span<const std::uint64_t> original,std::uint32_t expectedUniqueHits);
    /** @brief Fixed input/canary uploads outside timing. @param original Declared source IDs. */
    void prepareTiming(std::span<const std::uint64_t> original,int healthy=1,int error=0);
    /** @brief Replay real pre-response PLE from the canonical reset words, then admit through the production GPU initializer. */
    void coldPrefill(std::span<const std::uint64_t> original,std::uint32_t expectedUniqueHits);
    /** @brief A first error rejects payload and survives actual response admission until captured request reset. */
    void stickyForwardError(std::span<const std::uint64_t> original);
    /** @brief Initialize only the response controller through the original backend, preserving prefill first-error state. */
    void initializeResponse(int expectedError);
    /** @brief Submit both original complete participant captures without observations. */
    void timingLaunch(){cohort_->launch();}
    /** @return Original source stream frozen by the shared cohort recorder. */
    void* measurementStream()const noexcept{return streams_[0];}
    /** @brief Final result collection only, never an inference-stage synchronization. */
    void timingJoin()const{join();}
    /** @brief Shared independent native/output oracle. @param original Fixed inputs. @param expectedUniqueHits Prescribed history. */
    void validateTiming(std::span<const std::uint64_t> original,std::optional<std::uint32_t> expectedUniqueHits,int expectedHealthy=1);
    PooledNgramReplayObservation lastObservation{}; ///< Terminal copied diagnostics, never execution state.
    /** @brief Close only through the original captured terminal operation after complete reader/physical retirement. */
    void close(bool explicitReset=false);
private:
    /** @return Actual production code bundle for precisely this endpoint's family. */
    static std::shared_ptr<const q::IPooledImmutableNativeKernels> bundle(DeviceId device){
#ifdef HAVE_CUDA
        if(device.is_cuda())return q::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())return q::createROCmPooledImmutableNativeKernels();
#endif
        throw std::runtime_error("Cross-vendor reader lacks its requested production native bundle");
    }
    /** @brief Native diagnostic joins are setup/final observations, never inference-stage waits. */
    void join()const{for(unsigned i=0;i<2;++i)if(streams_[i])workers_[i]->submitAndWait([&,i]{if(!backends_[i]->synchronizeStream(streams_[i],devices_[i].ordinal))std::terminate();});}
    /** @brief Build actual source/publication/read/terminal stages, then capture each participant exactly once. */
    void build();
    /** @return Final diagnostic data from one exact original endpoint; no state is reuploaded. */
    template<class T>std::vector<T> observe(unsigned endpoint,const void* from,size_t count)const{std::vector<T> value(count);if(count)workers_[endpoint]->submitAndWait([&,endpoint]{requireProvider(backends_[endpoint]->deviceToHostOnStream(value.data(),from,count*sizeof(T),devices_[endpoint].ordinal,streams_[endpoint]),"Read component final observation failed");});join();return value;}
    const std::uint32_t Maximum,Unique; ///< Immutable retained measurement geometry; correctness defaults stay64/16.
    ProviderFile& file_;std::array<DeviceId,2> devices_;std::array<std::shared_ptr<IWorkerGPUContext>,2> workers_;std::array<IBackend*,2> backends_{};std::array<void*,2> streams_{};std::uint32_t slots_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;PhysicalMemoryAllocationLease activation_;
    std::shared_ptr<HostImmutableRowProvider> provider_;std::shared_ptr<q::ImmutableLookupService> service_;
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator_;std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner_;
    std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary_;std::array<std::shared_ptr<const q::IPooledImmutableNativeKernels>,2> code_;
    std::shared_ptr<const q::PooledImmutableCapturedPinEdge> forward_;std::shared_ptr<const q::PooledImmutableCapturedPinReturn> returned_;
    std::shared_ptr<const q::PooledImmutableCapturedFreeWave> free_;std::shared_ptr<const q::PooledImmutableCapturedVictimWave> victim_;
    std::shared_ptr<const q::PooledImmutableCapturedSourceWave> source_;std::shared_ptr<const q::PooledImmutableCapturedPublicationWave> publication_;
    std::shared_ptr<const q::PooledImmutableCapturedReadWave> read_;std::shared_ptr<const q::PooledImmutableCapturedRetirementWave> retirement_;
    std::optional<DeviceGenerationControlBinding> control_;std::array<std::shared_ptr<BufferArena>,2> arenas_;std::array<std::shared_ptr<ComputeGraph>,2> graphs_;
    std::array<std::shared_ptr<IDeviceContext>,2> contexts_;
    IGPUGraphCapture *terminal_=nullptr,*reset_=nullptr;
    CrossVendorReadReplayGraphs& definitions_; ///< Sole original graph owners outlive these embedded payloads.
    const PreparedCrossVendorReadReplayProgram& originalProgram_;
    PooledImmutableRetirementCompletion completion_;PooledPinParticipantNodes ownerRetired_;
    std::shared_ptr<q::PooledImmutableCompleteCohort> cohort_;
    std::vector<float> expected_;std::vector<std::byte> expectedResponses_,expectedPacket_,expectedOwnerPacket_;std::uint64_t identity_=0,replayIndex_=0;const int terminalComplete_=1;
};
/** @brief Complete authored case set shares ONE actual source/PMA and prices every retained native instance before roots. */
class CrossVendorReadReplayPrograms final {
public:
    /** @brief Freeze the actual immutable case set before selected worker/source creation. */
    explicit CrossVendorReadReplayPrograms(std::vector<CrossVendorReadReplaySpec> specs):specs_(std::move(specs)){
        requireProvider(!specs_.empty(),"Cross-vendor declaration must include its actual cases");for(const auto& spec:specs_)spec.requireComplete();
    }
    CrossVendorReadReplayPrograms(const CrossVendorReadReplayPrograms&)=delete;
    CrossVendorReadReplayPrograms(CrossVendorReadReplayPrograms&&)noexcept=default;
    /** @brief Preserve original retirement after partial pre-submission construction failure. */
    ~CrossVendorReadReplayPrograms()noexcept{try{retire();}catch(...){std::terminate();}}
    /** @brief Observe only the explicit source/storage endpoints, acquire/freeze their same source, then admit once. */
    void prepare(DeviceId source,DeviceId owner){
        if(session_)throw std::logic_error("Cross-vendor source program preparation is one-shot");
        requireProvider(source.is_gpu() && owner.is_gpu() && source.type!=owner.type,"Cross-vendor program requires opposite-vendor rank-local storage");
        const std::array selected{source,owner};session_=std::shared_ptr<PreparedNativeLookupFixture>(PreparedNativeLookupFixture::prepare(selected));
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        const auto io=RetainedHostServiceCalibrationAccess::program(RetainedHostServiceRole::ImmutableLookupIO);
#else
        const auto evidence=RetainedHostServiceProgram::originalEvidence();
        const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO,evidence);
#endif
        RetainedHostServiceDeclarations::Builder ioDeclarations(0);std::vector<RetainedHostServiceInstance> selectedIO;
        std::vector<CrossVendorReadReplayGeometry> declarations;
        for(std::size_t index=0;const auto& spec:specs_){
            const RetainedHostServiceInstance instance{0,source,RetainedHostServiceRole::ImmutableLookupIO,index++};selectedIO.push_back(instance);ioDeclarations.add(instance,io);
            auto geometry=CrossVendorReadReplayGeometry::create(spec,source,owner,*session_,io);declarations.push_back(geometry);
            programs_.push_back({spec,std::move(geometry),session_,source,owner,{},instance});
        }
        ioDeclarations_=std::move(ioDeclarations).freeze();session_->admit(crossVendorReadProgramBOM(declarations));
        ioBinding_=std::shared_ptr<RetainedHostServiceBinding>(ioDeclarations_->bind(session_->memory(),selectedIO));
        for(std::size_t index=0;auto& program:programs_){program.ioBinding=ioBinding_;
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
            if(program.geometry.remoteProgram)program.calibration=std::make_shared<PooledRemoteGraphCalibration>(session_->permit(owner),*program.geometry.remoteProgram,index,program.spec.handshake);
#endif
            ++index;
        }
        consumed_.assign(programs_.size(),false);
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        nlohmann::json originalKeys=nlohmann::json::array();
        for(const auto device:selected){const auto& original=session_->permit(device);const auto key=original.provenance().originalBirthKey();if(!key)throw std::logic_error("Cross-vendor program lost its original source key");
            originalKeys.push_back({{"device",device.toString()},{"generation",original.generation()},{"birth_key",nativeSourceBirthKeyJson(*key)}});}
        std::cout<<"POOLED_CROSS_SOURCE "<<nlohmann::json{{"qualified",false},{"original_keys",std::move(originalKeys)},
            {"auxiliary_program_family",NativeAuxiliaryResourceCatalog::resourceSourceIdentity()},{"original_pma",reinterpret_cast<std::uintptr_t>(session_->memory().get())}}.dump()<<'\n';
        std::cout<<"POOLED_CROSS_ADMISSION "<<nlohmann::json{{"qualified",false},{"same_original_source",true},{"same_pma",true},
            {"compute_continuation_devices",{source.toString()}},{"storage_owner_devices",{owner.toString()}},
            {"declared_cases",programs_.size()},{"general_source_slots_per_case",3},{"general_remote_slots_per_case",1},
            {"legacy_remote_retirement_events_per_case",1},{"whole_native_program_qualified",false},
            {"unknown_native_populations",{"general_graph_host","activation_publication_and_capture_dependencies","diagnostic_timing_events","cpu_io_service_libc_tls"}}}.dump()<<'\n';
#endif
    }
    /** @return Immutable same-source declaration, never an ordinary getContext/new-PMA compatibility arm. */
    const PreparedCrossVendorReadReplayProgram& program(std::size_t index)const{return programs_.at(index);}
    /** @brief Execute one declared instance; repeat requests reuse its original captures. */
    template<class Body>void run(std::size_t index,Body&& body){
        if(consumed_.at(index))throw std::logic_error("Cross-vendor native instance already consumed its original declaration");
        consumed_[index]=true;const auto& declared=program(index);CrossVendorReadReplayGraphs definitions(declared);
        std::unique_ptr<CrossVendorReadReplay> replay;std::exception_ptr primary;
        try{replay=std::make_unique<CrossVendorReadReplay>(declared,definitions);body(*replay);}
        catch(...){primary=std::current_exception();}
        if(replay && !primary && replay->hasActiveInterval())
            primary=std::make_exception_ptr(std::logic_error("Cross-vendor fixture body omitted its terminal retirement"));
        // Report the first body failure BEFORE cleanup: a genuine native failure
        // must never erase its diagnostic while enforcing strict reader safety.
        const auto text=[](const std::exception_ptr& error){try{std::rethrow_exception(error);}catch(const std::exception& e){return std::string(e.what());}catch(...){return std::string("non-standard cross-vendor fixture failure");}};
        if(primary)std::cerr<<"Cross-vendor original body failure: "<<text(primary)<<'\n';
        std::exception_ptr terminal;
        if(replay){
            replay->timingJoin();
            try{replay->finishOutstandingRequest();}catch(...){terminal=std::current_exception();std::cerr<<"Cross-vendor terminal cleanup failure: "<<text(terminal)<<'\n';}
            // This remains strict: if the original reset/join did not actually
            // retire readers, retireCohort/destruction still reject that state.
            replay->retireCohort();
        }
        definitions.retire();replay.reset();
        session_->requirePayloadRetired();
        if(primary && terminal)throw std::runtime_error(text(primary)+"; cross-vendor terminal cleanup failed: "+text(terminal));
        if(primary)std::rethrow_exception(primary);
        if(terminal)std::rethrow_exception(terminal);
    }
    /** @brief Retire original contexts only after all actual captures/roots and their original permits disappeared. */
    void retire(){if(session_){session_->retire();
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(!retirementReported_)for(const auto& program:programs_)if(program.calibration)program.calibration->checkpoint("original_contexts_reset",{{"physical_release_claimed",false}});
#endif
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(!retirementReported_){for(const auto& result:session_->retirementReceipts()){
            requireProvider(result.runtime_reset_invoked && result.runtime_post_reset_state==DeviceRuntimePostResetState::Quiescent &&
                result.retired_context_generation && result.retired_runtime_generation && result.successor_runtime_generation==result.retired_runtime_generation+1 &&
                !result.canonical_allocations_before_runtime_reset && !result.canonical_allocation_bytes_before_runtime_reset,
                "Cross-vendor original source reset receipt incomplete");
            std::cout<<"POOLED_CROSS_RETIRED "<<nlohmann::json{{"qualified",false},{"device",result.device.toString()},
                {"retired_context_generation",result.retired_context_generation},{"retired_runtime_generation",result.retired_runtime_generation},
                {"successor_runtime_generation",result.successor_runtime_generation},{"runtime_reset_invoked",true},{"runtime_post_reset_state","quiescent"},
                {"canonical_allocations_before_runtime_reset",result.canonical_allocations_before_runtime_reset},
                {"canonical_bytes_before_runtime_reset",result.canonical_allocation_bytes_before_runtime_reset},
                {"general_device_tokens_after_reset",session_->memory()->reservedBytes(result.device,PhysicalMemoryOwner::NativeGraphExecutable)},
                {"physical_release_claimed",false}}.dump()<<'\n';
        }retirementReported_=true;}
#endif
    }}
private:
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    bool retirementReported_=false; ///< Immutable terminal diagnostic publication only, never model progress.
#endif
    std::shared_ptr<const RetainedHostServiceDeclarations> ioDeclarations_;
    std::shared_ptr<RetainedHostServiceBinding> ioBinding_;
    std::vector<CrossVendorReadReplaySpec> specs_;
    std::shared_ptr<PreparedNativeLookupFixture> session_;
    std::vector<PreparedCrossVendorReadReplayProgram> programs_;
    std::vector<bool> consumed_; ///< CPU-owned one-shot cold instance lifetime, never a GPU model/cache mirror.
};
/** @brief Execute one complete original transaction and prove producer/receiver live bytes plus untouched tails. */
void CrossVendorReadReplay::replay(std::span<const std::uint64_t> original,std::uint32_t expectedUniqueHits){
    prepareTiming(original);
    timingLaunch();timingJoin();validateTiming(original,expectedUniqueHits);
}
/** @brief Actual captured source/read/retirement must accept cold prefill without manufacturing response admission. */
void CrossVendorReadReplay::coldPrefill(std::span<const std::uint64_t> original,std::uint32_t expectedUniqueHits){
    prepareTiming(original,0,0);timingLaunch();timingJoin();validateTiming(original,expectedUniqueHits,0);
    const auto cold=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);
    q::QwenAdaptiveCounterTotals before{};ASSERT_TRUE(q::readQwenAdaptiveCounters(cold.data(),cold.size(),before));
    EXPECT_EQ(before.calls,1u);EXPECT_EQ(before.completed_live_ids,original.size());EXPECT_EQ(before.failed_frames,0u);
    initializeResponse(0);
    const auto admitted=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);
    q::QwenAdaptiveCounterTotals after{};ASSERT_TRUE(q::readQwenAdaptiveCounters(admitted.data(),admitted.size(),after));
    EXPECT_EQ(after.calls,before.calls);EXPECT_EQ(after.completed_live_ids,before.completed_live_ids);
    const auto output=arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
    const auto unchanged=observe<float>(0,output->gpu_data_ptr(),expected_.size());
    EXPECT_EQ(std::memcmp(unchanged.data(),expected_.data(),expected_.size()*sizeof(float)),0);
}
/** @brief Real backend launches the production initialize_device_generation_control kernel on the original stream. */
void CrossVendorReadReplay::initializeResponse(int expectedError){
    workers_[0]->submitAndWait([&]{
        sampling_math::DeviceGenerationInitialization initialization;
        initialization.forward_failure_policy=sampling_math::DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;
        // The response owns an actual admitted INT32 row even though this
        // initializer leaves its inactive token storage untouched.
        const auto response=arenas_[0]->getSharedTensor(BufferId::MTP_GENERATION_RESPONSE_TOKENS);
        requireProvider(backends_[0]->enqueueInitializeDeviceGeneration(1,1,sampling_math::DeviceGenerationPolicy::ordinary(),
            sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse,1,response->gpu_data_ptr(),
            sampling_math::kDeviceGenerationControlCount,control_->row(),devices_[0].ordinal,streams_[0],initialization),
            "Cross-vendor actual response initialization failed to enqueue");
        TransferEngine::publishDeviceWrite(control_->owner(),devices_[0],streams_[0]);
    });join();
    const auto row=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);
    EXPECT_EQ(row[sampling_math::kDeviceGenerationControlErrorCode],expectedError);
    EXPECT_EQ(row[sampling_math::kDeviceGenerationControlOk],expectedError?0:1);
}
/** @brief Existing first error must prevent output publication while every owner still releases its pins/reservations. */
void CrossVendorReadReplay::stickyForwardError(std::span<const std::uint64_t> original){
    const int first=static_cast<int>(sampling_math::DeviceGenerationError::InvalidInitialization);
    const auto output=arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
    const auto before=observe<float>(0,output->gpu_data_ptr(),expected_.size());
    prepareTiming(original,0,first);timingLaunch();timingJoin();
    const auto row=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);
    EXPECT_EQ(row[sampling_math::kDeviceGenerationControlOk],0);EXPECT_EQ(row[sampling_math::kDeviceGenerationControlErrorCode],first);
    const auto after=observe<float>(0,output->gpu_data_ptr(),expected_.size());
    EXPECT_EQ(std::memcmp(before.data(),after.data(),before.size()*sizeof(float)),0);
    const auto& plan=owner_->plan().parameters();const auto roots=owner_->roots();
    const auto physical=observe<p::OwnerFrame>(1,static_cast<const std::byte*>(roots.frame->gpu_data_ptr())+plan.frame.control,1).front();
    EXPECT_EQ(physical.held_pins,0u);EXPECT_EQ(physical.reservations,0u);EXPECT_EQ(physical.phase,p::OwnerPhase::Idle);
    initializeResponse(first); // Admission cannot erase the original prefill failure.
    // A failed service is terminal, never a reusable generation. The existing
    // reset close must still retire native readers and join that lane before
    // its precise failure is consumed. No later admission borrows this cohort.
    try {close(true);}
    catch(const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("Immutable lookup retained-generation storage failed:"),std::string::npos);
    }
    EXPECT_FALSE(boundary_->hasActiveInterval());
}
/** @brief Shared original input/canary publication outside timing. @param original Fixed trace. */
void CrossVendorReadReplay::prepareTiming(std::span<const std::uint64_t> original,int healthy,int error){
    requireProvider(original.size()<=Maximum && original.size()%16==0,"Cross-vendor PLE requires its complete original token/head count");
    file_.requireAuthored(original); // Reject a missing independent oracle before any upload/native enqueue.
    workers_[0]->submitAndWait([&]{
        auto ids=arenas_[0]->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arenas_[0]->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY),controller=arenas_[0]->getSharedTensor(BufferId::MTP_GENERATION_CONTROL);
        std::memset(ids->raw_mutable_data(),0xff,ids->size_bytes());if(!original.empty())std::memcpy(ids->raw_mutable_data(),original.data(),original.size_bytes());
        *static_cast<int*>(count->raw_mutable_data())=static_cast<int>(original.size()/16);
        std::memset(controller->raw_mutable_data(),0,controller->size_bytes());static_cast<int*>(controller->raw_mutable_data())[sampling_math::kDeviceGenerationControlOk]=healthy;
        static_cast<int*>(controller->raw_mutable_data())[sampling_math::kDeviceGenerationControlErrorCode]=error;
        for(const auto& root:{ids,count,controller}){TransferEngine::prepareDeviceInput(root.get(),devices_[0],streams_[0]);TransferEngine::requireDeviceInput(root.get(),devices_[0],streams_[0]);}
    });
    // This retired diagnostic boundary refreshes only the two dead packet
    // banks, never private rows, physical cache state or a production count.
    // Every replay starts with distinct sender/receiver inactive bytes, so even
    // a stale full3 copy after a small1/empty0 message necessarily changes a tail.
    ++replayIndex_;
    expectedPacket_.assign(slots_*file_.row_bytes,static_cast<std::byte>(0xa0u+(replayIndex_%16)));
    expectedOwnerPacket_.assign(slots_*file_.row_bytes,static_cast<std::byte>(0x50u+(replayIndex_%16)));
    if(slots_){
        workers_[0]->submitAndWait([&]{const auto roots=boundary_->pooledStorage()->roots();const auto& plan=boundary_->pooledStorage()->geometry().parameters();
            requireProvider(backends_[0]->hostToDeviceOnStream(static_cast<std::byte*>(roots.frame->gpu_data_ptr())+plan.offsets.payload,expectedPacket_.data(),expectedPacket_.size(),devices_[0].ordinal,streams_[0]),"Fresh source packet canary upload failed");TransferEngine::publishDeviceWrite(roots.frame.get(),devices_[0],streams_[0]);});
        workers_[1]->submitAndWait([&]{const auto roots=owner_->roots();const auto& plan=owner_->plan().parameters();
            requireProvider(backends_[1]->hostToDeviceOnStream(static_cast<std::byte*>(roots.native->gpu_data_ptr())+plan.native.source_payload,expectedOwnerPacket_.data(),expectedOwnerPacket_.size(),devices_[1].ordinal,streams_[1]),"Fresh owner packet canary upload failed");TransferEngine::publishDeviceWrite(roots.native.get(),devices_[1],streams_[1]);});
        join(); // Retire uploads before these persistent canaries become result oracles.
    }
}
/** @brief Shared actual native byte/extent oracle, outside timing. @param original Fixed trace. @param expectedUniqueHits Prescribed cache-history witness. */
void CrossVendorReadReplay::validateTiming(std::span<const std::uint64_t> original,std::optional<std::uint32_t> expectedUniqueHits,int expectedHealthy){
    const auto co=coordinator_->roots(),ow=owner_->roots(),src=boundary_->pooledStorage()->roots();const auto& cp=coordinator_->plan().parameters();const auto& op=owner_->plan().parameters();const auto& sp=boundary_->pooledStorage()->geometry().parameters();
    for(unsigned i=0;i<2;++i)workers_[i]->submitAndWait([&,i]{const auto roots=i?ow:co;for(const auto& root:{roots.cache,roots.frame,roots.native})if(root)TransferEngine::publishDeviceWrite(root.get(),devices_[i],streams_[i]);
        if(!i){for(const auto& root:{src.frame,src.native,arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING),arenas_[0]->getSharedTensor(BufferId::MTP_GENERATION_CONTROL)})TransferEngine::publishDeviceWrite(root.get(),devices_[0],streams_[0]);}});
    const auto lookup=observe<p::LookupFrame>(0,static_cast<const std::byte*>(co.frame->gpu_data_ptr())+cp.frame.control,1).front();
    EXPECT_EQ(lookup.phase,p::LookupPhase::Idle);EXPECT_EQ(lookup.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(lookup.original_ids,original.size());
    const auto unique=observe<p::CoordinatorUnique>(0,static_cast<const std::byte*>(co.frame->gpu_data_ptr())+cp.frame.unique,lookup.unique_rows);
    const auto coordinates=observe<std::uint32_t>(0,static_cast<const std::byte*>(src.frame->gpu_data_ptr())+sp.offsets.coordinates,lookup.unique_rows);
    const auto requests=observe<n::DirectedSlice>(0,static_cast<const std::byte*>(co.native->gpu_data_ptr())+cp.native.pin_slices,1).front();
    std::uint32_t hits=0,misses=0;std::vector<p::Grant> expectedGrants;
    for(const auto& row:unique)if(row.phase==p::UniquePhase::Read){++hits;expectedGrants.push_back(row.grant);}else ++misses;
    // Correctness callers prescribe exact hits; adaptive timing declares only input
    // history. Packet extents always match the independently enumerated Read grants.
    if(expectedUniqueHits){EXPECT_EQ(hits,*expectedUniqueHits);if(*expectedUniqueHits)EXPECT_GT(hits,0u);}
    if(original.empty())EXPECT_EQ(hits,0u);
    EXPECT_EQ(requests.live_bytes,std::uint64_t{hits}*sizeof(p::Grant));
    const auto ingress=observe<n::OwnerIngress>(1,static_cast<const std::byte*>(ow.native->gpu_data_ptr())+op.native.ingress,1).front();
    const auto received=observe<std::uint64_t>(0,static_cast<const std::byte*>(src.native->gpu_data_ptr())+sp.offsets.payload_bytes,1).front();
    EXPECT_EQ(ingress.payload_bytes,std::uint64_t{hits}*file_.row_bytes);EXPECT_EQ(received,ingress.payload_bytes);
    if(slots_){EXPECT_EQ(ingress.record_bytes,requests.live_bytes);const auto grants=observe<p::Grant>(1,static_cast<const std::byte*>(ow.frame->gpu_data_ptr())+op.frame.commands,hits);
        ASSERT_EQ(grants.size(),expectedGrants.size());for(size_t i=0;i<grants.size();++i)EXPECT_EQ(std::memcmp(&grants[i],&expectedGrants[i],sizeof(p::Grant)),0);
    }else {EXPECT_EQ(hits,0u);EXPECT_EQ(requests.live_bytes,0u);}
    const auto controller=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);
    EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlOk],expectedHealthy);EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlErrorCode],0);
    const auto frontier=observe<n::SourceFrontier>(0,static_cast<const std::byte*>(src.native->gpu_data_ptr())+sp.offsets.frontier,1).front();EXPECT_EQ(frontier.status,q::adaptive_lookup::Status::Success);
    const auto io=observe<q::ImmutableLookupDeviceState>(0,src.native->gpu_data_ptr(),1).front();EXPECT_EQ(io.status,0u);EXPECT_EQ(io.idCount,misses);EXPECT_EQ(io.nativeBytes,std::uint64_t{misses}*file_.row_bytes);EXPECT_EQ(io.issuedSequence,io.consumedSequence);
    // Preserve the private miss prefix and append only initial hits, exactly
    // as the actual source planner/import bodies specify. These vectors are
    // terminal test oracles and are never uploaded or used to drive execution.
    std::uint32_t privateRow=0;
    for(const auto& row:unique)if(row.phase!=p::UniquePhase::Read){const auto bytes=file_.original(row.row);std::memcpy(expectedResponses_.data()+std::size_t{privateRow++}*file_.row_bytes,bytes.data(),bytes.size());}
    for(const auto& row:unique)if(row.phase==p::UniquePhase::Read){const auto bytes=file_.original(row.row);std::memcpy(expectedResponses_.data()+std::size_t{privateRow++}*file_.row_bytes,bytes.data(),bytes.size());}
    EXPECT_EQ(privateRow,lookup.unique_rows);
    for(size_t i=0;i<unique.size();++i){ASSERT_LT(coordinates[i],lookup.unique_rows);const auto bytes=file_.original(unique[i].row);EXPECT_EQ(std::memcmp(expectedResponses_.data()+std::size_t{coordinates[i]}*file_.row_bytes,bytes.data(),bytes.size()),0);}
    const auto rows=observe<std::byte>(0,static_cast<const std::byte*>(src.frame->gpu_data_ptr())+sp.offsets.responses,expectedResponses_.size());EXPECT_EQ(rows,expectedResponses_);
    // Before ReadRows, the real Fill packet updates only this owner's actual
    // reservation rows. Live hit rows then replace its prefix. Distinct initial
    // source/owner payload poison makes an accidental capacity copy observable.
    std::vector<const p::CoordinatorUnique*> fills;
    for(const auto& row:unique)if(row.phase==p::UniquePhase::FreeGranted || row.phase==p::UniquePhase::VictimGranted)fills.push_back(&row);
    std::sort(fills.begin(),fills.end(),[](const auto* a,const auto* b){const auto waveA=a->phase==p::UniquePhase::VictimGranted,waveB=b->phase==p::UniquePhase::VictimGranted;return waveA!=waveB?waveA<waveB:a->decision_index<b->decision_index;});
    std::uint32_t packetRow=0;
    for(const auto* row:fills){const auto bytes=file_.original(row->row);
        std::memcpy(expectedPacket_.data()+std::size_t{packetRow}*file_.row_bytes,bytes.data(),bytes.size());std::memcpy(expectedOwnerPacket_.data()+std::size_t{packetRow++}*file_.row_bytes,bytes.data(),bytes.size());}
    packetRow=0;for(const auto& row:unique)if(row.phase==p::UniquePhase::Read){const auto bytes=file_.original(row.row);std::memcpy(expectedPacket_.data()+std::size_t{packetRow}*file_.row_bytes,bytes.data(),bytes.size());std::memcpy(expectedOwnerPacket_.data()+std::size_t{packetRow++}*file_.row_bytes,bytes.data(),bytes.size());}
    if(slots_){EXPECT_EQ(observe<std::byte>(0,static_cast<const std::byte*>(src.frame->gpu_data_ptr())+sp.offsets.payload,expectedPacket_.size()),expectedPacket_);
        EXPECT_EQ(observe<std::byte>(1,static_cast<const std::byte*>(ow.native->gpu_data_ptr())+op.native.source_payload,expectedOwnerPacket_.size()),expectedOwnerPacket_);}
    if(!original.empty()){std::vector<std::byte> native(original.size()*file_.row_bytes);for(size_t i=0;i<original.size();++i){const auto bytes=file_.original(original[i]);std::memcpy(native.data()+i*file_.row_bytes,bytes.data(),bytes.size());}
        q::decodeCPUImmutableLookupRows(file_.type,file_.columns,file_.row_bytes,native.data(),original.size(),expected_.data());}
    const auto output=arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);const auto actualOutput=observe<float>(0,output->gpu_data_ptr(),expected_.size());EXPECT_EQ(std::memcmp(actualOutput.data(),expected_.data(),expected_.size()*sizeof(float)),0);
    const auto frame=observe<p::OwnerFrame>(1,static_cast<const std::byte*>(ow.frame->gpu_data_ptr())+op.frame.control,1).front();EXPECT_EQ(frame.phase,p::OwnerPhase::Idle);EXPECT_EQ(frame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(frame.held_pins,0u);EXPECT_EQ(frame.reservations,0u);
    q::QwenAdaptiveCounterTotals counters{};ASSERT_TRUE(q::readQwenAdaptiveCounters(controller.data(),controller.size(),counters));
    lastObservation={original.size(),lookup.unique_rows,hits,io.idCount,counters.promotions,counters.evictions,io.nativeBytes,ingress.payload_bytes};
    EXPECT_EQ(read_->captureIdentity(),identity_);std::cout<<"POOLED_CROSS_READ source="<<devices_[0].toString()<<" owner="<<devices_[1].toString()<<" native_format="<<static_cast<unsigned>(file_.format)<<" row_bytes="<<file_.row_bytes<<" original="<<original.size()<<" unique="<<lookup.unique_rows<<" private_misses="<<misses<<" hit_grant_bytes="<<requests.live_bytes<<" sent_hit_bytes="<<ingress.payload_bytes<<" received_hit_bytes="<<received<<" capacity_rows="<<slots_<<" captured=1 native_peer_factory_qualified=0\n";
}
/** @brief Test terminal input names the original completed generation; close itself is captured and device-validated. */
void CrossVendorReadReplay::close(bool explicitReset){workers_[0]->submitAndWait([&]{requireProvider(backends_[0]->hostToDeviceOnStream(control_->row()+sampling_math::kDeviceGenerationControlRequestComplete,&terminalComplete_,sizeof(terminalComplete_),devices_[0].ordinal,streams_[0]),"Cross-vendor terminal input failed");requireProvider(definitions_.source.launch(explicitReset?reset_:terminal_),"Cross-vendor retained close/reset failed");});join();boundary_->awaitTerminal();}
/** @brief Record the actual complete original two-participant DAG, including final reader acknowledgment and physical retirement. */
void CrossVendorReadReplay::build(){
    PooledPinParticipantNodes ownerVictims,coordinatorVictims;
    workers_[0]->submitAndWait([&]{
        const auto ids=arenas_[0]->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arenas_[0]->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
        const auto& cp=coordinator_->plan().parameters();const auto& sp=boundary_->pooledStorage()->geometry().parameters();q::PooledNativeOperandSpans inputs{};
        inputs[static_cast<size_t>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,Maximum*sizeof(std::uint64_t)};
        inputs[static_cast<size_t>(q::PooledNativeOperand::OriginalCount)]=q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,sizeof(int)};
        inputs[static_cast<size_t>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{boundary_->pooledStorage()->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,sp.offsets.protocol+offsetof(q::ImmutableLookupDeviceState,admissionGeneration),sizeof(std::uint64_t)};
        const auto rows=DeviceRowRange::deviceCounted(Maximum/16,static_cast<const int*>(count->gpu_data_ptr())).repeatEachRow(16);
        graphs_[0]->addNode("cross/begin",std::make_unique<PooledImmutableNativePhaseStage>(*arenas_[0],code_[0],PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::Begin,coordinator_,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::Begin,coordinator_->plan(),n::OriginalRows{rows,n::OriginalIdWidth::UInt64},inputs)}),devices_[0]);
        graphs_[0]->addNode("cross/commands",std::make_unique<PooledImmutableNativePhaseStage>(*arenas_[0],code_[0],PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::PinCommands,coordinator_,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::PinCommands,coordinator_->plan(),{},{})}),devices_[0]);graphs_[0]->addDependency("cross/commands","cross/begin");
        const auto sent=appendPooledPinSender(*graphs_[0],*arenas_[0],forward_,"cross/begin","cross/commands","cross/pin_send");
        const std::array returns{returned_};const std::array sentNodes{sent.last};const auto pin=appendPooledPinCoordinatorReturn(*graphs_[0],*arenas_[0],returns,code_[0],sentNodes,"cross/commands","cross/pin_return");
        const std::array frees{free_};const auto free=appendPooledFreeCoordinatorWave(*graphs_[0],*arenas_[0],frees,code_[0],pin.last,"cross/free");
        const std::array victims{victim_};coordinatorVictims=appendPooledVictimCoordinatorWave(*graphs_[0],*arenas_[0],victims,code_[0],free.last,"cross/victims");
    });
    workers_[1]->submitAndWait([&]{
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(originalProgram_.geometry.remoteProgram)return;
#endif
        const auto received=appendPooledPinReceiver(*graphs_[1],*arenas_[1],forward_,code_[1],"cross/pin");
        const auto returned=appendPooledPinOwnerReturn(*graphs_[1],*arenas_[1],returned_,code_[1],received.last,"cross/pin_return");
        const auto free=appendPooledFreeOwnerWave(*graphs_[1],*arenas_[1],free_,code_[1],returned.last,"cross/free");ownerVictims=appendPooledVictimOwnerWave(*graphs_[1],*arenas_[1],victim_,code_[1],free.last,"cross/victims");
    });
    workers_[0]->submitAndWait([&]{
        const std::array sourceMembers{q::PooledImmutableSourceParticipant{source_}};
        const auto fill=appendPooledSourceCoordinatorMembers(*graphs_[0],*arenas_[0],sourceMembers,boundary_,code_[0],*control_,coordinatorVictims.last,"cross/source");
        const std::array publicationMembers{q::PooledImmutablePublicationParticipant{publication_}};const auto ready=appendPooledPublicationCoordinator(*graphs_[0],*arenas_[0],fill,publicationMembers,code_[0],"cross/publication");
        const auto output=arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);const std::array readMembers{q::PooledImmutableReadParticipant{read_}};
        const auto read=appendPooledReadCoordinator(*graphs_[0],*arenas_[0],ready,readMembers,{output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()},"cross/read","cross/original_order_ple");
        requirePooledReadCompletion(*graphs_[0],read);retirement_=q::PooledImmutableCapturedRetirementWave::bind(read_,memory_);
        const std::array retiredMembers{q::PooledImmutableRetirementParticipant{retirement_}};completion_=appendPooledRetirementCoordinator(*graphs_[0],*arenas_[0],read,retiredMembers,code_[0],"cross/retirement");requirePooledRetirementCompletion(*graphs_[0],completion_);
        graphs_[0]->setRetainedStorageBoundary(boundary_);
    });
    workers_[1]->submitAndWait([&]{
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        if(originalProgram_.geometry.remoteProgram){
            const q::PooledImmutableRemoteTransport remote{0,forward_,returned_,free_,victim_,source_,publication_,read_,retirement_};
            ownerRetired_=appendPooledImmutableDirectedOwnerGraph(*graphs_[1],*arenas_[1],remote,code_[1],*originalProgram_.geometry.remoteProgram,"cross/authored_owner");return;
        }
#endif
        const auto fill=appendPooledSourceOwnerFill(*graphs_[1],*arenas_[1],source_,code_[1],ownerVictims.last,"cross/source");
        const auto ready=appendPooledPublicationOwner(*graphs_[1],*arenas_[1],publication_,code_[1],fill.last,"cross/publication");
        const auto read=appendPooledReadOwner(*graphs_[1],*arenas_[1],read_,code_[1],ready.last,"cross/read");
        ownerRetired_=appendPooledRetirementOwner(*graphs_[1],*arenas_[1],retirement_,code_[1],read.last,"cross/retirement");requirePooledRetirementOwnerCompletion(*graphs_[1],retirement_,ownerRetired_.last);
    });
    // Only the original cohort owner creates and records the actual programs.
    // No caller executable (including an unrelated same-stream graph) is accepted.
    q::PooledImmutableCohortParticipant sourceProgram{workers_[0],contexts_[0],arenas_[0],graphs_[0],streams_[0],definitions_.source.definition(0)};
    std::vector<q::PooledImmutableCohortRemote> ownerPrograms;
    ownerPrograms.push_back({0,{workers_[1],contexts_[1],arenas_[1],graphs_[1],streams_[1],definitions_.owner.definition(0)},ownerRetired_.last});
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    if(originalProgram_.calibration)originalProgram_.calibration->checkpoint("before_cohort_recording");
#endif
    cohort_=q::PooledImmutableCompleteCohort::bind(std::move(sourceProgram),completion_,std::move(ownerPrograms));
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    if(originalProgram_.calibration)originalProgram_.calibration->checkpoint("after_cohort_recording");
#endif
    for(unsigned i=0;i<2;++i){
        const auto nodes=i?cohort_->ownerKernelInventory(0):cohort_->sourceKernelInventory();
        requireProvider(!nodes.empty(),"Cross-vendor original native program has no kernel inventory");
        for(const auto& node:nodes){requireProvider(node.valid(),"Cross-vendor actual native kernel inventory incomplete");std::cout<<"POOLED_CROSS_READ_RESOURCE participant="<<devices_[i].toString()<<" name="<<node.name<<" threads="<<node.block_x<<" registers="<<node.registers_per_thread<<" local_bytes="<<node.local_memory_bytes_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes<<" active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';}
    }
    workers_[0]->submitAndWait([&]{terminal_=definitions_.source.capture(1);requireProvider(bool(terminal_),"Cross-vendor retained terminal unavailable");
        {ScopedBackendGraphCapture capture(*workers_[0],*terminal_,"cross-vendor original complete-parent close");requireProvider(capture.begin(),"Cross-vendor close capture begin failed");boundary_->recordTerminalClose(streams_[0],control_->healthy(),control_->row()+sampling_math::kDeviceGenerationControlRequestComplete);capture.finish();}
        std::string detail;requireProvider(terminal_->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());requireProvider(terminal_->instantiate(),"Cross-vendor retained close instantiation failed");
    });
    workers_[0]->submitAndWait([&]{reset_=definitions_.source.capture(2);requireProvider(bool(reset_),"Cross-vendor retained reset unavailable");
        {ScopedBackendGraphCapture capture(*workers_[0],*reset_,"cross-vendor original typed complete-cohort reset");requireProvider(capture.begin(),"Cross-vendor reset capture begin failed");boundary_->recordTerminalClose(streams_[0]);capture.finish();}
        std::string detail;requireProvider(reset_->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());requireProvider(reset_->instantiate(),"Cross-vendor retained reset instantiation failed");
        // Closing an initialized idle source before its first admission must not
        // create a ticket, consume a row or require an armed service.
        requireProvider(definitions_.source.launch(reset_),"Cross-vendor initial idle reset failed");
    });join();
    requireProvider(!boundary_->hasActiveInterval(),"Cold reset invented an active service interval");
}
/**
 * @brief One explicit capacity shard retains every original format and source/read/reset oracle.
 * @param sourceDevice Actual original continuation endpoint.
 * @param ownerDevice Actual opposite-vendor storage-only endpoint.
 * @param selectedSlots Explicit C0 or positive3 geometry; each native registration keeps its30s deadline.
 * @details Capacity shards divide independent cases, never a transaction, format or replay trajectory.
 */
void crossVendorReadFormats(DeviceId sourceDevice,DeviceId ownerDevice,std::uint32_t selectedSlots){
    requireProvider(selectedSlots==0 || selectedSlots==3,"All-format reader requires explicit C0 or positive geometry");
    if(!hasCPUBackend())initCPUBackend(-1);
    std::vector<CrossVendorReadReplaySpec> cases;std::size_t supported=0,unsupported=0;
    for(const auto& format:embeddingVerifierFormats()){
        const std::size_t columns=format.prepared_native_quantized?768:257;auto original=format.create({97,columns},1701);const auto gguf=providerFormat(original->native_type());
        if(!gguf){requireProvider(original->native_type()==TensorType::Q8_1,"Original source format disappeared from cross-vendor coverage");++unsupported;continue;}
        ++supported;auto file=std::make_shared<ProviderFile>(*original,*gguf,columns);cases.push_back({file,selectedSlots});
    }
    const auto sparse=std::make_shared<ProviderFile>();
    EXPECT_NE(std::memcmp(sparse->original(7).data(),sparse->original(ProviderFile::High).data(),sparse->row_bytes),0);
    const auto sparseBegin=cases.size();cases.push_back({sparse,selectedSlots});
    CrossVendorReadReplayPrograms programs(std::move(cases));programs.prepare(sourceDevice,ownerDevice);
    for(std::size_t index=0;index<supported;++index){const auto slots=programs.program(index).spec.slots;
        programs.run(index,[&](CrossVendorReadReplay& replay){
            std::vector<std::uint64_t> warm(16,1),large(64,1),fullHits(64,1),small(16,2);for(std::size_t i=0;i<warm.size();++i)if(i%2)warm[i]=2;
            for(std::size_t i=0;i<large.size();++i)large[i]=1+i%5;for(std::size_t i=0;i<fullHits.size();++i)fullHits[i]=1+i%3;
            for(std::uint64_t generation=1;generation<=3;++generation){replay.admit(generation);replay.coldPrefill(warm,slots?(generation==1?0u:2u):0u);replay.replay(large,slots?(generation==1?2u:3u):0u);replay.replay(fullHits,slots?3u:0u);replay.replay(small,slots?1u:0u);replay.replay({},0);if(generation==3)replay.stickyForwardError(warm);else replay.close(generation==2);}
        });
    }
    // Sparse high-row storage is part of the same predeclared native/PMA
    // population, not a second source adopted after process-retained graphs.
    for(std::size_t index=sparseBegin;index<sparseBegin+1;++index){const auto slots=programs.program(index).spec.slots;programs.run(index,[&](CrossVendorReadReplay& replay){
        std::vector<std::uint64_t> warm(16,7),large(64,7),full(64,7),small(16,ProviderFile::High);
        const std::array<std::uint64_t,5> rows{7,ProviderFile::High,700,701,900};for(std::size_t i=0;i<warm.size();++i)if(i%2)warm[i]=ProviderFile::High;
        for(std::size_t i=0;i<large.size();++i)large[i]=rows[i%rows.size()];for(std::size_t i=0;i<full.size();++i)full[i]=rows[i%3];
        for(std::uint64_t generation=1;generation<=2;++generation){replay.admit(generation);replay.replay(warm,slots?(generation==1?0u:2u):0u);replay.replay(large,slots?(generation==1?2u:3u):0u);replay.replay(full,slots?3u:0u);replay.replay(small,slots?1u:0u);replay.replay({},0);replay.close(generation==2);}
    });}
    programs.retire();EXPECT_EQ(unsupported,1u);EXPECT_EQ(supported+unsupported,embeddingVerifierFormats().size());
}
/** @brief Focused first-error regression on the same real provider/source/cohort path, without an all-format campaign. */
void crossVendorColdPrefillLifecycle(DeviceId sourceDevice,DeviceId ownerDevice){
    if(!hasCPUBackend())initCPUBackend(-1);
    const auto file=std::make_shared<ProviderFile>();CrossVendorReadReplayPrograms programs({{file,0},{file,3},{file,0}});
    programs.prepare(sourceDevice,ownerDevice);
    // An original body exception after admission must survive original captured
    // reset/join cleanup, not be replaced by an active-cohort destructor abort.
    try{programs.run(2,[](CrossVendorReadReplay& replay){replay.admit(1);throw std::runtime_error("recorded failure before first lookup");});FAIL()<<"Missing original fixture failure";}
    catch(const std::runtime_error& error){EXPECT_STREQ(error.what(),"recorded failure before first lookup");}
    for(std::size_t index=0;index<2;++index){const auto slots=programs.program(index).spec.slots;
        programs.run(index,[&](CrossVendorReadReplay& replay){
            const std::vector<std::uint64_t> missing(16,1);
            EXPECT_THROW(replay.prepareTiming(missing,0,0),std::invalid_argument);
            std::vector<std::uint64_t> warm(16,7),small(16,ProviderFile::High);
            for(std::size_t i=0;i<warm.size();++i)if(i%2)warm[i]=ProviderFile::High;
            for(std::uint64_t generation=1;generation<=3;++generation){
                replay.admit(generation);replay.coldPrefill(warm,slots && generation>1?2u:0u);
                replay.replay(warm,slots?2u:0u);replay.replay(small,slots?1u:0u);replay.replay({},0);
                if(generation==3)replay.stickyForwardError(warm);
                else replay.close(generation==2);
            }
        });
    }
    programs.retire();
}
