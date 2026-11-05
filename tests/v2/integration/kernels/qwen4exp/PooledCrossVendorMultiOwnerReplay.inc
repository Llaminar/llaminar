/**
 * @file PooledCrossVendorMultiOwnerReplay.inc
 * @brief Full original source/read/retirement capture over two or four remote owners.
 *
 * Every edge is an existing real cross-vendor TransferEngine channel. Owners
 * retain disjoint roots under one original PMA, explicit streams and private CompleteCohort
 * captures. Host copies are diagnostic ingress/terminal oracles only. Native
 * same-vendor factories and real model serving are not certified here. Every
 * worker is created by the original prepared source before payload admission;
 * payloads and retained native definitions consume that same original PMA. The existing one-owner/performance fixture
 * remains unchanged so its workload and observations stay comparable.
 */
class CrossVendorMultiOwnerReplay final {
public:
    /** @brief Admit exact original roots for the ordered cross-vendor owner cohort before materialization. */
    CrossVendorMultiOwnerReplay(PreparedNativeLookupFixture& session,ProviderFile& file,DeviceId sourceDevice,
        std::vector<DeviceId> ownerDevices,std::vector<std::uint32_t> slots)
        :session_(session),file_(file),slots_(std::move(slots)) {
        requireProvider(ownerDevices.size()==slots_.size() && (slots_.size()==2 || slots_.size()==4),"Multi-owner fixture requires two or four original owners");
        devices_.push_back(sourceDevice);
        for(const auto device:ownerDevices){requireProvider(device.is_gpu() && sourceDevice.is_gpu() && device.type!=sourceDevice.type && std::find(devices_.begin(),devices_.end(),device)==devices_.end(),"Multi-owner fixture requires unique actual cross-vendor owners");devices_.push_back(device);}
        const auto count=devices_.size();workers_.resize(count);backends_.resize(count);streams_.resize(count);arenas_.resize(count);graphs_.resize(count);contexts_.resize(count);code_.resize(count);
        for(std::size_t i=0;i<count;++i)
            workers_[i]=session_.permit(devices_[i]).worker();

        struct ConstructionRetirement {CrossVendorMultiOwnerReplay& self;~ConstructionRetirement(){self.join();}} retirement{*this};
        const PhysicalMemoryAllocatorIdentity sourceId{0,sourceDevice},hostId{0,DeviceId::cpu()};
        std::vector<PhysicalMemoryAllocatorIdentity> ownerIds;std::vector<ImmutableRowTableLayoutOwner> layoutOwners;
        bool positive=false;for(std::size_t i=0;i<slots_.size();++i){ownerIds.push_back({0,devices_[i+1]});layoutOwners.push_back({{ownerIds.back(),ImmutableRowStorage::VRAM},0,slots_[i],{}});positive|=slots_[i]!=0;}
        layoutOwners.push_back({{hostId,ImmutableRowStorage::RAM},0,2,{}});
        const auto source=file_.descriptor();const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,std::move(layoutOwners),{sourceId},{},{{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},positive?std::optional(sourceId):std::nullopt,Unique);
        const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,ownerIds,sourceId);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},Maximum);
        const auto cpu=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,sourceId);
        const auto src=session_.resource(sourceDevice),host=session_.resource(DeviceId::cpu());
        auto io=llaminar2::test::RetainedHostServiceFixture::original(RetainedHostServiceRole::ImmutableLookupIO,{0,sourceDevice});
        PhysicalMemoryPlanBuilder bom;const auto add=[&](const PhysicalMemoryPlan& plan){for(const auto& resource:plan.resources())bom.add(resource);};
        add(io.bom(host));add(cpu.bom(host));add(policy.policyBOM(src));add(policy.mandatoryBOM(src));add(q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,src));add(q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,source,src));
        const auto ordering=NativeOrderingEventBudgetProjection::installed();
        add(ordering.bom(q::PooledImmutableCoordinatorNativePlan::orderingProgram(physical,policy),host,src));
        add(ordering.bom(q::PooledImmutableSourceGeometry::orderingProgram(physical,policy,source),host,src));
        add(ordering.bom(NativeOrderingEventProgram::lookupSubmissionRetirement(),host,src));
        std::vector<PhysicalMemoryResource> remoteOrdering;
        for(std::size_t i=0;i<slots_.size();++i)remoteOrdering.push_back(session_.resource(devices_[i+1]));
        add(q::PooledImmutableCompleteCohort::submissionRetirementBOM(host,remoteOrdering));
        for(std::size_t i=0;i<slots_.size();++i){const auto own=session_.resource(devices_[i+1]);add(physical.cacheBOM(i,own));add(physical.mandatoryBOM(i,own));add(q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,i,own));
            add(ordering.bom(q::PooledImmutableOwnerNativePlan::orderingProgram(physical,policy,i),host,own));
            const auto edge=q::PooledImmutablePinEdge::fromPlans(q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,701,17),q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,i,701,17));
            for(const auto channel:{edge.channelMemory(),edge.returnChannelMemory()}){bom.add(host,PhysicalMemoryOwner::ActivationTransportStaging,channel.mapped_host_bytes);bom.add(src,PhysicalMemoryOwner::ActivationTransportStaging,channel.cursor_bytes_per_device);bom.add(own,PhysicalMemoryOwner::ActivationTransportStaging,channel.cursor_bytes_per_device);}
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
            bom.add(own,PhysicalMemoryOwner::NativeGraphExecutable,
                GPUGraphMemoryContract::reservationBytesPerExecutable(devices_[i+1],GPUGraphExecutableClass::General));
#else
            add(RetainedParticipantGraphExecutable::generalBOM(host,own,1));
#endif
        }
        const auto channel=q::ImmutableLookupLayout::createSubordinate(Unique,source.rowBytes());bom.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
        const std::array declarations{CPUTensorStorageDeclaration(1,Maximum*2,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),CPUTensorStorageDeclaration(Maximum/16,16*file_.columns,BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
        const auto activation=Maximum*sizeof(std::uint64_t)+sizeof(int)+sampling_math::kDeviceGenerationControlCount*sizeof(int)+Maximum*file_.columns*sizeof(float);
        bom.add(src,PhysicalMemoryOwner::ActivationArena,activation);for(const auto& declaration:declarations)bom.add(host,PhysicalMemoryOwner::ActivationArena,declaration.physicalBytes());
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
        bom.add(src,PhysicalMemoryOwner::NativeGraphExecutable,
            3*GPUGraphMemoryContract::reservationBytesPerExecutable(sourceDevice,GPUGraphExecutableClass::General));
#else
        add(RetainedParticipantGraphExecutable::generalBOM(host,src,3));
#endif
        session_.admit(bom.build());
        memory_=session_.memory();
        for(std::size_t i=0;i<count;++i){
            const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(devices_[i],memory_);
            original.requireWorker(workers_[i]);
            arenas_[i]=std::make_shared<BufferArena>();graphs_[i]=std::make_shared<ComputeGraph>();
            backends_[i]=getBackendFor(devices_[i]);requireProvider(backends_[i],"Multi-owner backend unavailable");
            workers_[i]->submitAndWait([&,i]{
                streams_[i]=ExplicitGPUStream(workers_[i]->defaultStream()).get();
                contexts_[i]=IDeviceContext::create(devices_[i],1);code_[i]=bundle(devices_[i]);
            });
            definitions_.push_back(std::make_unique<ProviderReplayGraphs>(session_.permit(devices_[i]),i==0?3:1));
        }
        leases_.push_back(memory_->claimNewAllocation(sourceDevice,PhysicalMemoryOwner::ActivationArena,activation));
        provider_=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,cpu,memory_));service_=q::ImmutableLookupService::create(provider_,channel,std::array{sourceDevice},memory_,DeviceId::cpu(),io.take(memory_));
        const auto binding=service_->bindSubordinate(sourceDevice);const auto table=binding.deviceView().base.table,generation=binding.storageGeneration();const auto cp=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,table,generation);
        workers_[0]->submitAndWait([&]{coordinator_=q::PooledImmutableCoordinatorPreparedStorage::create(cp,memory_,streams_[0]);code_[0]->initialize(*coordinator_,streams_[0]);});
        std::vector<q::PooledImmutableOwnerNativePlan> plans;
        for(std::size_t i=0;i<slots_.size();++i){plans.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,i,table,generation));std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner;workers_[i+1]->submitAndWait([&,i]{owner=q::PooledImmutableOwnerPreparedStorage::create(plans[i],memory_,streams_[i+1]);code_[i+1]->initialize(*owner,streams_[i+1]);bindPooledNativeRoots(*arenas_[i+1],*owner);});owners_.push_back(std::move(owner));}
        workers_[0]->submitAndWait([&]{boundary_=q::ImmutableLookupExecutionBoundary::createPooled(service_,q::PooledImmutableSourceGeometry::fromPlans(plans,cp,source,1),owners_,coordinator_,memory_,streams_[0]);
            const std::array roles{BufferId::QFN_NGRAM_ROW_IDS,BufferId::PREFILL_CHUNK_GEOMETRY,BufferId::MTP_GENERATION_CONTROL,BufferId::QFN_PLE_EMBEDDING};for(std::size_t i=0;i<roles.size();++i){std::shared_ptr<TensorBase> root(declarations[i].materialize(memory_));if(i==3)std::fill_n(static_cast<float*>(root->raw_mutable_data()),Maximum*file_.columns,-12345.f);TransferEngine::prepareDeviceInput(root.get(),sourceDevice,streams_[0]);TransferEngine::requireDeviceInput(root.get(),sourceDevice,streams_[0]);requireProvider(arenas_[0]->registerExternalBuffer(roles[i],root),"Multi-owner activation role replaced");}
            bindPooledNativeRoots(*arenas_[0],*coordinator_);bindPooledSourceRoots(*arenas_[0],*boundary_);control_.emplace(DeviceGenerationControlBinding::admit(*arenas_[0],sourceDevice,0));boundary_->bindPooledGenerationControl(*control_);
            for(std::size_t i=0;i<owners_.size();++i){forward_.push_back(q::PooledImmutableCapturedPinEdge::bind(coordinator_,owners_[i],memory_,streams_[0],streams_[i+1]));returned_.push_back(q::PooledImmutableCapturedPinReturn::bind(forward_[i],memory_,streams_[i+1],streams_[0]));free_.push_back(q::PooledImmutableCapturedFreeWave::bind(returned_[i],memory_));victim_.push_back(q::PooledImmutableCapturedVictimWave::bind(free_[i],memory_));source_.push_back(q::PooledImmutableCapturedSourceWave::bind(victim_[i],boundary_,memory_));publication_.push_back(q::PooledImmutableCapturedPublicationWave::bind(source_[i],memory_));read_.push_back(q::PooledImmutableCapturedReadWave::bind(publication_[i],memory_));}});
        expected_.assign(Maximum*file_.columns,-12345.f);expectedResponses_.assign(Unique*file_.row_bytes,std::byte{0xa5});expectedPackets_.resize(owners_.size());
        workers_[0]->submitAndWait([&]{const auto root=boundary_->pooledStorage()->roots().frame;std::vector<std::byte> poison(root->size_bytes(),std::byte{0xa5});requireProvider(backends_[0]->hostToDeviceOnStream(root->gpu_data_ptr(),poison.data(),poison.size(),sourceDevice.ordinal,streams_[0]),"Multi-owner source poison failed");TransferEngine::publishDeviceWrite(root.get(),sourceDevice,streams_[0]);requireProvider(backends_[0]->synchronizeStream(streams_[0],sourceDevice.ordinal),"Multi-owner source poison did not retire");});
        build();
    }
    /** @brief Drain actual submissions before member aliases and source backing retire. */
    ~CrossVendorMultiOwnerReplay(){
        join();if(boundary_ && boundary_->hasActiveInterval())std::terminate();
        cohort_.reset();
        try{for(const auto& definition:definitions_)definition->retire();}catch(...){std::terminate();}
    }
    /** @brief Arm the original generation only outside capture. */
    void admit(std::uint64_t generation){workers_[0]->submitAndWait([&]{boundary_->beginAdmission(generation,streams_[0]);});join();}
    /** @brief Run the complete actual captured transaction and terminal byte/extent oracle. */
    void replay(std::span<const std::uint64_t> ids);
    std::vector<std::size_t> lastHits,lastFills; ///< Terminal diagnostics only; never protocol inputs.
    /** @brief Close/reset through the original captured terminal after all participants retire. */
    void close(bool reset){workers_[0]->submitAndWait([&]{requireProvider(backends_[0]->hostToDeviceOnStream(control_->row()+sampling_math::kDeviceGenerationControlRequestComplete,&terminalComplete_,sizeof(terminalComplete_),devices_[0].ordinal,streams_[0]),"Multi-owner close input failed");requireProvider(definitions_.front()->launch(reset?reset_:terminal_),"Multi-owner terminal launch failed");});join();boundary_->awaitTerminal();}
private:
    /** @return Existing production kernels for the exact backend, without substitutions. */
    static std::shared_ptr<const q::IPooledImmutableNativeKernels> bundle(DeviceId device){
#ifdef HAVE_CUDA
        if(device.is_cuda())return q::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())return q::createROCmPooledImmutableNativeKernels();
#endif
        throw std::runtime_error("Multi-owner endpoint backend unavailable");
    }
    /** @brief Join only at diagnostic/setup/terminal boundaries, never within protocol execution. */
    void join()const{for(std::size_t i=0;i<streams_.size();++i)if(streams_[i])workers_[i]->submitAndWait([&,i]{if(!backends_[i]->synchronizeStream(streams_[i],devices_[i].ordinal))std::terminate();});}
    /** @brief Record original participant graphs once under their real complete owner. */
    void build();
    /** @return Exact terminal diagnostic bytes, never an input driving a later device decision. */
    template<class T>std::vector<T> observe(std::size_t endpoint,const void* address,std::size_t count)const{std::vector<T> value(count);if(count)workers_[endpoint]->submitAndWait([&,endpoint]{requireProvider(backends_[endpoint]->deviceToHostOnStream(value.data(),address,count*sizeof(T),devices_[endpoint].ordinal,streams_[endpoint]),"Multi-owner observation failed");});join();return value;}
    static constexpr std::uint32_t Maximum=64,Unique=16;
    PreparedNativeLookupFixture& session_; ///< Original source outlives every payload and native definition.
    std::vector<std::unique_ptr<ProviderReplayGraphs>> definitions_; ///< Destroyed after all member payload aliases.
    ProviderFile& file_;std::vector<std::uint32_t> slots_;std::vector<DeviceId> devices_;std::vector<std::shared_ptr<IWorkerGPUContext>> workers_;std::vector<IBackend*> backends_;std::vector<void*> streams_;
    std::vector<std::shared_ptr<BufferArena>> arenas_;std::vector<std::shared_ptr<ComputeGraph>> graphs_;std::vector<std::shared_ptr<IDeviceContext>> contexts_;std::vector<std::shared_ptr<const q::IPooledImmutableNativeKernels>> code_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;std::vector<PhysicalMemoryAllocationLease> leases_;
    std::shared_ptr<HostImmutableRowProvider> provider_;std::shared_ptr<q::ImmutableLookupService> service_;std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator_;std::vector<std::shared_ptr<q::PooledImmutableOwnerPreparedStorage>> owners_;std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary_;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinEdge>> forward_;std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> returned_;std::vector<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> free_;std::vector<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> victim_;std::vector<std::shared_ptr<const q::PooledImmutableCapturedSourceWave>> source_;std::vector<std::shared_ptr<const q::PooledImmutableCapturedPublicationWave>> publication_;std::vector<std::shared_ptr<const q::PooledImmutableCapturedReadWave>> read_;std::vector<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>> retirement_;
    std::optional<DeviceGenerationControlBinding> control_;std::shared_ptr<q::PooledImmutableCompleteCohort> cohort_;PooledImmutableRetirementCompletion completion_;IGPUGraphCapture *terminal_=nullptr,*reset_=nullptr;
    std::vector<float> expected_;std::vector<std::byte> expectedResponses_;std::vector<std::vector<std::byte>> expectedPackets_;std::uint64_t replayIndex_=0;const int terminalComplete_=1;
};

/** @brief Compose every real owner and record the exclusive complete original cohort. */
void CrossVendorMultiOwnerReplay::build(){
    std::vector<PooledPinParticipantNodes> ownerVictims(owners_.size()),ownerRetired(owners_.size());
    PooledPinParticipantNodes coordinatorVictims;
    workers_[0]->submitAndWait([&]{
        const auto ids=arenas_[0]->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arenas_[0]->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
        const auto& cp=coordinator_->plan().parameters();const auto& sp=boundary_->pooledStorage()->geometry().parameters();q::PooledNativeOperandSpans inputs{};
        inputs[static_cast<size_t>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,Maximum*sizeof(std::uint64_t)};
        inputs[static_cast<size_t>(q::PooledNativeOperand::OriginalCount)]=q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,sizeof(int)};
        inputs[static_cast<size_t>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{boundary_->pooledStorage()->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,sp.offsets.protocol+offsetof(q::ImmutableLookupDeviceState,admissionGeneration),sizeof(std::uint64_t)};
        const auto rows=DeviceRowRange::deviceCounted(Maximum/16,static_cast<const int*>(count->gpu_data_ptr())).repeatEachRow(16);
        graphs_[0]->addNode("cross/begin",std::make_unique<PooledImmutableNativePhaseStage>(*arenas_[0],code_[0],PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::Begin,coordinator_,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::Begin,coordinator_->plan(),n::OriginalRows{rows,n::OriginalIdWidth::UInt64},inputs)}),devices_[0]);
        graphs_[0]->addNode("cross/commands",std::make_unique<PooledImmutableNativePhaseStage>(*arenas_[0],code_[0],PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::PinCommands,coordinator_,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::PinCommands,coordinator_->plan(),{},{})}),devices_[0]);graphs_[0]->addDependency("cross/commands","cross/begin");
        std::vector<std::string> sentNodes;for(std::size_t i=0;i<owners_.size();++i){const auto sent=appendPooledPinSender(*graphs_[0],*arenas_[0],forward_[i],"cross/begin","cross/commands","cross/pin_send"+std::to_string(i));sentNodes.push_back(sent.last);}
        const auto pin=appendPooledPinCoordinatorReturn(*graphs_[0],*arenas_[0],returned_,code_[0],sentNodes,"cross/commands","cross/pin_return");
        const auto free=appendPooledFreeCoordinatorWave(*graphs_[0],*arenas_[0],free_,code_[0],pin.last,"cross/free");
        coordinatorVictims=appendPooledVictimCoordinatorWave(*graphs_[0],*arenas_[0],victim_,code_[0],free.last,"cross/victims");
    });
    for(std::size_t i=0;i<owners_.size();++i)workers_[i+1]->submitAndWait([&,i]{const auto received=appendPooledPinReceiver(*graphs_[i+1],*arenas_[i+1],forward_[i],code_[i+1],"cross/pin");
        const auto returned=appendPooledPinOwnerReturn(*graphs_[i+1],*arenas_[i+1],returned_[i],code_[i+1],received.last,"cross/pin_return");
        const auto free=appendPooledFreeOwnerWave(*graphs_[i+1],*arenas_[i+1],free_[i],code_[i+1],returned.last,"cross/free");ownerVictims[i]=appendPooledVictimOwnerWave(*graphs_[i+1],*arenas_[i+1],victim_[i],code_[i+1],free.last,"cross/victims");
    });
    workers_[0]->submitAndWait([&]{
        std::vector<q::PooledImmutableSourceParticipant> sourceMembers;for(const auto& value:source_)sourceMembers.emplace_back(value);
        const auto fill=appendPooledSourceCoordinatorMembers(*graphs_[0],*arenas_[0],sourceMembers,boundary_,code_[0],*control_,coordinatorVictims.last,"cross/source");
        std::vector<q::PooledImmutablePublicationParticipant> publicationMembers;for(const auto& value:publication_)publicationMembers.emplace_back(value);const auto ready=appendPooledPublicationCoordinator(*graphs_[0],*arenas_[0],fill,publicationMembers,code_[0],"cross/publication");
        const auto output=arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);std::vector<q::PooledImmutableReadParticipant> readMembers;for(const auto& value:read_)readMembers.emplace_back(value);
        const auto read=appendPooledReadCoordinator(*graphs_[0],*arenas_[0],ready,readMembers,{output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()},"cross/read","cross/original_order_ple");
        requirePooledReadCompletion(*graphs_[0],read);for(const auto& value:read_)retirement_.push_back(q::PooledImmutableCapturedRetirementWave::bind(value,memory_));
        std::vector<q::PooledImmutableRetirementParticipant> retiredMembers;for(const auto& value:retirement_)retiredMembers.emplace_back(value);completion_=appendPooledRetirementCoordinator(*graphs_[0],*arenas_[0],read,retiredMembers,code_[0],"cross/retirement");requirePooledRetirementCompletion(*graphs_[0],completion_);
        graphs_[0]->setRetainedStorageBoundary(boundary_);
    });
    for(std::size_t i=0;i<owners_.size();++i)workers_[i+1]->submitAndWait([&,i]{const auto fill=appendPooledSourceOwnerFill(*graphs_[i+1],*arenas_[i+1],source_[i],code_[i+1],ownerVictims[i].last,"cross/source");
        const auto ready=appendPooledPublicationOwner(*graphs_[i+1],*arenas_[i+1],publication_[i],code_[i+1],fill.last,"cross/publication");
        const auto read=appendPooledReadOwner(*graphs_[i+1],*arenas_[i+1],read_[i],code_[i+1],ready.last,"cross/read");
        ownerRetired[i]=appendPooledRetirementOwner(*graphs_[i+1],*arenas_[i+1],retirement_[i],code_[i+1],read.last,"cross/retirement");requirePooledRetirementOwnerCompletion(*graphs_[i+1],retirement_[i],ownerRetired[i].last);
    });
    // Only the original cohort owner creates and records the actual programs.
    // No caller executable (including an unrelated same-stream graph) is accepted.
    q::PooledImmutableCohortParticipant sourceProgram{workers_[0],contexts_[0],arenas_[0],graphs_[0],streams_[0],definitions_[0]->definition(0)};
    std::vector<q::PooledImmutableCohortRemote> ownerPrograms;
    for(std::size_t i=0;i<owners_.size();++i)ownerPrograms.push_back({static_cast<std::uint32_t>(i),{workers_[i+1],contexts_[i+1],arenas_[i+1],graphs_[i+1],streams_[i+1],definitions_[i+1]->definition(0)},ownerRetired[i].last});
    cohort_=q::PooledImmutableCompleteCohort::bind(std::move(sourceProgram),completion_,std::move(ownerPrograms));
    for(std::size_t i=0;i<devices_.size();++i){
        const auto nodes=i?cohort_->ownerKernelInventory(i-1):cohort_->sourceKernelInventory();
        requireProvider(!nodes.empty(),"Cross-vendor original native program has no kernel inventory");
        for(const auto& node:nodes){requireProvider(node.valid(),"Cross-vendor actual native kernel inventory incomplete");std::cout<<"POOLED_CROSS_READ_RESOURCE participant="<<devices_[i].toString()<<" name="<<node.name<<" threads="<<node.block_x<<" registers="<<node.registers_per_thread<<" local_bytes="<<node.local_memory_bytes_per_thread<<" shared_bytes="<<node.static_shared_memory_bytes+node.dynamic_shared_memory_bytes<<" active_blocks_per_sm="<<node.max_active_blocks_per_sm<<'\n';}
    }
    workers_[0]->submitAndWait([&]{terminal_=definitions_[0]->capture(1);requireProvider(bool(terminal_),"Cross-vendor retained terminal unavailable");
        {ScopedBackendGraphCapture capture(*workers_[0],*terminal_,"cross-vendor original complete-parent close");requireProvider(capture.begin(),"Cross-vendor close capture begin failed");boundary_->recordTerminalClose(streams_[0],control_->healthy(),control_->row()+sampling_math::kDeviceGenerationControlRequestComplete);capture.finish();}
        std::string detail;requireProvider(terminal_->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());requireProvider(terminal_->instantiate(),"Cross-vendor retained close instantiation failed");
    });
    workers_[0]->submitAndWait([&]{reset_=definitions_[0]->capture(2);requireProvider(bool(reset_),"Cross-vendor retained reset unavailable");
        {ScopedBackendGraphCapture capture(*workers_[0],*reset_,"cross-vendor original typed complete-cohort reset");requireProvider(capture.begin(),"Cross-vendor reset capture begin failed");boundary_->recordTerminalClose(streams_[0]);capture.finish();}
        std::string detail;requireProvider(reset_->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());requireProvider(reset_->instantiate(),"Cross-vendor retained reset instantiation failed");
        // Closing an initialized idle source before its first admission must not
        // create a ticket, consume a row or require an armed service.
        requireProvider(definitions_[0]->launch(reset_),"Cross-vendor initial idle reset failed");
    });join();
    requireProvider(!boundary_->hasActiveInterval(),"Cold reset invented an active service interval");
}
/**
 * @brief Replay one exact live query and independently check each owner's actual traffic and inactive tails.
 * @details Host observations occur only after all captured participants complete;
 * they never select requests, grants, source windows or device cache decisions.
 */
void CrossVendorMultiOwnerReplay::replay(std::span<const std::uint64_t> ids){
    requireProvider(ids.size()<=Maximum && ids.size()%16==0,"Multi-owner query must retain complete original heads");
    workers_[0]->submitAndWait([&]{const auto input=arenas_[0]->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arenas_[0]->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY),control=arenas_[0]->getSharedTensor(BufferId::MTP_GENERATION_CONTROL);std::memset(input->raw_mutable_data(),0xff,input->size_bytes());if(!ids.empty())std::memcpy(input->raw_mutable_data(),ids.data(),ids.size_bytes());*static_cast<int*>(count->raw_mutable_data())=ids.size()/16;std::memset(control->raw_mutable_data(),0,control->size_bytes());static_cast<int*>(control->raw_mutable_data())[sampling_math::kDeviceGenerationControlOk]=1;for(const auto& root:{input,count,control}){TransferEngine::prepareDeviceInput(root.get(),devices_[0],streams_[0]);TransferEngine::requireDeviceInput(root.get(),devices_[0],streams_[0]);}});
    ++replayIndex_;const auto maximumSlots=*std::max_element(slots_.begin(),slots_.end());std::vector<std::byte> sourcePacket(maximumSlots*file_.row_bytes,static_cast<std::byte>(0xa0+replayIndex_%16));
    workers_[0]->submitAndWait([&]{if(!sourcePacket.empty()){const auto root=boundary_->pooledStorage()->roots().frame;requireProvider(backends_[0]->hostToDeviceOnStream(static_cast<std::byte*>(root->gpu_data_ptr())+boundary_->pooledStorage()->geometry().parameters().offsets.payload,sourcePacket.data(),sourcePacket.size(),devices_[0].ordinal,streams_[0]),"Multi-owner source canary upload failed");TransferEngine::publishDeviceWrite(root.get(),devices_[0],streams_[0]);}});
    for(std::size_t i=0;i<owners_.size();++i){expectedPackets_[i].assign(slots_[i]*file_.row_bytes,static_cast<std::byte>(0x50+(replayIndex_+i)%16));workers_[i+1]->submitAndWait([&,i]{if(!expectedPackets_[i].empty()){const auto root=owners_[i]->roots().native;requireProvider(backends_[i+1]->hostToDeviceOnStream(static_cast<std::byte*>(root->gpu_data_ptr())+owners_[i]->plan().parameters().native.source_payload,expectedPackets_[i].data(),expectedPackets_[i].size(),devices_[i+1].ordinal,streams_[i+1]),"Multi-owner receiver canary upload failed");TransferEngine::publishDeviceWrite(root.get(),devices_[i+1],streams_[i+1]);}});}
    join();cohort_->launch();join();
    const auto co=coordinator_->roots(),src=boundary_->pooledStorage()->roots();const auto& cp=coordinator_->plan().parameters();const auto& sp=boundary_->pooledStorage()->geometry().parameters();
    const auto lookup=observe<p::LookupFrame>(0,static_cast<const std::byte*>(co.frame->gpu_data_ptr())+cp.frame.control,1).front();EXPECT_EQ(lookup.phase,p::LookupPhase::Idle);EXPECT_EQ(lookup.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(lookup.original_ids,ids.size());requireProvider(lookup.unique_rows<=Unique,"Multi-owner unique bound corrupted");
    const auto unique=observe<p::CoordinatorUnique>(0,static_cast<const std::byte*>(co.frame->gpu_data_ptr())+cp.frame.unique,lookup.unique_rows);const auto coordinates=observe<std::uint32_t>(0,static_cast<const std::byte*>(src.frame->gpu_data_ptr())+sp.offsets.coordinates,lookup.unique_rows);
    const auto requests=observe<n::DirectedSlice>(0,static_cast<const std::byte*>(co.native->gpu_data_ptr())+cp.native.pin_slices,owners_.size());
    std::vector<std::vector<const p::CoordinatorUnique*>> hit(owners_.size()),fill(owners_.size());std::uint32_t misses=0,hits=0;
    for(const auto& row:unique){if(row.phase==p::UniquePhase::Read){requireProvider(row.grant.owner<owners_.size(),"Multi-owner hit has foreign owner");hit[row.grant.owner].push_back(&row);++hits;}else{++misses;if(row.phase==p::UniquePhase::FreeGranted || row.phase==p::UniquePhase::VictimGranted){requireProvider(row.allocation_owner<owners_.size(),"Multi-owner fill has foreign owner");fill[row.allocation_owner].push_back(&row);}}}
    // The source packet is a shared dead bank: fills visit owners in order,
    // followed by reads in order. Preserve earlier tails instead of treating
    // capacity as transferred data. Each owner retains its own independent bank.
    const auto update=[&](std::vector<std::byte>& destination,const std::vector<const p::CoordinatorUnique*>& rows){requireProvider(rows.size()*file_.row_bytes<=destination.size(),"Multi-owner packet exceeds admitted capacity");for(std::size_t j=0;j<rows.size();++j){const auto original=file_.original(rows[j]->row);std::memcpy(destination.data()+j*file_.row_bytes,original.data(),original.size());}};
    for(std::size_t i=0;i<owners_.size();++i){std::sort(fill[i].begin(),fill[i].end(),[](const auto* a,const auto* b){const bool av=a->phase==p::UniquePhase::VictimGranted,bv=b->phase==p::UniquePhase::VictimGranted;return av!=bv?av<bv:a->decision_index<b->decision_index;});update(sourcePacket,fill[i]);update(expectedPackets_[i],fill[i]);}
    lastHits.assign(owners_.size(),0);lastFills.assign(owners_.size(),0);
    std::uint64_t payloadTotal=0;
    for(std::size_t i=0;i<owners_.size();++i){const auto roots=owners_[i]->roots();const auto& op=owners_[i]->plan().parameters();const auto ingress=observe<n::OwnerIngress>(i+1,static_cast<const std::byte*>(roots.native->gpu_data_ptr())+op.native.ingress,1).front();
        lastHits[i]=hit[i].size();lastFills[i]=fill[i].size();EXPECT_EQ(requests[i].live_bytes,hit[i].size()*sizeof(p::Grant));EXPECT_EQ(ingress.payload_bytes,hit[i].size()*file_.row_bytes);payloadTotal+=ingress.payload_bytes;
        if(slots_[i]){EXPECT_EQ(ingress.record_bytes,requests[i].live_bytes);const auto grants=observe<p::Grant>(i+1,static_cast<const std::byte*>(roots.frame->gpu_data_ptr())+op.frame.commands,hit[i].size());for(std::size_t j=0;j<grants.size();++j)EXPECT_EQ(std::memcmp(&grants[j],&hit[i][j]->grant,sizeof(p::Grant)),0);}else{EXPECT_TRUE(hit[i].empty());EXPECT_TRUE(fill[i].empty());}
        update(sourcePacket,hit[i]);update(expectedPackets_[i],hit[i]);if(!expectedPackets_[i].empty())EXPECT_EQ(observe<std::byte>(i+1,static_cast<const std::byte*>(roots.native->gpu_data_ptr())+op.native.source_payload,expectedPackets_[i].size()),expectedPackets_[i]);
        const auto frame=observe<p::OwnerFrame>(i+1,static_cast<const std::byte*>(roots.frame->gpu_data_ptr())+op.frame.control,1).front();EXPECT_EQ(frame.phase,p::OwnerPhase::Idle);EXPECT_EQ(frame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(frame.held_pins,0u);EXPECT_EQ(frame.reservations,0u);
        std::cout<<"POOLED_MULTI_OWNER owner="<<i<<" device="<<devices_[i+1].toString()<<" slots="<<slots_[i]<<" hit_rows="<<hit[i].size()<<" fill_rows="<<fill[i].size()<<" grant_bytes="<<requests[i].live_bytes<<" payload_bytes="<<ingress.payload_bytes<<'\n';
    }
    EXPECT_EQ(payloadTotal,std::uint64_t{hits}*file_.row_bytes);if(!sourcePacket.empty())EXPECT_EQ(observe<std::byte>(0,static_cast<const std::byte*>(src.frame->gpu_data_ptr())+sp.offsets.payload,sourcePacket.size()),sourcePacket);
    const auto received=observe<std::uint64_t>(0,static_cast<const std::byte*>(src.native->gpu_data_ptr())+sp.offsets.payload_bytes,1).front();EXPECT_EQ(received,hit.back().size()*file_.row_bytes);
    const auto io=observe<q::ImmutableLookupDeviceState>(0,src.native->gpu_data_ptr(),1).front();EXPECT_EQ(io.status,0u);EXPECT_EQ(io.idCount,misses);EXPECT_EQ(io.nativeBytes,std::uint64_t{misses}*file_.row_bytes);EXPECT_EQ(io.issuedSequence,io.consumedSequence);
    for(std::size_t i=0;i<unique.size();++i){requireProvider(coordinates[i]<lookup.unique_rows,"Multi-owner coordinate escaped actual response extent");const auto original=file_.original(unique[i].row);std::memcpy(expectedResponses_.data()+coordinates[i]*file_.row_bytes,original.data(),original.size());}EXPECT_EQ(observe<std::byte>(0,static_cast<const std::byte*>(src.frame->gpu_data_ptr())+sp.offsets.responses,expectedResponses_.size()),expectedResponses_);
    if(!ids.empty()){std::vector<std::byte> native(ids.size()*file_.row_bytes);for(std::size_t i=0;i<ids.size();++i){const auto original=file_.original(ids[i]);std::memcpy(native.data()+i*file_.row_bytes,original.data(),original.size());}q::decodeCPUImmutableLookupRows(file_.type,file_.columns,file_.row_bytes,native.data(),ids.size(),expected_.data());}
    const auto actual=observe<float>(0,arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING)->gpu_data_ptr(),expected_.size());EXPECT_EQ(std::memcmp(actual.data(),expected_.data(),expected_.size()*sizeof(float)),0);
    const auto control=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);EXPECT_EQ(control[sampling_math::kDeviceGenerationControlOk],1);EXPECT_EQ(control[sampling_math::kDeviceGenerationControlErrorCode],0);
    const auto frontier=observe<n::SourceFrontier>(0,static_cast<const std::byte*>(src.native->gpu_data_ptr())+sp.offsets.frontier,1).front();EXPECT_EQ(frontier.status,q::adaptive_lookup::Status::Success);
}
/**
 * @brief One partitioned explicit topology/capacity trajectory, with no hardware skips or mode substitution.
 * @param source Source/continuation endpoint, independent of every physical cache owner.
 * @param owners Original physical owner order, deliberately reversed from ordinal order.
 * @param slots Exact per-owner capacities including empty C0 participants.
 * @details Large duplicate-bearing work fills every positive owner, then replays
 * hot hits, one-row skew and empty work. Retained generation reset reuses the
 * same captured programs and proves packet tails after shrinking. The exact
 * sparse IQ4_NL source includes keys differing above bit32; all-format single-
 * owner coverage remains the separate unchanged canonical gate.
 */
void crossVendorMultiOwnerTrajectory(DeviceId source,std::vector<DeviceId> owners,std::vector<std::uint32_t> slots){
    std::vector<DeviceId> devices{source};devices.insert(devices.end(),owners.begin(),owners.end());for(const auto device:devices){const auto* backend=getBackendFor(device);requireProvider(backend && backend->deviceCount()>device.ordinal,"Requested multi-owner cross-vendor hardware unavailable; cell is unqualified");}
    if(!hasCPUBackend())initCPUBackend(-1);
    auto session=PreparedNativeLookupFixture::prepare(devices);
    {ProviderFile file;CrossVendorMultiOwnerReplay replay(*session,file,source,owners,slots);
        std::vector<std::uint64_t> large(64),skew(16,2000),high(16,ProviderFile::High);for(std::size_t i=0;i<large.size();++i)large[i]=i%16==15?ProviderFile::High:2000+i%16;
        for(std::uint64_t generation=1;generation<=2;++generation){replay.admit(generation);
            // Admit the high-bit identity into an actually free slot first.
            // Every later hot-set member remains within this same sixteen-row
            // set, including across admissions. Therefore hot coverage proves
            // each owner without assuming that TinyLFU evicts an outside row.
            replay.replay(high);replay.replay(high);
            replay.replay(large);replay.replay(large);
            for(std::size_t owner=0;owner<slots.size();++owner)EXPECT_EQ(replay.lastHits[owner],slots[owner]);
            replay.replay(skew);std::size_t skewHits=0;for(const auto hits:replay.lastHits)skewHits+=hits;EXPECT_LE(skewHits,1u);
            replay.replay(high);replay.replay(high);std::size_t highHits=0;for(const auto hits:replay.lastHits)highHits+=hits;EXPECT_EQ(highHits,std::any_of(slots.begin(),slots.end(),[](auto capacity){return capacity!=0;})?1u:0u);
            replay.replay({});for(std::size_t owner=0;owner<slots.size();++owner){EXPECT_EQ(replay.lastHits[owner],0u);EXPECT_EQ(replay.lastFills[owner],0u);}replay.close(generation==2);
        }
    }
    session->requirePayloadRetired();
    session->retire();
    session->requireExactRetired();
}
