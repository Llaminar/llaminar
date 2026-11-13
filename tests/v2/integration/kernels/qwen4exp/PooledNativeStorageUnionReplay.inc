/**
 * @file PooledNativeStorageUnionReplay.inc
 * @brief Original-rank source and storage-only native union capture/reset replay.
 *
 * Included after the ordinary provider helpers. The sole query participant is
 * the original source; every other actual GPU is a physical storage owner only.
 * Real prepared workers, source factories, PMA declarations and directed graph
 * builders provide all execution authority. Host copies occur at trace input
 * and completed terminal observation boundaries, never inside lookup execution.
 */
#ifndef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
/** @brief Print the first fixture failure before any strict original-owner cleanup can terminate. */
void reportNativeStorageUnionFailure(const char* phase,const std::exception_ptr& failure)noexcept {
    try{std::rethrow_exception(failure);}
    catch(const std::exception& error){std::cerr<<"Native storage union "<<phase<<" failure: "<<error.what()<<'\n';}
    catch(...){std::cerr<<"Native storage union "<<phase<<" failure: non-standard exception\n";}
}
/** @brief Complete original storage-union lifetime for one immutable case. */
class NativeStorageUnionReplay final {
public:
    /**
     * @brief Admit and capture the actual source/storage population once.
     * @param devices Actual distinct GPUs from one original MPI/NUMA owner.
     * @param empty All owners C0, otherwise retain positive and an empty owner.
     */
    NativeStorageUnionReplay(std::span<const DeviceId> devices,bool empty)
        :devices_(devices.begin(),devices.end()),rank_(MPIContextFactory::global()->rank()),
         session_(PreparedNativeLookupFixture::prepare(devices_,rank_)),memory_(session_->memory()),
         owners_(devices_.size()-1),arenas_(devices_.size()),graphs_(devices_.size()),
         contexts_(devices_.size()),streams_(devices_.size()),definitions_(devices_.size()) {
        try {
        requireProvider(devices_.size()>1,"Native storage replay requires an actual remote owner");
        const auto source=file_.descriptor();const PhysicalMemoryAllocatorIdentity sourceId{rank_,devices_[0]},hostId{rank_,DeviceId::cpu()};
        std::vector<ImmutableRowTableLayoutOwner> capacities;
        std::vector<PhysicalMemoryAllocatorIdentity> originalOwners;
        std::vector<AdaptiveGPUCacheOwnerRequest> placementOwners;
        std::vector<PhysicalMemoryResource> gpu;
        for(std::size_t i=0;i<devices_.size();++i)gpu.push_back(session_->resource(devices_[i]));
        for(std::size_t i=1;i<devices_.size();++i){
            const PhysicalMemoryAllocatorIdentity owner{rank_,devices_[i]};originalOwners.push_back(owner);
            const auto slots=empty || (devices_.size()>2 && i==2)?0u:3u;
            capacities.push_back({{owner,ImmutableRowStorage::VRAM},0,slots,{}});
            placementOwners.push_back({{rank_,devices_[i].type,devices_[i].ordinal},slots,0});
        }
        capacities.push_back({{hostId,ImmutableRowStorage::RAM},0,2,{}});
        const AdaptiveGPUCacheParticipant coordinatorParticipant{rank_,devices_[0].type,devices_[0].ordinal};
        const AdaptiveGPUCachePlacementRequest placement{AdaptiveGPUCachePlacementMode::DeclaredRankLocal,placementOwners,coordinatorParticipant};
        const auto inventory=MPIContextFactory::global()->clusterInventory();
        requireProvider(bool(inventory),"Native storage replay omitted original discovery");
        const auto peer=ImmutableRowTableDirectedTransportPolicy::observe(placement,inventory->getRank(rank_));
        requireProvider(peer.has_value(),"Native union requires its actual same-vendor ownership observation");
        const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
            ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,capacities,{sourceId},{},
            {{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},
            empty?std::nullopt:std::optional(sourceId),Unique,ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator,peer);
        const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,originalOwners,sourceId);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},Maximum);
        const auto cpu=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,sourceId);
        const auto channel=q::ImmutableLookupLayout::createSubordinate(Unique,source.rowBytes());
        const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
        const RetainedHostServiceInstance ioKey{rank_,devices_[0],RetainedHostServiceRole::ImmutableLookupIO,0};
        RetainedHostServiceDeclarations::Builder ioBuilder(rank_);ioBuilder.add(ioKey,io);
        const auto ioDeclarations=std::move(ioBuilder).freeze();
        const std::array declarations{
            CPUTensorStorageDeclaration(1,Maximum*2,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(Maximum/16,16*file_.columns,BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
            CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
        PhysicalMemoryPlanBuilder bom;const auto add=[&](const PhysicalMemoryPlan& plan){for(const auto& row:plan.resources())bom.add(row);};
        add(cpu.bom(session_->resource(DeviceId::cpu())));add(policy.policyBOM(gpu[0]));add(policy.mandatoryBOM(gpu[0]));
        add(q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,gpu[0]));
        add(q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,source,gpu[0]));
        add(q::PooledImmutableDirectedTransport::completeBOM(physical,policy,session_->resource(DeviceId::cpu()),gpu));
        add(io.bom(session_->resource(DeviceId::cpu())));
        bom.add(session_->resource(DeviceId::cpu()),PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
        for(const auto& declaration:declarations)bom.add(session_->resource(DeviceId::cpu()),PhysicalMemoryOwner::ActivationArena,declaration.physicalBytes());
        activationBytes_=Maximum*sizeof(std::uint64_t)+sizeof(int)+sampling_math::kDeviceGenerationControlCount*sizeof(int)+Maximum*file_.columns*sizeof(float)+sizeof(int);
        bom.add(gpu[0],PhysicalMemoryOwner::ActivationArena,activationBytes_);
        for(std::size_t i=0;i<devices_.size();++i){
            add(RetainedParticipantGraphExecutable::generalBOM(session_->resource(DeviceId::cpu()),gpu[i],i?1:3));
            if(i){add(physical.cacheBOM(i-1,gpu[i]));add(physical.mandatoryBOM(i-1,gpu[i]));
                add(q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,i-1,gpu[i]));
                add(NativeOrderingEventGroup::bom(q::PooledImmutableOwnerNativePlan::orderingProgram(physical,policy,i-1),session_->resource(DeviceId::cpu()),gpu[i]));}
            add(NativeOrderingEventGroup::bom(NativeOrderingEventProgram::lookupSubmissionRetirement(),session_->resource(DeviceId::cpu()),gpu[i]));
        }
        for(const auto program:{q::PooledImmutableCoordinatorNativePlan::orderingProgram(physical,policy),q::PooledImmutableSourceGeometry::orderingProgram(physical,policy,source)})
            add(NativeOrderingEventGroup::bom(program,session_->resource(DeviceId::cpu()),gpu[0]));
        session_->admit(bom.build());
        activation_=memory_->claimNewAllocation(devices_[0],PhysicalMemoryOwner::ActivationArena,activationBytes_);
        auto ioBinding=ioDeclarations->bind(memory_,std::span{&ioKey,std::size_t{1}});
        provider_=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(source,cpu,memory_));
        service_=q::ImmutableLookupService::create(provider_,channel,std::array{devices_[0]},memory_,DeviceId::cpu(),ioBinding->take(ioKey));
        const auto issued=service_->bindSubordinate(devices_[0]);
        const auto cp=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,issued.deviceView().base.table,issued.storageGeneration());
        std::vector<q::PooledImmutableOwnerNativePlan> plans;
        for(std::size_t i=0;i<owners_.size();++i)plans.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,i,issued.deviceView().base.table,issued.storageGeneration()));
        for(std::size_t i=0;i<devices_.size();++i){
            arenas_[i]=std::make_shared<BufferArena>();graphs_[i]=std::make_shared<ComputeGraph>();
            definitions_[i]=std::make_unique<ProviderReplayGraphs>(session_->permit(devices_[i]),i?1:3);
            worker(i)->submitAndWait([&,i]{streams_[i]=ExplicitGPUStream(worker(i)->defaultStream()).get();contexts_[i]=IDeviceContext::create(devices_[i],1);
                if(i){owners_[i-1]=q::PooledImmutableOwnerPreparedStorage::create(plans[i-1],memory_,streams_[i]);code(i)->initialize(*owners_[i-1],streams_[i]);}
                else {coordinator_=q::PooledImmutableCoordinatorPreparedStorage::create(cp,memory_,streams_[0]);code(0)->initialize(*coordinator_,streams_[0]);}});
        }
        worker(0)->submitAndWait([&]{
            boundary_=q::ImmutableLookupExecutionBoundary::createPooled(service_,q::PooledImmutableSourceGeometry::fromPlans(plans,cp,source,1),owners_,coordinator_,memory_,streams_[0]);
            PooledImmutableFactoryReplayAccess::bindOriginalPreparedWorkers(*boundary_->pooledStorage(),session_->source());
            const std::array roles{BufferId::QFN_NGRAM_ROW_IDS,BufferId::PREFILL_CHUNK_GEOMETRY,BufferId::MTP_GENERATION_CONTROL,BufferId::QFN_PLE_EMBEDDING,BufferId::MTP_GENERATION_RESPONSE_TOKENS};
            for(std::size_t i=0;i<roles.size();++i){std::shared_ptr<TensorBase> root(declarations[i].materialize(memory_));
                std::memset(root->raw_mutable_data(),0,root->size_bytes());TransferEngine::prepareDeviceInput(root.get(),devices_[0],streams_[0]);TransferEngine::requireDeviceInput(root.get(),devices_[0],streams_[0]);
                requireProvider(arenas_[0]->registerExternalBuffer(roles[i],root),"Native source activation role changed");}
            control_.emplace(DeviceGenerationControlBinding::admit(*arenas_[0],devices_[0],0));boundary_->bindPooledGenerationControl(*control_);
        });
        transport_=q::PooledImmutableDirectedTransport::create(boundary_,inventory);
        EXPECT_FALSE(transport_->readerGeometry());EXPECT_TRUE(transport_->readers().empty());
        EXPECT_EQ(boundary_->pooledStorage()->geometry().parameters().consumers,1u);
        EXPECT_EQ(transport_->remotes().size(),devices_.size()-1);
        for(const auto& remote:transport_->remotes())EXPECT_TRUE(remote.pin->binding(q::PooledPinWireRole::Invocation,CapturedTransferEndpoint::Producer).nativeCollective());
        build();
        join();
        } catch(...) {
            // Locals in the construction scope have already unwound. Retire
            // member-owned native definitions while their original roots and
            // transport authority still exist; a join alone cannot release them.
            const auto primary=std::current_exception();
            reportNativeStorageUnionFailure("construction",primary);
            try{finish();}
            catch(...){reportNativeStorageUnionFailure("construction retirement",std::current_exception());std::terminate();}
            std::rethrow_exception(primary);
        }
    }
    /** @brief Cleanup uses original typed reset and rejects missing native retirement proof. */
    ~NativeStorageUnionReplay(){try{finish();}catch(...){reportNativeStorageUnionFailure("destructor retirement",std::current_exception());std::terminate();}}
    /** @brief Submit one actual service admission; topology/captures remain unchanged. */
    void admit(std::uint64_t generation){worker(0)->submitAndWait([&]{boundary_->beginAdmission(generation,streams_[0]);});join();}
    /** @brief Trace inputs precede the retained captured source/owner transaction. */
    void replay(std::span<const std::uint64_t> ids,bool cold=false,bool requireHit=false){
        prepare(ids,cold?0:1,0);poisonOwnerPackets();cohort_->launch();join();
        const auto hits=validate(ids,cold?0:1);if(requireHit)EXPECT_GT(hits,0u);
        if(cold)initializeResponse(0);
    }
    /** @brief A recorded GPU first error drains all native storage work and survives response admission. */
    void failedForward(std::span<const std::uint64_t> ids){
        admit(100);const int first=static_cast<int>(sampling_math::DeviceGenerationError::InvalidInitialization);
        const auto output=arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
        const auto before=observe<float>(0,output->gpu_data_ptr(),output->numel());
        prepare(ids,0,first);cohort_->launch();join();
        const auto after=observe<float>(0,output->gpu_data_ptr(),output->numel());EXPECT_EQ(before,after);
        for(std::size_t i=0;i<owners_.size();++i){const auto& plan=owners_[i]->plan().parameters();const auto root=owners_[i]->roots().frame;
            const auto frame=observe<p::OwnerFrame>(i+1,static_cast<const std::byte*>(root->gpu_data_ptr())+plan.frame.control,1).front();
            EXPECT_EQ(frame.held_pins,0u);EXPECT_EQ(frame.reservations,0u);EXPECT_EQ(frame.phase,p::OwnerPhase::Idle);}
        initializeResponse(first);
        bool rejected=false;try{close(true);}catch(const std::runtime_error& error){
            rejected=std::string(error.what()).find("Immutable lookup retained-generation storage failed:")!=std::string::npos;
            if(!rejected)throw;}
        EXPECT_TRUE(rejected);EXPECT_FALSE(boundary_->hasActiveInterval());
    }
    /** @brief Typed captured close/reset drains the exact original service interval. */
    void close(bool reset){
        worker(0)->submitAndWait([&]{requireProvider(getBackendFor(devices_[0])->hostToDeviceOnStream(control_->row()+sampling_math::kDeviceGenerationControlRequestComplete,&terminalComplete_,sizeof(terminalComplete_),devices_[0].ordinal,streams_[0]),"Native terminal input failed");
            requireProvider(definitions_[0]->launch(definitions_[0]->capture(reset?2:1)),"Native terminal graph submission failed");});
        join();boundary_->awaitTerminal();EXPECT_FALSE(boundary_->hasActiveInterval());
    }
    /** @brief Preserve the primary failure while original request reset joins each storage endpoint. */
    void recordedCancellation(){admit(99);try{throw std::runtime_error("recorded native storage body failure");}catch(const std::runtime_error& error){close(true);EXPECT_STREQ(error.what(),"recorded native storage body failure");}}
    /** @brief Release graph borrowers before roots, native union and original PMA/context retirement. */
    void finish(){
        if(!session_)return;join();if(boundary_ && boundary_->hasActiveInterval())close(true);
        const bool admitted=memory_->phase()==PhysicalMemoryAuthorityPhase::Admitted;
        cohort_.reset();
        // Definitions may contain captured operand addresses even when cold
        // recording failed before a complete cohort could be published. Their
        // sole original owners retire before graph stages, arenas and transport.
        for(auto& definition:definitions_)if(definition){definition->retire();definition.reset();}
        for(auto& graph:graphs_)graph.reset();
        contexts_.clear();arenas_.clear();control_.reset();transport_.reset();boundary_.reset();owners_.clear();coordinator_.reset();service_.reset();provider_.reset();activation_={};
        if(admitted)session_->requirePayloadRetired();
        // The original fixture handles Preparing via retireUnadmitted and
        // Admitted via exclusive native retirement; do not forge either proof.
        session_->retire();session_->requireExactRetired();
        if(admitted){
            EXPECT_EQ(session_->retirementReceipts().size(),devices_.size());
            for(const auto& receipt:session_->retirementReceipts()){EXPECT_TRUE(receipt.runtime_reset_invoked);EXPECT_EQ(receipt.canonical_allocation_bytes_before_runtime_reset,0u);}
        }
        session_.reset();
    }
private:
    /** @return Actual original worker; never creates an ordinary context. */
    std::shared_ptr<IWorkerGPUContext> worker(std::size_t i)const{return session_->permit(devices_[i]).worker();}
    /** @return Original backend kernel bundle, with symmetric native operations. */
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code(std::size_t i)const{
#ifdef HAVE_CUDA
        if(devices_[i].is_cuda())return q::createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
        if(devices_[i].is_rocm())return q::createROCmPooledImmutableNativeKernels();
#endif
        throw std::runtime_error("Native storage source backend was not compiled");
    }
    /** @brief This synchronization is solely the component trace's terminal observation boundary. */
    void join()const{if(!session_)return;for(std::size_t i=0;i<devices_.size();++i)if(streams_[i])worker(i)->submitAndWait([&,i]{requireProvider(getBackendFor(devices_[i])->synchronizeStream(streams_[i],devices_[i].ordinal),"Native storage terminal join failed");});}
    /** @return Terminal copied data; it never influences execution or resets device state. */
    template<class T>std::vector<T> observe(std::size_t i,const void* pointer,std::size_t count)const{
        std::vector<T> result(count);if(count)worker(i)->submitAndWait([&,i]{requireProvider(getBackendFor(devices_[i])->deviceToHostOnStream(result.data(),pointer,count*sizeof(T),devices_[i].ordinal,streams_[i]),"Native storage terminal observation failed");});join();return result;
    }
    /** @brief Publish only recorded original inputs and canonical initial controller words outside execution. */
    void prepare(std::span<const std::uint64_t> ids,int healthy,int error){
        file_.requireAuthored(ids);requireProvider(ids.size()<=Maximum && !(ids.size()%16),"Native source trace violates admitted head geometry");
        worker(0)->submitAndWait([&]{
            const auto input=arenas_[0]->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arenas_[0]->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY),controller=arenas_[0]->getSharedTensor(BufferId::MTP_GENERATION_CONTROL);
            std::memset(input->raw_mutable_data(),0xff,input->size_bytes());if(!ids.empty())std::memcpy(input->raw_mutable_data(),ids.data(),ids.size_bytes());
            *static_cast<int*>(count->raw_mutable_data())=ids.size()/16;std::memset(controller->raw_mutable_data(),0,controller->size_bytes());
            static_cast<int*>(controller->raw_mutable_data())[sampling_math::kDeviceGenerationControlOk]=healthy;
            static_cast<int*>(controller->raw_mutable_data())[sampling_math::kDeviceGenerationControlErrorCode]=error;
            for(const auto& root:{input,count,controller}){TransferEngine::prepareDeviceInput(root.get(),devices_[0],streams_[0]);TransferEngine::requireDeviceInput(root.get(),devices_[0],streams_[0]);}
        });
    }
    /** @brief Invoke the production response initializer, preserving cold-prefill/first-error state. */
    void initializeResponse(int error){
        const auto before=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);
        q::QwenAdaptiveCounterTotals prior{};ASSERT_TRUE(q::readQwenAdaptiveCounters(before.data(),before.size(),prior));
        worker(0)->submitAndWait([&]{sampling_math::DeviceGenerationInitialization initialization;
            initialization.forward_failure_policy=sampling_math::DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;
            const auto response=arenas_[0]->getSharedTensor(BufferId::MTP_GENERATION_RESPONSE_TOKENS);
            requireProvider(getBackendFor(devices_[0])->enqueueInitializeDeviceGeneration(1,1,sampling_math::DeviceGenerationPolicy::ordinary(),
                sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse,1,response->gpu_data_ptr(),sampling_math::kDeviceGenerationControlCount,
                control_->row(),devices_[0].ordinal,streams_[0],initialization),"Native source response initializer failed");
            TransferEngine::publishDeviceWrite(control_->owner(),devices_[0],streams_[0]);});join();
        const auto after=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);
        q::QwenAdaptiveCounterTotals current{};ASSERT_TRUE(q::readQwenAdaptiveCounters(after.data(),after.size(),current));
        EXPECT_EQ(current.calls,prior.calls);EXPECT_EQ(current.completed_live_ids,prior.completed_live_ids);
        EXPECT_EQ(after[sampling_math::kDeviceGenerationControlErrorCode],error);EXPECT_EQ(after[sampling_math::kDeviceGenerationControlOk],error?0:1);
    }
    /** @brief Fresh distinct dead-bank canaries make a capacity copy observable after shrinking work. */
    void poisonOwnerPackets(){
        ++replay_;poison_.assign(owners_.size(),{});
        std::size_t maximumSlots=0;for(const auto& owner:owners_)maximumSlots=std::max(maximumSlots,std::size_t(owner->plan().parameters().owner.capacity()));
        sourcePoison_.assign(maximumSlots*file_.row_bytes,static_cast<std::byte>(0xa0+(replay_%16)));
        if(!sourcePoison_.empty())worker(0)->submitAndWait([&]{const auto root=boundary_->pooledStorage()->roots().frame;
            const auto offset=boundary_->pooledStorage()->geometry().parameters().offsets.payload;
            requireProvider(getBackendFor(devices_[0])->hostToDeviceOnStream(static_cast<std::byte*>(root->gpu_data_ptr())+offset,sourcePoison_.data(),sourcePoison_.size(),devices_[0].ordinal,streams_[0]),"Native source packet canary failed");
            TransferEngine::publishDeviceWrite(root.get(),devices_[0],streams_[0]);});
        for(std::size_t i=0;i<owners_.size();++i){const auto bytes=owners_[i]->plan().parameters().owner.capacity()*file_.row_bytes;if(!bytes)continue;
            poison_[i].assign(bytes,static_cast<std::byte>(0x50+(replay_%16)));
            worker(i+1)->submitAndWait([&,i]{const auto root=owners_[i]->roots().native;const auto offset=owners_[i]->plan().parameters().native.source_payload;
                requireProvider(getBackendFor(devices_[i+1])->hostToDeviceOnStream(static_cast<std::byte*>(root->gpu_data_ptr())+offset,poison_[i].data(),poison_[i].size(),devices_[i+1].ordinal,streams_[i+1]),"Native storage packet canary failed");TransferEngine::publishDeviceWrite(root.get(),devices_[i+1],streams_[i+1]);});}
        join();
    }
    /** @brief Build production participant-local source/owner graphs and capture each original definition once. */
    void build(){
        std::optional<PooledImmutableDirectedLookupEndpoints> endpoints;
        worker(0)->submitAndWait([&]{
            const auto ids=arenas_[0]->getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS),count=arenas_[0]->getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY),output=arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
            const auto& sp=boundary_->pooledStorage()->geometry().parameters();q::PooledNativeOperandSpans inputs{};
            inputs[static_cast<std::size_t>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,Maximum*sizeof(std::uint64_t)};
            inputs[static_cast<std::size_t>(q::PooledNativeOperand::OriginalCount)]=q::PooledNativeTensorSpan{count,BufferId::PREFILL_CHUNK_GEOMETRY,0,sizeof(int)};
            inputs[static_cast<std::size_t>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{boundary_->pooledStorage()->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,sp.offsets.protocol+offsetof(q::ImmutableLookupDeviceState,admissionGeneration),sizeof(std::uint64_t)};
            const auto rows=DeviceRowRange::deviceCounted(Maximum/16,static_cast<const int*>(count->gpu_data_ptr())).repeatEachRow(16);
            endpoints=appendPooledImmutableDirectedLookupGraph(*graphs_[0],*arenas_[0],boundary_,transport_,code(0),{rows,n::OriginalIdWidth::UInt64},inputs,*control_,
                {output,BufferId::QFN_PLE_EMBEDDING,0,output->size_bytes()},"native_storage/source","native_storage/output");
        });
        std::vector<q::PooledImmutableCohortRemote> remotes;
        for(const auto& remote:transport_->remotes()){
            const auto i=remote.coordinate+1;PooledPinParticipantNodes terminal;
            worker(i)->submitAndWait([&,i]{terminal=appendPooledImmutableNativeOwnerGraph(*graphs_[i],*arenas_[i],remote,code(i),"native_storage/owner");});
            EXPECT_NO_THROW(q::PooledImmutableDirectedOwnerProgram::requireNativeActual(*graphs_[i],owners_[remote.coordinate]->plan(),coordinator_->plan()));
            // Actual native operations cannot exchange equally sized roles or
            // change DAG order while retaining the original source declaration.
            const auto order=graphs_[i]->getExecutionOrder();requireProvider(order.size()>1,"Native owner recipe is empty");
            auto* first=graphs_[i]->getNode(order.front());auto* last=graphs_[i]->getNode(order.back());
            std::swap(first->stage,last->stage);
            EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::requireNativeActual(*graphs_[i],owners_[remote.coordinate]->plan(),coordinator_->plan()),std::logic_error);
            std::swap(first->stage,last->stage);
            EXPECT_NO_THROW(q::PooledImmutableDirectedOwnerProgram::requireNativeActual(*graphs_[i],owners_[remote.coordinate]->plan(),coordinator_->plan()));
            remotes.push_back({remote.coordinate,{worker(i),contexts_[i],arenas_[i],graphs_[i],streams_[i],definitions_[i]->definition(0)},terminal.last});
        }
        cohort_=q::PooledImmutableCompleteCohort::bind({worker(0),contexts_[0],arenas_[0],graphs_[0],streams_[0],definitions_[0]->definition(0)},std::get<PooledImmutableRetirementCompletion>(endpoints->retirement),std::move(remotes));
        captureIdentity_=transport_->remotes().front().read->captureIdentity();
        worker(0)->submitAndWait([&]{for(unsigned index=1;index<3;++index){auto* native=definitions_[0]->capture(index);
            {ScopedBackendGraphCapture capture(*worker(0),*native,"native storage original terminal/reset");requireProvider(capture.begin(),"Native storage terminal capture failed");
                if(index==1)boundary_->recordTerminalClose(streams_[0],control_->healthy(),control_->row()+sampling_math::kDeviceGenerationControlRequestComplete);else boundary_->recordTerminalClose(streams_[0]);capture.finish();}
            std::string detail;requireProvider(native->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());requireProvider(native->instantiate(),"Native storage terminal instantiation failed");}
            requireProvider(definitions_[0]->launch(definitions_[0]->capture(2)),"Native storage cold idle reset failed");});
        join();EXPECT_FALSE(boundary_->hasActiveInterval());
    }
    /** @brief Check output, actual native receipts, owner retirement and untouched inactive byte tails. */
    std::uint32_t validate(std::span<const std::uint64_t> ids,int healthy){
        const auto cr=coordinator_->roots();const auto& cp=coordinator_->plan().parameters();
        const auto lookup=observe<p::LookupFrame>(0,static_cast<const std::byte*>(cr.frame->gpu_data_ptr())+cp.frame.control,1).front();
        EXPECT_EQ(lookup.original_ids,ids.size());EXPECT_EQ(lookup.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(lookup.phase,p::LookupPhase::Idle);
        const auto unique=observe<p::CoordinatorUnique>(0,static_cast<const std::byte*>(cr.frame->gpu_data_ptr())+cp.frame.unique,lookup.unique_rows);
        const auto slices=observe<n::DirectedSlice>(0,static_cast<const std::byte*>(cr.native->gpu_data_ptr())+cp.native.pin_slices,owners_.size());
        std::uint32_t totalHits=0;
        std::vector<std::vector<const p::CoordinatorUnique*>> allHits(owners_.size()),allFills(owners_.size());
        for(std::size_t owner=0;owner<owners_.size();++owner){
            const auto i=owner+1;const auto& op=owners_[owner]->plan().parameters();const auto roots=owners_[owner]->roots();
            std::vector<const p::CoordinatorUnique*> hits,fills;
            for(const auto& row:unique){
                if(row.phase==p::UniquePhase::Read && row.grant.owner==owner)hits.push_back(&row);
                else if((row.phase==p::UniquePhase::FreeGranted || row.phase==p::UniquePhase::VictimGranted) && row.allocation_owner==owner)fills.push_back(&row);
            }
            allHits[owner]=hits;allFills[owner]=fills;
            totalHits+=hits.size();EXPECT_EQ(slices[owner].live_bytes,hits.size()*sizeof(p::Grant));
            const auto ingress=observe<n::OwnerIngress>(i,static_cast<const std::byte*>(roots.native->gpu_data_ptr())+op.native.ingress,1).front();
            EXPECT_EQ(ingress.payload_bytes,hits.size()*file_.row_bytes);
            const auto frame=observe<p::OwnerFrame>(i,static_cast<const std::byte*>(roots.frame->gpu_data_ptr())+op.frame.control,1).front();
            EXPECT_EQ(frame.phase,p::OwnerPhase::Idle);EXPECT_EQ(frame.status,q::adaptive_lookup::Status::Success);EXPECT_EQ(frame.held_pins,0u);EXPECT_EQ(frame.reservations,0u);
            if(op.owner.capacity()){
                const auto& binding=transport_->remote(owner).read->binding(q::PooledReadWireRole::Rows,CapturedTransferEndpoint::Consumer);
                EXPECT_TRUE(binding.nativeCollective());const auto view=binding.nativeCompletion();requireProvider(view.valid(),"Native source receive lost its actual receipt");
                const auto receipt=observe<ncclLlaminarByteCompletion>(0,view.receipt(),1).front();
                EXPECT_EQ(receipt.bytes,hits.size()*file_.row_bytes);EXPECT_EQ(receipt.message,binding.message().key);EXPECT_EQ(receipt.phase,ncclLlaminarCompletionPhase::Completed);
                EXPECT_EQ(receipt.nativeAbortFlag,view.authority().abortFlag);EXPECT_EQ(receipt.nativeCommunicator,view.authority().deviceCommunicator);
                std::sort(fills.begin(),fills.end(),[](const auto* a,const auto* b){const auto av=a->phase==p::UniquePhase::VictimGranted,bv=b->phase==p::UniquePhase::VictimGranted;return av!=bv?av<bv:a->decision_index<b->decision_index;});
                std::size_t row=0;for(const auto* item:fills){const auto bytes=file_.original(item->row);std::memcpy(poison_[owner].data()+row++*file_.row_bytes,bytes.data(),bytes.size());}
                row=0;for(const auto* item:hits){const auto bytes=file_.original(item->row);std::memcpy(poison_[owner].data()+row++*file_.row_bytes,bytes.data(),bytes.size());}
                EXPECT_EQ(observe<std::byte>(i,static_cast<const std::byte*>(roots.native->gpu_data_ptr())+op.native.source_payload,poison_[owner].size()),poison_[owner]);
            }else {EXPECT_TRUE(hits.empty());EXPECT_TRUE(fills.empty());EXPECT_EQ(ingress.payload_bytes,0u);}
        }
        const auto& sp=boundary_->pooledStorage()->geometry().parameters();const auto sr=boundary_->pooledStorage()->roots();
        const auto updateSource=[&](const auto& rows){requireProvider(rows.size()*file_.row_bytes<=sourcePoison_.size(),"Native source terminal packet escaped its declared storage");
            for(std::size_t i=0;i<rows.size();++i){const auto bytes=file_.original(rows[i]->row);std::memcpy(sourcePoison_.data()+i*file_.row_bytes,bytes.data(),bytes.size());}};
        // The real source fills every owner before reading every owner. Its one
        // shared dead packet bank must preserve earlier inactive tails in that order.
        for(auto& rows:allFills){std::sort(rows.begin(),rows.end(),[](const auto* a,const auto* b){const auto av=a->phase==p::UniquePhase::VictimGranted,bv=b->phase==p::UniquePhase::VictimGranted;return av!=bv?av<bv:a->decision_index<b->decision_index;});updateSource(rows);}
        for(const auto& rows:allHits)updateSource(rows);
        if(!sourcePoison_.empty())EXPECT_EQ(observe<std::byte>(0,static_cast<const std::byte*>(sr.frame->gpu_data_ptr())+sp.offsets.payload,sourcePoison_.size()),sourcePoison_);
        const auto io=observe<q::ImmutableLookupDeviceState>(0,static_cast<const std::byte*>(sr.native->gpu_data_ptr())+sp.offsets.protocol,1).front();
        EXPECT_EQ(io.idCount,lookup.unique_rows-totalHits);EXPECT_EQ(io.nativeBytes,std::uint64_t(io.idCount)*file_.row_bytes);
        const auto controller=observe<int>(0,control_->row(),sampling_math::kDeviceGenerationControlCount);EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlOk],healthy);EXPECT_EQ(controller[sampling_math::kDeviceGenerationControlErrorCode],0);
        std::vector<std::byte> bytes(ids.size()*file_.row_bytes);for(std::size_t i=0;i<ids.size();++i){const auto row=file_.original(ids[i]);std::memcpy(bytes.data()+i*file_.row_bytes,row.data(),row.size());}
        std::vector<float> expected(ids.size()*file_.columns);if(!ids.empty())q::decodeCPUImmutableLookupRows(file_.type,file_.columns,file_.row_bytes,bytes.data(),ids.size(),expected.data());
        const auto output=arenas_[0]->getSharedTensor(BufferId::QFN_PLE_EMBEDDING);const auto actual=observe<float>(0,output->gpu_data_ptr(),expected.size());
        if(!expected.empty())EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)),0);EXPECT_EQ(transport_->remotes().front().read->captureIdentity(),captureIdentity_);
        std::cout<<"POOLED_NATIVE_STORAGE_UNION original_rank="<<rank_<<" backend="<<devices_[0].toString()<<" native_members="<<devices_.size()<<" query_readers=1 storage_only="<<owners_.size()<<" original_ids="<<ids.size()<<" hits="<<totalHits<<" native_peer=1 mapped_payload_bytes=0 retained_capture=1\n";
        return totalHits;
    }
    static constexpr std::uint32_t Maximum=64,Unique=16;
    const int terminalComplete_=1; ///< Terminal input survives its asynchronous copy through the final join.
    ProviderFile file_;std::vector<DeviceId> devices_;int rank_;
    std::unique_ptr<PreparedNativeLookupFixture> session_;std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::size_t activationBytes_=0;PhysicalMemoryAllocationLease activation_;
    std::shared_ptr<HostImmutableRowProvider> provider_;std::shared_ptr<q::ImmutableLookupService> service_;
    std::vector<std::shared_ptr<q::PooledImmutableOwnerPreparedStorage>> owners_;
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator_;
    std::vector<std::shared_ptr<BufferArena>> arenas_;std::vector<std::shared_ptr<ComputeGraph>> graphs_;
    std::vector<std::shared_ptr<IDeviceContext>> contexts_;std::vector<void*> streams_;
    std::vector<std::unique_ptr<ProviderReplayGraphs>> definitions_;
    std::optional<DeviceGenerationControlBinding> control_;std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary_;
    std::shared_ptr<const q::PooledImmutableDirectedTransport> transport_;std::shared_ptr<q::PooledImmutableCompleteCohort> cohort_;
    std::vector<std::vector<std::byte>> poison_;std::vector<std::byte> sourcePoison_;std::uint64_t captureIdentity_=0,replay_=0;
};
/** @brief Two immutable source geometries require independent original process lifetimes. */
enum class NativeStorageUnionOwnerCapacity {PositiveWithEmptyOwners, AllEmpty};
/**
 * @brief Replay one admitted original physical layout per canonical MPI process.
 * @param backend Actual original NUMA-local vendor.
 * @param participants Complete native transport population, including the source.
 * @param capacity Exact immutable physical owner capacity declaration.
 * @note Context reset retires native handles but preserves process-root PMA
 * commitments. The C0 and positive layouts are separate CTest MPI2 cases;
 * neither replaces another original source authority in this process.
 */
void nativeStorageUnionReplay(DeviceType backend,std::size_t participants,NativeStorageUnionOwnerCapacity capacity){
    const bool empty=capacity==NativeStorageUnionOwnerCapacity::AllEmpty;
    originalRankLocalReplay(backend,participants,[empty](const auto& devices){
        if(!hasCPUBackend())initCPUBackend(-1);
        // Keep the lifetime owner outside the body catch so the original
        // exception is visible BEFORE reset/join/definition retirement.
        std::unique_ptr<NativeStorageUnionReplay> replay;std::exception_ptr primary;
        try{
            replay=std::make_unique<NativeStorageUnionReplay>(devices,empty);
            const std::array<std::uint64_t,5> rows{7,ProviderFile::High,700,701,900};
            std::vector<std::uint64_t> large(64),small(16,7);for(std::size_t i=0;i<large.size();++i)large[i]=rows[i%rows.size()];
            for(std::uint64_t generation=1;generation<=3;++generation){replay->admit(generation);replay->replay(small,true);replay->replay(small,false,!empty);replay->replay(large);replay->replay(small);replay->replay({});replay->close(generation==2);}
            replay->recordedCancellation();replay->failedForward(small);
        }catch(...){primary=std::current_exception();if(replay)reportNativeStorageUnionFailure("body",primary);}
        try{if(replay)replay->finish();}
        catch(...){reportNativeStorageUnionFailure("body retirement",std::current_exception());std::terminate();}
        if(primary)std::rethrow_exception(primary);
    });
}
TEST(PooledNativeStorageUnion,CUDA2OriginalSourceCaptureResetReplay){nativeStorageUnionReplay(DeviceType::CUDA,2,NativeStorageUnionOwnerCapacity::PositiveWithEmptyOwners);}
TEST(PooledNativeStorageUnion,ROCm2OriginalSourceCaptureResetReplay){nativeStorageUnionReplay(DeviceType::ROCm,2,NativeStorageUnionOwnerCapacity::PositiveWithEmptyOwners);}
TEST(PooledNativeStorageUnion,ROCm4OriginalSourceCaptureResetReplay){nativeStorageUnionReplay(DeviceType::ROCm,4,NativeStorageUnionOwnerCapacity::PositiveWithEmptyOwners);}
TEST(PooledNativeStorageUnion,CUDA2OriginalSourceCaptureResetC0Replay){nativeStorageUnionReplay(DeviceType::CUDA,2,NativeStorageUnionOwnerCapacity::AllEmpty);}
TEST(PooledNativeStorageUnion,ROCm2OriginalSourceCaptureResetC0Replay){nativeStorageUnionReplay(DeviceType::ROCm,2,NativeStorageUnionOwnerCapacity::AllEmpty);}
TEST(PooledNativeStorageUnion,ROCm4OriginalSourceCaptureResetC0Replay){nativeStorageUnionReplay(DeviceType::ROCm,4,NativeStorageUnionOwnerCapacity::AllEmpty);}
#endif
