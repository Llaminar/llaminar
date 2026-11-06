/**
 * @file LocalHotFutureSourceReplay.inc
 * @brief Genuine bounded SSD/source future lookup, then actual-demand adoption or drained discard.
 *
 * Uses the same original singleton provider, CPU I/O worker, shared coordinator,
 * compact pack/ACK and native parent/PMA owners as the current-source fixture.
 * No response injection or alternate source/launcher is used. This proves source
 * machinery, not a real-GGUF model, multi-consumer Rank cohort or performance.
 */
namespace llaminar2::test::future_l1_source {
namespace q=qwen4exp;namespace h=q::local_hot;namespace t=q::local_hot_transport;namespace n=q::pooled_native;
using original_l1_fork::ProviderFile;
using original_l1_fork::requireProvider;
/** @brief Execute source machinery on the exact requested backend; runtime remains root-owned. */
void replayFutureSource(DeviceId device,bool sourceAfterAdoption=false)
{
    if(!hasCPUBackend())initCPUBackend(-1);
    const std::array devices{device};auto session=PreparedNativeLookupFixture::prepare(devices);
    const auto memory=session->memory();const auto gpu=session->resource(device),host=session->resource(DeviceId::cpu());
    const PhysicalMemoryAllocatorIdentity gpuId{0,device},hostId{0,DeviceId::cpu()};
    const std::array<std::uint64_t,6> authored{7,8,9,10,11,12};ProviderFile file(32,authored);const auto source=file.descriptor();
    constexpr std::uint32_t Maximum=32,Slots=2;
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{gpuId,ImmutableRowStorage::VRAM},0,2,{}},{{hostId,ImmutableRowStorage::RAM},0,2,{}}},{gpuId},{},
        {{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},gpuId,Maximum);
    const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{gpuId},gpuId);
    const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},Maximum);
    const auto cpu=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,gpuId);
    const auto channel=q::ImmutableLookupLayout::createSubordinate(Maximum,source.rowBytes());
    const auto localGeometry=q::PooledImmutableLocalHotGeometry::declare(source,gpuId,Maximum,Slots,{65536,1,2});
    const auto compact=q::PooledImmutableLocalHotTransportGeometry::singleton(gpuId,Maximum,
        ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()));
    const auto io=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
    const std::array declarations{
        CPUTensorStorageDeclaration(1,Maximum*2,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu()),
        CPUTensorStorageDeclaration(Maximum,static_cast<int>(file.columns),BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu())};
    PhysicalMemoryPlanBuilder bom;
    for(const auto& part:{cpu.bom(host),physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),policy.policyBOM(gpu),policy.mandatoryBOM(gpu),
        q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,0,gpu),q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,gpu),
        q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,source,gpu),localGeometry.mandatoryBOM(gpu,host),localGeometry.slotDependentBOM(gpu,host),
        compact.completeBOM(host,std::array{gpu}),io.bom(host),RetainedParticipantGraphExecutable::generalBOM(host,gpu,sourceAfterAdoption?6:4)})
        for(const auto& row:part.resources())bom.add(row);
    bom.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
    bom.add(gpu,PhysicalMemoryOwner::ActivationArena,Maximum*8+4+sampling_math::kDeviceGenerationControlCount*4+Maximum*file.columns*4);
    for(const auto& declaration:declarations)bom.add(host,PhysicalMemoryOwner::ActivationArena,declaration.physicalBytes());
    for(const auto program:originalPooledSourceSetupOrderingPrograms(physical,policy,source)){
        const auto part=NativeOrderingEventGroup::bom(program,host,gpu);for(const auto& row:part.resources())bom.add(row);
    }
    session->admit(bom.build());
    const auto& original=session->permit(device);auto worker=original.worker();auto* backend=getBackendFor(device);
    const auto stream=ExplicitGPUStream(worker->defaultStream()).get();
    const RetainedHostServiceInstance ioKey{0,device,RetainedHostServiceRole::ImmutableLookupIO,0};
    RetainedHostServiceDeclarations::Builder declaration(0);declaration.add(ioKey,io);const auto declared=std::move(declaration).freeze();
    auto ioBinding=declared->bind(memory,std::span{&ioKey,std::size_t{1}});
    {
        CpuAdaptiveImmutableRowCache* cpuCache=nullptr; // Passive fixture borrow; provider retains the sole unique facade.
        std::shared_ptr<HostImmutableRowProvider> provider;std::shared_ptr<q::ImmutableLookupService> service;
        std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary;std::shared_ptr<q::PooledImmutableLocalHotPreparedStorage> local;
        std::shared_ptr<const q::PooledImmutableDirectedTransport> transport;
        std::shared_ptr<const q::IPooledImmutableNativeKernels> poolCode;
        std::shared_ptr<const q::IPooledImmutableLocalHotKernels> localCode;
        std::array<std::shared_ptr<TensorBase>,4> roots;BufferArena arena;
        std::optional<DeviceGenerationControlBinding> control;
        worker->submitAndWait([&]{
            auto ownedCache=CpuAdaptiveImmutableRowCache::create(source,cpu,memory);
            cpuCache=ownedCache.get();
            provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(std::move(ownedCache));
            service=q::ImmutableLookupService::create(provider,channel,std::array{device},memory,DeviceId::cpu(),ioBinding->take(ioKey));
            const auto binding=service->bindSubordinate(device);
            const auto ownPlan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,binding.deviceView().base.table,binding.storageGeneration());
            const auto coordPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,binding.deviceView().base.table,binding.storageGeneration());
            auto owner=q::PooledImmutableOwnerPreparedStorage::create(ownPlan,memory,stream);
            auto coordinator=q::PooledImmutableCoordinatorPreparedStorage::create(coordPlan,memory,stream);
#ifdef HAVE_CUDA
            if(device.is_cuda()){poolCode=q::createCUDAPooledImmutableNativeKernels();localCode=q::createCUDAPooledImmutableLocalHotKernels();}
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm()){poolCode=q::createROCmPooledImmutableNativeKernels();localCode=q::createROCmPooledImmutableLocalHotKernels();}
#endif
            requireProvider(poolCode && localCode,"Real local/source native bundles missing");
            poolCode->initialize(*owner,stream);poolCode->initialize(*coordinator,stream);
            const auto sourcePlan=q::PooledImmutableSourceGeometry::fromPlans(ownPlan,coordPlan,source,1);
            boundary=q::ImmutableLookupExecutionBoundary::createPooled(service,sourcePlan,owner,coordinator,memory,stream);
            PooledImmutableFactoryReplayAccess::bindOriginalPreparedWorkers(*boundary->pooledStorage(),session->source());
            const auto& witness=*boundary->pooledStorage()->retainedPreparedSourceWorkers();
            requireProvider(witness.source()==session->source() && witness.source()->memory()==memory &&
                witness.rootIdentity()==boundary->pooledStorage()->captureIdentity() && witness.coordinator().worker()==worker &&
                witness.coordinator().generation()==original.generation() && witness.owners().size()==1 &&
                witness.owners()[0].worker()==worker && witness.owners()[0].generation()==original.generation(),
                "Actual original source preparation identity changed");
            local=q::PooledImmutableLocalHotPreparedStorage::create(localGeometry,boundary->pooledStorage(),memory,stream);localCode->initialize(*local,stream);
            const std::array roles{BufferId::QFN_NGRAM_ROW_IDS,BufferId::PREFILL_CHUNK_GEOMETRY,BufferId::MTP_GENERATION_CONTROL,BufferId::QFN_PLE_EMBEDDING};
            for(std::size_t i=0;i<roots.size();++i){
                roots[i]=std::shared_ptr<TensorBase>(declarations[i].materialize(memory));std::memset(roots[i]->raw_mutable_data(),0,roots[i]->size_bytes());
                TransferEngine::prepareDeviceInput(roots[i].get(),device,stream);TransferEngine::requireDeviceInput(roots[i].get(),device,stream);
                requireProvider(arena.registerExternalBuffer(roles[i],roots[i]),"Real local/source arena binding failed");
            }
            control.emplace(DeviceGenerationControlBinding::admit(arena,device,0));
        });
        transport=q::PooledImmutableDirectedTransport::create(boundary,{},{},{},compact);
        using original_l1_fork::requireProvider;
        ComputeGraph producerGraph;
        DeviceGraphExecutor executor;
        executor.setArena(&arena);
        std::unique_ptr<IDeviceContext> context;
        worker->submitAndWait([&]{context=IDeviceContext::create(device,1);});
        const auto tokenRows=DeviceRowRange::deviceCounted(Maximum/16,static_cast<const int*>(roots[1]->gpu_data_ptr()));
        const q::PooledNativeTensorSpan hashes{roots[0],BufferId::QFN_NGRAM_ROW_IDS,0,Maximum*8};
        const q::PooledNativeTensorSpan count{roots[1],BufferId::PREFILL_CHUNK_GEOMETRY,0,4};
        const q::PooledNativeTensorSpan output{roots[3],BufferId::QFN_PLE_EMBEDDING,0,roots[3]->size_bytes()};
        const auto body=appendPooledImmutableLocalHotFutureSourceBody(producerGraph,arena,transport,
            std::array{local},poolCode,local,localCode,tokenRows,hashes,count,*control,"actual_future_source");
        requirePooledSourceRetirementCompletion(producerGraph,body.pool_retirement);
        requirePooledLocalHotAuxiliary(producerGraph,body.auxiliary,transport,local);
        EXPECT_FALSE(producerGraph.retainedParticipantSourcePlan());
        EXPECT_FALSE(producerGraph.retainedParticipantAuxiliaryRange());
        std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>> parents(sourceAfterAdoption?6:4);
        for(auto& parent:parents)parent=RetainedParticipantGraphExecutableFixtureAccess::original(original,stream);
        std::vector<std::uint64_t> ordinals(parents.size());
        const auto join=[&]{worker->submitAndWait([&]{
            requireProvider(backend->synchronizeStream(stream,device.ordinal),"Future source diagnostic join failed");
        });};
        bool closed=false;
        const auto consumer=compact.consumer(0,local->consumerStorageIdentity());
        auto adoption=q::PooledImmutableLocalHotOperands::adoption(*local,consumer,tokenRows,hashes,count,
            transport->localHotAdmission(device),*control);
        auto read=q::PooledImmutableLocalHotOperands::read(*local,consumer,output,*control);
        auto frame=q::PooledImmutableLocalHotOperands::frame(*local,consumer,*control);
        struct Close {
            std::function<void()> finish;
            bool& closed;
            ~Close(){if(!closed)try{finish();}catch(...){std::terminate();}}
        } close{[&]{join();if(boundary->hasActiveInterval()){
            boundary->recordTerminalClose(stream);join();boundary->awaitTerminal();}parents={};},closed};
        worker->submitAndWait([&]{
            adoption.seal();read.seal();frame.seal();
            for(const auto& name:producerGraph.getExecutionOrder())
                requireProvider(producerGraph.getNode(name)->stage->prepareGraphLaunch(context.get(),stream),"Future source cold stage binding failed");
            requireProvider(executor.prepareGraphStorageForCapture(producerGraph,context.get(),stream,"actual future source"),"Future source storage binding failed");
            const auto names=producerGraph.getExecutionOrder();
            std::vector<IComputeStage*> stages;
            for(const auto& name:names)stages.push_back(producerGraph.getNode(name)->stage.get());
            for(const unsigned producerProfile:{0u,4u}){
                if(producerProfile==4 && !sourceAfterAdoption)continue;
                auto ledger=executor.planGraphCaptureDependencies(producerGraph,names,device,stream,"actual future source");
                requireProvider(bool(ledger),"Future source dependency plan missing");
                ScopedGraphCaptureStageActivity activity(stages,context.get(),stream,"actual future source");
                requireProvider(activity.begin(),"Future source activity begin failed");
                {
                    auto& native=RetainedParticipantGraphExecutableFixtureAccess::capture(parents[producerProfile]);
                    ScopedBackendGraphCapture capture(*worker,native,"actual future source",ledger.get());
                    requireProvider(capture.begin(),"Future source capture begin failed");
                    for(const auto& name:names){
                        {
                            auto* stage=producerGraph.getNode(name)->stage.get();
                            ScopedGraphCaptureStage scope(stage,stream);
                            requireProvider(stage->execute(context.get()),"Future source native stage recording failed");
                            scope.complete();
                        }
                        // Same actual demand IDs, after the genuine Probe and
                        // before immutable Future metadata/Union/Begin/Pack.
                        // No source completion or response is manufactured.
                        if(producerProfile==4 && name==body.probe)
                            localCode->enqueue(q::PooledImmutableLocalHotAction::Adopt,*local,adoption,stream);
                    }
                    capture.finish();
                    requireProvider(native.prepareRuntimeContextStorage(memory)&&native.instantiate(),"Future source original instantiation failed");
                }
                requireProvider(activity.finish(GraphCaptureActivityTransition::Completed),"Future source activity finish failed");
            }
            for(unsigned profile=1;profile<parents.size();++profile){
                if(profile==4)continue;
                auto& native=RetainedParticipantGraphExecutableFixtureAccess::capture(parents[profile]);
                ScopedBackendGraphCapture capture(*worker,native,"actual future consumer");
                requireProvider(capture.begin(),"Future consumer capture begin failed");
                if(profile!=3 && profile!=5)localCode->enqueue(q::PooledImmutableLocalHotAction::Adopt,*local,adoption,stream);
                if(profile==1 || profile==5){
                    local->enqueueReadyWait(stream);
                    localCode->enqueue(q::PooledImmutableLocalHotAction::LocalRead,*local,read,stream);
                    local->enqueueLocalReadCompleteWait(stream);
                    for(const auto action:{q::PooledImmutableLocalHotAction::ConsumeDemand,
                        q::PooledImmutableLocalHotAction::CopyFills,q::PooledImmutableLocalHotAction::CompleteFills})
                        localCode->enqueue(action,*local,frame,stream);
                }else if(profile==3)localCode->enqueue(q::PooledImmutableLocalHotAction::Discard,*local,frame,stream);
                // The source parent is physically observed before any consumer
                // submission below. No host boolean mints a source completion.
                localCode->enqueue(q::PooledImmutableLocalHotAction::Retire,*local,frame,stream);
                capture.finish();
                requireProvider(native.prepareRuntimeContextStorage(memory)&&native.instantiate(),"Future consumer original instantiation failed");
            }
        });
        join();
        const auto submit=[&](unsigned profile){
            requireProvider(RetainedParticipantGraphExecutableFixtureAccess::submit(parents[profile],++ordinals[profile]),"Future original parent submission failed");
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(parents[profile])){
                requireProvider(std::chrono::steady_clock::now()<deadline,"Future original parent terminal deadline");
                std::this_thread::yield();
            }
        };
        const auto observe=[&]<class T>(const std::shared_ptr<TensorBase>& root,std::size_t offset){
            T value{};
            worker->submitAndWait([&]{requireProvider(backend->deviceToHostOnStream(&value,
                static_cast<const std::byte*>(root->gpu_data_ptr())+offset,sizeof(T),device.ordinal,stream),"Future terminal observation failed");});
            join();return value;
        };
        // Observe only at already-proven native parent terminals. Normalize
        // the genuine pin flag; every policy, directory and payload byte must
        // otherwise remain unchanged by an unused future producer.
        const auto cacheSnapshot=[&]{
            std::vector<std::byte> bytes(localGeometry.cacheOffsets().bytes);
            worker->submitAndWait([&]{requireProvider(backend->deviceToHostOnStream(bytes.data(),
                local->roots().cache->gpu_data_ptr(),bytes.size(),device.ordinal,stream),
                "Future terminal cache observation failed");});
            join();
            for(std::uint32_t slot=0;slot<Slots;++slot){
                const auto at=localGeometry.cacheOffsets().slots+slot*sizeof(q::adaptive_lookup::Slot32);
                q::adaptive_lookup::Slot32 value{};std::memcpy(&value,bytes.data()+at,sizeof(value));
                value.flags&=~q::adaptive_lookup::FramePin;std::memcpy(bytes.data()+at,&value,sizeof(value));
            }
            return bytes;
        };
        const auto upload=[&](std::span<const std::uint64_t> ids){
            requireProvider(ids.size()%16==0&&ids.size()<=Maximum,"Future original extent invalid");
            file.requireAuthored(ids);
            std::array<std::uint64_t,Maximum> complete{};std::copy(ids.begin(),ids.end(),complete.begin());
            const int tokens=static_cast<int>(ids.size()/16);
            worker->submitAndWait([&]{
                requireProvider(backend->hostToDeviceOnStream(roots[0]->gpu_data_ptr(),complete.data(),sizeof(complete),device.ordinal,stream),"Future authored hashes upload failed");
                requireProvider(backend->hostToDeviceOnStream(roots[1]->gpu_data_ptr(),&tokens,4,device.ordinal,stream),"Future authored count upload failed");
                for(const auto& root:{roots[0],roots[1]})TransferEngine::publishDeviceWrite(root.get(),device,stream);
            });
        };
        const auto cacheIdentity=local->captureIdentity();
        const auto coordinator=boundary->pooledStorage()->coordinator();
        for(std::uint64_t request=1;request<=2;++request){
            // Host source/cache observations belong to a retired service
            // interval. Native parent completion alone does not transfer the
            // retained CPU worker's statistics ownership to this fixture.
            const auto cpuBefore=cpuCache->stats();
            const auto serviceBefore=service->stats();
            std::uint64_t expectedSourceRequests=0,expectedSourceRows=0;
            boundary->beginAdmission(request,stream);
            worker->submitAndWait([&]{transport->enqueueLocalHotAdmission(local,localCode,*control,stream);});
            join();
            /** @brief Fetch through actual union/source/ACK, then adopt, mismatch or discard after exact source terminal. */
            const auto replay=[&](std::span<const std::uint64_t> forecast,std::span<const std::uint64_t> demand,
                unsigned consumerProfile,std::uint64_t expectedNativeRows,bool requirePressure=false){
                const auto lowerBefore=observe.template operator()<q::adaptive_lookup::pooled::CoordinatorPolicyControl>(
                    coordinator->roots().cache,coordinator->plan().parameters().cache.control);
                const auto countersBefore=observe.template operator()<h::Counters>(local->roots().frame,localGeometry.frameOffsets().counters);
                std::vector<float> poison(Maximum*file.columns,-12345.f);
                worker->submitAndWait([&]{
                    requireProvider(backend->hostToDeviceOnStream(roots[3]->gpu_data_ptr(),poison.data(),poison.size()*4,device.ordinal,stream),"Future output poison upload failed");
                    TransferEngine::publishDeviceWrite(roots[3].get(),device,stream);
                });
                const auto localPolicyBefore=cacheSnapshot();
                upload(forecast);submit(sourceAfterAdoption && consumerProfile==1?4:0);
                EXPECT_EQ(cacheSnapshot(),localPolicyBefore);
                const auto futureCounters=observe.template operator()<h::Counters>(local->roots().frame,localGeometry.frameOffsets().counters);
                EXPECT_EQ(futureCounters.promotions,countersBefore.promotions);
                EXPECT_EQ(futureCounters.evictions,countersBefore.evictions);
                EXPECT_EQ(futureCounters.consumed_original_ids,countersBefore.consumed_original_ids);
                EXPECT_EQ(futureCounters.read_original_ids,countersBefore.read_original_ids);
                const auto fetched=observe.template operator()<h::FrameIdentity>(local->roots().frame,localGeometry.frameOffsets().identity);
                const auto gathered=observe.template operator()<q::adaptive_lookup::FrameControl>(local->roots().frame,localGeometry.frameOffsets().control);
                EXPECT_EQ(fetched.disposition,h::Disposition::DataReady);
                if(sourceAfterAdoption && consumerProfile==1){
                    EXPECT_EQ(fetched.intent,q::PooledImmutableLocalHotAccessIntent::CurrentInvocation);
                    EXPECT_NE(fetched.flags&h::FutureOrigin,0u);
                    EXPECT_EQ((fetched.flags&h::LateAdoption)!=0,expectedNativeRows!=0);
                }
                EXPECT_EQ(gathered.source_rows,expectedNativeRows);
                EXPECT_EQ(fetched.retained_native_bytes,expectedNativeRows*file.row_bytes);
                const auto lowerAfter=observe.template operator()<q::adaptive_lookup::pooled::CoordinatorPolicyControl>(
                    coordinator->roots().cache,coordinator->plan().parameters().cache.control);
                EXPECT_EQ(std::memcmp(&lowerBefore,&lowerAfter,sizeof(lowerBefore)),0);
                expectedSourceRequests+=expectedNativeRows?1u:0u;
                expectedSourceRows+=expectedNativeRows;
                // Deliberate mismatch supplies real ordered demand with the SAME
                // set/count; no manufactured response or source receipt follows.
                upload(demand);submit(sourceAfterAdoption && consumerProfile==1?5:consumerProfile);
                const auto retired=observe.template operator()<h::FrameIdentity>(local->roots().frame,localGeometry.frameOffsets().identity);
                const auto finalFrame=observe.template operator()<q::adaptive_lookup::FrameControl>(local->roots().frame,localGeometry.frameOffsets().control);
                const auto countersAfter=observe.template operator()<h::Counters>(local->roots().frame,localGeometry.frameOffsets().counters);
                EXPECT_EQ(retired.disposition,h::Disposition::Idle);EXPECT_EQ(finalFrame.held_pins,0u);
                EXPECT_EQ(retired.terminal_error,q::adaptive_lookup::Status::Success);
                EXPECT_EQ(local->captureIdentity(),cacheIdentity);
                std::vector<float> actual(poison.size());
                worker->submitAndWait([&]{requireProvider(backend->deviceToHostOnStream(actual.data(),roots[3]->gpu_data_ptr(),actual.size()*4,device.ordinal,stream),"Future LocalRead terminal bytes failed");});
                join();
                if(consumerProfile==1){
                    std::vector<std::byte> rows;
                    for(const auto id:demand){const auto row=file.original(id);rows.insert(rows.end(),row.begin(),row.end());}
                    std::vector<float> expected(demand.size()*file.columns);
                    if(!demand.empty())q::decodeCPUImmutableLookupRows(file.type,static_cast<int>(file.columns),file.row_bytes,rows.data(),demand.size(),expected.data());
                    if(!expected.empty())EXPECT_EQ(std::memcmp(actual.data(),expected.data(),expected.size()*4),0);
                    for(std::size_t at=expected.size();at<actual.size();++at)EXPECT_EQ(actual[at],poison[at]);
                    if(requirePressure){
                        EXPECT_GT(countersAfter.evictions,countersBefore.evictions);
                        EXPECT_GT(countersAfter.promotions,countersBefore.promotions);
                    }
                    EXPECT_EQ(countersAfter.read_original_ids-countersBefore.read_original_ids,demand.size());
                    EXPECT_EQ(countersAfter.consumed_original_ids-countersBefore.consumed_original_ids,demand.size());
                    EXPECT_EQ(countersAfter.future_useful_native_rows-countersBefore.future_useful_native_rows,expectedNativeRows);
                }else{
                    EXPECT_EQ(cacheSnapshot(),localPolicyBefore);
                    EXPECT_EQ(actual,poison);
                    EXPECT_EQ(countersAfter.read_original_ids,countersBefore.read_original_ids);
                    EXPECT_EQ(countersAfter.consumed_original_ids,countersBefore.consumed_original_ids);
                    EXPECT_EQ(countersAfter.local_hit_original_ids,countersBefore.local_hit_original_ids);
                    EXPECT_EQ(countersAfter.all_hit_wait_avoided_original_ids,countersBefore.all_hit_wait_avoided_original_ids);
                    EXPECT_EQ(countersAfter.promotions,countersBefore.promotions);EXPECT_EQ(countersAfter.evictions,countersBefore.evictions);
                    EXPECT_EQ(countersAfter.future_wasted_native_rows-countersBefore.future_wasted_native_rows,expectedNativeRows);
                    if(consumerProfile==2)EXPECT_NE(retired.flags&h::PredictionDiscard,0u);
                }
            };
            std::array<std::uint64_t,16> hot{};hot.fill(7);
            std::array<std::uint64_t,32> mixed{};std::fill_n(mixed.begin(),16,7);std::fill_n(mixed.begin()+16,16,8);
            std::array<std::uint64_t,32> unused{};std::fill_n(unused.begin(),16,9);std::fill_n(unused.begin()+16,16,10);
            auto reversed=unused;std::rotate(reversed.begin(),reversed.begin()+16,reversed.end());
            replay(hot,hot,1,request==1?1:0);
            replay(hot,hot,3,0); // Unused initial hit pins survive fetch, then retire without policy credit.
            replay(hot,hot,1,0);
            replay(mixed,mixed,1,request==1?1:0);
            replay(unused,unused,3,2);
            replay(unused,reversed,2,2);
            replay({}, {},1,0);
            // Both slots are occupied by 7/8. The actual Future probe pins 7;
            // unused forecast must not evict 8, then real demand may replace
            // only that unpinned row. Repeated read proves native slot reuse.
            if(request==2){
                auto pressure=mixed;std::fill_n(pressure.begin()+16,16,9);
                replay(pressure,pressure,3,1);
                // The unchanged scan-resistant admission rule requires strictly
                // greater demand frequency than resident 8 (two prior reads).
                // Future fetch/discard contributes nothing to that frequency.
                replay(pressure,pressure,1,1);
                replay(pressure,pressure,1,1);
                replay(pressure,pressure,1,1,true);
                replay(pressure,pressure,1,0);
            }
            boundary->recordTerminalClose(stream);join();boundary->awaitTerminal();
            const auto cpuAfter=cpuCache->stats();
            const auto serviceAfter=service->stats();
            EXPECT_EQ(serviceAfter.requests-serviceBefore.requests,expectedSourceRequests);
            EXPECT_EQ(serviceAfter.ids-serviceBefore.ids,expectedSourceRows);
            EXPECT_EQ(serviceAfter.idBytes-serviceBefore.idBytes,expectedSourceRows*sizeof(std::uint64_t));
            EXPECT_EQ(serviceAfter.nativeBytes-serviceBefore.nativeBytes,expectedSourceRows*file.row_bytes);
            EXPECT_EQ(cpuAfter.completed_rows-cpuBefore.completed_rows,expectedSourceRows);
            EXPECT_EQ(cpuAfter.completed_bytes-cpuBefore.completed_bytes,expectedSourceRows*file.row_bytes);
            EXPECT_EQ(cpuAfter.admissions,cpuBefore.admissions);EXPECT_EQ(cpuAfter.evictions,cpuBefore.evictions);
            EXPECT_EQ(cpuAfter.invalidations,cpuBefore.invalidations);EXPECT_EQ(cpuAfter.resident_rows,0u);
        }
        parents={};producerGraph.clear();context.reset();transport.reset();local.reset();boundary.reset();
        service.reset();cpuCache=nullptr;provider.reset();roots={};join();closed=true;
        std::cout<<"LOCAL_HOT_FUTURE_SOURCE device="<<device.toString()<<" real_ssd_source=1 source_drained_before_consumer=1 injected_response=0 requests=2 multi_consumer_model_qualified=0\n";
    }
    session->requirePayloadRetired();worker.reset();session->retire();session->requireExactRetired();
}
} // namespace llaminar2::test::future_l1_source
namespace llaminar2::test {
#ifdef HAVE_CUDA
TEST(LocalHotNativeParentFork,CUDA_FutureOriginalSourceAdoptDiscard){future_l1_source::replayFutureSource(DeviceId::cuda(0));}
TEST(LocalHotNativeParentFork,CUDA_FutureOriginalSourceAfterAdoption){future_l1_source::replayFutureSource(DeviceId::cuda(0),true);}
#endif
#ifdef HAVE_ROCM
TEST(LocalHotNativeParentFork,ROCm_FutureOriginalSourceAdoptDiscard){future_l1_source::replayFutureSource(DeviceId::rocm(0));}
TEST(LocalHotNativeParentFork,ROCm_FutureOriginalSourceAfterAdoption){future_l1_source::replayFutureSource(DeviceId::rocm(0),true);}
#endif

} // namespace llaminar2::test
