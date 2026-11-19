/**
 * @file MoEPromoted23NativeLayoutReplay.inc
 * @brief Original CPU-to-ROCm fragment arrival, epoch publication and native FFN replay.
 *
 * The existing physical fabric owns all source loading, conversion, destination
 * slots and completion. Both captured implementations borrow its same published
 * candidate bank; no test-authored descriptor can create execution readiness.
 */

namespace promoted23_replay {
using namespace original_fragment_fixture;

/** @brief Cold publication boundary upgraded to the actual committed inference event when full graphs exist. */
class ColdPublicationBoundary final : public IMoEOverlayDeviceInferenceBoundary,
                                      public IMoEOverlayDeviceInitialRuntimePublisher {
    std::function<bool(void*)> inference_terminal_;
public:
    /** @brief Replace the cold boundary with the actual already-captured inference event authority. */
    void bindInferenceTerminal(std::function<bool(void*)> terminal) {inference_terminal_=std::move(terminal);}
    /** @return The original sole layer was published before this callback. */
    bool publishMoEOverlayDeviceInitialRuntime(void* stream) override { return stream!=nullptr; }
    /** @return Maintenance follows the exact committed inference event, or the original cold no-inference state. */
    MoEOverlayInferenceBoundaryStatus enqueueMoEOverlayDeviceInferenceBoundary(
        void* stream,MoEOverlayInferenceBoundaryRequest) override {
        if(!stream)return MoEOverlayInferenceBoundaryStatus::Failed;
        if(inference_terminal_&&!inference_terminal_(stream))return MoEOverlayInferenceBoundaryStatus::Failed;
        return MoEOverlayInferenceBoundaryStatus::Submitted;
    }
    /** @return Whether the actual transfer progress owner is present. */
    bool installMoEOverlayTransferProgressEpoch(std::shared_ptr<MappedTransferProgressEpoch> epoch) override {
        return bool(epoch);
    }
};

/** @brief Poll one existing event-based lifecycle without stream/device waits. */
template<class Poll> void await(Poll&& poll,const char* what) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        std::string error;
        const auto state=poll(&error);
        require(state!=Progress::Failed,std::string(what)+": "+error);
        if(state==Progress::Ready)return;
        require(std::chrono::steady_clock::now()<deadline,what);
        std::this_thread::yield();
    }
}

/** @brief Select original standalone FFN replay or retained full numerical epoch transition. */
enum class NumericalProgram { NativeLayout, FullHCFFNEpoch };
/** @brief Genuine source route for this numerical fixture, never a destination format override. */
enum class EpochSource { CPUPromoted23, GPUNative5_7, GPUNativeCohort2, GPUNativeCohort2WithDescriptorReader, GPUNativeRealTP4, GPUNativeCohort2Q4Q8, GPUNativeCohort2Q5Q8 };
/**
 * @brief Execute the original physical fabric and retained numerical publication lifecycle.
 * @param program Existing standalone or complete captured numerical observer.
 * @param pci Optional exact diagnostic GPU cohort identities; rank ownership stays original.
 * @param epoch_source Genuine CPU-promoted or native GPU source/geometry selection.
 */
void run(NumericalProgram program=NumericalProgram::NativeLayout,std::span<const std::string_view> pci={},
    EpochSource epoch_source=EpochSource::CPUPromoted23,RealRetainedTPSourceWindow* real_window=nullptr) {
    const bool gpu_native=epoch_source!=EpochSource::CPUPromoted23;
    const bool q5_q8=epoch_source==EpochSource::GPUNativeCohort2Q5Q8;
    const bool q8_down=q5_q8||epoch_source==EpochSource::GPUNativeCohort2Q4Q8;
    const bool cohort2=q8_down||epoch_source==EpochSource::GPUNativeCohort2||
        epoch_source==EpochSource::GPUNativeCohort2WithDescriptorReader;
    const bool real_tp4=epoch_source==EpochSource::GPUNativeRealTP4;
    const bool native_cohorts=cohort2||real_tp4;
    const std::size_t degree=cohort2?2u:1u;
    require(pci.empty()||pci.size()==(real_tp4?6u:gpu_native?3u*degree:2u),
        "original source fixture declares one PCI per requested GPU cohort");
    require(!gpu_native||program==NumericalProgram::FullHCFFNEpoch,
        "native GPU-source epoch fixture requires the retained complete numerical program");
    require(!real_window||(gpu_native&&!cohort2),"original real TP4 sources move across declared native owners");
    require(!real_tp4||(real_window&&real_window->originalLayer()==0&&pci.size()==6),
        "complete real TP4 movement first proves actual layer0 rotation and six original GPU owners");
    const OriginalFFNFixtureGeometry geometry=real_tp4?
        OriginalFFNFixtureGeometry{2560,640,640,512,10,OriginalFFNRouterSource::TieControl,OriginalFFNGeometryContract::QFNTensorSharded}:
        real_window?real_window->geometry():gpu_native
        ?OriginalFFNFixtureGeometry{2560,cohort2?512:256,cohort2?384:192,512,10,OriginalFFNRouterSource::TieControl}
        :OriginalFFNFixtureGeometry{256,640,256,6,2,OriginalFFNRouterSource::TieControl};
    std::map<DeviceId,FrozenModelWeightSet> shared;
    std::map<DeviceId,std::shared_ptr<PublishedFFNEpoch>> published;
    std::vector<std::unique_ptr<ColdPublicationBoundary>> boundaries;
#if defined(HAVE_ROCM)
    std::map<DeviceId,std::shared_ptr<FullEpochNumericalPair>> numerical;
#endif
    // Shared graph weights must retire before OriginalFragmentFabric retires
    // its original native contexts, including exceptional setup exits.
    struct NativeDomain {
        std::vector<DeviceId> devices;
        std::shared_ptr<ILocalTPContext> communicator;
        TensorParallelConfig shared_partition;
    };
    std::map<std::string,NativeDomain> local_domains;
    std::map<DeviceId,std::string> domain_of;
    struct SourceBorrowRetirement {
        std::map<std::string,NativeDomain>& domains;
        std::map<DeviceId,FrozenModelWeightSet>& weights;
        std::map<DeviceId,std::shared_ptr<PublishedFFNEpoch>>& epochs;
        std::vector<std::unique_ptr<ColdPublicationBoundary>>& boundaries;
#if defined(HAVE_ROCM)
        std::map<DeviceId,std::shared_ptr<FullEpochNumericalPair>>& numerical;
#endif
        /** @brief All native graph/transaction locals retire before their original source context. */
        ~SourceBorrowRetirement(){
#if defined(HAVE_ROCM)
            numerical.clear();
#endif
            domains.clear();epochs.clear();boundaries.clear();weights.clear();}
    };
    OriginalFragmentPreparationHooks hooks;
    if(native_cohorts)hooks.authored_plan=[&](const MoERoutedExpertPlacementPlan& plan,int rank) {
        for(const auto& d:plan.domains) {
            if(d.owner_rank!=rank)continue;
            require(d.routed_compute_policy==RoutedExpertComputePolicy::TensorSharded&&
                d.participants.size()==(real_tp4&&d.name=="warm"?4u:degree),
                "actual complete original local GPU cohort");
            auto& domain=local_domains[d.name];
            for(const auto& member:d.participants) {
                domain.devices.push_back(member.toLocalDeviceId());
                require(domain_of.emplace(domain.devices.back(),d.name).second,"original disjoint GPU cohorts");
            }
            domain.shared_partition=TensorParallelConfig::equalSplit(domain.devices.size(),8,2,geometry.shared,320,domain.devices);
        }
    };
    hooks.geometry=OriginalFragmentSourceGeometry{geometry.hidden,geometry.routed,geometry.shared,geometry.experts,geometry.top_k};
    hooks.shared_type=GGUFTensorType::Q8_0;
    if(q8_down) {
        const auto gate=q5_q8?GGUFTensorType::Q5_K:GGUFTensorType::Q4_K;
        hooks.routed_source_types=std::array{gate,gate,GGUFTensorType::Q8_0};
        hooks.generate_declared_routed_sources=true;
    }
    if(real_window) {
        const auto format=real_window->format();
        const auto gate=PlanningGGUFFixture::sourceType(format.type);
        hooks.routed_source_types=std::array{gate,gate,
            PlanningGGUFFixture::sourceType(format.down_quantized->tensor_type)};
    }
    hooks.source=[&](PlanningGGUFFixture& file) {
        if(real_window)real_window->copyTo(file,geometry,real_window->format(),RealRetainedWindowRoles::NativeProjections,
            real_tp4?RealRetainedWindowExtent::WholeLayer:RealRetainedWindowExtent::OriginalShard);
        // Author only shared/router tensors; routed source remains the exact
        // existing Q4_K/Q4_K/Q5_1 physical-fabric fixture payload.
        for(size_t role=real_window?3u:0u;role<kSharedNames.size();++role) {
            if(role<3) {
                const auto n=role==2?geometry.hidden:geometry.shared;
                const auto k=role==2?geometry.shared:geometry.hidden;
                const auto values=TestTensorFactory::createQ8_0Random({size_t(n),size_t(k)},7103+role);
                file.writePayload(kSharedNames[role],0,{static_cast<const uint8_t*>(values->raw_data()),values->size_bytes()});
            } else {
                std::vector<float> values(role==3?geometry.hidden:geometry.experts*geometry.hidden,0.f);
                if(role==3)for(size_t i=0;i<values.size();++i)values[i]=float(int(i%19)-9)/128.f;
                else for(int expert=0;expert<geometry.experts;++expert) {
                    if(program==NumericalProgram::FullHCFFNEpoch) {
                        // Genuine router source, not selected-ID substitution: a
                        // positive HC mix gives distinct bounded4/3 logits.
                        for(int h=0;h<geometry.hidden;++h)values[size_t(expert)*geometry.hidden+h]=
                            gpu_native?(expert<6?float(8-expert)/512.f:-1.f/128.f):
                            (expert==4?1.f/64.f:expert==3?1.f/128.f:-1.f/128.f);
                    } else values[size_t(expert)*geometry.hidden]=expert==4?2.f:expert==3?1.f:-1.f;
                }
                file.writePayload(kSharedNames[role],0,{reinterpret_cast<const uint8_t*>(values.data()),values.size()*sizeof(float)});
            }
        }
    };
    hooks.before_admission=[&](ModelContext& context,PhysicalMemoryPlanBuilder& budget,
        const std::vector<PhysicalMemoryResource>& resources,const std::vector<DeviceId>& devices) {
        auto& manager=*context.concreteWeightManager();
        manager.setWeightShardingConfig(Qwen35MoESchemaFactory{}.getWeightShardingConfig());
        const ModelContextId model{reinterpret_cast<uint64_t>(&context)};
        // GPU-only preparation never enters the CPU source population which
        // binds this store automatically in the physical23 fixture. Use the
        // same original model identity before declaring or preparing any weights.
        require(manager.preparedWeightStore()->bindModelIdIfUnset(model),
            "physical epoch shares one original model store across source roles");
        const auto host=std::find_if(resources.begin(),resources.end(),[](const auto& r){return r.device.is_cpu();});
        require(host!=resources.end(),"promoted original CPU allocator");
        for(const auto device:devices) {
            // All local source cohorts are singleton; this does not repartition
            // an existing source cohort or invent a representative member.
            const std::vector<DeviceId> singleton{device};
            const auto& participants=native_cohorts?local_domains.at(domain_of.at(device)).devices:singleton;
            manager.setTensorParallelConfig(std::make_shared<TensorParallelConfig>(
                TensorParallelConfig::equalSplit(participants.size(),8,2,geometry.shared,320,participants)));
            auto plan_source=sharedPlan(model,participants,device);
            // This source is an independently declared singleton replica. Its
            // full range is authored by SingleDevice, with no invented TP row.
            shared.emplace(device,manager.materialize(plan_source));
            WeightVRAMPool plan;
            for(const auto& binding:shared.at(device).bindings()) {
                const auto* bounded=boundedGPUWeightSource(*binding.tensor);
                const auto* native=gpuWeightSourceFormat(*binding.tensor);
                const auto source_bytes=bounded?bounded->sourceBytes():binding.tensor->size_bytes();
                if(native)plan.planWeight(binding.identity.canonical_name,binding.tensor->rows(),binding.tensor->cols(),
                    native->payload_bytes,canonicalDeviceVnniFormat(*native).is_asymmetric,native->has_emins,source_bytes);
                else plan.planRawWeight(binding.identity.canonical_name,binding.tensor->rows(),binding.tensor->cols(),binding.tensor->size_bytes());
            }
            const auto gpu=std::find_if(resources.begin(),resources.end(),[&](const auto& r){return r.device==device;});
            require(gpu!=resources.end(),"promoted original GPU allocator");
            const auto staging=resolveGPUWeightLoadMemoryGeometry(plan.maximumPlannedStagingBytes(),configuredGPUWeightLoadMemoryPolicy());
            const auto route_bytes=DeviceMoESerialRouteScratchArena::plannedAllocationBytes(
                {device,geometry.experts,geometry.top_k,1});
            const auto runtime_templates=DeviceMoERuntimeTable::kStateTemplateCopies*sizeof(DeviceMoELayerRuntime);
            // This API's host component consists only of object/templates,
            // flags, one scratch-binding record and one family shared_ptr;
            // it contains no route-count terms (MoERuntimeTable.cpp). The
            // device route allocation above uses the actual source topK geometry.
            const auto runtime_host=DeviceMoERuntimeTable::preparedFragmentServiceStorageGeometry(device,geometry.experts,1).host_bytes;
            budget.add(*gpu,PhysicalMemoryOwner::PrimaryModelWeights,alignGPUWeightLoadAllocation(plan.totalPlannedBytes()))
                .add(*gpu,PhysicalMemoryOwner::WeightLoadStaging,staging.staging_bytes)
                .add(*host,PhysicalMemoryOwner::WeightLoadStaging,staging.host_staging_bytes)
                .add(*host,PhysicalMemoryOwner::ExecutionWorkspace,runtime_host)
                .add(*gpu,PhysicalMemoryOwner::ExecutionWorkspace,route_bytes+runtime_templates+
                    DeviceMoEOverlayEpochArena::allocationBytesForSlots(1));
        }
    };
    hooks.prepared=[&](ModelContext& context,const std::vector<DeviceId>& devices) {
        for(const auto device:devices)
            require(context.concreteWeightManager()->prepareWeightsForDevice(shared.at(device),device),
                "prepare actual shared Q8/router source under original PMA");
    };
    // GPU-source qualification retains real rank ownership even when its
    // closed rotation crosses sockets. The existing CPU-source fixture keeps
    // its original two same-rank GPU cohorts and real dual-socket capacity.
    std::array requests{
        OriginalFragmentTargetRequest{"hot",DeviceType::ROCm,degree,{},!cohort2&&!pci.empty()?std::string(pci[0]):std::string{}},
        OriginalFragmentTargetRequest{"warm",DeviceType::ROCm,degree,gpu_native?std::string{}:std::string{"hot"},
            !cohort2&&!pci.empty()?std::string(pci[1]):std::string{}},
        gpu_native
            ?OriginalFragmentTargetRequest{"capacity",DeviceType::ROCm,degree,{},!cohort2&&!pci.empty()?std::string(pci[2]):std::string{}}
            :OriginalFragmentTargetRequest{"capacity",DeviceType::CPU,2}};
    if(real_tp4) {
        requests[0].pci_address=std::string(pci[0]);
        requests[1].pci_address.clear();
        requests[1].degree=4;
        requests[1].pci_addresses.assign(pci.begin()+1,pci.begin()+5);
        requests[2].pci_address=std::string(pci[5]);
    }
    if(cohort2)for(std::size_t i=0;i<requests.size();++i) {
        require(!pci.empty(),"exact original PCI roster for this NUMA-local diagnostic");
        requests[i].pci_addresses.assign(pci.begin()+i*degree,pci.begin()+(i+1)*degree);
    }
    OriginalFragmentFabric original(requests,true,hooks);
    SourceBorrowRetirement source_retirement{local_domains,shared,published,boundaries
#if defined(HAVE_ROCM)
        ,numerical
#endif
    };
    for(auto& [name,domain]:local_domains) {
        (void)name;
        std::vector<GlobalDeviceAddress> addresses;
        for(const DeviceId device:domain.devices)addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
        const bool collective=domain.devices.size()>1;
        domain.communicator=createLocalTPContext(addresses,{},collective?CollectiveBackendType::RCCL:CollectiveBackendType::AUTO);
        require(domain.communicator&&domain.communicator->degree()==int(domain.devices.size())&&
            (!collective||domain.communicator->backend()==CollectiveBackendType::RCCL),"actual original native local cohort");
    }
    std::exception_ptr native_failure;
    const auto abort_native=[&](std::exception_ptr first) {
        if(!native_failure)native_failure=first;
        for(auto& [name,domain]:local_domains) {
            (void)name;
            if(domain.communicator&&!domain.communicator->signalNativeCompletionAbort())std::terminate();
        }
    };
    const auto source_metadata=PlanningModelMetadata::fromLoader(original.context->concreteLoader());
    require(source_metadata.memoryProfile().d_model==geometry.hidden &&
        source_metadata.memoryProfile().expert_count==geometry.experts &&
        source_metadata.memoryProfile().expert_used_count==geometry.top_k &&
        source_metadata.memoryProfile().expert_feed_forward_length==geometry.routed,
        "promoted replay uses actual original model geometry");
    // The original reverse rotation moves capacity expert4 to the warm
    // cohort and preserves warm expert3. All source/destination cuts remain
    // authored by the existing physical family and movement authority.
    const auto transaction=original.transaction(true);
    require(transaction.valid() && transaction.previous->epoch==1 && transaction.candidate->epoch==2,
        "original physical fabric authors the exact first candidate epoch");
    ComponentWave wave(original,transaction);
    pollComponent(wave);
    std::string error;
    for(const int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);
        auto bank=endpoint->cloneCandidate(transaction.previous->epoch,transaction.candidate->epoch);
        for(const auto& migration:transaction.migrations)
            if(bank.layerForModelLayer(migration.layer_idx).resident_mask.at(migration.expert_id))
                bank.layerForModelLayer(migration.layer_idx).clearExpert(migration.expert_id);
        for(size_t i=0;i<transaction.migrations.size();++i)
            for(const auto& destination:wave.prepared.migrations[i].destination_arrivals)if(destination.participant_id==id) {
                MoEOverlayPreparedFragmentPayload payload;
                require(destination.arrival->completeFragment(payload,&error),"complete original physical fragment: "+error);
                bank.layerForModelLayer(transaction.migrations[i].layer_idx).setResidentFragment(transaction.migrations[i].expert_id,std::move(payload));
            }
        auto ready=endpoint->prepareReadyBank(std::move(bank),&error);
        require(ready.has_value(),"prepare original candidate bank: "+error);
        require(endpoint->installReadyBank(std::move(*ready),&error)==MoEOverlayParticipantBankInstallStatus::Installed,
            "install original candidate bank: "+error);
    }
    std::vector<MoEOverlayDeviceControllerRuntimeBinding> bindings;
    std::size_t local_promoted=0;
    // Every runtime below belongs to the original source/PMA and is retained
    // until the device-bank publication and all imported inference retire.
    for(const auto device:original.local_gpus) {
        const auto& members=original.registry->initialOwnerMap().participants();
        const auto found=std::find_if(members.begin(),members.end(),[&](const auto& p){return p.device==device&&p.world_rank==original.mpi->rank();});
        require(found!=members.end(),"published runtime original member");
        const auto* member=&*found;
        auto state=std::make_shared<PublishedFFNEpoch>();
        auto initial=original.registry->endpoint(member->participant_id)->acquire(1);
        state->bank=original.registry->endpoint(member->participant_id)->acquire(transaction.candidate->epoch);
        require(initial&&state->bank,"retain actual original and candidate banks");
        auto& worker=GPUDeviceContextPool::instance().getContext(device);
        worker.submitAndWait([&] {
            state->arena=std::make_shared<DeviceMoEOverlayEpochArena>(DeviceMoEOverlayEpochArena::Config{
                .device_id=device,.memory_authority=original.memory,.initial_epoch=1,.initial_bank=1,.request_slot_capacity=1});
            state->runtime=std::make_shared<DeviceMoERuntimeTable>(DeviceMoERuntimeTable::Config{
                .device_id=device,.num_layers=1,.num_experts=geometry.experts,.top_k=geometry.top_k,
                .mirror_to_device=true,.prefill_token_capacity=1,.overlay_epoch_arena=state->arena,
                .fragment_families={initial->layers.front().fragment_family}});
            const auto stream=worker.defaultStream();
            publishInitialMoEIntermediateShardRuntimeLayer(*state->runtime,original.registry->initialOwnerMap(),
                initial->layerForModelLayer(0),0,stream);
            state->runtime->sealAndPublishCompleteInitialRuntimeState(stream);
            ScopedGPUCompletionEvent completion(getBackendFor(device),device,stream);
            require(completion.join(),"complete original runtime setup before cold publication");
        });
        for(const auto& migration:transaction.migrations) {
            const auto& source_address=migration.source.shard_cohort->participants().front().address;
            const bool expected_source=gpu_native?source_address.toLocalDeviceId().is_gpu():source_address.isCPU();
            if(expected_source&&state->bank->layers.front().resident_mask.at(migration.expert_id)) {
                state->promoted_experts.push_back(migration.expert_id);
                if(gpu_native) {
                    const auto& fragment=state->bank->layerForModelLayer(0).fragments.at(migration.expert_id);
                    std::array<DeviceNativeQuantisedMatrixDesc,3> arrived;
                    const auto declaration=MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(
                        state->bank->layers.front().fragment_family);
                    require(fragment.gate()->exportNativeQuantisedMatrixDesc(arrived[0])&&
                        fragment.up()->exportNativeQuantisedMatrixDesc(arrived[1])&&
                        fragment.down()->exportNativeQuantisedMatrixDesc(arrived[2])&&
                        declaration.matchesNativeGateUp(arrived[0],arrived[1])&&declaration.matchesNativeDown(arrived[2]),
                        "actual GPU-source numerical arrival retains its original geometry, provenance and physical decoder");
                }
            }
        }
        // The existing rotation retains expert3 on the warm GPU while
        // expert4 arrives from its declared capacity source. Discover that unchanged resident from the real banks,
        // rather than assigning an execution codebook or editing a route table.
        for(size_t expert=0;expert<state->bank->layers.front().resident_mask.size();++expert)
            if(initial->layers.front().resident_mask.at(expert)&&
               state->bank->layers.front().resident_mask.at(expert)&&
               std::find(state->promoted_experts.begin(),state->promoted_experts.end(),int(expert))==state->promoted_experts.end())
                state->canonical_experts.push_back(int(expert));
        if(!state->promoted_experts.empty())require(!state->canonical_experts.empty(),
            "original rotation retains a canonical GPU expert beside its physical arrival");
        local_promoted+=state->promoted_experts.size();
        auto boundary=std::make_unique<ColdPublicationBoundary>();
        bindings.push_back({.device=device,.runtime_layers_device=state->runtime->deviceLayerState(0),
            .runtime_table_host=state->runtime.get(),.overlay_participant_id=member->participant_id,
            .domain_participant_id=uint32_t(member->domain_participant_index),
            .domain_participant_count=uint32_t(state->bank->layers.front().fragment_family->cohort()->participants().size()),
            .layer_count=1,.expert_count=geometry.experts,.top_k=geometry.top_k,
            .epoch_control=state->arena->control(),.maintenance_epoch=state->arena->maintenanceEpoch(),
            .maintenance_status=state->arena->maintenanceStatus(),.inference_boundary=boundary.get(),
            .initial_runtime_publisher=boundary.get()});
        boundaries.push_back(std::move(boundary));published.emplace(device,std::move(state));
    }
    for(const auto& [device,state]:published) {
        (void)device;
        require(state->bank->epoch==transaction.candidate->epoch,
            "published host bank retains the actual migration candidate epoch");
    }
    std::shared_ptr<MoEOverlayHostAuthorityDeviceBankPublisher> publisher;
    std::unique_ptr<IMoEOverlayInactiveBankTransaction> publication;
    if(!bindings.empty()) {
        publisher=std::make_shared<MoEOverlayHostAuthorityDeviceBankPublisher>(MoEOverlayHostAuthorityDeviceBankPublisher::Config{
            .runtime_bindings=bindings,.registry=original.registry,.perf_device="original-promoted23-native-layout"});
        publication=publisher->createTransaction(transaction.previous,transaction.candidate,&error);
        require(bool(publication)&&publication->beginPrepare(&error),"prepare actual device-bank publication: "+error);
        await([&](std::string* e){return publication->pollPrepare(e);},"candidate prepare event");
#if defined(HAVE_ROCM)
        try {
        std::vector<std::future<void>> preparing;
        std::vector<std::pair<DeviceId,std::shared_ptr<FullEpochNumericalPair>>> prepared_pairs(published.size());
        unsigned prepared_count=0;
        if(program==NumericalProgram::FullHCFFNEpoch)for(const auto& [device,state]:published)
            if(!state->promoted_experts.empty()&&(!real_tp4||domain_of.at(device)=="warm")) {
            const auto index=prepared_count++;
            preparing.push_back(std::async(std::launch::async,[&,device,state,index] {
            const auto& members=original.registry->initialOwnerMap().participants();
            const auto member=std::find_if(members.begin(),members.end(),[&](const auto& p){return p.device==device&&p.world_rank==original.mpi->rank();});
            require(member!=members.end(),"full epoch original source member");
            auto initial=std::make_shared<PublishedFFNEpoch>();initial->runtime=state->runtime;initial->arena=state->arena;
            initial->bank=original.registry->endpoint(member->participant_id)->acquire(transaction.previous->epoch);
            require(initial->bank&&initial->epoch()==1,"full epoch retains actual initially published source lease");
            std::shared_ptr<ILocalTPContext> tp;
            TensorParallelConfig partition;
            if(native_cohorts) {
                const auto& domain=local_domains.at(domain_of.at(device));
                tp=domain.communicator;partition=domain.shared_partition;
            } else tp=createLocalTPContext({GlobalDeviceAddress::fromLocalDeviceId(device)},{},CollectiveBackendType::AUTO);
            auto pair=std::make_shared<FullEpochNumericalPair>(device,std::move(tp),initial,shared.at(device),partition,
                *original.context->concreteWeightManager()->preparedWeightStore(),geometry,
                epoch_source==EpochSource::GPUNativeCohort2WithDescriptorReader?EpochNumericalBody::RetainedWithDescriptorReader:
                    (gpu_native?EpochNumericalBody::RetainedBank:EpochNumericalBody::OriginalNative));
            prepared_pairs[index]={device,std::move(pair)};
            }));
        }
        joinOriginalNativeWorkerSubmissions(preparing,abort_native,std::chrono::steady_clock::now()+std::chrono::seconds(30));
        if(native_failure)std::rethrow_exception(native_failure);
        if(native_cohorts) {
            for(const auto& [name,domain]:local_domains) {
                if(real_tp4&&name!="warm")continue;
                std::vector<std::shared_ptr<FullEpochNumericalPair>> members;
                for(const auto& device:domain.devices) {
                    const auto found=std::find_if(prepared_pairs.begin(),prepared_pairs.begin()+prepared_count,
                        [&](const auto& item){return item.first==device;});
                    require(found!=prepared_pairs.begin()+prepared_count,"actual original member has cold capture resources");
                    members.push_back(found->second);
                }
                FullEpochNumericalPair::captureOriginalCohort(members);
            }
        } else for(unsigned index=0;index<prepared_count;++index) {
            const std::array singleton{prepared_pairs[index].second};
            FullEpochNumericalPair::captureOriginalCohort(singleton);
        }
        std::vector<std::future<void>> initial_replays;
        for(unsigned index=0;index<prepared_count;++index)
            initial_replays.push_back(std::async(std::launch::async,[pair=prepared_pairs[index].second]{pair->replayInitial();}));
        joinOriginalNativeWorkerSubmissions(initial_replays,abort_native,std::chrono::steady_clock::now()+std::chrono::seconds(30));
        if(native_failure)std::rethrow_exception(native_failure);
        for(unsigned index=0;index<prepared_count;++index) {
            auto [device,pair]=std::move(prepared_pairs[index]);
            const std::weak_ptr<FullEpochNumericalPair> weak=pair;
            const auto binding=std::find_if(bindings.begin(),bindings.end(),[&](const auto& item){return item.device==device;});
            require(binding!=bindings.end(),"actual full publisher boundary binding");
            auto* boundary=dynamic_cast<ColdPublicationBoundary*>(binding->inference_boundary);
            require(boundary,"original publication boundary owner");
            boundary->bindInferenceTerminal([weak](void* stream){const auto retained=weak.lock();
                return retained&&retained->enqueueMaintenanceBoundary(stream);});
            numerical.emplace(device,std::move(pair));
        }
        } catch(const std::exception& failure) {
            std::osyncstream(std::cerr)<<"Original epoch1 numerical binding failed: "<<failure.what()<<'\n';
            publication->abort();
            await([&](std::string* e){return publication->pollAbort(e);},"abort original staged publication after diagnostic failure");
            throw;
        }
#endif
        require(publication->beginPublication(&error),"publish actual epoch2 selector: "+error);
        await([&](std::string* e){return publication->pollPublication(e);},"candidate publication event");
    }
    unsigned long long local_count=local_promoted,global_count=0;
    if(q8_down)
        std::osyncstream(std::cout)<<"QFN_Q8_DOWN_MOVEMENT_SOURCE,gate="<<(q5_q8?"Q5_K":"Q4_K")
            <<",up="<<(q5_q8?"Q5_K":"Q4_K")<<",down=Q8_0,source_F="<<geometry.routed
            <<",source_S="<<geometry.shared<<",cohort_degree="<<degree<<'\n';
    const unsigned long long expected_arrivals=real_tp4?6u:gpu_native?requests.size()*degree:1u;
    require(MPI_Allreduce(&local_count,&global_count,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,MPI_COMM_WORLD)==MPI_SUCCESS&&
            global_count==expected_arrivals,
        "every declared physical arrival has its original numerical destination witness");
    if(MPI_Barrier(MPI_COMM_WORLD)!=MPI_SUCCESS)throw std::runtime_error("promoted physical source publication rendezvous");
    std::vector<std::future<void>> replaying;
    for(const auto& [device,state]:published)if(!state->promoted_experts.empty()) {
        if(real_tp4&&domain_of.at(device)!="warm")continue;
        if(native_cohorts) {
            require(numerical.contains(device),"same retained component graph with its original RCCL cohort");
            replaying.push_back(std::async(std::launch::async,[&,device,state]{numerical.at(device)->replay(*state,true);}));
            continue;
        }
        const auto& members=original.registry->initialOwnerMap().participants();
        const auto found=std::find_if(members.begin(),members.end(),[&](const auto& p){return p.device==device&&p.world_rank==original.mpi->rank();});
        require(found!=members.end(),"replay target retains its original published member");
        auto tp=createLocalTPContext({GlobalDeviceAddress::fromLocalDeviceId(device)},{},CollectiveBackendType::AUTO);
        require(bool(tp),"original singleton native collective");
        std::barrier rendezvous(1);
        TensorParallelConfig partition;
#if defined(HAVE_ROCM)
        if(program==NumericalProgram::FullHCFFNEpoch) {
            require(numerical.contains(device),"same full numerical graph pair retained through actual publisher");
            numerical.at(device)->replay(*state,true);
        } else
#endif
        replayResidentParticipant(device,0,*tp,rendezvous,*original.registry,found->participant_id,
            shared.at(device),partition,*original.context->concreteWeightManager()->preparedWeightStore(),
            ResidentFFNObservation::Replay,"Q4_K_Q5_1_CPU_Promoted23",LocalFFNReplayCandidate::Resident,
            LocalFFNProgram::Original,geometry,true,state.get());
    }
    joinOriginalNativeWorkerSubmissions(replaying,abort_native,std::chrono::steady_clock::now()+std::chrono::seconds(30));
    if(native_failure)std::rethrow_exception(native_failure);
    if(publication) {
        await([&](std::string* e){return publication->pollRetirementFence(e);},"original epoch reader retirement");
        publication->retirePrevious();publication.reset();publisher.reset();
    }
#if defined(HAVE_ROCM)
    numerical.clear();
#endif
    published.clear();boundaries.clear();shared.clear();
    wave.abortAndDrain();
    if(MPI_Barrier(MPI_COMM_WORLD)!=MPI_SUCCESS)throw std::runtime_error("promoted physical replay retirement rendezvous");
}
} // namespace promoted23_replay

#if defined(HAVE_ROCM) && defined(LLAMINAR_CASTAGNA_FAST_EPOCH)
/** @brief Own six original PCI addresses across the live movement replay. */
struct RetainedClosedMovementPCISelection {
    std::array<std::string, 6> addresses;

    /** @return A borrowed view for the retained replay whose selection stays in scope. */
    std::array<std::string_view, 6> views() const & {
        std::array<std::string_view, 6> result;
        std::copy(addresses.begin(),addresses.end(),result.begin());
        return result;
    }
    std::array<std::string_view, 6> views() const && = delete;
};

/**
 * @brief Use six original gfx906 devices in three wholly rank-local pairs.
 * @param env Optional explicit ordered cohort reservation used by the site runner.
 * @param inventory Original rank inventory retained by the enclosing replay.
 * @return Owned original PCI addresses in the declared movement order.
 */
RetainedClosedMovementPCISelection originalRetainedClosedMovementPCIs(
    const char* env, const ClusterInventory& inventory) {
    RetainedClosedMovementPCISelection result;
    std::size_t selected=0;
    if(env&&*env) {
        auto remaining=std::string_view(env);
        while(selected<result.addresses.size()) {
            const auto separator=remaining.find(',');
            const auto address=remaining.substr(0,separator);
            require(!address.empty() &&
                    (separator==std::string_view::npos)==(selected+1==result.addresses.size()),
                "retained movement PCI override requires six comma-separated original GPU addresses");
            result.addresses[selected++]=address;
            if(separator!=std::string_view::npos)remaining.remove_prefix(separator+1);
        }
        return result;
    }
    for(const auto& rank:inventory.ranks) {
        std::vector<std::string_view> local;
        for(const auto& gpu:rank.gpus) {
            if(gpu.type==DeviceType::ROCm&&rankOwnsGPUDevice(rank,gpu)&&!gpu.pci_bus_address.empty()&&
               std::string_view(gpu.native_isa).substr(0,gpu.native_isa.find(':'))=="gfx906")
                local.push_back(gpu.pci_bus_address);
        }
        const auto count=std::min(local.size() / 2 * 2,result.addresses.size()-selected);
        std::copy_n(local.begin(),count,result.addresses.begin()+selected);
        selected+=count;
        if(selected==result.addresses.size())return result;
    }
    throw std::runtime_error("Retained closed movement requires three rank-local gfx906 pairs from original inventory");
}
/** @brief Keep every original participant aligned while testing the declared native arrival format. */
void replayOriginalRetainedClosedCohort2(promoted23_replay::EpochSource source) {
    try {
        // MPIContextFactory keeps a weak reference. Retain the original context
        // through fabric construction and replay so selection and fabric use
        // the very same original discovery publication.
        const auto mpi=MPIContextFactory::global();
        require(mpi!=nullptr,"retained movement uses its original MPI context");
        const auto inventory=mpi->clusterInventory();
        require(inventory!=nullptr,"retained movement uses actual MPI rank inventory");
        const auto pci=originalRetainedClosedMovementPCIs(
            std::getenv("LLAMINAR_TEST_RETAINED_HC_FFN_MOVEMENT_PCIS"),*inventory);
        promoted23_replay::run(promoted23_replay::NumericalProgram::FullHCFFNEpoch,pci.views(),
            source);
        if(::testing::Test::HasFailure())MPI_Abort(MPI_COMM_WORLD,2);
    } catch(const std::exception& error){ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;}
}

/** @test A real closed movement cohort retains two actual native RCCL members. */
TEST(FullNativeHCFFN,ROCmRetainedFastGPUOriginalCohort2EpochSwitchReplay) {
    replayOriginalRetainedClosedCohort2(promoted23_replay::EpochSource::GPUNativeCohort2);
}

/** @test Both native Q8-down leaves consume changed expert bytes through the same retained capture. */
TEST(FullNativeHCFFN,ROCmRetainedFastGPUQ4KQ8EpochSwitchReplay) {
    replayOriginalRetainedClosedCohort2(promoted23_replay::EpochSource::GPUNativeCohort2Q4Q8);
}
TEST(FullNativeHCFFN,ROCmRetainedFastGPUQ5KQ8EpochSwitchReplay) {
    replayOriginalRetainedClosedCohort2(promoted23_replay::EpochSource::GPUNativeCohort2Q5Q8);
}
#endif
