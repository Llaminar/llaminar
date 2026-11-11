/**
 * @file NativeSharedUnionPacketReplay.inc
 * @brief Source-only union publication on three authentic original-rank GPUs.
 *
 * Two real consumer grants share one physical dispatch matrix. Ten distinct
 * admitted source families retain the original forward/reverse target order
 * and65/1/0/17/65 recorded live-row cases. Each family is intentionally cancelled
 * after its actual source terminal because this component does not fabricate a
 * follower return. One original PMA owns every bank, grant and native definition.
 */
namespace {
/** Source-only publication proof; the complete roundtrip is a separate native fixture. */
class OriginalSharedUnionSourceReplay final {
    static constexpr int rows=65,width=8,topk=2;
    static constexpr std::array<int,5> live_rows{65,1,0,17,65};
    using Graph=test::RetainedParticipantGraphExecutableFixtureAccess;
    struct Family {
        std::vector<MoEOverlayMappedActivationDeviceLane> lanes;
        std::shared_ptr<MoEOverlayActivationLaneBatchState> batch;
        std::shared_ptr<IComputeStage> clear,dispatch;
        std::shared_ptr<RetainedParticipantGraphExecutable> graph;
        std::vector<std::unique_ptr<MoEOverlayActivationEpochProtocol>> protocols;
        std::vector<std::optional<MoEOverlayActivationEpochIdentity>> identities;
        bool pending=false,submitted=false;
        std::uint64_t signature=0;
    };
public:
    /** Retain every observed original endpoint before any native allocation. */
    explicit OriginalSharedUnionSourceReplay(std::span<const DeviceId> devices):devices_(devices.begin(),devices.end()) {
        require(devices_.size()==3,"Shared union requires one source and two real consumers");
        original_=test::PreparedNativeLookupFixture::prepare(devices_,MPIContextFactory::global()->rank());
    }
    /** Absorbing failure cancellation precedes native/source retirement. */
    ~OriginalSharedUnionSourceReplay() noexcept {try{dispose();}catch(...){std::terminate();}}
    /** Declare exact original banks, graph roster and canonical Batch resource contracts. */
    void prepare();
    /** Preserve reversed order, all live counts, exact bytes and the one-writer native DAG. */
    void replay();
private:
    /** Poll only the original source terminal stream. */
    void await(){
        if(!worker_)return;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;){GPUStreamExecutionState state=GPUStreamExecutionState::Pending;
            worker_->submitAndWait([&]{state=worker_->queryStreamExecutionState(stream_,"shared union source terminal");});
            if(state==GPUStreamExecutionState::Complete)return;
            require(std::chrono::steady_clock::now()<deadline,"Original shared union source terminal timed out");
            std::this_thread::yield();}
    }
    /** Retire all actual definitions off-worker before mappings, grants and original context. */
    void dispose();
    std::vector<DeviceId> devices_;
    std::unique_ptr<test::PreparedNativeLookupFixture> original_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::shared_ptr<IWorkerGPUContext> worker_;
    std::unique_ptr<IDeviceContext> context_;
    void* stream_=nullptr;
    NativeExactPacketTensor hidden_{65,8},indices_{65,2},weights_{65,2},canonical_{130,8},zero_{130,8};
    NativeExactPacketTensor placement_{1,sizeof(NativeExactPacketPlacement)/4,BufferTensorType::INT32};
    MoEOverlayActivationChannelPlan plan_;
    MoEOverlayInferenceTopologyIdentity topology_;
    std::shared_ptr<IMoEOverlayRankBatchTransport> transport_;
    std::shared_ptr<IMoEOverlayMappedActivationTransport> mapped_;
    std::shared_ptr<MappedHostTransferRegion> snapshot_;
    std::optional<PhysicalMemoryAllocationLease> snapshot_claim_;
    std::array<Family,10> families_;
    std::vector<int> targets_;
    bool admitted_=false,disposed_=false;
};

void OriginalSharedUnionSourceReplay::prepare(){
    const auto mpi=MPIContextFactory::global();const int rank=mpi->rank();
    MoERoutedExpertPlacementPlan placement;placement.enabled=true;placement.topology=RoutedExpertPlacementTopology::TieredOverlay;
    placement.continuation_domain=placement.base_model_domain=placement.shared_expert_domain="source";
    placement.continuation_domain_spec.domain="source";placement.continuation_domain_spec.logical_root_participant=0;
    for(int domain_index=0;domain_index<2;++domain_index){
        RoutedExpertDomain domain;domain.name=domain_index?"consumers":"source";domain.scope=ExecutionDomainScope::RANK_LOCAL;
        domain.owner_rank=rank;domain.world_ranks={rank};domain.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
        if(domain_index)domain.participants={GlobalDeviceAddress::fromLocalDeviceId(devices_[1]),GlobalDeviceAddress::fromLocalDeviceId(devices_[2])};
        else domain.participants={GlobalDeviceAddress::fromLocalDeviceId(devices_[0])};
        placement.domains.push_back(domain);placement.routed_tiers.push_back({.name=domain.name,.domain=domain.name,.priority=domain_index?30:80});
    }
    placement.authority_execution=resolveMoEOverlayAuthorityExecutionKind(placement);
    plan_=MoEOverlayActivationChannelPlanner::plan({.placement_plan=&placement,.cluster_inventory=mpi->clusterInventory().get(),
        .row_capacity=rows,.d_model=width,.top_k=topk,.graph_family_count=families_.size()});
    require(plan_.channels.size()==1,"Original shared consumers must share one physical rank-pair channel");
    const auto& channel=plan_.channels.front();targets_=channel.targetParticipantIds();
    require(targets_.size()==2 && channel.source_device==devices_[0] && channel.source_world_rank==rank && channel.target_world_rank==rank,
        "Shared union changed original source or actual consumer population");
    const auto host=original_->resource(DeviceId::cpu());PhysicalMemoryPlanBuilder bill;
    for(auto* tensor:{&hidden_,&indices_,&weights_,&canonical_,&zero_,&placement_})tensor->declare(bill,host,original_->resource(devices_[0]));
    for(const auto device:devices_)bill.add(original_->resource(device),PhysicalMemoryOwner::ActivationTransportStaging,plan_.stagingBytesFor(rank,device));
    bill.add(host,PhysicalMemoryOwner::ActivationTransportStaging,plan_.stagingBytesFor(rank,DeviceId::cpu())+
        TransferEngine::mappedHostRegionAllocationBytes(2*sizeof(MoEOverlayActivationDeviceEpochGrant)));
    const auto add=[&](const PhysicalMemoryPlan& value){for(const auto& line:value.resources())bill.add(line);};
    add(RetainedParticipantGraphExecutable::generalBOM(host,original_->resource(devices_[0]),families_.size()));
    add(MoEOverlayActivationBatchResources::sharedLaneStreamsBOM(channel,host,original_->resource(devices_[0])));
    for(std::size_t f=0;f<families_.size();++f)add(MoEOverlayActivationBatchResources::bom(channel,rows,0,host,original_->resource(devices_[0])));
    for(std::size_t p=1;p<devices_.size();++p)add(MoEOverlayActivationBatchResources::coldPublicationBOM(host,original_->resource(devices_[p])));
    original_->admit(bill.build());admitted_=true;memory_=original_->memory();
    worker_=original_->permit(devices_[0]).worker();stream_=worker_->defaultStream();context_=IDeviceContext::create(devices_[0]);
    worker_->submitAndWait([&]{for(auto* tensor:{&hidden_,&indices_,&weights_,&canonical_,&zero_,&placement_})tensor->prepare(memory_,devices_[0],stream_);});
    snapshot_claim_.emplace(memory_->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationTransportStaging,
        TransferEngine::mappedHostRegionAllocationBytes(2*sizeof(MoEOverlayActivationDeviceEpochGrant))));
    const std::array source_device{devices_[0]};snapshot_=TransferEngine::instance().allocateMappedHostRegion(2*sizeof(MoEOverlayActivationDeviceEpochGrant),source_device);
    topology_={.workspace_generation=original_->permit(devices_[0]).generation(),.topology_fingerprint_low=0x756e696f6e,.topology_fingerprint_high=0x7265706c6179,
        .source_world_rank=rank,.target_world_rank=rank};
    MoEOverlayRankBatchTransportConfig config;config.mpi_ctx=mpi;config.endpoint_scope=MoEOverlayActivationEndpointScope::RankLocalGPU;
    config.rank_local_source_device=devices_[0];config.source_world_rank=config.target_world_rank=rank;
    config.workspace=std::make_shared<MoEOverlayRankBatchWireWorkspace>(MoEOverlayRankBatchWireWorkspace::Config{
        .participant_ids=targets_,.max_total_rows=rows*2,.max_total_entries=rows*topk,.d_model=width,.top_k=topk});
    config.max_rows_per_participant=rows;config.max_entries_per_participant=rows*topk;config.d_model=width;config.top_k=topk;
    config.tier_index=channel.tier_index;config.domain_ordinal=channel.domain_ordinal;config.channel_identity="original_shared_union_"+std::to_string(::getpid());
    config.transaction_topology=topology_;config.transaction_slot_count=32;
    config.source_endpoint={.world_rank=rank,.participant_id=0,.tier_priority=80,.domain_ordinal=0};config.target_tier_priority=30;
    config.activation_layout=channel.mapping_layout;config.local_lanes=channel.localLanes(rank);config.physical_memory_authority=memory_;
    for(std::size_t f=0;f<families_.size();++f)config.activation_graph_families.push_back({.graph_role_mask=1u<<unsigned(MoEOverlayInferenceGraphRole::MainPrefill),.model_layer_indices={5,9,14}});
    transport_=createMoEOverlayRankLocalGPUActivationTransport(std::move(config));mapped_=std::dynamic_pointer_cast<IMoEOverlayMappedActivationTransport>(transport_);
    require(bool(mapped_),"Shared union lost actual mapped transport");
    worker_->submitAndWait([&]{
        for(int row=0;row<rows;++row)for(int column=0;column<width;++column)hidden_.host<float>()[row*width+column]=float(row*100+column)+.25f;
        constexpr std::array<float,8> weights{.75f,.25f,.6f,.4f,.55f,.45f,.9f,.1f};
        for(int slot=0;slot<rows*topk;++slot){indices_.host<float>()[slot]=float(10+slot%8);weights_.host<float>()[slot]=weights[slot%8];}
        auto* metadata=placement_.host<NativeExactPacketPlacement>();*metadata={};
        constexpr std::array<int,8> roles{1,2,0,0,1,1,2,0};
        for(int slot=0;slot<8;++slot)metadata->routes[1][10+slot]=roles[slot]?targets_[roles[slot]-1]:0;
        metadata->epochs[0]=metadata->epochs[1]=41;metadata->ticket={.epoch=41,.selector=deviceMoEOverlayEpochSelector(3,1)};
        metadata->status={.epoch=41,.selector=metadata->ticket.selector,.operation=unsigned(DeviceMoEOverlayEpochOperation::Acquire),
            .code=unsigned(DeviceMoEOverlayEpochStatusCode::Success),.bank=1,.observed_state=unsigned(DeviceMoEOverlayEpochBankState::Published)};
        for(auto* tensor:{&hidden_,&indices_,&weights_,&canonical_,&zero_,&placement_}){TransferEngine::publishHostWrite(tensor->tensor.get());TransferEngine::prepareDeviceInput(tensor->tensor.get(),devices_[0],stream_);}
    });await();
    for(std::size_t index=0;index<families_.size();++index){auto& family=families_[index];
        auto ordered=targets_;if(index>=live_rows.size())std::reverse(ordered.begin(),ordered.end());
        for(const int target:ordered){family.lanes.push_back(mapped_->activationDeviceLane(target,index,devices_[0]));
            auto config=mapped_->activationEpochConfig(target,index);family.protocols.push_back(std::make_unique<MoEOverlayActivationEpochProtocol>(mapped_->activationEpochControl(target,index),config));family.identities.emplace_back();}
        family.batch=std::make_shared<MoEOverlayActivationLaneBatchState>(devices_[0],family.lanes,rows,0,5,memory_);
        ResidualAddStage::Params clear;clear.device_id=devices_[0];clear.input=zero_.tensor.get();clear.output=canonical_.tensor.get();clear.num_elements=rows*topk*width;
        family.clear=std::make_shared<ResidualAddStage>(clear);
        auto* metadata=static_cast<NativeExactPacketPlacement*>(placement_.tensor->gpu_data_ptr());
        MoEOverlayActivationDispatchPackBatchStage::Params dispatch;dispatch.device_id=devices_[0];dispatch.transaction=family.batch;
        dispatch.hidden=hidden_.tensor.get();dispatch.routing_indices=indices_.tensor.get();dispatch.routing_weights=weights_.tensor.get();
        dispatch.placement={.banks={{metadata->routes[0],&metadata->epochs[0]},{metadata->routes[1],&metadata->epochs[1]}},.ticket=&metadata->ticket,.status=&metadata->status,.expert_count=18};
        dispatch.active_row_count_device=&metadata->active_rows;family.dispatch=std::make_shared<MoEOverlayActivationDispatchPackBatchStage>(dispatch);
        family.graph=Graph::original(original_->permit(devices_[0]),stream_);
        std::vector<GraphCaptureDependencyLedger::StagePlan> plans{
            {.stage_identity=family.clear.get(),.stage_name=family.clear->name(),.external_inputs={zero_.tensor.get()},.outputs={canonical_.tensor.get()},.recording_stream=stream_},
            {.stage_identity=family.dispatch.get(),.stage_name=family.dispatch->name(),.external_inputs={hidden_.tensor.get(),indices_.tensor.get(),weights_.tensor.get()},.recording_stream=stream_}};
        GraphCaptureDependencyLedger ledger(devices_[0],stream_,plans,"original shared union source-only packet");
        worker_->submitAndWait([&]{
            for(const auto& stage:{family.clear,family.dispatch})require(stage->prepareGraphLaunch(context_.get(),stream_),"Shared union source preparation failed");
            for(auto* tensor:{&hidden_,&indices_,&weights_,&canonical_,&zero_,&placement_})TransferEngine::requireDeviceInput(tensor->tensor.get(),devices_[0],stream_);
            auto& graph=Graph::capture(family.graph);ScopedBackendGraphCapture recording(*worker_,graph,"shared union source-only original capture",&ledger);
            require(recording.begin(),"Shared union source capture failed");
            for(const auto& stage:{family.clear,family.dispatch}){ScopedGraphCaptureStage scope(stage.get(),stream_);require(stage->execute(context_.get()),"Shared union source recording failed");scope.complete();}
            recording.finish();require(graph.prepareRuntimeContextStorage(memory_) && graph.instantiate(),"Shared union source instantiate failed");
            std::vector<GPUGraphKernelNodeInfo> kernels;std::string error;require(graph.inspectKernelNodes(kernels,&error),error.c_str());
            EXPECT_EQ(std::count_if(kernels.begin(),kernels.end(),[](const auto& node){return node.name.find("packSharedDispatchUnionKernel")!=std::string::npos;}),1);
            if(devices_[0].is_rocm())for(const auto& node:kernels)EXPECT_EQ(node.nesting_depth,0u);
            family.signature=family.dispatch->graphCaptureVariantSignature();
        });
    }
}

void OriginalSharedUnionSourceReplay::replay(){
    for(std::size_t index=0;index<families_.size();++index){auto& family=families_[index];const int live=live_rows[index%live_rows.size()];
        SCOPED_TRACE(::testing::Message()<<"original_rank="<<MPIContextFactory::global()->rank()<<" live="<<live<<" reverse="<<(index>=live_rows.size()));
        worker_->submitAndWait([&]{auto* metadata=placement_.host<NativeExactPacketPlacement>();metadata->active_rows=live;
            TransferEngine::publishHostWrite(placement_.tensor.get());TransferEngine::prepareDeviceInput(placement_.tensor.get(),devices_[0],stream_);});
        auto* matrix=static_cast<float*>(family.lanes.front().mapped_region->mutableHostData(family.lanes.front().shared_dispatch_hidden_offset));
        std::fill_n(matrix,rows*width,-137.f);
        const MoEOverlayInferenceCommandIdentity command{.request_generation=1,.command_id=1,.initial_placement_epoch=40};
        const auto ticket=makeMoEOverlayInferenceExecutionTicket(topology_,command,1,1,40,MoEOverlayInferenceGraphRole::MainPrefill,1,rows,rows);
        for(std::size_t lane=0;lane<family.lanes.size();++lane){std::string error;family.identities[lane]=family.protocols[lane]->arm(ticket,1,10'000'000'000ULL,&error);require(bool(family.identities[lane]),error.c_str());}
        worker_->submitAndWait([&]{family.pending=true;family.submitted=true;require(Graph::submit(family.graph,1),"Shared union source submit failed");TransferEngine::publishDeviceWrite(canonical_.tensor.get(),devices_[0],stream_);});
        await();require(Graph::observeTerminal(family.graph),"Shared union source lacks actual terminal proof");family.pending=false;
        worker_->submitAndWait([&]{for(std::size_t lane=0;lane<family.lanes.size();++lane)require(getBackendFor(devices_[0])->deviceToHostOnStream(
            static_cast<std::byte*>(snapshot_->mutableHostData())+lane*sizeof(MoEOverlayActivationDeviceEpochGrant),family.lanes[lane].grant_device,
            sizeof(MoEOverlayActivationDeviceEpochGrant),devices_[0].ordinal,stream_),"Shared union terminal grant observation failed");});await();
        const auto* grants=static_cast<const MoEOverlayActivationDeviceEpochGrant*>(snapshot_->mutableHostData());std::uint64_t actual_bytes=0,union_rows=0;
        constexpr std::array<int,8> role{1,2,0,0,1,1,2,0};
        for(int row=0;row<live;++row)union_rows+=role[(row*topk)%8] || role[(row*topk+1)%8];
        for(std::size_t lane=0;lane<family.lanes.size();++lane){const int target=family.lanes[lane].target_participant_id;
            const auto which=std::find(targets_.begin(),targets_.end(),target)-targets_.begin()+1;
            std::uint64_t count_rows=0,entries=0;
            for(int row=0;row<live;++row){bool seen=false;for(int slot=0;slot<topk;++slot)if(role[(row*topk+slot)%8]==which){++entries;seen=true;}count_rows+=seen;}
            const auto& descriptor=family.lanes[lane].control_host->buffers[0].dispatch_descriptor;const auto& grant=grants[lane];actual_bytes+=grant.published_shared_hidden_bytes;
            EXPECT_EQ(grant.state,unsigned(MoEOverlayActivationEndpointState::Active));EXPECT_EQ(grant.code,unsigned(MoEOverlayActivationStatusCode::Success));
            EXPECT_EQ(grant.generation,1u);EXPECT_EQ(grant.last_published_stage,0);EXPECT_EQ(grant.published_stage_count,1u);
            EXPECT_EQ(descriptor.digest,grant.digest);EXPECT_EQ(descriptor.stage_ordinal,0u);EXPECT_EQ(descriptor.model_layer_index,5);
            EXPECT_EQ(descriptor.placement_epoch,41u);EXPECT_EQ(descriptor.live_rows,count_rows);EXPECT_EQ(descriptor.live_entries,entries);
            EXPECT_EQ(descriptor.payload_bytes,moeOverlayDispatchPayloadBytes(count_rows,entries,width));
        }
        for(int row=0;row<rows;++row)for(int column=0;column<width;++column){const bool useful=row<live && (role[(row*topk)%8] || role[(row*topk+1)%8]);
            EXPECT_EQ(matrix[row*width+column],useful?float(row*100+column)+.25f:-137.f);}
        EXPECT_EQ(actual_bytes,union_rows*width*sizeof(float));EXPECT_EQ(family.signature,family.dispatch->graphCaptureVariantSignature());
        for(std::size_t lane=0;lane<family.lanes.size();++lane){std::string error;require(family.protocols[lane]->cancelOriginalInvocation(*family.identities[lane],MoEOverlayActivationStatusCode::ExplicitAbort,&error),error.c_str());}
    }
}

void OriginalSharedUnionSourceReplay::dispose(){
    if(disposed_)return;
    if(mapped_)for(std::size_t f=0;f<families_.size();++f)if(families_[f].pending)for(const int target:targets_){auto& control=mapped_->activationEpochControl(target,f);
        for(auto& bank:control.buffers){std::atomic_ref<std::uint64_t>(bank.dispatch_signal.value).store(kMoEOverlayActivationAbortTimeline,std::memory_order_release);std::atomic_ref<std::uint64_t>(bank.return_signal.value).store(kMoEOverlayActivationAbortTimeline,std::memory_order_release);}
        std::atomic_ref<std::uint64_t>(control.admission.ready_signal).store(kMoEOverlayActivationAbortTimeline,std::memory_order_release);}
    await();for(auto& family:families_)if(family.graph && family.submitted)require(Graph::observeTerminal(family.graph),"Shared union retirement lost original terminal");
    for(auto& family:families_){family.graph.reset();family.clear.reset();family.dispatch.reset();family.batch.reset();family.lanes.clear();family.identities.clear();family.protocols.clear();}
    mapped_.reset();transport_.reset();snapshot_.reset();snapshot_claim_.reset();
    for(auto* tensor:{&hidden_,&indices_,&weights_,&canonical_,&zero_,&placement_})tensor->tensor.reset();context_.reset();worker_.reset();
    if(admitted_)original_->requirePayloadRetired();original_->retire();original_->requireExactRetired();disposed_=true;
}
/** Select every real endpoint through the original same-rank inventory authority. */
void nativeSharedUnion(DeviceType family){originalRankLocalReplay(family,3,[&](const auto& devices){OriginalSharedUnionSourceReplay replay(devices);replay.prepare();replay.replay();});}
}
#ifdef HAVE_CUDA
TEST(Test__MappedActivationPacketCUDAAndROCm,CUDASharedPhysicalUnionHasOneWriterAndOnlyLivePayload){nativeSharedUnion(DeviceType::CUDA);}
#endif
#ifdef HAVE_ROCM
TEST(Test__MappedActivationPacketCUDAAndROCm,ROCmSharedPhysicalUnionHasOneWriterAndOnlyLivePayload){nativeSharedUnion(DeviceType::ROCm);}
#endif
