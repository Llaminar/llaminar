/**
 * @file FullNativeHCFFNReplay.inc
 * @brief Actual production full HC/FFN lowering versus its original staged declarations.
 *
 * Included by the existing FFN source/arena/worker test owner under HAVE_ROCM.
 * Both arms retain original prepared weights, device-owned row counts and exact
 * stream/capture authorities. No external engine, diagnostic kernel or host
 * arithmetic substitute participates in the mandatory numerical regression.
 */
/** @brief Inputs and public prefix values remain caller-owned through all captures. */
struct FullProductionHCPrefix {
    std::shared_ptr<FP32Tensor> residual,previous,raw,gain,post,normalized,down,activated,logits;
    std::shared_ptr<TensorBase> down_source,up_source,inject_source;
    std::shared_ptr<FP32Tensor> injection,next_trunk,reader_mixed; ///< Original middle-read exported values, when selected.
    GpuPreparedGemm down_prepared,up_prepared,inject_prepared;
};
/**
 * @brief Terminal observations of the actual device runtime publication, never an execution mirror.
 * @details Host vectors are reserved before the terminal event guard. Only the
 * selected descriptor rows are downloaded; pointer-bearing banks are not copied
 * wholesale or uploaded. Each descriptor is compared with its own acquired bank,
 * so independent prepared-owner addresses are never compared across arms.
 */
struct FullProductionRuntimeObservation {
    std::array<std::uint32_t,4> header{}; ///< Active bank/epoch and exact expert/top-k dimensions.
    std::array<std::uint32_t,2> bank_header{}; ///< Acquired bank epoch and expert extent.
    std::vector<std::int32_t> ids,participants;
    std::vector<float> weights;
    std::vector<std::uint64_t> selected,local,expected_selected,expected_local;
    std::vector<DeviceMoEExpertDescriptor> acquired;
    std::array<std::vector<DeviceNativeQuantisedMatrixDesc>,3> published;
};
/** @brief Admit a real persistent FP32 owner and its original input publication. */
std::shared_ptr<FP32Tensor> fullProductionHCFloat(std::size_t count,DeviceId device,void* stream) {
    auto tensor=std::make_shared<FP32Tensor>(std::vector<std::size_t>{1,count},DeviceId::cpu());
    std::fill_n(tensor->mutable_data(),count,.125f);
    // Return only after the original transfer prefix retires: allocation or
    // upload failure must not release a still-borrowed local source owner.
    ScopedGPUCompletionEvent completion(getBackendFor(device),device,stream);
    TransferEngine::prepareDeviceInput(tensor.get(),device,stream);
    require(completion.join(),"cold exact source upload");return tensor;
}
/** @brief Append actual previous residual-write and original read; no next-layer injection is timed.
 * @param hidden Original declared source H; the default preserves existing full-family geometry. */
void appendFullProductionHCPrefix(Lane& lane,FullProductionHCPrefix& p,std::map<BufferId,TensorBase*>& arenas,
    DeviceId device,FullProductionHCProbe production=FullProductionHCProbe::Disabled,unsigned hidden=2560) {
    using F=Qwen4ExpFloatBinding;using Stage=Qwen4ExpPrimitiveStage;
    lane.hc_up_observation=p.logits.get();
    const auto count=static_cast<const std::int32_t*>(lane.active->gpu_data_ptr());
    const Qwen4ExpStageRows rows{1,Qwen4ExpIntInput{Qwen4ExpNativeIntInput{device,count,1}}};
    // The actual model reuses its previous attention-result tensor as the
    // later FFN output. Retain the same tensor AND graph buffer identity.
    const bool reuses_block=p.previous.get()==lane.combined.get();
    const auto block_id=reuses_block?BufferId::FFN_OUTPUT:BufferId::ATTN_OUTPUT;
    lane.graph.addNode("ab_previous_write",std::make_unique<Stage>(Stage::Params{.device_id=device,.rows=rows,
        .operation=Stage::ResidualWrite{int(hidden),F(*p.residual,BufferId::RESIDUAL),
            F(*p.previous,block_id),F(*p.raw,BufferId::QFN_HC_WRITE_LOGITS),
            F(*p.post,BufferId::ATTN_PROJ)}}),device);
    const auto down=p.down_prepared.store->binding(p.down_prepared.ref);
    const auto up=p.up_prepared.store->binding(p.up_prepared.ref);
    require(down&&up,"original HC prepared binding ownership");
    const Qwen4ExpHyperconnectionContext context{.device=device,.prefix="ab_hc_",.rows=rows,.hidden=int(hidden),
        .lowRank=320,.epsilon=1e-6f,.arithmetic=Qwen4ExpProjectionArithmetic::Ordinary,
        .upOperandPrecision=QuantizedActivationOperandPrecision::TwoTermQ8,
        .upProjectionRows=DeviceRowRange::deviceCounted(1,count)};
    const Qwen4ExpHyperconnectionReadWeights weights{
        Qwen4ExpImmutableWeightBinding(*p.gain),Qwen4ExpPreparedProjection(*down,*p.down_prepared.store,device),
        Qwen4ExpPreparedProjection(*up,*p.up_prepared.store,device)};
    const Qwen4ExpHyperconnectionReadBuffers buffers{
        F(*p.post,BufferId::ATTN_PROJ),F(*p.normalized,BufferId::QFN_HC_NORMALIZED),
        F(*p.down,BufferId::QFN_HC_DOWN),F(*p.activated,BufferId::UP_PROJ),
        F(*p.logits,BufferId::QFN_HC_READ_LOGITS),F(*lane.hidden,BufferId::NORMALIZED)};
    std::string read;
    if(production==FullProductionHCProbe::MiddleRead) {
        const auto inject=p.inject_prepared.store->binding(p.inject_prepared.ref);
        require(inject&&p.injection&&p.next_trunk,"original middle-read injection binding");
        auto capability=Qwen4ExpHyperconnectionBuilder::appendRead(lane.graph,context,
            {weights,Qwen4ExpPreparedProjection(*inject,*p.inject_prepared.store,device)},buffers,{"ab_previous_write"});
        read=capability.terminal();
        // A following production write consumes the POST-collective result.
        // Middle correctness therefore records this original collective too;
        // only final-read economy stops at the participant-local output.
        const std::string block="m1_decode/ffn_sum";
        const auto* sum=lane.graph.getNode(block);
        require(sum&&sum->stage->isCollectiveStage(),"middle read retains its actual final collective producer");
        const auto injection_id=p.injection.get()==p.raw.get()?BufferId::QFN_HC_WRITE_LOGITS:BufferId::GATE_PROJ;
        Qwen4ExpHyperconnectionBuilder::appendWrite(lane.graph,std::move(capability),
            {F(*lane.combined,BufferId::FFN_OUTPUT),F(*p.injection,injection_id),
             F(*p.next_trunk,BufferId::HIDDEN_STATE)},{block});
    } else read=Qwen4ExpHyperconnectionBuilder::appendFinalRead(lane.graph,context,weights,buffers,{"ab_previous_write"});
    lane.graph.addDependency(lane.graph.getNode("router")?"router":"shared",read);
    if(production!=FullProductionHCProbe::Disabled&&!lane.graph.getNode("router"))
        lowerFullNativeHCFFN(lane.graph,"shared","ab_previous_write",read);
    arenas={{BufferId::RESIDUAL,p.residual.get()},{block_id,p.previous.get()},
        {BufferId::QFN_HC_WRITE_LOGITS,p.raw.get()},{BufferId::ATTN_PROJ,p.post.get()},
        {BufferId::QFN_HC_NORMALIZED,p.normalized.get()},{BufferId::QFN_HC_DOWN,p.down.get()},
        {BufferId::UP_PROJ,p.activated.get()},{BufferId::QFN_HC_READ_LOGITS,p.logits.get()}};
    // makeLane already binds this exact combined tensor. Do not register a
    // second arena entry or relax its distinct-binding validation.
    if(reuses_block)arenas.erase(block_id);
    if(production==FullProductionHCProbe::MiddleRead) {
        // A declared model reuse keeps the SAME tensor and BufferId for the
        // incoming gate and later injection. Distinct control storage retains
        // its original independent identity.
        arenas[p.injection.get()==p.raw.get()?BufferId::QFN_HC_WRITE_LOGITS:BufferId::GATE_PROJ]=p.injection.get();
        arenas[BufferId::HIDDEN_STATE]=p.next_trunk.get();
    }
}

/**
 * @brief Resolve the full entry from original immutable physical format authority.
 * @param source Original routed source family. @param shared Original shared sources.
 * @param store Their already-prepared owner. @param device Exact participant.
 * @return Complete full-entry function, never a standalone HC or FFN function.
 */
const void* fullProductionHCFunction(const MoEOverlayParticipantLayerBank& source,
    const FrozenModelWeightSet& shared,PreparedWeightStore& store,DeviceId device,unsigned expected_groups=0) {
    const auto declaration=MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(source.fragment_family);
    auto mask=declaration.nativeGateUpEnvelope().execution_codebook_mask|declaration.nativeDownEnvelope().execution_codebook_mask;
    for(const char* name:{"ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"}) {
        const auto* binding=shared.optionalLayerForDevice(0,name,device);require(binding,"original shared full-entry source");
        const auto ref=store.preparedRefForBinding(binding->binding_id,device);require(ref.has_value(),"original shared full-entry prepared ref");
        auto* engine=store.gemmKernel(*ref);DeviceNativeQuantisedMatrixDesc descriptor;
        require(engine&&engine->weights_converted()&&engine->inspectPreparedNativeQuantisedMatrixDesc(descriptor)&&descriptor.valid(),
            "full-entry source authentication never performs weight preparation");
        mask|=std::uint32_t(1)<<descriptor.codebook_id;
    }
    const auto program=rocm::ResidentMoEFFNDecoderProgram::fromExecutionMask(mask);
    require(!expected_groups||program.groups()==expected_groups,"full production actual source envelope selects expected decoder group union");
    return rocm::fullNativeHCFFNKernelFunction(program);
}
/**
 * @brief Reject stale real publication seals without adding a native command.
 * @param lane Actual prepared full-region owner with its original canonical arena.
 * @details Both failures use the unchanged real snapshot epoch. Failed attempts
 * must leave the valid seal intact, then the actual executor's same selection
 * must reseal idempotently before capture. No private policy/oracle is installed.
 */
void requireFullProductionOutputDemandSealNegatives(Lane& lane) {
    auto* node=lane.graph.getNode("shared");
    auto* demand=node?dynamic_cast<IGraphOutputDemand*>(node->stage.get()):nullptr;
    require(demand,"actual full stage exposes the cold publication interface");
    const auto declaration=demand->coldSnapshotPublicationDeclaration();
    require(declaration.has_value(),"actual full stage declares original possible observations");
    const auto epoch=lane.output_demand_executor.snapshotConfigurationEpoch();
    const auto reject=[&](std::span<const std::string> selected,const char* why) {
        bool rejected=false;
        try {demand->sealGraphOutputDemand(lane.graph,lane.output_arena,selected,epoch);}
        catch(const std::logic_error&){rejected=true;}
        require(rejected,why);
    };
    const std::array<std::string,1> unknown{"__invalid_full_hc_observation__"};
    reject(unknown,"same-epoch unknown observation must reject a prepared full seal");
    std::vector<std::string> changed;
    if(lane.output_observation!=FullProductionHCObservation::ObservedAll)
        for(const auto& output:declaration->outputs)changed.emplace_back(output.name);
    reject(changed,"same-epoch changed selected roles must reject a prepared full seal");
    lane.output_demand_executor.sealGraphOutputDemands(lane.graph);
    std::osyncstream(std::cout)<<"FULL_PRODUCTION_OUTPUT_SEAL_NEGATIVES,unknown_same_epoch=1,changed_roles_same_epoch=1,unchanged_executor_reseal=1\n";
}
/**
 * @brief Authenticate the actual retained kernel's two by-value argument records.
 * @param capture Original instantiated local graph. @param function Expected cold source envelope.
 * @param p Original prefix owners. @param lane Original FFN/count owner.
 * @return Exact full-bank base and actual admitted grid, for cold/terminal slot observation.
 */
std::pair<std::uint32_t*,unsigned> requireFullProductionHCBinding(IGPUGraphCapture& capture,const void* function,
    const FullProductionHCPrefix& p,const Lane& lane) {
    auto* native=dynamic_cast<HIPGraphCapture*>(&capture);require(native&&native->graph(),"full production native HIP graph");
    std::size_t count=0;require(hipGraphGetNodes(native->graph(),nullptr,&count)==hipSuccess,"count full production native nodes");
    std::vector<hipGraphNode_t> nodes(count);require(hipGraphGetNodes(native->graph(),nodes.data(),&count)==hipSuccess,"read full production native nodes");
    unsigned matches=0;
    std::pair<std::uint32_t*,unsigned> barrier{};
    static_assert(FullNativeHCFFNBarrierLayout::prefix_phases==3&&FullNativeHCFFNBarrierLayout::suffix_phases==2&&
        FullNativeHCFFNBarrierLayout::prefix_words==6&&FullNativeHCFFNBarrierLayout::suffix_word_offset==10&&
        FullNativeHCFFNBarrierLayout::words==14&&FullNativeHCFFNBarrierLayout::bytes==56);
    for(const auto node:nodes) {
        hipGraphNodeType type;require(hipGraphNodeGetType(node,&type)==hipSuccess,"full production native node type");
        if(type!=hipGraphNodeTypeKernel)continue;
        hipKernelNodeParams params{};require(hipGraphKernelNodeGetParams(node,&params)==hipSuccess,"full production actual kernel parameters");
        if(params.func!=function)continue;
        ++matches;require(params.kernelParams&&params.kernelParams[0]&&params.kernelParams[1]&&params.blockDim.x==1024,
            "full production uses two by-value args and complete physical CTA");
        // Native arguments already contain valid typed row records. Copy their
        // complete representation without inventing an invalid default row owner.
        std::array<std::byte,sizeof(rocm::NativeLayoutRouterFFNArgs)> suffix_bytes;
        std::array<std::byte,sizeof(rocm::FullNativeHCFFNPrefixArgs)> prefix_bytes;
        std::memcpy(suffix_bytes.data(),params.kernelParams[0],suffix_bytes.size());
        std::memcpy(prefix_bytes.data(),params.kernelParams[1],prefix_bytes.size());
        const auto suffix=std::bit_cast<rocm::NativeLayoutRouterFFNArgs>(suffix_bytes);
        const auto prefix=std::bit_cast<rocm::FullNativeHCFFNPrefixArgs>(prefix_bytes);
        require(prefix.write.residual==p.residual->gpu_data_ptr()&&prefix.write.block==p.previous->gpu_data_ptr()&&
            prefix.write.gateLogits==p.raw->gpu_data_ptr()&&prefix.write.output==p.post->gpu_data_ptr()&&
            prefix.hc.norm.input==p.post->gpu_data_ptr()&&prefix.hc.norm.output==p.normalized->gpu_data_ptr()&&
            prefix.hc.raw_down==p.down->gpu_data_ptr()&&
            prefix.hc.publication.activated==(prefix.output_demand.contains(FullNativeHCFFNOutputRole::HCActivated)?p.activated->gpu_data_ptr():nullptr)&&
            prefix.hc.up.output==(prefix.output_demand.contains(FullNativeHCFFNOutputRole::HCUp)?p.logits->gpu_data_ptr():nullptr)&&
            prefix.hc.mixed==lane.hidden->gpu_data_ptr()&&
            suffix.ffn.hidden==prefix.hc.mixed&&suffix.ffn.combined_output==lane.combined->gpu_data_ptr(),
            "full production captured exact original public producer/consumer pointers");
        require(prefix.output_demand==fullProductionExpectedDemand(lane.output_observation),
            "actual captured demand follows original graph and executor observer policy");
        const auto optional_pointer=[&](FullNativeHCFFNOutputRole role,const void* actual,const void* original) {
            require(actual==(prefix.output_demand.contains(role)?original:nullptr),"captured optional pointer follows exact role demand");
        };
        optional_pointer(FullNativeHCFFNOutputRole::RouteContributions,suffix.ffn.route_contributions,lane.routes->gpu_data_ptr());
        optional_pointer(FullNativeHCFFNOutputRole::Routed,suffix.ffn.routed_output,lane.routed->gpu_data_ptr());
        optional_pointer(FullNativeHCFFNOutputRole::Shared,suffix.ffn.shared_output,lane.shared->gpu_data_ptr());
        optional_pointer(FullNativeHCFFNOutputRole::SharedGate,suffix.ffn.shared_gate_output,lane.gate_scratch->gpu_data_ptr());
        optional_pointer(FullNativeHCFFNOutputRole::SharedUp,suffix.ffn.shared_up_output,lane.up_scratch->gpu_data_ptr());
        require(prefix.barrier_state&&suffix.barrier_state==prefix.barrier_state+FullNativeHCFFNBarrierLayout::suffix_word_offset&&
            prefix.hc.norm.extent.liveRows==lane.active->gpu_data_ptr()&&
            prefix.hc.publication.rows.countOwner()==lane.active->gpu_data_ptr()&&
            prefix.hc.up.rows.countOwner()==lane.active->gpu_data_ptr(),"full production retains original count and distinct instance barrier subregions");
        const auto begin=reinterpret_cast<std::uintptr_t>(lane.workspace->getBuffer(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER));
        const auto address=reinterpret_cast<std::uintptr_t>(prefix.barrier_state);
        const auto bytes=lane.workspace->getBufferSize(MoEWorkspaceBuffers::NATIVE_LAYOUT_BARRIER);
        require(begin&&address>=begin&&bytes>=FullNativeHCFFNBarrierLayout::bytes&&
            address-begin<=bytes-FullNativeHCFFNBarrierLayout::bytes&&
            params.gridDim.x==prefix.hc.resident_grid&&params.gridDim.x==suffix.ffn.resident_grid,
            "actual three-prefix/two-suffix bank fits retained workspace and exact admitted grid");
        barrier={prefix.barrier_state,params.gridDim.x};
        // Cold captured metadata comes from these already-decoded actual args;
        // do not probe hardware or instrument the full kernel to recover it.
        std::osyncstream(std::cout)<<"FULL_PRODUCTION_CAPTURE_BINDING,grid_x="<<params.gridDim.x
            <<",grid_y="<<params.gridDim.y<<",grid_z="<<params.gridDim.z
            <<",block_x="<<params.blockDim.x<<",block_y="<<params.blockDim.y<<",block_z="<<params.blockDim.z
            <<",dynamic_lds_bytes="<<params.sharedMemBytes<<",H="<<suffix.ffn.hidden_columns
            <<",F="<<suffix.ffn.routed_columns<<",S="<<suffix.ffn.shared_columns
            <<",E="<<suffix.ffn.experts<<",topK="<<suffix.ffn.top_k
            <<",output_demand_bits="<<unsigned(prefix.output_demand.bits())<<",hc_rows="<<prefix.hc.norm.extent.rows<<",ffn_rows="<<suffix.ffn.rows.physicalRows()
            <<",hc_down_columns="<<prefix.hc.down_weights.n<<",hc_down_K="<<prefix.hc.down_weights.k
            <<",hc_down_P="<<prefix.hc.partitions<<",hc_up_K32_groups="<<prefix.hc.up.weights.blocks_per_row
            <<",hc_grid_policy="<<unsigned(prefix.hc.grid_policy)<<",injection="<<bool(prefix.hc.injection.weights)
            <<",injection_load_pairs="<<prefix.hc.injection.load_pairs
            <<",panel_rows="<<suffix.full_entry_panel.rows<<",panel_blocks="<<suffix.full_entry_panel.blocks
            <<",prefix_joins="<<FullNativeHCFFNBarrierLayout::prefix_phases
            <<",suffix_joins="<<FullNativeHCFFNBarrierLayout::suffix_phases
            <<",suffix_word_offset="<<FullNativeHCFFNBarrierLayout::suffix_word_offset<<'\n';
        const auto report_policy=[](const char* role,const rocm::ResidentMoEPartitionPolicy& policy,std::uint32_t mask) {
            std::osyncstream(std::cout)<<"FULL_PRODUCTION_CAPTURE_POLICY,role="<<role
                <<",p0="<<policy.p0<<",p1="<<policy.p1<<",p2="<<policy.p2<<",p3="<<policy.p3
                <<",maximum="<<policy.maximum<<",execution_mask="<<mask<<'\n';
        };
        report_policy("routed_gate",suffix.ffn.routed_gate_policy,suffix.ffn.execution.routed_gate);
        report_policy("routed_down",suffix.ffn.routed_down_policy,suffix.ffn.execution.routed_down);
        report_policy("shared_gate",suffix.ffn.shared_gate_policy,suffix.ffn.execution.shared_gate);
        report_policy("shared_down",suffix.ffn.shared_down_policy,suffix.ffn.execution.shared_down);
        require(p.injection?prefix.hc.injection.output==p.injection->gpu_data_ptr():
            (!prefix.hc.injection.weights&&!prefix.hc.injection.output),
            "full production optional original injection identity");
    }
    require(matches==1,"exactly one actual complete production kernel, no standalone prefix/suffix replay");
    return barrier;
}

/**
 * @brief Exercise one original source family through staged and full production captures.
 * @param device Exact admitted ROCm participant. @param member Original local coordinate.
 * @param tp Existing native collective owner; excluded from both local captures.
 * @param rendezvous Original per-device test lifecycle boundary.
 * @param banks Original source family authority. @param original_member Its retained source coordinate.
 * @param shared Original shared expert weight owners. @param partition Original typed TP declaration.
 * @param store Existing prepared store. @param shape Same actual local geometry on both arms.
 * @param mode Explicit final or middle read; no diagnostic launch mode is used.
 * @param storage_reuse Explicit original read-before-write tensor reuse; distinct remains the default.
 * @param expected_groups Optional exact union derived from original source declarations, never a forced program.
 * @param timing Explicit paired event observation of final/precollective fully-active operations only.
 * @param observation Existing cold selector or actual original outside-reader demand.
 */
void replayFullNativeHCFFN(DeviceId device,int member,ILocalTPContext& tp,std::barrier<>& rendezvous,
    MoEOverlayParticipantResidencyRegistry& banks,int original_member,const FrozenModelWeightSet& shared,
    const TensorParallelConfig& partition,PreparedWeightStore& store,OriginalFFNFixtureGeometry shape,
    FullProductionHCProbe mode,FullProductionHCStorageReuse storage_reuse=FullProductionHCStorageReuse::Distinct,
    unsigned expected_groups=0,bool timing=false,
    FullProductionHCObservation observation=FullProductionHCObservation::ObservedAll)
{
    require(device.is_rocm()&&tp.degree()==1&&member==0&&shape.hidden==2560&&mode!=FullProductionHCProbe::Disabled,
        "full production replay uses its explicit ordinary original M1 local source");
    require(!timing||(mode==FullProductionHCProbe::FinalRead&&storage_reuse==FullProductionHCStorageReuse::Distinct&&
        expected_groups>0&&expected_groups<16),"family economy requires explicit source group and final independent-input boundary");
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&] {
        auto* stream=worker.getOrCreateAuxiliaryStream("full_native_hc_ffn_original_replay");
        auto* backend=getBackendFor(device);require(backend&&stream,"full production exact stream/backend");
        auto context=IDeviceContext::create(device,1);
        constexpr unsigned H=2560,K=4*H,R=320,planes=15;
        const bool middle=mode==FullProductionHCProbe::MiddleRead;
        const bool observed_all=observation==FullProductionHCObservation::ObservedAll;
        const bool retained=observation==FullProductionHCObservation::RetainedDemandedOnly;
        const bool up_observed=fullProductionExpectedDemand(observation).contains(FullNativeHCFFNOutputRole::HCUp);
        const bool outside_reader=observation==FullProductionHCObservation::OutsideHCUpReader;
        require(!timing||!outside_reader,"outside reader is a functional scope, never a timing substitution");
        const std::array<std::size_t,planes> widths{K,K,R,R,K,H,std::size_t(shape.experts),H,middle?4u:0u,middle?K:0u,
            std::size_t(shape.top_k)*H,H,H,std::size_t(shape.shared),std::size_t(shape.shared)};
        std::array<FullProductionHCPrefix,2> prefix;
        std::array<std::unique_ptr<Lane>,2> lanes;
        std::array<std::unique_ptr<IGPUGraphCapture>,2> captures,children,timed;
        std::array<std::array<std::vector<float>,planes>,2> observed,baseline,reset;
        std::array<std::vector<float>,planes> omitted_poison;
        std::array<std::vector<float>,2> reader_observed;
        std::array<std::vector<float>,2> ids,weights,baseline_ids,baseline_weights,reset_ids,reset_weights;
        std::array<std::vector<float>,3> inputs;
        std::array<FullProductionRuntimeObservation,2> runtime_observed;
        int published_live_count=0;
        constexpr std::array table_names{MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
            MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE};
        constexpr unsigned char descriptor_poison=0xa5u;
        std::vector<unsigned char> closed_table(retained?shape.experts*sizeof(DeviceNativeQuantisedMatrixDesc):0);
        std::pair<std::uint32_t*,unsigned> barrier_binding{};
        std::array<std::uint32_t,FullNativeHCFFNBarrierLayout::words> barrier_observed{},barrier_expected{};
        const std::array<std::uint32_t,4> retired_barrier_poison{0x4d310601u,0x4d310702u,0x4d310803u,0x4d310904u};
        inputs[0].resize(K);inputs[1].resize(H);inputs[2].resize(4);
        for(unsigned plane=0;plane<planes;++plane) {
            omitted_poison[plane].resize(widths[plane]);
            for(unsigned i=0;i<widths[plane];++i)omitted_poison[plane][i]=-8192.f-float(32*plane)-float(i%7)/16.f;
        }
        if(outside_reader)for(auto& values:reader_observed)values.resize(H);
        for(unsigned side=0;side<2;++side) {
            for(unsigned plane=0;plane<planes;++plane)observed[side][plane].resize(widths[plane]);
            ids[side].resize(shape.top_k);weights[side].resize(shape.top_k);
            auto& state=runtime_observed[side];
            state.ids.resize(shape.top_k);state.participants.resize(shape.top_k);state.weights.resize(shape.top_k);
            state.selected.resize(shape.experts);state.local.resize(shape.experts);
            state.expected_selected.resize(shape.experts);state.expected_local.resize(shape.experts);
            state.acquired.resize(shape.top_k);
            for(auto& table:state.published)table.resize(shape.top_k);
        }
        auto frame=banks.initialJointLayer(original_member,0);require(bool(frame),"original HC/FFN source bank");
        auto setup=frame->acquireInitialConstruction();require(bool(setup),"original HC/FFN construction lease");
        const auto& source=setup->layerForModelLayer(0);
        const auto full_function=retained?
            rocm::retainedM1HCFFNKernelFunction(FullNativeHCFFNOutputDemand::none(),MoESelectedDescriptorDemand::AcquiredBankOnly):
            fullProductionHCFunction(source,shared,store,device,expected_groups);
        // All sources, captures and asynchronous destinations precede the guard.
        // Its destructor retires work before any borrowed storage can be released.
        ScopedGPUCompletionEvent completion(backend,device,stream);
        struct Retire {
            ScopedGPUCompletionEvent& event;std::array<std::unique_ptr<Lane>,2>& lanes;
            ~Retire(){if(!event.join())std::terminate();for(auto& lane:lanes)if(lane){lane->stream=nullptr;lane->worker=nullptr;}}
        } retire{completion,lanes};
        for(unsigned side=0;side<2;++side) {
            auto& p=prefix[side];
            p.residual=fullProductionHCFloat(K,device,stream);
            if(storage_reuse==FullProductionHCStorageReuse::Distinct)p.previous=fullProductionHCFloat(H,device,stream);
            p.raw=fullProductionHCFloat(4,device,stream);p.gain=fullProductionHCFloat(K,device,stream);
            for(unsigned i=0;i<K;++i)p.gain->mutable_data()[i]=.75f+float(i%17)/64.f;
            TransferEngine::prepareDeviceInput(p.gain.get(),device,stream);require(completion.join(),"retire original gain upload");
            p.post=fullProductionHCFloat(K,device,stream);p.normalized=fullProductionHCFloat(K,device,stream);
            p.down=fullProductionHCFloat(R,device,stream);p.activated=fullProductionHCFloat(R,device,stream);
            p.logits=fullProductionHCFloat(K,device,stream);
            if(outside_reader)p.reader_mixed=fullProductionHCFloat(H,device,stream);
            p.down_source=TestTensorFactory::createQ8_0Random({R,K},864411u);
            p.up_source=TestTensorFactory::createQ8_0Random({K,R},864412u);
            p.down_prepared=makeGpuPreparedGemm(p.down_source.get(),device,"full.hc.down");
            p.up_prepared=makeGpuPreparedGemm(p.up_source.get(),device,"full.hc.up");
            p.down_prepared.orchestrator->finalize();p.up_prepared.orchestrator->finalize();
            if(middle) {
                p.injection=storage_reuse==FullProductionHCStorageReuse::ModelReadBeforeWrite?p.raw:fullProductionHCFloat(4,device,stream);
                p.next_trunk=fullProductionHCFloat(K,device,stream);
                p.inject_source=TestTensorFactory::createFP32Random({4,size_t(K)},-.125f,.125f,864394u);
                p.inject_prepared=makeGpuPreparedFloatingPointGemm(p.inject_source.get(),device,"full.hc.injection");
                p.inject_prepared.orchestrator->finalize();
            }
            lanes[side]=makeLane(1,false,device,member,tp,worker,*context,stream,source,shared,partition,store,
                side?MoEIntermediateShardFFNExecution::ResidentCandidate:MoEIntermediateShardFFNExecution::Staged,
                MoEGateUpPublicationPolicy::Staged,OriginalM1HiddenQ8FixturePolicy::CountedOriginal,{},{},
                MoEDownPublicationPolicy::Staged,CompletionSerialFamilyReplay::Disabled,shape,side&&!retained,
                NativeRouterAliasProbe::None,nullptr,OriginalFFNCollectiveWitness::ProductionPayloadOnly,
                [&](Lane& lane,std::map<BufferId,TensorBase*>& arenas){
                    if(storage_reuse==FullProductionHCStorageReuse::ModelReadBeforeWrite) {
                        p.previous=lane.combined;
                        if(middle)p.injection=p.raw;
                        require(p.previous.get()==lane.combined.get()&&p.previous->gpu_data_ptr()==lane.combined->gpu_data_ptr(),
                            "previous block reuses exact original combined-output tensor");
                        require(!middle||p.injection.get()==p.raw.get(),"incoming gate and later injection share exact original tensor");
                    }
                    appendFullProductionHCPrefix(lane,p,arenas,device,mode);
                    if(side&&retained)lowerRetainedM1HCFFN(lane.graph,"ab_previous_write");
                    if(outside_reader) {
                        using Stage=Qwen4ExpPrimitiveStage;using F=Qwen4ExpFloatBinding;
                        const auto* count=static_cast<const int32_t*>(lane.active->gpu_data_ptr());
                        const Qwen4ExpStageRows rows{1,Qwen4ExpIntInput{Qwen4ExpNativeIntInput{device,count,1}}};
                        lane.graph.addNode("ab_hc_up_reader",std::make_unique<Stage>(Stage::Params{.device_id=device,.rows=rows,
                            .operation=Stage::ResidualRead{H,F(*p.normalized,BufferId::QFN_HC_NORMALIZED),
                                F(*p.logits,BufferId::QFN_HC_READ_LOGITS),F(*p.reader_mixed,BufferId::Q_PROJ)}}),device);
                        lane.graph.addDependency("ab_hc_up_reader","shared");
                        require(arenas.emplace(BufferId::Q_PROJ,p.reader_mixed.get()).second,"outside read keeps independent original output");
                        lane.outside_hc_up_reader=true;
                    }
                },side?observation:FullProductionHCObservation::ObservedAll);
            if(side)requireFullProductionOutputDemandSealNegatives(*lanes[side]);
        }
        const auto required_plane=[&](unsigned plane) {
            if(observed_all)return true;
            if(plane==6||plane>=10||plane==3)return false;
            if(plane==4)return up_observed;
            if(plane==2&&prefix[1].activated.get()==prefix[1].down.get())return false;
            return true;
        };
        const auto poison_plane=[&](unsigned plane) {
            if(required_plane(plane)||plane==6)return false;
            if((plane==2||plane==3)&&prefix[1].activated.get()==prefix[1].down.get())return false;
            return true;
        };
        if(!observed_all) {
            auto& p=prefix[1];auto& lane=*lanes[1];
            const std::array<FP32Tensor*,planes> tensors{p.post.get(),p.normalized.get(),p.down.get(),p.activated.get(),p.logits.get(),
                lane.hidden.get(),nullptr,lane.combined.get(),p.injection.get(),p.next_trunk.get(),lane.routes.get(),lane.routed.get(),
                lane.shared.get(),lane.gate_scratch.get(),lane.up_scratch.get()};
            for(unsigned plane=0;plane<planes;++plane)if(widths[plane]&&poison_plane(plane)) {
                require(tensors[plane],"omitted exclusive tensor exists for poison proof");
                std::memcpy(tensors[plane]->mutable_data(),omitted_poison[plane].data(),widths[plane]*sizeof(float));
                TransferEngine::prepareDeviceInput(tensors[plane],device,stream);
            }
        }
        require(completion.join(),"retire original cold source preparation and omitted-bank poison before capture");
        // This explicit fixture uses embedded request histograms, not an enabled
        // asynchronous maintenance bank. Observe its initial device counters so
        // cold preparation cannot be mistaken for captured inference demand.
        for(unsigned side=0;side<2;++side) {
            const auto& initial=lanes[side]->runtime->hostLayerState(0);
            require(!initial.runtime_histogram_banks&&!initial.overlay_epoch_ticket&&!initial.overlay_placement_banks,
                "original embedded-bank fixture retains its declared histogram/epoch topology");
            const auto* state=reinterpret_cast<const std::byte*>(lanes[side]->runtime->deviceLayerState(0));
            auto& observed_runtime=runtime_observed[side];
            require(backend->deviceToHostOnStream(observed_runtime.expected_selected.data(),
                state+offsetof(DeviceMoELayerRuntime,decode_histogram),shape.experts*sizeof(std::uint64_t),device.ordinal,stream),
                "observe cold selected histogram baseline");
            require(backend->deviceToHostOnStream(observed_runtime.expected_local.data(),
                state+offsetof(DeviceMoELayerRuntime,decode_local_histogram),shape.experts*sizeof(std::uint64_t),device.ordinal,stream),
                "observe cold local histogram baseline");
        }
        require(completion.join(),"retire cold runtime observations before capture");
        for(unsigned side=0;side<2;++side) {
            auto& lane=*lanes[side];captures[side]=worker.createGraphCapture(stream);
            for(const auto& name:lane.graph.getExecutionOrder()) {
                auto* stage=lane.graph.getNode(name)->stage.get();
                if(auto* consumer=dynamic_cast<IWorkspaceConsumer*>(stage))consumer->bindWorkspace(lane.workspace.get());
                require(stage->prepareGraphLaunch(context.get(),stream),"prepare original full production region");
            }
            auto local_plan=lane.plan;
            // capturePlan's internal producer indices name this complete
            // sequence. Final mode removes only its terminal collective, while
            // middle mode retains the real producer of the following write.
            if(!middle)std::erase_if(local_plan,[](const auto& step){return static_cast<const IComputeStage*>(step.stage_identity)->isCollectiveStage();});
            GraphCaptureDependencyLedger ledger(device,stream,local_plan,"full_production_hc_ffn_replay",
                GraphCaptureDependencyLedger::ExternalInputAuthority::BindDeclaredAddressesOnly);
            ScopedBackendGraphCapture capture(worker,*captures[side],"full_production_hc_ffn_replay",&ledger);
            require(capture.begin(),"begin original full production capture");
            for(const auto& name:lane.graph.getExecutionOrder()) {
                auto* stage=lane.graph.getNode(name)->stage.get();if(!middle&&stage->isCollectiveStage())continue;
                ScopedGraphCaptureStage scope(stage);require(stage->execute(context.get()),"capture original full production stage");scope.complete();
            }
            capture.finish();require(captures[side]->instantiate(),"instantiate original full production region");
            std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
            require(captures[side]->inspectKernelNodes(nodes,&error)&&!nodes.empty(),"inspect full production capture: "+error);
            if(side) {
                require(middle?nodes.size()>=2u:nodes.size()==(outside_reader?2u:1u),
                    "one full production entry, plus original collective/next write only for middle correctness");
                if(!retained)barrier_binding=requireFullProductionHCBinding(*captures[side],full_function,prefix[side],lane);
                const auto body=std::find_if(nodes.begin(),nodes.end(),[&](const auto& n){return n.function_identity==reinterpret_cast<std::uintptr_t>(full_function);});
                require(body!=nodes.end()&&body->local_memory_bytes_per_thread==0&&body->block_x==1024,
                    "actual full production entry retains zero private bytes and admitted physical geometry");
            }
            std::osyncstream(std::cout)<<"FULL_PRODUCTION_HC_SCOPE,arm="<<side<<",middle="<<middle
                <<",storage_reuse="<<(storage_reuse==FullProductionHCStorageReuse::ModelReadBeforeWrite)<<",kernels="<<nodes.size()<<",source=original_native,ordinary_rows=1,collective="
                <<(middle?"inside_middle_correctness_capture":"outside_final_local_capture")<<'\n';
        }
        if(!retained) {
            require(barrier_binding.first&&barrier_binding.second,"full capture retains its actual barrier owner");
            require(backend->deviceToHostOnStream(barrier_expected.data(),barrier_binding.first,sizeof(barrier_expected),
                device.ordinal,stream),"observe retired full barrier baseline");
            require(completion.join(),"retire barrier baseline before cold poison authoring");
            for(unsigned phase:{0u,1u,2u,5u,6u})require(barrier_expected[2*phase]==0&&barrier_expected[2*phase+1]<=1,
                "captured production active barrier slots start retired");
            std::copy(retired_barrier_poison.begin(),retired_barrier_poison.end(),
                barrier_expected.begin()+FullNativeHCFFNBarrierLayout::prefix_words);
            // These four arena words are retired capacity, never execution state.
            // The exact-stream cold upload intentionally poisons only unused slots;
            // it cannot reset an active arrival/sense or enter a captured bracket.
            require(backend->hostToDeviceOnStream(barrier_binding.first+FullNativeHCFFNBarrierLayout::prefix_words,
                retired_barrier_poison.data(),sizeof(retired_barrier_poison),device.ordinal,stream),"poison only retired prefix barrier slots");
            require(completion.join(),"retire unused-slot poison before graph replay");
        }
        if(retained)for(const auto name:table_names) {
            auto* table=lanes[1]->workspace->getBuffer(name);
            require(table&&backend->memset(table,descriptor_poison,closed_table.size(),device.ordinal,stream),
                "poison closed descriptor table outside retained capture");
        }
        const auto same=[](const std::vector<float>& a,const std::vector<float>& b) {
            return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
        };
        const auto publish=[&](unsigned history,int count) {
            published_live_count=count;
            for(unsigned i=0;i<K;++i)inputs[0][i]=.2f+float(int((i*13+history*17)%127)-63)/256.f;
            for(unsigned i=0;i<H;++i)inputs[1][i]=float(int((i*19+history*23)%193)-96)/512.f;
            for(unsigned i=0;i<4;++i)inputs[2][i]=float(int((history+i*3)%13)-6)/8.f;
            for(unsigned side=0;side<2;++side) {
                const std::array tensors{prefix[side].residual,prefix[side].previous,prefix[side].raw};
                for(unsigned i=0;i<tensors.size();++i){std::memcpy(tensors[i]->mutable_data(),inputs[i].data(),inputs[i].size()*sizeof(float));
                    TransferEngine::prepareDeviceInput(tensors[i].get(),device,stream);}
                lanes[side]->active->mutable_typed_data()[0]=count;
                TransferEngine::publishHostWrite(lanes[side]->active.get());
                TransferEngine::prepareDeviceInput(lanes[side]->active.get(),device,stream);
            }
        };
        const auto restore_previous_inputs=[&](unsigned side) {
            if(storage_reuse!=FullProductionHCStorageReuse::ModelReadBeforeWrite)return;
            // This is the next launch's external attention-result input, not a
            // download of the preceding FFN output. In middle mode the original
            // incoming four gates are also restored before the injector reuses
            // that same tensor. Seeded input owners outlive upload retirement.
            std::memcpy(prefix[side].previous->mutable_data(),inputs[1].data(),inputs[1].size()*sizeof(float));
            TransferEngine::prepareDeviceInput(prefix[side].previous.get(),device,stream);
            if(middle) {
                std::memcpy(prefix[side].raw->mutable_data(),inputs[2].data(),inputs[2].size()*sizeof(float));
                TransferEngine::prepareDeviceInput(prefix[side].raw.get(),device,stream);
            }
        };
        const auto observe=[&](unsigned side,unsigned operations=1) {
            const auto& p=prefix[side];const auto& lane=*lanes[side];
            const std::array<const void*,planes> pointers{p.post->gpu_data_ptr(),p.normalized->gpu_data_ptr(),p.down->gpu_data_ptr(),
                p.activated->gpu_data_ptr(),p.logits->gpu_data_ptr(),lane.hidden->gpu_data_ptr(),
                lane.workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS),lane.combined->gpu_data_ptr(),
                p.injection?p.injection->gpu_data_ptr():nullptr,p.next_trunk?p.next_trunk->gpu_data_ptr():nullptr,
                lane.routes->gpu_data_ptr(),lane.routed->gpu_data_ptr(),lane.shared->gpu_data_ptr(),
                lane.gate_scratch->gpu_data_ptr(),lane.up_scratch->gpu_data_ptr()};
            for(unsigned plane=0;plane<planes;++plane)if(widths[plane])
                require(backend->deviceToHostOnStream(observed[side][plane].data(),pointers[plane],widths[plane]*sizeof(float),
                    device.ordinal,stream),"observe original HC/public plane");
            require(backend->deviceToHostOnStream(ids[side].data(),lane.indices->gpu_data_ptr(),ids[side].size()*sizeof(float),device.ordinal,stream),"observe original IDs");
            require(backend->deviceToHostOnStream(weights[side].data(),lane.weights->gpu_data_ptr(),weights[side].size()*sizeof(float),device.ordinal,stream),"observe original probabilities");
            auto& runtime=runtime_observed[side];
            const auto* state=reinterpret_cast<const std::byte*>(lane.runtime->deviceLayerState(0));
            const auto copy_runtime=[&](void* target,std::size_t offset,std::size_t bytes,const char* why) {
                require(backend->deviceToHostOnStream(target,state+offset,bytes,device.ordinal,stream),why);
            };
            copy_runtime(runtime.header.data(),0,sizeof(runtime.header),"observe actual device runtime dimensions/epoch");
            copy_runtime(runtime.ids.data(),offsetof(DeviceMoELayerRuntime,topk_expert_ids),shape.top_k*sizeof(std::int32_t),
                "observe actual readiness-filtered selected IDs");
            copy_runtime(runtime.weights.data(),offsetof(DeviceMoELayerRuntime,topk_weights),shape.top_k*sizeof(float),
                "observe actual readiness-filtered selected weights");
            copy_runtime(runtime.selected.data(),offsetof(DeviceMoELayerRuntime,decode_histogram),shape.experts*sizeof(std::uint64_t),
                "observe actual selected demand");
            copy_runtime(runtime.local.data(),offsetof(DeviceMoELayerRuntime,decode_local_histogram),shape.experts*sizeof(std::uint64_t),
                "observe actual local demand");
            require(backend->deviceToHostOnStream(runtime.participants.data(),lane.runtime->hostLayerState(0).route_participant_ids,
                shape.top_k*sizeof(std::int32_t),device.ordinal,stream),"observe original route participant ledger");
            if(side&&!retained)require(backend->deviceToHostOnStream(barrier_observed.data(),barrier_binding.first,sizeof(barrier_observed),
                device.ordinal,stream),"observe actual active and retired full barrier words");
            if(outside_reader)require(backend->deviceToHostOnStream(reader_observed[side].data(),p.reader_mixed->gpu_data_ptr(),
                H*sizeof(float),device.ordinal,stream),"observe actual outside HC-up reader");
            require(completion.join(),"retire original full production observations");
            if(side&&!observed_all)for(unsigned plane=0;plane<planes;++plane)if(widths[plane]&&poison_plane(plane))
                require(same(observed[side][plane],omitted_poison[plane]),"omitted optional tensor retains authored poison");
            if(outside_reader)require(same(reader_observed[side],observed[side][5]),
                "actual production outside reader consumes HC-up and reproduces original mixed bytes");
            if(side&&!retained) {
                for(unsigned phase:{0u,1u,2u,5u,6u}) {
                    barrier_expected[2*phase+1]^=operations&1u;
                    require(barrier_observed[2*phase]==0&&barrier_observed[2*phase+1]==barrier_expected[2*phase+1],
                        "all and only three HC/two FFN joins advance once per captured operation");
                }
                for(unsigned word=FullNativeHCFFNBarrierLayout::prefix_words;
                    word<FullNativeHCFFNBarrierLayout::suffix_word_offset;++word)
                    require(barrier_observed[word]==barrier_expected[word],"unused retained prefix slots never change");
            }
            require(runtime.header[0]<2&&runtime.header[1]>0&&runtime.header[2]==unsigned(shape.experts)&&
                runtime.header[3]==unsigned(shape.top_k),"actual runtime shape and acquired epoch");
            const std::size_t bank_offset=offsetof(DeviceMoELayerRuntime,banks)+
                runtime.header[0]*sizeof(DeviceMoEPlacementBank);
            copy_runtime(runtime.bank_header.data(),bank_offset+offsetof(DeviceMoEPlacementBank,epoch),
                sizeof(runtime.bank_header),"observe actual acquired bank identity");
            const bool closed=side&&retained;
            if(closed)for(const auto name:table_names) {
                const auto* table=lane.workspace->getBuffer(name);
                require(backend->deviceToHostOnStream(closed_table.data(),table,closed_table.size(),device.ordinal,stream),
                    "observe entire closed selected descriptor table");
                require(completion.join(),"retire exact closed descriptor observation");
                require(std::all_of(closed_table.begin(),closed_table.end(),[&](auto byte){return byte==descriptor_poison;}),
                    "closed retained capture leaves every selected descriptor table byte untouched");
            }
            for(unsigned slot=0;slot<unsigned(shape.top_k);++slot) {
                require(ids[side][slot]==float(runtime.ids[slot]),"runtime selected ID matches original public router slot");
                require(std::memcmp(&weights[side][slot],&runtime.weights[slot],sizeof(float))==0,
                    "runtime selected weight matches original public router bits");
                const int expert=runtime.ids[slot];
                if(!published_live_count) {
                    require(expert==-1&&runtime.participants[slot]==-1&&runtime.weights[slot]==0.f,
                        "empty replay clears selected IDs/weights/participants without stale dispatch");
                    continue;
                }
                require(expert>=0&&expert<shape.experts&&runtime.participants[slot]==int(lane.runtime->hostLayerState(0).participant_id),
                    "singleton original intermediate shard dispatch is local and live");
                runtime.expected_selected[expert]+=operations;runtime.expected_local[expert]+=operations;
                copy_runtime(&runtime.acquired[slot],bank_offset+offsetof(DeviceMoEPlacementBank,experts)+
                    std::size_t(expert)*sizeof(DeviceMoEExpertDescriptor),sizeof(DeviceMoEExpertDescriptor),
                    "observe only selected acquired descriptor rows");
                if(!closed)for(unsigned role=0;role<3;++role) {
                    const auto* table=static_cast<const DeviceNativeQuantisedMatrixDesc*>(lane.workspace->getBuffer(table_names[role]));
                    require(table,"original selected descriptor publication bank exists");
                    require(backend->deviceToHostOnStream(&runtime.published[role][slot],table+expert,
                        sizeof(DeviceNativeQuantisedMatrixDesc),device.ordinal,stream),"observe selected published projection descriptor");
                }
            }
            require(completion.join(),"retire selected descriptor observations before comparison");
            require(runtime.bank_header[0]==runtime.header[1]&&runtime.bank_header[1]==unsigned(shape.experts),
                "published descriptors belong to the actual active epoch/bank");
            require(runtime.selected==runtime.expected_selected&&runtime.local==runtime.expected_local,
                "each captured live route increments selected/local demand once; empty rows increment neither");
            if(published_live_count)for(unsigned slot=0;slot<unsigned(shape.top_k);++slot) {
                const auto& acquired=runtime.acquired[slot];
                require(acquired.logical_expert_id==runtime.ids[slot]&&acquired.weight_format==DeviceMoEWeightFormat::NativeQuantised,
                    "selected bank descriptor retains original expert and native source identity");
                const std::array<const DeviceNativeQuantisedMatrixDesc*,3> expected{&acquired.gate,&acquired.up,&acquired.down};
                if(!closed)for(unsigned role=0;role<3;++role)require(std::memcmp(&runtime.published[role][slot],expected[role],
                    sizeof(DeviceNativeQuantisedMatrixDesc))==0,"selected projection table is copied from the actual acquired bank");
            }
        };
        const auto exact=[&](const char* context,bool live=true) {
            // Empty rows have no consumable router scratch or per-route/shared
            // numerical outputs. ROUTE_LOGITS may also alias retired HC quant
            // bytes. Active/reset rows compare every suffix publication; the
            // original retained HC/combined/selected outputs remain checked here.
            for(unsigned plane=0;plane<planes;++plane)if(required_plane(plane)&&(live||(plane!=6&&plane<10)))require(same(observed[0][plane],observed[1][plane]),
                std::string(context)+": exact staged HC/public plane "+std::to_string(plane));
            require(same(ids[0],ids[1])&&same(weights[0],weights[1]),std::string(context)+": exact staged routing");
            require(runtime_observed[0].header==runtime_observed[1].header&&
                runtime_observed[0].bank_header==runtime_observed[1].bank_header&&
                runtime_observed[0].ids==runtime_observed[1].ids&&runtime_observed[0].participants==runtime_observed[1].participants&&
                runtime_observed[0].selected==runtime_observed[1].selected&&runtime_observed[0].local==runtime_observed[1].local,
                std::string(context)+": exact staged acquired-runtime publications and demand");
        };
        for(unsigned history=0;history<20;++history) {
            publish(history==13?0:history,1);
            for(unsigned side=0;side<2;++side) {
                restore_previous_inputs(side);require(captures[side]->launch(),"launch retained original full production region");observe(side);
                baseline[side]=observed[side];baseline_ids[side]=ids[side];baseline_weights[side]=weights[side];
                restore_previous_inputs(side);require(captures[side]->launch(),"repeat retained original full production region");observe(side);
                for(unsigned plane=0;plane<planes;++plane)require(same(observed[side][plane],baseline[side][plane]),"same-engine repeated public plane");
                require(same(ids[side],baseline_ids[side])&&same(weights[side],baseline_weights[side]),"same-engine repeated route bytes");
            }
            exact("active replay");
            for(const auto& arm:observed)for(const auto& plane:arm)for(float value:plane)require(std::isfinite(value),"finite original full production public output");
            if(history==0){reset=observed;reset_ids=ids;reset_weights=weights;}
            if(history==1)for(unsigned side=0;side<2;++side)
                require(!same(observed[side][0],reset[side][0]),"both production arms consume changed external residual/write input");
            if(history==13)for(unsigned side=0;side<2;++side) {
                for(unsigned plane=0;plane<planes;++plane)require(same(observed[side][plane],reset[side][plane]),"request reset public bytes without recapture");
                require(same(ids[side],reset_ids[side])&&same(weights[side],reset_weights[side]),"request reset route bytes");
            }
        }
        publish(21,0);
        for(unsigned repeat=0;repeat<2;++repeat) {
            for(unsigned side=0;side<2;++side){restore_previous_inputs(side);require(captures[side]->launch(),"empty original full production replay");observe(side);}
            exact("empty replay",false);
        }
        publish(0,1);
        for(unsigned side=0;side<2;++side) {
            restore_previous_inputs(side);require(captures[side]->launch(),"active after empty without recapture");observe(side);
            for(unsigned plane=0;plane<planes;++plane)require(same(observed[side][plane],reset[side][plane]),"active reset after empty public bytes");
            require(same(ids[side],reset_ids[side])&&same(weights[side],reset_weights[side]),"active reset after empty route bytes");
        }
        {
            std::osyncstream proof(std::cout);
            proof<<"FULL_PRODUCTION_HC_REPLAY,middle="<<middle
                <<",storage_reuse="<<(storage_reuse==FullProductionHCStorageReuse::ModelReadBeforeWrite)
                <<",input_restored_before_each_launch="<<(storage_reuse==FullProductionHCStorageReuse::ModelReadBeforeWrite)
                <<",histories=20,repeated=1,reset=1,empty=2,active_after_empty=1,planes="<<planes
                <<",runtime_bank=1,selected_descriptors="<<(!retained)<<",histogram_per_launch=1,dirty_input=1";
            if(retained)proof<<",closed_descriptor_poison=1";
            else proof<<",prefix_joins=3,suffix_joins=2,retired_barrier_words=4";
            proof<<",byte_oracle=original_staged,grouped_qualified=0\n";
        }
        std::osyncstream(std::cout)<<"FULL_PRODUCTION_OUTPUT_DEMAND,observation="<<unsigned(observation)
            <<",bits="<<unsigned(fullProductionExpectedDemand(observation).bits())<<",histories=20,poison_checked="<<(!observed_all)
            <<",outside_reader="<<outside_reader<<",raw_activated_alias="<<(prefix[1].down.get()==prefix[1].activated.get())<<"\n";
        if(timing) {
            for(unsigned side=0;side<2;++side) {
                children[side]=worker.createGraphCapture(stream);
                std::array<std::string,32> names;std::vector<GPUOrderedTimelineStep> steps;
                for(unsigned i=0;i<32;++i){names[i]="full_operation_"+std::to_string(i);
                    steps.push_back({.name=names[i].c_str(),.capture=captures[side].get()});}
                require(children[side]->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::Disabled),
                    "build retained32 full production operations without internal event markers");
                timed[side]=worker.createGraphCapture(stream);
                const GPUOrderedTimelineStep parent{.name="full32",.capture=children[side].get()};
                require(timed[side]->buildOrderedTimelineTransaction(std::span(&parent,1),GPUOrderedTimelineInstrumentation::PerStepEvents)&&
                    timed[side]->instantiate(),"one event pair around complete32-operation family parent");
                require(timed[side]->nodeCount()==children[side]->nodeCount()+2,"exactly two parent event markers");
            }
            std::osyncstream(std::cout)<<"FULL_NATIVE_FAMILY_SCOPE,groups="<<expected_groups<<",H="<<H
                <<",F="<<shape.routed<<",S="<<shape.shared<<",E="<<shape.experts<<",topK="<<shape.top_k
                <<",arms=staged:0;full:1,scope=post+HC+router+localFFN,boundary=before_collective"
                <<",operations=32,records=4,warm_rounds=5,measured_rounds=32,samples_per_arm=64"
                <<",event_pairs_per_parent=1,static_histogram=true,isa=gfx906\n";
            for(unsigned round=0;round<37;++round) {
                const auto order=native_quantised_dispatch::pairedRecordTimingRound<4>(round+5);
                publish(unsigned(order.record),1);
                // Prime/check each actual record outside the only timed bracket.
                // Inputs are independent of every output, so all32 retained
                // operations execute identical original work without resets.
                for(unsigned side=0;side<2;++side) {
                    require(captures[side]->launch(),"prime actual family input record");observe(side);
                    baseline[side]=observed[side];baseline_ids[side]=ids[side];baseline_weights[side]=weights[side];
                }
                exact("paired family timing input record");
                const bool abba=order.order==native_quantised_dispatch::PairedTimingOrder::ABBA;
                const std::array<unsigned,4> sequence=abba?std::array{0u,1u,1u,0u}:std::array{1u,0u,0u,1u};
                for(unsigned slot=0;slot<sequence.size();++slot) {
                    const auto side=sequence[slot];require(completion.join(),"cold family preparation outside event interval");
                    require(timed[side]->launch(),"launch whole32-operation family parent");observe(side,32);
                    for(unsigned plane=0;plane<planes;++plane)require(same(observed[side][plane],baseline[side][plane]),
                        "actual32-parent original public bytes");
                    require(same(ids[side],baseline_ids[side])&&same(weights[side],baseline_weights[side]),
                        "actual32-parent original route bytes");
                    const auto snapshot=timed[side]->consumeOrderedTimelineTiming();
                    require(snapshot.state==GPUOrderedTimelineTimingState::Complete&&snapshot.samples.size()==1,
                        "one complete genuine32-operation event interval");
                    if(round>=5)std::osyncstream(std::cout)<<std::setprecision(12)
                        <<"FULL_NATIVE_FAMILY_SAMPLE,groups="<<expected_groups<<",arm="<<side
                        <<",round="<<round-5<<",record="<<order.record<<",order="<<(abba?"ABBA":"BAAB")
                        <<",slot="<<slot<<",operations=32,total_ms="<<snapshot.samples[0].elapsed_ms
                        <<",us_per_op="<<snapshot.samples[0].elapsed_ms*1000.0/32<<'\n';
                }
            }
        }
        require(completion.join(),"retire full production capture owner");
        rendezvous.arrive_and_wait();
    });
}
