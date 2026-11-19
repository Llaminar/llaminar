/**
 * @file MoEIntermediateShardResidentFFNReplay.inc
 * @brief Original-source retained replay of one local resident FFN against unchanged staged M1.
 *
 * Included by the original finalization fixture after its source, worker and
 * capture helpers. Both programs borrow the same admitted ModelContext source
 * family, prepared shared matrices and canonical collective implementation.
 * Twenty dirty/full/empty/reset histories compare all six original public
 * planes. The actual native function must match the immutable source envelope
 * (including promotions), not only a symbol prefix or current residency.
 * The final lifetime adversary releases names on the SAME workspace
 * manager while captured handles and retained leases remain alive; signature
 * validation must reject before another native submission can occur.
 */

#ifdef HAVE_ROCM
/**
 * @brief Query the original generated serial-M1 policy, without loading device data.
 * @param arithmetic_policy_codebook_id Canonical original source arithmetic policy.
 * @param N Original output width.
 * @param K Original reduction width.
 * @param kb Receives the original fixed partition count.
 * @param target_waves_per_cu Optional original launch-policy observation.
 * @return Whether the original generated policy exists for this exact geometry.
 */
extern "C" bool rocmGemv_native_quantised_query_serial_m1_config_with_policy(
    uint8_t arithmetic_policy_codebook_id,int N,int K,int* kb,int* target_waves_per_cu);
#endif

/** @brief Select the established replay gate or explicit complete-operation native-event observation. */
enum class ResidentFFNObservation {Replay,PairedEvents,RepeatedParent};
/** @brief One explicit local native program; mutually exclusive diagnostic choices. */
enum class LocalFFNProgram {Original,PartitionLocal,OrdinaryPhases};
/** @brief Exactly one local composition differs from the unchanged staged oracle. */
enum class LocalFFNReplayCandidate {Resident,GateUpLastProducer,GateUpPairedK32,DownLastProducer};
/**
 * @brief Qualify one participant's actual resident local FFN and explicit final sum.
 * @param device Original admitted ROCm worker.
 * @param member Original LocalTP coordinate, distinct from the source member ID.
 * @param tp Existing native collective context, never a handwritten exchange.
 * @param rendezvous Original cross-worker fixture lifecycle barrier.
 * @param banks Actual original-source residency registry.
 * @param original_member Original closed source-family member.
 * @param shared Same model's prepared participant-local shared source bindings.
 * @param shared_partition Original N/N/K slicing authority.
 * @param store Original prepared weight owner retained throughout both captures.
 * @param observation Replay gate or explicit separately selected Performance observation.
 * @param source_name Original fixture source label, never inferred from a kernel symbol.
 * @param native_layout_router Select the original-layout router/local-FFN closure through its actual graph lowerer.
 * @param published Optional genuine promoted fragment bank/runtime, retained through all native replay.
 * @param expected_native_layout_groups Optional original-source compiled-family assertion, never a dispatch override.
 */
void replayResidentParticipant(DeviceId device,int member,ILocalTPContext &tp,std::barrier<> &rendezvous,
    MoEOverlayParticipantResidencyRegistry &banks,int original_member,
    const FrozenModelWeightSet &shared,const TensorParallelConfig &shared_partition,PreparedWeightStore &store,
    ResidentFFNObservation observation=ResidentFFNObservation::Replay,std::string_view source_name={},
    LocalFFNReplayCandidate replay_candidate=LocalFFNReplayCandidate::Resident,LocalFFNProgram local_program=LocalFFNProgram::Original,
    OriginalFFNFixtureGeometry shape={},bool native_layout_router=false,
    const PublishedFFNEpoch* published=nullptr,std::uint8_t expected_native_layout_groups=0)
{
#ifdef HAVE_ROCM
    require(device.is_rocm(),"resident FFN replay requires its implemented ROCm participant");
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&] {
        const int kH=shape.hidden,kTopK=shape.top_k;
        const bool partition_local=local_program==LocalFFNProgram::PartitionLocal;
        const bool ordinary_phases=local_program==LocalFFNProgram::OrdinaryPhases;
        const rocm::ScopedResidentMoEFFNProgram resident_program(ordinary_phases
            ?rocm::ResidentMoEFFNProgram::OrdinaryPhases:partition_local
            ?rocm::ResidentMoEFFNProgram::PartitionLocal:rocm::ResidentMoEFFNProgram::Original);
        require(!(partition_local||ordinary_phases)||replay_candidate==LocalFFNReplayCandidate::Resident,
            "partition-local program only applies to the complete resident candidate");
        if(published)require(native_layout_router&&observation==ResidentFFNObservation::Replay,
            "promoted epoch qualification retains the complete NativeLayout byte replay");
        // The actual worker, not its submitting test thread, owns cold launch policy.
        const rocm::ScopedGateUpProducerLoads producer_loads(replay_candidate==LocalFFNReplayCandidate::GateUpPairedK32
            ?rocm::GateUpProducerLoads::PairedK32:rocm::GateUpProducerLoads::Original);
        auto *local_tp=dynamic_cast<LocalTPContext *>(&tp);
        require(local_tp!=nullptr,"resident replay original native LocalTP implementation");
        local_tp->setCurrentDeviceIndex(member);
        auto *stream=worker.getOrCreateAuxiliaryStream("original_intermediate_ffn_finalizer");
        auto context=IDeviceContext::create(device,1);
        auto *backend=getBackendFor(device);
        require(backend!=nullptr,"resident replay original native backend");
        ScopedGPUCompletionEvent completion(backend,device,stream);
        // Authenticate the selected compiled entry from the SAME immutable
        // source family and original prepared shared engines as the builder.
        // This is setup metadata only; no current routed descriptor is read.
        std::array<ITensorGemm*,3> shared_engines{};
        constexpr std::array shared_names{"ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
        for(std::size_t role=0;role<shared_names.size();++role)
        {
            const auto* binding=shared.optionalLayerForDevice(0,shared_names[role],device);
            require(binding!=nullptr,"resident original shared source binding for decoder authentication");
            const auto ref=store.preparedRefForBinding(binding->binding_id,device);
            require(ref.has_value(),"resident original shared prepared reference for decoder authentication");
            shared_engines[role]=store.gemmKernel(*ref);
        }
        const auto* shared_gate_source=shared.optionalLayerForDevice(0,"ffn_gate_shexp.weight",device);
        require(shared_gate_source&&shared_gate_source->tensor,"original shared source owns local descriptor width");
        const int shared_columns=static_cast<int>(shared_gate_source->tensor->rows());
        require(shared_columns>0&&shared_columns%32==0,"original local shared descriptor width");
        DeviceMoEExpertDescriptor original_shared;
        require(exportDeviceMoEExpertWeightDescriptors(shared_engines[0],shared_engines[1],shared_engines[2],
            kH,shared_columns,original_shared),"resident original shared descriptor triple for decoder authentication");
        NativeQuantisedExecutionFormatEnvelope shared_gate_envelope,shared_down_envelope;
        const auto add_shared=[](NativeQuantisedExecutionFormatEnvelope& envelope,const DeviceNativeQuantisedMatrixDesc& descriptor) {
            return addNativeQuantisedExecutionFormat(envelope,descriptor.codebook_id,
                {.codebook_id=descriptor.source_codebook_id,.is_superblock=bool(descriptor.source_is_superblock),
                 .present=bool(descriptor.source_identity_present)},false);
        };
        require(add_shared(shared_gate_envelope,original_shared.gate)&&add_shared(shared_gate_envelope,original_shared.up)&&
            add_shared(shared_down_envelope,original_shared.down),"resident immutable shared physical decoder envelopes");
        const void* expected_function=nullptr;
        std::array<const void*,2> expected_ordinary_functions{};
        std::uint64_t expected_ordinary_gate_grid=0;
        std::array<std::size_t,2> expected_ordinary_dynamic_bytes{};
        int routed_columns=0;
        const bool down_completion=replay_candidate==LocalFFNReplayCandidate::DownLastProducer;
        const bool tile_completion=replay_candidate!=LocalFFNReplayCandidate::Resident;
        const auto* counter_name=down_completion?MoEWorkspaceBuffers::DOWN_TILE_COMPLETION:MoEWorkspaceBuffers::GATEUP_TILE_COMPLETION;
        const auto build=[&](MoEIntermediateShardFFNExecution execution,
            MoEGateUpPublicationPolicy publication=MoEGateUpPublicationPolicy::Staged,
            MoEDownPublicationPolicy down_publication=MoEDownPublicationPolicy::Staged,
            NativeRouterAliasProbe alias_probe=NativeRouterAliasProbe::None) {
            auto frame=published?banks.jointLayer(original_member,0,published->epoch()):banks.initialJointLayer(original_member,0);
            require(frame!=nullptr,"resident original joint layer");
            auto setup=published?frame->bankLease():frame->acquireInitialConstruction();
            require(bool(setup),"resident setup-only original source reader");
            const auto& source=setup->layerForModelLayer(0);
            if(published) {
                require(published->runtime && published->arena && setup->epoch==published->epoch() &&
                    !published->promoted_experts.empty(),"promoted replay retains exact completed epoch and actual CPU-arrival IDs");
                int original_parts=0;
                int original_n=0,original_k=0;
                require(!published->canonical_experts.empty(),"mixed physical packet retains an original GPU resident");
                for(const int expert:published->canonical_experts) {
                    require(expert>=0&&expert<int(source.fragments.size())&&source.resident_mask.at(expert),
                        "canonical packet expert belongs to the actual published bank");
                    const auto& payload=source.fragments.at(expert);
                    DeviceNativeQuantisedMatrixDesc descriptor;
                    require(payload.down()&&payload.down()->exportNativeQuantisedMatrixDesc(descriptor)&&
                        descriptor.codebook_id==7&&descriptor.source_identity_present&&descriptor.source_codebook_id==7&&
                        !descriptor.source_is_superblock&&descriptor.mins&&!descriptor.emins,
                        "retained original GPU Q5_1 keeps physical7 and source7");
                    int parts=0;
                    require(rocmGemv_native_quantised_query_serial_m1_config_with_policy(descriptor.source_codebook_id,
                        descriptor.n,descriptor.k,&parts,nullptr)&&parts>0,
                        "canonical GPU resident resolves original source7 partitions");
                    if(original_parts)require(parts==original_parts&&descriptor.n==original_n&&descriptor.k==original_k,
                        "retained canonical packet experts have the same original geometry and partitions");
                    original_parts=parts;original_n=descriptor.n;original_k=descriptor.k;
                }
                for(const int expert:published->promoted_experts) {
                    require(expert>=0 && expert<int(source.fragments.size()) && source.resident_mask.at(expert),
                        "promoted expert belongs to this actual destination bank");
                    const auto& payload=source.fragments.at(expert);
                    DeviceNativeQuantisedMatrixDesc descriptor;
                    require(payload.arrivalEpoch()==published->epoch() && payload.down() &&
                        payload.down()->exportNativeQuantisedMatrixDesc(descriptor) &&
                        descriptor.codebook_id==kNativeQuantisedExpandedInt8MinCodebook &&
                        descriptor.source_identity_present && descriptor.source_codebook_id==7 &&
                        !descriptor.source_is_superblock && descriptor.mins && !descriptor.emins,
                        "actual Q5_1 CPU arrival retains physical23 and original source7 policy");
                    int parts=0;
                    require(rocmGemv_native_quantised_query_serial_m1_config_with_policy(descriptor.source_codebook_id,
                        descriptor.n,descriptor.k,&parts,nullptr)&&parts==original_parts&&
                        descriptor.n==original_n&&descriptor.k==original_k,
                        "physical23 arrival preserves the exact original physical7 source-policy partitions and K intervals");
                }
            }

            const auto geometry=DeviceMoEIntermediateShardBinding::fromOriginalFamily(*source.fragment_family);
            require(geometry.modelColumns()==kH,"native-event fixture original H authority");
            if(routed_columns)require(routed_columns==int(geometry.localIntermediateColumns()),"timed lanes retain original local F");
            routed_columns=int(geometry.localIntermediateColumns());
            if(shape.contract==OriginalFFNGeometryContract::QFNTensorSharded||
                shape.contract==OriginalFFNGeometryContract::QFNSingletonShardShape) {
                require(native_layout_router&&(tp.degree()==1||tp.degree()==2||tp.degree()==4||tp.degree()==8)&&
                    shared_columns==shared_partition.forDevice(device).d_ff_count&&
                    original_shared.gate.n==shared_columns&&original_shared.gate.k==kH&&
                    original_shared.up.n==shared_columns&&original_shared.up.k==kH&&
                    original_shared.down.n==kH&&original_shared.down.k==shared_columns,
                    "exact TensorSharded independently prepared routed and shared descriptor geometry");
                if(shape.contract==OriginalFFNGeometryContract::QFNSingletonShardShape)
                    require(tp.degree()==1&&(routed_columns==64||routed_columns==96)&&
                        routed_columns==shape.routed&&shared_columns==shape.shared&&routed_columns+shared_columns==160,
                        "TP8 local-shape component retains its independently generated singleton source scope");
                const auto check_source=[](const DeviceNativeQuantisedMatrixDesc& descriptor,
                    int n,int k,uint8_t codebook,bool superblock,int expected_parts) {
                    int parts=0;
                    require(descriptor.n==n&&descriptor.k==k&&descriptor.codebook_id==codebook&&descriptor.source_identity_present&&
                        descriptor.source_codebook_id==codebook&&bool(descriptor.source_is_superblock)==superblock&&
                        rocmGemv_native_quantised_query_serial_m1_config_with_policy(
                            descriptor.source_codebook_id,n,k,&parts,nullptr)&&parts==expected_parts,
                        "exact TP original source descriptor and production-resolved partitions");
                };
                check_source(original_shared.gate,shared_columns,kH,19,false,16);
                check_source(original_shared.up,shared_columns,kH,19,false,16);
                check_source(original_shared.down,kH,shared_columns,19,false,std::min(16,shared_columns/32));
                require(source.fragments.size()==size_t(shape.experts),"exact TP complete original expert family");
                for(const auto& payload:source.fragments) {
                    DeviceNativeQuantisedMatrixDesc gate,up,down;
                    require(payload.gate()&&payload.up()&&payload.down()&&
                        payload.gate()->exportNativeQuantisedMatrixDesc(gate)&&
                        payload.up()->exportNativeQuantisedMatrixDesc(up)&&
                        payload.down()->exportNativeQuantisedMatrixDesc(down),
                        "exact TP exports actual original prepared expert triplets");
                    check_source(gate,routed_columns,kH,5,true,16);
                    check_source(up,routed_columns,kH,5,true,16);
                    check_source(down,kH,routed_columns,7,false,std::min(16,routed_columns/32));
                }
            }
            const auto declaration=MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(source.fragment_family);
            const auto required=declaration.nativeGateUpEnvelope().execution_codebook_mask|
                declaration.nativeDownEnvelope().execution_codebook_mask|shared_gate_envelope.execution_codebook_mask|
                shared_down_envelope.execution_codebook_mask;
            if(published) require((declaration.nativeDownEnvelope().execution_codebook_mask&(uint32_t(1)<<23)) &&
                (declaration.nativeDownEnvelope().policy_codebook_mask&(uint32_t(1)<<7)) &&
                !(declaration.nativeDownEnvelope().policy_codebook_mask&(uint32_t(1)<<23)),
                "promoted descriptor retains original source7 partition policy and independently admits physical23");
            const auto program=rocm::ResidentMoEFFNDecoderProgram::fromExecutionMask(required);
            if(expected_native_layout_groups) {
                require(native_layout_router&&program.groups()==expected_native_layout_groups,
                    "original source selects the explicitly required NativeLayout decoder family");
                require(expected_native_layout_groups!=15 || program.executionMask()==25952241u,
                    "full-union original source selects all18 physical decoder implementations");
            }
            const auto maximum_tasks=rocm::ResidentMoEFFNTaskGeometry::projection(
                std::max({kH,routed_columns,shared_columns}),kTopK,NativeQuantisedGroupedDecodePolicy::maximum_k_partitions);
            require(maximum_tasks.uses32BitScheduler(),"focused original source geometry selects its checked32 scheduler");
            if(ordinary_phases)
            {
                expected_ordinary_functions=rocm::ordinaryMoEFFNKernelFunctions(program);
                const auto maximum_partitions=[&](int columns,int reduction,uint32_t policies) {
                    int maximum=0;
                    for(unsigned bit=0;bit<32;++bit)if(policies&(uint32_t(1)<<bit))
                    {
                        int partitions=0;
                        require(rocmGemv_native_quantised_query_serial_m1_config_with_policy(
                            static_cast<uint8_t>(bit),columns,reduction,&partitions,nullptr)&&partitions>0&&partitions<=NativeQuantisedGroupedDecodePolicy::maximum_k_partitions,
                            "ordinary native grid resolves exact original source partition policy");
                        maximum=std::max(maximum,partitions);
                    }
                    require(maximum>0,"ordinary native grid retains a nonempty original policy envelope");
                    return maximum;
                };
                const auto tiles=[&](int columns,uint32_t policies) {
                    const auto columns_per_tile=256/maximum_partitions(columns,kH,policies);
                    return (uint64_t(columns)+columns_per_tile-1)/columns_per_tile;
                };
                expected_ordinary_gate_grid=tiles(routed_columns,declaration.nativeGateUpEnvelope().policy_codebook_mask)*kTopK+
                    tiles(shared_columns,shared_gate_envelope.policy_codebook_mask);
                const auto routed_down_parts=maximum_partitions(kH,routed_columns,declaration.nativeDownEnvelope().policy_codebook_mask);
                const auto shared_down_parts=maximum_partitions(kH,shared_columns,shared_down_envelope.policy_codebook_mask);
                const auto routed_workers=std::min(8,routed_down_parts);
                const auto routed_batch_capacity=8/routed_workers;
                const auto values=size_t(kTopK)*routed_columns+shared_columns;
                expected_ordinary_dynamic_bytes={size_t(kH)+size_t(kH/32)*8,
                    values+values/32*8+size_t(32)*std::max(routed_batch_capacity*routed_down_parts,shared_down_parts)*sizeof(float)+
                        size_t(kTopK)*32*sizeof(float)};
            }
            const auto* function=native_layout_router?rocm::nativeLayoutRouterFFNKernelFunction(program):replay_candidate==LocalFFNReplayCandidate::Resident
                ? (ordinary_phases?expected_ordinary_functions[0]:rocm::residentMoEFFNKernelFunction(program,rocm::ResidentMoEFFNSchedulerWidth::UInt32))
                : down_completion?rocm::downLastProducerKernelFunction():rocm::gateUpLastProducerKernelFunction();
            if(expected_function)require(function==expected_function,"original and candidate retain the same cold source-envelope program");
            expected_function=function;

            auto lane=makeLane(1,false,device,member,tp,worker,*context,stream,
                source,shared,shared_partition,store,execution,publication,
                OriginalM1HiddenQ8FixturePolicy::CountedOriginal,{},{},down_publication,
                tile_completion&&observation==ResidentFFNObservation::Replay&&
                    (publication==MoEGateUpPublicationPolicy::LastProducer||down_publication==MoEDownPublicationPolicy::LastProducerRouteFold)
                    ?CompletionSerialFamilyReplay::PrefillScratchBeforeDecode:CompletionSerialFamilyReplay::Disabled,shape,
                native_layout_router&&execution==MoEIntermediateShardFFNExecution::ResidentCandidate,alias_probe,published,
                observation==ResidentFFNObservation::PairedEvents
                    ?OriginalFFNCollectiveWitness::ProductionPayloadOnly:OriginalFFNCollectiveWitness::DiagnosticSideband);
            setup={}; // Initial source reader never survives into recording.
            captureLane(*lane,false,device,member,worker,*context,stream,rendezvous);
            return lane;
        };
        if(native_layout_router) {
            for(const auto probe:{NativeRouterAliasProbe::RouterOutput,NativeRouterAliasProbe::FFNOutput}) {
                bool rejected=false;
                try {
                    (void)build(MoEIntermediateShardFFNExecution::ResidentCandidate,
                        MoEGateUpPublicationPolicy::Staged,MoEDownPublicationPolicy::Staged,probe);
                } catch(const std::invalid_argument& error) {
                    const std::string expected=probe==NativeRouterAliasProbe::RouterOutput
                        ?"Native-layout router output aliases its original router weights"
                        :"Native-layout FFN output aliases original router weights";
                    require(error.what()==expected,"alias probe reached the actual native-layout owner overlap guard");
                    rejected=true;
                }
                require(rejected,"native-layout owner rejects output overlap with router weights before capture");
            }
        }
        auto original=build(MoEIntermediateShardFFNExecution::Staged);
        auto candidate=replay_candidate==LocalFFNReplayCandidate::Resident
            ? build(MoEIntermediateShardFFNExecution::ResidentCandidate)
            : down_completion
                ? build(MoEIntermediateShardFFNExecution::Staged,MoEGateUpPublicationPolicy::Staged,MoEDownPublicationPolicy::LastProducerRouteFold)
                : build(MoEIntermediateShardFFNExecution::Staged,MoEGateUpPublicationPolicy::LastProducer);
        const std::string_view candidate_symbol=native_layout_router?"nativeLayoutRouterFFNKernel":replay_candidate==LocalFFNReplayCandidate::Resident
            ? (ordinary_phases?"rocmOrdinaryMoEFFN":partition_local?"rocmPersistentMoEFFNKernel":"rocmResidentMoEFFNKernel"):down_completion?"moe_grouped_native_quantised_down_last_producer_kernel":
              replay_candidate==LocalFFNReplayCandidate::GateUpPairedK32
                ?"moe_grouped_native_quantised_gate_up_paired_k32_last_producer_kernel":
                 "moe_grouped_native_quantised_gate_up_last_producer_kernel";
        const auto planes=[](Lane &lane) {
            return std::array<FP32Tensor *,6>{lane.routes.get(),lane.routed.get(),lane.shared.get(),
                lane.combined.get(),lane.gate_scratch.get(),lane.up_scratch.get()};
        };
        const auto signatures=[](Lane &lane) {
            std::vector<uint64_t> result;
            for(const auto &name:lane.graph.getExecutionOrder())
                result.push_back(lane.graph.getNode(name)->stage->graphCaptureVariantSignature());
            return result;
        };
        const auto original_signature=signatures(*original),candidate_signature=signatures(*candidate);
        if(native_layout_router)require(candidate->graph.getNode("router")==nullptr&&candidate->graph.size()==(published?4u:2u),
            "native-layout router FFN contains actual router once and preserves final collective");
        auto *original_native=dynamic_cast<HIPGraphCapture *>(original->capture.get());
        auto *candidate_native=dynamic_cast<HIPGraphCapture *>(candidate->capture.get());
        require(original_native && candidate_native,"resident replay retained actual native HIP owners");
        const auto original_graph=original_native->graph(),candidate_graph=candidate_native->graph();
        const auto original_executable=original_native->executable(),candidate_executable=candidate_native->executable();
        require(original_graph && candidate_graph && original_executable && candidate_executable,
            "resident replay complete native graph/executable handles");
        std::vector<GPUGraphKernelNodeInfo> original_nodes,candidate_nodes;std::string error;
        require(original->capture->inspectKernelNodes(original_nodes,&error),"inspect original FFN kernels: "+error);
        require(candidate->capture->inspectKernelNodes(candidate_nodes,&error),"inspect resident FFN kernels: "+error);
        const auto resident_nodes=std::count_if(candidate_nodes.begin(),candidate_nodes.end(),[&](const auto &node) {
            return node.name_resolved && node.name.find(candidate_symbol)!=std::string::npos;
        });
        require(resident_nodes==(ordinary_phases?2:1),"exact ordered native FFN entries in the retained candidate");
        require(candidate_nodes.size()<original_nodes.size() && candidate->capture->nodeCount()<original->capture->nodeCount(),
            "resident lowering removes actual native nodes while retaining router and collective");
        if(native_layout_router)
            for(const auto& node:candidate_nodes)
                require(node.name.find("gwsInit")==std::string::npos,"ordinary resident capture contains no GWS initialization dispatch");
        for(const auto &node:candidate_nodes)
        {
            require(node.valid(),"resident retained native node has complete resources");
            if(node.name.find(candidate_symbol)!=std::string::npos)
            {
                require(native_layout_router?node.function_identity==reinterpret_cast<std::uintptr_t>(expected_function): (ordinary_phases
                    ?std::any_of(expected_ordinary_functions.begin(),expected_ordinary_functions.end(),[&](const void* function){
                        return node.function_identity==reinterpret_cast<std::uintptr_t>(function);})
                    :node.function_identity==reinterpret_cast<std::uintptr_t>(expected_function)),
                    "resident FFN captured the exact smallest full-source-envelope decoder family and scheduler entry");
                require(node.local_memory_bytes_per_thread==0 && node.block_x==
                    (native_layout_router?1024u:replay_candidate==LocalFFNReplayCandidate::Resident?256u:
                        down_completion?NativeDownCompletionPlan::columns_per_tile:64u),
                    "resident FFN retains its admitted zero-private-memory physical CTA");
            }
        }
        if(tile_completion)
        {
            require(candidate_nodes.size()+(down_completion?2:1)==original_nodes.size(),"last-producer replaces the exact original native arithmetic region with one");
            const auto original_producer=std::find_if(original_nodes.begin(),original_nodes.end(),[&](const auto& node) {
                return node.name.find(down_completion?"moe_grouped_native_quantised_down_descriptor_kpart_kernel":
                    "moe_grouped_native_quantised_gate_up_descriptor_kpart_kernel")!=std::string::npos&&
                    node.grid_y==kTopK;
            });
            const auto candidate_producer=std::find_if(candidate_nodes.begin(),candidate_nodes.end(),[&](const auto& node) {
                return node.name.find(candidate_symbol)!=std::string::npos;
            });
            require(original_producer!=original_nodes.end()&&candidate_producer!=candidate_nodes.end(),"same original routed split-K producers");
            const unsigned paired=down_completion?2u:1u;
            require((original_producer->grid_x/paired+(original_producer->grid_x%paired!=0))==candidate_producer->grid_x&&
                original_producer->grid_y==candidate_producer->grid_y&&original_producer->grid_z==candidate_producer->grid_z&&
                original_producer->block_x*paired==candidate_producer->block_x,
                "last-producer preserves every original64-column logical task and unchanged split-K policy");
        }
        // Ordinary resident submission must never import a cooperative attribute
        // or its runtime GWS side dispatch. This runs outside captured execution.
        const std::uint32_t* candidate_barrier_state=nullptr;
        const auto require_direct_graph=[&](hipGraph_t graph,size_t expected_instances=1) {
            size_t found=0;
            const auto inspect=[&](auto&& self,hipGraph_t current)->void {
                size_t count=0;require(hipGraphGetNodes(current,nullptr,&count)==hipSuccess,"count direct nodes");
                std::vector<hipGraphNode_t> nodes(count);
                require(hipGraphGetNodes(current,nodes.data(),&count)==hipSuccess,"read direct nodes");
                for(auto node:nodes) {
                    hipGraphNodeType type;require(hipGraphNodeGetType(node,&type)==hipSuccess,"direct node type");
                    if(type==hipGraphNodeTypeGraph) {
                        hipGraph_t child;require(hipGraphChildGraphNodeGetGraph(node,&child)==hipSuccess,"direct child");self(self,child);
                    } else if(type==hipGraphNodeTypeKernel) {
                        hipKernelNodeParams params{};require(hipGraphKernelNodeGetParams(node,&params)==hipSuccess,"direct params");
                        if(params.func==expected_function) {
                            hipKernelNodeAttrValue attribute{};
                            require(hipGraphKernelNodeGetAttribute(node,hipKernelNodeAttributeCooperative,&attribute)==hipSuccess&&
                                !attribute.cooperative,"resident ordinary node must be noncooperative");
                            require(params.kernelParams&&params.kernelParams[0],"direct node retains typed parameters");
                            const auto& binding=*static_cast<const rocm::NativeLayoutRouterFFNArgs*>(params.kernelParams[0]);
                            require(binding.barrier_state!=nullptr,"direct node retains barrier slot");
                            if(candidate_barrier_state)require(candidate_barrier_state==binding.barrier_state,"import preserves exact barrier slot");
                            candidate_barrier_state=binding.barrier_state;
                            ++found;
                        }
                    }
                }
            };
            inspect(inspect,graph);require(found==expected_instances,"exact direct resident operation count");
        };
        // Inspect native attributes cold, including imported copies below. A
        // source-level ordinary launcher is insufficient proof of execution mode.
        const auto require_ordinary_graph=[&](hipGraph_t graph,size_t expected_instances=1) {
            std::array<unsigned,2> found{};
            const auto inspect=[&](auto&& self,hipGraph_t current)->void {
                size_t count=0;
                require(hipGraphGetNodes(current,nullptr,&count)==hipSuccess,"count native ordinary phase nodes");
                std::vector<hipGraphNode_t> nodes(count);
                std::array<std::vector<hipGraphNode_t>,2> phase_nodes;
                require(hipGraphGetNodes(current,nodes.data(),&count)==hipSuccess,"inspect native ordinary phase nodes");
                for(const auto node:nodes)
                {
                    hipGraphNodeType type;
                    require(hipGraphNodeGetType(node,&type)==hipSuccess,"inspect ordinary node type");
                    if(type==hipGraphNodeTypeGraph)
                    {
                        hipGraph_t child=nullptr;
                        require(hipGraphChildGraphNodeGetGraph(node,&child)==hipSuccess&&child,"inspect ordinary phase child");
                        self(self,child);
                    }
                    else if(type==hipGraphNodeTypeKernel)
                    {
                        hipKernelNodeParams params{};
                        require(hipGraphKernelNodeGetParams(node,&params)==hipSuccess,"inspect ordinary phase launch");
                        for(size_t phase=0;phase<expected_ordinary_functions.size();++phase)
                            if(params.func==expected_ordinary_functions[phase])
                            {
                                ++found[phase];phase_nodes[phase].push_back(node);
                                hipKernelNodeAttrValue attribute{};
                                require(hipGraphKernelNodeGetAttribute(node,hipKernelNodeAttributeCooperative,&attribute)==hipSuccess&&
                                    !attribute.cooperative,"ordinary phase must retain cooperative=false");
                                const auto grid=phase==0
                                    ?expected_ordinary_gate_grid
                                    :(uint64_t(kH)+31)/32;
                                require(params.gridDim.x==grid&&params.gridDim.y==1&&params.gridDim.z==1&&
                                    params.blockDim.x==256&&params.blockDim.y==1&&params.blockDim.z==1&&params.sharedMemBytes==expected_ordinary_dynamic_bytes[phase],
                                    "ordinary phases retain exact useful output tiles without resident-grid clamping");
                            }
                    }
                }
                require(phase_nodes[0].size()==phase_nodes[1].size(),
                    "each imported child contains complete ordinary operations, never a split phase pair");
                if(!phase_nodes[0].empty())
                {
                    size_t edges=0;
                    require(hipGraphGetEdges(current,nullptr,nullptr,&edges)==hipSuccess,"inspect ordinary phase ordering edges");
                    std::vector<hipGraphNode_t> from(edges),to(edges);
                    require(hipGraphGetEdges(current,from.data(),to.data(),&edges)==hipSuccess,"read ordinary phase ordering edges");
                    std::map<hipGraphNode_t,std::vector<hipGraphNode_t>> descendants;
                    for(const auto& phase:phase_nodes)for(const auto source:phase)
                    {
                        std::vector<hipGraphNode_t> frontier{source};
                        auto& visited=descendants[source];
                        for(size_t next=0;next<frontier.size();++next)
                        {
                            const auto current_node=frontier[next];
                            if(std::find(visited.begin(),visited.end(),current_node)!=visited.end())continue;
                            visited.push_back(current_node);
                            for(size_t edge=0;edge<edges;++edge)if(from[edge]==current_node)frontier.push_back(to[edge]);
                        }
                    }
                    const auto reaches=[&](hipGraphNode_t source,hipGraphNode_t target) {
                        const auto& visited=descendants.at(source);
                        return std::find(visited.begin(),visited.end(),target)!=visited.end();
                    };
                    // Imported timelines may flatten all32 leaves. Establish
                    // graph order rather than trusting enumeration or checking
                    // only the final pair. Complete operations are serial here.
                    for(auto& phase:phase_nodes)
                    {
                        for(size_t left=0;left<phase.size();++left)
                            for(size_t right=left+1;right<phase.size();++right)
                                require(reaches(phase[left],phase[right])!=reaches(phase[right],phase[left]),
                                    "every repeated ordinary phase has an unambiguous acyclic graph order");
                        std::sort(phase.begin(),phase.end(),[&](auto left,auto right){return left!=right&&reaches(left,right);});
                    }
                    for(size_t operation=0;operation<phase_nodes[0].size();++operation)
                    {
                        require(reaches(phase_nodes[0][operation],phase_nodes[1][operation]),
                            "each ordinary gate/up publication precedes its matching down-phase reads");
                        if(operation)
                            require(reaches(phase_nodes[1][operation-1],phase_nodes[0][operation]),
                                "each next ordinary operation follows the prior complete operation before scratch reuse");
                    }
                }
            };
            inspect(inspect,graph);
            require(found[0]==expected_instances&&found[1]==expected_instances,"each exact ordinary phase appears once per operation in retained native graph");
        };
        if(ordinary_phases)require_ordinary_graph(candidate_graph);
        if(native_layout_router)require_direct_graph(candidate_graph);
        const auto require_epoch_edges=[&](HIPGraphCapture& capture) {
            if(!published)return;
            const auto graph=capture.graph();
            size_t count=0;
            require(hipGraphGetNodes(graph,nullptr,&count)==hipSuccess,"enumerate actual epoch-bound graph");
            std::vector<hipGraphNode_t> nodes(count);
            require(hipGraphGetNodes(graph,nodes.data(),&count)==hipSuccess,"read actual epoch-bound graph");
            hipGraphNode_t acquire=nullptr,release=nullptr,body=nullptr;
            for(const auto node:nodes) {
                hipGraphNodeType type;
                require(hipGraphNodeGetType(node,&type)==hipSuccess,"inspect actual epoch node type");
                if(type!=hipGraphNodeTypeKernel)continue;
                hipKernelNodeParams parameters{};
                require(hipGraphKernelNodeGetParams(node,&parameters)==hipSuccess,"inspect epoch kernel parameters");
                // The existing inventory owns exact kernel names; find the
                // matching native identity without guessing mangled symbols.
                const auto info=std::find_if(candidate_nodes.begin(),candidate_nodes.end(),[&](const auto& n){
                    return n.function_identity==reinterpret_cast<std::uintptr_t>(parameters.func);});
                if(info==candidate_nodes.end())continue;
                if(info->name.find("acquireEpochKernel")!=std::string::npos)acquire=node;
                if(info->name.find("releaseEpochKernel")!=std::string::npos)release=node;
                if(parameters.func==expected_function)body=node;
            }
            require(acquire&&release&&body,"actual captured epoch acquire/body/release kernels");
            size_t edges=0;
            require(hipGraphGetEdges(graph,nullptr,nullptr,&edges)==hipSuccess,"count actual epoch graph edges");
            std::vector<hipGraphNode_t> from(edges),to(edges);
            require(hipGraphGetEdges(graph,from.data(),to.data(),&edges)==hipSuccess,"read actual epoch graph edges");
            const auto reaches=[&](hipGraphNode_t first,hipGraphNode_t last) {
                std::vector<hipGraphNode_t> work{first},seen;
                for(size_t index=0;index<work.size();++index) {
                    const auto current=work[index];
                    if(current==last)return true;
                    if(std::find(seen.begin(),seen.end(),current)!=seen.end())continue;
                    seen.push_back(current);
                    for(size_t edge=0;edge<edges;++edge)if(from[edge]==current)work.push_back(to[edge]);
                }
                return false;
            };
            require(reaches(acquire,body)&&reaches(body,release),"native dependency path retains exact epoch around complete FFN");
        };
        require_epoch_edges(*candidate_native);
        const auto original_node_count=original->capture->nodeCount(),candidate_node_count=candidate->capture->nodeCount();
        /** @brief Retire imported work on its actual scheduler stream before the cloned graph retires. */
        struct ImportedReplay final
        {
            IWorkerGPUContext& worker;ILocalTPContext& tp;void* stream;
            std::unique_ptr<IGPUGraphCapture> capture;
            std::unique_ptr<ScopedGPUCompletionEvent> completion;
            int exceptions=std::uncaught_exceptions();
            /** @brief Fixture-only terminal join, including collective failure paths. */
            ~ImportedReplay()
            {
                if(std::uncaught_exceptions()>exceptions)tp.requestAbort();
                if(completion&&!completion->join())std::terminate();
            }
        };
        std::unique_ptr<ImportedReplay> imported;
        if(tile_completion||ordinary_phases||native_layout_router)
        {
            auto* execution_stream=worker.getOrCreateAuxiliaryStream("gateup_completion_imported_parent");
            require(execution_stream!=stream,"first counter consumer uses a distinct explicit parent stream");
            imported=std::make_unique<ImportedReplay>(worker,tp,execution_stream,worker.createGraphCapture(execution_stream));
            imported->completion=std::make_unique<ScopedGPUCompletionEvent>(backend,device,execution_stream);
            require(bool(imported->capture),"cold imported candidate native parent");
            const GPUOrderedTimelineStep leaf{.name="original local gate/up completion leaf",.capture=candidate->capture.get()};
            require(imported->capture->buildOrderedTimelineTransaction(std::span(&leaf,1)),
                "import the actual original-source completion leaf without executing it");
            require(imported->capture->instantiate(),"instantiate retained parent on a different scheduler stream");
            std::vector<GPUGraphKernelNodeInfo> imported_nodes;
            require(imported->capture->inspectKernelNodes(imported_nodes,&error)&&imported_nodes.size()==candidate_nodes.size(),
                "native import preserves every original candidate kernel exactly once");
            if(published) {
                auto* native_import=dynamic_cast<HIPGraphCapture*>(imported->capture.get());
                require(native_import!=nullptr,"promoted imported parent has native HIP ownership");
                require_epoch_edges(*native_import);
            }
            if(ordinary_phases) {
                auto* native_import=dynamic_cast<HIPGraphCapture*>(imported->capture.get());
                require(native_import!=nullptr,"ordinary imported parent has native HIP ownership");
                require_ordinary_graph(native_import->graph());
            }
            if(native_layout_router) {
                auto* direct=dynamic_cast<HIPGraphCapture*>(imported->capture.get());
                require(direct!=nullptr,"direct imported parent retains HIP ownership");
                require_direct_graph(direct->graph());
            }
        }
        std::array<std::vector<float>,6> expected,actual;
        const auto original_planes=planes(*original),candidate_planes=planes(*candidate);
        for(std::size_t plane=0;plane<expected.size();++plane)
        {
            require(original_planes[plane]->numel()==candidate_planes[plane]->numel(),"same original public FFN plane extent");
            expected[plane].resize(original_planes[plane]->numel());actual[plane].resize(expected[plane].size());
        }
        const auto completion_bytes=down_completion?NativeDownCompletionPlan::make(kH,kTopK,1).bytes():
            NativeTileCompletionPlan::make(routed_columns,kTopK,1).bytes();
        std::vector<std::uint32_t> completion_values(completion_bytes/sizeof(std::uint32_t));
        std::array<std::uint32_t,NativeLayoutBarrierLayout::words> barrier_values{};
        std::vector<float> input(kH);
        std::array<std::vector<float>,2> native_indices,native_probabilities,native_logits;
        if(native_layout_router)for(unsigned side=0;side<2;++side) {
            native_indices[side].resize(kTopK);native_probabilities[side].resize(kTopK);native_logits[side].resize(shape.experts);
        }
        const std::array live_history{1,0,1,1,0,1};
        constexpr std::array plane_names{"route contributions","routed output","shared output","combined output","shared gate","shared up"};
        const std::array scratch_names{MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
            MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,MoEWorkspaceBuffers::DOWN_PARTIALS,
            MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES};
        const size_t structural_values=size_t(kTopK)*routed_columns+candidate->gate_scratch->numel();
        const std::array<size_t,7> structural_live_bytes{native_layout_router?0u:size_t(kH),native_layout_router?0u:size_t(kH/32)*sizeof(float),
                structural_values*sizeof(float),native_layout_router?0u:structural_values+structural_values/32*8,
                native_layout_router?sizeof(float):size_t(kH/32)*sizeof(int32_t),0,0};
        std::array<std::vector<uint8_t>,7> scratch_guards;
        if(partition_local||ordinary_phases||native_layout_router)
            for(size_t bank=0;bank<scratch_names.size();++bank)
            {
                scratch_guards[bank].resize(candidate->workspace->getBufferSize(scratch_names[bank]));
                require(structural_live_bytes[bank]<=scratch_guards[bank].size(),"structural scratch exact live prefix fits its original named bank");
            }
        bool executed_promoted=false,nonzero_promoted=false,mixed_packet_executed=false;
        for(int replay=0;replay<20;++replay)
        {
            const int live=live_history[replay%live_history.size()];
            // Sign changes deliberately change router selection and reuse the
            // same captured geometry; small fractions exercise Q8 rounding.
            for(int column=0;column<kH;++column)
                input[column]=column==0 ? (replay%3==0 ? -1.f : 1.f)+float(replay%7)/32.f
                    : float((column*13+replay*7)%37-18)/float(64+(replay%4)*32);
            if(replay==7 || replay==13)
                for(auto *lane:{original.get(),candidate.get()})
                    for(const auto &name:lane->graph.getExecutionOrder())
                        lane->graph.getNode(name)->stage->resetSessionStatePreservingCapturedReplay();
            require(signatures(*original)==original_signature && signatures(*candidate)==candidate_signature,
                "request reset preserves complete original/resident stage signatures");
            const auto run=[&](Lane &lane,std::array<std::vector<float>,6> &observed) {
                const bool imported_run=imported && &lane==candidate.get();
                void* const operation_stream=imported_run?imported->stream:stream;
                auto& operation=imported_run?*imported->capture:*lane.capture;
                for(auto *tensor:planes(lane))
                    require(backend->memset(tensor->gpu_data_ptr(),0xa5+replay%8,tensor->size_bytes(),device.ordinal,operation_stream),
                        "dirty every original public FFN plane");
                if(native_layout_router) {
                    require(backend->memset(lane.indices->gpu_data_ptr(),0xa5+replay%8,lane.indices->size_bytes(),device.ordinal,operation_stream),"dirty original router IDs");
                    require(backend->memset(lane.weights->gpu_data_ptr(),0xa5+replay%8,lane.weights->size_bytes(),device.ordinal,operation_stream),"dirty original router weights");
                    require(backend->memset(lane.workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS),0xa5+replay%8,size_t(shape.experts)*sizeof(float),device.ordinal,operation_stream),"dirty original router probabilities");
                }
                for(const auto *name:scratch_names)
                {
                    auto *pointer=lane.workspace->getBuffer(name);const auto bytes=lane.workspace->getBufferSize(name);
                    require(pointer && bytes,"resident replay original named scratch admission");
                    require(backend->memset(pointer,0xc1+replay%16,bytes,device.ordinal,operation_stream),"dirty original reused FFN scratch");
                }
                require(backend->hostToDeviceOnStream(lane.hidden->gpu_data_ptr(),input.data(),input.size()*sizeof(float),
                    device.ordinal,operation_stream),"publish varied original FFN input");
                require(backend->hostToDeviceOnStream(lane.active->gpu_data_ptr(),&live,sizeof(live),device.ordinal,operation_stream),
                    "publish sole device-owned ordinary M1 count");
                require(backend->memset(lane.receipt->gpu_data_ptr(),0,lane.receipt->size_bytes(),device.ordinal,operation_stream),
                    "reset actual compact final-sum byte receipt");
                const int sideband=member+1;
                if(lane.sideband)
                    require(backend->hostToDeviceOnStream(lane.sideband->gpu_data_ptr(),&sideband,sizeof(sideband),device.ordinal,operation_stream),
                        "reset same final native sum sideband");
                rendezvous.arrive_and_wait();
                if(lane.serial_prefill_scratch)
                    require(lane.serial_prefill_scratch->launchOnStream(operation_stream),
                        "replay alternate original prefill scratch before retained completion decode");
                require(operation.launch(),"replay original source local FFN retained executable");
                if(native_layout_router) {
                    const unsigned side=&lane==candidate.get()?1:0;
                    require(backend->deviceToHostOnStream(native_indices[side].data(),lane.indices->gpu_data_ptr(),kTopK*sizeof(float),device.ordinal,operation_stream),"observe original/native router IDs");
                    require(backend->deviceToHostOnStream(native_probabilities[side].data(),lane.weights->gpu_data_ptr(),kTopK*sizeof(float),device.ordinal,operation_stream),"observe original/native router weights");
                    require(backend->deviceToHostOnStream(native_logits[side].data(),lane.workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS),shape.experts*sizeof(float),device.ordinal,operation_stream),"observe original/native router probabilities");
                }
                if(native_layout_router && &lane==candidate.get())
                    require(backend->deviceToHostOnStream(barrier_values.data(),
                        candidate_barrier_state,sizeof(barrier_values),
                        device.ordinal,operation_stream),"observe all instance-owned barrier slots");
                const auto outputs=planes(lane);
                for(std::size_t plane=0;plane<outputs.size();++plane)
                    require(backend->deviceToHostOnStream(observed[plane].data(),outputs[plane]->gpu_data_ptr(),
                        observed[plane].size()*sizeof(float),device.ordinal,operation_stream),"observe original public FFN plane");
                unsigned long long sent=0;int sideband_sum=0;
                require(backend->deviceToHostOnStream(&sent,lane.receipt->gpu_data_ptr(),sizeof(sent),device.ordinal,operation_stream),
                    "observe actual compact collective useful bytes");
                if(lane.sideband)
                    require(backend->deviceToHostOnStream(&sideband_sum,lane.sideband->gpu_data_ptr(),sizeof(sideband_sum),device.ordinal,operation_stream),
                        "observe original final collective sideband");
                if(tile_completion && &lane==candidate.get())
                    require(backend->deviceToHostOnStream(completion_values.data(),
                        lane.workspace->getBuffer(counter_name),completion_bytes,device.ordinal,operation_stream),
                        "observe exact last-producer protocol counters outside execution");
                if((partition_local||ordinary_phases||native_layout_router) && &lane==candidate.get())
                    for(size_t bank=0;bank<scratch_names.size();++bank)
                        require(backend->deviceToHostOnStream(scratch_guards[bank].data(),lane.workspace->getBuffer(scratch_names[bank]),
                            scratch_guards[bank].size(),device.ordinal,operation_stream),"observe structural untouched scratch suffix");
                require(imported_run?imported->completion->join():completion.join(),"join terminal resident FFN fixture observations");
                if((partition_local||ordinary_phases||native_layout_router) && &lane==candidate.get())
                    for(size_t bank=0;bank<scratch_names.size();++bank)
                        require(std::all_of(scratch_guards[bank].begin()+structural_live_bytes[bank],scratch_guards[bank].end(),
                            [replay](uint8_t value){return value==uint8_t(0xc1+replay%16);}),
                            "structural scratch suffix remains untouched: "+std::string(scratch_names[bank]));
                if(tile_completion && &lane==candidate.get())
                    require(std::all_of(completion_values.begin(),completion_values.end(),[](auto value){return value==0;}),
                        "every full/partial/empty/reset tile retires and resets without an extra native clearing node");
                if(native_layout_router && &lane==candidate.get())
                    for(unsigned phase=0;phase<NativeLayoutBarrierLayout::phases;++phase)
                        require(barrier_values[2*phase]==0 && barrier_values[2*phase+1]==unsigned((replay+1)&1),
                            "all full/empty/reset/imported joins retire and toggle exactly once without host reset");
                require(sent==static_cast<unsigned long long>(live)*kH*sizeof(float)*(tp.degree()-1),
                    "explicit final collective sends exactly actual live H data");
                if(lane.sideband)
                    require(sideband_sum==tp.degree()*(tp.degree()+1)/2,"same explicit collective preserves sidebands on empty/full replay");
                rendezvous.arrive_and_wait();
            };
            run(*original,expected);run(*candidate,actual);
            if(published && live) {
                for(size_t slot=0;slot<native_indices[1].size();++slot)
                    if(std::find(published->promoted_experts.begin(),published->promoted_experts.end(),int(native_indices[1][slot]))!=published->promoted_experts.end()) {
                        executed_promoted=true;
                        for(int column=0;column<kH;++column)
                            nonzero_promoted=nonzero_promoted||actual[0][slot*size_t(kH)+column]!=0.f;
                    }
            }
            if(published&&live) {
                // EB4 processes consecutive router slots. Require both actual
                // physical representations in one observed packet, with nonzero
                // public contributions, rather than inferring use from residency.
                for(size_t first=0;first<native_indices[1].size();first+=4) {
                    bool canonical=false,promoted=false;
                    for(size_t slot=first;slot<std::min(first+4,native_indices[1].size());++slot) {
                        const int expert=int(native_indices[1][slot]);
                        bool nonzero=false;
                        for(int column=0;column<kH;++column)
                            nonzero=nonzero||actual[0][slot*size_t(kH)+column]!=0.f;
                        canonical=canonical||(nonzero&&std::find(published->canonical_experts.begin(),published->canonical_experts.end(),expert)!=published->canonical_experts.end());
                        promoted=promoted||(nonzero&&std::find(published->promoted_experts.begin(),published->promoted_experts.end(),expert)!=published->promoted_experts.end());
                    }
                    mixed_packet_executed=mixed_packet_executed||(canonical&&promoted);
                }
            }
            if(native_layout_router) {
                require(std::memcmp(native_indices[0].data(),native_indices[1].data(),kTopK*sizeof(float))==0,"native-layout original router ID bytes");
                require(std::memcmp(native_probabilities[0].data(),native_probabilities[1].data(),kTopK*sizeof(float))==0,"native-layout original router weight bytes");
                require(std::memcmp(native_logits[0].data(),native_logits[1].data(),shape.experts*sizeof(float))==0,"native-layout original router probability bytes");
            }
            for(std::size_t plane=0;plane<expected.size();++plane)
                for(std::size_t element=0;element<expected[plane].size();++element)
                    if(!std::isfinite(actual[plane][element]) ||
                        std::bit_cast<uint32_t>(actual[plane][element])!=std::bit_cast<uint32_t>(expected[plane][element]))
                        throw std::runtime_error("Resident FFN original bytes differ: member="+std::to_string(member)+
                            " replay="+std::to_string(replay)+" live="+std::to_string(live)+" plane="+plane_names[plane]+
                            " element="+std::to_string(element)+" resident="+std::to_string(actual[plane][element])+
                            " original="+std::to_string(expected[plane][element]));
            require(original_native->graph()==original_graph && original_native->executable()==original_executable &&
                candidate_native->graph()==candidate_graph && candidate_native->executable()==candidate_executable &&
                original->capture->nodeCount()==original_node_count && candidate->capture->nodeCount()==candidate_node_count,
                "twenty histories retain identical native graph/executable handles and topology");
        }
        if(observation!=ResidentFFNObservation::Replay)
        {
#include "MoEPartitionLocalFFNTiming.inc"
        }
        rendezvous.arrive_and_wait();
        require(!published||(executed_promoted&&nonzero_promoted),"retained NativeLayout actually executed a nonzero promoted physical23 route");
        require(!published||mixed_packet_executed,"retained NativeLayout executed nonzero physical7 and23 routes in the same original EB4 packet");
        if(tile_completion||ordinary_phases||native_layout_router)
        {
            // A native executable borrows immutable backend identity. Changing
            // only the backend stream must fail through stage validation before
            // a graph launch; restoring the original stream restores admission.
            auto* alternate=worker.getOrCreateAuxiliaryStream("gateup_completion_rebind_adversary");
            require(alternate!=stream,"stream adversary owns a different original worker stream");
            candidate->owner->kernel->setGPUStream(alternate);
            bool stream_rejected=false;
            try {(void)signatures(*candidate);}catch(const std::logic_error&){stream_rejected=true;}
            candidate->owner->kernel->setGPUStream(stream);
            require(stream_rejected&&signatures(*candidate)==candidate_signature,
                "last-producer signature rejects stream rebinding before retained submission");
            // Keep the actual descriptor owner and all its leases alive while
            // substituting an unprepared owner of the same backend/device.
            // Matching type and geometry cannot mint the original publication.
            if(ordinary_phases)
            {
                const auto rejects_publisher=[&](DeviceMoELayerRuntime* runtime,int gate,int down) {
                    bool rejected=false;
                    try {(void)candidate->owner->kernel->preparePairedRouterDecodeDescriptors(runtime,gate,down,kTopK);}
                    catch(const std::logic_error&){rejected=true;}
                    require(rejected,"ordinary FFN rejects a missing or mismatched original router publication");
                    require(signatures(*candidate)==candidate_signature,"rejected publisher change leaves the original graph seal intact");
                };
                auto* runtime=candidate->runtime->deviceLayerState(0);
                rejects_publisher(runtime,-1,-1);
                rejects_publisher(original->runtime->deviceLayerState(0),0,0);
                int foreign_tables_rejected=0;
                for(int table=0;table<2;++table)
                {
                    bool rejected=false;
                    try {(void)candidate->owner->kernel->preparePairedRouterDecodeDescriptors(runtime,table,table,kTopK);}
                    catch(const std::logic_error&){rejected=true;}
                    foreign_tables_rejected+=rejected;
                    require(signatures(*candidate)==candidate_signature,
                        "table probe preserves exact paired publication identity");
                }
                require(foreign_tables_rejected>0,"at least one foreign prepared table pair is rejected");
            }
            auto replacement=llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            require(bool(replacement),"admit separate descriptor-owner adversary");
            replacement->setGPUStream(stream);
            if(ordinary_phases)
            {
                bool missing_rejected=false;
                try {(void)replacement->preparePairedRouterDecodeDescriptors(candidate->runtime->deviceLayerState(0),0,0,kTopK);}
                catch(const std::logic_error&){missing_rejected=true;}
                require(missing_rejected,"unprepared same-backend owner cannot mint the required router publication");
            }
            auto original_owner=std::move(candidate->owner->kernel);
            candidate->owner->kernel=std::move(replacement);
            bool owner_rejected=false;
            try {(void)signatures(*candidate);}catch(const std::logic_error&){owner_rejected=true;}
            replacement=std::move(candidate->owner->kernel);
            candidate->owner->kernel=std::move(original_owner);
            require(owner_rejected&&signatures(*candidate)==candidate_signature,
                "last-producer signature rejects descriptor owner substitution before retained submission");
        }
        // No native work remains outstanding. Keep the graph and its leases
        // alive, withdraw only the SAME manager's named storage authority, and
        // require graph identity validation to reject before another launch.
        const auto workspace_id=candidate->workspace->id();
        candidate->workspace->release();
        require(candidate->workspace->id()==workspace_id && !candidate->workspace->isAllocated(),
            "lifetime adversary retains manager identity while releasing names");
        bool rejected=false;
        try {(void)signatures(*candidate);}catch(const std::logic_error &){rejected=true;}
        require(rejected,"resident signature must reject same-manager storage release before retained submission");
        require(candidate_native->graph()==candidate_graph && candidate_native->executable()==candidate_executable,
            "storage release test retains original native handles until normal graph retirement");
        rendezvous.arrive_and_wait();
    });
#else
    (void)device;(void)member;(void)tp;(void)rendezvous;(void)banks;(void)original_member;
    (void)shared;(void)shared_partition;(void)store;(void)observation;(void)source_name;(void)replay_candidate;(void)local_program;(void)shape;(void)native_layout_router;(void)published;
    throw std::invalid_argument("Resident FFN replay requires compiled ROCm production kernels");
#endif
}
