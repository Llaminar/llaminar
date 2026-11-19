/**
 * @file MoEPartitionLocalFFNTiming.inc
 * @brief Retained complete FFN event timing and same-source observability control.
 * @details Included in the established original-source replay owner after its
 * six-plane byte gate. PairedEvents imports32 complete router/localFFN/collective
 * operations into a Disabled child, then times that child as ONE parent step.
 * Exactly one event pair brackets the whole32-operation group; samples report
 * parent latency and its amortized per-operation value, never32 fabricated
 * independent intervals. Only this economy mode omits the fixture-only INT32
 * sideband; the production FP32 final collective and receipt remain explicit.
 * No allocation, copy, reset or rendezvous occurs between measured operations.
 * Each input record visits both ABBA/BAAB orders equally over32 measured
 * rounds (64 whole-parent observations per side), independent of record parity.
 * Each measured output is inspected directly after its parent; verification
 * never relaunches a graph.
 * RepeatedParent additionally imports the same32 operations with instrumentation
 * disabled. This distinguishes event-node submission effects from native work:
 * both policies retain the same numerical/receipt oracle, with no speed assertion.
 * Host complete-parent latency includes launch through the existing terminal
 * completion event, excluding setup, copies, resets and participant rendezvous.
 */
            if(native_layout_router)
                std::osyncstream(std::cout)<<"NATIVE_LAYOUT_ROUTER_FFN_SCOPE,"<<device.toString()<<','<<member
                    <<",router_and_local_ffn,original_prepared_layout,explicit_final_collective,HC_prefix_excluded,"
                    <<(observation==ResidentFFNObservation::PairedEvents?"production_payload_only,whole_parent_events_v2":"diagnostic_sideband_retained,observability_control_v1")<<'\n';
            constexpr size_t operations=32;
            constexpr size_t paired_records=4,measured_rounds=32,retained_samples=2*measured_rounds;
            static_assert(measured_rounds%(2*paired_records)==0);
            const bool observability_control=observation==ResidentFFNObservation::RepeatedParent;
            const bool parent_events=observation==ResidentFFNObservation::PairedEvents;
            const size_t instrumentation_count=observability_control?2:1;
            // Both immutable parents are built once before replay and borrow
            // the same original lane; they never execute concurrently.
            // Parents retire before their borrowed children; the terminal event
            // below is destroyed first and joins work before either graph owner.
            std::array<std::unique_ptr<IGPUGraphCapture>,2> economy_children;
            std::array<std::pair<hipGraph_t,hipGraphExec_t>,2> economy_child_identity{};
            std::array<std::array<std::unique_ptr<IGPUGraphCapture>,2>,2> batches;
            std::array<std::array<std::pair<hipGraph_t,hipGraphExec_t>,2>,2> batch_identity{};
            std::array<std::array<size_t,2>,2> batch_node_counts{};
            std::array<std::vector<std::array<uint64_t,9>>,2> instrumented_kernel_inventory;
            ScopedGPUCompletionEvent batch_retirement(backend,device,stream);
            std::array<Lane*,2> lanes{original.get(),candidate.get()};
            for(size_t instrumentation=0;instrumentation<instrumentation_count;++instrumentation)
            for(size_t side=0;side<lanes.size();++side)
            {
                auto& batch=batches[instrumentation][side];
                batch=worker.createGraphCapture(stream);
                require(bool(batch),"admit complete-operation event parent");
                std::vector<GPUOrderedTimelineStep> steps;
                steps.reserve(operations);
                std::array<std::string,operations> names;
                for(size_t operation=0;operation<operations;++operation)
                {
                    names[operation]="complete_original_ffn_"+std::to_string(operation);
                    steps.push_back({.name=names[operation].c_str(),.capture=lanes[side]->capture.get()});
                }
                if(parent_events)
                {
                    require(!lanes[side]->sideband,"economy lane omits the fixture-only sideband declaration and allocation");
                    auto& child=economy_children[side];
                    child=worker.createGraphCapture(stream);
                    require(bool(child)&&child->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::Disabled),
                        "import32 complete production-payload operations without internal timing markers");
                    // Import the retained definition directly; only the outer
                    // measured parent needs an executable or native submission.
                    auto* native_child=dynamic_cast<HIPGraphCapture*>(child.get());
                    require(native_child&&native_child->graph()&&!native_child->executable(),"economy child retains only its native graph definition");
                    economy_child_identity[side]={native_child->graph(),native_child->executable()};
                    require(child->consumeOrderedTimelineTiming().state==GPUOrderedTimelineTimingState::Disabled,
                        "economy child contributes no internal event intervals");
                    const GPUOrderedTimelineStep group{.name="whole32_operation_ffn_parent",.capture=child.get()};
                    require(batch->buildOrderedTimelineTransaction(std::span(&group,1),GPUOrderedTimelineInstrumentation::PerStepEvents)&&
                        batch->nodeCount()==child->nodeCount()+2,
                        "exactly one event pair brackets the complete32-operation parent");
                }
                else
                    require(batch->buildOrderedTimelineTransaction(steps,instrumentation==0?GPUOrderedTimelineInstrumentation::PerStepEvents:GPUOrderedTimelineInstrumentation::Disabled),
                        "import same32 complete native operations with declared instrumentation");
                require(batch->instantiate(),"retain complete-operation measured native parent");
                std::vector<GPUGraphKernelNodeInfo> measured_nodes;
                std::string measured_error;
                require(batch->inspectKernelNodes(measured_nodes,&measured_error)&&
                    measured_nodes.size()==operations*(side==0?original_nodes.size():candidate_nodes.size()),
                    "measured parent retains exactly32 complete native operations: "+measured_error);
                std::vector<std::array<uint64_t,9>> inventory;
                for(const auto& node:measured_nodes)
                    inventory.push_back({node.function_identity,node.grid_x,node.grid_y,node.grid_z,
                        node.block_x,node.block_y,node.block_z,node.dynamic_shared_memory_bytes,node.static_shared_memory_bytes});
                std::sort(inventory.begin(),inventory.end());
                if(instrumentation==0)instrumented_kernel_inventory[side]=std::move(inventory);
                else require(inventory==instrumented_kernel_inventory[side],
                    "disabled parent retains identical native functions, work grids and shared-memory extents");
                auto* retained_native=dynamic_cast<HIPGraphCapture*>(batch.get());
                require(retained_native&&retained_native->graph()&&retained_native->executable(),
                    "observability parent retains actual native graph/executable");
                batch_identity[instrumentation][side]={retained_native->graph(),retained_native->executable()};
                batch_node_counts[instrumentation][side]=batch->nodeCount();
                if(instrumentation==1)
                    require(batch->consumeOrderedTimelineTiming().state==GPUOrderedTimelineTimingState::Disabled,
                        "uninstrumented parent contains no timeline timing state");
                if(native_layout_router&&side==1)require_direct_graph(retained_native->graph(),operations);
                if(ordinary_phases&&side==1)
                {
                    auto* measured_native=dynamic_cast<HIPGraphCapture*>(batch.get());
                    require(measured_native!=nullptr,"ordinary measured parent retains native HIP ownership");
                    require_ordinary_graph(measured_native->graph(),operations);
                }
            }
            for(size_t side=0;side<lanes.size();++side)
                for(const auto& node:side==0?original_nodes:candidate_nodes)
                    std::osyncstream(std::cout)<<"PARTITION_LOCAL_FFN_RESOURCE,"<<device.toString()<<','<<member<<','<<source_name
                        <<",generated_original_source,1,"<<kH<<','<<routed_columns<<','<<candidate->gate_scratch->numel()
                        <<','<<kTopK<<','<<tp.degree()<<','<<(side==0?"staged":"candidate")<<','<<std::quoted(node.name)
                        <<','<<node.grid_x<<','<<node.grid_y<<','<<node.grid_z<<','<<node.block_x<<','<<node.block_y<<','<<node.block_z
                        <<','<<node.registers_per_thread<<','<<node.static_shared_memory_bytes<<','<<node.dynamic_shared_memory_bytes
                        <<','<<node.local_memory_bytes_per_thread<<','<<node.max_active_blocks_per_sm<<'\n';
            std::array<std::vector<int>,paired_records> selected_experts;
            std::vector<float> selected_indices(kTopK);
            std::vector<float> native_expected_indices(kTopK),native_expected_weights(kTopK),native_expected_logits(shape.experts);
            std::array<std::vector<float>,2> samples,parent_samples;
            for(auto& values:samples)values.reserve(parent_events?retained_samples:retained_samples*operations);
            if(parent_events)for(auto& values:parent_samples)values.reserve(retained_samples);
            const auto prepare_sample=[&](Lane& lane,int record) {
                for(int column=0;column<kH;++column)
                    input[column]=column==0?1.125f+float(record%3)/32.f:
                        float((column*13+record*7)%37-18)/float(128+(record%3)*32);
                for(auto* tensor:planes(lane))
                    require(backend->memset(tensor->gpu_data_ptr(),0xa5,tensor->size_bytes(),device.ordinal,stream),
                        "dirty complete-operation measured public planes outside event parent");
                for(const auto* name:scratch_names)
                    require(backend->memset(lane.workspace->getBuffer(name),0xc7,lane.workspace->getBufferSize(name),
                        device.ordinal,stream),"dirty measured scratch suffixes outside event parent");
                if(native_layout_router) {
                    require(backend->memset(lane.indices->gpu_data_ptr(),0xa5,lane.indices->size_bytes(),device.ordinal,stream),"dirty measured router ID output outside event parent");
                    require(backend->memset(lane.weights->gpu_data_ptr(),0xa5,lane.weights->size_bytes(),device.ordinal,stream),"dirty measured router weight output outside event parent");
                    require(backend->memset(lane.workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS),0xa5,size_t(shape.experts)*sizeof(float),device.ordinal,stream),"dirty measured router probabilities outside event parent");
                }
                const int live=1;
                // Zero is a legitimate sideband payload and remains zero over
                //32 exact allreduces. Nonzero/empty sidebands already passed the
                // independent20-history gate; this avoids exponential INT32
                // amplification when importing the SAME complete operation.
                const int sideband=0;
                require(backend->hostToDeviceOnStream(lane.hidden->gpu_data_ptr(),input.data(),input.size()*sizeof(float),
                    device.ordinal,stream),"publish paired complete-operation input record");
                require(backend->hostToDeviceOnStream(lane.active->gpu_data_ptr(),&live,sizeof(live),device.ordinal,stream),
                    "publish paired full live row");
                if(lane.sideband)
                    require(backend->hostToDeviceOnStream(lane.sideband->gpu_data_ptr(),&sideband,sizeof(sideband),device.ordinal,stream),
                        "publish stable original measured collective sideband");
                require(backend->memset(lane.receipt->gpu_data_ptr(),0,lane.receipt->size_bytes(),device.ordinal,stream),
                    "reset measured original compact receipt outside event parent");
                require(completion.join(),"terminal event retires sample setup outside measured parent");
            };
            const auto observe_sample=[&](Lane& lane,std::array<std::vector<float>,6>& output,size_t completed_operations) {
                const auto output_planes=planes(lane);
                for(size_t plane=0;plane<output.size();++plane)
                    require(backend->deviceToHostOnStream(output[plane].data(),output_planes[plane]->gpu_data_ptr(),
                        output[plane].size()*sizeof(float),device.ordinal,stream),"observe actual measured public FFN output without relaunch");
                unsigned long long sent=0;int sideband=1;
                require(backend->deviceToHostOnStream(&sent,lane.receipt->gpu_data_ptr(),sizeof(sent),device.ordinal,stream),
                    "observe measured native compact receipt");
                if(lane.sideband)
                    require(backend->deviceToHostOnStream(&sideband,lane.sideband->gpu_data_ptr(),sizeof(sideband),device.ordinal,stream),
                        "observe measured native sideband");
                if((partition_local||ordinary_phases||native_layout_router) && &lane==candidate.get())
                    for(size_t bank=0;bank<scratch_names.size();++bank)
                        require(backend->deviceToHostOnStream(scratch_guards[bank].data(),lane.workspace->getBuffer(scratch_names[bank]),
                            scratch_guards[bank].size(),device.ordinal,stream),"observe measured structural scratch guard without relaunch");
                if(native_layout_router) {
                    const unsigned side=&lane==candidate.get()?1:0;
                    require(backend->deviceToHostOnStream(native_indices[side].data(),lane.indices->gpu_data_ptr(),kTopK*sizeof(float),device.ordinal,stream),"observe measured native router IDs");
                    require(backend->deviceToHostOnStream(native_probabilities[side].data(),lane.weights->gpu_data_ptr(),kTopK*sizeof(float),device.ordinal,stream),"observe measured native route weights");
                    require(backend->deviceToHostOnStream(native_logits[side].data(),lane.workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS),shape.experts*sizeof(float),device.ordinal,stream),"observe measured native probabilities");
                }
                require(completion.join(),"terminal event joins measured output/guard observation");
                const auto per_operation=static_cast<unsigned long long>(kH)*sizeof(float)*(tp.degree()-1);
                require(completed_operations>0&&(!per_operation||completed_operations<=std::numeric_limits<unsigned long long>::max()/per_operation),
                    "completed parent collective receipt extent is representable");
                const auto expected_sent=per_operation*completed_operations;
                require(sent==expected_sent,"completed parent retains exact live compact collective bytes: actual="+
                    std::to_string(sent)+", expected="+std::to_string(expected_sent)+", completed_operations="+
                    std::to_string(completed_operations)+", bytes_per_operation="+std::to_string(per_operation));
                if(lane.sideband)require(sideband==0,"measured original collective preserves stable zero sideband");
                if((partition_local||ordinary_phases||native_layout_router) && &lane==candidate.get())
                    for(size_t bank=0;bank<scratch_names.size();++bank)
                        require(std::all_of(scratch_guards[bank].begin()+structural_live_bytes[bank],scratch_guards[bank].end(),
                            [](uint8_t value){return value==0xc7;}),"measured structural unused suffix remains untouched");
            };
            /**
             * @brief Replay one already-built parent with identical source and byte oracle.
             * @param instrumentation Zero is the declared event policy; one is the old Disabled control.
             * @param side Original staged or candidate lane.
             * @param record Original paired input identifier, prepared before the bracket.
             * @param retain Whether to retain this complete-parent event observation.
             * @param round Balanced-order round identifier.
             * @param slot ABBA/BAAB position within this parent policy.
             * @details The host bracket contains only launch and the original
             * terminal event completion. It measures complete-parent latency,
             * not kernel service or fabricated per-operation device intervals.
             */
            const auto sample=[&](size_t instrumentation,int side,int record,bool retain,int round,int slot) {
                prepare_sample(*lanes[side],record);
                rendezvous.arrive_and_wait();
                auto& batch=batches[instrumentation][side];
                const auto host_begin=std::chrono::steady_clock::now();
                const bool launched=batch->launch();
                const bool completed=completion.join();
                const auto host_end=std::chrono::steady_clock::now();
                require(launched&&completed,"terminal event retires actual retained32-operation parent");
                const auto timing=batch->consumeOrderedTimelineTiming();
                if(instrumentation==0)
                    require(timing.state==GPUOrderedTimelineTimingState::Complete&&timing.samples.size()==(parent_events?1:operations),
                        "event interval count matches the declared whole-parent or original observability scope");
                else require(timing.state==GPUOrderedTimelineTimingState::Disabled&&timing.samples.empty(),
                    "disabled parent exposes no fabricated device intervals");
                auto* retained_native=dynamic_cast<HIPGraphCapture*>(batch.get());
                require(retained_native&&batch_identity[instrumentation][side]==
                    std::pair{retained_native->graph(),retained_native->executable()}&&
                    batch->nodeCount()==batch_node_counts[instrumentation][side],
                    "observability replay preserves native graph/executable identity and topology");
                if(parent_events) {
                    auto* child=dynamic_cast<HIPGraphCapture*>(economy_children[side].get());
                    require(child&&economy_child_identity[side]==std::pair{child->graph(),child->executable()}&&
                        economy_children[side]->consumeOrderedTimelineTiming().state==GPUOrderedTimelineTimingState::Disabled,
                        "whole-parent samples preserve the original uninstrumented child identity");
                }
                if(observability_control) {
                    const auto parent_us=std::chrono::duration<double,std::micro>(host_end-host_begin).count();
                    require(std::isfinite(parent_us)&&parent_us>0,"positive finite complete-parent host latency");
                    std::osyncstream(std::cout)<<"NATIVE_LAYOUT_OBSERVABILITY_PARENT_HOST_US,"<<device.toString()<<','<<member
                        <<','<<source_name<<','<<tp.degree()<<','<<(instrumentation==0?"PerStepEvents":"Disabled")
                        <<','<<(side==0?"staged":"candidate")<<",record="<<record<<",round="<<round<<",slot="<<slot
                        <<",operations="<<operations<<",us="<<std::setprecision(12)<<parent_us
                        <<",launch_through_original_terminal_event,no_setup_copy_reset_rendezvous_inside_bracket,not_kernel_service\n";
                }
                observe_sample(*lanes[side],actual,operations);
                if(native_layout_router)require(std::memcmp(native_indices[side].data(),native_expected_indices.data(),kTopK*sizeof(float))==0&&
                    std::memcmp(native_probabilities[side].data(),native_expected_weights.data(),kTopK*sizeof(float))==0&&
                    std::memcmp(native_logits[side].data(),native_expected_logits.data(),shape.experts*sizeof(float))==0,
                    "actual measured native-layout parent retains original router outputs");
                for(size_t plane=0;plane<expected.size();++plane)
                        for(size_t element=0;element<expected[plane].size();++element)
                            require(std::isfinite(actual[plane][element])&&
                                std::bit_cast<uint32_t>(actual[plane][element])==std::bit_cast<uint32_t>(expected[plane][element]),
                                "actual measured complete-operation bytes: "+std::string(plane_names[plane]));
                if(retain&&instrumentation==0&&parent_events)
                {
                    const float parent_us=float(timing.samples.front().elapsed_ms*1000.0);
                    const float amortized_us=parent_us/float(operations);
                    require(std::isfinite(parent_us)&&parent_us>0,"positive finite whole-parent event interval");
                    parent_samples[side].push_back(parent_us);samples[side].push_back(amortized_us);
                    std::osyncstream(std::cout)<<"PARTITION_LOCAL_FFN_PARENT_EVENT_SAMPLE_V2,"<<device.toString()<<','<<member<<','<<source_name
                        <<",generated_original_source,1,"<<kH<<','<<routed_columns<<','<<candidate->gate_scratch->numel()
                        <<','<<kTopK<<','<<tp.degree()<<",round="<<round<<",slot="<<slot<<",record="<<record<<','
                        <<(side==0?"staged":"candidate")<<",operations="<<operations<<",parent_us="<<std::setprecision(12)<<parent_us
                        <<",amortized_us_per_op="<<amortized_us<<",one_event_pair,production_payload_only,no_fixture_sideband,cache_hot_within_parent\n";
                }
                else if(retain&&instrumentation==0)
                    for(size_t operation=0;operation<timing.samples.size();++operation)
                    {
                        const float us=float(timing.samples[operation].elapsed_ms*1000.0);
                        require(std::isfinite(us)&&us>0,"positive finite internal complete-operation event interval");
                        samples[side].push_back(us);
                        std::osyncstream(std::cout)<<"PARTITION_LOCAL_FFN_EVENT_SAMPLE,"<<device.toString()<<','<<member<<','<<source_name
                            <<",generated_original_source,1,"<<kH<<','<<routed_columns<<','<<candidate->gate_scratch->numel()
                            <<','<<kTopK<<','<<tp.degree()<<','<<round<<','<<slot<<','<<operation<<','
                            <<(side==0?"staged":"candidate")<<','<<std::setprecision(9)<<us<<'\n';
                    }
                require(signatures(*original)==original_signature&&signatures(*candidate)==candidate_signature,
                    "measured complete-operation owner signatures remain sealed");
                rendezvous.arrive_and_wait();
            };
            // Author an independent staged oracle for EACH input record before
            //the balanced measured order, outside all retained event intervals.
            const bool timing=observation==ResidentFFNObservation::PairedEvents;
            for(int round=timing?-5:0;round<(timing?static_cast<int>(measured_rounds):1);++round)
            {
                const auto schedule=llaminar2::test::native_quantised_dispatch::
                    pairedRecordTimingRound<paired_records>(static_cast<size_t>(round+5));
                const int record=static_cast<int>(schedule.record);
                prepare_sample(*original,record);
                rendezvous.arrive_and_wait();
                require(original->capture->launch(),"author paired input's independent staged output outside timing");
                observe_sample(*original,expected,1);
                if(native_layout_router){native_expected_indices=native_indices[0];native_expected_weights=native_probabilities[0];native_expected_logits=native_logits[0];}
                if(shape.router_source==OriginalFFNRouterSource::Dense)
                {
                    require(backend->deviceToHostOnStream(selected_indices.data(),original->indices->gpu_data_ptr(),
                        selected_indices.size()*sizeof(float),device.ordinal,stream),"observe actual original selected expert working set outside timing");
                    require(completion.join(),"terminal event joins original route-index observation");
                    auto& selected=selected_experts[record];selected.clear();
                    for(const float expert:selected_indices)
                    {
                        require(std::isfinite(expert)&&expert>=0&&expert<shape.experts&&expert==std::trunc(expert),
                            "original router publishes valid exact expert indices");
                        selected.push_back(static_cast<int>(expert));
                    }
                }
                rendezvous.arrive_and_wait();
                if(round==7||round==19)
                    for(auto* lane:lanes)
                        for(const auto& name:lane->graph.getExecutionOrder())
                            lane->graph.getNode(name)->stage->resetSessionStatePreservingCapturedReplay();
                // Alternate order on each record's next visit, not on round
                // parity (which aliases even record counts to one order).
                // The single non-economy observability control keeps its ABBA.
                const auto order=timing?schedule.sides():std::array<int,4>{0,1,1,0};
                for(size_t instrumentation=0;instrumentation<instrumentation_count;++instrumentation)
                    for(size_t slot=0;slot<order.size();++slot)
                        sample(instrumentation,order[slot],record,round>=0,round,int(slot));
            }
            if(timing&&shape.router_source==OriginalFFNRouterSource::Dense)
            {
                std::vector<int> working_set;
                for(const auto& selected:selected_experts)working_set.insert(working_set.end(),selected.begin(),selected.end());
                std::sort(working_set.begin(),working_set.end());
                working_set.erase(std::unique(working_set.begin(),working_set.end()),working_set.end());
                require(working_set.size()>size_t(kTopK),"paired input records exercise a genuine changing original expert working set");
                std::osyncstream(std::cout)<<"PARTITION_LOCAL_FFN_SOURCE_WORKING_SET,"<<device.toString()<<','<<member
                    <<",original_experts="<<shape.experts<<",records=4,unique_selected_experts="<<working_set.size()<<'\n';
            }
            if(timing)for(size_t side=0;side<samples.size();++side)
            {
                require(parent_events&&samples[side].size()==retained_samples&&parent_samples[side].size()==retained_samples,
                    "32 record/order-balanced rounds retain64 whole-parent observations per side");
                auto ordered=samples[side],ordered_parent=parent_samples[side];
                std::sort(ordered.begin(),ordered.end());std::sort(ordered_parent.begin(),ordered_parent.end());
                const float median=(ordered[ordered.size()/2-1]+ordered[ordered.size()/2])*0.5f;
                const float parent_median=(ordered_parent[ordered_parent.size()/2-1]+ordered_parent[ordered_parent.size()/2])*0.5f;
                std::osyncstream(std::cout)<<"PARTITION_LOCAL_FFN_PARENT_EVENT_MEDIAN_V2,"<<device.toString()<<','<<member<<','<<source_name
                    <<",generated_original_source,1,"<<kH<<','<<routed_columns<<','<<candidate->gate_scratch->numel()
                    <<','<<kTopK<<','<<tp.degree()<<','<<(side==0?"staged":"candidate")<<','<<ordered.size()<<','
                    <<"parent_us="<<std::setprecision(12)<<parent_median<<",amortized_us_per_op="<<median
                    <<",amortized_min_us="<<ordered.front()<<",amortized_max_us="<<ordered.back()
                    <<",operations_per_parent=32,one_event_pair,production_payload_only,no_fixture_sideband,cache_hot_within_parent,"
                    <<"4_paired_input_records,same_admitted_original_weights,5_warm_32_record_balanced_ABBA_BAAB_rounds,"
                    <<"measured_outputs_checked_without_relaunch,6_public_planes,terminal_event_only,no_host_submission_inside_interval\n";
            }
