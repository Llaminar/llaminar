/**
 * @file PooledIdentityRejectionMethods.inc
 * @brief Captured original-owner rejection equivalence over recorded native initial state.
 *
 * Included only in the existing ProviderReplay test owner. Recorded-state uploads
 * are kernel-replay inputs, never production recovery or evidence of retirement.
 * Retained failures separately exercise the actual supported cancel/release drain.
 */
    /** @brief Capture original prefix stages and exactly the selected actual production phase. */
    void buildIdentityRejection(){
        const auto purpose=*originalProgram.spec.identityRejection;
        bool found=false;
        std::string previousBefore,previousFault;
        for(const auto& name:compute.getExecutionOrder()){
            const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(compute.getNode(name)->stage.get());
            requireProvider(stage,"Identity rejection prefix contains an unrecognized non-native stage");
            const auto* own=std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase());
            const bool selected=own && own->action==purpose.action;
            const auto append=[&](ComputeGraph& destination,std::string& previous,std::unique_ptr<IComputeStage> item,const std::string& node){
                destination.addNode(node,std::move(item),device);
                if(!previous.empty())destination.addDependency(node,previous);
                previous=node;
            };
            if(selected){
                requireProvider(own->storage==owner && own->operands.sourceCondition()==n::SourceCondition::Always,"Identity rejection lost original owner/producer");
                identitySelected=own;
                if(purpose.action==n::OwnerAction::Pin){
                    const auto& input=own->operands.spans()[static_cast<unsigned>(q::PooledNativeOperand::Records)];
                    requireProvider(input && input->root==coordinator->roots().native && input->offset==coordinator->plan().parameters().native.pin_commands,
                        "Identity fault producer differs from the actual original Pin command span");
                }
                append(identityFaultCompute,previousFault,std::make_unique<IdentityFaultStage>(owner,coordinator,purpose),name+"/recorded_invalid_input");
                append(identityFaultCompute,previousFault,std::make_unique<PooledImmutableNativePhaseStage>(*arena,code,stage->phase()),name);
                found=true;break;
            }
            append(identityBeforeCompute,previousBefore,std::make_unique<PooledImmutableNativePhaseStage>(*arena,code,stage->phase()),name);
            append(identityFaultCompute,previousFault,std::make_unique<PooledImmutableNativePhaseStage>(*arena,code,stage->phase()),name);
        }
        requireProvider(found && identitySelected,"Identity rejection selected phase is absent");
        identityFaultGraph=graphs.capture(3);
        const auto record=[&](ComputeGraph& definition,DeviceGraphExecutor& recorder,IGPUGraphCapture* facade){
            requireProvider(facade,"Identity rejection lacks its preadmitted original graph");
            recorder.setArena(arena.get());
            for(const auto& name:definition.getExecutionOrder())requireProvider(definition.getNode(name)->stage->prepareGraphLaunch(context.get(),stream),"Identity rejection preparation failed");
            requireProvider(recorder.prepareGraphStorageForCapture(definition,context.get(),stream,"original native identity rejection"),"Identity rejection root preparation failed");
            auto dependencies=recorder.planGraphCaptureDependencies(definition,definition.getExecutionOrder(),device,stream,"original identity rejection dependencies");
            requireProvider(bool(dependencies),"Identity rejection dependency plan missing");
            {ScopedBackendGraphCapture capture(worker,*facade,"original identity rejection",dependencies.get());
                requireProvider(capture.begin(),"Identity rejection capture failed");
                requireProvider(recorder.recordSetupGraphMaterialization(definition,context.get()),"Identity rejection actual stage recording failed");capture.finish();}
        };
        record(identityBeforeCompute,identityBeforeExecutor,overbound);
        record(identityFaultCompute,identityFaultExecutor,identityFaultGraph);
        std::string detail;
        requireProvider(identityFaultGraph->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
        requireProvider(identityFaultGraph->instantiate(),"Identity rejection graph instantiation failed");
        if(purpose.action!=n::OwnerAction::Pin){
            identityDrainGraph=graphs.capture(4);
            std::string previous;
            for(const auto action:{purpose.action==n::OwnerAction::AcceptFree?n::OwnerAction::CancelFree:n::OwnerAction::CancelVictim,n::OwnerAction::Release}){
                q::PooledNativeOperandSpans spans{};
                spans[static_cast<unsigned>(q::PooledNativeOperand::Invocation)]=identitySelected->operands.spans()[static_cast<unsigned>(q::PooledNativeOperand::Invocation)];
                q::PooledImmutableOwnerOperands operands(action,owner->plan(),std::move(spans));
                auto node=std::string("original_rejection_drain/")+std::to_string(static_cast<unsigned>(action));
                identityDrainCompute.addNode(node,std::make_unique<PooledImmutableNativePhaseStage>(*arena,code,PooledImmutableNativePhaseStage::Owner{action,owner,std::move(operands)}),device);
                if(!previous.empty())identityDrainCompute.addDependency(node,previous);previous=node;
            }
            record(identityDrainCompute,identityDrainExecutor,identityDrainGraph);
            requireProvider(identityDrainGraph->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&detail),detail.c_str());
            requireProvider(identityDrainGraph->instantiate(),"Identity rejection drain instantiation failed");
        }
    }
    /** @brief One immutable recorded native root; this is diagnostic input, never a live ownership mirror. */
    struct IdentityRootRecord final {std::shared_ptr<TensorBase> root;PooledImmutableNativeReplayBytes bytes;};
    using IdentitySnapshot=std::vector<IdentityRootRecord>;
    /** @return Completed original device bytes for this kernel-replay starting state only. */
    IdentitySnapshot identitySnapshot(){
        IdentitySnapshot result;
        for(const auto& family:{owner->roots(),coordinator->roots(),boundary->pooledStorage()->roots()})
            for(const auto& root:{family.cache,family.frame,family.native})if(root){
                IdentityRootRecord record{root,PooledImmutableNativeReplayBytes(root->size_bytes())};
                requireProvider(backend->deviceToHostOnStream(record.bytes.data(),root->gpu_data_ptr(),record.bytes.size(),device.ordinal,stream),"Recorded native input read failed");
                join();result.push_back(std::move(record));
            }
        return result;
    }
    /** @brief Restore recorded kernel-replay input after native completion, outside capture.
     * @param snapshot Original completed starting bytes; never a fabricated recovery result.
     */
    void restoreIdentityInput(const IdentitySnapshot& snapshot){
        join();
        for(const auto& entry:snapshot){
            requireProvider(entry.root->size_bytes()==entry.bytes.size(),"Recorded identity input geometry changed");
            requireProvider(backend->hostToDeviceOnStream(entry.root->gpu_data_ptr(),entry.bytes.data(),entry.bytes.size(),device.ordinal,stream),"Recorded identity input upload failed");
            TransferEngine::publishDeviceWrite(entry.root.get(),device,stream);
        }
        join();
    }
    /** @brief Locate one exact original root in a completed diagnostic trace.
     * @param snapshot Completed trace with original root controlblocks.
     * @param root Original root.
     * @return Mutable diagnostic bytes; no production Tensor pointer is changed.
     */
    static PooledImmutableNativeReplayBytes& identityBytes(IdentitySnapshot& snapshot,const std::shared_ptr<TensorBase>& root){
        for(auto& record:snapshot)if(record.root==root && !record.root.owner_before(root) && !root.owner_before(record.root))return record.bytes;
        throw std::logic_error("Identity trace has a missing or foreign original root");
    }
    /** @brief Compare actual captured rejection to the existing scalar owner on the same decoded trace.
     * @param rows Full original authored hot-row prefix; mutation coordinate lies inside it.
     */
    void replayIdentityRejection(std::span<const std::uint64_t> rows){
        const auto purpose=*originalProgram.spec.identityRejection;
        requireProvider(rows.size()>purpose.position && identitySelected && identityFaultGraph,"Identity rejection has no live fault or original phase");
        input(rows);
        auto initial=identitySnapshot();
        requireProvider(graphs.launch(overbound),"Original before-phase prefix launch failed");join();
        auto before=identitySnapshot();auto expected=before;
        auto& expectedCache=identityBytes(expected,owner->roots().cache);
        auto& expectedFrame=identityBytes(expected,owner->roots().frame);
        auto& expectedNative=identityBytes(expected,owner->roots().native);
        const auto& plan=owner->plan().parameters();
        const auto pointer=[&](q::PooledNativeOperand role)->const void*{
            const auto& span=identitySelected->operands.spans()[static_cast<unsigned>(role)];
            if(!span)return nullptr;
            auto& bytes=identityBytes(expected,span->root);
            requireProvider(span->offset<=bytes.size() && span->bytes<=bytes.size()-span->offset,"Identity trace span exceeds its original root");
            return bytes.data()+span->offset;
        };
        const auto& records=identitySelected->operands.spans()[static_cast<unsigned>(q::PooledNativeOperand::Records)];
        const auto layout=identitySelected->operands.recordLayout();
        n::OwnerInputs inputs{static_cast<const n::Invocation*>(pointer(q::PooledNativeOperand::Invocation)),pointer(q::PooledNativeOperand::Records),
            static_cast<const std::uint64_t*>(pointer(q::PooledNativeOperand::RecordBytes)),pointer(q::PooledNativeOperand::Frontier),nullptr,nullptr,
            n::SourceCondition::Always,nullptr,layout,static_cast<const n::DirectedSlice*>(pointer(q::PooledNativeOperand::AllocationSlice)),
            layout==n::RecordLayout::DirectedAllocationBank?records->offset:0,
            (layout==n::RecordLayout::DirectedAllocationBank || layout==n::RecordLayout::DirectedCommandBank)?records->bytes:0};
        n::OwnerArgs args{plan,expectedCache.data(),expectedFrame.data(),expectedNative.data(),inputs};
        const auto offset=purpose.position;
        if(purpose.action==n::OwnerAction::Pin){
            requireProvider(layout==n::RecordLayout::Compact || (layout==n::RecordLayout::DirectedCommandBank && inputs.allocation_slice),"Identity Pin trace lost its original command span");
            const auto offsetBytes=layout==n::RecordLayout::DirectedCommandBank?inputs.allocation_slice->byte_offset:0;
            auto* commands=reinterpret_cast<p::OwnerCommand*>(const_cast<std::byte*>(static_cast<const std::byte*>(inputs.records))+offsetBytes);
            const auto count=layout==n::RecordLayout::DirectedCommandBank?inputs.allocation_slice->live_bytes/sizeof(p::OwnerCommand):*inputs.record_bytes/sizeof(p::OwnerCommand);
            requireProvider(count==rows.size(),"Identity rejection lacks its exact original live Pin prefix");
            auto& bad=commands[offset];const auto original=commands[0];
            if(purpose.fault==identity_fixture::IdentityFault::DuplicateRow){const auto unique=bad.target.unique;bad.target=original.target;bad.target.unique=unique;bad.requested_row=original.requested_row;}
            else if(purpose.fault==identity_fixture::IdentityFault::DuplicateUnique)bad.target.unique=original.target.unique;
            else if(purpose.fault==identity_fixture::IdentityFault::DuplicateSlot)bad.target.slot=original.target.slot;
        }else{
            auto* pins=n::at<p::Grant>(args.frame,plan.frame.initial_pins);
            if(purpose.fault==identity_fixture::IdentityFault::DuplicateRow)pins[offset].row=pins[0].row;
            else if(purpose.fault==identity_fixture::IdentityFault::DuplicateUnique)pins[offset].unique=pins[0].unique;
            else if(purpose.fault==identity_fixture::IdentityFault::ForeignOwner)++pins[offset].owner;
            else if(purpose.fault==identity_fixture::IdentityFault::ForeignInterval)++pins[offset].interval.sequence;
            const auto decisionBytes=purpose.action==n::OwnerAction::AcceptFree?static_cast<const n::FreePlanHeader*>(inputs.frontier)->decision_bytes:
                static_cast<const n::VictimPlanHeader*>(inputs.frontier)->decision_bytes;
            requireProvider(decisionBytes==0 && n::allocationFrame(args).count==0 &&
                n::at<p::OwnerFrame>(args.frame,plan.frame.control)->initial_pin_count==rows.size(),
                "Retained identity rejection requires all original pins and ZERO incoming decisions");
        }
        const auto scalar=[&]<class Slot>(){
            switch(purpose.action){
            case n::OwnerAction::Pin:return n::owner<n::OwnerAction::Pin,Slot>(args);
            case n::OwnerAction::AcceptFree:return n::owner<n::OwnerAction::AcceptFree,Slot>(args);
            case n::OwnerAction::AcceptVictim:return n::owner<n::OwnerAction::AcceptVictim,Slot>(args);
            default:throw std::logic_error("Identity rejection has a foreign scalar role");
            }
        };
        const bool accepted=plan.owner.rowWidth()==q::adaptive_lookup::RowKeyWidth::UInt32?scalar.template operator()<q::adaptive_lookup::Slot32>():scalar.template operator()<q::adaptive_lookup::Slot64>();
        requireProvider(!accepted,"Recorded malformed identity unexpectedly passed original scalar validation");
        restoreIdentityInput(initial);
        requireProvider(graphs.launch(identityFaultGraph),"Actual captured owner rejection launch failed");join();
        auto actual=identitySnapshot();
        const auto& actualCache=identityBytes(actual,owner->roots().cache);
        ASSERT_EQ(actualCache.size(),expectedCache.size());
        EXPECT_EQ(std::memcmp(actualCache.data(),expectedCache.data(),expectedCache.size()),0)<<"Rejected owner changed original physical slot/payload bytes";
        const auto& actualFrame=identityBytes(actual,owner->roots().frame);
        const auto& beforeFrame=identityBytes(before,owner->roots().frame);
        const auto liveKeys=rows.size()*sizeof(std::uint64_t);
        const auto inactiveKeys=(std::size_t{plan.owner.maximumUnique()}-rows.size())*sizeof(std::uint64_t);
        for(const auto bank:{plan.frame.validation_a,plan.frame.validation_b})
            EXPECT_EQ(std::memcmp(actualFrame.data()+bank+liveKeys,beforeFrame.data()+bank+liveKeys,inactiveKeys),0)<<"Identity validation touched inactive admitted scratch";
        for(const auto range:std::array<std::pair<std::size_t,std::size_t>,5>{{{plan.frame.control,sizeof(p::OwnerFrame)},
            {plan.frame.allocation_control,sizeof(p::AllocationFrame)},{plan.frame.initial_pins,std::size_t{plan.owner.maximumUnique()}*sizeof(p::Grant)},
            {plan.frame.accepted_plan,std::size_t{plan.owner.maximumUnique()}*sizeof(p::AllocationDecision)},
            {plan.frame.replies,std::size_t{plan.owner.maximumUnique()}*sizeof(p::Grant)}}})
            EXPECT_EQ(std::memcmp(actualFrame.data()+range.first,expectedFrame.data()+range.first,range.second),0);
        for(auto& expectedRoot:expected){
            if(expectedRoot.root==owner->roots().cache || expectedRoot.root==owner->roots().frame || expectedRoot.root==owner->roots().native)continue;
            const auto& observed=identityBytes(actual,expectedRoot.root);
            ASSERT_EQ(observed.size(),expectedRoot.bytes.size());
            EXPECT_EQ(std::memcmp(observed.data(),expectedRoot.bytes.data(),observed.size()),0)<<"Captured owner rejection changed another original root";
        }
        const auto& actualNative=identityBytes(actual,owner->roots().native);
        ASSERT_EQ(actualNative.size(),expectedNative.size());
        EXPECT_EQ(std::memcmp(actualNative.data(),expectedNative.data(),expectedNative.size()),0)<<"Captured rejection changed original publication/grant outputs";
        const auto actualControl=*n::at<p::OwnerFrame>(const_cast<std::byte*>(actualFrame.data()),plan.frame.control);
        EXPECT_EQ(actualControl.status,q::adaptive_lookup::Status::Invariant);
        EXPECT_EQ(actualControl.held_pins,purpose.action==n::OwnerAction::Pin?0u:rows.size());
        if(identityDrainGraph){
            // Restore ONLY the deliberately invalid Grant input. Actual cache pin
            // bits and sticky Invariant remain; original cancel/release drains them.
            auto& originalFrame=identityBytes(before,owner->roots().frame);
            const auto bytes=std::size_t{plan.owner.maximumUnique()}*sizeof(p::Grant);
            requireProvider(backend->hostToDeviceOnStream(static_cast<std::byte*>(owner->roots().frame->gpu_data_ptr())+plan.frame.initial_pins,
                originalFrame.data()+plan.frame.initial_pins,bytes,device.ordinal,stream),"Original grant input restore failed");
            TransferEngine::publishDeviceWrite(owner->roots().frame.get(),device,stream);join();
            requireProvider(graphs.launch(identityDrainGraph),"Original rejection drain launch failed");join();
            const auto drained=read<p::OwnerFrame>(static_cast<const std::byte*>(owner->roots().frame->gpu_data_ptr())+plan.frame.control,1).front();
            const auto progress=read<n::OwnerProgress>(static_cast<const std::byte*>(owner->roots().native->gpu_data_ptr())+plan.native.progress,1).front();
            EXPECT_EQ(drained.status,q::adaptive_lookup::Status::Invariant);EXPECT_EQ(drained.held_pins,0u);EXPECT_EQ(drained.reservations,0u);
            EXPECT_EQ(progress.released,rows.size());EXPECT_EQ(progress.release_bytes,rows.size()*sizeof(p::ReleaseReceipt));
            const auto checkPins=[&]<class Slot>(){
                const auto physical=read<Slot>(static_cast<const std::byte*>(owner->roots().cache->gpu_data_ptr())+plan.cache.slots,plan.owner.capacity());
                for(const auto& slot:physical)EXPECT_FALSE(q::adaptive_lookup::PhysicalSlot<Slot>::pinned(slot));
            };
            if(plan.owner.rowWidth()==q::adaptive_lookup::RowKeyWidth::UInt32)checkPins.template operator()<q::adaptive_lookup::Slot32>();
            else checkPins.template operator()<q::adaptive_lookup::Slot64>();
        }
        // This is solely the next recorded kernel-replay input, NOT a proof that
        // an Invariant failure can retire or recover through normal serving.
        restoreIdentityInput(initial);
    }
