/**
 * @file Test__PooledImmutableReadGraph.cpp
 * @brief Actual original root/PMA/read DAG insertion replay over backend API doubles.
 *
 * This isolated process retains the real provider/worker, prepared factories and
 * graph stages. No native launch, graph capture or vendor runtime is simulated
 * by a successful insertion. Corruptions must reject before successor mutation.
 */
#include "PooledImmutableOriginalProgramReplay.h"
#include "execution/qwen4exp/PooledImmutableParticipantReadGraph.h"
namespace llaminar2::test {
/** @return Original local or remote read holders, without substituting a storage/channel owner. */
std::vector<q::PooledImmutableReadParticipant> readMembers(SourceGraphCase& c,const PooledImmutablePublicationCompletion& ready){
    std::vector<q::PooledImmutableReadParticipant> out;for(size_t i=0;i<c.owners.size();++i){if(i==c.localCoordinate)out.emplace_back(c.localBinding);else out.emplace_back(q::PooledImmutableCapturedReadWave::bind(std::get<std::shared_ptr<const q::PooledImmutableCapturedPublicationWave>>(ready.owners[i]),c.memory));}return out;
}
/** @return Actual source→publication→Ready, using the concrete original source/owner chain. */
PooledImmutablePublicationCompletion readReady(SourceGraphCase& c){const auto source=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",c.mixedVictim?&*c.mixedVictim:nullptr);const auto members=publicationMembers(c);return appendPooledPublicationCoordinator(c.graph,c.arena,source,members,c.coordCode,"publication");}
/** @return Exact original FP32 consumer declaration; cold construction reads no live device data. */
q::PooledNativeTensorSpan readOutput(SourceGraphCase& c){return {c.readOutput,BufferId::QFN_PLE_EMBEDDING,0,16*160*sizeof(float)};}
TEST(PooledImmutableReadGraph, ActualOriginalRootFactoriesAndAllIssuedEdgesRetainSuccessAndFailureDrain){
    std::vector<PooledOriginalRecipe> cases;
    for(size_t count:{1u,2u,4u,8u})for(uint32_t local=0;local<=count;++local)for(bool zero:{false,true}){
        if(count==1 && local==0)continue;std::vector<uint32_t> capacities(count);for(size_t i=0;i<count;++i)capacities[i]=zero?0:count==1?3:i%3==0?0:i+1;
        cases.push_back({capacities,true,local==count?q::adaptive_lookup::None:local});
    }
    PooledImmutableOriginalPrograms programs(cases);
    for(const auto& recipe:cases){const auto& capacities=recipe.capacities;const auto count=capacities.size();
        auto c=programs.make(recipe);const auto ready=readReady(c);const auto members=readMembers(c,ready);
        const auto reads=appendPooledReadCoordinator(c.graph,c.arena,ready,members,readOutput(c),"reads","original_order_read");
        EXPECT_NO_THROW(requirePooledReadCompletion(c.graph,reads));EXPECT_EQ(reads.coordinator.last,reads.output);
        EXPECT_EQ(c.graph.getNode(reads.output)->dependencies,std::vector<std::string>{reads.sealed});
        for(size_t i=0;i<count;++i)if(i!=c.localCoordinate){const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(members[i]);
            EXPECT_EQ(edge->source(q::PooledReadWireRole::Receipt).bytes,64u);EXPECT_EQ(edge->destination(q::PooledReadWireRole::Receipt).bytes,64u);
            const auto fill=appendPooledSourceOwnerFill(*c.ownerGraphs[i],*c.ownerArenas[i],c.sources[i],c.ownerCode,c.ownerVictims[i].last,"source");
            const auto published=appendPooledPublicationOwner(*c.ownerGraphs[i],*c.ownerArenas[i],edge->publication(),c.ownerCode,fill.last,"publication");
            const auto owner=appendPooledReadOwner(*c.ownerGraphs[i],*c.ownerArenas[i],edge,c.ownerCode,published.last,"reads");
            EXPECT_EQ(owner.last,capacities[i]?"reads/rows_send":"reads/receipt_send");EXPECT_NO_THROW(requirePooledReadOwnerCompletion(*c.ownerGraphs[i],edge,owner.last));
            if(capacities[i]){EXPECT_EQ(edge->destination(q::PooledReadWireRole::Requests).bytes,capacities[i]*64u);EXPECT_EQ(edge->source(q::PooledReadWireRole::Rows).bytes,capacities[i]*90u);}
        }
    }
}
TEST(PooledImmutableReadGraph, MissingOriginalReadyRejectsBeforeAnyReaderMutation){
    const PooledOriginalRecipe recipe{{2,0,3,1},true,1};PooledImmutableOriginalPrograms programs(std::vector<PooledOriginalRecipe>(3,recipe));
    for(unsigned defect=0;defect<3;++defect){auto c=programs.make(recipe);auto ready=readReady(c);const auto members=readMembers(c,ready);
        if(defect==0)c.graph.getNode(ready.ready)->dependencies.clear();if(defect==1)c.graph.getNode(ready.acquiredValidated)->dependencies={ready.publicationValidated};if(defect==2)ready.owners[0]=ready.owners[3];
        const auto before=c.graph.getExecutionOrder();EXPECT_THROW(appendPooledReadCoordinator(c.graph,c.arena,ready,members,readOutput(c),"reads","original_order_read"),std::invalid_argument);EXPECT_EQ(c.graph.getExecutionOrder(),before);
    }
}
TEST(PooledImmutableReadGraph, MissingIssuedReadImportOutputOrOriginalSpanRejectsBeforeTerminalMutation){
    const PooledOriginalRecipe recipe{{2,0,3,1},true,1};PooledImmutableOriginalPrograms programs(std::vector<PooledOriginalRecipe>(8,recipe));
    for(unsigned defect=0;defect<8;++defect){auto c=programs.make(recipe);const auto ready=readReady(c);const auto members=readMembers(c,ready);auto reads=appendPooledReadCoordinator(c.graph,c.arena,ready,members,readOutput(c),"reads","original_order_read");
        ASSERT_NO_THROW(requirePooledReadCompletion(c.graph,reads));
        if(defect==0)c.graph.getNode(reads.requestSent[3])->dependencies.clear();
        if(defect==1)c.graph.getNode(reads.receiptReceived[0])->dependencies={reads.planned};
        if(defect==2)c.graph.getNode(reads.imported[0])->dependencies={reads.receiptReceived[0]};
        if(defect==3)c.graph.getNode(reads.sealed)->dependencies={reads.imported[0]};
        if(defect==4)c.graph.getNode(reads.output)->dependencies={reads.planned};
        if(defect==5)reads.rowsReceived[0].clear();
        if(defect==6)reads.outputSpan.bytes-=160*sizeof(float);
        if(defect==7)reads.owners[0]=reads.owners[3];
        const auto before=c.graph.getExecutionOrder();EXPECT_THROW(requirePooledReadCompletion(c.graph,reads),std::invalid_argument);EXPECT_EQ(c.graph.getExecutionOrder(),before);
    }
}
TEST(PooledImmutableReadGraph, MissingOriginalOwnerCopyReceiptOrReadyRejectsBeforeRelease){
    std::vector<PooledOriginalRecipe> cases;for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<3;++defect)cases.push_back({{capacity,1},true});PooledImmutableOriginalPrograms programs(cases);
    for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<3;++defect){auto c=programs.make({{capacity,1},true});const auto ready=readReady(c);const auto members=readMembers(c,ready);const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(members[0]);auto& graph=*c.ownerGraphs[0];auto& arena=*c.ownerArenas[0];
        const auto fill=appendPooledSourceOwnerFill(graph,arena,c.sources[0],c.ownerCode,c.ownerVictims[0].last,"source");const auto published=appendPooledPublicationOwner(graph,arena,edge->publication(),c.ownerCode,fill.last,"publication");const auto read=appendPooledReadOwner(graph,arena,edge,c.ownerCode,published.last,"reads");ASSERT_NO_THROW(requirePooledReadOwnerCompletion(graph,edge,read.last));
        if(defect==0)graph.getNode("reads/read_rows")->dependencies.clear();if(defect==1)graph.getNode("reads/receipt_send")->dependencies={published.last};if(defect==2)graph.getNode("reads/outcome_receive")->dependencies={fill.last};
        const auto before=graph.getExecutionOrder();EXPECT_THROW(requirePooledReadOwnerCompletion(graph,edge,read.last),std::invalid_argument);EXPECT_EQ(graph.getExecutionOrder(),before);
    }
}
TEST(PooledImmutableReadGraph, SingleRemoteOwnerWithIndependentCoordinatorKeepsRealDataEdges){
    PooledImmutableOriginalPrograms programs({{{0},false},{{3},false}});
    for(const uint32_t capacity:{0u,3u}){auto c=programs.make({{capacity},false});const auto ready=readReady(c);const auto members=readMembers(c,ready);
        const auto read=appendPooledReadCoordinator(c.graph,c.arena,ready,members,readOutput(c),"reads","original_order_read");
        EXPECT_NO_THROW(requirePooledReadCompletion(c.graph,read));EXPECT_EQ(read.owners.size(),1u);EXPECT_FALSE(read.receiptReceived[0].empty());
        EXPECT_EQ(read.requestSent[0].empty(),capacity==0);EXPECT_EQ(read.rowsReceived[0].empty(),capacity==0);
        const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(members[0]);
        EXPECT_NE(q::pooledSourceOwner(edge->publication()->sourceWave())->plan().allocator(),c.coordinator->plan().allocator());
        const auto fill=appendPooledSourceOwnerFill(*c.ownerGraphs[0],*c.ownerArenas[0],c.sources[0],c.ownerCode,c.ownerVictims[0].last,"source");
        const auto published=appendPooledPublicationOwner(*c.ownerGraphs[0],*c.ownerArenas[0],edge->publication(),c.ownerCode,fill.last,"publication");
        const auto owner=appendPooledReadOwner(*c.ownerGraphs[0],*c.ownerArenas[0],edge,c.ownerCode,published.last,"reads");
        EXPECT_NO_THROW(requirePooledReadOwnerCompletion(*c.ownerGraphs[0],edge,owner.last));
    }
}

}

namespace llaminar2::test {
/** @test Original partial source/boundary setup retains same-PMA obligations until actual handles and generation retire. */
TEST(PooledImmutableReadGraph, RecordedOrderingPartialSourceSetupRetainsOriginalClaimsUntilReset){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(std::size_t prefix=1;prefix<=4;++prefix){
        ASSERT_EXIT(([&]{
            const PooledOriginalRecipe recipe{{2u,0u},false,q::adaptive_lookup::None,false,type};PooledImmutableOriginalPrograms programs({recipe});
            auto& backends=programs.backends();auto& backend=type==DeviceType::CUDA?backends.cuda:backends.rocm;
            backend.failEventCreationCall=prefix;std::shared_ptr<PhysicalMemoryAuthority> memory;std::size_t waits=0;
            backend.beforeRetirementWait=[&]{++waits;ASSERT_NE(memory,nullptr);
                EXPECT_GT(memory->reservedBytes({type,0},PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
                EXPECT_GT(memory->claimedBytes({type,0},PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);};
            EXPECT_THROW((void)programs.make(recipe,&memory),std::exception);
            ASSERT_NE(memory,nullptr);EXPECT_EQ(backend.eventCreates,prefix);
            EXPECT_EQ(backends.cuda.liveAllocations(),0u);EXPECT_EQ(backends.rocm.liveAllocations(),0u);
            EXPECT_GT(memory->reservedBytes({type,0},PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
            EXPECT_EQ(memory->claimedBytes({type,0},PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);
            EXPECT_EQ(waits==0,prefix==1);backend.beforeRetirementWait={};programs.retire();
            EXPECT_EQ(memory->reservedBytes({type,0},PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
            EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
            std::_Exit(::testing::Test::HasFailure()?2:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}
/**
 * @test Actual source factories preserve storage across serial canonical controller lifetimes.
 * Backend doubles cover metadata ownership; no native completion is fabricated.
 */
TEST(PooledImmutableReadGraph, CanonicalGenerationBorrowRetiresBeforeRunnerAddressReplacement) {
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm}) {
        ASSERT_EXIT(([&]{
            const PooledOriginalRecipe recipe{{2u},false,0,false,type};
            PooledImmutableOriginalPrograms programs({recipe});auto c=programs.make(recipe);
            ASSERT_FALSE(c.localBinding);ASSERT_TRUE(c.sourceMembers.empty());
            ASSERT_TRUE(c.ownerGraphs.empty());ASSERT_TRUE(c.forwards.front()==nullptr);
            ASSERT_EQ(c.graph.retainedStorageBoundary(),c.boundary);
            ASSERT_FALSE(c.complete.first.empty());ASSERT_FALSE(c.complete.last.empty());
            const auto device=DeviceId{type,0};const auto source=c.boundary->pooledStorage();
            const auto identity=c.boundary->captureIdentity();
            const auto bytes=c.memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation);
            const auto retire=[&](RetainedStorageGenerationControl words){
                PooledImmutableFactoryReplayAccess::retireGenerationControl(*c.boundary,words);
            };
            const RetainedStorageGenerationControl original{c.control->healthy(),c.control->error()};
            EXPECT_THROW(retire({original.error,original.healthy}),std::logic_error);
            {
                GraphCaptureGuard recording;
                EXPECT_THROW(retire(original),std::logic_error);
                EXPECT_EQ(c.boundary->generationControl().healthy,original.healthy);
            }
            // Release the actual fixture topology before its graph-retirement frontier.
            c.graph.clear();for(auto& graph:c.ownerGraphs)if(graph)graph->clear();
            retire(original);
            EXPECT_THROW(c.boundary->generationControl(),std::logic_error);
            EXPECT_THROW(retire(original),std::logic_error);
            // Reuse freed, already admitted output capacity for a second controller;
            // retain the first owner so both canonical addresses remain distinct.
            c.arena=BufferArena{};c.readOutput.reset();
            ASSERT_TRUE(c.arena.registerExternalBuffer(BufferId::MTP_GENERATION_CONTROL,c.controller));
            c.control=DeviceGenerationControlBinding::admit(c.arena,device,0);
            const CPUTensorStorageDeclaration declaration(1,sampling_math::kDeviceGenerationControlCount,
                BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu());
            auto lease=c.memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,
                sampling_math::kDeviceGenerationControlCount*sizeof(int32_t));
            std::shared_ptr<TensorBase> controller=declaration.materialize(c.memory);
            TransferEngine::prepareDeviceInput(controller.get(),device,c.originalSourceStream);
            TransferEngine::requireDeviceInput(controller.get(),device,c.originalSourceStream);
            BufferArena arena;ASSERT_TRUE(arena.registerExternalBuffer(BufferId::MTP_GENERATION_CONTROL,controller));
            const auto next=DeviceGenerationControlBinding::admit(arena,device,0);
            ASSERT_NE(next.healthy(),original.healthy);ASSERT_NE(next.error(),original.error);
            for(int lifetime=0;lifetime<20;++lifetime){
                SCOPED_TRACE(lifetime);
                const auto& selected=lifetime%2?*c.control:next;
                const auto& other=lifetime%2?next:*c.control;
                const RetainedStorageGenerationControl words{selected.healthy(),selected.error()};
                c.boundary->bindPooledGenerationControl(selected);
                EXPECT_NO_THROW(c.boundary->bindPooledGenerationControl(selected));
                EXPECT_THROW(c.boundary->bindPooledGenerationControl(other),std::invalid_argument);
                EXPECT_THROW(c.boundary->bindGenerationControl({other.healthy(),other.error()}),std::invalid_argument);
                EXPECT_THROW(retire({other.healthy(),other.error()}),std::logic_error);
                EXPECT_EQ(c.boundary->generationControl().healthy,words.healthy);
                EXPECT_EQ(c.boundary->generationControl().error,words.error);
                EXPECT_EQ(c.boundary->pooledStorage(),source);EXPECT_EQ(c.boundary->captureIdentity(),identity);
                EXPECT_EQ(c.memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,
                    PhysicalMemoryMaterializationKind::NewAllocation),bytes);
                retire(words);
                EXPECT_THROW(c.boundary->generationControl(),std::logic_error);
                EXPECT_THROW(retire(words),std::logic_error);
            }
            std::_Exit(::testing::Test::HasFailure()?2:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}

}
