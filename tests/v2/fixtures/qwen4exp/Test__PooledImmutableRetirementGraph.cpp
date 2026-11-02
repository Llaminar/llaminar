/**
 * @file Test__PooledImmutableRetirementGraph.cpp
 * @brief Real terminal root/PMA/graph insertion replay over explicit hardware-free backend APIs.
 *
 * The original provider, worker, storage factories, captured bindings and graph
 * stages execute as authored. Graph construction does not certify native storage,
 * GPU execution or wire traffic; every broken predecessor must reject before mutation.
 */
#include "PooledImmutableOriginalProgramReplay.h"
#include "execution/qwen4exp/PooledImmutableParticipantRetirementGraph.h"
namespace llaminar2::test {
/** @return Actual source→publication→Read/output descriptor under original ordered owners. */
PooledImmutableReadCompletion retirementRead(SourceGraphCase& c){
    const auto source=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",c.mixedVictim?&*c.mixedVictim:nullptr);
    const auto ready=appendPooledPublicationCoordinator(c.graph,c.arena,source,publicationMembers(c),c.coordCode,"publication");
    std::vector<q::PooledImmutableReadParticipant> members;for(size_t i=0;i<c.owners.size();++i){if(i==c.localCoordinate)members.emplace_back(c.localBinding);else members.emplace_back(q::PooledImmutableCapturedReadWave::bind(std::get<std::shared_ptr<const q::PooledImmutableCapturedPublicationWave>>(ready.owners[i]),c.memory));}
    return appendPooledReadCoordinator(c.graph,c.arena,ready,members,{c.readOutput,BufferId::QFN_PLE_EMBEDDING,0,16*160*sizeof(float)},"reads","actual_output");
}
/** @return Actual local binding or original remote retirement holder; never a replacement/root copy. */
std::vector<q::PooledImmutableRetirementParticipant> retirementMembers(SourceGraphCase& c,const PooledImmutableReadCompletion& read){
    std::vector<q::PooledImmutableRetirementParticipant> result;for(size_t i=0;i<c.owners.size();++i){if(i==c.localCoordinate)result.emplace_back(c.localBinding);else result.emplace_back(q::PooledImmutableCapturedRetirementWave::bind(std::get<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(read.owners[i]),c.memory));}return result;
}
/** @return Actual remote source/fill/publication/read tail before adding any physical release. */
PooledPinParticipantNodes retirementOwnerRead(SourceGraphCase& c,const std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>& holder,size_t coordinate){
    auto& graph=*c.ownerGraphs[coordinate];auto& arena=*c.ownerArenas[coordinate];
    const auto filled=appendPooledSourceOwnerFill(graph,arena,c.sources[coordinate],c.ownerCode,c.ownerVictims[coordinate].last,"source");
    const auto ready=appendPooledPublicationOwner(graph,arena,holder->read()->publication(),c.ownerCode,filled.last,"publication");
    return appendPooledReadOwner(graph,arena,holder->read(),c.ownerCode,ready.last,"reads");
}
/**
 * @brief Replay a disjoint original cohort range without reducing its assertions or repetitions.
 * @param count Original participant count.
 * @param first First local-owner coordinate; count denotes the independent remote coordinator.
 * @param past Exclusive end of the local-owner coordinate range.
 * @note Separate registrations keep the eight-owner cases well inside the normal timeout.
 */
void retirementCohortRange(size_t count,uint32_t first,uint32_t past){
    ASSERT_LT(first,past);ASSERT_LE(past,count+1);
    std::vector<PooledOriginalRecipe> cases;
    for(uint32_t local=first;local<past;++local)for(bool empty:{false,true}){
        if(count==1 && local==0)continue;std::vector<uint32_t> capacities(count);for(size_t i=0;i<count;++i)capacities[i]=empty?0:count==1?3:i%3==1?0:i+1;
        cases.push_back({capacities,true,local==count?q::adaptive_lookup::None:local});
    }
    PooledImmutableOriginalPrograms programs(cases);
    for(const auto& recipe:cases){const auto& capacities=recipe.capacities;
        SCOPED_TRACE(count);
        SCOPED_TRACE(recipe.local);
        SCOPED_TRACE(std::all_of(capacities.begin(),capacities.end(),[](auto n){return n==0;}));
        auto c=programs.make(recipe);const auto read=retirementRead(c);const auto members=retirementMembers(c,read);
        const auto terminal=appendPooledRetirementCoordinator(c.graph,c.arena,read,members,c.coordCode,"terminal");
        EXPECT_NO_THROW(requirePooledRetirementCompletion(c.graph,terminal));EXPECT_EQ(terminal.coordinator.last,terminal.observed);
        EXPECT_EQ(c.graph.getNode(terminal.readersRetired)->dependencies,std::vector<std::string>{read.output});
        for(size_t i=0;i<count;++i){if(i==c.localCoordinate){EXPECT_TRUE(terminal.gateSent[i].empty());EXPECT_TRUE(terminal.releaseReceived[i].empty());EXPECT_TRUE(terminal.retiredReceived[i].empty());continue;}
            const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(members[i]);
            EXPECT_EQ(edge->source(q::PooledRetirementWireRole::ReadersRetired).bytes,64u);EXPECT_EQ(edge->destination(q::PooledRetirementWireRole::ReadersRetired).bytes,64u);
            EXPECT_EQ(edge->source(q::PooledRetirementWireRole::Retired).bytes,60u);EXPECT_EQ(edge->destination(q::PooledRetirementWireRole::Retired).bytes,60u);
            EXPECT_EQ(edge->source(q::PooledRetirementWireRole::Releases).bytes,capacities[i]*72u);EXPECT_EQ(edge->destination(q::PooledRetirementWireRole::Releases).bytes,capacities[i]*72u);
            const auto sent=retirementOwnerRead(c,edge,i);const auto retired=appendPooledRetirementOwner(*c.ownerGraphs[i],*c.ownerArenas[i],edge,c.ownerCode,sent.last,"terminal");
            EXPECT_NO_THROW(requirePooledRetirementOwnerCompletion(*c.ownerGraphs[i],edge,retired.last));
            if(!capacities[i])EXPECT_THROW(edge->binding(q::PooledRetirementWireRole::Releases,CapturedTransferEndpoint::Producer),std::invalid_argument);
        }
    }
}
TEST(PooledImmutableRetirementGraph, CohortP1Remote){retirementCohortRange(1,1,2);}
TEST(PooledImmutableRetirementGraph, CohortP2){retirementCohortRange(2,0,3);}
TEST(PooledImmutableRetirementGraph, CohortP4){retirementCohortRange(4,0,5);}
TEST(PooledImmutableRetirementGraph, CohortP8LocalFirst){retirementCohortRange(8,0,4);}
TEST(PooledImmutableRetirementGraph, CohortP8LocalLast){retirementCohortRange(8,4,8);}
TEST(PooledImmutableRetirementGraph, CohortP8Remote){retirementCohortRange(8,8,9);}
TEST(PooledImmutableRetirementGraph, MissingOutputOrIssuedReadRejectsBeforeAnyTerminalNode){
    const PooledOriginalRecipe recipe{{2,0,3,1},true,1};PooledImmutableOriginalPrograms programs(std::vector<PooledOriginalRecipe>(4,recipe));
    for(unsigned defect=0;defect<4;++defect){auto c=programs.make(recipe);auto read=retirementRead(c);const auto members=retirementMembers(c,read);
        if(defect==0)c.graph.getNode(read.output)->dependencies.clear();
        if(defect==1)c.graph.getNode(read.sealed)->dependencies={read.imported[0]};
        if(defect==2)read.owners[0]=read.owners[3];
        if(defect==3)read.outputSpan.bytes-=160*sizeof(float);
        const auto before=c.graph.getExecutionOrder();EXPECT_THROW(appendPooledRetirementCoordinator(c.graph,c.arena,read,members,c.coordCode,"terminal"),std::invalid_argument);EXPECT_EQ(c.graph.getExecutionOrder(),before);EXPECT_EQ(c.graph.getNode("terminal/readers_retired"),nullptr);
    }
}
TEST(PooledImmutableRetirementGraph, GateAndSerialReadersMustRetireBeforeEverySharedBankOverwrite){
    const PooledOriginalRecipe recipe{{2,0,3,1},true,1};PooledImmutableOriginalPrograms programs(std::vector<PooledOriginalRecipe>(8,recipe));
    for(unsigned defect=0;defect<8;++defect){auto c=programs.make(recipe);const auto read=retirementRead(c);const auto members=retirementMembers(c,read);auto terminal=appendPooledRetirementCoordinator(c.graph,c.arena,read,members,c.coordCode,"terminal");
        ASSERT_NO_THROW(requirePooledRetirementCompletion(c.graph,terminal));
        if(defect==0)c.graph.getNode(terminal.gatePacked[0])->dependencies={read.output};
        if(defect==1)c.graph.getNode(terminal.releaseReceived[0])->dependencies={terminal.planned};
        if(defect==2)c.graph.getNode(terminal.releaseImported[1])->dependencies={terminal.releaseImported[0]};
        if(defect==3)c.graph.getNode(terminal.localRetire)->dependencies={terminal.localRelease};
        if(defect==4)c.graph.getNode(terminal.retiredReceived[0])->dependencies={terminal.reconciled};
        if(defect==5)c.graph.getNode(terminal.retiredImported[1])->dependencies={terminal.retiredImported[0]};
        if(defect==6)c.graph.getNode(terminal.finished)->dependencies={terminal.snapshot};
        if(defect==7)c.graph.getNode(terminal.observed)->dependencies={terminal.finished,read.output};
        const auto before=c.graph.getExecutionOrder();EXPECT_THROW(requirePooledRetirementCompletion(c.graph,terminal),std::invalid_argument);EXPECT_EQ(c.graph.getExecutionOrder(),before);
    }
}
TEST(PooledImmutableRetirementGraph, ActualOwnerGateDrainAndBothSendReadersCannotBeSkipped){
    std::vector<PooledOriginalRecipe> cases;for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<4;++defect)cases.push_back({{capacity,1},true});PooledImmutableOriginalPrograms programs(cases);
    for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<4;++defect){auto c=programs.make({{capacity,1},true});const auto read=retirementRead(c);const auto members=retirementMembers(c,read);const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedRetirementWave>>(members[0]);const auto rows=retirementOwnerRead(c,edge,0);auto& graph=*c.ownerGraphs[0];auto& arena=*c.ownerArenas[0];const auto terminal=appendPooledRetirementOwner(graph,arena,edge,c.ownerCode,rows.last,"terminal");
        ASSERT_NO_THROW(requirePooledRetirementOwnerCompletion(graph,edge,terminal.last));
        if(defect==0)graph.getNode("terminal/reader_gate_receive")->dependencies={"publication/ready_send"};
        if(defect==1)graph.getNode("terminal/drain")->dependencies={"terminal/reader_gate_receive"};
        if(defect==2)graph.getNode("terminal/retire")->dependencies={capacity?"terminal/release":"terminal/drain"};
        if(defect==3)graph.getNode("terminal/retired_send")->dependencies={"terminal/retire"};
        const auto before=graph.getExecutionOrder();EXPECT_THROW(requirePooledRetirementOwnerCompletion(graph,edge,terminal.last),std::invalid_argument);EXPECT_EQ(graph.getExecutionOrder(),before);
    }
}
}
