/**
 * @file Test__PooledImmutableOwnerPinChain.cpp
 * @brief Replay actual pooled owner/source ordering against one predeclared original authority per process.
 *
 * The shared fixture owns actual prepared worker provenance, immutable IO case
 * identities, and same-PMA reservations. Canonical runners isolate process roots.
 */
#include "fixtures/qwen4exp/PooledImmutableOriginalProgramReplay.h"
#include "loaders/ImmutableLookupSetupObservation.h"
#include "execution/qwen4exp/PooledImmutableDemandFrontier.h"
#include "execution/qwen4exp/PooledImmutableDirectedLookupGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantRetirementGraph.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLEBuilder.h"
#include <typeinfo>
namespace llaminar2::test {
TEST(PooledImmutableOwnerPinChain, OriginalConcretePositiveAndC0ChainsPass){for(uint32_t capacity:{0u,3u}){OwnerChainCase c(capacity);
    EXPECT_NO_THROW(requirePooledPinOwnerCompletion(c.graph,c.returned,c.complete.last));const auto before=c.graph.getExecutionOrder().size();
    EXPECT_NO_THROW(appendPooledFreeOwnerWave(c.graph,c.arena,c.free,c.code,c.complete.last,"free"));
    EXPECT_GT(c.graph.getExecutionOrder().size(),before);}}

TEST(PooledImmutableOwnerPinChain, SameRootForeignMissingAndOutOfOrderRejectBeforeMutation){for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<4;++defect){OwnerChainCase c(capacity,defect==1);const std::array<std::string,4> predecessors{"","foreign/frontier_receive","returned/header_send",c.incoming.last};const auto acquired=c.acquire(predecessors[defect]);const auto before=c.graph.getExecutionOrder();
    EXPECT_THROW(appendPooledFreeOwnerWave(c.graph,c.arena,c.free,c.code,acquired,"free"),std::invalid_argument)<<capacity<<"/"<<defect;
    EXPECT_EQ(c.graph.getExecutionOrder(),before);
    EXPECT_EQ(c.graph.getNode("free/availability"),nullptr);}}

TEST(PooledImmutableOwnerPinChain, IntermediateForeignReturnOrMissingSendRejectBeforeMutation){for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<10;++defect){OwnerChainCase c(capacity,defect==0);auto* frontier=c.graph.getNode("returned/frontier_receive");
    if(defect==0)frontier->dependencies={"foreign/header_send"};
    if(defect==1)frontier->dependencies.clear();
    if(defect==2)c.graph.getNode("returned/seal_pins")->dependencies={"incoming/owner_begin"};
    if(defect==3)c.graph.getNode("incoming/owner_pin")->dependencies.clear();
    if(defect==4)c.graph.getNode("incoming/owner_begin")->dependencies.clear();
    if(defect==5)c.graph.getNode("incoming/owner_pin")->dependencies={"incoming/invocation_receive"};
    if(defect==6)c.graph.getNode("returned/header_send")->dependencies={"incoming/owner_pin"};
    if(defect==7)c.graph.getNode("returned/seal_pins")->dependencies.clear();
    if(defect==8){if(capacity)c.graph.getNode("incoming/pin_receive")->dependencies={"incoming/invocation_receive"};else c.graph.getNode("incoming/owner_begin")->device=DeviceId::cuda(0);}
    if(defect==9)c.graph.getNode("incoming/invocation_receive")->device=DeviceId::cuda(0);
    const auto before=c.graph.getExecutionOrder();
    EXPECT_THROW(appendPooledFreeOwnerWave(c.graph,c.arena,c.free,c.code,c.complete.last,"free"),std::invalid_argument)<<capacity<<"/"<<defect;
    EXPECT_EQ(c.graph.getExecutionOrder(),before);}}

TEST(PooledImmutableOwnerPinChain, VictimWaveConcreteOriginalFreeChainsPass)
{
    for(uint32_t capacity:{0u,3u}){OwnerChainCase c(capacity);const auto completed=appendPooledFreeOwnerWave(c.graph,c.arena,c.free,c.code,c.complete.last,"free");const auto victim=q::PooledImmutableCapturedVictimWave::bind(c.free,c.memory);
        EXPECT_NO_THROW(requirePooledFreeOwnerCompletion(c.graph,c.free,completed.last));
        const auto before=c.graph.getExecutionOrder().size();PooledPinParticipantNodes nodes;
        EXPECT_NO_THROW(nodes=appendPooledVictimOwnerWave(c.graph,c.arena,victim,c.code,completed.last,"victim"));
        EXPECT_EQ(nodes.last,"victim/receipt_send");EXPECT_GT(c.graph.getExecutionOrder().size(),before);EXPECT_EQ(c.graph.getNode("victim/frontier_receive"),nullptr);EXPECT_EQ(c.graph.getNode("victim/acquire_victims"),nullptr);
    }
}

TEST(PooledImmutableOwnerPinChain, VictimWaveRejectsMissingForeignAndOutOfOrderFreeChains)
{
    for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<8;++defect){OwnerChainCase c(capacity);auto actual=c.free;
        if(defect==1)actual=q::PooledImmutableCapturedFreeWave::bind(c.returned,c.memory);
        const auto completed=appendPooledFreeOwnerWave(c.graph,c.arena,actual,c.code,c.complete.last,"free");const auto victim=q::PooledImmutableCapturedVictimWave::bind(c.free,c.memory);auto acquired=completed.last;
        if(defect==0){acquired="manual_free";c.graph.addNode(acquired,std::make_unique<PooledImmutableNativePhaseStage>(c.arena,c.code,PooledImmutableNativePhaseStage::Owner{n::OwnerAction::AcquireFree,c.owner,q::PooledImmutableOwnerOperands(n::OwnerAction::AcquireFree,c.ownerPlan,c.free->ownerInputs(n::OwnerAction::AcquireFree))}),c.ownerPlan.allocator().device);}
        if(defect==2)c.graph.getNode("free/frontier_receive")->dependencies={"free/execute_free"};
        if(defect==3)c.graph.getNode("free/receipt_send")->dependencies={"free/execute_free"};
        if(defect==4)c.graph.getNode("free/accept_free")->dependencies={c.complete.last};
        if(defect==5)c.graph.getNode("free/availability")->dependencies={c.incoming.last};
        if(defect==6){if(capacity)c.graph.getNode("free/grant_send")->dependencies={"free/accept_free"};else c.graph.getNode("free/execute_free")->dependencies.clear();}
        if(defect==7)c.graph.getNode("free/plan_receive")->device=DeviceId::cuda(0);
        const auto before=c.graph.getExecutionOrder();
        EXPECT_THROW(appendPooledVictimOwnerWave(c.graph,c.arena,victim,c.code,acquired,"victim"),std::invalid_argument)<<capacity<<"/"<<defect;
        EXPECT_EQ(c.graph.getExecutionOrder(),before);EXPECT_EQ(c.graph.getNode("victim/plan_receive"),nullptr);
    }
}

TEST(PooledImmutableOwnerPinChain, VictimWaveConcreteCoordinatorFreeChainsPass)
{
    for(uint32_t capacity:{0u,3u}){VictimCoordinatorCase c(capacity);const std::array freeEdges{c.owner.free};
        EXPECT_NO_THROW(requirePooledFreeCoordinatorCompletion(c.graph,freeEdges,c.free.last));
        const auto victim=q::PooledImmutableCapturedVictimWave::bind(c.owner.free,c.owner.memory);const std::array edges{victim};PooledPinParticipantNodes nodes;
        EXPECT_NO_THROW(nodes=appendPooledVictimCoordinatorWave(c.graph,c.arena,edges,c.code,c.free.last,"victim"));
        EXPECT_EQ(nodes.last,"victim/complete_victims");EXPECT_EQ(c.graph.getNode("victim/owner0/frontier_send"),nullptr);
    }
}

TEST(PooledImmutableOwnerPinChain, VictimWaveCoordinatorRejectsForeignMissingAndOutOfOrderFreeChains)
{
    for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<6;++defect){VictimCoordinatorCase c(capacity);auto sent=c.free.last;const auto victim=q::PooledImmutableCapturedVictimWave::bind(c.owner.free,c.owner.memory);const std::array edges{victim};
        if(defect==0)sent="free/complete_free";
        if(defect==1)sent="free/owner0/plan_send";
        if(defect==2)c.graph.getNode(c.free.last)->dependencies.clear();
        if(defect==3)c.graph.getNode("free/complete_free")->dependencies={"free/owner0/plan_send"};
        if(defect==4)c.graph.getNode(capacity?"free/owner0/grant_receive":"free/owner0/receipt_receive")->dependencies.clear();
        if(defect==5){const auto other=q::PooledImmutableCapturedFreeWave::bind(c.owner.returned,c.owner.memory);const std::array others{other};sent=appendPooledFreeCoordinatorWave(c.graph,c.arena,others,c.code,c.pin.last,"other_free").last;}
        const auto before=c.graph.getExecutionOrder();
        EXPECT_THROW(appendPooledVictimCoordinatorWave(c.graph,c.arena,edges,c.code,sent,"victim"),std::invalid_argument)<<capacity<<"/"<<defect;
        EXPECT_EQ(c.graph.getExecutionOrder(),before);EXPECT_EQ(c.graph.getNode("victim/plan_victims"),nullptr);
    }
}

TEST(PooledImmutableOwnerPinChain, VictimWaveMultiOwnerOriginalFactoriesPass){victimMultiCohorts([](const auto& capacities,bool reversed){VictimMultiCoordinatorCase c(capacities,reversed);EXPECT_EQ(c.ownerPlans.size(),capacities.size());for(size_t i=0;i<capacities.size();++i)EXPECT_EQ(c.ownerPlans[i].allocator().device,DeviceId::rocm(static_cast<int>(reversed?capacities.size()-1-i:i)));
    EXPECT_NO_THROW(requirePooledFreeCoordinatorCompletion(c.graph,c.frees,c.free.last));
    const auto before=c.graph.getExecutionOrder().size();PooledPinParticipantNodes nodes;
    EXPECT_NO_THROW(nodes=appendPooledVictimCoordinatorWave(c.graph,c.arena,c.victims,c.code,c.free.last,"victim"));EXPECT_EQ(nodes.last,"victim/complete_victims");EXPECT_GT(c.graph.getExecutionOrder().size(),before);for(size_t i=0;i<capacities.size();++i)EXPECT_EQ(c.graph.getNode("victim/owner"+std::to_string(i)+"/frontier_send"),nullptr);
});}

TEST(PooledImmutableOwnerPinChain, VictimWaveMultiOwnerMissingPreviousReceiptRejectsBeforeMutation){victimMultiCohorts([](const auto& capacities,bool reversed){for(size_t owner=1;owner<capacities.size();++owner){VictimMultiCoordinatorCase c(capacities,reversed);auto& deps=c.read(owner)->dependencies;const auto prior="free/owner"+std::to_string(owner-1)+"/receipt_receive";const auto size=deps.size();std::erase(deps,prior);ASSERT_EQ(deps.size()+1,size);const auto before=c.graph.getExecutionOrder();
    EXPECT_THROW(appendPooledVictimCoordinatorWave(c.graph,c.arena,c.victims,c.code,c.free.last,"victim"),std::invalid_argument)<<owner;EXPECT_EQ(c.graph.getExecutionOrder(),before);EXPECT_EQ(c.graph.getNode("victim/plan_victims"),nullptr);}});}

TEST(PooledImmutableOwnerPinChain, VictimWaveMultiOwnerMissingAnyGlobalSenderRejectsBeforeMutation){victimMultiCohorts([](const auto& capacities,bool reversed){const size_t owner=capacities.size()-1;for(size_t producer=0;producer<capacities.size();++producer){VictimMultiCoordinatorCase c(capacities,reversed);auto& deps=c.read(owner)->dependencies;const auto size=deps.size();std::erase(deps,c.sender(producer));ASSERT_EQ(deps.size()+1,size);const auto before=c.graph.getExecutionOrder();
    EXPECT_THROW(appendPooledVictimCoordinatorWave(c.graph,c.arena,c.victims,c.code,c.free.last,"victim"),std::invalid_argument)<<owner<<"/"<<producer;EXPECT_EQ(c.graph.getExecutionOrder(),before);EXPECT_EQ(c.graph.getNode("victim/plan_victims"),nullptr);}});}

TEST(PooledImmutableOwnerPinChain, SourceFillActualWorkerOriginalGrantsAndSourceOnly){
std::vector<PooledOriginalRecipe> originalRecipes;
for(const auto capacities:{std::vector<uint32_t>{0},std::vector<uint32_t>{2,0},std::vector<uint32_t>{0,1,0,1},std::vector<uint32_t>{0,1,0,1,0,1,0,1}}){
originalRecipes.push_back(PooledOriginalRecipe{capacities,true});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(const auto capacities:{std::vector<uint32_t>{0},std::vector<uint32_t>{2,0},std::vector<uint32_t>{0,1,0,1},std::vector<uint32_t>{0,1,0,1,0,1,0,1}}){
auto c=programs.make(PooledOriginalRecipe{capacities,true});SourceWorkerTrace trace(c);const auto ticket=c.service->arm(17);const std::array<uint64_t,6> ids{19,7,19,3,5,7};trace.cold(ids,ticket.generation);trace.fill(ticket);

}}

TEST(PooledImmutableOwnerPinChain, SourceFillActualWorkerMixedSealCancellation){
std::vector<PooledOriginalRecipe> originalRecipes;
namespace source=q::pooled_source;
originalRecipes.push_back(PooledOriginalRecipe{std::vector<uint32_t>{2,1}});
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
namespace source=q::pooled_source;
auto c=programs.make(PooledOriginalRecipe{std::vector<uint32_t>{2,1}});SourceWorkerTrace trace(c);const auto ticket=c.service->arm(17);const std::array<uint64_t,4> ids{3,7,5,9};trace.cold(ids,ticket.generation);c.boundary->pooledKernels().admit(*c.boundary->pooledStorage(),ticket.generation,ticket.invocation,c.originalSourceStream);ASSERT_TRUE(source::planRequest(trace.source));trace.coordinator.input.source=&source::frontier(trace.source);trace.run<n::CoordinatorAction::SealSource>();trace.coordinator.input={};
    auto& sealed=trace.owners[0];sealed.input.frontier=&source::frontier(trace.source);SourceWorkerTrace::run<n::OwnerAction::SealSource>(sealed);ASSERT_TRUE(source::acquireSeal(trace.source,*n::at<n::OwnerSourceSealReceipt>(sealed.native,sealed.plan.native.header),0));
    auto& failed=trace.owners[1];SourceWorkerTrace::run<n::OwnerAction::CancelVictim>(failed);failed.input.frontier=&source::frontier(trace.source);EXPECT_FALSE((n::owner<n::OwnerAction::SealSource,q::adaptive_lookup::Slot32>(failed)));EXPECT_FALSE(source::acquireSeal(trace.source,*n::at<n::OwnerSourceSealReceipt>(failed.native,failed.plan.native.header),1));ASSERT_TRUE(c.service->cancel(ticket));EXPECT_FALSE(c.service->await(ticket));
    struct ActualWorkerPublications{q::ImmutableLookupAcknowledgement* ack;q::ImmutableLookupCompletion* complete;uint64_t retiredInvocation(){return std::atomic_ref(ack->retiredInvocation).load(std::memory_order_acquire);}uint64_t retiredGeneration(){return std::atomic_ref(ack->retiredGeneration).load(std::memory_order_acquire);}uint64_t completion(){return std::atomic_ref(complete->publication).load(std::memory_order_acquire);}} observed{trace.source.channel.base.acknowledgement,trace.source.channel.base.completion};
    ASSERT_TRUE(source::retireAborted(trace.source,observed));EXPECT_EQ(source::protocol(trace.source).issuedSequence,0u);EXPECT_EQ(source::protocol(trace.source).consumedSequence,0u);EXPECT_EQ(observed.completion(),0u);
    for(auto& a:trace.owners){a.input.frontier=&source::frontier(trace.source);a.input.condition=n::SourceCondition::Always;a.input.outcome=nullptr;EXPECT_TRUE((n::owner<n::OwnerAction::DrainSourceFailure,q::adaptive_lookup::Slot32>(a)));EXPECT_EQ(n::progress(a).header_bytes,sizeof(n::p::PublishHeader));EXPECT_NE(n::at<n::p::PublishHeader>(a.native,a.plan.native.header)->status,q::adaptive_lookup::Status::Success);// Allocation cancellation preserves original pin ownership until the explicit
        // later release phase. The source/fill handoff must not manufacture retirement.
        EXPECT_EQ(n::ownerView<q::adaptive_lookup::Slot32>(a).frame->held_pins,n::allocationFrame(a).count);}
    EXPECT_EQ(n::progress(sealed).retirement,n::p::SourceRetirement::Aborted);EXPECT_EQ(n::progress(failed).retirement,n::p::SourceRetirement::Unpublished);
    // Existing subsequent owner lifecycle is replayed only to prove actual reader
    // release; no distributed publication/Finish graph is claimed here.
    for(auto& a:trace.owners){a.input={a.input.invocation,nullptr,nullptr,nullptr,nullptr,nullptr};SourceWorkerTrace::run<n::OwnerAction::Release>(a);SourceWorkerTrace::run<n::OwnerAction::Retire>(a);EXPECT_EQ(n::ownerView<q::adaptive_lookup::Slot32>(a).frame->held_pins,0u);}


}

TEST(PooledImmutableOwnerPinChain, SourceFillOwnerRejectsBrokenOriginalVictimChainBeforeMutation){
std::vector<PooledOriginalRecipe> originalRecipes;
for(const auto capacity:{0u,3u})for(unsigned defect=0;defect<3;++defect){
originalRecipes.push_back(PooledOriginalRecipe{std::vector<uint32_t>{capacity,1}});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(const auto capacity:{0u,3u})for(unsigned defect=0;defect<3;++defect){
auto c=programs.make(PooledOriginalRecipe{std::vector<uint32_t>{capacity,1}});auto& graph=*c.ownerGraphs[0];auto* last=const_cast<ComputeNode*>(graph.getNode(c.ownerVictims[0].last));ASSERT_EQ(last->dependencies.size(),1u);if(defect==0)last->dependencies.clear();if(defect==1)last->dependencies[0]="free/acquire_free";if(defect==2)last->dependencies.push_back("pin/begin");const auto before=graph.getExecutionOrder();
    EXPECT_THROW(appendPooledSourceOwnerFill(graph,*c.ownerArenas[0],c.sources[0],c.ownerCode,c.ownerVictims[0].last,"source"),std::invalid_argument);EXPECT_EQ(graph.getExecutionOrder(),before);EXPECT_EQ(graph.getNode("source/initial_receive"),nullptr);

}}

TEST(PooledImmutableOwnerPinChain, SourceFillCoordinatorRejectsBrokenOriginalVictimChainBeforeMutation){
std::vector<PooledOriginalRecipe> originalRecipes;
for(const auto count:{2u,4u,8u})for(unsigned defect=0;defect<2;++defect){std::vector<uint32_t> capacities(count,1);capacities[0]=0;
originalRecipes.push_back(PooledOriginalRecipe{capacities,true});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(const auto count:{2u,4u,8u})for(unsigned defect=0;defect<2;++defect){std::vector<uint32_t> capacities(count,1);capacities[0]=0;
auto c=programs.make(PooledOriginalRecipe{capacities,true});const auto last=count-1;const auto name="victim/owner"+std::to_string(last)+"/grant_receive";auto* node=const_cast<ComputeNode*>(c.graph.getNode(name));ASSERT_NE(node,nullptr);auto& deps=node->dependencies;const auto original=deps.size();const auto missing=defect?"victim/owner0/plan_send":"victim/owner"+std::to_string(last-1)+"/receipt_receive";std::erase(deps,missing);ASSERT_EQ(deps.size()+1,original);const auto before=c.graph.getExecutionOrder();
    EXPECT_THROW(appendPooledSourceCoordinatorFill(c.graph,c.arena,c.sources,c.coordCode,*c.control,c.complete.last,"source"),std::invalid_argument);EXPECT_EQ(c.graph.getExecutionOrder(),before);EXPECT_EQ(c.graph.getNode("source/plan_source"),nullptr);

}}

TEST(PooledImmutableOwnerPinChain, SourceFillConcreteSharedPacketSendReadersRetireBeforeReuse){
std::vector<PooledOriginalRecipe> originalRecipes;

originalRecipes.push_back(PooledOriginalRecipe{std::vector<uint32_t>{2,0,3,1},true});
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));

auto c=programs.make(PooledOriginalRecipe{std::vector<uint32_t>{2,0,3,1},true});appendPooledSourceCoordinatorFill(c.graph,c.arena,c.sources,c.coordCode,*c.control,c.complete.last,"source");for(size_t i=1;i<c.sources.size();++i){const auto* packed=c.graph.getNode("source/owner"+std::to_string(i)+"/pack");ASSERT_NE(packed,nullptr);const auto expected="source/owner"+std::to_string(i-1)+(c.sources[i-1]->hasPayload()?"/payload_send":"/pack");EXPECT_EQ(packed->dependencies,std::vector<std::string>{expected});if(c.sources[i-1]->hasPayload())EXPECT_EQ(c.graph.getNode(expected)->dependencies,std::vector<std::string>{"source/owner"+std::to_string(i-1)+"/fill_send"});}EXPECT_EQ(c.graph.getNode("source/retire_response")->dependencies,std::vector<std::string>{"source/owner3/payload_send"});EXPECT_EQ(c.graph.getNode("source/owner0/terminal_send")->dependencies,std::vector<std::string>{"source/retire_response"});

}

TEST(PooledImmutableOwnerPinChain, SourceFillConcreteOriginalRemoteCohortsAndExtents){
std::vector<PooledOriginalRecipe> originalRecipes;
for(size_t count:{1u,2u,4u,8u})for(bool reverse:{false,true})for(bool empty:{false,true}){std::vector<uint32_t> capacities;for(size_t i=0;i<count;++i)capacities.push_back(empty || i%3==0?0:i+1);
originalRecipes.push_back(PooledOriginalRecipe{capacities,reverse});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(size_t count:{1u,2u,4u,8u})for(bool reverse:{false,true})for(bool empty:{false,true}){std::vector<uint32_t> capacities;for(size_t i=0;i<count;++i)capacities.push_back(empty || i%3==0?0:i+1);
auto c=programs.make(PooledOriginalRecipe{capacities,reverse});const auto co=appendPooledSourceCoordinatorFill(c.graph,c.arena,c.sources,c.coordCode,*c.control,c.complete.last,"source");EXPECT_EQ(co.last,"source/owner"+std::to_string(count-1)+"/terminal_send");for(size_t i=0;i<count;++i){const auto own=appendPooledSourceOwnerFill(*c.ownerGraphs[i],*c.ownerArenas[i],c.sources[i],c.ownerCode,c.ownerVictims[i].last,"source");EXPECT_EQ(own.last,"source/fill_outcome");const auto& p=c.owners[i]->plan().parameters();EXPECT_EQ(c.sources[i]->source(q::PooledSourceWireRole::InitialFrontier).bytes,64u);EXPECT_EQ(c.sources[i]->source(q::PooledSourceWireRole::SealReceipt).bytes,64u);EXPECT_EQ(c.sources[i]->source(q::PooledSourceWireRole::Fills).bytes,p.native.reply_capacity*96u);EXPECT_EQ(c.sources[i]->source(q::PooledSourceWireRole::Payload).bytes,p.native.reply_capacity*90u);EXPECT_NE(c.sources[i]->destination(q::PooledSourceWireRole::TerminalFrontier).offset,p.native.header);EXPECT_EQ(c.sources[i]->destination(q::PooledSourceWireRole::TerminalFrontier).offset,p.native.source_outcome);}

}
}

TEST(PooledImmutableOwnerPinChain, SourceFillIncompleteFactoryAndInvalidDirectOperationsRejectBeforeLaunch)
{
std::vector<PooledOriginalRecipe> originalRecipes;


originalRecipes.push_back(PooledOriginalRecipe{std::vector{2u,0u}});
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));


auto c=programs.make(PooledOriginalRecipe{std::vector{2u,0u}});
    const auto& code=dynamic_cast<const SourceGuardCode&>(c.boundary->pooledKernels());
    for(const auto action:{q::PooledSourceAction::Read,q::PooledSourceAction::BeforeReady})
        EXPECT_THROW(code.enqueue(action,*c.boundary->pooledStorage(),*c.control,{},
                                  c.originalSourceStream),std::invalid_argument);
    EXPECT_THROW(code.closeForReset(*c.boundary->pooledStorage(),*c.control,
                                   c.originalSourceStream),std::invalid_argument);
    EXPECT_THROW(c.boundary->beginAdmission(1,c.originalSourceStream),std::logic_error);
    EXPECT_EQ(code.launches,0u);
    EXPECT_EQ(code.closes,0u);
    EXPECT_FALSE(c.boundary->hasActiveInterval());


}

TEST(PooledImmutableOwnerPinChain, SourceTerminalImplementedEntriesReachNativeDispatch)
{
std::vector<PooledOriginalRecipe> originalRecipes;


originalRecipes.push_back(PooledOriginalRecipe{std::vector{2u,0u}});
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));


auto c=programs.make(PooledOriginalRecipe{std::vector{2u,0u}});
    const auto& code=dynamic_cast<const SourceGuardCode&>(c.boundary->pooledKernels());
    const auto expectEntry=[](auto&& call,const char* message) {
        try {call();FAIL()<<"Expected the isolated native code-entry double";}
        catch(const std::logic_error& error){EXPECT_STREQ(error.what(),message);}
    };
    for(const auto action:{q::PooledSourceAction::ReadersRetired,q::PooledSourceAction::ObserveRetired})
        expectEntry([&]{code.enqueue(action,*c.boundary->pooledStorage(),*c.control,{},
                                     c.originalSourceStream);},
                    "Unit source attempted native launch");
    expectEntry([&]{code.closeWhenTerminal(*c.boundary->pooledStorage(),*c.control,
                                          c.originalSourceStream);},
                "Unit source attempted native close");
    EXPECT_EQ(code.launches,2u);
    EXPECT_EQ(code.closes,1u);
    EXPECT_FALSE(c.boundary->hasActiveInterval());


}

TEST(PooledImmutableOwnerPinChain, MixedLocalPreSourceOriginalFactoriesPass)
{
    for(size_t count:{2u,4u,8u})for(size_t local=0;local<count;++local)for(bool empty:{false,true})for(bool reversed:{false,true}){
        std::vector<uint32_t> capacities(count);for(size_t i=0;i<count;++i)capacities[i]=empty || i%3==0?0:static_cast<uint32_t>(i+1);
        if(reversed)std::reverse(capacities.begin(),capacities.end());
        VictimMultiCoordinatorCase c(capacities,reversed,local);
        ASSERT_TRUE(c.localBinding);
        EXPECT_EQ(c.ownerPlans[local].allocator(),c.coordinatorPlan.allocator());
        EXPECT_FALSE(c.forwards[local]);
        EXPECT_FALSE(c.returns[local]);
        EXPECT_FALSE(c.frees[local]);
        EXPECT_FALSE(c.victims[local]);
        ASSERT_TRUE(c.mixedPin);
        ASSERT_TRUE(c.mixedFree);
        EXPECT_NO_THROW(requirePooledMixedPinCompletion(c.graph,*c.mixedPin));
        EXPECT_NO_THROW(requirePooledMixedFreeCompletion(c.graph,*c.mixedFree));
        const auto victim=appendPooledVictimCoordinatorMembers(c.graph,c.arena,c.mixedVictims,c.code,c.free.last,"mixed_victim",&*c.mixedFree);
        EXPECT_NO_THROW(requirePooledMixedVictimCompletion(c.graph,victim));
        EXPECT_EQ(victim.coordinator.last,"mixed_victim/complete_victims");
        EXPECT_NO_THROW((void)c.graph.getExecutionOrder());
    }
}

TEST(PooledImmutableOwnerPinChain, MixedLocalP1RetainsOriginalUnallocatedFrontier)
{
    const auto physical=VictimMultiCoordinatorCase::declared(std::vector<uint32_t>{3},false,0);
    const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},16);
    const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,701,17);
    const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,701,17);
    EXPECT_EQ(coordinator.parameters().native.ingress_frontier,0u);
    EXPECT_EQ(coordinator.parameters().native.ingress_count,0u);
    EXPECT_EQ(owner.parameters().native.ingress,0u);
    EXPECT_FALSE(q::PooledImmutablePinEdge::fromPlans(coordinator,owner).remote());
    EXPECT_THROW((VictimMultiCoordinatorCase(std::vector<uint32_t>{3},false,0)),std::invalid_argument);
}

TEST(PooledImmutableOwnerPinChain, MixedLocalMetadataAndFrontierDependenciesRejectBeforeMutation)
{
    for(size_t count:{2u,4u,8u})for(size_t local=0;local<count;++local)for(bool empty:{false,true})for(unsigned defect=0;defect<6;++defect){
        VictimMultiCoordinatorCase c(std::vector<uint32_t>(count,empty?0:3),false,local);
        std::string consumer,producer;
        if(defect==0){consumer=c.mixedPin->replies[local].headerReceived;producer=c.mixedPin->localSeal;}
        if(defect==1){consumer=c.mixedFree->replies[local].headerReceived;producer=c.mixedFree->localReceipt;}
        if(defect==2){consumer=c.mixedPin->localAcquire;producer=local?c.mixedPin->frontiers[local-1]:c.mixedPin->seal;}
        if(defect==3){consumer=c.mixedFree->localAcquire;producer=local?c.mixedFree->frontiers[local-1]:c.mixedFree->completed;}
        if(defect==4){consumer=c.mixedFree->localAvailability;producer=c.pin.last;}
        if(defect==5)c.graph.addDependency(c.mixedFree->localExecute,c.mixedFree->localAvailability);
        else ASSERT_EQ(std::erase(c.graph.getNode(consumer)->dependencies,producer),1u);
        const auto before=c.graph.getExecutionOrder();
        EXPECT_THROW(appendPooledVictimCoordinatorMembers(c.graph,c.arena,c.mixedVictims,c.code,c.free.last,"bad_metadata_victim",&*c.mixedFree),std::invalid_argument);
        EXPECT_EQ(c.graph.getExecutionOrder(),before);
        EXPECT_EQ(c.graph.getNode("bad_metadata_victim/plan_victims"),nullptr);
    }
}

TEST(PooledImmutableOwnerPinChain, MixedLocalOriginalOwnersSurviveCallerDropAndReleaseAllClaims)
{
    for(size_t count:{2u,4u,8u})for(size_t local:{size_t{0},count/2,count-1})for(bool empty:{false,true}){
        std::shared_ptr<PhysicalMemoryAuthority> memory;std::weak_ptr<TensorBase> coordinatorNative,ownerFrame;
        {
            auto c=std::make_unique<VictimMultiCoordinatorCase>(std::vector<uint32_t>(count,empty?0:3),false,local);
            auto victim=appendPooledVictimCoordinatorMembers(c->graph,c->arena,c->mixedVictims,c->code,c->free.last,"lifetime_victim",&*c->mixedFree);
            EXPECT_NO_THROW(requirePooledMixedVictimCompletion(c->graph,victim));
            memory=c->memory;coordinatorNative=c->coordinator->roots().native;ownerFrame=c->owners[local]->roots().frame;
            c->forwards.clear();c->returns.clear();c->frees.clear();c->victims.clear();c->owners.clear();c->coordinator.reset();c->localBinding.reset();
            c->mixedPin.reset();c->mixedFree.reset();c->mixedVictims.clear();victim={};
            EXPECT_FALSE(coordinatorNative.expired());
            EXPECT_FALSE(ownerFrame.expired());
            for(const auto& name:c->graph.getExecutionOrder())EXPECT_NO_THROW((void)c->graph.getNode(name)->stage->graphCaptureVariantSignature());
        }
        EXPECT_TRUE(coordinatorNative.expired());
            EXPECT_TRUE(ownerFrame.expired());
        for(const auto& line:memory->rankAttestation()){
            EXPECT_EQ(line.totalMaterializedNewBytes(),0u);
            EXPECT_EQ(line.totalCommittedNewBytes(),0u);
            EXPECT_EQ(line.adopted_resident_bytes,0u);
            EXPECT_EQ(line.native_declared_preclaimed_bytes,0u);
        }
    }
}

TEST(PooledImmutableOwnerPinChain, MixedLocalForeignAuthorityAndReorderedCohortReject)
{
    for(size_t count:{2u,4u,8u})for(size_t local=0;local<count;++local){
        VictimMultiCoordinatorCase c(std::vector<uint32_t>(count,3),false,local);const auto before=c.graph.getExecutionOrder();
        PhysicalMemoryPlanBuilder other;other.add(chainObserved(c.coordinatorPlan.allocator()),PhysicalMemoryOwner::ImmutableLookupStaging,1);
        auto foreign=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(other.build()),0);
        EXPECT_THROW(q::PooledImmutableLocalOwnerBinding::bind(c.coordinator,c.owners[local],foreign),std::logic_error);
        auto members=c.mixedPin->cohort->owners();std::swap(members[0],members[1]);
        EXPECT_THROW(q::PooledImmutableMixedCohort::fromOriginal(std::move(members)),std::invalid_argument);
        EXPECT_EQ(c.graph.getExecutionOrder(),before);
    }
}

TEST(PooledImmutableOwnerPinChain, MixedLocalMissingCommandReaderRejectsBeforeVictimMutation)
{
    for(size_t count:{2u,4u,8u})for(size_t local=0;local<count;++local)for(bool empty:{false,true}){
        std::vector<uint32_t> capacities(count,empty?0:3);VictimMultiCoordinatorCase c(capacities,false,local);
        const auto& last=c.mixedFree->replies.back();const bool direct=last.path==PooledPinReplyOwnerNodes::Path::DirectLocal;
        const auto read=direct && !last.replyAcquired.empty()?last.replyAcquired:!last.replyReceived.empty()?last.replyReceived:last.headerReceived;
        auto* node=c.graph.getNode(read);const auto removed=std::erase(node->dependencies,c.mixedFree->localAccept);
        ASSERT_EQ(removed,1u);
        const auto before=c.graph.getExecutionOrder();
        EXPECT_THROW(appendPooledVictimCoordinatorMembers(c.graph,c.arena,c.mixedVictims,c.code,c.free.last,"rejected_victim",&*c.mixedFree),std::invalid_argument);
        EXPECT_EQ(c.graph.getExecutionOrder(),before);
        EXPECT_EQ(c.graph.getNode("rejected_victim/plan_victims"),nullptr);
    }
}

TEST(PooledImmutableOwnerPinChain, SourceFillMixedOriginalCompletionAndLocalLastReaders){
std::vector<PooledOriginalRecipe> originalRecipes;
for(size_t count:{2u,4u,8u})for(uint32_t local=0;
local<count;
++local)for(bool zero:{false,true}){std::vector<uint32_t> caps(count);
for(size_t i=0;
i<count;
++i)caps[i]=zero || i%3==0?0:static_cast<uint32_t>(i+1);

originalRecipes.push_back(PooledOriginalRecipe{caps,true,local});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(size_t count:{2u,4u,8u})for(uint32_t local=0;
local<count;
++local)for(bool zero:{false,true}){std::vector<uint32_t> caps(count);
for(size_t i=0;
i<count;
++i)caps[i]=zero || i%3==0?0:static_cast<uint32_t>(i+1);

auto c=programs.make(PooledOriginalRecipe{caps,true,local});
const auto completed=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",&*c.mixedVictim);
EXPECT_NO_THROW(requirePooledSourceCoordinatorCompletion(c.graph,completed));
EXPECT_EQ(completed.coordinator.last,completed.localOutcome);
EXPECT_EQ(c.graph.getNode(completed.retired)->dependencies,std::vector<std::string>{completed.packed[local]});
EXPECT_EQ(c.graph.getNode(completed.localApply)->dependencies,std::vector<std::string>{completed.localDrain});
EXPECT_EQ(c.graph.getNode(completed.localValidate)->dependencies,std::vector<std::string>{completed.localApply});
for(size_t i=0;
i<count;
++i)if(i!=local){const auto own=appendPooledSourceOwnerFill(*c.ownerGraphs[i],*c.ownerArenas[i],c.sources[i],c.ownerCode,c.ownerVictims[i].last,"source");
EXPECT_NO_THROW(requirePooledSourceOwnerCompletion(*c.ownerGraphs[i],c.sources[i],own.last));
}EXPECT_EQ(c.boundary->pooledStorage()->geometry().parameters().local_owner,local);


}}

TEST(PooledImmutableOwnerPinChain, SourceFillCompletionRejectsBrokenOriginalProducer){
std::vector<PooledOriginalRecipe> originalRecipes;
for(bool local:{false,true})for(unsigned defect=0;
defect<4;
++defect){
originalRecipes.push_back(PooledOriginalRecipe{{2,0,3},true,local?1:q::adaptive_lookup::None});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(bool local:{false,true})for(unsigned defect=0;
defect<4;
++defect){
auto c=programs.make(PooledOriginalRecipe{{2,0,3},true,local?1:q::adaptive_lookup::None});
auto completed=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",c.mixedVictim?&*c.mixedVictim:nullptr);
ASSERT_NO_THROW(requirePooledSourceCoordinatorCompletion(c.graph,completed));
if(defect==0)c.graph.getNode(completed.retired)->dependencies.clear();
if(defect==1)c.graph.getNode(completed.sealAcquired[0])->dependencies={completed.sealed};
if(defect==2)c.graph.getNode(local?completed.localApply:completed.payloadSent[0])->dependencies={completed.acquired};
if(defect==3)completed.owners[0]=completed.owners[2];
EXPECT_THROW(requirePooledSourceCoordinatorCompletion(c.graph,completed),std::invalid_argument);


}}

TEST(PooledImmutableOwnerPinChain, PublicationReadyConcreteOriginalMixedAndRemoteCohorts){
std::vector<PooledOriginalRecipe> originalRecipes;
for(size_t count:{1u,2u,4u,8u})for(uint32_t local=0;local<=count;++local)for(bool zero:{false,true}){if(count==1 && local==0)continue;std::vector<uint32_t> caps(count);for(size_t i=0;i<count;++i)caps[i]=zero || i%3==0?0:static_cast<uint32_t>(i+1);
originalRecipes.push_back(PooledOriginalRecipe{caps,true,local==count?q::adaptive_lookup::None:local});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(size_t count:{1u,2u,4u,8u})for(uint32_t local=0;local<=count;++local)for(bool zero:{false,true}){if(count==1 && local==0)continue;std::vector<uint32_t> caps(count);for(size_t i=0;i<count;++i)caps[i]=zero || i%3==0?0:static_cast<uint32_t>(i+1);
auto c=programs.make(PooledOriginalRecipe{caps,true,local==count?q::adaptive_lookup::None:local});
    const auto fill=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",c.mixedVictim?&*c.mixedVictim:nullptr);auto members=publicationMembers(c);const auto ready=appendPooledPublicationCoordinator(c.graph,c.arena,fill,members,c.coordCode,"publication");
    EXPECT_NO_THROW(requirePooledPublicationCompletion(c.graph,ready));
    EXPECT_EQ(ready.coordinator.last,ready.ready);EXPECT_EQ(c.graph.getNode(ready.ready)->dependencies,std::vector<std::string>{ready.readyValidated});
    for(size_t i=0;i<count;++i)if(i!=c.localCoordinate){const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedPublicationWave>>(members[i]);for(auto role:{q::PooledPublicationWireRole::PublicationHeader,q::PooledPublicationWireRole::AcquiredHeader,q::PooledPublicationWireRole::ReadyHeader}){EXPECT_EQ(edge->source(role).bytes,60u);EXPECT_EQ(edge->destination(role).bytes,60u);}EXPECT_EQ(edge->source(q::PooledPublicationWireRole::PublicationFrontier).bytes,count*56u);EXPECT_EQ(edge->source(q::PooledPublicationWireRole::ReadyGate).bytes,64u);
        const auto ownerFill=appendPooledSourceOwnerFill(*c.ownerGraphs[i],*c.ownerArenas[i],c.sources[i],c.ownerCode,c.ownerVictims[i].last,"source");const auto ownerReady=appendPooledPublicationOwner(*c.ownerGraphs[i],*c.ownerArenas[i],edge,c.ownerCode,ownerFill.last,"publication");EXPECT_EQ(ownerReady.last,"publication/ready_send");
        EXPECT_NO_THROW(requirePooledPublicationOwnerCompletion(*c.ownerGraphs[i],edge,ownerReady.last));}
    if(c.localBinding){EXPECT_EQ(c.graph.getNode(ready.localAcquire)->dependencies,std::vector<std::string>{ready.publicationFrontierSent.back().empty()?ready.publicationFrontierSent[count-2]:ready.publicationFrontierSent.back()});EXPECT_EQ(c.graph.getNode(ready.localReadyReceipt)->dependencies,std::vector<std::string>{ready.localMakeReady});}


}}

TEST(PooledImmutableOwnerPinChain, PublicationReadyConcreteRejectsMissingOriginalReadersAndAcknowledgments){
std::vector<PooledOriginalRecipe> originalRecipes;
for(bool local:{false,true})for(unsigned defect=0;defect<6;++defect){
originalRecipes.push_back(PooledOriginalRecipe{{2,0,3,1},true,local?1:q::adaptive_lookup::None});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(bool local:{false,true})for(unsigned defect=0;defect<6;++defect){
auto c=programs.make(PooledOriginalRecipe{{2,0,3,1},true,local?1:q::adaptive_lookup::None});const auto fill=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",c.mixedVictim?&*c.mixedVictim:nullptr);auto members=publicationMembers(c);auto ready=appendPooledPublicationCoordinator(c.graph,c.arena,fill,members,c.coordCode,"publication");
    ASSERT_NO_THROW(requirePooledPublicationCompletion(c.graph,ready));
    if(defect==0)c.graph.getNode(ready.acquiredReceived[0])->dependencies={ready.publicationValidated};
    if(defect==1)c.graph.getNode(ready.acquiredValidated)->dependencies={ready.publicationValidated};
    if(defect==2)c.graph.getNode(ready.ready)->dependencies={ready.acquiredValidated};
    if(defect==3)c.graph.getNode(ready.readyImported[3])->dependencies={ready.readyReceived[0]};
    if(defect==4)ready.owners[0]=ready.owners[3];
    if(defect==5)c.graph.getNode(fill.retired)->dependencies={fill.acquired};
    const auto before=c.graph.getExecutionOrder();EXPECT_THROW(requirePooledPublicationCompletion(c.graph,ready),std::invalid_argument);EXPECT_EQ(c.graph.getExecutionOrder(),before);


}}

TEST(PooledImmutableOwnerPinChain, SourceFillMixedActualWorkerRetainsLocalPacketThroughConsumed){
std::vector<PooledOriginalRecipe> originalRecipes;
for(size_t count:{2u,4u,8u})for(uint32_t local=0;local<count;++local)for(bool c0:{false,true}){std::vector<uint32_t> capacities(count,1);if(c0)capacities[local]=0;
originalRecipes.push_back(PooledOriginalRecipe{capacities,true,local});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(size_t count:{2u,4u,8u})for(uint32_t local=0;local<count;++local)for(bool c0:{false,true}){std::vector<uint32_t> capacities(count,1);if(c0)capacities[local]=0;
auto c=programs.make(PooledOriginalRecipe{capacities,true,local});SourceWorkerTrace trace(c);const auto ticket=c.service->arm(17);const std::array<uint64_t,6> ids{19,7,19,3,5,7};trace.cold(ids,ticket.generation);trace.fill(ticket);EXPECT_EQ(q::pooled_source::stage(trace.source),q::pooled_source::ResponseStage::Consumed);

}}

TEST(PooledImmutableOwnerPinChain, PublicationReadyOwnerRejectsMissingActualAcquisition){
std::vector<PooledOriginalRecipe> originalRecipes;
for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<3;++defect){
originalRecipes.push_back(PooledOriginalRecipe{{capacity,1},true});}
PooledImmutableOriginalPrograms programs(std::move(originalRecipes));
for(uint32_t capacity:{0u,3u})for(unsigned defect=0;defect<3;++defect){
auto c=programs.make(PooledOriginalRecipe{{capacity,1},true});const auto members=publicationMembers(c);const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedPublicationWave>>(members[0]);auto& graph=*c.ownerGraphs[0];const auto fill=appendPooledSourceOwnerFill(graph,*c.ownerArenas[0],c.sources[0],c.ownerCode,c.ownerVictims[0].last,"source");const auto ready=appendPooledPublicationOwner(graph,*c.ownerArenas[0],edge,c.ownerCode,fill.last,"publication");
    ASSERT_NO_THROW(requirePooledPublicationOwnerCompletion(graph,edge,ready.last));
    if(defect==0)graph.getNode("publication/acquire_publication")->dependencies.clear();
    if(defect==1)graph.getNode("publication/ready_receipt")->dependencies={"publication/publication_frontier"};
    if(defect==2)graph.getNode("source/validate_fills")->dependencies={"source/terminal_receive"};
    const auto before=graph.getExecutionOrder();EXPECT_THROW(requirePooledPublicationOwnerCompletion(graph,edge,ready.last),std::invalid_argument);EXPECT_EQ(graph.getExecutionOrder(),before);


}}

TEST(PooledImmutableOwnerPinChain, RecordedOrderingRejectsPublicOrdinaryAndForeignClaimsBeforeCreators){
    for(const uint32_t capacity:{0u,3u}){
        OwnerChainBackendScope backends;const auto physical=chainGeometry(capacity);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},16);
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,701,17);
        const auto plan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,701,17);
        const auto memory=chainAuthority(physical,policy,coordinator,plan,1,false);auto* stream=reinterpret_cast<void*>(0x202);
        (void)GPUDeviceContextPool::instance().getContext(plan.allocator().device);
        try{(void)q::PooledImmutableOwnerPreparedStorage::create(plan,memory,stream);FAIL()<<"Ordinary worker manufactured a prepared source";}
        catch(const std::logic_error& error){EXPECT_NE(std::string(error.what()).find("No live original prepared source worker"),std::string::npos);}
        EXPECT_EQ(backends.rocm.allocationCalls,0u);EXPECT_EQ(backends.rocm.eventCreates,0u);
        for(unsigned defect=0;defect<(capacity?4u:3u);++defect){
            auto group=PooledImmutableFactoryReplayAccess::claim(memory,plan.allocator().device,
                defect==2?coordinator.orderingProgram():defect==3?NativeOrderingEventProgram::pooledOwner(false):plan.orderingProgram());
            auto target=memory;
            if(defect==0)target=std::make_shared<PhysicalMemoryAuthority>(memory->admission(),0);
            if(defect==1)target=std::shared_ptr<PhysicalMemoryAuthority>(memory.get(),[](PhysicalMemoryAuthority*){});
            EXPECT_THROW((void)PooledImmutableFactoryReplayAccess::ownerClaim(plan,target,stream,std::move(group)),std::logic_error);
            EXPECT_EQ(backends.rocm.allocationCalls,0u);EXPECT_EQ(backends.rocm.eventCreates,0u);
            EXPECT_EQ(memory->reservedBytes(plan.allocator().device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
        }
    }
}

TEST(PooledImmutableOwnerPinChain, RecordedOrderingPartialCreationAndFinalAliasKeepOneAuthority){
    for(const uint32_t capacity:{0u,3u}){
        OwnerChainBackendScope backends;const auto physical=chainGeometry(capacity);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},16);
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,701,17);
        const auto plan=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,701,17);
        const auto memory=chainAuthority(physical,policy,coordinator,plan,1,false);auto* stream=reinterpret_cast<void*>(0x202);
        for(unsigned prefix=1;prefix<=(capacity?3u:2u);++prefix){
            backends.rocm.failAllocationCall=backends.rocm.allocationCalls+prefix;
            EXPECT_THROW((void)PooledImmutableFactoryReplayAccess::owner(plan,memory,stream),std::exception);
            EXPECT_EQ(backends.rocm.liveAllocations(),0u);EXPECT_EQ(backends.rocm.eventCreates,0u);
            EXPECT_EQ(memory->reservedBytes(plan.allocator().device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
        }
        backends.rocm.failAllocationCall.reset();
        auto owner=PooledImmutableFactoryReplayAccess::owner(plan,memory,stream);auto alias=owner->roots().native;
        const auto live=backends.rocm.liveAllocations();ASSERT_EQ(live,capacity?3u:2u);
        owner.reset();EXPECT_EQ(backends.rocm.liveAllocations(),live);
        EXPECT_GT(memory->reservedBytes(plan.allocator().device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
        EXPECT_THROW((void)GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(plan.allocator().device),std::logic_error);
        alias.reset();EXPECT_EQ(backends.rocm.liveAllocations(),0u);
        EXPECT_EQ(memory->reservedBytes(plan.allocator().device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
    }
}
}

namespace llaminar2::test {
/** @test Actual original source/root leases partition setup capacities once, independent of request-data replay. */
TEST(PooledImmutableOwnerPinChain, ImmutableSetupCapacityUsesOriginalDisjointRootsAndRejectsPartialSources) {
    struct CollectorScope {
        std::optional<std::string> enabled,filter;
        CollectorScope() {
            if(const auto* old=std::getenv("LLAMINAR_PERF_STATS_JSON"))enabled=old;
            if(const auto* old=std::getenv("LLAMINAR_PERF_STATS_FILTER"))filter=old;
            ::setenv("LLAMINAR_PERF_STATS_JSON","1",1);::setenv("LLAMINAR_PERF_STATS_FILTER","immutable_lookup_setup.*",1);
            PerfStatsCollector::reloadConfigurationFromEnvironment();PerfStatsCollector::reset();
        }
        ~CollectorScope() {
            if(enabled)::setenv("LLAMINAR_PERF_STATS_JSON",enabled->c_str(),1);else ::unsetenv("LLAMINAR_PERF_STATS_JSON");
            if(filter)::setenv("LLAMINAR_PERF_STATS_FILTER",filter->c_str(),1);else ::unsetenv("LLAMINAR_PERF_STATS_FILTER");
            PerfStatsCollector::reloadConfigurationFromEnvironment();PerfStatsCollector::reset();
        }
    } collector;
    const std::vector<PooledOriginalRecipe> recipes{{{0},false},{{2,0},true},{{2,1,0,3},false,0}};
    PooledImmutableOriginalPrograms programs(recipes);
    for(const auto& recipe:recipes) {
        auto c=programs.make(recipe);
        const std::array<std::shared_ptr<const q::PooledImmutableSourcePreparedStorage>,1> sources{c.boundary->pooledStorage()};
        const auto capture=[&](const auto& population) {
            return ImmutableLookupSetupObservation::captureAdaptive(c.layout,c.cpuGeometry,c.service,population);
        };
        const auto observed=capture(sources);
        PerfStatsCollector::reset();observed.publish();
        const auto records=PerfStatsCollector::snapshot({"immutable_lookup_setup"});ASSERT_EQ(records.size(),observed.owners().size());
        for(const auto& record:records){EXPECT_EQ(record.name,"admitted_owner");EXPECT_EQ(record.count,1u);EXPECT_EQ(record.value,1);EXPECT_EQ(record.tags.at("source_row_bytes"),std::to_string(c.file.descriptor().rowBytes()));}
        PerfStatsCollector::addCounter("immutable_lookup_setup","unrelated_warmup",7,"warmup","CPU");
        PerfStatsCollector::resetPreserving({},{{"immutable_lookup_setup","admitted_owner"}});
        const auto retained=PerfStatsCollector::snapshot({"immutable_lookup_setup"});EXPECT_EQ(retained.size(),records.size());
        std::size_t hostRows=0,cacheRows=0,tableBytes=0,stagingBytes=0;
        for(const auto& row:observed.owners()) {
            EXPECT_EQ(row.table_storage_bytes,row.source_storage_bytes+row.payload_bytes+row.metadata_bytes);
            tableBytes+=row.table_storage_bytes;stagingBytes+=row.lookup_staging_bytes;
            if(row.role==ImmutableLookupSetupObservation::Role::SharedHostSource) {
                ++hostRows;EXPECT_EQ(row.allocator.device,DeviceId::cpu());
                EXPECT_EQ(row.source_storage_bytes,c.cpuGeometry.sourceBytes());
                EXPECT_EQ(row.admitted_slots,c.cpuGeometry.slots());
            }
            if(row.role==ImmutableLookupSetupObservation::Role::GPUCache) {
                ++cacheRows;
                const auto owner=std::find_if(c.owners.begin(),c.owners.end(),[&](const auto& owner){return owner->plan().allocator()==row.allocator;});
                ASSERT_NE(owner,c.owners.end());EXPECT_EQ(row.admitted_slots,(*owner)->plan().parameters().owner.capacity());
                EXPECT_EQ(row.table_storage_bytes,(*owner)->retainedOwnerObservation().tableStorageBytes());
            }
        }
        EXPECT_EQ(hostRows,1u);EXPECT_EQ(cacheRows,c.owners.size());
        std::size_t actualTable=0,actualStaging=0;
        for(const auto& lease:c.memory->rankAttestation()) {
            if(lease.owner==PhysicalMemoryOwner::ImmutableTableStorage)actualTable+=lease.materialized_new_bytes+lease.adopted_resident_bytes;
            if(lease.owner==PhysicalMemoryOwner::ImmutableLookupStaging)actualStaging+=lease.materialized_new_bytes+lease.adopted_resident_bytes;
        }
        EXPECT_EQ(tableBytes,actualTable);
        // Fixture query/control tensors are other original staging owners; setup reports only lookup roots/service.
        EXPECT_LE(stagingBytes,actualStaging);
        for(unsigned replay=0;replay<20;++replay)EXPECT_TRUE(observed.sameCapacity(capture(sources)));
        const std::span<const std::shared_ptr<const q::PooledImmutableSourcePreparedStorage>> missing;
        EXPECT_THROW(capture(missing),std::invalid_argument);
        const std::array<std::shared_ptr<const q::PooledImmutableSourcePreparedStorage>,1> null{nullptr};
        EXPECT_THROW(capture(null),std::invalid_argument);
        const std::array duplicate{sources[0],sources[0]};EXPECT_THROW(capture(duplicate),std::invalid_argument);
        EXPECT_THROW(ImmutableLookupSetupObservation::captureAdaptive(c.layout,c.cpuGeometry,{},sources),std::invalid_argument);
        auto foreignCapacities=recipe.capacities;++foreignCapacities[0];
        const auto foreign=SourceGraphCase::declare(c.file.descriptor(),foreignCapacities,recipe.reverse,recipe.local,recipe.source);
        EXPECT_THROW(ImmutableLookupSetupObservation::captureAdaptive(foreign,c.cpuGeometry,c.service,sources),std::invalid_argument);
        const auto host=ImmutableLookupSetupObservation::hostOwner(c.cpuGeometry,c.service);
        EXPECT_EQ(host.source_storage_bytes,c.cpuGeometry.sourceBytes());
        // The emitted historical value survives request resets; a changed actual
        // source cannot issue a fresh observation or replace its provenance.
        {std::ofstream changed(c.file.path,std::ios::binary|std::ios::app);changed.put('x');}
        EXPECT_THROW(capture(sources),std::exception);
        EXPECT_TRUE(observed.sameCapacity(observed));


    }
}
} // namespace llaminar2::test

namespace llaminar2::test {
namespace {
/** @brief Independent same-device trunk marker; only original protocol nodes claim source/native effects. */
class CurrentDemandTrunk final:public IComputeStage {
public:
    explicit CurrentDemandTrunk(DeviceId device):IComputeStage(device){}
    bool execute(IDeviceContext*)override{return true;}
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_SOURCE_PHASE;}
    bool supportsBackend(ComputeBackendType)const override{return false;}
    StageBufferRequirements getBufferRequirements()const override{return {};}
    StageBufferContract bufferContract()const override{return {};}
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
};
/** @brief Declare two ordinary trunk nodes after the actual protocol prefix, preserving order. */
void declareDemandTrunk(ComputeGraph& graph,DeviceId device,const std::string& prefix){
    graph.addNode("trunk/root",std::make_unique<CurrentDemandTrunk>(device),device);
    graph.addNode("trunk/terminal",std::make_unique<CurrentDemandTrunk>(device),device);
    graph.addDependency("trunk/root",prefix);graph.addDependency("trunk/terminal","trunk/root");
}
/** @return Actual topological position, independent of node insertion order. */
std::size_t demandPosition(const ComputeGraph& graph,const std::string& name){
    const auto& order=graph.getExecutionOrder();return std::find(order.begin(),order.end(),name)-order.begin();
}
}
/** @test Current source demand issues once before trunk, consumes late and retains exact original source lifecycle. */
TEST(PooledImmutableCurrentDemand, SourcePublishedBeforeTrunkAndResponseAfterTrunkLifecycle){
    std::vector<PooledOriginalRecipe> recipes;
    for(const bool zero:{true,false})recipes.push_back(PooledOriginalRecipe{zero?std::vector<uint32_t>{0,0}:std::vector<uint32_t>{1,3},true});
    PooledImmutableOriginalPrograms programs(recipes);
    for(const auto& recipe:recipes){
        auto c=programs.make(recipe);
        auto source=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",nullptr);
        const auto ready=appendPooledPublicationCoordinator(c.graph,c.arena,source,publicationMembers(c),c.coordCode,"publication");
        std::vector<q::PooledImmutableReadParticipant> readers;
        for(const auto& member:ready.owners)readers.emplace_back(q::PooledImmutableCapturedReadWave::bind(std::get<std::shared_ptr<const q::PooledImmutableCapturedPublicationWave>>(member),c.memory));
        const auto read=appendPooledReadCoordinator(c.graph,c.arena,ready,readers,
            {c.readOutput,BufferId::QFN_PLE_EMBEDDING,0,16*160*sizeof(float)},"read","original_output");
        std::vector<q::PooledImmutableRetirementParticipant> retiring;
        for(const auto& member:read.owners)retiring.emplace_back(q::PooledImmutableCapturedRetirementWave::bind(std::get<std::shared_ptr<const q::PooledImmutableCapturedReadWave>>(member),c.memory));
        const auto retired=appendPooledRetirementCoordinator(c.graph,c.arena,read,retiring,c.coordCode,"retirement");
        c.graph.setRetainedStorageBoundary(c.boundary);
        PooledImmutableDemandFrontier frontier{PooledImmutableDemandRole::Source,source.published,source.acquired};
        PooledImmutableDemandHandoff handoff(c.graph,frontier,71,91);
        EXPECT_THROW(handoff.requireConsumed(),std::logic_error);
        declareDemandTrunk(c.graph,c.boundary->device(),source.published);
        EXPECT_THROW(handoff.consume(c.graph,72,91,{"trunk/terminal"}),std::invalid_argument);
        EXPECT_THROW(handoff.consume(c.graph,71,92,{"trunk/terminal"}),std::invalid_argument);
        ComputeGraph foreign;
        EXPECT_THROW(handoff.consume(foreign,71,91,{"trunk/terminal"}),std::invalid_argument);
        EXPECT_THROW(handoff.consume(c.graph,71,91,{source.published}),std::invalid_argument);
        {GraphCaptureGuard capture;EXPECT_THROW(handoff.consume(c.graph,71,91,{"trunk/terminal"}),std::logic_error);}
        const auto beforeSuffixJoin=c.graph.topologyGeneration();
        const auto beforeSuffixOrder=c.graph.getExecutionOrder();
        EXPECT_THROW(handoff.consume(c.graph,71,91,{retired.coordinator.last}),std::invalid_argument);
        EXPECT_EQ(c.graph.topologyGeneration(),beforeSuffixJoin);EXPECT_EQ(c.graph.getExecutionOrder(),beforeSuffixOrder);
        // A second ancestry branch reaches response even though the first reaches Publish.
        c.graph.addNode("trunk/bad_join",std::make_unique<CurrentDemandTrunk>(c.boundary->device()),c.boundary->device());
        c.graph.addDependency("trunk/bad_join",source.published);c.graph.addDependency("trunk/bad_join",source.acquired);
        const auto beforeMixedJoin=c.graph.topologyGeneration();
        EXPECT_THROW(handoff.consume(c.graph,71,91,{"trunk/bad_join"}),std::invalid_argument);
        EXPECT_EQ(c.graph.topologyGeneration(),beforeMixedJoin);
        EXPECT_NO_THROW(handoff.consume(c.graph,71,91,{"trunk/terminal"}));
        EXPECT_NO_THROW(handoff.requireConsumed());
        EXPECT_THROW(handoff.consume(c.graph,71,91,{"trunk/terminal"}),std::invalid_argument);
        EXPECT_LT(demandPosition(c.graph,source.published),demandPosition(c.graph,"trunk/root"));
        EXPECT_LT(demandPosition(c.graph,"trunk/terminal"),demandPosition(c.graph,source.acquired));
        EXPECT_NO_THROW(requirePooledSourceCoordinatorCompletion(c.graph,source));
        EXPECT_NO_THROW(requirePooledRetirementCompletion(c.graph,retired));
        const auto order=c.graph.getExecutionOrder();
        for(int replay=0;replay<20;++replay){c.graph.reset();EXPECT_EQ(c.graph.getExecutionOrder(),order);EXPECT_NO_THROW(handoff.requireConsumed());}
        auto* response=c.graph.getNode(source.acquired);
        response->dependencies.push_back("trunk/root");
        EXPECT_THROW(requirePooledRetirementCompletion(c.graph,retired),std::invalid_argument);
        response->dependencies.pop_back();
        response->dependencies.erase(std::remove(response->dependencies.begin(),response->dependencies.end(),source.published),response->dependencies.end());
        EXPECT_THROW(requirePooledSourceCoordinatorCompletion(c.graph,source),std::invalid_argument);
        EXPECT_THROW(requirePooledRetirementCompletion(c.graph,retired),std::invalid_argument);
        response->dependencies.push_back(source.sealed);
        EXPECT_THROW(requirePooledRetirementCompletion(c.graph,retired),std::invalid_argument);
    }
}
/** @test The actual addressed owner stops at seal-send, including C0, with exactly one original protocol predecessor. */
TEST(PooledImmutableCurrentDemand, InlineSealBeforeTrunkFirstResponseAfterTrunkPreservesOriginalProof){
    const std::vector<PooledOriginalRecipe> recipes{{{0,0},true},{{1,3},true}};
    PooledImmutableOriginalPrograms programs(recipes);
    for(const auto& recipe:recipes){auto c=programs.make(recipe);
        for(std::size_t i=0;i<c.owners.size();++i){
            auto& graph=*c.ownerGraphs[i];
            PooledImmutableDemandFrontier frontier;
            const auto source=appendPooledSourceOwnerFill(graph,*c.ownerArenas[i],c.sources[i],c.ownerCode,c.ownerVictims[i].last,"source",&frontier);
            graph.setRetainedParticipantDefinitionDomain({c.boundary->device(),c.sources[i].get()});
            PooledImmutableDemandHandoff handoff(graph,frontier,71,91);
            const bool positive=c.sources[i]->hasPayload();
            EXPECT_EQ(frontier.published,"source/seal_receipt_send");
            EXPECT_EQ(frontier.consume,positive?"source/fill_receive":"source/terminal_receive");
            declareDemandTrunk(graph,c.owners[i]->plan().allocator().device,frontier.published);
            const auto beforeSuffixJoin=graph.topologyGeneration();const auto beforeSuffixOrder=graph.getExecutionOrder();
            EXPECT_THROW(handoff.consume(graph,71,91,{source.last}),std::invalid_argument);
            EXPECT_EQ(graph.topologyGeneration(),beforeSuffixJoin);EXPECT_EQ(graph.getExecutionOrder(),beforeSuffixOrder);
            EXPECT_NO_THROW(handoff.consume(graph,71,91,{"trunk/terminal"}));
            EXPECT_NO_THROW(requirePooledSourceOwnerCompletion(graph,c.sources[i],source.last));
            EXPECT_LT(demandPosition(graph,frontier.published),demandPosition(graph,"trunk/root"));
            EXPECT_LT(demandPosition(graph,"trunk/terminal"),demandPosition(graph,frontier.consume));
            const auto order=graph.getExecutionOrder();for(int replay=0;replay<20;++replay){graph.reset();EXPECT_EQ(graph.getExecutionOrder(),order);}
            auto* response=graph.getNode(frontier.consume);
            response->dependencies.push_back("trunk/root");
            EXPECT_THROW(requirePooledSourceOwnerCompletion(graph,c.sources[i],source.last),std::invalid_argument);
            response->dependencies.pop_back();
            response->dependencies.erase(std::remove(response->dependencies.begin(),response->dependencies.end(),"trunk/terminal"),response->dependencies.end());
            EXPECT_THROW(requirePooledSourceOwnerCompletion(graph,c.sources[i],source.last),std::invalid_argument);
        }
    }
}
/** @test Foreign/pure-reader descriptors cannot turn an owner/source node into another protocol role. */
TEST(PooledImmutableCurrentDemand, ActualEndpointRolesRejectForgedPureReaderAndWrongCut){
    const PooledOriginalRecipe recipe{{0,1},true};PooledImmutableOriginalPrograms programs({recipe});auto c=programs.make(recipe);
    const auto source=appendPooledSourceCoordinatorMembers(c.graph,c.arena,c.sourceMembers,c.boundary,c.coordCode,*c.control,c.complete.last,"source",nullptr);
    c.graph.setRetainedStorageBoundary(c.boundary);
    EXPECT_THROW((PooledImmutableDemandHandoff(c.graph,{PooledImmutableDemandRole::Source,source.planned,source.acquired},71,91)),std::invalid_argument);
    EXPECT_THROW((PooledImmutableDemandHandoff(c.graph,{PooledImmutableDemandRole::PureReader,{},source.acquired},71,91)),std::invalid_argument);
    EXPECT_THROW((PooledImmutableDemandHandoff(c.graph,{PooledImmutableDemandRole::Source,source.published,source.acquired},0,91)),std::invalid_argument);
}
}

namespace llaminar2::test {
/** @test Real factory allocations may differ while complete source/query geometry stays identical. */
TEST(PooledImmutableCurrentDemand, RealFactoryRegenerationConsumesOriginalFrameAndRejectsPrecisionDrift){
    const PooledOriginalRecipe recipe{{3},false,0,true,DeviceType::CUDA,16};
    PooledImmutableOriginalPrograms programs({recipe});auto c=programs.make(recipe);
    ASSERT_TRUE(c.queryTokens);ASSERT_TRUE(c.queryPrior);
    const auto& declared=c.boundary->pooledStorage()->geometry().parameters();
    Qwen4ExpPLEContext context{.device=DeviceId::cuda(0),.prefix="L1_ple_",.rows={1},.hidden=1,.epsilon=1e-6f,
        .nativeFormat=declared.format,.nativeRowElements=static_cast<int>(declared.row_elements),.nativeRowBytes=declared.owner.row_bytes};
    context.hash.multipliers={9007199254740993ULL,9007199254740995ULL,9007199254740997ULL};
    for(int head=0;head<16;++head){context.hash.moduli[head]=7;context.hash.offsets[head]=7*head;}
    const Qwen4ExpFloatBinding output(*c.readOutput,BufferId::QFN_PLE_EMBEDDING);
    const Qwen4ExpPLEBuffers buffers{
        Qwen4ExpNativeIntInput{context.device,static_cast<const std::int32_t*>(c.queryTokens->gpu_data_ptr()),1},
        Qwen4ExpIntBinding(*c.queryPrior,BufferId::QFN_PLE_TOKEN_HISTORY),Qwen4ExpIntBinding(*c.ids,BufferId::QFN_NGRAM_ROW_IDS),
        output,output,output,output,output,output,output,output};
    std::shared_ptr<const q::IPooledImmutableNativeKernels> first=q::createCUDAPooledImmutableNativeKernels();
    std::shared_ptr<const q::IPooledImmutableNativeKernels> second=q::createCUDAPooledImmutableNativeKernels();
    ASSERT_TRUE(first);ASSERT_TRUE(second);EXPECT_NE(first.get(),second.get());EXPECT_EQ(typeid(*first),typeid(*second));
    Qwen4ExpPooledLookupStorage storage{&c.arena,c.boundary,first,*c.control,DeviceRowRange::fullyActive(1),{}, {}};
    ComputeGraph graph;
    auto frame=Qwen4ExpPooledReadAheadFrame::publish(graph,context,buffers,storage);
    ASSERT_TRUE(graph.getNode("L1_ple_hash"));
    EXPECT_THROW(frame->requireConsumed(),std::logic_error);
    declareDemandTrunk(graph,context.device,frame->producer());
    const auto hash=graph.getNode("L1_ple_hash")->stage->graphCaptureVariantSignature();
    auto regenerated=storage;regenerated.code=second;
    auto changed=context;changed.nativeRowBytes++;
    EXPECT_THROW(frame->consume(graph,changed,buffers,regenerated,hash,{"trunk/terminal"}),std::invalid_argument);
    changed=context;changed.nativeRowElements++;
    EXPECT_THROW(frame->consume(graph,changed,buffers,regenerated,hash,{"trunk/terminal"}),std::invalid_argument);
    changed=context;changed.nativeFormat=TensorType::FP32;
    EXPECT_THROW(frame->consume(graph,changed,buffers,regenerated,hash,{"trunk/terminal"}),std::invalid_argument);
    auto foreign=regenerated;foreign.boundary.reset();
    EXPECT_THROW(frame->consume(graph,context,buffers,foreign,hash,{"trunk/terminal"}),std::invalid_argument);
    EXPECT_NO_THROW(frame->consume(graph,context,buffers,regenerated,hash,{"trunk/terminal"}));
    EXPECT_NO_THROW(frame->requireConsumed());
    EXPECT_THROW(frame->consume(graph,context,buffers,regenerated,hash,{"trunk/terminal"}),std::invalid_argument);
    EXPECT_LT(demandPosition(graph,"trunk/terminal"),demandPosition(graph,"L1_ple_gather"));
}
}

namespace llaminar2::test {
/** @test Counted prefill retains original count, sixteen-head expansion and exact graph identity across partial/empty reset records. */
TEST(PooledImmutableCurrentDemand, CountedPrefillKeepsOriginalLiveExtentAuthorityAcrossResetRecords){
    const PooledOriginalRecipe recipe{{3},false,0,true,DeviceType::CUDA,64};
    PooledImmutableOriginalPrograms programs({recipe});auto c=programs.make(recipe);
    ASSERT_TRUE(c.queryLive);const auto device=c.boundary->device();
    auto* live=static_cast<std::int32_t*>(c.queryLive->gpu_data_ptr());ASSERT_NE(live,nullptr);
    const Qwen4ExpNativeIntInput liveInput{device,live,1};
    const auto& declared=c.boundary->pooledStorage()->geometry().parameters();
    Qwen4ExpPLEContext context{.device=device,.prefix="L1_ple_",.rows={4,liveInput},.hidden=1,.epsilon=1e-6f,
        .nativeFormat=declared.format,.nativeRowElements=static_cast<int>(declared.row_elements),.nativeRowBytes=declared.owner.row_bytes};
    context.hash.multipliers={9007199254740993ULL,9007199254740995ULL,9007199254740997ULL};
    for(int head=0;head<16;++head){context.hash.moduli[head]=7;context.hash.offsets[head]=7*head;}
    const Qwen4ExpFloatBinding output(*c.readOutput,BufferId::QFN_PLE_EMBEDDING);
    const Qwen4ExpPLEBuffers buffers{
        Qwen4ExpNativeIntInput{device,static_cast<const std::int32_t*>(c.queryTokens->gpu_data_ptr()),4},
        Qwen4ExpIntBinding(*c.queryPrior,BufferId::QFN_PLE_TOKEN_HISTORY),Qwen4ExpIntBinding(*c.ids,BufferId::QFN_NGRAM_ROW_IDS),
        output,output,output,output,output,output,output,output};
    const auto rows=DeviceRowRange::deviceCounted(4,live);
    const Qwen4ExpPooledLookupStorage storage{&c.arena,c.boundary,c.coordCode,*c.control,rows,
        Qwen4ExpIntBinding(*c.queryLive,BufferId::PREFILL_CHUNK_GEOMETRY),{}};
    ComputeGraph graph;auto frame=Qwen4ExpPooledReadAheadFrame::publish(graph,context,buffers,storage);
    declareDemandTrunk(graph,device,frame->producer());
    const auto hash=graph.getNode("L1_ple_hash")->stage->graphCaptureVariantSignature();
    EXPECT_NO_THROW(frame->consume(graph,context,buffers,storage,hash,{"trunk/terminal"}));
    const PooledImmutableNativePhaseStage::Coordinator* begin=nullptr;
    for(const auto& name:graph.getExecutionOrder()){
        const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(graph.getNode(name)->stage.get());
        if(!stage)continue;const auto* candidate=std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase());
        if(candidate && candidate->action==n::CoordinatorAction::Begin){ASSERT_EQ(begin,nullptr);begin=candidate;}
    }
    ASSERT_NE(begin,nullptr);
    const q::PooledImmutableCoordinatorOperands expected(n::CoordinatorAction::Begin,c.coordinator->plan(),
        n::OriginalRows{rows.repeatEachRow(16),n::OriginalIdWidth::UInt64},begin->operands.spans());
    const auto query=begin->operands.declarationIdentity();
    EXPECT_EQ(query,expected.declarationIdentity());
    const auto order=graph.getExecutionOrder();
    for(int replay=0;replay<20;++replay)for(const int count:{4,1,0,3,0}){
        // Recorded API-double device memory, not a production host mirror or receipt.
        *live=count;graph.reset();EXPECT_EQ(graph.getExecutionOrder(),order);
        EXPECT_EQ(graph.getNode("L1_ple_hash")->stage->graphCaptureVariantSignature(),hash);
        EXPECT_EQ(begin->operands.declarationIdentity(),query);
        EXPECT_NO_THROW(frame->requireConsumed());
    }
}
}

namespace llaminar2::test {
/**
 * @test The actual P1 directed factory uses the complete local source graph,
 * preserving its original unallocated distributed frontier and late model joins.
 * Native SDK operations are doubled; this proves construction, not execution.
 */
TEST(PooledImmutableOwnerPinChain, SoleLocalDirectedSourceKeepsOriginalPhasesAndModelJoins)
{
    std::vector<PooledOriginalRecipe> recipes;
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const auto slots:{0u,3u})
        recipes.push_back({{slots},false,0,true,type,64});
    PooledImmutableOriginalPrograms programs(recipes);
    for(const auto& recipe:recipes){
        auto c=programs.makeWithPreparedWorkers(recipe);const auto device=c.boundary->device();
        const auto transport=q::PooledImmutableDirectedTransport::create(c.boundary);
        ASSERT_TRUE(transport->remotes().empty());ASSERT_TRUE(transport->readers().empty());
        EXPECT_THROW(q::PooledImmutableLocalOwnerBinding::bind(c.coordinator,c.owners.front(),c.memory),std::invalid_argument);

        EXPECT_EQ(c.coordinator->plan().parameters().native.ingress_frontier,0u);
        for(const bool counted:{false,true}){
            auto* live=static_cast<std::int32_t*>(c.queryLive->gpu_data_ptr());
            const auto tokens=counted?DeviceRowRange::deviceCounted(4,live):DeviceRowRange::fullyActive(4);
            q::PooledNativeOperandSpans begin{};
            begin[static_cast<unsigned>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{c.ids,BufferId::QFN_NGRAM_ROW_IDS,0,64*sizeof(std::uint64_t)};
            if(counted)begin[static_cast<unsigned>(q::PooledNativeOperand::OriginalCount)]=q::PooledNativeTensorSpan{c.queryLive,BufferId::PREFILL_CHUNK_GEOMETRY,0,sizeof(std::int32_t)};
            const auto& p=c.boundary->pooledStorage()->geometry().parameters();
            begin[static_cast<unsigned>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{
                c.boundary->pooledStorage()->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,
                p.offsets.protocol+offsetof(q::ImmutableLookupDeviceState,admissionGeneration),sizeof(std::uint64_t)};
            ComputeGraph graph;
            const auto completed=appendPooledImmutableDirectedLookupGraph(graph,c.arena,c.boundary,transport,c.coordCode,
                {tokens.repeatEachRow(16),n::OriginalIdWidth::UInt64},begin,*c.control,
                {c.readOutput,BufferId::QFN_PLE_EMBEDDING,0,64*160*sizeof(float)},"p1","p1/output");
            const auto local=std::get<std::shared_ptr<const PooledImmutableLocalRetirementCompletion>>(completed.retirement);
            ASSERT_TRUE(local);EXPECT_NO_THROW(requirePooledSourceRetirementCompletion(graph,completed.retirement));
            // Use the real model hash constructor and the same exact typed
            // demand/trunk handoff as Qwen4ExpPooledReadAheadFrame.
            q::NgramHashPolicy policy;policy.multipliers={9007199254740993ULL,9007199254740995ULL,9007199254740997ULL};
            for(unsigned head=0;head<16;++head){policy.moduli[head]=7;policy.offsets[head]=7*head;}
            Qwen4ExpStageRows rows{4};
            if(counted)rows.liveRows=Qwen4ExpNativeIntInput{device,live,1};
            graph.addNode("model/hash",std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
                .device_id=device,.rows=rows,.operation=Qwen4ExpPrimitiveStage::NgramHash{
                    Qwen4ExpNativeIntInput{device,static_cast<const std::int32_t*>(c.queryTokens->gpu_data_ptr()),4},
                    Qwen4ExpIntBinding(*c.queryPrior,BufferId::QFN_PLE_TOKEN_HISTORY),
                    Qwen4ExpIntBinding(*c.ids,BufferId::QFN_NGRAM_ROW_IDS),policy}}),device);
            graph.addDependency(completed.source.first,"model/hash");
            ASSERT_TRUE(completed.source.demand);
            declareDemandTrunk(graph,device,completed.source.demand->published);
            bindPooledDemandConsumption(graph,*completed.source.demand,{"trunk/terminal"});
            EXPECT_NO_THROW(requirePooledSourceRetirementCompletion(graph,completed.retirement));
            // Each internal phase still needs its original predecessor. A
            // missing release/read/retirement edge cannot become clean reuse.
            for(const auto& name:graph.getExecutionOrder()){
                if(name==completed.source.first || name.starts_with("model/") || name.starts_with("trunk/"))continue;
                auto* node=graph.getNode(name);if(node->dependencies.empty())continue;
                const auto saved=node->dependencies;node->dependencies.clear();
                EXPECT_THROW(requirePooledSourceRetirementCompletion(graph,completed.retirement),std::invalid_argument)<<name;
                node->dependencies=saved;
            }
            auto* first=graph.getNode(completed.source.first);first->dependencies={"trunk/terminal"};
            EXPECT_THROW(requirePooledSourceRetirementCompletion(graph,completed.retirement),std::invalid_argument);
            first->dependencies={"model/hash"};
            auto* read=graph.getNode(local->readNode());read->device=DeviceId::cpu();
            EXPECT_THROW(requirePooledSourceRetirementCompletion(graph,completed.retirement),std::invalid_argument);
            read->device=device;
            EXPECT_NO_THROW(requirePooledSourceRetirementCompletion(graph,completed.retirement));
        }
    }
}
}
