/**
 * @file Test__PooledImmutableParticipantPin.cpp
 * @brief Device-free original-cohort initial-wave geometry, authority and economical wire contracts.
 *
 * These declarations and real PMA leases do not construct prepared GPU roots,
 * load native runtimes or qualify captured peer execution. The separate native
 * phase replay consumes decoded original ingress through the production parsers.
 */
#include "execution/qwen4exp/PooledImmutableParticipantPinGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantFreeGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantVictimGraph.h"
#include <gtest/gtest.h>
#include "kernels/qwen4exp/PooledImmutableNativeAliasPolicy.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include <array>
#include <algorithm>
#include <type_traits>
#include "PooledNativeAliasGuards.inc"
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace a=q::adaptive_lookup;namespace p=a::pooled;
/** @brief Original allocator observations only; no backend discovery occurs. */
PhysicalMemoryResource observed(PhysicalMemoryAllocatorIdentity id){return {id.world_rank,id.device,1ULL<<30,1ULL<<30};}
/** @brief Project the original checked layout and caller's exact unsorted owner order. */
q::PooledImmutablePhysicalOwnerGeometry declared(std::span<const uint32_t> capacities,
    const std::vector<PhysicalMemoryAllocatorIdentity>& order,PhysicalMemoryAllocatorIdentity coordinator)
{
    std::vector<ImmutableRowTableLayoutOwner> owners;bool positive=false;
    for(size_t i=0;i<capacities.size();++i){owners.push_back({{order.at(i),ImmutableRowStorage::VRAM},0,capacities[i],{}});positive|=capacities[i]!=0;}
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,
        {{0,DeviceId::cpu()},coordinator},{},{{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},
        {},positive?std::optional(coordinator):std::nullopt,16);
    return q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
}
/** @brief Immutable common policy from exactly the declared source/owner cohort. */
q::PooledImmutableCoordinatorGeometry policy(const q::PooledImmutablePhysicalOwnerGeometry& physical)
{return q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},64);}
}
/** @brief Every original coordinate retains source bank, descriptor, exact receiver span and control/count roles. */
TEST(PooledImmutableParticipantPin, OrderedUnevenAndEmptyOwnersKeepExactOriginalWireOperands)
{
    static_assert(std::is_same_v<n::DirectedSlice,CapturedTransferSourceSlice>);
    static_assert(sizeof(n::Invocation)==48 && sizeof(n::OwnerIngress)==64);
    for(const auto count:{1u,2u,4u,8u})for(const bool reversed:{false,true}){
        const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(19)};
        std::vector<uint32_t> capacities;std::vector<PhysicalMemoryAllocatorIdentity> order;
        for(uint32_t i=0;i<count;++i){capacities.push_back(i%3?i+1:0);order.push_back({0,i%2?DeviceId::cuda(i):DeviceId::rocm(i)});}
        if(reversed){std::reverse(capacities.begin(),capacities.end());std::reverse(order.begin(),order.end());}
        const auto physical=declared(capacities,order,coordinator);
        const auto canonical=policy(physical);const auto source=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,canonical,701,17);
        for(uint32_t i=0;i<count;++i){
            auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,canonical,i,701,17);
            const auto edge=q::PooledImmutablePinEdge::fromPlans(source,owner);
            SCOPED_TRACE(::testing::Message()<<count<<"/"<<reversed<<"/"<<i);
            EXPECT_EQ(edge.sender(),coordinator);EXPECT_EQ(edge.receiver(),order[i]);EXPECT_EQ(edge.coordinate(),i);EXPECT_TRUE(edge.remote());
            EXPECT_EQ(edge.invocationSource().bytes,48u);EXPECT_EQ(edge.invocationDestination().bytes,48u);
            EXPECT_EQ(edge.invocationDestination().offset,owner.parameters().native.ingress);
            EXPECT_EQ(edge.commandSlice().offset,source.parameters().native.pin_slices+i*sizeof(n::DirectedSlice));
            EXPECT_EQ(edge.commandSlice().bytes,16u);EXPECT_EQ(edge.commandSource().offset,source.parameters().native.pin_commands);
            EXPECT_EQ(edge.maximumCommandBytes(),std::min(capacities[i],physical.maximumUniqueRows())*sizeof(p::OwnerCommand));
            EXPECT_EQ(edge.commandDestination().offset,owner.parameters().frame.commands);EXPECT_EQ(edge.commandDestination().bytes,edge.maximumCommandBytes());
            EXPECT_LE(edge.maximumCommandBytes(),edge.commandSource().bytes);EXPECT_EQ(edge.hasCommands(),capacities[i]!=0);
            EXPECT_NE(edge.messageKey(q::PooledPinWireRole::Invocation),edge.messageKey(q::PooledPinWireRole::Commands));
            if(edge.hasCommands()){
                ASSERT_TRUE(edge.incomingCount());EXPECT_EQ(edge.incomingCount()->bytes,8u);
                EXPECT_EQ(edge.incomingCount()->offset,owner.parameters().native.ingress+48);
                EXPECT_NE(edge.incomingCount()->offset,owner.parameters().native.progress+offsetof(n::OwnerProgress,reply_bytes));
            }else EXPECT_FALSE(edge.incomingCount());
            if(edge.capturedCrossVendor())EXPECT_NO_THROW(edge.channelMemory());else EXPECT_THROW(edge.channelMemory(),std::invalid_argument);
        }
    }
}
/** @brief The actual existing layout factory owns ingress/BOM once; ordinary local storage is unchanged. */
TEST(PooledImmutableParticipantPin, OriginalLayoutPricesRemoteIngressExactlyOnceAndLocalGeometryUnchanged)
{
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(7)};
    for(uint32_t slots:{0u,1u,3u,19u})for(bool remote:{false,true}){
        const std::array capacities{slots};const std::vector order{remote?PhysicalMemoryAllocatorIdentity{0,DeviceId::rocm(2)}:coordinator};
        const auto physical=declared(capacities,order,coordinator);const auto geometry=policy(physical);
        const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,17);const auto offsets=owner.parameters().native;
        const auto baseline=128u+64u*std::min(slots,physical.maximumUniqueRows());
        EXPECT_EQ(offsets.progress,0u);EXPECT_EQ(offsets.header,64u);EXPECT_EQ(offsets.replies,slots?128u:0u);
        const auto aligned=[](size_t bytes){return (bytes+63)/64*64;};const auto r=std::min(slots,physical.maximumUniqueRows());
        EXPECT_EQ(offsets.ingress,remote?baseline:0u);
        EXPECT_EQ(offsets.source_outcome,remote?baseline+64u:0u);
        EXPECT_EQ(offsets.source_fills,remote && r?baseline+128u:0u);
        EXPECT_EQ(offsets.source_payload,remote && r?aligned(baseline+128u+96u*r):0u);
        EXPECT_EQ(offsets.bytes,remote?aligned((r?offsets.source_payload:baseline+128u)+90u*r):baseline);
        EXPECT_EQ(q::PooledImmutableOwnerNativePlan::rootOffsets(physical,geometry,0),offsets);
        const auto resource=observed(owner.allocator());const auto bom=owner.mandatoryBOM(resource);
        EXPECT_EQ(bom.totalBytes(),offsets.bytes);
        EXPECT_EQ(q::PooledImmutableOwnerNativePlan::mandatoryBOM(physical,geometry,0,resource).totalBytes(),bom.totalBytes());
        auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bom);PhysicalMemoryAuthority memory(certificate,0);
        EXPECT_EQ(memory.plannedBytes(order[0].device,PhysicalMemoryOwner::ImmutableLookupStaging),offsets.bytes);
        {
            auto lease=memory.claimNewAllocation(order[0].device,PhysicalMemoryOwner::ImmutableLookupStaging,offsets.bytes);
            EXPECT_EQ(memory.claimedBytes(order[0].device,PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation),offsets.bytes);
            EXPECT_ANY_THROW(memory.claimNewAllocation(order[0].device,PhysicalMemoryOwner::ImmutableLookupStaging,1));
        }
        EXPECT_EQ(memory.claimedBytes(order[0].device,PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
/** @brief Geometry equality cannot substitute another owner, cohort, source table, storage generation or ordered coordinate. */
TEST(PooledImmutableParticipantPin, RejectsForeignOwnersBeforeAnyBindingOrAllocation)
{
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(7)};
    const std::array capacities{3u,0u,2u};const std::vector<PhysicalMemoryAllocatorIdentity> order{{0,DeviceId::rocm(4)},{0,DeviceId::cuda(2)},{0,DeviceId::rocm(1)}};
    const auto physical=declared(capacities,order,coordinator);const auto geometry=policy(physical);
    const auto source=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);
    for(uint32_t coordinate=0;coordinate<capacities.size();++coordinate){
        EXPECT_NO_THROW(q::PooledImmutablePinEdge::fromPlans(source,q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,coordinate,701,17)));
        for(const auto pair:{std::pair{702ULL,17ULL},std::pair{701ULL,18ULL}})
            EXPECT_THROW(q::PooledImmutablePinEdge::fromPlans(source,q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,coordinate,pair.first,pair.second)),std::invalid_argument);
        for(uint32_t changed=0;changed<capacities.size();++changed){
            auto foreignOrder=order;foreignOrder[changed].device.ordinal+=32;
            const auto foreign=declared(capacities,foreignOrder,coordinator);const auto foreignGeometry=policy(foreign);
            EXPECT_THROW(q::PooledImmutablePinEdge::fromPlans(source,q::PooledImmutableOwnerNativePlan::fromGeometry(foreign,foreignGeometry,coordinate,701,17)),std::invalid_argument);
        }
    }
    EXPECT_THROW(q::PooledImmutableCapturedPinEdge::bind({},{},{},nullptr,nullptr),std::invalid_argument);
}
/** @brief Metadata supports native/rank edges without silently materializing a mapped substitute. */
TEST(PooledImmutableParticipantPin, NativeAndCrossRankTransportPrerequisitesStayHardErrors)
{
    const PhysicalMemoryAllocatorIdentity source{0,DeviceId::cuda(7)};const std::array capacities{3u};
    for(const auto target:{PhysicalMemoryAllocatorIdentity{0,DeviceId::cuda(2)},PhysicalMemoryAllocatorIdentity{1,DeviceId::rocm(1)},source}){
        const auto physical=declared(capacities,std::vector{target},source);const auto geometry=policy(physical);
        auto edge=q::PooledImmutablePinEdge::fromPlans(q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17),
            q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,17));
        EXPECT_FALSE(edge.capturedCrossVendor());EXPECT_THROW(edge.channelMemory(),std::invalid_argument);
        if(!edge.remote()){
            ASSERT_TRUE(edge.incomingCount());EXPECT_EQ(edge.incomingCount()->buffer,BufferId::QFN_POOLED_COORDINATOR_NATIVE);
            EXPECT_EQ(edge.incomingCount()->offset,edge.commandSlice().offset+8);EXPECT_EQ(edge.invocationSource(),edge.invocationDestination());
        }
    }
}
/** @brief Existing physical/native BOM producers merge once into the single original authority across all owners. */
TEST(PooledImmutableParticipantPin, OneOriginalPMAOwnsEveryDistinctOwnerRootAndRejectsDuplicateClaims)
{
    for(uint32_t participants:{1u,2u,4u,8u}){
        const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(19)};
        std::vector<uint32_t> capacities;std::vector<PhysicalMemoryAllocatorIdentity> order;
        for(uint32_t i=0;i<participants;++i){capacities.push_back(i%3?i+1:0);order.push_back({0,i%2?DeviceId::cuda(i):DeviceId::rocm(i)});}
        std::reverse(order.begin(),order.end());
        const auto physical=declared(capacities,order,coordinator);const auto geometry=policy(physical);
        PhysicalMemoryPlanBuilder builder;
        const auto add=[&](const PhysicalMemoryPlan& plan){for(const auto& resource:plan.resources())builder.add(resource);};
        add(geometry.policyBOM(observed(coordinator)));add(geometry.mandatoryBOM(observed(coordinator)));
        const auto source=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);
        add(source.mandatoryBOM(observed(coordinator)));
        for(uint32_t i=0;i<participants;++i){
            const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,i,701,17);
            add(physical.cacheBOM(i,observed(order[i])));add(physical.mandatoryBOM(i,observed(order[i])));add(owner.mandatoryBOM(observed(order[i])));
        }
        auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build());PhysicalMemoryAuthority memory(certificate,0);
        EXPECT_EQ(certificate->plan().resources().size(),participants+1);
        std::vector<PhysicalMemoryAllocationLease> leases;
        for(uint32_t i=0;i<participants;++i){
            const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,i,701,17);
            const auto& native=owner.parameters().native;const auto& original=physical.owners()[i];
            EXPECT_EQ(memory.plannedBytes(order[i].device,PhysicalMemoryOwner::ImmutableLookupStaging),original.frame.bytes+native.bytes);
            leases.push_back(memory.claimNewAllocation(order[i].device,PhysicalMemoryOwner::ImmutableLookupStaging,original.frame.bytes));
            leases.push_back(memory.claimNewAllocation(order[i].device,PhysicalMemoryOwner::ImmutableLookupStaging,native.bytes));
            EXPECT_ANY_THROW(memory.claimNewAllocation(order[i].device,PhysicalMemoryOwner::ImmutableLookupStaging,native.bytes));
        }
        leases.clear();
        for(const auto& id:order)EXPECT_EQ(memory.claimedBytes(id.device,PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}

namespace {
/** @brief Actual CPU backing/leases exercise production alias declarations without a fake GPU storage owner or seal. */
struct AliasRoots
{
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    AliasRoots(){PhysicalMemoryPlanBuilder b;b.add(observed({0,DeviceId::cpu()}),PhysicalMemoryOwner::ImmutableLookupStaging,1u<<20);
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(b.build()),0);}
    /** @return One actually allocated original canonical CPU Tensor; no runtime/backend state is forged. */
    std::shared_ptr<TensorBase> root(size_t bytes){
        if(!bytes)return {};if(bytes%4)throw std::logic_error("fixture root requires actual INT32 packing");
        return CPUTensorStorageDeclaration(1,bytes/4,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu()).materialize(memory);
    }
    /** @return Original source-owned root shapes, retaining real CPU PMA leases for metadata-only checks. */
    template<class Plan> q::PooledImmutableNativeRoots roots(const Plan& plan){const auto& p=plan.parameters();return {root(p.cache.bytes),root(p.frame.bytes),root(p.native.bytes)};}
};
/** @brief Set one exact schema role using its original canonical owner. */
void span(q::PooledNativeOperandSpans& inputs,q::PooledNativeOperand role,std::shared_ptr<TensorBase> root,BufferId id,size_t offset,size_t bytes)
{if(bytes)inputs[static_cast<size_t>(role)]=q::PooledNativeTensorSpan{std::move(root),id,offset,bytes};}
}
/** @brief The exact guard called by native enqueue permits only original ingress roles/actions and rejects physical authority aliases. */
TEST(PooledImmutableParticipantPin, ProductionOwnerAliasGuardChecksIngressActionRootGenerationAndPartialOverlap)
{
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(7)};const std::array capacities{3u};
    const auto physical=declared(capacities,std::vector<PhysicalMemoryAllocatorIdentity>{{0,DeviceId::rocm(2)}},coordinator);const auto geometry=policy(physical);
    const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,17);const auto& p=owner.parameters();AliasRoots storage;const auto roots=storage.roots(owner);
    q::PooledNativeOperandSpans original{};
    span(original,q::PooledNativeOperand::Invocation,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress,48);
    span(original,q::PooledNativeOperand::Records,roots.frame,BufferId::QFN_POOLED_OWNER_FRAME,p.frame.commands,p.native.reply_capacity*sizeof(p::OwnerCommand));
    span(original,q::PooledNativeOperand::RecordBytes,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress+48,8);
    q::PooledImmutableOwnerOperands inputs(n::OwnerAction::Pin,owner,original);
    EXPECT_NO_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::Pin,inputs,roots,owner));
    EXPECT_THROW(q::alias_original::ownerAliases(inputs,roots,owner.parameters()),std::invalid_argument);
    const auto* real=static_cast<const std::byte*>(roots.native->raw_data());
    EXPECT_TRUE(q::alias_candidate::overlaps(real,64,real+8,16));
    EXPECT_FALSE(q::alias_candidate::overlaps(real,64,real+64,16));
    EXPECT_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::Begin,inputs,roots,owner),std::invalid_argument);
    const auto stale=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,18);
    EXPECT_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::Pin,inputs,roots,stale),std::invalid_argument);
    for(unsigned defect=0;defect<5;++defect){auto changed=original;
        if(defect==0)changed[static_cast<size_t>(q::PooledNativeOperand::Invocation)]->offset-=8;
        if(defect==1)changed[static_cast<size_t>(q::PooledNativeOperand::RecordBytes)]->offset=p.native.progress+offsetof(n::OwnerProgress,reply_bytes);
        if(defect==2){const auto foreign=storage.root(p.native.bytes);changed[static_cast<size_t>(q::PooledNativeOperand::Invocation)]->root=foreign;changed[static_cast<size_t>(q::PooledNativeOperand::RecordBytes)]->root=foreign;}
        if(defect==3)changed[static_cast<size_t>(q::PooledNativeOperand::Records)]->offset=p.frame.accepted_plan;
        if(defect==4)changed[static_cast<size_t>(q::PooledNativeOperand::Records)]->offset+=8;
        q::PooledImmutableOwnerOperands forged(n::OwnerAction::Pin,owner,changed);
        EXPECT_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::Pin,forged,roots,owner),std::invalid_argument)<<defect;
    }
    q::PooledNativeOperandSpans frontier{};span(frontier,q::PooledNativeOperand::Invocation,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress,48);
    span(frontier,q::PooledNativeOperand::Frontier,roots.frame,BufferId::QFN_POOLED_OWNER_FRAME,p.frame.pin_headers,sizeof(p::PinHeader));
    q::PooledImmutableOwnerOperands acquire(n::OwnerAction::AcquirePins,owner,frontier);
    EXPECT_NO_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::AcquirePins,acquire,roots,owner));
}
/** @brief Real coordinator guard admits only the serial receive union and its exact phase frontier, never outgoing metadata or a foreign root. */
TEST(PooledImmutableParticipantPin, ProductionCoordinatorAliasGuardChecksSharedBankRolesAndSourceGeneration)
{
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(7)};const std::array capacities{3u,0u,2u,1u};
    const auto physical=declared(capacities,std::vector<PhysicalMemoryAllocatorIdentity>{{0,DeviceId::rocm(2)},{0,DeviceId::rocm(3)},{0,DeviceId::rocm(4)},{0,DeviceId::rocm(5)}},coordinator);const auto geometry=policy(physical);
    const auto plan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);const auto& p=plan.parameters();AliasRoots storage;const auto roots=storage.roots(plan);
    q::PooledNativeOperandSpans original{};
    span(original,q::PooledNativeOperand::Records,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,p.native.pin_commands,p.native.reply_capacity*sizeof(p::Grant));
    span(original,q::PooledNativeOperand::RecordBytes,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,p.native.ingress_count,8);
    q::PooledImmutableCoordinatorOperands inputs(n::CoordinatorAction::AcquirePins,plan,{},original);
    EXPECT_NO_THROW(q::alias_candidate::coordinatorAliases(n::CoordinatorAction::AcquirePins,inputs,roots,plan));
    EXPECT_THROW(q::alias_original::coordinatorAliases(inputs,roots),std::invalid_argument);
    EXPECT_THROW(q::alias_candidate::coordinatorAliases(n::CoordinatorAction::PinCommands,inputs,roots,plan),std::invalid_argument);
    const auto stale=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,18);
    EXPECT_THROW(q::alias_candidate::coordinatorAliases(n::CoordinatorAction::AcquirePins,inputs,roots,stale),std::invalid_argument);
    for(unsigned defect=0;defect<4;++defect){auto changed=original;
        if(defect==0)changed[static_cast<size_t>(q::PooledNativeOperand::Records)]->offset=p.native.sorted_releases;
        if(defect==1)changed[static_cast<size_t>(q::PooledNativeOperand::RecordBytes)]->offset=p.native.pin_cursors;
        if(defect==2){const auto foreign=storage.root(p.native.bytes);for(auto& input:changed)if(input)input->root=foreign;}
        if(defect==3)changed[static_cast<size_t>(q::PooledNativeOperand::Records)]->offset+=8;
        q::PooledImmutableCoordinatorOperands forged(n::CoordinatorAction::AcquirePins,plan,{},changed);
        EXPECT_THROW(q::alias_candidate::coordinatorAliases(n::CoordinatorAction::AcquirePins,forged,roots,plan),std::invalid_argument)<<defect;
    }
    q::PooledNativeOperandSpans frontier{};span(frontier,q::PooledNativeOperand::Frontier,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,p.native.ingress_frontier,capacities.size()*sizeof(p::PinHeader));
    q::PooledImmutableCoordinatorOperands seal(n::CoordinatorAction::SealPins,plan,{},frontier);
    EXPECT_NO_THROW(q::alias_candidate::coordinatorAliases(n::CoordinatorAction::SealPins,seal,roots,plan));
    frontier[static_cast<size_t>(q::PooledNativeOperand::Frontier)]->offset+=8;
    q::PooledImmutableCoordinatorOperands partial(n::CoordinatorAction::SealPins,plan,{},frontier);
    EXPECT_THROW(q::alias_candidate::coordinatorAliases(n::CoordinatorAction::SealPins,partial,roots,plan),std::invalid_argument);
}

namespace {
/** @brief CPU-only graph node for observing the actual production dependency helper, never a native compute substitute. */
class PinTopologyNode final:public IComputeStage
{
public:
    /** @brief Ordinary CPU topology fixture, with no GPU worker/context. */
    PinTopologyNode():IComputeStage(DeviceId::cpu()){}
    /** @return Topology-only fixture success; native parser execution has its own actual-source replay. */
    bool execute(IDeviceContext*)override{return true;}
    /** @return Existing nominal CPU stage type, unrelated to native capability. */
    ComputeStageType type()const override{return ComputeStageType::GEMM;}
    /** @return Immutable fixture name. */
    std::string name()const override{return "PinTopologyFixture";}
    /** @return CPU-only test scheduling; no native execution capability. */
    bool supportsBackend(ComputeBackendType backend)const override{return backend==ComputeBackendType::CPU;}
    /** @return No live model/device observation. */
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
};
/** @return Whether the real graph's dependency state currently permits this exact node. */
bool ready(const ComputeGraph& graph,const std::string& name){const auto nodes=graph.getReadyNodes();return std::find(nodes.begin(),nodes.end(),name)!=nodes.end();}
}
/** @brief An early owner receive cannot overwrite a late owner's command, and each prefix is consumed before reuse/reset. */
TEST(PooledImmutableParticipantPin, ActualGraphLifetimeBlocksAsymmetricSendProgressAndSerializesEachReceive)
{
    const PhysicalMemoryAllocatorIdentity source{0,DeviceId::cuda(7)};
    for(unsigned participants:{1u,2u,4u,8u})for(bool allEmpty:{false,true}){
        std::vector<uint32_t> capacities(participants,allEmpty?0u:2u);if(participants>2)capacities[1]=0;
        std::vector<PhysicalMemoryAllocatorIdentity> order;for(unsigned i=0;i<participants;++i)order.push_back({0,DeviceId::rocm(participants-i)});
        const auto physical=declared(capacities,order,source);const auto geometry=policy(physical);const auto original=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);
        ComputeGraph graph;const auto add=[&](const std::string& name){graph.addNode(name,std::make_unique<PinTopologyNode>(),DeviceId::cpu());};add("producer");
        std::vector<PooledPinReplyOwnerNodes> owners;
        for(unsigned i=0;i<participants;++i){const auto prefix=std::to_string(i);PooledPinReplyOwnerNodes n{i,"sent"+prefix,capacities[i]?"received"+prefix:"",capacities[i]?"acquired"+prefix:"","header"+prefix};
            add(n.commandSent);if(capacities[i]){add(n.replyReceived);add(n.replyAcquired);graph.addDependency(n.commandSent,"producer");}else graph.addDependency("producer",n.commandSent);add(n.headerReceived);owners.push_back(n);}
        ASSERT_NO_THROW(wirePooledPinReplyBankLifetime(graph,original,owners,"producer"));const auto topology=graph.getExecutionOrder();
        for(unsigned replay=0;replay<3;++replay){graph.reset();
            for(const auto& o:owners)if(o.replyReceived.empty()){ASSERT_TRUE(ready(graph,o.commandSent));graph.markCompleted(o.commandSent);}
            const auto first=owners[0].replyReceived.empty()?owners[0].headerReceived:owners[0].replyReceived;
            EXPECT_FALSE(ready(graph,first));ASSERT_TRUE(ready(graph,"producer"));graph.markCompleted("producer");
            size_t remaining=std::count_if(owners.begin(),owners.end(),[](const auto& o){return !o.replyReceived.empty();});
            for(const auto& o:owners)if(!o.replyReceived.empty()){
                EXPECT_FALSE(ready(graph,first));ASSERT_TRUE(ready(graph,o.commandSent));graph.markCompleted(o.commandSent);--remaining;
                if(remaining)EXPECT_FALSE(ready(graph,first));
            }
            for(size_t i=0;i<owners.size();++i){const auto& o=owners[i];
                if(!o.replyReceived.empty()){
                    ASSERT_TRUE(ready(graph,o.replyReceived));graph.markCompleted(o.replyReceived);
                    if(i+1<owners.size())EXPECT_FALSE(ready(graph,owners[i+1].replyReceived.empty()?owners[i+1].headerReceived:owners[i+1].replyReceived));
                    ASSERT_TRUE(ready(graph,o.replyAcquired));graph.markCompleted(o.replyAcquired);
                }
                ASSERT_TRUE(ready(graph,o.headerReceived));graph.markCompleted(o.headerReceived);
            }
            EXPECT_EQ(graph.getExecutionOrder(),topology);
        }
        auto partial=owners;partial.back().coordinate=participants+1;
        EXPECT_THROW(wirePooledPinReplyBankLifetime(graph,original,partial,"producer"),std::invalid_argument);
    }
}
/** @brief Record/control wire extents are independent of original reusable source capacity and retained frontier alignment. */
TEST(PooledImmutableParticipantPin, PinReplyExtentsUseOriginalGrantCountAndExactFullCohortHeaderArray)
{
    const PhysicalMemoryAllocatorIdentity source{0,DeviceId::cuda(7)};
    for(unsigned participants:{1u,2u,4u,8u}){
        std::vector<uint32_t> capacities;std::vector<PhysicalMemoryAllocatorIdentity> order;
        for(unsigned i=0;i<participants;++i){capacities.push_back(i%3?i:0);order.push_back({0,DeviceId::rocm(participants-i)});}
        const auto physical=declared(capacities,order,source);const auto geometry=policy(physical);const auto original=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);
        for(unsigned i=0;i<participants;++i){const auto edge=q::PooledImmutablePinEdge::fromPlans(original,q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,i,701,17));
            EXPECT_EQ(edge.replySource().bytes,std::min(capacities[i],physical.maximumUniqueRows())*sizeof(p::Grant));
            EXPECT_EQ(edge.replyCountSource().bytes,8u);EXPECT_EQ(edge.replyCountSource().offset,edge.owner().parameters().native.progress+offsetof(n::OwnerProgress,reply_bytes));
            EXPECT_EQ(edge.replyDestination().offset,edge.commandSource().offset);EXPECT_LE(edge.replyDestination().bytes,edge.commandSource().bytes);
            EXPECT_EQ(edge.pinHeaderSource().bytes,32u);EXPECT_EQ(edge.pinHeaderDestination().offset,original.parameters().native.ingress_frontier+i*32);
            EXPECT_EQ(edge.frontierSource().bytes,participants*32u);EXPECT_EQ(edge.frontierDestination().bytes,participants*32u);
            EXPECT_EQ(edge.frontierDestination().offset,edge.owner().parameters().frame.pin_headers);
            EXPECT_EQ(bool(edge.replyCountDestination()),capacities[i]!=0);
            EXPECT_NE(edge.messageKey(q::PooledPinWireRole::Replies),edge.messageKey(q::PooledPinWireRole::Header));
            EXPECT_EQ(original.parameters().native.ingress_count,original.parameters().native.reply_capacity?original.parameters().native.ingress_frontier+participants*sizeof(p::PublishHeader):0u);
            EXPECT_NO_THROW(edge.returnChannelMemory());
        }
    }
}

}

namespace llaminar2::test {
/**
 * @brief Every real wire action and endpoint retains a distinct immutable stage variant.
 *
 * Boundary and deterministic replayed binding words exercise the production
 * helper used by both actual directed stage classes. Mutable payload/count
 * values are intentionally absent. This is a CPU capture-key regression, not
 * native storage, endpoint capture, communication or factory qualification.
 */
TEST(PooledImmutableParticipantPin, EndpointActionCaptureIdentityRetainsEveryWireRole)
{
    namespace q=qwen4exp;
    constexpr std::array roles{q::PooledPinWireRole::Invocation,q::PooledPinWireRole::Commands,
        q::PooledPinWireRole::Frontier,q::PooledPinWireRole::Replies,q::PooledPinWireRole::Header};
    constexpr std::array endpoints{CapturedTransferEndpoint::Producer,CapturedTransferEndpoint::Consumer};
    static_assert(static_cast<unsigned>(CapturedTransferEndpoint::Producer)==0);
    static_assert(static_cast<unsigned>(CapturedTransferEndpoint::Consumer)==1);
    std::array<uint64_t,264> identities{0,1,2,3,UINT64_MAX,UINT64_MAX-1,1ULL<<63,(1ULL<<63)-1};
    uint64_t state=0xc17a24b69fd308e5ULL;
    for(size_t i=8;i<identities.size();++i){state^=state<<13;state^=state>>7;state^=state<<17;identities[i]=state;}
    for(const auto identity:identities){
        SCOPED_TRACE(identity);std::array<uint64_t,roles.size()*endpoints.size()> signatures{};size_t index=0;
        for(const auto role:roles)for(const auto endpoint:endpoints){
            const auto signature=q::pooledPinEndpointCaptureIdentity(identity,role,endpoint);
            EXPECT_NE(signature,0u);EXPECT_EQ(signature,q::pooledPinEndpointCaptureIdentity(identity,role,endpoint));
            for(size_t prior=0;prior<index;++prior)EXPECT_NE(signature,signatures[prior]);
            signatures[index++]=signature;
        }
    }
}
}

namespace llaminar2::test {
/** @brief Original one/two/four/eight owner wire spans keep source origins, phase union roles and exact metadata extents. */
TEST(PooledImmutableParticipantPin, FreeWaveGeometryRetainsOriginalBanksAndExactMessageRoles)
{
    const PhysicalMemoryAllocatorIdentity source{0,DeviceId::cuda(7)};
    constexpr std::array roles{q::PooledFreeWireRole::Availability,q::PooledFreeWireRole::Plan,q::PooledFreeWireRole::Decisions,q::PooledFreeWireRole::Grants,q::PooledFreeWireRole::Receipt,q::PooledFreeWireRole::Frontier};
    for(unsigned participants:{1u,2u,4u,8u})for(bool reversed:{false,true}){
        std::vector<uint32_t> capacities;std::vector<PhysicalMemoryAllocatorIdentity> order;for(unsigned i=0;i<participants;++i){capacities.push_back(i%3?i+1:0);order.push_back({0,DeviceId::rocm(i)});}if(reversed){std::reverse(capacities.begin(),capacities.end());std::reverse(order.begin(),order.end());}
        const auto physical=declared(capacities,order,source);const auto geometry=policy(physical);const auto original=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);
        for(unsigned i=0;i<participants;++i){const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,i,701,17);const auto pin=q::PooledImmutablePinEdge::fromPlans(original,owner);const auto edge=q::PooledImmutableFreeEdge::fromOriginal(pin);const auto& c=original.parameters();const auto& o=owner.parameters();
        EXPECT_EQ(edge.source(q::PooledFreeWireRole::Availability).bytes,40u);
        EXPECT_EQ(edge.destination(q::PooledFreeWireRole::Availability).offset,c.native.ingress_frontier+i*40);
        EXPECT_EQ(edge.source(q::PooledFreeWireRole::Plan).bytes,48u);
        EXPECT_EQ(edge.destination(q::PooledFreeWireRole::Plan).offset,o.native.header);
        EXPECT_EQ(edge.source(q::PooledFreeWireRole::Decisions).offset,c.frame.decisions);
        EXPECT_EQ(edge.source(q::PooledFreeWireRole::Decisions).buffer,BufferId::QFN_POOLED_COORDINATOR_FRAME);
        EXPECT_EQ(edge.decisionSlice().offset,c.native.free_slices+i*sizeof(n::DirectedSlice));
        EXPECT_EQ(edge.decisionSlice().bytes,16u);
        EXPECT_EQ(edge.destination(q::PooledFreeWireRole::Decisions).offset,o.frame.commands);
        EXPECT_EQ(edge.maximumDecisionBytes(),std::min(capacities[i],physical.maximumUniqueRows())*sizeof(p::AllocationDecision));
        EXPECT_EQ(edge.source(q::PooledFreeWireRole::Grants),pin.replySource());
        EXPECT_EQ(edge.destination(q::PooledFreeWireRole::Grants),pin.replyDestination());
        EXPECT_EQ(edge.source(q::PooledFreeWireRole::Receipt).bytes,40u);
        EXPECT_EQ(edge.source(q::PooledFreeWireRole::Frontier).bytes,participants*40u);
        EXPECT_EQ(edge.destination(q::PooledFreeWireRole::Frontier).offset,o.frame.free_headers);
        EXPECT_EQ(edge.destination(q::PooledFreeWireRole::Frontier).bytes,participants*40u);
        EXPECT_EQ(bool(edge.decisionCount()),capacities[i]!=0);
        EXPECT_EQ(bool(edge.incomingGrantCount()),capacities[i]!=0);if(edge.hasPayload())EXPECT_EQ(edge.decisionCount()->offset,o.native.ingress+48);
            std::vector<uint64_t> keys,signatures;for(const auto role:roles){const auto key=edge.messageKey(role);for(auto prior:keys)EXPECT_NE(key,prior);keys.push_back(key);
        EXPECT_NE(key,pin.messageKey(q::PooledPinWireRole::Header));
                for(auto endpoint:{CapturedTransferEndpoint::Producer,CapturedTransferEndpoint::Consumer}){const auto variant=q::pooledFreeEndpointCaptureIdentity(pin.identity(),role,endpoint);for(auto prior:signatures)EXPECT_NE(variant,prior);signatures.push_back(variant);}
                const auto extent=role==q::PooledFreeWireRole::Decisions?edge.maximumDecisionBytes():edge.source(role).bytes;const bool reverse=role==q::PooledFreeWireRole::Availability || role==q::PooledFreeWireRole::Grants || role==q::PooledFreeWireRole::Receipt;
        EXPECT_LE(extent,reverse?edge.reverseChannelMaximum():edge.forwardChannelMaximum());}
        }
    }
    EXPECT_THROW(q::PooledImmutableCapturedFreeWave::bind({},{}),std::invalid_argument);
    const auto localPhysical=declared(std::array{3u},std::vector{source},source);const auto localPolicy=policy(localPhysical);
    const auto localCoordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(localPhysical,localPolicy,701,17);
    const auto localOwner=q::PooledImmutableOwnerNativePlan::fromGeometry(localPhysical,localPolicy,0,701,17);
    EXPECT_THROW(q::PooledImmutableFreeEdge::fromOriginal(q::PooledImmutablePinEdge::fromPlans(localCoordinator,localOwner)),std::invalid_argument);
}
/** @brief Actual extracted enqueue guards admit only the new exact nominal header/count roles; old code causally rejects them. */
TEST(PooledImmutableParticipantPin, FreeWaveProductionGuardChecksTransportedHeaderCountRootAndGeneration)
{
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(7)};const auto physical=declared(std::array{3u,0u},std::vector<PhysicalMemoryAllocatorIdentity>{{0,DeviceId::rocm(2)},{0,DeviceId::rocm(3)}},coordinator);const auto geometry=policy(physical);const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,17);const auto& p=owner.parameters();AliasRoots storage;const auto roots=storage.roots(owner);
    q::PooledNativeOperandSpans spans{};span(spans,q::PooledNativeOperand::Invocation,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress,48);span(spans,q::PooledNativeOperand::Frontier,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.header,48);span(spans,q::PooledNativeOperand::Records,roots.frame,BufferId::QFN_POOLED_OWNER_FRAME,p.frame.commands,p.native.reply_capacity*sizeof(p::AllocationDecision));span(spans,q::PooledNativeOperand::RecordBytes,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress+48,8);
    const auto operands=[&](auto inputs){return q::PooledImmutableOwnerOperands(n::OwnerAction::AcceptFree,owner,std::move(inputs),n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);};
    const auto original=operands(spans);
        EXPECT_NO_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::AcceptFree,original,roots,owner));
        EXPECT_THROW(q::alias_original::ownerAliases(original,roots,owner.parameters()),std::invalid_argument);
        EXPECT_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::AcceptVictim,original,roots,owner),std::invalid_argument);
    const auto stale=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,18);
        EXPECT_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::AcceptFree,original,roots,stale),std::invalid_argument);
    for(unsigned defect=0;defect<5;++defect){auto changed=spans;
        if(defect==0)changed[static_cast<size_t>(q::PooledNativeOperand::Frontier)]->offset+=8;
        if(defect==1)changed[static_cast<size_t>(q::PooledNativeOperand::RecordBytes)]->offset=p.native.progress+offsetof(n::OwnerProgress,reply_bytes);
        if(defect==2)changed[static_cast<size_t>(q::PooledNativeOperand::Records)]->offset+=8;
        if(defect==3)changed[static_cast<size_t>(q::PooledNativeOperand::Records)]->offset=p.frame.accepted_plan;
        if(defect==4){const auto foreign=storage.root(p.native.bytes);for(auto& input:changed)if(input && input->buffer==BufferId::QFN_POOLED_OWNER_NATIVE)input->root=foreign;}
        const auto forged=operands(changed);
        EXPECT_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::AcceptFree,forged,roots,owner),std::invalid_argument)<<defect;
    }
    const auto schema=q::pooledNativeSchema(n::OwnerAction::AcceptFree,p,n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);
        EXPECT_EQ(schema[static_cast<size_t>(q::PooledNativeOperand::RecordBytes)].bytes,8u);
        EXPECT_THROW(q::PooledImmutableOwnerOperands(n::OwnerAction::AcceptFree,owner,spans),std::invalid_argument); // Local compact mode cannot silently acquire a transport count.
}
/** @brief Actual Availability source reuse and shared plan read frontier block asymmetric peer progress across reset. */
TEST(PooledImmutableParticipantPin, FreeWaveActualGraphLifetimeRetiresPinAndEveryPlanBeforeSharedIngressReuse)
{
    const PhysicalMemoryAllocatorIdentity source{0,DeviceId::cuda(7)};
    for(unsigned count:{1u,2u,4u,8u})for(bool allEmpty:{false,true}){
        std::vector<uint32_t> capacities(count,allEmpty?0u:2u);if(count>2)capacities[1]=0;std::vector<PhysicalMemoryAllocatorIdentity> order;for(unsigned i=0;i<count;++i)order.push_back({0,DeviceId::rocm(i)});
        const auto physical=declared(capacities,order,source);const auto geometry=policy(physical);const auto original=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);
        ComputeGraph graph;const auto add=[&](std::string name){graph.addNode(name,std::make_unique<PinTopologyNode>(),DeviceId::cpu());};add("pin_frontier");add("plan_free");std::vector<std::string> available;std::vector<PooledPinReplyOwnerNodes> replies;
        for(unsigned i=0;i<count;++i){auto suffix=std::to_string(i);available.push_back("available"+suffix);add(available.back());PooledPinReplyOwnerNodes nodes{i,"plan_sent"+suffix,capacities[i]?"grant"+suffix:"",capacities[i]?"acquire"+suffix:"","receipt"+suffix};add(nodes.commandSent);graph.addDependency(nodes.commandSent,"plan_free");if(capacities[i]){add(nodes.replyReceived);add(nodes.replyAcquired);}add(nodes.headerReceived);replies.push_back(nodes);}
        ASSERT_NO_THROW(wirePooledFreeAvailabilityLifetime(graph,original,available,"pin_frontier","plan_free"));
        ASSERT_NO_THROW(wirePooledPinReplyBankLifetime(graph,original,replies,"plan_free"));const auto topology=graph.getExecutionOrder();
        for(unsigned replay=0;replay<3;++replay){graph.reset();for(const auto& name:available)EXPECT_FALSE(ready(graph,name));
        ASSERT_TRUE(ready(graph,"pin_frontier"));graph.markCompleted("pin_frontier");for(const auto& name:available){ASSERT_TRUE(ready(graph,name));
        EXPECT_FALSE(ready(graph,"plan_free"));graph.markCompleted(name);}ASSERT_TRUE(ready(graph,"plan_free"));graph.markCompleted("plan_free");
            const auto first=replies[0].replyReceived.empty()?replies[0].headerReceived:replies[0].replyReceived;for(size_t i=0;i<replies.size();++i){EXPECT_FALSE(ready(graph,first));
        ASSERT_TRUE(ready(graph,replies[i].commandSent));graph.markCompleted(replies[i].commandSent);if(i+1<replies.size())EXPECT_FALSE(ready(graph,first));}
            for(const auto& nodes:replies){if(!nodes.replyReceived.empty()){ASSERT_TRUE(ready(graph,nodes.replyReceived));graph.markCompleted(nodes.replyReceived);
        ASSERT_TRUE(ready(graph,nodes.replyAcquired));graph.markCompleted(nodes.replyAcquired);}ASSERT_TRUE(ready(graph,nodes.headerReceived));graph.markCompleted(nodes.headerReceived);}EXPECT_EQ(graph.getExecutionOrder(),topology);
        }
        auto partial=available;partial.pop_back();
        EXPECT_THROW(wirePooledFreeAvailabilityLifetime(graph,original,partial,"pin_frontier","plan_free"),std::invalid_argument);
    }
}
}

namespace llaminar2::test {
/** @brief Actual complete coordinator guard authenticates acquired free Availability/Grant/receipt roles before submission. */
TEST(PooledImmutableParticipantPin, FreeWaveProductionCoordinatorGuardChecksEveryAcquiredOriginalFrontier)
{
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(7)};
    const auto physical=declared(std::array{3u,0u},std::vector<PhysicalMemoryAllocatorIdentity>{{0,DeviceId::rocm(2)},{0,DeviceId::rocm(3)}},coordinator);const auto geometry=policy(physical);const auto plan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);const auto& p=plan.parameters();AliasRoots storage;const auto roots=storage.roots(plan);
    for(const auto action:{n::CoordinatorAction::PlanFree,n::CoordinatorAction::CompleteFree,n::CoordinatorAction::AcquireFree}){
        q::PooledNativeOperandSpans spans{};
        if(action==n::CoordinatorAction::AcquireFree){span(spans,q::PooledNativeOperand::Records,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,p.native.pin_commands,p.native.reply_capacity*sizeof(p::Grant));span(spans,q::PooledNativeOperand::RecordBytes,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,p.native.ingress_count,8);}
        else span(spans,q::PooledNativeOperand::Frontier,roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,p.native.ingress_frontier,p.policy.participants()*40);
        const auto make=[&](auto inputs){return q::PooledImmutableCoordinatorOperands(action,plan,{},std::move(inputs));};const auto inputs=make(spans);
        EXPECT_NO_THROW(q::alias_candidate::coordinatorAliases(action,inputs,roots,plan));
        EXPECT_THROW(q::alias_original::coordinatorAliases(inputs,roots),std::invalid_argument);
        auto changed=spans;const auto role=action==n::CoordinatorAction::AcquireFree?q::PooledNativeOperand::Records:q::PooledNativeOperand::Frontier;changed[static_cast<size_t>(role)]->offset+=8;const auto partial=make(changed);
        EXPECT_THROW(q::alias_candidate::coordinatorAliases(action,partial,roots,plan),std::invalid_argument);
        changed=spans;const auto foreign=storage.root(p.native.bytes);for(auto& span:changed)if(span)span->root=foreign;const auto replaced=make(changed);
        EXPECT_THROW(q::alias_candidate::coordinatorAliases(action,replaced,roots,plan),std::invalid_argument);
    }
}
}

namespace llaminar2::test {
/** @brief Exact actual argument assembly preserves transported zero coordinates, C0 absence and original Directed origins. */
TEST(PooledImmutableParticipantPin, FreeWaveActualArgumentAssemblerKeepsTransportedCoordinatesAndC0Absent)
{
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(7)};
    for(uint32_t capacity:{0u,3u}){
        const auto physical=declared(std::array{capacity},std::vector<PhysicalMemoryAllocatorIdentity>{{0,DeviceId::rocm(2)}},coordinator);const auto geometry=policy(physical);const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,17);const auto& p=owner.parameters();AliasRoots storage;const auto roots=storage.roots(owner);q::PooledNativeOperandSpans spans{};
        span(spans,q::PooledNativeOperand::Invocation,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress,48);span(spans,q::PooledNativeOperand::Frontier,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.header,48);
        if(capacity){span(spans,q::PooledNativeOperand::Records,roots.frame,BufferId::QFN_POOLED_OWNER_FRAME,p.frame.commands,p.native.reply_capacity*sizeof(p::AllocationDecision));span(spans,q::PooledNativeOperand::RecordBytes,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress+48,8);}
        const auto actual=q::arguments_candidate::ownerNativeInputs(spans,n::OwnerAction::AcceptFree,n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);
        EXPECT_EQ(actual.record_layout,n::RecordLayout::TransportedAllocationPrefix);EXPECT_EQ(actual.record_origin,0u);EXPECT_EQ(actual.record_capacity,0u);EXPECT_EQ(actual.allocation_slice,nullptr);
        EXPECT_EQ(actual.records,capacity?static_cast<const std::byte*>(roots.frame->raw_data())+p.frame.commands:nullptr);EXPECT_EQ(actual.record_bytes,capacity?reinterpret_cast<const uint64_t*>(static_cast<const std::byte*>(roots.native->raw_data())+p.native.ingress+48):nullptr);
        const auto compact=q::arguments_candidate::ownerNativeInputs(spans,n::OwnerAction::AcceptFree,n::SourceCondition::Always,n::RecordLayout::Compact);EXPECT_EQ(compact.record_origin,0u);EXPECT_EQ(compact.record_capacity,0u);
        if(capacity){
            // The exact original non-Compact branch incorrectly selects
            // directed coordinates for this new counted transport mode.
            const auto original=q::arguments_original::ownerNativeInputs(spans,n::OwnerAction::AcceptFree,n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);EXPECT_NE(original.record_capacity,0u);
            const auto coordinatorPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,geometry,701,17);const auto coordinatorRoots=storage.roots(coordinatorPlan);const auto& c=coordinatorPlan.parameters();auto directedSpans=spans;directedSpans[static_cast<size_t>(q::PooledNativeOperand::RecordBytes)].reset();
            span(directedSpans,q::PooledNativeOperand::Frontier,coordinatorRoots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.free_headers,48);
            span(directedSpans,q::PooledNativeOperand::Records,coordinatorRoots.frame,BufferId::QFN_POOLED_COORDINATOR_FRAME,c.frame.decisions,p.owner.maximumUnique()*sizeof(p::AllocationDecision));
            span(directedSpans,q::PooledNativeOperand::AllocationSlice,coordinatorRoots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,c.native.free_slices,16);
            const q::PooledImmutableOwnerOperands declaredDirected(n::OwnerAction::AcceptFree,owner,directedSpans,n::SourceCondition::Always,n::RecordLayout::DirectedAllocationBank);
            const auto directed=q::arguments_candidate::ownerNativeInputs(declaredDirected.spans(),n::OwnerAction::AcceptFree,n::SourceCondition::Always,n::RecordLayout::DirectedAllocationBank);EXPECT_EQ(directed.record_origin,c.frame.decisions);EXPECT_EQ(directed.record_capacity,p.owner.maximumUnique()*sizeof(p::AllocationDecision));EXPECT_NE(directed.allocation_slice,nullptr);
        }
    }
}
}

namespace llaminar2::test {
/** @brief Victim suffixes retain original count/descriptor/root roles and require no new channel capacity or owner fanout. */
TEST(PooledImmutableParticipantPin, VictimWaveGeometryPreservesOriginalSuffixAndNominalMessageRoles)
{
    const PhysicalMemoryAllocatorIdentity source{0,DeviceId::cuda(7)};
    for(unsigned participants:{1u,2u,4u,8u})for(bool reversed:{false,true}){std::vector<uint32_t> capacities;std::vector<PhysicalMemoryAllocatorIdentity> order;for(unsigned i=0;i<participants;++i){capacities.push_back(i%3?i+1:0);order.push_back({0,DeviceId::rocm(i)});}if(reversed){std::reverse(capacities.begin(),capacities.end());std::reverse(order.begin(),order.end());}const auto physical=declared(capacities,order,source);const auto policy_=policy(physical);const auto original=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy_,701,17);
        for(unsigned i=0;i<participants;++i){const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy_,i,701,17);const auto pin=q::PooledImmutablePinEdge::fromPlans(original,owner);const auto free=q::PooledImmutableFreeEdge::fromOriginal(pin);const auto edge=q::PooledImmutableVictimEdge::fromFree(free);const auto& c=original.parameters();
    EXPECT_EQ(edge.source(q::PooledVictimWireRole::Plan).offset,c.native.victim_headers+i*sizeof(n::VictimPlanHeader));
    EXPECT_EQ(edge.source(q::PooledVictimWireRole::Plan).bytes,48u);
    EXPECT_EQ(edge.decisionSlice().offset,c.native.victim_slices+i*sizeof(n::DirectedSlice));
    EXPECT_EQ(edge.decisionSlice().bytes,16u);
    EXPECT_EQ(edge.source(q::PooledVictimWireRole::Decisions),free.source(q::PooledFreeWireRole::Decisions));
    EXPECT_EQ(edge.destination(q::PooledVictimWireRole::Decisions).offset,owner.parameters().frame.commands);
    EXPECT_EQ(edge.source(q::PooledVictimWireRole::Grants),pin.replySource());
    EXPECT_EQ(edge.destination(q::PooledVictimWireRole::Grants),pin.replyDestination());
    EXPECT_EQ(edge.source(q::PooledVictimWireRole::Receipt).bytes,40u);
    EXPECT_EQ(edge.destination(q::PooledVictimWireRole::Receipt).offset,c.native.ingress_frontier+i*40u);
            std::vector<uint64_t> keys,variants;for(auto role:{q::PooledVictimWireRole::Plan,q::PooledVictimWireRole::Decisions,q::PooledVictimWireRole::Grants,q::PooledVictimWireRole::Receipt}){const auto key=edge.messageKey(role);for(auto old:keys)EXPECT_NE(key,old);keys.push_back(key);
    EXPECT_NE(key,pin.messageKey(q::PooledPinWireRole::Header));
    EXPECT_NE(key,free.messageKey(q::PooledFreeWireRole::Plan));for(auto endpoint:{CapturedTransferEndpoint::Producer,CapturedTransferEndpoint::Consumer}){const auto value=q::pooledVictimEndpointCaptureIdentity(pin.identity(),role,endpoint);for(auto old:variants)EXPECT_NE(value,old);variants.push_back(value);}const auto bytes=role==q::PooledVictimWireRole::Decisions?edge.maximumDecisionBytes():edge.source(role).bytes;
    EXPECT_LE(bytes,role==q::PooledVictimWireRole::Grants || role==q::PooledVictimWireRole::Receipt?free.reverseChannelMaximum():free.forwardChannelMaximum());}
        }
    }
    EXPECT_THROW(q::PooledImmutableCapturedVictimWave::bind({},{}),std::invalid_argument);
}
/** @brief The real enqueue guard/argument assembler allow only exact nominal counted victim inputs, with no directed coordinates. */
TEST(PooledImmutableParticipantPin, VictimWaveActualGuardAndAssemblerKeepOriginalCountHeaderAndC0)
{
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(7)};for(uint32_t capacity:{0u,3u}){const auto physical=declared(std::array{capacity},std::vector<PhysicalMemoryAllocatorIdentity>{{0,DeviceId::rocm(2)}},coordinator);const auto geometry=policy(physical);const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,geometry,0,701,17);const auto& p=owner.parameters();AliasRoots storage;const auto roots=storage.roots(owner);q::PooledNativeOperandSpans spans{};
        span(spans,q::PooledNativeOperand::Invocation,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress,48);span(spans,q::PooledNativeOperand::Frontier,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.header,48);if(capacity){span(spans,q::PooledNativeOperand::Records,roots.frame,BufferId::QFN_POOLED_OWNER_FRAME,p.frame.commands,p.native.reply_capacity*sizeof(p::AllocationDecision));span(spans,q::PooledNativeOperand::RecordBytes,roots.native,BufferId::QFN_POOLED_OWNER_NATIVE,p.native.ingress+48,8);}
        const q::PooledImmutableOwnerOperands inputs(n::OwnerAction::AcceptVictim,owner,spans,n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);
    EXPECT_NO_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::AcceptVictim,inputs,roots,owner));
    EXPECT_THROW(q::alias_original::ownerAliases(inputs,roots,owner.parameters()),std::invalid_argument);const auto actual=q::arguments_candidate::ownerNativeInputs(spans,n::OwnerAction::AcceptVictim,n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);
    EXPECT_EQ(actual.record_origin,0u);
    EXPECT_EQ(actual.record_capacity,0u);
    EXPECT_EQ(bool(actual.record_bytes),capacity!=0);
        auto partial=spans;partial[static_cast<size_t>(q::PooledNativeOperand::Frontier)]->offset+=8;const q::PooledImmutableOwnerOperands shifted(n::OwnerAction::AcceptVictim,owner,partial,n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);
    EXPECT_THROW(q::alias_candidate::ownerAliases(n::OwnerAction::AcceptVictim,shifted,roots,owner),std::invalid_argument);
    }
}
}

namespace llaminar2::test {
/** @brief The one original channel factory prices exact all-owner Publish56 metadata even for C0; no second transport allowance. */
TEST(PooledImmutableParticipantPin, PublicationReadyOriginalChannelPricesFullLiveFrontierOnce){
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(19)};
    for(uint32_t participants:{1u,2u,4u,8u}){std::vector<uint32_t> capacities(participants,0);std::vector<PhysicalMemoryAllocatorIdentity> order;for(uint32_t i=0;i<participants;++i)order.push_back({0,DeviceId::rocm(i)});const auto physical=declared(capacities,order,coordinator);const auto canonical=policy(physical);const auto coord=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,canonical,701,17);
        for(uint32_t i=0;i<participants;++i){const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,canonical,i,701,17);const auto pin=q::PooledImmutablePinEdge::fromPlans(coord,owner);const auto edge=q::PooledImmutableFreeEdge::fromOriginal(pin);const auto live=uint64_t{participants}*sizeof(p::PublishHeader);const auto maximum=std::max<uint64_t>(sizeof(n::SourceFrontier),live);EXPECT_EQ(edge.forwardChannelMaximum(),maximum);EXPECT_EQ(edge.reverseChannelMaximum(),64u);const auto declared=pin.channelMemory(),actual=CapturedTransferChannel::memoryFor(maximum);EXPECT_EQ(declared.mapped_host_bytes,actual.mapped_host_bytes);EXPECT_EQ(declared.cursor_bytes_per_device,actual.cursor_bytes_per_device);EXPECT_EQ(owner.parameters().native.bytes,256u);EXPECT_EQ(edge.maximumDecisionBytes(),0u);}
    }
}
}
namespace llaminar2::test {
/** @brief The original reverse channel prices the later exact live Release72 prefix once, alongside nominal60 metadata. */
TEST(PooledImmutableParticipantPin, PublicationReadyOriginalReverseChannelIncludesExactReleasePrefix){const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(19)};for(uint32_t participants:{1u,2u,4u,8u}){std::vector<uint32_t> capacities;std::vector<PhysicalMemoryAllocatorIdentity> order;for(uint32_t i=0;i<participants;++i){capacities.push_back(i%3?i+1:0);order.push_back({0,DeviceId::rocm(participants-i)});}const auto physical=declared(capacities,order,coordinator);const auto canonical=policy(physical);const auto coord=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,canonical,701,17);for(uint32_t i=0;i<participants;++i){const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,canonical,i,701,17);const auto pin=q::PooledImmutablePinEdge::fromPlans(coord,owner);const auto edge=q::PooledImmutableFreeEdge::fromOriginal(pin);const auto exact=std::max({size_t{64},size_t{owner.parameters().native.reply_capacity}*sizeof(p::ReleaseReceipt),size_t{owner.parameters().native.reply_capacity}*owner.parameters().row_bytes});EXPECT_EQ(edge.reverseChannelMaximum(),exact);const auto maximum=CapturedTransferChannel::memoryFor(exact),original=pin.returnChannelMemory();EXPECT_EQ(original.mapped_host_bytes,maximum.mapped_host_bytes);EXPECT_EQ(original.cursor_bytes_per_device,maximum.cursor_bytes_per_device);}}}
}

namespace llaminar2::test {
/** @brief One original reverse maximum covers exact native hit payloads and Release72, including C0 and large declared row counts. */
TEST(PooledImmutableParticipantPin, PublicationReadyReadChannelRepresentativeFormatsRowsAndC0){
    struct Shape{GGUFTensorType format;size_t bytes;};
    const std::array shapes{Shape{GGUFTensorType::F32,4},Shape{GGUFTensorType::F32,256},Shape{GGUFTensorType::F16,2},Shape{GGUFTensorType::F16,128},Shape{GGUFTensorType::IQ4_NL,18},Shape{GGUFTensorType::IQ4_NL,90},Shape{GGUFTensorType::Q8_0,34},Shape{GGUFTensorType::Q8_0,136}};
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(19)};
    for(const auto shape:shapes)for(uint32_t maximumRows:{1u,16u,4096u})for(uint32_t participants:{1u,2u,4u,8u}){
        std::vector<PhysicalMemoryAllocatorIdentity> order;std::vector<ImmutableRowTableLayoutOwner> declarations;bool positive=false;
        for(uint32_t i=0;i<participants;++i){const auto id=PhysicalMemoryAllocatorIdentity{0,DeviceId::rocm(participants-i)};const uint32_t capacity=i%3==0?0:i%3==1?1:maximumRows*2+1;positive|=capacity!=0;order.push_back(id);declarations.push_back({{id,ImmutableRowStorage::VRAM},0,capacity,{}});}
        const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(shape.format,1000000,shape.bytes),ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,declarations,{{0,DeviceId::cpu()},coordinator},{},{{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},positive?std::optional(coordinator):std::nullopt,maximumRows);
        const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);const auto canonical=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},std::max(64u,maximumRows));const auto coord=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,canonical,701,17);
        for(uint32_t i=0;i<participants;++i){const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,canonical,i,701,17);const auto pin=q::PooledImmutablePinEdge::fromPlans(coord,owner);const auto edge=q::PooledImmutableFreeEdge::fromOriginal(pin);const auto liveRows=size_t{owner.parameters().native.reply_capacity};const auto exact=std::max({size_t{64},liveRows*sizeof(p::ReleaseReceipt),liveRows*shape.bytes});
            SCOPED_TRACE(::testing::Message()<<static_cast<int>(shape.format)<<"/"<<shape.bytes<<"/"<<maximumRows<<"/"<<participants<<"/"<<i);
            EXPECT_EQ(owner.parameters().row_bytes,shape.bytes);EXPECT_EQ(edge.reverseChannelMaximum(),exact);
            const auto expected=CapturedTransferChannel::memoryFor(exact),actual=pin.returnChannelMemory();EXPECT_EQ(actual.mapped_host_bytes,expected.mapped_host_bytes);EXPECT_EQ(actual.cursor_bytes_per_device,expected.cursor_bytes_per_device);
            EXPECT_LE(liveRows*shape.bytes,edge.reverseChannelMaximum());EXPECT_LE(liveRows*sizeof(p::ReleaseReceipt),edge.reverseChannelMaximum());if(!liveRows)EXPECT_EQ(exact,64u);
        }
    }
}
}
