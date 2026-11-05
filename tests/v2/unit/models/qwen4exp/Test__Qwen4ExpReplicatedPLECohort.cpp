/**
 * @file Test__Qwen4ExpReplicatedPLECohort.cpp
 * @brief Device-free original source/head/communicator declaration checks.
 *
 * The real Rank constructor calls this validator. It returns no cohort/query
 * capability and cannot certify device token/history replication; canonical
 * full-model Rank lifecycle qualification remains mandatory.
 */
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include "Qwen4ExpMetadataFixture.h"
#include "../../../mocks/MockLocalTPContext.h"
#include "../../../mocks/MockModelContext.h"
#include "../../../fixtures/models/Qwen4ExpReplicatedPLECohortReplayAccess.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include <array>
#include <chrono>
#include <exception>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <string>
#include <gtest/gtest.h>
#include <algorithm>
using namespace llaminar2;
using namespace llaminar2::test;
namespace {
/** @return Published metadata geometry with unequal original head domains. */
GGUFModel sourceMetadata()
{
    auto model=qwen4MetadataFixture(false);
    std::vector<std::uint64_t> sizes(16),offsets(16);std::uint64_t next=5;
    for(std::size_t h=0;h<16;++h){offsets[h]=next;sizes[h]=h+3;next+=sizes[h]+(h==7?7:1);}
    model.metadata["qwen4exp.ple.head_offsets"]=qwen4Array(GGUFValueType::UINT64,offsets);
    model.metadata["qwen4exp.ple.head_vocab_sizes"]=qwen4Array(GGUFValueType::UINT64,sizes);
    return model;
}
/** @return Source-ordered owners, deliberately reversing physical ordinals. */
std::vector<PhysicalMemoryAllocatorIdentity> owners(int degree,DeviceType type)
{
    std::vector<PhysicalMemoryAllocatorIdentity> result;
    for(int p=0;p<degree;++p)result.push_back({0,DeviceId(type,degree-p-1)});
    return result;
}
/** @brief Configure immutable mock metadata without starting a backend. */
void configure(MockLocalTPContext &context,const std::vector<PhysicalMemoryAllocatorIdentity> &members)
{
    std::vector<GlobalDeviceAddress> addresses;
    for(const auto &member:members)addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(member.device));
    context.setDevices(std::move(addresses));
    context.setBackend(members.front().device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL);
    context.setRawAllgatherGraphCaptureSupported(true);
}
/** @return Sole checked layout declaration; no owner or memory claim is minted. */
auto declaration(const Qwen4ExpConfig &source,const std::vector<PhysicalMemoryAllocatorIdentity> &members)
{
    const auto &n=source.ngrams();
    return Qwen4ExpFixedHeadShards::declare(
        ImmutableRowTableExtent::fromNative(n.table.type,n.table.dimensions[0],n.row_bytes),
        n.head_offsets,n.head_vocab_sizes,members);
}
}
/** @test Source-head coordinate follows actual collective order, independent of range sizes. */
TEST(Qwen4ExpReplicatedPLECohort, ExactSourceOrderAndNativeCollectiveCoordinateAreRequired)
{
    const Qwen4ExpConfig source(sourceMetadata());
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const int degree:{2,4,8})
    {
        auto members=owners(degree,type);const auto placement=declaration(source,members);MockLocalTPContext context;
        configure(context,members);
        EXPECT_NO_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source));
        EXPECT_NE(placement->fixedHeadSourceRangeFor(members.front()).nativeBytes(),
            placement->fixedHeadSourceRangeFor(members.back()).nativeBytes());
        std::reverse(members.begin(),members.end());configure(context,members);
        EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
        const auto reversed=declaration(source,members);
        EXPECT_NO_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*reversed,source));
    }
}
/** @test Foreign membership, source and unsupported transport reject before graph mutation. */
TEST(Qwen4ExpReplicatedPLECohort, ForeignMembershipSourceAndCaptureTransportRejectBeforeMutation)
{
    const Qwen4ExpConfig source(sourceMetadata());const auto members=owners(2,DeviceType::CUDA);
    const auto placement=declaration(source,members);MockLocalTPContext context;configure(context,members);
    context.setRawAllgatherGraphCaptureSupported(false);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    configure(context,members);context.setBackend(CollectiveBackendType::RCCL);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    auto foreign=members;foreign[1].device=DeviceId::cuda(9);configure(context,foreign);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    foreign=members;foreign[1]=foreign[0];configure(context,foreign);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    foreign=members;foreign[1].device=DeviceId::rocm(0);configure(context,foreign);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    configure(context,owners(4,DeviceType::CUDA));
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    configure(context,members);const Qwen4ExpConfig different(qwen4MetadataFixture(false));
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,different),std::invalid_argument);
    EXPECT_EQ(context.allreduceCallCount(),0);
}

namespace {
/** @brief Test-only original LocalTP barrier/abort observations; no native transport is simulated. */
class ColdSourceContext final:public MockLocalTPContext {
public:
    struct Call {std::string phase;int participant;};
    /** @brief Preserve the actual method's two named 30-second original rendezvous contracts. */
    bool graphCaptureBoundaryRendezvous(const std::string& name,int participant,int timeout)override
    {
        EXPECT_EQ(timeout,30000);
        const int phase=name=="qfn-shared-source-materialized"?0:name=="qfn-shared-readers-materialized"?1:-1;
        if(phase<0 || participant<0 || participant>=2)throw std::invalid_argument("Unexpected original cold rendezvous");
        std::unique_lock lock(mutex_);
        calls_.push_back({name,participant});arrived_[phase][participant]=true;condition_.notify_all();
        return condition_.wait_for(lock,std::chrono::milliseconds(timeout),[&]{return aborted_ ||
            (arrived_[phase][0] && arrived_[phase][1]);}) && !aborted_;
    }
    /** @brief Observe the actual cohort owner's original abort and release blocked peers. */
    void requestAbort()override
    {std::lock_guard lock(mutex_);aborted_=true;condition_.notify_all();}
    /** @return Actual observed abort, not a successful preparation receipt. */
    bool isAbortRequested()const override{std::lock_guard lock(mutex_);return aborted_;}
    /** @brief Signal completion even when an old implementation omitted its rendezvous. */
    void finished(int participant)
    {std::lock_guard lock(mutex_);finished_[participant]=true;condition_.notify_all();}
    /** @return Whether first arrival reached phase one before returning, with bounded original timeout. */
    bool awaitFirstPhaseOrDone(int participant)
    {
        std::unique_lock lock(mutex_);
        const bool observed=condition_.wait_for(lock,std::chrono::seconds(30),[&]{return arrived_[0][participant] || finished_[participant];});
        return observed && arrived_[0][participant];
    }
    /** @return Immutable terminal call history after original participant threads have joined. */
    std::vector<Call> calls()const{std::lock_guard lock(mutex_);return calls_;}
private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::array<std::array<bool,2>,2> arrived_{};
    std::array<bool,2> finished_{};
    std::vector<Call> calls_;
    bool aborted_=false;
};

/** @return Actual validated shared metadata, with a nonzero coordinator and third storage-only GPU. */
std::shared_ptr<const ResolvedImmutableRowTablePlacement> coldSharedDeclaration(const Qwen4ExpConfig& source)
{
    ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
    policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement("0:rocm:2/0/0@0:rocm:1");
    const auto resource=[](DeviceId device){return PhysicalMemoryResource{0,device,1ull<<30,1ull<<29};};
    const std::array devices{resource(DeviceId::rocm(0)),resource(DeviceId::rocm(1)),resource(DeviceId::rocm(2))};
    const std::array consumers{devices[1],devices[0]};
    const std::array readerEdges{ImmutableRowTableReaderPeerAccess{
        {0,DeviceId::rocm(1)}, {0,DeviceId::rocm(0)}, PeerAccessCoverage::None}};
    const auto& n=source.ngrams();
    const auto input=ImmutableRowTablePlacementInput::forConsumers(
        ImmutableRowTableExtent::fromNative(n.table.type,n.table.dimensions[0],n.row_bytes),resource(DeviceId::cpu()),consumers)
        .withGPUCacheResources(policy,devices).withGPUCachePeerAccess(PeerAccessCoverage::None)
        .withReaderPeerAccess(readerEdges);
    const std::array<size_t,1> empty{};
    return Qwen4ExpReplicatedPLECohortReplayAccess::declaration(input.gpuAdaptiveLayout(policy,empty,0,16),
        policy.gpu_adaptive,n.row_elements);
}

/** @brief Join one CPU participant call and preserve its actual exception for an exact oracle. */
std::exception_ptr joinColdCall(std::future<void>& call)
{try{call.get();return {};}catch(...){return std::current_exception();}}
}

/** @test The authored source prepares first for reversed arrivals and original communicator orders. */
TEST(Qwen4ExpReplicatedPLECohort, SharedColdSourceUsesAuthoredCoordinatorAndBothOriginalPhases)
{
    const Qwen4ExpConfig source(sourceMetadata());
    const auto placement=coldSharedDeclaration(source);
    MockModelContext model;
    for(bool reversedMembers:{false,true})for(bool sourceFirst:{false,true}){
        ColdSourceContext context;
        std::vector<PhysicalMemoryAllocatorIdentity> members{{0,DeviceId::rocm(0)},{0,DeviceId::rocm(1)}};
        if(reversedMembers)std::reverse(members.begin(),members.end());
        configure(context,members);
        auto cohort=Qwen4ExpReplicatedPLECohortReplayAccess::cohort(context,model,placement,source);
        std::atomic<bool> sourceReady=false;
        std::atomic<int> sources=0,readers=0;
        const auto index=[&](DeviceId device){return int(std::find_if(members.begin(),members.end(),[&](auto member){return member.device==device;})-members.begin());};
        const auto start=[&](DeviceId device){return std::async(std::launch::async,[&,device]{
            try{
                cohort->prepareSharedSource(device,[&]{
                    if(device==DeviceId::rocm(1)){++sources;sourceReady.store(true);}
                    else{EXPECT_TRUE(sourceReady.load());++readers;}
                });
                context.finished(index(device));
            }catch(...){context.finished(index(device));throw;}
        });};
        const auto first=sourceFirst?DeviceId::rocm(1):DeviceId::rocm(0);
        const auto second=sourceFirst?DeviceId::rocm(0):DeviceId::rocm(1);
        auto firstCall=start(first);
        EXPECT_TRUE(context.awaitFirstPhaseOrDone(index(first)));
        auto secondCall=start(second);
        EXPECT_FALSE(joinColdCall(firstCall));EXPECT_FALSE(joinColdCall(secondCall));
        EXPECT_EQ(sources.load(),1);EXPECT_EQ(readers.load(),1);
        EXPECT_FALSE(context.isAbortRequested());
        const auto calls=context.calls();ASSERT_EQ(calls.size(),4u);
        for(int participant=0;participant<2;++participant){
            std::vector<std::string> phases;
            for(const auto& call:calls)if(call.participant==participant)phases.push_back(call.phase);
            EXPECT_EQ(phases,(std::vector<std::string>{"qfn-shared-source-materialized","qfn-shared-readers-materialized"}));
        }
        EXPECT_EQ(cohort->modelOwner(),&model);
        EXPECT_THROW(cohort->prepareSharedSource(DeviceId::rocm(2),[]{}),std::invalid_argument);
        EXPECT_THROW(cohort->prepareSharedSource(DeviceId::rocm(0),{}),std::invalid_argument);
    }
}

/** @test A genuine source preparation error requests original TP abort before joining the blocked foreign peer. */
TEST(Qwen4ExpReplicatedPLECohort, SharedColdSourceFailureAbortsAndUnblocksOriginalPeer)
{
    const Qwen4ExpConfig source(sourceMetadata());const auto placement=coldSharedDeclaration(source);
    MockModelContext model;ColdSourceContext context;
    const std::vector<PhysicalMemoryAllocatorIdentity> members{{0,DeviceId::rocm(0)},{0,DeviceId::rocm(1)}};
    configure(context,members);
    auto cohort=Qwen4ExpReplicatedPLECohortReplayAccess::cohort(context,model,placement,source);
    std::atomic<int> readerCalls=0;
    auto peer=std::async(std::launch::async,[&]{try{
        cohort->prepareSharedSource(DeviceId::rocm(0),[&]{++readerCalls;});context.finished(0);
    }catch(...){context.finished(0);throw;}});
    EXPECT_TRUE(context.awaitFirstPhaseOrDone(0));
    auto coordinator=std::async(std::launch::async,[&]{try{
        cohort->prepareSharedSource(DeviceId::rocm(1),[]{throw std::runtime_error("original shared source preparation failure");});
        context.finished(1);
    }catch(...){context.finished(1);throw;}});
    const auto sourceFailure=joinColdCall(coordinator);EXPECT_TRUE(sourceFailure);
    if(sourceFailure)try{std::rethrow_exception(sourceFailure);}catch(const std::runtime_error& error){EXPECT_STREQ(error.what(),"original shared source preparation failure");}
    const bool originalAbort=context.isAbortRequested();EXPECT_TRUE(originalAbort);
    // Failed implementations still release test threads, but this cleanup
    // cannot pass the preceding assertion or manufacture source readiness.
    if(!originalAbort)context.requestAbort();
    const auto peerFailure=joinColdCall(peer);EXPECT_TRUE(peerFailure);
    EXPECT_EQ(readerCalls.load(),0);
    const auto calls=context.calls();ASSERT_EQ(calls.size(),1u);
    EXPECT_EQ(calls.front().phase,"qfn-shared-source-materialized");EXPECT_EQ(calls.front().participant,0);
}

/** @test Parent deferral is exact immutable metadata, never native readiness. */
TEST(Qwen4ExpReplicatedPLECohort, SharedNativeFactoryRequiresItsOriginalRankOwnedMaterialization)
{
    const Qwen4ExpConfig source(sourceMetadata());
    MockModelContext model;
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm}) {
        const std::string backend = type == DeviceType::CUDA ? "cuda" : "rocm";
        ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
        policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement(
            "0:"+backend+":0/0/0,0:"+backend+":1/0/0@0:"+backend+":1");
        const auto resource=[](DeviceId device){return PhysicalMemoryResource{0,device,1ull<<30,1ull<<29};};
        const std::array devices{resource(DeviceId(type,0)),resource(DeviceId(type,1))};
        const std::array consumers{devices[1],devices[0]};
        const std::array readerEdges{ImmutableRowTableReaderPeerAccess{
            {0,DeviceId(type,1)},{0,DeviceId(type,0)},PeerAccessCoverage::None}};
        const auto& n=source.ngrams();
        const auto input=ImmutableRowTablePlacementInput::forConsumers(
            ImmutableRowTableExtent::fromNative(n.table.type,n.table.dimensions[0],n.row_bytes),resource(DeviceId::cpu()),consumers)
            .withGPUCacheResources(policy,devices)
            .withReaderPeerAccess(readerEdges);
        const std::array<size_t,2> empty{};
        auto placement=Qwen4ExpReplicatedPLECohortReplayAccess::declaration(
            input.gpuAdaptiveLayout(policy,empty,0,16),policy.gpu_adaptive,n.row_elements);
        MockLocalTPContext context, foreign;
        const auto members=owners(2,type);configure(context,members);configure(foreign,members);
        auto cohort=Qwen4ExpReplicatedPLECohortReplayAccess::cohort(context,model,placement,source);
        for(int coordinate=0;coordinate<2;++coordinate) {
            EXPECT_TRUE(cohort->requiresParentOwnedNativeMaterialization(context,members[coordinate].device,coordinate));
            EXPECT_THROW(cohort->requiresParentOwnedNativeMaterialization(foreign,members[coordinate].device,coordinate),std::invalid_argument);
            EXPECT_THROW(cohort->requiresParentOwnedNativeMaterialization(context,members[coordinate].device,1-coordinate),std::invalid_argument);
        }
    }
    MockLocalTPContext dedicated;configure(dedicated,owners(2,DeviceType::ROCm));
    auto placement=coldSharedDeclaration(source);
    auto cohort=Qwen4ExpReplicatedPLECohortReplayAccess::cohort(dedicated,model,placement,source);
    EXPECT_TRUE(cohort->requiresParentOwnedNativeMaterialization(dedicated,DeviceId::rocm(1),0));
}

/** @test Original Rank owns native materialization even when coordinator or sibling is a pure reader. */
TEST(Qwen4ExpReplicatedPLECohort, SharedOwnerSubsetsKeepParentMaterializationAndExactPhysicalGeometry)
{
    const Qwen4ExpConfig source(sourceMetadata());MockModelContext model;
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})
        for(unsigned mask=1;mask<4;++mask)
            for(int coordinator=0;coordinator<2;++coordinator)
                for(const std::size_t capacity:{0u,2u})
                    for(const auto backing:{AdaptiveImmutableRowBacking::RAM,AdaptiveImmutableRowBacking::SSD}){
                        SCOPED_TRACE(::testing::Message()<<int(type)<<":"<<mask<<":"<<coordinator<<":"<<capacity<<":"<<int(backing));
                        const std::string backend=type==DeviceType::CUDA?"cuda":"rocm";
                        std::string declaration;
                        for(unsigned owner=0;owner<2;++owner)if(mask&(1u<<owner)){
                            if(!declaration.empty())declaration+=",";
                            declaration+="0:"+backend+":"+std::to_string(owner)+"/"+std::to_string(capacity)+"/"+std::to_string(capacity?1048576:0);
                        }
                        ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
                        policy.gpu_adaptive.source_backing=backing;
                        policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement(declaration+"@0:"+backend+":"+std::to_string(coordinator));
                        const auto resource=[](DeviceId device){return PhysicalMemoryResource{0,device,1ull<<30,1ull<<29};};
                        const std::array devices{resource(DeviceId(type,0)),resource(DeviceId(type,1))};
                        const std::array consumers{devices[1],devices[0]};const auto& n=source.ngrams();
                        const auto input=ImmutableRowTablePlacementInput::forConsumers(
                            ImmutableRowTableExtent::fromNative(n.table.type,n.table.dimensions[0],n.row_bytes),resource(DeviceId::cpu()),consumers)
                            .withGPUCacheResources(policy,devices);
                        const std::vector<std::size_t> slots(policy.gpu_adaptive.placement.owners.size(),capacity);
                        const auto layout=input.gpuAdaptiveLayout(policy,slots,0,16);
                        EXPECT_FALSE(layout.peerAccess());EXPECT_TRUE(layout.readerPeerAccess().empty());
                        for(unsigned participant=0;participant<2;++participant){
                            const auto found=std::find_if(layout.owners().begin(),layout.owners().end(),[&](const auto& owner){
                                return owner.key.tier==ImmutableRowStorage::VRAM && owner.key.allocator.device==DeviceId(type,int(participant));});
                            if(mask&(1u<<participant)){
                                ASSERT_NE(found,layout.owners().end());EXPECT_EQ(found->cache_rows,capacity);
                                EXPECT_EQ(layout.nativePayloadBytes(found->key),capacity*n.row_bytes);
                            }else EXPECT_EQ(found,layout.owners().end());
                        }
                        auto placement=Qwen4ExpReplicatedPLECohortReplayAccess::declaration(layout,policy.gpu_adaptive,n.row_elements);
                        MockLocalTPContext context,foreign;const auto members=owners(2,type);configure(context,members);configure(foreign,members);
                        auto cohort=Qwen4ExpReplicatedPLECohortReplayAccess::cohort(context,model,placement,source);
                        for(int coordinate=0;coordinate<2;++coordinate){
                            EXPECT_TRUE(cohort->requiresParentOwnedNativeMaterialization(context,members[coordinate].device,coordinate));
                            EXPECT_THROW(cohort->requiresParentOwnedNativeMaterialization(foreign,members[coordinate].device,coordinate),std::invalid_argument);
                        }
                    }
}

/** @test Every original addressed reader keeps Rank-owned materialization without manufacturing a physical cache role. */
TEST(Qwen4ExpReplicatedPLECohort, AddressedReadersEveryDegreeKeepOriginalParentMaterialization) {
    const Qwen4ExpConfig source(sourceMetadata());MockModelContext model;const auto& n=source.ngrams();
    const auto resource=[](DeviceId d){return PhysicalMemoryResource{0,d,1ull<<30,1ull<<29};};
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const int degree:{2,4,8})for(const int coordinator:{0,degree-1})
        for(const bool ownerIsSource:{false,true})for(const std::size_t capacity:{0u,2u}) {
            const std::string backend=type==DeviceType::CUDA?"cuda":"rocm";const int owner=ownerIsSource?coordinator:(coordinator+1)%degree;
            ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
            policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement("0:"+backend+":"+std::to_string(owner)+"/"+
                std::to_string(capacity)+"/"+std::to_string(capacity?1048576:0)+"@0:"+backend+":"+std::to_string(coordinator));
            const auto members=owners(degree,type);std::vector<PhysicalMemoryResource> resources;
            for(const auto member:members)resources.push_back(resource(member.device));
            const auto input=ImmutableRowTablePlacementInput::forConsumers(
                ImmutableRowTableExtent::fromNative(n.table.type,n.table.dimensions[0],n.row_bytes),resource(DeviceId::cpu()),resources)
                .withGPUCacheResources(policy,resources);
            const std::array<std::size_t,1> slots{capacity};
            auto placement=Qwen4ExpReplicatedPLECohortReplayAccess::declaration(input.gpuAdaptiveLayout(policy,slots,0,16),policy.gpu_adaptive,n.row_elements);
            MockLocalTPContext context,foreign;configure(context,members);configure(foreign,members);
            auto cohort=Qwen4ExpReplicatedPLECohortReplayAccess::cohort(context,model,placement,source);
            for(int coordinate=0;coordinate<degree;++coordinate){
                EXPECT_TRUE(cohort->requiresParentOwnedNativeMaterialization(context,members[coordinate].device,coordinate));
                EXPECT_THROW(cohort->requiresParentOwnedNativeMaterialization(foreign,members[coordinate].device,coordinate),std::invalid_argument);
            }
        }
}
