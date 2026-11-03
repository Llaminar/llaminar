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
