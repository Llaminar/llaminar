/**
 * @file PooledImmutableReadGraphFixture.h
 * @brief Original source/owner graph insertion and retirement-chain replay over declared original authorities.
 *
 * Backend API doubles replace only the native loader boundary. Actual root
 * factories, PMA claims/borrows, graph stages and insertion helpers execute
 * unchanged. Host allocations stand in for device APIs, proving ownership
 * interfaces rather than native storage, transport, capture or GPU work.
 */
#pragma once
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "execution/qwen4exp/PooledImmutableParticipantFreeGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantVictimGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantLocalGraph.h"
#include "execution/qwen4exp/PooledImmutableLocalLookupGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantSourceGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantPublicationGraph.h"
#include "loaders/HostImmutableRowProvider.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include <fstream>
#include <atomic>
#include <chrono>
#include <thread>
#include <source_location>
#include "kernels/qwen4exp/PooledImmutableSourceOps.h"
#include <cstdio>
#include <filesystem>
#include <unistd.h>
#include "backends/BackendManager.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "mocks/MockBackend.h"
#include "mocks/MockWorkerGPUContext.h"
#include "fixtures/qwen4exp/PooledImmutableFactoryReplayAccess.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
namespace llaminar2::test {
namespace q=qwen4exp;namespace n=q::pooled_native;
/** @brief Actual allocator records around aligned host API-double allocations. */
class OwnerChainBackend final:public MockBackend {
public:
    /** @brief Install a family without probing hardware or initializing a vendor runtime. */
    explicit OwnerChainBackend(DeviceType type):MockBackend(type),type_(type){setDeviceCount(8);}
    /** @brief No mocked physical owner may outlive its exact backend record. */
    ~OwnerChainBackend()override{if(!allocations_.empty() || !registrations_.empty() || !mapped_.empty())std::terminate();}
    /** @brief Allocate exactly the requested bytes; each live record has one original capture token. */
    void* allocate(size_t bytes,int ordinal)override{++allocationCalls;if(failAllocationCall && allocationCalls==*failAllocationCall)return nullptr;void* p=nullptr;if(posix_memalign(&p,64,bytes))return nullptr;allocations_.emplace(p,Record{bytes,ordinal,std::make_shared<const NativeAllocationCaptureIdentity>()});return p;}
    /** @brief Reject borrowed reclamation before freeing the actual host-double allocation. */
    void free(void* p,int ordinal)override{if(!p)return;auto it=allocations_.find(p);if(it==allocations_.end() || it->second.ordinal!=ordinal)std::terminate();NativeAllocationCaptureLease::requireUnborrowed(it->second.identity);allocations_.erase(it);std::free(p);}
    /** @brief Borrow the exact actual allocator record, including legal interior spans. */
    NativeAllocationCaptureLease retainNativeAllocation(const void* p,size_t bytes,int ordinal)override{
        const auto start=reinterpret_cast<uintptr_t>(p);for(const auto& [base,r]:allocations_){const auto first=reinterpret_cast<uintptr_t>(base);if(start>=first && start-first<r.bytes)return NativeAllocationCaptureLease::acquire(r.identity,base,r.bytes,DeviceId{type_,r.ordinal},p,bytes,DeviceId{type_,ordinal});}throw std::invalid_argument("Owner chain mock allocation record is foreign");
    }
    /** @brief Actual service mapped-allocation API double, preserving original64-byte protocol alignment. */
    void* allocateMapped(size_t bytes,int,void** alias)override{void* p=std::aligned_alloc(64,(bytes+63)/64*64);if(p)mapped_.insert(p);if(alias)*alias=p;return p;}
    /** @brief Final original source backing retires only after all actual worker/binding owners release it. */
    void freeMapped(void* p,int)override{if(mapped_.erase(p)!=1)std::terminate();std::free(p);}
    /** @brief Register only the original host mapping; the test never emulates a native peer route. */
    bool registerExternalMappedHostMemory(void* p,size_t,int,MappedHostRegistrationScope)override{return registrations_.insert(p).second;}
    /** @brief Expose the same host-double mapping to the matching backend. */
    bool externalMappedHostDevicePointer(void* p,int,void** alias)override{if(!alias || !registrations_.contains(p))return false;*alias=p;return true;}
    /** @brief Remove registration before its original host mapping is retired. */
    bool unregisterExternalMappedHostMemory(void* p,int)override{return registrations_.erase(p)==1;}
    /** @return Stable API-double runtime generation for this exact backend lifetime, never a vendor observation. */
    uint64_t deviceRuntimeGeneration(int ordinal)const override{return ordinal>=0 && ordinal<8?generation_:0;}
    /** @brief Inject one actual setup join failure; record exact ordinary backend calls otherwise. */
    bool streamWaitEvent(void* stream,void* event,int ordinal)override {
        if(failJoinOrdinal && *failJoinOrdinal==ordinal){failJoinOrdinal.reset();return false;}
        return MockBackend::streamWaitEvent(stream,event,ordinal);
    }
    std::size_t allocationCalls=0;
    std::optional<std::size_t> failAllocationCall;
    std::optional<int> failJoinOrdinal,failEventCreationOrdinal;
    /** @brief One failed native event creation is observed before any recording work. */
    void* createEvent(int ordinal)override {
        ++eventCreates;if(failEventCreationCall && eventCreates==*failEventCreationCall)return nullptr;
        if(failEventCreationOrdinal && *failEventCreationOrdinal==ordinal){failEventCreationOrdinal.reset();return nullptr;}
        return MockBackend::createEvent(ordinal);
    }
    std::size_t eventCreates=0;
    std::optional<std::size_t> failEventCreationCall;
    /** @return Actual live physical allocations, not a duplicate byte ledger. */
    std::size_t liveAllocations()const{return allocations_.size();}
    std::function<void()> beforeRetirementWait;
    bool waitForEvent(void* event,int ordinal)override{if(beforeRetirementWait)beforeRetirementWait();return MockBackend::waitForEvent(event,ordinal);}
    unsigned capturedTransfers=0;
    /** @brief Record the actual original transfer entry under the executor's capture guard, without executing a wire. */
    bool enqueueCapturedTransferChannel(const CapturedTransferChannelDeviceBinding&,void*,const void*,int,void* stream)override {
        if(!isGraphCaptureActive() || !stream)return false;++capturedTransfers;return true;
    }
    /** @brief Cold protocol geometry only; no kernel/vendor API is submitted. */
    bool prepareCapturedTransferChannelKernels(int,int timeout,uint64_t* ticks)override{if(timeout!=30000 || !ticks)return false;*ticks=42;return true;}
    /** @brief Materialize only the original cursor setup bytes using the explicit host API double. */
    bool copyDeviceVisibleRegionByKernelOnStream(void* dst,const void* src,size_t bytes,int,void* stream)override{if(!stream || reinterpret_cast<uintptr_t>(stream)<=2)return false;std::memcpy(dst,src,bytes);return true;}
    /** @brief Observe only cold setup-event readiness. */
    bool queryEvent(void*,int,bool* ready)override{*ready=true;return true;}
private:
    struct Record{size_t bytes;int ordinal;std::shared_ptr<const NativeAllocationCaptureIdentity> identity;};
    inline static std::atomic<uint64_t> nextGeneration_{1};
    const uint64_t generation_=nextGeneration_.fetch_add(1);
    DeviceType type_;std::map<void*,Record> allocations_;std::set<void*> registrations_,mapped_;
};
OwnerChainBackend* chainCuda=nullptr;OwnerChainBackend* chainRocm=nullptr;
/** @brief Isolate both API doubles until all concrete graph/storage owners are destroyed. */
struct OwnerChainBackendScope {
    OwnerChainBackend cudaStorage{DeviceType::CUDA},rocmStorage{DeviceType::ROCm};
    std::shared_ptr<OwnerChainBackendScope> original;
    OwnerChainBackend& cuda;OwnerChainBackend& rocm;
    explicit OwnerChainBackendScope(std::shared_ptr<OwnerChainBackendScope> shared={})
        :original(std::move(shared)),cuda(original?original->cuda:cudaStorage),rocm(original?original->rocm:rocmStorage){
        if(!original){testing::installHardwareFreeGPUContextFactories();chainCuda=&cuda;chainRocm=&rocm;}
    }
    ~OwnerChainBackendScope(){if(!original){PooledImmutableFactoryReplayAccess::retireFixtureGenerations();chainCuda=nullptr;chainRocm=nullptr;}}
};
/** @brief Host API boundary: actual common setup initializes metadata; capture records entrypoints without executing GPU kernels. */
class OwnerChainCode final:public q::IPooledImmutableNativeKernels {
public:
    explicit OwnerChainCode(DeviceType family=DeviceType::ROCm):family_(family){}
    uint64_t codeIdentity()const noexcept override{return 0x434841494eULL^static_cast<uint64_t>(family_);}
    DeviceType backend()const noexcept override{return family_;}
protected:
    /** @brief Host API-double kernel entry executes actual canonical metadata setup, never a vendor launch. */
    void initializeOwner(const n::OwnerArgs& a,void*)const override{
        n::seedOwner<q::adaptive_lookup::Slot32>(a);auto operation=n::p::OwnerOperations<q::adaptive_lookup::Slot32>(n::ownerView<q::adaptive_lookup::Slot32>(a));
        for(uint32_t slot=0;slot<a.plan.owner.capacity();++slot)operation.initializeSlot(slot);
        if(!n::sealOwner<q::adaptive_lookup::Slot32>(a))throw std::logic_error("Actual owner setup rejected");
    }
    /** @brief Keep the original declared owner order and actual coordinator initializer. */
    void initializeCoordinator(const n::CoordinatorSetupArgs& a,std::span<const n::p::CoordinatorOwnerGeometry> owners,void*)const override{
        n::seedCoordinator(a);std::copy(owners.begin(),owners.end(),n::at<n::p::CoordinatorOwnerGeometry>(a.frame,a.plan.frame.owners));
        if(!n::p::CoordinatorOperations(n::coordinatorView(n::CoordinatorArgs{a.plan,a.cache,a.frame,a.native,{}})).initialize())throw std::logic_error("Actual coordinator setup rejected");
    }
    void launch(n::OwnerAction,const n::OwnerArgs&,void*)const override{if(!isGraphCaptureActive())throw std::logic_error("Unit cannot execute a native kernel");++recorded;}
    void launch(n::CoordinatorAction,const n::CoordinatorArgs&,const n::CoordinatorQuery*,void*)const override{if(!isGraphCaptureActive())throw std::logic_error("Unit cannot execute a native kernel");++recorded;}
public:
    mutable unsigned recorded=0;
private:
    DeviceType family_;
};
/** @brief Exact original one-owner declarations, retaining C0 and original independent allocator roles. */
q::PooledImmutablePhysicalOwnerGeometry chainGeometry(uint32_t capacity){
    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(0)},owner{0,DeviceId::rocm(0)};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,1000,90),ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
        {{{owner,ImmutableRowStorage::VRAM},0,capacity,{}}},{{0,DeviceId::cpu()},coordinator},{},{{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},capacity?std::optional(coordinator):std::nullopt,16);
    const std::array order{owner};return q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
}
/** @brief Original test observations only; no device or driver query. */
PhysicalMemoryResource chainObserved(PhysicalMemoryAllocatorIdentity id){return {id.world_rank,id.device,1ULL<<28,1ULL<<28};}
/** @brief Price actual original native roots and exact original channel geometries once before materialization. */
std::shared_ptr<PhysicalMemoryAuthority> chainAuthority(const q::PooledImmutablePhysicalOwnerGeometry& physical,const q::PooledImmutableCoordinatorGeometry& policy,const q::PooledImmutableCoordinatorNativePlan& coordinator,const q::PooledImmutableOwnerNativePlan& owner,unsigned reverseChannels,bool queryInputs){
    PhysicalMemoryPlanBuilder b;const auto add=[&](const PhysicalMemoryPlan& plan){for(const auto& resource:plan.resources())b.add(resource);};
    add(policy.policyBOM(chainObserved(coordinator.allocator())));add(policy.mandatoryBOM(chainObserved(coordinator.allocator())));add(coordinator.mandatoryBOM(chainObserved(coordinator.allocator())));
    llaminar2::test::PooledImmutableFactoryReplayAccess::addOrdering(b,coordinator.orderingProgram(),chainObserved({0,DeviceId::cpu()}),chainObserved(coordinator.allocator()));
    add(physical.cacheBOM(0,chainObserved(owner.allocator())));add(physical.mandatoryBOM(0,chainObserved(owner.allocator())));add(owner.mandatoryBOM(chainObserved(owner.allocator())));
    llaminar2::test::PooledImmutableFactoryReplayAccess::addOrdering(b,owner.orderingProgram(),chainObserved({0,DeviceId::cpu()}),chainObserved(owner.allocator()));
    if(queryInputs){
        const CPUTensorStorageDeclaration ids(1,8,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu());
        const CPUTensorStorageDeclaration admission(1,2,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu());
        b.add(chainObserved({0,DeviceId::cpu()}),PhysicalMemoryOwner::ImmutableLookupStaging,ids.physicalBytes()+admission.physicalBytes());
    }
    const auto edge=q::PooledImmutablePinEdge::fromPlans(coordinator,owner);
    for(unsigned channel=0;channel<=reverseChannels;++channel){const auto geometry=channel?edge.returnChannelMemory():edge.channelMemory();
        b.add(chainObserved({0,DeviceId::cpu()}),PhysicalMemoryOwner::ActivationTransportStaging,geometry.mapped_host_bytes);
        b.add(chainObserved(coordinator.allocator()),PhysicalMemoryOwner::ActivationTransportStaging,geometry.cursor_bytes_per_device);
        b.add(chainObserved(owner.allocator()),PhysicalMemoryOwner::ActivationTransportStaging,geometry.cursor_bytes_per_device);
    }
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(b.build()),0);
}
/** @brief Concrete original roots/channels and arenas; no adoption hook or fake prepared owner is used. */
struct OwnerChainCase {
    OwnerChainBackendScope backends;
    q::PooledImmutablePhysicalOwnerGeometry geometry;
    q::PooledImmutableCoordinatorGeometry policy;
    q::PooledImmutableCoordinatorNativePlan coordinatorPlan;
    q::PooledImmutableOwnerNativePlan ownerPlan;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
    std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> owner;
    std::shared_ptr<const q::PooledImmutableCapturedPinEdge> forward;
    std::shared_ptr<const q::PooledImmutableCapturedPinReturn> returned,foreign;
    std::shared_ptr<const q::PooledImmutableCapturedFreeWave> free;
    std::shared_ptr<OwnerChainCode> code=std::make_shared<OwnerChainCode>();
    BufferArena arena;ComputeGraph graph;PooledPinParticipantNodes incoming,complete,otherComplete;
    explicit OwnerChainCase(uint32_t capacity,bool extraReturn=false,bool queryInputs=false):geometry(chainGeometry(capacity)),policy(q::PooledImmutableCoordinatorGeometry::fromOwners(geometry,{65536,1,2},16)),coordinatorPlan(q::PooledImmutableCoordinatorNativePlan::fromGeometry(geometry,policy,701,17)),ownerPlan(q::PooledImmutableOwnerNativePlan::fromGeometry(geometry,policy,0,701,17)),memory(chainAuthority(geometry,policy,coordinatorPlan,ownerPlan,extraReturn?2u:1u,queryInputs)){
        auto* sourceStream=reinterpret_cast<void*>(0x101);auto* ownerStream=reinterpret_cast<void*>(0x202);
        coordinator=llaminar2::test::PooledImmutableFactoryReplayAccess::coordinator(coordinatorPlan,memory,sourceStream);owner=llaminar2::test::PooledImmutableFactoryReplayAccess::owner(ownerPlan,memory,ownerStream);
        forward=q::PooledImmutableCapturedPinEdge::bind(coordinator,owner,memory,sourceStream,ownerStream);returned=q::PooledImmutableCapturedPinReturn::bind(forward,memory,ownerStream,sourceStream);if(extraReturn)foreign=q::PooledImmutableCapturedPinReturn::bind(forward,memory,ownerStream,sourceStream);free=q::PooledImmutableCapturedFreeWave::bind(returned,memory);
        bindPooledNativeRoots(arena,*owner);incoming=appendPooledPinReceiver(graph,arena,forward,code,"incoming");complete=appendPooledPinOwnerReturn(graph,arena,returned,code,incoming.last,"returned");if(extraReturn)otherComplete=appendPooledPinOwnerReturn(graph,arena,foreign,code,incoming.last,"foreign");
    }
    /** @brief Construct the reported same-root AcquirePins phase with a caller-selected predecessor. */
    std::string acquire(const std::string& dependency){const std::string name="manual_acquire";graph.addNode(name,std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{n::OwnerAction::AcquirePins,owner,q::PooledImmutableOwnerOperands(n::OwnerAction::AcquirePins,ownerPlan,forward->ownerInputs(n::OwnerAction::AcquirePins))}),ownerPlan.allocator().device);if(!dependency.empty())graph.addDependency(name,dependency);return name;}
};
/** @brief Both real positive-capacity and C0 Pin return chains can append the concrete ordinary free wave. */

/** @brief Same-root AcquirePins with missing, foreign or out-of-order predecessors rejects before any free graph mutation. */

/** @brief An intermediate predecessor replacement cannot masquerade as the original channel retirement. */

}
namespace llaminar2::test {
/** @brief Original decoded initial query declarations for cold source-stage construction, not a fabricated source service. */
struct VictimCoordinatorCase {
    OwnerChainCase owner;
    std::shared_ptr<TensorBase> ids,admission;
    std::shared_ptr<OwnerChainCode> code=std::make_shared<OwnerChainCode>(DeviceType::CUDA);
    BufferArena arena;ComputeGraph graph;PooledPinParticipantNodes pin,free;
    explicit VictimCoordinatorCase(uint32_t capacity):owner(capacity,false,true){
        ids=CPUTensorStorageDeclaration(1,8,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu()).materialize(owner.memory);
        admission=CPUTensorStorageDeclaration(1,2,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu()).materialize(owner.memory);
        bindPooledNativeRoots(arena,*owner.coordinator);if(!arena.registerExternalBuffer(BufferId::QFN_NGRAM_ROW_IDS,ids) || !arena.registerExternalBuffer(BufferId::QFN_POOLED_SOURCE_NATIVE,admission))throw std::logic_error("Initial query fixture could not retain its original decoded declarations");
        q::PooledNativeOperandSpans begin{};begin[static_cast<size_t>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,32};begin[static_cast<size_t>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{admission,BufferId::QFN_POOLED_SOURCE_NATIVE,0,8};
        graph.addNode("begin",std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::Begin,owner.coordinator,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::Begin,owner.coordinatorPlan,n::OriginalRows{DeviceRowRange::fullyActive(4),n::OriginalIdWidth::UInt64},begin)}),DeviceId::cuda(0));
        graph.addNode("commands",std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::PinCommands,owner.coordinator,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::PinCommands,owner.coordinatorPlan,{}, {})}),DeviceId::cuda(0));graph.addDependency("commands","begin");
        const auto sent=appendPooledPinSender(graph,arena,owner.forward,"begin","commands","pin_send");const std::array edges{owner.returned};const std::array sentNames{sent.last};pin=appendPooledPinCoordinatorReturn(graph,arena,edges,code,sentNames,"commands","pin_return");const std::array freeEdges{owner.free};free=appendPooledFreeCoordinatorWave(graph,arena,freeEdges,code,pin.last,"free");
    }
};
/** @brief Actual positive/C0 original free chains append only their real victim endpoints; no owner receipt fanout exists. */

/** @brief Same-root free completion without the original header/count/read chain fails before any victim node mutation. */

}
namespace llaminar2::test {
/** @brief Concrete original coordinator free send/read chains authorize only the real victim gather; no unused fanout. */

/** @brief Skipped/foreign free frontier, missing read barriers and out-of-order receipt consumers reject before victim mutation. */

}

namespace llaminar2::test {
/** @brief Actual original multi-owner root/channel cohort, preserving physical order and C0 declarations.
 *
 * Only the backend API boundary is doubled. Original checked layout factories,
 * PMA materialization, independent channels, root contracts, native stages and
 * participant-local graph builders execute unchanged. No vendor runtime runs.
 */
struct VictimMultiCoordinatorCase {
    OwnerChainBackendScope backends;
    q::PooledImmutablePhysicalOwnerGeometry geometry;
    q::PooledImmutableCoordinatorGeometry policy;
    q::PooledImmutableCoordinatorNativePlan coordinatorPlan;
    std::vector<q::PooledImmutableOwnerNativePlan> ownerPlans;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
    std::vector<std::shared_ptr<q::PooledImmutableOwnerPreparedStorage>> owners;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinEdge>> forwards;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> returns;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> frees;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> victims;
    std::shared_ptr<const q::PooledImmutableLocalOwnerBinding> localBinding;
    std::optional<PooledImmutableMixedPinCompletion> mixedPin;
    std::optional<PooledImmutableMixedFreeCompletion> mixedFree;
    std::vector<q::PooledImmutableVictimParticipant> mixedVictims;
    std::shared_ptr<TensorBase> ids,admission;
    std::shared_ptr<OwnerChainCode> code=std::make_shared<OwnerChainCode>(DeviceType::CUDA);
    BufferArena arena;ComputeGraph graph;PooledPinParticipantNodes pin,free;
    /** @brief Derive the original layout once from exact caller membership/capacities, including reversed device order. */
    static q::PooledImmutablePhysicalOwnerGeometry declared(const std::vector<uint32_t>& capacities,bool reversed,std::optional<size_t> localCoordinate=std::nullopt){
        const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId::cuda(0)};
        std::vector<PhysicalMemoryAllocatorIdentity> order;std::vector<ImmutableRowTableLayoutOwner> owners;bool positive=false;
        for(size_t i=0;i<capacities.size();++i){const PhysicalMemoryAllocatorIdentity id=localCoordinate && i==*localCoordinate?coordinator:PhysicalMemoryAllocatorIdentity{0,DeviceId::rocm(static_cast<int>(reversed?capacities.size()-1-i:i))};order.push_back(id);owners.push_back({{id,ImmutableRowStorage::VRAM},0,capacities[i],{}});positive|=capacities[i]!=0;}
        const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
        const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,1000,90),ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,{{0,DeviceId::cpu()},coordinator},{},{{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},positive?std::optional(coordinator):std::nullopt,16);
        return q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
    }
    /** @brief Price each original physical root and each original forward/reverse channel exactly once in one PMA. */
    explicit VictimMultiCoordinatorCase(const std::vector<uint32_t>& capacities,bool reversed,std::optional<size_t> localCoordinate=std::nullopt):geometry(declared(capacities,reversed,localCoordinate)),policy(q::PooledImmutableCoordinatorGeometry::fromOwners(geometry,{65536,1,2},16)),coordinatorPlan(q::PooledImmutableCoordinatorNativePlan::fromGeometry(geometry,policy,701,17)){
        PhysicalMemoryPlanBuilder b;const auto add=[&](const PhysicalMemoryPlan& p){for(const auto& r:p.resources())b.add(r);};
        add(policy.policyBOM(chainObserved(coordinatorPlan.allocator())));add(policy.mandatoryBOM(chainObserved(coordinatorPlan.allocator())));add(coordinatorPlan.mandatoryBOM(chainObserved(coordinatorPlan.allocator())));
        llaminar2::test::PooledImmutableFactoryReplayAccess::addOrdering(b,coordinatorPlan.orderingProgram(),chainObserved({0,DeviceId::cpu()}),chainObserved(coordinatorPlan.allocator()));
        const CPUTensorStorageDeclaration idDeclaration(1,8,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu()),admissionDeclaration(1,2,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu());
        b.add(chainObserved({0,DeviceId::cpu()}),PhysicalMemoryOwner::ImmutableLookupStaging,idDeclaration.physicalBytes()+admissionDeclaration.physicalBytes());
        for(size_t i=0;i<capacities.size();++i){ownerPlans.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(geometry,policy,i,701,17));const auto& o=ownerPlans.back();add(geometry.cacheBOM(i,chainObserved(o.allocator())));add(geometry.mandatoryBOM(i,chainObserved(o.allocator())));add(o.mandatoryBOM(chainObserved(o.allocator())));llaminar2::test::PooledImmutableFactoryReplayAccess::addOrdering(b,o.orderingProgram(),chainObserved({0,DeviceId::cpu()}),chainObserved(o.allocator()));
            const auto e=q::PooledImmutablePinEdge::fromPlans(coordinatorPlan,o);if(e.remote())for(const auto channel:{e.channelMemory(),e.returnChannelMemory()}){b.add(chainObserved({0,DeviceId::cpu()}),PhysicalMemoryOwner::ActivationTransportStaging,channel.mapped_host_bytes);b.add(chainObserved(coordinatorPlan.allocator()),PhysicalMemoryOwner::ActivationTransportStaging,channel.cursor_bytes_per_device);b.add(chainObserved(o.allocator()),PhysicalMemoryOwner::ActivationTransportStaging,channel.cursor_bytes_per_device);}}
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(b.build()),0);
        auto* stream=reinterpret_cast<void*>(0x101);coordinator=llaminar2::test::PooledImmutableFactoryReplayAccess::coordinator(coordinatorPlan,memory,stream);
        forwards.resize(ownerPlans.size());returns.resize(ownerPlans.size());frees.resize(ownerPlans.size());victims.resize(ownerPlans.size());
        for(size_t i=0;i<ownerPlans.size();++i){auto* ownerStream=reinterpret_cast<void*>(0x202);owners.push_back(llaminar2::test::PooledImmutableFactoryReplayAccess::owner(ownerPlans[i],memory,ownerStream));
            if(localCoordinate && i==*localCoordinate)localBinding=q::PooledImmutableLocalOwnerBinding::bind(coordinator,owners.back(),memory);
            else{forwards[i]=q::PooledImmutableCapturedPinEdge::bind(coordinator,owners.back(),memory,stream,ownerStream);returns[i]=q::PooledImmutableCapturedPinReturn::bind(forwards[i],memory,ownerStream,stream);frees[i]=q::PooledImmutableCapturedFreeWave::bind(returns[i],memory);victims[i]=q::PooledImmutableCapturedVictimWave::bind(frees[i],memory);}}

        ids=idDeclaration.materialize(memory);admission=admissionDeclaration.materialize(memory);bindPooledNativeRoots(arena,*coordinator);
        if(localBinding)bindPooledNativeRoots(arena,*localBinding->ownerStorage());
        if(!arena.registerExternalBuffer(BufferId::QFN_NGRAM_ROW_IDS,ids) || !arena.registerExternalBuffer(BufferId::QFN_POOLED_SOURCE_NATIVE,admission))throw std::logic_error("Multi-owner initial query declarations were replaced");
        q::PooledNativeOperandSpans begin{};begin[static_cast<size_t>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,32};begin[static_cast<size_t>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{admission,BufferId::QFN_POOLED_SOURCE_NATIVE,0,8};
        graph.addNode("begin",std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::Begin,coordinator,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::Begin,coordinatorPlan,n::OriginalRows{DeviceRowRange::fullyActive(4),n::OriginalIdWidth::UInt64},begin)}),DeviceId::cuda(0));
        graph.addNode("commands",std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::PinCommands,coordinator,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::PinCommands,coordinatorPlan,{}, {})}),DeviceId::cuda(0));graph.addDependency("commands","begin");
        std::vector<std::string> sent(forwards.size());for(size_t i=0;i<forwards.size();++i)if(forwards[i])sent[i]=appendPooledPinSender(graph,arena,forwards[i],"begin","commands","pin_send"+std::to_string(i)).last;
        if(localBinding){
            std::vector<q::PooledImmutablePinParticipant> pinMembers;std::vector<q::PooledImmutableFreeParticipant> freeMembers;
            for(size_t i=0;i<owners.size();++i){if(i==localBinding->coordinate()){pinMembers.emplace_back(localBinding);freeMembers.emplace_back(localBinding);mixedVictims.emplace_back(localBinding);}else{pinMembers.emplace_back(returns[i]);freeMembers.emplace_back(frees[i]);mixedVictims.emplace_back(victims[i]);}}
            mixedPin=appendPooledPinCoordinatorMembers(graph,arena,pinMembers,code,sent,"begin","commands","pin_return");pin=mixedPin->coordinator;
            mixedFree=appendPooledFreeCoordinatorMembers(graph,arena,freeMembers,code,pin.last,"free",&*mixedPin);free=mixedFree->coordinator;
        }else{pin=appendPooledPinCoordinatorReturn(graph,arena,returns,code,sent,"commands","pin_return");free=appendPooledFreeCoordinatorWave(graph,arena,frees,code,pin.last,"free");}
    }
    /** @return Real shared-bank read endpoint for an original owner, including payload-free C0. */
    ComputeNode* read(size_t owner){return graph.getNode("free/owner"+std::to_string(owner)+(frees[owner]->declaration().hasPayload()?"/grant_receive":"/receipt_receive"));}
    /** @return Exact original sender whose completion the serial reply bank must await. */
    std::string sender(size_t owner)const{return "free/owner"+std::to_string(owner)+(frees[owner]->declaration().hasPayload()?"/decisions_send":"/plan_send");}
};
/** @brief Exercise actual 2/4/8-owner cohorts with unequal positive/C0 banks, reversed physical order and all-empty cohorts. */
template<class F> void victimMultiCohorts(F&& replay){for(size_t count:{2u,4u,8u})for(bool reversed:{false,true})for(bool empty:{false,true}){std::vector<uint32_t> capacities;for(size_t i=0;i<count;++i)capacities.push_back(empty || i%3==0?0:static_cast<uint32_t>(i+1));if(reversed)std::reverse(capacities.begin(),capacities.end());replay(capacities,reversed);}}
/** @brief Real original multi-owner free chains authenticate and append the actual victim gather before no unused owner fanout. */

/** @brief A later reply cannot overwrite the shared bank before the actual previous-owner receipt retires. */

/** @brief Every later owner waits for every original sender, including distant positive and C0 producers, before bank reuse. */

}

namespace llaminar2::test {
/** @brief Only native code entrypoints are doubled: original service/provider/geometry/claims still execute. */
class SourceGuardCode final:public q::IPooledImmutableSourceKernels {
public:
    SourceGuardCode(q::PooledImmutableSourceGeometry geometry,q::ImmutableSubordinateLookupBinding source):geometry_(std::move(geometry)),source_(std::move(source)){}
    const q::PooledImmutableSourceGeometry& geometry()const noexcept override{return geometry_;}
    const q::ImmutableSubordinateLookupBinding& source()const noexcept override{return source_;}
    mutable unsigned launches=0,closes=0;
    mutable bool failAdmission=false;
    mutable std::optional<q::PooledSourceAction> failRecordedAction;
protected:
    uint64_t codeIdentity()const noexcept override{return 0x534f555243454450ULL^geometry_.parameters().identity;}
    void initializeNative(const q::pooled_source::DeviceArgs& a,void*)const override{q::pooled_source::protocol(a)={};q::pooled_source::stage(a)=q::pooled_source::ResponseStage::Idle;q::pooled_source::progress(a)={};q::pooled_source::frontier(a)={};}
    void admitNative(const q::pooled_source::DeviceArgs& a,uint64_t generation,uint64_t ticket,void*)const override{if(!q::pooled_source::admitProtocol(a,generation,ticket))throw std::logic_error("Recorded source I/O admission rejected");if(failAdmission)throw std::runtime_error("Injected source admission publication failure");}
    void launch(q::PooledSourceAction action,const q::pooled_source::DeviceArgs&,q::PooledSourceQuery,void*)const override{++launches;if(!isGraphCaptureActive())throw std::logic_error("Unit source attempted native launch");if(failRecordedAction && *failRecordedAction==action)throw std::runtime_error("Injected exact native recording failure");}
    void closeNative(const q::pooled_source::DeviceArgs&,const int*,void*)const override{++closes;throw std::logic_error("Unit source attempted native close");}
private:
    q::PooledImmutableSourceGeometry geometry_;q::ImmutableSubordinateLookupBinding source_;
};
/** @brief Small authored native file; no real model/corpus payload is read. */
struct SourceGraphFile {
    std::string path;
    SourceGraphFile(){char name[]="/tmp/qfn-source-fill-replay-XXXXXX";const int fd=mkstemp(name);if(fd<0)throw std::runtime_error("Source fixture file creation failed");close(fd);path=name;std::vector<char> bytes(37+1000*90);for(size_t i=37;i<bytes.size();++i)bytes[i]=static_cast<char>((i-37)%251);std::ofstream f(path,std::ios::binary);f.write(bytes.data(),bytes.size());if(!f)throw std::runtime_error("Source fixture file write failed");}
    ~SourceGraphFile(){if(!path.empty() && std::remove(path.c_str()))std::terminate();}
    ImmutableRowTableSource descriptor()const{return {.path=path,.tensor_name="per_layer_token_embd.weight",.file_offset=37,.rows=1000,.row_elements=160,.format=GGUFTensorType::IQ4_NL};}
};
/** @brief Actual original root/provider/worker/channel factories over exact admitted declarations, with no native execution. */
/** @brief Explicit pure-declaration pass versus original fixture materialization. */
enum class SourceGraphRecipePhase { Materialize, DeclareOnly };
struct SourceGraphCase {
    OwnerChainBackendScope backends;SourceGraphFile file;uint32_t localCoordinate;
    std::shared_ptr<const q::PooledImmutableLocalOwnerBinding> localBinding;
    std::optional<PooledImmutableMixedVictimCompletion> mixedVictim;
    std::vector<q::PooledImmutableSourceParticipant> sourceMembers;
    ImmutableRowTableLayoutGeometry layout;
    q::PooledImmutablePhysicalOwnerGeometry physical;
    q::PooledImmutableCoordinatorGeometry policy;
    CpuAdaptiveImmutableRowCacheGeometry cpuGeometry;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    void* originalSourceStream=nullptr; ///< Exact retained producer setup/capture stream.
    std::vector<void*> originalOwnerStreams;
    llaminar2::test::RetainedHostServiceFixture io=llaminar2::test::RetainedHostServiceFixture::recorded(RetainedHostServiceRole::ImmutableLookupIO,{0,DeviceId::cuda(0)});
    PhysicalMemoryPlan declaredProgram; ///< Pure original fixture recipe, available before any root materialization.
    std::shared_ptr<HostImmutableRowProvider> provider;
    std::shared_ptr<q::ImmutableLookupService> service;
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
    std::vector<std::shared_ptr<q::PooledImmutableOwnerPreparedStorage>> owners;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinEdge>> forwards;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> returns;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedFreeWave>> frees;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedVictimWave>> victims;
    std::shared_ptr<q::ImmutableLookupExecutionBoundary> boundary;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedSourceWave>> sources;
    PhysicalMemoryAllocationLease controlLease,readOutputLease,readIdsLease,queryTokenLease,queryPriorLease,queryLiveLease;
    std::shared_ptr<TensorBase> ids,controller,readOutput;
    std::shared_ptr<TensorBase> queryTokens,queryPrior,queryLive; ///< Optional original PLE query/history roots, never mirrored inference state.
    std::shared_ptr<OwnerChainCode> coordCode=std::make_shared<OwnerChainCode>(DeviceType::CUDA),ownerCode=std::make_shared<OwnerChainCode>(DeviceType::ROCm);
    BufferArena arena;ComputeGraph graph;std::optional<DeviceGenerationControlBinding> control;
    std::vector<std::unique_ptr<BufferArena>> ownerArenas;
    std::vector<std::unique_ptr<ComputeGraph>> ownerGraphs;
    std::vector<PooledPinParticipantNodes> ownerVictims;
    PooledPinParticipantNodes complete;
    /** @brief Original source/CPU cache/coordinator/owners declared once, preserving reverse physical order. */
    static ImmutableRowTableLayoutGeometry declare(const ImmutableRowTableSource& source,const std::vector<uint32_t>& capacities,bool reverse,uint32_t local=q::adaptive_lookup::None,DeviceType sourceType=DeviceType::CUDA){
        const PhysicalMemoryAllocatorIdentity host{0,DeviceId::cpu()},coord{0,DeviceId{sourceType,0}};std::vector<ImmutableRowTableLayoutOwner> owners;
        for(size_t i=0;i<capacities.size();++i)owners.push_back({{{0,i==local?DeviceId{sourceType,0}:DeviceId{sourceType==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA,static_cast<int>(reverse?capacities.size()-1-i:i)}},ImmutableRowStorage::VRAM},0,capacities[i],{}});
        owners.push_back({{host,ImmutableRowStorage::RAM},0,2,{}});const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};const bool positive=std::any_of(capacities.begin(),capacities.end(),[](auto c){return c!=0;});
        return ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,{coord},{},{{host,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},{},positive?std::optional(coord):std::nullopt,16);
    }
    /** @brief Original caller order, never sorted by device or cache capacity. */
    static q::PooledImmutablePhysicalOwnerGeometry project(const ImmutableRowTableLayoutGeometry& layout,const std::vector<uint32_t>& capacities,bool reverse,uint32_t local=q::adaptive_lookup::None,DeviceType sourceType=DeviceType::CUDA){std::vector<PhysicalMemoryAllocatorIdentity> order;for(size_t i=0;i<capacities.size();++i)order.push_back({0,i==local?DeviceId{sourceType,0}:DeviceId{sourceType==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA,static_cast<int>(reverse?capacities.size()-1-i:i)}});return q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,{0,DeviceId{sourceType,0}});}
    explicit SourceGraphCase(const std::vector<uint32_t>& capacities,bool reverse=false,uint32_t local=q::adaptive_lookup::None,bool prepareNativeIds=false,DeviceType sourceType=DeviceType::CUDA,std::shared_ptr<OwnerChainBackendScope> sharedBackends={},std::shared_ptr<PhysicalMemoryAuthority>* originalMemoryObservation=nullptr,std::shared_ptr<PhysicalMemoryAuthority> originalAuthority={},SourceGraphRecipePhase phase=SourceGraphRecipePhase::Materialize,std::uint64_t ioOrdinal=0,std::uint32_t originalIds=4,std::shared_ptr<NativeContextPreparation> originalSource={}):backends(std::move(sharedBackends)),localCoordinate(local),layout(declare(file.descriptor(),capacities,reverse,local,sourceType)),physical(project(layout,capacities,reverse,local,sourceType)),policy(q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},std::max(16u,originalIds))),cpuGeometry(CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,{0,DeviceId::cpu()},{0,DeviceId{sourceType,0}})){
        if(!originalIds || originalIds>64 || (originalIds!=4 && originalIds%16!=0))throw std::invalid_argument("Original graph fixture query exceeds its admitted 64-ID source envelope");
        coordCode=std::make_shared<OwnerChainCode>(sourceType);ownerCode=std::make_shared<OwnerChainCode>(sourceType==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA);
        const auto descriptor=file.descriptor();const PhysicalMemoryAllocatorIdentity coord{0,DeviceId{sourceType,0}};const auto resource=[&](PhysicalMemoryAllocatorIdentity id){return PhysicalMemoryResource{id.world_rank,id.device,1ULL<<30,1ULL<<30};};PhysicalMemoryPlanBuilder bom;const auto add=[&](const PhysicalMemoryPlan& p){for(const auto& r:p.resources())bom.add(r);};
        io=llaminar2::test::RetainedHostServiceFixture::recorded(RetainedHostServiceRole::ImmutableLookupIO,coord,ioOrdinal);add(io.bom(resource({0,DeviceId::cpu()})));
        add(cpuGeometry.bom(resource({0,DeviceId::cpu()})));add(policy.policyBOM(resource(coord)));add(policy.mandatoryBOM(resource(coord)));add(q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,resource(coord)));add(q::PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,descriptor,resource(coord)));
        const auto channel=q::ImmutableLookupLayout::createSubordinate(physical.maximumUniqueRows(),descriptor.rowBytes());bom.add(resource({0,DeviceId::cpu()}),PhysicalMemoryOwner::ImmutableLookupStaging,q::ImmutableLookupService::hostBytes(channel));
        const auto declarationCoord=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,701,17);
        std::vector<q::PooledImmutableOwnerNativePlan> initialOwners;
        for(size_t i=0;i<capacities.size();++i)initialOwners.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,i,701,17));
        const auto sourceDeclaration=q::PooledImmutableSourceGeometry::fromPlans(initialOwners,declarationCoord,descriptor,1);
        llaminar2::test::PooledImmutableFactoryReplayAccess::addOrdering(bom,declarationCoord.orderingProgram(),resource({0,DeviceId::cpu()}),resource(coord));
        llaminar2::test::PooledImmutableFactoryReplayAccess::addOrdering(bom,sourceDeclaration.orderingProgram(),resource({0,DeviceId::cpu()}),resource(coord));
        llaminar2::test::PooledImmutableFactoryReplayAccess::addOrdering(bom,NativeOrderingEventProgram::lookupSubmissionRetirement(),resource({0,DeviceId::cpu()}),resource(coord));
        for(size_t i=0;i<capacities.size();++i){const auto allocator=physical.owners()[i].allocator;add(physical.cacheBOM(i,resource(allocator)));add(physical.mandatoryBOM(i,resource(allocator)));const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,i,701,17);add(owner.mandatoryBOM(resource(allocator)));llaminar2::test::PooledImmutableFactoryReplayAccess::addOrdering(bom,owner.orderingProgram(),resource({0,DeviceId::cpu()}),resource(allocator));if(i==local)continue;const auto edge=q::PooledImmutablePinEdge::fromPlans(declarationCoord,owner);for(const auto c:{edge.channelMemory(),edge.returnChannelMemory()}){bom.add(resource({0,DeviceId::cpu()}),PhysicalMemoryOwner::ActivationTransportStaging,c.mapped_host_bytes);bom.add(resource(coord),PhysicalMemoryOwner::ActivationTransportStaging,c.cursor_bytes_per_device);bom.add(resource(allocator),PhysicalMemoryOwner::ActivationTransportStaging,c.cursor_bytes_per_device);}}
        const CPUTensorStorageDeclaration idDeclaration(1,originalIds*2,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu()),controllerDeclaration(1,sampling_math::kDeviceGenerationControlCount,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu());
        if(prepareNativeIds)bom.add(resource(coord),PhysicalMemoryOwner::ImmutableLookupStaging,originalIds*sizeof(std::uint64_t));
        const CPUTensorStorageDeclaration tokenDeclaration(1,std::max(1u,originalIds/16),BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu()),priorDeclaration(1,2,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu()),liveDeclaration(1,1,BufferTensorType::INT32,PhysicalMemoryOwner::ImmutableLookupStaging,DeviceId::cpu());
        if(originalIds>4){bom.add(resource({0,DeviceId::cpu()}),PhysicalMemoryOwner::ImmutableLookupStaging,tokenDeclaration.physicalBytes()+priorDeclaration.physicalBytes()+liveDeclaration.physicalBytes());
            bom.add(resource(coord),PhysicalMemoryOwner::ImmutableLookupStaging,(std::max(1u,originalIds/16)+3)*sizeof(std::int32_t));}
        bom.add(resource({0,DeviceId::cpu()}),PhysicalMemoryOwner::ImmutableLookupStaging,idDeclaration.physicalBytes());bom.add(resource({0,DeviceId::cpu()}),PhysicalMemoryOwner::ActivationArena,controllerDeclaration.physicalBytes());bom.add(resource(coord),PhysicalMemoryOwner::ActivationArena,sampling_math::kDeviceGenerationControlCount*sizeof(int32_t));
        const CPUTensorStorageDeclaration readOutputDeclaration(std::max(16u,originalIds),160,BufferTensorType::FP32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu());
        bom.add(resource({0,DeviceId::cpu()}),PhysicalMemoryOwner::ActivationArena,readOutputDeclaration.physicalBytes());
        bom.add(resource(coord),PhysicalMemoryOwner::ActivationArena,std::max(16u,originalIds)*160*sizeof(float));
        declaredProgram=bom.build();if(phase==SourceGraphRecipePhase::DeclareOnly)return;
        if(!originalAuthority)throw std::invalid_argument("Source graph fixture requires its declared original preparing/admitted program set");
        originalSourceStream=GPUDeviceContextPool::instance().retainExistingPreparedContext(coord.device,originalAuthority).worker()->defaultStream();
        for(const auto& owner:physical.owners())originalOwnerStreams.push_back(GPUDeviceContextPool::instance().retainExistingPreparedContext(owner.allocator.device,originalAuthority).worker()->defaultStream());
        memory=std::move(originalAuthority);if(originalMemoryObservation)*originalMemoryObservation=memory;provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(CpuAdaptiveImmutableRowCache::create(descriptor,cpuGeometry,memory));service=q::ImmutableLookupService::create(provider,channel,std::array{DeviceId{sourceType,0}},memory,DeviceId::cpu(),io.take(memory));const auto binding=service->bindSubordinate(DeviceId{sourceType,0});const auto table=binding.deviceView().base.table,generation=binding.storageGeneration();
        const auto coordinatorPlan=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,table,generation);coordinator=llaminar2::test::PooledImmutableFactoryReplayAccess::coordinator(coordinatorPlan,memory,originalSourceStream);std::vector<q::PooledImmutableOwnerNativePlan> declarations;
        for(size_t i=0;i<capacities.size();++i){declarations.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,i,table,generation));owners.push_back(llaminar2::test::PooledImmutableFactoryReplayAccess::owner(declarations.back(),memory,originalOwnerStreams[i]));if(i==local){if(capacities.size()>1)localBinding=q::PooledImmutableLocalOwnerBinding::bind(coordinator,owners.back(),memory);forwards.push_back({});returns.push_back({});frees.push_back({});victims.push_back({});continue;}forwards.push_back(q::PooledImmutableCapturedPinEdge::bind(coordinator,owners.back(),memory,originalSourceStream,originalOwnerStreams[i]));returns.push_back(q::PooledImmutableCapturedPinReturn::bind(forwards.back(),memory,originalOwnerStreams[i],originalSourceStream));frees.push_back(q::PooledImmutableCapturedFreeWave::bind(returns.back(),memory));victims.push_back(q::PooledImmutableCapturedVictimWave::bind(frees.back(),memory));}
        const auto packets=q::PooledImmutableSourceGeometry::fromPlans(declarations,coordinatorPlan,descriptor,1);boundary=llaminar2::test::PooledImmutableFactoryReplayAccess::boundary(service,packets,owners,coordinator,memory,originalSourceStream);for(size_t i=0;i<victims.size();++i){if(i==local){sources.push_back({});if(localBinding)sourceMembers.emplace_back(localBinding);}else{sources.push_back(q::PooledImmutableCapturedSourceWave::bind(victims[i],boundary,memory));sourceMembers.emplace_back(sources.back());}}
        if(originalSource)PooledImmutableFactoryReplayAccess::bindOriginalPreparedWorkers(*boundary->pooledStorage(),originalSource);
        readOutput=readOutputDeclaration.materialize(memory);if(!arena.registerExternalBuffer(BufferId::QFN_PLE_EMBEDDING,readOutput))throw std::logic_error("Reader original output role replaced");
        ids=idDeclaration.materialize(memory);
        if(prepareNativeIds){readIdsLease=memory->claimNewAllocation(DeviceId{sourceType,0},PhysicalMemoryOwner::ImmutableLookupStaging,originalIds*sizeof(std::uint64_t));TransferEngine::prepareDeviceInput(ids.get(),DeviceId{sourceType,0},originalSourceStream);TransferEngine::requireDeviceInput(ids.get(),DeviceId{sourceType,0},originalSourceStream);}
        if(originalIds>4){
            queryTokens=tokenDeclaration.materialize(memory);queryPrior=priorDeclaration.materialize(memory);queryLive=liveDeclaration.materialize(memory);
            queryTokenLease=memory->claimNewAllocation(DeviceId{sourceType,0},PhysicalMemoryOwner::ImmutableLookupStaging,std::max(1u,originalIds/16)*sizeof(std::int32_t));
            queryPriorLease=memory->claimNewAllocation(DeviceId{sourceType,0},PhysicalMemoryOwner::ImmutableLookupStaging,2*sizeof(std::int32_t));
            queryLiveLease=memory->claimNewAllocation(DeviceId{sourceType,0},PhysicalMemoryOwner::ImmutableLookupStaging,sizeof(std::int32_t));
            if(!arena.registerExternalBuffer(BufferId::PREFILL_CHUNK_GEOMETRY,queryLive))throw std::logic_error("Original query count replaced its arena role");
            for(const auto& root:{queryTokens,queryPrior,queryLive}){TransferEngine::prepareDeviceInput(root.get(),DeviceId{sourceType,0},originalSourceStream);TransferEngine::requireDeviceInput(root.get(),DeviceId{sourceType,0},originalSourceStream);}
        }
        controlLease=memory->claimNewAllocation(DeviceId{sourceType,0},PhysicalMemoryOwner::ActivationArena,sampling_math::kDeviceGenerationControlCount*sizeof(int32_t));controller=controllerDeclaration.materialize(memory);static_cast<int32_t*>(controller->raw_mutable_data())[sampling_math::kDeviceGenerationControlOk]=1;TransferEngine::prepareDeviceInput(controller.get(),DeviceId{sourceType,0},originalSourceStream);TransferEngine::requireDeviceInput(controller.get(),DeviceId{sourceType,0},originalSourceStream);
        bindPooledNativeRoots(arena,*coordinator);if(localBinding)bindPooledNativeRoots(arena,*localBinding->ownerStorage());else if(capacities.size()==1 && local==0)bindPooledNativeRoots(arena,*owners.front());bindPooledSourceRoots(arena,*boundary);
        if(!arena.registerExternalBuffer(BufferId::QFN_NGRAM_ROW_IDS,ids))throw std::logic_error("Source graph query root replacement");
        if(!arena.registerExternalBuffer(BufferId::MTP_GENERATION_CONTROL,controller))throw std::logic_error("Source graph control root replacement");control.emplace(DeviceGenerationControlBinding::admit(arena,DeviceId{sourceType,0},0));boundary->bindPooledGenerationControl(*control);
        q::PooledNativeOperandSpans begin{};begin[static_cast<size_t>(q::PooledNativeOperand::Ids)]=q::PooledNativeTensorSpan{ids,BufferId::QFN_NGRAM_ROW_IDS,0,originalIds*sizeof(std::uint64_t)};begin[static_cast<size_t>(q::PooledNativeOperand::Admission)]=q::PooledNativeTensorSpan{boundary->pooledStorage()->roots().native,BufferId::QFN_POOLED_SOURCE_NATIVE,packets.parameters().offsets.protocol+offsetof(q::ImmutableLookupDeviceState,admissionGeneration),8};
        // A sole local owner uses the production local transaction. It has no
        // distributed frontier, self channel, mixed binding or remote participant.
        if(capacities.size()==1 && local==0){
            const auto localGraph=appendPooledImmutableLocalLookupGraph(graph,arena,boundary,coordCode,
                n::OriginalRows{DeviceRowRange::fullyActive(originalIds),n::OriginalIdWidth::UInt64},begin,*control,
                {readOutput,BufferId::QFN_PLE_EMBEDDING,0,originalIds*160*sizeof(float)},"local_lookup");
            complete={localGraph.first,localGraph.last};
            return;
        }
        graph.addNode("begin",std::make_unique<PooledImmutableNativePhaseStage>(arena,coordCode,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::Begin,coordinator,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::Begin,coordinatorPlan,n::OriginalRows{DeviceRowRange::fullyActive(originalIds),n::OriginalIdWidth::UInt64},begin)}),DeviceId{sourceType,0});graph.addNode("commands",std::make_unique<PooledImmutableNativePhaseStage>(arena,coordCode,PooledImmutableNativePhaseStage::Coordinator{n::CoordinatorAction::PinCommands,coordinator,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::PinCommands,coordinatorPlan,{}, {})}),DeviceId{sourceType,0});graph.addDependency("commands","begin");std::vector<std::string> sent;
        for(size_t i=0;i<owners.size();++i){if(i==local){sent.emplace_back();ownerArenas.push_back({});ownerGraphs.push_back({});ownerVictims.push_back({});continue;}sent.push_back(appendPooledPinSender(graph,arena,forwards[i],"begin","commands","pin_send"+std::to_string(i)).last);ownerArenas.push_back(std::make_unique<BufferArena>());ownerGraphs.push_back(std::make_unique<ComputeGraph>());auto& a=*ownerArenas.back();auto& g=*ownerGraphs.back();bindPooledNativeRoots(a,*owners[i]);const auto incoming=appendPooledPinReceiver(g,a,forwards[i],ownerCode,"pin");const auto pin=appendPooledPinOwnerReturn(g,a,returns[i],ownerCode,incoming.last,"pin_return");const auto free=appendPooledFreeOwnerWave(g,a,frees[i],ownerCode,pin.last,"free");ownerVictims.push_back(appendPooledVictimOwnerWave(g,a,victims[i],ownerCode,free.last,"victim"));}
        if(localBinding){std::vector<q::PooledImmutablePinParticipant> pp;std::vector<q::PooledImmutableFreeParticipant> fp;std::vector<q::PooledImmutableVictimParticipant> vp;for(size_t i=0;i<owners.size();++i){if(i==local){pp.emplace_back(localBinding);fp.emplace_back(localBinding);vp.emplace_back(localBinding);}else{pp.emplace_back(returns[i]);fp.emplace_back(frees[i]);vp.emplace_back(victims[i]);}}const auto pin=appendPooledPinCoordinatorMembers(graph,arena,pp,coordCode,sent,"begin","commands","pin_return");const auto free=appendPooledFreeCoordinatorMembers(graph,arena,fp,coordCode,pin.coordinator.last,"free",&pin);mixedVictim=appendPooledVictimCoordinatorMembers(graph,arena,vp,coordCode,free.coordinator.last,"victim",&free);complete=mixedVictim->coordinator;}
        else{const auto pin=appendPooledPinCoordinatorReturn(graph,arena,returns,coordCode,sent,"commands","pin_return");const auto free=appendPooledFreeCoordinatorWave(graph,arena,frees,coordCode,pin.last,"free");complete=appendPooledVictimCoordinatorWave(graph,arena,victims,coordCode,free.last,"victim");}
    }
};

/**
 * @brief Decode actual native phase producers over the original factory-owned API-double roots.
 * This fixture invokes the actual CPU I/O worker. CPU memcpy represents explicit
 * acquired transport/native copy observations; it does not qualify GPU storage,
 * capture, transfer traffic or the still-missing publication/read transaction.
 */
struct SourceWorkerTrace {
    SourceGraphCase& storage;
    n::CoordinatorArgs coordinator;
    std::vector<n::OwnerArgs> owners;
    q::pooled_source::DeviceArgs source;
    int healthy=1,error=0;
    static std::byte* data(const std::shared_ptr<TensorBase>& tensor){return tensor?static_cast<std::byte*>(tensor->gpu_data_ptr()):nullptr;}
    static void require(bool ok,const std::source_location& at=std::source_location::current()){if(!ok)throw std::logic_error("Actual decoded source/owner phase rejected at line "+std::to_string(at.line()));}
    explicit SourceWorkerTrace(SourceGraphCase& c):storage(c),coordinator{c.coordinator->plan().parameters(),nullptr,nullptr,nullptr,{}},source{c.boundary->pooledKernels().geometry().parameters(),{c.boundary->pooledKernels().geometry().parameters().owner,nullptr,nullptr,nullptr,{}},coordinator,nullptr,nullptr,c.boundary->pooledKernels().source().deviceView(),nullptr,nullptr,nullptr}{
        const auto cr=c.coordinator->roots();coordinator={c.coordinator->plan().parameters(),data(cr.cache),data(cr.frame),data(cr.native),{}};
        n::seedCoordinator({coordinator.plan,coordinator.cache,coordinator.frame,coordinator.native});
        std::copy(c.coordinator->plan().owners().begin(),c.coordinator->plan().owners().end(),n::at<n::p::CoordinatorOwnerGeometry>(coordinator.frame,coordinator.plan.frame.owners));
        require(n::p::CoordinatorOperations(n::coordinatorView(coordinator)).initialize());
        for(const auto& original:c.owners){const auto root=original->roots();owners.push_back({original->plan().parameters(),data(root.cache),data(root.frame),data(root.native),{}});auto& a=owners.back();n::seedOwner<q::adaptive_lookup::Slot32>(a);auto operation=n::p::OwnerOperations<q::adaptive_lookup::Slot32>(n::ownerView<q::adaptive_lookup::Slot32>(a));for(uint32_t slot=0;slot<a.plan.owner.capacity();++slot)operation.initializeSlot(slot);require(n::sealOwner<q::adaptive_lookup::Slot32>(a));}
        const auto roots=c.boundary->pooledStorage()->roots();const auto& code=c.boundary->pooledKernels();
        source={code.geometry().parameters(),{code.geometry().parameters().owner,nullptr,nullptr,nullptr,{}},coordinator,data(roots.frame),data(roots.native),code.source().deviceView(),&healthy,&error,nullptr};
        if(c.localBinding)source.owner=owners[c.localCoordinate];
    }
    template<n::OwnerAction Action>static void run(n::OwnerArgs& a){if(!n::owner<Action,q::adaptive_lookup::Slot32>(a))throw std::logic_error("Actual owner action "+std::to_string(static_cast<unsigned>(Action))+" coordinate "+std::to_string(a.plan.owner.coordinate())+" rejected status "+std::to_string(static_cast<unsigned>(n::ownerView<q::adaptive_lookup::Slot32>(a).frame->status)));}
    template<n::CoordinatorAction Action>void run(){if constexpr(Action==n::CoordinatorAction::PlanFree){auto v=n::coordinatorView(coordinator);constexpr auto width=n::p::OriginalObservationRuns::PartitionRows;for(uint32_t first=0;first<v.frame->original_ids;first+=width){const auto end=std::min(v.frame->original_ids,first+width);uint32_t count=0;for(auto pos=first;pos<end;){const auto unique=v.original_to_unique[pos];auto next=pos+1;while(next<end && v.original_to_unique[next]==unique)++next;v.sort_a[first+count++]={v.unique[unique].row,pos,next-pos};pos=next;}v.sort_b[first/width].unique=count;}}if(!n::coordinator<Action>(coordinator))throw std::logic_error("Actual coordinator action "+std::to_string(static_cast<unsigned>(Action))+" rejected status "+std::to_string(static_cast<unsigned>(n::coordinatorView(coordinator).frame->status)));}
    template<class Header>std::vector<Header> headers(){std::vector<Header> out;for(const auto& a:owners)out.push_back(*n::at<Header>(a.native,a.plan.native.header));return out;}
    template<n::CoordinatorAction Action>void acquire(bool pins=false){std::vector<n::p::Grant> records;for(auto& a:owners){const auto count=n::progress(a).reply_bytes/sizeof(n::p::Grant);const auto* from=pins?n::at<n::p::Grant>(a.frame,a.plan.frame.initial_pins):n::grants(a)+n::allocationFrame(a).wave_offset;records.insert(records.end(),from,from+count);}const uint64_t bytes=records.size()*sizeof(n::p::Grant);coordinator.input.records=records.data();coordinator.input.record_bytes=coordinator.plan.native.reply_capacity?&bytes:nullptr;run<Action>();coordinator.input={};}
    /** @brief Original Begin/Pin/Free/Victim producers create every accepted grant; no synthetic slot/reservation records. */
    void cold(std::span<const uint64_t> ids,uint64_t generation){
        require(n::p::CoordinatorOperations(n::coordinatorView(coordinator)).begin(generation,ids.data(),ids.size()));n::invocation(coordinator);const auto* invocation=n::at<n::Invocation>(coordinator.native,coordinator.plan.native.invocation);
        for(auto& a:owners){a.input.invocation=invocation;run<n::OwnerAction::Begin>(a);}run<n::CoordinatorAction::PinCommands>();
        for(size_t i=0;i<owners.size();++i){auto& a=owners[i];const auto& slice=n::at<n::DirectedSlice>(coordinator.native,coordinator.plan.native.pin_slices)[i];a.input.records=a.plan.native.reply_capacity?coordinator.native+coordinator.plan.native.pin_commands+slice.byte_offset:nullptr;a.input.record_bytes=a.plan.native.reply_capacity?&slice.live_bytes:nullptr;run<n::OwnerAction::Pin>(a);a.input.records=nullptr;a.input.record_bytes=nullptr;}
        acquire<n::CoordinatorAction::AcquirePins>(true);for(auto& a:owners)run<n::OwnerAction::SealPins>(a);const auto pins=headers<n::p::PinHeader>();coordinator.input.frontier=pins.data();run<n::CoordinatorAction::SealPins>();coordinator.input={};for(auto& a:owners){a.input.frontier=pins.data();run<n::OwnerAction::AcquirePins>(a);a.input.frontier=nullptr;}
        for(auto& a:owners)run<n::OwnerAction::Availability>(a);const auto available=headers<n::Availability>();coordinator.input.frontier=available.data();run<n::CoordinatorAction::PlanFree>();coordinator.input={};
        for(size_t i=0;i<owners.size();++i){auto& a=owners[i];const auto slice=n::at<n::DirectedSlice>(coordinator.native,coordinator.plan.native.free_slices)[i];a.input.frontier=n::at<n::FreePlanHeader>(coordinator.native,coordinator.plan.native.free_headers)+i;a.input.records=coordinator.frame+slice.byte_offset;run<n::OwnerAction::AcceptFree>(a);a.input.records=nullptr;a.input.frontier=nullptr;run<n::OwnerAction::ExecuteFree>(a);}
        acquire<n::CoordinatorAction::AcquireFree>();for(auto& a:owners)run<n::OwnerAction::FreeReceipt>(a);const auto free=headers<n::p::FreeAllocationReceipt>();coordinator.input.frontier=free.data();run<n::CoordinatorAction::CompleteFree>();coordinator.input={};for(auto& a:owners){a.input.frontier=free.data();run<n::OwnerAction::AcquireFree>(a);a.input.frontier=nullptr;}
        run<n::CoordinatorAction::PlanVictims>();for(size_t i=0;i<owners.size();++i){auto& a=owners[i];const auto slice=n::at<n::DirectedSlice>(coordinator.native,coordinator.plan.native.victim_slices)[i];a.input.frontier=n::at<n::VictimPlanHeader>(coordinator.native,coordinator.plan.native.victim_headers)+i;a.input.records=coordinator.frame+slice.byte_offset;run<n::OwnerAction::AcceptVictim>(a);a.input.records=nullptr;a.input.frontier=nullptr;run<n::OwnerAction::ExecuteVictim>(a);}
        acquire<n::CoordinatorAction::AcquireVictims>();for(auto& a:owners)run<n::OwnerAction::VictimReceipt>(a);const auto victims=headers<n::p::VictimAllocationReceipt>();coordinator.input.frontier=victims.data();run<n::CoordinatorAction::CompleteVictims>();coordinator.input={};
    }
    /** @brief Exact real-worker publication wait; test-only bounded wait never appears in a production GPU path. */
    static void wait(uint64_t& word,uint64_t value){const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);while(std::atomic_ref(word).load(std::memory_order_acquire)!=value){if(std::chrono::steady_clock::now()>=until)throw std::runtime_error("Actual source worker did not publish expected original tuple");std::this_thread::yield();}}
    /** @brief Publish production compact request body and acquire actual worker bytes/ACKs before packing each original owner. */
    void fill(q::ImmutableLookupService::Ticket ticket){namespace s=q::pooled_source;auto& io=s::protocol(source);auto& wire=source.channel.base;storage.boundary->pooledKernels().admit(*storage.boundary->pooledStorage(),ticket.generation,ticket.invocation,storage.originalSourceStream);require(s::planRequest(source));
        coordinator.input.source=&s::frontier(source);run<n::CoordinatorAction::SealSource>();coordinator.input={};
        for(size_t i=0;i<owners.size();++i){auto& a=owners[i];a.input.frontier=&s::frontier(source);run<n::OwnerAction::SealSource>(a);a.input.frontier=nullptr;require(s::acquireSeal(source,*n::at<n::OwnerSourceSealReceipt>(a.native,a.plan.native.header),i));}
        ASSERT_GT(io.idCount,0u);const auto body=s::requestBody(source);std::memcpy(reinterpret_cast<std::byte*>(wire.request)+8,reinterpret_cast<const std::byte*>(&body)+8,q::kLookupRequestLiveBytes-8);io.issuedSequence=body.publication;s::stage(source)=s::ResponseStage::Published;std::atomic_ref(wire.request->publication).store(body.publication,std::memory_order_release);wait(wire.completion->publication,body.publication);
        const auto completion=*wire.completion;require(s::acquiredCompletion(source,completion,std::atomic_ref(wire.completion->publication).load(std::memory_order_acquire)));ASSERT_EQ(completion.status,0u);io.nativeBytes=completion.nativeBytes;io.status=completion.status;s::stage(source)=s::ResponseStage::Acquired;
        std::memcpy(source.frame+source.plan.offsets.responses,wire.nativeRows,io.nativeBytes);std::memcpy(source.frame+source.plan.offsets.invalidations,source.channel.acknowledgements,io.idCount*sizeof(ImmutableRowInvalidationAcknowledgement));require(s::validateResponse(source));
        uint64_t totalRecords=0;for(uint32_t ordinal=0;ordinal<owners.size();++ordinal){const auto i=s::packetOwner(source,ordinal);auto& a=owners[i];require(s::prepareDelivery(source,i));const auto records=*n::at<uint64_t>(source.native,source.plan.offsets.record_bytes),payload=*n::at<uint64_t>(source.native,source.plan.offsets.payload_bytes);ASSERT_EQ(records%sizeof(n::Fill),0u);ASSERT_EQ(payload,records/sizeof(n::Fill)*a.plan.row_bytes);totalRecords+=records;
            auto* to=source.frame+source.plan.offsets.payload;for(uint64_t row=0;row<records/sizeof(n::Fill);++row)std::memcpy(to+row*a.plan.row_bytes,s::deliveryRow(source,row),a.plan.row_bytes);
            if(i!=storage.localCoordinate){if(records)std::memcpy(a.native+a.plan.native.source_fills,source.frame+source.plan.offsets.fills,records);if(payload)std::memcpy(a.native+a.plan.native.source_payload,to,payload);auto* ingress=n::at<n::OwnerIngress>(a.native,a.plan.native.ingress);ingress->record_bytes=records;ingress->payload_bytes=payload;}
        }
        EXPECT_LE(totalRecords/sizeof(n::Fill),io.idCount);require(s::retirementDecision(source)==s::ResponseRetirement::Consume);std::atomic_ref(wire.acknowledgement->consumedSequence).store(io.issuedSequence,std::memory_order_release);require(s::retireConsumed(source));
        for(size_t i=0;i<owners.size();++i){auto& a=owners[i];const bool local=i==storage.localCoordinate;auto* terminal=local?&s::frontier(source):n::at<n::SourceFrontier>(a.native,a.plan.native.source_outcome);if(!local)*terminal=s::frontier(source);const auto before=*terminal;auto* ingress=local?nullptr:n::at<n::OwnerIngress>(a.native,a.plan.native.ingress);a.input.condition=n::SourceCondition::AfterSuccess;a.input.outcome=terminal;a.input.frontier=terminal;a.input.records=a.plan.native.reply_capacity?(local?source.frame+source.plan.offsets.fills:a.native+a.plan.native.source_fills):nullptr;a.input.record_bytes=a.plan.native.reply_capacity?(local?n::at<uint64_t>(source.native,source.plan.offsets.record_bytes):&ingress->record_bytes):nullptr;a.input.payload=a.plan.native.reply_capacity?(local?source.frame+source.plan.offsets.payload:a.native+a.plan.native.source_payload):nullptr;a.input.payload_bytes=a.plan.native.reply_capacity?(local?n::at<uint64_t>(source.native,source.plan.offsets.payload_bytes):&ingress->payload_bytes):nullptr;run<n::OwnerAction::ApplySourceDelivery>(a);
            const auto count=a.input.record_bytes?*a.input.record_bytes/sizeof(n::Fill):0;const auto* fills=static_cast<const n::Fill*>(a.input.records);for(uint64_t row=0;row<count;++row){auto* destination=a.cache+a.plan.cache.payload+fills[row].original.slot*a.plan.slot_stride;std::memcpy(destination,a.input.payload+row*a.plan.row_bytes,a.plan.row_bytes);for(uint64_t byte=0;byte<a.plan.row_bytes;++byte)EXPECT_EQ(destination[byte],std::byte((fills[row].original.row*a.plan.row_bytes+byte)%251));}require(n::copied<q::adaptive_lookup::Slot32>(a));run<n::OwnerAction::ValidateFills>(a);EXPECT_EQ(n::progress(a).header_bytes,sizeof(n::p::PublishHeader));EXPECT_EQ(n::at<n::p::PublishHeader>(a.native,a.plan.native.header)->status,q::adaptive_lookup::Status::Success);EXPECT_EQ(std::memcmp(terminal,&before,sizeof(before)),0);}
        std::atomic_ref(wire.acknowledgement->closedGeneration).store(ticket.generation,std::memory_order_release);EXPECT_TRUE(storage.service->await(ticket));wait(wire.acknowledgement->retiredInvocation,ticket.invocation);EXPECT_EQ(wire.acknowledgement->retiredGeneration,ticket.generation);EXPECT_EQ(wire.acknowledgement->consumedSequence,io.issuedSequence);
    }
};
/** @brief Real original source worker serves accepted grant/token rows and SourceOnly rows without a fake provider or host ledger. */


/** @brief Real worker cancellation supplies exact retirement before mixed sealed/unsealed owners drain their actual grants. */

/** @brief Concrete original owner predecessor chains reject missing/foreign/out-of-order handoff before any source nodes appear. */

/** @brief Complete all-owner source/read frontiers authenticate actual later-owner receipt ordering before source mutation. */

/** @brief Every shared packet writer follows both send readers and terminal publication follows final private-reader retirement. */

/** @brief Actual original all-remote graphs retain source/owner roots and exact count spans across C0 and reversed1/2/4/8 cohorts. */

/** @brief Actual enqueue/close guards reject all unfinished distributed transaction operations before native code runs. */

}
namespace llaminar2::qwen4exp {
/** @brief Isolated executable code-entry double; original service/root/PMA factories remain actual production bodies. */
std::unique_ptr<IPooledImmutableSourceKernels> createCUDAPooledImmutableSourceKernels(PooledImmutableSourceGeometry geometry,ImmutableSubordinateLookupBinding source){return std::make_unique<test::SourceGuardCode>(std::move(geometry),std::move(source));}
std::unique_ptr<IPooledImmutableSourceKernels> createROCmPooledImmutableSourceKernels(PooledImmutableSourceGeometry geometry,ImmutableSubordinateLookupBinding source){return std::make_unique<test::SourceGuardCode>(std::move(geometry),std::move(source));}
}
// This standalone hardware-free executable owns its entire backend-resolution
// boundary. No production target links these symbols and no vendor API loads.
namespace llaminar2 {
IBackend* getCUDABackend(){return test::chainCuda;}
IBackend* getROCmBackend(){return test::chainRocm;}
IBackend* getBackendFor(DeviceId id){return id.is_cpu()?getCPUBackend():id.is_cuda()?getCUDABackend():id.is_rocm()?getROCmBackend():nullptr;}
namespace test {
/** @brief Actual original factories retain one local owner without allocating or binding a self channel. */

/** @brief P1 does not acquire a distributed frontier/channel or enter the mixed adapter. */

/** @brief Real local metadata producers and full-frontier readers cannot be removed or replaced by extra cold edges. */

/** @brief Actual original roots remain on their sole PMA until graph/arena owners retire. */

/** @brief Neither a new authority nor reordered original membership can be substituted before any graph mutation. */

/** @brief Dropping the actual local command read cannot authorize any later shared-bank reuse. */


} // namespace test

}

namespace llaminar2::test {
/** @brief Concrete original mixed and remote producers authenticate, with remote packaging first and true local packet held through Apply. */

/** @brief Actual source completion rejects missing packet retirement, wrong original endpoint and broken local Apply before successor graph mutation. */

}

namespace llaminar2::test {
/** @return Actual nominal metadata holders over the original local relation or original source channels. */
std::vector<q::PooledImmutablePublicationParticipant> publicationMembers(SourceGraphCase& c){std::vector<q::PooledImmutablePublicationParticipant> result;for(size_t i=0;i<c.owners.size();++i){if(i==c.localCoordinate)result.emplace_back(c.localBinding);else result.emplace_back(q::PooledImmutableCapturedPublicationWave::bind(c.sources[i],c.memory));}return result;}
/** @brief Real mixed/all-remote graph stages retain exact original roots, cohorts and nominal wire extents through Ready. */

/** @brief Missing full-frontier sender or compute acknowledgment cannot pass as Ready authority, before any reader successor mutates graph. */

}

namespace llaminar2::test {
/** @brief Actual worker/provider plus original mixed roots execute local-last Fill96/count/payload through real local Apply and copy completion. */

}
namespace llaminar2::test {
/** @brief Read insertion cannot accept a local Ready send whose actual source/acquisition chain was removed or reordered. */

}
