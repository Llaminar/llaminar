/**
 * @file Test__PooledImmutableOrderingRootLifetime.cpp
 * @brief Replay the actual root allocator, lazy Tensor publication and final-alias teardown against backend API doubles.
 *
 * Only recorded certificate bounds and native API bodies are doubled. Production
 * group claims, original pool provenance, root allocation seals, canonical Tensor
 * event publication, partial-prefix cleanup and validated generation retirement
 * run unchanged. This is an isolated loader boundary, never native qualification.
 */
#include "kernels/qwen4exp/PooledImmutableNativeRootStorage.h"
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "backends/BackendManager.h"
#include "mocks/MockBackend.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <map>
#include <set>
#include <optional>
namespace llaminar2::test {
/** @brief Named private replay consumes a sealed claim in the same production materialization body. */
struct PooledImmutableNativeRootStorageReplayAccess {
    static auto create(PhysicalMemoryAllocatorIdentity allocator,std::array<std::size_t,3> bytes,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream,NativeOrderingEventProgram program,
        std::unique_ptr<NativeOrderingEventGroup> group){
        return qwen4exp::detail::PooledNativeRootStorage::createWithOrderingGroup(
            allocator,bytes,701,std::move(memory),stream,program,std::move(group));
    }
};
namespace {
/** @brief Actual host-double allocation records and exact existing event creators. */
class RootBackend final:public MockBackend {
public:
    /** @brief Install only an explicit native API double; no SDK or inventory query occurs. */
    explicit RootBackend(DeviceType type):MockBackend(type),type_(type){setDeviceCount(8);}
    /** @brief Reject leaked actual API-double allocation or event handles. */
    ~RootBackend()override{if(!allocations.empty() || !events.empty())std::terminate();}
    /** @copydoc IBackend::allocate
     * @brief Record exact aligned allocation extents and their original capture authority. */
    void* allocate(std::size_t bytes,int ordinal)override{
        void* p=nullptr;if(posix_memalign(&p,64,bytes))return nullptr;
        allocations.emplace(p,Allocation{bytes,ordinal,std::make_shared<const NativeAllocationCaptureIdentity>()});++allocationCalls;return p;
    }
    /** @copydoc IBackend::free
     * @brief Reject reclamation while the recorded original producer or native borrowers remain live. */
    void free(void* p,int ordinal)override{
        if(!p)return;auto found=allocations.find(p);
        if(found==allocations.end() || found->second.ordinal!=ordinal || pending)std::terminate();
        NativeAllocationCaptureLease::requireUnborrowed(found->second.identity);allocations.erase(found);std::free(p);++frees;
    }
    /** @copydoc IBackend::retainNativeAllocation */
    NativeAllocationCaptureLease retainNativeAllocation(const void* p,std::size_t bytes,int ordinal)override{
        const auto value=reinterpret_cast<std::uintptr_t>(p);
        for(const auto& [base,record]:allocations){const auto first=reinterpret_cast<std::uintptr_t>(base);
            if(value>=first && value-first<record.bytes)return NativeAllocationCaptureLease::acquire(record.identity,base,record.bytes,
                {type_,record.ordinal},p,bytes,{type_,ordinal});}
        throw std::logic_error("Root replay has a foreign actual allocation");
    }
    /** @copydoc IBackend::deviceRuntimeGeneration */
    std::uint64_t deviceRuntimeGeneration(int ordinal)const override{return ordinal>=0 && ordinal<8?1:0;}
    /** @copydoc IBackend::createEvent
     * @brief Exercise the recorded exact creator failure prefix. */
    void* createEvent(int)override{
        ++creates;if(observeCreator)observeCreator();
        if(creates==failCreate)return nullptr;
        auto* event=reinterpret_cast<void*>(0x1000+creates*16);events.insert(event);return event;
    }
    /** @copydoc IBackend::destroyEvent
     * @brief Observe commitment and worker retention at every original handle destroy. */
    void destroyEvent(void* event,int)override{
        if(observeDestroy)observeDestroy();if(events.erase(event)!=1)std::terminate();++destroys;
    }
    /** @copydoc IBackend::recordEvent
     * @brief Preserve exact producer stream and a one-shot publication failure. */
    bool recordEvent(void* event,int,void* stream)override{
        if(!events.contains(event) || stream!=producer)std::terminate();++records;return records!=failRecord;
    }
    /** @copydoc IBackend::waitForEvent
     * @brief Complete only the recorded exact setup producer after observing its live roots. */
    bool waitForEvent(void* event,int)override{
        if(!events.contains(event))std::terminate();if(observeWait)observeWait();++waits;pending=false;return true;
    }
    std::function<void()> observeCreator,observeDestroy,observeWait;
    void* producer=nullptr;bool pending=false;
    std::size_t creates=0,destroys=0,records=0,waits=0,allocationCalls=0,frees=0,failCreate=0,failRecord=0;
    std::set<void*> events;
private:
    struct Allocation{std::size_t bytes;int ordinal;std::shared_ptr<const NativeAllocationCaptureIdentity> identity;};
    std::map<void*,Allocation> allocations;
    DeviceType type_;
};
RootBackend* cudaBackend=nullptr;RootBackend* rocmBackend=nullptr;
/** @brief Isolated original worker plus exact same-PMA buffers and recorded event group. */
struct RootCase {
    DeviceId device;RootBackend backend;NativeOrderingEventProgram program;
    std::array<std::size_t,3> bytes;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::optional<GPUContextProvenance> original;
    std::shared_ptr<qwen4exp::detail::PooledNativeRootStorage> root;
    /** @brief Build one exact original same-PMA program before creators.
     * @param type Recorded backend family. @param p Closed original root program. */
    explicit RootCase(DeviceType type,NativeOrderingEventProgram p):device(type,0),backend(type),program(p),
        bytes{p.nonemptyRootMask()&1?64u:0u,64,64}{
        if(device.is_cuda())cudaBackend=&backend;else rocmBackend=&backend;
        testing::installHardwareFreeGPUContextFactories();auto& pool=GPUDeviceContextPool::instance();pool.getContext(device);
        original.emplace(pool.retainExistingContextProvenance(device));backend.producer=original->worker()->defaultStream();
        const std::array devices{device};const auto projection=NativeOrderingEventGroupReplayAccess::projection(devices);
        const PhysicalMemoryResource host{0,DeviceId::cpu(),1u<<20,1u<<20},gpu{0,device,1u<<20,1u<<20};
        PhysicalMemoryPlanBuilder plan(projection.bom(program,host,gpu));
        plan.add(gpu,PhysicalMemoryOwner::ImmutableTableStorage,bytes[0]);
        plan.add(gpu,PhysicalMemoryOwner::ImmutableLookupStaging,bytes[1]+bytes[2]);
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        auto group=NativeOrderingEventGroupReplayAccess::claim(memory,*original,program,projection);
        root=PooledImmutableNativeRootStorageReplayAccess::create({0,device},bytes,memory,backend.producer,program,std::move(group));
        EXPECT_EQ(backend.creates,0u);EXPECT_EQ(backend.records,0u);
        backend.observeCreator=[&]{EXPECT_EQ(liveRoots(),bytes[0]+bytes[1]+bytes[2]);EXPECT_GT(groupBytes(),0u);};
        backend.observeDestroy=[&]{EXPECT_GT(groupBytes(),0u);
            EXPECT_THROW((void)GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device),std::logic_error);};
        backend.observeWait=[&]{EXPECT_EQ(liveRoots(),bytes[0]+bytes[1]+bytes[2]);EXPECT_GT(groupBytes(),0u);};
    }
    /** @return Actual direct buffer claims from the original PMA, without another ledger. */
    std::size_t liveRoots()const{return memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation)+
        memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,PhysicalMemoryMaterializationKind::NewAllocation);}
    /** @return Actual original lazy reservations held through the certified physical lifetime. */
    std::size_t groupBytes()const{return memory->reservedBytes(device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent);}
    /** @brief Destroy roots, then replay the existing validated generation-reset discharge. */
    void reset(){
        root.reset();EXPECT_EQ(liveRoots(),0u);
        if(backend.creates)EXPECT_GT(groupBytes(),0u);else EXPECT_EQ(groupBytes(),0u);
        original.reset();
        {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);NativeOrderingEventGroupReplayAccess::validatedReset(scope);}
        EXPECT_EQ(groupBytes(),0u);EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
    }
    /** @brief Require explicit original physical retirement before withdrawing backend doubles. */
    ~RootCase(){if(root || original)std::terminate();if(device.is_cuda())cudaBackend=nullptr;else rocmBackend=nullptr;}
};
constexpr std::array programs{NativeOrderingEventProgram::pooledOwner(false),NativeOrderingEventProgram::pooledOwner(true),
    NativeOrderingEventProgram::pooledCoordinator(false),NativeOrderingEventProgram::pooledCoordinator(true),NativeOrderingEventProgram::pooledSource()};
/** @test Actual roots rejected before initialization create no events and leave no opaque reservation. */
TEST(PooledImmutableOrderingRootLifetime, UninitializedRootsCancelGroupBeforeReset){
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const auto program:programs){
        RootCase c(type,program);
        EXPECT_GT(c.liveRoots(),0u);EXPECT_GT(c.groupBytes(),0u);
        c.root.reset();
        EXPECT_EQ(c.backend.creates,0u);EXPECT_EQ(c.backend.destroys,0u);EXPECT_EQ(c.backend.waits,0u);
        EXPECT_EQ(c.liveRoots(),0u);EXPECT_EQ(c.groupBytes(),0u);
        EXPECT_EQ(c.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
        c.reset();
    }
}
/** @test Last root alias owns all real Tensor events and commitments until their exact setup producer retires. */
TEST(PooledImmutableOrderingRootLifetime, OriginalTensorEventsFollowLastAliasAndReset){
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const auto program:programs){
        RootCase c(type,program);c.root->initialize(type,c.backend.producer,[&]{c.backend.pending=true;});
        EXPECT_EQ(c.backend.creates,program.orderingEvents());EXPECT_EQ(c.backend.records,program.orderingEvents());
        auto aliases=c.root->roots();std::weak_ptr retained=c.root;
        c.original.reset(); // The actual group is now the sole external worker borrower.
        c.root.reset();
        EXPECT_FALSE(retained.expired());EXPECT_EQ(c.backend.destroys,0u);EXPECT_GT(c.liveRoots(),0u);
        aliases.cache.reset();aliases.frame.reset();EXPECT_FALSE(retained.expired());
        aliases.native.reset();EXPECT_TRUE(retained.expired());EXPECT_EQ(c.backend.destroys,program.orderingEvents());
        EXPECT_EQ(c.backend.waits,1u);EXPECT_FALSE(c.backend.pending);EXPECT_EQ(c.backend.frees,program.orderingEvents()-1);
        c.reset();
    }
}
/** @test Every creator and publication-record prefix preserves original roots/group until exact producer completion. */
TEST(PooledImmutableOrderingRootLifetime, PartialCreatorAndRecordFailuresRetainOriginalProgram){
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const auto program:programs){
        for(std::size_t fail=1;fail<=program.orderingEvents();++fail){
            RootCase c(type,program);c.backend.failCreate=fail;
            EXPECT_THROW(c.root->initialize(type,c.backend.producer,[&]{c.backend.pending=true;}),std::exception);
            EXPECT_THROW(c.root->requireInitialized(),std::logic_error);
            const auto attempts=c.backend.creates;
            EXPECT_THROW(c.root->initialize(type,c.backend.producer,[]{}),std::logic_error);
            EXPECT_EQ(c.backend.creates,attempts);c.root.reset();
            EXPECT_EQ(c.backend.destroys,fail-1);EXPECT_FALSE(c.backend.pending);c.reset();
        }
        for(std::size_t fail=1;fail<=program.orderingEvents();++fail){
            RootCase c(type,program);c.backend.failRecord=fail;
            EXPECT_THROW(c.root->initialize(type,c.backend.producer,[&]{c.backend.pending=true;}),std::exception);
            c.root.reset();EXPECT_EQ(c.backend.creates,c.backend.destroys);EXPECT_EQ(c.backend.waits,1u);EXPECT_FALSE(c.backend.pending);c.reset();
        }
        RootCase c(type,program);
        EXPECT_THROW(c.root->initialize(type,c.backend.producer,[&]{c.backend.pending=true;throw std::runtime_error("recorded partial metadata submission");}),std::runtime_error);
        c.root.reset();EXPECT_EQ(c.backend.creates,1u);EXPECT_EQ(c.backend.destroys,1u);EXPECT_EQ(c.backend.waits,1u);c.reset();
    }
}
}
}
namespace llaminar2 {
IBackend* getCUDABackend(){return test::cudaBackend;}
IBackend* getROCmBackend(){return test::rocmBackend;}
IBackend* getBackendFor(DeviceId id){return id.is_cpu()?getCPUBackend():id.is_cuda()?getCUDABackend():id.is_rocm()?getROCmBackend():nullptr;}
}
