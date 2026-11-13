/**
 * @file Test__ReadAheadNativeResourceFamily.cpp
 * @brief Device-free replay of original PMA commitments, native failure rollback and event retirement.
 *
 * Private replay certificates carry synthetic bounds solely to exercise the
 * actual resource owner without changing the public production catalog. A mock
 * worker authenticates pre-create reservations and exact event/stream identity;
 * no GPU runtime or manufactured resource qualification is involved.
 */
#include "backends/ReadAheadNativeResourceFamily.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include <cstdlib>
#include <barrier>
#include <thread>
#include <atomic>
#include "backends/ReadAheadNativeParentTerminal.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <gtest/gtest.h>
#include <future>
#include <vector>
namespace llaminar2::test {
/** @brief The only private injection path; production cannot mint a resource certificate. */
struct ReadAheadNativeResourceReplayAccess {
    /** @brief Simulate only the original publisher capability, never a native certificate. */
    static auto published(const std::shared_ptr<ReadAheadNativeResourceFamily>& family,std::shared_ptr<void> event,void* stream){
        return ReadAheadNativeResourceFamily::PublishedTerminal(family,std::move(event),stream);
    }
    /** @brief Replay actual prebirth slots with an intentional publication interleaving. */
    static NativeContextRetirementSlot pending(const GPUContextProvenance& original,const std::shared_ptr<PhysicalMemoryAuthority>& memory){
        auto slot=original.reserveNativeRetirementSlot(memory,"recorded-closed-population");
        EXPECT_FALSE(slot.tryBorrowProcessRuntime("read-ahead-one-stream-two-ordering-events"));return slot;
    }
    /** @brief Move original synthetic test reservations through the real production adoption. */
    static void adopt(NativeContextRetirementSlot& slot,const GPUContextProvenance& original,const std::shared_ptr<PhysicalMemoryAuthority>& memory){
        auto host=memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadFamily,1024);
        auto gpu=memory->reserveNewAllocations(original.device(),PhysicalMemoryOwner::NativeReadAheadFamily,2048);
        auto leaf=std::unique_ptr<ReadAheadNativeResourceObligation>(new ReadAheadNativeResourceObligation(memory,original.device(),original.generation(),
            "recorded-closed-population",std::move(host),std::optional<PhysicalMemoryOwnerReservation>(std::move(gpu)),NativePhysicalLifetime::ProcessRuntime));
        slot.adopt(std::move(leaf));
    }
    /** @brief Replay an isolated certified-owner state machine without creating a production proof. */
    static std::shared_ptr<ReadAheadNativeResourceFamily> create(std::shared_ptr<PhysicalMemoryAuthority> memory,
        std::shared_ptr<IWorkerGPUContext> worker,ReadAheadNativeResourceRequest request,
        ReadAheadNativeRetirementPolicy retirement=ReadAheadNativeRetirementPolicy::HandleDestruction){
        const auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(request.closure.device);
        if(!request.closure.contextGeneration)request.closure.contextGeneration=original.generation();
        auto certificate=ReadAheadNativeResourceCertificate(request.closure,1024,2048,"device-free replay only; not native evidence",retirement);
        return ReadAheadNativeResourceFamily::materializeCertified(std::move(memory),std::move(worker),std::move(request),std::move(certificate),original);
    }
};
/** @brief Private positive replay for the independent one-event baseline, never a production certificate. */
struct ReadAheadNativeParentTerminalReplayAccess {
    static std::shared_ptr<ReadAheadNativeParentTerminal> create(std::shared_ptr<PhysicalMemoryAuthority> memory,
        std::shared_ptr<IWorkerGPUContext> worker,ReadAheadNativeResourceClosure closure,std::size_t deviceBytes=1024,
        ReadAheadNativeRetirementPolicy retirement=ReadAheadNativeRetirementPolicy::HandleDestruction){
        const auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(closure.device);
        if(!closure.contextGeneration)closure.contextGeneration=original.generation();
        return ReadAheadNativeParentTerminal::materializeCertified(std::move(memory),std::move(worker),
            {std::move(closure),512,deviceBytes,"device-free parent baseline only; not native evidence",retirement},original);
    }
};
/** @brief Replay only the trusted reset-observer boundary; no native reset qualification. */
struct NativeContextRetirementReplayAccess {
    static void validatedReset(ExclusiveGPUDeviceGenerationRetirement& scope){scope.retireNativeObligationsAfterValidatedReset();}
};
namespace {
/** @brief Stable fake handles are ordinary test addresses, not native API objects. */
void* handle(std::uintptr_t value){return reinterpret_cast<void*>(value);}
/** @brief One actual complete source plan, shared by host and GPU commitments. */
std::shared_ptr<PhysicalMemoryAuthority> authority(std::size_t host=1024,std::size_t device=2048,DeviceId gpu=DeviceId::cuda(0),bool includeParent=false){
    PhysicalMemoryPlanBuilder plan;plan.add({0,DeviceId::cpu(),1u<<20,1u<<20},PhysicalMemoryOwner::NativeReadAheadFamily,host);
    plan.add({0,gpu,1u<<20,1u<<20},PhysicalMemoryOwner::NativeReadAheadFamily,device);
    if(includeParent){
        plan.add({0,DeviceId::cpu(),1u<<20,1u<<20},PhysicalMemoryOwner::NativeReadAheadParentTerminal,512);
        plan.add({0,gpu,1u<<20,1u<<20},PhysicalMemoryOwner::NativeReadAheadParentTerminal,1024);
    }
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
}
/** @brief Complete immutable fake closure; no field is inferred from a worker name. */
ReadAheadNativeResourceRequest request(){return {{DeviceId::cuda(0),0,"uuid","arch","driver","runtime","backend-source","resource-source",std::string(64,'b')},handle(100)};}
/** @brief Actual owner API double with pre-create accounting and fault injection. */
class Worker final:public IWorkerGPUContext {
public:
    DeviceId device=DeviceId::cuda(0);
    std::shared_ptr<PhysicalMemoryAuthority> memory;std::size_t parentDeviceBytes=1024;unsigned creates=0,destroys=0,records=0,failAt=0,throwAt=0;bool parent=false,ready=false,querySuccess=true,recordSuccess=true,aliasStream=false,aliasEvents=false;std::vector<void*> destroyed;
    int deviceOrdinal()const override{return 0;}
    std::string deviceName()const override{return "device-free";}
    bool isInitialized()const override{return true;}
    bool ownsCurrentThread()const noexcept override{return true;}
    std::future<void> submitAsync(std::function<void()> work)override{std::packaged_task<void()> task(std::move(work));auto result=task.get_future();task();return result;}
    void* defaultStream()override{return handle(100);}
    /** @brief Assert the original same-PMA commitments existed before every native creator. */
    bool create(){EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),parent?PhysicalMemoryOwner::NativeReadAheadParentTerminal:PhysicalMemoryOwner::NativeReadAheadFamily),parent?512u:1024u);EXPECT_EQ(memory->reservedBytes(device,parent?PhysicalMemoryOwner::NativeReadAheadParentTerminal:PhysicalMemoryOwner::NativeReadAheadFamily),parent?parentDeviceBytes:2048u);++creates;if(creates==throwAt)throw std::runtime_error("Actual creator exception replay");return creates!=failAt;}
    /** @brief Physical commitment must outlive each actual native cleanup call. */
    void assertCommitted(){EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),parent?PhysicalMemoryOwner::NativeReadAheadParentTerminal:PhysicalMemoryOwner::NativeReadAheadFamily),parent?512u:1024u);EXPECT_EQ(memory->reservedBytes(device,parent?PhysicalMemoryOwner::NativeReadAheadParentTerminal:PhysicalMemoryOwner::NativeReadAheadFamily),parent?parentDeviceBytes:2048u);}
    void* createStream()override{return create()?(aliasStream?handle(100):handle(200)):nullptr;}
    void destroyStream(void* stream)override{assertCommitted();destroyed.push_back(stream);++destroys;}
    void* getOrCreateAuxiliaryStream(const std::string&,bool* =nullptr)override{throw std::logic_error("Unaccounted auxiliary stream path forbidden");}
    void resetAuxiliaryStreams()override{}
    void* createEvent(GPUEventPurpose purpose=GPUEventPurpose::Ordering)override{EXPECT_EQ(purpose,GPUEventPurpose::Ordering);return create()?(aliasEvents?handle(300):handle(298+creates)):nullptr;}
    void destroyEvent(void* event)override{assertCommitted();destroyed.push_back(event);++destroys;}
    void recordEvent(void* event,void* stream)override{EXPECT_EQ(event,handle(400));EXPECT_EQ(stream,handle(100));}
    bool recordEventChecked(void* event,void* stream)override{++records;recordEvent(event,stream);return recordSuccess;}
    bool queryEventChecked(void* event,bool& complete)override{EXPECT_EQ(event,parent?handle(299):handle(400));complete=ready;return querySuccess;}
    void waitEvent(void*,void*)override{}
    void synchronizeEvent(void*)override{ADD_FAILURE()<<"Resource retirement cannot synchronize";}
    float eventElapsedTime(void*,void*)override{return 0;}
    void* blasHandle()override{return nullptr;}
    void* blasLtHandle()override{return nullptr;}
    void setCollectiveComm(void*)override{}
    void* collectiveComm()const override{return nullptr;}
    void synchronize()override{ADD_FAILURE()<<"No device synchronization";}
    void synchronizeStream(void*)override{ADD_FAILURE()<<"No stream synchronization";}
    GPUStreamExecutionState queryStreamExecutionState(void*,std::string_view)override{return GPUStreamExecutionState::Complete;}
    bool insertStreamDependency(void*,void*)override{return true;}
    std::unique_ptr<IGPUGraphCapture> createGraphCapture()override{return nullptr;}
    std::unique_ptr<IGPUGraphCapture> createGraphCapture(void*)override{return nullptr;}
};
/** @brief Dedicated replay process alone owns these fake canonical pool factories.
 * No shared native test executable is mutated and no native backend is discovered.
 */
std::shared_ptr<Worker> makeWorker(DeviceId device=DeviceId::cuda(0)){
    auto& pool=GPUDeviceContextPool::instance();
    pool.shutdown();
    auto factory=[device](int){auto result=std::make_unique<Worker>();result->device=device;return result;};
    if(device.is_cuda())pool.registerNvidiaFactory(factory,1);else pool.registerAMDFactory(factory,1);
    return std::static_pointer_cast<Worker>(pool.retainContext(device));
}
/** @brief Verify all original commitments retired without fabricating materialized allocations. */
void empty(const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId gpu=DeviceId::cuda(0)){for(const auto device:{DeviceId::cpu(),gpu}){EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeReadAheadFamily),0u);EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::NativeReadAheadFamily,PhysicalMemoryMaterializationKind::NewAllocation),0u);}}
/** @test Missing original prepared source and malformed recorded closure reject before native resource birth. */
TEST(ReadAheadNativeResourceFamily,AbsentPreparedSourceFailsBeforeNativeCreate){auto memory=authority();auto worker=makeWorker();worker->memory=memory;
    EXPECT_THROW(ReadAheadNativeResourceFamily::materialize(memory,worker,DeviceId::cuda(0),handle(100)),std::logic_error);auto invalid=request();invalid.closure.driver.clear();
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,invalid),std::invalid_argument);EXPECT_EQ(worker->creates,0u);empty(memory);}
/** @test Both canonical allocator commitments must succeed before the very first creator. */
TEST(ReadAheadNativeResourceFamily,InsufficientEitherAllocatorRollsBackBeforeCreate){for(const auto host:{false,true}){auto memory=authority(host?1023:1024,host?2048:2047);auto worker=makeWorker();worker->memory=memory;
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,request()),std::exception);EXPECT_EQ(worker->creates,0u);empty(memory);}}
/** @test Every partial native birth rolls back exactly the successfully created prefix, in reverse order. */
TEST(ReadAheadNativeResourceFamily,EveryCreatorFailureReleasesOnlyAfterNativeCleanup){for(unsigned failure=1;failure<=3;++failure){auto memory=authority();auto worker=makeWorker();worker->memory=memory;worker->failAt=failure;
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,request()),std::runtime_error);EXPECT_EQ(worker->creates,failure);EXPECT_EQ(worker->destroys,failure-1);empty(memory);}}
/** @test Aliased invalid creator results never destroy the borrowed primary or an already-owned event twice. */
TEST(ReadAheadNativeResourceFamily,CreatorAliasCannotRetireBorrowedResources){for(const bool stream:{false,true}){auto memory=authority();auto worker=makeWorker();worker->memory=memory;worker->aliasStream=stream;worker->aliasEvents=!stream;
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,request()),std::runtime_error);EXPECT_EQ(worker->destroys,stream?0u:2u);for(void* destroyed:worker->destroyed)EXPECT_NE(destroyed,handle(100));empty(memory);}}
/** @test Reset/replay retains exact resources; only the latest bound terminal completion permits retirement. */
TEST(ReadAheadNativeResourceFamily,TwentyReplaysKeepCommitmentsUntilActualTerminal){auto memory=authority();auto worker=makeWorker();worker->memory=memory;{auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request());const auto identity=family->captureIdentity();EXPECT_EQ(worker->creates,3u);EXPECT_EQ(memory->claimedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadFamily,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_THROW(family->beginSubmission(),std::logic_error);family->bindTerminal(std::shared_ptr<void>(handle(400),[](void*){}));
    EXPECT_THROW(family->bindTerminal(std::shared_ptr<void>(handle(400),[](void*){})),std::logic_error);for(unsigned replay=0;replay<20;++replay){worker->ready=false;family->beginSubmission();
    EXPECT_THROW(family->beginSubmission(),std::logic_error);
    EXPECT_THROW(family->observeTerminal(),std::logic_error);family->recordTerminal();EXPECT_FALSE(family->observeTerminal());EXPECT_EQ(worker->destroys,0u);worker->ready=true;EXPECT_TRUE(family->observeTerminal());EXPECT_EQ(family->captureIdentity(),identity);}EXPECT_EQ(worker->creates,3u);}EXPECT_EQ(worker->destroys,3u);empty(memory);}
/** @test Both native backends use their original exact allocator without creating replacement contexts. */
TEST(ReadAheadNativeResourceFamily,BackendSymmetryUsesOriginalHostAndDeviceOwners){
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}){
        auto memory=authority(1024,2048,device);auto worker=makeWorker(device);worker->memory=memory;worker->device=device;
        auto input=request();input.closure.device=device;
        {auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,input);EXPECT_EQ(family->device(),device);EXPECT_EQ(worker->creates,3u);}
        EXPECT_EQ(worker->destroys,3u);empty(memory,device);
    }
}
/** @test Exceptions from each actual creator retain the same reverse-prefix rollback as null failures. */
TEST(ReadAheadNativeResourceFamily,CreatorExceptionsUnwindSameOriginalPrefix){
    for(unsigned failure=1;failure<=3;++failure){auto memory=authority();auto worker=makeWorker();worker->memory=memory;worker->throwAt=failure;
        EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,request()),std::runtime_error);
        EXPECT_EQ(worker->destroys,failure-1);empty(memory);
    }
}
/** @test Default primary streams reject before either commitments or native resource creation. */
TEST(ReadAheadNativeResourceFamily,DefaultPrimaryStreamsCannotSpendCertificate){
    for(std::uintptr_t sentinel=0;sentinel<=2;++sentinel){auto memory=authority();auto worker=makeWorker();worker->memory=memory;auto input=request();input.primaryStream=handle(sentinel);
        EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,input),std::invalid_argument);
        EXPECT_EQ(worker->creates,0u);empty(memory);
    }
}
/** @test Native capture records borrowed resources but cannot create or retire a request lifecycle. */
TEST(ReadAheadNativeResourceFamily,CaptureCannotCreateOrPublishHostSubmissionState){
    auto memory=authority();auto worker=makeWorker();worker->memory=memory;
    auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request());
    {GraphCaptureGuard recording;
        EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,request()),std::invalid_argument);
        EXPECT_THROW(family->beginSubmission(),std::logic_error);
        EXPECT_THROW(family->observeTerminal(),std::logic_error);
    }
    EXPECT_EQ(worker->creates,3u);EXPECT_FALSE(family->hasOutstandingSubmission());
}
/** @test Serialized variants retain the same terminal owner without changing embedded capture identity. */
TEST(ReadAheadNativeResourceFamily,OriginalTerminalOwnerOutlivesFamilyAndCompatibleBindings){
    auto memory=authority();auto worker=makeWorker();worker->memory=memory;
    auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request());const auto identity=family->captureIdentity();
    auto event=std::shared_ptr<void>(handle(400),[](void*){});std::weak_ptr<void> weak=event;
    family->bindTerminal(event);family->bindTerminal(event);event.reset();EXPECT_FALSE(weak.expired());
    EXPECT_EQ(family->captureIdentity(),identity);family->beginSubmission();EXPECT_TRUE(family->hasOutstandingSubmission());
    family->recordTerminal();worker->ready=true;EXPECT_TRUE(family->observeTerminal());EXPECT_FALSE(family->hasOutstandingSubmission());
    family.reset();EXPECT_TRUE(weak.expired());empty(memory);
}
/** @test Same ordinals, forged shared ownership and stale generation cannot create native handles. */
TEST(ReadAheadNativeResourceFamily,ExactPoolProvenanceRejectsBackendControlBlockAndStaleGeneration){
    auto memory=authority();auto worker=makeWorker();worker->memory=memory;
    auto& pool=GPUDeviceContextPool::instance();
    pool.registerAMDFactory([](int){auto result=std::make_unique<Worker>();result->device=DeviceId::rocm(0);return result;},1);
    auto remote=pool.retainContext(DeviceId::rocm(0));
    auto wrong=request();wrong.closure.device=DeviceId::rocm(0);
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,wrong),std::logic_error);
    auto forged=std::shared_ptr<IWorkerGPUContext>(worker.get(),[](auto*){});
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,forged,request()),std::logic_error);
    auto stale=request();stale.closure.contextGeneration=pool.retainExistingContextProvenance(DeviceId::cuda(0)).generation()+1;
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,stale),std::invalid_argument);
    EXPECT_EQ(worker->creates,0u);empty(memory);
}
/** @test Existing terminal publisher remains the sole event producer across stale-proof replay. */
TEST(ReadAheadNativeResourceFamily,OriginalPublicationProofNeverDuplicatesEventProduction){
    auto memory=authority();auto worker=makeWorker();worker->memory=memory;
    auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request());
    auto event=std::shared_ptr<void>(handle(400),[](void*){});family->bindTerminal(event);
    family->beginSubmission();
    EXPECT_TRUE(worker->recordEventChecked(event.get(),handle(100)));
    auto proof=ReadAheadNativeResourceReplayAccess::published(family,event,handle(100));
    family->acceptTerminalPublication(proof);
    EXPECT_EQ(worker->records,1u);
    EXPECT_THROW(family->acceptTerminalPublication(proof),std::logic_error);
    worker->ready=true;EXPECT_TRUE(family->observeTerminal());
    family->beginSubmission();
    EXPECT_THROW(family->acceptTerminalPublication(proof),std::logic_error);
    auto forgedFamily=std::shared_ptr<ReadAheadNativeResourceFamily>(family.get(),[](auto*){});
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::published(forgedFamily,event,handle(100)),std::logic_error);
    auto forged=std::shared_ptr<void>(event.get(),[](void*){});
    auto bad=ReadAheadNativeResourceReplayAccess::published(family,forged,handle(100));
    EXPECT_THROW(family->acceptTerminalPublication(bad),std::logic_error);
    auto wrong=ReadAheadNativeResourceReplayAccess::published(family,event,handle(200));
    EXPECT_THROW(family->acceptTerminalPublication(wrong),std::logic_error);
    EXPECT_TRUE(worker->recordEventChecked(event.get(),handle(100)));
    auto current=ReadAheadNativeResourceReplayAccess::published(family,event,handle(100));
    family->acceptTerminalPublication(current);EXPECT_TRUE(family->observeTerminal());
    EXPECT_EQ(worker->records,2u);family.reset();empty(memory);
}
/** @test Active native readers cannot unwind original commitments or captured handles. */
TEST(ReadAheadNativeResourceFamily,OutstandingSubmissionDestructionIsFatal){GTEST_FLAG_SET(death_test_style,"threadsafe");EXPECT_DEATH({auto memory=authority();auto worker=makeWorker();worker->memory=memory;auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request());family->bindTerminal(std::shared_ptr<void>(handle(400),[](void*){}));family->beginSubmission();family.reset();},"");}
/** @brief Independent baseline event admission; neither graph nor family bytes fund this owner. */
std::shared_ptr<PhysicalMemoryAuthority> parentAuthority(std::size_t host=512,std::size_t gpu=1024){
    PhysicalMemoryPlanBuilder plan;plan.add({0,DeviceId::cpu(),1u<<20,1u<<20},PhysicalMemoryOwner::NativeReadAheadParentTerminal,host);
    plan.add({0,DeviceId::cuda(0),1u<<20,1u<<20},PhysicalMemoryOwner::NativeReadAheadParentTerminal,gpu);
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
}
/** @test Unqualified baseline cannot create an event or silently spend another BOM owner. */
TEST(ReadAheadNativeParentTerminal,MissingOriginalPreparedSourceRejectsBeforeCreate){auto memory=parentAuthority();auto worker=makeWorker();worker->memory=memory;worker->parent=true;
    EXPECT_THROW(ReadAheadNativeParentTerminal::materialize(memory,worker,DeviceId::cuda(0)),std::logic_error);
    EXPECT_EQ(worker->creates,0u);EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);
}
/** @test Both independent allocator bounds and the actual creator are failure-atomic. */
TEST(ReadAheadNativeParentTerminal,AllocatorAndCreatorFailureAreAtomic){for(unsigned fault=0;fault<3;++fault){auto memory=parentAuthority(fault==0?511:512,fault==1?1023:1024);auto worker=makeWorker();worker->memory=memory;worker->parent=true;if(fault==2)worker->failAt=1;
    EXPECT_THROW(ReadAheadNativeParentTerminalReplayAccess::create(memory,worker,request().closure),std::exception);
    EXPECT_EQ(worker->creates,fault==2?1u:0u);for(const auto device:{DeviceId::cpu(),DeviceId::cuda(0)})EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);}
}
/** @test A borrowed event strongly retains its original owner and separate commitments through actual completion. */
TEST(ReadAheadNativeParentTerminal,StrongAliasRetainsOriginalCommitment){auto memory=parentAuthority();auto worker=makeWorker();worker->memory=memory;worker->parent=true;auto owner=ReadAheadNativeParentTerminalReplayAccess::create(memory,worker,request().closure);auto event=owner->eventOwner();EXPECT_EQ(event.get(),handle(299));owner.reset();EXPECT_EQ(worker->destroys,0u);EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadParentTerminal),1024u);worker->ready=true;event.reset();EXPECT_EQ(worker->destroys,1u);EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);}
/** @test Original canonical tokens survive every successful and partial handle teardown until reset. */
TEST(ReadAheadNativeResourceFamily,ContextResetRetainsOriginalTokensAcrossSuccessAndCreatorFailure){
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})for(unsigned failure=0;failure<=3;++failure){
        auto memory=authority(1024,2048,device);auto worker=makeWorker(device);worker->memory=memory;worker->failAt=failure;
        auto input=request();input.closure.device=device;
        if(failure){EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,input,
            ReadAheadNativeRetirementPolicy::OriginalContextReset),std::runtime_error);}
        else {auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,input,
            ReadAheadNativeRetirementPolicy::OriginalContextReset);family.reset();}
        EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadFamily),1024u);
        EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
        EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,input,
            ReadAheadNativeRetirementPolicy::OriginalContextReset),std::logic_error);
        worker.reset();auto& pool=GPUDeviceContextPool::instance();
        EXPECT_THROW(pool.shutdown(),std::logic_error);
        EXPECT_THROW(pool.retireExclusiveGeneration(device),std::logic_error);
        auto scope=pool.beginExclusiveGenerationRetirement(device);
        EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
        NativeContextRetirementReplayAccess::validatedReset(scope);empty(memory,device);
    }
}
/** @test One-event fixture's explicit zero device bound creates no zero-byte reservation or false claim. */
TEST(ReadAheadNativeParentTerminal,ZeroDeviceFixtureObligationRetiresAtOriginalReset){
    auto memory=parentAuthority(512,0);auto worker=makeWorker();worker->memory=memory;worker->parent=true;worker->parentDeviceBytes=0;
    auto owner=ReadAheadNativeParentTerminalReplayAccess::create(memory,worker,request().closure,0,
        ReadAheadNativeRetirementPolicy::OriginalContextReset);
    worker->ready=true;owner.reset();
    EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadParentTerminal),512u);
    EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);
    worker.reset();auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));
    NativeContextRetirementReplayAccess::validatedReset(scope);
    EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);
}
/** @test An equal-looking second authority cannot adopt into the original generation. */
TEST(ReadAheadNativeResourceFamily,ContextResetRejectsForeignAuthorityBeforeCreators){
    auto first=authority();auto worker=makeWorker();worker->memory=first;
    auto family=ReadAheadNativeResourceReplayAccess::create(first,worker,request(),ReadAheadNativeRetirementPolicy::OriginalContextReset);family.reset();
    auto second=authority();worker->memory=second;
    EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(second,worker,request(),ReadAheadNativeRetirementPolicy::OriginalContextReset),std::logic_error);
    EXPECT_EQ(worker->creates,3u);empty(second);worker->memory=first;worker.reset();
    auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));
    NativeContextRetirementReplayAccess::validatedReset(scope);empty(first);
}
/** @test A worker-only scope cannot silently drop retained native tokens without reset. */
TEST(ReadAheadNativeResourceFamily,MissingOriginalResetIsFatal){
    GTEST_FLAG_SET(death_test_style,"threadsafe");EXPECT_DEATH({
        auto memory=authority();auto worker=makeWorker();worker->memory=memory;
        auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request(),ReadAheadNativeRetirementPolicy::OriginalContextReset);family.reset();worker.reset();
        {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));}
    },"");
}
/** @test Actual pending parent use cannot be destroyed or converted into a released physical obligation. */
TEST(ReadAheadNativeParentTerminal,PendingDestructionIsFatal){GTEST_FLAG_SET(death_test_style,"threadsafe");EXPECT_DEATH({auto memory=parentAuthority();auto worker=makeWorker();worker->memory=memory;worker->parent=true;auto owner=ReadAheadNativeParentTerminalReplayAccess::create(memory,worker,request().closure);owner.reset();},"");}
/** @test Original process tokens survive20actual owner resets and lexical process cleanup. */
TEST(ReadAheadNativeProcessRuntime,SameAuthorityTwentyResetUsesNeverRechargeOrRelease){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}){
        ASSERT_EXIT(([&]{
            auto memory=authority(1024,2048,device);
            for(unsigned replay=0;replay<20;++replay){
                auto worker=makeWorker(device);worker->memory=memory;auto input=request();input.closure.device=device;
                auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,input,ReadAheadNativeRetirementPolicy::OriginalProcessRuntime);
                EXPECT_EQ(worker->creates,3u);family.reset();
                EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
                worker.reset();{auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);NativeContextRetirementReplayAccess::validatedReset(scope);}
                EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadFamily),1024u);
            }
            std::weak_ptr<PhysicalMemoryAuthority> original=memory;memory.reset();EXPECT_FALSE(original.expired());
            GPUDeviceProcessLifetime::retireActive();EXPECT_FALSE(original.expired());
            EXPECT_THROW(GPUDeviceProcessLifetime::requireActive(),std::logic_error);
            std::_Exit(::testing::Test::HasFailure()?2:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}
/** @test Borrowing authenticates source, original authority/controlblock and exclusive generation use. */
TEST(ReadAheadNativeProcessRuntime,RejectForeignSourceAuthorityAndOverlappingUseBeforeCreators){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        auto memory=authority();auto worker=makeWorker();worker->memory=memory;
        auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request(),ReadAheadNativeRetirementPolicy::OriginalProcessRuntime);
        EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,request(),ReadAheadNativeRetirementPolicy::OriginalProcessRuntime),std::logic_error);
        family.reset();worker.reset();{auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));NativeContextRetirementReplayAccess::validatedReset(scope);}
        worker=makeWorker();worker->memory=memory;auto foreign=request();foreign.closure.resourceSource="foreign-source";
        EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,foreign,ReadAheadNativeRetirementPolicy::OriginalProcessRuntime),std::logic_error);
        auto second=authority();EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(second,worker,request(),ReadAheadNativeRetirementPolicy::OriginalProcessRuntime),std::logic_error);
        auto alias=std::shared_ptr<PhysicalMemoryAuthority>(memory.get(),[](auto*){});
        EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(alias,worker,request(),ReadAheadNativeRetirementPolicy::OriginalProcessRuntime),std::logic_error);
        EXPECT_EQ(worker->creates,0u);empty(second);
        family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request(),ReadAheadNativeRetirementPolicy::OriginalProcessRuntime);family.reset();worker.reset();
        {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));NativeContextRetirementReplayAccess::validatedReset(scope);}
        GPUDeviceProcessLifetime::retireActive();
        EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @test Every partial native creation retains the original process obligation through reset. */
TEST(ReadAheadNativeProcessRuntime,PartialCreationAndParentZeroDeviceKeepOriginalProcessTokens){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        auto memory=authority(1024,2048,DeviceId::cuda(0),true);
        for(unsigned failure=1;failure<=3;++failure){
            auto worker=makeWorker();worker->memory=memory;worker->failAt=failure;
            EXPECT_THROW(ReadAheadNativeResourceReplayAccess::create(memory,worker,request(),ReadAheadNativeRetirementPolicy::OriginalProcessRuntime),std::runtime_error);
            EXPECT_EQ(worker->destroys,failure-1);worker.reset();
            {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));NativeContextRetirementReplayAccess::validatedReset(scope);}
            EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
        }
        // Independent terminal population has its own original authority and explicit zero GPU token.
        auto parentMemory=memory;auto worker=makeWorker();worker->memory=parentMemory;worker->parent=true;worker->parentDeviceBytes=0;
        auto parent=ReadAheadNativeParentTerminalReplayAccess::create(parentMemory,worker,request().closure,0,ReadAheadNativeRetirementPolicy::OriginalProcessRuntime);
        worker->ready=true;parent.reset();worker.reset();
        {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));NativeContextRetirementReplayAccess::validatedReset(scope);}
        GPUDeviceProcessLifetime::retireActive();
        EXPECT_EQ(parentMemory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeReadAheadParentTerminal),512u);
        EXPECT_EQ(parentMemory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @test Two pending original slots cannot publish one bounded process population twice. */
TEST(ReadAheadNativeProcessRuntime,InterleavedFirstPublicationRejectsDuplicateBeforeNativeAttempt){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        auto memory=authority(2048,4096);auto worker=makeWorker();
        {auto original=GPUDeviceContextPool::instance().retainExistingContextProvenance(DeviceId::cuda(0));
            auto first=ReadAheadNativeResourceReplayAccess::pending(original,memory);
            auto second=ReadAheadNativeResourceReplayAccess::pending(original,memory);
            ReadAheadNativeResourceReplayAccess::adopt(first,original,memory);
            EXPECT_THROW(ReadAheadNativeResourceReplayAccess::adopt(second,original,memory),std::logic_error);
            EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
            first.completeNativeHandles();
        }
        worker.reset();{auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));NativeContextRetirementReplayAccess::validatedReset(scope);}
        GPUDeviceProcessLifetime::retireActive();
        EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @test Concurrent same-source generation borrowers serialize at the existing process authority. */
TEST(ReadAheadNativeProcessRuntime,ConcurrentBorrowAndCancellationKeepSingleOriginalCharge){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        auto memory=authority();
        for(unsigned replay=0;replay<20;++replay){
            auto worker=makeWorker();worker->memory=memory;std::barrier start(3);std::atomic<unsigned> accepted=0,rejected=0;
            auto attempt=[&]{start.arrive_and_wait();try{
                auto family=ReadAheadNativeResourceReplayAccess::create(memory,worker,request(),ReadAheadNativeRetirementPolicy::OriginalProcessRuntime);++accepted;
            }catch(const std::logic_error&){++rejected;}catch(const PhysicalMemoryCapacityExhausted&){++rejected;}};
            std::thread a(attempt),b(attempt);start.arrive_and_wait();a.join();b.join();
            EXPECT_EQ(accepted.load(),1u);EXPECT_EQ(rejected.load(),1u);EXPECT_EQ(worker->creates,3u);
            worker.reset();{auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(DeviceId::cuda(0));NativeContextRetirementReplayAccess::validatedReset(scope);}
            EXPECT_EQ(memory->reservedBytes(DeviceId::cuda(0),PhysicalMemoryOwner::NativeReadAheadFamily),2048u);
        }
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
} // namespace
} // namespace llaminar2::test
