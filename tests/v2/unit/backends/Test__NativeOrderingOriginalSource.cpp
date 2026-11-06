/** @file Test__NativeOrderingOriginalSource.cpp
 * @brief Actual installed-origin first-claim rejects foreign PMA before reservation or native creators.
 *
 * Failed reservations retain the original admission exception and cause, with
 * the exact native program and context attached for model-loading diagnosis.
 */
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "backends/ReadAheadNativeResourceFamily.h"
#include "backends/ReadAheadNativeParentTerminal.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include <gtest/gtest.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdio>
#include <type_traits>
#include "execution/local_execution/graph/PreparedFollowerCaptureTerminals.h"
namespace llaminar2::test { namespace {
static_assert(!std::is_copy_constructible_v<PreparedFollowerCaptureTerminalPopulation>);
static_assert(!std::is_move_constructible_v<PreparedFollowerCaptureTerminalPopulation>);
static_assert(!std::is_copy_assignable_v<PreparedFollowerCaptureTerminalPopulation>);
static_assert(!std::is_move_assignable_v<PreparedFollowerCaptureTerminalPopulation>);
static_assert(!std::is_copy_constructible_v<PreparedFollowerCaptureTerminalOwner>);
static_assert(!std::is_move_constructible_v<PreparedFollowerCaptureTerminalOwner>);
static_assert(!std::is_copy_assignable_v<PreparedFollowerCaptureTerminalOwner>);
static_assert(!std::is_move_assignable_v<PreparedFollowerCaptureTerminalOwner>);

/** @brief Recorded original worker placement and exact stack without SDK resource creation. */
class SourceBoundWorker final:public testing::MockWorkerGPUContext {
public:
    SourceBoundWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(ordinal){thread_.start(std::move(root),[](void*){},nullptr);}
    ~SourceBoundWorker()override{thread_.join();}
    /** @brief Replay a precise native creator failure after earlier handles were accepted. */
    void* createEvent(GPUEventPurpose purpose=GPUEventPurpose::Ordering)override{
        ++eventAttempts;
        if(failEventAttempt && eventAttempts==failEventAttempt)return nullptr;
        return MockWorkerGPUContext::createEvent(purpose);
    }
    /** @brief Preserve original destroy behavior while observing partial accepted-handle cleanup. */
    void destroyEvent(void* event)override{++eventDestructions;MockWorkerGPUContext::destroyEvent(event);}
    int eventAttempts=0,eventDestructions=0,failEventAttempt=0;
private:PreparedGPUWorkerThread thread_;
};
}
/** @test First installed-origin call cannot bind another PMA, including same-address foreign shared ownership. */
TEST(NativeOrderingOriginalSource, FirstInstalledCommonClaimAuthenticatesActualSourceBeforeReservations){
    using Access=NativeContextPreparationReplayAccess;
    for(const auto device:{DeviceId::cuda(51),DeviceId::rocm(51)}){
        testing::installHardwareFreeGPUContextFactories();
        Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root){return Access::construct<SourceBoundWorker>(std::move(root),device.gpu_ordinal());});
        auto source=Access::source<SourceBoundWorker>(device);auto memory=source->memory();
        PhysicalMemoryPlanBuilder builder;for(const auto target:{DeviceId::cpu(),device}){
            PhysicalMemoryResource r{0,target,64u<<20,64u<<20};
            builder.add(r,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,1024);
            builder.add(r,PhysicalMemoryOwner::NativeReadAheadFamily,1024);
            builder.add(r,PhysicalMemoryOwner::NativeReadAheadParentTerminal,1024);
        }
        auto foreign=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
        auto alias=std::shared_ptr<PhysicalMemoryAuthority>(memory.get(),[](auto*){});
        {
            auto permit=source->acquire(device);const auto original=permit.provenance();
            const auto program=NativeOrderingEventProgram::lookupSubmissionRetirement();
            const auto installed=NativeOrderingEventBudgetProjection::installed();
            for(const auto& wrong:{foreign,alias}){
                EXPECT_THROW(NativeOrderingEventGroup::claim(wrong,original,program),std::logic_error);
                EXPECT_THROW(NativeOrderingEventGroupReplayAccess::claim(wrong,original,program,installed),std::logic_error);
                EXPECT_THROW(ReadAheadNativeResourceFamily::materialize(wrong,permit.worker(),device,reinterpret_cast<void*>(100)),std::logic_error);
                EXPECT_THROW(ReadAheadNativeParentTerminal::materialize(wrong,permit.worker(),device),std::logic_error);
            }
            for(const auto owner:{PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,PhysicalMemoryOwner::NativeReadAheadFamily,PhysicalMemoryOwner::NativeReadAheadParentTerminal}){
                EXPECT_EQ(foreign->reservedBytes(DeviceId::cpu(),owner),0u);EXPECT_EQ(foreign->reservedBytes(device,owner),0u);
            }
            // Supported ordering families still require final admission on the SAME original source.
            EXPECT_EQ(memory->phase(),PhysicalMemoryAuthorityPhase::Preparing);
            try {
                auto premature=NativeOrderingEventGroupReplayAccess::claim(memory,original,program,installed);
                FAIL()<<"An original Preparing source claimed an ordering group before final admission";
            } catch(const std::logic_error& error) {
                EXPECT_EQ(std::string(error.what()),
                    "Native ordering reservation failed: program_kind=3 root_mask=0 event_count=1 device="+
                    device.toString()+" context_generation="+std::to_string(original.generation())+
                    " allocation_device=CPU cause={Ordinary physical storage requires the final Admitted authority phase}");
            }
            EXPECT_EQ(memory->phase(),PhysicalMemoryAuthorityPhase::Preparing);
            // Supported side/parent families cannot allocate from a Preparing source.

            EXPECT_THROW(ReadAheadNativeResourceFamily::materialize(memory,permit.worker(),device,reinterpret_cast<void*>(100)),std::logic_error);
            EXPECT_THROW(ReadAheadNativeParentTerminal::materialize(memory,permit.worker(),device),std::logic_error);
            // Ledger reads themselves require Admitted. Freeze only the original
            // source BOM before proving that the rejected calls reserved nothing.
            source->freezeForPlanning();
            source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(source->mandatoryBOM()));
            // Mandatory-only admission grants no auxiliary-family budget.
            EXPECT_THROW(ReadAheadNativeResourceFamily::materialize(memory,permit.worker(),device,reinterpret_cast<void*>(100)),std::logic_error);
            EXPECT_THROW(ReadAheadNativeParentTerminal::materialize(memory,permit.worker(),device),std::logic_error);
            for(const auto owner:{PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,PhysicalMemoryOwner::NativeReadAheadFamily,PhysicalMemoryOwner::NativeReadAheadParentTerminal}){
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),owner),0u);
                EXPECT_EQ(memory->reservedBytes(device,owner),0u);
            }
        }
        {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}
    }
}
/** @test Two real cache creators claim two distinct populations from their SAME admitted source. */
TEST(NativeOrderingOriginalSource, CaptureTerminalSingletonsUseOriginalPMAAndExactCreatorPopulation){
    using Access=NativeContextPreparationReplayAccess;
    using Cache=DeviceGraphExecutor::GraphSegmentCache;
    const auto program=NativeOrderingEventProgram::captureSubmissionRetirement();
    EXPECT_EQ(program.orderingEvents(),1u);
    const std::array devices{DeviceId::cuda(52),DeviceId::rocm(52)};
    testing::installHardwareFreeGPUContextFactories();
    for(const auto device:devices)
        Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root){return Access::construct<SourceBoundWorker>(std::move(root),device.gpu_ordinal());});
    const PhysicalMemoryResource host{0,DeviceId::cpu(),128u<<20,128u<<20};
    const std::array observed{host,PhysicalMemoryResource{0,devices[0],64u<<20,64u<<20},PhysicalMemoryResource{0,devices[1],64u<<20,64u<<20}};
    auto source=Access::source<SourceBoundWorker>(std::span<const DeviceId>{devices},observed);
    auto memory=source->memory();
    {
        std::vector<PreparedGPUContextUsePermit> permits;
        for(const auto device:devices)permits.push_back(source->acquire(device));
        const auto geometry=MoEOverlayInferenceOriginalSlotGeometry::forRetainedDraftCapacity(3);
        PhysicalMemoryPlanBuilder extra;
        for(std::size_t i=0;i<devices.size();++i){
            const auto one=NativeOrderingEventGroup::bom(program,host,observed[i+1]);
            for(int count=0;count<2;++count)for(const auto& row:one.resources())extra.add(row);
            const auto recurrent=OriginalSlotCaptureTerminals::extraEventBOM(geometry,host,observed[i+1]);
            for(const auto& row:recurrent.resources())extra.add(row);
            // A separate cold partial-creator failure retains its own full
            // opaque allowance; never borrow the successful cache population.
            for(const auto& row:one.resources())extra.add(row);
            for(const auto& row:recurrent.resources())extra.add(row);
        }
        source->freezeForPlanning();
        PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());const auto extraPlan=extra.build();
        for(const auto& row:extraPlan.resources())plan.add(row);
        source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
        for(std::size_t i=0;i<devices.size();++i){
            const auto device=devices[i];auto worker=permits[i].worker();
            const auto baseline=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent);
            const MoEOverlayInferenceTopologyIdentity topology{.workspace_generation=71,
                .topology_fingerprint_low=0x1234,.topology_fingerprint_high=0x5678,
                .source_world_rank=0,.target_world_rank=1};
            MoEOverlayInferenceTransactionProtocol protocol({.topology=topology,
                .slot_count=geometry.slotCount(),.max_request_count=1,.max_rows_per_request=1,.max_mtp_draft_depth=3});
            // Actual protocol outlives every cache drain; no receipt survives its original owner.
            Cache first,second,unpriced,partial;int firstStream=1,secondStream=2;
            ASSERT_TRUE(first.bindBorrowedCaptureStream(worker.get(),&firstStream,device));
            ASSERT_TRUE(second.bindBorrowedCaptureStream(worker.get(),&secondStream,device));
            ASSERT_TRUE(unpriced.bindBorrowedCaptureStream(worker.get(),&firstStream,device));
            ASSERT_TRUE(partial.bindBorrowedCaptureStream(worker.get(),&secondStream,device));
            worker->submitAndWait([&]{
                auto alias=std::shared_ptr<PhysicalMemoryAuthority>(memory.get(),[](auto*){});
                EXPECT_THROW(first.prepareCaptureStreamTerminal(worker.get(),alias),std::logic_error);
                EXPECT_EQ(first.terminal_event,nullptr);
                ASSERT_TRUE(first.prepareCaptureStreamTerminal(worker.get(),memory));
                const auto one=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent)-baseline;
                EXPECT_GT(one,0u);
                ASSERT_TRUE(first.prepareCaptureStreamTerminal(worker.get(),memory));
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),baseline+one);
                ASSERT_TRUE(second.prepareCaptureStreamTerminal(worker.get(),memory));
                EXPECT_NE(first.terminal_event,second.terminal_event);
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),baseline+2*one);
                EXPECT_THROW(first.bindOriginalCaptureTerminalSlots(protocol.originalSlotDomain(),memory,
                    MoEOverlayInferenceOriginalSlotGeometry::forRetainedDraftCapacity(1)),std::logic_error);
                first.bindOriginalCaptureTerminalSlots(protocol.originalSlotDomain(),memory,geometry);
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),
                    baseline+(geometry.slotCount()+1)*one);
                ASSERT_TRUE(partial.prepareCaptureStreamTerminal(worker.get(),memory));
                auto& observedWorker=dynamic_cast<SourceBoundWorker&>(*worker);
                const auto destroyed=observedWorker.eventDestructions;
                observedWorker.failEventAttempt=observedWorker.eventAttempts+2;
                EXPECT_THROW(partial.bindOriginalCaptureTerminalSlots(protocol.originalSlotDomain(),memory,geometry),std::runtime_error);
                observedWorker.failEventAttempt=0;
                EXPECT_EQ(observedWorker.eventDestructions,destroyed+1);
                EXPECT_FALSE(partial.original_slot_terminals);
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),
                    baseline+(2*geometry.slotCount()+1)*one);
                EXPECT_THROW(unpriced.prepareCaptureStreamTerminal(worker.get(),memory),std::logic_error);
                EXPECT_EQ(unpriced.terminal_event,nullptr);
                // Device budgets are separate, while the one original source/PMA
                // owns the combined host commitment for both backend populations.
            });
            for(std::uint64_t request=1;request<=20;++request){
                const MoEOverlayInferenceCommandIdentity command{request,request,41};
                ASSERT_TRUE(protocol.beginCommand(command));
                std::vector<Cache::CaptureStreamTerminalTicket> receipts;
                std::vector<MoEOverlayInferenceOriginalSlot> slots;
                for(std::uint64_t ordinal=1;ordinal<=3;++ordinal){
                    const auto ticket=makeMoEOverlayInferenceExecutionTicket(topology,command,ordinal,ordinal,41,
                        MoEOverlayInferenceGraphRole::MTPDraft,1,1,1,3,int(ordinal-1));
                    const auto admitted=protocol.accept(ticket);ASSERT_TRUE(admitted.accepted());
                    ASSERT_TRUE(protocol.markSubmitted(admitted.slot_index,ticket));
                    slots.push_back(protocol.originalSubmittedSlot(ticket));
                    worker->submitAndWait([&]{receipts.push_back(first.publishCaptureStreamTerminal(slots.back()));});
                }
                // Moving the actual cache cannot replace event/token identity
                // or drop any outstanding invocation's physical ownership.
                if(request==10){Cache moved(std::move(first));first=std::move(moved);}
                // All three exact records exist before the first host observation.
                worker->submitAndWait([&]{first.enqueuePublishedCaptureStreamTerminalWait(receipts.back(),&secondStream,worker.get());});
                for(const auto& receipt:receipts)
                    first.waitForPublishedCaptureStreamTerminal(receipt,Cache::HostFenceWaitPolicy::ActiveProgress);
                worker->submitAndWait([&]{first.enqueuePublishedCaptureStreamTerminalWait(receipts.back(),&secondStream,worker.get());});
                for(const auto& slot:slots){
                    ASSERT_TRUE(protocol.markReturnReady(slot.index(),slot.ticket()));
                    ASSERT_TRUE(protocol.retire(slot.index(),slot.ticket()));
                }
                const auto complete=makeMoEOverlayInferenceTerminalTicket(topology,command,4,41,
                    MoEOverlayInferenceTransactionAction::Complete);
                ASSERT_TRUE(protocol.accept(complete).completed());
                first.reset(Cache::StreamResetPolicy::Preserve);
            }
            const MoEOverlayInferenceCommandIdentity failedCommand{21,21,41};
            ASSERT_TRUE(protocol.beginCommand(failedCommand));
            std::vector<MoEOverlayInferenceOriginalSlot> failedSlots;
            for(std::uint64_t ordinal=1;ordinal<=2;++ordinal){
                const auto ticket=makeMoEOverlayInferenceExecutionTicket(topology,failedCommand,ordinal,ordinal,41,
                    MoEOverlayInferenceGraphRole::MTPDraft,1,1,1,3,int(ordinal-1));
                const auto admitted=protocol.accept(ticket);ASSERT_TRUE(admitted.accepted());
                ASSERT_TRUE(protocol.markSubmitted(admitted.slot_index,ticket));
                failedSlots.push_back(protocol.originalSubmittedSlot(ticket));
                worker->submitAndWait([&]{(void)first.publishCaptureStreamTerminal(failedSlots.back());});
            }
            const auto aborted=makeMoEOverlayInferenceTerminalTicket(topology,failedCommand,3,41,
                MoEOverlayInferenceTransactionAction::Abort,91);
            ASSERT_EQ(protocol.accept(aborted).status,MoEOverlayInferenceAdmissionStatus::Aborted);
            EXPECT_THROW(protocol.originalSlotDomain()->requireActiveAdmission(failedSlots.front()),std::logic_error);
            // The actual cache destructor/reset drains both retained native
            // records even after logical abort, before original slots retire.
            first.reset(Cache::StreamResetPolicy::Destroy);
            for(const auto& slot:failedSlots){
                ASSERT_TRUE(protocol.markReturnReady(slot.index(),slot.ticket()));
                ASSERT_TRUE(protocol.retire(slot.index(),slot.ticket()));
            }
            second.reset(Cache::StreamResetPolicy::Destroy);
            partial.reset(Cache::StreamResetPolicy::Destroy);
        }
    }
    for(const auto device:devices){auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}

}

/** @test Runner recreation borrows the SAME actual event population with fresh original protocol receipts. */
TEST(NativeOrderingOriginalSource, PreparedFollowerRecreationRetainsOneOriginalEventPopulation){
    // ProcessRuntime commitments deliberately cannot be adopted by a second
    // independent source in one process. Re-exec the actual fixture instead.
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT(([]{
        const rlimit limit{0,0};
        if(setrlimit(RLIMIT_CORE,&limit)!=0){std::fputs("could not disable child core dumps\n",stderr);_exit(2);}
        {
    using Access=NativeContextPreparationReplayAccess;
    using Cache=DeviceGraphExecutor::GraphSegmentCache;
    const std::array devices{DeviceId::cuda(53),DeviceId::rocm(53)};
    testing::installHardwareFreeGPUContextFactories();
    for(const auto device:devices)Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> stack){
        return Access::construct<SourceBoundWorker>(std::move(stack),device.gpu_ordinal());});
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),128u<<20,128u<<20};
    const std::array resources{cpu,PhysicalMemoryResource{0,devices[0],64u<<20,64u<<20},PhysicalMemoryResource{0,devices[1],64u<<20,64u<<20}};
    auto source=Access::source<SourceBoundWorker>(std::span<const DeviceId>{devices},resources);
    auto memory=source->memory();
    {
        std::array permits{source->acquire(devices[0]),source->acquire(devices[1])};
        const auto geometry=MoEOverlayInferenceOriginalSlotGeometry::forRetainedDraftCapacity(3);
        PhysicalMemoryPlanBuilder extra;
        extra.add(cpu,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,PreparedFollowerCaptureTerminalOwner::physicalBytes());
        for(std::size_t i=0;i<devices.size();++i){
            extra.add(cpu,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,PreparedFollowerCaptureTerminalPopulation::physicalBytes(geometry));
            for(const auto program:{NativeOrderingEventProgram::captureSubmissionRetirement(),NativeOrderingEventProgram::retainedCaptureTerminals(geometry)}){
                const auto bom=NativeOrderingEventGroup::bom(program,cpu,resources[i+1]);
                for(const auto& row:bom.resources())extra.add(row);
            }
        }
        source->freezeForPlanning();PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
        const auto extraPlan=extra.build();for(const auto& row:extraPlan.resources())plan.add(row);
        source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
        auto owner=PreparedFollowerCaptureTerminalOwner::create(memory,geometry);
        std::array<void*,2> originalEvents{};std::uint64_t reserved=0;
        for(int cycle=0;cycle<20;++cycle){
            std::array<Cache,2> caches;
            std::array<std::unique_ptr<MoEOverlayInferenceTransactionProtocol>,2> protocols;
            for(std::size_t i=0;i<devices.size();++i){
                const MoEOverlayInferenceTopologyIdentity topology{.workspace_generation=std::uint64_t(71+cycle),
                    .topology_fingerprint_low=0x1234,.topology_fingerprint_high=0x5678,.source_world_rank=0,.target_world_rank=1};
                protocols[i]=std::make_unique<MoEOverlayInferenceTransactionProtocol>(MoEOverlayInferenceTransactionProtocol::Config{
                    .topology=topology,.slot_count=geometry.slotCount(),.max_request_count=1,.max_rows_per_request=1,.max_mtp_draft_depth=3});
                auto worker=permits[i].worker();auto* stream=worker->defaultStream();
                ASSERT_TRUE(caches[i].bindBorrowedCaptureStream(worker.get(),stream,devices[i]));
                const PreparedFollowerCaptureTerminalKey key{int(i),0,1,devices[i]};
                worker->submitAndWait([&]{
#ifdef LLAMINAR_OLD_FOLLOWER_TERMINAL_RECREATION
                    ASSERT_TRUE(caches[i].prepareCaptureStreamTerminal(worker.get(),memory));
                    caches[i].bindOriginalCaptureTerminalSlots(protocols[i]->originalSlotDomain(),memory,geometry);
#else
                    auto population=owner->acquire(key,permits[i],stream);
                    EXPECT_THROW(owner->acquire(key,permits[i],stream),std::logic_error);
                    caches[i].bindPreparedFollowerCaptureTerminals(protocols[i]->originalSlotDomain(),std::move(population),memory);
#endif
                });
#ifndef LLAMINAR_OLD_FOLLOWER_TERMINAL_RECREATION
                if(!cycle)originalEvents[i]=caches[i].terminal_event;else EXPECT_EQ(caches[i].terminal_event,originalEvents[i]);
#endif
                const MoEOverlayInferenceCommandIdentity command{1,1,41};ASSERT_TRUE(protocols[i]->beginCommand(command));
                const auto ticket=makeMoEOverlayInferenceExecutionTicket(topology,command,1,1,41,MoEOverlayInferenceGraphRole::MTPDraft,1,1,1,3,0);
                const auto admitted=protocols[i]->accept(ticket);ASSERT_TRUE(admitted.accepted());ASSERT_TRUE(protocols[i]->markSubmitted(admitted.slot_index,ticket));
                const auto slot=protocols[i]->originalSubmittedSlot(ticket);
                std::optional<Cache::CaptureStreamTerminalTicket> receipt;
                worker->submitAndWait([&]{receipt=caches[i].publishCaptureStreamTerminal(slot);});
                caches[i].waitForPublishedCaptureStreamTerminal(*receipt,Cache::HostFenceWaitPolicy::ActiveProgress);
                ASSERT_TRUE(protocols[i]->markReturnReady(slot.index(),ticket));ASSERT_TRUE(protocols[i]->retire(slot.index(),ticket));
                const auto complete=makeMoEOverlayInferenceTerminalTicket(topology,command,2,41,MoEOverlayInferenceTransactionAction::Complete);
                ASSERT_TRUE(protocols[i]->accept(complete).completed());
            }
#ifndef LLAMINAR_OLD_FOLLOWER_TERMINAL_RECREATION
            owner->seal();
#endif
            const auto current=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent);
            if(!cycle)reserved=current;else EXPECT_EQ(current,reserved);
            for(auto& cache:caches)cache.reset(Cache::StreamResetPolicy::Destroy);
#ifndef LLAMINAR_OLD_FOLLOWER_TERMINAL_RECREATION
            EXPECT_THROW(owner->seal(),std::logic_error); // No current runner may omit the original roster.
#endif
            for(std::size_t i=0;i<devices.size();++i){
                auto& worker=dynamic_cast<SourceBoundWorker&>(*permits[i].worker());
                // bindBorrowedCaptureStream retains its existing separate sync event;
                // that cache-local event is created/destroyed once per runner.
#ifndef LLAMINAR_OLD_FOLLOWER_TERMINAL_RECREATION
                EXPECT_EQ(worker.eventAttempts,int(geometry.slotCount())+cycle+1);
                EXPECT_EQ(worker.eventDestructions,cycle+1);
#endif
            }
        }
        auto worker=permits[0].worker();auto* stream=worker->defaultStream();
        worker->submitAndWait([&]{
            EXPECT_THROW(owner->acquire({0,0,2,devices[0]},permits[0],stream),std::invalid_argument);
            EXPECT_THROW(owner->acquire({0,0,1,devices[0]},permits[0],reinterpret_cast<void*>(0x1234)),std::invalid_argument);
        });
#ifndef LLAMINAR_OLD_FOLLOWER_TERMINAL_RECREATION
        const auto changedGeometry=MoEOverlayInferenceOriginalSlotGeometry::forRetainedDraftCapacity(1);
        MoEOverlayInferenceTransactionProtocol changed({.topology={.workspace_generation=999,
            .topology_fingerprint_low=0x1234,.topology_fingerprint_high=0x5678,.source_world_rank=0,.target_world_rank=1},
            .slot_count=changedGeometry.slotCount(),.max_request_count=1,.max_rows_per_request=1,.max_mtp_draft_depth=1});
        Cache wrongGeometry;ASSERT_TRUE(wrongGeometry.bindBorrowedCaptureStream(worker.get(),stream,devices[0]));
        worker->submitAndWait([&]{auto population=owner->acquire({0,0,1,devices[0]},permits[0],stream);
            EXPECT_THROW(wrongGeometry.bindPreparedFollowerCaptureTerminals(changed.originalSlotDomain(),std::move(population),memory),std::invalid_argument);});
        wrongGeometry.reset(Cache::StreamResetPolicy::Destroy);
#endif
        owner.reset();
        for(std::size_t i=0;i<devices.size();++i)EXPECT_EQ(dynamic_cast<SourceBoundWorker&>(*permits[i].worker()).eventDestructions,
            dynamic_cast<SourceBoundWorker&>(*permits[i].worker()).eventAttempts);
    }
    for(const auto device:devices){auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}

        }
        _exit(::testing::Test::HasFailure()?1:0);
    }()),::testing::ExitedWithCode(0),"");
}

/** @test A partial original creator is retired once and its source failure remains absorbing. */
TEST(NativeOrderingOriginalSource, PreparedFollowerPartialCreatorFailureCannotAllocateReplacement){
    // ProcessRuntime commitments deliberately cannot be adopted by a second
    // independent source in one process. Re-exec the actual fixture instead.
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    EXPECT_EXIT(([]{
        const rlimit limit{0,0};
        if(setrlimit(RLIMIT_CORE,&limit)!=0){std::fputs("could not disable child core dumps\n",stderr);_exit(2);}
        {
    using Access=NativeContextPreparationReplayAccess;
    const std::array devices{DeviceId::cuda(54),DeviceId::rocm(54)};
    testing::installHardwareFreeGPUContextFactories();
    for(const auto device:devices)Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> stack){
        return Access::construct<SourceBoundWorker>(std::move(stack),device.gpu_ordinal());});
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),128u<<20,128u<<20};
    const std::array resources{cpu,PhysicalMemoryResource{0,devices[0],64u<<20,64u<<20},PhysicalMemoryResource{0,devices[1],64u<<20,64u<<20}};
    auto source=Access::source<SourceBoundWorker>(std::span<const DeviceId>{devices},resources);auto memory=source->memory();
    {
        std::array permits{source->acquire(devices[0]),source->acquire(devices[1])};
        const auto geometry=MoEOverlayInferenceOriginalSlotGeometry::forRetainedDraftCapacity(3);
        PhysicalMemoryPlanBuilder extra;
        for(std::size_t i=0;i<devices.size();++i){
            extra.add(cpu,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,
                PreparedFollowerCaptureTerminalOwner::physicalBytes()+PreparedFollowerCaptureTerminalPopulation::physicalBytes(geometry));
            for(const auto program:{NativeOrderingEventProgram::captureSubmissionRetirement(),NativeOrderingEventProgram::retainedCaptureTerminals(geometry)}){
                const auto bom=NativeOrderingEventGroup::bom(program,cpu,resources[i+1]);for(const auto& row:bom.resources())extra.add(row);
            }
        }
        source->freezeForPlanning();PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
        const auto additions=extra.build();for(const auto& row:additions.resources())plan.add(row);
        source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
        for(std::size_t i=0;i<devices.size();++i){
            auto owner=PreparedFollowerCaptureTerminalOwner::create(memory,geometry);
            auto worker=permits[i].worker();auto& observed=dynamic_cast<SourceBoundWorker&>(*worker);
            observed.failEventAttempt=2;
            worker->submitAndWait([&]{
                const PreparedFollowerCaptureTerminalKey key{int(i),0,1,devices[i]};
                EXPECT_THROW(owner->acquire(key,permits[i],worker->defaultStream()),std::runtime_error);
                EXPECT_EQ(observed.eventAttempts,2);EXPECT_EQ(observed.eventDestructions,1);
                const auto reserved=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent);
                const auto claimed=memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,PhysicalMemoryMaterializationKind::NewAllocation);
                for(int retry=0;retry<20;++retry)EXPECT_THROW(owner->acquire(key,permits[i],worker->defaultStream()),std::runtime_error);
                EXPECT_THROW(owner->seal(),std::runtime_error);
                EXPECT_EQ(observed.eventAttempts,2);EXPECT_EQ(observed.eventDestructions,1);
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),reserved);
                EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,PhysicalMemoryMaterializationKind::NewAllocation),claimed);
            });
        }
    }
    for(const auto device:devices){auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}

        }
        _exit(::testing::Test::HasFailure()?1:0);
    }()),::testing::ExitedWithCode(0),"");
}

}
