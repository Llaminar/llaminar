/** @file Test__NativeOrderingOriginalSource.cpp
 * @brief Actual installed-origin first-claim rejects foreign PMA before reservation or native creators.
 */
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "backends/ReadAheadNativeResourceFamily.h"
#include "backends/ReadAheadNativeParentTerminal.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include <gtest/gtest.h>
namespace llaminar2::test { namespace {
/** @brief Recorded original worker placement and exact stack without SDK resource creation. */
class SourceBoundWorker final:public testing::MockWorkerGPUContext {
public:
    SourceBoundWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(ordinal){thread_.start(std::move(root),[](void*){},nullptr);}
    ~SourceBoundWorker()override{thread_.join();}
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
                EXPECT_STREQ(error.what(),"Ordinary physical storage requires the final Admitted authority phase");
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
        PhysicalMemoryPlanBuilder extra;
        for(std::size_t i=0;i<devices.size();++i){
            const auto one=NativeOrderingEventGroup::bom(program,host,observed[i+1]);
            for(int count=0;count<2;++count)for(const auto& row:one.resources())extra.add(row);
        }
        source->freezeForPlanning();
        PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());const auto extraPlan=extra.build();
        for(const auto& row:extraPlan.resources())plan.add(row);
        source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
        for(std::size_t i=0;i<devices.size();++i){
            const auto device=devices[i];auto worker=permits[i].worker();
            const auto baseline=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent);
            Cache first,second,unpriced;int firstStream=1,secondStream=2;
            ASSERT_TRUE(first.bindBorrowedCaptureStream(worker.get(),&firstStream,device));
            ASSERT_TRUE(second.bindBorrowedCaptureStream(worker.get(),&secondStream,device));
            ASSERT_TRUE(unpriced.bindBorrowedCaptureStream(worker.get(),&firstStream,device));
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
                EXPECT_THROW(unpriced.prepareCaptureStreamTerminal(worker.get(),memory),std::logic_error);
                EXPECT_EQ(unpriced.terminal_event,nullptr);
                // Device budgets are separate, while the one original source/PMA
                // owns the combined host commitment for both backend populations.
            });
            first.reset(Cache::StreamResetPolicy::Destroy);second.reset(Cache::StreamResetPolicy::Destroy);
        }
    }
    for(const auto device:devices){auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}

}

}
