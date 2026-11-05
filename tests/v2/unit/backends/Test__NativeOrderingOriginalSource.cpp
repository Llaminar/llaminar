/** @file Test__NativeOrderingOriginalSource.cpp
 * @brief Actual installed-origin first-claim rejects foreign PMA before reservation or native creators.
 */
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "backends/ReadAheadNativeResourceFamily.h"
#include "backends/ReadAheadNativeParentTerminal.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
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
}
