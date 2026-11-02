/**
 * @file Test__PreparedSourceProcessLifetime.cpp
 * @brief Replay original selected-worker process backing through real source/pool/PMA operations.
 *
 * Only native constructor bodies and reviewed limits are recorded fixtures.
 * Production source birth, exact mapped thread/object, canonical reservations,
 * publication, failed prefix, reset and original process retention all execute.
 */
#include "backends/NativeContextPreparation.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "backends/GPUDeviceContextPool.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
namespace llaminar2::test {
namespace {
/** @brief Recorded actual original mapped-stack worker with no native accelerator calls. */
class ProcessSourceWorker final:public testing::MockWorkerGPUContext {
public:
    ProcessSourceWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> stack):MockWorkerGPUContext(ordinal){worker_.start(std::move(stack),[](void*){},nullptr);}
    ~ProcessSourceWorker()override{worker_.join();}
private:PreparedGPUWorkerThread worker_;
};
/** @return Actual canonical opaque commitments, without a parallel live budget. */
std::size_t opaque(const std::shared_ptr<PhysicalMemoryAuthority>& memory){std::size_t total=0;for(const auto& value:memory->rankAttestation())total+=value.opaque_runtime_committed_bytes;return total;}
/** @return Actual still-mapped preparation roots from the canonical ledger. */
std::size_t exact(const std::shared_ptr<PhysicalMemoryAuthority>& memory){std::size_t total=0;for(const auto& value:memory->rankAttestation())total+=value.preparation_materialized_bytes;return total;}
}
/** @test Same original process tokens survive both quota failure and admitted worker reset on both backends. */
TEST(PreparedSourceProcessLifetime, OriginalTokensSurvivePreAdmissionFailureAndAdmittedReset){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto device:{DeviceId::cuda(77),DeviceId::rocm(77)})for(const bool admit:{false,true}){
        ASSERT_EXIT(([device,admit]{
            using Access=NativeContextPreparationReplayAccess;
            Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root){return Access::construct<ProcessSourceWorker>(std::move(root),device.gpu_ordinal());});
            auto source=Access::source<ProcessSourceWorker>(device,64u<<20,NativePhysicalLifetime::ProcessRuntime);
            auto memory=source->memory();EXPECT_EQ(opaque(memory),12288u);EXPECT_GT(exact(memory),0u);
            EXPECT_NO_THROW(Access::requireProcessRetention(source,memory));
            auto alias=std::shared_ptr<PhysicalMemoryAuthority>(memory.get(),[](auto*){});
            EXPECT_THROW(Access::requireProcessRetention(source,alias),std::logic_error);alias.reset();
            {auto permit=source->acquire(device);EXPECT_EQ(permit.memory(),memory);}
            source->freezeForPlanning();auto certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(source->mandatoryBOM());
            if(admit)source->admitFinal(certificate);
            else {
                PhysicalMemoryPlanBuilder tooLarge(source->mandatoryBOM());
                tooLarge.add(source->originalResource(device),PhysicalMemoryOwner::PrimaryModelWeights,64u<<20);
                EXPECT_FALSE(tooLarge.build().fits());
                EXPECT_THROW(source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(tooLarge.build())),std::exception);
                EXPECT_EQ(memory->phase(),PhysicalMemoryAuthorityPhase::Preparing);
                EXPECT_THROW(Access::cancel(source),std::logic_error);
            }
            {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}
            EXPECT_EQ(opaque(memory),12288u);EXPECT_EQ(exact(memory),0u);
            EXPECT_THROW(source->acquire(device),std::logic_error);
            EXPECT_THROW(Access::source<ProcessSourceWorker>(device),std::logic_error);
            const auto different=device.is_cuda()?DeviceId::rocm(76):DeviceId::cuda(76);
            EXPECT_THROW(Access::source<ProcessSourceWorker>(different),std::logic_error);
            if(!admit)EXPECT_THROW(source->admitFinal(certificate),std::logic_error);
            std::weak_ptr<PhysicalMemoryAuthority> retained=memory;source.reset();memory.reset();EXPECT_FALSE(retained.expired());
            GPUDeviceProcessLifetime::retireActive();EXPECT_FALSE(retained.expired());
            std::_Exit(::testing::Test::HasFailure()?2:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}
/** @test A failed original creator prefix retains the same unpublished process commitment after reset. */
TEST(PreparedSourceProcessLifetime, FailedCreatorPrefixCannotCancelOrDischargeProcessBacking){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto device:{DeviceId::cuda(78),DeviceId::rocm(78)}){
        ASSERT_EXIT(([device]{
            using Access=NativeContextPreparationReplayAccess;
            Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root)->std::shared_ptr<IWorkerGPUContext>{
                auto initialized=Access::construct<ProcessSourceWorker>(std::move(root),device.gpu_ordinal());
                throw std::runtime_error("recorded failure after initialized original worker prefix");
            });
            auto source=Access::source<ProcessSourceWorker>(device,64u<<20,NativePhysicalLifetime::ProcessRuntime);auto memory=source->memory();
            EXPECT_THROW(source->acquire(device),std::runtime_error);EXPECT_EQ(opaque(memory),12288u);
            EXPECT_THROW(source->freezeForPlanning(),std::logic_error);
            EXPECT_THROW(Access::cancel(source),std::logic_error);
            {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}
            EXPECT_EQ(exact(memory),0u);EXPECT_EQ(opaque(memory),12288u);
            EXPECT_THROW(Access::source<ProcessSourceWorker>(device),std::logic_error);
            source.reset();GPUDeviceProcessLifetime::retireActive();EXPECT_EQ(opaque(memory),12288u);
            std::_Exit(::testing::Test::HasFailure()?2:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}
}
