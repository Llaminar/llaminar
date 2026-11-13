/** @file Test__NativeSourceProbePma.cpp
 * @brief Replay the actual diagnostic projection through real Preparing/admitted source PMA states.
 */
#include "integration/backends/NativeContextSourceProbePma.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
namespace llaminar2::test {
namespace {
/** @brief Recorded original placement object/stack owner, without a native SDK call. */
class ProjectionWorker final:public testing::MockWorkerGPUContext {
public:
    ProjectionWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(ordinal){thread_.start(std::move(root),[](void*){},nullptr);}
    ~ProjectionWorker()override{thread_.join();}
private:PreparedGPUWorkerThread thread_;
};
}

/** @test Both backend allocator identities retain the same named attestation across final admission. */
TEST(NativeSourceProbePma, PreparingAndAdmittedUseActualCanonicalAttestation){
    using Access=NativeContextPreparationReplayAccess;
    for(const auto device:{DeviceId::cuda(52),DeviceId::rocm(52)}){
        testing::installHardwareFreeGPUContextFactories();
        Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root){return Access::construct<ProjectionWorker>(std::move(root),device.gpu_ordinal());});
        auto source=Access::source<ProjectionWorker>(device);
        auto memory=source->memory();
        EXPECT_EQ(memory->phase(),PhysicalMemoryAuthorityPhase::Preparing);
        EXPECT_THROW((void)memory->reservedBytes(device,PhysicalMemoryOwner::NativeExecutionContext),std::logic_error);
        EXPECT_EQ(nativeSourcePmaProjection(memory,device)["device_opaque_bytes"],8192u);
        {auto permit=source->acquire(device);EXPECT_EQ(permit.memory(),memory);}
        const auto before=nativeSourcePmaProjection(memory,device);
        EXPECT_EQ(before["phase"],"Preparing");EXPECT_EQ(before["host_opaque_bytes"],4096u);EXPECT_EQ(before["device_opaque_bytes"],8192u);
        EXPECT_EQ(before["opaque_committed_bytes"],12288u);EXPECT_GT(before["exact_preparation_bytes"].get<std::size_t>(),0u);
        const auto other=device.is_cuda()?DeviceId::rocm(52):DeviceId::cuda(52);
        EXPECT_EQ(nativeSourcePmaProjection(memory,other)["device_opaque_bytes"],0u);
        source->freezeForPlanning();source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(source->mandatoryBOM()));
        auto after=nativeSourcePmaProjection(memory,device);EXPECT_EQ(after["phase"],"Admitted");after["phase"]="Preparing";EXPECT_EQ(after,before);
        EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeExecutionContext),8192u);
        {auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}
    }
}
}
