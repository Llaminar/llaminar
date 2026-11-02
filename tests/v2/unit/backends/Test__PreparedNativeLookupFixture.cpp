/** @file Test__PreparedNativeLookupFixture.cpp
 * @brief Device-free actual source acquisition/freeze/admission with original mapped worker and payload roots.
 */
#include "fixtures/backends/PreparedNativeLookupFixtureReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "tensors/TensorClasses.h"
#include <gtest/gtest.h>
namespace llaminar2::test {namespace {
/** @brief Actual prepared stack/object ownership with recorded accelerator constructor inputs. */
class LookupFixtureWorker final:public testing::MockWorkerGPUContext {
public:
    LookupFixtureWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(ordinal){thread_.start(std::move(root),[](void*){},nullptr);}
    ~LookupFixtureWorker()override{thread_.join();}
private:PreparedGPUWorkerThread thread_;
};
}
/** @test Original source and physical CPU payload remain on the same authority through actual admission. */
TEST(PreparedNativeLookupFixture, OriginalSourceFreezeAdmitAndPayloadResetReplay){
    if(!hasCPUBackend())initCPUBackend(1);
    using Source=NativeContextPreparationReplayAccess;using Fixture=PreparedNativeLookupFixtureReplayAccess;
    for(const auto device:{DeviceId::cuda(49),DeviceId::rocm(49)}){
        testing::installHardwareFreeGPUContextFactories();
        Source::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root){return Source::construct<LookupFixtureWorker>(std::move(root),device.gpu_ordinal());});
        auto source=Source::source<LookupFixtureWorker>(device);const auto original=source->memory();const std::array devices{device};
        auto fixture=Fixture::prepare(source,devices);EXPECT_THROW(fixture->retirementReceipts(),std::logic_error);EXPECT_EQ(fixture->memory(),original);EXPECT_EQ(fixture->source(),source);
        EXPECT_EQ(original->phase(),PhysicalMemoryAuthorityPhase::Preparing);EXPECT_EQ(fixture->permit(device).memory(),original);
        EXPECT_EQ(fixture->resource(DeviceId::cpu()).device,DeviceId::cpu());EXPECT_EQ(fixture->resource(device).device,device);
        const CPUTensorStorageDeclaration declaration(1,16,BufferTensorType::INT32,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu());
        const CPUTensorStorageDeclaration sameOwner(1,16,BufferTensorType::INT32,PhysicalMemoryOwner::LocalCollective,DeviceId::cpu());
        PhysicalMemoryPlanBuilder payload;payload.add(fixture->resource(DeviceId::cpu()),PhysicalMemoryOwner::ActivationArena,declaration.physicalBytes());
        payload.add(fixture->resource(DeviceId::cpu()),PhysicalMemoryOwner::LocalCollective,sameOwner.physicalBytes());
        fixture->admit(payload.build());EXPECT_EQ(original->phase(),PhysicalMemoryAuthorityPhase::Admitted);EXPECT_EQ(fixture->memory(),original);
        EXPECT_THROW(fixture->admit(payload.build()),std::logic_error);
        EXPECT_GT(original->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::LocalCollective,PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_NO_THROW(fixture->requirePayloadRetired());
        EXPECT_THROW(fixture->requireExactRetired(),std::logic_error);
        {std::shared_ptr<TensorBase> root(declaration.materialize(original));EXPECT_NE(root->raw_data(),nullptr);EXPECT_THROW(fixture->requirePayloadRetired(),std::logic_error);}
        {std::shared_ptr<TensorBase> root(sameOwner.materialize(original));EXPECT_THROW(fixture->requirePayloadRetired(),std::logic_error);}
        EXPECT_NO_THROW(fixture->requirePayloadRetired());
        Fixture::retire(*fixture);EXPECT_TRUE(fixture->retirementReceipts().empty()); // Recorded physical reset cannot mint an actual native TransferEngine receipt.
        std::size_t exact=0;for(const auto& row:original->rankAttestation())exact+=row.preparation_materialized_bytes;EXPECT_EQ(exact,0u);
        EXPECT_THROW(fixture->permit(device),std::logic_error);
        EXPECT_NO_THROW(fixture->requireExactRetired());
        EXPECT_THROW(fixture->requirePayloadRetired(),std::logic_error);
    }
}
}
