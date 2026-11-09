/** @file Test__PreparedNativeLookupFixture.cpp
 * @brief Device-free actual source acquisition/freeze/admission with original mapped worker and payload roots.
 */
#include "../../utils/HostNUMANodes.h"
#include "fixtures/backends/PreparedNativeLookupFixtureReplayAccess.h"
#include "fixtures/backends/NativeLookupCaseAdmission.h"
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
    if(!hasCPUBackend())initCPUBackend(llaminar2::test::hostLastNUMANode());
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
/** @test Serialized exact payloads share their maximum while all original retained instances remain charged. */
TEST(PreparedNativeLookupFixture, SerialLookupCasesPreserveRetainedPopulationAndExactMaximum)
{
    const PhysicalMemoryResource host{0,DeviceId::cpu(),1u<<20,1u<<20};
    const PhysicalMemoryResource gpu{0,DeviceId::cuda(49),1u<<20,1u<<20};
    std::vector<PhysicalMemoryPlan> cases;
    for(const auto count:{1u,4u,2u}){
        PhysicalMemoryPlanBuilder original;
        original.add(host,PhysicalMemoryOwner::ImmutableTableStorage,count*100);
        original.add(gpu,PhysicalMemoryOwner::ImmutableLookupStaging,count*32);
        original.add(host,PhysicalMemoryOwner::RetainedHostServiceStorage,count*64);
        original.add(host,PhysicalMemoryOwner::RetainedHostServiceRuntime,count*16);
        original.add(host,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,count*8);
        original.add(gpu,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,count*4);
        cases.push_back(original.build());
    }
    const auto envelope=serialNativeLookupCaseEnvelope(cases);
    const auto* h=envelope.find({0,DeviceId::cpu()});const auto* g=envelope.find({0,DeviceId::cuda(49)});
    ASSERT_NE(h,nullptr);ASSERT_NE(g,nullptr);
    EXPECT_EQ(h->bytes(PhysicalMemoryOwner::ImmutableTableStorage),400u);
    EXPECT_EQ(g->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),128u);
    EXPECT_EQ(h->bytes(PhysicalMemoryOwner::RetainedHostServiceStorage),256u);
    EXPECT_EQ(h->bytes(PhysicalMemoryOwner::RetainedHostServiceRuntime),112u);
    EXPECT_EQ(h->bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),56u);
    EXPECT_EQ(g->bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),28u);
    EXPECT_THROW(serialNativeLookupCaseEnvelope({}),std::invalid_argument);
    auto foreign=host;--foreign.total_bytes;foreign.admission_available_bytes=foreign.total_bytes;
    PhysicalMemoryPlanBuilder wrong;wrong.add(foreign,PhysicalMemoryOwner::ImmutableTableStorage,1);
    auto changed=cases;changed.push_back(wrong.build());
    EXPECT_THROW(serialNativeLookupCaseEnvelope(changed),std::invalid_argument);
    // Replaying declaration composition cannot consume or mutate any original case.
    for(int reset=0;reset<20;++reset)EXPECT_EQ(serialNativeLookupCaseEnvelope(cases).summary(),envelope.summary());
}

}
