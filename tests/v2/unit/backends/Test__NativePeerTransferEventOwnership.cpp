/**
 * @file Test__NativePeerTransferEventOwnership.cpp
 * @brief Original-source peer event flags, admission and partial-creation retirement replay.
 *
 * Recorded workers call the actual closed ownership implementation and canonical
 * PMA. Fake handles exercise lifetime/error ownership only; they do not certify
 * CUDA/HIP flags or native physical ceilings. The original native peer cases
 * separately prove completion, real elapsed time and exact transfer bytes.
 */
#include "backends/NativePeerTransferEventOwnership.h"
#include "execution/moe/ExpertTierGpuPeerTransferLane.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
namespace llaminar2::test {
/** @brief Only native SDK handles are substituted; source/PMA and owner lifecycle remain actual. */
struct NativePeerTransferEventOwnershipReplayAccess {
    static auto materialize(const PreparedGPUContextUsePermit& original,bool timing,const std::string& instance)
    { return NativePeerTransferEventOwnership::materialize(original,timing,instance); }
};
namespace {
/** @brief Real original prepared worker/stack with recorded native event attempts. */
class PeerEventWorker final:public testing::MockWorkerGPUContext {
public:
    /** @brief Bind the exact source-prepared worker stack before its first creator. */
    PeerEventWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(ordinal)
    { thread_.start(std::move(root),[](void*){},nullptr); }
    ~PeerEventWorker()override { thread_.join(); }
    /** @brief Record exact original purpose and one selected partial creator rejection. */
    void* createEvent(GPUEventPurpose purpose=GPUEventPurpose::Ordering)override {
        purposes.push_back(purpose);++attempts;
        if(failAt && attempts==failAt)return nullptr;
        return MockWorkerGPUContext::createEvent(purpose);
    }
    /** @brief Observe accepted-handle destruction before the original commitment closes. */
    void destroyEvent(void* event)override { EXPECT_NE(event,nullptr);++destroyed;MockWorkerGPUContext::destroyEvent(event); }
    std::vector<GPUEventPurpose> purposes;
    int attempts=0,destroyed=0,failAt=0;
private:
    PreparedGPUWorkerThread thread_;
};
}
/** @test Exact flag family and accepted handles remain paired through success and every failed creator. */
TEST(NativePeerTransferEvents, OriginalFlagsAndPartialCreationReplay)
{
    using Access=NativeContextPreparationReplayAccess;
    const std::array devices{DeviceId::cuda(61),DeviceId::rocm(61)};
    testing::installHardwareFreeGPUContextFactories();
    for(const auto device:devices) {
        Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root) {
            return Access::construct<PeerEventWorker>(std::move(root),device.gpu_ordinal());
        });
    }
    // One process has one original source; both recorded backend workers must
    // exist before that source closes its mandatory declaration for planning.
    const std::array observed{
        PhysicalMemoryResource{0,DeviceId::cpu(),256u<<20,256u<<20},
        PhysicalMemoryResource{0,devices[0],64u<<20,64u<<20},
        PhysicalMemoryResource{0,devices[1],64u<<20,64u<<20}};
    auto source=Access::source<PeerEventWorker>(devices,observed);
    auto memory=source->memory();
    /** @brief Retire every published recorded context even when setup or an assertion exits early. */
    struct RetirementGuard {
        const std::array<DeviceId,2>& devices;
        std::array<bool,2> live{};
        /** @brief All inner handle/permit scopes unwind before exact original context retirement. */
        ~RetirementGuard() {
            for(std::size_t i=0;i<devices.size();++i)if(live[i]) {
                auto retirement=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(devices[i]);
                Access::physicalReset(retirement);
            }
        }
    } retirementGuard{devices};
    {
        std::vector<PreparedGPUContextUsePermit> preparing;
        for(const auto device:devices) {
            preparing.push_back(source->acquire(device));
            retirementGuard.live[device==devices[0]?0:1]=true;
            auto worker=std::dynamic_pointer_cast<PeerEventWorker>(preparing.back().worker());
            ASSERT_NE(worker,nullptr);
            EXPECT_THROW(NativePeerTransferEventOwnershipReplayAccess::materialize(preparing.back(),true,"preparing"),std::logic_error);
            EXPECT_EQ(worker->attempts,0);
        }
        source->freezeForPlanning();
        PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
        for(const auto device:devices) {
            const auto host=source->originalResource(DeviceId::cpu());
            const auto gpu=source->originalResource(device);
            const auto timed=ExpertTierGpuPeerTransferLane::nativeEventsBOM(host,gpu,true);
            const auto untimed=ExpertTierGpuPeerTransferLane::nativeEventsBOM(host,gpu,false);
            for(int instance=0;instance<4;++instance)for(const auto& row:timed.resources())plan.add(row);
            for(const auto& row:untimed.resources())plan.add(row);
        }
        source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
    }
    for(const auto device:devices) {
        std::size_t retainedHost=0,retainedGPU=0;
        {
            auto original=source->acquire(device);
            auto worker=std::dynamic_pointer_cast<PeerEventWorker>(original.worker());
            if(!worker)throw std::logic_error("Peer event replay lost original recorded worker");
            const auto gpu=source->originalResource(device);
            for(int failedCreator=0;failedCreator<=3;++failedCreator) {
                SCOPED_TRACE(::testing::Message()<<device.toString()<<" failedCreator="<<failedCreator);
                const auto attempts=worker->attempts,destroyed=worker->destroyed;
                worker->failAt=failedCreator?attempts+failedCreator:0;
                const auto instance="actual_lane_"+std::to_string(failedCreator);
                if(failedCreator) {
                    EXPECT_THROW(NativePeerTransferEventOwnershipReplayAccess::materialize(original,true,instance),std::runtime_error);
                    EXPECT_EQ(worker->attempts-attempts,failedCreator);
                    EXPECT_EQ(worker->destroyed-destroyed,failedCreator-1);
                } else {
                    auto owner=NativePeerTransferEventOwnershipReplayAccess::materialize(original,true,instance);
                    EXPECT_EQ(worker->attempts-attempts,3);EXPECT_EQ(worker->destroyed,destroyed);
                    const auto activeCharge=memory->reservedBytes(device,PhysicalMemoryOwner::NativeTransferEvent);
                    EXPECT_THROW(NativePeerTransferEventOwnershipReplayAccess::materialize(original,true,instance),std::logic_error);
                    EXPECT_EQ(worker->attempts-attempts,3);
                    EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeTransferEvent),activeCharge);
                    owner.reset();EXPECT_EQ(worker->destroyed-destroyed,3);
                    const auto charged=memory->reservedBytes(device,PhysicalMemoryOwner::NativeTransferEvent);
                    EXPECT_THROW(NativePeerTransferEventOwnershipReplayAccess::materialize(original,true,instance),std::logic_error);
                    EXPECT_EQ(worker->attempts-attempts,3);
                    EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeTransferEvent),charged);
                    EXPECT_THROW(NativePeerTransferEventOwnershipReplayAccess::materialize(original,false,instance),std::logic_error);
                    EXPECT_EQ(worker->attempts-attempts,3);
                }
                EXPECT_EQ(worker->purposes[attempts],GPUEventPurpose::Ordering);
                for(int i=1;i<worker->attempts-attempts;++i)EXPECT_EQ(worker->purposes[attempts+i],GPUEventPurpose::Timing);
            }
            worker->failAt=0;const auto attempts=worker->attempts,destroyed=worker->destroyed;
            {auto owner=NativePeerTransferEventOwnershipReplayAccess::materialize(original,false,"untimed_lane");
                EXPECT_EQ(worker->attempts-attempts,1);EXPECT_EQ(worker->purposes.back(),GPUEventPurpose::Ordering);}
            EXPECT_EQ(worker->destroyed-destroyed,1);
            // Physical runtime commitments survive successful handle teardown;
            // an unpriced further population cannot create another handle.
            EXPECT_THROW(NativePeerTransferEventOwnershipReplayAccess::materialize(original,true,"unpriced_lane"),std::logic_error);
            EXPECT_EQ(worker->attempts-attempts,1);
            EXPECT_THROW((ExpertTierGpuPeerTransferLane::nativeEventsBOM({1,DeviceId::cpu(),1,1},gpu,true)),std::invalid_argument);
            EXPECT_NE(NativeAuxiliaryResourceProgram::peerTransferEvents(true),NativeAuxiliaryResourceProgram::peerTransferEvents(false));
            EXPECT_FALSE(NativeAuxiliaryResourceProgram::peerTransferEvents(true).isOrdering());
            EXPECT_THROW(NativePeerTransferEventOwnershipReplayAccess::materialize(original,false,""),std::invalid_argument);
            retainedHost=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeTransferEvent);
            retainedGPU=memory->reservedBytes(device,PhysicalMemoryOwner::NativeTransferEvent);
            EXPECT_GT(retainedHost,0u);EXPECT_GT(retainedGPU,0u);
        }
        {auto retirement=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(retirement);}
        retirementGuard.live[device==devices[0]?0:1]=false;
        // Handle and genuine context retirement do not claim process-pool
        // reclamation; the original canonical PMA keeps exactly these tokens.
        EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeTransferEvent),retainedHost);
        EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeTransferEvent),retainedGPU);
    }
}
}
