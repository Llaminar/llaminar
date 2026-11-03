/**
 * @file FiniteBodySiblingNativeReplay.inc
 * @brief Original same-worker native main/storage graph composition and terminal replay.
 *
 * The fixture owns ordinary admitted General definitions and real byte copies.
 * It proves the native DAG and lifecycle primitive independently of the remote
 * source protocol. It mints no ModelContext witness, SourceSeal or fake ACK and
 * does not certify a full colocated model. That proof remains a canonical cell.
 */
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/GPUGraphMemoryContract.h"
namespace llaminar2::test {
namespace {
/** @brief Import the same original finite definition on CUDA progress and ROCm native SDMA. */
void replayNativeFiniteBody(DeviceId device,std::size_t liveBytes){
    constexpr std::size_t capacity=8193,header=sizeof(std::uint64_t),stride=header+capacity;
    ASSERT_LE(liveBytes,capacity);
    const std::array devices{device};auto session=PreparedNativeLookupFixture::prepare(devices);
    const auto& original=session->permit(device);auto worker=original.worker();auto* backend=getBackendFor(device);
    const auto general=RetainedParticipantGraphExecutable::generalBOM(session->resource(DeviceId::cpu()),session->resource(device),3);
    const auto mappedBytes=device.is_cuda()?TransferEngine::mappedHostRegionAllocationBytes(sizeof(MappedTransferProgressCommand)+sizeof(MappedTransferProgressCompletion))+
        TransferEngine::mappedHostRegionAllocationBytes(sizeof(std::uint64_t)):0;
    const auto gpuBytes=3*stride+(device.is_cuda()?sizeof(MappedTransferServiceCursor):0);
    PhysicalMemoryPlanBuilder bom(general);bom.add(session->resource(device),PhysicalMemoryOwner::ActivationArena,gpuBytes);
    if(mappedBytes)bom.add(session->resource(DeviceId::cpu()),PhysicalMemoryOwner::ActivationArena,mappedBytes);
    session->admit(bom.build());
    {
        auto gpuClaim=session->memory()->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,gpuBytes);
        std::optional<PhysicalMemoryAllocationLease> hostClaim;
        if(mappedBytes)hostClaim.emplace(session->memory()->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,mappedBytes));
        TransferEngine transfers;auto source=transfers.allocateDeviceTransferBuffer(stride,device);
        auto output=transfers.allocateDeviceTransferBuffer(2*stride,device);
        std::shared_ptr<MappedTransferProgressEpoch> epoch;
        if(device.is_cuda())epoch=MappedTransferProgressEpoch::create({.device=device,.slot_capacity=1,.execution_lane_capacity=1,
            .execution_streams=transfers.allocatePersistentTransferExecutionLanes(1,device,"finite_storage_native"),
            .maximum_bytes=capacity,.name="finite_storage_native",.perf_device=device.toString()});
        const auto factory=epoch?epoch->graphBranchFactory():GraphCaptureAuxiliaryBranchFactory{};
        auto progress=factory.valid()?factory.create():nullptr;
        if(device.is_cuda()){ASSERT_TRUE(factory.valid());ASSERT_TRUE(progress);}
        else{ASSERT_TRUE(factory.empty());ASSERT_FALSE(progress);}
        auto parent=RetainedParticipantGraphExecutableFixtureAccess::original(original,worker->defaultStream());
        auto storage=RetainedParticipantGraphExecutableFixtureAccess::original(original,worker->defaultStream());
        auto executable=RetainedParticipantGraphExecutableFixtureAccess::original(original,worker->defaultStream());
        std::vector<std::byte> input(stride),actual(2*stride),poison(2*stride,std::byte{0xa5});
        const auto count=static_cast<std::uint64_t>(liveBytes);std::memcpy(input.data(),&count,header);
        for(std::size_t i=0;i<capacity;++i)input[header+i]=std::byte((i*43u+17u)&255u);
        bool complete=false;
        const auto join=[&]{worker->submitAndWait([&]{if(!backend->synchronizeStream(worker->defaultStream(),device.ordinal))throw std::runtime_error("Finite-body diagnostic retirement failed");});};
        struct Cleanup{std::function<void()> join;bool& complete;~Cleanup(){if(!complete)try{join();}catch(...){std::terminate();}}} cleanup{join,complete};
        worker->submitAndWait([&]{
            const auto stream=worker->defaultStream();
            ASSERT_TRUE(backend->hostToDeviceOnStream(source->mutableDeviceData(),input.data(),input.size(),device.ordinal,stream));
            const auto record=[&](const auto& owner,std::size_t destination){
                auto& graph=RetainedParticipantGraphExecutableFixtureAccess::capture(owner);
                ScopedBackendGraphCapture capture(*worker,graph,"original finite native body");ASSERT_TRUE(capture.begin());
                // Capture only the original asynchronous live copy. Completion
                // belongs to the retained parent's terminal event, not the
                // blocking compatibility copy API's external receipt.
                ASSERT_TRUE(backend->deviceCopyAsync(output->mutableDeviceData(destination),source->deviceData(),header,device.ordinal,stream));
                if(liveBytes)ASSERT_TRUE(backend->deviceCopyAsync(output->mutableDeviceData(destination+header),source->deviceData(header),liveBytes,device.ordinal,stream));
                capture.finish();
            };
            record(parent,0);
            const auto emptyBefore=RetainedParticipantGraphExecutableFixtureAccess::capture(parent).nodeCount();
            const std::array emptyBody{storage};
            EXPECT_FALSE(RetainedParticipantGraphExecutableFixtureAccess::composeRecordedNativeFixtureBodies(parent,emptyBody,progress.get()));
            EXPECT_EQ(RetainedParticipantGraphExecutableFixtureAccess::capture(parent).nodeCount(),emptyBefore);
            record(storage,stride);record(executable,stride);
            ASSERT_TRUE(RetainedParticipantGraphExecutableFixtureAccess::capture(executable).instantiate());
            const auto before=RetainedParticipantGraphExecutableFixtureAccess::capture(parent).nodeCount();
            using Owner=std::shared_ptr<RetainedParticipantGraphExecutable>;
            for(const auto invalid:{std::vector<Owner>{parent},std::vector<Owner>{storage,storage},std::vector<Owner>{executable},std::vector<Owner>{nullptr}}){
                EXPECT_FALSE(RetainedParticipantGraphExecutableFixtureAccess::composeRecordedNativeFixtureBodies(parent,invalid,progress.get()));
                EXPECT_EQ(RetainedParticipantGraphExecutableFixtureAccess::capture(parent).nodeCount(),before);
            }
            // Empty progress-free import is invalid. An actual progress branch's
            // empty siblings mean its original ordinary attach, so do not call it
            // as an invalid control and consume that real one-shot branch.
            if(!progress)EXPECT_FALSE(RetainedParticipantGraphExecutableFixtureAccess::composeRecordedNativeFixtureBodies(parent,{},nullptr));
            const std::array bodies{storage};
            ASSERT_TRUE(RetainedParticipantGraphExecutableFixtureAccess::composeRecordedNativeFixtureBodies(parent,bodies,progress.get()));
            EXPECT_EQ(RetainedParticipantGraphExecutableFixtureAccess::capture(parent).nodeCount(),before+(progress?5u:1u));
            EXPECT_FALSE(RetainedParticipantGraphExecutableFixtureAccess::capture(storage).hasExecutable());
            ASSERT_TRUE(RetainedParticipantGraphExecutableFixtureAccess::capture(parent).instantiate());
            EXPECT_FALSE(RetainedParticipantGraphExecutableFixtureAccess::capture(storage).hasExecutable());
        });
        join();
        for(std::uint64_t invocation=1;invocation<=20;++invocation){
            worker->submitAndWait([&]{
                ASSERT_TRUE(backend->hostToDeviceOnStream(output->mutableDeviceData(),poison.data(),poison.size(),device.ordinal,worker->defaultStream()));
                ASSERT_TRUE(RetainedParticipantGraphExecutableFixtureAccess::submit(parent,invocation));
            });
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(parent)){
                ASSERT_LT(std::chrono::steady_clock::now(),deadline);std::this_thread::yield();
            }
            worker->submitAndWait([&]{ASSERT_TRUE(backend->deviceToHostOnStream(actual.data(),output->deviceData(),actual.size(),device.ordinal,worker->defaultStream()));});join();
            for(const auto offset:{std::size_t{0},stride}){
                EXPECT_EQ(std::memcmp(actual.data()+offset,input.data(),header+liveBytes),0);
                EXPECT_TRUE(std::all_of(actual.begin()+offset+header+liveBytes,actual.begin()+offset+stride,[](std::byte value){return value==std::byte{0xa5};}));
            }
        }
        // Original parent native handle goes first; progress fragments and the
        // immutable storage definition outlive every cloned execution.
        parent.reset();progress.reset();storage.reset();executable.reset();epoch.reset();join();complete=true;
    }
    session->requirePayloadRetired();worker.reset();session->retire();session->requireExactRetired();
}
/** @brief Actual foreign-device native handles reject before changing the original parent DAG. */
void rejectForeignNativeFiniteBody(DeviceId device,DeviceId foreign){
    const std::array devices{device,foreign};auto session=PreparedNativeLookupFixture::prepare(devices);
    PhysicalMemoryPlanBuilder bom;
    for(const auto selected:devices){
        const auto graph=RetainedParticipantGraphExecutable::generalBOM(session->resource(DeviceId::cpu()),session->resource(selected),1);
        for(const auto& row:graph.resources())bom.add(row);
        bom.add(session->resource(selected),PhysicalMemoryOwner::ActivationArena,sizeof(std::uint64_t));
    }
    session->admit(bom.build());
    {
        std::array<std::optional<PhysicalMemoryAllocationLease>,2> claims;
        std::array<std::shared_ptr<DeviceTransferBuffer>,2> bytes;
        std::array<std::shared_ptr<RetainedParticipantGraphExecutable>,2> graphs;
        TransferEngine transfers;
        for(std::size_t i=0;i<devices.size();++i){
            const auto selected=devices[i];const auto& original=session->permit(selected);
            claims[i].emplace(session->memory()->claimNewAllocation(selected,PhysicalMemoryOwner::ActivationArena,sizeof(std::uint64_t)));
            bytes[i]=transfers.allocateDeviceTransferBuffer(sizeof(std::uint64_t),selected);
            graphs[i]=RetainedParticipantGraphExecutableFixtureAccess::original(original,original.worker()->defaultStream());
            original.worker()->submitAndWait([&,i,selected]{
                auto& capture=RetainedParticipantGraphExecutableFixtureAccess::capture(graphs[i]);
                ScopedBackendGraphCapture record(*session->permit(selected).worker(),capture,"foreign finite-body input");ASSERT_TRUE(record.begin());
                ASSERT_TRUE(getBackendFor(selected)->memset(bytes[i]->mutableDeviceData(),0,sizeof(std::uint64_t),selected.ordinal,session->permit(selected).worker()->defaultStream()));
                record.finish();
            });
        }
        session->permit(device).worker()->submitAndWait([&]{
            const auto before=RetainedParticipantGraphExecutableFixtureAccess::capture(graphs[0]).nodeCount();
            const std::array sibling{graphs[1]};
            EXPECT_FALSE(RetainedParticipantGraphExecutableFixtureAccess::composeRecordedNativeFixtureBodies(graphs[0],sibling,nullptr));
            EXPECT_EQ(RetainedParticipantGraphExecutableFixtureAccess::capture(graphs[0]).nodeCount(),before);
        });
        graphs={};bytes={};claims={};
    }
    session->requirePayloadRetired();session->retire();session->requireExactRetired();
}

}
#ifdef HAVE_CUDA
TEST(FiniteBodySiblingNativeIntegration, CUDA_RejectsOriginalForeignDevice){rejectForeignNativeFiniteBody(DeviceId::cuda(0),DeviceId::cuda(1));}
TEST(FiniteBodySiblingNativeIntegration, CUDA_Live0){replayNativeFiniteBody(DeviceId::cuda(0),0);}
TEST(FiniteBodySiblingNativeIntegration, CUDA_Live1){replayNativeFiniteBody(DeviceId::cuda(0),1);}
TEST(FiniteBodySiblingNativeIntegration, CUDA_Live257){replayNativeFiniteBody(DeviceId::cuda(0),257);}
TEST(FiniteBodySiblingNativeIntegration, CUDA_Live8193){replayNativeFiniteBody(DeviceId::cuda(0),8193);}
#endif
#ifdef HAVE_ROCM
TEST(FiniteBodySiblingNativeIntegration, ROCm_RejectsOriginalForeignDevice){rejectForeignNativeFiniteBody(DeviceId::rocm(0),DeviceId::rocm(1));}
TEST(FiniteBodySiblingNativeIntegration, ROCm_Live0){replayNativeFiniteBody(DeviceId::rocm(0),0);}
TEST(FiniteBodySiblingNativeIntegration, ROCm_Live1){replayNativeFiniteBody(DeviceId::rocm(0),1);}
TEST(FiniteBodySiblingNativeIntegration, ROCm_Live257){replayNativeFiniteBody(DeviceId::rocm(0),257);}
TEST(FiniteBodySiblingNativeIntegration, ROCm_Live8193){replayNativeFiniteBody(DeviceId::rocm(0),8193);}
#endif
} // namespace llaminar2::test
