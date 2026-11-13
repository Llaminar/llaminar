/** @file Test__PooledCrossOriginalProgramPlan.cpp
 * @brief Actual program admission envelope and additive native obligations on recorded original source PMA.
 */
#include "execution/qwen4exp/PooledImmutableLocalLookupGraph.h"
#include "execution/qwen4exp/PooledImmutableParticipantReadGraph.h"
#include "execution/qwen4exp/PooledImmutableCompleteCohort.h"
#include "execution/qwen4exp/PooledImmutableParticipantRetirementGraph.h"
#include "integration/kernels/qwen4exp/PooledACKReadyCancellation.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "tensors/NativeTensorExtent.h"
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "backends/NativeOrderingEventGroup.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "transfer/TransferEngine.h"
#include "utils/EmbeddingVerifierFormats.h"
#include "utils/PooledNgramReplayTrace.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <map>
#include <type_traits>
#include <unistd.h>

#include "fixtures/backends/PreparedNativeLookupFixtureReplayAccess.h"
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
namespace llaminar2::test {namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=n::p;
#include "integration/kernels/qwen4exp/PooledImmutableProviderFile.inc"
#include "integration/kernels/qwen4exp/PooledCrossVendorReadReplayGeometry.inc"
/** @brief Actual original mapped worker object and stack; no device API is invoked. */
class CrossPlanWorker final:public testing::MockWorkerGPUContext {
public:
    CrossPlanWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(ordinal){thread_.start(std::move(root),[](void*){},nullptr);}
    ~CrossPlanWorker()override{thread_.join();}
private:PreparedGPUWorkerThread thread_;
};
TEST(PooledCrossOriginalProgramPlan, SourceAndStorageRolesPriceBothChannelsBeforeOriginalAdmission){
    using Source=NativeContextPreparationReplayAccess;using Fixture=PreparedNativeLookupFixtureReplayAccess;
    for(const bool reverse:{false,true}){
        const auto sourceDevice=reverse?DeviceId::rocm(46):DeviceId::cuda(46),ownerDevice=reverse?DeviceId::cuda(46):DeviceId::rocm(46);
        const std::array devices{sourceDevice,ownerDevice};testing::installHardwareFreeGPUContextFactories();
        for(const auto device:devices)Source::seed(device,[device](auto root){return Source::construct<CrossPlanWorker>(std::move(root),device.ordinal);});
        const std::array observations{PhysicalMemoryResource{0,DeviceId::cpu(),1ull<<34,1ull<<34},PhysicalMemoryResource{0,sourceDevice,1ull<<30,1ull<<30},PhysicalMemoryResource{0,ownerDevice,1ull<<30,1ull<<30}};
        auto source=Source::source<CrossPlanWorker>(devices,observations);const auto original=source->memory();
        auto session=Fixture::prepare(source,devices);const auto ordering=NativeOrderingEventGroupReplayAccess::projection(devices);
        const auto io=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ImmutableLookupIO,4096);
        const std::array<std::uint64_t,2> rows{7,101};auto file=std::make_shared<ProviderFile>(4096,rows);
        const std::array specs{CrossVendorReadReplaySpec{file,0,64,16,0},CrossVendorReadReplaySpec{std::make_shared<ProviderFile>(),3,128,32,2}};
        std::vector<CrossVendorReadReplayGeometry> declarations;for(const auto& spec:specs)declarations.push_back(CrossVendorReadReplayGeometry::create(spec,sourceDevice,ownerDevice,*session,io,ordering));
        EXPECT_THROW(crossVendorReadProgramBOM(std::span<const CrossVendorReadReplayGeometry>{}),std::invalid_argument);
        EXPECT_NE(specs[0].file->descriptor().path,specs[1].file->descriptor().path);
        EXPECT_NE(specs[0].file->rows,specs[1].file->rows); // Dense-width and sparse Slot64 programs share ONE original PMA.
        // Same source/remote owners retain an actual RAM source with no redundant row cache.
        auto ramSpec=specs[0];ramSpec.sourceBacking=AdaptiveImmutableRowBacking::RAM;ramSpec.slots=3;
        const auto ramGeometry=CrossVendorReadReplayGeometry::create(ramSpec,sourceDevice,ownerDevice,*session,io,ordering);
        EXPECT_EQ(ramGeometry.cpuGeometry.sourceStorage(),ImmutableRowStorage::RAM);
        EXPECT_EQ(ramGeometry.cpuGeometry.storageBytes(),0u);
        EXPECT_EQ(ramGeometry.payload.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ImmutableTableStorage),ramGeometry.cpuGeometry.sourceBytes());
        EXPECT_EQ(ramGeometry.cpuGeometry.sourceBytes(),ImmutableRowTable::physicalBytes(file->descriptor(),providerReplaySourcePolicy(ramSpec.sourceBacking)));
        ramSpec.ramSlots=1;EXPECT_THROW(ramSpec.requireComplete(),std::runtime_error);
        const auto bom=crossVendorReadProgramBOM(declarations);const auto* compute=bom.find({0,sourceDevice});const auto* storage=bom.find({0,ownerDevice});ASSERT_NE(compute,nullptr);ASSERT_NE(storage,nullptr);
        EXPECT_EQ(compute->bytes(PhysicalMemoryOwner::NativeGraphExecutable),6*GPUGraphMemoryContract::reservationBytesPerExecutable(sourceDevice,GPUGraphExecutableClass::General));
        EXPECT_EQ(storage->bytes(PhysicalMemoryOwner::NativeGraphExecutable),2*GPUGraphMemoryContract::reservationBytesPerExecutable(ownerDevice,GPUGraphExecutableClass::General));
        const auto sourceGeneral=RetainedParticipantGraphExecutable::generalBOM(session->resource(DeviceId::cpu()),session->resource(sourceDevice),6);
        const auto ownerGeneral=RetainedParticipantGraphExecutable::generalBOM(session->resource(DeviceId::cpu()),session->resource(ownerDevice),2);
        EXPECT_EQ(bom.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::NativeGraphExecutable),
            sourceGeneral.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::NativeGraphExecutable)+ownerGeneral.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::NativeGraphExecutable));
        EXPECT_EQ(declarations.back().physical.owners().size(),1u);EXPECT_EQ(declarations.back().physical.owners()[0].allocator.device,ownerDevice);
        EXPECT_EQ(declarations.back().policy.coordinator().device,sourceDevice);
        const auto channel=q::PooledImmutableDirectedTransport::completeBOM(declarations.back().physical,declarations.back().policy,session->resource(DeviceId::cpu()),std::array{session->resource(sourceDevice),session->resource(ownerDevice)});
        for(const auto device:devices){const auto* actual=bom.find({0,device});const auto* wanted=channel.find({0,device});ASSERT_NE(wanted,nullptr);EXPECT_GE(actual->bytes(PhysicalMemoryOwner::ActivationTransportStaging),wanted->bytes(PhysicalMemoryOwner::ActivationTransportStaging));}
        session->admit(bom);EXPECT_EQ(session->memory(),original);EXPECT_EQ(original->phase(),PhysicalMemoryAuthorityPhase::Admitted);
        for(const auto device:devices){const auto current=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,original);current.requireWorker(session->permit(device).worker());EXPECT_EQ(current.generation(),session->permit(device).generation());}
        Fixture::retire(*session);
    }
}
/** @test Sparse backing cannot invent terminal reference bytes for unauthored rows. */
TEST(PooledCrossOriginalProgramPlan,SparseOracleRejectsUnauthoredRowsBeforeNativeInput){
    ProviderFile file;const std::array valid{std::uint64_t{7},ProviderFile::High};
    EXPECT_NO_THROW(file.requireAuthored(valid));
    const std::array invalid{std::uint64_t{1},ProviderFile::High};
    EXPECT_THROW(file.requireAuthored(invalid),std::invalid_argument);
    EXPECT_NO_THROW(file.requireAuthored(std::span<const std::uint64_t>{}));
    EXPECT_EQ(file.original(7).size(),file.row_bytes);EXPECT_EQ(file.original(ProviderFile::High).size(),file.row_bytes);
}
}}
