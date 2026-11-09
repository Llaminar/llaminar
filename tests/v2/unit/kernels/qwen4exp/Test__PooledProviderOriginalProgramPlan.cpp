/** @file Test__PooledProviderOriginalProgramPlan.cpp
 * @brief Actual program admission envelope and additive native obligations on recorded original source PMA.
 */
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
#include "execution/qwen4exp/PooledImmutableLocalLookupGraph.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLEBuilder.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "memory/CPUMappedPublicationRoot.h"
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
namespace llaminar2::test {namespace {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace p=n::p;
#include "integration/kernels/qwen4exp/PooledImmutableProviderReplay.inc"
/** @brief Actual source stack/object owner with no accelerator API in its recorded constructor. */
class ProgramPlanWorker final:public testing::MockWorkerGPUContext {
public:
    ProgramPlanWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(ordinal){thread_.start(std::move(root),[](void*){},nullptr);}
    ~ProgramPlanWorker()override{thread_.join();}
private:PreparedGPUWorkerThread thread_;
};
/** @test Serialized payload is shared while each actual graph/event program retains its own named commitment. */
// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
TEST(PooledProviderOriginalProgramPlan, PayloadEnvelopeAndAdditiveNativeProgramsUseOneOriginalAuthority){
    using Source=NativeContextPreparationReplayAccess;using Fixture=PreparedNativeLookupFixtureReplayAccess;
    for(const auto device:{DeviceId::cuda(47),DeviceId::rocm(47)}){
        testing::installHardwareFreeGPUContextFactories();
        Source::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> root){return Source::construct<ProgramPlanWorker>(std::move(root),device.gpu_ordinal());});
        const std::array observations{PhysicalMemoryResource{0,DeviceId::cpu(),16ULL<<30,16ULL<<30},PhysicalMemoryResource{0,device,1u<<30,1u<<30}};
        auto source=Source::source<ProgramPlanWorker>(device,observations);const auto memory=source->memory();const std::array devices{device};
        auto session=Fixture::prepare(source,devices);const auto projection=NativeOrderingEventGroupReplayAccess::projection(devices);
        const std::array<std::uint64_t,2> rows{7,101};auto file=std::make_shared<ProviderFile>(4096,rows);
        const std::array specs{ProviderReplaySpec{file,32,0,0,n::OriginalIdWidth::UInt64},ProviderReplaySpec{file,64,4,2,n::OriginalIdWidth::UInt64},
            ProviderReplaySpec{file,8192,4,2,n::OriginalIdWidth::UInt64,true}};
        EXPECT_THROW(ProviderReplayPrograms::programBOM(std::span<const ProviderReplayGeometry>{}),std::invalid_argument);
        auto malformed=specs.back();malformed.maximum=33;
        EXPECT_THROW(ProviderReplayPrograms(std::vector{malformed}),std::runtime_error);
        std::vector<ProviderReplayGeometry> geometries;for(const auto& spec:specs)geometries.push_back(ProviderReplayGeometry::create(spec,device,*session,RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ImmutableLookupIO),projection));
        // Full RAM changes the original source root, never the bounded host cache.
        auto ramSpec=specs[1];ramSpec.sourceBacking=AdaptiveImmutableRowBacking::RAM;ramSpec.hostSlots=0;
        const auto ramGeometry=ProviderReplayGeometry::create(ramSpec,device,*session,
            RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ImmutableLookupIO),projection);
        EXPECT_EQ(ramGeometry.cpu.sourceStorage(),ImmutableRowStorage::RAM);
        EXPECT_EQ(ramGeometry.cpu.storageBytes(),0u);
        EXPECT_EQ(ramGeometry.payload.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ImmutableTableStorage),ramGeometry.cpu.sourceBytes());
        EXPECT_EQ(ramGeometry.cpu.sourceBytes(),ImmutableRowTable::physicalBytes(file->descriptor(),providerReplaySourcePolicy(ramSpec.sourceBacking)));
        ramSpec.hostSlots=1;EXPECT_THROW(ramSpec.requireComplete(),std::runtime_error);
        const auto full=ProviderReplayPrograms::programBOM(geometries);const auto* gpu=full.find({0,device});const auto* host=full.find({0,DeviceId::cpu()});
        ASSERT_NE(gpu,nullptr);ASSERT_NE(host,nullptr);
        EXPECT_EQ(gpu->bytes(PhysicalMemoryOwner::NativeGraphExecutable),10*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General));
        const auto general=RetainedParticipantGraphExecutable::generalBOM(session->resource(DeviceId::cpu()),session->resource(device),10);
        EXPECT_EQ(host->bytes(PhysicalMemoryOwner::NativeGraphExecutable),general.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::NativeGraphExecutable));
        // Recorded closed rows: C0 is 3+3+3+1; each positive case is4+4+3+1.
        EXPECT_EQ(host->bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),38912u);
        EXPECT_EQ(gpu->bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),77824u);
        const auto* largest=geometries.back().payload.find({0,device});ASSERT_NE(largest,nullptr);
        EXPECT_EQ(gpu->bytes(PhysicalMemoryOwner::ActivationArena),largest->bytes(PhysicalMemoryOwner::ActivationArena));
        EXPECT_EQ(gpu->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),largest->bytes(PhysicalMemoryOwner::ImmutableLookupStaging));
        session->admit(full);EXPECT_EQ(session->memory(),memory);EXPECT_EQ(memory->phase(),PhysicalMemoryAuthorityPhase::Admitted);
        EXPECT_EQ(memory->admission()->plan().requiredFootprintMismatch(PhysicalMemoryPlanBuilder(source->mandatoryBOM()).addMutuallyExclusive(std::array{full}).build()),std::nullopt);
        Fixture::retire(*session);
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)
}}
