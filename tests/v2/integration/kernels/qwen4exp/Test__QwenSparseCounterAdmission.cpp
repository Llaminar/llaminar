/**
 * @file Test__QwenSparseCounterAdmission.cpp
 * @brief Native canonical controller borrow, exact request-row identity and malformed-owner rejection.
 *
 * Setup materializes only the original admitted arena allocation. Counter
 * writers remain outside per-stage tensor publication; reset and observation
 * retain the original complete generation-parent lifetime.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */
#include "../../../utils/CPUActivationArenaFixture.h"
#include "execution/qwen4exp/QwenSparseCounterBinding.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "tensors/TensorFactory.h"
#include "transfer/TransferEngine.h"
#include "../../../mocks/MockMPIContext.h"
#include <gtest/gtest.h>
namespace llaminar2::test {
/** @brief Positive admission uses only a real owned native arena bank; wrong geometry/participant cannot bind. */
void proveCounterAdmission(DeviceId device){
    auto* backend=getBackendFor(device);if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP();
    if(!hasCPUBackend())initCPUBackend(-1);
    auto& context=GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]{
        void* stream=context.defaultStream();ASSERT_NE(stream,nullptr);MockMPIContext mpi;TensorFactory factory(mpi);
        BufferArena arena(::llaminar2::test::cpuArenaFixtureConfig(&factory));
        ASSERT_TRUE(arena.registerBuffer(BufferId::MTP_GENERATION_CONTROL,3,sampling_math::kDeviceGenerationControlCount,"INT32",device));
        ASSERT_TRUE(arena.allocate());auto owner=arena.getSharedTensor(BufferId::MTP_GENERATION_CONTROL);ASSERT_TRUE(owner);
        EXPECT_THROW(QwenSparseCounterBinding::admit(arena,device,0,48,16),std::invalid_argument);
        TransferEngine::allocateDeviceStorage(owner.get(),device);
        const auto first=QwenSparseCounterBinding::admit(arena,device,0,48,16);
        const auto second=QwenSparseCounterBinding::admit(arena,device,1,48,16);
        EXPECT_EQ(first.native().control,owner->gpu_data_ptr());
        EXPECT_EQ(second.native().control,first.native().control+sampling_math::kDeviceGenerationControlCount);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.native().control)%8,0u);
        EXPECT_NE(first.captureIdentity(),second.captureIdentity());
        const auto identity=first.captureIdentity();ASSERT_TRUE(backend->memset(owner->gpu_data_ptr(),0xff,owner->size_bytes(),device.ordinal,stream));
        EXPECT_EQ(first.captureIdentity(),identity);ASSERT_NO_THROW(first.requireFor(device));
        ASSERT_NO_THROW(second.requireRequest(device,1));
        EXPECT_THROW(second.requireRequest(device,0),std::invalid_argument);
        EXPECT_THROW(first.requireFor(DeviceId::cpu()),std::invalid_argument);
        const auto other=device.is_cuda()?DeviceId::rocm(device.ordinal):DeviceId::cuda(device.ordinal);
        EXPECT_THROW(first.requireFor(other),std::invalid_argument);
        EXPECT_THROW(QwenSparseCounterBinding::admit(arena,other,0,48,16),std::invalid_argument);
        EXPECT_THROW(QwenSparseCounterBinding::admit(arena,device,-1,48,16),std::invalid_argument);
        EXPECT_THROW(QwenSparseCounterBinding::admit(arena,device,3,48,16),std::invalid_argument);
        EXPECT_THROW(QwenSparseCounterBinding::admit(arena,device,0,INT_MAX,INT_MAX),std::invalid_argument);
        ASSERT_TRUE(backend->synchronizeStream(stream,device.ordinal));
    });
}
/** @brief Explicit CUDA setup proof; never part of Unit. */
TEST(QwenSparseCounterAdmission, CUDA){proveCounterAdmission(DeviceId::cuda(0));}
/** @brief Explicit ROCm setup proof; never part of Unit. */
TEST(QwenSparseCounterAdmission, ROCm){proveCounterAdmission(DeviceId::rocm(0));}
}
