/**
 * @file Test__QwenSparseCounterBinding.cpp
 * @brief Device-free exact canonical arena extent and foreign-owner rejection.
 *
 * No GPU backend initializes or device storage is materialized. Native-positive
 * admission is proved separately in the explicit CUDA/ROCm integration fixture.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */
#include "../../../utils/CPUActivationArenaFixture.h"
#include "execution/qwen4exp/QwenSparseCounterBinding.h"
#include "tensors/TensorFactory.h"
#include "mocks/MockMPIContext.h"
#include <gtest/gtest.h>
#include <type_traits>
using namespace llaminar2;
/** @brief An arbitrary native pointer/interior view cannot represent an admitted control owner. */
TEST(QwenSparseCounterBinding, MissingForeignExternalAndCPUOwnersAreRejectedWithoutMaterialization){
    static_assert(!std::is_constructible_v<QwenSparseCounterBinding,std::int32_t*,int>);
    BufferArena absent{::llaminar2::test::cpuArenaFixtureConfig()};
    EXPECT_THROW(QwenSparseCounterBinding::admit(absent,DeviceId::cuda(0),0,48,16),std::invalid_argument);
    INT32Tensor foreign({3,sampling_math::kDeviceGenerationControlCount});BufferArena borrowed{::llaminar2::test::cpuArenaFixtureConfig()};
    ASSERT_TRUE(borrowed.registerExternalBuffer(BufferId::MTP_GENERATION_CONTROL,&foreign));
    EXPECT_THROW(QwenSparseCounterBinding::admit(borrowed,DeviceId::cuda(0),0,48,16),std::invalid_argument);
    EXPECT_EQ(foreign.gpu_data_ptr(),nullptr);
}
/** @brief The canonical arena owns108 INT32 words: immutable scope and32B utility state/events share the existing BOM. */
TEST(QwenSparseCounterBinding, CanonicalOwnedArenaAndTerminalGeometryHaveOneExactExtent){
    static_assert(sampling_math::kDeviceGenerationControlDepthUtilityPhase==100);
    static_assert(sampling_math::kDeviceGenerationControlDepthUtilityProbes-sampling_math::kDeviceGenerationControlDepthUtilityPhase==5);
    static_assert(sampling_math::kDeviceGenerationControlCount-sampling_math::kDeviceGenerationControlDepthUtilityProbes==3);
    llaminar2::test::MockMPIContext mpi;TensorFactory factory(mpi);BufferArena arena(::llaminar2::test::cpuArenaFixtureConfig(&factory));
    ASSERT_TRUE(arena.registerBuffer(BufferId::MTP_GENERATION_CONTROL,3,sampling_math::kDeviceGenerationControlCount,"INT32",DeviceId::cpu()));
    ASSERT_TRUE(arena.allocate());auto tensor=arena.getSharedTensor(BufferId::MTP_GENERATION_CONTROL);ASSERT_TRUE(tensor);
    EXPECT_EQ(tensor->size_bytes(),3u*(208+32+48+72+40+20+12));EXPECT_EQ(arena.getRows(BufferId::MTP_GENERATION_CONTROL),3u);
    EXPECT_EQ(arena.getCols(BufferId::MTP_GENERATION_CONTROL),108u);
    EXPECT_EQ(tensor->gpu_data_ptr(),nullptr);
    EXPECT_THROW(QwenSparseCounterBinding::admit(arena,DeviceId::cpu(),0,48,16),std::invalid_argument);
    EXPECT_THROW(QwenSparseCounterBinding::admit(arena,DeviceId::cuda(0),0,48,16),std::invalid_argument);
    EXPECT_THROW(QwenSparseCounterBinding::admit(arena,DeviceId::cuda(0),-1,48,16),std::invalid_argument);
    EXPECT_THROW(QwenSparseCounterBinding::admit(arena,DeviceId::cuda(0),3,48,16),std::invalid_argument);
    EXPECT_EQ(tensor->gpu_data_ptr(),nullptr);
}
