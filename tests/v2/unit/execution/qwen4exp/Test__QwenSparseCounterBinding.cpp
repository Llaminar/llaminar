/**
 * @file Test__QwenSparseCounterBinding.cpp
 * @brief Device-free exact canonical arena extent and foreign-owner rejection.
 *
 * No GPU backend initializes or device storage is materialized. Native-positive
 * admission is proved separately in the explicit CUDA/ROCm integration fixture.
 */
#include "execution/qwen4exp/QwenSparseCounterBinding.h"
#include "tensors/TensorFactory.h"
#include "mocks/MockMPIContext.h"
#include <gtest/gtest.h>
#include <type_traits>
using namespace llaminar2;
/** @brief An arbitrary native pointer/interior view cannot represent an admitted control owner. */
TEST(QwenSparseCounterBinding, MissingForeignExternalAndCPUOwnersAreRejectedWithoutMaterialization){
    static_assert(!std::is_constructible_v<QwenSparseCounterBinding,std::int32_t*,int>);
    BufferArena absent;
    EXPECT_THROW(QwenSparseCounterBinding::admit(absent,DeviceId::cuda(0),0,48,16),std::invalid_argument);
    INT32Tensor foreign({3,sampling_math::kDeviceGenerationControlCount});BufferArena borrowed;
    ASSERT_TRUE(borrowed.registerExternalBuffer(BufferId::MTP_GENERATION_CONTROL,&foreign));
    EXPECT_THROW(QwenSparseCounterBinding::admit(borrowed,DeviceId::cuda(0),0,48,16),std::invalid_argument);
    EXPECT_EQ(foreign.gpu_data_ptr(),nullptr);
}
/** @brief The actual canonical arena admits exactly90 INT32 words per request, without a second BOM. */
TEST(QwenSparseCounterBinding, CanonicalOwnedArenaAndTerminalGeometryHaveOneExactExtent){
    llaminar2::test::MockMPIContext mpi;TensorFactory factory(mpi);BufferArena arena(ArenaConfig{.factory=&factory});
    ASSERT_TRUE(arena.registerBuffer(BufferId::MTP_GENERATION_CONTROL,3,sampling_math::kDeviceGenerationControlCount,"INT32",DeviceId::cpu()));
    ASSERT_TRUE(arena.allocate());auto tensor=arena.getSharedTensor(BufferId::MTP_GENERATION_CONTROL);ASSERT_TRUE(tensor);
    EXPECT_EQ(tensor->size_bytes(),3u*(208+32+48+72));EXPECT_EQ(arena.getRows(BufferId::MTP_GENERATION_CONTROL),3u);
    EXPECT_EQ(arena.getCols(BufferId::MTP_GENERATION_CONTROL),90u);
    EXPECT_EQ(tensor->gpu_data_ptr(),nullptr);
    EXPECT_THROW(QwenSparseCounterBinding::admit(arena,DeviceId::cpu(),0,48,16),std::invalid_argument);
    EXPECT_THROW(QwenSparseCounterBinding::admit(arena,DeviceId::cuda(0),0,48,16),std::invalid_argument);
    EXPECT_THROW(QwenSparseCounterBinding::admit(arena,DeviceId::cuda(0),-1,48,16),std::invalid_argument);
    EXPECT_THROW(QwenSparseCounterBinding::admit(arena,DeviceId::cuda(0),3,48,16),std::invalid_argument);
    EXPECT_EQ(tensor->gpu_data_ptr(),nullptr);
}
