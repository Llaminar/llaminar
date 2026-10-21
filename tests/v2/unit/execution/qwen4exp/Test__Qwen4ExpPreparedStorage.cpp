/**
 * @file Test__Qwen4ExpPreparedStorage.cpp
 * @brief Device-free declaration/preparation/seal/stale-storage lifecycle proof.
 *
 * The existing host-memory MockBackend supplies canonical storage and runtime
 * generations. No native driver discovery, accelerator work or model is used.
 */
#include "execution/qwen4exp/Qwen4ExpPreparedStorage.h"
#include "transfer/TransferEngine.h"
#include "tensors/Tensors.h"
#include "mocks/MockBackend.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
using namespace llaminar2;
namespace {
/** @brief Host-memory backend with an explicit native runtime generation authority. */
struct Backend final:test::MockBackend {
 explicit Backend(DeviceType type):MockBackend(type){}
 std::uint64_t generation=1;
 std::uint64_t deviceRuntimeGeneration(int)const override{return generation;}
};
/** @brief Exact non-default mock stream; no native submission occurs. */
void*stream(){return reinterpret_cast<void*>(0x12340);}
}
/** @test Declaration inspection never materializes storage or changes after ordinary preparation. */
TEST(Qwen4ExpPreparedStorage, DeclarationPrepareSealReplayBothGPUAuthorities){
 llaminar2::testing::installHardwareFreeGPUContextFactories();
 for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}){
  Backend backend(device.type);FP32Tensor first({2,8}),second({2,8});first.setBackendForTesting(&backend);second.setBackendForTesting(&backend);
  std::array<TensorBase*,3>roots{&first,&second,&first};Qwen4ExpPreparedStorage storage(device,roots);
  EXPECT_EQ(first.gpu_data_ptr(),nullptr);
 EXPECT_FALSE(storage.isSealed());
 EXPECT_NO_THROW(storage.validateIfSealed());
 EXPECT_EQ(backend.getAllocationCount(),0);
  EXPECT_THROW(storage.seal(),std::logic_error);
 EXPECT_FALSE(storage.isSealed());
  TransferEngine::prepareDeviceInput(&first,device,stream());
 EXPECT_THROW(storage.seal(),std::logic_error);
 EXPECT_FALSE(storage.isSealed());
  TransferEngine::prepareDeviceOutput(&second,device,stream());
  const auto allocations=backend.getAllocationCount();const auto bytes=backend.getTransferStats().h2d_bytes;
  EXPECT_NO_THROW(storage.seal());
 EXPECT_TRUE(storage.isSealed());
  for(int replay=0;replay<3;++replay){EXPECT_NO_THROW(storage.validateIfSealed());
 EXPECT_NO_THROW(storage.seal());}
  EXPECT_EQ(backend.getAllocationCount(),allocations);
 EXPECT_EQ(backend.getTransferStats().h2d_bytes,bytes);
  backend.generation=2;
 EXPECT_THROW(storage.validateIfSealed(),std::logic_error);
 EXPECT_THROW(storage.seal(),std::logic_error);
  backend.generation=1;
 EXPECT_NO_THROW(storage.validate());
 ASSERT_TRUE(first.releaseDeviceMemory());
 EXPECT_THROW(storage.validateIfSealed(),std::logic_error);
 }
}
/** @test Runtime identity is authenticated even when the native pointer itself did not change. */
TEST(Qwen4ExpPreparedStorage, RejectZeroGenerationAndBackendReplacement){
 llaminar2::testing::installHardwareFreeGPUContextFactories();
 for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}){
  Backend backend(device.type),other(device.type);FP32Tensor value({1,8});value.setBackendForTesting(&backend);
  std::array<TensorBase*,1>roots{&value};Qwen4ExpPreparedStorage storage(device,roots);
  TransferEngine::prepareDeviceInput(&value,device,stream());backend.generation=0;
 EXPECT_THROW(storage.seal(),std::logic_error);
 EXPECT_FALSE(storage.isSealed());
  backend.generation=1;storage.seal();value.setBackendForTesting(&other);
 EXPECT_THROW(storage.validate(),std::logic_error);
  value.setBackendForTesting(&backend);
 EXPECT_NO_THROW(storage.validate());
 }
}
/** @test CPU preparation keeps the same declaration but release invalidates its physical seal. */
TEST(Qwen4ExpPreparedStorage, CPURootReleaseAndInvalidDeclarations){
 FP32Tensor value({1,8});std::array<TensorBase*,1>roots{&value};Qwen4ExpPreparedStorage storage(DeviceId::cpu(),roots);
 EXPECT_NO_THROW(storage.validateIfSealed());
 EXPECT_THROW(storage.validate(),std::logic_error);storage.seal();
 EXPECT_NO_THROW(storage.validate());
 value.release_raw_data();
 EXPECT_THROW(storage.validateIfSealed(),std::logic_error);
 EXPECT_THROW((Qwen4ExpPreparedStorage{DeviceId::invalid(),roots}),std::invalid_argument);
 std::array<TensorBase*,1>missing{nullptr};
 EXPECT_THROW((Qwen4ExpPreparedStorage{DeviceId::cpu(),missing}),std::invalid_argument);
}
