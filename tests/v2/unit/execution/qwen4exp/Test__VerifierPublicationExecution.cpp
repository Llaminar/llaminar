/**
 * @file Test__VerifierPublicationExecution.cpp
 * @brief Device-free immutable publication-token role and recording-concurrency proofs.
 */
#include "execution/compute_stages/IComputeStage.h"
#include <gtest/gtest.h>
#include <array>
#include <thread>
using namespace llaminar2;
namespace {
/** @brief Metadata-only existing retained-publication roles, without native resources. */
class MetadataStage final:public IComputeStage {
 bool captured_,required_;
public:
 /** @brief Freeze an explicit participant and existing typed publication roles. */
 MetadataStage(DeviceId d,bool captured=true,bool required=true):IComputeStage(d),captured_(captured),required_(required){}
 bool execute(IDeviceContext*)override{return true;}
 ComputeStageType type()const override{return ComputeStageType::COPY;}
 std::size_t estimatedFlops()const override{return 0;}
 std::size_t estimatedMemoryBytes()const override{return 0;}
 bool supportsBackend(ComputeBackendType)const override{return true;}
 StageBufferRequirements getBufferRequirements()const override{return {};}
 bool hasVerifierStateCapture()const override{return captured_;}
 bool requiresVerifierStateCaptureForPublication()const override{return required_;}
 StageBufferContract bufferContract()const override{return {};}
 StageDumpInfo buildDumpInfoImpl()const override{return {};}
};
}
/** @test Explicit unbound publication is legal; default/CPU/wrong-role input never creates an ordinary binding. */
TEST(VerifierPublicationExecution, ExplicitUnboundInvocationRejectsDefaultsAndWrongRole){
 for(auto device:{DeviceId::cuda(0),DeviceId::rocm(1)}){
  MetadataStage stage(device);auto*ordinary=reinterpret_cast<void*>(0x1000);auto*publication=reinterpret_cast<void*>(0x2000);
  const auto unbound=stage.gpuExecutionForVerifierPublication(publication);
  EXPECT_EQ(unbound.nativeStream(),publication);EXPECT_EQ(unbound.device(),device);EXPECT_FALSE(stage.hasGPUStream());
  stage.setGPUStream(ordinary);
  for(std::uintptr_t value:{0u,1u,2u})EXPECT_THROW((void)stage.gpuExecutionForVerifierPublication(reinterpret_cast<void*>(value)),std::invalid_argument);
  EXPECT_EQ(stage.gpuStream(),ordinary);
  for(auto roles:{std::array{false,true},std::array{true,false}}){MetadataStage invalid(device,roles[0],roles[1]);invalid.setGPUStream(ordinary);
   EXPECT_THROW((void)invalid.gpuExecutionForVerifierPublication(publication),std::invalid_argument);EXPECT_EQ(invalid.gpuStream(),ordinary);}
 }
 MetadataStage cpu(DeviceId::cpu());EXPECT_THROW((void)cpu.gpuExecutionForVerifierPublication(reinterpret_cast<void*>(0x2000)),std::invalid_argument);
}
/** @test Each invocation owns immutable stream/device metadata and leaves ordinary capture binding unchanged. */
TEST(VerifierPublicationExecution, PerCallAuthorityDoesNotMutateOrdinaryBinding){
 for(auto device:{DeviceId::cuda(1),DeviceId::rocm(0)}){
  MetadataStage stage(device);auto*ordinary=reinterpret_cast<void*>(0x1000);stage.setGPUStream(ordinary);const auto signature=stage.graphCaptureVariantSignature();
  const auto first=stage.gpuExecutionForVerifierPublication(reinterpret_cast<void*>(0x2000));
  const auto second=stage.gpuExecutionForVerifierPublication(reinterpret_cast<void*>(0x3000));
  EXPECT_EQ(first.device(),device);EXPECT_EQ(second.device(),device);EXPECT_EQ(first.nativeStream(),reinterpret_cast<void*>(0x2000));EXPECT_EQ(second.nativeStream(),reinterpret_cast<void*>(0x3000));
  EXPECT_EQ(stage.gpuStream(),ordinary);EXPECT_EQ(stage.graphCaptureVariantSignature(),signature);
 }
}
/** @test Two recording threads read the same frozen stage while constructing independent invocation values. */
TEST(VerifierPublicationExecution, ConcurrentRecordingUsesIndependentImmutableTokens){
 MetadataStage stage(DeviceId::rocm(1));auto*ordinary=reinterpret_cast<void*>(0x1000);stage.setGPUStream(ordinary);
 std::array<void*,2>observed{};std::array<DeviceId,2>devices{DeviceId::cpu(),DeviceId::cpu()};
 std::array<std::thread,2>threads;
 for(int i=0;i<2;++i)threads[i]=std::thread([&,i]{for(int repeat=0;repeat<1000;++repeat){const auto token=stage.gpuExecutionForVerifierPublication(reinterpret_cast<void*>(std::uintptr_t(0x2000+i*0x1000)));observed[i]=token.nativeStream();devices[i]=token.device();}});
 // Ordinary recording configuration is independent of immutable invocation
 // construction. The factory must not even read this concurrently written field.
 std::thread ordinaryWriter([&]{for(int i=0;i<1000;++i)stage.setGPUStream(reinterpret_cast<void*>(std::uintptr_t(0x4000+(i%2)*0x1000)));stage.setGPUStream(ordinary);});
 for(auto&thread:threads)thread.join();ordinaryWriter.join();EXPECT_EQ(stage.gpuStream(),ordinary);
 for(int i=0;i<2;++i){EXPECT_EQ(observed[i],reinterpret_cast<void*>(std::uintptr_t(0x2000+i*0x1000)));EXPECT_EQ(devices[i],DeviceId::rocm(1));}
}
