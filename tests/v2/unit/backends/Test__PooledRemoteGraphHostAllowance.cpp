/** @file Test__PooledRemoteGraphHostAllowance.cpp
 * @brief Coarse declared-resource pricing and actual installed selector replay, without native SDK work.
 */
#include "backends/PooledRemoteGraphHostAllowance.h"
#include "backends/NativeAuxiliaryQualifiedLimits.h"
#include "fixtures/backends/RecordedNativeContextBirthKey.h"
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "execution/qwen4exp/PooledImmutableDirectedOwnerProgram.h"
#include "loaders/ModelLoader.h"
#include <gtest/gtest.h>
namespace llaminar2::test {
namespace {
/** @return Actual descriptor matching the measured P1 independent-coordinator program, not caller node integers. */
PooledRemoteGraphProgram measuredRemoteProgram(DeviceId remote,std::uint32_t capacity,std::size_t rowBytes=90){
    const PhysicalMemoryAllocatorIdentity coordinator{0,remote.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0)},owner{0,remote};
    const std::array order{owner};
    std::vector<ImmutableRowTableLayoutOwner> owners{{{owner,ImmutableRowStorage::VRAM},0,capacity,{}}};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,1048576,rowBytes),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,std::move(owners),{coordinator},{},
        {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},
        capacity?std::optional(coordinator):std::nullopt,8192);
    const auto physical=qwen4exp::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
    const auto policy=qwen4exp::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},8192);
    return qwen4exp::PooledImmutableDirectedOwnerProgram::declare(physical,policy,0);
}
}

TEST(PooledRemoteGraphHostAllowance,StartedBucketsAndCheckedArithmetic){
 const auto unit=std::size_t{1}<<30;
 for(auto nodes:{1ull,53ull,69ull,70ull,1023ull,1024ull})EXPECT_EQ(PooledRemoteGraphHostAllowance::bytesForDeclaredNodes(nodes),unit);
 for(auto nodes:{1025ull,2048ull})EXPECT_EQ(PooledRemoteGraphHostAllowance::bytesForDeclaredNodes(nodes),2*unit);
 EXPECT_EQ(PooledRemoteGraphHostAllowance::bytesForDeclaredNodes(2049),3*unit);
 EXPECT_THROW(PooledRemoteGraphHostAllowance::bytesForDeclaredNodes(0),std::invalid_argument);
 EXPECT_THROW(PooledRemoteGraphHostAllowance::scaledBytes(1,0),std::invalid_argument);
 EXPECT_THROW(PooledRemoteGraphHostAllowance::bytesForDeclaredNodes(UINT64_MAX),std::overflow_error);
 EXPECT_THROW(PooledRemoteGraphHostAllowance::scaledBytes(1025,SIZE_MAX),std::overflow_error);
 EXPECT_EQ(PooledRemoteGraphHostAllowance::scaledBytes(1024,SIZE_MAX),SIZE_MAX);
}
TEST(PooledRemoteGraphHostAllowance,ActualC0PositiveAndLargerProgramsUseInstalledBackendFamily){
 for(auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
  auto key=recordedNativeContextBirthKey(backend,NativeContextBirthPopulation::AdditionalContext);
  for(auto capacity:{0u,8u,16u,64u}){
   const auto descriptor=measuredRemoteProgram(key.device,capacity,capacity==64?9000:90);
   const auto request=NativeAuxiliaryResourceProgram::pooledRemoteOwner(descriptor);
   const auto selected=NativeOrderingEventGroupReplayAccess::select(key,9,request,"evidence",detail::kNativeAuxiliaryQualifiedClasses);
   EXPECT_EQ(selected.hostBytes,PooledRemoteGraphHostAllowance::bytesFor(descriptor));EXPECT_EQ(selected.deviceBytes,0u);
   EXPECT_EQ(selected.lifetime,NativePhysicalLifetime::ProcessRuntime);EXPECT_EQ(selected.program,request);EXPECT_EQ(selected.contextGeneration,9u);
   ASSERT_TRUE(selected.remoteFamily);
   auto larger=descriptor.limits();larger.maximumNodes=2049;larger.maximumEdges=2048;
   auto args=PooledRemoteGraphProgramClass::argumentLayout();++args.bytes;
   EXPECT_TRUE(selected.remoteFamily->accepts(larger,args));
   larger.maximumEdges--;EXPECT_FALSE(selected.remoteFamily->accepts(larger,args));
   args.alignment=3;EXPECT_FALSE(selected.remoteFamily->accepts(descriptor.limits(),args));
  }
 }
}
}
