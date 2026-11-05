/** @file Test__NativeAllowanceEligibility.cpp
 * @brief Replay tested native resource eligibility without application-source admission dependencies.
 * No native runtime or installed limits are manufactured by these metadata-only tests.
 */
#include "backends/NativeContextQualifiedLimits.h"
#include "fixtures/backends/RecordedNativeContextBirthKey.h"
#include "backends/NativeAuxiliaryQualifiedLimits.h"
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include <gtest/gtest.h>
#include <algorithm>
namespace llaminar2::test {
namespace {
/** @return Complete recorded runtime/hardware/count metadata from an installed tested class; no native creator runs. */
NativeContextBirthKey recorded(const detail::NativeContextQualifiedClass& row){return recordedNativeContextBirthKey(row.backend,row.population);}

}
TEST(NativeAllowanceEligibility, ApplicationSourceChangesDoNotAffectAllowanceOrPhysicalIdentity){
 ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
 for(const auto& row:detail::kNativeContextQualifiedClasses){
  auto first=recorded(row),second=first;second.source_sha256="unrelated implementation revision";
  const auto a=NativeContextOpaqueCatalog::require(first),b=NativeContextOpaqueCatalog::require(second);
  EXPECT_EQ(a.hostUpperBytes(),b.hostUpperBytes());EXPECT_EQ(a.deviceUpperBytes(),b.deviceUpperBytes());
  EXPECT_EQ(b.key(),second);EXPECT_NE(a.key(),b.key());
  auto absent=second;absent.source_sha256.clear();EXPECT_NO_THROW(NativeContextOpaqueCatalog::require(absent));
  EXPECT_EQ(NativeAuxiliaryResourceCatalog::liveSourceIdentity(first),NativeAuxiliaryResourceCatalog::liveSourceIdentity(absent));
  const auto program=NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(false));
  const auto selected=NativeOrderingEventGroupReplayAccess::select(second,7,program,std::string(64,'e'),detail::kNativeAuxiliaryQualifiedClasses);
  EXPECT_GT(selected.hostBytes,0u);EXPECT_EQ(selected.sourceIdentity,NativeAuxiliaryResourceCatalog::liveSourceIdentity(second));
  EXPECT_EQ(NativeAuxiliaryResourceCatalog::portableSourceIdentity(first),NativeAuxiliaryResourceCatalog::portableSourceIdentity(second));
  EXPECT_EQ(NativeAuxiliaryResourceCatalog::liveSourceIdentity(first),NativeAuxiliaryResourceCatalog::liveSourceIdentity(second));
 }
}
/** @test Application evidence changes do not affect auxiliary limits; the selected program and original generation remain exact. */
TEST(NativeAllowanceEligibility, AuxiliaryResourceSelectionIgnoresApplicationEvidence){
 ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
 for(const auto& installed:detail::kNativeContextQualifiedClasses){
  const auto first=recorded(installed);auto second=first;second.source_sha256="unrelated implementation revision";
  const auto program=NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(false));
  const auto historical=std::string(64,'a'),current=std::string(64,'b');
  std::array rows{detail::NativeAuxiliaryQualifiedClass{first.device.type,program,3072,6144,historical,NativePhysicalLifetime::ProcessRuntime}};
  const auto a=NativeOrderingEventGroupReplayAccess::select(first,1,program,historical,rows);
  const auto b=NativeOrderingEventGroupReplayAccess::select(second,2,program,current,rows);
  EXPECT_EQ(a.hostBytes,b.hostBytes);EXPECT_EQ(a.deviceBytes,b.deviceBytes);
  EXPECT_EQ(a.sourceIdentity,b.sourceIdentity);EXPECT_EQ(b.resourceSource,current);EXPECT_EQ(b.contextGeneration,2u);
  EXPECT_THROW(NativeOrderingEventGroupReplayAccess::select(first,1,NativeAuxiliaryResourceProgram::readAheadSide(),current,rows),std::runtime_error);
 }
}
/** @test Valid software/hardware/object variations do not cause synthetic OOM; malformed counts remain invalid. */
TEST(NativeAllowanceEligibility, ResourceFamiliesIgnoreDevelopmentFingerprintsButKeepCreatorCounts){
 ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
 for(const auto& row:detail::kNativeContextQualifiedClasses){
  auto key=recorded(row);key.backend_sha256=key.runtime_sha256=key.driver_sha256=key.runtime_policy_sha256=std::string(64,'f');
  key.runtime_version+=".compatible-patch";key.driver_version+=".compatible-patch";key.architecture+="-supported";key.compute_units++;key.native_wave_width*=2;
  key.context_object_bytes++;key.context_object_alignment*=2;key.worker_stack_bytes+=4096;key.pthread_defaults_identity="different-valid-defaults";
  const auto selected=NativeContextOpaqueCatalog::require(key);EXPECT_EQ(selected.hostUpperBytes(),row.host_upper_bytes);EXPECT_EQ(selected.deviceUpperBytes(),row.device_upper_bytes);EXPECT_EQ(selected.key(),key);
  const auto program=NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(false));
  EXPECT_NO_THROW(NativeOrderingEventGroupReplayAccess::select(key,1,program,"native-auxiliary-operations",detail::kNativeAuxiliaryQualifiedClasses));
  key.dependency_events++;EXPECT_THROW(NativeContextOpaqueCatalog::require(key),std::invalid_argument);
 }
}
/** @test Explicit compiled object pages remain independently priced even when opaque units stay fixed. */
TEST(NativeAllowanceEligibility, ExplicitObjectPagesAndRealCapacityRemainOwned){
 const auto small=NativeContextPreparationReplayAccess::objectProgram(1,8);
 const auto large=NativeContextPreparationReplayAccess::objectProgram(small.pageBytes()+1,8);
 EXPECT_EQ(large.contextObjectBytes(),small.pageBytes()+1);EXPECT_EQ(large.allocationBytes(),small.allocationBytes()+small.pageBytes());
 const PhysicalMemoryResource host{0,DeviceId::cpu(),large.allocationBytes(),large.allocationBytes()};
 PhysicalMemoryPlanBuilder plan;plan.add(host,PhysicalMemoryOwner::NativeExecutionContext,large.allocationBytes());
 auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
 auto owned=memory->reserveNewAllocations(host.device,PhysicalMemoryOwner::NativeExecutionContext,large.allocationBytes());
 EXPECT_EQ(memory->reservedBytes(host.device,PhysicalMemoryOwner::NativeExecutionContext),large.allocationBytes());
 EXPECT_THROW((void)memory->reserveNewAllocations(host.device,PhysicalMemoryOwner::NativeExecutionContext,1),std::logic_error);
 PhysicalMemoryPlanBuilder oversized;oversized.add(host,PhysicalMemoryOwner::NativeExecutionContext,large.allocationBytes()+1);
 EXPECT_THROW((void)PhysicalMemoryPlanAdmissionCertificate(oversized.build()),PhysicalMemoryCapacityExhausted);
 EXPECT_THROW((void)memory->reserveNewAllocations(DeviceId::cuda(0),PhysicalMemoryOwner::NativeExecutionContext,1),std::out_of_range);
 EXPECT_THROW((void)memory->reserveNewAllocations(host.device,PhysicalMemoryOwner::Count,1),std::invalid_argument);
}
/** @test Backend/resource/lifetime errors remain errors independently from developer fingerprints. */
TEST(NativeAllowanceEligibility, InvalidResourceFamilyAndLifetimeStillReject){
 ASSERT_FALSE(detail::kNativeContextQualifiedClasses.empty());
 const auto key=recorded(detail::kNativeContextQualifiedClasses.front());const auto p=NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(false));
 const std::string evidence(64,'a');std::array rows{detail::NativeAuxiliaryQualifiedClass{DeviceType::CPU,p,4096,8192,evidence,NativePhysicalLifetime::ProcessRuntime}};
 EXPECT_THROW(NativeOrderingEventGroupReplayAccess::select(key,1,p,"family",rows),std::runtime_error);
 rows[0].backend=key.device.type;rows[0].lifetime=static_cast<NativePhysicalLifetime>(-1);
 EXPECT_THROW(NativeOrderingEventGroupReplayAccess::select(key,1,p,"family",rows),std::logic_error);
 rows[0].lifetime=NativePhysicalLifetime::ProcessRuntime;
 EXPECT_THROW(NativeOrderingEventGroupReplayAccess::select(key,0,p,"family",rows),std::invalid_argument);
 EXPECT_THROW(NativeOrderingEventGroupReplayAccess::select(key,1,NativeAuxiliaryResourceProgram::readAheadSide(),"family",rows),std::runtime_error);
}

/** @test Installed side/parent limits ignore fingerprints while live provenance stays exact. */
TEST(NativeAllowanceEligibility, ReadAheadInstalledFamiliesKeepOriginalIdentityWithoutFingerprintSelection){
 for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
  const auto found=std::find_if(detail::kNativeContextQualifiedClasses.begin(),detail::kNativeContextQualifiedClasses.end(),
      [&](const auto& row){return row.backend==backend;});
  ASSERT_NE(found,detail::kNativeContextQualifiedClasses.end());
  const auto first=recorded(*found);auto changed=first;
  changed.source_sha256="independent application revision";
  changed.backend_sha256=changed.runtime_sha256=changed.driver_sha256=changed.runtime_policy_sha256=std::string(64,'f');
  changed.runtime_version+=".patch";changed.driver_version+=".patch";
  changed.architecture+="-supported";changed.compute_units++;changed.native_wave_width*=2;
  changed.context_object_bytes++;changed.context_object_alignment*=2;
  changed.worker_stack_bytes+=4096;changed.pthread_defaults_identity="different-valid-defaults";
  for(const auto program:{NativeAuxiliaryResourceProgram::readAheadSide(),
      NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::readAheadParentTerminal())}){
   const auto a=NativeOrderingEventGroupReplayAccess::select(first,11,program,"original-resource-source",detail::kNativeAuxiliaryQualifiedClasses);
   const auto b=NativeOrderingEventGroupReplayAccess::select(changed,12,program,"changed-resource-source",detail::kNativeAuxiliaryQualifiedClasses);
   const auto installed=NativeAuxiliaryResourceCatalog::allowance(backend,program);
   EXPECT_EQ(a.hostBytes,installed.hostBytes);EXPECT_EQ(b.hostBytes,a.hostBytes);
   EXPECT_EQ(a.deviceBytes,installed.deviceBytes);EXPECT_EQ(b.deviceBytes,a.deviceBytes);
   EXPECT_EQ(a.evidence,b.evidence);EXPECT_EQ(b.lifetime,NativePhysicalLifetime::ProcessRuntime);
   EXPECT_EQ(a.program,program);EXPECT_EQ(b.program,program);
   EXPECT_EQ(a.contextGeneration,11u);EXPECT_EQ(b.contextGeneration,12u);
   EXPECT_EQ(a.sourceIdentity,NativeAuxiliaryResourceCatalog::liveSourceIdentity(first));
   EXPECT_EQ(b.sourceIdentity,NativeAuxiliaryResourceCatalog::liveSourceIdentity(changed));
   EXPECT_NE(a.sourceIdentity,b.sourceIdentity);
   EXPECT_EQ(b.resourceSource,"changed-resource-source");
   EXPECT_THROW(NativeOrderingEventGroupReplayAccess::select(changed,0,program,"source",detail::kNativeAuxiliaryQualifiedClasses),std::invalid_argument);
  }
 }
}

}
