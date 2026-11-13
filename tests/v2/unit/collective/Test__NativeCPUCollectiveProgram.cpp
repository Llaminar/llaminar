/** @file Test__NativeCPUCollectiveProgram.cpp
 * @brief Actual program/PMA/session/shutdown bodies with native MPI boundary doubles; no communicator readiness.
 */
#include "collective/NativeCPUCollectiveProgram.h"
#include "app/MPIProcessSession.h"
#include "app/MPIShutdown.h"
#include "collective/BackendRouter.h"
#include "kernels/KernelFactory.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <functional>
#include <optional>
using namespace llaminar2;
namespace {
int initialized=0,finalized=0,finalize_result=MPI_SUCCESS,finalize_calls=0;
std::function<void()> during_finalize;
void freshMPI(){initialized=0;finalized=0;finalize_result=MPI_SUCCESS;finalize_calls=0;during_finalize={};}
NativeCPUCollectiveProgram spec(){return NativeCPUCollectiveProgram::expertShardRoutePartials(7,CollectiveBackendType::UPI,{{0,GlobalDeviceAddress::cpu(0)},{1,GlobalDeviceAddress::cpu(1)}});}
std::shared_ptr<PhysicalMemoryAuthority> memory(const NativeCPUCollectiveProgram& p,int domains=1){
 PhysicalMemoryResource cpu{0,DeviceId::cpu(),1024u*1024u*1024u,1024u*1024u*1024u};
 PhysicalMemoryPlanBuilder plan;
 for(int i=0;i<domains;++i)p.contributeDomainBOM(plan,cpu,0);
 NativeCPUCollectiveProgram::contributeProcessPoolBOM(plan,cpu);
 return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
}
std::size_t reserved(const std::shared_ptr<PhysicalMemoryAuthority>& m,PhysicalMemoryOwner o){return m->reservedBytes(DeviceId::cpu(),o);}
PhysicalMemoryOwnerAttestation poolRow(const std::shared_ptr<PhysicalMemoryAuthority>& m){for(auto row:m->rankAttestation())if(row.owner==PhysicalMemoryOwner::NativeCPUCollectiveProcessPool)return row;throw std::logic_error("no pool row");}
MPIProcessSession session(){int argc=0;char** argv=nullptr;return MPIProcessSession::initialize(argc,argv);}
}
extern "C" {
int MPI_Initialized(int* out){*out=initialized;return MPI_SUCCESS;}
int MPI_Finalized(int* out){*out=finalized;return MPI_SUCCESS;}
int MPI_Init_thread(int*,char***,int,int* out){initialized=1;*out=MPI_THREAD_MULTIPLE;return MPI_SUCCESS;}
int MPI_Allreduce(const void*,void*,int,MPI_Datatype,MPI_Op,MPI_Comm){return MPI_SUCCESS;}
int MPI_Finalize(){++finalize_calls;if(during_finalize)during_finalize();if(finalize_result==MPI_SUCCESS)finalized=1;return finalize_result;}
}
// These are external native/cache boundaries of the actual MPIShutdown body;
// this replay creates no backend, model, GPU context or communicator.
void llaminar2::GlobalBackendRouter::shutdown(){}
void llaminar::v2::kernels::KernelFactory::clearCache(){}
void llaminar2::GPUDeviceProcessLifetime::retireActive() noexcept{}
namespace llaminar2::test {
struct NativeCPUCollectiveProgramReplayAccess {
 static void begin(NativeCPUCollectiveProgram::DomainUse& use){use.beginNative();}
 static void retire(NativeCPUCollectiveProgram::DomainUse& use){use.retireAfterNativeDestruction();}
 static void begin(NativeCPUCollectiveProgram::ProcessUse& use){use.beginNative();}
 static void retire(NativeCPUCollectiveProgram::ProcessUse& use){use.retireAfterNativeSplit();}
};
}
using Access=llaminar2::test::NativeCPUCollectiveProgramReplayAccess;
TEST(NativeCPUCollectiveProgram, IndependentAdmissionsBorrowOneOriginalPoolUntilFinalize){
 freshMPI();auto mpi=session();auto p=spec();auto a=memory(p);std::weak_ptr<PhysicalMemoryAuthority> first=a;
 {auto use=p.admit(a,0);Access::begin(use);EXPECT_EQ(reserved(a,PhysicalMemoryOwner::NativeCPUCollectiveDomain),p.domainBytes());Access::retire(use);}
 auto second=memory(p);{auto use=p.admit(second,0);Access::begin(use);auto row=poolRow(second);
 EXPECT_EQ(row.borrowed_opaque_runtime_committed_bytes,p.processPoolBytes());EXPECT_EQ(row.committed_new_bytes,p.processPoolBytes());
 EXPECT_TRUE(second->rankCommitted());EXPECT_FALSE(second->rankComplete());EXPECT_EQ(row.opaque_runtime_committed_bytes,0u);
 EXPECT_EQ(row.materialized_new_bytes,0u);EXPECT_EQ(row.adopted_resident_bytes,0u);EXPECT_EQ(reserved(second,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),0u);
 EXPECT_THROW((void)second->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::NativeCPUCollectiveProcessPool,1),std::logic_error);
 EXPECT_THROW((void)second->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::NativeCPUCollectiveProcessPool,1),std::logic_error);
 EXPECT_EQ(reserved(a,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),p.processPoolBytes());Access::retire(use);}
 a.reset();EXPECT_TRUE(first.expired()); // Reservation retains canonical ledger, not dead model wrapper.
 during_finalize=[&]{EXPECT_EQ(poolRow(second).borrowed_opaque_runtime_committed_bytes,p.processPoolBytes());};
 mpiShutdown();EXPECT_EQ(finalize_calls,1);EXPECT_EQ(poolRow(second).borrowed_opaque_runtime_committed_bytes,0u);EXPECT_EQ(poolRow(second).committed_new_bytes,0u);
}
TEST(NativeCPUCollectiveProgram, PendingFailureAndRepeatedDomainRetirement){
 freshMPI();auto mpi=session();auto p=spec();auto m=memory(p,2);
 auto a=p.admit(m,0);Access::begin(a);auto b=p.admit(m,0);Access::begin(b);
 EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveDomain),2*p.domainBytes());
 EXPECT_THROW(mpiShutdown(),std::logic_error);EXPECT_EQ(finalize_calls,0);EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),p.processPoolBytes());
 Access::retire(a);EXPECT_THROW(Access::retire(a),std::logic_error);Access::retire(b);
 EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);mpiShutdown();EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),0u);
}
TEST(NativeCPUCollectiveProgram, NoNativeEntryRetiresOnlyDomainButSessionKeepsPool){
 freshMPI();auto mpi=session();auto p=spec();auto m=memory(p);{auto unused=p.admit(m,0);}
 EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
 EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),p.processPoolBytes());
 mpiShutdown();EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),0u);
}
TEST(NativeCPUCollectiveProgram, FailedFinalizeRetainsPoolAndCannotRetryOrClaimRetirement){
 EXPECT_DEATH({
  freshMPI();auto mpi=session();auto p=spec();auto m=memory(p);{auto unused=p.admit(m,0);}
  finalize_result=MPI_ERR_OTHER;
  try {mpiShutdown();std::_Exit(0);}catch(const std::runtime_error&){}
  if(finalize_calls!=1 || reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool)!=p.processPoolBytes())std::_Exit(0);
  // Scope destruction must fail before a second original MPI_Finalize call.
 }, "Previous MPI_Finalize has no original successful retirement");
}
TEST(NativeCPUCollectiveProgram, GeometryNoAdmissionAndOriginalNativeOrdering){
 freshMPI();auto p=spec();auto m=memory(p);EXPECT_THROW((void)p.admit(m,0),std::logic_error);EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
 auto mpi=session();EXPECT_THROW((void)p.admit(m,1),std::logic_error);
 PhysicalMemoryPlanBuilder missingPool;
 p.contributeDomainBOM(missingPool,{0,DeviceId::cpu(),1024u*1024u*1024u,1024u*1024u*1024u},0);
 auto incomplete=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(missingPool.build()),0);
 EXPECT_THROW((void)p.admit(incomplete,0),std::logic_error);
 EXPECT_EQ(reserved(incomplete,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
 EXPECT_EQ(reserved(incomplete,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),0u);
 auto unrelated_control=std::shared_ptr<PhysicalMemoryAuthority>(m.get(),[](auto*){});
 EXPECT_THROW((void)p.admit(unrelated_control,0),std::logic_error);
 EXPECT_THROW((void)NativeCPUCollectiveProgram::expertShardRoutePartials(0,CollectiveBackendType::AUTO,{}),std::invalid_argument);
 EXPECT_THROW((void)NativeCPUCollectiveProgram::expertShardRoutePartials(0,CollectiveBackendType::MPI,{{0,GlobalDeviceAddress::cpu(-1)}}),std::invalid_argument);
 EXPECT_EQ(p.communicatorCount(),2u);mpiShutdown();EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),0u);
}

TEST(NativeCPUCollectiveProgram, PartialSetupIsBoundedAndNeverRetiresAnArmedNativeUseByScope){
 freshMPI();auto mpi=session();auto p=spec();auto m=memory(p,2);
 { auto first=p.admit(m,0);Access::begin(first);
   {auto not_entered=p.admit(m,0);} // Second creator failed before any MPI native call.
   EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveDomain),p.domainBytes());
   Access::retire(first);
 }
 EXPECT_DEATH({ auto unproven=p.admit(m,0);Access::begin(unproven); }, "");
 mpiShutdown();EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
}

namespace {
/** @brief Recorded excluded-rank PMA input, without a fabricated CPU-domain endpoint or communicator. */
std::shared_ptr<PhysicalMemoryAuthority> excludedMemory(){
 PhysicalMemoryPlanBuilder plan;
 NativeCPUCollectiveProgram::contributeProcessPoolBOM(plan,{2,DeviceId::cpu(),1024u*1024u*1024u,1024u*1024u*1024u});
 return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()),2);
}
}
TEST(NativeCPUCollectiveProgram, ExcludedParentBorrowsOnlyOriginalProcessPoolAcrossAdmissions){
 freshMPI();auto mpi=session();auto p=spec();auto a=excludedMemory();auto b=excludedMemory();
 EXPECT_THROW((void)p.admit(a,2),std::logic_error);
 {auto split=p.admitExcludedParent(a,2);Access::begin(split);
  EXPECT_EQ(split.originalWorldRank(),2);EXPECT_EQ(split.originalMemory(),a);EXPECT_EQ(split.program(),p);
  EXPECT_EQ(reserved(a,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
  EXPECT_EQ(reserved(a,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),p.processPoolBytes());Access::retire(split);}
 {auto split=p.admitExcludedParent(b,2);Access::begin(split);
  EXPECT_EQ(reserved(b,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
  EXPECT_EQ(reserved(b,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),0u);
  EXPECT_TRUE(b->rankCommitted());EXPECT_FALSE(b->rankComplete());
  EXPECT_EQ(poolRow(b).borrowed_opaque_runtime_committed_bytes,p.processPoolBytes());
  std::weak_ptr<PhysicalMemoryAuthority> first=a;a.reset();EXPECT_TRUE(first.expired());Access::retire(split);}
 during_finalize=[&]{EXPECT_EQ(poolRow(b).borrowed_opaque_runtime_committed_bytes,p.processPoolBytes());};
 mpiShutdown();EXPECT_EQ(poolRow(b).committed_new_bytes,0u);
}
TEST(NativeCPUCollectiveProgram, ExcludedSplitAndPendingNativeEntryHonorOriginalSessionClosure){
 freshMPI();auto mpi=session();auto p=spec();auto m=excludedMemory();
 auto split=p.admitExcludedParent(m,2);Access::begin(split);
 {auto pending=p.admitExcludedParent(m,2);
  EXPECT_THROW(mpiShutdown(),std::logic_error);EXPECT_EQ(finalize_calls,0);
  EXPECT_THROW(Access::begin(pending),std::logic_error);
  EXPECT_THROW((void)p.admitExcludedParent(m,2),std::logic_error);
 }
 Access::retire(split);EXPECT_THROW(Access::retire(split),std::logic_error);
 EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
 mpiShutdown();EXPECT_EQ(finalize_calls,1);EXPECT_EQ(poolRow(m).committed_new_bytes,0u);
}

TEST(NativeCPUCollectiveProgram, VariableOriginalPopulationsUseSameLifetimeContract){
 // Recorded endpoint metadata tests no new socket discovery/affinity or native communicator readiness.
 for(int population=1;population<=17;++population){
  SCOPED_TRACE(population);freshMPI();auto mpi=session();
  std::vector<NativeCPUCollectiveProgram::Participant> members;
  for(int i=0;i<population;++i)members.push_back({i,GlobalDeviceAddress::cpu(i)});
  auto p=NativeCPUCollectiveProgram::expertShardRoutePartials(11,CollectiveBackendType::MPI,std::move(members));
  EXPECT_EQ(p.participants().size(),static_cast<size_t>(population));
  EXPECT_EQ(p.communicatorCount(),population==1?1u:2u);
  auto m=memory(p);{auto token=p.admit(m,0);Access::begin(token);EXPECT_TRUE(m->rankCommitted());Access::retire(token);}
  EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
  mpiShutdown();EXPECT_EQ(reserved(m,PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),0u);
 }
 std::vector<NativeCPUCollectiveProgram::Participant> invalid{{0,GlobalDeviceAddress::cpu(3)},
     {1,GlobalDeviceAddress::cpu(3)}};
 EXPECT_THROW((void)NativeCPUCollectiveProgram::expertShardRoutePartials(11,CollectiveBackendType::MPI,std::move(invalid)),std::invalid_argument);
}
