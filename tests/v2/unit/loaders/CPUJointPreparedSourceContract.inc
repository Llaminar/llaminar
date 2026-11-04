/**
 * @file CPUJointPreparedSourceContract.inc
 * @brief Verify selected original CPU fragment contracts through production views and registries.
 *
 * Real tensor bytes retain source E512 and sparse logical IDs; geometry-only
 * engine fixtures prove ownership/adoption, not native arithmetic or NUMA
 * execution. The separate CPUExpertShardExecution native matrix replay owns
 * arithmetic. No GPU, communicator or serving readiness is manufactured here.
 */
#include "execution/moe/CPUExpertShardSourceSelection.h"
#include "../../utils/NativeWeightSourceTestRead.h"
#include "loaders/PreparedWeightIdentityContract.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "execution/moe/MoEOverlayPreparedWeightSource.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "loaders/ModelLoader.h"
#include "kernels/cpu/gemm/CPUWeightPreparationGeometry.h"
#include "loaders/WeightManager.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "utils/CPUProjectionTestWorkspace.h"
#include "utils/CPUJointSourceTestPlan.h"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <unistd.h>
namespace {
/** @return Canonical tier quota64+448 over the unchanged original source E512. */
llaminar2::MoERoutedExpertPlacementPlan jointSourcePlan() { return llaminar2::test::cpuJointSourceTestPlan(); }
/** @return Same actual cohort source for each independent canonical map construction. */
llaminar2::MoEOverlayLayerWeightManifest jointManifest() {
 using namespace llaminar2;const auto fp=ExpertWeightFormat::floating(TensorType::FP32);
 return {.layer_idx=7,.projections={{{ExpertTierWeightProjection::Gate,20,8,fp},
  {ExpertTierWeightProjection::Up,20,8,fp},{ExpertTierWeightProjection::Down,8,20,fp}}}};
}
/** @return Actual private original-map selection; never call the private constructor. */
std::shared_ptr<const llaminar2::CPUExpertShardSourceSelection> jointSelection(int member) {
 using namespace llaminar2;const auto map=MoEExpertOwnerMap::buildShardPlacement(jointSourcePlan(),{jointManifest()});
 for(const auto& p:map.participants())if(p.domain_name=="cpu_joint" && p.domain_participant_index==member)
  return CPUExpertShardSourceSelection::fromOriginalPlacement(map,7,p.participant_id);
 throw std::logic_error("Joint source test lost its canonical member");
}
/** @return Original member metadata with only the tested actor backed by this process's real NUMA node. */
std::shared_ptr<const llaminar2::CPUExpertShardSourceSelection> localJointSelection(int member) {
 using namespace llaminar2;auto plan=jointSourcePlan();
 for(auto& domain:plan.domains)if(domain.name=="cpu_joint")
  for(size_t p=0;p<domain.participants.size();++p)
   domain.participants[p]=GlobalDeviceAddress::cpu(test::localNativeSourceTestNUMANode()+int((p+domain.participants.size()-size_t(member))%domain.participants.size()));
 const auto map=MoEExpertOwnerMap::buildShardPlacement(plan,{jointManifest()});
 for(const auto& p:map.participants())if(p.domain_name=="cpu_joint" && p.domain_participant_index==member)
  return CPUExpertShardSourceSelection::fromOriginalPlacement(map,7,p.participant_id);
 throw std::logic_error("Local source actor lost its original member");
}
/** @brief Real source/consumer owner retained explicitly beside the committed binding vector. */
struct JointSourceActor {
 std::filesystem::path path;
 std::shared_ptr<llaminar2::MPIContext> mpi;
 std::unique_ptr<llaminar2::TensorFactory> factory;
 std::unique_ptr<llaminar2::ModelLoader> loader;
 std::shared_ptr<llaminar2::PhysicalMemoryAuthority> memory;
 std::unique_ptr<llaminar2::WeightManager> manager;
 ~JointSourceActor(){manager.reset();loader.reset();factory.reset();std::error_code error;std::filesystem::remove(path,error);}
};
/** @brief Bindings and their authentic Store; no fake native engine or source-window parent. */
struct JointSourceBindings:std::vector<llaminar2::WeightBinding> {
 std::shared_ptr<JointSourceActor> actor;
 void registerInto(llaminar2::ExpertGemmRegistry& registry,size_t first=0,size_t count=SIZE_MAX)const {
  const auto end=std::min(size(),first+std::min(count,size()-first));
  for(size_t i=first;i<end;++i)registry.registerCommittedCPUProjection(at(i),*actor->manager->preparedWeightStore());
 }
};
/**
 * @return Actual GGUF→common reader→Manager/PMA/Store records for one local component actor.
 * @note Scripted rank metadata does not claim MPI/socket execution. Native integration owns that proof.
 */
JointSourceBindings jointSources(std::shared_ptr<const llaminar2::CPUExpertShardSourceSelection> selected,
 size_t arena_bytes=0,size_t workspace_bytes=0) {
 using namespace llaminar2;auto actor=std::make_shared<JointSourceActor>();
 static size_t serial=0;
 actor->path=std::filesystem::temp_directory_path()/("llaminar_joint_actor_"+std::to_string(getpid())+"_"+std::to_string(++serial)+".gguf");
 const auto scalar=[](std::ofstream& out,auto value){out.write(reinterpret_cast<const char*>(&value),sizeof(value));};
 const auto string=[&](std::ofstream& out,const std::string& value){scalar(out,uint64_t(value.size()));out.write(value.data(),value.size());};
 {std::ofstream out(actor->path,std::ios::binary|std::ios::trunc);
  out.write("GGUF",4);scalar(out,uint32_t{3});scalar(out,uint64_t{3});scalar(out,uint64_t{0});
  for(int role=0;role<3;++role){string(out,std::string("blk.7.ffn_")+(role==0?"gate":role==1?"up":"down")+"_exps.weight");
   scalar(out,uint32_t{3});scalar(out,uint64_t(role==2?20:8));scalar(out,uint64_t(role==2?8:20));
   scalar(out,uint64_t{512});scalar(out,uint32_t{0});scalar(out,uint64_t(role)*512*160*sizeof(float));}
  const size_t at=size_t(out.tellp());std::vector<char> padding((at+31)/32*32-at);out.write(padding.data(),padding.size());
  for(int role=0;role<3;++role)for(int expert=0;expert<512;++expert)for(int element=0;element<160;++element)scalar(out,float(expert));
  if(!out)throw std::runtime_error("Failed writing actual local source fixture");}
 actor->mpi=std::make_shared<MPIContext>(selected->participant().world_rank,2,MPI_COMM_NULL);
 actor->factory=std::make_unique<TensorFactory>(*actor->mpi);actor->loader=std::make_unique<ModelLoader>(actor->factory.get());
 if(!actor->loader->loadModel(actor->path.string()))throw std::runtime_error("Failed parsing actual local source fixture");
 const auto placement=CPUStoragePlacement::onNode(selected->participant().address.numa_node);
 size_t weights=0;InferenceStrategy strategy{.model_id=ModelContextId{155}};
 WeightPlan plan(strategy,PhysicalMemoryOwner::RoutedExpertWeights);
 for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown}) {
  const auto projection=selected->ownership().projection(role);
  weights+=selected->expertIds().size()*placement.physicalBytes<float>(size_t(projection.rows)*projection.columns);
  const auto name=std::string("blk.7.ffn_")+(role==WeightRole::MoEExpertGate?"gate":role==WeightRole::MoEExpertUp?"up":"down")+"_exps.weight";
  plan.add(test::nativeJointRequirementForTest(selected,role,name));
 }
 const auto total=weights+arena_bytes+workspace_bytes;PhysicalMemoryPlanBuilder budget;
 const PhysicalMemoryResource host{selected->participant().world_rank,selected->participant().device,total,total};
 budget.add(host,PhysicalMemoryOwner::RoutedExpertWeights,weights);
 if(arena_bytes)budget.add(host,PhysicalMemoryOwner::ActivationArena,arena_bytes);
 if(workspace_bytes)budget.add(host,PhysicalMemoryOwner::ExecutionWorkspace,workspace_bytes);
 actor->memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),host.world_rank);
 actor->manager=std::make_unique<WeightManager>(*actor->loader,actor->mpi);actor->manager->installPhysicalMemoryAuthority(actor->memory);
 const auto frozen=actor->manager->materialize(plan);
 JointSourceBindings result;result.actor=std::move(actor);result.assign(frozen.bindings().begin(),frozen.bindings().end());return result;
}
}
/** @test Native source requests retain E512 while materializing only the selected448 fragments. */
TEST(CPUJointPreparedSource, ExactSelectedOriginalSourceAndSparseRegistryLease) {
 using namespace llaminar2;
 for(int member:{0,1}) {
  const auto metadata=jointSelection(member);ASSERT_EQ(metadata->expertIds().size(),448u);
  EXPECT_EQ(metadata->participant().address.numa_node,member);
  const auto selected=localJointSelection(member);
  EXPECT_EQ(selected->ownership().geometry().experts,512);EXPECT_EQ(selected->expertIds().front(),64);
  EXPECT_EQ(selected->expertIds().back(),511);EXPECT_EQ(selected->participant().device,DeviceId::cpu());

  WeightRequirement request;request.canonical_name="blk.7.ffn_down_exps.weight";request.layer=7;
  request.role=WeightRole::MoEExpertDown;request.derivation=WeightDerivationKind::ExpertSlice;
  request.bypass_tensor_parallel=true;request.expected_prepared_kind=PreparedWeightKind::MoeExpertSlab;
  request.tp_rank_or_device_index=request.overlay_participant_index=member;
  request.overlay_participant_world_rank=selected->participant().world_rank;request.overlay_domain="cpu_joint";
  request.cpu_joint_source=selected;request.projection_ownership=selected->ownership();
  WeightPlan plan(InferenceStrategy{});plan.add(request);const auto& slice=plan.requirements().front().slice;
  EXPECT_EQ(slice.expert_ids,std::vector<int>(selected->expertIds().begin(),selected->expertIds().end()));
  EXPECT_EQ(slice.expert_count,448u);EXPECT_FALSE(slice.inner_is_presliced);
  EXPECT_EQ(slice.row_start,size_t(member*10));EXPECT_EQ(slice.row_count,10u);EXPECT_EQ(slice.source_rows,20u);
  auto sources=jointSources(selected);ASSERT_EQ(sources.size(),1344u);
  for(const auto& source:sources) {
   EXPECT_EQ(static_cast<const float*>(source.tensor->raw_data())[0],float(source.identity.expert));
   EXPECT_FALSE(source.projection_source_owner);ASSERT_TRUE(source.native_source);
   EXPECT_EQ(source.native_source->original.original_shape[2],512u);
  }
  ExpertGemmRegistry registry;const auto scope=ExpertGemmRegistry::staticProjectionScope(sources.front());
  EXPECT_FALSE(registry.acquireStaticProjectionLayer(scope,selected->ownership(),sources));
  sources.registerInto(registry);
  const auto graph_lease=registry.acquireCPUJointLayer(sources.front().identity.model_id,selected->participant(),7);
  ASSERT_TRUE(graph_lease);EXPECT_EQ(graph_lease->cpu_joint_source,selected);
  EXPECT_THROW(registry.acquireCPUJointLayer(ModelContextId{156},selected->participant(),7),std::invalid_argument);
  auto foreign_descriptor=selected->participant();foreign_descriptor.address.numa_node+=2;
  EXPECT_THROW(registry.acquireCPUJointLayer(sources.front().identity.model_id,foreign_descriptor,7),std::invalid_argument);
  auto lease=registry.acquireStaticProjectionLayer(scope,selected->ownership(),sources);ASSERT_TRUE(lease);
  EXPECT_EQ(lease->cpu_joint_source,selected);ASSERT_EQ(lease->experts.size(),512u);
  for(int e=0;e<512;++e)EXPECT_EQ(lease->experts[size_t(e)].complete(),e>=64);
  EXPECT_THROW(registry.getEngineForParticipant(scope.domain_name,scope.device,scope.participant_world_rank,
    scope.participant_index,7,64,Role::DOWN),std::logic_error);
  const auto old=lease->experts[511].down;registry.clear();sources.clear();
  EXPECT_EQ(lease->experts[511].down,old);EXPECT_TRUE(lease->experts[511].complete());
 }
}
/** @test A selection/purpose cannot be replaced by equal compact shape or a different original member. */
TEST(CPUJointPreparedSource, RejectsForeignWholeAndIncompletePurpose) {
 using namespace llaminar2;const auto selected=localJointSelection(0),same=localJointSelection(0),foreign=jointSelection(1);
 EXPECT_EQ(*selected,*same);EXPECT_NE(*selected,*foreign);
 auto sources=jointSources(selected);ExpertGemmRegistry registry;
 const auto scope=ExpertGemmRegistry::staticProjectionScope(sources.front());
 auto changed=sources.front();changed.cpu_joint_source.reset();
 EXPECT_THROW(ExpertGemmRegistry::staticProjectionScope(changed),std::invalid_argument);
 changed=sources.front();changed.cpu_joint_source=foreign;
 EXPECT_THROW(ExpertGemmRegistry::staticProjectionScope(changed),std::invalid_argument);
 const auto identity=PreparedWeightIdentityContract::fromBinding(sources.front());ASSERT_TRUE(identity);
 changed=sources.front();changed.cpu_joint_source=selected;
 EXPECT_TRUE(identity->compatible(*PreparedWeightIdentityContract::fromBinding(changed)));
 // Structural equality is not authority to substitute a newly built cohort.
 changed.cpu_joint_source=same;
 EXPECT_FALSE(identity->compatible(*PreparedWeightIdentityContract::fromBinding(changed)));
 changed.cpu_joint_source=foreign;EXPECT_FALSE(identity->compatible(*PreparedWeightIdentityContract::fromBinding(changed)));
 changed.cpu_joint_source.reset();EXPECT_FALSE(identity->compatible(*PreparedWeightIdentityContract::fromBinding(changed)));
 auto missing=sources;missing.pop_back();
 EXPECT_THROW(registry.acquireStaticProjectionLayer(scope,selected->ownership(),missing),std::invalid_argument);
 auto duplicate=sources;duplicate.back()=duplicate.front();
 EXPECT_THROW(registry.acquireStaticProjectionLayer(scope,selected->ownership(),duplicate),std::invalid_argument);
 auto wrong=sources;wrong.front().identity.expert=0;
 EXPECT_THROW(registry.acquireStaticProjectionLayer(scope,selected->ownership(),wrong),std::invalid_argument);
 EXPECT_FALSE(registry.acquireStaticProjectionLayer(scope,selected->ownership(),sources));
 sources.registerInto(registry,0,1);
 EXPECT_THROW(registry.acquireCPUJointLayer(sources.front().identity.model_id,selected->participant(),7),std::logic_error);
}

/** @test Source-free compilation cannot manufacture a full expert for a CPU shard cohort. */
TEST(CPUJointPreparedSource, SourceFreePreparationRejectsCPUJointMode) {
 using namespace llaminar2;
 const auto original=std::make_shared<const MoERoutedExpertPlacementPlan>(jointSourcePlan());
 const auto runtime=resolveMoEExpertOverlayRuntimePlan(original,{.current_world_rank=0});ASSERT_TRUE(runtime);
 EXPECT_THROW(MoEExpertOverlayPreparationPlan::build(*runtime),std::invalid_argument);
}

/** @test The actual expert factory/store/registry retain one original allocated CPU engine after store clearing. */
TEST(CPUJointPreparedSource, ActualPreparedExpertHandleSurvivesStoreRetirement) {
 using namespace llaminar2;
 const auto selected=localJointSelection(0);auto sources=jointSources(selected);auto binding=sources.front();
 const auto memory=sources.actor->memory;auto& store=*sources.actor->manager->preparedWeightStore();
 auto engine=store.gemmKernelLifetime(*binding.prepared);ASSERT_TRUE(engine);
 MoEOverlayPreparedWeightSource prepared;std::string diagnostic;
 ASSERT_TRUE(resolveMoEOverlayPreparedWeightSource(engine,DeviceId::cpu(),prepared,&diagnostic))<<diagnostic;
 EXPECT_EQ(prepared.format,ExpertWeightFormat::floating(TensorType::FP32));
 EXPECT_EQ(engine->get_n(),int(binding.tensor->shape()[0]));EXPECT_EQ(engine->get_k(),int(binding.tensor->shape()[1]));
 ExpertGemmRegistry registry;sources.registerInto(registry,0,1);auto* raw=engine.get();
 store.clear();sources.actor->manager->expertGemmRegistry().clear();sources.actor->manager.reset();sources.clear();
 EXPECT_EQ(engine.get(),raw);EXPECT_TRUE(engine->weights_converted());EXPECT_NE(binding.tensor->raw_data(),nullptr);
 registry.clear();binding={};engine.reset();prepared={};
 EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);

}

namespace {
/** @brief Serialize one native scalar without changing the GGUF payload representation. */
template<class T> void jointScalar(std::ofstream& stream,T value) {
 stream.write(reinterpret_cast<const char*>(&value),sizeof(value));
}
/** @brief Serialize a GGUF length-delimited source identity. */
void jointString(std::ofstream& stream,const std::string& value) {
 jointScalar(stream,uint64_t(value.size()));stream.write(value.data(),std::streamsize(value.size()));
}
/** @brief Observe the actual original staging lease and inject failure only after real native loading. */
class JointWindowObservedLoader final:public llaminar2::ModelLoader {
public:
 using ModelLoader::ModelLoader;
 size_t calls=0,fail_on_call=0,highwater=0;
 std::shared_ptr<llaminar2::PhysicalMemoryAuthority> observed_memory;
 /** @brief Observe the SAME common reader after real I/O; never return fake source ownership. */
 void readNativeWeightSourceWindow(const std::string& name,const llaminar2::NativeWeightSourceWindow& window,
  std::span<std::byte> bytes) override {
  ModelLoader::readNativeWeightSourceWindow(name,window,bytes);
  if(observed_memory)highwater=std::max(highwater,observed_memory->claimedBytes(llaminar2::DeviceId::cpu(),
   llaminar2::PhysicalMemoryOwner::WeightLoadStaging,llaminar2::PhysicalMemoryMaterializationKind::NewAllocation));
  ++calls;if(calls==fail_on_call)throw std::runtime_error("Injected failure after actual native source window load");
 }

};
/** @brief Own only this fixture's generated source, removed after loader/mapping retirement. */
struct JointOriginalFile {
 inline static std::atomic<uint64_t> next_id{0};
 JointOriginalFile()=default;
 JointOriginalFile(const JointOriginalFile&)=delete;
 JointOriginalFile& operator=(const JointOriginalFile&)=delete;
 // Several live source contexts can share a process. Each owns only its
 // unique file, so retiring an older fixture cannot unlink a newer source.
 std::filesystem::path path=std::filesystem::temp_directory_path()/
  ("llaminar_cpu_joint_original_"+std::to_string(getpid())+"_"+
   std::to_string(next_id.fetch_add(1,std::memory_order_relaxed))+".gguf");
 ~JointOriginalFile(){std::error_code error;std::filesystem::remove(path,error);}
};
}
/** @test Actual GGUF metadata/loader/frozen materializer retain the one original map and exact selected bytes. */
TEST(CPUJointPreparedSource, ActualLoaderRetainsOriginalMapAndSelectedNativeBytes) {
 using namespace llaminar2;JointOriginalFile file;
 constexpr size_t H=64,F=80,local_elements=H*F/2,direct_cap=32768;
 struct OriginalStartupBudget {
  int previous=mutableDebugEnv().rocm.repack_budget_mb;
  OriginalStartupBudget(){mutableDebugEnv().rocm.repack_budget_mb=1;}
  ~OriginalStartupBudget(){mutableDebugEnv().rocm.repack_budget_mb=previous;}
 } startup_budget;
 {
  std::ofstream stream(file.path,std::ios::binary|std::ios::trunc);ASSERT_TRUE(stream);
  stream.write("GGUF",4);jointScalar(stream,uint32_t{3});jointScalar(stream,uint64_t{3});jointScalar(stream,uint64_t{0});
  for(int role=0;role<3;++role) {
   const std::string suffix=role==0?"gate":role==1?"up":"down";
   jointString(stream,"blk.7.ffn_"+suffix+"_exps.weight");jointScalar(stream,uint32_t{3});
   jointScalar(stream,uint64_t(role==2?F:H));jointScalar(stream,uint64_t(role==2?H:F));
   jointScalar(stream,uint64_t{512});jointScalar(stream,uint32_t{0});
   jointScalar(stream,uint64_t(role)*512*F*H*sizeof(float));
  }
  const size_t end=size_t(stream.tellp()),aligned=(end+31)/32*32;
  std::vector<char> padding(aligned-end,0);stream.write(padding.data(),std::streamsize(padding.size()));
  for(int role=0;role<3;++role)for(int expert=0;expert<512;++expert)
   for(int i=0;i<int(F*H);++i)jointScalar(stream,float(expert));
  ASSERT_TRUE(stream);
 }
 const auto mpi=std::make_shared<MPIContext>(0,1,MPI_COMM_NULL);TensorFactory factory(*mpi);JointWindowObservedLoader loader(&factory);
 ASSERT_TRUE(loader.loadModel(file.path.string()));
 auto sparse_plan=test::localCPUJointSourceTestPlan();
 for(size_t expert=0;expert<512;++expert)sparse_plan.placements.front().routed_expert_tier[expert]=expert%8==0?0:1;
 const auto runtime=resolveMoEExpertOverlayRuntimePlan(std::make_shared<const MoERoutedExpertPlacementPlan>(sparse_plan),
    {.current_world_rank=0});ASSERT_TRUE(runtime);
 const auto prepared=MoEExpertOverlayPreparationPlan::build(*runtime,loader);ASSERT_TRUE(prepared.cpuJointOwnerMap());
 const auto local=prepared.filteredForDevice(DeviceId::cpu());
 EXPECT_EQ(local.cpuJointOwnerMap(),prepared.cpuJointOwnerMap());
 ASSERT_EQ(local.requests().size(),448u*3);
 const auto selected=local.requests().front().cpu_joint_source;ASSERT_TRUE(selected);
 EXPECT_EQ(selected->cohort(),prepared.cpuJointOwnerMap()->placementFor(7,1)->shard_cohort);
 EXPECT_EQ(local.cpuShardSourceFor(7,selected->participant()),selected);
 const auto plan=local.sourceWeightPlan(loader,ModelContextId{175});ASSERT_EQ(plan.requirements().size(),3u);
 // Budget original raw staging and final prepared payload before any source load.
 size_t routed_bytes=0;
 for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown}) {
  const auto projection=selected->ownership().projection(role);
  routed_bytes+=448*resolveExpertPreparedProjectionMemoryGeometry(projection.rows,projection.columns,
   ExpertWeightFormat::floating(TensorType::FP32)).cpu_bytes;
 }
 const auto placement=CPUStoragePlacement::onNode(selected->participant().address.numa_node);
 const auto original=loader.nativeWeightSourceGeometry("blk.7.ffn_gate_exps.weight");
 const auto projected=projectNativeWeightSource(plan.requirements().front(),original);
 ASSERT_EQ(projected.size(),448u);
 const auto input_quote=test::nativeReadRequestForTest(original,projected.front().range);
 const auto staging=resolveNativeWeightLoadStaging(input_quote,placement,direct_cap);
 PhysicalMemoryPlanBuilder direct_budget;
 direct_budget.add({0,DeviceId::cpu(),staging.physical_bytes,staging.physical_bytes},
  PhysicalMemoryOwner::WeightLoadStaging,staging.physical_bytes);
 const auto direct_memory=std::make_shared<PhysicalMemoryAuthority>(
  std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(direct_budget.build()),0);
 loader.observed_memory=direct_memory;
 for(const auto& projection:projected) {
  test::visitNativeSourceForTest(loader,"blk.7.ffn_gate_exps.weight",projection.range,input_quote,placement,
   direct_memory,DeviceId::cpu(),direct_cap,[&](const auto& window,auto bytes,auto) {
    ASSERT_EQ(bytes.size()%sizeof(float),0u);
    const auto* values=reinterpret_cast<const float*>(bytes.data());
    for(size_t i=0;i<bytes.size()/sizeof(float);++i)EXPECT_EQ(values[i],float(window.original_expert));
   });
  EXPECT_EQ(direct_memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
   PhysicalMemoryMaterializationKind::NewAllocation),0u);
 }
 // Native floating consumers read bounded windows directly into the final
 // admitted tensor: there is no transient input arena to reserve or retain.
 ASSERT_FALSE(quoteCPUWeightPreparation(TensorType::FP32,F/2,H,WeightPrecision::NATIVE,CPUWeightConsumer::PreparedGemm));
 PhysicalMemoryPlanBuilder budget;
 budget.add({0,DeviceId::cpu(),routed_bytes,routed_bytes},PhysicalMemoryOwner::RoutedExpertWeights,routed_bytes);
 const auto memory=std::make_shared<PhysicalMemoryAuthority>(
  std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
 loader.observed_memory=memory;
 WeightManager manager(loader,mpi);manager.installPhysicalMemoryAuthority(memory);
 const auto before_calls=loader.calls;
 const auto frozen=manager.materialize(plan);EXPECT_GT(loader.calls-before_calls,3u);ASSERT_EQ(frozen.bindings().size(),448u*3);
 const auto matrices=frozen.projectedExpertMatrices();ASSERT_EQ(matrices.bindings().size(),448u*3);
 for(const auto& source:matrices.bindings()) {
  EXPECT_EQ(source.cpu_joint_source,selected);ASSERT_TRUE(source.native_source);EXPECT_FALSE(source.projection_source_owner);
  EXPECT_EQ(source.projection_ownership->geometry().experts,512);
  EXPECT_EQ(source.tensor->shape().size(),2u);
  EXPECT_FALSE(source.tensor->is_raw_data_released());ASSERT_NE(source.tensor->raw_data(),nullptr);
  EXPECT_EQ(source.native_source->range.original_expert,size_t(source.identity.expert));
  EXPECT_EQ(static_cast<const float*>(source.tensor->raw_data())[0],float(source.identity.expert));
 }
 EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);
 const FrozenModelWeightSet aggregate(frozen.strategy(),frozen.bindings(),PhysicalMemoryOwner::PrimaryModelWeights);
 const auto original_preparation=std::make_shared<const MoEExpertOverlayPreparationPlan>(prepared);
 ASSERT_TRUE(manager.prepareMoEExpertOverlayWeights(*runtime,DeviceId::cpu(),&aggregate,nullptr,
  PreparedWeightAdmission::AllocateCompleteSet,original_preparation));
 EXPECT_EQ(aggregate.physicalMemoryOwner(),PhysicalMemoryOwner::PrimaryModelWeights);
 EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);
 EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
  PhysicalMemoryMaterializationKind::NewAllocation),routed_bytes);
 const auto lease=manager.expertGemmRegistry().acquireCPUJointLayer(ModelContextId{175},selected->participant(),7);
 ASSERT_TRUE(lease);EXPECT_EQ(lease->cpu_joint_source,selected);
 // Copying a source-bound prepared lease retains the actual engines, not raw
 // source windows. Clearing both producer registries cannot invalidate it.
 auto copied=*lease;ASSERT_TRUE(copied.experts[1].complete());
 auto* engine=copied.experts[1].gate.get();
 auto input=llaminar2::test::TestTensorFactory::createFP32Zeros({size_t(engine->get_k()),1});
 auto output=llaminar2::test::TestTensorFactory::createFP32Zeros({size_t(engine->get_n()),1});
 std::fill_n(input->mutable_data(),engine->get_k(),1.f);
 const auto requirements=llaminar2::test::cpuProjectionTestRequirements(1,{engine});
 std::unique_ptr<llaminar2::test::CPUProjectionTestWorkspace> workspace;
 if(requirements.total_bytes_with_alignment())workspace=std::make_unique<llaminar2::test::CPUProjectionTestWorkspace>(requirements);
 const auto run=[&] {
  engine->multiply_tensor(input.get(),output.get(),1,engine->get_n(),engine->get_k(),true,
   1.f,0.f,nullptr,mpi.get(),-1,workspace?workspace->get():nullptr);
  for(int row=0;row<engine->get_n();++row)EXPECT_EQ(static_cast<const float*>(output->raw_data())[row],float(engine->get_k()));
 };
 run();manager.expertGemmRegistry().clear();manager.preparedWeightStore()->clear();run();
 EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);

 // A different existing startup cap changes only physical batching, never
 // logical matrix identity or the prepared arithmetic result.
 mutableDebugEnv().rocm.repack_budget_mb=2;
 const auto wider_memory=std::make_shared<PhysicalMemoryAuthority>(
  std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
 loader.observed_memory=wider_memory;
 {
  WeightManager second(loader,mpi);second.installPhysicalMemoryAuthority(wider_memory);
  const auto second_frozen=second.materialize(plan);
  ASSERT_EQ(second_frozen.bindings().size(),frozen.bindings().size());
  for(size_t i=0;i<frozen.bindings().size();++i) {
   EXPECT_EQ(second_frozen.bindings()[i].binding_id,frozen.bindings()[i].binding_id);
   EXPECT_EQ(second_frozen.bindings()[i].identity,frozen.bindings()[i].identity);
   EXPECT_FALSE(second_frozen.bindings()[i].tensor->is_raw_data_released());
   EXPECT_FALSE(second_frozen.bindings()[i].projection_source_owner);
  }
  const auto second_lease=second.expertGemmRegistry().acquireCPUJointLayer(ModelContextId{175},selected->participant(),7);
  ASSERT_TRUE(second_lease);engine=second_lease->experts[1].gate.get();run();
  second.expertGemmRegistry().clear();second.preparedWeightStore()->clear();run();
 }
 EXPECT_EQ(wider_memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);
 EXPECT_EQ(wider_memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);
 mutableDebugEnv().rocm.repack_budget_mb=1;
 const auto failed_memory=std::make_shared<PhysicalMemoryAuthority>(
  std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
 loader.observed_memory=failed_memory;loader.fail_on_call=loader.calls+2;
 {
  WeightManager failing(loader,mpi);failing.installPhysicalMemoryAuthority(failed_memory);
  EXPECT_THROW(failing.materialize(plan),std::runtime_error);
  EXPECT_EQ(failed_memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
   PhysicalMemoryMaterializationKind::NewAllocation),0u);
  EXPECT_THROW(failing.expertGemmRegistry().acquireCPUJointLayer(ModelContextId{175},selected->participant(),7),
   std::logic_error);
 }
 EXPECT_EQ(failed_memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);
 EXPECT_EQ(failed_memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
  PhysicalMemoryMaterializationKind::NewAllocation),0u);
 EXPECT_EQ(loader.highwater,staging.physical_bytes);


}

/** @test Physical windows retain sparse original IDs and never reinterpret their ordinal as a file ID. */
TEST(CPUJointPreparedSource, BoundedWindowsRetainSparseOriginalIdsAndNativeCuts) {
 using namespace llaminar2;
 auto plan=jointSourcePlan();
 auto& tiers=plan.placements.front().routed_expert_tier;
 for(size_t expert=0;expert<tiers.size();++expert)tiers[expert]=expert%8==0?0:1;
 const auto original=MoEExpertOwnerMap::buildShardPlacement(plan,{jointManifest()});
 const auto participant=std::find_if(original.participants().begin(),original.participants().end(),
  [](const auto& p){return p.domain_name=="cpu_joint" && p.domain_participant_index==0;});
 ASSERT_NE(participant,original.participants().end());
 const auto selected=CPUExpertShardSourceSelection::fromOriginalPlacement(original,7,participant->participant_id);
 ASSERT_EQ(selected->expertIds().size(),448u);
 EXPECT_EQ(selected->ownership().geometry().experts,512);
 for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown}) {
  const auto extent=selected->ownership().projection(role);
  const auto geometry=NativeWeightSourceGeometry{{size_t(extent.source_columns),size_t(extent.source_rows),512},TensorType::FP32,1,sizeof(float)};
  const auto name=std::string("blk.7.ffn_")+(role==WeightRole::MoEExpertGate?"gate":role==WeightRole::MoEExpertUp?"up":"down")+"_exps.weight";
  const auto request=test::nativeJointRequirementForTest(selected,role,name);
  const auto projections=projectNativeWeightSource(request,geometry);ASSERT_EQ(projections.size(),448u);
  size_t seen=0;
  for(size_t begin=0;begin<448;begin+=17)for(size_t ordinal=begin;ordinal<std::min(begin+17,size_t{448});++ordinal) {
   const auto& range=projections[ordinal].range;
   EXPECT_EQ(range.original_expert,size_t(selected->expertIds()[ordinal]));EXPECT_NE(range.original_expert%8,0u);
   EXPECT_EQ(range.rows,size_t(extent.rows));EXPECT_EQ(range.columns,size_t(extent.columns));
   EXPECT_EQ(range.first_row,request.slice.row_start);EXPECT_EQ(range.first_column,request.slice.col_start);++seen;
  }
  EXPECT_EQ(seen,448u);
  auto changed=request;changed.slice.expert_ids.push_back(512);
  EXPECT_THROW(projectNativeWeightSource(changed,geometry),std::invalid_argument);
 }
}
