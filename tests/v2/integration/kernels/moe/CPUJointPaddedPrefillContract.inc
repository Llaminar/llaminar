/**
 * @file CPUJointPaddedPrefillContract.inc
 * @brief Actual original-source rooted stage preflight with native mapped canonical tickets.
 *
 * Real CUDA/ROCm mappings are created by the public transfer owner. The CPU TP
 * context supplies only original cold membership: this test never executes a
 * collective, claims native TP readiness or launches a graph/model/kernel.
 */
#include "execution/compute_stages/stages/MoECPUExpertShardRouteStage.h"
#include "execution/local_execution/engine/PrefillGraphCache.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEOverlayCapacityResolver.h"
#include "loaders/WeightManager.h"
#include "kernels/cpu/gemm/CPUWeightPreparationGeometry.h"
#include "../../../utils/PlanningGGUFFixture.h"
#include "../../../utils/CPUJointSourceTestPlan.h"
#include "../../../utils/CPUForwardSamplingTestFixture.h"
namespace llaminar2::test { namespace {
/** @brief Borrow only actual original cohort metadata; no collective/native-ready result is supplied. */
class PaddedRootPlanningContext final:public ScriptedGlobalTPContext {
public:
 /** @param source Actual loader-derived original member and cohort.
  * @param backend Original declared CPU communication family.
  * @param domain Original declaration ordinal; no communicator is created. */
 explicit PaddedRootPlanningContext(std::shared_ptr<const CPUExpertShardSourceSelection> source,CollectiveBackendType backend,int domain):source_(std::move(source)),backend_(backend),domain_(domain) {
  for(const auto& member:source_->cohort()->participants())ranks_.push_back(member.world_rank);
 }
 /** @return Original declared cohort population. */
 int degree()const override{return int(ranks_.size());}
 /** @return Original source member ordinal. */
 int myIndex()const override{return int(source_->participantIndex());}
 /** @return Declared NODE_LOCAL expert scope, not native readiness. */
 TPScope scope()const override{return TPScope::NODE_LOCAL;}
 /** @return Original declared communication family; no backend is materialized. */
 CollectiveBackendType backend()const override{return backend_;}
 /** @return Original declared domain ordinal. */
 int domainId()const override{return domain_;}
 /** @brief Reject any accidental execution: this fixture owns cold metadata only. */
 bool gatherVariableFloatRecordsToRoot(const float*,size_t,float*,size_t,size_t,int,size_t&,
  const std::string&,std::span<RootedFloatRecordRange>)override {
  ++gathers_;throw std::logic_error("Planning-only context cannot execute a rooted collective");
 }
 /** @return Zero unless an invalid preflight unexpectedly executed a collective. */
 int gatherCalls()const noexcept{return gathers_;}
 /** @return Exact original CPU0/NUMA address. */
 GlobalDeviceAddress localDevice()const override{return source_->participant().address;}
 /** @return Exact original source rank order. */
 const std::vector<int>& worldRanks()const override{return ranks_;}
private:
 std::shared_ptr<const CPUExpertShardSourceSelection> source_;
 std::vector<int> ranks_;
 CollectiveBackendType backend_;int domain_;int gathers_=0;
};
/** @brief Construct real original CPU fragments and mapped publication, then check original device/cache contracts.
 * @param device Exact native mapped continuation endpoint.
 */
void rootedPaddedPrefillContract(DeviceId device) {
 constexpr int rows=64,hidden=256,experts=8,intermediate=16,top_k=2,slots=rows*top_k;
 if(!hasCPUBackend())initCPUBackend(-1);
 auto* backend=getBackendFor(device);ASSERT_NE(backend,nullptr);ASSERT_GT(backend->deviceCount(),device.ordinal);
 PlanningGGUFFixture file(true,false,GGUFTensorType::F32,intermediate,std::nullopt,8,
  PlanningGGUFFixture::Geometry::Directory,1);
 const auto mpi=std::make_shared<MPIContext>(0,1,MPI_COMM_NULL);TensorFactory factory(*mpi);ModelLoader loader(&factory);
 loader.setPayloadAccessPattern(ModelPayloadAccessPattern::SparseSelection);ASSERT_TRUE(loader.loadModel(file.path()));
 auto plan=cpuJointSourceTestPlan();plan.first_model_layer=0;
 plan.domains.front().participants={device.is_cuda()?GlobalDeviceAddress::cuda(device.ordinal):GlobalDeviceAddress::rocm(device.ordinal)};
 plan.domains.front().backend=device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
 plan.routed_tiers[0].max_experts_per_layer=2;plan.routed_tiers[1].max_experts_per_layer=6;
 plan.placements={{.layer=0,.routed_expert_tier={0,0,1,1,1,1,1,1}}};
 const auto runtime=resolveMoEExpertOverlayRuntimePlan(std::make_shared<const MoERoutedExpertPlacementPlan>(plan),
  {.current_world_rank=0,.validate_mvp_root_reachability=false});ASSERT_TRUE(runtime);
 const auto original=std::make_shared<const MoEExpertOverlayPreparationPlan>(MoEExpertOverlayPreparationPlan::build(*runtime,loader));
 const auto local=original->filteredForDevice(DeviceId::cpu());ASSERT_FALSE(local.requests().empty());
 const auto selected=local.requests().front().cpu_joint_source;ASSERT_TRUE(selected);
 const auto manifest=selected->cohort()->localManifest(selected->participantIndex());
 const auto prepared_bytes=6*MoEOverlayCapacityResolver::preparedFootprints({manifest}).front().cpu_live_bytes;
 size_t staging_bytes=0;
 for(const auto& projection:manifest.projections) {
  const auto type=projection.format.floatingTensorType();ASSERT_TRUE(type);
  const auto quote=quoteCPUWeightPreparation(*type,projection.N,projection.K,WeightPrecision::NATIVE,
   CPUWeightConsumer::PreparedGemm);
  if(quote)staging_bytes=std::max(staging_bytes,resolveNativeWeightLoadStaging(*quote,
   CPUStoragePlacement::onNode(selected->participant().address.numa_node),
   configuredNativeWeightLoadStagingBudgetBytes()).physical_bytes);
 }
 EXPECT_EQ(staging_bytes,0u); // Genuine final floating roots need no source shadow.
 CPUJointExpertRouteStorageLayout joint(selected->cohort(),selected->participantIndex(),rows,top_k);
 const auto layout=std::make_shared<const CPUSerialExpertStorageLayout>(DeviceId::cpu(),rows,hidden,top_k,
  std::vector<size_t>{rows},true,intermediate/2,experts,device,joint);
 const auto total=prepared_bytes+staging_bytes+layout->physicalBytes()+layout->workspaceBytes();
 PhysicalMemoryPlanBuilder budget;const PhysicalMemoryResource host{0,DeviceId::cpu(),total,total};
 budget.add(host,PhysicalMemoryOwner::RoutedExpertWeights,prepared_bytes).add(host,PhysicalMemoryOwner::WeightLoadStaging,staging_bytes)
  .add(host,PhysicalMemoryOwner::ActivationArena,layout->physicalBytes()).add(host,PhysicalMemoryOwner::ExecutionWorkspace,layout->workspaceBytes());
 const auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
 WeightManager manager(loader,mpi);manager.installPhysicalMemoryAuthority(memory);
 const ModelContextId model{454};const auto weights=manager.materialize(local.sourceWeightPlan(loader,model));
 auto residency=std::make_shared<MoEOverlayParticipantResidencyRegistry>(MoEOverlayParticipantResidencyRegistry::Config{
  .local_participant_ids={selected->participant().participant_id},.num_layers=1,.num_experts=experts,.initial_epoch=1,
  .projection_preparation=original,.first_model_layer=0,.original_model=model});
 std::string error;ASSERT_TRUE(residency->registerInitialCPUJointBank(manager.expertGemmRegistry(),model,
  selected->participant().participant_id,&error))<<error;
 const auto layer=residency->initialCPUJointLayer(selected->participant().participant_id,0);ASSERT_TRUE(layer);
 auto arena=std::make_shared<MoELocalExpertSerialBufferArena>(MoELocalExpertSerialBufferArena::Config{
  .device_id=DeviceId::cpu(),.row_capacity=rows,.row_capacity_buckets={rows},.d_model=hidden,.routing_top_k=top_k,
  .cpu_canonical_route_storage=MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
  .cpu_canonical_route_gpu_consumer=device,
  .cpu_grouped_scratch_storage=MoELocalExpertSerialBufferArena::CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
  .num_experts=experts,.expert_intermediate=intermediate/2,.logical_participant_id=selected->participant().participant_id,
  .debug_name="rooted_padded_contract",.cpu_storage_layout=layout,.physical_memory_authority=memory});
 const std::array<DeviceId,1> endpoints{device};auto metadata=TransferEngine::instance().createMappedHostArena(endpoints);
 auto ticket=std::make_shared<MoEOverlayCanonicalRouteReturnTicketStorage>();
 const auto publication=arena->mappedCPUCanonicalRoutes(device);
 ticket->bindFixedCapacity(0,slots,hidden,device,1,publication,metadata);ASSERT_TRUE(ticket->hasValidBoundIdentity());
 MoEOverlayCollectiveWorkspace packets({.max_rows=rows,.max_entries=slots,.d_model=hidden,.top_k=top_k,
  .device=DeviceId::cpu(),.return_layout=MoEOverlayReturnLayout::CanonicalExpertRoutes});
 auto input=packets.localExpertInput(0,selected->participant().participant_id);
 auto output=packets.localExpertOutput(0,selected->participant().participant_id);
 input.live_row_count=9;input.live_entry_count=9;input.residency_epoch=1;
 input.source_participant=0;input.target_participant=selected->participant().participant_id;
 for(int row=0;row<9;++row) {
  input.row_ids_host[row]=row;input.entry_offsets_host[row]=row;
  input.expert_ids_host[row]=selected->expertIds().front();input.route_weights_host[row]=1.f;
  input.original_route_slots_host[row]=row*top_k;input.compact_route_slots_host[row]=row*top_k;
 }
 input.entry_offsets_host[9]=9;
 const auto context=std::make_shared<PaddedRootPlanningContext>(selected,plan.domains.back().backend,int(plan.domains.size()-1));
 MoECPUExpertShardRouteStage::Params params{.original_source=selected,.tp=context.get(),
  .packed_records=arena->cpuJointPackedRecords().get(),.combined_routes=arena->cpuCanonicalRoutes().get(),
  .source_ranges=arena->cpuJointSourceRanges(),.participant_route_maps=arena->cpuJointRouteMaps(),
  .required_route_weights=arena->cpuJointRequiredRoutes(),.rows=rows,.top_k=top_k,.columns=hidden,.root_participant=0,
  .stage_name="rooted_padded_contract",.original_return_rows=&output,.original_return_route_capacity=slots,
  .original_sparse_routes=&input,.original_prepared_layer=layer,.original_tp_owner=context,
  .original_canonical_ticket=ticket,.original_continuation_device=device};
 auto stage=std::make_unique<MoECPUExpertShardRouteStage>(params);const auto* actual=stage.get();
 const auto foreign_ordinal=device.is_cuda()?DeviceId::cuda(device.ordinal+1):DeviceId::rocm(device.ordinal+1);
 const auto foreign_vendor=device.is_cuda()?DeviceId::rocm(device.ordinal):DeviceId::cuda(device.ordinal);
 EXPECT_TRUE(actual->acceptsSchedulingContext(DeviceId::cpu()));
 EXPECT_TRUE(actual->acceptsSchedulingContext(device));
 EXPECT_FALSE(actual->acceptsSchedulingContext(foreign_ordinal));
 EXPECT_FALSE(actual->acceptsSchedulingContext(foreign_vendor));
 EXPECT_FALSE(actual->acceptsSchedulingContext(DeviceId::invalid()));
 EXPECT_FALSE(actual->isGraphCapturable());EXPECT_TRUE(actual->isManualGraphBoundary());
 EXPECT_EQ(actual->manualGraphBoundaryScheduling(),ManualGraphBoundaryScheduling::ConcurrentTicketService);
 EXPECT_TRUE(actual->supportsPaddedPrefillGraphCapturePreflight());EXPECT_TRUE(actual->supportsPaddedPrefillRealLengthContract());
 ComputeGraph graph;graph.addNode("rooted",std::move(stage),DeviceId::cpu());
 PrefillGraphConfig config;config.enabled=true;config.buckets_enabled=true;config.minimum_padded_bucket_seq_len=64;
 PrefillGraphCache cache(config);PrefillGraphCacheKey key;key.device_id=device;key.seq_len=rows;
 const std::unordered_set<std::string> collectives{"rooted"};
 for(const auto mode:{PrefillGraphPreflightMode::ColdPaddedSupport,PrefillGraphPreflightMode::CaptureReady})
  EXPECT_EQ(cache.preflight(graph,key,&collectives,false,false,9,64,mode,false,true),PrefillGraphRejectReason::None);
 EXPECT_NE(cache.preflight(graph,key,&collectives,false,false,9,64,PrefillGraphPreflightMode::CaptureReady,false,false),
  PrefillGraphRejectReason::None);
 params.original_canonical_ticket.reset();params.original_preweighted_routes=static_cast<float*>(publication->mutableHostData());
 auto no_ticket=std::make_unique<MoECPUExpertShardRouteStage>(params);
 EXPECT_TRUE(no_ticket->acceptsSchedulingContext(DeviceId::cpu()));
 EXPECT_FALSE(no_ticket->acceptsSchedulingContext(device));
 EXPECT_FALSE(no_ticket->acceptsSchedulingContext(foreign_ordinal));
 EXPECT_FALSE(no_ticket->acceptsSchedulingContext(foreign_vendor));
 EXPECT_FALSE(no_ticket->acceptsSchedulingContext(DeviceId::invalid()));
 EXPECT_FALSE(no_ticket->isGraphCapturable());EXPECT_FALSE(no_ticket->supportsPaddedPrefillGraphCapturePreflight());
 EXPECT_FALSE(no_ticket->supportsPaddedPrefillRealLengthContract());
 ComputeGraph ordinary;ordinary.addNode("rooted",std::move(no_ticket),DeviceId::cpu());
 for(const auto mode:{PrefillGraphPreflightMode::ColdPaddedSupport,PrefillGraphPreflightMode::CaptureReady})
  EXPECT_NE(cache.preflight(ordinary,key,&collectives,false,false,9,64,mode,false,true),PrefillGraphRejectReason::None);
 EXPECT_EQ(context->communicator(),MPI_COMM_NULL);EXPECT_EQ(context->gatherCalls(),0);
}
}
#ifdef HAVE_CUDA
TEST(CPUJointPaddedPrefillContract, OriginalMappedTicketCUDA){rootedPaddedPrefillContract(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(CPUJointPaddedPrefillContract, OriginalMappedTicketROCm){rootedPaddedPrefillContract(DeviceId::rocm(0));}
#endif
}
