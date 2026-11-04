/**
 * @file Test__MoEOverlayCPUCohortFabric.cpp
 * @brief Genuine two-socket whole/cohort physical staging and abort.
 *
 * One original GGUF/context/preparation supplies all real prepared banks and
 * PMA roots. Polling gates delay/fail an actual piece, retaining its copy, ACK
 * and abort/drain. No global candidate publication or GPU actor is invented.
 */
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "execution/moe/MoEOverlayPhysicalLaneDemand.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "execution/moe/MoEOverlayCapacityResolver.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "loaders/ModelContext.h"
#include "loaders/WeightManager.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/NativeWeightSourceRange.h"
#include "backends/HostMemoryCapacity.h"
#include "backends/BackendManager.h"
#include "utils/NodeTopology.h"
#include "utils/NUMATopology.h"
#include "utils/MPIContext.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/ServingGGUFFixturePayload.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include "../../utils/TestTensorFactory.h"
#include "../../fixtures/collective/CPUExpertDomainSocketMembership.h"
#include "../../fixtures/moe/MoEOverlayJointPlacementCandidateTestAccess.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test {
namespace {
constexpr int kExperts=4,kH=256,kF=96;
constexpr size_t kStaging=4093;
using Progress=MoEOverlayResidencyWaveProgress;
/** @return Same-host distinct physical socket observations in actual world-rank order. */
std::vector<CPUExpertSocketEndpoint> originalFabricSockets(const MPIContext& mpi) {
    if(mpi.world_size()!=2)throw std::invalid_argument("CPU cohort fabric requires two real MPI ranks");
    const auto affinity=NUMATopology::detectLocalNUMANode();
    if(!affinity.detection_succeeded || affinity.local_numa_node<0)
        throw std::runtime_error("CPU cohort fabric rank has no original NUMA affinity");
    const auto topology=NodeTopology::detect();
    const auto node=std::find_if(topology.numaNodes().begin(),topology.numaNodes().end(),
        [&](const auto& item){return item.node_id==affinity.local_numa_node;});
    if(node==topology.numaNodes().end() || node->socket_id<0)
        throw std::runtime_error("CPU cohort fabric rank has no original physical socket");
    std::array<int,2> local{affinity.local_numa_node,node->socket_id};std::array<int,4> all{};
    if(MPI_Allgather(local.data(),2,MPI_INT,all.data(),2,MPI_INT,MPI_COMM_WORLD)!=MPI_SUCCESS)
        throw std::runtime_error("CPU cohort fabric socket observation failed");
    std::array<char,MPI_MAX_PROCESSOR_NAME> host{};int length=0;
    if(MPI_Get_processor_name(host.data(),&length)!=MPI_SUCCESS)
        throw std::runtime_error("CPU cohort fabric hostname observation failed");
    std::array<char,2*MPI_MAX_PROCESSOR_NAME> hosts{};
    if(MPI_Allgather(host.data(),MPI_MAX_PROCESSOR_NAME,MPI_CHAR,hosts.data(),MPI_MAX_PROCESSOR_NAME,
        MPI_CHAR,MPI_COMM_WORLD)!=MPI_SUCCESS || std::string(hosts.data())!=std::string(hosts.data()+MPI_MAX_PROCESSOR_NAME))
        throw std::runtime_error("CPU cohort fabric requires original same-host endpoints");
    std::vector<CPUExpertSocketEndpoint> result{{0,all[0],all[1]},{1,all[2],all[3]}};
    if(!validCPUExpertSocketMembership(result,2))
        throw std::runtime_error("CPU cohort fabric requires one actual MPI rank per distinct physical socket");
    return result;
}
/** @return Fixed admitted tiers and cuts, never fabricated rank/socket execution. */
MoERoutedExpertPlacementPlan fabricOriginalPlan(const std::vector<CPUExpertSocketEndpoint>& sockets) {
    MoERoutedExpertPlacementPlan plan;
    plan.enabled=true;plan.first_model_layer=0;plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
    plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain="whole";
    RoutedExpertDomain whole;
    whole.name="whole";whole.scope=ExecutionDomainScope::RANK_LOCAL;
    whole.backend=CollectiveBackendType::UPI;whole.owner_rank=0;
    whole.participants={GlobalDeviceAddress::cpu(sockets.at(0).numa_node)};
    whole.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
    RoutedExpertDomain cohort;
    cohort.name="cohort";cohort.scope=ExecutionDomainScope::NODE_LOCAL;
    cohort.backend=CollectiveBackendType::UPI;cohort.owner_rank=-1;
    cohort.participants={GlobalDeviceAddress::cpu(sockets.at(0).numa_node),GlobalDeviceAddress::cpu(sockets.at(1).numa_node)};
    cohort.world_ranks={0,1};cohort.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
    cohort.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
    cohort.routed_decode_assignment_policy=cohort.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
    plan.domains={whole,cohort};
    plan.routed_tiers={{.name="whole",.domain="whole",.priority=0,.max_experts_per_layer=2},
        {.name="cohort",.domain="cohort",.priority=1,.max_experts_per_layer=2,.fallback=true}};
    plan.placements={{.layer=0,.routed_expert_tier={0,0,1,1}}};
    plan.residency_policy=RoutedExpertResidencyPolicy::StaticById;
    plan.authority_execution=resolveMoEOverlayAuthorityExecutionKind(plan);
    return plan;
}
/** @brief Real original source and physical owners, destroyed after every staged operation drains. */
struct FabricOriginal {
    PlanningGGUFFixture file{true,false,GGUFTensorType::F32,kF,std::nullopt,8,
        PlanningGGUFFixture::Geometry::Directory,1,kExperts};
    std::shared_ptr<MPIContext> mpi;
    std::shared_ptr<TensorFactory> tensors;
    std::shared_ptr<ModelContext> context;
    std::shared_ptr<const MoERoutedExpertPlacementPlan> plan;
    std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation;
    std::shared_ptr<MoEOverlayResidencyAuthority> authority;
    std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<MoEOverlayMPIRemoteProjectionTransport> remote;
    std::shared_ptr<MoEOverlayPhysicalResidencyFabric> fabric;
    std::vector<MoEOverlayLayerWeightManifest> manifests;
    int whole=-1,last_shard=-1;
    ModelContextId model;
    /** @brief Parse once, admit exact source/shadow/lane roots and publish genuine initial banks. */
    FabricOriginal() {
        int rank=-1,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
        mpi=std::make_shared<MPIContext>(rank,size,MPI_COMM_WORLD);
        const auto sockets=originalFabricSockets(*mpi);
        if(!hasCPUBackend())initCPUBackend(sockets.at(size_t(rank)).numa_node);
        populateServingGGUFFixture(file);
        tensors=std::make_shared<TensorFactory>(*mpi);
        ModelContextConfig source;source.mpi_ctx=mpi;source.factory=tensors.get();source.use_mmap=false;
        context=ModelContext::create(file.path(),source);
        if(!context)throw std::runtime_error("CPU cohort fabric original context did not load");
        model=ModelContextId{reinterpret_cast<uint64_t>(context.get())};
        plan=std::make_shared<const MoERoutedExpertPlacementPlan>(fabricOriginalPlan(sockets));
        const auto runtime=resolveMoEExpertOverlayRuntimePlan(plan,{.current_world_rank=rank,.validate_mvp_root_reachability=false});
        if(!runtime)throw std::runtime_error("CPU cohort fabric original runtime is absent");
        auto& loader=context->concreteLoader();
        preparation=std::make_shared<const MoEExpertOverlayPreparationPlan>(MoEExpertOverlayPreparationPlan::build(*runtime,loader));
        authority=std::make_shared<MoEOverlayResidencyAuthority>(MoEOverlayResidencyAuthority::Config{
            .initial_plan=*plan,.model_metadata={.num_layers=1,.num_experts=kExperts,.d_model=kH,
                .routed_intermediate_size=kF,.main_inference_layer_count=1},
            .projection_preparation=preparation,.maintenance_mode=MoERebalanceRuntimeMode::Off,
            .participant_rebalance_policy={.enabled=false}});
        const auto& topology=authority->snapshot()->placementOwnerMap();
        std::vector<int> local;
        for(const auto& member:topology.participants()) {
            if(member.domain_name=="whole")whole=member.participant_id;
            if(member.domain_name=="cohort" && member.domain_participant_index==1)last_shard=member.participant_id;
            if(member.world_rank_known && member.world_rank==rank)local.push_back(member.participant_id);
        }
        if(whole<0 || last_shard<0 || local.empty())throw std::runtime_error("CPU cohort fabric lost original targets");
        std::sort(local.begin(),local.end());
        auto loading=preparation->filteredForDevice(DeviceId::cpu()).sourceWeightPlan(loader,model);
        size_t resident_bytes=0;
        const auto placement=loader.nativeWeightSourcePlacement();
        for(const auto& request:loading.requirements())
            for(const auto& projection:projectNativeWeightSource(request,loader.nativeWeightSourceGeometry(request.source_name.empty()?request.canonical_name:request.source_name)))
                resident_bytes+=CPUStorageRoot<float>(projection.finalRows()*projection.finalColumns(),
                    PhysicalMemoryOwner::RoutedExpertWeights,DeviceId::cpu(),placement).physicalBytes();
        manifests=buildMoEOverlayLayerWeightManifestFromLoader(loader,1,kExperts);
        size_t shadow_bytes=0;
        for(int id:local) {
            const auto* member=topology.participantForId(id);
            const auto selected=member->domain_name=="cohort"?preparation->cpuShardSourceFor(0,*member):nullptr;
            const auto local_manifest=selected?selected->cohort()->localManifest(selected->participantIndex()):manifests.front();
            shadow_bytes+=MoEOverlayCapacityResolver::preparedFootprints({local_manifest}).front().cpu_shadow_bytes;
        }
        const MoEOverlayCapacityAdmissionPolicy storage{
            .migration_storage=MoEOverlayMigrationStorageKind::PhysicalResidencyFabric,
            .shadow_slots_per_endpoint_layer=1,.staging_capacity_bytes=kStaging,
            .maximum_concurrent_cycles=1,.maximum_execution_streams=1,.maximum_cycles_per_layer=1,
            .distributed_transport=true,.overlay_world_size=2};
        const auto staging=MoEOverlayCapacityAdmission::transferStagingBOM(*plan,2,storage,manifests,kExperts);
        size_t migration_bytes=0;
        for(const auto& charge:staging)if(charge.world_rank==rank) {
            if(!charge.device.is_cpu())throw std::runtime_error("CPU-only fabric unexpectedly priced a GPU allocator");
            migration_bytes+=charge.bytes;
        }
        const auto capacity=observeSystemMemoryCapacity();
        const PhysicalMemoryResource cpu{rank,DeviceId::cpu(),capacity.total_bytes,capacity.admission_available_bytes};
        PhysicalMemoryPlanBuilder budget;
        budget.add(cpu,PhysicalMemoryOwner::RoutedExpertWeights,resident_bytes);
        budget.add(cpu,PhysicalMemoryOwner::ExpertShadowSlots,shadow_bytes);
        budget.add(cpu,PhysicalMemoryOwner::ExpertMigrationStaging,migration_bytes);
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(budget.build()),rank);
        auto manager=context->concreteWeightManager();manager->installPhysicalMemoryAuthority(memory);
        const auto frozen=manager->materialize(loading);
        if(!manager->prepareMoEExpertOverlayWeights(*runtime,DeviceId::cpu(),&frozen,nullptr,
            PreparedWeightAdmission::AllocateCompleteSet,preparation))
            throw std::runtime_error("CPU cohort fabric original preparation did not complete");
        registry=std::make_shared<MoEOverlayParticipantResidencyRegistry>(MoEOverlayParticipantResidencyRegistry::Config{
            .local_participant_ids=local,.num_layers=1,.num_experts=kExperts,.initial_epoch=1,
            .retained_epoch_capacity=3,.projection_preparation=preparation,.original_model=model});
        std::string error;
        if(!registry->bindOriginalPreparedContext(context,preparation,&error) ||
            !registry->finalizeInitialBanksFromPreparedRegistry(manager->expertGemmRegistry(),&error))throw std::runtime_error(error);
        const auto demand=configuredMoEOverlayLaneDemand(*plan,manifests,kExperts,2,1,1);
        remote=std::make_shared<MoEOverlayMPIRemoteProjectionTransport>(MoEOverlayMPIRemoteProjectionTransport::Config{
            .mpi_context=mpi,.lane_budget=remoteMoEOverlayProjectionLaneBudget(demand),
            .staging_capacity_bytes=kStaging,.perf_device="native-cpu-cohort-fabric"});
        fabric=MoEOverlayPhysicalResidencyFabric::create({.memory_authority=memory,.registry=registry,
            .initial_snapshot=authority->snapshot(),.layer_weight_manifest=manifests,.remote_projection_transport=remote,
            .shadow_slots_per_endpoint_layer=1,.staging_capacity_bytes=kStaging,.maximum_concurrent_cycles=1,
            .maximum_execution_streams=1,.maximum_cycles_per_layer=1,.perf_device="native-cpu-cohort-fabric"});
        if(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
            PhysicalMemoryMaterializationKind::NewAllocation)!=resident_bytes ||
            memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExpertShadowSlots,
                PhysicalMemoryMaterializationKind::NewAllocation)!=shadow_bytes)
            throw std::runtime_error("CPU cohort fabric source/shadow roots differ from their original exact PMA declarations");
    }
    /** @return Original authority-authored swap, without publishing any candidate bank. */
    MoEOverlayResidencyTransaction transaction() const {
        auto placements=plan->placements;
        std::swap(placements.front().routed_expert_tier[0],placements.front().routed_expert_tier[2]);
        return MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(*authority,std::move(placements));
    }
};
/** @brief Own every real operation until normal completion or actual abort/drain. */
class ComponentWave {
public:
    MoEOverlayParticipantPreparedTransfers prepared;
    /** @brief Reserve full physical work, preserving the unchanged global authority. */
    ComponentWave(FabricOriginal& original,const MoEOverlayResidencyTransaction& transaction)
        :prepared(original.fabric->prepareTransfers(transaction,original.registry->localParticipantIds())) {
        if(prepared.status!=MoEOverlayResidencyStageStartStatus::Started) {
            const auto error=prepared.error;
            abortAndDrain(); // Failed prefixes are inert today, but still consume the real cleanup interface.
            throw std::runtime_error(error);
        }
    }
    /** @brief Drain before any assertion unwinds slots or MPI owners. */
    ~ComponentWave() {if(!drained_) {try{abortAndDrain();}catch(...){std::terminate();}}}
    /** @brief Delegate abort/drain to original operations, never manufacture readiness. */
    void abortAndDrain() {
        for(auto& migration:prepared.migrations)for(auto& piece:migration.operations)if(piece.operation)piece.operation->abort();
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;) {
            bool ready=true;std::string error;
            for(auto& migration:prepared.migrations)for(auto& piece:migration.operations) {
                if(!piece.operation)continue;
                const auto status=piece.operation->pollAbort(&error);
                if(status==Progress::Failed)throw std::runtime_error(error);
                ready=ready && status==Progress::Ready;
            }
            if(ready){drained_=true;return;}
            if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("CPU cohort fabric abort exceeded standard deadline");
            std::this_thread::yield();
        }
    }
private:bool drained_=false;
};
/** @brief Gate visibility of a real piece; source/copy/ACK/abort remain production-owned. */
class LastPieceGate final:public IMoEOverlayTierTransferOperation {
public:
    /** @brief Retain the actual fabric operation and all its source lifetimes. */
    explicit LastPieceGate(std::unique_ptr<IMoEOverlayTierTransferOperation> operation):operation_(std::move(operation)){}
    bool released=false,failed=false;
    /** @brief Inject only the deterministic polling edge before genuine progress. */
    Progress poll(std::string* error) noexcept override {
        if(failed){if(error)*error="component injected final-piece failure";return Progress::Failed;}
        return released?operation_->poll(error):Progress::Pending;
    }
    /** @brief Preserve actual asynchronous cancellation. */
    void abort() noexcept override {operation_->abort();}
    /** @brief Preserve real cancellation quiescence. */
    Progress pollAbort(std::string* error) noexcept override{return operation_->pollAbort(error);}
    /** @return Original completed physical measurement. */
    std::optional<ExpertTierProjectionTransferMeasurement> completedMeasurement() const noexcept override{return operation_->completedMeasurement();}
private:std::unique_ptr<IMoEOverlayTierTransferOperation> operation_;
};
/** @return Exact remote last-member Down piece, named identically on both actual ranks. */
LastPieceGate* delayLastPiece(ComponentWave& wave,const FabricOriginal& original,const MoEOverlayResidencyTransaction& transaction) {
    for(size_t index=0;index<wave.prepared.migrations.size();++index)
        if(transaction.migrations[index].expert_id==2)
            for(auto& piece:wave.prepared.migrations[index].operations)
                if(piece.projection==ExpertTierWeightProjection::Down && piece.source_participant==original.last_shard && piece.destination_participant==original.whole) {
                    auto gate=std::make_unique<LastPieceGate>(std::move(piece.operation));
                    auto* result=gate.get();piece.operation=std::move(gate);return result;
                }
    throw std::runtime_error("CPU cohort fabric omitted its original remote last piece");
}
/** @brief Poll actual pieces with symmetric test-only consensus, excluding the named gate. */
void pollComponent(ComponentWave& wave,const LastPieceGate* withheld=nullptr) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        int all_local=1;std::string error;
        for(auto& migration:wave.prepared.migrations)for(auto& piece:migration.operations) {
            if(piece.operation.get()==withheld)continue;
            const auto progress=piece.operation->poll(&error);
            if(progress==Progress::Failed)throw std::runtime_error(error);
            if(progress!=Progress::Ready)all_local=0;
        }
        int all_ranks=0;
        if(MPI_Allreduce(&all_local,&all_ranks,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD)!=MPI_SUCCESS)
            throw std::runtime_error("CPU cohort fabric component progress vote failed");
        if(all_ranks)return;
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("CPU cohort fabric staging exceeded standard deadline");
        std::this_thread::yield();
    }
}
/** @brief Actual previously observed terminal boundary, never an admission/readiness token. */
enum class ComponentTerminalQuery { TransferReady, AbortDrained };
/**
 * @brief Re-query only genuine terminal operations without consuming a new reservation.
 * @param wave Actual staged operations, already wholly Ready or abort-drained.
 * @param remote Original real transport supplying immutable traffic/counter observations.
 * @param phase Actual terminal boundary already observed through the original operation.
 */
void expectStableTerminalQueries(ComponentWave& wave,
    const MoEOverlayMPIRemoteProjectionTransport& remote,ComponentTerminalQuery phase)
{
    const auto before=remote.stats();
    for(int repeat=0;repeat<20;++repeat)
        for(auto& migration:wave.prepared.migrations)for(auto& piece:migration.operations) {
            ASSERT_TRUE(piece.operation);
            std::string error;
            EXPECT_EQ(phase==ComponentTerminalQuery::AbortDrained?piece.operation->pollAbort(&error):piece.operation->poll(&error),Progress::Ready)
                << error;
        }
    const auto after=remote.stats();
    EXPECT_EQ(after.bytes_sent,before.bytes_sent);EXPECT_EQ(after.bytes_received,before.bytes_received);
    EXPECT_EQ(after.operations_completed,before.operations_completed);
    EXPECT_EQ(after.operations_aborted,before.operations_aborted);
    EXPECT_EQ(after.protocol_failures,before.protocol_failures);
    EXPECT_EQ(after.active_mpi_requests,0u);
}
/** @return Independent originally authored float at exact original expert/N/K coordinates. */
float sourceValue(int expert,int row,int column,int columns,int rows) {
    const size_t index=(size_t(expert)*rows+size_t(row))*columns+size_t(column);
    return 0.001f*float(1+index%29)/29.0f;
}
/** @brief Check one final owner, original-coordinate bytes and real one-hot GEMM. */
void expectDestination(const FabricOriginal& original,const MoEOverlayResidencyTransaction& transaction,
    size_t migration_index,const MoEOverlayParticipantDestinationArrival& destination) {
    const auto& edge=transaction.migrations[migration_index];
    std::array<std::shared_ptr<ITensorGemm>,3> engines;
    std::array<MoEExpertProjectionOwnership::Projection,3> projections;
    std::string error;
    if(destination.arrival->isCPUJointFragment()) {
        MoEOverlayPreparedCPUFragmentPayload payload;
        ASSERT_TRUE(destination.arrival->completeCPUFragment(payload,&error))<<error;
        EXPECT_EQ(payload.model(),original.model);EXPECT_EQ(payload.preparation(),original.preparation);
        ASSERT_EQ(payload.cohort(),edge.destination.shard_cohort);
        const auto member=std::find_if(payload.cohort()->participants().begin(),payload.cohort()->participants().end(),
            [&](const auto& p){return p.participant_id==destination.participant_id;});
        ASSERT_NE(member,payload.cohort()->participants().end());
        const auto ordinal=size_t(member-payload.cohort()->participants().begin());
        engines={payload.gate(),payload.up(),payload.down()};
        const auto& owned=payload.cohort()->ownership(ordinal);
        projections={owned.projection(WeightRole::MoEExpertGate),owned.projection(WeightRole::MoEExpertUp),owned.projection(WeightRole::MoEExpertDown)};
        MoEOverlayPreparedExpertPayload invalid;
        EXPECT_FALSE(destination.arrival->completePayload(invalid,&error));
    } else {
        MoEOverlayPreparedExpertPayload payload;
        ASSERT_TRUE(destination.arrival->completePayload(payload,&error))<<error;
        engines={payload.gate(),payload.up(),payload.down()};
        for(size_t role=0;role<3;++role) {
            const auto& matrix=original.manifests.front().projections[role];
            projections[role]={MoEProjectionResidence::ExpertOwner,matrix.N,matrix.K,0,matrix.N,0,matrix.K};
        }
    }
    for(size_t role=0;role<3;++role) {
        const auto& engine=engines[role];ASSERT_TRUE(engine);
        ContiguousFloatingPointWeightDescriptor bytes;
        ASSERT_TRUE(engine->exportContiguousFloatingPointWeights(bytes));ASSERT_TRUE(bytes.valid());
        ASSERT_EQ(bytes.type,TensorType::FP32);const auto& owned=projections[role];
        EXPECT_EQ(bytes.n,owned.rows);EXPECT_EQ(bytes.k,owned.columns);
        const auto* values=static_cast<const float*>(bytes.data);
        for(int row=0;row<owned.rows;++row)for(int column=0;column<owned.columns;++column)
            ASSERT_EQ(std::bit_cast<uint32_t>(values[size_t(row)*owned.columns+column]),
                std::bit_cast<uint32_t>(sourceValue(edge.expert_id,owned.first_row+row,owned.first_column+column,
                    owned.source_columns,owned.source_rows)));
        auto input=TestTensorFactory::createFP32Zeros({1,size_t(bytes.k)});
        auto output=TestTensorFactory::createFP32Zeros({1,size_t(bytes.n)});
        const auto requirements=cpuProjectionTestRequirements(1,{engine.get()});
        ASSERT_EQ(requirements.total_bytes_with_alignment(),0u); // Native FP32 execution owns no transform scratch.
        for(int column:{0,bytes.k/2,bytes.k-1}) {
            std::fill_n(input->mutable_data(),bytes.k,0.f);input->mutable_data()[column]=1.f;
            ASSERT_TRUE(engine->multiply_tensor(input.get(),output.get(),1,bytes.n,bytes.k,true,1.f,0.f,nullptr,
                original.mpi.get(),-1,nullptr));
            for(int row=0;row<bytes.n;++row)
                EXPECT_EQ(std::bit_cast<uint32_t>(static_cast<const float*>(output->raw_data())[row]),
                    std::bit_cast<uint32_t>(values[size_t(row)*bytes.k+column]));
        }
    }
}
} // namespace
/** @test Genuine staged whole/cohort pieces withhold readiness until the last real piece completes. */
TEST(MoEOverlayCPUCohortFabric, RealTwoSocketStageDelaysFinalPieceAndExecutesExactDestinations) {
    FabricOriginal original;
    RecordProperty("physical_sockets",2);RecordProperty("mpi_ranks",2);
    const auto transaction=original.transaction();ASSERT_TRUE(transaction.valid());
    ASSERT_EQ(transaction.migrations.size(),2u);
    ComponentWave wave(original,transaction);auto* gate=delayLastPiece(wave,original,transaction);ASSERT_TRUE(gate);
    pollComponent(wave,gate);
    if(original.mpi->rank()==0)
        for(size_t index=0;index<wave.prepared.migrations.size();++index)if(transaction.migrations[index].expert_id==2)
            for(const auto& destination:wave.prepared.migrations[index].destination_arrivals) {
                MoEOverlayPreparedExpertPayload incomplete;std::string error;
                EXPECT_FALSE(destination.arrival->completePayload(incomplete,&error));
            }
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
    gate->released=true;pollComponent(wave);
    expectStableTerminalQueries(wave,*original.remote,ComponentTerminalQuery::TransferReady);
    for(size_t index=0;index<wave.prepared.migrations.size();++index)
        for(const auto& destination:wave.prepared.migrations[index].destination_arrivals)
            expectDestination(original,transaction,index,destination);
    const auto stats=original.remote->stats();
    const uint64_t expected_remote_bytes=uint64_t(3)*kH*kF*sizeof(float)/2;
    EXPECT_EQ(stats.bytes_sent,expected_remote_bytes);EXPECT_EQ(stats.bytes_received,expected_remote_bytes);
    EXPECT_EQ(stats.active_mpi_requests,0u);EXPECT_EQ(stats.inference_stream_waits,0u);
    EXPECT_EQ(stats.blocking_synchronizations,0u);
    wave.abortAndDrain();
    expectStableTerminalQueries(wave,*original.remote,ComponentTerminalQuery::AbortDrained);
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
    for(int id:original.registry->localParticipantIds())EXPECT_FALSE(original.registry->endpoint(id)->acquire(2));
}
/** @test Failed last-piece staging drains real operations before component source-node reclamation. */
TEST(MoEOverlayCPUCohortFabric, RealTwoSocketFailedFinalPieceDrainsAndRetainsOriginalSourceBank) {
    FabricOriginal original;const auto transaction=original.transaction();ASSERT_TRUE(transaction.valid());
    ComponentWave wave(original,transaction);auto* gate=delayLastPiece(wave,original,transaction);ASSERT_TRUE(gate);
    pollComponent(wave,gate);
    for(int id:original.registry->localParticipantIds())
        EXPECT_FALSE(original.registry->endpoint(id)->retiredEpochReaders(1)); // Still published, so not a retired-node observation.
    auto manager=original.context->concreteWeightManager();manager->clearCache();manager->preparedWeightStore()->clear();manager->expertGemmRegistry().clear();
    // Deliberate registry-lifetime adversary, never global publication or a
    // fabricated inference ticket. All actual source readers were acquired first.
    for(int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);(void)endpoint->retire(1);
        EXPECT_EQ(endpoint->retainedEpochCount(),0u); // Withdrawal is immediate, even with a real old reader.
        EXPECT_FALSE(endpoint->acquire(1)); // Observation must never resurrect source readiness.
        EXPECT_FALSE(endpoint->retiredEpochReaders(0));
        EXPECT_FALSE(endpoint->retiredEpochReaders(2)); // No candidate node was installed.
    }
    if(original.mpi->rank()==1) {
        const auto readers=original.registry->endpoint(original.last_shard)->retiredEpochReaders(1);
        ASSERT_TRUE(readers);EXPECT_EQ(*readers,1u); // The fabric's ONE original member borrow, not an engine ref.
    } else {
        EXPECT_FALSE(original.registry->endpoint(original.whole)->retiredEpochReaders(1));
    }
    gate->failed=true;std::string error;
    EXPECT_EQ(gate->poll(&error),Progress::Failed);EXPECT_NE(error.find("final-piece"),std::string::npos);
    wave.abortAndDrain();
    expectStableTerminalQueries(wave,*original.remote,ComponentTerminalQuery::AbortDrained);
    if(original.mpi->rank()==1) {
        const auto drained=original.registry->endpoint(original.last_shard)->retiredEpochReaders(1);
        ASSERT_TRUE(drained);EXPECT_EQ(*drained,0u); // Actual abort/drain released the original reader before reclamation.
    }
    for(int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);(void)endpoint->retire(1); // Existing maintenance sweep, no diagnostic mutation.
        EXPECT_EQ(endpoint->retainedEpochCount(),0u);EXPECT_FALSE(endpoint->acquire(2));
        EXPECT_FALSE(endpoint->retiredEpochReaders(1)); // The real retired node is now reclaimed.
    }
    EXPECT_EQ(original.remote->stats().active_mpi_requests,0u);
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
    wave.prepared={}; // Drop actual arrivals, operation callbacks and their engine owners after drain.
    original.fabric.reset();original.registry.reset();original.remote.reset();
    original.context.reset();manager.reset();
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExpertShadowSlots,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
/** @test A retained whole-wave reservation rejects new setup and aborts without starting any copy. */
TEST(MoEOverlayCPUCohortFabric, RealTwoSocketSetupFailureDrainsWithoutStartingPhysicalWork) {
    FabricOriginal original;const auto transaction=original.transaction();ASSERT_TRUE(transaction.valid());
    ComponentWave reserved(original,transaction);
    // Same admitted slots/lanes cannot serve two overlapping waves. The failed
    // constructor drains its real cleanup prefix; neither wave has been polled.
    EXPECT_THROW((void)ComponentWave(original,transaction),std::runtime_error);
    const auto before=original.remote->stats();
    EXPECT_EQ(before.bytes_sent,0u);EXPECT_EQ(before.bytes_received,0u);
    EXPECT_EQ(before.active_mpi_requests,0u);
    EXPECT_EQ(original.fabric->stats().active_cpu_copy_operations,0u);
    reserved.abortAndDrain();
    expectStableTerminalQueries(reserved,*original.remote,ComponentTerminalQuery::AbortDrained);
    const auto after=original.remote->stats();
    EXPECT_EQ(after.bytes_sent,0u);EXPECT_EQ(after.bytes_received,0u);
    EXPECT_EQ(after.active_mpi_requests,0u);
    for(int id:original.registry->localParticipantIds()) {
        EXPECT_TRUE(original.registry->endpoint(id)->acquire(1));
        EXPECT_FALSE(original.registry->endpoint(id)->acquire(2));
    }
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
}

} // namespace llaminar2::test
