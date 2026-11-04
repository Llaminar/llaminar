/**
 * @file CPUJointPreparedComputeServiceContract.inc
 * @brief Original-source two-socket input-ready/start-release to canonical-output probe.
 *
 * This setup probe deliberately excludes serving dispatch and device ingress.
 * Actual context/Store/fragment banks and one original admitted native CPU domain
 * supply the production producer. The independent FP32 oracle is terminal test
 * arithmetic only; it is never an inference or calibration execution path.
 */
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/CPUExpertShardSourceSelection.h"
#include "backends/BackendManager.h"
#include "loaders/WeightManager.h"
#include "../../utils/CPUJointSourceTestPlan.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/ServingGGUFFixturePayload.h"
#include "loaders/ModelContext.h"
#include "tensors/NativeWeightSourceRange.h"
#include "kernels/cpu/gemm/CPUProjectionWorkspaceContract.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
/** @brief Preserve one original failure instead of allowing a peer to enter another collective.
 * @param condition Actual source/lifecycle predicate.
 * @param diagnostic Original rejected condition. */
void requirePreparedCompute(bool condition,const char* diagnostic)
{if(!condition)throw std::runtime_error(diagnostic);}
/** @brief Abort only the original native job before admitted domain owners unwind.
 * @param diagnostic Original producer failure, never a synthetic completion. */
[[noreturn]] void abortPreparedCompute(const char* diagnostic)
{std::fprintf(stderr,"Prepared CPU target probe failed: %s\n",diagnostic);MPI_Abort(MPI_COMM_WORLD,1);std::abort();}
/** @brief Independent terminal full-source FP32 FFN for this tiny authored file.
 * @param row Same deterministic input row passed by the setup producer.
 * @param column Original output column.
 * @param expert Original logical expert, never a compact slot.
 * @return FP32 unweighted output over all 64 original intermediate columns, independently accumulated.
 */
float preparedComputeOracle(int row,int column,int expert)
{
    constexpr int h=256,f=64;
    const auto weight=[](size_t index){return 0.001f*static_cast<float>(1+index%29)/29.0f;};
    float result=0;
    for(int n=0;n<f;++n) {
        float gate=0;
        for(int k=0;k<h;++k) {
            const float input=0.0001f*static_cast<float>((k%31*13+row*7)%31-15);
            gate=std::fma(input,weight((static_cast<size_t>(expert)*f+n)*h+k),gate);
        }
        const float activated=(gate/(1.0f+std::exp(-gate)))*gate;
        result=std::fma(activated,weight((static_cast<size_t>(expert)*h+column)*f+n),result);
    }
    return result;
}

/** @test Genuine source/bank/domain owners, all three phase families and root-only complete output. */
TEST_F(CPUExpertDomainNative, PreparedCPUJointComputeRootWallAndRetirement)
{
    // Setup is rank-local until it enters the original native collectives.
    // Any setup exception must abort the same MPI job before a peer advances.
    try {
    using Service=MoEOverlayCPUServiceMeasurement;
    constexpr int h=256,f=64,e=4,selected_expert=2;
    int rank=-1,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    requirePreparedCompute(size==2,"prepared-compute requires two original physical CPU ranks");
    const auto world=MPIContextFactory::global();const auto inventory=world->clusterInventory();
    requirePreparedCompute(inventory && inventory->ranks.size()==2,"original prepared-compute inventory");
    const auto affinity=NUMATopology::detectLocalNUMANode();
    requirePreparedCompute(affinity.detection_succeeded && affinity.local_numa_node>=0,"original member NUMA binding");
    const auto topology=NodeTopology::detect();
    const auto node=std::find_if(topology.numaNodes().begin(),topology.numaNodes().end(),
        [&](const auto& item){return item.node_id==affinity.local_numa_node;});
    requirePreparedCompute(node!=topology.numaNodes().end() && node->socket_id>=0,"original physical socket");
    std::array<int,2> nodes{},sockets{};
    requirePreparedCompute(MPI_Allgather(&affinity.local_numa_node,1,MPI_INT,nodes.data(),1,MPI_INT,MPI_COMM_WORLD)==MPI_SUCCESS &&
        MPI_Allgather(&node->socket_id,1,MPI_INT,sockets.data(),1,MPI_INT,MPI_COMM_WORLD)==MPI_SUCCESS,"original socket exchange");
    const std::array<test::CPUExpertSocketEndpoint,2> endpoints{{{0,nodes[0],sockets[0]},{1,nodes[1],sockets[1]}}};
    requirePreparedCompute(test::validCPUExpertSocketMembership(endpoints,2),"distinct physical CPU sockets");
    requirePreparedCompute(inventory->ranks[0].hostname==inventory->ranks[1].hostname,"same original node");
    initCPUBackend(nodes[rank]);
    // Repeat the entire source/domain lifetime inside the one original MPI
    // process session. MPI cannot be reinitialized after GTest suite teardown.
    for(int lifecycle=0;lifecycle<20;++lifecycle) {
    SCOPED_TRACE(lifecycle);
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::weak_ptr<ITensorGemm> retired_gate;
    {
        test::PlanningGGUFFixture file(true,false,GGUFTensorType::F32,f,std::nullopt,8,
            test::PlanningGGUFFixture::Geometry::Directory,1,e);
        test::populateServingGGUFFixture(file);
        auto factory=std::make_shared<TensorFactory>(*world);
        ModelContextConfig source_config;source_config.mpi_ctx=world;source_config.factory=factory.get();source_config.use_mmap=false;
        auto original=ModelContext::create(file.path(),source_config);
        requirePreparedCompute(bool(original),"actual prepared-compute model context");
        const ModelContextId model{reinterpret_cast<uint64_t>(original.get())};
        auto plan=test::cpuJointSourceTestPlan();plan.first_model_layer=0;
        plan.domains.front().participants={GlobalDeviceAddress::cpu(nodes[0])};
        plan.domains.front().backend=CollectiveBackendType::UPI;
        plan.domains.back().participants={GlobalDeviceAddress::cpu(nodes[0]),GlobalDeviceAddress::cpu(nodes[1])};
        plan.domains.back().world_ranks={0,1};
        plan.routed_tiers[0].max_experts_per_layer=2;plan.routed_tiers[1].max_experts_per_layer=2;
        plan.placements={{.layer=0,.routed_expert_tier={0,0,1,1}}};
        plan.residency_policy=RoutedExpertResidencyPolicy::StaticById;
        plan.authority_execution=resolveMoEOverlayAuthorityExecutionKind(plan);
        const auto runtime=resolveMoEExpertOverlayRuntimePlan(std::make_shared<const MoERoutedExpertPlacementPlan>(plan),
            {.current_world_rank=rank,.validate_mvp_root_reachability=false});
        requirePreparedCompute(bool(runtime),"original CPU whole/cohort runtime plan");
        auto& loader=original->concreteLoader();
        auto preparation=std::make_shared<const MoEExpertOverlayPreparationPlan>(MoEExpertOverlayPreparationPlan::build(*runtime,loader));
        const auto loading=preparation->filteredForDevice(DeviceId::cpu()).sourceWeightPlan(loader,model);
        std::vector<int> local_ids;int participant=-1,whole=-1;
        for(const auto& descriptor:preparation->cpuJointOwnerMap()->participants()) {
            if(descriptor.world_rank_known && descriptor.world_rank==rank) {
                local_ids.push_back(descriptor.participant_id);
                if(descriptor.domain_name=="cpu_joint")participant=descriptor.participant_id;
                else whole=descriptor.participant_id;
            }
        }
        requirePreparedCompute(participant>=0,"original local CPU fragment participant");
        const auto selected=preparation->cpuShardSourceFor(0,*preparation->cpuJointOwnerMap()->participantForId(participant));
        requirePreparedCompute(selected && selected->contains(selected_expert),"same global selected expert");
        std::vector<NativeCPUCollectiveProgram::Participant> members;
        for(int member=0;member<2;++member)members.push_back({member,GlobalDeviceAddress::cpu(nodes[member],inventory->ranks[member].hostname)});
        const auto program=NativeCPUCollectiveProgram::expertShardRoutePartials(1,CollectiveBackendType::UPI,members);
        size_t weights_bytes=0;
        for(const auto& request:loading.requirements())
            for(const auto& projection:projectNativeWeightSource(request,loader.nativeWeightSourceGeometry(
                request.source_name.empty()?request.canonical_name:request.source_name)))
                weights_bytes+=CPUStorageRoot<float>(projection.finalRows()*projection.finalColumns(),
                    PhysicalMemoryOwner::RoutedExpertWeights,DeviceId::cpu(),loader.nativeWeightSourcePlacement()).physicalBytes();
        // The real existing source-native projection/SwiGLU workspace declaration,
        // not a reserve or a copied allocation formula, prices both probe families.
        const auto invocationBytes=[](int intermediate) {
            auto requirements=cpuSwiGLUWorkspaceRequirements(Service::kMaximumRows,intermediate);
            for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown})
                requirements.merge(CPUProjectionWorkspaceContract::sourceNative(TensorType::FP32,
                    {.rows=Service::kMaximumRows,.n=role==WeightRole::MoEExpertDown?h:intermediate,
                     .k=role==WeightRole::MoEExpertDown?intermediate:h,.workers=1,
                     .execution=CPUExecutionGeometry::local(),.numerical_policy=CPUProjectionNumericalPolicy::GPUAlignedExpert}));
            return requirements.total_bytes_with_alignment();
        };
        size_t probe_bytes=0;
        for(const auto phase:{ExpertHistogramSource::PrefillChunk,ExpertHistogramSource::DecodeToken,ExpertHistogramSource::GroupedVerifier})
            probe_bytes=std::max(probe_bytes,Service::cpuJointAllocationBytes(selected->cohort(),selected->participantIndex(),phase,invocationBytes(f/2)));
        if(rank==0)probe_bytes=std::max(probe_bytes,Service::allocationBytes({h,f},invocationBytes(f)));
        const CPUStorageRoot<float> diagnostic_root(rank==0?Service::kMaximumRows*h+1:0,
            PhysicalMemoryOwner::ExecutionWorkspace,DeviceId::cpu());
        const auto capacity=observeSystemMemoryCapacity();
        const PhysicalMemoryResource host{rank,DeviceId::cpu(),capacity.total_bytes,capacity.admission_available_bytes};
        PhysicalMemoryPlanBuilder bill;
        bill.add(host,PhysicalMemoryOwner::RoutedExpertWeights,weights_bytes)
            .add(host,PhysicalMemoryOwner::ExecutionWorkspace,probe_bytes+diagnostic_root.physicalBytes());
        program.contributeDomainBOM(bill,host,rank);NativeCPUCollectiveProgram::contributeProcessPoolBOM(bill,host);
        if(rank==0)bill.add(host,PhysicalMemoryOwner::ActivationTransportStaging,ShmemSpinArena::allocationGeometry(2).mapped_bytes);
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bill.build()),rank);
        auto manager=original->concreteWeightManager();manager->installPhysicalMemoryAuthority(memory);
        const auto frozen=manager->materialize(loading);
        requirePreparedCompute(manager->prepareMoEExpertOverlayWeights(*runtime,DeviceId::cpu(),&frozen,nullptr,
            PreparedWeightAdmission::AllocateCompleteSet,preparation),"actual original prepared CPU engines");
        auto registry=std::make_shared<MoEOverlayParticipantResidencyRegistry>(MoEOverlayParticipantResidencyRegistry::Config{
            .local_participant_ids=local_ids,.num_layers=1,.num_experts=e,.initial_epoch=1,
            .projection_preparation=preparation,.first_model_layer=0,.original_model=model});
        std::string error;
        requirePreparedCompute(registry->bindOriginalPreparedContext(original,preparation,&error),error.c_str());
        requirePreparedCompute(registry->finalizeInitialBanksFromPreparedRegistry(manager->expertGemmRegistry(),&error),error.c_str());
        {const auto frame=registry->initialCPUJointLayer(participant,0)->acquireEpoch(1);
         requirePreparedCompute(bool(frame),"actual original source bank");retired_gate=frame->bankLease()->layerForModelLayer(0).cpu_fragments[selected_expert].gate();}
        std::variant<NativeCPUCollectiveProgram::DomainUse,NativeCPUCollectiveProgram::ProcessUse> use(
            std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,program.admit(memory,rank));
        if(!observed_original_pool){original_pool_admission=memory->admission();observed_original_pool=true;}
        auto domain=GlobalTPContext::createAdmittedCPUExpertDomain(MPI_COMM_WORLD,program,std::move(use));
        requirePreparedCompute(bool(domain),"actual original admitted CPU domain");
        try {
            auto diagnostic=diagnostic_root.allocate(memory);
            const auto baseline=memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation);
            for(int replay=0;replay<2;++replay)
                for(const auto phase:{ExpertHistogramSource::PrefillChunk,ExpertHistogramSource::DecodeToken,ExpertHistogramSource::GroupedVerifier}) {
                    const int rows=Service::rowsForPhase(phase);
                    std::fill(diagnostic.begin(),diagnostic.end(),-991.0f);
                    const auto span=rank==0?std::span<float>(diagnostic.data(),size_t(rows)*h):std::span<float>{};
                    const auto observed=Service::measurePreparedCPUJoint(*registry,1,participant,0,selected_expert,phase,domain,memory,span);
                    EXPECT_EQ(bool(observed),rank==0);
                    if(observed) {
                        EXPECT_EQ(observed->target.word(),MoEOverlayRouteTarget::cpuCohort(1).word());
                        EXPECT_EQ(observed->expert,selected_expert);EXPECT_EQ(observed->logical_rows_per_invocation,uint64_t(rows));
                        EXPECT_EQ(observed->measured_invocations,Service::kMeasuredSamples);EXPECT_GT(observed->total_root_wall_nanoseconds,0u);
                        EXPECT_EQ(observed->boundary,Service::PreparedComputeBoundary::InputReadyStartReleaseToCanonicalOutput);
                        for(int row=0;row<rows;++row)for(int col=0;col<h;++col) {
                            const auto expected=preparedComputeOracle(row,col,selected_expert);
                            EXPECT_GT(expected,0.0f);
                            EXPECT_NEAR(span[size_t(row)*h+col],expected,std::max(1.0e-20f,std::abs(expected)*3.0e-4f));
                        }
                        EXPECT_EQ(diagnostic[size_t(rows)*h],-991.0f);
                    }
                    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation),baseline);
                }
            // A raw-pointer alias cannot borrow the original native owner.
            // It is rejected before arming a job or entering any collective;
            // unlike a rank-local failure after admission it does not poison
            // the actual domain used by the following whole/control teardown.
            std::fill(diagnostic.begin(),diagnostic.end(),-993.0f);
            const auto foreign=std::shared_ptr<IGlobalTPContext>(domain.get(),[](IGlobalTPContext*){});
            const auto terminal=rank==0?std::span<float>(diagnostic.data(),h):std::span<float>{};
            EXPECT_THROW((void)Service::measurePreparedCPUJoint(*registry,1,participant,0,selected_expert,
                ExpertHistogramSource::DecodeToken,foreign,memory,terminal),std::invalid_argument);
            if(rank==0) {
                EXPECT_EQ(diagnostic[0],-993.0f);
                requirePreparedCompute(whole>=0,"actual whole-owner control participant");
                const auto bank=registry->endpoint(whole)->acquire(1);
                EXPECT_GT(Service::measurePrepared(bank->layerForModelLayer(0).experts[0],DeviceId::cpu(),0,
                    ExpertHistogramSource::DecodeToken,{h,f},memory),0u);
            }
            // No borrowed domain survives this producer; all stages and views
            // already retired. The real bank/context still retain source roots.
            EXPECT_FALSE(retired_gate.expired());
            requirePreparedCompute(domain->retireOriginalCPUExpertDomain(),"original domain retirement after prepared compute");
        } catch(const std::exception& failure){abortPreparedCompute(failure.what());}
          catch(...){abortPreparedCompute("unknown prepared-compute failure with an admitted native domain");}
    }
    EXPECT_TRUE(retired_gate.expired());
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
    // The next independent source may start only after every old domain retires.
    requirePreparedCompute(MPI_Barrier(MPI_COMM_WORLD)==MPI_SUCCESS,"prepared-compute lifecycle retirement join");
    }
    ::testing::Test::RecordProperty("service_boundary","input_ready_start_release_to_canonical_output");
    ::testing::Test::RecordProperty("source_domain_lifetimes",20);
    ::testing::Test::RecordProperty("dispatch_and_ingress_measured",false);
    ::testing::Test::RecordProperty("whole_control_boundary","prepared_ffn_without_collective_start_release");
    } catch(const std::exception& failure){abortPreparedCompute(failure.what());}
      catch(...){abortPreparedCompute("unknown prepared-compute source/setup failure");}
}
} // namespace
