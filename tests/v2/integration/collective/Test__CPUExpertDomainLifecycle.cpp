/**
 * @file Test__CPUExpertDomainLifecycle.cpp
 * @brief Actual same-PMA CPU domain/split/rooted gather/explicit-retirement replay.
 *
 * The fixture owns the original MPIProcessSession and existing CPU backends.
 * Every native participant occupies a distinct discovered physical socket.
 * Absent socket populations are not launched or qualified.
 * Domain-only cases construct no model. The CUDA/ROCm replay cases additionally
 * prepares a tiny genuine source and joins its actual GPU dispatch, CPU shards
 * and rooted return. It supplies no fabricated completion/readiness and makes
 * no full-model accuracy or deployment qualification claim. The prepared CPU
 * service case uses a genuine tiny source at the explicitly dispatch-exclusive
 * input-ready/start-release to canonical-output boundary.
 */
#include "app/MPIProcessSession.h"
#include "collective/GlobalTPContext.h"
#include "collective/NativeCPUCollectiveProgram.h"
#include "collective/backends/ShmemSpinBackend.h"
#include "memory/CPUStorageRoot.h"
#include "backends/HostMemoryCapacity.h"
#include "utils/MPIContext.h"
#include "utils/NUMATopology.h"
#include "utils/NodeTopology.h"
#include "../../fixtures/collective/CPUExpertDomainSocketMembership.h"
#include <gtest/gtest.h>
#include <set>
#include <optional>
#include <variant>
#include <algorithm>
#include "CPUJointCapturedPacketLifecycle.inc"
using namespace llaminar2;
namespace {
class CPUExpertDomainNative : public ::testing::Test {
protected:
    /** @brief Adversarial original-job inputs used only by owned failure children. */
    enum class PreparedServiceRejection { None, JointInitialEpoch, TargetsMissingPMA };
    /** @brief Run original source/domain lifecycle or one intentional rejected input.
     * @param rejection Exact API rejection; None retains the ordinary20 lifecycle test.
     */
    void preparedComputeSourceAndRetirement(PreparedServiceRejection rejection);
    static inline std::optional<MPIProcessSession> session;
    static inline std::weak_ptr<const PhysicalMemoryPlanAdmissionCertificate> original_pool_admission;
    static inline bool observed_original_pool = false;
    static void SetUpTestSuite() {
        int argc=0;char** argv=nullptr;
        session.emplace(MPIProcessSession::initialize(argc,argv));
    }
    static void TearDownTestSuite() {
        session.reset();
        EXPECT_TRUE(original_pool_admission.expired());
    }
    /** @brief Run original backends with exact observed endpoints and live per-source ranges. */
    void originalDomain(CollectiveBackendType backend,int processes) {
        int rank=-1,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
        ASSERT_EQ(size,processes);
        const auto world=MPIContextFactory::global();const auto inventory=world->clusterInventory();
        ASSERT_EQ(inventory->ranks.size(),size_t(size));
        const auto affinity=NUMATopology::detectLocalNUMANode();
        ASSERT_TRUE(affinity.detection_succeeded);ASSERT_GE(affinity.local_numa_node,0);
        const auto topology=NodeTopology::detect();
        const auto node=std::find_if(topology.numaNodes().begin(),topology.numaNodes().end(),
            [&](const auto& item){return item.node_id==affinity.local_numa_node;});
        ASSERT_NE(node,topology.numaNodes().end());
        ASSERT_GE(node->socket_id,0);
        std::vector<int> observed_sockets(size);
        ASSERT_EQ(MPI_Allgather(&node->socket_id,1,MPI_INT,observed_sockets.data(),1,MPI_INT,MPI_COMM_WORLD),MPI_SUCCESS);
        std::vector<int> observed_nodes(size);
        ASSERT_EQ(MPI_Allgather(&affinity.local_numa_node,1,MPI_INT,observed_nodes.data(),1,MPI_INT,MPI_COMM_WORLD),MPI_SUCCESS);
        std::vector<NativeCPUCollectiveProgram::Participant> members;
        std::vector<test::CPUExpertSocketEndpoint> endpoints;
        for(const auto &original:inventory->ranks) {
            ASSERT_GE(original.rank,0);ASSERT_LT(original.rank,size);
            const int node=observed_nodes[original.rank];
            if(original.cpu.numa_node>=0)ASSERT_EQ(original.cpu.numa_node,node);
            ASSERT_EQ(original.hostname,inventory->ranks.front().hostname);
            endpoints.push_back({original.rank,node,observed_sockets[original.rank]});
            members.push_back({original.rank,GlobalDeviceAddress::cpu(node,original.hostname)});
        }
        ASSERT_TRUE(test::validCPUExpertSocketMembership(endpoints,processes))
            << "Native CPU expert domain requires one original MPI rank per distinct physical socket";
        ASSERT_EQ(members.size(),size_t(size));
        RecordProperty("actual_mpi_processes",size);RecordProperty("actual_distinct_numa_nodes",size);
        RecordProperty("actual_distinct_physical_sockets",size);
        RecordProperty("expert_socket_hardware_qualification","original_distinct_socket_endpoints");
        const auto program=NativeCPUCollectiveProgram::expertShardRoutePartials(27,backend,members);
        const auto member=std::find_if(members.begin(),members.end(),[&](const auto& p){return p.world_rank==rank;});
        ASSERT_NE(member,members.end());const auto ordinal=size_t(member-members.begin());
        constexpr size_t width=5;
        constexpr size_t per_peer=ShmemSpinArena::CHUNK_CAPACITY/width+17; // Real multi-chunk large→small replay.
        const auto record_capacity=ordinal==0?per_peer*members.size():per_peer;
        const auto capacity=observeSystemMemoryCapacity();
        ASSERT_GT(capacity.total_bytes,0u);ASSERT_GT(capacity.admission_available_bytes,0u);
        const PhysicalMemoryResource cpu{rank,DeviceId::cpu(),capacity.total_bytes,capacity.admission_available_bytes};
        // The session reservation retains the admission/ledger, not its
        // temporary PMA facade. Observe the first root exactly once.

        for(int replay=0;replay<20;++replay) {
            PhysicalMemoryPlanBuilder plan;
            program.contributeDomainBOM(plan,cpu,rank);
            NativeCPUCollectiveProgram::contributeProcessPoolBOM(plan,cpu);
            if(ordinal==0 && backend==CollectiveBackendType::UPI && members.size()>1)
                plan.add(cpu,PhysicalMemoryOwner::ActivationTransportStaging,
                    ShmemSpinArena::allocationGeometry(int(members.size())).mapped_bytes);
            std::optional<CPUStorageRoot<float>> records_decl;
            std::optional<CPUStorageRoot<RootedFloatRecordRange>> ranges_decl;
            {
                const auto placement=CPUStoragePlacement::onNode(member->address.numa_node);
                records_decl.emplace(record_capacity*width,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
                plan.add(cpu,PhysicalMemoryOwner::ActivationArena,records_decl->physicalBytes());
                if(ordinal==0) {
                    ranges_decl.emplace(members.size(),PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
                    plan.add(cpu,PhysicalMemoryOwner::ActivationArena,ranges_decl->physicalBytes());
                }
            }
            auto memory=std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()),rank);
            AlignedVector<float> records;AlignedVector<RootedFloatRecordRange> ranges;
            if(records_decl)records=records_decl->allocate(memory);
            if(ranges_decl)ranges=ranges_decl->allocate(memory);
            std::variant<NativeCPUCollectiveProgram::DomainUse,NativeCPUCollectiveProgram::ProcessUse> use=
                decltype(use)(std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,program.admit(memory,rank));
            if (!observed_original_pool) {
                original_pool_admission = memory->admission();
                observed_original_pool = true;
            }
            auto context=GlobalTPContext::createAdmittedCPUExpertDomain(MPI_COMM_WORLD,program,std::move(use));
            ASSERT_TRUE(context);
            if(context) {
                ASSERT_TRUE(context->authenticatesOriginalCPUExpertBorrow(context));
                {auto borrowed=std::shared_ptr<IGlobalTPContext>(context);
                 EXPECT_THROW(context->retireOriginalCPUExpertDomain(),std::logic_error);}
                for(int shape=0;shape<4;++shape) {
                    const size_t count=shape==0?per_peer:shape==1?0:shape==2?(ordinal%2?7:1):3;
                    std::fill_n(records.data(),records.size(),-991.0f);
                    for(size_t element=0;element<count*width;++element)records[element]=float(rank*1000000+element);
                    size_t gathered=0;
                    ASSERT_TRUE(context->gatherVariableFloatRecordsToRoot(records.data(),count,records.data(),
                        record_capacity,width,0,gathered,"original_cpu_domain_lifecycle",{ranges.data(),ranges.size()}));
                    if(ordinal==0) {
                        size_t expected=0;
                        for(size_t source=0;source<members.size();++source) {
                            const size_t source_count=shape==0?per_peer:shape==1?0:shape==2?(source%2?7:1):3;
                            EXPECT_EQ(ranges[source].participant,int(source));EXPECT_EQ(ranges[source].record_count,source_count);
                            for(size_t element=0;element<source_count*width;++element)
                                ASSERT_EQ(records[ranges[source].first_record*width+element],float(members[source].world_rank*1000000+element));
                            expected+=source_count;
                        }
                        EXPECT_EQ(gathered,expected);if(gathered<record_capacity)EXPECT_EQ(records[gathered*width],-991.0f);
                    } else EXPECT_EQ(gathered,0u);
                }
                ASSERT_TRUE(context->retireOriginalCPUExpertDomain());
                EXPECT_TRUE(context->retireOriginalCPUExpertDomain()); // No second SDK/free/collective.
                EXPECT_FALSE(context->authenticatesOriginalCPUExpertBorrow(context));
                context.reset();
            }
            EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ActivationTransportStaging,PhysicalMemoryMaterializationKind::NewAllocation),0u);
            MPI_Barrier(MPI_COMM_WORLD); // Original test iteration boundary after every member's explicit retirement.
        }
        EXPECT_FALSE(original_pool_admission.expired()); // The original process pool intentionally lives until MPI_Finalize.
    }
};
TEST_F(CPUExpertDomainNative, UPIOriginalDomainRepeatedAndRetired){originalDomain(CollectiveBackendType::UPI,2);}
TEST_F(CPUExpertDomainNative, MPIOriginalDomainRepeatedAndRetired){originalDomain(CollectiveBackendType::MPI,2);}
TEST_F(CPUExpertDomainNative, UPIFourPhysicalSockets){originalDomain(CollectiveBackendType::UPI,4);}
TEST_F(CPUExpertDomainNative, MPIFourPhysicalSockets){originalDomain(CollectiveBackendType::MPI,4);}
TEST_F(CPUExpertDomainNative, UPIEightPhysicalSockets){originalDomain(CollectiveBackendType::UPI,8);}
TEST_F(CPUExpertDomainNative, MPIEightPhysicalSockets){originalDomain(CollectiveBackendType::MPI,8);}
#ifdef HAVE_CUDA
/** @test Actual CUDA/CPU2 capture reuse across twenty successful 9/0/1/9 transactions. */
TEST_F(CPUExpertDomainNative, CUDAPreparedRootedPacketReplay) { rooted_packet_lifecycle::run(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test Actual ROCm/CPU2 capture reuse across twenty successful 9/0/1/9 transactions. */
TEST_F(CPUExpertDomainNative, ROCmPreparedRootedPacketReplay) { rooted_packet_lifecycle::run(DeviceId::rocm(0)); }
#endif

#ifdef HAVE_CUDA
/** @test Captured MainDecode packet/reset/live-byte path on actual CUDA and two CPU sockets. */
TEST_F(CPUExpertDomainNative, CUDAPreparedRootedDecodeReplay)
{ rooted_packet_lifecycle::run(DeviceId::cuda(0), rooted_packet_lifecycle::PacketPhaseScenario::Decode); }
/** @test Actual CUDA verifier2/3/4 and partial/empty work is byte-identical to same-input serial decode. */
TEST_F(CPUExpertDomainNative, CUDAPreparedRootedVerifierReplay)
{ rooted_packet_lifecycle::run(DeviceId::cuda(0), rooted_packet_lifecycle::PacketPhaseScenario::GroupedVerifier); }
#endif
#ifdef HAVE_ROCM
/** @test Captured MainDecode packet/reset/live-byte path on actual ROCm and two CPU sockets. */
TEST_F(CPUExpertDomainNative, ROCmPreparedRootedDecodeReplay)
{ rooted_packet_lifecycle::run(DeviceId::rocm(0), rooted_packet_lifecycle::PacketPhaseScenario::Decode); }
/** @test Actual ROCm verifier2/3/4 and partial/empty work is byte-identical to same-input serial decode. */
TEST_F(CPUExpertDomainNative, ROCmPreparedRootedVerifierReplay)
{ rooted_packet_lifecycle::run(DeviceId::rocm(0), rooted_packet_lifecycle::PacketPhaseScenario::GroupedVerifier); }
#endif

}

#include "CPUJointPreparedComputeServiceContract.inc"
