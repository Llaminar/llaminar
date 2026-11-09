/**
 * @file Perf__CPURootedPublicationTransport.cpp
 * @brief Measure same-node CPU rooted-publication transport crossovers.
 *
 * This focused two-rank harness compares the bounded native shared-memory
 * protocols with MPI payload transport for the two communication operations
 * used by exact CPU MoE publication: variable packed-record gather and compact
 * output broadcast. It sweeps payloads from decode-sized messages through
 * long-prefill messages and reports slowest-rank p50/p95 latency and effective
 * payload bandwidth.
 *
 * Profiler or benchmark output from this executable is evidence for a typed
 * transport policy. It is not a correctness substitute: production graph and
 * all-format byte-equivalence tests remain the authority for arithmetic.
 * The OriginalNUMAVariableRecordGatherReplay case additionally uses admitted
 * original physical socket domains, production matched-envelope MPI gather,
 * node-certified ordinary packed roots, root aliasing and live/empty replays.
 * Its live-copy boundary cases cover native socket-team copies and single-caller
 * entry from an existing OpenMP region without changing the production selector.
 * The original admitted backend also replays absorbing abort publication and
 * local-abort precedence without a callback, host hook, race or timing assumption.
 * RootCombinePreweightReadyRecordEconomy separately measures the exact CPU
 * arithmetic on PMA/NUMA cached outputs. It is an authored mechanism control,
 * not a mapped-ticket, gather, topology or full-model speed certificate.
 * An explicit sampled-profile repeat isolates only current fused/live3208 work
 * after the original proof/warmup. External perf still observes process setup
 * and final byte checks; no canonical timing samples are emitted in that mode.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>

#include "collective/DeviceGroup.h"
#include "collective/GlobalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "memory/CPUStorageRoot.h"
#include "backends/HostMemoryCapacity.h"
#include "kernels/cpu/moe/CPUCanonicalShardRouteRecords.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "utils/MPIContext.h"
#include "utils/NUMATopology.h"
#include "utils/NodeTopology.h"
#include "utils/DebugEnv.h"
#include "../../fixtures/collective/CPUExpertDomainSocketMembership.h"
#include "collective/backends/ShmemSpinBackend.h"
#include "collective/backends/UPIBackend.h"
#include "../../utils/CPUPerfStatsTimingCollectionScope.h"

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <span>
#include <variant>
#include <chrono>
#include <charconv>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test
{
    /** @brief Bounded test operation borrowing one actual admitted shared backend.
     * This fixture-only seam returns no backend pointer and retains no borrower.
     * Public production transport selection and original context retirement are
     * unchanged. Failed native execution marks the same original context aborted.
     */
    struct CPURootedPublicationTransportReplayAccess
    {
        /**
         * @brief Replay native epoch transitions on the same original admitted backend.
         * @param context Original live CPU domain whose mapping remains owned here.
         *
         * All ranks first advance two healthy epochs. After an outside barrier,
         * only rank zero injects the terminal shared witness representing abort
         * between the old local precheck and normal publication. No peer worker
         * or timing hook is needed: rank one retains its already-ready epoch.
         * Repeated attempts must retain both the witness and local owner epoch;
         * local abort must reject that ready peer rather than report completion.
         */
        static void replayEpochAbort(const std::shared_ptr<GlobalTPContext>& context)
        {
            ASSERT_TRUE(context);
            ASSERT_TRUE(context->authenticatesOriginalCPUExpertBorrow(context));
            auto* backend=dynamic_cast<ShmemSpinBackend*>(context->backend_.get());
            ASSERT_NE(backend,nullptr);
            ASSERT_EQ(backend->num_ranks_,2);
            EXPECT_TRUE(backend->publishNextEpoch());
            EXPECT_TRUE(backend->publishNextEpoch());
            EXPECT_EQ(backend->currentEpoch(),2u);
            ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD),MPI_SUCCESS);
            if(backend->my_rank_!=0)return;
            auto* slot=backend->arena()->epoch_at(0);
            constexpr auto terminal=std::numeric_limits<uint64_t>::max();
            const auto original_epoch=backend->currentEpoch();
            // Exact shared witness of an abort that wins after the old precheck.
            // The local flag deliberately remains clear until production rejects
            // this first publication, so a flag-only check cannot satisfy replay.
            slot->epoch.store(terminal,std::memory_order_release);
            for(int replay=0;replay<20;++replay)
            {
                EXPECT_FALSE(backend->publishNextEpoch())<<replay;
                EXPECT_EQ(slot->epoch.load(std::memory_order_acquire),terminal)<<replay;
                EXPECT_EQ(backend->currentEpoch(),original_epoch)<<replay;
            }
            backend->abort();
            // Rank one is paused at the outer barrier with a genuinely ready
            // epoch. An observed local abort must beat its successful observation.
            for(int replay=0;replay<20;++replay)
            {
                EXPECT_FALSE(backend->waitForPeerEpoch(1,2u,"replaying local abort with ready peer"))<<replay;
                EXPECT_EQ(slot->epoch.load(std::memory_order_acquire),terminal)<<replay;
            }
        }

        /** @brief Invoke only the existing shared gather of a live admitted UPI context.
         * @param context Original retained domain; foreign, null and retired owners reject.
         * @param local Original NUMA-packed local prefix, aliased to receive storage on root.
         * @param count Actual local records, never declaration capacity.
         * @param root Original root NUMA-packed receive allocation, null on contributor.
         * @param capacity Common declared record capacity used by both transport arms.
         * @param width Native packed record width.
         * @param root_index Original communicator member selected as gather root.
         * @param gathered Successful root live count; reset before any rejected borrow.
         * @param ranges Original root-only per-source boundaries, reset on rejected borrow.
         * @return Existing native result, or false before entry for an invalid owner.
         */
        static bool gatherSharedOriginal(const std::shared_ptr<GlobalTPContext>& context,
            const float* local,size_t count,float* root,size_t capacity,size_t width,
            int root_index,size_t& gathered,std::span<RootedFloatRecordRange> ranges)
        {
            gathered=0;
            for(auto& range:ranges)range={};
            if(!context || !context->authenticatesOriginalCPUExpertBorrow(context) ||
               context->isAbortRequested() || context->backend()!=CollectiveBackendType::UPI)
                return false;
            auto* backend=dynamic_cast<ShmemSpinBackend*>(context->backend_.get());
            if(!backend)return false;
            if(!backend->gatherVariableFloatRecordsToRoot(local,count,root,capacity,width,
                                                        root_index,gathered,ranges)) {
                context->requestAbort();return false;
            }
            return true;
        }
    };

    namespace
    {
        using Clock = std::chrono::steady_clock;

        /** Return this process's rank in the benchmark communicator. */
        int worldRank()
        {
            int rank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank;
        }

        /** Return the process count in the benchmark communicator. */
        int worldSize()
        {
            int size = 0;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return size;
        }

        /** Construct the CPU device group consumed by both transport backends. */
        DeviceGroup makeCpuGroup(int size)
        {
            DeviceGroup group;
            group.name = "perf_cpu_rooted_publication_transport";
            group.scope = CollectiveScope::LOCAL;
            for (int rank = 0; rank < size; ++rank)
                group.devices.push_back(DeviceId::cpu());
            return group;
        }

        /** Choose enough samples for stable latency without making large sweeps slow. */
        int repetitionsForBytes(size_t bytes)
        {
            if (bytes <= 256u * 1024u)
                return 200;
            if (bytes <= 4u * 1024u * 1024u)
                return 60;
            return 20;
        }

        /** Select a nearest-rank percentile from an already sorted sample set. */
        double percentile(
            const std::vector<double> &sorted_samples,
            size_t numerator)
        {
            const size_t rank =
                (numerator * sorted_samples.size() + 99u) / 100u;
            return sorted_samples[std::max<size_t>(1u, rank) - 1u];
        }

        /** Slowest-rank latency summary for one exact transport and payload. */
        struct LatencySummary
        {
            double p50_us = 0.0;
            double p95_us = 0.0;
        };

        /**
         * Combine rank-local samples by maximum before calculating percentiles.
         *
         * A rooted transaction is complete only when its slowest participant is
         * complete. Reducing the full sample vector once after measurement avoids
         * placing an MPI timing reduction inside every canonical sample.
         */
        LatencySummary summarizeSlowestRank(
            const std::vector<double> &local_samples)
        {
            std::vector<double> slowest_samples(local_samples.size(), 0.0);
            EXPECT_EQ(
                MPI_Allreduce(
                    local_samples.data(),
                    slowest_samples.data(),
                    static_cast<int>(local_samples.size()),
                    MPI_DOUBLE,
                    MPI_MAX,
                    MPI_COMM_WORLD),
                MPI_SUCCESS);
            std::sort(slowest_samples.begin(), slowest_samples.end());
            return {
                .p50_us = percentile(slowest_samples, 50u),
                .p95_us = percentile(slowest_samples, 95u),
            };
        }

        /** Emit one stable CSV row from rank zero. */
        void printResult(
            std::string_view operation,
            size_t bytes,
            std::string_view transport,
            const LatencySummary &summary)
        {
            if (worldRank() != 0)
                return;
            const double p50_gbps =
                static_cast<double>(bytes) / summary.p50_us / 1000.0;
            std::cout << operation << ',' << bytes << ',' << transport << ','
                      << std::fixed << std::setprecision(3)
                      << summary.p50_us << ',' << summary.p95_us << ','
                      << p50_gbps << '\n';
        }
    } // namespace

    /**
     * @test OriginalNUMAVariableRecordGatherReplay
     * Compare the retained SharedSpin protocol with production MPI matched
     * gather on the SAME certified ordinary NUMA roots and live record counts.
     * Only the named test seam bypasses the production crossover for the shared
     * arm. Original contexts own their split, arena, PMA and explicit retirement.
     * Setup/poisoning/validation and readiness barriers are outside each timed
     * call; this measures ready-data transport service, not model arrival skew.
     */
    TEST(Perf__CPURootedPublicationTransport, OriginalNUMAVariableRecordGatherReplay)
    {
        constexpr int root=1;
        constexpr size_t width=2561;
        constexpr size_t record_bytes=width*sizeof(float);
        constexpr size_t one_mib=(1u<<20)/record_bytes;
        constexpr size_t four_mib=(4u<<20)/record_bytes;
        constexpr size_t sixteen_mib=(16u<<20)/record_bytes;
        constexpr size_t capacity=2*sixteen_mib;
        constexpr size_t parallel_threshold_records=(64u*1024u)/width;
        constexpr size_t shape_count=16;
        constexpr int warmups=3,repetitions=12;
        const int rank=worldRank();
        ASSERT_EQ(worldSize(),2)<<"Requires two actual physical socket workers";
        ASSERT_EQ(collective_timeout_policy::effectiveCollectTimeoutMs(
            debugEnv().tp_collect_timeout_ms),30000)<<"Requires the original collective deadline";
        const auto inventory=MPIContextFactory::global()->clusterInventory();
        ASSERT_EQ(inventory->ranks.size(),2u);
        const auto affinity=NUMATopology::detectLocalNUMANode();
        ASSERT_TRUE(affinity.detection_succeeded);
        const auto topology=NodeTopology::detect();
        const auto node=std::find_if(topology.numaNodes().begin(),topology.numaNodes().end(),
            [&](const auto& item){return item.node_id==affinity.local_numa_node;});
        ASSERT_NE(node,topology.numaNodes().end());
        std::array<int,2> sockets{},nodes{};
        ASSERT_EQ(MPI_Allgather(&node->socket_id,1,MPI_INT,sockets.data(),1,MPI_INT,MPI_COMM_WORLD),MPI_SUCCESS);
        ASSERT_EQ(MPI_Allgather(&affinity.local_numa_node,1,MPI_INT,nodes.data(),1,MPI_INT,MPI_COMM_WORLD),MPI_SUCCESS);
        std::vector<NativeCPUCollectiveProgram::Participant> members;
        std::vector<CPUExpertSocketEndpoint> endpoints;
        for(const auto& observed:inventory->ranks) {
            ASSERT_GE(observed.rank,0);ASSERT_LT(observed.rank,2);
            ASSERT_EQ(observed.hostname,inventory->ranks.front().hostname);
            ASSERT_EQ(observed.cpu.numa_node,nodes[observed.rank]);
            members.push_back({observed.rank,GlobalDeviceAddress::cpu(nodes[observed.rank],observed.hostname)});
            endpoints.push_back({observed.rank,nodes[observed.rank],sockets[observed.rank]});
        }
        ASSERT_TRUE(validCPUExpertSocketMembership(endpoints,2));
        // Original inventory order must match MPI member order; never fabricate
        // a socket from a rank number or silently repair an unexpected ordering.
        ASSERT_EQ(members[0].world_rank,0);ASSERT_EQ(members[1].world_rank,1);
        RecordProperty("actual_distinct_physical_sockets",2);
        RecordProperty("actual_rank_numa_node",nodes[rank]);
        RecordProperty("actual_root_world_rank",root);
        RecordProperty("native_record_width_elements",width);
        RecordProperty("original_omp_max_workers",omp_get_max_threads());
        RecordProperty("original_omp_dynamic",omp_get_dynamic());
        const auto shared_program=NativeCPUCollectiveProgram::expertShardRoutePartials(
            73002,CollectiveBackendType::UPI,members);
        const auto mpi_program=NativeCPUCollectiveProgram::expertShardRoutePartials(
            73003,CollectiveBackendType::MPI,members);
        const auto observed_capacity=observeSystemMemoryCapacity();
        ASSERT_TRUE(observed_capacity.valid());
        ASSERT_GT(observed_capacity.admission_available_bytes,0u);
        const PhysicalMemoryResource cpu{rank,DeviceId::cpu(),observed_capacity.total_bytes,
                                         observed_capacity.admission_available_bytes};
        const size_t local_capacity=rank==root?capacity:sixteen_mib;
        const auto placement=CPUStoragePlacement::onNode(nodes[rank]);
        const CPUStorageRoot<float> packed_root(local_capacity*width+1,
            PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
        const CPUStorageRoot<RootedFloatRecordRange> ranges_root(rank==root?2:0,
            PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
        PhysicalMemoryPlanBuilder bill;
        shared_program.contributeDomainBOM(bill,cpu,rank);
        mpi_program.contributeDomainBOM(bill,cpu,rank);
        NativeCPUCollectiveProgram::contributeProcessPoolBOM(bill,cpu);
        bill.add(cpu,PhysicalMemoryOwner::ActivationArena,packed_root.physicalBytes()+ranges_root.physicalBytes());
        if(rank==0)bill.add(cpu,PhysicalMemoryOwner::ActivationTransportStaging,
                            ShmemSpinArena::allocationGeometry(2).mapped_bytes);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bill.build()),rank);
        auto records=packed_root.allocate(memory);
        auto ranges=ranges_root.allocate(memory);
        using Use=std::variant<NativeCPUCollectiveProgram::DomainUse,NativeCPUCollectiveProgram::ProcessUse>;
        auto shared=GlobalTPContext::createAdmittedCPUExpertDomain(MPI_COMM_WORLD,shared_program,
            Use(std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,shared_program.admit(memory,rank)));
        auto mpi=GlobalTPContext::createAdmittedCPUExpertDomain(MPI_COMM_WORLD,mpi_program,
            Use(std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,mpi_program.admit(memory,rank)));
        ASSERT_TRUE(shared);ASSERT_TRUE(mpi);
        ASSERT_EQ(shared->scope(),TPScope::NODE_LOCAL);ASSERT_EQ(mpi->scope(),TPScope::NODE_LOCAL);
        ASSERT_EQ(mpi->backend(),CollectiveBackendType::MPI);
        ASSERT_EQ(CPURootedPublicationTransportPolicy::select(
            CPURootedPublicationOperation::PackedRecordGather,capacity*record_bytes),
            CPURootedPublicationTransport::MPIBandwidth);
        const auto range_span=std::span(ranges.data(),ranges.size());
        size_t rejected=99;
        EXPECT_FALSE(CPURootedPublicationTransportReplayAccess::gatherSharedOriginal(
            {},records.data(),0,rank==root?records.data():nullptr,capacity,width,root,rejected,range_span));
        EXPECT_EQ(rejected,0u);
        // An admitted MPI owner cannot masquerade as a retained SharedSpin owner.
        rejected=99;
        EXPECT_FALSE(CPURootedPublicationTransportReplayAccess::gatherSharedOriginal(
            mpi,records.data(),0,rank==root?records.data():nullptr,capacity,width,root,rejected,range_span));
        EXPECT_EQ(rejected,0u);EXPECT_FALSE(mpi->isAbortRequested());

        struct Shape {const char* name;size_t root_records,remote_records;};
        const std::array<Shape,shape_count> shapes={{{"equal_1MiB",one_mib,one_mib},
            {"equal_4MiB",four_mib,four_mib},{"equal_16MiB",sixteen_mib,sixteen_mib},
            {"small_after_large",1,7},{"all_empty",0,0},{"mixed_root_small",3,four_mib},
            {"remote_empty",four_mib,0},{"root_empty",0,four_mib},
            {"equal_16MiB_replay",sixteen_mib,sixteen_mib},{"small_replay",1,7},
            {"parallel_below",parallel_threshold_records,parallel_threshold_records},
            {"parallel_above",parallel_threshold_records+1,parallel_threshold_records+1},
            {"chunk_plus_tail",four_mib,four_mib+1},
            {"root_empty_chunk_tail",0,four_mib+1},
            {"root_large_remote_small",sixteen_mib,1},{"small_after_parallel",1,1}}};
        const std::string stage_name="perf_original_numa_variable_records";
        /** Source-tagged exact FP32 bytes, with the native flat-slot coordinate last. */
        const auto valueAt=[](int source,size_t record,size_t column) {
            return column==width-1?float(record):float(size_t(source)*8000000+record*width+column);
        };
        if(rank==root)std::cout<<"original_numa_gather,shape,transport,root,root_numa,remote_numa,width,common_capacity,root_local_bytes,remote_bytes,total_live_root_bytes,root_p50_us,root_p95_us,slowest_p50_us,slowest_p95_us,remote_effective_gbps\n";
        for(const bool shared_arm:{true,false}) {
            // Small timing vectors only; payload and range roots remain the same
            // PMA-owned node-certified allocations throughout both native arms.
            std::array<std::vector<double>,shape_count> samples;
            for(auto& sample:samples)sample.resize(repetitions);
            for(int iteration=-warmups;iteration<repetitions;++iteration) {
                for(size_t index=0;index<shapes.size();++index) {
                    const auto& shape=shapes[index];
                    const size_t count=rank==root?shape.root_records:shape.remote_records;
                    const size_t total=shape.root_records+shape.remote_records;
                    const float poison=-991.0f-float(rank);
                    std::fill_n(records.data(),records.size(),poison);
                    for(size_t record=0;record<count;++record)
                        for(size_t column=0;column<width;++column)
                            records[record*width+column]=valueAt(rank,record,column);
                    size_t gathered=99;
                    // Outside the measured call: both original producers are ready.
                    // No count exchange, payload, allocation or barrier is added to
                    // either native gather. This is not model arrival-skew evidence.
                    ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD),MPI_SUCCESS);
                    const auto start=Clock::now();
                    const bool success=shared_arm?
                        CPURootedPublicationTransportReplayAccess::gatherSharedOriginal(shared,
                            records.data(),count,rank==root?records.data():nullptr,capacity,width,root,gathered,range_span):
                        mpi->gatherVariableFloatRecordsToRoot(records.data(),count,
                            rank==root?records.data():nullptr,capacity,width,root,gathered,stage_name,range_span);
                    const auto stop=Clock::now();
                    ASSERT_TRUE(success);
                    if(iteration>=0)samples[index][iteration]=
                        std::chrono::duration<double,std::micro>(stop-start).count();
                    /** Validate source counts/bytes and every unused element, including native count-word tail. */
                    const auto outputValid=[&] {
                        const size_t used=rank==root?total:count;
                        if(gathered!=(rank==root?total:0))return false;
                        if(rank==root) {
                            if(!validRootedFloatRecordRanges(range_span,total) ||
                               ranges[root].first_record!=0 || ranges[root].record_count!=shape.root_records ||
                               ranges[1-root].first_record!=shape.root_records || ranges[1-root].record_count!=shape.remote_records)
                                return false;
                            for(const auto& range:range_span)
                                for(size_t record=0;record<range.record_count;++record)
                                    for(size_t column=0;column<width;++column)
                                        if(std::bit_cast<uint32_t>(records[(range.first_record+record)*width+column])!=
                                           std::bit_cast<uint32_t>(valueAt(range.participant,record,column)))return false;
                        } else for(size_t record=0;record<count;++record)
                            for(size_t column=0;column<width;++column)
                                if(std::bit_cast<uint32_t>(records[record*width+column])!=
                                   std::bit_cast<uint32_t>(valueAt(rank,record,column)))return false;
                        return std::all_of(records.data()+used*width,records.data()+records.size(),
                            [&](float value){return std::bit_cast<uint32_t>(value)==std::bit_cast<uint32_t>(poison);});
                    };
                    EXPECT_TRUE(outputValid())<<shape.name<<" shared="<<shared_arm;
                    if(iteration==0 && index==0) {
                        const size_t tail=(rank==root?total:count)*width;
                        records[tail]=42;EXPECT_FALSE(outputValid());records[tail]=poison;
                        const float saved=records[0];records[0]=-1;
                        EXPECT_FALSE(outputValid());records[0]=saved;
                        EXPECT_TRUE(outputValid());
                    }
                }
            }
            for(size_t index=0;index<shapes.size();++index) {
                const auto slowest=summarizeSlowestRank(samples[index]);
                if(rank!=root)continue;
                auto& sample=samples[index];
                const auto& shape=shapes[index];
                for(size_t iteration=0;iteration<sample.size();++iteration)
                    std::cout<<"original_numa_gather_raw,"<<shape.name<<','
                        <<(shared_arm?"shared_spin":"production_mpi_bandwidth")<<','<<iteration<<','
                        <<std::fixed<<std::setprecision(3)<<sample[iteration]<<'\n';
                std::sort(sample.begin(),sample.end());
                const auto root_p50=percentile(sample,50),root_p95=percentile(sample,95);
                const size_t remote_bytes=shape.remote_records*record_bytes;
                std::cout<<"original_numa_gather,"<<shape.name<<','<<(shared_arm?"shared_spin":"production_mpi_bandwidth")
                    <<','<<root<<','<<nodes[root]<<','<<nodes[1-root]<<','<<width<<','<<capacity
                    <<','<<shape.root_records*record_bytes<<','<<remote_bytes<<','<<(shape.root_records+shape.remote_records)*record_bytes
                    <<','<<std::fixed<<std::setprecision(3)<<root_p50<<','<<root_p95<<','<<slowest.p50_us<<','<<slowest.p95_us
                    <<','<<(remote_bytes?double(remote_bytes)/slowest.p50_us/1000.0:0.0)<<'\n';
            }
        }
        // A backend may be entered by ONE member of an unrelated existing
        // team. Only that original caller executes the protocol; it must not
        // conscript absent siblings into a nested collective workshare.
        {
            const size_t count=four_mib+1u;
            const float poison=-771.0f-float(rank);
            std::fill_n(records.data(),records.size(),poison);
            for(size_t record=0;record<count;++record)
                for(size_t column=0;column<width;++column)
                    records[record*width+column]=valueAt(rank,record,column);
            size_t gathered=0;
            bool success=false;
            ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD),MPI_SUCCESS);
#pragma omp parallel num_threads(2) shared(success,gathered)
            {
#pragma omp master
                {
                    success=CPURootedPublicationTransportReplayAccess::gatherSharedOriginal(shared,
                        records.data(),count,rank==root?records.data():nullptr,capacity,width,root,gathered,range_span);
                }
            }
            ASSERT_TRUE(success);
            EXPECT_EQ(gathered,rank==root?2u*count:0u);
            const size_t used=rank==root?2u*count:count;
            if(rank==root)
                for(const auto& range:range_span)
                    for(size_t record=0;record<range.record_count;++record)
                        for(size_t column=0;column<width;++column)
                            ASSERT_EQ(std::bit_cast<uint32_t>(records[(range.first_record+record)*width+column]),
                                std::bit_cast<uint32_t>(valueAt(range.participant,record,column)));
            EXPECT_TRUE(std::all_of(records.data()+used*width,records.data()+records.size(),
                [&](float value){return std::bit_cast<uint32_t>(value)==std::bit_cast<uint32_t>(poison);}));
        }
        // Retained original borrowers must prevent retirement BEFORE any
        // native unmap/acknowledgement; the same owner remains usable afterwards.
        {
            auto borrower=std::shared_ptr<IGlobalTPContext>(shared);
            EXPECT_THROW(shared->retireOriginalCPUExpertDomain(),std::logic_error);
            EXPECT_TRUE(shared->authenticatesOriginalCPUExpertBorrow(borrower));
        }
        ASSERT_TRUE(shared->retireOriginalCPUExpertDomain());
        EXPECT_TRUE(shared->retireOriginalCPUExpertDomain());
        ASSERT_TRUE(mpi->retireOriginalCPUExpertDomain());
        EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ActivationTransportStaging,
                                      PhysicalMemoryMaterializationKind::NewAllocation),0u);
        rejected=99;
        EXPECT_FALSE(CPURootedPublicationTransportReplayAccess::gatherSharedOriginal(shared,
            records.data(),0,rank==root?records.data():nullptr,capacity,width,root,rejected,range_span));
        EXPECT_EQ(rejected,0u);
        EXPECT_TRUE(std::all_of(range_span.begin(),range_span.end(),[](const auto& range){return range.participant==-1 && range.first_record==0 && range.record_count==0;}));
        // No backend pointer escapes. Original session owns the process pool
        // until mpi_gtest_main performs its unchanged production finalization.
        shared.reset();mpi.reset();
    }

    /**
     * @test NativeEpochAbortWitnessReplay
     * Replay the original admitted CPU backend's terminal epoch and ready-peer
     * boundaries. The same original mapping, PMA and communicator retire only
     * after both ranks leave the replay; no payload or alternative owner is made.
     */
    TEST(Perf__CPURootedPublicationTransport,NativeEpochAbortWitnessReplay)
    {
        const int rank=worldRank();
        ASSERT_EQ(worldSize(),2);
        const auto inventory=MPIContextFactory::global()->clusterInventory();
        ASSERT_EQ(inventory->ranks.size(),2u);
        std::vector<NativeCPUCollectiveProgram::Participant> members;
        for(const auto& observed:inventory->ranks)
        {
            ASSERT_EQ(observed.hostname,inventory->ranks.front().hostname);
            members.push_back({observed.rank,
                GlobalDeviceAddress::cpu(observed.cpu.numa_node,observed.hostname)});
        }
        ASSERT_EQ(members[0].world_rank,0);ASSERT_EQ(members[1].world_rank,1);
        const auto program=NativeCPUCollectiveProgram::expertShardRoutePartials(
            73004,CollectiveBackendType::UPI,members);
        const auto observed=observeSystemMemoryCapacity();
        ASSERT_TRUE(observed.valid());ASSERT_GT(observed.admission_available_bytes,0u);
        const PhysicalMemoryResource cpu{rank,DeviceId::cpu(),observed.total_bytes,
            observed.admission_available_bytes};
        PhysicalMemoryPlanBuilder bill;
        program.contributeDomainBOM(bill,cpu,rank);
        NativeCPUCollectiveProgram::contributeProcessPoolBOM(bill,cpu);
        if(rank==0)bill.add(cpu,PhysicalMemoryOwner::ActivationTransportStaging,
            ShmemSpinArena::allocationGeometry(2).mapped_bytes);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bill.build()),rank);
        using Use=std::variant<NativeCPUCollectiveProgram::DomainUse,NativeCPUCollectiveProgram::ProcessUse>;
        auto context=GlobalTPContext::createAdmittedCPUExpertDomain(MPI_COMM_WORLD,program,
            Use(std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,program.admit(memory,rank)));
        ASSERT_TRUE(context);ASSERT_EQ(context->scope(),TPScope::NODE_LOCAL);
        CPURootedPublicationTransportReplayAccess::replayEpochAbort(context);
        ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD),MPI_SUCCESS);
        ASSERT_TRUE(context->retireOriginalCPUExpertDomain());
        EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeCPUCollectiveDomain),0u);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }

    /**
     * @test SweepNativeSharedMemoryAndMPIByPayloadSize
     *
     * Measures each candidate after independent warmup. Gather payload bytes are
     * the sum of both participant contributions; broadcast bytes are the compact
     * result size delivered to the peer. Canonical timing excludes initialization,
     * allocation, validation output, and cross-rank sample reduction.
     */
    TEST(
        Perf__CPURootedPublicationTransport,
        SweepNativeSharedMemoryAndMPIByPayloadSize)
    {
        const int rank = worldRank();
        const int size = worldSize();
        if (size != 2)
            GTEST_SKIP() << "Rooted publication crossover requires exactly two ranks";

        auto upi =
            std::make_unique<UPICollectiveBackend>(MPI_COMM_WORLD, nullptr);
        ShmemSpinBackend shmem(/*domain_id=*/73001, rank, std::move(upi));
        ASSERT_TRUE(shmem.initialize(makeCpuGroup(size))) << shmem.lastError();

        const std::vector<size_t> payload_bytes = {
            8u * 1024u,
            16u * 1024u,
            32u * 1024u,
            64u * 1024u,
            128u * 1024u,
            256u * 1024u,
            512u * 1024u,
            1u * 1024u * 1024u,
            2u * 1024u * 1024u,
            4u * 1024u * 1024u,
            8u * 1024u * 1024u,
            16u * 1024u * 1024u,
            32u * 1024u * 1024u,
        };

        if (rank == 0)
        {
            std::cout << "operation,bytes,transport,p50_us,p95_us,p50_gbps\n";
        }

        for (const size_t bytes : payload_bytes)
        {
            ASSERT_EQ(bytes % (2u * sizeof(float)), 0u);
            const size_t total_elements = bytes / sizeof(float);
            const size_t local_elements = total_elements / 2u;
            std::vector<float> local(local_elements);
            std::iota(
                local.begin(),
                local.end(),
                static_cast<float>(rank * 1024));
            std::vector<float> root_records;
            if (rank == 0)
                root_records.resize(total_elements);

            const int repetitions = repetitionsForBytes(bytes);
            const auto run_gather = [&](bool native_shared_memory) {
                std::vector<double> local_samples(
                    static_cast<size_t>(repetitions));
                for (int warmup = 0; warmup < 3; ++warmup)
                {
                    size_t gathered = 0u;
                    if (native_shared_memory)
                    {
                        EXPECT_TRUE(shmem.gatherVariableFloatRecordsToRoot(
                            local.data(),
                            local_elements,
                            rank == 0 ? root_records.data() : nullptr,
                            total_elements,
                            /*record_width_elements=*/1u,
                            /*root_rank=*/0,
                            gathered));
                    }
                    else if (rank == 0)
                    {
                        std::memcpy(
                            root_records.data(),
                            local.data(),
                            local_elements * sizeof(float));
                        MPI_Request request = MPI_REQUEST_NULL;
                        EXPECT_EQ(
                            MPI_Irecv(
                                root_records.data() + local_elements,
                                static_cast<int>(local_elements),
                                MPI_FLOAT,
                                1,
                                7301,
                                MPI_COMM_WORLD,
                                &request),
                            MPI_SUCCESS);
                        EXPECT_EQ(
                            MPI_Wait(&request, MPI_STATUS_IGNORE),
                            MPI_SUCCESS);
                    }
                    else
                    {
                        MPI_Request request = MPI_REQUEST_NULL;
                        EXPECT_EQ(
                            MPI_Isend(
                                local.data(),
                                static_cast<int>(local_elements),
                                MPI_FLOAT,
                                0,
                                7301,
                                MPI_COMM_WORLD,
                                &request),
                            MPI_SUCCESS);
                        EXPECT_EQ(
                            MPI_Wait(&request, MPI_STATUS_IGNORE),
                            MPI_SUCCESS);
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                for (int repetition = 0; repetition < repetitions; ++repetition)
                {
                    size_t gathered = 0u;
                    const auto start = Clock::now();
                    if (native_shared_memory)
                    {
                        EXPECT_TRUE(shmem.gatherVariableFloatRecordsToRoot(
                            local.data(),
                            local_elements,
                            rank == 0 ? root_records.data() : nullptr,
                            total_elements,
                            /*record_width_elements=*/1u,
                            /*root_rank=*/0,
                            gathered));
                    }
                    else if (rank == 0)
                    {
                        std::memcpy(
                            root_records.data(),
                            local.data(),
                            local_elements * sizeof(float));
                        MPI_Request request = MPI_REQUEST_NULL;
                        EXPECT_EQ(
                            MPI_Irecv(
                                root_records.data() + local_elements,
                                static_cast<int>(local_elements),
                                MPI_FLOAT,
                                1,
                                7301,
                                MPI_COMM_WORLD,
                                &request),
                            MPI_SUCCESS);
                        EXPECT_EQ(
                            MPI_Wait(&request, MPI_STATUS_IGNORE),
                            MPI_SUCCESS);
                    }
                    else
                    {
                        MPI_Request request = MPI_REQUEST_NULL;
                        EXPECT_EQ(
                            MPI_Isend(
                                local.data(),
                                static_cast<int>(local_elements),
                                MPI_FLOAT,
                                0,
                                7301,
                                MPI_COMM_WORLD,
                                &request),
                            MPI_SUCCESS);
                        EXPECT_EQ(
                            MPI_Wait(&request, MPI_STATUS_IGNORE),
                            MPI_SUCCESS);
                    }
                    const auto finish = Clock::now();
                    local_samples[static_cast<size_t>(repetition)] =
                        std::chrono::duration<double, std::micro>(
                            finish - start)
                            .count();
                }
                return summarizeSlowestRank(local_samples);
            };

            printResult("gather", bytes, "shmem", run_gather(true));
            printResult("gather", bytes, "mpi", run_gather(false));

            std::vector<float> broadcast_payload(total_elements, 0.0f);
            if (rank == 0)
                std::iota(broadcast_payload.begin(), broadcast_payload.end(), 1.0f);
            const auto run_broadcast = [&](bool native_shared_memory) {
                std::vector<double> local_samples(
                    static_cast<size_t>(repetitions));
                for (int warmup = 0; warmup < 3; ++warmup)
                {
                    if (native_shared_memory)
                    {
                        EXPECT_TRUE(shmem.broadcast(
                            broadcast_payload.data(),
                            total_elements,
                            CollectiveDataType::FLOAT32,
                            0));
                    }
                    else
                    {
                        EXPECT_EQ(
                            MPI_Bcast(
                                broadcast_payload.data(),
                                static_cast<int>(total_elements),
                                MPI_FLOAT,
                                0,
                                MPI_COMM_WORLD),
                            MPI_SUCCESS);
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                for (int repetition = 0; repetition < repetitions; ++repetition)
                {
                    const auto start = Clock::now();
                    if (native_shared_memory)
                    {
                        EXPECT_TRUE(shmem.broadcast(
                            broadcast_payload.data(),
                            total_elements,
                            CollectiveDataType::FLOAT32,
                            0));
                    }
                    else
                    {
                        EXPECT_EQ(
                            MPI_Bcast(
                                broadcast_payload.data(),
                                static_cast<int>(total_elements),
                                MPI_FLOAT,
                                0,
                                MPI_COMM_WORLD),
                            MPI_SUCCESS);
                    }
                    const auto finish = Clock::now();
                    local_samples[static_cast<size_t>(repetition)] =
                        std::chrono::duration<double, std::micro>(
                            finish - start)
                            .count();
                }
                return summarizeSlowestRank(local_samples);
            };

            printResult("broadcast", bytes, "shmem", run_broadcast(true));
            printResult("broadcast", bytes, "mpi", run_broadcast(false));
        }

        shmem.shutdown();
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * @test Cached-output mechanism control for original combine+preweight.
     * Inputs are authored ready records at observed live-row scales, not a
     * recorded route/gather replay. All large banks use the existing PMA/NUMA
     * authority. Native mapped-ticket/model qualification remains separate.
     */
    TEST(Perf__CPURootedPublicationTransport, RootCombinePreweightReadyRecordEconomy)
    {
        ASSERT_EQ(worldSize(),1)<<"One original session worker on one physical socket is required";
        constexpr int maximum_rows=448,top_k=10,columns=2560;
        constexpr size_t slots=size_t(maximum_rows)*top_k,live_max=3208,record_width=columns+1;
        const int workers=omp_get_max_threads();
        ASSERT_GT(workers,0);
        int profile_repetitions=0;
        if(const char* raw=std::getenv("LLAMINAR_CPU_ROOTED_PUBLICATION_PROFILE_ITERATIONS")) {
            const std::string_view value(raw);
            const auto parsed=std::from_chars(value.data(),value.data()+value.size(),profile_repetitions);
            ASSERT_TRUE(parsed.ec==std::errc{} && parsed.ptr==value.data()+value.size() &&
                        profile_repetitions>0 && profile_repetitions<=2000)
                <<"Explicit root-publication profile iterations must be1..2000";
            ASSERT_EQ(workers,28)<<"This bounded diagnostic requires28actual physical workers";
            ASSERT_EQ(omp_get_level(),0)<<"Serial map validation retains standalone production entry";
        }
        const auto affinity=NUMATopology::detectLocalNUMANode();
        ASSERT_TRUE(affinity.detection_succeeded);
        const auto observed=observeSystemMemoryCapacity();
        ASSERT_TRUE(observed.valid());
        const PhysicalMemoryResource cpu{0,DeviceId::cpu(),observed.total_bytes,observed.admission_available_bytes};
        const auto placement=CPUStoragePlacement::onNode(affinity.local_numa_node);
        const CPUStorageRoot<float> records_root(2*live_max*record_width,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
        const CPUStorageRoot<float> raw0_root(slots*columns,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
        const CPUStorageRoot<float> raw1_root(slots*columns,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
        const CPUStorageRoot<float> weighted0_root(slots*columns,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
        const CPUStorageRoot<float> weighted1_root(slots*columns,PhysicalMemoryOwner::ActivationArena,DeviceId::cpu(),placement);
        PhysicalMemoryPlanBuilder bill;
        bill.add(cpu,PhysicalMemoryOwner::ActivationArena,records_root.physicalBytes()+raw0_root.physicalBytes()+
            raw1_root.physicalBytes()+weighted0_root.physicalBytes()+weighted1_root.physicalBytes());
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bill.build()),0);
        auto records=records_root.allocate(memory),raw0=raw0_root.allocate(memory),raw1=raw1_root.allocate(memory),
            weighted0=weighted0_root.allocate(memory),weighted1=weighted1_root.allocate(memory);
        // First-touch large banks on the workers that consume them.
        int actual_workers=0;
#pragma omp parallel
        {
#pragma omp single
            actual_workers=omp_get_num_threads();
#pragma omp for schedule(static)
            for(long long i=0;i<static_cast<long long>(records.size());++i)records.data()[i]=float(i%31)*.03125f;
#pragma omp for schedule(static)
            for(long long i=0;i<static_cast<long long>(raw0.size());++i) {
                raw0.data()[i]=raw1.data()[i]=-991.0f;
                weighted0.data()[i]=weighted1.data()[i]=-993.0f;
            }
        }
        ASSERT_EQ(actual_workers,workers)<<"Dynamic/narrowed team is not the requested measurement";
        RecordProperty("actual_omp_workers",actual_workers);
        std::vector<int32_t> original(live_max),compact(live_max),experts(live_max,1),maps(2*slots);
        std::vector<float> weights(live_max,.3f),required(slots);
        const std::array<int,1> selected{1};
        std::array<RootedFloatRecordRange,2> ranges;
        constexpr int warmups=16,samples=30,repeats=8;
        RecordProperty("output_storage","cached_PMA_root_mechanism_control");
        RecordProperty("route_provenance","authored_ready_records_not_route_replay");
        RecordProperty("workers",workers);RecordProperty("numa_node",affinity.local_numa_node);
        for(const size_t live:{size_t{453},live_max}) {
            if(profile_repetitions && live!=live_max)continue;
            // Retained maximum storage is not the logical incoming chunk.
            // These live-count scales correspond to M64 and M448 respectively.
            const int rows=live==453?64:maximum_rows;
            const size_t live_slot_extent=static_cast<size_t>(rows)*top_k;
            for(size_t entry=0;entry<live;++entry) {
                // A sparse prefix at this scale has real holes in both output
                // coordinate systems. Neither capacity nor padding is work.
                original[entry]=static_cast<int32_t>(entry+entry/16);
                compact[entry]=static_cast<int32_t>(entry+entry/16);
                ASSERT_LT(original[entry],int32_t(live_slot_extent));
                for(size_t participant=0;participant<2;++participant)
                    ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(
                        records.data()+(participant*live+entry)*record_width,columns,size_t(original[entry])));
            }
            ranges={RootedFloatRecordRange{0,0,live},RootedFloatRecordRange{1,live,live}};
            const auto os=std::span<const int32_t>(original).first(live);
            const auto cs=std::span<const int32_t>(compact).first(live);
            const auto es=std::span<const int32_t>(experts).first(live);
            const auto ws=std::span<const float>(weights).first(live);
            cpu::moe::CanonicalShardRouteRecordArgs baseline{{records.data(),2*live*record_width},ranges,{},{},selected,
                maps,required,{raw0.data(),raw0.size()},512,rows,top_k,columns,
                cpu::moe::CanonicalShardSparseRouteView{os,es,ws}};
            auto fused=baseline;fused.combined_route_rows={raw1.data(),raw1.size()};
            fused.preweighted_output=cpu::moe::CanonicalRoutePreweightedOutput{
                {weighted1.data(),weighted1.size()},os,cs,ws};
            MoEOverlaySparseRows input;
            input.row_capacity=rows;input.entry_capacity=live_slot_extent;input.live_row_count=rows;
            input.live_entry_count=live;input.d_model=columns;input.top_k=top_k;
            input.original_route_slots_host=original.data();input.compact_route_slots_host=compact.data();input.route_weights_host=weights.data();
            const auto run=[&](int candidate) {
                if(candidate==1)return cpu::moe::combineCanonicalShardRouteRecords(fused);
                return cpu::moe::combineCanonicalShardRouteRecords(baseline) &&
                    preweightMoEOverlayCanonicalRouteRows(input,{raw0.data(),raw0.size()},
                        MoEOverlayCanonicalRouteSourceLayout::originalSlots(),{weighted0.data(),weighted0.size()},
                        MoEOverlayCanonicalRouteSourceIndex::CompactRouteSlot);
            };
            ASSERT_TRUE(run(0));ASSERT_TRUE(run(1));
            ASSERT_EQ(std::memcmp(raw0.data(),raw1.data(),raw0.size()*sizeof(float)),0);
            ASSERT_EQ(std::memcmp(weighted0.data(),weighted1.data(),weighted0.size()*sizeof(float)),0);
            if(profile_repetitions) {
                {
                    CPUExactTimingCollectionScope recording_pause;
                    for(int warm=0;warm<warmups;++warm)ASSERT_TRUE(run(1));
                }
                ASSERT_TRUE(run(1)); // Fresh original bytes, outside repeat body.
                ASSERT_EQ(std::memcmp(raw0.data(),raw1.data(),raw0.size()*sizeof(float)),0);
                ASSERT_EQ(std::memcmp(weighted0.data(),weighted1.data(),weighted0.size()*sizeof(float)),0);
                {
                    CPUExactTimingCollectionScope recording_pause;
                    // Exactly one unchanged operation is repeated. Do not put
                    // an outer parallel region around it: map validation is
                    // serial, then the original reducer opens its own team.
                    // This mode has no Clock interval or canonical samples.
                    for(int iteration=0;iteration<profile_repetitions;++iteration)
                        ASSERT_TRUE(run(1))<<"fused profile iteration="<<iteration;
                }
                ASSERT_EQ(std::memcmp(raw0.data(),raw1.data(),raw0.size()*sizeof(float)),0);
                ASSERT_EQ(std::memcmp(weighted0.data(),weighted1.data(),weighted0.size()*sizeof(float)),0);
                RecordProperty("profile_body_calls",profile_repetitions);
                RecordProperty("profile_live_records_per_participant",live);
                RecordProperty("profile_output_storage","cached_PMA_root_mechanism_control");
                std::cout<<"ROOT_COMBINE_PREWEIGHT_PROFILE,candidate=fused,profile_kind=sampled_IP_repeat"
                    <<",profile_body_calls="<<profile_repetitions
                    <<",live="<<live<<",H="<<columns<<",M="<<rows<<",top_k="<<top_k
                    <<",participants=2,workers="<<actual_workers
                    <<",team_entry=standalone,live_slot_extent="<<live_slot_extent
                    <<",storage_slots="<<slots<<",preconditioning_calls="<<warmups
                    <<",fused_proof_calls=2,control_proof_calls=1"
                    <<",gather_calls=0,ticket_publication_calls=0"
                    <<",sampling_scope=whole_lifetime_setup_and_oracles_included"
                    <<",body_scope=ready_record_map_validation_reduce_preweight"
                    <<",output_storage=cached_PMA_root_mechanism_control"
                    <<",route_provenance=authored_ready_records_not_route_replay"
                    <<",both_outputs_byte_exact=1,canonical_timing_samples=0\n";
                return; // Original two-shape/two-arm timing path remains below.
            }
            for(int warm=0;warm<warmups;++warm){ASSERT_TRUE(run(warm%2));}
            std::array<std::array<double,samples>,2> timings{};
            for(int sample=0;sample<samples;++sample)for(int offset=0;offset<2;++offset) {
                const int candidate=(sample+offset)%2;
                const auto begin=std::chrono::steady_clock::now();
                bool ok=true;for(int repeat=0;repeat<repeats;++repeat)ok=run(candidate)&&ok;
                const auto end=std::chrono::steady_clock::now();
                ASSERT_TRUE(ok);
                timings[candidate][sample]=std::chrono::duration<double,std::micro>(end-begin).count()/repeats;
            }
            ASSERT_EQ(std::memcmp(raw0.data(),raw1.data(),raw0.size()*sizeof(float)),0);
            ASSERT_EQ(std::memcmp(weighted0.data(),weighted1.data(),weighted0.size()*sizeof(float)),0);
            for(int candidate=0;candidate<2;++candidate) {
                auto sorted=timings[candidate];std::sort(sorted.begin(),sorted.end());
                std::cout<<"ROOT_COMBINE_PREWEIGHT,cached_output_mechanism_control,authored_ready_records"
                    <<",candidate="<<(candidate?"fused":"original_two_pass")<<",live="<<live<<",H="<<columns
                    <<",M="<<rows<<",live_slot_extent="<<live_slot_extent<<",storage_slots="<<slots
                    <<",workers="<<workers<<",median_us="<<(sorted[14]+sorted[15])*.5
                    <<",p10_us="<<sorted[3]<<",p90_us="<<sorted[27]<<",raw_us=";
                for(const double value:timings[candidate])std::cout<<value<<';';
                std::cout<<",both_outputs_byte_exact=1\n";
            }
        }
    }
} // namespace llaminar2::test
