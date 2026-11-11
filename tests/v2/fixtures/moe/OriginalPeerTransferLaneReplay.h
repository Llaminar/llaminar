/**
 * @file OriginalPeerTransferLaneReplay.h
 * @brief Original-MPI/PMA ownership for the existing exact same-backend peer-lane oracles.
 *
 * One observed NUMA owner prepares two real GPUs. The existing production lane
 * retains its exact admitted background stream and native events; this fixture
 * adds only initial byte I/O, a genuine source readiness event and terminal
 * observations. No graph/communicator or synthetic participant is introduced.
 */
#pragma once
#include "fixtures/backends/OriginalRankLocalNativeReplay.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "backends/NativePeerTransferEventOwnership.h"
#include "planning/RankHardwareOwnership.h"
#include "execution/moe/ExpertTierGpuPeerTransferLane.h"
#include "execution/moe/GPUExpertTransferBackend.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
namespace llaminar2::test {
/** @brief Only a fixed one-ordering-event set may be created for original source I/O readiness. */
struct OriginalPeerTransferEventFixtureAccess final {
    static auto sourceReady(const PreparedGPUContextUsePermit& original,const std::string& instance)
    { return NativePeerTransferEventOwnership::materialize(original,false,instance); }
    static void* event(const NativePeerTransferEventOwnership& original){return original.event(0);}
};
/** @brief Print the first component failure before any required native-retirement diagnostic. */
inline void requireOriginalPeer(bool ok,const std::string& error)
{ if(!ok){std::cerr<<"Original peer transfer: "<<error<<'\n';throw std::runtime_error(error);} }
/** @brief Exact terminal queries retire all native I/O before event and allocation release. */
struct OriginalPeerTransferRetirement final {
    std::function<void()> retire;
    ~OriginalPeerTransferRetirement()noexcept {
        try {if(retire)retire();}catch(const std::exception& error){std::cerr<<"Original peer retirement: "<<error.what()<<'\n';std::terminate();}
    }
};
/** @brief Preserve packed or FP16/BF16/FP32 bit-exact production-lane semantics on the original owner. */
inline void originalPeerTransferReplay(DeviceType family,const char* precision=nullptr,
    std::size_t element_bytes=0,std::uint32_t seed=0,
    const std::function<bool(void*,int)>& background_priority={})
{
    const std::array families{NativeReplayFamily{family,2,{}}};
    originalRankLocalReplay(std::span<const NativeReplayFamily>(families),[&](const auto& selected) {
        const auto mpi=MPIContextFactory::global();
        const RankInventory* inventory=nullptr;
        for(const auto& rank:mpi->clusterInventory()->ranks)if(rank.rank==mpi->rank())inventory=&rank;
        requireOriginalPeer(inventory!=nullptr,"Original peer replay lost its original rank inventory");
        // Admission and execution share one directed observation predicate;
        // no failed candidate is materialized before selecting its true owner.
        const auto pair=observedNativeDirectedPeerPair(*inventory,family,
            [](DeviceId destination,DeviceId source) {
                return DeviceManager::instance().peerAccessAvailable(destination,source);
            });
        requireOriginalPeer(pair.has_value() && selected.size()==2,"Original NUMA-owned cohort has no directed peer edge");
        const std::vector<DeviceId> devices{(*pair)[0],(*pair)[1]};
        const auto source=devices[0],destination=devices[1];
        auto original=PreparedNativeLookupFixture::prepare(devices,mpi->rank());
        constexpr int n=2048,k=256;constexpr std::uint8_t payload_per_block=16;
        constexpr std::size_t blocks=std::size_t(n)*(k/32);
        const bool packed=precision==nullptr;
        requireOriginalPeer(packed || element_bytes==2 || element_bytes==4,"Original floating peer format has invalid scalar width");
        const std::array<std::size_t,4> parts=packed
            ?std::array<std::size_t,4>{blocks*payload_per_block,blocks*2,blocks*2,blocks*4}
            :std::array<std::size_t,4>{32771*element_bytes,0,0,0};
        std::array<std::size_t,4> offsets{};std::size_t bytes=0;
        for(std::size_t i=0;i<parts.size();++i){offsets[i]=bytes;bytes+=parts[i];}
        PhysicalMemoryPlanBuilder plan;
        const auto mapping_bytes=TransferEngine::mappedHostRegionAllocationBytes(2*bytes);
        plan.add(original->resource(DeviceId::cpu()),PhysicalMemoryOwner::ExpertMigrationStaging,mapping_bytes);
        for(const auto device:devices)plan.add(original->resource(device),PhysicalMemoryOwner::ExpertMigrationStaging,bytes);
        const auto add=[&](const PhysicalMemoryPlan& quote){for(const auto& row:quote.resources())plan.add(row);};
        add(TransferEngine::persistentTransferExecutionLanesBOM(1,original->resource(DeviceId::cpu()),original->resource(destination)));
        add(ExpertTierGpuPeerTransferLane::nativeEventsBOM(original->resource(DeviceId::cpu()),original->resource(source),false));
        add(ExpertTierGpuPeerTransferLane::nativeEventsBOM(original->resource(DeviceId::cpu()),original->resource(destination),true));
        original->admit(plan.build());
        {
            auto memory=original->memory();TransferEngine& transfer=TransferEngine::instance();
            const auto producer=original->permit(source).worker(),consumer=original->permit(destination).worker();
            void* const producer_stream=producer->defaultStream();void* const consumer_stream=consumer->defaultStream();
            requireOriginalPeer(reinterpret_cast<std::uintptr_t>(producer_stream)>2 && reinterpret_cast<std::uintptr_t>(consumer_stream)>2,
                "Original peer fixture requires admitted explicit non-default native streams");
            // These exact claims outlive mapped/native allocations and every event borrower.
            std::vector<PhysicalMemoryAllocationLease> claims;
            claims.push_back(memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ExpertMigrationStaging,mapping_bytes));
            std::shared_ptr<MappedHostTransferRegion> mapped;
            std::array<std::shared_ptr<DeviceTransferBuffer>,2> buffers;
            producer->submitAndWait([&]{mapped=transfer.allocateMappedHostRegion(2*bytes,devices);});
            for(std::size_t i=0;i<devices.size();++i) {
                claims.push_back(memory->claimNewAllocation(devices[i],PhysicalMemoryOwner::ExpertMigrationStaging,bytes));
                original->permit(devices[i]).worker()->submitAndWait([&,i]{buffers[i]=transfer.allocateDeviceTransferBuffer(bytes,devices[i]);});
            }
            auto* expected=static_cast<std::uint8_t*>(mapped->mutableHostData());
            auto* actual=expected+bytes;
            if(packed) {
                std::mt19937 rng(family==DeviceType::CUDA?0xC0FFEEu:0xBADC0DEu);
                for(std::size_t i=0;i<parts[0];++i)expected[i]=static_cast<std::uint8_t>(rng());
                for(std::size_t region=1;region<3;++region)for(std::size_t i=0;i<blocks;++i)
                    reinterpret_cast<std::uint16_t*>(expected+offsets[region])[i]=static_cast<std::uint16_t>(rng());
                for(std::size_t i=0;i<blocks;++i)reinterpret_cast<std::uint32_t*>(expected+offsets[3])[i]=rng();
            } else for(std::size_t i=0;i<bytes;++i)expected[i]=static_cast<std::uint8_t>(family==DeviceType::CUDA
                ?(i*43u+seed*17u+(i>>3u))&0xffu:(i*47u+seed*19u+(i>>2u))&0xffu);
            auto ready=OriginalPeerTransferEventFixtureAccess::sourceReady(original->permit(source),
                std::string("original_peer_producer_ready:")+(precision?precision:"packed"));
            auto execution=transfer.allocatePersistentTransferExecutionLanes(1,destination,
                std::string("original_peer_")+(precision?precision:"packed"),original->permit(destination));
            if(background_priority)consumer->submitAndWait([&]{requireOriginalPeer(
                background_priority(execution.front().stream(),destination.ordinal),
                "Original peer stream did not retain nonblocking lowest-priority scheduling");});
            consumer->submitAndWait([&] {
                void* const stable=execution.front().stream();
                EXPECT_THROW(consumer->resetAuxiliaryStreams(),std::logic_error);
                EXPECT_EQ(execution.front().stream(),stable);
            });
            std::unique_ptr<ExpertTierGpuPeerTransferLane> lane;
            const auto awaitStream=[&](const std::shared_ptr<IWorkerGPUContext>& worker,void* stream) {
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
                for(;;){GPUStreamExecutionState state=GPUStreamExecutionState::Pending;
                    worker->submitAndWait([&]{state=worker->queryStreamExecutionState(stream,"original peer fixture terminal");});
                    if(state==GPUStreamExecutionState::Complete)break;
                    requireOriginalPeer(std::chrono::steady_clock::now()<deadline,"Original peer fixture terminal timed out");std::this_thread::yield();}
            };
            const auto poll=[&] {
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
                std::string error;
                while(lane->progress()==ExpertTierGpuPeerTransferProgress::Pending) {
                    const auto state=lane->poll(&error);
                    requireOriginalPeer(state!=ExpertTierGpuPeerTransferProgress::Failed,error);
                    requireOriginalPeer(std::chrono::steady_clock::now()<deadline,"Original peer completion event timed out");std::this_thread::yield();
                }
                requireOriginalPeer(lane->progress()==ExpertTierGpuPeerTransferProgress::Ready,"Original peer event did not publish Ready");
            };
            OriginalPeerTransferRetirement retirement{[&]{
                if(lane && lane->progress()==ExpertTierGpuPeerTransferProgress::Pending)poll();
                awaitStream(producer,producer_stream);awaitStream(consumer,consumer_stream);awaitStream(consumer,execution.front().stream());
                requireOriginalPeer(!lane || lane->quiescent(),"Original peer lane retained an unfenced native obligation");
            }};
            const auto ready_event=OriginalPeerTransferEventFixtureAccess::event(*ready);
            producer->submitAndWait([&]{
                transfer.enqueueMappedHostToPersistentDeviceRegion(*mapped,0,buffers[0]->mutableDeviceData(),bytes,0,bytes,source,producer_stream);
                producer->recordEvent(ready_event,producer_stream);
            });
            lane=std::make_unique<ExpertTierGpuPeerTransferLane>(ExpertTierGpuPeerTransferLane::Config{
                .source_device=source,.destination_device=destination,.execution=execution.front(),
                .lane_name=std::string("original_peer_")+(precision?precision:"packed"),.perf_device="original-peer",.collect_timing_measurements=true});
            std::string error;requireOriginalPeer(lane->materialize(&error),error);
            const auto events_host=memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeTransferEvent);
            const auto events_source=memory->reservedBytes(source,PhysicalMemoryOwner::NativeTransferEvent);
            const auto events_destination=memory->reservedBytes(destination,PhysicalMemoryOwner::NativeTransferEvent);
            const auto requireSameMaterialization=[&] {
                requireOriginalPeer(lane->materialize(&error),error);
                EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeTransferEvent),events_host);
                EXPECT_EQ(memory->reservedBytes(source,PhysicalMemoryOwner::NativeTransferEvent),events_source);
                EXPECT_EQ(memory->reservedBytes(destination,PhysicalMemoryOwner::NativeTransferEvent),events_destination);
            };
            requireSameMaterialization();
            auto* backend=getBackendFor(source);requireOriginalPeer(backend && backend->setDevice(source.ordinal),"Original peer source device selection failed");
            const auto current=[&] {
#ifdef HAVE_CUDA
                if(family==DeviceType::CUDA)return detail::currentCUDADeviceOrdinal();
#endif
#ifdef HAVE_ROCM
                if(family==DeviceType::ROCm)return detail::currentROCmDeviceOrdinal();
#endif
                throw std::logic_error("Original peer replay has no compiled native backend");
            };
            const auto descriptor=[&](std::size_t i) {
                GpuExpertPackedDescriptor d;d.ptrs.d_vnni=static_cast<std::uint8_t*>(buffers[i]->mutableDeviceData(offsets[0]));
                d.ptrs.d_scales=static_cast<std::uint16_t*>(buffers[i]->mutableDeviceData(offsets[1]));
                d.ptrs.d_mins=static_cast<std::uint16_t*>(buffers[i]->mutableDeviceData(offsets[2]));
                d.ptrs.d_emins=static_cast<std::uint32_t*>(buffers[i]->mutableDeviceData(offsets[3]));
                d.n=n;d.k=k;d.blocks_per_row=k/32;d.codebook_id=4;d.payload_bytes_per_block=payload_per_block;d.is_asymmetric=true;d.has_emins=true;
                d.vnni_bytes=parts[0];d.scales_bytes=parts[1];d.mins_bytes=parts[2];d.emins_bytes=parts[3];return d;
            };
            if(packed)requireOriginalPeer(lane->start(descriptor(0),descriptor(1),ExpertTierSourceReadiness::producerEvent(ready_event),&error),error);
            else requireOriginalPeer(lane->startContiguous(buffers[0]->deviceData(),buffers[1]->mutableDeviceData(),bytes,
                ExpertTierSourceReadiness::producerEvent(ready_event),&error),error);
            EXPECT_EQ(current(),source.ordinal);poll();requireSameMaterialization();
            if(packed) {
                requireOriginalPeer(backend->setDevice(source.ordinal),"Original peer bank reuse source selection failed");
                requireOriginalPeer(lane->start(descriptor(0),descriptor(1),ExpertTierSourceReadiness::publishedResidencyBank(17),&error),error);
                poll();EXPECT_EQ(current(),source.ordinal);requireSameMaterialization();
            }
            const auto stats=lane->stats();const auto runs=packed?2u:1u;
            EXPECT_EQ(stats.transfers_started,runs);EXPECT_EQ(stats.transfers_completed,runs);EXPECT_EQ(stats.bytes_submitted,bytes*runs);
            EXPECT_EQ(stats.producer_event_waits,1u);EXPECT_EQ(stats.published_bank_sources,packed?1u:0u);EXPECT_EQ(stats.failed_transfers,0u);
            EXPECT_EQ(stats.timing_measurement_failures,0u);EXPECT_TRUE(stats.last_measurement.valid());EXPECT_EQ(stats.last_measurement.bytes,bytes);
            EXPECT_GT(stats.last_measurement.device_nanoseconds,0u);EXPECT_EQ(stats.inference_stream_waits,0u);EXPECT_EQ(stats.blocking_synchronizations,0u);
            consumer->submitAndWait([&]{transfer.enqueuePersistentDeviceRegionToMappedHost(buffers[1]->deviceData(),bytes,0,*mapped,bytes,bytes,destination,consumer_stream);});
            awaitStream(consumer,consumer_stream);
            for(std::size_t region=0;region<parts.size();++region)if(parts[region]) {
                SCOPED_TRACE(::testing::Message()<<"region="<<region<<" bytes="<<parts[region]);
                EXPECT_EQ(std::memcmp(expected+offsets[region],actual+offsets[region],parts[region]),0);
            }
        }
        original->requirePayloadRetired();original->retire();original->requireExactRetired();
    },[&](const RankInventory &rank) {
        return observedNativeDirectedPeerPair(rank,family,
            [](DeviceId destination,DeviceId source) {
                return DeviceManager::instance().peerAccessAvailable(destination,source);
            }).has_value();
    });
}
}
