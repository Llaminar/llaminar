/**
 * @file MoEFollowerPredecessorCaptureReplay.inc
 * @brief Actual two-stream capture replay protecting one original GPU epoch slot.
 *
 * Both retained native graphs acquire/release the SAME arena request ticket0.
 * An asynchronous phase gate proves the producer holds ticket0 before the
 * consumer starts. An explicit no-edge control proves original concurrent
 * reuse fails, while the production edge makes the same captures succeed. The production cache
 * consumer API must join its exact original terminal before a different cache
 * can touch that shared slot. Request-reset ingress and terminal observations
 * occur outside the graph transaction; the consumer API performs no host fence.
 * This is native event/epoch ownership proof, not learned-model inference.
 */
#include "backends/NativeContextPreparation.h"
#include "backends/NativeOrderingEventGroup.h"
#include "backends/GPUGraphMemoryContract.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "backends/ComputeBackend.h"
#include "fixtures/backends/FollowerPredecessorDiagnosticGraphResources.h"
#include <optional>
#include <iostream>

namespace llaminar2::test { namespace {
/** @brief Report a concrete native component failure without skipping admitted work. */
void requireFollowerPredecessor(bool valid,const char *error)
{ if(!valid)throw std::runtime_error(error); }

/** @brief Original startup inventory observation, never fabricated device capacity. */
PhysicalMemoryResource followerPredecessorResource(DeviceId device)
{
    auto &manager=DeviceManager::instance();manager.initialize(-1,false);
    const auto backend=device.is_cpu()?ComputeBackendType::CPU:
        device.is_cuda()?ComputeBackendType::GPU_CUDA:ComputeBackendType::GPU_ROCM;
    const auto &devices=manager.devices();
    const auto found=std::find_if(devices.begin(),devices.end(),[&](const auto &entry) {
        return entry.type==backend && (device.is_cpu() || entry.device_id==device.ordinal);
    });
    requireFollowerPredecessor(found!=devices.end(),"Native predecessor endpoint is absent from original startup inventory");
    return {0,device,found->total_memory_bytes,found->free_memory_bytes};
}

/**
 * @brief Replay two actual cache streams over the same slot, then failed admission and reset.
 * @param device One available physical GPU; no participant emulation is used.
 */
void replayFollowerPredecessorCapture(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<1)GTEST_SKIP()<<"Requires one actual "<<device.toString()<<" endpoint";
    using Cache=DeviceGraphExecutor::GraphSegmentCache;
    const auto cpu=followerPredecessorResource(DeviceId::cpu());
    const auto gpu=followerPredecessorResource(device);
    const std::array resources{cpu,gpu};const std::array selected{device};
    auto source=NativeContextPreparation::create(resources,0,selected);
    std::optional<PreparedGPUContextUsePermit> permit(source->acquire(device));
    const auto memory=source->memory();
    source->freezeForPlanning();
    PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
    const auto terminal_program=NativeOrderingEventProgram::captureSubmissionRetirement();
    // Each actual cache owns one independently certified original event group.
    const auto terminal_bom=NativeOrderingEventGroup::bom(terminal_program,cpu,gpu);
    for(int cache=0;cache<2;++cache)
        for(const auto &entry:terminal_bom.resources())plan.add(entry);
    const auto arena_bytes=DeviceMoEOverlayEpochArena::allocationBytesForSlots(1u);
    const std::size_t gate_bytes=device.is_rocm()?sizeof(std::uint64_t):sizeof(std::uint32_t);
    const auto diagnostic_graph_bom=FollowerPredecessorDiagnosticGraphResources::bom(cpu,gpu);
    for(const auto &entry:diagnostic_graph_bom.resources())plan.add(entry);
    plan.add(gpu,PhysicalMemoryOwner::ExecutionWorkspace,arena_bytes);
    plan.add(gpu,PhysicalMemoryOwner::ActivationArena,gate_bytes);
    source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
    // Raw diagnostic graphs retain both original General reservations through
    // ProcessRuntime. This does not claim the production retained-owner API.
    std::array<std::optional<NativeContextRetirementSlot>,2> graph_backing;
    for(auto &slot:graph_backing)slot.emplace(FollowerPredecessorDiagnosticGraphResources::prepare(*permit));
    auto worker_owner=permit->worker();permit.reset();
    auto &worker=*worker_owner;
    std::exception_ptr failure;
    {
        std::shared_ptr<DeviceMoEOverlayEpochArena> arena;
        std::array<Cache,2> caches;
        std::array<std::unique_ptr<IGPUGraphCapture>,2> graphs;
        std::array<PhysicalMemoryAllocationLease,2> graph_object_claims;
        std::array<bool,2> native_attempted{};
        bool producer_launched=false;
        void *gate=nullptr;
        std::optional<PhysicalMemoryAllocationLease> gate_claim;
        std::array<Cache::CaptureStreamTerminalTicket,2> pending{};
        std::size_t waits=0u,successful_pairs=0u,failed_successors=0u,no_edge_controls=0u;
        // Every exceptional exit releases only the deliberate fixture gate,
        // then observes every actually published original terminal before free.
        const auto release_and_drain=[&] {
            worker.submitAndWait([&] {
                if(gate && pending[0].valid() && producer_launched) {
                    // Device-order release AFTER A's actual phase1 publication.
                    // Publishing2 earlier could be overwritten by A's later1.
                    requireFollowerPredecessor(backend->streamWaitTimelineSignal32(
                        worker.defaultStream(),gate,1u,device.ordinal),"Native predecessor fixture release dependency failed");
                    requireFollowerPredecessor(backend->streamPublishTimelineSignal32(
                        worker.defaultStream(),gate,2u,device.ordinal),"Native predecessor fixture gate release failed");
                }
            });
            for(std::size_t index=0;index<pending.size();++index)if(pending[index].valid()) {
                caches[index].waitForPublishedCaptureStreamTerminal(pending[index],Cache::HostFenceWaitPolicy::ActiveProgress);
                pending[index]={};
                if(index==0u)producer_launched=false;
            }
        };
        try {
            worker.submitAndWait([&] {
                arena=std::make_shared<DeviceMoEOverlayEpochArena>(DeviceMoEOverlayEpochArena::Config{
                    .device_id=device,.memory_authority=memory});
                gate_claim.emplace(memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,gate_bytes));
                requireFollowerPredecessor(backend->supportsStreamTimelineSignal32(device.ordinal),"Native source input gate is unavailable");
                gate=backend->allocateStreamTimelineSignal32(device.ordinal);
                requireFollowerPredecessor(gate!=nullptr,"Native predecessor input gate allocation failed");
                for(std::size_t index=0;index<caches.size();++index) {
                    requireFollowerPredecessor(caches[index].ensureCaptureStream(&worker,device,false),"Native cache stream preparation failed");
                    requireFollowerPredecessor(caches[index].prepareCaptureStreamTerminal(&worker,memory),"Certified native cache terminal preparation failed");
                    graph_object_claims[index]=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,
                        nativeGraphCaptureObjectBytes(device));
                    graph_backing[index]->armNativeAttempt();native_attempted[index]=true;
                    graphs[index]=worker.createGraphCapture(caches[index].capture_stream);
                    requireFollowerPredecessor(graphs[index] && graphs[index]->beginCapture(),"Native shared-slot graph capture failed");
                    auto kernel=llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
                    requireFollowerPredecessor(bool(kernel),"Original epoch kernel is unavailable");
                    const MoEKernelLaunchContext launch{.stream=caches[index].capture_stream};
                    auto *status=index==0u?arena->requestStatus(0u):arena->maintenanceStatus();
                    // Consumer eligibility follows actual producer acquisition,
                    // so removing only the terminal edge deterministically exposes
                    // concurrent reuse of the SAME original ticket (the control below).
                    bool okay=true;
                    if(index==1u)okay=backend->streamWaitTimelineSignal32(caches[index].capture_stream,gate,1u,device.ordinal);
                    okay=kernel->acquireMoEOverlayEpoch(launch,arena->control(),arena->requestTicket(0u),status) && okay;
                    if(index==0u) {
                        okay=backend->streamPublishTimelineSignal32(caches[index].capture_stream,gate,1u,device.ordinal) && okay;
                        okay=backend->streamWaitTimelineSignal32(caches[index].capture_stream,gate,2u,device.ordinal) && okay;
                    }
                    okay=kernel->releaseMoEOverlayEpoch(launch,arena->control(),arena->requestTicket(0u),status) && okay;
                    const bool ended=graphs[index]->endCapture();
                    requireFollowerPredecessor(okay && ended,"Native shared-slot body did not capture completely");
                    std::string detail;
                    requireFollowerPredecessor(graphs[index]->validateExecutableMemoryClass(
                        GPUGraphExecutableClass::General,&detail),detail.c_str());
                    requireFollowerPredecessor(graphs[index]->instantiate(),"Native shared-slot graph instantiate failed");
                }
                requireFollowerPredecessor(caches[0].capture_stream!=caches[1].capture_stream,"Native predecessor fixture caches alias one stream");
            });
            const auto reset_input=[&] {
                // Prior graph receipts have already retired. This is request
                // ingress, not a fence in the consumer or captured hot path.
                Cache::CaptureStreamTerminalTicket reset;
                worker.submitAndWait([&] {
                    requireFollowerPredecessor(backend->streamPublishTimelineSignal32(caches[0].capture_stream,
                        gate,0u,device.ordinal),"Native predecessor input reset failed");
                    reset=caches[0].publishCaptureStreamTerminal();
                });
                caches[0].waitForPublishedCaptureStreamTerminal(reset,Cache::HostFenceWaitPolicy::ActiveProgress);
            };
            // Original/no-edge control uses the SAME captured graph bodies.
            // Observe only B while A is physically held at phase1. B necessarily
            // sees the live shared ticket; its release consumes that ticket and
            // A's later release reports InvalidTicket. No production path calls
            // this deliberately unordered branch or its diagnostic observation.
            reset_input();
            worker.submitAndWait([&] {
                const bool producer=graphs[0]->launchOnStream(caches[0].capture_stream);
                producer_launched=producer;
                pending[0]=caches[0].publishCaptureStreamTerminal();
                requireFollowerPredecessor(producer,"No-edge producer launch failed");
                const bool consumer=graphs[1]->launchOnStream(caches[1].capture_stream);
                pending[1]=caches[1].publishCaptureStreamTerminal();
                requireFollowerPredecessor(consumer,"No-edge consumer launch failed");
            });
            caches[1].waitForPublishedCaptureStreamTerminal(pending[1],Cache::HostFenceWaitPolicy::ActiveProgress);
            pending[1]={};
            worker.submitAndWait([&] {
                bool complete=true;
                requireFollowerPredecessor(worker.queryEventChecked(caches[0].terminal_event,complete) && !complete,
                    "No-edge control lost the physically held original producer");
            });
            release_and_drain();
            DeviceMoEOverlayEpochStatus no_edge_status{};
            worker.submitAndWait([&] {
                requireFollowerPredecessor(backend->deviceToHost(&no_edge_status,arena->requestStatus(0u),
                    sizeof(no_edge_status),device.ordinal,caches[0].capture_stream),"No-edge original status output failed");
            });
            EXPECT_EQ(no_edge_status.operation,static_cast<std::uint32_t>(DeviceMoEOverlayEpochOperation::Release));
            EXPECT_EQ(no_edge_status.code,static_cast<std::uint32_t>(DeviceMoEOverlayEpochStatusCode::InvalidTicket));
            ++no_edge_controls;
            const auto replay_pair=[&](int replay) {
                SCOPED_TRACE(replay);reset_input();
                const auto published_before=caches[0].terminal_fence_published_generation;
                worker.submitAndWait([&] {
                    const bool launched=graphs[0]->launchOnStream(caches[0].capture_stream);
                    producer_launched=launched;
                    pending[0]=caches[0].publishCaptureStreamTerminal();
                    requireFollowerPredecessor(launched,"Native producer graph launch failed");
                    const auto observed=caches[0].terminal_fence_observed_generation;
                    caches[0].enqueuePublishedCaptureStreamTerminalWait(pending[0],caches[1].capture_stream,&worker);
                    ++waits;
                    requireFollowerPredecessor(caches[0].terminal_fence_observed_generation==observed &&
                        caches[0].terminal_fence_published_generation==published_before+1u,
                        "Consumer event edge observed or rerecorded the producer");
                    const bool consumer=graphs[1]->launchOnStream(caches[1].capture_stream);
                    pending[1]=caches[1].publishCaptureStreamTerminal();
                    requireFollowerPredecessor(consumer,"Native consumer graph launch failed");
                    bool complete=true;
                    requireFollowerPredecessor(worker.queryEventChecked(caches[1].terminal_event,complete) && !complete,
                        "Shared-slot consumer completed before its delayed original predecessor");
                });
                release_and_drain();
                DeviceMoEOverlayEpochControl control{};DeviceMoEOverlayEpochStatus first{},second{};
                worker.submitAndWait([&] {
                    requireFollowerPredecessor(backend->deviceToHost(&control,arena->control(),sizeof(control),device.ordinal,caches[1].capture_stream),"Native shared control output failed");
                    requireFollowerPredecessor(backend->deviceToHost(&first,arena->requestStatus(0u),sizeof(first),device.ordinal,caches[1].capture_stream),"Native producer status output failed");
                    requireFollowerPredecessor(backend->deviceToHost(&second,arena->maintenanceStatus(),sizeof(second),device.ordinal,caches[1].capture_stream),"Native consumer status output failed");
                });
                EXPECT_TRUE(first.succeeded());EXPECT_TRUE(second.succeeded());
                EXPECT_EQ(control.bank_readers[0],0u);EXPECT_EQ(control.bank_readers[1],0u);
                EXPECT_EQ(control.acquisitions_in_flight,0u);++successful_pairs;
            };
            for(int replay=0;replay<20;++replay)replay_pair(replay);
            // A rejected successor creates no second native publication. Its
            // already submitted producer still retires through the original event.
            reset_input();
            worker.submitAndWait([&] {
                const bool launched=graphs[0]->launchOnStream(caches[0].capture_stream);
                producer_launched=launched;
                pending[0]=caches[0].publishCaptureStreamTerminal();
                requireFollowerPredecessor(launched,"Partial native producer launch failed");
                const auto before=caches[1].terminal_fence_published_generation;
                EXPECT_THROW(caches[0].enqueuePublishedCaptureStreamTerminalWait(pending[0],nullptr,&worker),std::logic_error);
                EXPECT_EQ(caches[1].terminal_fence_published_generation,before);++failed_successors;
            });
            const auto observed_producer=pending[0];
            release_and_drain();
            worker.submitAndWait([&] {
                // A completed SAME current native receipt remains an authentic
                // redundant device edge; it consumes no new host observation.
                const auto observed=caches[0].terminal_fence_observed_generation;
                const auto published=caches[0].terminal_fence_published_generation;
                EXPECT_NO_THROW(caches[0].enqueuePublishedCaptureStreamTerminalWait(
                    observed_producer,caches[1].capture_stream,&worker));
                EXPECT_EQ(caches[0].terminal_fence_observed_generation,observed);
                EXPECT_EQ(caches[0].terminal_fence_published_generation,published);
            });
            reset_input(); // A newer real publication makes the old receipt stale.
            worker.submitAndWait([&] {
                EXPECT_THROW(caches[0].enqueuePublishedCaptureStreamTerminalWait(
                    observed_producer,caches[1].capture_stream,&worker),std::logic_error);
            });
            replay_pair(20); // A fresh request reuses both original captures after the rejected admission drained.
        } catch(...) { failure=std::current_exception(); }
        try { release_and_drain(); }
        catch(...) { std::terminate(); }
        // One last actual cache reset observes any last published ticket. It
        // destroys the same event handles only after the exact native proof.
        worker.submitAndWait([&] {
            for(std::size_t index=0;index<graphs.size();++index) {
                graphs[index].reset();graph_object_claims[index]={};
                if(native_attempted[index])graph_backing[index]->completeNativeHandles();
            }
            for(auto &cache:caches)cache.reset(Cache::StreamResetPolicy::Destroy);
            arena.reset();
            if(gate)backend->freeStreamTimelineSignal32(gate,device.ordinal);
            gate=nullptr;gate_claim.reset();
        });
        EXPECT_EQ(waits,21u);EXPECT_EQ(successful_pairs,21u);EXPECT_EQ(failed_successors,1u);EXPECT_EQ(no_edge_controls,1u);
    }
    for(auto &slot:graph_backing)slot.reset();
    worker_owner.reset();
    requireFollowerPredecessor(source->retireReleasedContexts()==1u,"Native predecessor original context did not retire");
    if(failure)std::rethrow_exception(failure);
    std::cout<<"follower_predecessor_native device="<<device.toString()
        <<" pairs=21 shared_slot=0 distinct_streams=2 no_edge_controls=1 partial_rejection=1 reset_without_recapture=1\n";
}
} // namespace
TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm, FollowerPredecessorCUDASharedSlotCapturedReplay)
{ replayFollowerPredecessorCapture(DeviceId::cuda(0)); }
TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm, FollowerPredecessorROCmSharedSlotCapturedReplay)
{ replayFollowerPredecessorCapture(DeviceId::rocm(0)); }
} // namespace llaminar2::test
