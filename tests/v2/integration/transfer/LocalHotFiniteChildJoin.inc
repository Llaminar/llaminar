/**
 * @file LocalHotFiniteChildJoin.inc
 * @brief Actual SDK finite-child and imported-main exit joins under one original General owner.
 *
 * Diagnostic copies stand for publication markers, not model history or a source
 * receipt. Four graph-only definitions share one executable and original PMA.
 * The explicit missing-edge control releases its held predecessor only AFTER
 * reading stale bytes, making the old missing dependency finite and deterministic.
 */
#include "backends/GPUFlatGraphStructure.h"

namespace llaminar2::test {
namespace {
/**
 * @brief Replay exact joined native definitions across two data-reset epochs.
 * @param device Actual GPU0 backend, with original prepared source/lifetime.
 * @param owned Use owned timeline main fragments or independently imported children.
 * @param missingRetire Explicit fixture-only negative control; never a production strategy.
 */
void nativeFiniteJoin(DeviceId device, bool owned, bool missingRetire)
{
    using Access = RetainedParticipantGraphExecutableFixtureAccess;
    const std::array devices{device};
    auto session = PreparedNativeLookupFixture::prepare(devices);
    const auto& original = session->permit(device);
    auto worker = original.worker();
    auto* backend = getBackendFor(device);
    const auto gpu = session->resource(device), cpu = session->resource(DeviceId::cpu());
    constexpr std::size_t GPUBytes = 6 * sizeof(std::uint64_t);
    constexpr std::size_t MappedRoots = 5;
    const auto definitionBytes = Access::NativeJoinDefinitions::Count * nativeGraphCaptureObjectBytes(device);
    PhysicalMemoryPlanBuilder bom(RetainedParticipantGraphExecutable::generalBOM(cpu,gpu,1));
    bom.add(gpu,PhysicalMemoryOwner::ActivationArena,GPUBytes);
    // Exact known graph-only wrapper objects are distinct from opaque SDK backing.
    bom.add(cpu,PhysicalMemoryOwner::NativeGraphExecutable,definitionBytes);
    bom.add(cpu,PhysicalMemoryOwner::ImmutableLookupStaging,
        MappedRoots * TransferEngine::mappedHostRegionAllocationBytes(sizeof(std::uint64_t)));
    session->admit(bom.build());
    {
        auto gpuClaim = session->memory()->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,GPUBytes);
        auto definitionClaim = session->memory()->claimNewAllocation(
            DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable,definitionBytes);
        TransferEngine transfers;
        auto bytes = transfers.allocateDeviceTransferBuffer(GPUBytes,device);
        const auto mapped = [&] { return qwen4exp::PooledImmutableLocalHotReadyRoot::create({0,device},session->memory()); };
        auto historyGate = mapped(), retireGate = mapped(), historyDone = mapped(), retireDone = mapped(), futureDone = mapped();
        const auto stream = ExplicitGPUStream(worker->defaultStream()).get();
        auto parent = Access::original(original,stream);
        Access::NativeJoinDefinitions definitions;
        bool drained = false;
        const auto drain = [&] {
            hostWord(historyGate).store(1,std::memory_order_release);
            hostWord(retireGate).store(1,std::memory_order_release);
            worker->submitAndWait([&] {
                requireFork(backend->synchronizeStream(stream,device.ordinal),"Native join diagnostic cleanup failed");
            });
        };
        struct Cleanup {
            std::function<void()> drain;
            bool& drained;
            /** @brief Release original diagnostic gates before any exceptional native teardown. */
            ~Cleanup() { if(!drained)try{drain();}catch(...){std::terminate();} }
        } cleanup{drain,drained};
        worker->submitAndWait([&] {
            const auto copy = [&](std::size_t destination,std::size_t source) {
                requireFork(backend->deviceCopyAsync(bytes->mutableDeviceData(destination),bytes->deviceData(source),
                    sizeof(std::uint64_t),device.ordinal,stream),"Native join exact marker copy failed");
            };
            definitions = Access::composeNativeJoinFixture(parent,owned,missingRetire,
                [&](IGPUGraphCapture& probe,IGPUGraphCapture& history,IGPUGraphCapture& retired,IGPUGraphCapture& future) {
                    {
                        ScopedBackendGraphCapture capture(*worker,probe,"original join Probe marker");
                        requireFork(capture.begin(),"Native join Probe capture failed");
                        copy(8,0);
                        capture.finish();
                    }
                    {
                        ScopedBackendGraphCapture capture(*worker,history,"original join main history marker");
                        requireFork(capture.begin(),"Native join history capture failed");
                        transfers.enqueueMappedTimelineWait64(*historyGate->mappedRegion(),0,1,device,stream);
                        copy(16,0);
                        transfers.enqueueMappedTimelinePublish64(*historyDone->mappedRegion(),0,1,device,stream);
                        capture.finish();
                    }
                    {
                        ScopedBackendGraphCapture capture(*worker,retired,"original join finite Retire marker");
                        requireFork(capture.begin(),"Native join prior finite capture failed");
                        transfers.enqueueMappedTimelineWait64(*retireGate->mappedRegion(),0,1,device,stream);
                        copy(24,8);
                        transfers.enqueueMappedTimelinePublish64(*retireDone->mappedRegion(),0,1,device,stream);
                        capture.finish();
                    }
                    {
                        ScopedBackendGraphCapture capture(*worker,future,"original joined next finite body");
                        requireFork(capture.begin(),"Native joined finite capture failed");
                        copy(32,16);
                        copy(40,24);
                        // Only the no-edge control releases the held prior body,
                        // after taking its stale snapshot. Positive graph uses
                        // the real prior-child exit and never executes this arm.
                        if(missingRetire)
                            transfers.enqueueMappedTimelinePublish64(*retireGate->mappedRegion(),0,1,device,stream);
                        transfers.enqueueMappedTimelinePublish64(*futureDone->mappedRegion(),0,1,device,stream);
                        capture.finish();
                    }
                });
            auto& recording = Access::capture(parent);
            requireFork(recording.prepareRuntimeContextStorage(session->memory()),"Native join original context preparation failed");
            requireFork(recording.instantiate(),"Native joined parent instantiation failed");
            requireFork(Access::rejectInstantiatedNativeJoin(parent,definitions),"Instantiated native parent accepted a finite join");
            for(const auto* graph : {definitions.probe.get(),definitions.history.get(),definitions.retired.get(),definitions.future.get()})
                requireFork(!graph->hasExecutable(),"Joined graph-only definition acquired an executable");
        });
        drain();
        const std::array<std::uint64_t,6> resetZeros{}; // Retained until every original async input copy retired.
        for(std::uint64_t invocation=1;invocation<=20;++invocation)
        {
            const bool reset = invocation==1 || invocation==11;
            hostWord(historyGate).store(0,std::memory_order_release);
            hostWord(retireGate).store(0,std::memory_order_release);
            hostWord(historyDone).store(0,std::memory_order_release);
            hostWord(retireDone).store(0,std::memory_order_release);
            hostWord(futureDone).store(0,std::memory_order_release);
            worker->submitAndWait([&] {
                if(reset) {
                    requireFork(backend->hostToDeviceOnStream(bytes->mutableDeviceData(),resetZeros.data(),sizeof(resetZeros),device.ordinal,stream),
                        "Native join original data reset failed");
                }
                requireFork(backend->hostToDeviceOnStream(bytes->mutableDeviceData(),&invocation,sizeof(invocation),device.ordinal,stream),
                    "Native join recorded input publication failed");
                requireFork(Access::submit(parent,invocation),"Native joined original parent submit failed");
            });
            const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(30);
            if(missingRetire) {
                hostWord(historyGate).store(1,std::memory_order_release);
            } else {
                const bool historyFirst = invocation%2==1;
                auto& gate = historyFirst?historyGate:retireGate;
                auto& done = historyFirst?historyDone:retireDone;
                hostWord(gate).store(1,std::memory_order_release);
                while(hostWord(done).load(std::memory_order_acquire)!=1) {
                    requireFork(std::chrono::steady_clock::now()<deadline,"Native first dependency marker exceeded standard deadline");
                    std::this_thread::yield();
                }
                EXPECT_FALSE(Access::observeTerminal(parent));
                hostWord(historyFirst?retireGate:historyGate).store(1,std::memory_order_release);
            }
            while(!Access::observeTerminal(parent)) {
                requireFork(std::chrono::steady_clock::now()<deadline,"Native joined parent exceeded standard terminal deadline");
                std::this_thread::yield();
            }
            std::array<std::uint64_t,6> observed{};
            worker->submitAndWait([&] {
                requireFork(backend->deviceToHostOnStream(observed.data(),bytes->deviceData(),sizeof(observed),device.ordinal,stream),
                    "Native joined terminal observation failed");
            });
            drain();
            for(std::size_t i=0;i<5;++i)EXPECT_EQ(observed[i],invocation)<<i;
            EXPECT_EQ(observed[5],missingRetire?(reset?0:invocation-1):invocation);
            EXPECT_EQ(hostWord(futureDone).load(std::memory_order_acquire),1u);
        }
        parent.reset();
        definitions = {};
        drain();
        drained = true;
        std::cout<<"NATIVE_FINITE_CHILD_JOIN device="<<device.toString()<<" owned="<<owned
                 <<" missing_retire_control="<<missingRetire<<" replays=20 data_reset_epochs=2 executables=1 model_executed=0\n";
    }
    session->requirePayloadRetired();
    worker.reset();session->retire();session->requireExactRetired();
}
}
#define NATIVE_FINITE_JOIN_TESTS(BACKEND,DEVICE) \
TEST(LocalHotNativeParentFork,BACKEND##_OwnedFiniteJoin){nativeFiniteJoin(DEVICE,true,false);} \
TEST(LocalHotNativeParentFork,BACKEND##_ImportedFiniteJoin){nativeFiniteJoin(DEVICE,false,false);} \
TEST(LocalHotNativeParentFork,BACKEND##_MissingFiniteJoinControl){nativeFiniteJoin(DEVICE,true,true);}
#ifdef HAVE_CUDA
NATIVE_FINITE_JOIN_TESTS(CUDA,DeviceId::cuda(0))
#endif
#ifdef HAVE_ROCM
NATIVE_FINITE_JOIN_TESTS(ROCm,DeviceId::rocm(0))
#endif
#undef NATIVE_FINITE_JOIN_TESTS
} // namespace llaminar2::test
