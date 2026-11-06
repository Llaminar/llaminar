/**
 * @file MoEFinitePlacementCaptureReplay.inc
 * @brief Recorded finite-parent inputs through original CUDA/HIP captured kernels.
 *
 * Each endpoint is a real physical GPU and owns its original PMA-backed epoch
 * arena. The fixed trace publishes one successor between two actual children,
 * checks old-E execution, closes the parent, and reclaims the original bank.
 * Twenty iterations reuse the same five native graphs without recapture.
 * This is kernel/protocol evidence, not a model-source or communicator proof.
 */

namespace llaminar2::test
{
namespace
{
    /** One real GPU's original control records and exact captured terminal. */
    struct FinitePlacementCaptureEndpoint
    {
        IBackend *backend;
        DeviceId device;
        std::unique_ptr<DeviceMoEOverlayEpochArena> arena;
        std::unique_ptr<ScopedGPUStream> stream;
        std::shared_ptr<void> terminal;
        std::unique_ptr<IMoEKernel> kernel;
        std::array<std::unique_ptr<IGPUGraphCapture>,5> graphs;
    };

    /**
     * @brief Replay the same immutable lifecycle trace on every actual device.
     * @param vendor CUDA or ROCm; no ordinal is used as a substitute owner.
     * @param count Actual physical endpoint count.
     */
    void replayFinitePlacementCapture(DeviceType vendor,int count)
    {
        IBackend *backend=vendor==DeviceType::CUDA?getCUDABackend():getROCmBackend();
        if(!backend || backend->deviceCount()<count)
            GTEST_SKIP()<<"Requires "<<count<<" actual "<<(vendor==DeviceType::CUDA?"CUDA":"ROCm")<<" devices";
        std::vector<std::unique_ptr<FinitePlacementCaptureEndpoint>> owners;
        for(int ordinal=0;ordinal<count;++ordinal) {
            auto owner=std::make_unique<FinitePlacementCaptureEndpoint>();
            owner->backend=backend;
            owner->device=vendor==DeviceType::CUDA?DeviceId::cuda(ordinal):DeviceId::rocm(ordinal);
            owner->arena=std::make_unique<DeviceMoEOverlayEpochArena>(
                admitEpochArenaForTest({.device_id=owner->device}));
            owner->stream=std::make_unique<ScopedGPUStream>(owner->device);
            owner->terminal=std::shared_ptr<void>(backend->createEvent(ordinal),[backend,ordinal](void *event) {
                if(event)backend->destroyEvent(event,ordinal);
            });
            owner->kernel=llaminar::v2::kernels::KernelFactory::createMoEKernel(owner->device);
            ASSERT_TRUE(owner->kernel && owner->stream->get() && owner->terminal);
            auto &worker=GPUDeviceContextPool::instance().getContext(owner->device);
            bool captured=true;
            worker.submitAndWait([&] {
                for(std::size_t graph_index=0u;graph_index<owner->graphs.size();++graph_index) {
                    auto &graph=owner->graphs[graph_index];
                    graph=worker.createGraphCapture(owner->stream->get());
                    if(!graph || !graph->beginCapture()) {captured=false;break;}
                    const MoEKernelLaunchContext launch{.stream=owner->stream->get()};
                    auto *control=owner->arena->control();
                    auto *status=owner->arena->maintenanceStatus();
                    auto *epoch=owner->arena->maintenanceEpoch();
                    bool accepted=false;
                    switch(graph_index) {
                    case 0u: accepted=owner->kernel->runMoEOverlayEpochParentAction(launch,control,
                        owner->arena->finiteParentStatus(),DeviceMoEOverlayEpochParentAction::Arm);break;
                    case 1u: accepted=owner->kernel->acquireMoEOverlayEpoch(launch,control,
                        owner->arena->requestTicket(0u),owner->arena->requestStatus(0u)) &&
                        owner->kernel->releaseMoEOverlayEpoch(launch,control,
                        owner->arena->requestTicket(0u),owner->arena->requestStatus(0u));break;
                    case 2u: accepted=owner->kernel->reserveMoEOverlayEpochCandidate(launch,control,epoch,status) &&
                        owner->kernel->markMoEOverlayEpochCandidateReady(launch,control,epoch,status) &&
                        owner->kernel->publishMoEOverlayEpochCandidate(launch,control,epoch,status);break;
                    case 3u: accepted=owner->kernel->runMoEOverlayEpochParentAction(launch,control,
                        owner->arena->finiteParentStatus(),DeviceMoEOverlayEpochParentAction::Close);break;
                    case 4u: accepted=owner->kernel->retireMoEOverlayEpoch(launch,control,epoch,status);break;
                    }
                    const bool ended=graph->endCapture();
                    captured=captured && accepted && ended && graph->instantiate();
                    if(!captured)break;
                }
            });
            ASSERT_TRUE(captured);
            owners.push_back(std::move(owner));
        }
        const auto submit_all=[&](std::size_t graph_index) {
            bool accepted=true;
            std::vector<bool> submitted(owners.size(),false);
            for(std::size_t i=0u;i<owners.size();++i) {
                auto &owner=*owners[i];
                auto &worker=GPUDeviceContextPool::instance().getContext(owner.device);
                worker.submitAndWait([&] {
                    submitted[i]=owner.graphs[graph_index]->launchOnStream(owner.stream->get()) &&
                        backend->recordEvent(owner.terminal.get(),owner.device.gpu_ordinal(),owner.stream->get());
                });
                accepted=accepted && submitted[i];
            }
            // Observe EVERY accepted native terminal before any assertion can
            // destroy an original arena. SDK callback completion is insufficient.
            for(std::size_t i=0u;i<owners.size();++i)
                if(submitted[i])accepted=await(backend,owners[i]->terminal.get(),
                    owners[i]->device.gpu_ordinal(),std::chrono::seconds(30)) && accepted;
            if(!accepted) {
                LOG_ERROR("Finite placement capture lost an original native terminal; retaining owners until process exit");
                std::terminate();
            }
            return accepted;
        };
        std::array<std::uint64_t,21> epoch_trace{};
        for(std::size_t i=0u;i<epoch_trace.size();++i)epoch_trace[i]=i+1u;
        const auto upload_epoch=[&](const std::uint64_t &input_epoch) {
            bool accepted=true;
            for(auto &owner:owners)
                accepted=backend->hostToDevice(owner->arena->maintenanceEpoch(),&input_epoch,
                    sizeof(input_epoch),owner->device.gpu_ordinal(),owner->stream->get()) && accepted;
            return accepted;
        };
        for(std::uint64_t replay=0u;replay<20u;++replay) {
            SCOPED_TRACE(replay);
            const auto base_epoch=replay+1u;
            ASSERT_TRUE(submit_all(0u));ASSERT_TRUE(submit_all(1u));
            ASSERT_TRUE(upload_epoch(epoch_trace[replay+1u]));ASSERT_TRUE(submit_all(2u));
            ASSERT_TRUE(submit_all(1u));
            for(auto &owner:owners) {
                DeviceMoEOverlayEpochStatus result{};
                ASSERT_TRUE(backend->deviceToHost(&result,owner->arena->requestStatus(0u),sizeof(result),
                    owner->device.gpu_ordinal(),owner->stream->get()));
                EXPECT_TRUE(result.succeeded());EXPECT_EQ(result.epoch,base_epoch);
                EXPECT_EQ(result.operation,static_cast<std::uint32_t>(DeviceMoEOverlayEpochOperation::Release));
            }
            ASSERT_TRUE(submit_all(3u));
            ASSERT_TRUE(upload_epoch(epoch_trace[replay]));ASSERT_TRUE(submit_all(4u));
            for(auto &owner:owners) {
                DeviceMoEOverlayEpochControl control{};DeviceMoEOverlayEpochStatus result{};
                ASSERT_TRUE(backend->deviceToHost(&control,owner->arena->control(),sizeof(control),
                    owner->device.gpu_ordinal(),owner->stream->get()));
                ASSERT_TRUE(backend->deviceToHost(&result,owner->arena->maintenanceStatus(),sizeof(result),
                    owner->device.gpu_ordinal(),owner->stream->get()));
                EXPECT_TRUE(result.succeeded());EXPECT_EQ(result.epoch,base_epoch);
                EXPECT_EQ(control.bank_readers[0],0u);EXPECT_EQ(control.bank_readers[1],0u);
                EXPECT_EQ(control.acquisitions_in_flight,0u);
                EXPECT_EQ(control.bank_epochs[deviceMoEOverlayEpochSelectorBank(control.published_selector)],base_epoch+1u);
            }
        }
    }
}

/** @test Two original CUDA owners replay scoped children across publication/reset. */
TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm, FinitePlacementCUDA2CapturedProtocolReplay)
{ replayFinitePlacementCapture(DeviceType::CUDA,2); }
/** @test ROCm uses the same original two-owner device lifecycle and exact extents. */
TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm, FinitePlacementROCm2CapturedProtocolReplay)
{ replayFinitePlacementCapture(DeviceType::ROCm,2); }
/** @test All four real ROCm owners close before original old-bank reclamation. */
TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm, FinitePlacementROCm4CapturedProtocolReplay)
{ replayFinitePlacementCapture(DeviceType::ROCm,4); }
}
