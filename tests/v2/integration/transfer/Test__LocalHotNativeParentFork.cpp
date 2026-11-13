/**
 * @file Test__LocalHotNativeParentFork.cpp
 * @brief Actual native ordered main units and post-Probe auxiliary fork/retirement replay.
 *
 * These model-free diagnostics exercise the production graph SDK lowerers and
 * original General/PMA lifecycle. Input generations and test gates are authored
 * diagnostic inputs. No model SourceSeal, Rank cohort or successful L2 receipt
 * is invented, and this does not qualify cache mechanics or model inference.
 */
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotReadyRoot.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/ExplicitGPUStream.h"
#include "transfer/TransferEngine.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "execution/moe/MoEOverlayRetainedParentComposer.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "utils/CPUJointSourceTestPlan.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <thread>

namespace llaminar2::test {
namespace {
/** @brief Actual profile boundaries; noEdge is confined to its explicit failing-control case. */
enum class ForkProfile { AllHit,Miss,Empty,OldNoEdge };
/** @brief Fail the requested native backend without a skip or alternate path. */
void requireFork(bool ok,const char* reason)
{
    if(!ok)throw std::runtime_error(reason);
}
/** @brief Diagnostic-only mapped input/observation, never a host shadow of inference state. */
std::atomic_ref<std::uint64_t> hostWord(const std::shared_ptr<qwen4exp::PooledImmutableLocalHotReadyRoot>& root)
{
    return std::atomic_ref(*static_cast<std::uint64_t*>(root->mappedRegion()->mutableHostData()));
}
/** @brief Declaration of the diagnostic copies recorded below; never a serving/source stage. */
class NativeForkDeclarationStage final : public IComputeStage {
public:
    /** @brief Bind the exact native diagnostic participant. */
    explicit NativeForkDeclarationStage(DeviceId device):IComputeStage(device){}
    /** @brief Planning only; invoking this stage would replace the actual captured SDK body. */
    bool execute(IDeviceContext*) override { throw std::logic_error("Diagnostic declaration cannot execute"); }
    /** @copydoc IComputeStage::type */
    ComputeStageType type() const override { return ComputeStageType::COPY; }
    /** @copydoc IComputeStage::name */
    std::string name() const override { return "native_fork_declaration"; }
    /** @copydoc IComputeStage::supportsBackend */
    bool supportsBackend(ComputeBackendType) const override { return true; }
    /** @copydoc IComputeStage::coherencePolicy */
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    /** @copydoc IComputeStage::isGraphCapturable */
    bool isGraphCapturable() const override { return true; }
protected:
    /** @copydoc IComputeStage::buildDumpInfoImpl */
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }
};

/** @brief Build actual resolver-authored mixed-GPU metadata for the recorded diagnostic.
 * @param device Exact participant owning the native copies; no foreign device is opened.
 * @return Complete metadata without a QFN native source/cohort permit.
 */
ComputeGraph nativeForkDeclaration(DeviceId device)
{
    auto authored = std::make_shared<MoERoutedExpertPlacementPlan>(cpuJointSourceTestPlan());
    auto& follower = authored->domains[1];
    follower.name="rocm_followers";follower.scope=ExecutionDomainScope::RANK_LOCAL;
    follower.backend=CollectiveBackendType::RCCL;follower.owner_rank=0;
    follower.participants={GlobalDeviceAddress::rocm(0),GlobalDeviceAddress::rocm(1)};
    follower.world_ranks.clear();authored->routed_tiers[1].domain=follower.name;
    authored->routed_tiers[1].fallback=false;
    const auto resolved=resolveMoEExpertOverlayRuntimePlan(authored,{.current_world_rank=0});
    const auto execution=std::make_shared<const MoEExpertOverlayExecutionPlan>(buildMoEExpertOverlayExecutionPlan(*resolved,1));
    static const int diagnostic_domain=0;
    ComputeGraph graph;
    graph.setRetainedParticipantDefinitionDomain({device.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0),&diagnostic_domain});
    for(const auto* name:{"probe","aux_result","main_read","aux_retire"})
        graph.addNode(name,std::make_unique<NativeForkDeclarationStage>(device),device);
    graph.addDependency("aux_result","probe");graph.addDependency("main_read","probe");
    graph.addDependency("aux_retire","aux_result");graph.addDependency("aux_retire","main_read");
    graph.setTerminalNode("main_read");
    graph.setRetainedParticipantAuxiliaryRange({.role=RetainedGraphParticipantAuxiliaryRole::Reader,
        .participant=device,.nodes={"aux_result","aux_retire"},.probe_ready="probe",.ready_publication="aux_result",
        .main_read="main_read",.retirement="aux_retire",.identity=781,.original_execution=execution});
    graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
    graph.setNativeTransactionCaptureContract({.identity="diagnostic_original_native_parent",.terminal_node="main_read"});
    return graph;
}
/** @brief Replay the genuine parent with distinct Probe/LocalRead units and a finite auxiliary.
 * @param device Actual prepared native participant.
 * @param owned Retain original native-owned or imported graph-only definitions.
 * @param profile Existing bounded diagnostic input/gate configuration.
 * @param originalComposer Derive/install the actual authenticated native-parent compiler plan.
 */
void nativeFork(DeviceId device,bool owned,ForkProfile profile,bool originalComposer = false)
{
    const std::array devices{device};
    auto session=PreparedNativeLookupFixture::prepare(devices);
    const auto& original=session->permit(device);
    auto worker=original.worker();
    auto* backend=getBackendFor(device);
    const auto gpu=session->resource(device),cpu=session->resource(DeviceId::cpu());
    PhysicalMemoryPlanBuilder bom(RetainedParticipantGraphExecutable::generalBOM(cpu,gpu,1));
    constexpr std::size_t GPUBytes=4*sizeof(std::uint64_t);
    bom.add(gpu,PhysicalMemoryOwner::ActivationArena,GPUBytes);
    // ready, producer input gate, service gate and diagnostic read-completion
    // observation are four independent actual mapped owners, priced once each.
    bom.add(cpu,PhysicalMemoryOwner::ImmutableLookupStaging,
        4*TransferEngine::mappedHostRegionAllocationBytes(sizeof(std::uint64_t)));
    session->admit(bom.build());
    {
        auto gpuClaim=session->memory()->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,GPUBytes);
        TransferEngine transfers;
        auto bytes=transfers.allocateDeviceTransferBuffer(GPUBytes,device);
        auto ready=qwen4exp::PooledImmutableLocalHotReadyRoot::create({0,device},session->memory());
        auto producerGate=qwen4exp::PooledImmutableLocalHotReadyRoot::create({0,device},session->memory());
        auto serviceGate=qwen4exp::PooledImmutableLocalHotReadyRoot::create({0,device},session->memory());
        auto readObservation=qwen4exp::PooledImmutableLocalHotReadyRoot::create({0,device},session->memory());
        const auto stream=ExplicitGPUStream(worker->defaultStream()).get();
        auto parent=RetainedParticipantGraphExecutableFixtureAccess::original(original,stream);
        RetainedParticipantGraphExecutableFixtureAccess::NativeForkDefinitions definitions;
        bool drained=false;
        const auto join=[&] {
            hostWord(producerGate).store(1,std::memory_order_release);
            hostWord(serviceGate).store(1,std::memory_order_release);
            worker->submitAndWait([&] {
                requireFork(backend->synchronizeStream(stream,device.ordinal),"Native fork diagnostic cleanup failed");
            });
        };
        struct Cleanup {
            std::function<void()> join;
            bool& drained;
            /** @brief Release both deliberate gates before diagnostic teardown on every exception. */
            ~Cleanup(){if(!drained)try{join();}catch(...){std::terminate();}}
        } cleanup{join,drained};
        const std::array<std::uint64_t,4> initial{};
        worker->submitAndWait([&] {
            requireFork(backend->hostToDeviceOnStream(bytes->mutableDeviceData(),initial.data(),sizeof(initial),device.ordinal,stream),
                        "Native fork original input setup failed");
            auto readyAlias=transfers.bindMappedTimelineKernelPublish64(*ready->mappedRegion(),0,1,device).deviceSignal();
            definitions=RetainedParticipantGraphExecutableFixtureAccess::composeNativeForkFixture(parent,owned,
                profile==ForkProfile::OldNoEdge,[&](IGPUGraphCapture& probe,IGPUGraphCapture& read,IGPUGraphCapture& aux) {
                {
                    ScopedBackendGraphCapture capture(*worker,probe,"native original Probe unit");
                    requireFork(capture.begin(),"Native Probe recording failed");
                    transfers.enqueueMappedTimelineWait64(*producerGate->mappedRegion(),0,1,device,stream);
                    requireFork(backend->memset(readyAlias,0,sizeof(std::uint64_t),device.ordinal,stream),"Native Probe ready reset failed");
                    requireFork(backend->deviceCopyAsync(bytes->mutableDeviceData(8),bytes->deviceData(0),8,device.ordinal,stream),
                                "Native Probe generation publication failed");
                    if(profile!=ForkProfile::Miss)
                        transfers.enqueueMappedTimelinePublish64(*ready->mappedRegion(),0,1,device,stream);
                    capture.finish();
                }
                {
                    ScopedBackendGraphCapture capture(*worker,read,"native later LocalRead unit");
                    requireFork(capture.begin(),"Native LocalRead recording failed");
                    transfers.enqueueMappedTimelineWait64(*ready->mappedRegion(),0,1,device,stream);
                    requireFork(backend->deviceCopyAsync(bytes->mutableDeviceData(16),bytes->deviceData(8),8,device.ordinal,stream),
                                "Native LocalRead observation failed");
                    transfers.enqueueMappedTimelinePublish64(*ready->mappedRegion(),0,2,device,stream);
                    transfers.enqueueMappedTimelinePublish64(*readObservation->mappedRegion(),0,1,device,stream);
                    capture.finish();
                }
                {
                    ScopedBackendGraphCapture capture(*worker,aux,"native finite compact-response and maintenance unit");
                    requireFork(capture.begin(),"Native auxiliary recording failed");
                    requireFork(backend->deviceCopyAsync(bytes->mutableDeviceData(24),bytes->deviceData(8),8,device.ordinal,stream),
                                "Native auxiliary current-Probe observation failed");
                    // In the no-edge control this releases the blocked Probe
                    // only AFTER copying the previous generation, making that
                    // original ordering defect deterministic and finite.
                    transfers.enqueueMappedTimelinePublish64(*producerGate->mappedRegion(),0,1,device,stream);
                    if(profile==ForkProfile::Miss)
                        transfers.enqueueMappedTimelinePublish64(*ready->mappedRegion(),0,1,device,stream);
                    transfers.enqueueMappedTimelineWait64(*ready->mappedRegion(),0,2,device,stream);
                    transfers.enqueueMappedTimelineWait64(*serviceGate->mappedRegion(),0,1,device,stream);
                    capture.finish();
                }
            }, originalComposer ? std::function<bool(IGPUGraphCapture&,const IGPUGraphCapture&,const IGPUGraphCapture&)>{
                [&](IGPUGraphCapture& destination,const IGPUGraphCapture& probe,const IGPUGraphCapture& read) {
                    auto declaration=nativeForkDeclaration(device);
                    DeviceGraphExecutor::DecodeCapturePolicy policy;
                    policy.allow_cached_graph_replay=true;
                    requireFork(DeviceGraphCaptureController::constrainReplayPolicyToNativeEnvelope(
                        declaration.nativeCaptureEnvelope(),policy),"Original native envelope policy failed");
                    requireFork(policy.graph_replay_plan_policy==DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph &&
                        !policy.retained_parent_composer,"Native envelope retained an unauthenticated composer");
                    auto plan=makeMoEOverlayRetainedParentPlan(declaration);
                    requireFork(plan.has_value(),"Original native auxiliary composer was not selected");
                    policy.graph_replay_plan_policy=plan->replay_policy;
                    policy.retained_parent_composer=std::move(plan->composer);
                    policy.defer_final_sync=true;
                    DeviceGraphExecutor::GraphSegmentCache cache;
                    DeviceGraphCaptureController::buildCapturePlan(declaration,cache,nullptr,false,false,policy.graph_replay_plan_policy);
                    requireFork(cache.segments.size()==2,"Original native auxiliary compiler did not expose the actual Probe frontier");
                    const std::array units{
                        DeviceGraphExecutor::GraphSegmentCache::RetainedCaptureUnitTemplateView{
                            .capture=&probe,.stream=stream,.stage_names=cache.segments[0].stage_names},
                        DeviceGraphExecutor::GraphSegmentCache::RetainedCaptureUnitTemplateView{
                            .capture=&read,.stream=stream,.stage_names=cache.segments[1].stage_names}};
                    return policy.retained_parent_composer(destination,declaration,units);
                }} : std::function<bool(IGPUGraphCapture&,const IGPUGraphCapture&,const IGPUGraphCapture&)>{});
            auto& recording=RetainedParticipantGraphExecutableFixtureAccess::capture(parent);
            requireFork(recording.prepareRuntimeContextStorage(session->memory()),"Native fork original context preparation failed");
            requireFork(recording.instantiate(),"Native fork parent instantiation failed");
            requireFork(!definitions.producer->hasExecutable() && !definitions.consumer->hasExecutable() && !definitions.auxiliary->hasExecutable(),
                        "A graph-only native definition acquired another executable");
        });
        join();
        for(std::uint64_t invocation=1;invocation<=20;++invocation) {
            // Deliberately start each replay at the previous read-complete2.
            // Only the CURRENT Probe dependency authorizes auxiliary entry.
            hostWord(ready).store(2,std::memory_order_release);
            hostWord(producerGate).store(profile==ForkProfile::OldNoEdge?0:1,std::memory_order_release);
            hostWord(serviceGate).store(profile==ForkProfile::AllHit?0:1,std::memory_order_release);
            hostWord(readObservation).store(0,std::memory_order_release);
            worker->submitAndWait([&] {
                requireFork(backend->hostToDeviceOnStream(bytes->mutableDeviceData(0),&invocation,8,device.ordinal,stream),
                            "Native fork recorded input publication failed");
                requireFork(RetainedParticipantGraphExecutableFixtureAccess::submit(parent,invocation),"Native fork original parent submit failed");
            });
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            if(profile==ForkProfile::AllHit) {
                while(hostWord(readObservation).load(std::memory_order_acquire)!=1) {
                    requireFork(std::chrono::steady_clock::now()<deadline,"All-hit main read waited for held owner service");
                    std::this_thread::yield();
                }
                EXPECT_FALSE(RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(parent));
                hostWord(serviceGate).store(1,std::memory_order_release);
            }
            while(!RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(parent)) {
                requireFork(std::chrono::steady_clock::now()<deadline,"Native fork exact parent terminal exceeded standard deadline");
                std::this_thread::yield();
            }
            std::array<std::uint64_t,4> observed{};
            worker->submitAndWait([&] {
                requireFork(backend->deviceToHostOnStream(observed.data(),bytes->deviceData(),sizeof(observed),device.ordinal,stream),
                            "Native fork terminal observation failed");
            });
            join();
            EXPECT_EQ(observed[0],invocation);EXPECT_EQ(observed[1],invocation);EXPECT_EQ(observed[2],invocation);
            if(profile==ForkProfile::OldNoEdge)EXPECT_EQ(observed[3],invocation-1);
            else EXPECT_EQ(observed[3],invocation);
            EXPECT_EQ(hostWord(ready).load(std::memory_order_acquire),2u);
        }
        parent.reset();definitions={};join();drained=true;
        std::cout<<"LOCAL_HOT_NATIVE_PARENT_FORK device="<<device.toString()<<" owned="<<owned
                 <<" profile="<<static_cast<unsigned>(profile)<<" replays=20 aux_executables=0 model_executed=0\n";
    }
    session->requirePayloadRetired();worker.reset();session->retire();session->requireExactRetired();
}
}
#define LOCAL_HOT_FORK_TESTS(BACKEND,DEVICE) \
TEST(LocalHotNativeParentFork,BACKEND##_OwnedAllHit){nativeFork(DEVICE,true,ForkProfile::AllHit);} \
TEST(LocalHotNativeParentFork,BACKEND##_ImportedAllHit){nativeFork(DEVICE,false,ForkProfile::AllHit);} \
TEST(LocalHotNativeParentFork,BACKEND##_OwnedMiss){nativeFork(DEVICE,true,ForkProfile::Miss);} \
TEST(LocalHotNativeParentFork,BACKEND##_ImportedEmpty){nativeFork(DEVICE,false,ForkProfile::Empty);} \
TEST(LocalHotNativeParentFork,BACKEND##_OldNoEdgeControl){nativeFork(DEVICE,true,ForkProfile::OldNoEdge);} \
TEST(LocalHotNativeParentFork,BACKEND##_OriginalNativeComposerAllHit){nativeFork(DEVICE,true,ForkProfile::AllHit,true);}
#ifdef HAVE_CUDA
LOCAL_HOT_FORK_TESTS(CUDA,DeviceId::cuda(0))
#endif
#ifdef HAVE_ROCM
LOCAL_HOT_FORK_TESTS(ROCm,DeviceId::rocm(0))
#endif
#undef LOCAL_HOT_FORK_TESTS
} // namespace llaminar2::test

#include "LocalHotFiniteChildJoin.inc"
#include "LocalHotOriginalSourceFork.inc"

#include "LocalHotFutureSourceReplay.inc"
