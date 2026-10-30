/**
 * @file Test__ResidentLookupPrefillLifecycle.cpp
 * @brief Actual DGO reset/prefill/admission ownership for resident and hosted lookup lifetimes.
 *
 * A tiny real DGO arena owns the only controller row and its existing reset,
 * forward-publication and response-admission events. Real captured resident
 * kernels read/write that row, never a host-seeded health mirror. The test
 * peer publishes the genuine lookup graph's completion as the model-free
 * forward producer; full-model arithmetic is deliberately outside this proof.
 * Hosted cases additionally exercise PMA-owned RAM/mmap native I/O, the original
 * interval guard and exact reset terminal event before the next API epoch.
 * Two-device cases deliberately change the runtime's current device before
 * every public reset without changing the actual DGO stream/root ownership.
 */
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/qwen4exp/ImmutableResidentLookupKernels.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "models/qwen/QwenStandardGraph.h"
#include "tensors/TensorFactory.h"
#include "transfer/TransferEngine.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "loaders/HostImmutableRowProvider.h"
#include "utils/PerfStatsCollector.h"
#include "../../../mocks/MockMPIContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
/** @brief Existing DGO test peer, defined only in this native executable's closure. */
struct DeviceGraphOrchestratorLiveStateTestAccess
{
    /** @brief Resolve the exact native DGO stream for diagnostic terminal retirement.
     * @param runner Actual participant owning admission/reset submission.
     * @return Explicit stream, without any publication or data operation.
     */
    static void *controllerStream(DeviceGraphOrchestrator &runner)
    {return runner.explicitGPUStreamForOperation("resident_prefill_fixture_retirement");}
    /** @brief Borrow the actual arena controller, without copying or modifying its words.
     * @param runner Actual participant retaining the canonical arena.
     * @return Sole native controller allocation, borrowed through test retirement.
     */
    static int *control(DeviceGraphOrchestrator &runner)
    {return runner.device_generation_storage_.control_device;}
    /** @brief Bind actual named workspace so native admission has a real generation.
     * @param runner Participant retaining the normal workspace allocator.
     * @param stage Setup-only consumer retaining its real workspace binding.
     * @return True only after canonical family allocation publishes a nonzero generation.
     */
    static bool workspace(DeviceGraphOrchestrator &runner,IComputeStage &stage)
    {
        auto *consumer=dynamic_cast<IWorkspaceConsumer *>(&stage);
        if(!consumer)return false;
        runner.workspace_allocator_=std::make_shared<WorkspaceAllocator>();
        ComputeGraph graph;WorkspaceSizingHints hints;hints.max_seq_len=1;
        return runner.workspace_allocator_->allocateForGraphFamily(graph,
            WorkspaceGraphParticipantRole::Prefill,{},hints,
            {{.consumer=consumer,.device=stage.device(),.m=1,.n=1,.k=1}},
            {.gpu_fraction=1.0f,.cpu_fraction=1.0f,.min_budget=256,.max_budget=256}) &&
            runner.workspace_allocator_->deviceGeneration(stage.device())!=0;
    }
    /** @brief Enter the real forward prelude on the captured lookup consumer stream.
     * @param runner Canonical request lifecycle owner.
     * @param stream Exact captured graph consumer stream.
     * @param device Original participant identity.
     * @param rows Declared prefill graph capacity; live rows remain device-owned.
     * @return Whether all original producer/reset handoffs were acquired.
     */
    static bool prepare(DeviceGraphOrchestrator &runner,void *stream,DeviceId device,int rows)
    {
        ForwardInput input;input.seq_len=rows;input.batch_size=1;
        input.execution_role=ForwardExecutionRole::MainInference;
        input.execution_phase=ForwardExecutionPhase::Prefill;
        return runner.prepareLiveStateForForwardGraphExecution(input,stream,device);
    }
    /** @brief Exercise scalar decode's actual prelude without replacing its device count.
     * @param runner Canonical participant owning the request epoch.
     * @param stream Exact lookup graph stream.
     * @param device Original GPU owner.
     * @return Whether the production scalar forward prelude acquired every producer.
     */
    static bool prepareDecode(DeviceGraphOrchestrator &runner,void *stream,DeviceId device)
    {
        ForwardInput input;input.seq_len=1;input.batch_size=1;
        input.execution_role=ForwardExecutionRole::MainInference;
        input.execution_phase=ForwardExecutionPhase::Decode;
        return runner.prepareLiveStateForForwardGraphExecution(input,stream,device);
    }
    /** @brief Borrow the real API epoch solely to test rejection of replacement before retirement. */
    static std::uint64_t epoch(const DeviceGraphOrchestrator &runner){return runner.session_epoch_;}
    /** @brief Complete the ordinary native-terminal sink after an actual lookup graph.
     * @param runner Canonical participant retaining its setup-created terminal event.
     * @param boundary Actual mapped I/O authority.
     * @param stream Exact forward stream, after every native response consumer.
     * @details No model/controller value is fabricated. This isolates the same
     * native-close/event/worker-join sink used after a complete generation parent;
     * the existing generation suites separately exercise that parent's decisions.
     */
    static void completeStorage(DeviceGraphOrchestrator &runner,
        qwen4exp::ImmutableLookupExecutionBoundary &boundary,void *stream)
    {
        auto *backend=getBackendFor(runner.state_.device_id);
        if(!backend || !runner.device_generation_terminal_host_ready_event_)
            throw std::logic_error("Hosted prefill has no setup-created terminal event");
        boundary.recordTerminalClose(stream);
        if(!backend->recordEvent(runner.device_generation_terminal_host_ready_event_.get(),runner.state_.device_id.ordinal,stream) ||
            !backend->waitForEvent(runner.device_generation_terminal_host_ready_event_.get(),runner.state_.device_id.ordinal))
            throw std::runtime_error("Hosted fixture terminal publication failed");
        runner.awaitRetainedStorageTerminal();
    }
    /** @brief Exercise the actual reset join without reset transaction's deliberate fatal wrapper.
     * @param runner Canonical participant retaining the request and source owners.
     * @param stream Exact reset stream.
     * @return False only after the production reset helper has joined/published a failed worker.
     */
    static bool joinReset(DeviceGraphOrchestrator &runner,void *stream)
    {return runner.joinPriorDeviceWorkForRequestStateReset(stream,"hosted_prefill_failure_retirement");}
    /** @brief Join canonical admission publication for terminal diagnostics, without a data transfer.
     * @param runner Exact arena/controller owner.
     * @param stream Explicit terminal observation stream.
     */
    static void joinControlObservation(DeviceGraphOrchestrator &runner,void *stream)
    {
        auto owner=runner.arena_->getSharedTensor(BufferId::MTP_GENERATION_CONTROL);
        TransferEngine::prepareDeviceInput(owner.get(),runner.state_.device_id,stream);
        TransferEngine::requireDeviceInput(owner.get(),runner.state_.device_id,stream);
    }
    /** @brief Publish the real graph producer event used by subsequent native admission.
     * @param runner Request owner retaining the durable forward event.
     * @param stream Exact stream which just submitted the captured lookup.
     * @param device Original participant identity.
     * @param rows Declared graph capacity, independent of mutable live count.
     * @param decode Whether the submitted producer was scalar decode rather than prefill.
     * @return Whether the production forward completion publication succeeded.
     */
    static bool publish(DeviceGraphOrchestrator &runner,void *stream,DeviceId device,int rows,bool decode=false)
    {
        ForwardOutput output;output.execution={.valid=true,.device=device,.stream=stream,
            .execution_role=ForwardExecutionRole::MainInference,.is_decode=decode,
            .all_position_logits=false,.graph_seq_len=rows,.graph_batch_size=1};
        return runner.publishForwardGraphOutputReady(output);
    }
};
namespace
{
using namespace qwen4exp;
using namespace sampling_math;
/** @brief Fail infrastructure without allowing an invalid captured owner to escape.
 * @param ok Required invariant.
 * @param message Precise failure diagnostic.
 */
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
/** @brief Model-free builder declares the same generic failure lifetime as direct VRAM consumers. */
class ForwardStatusGraph final:public QwenStandardGraph
{
public:
    /** @brief Retain the ordinary real schema, without loading model weights.
     * @param config Explicit minimal real arena/cache geometry.
     */
    explicit ForwardStatusGraph(const GraphConfig &config):QwenStandardGraph(config,nullptr){}
    /** @copydoc IGraphBuilder::deviceGenerationForwardFailurePolicy */
    DeviceGenerationForwardFailurePolicy deviceGenerationForwardFailurePolicy()const override
    {return DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;}
};
/** @brief Tiny real workspace consumer authenticates DGO admission's actual allocation generation. */
class AdmissionWorkspace final:public IComputeStage,public IWorkspaceConsumer
{
public:
    /** @brief Declare one exact-device diagnostic workspace consumer.
     * @param device Original GPU owning the concrete workspace allocation.
     */
    explicit AdmissionWorkspace(DeviceId device):IComputeStage(device){}
    /** @copydoc IComputeStage::execute
     * @details This setup-only consumer is never submitted as model arithmetic.
     */
    bool execute(IDeviceContext *)override{return false;}
    /** @brief Use the existing diagnostic copy-stage category. */
    ComputeStageType type()const override{return ComputeStageType::COPY;}
    /** @brief Return the immutable setup-only consumer identity. */
    std::string name()const override{return "resident_prefill_admission_workspace";}
    /** @brief This metadata-only stage owns no tensor coherence transition. */
    CoherencePolicy coherencePolicy()const override{return CoherencePolicy::NONE;}
    /** @copydoc IComputeStage::supportsBackend */
    bool supportsBackend(ComputeBackendType backend)const override
    {return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    /** @brief This setup-only workspace has no semantic model output. */
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
    /** @copydoc IWorkspaceConsumer::getWorkspaceRequirements
     * @details One named sixteen-byte block authenticates workspace generation;
     * it never duplicates controller, token, history or model-output storage.
     */
    WorkspaceRequirements getWorkspaceRequirements(int,int,int)const override
    {WorkspaceRequirements requirements;requirements.buffers.push_back({"resident_admission_identity",16,256,true});return requirements;}
    /** @copydoc IWorkspaceConsumer::bindWorkspace */
    void bindWorkspace(DeviceWorkspaceManager *workspace)override{workspace_=workspace;}
    /** @brief Report the actual setup workspace binding, without allocating one. */
    bool hasWorkspace()const override{return workspace_!=nullptr;}
    /** @brief Borrow the normal workspace manager retained by the DGO. */
    DeviceWorkspaceManager *getWorkspace()const override{return workspace_;}
private:
    DeviceWorkspaceManager *workspace_=nullptr;
};
/** @brief Retire real graph work before source allocations and before the DGO arena. */
struct Storage
{
    DeviceId device;IBackend *backend;void *stream;void *controllerStream;
    std::vector<std::shared_ptr<DeviceTransferBuffer>> owners;
    std::unique_ptr<IImmutableResidentLookupKernels> kernels;
    std::shared_ptr<TensorBase> hostedProtocol; ///< Original INT32 root retained beyond DGO and graph retirement.
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> hostedBoundary; ///< Retained through graph retirement and before native operands.
    std::unique_ptr<IGPUGraphCapture> graph;
    /** @brief Complete native test work and retire recordings before captured storage. */
    ~Storage()noexcept
    {
        if(!backend->synchronizeStream(stream,device.ordinal) ||
            !backend->synchronizeStream(controllerStream,device.ordinal))std::terminate();
        graph.reset();
    }
    /** @brief Allocate one persistent fixture operand before capture.
     * @param count Exact typed capacity.
     * @return Stable owned native pointer.
     */
    template<class T>T *allocate(std::size_t count)
    {auto owner=TransferEngine::instance().allocateDeviceTransferBuffer(count*sizeof(T),device);require(owner&&owner->isBound(),"Prefill fixture allocation failed");auto *p=static_cast<T *>(owner->mutableDeviceData());owners.push_back(std::move(owner));return p;}
    /** @brief Upload immutable source/IDs or diagnostic live count outside capture.
     * @param destination Exact native operand.
     * @param source Caller-retained values through setup completion.
     */
    template<class T>void upload(T *destination,const std::vector<T> &source)
    {require(backend->hostToDeviceOnStream(destination,source.data(),source.size()*sizeof(T),device.ordinal,stream),"Prefill fixture upload failed");require(backend->synchronizeStream(stream,device.ordinal),"Prefill setup completion failed");}
    /** @brief Observe native terminal evidence only after the tested operations.
     * @param source Exact borrowed device pointer.
     * @param count Required diagnostic elements.
     * @return Completed host evidence, never fed to inference.
     */
    template<class T>std::vector<T> download(const T *source,std::size_t count)
    {std::vector<T> result(count);require(backend->deviceToHostOnStream(result.data(),source,count*sizeof(T),device.ordinal,stream),"Prefill terminal observation failed");require(backend->synchronizeStream(stream,device.ordinal),"Prefill terminal completion failed");return result;}
};
/** @brief Exercise production DGO request reset and native admission around actual captured lookup.
 * @param device Exact single participant, isolated by the explicit native test alias.
 */
void prove(DeviceId device)
{
    auto *backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested native backend unavailable";
    if(!hasCPUBackend())initCPUBackend(-1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        constexpr int capacity=3,width=160,tableRows=37;
        constexpr std::size_t stride=5*sizeof(IQ4_NLBlock),idCapacity=capacity*16,outputElements=idCapacity*width;
        GraphConfig config;config.default_device=device;config.d_model=32;config.d_ff=64;
        config.n_heads=config.n_kv_heads=config.local_n_heads=config.local_n_kv_heads=1;
        config.head_dim=32;config.n_layers=1;config.vocab_size=32;
        config.max_seq_len=16;config.max_request_count=1;config.max_activation_rows=capacity;
        auto mpi=std::make_shared<test::MockMPIContext>();TensorFactory factory(*mpi);
        auto builder=std::make_shared<ForwardStatusGraph>(config);
        AdmissionWorkspace workspace(device);
        DeviceGraphOrchestrator runner(builder,mpi);runner.setTensorFactory(&factory);
        // Declare retirement before even initial DGO setup can submit work.
        // Its actual operation stream is the worker default, checked below.
        Storage storage{device,backend,ExplicitGPUStream(worker.getOrCreateAuxiliaryStream("resident_prefill_lifecycle")).get(),
            ExplicitGPUStream(worker.defaultStream()).get()};
        ASSERT_TRUE(runner.initializeInferenceStateFromArena(1,16,device));
        using Peer=DeviceGraphOrchestratorLiveStateTestAccess;
        ASSERT_EQ(Peer::controllerStream(runner),storage.controllerStream);
        ASSERT_TRUE(Peer::workspace(runner,workspace));
        auto *const control=Peer::control(runner);ASSERT_NE(control,nullptr);
        std::vector<IQ4_NLBlock> blocks(tableRows*5);
        for(std::size_t i=0;i<blocks.size();++i){blocks[i].d=0x2000;for(std::size_t q=0;q<16;++q)blocks[i].qs[q]=static_cast<std::uint8_t>(i*13+q*29);}
        std::vector<std::byte> source(blocks.size()*sizeof(IQ4_NLBlock));std::memcpy(source.data(),blocks.data(),source.size());
        auto *table=storage.allocate<std::byte>(source.size());storage.upload(table,source);
        auto *ids=storage.allocate<std::int64_t>(idCapacity);auto *live=storage.allocate<int>(1);
        auto *output=storage.allocate<float>(outputElements);
        const std::vector<float> poison(outputElements,-12345.0f);storage.upload(output,poison);
        const auto view=ImmutableResidentTableView::bind(device,TensorType::IQ4_NL,table,tableRows,width,stride,source.size(),1);
#ifdef HAVE_CUDA
        if(device.is_cuda())storage.kernels=createCUDAImmutableResidentLookupKernels(view);
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())storage.kernels=createROCmImmutableResidentLookupKernels(view);
#endif
        ASSERT_NE(storage.kernels,nullptr);
        const ImmutableResidentGatherArgs args{{capacity,live},ids,idCapacity,output,outputElements,
            control+kDeviceGenerationControlOk,control+kDeviceGenerationControlErrorCode};
        storage.graph=worker.createGraphCapture(storage.stream);
        {ScopedBackendGraphCapture capture(worker,*storage.graph,"resident pre-admission forward");ASSERT_TRUE(capture.begin());storage.kernels->gather(args,storage.stream);capture.finish();}
        ASSERT_TRUE(storage.graph->instantiate());ASSERT_EQ(storage.graph->nodeCount(),2u);
        const auto owners=storage.owners.size();
        for(int repeat=0;repeat<3;++repeat)
        {
            runner.resetInferenceState(InferenceStateResetRequest::requestBoundary("resident_prefill_regression"));
            // The actual DGO reset clears status and publishes its exact reset
            // event. The actual forward prelude joins it on the distinct stream.
            for(const int rows:{3,1,0,2,3})
            {
                SCOPED_TRACE(::testing::Message()<<device.toString()<<" repeat="<<repeat<<" live="<<rows);
                std::vector<std::int64_t> requested(idCapacity,-1);const auto count=static_cast<std::size_t>(rows)*16;
                for(std::size_t i=0;i<count;++i)requested[i]=static_cast<std::int64_t>(i%tableRows);
                storage.upload(ids,requested);storage.upload(live,std::vector{rows});storage.upload(output,poison);
                ASSERT_TRUE(Peer::prepare(runner,storage.stream,device,capacity));ASSERT_TRUE(storage.graph->launch());
                ASSERT_TRUE(Peer::publish(runner,storage.stream,device,capacity));
                auto expected=poison;std::vector<std::byte> gathered(count*stride);
                for(std::size_t i=0;i<count;++i)std::memcpy(gathered.data()+i*stride,source.data()+requested[i]*stride,stride);
                decodeCPUImmutableLookupRows(TensorType::IQ4_NL,width,stride,gathered.data(),count,expected.data());
                const auto actual=storage.download(output,outputElements);EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
                const auto observed=storage.download(control,kDeviceGenerationControlCount);
                EXPECT_EQ(observed[kDeviceGenerationControlOk],0);EXPECT_EQ(observed[kDeviceGenerationControlErrorCode],0);
            }
            std::vector<std::int64_t> invalid(idCapacity,0);invalid.back()=tableRows;
            storage.upload(ids,invalid);storage.upload(live,std::vector{capacity});storage.upload(output,poison);
            ASSERT_TRUE(Peer::prepare(runner,storage.stream,device,capacity));ASSERT_TRUE(storage.graph->launch());
            ASSERT_TRUE(Peer::publish(runner,storage.stream,device,capacity));
            // Actual DGO response admission must consume that producer event,
            // pass the builder's static policy and retain the first failure.
            ASSERT_TRUE(runner.beginDeviceResidentGeneration({.request_count=1,.max_new_tokens=0,
                .depth_policy=DeviceGenerationPolicy::forwardOnly(),
                .initial_leading_row_disposition=DeviceGenerationLeadingRowDisposition::AlreadyEmitted}));
            Peer::joinControlObservation(runner,storage.stream);
            auto observed=storage.download(control,kDeviceGenerationControlCount);
            EXPECT_EQ(observed[kDeviceGenerationControlOk],0);
            EXPECT_EQ(observed[kDeviceGenerationControlRequestComplete],1);
            EXPECT_EQ(observed[kDeviceGenerationControlErrorCode],static_cast<int>(DeviceGenerationError::ImmutableStorageFailure));
            const auto failed=storage.download(output,outputElements);EXPECT_EQ(std::memcmp(failed.data(),poison.data(),failed.size()*sizeof(float)),0);
            runner.resetInferenceState(InferenceStateResetRequest::requestBoundary("resident_prefill_after_failure"));
            ASSERT_TRUE(Peer::prepare(runner,storage.stream,device,capacity));
            observed=storage.download(control,kDeviceGenerationControlCount);
            EXPECT_EQ(observed[kDeviceGenerationControlOk],0);EXPECT_EQ(observed[kDeviceGenerationControlErrorCode],0);
            ASSERT_TRUE(runner.beginDeviceResidentGeneration({.request_count=1,.max_new_tokens=0,
                .depth_policy=DeviceGenerationPolicy::forwardOnly(),
                .initial_leading_row_disposition=DeviceGenerationLeadingRowDisposition::AlreadyEmitted}));
            Peer::joinControlObservation(runner,storage.stream);
            observed=storage.download(control,kDeviceGenerationControlCount);
            EXPECT_EQ(observed[kDeviceGenerationControlOk],1);EXPECT_EQ(observed[kDeviceGenerationControlErrorCode],0);
            EXPECT_EQ(storage.owners.size(),owners);EXPECT_EQ(storage.kernels->table(),view);
        }
        // Retire the last admitted frontier through the same production reset.
        runner.resetInferenceState(InferenceStateResetRequest::requestBoundary("resident_prefill_terminal"));
        ASSERT_TRUE(Peer::prepare(runner,storage.stream,device,capacity));
    }).get();
}
/** @brief Collect only terminal I/O evidence, preserving the process policy. */
struct HostedEvidence
{
    std::array<const char *,2> names{"LLAMINAR_PERF_STATS_JSON","LLAMINAR_PERF_STATS_FILTER"};
    std::array<std::optional<std::string>,2> prior;
    /** @brief Configure before the storage worker starts, outside native capture. */
    HostedEvidence()
    {
        for(std::size_t i=0;i<names.size();++i)if(const char *v=std::getenv(names[i]))prior[i]=v;
        ::setenv(names[0],"1",1);::setenv(names[1],"immutable_lookup",1);PerfStatsCollector::reset();
    }
    /** @brief Restore environment only after every worker and native owner retired. */
    ~HostedEvidence()
    {
        for(std::size_t i=0;i<names.size();++i)
        {if(prior[i])::setenv(names[i],prior[i]->c_str(),1);else ::unsetenv(names[i]);}
        PerfStatsCollector::reset();
    }
    /** @brief Sum actual immutable terminal records, including explicit reset closes/waits.
     * @param name Exact bounded counter name.
     * @return Completed value in this fixture's isolated collection epoch.
     */
    static double total(const char *name)
    {
        double result=0;
        for(const auto &record:PerfStatsCollector::snapshot({"immutable_lookup"}))
            if(record.name==name)result+=record.value;
        return result;
    }
};
/** @brief Tiny original native source, removed after every source/boundary/graph owner. */
struct HostedSource
{
    std::string path;int fd=-1;std::array<IQ4_NLBlock,37*5> blocks{};
    /** @brief Publish a complete small IQ4_NL table before actual PMA materialization. */
    HostedSource()
    {
        char pattern[]="/tmp/llaminar-hosted-reset-XXXXXX";fd=::mkstemp(pattern);
        if(fd<0)throw std::runtime_error("Hosted reset source creation failed");path=pattern;
        for(std::size_t i=0;i<blocks.size();++i)
        {blocks[i].d=0x2000;for(std::size_t q=0;q<16;++q)blocks[i].qs[q]=static_cast<std::uint8_t>(i*13+q*29);}
        if(::write(fd,blocks.data(),sizeof(blocks))!=static_cast<ssize_t>(sizeof(blocks)))
            throw std::runtime_error("Hosted reset source publication failed");
    }
    /** @brief Retire the source only after all actual lookup owners have joined. */
    ~HostedSource(){::close(fd);::unlink(path.c_str());}
    /** @brief Original source-native geometry, without a synthetic quantization conversion. */
    ImmutableRowTableSource source()const
    {return {.path=path,.tensor_name="per_layer_token_embd.weight",.rows=37,.row_elements=160,.format=GGUFTensorType::IQ4_NL};}
};
/** @brief Real minimal model graph declaring its one pre-created hosted boundary. */
class HostedStatusGraph final:public QwenStandardGraph
{
public:
    /** @brief Retain the ordinary schema; no complete generation parent is prepared by this fixture. */
    explicit HostedStatusGraph(const GraphConfig &config):QwenStandardGraph(config,nullptr){}
    /** @brief Declare the sole boundary during setup, before any live request uses it. */
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary;
    /** @copydoc IGraphBuilder::retainedStorageBoundary */
    std::shared_ptr<IRetainedGraphStorageBoundary> retainedStorageBoundary(DeviceId device)override
    {
        if(boundary&&boundary->device()!=device)throw std::logic_error("Hosted fixture boundary changed device");
        return boundary;
    }
};
/** @brief Retire the external fixture recording before its borrowed DGO arena on every exit. */
struct HostedGraphBeforeArena
{
    Storage &storage;
    /** @brief Join only fixture terminal diagnostics before releasing a recording's borrowed pointers. */
    ~HostedGraphBeforeArena() noexcept
    {
        if(!storage.backend->synchronizeStream(storage.stream,storage.device.ordinal) ||
            !storage.backend->synchronizeStream(storage.controllerStream,storage.device.ordinal))std::terminate();
        storage.graph.reset();
    }
};
/** @brief Exercise actual hosted DGO prefill/reset/readmission and completed/failed retirement.
 * @param device Exact CUDA or ROCm participant selected by its canonical native alias.
 * @param foreignCurrentDevice Select another actual device immediately before every public reset.
 * @details Captured production publish/consume kernels really exchange native
 * rows with the PMA-owned CPU worker. The forward publication is the exact
 * producer joined by the production reset. Nothing in reset reads a live count.
 */
void proveHostedReset(DeviceId device,bool foreignCurrentDevice=false)
{
    auto *backend=getBackendFor(device);if(!backend||backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"Requested native backend unavailable";
    if(foreignCurrentDevice && backend->deviceCount()<2)
    {
        GTEST_SKIP()<<"Foreign-current reset requires two physical devices of the requested backend";
    }
    if(!hasCPUBackend())initCPUBackend(-1);
    HostedEvidence evidence;
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        using Peer=DeviceGraphOrchestratorLiveStateTestAccess;
        for(const auto mode:{ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})
        for(const int capacity:{1,3})
        {
            SCOPED_TRACE(::testing::Message()<<device.toString()<<" capacity="<<capacity<<" tier="<<static_cast<int>(mode));
            constexpr int width=160;constexpr std::size_t stride=5*sizeof(IQ4_NLBlock);
            const auto idCapacity=static_cast<std::size_t>(capacity)*16,outputElements=idCapacity*width;
            HostedSource file;
            // This owner precedes DGO construction, so its graphs/operands remain
            // alive through the runner's event-aware arena/topology destruction.
            Storage storage{device,backend,ExplicitGPUStream(worker.getOrCreateAuxiliaryStream("hosted_prefill_reset")).get(),
                ExplicitGPUStream(worker.defaultStream()).get()};
            auto protocolOwner=std::make_shared<INT32Tensor>(std::vector<std::size_t>{16});
            storage.hostedProtocol=protocolOwner;
            TransferEngine::allocateDeviceStorage(protocolOwner.get(),device);
            auto *state=static_cast<ImmutableLookupDeviceState *>(protocolOwner->gpu_data_ptr());
            GraphConfig config;config.default_device=device;config.d_model=32;config.d_ff=64;
            config.n_heads=config.n_kv_heads=config.local_n_heads=config.local_n_kv_heads=1;
            config.head_dim=32;config.n_layers=1;config.vocab_size=32;
            config.max_seq_len=16;config.max_request_count=1;config.max_activation_rows=capacity;
            auto mpi=std::make_shared<test::MockMPIContext>();TensorFactory factory(*mpi);
            auto builder=std::make_shared<HostedStatusGraph>(config);
            AdmissionWorkspace workspace(device);
            DeviceGraphOrchestrator runner(builder,mpi);runner.setTensorFactory(&factory);
            HostedGraphBeforeArena graphRetirement{storage};
            ASSERT_TRUE(runner.initializeInferenceStateFromArena(1,16,device));
            ASSERT_EQ(Peer::controllerStream(runner),storage.controllerStream);
            ASSERT_TRUE(Peer::workspace(runner,workspace));
            auto *live=storage.allocate<int>(1);auto *ids=storage.allocate<std::int64_t>(idCapacity);
            auto *scratch=storage.allocate<std::byte>(idCapacity*stride);auto *output=storage.allocate<float>(outputElements);
            const auto layout=ImmutableLookupLayout::create(idCapacity,stride);
            const auto policy=ImmutableRowTableStoragePolicy::forStorage(mode,4096,2);
            const PhysicalMemoryResource host{.world_rank=0,.device=DeviceId::cpu(),.total_bytes=1u<<24,.admission_available_bytes=1u<<24};
            PhysicalMemoryPlanBuilder plan;
            plan.add(host,PhysicalMemoryOwner::ImmutableTableStorage,ImmutableRowTable::physicalBytes(file.source(),policy));
            plan.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,ImmutableLookupService::hostBytes(layout));
            auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
            auto source=HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(file.source(),policy,
                DeviceId::cpu(),memory,ImmutableRowTableHostPlacement::taskPolicy()));
            auto service=ImmutableLookupService::create(source,layout,std::array{device},memory,DeviceId::cpu());
            std::unique_ptr<IImmutableLookupKernels> kernels;
#ifdef HAVE_CUDA
            if(device.is_cuda())kernels=createCUDAImmutableLookupKernels(device,TensorType::IQ4_NL,width,stride);
#endif
#ifdef HAVE_ROCM
            if(device.is_rocm())kernels=createROCmImmutableLookupKernels(device,TensorType::IQ4_NL,width,stride);
#endif
            auto *generation=reinterpret_cast<std::uint64_t *>(reinterpret_cast<std::byte *>(state)+offsetof(ImmutableLookupDeviceState,admissionGeneration));
            auto boundary=std::make_shared<ImmutableLookupExecutionBoundary>(service,device,std::move(kernels),protocolOwner);
            builder->boundary=boundary;storage.hostedBoundary=boundary;
            auto *control=Peer::control(runner);ASSERT_NE(control,nullptr);
            boundary->bindGenerationControl({control+kDeviceGenerationControlOk,control+kDeviceGenerationControlErrorCode});
            storage.graph=worker.createGraphCapture(storage.stream);
            {ScopedBackendGraphCapture recording(worker,*storage.graph,"hosted lookup forward without generation parent");
                ASSERT_TRUE(recording.begin());
                boundary->kernels().publish({{capacity,live},ids,generation,state,boundary->channel()},storage.stream);
                boundary->kernels().consume({state,boundary->channel(),scratch,output,
                    control+kDeviceGenerationControlOk,control+kDeviceGenerationControlErrorCode},storage.stream);
                recording.finish();}
            ASSERT_TRUE(storage.graph->instantiate());ASSERT_EQ(storage.graph->nodeCount(),7u);
            const auto allocations=storage.owners.size();
            const std::vector<float> poison(outputElements,-12345.0f);
            // Real capture, capacity-poisoned IDs and exact live outputs. No
            // terminal copy is needed between launch/publication and reset;
            // the reset must inherit the actual forward producer itself.
            const auto launch=[&](int rows,bool decode,bool bad=false) {
                std::vector<std::int64_t> requested(idCapacity,-1);
                for(std::size_t i=0;i<static_cast<std::size_t>(rows)*16;++i)requested[i]=static_cast<std::int64_t>(i%37);
                if(bad)requested.front()=37;
                storage.upload(ids,requested);storage.upload(live,std::vector{rows});storage.upload(output,poison);
                require(decode?Peer::prepareDecode(runner,storage.stream,device):Peer::prepare(runner,storage.stream,device,capacity),
                    "Hosted forward admission failed");
                require(boundary->hasActiveInterval(),"Hosted prelude failed to arm its actual I/O interval");
                require(storage.graph->launch(),"Hosted captured lookup launch failed");
                require(Peer::publish(runner,storage.stream,device,decode?1:capacity,decode),"Hosted forward event publication failed");
            };
            const auto reset=[&] {
                if(foreignCurrentDevice)
                {
                    const int foreign=device.ordinal==0 ? 1 : 0;
                    require(foreign!=device.ordinal && backend->setDevice(foreign),
                        "Hosted reset fixture could not select its real foreign current device");
                }
                // The exact admitted DGO stream/event/root owners stay intact.
                // No worker hop, host count readback or additional fence repairs
                // the current device before this actual public lifecycle call.
                runner.resetInferenceState(InferenceStateResetRequest::requestBoundary("hosted_prefill_interval_regression"));
            };
            auto closes=HostedEvidence::total("reset_terminal_closes"),waits=HostedEvidence::total("reset_terminal_waits");
            auto intervals=HostedEvidence::total("intervals");
            // Standalone prefill -> reset, before any parent preparation or
            // response admission: old code leaves armed=true and this fails.
            launch(capacity,false);
            EXPECT_THROW(boundary->beginAdmission(Peer::epoch(runner)+1,storage.stream),std::logic_error);
            EXPECT_THROW(service->stats(),std::logic_error);
            const auto oldEpoch=Peer::epoch(runner);
            reset();ASSERT_FALSE(boundary->hasActiveInterval());EXPECT_GT(Peer::epoch(runner),oldEpoch);
            auto terminal=boundary->terminalObservation();EXPECT_EQ(terminal.generation,1u);
            EXPECT_EQ(terminal.counters.requests,1u);EXPECT_EQ(terminal.counters.ids,idCapacity);
            EXPECT_EQ(terminal.counters.nativeBytes,idCapacity*stride);
            // Reset closes the GPU interval and joins its exact CPU worker;
            // both the close word and the two-word retirement witness count.
            EXPECT_EQ(terminal.counters.terminalBytes,3u*sizeof(std::uint64_t));
            EXPECT_EQ(HostedEvidence::total("reset_terminal_closes"),++closes);
            EXPECT_EQ(HostedEvidence::total("reset_terminal_waits"),++waits);
            EXPECT_EQ(HostedEvidence::total("intervals"),++intervals);
            auto actual=storage.download(output,outputElements);std::vector<std::byte> packed(idCapacity*stride);
            for(std::size_t i=0;i<idCapacity;++i)std::memcpy(packed.data()+i*stride,file.blocks.data()+(i%37)*5,stride);
            std::vector<float> expected(outputElements);decodeCPUImmutableLookupRows(TensorType::IQ4_NL,width,stride,packed.data(),idCapacity,expected.data());
            EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
            // No active lane: repeated reset must do no close/wait/publication.
            reset();EXPECT_FALSE(boundary->hasActiveInterval());
            EXPECT_EQ(HostedEvidence::total("reset_terminal_closes"),closes);EXPECT_EQ(HostedEvidence::total("reset_terminal_waits"),waits);
            EXPECT_EQ(HostedEvidence::total("intervals"),intervals);
            // New prefill + scalar history forward share exactly one interval.
            launch(0,false);launch(1,true);
            Peer::completeStorage(runner,*boundary,storage.stream);
            ASSERT_FALSE(boundary->hasActiveInterval());terminal=boundary->terminalObservation();
            EXPECT_EQ(terminal.generation,2u);EXPECT_EQ(terminal.counters.requests,2u);
            EXPECT_EQ(terminal.counters.ids,16u);EXPECT_EQ(terminal.counters.nativeBytes,16*stride);
            EXPECT_EQ(HostedEvidence::total("intervals"),++intervals);
            reset();EXPECT_EQ(HostedEvidence::total("reset_terminal_closes"),closes);
            EXPECT_EQ(HostedEvidence::total("reset_terminal_waits"),waits);EXPECT_EQ(HostedEvidence::total("intervals"),intervals);
            // Dirty large->small->empty replay retains the exact graph/buffers.
            for(const int rows:{capacity,1,0})
            {
                launch(rows,false);reset();ASSERT_FALSE(boundary->hasActiveInterval());
                terminal=boundary->terminalObservation();EXPECT_EQ(terminal.counters.ids,static_cast<std::uint64_t>(rows)*16);
                EXPECT_EQ(terminal.counters.nativeBytes,static_cast<std::uint64_t>(rows)*16*stride);
                actual=storage.download(output,outputElements);
                EXPECT_TRUE(std::all_of(actual.begin()+static_cast<std::size_t>(rows)*16*width,actual.end(),[](float x){return x==-12345.0f;}));
                EXPECT_EQ(HostedEvidence::total("reset_terminal_closes"),++closes);
                EXPECT_EQ(HostedEvidence::total("reset_terminal_waits"),++waits);
                EXPECT_EQ(HostedEvidence::total("intervals"),++intervals);
            }
            // A failed provider still owes a worker join. Exercise the real
            // reset prelude before its deliberate fatal whole-reset wrapper.
            launch(1,false,true);
            EXPECT_FALSE(Peer::joinReset(runner,storage.controllerStream));
            ASSERT_FALSE(boundary->hasActiveInterval());terminal=boundary->terminalObservation();
            EXPECT_EQ(terminal.counters.failures,1u);EXPECT_EQ(terminal.counters.nativeBytes,0u);
            EXPECT_EQ(HostedEvidence::total("reset_terminal_closes"),++closes);
            EXPECT_EQ(HostedEvidence::total("reset_terminal_waits"),++waits);
            EXPECT_EQ(HostedEvidence::total("intervals"),++intervals);
            EXPECT_THROW(boundary->requireArmedForLaunch(),std::runtime_error);
            reset();EXPECT_EQ(HostedEvidence::total("reset_terminal_closes"),closes);
            EXPECT_EQ(HostedEvidence::total("reset_terminal_waits"),waits);EXPECT_EQ(HostedEvidence::total("intervals"),intervals);
            // The retired provider retains its original fatal I/O cause.
            // Readmission must fail at health validation, before arming a new
            // source request, rather than report a caller-lifecycle mismatch.
            const auto failedRequests = service->stats().requests;
            try
            {
                boundary->beginAdmission(Peer::epoch(runner), storage.stream);
                FAIL() << "Fatal immutable source accepted readmission";
            }
            catch (const std::runtime_error &error)
            {
                const auto cause = "fatal I/O protocol status " +
                    std::to_string(static_cast<std::uint64_t>(ImmutableLookupStatus::InvalidRowId)) + ";";
                EXPECT_NE(std::string(error.what()).find(cause), std::string::npos);
            }
            EXPECT_FALSE(boundary->hasActiveInterval());
            EXPECT_EQ(service->stats().requests, failedRequests);
            EXPECT_EQ(storage.owners.size(),allocations);
        }
    }).get();
}
#ifdef HAVE_CUDA
/** @test CUDA actual hosted prefill retires its interval before request reset/new admission. */
TEST(ResidentLookupPrefillLifecycle,CUDAHostedTerminalReset){proveHostedReset(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test ROCm uses the identical exact-stream native close/event/worker-join reset boundary. */
TEST(ResidentLookupPrefillLifecycle,ROCmHostedTerminalReset){proveHostedReset(DeviceId::rocm(0));}
#endif
#ifdef HAVE_CUDA
/** @test CUDA cold prefill, error admission and actual DGO request reset share one controller. */
TEST(ResidentLookupPrefillLifecycle,CUDACanonicalController){prove(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @test ROCm enforces the same captured prefill and canonical first-error lifetime. */
TEST(ResidentLookupPrefillLifecycle,ROCmCanonicalController){prove(DeviceId::rocm(0));}
#endif

#ifdef HAVE_CUDA
/** @test Both actual CUDA owners retire/reset from a different current device, including empty/repeated and dirty capacity replays. */
TEST(ResidentLookupPrefillLifecycle,CUDAForeignCurrentDeviceHostedTerminalReset)
{
    proveHostedReset(DeviceId::cuda(0),true);
    proveHostedReset(DeviceId::cuda(1),true);
}
#endif
#ifdef HAVE_ROCM
/** @test Both actual HIP owners establish their declared device before native close and exact terminal-event retirement. */
TEST(ResidentLookupPrefillLifecycle,ROCmForeignCurrentDeviceHostedTerminalReset)
{
    proveHostedReset(DeviceId::rocm(0),true);
    proveHostedReset(DeviceId::rocm(1),true);
}
#endif
} // namespace
} // namespace llaminar2
