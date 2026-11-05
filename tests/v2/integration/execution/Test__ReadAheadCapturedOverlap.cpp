/**
 * @file Test__ReadAheadCapturedOverlap.cpp
 * @brief Causal two-lane proof through the actual retained production executor.
 *
 * The first side gather waits on a native source gate. Useful main work alone
 * publishes that gate, so same-stream serialization creates a causal cycle.
 * The real production resident decoder then feeds another real resident decoder
 * on the consuming main lane. Terminal bytes and untouched inactive tails prove
 * both joins. All native creation and data ingress precede capture; observations
 * occur at the exact complete-parent terminal. No profiler attachment is used.
 *
 * An absent installed stream/event certificate is an explicit qualification
 * failure before native family creation. This file cannot mint test certificates.
 */
#include "backends/ReadAheadNativeResourceFamily.h"
#include "backends/NativeContextPreparation.h"
#include "backends/NativeAuxiliaryResourceCatalog.h"
#include "backends/ReadAheadNativeParentTerminal.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "kernels/qwen4exp/ImmutableResidentLookupKernels.h"
#include "kernels/common/SamplingMath.h"
#include "memory/BufferArena.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llaminar2::test { namespace {
using namespace qwen4exp;
/** @brief Fail the selected native qualification instead of skipping a path. */
void require(bool ok,const char* why){if(!ok){std::cerr<<"Read-ahead qualification failure: "<<why<<'\n';throw std::runtime_error(why);}}
/** @brief Bound one diagnostic terminal observation without unwinding live native storage. */
bool terminalReady(IWorkerGPUContext& worker,void* event)
{
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    do
    {
        bool ready=false;if(!worker.queryEventChecked(event,ready))return false;
        if(ready)return true;
        std::this_thread::yield();
    }while(std::chrono::steady_clock::now()<deadline);
    return false;
}
/**
 * @brief Diagnostic-only submission guard; a failed causal proof can never destroy live graphs.
 *
 * Normal retirement uses exactly the admitted terminal. On timeout only, both
 * existing exact lanes receive a source-gate release, followed by one bounded
 * drain. A released failure remains a hard test failure and is never timed or
 * reported as overlap. An undrainable driver exits without C++/GPU destruction.
 */
class Submission final
{
public:
    /** @brief Arm before the first actual native launch in this serialized batch. */
    Submission(IWorkerGPUContext& worker,IBackend& backend,DeviceId device,void* main,void* side,
        void* gate,std::shared_ptr<void> terminal,ReadAheadNativeResourceFamily* family)
        :worker_(worker),backend_(backend),device_(device),main_(main),side_(side),gate_(gate),
         terminal_(std::move(terminal)),family_(family)
    {if(family_)family_->beginSubmission();}
    /** @brief Avoid unwinding any native graph/allocation after an unretired launch exception. */
    ~Submission(){if(!retired_){std::cerr<<"Unretired native diagnostic submission: exiting without destroying live graph/storage\n";std::quick_exit(4);}}
    /** @brief Record and observe the actual complete parent; timeout recovery always fails. */
    void retire()
    {
        if(family_)family_->recordTerminal();
        else require(worker_.recordEventChecked(terminal_.get(),main_),"Matched serial terminal failed");
        if(!terminalReady(worker_,terminal_.get()))
        {
            const bool releaseMain=backend_.streamPublishTimelineSignal32(main_,gate_,1,device_.ordinal);
            const bool releaseSide=backend_.streamPublishTimelineSignal32(side_,gate_,1,device_.ordinal);
            if(!releaseMain || !releaseSide || !terminalReady(worker_,terminal_.get()))std::quick_exit(5);
            if(family_ && !family_->observeTerminal())std::quick_exit(6);
            retired_=true;
            throw std::runtime_error("Causal read-ahead proof failed; diagnostic-only source release drained the failed parent");
        }
        if(family_)require(family_->observeTerminal(),"Read-ahead native readers remain live after terminal");
        retired_=true;
    }
private:
    IWorkerGPUContext& worker_;IBackend& backend_;DeviceId device_;void *main_,*side_,*gate_;
    std::shared_ptr<void> terminal_;ReadAheadNativeResourceFamily* family_;bool retired_=false;
};
/**
 * @brief Test-only logical source-gate epochs, reset solely after complete native retirement.
 *
 * Every epoch is a new diagnostic source transaction over stable admitted
 * storage. Its signal is monotonic 0->1 while live. Reusing the physical word
 * starts a new epoch only after the original exact terminal and all readers
 * have retired; it never rewrites a live native wait or changes captured nodes.
 */
class CausalGateEpoch final
{
public:
    /** @brief Bind the original signal, exact streams and sole retirement authority. */
    CausalGateEpoch(IWorkerGPUContext& worker,IBackend& backend,DeviceId device,void* stream,
        void* gate,std::shared_ptr<void> terminal,std::shared_ptr<ReadAheadNativeResourceFamily> family)
        :worker_(worker),backend_(backend),device_(device),stream_(stream),gate_(gate),terminal_(std::move(terminal)),family_(std::move(family)){}
    /** @brief Begin a fresh logical source epoch; no old participant may remain live. */
    void resetAfterRetirement()
    {
        require(!family_->hasOutstandingSubmission(),"A live source epoch cannot reset its gate");
        if(epoch_)
        {
            bool complete=false;require(worker_.queryEventChecked(terminal_.get(),complete) && complete,
                "Source epoch reset requires the original completed parent terminal");
        }
        require(backend_.streamPublishTimelineSignal32(stream_,gate_,0,device_.ordinal),"Diagnostic source epoch initialization failed");
        ++epoch_;
    }
private:
    IWorkerGPUContext& worker_;IBackend& backend_;DeviceId device_;void *stream_,*gate_;
    std::shared_ptr<void> terminal_;std::shared_ptr<ReadAheadNativeResourceFamily> family_;std::uint64_t epoch_=0;
};
/** @brief One native gate and original/output storage, retained after all graphs. */
struct Storage
{
    DeviceId device;IBackend* backend;void* main;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::optional<ExclusiveModelRetirementTicket>& retirement;
    std::vector<PhysicalMemoryAllocationLease> claims;
    std::vector<std::shared_ptr<DeviceTransferBuffer>> buffers;
    void* gate=nullptr;
    /** @brief Observe original live allocations before releasing any fixture storage.
     * The ticket uses the exact original TransferEngine buffer extents. Graphs,
     * snapshots and native handles have already retired; opaque pool tokens
     * remain charged until the outer scope consumes the actual reset receipt.
     */
    ~Storage()
    {
        if(!buffers.empty())
        {
            std::size_t bytes=0;
            for(const auto& buffer:buffers)bytes+=buffer->sizeBytes();
            retirement.emplace(TransferEngine::instance().beginExclusiveModelRetirement(
                {.device=device,.reusable_workspace_bytes=bytes}));
        }
        if(gate)backend->freeStreamTimelineSignal32(gate,device.ordinal);
    }
    /** @brief Claim the exact persistent allocation before using the public transfer authority. */
    template<class T>T* allocate(std::size_t count)
    {
        claims.push_back(memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,count*sizeof(T)));
        auto owner=TransferEngine::instance().allocateDeviceTransferBuffer(count*sizeof(T),device);
        require(owner && owner->isBound(),"Read-ahead fixture storage failed");
        auto* result=static_cast<T*>(owner->mutableDeviceData());buffers.push_back(std::move(owner));return result;
    }
    /** @brief Initial/reset publication is fixture ingress, never captured inference work. */
    template<class T>void upload(T* target,const std::vector<T>& values)
    {
        require(backend->hostToDeviceOnStream(target,values.data(),values.size()*sizeof(T),device.ordinal,main),"Read-ahead ingress failed");
        require(backend->synchronizeStream(main,device.ordinal),"Read-ahead ingress terminal failed");
    }
    /** @brief Observe only after the exact complete-parent terminal; test evidence is intentional final I/O. */
    template<class T>std::vector<T> download(const T* source,std::size_t count)
    {
        std::vector<T> result(count);
        require(backend->deviceToHostOnStream(result.data(),source,count*sizeof(T),device.ordinal,main),"Read-ahead terminal evidence failed");
        require(backend->synchronizeStream(main,device.ordinal),"Read-ahead terminal evidence completion failed");return result;
    }
};
/** @brief Snapshot metadata borrowing the fixture's original PMA-owned native bytes.
 * No host mirror, allocation, coherence mutation or second owner is created. */
class SnapshotView final : public ITensor {
public:
    /** @brief Bind immutable geometry to an existing retained test allocation. */
    SnapshotView(DeviceId device,float* data,std::size_t rows,std::size_t cols):device_(device),data_(data),shape_{rows,cols}{}
    int native_type_id()const override{return TensorTypeId::FP32;}
    const std::vector<std::size_t>& shape()const override{return shape_;}
    std::size_t numel()const override{return shape_[0]*shape_[1];}
    std::size_t size_bytes()const override{return numel()*sizeof(float);}
    DeviceId home_device()const override{return device_;}
    bool isDeviceValid()const override{return true;}
    const float* data()const override{throw std::logic_error("Native snapshot view has no host data");}
    float* mutable_data()override{throw std::logic_error("Native snapshot view has no mutable host data");}
    const void* raw_data()const override{return nullptr;}
    void* raw_mutable_data()override{throw std::logic_error("Native snapshot view has no host storage");}
    const void* gpu_data_ptr()const override{return data_;}
    void* gpu_data_ptr()override{return data_;}
    void to_fp32(float*)const override{throw std::logic_error("Acquire immutable snapshot through executor publication");}
private:
    DeviceId device_;float* data_;std::vector<std::size_t> shape_;
};
/** @brief Thin fixture operations exercise generic production scheduling and unchanged production gather kernels. */
class Stage final:public IComputeStage
{
public:
    enum class Operation{Frontier,Gather,Trunk,Consume};
    /** @brief Bind immutable test operation metadata; no native work or allocation occurs here. */
    Stage(DeviceId device,Operation operation,IBackend* backend,void* gate,int* marker,
        IImmutableResidentLookupKernels* kernels,ImmutableResidentGatherArgs args)
        :IComputeStage(device),operation_(operation),backend_(backend),gate_(gate),marker_(marker),kernels_(kernels),args_(args){}
    /** @brief Enqueue only native capturable work on the executor's exact lane. */
    bool execute(IDeviceContext*)override
    {
        require(gpuStream()!=nullptr,"Read-ahead fixture lost its exact stream");
        switch(operation_)
        {
        case Operation::Frontier:return true;
        case Operation::Gather:
            require(backend_->streamWaitTimelineSignal32(gpuStream(),gate_,1,device().ordinal),
                "Native captured source gate is unavailable; read-ahead overlap is unqualified");
            kernels_->gather(args_,gpuStream());return true;
        case Operation::Trunk:
            require(backend_->memset(marker_,0x2a,sizeof(int),device().ordinal,gpuStream()),"Useful primary work failed");
            return backend_->streamPublishTimelineSignal32(gpuStream(),gate_,1,device().ordinal);
        case Operation::Consume:kernels_->gather(args_,gpuStream());return true;
        }
        throw std::logic_error("Unknown fixture operation");
    }
    /** @brief Fixture operations have no undeclared tensor authority; raw buffers are explicit admitted test infrastructure. */
    CoherencePolicy coherencePolicy()const override{return CoherencePolicy::NONE;}
    /** @brief Keep runtime backend selection explicit. */
    bool supportsBackend(ComputeBackendType backend)const override
    {return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    /** @brief Identify unchanged native gather arithmetic in execution diagnostics. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_NGRAM_LOOKUP;}
    /** @brief Expose only explicitly bound original native output metadata for snapshot qualification. */
    StageDumpInfo buildDumpInfoImpl()const override{
        StageDumpInfo info;
        if(snapshot_)info.outputs.push_back({"gathered_rows",args_.output,
            snapshot_->shape()[0],snapshot_->shape()[1],"FP32",sizeof(float),snapshot_->size_bytes(),snapshot_});
        return info;
    }
    /** @brief Cold-only binding of original source metadata, retained through capture. */
    void bindSnapshot(ITensor& view){snapshot_=&view;}
private:
    ITensor* snapshot_=nullptr;
    Operation operation_;IBackend* backend_;void* gate_;int* marker_;
    IImmutableResidentLookupKernels* kernels_;ImmutableResidentGatherArgs args_;
};
/** @brief Project immutable startup capacities; certificate catalog access remains source-owned. */
PhysicalMemoryResource observed(DeviceId device)
{
    auto& manager=DeviceManager::instance();manager.initialize(-1,false);
    const auto backend=device.is_cpu()?ComputeBackendType::CPU:device.is_cuda()?ComputeBackendType::GPU_CUDA:ComputeBackendType::GPU_ROCM;
    const auto& devices=manager.devices();const auto found=std::find_if(devices.begin(),devices.end(),[&](const auto& value){return value.type==backend && (device.is_cpu() || value.device_id==device.ordinal);});
    require(found!=devices.end(),"Read-ahead fixture endpoint absent from startup inventory");
    return {0,device,found->total_memory_bytes,found->free_memory_bytes};
}
/** @brief Replay one frozen two-lane parent, then matched serial/concurrent diagnostic batches. */
void prove(DeviceId device,bool snapshots=false)
{
    constexpr int capacity=4,width=32,idsCapacity=capacity*16;
    constexpr std::size_t elements=idsCapacity*width;
    constexpr std::size_t payloadBytes=(elements*3+idsCapacity*4+4)*sizeof(float);
    // HIP's timeline API owns a native 64-bit HSA signal even when only its
    // low 32 bits participate. CUDA's corresponding allocation is four bytes.
    const std::size_t signalBytes=device.is_rocm()?sizeof(std::uint64_t):sizeof(std::uint32_t);
    const auto cpu=observed(DeviceId::cpu()),gpu=observed(device);
    // No declaration can supply a byte allowance or bypass source certification.
    auto familyPlan=ReadAheadNativeResourceFamily::bom(device,cpu,gpu);
    const std::array observations{cpu,gpu};
    const std::array selected{device};
    auto source=NativeContextPreparation::create(observations,0,selected);
    std::optional<PreparedGPUContextUsePermit> permit(source->acquire(device));
    const auto memory=source->memory();
    source->freezeForPlanning();
    PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
    for(const auto& resource:familyPlan.resources())plan.add(resource);
    const auto terminalPlan=ReadAheadNativeParentTerminal::bom(device,cpu,gpu);
    for(const auto& resource:terminalPlan.resources())plan.add(resource);
    plan.add(gpu,PhysicalMemoryOwner::ActivationArena,payloadBytes+signalBytes);
    if(snapshots)plan.add(gpu,PhysicalMemoryOwner::GraphSnapshotArena,2*elements*sizeof(float));
    plan.add(gpu,PhysicalMemoryOwner::NativeGraphExecutable,2*GPUGraphMemoryContract::reservationBytesPerExecutable(device));
    source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
    // A BOM entry alone does not reserve a physical obligation. Retain the
    // existing graph-contract tokens through original context reset, including
    // driver backing that survives destruction of both retained executables.
    auto graphReservation=memory->reserveNewAllocations(device,PhysicalMemoryOwner::NativeGraphExecutable,
        2*GPUGraphMemoryContract::reservationBytesPerExecutable(device));
    auto* backend=getBackendFor(device);require(backend!=nullptr,"Read-ahead native backend unavailable");
    {const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);
        original.requireWorker(permit->worker());}
    auto strongWorker=permit->worker();
    auto& worker=*strongWorker;
    // The preparation borrow ends before Storage obtains its exclusive reset ticket.
    permit.reset();
    require(strongWorker!=nullptr,"Read-ahead original worker is unavailable");
    std::optional<ExclusiveModelRetirementTicket> retirement;
    std::exception_ptr failure;
    try{worker.submitAndWait([&]{
        const auto main=worker.defaultStream();require(main!=nullptr,"Read-ahead parent stream is null");
        Storage storage{device,backend,main,memory,retirement};
        auto* table=storage.allocate<float>(elements);auto* output=storage.allocate<float>(elements);
        auto* consumed=storage.allocate<float>(elements);auto* ids=storage.allocate<std::int64_t>(idsCapacity);auto* consumeIds=storage.allocate<std::int64_t>(idsCapacity);
        auto* control=storage.allocate<int>(3);auto* marker=storage.allocate<int>(1);
        storage.claims.push_back(memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,signalBytes));
        require(backend->supportsStreamTimelineSignal32(device.ordinal),"Native timeline source gate is unavailable");
        storage.gate=backend->allocateStreamTimelineSignal32(device.ordinal);require(storage.gate!=nullptr,"Native source gate allocation failed");
        std::vector<float> source(elements);for(std::size_t i=0;i<elements;++i)source[i]=static_cast<float>(i+1)*0.125f;
        std::vector<std::int64_t> identity(idsCapacity);for(int i=0;i<idsCapacity;++i)identity[i]=i;
        const std::vector<float> poison(elements,-12345.0f);
        storage.upload(table,source);storage.upload(ids,identity);storage.upload(consumeIds,identity);storage.upload(control,std::vector<int>{capacity,1,0});
        storage.upload(output,poison);storage.upload(consumed,poison);
        const auto sourceView=ImmutableResidentTableView::bind(device,TensorType::FP32,reinterpret_cast<const std::byte*>(table),idsCapacity,width,width*sizeof(float),elements*sizeof(float),1);
        const auto consumerView=ImmutableResidentTableView::bind(device,TensorType::FP32,reinterpret_cast<const std::byte*>(output),idsCapacity,width,width*sizeof(float),elements*sizeof(float),1);
        std::unique_ptr<IImmutableResidentLookupKernels> gather,consume;
#ifdef HAVE_CUDA
        if(device.is_cuda()){gather=createCUDAImmutableResidentLookupKernels(sourceView);consume=createCUDAImmutableResidentLookupKernels(consumerView);}
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm()){gather=createROCmImmutableResidentLookupKernels(sourceView);consume=createROCmImmutableResidentLookupKernels(consumerView);}
#endif
        require(gather && consume,"Read-ahead production resident kernels unavailable");
        // The terminal belongs to its separately certified baseline population;
        // the read-ahead family remains exactly one stream plus fork/join events.
        auto parentTerminal=ReadAheadNativeParentTerminal::materialize(memory,strongWorker,device);
        auto terminal=parentTerminal->eventOwner();
        require(terminal!=nullptr,"Qualified read-ahead parent terminal unavailable");
        // Establish an observed idle terminal during fixture setup. A later
        // creator/capture failure can then retire this owner without treating
        // an unrecorded event as evidence that readers completed.
        require(worker.recordEventChecked(terminal.get(),main) && terminalReady(worker,terminal.get()),
            "Initial read-ahead parent terminal did not retire");
        auto family=ReadAheadNativeResourceFamily::materialize(memory,strongWorker,device,main);family->bindTerminal(terminal);
        std::unique_ptr<IDeviceContext> context;
#ifdef HAVE_CUDA
        if(device.is_cuda())context=std::make_unique<CUDADeviceContext>(device,device.ordinal);
#endif
#ifdef HAVE_ROCM
        if(device.is_rocm())context=std::make_unique<ROCmDeviceContext>(device,device.ordinal);
#endif
        require(context!=nullptr,"Read-ahead device context unavailable");
        SnapshotView outputView(device,output,idsCapacity,width),consumedView(device,consumed,idsCapacity,width);
        std::vector<float> expectedSnapshot;std::size_t snapshotCount=0;
        GraphExecutorConfig config{.default_device=device};
        if(snapshots){
            config.require_snapshot_memory_authority=true;
            config.snapshot_callback=[&](const std::string& name,const StageDumpInfo& info){
                if(name!="gather" && name!="consume")return;
                ASSERT_EQ(info.outputs.size(),1u);
                ASSERT_NE(info.outputs[0].data,nullptr);
                EXPECT_EQ(std::memcmp(info.outputs[0].data,expectedSnapshot.data(),elements*sizeof(float)),0)<<name;
                ++snapshotCount;
            };
        }
        BufferArena arena;DeviceGraphExecutor executor(config);executor.setArena(&arena);
        if(snapshots)executor.setGraphSnapshotMemoryReservation(std::make_shared<PhysicalMemoryOwnerReservation>(
            memory->reserveNewAllocations(device,PhysicalMemoryOwner::GraphSnapshotArena,2*elements*sizeof(float))));
        const ImmutableResidentGatherArgs arguments{{capacity,control},ids,idsCapacity,output,elements,control+1,control+2};
        auto consuming=arguments;consuming.output=consumed;consuming.ids=consumeIds;
        auto build=[&](bool side){
            ComputeGraph graph;
            graph.setReadAheadPhasePolicy(ForwardExecutionPhase::Decode,side?
                ImmutableLookupInvocationSchedule::ResidentSideLane:ImmutableLookupInvocationSchedule::ConsumerScheduled);
            graph.addNode("frontier",std::make_unique<Stage>(device,Stage::Operation::Frontier,backend,storage.gate,marker,nullptr,arguments),device);
            graph.addNode("gather",std::make_unique<Stage>(device,Stage::Operation::Gather,backend,storage.gate,marker,gather.get(),arguments),device).addDependency("gather","frontier");
            graph.addNode("trunk",std::make_unique<Stage>(device,Stage::Operation::Trunk,backend,storage.gate,marker,nullptr,arguments),device).addDependency("trunk","frontier");
            graph.addNode("consume",std::make_unique<Stage>(device,Stage::Operation::Consume,backend,storage.gate,marker,consume.get(),consuming),device).addDependency("consume","gather").addDependency("consume","trunk");
            if(snapshots){
                static_cast<Stage*>(graph.getNode("gather")->stage.get())->bindSnapshot(outputView);
                static_cast<Stage*>(graph.getNode("consume")->stage.get())->bindSnapshot(consumedView);
            }
            if(side)graph.setReadAheadBranch({"frontier","gather","consume",family});
            else graph.addDependency("trunk","gather");
            return graph;
        };
        auto graph=build(true);auto native=worker.createGraphCapture(main);require(native!=nullptr,"Read-ahead capture unavailable");
        require(executor.captureRetainedGraphFragment(graph,context.get(),native.get(),"causal read-ahead parent"),"Production read-ahead capture failed");
        std::vector<GPUGraphKernelNodeInfo> nativeKernels;std::string nativeDetail;
        require(native->inspectKernelNodes(nativeKernels,&nativeDetail),nativeDetail.c_str());
        require(nativeKernels.size()==4,"Causal parent must retain exactly two production validator/decoder pairs");
        for(const auto& kernel:nativeKernels)
            require(kernel.local_memory_bytes_per_thread==0,"Production resident replay unexpectedly uses local storage");
        require(native->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&nativeDetail),nativeDetail.c_str());
        require(native->instantiate(),"Causal parent instantiation failed");
        const auto originalIdentity=graph.readAheadCaptureIdentity();
        CausalGateEpoch gateEpoch(worker,*backend,device,main,storage.gate,terminal,family);
        std::uint64_t replayIndex=0;
        for(const int rows:{capacity,1,0,3,capacity,1,-1,capacity+1,1})
        {
            gateEpoch.resetAfterRetirement();
            std::vector<std::int64_t> query(idsCapacity,-1);
            const bool valid=rows>=0 && rows<=capacity;
            if(valid)for(int i=0;i<rows*16;++i)query[i]=(i*17+replayIndex*13)%idsCapacity;
            ++replayIndex;
            storage.upload(ids,query);storage.upload(control,std::vector<int>{rows,1,0});
            storage.upload(marker,std::vector<int>{0});storage.upload(output,poison);storage.upload(consumed,poison);
            // Every changed query/count/history epoch re-proves the same
            // ordering: only useful primary work releases its delayed source.
            {Submission submission(worker,*backend,device,main,family->stream(),storage.gate,terminal,family.get());
             require(native->launch(),"Causal retained parent launch failed");submission.retire();}
            auto expected=poison;
            if(valid)for(int i=0;i<rows*16;++i)
                std::copy_n(source.begin()+query[i]*width,width,expected.begin()+i*width);
            if(snapshots){
                expectedSnapshot=expected;const auto before=snapshotCount;
                require(executor.publishSnapshotsAfterGraphExecution(graph,main,"retired read-ahead snapshots"),"Captured snapshot publication failed");
                EXPECT_EQ(snapshotCount-before,2u);
            }
            const auto actual=storage.download(consumed,elements);
            EXPECT_EQ(std::memcmp(actual.data(),expected.data(),elements*sizeof(float)),0);
            const auto produced=storage.download(output,elements);
            EXPECT_EQ(std::memcmp(produced.data(),expected.data(),elements*sizeof(float)),0);
            EXPECT_EQ(storage.download(control,3),(std::vector<int>{rows,valid?1:0,valid?0:static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure)}));
            EXPECT_EQ(storage.download(marker,1).front(),0x2a2a2a2a);
            EXPECT_EQ(graph.readAheadCaptureIdentity(),originalIdentity);
            graph.reset();
        }
        if(snapshots){native.reset();return;}
        // The same source/native kernels, pointers, counts and trunk work form
        // the serial timing control. The gate is already monotonically released.
        auto serialGraph=build(false);auto serial=worker.createGraphCapture(main);
        require(serial && executor.captureRetainedGraphFragment(serialGraph,context.get(),serial.get(),"matched serial parent") &&
            serial->validateExecutableMemoryClass(GPUGraphExecutableClass::General,&nativeDetail) && serial->instantiate(),"Matched serial control failed");
        for(const auto& candidate:std::array<std::pair<const char*,IGPUGraphCapture*>,2>{{{"serial",serial.get()},{"read_ahead",native.get()}}})
        {
            constexpr int repeats=20;const auto begin=std::chrono::steady_clock::now();
            const bool side=candidate.second==native.get();
            Submission submission(worker,*backend,device,main,family->stream(),storage.gate,terminal,side?family.get():nullptr);
            for(int replay=0;replay<repeats;++replay)require(candidate.second->launch(),"Matched unprofiled replay failed");
            submission.retire();
            const auto us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/repeats;
            std::cout<<"read_ahead_diagnostic schedule="<<candidate.first<<" mean_wall_us="<<us<<" rows=1 width=32 profiler=none\n";
        }
        serial.reset();native.reset();
    });}catch(...){failure=std::current_exception();}
    strongWorker.reset();
    if(retirement)
    {
        const auto retired=TransferEngine::instance().completeExclusiveModelRetirement(std::move(*retirement));
        require(retired.runtime_reset_invoked && retired.retired_context_generation &&
            retired.retired_runtime_generation && retired.successor_runtime_generation &&
            retired.runtime_post_reset_state==DeviceRuntimePostResetState::Quiescent,
            "Read-ahead native fixture did not retire its original physical context");
        graphReservation={};
        for(const auto allocator:{DeviceId::cpu(),device})
            for(const auto owner:{PhysicalMemoryOwner::NativeGraphExecutable,PhysicalMemoryOwner::ActivationArena,
                                 PhysicalMemoryOwner::GraphSnapshotArena})
            {
                EXPECT_EQ(memory->reservedBytes(allocator,owner),0u);
                EXPECT_EQ(memory->claimedBytes(allocator,owner,PhysicalMemoryMaterializationKind::NewAllocation),0u);
            }
        // Destroyed handles do not release opaque process-runtime commitments.
        // The SAME original PMA retains exactly the installed typed allowances.
        for(const auto& entry:std::array{
            std::pair{PhysicalMemoryOwner::NativeReadAheadFamily,NativeAuxiliaryResourceProgram::readAheadSide()},
            std::pair{PhysicalMemoryOwner::NativeReadAheadParentTerminal,NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::readAheadParentTerminal())}})
        {
            const auto allowance=NativeAuxiliaryResourceCatalog::allowance(device.type,entry.second);
            ASSERT_EQ(allowance.lifetime,NativePhysicalLifetime::ProcessRuntime);
            EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),entry.first),allowance.hostBytes);
            EXPECT_EQ(memory->reservedBytes(device,entry.first),allowance.deviceBytes);
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),entry.first,PhysicalMemoryMaterializationKind::NewAllocation),0u);
            EXPECT_EQ(memory->claimedBytes(device,entry.first,PhysicalMemoryMaterializationKind::NewAllocation),0u);
        }
        std::cout<<"read_ahead_context_retired device="<<device.toString()
            <<" generation="<<retired.retired_context_generation<<" reset=1 same_pma_process_commitments_retained=1\n";
    }
    else require(bool(failure),"Read-ahead fixture omitted its original storage retirement ticket");
    if(failure)std::rethrow_exception(failure);
}
#ifdef GPU_CONTEXT_TEST_BACKEND_CUDA
TEST(ReadAheadCapturedOverlap, CUDACapturedSnapshotPublication){prove(DeviceId::cuda(0),true);}
TEST(ReadAheadCapturedOverlap, CUDACausalForkJoinAndMatchedReplay){prove(DeviceId::cuda(0));}
#endif
#ifdef GPU_CONTEXT_TEST_BACKEND_ROCM
TEST(ReadAheadCapturedOverlap, ROCmCapturedSnapshotPublication){prove(DeviceId::rocm(0),true);}
TEST(ReadAheadCapturedOverlap, ROCmCausalForkJoinAndMatchedReplay){prove(DeviceId::rocm(0));}
#endif
}}
