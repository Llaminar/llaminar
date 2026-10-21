/**
 * @file Test__ImmutableFixedHeadNativeCohort.cpp
 * @brief Real captured fixed-head native packets and both NCCL/RCCL exchanges.
 *
 * This is the model-free component boundary: original source ranges are declared,
 * PMA-admitted and loaded by the production immutable owner; every participant
 * produces its own validation header and packed native payload. Neither peer
 * headers nor received payloads are staged by the host. Fixture-owned original
 * query inputs do not certify Rank/DGO replicated token/history issuance, sampler,
 * MTP acceptance, prefix restore or complete model serving. Those require the
 * separately registered canonical full-model Rank qualification.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "collective/LocalTPContext.h"
#include "execution/compute_stages/stages/NativeAllGatherStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include "execution/mtp/DeviceGenerationCollectiveReceipt.h"
#include "kernels/qwen4exp/QwenFixedHeadTerminalCounters.h"
#include "kernels/qwen4exp/ImmutableFixedHeadLookupKernels.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "loaders/ResidentImmutableRowTable.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "memory/BufferArena.h"
#include "models/qwen4exp/Qwen4ExpFixedHeadShards.h"
#include "models/qwen4exp/Qwen4ExpPublishedGeometry.h"
#include "tensors/TensorFactory.h"
#include "transfer/TransferEngine.h"
#include "../../../mocks/MockMPIContext.h"
#include "../../../utils/EmbeddingVerifierFormats.h"
#include <algorithm>
#include <array>
#include <barrier>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <thread>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace {
/** @brief Reject failure while captured owners still exist. @param ok Required invariant. @param why Precise failure diagnostic. */
void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
/** @return Unequal original head vocabularies, including source gaps and a trailing unqueried tail. */
NgramHashPolicy sourcePolicy() {
    NgramHashPolicy p;p.multipliers={1,3,7};std::uint64_t end=0;
    for(int h=0;h<16;++h){if(h==8)end+=2;p.offsets[h]=end;p.moduli[h]=h+3;end+=h+3;}return p;
}
/** @brief Tiny original IQ4_NL file, never a replacement model or a full-model source claim. */
class NativeFile final {
public:
    static constexpr std::size_t rows=178,width=qwen4exp::published::RowElements,rowBytes=90,offset=37;
    std::string path;std::vector<std::byte> original;
    /** @brief Preserve canonical format-factory native bytes unchanged in a private immutable file. */
    NativeFile() {
        const auto &formats=test::embeddingVerifierFormats();
        const auto found=std::find_if(formats.begin(),formats.end(),[](const auto &f){return std::string_view(f.label)=="IQ4_NL";});
        require(found!=formats.end(),"Canonical IQ4_NL fixture absent");auto tensor=found->create({rows,width},913);
        require(tensor->native_type()==TensorType::IQ4_NL&&tensor->size_bytes()==rows*rowBytes,"Native fixture source geometry changed");
        const auto *bytes=static_cast<const std::byte *>(tensor->raw_data());original.assign(bytes,bytes+tensor->size_bytes());
        char name[]="/tmp/qfn-fixed-native-cohort-XXXXXX";int fd=::mkstemp(name);
        require(fd>=0,"Cannot create original native source");path=name;
        try {
            require(::ftruncate(fd,offset+original.size())==0,"Cannot size native source");
            require(::pwrite(fd,original.data(),original.size(),offset)==static_cast<ssize_t>(original.size()),"Cannot publish native source");
            const int closed=::close(fd);fd=-1;require(closed==0,"Cannot close native source");
        } catch(...){if(fd>=0)::close(fd);::unlink(path.c_str());throw;}
    }
    /** @brief Remove the private file only after all native uploads/consumers retire. */
    ~NativeFile(){::unlink(path.c_str());}
    /** @return Complete original descriptor; a selected range never relabels its source. */
    ImmutableRowTableSource source() const {return {.path=path,.tensor_name="per_layer_token_embd.weight",.file_offset=offset,.rows=rows,.row_elements=width,.format=GGUFTensorType::IQ4_NL};}
};
/** @brief Observe exact canonical native send receipt binding without adding an instrumentation bank. */
class ObservedCohort final:public LocalTPContext {
public:
    using LocalTPContext::LocalTPContext;
    /** @brief Bind original controller pairs for the next retired-frame setup; never retain live values.
     * @param participant Actual collective coordinate. @param header Original raw native header-send pair.
     * @param payload Original raw native payload-send pair.
     */
    void bindReceipts(int participant,unsigned long long *header,unsigned long long *payload) {
        require(participant>=0&&participant<degree()&&header&&payload&&header!=payload,"Invalid original controller receipt binding");
        headers_[participant]=header;payloads_[participant]=payload;
    }
    /** @brief The mandatory header must exercise the counted native receipt entrypoint, never raw fixed transport.
     * @param send Original send pointer. @param receive Original receive pointer. @param count Original elements.
     * @param dtype Original datatype. @param participant Actual coordinate. @param stream Exact stream.
     * @param stage Actual stage. @return No raw operation is supported in this component.
     */
    bool allgatherRawOnStream(const void *,void *,std::size_t,CollectiveDataType,int,void *,const std::string &) override {
        throw std::runtime_error("Fixed-head receipt test reached a raw header without native send instrumentation");
    }
    /** @brief Forward the exact production counted operation with its original typed controller receipt.
     * @param operation Actual collective. @param send Produced native packet. @param receive Native rank-major bank.
     * @param rows Original transport geometry+canonical receipt. @param dtype Original type.
     * @param reduction Native operation. @param participant Actual coordinate. @param stream Exact stream.
     * @param stage Actual immutable stage. @param counter Must be absent; the canonical row owns its receipt.
     * @return Unmodified native result; no test-authored byte/count value is substituted.
     */
    bool nativeRowsOnStream(NativeRowCollective operation,const void *send,void *receive,const NativeCollectiveRows &rows,
        CollectiveDataType dtype,CollectiveOp reduction,int participant,void *stream,const std::string &stage,
        unsigned long long *counter=nullptr) override {
        require(participant>=0&&participant<degree()&&!counter&&dtype==CollectiveDataType::INT8&&
            operation==NativeRowCollective::AllGather&&reduction==CollectiveOp::ALLGATHER,"Foreign native operation/receipt authority");
        const bool header=stage=="fixed_header_allgather";
        require(header||stage=="fixed_payload_allgather","Foreign fixed-head transport stage");
        require(rows.payloadReceipt()==(header?headers_[participant]:payloads_[participant]),"Native receipt escaped the original canonical controller");
        if(header)require(rows.rows().capacity()==1&&rows.elementsPerRow()==sizeof(FixedHeadValidationHeader),"Header did not submit the exact counted8B protocol");
        return LocalTPContext::nativeRowsOnStream(operation,send,receive,rows,dtype,reduction,participant,stream,stage,counter);
    }
private:
    std::array<unsigned long long *,8> headers_{},payloads_{}; ///< Setup-only addresses, never a live host count mirror.
};
/** @brief Direct production-kernel component node; no Rank/DGO proof is minted or emulated here. */
class KernelNode final:public IComputeStage {
public:
    /** @brief Exactly one local operation, with collective nodes declared separately. */
    enum class Op { Local,Consensus,Pack,Decode };
    /**
     * @brief Borrow admitted component operands through complete graph retirement.
     * @param device Actual participant. @param op One local production kernel operation.
     * @param kernels Retained installed native bundle. @param args Exact immutable frame.
     * @param label Stable graph identity. @param inputs Actual producer roots.
     * @param outputs Exact roots published by this operation.
     */
    KernelNode(DeviceId device,Op op,const IImmutableFixedHeadLookupKernels &kernels,const ImmutableFixedHeadLookupArgs &args,
        std::string label,std::vector<TensorBase *> inputs,std::vector<TensorBase *> outputs)
        :IComputeStage(device),op_(op),kernels_(kernels),args_(args),label_(std::move(label)),inputs_(std::move(inputs)),outputs_(std::move(outputs)){}
    /** @brief Run the installed operation and publish genuine outputs. @param context Exact native execution context. @return True after successful submission/publication. */
    bool execute(IDeviceContext *context) override {
        require(context&&context->deviceId()==device(),"Foreign native component execution context");const auto execution=gpuExecution();
        for(auto *input:inputs_)execution.requirePreparedInput(input);for(auto *output:outputs_)execution.requirePreparedOutput(output);
        switch(op_){case Op::Local:kernels_.validateLocal(args_,execution.nativeStream());break;
            case Op::Consensus:kernels_.validateCohort(args_,execution.nativeStream());break;
            case Op::Pack:kernels_.pack(args_,execution.nativeStream());break;
            case Op::Decode:kernels_.decode(args_,execution.nativeStream());break;}
        for(auto *output:outputs_)execution.publish(output);return true;
    }
    /** @return Component-local stable graph label. */
    std::string name()const override{return label_;}
    /** @return Existing native lookup classification. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_NGRAM_LOOKUP;}
    /** @param type Queried backend. @return Only this exact component node's actual native backend. */
    bool supportsBackend(ComputeBackendType type)const override {return device().is_cuda()?type==ComputeBackendType::GPU_CUDA:type==ComputeBackendType::GPU_ROCM;}
    /** @return No semantic model checkpoint is fabricated by a component kernel wrapper. */
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
private:
    Op op_;const IImmutableFixedHeadLookupKernels &kernels_;const ImmutableFixedHeadLookupArgs &args_;
    std::string label_;std::vector<TensorBase *> inputs_,outputs_;
};
/** @brief Test-only host failure point; it creates no production query/cohort authority. */
enum class HostFailureInjection { None, DuringArenaUpload, BeforeFirstRetainedLaunch };
/** @brief Distinguish the intentional peer drop from every unrelated native/fixture failure. */
class InjectedHostFailure final:public std::runtime_error {
public:
    /** @brief Emit a unique typed fixture-only failure before the launch rendezvous. */
    InjectedHostFailure():std::runtime_error("Intentional fixed-head host participant drop"){}
};
/** @brief Passive terminal observations, written only by their own worker and read after join. */
struct ParticipantObservation {
    std::size_t setupJoins=0; ///< Exact setup-stream completions before upload owners retire.
    std::size_t instantiated=0; ///< Actual successful native executable instantiations.
    std::size_t validated=0; ///< Actual complete kernel/resource validations.
    std::size_t launchAttempts=0; ///< Entries to the guarded native executable submission boundary.
    std::size_t terminalJoins=0; ///< Successful exact-stream retirement joins.
    std::size_t retiredGraphs=0; ///< Executables reset before captured operand destruction.
};
/** @brief Cover asynchronous arena setup BEFORE any captured-work owner exists.
 * @details The later Retirement still fences/reset recorded work before kernels
 * and stages destruct. This outer guard separately covers a setup throw between
 * the first H2D and construction of that later guard; it owns no graph or event.
 */
class SetupStreamRetirement final {
public:
    /** @brief Borrow exact stream/abort authority while the arena outlives this scope. */
    SetupStreamRetirement(IWorkerGPUContext &worker,ObservedCohort &tp,void *stream,ParticipantObservation &observation)
        :worker_(worker),tp_(tp),stream_(stream),observation_(observation),exceptions_(std::uncaught_exceptions()){}
    /** @brief Complete pending setup uploads before their source/destination arena roots retire. */
    ~SetupStreamRetirement()noexcept {
        if(std::uncaught_exceptions()>exceptions_)tp_.requestAbort();
        try{if(!worker_.synchronizeStreamChecked(stream_))std::terminate();++observation_.setupJoins;}
        catch(...){std::terminate();}
    }
private:
    IWorkerGPUContext &worker_;ObservedCohort &tp_;void *stream_;ParticipantObservation &observation_;int exceptions_;
};
/** @brief Graph retirement precedes every borrowed arena, read token, stage and source owner on all exits. */
class Retirement final {
public:
    /**
     * @brief Retain terminal lifetime without submitting native work.
     * @param worker Exact native owner. @param tp Actual native communicator.
     * @param graph Retained executable. @param stream Exact captured stream.
     * @param rendezvous Existing fixture participant barrier. @param dropped One-shot abort/drop observation.
     * @param observation Passive per-worker terminal receipt, never execution policy.
     */
    Retirement(IWorkerGPUContext &worker,ObservedCohort &tp,std::unique_ptr<IGPUGraphCapture> &graph,
        void *stream,std::barrier<> &rendezvous,bool &dropped,ParticipantObservation &observation)
        :worker_(worker),tp_(tp),graph_(graph),stream_(stream),barrier_(rendezvous),dropped_(dropped),observation_(observation),exceptions_(std::uncaught_exceptions()){}
    /** @brief Abort peers on failure, release host rendezvous, then join this terminal diagnostic boundary. */
    ~Retirement()noexcept {
        if(std::uncaught_exceptions()>exceptions_){tp_.requestAbort();barrier_.arrive_and_drop();dropped_=true;}
        try{if(!worker_.synchronizeStreamChecked(stream_))std::terminate();++observation_.terminalJoins;
            if(graph_){graph_->reset();graph_.reset();++observation_.retiredGraphs;}}
        catch(...){std::terminate();}
    }
private:
    IWorkerGPUContext &worker_;ObservedCohort &tp_;std::unique_ptr<IGPUGraphCapture> &graph_;void *stream_;
    std::barrier<> &barrier_;bool &dropped_;ParticipantObservation &observation_;int exceptions_;
};
/** @brief Exact registered role geometry, reused for the canonical activation BOM and arena materialization. */
struct Role {BufferId id;std::size_t rows,columns;const char *type;};
/** @brief Declare complete original operands and disjoint packet roots. @param capacity Physical token rows. @param degree Actual native cohort degree. @return Exact typed role inventory, without anonymous reserve. */
std::vector<Role> roles(int capacity,int degree) {
    const auto packet=static_cast<std::size_t>(capacity)*(16/degree)*NativeFile::rowBytes;
    require(packet%4==0,"Native source packet must have exact INT32 backing");
    return {{BufferId::PREFILL_CHUNK_GEOMETRY,1,1,"INT32"},
        {BufferId::MTP_GENERATION_CONTROL,1,sampling_math::kDeviceGenerationControlCount,"INT32"},
        {BufferId::QFN_NGRAM_ROW_IDS,static_cast<std::size_t>(capacity),32,"INT32"},
        {BufferId::QFN_NGRAM_SHARD_LOCAL_HEADER,1,2,"INT32"},
        {BufferId::QFN_NGRAM_SHARD_HEADERS,static_cast<std::size_t>(degree),2,"INT32"},
        {BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS,1,1,"INT32"},
        {BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,1,packet/4,"INT32"},
        {BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE,1,packet*degree/4,"INT32"},
        {BufferId::QFN_PLE_EMBEDDING,static_cast<std::size_t>(capacity),16*NativeFile::width,"FP32"}};
}
/** @brief Price the same registered component roles. @param capacity Physical token rows. @param degree Actual native cohort degree. @return Exact backing bytes, never communication extent. */
std::size_t arenaBytes(int capacity,int degree){std::size_t bytes=0;for(const auto &r:roles(capacity,degree))bytes+=r.rows*r.columns*4;return bytes;}
/**
 * @brief Prove native component exchanges or their prelaunch host-failure retirement.
 * @param first Explicit backend selector. @param degree Actual homogeneous participant count2/4/8.
 * @param injection Typed fixture-only failure point; ordinary tests retain every existing replay.
 */
void prove(DeviceId first,int degree,HostFailureInjection injection=HostFailureInjection::None) {
    auto *backend=getBackendFor(first);if(!backend||backend->deviceCount()<degree)GTEST_SKIP()<<"Native TP"<<degree<<" hardware unavailable";
    if(!hasCPUBackend())initCPUBackend(-1);NativeFile file;const auto source=file.source();const auto policy=sourcePolicy();
    std::vector<DeviceId> devices;std::vector<GlobalDeviceAddress> members;std::vector<PhysicalMemoryAllocatorIdentity> owners;
    std::vector<PhysicalMemoryResource> resources;
    for(int p=0;p<degree;++p){const auto d=first.is_cuda()?DeviceId::cuda(degree-1-p):DeviceId::rocm(degree-1-p);
        devices.push_back(d);members.push_back(GlobalDeviceAddress::fromLocalDeviceId(d));owners.push_back({0,d});
        resources.push_back({.world_rank=0,.device=d,.total_bytes=backend->deviceMemoryTotal(d.ordinal),.admission_available_bytes=backend->deviceMemoryFree(d.ordinal)});}
    auto placement=Qwen4ExpFixedHeadShards::declare(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),policy.offsets,policy.moduli,owners);
    auto tp=std::make_unique<ObservedCohort>(members,std::vector<float>{},first.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL);
    constexpr std::size_t slot=4096;PhysicalMemoryPlanBuilder plan;
    const auto host=PhysicalMemoryResource{.world_rank=0,.device=DeviceId::cpu(),.total_bytes=getCPUBackend()->deviceMemoryTotal(0),.admission_available_bytes=getCPUBackend()->deviceMemoryFree(0)};
    plan.add(host,PhysicalMemoryOwner::WeightLoadStaging,slot*degree);
    // BufferArena's ordinary host allocations and device allocations share one
    // owner inventory; the host mirror is test setup storage, never a service.
    plan.add(host,PhysicalMemoryOwner::ActivationArena,arenaBytes(17,degree)*degree);
    for(const auto &resource:resources){plan.add(resource,PhysicalMemoryOwner::PrimaryModelWeights,slot);
        plan.add(resource,PhysicalMemoryOwner::WeightLoadStaging,slot);plan.add(resource,PhysicalMemoryOwner::ActivationArena,arenaBytes(17,degree));}
    const auto native=placement->fixedHeadShardStorageBOM(resources);for(const auto &bom:native.resources())plan.add(bom);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
    std::barrier rendezvous(degree);std::vector<std::exception_ptr> errors(degree);std::vector<ParticipantObservation> observations(degree);std::vector<std::thread> threads;
    for(int participant=0;participant<degree;++participant)threads.emplace_back([&,participant]{bool dropped=false;
        try {auto &worker=GPUDeviceContextPool::instance().getContext(devices[participant]);worker.submitAndWait([&]{
            const auto device=devices[participant];void *stream=worker.getOrCreateAuxiliaryStream("fixed_native_cohort");
            void *uploadStream=worker.getOrCreateAuxiliaryStream("fixed_native_source_upload");
            require(stream&&uploadStream&&stream!=uploadStream,"Missing exact distinct source producer/consumer streams");
            LoadOrchestrator loader(backend,memory,PhysicalMemoryOwner::PrimaryModelWeights);loader.addDevice(device.ordinal);
            loader.planRawWeight(device.ordinal,"ordinary.fixture",1,slot/sizeof(float),slot);loader.allocate(slot,1);
            const std::vector<float> ordinary(slot/sizeof(float),.25F);loader.addWeightJob(device.ordinal,{.name="ordinary.fixture",.host_raw_data=ordinary.data(),
                .raw_bytes=slot,.format=RepackFormat::RAW_FP,.N=1,.K=slot/sizeof(float),.is_asymmetric=false});loader.load();loader.finalize();
            auto table=ResidentImmutableRowTable::prepareRange(source,*placement,memory,device,loader.releasedWeightLoadStaging(device.ordinal),uploadStream,TransferEngine::instance());
            const auto assignment=Qwen4ExpFixedHeadShards::assignmentFor(*placement,{0,device});require(assignment.participant==static_cast<std::size_t>(participant),"Physical owner sort changed source/head coordinate");
            require(table->source()==source&&table->sourceRange()==assignment.source&&table->nativeBytes()==assignment.source.nativeBytes(),"Range owner changed original source");
            auto read=table->acquire(stream,TransferEngine::instance());
            TransferEngine::instance().requireImmutableNativeDeviceInput(read,device,stream);const auto view=table->rangeView();
            for(const int capacity:{1,17}) {
                auto hostClaim=memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,arenaBytes(capacity,degree));
                auto deviceClaim=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationArena,arenaBytes(capacity,degree));
                test::MockMPIContext mpi;TensorFactory factory(mpi);BufferArena arena(ArenaConfig{.factory=&factory});
                for(const auto &r:roles(capacity,degree))require(arena.registerBuffer(r.id,r.rows,r.columns,r.type,device),"Register component role");require(arena.allocate(),"Allocate component arena");
                SetupStreamRetirement setupRetirement(worker,*tp,stream,observations[participant]);
                for(const auto &r:roles(capacity,degree)){auto *t=arena.getTensor(r.id);std::memset(t->raw_mutable_data(),0,t->size_bytes());
                    if(r.id==BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE)std::memset(t->raw_mutable_data(),0xa7,t->size_bytes());
                    if(r.id==BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE)std::memset(t->raw_mutable_data(),0x6d,t->size_bytes());
                    if(r.id==BufferId::QFN_PLE_EMBEDDING)std::fill_n(static_cast<float *>(t->raw_mutable_data()),t->numel(),-12345.F);
                    TransferEngine::prepareDeviceInput(t,device,stream);TransferEngine::requireDeviceInput(t,device,stream);
                    if(injection==HostFailureInjection::DuringArenaUpload && capacity==1 && r.id==BufferId::PREFILL_CHUNK_GEOMETRY) {
                        // Both first H2D enqueues have real owners before the
                        // deliberate failure. This readiness edge prevents a fast
                        // peer's abort from racing the other's setup validation.
                        rendezvous.arrive_and_wait();
                        require(!tp->isAbortRequested(),"Peer failed during initial arena upload");
                        if(participant==degree-1)throw InjectedHostFailure{};
                        // The failed peer drops this separate exit rendezvous
                        // only AFTER its setup-stream retirement has completed.
                        rendezvous.arrive_and_wait();
                        require(!tp->isAbortRequested(),"Peer failed during initial arena upload");
                    }
                }
                const auto tensor=[&](BufferId id){return arena.getSharedTensor(id).get();};
                auto *count=tensor(BufferId::PREFILL_CHUNK_GEOMETRY),*ids=tensor(BufferId::QFN_NGRAM_ROW_IDS),*header=tensor(BufferId::QFN_NGRAM_SHARD_LOCAL_HEADER),
                    *headers=tensor(BufferId::QFN_NGRAM_SHARD_HEADERS),*transport=tensor(BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS),
                    *packet=tensor(BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE),*gathered=tensor(BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE),*output=tensor(BufferId::QFN_PLE_EMBEDDING),*control=tensor(BufferId::MTP_GENERATION_CONTROL);
                const auto binding=DeviceGenerationControlBinding::admit(arena,device,0);
                const auto headerReceipt=DeviceGenerationCollectiveReceipt::fixedHead(binding,DeviceGenerationCollectiveReceiptKind::FixedHeadHeader,capacity,16/degree,NativeFile::rowBytes);
                const auto payloadReceipt=DeviceGenerationCollectiveReceipt::fixedHead(binding,DeviceGenerationCollectiveReceiptKind::FixedHeadPayload,capacity,16/degree,NativeFile::rowBytes);
                tp->bindReceipts(participant,headerReceipt.native(device),payloadReceipt.native(device));
                const std::array roots{count,ids,header,headers,transport,packet,gathered,output,control};
                std::array<const void *,9> rootPointers{};
                for(std::size_t index=0;index<roots.size();++index)rootPointers[index]=roots[index]->gpu_data_ptr();
                const auto allocatedBuffers=arena.stats().total_buffers,allocatedBytes=arena.stats().total_bytes;
                const auto frozen=ImmutableFixedHeadLookupPlan::bind(view,policy,degree,participant);
                std::unique_ptr<IImmutableFixedHeadLookupKernels> kernels;
#ifdef HAVE_CUDA
                if(device.is_cuda())kernels=createCUDAImmutableFixedHeadLookupKernels(frozen);
#endif
#ifdef HAVE_ROCM
                if(device.is_rocm())kernels=createROCmImmutableFixedHeadLookupKernels(frozen);
#endif
                require(bool(kernels),"Native fixed-head bundle absent");const auto packetBytes=static_cast<std::size_t>(capacity)*frozen.heads()*NativeFile::rowBytes;
                ImmutableFixedHeadLookupArgs args{DeviceRowRange::deviceCounted(capacity,static_cast<int *>(count->gpu_data_ptr())),static_cast<std::int64_t *>(ids->gpu_data_ptr()),static_cast<std::size_t>(capacity)*16,
                    static_cast<FixedHeadValidationHeader *>(header->gpu_data_ptr()),static_cast<FixedHeadValidationHeader *>(headers->gpu_data_ptr()),static_cast<std::size_t>(degree),static_cast<int *>(transport->gpu_data_ptr()),
                    static_cast<std::byte *>(packet->gpu_data_ptr()),packetBytes,static_cast<std::byte *>(gathered->gpu_data_ptr()),packetBytes*degree,static_cast<float *>(output->gpu_data_ptr()),static_cast<std::size_t>(capacity)*16*NativeFile::width,binding.healthy(),binding.error(),{binding.row(),sampling_math::kDeviceGenerationControlCount}};
                const auto identity=kernels->captureIdentity(args,stream);auto context=IDeviceContext::create(device,1);
                require(identity.originalRows==DeviceRowRange::deviceCounted(capacity,static_cast<int *>(count->gpu_data_ptr())).captureIdentity() &&
                    args.transportRows!=args.tokenRows.countOwner(),"Transport extent replaced the original logical request authority");
                std::vector<std::unique_ptr<IComputeStage>> nodes;
                nodes.push_back(std::make_unique<KernelNode>(device,KernelNode::Op::Local,*kernels,args,"validate_local",std::vector{ids,count,control},std::vector{header,transport,control}));
                nodes.push_back(std::make_unique<NativeAllGatherStage>(NativeAllGatherStage::Params{.device_id=device,.tp_ctx=tp.get(),.local_input=header,.rank_major_output=headers,.bytes_per_participant=sizeof(FixedHeadValidationHeader),.participant=participant,.stage_name="fixed_header_allgather",.input_buffer_id=BufferId::QFN_NGRAM_SHARD_LOCAL_HEADER,.output_buffer_id=BufferId::QFN_NGRAM_SHARD_HEADERS,.arena_rows=NativeAllGatherStage::ArenaRows{transport,BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS,0,1,sizeof(FixedHeadValidationHeader)},.payload_receipt=headerReceipt}));
                nodes.push_back(std::make_unique<KernelNode>(device,KernelNode::Op::Consensus,*kernels,args,"consensus",std::vector{headers,count,control},std::vector{transport,control}));
                nodes.push_back(std::make_unique<KernelNode>(device,KernelNode::Op::Pack,*kernels,args,"pack_native",std::vector{ids,count,control,transport},std::vector{packet,control}));
                nodes.push_back(std::make_unique<NativeAllGatherStage>(NativeAllGatherStage::Params{.device_id=device,.tp_ctx=tp.get(),.local_input=packet,.rank_major_output=gathered,.bytes_per_participant=packetBytes,.participant=participant,.stage_name="fixed_payload_allgather",.input_buffer_id=BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,.output_buffer_id=BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE,.arena_rows=NativeAllGatherStage::ArenaRows{transport,BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS,0,capacity,static_cast<std::size_t>(frozen.heads())*NativeFile::rowBytes},.payload_receipt=payloadReceipt}));
                nodes.push_back(std::make_unique<KernelNode>(device,KernelNode::Op::Decode,*kernels,args,"decode",std::vector{ids,count,control,header,headers,transport,packet,gathered},std::vector{output,control}));
                std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
                const std::array<std::vector<TensorBase *>,6> inputs{std::vector{ids,count,control},std::vector{header,transport,control},std::vector{headers,count,control},std::vector{ids,count,control,transport},std::vector{packet,transport,control},std::vector{ids,count,control,header,headers,transport,packet,gathered}};
                const std::array<std::vector<TensorBase *>,6> outputs{std::vector{header,transport,control},std::vector{headers,control},std::vector{transport,control},std::vector{packet,control},std::vector{gathered,control},std::vector{output,control}};
                for(std::size_t index=0;index<nodes.size();++index){require(nodes[index]->prepareGraphLaunch(context.get(),stream),"Prepare actual component node");GraphCaptureDependencyLedger::StagePlan p;p.stage_identity=nodes[index].get();p.stage_name=nodes[index]->name();
                    for(auto *input:inputs[index]){std::size_t producer=index;while(producer&&std::find(outputs[producer-1].begin(),outputs[producer-1].end(),input)==outputs[producer-1].end())--producer;
                        if(producer)p.internal_inputs.push_back({input,producer-1});else p.external_inputs.push_back(input);}
                    p.outputs.assign(outputs[index].begin(),outputs[index].end());plans.push_back(std::move(p));}
                const auto headerSignature=nodes[1]->graphCaptureVariantSignature(),payloadSignature=nodes[4]->graphCaptureVariantSignature();
                GraphCaptureDependencyLedger ledger(device,stream,std::move(plans),"fixed_native_two_allgathers");std::unique_ptr<IGPUGraphCapture> graph;
                // These host setup inputs outlive the terminal retirement guard even
                // when a later asynchronous publication throws before observation.
                std::vector<std::int64_t> originalIds(static_cast<std::size_t>(capacity)*16,-1);
                std::vector<int> controlWords(sampling_math::kDeviceGenerationControlCount);
                int originalCount=0;
                Retirement retirement(worker,*tp,graph,stream,rendezvous,dropped,observations[participant]);graph=worker.createGraphCapture(stream);rendezvous.arrive_and_wait();
                // A dropped peer satisfies the host barrier but cannot participate
                // in native capture. Abort is not permission to enqueue without it.
                require(!tp->isAbortRequested(),"Peer failed before native component capture");
                {ScopedBackendGraphCapture recording(worker,*graph,"fixed_native_two_allgathers",&ledger);require(recording.begin(),"Begin real component capture");
                    for(auto &node:nodes){ScopedGraphCaptureStage stage(node.get());require(node->execute(context.get()),"Record actual component operation");stage.complete();}recording.finish();}
                require(graph->instantiate(),"Instantiate real component graph");++observations[participant].instantiated;
                std::vector<GPUGraphKernelNodeInfo> kernelNodes;std::string resourceDetail;
                require(graph->inspectKernelNodes(kernelNodes,&resourceDetail),"Inspect actual component kernels/resources");
                std::array<bool,5> observedProjectKernels{};constexpr std::array<std::string_view,5> symbols{
                    "initializeHeader","validateOwnedIds","validateHeaders","packNative","decodeRankMajor"};
                for(const auto &node:kernelNodes)for(std::size_t index=0;index<symbols.size();++index)
                    if(node.name.find(symbols[index])!=std::string::npos){observedProjectKernels[index]=true;
                        require(!node.local_memory_bytes_per_thread,"Fixed-head component kernel acquired private local memory");}
                require(std::all_of(observedProjectKernels.begin(),observedProjectKernels.end(),[](bool found){return found;}),
                    "Runtime resource evidence omitted an actual fixed-head kernel symbol");
                ++observations[participant].validated;
                const auto upload=[&](void *dst,const void *src,std::size_t bytes){if(bytes)require(backend->hostToDeviceOnStream(dst,src,bytes,device.ordinal,stream),"Publish component request input");};
                const auto observe=[&](const void *src,std::size_t bytes){std::vector<std::byte> out(bytes);
                    struct Join {IWorkerGPUContext &worker;void *stream;~Join()noexcept{if(!worker.synchronizeStreamChecked(stream))std::terminate();}} join{worker,stream};
                    if(bytes)require(backend->deviceToHostOnStream(out.data(),src,bytes,device.ordinal,stream),"Observe component terminal bytes");
                    require(worker.synchronizeStreamChecked(stream),"Join component terminal observation");return out;};
                for(int replay=0;replay<16;++replay) {
                    const std::array<int,8> rows{capacity,std::min(15,capacity),1,0,std::max(1,capacity-1),0,capacity,1};
                    const int live=rows[replay%rows.size()];const bool failure=replay>=12;originalCount=live;
                    if(replay==13&&participant==degree-1)originalCount=capacity+1;
                    if(replay==14&&participant==degree-1)originalCount=0;
                    std::fill(originalIds.begin(),originalIds.end(),-1);
                    for(int row=0;row<capacity;++row)for(int h=0;h<16;++h)if(row<live)originalIds[row*16+h]=policy.offsets[h]+(row+replay*3+h)%policy.moduli[h];
                    if(replay==12&&participant==degree-1)originalIds[frozen.firstHead()]=static_cast<std::int64_t>(policy.offsets[frozen.firstHead()]+policy.moduli[frozen.firstHead()]);
                    std::fill(controlWords.begin(),controlWords.end(),0);
                    if(replay==15&&participant==degree-1)controlWords[static_cast<std::size_t>(binding.error()-binding.row())]=37;
                    upload(count->gpu_data_ptr(),&originalCount,sizeof(originalCount));upload(ids->gpu_data_ptr(),originalIds.data(),ids->size_bytes());upload(control->gpu_data_ptr(),controlWords.data(),control->size_bytes());
                    const auto oldPacket=observe(packet->gpu_data_ptr(),packet->size_bytes()),oldGathered=observe(gathered->gpu_data_ptr(),gathered->size_bytes()),oldOutput=observe(output->gpu_data_ptr(),output->size_bytes());
                    binding.requireRequest(device,0);TransferEngine::instance().requireImmutableNativeDeviceInput(read,device,stream);
                    if(injection==HostFailureInjection::BeforeFirstRetainedLaunch&&capacity==1&&replay==0) {
                        // All real graphs and source/read/operand validations are complete.
                        // The last participant then fails before the launch rendezvous;
                        // no fake collective or test-produced peer header is substituted.
                        rendezvous.arrive_and_wait();
                        require(!tp->isAbortRequested(),"Peer failed before retained native component launch");
                        if(participant==degree-1)throw InjectedHostFailure{};
                    }
                    rendezvous.arrive_and_wait();
                    require(!tp->isAbortRequested(),"Peer failed before retained native component launch");
                    ++observations[participant].launchAttempts;
                    // A removed abort guard must fail here without deliberately entering
                    // an unmatched vendor collective or waiting for its native timeout.
                    require(injection==HostFailureInjection::None,"Aborted cohort reached retained native submission");
                    require(graph->launch(),"Submit retained real two-allgather graph");
                    // A second same-stream real lookup accumulates native atomic
                    // sends in the SAME original request row (no reset/overwrite).
                    // Next replay resets the full row and cannot republish this window.
                    const std::uint64_t calls=replay==0?2:1;
                    if(replay==0){rendezvous.arrive_and_wait();require(!tp->isAbortRequested(),"Peer failed before second native contribution");
                        ++observations[participant].launchAttempts;require(graph->launch(),"Accumulate second native lookup without counter reset");}
                    const auto actual=observe(output->gpu_data_ptr(),output->size_bytes());const auto packed=observe(packet->gpu_data_ptr(),packet->size_bytes());const auto received=observe(gathered->gpu_data_ptr(),gathered->size_bytes());
                    const auto countBytes=observe(transport->gpu_data_ptr(),sizeof(int)),controlBytes=observe(control->gpu_data_ptr(),control->size_bytes()),headerBytes=observe(headers->gpu_data_ptr(),headers->size_bytes());
                    int extent=0,error=0;std::memcpy(&extent,countBytes.data(),sizeof(int));
                    std::vector<std::int32_t> terminal(sampling_math::kDeviceGenerationControlCount);std::memcpy(terminal.data(),controlBytes.data(),controlBytes.size());
                    error=terminal[sampling_math::kDeviceGenerationControlErrorCode];
                    const bool rejected=failure;require(extent==(rejected?0:live),"Consensus replaced or failed to clip original query extent");require((error!=0)==rejected,"Native cohort error consensus is incorrect");
                    QwenFixedHeadNativeSendTotals raw;require(readQwenFixedHeadNativeSendCounters(terminal.data(),static_cast<int>(terminal.size()),raw),"Missing actual native terminal row");
                    const auto ownedIDs=static_cast<std::uint64_t>(rejected?0:live)*frozen.heads()*calls;
                    const auto contributed=ownedIDs*NativeFile::rowBytes;
                    require(raw==QwenFixedHeadNativeSendTotals{calls,ownedIDs,contributed,rejected?calls:0,8*calls*(degree-1),contributed*(degree-1)},
                        "Actual native ring sends/calls/owned rows/failures differ from successful source contribution");
                    QwenFixedHeadCounterTotals logical;require(projectQwenFixedHeadTerminalContributions(raw,degree,NativeFile::rowBytes,logical),"Raw native ring sends failed exact terminal projection");
                    require(logical==QwenFixedHeadCounterTotals{calls,ownedIDs,contributed,rejected?calls:0,8*calls,contributed},"Logical contribution metrics include link/receive/decode amplification");
                    auto expected=oldOutput;
                    if(!rejected&&live){std::vector<std::byte> nativeRows(static_cast<std::size_t>(live)*16*NativeFile::rowBytes);
                        for(std::size_t i=0;i<static_cast<std::size_t>(live)*16;++i)std::memcpy(nativeRows.data()+i*NativeFile::rowBytes,file.original.data()+originalIds[i]*NativeFile::rowBytes,NativeFile::rowBytes);
                        std::vector<float> decoded(static_cast<std::size_t>(live)*16*NativeFile::width);
                        decodeCPUImmutableLookupRows(TensorType::IQ4_NL,NativeFile::width,NativeFile::rowBytes,nativeRows.data(),static_cast<std::size_t>(live)*16,decoded.data());std::memcpy(expected.data(),decoded.data(),decoded.size()*sizeof(float));}
                    require(actual==expected,"Live original native decode or inactive output suffix changed");
                    if(rejected||!live){require(packed==oldPacket&&received==oldGathered,"Empty/error transport overwrote stale successful packet/receive data");}
                    else {
                        const auto used=static_cast<std::size_t>(live)*frozen.heads()*NativeFile::rowBytes;
                        require(std::equal(packed.begin()+used,packed.end(),oldPacket.begin()+used),"Pack touched inactive native suffix");
                        for(int peer=0;peer<degree;++peer){const auto headFirst=peer*frozen.heads();
                            for(int row=0;row<live;++row)for(int localHead=0;localHead<frozen.heads();++localHead){const auto global=originalIds[row*16+headFirst+localHead];const auto at=peer*packetBytes+(row*frozen.heads()+localHead)*NativeFile::rowBytes;
                                require(!std::memcmp(received.data()+at,file.original.data()+global*NativeFile::rowBytes,NativeFile::rowBytes),"Actual peer native bytes differ from original source");}
                            require(std::equal(received.begin()+peer*packetBytes+used,received.begin()+(peer+1)*packetBytes,oldGathered.begin()+peer*packetBytes+used),"Allgather touched unused rank-bank suffix");}
                    }
                    for(int peer=0;peer<degree;++peer){FixedHeadValidationHeader headerValue{};std::memcpy(&headerValue,headerBytes.data()+peer*sizeof(headerValue),sizeof(headerValue));
                        require(headerValue.originalLiveRows==((replay==13&&peer==degree-1)?capacity+1:(replay==14&&peer==degree-1)?0:live),"Actual peer header lost original query count");
                        const bool peerLocalError=peer==degree-1&&(replay==12||replay==13||replay==15);
                        require((headerValue.firstError!=0)==peerLocalError,"Real mandatory header exchange lost local failure metadata");
                        if(replay==15&&peer==degree-1)require(headerValue.firstError==37,"Pre-existing first error changed in native header");}
                    require(kernels->captureIdentity(args,stream)==identity&&nodes[1]->graphCaptureVariantSignature()==headerSignature&&nodes[4]->graphCaptureVariantSignature()==payloadSignature,"Replay changed complete component capture identity");
                    require(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation)==assignment.source.nativeBytes(),"Replay grew/changed source-owned native storage");
                    require(table->rangeView()==view&&arena.stats().total_buffers==allocatedBuffers&&arena.stats().total_bytes==allocatedBytes,
                        "Replay changed source generation/arena allocation inventory");
                    for(std::size_t index=0;index<roots.size();++index)require(roots[index]->gpu_data_ptr()==rootPointers[index],"Captured arena operand was rebound");
                    require(memory->claimedBytes(device,PhysicalMemoryOwner::ActivationArena,PhysicalMemoryMaterializationKind::NewAllocation)==arenaBytes(capacity,degree),"Replay changed admitted activation capacity");
                    rendezvous.arrive_and_wait();
                    require(!tp->isAbortRequested(),"Peer failed during native component observation");
                }
            }
            require(memory->claimedBytes(device,PhysicalMemoryOwner::ActivationArena,PhysicalMemoryMaterializationKind::NewAllocation)==0,"Arena outlived captured graphs");
            table.reset();require(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,PhysicalMemoryMaterializationKind::NewAllocation)==assignment.source.nativeBytes(),"Actual read token failed to retain native storage");
        });}catch(...){errors[participant]=std::current_exception();tp->requestAbort();if(!dropped)rendezvous.arrive_and_drop();}});
    for(auto &thread:threads)thread.join();
    for(const auto &owner:memory->rankAttestation())require(!owner.materialized_new_bytes&&!owner.committed_new_bytes,"Component retirement leaked a PMA allocation");
    if(injection==HostFailureInjection::DuringArenaUpload) {
        require(tp->isAbortRequested(),"Initial upload failure did not publish cancellation");
        std::size_t injected=0,guarded=0;
        for(int participant=0;participant<degree;++participant) {
            const auto &observation=observations[participant];
            require(observation.setupJoins==1,"Initial upload owner retired before its exact stream joined");
            require(!observation.instantiated&&!observation.validated&&!observation.launchAttempts&&
                !observation.terminalJoins&&!observation.retiredGraphs,"Setup failure reached captured-work lifecycle");
            require(bool(errors[participant]),"Initial upload failure left a successful unmatched participant");
            try{std::rethrow_exception(errors[participant]);}
            catch(const InjectedHostFailure &){require(participant==degree-1,"Wrong initial upload peer failed");++injected;}
            catch(const std::runtime_error &error){require(participant!=degree-1&&
                std::string_view(error.what())=="Peer failed during initial arena upload","Unrelated initial-upload failure");++guarded;}
        }
        require(injected==1&&guarded==static_cast<std::size_t>(degree-1),"Initial-upload failure did not stop every peer");
        return;
    }
    if(injection==HostFailureInjection::BeforeFirstRetainedLaunch) {
        require(tp->isAbortRequested(),"Injected host failure did not retire collective admission");
        std::size_t injected=0,guarded=0;
        for(int participant=0;participant<degree;++participant) {
            const auto &observation=observations[participant];
            require(observation.setupJoins==1,"Setup upload stream did not retire before captured owners");
            require(observation.instantiated==1&&observation.validated==1,"Host failure occurred before actual graph capture/validation");
            require(observation.launchAttempts==0,"Surviving worker reached retained native work after peer drop");
            require(observation.terminalJoins==1&&observation.retiredGraphs==1,"Host drop did not join/reset captured graph before owner retirement");
            require(bool(errors[participant]),"Dropped native cohort silently completed");
            try{std::rethrow_exception(errors[participant]);}
            catch(const InjectedHostFailure &){require(participant==degree-1,"Wrong participant injected a host failure");++injected;}
            catch(const std::runtime_error &error) {
                require(participant!=degree-1&&std::string_view(error.what())=="Peer failed before retained native component launch",
                    "Host-drop test encountered an unrelated failure instead of the exact prelaunch abort guard");++guarded;
            }
        }
        require(injected==1&&guarded==static_cast<std::size_t>(degree-1),"Host-drop did not reject every surviving participant");
        return;
    }
    for(const auto &error:errors)if(error)std::rethrow_exception(error);
}
} // namespace
/** @test CUDA2 original source ranges and real two-allgather captured component. */
TEST(ImmutableFixedHeadNativeCohort, CUDA2OriginalPackets){prove(DeviceId::cuda(0),2);}
/** @test CUDA4 original source ranges and real two-allgather captured component. */
TEST(ImmutableFixedHeadNativeCohort, CUDA4OriginalPackets){prove(DeviceId::cuda(0),4);}
/** @test CUDA8 original source ranges and real two-allgather captured component. */
TEST(ImmutableFixedHeadNativeCohort, CUDA8OriginalPackets){prove(DeviceId::cuda(0),8);}
/** @test ROCm2 original source ranges and real two-allgather captured component. */
TEST(ImmutableFixedHeadNativeCohort, ROCm2OriginalPackets){prove(DeviceId::rocm(0),2);}
/** @test ROCm4 original source ranges and real two-allgather captured component. */
TEST(ImmutableFixedHeadNativeCohort, ROCm4OriginalPackets){prove(DeviceId::rocm(0),4);}
/** @test ROCm8 original source ranges and real two-allgather captured component. */
TEST(ImmutableFixedHeadNativeCohort, ROCm8OriginalPackets){prove(DeviceId::rocm(0),8);}
/** @test CUDA2 early-upload and captured-prelaunch peer drops retire exact owners before any unmatched launch. */
TEST(ImmutableFixedHeadNativeCohort, CUDA2HostDropBeforeLaunch){
    prove(DeviceId::cuda(0),2,HostFailureInjection::DuringArenaUpload);if(::testing::Test::IsSkipped())return;
    for(int repetition=0;repetition<3;++repetition){SCOPED_TRACE(repetition);prove(DeviceId::cuda(0),2,HostFailureInjection::BeforeFirstRetainedLaunch);if(::testing::Test::IsSkipped())return;}}
/** @test ROCm2 early-upload and captured-prelaunch peer drops retire exact owners before any unmatched launch. */
TEST(ImmutableFixedHeadNativeCohort, ROCm2HostDropBeforeLaunch){
    prove(DeviceId::rocm(0),2,HostFailureInjection::DuringArenaUpload);if(::testing::Test::IsSkipped())return;
    for(int repetition=0;repetition<3;++repetition){SCOPED_TRACE(repetition);prove(DeviceId::rocm(0),2,HostFailureInjection::BeforeFirstRetainedLaunch);if(::testing::Test::IsSkipped())return;}}
