/**
 * @file RecordedHeadProjection.cpp
 * @brief Real GGUF output-weight projection of a qualified native FINAL_NORM row.
 *
 * This opt-in branch of the existing prepared-projection executable loads one
 * original row slice, prepares it through admitted production upload/store APIs,
 * and proves the actual captured LMHeadStage against its recorded oracle. The
 * plain GEMM adapter then proves byte equivalence before event timing. Restore
 * runs before the start event. Every raw compute-only sample and captured leaf
 * resource is retained. Local requests remain local_shape. Explicit cohorts
 * concurrently submit participant-local production compute graphs and report
 * actual_tp only for that partitioned compute boundary, never complete head
 * publication or whole inference. Original graph/source owners outlive replay.
 */
#include "RecordedHeadCohort.h"
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
#include "execution/diagnostics/PreparedGemmIsolationAdapter.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "backends/BackendManager.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/KernelFactory.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmWorkspaceContract.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedGemmWorkspaceContract.h"
#endif
#include "loaders/ModelLoader.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/GPUVramPreflight.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "tensors/Tensors.h"
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <span>
#include <string_view>
#include <set>
#include <sys/stat.h>
#include <thread>
#include <future>
#include <exception>
#include <optional>
#endif

namespace llaminar2::test::recorded_head
{
bool requested(int argc,char **argv) noexcept
{ return argc>1 && std::string_view(argv[1])=="--recorded-head-request"; }
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
namespace
{
using namespace diagnostics;
using Factory=llaminar::v2::kernels::KernelFactory;
/** @brief Reject native failure without changing backend, precision, input or geometry. */
void checked(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
/** @brief Authenticate complete immutable file metadata; no binary/model payload hash. */
void checkFile(const FileIdentity &f)
{
    struct stat s{};checked(::stat(f.path.c_str(),&s)==0 && S_ISREG(s.st_mode),"Recorded input/source is not a regular file");
    checked(std::uint64_t(s.st_dev)==f.device && std::uint64_t(s.st_ino)==f.inode && std::uint64_t(s.st_size)==f.bytes &&
        std::int64_t(s.st_mtim.tv_sec)*1000000000+s.st_mtim.tv_nsec==f.mtime_ns &&
        std::int64_t(s.st_ctim.tv_sec)*1000000000+s.st_ctim.tv_nsec==f.ctime_ns,
        "Recorded input/source file identity changed");
}
/** @brief Recheck all request-bound external assets before setup and after retirement. */
void checkFiles(const Request &r)
{
    checkFile(r.input);checkFile(r.logits);checkFile(r.snapshot_manifest);checkFile(r.observation_receipt);
    for(const auto &f:r.model_artifacts)checkFile(f);
}
/** @brief Read bounded metadata only; reject a payload-sized JSON/CSV request. */
std::string metadata(const std::string &path)
{
    std::ifstream f(path,std::ios::binary|std::ios::ate);checked(bool(f),"Recorded metadata unavailable");
    const auto n=f.tellg();checked(n>0 && n<=4*1024*1024,"Recorded metadata exceeds bounded input contract");
    std::string result(std::size_t(n),'\0');f.seekg(0);f.read(result.data(),std::streamsize(n));checked(bool(f),"Recorded metadata read failed");return result;
}
/** @brief Read only the selected exact FP32 row/slice during setup, preserving every bit. */
void readWords(const FileIdentity &f,std::size_t first,std::span<std::uint32_t> words)
{
    checked(first<=f.bytes/4 && words.size()<=f.bytes/4-first,"Recorded FP32 subrange exceeds original file");
    std::ifstream in(f.path,std::ios::binary);checked(bool(in),"Recorded FP32 file unavailable");
    in.seekg(std::streamoff(first*4));in.read(reinterpret_cast<char *>(words.data()),std::streamsize(words.size_bytes()));
    checked(bool(in),"Recorded FP32 subrange read failed");
}
/** @brief Poll one exact event with the standard bounded terminal retirement. */
void completed(IBackend &backend,DeviceId device,void *event)
{
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;)
    {
        bool ready=false;checked(backend.queryEvent(event,device.ordinal,&ready),"Recorded projection event query failed");
        if(ready)return;
        checked(std::chrono::steady_clock::now()<deadline,"Recorded projection exceeded30s retirement");
        std::this_thread::yield();
    }
}
/** @brief Timing events are native-owned; no profiler is initialized or attached. */
struct Events final
{
    IBackend &backend;DeviceId device;void *start=nullptr,*stop=nullptr;
    explicit Events(IBackend &b,DeviceId d):backend(b),device(d)
    {start=backend.createTimingEvent(device.ordinal);stop=backend.createTimingEvent(device.ordinal);
     if(!start||!stop){if(start)backend.destroyEvent(start,device.ordinal);if(stop)backend.destroyEvent(stop,device.ordinal);throw std::runtime_error("Recorded timing event creation failed");}}
    ~Events(){backend.destroyEvent(stop,device.ordinal);backend.destroyEvent(start,device.ordinal);}
};
/** @brief Query the existing workspace ABI without allocating or changing its policy. */
WorkspaceRequirements workspaceRequirements(DeviceId device,int n,int k)
{
#ifdef HAVE_CUDA
    if(device.is_cuda())return cuda::quantized_gemm_workspace::projectionRequirements(1,n,k,device.ordinal,
        cuda::quantized_gemm_workspace::NativeCodebooks{19,19});
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())return rocm::quantized_gemm_workspace::projectionRequirements(1,n,k);
#endif
    throw std::invalid_argument("Recorded head selected an uncompiled backend");
}
/** @brief Compare the complete named workspace ABI, not only its byte sum. */
bool sameWorkspace(const WorkspaceRequirements &a,const WorkspaceRequirements &b)
{
    if(a.buffers.size()!=b.buffers.size())return false;
    for(std::size_t i=0;i<a.buffers.size();++i)
    {
        const auto &x=a.buffers[i],&y=b.buffers[i];
        if(std::tie(x.name,x.size_bytes,x.alignment,x.required,x.regime,x.content_lifetime)!=
            std::tie(y.name,y.size_bytes,y.alignment,y.required,y.regime,y.content_lifetime))return false;
    }
    return true;
}
/** @brief Declare a single exact source weight using the allocator's canonical layout. */
void planWeight(LoadOrchestrator &pool,const Request &r,DeviceId device)
{
    pool.addDevice(device.ordinal);
    const auto &f=native_vnni_formats::Q8_0;
    pool.planWeightForOwner(device.ordinal,r.source_tensor,r.n,r.k,f.payload_bytes,
        canonicalDeviceVnniFormat(f).is_asymmetric,f.has_emins,std::size_t(r.n)*(r.k/32)*34,
        PhysicalMemoryOwner::PrimaryModelWeights);
}
/** @brief A typed single-node capture remains the original complete LM-head oracle. */
std::unique_ptr<IGPUGraphCapture> captureGraph(ComputeGraph &graph,DeviceGraphExecutor &executor,
    IDeviceContext &context,IWorkerGPUContext &worker,void *stream,const std::shared_ptr<PhysicalMemoryAuthority> &memory)
{
    checked(executor.prepareGraphStorageForCapture(graph,&context,stream,"recorded original LM head"),"Recorded head frontier preparation failed");
    auto result=worker.createGraphCapture(stream);checked(bool(result),"Recorded head graph creation failed");
    ScopedBackendGraphCapture capture(worker,*result,"recorded original LMHeadStage");
    checked(capture.begin(),"Recorded original head capture failed");
    checked(executor.recordSetupGraphMaterialization(graph,&context),"Recorded original head stage recording failed");capture.finish();
    checked(result->prepareRuntimeContextStorage(memory) && result->instantiate(),"Recorded original head instantiation failed");return result;
}
/** @brief Retain exact native nodes/resources; never call occupancy potential measured occupancy. */
nlohmann::json inventory(IGPUGraphCapture &graph,DeviceId device)
{
    std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
    checked(graph.inspectKernelNodes(nodes,&error) && !nodes.empty(),"Recorded compute native inventory unavailable");
    nlohmann::json out=nlohmann::json::array();bool projection=false;
    for(const auto &n:nodes)
    {
        checked(n.name_resolved && n.valid(),"Recorded compute kernel identity/resources incomplete");
        checked(n.name.find("nccl")==std::string::npos && n.name.find("capturedTransfer")==std::string::npos,
            "Recorded compute contains unrelated communication");
        projection|=device.is_cuda()
            ? n.name.find("nativeVnniGemv_")!=std::string::npos || n.name.find("nativeVnniGemvTensorCoreSmallM")!=std::string::npos
            : n.name.find("gemv_native_vnni_kernel")!=std::string::npos;
        out.push_back({{"symbol",n.name},{"function",n.function_identity},{"grid",{n.grid_x,n.grid_y,n.grid_z}},
            {"block",{n.block_x,n.block_y,n.block_z}},{"dynamic_shared_bytes",n.dynamic_shared_memory_bytes},
            {"static_shared_bytes",n.static_shared_memory_bytes},{"local_bytes_per_thread",n.local_memory_bytes_per_thread},
            {"registers_per_thread",n.registers_per_thread},{"potential_max_active_blocks_per_sm",n.max_active_blocks_per_sm}});
    }
    checked(projection,"Recorded compute did not retain a NativeVNNI projection");return out;
}
/** @brief One exact canonical declaration, before any participant tensor materializes. */
struct ParticipantDeclaration
{
    DeviceId device;
    IBackend *backend=nullptr;
    WorkspaceRequirements requirements;
    PhysicalMemoryResource gpu;
    std::size_t tensor_bytes=0,host_bytes=0,raw=0,graph_bytes=0,persistent_bytes=0,staging_bytes=0,host_staging_bytes=0;
};
/**
 * @brief Declare original allocator geometry on the actual device's own worker.
 * @param r Validated source/slice/policy request.
 * @param cohort Whether the retained family includes the extra timing parent.
 * @return Typed owners and observed device resource; no tensor is materialized.
 */
ParticipantDeclaration declareParticipant(const Request &r,bool cohort)
{
    ParticipantDeclaration d;
    d.device=r.backend=="rocm"?DeviceId::rocm(r.ordinal):DeviceId::cuda(r.ordinal);
    d.backend=getBackendFor(d.device);checked(d.backend,"Recorded head backend unavailable");
    checked(r.ordinal<d.backend->deviceCount(),"Recorded cohort names an unavailable physical device");
    d.requirements=workspaceRequirements(d.device,r.n,r.k);
    std::size_t scratch=0;for(const auto &b:d.requirements.buffers)scratch+=((b.size_bytes+3)/4)*4;
    d.tensor_bytes=2*std::size_t(r.k)*4+2*std::size_t(r.n)*4+scratch;
    d.host_bytes=d.tensor_bytes+(2*std::size_t(r.k)+3*std::size_t(r.n))*4;
    d.raw=std::size_t(r.n)*(r.k/32)*34;
    LoadOrchestrator declaration;planWeight(declaration,r,d.device);
    const auto staging=resolveGPUWeightLoadMemoryGeometry(d.raw,{.staging_stream_count=3,.staging_budget_bytes=0});
    d.staging_bytes=staging.staging_bytes;d.host_staging_bytes=staging.host_staging_bytes;
    d.graph_bytes=(cohort?4u:3u)*GPUGraphMemoryContract::reservationBytesPerExecutable(d.device);
    d.persistent_bytes=declaration.plannedPersistentBytes(d.device.ordinal,PhysicalMemoryOwner::PrimaryModelWeights);
    d.gpu={0,d.device,d.backend->deviceMemoryTotal(d.device.ordinal),d.backend->deviceMemoryFree(d.device.ordinal)};
    return d;
}
/**
 * @brief Add concurrent owners to the ONE canonical cohort ledger.
 * @param plan Canonical aggregate admission builder, never a second live ledger.
 * @param d One actual participant's source, storage and graph declaration.
 * @param cpu Same observed host allocator for all concurrently retained owners.
 */
void addDeclaration(PhysicalMemoryPlanBuilder &plan,const ParticipantDeclaration &d,const PhysicalMemoryResource &cpu)
{
    plan.add(d.gpu,PhysicalMemoryOwner::PrimaryModelWeights,d.persistent_bytes);
    plan.add(d.gpu,PhysicalMemoryOwner::WeightLoadStaging,d.staging_bytes);
    plan.add(cpu,PhysicalMemoryOwner::WeightLoadStaging,d.host_staging_bytes);
    plan.add(d.gpu,PhysicalMemoryOwner::ExecutionWorkspace,d.tensor_bytes+d.requirements.total_bytes_with_alignment());
    plan.add(cpu,PhysicalMemoryOwner::ExecutionWorkspace,d.host_bytes);
    plan.add(d.gpu,PhysicalMemoryOwner::NativeGraphExecutable,d.graph_bytes);
}
/** @brief Authenticate/seal one metadata owner reused by every original equal source slice. */
struct RecordedSource final
{
    ModelLoader loader;
    const GGUFTensorInfo *info=nullptr;
    /**
     * @brief Preserve original file/receipt/manifest/source checks before admission.
     * @param r Exact externally bound source observation and source artifact stats.
     * @throws std::runtime_error If source metadata or qualification has drifted.
     */
    explicit RecordedSource(const Request &r)
    {
        checked(std::endian::native==std::endian::little,"Recorded FP32 input requires little-endian native host");
        checkFiles(r);
        std::vector<std::string> lines;std::istringstream csv(metadata(r.snapshot_manifest.path));std::string line;
        while(std::getline(csv,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();lines.push_back(line);}
        validateSnapshotManifest(r,lines);
        validateObservationReceipt(r,nlohmann::json::parse(metadata(r.observation_receipt.path)));
        loader.setUseMmap(true);loader.setPayloadAccessPattern(ModelPayloadAccessPattern::SparseSelection);
        checked(loader.loadModel(r.model_artifacts.front().path),"Original GGUF metadata load failed");
        std::set<std::string> expected_paths,actual_paths;
        for(const auto &f:r.model_artifacts)expected_paths.insert(std::filesystem::canonical(f.path).string());
        for(const auto &p:loader.artifactPaths())actual_paths.insert(std::filesystem::canonical(p).string());
        checked(actual_paths==expected_paths,"Original GGUF shard closure differs from request");
        info=loader.tensorInfo(r.source_tensor);
        checked(info && info->dimensions==std::vector<std::uint64_t>{std::uint64_t(r.original_n),std::uint64_t(r.k)} &&
            info->type==GGUFTensorType::Q8_0,"Original GGUF head metadata differs from recorded source");
        loader.sealTensorSources();
    }
};
/**
 * @brief Read submission-skew metadata, never a cross-device event clock.
 * @return Monotonic host nanoseconds; no device is queried or synchronized.
 */
std::int64_t hostNow()
{return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
/**
 * @brief Compare complete node multisets while retaining original inventory records.
 * @param a Actual parent inventory copied solely for comparison.
 * @param b Original compute inventory including exact multiplicity and resources.
 * @return Whether composition retained every original compute kernel unchanged.
 */
bool sameKernels(nlohmann::json a,nlohmann::json b)
{
    const auto less=[](const auto &x,const auto &y){return x.dump()<y.dump();};
    std::sort(a.begin(),a.end(),less);std::sort(b.begin(),b.end(),less);return a==b;
}
/**
 * @brief Retained participant-local setup shared by local and concurrent diagnostics.
 *
 * Construction claims canonical physical owners before tensor/vector allocation;
 * prepare uses the original one-source LMHead/GEMM path. All asynchronous work
 * and final destruction stay on the original GPU worker. No owner is replaced
 * between histories, warmups and measured rounds.
 */
class PreparedParticipant final
{
public:
    /**
     * @brief Retain source/context/claims before any asynchronous tensor work.
     * @param request Exact immutable member of the validated request/cohort.
     * @param declaration Canonical storage/graph BOM for this actual participant.
     * @param owner Actual device worker that also owns final destruction.
     * @param authority The one shared admission authority for the entire cohort.
     * @note Preparation is a separate fallible call so its owner is complete
     *       before the first enqueue and can join every exceptional setup path.
     */
    PreparedParticipant(const Request &request,ParticipantDeclaration declaration,IWorkerGPUContext &owner,
        std::shared_ptr<PhysicalMemoryAuthority> authority)
        :r(request),d(std::move(declaration)),worker(owner),memory(std::move(authority)),
         host_claim(memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,d.host_bytes)),
         gpu_claim(memory->claimNewAllocation(d.device,PhysicalMemoryOwner::ExecutionWorkspace,d.tensor_bytes)),
         graph_claim(memory->claimNewAllocation(d.device,PhysicalMemoryOwner::NativeGraphExecutable,d.graph_bytes)),
         words(r.k),restored_input(r.k),expected(r.n),actual(r.n),recorded(r.n)
    {
        stream=worker.defaultStream();checked(reinterpret_cast<std::uintptr_t>(stream)>2,"Recorded head requires an exact nondefault stream");
    }
    /** @brief Join before every graph/stage/workspace/tensor/claim unwind, including failed setup. */
    ~PreparedParticipant()
    {
        if(!worker.ownsCurrentThread() || !d.backend->synchronizeStream(stream,d.device.ordinal))std::terminate();
        timed_parent.reset();compute.reset();restore.reset();oracle.reset();
    }
    /**
     * @brief Prepare the original source slice and production stage pair once.
     * @param observation Shared canonical source metadata owner, retained through retirement.
     * @param cohort Build an additional fresh timing parent only for cohort mode.
     * @throws std::runtime_error On source, admission, native capture or byte disagreement.
     * @note Original LMHead and adapter captures remain independent byte oracles;
     *       the parent composes their retained children without recapture or rebinding.
     */
    void prepare(RecordedSource &observation,bool cohort)
    {
        auto &loader=observation.loader;const auto device=d.device;auto *backend=d.backend;
        const auto &requirements=d.requirements;const auto raw=d.raw;
        context=IDeviceContext::create(device,1);events=std::make_unique<Events>(*backend,device);
        source=loader.loadTensorRowSlice(r.source_tensor,r.first,r.first+r.n,DeviceId::cpu(),WeightPrecision::NATIVE);
        checked(source && source->native_type()==TensorType::Q8_0 && source->rows()==std::size_t(r.n) && source->cols()==std::size_t(r.k),"Original native GGUF slice load failed");
        pool=std::make_shared<LoadOrchestrator>(backend,memory,PhysicalMemoryOwner::PrimaryModelWeights);planWeight(*pool,r,device);
        pool->allocate(raw,3);pool->addWeightJob(device.ordinal,{.name=r.source_tensor,.host_raw_data=source->raw_data(),
            .raw_bytes=raw,.format=RepackFormat::Q8_0,.N=r.n,.K=r.k,.is_asymmetric=false});
        pool->load();pool->finalize();
        auto engine=Factory::createGemmFromGPUWeightPool(*source,device,pool,r.source_tensor);
        // The canonical static ABI is a precursor BOM, not arithmetic authority.
        // Price/validate the prepared engine under the actual source policy
        // before allocating any workspace or recording any graph. A differing
        // requirement is an unsupported admission, never reserve inflation.
        {
            auto arithmetic=engine->beginOutputPartitionEquivalenceScope(r.n,r.serial_width?r.serial_width:r.n);
            auto *consumer=dynamic_cast<IWorkspaceConsumer *>(engine.get());
            checked(consumer && sameWorkspace(requirements,consumer->getWorkspaceRequirements(1,r.n,r.k)),
                "Prepared original-partition workspace differs from its canonical admitted precursor");
        }
        store=std::make_shared<PreparedWeightStore>(ModelContextId{r.model_context});store->installPhysicalMemoryAuthority(memory);
        ModelWeightSetBuilder bindings(InferenceStrategy{.model_id=store->modelId(),.tp_degree=r.degree,.devices={device}});
        WeightBinding binding;
        binding.identity=makeSourceWeightIdentity(r.source_tensor,store->modelId());
        binding.identity.derivation=r.degree==1?WeightDerivationKind::Source:WeightDerivationKind::RowSlice;
        binding.identity.tp_rank_or_device_index=r.participant;
        binding.slice={.source_rows=std::size_t(r.original_n),.source_cols=std::size_t(r.k),.row_start=std::size_t(r.first),
            .row_count=std::size_t(r.n),.col_start=0,.col_count=std::size_t(r.k)};
        binding.tensor_owner=source;binding.tensor=source.get();binding.residency.home_device=device;binding.residency.resident_device=device;
        bindings.addBinding(std::move(binding));binding=bindings.freezeBindings().at(0);
        auto handle=std::make_shared<Factory::PreparedGemmHandle>();handle->tensor=source.get();handle->device_id=device;
        handle->kind=device.is_cuda()?Factory::GemmPreparationKind::CUDA_INT8_PACKED:Factory::GemmPreparationKind::ROCM_INT8_PACKED;
        handle->variant=int(source->native_type());handle->prepared_weights=std::make_shared<Factory::PreparedGemmWeights>();
        handle->prepared_weights->kind=handle->kind;handle->prepared_weights->owned_kernel=std::shared_ptr<ITensorGemm>(std::move(engine));
        handle->prepared_weights->kernel=handle->prepared_weights->owned_kernel.get();
        ref=store->registerPreparedGemmHandle(binding,device.is_cuda()?PreparedWeightKind::CudaInt8PackedGemm:PreparedWeightKind::RocmInt8PackedGemm,device,std::move(handle));
        input=std::make_shared<FP32Tensor>(std::vector<std::size_t>{1,std::size_t(r.k)});
        baseline=std::make_shared<FP32Tensor>(std::vector<std::size_t>{1,std::size_t(r.k)});
        output=std::make_shared<FP32Tensor>(std::vector<std::size_t>{1,std::size_t(r.n)});
        original=std::make_shared<FP32Tensor>(std::vector<std::size_t>{1,std::size_t(r.n)});
        readWords(r.input,std::size_t(r.selected_row)*r.k,words);readWords(r.logits,r.first,recorded);
        std::memcpy(input->mutable_data(),words.data(),words.size()*4);std::memcpy(baseline->mutable_data(),words.data(),words.size()*4);
        for(auto *t:{input.get(),baseline.get(),output.get(),original.get()})TransferEngine::prepareDeviceInput(t,device,stream);
        workspace=std::make_shared<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment(),memory);
        checked(workspace->allocate(requirements),"Recorded projection exact workspace allocation failed");
        gemm=std::make_unique<GEMMStage>(GEMMStage::Params{.device_id=device,.A=input.get(),.B=source.get(),.C=output.get(),
            .m=1,.n=r.n,.k=r.k,.alpha=1,.beta=0,.transpose_B=true,.gemm_context=GemmContext::LM_HEAD,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,.prepared_ref=ref,.prepared_store=store.get()});
        gemm->setGPUStream(stream);gemm->bindWorkspace(workspace.get());
        lm=std::make_unique<LMHeadStage>(LMHeadStage::Params{.device_id=device,.hidden_states=input.get(),.lm_head_weight=source.get(),
            .logits=original.get(),.seq_len=1,.d_model=r.k,.vocab_size=r.n,.serial_equivalent_partition_width=r.serial_width,
            .use_prefill_replay_row_offset=false,.input_buffer_id=BufferId::NORMALIZED,.output_buffer_id=BufferId::LOGITS,
            .prepared_ref=ref,.prepared_store=store.get()});
        lm->setGPUStream(stream);lm->bindWorkspace(workspace.get());
        {
            auto arithmetic=store->gemmKernel(ref)->beginOutputPartitionEquivalenceScope(r.n,r.serial_width?r.serial_width:r.n);
            checked(sameWorkspace(gemm->getWorkspaceRequirements(1,r.n,r.k),requirements) &&
                sameWorkspace(lm->getWorkspaceRequirements(1,r.n,r.k),requirements),
                "Recorded stage original-policy workspace differs from admitted ABI");
        }
        checked(arena.registerExternalBuffer(BufferId::NORMALIZED,input.get()) &&
            arena.registerExternalBuffer(BufferId::GATE_PROJ,output.get()) && arena.registerExternalBuffer(BufferId::LOGITS,original.get()),"Recorded canonical arena binding failed");
        executor.setArena(&arena);
        std::vector<IsolationWorkspaceBaseline> scratch_baselines;
        for(const auto &b:requirements.buffers)
        {
            if(!b.size_bytes)continue;
            scratch_owners.push_back(std::make_shared<FP32Tensor>(std::vector<std::size_t>{(b.size_bytes+3)/4}));
            auto &t=scratch_owners.back();
            std::fill(t->mutable_data(),t->mutable_data()+t->size_bytes()/sizeof(float),0.f);
            TransferEngine::prepareDeviceInput(t.get(),device,stream);
            scratch_baselines.push_back({b.name,{t,0,b.size_bytes}});
        }
        oracle_graph.addNode("recorded_original_lm_head",std::move(lm),device);
        adapter_owner=std::make_unique<PreparedGemmIsolationAdapter>(std::move(gemm),
            std::array<std::shared_ptr<ITensor>,3>{input,source,output},store,workspace,arena,
            IsolationGemmShape{1,r.original_n,r.k,r.first,r.n,r.participant,r.degree},
            IsolationInputProvenance{IsolationInputOrigin::QualifiedInferenceObservation,r.observation,"final_norm",r.generation,baseline},
            IsolationResidentSlot{baseline,0,baseline->size_bytes()},std::move(scratch_baselines),
            IsolationOutputPartitionPolicy{.serial_columns=r.serial_width});
        auto &adapter=*adapter_owner;
        // Every named scratch region has an explicit immutable initial baseline.
        std::size_t scratch_index=0;
        for(const auto &b:requirements.buffers)if(b.size_bytes)
            checked(backend->deviceCopyAsync(workspace->getBuffer(b.name),scratch_owners.at(scratch_index++)->gpu_data_ptr(),b.size_bytes,device.ordinal,stream),"Recorded initial scratch publication failed");
        adapter.freezeScratchBaselines(*backend,stream);adapter.prepareRecording(executor,context.get(),stream);
        oracle=captureGraph(oracle_graph,executor,*context,worker,stream,memory);
        restore=worker.createGraphCapture(stream);compute=worker.createGraphCapture(stream);checked(restore&&compute,"Recorded graph owners missing");
        {ScopedBackendGraphCapture capture(worker,*restore,"recorded restore only");checked(capture.begin(),"Recorded restore capture failed");adapter.recordRestore(*backend,*restore,stream);capture.finish();}
        {ScopedBackendGraphCapture capture(worker,*compute,"recorded GEMM compute only");checked(capture.begin(),"Recorded compute capture failed");adapter.recordCompute(executor,context.get(),*compute,stream);capture.finish();}
        checked(restore->prepareRuntimeContextStorage(memory) && compute->prepareRuntimeContextStorage(memory) && restore->instantiate() && compute->instantiate(),"Recorded compute/restore instantiation failed");
        checkGraphMemory();
        kernels=inventory(*compute,device);oracle_kernels=inventory(*oracle,device);
        checked(kernels==oracle_kernels,"LMHeadStage/plain GEMM retained different native leaf families/resources");
        checked(restore->launch() && oracle->launch(),"Recorded original head launch failed");
        checked(backend->deviceToHostOnStream(expected.data(),original->gpu_data_ptr(),expected.size()*4,device.ordinal,stream) &&
            backend->recordEvent(events->stop,device.ordinal,stream),"Recorded original head observation failed");completed(*backend,device,events->stop);
        checked(expected==recorded,"Original captured LMHeadStage disagrees with qualified recorded LM_HEAD slice");
        for(int replay=0;replay<20;++replay)
        {poison(replay);adapter_owner->launchRestored(*restore,*compute,stream);observe();}
        if(cohort)
        {
            timed_parent=worker.createGraphCapture(stream);checked(bool(timed_parent),"Recorded cohort timing owner unavailable");
            const std::array<GPUOrderedTimelineStep,2> steps{{
                {.name="recorded_restore",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=restore.get()},
                {.name="recorded_projection",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=compute.get()}}};
            checked(timed_parent->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::PerStepEvents) &&
                timed_parent->prepareRuntimeContextStorage(memory) && timed_parent->instantiate(),"Recorded cohort in-graph timing unavailable");
            parent_kernels=inventory(*timed_parent,device);
            checked(sameKernels(parent_kernels,kernels),"Recorded timing parent changed the production compute kernels");
            checkGraphMemory();
            for(int replay=0;replay<20;++replay){poison(replay);launchTimed();(void)consumeTiming();observe();}
        }
    }
    /** @brief Validate the original immutable children before any timed-parent launch. */
    void validateReplay() const {adapter_owner->validateRecordedReplay(*restore,*compute,stream);}
    /** @brief Launch the one private immutable restore→compute timing parent. */
    void launchTimed()
    {validateReplay();checked(timed_parent && timed_parent->launch(),"Recorded cohort timing parent launch failed");}
    /**
     * @brief Consume exactly one completed native timing replay without blocking synchronization.
     * @return Compute/reset microseconds; only compute is the scaling interval.
     */
    std::pair<double,double> consumeTiming()
    {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;)
        {
            auto value=timed_parent->consumeOrderedTimelineTiming();
            if(value.state==GPUOrderedTimelineTimingState::Pending)
            {checked(std::chrono::steady_clock::now()<deadline,"Recorded cohort timing exceeded30s retirement");std::this_thread::yield();continue;}
            checked(value.state==GPUOrderedTimelineTimingState::Complete && value.samples.size()==2,
                "Recorded cohort timing is absent, failed, stale or incomplete");
            const auto &reset=value.samples[0],&projection=value.samples[1];
            checked(reset.name=="recorded_restore" && projection.name=="recorded_projection" &&
                reset.kind==GPUOrderedTimelineStepKind::CapturedFragment && projection.kind==GPUOrderedTimelineStepKind::CapturedFragment &&
                std::isfinite(reset.elapsed_ms) && reset.elapsed_ms>=0 && std::isfinite(projection.elapsed_ms) && projection.elapsed_ms>0,
                "Recorded cohort timing step identity or interval is invalid");
            return {projection.elapsed_ms*1000.0,reset.elapsed_ms*1000.0};
        }
    }
    /** @brief Observe bytes only after all cohort timing callbacks have completed. */
    void observe()
    {
        checked(d.backend->deviceToHostOnStream(actual.data(),output->gpu_data_ptr(),actual.size()*4,d.device.ordinal,stream) &&
            d.backend->deviceToHostOnStream(restored_input.data(),input->gpu_data_ptr(),restored_input.size()*4,d.device.ordinal,stream) &&
            d.backend->recordEvent(events->stop,d.device.ordinal,stream),"Recorded GEMM byte observation failed");completed(*d.backend,d.device,events->stop);
        checked(actual==expected && restored_input==words,"Prepared GEMM output/input differs bytewise from its original oracle");
    }
    /**
     * @brief Preserve the original local loop, host-event interval and raw failure log.
     * @return Every finite positive original local sample, with no cohort claim.
     */
    nlohmann::json measureLocal()
    {
        for(int w=0;w<r.warmup;++w)adapter_owner->launchRestored(*restore,*compute,stream);
        nlohmann::json samples=nlohmann::json::array();
        for(int sample=0;sample<r.samples;++sample)
        {
            checked(restore->launch(),"Recorded untimed restore launch failed");
            checked(d.backend->recordEvent(events->start,d.device.ordinal,stream) && compute->launch() &&
                d.backend->recordEvent(events->stop,d.device.ordinal,stream),"Recorded compute event submission failed");completed(*d.backend,d.device,events->stop);
            float ms=0;checked(d.backend->eventElapsedTimeMs(events->start,events->stop,d.device.ordinal,&ms) && std::isfinite(ms) && ms>0,"Recorded native elapsed time invalid");
            nlohmann::json raw_sample={{"sample",sample},{"rank",r.participant},{"latency_us",double(ms)*1000.0}};
            samples.push_back(raw_sample);std::cout<<"RECORDED_HEAD_SAMPLE "<<raw_sample.dump()<<'\n'<<std::flush;observe();
        }
        return samples;
    }
    /**
     * @brief Preserve source/prepared/native identities without a model-topology claim.
     * @param info Original loader-normalized source descriptor.
     * @param samples Original local samples; aggregate cohort samples live outside members.
     * @param cohort Label this retained participant as one member, not an independent TP result.
     * @return Complete byte/resource evidence after all original replays have passed.
     */
    nlohmann::json describe(const GGUFTensorInfo &info,nlohmann::json samples,bool cohort) const
    {
        DeviceNativeVNNIMatrixDesc prepared{};
        checked(store->gemmKernel(ref)->inspectPreparedNativeVNNIMatrixDesc(prepared) && prepared.valid() &&
            prepared.codebook_id==19 && prepared.source_identity_present && prepared.source_codebook_id==19,
            "Recorded head lost original/prepared Q8 source identity");
        nlohmann::json result={{"schema","llaminar.recorded_head_projection.result.v1"},{"scope",cohort?"cohort_participant":"local_shape"},{"request",r.document},
            {"status","PASS"},{"byte_oracle","captured_LMHeadStage_and_original_recorded_LM_HEAD"},{"poison_replays",20},
            {"source_descriptor_actual",{{"name",info.name},{"loader_normalized_n_k",info.dimensions},{"gguf_type",int(info.type)},
                {"artifact_split_index",info.split_idx},{"offset",info.offset},{"bytes",info.size_bytes}}},
            {"prepared_actual",{{"n",prepared.n},{"k",prepared.k},{"codebook",prepared.codebook_id},
                {"source_codebook",prepared.source_codebook_id},{"serial_equivalent_partition_width",adapter_owner->contract().outputPartitionPolicy().serial_columns},{"source_is_superblock",prepared.source_is_superblock},
                {"local_original_raw_bytes",d.raw},{"binding_id",ref.binding_id},{"model_context",store->modelId().value}}},
            {"samples",std::move(samples)},{"compute_kernels",kernels},{"original_head_kernels",oracle_kernels},
            {"restore_timed",false},{"host_submission_gaps_may_be_included",!cohort},
            {"source_qualification_scope","external_root_bound_focused_canonical_export"},
            {"contains_collective",false},{"actual_tp_claim",false}};
        if(cohort){result["timed_parent_kernels"]=parent_kernels;result["timed_parent_poison_replays"]=20;}
        return result;
    }
private:
    /** @brief Check the complete materialized graph family against its exact admitted owner. */
    void checkGraphMemory() const
    {
        const auto bytes=oracle->residentMemoryBytes()+restore->residentMemoryBytes()+compute->residentMemoryBytes()+
            (timed_parent?timed_parent->residentMemoryBytes():0);
        checked(GPUGraphMemoryContract::acceptsFamilyObservation(d.device,bytes,d.graph_bytes),"Recorded graph family exceeded original BOM");
    }
    /**
     * @brief Publish original poison data outside every measured compute interval.
     * @param replay Distinct bounded history ordinal for exact input/output restoration.
     */
    void poison(int replay)
    {
        std::fill(actual.begin(),actual.end(),0x7fc00000u+unsigned(replay));
        std::fill(restored_input.begin(),restored_input.end(),0x3f800000u+unsigned(replay));
        checked(d.backend->hostToDeviceOnStream(output->gpu_data_ptr(),actual.data(),actual.size()*4,d.device.ordinal,stream) &&
            d.backend->hostToDeviceOnStream(input->gpu_data_ptr(),restored_input.data(),restored_input.size()*4,d.device.ordinal,stream),"Recorded input/output poison upload failed");
    }
    const Request r;
    const ParticipantDeclaration d;
    IWorkerGPUContext &worker;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    PhysicalMemoryAllocationLease host_claim,gpu_claim,graph_claim;
    std::unique_ptr<IDeviceContext> context;
    void *stream=nullptr;
    std::shared_ptr<TensorBase> source;
    std::shared_ptr<LoadOrchestrator> pool;
    std::shared_ptr<PreparedWeightStore> store;
    PreparedWeightRef ref{};
    std::shared_ptr<FP32Tensor> input,baseline,output,original;
    std::vector<std::uint32_t> words,restored_input,expected,actual,recorded;
    std::vector<std::shared_ptr<FP32Tensor>> scratch_owners;
    std::shared_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<GEMMStage> gemm;
    std::unique_ptr<LMHeadStage> lm;
    BufferArena arena;
    DeviceGraphExecutor executor;
    ComputeGraph oracle_graph;
    std::unique_ptr<Events> events;
    std::unique_ptr<PreparedGemmIsolationAdapter> adapter_owner;
    std::unique_ptr<IGPUGraphCapture> oracle,restore,compute,timed_parent;
    nlohmann::json kernels,oracle_kernels,parent_kernels;
};
/** @brief Retire every retained participant on its actual worker before common source/PMA destruction. */
struct ParticipantRetirement final
{
    std::vector<IWorkerGPUContext *> &workers;
    std::vector<std::unique_ptr<PreparedParticipant>> &states;
    /** @brief Complete cleanup explicitly before final source-file revalidation; idempotent. */
    void retire() noexcept
    {
        for(std::size_t rank=0;rank<states.size();++rank)if(states[rank])
        {
            try{workers[rank]->submitAndWait([&,rank]{states[rank].reset();});}catch(...){std::terminate();}
        }
    }
    /** @brief Preserve every admitted owner on all exceptional paths until original-stream retirement. */
    ~ParticipantRetirement(){retire();}
};
/**
 * @brief Drain every queued task even when a participant or controller failed.
 * @param tasks Actual worker futures whose referenced state must remain alive.
 * @param error Retain the first error for rethrow only after every future is joined.
 */
void joinAll(std::vector<std::future<void>> &tasks,std::exception_ptr &error)
{
    for(auto &task:tasks)try{task.get();}catch(...){if(!error)error=std::current_exception();}
}
/**
 * @brief Reuse participant setup for one local or one closed concurrent cohort.
 * @param source Original local request or exact immutable cohort template.
 * @param cohort Validated concurrent membership; null preserves original local mode.
 * @return Passed source/byte/resource/sample evidence, after all graphs retire.
 * @note One host CPU backend/admission budget is initialized before any cohort
 *       materialization. No cross-device operation is hidden in a compute stage.
 */
nlohmann::json measure(const Request &source,const CohortRequest *cohort)
{
    RecordedSource observation(source);
    const std::vector<Request> requests=cohort?cohort->members():std::vector<Request>{source};
    std::vector<IWorkerGPUContext *> workers;std::vector<ParticipantDeclaration> declarations;
    // GPU access is lazy; the standalone CPU PMA owner is deliberately not.
    // Match canonical local startup once for the aggregate host-memory budget,
    // before participant setup or the shared CPU allocator observation.
    const auto first_device=source.backend=="rocm"?DeviceId::rocm(source.ordinal):DeviceId::cuda(source.ordinal);
    auto *first_backend=getBackendFor(first_device);
    checked(first_backend && first_device.ordinal<first_backend->deviceCount(),"Recorded head selected GPU backend/device unavailable");
    initCPUBackend(-1);
    auto *host=getBackendFor(DeviceId::cpu());checked(host,"Recorded head host-memory backend unavailable");
    for(const auto &r:requests)
    {
        const auto device=r.backend=="rocm"?DeviceId::rocm(r.ordinal):DeviceId::cuda(r.ordinal);
        auto *backend=getBackendFor(device);
        checked(backend && r.ordinal<backend->deviceCount(),"Recorded cohort backend/device is unavailable");
        auto *worker=&GPUDeviceContextPool::instance().getContext(device);
        checked(worker->deviceOrdinal()==r.ordinal && std::find(workers.begin(),workers.end(),worker)==workers.end(),
            "Recorded cohort worker does not uniquely own its physical member");
        workers.push_back(worker);
        workers.back()->submitAndWait([&]{declarations.push_back(declareParticipant(r,cohort!=nullptr));});
    }
    // Observe one shared CPU allocator AFTER every worker context is resident.
    // Reusing this exact observation makes concurrent host BOM sums canonical.
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),host->deviceMemoryTotal(0),host->deviceMemoryFree(0)};
    PhysicalMemoryPlanBuilder plan;
    for(const auto &declaration:declarations)addDeclaration(plan,declaration,cpu);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
    std::vector<std::unique_ptr<PreparedParticipant>> states(requests.size());
    ParticipantRetirement retirement{workers,states};
    for(std::size_t rank=0;rank<requests.size();++rank)
        workers[rank]->submitAndWait([&,rank]
        {
            // If setup throws before publication, this local owner joins on
            // this same worker before any retained field can unwind.
            auto state=std::make_unique<PreparedParticipant>(requests[rank],declarations[rank],*workers[rank],memory);
            state->prepare(observation,cohort!=nullptr);states[rank]=std::move(state);
        });
    nlohmann::json result;
    if(!cohort)
        workers.front()->submitAndWait([&]{result=states.front()->describe(*observation.info,states.front()->measureLocal(),false);});
    else
    {
        nlohmann::json samples=nlohmann::json::array(),rounds=nlohmann::json::array();
        const auto workload_identity=cohortWorkloadIdentity(source);const auto workload=workload_identity.dump();
        for(int iteration=0;iteration<source.warmup+source.samples;++iteration)
        {
            const bool measured=iteration>=source.warmup;const int round=measured?iteration-source.warmup:iteration;
            CohortRoundGate gate(source.degree);
            std::vector<nlohmann::json> raw(requests.size());std::vector<CohortRankInterval> intervals(requests.size());
            std::vector<std::future<void>> tasks;tasks.reserve(requests.size());std::exception_ptr error;
            const auto ready_ns=hostNow();
            try
            {
                for(std::size_t rank=0;rank<requests.size();++rank)
                {
                    auto task=workers[rank]->submitAsync([&,rank]
                    {
                        try
                        {
                            states[rank]->validateReplay();const auto admitted_ns=hostNow();gate.arriveAndWait(int(rank));
                            const auto submit_ns=hostNow();states[rank]->launchTimed();
                            const auto [us,reset_us]=states[rank]->consumeTiming();
                            intervals[rank]={round,int(rank),us};
                            raw[rank]={{"sample",round},{"rank",rank},{"device_ordinal",requests[rank].ordinal},{"degree",source.degree},
                                {"scope","actual_tp"},{"disposition","partitioned"},{"operation_boundary","partitioned_projection_compute"},
                                {"operation","recorded_native_q8_output_projection"},{"workload",workload},{"latency_us",us},{"restore_interval_us_excluded",reset_us},{"admitted_host_ns",admitted_ns},
                                {"submit_host_ns",submit_ns},{"complete_host_ns",hostNow()}};
                        }
                        catch(...){gate.abort();throw;}
                    });
                    requireQueuedCohortAdmission(task);tasks.push_back(std::move(task));
                }
                gate.releaseWhenReady();
            }
            catch(...){error=std::current_exception();gate.abort();}
            joinAll(tasks,error);
            // Completed values survive a later participant/byte failure in the
            // owned native journal. Never interleave partial worker JSON writes.
            for(const auto &value:raw)if(!value.is_null())
                std::cout<<"RECORDED_HEAD_COHORT_SAMPLE "<<nlohmann::json{{"measured",measured},{"interval",value}}.dump()<<'\n'<<std::flush;
            if(error)std::rethrow_exception(error);
            const double maximum=cohortMaximum(intervals,source.degree,round);
            tasks.clear();
            try{for(std::size_t rank=0;rank<requests.size();++rank)tasks.push_back(workers[rank]->submitAsync([&,rank]{states[rank]->observe();}));}
            catch(...){error=std::current_exception();}
            joinAll(tasks,error);if(error)std::rethrow_exception(error);
            if(measured)
            {
                for(auto &value:raw)samples.push_back(std::move(value));
                rounds.push_back({{"sample",round},{"maximum_rank_latency_us",maximum},{"host_round_begin_ns",ready_ns},{"release_host_ns",gate.releaseHostNanoseconds()},{"host_round_retired_ns",hostNow()}});
            }
        }
        nlohmann::json participants=nlohmann::json::array();
        for(std::size_t rank=0;rank<states.size();++rank)
            workers[rank]->submitAndWait([&,rank]{participants.push_back(states[rank]->describe(*observation.info,nlohmann::json::array(),true));});
        result={{"schema","llaminar.recorded_head_cohort.result.v1"},{"status","PASS"},{"scope","actual_tp"},
            {"operation_boundary","partitioned_projection_compute"},{"complete_head_or_inference_claim",false},{"contains_collective",false},
            {"actual_tp_claim",true},{"request",cohort->document()},{"workload_identity",workload_identity},{"participants",std::move(participants)},{"samples",std::move(samples)},
            {"rounds",std::move(rounds)},{"reduction","max_rank_per_round_before_median"},{"restore_timed",false},
            {"timing_interval","native_in_graph_events_around_production_compute_fragment"},{"host_submission_gaps_may_be_included",false},{"runtime_graph_scheduling_gaps_may_be_included",true},
            {"pure_kernel_duration_claim",false},{"hardware_overlap_claim",false},{"source_qualification_scope","external_root_bound_focused_canonical_export"},
            {"original_full_logit_columns_byte_proved",source.original_n},{"poison_replays_per_participant",20},{"timed_parent_poison_replays_per_participant",20}};
    }
    retirement.retire();checkFiles(source);return result;
}
} // namespace
#endif
int run(int argc,char **argv)
{
    require(argc==3 && requested(argc,argv),"Use --recorded-head-request <absolute request.json> only");
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    const auto document=nlohmann::json::parse(metadata(argv[2]));
    std::optional<CohortRequest> cohort;
    if(document.at("schema")=="llaminar.recorded_head_cohort.v1")cohort.emplace(CohortRequest::fromJson(document));
    const auto request=cohort?cohort->source():parse(document);
    require(!std::filesystem::exists(request.output),"Recorded output must be fresh");
    const auto temporary=request.output+".tmp";require(!std::filesystem::exists(temporary),"Interrupted recorded output already exists");
    require(std::filesystem::is_directory(std::filesystem::path(request.output).parent_path()),"Recorded output parent directory must already exist");
    std::cout<<"RECORDED_HEAD_IDENTITY "<<document.dump()<<'\n'<<std::flush;
    const auto result=measure(request,cohort?&*cohort:nullptr);
    std::ofstream output(temporary,std::ios::binary);output<<result.dump(2)<<'\n';output.close();checked(bool(output),"Recorded result publication failed");
    std::filesystem::rename(temporary,request.output);return 0;
#else
    throw std::invalid_argument("Recorded prepared head requires a compiled GPU backend");
#endif
}
} // namespace llaminar2::test::recorded_head
