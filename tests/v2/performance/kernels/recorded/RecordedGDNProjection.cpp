/**
 * @file RecordedGDNProjection.cpp
 * @brief Captured complete GDN projection compute on qualified original native inputs.
 *
 * Source slicing uses WeightManager's production GDN loader and head assignment.
 * Half controls use the production original-source materializer and operand
 * policy. The real GDNProjectionStage selects its original fused subgroups.
 * The shared recorded runtime owns PMA admission, native inventory and complete
 * concurrent rounds. Every output byte on every participant is checked against
 * the original full native snapshot in canonical packed order before timing.
 */
#include "RecordedGDNRequest.h"
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
#include "RecordedProjectionRuntime.h"
#include "RecordedGDNSourceGeometry.h"
#include "execution/diagnostics/PreparedGDNIsolationAdapter.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "kernels/KernelFactory.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmWorkspaceContract.h"
#include "kernels/rocm/gemm/ROCmFloatingPointGemmWorkspaceContract.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedGemmWorkspaceContract.h"
#include "kernels/cuda/gemm/CUDAFloatingPointGemmWorkspaceContract.h"
#endif
#include "loaders/ModelLoader.h"
#include "loaders/WeightManager.h"
#include "loaders/PreparedHalfWeights.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "models/qwen35/Qwen35Schema.h"
#include "tensors/Tensors.h"
#endif

namespace llaminar2::test::recorded_gdn
{
bool requested(int argc,char **argv) noexcept
{return argc>1 && std::string_view(argv[1])=="--recorded-gdn-request";}
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
namespace
{
using namespace diagnostics;
using namespace recorded_runtime;
using Factory=llaminar::v2::kernels::KernelFactory;
/** @brief Recheck exactly the complete original observation/source asset closure. */
void checkGDNFiles(const Request &r)
{
    checkFile(r.input);checkFile(r.snapshot_manifest);checkFile(r.observation_receipt);
    for(const auto &s:r.sources)checkFile(s.oracle);
    for(const auto &f:r.model_artifacts)checkFile(f);
}
/** @brief Resolve actual local widths solely by concatenating canonical ownership spans. */
std::array<int,4> localWidths(const Request &r)
{
    const auto spans=sourceSpans(r);std::array<int,4> widths{};
    for(std::size_t i=0;i<4;++i)for(const auto span:spans[i])widths[i]+=span.count;
    return widths;
}
/** @brief Declare the exact canonical pool layout for both native and half-prepared siblings. */
void planGDNWeights(LoadOrchestrator &pool,const Request &r,DeviceId device)
{
    pool.addDevice(device.ordinal);const auto widths=localWidths(r);const auto &format=native_vnni_formats::Q8_0;
    for(std::size_t i=0;i<4;++i)
    {
        if(i<2)pool.planWeightForOwner(device.ordinal,r.sources[i].name,widths[i],r.k,format.payload_bytes,
            canonicalDeviceVnniFormat(format).is_asymmetric,format.has_emins,std::size_t(widths[i])*(r.k/32)*34,
            PhysicalMemoryOwner::PrimaryModelWeights);
        else pool.planRawWeightForOwner(device.ordinal,r.sources[i].name,widths[i],r.k,std::size_t(widths[i])*r.k*2,
            PhysicalMemoryOwner::PrimaryModelWeights);
    }
}
/** @brief Compose the real four-engine workspace ABI before any weight/tensor allocation. */
WorkspaceRequirements gdnWorkspace(DeviceId device,const Request &r)
{
    const auto widths=localWidths(r);WorkspaceRequirements requirements;
    for(std::size_t i=0;i<4;++i)
    {
#ifdef HAVE_ROCM
        if(device.is_rocm())
        {
            requirements.merge(i<2?rocm::quantized_gemm_workspace::projectionRequirements(r.rows,widths[i],r.k):
                rocm::floating_gemm_workspace::projectionRequirements(r.rows,r.k,FloatingActivationOperandPrecision::FP16RNE));
            continue;
        }
#endif
#ifdef HAVE_CUDA
        if(device.is_cuda())
        {
            requirements.merge(i<2?cuda::quantized_gemm_workspace::projectionRequirements(r.rows,widths[i],r.k,device.ordinal,
                cuda::quantized_gemm_workspace::NativeCodebooks{19,19}):
                cuda::floating_gemm_workspace::projectionRequirements(r.rows,widths[i],r.k,FloatingActivationOperandPrecision::FP16RNE));
            continue;
        }
#endif
        throw std::invalid_argument("Recorded GDN selected an uncompiled backend");
    }
#ifdef HAVE_ROCM
    if(device.is_rocm())rocm::quantized_gemm_workspace::appendFusedProjectionRequirements(requirements,r.rows,widths);
#endif
#ifdef HAVE_CUDA
    if(device.is_cuda())cuda::quantized_gemm_workspace::appendFusedProjectionRequirements(requirements,r.rows,widths,r.k);
#endif
    addCudaConcurrentDecodeGemvSideStreamWorkspace(requirements,device,r.rows,4);
    return requirements;
}
/** @brief Bind an immutable original manifest/receipt and the real normalized GGUF header closure. */
struct GDNSource final
{
    ModelLoader loader;
    std::unique_ptr<const Qwen4ExpConfig> profile;
    std::array<const GGUFTensorInfo *,4> info{};
    /** @brief Authenticate source metadata without selecting or initializing a backend. */
    explicit GDNSource(const Request &r)
    {
        checked(std::endian::native==std::endian::little,"Recorded GDN requires little-endian FP32 snapshots");checkGDNFiles(r);
        std::vector<std::string> lines;std::istringstream csv(metadata(r.snapshot_manifest.path));std::string line;
        while(std::getline(csv,line)){if(!line.empty() && line.back()=='\r')line.pop_back();lines.push_back(line);}
        validateSnapshotManifest(r,lines);validateObservationReceipt(r,nlohmann::json::parse(metadata(r.observation_receipt.path)));
        loader.setUseMmap(true);loader.setPayloadAccessPattern(ModelPayloadAccessPattern::SparseSelection);
        checked(loader.loadModel(r.model_artifacts.front().path),"Recorded original GDN GGUF metadata unavailable");
        std::set<std::string> expected,actual;
        for(const auto &f:r.model_artifacts)expected.insert(std::filesystem::canonical(f.path).string());
        for(const auto &f:loader.artifactPaths())actual.insert(std::filesystem::canonical(f).string());
        checked(expected==actual,"Recorded GDN original GGUF shard closure differs");
        const auto arch=loader.architecture();checked(arch=="qwen4exp","Recorded GDN first bundle adapter requires its original QFN schema");
        checked(loader.ggufDirectory(),"Recorded GDN original GGUF directory is unavailable");
        profile=std::make_unique<const Qwen4ExpConfig>(*loader.ggufDirectory());
        (void)originalSourceDimensions(profile->sparseAttention(),profile->deltaNet());
        checked(profile->mixers().at(0)==Qwen4ExpMixer::GatedDeltaNet,"Recorded GDN source layer zero is not a GDN block");
        checked(loader.getInt(arch+".ssm.group_count",0)==r.key_heads &&
            loader.getInt(arch+".ssm.time_step_rank",0)==r.value_heads &&
            loader.getInt(arch+".ssm.state_size",0)==r.state && loader.embeddingLength()==std::uint64_t(r.k) &&
            loader.getInt(arch+".ssm.inner_size",0)==r.value_heads*r.state,"Recorded GDN geometry differs from original GGUF metadata");
        for(std::size_t i=0;i<4;++i)
        {
            info[i]=loader.tensorInfo(r.sources[i].name);
            checked(info[i] && info[i]->dimensions==std::vector<std::uint64_t>{std::uint64_t(r.sources[i].original_n),std::uint64_t(r.k)} &&
                int(info[i]->type)==r.sources[i].gguf_type,"Recorded GDN source metadata differs from the original native matrix");
        }
        loader.sealTensorSources();
    }
};
/** @brief Declare only the actual complete bundle's source, baseline, scratch and graph owners. */
ParticipantDeclaration declareGDN(const Request &r,bool cohort)
{
    ParticipantDeclaration d;d.device=r.backend=="rocm"?DeviceId::rocm(r.ordinal):DeviceId::cuda(r.ordinal);
    d.backend=getBackendFor(d.device);checked(d.backend && r.ordinal<d.backend->deviceCount(),"Recorded GDN device unavailable");
    d.requirements=gdnWorkspace(d.device,r);const auto widths=localWidths(r);
    std::size_t scratch=0,output_words=0,source_bytes=0,half_bytes=0,concat_peak=0;
    for(const auto &b:d.requirements.buffers)scratch+=((b.size_bytes+3)/4)*4;
    for(std::size_t i=0;i<4;++i)
    {
        output_words+=std::size_t(r.rows)*widths[i];
        const auto raw=std::size_t(widths[i])*(i<2?std::size_t(r.k/32)*34:std::size_t(r.k)*4);
        source_bytes+=raw;concat_peak=std::max(concat_peak,raw);
        const auto prepared=i<2?raw:std::size_t(widths[i])*r.k*2;
        d.raw=std::max(d.raw,prepared);if(i>=2)half_bytes+=prepared;
    }
    const auto input_words=std::size_t(r.rows)*r.k;
    d.tensor_bytes=(2*input_words+output_words)*4+scratch;
    // CPU tensor mirrors, immutable recorded/observed word banks, retained packed
    // sources and half controls coexist. One source-sized concat bank is transient
    // in the production loader; it is named here rather than an anonymous reserve.
    d.host_bytes=d.tensor_bytes+(2*input_words+2*output_words)*4+source_bytes+half_bytes+concat_peak;
    LoadOrchestrator declaration;planGDNWeights(declaration,r,d.device);
    d.persistent_bytes=declaration.plannedPersistentBytes(d.device.ordinal,PhysicalMemoryOwner::PrimaryModelWeights);
    const auto staging=resolveGPUWeightLoadMemoryGeometry(d.raw,{.staging_stream_count=3,.staging_budget_bytes=0});
    d.staging_bytes=staging.staging_bytes;d.host_staging_bytes=staging.host_staging_bytes;
    d.graph_bytes=(cohort?3u:2u)*GPUGraphMemoryContract::reservationBytesPerExecutable(d.device);
    d.gpu={0,d.device,d.backend->deviceMemoryTotal(d.device.ordinal),d.backend->deviceMemoryFree(d.device.ordinal)};
    return d;
}
/** @brief One retained original source/prepared stage and immutable captured replay family. */
class GDNParticipant final
{
public:
    /** @brief Claim canonical physical owners before constructing the participant's buffers. */
    GDNParticipant(const Request &request,ParticipantDeclaration declaration,IWorkerGPUContext &owner,
        std::shared_ptr<PhysicalMemoryAuthority> authority)
        :r(request),d(std::move(declaration)),worker(owner),memory(std::move(authority)),
         host_claim(memory->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,d.host_bytes)),
         gpu_claim(memory->claimNewAllocation(d.device,PhysicalMemoryOwner::ExecutionWorkspace,d.tensor_bytes)),
         graph_claim(memory->claimNewAllocation(d.device,PhysicalMemoryOwner::NativeGraphExecutable,d.graph_bytes)),widths(localWidths(r)),
         words(std::size_t(r.rows)*r.k),restored_input(words.size())
    {
        stream=worker.defaultStream();checked(reinterpret_cast<std::uintptr_t>(stream)>2,"Recorded GDN requires its exact nondefault worker stream");
        for(std::size_t i=0;i<4;++i){recorded[i].resize(std::size_t(r.rows)*widths[i]);actual[i].resize(recorded[i].size());}
    }
    /** @brief Join original native work before releasing any retained prepared/reset/graph owner. */
    ~GDNParticipant()
    {
        if(!worker.ownsCurrentThread() || !d.backend->synchronizeStream(stream,d.device.ordinal))std::terminate();
        timed_parent.reset();compute.reset();restore.reset();adapter.reset();
    }
    /** @brief Load canonical GDN shards, prepare original arithmetic and prove the captured full bundle. */
    void prepare(GDNSource &observation,bool cohort)
    {
        auto &loader=observation.loader;const auto device=d.device;
        context=IDeviceContext::create(device,1);events=std::make_unique<Events>(*d.backend,device);
        // TP1 authenticates fused-QKV through the ordinary model slicer, so
        // retain the canonical profile's complete original attention AND GDN
        // axes. Equal source assignments use the actual original query heads;
        // their GDN intervals remain owned by GDNHeadAssignment::fromPartition.
        WeightManagerConfig config;config.sharding=Qwen35SchemaFactory{}.getWeightShardingConfig();
        config.dimensions=originalSourceDimensions(observation.profile->sparseAttention(),observation.profile->deltaNet());
        const auto assignments=originalSourceAssignments(config.dimensions,r.degree);
        config.tp_config=std::make_shared<TensorParallelConfig>(assignments);
        source_loader=std::make_unique<WeightManager>(loader);source_loader->configure(config);
        source_loader->installPhysicalMemoryAuthority(memory);
        for(std::size_t i=0;i<4;++i)
        {
            source[i]=source_loader->getShardedWeightForAssignment(r.sources[i].name,DeviceId::cpu(),assignments.at(r.participant),0);
            checked(source[i] && source[i]->rows()==std::size_t(widths[i]) && source[i]->cols()==std::size_t(r.k) &&
                source[i]->native_type()==(i<2?TensorType::Q8_0:TensorType::FP32),"Canonical GDN source slicer returned another source geometry/type");
            prepared_source[i]=i<2?source[i]:materializePreparedHalfWeights(*source[i]);
        }
        pool=std::make_shared<LoadOrchestrator>(d.backend,memory,PhysicalMemoryOwner::PrimaryModelWeights);
        planGDNWeights(*pool,r,device);pool->allocate(d.raw,3);
        for(std::size_t i=0;i<4;++i)pool->addWeightJob(device.ordinal,{.name=r.sources[i].name,
            .host_raw_data=prepared_source[i]->raw_data(),.raw_bytes=prepared_source[i]->size_bytes(),
            .format=i<2?RepackFormat::Q8_0:RepackFormat::RAW_FP,.N=widths[i],.K=r.k,.is_asymmetric=false});
        pool->load();pool->finalize();
        store=std::make_shared<PreparedWeightStore>(ModelContextId{r.model_context});store->installPhysicalMemoryAuthority(memory);
        ModelWeightSetBuilder bindings(InferenceStrategy{.model_id=store->modelId(),.tp_degree=r.degree,.devices={device}});
        for(std::size_t i=0;i<4;++i)
        {
            const auto role=inferWeightRole(r.sources[i].name);
            const auto operand=PreparedWeightRepresentationContract::operandPrecision(role,r.sources[i].name,loader.architecture(),GDNProjectionArithmetic::FP16MoEAlphaBeta);
            checked(operand==(i<2?FloatingActivationOperandPrecision::Native:FloatingActivationOperandPrecision::FP16RNE),"Recorded GDN source operand policy differs from production");
            auto engine=Factory::createGemmFromGPUWeightPool(*prepared_source[i],device,pool,r.sources[i].name,0,GPUPreparedWeightPoolLayout::SourceNative,operand);
            WeightBinding binding;binding.identity=makeSourceWeightIdentity(r.sources[i].name,store->modelId());
            binding.identity.derivation=r.degree==1?WeightDerivationKind::Source:WeightDerivationKind::FusedSubblockConcat;
            binding.identity.tp_rank_or_device_index=r.participant;binding.floating_operand_precision=operand;
            binding.slice={.source_rows=std::size_t(r.sources[i].original_n),.source_cols=std::size_t(r.k),.row_count=std::size_t(widths[i]),
                .col_count=std::size_t(r.k),.inner_is_presliced=r.degree>1};
            binding.tensor_owner=prepared_source[i];binding.tensor=prepared_source[i].get();
            binding.residency.home_device=device;binding.residency.resident_device=device;bindings.addBinding(std::move(binding));
            engines[i]=std::move(engine);
        }
        const auto frozen=bindings.freezeBindings();
        for(std::size_t i=0;i<4;++i)
        {
            auto handle=std::make_shared<Factory::PreparedGemmHandle>();handle->tensor=prepared_source[i].get();handle->device_id=device;
            handle->kind=i<2?(device.is_cuda()?Factory::GemmPreparationKind::CUDA_INT8_PACKED:Factory::GemmPreparationKind::ROCM_INT8_PACKED):Factory::GemmPreparationKind::FLOATING_POINT;
            handle->variant=int(prepared_source[i]->native_type());handle->prepared_weights=std::make_shared<Factory::PreparedGemmWeights>();
            handle->prepared_weights->kind=handle->kind;handle->prepared_weights->owned_kernel=std::shared_ptr<ITensorGemm>(std::move(engines[i]));
            handle->prepared_weights->kernel=handle->prepared_weights->owned_kernel.get();
            refs[i]=store->registerPreparedGemmHandle(frozen[i],device.is_cuda()?PreparedWeightKind::CudaInt8PackedGemm:PreparedWeightKind::RocmInt8PackedGemm,device,std::move(handle));
        }
        input=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(r.rows),std::size_t(r.k)});
        baseline=std::make_shared<FP32Tensor>(input->shape());readWords(r.input,0,words);
        std::memcpy(input->mutable_data(),words.data(),words.size()*4);std::memcpy(baseline->mutable_data(),words.data(),words.size()*4);
        TransferEngine::prepareDeviceInput(input.get(),device,stream);TransferEngine::prepareDeviceInput(baseline.get(),device,stream);
        checked(arena.registerExternalBuffer(BufferId::NORMALIZED,input.get()),"Recorded GDN input arena binding failed");
        const auto spans=sourceSpans(r);
        for(std::size_t i=0;i<4;++i)
        {
            outputs[i]=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(r.rows),std::size_t(widths[i])});
            TransferEngine::prepareDeviceInput(outputs[i].get(),device,stream);
            checked(arena.registerExternalBuffer(ids[i],outputs[i].get()),"Recorded GDN output arena binding failed");
            for(int row=0;row<r.rows;++row)
            {
                std::size_t local=std::size_t(row)*widths[i];
                for(const auto span:spans[i])
                {
                    readWords(r.sources[i].oracle,std::size_t(row)*r.sources[i].original_n+span.start,
                        std::span<std::uint32_t>(recorded[i]).subspan(local,span.count));local+=span.count;
                }
            }
        }
        auto stage=std::make_unique<GDNProjectionStage>(GDNProjectionStage::Params{.device_id=device,.input=input.get(),.m=r.rows,.k=r.k,
            .w_qkv=prepared_source[0].get(),.output_qkv=outputs[0].get(),.n_qkv=widths[0],
            .w_z=prepared_source[1].get(),.output_z=outputs[1].get(),.n_z=widths[1],
            .w_a=prepared_source[2].get(),.output_a=outputs[2].get(),.n_a=widths[2],
            .w_b=prepared_source[3].get(),.output_b=outputs[3].get(),.n_b=widths[3],
            .input_buffer_id=BufferId::NORMALIZED,.output_qkv_buffer_id=ids[0],.output_z_buffer_id=ids[1],.output_a_buffer_id=ids[2],.output_b_buffer_id=ids[3],
            .prepared_ref_qkv=refs[0],.prepared_ref_z=refs[1],.prepared_ref_a=refs[2],.prepared_ref_b=refs[3],.prepared_store=store.get()});
        stage->setGPUStream(stream);
        checked(sameWorkspace(stage->getWorkspaceRequirements(r.rows,0,r.k),d.requirements),"Actual GDN stage workspace differs from admitted production ABI");
        workspace=std::make_shared<DeviceWorkspaceManager>(device,d.requirements.total_bytes_with_alignment(),memory);
        checked(workspace->allocate(d.requirements),"Recorded GDN exact workspace allocation failed");stage->bindWorkspace(workspace.get());
        std::vector<IsolationWorkspaceBaseline> scratch;
        for(const auto &b:d.requirements.buffers)if(b.size_bytes)
        {
            auto t=std::make_shared<FP32Tensor>(std::vector<std::size_t>{(b.size_bytes+3)/4});std::fill(t->mutable_data(),t->mutable_data()+t->numel(),0.f);
            TransferEngine::prepareDeviceInput(t.get(),device,stream);
            checked(d.backend->deviceCopyAsync(workspace->getBuffer(b.name),t->gpu_data_ptr(),b.size_bytes,device.ordinal,stream),"Recorded GDN initial scratch publication failed");
            scratch.push_back({b.name,{t,0,b.size_bytes}});scratch_owners.push_back(std::move(t));
        }
        adapter=std::make_unique<PreparedGDNIsolationAdapter>(std::move(stage),input,
            std::array<std::shared_ptr<ITensor>,4>{outputs[0],outputs[1],outputs[2],outputs[3]},
            std::array<std::shared_ptr<ITensor>,4>{prepared_source[0],prepared_source[1],prepared_source[2],prepared_source[3]},store,workspace,arena,
            IsolationInputProvenance{IsolationInputOrigin::QualifiedInferenceObservation,r.observation,"layer0_attn_norm",r.generation,baseline},
            IsolationResidentSlot{baseline,0,baseline->size_bytes()},std::move(scratch));
        executor.setArena(&arena);adapter->freezeScratchBaselines(*d.backend,stream);adapter->prepareRecording(executor,context.get(),stream);
        restore=worker.createGraphCapture(stream);compute=worker.createGraphCapture(stream);checked(restore && compute,"Recorded GDN graph owners unavailable");
        {ScopedBackendGraphCapture capture(worker,*restore,"recorded GDN restore");checked(capture.begin(),"Recorded GDN restore capture failed");adapter->recordRestore(*d.backend,*restore,stream);capture.finish();}
        {ScopedBackendGraphCapture capture(worker,*compute,"recorded original GDN bundle");checked(capture.begin(),"Recorded GDN compute capture failed");adapter->recordCompute(executor,context.get(),*compute,stream);capture.finish();}
        checked(restore->prepareRuntimeContextStorage(memory) && compute->prepareRuntimeContextStorage(memory) && restore->instantiate() && compute->instantiate(),"Recorded GDN graph instantiation failed");
        kernels=inventory(*compute,device,InventoryPolicy::PreparedStageBundle);checkGraphMemory();
        std::cout<<"RECORDED_GDN_CAPTURE_IDENTITY "<<nlohmann::json{{"rank",r.participant},{"degree",r.degree},
            {"physical_rows",r.rows},{"local_output_columns",widths},{"compute_kernels",kernels}}.dump()<<'\n'<<std::flush;
        for(int replay=0;replay<20;++replay){poison(replay);adapter->launchRestored(*restore,*compute,stream);observe();}
        if(cohort)
        {
            timed_parent=worker.createGraphCapture(stream);checked(bool(timed_parent),"Recorded GDN timing parent unavailable");
            const std::array<GPUOrderedTimelineStep,2> steps{{
                {.name="recorded_restore",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=restore.get()},
                {.name="recorded_projection",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=compute.get()}}};
            checked(timed_parent->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::PerStepEvents) &&
                timed_parent->prepareRuntimeContextStorage(memory) && timed_parent->instantiate(),"Recorded GDN native timing parent failed");
            parent_kernels=inventory(*timed_parent,device,InventoryPolicy::PreparedStageBundle);checked(sameKernels(parent_kernels,kernels),"Recorded GDN timing changed original compute leaves");checkGraphMemory();
            for(int replay=0;replay<20;++replay){poison(replay);launchTimed();(void)consumeTiming();observe();}
        }
    }
    /** @brief Validate original complete replay family without submitting native work. */
    void validateReplay() const{adapter->validateRecordedReplay(*restore,*compute,stream);}
    /** @brief Submit the exact retained restore/compute native event transaction. */
    void launchTimed(){validateReplay();checked(timed_parent && timed_parent->launch(),"Recorded GDN native timing replay failed");}
    /** @return Native compute/reset intervals from the common original timeline consumer. */
    std::pair<double,double> consumeTiming(){return recorded_runtime::consumeTiming(*timed_parent);}
    /** @brief Byte-check every sibling and the restored complete original input after retirement. */
    void observe()
    {
        for(std::size_t i=0;i<4;++i)checked(d.backend->deviceToHostOnStream(actual[i].data(),outputs[i]->gpu_data_ptr(),actual[i].size()*4,d.device.ordinal,stream),"Recorded GDN output observation failed");
        checked(d.backend->deviceToHostOnStream(restored_input.data(),input->gpu_data_ptr(),restored_input.size()*4,d.device.ordinal,stream) &&
            d.backend->recordEvent(events->stop,d.device.ordinal,stream),"Recorded GDN input observation failed");completed(*d.backend,d.device,events->stop);
        for(std::size_t i=0;i<4;++i)if(actual[i]!=recorded[i])
        {
            const auto first=std::size_t(std::mismatch(actual[i].begin(),actual[i].end(),recorded[i].begin()).first-actual[i].begin());
            throw std::runtime_error("Captured original GDN bundle byte mismatch: "+r.sources[i].key+
                " rank="+std::to_string(r.participant)+" degree="+std::to_string(r.degree)+
                " row="+std::to_string(first/std::size_t(widths[i]))+" local_column="+std::to_string(first%std::size_t(widths[i]))+
                " actual_word="+std::to_string(actual[i][first])+" recorded_word="+std::to_string(recorded[i][first]));
        }
        checked(restored_input==words,"Recorded GDN original physical input was not restored bytewise");
    }
    /** @brief Preserve explicit local-shape timing without claiming concurrent TP. */
    nlohmann::json measureLocal()
    {
        for(int w=0;w<r.warmup;++w)adapter->launchRestored(*restore,*compute,stream);
        nlohmann::json samples=nlohmann::json::array();
        for(int sample=0;sample<r.samples;++sample)
        {
            validateReplay();checked(restore->launch() && d.backend->recordEvent(events->start,d.device.ordinal,stream) && compute->launch() &&
                d.backend->recordEvent(events->stop,d.device.ordinal,stream),"Recorded GDN local event submission failed");completed(*d.backend,d.device,events->stop);
            float ms=0;checked(d.backend->eventElapsedTimeMs(events->start,events->stop,d.device.ordinal,&ms) && std::isfinite(ms) && ms>0,"Recorded GDN local interval invalid");
            nlohmann::json raw={{"sample",sample},{"rank",r.participant},{"latency_us",double(ms)*1000.0}};
            samples.push_back(raw);std::cout<<"RECORDED_GDN_SAMPLE "<<raw.dump()<<'\n'<<std::flush;observe();
        }
        return samples;
    }
    /** @brief Describe original source/slice/kernel identities after mandatory byte proof. */
    nlohmann::json describe(const GDNSource &source_observation,nlohmann::json samples,bool cohort) const
    {
        nlohmann::json sources=nlohmann::json::array();const auto spans=sourceSpans(r);
        for(std::size_t i=0;i<4;++i)
        {
            nlohmann::json intervals=nlohmann::json::array();for(const auto span:spans[i])intervals.push_back({{"first",span.start},{"count",span.count}});
            nlohmann::json prepared;auto *engine=store->gemmKernel(refs[i]);
            if(i<2)
            {
                DeviceNativeVNNIMatrixDesc descriptor{};
                checked(engine && engine->inspectPreparedNativeVNNIMatrixDesc(descriptor) && descriptor.valid() &&
                    descriptor.source_identity_present && descriptor.source_codebook_id==19 && descriptor.codebook_id==19,
                    "Recorded GDN native source identity changed before publication");
                prepared=nativePreparedSourceMetadata(descriptor);
            }
            else
            {
                ContiguousFloatingPointWeightDescriptor descriptor{};
                checked(engine && engine->exportContiguousFloatingPointWeights(descriptor) && descriptor.valid() &&
                    descriptor.type==TensorType::FP16 && engine->floatingActivationOperandPrecision()==FloatingActivationOperandPrecision::FP16RNE,
                    "Recorded GDN floating source/operand identity changed before publication");
                prepared={{"n",descriptor.n},{"k",descriptor.k},{"bytes",descriptor.bytes},{"format","FP16"},{"activation_operand","half_rne"}};
            }
            const auto &info=*source_observation.info[i];sources.push_back({{"name",info.name},{"loader_normalized_n_k",info.dimensions},
                {"gguf_type",int(info.type)},{"artifact_split_index",info.split_idx},{"offset",info.offset},{"bytes",info.size_bytes},
                {"local_n",widths[i]},{"source_row_spans",intervals},{"prepared_format",i<2?"NativeVNNI_Q8_0":"FP16"},
                {"activation_operand",i<2?"single_q8":"half_rne"},{"binding_id",refs[i].binding_id},{"prepared_actual",prepared}});
        }
        return {{"schema","llaminar.recorded_gdn_projection.result.v1"},{"scope",cohort?"cohort_participant":"local_shape"},
            {"status","PASS"},{"request",r.document},{"byte_oracle","original_recorded_complete_native_GDN_bundle_outputs"},
            {"all_four_outputs_byte_proved",true},{"original_input_byte_proved",true},{"poison_replays",20},
            {"timed_parent_poison_replays",cohort?20:0},{"sources_actual",sources},{"compute_kernels",kernels},
            {"timed_parent_kernels",parent_kernels},{"samples",std::move(samples)},{"contains_collective",false},{"actual_tp_claim",false},
            {"restore_timed",false},{"host_submission_gaps_may_be_included",!cohort},
            {"source_qualification_scope","external_root_bound_focused_canonical_export"}};
    }
private:
    /** @brief Reject graph storage outside the exact original admitted family BOM. */
    void checkGraphMemory() const
    {checked(GPUGraphMemoryContract::acceptsFamilyObservation(d.device,restore->residentMemoryBytes()+compute->residentMemoryBytes()+
        (timed_parent?timed_parent->residentMemoryBytes():0),d.graph_bytes),"Recorded GDN graph family exceeds its exact admitted BOM");}
    /** @brief Exercise distinct dirty input/output histories outside every timed interval. */
    void poison(int replay)
    {
        std::fill(restored_input.begin(),restored_input.end(),0x3f800000u+unsigned(replay));
        checked(d.backend->hostToDeviceOnStream(input->gpu_data_ptr(),restored_input.data(),restored_input.size()*4,d.device.ordinal,stream),"Recorded GDN input poison failed");
        for(std::size_t i=0;i<4;++i)
        {std::fill(actual[i].begin(),actual[i].end(),0x7fc00000u+unsigned(replay));checked(d.backend->hostToDeviceOnStream(outputs[i]->gpu_data_ptr(),actual[i].data(),actual[i].size()*4,d.device.ordinal,stream),"Recorded GDN output poison failed");}
    }
    const Request r;const ParticipantDeclaration d;IWorkerGPUContext &worker;std::shared_ptr<PhysicalMemoryAuthority> memory;
    PhysicalMemoryAllocationLease host_claim,gpu_claim,graph_claim;
    const std::array<int,4> widths;
    static constexpr std::array<BufferId,4> ids={BufferId::GDN_QKV,BufferId::GDN_Z,BufferId::GDN_ALPHA,BufferId::GDN_BETA};
    void *stream=nullptr;std::unique_ptr<IDeviceContext> context;
    std::unique_ptr<WeightManager> source_loader;
    std::array<std::shared_ptr<TensorBase>,4> source,prepared_source;
    std::array<std::unique_ptr<ITensorGemm>,4> engines;
    std::array<PreparedWeightRef,4> refs;
    std::shared_ptr<LoadOrchestrator> pool;std::shared_ptr<PreparedWeightStore> store;
    std::shared_ptr<FP32Tensor> input,baseline;std::array<std::shared_ptr<FP32Tensor>,4> outputs;
    std::vector<std::uint32_t> words,restored_input;std::array<std::vector<std::uint32_t>,4> recorded,actual;
    std::vector<std::shared_ptr<FP32Tensor>> scratch_owners;
    std::shared_ptr<DeviceWorkspaceManager> workspace;BufferArena arena;DeviceGraphExecutor executor;
    std::unique_ptr<Events> events;std::unique_ptr<PreparedGDNIsolationAdapter> adapter;
    std::unique_ptr<IGPUGraphCapture> restore,compute,timed_parent;
    // An unused local timing parent has an empty native inventory, never JSON null.
    nlohmann::json kernels=nlohmann::json::array(),parent_kernels=nlohmann::json::array();
};
/** @brief Bind the complete GDN bundle to the same cohort scheduler and native observation helpers as head. */
struct GDNOperation
{
    using Request=recorded_gdn::Request;using Cohort=recorded_gdn::CohortRequest;using Source=GDNSource;using Participant=GDNParticipant;
    static constexpr const char *boundary="gdn_projection_bundle_compute";
    static constexpr const char *operation="recorded_native_gdn_projection_bundle";
    static constexpr const char *sample_prefix="RECORDED_GDN_COHORT_SAMPLE ";
    /** @brief Declare canonical source/prepared/workspace geometry for one participant. */
    static ParticipantDeclaration declare(const Request &r,bool cohort){return declareGDN(r,cohort);}
    /** @brief Revalidate source metadata after all native owners retire. */
    static void checkFiles(const Request &r){checkGDNFiles(r);}
    /** @brief Remove only varying physical cohort membership from the original work identity. */
    static nlohmann::json workloadIdentity(const Request &r)
    {auto identity=r.document;for(const char *key:{"output","device_ordinal","slice_degree","slice_participant"})identity.erase(key);return identity;}
    /** @brief Publish the actual complete participant oracle/source proof. */
    static nlohmann::json describe(const Participant &p,const Source &s,nlohmann::json samples,bool cohort){return p.describe(s,std::move(samples),cohort);}
    /** @brief Scope the concurrent result to compute only, preserving every raw participant interval. */
    static nlohmann::json describeCohort(const Cohort &cohort,const nlohmann::json &identity,nlohmann::json participants,nlohmann::json samples,nlohmann::json rounds)
    {
        return {{"schema","llaminar.recorded_gdn_cohort.result.v1"},{"status","PASS"},{"scope","actual_tp"},
            {"operation_boundary",boundary},{"complete_inference_claim",false},{"contains_collective",false},{"actual_tp_claim",true},
            {"request",cohort.document()},{"workload_identity",identity},{"participants",std::move(participants)},
            {"samples",std::move(samples)},{"rounds",std::move(rounds)},{"reduction","max_rank_per_round_before_median"},
            {"restore_timed",false},{"timing_interval","native_in_graph_events_around_production_compute_fragment"},
            {"host_submission_gaps_may_be_included",false},{"runtime_graph_scheduling_gaps_may_be_included",true},
            {"pure_kernel_duration_claim",false},{"hardware_overlap_claim",false},
            {"all_original_output_columns_byte_proved",true},{"poison_replays_per_participant",20},{"timed_parent_poison_replays_per_participant",20}};
    }
};
} // namespace
#endif
int run(int argc,char **argv)
{
    require(argc==3 && requested(argc,argv),"Use --recorded-gdn-request <absolute request.json> only");
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    const auto document=nlohmann::json::parse(recorded_runtime::metadata(argv[2]));std::optional<CohortRequest> cohort;
    if(document.at("schema")=="llaminar.recorded_gdn_cohort.v1")cohort.emplace(CohortRequest::fromJson(document));
    const auto request=cohort?cohort->source():parse(document);
    require(!std::filesystem::exists(request.output) && !std::filesystem::exists(request.output+".tmp"),"Recorded GDN output must be fresh");
    require(std::filesystem::is_directory(std::filesystem::path(request.output).parent_path()),"Recorded GDN output parent is unavailable");
    std::cout<<"RECORDED_GDN_IDENTITY "<<document.dump()<<'\n'<<std::flush;
    const auto result=recorded_runtime::measureOperation<GDNOperation>(request,cohort?&*cohort:nullptr);
    const auto temporary=request.output+".tmp";std::ofstream output(temporary,std::ios::binary);output<<result.dump(2)<<'\n';output.close();
    recorded_runtime::checked(bool(output),"Recorded GDN result publication failed");std::filesystem::rename(temporary,request.output);return 0;
#else
    throw std::invalid_argument("Recorded GDN projection requires a compiled GPU backend");
#endif
}
} // namespace llaminar2::test::recorded_gdn
