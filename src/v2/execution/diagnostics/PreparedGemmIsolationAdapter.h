/**
 * @file PreparedGemmIsolationAdapter.h
 * @brief Real prepared GEMMStage recording with separate resident baseline/restore ownership.
 *
 * This explicit diagnostic owns a one-node ComputeGraph and calls the ordinary
 * setup-materialization executor. It does not suppress stages in generation.
 * Callers admit all tensors/workspace with their existing PMA and retain both
 * reset and compute native graphs until terminal stream retirement. No hidden
 * input/routing/stateful GEMM form is accepted by this initial adapter.
 */
#pragma once
#include "CapturedComputeIsolation.h"
#include "../compute_stages/stages/GEMMStage.h"
#include "../local_execution/graph/DeviceGraphExecutor.h"
#include "../local_execution/graph/GraphCaptureGuard.h"
#include "../../backends/IBackend.h"
#include "../../loaders/PreparedWeightStore.h"
#include "../../tensors/ITensor.h"
#include "../../transfer/TransferEngine.h"
#include "../../tensors/NativeVnniFormatInfo.h"
#include <tuple>
#include <map>

namespace llaminar2::diagnostics
{
/** @brief One byte-exact GPU slot in a retained already-admitted tensor/manifest arena. */
struct IsolationResidentSlot
{
    std::shared_ptr<ITensor> tensor;
    std::size_t offset=0, bytes=0;
};
/** @brief Actual workspace name plus pre-execution baseline in an admitted resident slot. */
struct IsolationWorkspaceBaseline
{
    std::string name;
    IsolationResidentSlot baseline;
};
/**
 * @brief Reject hidden or protocol-bearing GEMM forms before looking up weights/storage.
 * @param params Actual production GEMM Params, not a parallel operation description.
 * @return Whether this initial adapter implements every read and execution side effect.
 * @note beta*C is absent from the current GEMM StageBufferContract, so beta0 is mandatory.
 */
[[nodiscard]] inline bool plainGemmIsolationFormSupported(const GEMMStage::Params &params) noexcept
{
    return params.alpha==1.f && params.beta==0.f && !params.mpi_ctx && !params.needs_allreduce &&
        !params.bias && !params.bias_tensor && !params.bias_required && !params.gate_input && !params.do_swiglu &&
        !params.verifier_row_range && !params.request_rows && !params.projection_row_work && params.output_range.empty();
}

/**
 * @brief Initial runnable adapter: static plain alpha1/beta0 prepared NativeVNNI dense GEMM only.
 *
 * The concrete Params and actual StageBufferContract jointly define closure.
 * Routes, live counts, recurrent state, raw bias and fused operations require
 * separate proven adapters and are rejected here, including nonnull MPI context.
 * Prepared source/store/engine and named workspace are retained and checked;
 * raw B GPU storage is never substituted for a prepared kernel payload.
 */
class PreparedGemmIsolationAdapter final
{
public:
    /**
     * @brief Freeze a real prepared local stage after its exact workspace has been admitted/bound.
     * @param stage Actual production stage, transferred into a fresh single-node graph.
     * @param tensors Exact retained A,B,C owners corresponding to the stage Params.
     * @param store Actual prepared binding owner, not an alternate engine registry.
     * @param workspace Actual already-bound workspace; no allocation occurs here.
     * @param arena Existing canonical binding authority for A/C.
     * @param shape Original/local work; its local axes must equal the real Params.
     * @param provenance External pre-execution inference observation or explicit synthetic fixture.
     * @param input_baseline Owned resident pre-execution A slot, optionally from an upstream snapshot arena.
     * @param scratch_baselines Exactly one pre-execution slot per declared nonempty workspace buffer.
     * @param partition Immutable original output-partition arithmetic (zero identity or positive serial width).
     * @throws std::invalid_argument For unsupported forms or incomplete/aliased ownership.
     */
    PreparedGemmIsolationAdapter(std::unique_ptr<GEMMStage> stage,
        std::array<std::shared_ptr<ITensor>,3> tensors, std::shared_ptr<PreparedWeightStore> store,
        std::shared_ptr<DeviceWorkspaceManager> workspace, BufferArena &arena,
        IsolationGemmShape shape, IsolationInputProvenance provenance,
        IsolationResidentSlot input_baseline, std::vector<IsolationWorkspaceBaseline> scratch_baselines,
        IsolationOutputPartitionPolicy partition={})
        : tensors_(std::move(tensors)), store_(std::move(store)), workspace_(std::move(workspace)),
          input_baseline_(std::move(input_baseline)), scratch_baselines_(std::move(scratch_baselines)),
          arena_(&arena), stage_(stage.get()),partition_(partition)
    {
        if(isGraphCaptureActive() || !stage_ || !store_ || !workspace_)
            throw std::invalid_argument("GEMM isolation requires cold actual prepared owners");
        const auto &p=stage_->getParams(); device_=p.device_id;
        validateIsolationOutputPartitionPolicy(shape,partition_);
        if((!device_.is_cuda() && !device_.is_rocm()) || device_.ordinal<0 || stage_->isCollectiveStage())
            throw std::invalid_argument("Isolation supports only a participant-local CUDA/ROCm plain GEMM");
        if(!p.A || !p.B || !p.C || tensors_[0].get()!=p.A || tensors_[1].get()!=p.B || tensors_[2].get()!=p.C ||
            p.prepared_store!=store_.get() || !p.prepared_ref || p.prepared_ref->device!=device_ ||
            !store_->contains(*p.prepared_ref) || !plainGemmIsolationFormSupported(p) ||
            !p.a_buffer_id || !p.c_buffer_id || *p.a_buffer_id==*p.c_buffer_id ||
            p.A->native_type()!=TensorType::FP32 || p.C->native_type()!=TensorType::FP32 ||
            shape.rows!=p.m || shape.local_columns!=p.n || shape.reduction_columns!=p.k ||
            arena.getTensor(*p.a_buffer_id)!=p.A || arena.getTensor(*p.c_buffer_id)!=p.C ||
            workspace_->device()!=device_ || !stage_->hasWorkspace() || stage_->getWorkspace()!=workspace_.get())
            throw std::invalid_argument("Unsupported or unbound plain prepared GEMM isolation input/state closure");
        const auto binding=store_->binding(*p.prepared_ref);
        if(!binding || !binding->immutable || binding->tensor!=p.B)
            throw std::invalid_argument("GEMM isolation requires the exact immutable prepared source binding");
        std::string error;
        if(!stage_->validatePreparedWeights(&error)) throw std::invalid_argument(error);
        engine_=store_->gemmKernel(*p.prepared_ref);
        if(!engine_ || dynamic_cast<ITensorGemm *>(stage_->getKernelAsWorkspaceConsumer())!=engine_)
            throw std::invalid_argument("GEMM isolation must resolve its actual engine before capture");
        if(!engine_->inspectPreparedNativeVNNIMatrixDesc(weight_) || !weight_.valid() ||
            !weight_.source_identity_present || weight_.n!=p.n || weight_.k!=p.k || p.k%32)
            throw std::invalid_argument("Initial isolation adapter requires exact prepared NativeVNNI metadata");
        std::optional<NativeVnniFormatInfo> physical;
        for(const auto &source:native_vnni_formats::kAllSourceFormats)
            if(source.metadata->codebook_id==weight_.source_codebook_id &&
                source.metadata->is_superblock==bool(weight_.source_is_superblock))
                for(const auto format:{canonicalDeviceVnniFormat(*source.metadata),migrationStableDeviceVnniFormat(*source.metadata)})
                    if(format.codebook_id==weight_.codebook_id)
                        // Physical storage view only; original source identity remains in weight_.
                        physical=NativeVnniFormatInfo{.codebook_id=format.codebook_id,
                            .payload_bytes=format.payload_bytes_per_block,.is_asymmetric=format.is_asymmetric,
                            .is_superblock=false,.has_emins=format.has_emins,
                            .max_abs_factor=source.metadata->max_abs_factor};
        if(!physical) throw std::invalid_argument("Initial isolation adapter cannot authenticate prepared format regions");
        const auto sizes=nativeVnniPackedRegionSizes(std::size_t(p.n),std::size_t(p.k),*physical);
        weights_={{{reinterpret_cast<std::uintptr_t>(weight_.payload),sizes.payload_bytes},
                  {reinterpret_cast<std::uintptr_t>(weight_.scales),sizes.scales_bytes},
                  {reinterpret_cast<std::uintptr_t>(weight_.mins),sizes.mins_bytes},
                  {reinterpret_cast<std::uintptr_t>(weight_.emins),sizes.emins_bytes}}};
        const auto contract=stage_->bufferContract();
        if(contract.inputs.size()!=1 || contract.outputs.size()!=1 || !contract.inouts.empty() ||
            !contract.weight_tensors.empty() || contract.prepared_weights.size()!=1 ||
            contract.inputs[0].id!=*p.a_buffer_id || contract.outputs[0].id!=*p.c_buffer_id ||
            contract.prepared_weights[0].source_tensor!=p.B || contract.prepared_weights[0].store!=store_.get())
            throw std::invalid_argument("GEMM isolation found an undeclared or extra stage input");
        input_bytes_=matrixBytes(p.m,p.k); output_bytes_=matrixBytes(p.m,p.n);
        const auto a=resident(tensors_[0],0,input_bytes_),c=resident(tensors_[2],0,output_bytes_);
        const auto baseline=slot(input_baseline_);
        if(provenance.owner.get()!=input_baseline_.tensor.get())
            throw std::invalid_argument("Isolation input provenance must retain the actual baseline tensor owner");
        if(input_baseline_.bytes!=input_bytes_)
            throw std::invalid_argument("GEMM isolation A baseline must cover its exact physical consumed rows");
        contract_.emplace(device_,shape,std::move(provenance),stage_->type(),contract,std::vector<IsolationBuffer>{
            {*p.a_buffer_id,a,baseline,input_bytes_,tensors_[0],input_baseline_.tensor},
            {*p.c_buffer_id,c,0,output_bytes_,tensors_[2],{}}},partition_);
        resets_.push_back({a,baseline,input_bytes_});
        std::map<std::string,IsolationResidentSlot> supplied;
        for(const auto &entry:scratch_baselines_)
            if(entry.name.empty() || !supplied.emplace(entry.name,entry.baseline).second)
                throw std::invalid_argument("GEMM isolation workspace baseline name is missing/duplicated");
        const auto requirements=[&]
        {auto arithmetic=beginOutputPartitionScope();return stage_->getWorkspaceRequirements(p.m,p.n,p.k);}();
        for(const auto &buffer:requirements.buffers)
        {
            if(!buffer.size_bytes) continue;
            const auto found=supplied.find(buffer.name);
            const auto actual=workspace_->getBuffer(buffer.name);
            if(found==supplied.end() || !actual || workspace_->getBufferSize(buffer.name)<buffer.size_bytes ||
                found->second.bytes!=buffer.size_bytes)
                throw std::invalid_argument("GEMM isolation requires each actual named workspace reset baseline");
            const auto lease=workspace_->retainBuffer(buffer.name,0,buffer.size_bytes);
            leases_.push_back(lease);
            resets_.push_back({reinterpret_cast<std::uintptr_t>(actual),slot(found->second),buffer.size_bytes});
            names_.push_back(buffer.name); supplied.erase(found);
        }
        if(!supplied.empty()) throw std::invalid_argument("GEMM isolation rejects unrelated workspace baselines");
        // Every reset source remains immutable while all compute destinations/scratch may change.
        for(std::size_t i=0;i<resets_.size();++i)
        {
            if(overlap(resets_[i].source,resets_[i].bytes,c,output_bytes_) ||
                (i>0 && overlap(resets_[i].destination,resets_[i].bytes,c,output_bytes_)))
                throw std::invalid_argument("GEMM output aliases an immutable reset baseline");
            for(std::size_t j=0;j<resets_.size();++j)
                if(overlap(resets_[i].source,resets_[i].bytes,resets_[j].destination,resets_[j].bytes) ||
                    (i!=j && (overlap(resets_[i].source,resets_[i].bytes,resets_[j].source,resets_[j].bytes) ||
                              overlap(resets_[i].destination,resets_[i].bytes,resets_[j].destination,resets_[j].bytes))))
                    throw std::invalid_argument("GEMM isolation reset sources/destinations overlap");
        }
        for(const auto &[address,bytes]:weights_)
        {
            if(!bytes) continue;
            if(!address || bytes>std::numeric_limits<std::uintptr_t>::max()-address ||
                overlap(address,bytes,c,output_bytes_))
                throw std::invalid_argument("GEMM output aliases immutable prepared weights");
            for(const auto &copy:resets_)
                if(overlap(address,bytes,copy.source,copy.bytes) || overlap(address,bytes,copy.destination,copy.bytes))
                    throw std::invalid_argument("Isolation reset/input/workspace aliases prepared weight storage");
        }
        graph_.addNode("isolated_prepared_gemm",std::move(stage),device_);
    }
    /** @brief Revoke the prepared engine's scratch borrow after caller retires both native captures. */
    ~PreparedGemmIsolationAdapter() {if(stage_)stage_->unbindWorkspace();}
    PreparedGemmIsolationAdapter(const PreparedGemmIsolationAdapter &)=delete;
    PreparedGemmIsolationAdapter &operator=(const PreparedGemmIsolationAdapter &)=delete;
    /** @return Frozen exact work/provenance; not an inference or byte certificate. */
    [[nodiscard]] const CapturedComputeIsolationContract &contract() const noexcept {return *contract_;}
    /**
     * @brief Copy actual pre-execution A/scratch into caller-admitted immutable slots outside capture.
     * @param backend Exact native backend selected by the caller's existing worker context.
     * @param stream Ordered producer stream after the qualified input observation, never default.
     * @note This is an explicit diagnostic setup copy, not an inference transfer or state mirror.
     *       Callers supplying an existing upstream A snapshot should use freezeScratchBaselines().
     */
    void freezeCurrentInputs(IBackend &backend,void *stream)
    { freeze(backend,stream,true); }
    /** @brief Retain an already-authenticated upstream A slot and freeze only actual workspace state.
     * @param backend Exact native copy backend.
     * @param stream Ordered snapshot/producer stream.
     */
    void freezeScratchBaselines(IBackend &backend,void *stream)
    { freeze(backend,stream,false); }
    /**
     * @brief Prepare the original stage frontier/dependency plan through the existing executor.
     * @param executor Canonical executor borrowing the actual arena.
     * @param context Actual participant GPU context.
     * @param stream Exact future recording stream, after baseline publication.
     */
    void prepareRecording(DeviceGraphExecutor &executor,IDeviceContext *context,void *stream)
    {
        requireReady(stream,false);
        if(!frozen_ || stream!=stream_ || !context || context->deviceId()!=device_ || executor.arena()!=arena_)
            throw std::invalid_argument("Isolation frontier preparation requires its frozen native input owner");
        stage_->setGPUStream(stream);
        auto arithmetic=beginOutputPartitionScope();
        if(!executor.prepareGraphStorageForCapture(graph_,context,stream,"isolated_prepared_gemm"))
            throw std::runtime_error("Canonical isolation stage frontier preparation failed");
        prepared_=true;
    }
    /**
     * @brief Record complete D2D restore nodes into a separate caller-owned native graph.
     * @param backend Exact copy backend, without native allocation or synchronization.
     * @param owner Fresh exclusive native graph currently recording the restore nodes.
     * @param stream Same exact stream as the subsequent isolated compute graph.
     */
    void recordRestore(IBackend &backend,IGPUGraphCapture &owner,void *stream)
    {
        requireReady(stream,true);
        requireRecordingOwner(owner,compute_owner_,restore_owner_,stream);
        for(const auto &copy:resets_)
            if(!backend.deviceCopyAsync(reinterpret_cast<void *>(copy.destination),reinterpret_cast<const void *>(copy.source),
                    copy.bytes,device_.ordinal,stream)) throw std::runtime_error("Isolated GEMM restore recording failed");
        restore_owner_=&owner;
    }
    /**
     * @brief Record the actual prepared GEMM through the production setup stage runner.
     * @param executor Existing canonical executor bound to this adapter's BufferArena.
     * @param context Exact participant GPU context used by native capture.
     * @param owner Fresh exclusive native graph currently recording the compute nodes.
     * @param stream Exact stream already recording the fresh caller-owned compute graph.
     * @note Instantiate/launch/native-inventory and byte proof remain required before timing.
     */
    void recordCompute(DeviceGraphExecutor &executor,IDeviceContext *context,IGPUGraphCapture &owner,void *stream)
    {
        requireReady(stream,true);
        requireRecordingOwner(owner,restore_owner_,compute_owner_,stream);
        if(!context || context->deviceId()!=device_ || executor.arena()!=arena_)
            throw std::invalid_argument("GEMM isolation requires its actual GPU context and canonical arena");
        stage_->setGPUStream(stream);
        auto arithmetic=beginOutputPartitionScope();
        if(!executor.recordSetupGraphMaterialization(graph_,context))
            throw std::runtime_error("Canonical GEMM setup materialization failed");
        compute_owner_=&owner;
    }
    /**
     * @brief Validate immutable storage before launching the separately recorded reset/compute pair.
     * @param restore Actual fresh reset executable recorded with recordRestore().
     * @param compute Actual fresh compute executable recorded with recordCompute().
     * @param stream Exact original stream; no communication or synchronization is performed.
     * @note Caller owns both graphs through terminal retirement and must not reset/rebind them.
     */
    void launchRestored(IGPUGraphCapture &restore,IGPUGraphCapture &compute,void *stream)
    {
        validateRecordedReplay(restore,compute,stream);
        if(!restore.launch() || !compute.launch())
            throw std::runtime_error("Isolation reset/compute replay submission failed");
    }
    /**
     * @brief Validate original immutable replay bindings without submitting native work.
     * @param restore Exact retained reset owner originally recorded by this adapter.
     * @param compute Exact retained compute owner originally recorded by this adapter.
     * @param stream Exact original execution stream on the same participant.
     * @note A diagnostic may compose these original graphs into one private
     *       immutable timed parent. That parent must retain the checked child
     *       owners and may not replace, reset or re-record their native graphs.
     *       This method grants no communication, alternate storage or launch policy.
     */
    void validateRecordedReplay(const IGPUGraphCapture &restore,const IGPUGraphCapture &compute,void *stream) const
    {
        requireReady(stream,false);
        if(!frozen_ || !prepared_ || stream!=stream_ || &restore==&compute ||
            &restore!=restore_owner_ || &compute!=compute_owner_ ||
            restore.executionStream()!=stream || compute.executionStream()!=stream ||
            !restore.hasExecutable() || !compute.hasExecutable())
            throw std::logic_error("Isolation replay requires two original retained executables and frozen input state");
    }
private:
    /**
     * @brief Retain the original source partition token through one workspace/frontier/recording call.
     * @return Existing backend scope; no new selector/global policy or arithmetic is implemented.
     * @note Identity mode explicitly requests actualN, so an unrelated outer
     *       arithmetic scope cannot silently change this adapter's frozen intent.
     *       Backend RAII restores its previous setting on both success and throw.
     */
    std::unique_ptr<ITensorGemm::OutputPartitionEquivalenceScope> beginOutputPartitionScope() const
    {
        const int actual=stage_->getParams().n;
        return engine_->beginOutputPartitionEquivalenceScope(actual,
            partition_.serial_columns?partition_.serial_columns:actual);
    }
    struct Copy {std::uintptr_t destination,source;std::size_t bytes;};
    /** @brief Bind only a distinct original graph on the exact recording stream, once per role.
     * @note The caller must retain exclusive graph ownership and never reset/re-record it;
     *       IGPUGraphCapture currently exposes no immutable graph-generation lease.
     */
    static void requireRecordingOwner(IGPUGraphCapture &owner,const IGPUGraphCapture *other,
        const IGPUGraphCapture *existing,void *stream)
    {
        if(existing || &owner==other || owner.executionStream()!=stream || owner.hasExecutable())
            throw std::logic_error("Isolation recording requires a fresh exclusive graph on its exact stream");
    }
    /** @brief Checked physical FP32 extent; no capacity or live-count inference. */
    static std::size_t matrixBytes(int rows,int columns)
    {
        if(rows<=0 || columns<=0 || std::size_t(rows)>std::numeric_limits<std::size_t>::max()/sizeof(float)/std::size_t(columns))
            throw std::invalid_argument("Isolated GEMM physical extent overflow");
        return std::size_t(rows)*std::size_t(columns)*sizeof(float);
    }
    /** @brief Resolve actual resident owner/device/capacity before pointer arithmetic. */
    std::uintptr_t resident(const std::shared_ptr<ITensor> &tensor,std::size_t offset,std::size_t bytes) const
    {
        if(!tensor || tensor->current_device()!=std::optional<DeviceId>(device_) || !tensor->gpu_data_ptr() ||
            !bytes || offset>tensor->size_bytes() || bytes>tensor->size_bytes()-offset)
            throw std::invalid_argument("Isolation slot lacks its real resident tensor/device/capacity");
        const auto address=reinterpret_cast<std::uintptr_t>(tensor->gpu_data_ptr());
        if(offset>std::numeric_limits<std::uintptr_t>::max()-address || bytes>std::numeric_limits<std::uintptr_t>::max()-address-offset)
            throw std::invalid_argument("Isolation resident slot pointer overflow");
        return address+offset;
    }
    /** @brief Resolve the complete borrowed resident slot through its actual tensor owner. */
    std::uintptr_t slot(const IsolationResidentSlot &value) const {return resident(value.tensor,value.offset,value.bytes);}
    /** @brief Compare validated native spans without overflowing their endpoints. */
    static bool overlap(std::uintptr_t a,std::size_t an,std::uintptr_t b,std::size_t bn)
    {return a<b+bn && b<a+an;}
    /** @brief Bind one nondefault stream and copy real baseline bytes before either native capture. */
    void freeze(IBackend &backend,void *stream,bool copy_input)
    {
        requireReady(stream,false);
        if(frozen_) throw std::logic_error("Isolation baselines are immutable after setup publication");
        TransferEngine::requireDeviceInput(tensors_[0].get(),device_,stream);
        if(!copy_input) TransferEngine::requireDeviceInput(input_baseline_.tensor.get(),device_,stream);
        for(std::size_t i=copy_input?0:1;i<resets_.size();++i)
        {
            const auto &copy=resets_[i];
            if(!backend.deviceCopyAsync(reinterpret_cast<void *>(copy.source),reinterpret_cast<const void *>(copy.destination),
                copy.bytes,device_.ordinal,stream)) throw std::runtime_error("Isolation baseline publication failed");
        }
        if(copy_input) TransferEngine::publishDeviceWrite(input_baseline_.tensor.get(),device_,stream);
        for(const auto &entry:scratch_baselines_) TransferEngine::publishDeviceWrite(entry.baseline.tensor.get(),device_,stream);
        stream_=stream; frozen_=true;
    }
    /** @brief Compare named immutable descriptor fields; C++ padding is never identity. */
    static bool sameDescriptor(const DeviceNativeVNNIMatrixDesc &a,const DeviceNativeVNNIMatrixDesc &b) noexcept
    {
        return std::tie(a.payload, a.scales, a.mins, a.emins, a.n, a.k, a.blocks_per_row, a.codebook_id, a.allocation_payload_bytes_per_block, a.allocation_has_mins, a.allocation_has_emins, a.source_codebook_id, a.source_is_superblock, a.source_identity_present, a.reserved, a.reserved_tail) ==
               std::tie(b.payload, b.scales, b.mins, b.emins, b.n, b.k, b.blocks_per_row, b.codebook_id, b.allocation_payload_bytes_per_block, b.allocation_has_mins, b.allocation_has_emins, b.source_codebook_id, b.source_is_superblock, b.source_identity_present, b.reserved, b.reserved_tail);
    }
    /** @brief Reject stale actual bindings and phase/stream errors before native enqueue. */
    void requireReady(void *stream,bool recording) const
    {
        const auto word=reinterpret_cast<std::uintptr_t>(stream);
        if(word<=2 || isGraphCaptureActive()!=recording || (recording && (!frozen_ || !prepared_ || stream!=stream_)))
            throw std::logic_error("Isolation requires frozen inputs and its exact explicit capture stream/phase");
        const auto &p=stage_->getParams();
        if(resident(tensors_[0],0,input_bytes_)!=resets_[0].destination || slot(input_baseline_)!=resets_[0].source ||
            resident(tensors_[2],0,output_bytes_)!=contract_->bindings()[1].address ||
            arena_->getTensor(*p.a_buffer_id)!=p.A || arena_->getTensor(*p.c_buffer_id)!=p.C ||
            store_->gemmKernel(*p.prepared_ref)!=engine_ || stage_->getWorkspace()!=workspace_.get() || !stage_->hasWorkspace())
            throw std::logic_error("Isolation prepared/input/workspace binding changed");
        DeviceNativeVNNIMatrixDesc current{};
        if(!engine_->inspectPreparedNativeVNNIMatrixDesc(current) || !sameDescriptor(current,weight_))
            throw std::logic_error("Isolation immutable prepared descriptor changed");
        for(const auto &entry:scratch_baselines_)
        {
            const auto found=std::find(names_.begin(),names_.end(),entry.name);
            if(found==names_.end() || slot(entry.baseline)!=resets_[std::size_t(found-names_.begin())+1].source)
                throw std::logic_error("Isolation workspace baseline binding changed");
        }
        for(std::size_t i=0;i<names_.size();++i)
            if(reinterpret_cast<std::uintptr_t>(workspace_->getBuffer(names_[i]))!=resets_[i+1].destination ||
                workspace_->getBufferSize(names_[i])<resets_[i+1].bytes)
                throw std::logic_error("Isolation workspace generation changed");
    }
    std::array<std::shared_ptr<ITensor>,3> tensors_;
    std::shared_ptr<PreparedWeightStore> store_;
    std::shared_ptr<DeviceWorkspaceManager> workspace_;
    IsolationResidentSlot input_baseline_;
    std::vector<IsolationWorkspaceBaseline> scratch_baselines_;
    std::vector<std::shared_ptr<const WorkspaceBufferLease>> leases_;
    std::vector<std::string> names_;
    std::vector<Copy> resets_;
    BufferArena *arena_;
    GEMMStage *stage_;
    const IsolationOutputPartitionPolicy partition_;
    DeviceId device_;
    ITensorGemm *engine_=nullptr;
    DeviceNativeVNNIMatrixDesc weight_{};
    std::array<std::pair<std::uintptr_t,std::size_t>,4> weights_;
    std::size_t input_bytes_=0,output_bytes_=0;
    std::optional<CapturedComputeIsolationContract> contract_;
    IGPUGraphCapture *restore_owner_=nullptr,*compute_owner_=nullptr; ///< Borrowed exclusive original graph owners.
    void *stream_=nullptr;
    bool frozen_=false,prepared_=false;
    ComputeGraph graph_; ///< Destroy stage before prepared/workspace/tensor owners.
};
} // namespace llaminar2::diagnostics
