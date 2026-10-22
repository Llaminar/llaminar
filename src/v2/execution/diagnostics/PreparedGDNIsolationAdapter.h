/**
 * @file PreparedGDNIsolationAdapter.h
 * @brief Isolate the complete original prepared GDN projection stage without changing its fusion.
 *
 * The stage owns subgroup selection and arithmetic. This adapter owns only a
 * declarative one-node compute graph and disjoint immutable input/workspace
 * baselines. All storage is already admitted and resident. Native recording
 * follows the production executor; no request loop, collective or eager path
 * is installed by this explicit diagnostic.
 */
#pragma once
#include "PreparedGemmIsolationAdapter.h"
#include "../compute_stages/stages/GDNProjectionStage.h"

namespace llaminar2::diagnostics
{
/**
 * @brief Static complete-bundle form implemented by the first GDN adapter.
 * @param p Real production stage params.
 * @return Whether every read and output is representable by fixed physical rows.
 */
inline bool staticGDNIsolationFormSupported(const GDNProjectionStage::Params &p) noexcept
{
    return !p.mpi_ctx && p.m>0 && p.k>0 && p.k%32==0 && p.n_qkv>0 && p.n_z>0 && p.n_a>0 && p.n_b>0 &&
        p.qkv_operand_precision==QuantizedActivationOperandPrecision::SingleQ8 &&
        !p.qkv_row_range && !p.verifier_row_range && !p.force_decode_equivalent_verifier_prefill;
}
/**
 * @brief Prove the exact Q8 physical planes before forming immutable alias spans.
 * @param n Actual descriptor exported by the retained prepared engine.
 * @throws std::invalid_argument For noncanonical source, geometry or allocation planes.
 *
 * Ordinary immutable pool weights leave reusable-slot allocation metadata zero;
 * their physical capacity follows codebook 19. Explicit exact-Q8 slots name 32
 * bytes per block. Neither representation permits additional metadata planes.
 */
inline void validateGDNIsolationQ8Allocation(const DeviceNativeVNNIMatrixDesc &n)
{
    if(!n.valid() || n.codebook_id!=19 || n.source_identity_present!=1 ||
        n.source_codebook_id!=19 || n.source_is_superblock || n.reserved || n.reserved_tail ||
        n.k%32 || n.blocks_per_row!=static_cast<std::uint32_t>(n.k/32) ||
        (n.allocation_payload_bytes_per_block!=0 && n.allocation_payload_bytes_per_block!=32) ||
        n.allocation_has_mins || n.allocation_has_emins || n.mins || n.emins)
        throw std::invalid_argument("GDN isolation Q8 prepared allocation differs from its canonical physical format");
}
/** @brief Complete native GDN stage plus retained original prepared/source/reset owners. */
class PreparedGDNIsolationAdapter final
{
public:
    /**
     * @brief Seal a real complete GDN stage after exact admitted workspace binding.
     * @param stage Original production stage, transferred to the diagnostic graph.
     * @param input Exact FP32 activation owner consumed by the stage.
     * @param outputs Four exact FP32 owners in QKV/Z/alpha/beta order.
     * @param weights Four immutable actual prepared source owners in the same order.
     * @param store Actual prepared handle authority.
     * @param workspace Exact stage-bound admitted scratch owner.
     * @param arena Canonical activation binding authority.
     * @param provenance Original upstream observation and retained baseline lifetime.
     * @param input_baseline Immutable original complete physical input row bank.
     * @param scratch_baselines Complete named workspace baseline banks.
     * @throws std::invalid_argument For an unimplemented form or incomplete ownership.
     */
    PreparedGDNIsolationAdapter(std::unique_ptr<GDNProjectionStage> stage,std::shared_ptr<ITensor> input,
        std::array<std::shared_ptr<ITensor>,4> outputs,std::array<std::shared_ptr<ITensor>,4> weights,
        std::shared_ptr<PreparedWeightStore> store,std::shared_ptr<DeviceWorkspaceManager> workspace,
        BufferArena &arena,IsolationInputProvenance provenance,IsolationResidentSlot input_baseline,
        std::vector<IsolationWorkspaceBaseline> scratch_baselines)
        :input_(std::move(input)),outputs_(std::move(outputs)),weights_(std::move(weights)),store_(std::move(store)),
         workspace_(std::move(workspace)),baseline_(std::move(input_baseline)),scratch_(std::move(scratch_baselines)),
         arena_(&arena),stage_(stage.get())
    {
        if(isGraphCaptureActive() || !stage_ || !store_ || !workspace_)
            throw std::invalid_argument("GDN isolation requires cold actual complete prepared owners");
        const auto &p=stage_->getParams();device_=p.device_id;
        if(!device_.is_gpu() || device_.ordinal<0 || !staticGDNIsolationFormSupported(p) ||
            p.prepared_store!=store_.get() || p.input!=input_.get() || !p.input_buffer_id ||
            !provenance.owner || provenance.owner.get()!=baseline_.tensor.get() || provenance.observation.empty() ||
            provenance.producer_stage.empty() || !provenance.generation ||
            (provenance.origin!=IsolationInputOrigin::QualifiedInferenceObservation && provenance.origin!=IsolationInputOrigin::SyntheticFixture) ||
            workspace_->device()!=device_ || stage_->getWorkspace()!=workspace_.get())
            throw std::invalid_argument("Unsupported GDN isolation device, row policy or prepared owner");
        provenance_=std::move(provenance);
        refs_={p.prepared_ref_qkv,p.prepared_ref_z,p.prepared_ref_a,p.prepared_ref_b};
        ids_={p.output_qkv_buffer_id,p.output_z_buffer_id,p.output_a_buffer_id,p.output_b_buffer_id};
        widths_={p.n_qkv,p.n_z,p.n_a,p.n_b};
        const std::array<const ITensor *,4> actual_weights={p.w_qkv,p.w_z,p.w_a,p.w_b};
        const std::array<const ITensor *,4> actual_outputs={p.output_qkv,p.output_z,p.output_a,p.output_b};
        input_bytes_=matrixBytes(p.m,p.k);
        if(!input_ || input_->native_type()!=TensorType::FP32 || baseline_.bytes!=input_bytes_)
            throw std::invalid_argument("GDN isolation requires exact original FP32 input extent");
        bindings_.push_back({*p.input_buffer_id,resident(input_,0,input_bytes_),slot(baseline_),input_bytes_,input_,baseline_.tensor});
        for(std::size_t i=0;i<4;++i)
        {
            if(!outputs_[i] || outputs_[i]->native_type()!=TensorType::FP32 || actual_outputs[i]!=outputs_[i].get() ||
                !weights_[i] || actual_weights[i]!=weights_[i].get() || !refs_[i] || !ids_[i] ||
                refs_[i]->device!=device_ || !store_->contains(*refs_[i]))
                throw std::invalid_argument("GDN isolation requires every original output/source/prepared binding");
            const auto binding=store_->binding(*refs_[i]);
            if(!binding || !binding->immutable || binding->tensor!=weights_[i].get())
                throw std::invalid_argument("GDN isolation prepared source binding is not the immutable original");
            engines_[i]=store_->gemmKernel(*refs_[i]);
            if(!engines_[i])throw std::invalid_argument("GDN isolation has no original prepared engine");
            if(i<2)
            {
                if(!engines_[i]->inspectPreparedNativeVNNIMatrixDesc(native_[i]) || !native_[i].valid() ||
                    !native_[i].source_identity_present || native_[i].source_codebook_id!=19 || native_[i].codebook_id!=19 ||
                    native_[i].n!=widths_[i] || native_[i].k!=p.k)
                    throw std::invalid_argument("GDN isolation QKV/Z require exact original Q8 prepared descriptors");
            }
            else if(!engines_[i]->exportContiguousFloatingPointWeights(floating_[i-2]) || !floating_[i-2].valid() ||
                floating_[i-2].type!=TensorType::FP16 || floating_[i-2].n!=widths_[i] || floating_[i-2].k!=p.k ||
                binding->floating_operand_precision!=FloatingActivationOperandPrecision::FP16RNE)
                throw std::invalid_argument("GDN isolation controls require original-source half-RNE prepared arithmetic");
            output_bytes_[i]=matrixBytes(p.m,widths_[i]);
            bindings_.push_back({*ids_[i],resident(outputs_[i],0,output_bytes_[i]),0,output_bytes_[i],outputs_[i],{}});
        }
        std::string error;if(!stage_->validatePreparedWeights(&error))throw std::invalid_argument(error);
        const auto contract=stage_->bufferContract();
        if(contract.inputs.size()!=1 || contract.outputs.size()!=4 || !contract.inouts.empty() ||
            !contract.weight_tensors.empty() || contract.prepared_weights.size()!=4)
            throw std::invalid_argument("GDN isolation found an undeclared or extra production stage operand");
        (void)validateIsolationBufferClosure(contract,bindings_);
        for(const auto &b:bindings_)if(arena_->getTensor(b.id)!=b.owner.get())
            throw std::invalid_argument("GDN isolation activation differs from the actual arena binding");
        copies_.push_back({bindings_[0].address,bindings_[0].baseline,input_bytes_});
        const auto requirements=stage_->getWorkspaceRequirements(p.m,0,p.k);
        std::map<std::string,IsolationResidentSlot> supplied;
        for(const auto &s:scratch_)if(s.name.empty() || !supplied.emplace(s.name,s.baseline).second)
            throw std::invalid_argument("GDN isolation workspace baseline name is empty or duplicated");
        for(const auto &b:requirements.buffers)if(b.size_bytes)
        {
            const auto found=supplied.find(b.name);auto *actual=workspace_->getBuffer(b.name);
            if(found==supplied.end() || !actual || workspace_->getBufferSize(b.name)<b.size_bytes || found->second.bytes!=b.size_bytes)
                throw std::invalid_argument("GDN isolation lacks an exact original named workspace baseline");
            leases_.push_back(workspace_->retainBuffer(b.name,0,b.size_bytes));
            names_.push_back(b.name);copies_.push_back({reinterpret_cast<std::uintptr_t>(actual),slot(found->second),b.size_bytes});
            supplied.erase(found);
        }
        if(!supplied.empty())throw std::invalid_argument("GDN isolation rejects unrelated workspace baselines");
        validateAliases();signature_=stage_->graphCaptureVariantSignature();
        graph_.addNode("isolated_prepared_gdn",std::move(stage),device_);
    }
    /** @brief Revoke all four engines' scratch borrows after original native graph retirement. */
    ~PreparedGDNIsolationAdapter(){if(stage_)stage_->unbindWorkspace();}
    /** @brief Freeze only scratch; the independently authenticated upstream input baseline is immutable. */
    void freezeScratchBaselines(IBackend &backend,void *stream)
    {
        ready(stream,false);
        if(frozen_)throw std::logic_error("GDN isolation baselines were already published");
        TransferEngine::requireDeviceInput(input_.get(),device_,stream);
        TransferEngine::requireDeviceInput(baseline_.tensor.get(),device_,stream);
        for(std::size_t i=1;i<copies_.size();++i)
            if(!backend.deviceCopyAsync(reinterpret_cast<void *>(copies_[i].source),reinterpret_cast<void *>(copies_[i].destination),copies_[i].bytes,device_.ordinal,stream))
                throw std::runtime_error("GDN isolation scratch baseline publication failed");
        for(const auto &s:scratch_)TransferEngine::publishDeviceWrite(s.baseline.tensor.get(),device_,stream);
        stream_=stream;frozen_=true;
    }
    /** @brief Prepare the actual production frontier before either native recording. */
    void prepareRecording(DeviceGraphExecutor &executor,IDeviceContext *context,void *stream)
    {
        ready(stream,false);contextMatches(executor,context);
        if(!frozen_ || stream!=stream_)throw std::logic_error("GDN isolation input publication is incomplete");
        // Storage admission does not provision launch resources. The production
        // stage lifecycle creates backend projection streams/events before capture;
        // the already prepared floating engines borrow context-owned BLAS handles.
        if(!stage_->prepareGraphLaunch(context,stream))
            throw std::runtime_error("GDN isolation production launch preparation failed");
        if(!executor.prepareGraphStorageForCapture(graph_,context,stream,"isolated_prepared_gdn"))
            throw std::runtime_error("GDN isolation production frontier preparation failed");
        prepared_=true;
    }
    /** @brief Record exact input/scratch restoration as a separate retained native graph. */
    void recordRestore(IBackend &backend,IGPUGraphCapture &owner,void *stream)
    {
        ready(stream,true);fresh(owner,restore_,compute_,stream);
        for(const auto &c:copies_)
            if(!backend.deviceCopyAsync(reinterpret_cast<void *>(c.destination),reinterpret_cast<void *>(c.source),c.bytes,device_.ordinal,stream))
                throw std::runtime_error("GDN isolation restore recording failed");
        restore_=&owner;
    }
    /** @brief Record the unchanged GDNProjectionStage through the production setup executor. */
    void recordCompute(DeviceGraphExecutor &executor,IDeviceContext *context,IGPUGraphCapture &owner,void *stream)
    {
        ready(stream,true);fresh(owner,compute_,restore_,stream);contextMatches(executor,context);stage_->setGPUStream(stream);
        if(!executor.recordSetupGraphMaterialization(graph_,context))throw std::runtime_error("GDN isolation production recording failed");
        compute_=&owner;
    }
    /** @brief Validate original retained graphs and unchanged complete bindings before native submission. */
    void validateRecordedReplay(const IGPUGraphCapture &restore,const IGPUGraphCapture &compute,void *stream) const
    {
        ready(stream,false);
        if(!frozen_ || !prepared_ || stream!=stream_ || &restore!=restore_ || &compute!=compute_ || &restore==&compute ||
            !restore.hasExecutable() || !compute.hasExecutable() || restore.executionStream()!=stream || compute.executionStream()!=stream)
            throw std::logic_error("GDN isolation replay requires its original immutable native graph pair");
    }
    /** @brief Submit reset and real compute on their exact original nondefault stream. */
    void launchRestored(IGPUGraphCapture &restore,IGPUGraphCapture &compute,void *stream)
    {
        validateRecordedReplay(restore,compute,stream);
        if(!restore.launch() || !compute.launch())throw std::runtime_error("GDN isolation replay submission failed");
    }
private:
    struct Copy{std::uintptr_t destination=0,source=0;std::size_t bytes=0;};
    /** @brief Reject nonrepresentable physical FP32 matrices before pointer arithmetic. */
    static std::size_t matrixBytes(int m,int n)
    {
        if(m<=0 || n<=0 || std::size_t(m)>std::numeric_limits<std::size_t>::max()/4/std::size_t(n))
            throw std::invalid_argument("GDN isolation physical matrix extent overflows");
        return std::size_t(m)*std::size_t(n)*4;
    }
    /** @brief Resolve only actual retained resident tensor storage on the exact execution device. */
    std::uintptr_t resident(const std::shared_ptr<ITensor> &t,std::size_t offset,std::size_t bytes) const
    {
        if(!t || t->current_device()!=std::optional<DeviceId>(device_) || !t->gpu_data_ptr() || !bytes ||
            offset>t->size_bytes() || bytes>t->size_bytes()-offset)
            throw std::invalid_argument("GDN isolation has no exact resident tensor extent");
        const auto address=reinterpret_cast<std::uintptr_t>(t->gpu_data_ptr());
        if(offset>std::numeric_limits<std::uintptr_t>::max()-address || bytes>std::numeric_limits<std::uintptr_t>::max()-address-offset)
            throw std::invalid_argument("GDN isolation resident pointer overflows");
        return address+offset;
    }
    /** @brief Resolve the immutable slot through its real retained tensor. */
    std::uintptr_t slot(const IsolationResidentSlot &s) const{return resident(s.tensor,s.offset,s.bytes);}
    /** @brief Require fresh, distinct original recording owners on the exact stream. */
    static void fresh(const IGPUGraphCapture &owner,const IGPUGraphCapture *existing,const IGPUGraphCapture *other,void *stream)
    {
        if(existing || &owner==other || owner.executionStream()!=stream || owner.hasExecutable())
            throw std::logic_error("GDN isolation recording owner is stale, shared or on another stream");
    }
    /** @brief Reject a foreign context or canonical arena. */
    void contextMatches(DeviceGraphExecutor &executor,IDeviceContext *context) const
    {if(!context || context->deviceId()!=device_ || executor.arena()!=arena_)throw std::invalid_argument("GDN isolation context/arena differs");}
    /** @brief Prove reset banks, writes, scratch and immutable prepared storage are pairwise safe. */
    void validateAliases() const
    {
        const auto overlap=[](std::uintptr_t a,std::size_t an,std::uintptr_t b,std::size_t bn){return a<b+bn && b<a+an;};
        for(std::size_t i=0;i<copies_.size();++i)
        {
            const auto &a=copies_[i];
            for(std::size_t j=0;j<copies_.size();++j)
            {
                const auto &b=copies_[j];
                if(overlap(a.source,a.bytes,b.destination,b.bytes) ||
                    (i!=j && (overlap(a.source,a.bytes,b.source,b.bytes) || overlap(a.destination,a.bytes,b.destination,b.bytes))))
                    throw std::invalid_argument("GDN isolation immutable reset sources or destinations alias");
            }
            for(std::size_t j=1;j<bindings_.size();++j)
                if(overlap(a.source,a.bytes,bindings_[j].address,bindings_[j].bytes) || overlap(a.destination,a.bytes,bindings_[j].address,bindings_[j].bytes))
                    throw std::invalid_argument("GDN isolation scratch/reset bank aliases an output");
        }
        std::vector<std::pair<std::uintptr_t,std::size_t>> regions;
        for(const auto &n:native_)
        {
            validateGDNIsolationQ8Allocation(n);
            const auto sizes=nativeVnniPackedRegionSizes(std::size_t(n.n),std::size_t(n.k),native_vnni_formats::Q8_0);
            regions.push_back({reinterpret_cast<std::uintptr_t>(n.payload),sizes.payload_bytes});
            regions.push_back({reinterpret_cast<std::uintptr_t>(n.scales),sizes.scales_bytes});
        }
        for(const auto &f:floating_)
        {
            if(!f.valid())throw std::invalid_argument("GDN isolation floating prepared region is invalid");
            regions.push_back({reinterpret_cast<std::uintptr_t>(f.data),f.bytes});
        }
        for(std::size_t i=0;i<regions.size();++i)
        {
            const auto [address,bytes]=regions[i];
            if(!address || !bytes || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
                throw std::invalid_argument("GDN isolation prepared region is absent or overflows");
            for(const auto &c:copies_)
                if(overlap(address,bytes,c.source,c.bytes) || overlap(address,bytes,c.destination,c.bytes))
                    throw std::invalid_argument("GDN isolation reset/input/scratch aliases immutable prepared weights");
            for(const auto &b:bindings_)
                if(overlap(address,bytes,b.address,b.bytes))
                    throw std::invalid_argument("GDN isolation output aliases immutable prepared weights");
            for(std::size_t j=0;j<i;++j)
                if(overlap(address,bytes,regions[j].first,regions[j].second))
                    throw std::invalid_argument("Independent GDN prepared source regions alias");
        }
    }
    /** @brief Compare named immutable native fields, never C++ descriptor padding. */
    static bool sameNative(const DeviceNativeVNNIMatrixDesc &a,const DeviceNativeVNNIMatrixDesc &b)
    {
        return std::tie(a.payload,a.scales,a.mins,a.emins,a.n,a.k,a.blocks_per_row,a.codebook_id,a.allocation_payload_bytes_per_block,
            a.allocation_has_mins,a.allocation_has_emins,a.source_codebook_id,a.source_is_superblock,a.source_identity_present,a.reserved,a.reserved_tail)==
            std::tie(b.payload,b.scales,b.mins,b.emins,b.n,b.k,b.blocks_per_row,b.codebook_id,b.allocation_payload_bytes_per_block,
            b.allocation_has_mins,b.allocation_has_emins,b.source_codebook_id,b.source_is_superblock,b.source_identity_present,b.reserved,b.reserved_tail);
    }
    /** @brief Validate exact capture phase, source descriptors and retained storage before enqueue. */
    void ready(void *stream,bool recording) const
    {
        if(reinterpret_cast<std::uintptr_t>(stream)<=2 || isGraphCaptureActive()!=recording ||
            (recording && (!frozen_ || !prepared_ || stream!=stream_)))throw std::logic_error("GDN isolation stream/phase is invalid");
        if(stage_->graphCaptureVariantSignature()!=signature_ || stage_->getWorkspace()!=workspace_.get() ||
            resident(input_,0,input_bytes_)!=bindings_[0].address || slot(baseline_)!=bindings_[0].baseline)
            throw std::logic_error("GDN isolation stage/input/workspace identity changed");
        for(const auto &b:bindings_)if(arena_->getTensor(b.id)!=b.owner.get())throw std::logic_error("GDN isolation arena binding changed");
        for(std::size_t i=0;i<4;++i)
        {
            if(store_->gemmKernel(*refs_[i])!=engines_[i] || resident(outputs_[i],0,output_bytes_[i])!=bindings_[i+1].address)
                throw std::logic_error("GDN isolation prepared engine/output binding changed");
            auto *consumer=dynamic_cast<IWorkspaceConsumer *>(engines_[i]);
            if(!consumer || consumer->getWorkspace()!=workspace_.get())throw std::logic_error("GDN isolation engine scratch changed");
            if(i<2)
            {
                DeviceNativeVNNIMatrixDesc current{};
                if(!engines_[i]->inspectPreparedNativeVNNIMatrixDesc(current) || !sameNative(current,native_[i]))
                    throw std::logic_error("GDN isolation native prepared descriptor changed");
            }
            else
            {
                ContiguousFloatingPointWeightDescriptor current{};const auto &old=floating_[i-2];
                if(!engines_[i]->exportContiguousFloatingPointWeights(current) ||
                    std::tie(current.data,current.type,current.n,current.k,current.bytes)!=std::tie(old.data,old.type,old.n,old.k,old.bytes))
                    throw std::logic_error("GDN isolation floating prepared descriptor changed");
            }
        }
        for(const auto &entry:scratch_)
        {
            const auto found=std::find(names_.begin(),names_.end(),entry.name);
            if(found==names_.end() || slot(entry.baseline)!=copies_[std::size_t(found-names_.begin())+1].source)
                throw std::logic_error("GDN isolation immutable scratch baseline binding changed");
        }
        for(std::size_t i=0;i<names_.size();++i)
            if(reinterpret_cast<std::uintptr_t>(workspace_->getBuffer(names_[i]))!=copies_[i+1].destination ||
                workspace_->getBufferSize(names_[i])<copies_[i+1].bytes)
                throw std::logic_error("GDN isolation workspace generation changed");
    }
    std::shared_ptr<ITensor> input_;
    std::array<std::shared_ptr<ITensor>,4> outputs_,weights_;
    std::shared_ptr<PreparedWeightStore> store_;
    std::shared_ptr<DeviceWorkspaceManager> workspace_;
    IsolationInputProvenance provenance_;
    IsolationResidentSlot baseline_;
    std::vector<IsolationWorkspaceBaseline> scratch_;
    std::vector<std::shared_ptr<const WorkspaceBufferLease>> leases_;
    std::vector<IsolationBuffer> bindings_;
    std::vector<Copy> copies_;
    std::vector<std::string> names_;
    std::array<std::optional<PreparedWeightRef>,4> refs_;
    std::array<std::optional<BufferId>,4> ids_;
    std::array<int,4> widths_{};
    std::array<std::size_t,4> output_bytes_{};
    std::array<ITensorGemm *,4> engines_{};
    std::array<DeviceNativeVNNIMatrixDesc,2> native_{};
    std::array<ContiguousFloatingPointWeightDescriptor,2> floating_{};
    BufferArena *arena_=nullptr;GDNProjectionStage *stage_=nullptr;
    DeviceId device_;std::size_t input_bytes_=0;std::uint64_t signature_=0;
    void *stream_=nullptr;bool frozen_=false,prepared_=false;
    IGPUGraphCapture *restore_=nullptr,*compute_=nullptr;
    ComputeGraph graph_;
};
} // namespace llaminar2::diagnostics
