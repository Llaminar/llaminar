/**
 * @file Qwen4ExpAuxiliaryState.cpp
 * @brief Borrow admitted QFN state through canonical cache lifecycle and archive boundaries.
 *
 * Local rollback copies only live raw tails and accepted lexical histories.
 * Native pool archives carry only complete compressed blocks from immutable
 * logical ranges. Direct backend copies are confined to this cache prefix-I/O
 * infrastructure boundary; TransferEngine remains the tensor event authority.
 */
#include "Qwen4ExpAuxiliaryState.h"
#include "../../kernels/qwen4exp/Qwen4ExpStateIO.h"
#include "../../kernels/qwen4exp/Qwen4ExpCheckpointGeometry.h"
#include "../../backends/BackendManager.h"
#include "../../backends/ExplicitGPUStream.h"
#include "../../transfer/TransferEngine.h"
#include "../../tensors/Tensors.h"
#include "../local_execution/graph/ComputeGraph.h"
#include "../local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
namespace
{
using CheckpointGeometry=qwen4exp::Qwen4ExpCheckpointGeometry;
constexpr std::size_t TailHeader=CheckpointGeometry::TailHeaderBytes,TailRecord=CheckpointGeometry::sparseLayerBytes();
/** @brief Reject invalid lifecycle/extent requests before state mutation. */
void require(bool condition,const char *message) { if(!condition)throw std::invalid_argument(message); }
/** @brief Validate an admitted native view without materialization or a second memory ledger. */
template<TensorType Type>void validate(Qwen4ExpArenaBinding<Type> b,std::size_t bytes)
{
    if(!b.tensor || b.tensor->native_type()!=Type || b.buffer>=BufferId::_COUNT || b.byteOffset%4!=0 ||
       b.byteOffset>b.tensor->size_bytes() || bytes>b.tensor->size_bytes()-b.byteOffset)
        throw std::invalid_argument("QFN auxiliary state view violates native arena contract: role="+
            std::string(bufferIdName(b.buffer))+" offset="+std::to_string(b.byteOffset)+
            " bytes="+std::to_string(bytes)+" capacity="+std::to_string(b.tensor?b.tensor->size_bytes():0)+
            " expected_type="+std::to_string(static_cast<int>(Type))+
            " actual_type="+std::to_string(b.tensor?static_cast<int>(b.tensor->native_type()):-1));
}
/** @brief Submit one exact CPU/GPU state copy through the selected backend implementation. */
void transfer(DeviceId device,const qwen4exp::QwenStateTransfer &p,void *stream)
{
    if(device.is_cpu()) { require(!stream,"CPU QFN state cannot use a GPU stream");qwen4exp::cpuQwenStateTransfer(p);return; }
#ifdef HAVE_CUDA
    if(device.is_cuda()) { qwen4exp::cudaQwenStateTransfer(p,stream);return; }
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm()) { qwen4exp::rocmQwenStateTransfer(p,stream);return; }
#endif
    throw std::invalid_argument("QFN auxiliary state backend was not compiled");
}
/** @brief Reset only the declared state words, preserving exact EOS-token integers. */
void fill(DeviceId device,const qwen4exp::QwenStateFill &p,void *stream)
{
    if(device.is_cpu()) { require(!stream,"CPU QFN state cannot use a GPU stream");qwen4exp::cpuQwenStateFill(p);return; }
#ifdef HAVE_CUDA
    if(device.is_cuda()) { qwen4exp::cudaQwenStateFill(p,stream);return; }
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm()) { qwen4exp::rocmQwenStateFill(p,stream);return; }
#endif
    throw std::invalid_argument("QFN auxiliary state backend was not compiled");
}
/** @brief Fixed native PLE next-bank promotion with explicit arena coherence declarations. */
class PLEPromotion final : public IComputeStage
{
    Qwen4ExpAuxiliaryState::PLEHistory history_;
    /** @brief Resolve already-prepared exact native view bytes. */
    template<TensorType Type>void *pointer(Qwen4ExpArenaBinding<Type> b) const
    {
        void *base=device().is_gpu() ? b.tensor->gpu_data_ptr() : b.tensor->raw_mutable_data();
        require(base,"QFN history promotion requires already prepared storage");
        return static_cast<std::byte *>(base)+b.byteOffset;
    }
public:
    /** @brief Seal one request's independent accepted/next banks. */
    PLEPromotion(DeviceId device,Qwen4ExpAuxiliaryState::PLEHistory h):IComputeStage(device),history_(h) {}
    /** @brief Publish exact lexical accepted bytes on the executor's stream. */
    bool execute(IDeviceContext *context) override
    {
        require(context && context->deviceId()==device(),"QFN history promotion context differs from its owner");
        void *stream=nullptr;
        if(device().is_gpu()) {
            auto execution=gpuExecution();stream=gpuStream();
            execution.requirePreparedInput(history_.nextTokens.tensor);execution.requirePreparedInput(history_.nextConvolution.tensor);
            execution.requirePreparedOutput(history_.acceptedTokens.tensor);execution.requirePreparedOutput(history_.acceptedConvolution.tensor);
        }
        transfer(device(),{pointer(history_.nextTokens),pointer(history_.acceptedTokens),8,1,
            qwen4exp::QwenStateRowSource::Fixed,nullptr,1,nullptr},stream);
        transfer(device(),{pointer(history_.nextConvolution),pointer(history_.acceptedConvolution),std::size_t(9)*history_.channels*4,1,
            qwen4exp::QwenStateRowSource::Fixed,nullptr,1,nullptr},stream);
        return true;
    }
    /** @brief Reuse the canonical native copy semantic tag. */
    ComputeStageType type()const override{return ComputeStageType::COPY;}
    std::size_t estimatedFlops()const override{return 0;}
    std::size_t estimatedMemoryBytes()const override{return 2*(8+std::size_t(9)*history_.channels*4);}
    bool supportsBackend(ComputeBackendType b)const override
    {return (device().is_cpu()&&b==ComputeBackendType::CPU)||(device().is_cuda()&&b==ComputeBackendType::GPU_CUDA)||(device().is_rocm()&&b==ComputeBackendType::GPU_ROCM);}
    StageBufferRequirements getBufferRequirements()const override{return {};}
    StageBufferContract bufferContract()const override
    {
        StageBufferContract c;
        const auto add=[&](BufferId in,BufferId out,const char *dtype) {
            if(in==out)c.addPreallocatedInOut(in,dtype);else {c.addInput(in,dtype);c.addPreallocatedOutput(out,dtype);}
        };
        add(history_.nextTokens.buffer,history_.acceptedTokens.buffer,"INT32");
        add(history_.nextConvolution.buffer,history_.acceptedConvolution.buffer,"FP32");return c;
    }
    StageDumpInfo buildDumpInfoImpl()const override{return buildDumpInfoFromContract();}
};
}

Qwen4ExpAuxiliaryState::Qwen4ExpAuxiliaryState(Params params):params_(std::move(params))
{
    require(params_.device.is_valid() && params_.requestCapacity>0 && params_.eosToken>=0 &&
        params_.requestCapacity<=params_.cache.request_capacity() && (!params_.sparseLayers.empty() || !params_.pleHistories.empty()),
        "QFN auxiliary state needs exact cache/device/request identity and nonempty model state");
    for(const auto &entry : params_.sparseLayers) {
        require(entry.layer>=0 && entry.cacheLayer>=0 && entry.sequence>=0 && entry.sequence<params_.requestCapacity &&
            entry.contextTokens==params_.cache.max_seq_len(),"QFN auxiliary sparse bank has invalid layer/request/context identity");
        validate(entry.pooledKeys,std::size_t((std::int64_t(entry.contextTokens)+3)/4)*128*4);
        validate(entry.partialKeys,4*128*4);
        if(entry.verifier) {
            const auto& v=*entry.verifier;
            require(v.rowCapacity>0 && v.prior.storage.buffer==BufferId::QFN_QSA_TRANSACTION_PRIOR &&
                v.priorTraffic.buffer==BufferId::QFN_QSA_TRANSACTION_PRIOR_TRAFFIC &&
                v.rawKeys.buffer==BufferId::QFN_QSA_CANDIDATE_RAW_KEYS &&
                v.candidateCopyBytes.buffer==BufferId::QFN_QSA_CANDIDATE_COPY_BYTES &&
                v.prior.storage.byteOffset%alignof(qwen4exp::QwenSparseTransactionPrior)==0 &&
                v.candidateCopyBytes.byteOffset%alignof(std::uint64_t)==0,
                "QFN sparse verifier storage requires exact dedicated typed roles/geometry");
            validate(v.prior.storage,sizeof(qwen4exp::QwenSparseTransactionPrior));
            validate(v.priorTraffic,sizeof(qwen4exp::QwenSparsePriorTraffic));
            validate(v.rawKeys,std::size_t(v.rowCapacity)*128*sizeof(float));
            validate(v.candidateCopyBytes,sizeof(std::uint64_t));
        }

        if(params_.device.is_gpu())require(params_.cache.deviceCachedTokenCountPtr(entry.cacheLayer,entry.sequence),
            "QFN auxiliary tail requires its canonical device frontier");
    }
    for(const auto &entry : params_.pleHistories) {
        require(entry.layer>=0 && entry.cacheLayer>=0 && entry.sequence>=0 && entry.sequence<params_.requestCapacity && entry.channels>0,
            "QFN auxiliary lexical history has invalid layer/request/width identity");
        validate(entry.acceptedTokens,8);validate(entry.nextTokens,8);
        validate(entry.acceptedConvolution,std::size_t(9)*entry.channels*4);validate(entry.nextConvolution,std::size_t(9)*entry.channels*4);
        require(entry.acceptedTokens.tensor!=entry.nextTokens.tensor || entry.acceptedTokens.byteOffset+8<=entry.nextTokens.byteOffset ||
            entry.nextTokens.byteOffset+8<=entry.acceptedTokens.byteOffset,"QFN lexical token history banks overlap");
        const auto extent=std::size_t(9)*entry.channels*4;
        require(entry.acceptedConvolution.tensor!=entry.nextConvolution.tensor || entry.acceptedConvolution.byteOffset+extent<=entry.nextConvolution.byteOffset ||
            entry.nextConvolution.byteOffset+extent<=entry.acceptedConvolution.byteOffset,"QFN lexical convolution history banks overlap");
    }
    struct Interval {TensorBase *tensor;BufferId buffer;std::size_t begin,end;};
    std::vector<Interval> intervals;
    const auto seal=[&](auto binding,std::size_t bytes,bool persistent=true) {
        for(const auto &prior:intervals)
            require((prior.tensor!=binding.tensor && prior.buffer!=binding.buffer) || binding.byteOffset>=prior.end || prior.begin>=binding.byteOffset+bytes,
                "QFN auxiliary state banks physically overlap another layer/request/bank");
        intervals.push_back({binding.tensor,binding.buffer,binding.byteOffset,binding.byteOffset+bytes});
        if(persistent&&std::find(owners_.begin(),owners_.end(),binding.tensor)==owners_.end())owners_.push_back(binding.tensor);
    };
    for(const auto &e:params_.sparseLayers) {
        seal(e.pooledKeys,std::size_t((std::int64_t(e.contextTokens)+3)/4)*128*4);seal(e.partialKeys,4*128*4);
        if(e.verifier) {
            const auto& v=*e.verifier;
            seal(v.prior.storage,sizeof(qwen4exp::QwenSparseTransactionPrior),false);
            seal(v.priorTraffic,sizeof(qwen4exp::QwenSparsePriorTraffic),false);
            seal(v.rawKeys,std::size_t(v.rowCapacity)*128*sizeof(float),false);
            seal(v.candidateCopyBytes,sizeof(std::uint64_t),false);
        }

    }
    for(const auto &e:params_.pleHistories) {
        seal(e.acceptedTokens,8);seal(e.nextTokens,8);
        seal(e.acceptedConvolution,std::size_t(9)*e.channels*4);seal(e.nextConvolution,std::size_t(9)*e.channels*4);
        if(e.verifier) {
            const auto&v=*e.verifier;
            require(v.rowCapacity>0&&std::size_t(v.rowCapacity)<=std::numeric_limits<std::size_t>::max()/4/std::size_t(e.channels),
                "QFN PLE verifier candidate geometry overflows");
            require(v.priorTokens.buffer==BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY&&
                v.priorConvolution.buffer==BufferId::QFN_PLE_PRIOR_CONV_HISTORY&&
                v.transactionLive.buffer==BufferId::QFN_PLE_TRANSACTION_LIVE&&
                v.retainedTokens.buffer==BufferId::QFN_PLE_RETAINED_TOKENS&&
                v.retainedNormalized.buffer==BufferId::QFN_PLE_RETAINED_NORMALIZED&&
                v.copiedBytes.buffer==BufferId::QFN_PLE_TRANSACTION_COPY_BYTES,
                "QFN verifier prior must use dedicated admitted PLEMTP roles");
            validate(v.priorTokens,8);validate(v.priorConvolution,std::size_t(9)*e.channels*4);
            validate(v.transactionLive,4);validate(v.retainedTokens,std::size_t(v.rowCapacity)*4);
            validate(v.retainedNormalized,std::size_t(v.rowCapacity)*e.channels*4);validate(v.copiedBytes,16);
            require(v.copiedBytes.byteOffset%8==0,"QFN PLE verifier byte observations must align to8");
            // Ephemeral priors/candidates are never reset/imported/exported or published as accepted state.
            // Their sole writer is the verifier snapshot/retention stage on the canonical transaction DAG.
            seal(v.priorTokens,8,false);seal(v.priorConvolution,std::size_t(9)*e.channels*4,false);
            seal(v.transactionLive,4,false);seal(v.retainedTokens,std::size_t(v.rowCapacity)*4,false);
            seal(v.retainedNormalized,std::size_t(v.rowCapacity)*e.channels*4,false);seal(v.copiedBytes,16,false);
        }
    }
    for(std::size_t i=0;i<params_.sparseLayers.size();++i)for(std::size_t j=0;j<i;++j)
        require(params_.sparseLayers[i].layer!=params_.sparseLayers[j].layer || params_.sparseLayers[i].sequence!=params_.sparseLayers[j].sequence,
            "QFN auxiliary state repeats a sparse layer/request bank");
    for(std::size_t i=0;i<params_.pleHistories.size();++i)for(std::size_t j=0;j<i;++j)
        require(params_.pleHistories[i].sequence!=params_.pleHistories[j].sequence,"QFN auxiliary state repeats a lexical request history");
}

template<typename T,TensorType Type>T *Qwen4ExpAuxiliaryState::pointer(Qwen4ExpArenaBinding<Type> b)const
{
    void *base=params_.device.is_gpu() ? b.tensor->gpu_data_ptr() : b.tensor->raw_mutable_data();
    require(base,"QFN auxiliary state requires already admitted/prepared native storage");
    return reinterpret_cast<T *>(static_cast<std::byte *>(base)+b.byteOffset);
}
const Qwen4ExpAuxiliaryState::SparseLayer &Qwen4ExpAuxiliaryState::sparse(int layer,int sequence)const
{
    for(const auto &entry : params_.sparseLayers)if(entry.layer==layer && entry.sequence==sequence)return entry;
    throw std::invalid_argument("QFN cache does not own the requested sparse layer/request state");
}
ImmutableVerifierPrior Qwen4ExpAuxiliaryState::appendSparseVerifierPrior(ComputeGraph& graph,
    const std::string& prefix,int globalLayer,int sequence,int rowCapacity,Qwen4ExpIntBinding replay,
    const std::vector<std::string>& dependencies,std::optional<QwenSparseCounterBinding> terminalCounters) const
{
    const auto& bank=sparse(globalLayer,sequence);
    if(terminalCounters)terminalCounters->requireRequest(params_.device,sequence);
    require(bank.verifier && rowCapacity>0 && rowCapacity<=bank.verifier->rowCapacity,
        "QFN sparse verifier lacks its exact admitted immutable transaction storage");
    const auto& v=*bank.verifier;
    const auto name=prefix+"qsa_snapshot_prior";
    require(!prefix.empty() && !graph.getNode(name),"QFN sparse prior snapshot producer collides");
    for(const auto& dependency:dependencies) {
        const auto* producer=graph.getNode(dependency);
        require(producer && producer->stage && producer->stage->device()==params_.device,
            "QFN sparse prior requires exact same-participant replay producer dependencies");
    }
    auto stage=std::make_unique<QwenSparseAttentionStage>(QwenSparseAttentionStage::Params{
        .device_id=params_.device,.rowCapacity=rowCapacity,.tokenCapacity=bank.contextTokens,.replay=replay,
        .operation=QwenSparseAttentionStage::SnapshotPrior{bank.partialKeys,v.prior,v.priorTraffic,terminalCounters}});
    graph.addNode(name,std::move(stage),params_.device);
    for(const auto& dependency:dependencies)graph.addDependency(name,dependency);
    return ImmutableVerifierPrior(graph,params_.device,name,{{v.prior.storage,sizeof(qwen4exp::QwenSparseTransactionPrior)}});
}

const Qwen4ExpAuxiliaryState::PLEHistory &Qwen4ExpAuxiliaryState::ple(int sequence)const
{
    for(const auto &entry : params_.pleHistories)if(entry.sequence==sequence)return entry;
    throw std::invalid_argument("QFN cache does not own the requested lexical request history");
}
void Qwen4ExpAuxiliaryState::publish(void *stream)const
{
    if(!params_.device.is_gpu())return;
    for(auto *tensor:owners_)TransferEngine::publishDeviceWrite(tensor,params_.device,stream);
}
void Qwen4ExpAuxiliaryState::initialize(void *stream)
{
    require(!initialized_,"QFN auxiliary state cannot initialize twice");
    require(!isGraphCaptureActive(),"QFN auxiliary state must materialize before graph capture");
    if(params_.device.is_gpu())
    {
        (void)ExplicitGPUStream(stream);
        // Arena registration owns capacity, but first graph materialization
        // normally establishes its device addresses later. Cache setup needs
        // these retained state banks earlier. Materialize their existing sole
        // owners through the same authority as BufferArena, without copying
        // unspecified host bytes. Reset below publishes the initial producer.
        for(auto *tensor:owners_)
            TransferEngine::allocateDeviceStorage(tensor,params_.device);
    }
    else require(!stream,"CPU auxiliary state cannot use a GPU stream");
    reset({0,std::numeric_limits<int>::max(),0,params_.requestCapacity},stream);initialized_=true;
}
void Qwen4ExpAuxiliaryState::requireReady(void *stream)const
{
    require(initialized_,"QFN auxiliary state must initialize before capture");
    if(!params_.device.is_gpu()) {require(!stream,"CPU auxiliary state has no GPU producer event");return;}
    for(auto *tensor:owners_)TransferEngine::requireDeviceInput(tensor,params_.device,stream);
}
void Qwen4ExpAuxiliaryState::reset(CacheAuxiliaryResetRange range,void *stream)
{
    require(range.firstLayer>=0 && range.layerCount>0 && range.firstSequence>=0 && range.sequenceCount>0 &&
        std::int64_t(range.firstSequence)+range.sequenceCount<=params_.requestCapacity,"QFN auxiliary reset range is invalid");
    const auto contains=[&](int layer,int sequence) {return layer>=range.firstLayer && std::int64_t(layer)<std::int64_t(range.firstLayer)+range.layerCount &&
        sequence>=range.firstSequence && sequence<range.firstSequence+range.sequenceCount;};
    for(const auto &e : params_.sparseLayers)if(contains(e.layer,e.sequence))fill(params_.device,{pointer<void>(e.partialKeys),4*128,0},stream);
    for(const auto &e : params_.pleHistories)if(contains(e.layer,e.sequence)) {
        fill(params_.device,{pointer<void>(e.acceptedTokens),2,std::uint32_t(params_.eosToken)},stream);
        fill(params_.device,{pointer<void>(e.nextTokens),2,std::uint32_t(params_.eosToken)},stream);
        fill(params_.device,{pointer<void>(e.acceptedConvolution),std::size_t(9)*e.channels,0},stream);
        fill(params_.device,{pointer<void>(e.nextConvolution),std::size_t(9)*e.channels,0},stream);
    }
    // Pool bytes are intentionally unspecified beyond the zeroed/restored
    // canonical frontier; newly completed pools overwrite before any read.
    publish(stream);
}
std::size_t Qwen4ExpAuxiliaryState::checkpointCapacity(int sequence)const
{
    require(sequence>=0 && sequence<params_.requestCapacity,"QFN auxiliary checkpoint request is invalid");
    std::size_t bytes=0;
    for(const auto &e : params_.sparseLayers)if(e.sequence==sequence)bytes+=TailRecord;
    for(const auto &e : params_.pleHistories)if(e.sequence==sequence)bytes+=CheckpointGeometry::pleHistoryBytes(e.channels);
    return bytes;
}
void Qwen4ExpAuxiliaryState::captureCheckpoint(int sequence,CacheAuxiliaryMemory destination,void *stream)const
{
    require(destination.data && destination.capacity>=checkpointCapacity(sequence) &&
        destination.domain==(params_.device.is_gpu() ? CacheAuxiliaryMemoryDomain::Device : CacheAuxiliaryMemoryDomain::Host),
        "QFN rollback checkpoint must retain admitted native execution-device capacity");
    auto *cursor=static_cast<std::byte *>(destination.data);
    for(const auto &e : params_.sparseLayers)if(e.sequence==sequence) {
        const int cpuCount=params_.device.is_cpu() ? params_.cache.get_cached_tokens(e.cacheLayer,sequence) : 0;
        const int *count=params_.device.is_gpu() ? params_.cache.deviceCachedTokenCountPtr(e.cacheLayer,sequence) : &cpuCount;
        transfer(params_.device,{pointer<void>(e.partialKeys),cursor+TailHeader,128*4,3,
            qwen4exp::QwenStateRowSource::CanonicalModuloFour,count,0,reinterpret_cast<int *>(cursor)},stream);
        cursor+=TailRecord;
    }
    for(const auto &e : params_.pleHistories)if(e.sequence==sequence) {
        transfer(params_.device,{pointer<void>(e.acceptedTokens),cursor,8,1,qwen4exp::QwenStateRowSource::Fixed,nullptr,1,nullptr},stream);cursor+=8;
        const auto bytes=std::size_t(9)*e.channels*4;
        transfer(params_.device,{pointer<void>(e.acceptedConvolution),cursor,bytes,1,qwen4exp::QwenStateRowSource::Fixed,nullptr,1,nullptr},stream);cursor+=bytes;
    }
}
void Qwen4ExpAuxiliaryState::restoreCheckpoint(int sequence,CacheAuxiliaryMemory source,void *stream)
{
    require(source.data && source.capacity>=checkpointCapacity(sequence) &&
        source.domain==(params_.device.is_gpu() ? CacheAuxiliaryMemoryDomain::Device : CacheAuxiliaryMemoryDomain::Host),
        "QFN rollback restore must consume its native opaque execution-device checkpoint");
    auto *cursor=static_cast<std::byte *>(source.data);
    for(const auto &e : params_.sparseLayers)if(e.sequence==sequence) {
        transfer(params_.device,{cursor+TailHeader,pointer<void>(e.partialKeys),128*4,3,
            qwen4exp::QwenStateRowSource::CheckpointHeader,reinterpret_cast<const int *>(cursor),0,nullptr},stream);cursor+=TailRecord;
    }
    for(const auto &e : params_.pleHistories)if(e.sequence==sequence) {
        transfer(params_.device,{cursor,pointer<void>(e.acceptedTokens),8,1,qwen4exp::QwenStateRowSource::Fixed,nullptr,1,nullptr},stream);cursor+=8;
        const auto bytes=std::size_t(9)*e.channels*4;
        transfer(params_.device,{cursor,pointer<void>(e.acceptedConvolution),bytes,1,qwen4exp::QwenStateRowSource::Fixed,nullptr,1,nullptr},stream);cursor+=bytes;
    }
    publish(stream);
}
void Qwen4ExpAuxiliaryState::archiveCopy(void *destination,const void *source,std::size_t bytes,
    CacheAuxiliaryMemoryDomain domain,bool exportData,void *stream)const
{
    if(!bytes)return;
    require(destination && source,"QFN archive native payload pointer is missing");
    if(params_.device.is_cpu()) { require(domain==CacheAuxiliaryMemoryDomain::Host && !stream,"CPU QFN archive requires host payload authority");std::memmove(destination,source,bytes);return; }
    IKVCache::requireGPUExecutionStream(stream,"QFN auxiliary prefix archive");
    auto *backend=getBackendFor(params_.device);require(backend,"QFN archive backend is unavailable");
    // Prefix owners retain the destination and publish their exact ready event
    // after every payload section is enqueued. A per-section blocking wrapper
    // would allocate/wait an event and hide this explicit consumer ordering.
    const bool ok=domain==CacheAuxiliaryMemoryDomain::Device ? backend->deviceCopyAsync(destination,source,bytes,params_.device.ordinal,stream) :
        exportData ? backend->deviceToHostOnStream(destination,source,bytes,params_.device.ordinal,stream) :
        backend->hostToDeviceOnStream(destination,source,bytes,params_.device.ordinal,stream);
    require(ok,"QFN exact native archive transfer failed");
}
std::size_t Qwen4ExpAuxiliaryState::blockArchiveBytes(CacheAuxiliaryBlockRange range)const
{
    require(range.sequence>=0 && range.sequence<params_.requestCapacity && range.firstToken>=0 && range.tokenCount>=0 &&
        std::int64_t(range.firstToken)+range.tokenCount<=params_.cache.max_seq_len(),"QFN auxiliary archive range exceeds admitted context");
    for(const auto &e : params_.sparseLayers)if(e.layer==range.layer && e.sequence==range.sequence)
        return qwen4exp::Qwen4ExpArchiveGeometry::blockBytes(range.firstToken,range.tokenCount);
    return 0;
}
void Qwen4ExpAuxiliaryState::exportBlock(CacheAuxiliaryBlockRange range,CacheAuxiliaryMemory destination,void *stream)const
{
    const auto bytes=blockArchiveBytes(range);require(destination.capacity>=bytes,"QFN pool archive destination is undersized");if(!bytes)return;
    const auto &e=sparse(range.layer,range.sequence);
    archiveCopy(destination.data,pointer<float>(e.pooledKeys)+std::size_t(range.firstToken/4)*128,bytes,destination.domain,true,stream);
}
void Qwen4ExpAuxiliaryState::importBlock(CacheAuxiliaryBlockRange range,CacheAuxiliaryMemory source,void *stream)
{
    const auto bytes=blockArchiveBytes(range);require(source.capacity>=bytes,"QFN pool archive source is undersized");if(!bytes)return;
    const auto &e=sparse(range.layer,range.sequence);
    archiveCopy(pointer<float>(e.pooledKeys)+std::size_t(range.firstToken/4)*128,source.data,bytes,source.domain,false,stream);publish(stream);
}
std::size_t Qwen4ExpAuxiliaryState::endpointArchiveBytes(int sequence,int logicalTokens)const
{
    require(sequence>=0 && sequence<params_.requestCapacity && logicalTokens>=0 && logicalTokens<=params_.cache.max_seq_len(),
        "QFN endpoint archive prefix length/request is invalid");
    return archiveGeometry(sequence).endpointBytes(logicalTokens);
}
qwen4exp::Qwen4ExpArchiveGeometry Qwen4ExpAuxiliaryState::archiveGeometry(int sequence) const
{
    require(sequence>=0 && sequence<params_.requestCapacity,"QFN archive geometry request is outside admitted capacity");
    qwen4exp::Qwen4ExpArchiveGeometry geometry;
    for(const auto &entry:params_.sparseLayers)if(entry.sequence==sequence)++geometry.sparse_layers;
    for(const auto &entry:params_.pleHistories)if(entry.sequence==sequence) {
        require(geometry.ple_channels==0,"QFN archive geometry has duplicate lexical history owners");
        geometry.ple_channels=entry.channels;
    }
    return geometry;
}
void Qwen4ExpAuxiliaryState::exportEndpoint(int sequence,int logicalTokens,CacheAuxiliaryMemory destination,void *stream)const
{
    require(destination.capacity>=endpointArchiveBytes(sequence,logicalTokens),"QFN compact endpoint archive is undersized");
    auto *cursor=static_cast<std::byte *>(destination.data);
    for(const auto &e : params_.sparseLayers)if(e.sequence==sequence) {
        const auto bytes=std::size_t(logicalTokens%4)*128*4;
        archiveCopy(cursor,pointer<void>(e.partialKeys),bytes,destination.domain,true,stream);if(bytes)cursor+=bytes;
    }
    for(const auto &e : params_.pleHistories)if(e.sequence==sequence) {
        archiveCopy(cursor,pointer<void>(e.acceptedTokens),8,destination.domain,true,stream);cursor+=8;
        const auto bytes=std::size_t(9)*e.channels*4;
        archiveCopy(cursor,pointer<void>(e.acceptedConvolution),bytes,destination.domain,true,stream);cursor+=bytes;
    }
}
void Qwen4ExpAuxiliaryState::importEndpoint(int sequence,int logicalTokens,CacheAuxiliaryMemory source,void *stream)
{
    require(source.capacity>=endpointArchiveBytes(sequence,logicalTokens),"QFN compact endpoint restore is undersized");
    auto *cursor=static_cast<std::byte *>(source.data);
    for(const auto &e : params_.sparseLayers)if(e.sequence==sequence) {
        const auto bytes=std::size_t(logicalTokens%4)*128*4;
        archiveCopy(pointer<void>(e.partialKeys),cursor,bytes,source.domain,false,stream);if(bytes)cursor+=bytes;
    }
    for(const auto &e : params_.pleHistories)if(e.sequence==sequence) {
        archiveCopy(pointer<void>(e.acceptedTokens),cursor,8,source.domain,false,stream);cursor+=8;
        const auto bytes=std::size_t(9)*e.channels*4;
        archiveCopy(pointer<void>(e.acceptedConvolution),cursor,bytes,source.domain,false,stream);cursor+=bytes;
    }
    publish(stream);
}
std::string Qwen4ExpAuxiliaryState::appendAcceptedSparseCommit(ComputeGraph &graph,const std::string &prefix,int sequence,
    const std::vector<SparseCandidates> &candidates,Qwen4ExpIntBinding replay,Qwen4ExpIntInput acceptedRows,
    const std::vector<std::string> &dependencies)const
{
    require(!prefix.empty() && !candidates.empty() && !dependencies.empty(),"QFN accepted sparse publication requires candidates and exact producer edges");
    std::vector<std::pair<std::string,std::unique_ptr<QwenSparseAttentionStage>>> pending;
    for(const auto &candidate : candidates) {
        require(candidate.sequence==sequence && candidate.rowCapacity>0 && candidate.rawKeys.buffer==BufferId::QFN_QSA_CANDIDATE_RAW_KEYS,
            "QFN accepted sparse publication requires exact retained verifier candidates");
        const auto &state=sparse(candidate.layer,sequence);
        const auto name=prefix+"qsa_commit_"+std::to_string(candidate.layer);
        require(!graph.getNode(name),"QFN accepted sparse node identity collides");
        pending.emplace_back(name,std::make_unique<QwenSparseAttentionStage>(QwenSparseAttentionStage::Params{
            .device_id=params_.device,.rowCapacity=candidate.rowCapacity,.tokenCapacity=state.contextTokens,.replay=replay,
            .operation=QwenSparseAttentionStage::CommitPartial{state.partialKeys,candidate.rawKeys,acceptedRows,state.partialKeys}}));
    }
    for(const auto &dependency : dependencies)require(graph.getNode(dependency),"QFN accepted sparse publication is missing a producer");
    std::string previous;
    for(auto &[name,stage] : pending) {
        graph.addNode(name,std::move(stage),params_.device);
        for(const auto &dependency : dependencies)graph.addDependency(name,dependency);
        if(!previous.empty())graph.addDependency(name,previous);previous=name;
    }
    return previous;
}

namespace {
/** @brief Seal one single-request verifier stage from the actual auxiliary storage authority. */
Qwen4ExpPLERetainedStage::Params pleVerifierParams(DeviceId device,
 const Qwen4ExpAuxiliaryState::PLEHistory&history,Qwen4ExpStageRows rows,
 Qwen4ExpIntInput tokens,Qwen4ExpFloatBinding normalized,Qwen4ExpPLERetainedStage::Operation operation)
{
 const auto&v=*history.verifier;auto counter=v.copiedBytes;
 if(operation==Qwen4ExpPLERetainedStage::Operation::RetainCandidates)counter.byteOffset+=8;
 return {.device_id=device,.operation=operation,.requests=1,.requestIndex=0,.rowCapacity=rows.capacity,
  .channels=history.channels,.liveCounts=rows.liveRows,.transactionLiveCounts=v.transactionLive,
  .acceptedTokens=history.acceptedTokens,.priorTokens=v.priorTokens,.retainedTokens=v.retainedTokens,
  .tokens=std::move(tokens),.acceptedConvolution=history.acceptedConvolution,
  .priorConvolution=v.priorConvolution,.normalized=normalized,.retainedNormalized=v.retainedNormalized,
  .copiedBytes=counter,.verifierOrigins=RequestRowOrigins::fixedStride(0,rows.capacity)};
}
}
Qwen4ExpAuxiliaryState::PLEVerifierPrior Qwen4ExpAuxiliaryState::appendVerifierPLEPrior(
 ComputeGraph&graph,const std::string&prefix,int sequence,Qwen4ExpStageRows rows,
 Qwen4ExpIntInput candidateTokens,Qwen4ExpFloatBinding candidateNormalized,
 const std::vector<std::string>&dependencies)const
{
 const auto&history=ple(sequence);
 require(sequence==0&&history.verifier&&rows.capacity>0&&rows.capacity<=history.verifier->rowCapacity,
  "QFN grouped PLE prior requires admitted single-request verifier storage");
 // A PLE-only pipeline participant begins with this snapshot. The canonical
 // pipeline composer encloses every model root with metadata/activation edges
 // after this fragment is built. Do not invent an earlier local producer, or
 // permit an omitted edge once another model node has already been declared.
 require(!prefix.empty()&&(!dependencies.empty()||graph.size()==0),
  "QFN PLE prior requires an empty graph entry or exact native producer dependencies");
 const auto name=prefix+"transaction_prior";require(!graph.getNode(name),"QFN PLE prior producer name collides");
 for(const auto&dependency:dependencies) {
  const auto*producer=graph.getNode(dependency);
  require(producer&&producer->stage&&producer->stage->device()==params_.device,
   "QFN PLE prior is missing its actual same-participant producer");
 }
 auto stage=std::make_unique<Qwen4ExpPLERetainedStage>(pleVerifierParams(params_.device,history,rows,
  std::move(candidateTokens),candidateNormalized,Qwen4ExpPLERetainedStage::Operation::SnapshotPrior));
 graph.addNode(name,std::move(stage),params_.device);for(const auto&dependency:dependencies)graph.addDependency(name,dependency);
 const auto&v=*history.verifier;
 return {sequence,std::move(rows),v.priorTokens,v.priorConvolution,
  ImmutableVerifierPrior(graph,params_.device,name,{{v.priorTokens,8},{v.priorConvolution,std::size_t(history.channels)*36},{v.transactionLive,4}})};
}
std::string Qwen4ExpAuxiliaryState::appendVerifierPLERetain(ComputeGraph&graph,const std::string&prefix,
 PLEVerifierPrior&&prior,Qwen4ExpIntInput candidateTokens,Qwen4ExpFloatBinding candidateNormalized,
 const std::vector<std::string>&dependencies)const
{
 prior.capability_.requireFor(graph,params_.device);const auto&history=ple(prior.sequence_);
 require(history.verifier&&prior.rows_.capacity<=history.verifier->rowCapacity&&!prefix.empty()&&!dependencies.empty(),
  "QFN PLE retention requires its original admitted transaction capability");
 const auto name=prefix+"retain_candidates";require(!graph.getNode(name),"QFN PLE retention producer name collides");
 for(const auto&dependency:dependencies)require(graph.getNode(dependency),"QFN PLE retention is missing normalized/token producer");
 auto stage=std::make_unique<Qwen4ExpPLERetainedStage>(pleVerifierParams(params_.device,history,prior.rows_,
  std::move(candidateTokens),candidateNormalized,Qwen4ExpPLERetainedStage::Operation::RetainCandidates));
 graph.addNode(name,std::move(stage),params_.device);for(const auto&dependency:dependencies)graph.addDependency(name,dependency);
 return std::move(prior.capability_).retainForPublication(graph,name);
}
std::string Qwen4ExpAuxiliaryState::appendAcceptedPLECommit(ComputeGraph &graph,const std::string &prefix,int sequence,
    Qwen4ExpStageRows rows,Qwen4ExpStageAcceptance acceptance,Qwen4ExpIntInput candidateTokens,
    Qwen4ExpFloatBinding candidateNormalized,const std::vector<std::string> &dependencies)const
{
    const auto &history=ple(sequence);
    require(!prefix.empty() && !dependencies.empty(),"QFN lexical accepted publication requires exact producer edges");
    const auto conv=prefix+"ple_commit_convolution",tokens=prefix+"ple_commit_tokens",promote=prefix+"ple_promote";
    for(const auto &name : {conv,tokens,promote})require(!graph.getNode(name),"QFN lexical publication node identity collides");
    for(const auto &dependency : dependencies)require(graph.getNode(dependency),"QFN lexical publication is missing a producer");
    auto convStage=std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{.device_id=params_.device,.rows=rows,
        .operation=Qwen4ExpPrimitiveStage::CommitConvHistory{history.channels,acceptance,history.acceptedConvolution,candidateNormalized,history.nextConvolution}});
    auto tokenStage=std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{.device_id=params_.device,.rows=rows,
        .operation=Qwen4ExpPrimitiveStage::CommitTokenHistory{acceptance,history.acceptedTokens,candidateTokens,history.nextTokens}});
    auto promotion=std::make_unique<PLEPromotion>(params_.device,history);
    graph.addNode(conv,std::move(convStage),params_.device);graph.addNode(tokens,std::move(tokenStage),params_.device);
    graph.addNode(promote,std::move(promotion),params_.device);
    for(const auto &dependency : dependencies) {graph.addDependency(conv,dependency);graph.addDependency(tokens,dependency);}
    graph.addDependency(promote,conv);graph.addDependency(promote,tokens);return promote;
}
} // namespace llaminar2
