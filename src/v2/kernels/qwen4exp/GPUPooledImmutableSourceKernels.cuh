/**
 * @file GPUPooledImmutableSourceKernels.cuh
 * @brief Captured actual RAM/SSD request, acquired ACK/payload copy and original-order native consumer.
 *
 * One source metadata writer forms compact first-occurrence messages from the
 * actual coordinator and immutable accepted owner grants. Independent byte
 * copies and consumer validation/decode are parallel. No host count/state read,
 * hot allocation, callback, recapture or alternate provider exists in this path.
 * Factory setup binds the standard deadline from the admitted physical clock.
 */
#pragma once
#if defined(QFN_POOLED_SOURCE_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#define QFN_SOURCE_STREAM cudaStream_t
#define QFN_SOURCE_NAMESPACE cuda_pooled_source
#define QFN_SOURCE_FACTORY createCUDAPooledImmutableSourceKernels
#define QFN_SOURCE_ERROR cudaGetLastError
#define QFN_SOURCE_SUCCESS cudaSuccess
#define QFN_SOURCE_ERROR_TEXT cudaGetErrorString
#else
#include <hip/hip_runtime.h>
#include "ImmutableLookupHardwareBinding.h"
#include <hip/hip_fp16.h>
#define QFN_SOURCE_STREAM hipStream_t
#define QFN_SOURCE_NAMESPACE rocm_pooled_source
#define QFN_SOURCE_FACTORY createROCmPooledImmutableSourceKernels
#define QFN_SOURCE_ERROR hipGetLastError
#define QFN_SOURCE_SUCCESS hipSuccess
#define QFN_SOURCE_ERROR_TEXT hipGetErrorString
#endif
#include "PooledImmutableSourceKernels.h"
#include "ImmutableLookupDecode.h"
#include "../common/SamplingMath.h"
#include <algorithm>
#include <stdexcept>
#include <string>
namespace llaminar2::qwen4exp {
namespace QFN_SOURCE_NAMESPACE {
namespace s=pooled_source;
namespace n=pooled_native;
namespace p=n::p;
using s::DeviceArgs;
using s::ResponseStage;
/** @brief System acquire of one actual source publication; ordinary cached loads cannot substitute. */
__device__ __forceinline__ std::uint64_t acquire(const std::uint64_t* pointer){
#if defined(QFN_POOLED_SOURCE_CUDA)
    cuda::atomic_ref<std::uint64_t,cuda::thread_scope_system> word(*const_cast<std::uint64_t*>(pointer));
    return word.load(cuda::memory_order_acquire);
#else
    return __hip_atomic_load(pointer,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @brief Release mapped readers/writers before publishing the exact original protocol word. */
__device__ __forceinline__ void release(std::uint64_t* pointer,std::uint64_t value){
#if defined(QFN_POOLED_SOURCE_CUDA)
    cuda::atomic_ref<std::uint64_t,cuda::thread_scope_system> word(*pointer);word.store(value,cuda::memory_order_release);
#else
    __hip_atomic_store(pointer,value,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @return Fresh source response bytes after acquire, avoiding stale device cache lines. */
template<class T> __device__ __forceinline__ T fresh(const T* pointer){
#if defined(QFN_POOLED_SOURCE_CUDA)
    return __ldcv(pointer);
#else
    return __builtin_nontemporal_load(pointer);
#endif
}
/** @return Fixed-rate backend wall clock for the standard thirty-second source deadline. */
__device__ __forceinline__ std::uint64_t clock(){
#if defined(QFN_POOLED_SOURCE_CUDA)
    std::uint64_t value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
#else
    return wall_clock64();
#endif
}
/** @brief Publish only the canonical request's first source error. */
__device__ __forceinline__ void unhealthy(const DeviceArgs& a){
    if(*a.error==0)*a.error=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);
    *a.healthy=0;
}
/** @return Worker retirement for exactly this original admitted generation and actual ticket. */
__device__ __forceinline__ bool retired(const DeviceArgs& a){
    const auto invocation=acquire(&a.channel.base.acknowledgement->retiredInvocation);
    const auto generation=acquire(&a.channel.base.acknowledgement->retiredGeneration);
    return ImmutableLookupProtocol::matchesInvocation(invocation,generation,s::protocol(a).admissionGeneration,s::invocation(a));
}
/** @brief Actual system-scope source publications, preserving the shared completion/retirement load order. */
struct ResponsePublications {
    const DeviceArgs& a;
    /** @return Fresh actual worker completion publication. */
    __device__ __forceinline__ std::uint64_t completion()const{return acquire(&a.channel.base.completion->publication);}
    /** @return Acquired retirement for this exact admitted generation/invocation only. */
    __device__ __forceinline__ bool retirement()const{return retired(a);}
};
/** @brief A retired worker without a matching acquired envelope permits only cancellation, never Consumed. */
__device__ __forceinline__ void aborted(const DeviceArgs& a){
    const auto status=s::protocol(a).status?static_cast<ImmutableLookupStatus>(s::protocol(a).status):ImmutableLookupStatus::Cancelled;
    s::reject(a,status);s::frontier(a).disposition=s::frontier(a).source.table?n::SourceDisposition::Aborted:n::SourceDisposition::Unpublished;
    s::stage(a)=ResponseStage::Aborted;unhealthy(a);
}
/** @brief Initialize only protocol/source metadata; inactive packet/payload banks remain untouched. */
__global__ void initialize(DeviceArgs a){
    s::protocol(a)={};s::frontier(a)={};s::invocation(a)=0;s::stage(a)=ResponseStage::Idle;
    *n::at<QwenAdaptiveCounterTotals>(a.native,a.plan.offsets.observation)={};
    *n::at<std::uint64_t>(a.native,a.plan.offsets.observation_ordinal)=0;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.reported_ordinal)=0;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.record_bytes)=0;
    *n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes)=0;
}
/** @brief Publish only original immutable service admission; request reset preserves all sequence history. */
__global__ void admit(DeviceArgs a,std::uint64_t generation,std::uint64_t invocation){
    auto& io=s::protocol(a);
    if(!generation || !invocation || generation<=io.admissionGeneration || io.issuedSequence!=io.consumedSequence ||
       (io.status && io.status!=static_cast<std::uint64_t>(ImmutableLookupStatus::Cancelled)) ||
       (s::stage(a)!=ResponseStage::Idle && s::stage(a)!=ResponseStage::ReadersComplete)) {
#if defined(QFN_POOLED_SOURCE_CUDA)
        __trap();
#else
        __builtin_trap();
#endif
        return;
    }
    io.admissionGeneration=io.generation=generation;io.idCount=io.nativeBytes=io.status=0;
    s::invocation(a)=invocation;
}
/** @brief Form exact compact source intents, preserving original stable unique and accepted-grant order. */
__global__ void plan(DeviceArgs a){
    if(!s::planRequest(a)){
        unhealthy(a);
        if(!s::rejectedPlan(a)){
#if defined(QFN_POOLED_SOURCE_CUDA)
            __trap();
#else
            __builtin_trap();
#endif
        }
    }
}
/** @brief Issue only after both actual owner/coordinator source seals, never before accepted waves drain. */
__global__ void publish(DeviceArgs a){
    auto& io=s::protocol(a);auto& frontier=s::frontier(a);
    const auto& owner=*n::at<p::OwnerFrame>(a.owner.frame,a.plan.owner.frame.control);
    const auto& coordinator=*n::coordinatorView(a.coordinator).frame;
    if(s::stage(a)==ResponseStage::Rejected)return;
    if(s::stage(a)!=ResponseStage::Planned || owner.phase!=p::OwnerPhase::Filling ||
       coordinator.phase!=p::LookupPhase::Source || !n::same(owner.source,frontier.source) ||
       !n::same(coordinator.source,frontier.source)) {s::reject(a,ImmutableLookupStatus::InvalidProtocol);s::stage(a)=ResponseStage::Rejected;unhealthy(a);return;}
    if(!*a.healthy){s::reject(a,ImmutableLookupStatus::Cancelled);s::stage(a)=ResponseStage::Rejected;return;}
    if(!io.idCount){s::stage(a)=ResponseStage::Acquired;return;}
    if(retired(a)){aborted(a);return;}
    const auto body=s::requestBody(a);const auto* from=reinterpret_cast<const std::uint64_t*>(&body);
    auto* to=reinterpret_cast<std::uint64_t*>(a.channel.base.request);
    for(unsigned word=1;word<kLookupRequestLiveBytes/sizeof(std::uint64_t);++word)to[word]=from[word];
    io.issuedSequence=body.publication;s::stage(a)=ResponseStage::Published;
    release(&a.channel.base.request->publication,body.publication);
}
/** @brief Acquire exact completion metadata; a worker witness without completion stays a distinct aborted outcome. */
__global__ void acquireResponse(DeviceArgs a,std::uint64_t timeout){
    __shared__ __align__(64) std::uint64_t metadata[sizeof(ImmutableLookupCompletion)/sizeof(std::uint64_t)];
    auto& completion=*reinterpret_cast<ImmutableLookupCompletion*>(metadata);
    __shared__ int copy;if(threadIdx.x==0)copy=0;__syncthreads();
    if(threadIdx.x==0 && s::stage(a)==ResponseStage::Published){
        const auto start=clock();const auto expected=s::protocol(a).issuedSequence;
        for(;;){
            ResponsePublications publications{a};
            const auto observation=s::acquireResponsePublication(publications,expected);
            if(observation==s::ResponsePublication::Completion){copy=1;break;}
            if(observation==s::ResponsePublication::WorkerRetired){aborted(a);break;}
            if(clock()-start>=timeout){s::reject(a,ImmutableLookupStatus::TimedOut);s::stage(a)=ResponseStage::Rejected;unhealthy(a);break;}
        }
    }
    __syncthreads();if(!copy)return;
    if(threadIdx.x<kLookupCompletionLiveBytes/sizeof(std::uint64_t))
        reinterpret_cast<std::uint64_t*>(&completion)[threadIdx.x]=fresh(reinterpret_cast<const std::uint64_t*>(a.channel.base.completion)+threadIdx.x);
    __syncthreads();
    if(threadIdx.x==0){
        const auto publication=acquire(&a.channel.base.completion->publication);
        if(!s::acquiredCompletion(a,completion,publication)){
            // Discarded malformed metadata is not a successful source window.
            // The final retirement producer waits for this exact worker after
            // response-byte readers release their original mapped interval.
            s::reject(a,ImmutableLookupStatus::InvalidProtocol);s::stage(a)=ResponseStage::Rejected;
        }else{
            auto& io=s::protocol(a);io.status=completion.status;io.nativeBytes=completion.nativeBytes;
            s::stage(a)=completion.status==0?ResponseStage::Acquired:ResponseStage::Private;
            if(completion.status){s::reject(a,static_cast<ImmutableLookupStatus>(completion.status));unhealthy(a);}
        }
    }
}
/** @brief Copy only actual successfully acquired live native rows and ACKs; partial failed responses are never read. */
__global__ void copyResponse(DeviceArgs a){
    if(s::stage(a)!=ResponseStage::Acquired || s::protocol(a).status)return;
    const auto bytes=s::protocol(a).nativeBytes;auto* to=n::at<std::byte>(a.frame,a.plan.offsets.responses);
    for(std::uint64_t index=std::uint64_t{blockIdx.x}*blockDim.x+threadIdx.x;index<bytes;index+=std::uint64_t{gridDim.x}*blockDim.x)
        to[index]=static_cast<std::byte>(fresh(reinterpret_cast<const unsigned char*>(a.channel.base.nativeRows)+index));
    auto* acknowledgements=n::at<ImmutableRowInvalidationAcknowledgement>(a.frame,a.plan.offsets.invalidations);
    for(std::uint64_t index=std::uint64_t{blockIdx.x}*blockDim.x+threadIdx.x;index<s::protocol(a).idCount;index+=std::uint64_t{gridDim.x}*blockDim.x)
        acknowledgements[index].destination_grant=fresh(&a.channel.acknowledgements[index].destination_grant);
}
/** @brief Validate every private ACK before emitting any owner Fill record or payload extent. */
__global__ void prepareDelivery(DeviceArgs a){
    if(s::stage(a)!=ResponseStage::Acquired)return;
    if(!s::prepareDelivery(a))unhealthy(a);
}
/** @brief Copy exact unchanged private source rows into original accepted grant order, excluding inactive capacity. */
__global__ void packetRows(DeviceArgs a){
    if(s::stage(a)!=ResponseStage::Acquired || s::protocol(a).status)return;
    const auto bytes=*n::at<std::uint64_t>(a.native,a.plan.offsets.payload_bytes);
    auto* to=n::at<std::byte>(a.frame,a.plan.offsets.payload);
    for(std::uint64_t index=std::uint64_t{blockIdx.x}*blockDim.x+threadIdx.x;index<bytes;index+=std::uint64_t{gridDim.x}*blockDim.x)
        to[index]=s::deliveryRow(a,static_cast<std::uint32_t>(index/a.plan.owner.row_bytes))[index%a.plan.owner.row_bytes];
}
/** @brief Retire actual mapped readers, then publish a truthful consumed or exact-worker-aborted source outcome. */
__global__ void retireResponse(DeviceArgs a,std::uint64_t timeout){
    auto& io=s::protocol(a);auto& frontier=s::frontier(a);const auto state=s::stage(a);
    if(state==ResponseStage::Aborted)return;
    if(state!=ResponseStage::Acquired && state!=ResponseStage::Private && state!=ResponseStage::Rejected)return;
    if(!io.idCount && state!=ResponseStage::Rejected){s::stage(a)=ResponseStage::Consumed;return;}
    if(state==ResponseStage::Rejected){
        // These words have one GPU writer. They stop only this admitted I/O
        // invocation; neither Consumed nor complete-parent close is fabricated.
        if(!retired(a)){
            release(&a.channel.base.acknowledgement->abortGeneration,io.admissionGeneration);
            release(&a.channel.base.acknowledgement->abortInvocation,s::invocation(a));
        }
        const auto start=clock();while(!retired(a)){
            if(clock()-start>=timeout){
                s::reject(a,ImmutableLookupStatus::TimedOut);unhealthy(a);
                // No retirement witness exists: accepted grants and roots stay
                // retained. A fatal source failure cannot impersonate Aborted.
#if defined(QFN_POOLED_SOURCE_CUDA)
                __trap();
#else
                __builtin_trap();
#endif
                return;
            }
        }
        aborted(a);return;
    }
    // Only an authenticated acquired envelope can release response bytes and
    // authorize Consumed. All independent private copies precede this edge.
    release(&a.channel.base.acknowledgement->consumedSequence,io.issuedSequence);io.consumedSequence=io.issuedSequence;
    frontier.disposition=n::SourceDisposition::Consumed;s::stage(a)=ResponseStage::Consumed;
    if(io.status || !*a.healthy){s::reject(a,io.status?static_cast<ImmutableLookupStatus>(io.status):ImmutableLookupStatus::Cancelled);unhealthy(a);}
}
/** @brief Observe canonical cancellation after ACK/fill and before the owner Ready edge. */
__global__ void beforeReady(DeviceArgs a){(void)s::beforeReady(a);}
/** @brief Authenticate all consumer rows in parallel before any output writer can run. */
template<class Slot> __global__ void validateReaders(DeviceArgs a){
    if(s::frontier(a).status!=adaptive_lookup::Status::Success || !*a.healthy)return;
    const auto unique=n::coordinatorView(a.coordinator).frame->unique_rows;
    for(std::uint64_t index=std::uint64_t{blockIdx.x}*blockDim.x+threadIdx.x;index<unique;index+=std::uint64_t{gridDim.x}*blockDim.x)
        if(!s::readableRow<Slot>(a,static_cast<std::uint32_t>(index))){
            // Integer failure publication is commutative. Slot/directory state
            // remains read-only; one following producer owns the source outcome.
            atomicExch(a.healthy,0);atomicCAS(a.error,0,static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
        }
}
/** @brief One source outcome writer incorporates the canonical failure before any output write. */
__global__ void readerOutcome(DeviceArgs a,PooledSourceQuery query){
    if(n::coordinatorView(a.coordinator).frame->original_ids>query.elements/a.plan.row_elements){
        s::reject(a,ImmutableLookupStatus::InvalidExtent);unhealthy(a);
    }
    if(!*a.healthy && s::frontier(a).status==adaptive_lookup::Status::Success)s::reject(a,ImmutableLookupStatus::Cancelled);
}
/** @brief Decode only original live positions through the unchanged stable coalescing permutation. */
template<TensorType Format,class Slot> __global__ void read(DeviceArgs a,PooledSourceQuery query,NativeEmbeddingIQTables tables){
    if(!*a.healthy || s::frontier(a).status!=adaptive_lookup::Status::Success)return;
    const auto view=n::coordinatorView(a.coordinator);const auto count=std::uint64_t{view.frame->original_ids}*a.plan.row_elements;
    for(std::uint64_t index=std::uint64_t{blockIdx.x}*blockDim.x+threadIdx.x;index<count;index+=std::uint64_t{gridDim.x}*blockDim.x){
        const auto unique=view.original_to_unique[index/a.plan.row_elements];
        const auto* row=s::readableRow<Slot>(a,unique);
        if(row)query.output[index]=lookup_detail::decode<Format>(row,index%a.plan.row_elements,tables);
    }
}
/** @brief Only the explicit last-reader edge permits later source-frame reuse; graph owners still retain physical backing. */
__global__ void readersRetired(DeviceArgs a){
    const bool observed=a.plan.owner.owner.rowWidth()==adaptive_lookup::RowKeyWidth::UInt32?
        s::snapshotObservation<adaptive_lookup::Slot32>(a):s::snapshotObservation<adaptive_lookup::Slot64>(a);
    if(!observed){s::reject(a,ImmutableLookupStatus::InvalidProtocol);unhealthy(a);return;}
    if(!*a.healthy && s::frontier(a).status==adaptive_lookup::Status::Success)s::reject(a,ImmutableLookupStatus::Cancelled);
    if(s::stage(a)==ResponseStage::Consumed || s::stage(a)==ResponseStage::Aborted)s::stage(a)=ResponseStage::ReadersComplete;
    else {s::reject(a,ImmutableLookupStatus::InvalidProtocol);unhealthy(a);}
}
/** @brief Commit actual reporting only after original reader/owner/coordinator retirement. */
__global__ void observeRetired(DeviceArgs a){
    if(!s::observeRetired(a)){s::reject(a,ImmutableLookupStatus::InvalidProtocol);unhealthy(a);}
}
/** @brief Complete-parent terminal close is separate from worker retirement and retains all original root owners. */
__global__ void close(DeviceArgs a,const int* complete){
    if(complete && *a.healthy && !*complete)return;
    if((s::stage(a)!=ResponseStage::ReadersComplete && s::stage(a)!=ResponseStage::Idle) ||
       (s::protocol(a).issuedSequence!=s::protocol(a).consumedSequence &&
        s::frontier(a).disposition!=n::SourceDisposition::Aborted)){
        s::reject(a,ImmutableLookupStatus::InvalidAcknowledgement);unhealthy(a);return;
    }
    release(&a.channel.base.acknowledgement->closedGeneration,s::protocol(a).admissionGeneration);
}
/** @brief Reject wrong current device or every default stream spelling before any native enqueue. */
void requireContext(DeviceId device,void* stream){
    if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Pooled source requires its exact non-default native stream");
    int current=-1;
#if defined(QFN_POOLED_SOURCE_CUDA)
    if(!device.is_cuda() || cudaGetDevice(&current)!=cudaSuccess || current!=device.ordinal)
#else
    if(!device.is_rocm() || hipGetDevice(&current)!=hipSuccess || current!=device.ordinal)
#endif
        throw std::invalid_argument("Pooled source current native device differs from its original participant");
}
/** @brief Preserve the exact failing native launch diagnostic. */
void check(){const auto status=QFN_SOURCE_ERROR();if(status!=QFN_SOURCE_SUCCESS)throw std::runtime_error(std::string("Pooled source launch failed: ")+QFN_SOURCE_ERROR_TEXT(status));}
/** @return General grid-stride launch geometry; capacity bounds launch only, never memory/communication extent. */
unsigned blocks(std::uint64_t size){return static_cast<unsigned>(std::max<std::uint64_t>(1,std::min<std::uint64_t>(65535,(size+255)/256)));}
/** @brief Fixed format/clock/codebook bundle retaining the actual service-issued provider binding. */
template<TensorType Format> class Kernels final:public IPooledImmutableSourceKernels {
public:
    Kernels(PooledImmutableSourceGeometry geometry,ImmutableSubordinateLookupBinding source,NativeEmbeddingIQTables tables,std::uint64_t timeout)
        :geometry_(std::move(geometry)),source_(std::move(source)),tables_(tables),timeout_(timeout){}
    const PooledImmutableSourceGeometry& geometry()const noexcept override{return geometry_;}
    const ImmutableSubordinateLookupBinding& source()const noexcept override{return source_;}
protected:
    std::uint64_t codeIdentity()const noexcept override{
        std::uint64_t h=1469598103934665603ULL;const auto mix=[&](std::uint64_t word){h=(h^word)*1099511628211ULL;};
        mix(4);mix(n::Version);mix(kImmutableLookupAcknowledgementVersion);mix(static_cast<std::uint64_t>(Format));mix(timeout_);mix(256);
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq3s));mix(reinterpret_cast<std::uintptr_t>(tables_.iq3xxs));
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq2s));mix(reinterpret_cast<std::uintptr_t>(tables_.iq2xs));
        mix(reinterpret_cast<std::uintptr_t>(tables_.iq2xxs));mix(reinterpret_cast<std::uintptr_t>(tables_.iq1s));
        return h?h:1;
    }
    void initializeNative(const DeviceArgs& a,void* stream)const override{requireContext(a.plan.coordinator.device,stream);QFN_SOURCE_NAMESPACE::initialize<<<1,1,0,static_cast<QFN_SOURCE_STREAM>(stream)>>>(a);check();}
    void admitNative(const DeviceArgs& a,std::uint64_t generation,std::uint64_t invocation,void* stream)const override{
        requireContext(a.plan.coordinator.device,stream);QFN_SOURCE_NAMESPACE::admit<<<1,1,0,static_cast<QFN_SOURCE_STREAM>(stream)>>>(a,generation,invocation);check();}
    void launch(PooledSourceAction action,const DeviceArgs& a,PooledSourceQuery query,void* exact)const override{
        requireContext(a.plan.coordinator.device,exact);const auto stream=static_cast<QFN_SOURCE_STREAM>(exact);const auto maximum=a.plan.coordinator.policy.maximumUnique();
        switch(action){
        case PooledSourceAction::Plan:plan<<<1,1,0,stream>>>(a);break;
        case PooledSourceAction::Publish:publish<<<1,1,0,stream>>>(a);break;
        case PooledSourceAction::Acquire:
            acquireResponse<<<1,256,0,stream>>>(a,timeout_);
            copyResponse<<<blocks(std::uint64_t{maximum}*a.plan.owner.row_bytes),256,0,stream>>>(a);
            QFN_SOURCE_NAMESPACE::prepareDelivery<<<1,1,0,stream>>>(a);
            packetRows<<<blocks(std::uint64_t{a.plan.owner.native.reply_capacity}*a.plan.owner.row_bytes),256,0,stream>>>(a);
            retireResponse<<<1,1,0,stream>>>(a,timeout_);break;
        case PooledSourceAction::Read:
            if(a.plan.owner.owner.rowWidth()==adaptive_lookup::RowKeyWidth::UInt32){
                validateReaders<adaptive_lookup::Slot32><<<blocks(maximum),256,0,stream>>>(a);readerOutcome<<<1,1,0,stream>>>(a,query);
                read<Format,adaptive_lookup::Slot32><<<blocks(std::uint64_t{a.plan.coordinator.policy.maximumOriginal()}*a.plan.row_elements),256,0,stream>>>(a,query,tables_);
            }else{
                validateReaders<adaptive_lookup::Slot64><<<blocks(maximum),256,0,stream>>>(a);readerOutcome<<<1,1,0,stream>>>(a,query);
                read<Format,adaptive_lookup::Slot64><<<blocks(std::uint64_t{a.plan.coordinator.policy.maximumOriginal()}*a.plan.row_elements),256,0,stream>>>(a,query,tables_);
            }break;
        case PooledSourceAction::ReadersRetired:readersRetired<<<1,1,0,stream>>>(a);break;
        case PooledSourceAction::ObserveRetired:QFN_SOURCE_NAMESPACE::observeRetired<<<1,1,0,stream>>>(a);break;
        case PooledSourceAction::BeforeReady:QFN_SOURCE_NAMESPACE::beforeReady<<<1,1,0,stream>>>(a);break;
        }
        check();
    }
    void closeNative(const DeviceArgs& a,const int* complete,void* exact)const override{requireContext(a.plan.coordinator.device,exact);close<<<1,1,0,static_cast<QFN_SOURCE_STREAM>(exact)>>>(a,complete);check();}
private:
    PooledImmutableSourceGeometry geometry_;ImmutableSubordinateLookupBinding source_;NativeEmbeddingIQTables tables_;std::uint64_t timeout_;
};
} // namespace QFN_SOURCE_NAMESPACE
/** @brief Setup-only binding of real source, codebooks and the authenticated standard backend deadline. */
std::unique_ptr<IPooledImmutableSourceKernels> QFN_SOURCE_FACTORY(PooledImmutableSourceGeometry geometry,ImmutableSubordinateLookupBinding source){
    source.requireFor(geometry);const auto& p=geometry.parameters();NativeEmbeddingIQTables tables;std::uint64_t timeout=30'000'000'000ULL;
#if defined(QFN_POOLED_SOURCE_CUDA)
    if(!geometry.allocator().device.is_cuda() || !prepareNativeEmbeddingTablesCUDA(geometry.allocator().device.ordinal,p.format,tables))
        throw std::invalid_argument("Pooled CUDA source/device/codebook mismatch");
#else
    if(!geometry.allocator().device.is_rocm() || !prepareNativeEmbeddingTablesROCm(geometry.allocator().device.ordinal,p.format,tables))
        throw std::invalid_argument("Pooled HIP source/device/codebook/wall-clock mismatch");
    timeout=rocmImmutableLookupTimeoutTicks(rocmHardwareMetadata().at(geometry.allocator().device.ordinal));
#endif
    using namespace QFN_SOURCE_NAMESPACE;
    switch(p.format){
#define QFN_SOURCE_CASE(F) case TensorType::F:return std::make_unique<Kernels<TensorType::F>>(geometry,std::move(source),tables,timeout);
        QFN_SOURCE_CASE(FP32) QFN_SOURCE_CASE(FP16) QFN_SOURCE_CASE(BF16) QFN_LOOKUP_QUANT_FORMATS(QFN_SOURCE_CASE)
#undef QFN_SOURCE_CASE
    default:throw std::invalid_argument("Pooled native source format unsupported");
    }
}
} // namespace llaminar2::qwen4exp
#undef QFN_SOURCE_STREAM
#undef QFN_SOURCE_NAMESPACE
#undef QFN_SOURCE_FACTORY
#undef QFN_SOURCE_ERROR
#undef QFN_SOURCE_SUCCESS
#undef QFN_SOURCE_ERROR_TEXT
