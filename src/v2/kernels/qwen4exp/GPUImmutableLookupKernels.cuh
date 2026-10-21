/**
 * @file GPUImmutableLookupKernels.cuh
 * @brief Exact CUDA/HIP immutable I/O publication, acquire, gather and decode.
 *
 * Mapped pages are touched only for live IDs, required protocol words and live
 * native response bytes. A single CTA waits for the independent CPU producer;
 * no grid of spin blocks can occupy the whole GPU. Every mapped payload writer
 * fences before the final system-release publication. Response cache-volatile
 * loads precede acknowledgement, then decoding reads private GPU scratch.
 */
#pragma once
#if defined(QFN_LOOKUP_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#define QFN_LOOKUP_STREAM cudaStream_t
#define QFN_LOOKUP_ERROR cudaGetLastError
#define QFN_LOOKUP_SUCCESS cudaSuccess
#define QFN_LOOKUP_ERROR_TEXT cudaGetErrorString
#define QFN_LOOKUP_FACTORY createCUDAImmutableLookupKernels
#define QFN_LOOKUP_NAMESPACE cuda_lookup
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define QFN_LOOKUP_STREAM hipStream_t
#define QFN_LOOKUP_ERROR hipGetLastError
#define QFN_LOOKUP_SUCCESS hipSuccess
#define QFN_LOOKUP_ERROR_TEXT hipGetErrorString
#define QFN_LOOKUP_FACTORY createROCmImmutableLookupKernels
#define QFN_LOOKUP_NAMESPACE rocm_lookup
#endif
#include "ImmutableLookupKernels.h"
#include "ImmutableLookupDecode.h"
#include "../common/SamplingMath.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2::qwen4exp
{
namespace QFN_LOOKUP_NAMESPACE
{
/** @brief System-acquire a CPU-owned protocol publication word. */
__device__ __forceinline__ std::uint64_t acquire(const std::uint64_t *pointer)
{
#if defined(QFN_LOOKUP_CUDA)
    cuda::atomic_ref<std::uint64_t,cuda::thread_scope_system> word(*const_cast<std::uint64_t *>(pointer));
    return word.load(cuda::memory_order_acquire);
#else
    return __hip_atomic_load(pointer,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @brief Release only after every mapped payload writer has completed its system fence. */
__device__ __forceinline__ void release(std::uint64_t *pointer,std::uint64_t value)
{
#if defined(QFN_LOOKUP_CUDA)
    cuda::atomic_ref<std::uint64_t,cuda::thread_scope_system> word(*pointer);
    word.store(value,cuda::memory_order_release);
#else
    __hip_atomic_store(pointer,value,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @brief Bypass stale GPU cache lines for each acquired host payload word. */
template<class T> __device__ __forceinline__ T fresh(const T *pointer)
{
#if defined(QFN_LOOKUP_CUDA)
    return __ldcv(pointer);
#else
    return __builtin_nontemporal_load(pointer);
#endif
}
/** @brief Fixed-frequency wall clock for the standard thirty-second protocol deadline. */
__device__ __forceinline__ std::uint64_t wallClock()
{
#if defined(QFN_LOOKUP_CUDA)
    std::uint64_t value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
#else
    return wall_clock64();
#endif
}
/** @brief Stop malformed device-owned geometry rather than exposing unrelated host bytes. */
__device__ __forceinline__ void invalid()
{
#if defined(QFN_LOOKUP_CUDA)
    __trap();
#else
    __builtin_trap();
#endif
}
/** @brief Start one GPU-owned monotonic transaction and fill only live request fields. */
__global__ void begin(ImmutableLookupPublishArgs a)
{
    const int rows=a.tokenRows.liveRows?*a.tokenRows.liveRows:a.tokenRows.rows;
    const auto generation=*a.generation;auto &state=*a.state;
    if(rows<0||rows>a.tokenRows.rows||generation==0||state.issuedSequence!=state.consumedSequence||
       state.issuedSequence==~std::uint64_t{0}||static_cast<std::uint64_t>(rows)>a.channel.maximumIds/16)invalid();
    const auto sequence=++state.issuedSequence;
    state.generation=generation;state.idCount=static_cast<std::uint64_t>(rows)*16;
    state.nativeBytes=state.idCount*a.channel.rowBytes;state.status=0;
    auto &request=*a.channel.request;
    request.sequenceComplement=~sequence;request.generation=generation;request.generationComplement=~generation;
    request.table=a.channel.table;request.tableComplement=~a.channel.table;
    request.idCount=state.idCount;request.idCountComplement=~state.idCount;
    request.idBytes=state.idCount*8;request.idBytesComplement=~request.idBytes;
    request.rowBytes=a.channel.rowBytes;request.rowBytesComplement=~a.channel.rowBytes;
    request.protocol=kImmutableLookupProtocol;request.protocolComplement=~kImmutableLookupProtocol;
    __threadfence_system();
}
/** @brief Copy exactly the live INT64 IDs; every writer publishes its own system fence. */
__global__ __launch_bounds__(256,2) void copyIds(ImmutableLookupPublishArgs a)
{
    for(std::uint64_t i=static_cast<std::uint64_t>(blockIdx.x)*256+threadIdx.x;i<a.state->idCount;i+=static_cast<std::uint64_t>(gridDim.x)*256)
        a.channel.ids[i]=a.ids[i];
    __threadfence_system();
}
/** @brief Release the complete ID request after ordered payload kernels have retired. */
__global__ void publishRequest(ImmutableLookupPublishArgs a)
{
    release(&a.channel.request->publication,a.state->issuedSequence);
}
/** @brief Wait with one CTA, then authenticate coalesced exact completion metadata. */
__global__ __launch_bounds__(256,2) void waitResponse(ImmutableLookupConsumeArgs a,std::uint64_t timeout)
{
    alignas(64) __shared__ std::uint64_t words[sizeof(ImmutableLookupCompletion)/8];
    __shared__ std::uint64_t acquired;
    if(threadIdx.x==0)
    {
        const auto start=wallClock();std::uint64_t sequence;
        while((sequence=acquire(&a.channel.completion->publication))<a.state->issuedSequence&&wallClock()-start<timeout)
        {
#if defined(QFN_LOOKUP_CUDA)
            __nanosleep(256);
#else
            __builtin_amdgcn_s_sleep(1);
#endif
        }
        acquired=sequence;
    }
    __syncthreads();
    if(threadIdx.x<kLookupCompletionLiveBytes/8)
        words[threadIdx.x]=fresh(reinterpret_cast<const std::uint64_t *>(a.channel.completion)+threadIdx.x);
    __syncthreads();
    if(threadIdx.x==0)
    {
        // Read no alignment tail: the acquired live words alone are the wire ABI.
        const auto &body=*reinterpret_cast<const ImmutableLookupCompletion *>(words);
        auto status=acquired<a.state->issuedSequence?ImmutableLookupStatus::TimedOut:
            ImmutableLookupProtocol::validateCompletion(body,acquired,a.state->issuedSequence,a.state->generation,
                a.channel.table,a.state->idCount,a.channel.rowBytes);
        a.state->status=static_cast<std::uint64_t>(status);
        if(status!=ImmutableLookupStatus::Success)
        {
            *a.healthy=0;
            *a.error=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);
        }
    }
}
/** @brief Coalesced fresh loads copy exactly native live bytes to persistent private scratch. */
__global__ __launch_bounds__(256,2) void copyNative(ImmutableLookupConsumeArgs a)
{
    if(a.state->status!=0)return;
    const auto bytes=a.state->nativeBytes;
    const auto index=static_cast<std::uint64_t>(blockIdx.x)*256+threadIdx.x;
    const auto stride=static_cast<std::uint64_t>(gridDim.x)*256;
    for(auto word=index;word<bytes/4;word+=stride)
        reinterpret_cast<std::uint32_t *>(a.nativeScratch)[word]=fresh(reinterpret_cast<const std::uint32_t *>(a.channel.nativeRows)+word);
    if(index<bytes%4)
        reinterpret_cast<unsigned char *>(a.nativeScratch)[bytes/4*4+index]=fresh(reinterpret_cast<const unsigned char *>(a.channel.nativeRows)+bytes/4*4+index);
}
/** @brief Retire the mapped response only after all native-byte readers have completed. */
__global__ void acknowledge(ImmutableLookupConsumeArgs a)
{
    a.state->consumedSequence=a.state->issuedSequence;
    release(&a.channel.acknowledgement->consumedSequence,a.state->consumedSequence);
}
/** @brief Native-format-specialized parallel decode using existing source codebooks. */
template<TensorType Format>
__global__ __launch_bounds__(256,2) void decodeRows(ImmutableLookupConsumeArgs a,int width,NativeEmbeddingIQTables tables)
{
    if(a.state->status!=0)return;
    const auto size=a.state->idCount*static_cast<std::uint64_t>(width);
    for(std::uint64_t i=static_cast<std::uint64_t>(blockIdx.x)*256+threadIdx.x;i<size;i+=static_cast<std::uint64_t>(gridDim.x)*256)
        a.output[i]=lookup_detail::decode<Format>(a.nativeScratch+(i/width)*a.channel.rowBytes,i%width,tables);
}
/** @brief Publish explicit complete-parent service close after all response acknowledgements. */
__global__ void terminalClose(ImmutableLookupCloseArgs a)
{
    if(a.healthy&&*a.healthy!=0&&*a.complete==0)return;
    if(*a.generation==0||a.state->consumedSequence!=a.state->issuedSequence)invalid();
    release(&a.channel.acknowledgement->closedGeneration,*a.generation);
}
/** @brief Publish only immutable API admission metadata, outside captured execution. */
__global__ void admission(ImmutableLookupAdmissionArgs a)
{
    if(a.initialization==ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce)
        *a.state=ImmutableLookupDeviceState{};
    *a.deviceGeneration=a.generation;
}
/** @brief Validate frozen mapped addresses, exact capacities and setup-owned stream. */
void validate(const ImmutableLookupDeviceView &view,void *stream)
{
    if(!stream||reinterpret_cast<std::uintptr_t>(stream)<=2||!view.request||!view.completion||!view.acknowledgement||
       !view.ids||!view.nativeRows||!view.table||!view.maximumIds||!view.rowBytes||
       view.maximumIds>std::numeric_limits<std::uint64_t>::max()/view.rowBytes||
       view.maximumIds>std::numeric_limits<std::uint64_t>::max()/8||
       reinterpret_cast<std::uintptr_t>(view.request)%64||reinterpret_cast<std::uintptr_t>(view.completion)%64||
       reinterpret_cast<std::uintptr_t>(view.acknowledgement)%64||reinterpret_cast<std::uintptr_t>(view.ids)%8||
       reinterpret_cast<std::uintptr_t>(view.nativeRows)%4)
        throw std::invalid_argument("Immutable lookup requires exact non-default stream and aligned complete channel bindings");
}
/** @brief Reject native submission errors without synchronizing capture or replay. */
void check()
{
    const auto error=QFN_LOOKUP_ERROR();
    if(error!=QFN_LOOKUP_SUCCESS)throw std::runtime_error(std::string("Immutable lookup GPU launch failed: ")+QFN_LOOKUP_ERROR_TEXT(error));
}
/** @brief Bounded grid independent of live count; every lane guards against exact device extent. */
unsigned blocks(std::uint64_t count)
{
    return static_cast<unsigned>(std::min<std::uint64_t>(65535,std::max<std::uint64_t>(1,(count+255)/256)));
}
/** @brief Immutable native format/geometry bundle; all mutable state is graph-owned. */
template<TensorType Format> class Kernels final : public IImmutableLookupKernels
{
public:
    /** @brief Freeze source codebooks, exact output width and backend wall-clock deadline. */
    Kernels(int width,std::size_t bytes,NativeEmbeddingIQTables tables,std::uint64_t timeout):width_(width),rowBytes_(bytes),tables_(tables),timeout_(timeout) {}
    /** @copydoc IImmutableLookupKernels::admit */
    void admit(const ImmutableLookupAdmissionArgs &a,void *stream) const override
    {
        if(!stream||reinterpret_cast<std::uintptr_t>(stream)<=2||!a.generation||!a.deviceGeneration||!a.state||
           reinterpret_cast<std::uintptr_t>(a.state)%64||
           (a.initialization!=ImmutableLookupAdmissionArgs::ProtocolInitialization::Preserve&&
            a.initialization!=ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce)||
           reinterpret_cast<std::uintptr_t>(a.deviceGeneration)%8)
            throw std::invalid_argument("Immutable lookup admission requires positive identity and exact stream/aligned device word");
        admission<<<1,1,0,static_cast<QFN_LOOKUP_STREAM>(stream)>>>(a);check();
    }
    /** @copydoc IImmutableLookupKernels::publish */
    void publish(const ImmutableLookupPublishArgs &a,void *stream) const override
    {
        validate(a.channel,stream);
        if(a.tokenRows.rows<=0||static_cast<std::uint64_t>(a.tokenRows.rows)>a.channel.maximumIds/16||!a.ids||!a.generation||!a.state||a.channel.rowBytes!=rowBytes_)
            throw std::invalid_argument("Immutable lookup ID publication exceeds admitted token geometry");
        auto native=static_cast<QFN_LOOKUP_STREAM>(stream);
        begin<<<1,1,0,native>>>(a);copyIds<<<blocks(a.channel.maximumIds),256,0,native>>>(a);
        publishRequest<<<1,1,0,native>>>(a);check();
    }
    /** @copydoc IImmutableLookupKernels::consume */
    void consume(const ImmutableLookupConsumeArgs &a,void *stream) const override
    {
        validate(a.channel,stream);
        if(!a.state||!a.nativeScratch||!a.output||!a.healthy||!a.error||a.channel.rowBytes!=rowBytes_||
           reinterpret_cast<std::uintptr_t>(a.nativeScratch)%4||
           a.channel.maximumIds>std::numeric_limits<std::uint64_t>::max()/static_cast<unsigned>(width_))
            throw std::invalid_argument("Immutable lookup response requires persistent scratch, output and canonical error bindings");
        auto native=static_cast<QFN_LOOKUP_STREAM>(stream);
        waitResponse<<<1,256,0,native>>>(a,timeout_);
        copyNative<<<blocks(a.channel.maximumIds*a.channel.rowBytes/4),256,0,native>>>(a);
        acknowledge<<<1,1,0,native>>>(a);
        decodeRows<Format><<<blocks(a.channel.maximumIds*width_),256,0,native>>>(a,width_,tables_);check();
    }
    /** @copydoc IImmutableLookupKernels::close */
    void close(const ImmutableLookupCloseArgs &a,void *stream) const override
    {
        validate(a.channel,stream);
        if(!a.generation||!a.state||bool(a.healthy)!=bool(a.complete))throw std::invalid_argument("Immutable lookup close requires canonical generation/state and paired terminal predicates");
        terminalClose<<<1,1,0,static_cast<QFN_LOOKUP_STREAM>(stream)>>>(a);check();
    }
private:
    int width_;std::size_t rowBytes_;NativeEmbeddingIQTables tables_;std::uint64_t timeout_;
};
} // namespace QFN_LOOKUP_NAMESPACE
std::unique_ptr<IImmutableLookupKernels> QFN_LOOKUP_FACTORY(DeviceId device,TensorType format,int width,std::size_t rowBytes)
{
    lookup_detail::validateGeometry(format,width,rowBytes);
    NativeEmbeddingIQTables tables;std::uint64_t timeout=30'000'000'000ULL;
#if defined(QFN_LOOKUP_CUDA)
    if(!device.is_cuda()||!prepareNativeEmbeddingTablesCUDA(device.ordinal,format,tables))
        throw std::runtime_error("Immutable lookup CUDA setup could not authenticate exact endpoint/codebooks");
#else
    int rate=0;
    if(!device.is_rocm()||!prepareNativeEmbeddingTablesROCm(device.ordinal,format,tables)||
       hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,device.ordinal)!=hipSuccess||rate<=0)
        throw std::runtime_error("Immutable lookup ROCm setup could not authenticate endpoint/codebooks/wall clock");
    timeout=static_cast<std::uint64_t>(rate)*30'000;
#endif
    using namespace QFN_LOOKUP_NAMESPACE;
    switch(format)
    {
#define QFN_LOOKUP_CASE(F) case TensorType::F:return std::make_unique<Kernels<TensorType::F>>(width,rowBytes,tables,timeout);
        QFN_LOOKUP_CASE(FP32) QFN_LOOKUP_CASE(FP16) QFN_LOOKUP_CASE(BF16)
        QFN_LOOKUP_QUANT_FORMATS(QFN_LOOKUP_CASE)
#undef QFN_LOOKUP_CASE
    default:throw std::invalid_argument("Immutable lookup GPU source format is unsupported");
    }
}
} // namespace llaminar2::qwen4exp
#undef QFN_LOOKUP_STREAM
#undef QFN_LOOKUP_ERROR
#undef QFN_LOOKUP_SUCCESS
#undef QFN_LOOKUP_ERROR_TEXT
#undef QFN_LOOKUP_FACTORY
#undef QFN_LOOKUP_NAMESPACE
