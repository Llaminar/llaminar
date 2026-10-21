/**
 * @file GPUImmutableFixedHeadLookupKernels.cuh
 * @brief Symmetric CUDA/HIP fixed-head validation, native packing and decode.
 *
 * Required fixed-header and counted-payload collectives remain separate graph
 * stages. Two ordered local-validation kernels avoid a grid-wide initializer
 * race. Cohort validation always rewrites its derived transport count. All
 * payload kernels consult canonical first-error, not generation-admission
 * health: cold prefill legitimately has healthy0/error0. No host work, scratch
 * allocation, transfer, reduction or source representation change occurs here.
 */
#pragma once
#if defined(QFN_FIXED_HEAD_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define QFN_HEAD_STREAM cudaStream_t
#define QFN_HEAD_ERROR cudaGetLastError
#define QFN_HEAD_SUCCESS cudaSuccess
#define QFN_HEAD_ERROR_TEXT cudaGetErrorString
#define QFN_HEAD_FACTORY createCUDAImmutableFixedHeadLookupKernels
#define QFN_HEAD_NAMESPACE cuda_fixed_head_lookup
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define QFN_HEAD_STREAM hipStream_t
#define QFN_HEAD_ERROR hipGetLastError
#define QFN_HEAD_SUCCESS hipSuccess
#define QFN_HEAD_ERROR_TEXT hipGetErrorString
#define QFN_HEAD_FACTORY createROCmImmutableFixedHeadLookupKernels
#define QFN_HEAD_NAMESPACE rocm_fixed_head_lookup
#endif
#include "ImmutableFixedHeadLookupKernels.h"
#include "../common/SamplingMath.h"
#include <algorithm>
#include <string>

namespace llaminar2::qwen4exp
{
namespace QFN_HEAD_NAMESPACE
{
/** @brief Small trivially copied launch frame; no duplicated mutable state. */
struct Frame
{
    int capacity,participants,heads,firstHead,width;
    const std::int32_t *originalCount;
    const std::int64_t *ids;
    FixedHeadValidationHeader *header;
    const FixedHeadValidationHeader *headers;
    std::int32_t *transport;
    const std::byte *source;
    std::uint64_t firstRow;
    std::size_t stride,rankStride;
    std::byte *local;
    const std::byte *gathered;
    float *output;
    int *healthy,*error;
    QwenFixedHeadCounterRow counters;
};
/** @brief Only local head bounds are embedded; no device std::array access or private policy copy. */
struct HeadBounds { std::uint64_t begin[8],size[8]; };
/** @brief Read original logical count, never the transport-derived count. */
__device__ __forceinline__ int originalRows(const Frame &a)
{
    return a.originalCount?*a.originalCount:a.capacity;
}
/** @brief Preserve the canonical first error and revoke health without creating another authority. */
__device__ __forceinline__ void fail(const Frame &a,int error)
{
    atomicCAS(a.error,0,error);atomicExch(a.healthy,0);
}
/** @brief Original storage failure code shared with resident and hosted lookup. */
constexpr int failure=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);

/** @brief Publish the mandatory header even on prior failure, before independent validation CTAs. */
__global__ __launch_bounds__(1) void initializeHeader(Frame a)
{
    // The same transport-only scalar is first an admitted constant-one
    // header extent, then consensus overwrites it with payload L/0. Neither
    // phase changes or substitutes the original query count.
    *a.transport=1;
    if(a.counters.bound() && !addNativeCounter64(a.counters.metric(QwenFixedHeadCounter::Calls),1))
        fail(a,failure);
    a.header->originalLiveRows=originalRows(a);
    a.header->firstError=*a.error;
    if(*a.error!=0)atomicExch(a.healthy,0);
}
/** @brief Check only owned IDs against their exact original per-head vocabulary, including gaps. */
__global__ __launch_bounds__(256,2) void validateOwnedIds(Frame a,HeadBounds bounds)
{
    if(*a.error!=0)return;
    const int rows=originalRows(a);
    if(rows<0||rows>a.capacity)
    {
        if(blockIdx.x==0&&threadIdx.x==0)
        {
            atomicCAS(&a.header->firstError,0,failure);fail(a,failure);
        }
        return;
    }
    const auto count=static_cast<std::uint64_t>(rows)*a.heads;
    for(auto i=static_cast<std::uint64_t>(blockIdx.x)*256+threadIdx.x;i<count;
        i+=static_cast<std::uint64_t>(gridDim.x)*256)
    {
        const int head=static_cast<int>(i%a.heads);
        const auto id=a.ids[(i/a.heads)*kNgramHeads+a.firstHead+head];
        // Constant-index parameter accesses avoid an addressable per-thread
        // copy of the small bound array on either device compiler. Resource
        // qualification still belongs to the complete architecture build.
        bool owned=false;
#pragma unroll
        for(int h=0;h<8;++h)
            if(head==h)owned=id>=0&&static_cast<std::uint64_t>(id)>=bounds.begin[h]&&
                static_cast<std::uint64_t>(id)-bounds.begin[h]<bounds.size[h];
        if(!owned)
        {
            atomicCAS(&a.header->firstError,0,failure);fail(a,failure);
        }
    }
}
/**
 * @brief Deterministic bounded metadata check after the mandatory header exchange.
 * @details One lane checks at most8 headers in rank order. No payload reduction
 * or model state is serialized. A stale successful transport extent is always
 * overwritten with zero before failure is published.
 */
__global__ __launch_bounds__(1) void validateHeaders(Frame a)
{
    *a.transport=0;
    const int rows=originalRows(a);
    int status=*a.error;
    if(rows<0||rows>a.capacity)status=status?status:failure;
    for(int rank=0;rank<a.participants;++rank)
    {
        const auto header=a.headers[rank];
        if(!status&&header.firstError)status=header.firstError;
        if(header.originalLiveRows<0||header.originalLiveRows>a.capacity||header.originalLiveRows!=rows)
            status=status?status:failure;
    }
    if(status) {
        // Exactly one lane/participant/lookup observes consensus failure.
        if(a.counters.bound() && !addNativeCounter64(a.counters.metric(QwenFixedHeadCounter::Failures),1))
            status=status?status:failure;
        fail(a,status);
    }
    else *a.transport=rows;
}
/** @brief Fail closed on stale/corrupt transport metadata before any payload address is evaluated. */
__device__ __forceinline__ int payloadRows(const Frame &a)
{
    if(*a.error!=0)return 0;
    const int rows=*a.transport;
    if(rows<0||rows>a.capacity||rows!=originalRows(a))
    {
        fail(a,failure);return 0;
    }
    return rows;
}
/** @brief Copy exact original bytes of validated live local head rows; no expansion or padding. */
__global__ __launch_bounds__(256,2) void packNative(Frame a)
{
    const int rows=payloadRows(a);
    const auto bytes=static_cast<std::uint64_t>(rows)*a.heads*a.stride;
    if(blockIdx.x==0 && threadIdx.x==0 && a.counters.bound()) {
        const auto ids=static_cast<std::uint64_t>(rows)*a.heads;
        if(!addNativeCounter64(a.counters.metric(QwenFixedHeadCounter::LiveIds),ids) ||
            !addNativeCounter64(a.counters.metric(QwenFixedHeadCounter::NativeBytes),bytes))fail(a,failure);
    }
    for(auto i=static_cast<std::uint64_t>(blockIdx.x)*256+threadIdx.x;i<bytes;
        i+=static_cast<std::uint64_t>(gridDim.x)*256)
    {
        const auto row=i/a.stride,column=i%a.stride;
        const auto id=a.ids[(row/a.heads)*kNgramHeads+a.firstHead+row%a.heads];
        a.local[i]=a.source[(static_cast<std::uint64_t>(id)-a.firstRow)*a.stride+column];
    }
}
/** @brief Restore token/global-head ordering from capacity-strided source banks using canonical decode. */
template<TensorType Format>
__global__ __launch_bounds__(128,2) void decodeRankMajor(Frame a,NativeEmbeddingIQTables tables)
{
    const int rows=payloadRows(a);
    const auto count=static_cast<std::uint64_t>(rows)*kNgramHeads;
    for(auto row=static_cast<std::uint64_t>(blockIdx.x);row<count;row+=gridDim.x)
    {
        const auto token=row/kNgramHeads,head=row%kNgramHeads;
        const auto rank=head/a.heads,localHead=head%a.heads;
        const auto *native=a.gathered+rank*a.rankStride+(token*a.heads+localHead)*a.stride;
        for(unsigned column=threadIdx.x;column<static_cast<unsigned>(a.width);column+=128)
            a.output[row*a.width+column]=lookup_detail::decode<Format>(native,column,tables);
    }
}
/** @brief Bounded capture-stable grid; live work is resolved exclusively on device. */
unsigned blocks(std::size_t work)
{
    return static_cast<unsigned>(std::min<std::size_t>(2048,std::max<std::size_t>(1,work)));
}
/** @brief Surface launch submission failure without synchronization or retry. */
void check()
{
    const auto status=QFN_HEAD_ERROR();
    if(status!=QFN_HEAD_SUCCESS)
        throw std::runtime_error(std::string("Fixed-head immutable lookup launch failed: ")+QFN_HEAD_ERROR_TEXT(status));
}
/** @brief Immutable geometry/codebook owner; all state and physical storage are borrowed. */
template<TensorType Format> class Kernels final:public IImmutableFixedHeadLookupKernels
{
public:
    /** @brief Retain setup-authenticated original source/codebook geometry. */
    Kernels(ImmutableFixedHeadLookupPlan plan,NativeEmbeddingIQTables tables):plan_(plan),tables_(tables)
    {
        for(int h=0;h<plan_.heads();++h)
        {
            bounds_.begin[h]=plan_.policy().offsets[plan_.firstHead()+h];
            bounds_.size[h]=plan_.policy().moduli[plan_.firstHead()+h];
        }
    }
    /** @copydoc IImmutableFixedHeadLookupKernels::plan */
    const ImmutableFixedHeadLookupPlan &plan() const noexcept override { return plan_; }
    /** @copydoc IImmutableFixedHeadLookupKernels::captureIdentity */
    ImmutableFixedHeadLookupCaptureIdentity captureIdentity(const ImmutableFixedHeadLookupArgs &a,void *stream) const override
    {
        auto identity=fixed_head_lookup_detail::validate(plan_,a,stream);
        identity.codebooks={tables_.iq3s,tables_.iq3xxs,tables_.iq2s,tables_.iq2xs,tables_.iq2xxs,tables_.iq1s};
        return identity;
    }
    /** @copydoc IImmutableFixedHeadLookupKernels::validateLocal */
    void validateLocal(const ImmutableFixedHeadLookupArgs &a,void *stream) const override
    {
        const auto f=frame(a,stream);const auto native=static_cast<QFN_HEAD_STREAM>(stream);
        initializeHeader<<<1,1,0,native>>>(f);
        validateOwnedIds<<<blocks((static_cast<std::size_t>(f.capacity)*f.heads+255)/256),256,0,native>>>(f,bounds_);
        check();
    }
    /** @copydoc IImmutableFixedHeadLookupKernels::validateCohort */
    void validateCohort(const ImmutableFixedHeadLookupArgs &a,void *stream) const override
    {
        validateHeaders<<<1,1,0,static_cast<QFN_HEAD_STREAM>(stream)>>>(frame(a,stream));check();
    }
    /** @copydoc IImmutableFixedHeadLookupKernels::pack */
    void pack(const ImmutableFixedHeadLookupArgs &a,void *stream) const override
    {
        const auto f=frame(a,stream);
        packNative<<<blocks((a.localNativeBytes-1)/256+1),256,0,static_cast<QFN_HEAD_STREAM>(stream)>>>(f);check();
    }
    /** @copydoc IImmutableFixedHeadLookupKernels::decode */
    void decode(const ImmutableFixedHeadLookupArgs &a,void *stream) const override
    {
        const auto f=frame(a,stream);
        decodeRankMajor<Format><<<blocks(static_cast<std::size_t>(f.capacity)*kNgramHeads),128,0,
            static_cast<QFN_HEAD_STREAM>(stream)>>>(f,tables_);check();
    }
private:
    /** @brief Authenticate the complete original frame before embedding any native pointer. */
    Frame frame(const ImmutableFixedHeadLookupArgs &a,void *stream) const
    {
        (void)fixed_head_lookup_detail::validate(plan_,a,stream);
        return {a.tokenRows.physicalRows(),plan_.participants(),plan_.heads(),plan_.firstHead(),plan_.table().rowElements(),
            a.tokenRows.countOwner(),a.ids,a.localHeader,a.gatheredHeaders,a.transportRows,plan_.table().nativeData(),
            plan_.table().globalRowBegin(),plan_.table().rowBytes(),a.localNativeBytes,a.localNative,a.gatheredNative,
            a.output,a.healthy,a.error,a.counters};
    }
    const ImmutableFixedHeadLookupPlan plan_;
    const NativeEmbeddingIQTables tables_;
    HeadBounds bounds_{};
};
}

std::unique_ptr<IImmutableFixedHeadLookupKernels> QFN_HEAD_FACTORY(ImmutableFixedHeadLookupPlan plan)
{
    NativeEmbeddingIQTables tables;
#if defined(QFN_FIXED_HEAD_CUDA)
    if(!plan.table().device().is_cuda()||
       !prepareNativeEmbeddingTablesCUDA(plan.table().device().ordinal,plan.table().format(),tables))
        throw std::runtime_error("Fixed-head CUDA source/codebook preparation failed");
#else
    if(!plan.table().device().is_rocm()||
       !prepareNativeEmbeddingTablesROCm(plan.table().device().ordinal,plan.table().format(),tables))
        throw std::runtime_error("Fixed-head ROCm source/codebook preparation failed");
#endif
    using namespace QFN_HEAD_NAMESPACE;
    switch(plan.table().format())
    {
#define QFN_HEAD_CASE(F) case TensorType::F:return std::make_unique<Kernels<TensorType::F>>(plan,tables);
        QFN_HEAD_CASE(FP32) QFN_HEAD_CASE(FP16) QFN_HEAD_CASE(BF16)
        QFN_LOOKUP_QUANT_FORMATS(QFN_HEAD_CASE)
#undef QFN_HEAD_CASE
    default:throw std::invalid_argument("Fixed-head lookup source format is unsupported");
    }
}
}
#undef QFN_HEAD_STREAM
#undef QFN_HEAD_ERROR
#undef QFN_HEAD_SUCCESS
#undef QFN_HEAD_ERROR_TEXT
#undef QFN_HEAD_FACTORY
#undef QFN_HEAD_NAMESPACE
