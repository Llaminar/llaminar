/**
 * @file GPUImmutableResidentLookupKernels.cuh
 * @brief Direct CUDA/HIP immutable table ID validation and original native decode.
 *
 * Validation and decode are ordered kernels on the exact captured stream. A
 * failed ID/count therefore suppresses every output write without a temporary
 * buffer or all-grid barrier. Neither kernel touches RAM/mmap service pages,
 * publishes a ticket, mirrors token/model state or synchronizes with the host.
 */
#pragma once
#if defined(QFN_LOOKUP_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define QFN_RESIDENT_STREAM cudaStream_t
#define QFN_RESIDENT_ERROR cudaGetLastError
#define QFN_RESIDENT_SUCCESS cudaSuccess
#define QFN_RESIDENT_ERROR_TEXT cudaGetErrorString
#define QFN_RESIDENT_FACTORY createCUDAImmutableResidentLookupKernels
#define QFN_RESIDENT_NAMESPACE cuda_resident_lookup
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define QFN_RESIDENT_STREAM hipStream_t
#define QFN_RESIDENT_ERROR hipGetLastError
#define QFN_RESIDENT_SUCCESS hipSuccess
#define QFN_RESIDENT_ERROR_TEXT hipGetErrorString
#define QFN_RESIDENT_FACTORY createROCmImmutableResidentLookupKernels
#define QFN_RESIDENT_NAMESPACE rocm_resident_lookup
#endif
#include "ImmutableResidentLookupKernels.h"
#include "../common/SamplingMath.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace llaminar2::qwen4exp
{
namespace QFN_RESIDENT_NAMESPACE
{
/** @brief Publish the canonical first storage failure without replacing an earlier error. */
__device__ __forceinline__ void fail(ImmutableResidentGatherArgs a)
{
    atomicCAS(a.error,0,static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    atomicExch(a.healthy,0);
}
/** @brief Validate only live original IDs; one failure suppresses the following decode kernel. */
__global__ __launch_bounds__(256,2) void validateIds(ImmutableResidentGatherArgs a,std::uint64_t tableRows)
{
    // Ok==0/error==0 is the legitimate prefill state before generation
    // admission. The parent owns forward liveness; only first-error status
    // suppresses this operation. Preserve that error and normalize health.
    if(*a.error!=0)
    {
        if(threadIdx.x==0)atomicExch(a.healthy,0);
        return;
    }
    const int rows=a.tokenRows.liveRows?*a.tokenRows.liveRows:a.tokenRows.rows;
    if(rows<0||rows>a.tokenRows.rows)
    {
        if(blockIdx.x==0&&threadIdx.x==0)fail(a);
        return;
    }
    const auto size=static_cast<std::uint64_t>(rows)*kNgramHeads;
    for(auto index=static_cast<std::uint64_t>(blockIdx.x)*256+threadIdx.x;
        index<size;index+=static_cast<std::uint64_t>(gridDim.x)*256)
    {
        const auto id=a.ids[index];
        if(id<0||static_cast<std::uint64_t>(id)>=tableRows)fail(a);
    }
}
/**
 * @brief Decode one source row per CTA, reusing its native cache lines across columns.
 * @details Each independent value uses the canonical original-format helper;
 * there is no arithmetic reduction, derived table, scale repacking or native
 * gather scratch. Grid-striding requested rows bounds launch dimensions while
 * every global source/output access remains inside the exact live prefix.
 */
template<TensorType Format>
__global__ __launch_bounds__(128,2) void gatherRows(ImmutableResidentGatherArgs a,
    const std::byte *table,std::size_t stride,int width,NativeEmbeddingIQTables codebooks)
{
    if(*a.error!=0)return;
    const int rows=a.tokenRows.liveRows?*a.tokenRows.liveRows:a.tokenRows.rows;
    const auto size=static_cast<std::uint64_t>(rows)*kNgramHeads;
    for(auto row=static_cast<std::uint64_t>(blockIdx.x);row<size;row+=gridDim.x)
    {
        const auto source=table+static_cast<std::uint64_t>(a.ids[row])*stride;
        for(auto column=static_cast<unsigned>(threadIdx.x);column<static_cast<unsigned>(width);column+=128)
            a.output[row*static_cast<unsigned>(width)+column]=lookup_detail::decode<Format>(source,column,codebooks);
    }
}
/**
 * @brief Cap resident work to a reusable CTA grid independent of inactive capacity.
 * @details The kernels stride over live rows on device. A bounded grid avoids
 * launching one idle CTA for every admitted row during partial/empty replay;
 * no count readback, mutable launch topology or arithmetic change is required.
 */
unsigned blocks(std::uint64_t count)
{
    return static_cast<unsigned>(std::min<std::uint64_t>(2048,std::max<std::uint64_t>(1,count)));
}
/** @brief Fail native launch submission without introducing a synchronization boundary. */
void check()
{
    const auto status=QFN_RESIDENT_ERROR();
    if(status!=QFN_RESIDENT_SUCCESS)
        throw std::runtime_error(std::string("Resident immutable lookup launch failed: ")+QFN_RESIDENT_ERROR_TEXT(status));
}
/** @brief Immutable source/codebook bundle; mutable execution state remains caller-owned. */
template<TensorType Format> class Kernels final : public IImmutableResidentLookupKernels
{
public:
    /** @brief Retain checked native geometry and setup-published backend codebook addresses. */
    Kernels(ImmutableResidentTableView table,NativeEmbeddingIQTables codebooks):table_(table),codebooks_(codebooks) {}
    /** @copydoc IImmutableResidentLookupKernels::table */
    const ImmutableResidentTableView &table() const noexcept override { return table_; }
    /** @copydoc IImmutableResidentLookupKernels::gather */
    void gather(const ImmutableResidentGatherArgs &a,void *stream) const override
    {
        resident_lookup_detail::validate(table_,a,stream);
        const auto ids=static_cast<std::uint64_t>(a.tokenRows.rows)*kNgramHeads;
        auto native=static_cast<QFN_RESIDENT_STREAM>(stream);
        validateIds<<<blocks((ids+255)/256),256,0,native>>>(a,table_.rowCount());
        gatherRows<Format><<<blocks(ids),128,0,native>>>(a,table_.nativeData(),table_.rowBytes(),table_.rowElements(),codebooks_);
        check();
    }
private:
    const ImmutableResidentTableView table_;
    const NativeEmbeddingIQTables codebooks_;
};
} // namespace QFN_RESIDENT_NAMESPACE

std::unique_ptr<IImmutableResidentLookupKernels> QFN_RESIDENT_FACTORY(ImmutableResidentTableView table)
{
    NativeEmbeddingIQTables codebooks;
#if defined(QFN_LOOKUP_CUDA)
    if(!table.device().is_cuda()||!prepareNativeEmbeddingTablesCUDA(table.device().ordinal,table.format(),codebooks))
        throw std::runtime_error("Resident immutable lookup CUDA endpoint/codebook preparation failed");
#else
    if(!table.device().is_rocm()||!prepareNativeEmbeddingTablesROCm(table.device().ordinal,table.format(),codebooks))
        throw std::runtime_error("Resident immutable lookup ROCm endpoint/codebook preparation failed");
#endif
    using namespace QFN_RESIDENT_NAMESPACE;
    switch(table.format())
    {
#define QFN_RESIDENT_CASE(F) case TensorType::F:return std::make_unique<Kernels<TensorType::F>>(table,codebooks);
        QFN_RESIDENT_CASE(FP32) QFN_RESIDENT_CASE(FP16) QFN_RESIDENT_CASE(BF16)
        QFN_LOOKUP_QUANT_FORMATS(QFN_RESIDENT_CASE)
#undef QFN_RESIDENT_CASE
    default:throw std::invalid_argument("Resident immutable lookup source format is unsupported");
    }
}
} // namespace llaminar2::qwen4exp
#undef QFN_RESIDENT_STREAM
#undef QFN_RESIDENT_ERROR
#undef QFN_RESIDENT_SUCCESS
#undef QFN_RESIDENT_ERROR_TEXT
#undef QFN_RESIDENT_FACTORY
#undef QFN_RESIDENT_NAMESPACE
