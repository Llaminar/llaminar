/**
 * @file GPUDiagnosticSnapshotCopy.cuh
 * @brief Native count-aware diagnostic metadata and authored-byte packing.
 *
 * One CTA captures native counts with a fixed integer scan; the next kernel
 * copies only authored words to admitted private diagnostic storage. No model
 * output padding is written. Metadata and payload share the producer's exact
 * stream, so later terminal observation cannot combine different transactions.
 */
#pragma once
#include "DiagnosticSnapshotCopy.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#if defined(DIAGNOSTIC_SNAPSHOT_CUDA)
#include <cuda_runtime.h>
#define DSC_STREAM cudaStream_t
#define DSC_ERROR cudaGetLastError
#define DSC_SUCCESS cudaSuccess
#define DSC_ERROR_TEXT cudaGetErrorString
#define DSC_NAMESPACE cuda_diagnostic_snapshot
#define DSC_LAUNCH launchCUDADiagnosticSnapshotCopy
#else
#include <hip/hip_runtime.h>
#define DSC_STREAM hipStream_t
#define DSC_ERROR hipGetLastError
#define DSC_SUCCESS hipSuccess
#define DSC_ERROR_TEXT hipGetErrorString
#define DSC_NAMESPACE rocm_diagnostic_snapshot
#define DSC_LAUNCH launchROCmDiagnosticSnapshotCopy
#endif

namespace llaminar2
{
namespace DSC_NAMESPACE
{
/** @brief Malformed execution-owned counts are fatal; never clamp a diagnostic extent. */
__device__ __forceinline__ void invalid()
{
#if defined(DIAGNOSTIC_SNAPSHOT_CUDA)
    __trap();
#else
    __builtin_trap();
#endif
}
/** @brief Snapshot native count metadata and scan only authored selection rows. */
__global__ __launch_bounds__(256,2) void captureMetadata(DiagnosticSnapshotCopyArgs a)
{
    const auto &g=a.geometry;
    if(g.kind==DiagnosticSnapshotCopyKind::AuthoredTokenRows)
    {
        if(threadIdx.x==0) {
            auto *records=reinterpret_cast<SnapshotAuthoredTokenRowRecord *>(a.metadata+1);
            const auto id=a.work.captureIdentity();
            std::uint64_t rows=0;
            for(int record=0;record<g.requests;++record) {
                const int request=snapshotProjectionFirstRequest(a.work)+record;
                const auto observed=a.work.observeDevice(request);
                const auto span=observed.span;
                records[record]={observed.frontier,observed.live,
                    request*id.logical_rows*id.repeats_per_row-id.first_physical_row,
                    id.logical_rows,id.source_capacity,id.first_required_logical_row,span.first,span.end,
                    observed.live==0||static_cast<std::int64_t>(observed.frontier)+observed.live<=id.first_required_logical_row
                        ?SnapshotEmptyReason::RankingNotRequired:SnapshotEmptyReason::None,0,rows};
                rows+=static_cast<unsigned>(span.size());
            }
            *a.metadata={0,static_cast<std::int32_t>(rows),rows*static_cast<unsigned>(g.columns)*g.elementBytes};
        }
        return;
    }
    if(g.kind!=DiagnosticSnapshotCopyKind::CountedSelectionRows)
    {
        if(threadIdx.x==0)
        {
            SnapshotObservedHeader header;
            header.liveRows=a.liveRows?*a.liveRows:g.rows;
            std::uint64_t rows=0;
            if(g.kind==DiagnosticSnapshotCopyKind::StatePrefixRows)
            {
                header.priorRows=a.frontier[0];header.liveRows=a.frontier[1];
                const auto total=static_cast<std::int64_t>(header.priorRows)+header.liveRows;
                if(header.priorRows<0||header.liveRows<0||total>g.sourceCapacity)invalid();
                rows=static_cast<std::uint64_t>(total/g.sourceRowsPerStateRow);
            }
            else
            {
                if(header.liveRows<0||header.liveRows>g.rows)invalid();
                rows=header.liveRows;
            }
            if(rows>static_cast<unsigned>(g.rows))invalid();
            header.payloadBytes=rows*static_cast<unsigned>(g.columns)*static_cast<unsigned>(g.elementBytes);
            *a.metadata=header;
        }
        return;
    }
    __shared__ std::uint64_t values[256];
    __shared__ std::uint64_t carry;
    const auto live=a.liveRows?*a.liveRows:g.rows;
    if(live<0||live>g.rows)invalid();
    auto *offsets=reinterpret_cast<std::uint64_t *>(a.metadata+1);
    if(threadIdx.x==0){carry=0;offsets[0]=0;}
    __syncthreads();
    for(int base=0;base<live;base+=256)
    {
        const int row=base+threadIdx.x;
        const int count=row<live?a.counts[row]:0;
        if(count<0||count>g.columns)invalid();
        values[threadIdx.x]=static_cast<unsigned>(count);
        __syncthreads();
        // Exact integer addition is associative. This fixed CTA scan also
        // preserves one execution order on wave32 and wave64 implementations.
        for(unsigned offset=1;offset<256;offset*=2)
        {
            const auto add=threadIdx.x>=offset?values[threadIdx.x-offset]:0;
            __syncthreads();
            values[threadIdx.x]+=add;
            __syncthreads();
        }
        if(row<live)offsets[row+1]=carry+values[threadIdx.x];
        __syncthreads();
        if(threadIdx.x==0)carry+=values[255];
        __syncthreads();
    }
    if(threadIdx.x==0)*a.metadata={0,live,carry*static_cast<unsigned>(g.elementBytes)};
}
/** @brief Coalesced bit copies skip every unowned source cell and retained destination suffix. */
__global__ __launch_bounds__(256,2) void copyAuthoredPayload(DiagnosticSnapshotCopyArgs a)
{
    const auto &g=a.geometry;
    const auto first=static_cast<std::uint64_t>(blockIdx.x)*256+threadIdx.x;
    const auto stride=static_cast<std::uint64_t>(gridDim.x)*256;
    const auto *source=static_cast<const std::uint32_t *>(a.source);
    auto *destination=static_cast<std::uint32_t *>(a.destination);
    if(g.kind==DiagnosticSnapshotCopyKind::AuthoredTokenRows)
    {
        const auto *records=reinterpret_cast<const SnapshotAuthoredTokenRowRecord *>(a.metadata+1);
        const auto rowWords=static_cast<std::uint64_t>(g.columns)*(g.elementBytes/4);
        const auto capacityWords=static_cast<std::uint64_t>(g.rows)*rowWords;
        for(auto word=first;word<capacityWords;word+=stride) {
            const int row=static_cast<int>(word/rowWords);
            const int request=a.work.requestForRow(row);
            const auto &r=records[request-snapshotProjectionFirstRequest(a.work)];
            // Inactive cells are skipped before source access; packed rows
            // retain their original coordinate only in acquired provenance.
            if(row>=r.firstRow&&row<r.endRow)
                destination[(r.payloadRowOffset+row-r.firstRow)*rowWords+word%rowWords]=source[word];
        }
        return;
    }
    if(g.kind!=DiagnosticSnapshotCopyKind::CountedSelectionRows)
    {
        const auto words=a.metadata->payloadBytes/4;
        for(auto word=first;word<words;word+=stride)destination[word]=source[word];
        return;
    }
    // Every CTA has one row owner and touches only its authored prefix.
    // CUDA benefits from one CTA looping over a row. Wave64 targets split
    // columns across CTAs so skewed long rows retain enough resident work.
    // Empty/inactive rows issue no payload traffic on either implementation.
    const auto row=static_cast<unsigned>(blockIdx.x);
    if(row>=static_cast<unsigned>(a.metadata->liveRows)) return;
    const auto *offsets=reinterpret_cast<const std::uint64_t *>(a.metadata+1);
    const auto wordsPerElement=static_cast<unsigned>(g.elementBytes/4);
    const auto begin=offsets[row],end=offsets[row+1];
    const auto sourceStart=static_cast<std::uint64_t>(row)*g.columns*wordsPerElement;
#if defined(DIAGNOSTIC_SNAPSHOT_CUDA)
    for(std::uint64_t word=threadIdx.x;word<(end-begin)*wordsPerElement;word+=256)
        destination[begin*wordsPerElement+word]=source[sourceStart+word];
#else
    const auto word=static_cast<std::uint64_t>(blockIdx.y)*256+threadIdx.x;
    if(word<(end-begin)*wordsPerElement)
        destination[begin*wordsPerElement+word]=source[sourceStart+word];
#endif
}
}
void DSC_LAUNCH(const DiagnosticSnapshotCopyArgs &a,void *stream)
{
    if(!stream||reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::invalid_argument("Native diagnostic snapshot copy requires an exact non-default stream");
    const auto words=static_cast<std::uint64_t>(a.geometry.rows)*a.geometry.columns*(a.geometry.elementBytes/4);
    const auto blocks=static_cast<unsigned>(std::max<std::uint64_t>(1,std::min<std::uint64_t>(65535,(words+255)/256)));
    DSC_NAMESPACE::captureMetadata<<<1,256,0,static_cast<DSC_STREAM>(stream)>>>(a);
#if defined(DIAGNOSTIC_SNAPSHOT_CUDA)
    const dim3 grid=a.geometry.kind==DiagnosticSnapshotCopyKind::CountedSelectionRows
        ? dim3(static_cast<unsigned>(a.geometry.rows)) : dim3(blocks);
#else
    const dim3 grid=a.geometry.kind==DiagnosticSnapshotCopyKind::CountedSelectionRows
        ? dim3(static_cast<unsigned>(a.geometry.rows),(static_cast<unsigned>(a.geometry.columns)*(a.geometry.elementBytes/4)+255)/256)
        : dim3(blocks);
#endif
    DSC_NAMESPACE::copyAuthoredPayload<<<grid,256,0,static_cast<DSC_STREAM>(stream)>>>(a);
    const auto error=DSC_ERROR();
    if(error!=DSC_SUCCESS)throw std::runtime_error(std::string("Native diagnostic snapshot copy launch failed: ")+DSC_ERROR_TEXT(error));
}
} // namespace llaminar2
#undef DSC_STREAM
#undef DSC_ERROR
#undef DSC_SUCCESS
#undef DSC_ERROR_TEXT
#undef DSC_NAMESPACE
#undef DSC_LAUNCH
