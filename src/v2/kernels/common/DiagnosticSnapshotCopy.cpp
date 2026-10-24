/**
 * @file DiagnosticSnapshotCopy.cpp
 * @brief Checked setup and device-free exact authored-byte snapshot copies.
 *
 * Metadata has one native authority. CPU execution may inspect its own counts;
 * GPU launch validation checks immutable geometry only. Host diagnostic packing
 * never initializes or repairs inactive production output cells.
 */
#include "DiagnosticSnapshotCopy.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
namespace
{
/** @brief Authenticate positive geometry and representable maximum byte extents. */
std::size_t capacityBytes(DiagnosticSnapshotCopyGeometry g)
{
    if(g.rows<=0||g.columns<=0||(g.elementBytes!=4&&g.elementBytes!=8)||
       static_cast<std::size_t>(g.columns)>std::numeric_limits<std::size_t>::max()/g.elementBytes||
       static_cast<std::size_t>(g.rows)>std::numeric_limits<std::size_t>::max()/(static_cast<std::size_t>(g.columns)*g.elementBytes))
        throw std::invalid_argument("Diagnostic snapshot copy requires positive, representable native geometry");
    switch(g.kind)
    {
    case DiagnosticSnapshotCopyKind::TokenRows:break;
    case DiagnosticSnapshotCopyKind::AuthoredTokenRows:
        if(g.requests<=0)throw std::invalid_argument("Authored snapshot requires original request capacity");
        break;
    case DiagnosticSnapshotCopyKind::StatePrefixRows:
        if(g.sourceCapacity<=0||g.sourceRowsPerStateRow<=0||
           (static_cast<std::uint64_t>(g.sourceCapacity)+g.sourceRowsPerStateRow-1)/g.sourceRowsPerStateRow>static_cast<unsigned>(g.rows))
            throw std::invalid_argument("Diagnostic state-prefix slot is smaller than its admitted source frontier");
        break;
    case DiagnosticSnapshotCopyKind::CountedSelectionRows:break;
    default:throw std::invalid_argument("Diagnostic snapshot copy has an unknown mathematical layout");
    }
    return static_cast<std::size_t>(g.rows)*g.columns*g.elementBytes;
}
/** @brief Reject overlap and address overflow before writing a persistent diagnostic slot. */
bool overlaps(const void *first,std::size_t firstBytes,const void *second,std::size_t secondBytes)
{
    const auto a=reinterpret_cast<std::uintptr_t>(first),b=reinterpret_cast<std::uintptr_t>(second);
    if(firstBytes>std::numeric_limits<std::uintptr_t>::max()-a||secondBytes>std::numeric_limits<std::uintptr_t>::max()-b)
        throw std::overflow_error("Diagnostic snapshot address extent overflows native pointers");
    return a<b+secondBytes&&b<a+firstBytes;
}
}
std::size_t diagnosticSnapshotMetadataCapacity(DiagnosticSnapshotCopyGeometry g)
{
    (void)capacityBytes(g);
    if(g.kind==DiagnosticSnapshotCopyKind::AuthoredTokenRows) {
        if(static_cast<std::size_t>(g.requests)>(std::numeric_limits<std::size_t>::max()-sizeof(SnapshotObservedHeader))/sizeof(SnapshotAuthoredTokenRowRecord))
            throw std::overflow_error("Authored request metadata exceeds addressable storage");
        return sizeof(SnapshotObservedHeader)+static_cast<std::size_t>(g.requests)*sizeof(SnapshotAuthoredTokenRowRecord);
    }
    if(g.kind!=DiagnosticSnapshotCopyKind::CountedSelectionRows)return sizeof(SnapshotObservedHeader);
    if(static_cast<std::size_t>(g.rows)+1>(std::numeric_limits<std::size_t>::max()-sizeof(SnapshotObservedHeader))/8)
        throw std::overflow_error("Diagnostic counted-row metadata exceeds addressable storage");
    return sizeof(SnapshotObservedHeader)+(static_cast<std::size_t>(g.rows)+1)*8;
}
void validateDiagnosticSnapshotCopy(const DiagnosticSnapshotCopyArgs &a)
{
    const auto bytes=capacityBytes(a.geometry),metadata=diagnosticSnapshotMetadataCapacity(a.geometry);
    if(a.geometry.kind==DiagnosticSnapshotCopyKind::AuthoredTokenRows&&
       (a.work.physicalRows()!=a.geometry.rows||snapshotProjectionRequestCount(a.work)!=a.geometry.requests||
        a.work.captureIdentity().source_capacity<=0||a.frontier||a.liveRows||a.counts))
        throw std::invalid_argument("Authored snapshot must borrow the exact original projection work");
    if(!a.source||!a.destination||!a.metadata||reinterpret_cast<std::uintptr_t>(a.source)%4||
       reinterpret_cast<std::uintptr_t>(a.destination)%4||reinterpret_cast<std::uintptr_t>(a.metadata)%8||
       overlaps(a.source,bytes,a.destination,bytes)||overlaps(a.source,bytes,a.metadata,metadata)||
       overlaps(a.destination,bytes,a.metadata,metadata)||
       (a.geometry.kind==DiagnosticSnapshotCopyKind::StatePrefixRows&&!a.frontier)||
       (a.geometry.kind==DiagnosticSnapshotCopyKind::CountedSelectionRows&&!a.counts)||
       (a.frontier&&reinterpret_cast<std::uintptr_t>(a.frontier)%4)||
       (a.liveRows&&reinterpret_cast<std::uintptr_t>(a.liveRows)%4)||
       (a.counts&&reinterpret_cast<std::uintptr_t>(a.counts)%4))
        throw std::invalid_argument("Diagnostic snapshot copy requires aligned disjoint payload/metadata and exact native count bindings");
}
void copyDiagnosticSnapshotCPU(const DiagnosticSnapshotCopyArgs &a)
{
    validateDiagnosticSnapshotCopy(a);const auto &g=a.geometry;
    const auto rowBytes=static_cast<std::size_t>(g.columns)*g.elementBytes;
    SnapshotObservedHeader header;
    if(g.kind==DiagnosticSnapshotCopyKind::AuthoredTokenRows)
    {
        auto *records=reinterpret_cast<SnapshotAuthoredTokenRowRecord *>(a.metadata+1);
        const auto id=a.work.captureIdentity();
        std::uint64_t rows=0;
        // Validate every publication before touching any payload, including a
        // corrupt later request after an otherwise valid first request.
        for(int record=0;record<g.requests;++record) {
            const int request=snapshotProjectionFirstRequest(a.work)+record;
            const auto observed=a.work.observeHost(request);
            const auto span=observed.span;
            records[record]={observed.frontier,observed.live,
                request*id.logical_rows*id.repeats_per_row-id.first_physical_row,
                id.logical_rows,id.source_capacity,id.first_required_logical_row,span.first,span.end,
                observed.live==0||static_cast<std::int64_t>(observed.frontier)+observed.live<=id.first_required_logical_row
                    ?SnapshotEmptyReason::RankingNotRequired:SnapshotEmptyReason::None,0,rows};
            rows+=static_cast<unsigned>(span.size());
        }
        for(int request=0;request<g.requests;++request) {
            const auto &r=records[request];
            const auto bytes=static_cast<std::size_t>(r.endRow-r.firstRow)*rowBytes;
            if(bytes)std::memcpy(static_cast<std::byte *>(a.destination)+r.payloadRowOffset*rowBytes,
                static_cast<const std::byte *>(a.source)+static_cast<std::size_t>(r.firstRow)*rowBytes,bytes);
        }
        *a.metadata={0,static_cast<std::int32_t>(rows),rows*rowBytes};return;
    }
    header.liveRows=a.liveRows?*a.liveRows:g.rows;
    std::size_t rows=0;
    if(g.kind==DiagnosticSnapshotCopyKind::StatePrefixRows)
    {
        header.priorRows=a.frontier[0];header.liveRows=a.frontier[1];
        const auto total=static_cast<std::int64_t>(header.priorRows)+header.liveRows;
        if(header.priorRows<0||header.liveRows<0||total>g.sourceCapacity)
            throw std::logic_error("Diagnostic state-prefix native frontier is invalid");
        rows=static_cast<std::size_t>(total/g.sourceRowsPerStateRow);
    }
    else
    {
        if(header.liveRows<0||header.liveRows>g.rows)throw std::logic_error("Diagnostic native live row count exceeds capture geometry");
        rows=header.liveRows;
    }
    if(g.kind==DiagnosticSnapshotCopyKind::CountedSelectionRows)
    {
        auto *offsets=reinterpret_cast<std::uint64_t *>(a.metadata+1);offsets[0]=0;
        for(std::size_t row=0;row<rows;++row)
        {
            const auto count=a.counts[row];
            if(count<0||count>g.columns)throw std::logic_error("Diagnostic native selected count exceeds its row capacity");
            offsets[row+1]=offsets[row]+static_cast<unsigned>(count);
        }
        header.payloadBytes=offsets[rows]*g.elementBytes;
        for(std::size_t row=0;row<rows;++row)
            std::memcpy(static_cast<std::byte *>(a.destination)+offsets[row]*g.elementBytes,
                static_cast<const std::byte *>(a.source)+row*rowBytes,(offsets[row+1]-offsets[row])*g.elementBytes);
    }
    else
    {
        if(rows>static_cast<std::size_t>(g.rows))throw std::logic_error("Diagnostic state prefix exceeds its admitted slot");
        header.payloadBytes=rows*rowBytes;
        if(header.payloadBytes)std::memcpy(a.destination,a.source,header.payloadBytes);
    }
    *a.metadata=header;
}
DiagnosticSnapshotCopy DiagnosticSnapshotCopy::prepare(DeviceId device)
{
#ifdef HAVE_CUDA
    if(device.is_cuda())return {device,&launchCUDADiagnosticSnapshotCopy};
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())return {device,&launchROCmDiagnosticSnapshotCopy};
#endif
    throw std::invalid_argument("Diagnostic snapshot copy requires an exact compiled native GPU backend");
}
void DiagnosticSnapshotCopy::capture(const DiagnosticSnapshotCopyArgs &a,void *stream) const
{
    validateDiagnosticSnapshotCopy(a);
    if(!stream||reinterpret_cast<std::uintptr_t>(stream)<=2)
        throw std::invalid_argument("Diagnostic snapshot capture requires an exact non-default native stream");
    launch_(a,stream);
}
} // namespace llaminar2
