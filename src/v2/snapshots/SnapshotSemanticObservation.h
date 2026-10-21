/**
 * @file SnapshotSemanticObservation.h
 * @brief Host-only reconstruction of validated, authored diagnostic observations.
 *
 * This owner lives for one callback. GPU payloads have already crossed the
 * diagnostic terminal boundary; CPU payloads read only host-owned authorities.
 * Padding is created here, never in inference buffers or transfer payloads.
 */
#pragma once
#include "../execution/compute_stages/IComputeStage.h"
#include "../kernels/common/DiagnosticSnapshotCopy.h"
#include "../tensors/ITensor.h"
#include <cstring>
#include <climits>
#include <vector>

namespace llaminar2
{
/** @brief Semantic axis retained after native count bindings leave scope. */
enum class SnapshotSemanticAxis : unsigned char { Legacy,TokenRows,StatePrefixRows,CountedSelectionRows };
/** @brief Return the mathematical axis without retaining live execution authorities. */
inline SnapshotSemanticAxis snapshotSemanticAxis(const SnapshotSemanticExtent &extent) noexcept
{ return static_cast<SnapshotSemanticAxis>(extent.index()); }
/** @brief Construct the native copy geometry from an immutable output descriptor. */
inline DiagnosticSnapshotCopyGeometry snapshotCopyGeometry(const StageDumpInfo::OutputBuffer &out)
{
    if(out.rows>INT_MAX||out.cols>INT_MAX) throw std::overflow_error("Snapshot native geometry exceeds INT32");
    DiagnosticSnapshotCopyGeometry g{.rows=static_cast<int>(out.rows),.columns=static_cast<int>(out.cols),.elementBytes=static_cast<int>(out.element_size)};
    if(const auto *s=std::get_if<SnapshotStatePrefixRows>(&out.semantic_extent)) {
        g.kind=DiagnosticSnapshotCopyKind::StatePrefixRows;g.sourceCapacity=s->sourceCapacity;g.sourceRowsPerStateRow=s->sourceRowsPerStateRow;
    } else if(std::holds_alternative<SnapshotCountedSelectionRows>(out.semantic_extent)) g.kind=DiagnosticSnapshotCopyKind::CountedSelectionRows;
    else if(!std::holds_alternative<SnapshotTokenRows>(out.semantic_extent)) throw std::logic_error("Legacy snapshot has no native counted-copy geometry");
    return g;
}
/** @brief Resolve a checked INT32 subview on its actual owner, never a stale host mirror. */
inline const std::int32_t *snapshotCountPointer(SnapshotInt32Binding binding,std::size_t words,bool gpu,DeviceId device=DeviceId::invalid())
{
    if(!binding.tensor) return nullptr;
    auto &t=*binding.tensor;
    if(t.native_type()!=TensorType::INT32||binding.byteOffset%4||binding.byteOffset>t.size_bytes()||words>(t.size_bytes()-binding.byteOffset)/4)
        throw std::invalid_argument("Snapshot native count binding exceeds INT32 storage");
    const auto owner=t.current_device();
    // current_device() names a GPU allocation and is intentionally absent
    // for ordinary CPU tensors. Host validity is the CPU read authority;
    // allocation residency alone cannot classify a current host observation.
    if(gpu ? (!owner||*owner!=device) : !t.isHostValid())
        throw std::logic_error("Snapshot count authority is not valid on its observation device");
    const auto *base=static_cast<const std::byte *>(gpu?t.gpu_data_ptr():t.raw_data());
    if(!base) throw std::logic_error("Snapshot count authority has no prepared native address");
    return reinterpret_cast<const std::int32_t *>(base+binding.byteOffset);
}
/** @brief Supply exact native count pointers to a setup-validated copy operation. */
inline DiagnosticSnapshotCopyArgs snapshotCopyArgs(const StageDumpInfo::OutputBuffer &out,void *destination,SnapshotObservedHeader *metadata,bool gpu,DeviceId device=DeviceId::invalid())
{
    DiagnosticSnapshotCopyArgs a{.geometry=snapshotCopyGeometry(out),.source=out.data,.destination=destination,.metadata=metadata};
    if(const auto *s=std::get_if<SnapshotTokenRows>(&out.semantic_extent)) a.liveRows=snapshotCountPointer(s->liveRows,1,gpu,device);
    else if(const auto *s=std::get_if<SnapshotStatePrefixRows>(&out.semantic_extent)) a.frontier=snapshotCountPointer(s->frontier,2,gpu,device);
    else if(const auto *s=std::get_if<SnapshotCountedSelectionRows>(&out.semantic_extent)) {
        a.liveRows=snapshotCountPointer(s->liveRows,1,gpu,device);a.counts=snapshotCountPointer(s->counts,out.rows,gpu,device);
    }
    return a;
}
/** @brief Scoped normalized view; its backing bytes outlive extraction in one callback. */
class SnapshotSemanticObservation final
{
public:
    /** @brief Validate native counts and expose only their exact mathematical shape. */
    explicit SnapshotSemanticObservation(const StageDumpInfo::OutputBuffer &source):view_(source)
    {
        if(std::holds_alternative<std::monostate>(source.semantic_extent)) return;
        const auto geometry=snapshotCopyGeometry(source);
        (void)diagnosticSnapshotMetadataCapacity(geometry);
        if(source.byte_size<source.rows*source.cols*source.element_size)
            throw std::invalid_argument("Semantic snapshot source omits its admitted native byte extent");
        const SnapshotObservedHeader *header=source.observed_header;
        auto offsets=source.observed_row_offsets;
        if(!header) {
            if(!source.data) throw std::logic_error("Host diagnostic observation lacks source bytes");
            if(source.tensor) {
                if(!source.tensor->isHostValid())
                    throw std::logic_error("GPU semantic snapshots require an acquired native observation header");
            }
            const auto g=geometry;
            payload_.resize(source.byte_size);
            metadata_.resize((diagnosticSnapshotMetadataCapacity(g)+7)/8);
            auto *m=reinterpret_cast<SnapshotObservedHeader *>(metadata_.data());
            copyDiagnosticSnapshotCPU(snapshotCopyArgs(source,payload_.data(),m,false));
            header=m;view_.data=payload_.data();
            if(std::holds_alternative<SnapshotCountedSelectionRows>(source.semantic_extent))
                offsets={reinterpret_cast<const std::uint64_t *>(m+1),static_cast<std::size_t>(m->liveRows)+1};
        }
        const auto shape=validateSnapshotObservedHeader(source.semantic_extent,*header,source.rows,source.cols,source.element_size);
        view_.rows=shape.rows;view_.byte_size=shape.rows*source.cols*source.element_size;view_.tensor=nullptr;
        if(std::holds_alternative<SnapshotCountedSelectionRows>(source.semantic_extent)) {
            validateSnapshotSelectionOffsets(offsets,shape,source.cols,source.element_size);
            if(!source.dtype||(std::strcmp(source.dtype,"INT32")&&std::strcmp(source.dtype,"INT64")))
                throw std::logic_error("Counted diagnostic selection requires signed native integer identities");
            canonical_.assign(view_.byte_size,std::byte{0xff});
            const auto *packed=static_cast<const std::byte *>(view_.data);
            for(std::size_t row=0;row<shape.rows;++row) {
                const auto bytes=(offsets[row+1]-offsets[row])*source.element_size;
                // A captured count is authenticated by its complete native ID
                // prefix. Reject -1/negative/unsorted authored IDs so the host
                // sentinel boundary remains exactly that captured count.
                std::int64_t previous=-1;
                for(auto index=offsets[row];index<offsets[row+1];++index) {
                    std::int64_t value;
                    if(source.element_size==8) std::memcpy(&value,packed+index*8,8);
                    else {std::int32_t narrow;std::memcpy(&narrow,packed+index*4,4);value=narrow;}
                    if(value<=previous) throw std::logic_error("Diagnostic authored selection IDs must be nonnegative and strictly ordered before padding");
                    previous=value;
                }
                if(bytes) std::memcpy(canonical_.data()+row*source.cols*source.element_size,packed+offsets[row]*source.element_size,bytes);
            }
            view_.data=canonical_.data();
        }
        view_.observed_header=nullptr;view_.observed_row_offsets={};
    }
    /** @return The validated view, valid while this scoped owner exists. */
    const StageDumpInfo::OutputBuffer &view() const noexcept{return view_;}
private:
    StageDumpInfo::OutputBuffer view_;
    std::vector<std::byte> payload_,canonical_;
    std::vector<std::uint64_t> metadata_;
};
} // namespace llaminar2
