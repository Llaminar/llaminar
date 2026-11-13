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
#include "../tensors/TensorClasses.h"
#include "SnapshotSemanticAxis.h"
#include "../execution/local_execution/device/ProjectionRowPublicationTable.h"
#include <optional>
#include <cstring>
#include <climits>
#include <vector>

namespace llaminar2
{
/** @brief Return the mathematical axis without retaining live execution authorities. */
inline SnapshotSemanticAxis snapshotSemanticAxis(const SnapshotSemanticExtent &extent) noexcept
{ return static_cast<SnapshotSemanticAxis>(extent.index()); }
/** @brief Require the real tensor coherence authority before reading host diagnostic bytes. */
inline bool snapshotHostReadAuthority(const ITensor &tensor)
{
    const auto *owner=dynamic_cast<const TensorBase *>(&tensor);
    if(!owner)throw std::logic_error("Semantic snapshot requires an original tensor coherence authority");
    return owner->hostValid();
}
/** @brief Construct the native copy geometry from an immutable output descriptor. */
inline DiagnosticSnapshotCopyGeometry snapshotCopyGeometry(const StageDumpInfo::OutputBuffer &out)
{
    if(out.rows>INT_MAX||out.cols>INT_MAX) throw std::overflow_error("Snapshot native geometry exceeds INT32");
    DiagnosticSnapshotCopyGeometry g{.rows=static_cast<int>(out.rows),.columns=static_cast<int>(out.cols),.elementBytes=static_cast<int>(out.element_size)};
    if(const auto *s=std::get_if<SnapshotStatePrefixRows>(&out.semantic_extent)) {
        g.kind=DiagnosticSnapshotCopyKind::StatePrefixRows;g.sourceCapacity=s->sourceCapacity;g.sourceRowsPerStateRow=s->sourceRowsPerStateRow;
    } else if(std::holds_alternative<SnapshotCountedSelectionRows>(out.semantic_extent)) g.kind=DiagnosticSnapshotCopyKind::CountedSelectionRows;
    else if(const auto *a=std::get_if<SnapshotAuthoredTokenRows>(&out.semantic_extent)) {
        g.kind=DiagnosticSnapshotCopyKind::AuthoredTokenRows;g.requests=snapshotProjectionRequestCount(a->work);
        if(a->work.physicalRows()!=static_cast<int>(out.rows))throw std::logic_error("Authored snapshot work has a foreign source slice");
    }
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
    if(gpu ? (!owner||*owner!=device) : !snapshotHostReadAuthority(t))
        throw std::logic_error("Snapshot count authority is not valid on its observation device");
    const auto *base=static_cast<const std::byte *>(gpu?t.gpu_data_ptr():t.raw_data());
    if(!base) throw std::logic_error("Snapshot count authority has no prepared native address");
    return reinterpret_cast<const std::int32_t *>(base+binding.byteOffset);
}
/**
 * @brief Resolve an exact inherited native scalar borrow without inspecting its value.
 * @param binding Original tensor-backed or explicit controller-owned publication.
 * @param gpu Native observation lane; host callers may only borrow CPU authorities.
 * @param device Exact observation participant, never inferred from a count.
 * @return Original scalar address, or null for explicit absence.
 * @throws std::logic_error For foreign, stale or malformed static ownership metadata.
 */
inline const std::int32_t *snapshotInheritedCountPointer(
    const SnapshotInheritedInt32Binding &binding,bool gpu,DeviceId device)
{
    if(std::holds_alternative<std::monostate>(binding))return nullptr;
    if(const auto *tensor=std::get_if<SnapshotInt32Binding>(&binding)) {
        if(!tensor->tensor)throw std::logic_error("Inherited tensor binding lacks its original owner");
        return snapshotCountPointer(*tensor,1,gpu,device);
    }
    const auto &native=std::get<SnapshotNativeInt32Borrow>(binding);
    if(!native.data||!native.capacity||native.wordOffset>=native.capacity||
       reinterpret_cast<std::uintptr_t>(native.data)%alignof(std::int32_t)||
       native.capacity>(std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(native.data))/sizeof(std::int32_t)||
       (gpu?native.device!=device||!native.device.is_gpu():!native.device.is_cpu()))
        throw std::logic_error("Inherited native count borrow has foreign or incomplete owner/subview geometry");
    return native.data+native.wordOffset;
}
/**
 * @brief Authenticate prepared original scalar bindings without downloading GPU state.
 * @param authored Exact work used by the producer, with tensor/subview borrows.
 * @param gpu Whether source/count authority is GPU-owned.
 * @param device Exact participant owning every borrowed native address.
 * @return Identical immutable work; no counts are read and no state is repaired.
 * @details For indirect work, the stage also retains the immutable address table
 * through its sealed SerialGraphFamily workspace publication. Its entry binding
 * and generation are part of the producer's capture identity, not a live ledger.
 */
inline ProjectionRowWork snapshotProjectionWork(const SnapshotAuthoredTokenRows &authored,bool gpu,DeviceId device)
{
    const auto id=authored.work.captureIdentity();
    if(id.source_capacity<=0||authored.publications.size()!=static_cast<std::size_t>(snapshotProjectionRequestCount(authored.work)))
        throw std::logic_error("Authored snapshot lacks every original request publication binding");
    if(id.table) {
        const auto &proof=authored.publicationTable;
        if(!proof||proof->data()!=id.table||proof->generation()!=id.table_publication_generation||
           proof->entries().size()!=static_cast<std::size_t>(id.requests)||
           proof->storage().device()!=device||
           proof->storage().sizeBytes()!=static_cast<std::size_t>(id.requests)*sizeof(ProjectionRequestPublication)||
           proof->storage().data()!=static_cast<const void *>(id.table))
            throw std::logic_error("Authored snapshot has a stale or foreign immutable publication-table lease");
    } else if(authored.publicationTable)
        throw std::logic_error("Strided authored snapshot unexpectedly carries an indirect publication owner");
    for(std::size_t record=0;record<authored.publications.size();++record) {
        const int request=snapshotProjectionFirstRequest(authored.work)+static_cast<int>(record);
        const auto &binding=authored.publications[record];
        // GPU setup binds allocated original addresses before their captured
        // producers run. The graph edge owns publication ordering; setup must
        // not require already-authored values or read a mutable host mirror.
        const auto *frontier=snapshotCountPointer(binding.frontier,1,gpu,device);
        const auto *live=snapshotCountPointer(binding.live,1,gpu,device);
        const auto *inherited=snapshotInheritedCountPointer(binding.inheritedLive,gpu,device);
        if(!frontier||!live)throw std::logic_error("Authored snapshot lacks original frontier/live owners");
        if(!id.table&&(frontier!=id.frontiers+static_cast<std::size_t>(request)*id.publication_stride_words||
                      live!=id.counts+static_cast<std::size_t>(request)*id.publication_stride_words))
            throw std::logic_error("Authored snapshot scalar subviews disagree with captured projection work");
        if(id.table) {
            const auto entry=authored.publicationTable->entries()[request];
            if(entry.frontier!=frontier||entry.live!=live)
                throw std::logic_error("Authored snapshot scalar owners differ from the sealed immutable table entries");
        }
        if(inherited!=(id.inherited_counts?id.inherited_counts+static_cast<std::size_t>(request)*id.inherited_count_stride:nullptr))
            throw std::logic_error("Authored snapshot changed inherited count agreement authority");
    }
    if(id.table&&!id.table_publication_generation)
        throw std::logic_error("Authored snapshot has an unsealed indirect publication table");
    return authored.work;
}
/** @brief Supply exact native count pointers to a setup-validated copy operation. */
inline DiagnosticSnapshotCopyArgs snapshotCopyArgs(const StageDumpInfo::OutputBuffer &out,void *destination,SnapshotObservedHeader *metadata,bool gpu,DeviceId device=DeviceId::invalid())
{
    DiagnosticSnapshotCopyArgs a{.geometry=snapshotCopyGeometry(out),.source=out.data,.destination=destination,.metadata=metadata};
    if(const auto *s=std::get_if<SnapshotTokenRows>(&out.semantic_extent)) a.liveRows=snapshotCountPointer(s->liveRows,1,gpu,device);
    else if(const auto *s=std::get_if<SnapshotStatePrefixRows>(&out.semantic_extent)) a.frontier=snapshotCountPointer(s->frontier,2,gpu,device);
    else if(const auto *s=std::get_if<SnapshotCountedSelectionRows>(&out.semantic_extent)) {
        a.liveRows=snapshotCountPointer(s->liveRows,1,gpu,device);a.counts=snapshotCountPointer(s->counts,out.rows,gpu,device);
    } else if(const auto *s=std::get_if<SnapshotAuthoredTokenRows>(&out.semantic_extent))
        a.work=snapshotProjectionWork(*s,gpu,device);
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
        auto authoredRows=source.observed_authored_rows;
        if(!header) {
            if(!source.data) throw std::logic_error("Host diagnostic observation lacks source bytes");
            if(source.tensor) {
                if(!snapshotHostReadAuthority(*source.tensor))
                    throw std::logic_error("GPU semantic snapshots require an acquired native observation header");
            }
            const auto g=geometry;
            payload_.resize(source.byte_size);
            metadata_.resize((diagnosticSnapshotMetadataCapacity(g)+7)/8);
            auto *m=reinterpret_cast<SnapshotObservedHeader *>(metadata_.data());
            copyDiagnosticSnapshotCPU(snapshotCopyArgs(source,payload_.data(),m,false,DeviceId::cpu()));
            header=m;view_.data=payload_.data();
            if(std::holds_alternative<SnapshotCountedSelectionRows>(source.semantic_extent))
                offsets={reinterpret_cast<const std::uint64_t *>(m+1),static_cast<std::size_t>(m->liveRows)+1};
            if(const auto *a=std::get_if<SnapshotAuthoredTokenRows>(&source.semantic_extent))
                authoredRows={reinterpret_cast<const SnapshotAuthoredTokenRowRecord *>(m+1),static_cast<std::size_t>(snapshotProjectionRequestCount(a->work))};
        }
        const auto shape=validateSnapshotObservedHeader(source.semantic_extent,*header,source.rows,source.cols,source.element_size);
        view_.rows=shape.rows;view_.byte_size=shape.rows*source.cols*source.element_size;view_.tensor=nullptr;
        if(const auto *a=std::get_if<SnapshotAuthoredTokenRows>(&source.semantic_extent))
            authored_=validateSnapshotAuthoredTokenRows(*a,authoredRows,shape);
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
        view_.observed_header=nullptr;view_.observed_row_offsets={};view_.observed_authored_rows={};
    }
    /** @return The validated view, valid while this scoped owner exists. */
    const StageDumpInfo::OutputBuffer &view() const noexcept{return view_;}
/** @return Acquired original-coordinate provenance; absence is distinct from valid-empty. */
    const std::optional<SnapshotAuthoredTokenRowsObservation> &authoredTokenRows() const noexcept{return authored_;}
private:
    std::optional<SnapshotAuthoredTokenRowsObservation> authored_;
    StageDumpInfo::OutputBuffer view_;
    std::vector<std::byte> payload_,canonical_;
    std::vector<std::uint64_t> metadata_;
};
} // namespace llaminar2
