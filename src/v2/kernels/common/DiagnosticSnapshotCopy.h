/**
 * @file DiagnosticSnapshotCopy.h
 * @brief Count-aware native snapshot copies into already admitted diagnostic slots.
 *
 * Only authored values and required count metadata move. Worst-case slots stay
 * stable across capture/replay; inactive bytes remain untouched. The observer
 * may later download those counts and exact payload spans, but never supplies
 * them to inference. These copies are absent unless diagnostics are requested.
 */
#pragma once
#include "../../backends/DeviceId.h"
#include "../../snapshots/SnapshotSemanticExtent.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
/** @brief Mathematical source layout, chosen before capture. */
enum class DiagnosticSnapshotCopyKind : std::uint8_t { TokenRows,StatePrefixRows,CountedSelectionRows,AuthoredTokenRows };
/** @brief Immutable maximum geometry with exact native element and grouping units. */
struct DiagnosticSnapshotCopyGeometry
{
    DiagnosticSnapshotCopyKind kind=DiagnosticSnapshotCopyKind::TokenRows;
    int rows=0,columns=0,elementBytes=4;
    int sourceCapacity=0,sourceRowsPerStateRow=1;
    int requests=0; ///< Original requests intersecting the source slice.
};
/** @brief Complete borrowed source/destination/count bindings embedded in a capture. */
struct DiagnosticSnapshotCopyArgs
{
    DiagnosticSnapshotCopyGeometry geometry;
    const void *source=nullptr;
    void *destination=nullptr; ///< Disjoint retained payload capacity; selection data is packed here.
    SnapshotObservedHeader *metadata=nullptr; ///< Header, followed by counted uint64 offsets or fixed original-request records.
    const std::int32_t *frontier=nullptr; ///< State prefix's two adjacent native words, prior then produced.
    const std::int32_t *liveRows=nullptr; ///< Token/selection rows; null is explicitly all active.
    const std::int32_t *counts=nullptr; ///< One authored selected-column count per live token row.
    ProjectionRowWork work=ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(1)); ///< Borrowed original work only for authored intervals.
};
/** @brief Exact retained metadata capacity, not a terminal communication extent.
 * @param geometry Frozen admitted dimensions and mathematical source layout.
 * @return Header plus worst-case counted-row offsets, with overflow checked.
 */
std::size_t diagnosticSnapshotMetadataCapacity(DiagnosticSnapshotCopyGeometry geometry);
/** @brief Reject malformed immutable bindings without reading a device count.
 * @param args Frozen admitted geometry and complete source/destination pointers.
 */
void validateDiagnosticSnapshotCopy(const DiagnosticSnapshotCopyArgs &args);
/** @brief Host-owned counterpart, copying only the exact live native bytes.
 * @param args Host-owned counts and retained destination capacity.
 */
void copyDiagnosticSnapshotCPU(const DiagnosticSnapshotCopyArgs &args);

/** @brief Setup-selected stateless native copy bundle, without mutable inference state. */
class DiagnosticSnapshotCopy final
{
public:
    /** @brief Select the exact compiled GPU implementation outside capture.
     * @param device Participant owning every native pointer passed to capture().
     * @return Immutable value-type function binding; no workspace allocation.
     */
    static DiagnosticSnapshotCopy prepare(DeviceId device);
    /** @brief Record metadata and exact authored payload copies on an exact stream.
     * @param args Stable source/count/destination bindings, including admitted capacity.
     * @param stream Exact non-null native execution stream; never replaced or synchronized.
     */
    void capture(const DiagnosticSnapshotCopyArgs &args,void *stream) const;
    /** @return Exact participant selected at setup. */
    DeviceId device() const noexcept{return device_;}
private:
    using Launch=void (*)(const DiagnosticSnapshotCopyArgs &,void *);
    /** @brief Construct only with an authenticated compiled native launcher. */
    DiagnosticSnapshotCopy(DeviceId device,Launch launch):device_(device),launch_(launch){}
    DeviceId device_;Launch launch_;
};
#ifdef HAVE_CUDA
/** @brief CUDA compilation boundary; requires a validated exact stream. */
void launchCUDADiagnosticSnapshotCopy(const DiagnosticSnapshotCopyArgs &args,void *stream);
#endif
#ifdef HAVE_ROCM
/** @brief HIP compilation boundary; requires a validated exact stream. */
void launchROCmDiagnosticSnapshotCopy(const DiagnosticSnapshotCopyArgs &args,void *stream);
#endif
} // namespace llaminar2
