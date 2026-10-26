/**
 * @file GDNCUDADenseCapturePlan.h
 * @brief Cold exact-stream CUDA capture scheduling over unchanged Dense retention arithmetic.
 *
 * Canonical hardware inventory (or an explicitly authenticated fixture observation)
 * selects one compiled address schedule before capture. Live rows remain device
 * owned; no property query, mutable cache or scheduling decision occurs at replay.
 */
#pragma once
#include "kernels/gdn/GDNDenseCompactSnapshots.h"
#include <string>
#include <utility>
namespace llaminar2 { struct ComputeDevice; }
namespace llaminar2::gdn
{
/** @brief Closed compiled store/residency program; neither state encoding nor arithmetic. */
enum class CUDADenseRetentionSchedule : std::uint32_t
{
    PartitionOneCTA = 0, ///< Original partition checkpoint pass; no forced two-block allocation.
    CheckpointWordTwoCTA = 1 ///< Qualified DK128 KeyMajor interleaved word stores and min2 bound.
};
/** @brief Immutable stream/device/geometry authority, owned by an existing cold capture owner. */
class CUDADenseRetentionCapturePlan final
{
  public:
    /** @brief Compute actual launch CTAs without device observation or live-row reads.
     * @param geometry Existing checked physical request/head/value geometry.
     * @return Checked R*H*ceil(DV/32), independent of live L and physical M.
     * @throws std::overflow_error If the CTA product cannot be represented. */
    static std::size_t captureCTAs(const GDNDenseCompactGeometry &geometry)
    {
        const auto owners = std::size_t(geometry.requests()) * std::size_t(geometry.heads());
        const auto columns = 1 + (std::size_t(geometry.valueWidth()) - 1) / 32;
        if (owners > std::numeric_limits<std::size_t>::max() / columns)
            throw std::overflow_error("CUDA Dense capture CTA product overflow");
        return owners * columns;
    }
    /** @brief Select only the qualified scheduling family from immutable physical geometry.
     * @param geometry Exact captured shape; native DK64/128 admitted here.
     * @param axes Exact valid state axes.
     * @param multiprocessors Positive native-visible SM count, never an inferred card model.
     * @return Original schedule for <=SM or other shapes; word/min2 only DK128 KeyMajor >SM.
     * @throws std::invalid_argument For unknown axes/width or missing hardware observation. */
    static CUDADenseRetentionSchedule chooseSchedule(const GDNDenseCompactGeometry &geometry,
        GDNRecurrenceStateLayout axes, int multiprocessors)
    {
        if (multiprocessors <= 0 || !validGDNRecurrenceStateLayout(axes) ||
            (geometry.keyWidth() != 64 && geometry.keyWidth() != 128))
            throw std::invalid_argument("CUDA Dense scheduling requires valid native geometry and SM count");
        return geometry.keyWidth() == 128 && axes == GDNRecurrenceStateLayout::KeyMajorValueContiguous &&
            captureCTAs(geometry) > std::size_t(multiprocessors) ?
            CUDADenseRetentionSchedule::CheckpointWordTwoCTA : CUDADenseRetentionSchedule::PartitionOneCTA;
    }
    /** @return Exact native capture-family identity for this scheduling/terminal program. */
    const char *captureFunctionFamily() const noexcept
    {
        if (schedule_ == CUDADenseRetentionSchedule::CheckpointWordTwoCTA)
            return terminal_ == GDNDenseCompactTerminalKind::DenseDestination ?
                "cuda_gdn_dense_compact_packed_capture_kernel" : "cuda_gdn_retained_prefixes_packed_capture_kernel";
        return terminal_ == GDNDenseCompactTerminalKind::DenseDestination ?
            "cuda_gdn_dense_compact_capture_kernel" : "cuda_gdn_retained_prefixes_capture_kernel";
    }
    /** @return Exact immutable compiled scheduling program. */
    CUDADenseRetentionSchedule schedule() const noexcept { return schedule_; }
    /** @return Native stream owner authenticated before capture. */
    int device() const noexcept { return device_; }
    /** @return Observed native-visible SM count; no live query is performed. */
    int multiprocessors() const noexcept { return multiprocessors_; }
    /** @return Native compute-capability identity observed by the existing inventory. */
    int computeCapability() const noexcept { return compute_capability_; }
    /** @return Canonical native UUID, including partition identity where applicable. */
    const std::string &deviceUUID() const noexcept { return uuid_; }
    /** @return Exact borrowed stream; its existing owner must outlive retained graphs/plan. */
    void *stream() const noexcept { return stream_; }
    /** @return Checked immutable total CTA count. */
    std::size_t gridCTAs() const noexcept { return ctas_; }
    /** @brief Validate recording launch tuple without ANY CUDA calls.
     * @param input Current unchanged public numerical tuple.
     * @param stream Exact caller stream, already authenticated cold.
     * @return Exact stream, dimensions, axes, retained policy and terminal-program match. */
    bool matches(const GDNDenseCompactCaptureInput &input, void *stream) const noexcept
    {
        const auto &g = input.transaction.geometry;
        const auto p = g.retentionProgram(), expected = geometry_.retentionProgram();
        return stream == stream_ && reinterpret_cast<std::uintptr_t>(stream) > 2 &&
            g.requests() == geometry_.requests() && g.rowsPerRequest() == geometry_.rowsPerRequest() &&
            g.heads() == geometry_.heads() && g.keyWidth() == geometry_.keyWidth() &&
            g.valueWidth() == geometry_.valueWidth() && input.transaction.layout == axes_ &&
            p.mode == expected.mode && p.compact_min_live_rows == expected.compact_min_live_rows &&
            input.terminal.kind() == terminal_;
    }
    /** @brief Compare exact cold scheduling identity for a retained cache owner.
     * @param other Proposed reused plan.
     * @return Every authority field matches; numeric pointer/workspace identity remains its owner's duty. */
    bool sameBinding(const CUDADenseRetentionCapturePlan &other) const noexcept
    {
        const auto a = geometry_.retentionProgram(), b = other.geometry_.retentionProgram();
        return device_ == other.device_ && multiprocessors_ == other.multiprocessors_ &&
            compute_capability_ == other.compute_capability_ && uuid_ == other.uuid_ && stream_ == other.stream_ &&
            schedule_ == other.schedule_ && ctas_ == other.ctas_ && axes_ == other.axes_ && terminal_ == other.terminal_ &&
            geometry_.requests() == other.geometry_.requests() && geometry_.rowsPerRequest() == other.geometry_.rowsPerRequest() &&
            geometry_.heads() == other.geometry_.heads() && geometry_.keyWidth() == other.geometry_.keyWidth() &&
            geometry_.valueWidth() == other.geometry_.valueWidth() && a.mode == b.mode &&
            a.compact_min_live_rows == b.compact_min_live_rows;
    }
  private:
    friend CUDADenseRetentionCapturePlan prepareCUDADenseRetentionCapturePlan(
        const GDNDenseCompactGeometry &, GDNRecurrenceStateLayout, GDNDenseCompactTerminalKind, int, void *);
    friend CUDADenseRetentionCapturePlan prepareObservedCUDADenseRetentionCapturePlan(
        const GDNDenseCompactGeometry &, GDNRecurrenceStateLayout, GDNDenseCompactTerminalKind,
        const llaminar2::ComputeDevice &, void *);
    /** @brief Construct only after cold native stream/hardware authentication.
     * @param geometry Complete immutable request/head/row/width geometry.
     * @param axes Exact persistent/checkpoint matrix orientation.
     * @param terminal Explicit required-Dense or retained-prefixes-only output contract.
     * @param observed Existing authenticated canonical/fixture hardware record.
     * @param stream Exact authenticated borrowed stream.
     * @details Defined in CUDA translation unit; no default/mutable invalid plan exists. */
    CUDADenseRetentionCapturePlan(const GDNDenseCompactGeometry &geometry,
        GDNRecurrenceStateLayout axes, GDNDenseCompactTerminalKind terminal,
        const llaminar2::ComputeDevice &observed, void *stream);
    GDNDenseCompactGeometry geometry_;
    GDNRecurrenceStateLayout axes_;
    GDNDenseCompactTerminalKind terminal_;
    int device_, multiprocessors_, compute_capability_;
    std::string uuid_;
    void *stream_;
    std::size_t ctas_;
    CUDADenseRetentionSchedule schedule_;
};
/** @brief Prepare from canonical cached inventory on the exact owning stream BEFORE capture.
 * @param geometry Immutable numerical request/head/row/width geometry.
 * @param axes Exact persistent/checkpoint matrix orientation.
 * @param terminal Explicit required-Dense or retained-prefixes-only output contract.
 * @param device Exact CUDA owner ordinal; never a DeviceManager vector index.
 * @param stream Exact explicit stream; must be idle with respect to capture recording.
 * @return Complete immutable scheduling binding; no global cache is created.
 * @throws std::runtime_error For missing cached hardware, mismatched stream/device or active capture. */
CUDADenseRetentionCapturePlan prepareCUDADenseRetentionCapturePlan(
    const GDNDenseCompactGeometry &geometry, GDNRecurrenceStateLayout axes,
    GDNDenseCompactTerminalKind terminal, int device, void *stream);
/** @brief Prepare a model-free fixture's explicit observation, authenticated against native hardware cold.
 * @param geometry Immutable numerical request/head/row/width geometry.
 * @param axes Exact persistent/checkpoint matrix orientation.
 * @param terminal Explicit required-Dense or retained-prefixes-only output contract.
 * @param observed Actual ComputeDevice observation authored from native properties outside capture.
 * @param stream Exact explicit stream; native owner/SM/UUID/capability must match the observation.
 * @return Same immutable binding type used by production; no fake inventory installation.
 * @throws std::runtime_error For malformed/foreign hardware, stream or recording state. */
CUDADenseRetentionCapturePlan prepareObservedCUDADenseRetentionCapturePlan(
    const GDNDenseCompactGeometry &geometry, GDNRecurrenceStateLayout axes,
    GDNDenseCompactTerminalKind terminal, const llaminar2::ComputeDevice &observed, void *stream);
}
