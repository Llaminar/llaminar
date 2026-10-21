/**
 * @file HostPipelineActivationRows.h
 * @brief Exact host-owned activation extent at a declared segmented pipeline boundary.
 *
 * A host boundary owns its current live rows, including an empty submission.
 * This immutable invocation value names that source and destination; it is not
 * a graph-cache field or a mirror of a GPU counter. Native pipeline graphs bind
 * their local hidden bank directly. A distinct device-counted source without
 * this independently admitted boundary is invalid, never a capacity-sized copy.
 */
#pragma once
#include "backends/DeviceId.h"
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
namespace llaminar2
{
class TensorBase;
/** @brief Source-bound admission for a single host-owned segmented handoff. */
class HostPipelineActivationRows final
{
public:
    /**
     * @brief Seal the live extent while the canonical host boundary owns it.
     * @param source Previous-stage tensor retained across the boundary.
     * @param destination Exact receiving participant.
     * @param requests Positive request count.
     * @param liveRowsPerRequest Current real rows, including zero, never a bucket capacity.
     * @return One immutable invocation value; refresh it at every host submission.
     */
    static HostPipelineActivationRows admitted(const TensorBase *source, DeviceId destination,
        int requests, int liveRowsPerRequest)
    {
        if (!source || (!destination.is_cpu() && !destination.is_gpu()) ||
            requests <= 0 || liveRowsPerRequest < 0 ||
            (liveRowsPerRequest && requests > std::numeric_limits<int>::max() / liveRowsPerRequest))
            throw std::invalid_argument("Host pipeline activation requires exact source/device and representable live rows");
        return HostPipelineActivationRows(source, destination, requests, liveRowsPerRequest);
    }
    /**
     * @brief Authenticate this boundary against the current physical invocation.
     * @param source Exact current source; a marker cannot authorize a foreign tensor.
     * @param destination Current consumer participant.
     * @param requests Current request count.
     * @param physicalRowsPerRequest Fixed invocation stride.
     * @return Exact contiguous live row extent without touching execution data.
     * @throws std::invalid_argument For stale identity, over-capacity rows or request-local holes.
     */
    std::size_t rowsFor(const TensorBase *source, DeviceId destination,
        int requests, int physicalRowsPerRequest) const
    {
        if (source != source_ || destination != destination_ || requests != requests_ ||
            physicalRowsPerRequest <= 0 || live_rows_ > physicalRowsPerRequest ||
            requests > std::numeric_limits<int>::max() / physicalRowsPerRequest)
            throw std::invalid_argument("Host pipeline activation marker does not match its source, participant or row capacity");
        // Current segmented rank handoffs are single-request. A full batched
        // matrix is also contiguous; partial request-major holes need explicit
        // graph-native row ownership, never a flattened-prefix approximation.
        if (requests > 1 && live_rows_ != 0 && live_rows_ != physicalRowsPerRequest)
            throw std::invalid_argument("Host pipeline activation cannot flatten padded request-local row holes");
        return static_cast<std::size_t>(requests_) * static_cast<std::size_t>(live_rows_);
    }
private:
    /** @brief Only checked admission constructs a source-bound row value. */
    HostPipelineActivationRows(const TensorBase *source, DeviceId destination, int requests, int rows)
        : source_(source), destination_(destination), requests_(requests), live_rows_(rows) {}
    const TensorBase *source_;
    DeviceId destination_;
    int requests_;
    int live_rows_;
};
/** @brief Materialization seals topology and must not read or copy live activation bytes. */
enum class HostPipelineBoundaryExecution { Execute, MaterializeOnly };
/**
 * @brief Reject a distinct counted alias without a source-bound host admission.
 * @param binding Current canonical host boundary, never cached.
 * @param source Current external source.
 * @param destination Exact receiving participant.
 * @param requests Current request count.
 * @param physicalRowsPerRequest Fixed invocation stride.
 * @param execution Setup binds addresses with zero live bytes; execution requires admission.
 * @return Exact admitted contiguous live rows, including zero.
 */
inline std::size_t requireHostPipelineActivationRows(
    const std::optional<HostPipelineActivationRows> &binding,
    const TensorBase *source, DeviceId destination, int requests, int physicalRowsPerRequest,
    HostPipelineBoundaryExecution execution = HostPipelineBoundaryExecution::Execute)
{
    if (!source || (!destination.is_cpu() && !destination.is_gpu()) ||
        requests <= 0 || physicalRowsPerRequest <= 0 ||
        requests > std::numeric_limits<int>::max() / physicalRowsPerRequest)
        throw std::invalid_argument("Host pipeline activation setup requires exact source/device and representable capacity");
    if (!binding && execution == HostPipelineBoundaryExecution::MaterializeOnly)
        return 0;
    if (!binding)
        throw std::invalid_argument(
            "Distinct PP activation requires an admitted host-boundary live extent; "
            "device-counted native PP must receive into its canonical hidden bank");
    return binding->rowsFor(source, destination, requests, physicalRowsPerRequest);
}
/** @brief First-use graph routing must not submit the already completed host prelude twice. */
enum class HostPipelineBoundarySubmission { Current, AlreadySubmitted };
/**
 * @brief Resolve current metadata once per host submission and copy only its real extent.
 * @param execution Explicit execution versus setup-only materialization.
 * @param submission Whether the same invocation already submitted its initial prelude.
 * @param resolve Current host resolver; its result is never stored in graph-cache identity.
 * @param copy Existing TransferEngine operation, invoked only for positive live bytes.
 *
 * The resolver validates bindings even during materialization. There is no
 * allocation, device count readback, native launch or second transfer authority
 * here; GPU native counted graphs retain their alias and never enter this copy.
 */
template<class Resolve, class Copy>
void consumeHostPipelineActivationBoundary(HostPipelineBoundaryExecution execution,
    HostPipelineBoundarySubmission submission, Resolve &&resolve, Copy &&copy)
{
    if (submission == HostPipelineBoundarySubmission::AlreadySubmitted)
        return;
    const auto info = resolve();
    if (execution == HostPipelineBoundaryExecution::Execute && info.needs_copy && info.copy_bytes > 0)
        copy(info);
}
} // namespace llaminar2
