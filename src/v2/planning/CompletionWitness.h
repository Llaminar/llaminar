/**
 * @file CompletionWitness.h
 * @brief Bounded, deterministic all-frame completion-order proof for pooled pages.
 *
 * A witness is a proof about legal future transitions of the page authority,
 * produced before a transaction is admitted. It is not a scheduler and it
 * carries no independent capacity ledger: every frame counted here is the
 * authority's P/O/G/T/F state (design §5.2). Ranking scores are not witnesses.
 *
 * Scope note: this contract is the page-frame resource subset of the complete
 * all-resource/topology witness (design §5.3, plan P0/P5). It deliberately
 * refuses anything it cannot prove for frame occupancy, and callers must
 * compose it with the remaining resource classes before execution admission.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2::planning
{
/** @brief One request's page-frame demand at its declared peak. */
struct WitnessRequest
{
    std::uint64_t generation = 0;  ///< Request generation (may repeat across records).
    std::size_t record = 0;        ///< Unique pool request-record identity for matching.
    std::size_t peak_frames = 0;   ///< Complete resident working set at peak.
    std::size_t occupied_frames = 0; ///< Live private frames plus outstanding grants when labelled.
    bool preserved = false;        ///< Lower-tier preservation available for this request.
    bool operator==(const WitnessRequest &) const = default;
};

/** @brief Physical occupancy outside any live request record, counted once each. */
struct WitnessBaseOccupancy
{
    std::size_t shared_frames = 0;   ///< Immutable shared frames retained by references.
    std::size_t retiring_frames = 0; ///< Pinned frames awaiting final borrower retirement.
    std::size_t total() const noexcept { return shared_frames + retiring_frames; }
    bool operator==(const WitnessBaseOccupancy &) const = default;
};

/** @brief One enforceable ordered phase in the completion proof. */
struct WitnessStep
{
    enum class Kind : std::uint8_t { RetireResident, Preserve, Restore, Advance, FinalRetire };
    Kind kind = Kind::RetireResident;
    std::uint64_t request_generation = 0;
    std::size_t frames_required = 0;
    std::size_t frames_freed = 0;
    bool operator==(const WitnessStep &) const = default;
};

/** @brief Concrete proof result; an empty plan means no provable completion order. */
struct CompletionWitness
{
    std::vector<WitnessStep> steps;
    std::string refusal; ///< Typed reason when steps are empty; not "hardware impossible" unless stated.
    bool valid() const noexcept { return !steps.empty(); }
};

/** @brief Fixed-size witness scratch; no per-admission heap allocation. */
class WitnessWorkspace final
{
public:
    /** @brief Bound exactly from admitted request-record geometry. */
    explicit WitnessWorkspace(std::size_t records)
        : requests_(records + 1), steps_((records + 1) * 2 + 1) {}
    std::span<WitnessRequest> requests() noexcept { return requests_; }
    /** @return Fixed step cells (at most 2 per request + terminal). */
    std::span<WitnessStep> steps() noexcept { return steps_; }
    /** @return Typed refusal buffer, fixed size. */
    char *refusal() noexcept { return refusal_; }
    static constexpr std::size_t refusalCapacity() noexcept { return sizeof(refusal_); }
private:
    std::vector<WitnessRequest> requests_;
    std::vector<WitnessStep> steps_;
    char refusal_[256] = {0};
};

/**
 * @brief Decide whether a completion order exists under the supplied geometry.
 * @param frames Total materialized frames P.
 * @param requests Declared peak/occupied frame demand, one per live request.
 * @return A bounded witness (or a typed refusal with no invented order).
 *
 * Invariant per phase: for every live request, occupied + its future peak
 * within the phase must fit frames already occupied plus provably freed
 * capacity in the same or earlier phase. Preservation lets a request continue
 * at its current extent while another advances, but never creates frames.
 */
CompletionWitness proveCompletionOrder(std::size_t frames,
    std::span<const WitnessRequest> requests);
/**
 * @brief Decide a completion order with canonical physical occupancy included.
 * @param frames Total materialized frames P.
 * @param base Occupancy outside live request records (shared, retiring, cache), counted once each.
 * @param requests Declared peak/occupied frame demand, one per live request.
 * @return A bounded witness (or a typed refusal with no invented order).
 */
CompletionWitness proveCompletionOrder(std::size_t frames, const WitnessBaseOccupancy &base,
    std::span<const WitnessRequest> requests);
/**
 * @brief No-heap witness decision into a fixed workspace.
 * @param workspace Preallocated request/step/refusal cells, no allocation here.
 * @return True when a completion order exists; refusal text lands in workspace.
 */
bool proveCompletionOrderInto(WitnessWorkspace &workspace, std::size_t frames,
    const WitnessBaseOccupancy &base, std::span<const WitnessRequest> requests);
/** @brief Bytes of one fixed workspace for the authority BOM. */
std::size_t witnessWorkspaceBytes(std::size_t records);

/** @brief Exact per-phase arithmetic check used by the proof and its tests. */
bool phaseFits(std::size_t frames, std::size_t occupied, std::size_t needed);
} // namespace llaminar2::planning
