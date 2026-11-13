/**
 * @file GDNPreparedRetainedInvocation.h
 * @brief Immutable GPU retained-recurrence invocation over stage-leased workspace.
 *
 * Native geometry is the sole sizing authority. The stage retains arena leases
 * and the kernel retains canonical live state ownership; this host object only
 * freezes borrowed pointer/policy/workspace-owner metadata before graph recording.
 */
#pragma once
#include "kernels/gdn/GDNDenseCompactSnapshots.h"
#include <span>
#include <memory>
#include <limits>
#include <stdexcept>
namespace llaminar2::gdn
{
/** @brief Actual arena views plus exact geometry, never inferred from allocation capacity. */
struct GDNRetainedInvocationBinding final
{
    GDNDenseCompactGeometry geometry;
    GDNDenseCompactTerminalKind terminal_kind;
    std::span<float> storage;
    std::span<float> terminal;
    std::uint64_t
        workspace_owner_identity; ///< Stable existing manager id, NOT allocator/device generation.
    void *capture_stream;
};
/** @brief Stage-owned immutable capture program borrowing backend state and arena/source owners.
 * @details Kernel and all leased/source owners must outlive this object and every
 * graph borrowing it. Capture uses its frozen exact stream. Accepted publication
 * may use another exact stream through existing StageGPUExecution device/event
 * authority; no stream binding is temporarily changed on the shared kernel.
 */
class IGDNPreparedRetainedInvocation
{
  public:
    virtual ~IGDNPreparedRetainedInvocation() = default;
    /** @return Complete unchanged native tuple, with backend-authoritative initial state. */
    virtual const GDNDenseCompactCaptureInput &captureInput() const noexcept = 0;
    /** @return Actual leased views/workspace-owner identity and capture stream; no allocations are
     * owned here. */
    virtual const GDNRetainedInvocationBinding &binding() const noexcept = 0;
    /** @return Canonical state destination for accepted publication, not an unaccepted terminal. */
    virtual float *publicationDestination() const noexcept = 0;
    /** @return Closed backend/terminal/compiled scheduling identity; full equality uses
     * sameCaptureIdentity. */
    virtual std::uint64_t captureProgramIdentity() const noexcept = 0;
    /** @brief Compare all frozen sources, owners, geometry, workspace-owner identity, stream and
     * compiled program.
     * @param other Prepared program proposed for retained reuse.
     * @return Exact immutable identity match, never a live-state comparison. */
    virtual bool
    sameCaptureIdentity(const IGDNPreparedRetainedInvocation &other) const noexcept = 0;
    /** @brief Record/enqueue exact recurrence from frozen raw merged operands.
     * @return Native launch succeeded; false is fatal at the stage, never a fallback request. */
    virtual bool capture() const = 0;
    /** @brief Publish one existing typed selection under the caller's exact publication DAG.
     * @param selection Existing immutable device metadata view, actual capacity and meaning.
     * @param stream Exact nondefault stream supplied by StageGPUExecution on this same device.
     * @return Native publication/coherence enqueues succeeded; no host payload
     * read/query/allocation.
     * @details Caller owns the existing device/context and producer-event wait.
     * This stream may differ from capture_stream; no shared-kernel stream swap occurs. */
    virtual bool publish(const GDNDenseAcceptedSelection &selection, void *stream) const = 0;
};
/** @brief Build one native tuple from leased capacity and backend-authoritative state.
 * @param sources Frozen source-only merged operands; legacy state/snapshots must be absent.
 * @param binding Existing geometry, actual arena views and stable workspace manager identity.
 * @param initial Backend-owned complete scalar or packed live bank, never a host mirror.
 * @param layout Immutable backend/cache state orientation.
 * @param arithmetic Exact backend arithmetic contract.
 * @return Validated native tuple using exactly geometry's required prefix of each leased view.
 * @throws std::invalid_argument For missing/undersized/aliased/overflowing immutable bindings.
 * @details The transaction's opaque generation token carries manager identity,
 * NOT WorkspaceAllocator deviceGeneration. Stage leases and the outer graph owner
 * retain allocation lifetime and existing device-generation freshness authority. */
inline GDNDenseCompactCaptureInput
makeGDNRetainedCaptureInput(const MergedRecurrenceInput &sources,
                            const GDNRetainedInvocationBinding &binding, const float *initial,
                            GDNRecurrenceStateLayout layout, GDNDenseCompactArithmetic arithmetic)
{
    const auto &g = binding.geometry;
    const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!binding.workspace_owner_identity ||
        reinterpret_cast<std::uintptr_t>(binding.capture_stream) <= 2 ||
        binding.storage.size() < g.bankFloats() || binding.storage.size() > maximum ||
        binding.terminal.size() > maximum)
        throw std::invalid_argument("Prepared GDN retained workspace/owner/stream is incomplete");
    planar_recurrence_detail::Range complete{};
    if (!planar_recurrence_detail::range(binding.storage.data(), binding.storage.size_bytes(),
                                         complete))
        throw std::invalid_argument("Prepared GDN retained view has an invalid address extent");
    const auto terminal = [&]
    {
        if (binding.terminal_kind == GDNDenseCompactTerminalKind::RetainedPrefixesOnly)
        {
            if (!binding.terminal.empty() || binding.terminal.data())
                throw std::invalid_argument(
                    "Prepared retained-only GDN cannot borrow a dummy terminal view");
            return GDNDenseCompactTerminalOutput::retainedPrefixesOnly();
        }
        if (binding.terminal_kind != GDNDenseCompactTerminalKind::DenseDestination ||
            binding.terminal.size() < g.denseBankFloats() ||
            !planar_recurrence_detail::range(binding.terminal.data(), binding.terminal.size_bytes(),
                                             complete))
            throw std::invalid_argument("Prepared GDN Dense terminal view is incomplete");
        return GDNDenseCompactTerminalOutput::denseDestination(binding.terminal.data());
    }();
    GDNDenseCompactCaptureInput input{sources,
                                      {g, initial, binding.storage.data(), g.bankFloats(),
                                       binding.workspace_owner_identity, layout, arithmetic,
                                       g.encoding()},
                                      terminal};
    if (!validGDNDenseCompactCaptureStorage(input))
        throw std::invalid_argument(
            "Prepared GDN retained source/state/workspace tuple is invalid");
    return input;
}
/** @brief Protect every cache-owned live bank from speculative capture writes.
 * @param input Validated source/retained tuple with required logical write extents.
 * @param state One actual cache-owned bank view; empty optional secondary is allowed.
 * @return Retained/output/terminal writes are disjoint from that persistent owner.
 * @details This span is overlap protection only, never allocation admission or communication
 * volume. */
inline bool retainedCaptureWritesDisjointFromState(const GDNDenseCompactCaptureInput &input,
                                                   std::span<const float> state) noexcept
{
    if (state.empty())
        return state.data() == nullptr;
    if (state.size() > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    using namespace planar_recurrence_detail;
    Range live{}, output{}, retained{}, terminal{};
    const auto &g = input.transaction.geometry;
    if (!range(state.data(), state.size_bytes(), live) ||
        !range(input.sources.output, g.outputFloats() * sizeof(float), output) ||
        !range(input.transaction.storage, g.bankBytes(), retained) || overlaps(live, output) ||
        overlaps(live, retained))
        return false;
    if (input.terminal.kind() == GDNDenseCompactTerminalKind::DenseDestination)
        return range(input.terminal.destination(), g.denseBankBytes(), terminal) &&
               !overlaps(live, terminal);
    return true;
}
/** @brief Reject a metadata view manufactured from a persistent state owner.
 * @param selection Actual typed device metadata and addressed span (not capacity traffic).
 * @param state One cache-owned bank, including an optional empty secondary.
 * @return Complete immutable source span is disjoint from that state bank. */
inline bool retainedSelectionDisjointFromState(const GDNDenseAcceptedSelection &selection,
                                               std::span<const float> state) noexcept
{
    if (state.empty())
        return state.data() == nullptr;
    if (state.size() > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    planar_recurrence_detail::Range metadata{}, live{};
    return planar_recurrence_detail::range(selection.data(), selection.addressedBytes(),
                                           metadata) &&
           planar_recurrence_detail::range(state.data(), state.size_bytes(), live) &&
           !planar_recurrence_detail::overlaps(metadata, live);
}
/** @brief Exact immutable numerical/source identity, never padded-object memcmp or payload hashing.
 * @param a First native retained tuple.
 * @param b Candidate retained tuple.
 * @return All source/count/policy/geometry/storage/state/output identities match. */
inline bool sameGDNRetainedCaptureInput(const GDNDenseCompactCaptureInput &a,
                                        const GDNDenseCompactCaptureInput &b) noexcept
{
    const auto &x = a.sources;
    const auto &y = b.sources;
    const auto &g = a.transaction.geometry;
    const auto &h = b.transaction.geometry;
    const auto p = g.retentionProgram(), q = h.retentionProgram();
    return g.heads() == h.heads() && g.keyWidth() == h.keyWidth() &&
           g.valueWidth() == h.valueWidth() && g.requests() == h.requests() &&
           g.rowsPerRequest() == h.rowsPerRequest() && p.mode == q.mode &&
           p.compact_min_live_rows == q.compact_min_live_rows &&
           a.transaction.initial_state == b.transaction.initial_state &&
           a.transaction.storage == b.transaction.storage &&
           a.transaction.storage_floats == b.transaction.storage_floats &&
           a.transaction.generation == b.transaction.generation &&
           a.transaction.layout == b.transaction.layout &&
           a.transaction.arithmetic == b.transaction.arithmetic &&
           a.transaction.encoding == b.transaction.encoding &&
           a.terminal.kind() == b.terminal.kind() &&
           a.terminal.destination() == b.terminal.destination() &&
           x.rows.requests() == y.rows.requests() &&
           x.rows.rowsPerRequest() == y.rows.rowsPerRequest() &&
           x.rows.lengthOwner() == y.rows.lengthOwner() && x.key_heads == y.key_heads &&
           x.value_heads == y.value_heads && x.key_width == y.key_width &&
           x.value_width == y.value_width && x.global_value_offset == y.global_value_offset &&
           x.use_qk_l2norm == y.use_qk_l2norm && x.merged == y.merged && x.alpha == y.alpha &&
           x.beta_raw == y.beta_raw && x.A_log == y.A_log && x.dt_bias == y.dt_bias &&
           x.output == y.output && x.initial_state == y.initial_state &&
           x.updated_state == y.updated_state && x.snapshots == y.snapshots &&
           x.snapshot_stride_floats == y.snapshot_stride_floats &&
           x.max_snapshot_rows == y.max_snapshot_rows && x.recurrence_layout == y.recurrence_layout;
}
/** @brief Compare actual leased view identity and stable workspace manager identity.
 * @param a First immutable binding.
 * @param b Proposed reuse binding.
 * @return Exact views/stream/owner token match; geometry is compared by native input equality. */
inline bool sameGDNRetainedBindingViews(const GDNRetainedInvocationBinding &a,
                                        const GDNRetainedInvocationBinding &b) noexcept
{
    return a.storage.data() == b.storage.data() && a.storage.size() == b.storage.size() &&
           a.terminal.data() == b.terminal.data() && a.terminal.size() == b.terminal.size() &&
           a.workspace_owner_identity == b.workspace_owner_identity &&
           a.capture_stream == b.capture_stream && a.terminal_kind == b.terminal_kind;
}
/** @brief Authenticate a ROCm capture stream cold without exposing HIP headers to generic
 * consumers.
 * @param device Exact native owner ordinal.
 * @param stream Explicit stream before recording begins.
 * @return Native stream belongs to device and capture is inactive; no kernel is enqueued. */
bool prepareROCmGDNRetainedCaptureStream(int device, void *stream);

} // namespace llaminar2::gdn
