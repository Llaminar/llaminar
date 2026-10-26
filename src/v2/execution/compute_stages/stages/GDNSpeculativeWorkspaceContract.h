/**
 * @file GDNSpeculativeWorkspaceContract.h
 * @brief Exact byte contract for retained GDN speculative-state workspace.
 *
 * Grouped MTP verification retains short-convolution history and recurrence
 * prefixes. Existing GPU programs keep full post-row images; an explicitly
 * typed GPU retention contract prices its original compact/full/split geometry.
 * CPU images include pending epoch history, so publication never silently
 * invents a checkpoint boundary. These buffers are persistent
 * members of the captured serial graph family: the graph stages allocate them
 * later, while model admission must price them before expert weights consume
 * VRAM.  This header is the single arithmetic authority used at both lifecycle
 * points so their byte counts cannot drift.
 */

#pragma once

#include "kernels/CPUGDNStateLayout.h"
#include "kernels/CPUSharedGDNSnapshots.h"
#include "kernels/gdn/GDNDenseCompactSnapshots.h"
#include <cstddef>
#include <optional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2::gdn_workspace
{

/**
 * @brief Raw byte footprint of one speculative-state buffer pair.
 *
 * Legacy GPU/short-convolution slots retain images per verifier row. Explicit
 * GPU recurrence retention delegates its complete payload to the checked native
 * geometry. CPU recurrence retains immutable epoch checkpoints and a shared raw
 * log. Work is present only when the selected output contract requires a mutable
 * Dense terminal bank. Arena placement applies alignment to these exact typed
 * payload contributions; capacities are never live communication extents.
 */
struct SpeculativeStateFootprint
{
    std::size_t slot_bytes = 0; ///< Persistent retained rollback payload.
    std::size_t work_bytes = 0; ///< Mutable request-local working images.

    /** @brief Return the unaligned sum while rejecting size_t overflow. */
    [[nodiscard]] std::size_t totalBytes() const
    {
        if (work_bytes > std::numeric_limits<std::size_t>::max() - slot_bytes)
        {
            throw std::overflow_error(
                "GDN speculative workspace footprint exceeds size_t");
        }
        return slot_bytes + work_bytes;
    }
};

namespace detail
{

/** @brief Convert one validated positive geometry scalar to size_t. */
[[nodiscard]] inline std::size_t positiveSize(
    int value,
    std::string_view field)
{
    if (value <= 0)
    {
        throw std::invalid_argument(
            "GDN speculative workspace requires positive " +
            std::string(field));
    }
    return static_cast<std::size_t>(value);
}

/** @brief Checked multiplication used by every shared byte formula. */
[[nodiscard]] inline std::size_t checkedMultiply(
    std::size_t left,
    std::size_t right,
    std::string_view contribution)
{
    if (left != 0u &&
        right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::overflow_error(
            "GDN speculative workspace overflow while multiplying " +
            std::string(contribution));
    }
    return left * right;
}

/**
 * @brief Expand one state image into row slots and request work buffers.
 * @param state_floats Number of FP32 values in one state image.
 * @param slot_rows Flattened verifier rows retained for rollback.
 * @param request_count Maximum request count represented concurrently.
 */
[[nodiscard]] inline SpeculativeStateFootprint footprintFromStateFloats(
    std::size_t state_floats,
    int slot_rows,
    int request_count)
{
    const std::size_t rows = positiveSize(slot_rows, "slot_rows");
    const std::size_t requests = positiveSize(request_count, "request_count");
    const std::size_t state_bytes = checkedMultiply(
        state_floats, sizeof(float), "state elements and FP32 bytes");
    return {
        .slot_bytes = checkedMultiply(
            rows, state_bytes, "verifier rows and state bytes"),
        .work_bytes = checkedMultiply(
            requests, state_bytes, "requests and state bytes"),
    };
}

} // namespace detail

/**
 * @brief Resolve exact physical GPU verifier request partitions from admitted capacity.
 * @param flattened_rows Original complete verifier row envelope, never allocation bytes or live rows.
 * @param request_capacity Authenticated generation request capacity, independent of prefill batch_size.
 * @return Checked fully-active coordinates whose physical rows exactly equal the envelope.
 * @throws std::invalid_argument For nonpositive or nondivisible coordinates.
 * @details This sole flattened-to-request conversion is shared by legacy GPU
 * work-bank pricing and explicit retained geometry. Runtime live counts remain
 * separately device-owned; this cold capacity value never samples them.
 */
[[nodiscard]] inline DeviceRequestRowRanges gpuVerifierRequestRows(
    int flattened_rows, int request_capacity)
{
    (void)detail::positiveSize(flattened_rows, "flattened retained rows");
    (void)detail::positiveSize(request_capacity, "retained request capacity");
    if (flattened_rows % request_capacity != 0)
        throw std::invalid_argument("GPU verifier rows must exactly partition the admitted requests");
    return DeviceRequestRowRanges::fullyActiveRequests(
        request_capacity, flattened_rows / request_capacity);
}

/**
 * @brief Immutable GPU retention geometry shared by cold admission and stage binding.
 * @details This value contributes typed bytes to existing workspace/PMA owners;
 * it owns no allocation, pointer, live count, generation or reservation ledger.
 * Persistent recurrence remains DeviceDense. CPU epoch and legacy full-row
 * contracts remain separate, explicit programs. A stage must carry the same
 * geometry/output contract to native capture and accepted publication; it must
 * never infer a program or logical row capacity from an allocation's size.
 */
class GPURetainedRecurrenceWorkspace final
{
  public:
    /**
     * @brief Bind an already-checked original GPU retention and terminal contract.
     * @param geometry Exact participant heads, DK/DV, requests, physical rows and policy.
     * @param terminal_kind Required Dense terminal or explicit retained-prefixes-only output.
     * @throws std::invalid_argument For unknown terminal mode or unsupported native DK.
     * @throws std::overflow_error If the combined slot/work contribution overflows.
     * @details Both implemented GPU backends retain DK64/128. No backend is selected
     * here: arithmetic identity and persistent axes remain with its native/cache owner.
     */
    explicit GPURetainedRecurrenceWorkspace(
        gdn::GDNDenseCompactGeometry geometry,
        gdn::GDNDenseCompactTerminalKind terminal_kind)
        : geometry_(geometry), terminal_kind_(terminal_kind)
    {
        if (geometry_.keyWidth() != 64 && geometry_.keyWidth() != 128)
            throw std::invalid_argument("GPU retained GDN workspace requires native DK64 or DK128");
        if (terminal_kind_ != gdn::GDNDenseCompactTerminalKind::DenseDestination &&
            terminal_kind_ != gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly)
            throw std::invalid_argument("GPU retained GDN workspace has an unknown terminal contract");
        (void)footprint().totalBytes();
    }

    /**
     * @brief Resolve an authenticated flattened verifier envelope without rounding.
     * @param slot_rows Exact flattened physical rows, not observed/live or aligned capacity.
     * @param request_count Exact retained request capacity from its generation owner.
     * @param value_heads Participant-local value-head count.
     * @param key_width Native DK64/128.
     * @param value_width Original positive DV1..512.
     * @param program Explicit measured retention policy; no live count is sampled here.
     * @param terminal_kind Explicit captured output requirement.
     * @return One checked shape whose request partitions exactly cover slot_rows.
     * @throws std::invalid_argument For nonpositive/nondivisible rows or invalid policy/shape.
     * @throws std::overflow_error For native coordinate or byte-extent overflow.
     * @details Generic prefill batch_size is not request-count authority. Callers
     * must supply the actual admitted verifier request capacity. There is no ceil
     * partition, padding-based recovery or duplicated slot-byte formula here.
     */
    [[nodiscard]] static GPURetainedRecurrenceWorkspace fromFlattenedRows(
        int slot_rows, int request_count, int value_heads, int key_width, int value_width,
        gdn::GDNDenseRetentionProgram program,
        gdn::GDNDenseCompactTerminalKind terminal_kind)
    {
        const auto rows = gpuVerifierRequestRows(slot_rows, request_count);
        return GPURetainedRecurrenceWorkspace(
            gdn::GDNDenseCompactGeometry(value_heads, key_width, value_width,
                rows.requests(), rows.rowsPerRequest(), program), terminal_kind);
    }

    /** @return The same immutable geometry consumed by the native capture tuple. */
    [[nodiscard]] const gdn::GDNDenseCompactGeometry &geometry() const noexcept { return geometry_; }
    /** @return Exact output contract; absence of a pointer never selects this value. */
    [[nodiscard]] gdn::GDNDenseCompactTerminalKind terminalKind() const noexcept { return terminal_kind_; }
    /** @return Checked complete physical row count; no device live scalar is consulted. */
    [[nodiscard]] int physicalRows() const noexcept
    {
        // GDNDenseCompactGeometry already proves this exact product fits INT ABI.
        return geometry_.requests() * geometry_.rowsPerRequest();
    }
    /**
     * @brief Contribute exact existing retained payload and explicit terminal bytes.
     * @return Unaligned typed workspace bytes, before the existing arena placement.
     * @details bankBytes owns cp0, relative checkpoints, records and exclusive
     * Full/Threshold capacity. DenseDestination separately needs the original
     * per-request Dense work bank. RetainedPrefixesOnly needs no dummy terminal.
     */
    [[nodiscard]] SpeculativeStateFootprint footprint() const noexcept
    {
        return {.slot_bytes = geometry_.bankBytes(),
            .work_bytes = terminal_kind_ == gdn::GDNDenseCompactTerminalKind::DenseDestination
                ? geometry_.denseBankBytes() : 0};
    }

  private:
    gdn::GDNDenseCompactGeometry geometry_; ///< Sole native retention-capacity arithmetic authority.
    gdn::GDNDenseCompactTerminalKind terminal_kind_; ///< Complete immutable output requirement.
};

/**
 * @brief Price one explicitly typed GPU retention workspace without legacy reinterpretation.
 * @param workspace Checked actual native geometry and output contract.
 * @return Same raw slot/work contribution used by its eventual stage materialization.
 * @details This overload does not change any current production caller or default.
 */
[[nodiscard]] inline SpeculativeStateFootprint recurrenceStateFootprint(
    const GPURetainedRecurrenceWorkspace &workspace) noexcept
{
    return workspace.footprint();
}

/**
 * @brief Select the single source-owned grouped GPU retained candidate cold.
 * @param flat_rows Original physical verifier rows, before alignment or padding inference.
 * @param requests Original generation/request owner capacity for this graph member.
 * @param heads Exact participant-local or replicated value-head assignment.
 * @param dk Original positive key width.
 * @param dv Original positive value width.
 * @return ExactCorrections/RetainedPrefixesOnly, or the explicit original program
 *         when positive checked geometry is outside retained DK64/128, DV1..512.
 *         The original program keeps its own native admission; no capability is added.
 * @throws std::invalid_argument For malformed rows/widths; std::overflow_error for extents.
 * @details Only grouped GPU callers invoke this shared selector. Ordinary GPU,
 * CPU and prepared scalar ROCm paths never enter it. Absence is a cold explicit
 * geometry decision with unchanged original admission, never recovery after a failed retained bind or launch.
 */
[[nodiscard]] inline std::optional<GPURetainedRecurrenceWorkspace> selectGPUVerifierWorkspace(
    int flat_rows, int requests, int heads, int dk, int dv)
{
    const auto rows = gpuVerifierRequestRows(flat_rows, requests);
    const auto state = detail::checkedMultiply(detail::checkedMultiply(
        detail::positiveSize(heads, "value_heads"), detail::positiveSize(dk, "key_width"),
        "value heads and key width"), detail::positiveSize(dv, "value_width"), "dense matrix width");
    (void)detail::footprintFromStateFloats(state, flat_rows, requests).totalBytes();
    if (std::size_t(requests) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(heads) ||
        state > std::size_t(std::numeric_limits<int>::max()))
        throw std::overflow_error("GPU verifier state/grid extent exceeds the original INT ABI");
    if (dk != 64 && dk != 128) return std::nullopt;
    if (dv > 512) return std::nullopt;
    return GPURetainedRecurrenceWorkspace(gdn::GDNDenseCompactGeometry(
        heads, dk, dv, rows.requests(), rows.rowsPerRequest(), {}),
        gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
}

/**
 * @brief Compute the exact GDN recurrence rollback footprint.
 * @param slot_rows Flattened verifier rows retained for rollback.
 * @param request_count Maximum request count represented concurrently.
 * @param value_heads Participant-local GDN value-head count.
 * @param key_width Per-head key-state width.
 * @param value_width Per-head value-state width.
 * @param storage CPU complete epoch history or device dense matrix representation.
 * @return Exact raw bytes for the persistent slot and mutable work buffers.
 */
[[nodiscard]] inline SpeculativeStateFootprint recurrenceStateFootprint(
    int slot_rows,
    int request_count,
    int value_heads,
    int key_width,
    int value_width,
    GDNStateStorage storage)
{
    const std::size_t state_floats = gdnRecurrenceStateFloats(storage, value_heads, key_width, value_width);
    if (storage == GDNStateStorage::CPUFixedEpoch) {
        const CPUSharedGDNLogicalCapacity capacity(slot_rows,request_count);
        const CPUGDNStateLayout layout(value_heads,key_width,value_width);
        return {.slot_bytes=detail::checkedMultiply(capacity.slotFloats(layout),
            sizeof(float),"aligned shared epoch bytes"),.work_bytes=0};
    }
    return detail::footprintFromStateFloats(
        state_floats, slot_rows, request_count);
}

/**
 * @brief Compute the exact short-convolution rollback footprint.
 * @param slot_rows Flattened verifier rows retained for rollback.
 * @param request_count Maximum request count represented concurrently.
 * @param channels Participant-local fused Q/K/V channel count.
 * @param kernel_size Causal convolution kernel width.
 * @return Exact raw bytes for the persistent slot and mutable work buffers.
 */
[[nodiscard]] inline SpeculativeStateFootprint shortConvStateFootprint(
    int slot_rows,
    int request_count,
    int channels,
    int kernel_size)
{
    if (kernel_size <= 1)
    {
        throw std::invalid_argument(
            "GDN short-convolution speculative workspace requires kernel_size > 1");
    }
    const std::size_t state_floats = detail::checkedMultiply(
        detail::positiveSize(channels, "channels"),
        static_cast<std::size_t>(kernel_size - 1),
        "short-convolution channels and history");
    return detail::footprintFromStateFloats(
        state_floats, slot_rows, request_count);
}

} // namespace llaminar2::gdn_workspace
