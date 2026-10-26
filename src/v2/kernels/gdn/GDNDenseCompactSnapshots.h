/**
 * @file GDNDenseCompactSnapshots.h
 * @brief Checked full-row or exact-correction retention over unchanged Dense GDN state.
 *
 * A retained transaction owns immutable cp0 and either every post-row matrix
 * or eight-row checkpoints with normalized K/exact delta/rounded decay records.
 * Device-owned live counts select the captured retention policy. The format has
 * no persistent phase, backward-query program or raw-V/beta interpretation.
 * Explicit split retention can instead retain each live post-row Dense image.
 * A captured threshold chooses per request from its existing device count; both
 * representations retain immutable cp0. DenseDestination preserves terminal work;
 * explicit RetainedPrefixesOnly omits only the separately unaccepted terminal bank.
 * Geometry contributes one capacity formula; it owns no memory or live ledger.
 * Device-authored counts and whole-kernel/event completion remain the existing
 * publication authority. Pure admission never reads native metadata/payloads.
 */
#pragma once
#include "kernels/gdn/GDNMergedRecurrence.h"
#include <cstddef>
#include <initializer_list>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_GDN_DENSE_COMPACT_HD __host__ __device__
#else
#define LLAMINAR_GDN_DENSE_COMPACT_HD
#endif
namespace llaminar2::gdn
{
/** @brief Closed transaction-only format; never a persistent prefix-cache encoding. */
enum class GDNDenseCompactEncoding : std::uint32_t
{
    ExactCorrectionsV1 = 0x44464331u, ///< DFC1: unchanged fixed-correction controls.
    SplitRetentionV1 = 0x44465331u ///< DFS1: explicit per-request retention and policy identity.
};
/** @brief Closed immutable retention policy; persistent state remains Dense. */
enum class GDNDenseRetentionMode : std::uint32_t
{
    ExactCorrections = 0, ///< Retain fixed eight-row checkpoints and exact updates.
    FullSnapshots = 1, ///< Retain cp0 and every actual post-row matrix.
    LiveRowThreshold = 2 ///< Full snapshots below the captured compact-row threshold.
};
/** @brief Captured policy, not a host mirror of a device-selected live count. */
struct GDNDenseRetentionProgram final
{
    GDNDenseRetentionMode mode = GDNDenseRetentionMode::ExactCorrections;
    unsigned compact_min_live_rows = 0; ///< Positive threshold only for LiveRowThreshold.
};
/** @brief Exact authored forward/update program; no unimplemented backend is advertised. */
enum class GDNDenseCompactArithmetic : std::uint32_t
{
    CUDAEightPartFusedUpdateV1 = 1, ///< Original CUDA eight-part forward, rounded decay then fused update.
    ROCmFourPartFusedUpdateV1 = 2 ///< Original gfx906 four-part forward, rounded decay then delta*K fused update.
};
/** @brief Recognize only implemented exact-forward arithmetic identities.
 * @param arithmetic Immutable backend-specific native reduction/update program.
 * @return Whether the closed representation has a supported arithmetic identity. */
LLAMINAR_GDN_DENSE_COMPACT_HD constexpr bool validGDNDenseCompactArithmetic(
    GDNDenseCompactArithmetic arithmetic) noexcept
{
    return arithmetic == GDNDenseCompactArithmetic::CUDAEightPartFusedUpdateV1 ||
        arithmetic == GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1;
}
/** @brief One live byte span relative to a compact head; padding is not payload. */
struct GDNDenseCompactByteSpan final
{
    std::size_t offset_bytes = 0; ///< Offset within the admitted allocation.
    std::size_t live_bytes = 0; ///< Authored bytes, excluding alignment/capacity.
};
/** @brief Exclusive contiguous element ownership for one matrix publication tile.
 * @details This is a pure coordinate view, not allocation or live-state accounting. */
struct GDNDenseCompactMatrixTileSpan final
{
    std::size_t first_element = 0; ///< First element in this request/head matrix.
    std::size_t live_elements = 0; ///< Exact tile payload, excluding every tail gap.
};
/** @brief Raw protocol words observed only after the producer's exact completion edge.
 * @details Full32-bit format/layout/arithmetic words deliberately precede enum
 * narrowing, so corrupt high bits cannot alias a valid small enum value. This
 * value is not a host mirror or an event/lifetime proof. */
struct GDNDenseCompactPublication final
{
    std::uint32_t format; ///< Exact DFC1 or versioned DFS1 identity, never epoch raw history.
    std::uint32_t published_rows; ///< Completed request-local rows.
    std::uint32_t planned_rows; ///< Exact producer-observed live count.
    std::uint64_t generation; ///< Immutable transaction generation.
    std::uint32_t layout; ///< Full native matrix-axis identity word.
    std::uint32_t arithmetic; ///< Full native arithmetic identity word.
    std::uint32_t retention_mode = 0; ///< DFS1 actual FullSnapshots or ExactCorrections.
    std::uint32_t retention_program = 0; ///< DFS1 immutable captured policy.
    std::uint32_t compact_min_live_rows = 0; ///< DFS1 immutable threshold; zero otherwise.
};
/** @brief Exact relative selection after protocol/count validation; no arithmetic is executed. */
struct GDNDenseCompactSelection final
{
    std::size_t checkpoint = 0; ///< Preceding immutable checkpoint index, including cp0.
    std::size_t first_record = 0; ///< First correction record after that checkpoint.
    unsigned update_count = 0; ///< Accepted suffix0..7, never a model-row replay count.
    GDNDenseRetentionMode retention_mode = GDNDenseRetentionMode::ExactCorrections; ///< Validated source representation.
};
/** @brief Sole immutable sizing authority for Dense input and compact transaction storage. */
class GDNDenseCompactGeometry final
{
  public:
    static constexpr unsigned checkpoint_interval = 8;
    static constexpr std::size_t alignment_bytes = 64;
    static constexpr std::size_t header_floats = 16;
    static constexpr unsigned format_word = 0, published_rows_word = 1, planned_rows_word = 2;
    static constexpr unsigned generation_low_word = 3, generation_high_word = 4;
    static constexpr unsigned layout_word = 5, arithmetic_word = 6, live_header_words = 7;
    static constexpr unsigned retention_mode_word = 7, retention_program_word = 8;
    static constexpr unsigned compact_min_live_rows_word = 9, split_live_header_words = 10;
    /** @brief Check all immutable shape/capacity products before storage exists.
     * @param heads Positive participant-local value heads.
     * @param dk Positive key width1..512; each native entry admits its own widths.
     * @param dv Positive value width1..512.
     * @param requests Independent positive request partitions.
     * @param rows Positive physical row capacity per request, never a live count.
     * @param program Immutable retention policy; default preserves DFC1 controls.
     * @throws std::invalid_argument For unsupported immutable dimensions/policy.
     * @throws std::overflow_error For INT coordinate or size_t element/byte overflow. */
    GDNDenseCompactGeometry(int heads, int dk, int dv, int requests, int rows,
        GDNDenseRetentionProgram program = {})
        : heads_(heads), dk_(dk), dv_(dv), requests_(requests), rows_(rows), program_(program)
    {
        if (heads <= 0 || dk <= 0 || dk > 512 || dv <= 0 || dv > 512 || requests <= 0 || rows <= 0)
            throw std::invalid_argument("Dense compact GDN requires positive requests/heads/rows and widths1..512");
        if ((program.mode != GDNDenseRetentionMode::ExactCorrections &&
                program.mode != GDNDenseRetentionMode::FullSnapshots &&
                program.mode != GDNDenseRetentionMode::LiveRowThreshold) ||
            (program.mode == GDNDenseRetentionMode::LiveRowThreshold
                ? (!program.compact_min_live_rows || program.compact_min_live_rows > unsigned(rows))
                : program.compact_min_live_rows != 0))
            throw std::invalid_argument("Dense retention policy/threshold is invalid");
        const auto imax = std::size_t(std::numeric_limits<int>::max());
        if (std::size_t(requests) > imax / std::size_t(heads) ||
            std::size_t(requests) > imax / std::size_t(rows))
            throw std::overflow_error("Dense compact request/head/row extent exceeds INT ABI");
        matrix_ = product(std::size_t(dk), std::size_t(dv));
        checkpoint_stride_ = align(matrix_);
        checkpoints_ = 1 + std::size_t(rows) / checkpoint_interval;
        record_ = sum(sum(std::size_t(dk), std::size_t(dv)), 1);
        record_stride_ = align(record_);
        records_offset_ = align(sum(header_floats, product(checkpoints_, checkpoint_stride_)));
        if (program.mode == GDNDenseRetentionMode::ExactCorrections)
            head_stride_ = align(sum(records_offset_, product(std::size_t(rows), record_stride_)));
        else if (program.mode == GDNDenseRetentionMode::FullSnapshots)
            head_stride_ = align(sum(header_floats,
                product(sum(std::size_t(rows), 1), checkpoint_stride_)));
        else
        {
            // Only live counts below the threshold can reach the full branch.
            // cp0 plus threshold-1 rows is exactly threshold matrix images.
            const auto full_head = align(sum(header_floats,
                product(program.compact_min_live_rows, checkpoint_stride_)));
            const auto correction_head = align(sum(records_offset_, product(std::size_t(rows), record_stride_)));
            head_stride_ = full_head > correction_head ? full_head : correction_head;
        }
        if (program.mode == GDNDenseRetentionMode::FullSnapshots)
            checkpoints_ = sum(std::size_t(rows), 1);
        else if (program.mode == GDNDenseRetentionMode::LiveRowThreshold &&
            program.compact_min_live_rows > checkpoints_)
            checkpoints_ = program.compact_min_live_rows;
        const auto request_heads = product(std::size_t(requests), std::size_t(heads));
        dense_ = product(request_heads, matrix_);
        compact_ = product(request_heads, head_stride_);
        gates_ = product(request_heads, std::size_t(rows));
        outputs_ = product(gates_, std::size_t(dv));
        for (const auto extent : {dense_, compact_, gates_, outputs_}) (void)product(extent, sizeof(float));
    }
    /** @return Exact captured policy; it contains no observed live count. */
    LLAMINAR_GDN_DENSE_COMPACT_HD GDNDenseRetentionProgram retentionProgram() const noexcept { return program_; }
    /** @return Required transaction format; default controls remain DFC1. */
    LLAMINAR_GDN_DENSE_COMPACT_HD GDNDenseCompactEncoding encoding() const noexcept
    {
        return program_.mode == GDNDenseRetentionMode::ExactCorrections ?
            GDNDenseCompactEncoding::ExactCorrectionsV1 : GDNDenseCompactEncoding::SplitRetentionV1;
    }
    /** @brief Derive a request's representation from its sole ordered device count.
     * @param live Observed request rows, including zero.
     * @param mode Reset to corrections on failure; no count is clamped.
     * @return Whether live is within the captured physical capacity. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool retentionForLiveRows(int live, GDNDenseRetentionMode &mode) const noexcept
    {
        mode = GDNDenseRetentionMode::ExactCorrections;
        if (live < 0 || live > rows_) return false;
        if (program_.mode == GDNDenseRetentionMode::FullSnapshots ||
            (program_.mode == GDNDenseRetentionMode::LiveRowThreshold &&
                unsigned(live) < program_.compact_min_live_rows)) mode = GDNDenseRetentionMode::FullSnapshots;
        return true;
    }
    /** @brief Identify exactly which post-row states a sink retains.
     * @param live Ordered request count.
     * @param completed Completed row count1..live, never a physical inactive row.
     * @param slot Checkpoint/image index, reset on rejection; cp0 is always separately required.
     * @return Whether this row owns a retained full matrix. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool checkpointForCompletedRow(int live, int completed,
        std::size_t &slot) const noexcept
    {
        slot = 0;
        GDNDenseRetentionMode mode{};
        if (!retentionForLiveRows(live, mode) || completed <= 0 || completed > live) return false;
        if (mode == GDNDenseRetentionMode::FullSnapshots) { slot = std::size_t(completed); return true; }
        if (unsigned(completed) % checkpoint_interval) return false;
        slot = unsigned(completed) / checkpoint_interval;
        return true;
    }
    /** @brief Identify live exact-correction records without duplicating policy in sinks.
     * @param live Ordered request count.
     * @param row Zero-based completed input row.
     * @return Whether this row owns an exact correction record. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool retainRecordForRow(int live, int row) const noexcept
    {
        GDNDenseRetentionMode mode{};
        return retentionForLiveRows(live, mode) && row >= 0 && row < live &&
            mode == GDNDenseRetentionMode::ExactCorrections;
    }
    /** @return Immutable local value-head count. */
    LLAMINAR_GDN_DENSE_COMPACT_HD int heads() const noexcept { return heads_; }
    /** @return Immutable key width. */
    LLAMINAR_GDN_DENSE_COMPACT_HD int keyWidth() const noexcept { return dk_; }
    /** @return Immutable value width. */
    LLAMINAR_GDN_DENSE_COMPACT_HD int valueWidth() const noexcept { return dv_; }
    /** @return Independent request count. */
    LLAMINAR_GDN_DENSE_COMPACT_HD int requests() const noexcept { return requests_; }
    /** @return Fixed row capacity per request. */
    LLAMINAR_GDN_DENSE_COMPACT_HD int rowsPerRequest() const noexcept { return rows_; }
    /** @return Unpadded Dense head elements; no persistent header/history is added. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t matrixFloats() const noexcept { return matrix_; }
    /** @brief Count independent tiles of this already checked matrix extent.
     * @param tile_elements Positive immutable native tile width, never an allocation reserve.
     * @return Exact nonzero tile count, or0 for an invalid zero width.
     * @details Subtract-before-divide avoids overflow even for SIZE_MAX tile width. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t matrixTileCount(std::size_t tile_elements) const noexcept
    {
        return tile_elements ? 1 + (matrix_ - 1) / tile_elements : 0;
    }
    /** @brief Derive one exclusive bounded matrix tile without reading native state.
     * @param tile_index Zero-based grid.y owner.
     * @param tile_elements Positive immutable native tile width.
     * @param span Reset on failure; otherwise exact first/live element coordinates.
     * @return Whether this tile exists; no clamping of a foreign owner occurs.
     * @details The index check proves index*width <= matrix-1 before multiplication. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool matrixTileFor(std::size_t tile_index,
        std::size_t tile_elements, GDNDenseCompactMatrixTileSpan &span) const noexcept
    {
        span = {};
        const auto count = matrixTileCount(tile_elements);
        if (!count || tile_index >= count) return false;
        span.first_element = tile_index * tile_elements;
        const auto remaining = matrix_ - span.first_element;
        span.live_elements = remaining < tile_elements ? remaining : tile_elements;
        return true;
    }
    /** @return Complete unchanged persistent Dense bank extent. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t denseBankFloats() const noexcept { return dense_; }
    /** @return Complete persistent Dense bytes. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t denseBankBytes() const noexcept { return dense_ * sizeof(float); }
    /** @return Offset of immutable cp0 within one compact head. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t checkpointOffsetFloats() const noexcept { return header_floats; }
    /** @return Aligned allocation stride between relative checkpoints. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t checkpointStrideFloats() const noexcept { return checkpoint_stride_; }
    /** @return Maximum reachable immutable matrix slots across the exclusive retained modes. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t checkpointCount() const noexcept { return checkpoints_; }
    /** @return Offset of normalized K within each exact-update record. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t recordKeyOffsetFloats() const noexcept { return 0; }
    /** @return Offset of exact delta[DV]; it is never raw V. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t recordCorrectionOffsetFloats() const noexcept { return std::size_t(dk_); }
    /** @return Offset of the exact rounded scalar decay. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t recordDecayOffsetFloats() const noexcept { return std::size_t(dk_) + dv_; }
    /** @return Live record words, excluding every alignment gap. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t recordFloats() const noexcept { return record_; }
    /** @return Aligned record allocation stride. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t recordStrideFloats() const noexcept { return record_stride_; }
    /** @return Correction-branch record origin; exclusive full snapshots may overlap this region. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t recordsOffsetFloats() const noexcept { return records_offset_; }
    /** @return Aligned compact head stride, not its communication extent. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t headStrideFloats() const noexcept { return head_stride_; }
    /** @return Complete allocation elements for the reachable exclusive retention modes. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t bankFloats() const noexcept { return compact_; }
    /** @return Complete retained allocation bytes, never the live communication extent. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t bankBytes() const noexcept { return compact_ * sizeof(float); }
    /** @return Complete physical output elements. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t outputFloats() const noexcept { return outputs_; }
    /** @return Complete physical projected-gate elements. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t gateFloats() const noexcept { return gates_; }
    /** @brief Check ordered native protocol against exact capture identity.
     * @param publication Already event-ordered protocol; no device read occurs here.
     * @param generation Expected nonzero captured generation.
     * @param layout Expected immutable matrix axes.
     * @param arithmetic Expected exact native forward/update contract.
     * @return Complete bounded counts and exact full-width identity words.
     * @details Does not replace the producer completion event or stage lease. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool validPublication(const GDNDenseCompactPublication &publication,
        std::uint64_t generation, GDNRecurrenceStateLayout layout, GDNDenseCompactArithmetic arithmetic) const noexcept
    {
        return generation && (layout == GDNRecurrenceStateLayout::KeyMajorValueContiguous ||
            layout == GDNRecurrenceStateLayout::ValueMajorKeyContiguous) &&
            validGDNDenseCompactArithmetic(arithmetic) &&
            publication.format == std::uint32_t(encoding()) &&
            publication.generation == generation && publication.layout == std::uint32_t(layout) &&
            publication.arithmetic == std::uint32_t(arithmetic) &&
            publication.published_rows == publication.planned_rows &&
            publication.published_rows <= unsigned(rows_) && validRetentionIdentity(publication);
    }
    /** @brief Select immutable base and bounded accepted suffix without executing updates.
     * @param publication Exact completed request/head protocol.
     * @param generation Captured generation.
     * @param layout Captured matrix axes.
     * @param arithmetic Captured native arithmetic.
     * @param accepted Count0..published; negative/out-of-range fails, never clamps.
     * @param selection Result reset before validation; valid only on true.
     * @return Whether the exact accepted Dense value can be reconstructed. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool select(const GDNDenseCompactPublication &publication,
        std::uint64_t generation, GDNRecurrenceStateLayout layout, GDNDenseCompactArithmetic arithmetic,
        int accepted, GDNDenseCompactSelection &selection) const noexcept
    {
        selection = {};
        if (!validPublication(publication, generation, layout, arithmetic) || accepted < 0 ||
            unsigned(accepted) > publication.published_rows) return false;
        return selectForLiveRows(int(publication.published_rows), accepted, selection);
    }
    /** @brief Derive selection after the owner's whole-plan authentication, without header reads.
     * @param live Same immutable request count authenticated by the publication owner.
     * @param accepted Accepted count0..live; never clamped.
     * @param selection Reset on failure; exact image/base and bounded correction suffix.
     * @return Whether both counts fit this captured program's physical geometry.
     * @details This pure coordinate helper is not a substitute for whole-plan protocol validation. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool selectForLiveRows(int live, int accepted,
        GDNDenseCompactSelection &selection) const noexcept
    {
        selection = {};
        GDNDenseRetentionMode mode{};
        if (!retentionForLiveRows(live, mode) || accepted < 0 || accepted > live) return false;
        selection.retention_mode = mode;
        if (mode == GDNDenseRetentionMode::FullSnapshots)
        {
            selection.checkpoint = std::size_t(accepted);
            return true;
        }
        selection.checkpoint = std::size_t(accepted) / checkpoint_interval;
        selection.first_record = selection.checkpoint * checkpoint_interval;
        selection.update_count = unsigned(accepted) % checkpoint_interval;
        return true;
    }
    /** @brief Enumerate exact published bytes, including immutable cp0 for live0.
     * @param live Ordered completed row count0..physical capacity.
     * @param index Protocol0, then actual checkpoints, then individual live records.
     * @param span Result reset on failure, excludes padding and unused slots.
     * @return Whether that exact span exists; no state or metadata is read. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool publishedSpanFor(unsigned live, std::size_t index,
        GDNDenseCompactByteSpan &span) const noexcept
    {
        span = {};
        if (live > unsigned(rows_)) return false;
        if (!index) { span = {0, (encoding() == GDNDenseCompactEncoding::ExactCorrectionsV1 ?
            live_header_words : split_live_header_words) * sizeof(std::uint32_t)}; return true; }
        GDNDenseRetentionMode mode{};
        if (!retentionForLiveRows(int(live), mode)) return false;
        const auto checkpoints = 1 + (mode == GDNDenseRetentionMode::FullSnapshots ?
            std::size_t(live) : std::size_t(live) / checkpoint_interval);
        if (index <= checkpoints)
            span = {(header_floats + (index - 1) * checkpoint_stride_) * sizeof(float), matrix_ * sizeof(float)};
        else if (mode == GDNDenseRetentionMode::ExactCorrections && index - checkpoints - 1 < live)
            span = {(records_offset_ + (index - checkpoints - 1) * record_stride_) * sizeof(float), record_ * sizeof(float)};
        else return false;
        return true;
    }
  private:
    /** @brief Authenticate split policy and actual representation; DFC1 ignores added words.
     * @param publication Already bounded completed protocol.
     * @return Exact captured policy and mode derived from that same published count. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool validRetentionIdentity(const GDNDenseCompactPublication &publication) const noexcept
    {
        if (encoding() == GDNDenseCompactEncoding::ExactCorrectionsV1) return true;
        GDNDenseRetentionMode mode{};
        return retentionForLiveRows(int(publication.published_rows), mode) &&
            publication.retention_mode == std::uint32_t(mode) &&
            publication.retention_program == std::uint32_t(program_.mode) &&
            publication.compact_min_live_rows == program_.compact_min_live_rows;
    }
    /** @brief Add immutable capacities before allocation.
     * @param a First extent.
     * @param b Second extent.
     * @return Checked sum; throws std::overflow_error on overflow. */
    static std::size_t sum(std::size_t a, std::size_t b)
    {
        if (b > std::numeric_limits<std::size_t>::max() - a) throw std::overflow_error("Dense compact sum overflow");
        return a + b;
    }
    /** @brief Multiply immutable capacities before allocation.
     * @param a First extent.
     * @param b Second extent.
     * @return Checked product; throws std::overflow_error on overflow. */
    static std::size_t product(std::size_t a, std::size_t b)
    {
        if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("Dense compact product overflow");
        return a * b;
    }
    /** @brief Round allocation extent to64-byte alignment.
     * @param a Unaligned FP32 element count.
     * @return Aligned count; throws std::overflow_error on overflow. */
    static std::size_t align(std::size_t a) { return sum(a, 15) & ~std::size_t(15); }
    int heads_, dk_, dv_, requests_, rows_;
    GDNDenseRetentionProgram program_;
    std::size_t matrix_, checkpoint_stride_, checkpoints_, record_, record_stride_, records_offset_;
    std::size_t head_stride_, dense_, compact_, gates_, outputs_;
};
/** @brief Meaning of one existing ordered device metadata value; no implicit conversion. */
enum class GDNDenseAcceptedSelectionKind : std::uint32_t
{
    AcceptedCounts = 0, ///< A0 restores immutable cp0.
    OwnRequestFlattenedRows = 1, ///< Negative means NoOp; positive rows stay in this request.
    RequestTerminalLengths = 2 ///< Length0 means NoOp, not cp0 restoration.
};
/** @brief Closed publication action, distinct from numeric accepted-prefix count. */
enum class GDNDenseAcceptedSelectionAction : std::uint32_t { NoOp = 0, Restore = 1 };
class GDNDenseAcceptedSelection;
/** @brief Pure decoded selection; only the descriptor can author a Restore result. */
class GDNDenseAcceptedSelectionResult final
{
  public:
    /** @return Whether the destination stays untouched or receives a retained prefix. */
    LLAMINAR_GDN_DENSE_COMPACT_HD GDNDenseAcceptedSelectionAction action() const noexcept { return action_; }
    /** @return Count for Restore (including explicit0); NoOp exposes0 but never requests restoration. */
    LLAMINAR_GDN_DENSE_COMPACT_HD unsigned acceptedCount() const noexcept { return accepted_; }
  private:
    friend class GDNDenseAcceptedSelection;
    GDNDenseAcceptedSelectionAction action_ = GDNDenseAcceptedSelectionAction::NoOp;
    unsigned accepted_ = 0;
};
/** @brief Borrowed typed view of the existing device selection owner, never a new buffer or ledger.
 * @details Cold construction validates pointer arithmetic only. The consumer loads
 * the indexed value on its exact ordered stream and passes it to resolveObserved;
 * this descriptor never downloads, copies or reads native metadata on the host.
 * Metadata and retained sources remain immutable under the existing publication
 * transaction lifetime. Whole-plan validation must precede destination writes.
 */
class GDNDenseAcceptedSelection final
{
  public:
    /** @brief Bind explicit accepted counts; zero means restore transaction cp0.
     * @param geometry Captured request count and physical row-width authority.
     * @param values Existing nonnull aligned device metadata owner.
     * @param available_elements Admitted metadata view capacity in int32 elements, not a transfer extent.
     * @param stride Positive element stride between request values.
     * @return Checked immutable view; no metadata value is read.
     * @throws std::invalid_argument For invalid pointer/stride; std::overflow_error for span overflow. */
    static GDNDenseAcceptedSelection acceptedCounts(const GDNDenseCompactGeometry &geometry,
        const std::int32_t *values, std::size_t available_elements, int stride = 1)
    {
        return {GDNDenseAcceptedSelectionKind::AcceptedCounts, geometry, values, available_elements, stride};
    }
    /** @brief Bind production request-owned flattened snapshot rows; negative means NoOp.
     * @param geometry Exact request count and physical request width used by the producer.
     * @param values Existing nonnull aligned device metadata owner.
     * @param available_elements Admitted metadata view capacity in int32 elements, not a transfer extent.
     * @param stride Positive element stride, including interleaved outcome fields.
     * @return Checked view; globally valid foreign-request rows are NOT admitted.
     * @throws std::invalid_argument For invalid pointer/stride; std::overflow_error for span overflow. */
    static GDNDenseAcceptedSelection ownRequestFlattenedRows(const GDNDenseCompactGeometry &geometry,
        const std::int32_t *values, std::size_t available_elements, int stride = 1)
    {
        return {GDNDenseAcceptedSelectionKind::OwnRequestFlattenedRows, geometry, values, available_elements, stride};
    }
    /** @brief Bind existing request terminal lengths; zero means NoOp.
     * @param geometry Exact request count and physical capacity.
     * @param values Existing nonnull aligned device length owner.
     * @param available_elements Admitted metadata view capacity in int32 elements, not a transfer extent.
     * @param stride Positive element stride between request lengths.
     * @return Checked view without reading values.
     * @throws std::invalid_argument For invalid pointer/stride; std::overflow_error for span overflow. */
    static GDNDenseAcceptedSelection requestTerminalLengths(const GDNDenseCompactGeometry &geometry,
        const std::int32_t *values, std::size_t available_elements, int stride = 1)
    {
        return {GDNDenseAcceptedSelectionKind::RequestTerminalLengths, geometry, values, available_elements, stride};
    }
    /** @return Closed meaning, part of captured binding identity. */
    LLAMINAR_GDN_DENSE_COMPACT_HD GDNDenseAcceptedSelectionKind kind() const noexcept { return kind_; }
    /** @return Exact borrowed device owner; only device code should load native values. */
    LLAMINAR_GDN_DENSE_COMPACT_HD const std::int32_t *data() const noexcept { return values_; }
    /** @return Immutable element stride, not a communication count. */
    LLAMINAR_GDN_DENSE_COMPACT_HD int stride() const noexcept { return stride_; }
    /** @return Exact addressed backing extent including interleaved holes, never transport volume. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t addressedBytes() const noexcept { return bytes_; }
    /** @return Exact admitted source view capacity, part of captured binding identity. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t capacityElements() const noexcept { return capacity_; }
    /** @brief Reject foreign captured request dimensions without reading data.
     * @param geometry Candidate transaction geometry.
     * @return Exact request count and physical row width match. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool matchesGeometry(const GDNDenseCompactGeometry &geometry) const noexcept
    {
        return requests_ == geometry.requests() && rows_ == geometry.rowsPerRequest();
    }
    /** @brief Compare every descriptor-owned embedded identity field.
     * @param other Candidate captured binding.
     * @return Exact source-kind/pointer/stride/request-geometry equality. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool sameBinding(const GDNDenseAcceptedSelection &other) const noexcept
    {
        return kind_ == other.kind_ && values_ == other.values_ && stride_ == other.stride_ &&
            requests_ == other.requests_ && rows_ == other.rows_ && bytes_ == other.bytes_ && capacity_ == other.capacity_;
    }
    /** @brief Derive one checked element offset before the device load.
     * @param request Request index in the captured geometry.
     * @param offset Reset on failure; otherwise exact request*stride element coordinate.
     * @return Whether the request is owned by this view. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool valueOffset(int request, std::size_t &offset) const noexcept
    {
        offset = 0;
        if (request < 0 || request >= requests_) return false;
        offset = std::size_t(request) * std::size_t(stride_);
        return true;
    }
    /** @brief Decode an already device-loaded value with no payload access or writes.
     * @param request Exact destination request.
     * @param published_live Authenticated published request count0..physical capacity.
     * @param observed Existing device value loaded from valueOffset(request).
     * @param result Reset to NoOp on invalid input; valid only when true is returned.
     * @return Valid NoOp or Restore; malformed positive foreign rows fail, never clamp.
     * @details Caller still authenticates all retained heads/counts before any
     * destination write. Explicit accepted0 restores cp0; legacy skip meanings
     * remain distinct and cannot silently restore an earlier transaction state. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool resolveObserved(int request, int published_live,
        std::int32_t observed, GDNDenseAcceptedSelectionResult &result) const noexcept
    {
        result = {};
        if (request < 0 || request >= requests_ || published_live < 0 || published_live > rows_) return false;
        unsigned accepted = 0;
        if (kind_ == GDNDenseAcceptedSelectionKind::OwnRequestFlattenedRows)
        {
            if (observed < 0) return true;
            const auto first = std::size_t(request) * std::size_t(rows_);
            const auto row = std::size_t(observed);
            if (row < first || row - first >= unsigned(published_live)) return false;
            accepted = unsigned(row - first) + 1;
        }
        else
        {
            if (observed < 0 || observed > published_live) return false;
            if (kind_ == GDNDenseAcceptedSelectionKind::RequestTerminalLengths && observed == 0) return true;
            accepted = unsigned(observed);
        }
        result.action_ = GDNDenseAcceptedSelectionAction::Restore;
        result.accepted_ = accepted;
        return true;
    }
  private:
    /** @brief Authenticate immutable address arithmetic without touching native storage.
     * @param kind Closed named-factory selection meaning.
     * @param geometry Existing checked request geometry.
     * @param values Nonnull aligned borrowed native pointer.
     * @param available_elements Admitted metadata view capacity in int32 elements, not a transfer extent.
     * @param stride Positive element stride.
     * @throws std::invalid_argument For missing/misaligned owner or nonpositive stride.
     * @throws std::overflow_error If the last addressed element or byte endpoint wraps. */
    GDNDenseAcceptedSelection(GDNDenseAcceptedSelectionKind kind, const GDNDenseCompactGeometry &geometry,
        const std::int32_t *values, std::size_t available_elements, int stride)
        : kind_(kind), values_(values), stride_(stride), requests_(geometry.requests()), rows_(geometry.rowsPerRequest()), capacity_(available_elements)
    {
        const auto pointer = reinterpret_cast<std::uintptr_t>(values);
        if (!values || pointer % alignof(std::int32_t) || stride <= 0)
            throw std::invalid_argument("Dense accepted selection requires an aligned owner and positive stride");
        const auto maximum = std::numeric_limits<std::size_t>::max();
        const auto last_request = std::size_t(requests_ - 1);
        if (last_request > (maximum - 1) / std::size_t(stride))
            throw std::overflow_error("Dense accepted selection element span overflow");
        const auto elements = last_request * std::size_t(stride) + 1;
        if (available_elements < elements)
            throw std::invalid_argument("Dense accepted selection metadata view is undersized");
        if (available_elements > maximum / sizeof(std::int32_t))
            throw std::overflow_error("Dense accepted selection admitted view overflow");
        if (elements > maximum / sizeof(std::int32_t))
            throw std::overflow_error("Dense accepted selection byte span overflow");
        bytes_ = elements * sizeof(std::int32_t);
        if (available_elements * sizeof(std::int32_t) > std::numeric_limits<std::uintptr_t>::max() - pointer)
            throw std::overflow_error("Dense accepted selection pointer span overflow");
    }
    GDNDenseAcceptedSelectionKind kind_;
    const std::int32_t *values_;
    int stride_, requests_, rows_;
    std::size_t bytes_, capacity_;
};

/** @brief Closed output contract independent of retained representation and acceptance lifetime. */
enum class GDNDenseCompactTerminalKind : std::uint32_t
{
    DenseDestination = 0, ///< Preserve the original required terminal bank and stores.
    RetainedPrefixesOnly = 1 ///< Retain every accepted prefix, without an unaccepted Dense terminal.
};
/** @brief Explicit cold output contract; a missing Dense pointer never selects another program.
 * @details Both contracts retain immutable cp0 and all completed prefix data,
 * including live0 and repeated/revised acceptance. This is not PublishOnce. */
class GDNDenseCompactTerminalOutput final
{
  public:
    /** @brief Declare a required Dense terminal owner, preserving existing behavior.
     * @param destination Nonnull complete Dense bank; exact initial alias is explicit/legal.
     * @return A typed required terminal destination.
     * @throws std::invalid_argument For a missing required owner. */
    static GDNDenseCompactTerminalOutput denseDestination(float *destination)
    {
        if (!destination) throw std::invalid_argument("Dense compact terminal destination is required");
        return GDNDenseCompactTerminalOutput(destination, GDNDenseCompactTerminalKind::DenseDestination);
    }
    /** @brief Select complete retained-prefix publication without an unaccepted terminal bank.
     * @return An explicit no-terminal program; immutable cp0 and all prefix/revision obligations remain.
     * @details Valid only with the complete checked retained transaction. No dummy
     * destination is bound, and no pointer absence is interpreted as a fallback. */
    static GDNDenseCompactTerminalOutput retainedPrefixesOnly() noexcept
    {
        return GDNDenseCompactTerminalOutput(nullptr, GDNDenseCompactTerminalKind::RetainedPrefixesOnly);
    }
    /** @return Closed captured output kind; unknown values fail cold admission. */
    LLAMINAR_GDN_DENSE_COMPACT_HD GDNDenseCompactTerminalKind kind() const noexcept { return kind_; }
    /** @return Required Dense destination, or null only under explicit RetainedPrefixesOnly. */
    LLAMINAR_GDN_DENSE_COMPACT_HD float *destination() const noexcept { return destination_; }
  private:
    /** @brief Only checked named factories construct the output contract.
     * @param destination Nonnull for DenseDestination, null for RetainedPrefixesOnly.
     * @param kind Exact corresponding cold output contract. */
    GDNDenseCompactTerminalOutput(float *destination, GDNDenseCompactTerminalKind kind) noexcept
        : destination_(destination), kind_(kind) {}
    float *destination_; ///< One caller-owned Dense destination or explicit absence.
    GDNDenseCompactTerminalKind kind_; ///< Captured policy, not inferred from the pointer.
};
/** @brief Immutable complete transaction identity; storage/lifetime remain caller-owned. */
struct GDNDenseCompactTransaction final
{
    GDNDenseCompactGeometry geometry;
    const float *initial_state; ///< Original complete Dense bank, not an epoch image.
    float *storage; ///< Separate64B-aligned retained arena; cp0 remains live even when empty.
    std::size_t storage_floats; ///< Exact admitted view extent, at least geometry.bankFloats().
    std::uint64_t generation; ///< Nonzero captured transaction generation, not a live host counter.
    GDNRecurrenceStateLayout layout; ///< Exact Dense checkpoint matrix axes.
    GDNDenseCompactArithmetic arithmetic; ///< Exact source arithmetic/native update identity.
    GDNDenseCompactEncoding encoding = GDNDenseCompactEncoding::ExactCorrectionsV1;
};
/** @brief Raw merged forward sources and explicit retained/terminal outputs. */
struct GDNDenseCompactCaptureInput final
{
    MergedRecurrenceInput sources; ///< Source-only: all legacy state/snapshot fields must be null/zero.
    GDNDenseCompactTransaction transaction;
    GDNDenseCompactTerminalOutput terminal; ///< Explicit Dense or complete retained-prefix-only output contract.
};
/** @brief Optional exact request-zero coherence edge, conditioned on the typed Restore action.
 * @details This borrowed view owns no allocation. Default None preserves existing
 * no-mirror callers. Only request zero writes the mirror, only after complete
 * immutable publication validation, and only for Restore (including Counts0).
 * Negative flattened rows and terminal-length0 leave both destinations untouched.
 */
class GDNDenseCompactRequestZeroMirror final
{
  public:
    /** @brief Construct the explicit no-mirror contract. */
    constexpr GDNDenseCompactRequestZeroMirror() noexcept = default;
    /** @return No second destination or communication edge. */
    static constexpr GDNDenseCompactRequestZeroMirror none() noexcept { return {}; }
    /** @brief Borrow one existing complete live-image destination; never allocation capacity traffic.
     * @param geometry Native request/head/matrix sizing authority.
     * @param destination Existing float-aligned request-zero mirror owner.
     * @param available_floats Actual borrowed view capacity, retained in capture identity.
     * @return Validated optional destination; only one exact image is written.
     * @throws std::invalid_argument For absent, short, overflowing or misaligned views. */
    static GDNDenseCompactRequestZeroMirror borrow(const GDNDenseCompactGeometry &geometry,
        float *destination, std::size_t available_floats)
    {
        GDNDenseCompactRequestZeroMirror result;
        result.destination_ = destination;
        result.available_floats_ = available_floats;
        if (!destination || !result.validFor(geometry))
            throw std::invalid_argument("GDN request-zero mirror requires a complete bounded live image");
        return result;
    }
    /** @return Exact optional borrowed destination, never a host state shadow. */
    LLAMINAR_GDN_DENSE_COMPACT_HD float *destination() const noexcept { return destination_; }
    /** @return Actual borrowed view capacity, not a transfer extent. */
    LLAMINAR_GDN_DENSE_COMPACT_HD std::size_t availableFloats() const noexcept { return available_floats_; }
    /** @return Whether request zero has a second ordered destination. */
    LLAMINAR_GDN_DENSE_COMPACT_HD bool enabled() const noexcept { return destination_ != nullptr; }
    /** @brief Validate immutable view arithmetic without reading any payload.
     * @param geometry Native geometry defining exactly one request's live image.
     * @return None is empty, or the complete float-aligned image fits the borrowed view. */
    bool validFor(const GDNDenseCompactGeometry &geometry) const noexcept
    {
        if (!destination_) return available_floats_ == 0;
        const auto required = geometry.denseBankFloats() / std::size_t(geometry.requests());
        planar_recurrence_detail::Range span{};
        return available_floats_ >= required &&
            available_floats_ <= std::numeric_limits<std::size_t>::max() / sizeof(float) &&
            reinterpret_cast<std::uintptr_t>(destination_) % alignof(float) == 0 &&
            planar_recurrence_detail::range(destination_, available_floats_ * sizeof(float), span);
    }
  private:
    float *destination_ = nullptr;
    std::size_t available_floats_ = 0;
};
/** @brief Typed device-selected restore or authenticated no-op; original Dense input is never read. */
struct GDNDenseCompactAcceptedPrefixInput final
{
    GDNDenseCompactTransaction transaction;
    GDNDenseAcceptedSelection selection; ///< Existing ordered metadata owner and exact typed meaning.
    float *destination_state; ///< Explicit complete Dense target; exact initial alias is legal.
    GDNDenseCompactRequestZeroMirror request_zero_mirror{}; ///< Optional Restore-only coherence edge.
};
/** @brief Check complete borrowed transaction extents without native payload reads.
 * @param t Immutable source, retained arena and capture identity.
 * @return Known format/arithmetic/layout, nonzero generation and disjoint bounded storage.
 * @details Original Dense banks retain float alignment; only compact arena
 * storage requires64-byte alignment. Original CUDA signed head/matrix indices
 * remain bounded. No allocation/lifetime decision is made. */
inline bool validGDNDenseCompactTransactionStorage(const GDNDenseCompactTransaction &t) noexcept
{
    using namespace planar_recurrence_detail;
    if (!t.generation || t.encoding != t.geometry.encoding() ||
        !validGDNDenseCompactArithmetic(t.arithmetic) ||
        !validGDNRecurrenceStateLayout(t.layout) || t.storage_floats < t.geometry.bankFloats() ||
        t.storage_floats > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        reinterpret_cast<std::uintptr_t>(t.storage) % t.geometry.alignment_bytes) return false;
    const auto maximum = std::size_t(std::numeric_limits<int>::max());
    const auto heads = std::size_t(t.geometry.heads());
    const auto dk = std::size_t(t.geometry.keyWidth());
    const auto dv = std::size_t(t.geometry.valueWidth());
    if (heads > maximum / dk || heads * dk > maximum / dv || heads > maximum / dv) return false;
    Range initial{}, compact{};
    return range(t.initial_state, t.geometry.denseBankBytes(), initial) &&
        range(t.storage, t.storage_floats * sizeof(float), compact) && !overlaps(initial, compact);
}
/** @brief Admit the exact-forward native capture storage before enqueue.
 * @param p Raw source-only tuple, complete transaction and explicit terminal contract.
 * @return Exact native shape and pairwise protected spans; no device values are read.
 * @details DenseDestination requires its complete protected bank. Explicit
 * RetainedPrefixesOnly admits only the complete retained transaction and has no
 * destination span; pointer absence never changes the program. Legacy dense
 * snapshots cannot coexist or fake retained storage. */
inline bool validGDNDenseCompactCaptureStorage(const GDNDenseCompactCaptureInput &p) noexcept
{
    const auto &s = p.sources;
    const auto &t = p.transaction;
    const auto &g = t.geometry;
    const auto kind = p.terminal.kind();
    const bool dense_terminal = kind == GDNDenseCompactTerminalKind::DenseDestination;
    if ((!dense_terminal && kind != GDNDenseCompactTerminalKind::RetainedPrefixesOnly) ||
        (dense_terminal ? !p.terminal.destination() : p.terminal.destination() != nullptr)) return false;
    if (!validGDNDenseCompactTransactionStorage(t) ||
        (g.keyWidth() != 64 && g.keyWidth() != 128) ||
        g.heads() != s.value_heads || g.keyWidth() != s.key_width || g.valueWidth() != s.value_width ||
        g.requests() != s.rows.requests() || g.rowsPerRequest() != s.rows.rowsPerRequest() ||
        t.layout != s.recurrence_layout || s.initial_state || s.updated_state || s.snapshots ||
        s.snapshot_stride_floats || s.max_snapshot_rows || !s.sourceOutputDisjoint()) return false;
    using namespace planar_recurrence_detail;
    // The original coordinate admission above already proves this exact merged
    // source product and byte extent representable. Other extents come only
    // from the checked geometry; none is a second memory-admission ledger.
    const auto source_width = 2 * std::size_t(s.key_heads) * s.key_width +
        std::size_t(s.value_heads) * s.value_width;
    const auto source_bytes = source_width * std::size_t(s.rows.physicalRows()) * sizeof(float);
    Range reads[7]{}, writes[3]{};
    const void *read_ptrs[]{s.merged, s.alpha, s.beta_raw, s.A_log, s.dt_bias, t.initial_state, s.rows.lengthOwner()};
    const std::size_t read_bytes[]{source_bytes, g.gateFloats() * sizeof(float),
        g.gateFloats() * sizeof(float), std::size_t(g.heads()) * sizeof(float),
        std::size_t(g.heads()) * sizeof(float), g.denseBankBytes(), std::size_t(g.requests()) * sizeof(std::int32_t)};
    for (int i = 0; i < 7; ++i)
        if (!(i == 6 && !read_ptrs[i]) && !range(read_ptrs[i], read_bytes[i], reads[i])) return false;
    if (!range(s.output, g.outputFloats() * sizeof(float), writes[0]) ||
        !range(t.storage, t.storage_floats * sizeof(float), writes[1])) return false;
    const int write_count = dense_terminal ? 3 : 2;
    if (dense_terminal && !range(p.terminal.destination(), g.denseBankBytes(), writes[2])) return false;
    for (int w = 0; w < write_count; ++w)
    {
        for (int previous = 0; previous < w; ++previous)
            if (overlaps(writes[w], writes[previous])) return false;
        for (int r = 0; r < 7; ++r)
            if (overlaps(writes[w], reads[r]) &&
                !(w == 2 && r == 5 && p.terminal.destination() == t.initial_state)) return false;
    }
    return true;
}
/** @brief Admit device-selected Dense reconstruction over immutable retained records.
 * @param p Complete transaction, native count owner and explicit Dense destination.
 * @return Exact known native identity and protected spans, without reading counts/protocol.
 * @details The kernel validates all protocol/counts before writes and waits on
 * the producer's existing event edge. It NEVER reads original Dense input,
 * even for live0; cp0 is always retained. Exact initial/destination alias is
 * legal for revised acceptance, while partial overlap is rejected. The count
 * authority is disjoint from both immutable retained and original Dense banks. */
inline bool validGDNDenseCompactAcceptedPrefixStorage(const GDNDenseCompactAcceptedPrefixInput &p) noexcept
{
    const auto &t = p.transaction;
    const auto &g = t.geometry;
    if (!validGDNDenseCompactTransactionStorage(t) || !p.selection.matchesGeometry(g) ||
        (g.keyWidth() != 64 && g.keyWidth() != 128)) return false;
    using namespace planar_recurrence_detail;
    Range compact{}, counts{}, initial{}, destination{};
    if (!range(t.storage, t.storage_floats * sizeof(float), compact) ||
        !range(p.selection.data(), p.selection.addressedBytes(), counts) ||
        !range(t.initial_state, g.denseBankBytes(), initial) ||
        !range(p.destination_state, g.denseBankBytes(), destination) ||
        overlaps(counts, compact) || overlaps(counts, initial) ||
        overlaps(destination, compact) || overlaps(destination, counts)) return false;
    if (overlaps(destination, initial) && p.destination_state != t.initial_state) return false;
    if (!p.request_zero_mirror.validFor(g)) return false;
    if (!p.request_zero_mirror.enabled()) return true;
    Range mirror{};
    const auto image_bytes = g.denseBankBytes() / std::size_t(g.requests());
    return range(p.request_zero_mirror.destination(), image_bytes, mirror) &&
        !overlaps(mirror, compact) && !overlaps(mirror, counts) &&
        !overlaps(mirror, initial) && !overlaps(mirror, destination);
}
/** @brief Admit CUDA capture with its exact eight-part arithmetic identity.
 * @param p Complete borrowed native transaction.
 * @return Complete common storage checks and CUDA-specific arithmetic admission. */
inline bool validCUDADenseCompactCaptureStorage(const GDNDenseCompactCaptureInput &p) noexcept
{
    return p.transaction.arithmetic == GDNDenseCompactArithmetic::CUDAEightPartFusedUpdateV1 &&
        validGDNDenseCompactCaptureStorage(p);
}
/** @brief Admit ROCm capture with the original four-part fused-update identity.
 * @param p Complete borrowed native transaction.
 * @return Complete common storage checks and ROCm-specific arithmetic admission. */
inline bool validROCmDenseCompactCaptureStorage(const GDNDenseCompactCaptureInput &p) noexcept
{
    return p.transaction.arithmetic == GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1 &&
        validGDNDenseCompactCaptureStorage(p);
}
/** @brief Admit CUDA accepted reconstruction without reading native counts.
 * @param p Immutable ordered transaction and explicit destination.
 * @return Complete common spans and exact CUDA arithmetic identity. */
inline bool validCUDADenseCompactAcceptedPrefixStorage(const GDNDenseCompactAcceptedPrefixInput &p) noexcept
{
    return p.transaction.arithmetic == GDNDenseCompactArithmetic::CUDAEightPartFusedUpdateV1 &&
        validGDNDenseCompactAcceptedPrefixStorage(p);
}
/** @brief Admit ROCm accepted reconstruction without reading native counts.
 * @param p Immutable ordered transaction and explicit destination.
 * @return Complete common spans and exact ROCm arithmetic identity. */
inline bool validROCmDenseCompactAcceptedPrefixStorage(const GDNDenseCompactAcceptedPrefixInput &p) noexcept
{
    return p.transaction.arithmetic == GDNDenseCompactArithmetic::ROCmFourPartFusedUpdateV1 &&
        validGDNDenseCompactAcceptedPrefixStorage(p);
}
/** @brief Host-only cold scheduling authority, defined without changing the numeric input tuple. */
class CUDADenseRetentionCapturePlan;
/** @brief Enqueue original dense forward/readout with explicit compact correction capture.
 * @param input Explicit Dense or retained-prefix-only output and complete borrowed transaction.
 * @param stream Exact nondefault native stream retained through terminal retirement.
 * @param plan Immutable exact-stream/device/geometry scheduling binding prepared before capture.
 * @return Cold admission and enqueue succeeded; no completion/economy claim. */
bool launchCUDADenseCompactRecurrence(const GDNDenseCompactCaptureInput &input, void *stream,
    const CUDADenseRetentionCapturePlan &plan);
/** @brief Enqueue exact accepted Dense reconstruction, never a model/M1 replay.
 * @param input Immutable completed compact transaction and native accepted-count authority.
 * @param stream Exact ordered nondefault native stream.
 * @return Cold admission and enqueue succeeded; no host count read or fallback. */
bool launchCUDADenseCompactAcceptedPrefix(const GDNDenseCompactAcceptedPrefixInput &input, void *stream);
/** @brief Enqueue original ROCm dense forward with explicit terminal contract and complete retention.
 * @param input Original raw sources, immutable transaction and explicit Dense/retained-only output.
 * @param stream Exact nondefault HIP stream through terminal retirement.
 * @return Cold admission/enqueue success; not a completion or economy receipt. */
bool launchROCmDenseCompactRecurrence(const GDNDenseCompactCaptureInput &input, void *stream);
/** @brief Reconstruct accepted ROCm Dense state from retained exact corrections only.
 * @param input Ordered compact transaction, native counts and exclusive destination.
 * @param stream Exact producer-ordered HIP stream through terminal retirement.
 * @return Cold admission/enqueue success; no original-state read or fallback. */
bool launchROCmDenseCompactAcceptedPrefix(const GDNDenseCompactAcceptedPrefixInput &input, void *stream);
static_assert(sizeof(float) == sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<GDNDenseCompactTerminalOutput>);
static_assert(std::is_standard_layout_v<GDNDenseCompactTerminalOutput>);
static_assert(std::is_trivially_copyable_v<GDNDenseCompactGeometry>);
static_assert(std::is_standard_layout_v<GDNDenseCompactGeometry>);
static_assert(std::is_trivially_copyable_v<GDNDenseCompactTransaction>);
static_assert(std::is_trivially_copyable_v<GDNDenseCompactCaptureInput>);
static_assert(std::is_trivially_copyable_v<GDNDenseAcceptedSelection>);
static_assert(std::is_standard_layout_v<GDNDenseAcceptedSelection>);
static_assert(std::is_trivially_copyable_v<GDNDenseCompactAcceptedPrefixInput>);
} // namespace llaminar2::gdn
#undef LLAMINAR_GDN_DENSE_COMPACT_HD
