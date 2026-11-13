/**
 * @file CPUNativeQuantisedPreparedFootprint.h
 * @brief Exact cache footprint contract for prepared CPU NativeQuantised matrices.
 *
 * NativeQuantised kernels consume a prepared representation, not the source GGUF
 * payload.  Cache policy therefore must be expressed in bytes of the prepared
 * interleaved stream and its live activation/output panels. IQ3_S source-scale
 * encoding retains decoded grid bytes and adds one raw coefficient vector.  Keeping those
 * quantities in a typed value prevents a compact source block size from being
 * mistaken for the substantially different execution layout.
 */

#pragma once

#include <cstdint>

#include "tensors/NativeQuantisedFormatInfo.h"

namespace llaminar2::cpu::native_quantised
{

    /**
     * @brief Physical encoding consumed by CPU NativeQuantised decode kernels.
     *
     * Source GGUF codebooks and prepared CPU encodings are deliberately
     * separate concepts. Several source formats share one prepared encoding,
     * while rotation converts every source format to expanded signed INT8.
     * Cache policy belongs to this execution encoding because it determines
     * the bytes and decode work seen by the hot kernel.
     */
    enum class CPUNativeQuantisedEncoding : std::uint8_t
    {
        NibbleLUT = 0,          ///< Four-bit payload decoded in the hot loop.
        ExpandedInt8 = 1,       ///< One prepared signed byte per logical weight.
        Q6KNativeDualScale = 2, ///< Six-bit payload, original half base, two int8 coefficients.
        CompactMultiScale = 3, ///< Native payload, two scales, optional two minima.
        NibbleLUTSourceScale = 4, ///< IQ4_XS nibble payload, original half base, raw coefficient.
        IQ3SExpandedSourceScale = 5, ///< Original IQ3_S signed grid, base and raw nibble.
    };

    /**
     * @brief Return whether an encoding value names a supported execution path.
     *
     * Keeping this validation beside the enum lets footprint construction fail
     * closed if an invalid serialized or test-injected value reaches policy
     * selection.
     */
    [[nodiscard]] constexpr bool isKnownPreparedEncoding(
        CPUNativeQuantisedEncoding encoding) noexcept
    {
        switch (encoding)
        {
        case CPUNativeQuantisedEncoding::NibbleLUT:
        case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
        case CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale:
        case CPUNativeQuantisedEncoding::ExpandedInt8:
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
        case CPUNativeQuantisedEncoding::CompactMultiScale:
            return true;
        }
        return false;
    }

    /**
     * @brief Return the pure weight-data bytes in one prepared CPU unit.
     *
     * A unit always spans 64 output columns and one 32-value K block. Keeping
     * this geometry beside the encoding enum gives CPU execution and
     * heterogeneous tier conversion one authoritative layout definition.
     */
    [[nodiscard]] constexpr std::uint32_t preparedDataStride(
        CPUNativeQuantisedEncoding encoding) noexcept
    {
        switch (encoding)
        {
        case CPUNativeQuantisedEncoding::NibbleLUT:
        case CPUNativeQuantisedEncoding::NibbleLUTSourceScale:
        case CPUNativeQuantisedEncoding::CompactMultiScale:
            return 1024;
        case CPUNativeQuantisedEncoding::ExpandedInt8:
        case CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale:
            return 2048;
        case CPUNativeQuantisedEncoding::Q6KNativeDualScale:
            return 1536;
        }
        return 0;
    }

    /**
     * @brief Return all bytes in one prepared CPU unit, including metadata.
     * @param encoding Physical execution encoding.
     * @param is_asymmetric Whether one FP16 minimum is stored per column.
     */
    [[nodiscard]] constexpr std::uint32_t preparedInterleavedBlockStride(
        CPUNativeQuantisedEncoding encoding,
        bool is_asymmetric) noexcept
    {
        const std::uint32_t data_stride = preparedDataStride(encoding);
        if (data_stride == 0)
            return 0;
        if (encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
            return 1408;
        if (encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale)
            return 2432;
        if (encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale)
            return 1792;
        // A fixed padded native payload plus two scale vectors and one optional
        // effective-minimum vector. Unused bytes are zero, not another ledger.
        if (encoding == CPUNativeQuantisedEncoding::CompactMultiScale)
            return 1536;
        return data_stride + 256u + (is_asymmetric ? 128u : 0u);
    }

    /**
     * @brief Physical unit-interleaving layout of a prepared CPU stream.
     *
     * The prepared encoding fixes the arithmetic families a kernel executes;
     * this layout fixes how those families' payload, compensation, scales,
     * minima and effective minima are packed into the resident unit. Layouts
     * are distinct physical byte formats for the same arithmetic: a view, seal
     * or movement link rejects a foreign layout instead of reinterpreting it.
     *
     * InterleavedCanonical is the production layout on every ISA today: a
     * fixed-size unit whose unused bytes are zero padding, sized for the
     * widest codebook of each encoding family. InterleavedCompactA64 is the
     * tight AArch64-only unit (design: docs/v2/projects/2026-11/arm-perf-review/
     * 08-arm-prepared-weight-layout-design.md) that trims bytes the NEON
     * decode programs never load; x86 keeps canonical bytes by construction.
     */
    enum class CPUNativeQuantisedLayout : std::uint8_t
    {
        InterleavedCanonical = 0,  ///< Fixed-width padded unit, one stride per encoding.
        InterleavedCompactA64 = 1, ///< Tight AArch64 unit; per-codebook payload width.
    };

    /**
     * @brief Return whether a layout value names a known physical layout.
     *
     * Knowing a value is not the same as the current build implementing it:
     * section tables decide implementability and fail closed otherwise.
     */
    [[nodiscard]] constexpr bool isKnownPreparedLayout(
        CPUNativeQuantisedLayout layout) noexcept
    {
        switch (layout)
        {
        case CPUNativeQuantisedLayout::InterleavedCanonical:
        case CPUNativeQuantisedLayout::InterleavedCompactA64:
            return true;
        }
        return false;
    }

/** @brief Stable diagnostic name of one prepared layout.
 * @param layout Known layout value.
 * @return Short lowercase name for typed refusal messages; never null.
 */
[[nodiscard]] constexpr const char *preparedLayoutName(
    CPUNativeQuantisedLayout layout) noexcept
{
    switch (layout)
    {
    case CPUNativeQuantisedLayout::InterleavedCanonical:
        return "interleaved-canonical";
    case CPUNativeQuantisedLayout::InterleavedCompactA64:
        return "interleaved-compact-a64";
    }
    return "unknown-prepared-layout";
}

    /**
     * @brief Layout this build's production packer, owner and kernels execute.
     *
     * Phase 0 spine: every ISA still prepares, admits and executes the
     * canonical interleaving, so every byte and stride is unchanged and this
     * constant is the sole selector that Phase 1 will move to the compact
     * unit on AArch64 builds only. x86 bytes can never change because this
     * constant never changes off AArch64.
     */
    inline constexpr CPUNativeQuantisedLayout kHostPreparedLayout =
        CPUNativeQuantisedLayout::InterleavedCanonical;

    /**
     * @brief Exact section placement inside one prepared interleaved unit.
     *
     * One authority for where every section lives: the packer writes through
     * these offsets, the view and owner read through them, and footprint
     * pricing derives its strides from them. Offsets are absolute byte
     * positions inside one unit; each one is meaningful only while its flag
     * is set, and isValid() must hold before any offset is consumed.
     */
    struct PreparedUnitSections
    {
        std::uint32_t payload_stride = 0; ///< Pure payload bytes before metadata.
        std::uint32_t block_stride = 0;   ///< Total bytes of one unit.
        std::uint32_t compensation_offset = 0; ///< Inline int16 compensation vector.
        std::uint32_t scales_offset = 0;  ///< Sixty-four primary FP16 words.
        std::uint32_t mins_offset = 0;    ///< Sixty-four secondary FP16 words.
        std::uint32_t emins_offset = 0;   ///< Sixty-four packed effective-minimum words.
        bool has_compensation = false;
        bool has_secondary = false;
        bool has_emins = false;

        /** @brief Return whether these sections describe a complete unit. */
        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return payload_stride > 0 && block_stride > 0 &&
                   scales_offset >= payload_stride &&
                   block_stride >= scales_offset + 128u &&
                   (!has_compensation ||
                       (compensation_offset >= payload_stride &&
                           scales_offset >= compensation_offset + 128u)) &&
                   (!has_secondary ||
                       (mins_offset >= scales_offset + 128u &&
                           block_stride >= mins_offset + 128u)) &&
                   (!has_emins ||
                       (emins_offset >= scales_offset + 128u &&
                           block_stride >= emins_offset + 256u));
        }
    };

    /**
     * @brief Return the exact canonical section placement of one unit.
     * @param layout Physical layout of the stream.
     * @param encoding Physical execution encoding of the stream.
     * @param is_asymmetric Whether the prepared stream stores FP16 minima.
     *   This is the prepared asymmetry, not the source's: rotation prepares
     *   asymmetric sources symmetrically.
     * @param source Original source format; the canonical table ignores it.
     *   The compact table reads its payload width and effective minima.
     *
     * Canonical anatomy: the compensation vector sits directly after the
     * payload for nibble and expanded encodings, Q6_K and multi-scale units
     * carry no compensation, scales follow (compensation ? 128 : 0) bytes
     * into the metadata, secondary words follow the scales, and multi-scale
     * units always reserve the packed effective-minimum word at payload+256
     * (zero for codebooks without live effective minima).
     *
     * The compact layout fails closed until its Phase 1 tables exist: it
     * returns an invalid section value no admission or pricing path can
     * mistake for canonical geometry.
     */
    [[nodiscard]] constexpr PreparedUnitSections preparedUnitSections(
        CPUNativeQuantisedLayout layout,
        CPUNativeQuantisedEncoding encoding,
        bool is_asymmetric,
        [[maybe_unused]] const NativeQuantisedFormatInfo &source) noexcept
    {
        PreparedUnitSections sections{};
        if (layout != CPUNativeQuantisedLayout::InterleavedCanonical)
            return sections;
        sections.payload_stride = preparedDataStride(encoding);
        sections.block_stride =
            preparedInterleavedBlockStride(encoding, is_asymmetric);
        if (sections.payload_stride == 0 || sections.block_stride == 0)
            return sections;
        sections.has_compensation = encoding != CPUNativeQuantisedEncoding::Q6KNativeDualScale &&
            encoding != CPUNativeQuantisedEncoding::CompactMultiScale;
        sections.has_secondary = is_asymmetric ||
            encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
            encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale ||
            encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale ||
            encoding == CPUNativeQuantisedEncoding::CompactMultiScale;
        sections.has_emins = encoding == CPUNativeQuantisedEncoding::CompactMultiScale;
        sections.compensation_offset = sections.has_compensation
            ? sections.payload_stride
            : 0u;
        sections.scales_offset = sections.payload_stride +
            (sections.has_compensation ? 128u : 0u);
        sections.mins_offset = sections.has_secondary
            ? sections.scales_offset + 128u
            : 0u;
        sections.emins_offset = sections.has_emins
            ? sections.payload_stride + 256u
            : 0u;
        return sections;
    }

    /**
     * @brief Layout-aware pure weight-data stride of one prepared unit.
     * @sa preparedDataStride(CPUNativeQuantisedEncoding)
     */
    [[nodiscard]] constexpr std::uint32_t preparedDataStride(
        CPUNativeQuantisedLayout layout,
        CPUNativeQuantisedEncoding encoding,
        bool is_asymmetric,
        const NativeQuantisedFormatInfo &source) noexcept
    {
        return preparedUnitSections(layout, encoding, is_asymmetric, source)
            .payload_stride;
    }

    /**
     * @brief Layout-aware total unit stride including metadata.
     * @sa preparedInterleavedBlockStride(CPUNativeQuantisedEncoding, bool)
     */
    [[nodiscard]] constexpr std::uint32_t preparedInterleavedBlockStride(
        CPUNativeQuantisedLayout layout,
        CPUNativeQuantisedEncoding encoding,
        bool is_asymmetric,
        const NativeQuantisedFormatInfo &source) noexcept
    {
        return preparedUnitSections(layout, encoding, is_asymmetric, source)
            .block_stride;
    }

    /**
     * @brief Whether a layout's section tables satisfy the source data contract.
     *
     * This is a DATA-requirement contract, never an encoding-only presence
     * table: it admits every layout whose tables carry the metadata the
     * source format actually needs and rejects only real data loss.
     *
     * - Secondary (min) words are genuine source metadata: every asymmetric
     *   source and the listed scale-carrying encodings must keep the plane.
     * - Embedded effective minima belong to the super-block family. A
     *   source that carries them (Q2_K) must retain the plane while it
     *   stays in that family; other CompactMultiScale sources may drop it
     *   or keep the canonical zero-filled plane, a re-encoded (rotated)
     *   preparation legitimately consumes the minima into its payload, and
     *   layouts outside the family must not invent the plane at all.
     * - Compensation is a derived x86-kernel aid, not source metadata. Its
     *   presence is owned by the layout tables themselves and checked only
     *   by section geometry; GPU conversion writes and reads the plane
     *   exactly when the endpoint layout has it, and x86 execution
     *   admission already requires the canonical layout. No encoding may
     *   demand or forbid it here.
     *
     * @param sections Layout table entry under validation.
     * @param encoding Prepared CPU encoding of the endpoint.
     * @param is_asymmetric Whether the source format carries min offsets.
     * @param source Original source format descriptor.
     * @return `true` when the tables keep every datum the source needs
     *     without inventing a family-forbidden plane.
     */
    [[nodiscard]] constexpr bool sectionPresenceSatisfiesSourceContract(
        const PreparedUnitSections &sections,
        CPUNativeQuantisedEncoding encoding,
        bool is_asymmetric,
        const NativeQuantisedFormatInfo &source) noexcept
    {
        const bool secondary_required =
            is_asymmetric || encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
            encoding == CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale ||
            encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale ||
            encoding == CPUNativeQuantisedEncoding::CompactMultiScale;
        if (sections.has_secondary != secondary_required)
            return false;
        // Q2_K and any other super-block source with embedded minima must
        // keep the plane while it stays in that family; a rotated or
        // otherwise re-encoded preparation consumes the minima into its
        // payload and legitimately drops the plane.
        if (source.has_emins &&
            encoding == CPUNativeQuantisedEncoding::CompactMultiScale &&
            !sections.has_emins)
            return false;
        // Only the super-block family may carry it; canonical zero-filled
        // planes for other CompactMultiScale sources stay admitted.
        return encoding == CPUNativeQuantisedEncoding::CompactMultiScale ||
               !sections.has_emins;
    }

    /**
     * @brief Exact resident bytes touched by one logical NativeQuantised tile unit.
     *
     * A weight unit is one 64-column N chunk by one 32-element K block and
     * includes every inline datum consumed by the kernel: packed values,
     * compensation, scales, and optional minima.  Activation and output units
     * describe one M row so a scheduler can account for the physical row tile
     * independently of the matrix's source codebook.
     */
    struct NativeQuantisedPreparedFootprint
    {
        CPUNativeQuantisedLayout layout =
            CPUNativeQuantisedLayout::InterleavedCanonical;
        CPUNativeQuantisedEncoding encoding =
            CPUNativeQuantisedEncoding::ExpandedInt8;
        bool is_asymmetric = false;
        std::uint64_t weight_bytes_per_n_chunk_k_block = 0;
        std::uint64_t activation_bytes_per_row_k_block = 0;
        std::uint64_t output_bytes_per_row_n_chunk = 0;

        /** @brief Return whether all execution-footprint dimensions are valid. */
        [[nodiscard]] constexpr bool isValid() const noexcept
        {
            return isKnownPreparedLayout(layout) &&
                   isKnownPreparedEncoding(encoding) &&
                   weight_bytes_per_n_chunk_k_block > 0 &&
                   activation_bytes_per_row_k_block > 0 &&
                   output_bytes_per_row_n_chunk > 0;
        }
    };

} // namespace llaminar2::cpu::native_quantised
