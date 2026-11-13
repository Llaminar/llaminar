#ifndef LLAMINAR2_TENSORS_NATIVEQUANTISEDFORMATINFO_H
#define LLAMINAR2_TENSORS_NATIVEQUANTISEDFORMATINFO_H

/**
 * @file NativeQuantisedFormatInfo.h
 * IQ3_S physical11 retains original half/nibble metadata with source11 arithmetic.
 * @brief Intrinsic format metadata for native-quantised quantization formats
 *
 * Describes encoding properties that are invariant for each quantization format:
 * payload size, symmetry, super-block layout, and magnitude bounds. These are
 * returned by IINT8Unpackable::vnniFormatInfo() on each tensor class, replacing
 * the external dispatch table.
 *
 * Source IDs retain GGUF arithmetic identity. Source-aware physical mapping
 * separately assigns IQ4_XS execution24 while IQ4_NL retains execution4.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace llaminar2
{
    /**
     * @brief Execution codebook for a normalized signed-INT8 block with a
     *        per-block asymmetric minimum.
     *
     * This is an accelerator execution format, not a GGUF source format. A
     * CPU tier expands several compact asymmetric source codebooks to signed
     * INT8. Promotion keeps those 32 payload bytes and the FP16 minimum as-is,
     * so CUDA and ROCm consume this identifier to select direct INT8 decode
     * plus the ordinary activation-sum correction.
     */
    inline constexpr std::uint8_t kNativeQuantisedExpandedInt8MinCodebook = 23;

    /** @brief IQ4_XS physical tuple: nibble payload, original half base, raw six-bit coefficient. */
    inline constexpr std::uint8_t kNativeQuantisedIQ4XSSourceScaleCodebook = 24;

    /** @brief Test the common IQ4 lookup-table payload family.
     * @param codebook Physical accelerator codebook.
     * @return Whether the unchanged IQ4 LUT decodes its nibble payload.
     */
    [[nodiscard]] inline constexpr bool isNativeQuantisedIQ4LUTCodebook(uint8_t codebook) noexcept
    {
        return codebook == 4 || codebook == kNativeQuantisedIQ4XSSourceScaleCodebook;
    }

    /** @brief Validate paired gate/up payload compatibility without conflating scales.
     * @param gate Physical gate codebook.
     * @param up Physical up codebook.
     * @return Equal codebooks or the two members of the IQ4 LUT family.
     */
    [[nodiscard]] inline constexpr bool areNativeQuantisedGateUpCodebooksCompatible(
        uint8_t gate, uint8_t up) noexcept
    {
        return gate == up || (isNativeQuantisedIQ4LUTCodebook(gate) && isNativeQuantisedIQ4LUTCodebook(up));
    }


    /**
     * @brief Identify compact payloads whose two native scales must survive CPU preparation.
     * @param codebook Canonical execution codebook, independent of source aliases.
     * @return Whether the common compact multi-scale CPU representation applies.
     *
     * Q6_K retains its bitplanes, original half base and signed coefficient pair. IQ2_S
     * retains its half base and original coefficient byte. The remaining
     * dual-scale formats retain their original payload, including IQ1_M signs;
     * Q2_K additionally retains the packed pair of effective minima.
     */
    [[nodiscard]] inline constexpr bool hasCompactMultiScaleVnniPayload(
        uint8_t codebook) noexcept
    {
        return codebook == 9 || codebook == 10 || codebook == 13 ||
               codebook == 14 || codebook == 17;
    }

    /**
     * @brief Metadata descriptor for a native-quantised quantization format
     *
     * Each native-quantised format (≤6-bit) has fixed metadata that determines:
     * - How the GEMV/GEMM GPU kernel decodes the payload (codebook_id)
     * - How many bytes each block's payload occupies (payload_bytes)
     * - Whether min-value correction is needed (is_asymmetric)
     * - Whether 256→8×32 sub-block decomposition is needed (is_superblock)
     * - Conservative element magnitude for INT8 re-quantization (max_abs_factor)
     */
    struct NativeQuantisedFormatInfo
    {
        uint8_t codebook_id;  ///< Kernel dispatch ID (0=Q4_0, 4=IQ4_NL, 5=Q4_1, ...)
        int payload_bytes;    ///< Bytes per 32-element block payload
        bool is_asymmetric;   ///< True if format has min-value offset
        bool is_superblock;   ///< True if format has 256-element super-blocks (8×32)
        bool has_emins;       ///< True if format needs separate embedded mins (Q2_K only)
        float max_abs_factor; ///< Conservative max |element| / scale
    };

    /**
     * @brief Unambiguous source-format provenance retained by a prepared engine.
     *
     * Accelerator execution canonicalizes several GGUF formats onto the same
     * codebook and byte layout.  Migration still needs the original codebook
     * plus superblock bit so a GPU-to-CPU stream can preserve format identity
     * and a later promotion can authenticate all 21 supported source formats.
     */
    struct NativeQuantisedSourceIdentity
    {
        uint8_t codebook_id = 0;
        bool is_superblock = false;
        bool present = false;

        /**
         * @brief Compare all provenance fields, including explicit presence.
         *
         * Written member-wise instead of `= default`: the x86-64 g++-14 of
         * the canonical build container rejects the defaulted form's
         * implicit instantiation in otherwise-identical translation units
         * that clang and the ARM g++-14 accept, while every member is a
         * scalar. The explicit comparison is unconditionally well-formed,
         * byte-identical in behavior, and keeps every existing
         * `==`/`!=` caller (C++20 equality rewriting supplies `!=`).
         */
        [[nodiscard]] bool operator==(
            const NativeQuantisedSourceIdentity &other) const noexcept
        {
            return codebook_id == other.codebook_id &&
                   is_superblock == other.is_superblock &&
                   present == other.present;
        }
    };

    /**
     * @brief One named source-format entry in the exhaustive NativeQuantised catalog.
     *
     * Source identity is intentionally separate from execution codebook
     * identity.  Several GGUF formats share the same 32-value accelerator
     * representation, while Q8_1 and Q8_K normalize to the Q8_0 execution
     * codebook during preparation.
     */
    struct NativeQuantisedSourceFormat
    {
        std::string_view quant_type;          ///< Canonical GGUF format name.
        const NativeQuantisedFormatInfo *metadata; ///< Immutable format descriptor.
    };

    /**
     * @brief Byte lengths of the four persistent native-quantised storage regions.
     *
     * GPU preparation stores each logical 32-value execution block as one
     * compressed payload plus a primary FP16 scale/base word. Asymmetric formats
     * and source-tuple formats add a secondary word (Q6_K signed coefficients,
     * IQ2_S source byte, IQ4_XS six-bit coefficient, otherwise FP16 scale/minimum)
     * and Q2_K adds one packed effective-minimum word. Keeping
     * this calculation beside the format catalog makes the loader, memory
     * planner, and tests consume exactly the same storage contract.
     */
    struct NativeQuantisedPackedRegionSizes
    {
        size_t payload_bytes = 0;
        size_t scales_bytes = 0;
        size_t mins_bytes = 0;
        size_t emins_bytes = 0;

        [[nodiscard]] constexpr size_t logicalBytes() const noexcept
        {
            return payload_bytes + scales_bytes + mins_bytes + emins_bytes;
        }
    };

    /**
     * @brief Canonical native-quantised format catalog.
     *
     * These constants are the sole host-side declaration of payload geometry.
     * Tensor classes advertise one of these entries, the device loader uses
     * the advertised entry to allocate its pool, and preflight planning looks
     * up the same entry by GGUF format name.
     */
    namespace native_quantised_formats
    {
        inline constexpr NativeQuantisedFormatInfo IQ4_NL{
            4, 16, false, false, false, 127.0f};
        inline constexpr NativeQuantisedFormatInfo Q8_0{
            19, 32, false, false, false, 127.0f};
        inline constexpr NativeQuantisedFormatInfo Q8_1{
            20, 32, false, false, false, 127.0f};
        inline constexpr NativeQuantisedFormatInfo Q4_0{
            0, 16, false, false, false, 8.0f};
        inline constexpr NativeQuantisedFormatInfo Q4_1{
            5, 16, true, false, false, 15.0f};
        inline constexpr NativeQuantisedFormatInfo Q5_0{
            6, 20, false, false, false, 16.0f};
        inline constexpr NativeQuantisedFormatInfo Q5_1{
            7, 20, true, false, false, 31.0f};
        inline constexpr NativeQuantisedFormatInfo Q6_K{
            8, 24, true, true, false, 32.0f};
        inline constexpr NativeQuantisedFormatInfo Q2_K{
            10, 8, true, true, true, 3.0f};
        inline constexpr NativeQuantisedFormatInfo Q5_K{
            7, 20, true, true, false, 31.0f};
        inline constexpr NativeQuantisedFormatInfo Q3_K{
            9, 12, true, true, false, 4.0f};
        inline constexpr NativeQuantisedFormatInfo Q4_K{
            5, 16, true, true, false, 15.0f};
        inline constexpr NativeQuantisedFormatInfo Q8_K{
            21, 32, false, true, false, 127.0f};
        inline constexpr NativeQuantisedFormatInfo IQ4_XS{
            4, 16, false, true, false, 127.0f};
        inline constexpr NativeQuantisedFormatInfo IQ2_XXS{
            15, 8, false, true, false, 43.0f};
        inline constexpr NativeQuantisedFormatInfo IQ2_XS{
            14, 9, true, true, false, 43.0f};
        inline constexpr NativeQuantisedFormatInfo IQ3_XXS{
            12, 12, false, true, false, 62.0f};
        inline constexpr NativeQuantisedFormatInfo IQ2_S{
            13, 9, true, true, false, 43.0f};
        inline constexpr NativeQuantisedFormatInfo IQ3_S{
            11, 13, false, true, false, 15.0f};
        inline constexpr NativeQuantisedFormatInfo IQ1_S{
            16, 6, true, true, false, 1.125f};
        inline constexpr NativeQuantisedFormatInfo IQ1_M{
            17, 6, true, true, false, 1.125f};

        /**
         * @brief Exhaustive set of quantized source formats accepted by the
         * NativeQuantised preparation pipeline.
         *
         * Tests and migration campaigns iterate this catalog instead of
         * maintaining private format lists.  Adding a tensor format without
         * adding it here therefore fails totality gates rather than silently
         * omitting it from cross-tier correctness coverage.
         */
        inline constexpr std::array<NativeQuantisedSourceFormat, 21>
            kAllSourceFormats{{
                {"IQ4_NL", &IQ4_NL},
                {"Q8_0", &Q8_0},
                {"Q8_1", &Q8_1},
                {"Q4_0", &Q4_0},
                {"Q4_1", &Q4_1},
                {"Q5_0", &Q5_0},
                {"Q5_1", &Q5_1},
                {"Q6_K", &Q6_K},
                {"Q2_K", &Q2_K},
                {"Q5_K", &Q5_K},
                {"Q3_K", &Q3_K},
                {"Q4_K", &Q4_K},
                {"Q8_K", &Q8_K},
                {"IQ4_XS", &IQ4_XS},
                {"IQ2_XXS", &IQ2_XXS},
                {"IQ2_XS", &IQ2_XS},
                {"IQ3_XXS", &IQ3_XXS},
                {"IQ2_S", &IQ2_S},
                {"IQ3_S", &IQ3_S},
                {"IQ1_S", &IQ1_S},
                {"IQ1_M", &IQ1_M},
            }};

        /**
         * @brief Resolve the unambiguous source identity carried by a prepared
         * accelerator projection.
         * @param source_codebook_id Codebook advertised by the source tensor.
         * @param is_superblock Whether its source storage uses 256-value blocks.
         * @return Matching catalog descriptor, or `nullptr` for an unsupported
         *         or internally ambiguous pair.
         */
        [[nodiscard]] inline constexpr const NativeQuantisedFormatInfo *
        forSourceIdentity(
            uint8_t source_codebook_id,
            bool is_superblock) noexcept
        {
            const NativeQuantisedFormatInfo *match = nullptr;
            for (const NativeQuantisedSourceFormat &entry : kAllSourceFormats)
            {
                if (entry.metadata->codebook_id != source_codebook_id ||
                    entry.metadata->is_superblock != is_superblock)
                {
                    continue;
                }
                // More than one match would make the wire identity incomplete.
                if (match != nullptr)
                    return nullptr;
                match = entry.metadata;
            }
            return match;
        }

        /**
         * @brief Resolve a GGUF quantization name to its canonical descriptor.
         *
         * Family suffixes such as Q4_K_S and Q5_K_M describe model-level
         * quantization choices while retaining the same execution block
         * geometry, so they intentionally map to the common family entry.
         */
        [[nodiscard]] inline constexpr const NativeQuantisedFormatInfo *
        forQuantType(std::string_view quant_type) noexcept
        {
            // Exact source names are resolved from the same iterable catalog
            // used by all-format tests and migration campaign generation.
            for (const NativeQuantisedSourceFormat &entry : kAllSourceFormats)
            {
                if (entry.quant_type == quant_type)
                    return entry.metadata;
            }

            // Model-level K-quant suffixes retain the same execution layout.
            if (quant_type == "Q5_K" ||
                quant_type == "Q5_K_S" ||
                quant_type == "Q5_K_M")
                return &Q5_K;
            if (quant_type == "Q3_K" ||
                quant_type == "Q3_K_S" ||
                quant_type == "Q3_K_M" ||
                quant_type == "Q3_K_L")
                return &Q3_K;
            if (quant_type == "Q4_K" ||
                quant_type == "Q4_K_S" ||
                quant_type == "Q4_K_M")
                return &Q4_K;
            return nullptr;
        }
    } // namespace native_quantised_formats

    /**
     * @brief Normalize source arithmetic aliases without selecting a physical tuple.
     * @param source_codebook_id Codebook advertised by the source tensor.
     * @return Canonical CUDA/ROCm execution codebook.
     *
     * Q8_0, Q8_1, and Q8_K use different source blocks, but their prepared
     * payload is always 32 signed bytes plus one FP16 scale.  The source-format
     * catalog remains responsible for distinguishing those three inputs.
     */
    [[nodiscard]] inline constexpr uint8_t canonicalDeviceVnniCodebookId(
        uint8_t source_codebook_id) noexcept
    {
        return source_codebook_id == 20 || source_codebook_id == 21
                   ? static_cast<uint8_t>(19)
                   : source_codebook_id;
    }

    /**
     * @brief GPU execution representation that survives a CPU-tier round trip.
     *
     * CPU NativeQuantised preserves nibble and multi-scale native payloads, including
     * Q6_K bitplanes, Q2_K effective minima, and IQ1_M signed deltas. Single-scale
     * sources expand to one signed byte per value without changing quantization;
     * asymmetric sources retain a separate FP16 minimum. Source provenance is
     * deliberately not part of this structure because the prepared engine
     * carries it independently for arithmetic-policy selection.
     */
    struct NativeQuantisedMigrationStableDeviceFormat
    {
        std::uint8_t codebook_id = 0;
        std::uint8_t payload_bytes_per_block = 0;
        bool is_asymmetric = false; ///< Physical secondary-plane presence, including raw source coefficients.
        bool has_emins = false;
    };

    /** @brief Derive canonical physical bytes from unambiguous original provenance.
     * @param source Original GGUF descriptor, including the superblock bit.
     * @return Device decoder and plane-presence metadata; is_asymmetric here names
     *         the historical secondary-plane storage flag, not additive-minimum math.
     */
    [[nodiscard]] inline constexpr NativeQuantisedMigrationStableDeviceFormat
    canonicalDeviceVnniFormat(const NativeQuantisedFormatInfo &source) noexcept
    {
        const bool iq4xs = source.codebook_id == 4 && source.is_superblock;
        return {
            .codebook_id = iq4xs ? kNativeQuantisedIQ4XSSourceScaleCodebook
                                : canonicalDeviceVnniCodebookId(source.codebook_id),
            .payload_bytes_per_block = static_cast<uint8_t>(source.payload_bytes),
            .is_asymmetric = source.is_asymmetric || iq4xs || source.codebook_id == 11,
            .has_emins = source.has_emins,
        };
    }

    /** @brief Resolve a physical codebook while retaining the source-only overload.
     * @param source Original GGUF descriptor.
     * @return Canonical physical execution ID, distinct from source arithmetic ID4.
     */
    [[nodiscard]] inline constexpr uint8_t canonicalDeviceVnniCodebookId(
        const NativeQuantisedFormatInfo &source) noexcept
    {
        return canonicalDeviceVnniFormat(source).codebook_id;
    }

    /** @brief Resolve canonical physical ID from an authenticated source tuple.
     * @param source Original source identity with explicit presence.
     * @return Physical ID, or255 for absent/uncataloged provenance.
     */
    [[nodiscard]] inline constexpr uint8_t canonicalDeviceVnniCodebookId(
        NativeQuantisedSourceIdentity source) noexcept
    {
        const auto *format = source.present ? native_quantised_formats::forSourceIdentity(
            source.codebook_id, source.is_superblock) : nullptr;
        return format ? canonicalDeviceVnniCodebookId(*format) : uint8_t{255};
    }

    /**
     * @brief Physical capacity required by a recyclable GPU expert slot.
     *
     * A loader-owned slot initially contains the source's canonical compact
     * representation, but a later CPU-to-GPU arrival may contain the normalized
     * migration representation. The slot therefore owns the union of both
     * layouts while its current descriptor names only the live encoding.
     */
    struct NativeQuantisedReusableDeviceAllocationFormat
    {
        std::uint8_t payload_bytes_per_block = 0;
        bool has_mins = false;
        bool has_emins = false;
    };

    /**
     * @brief Derive the one accelerator format produced by CPU-tier promotion.
     * @param source Exact GGUF source-format metadata.
     * @return CUDA/ROCm execution metadata stable across repeated tier cycles.
     */
    [[nodiscard]] inline constexpr NativeQuantisedMigrationStableDeviceFormat
    migrationStableDeviceVnniFormat(
        const NativeQuantisedFormatInfo &source) noexcept
    {
        if (source.codebook_id == 0 || source.codebook_id == 4 ||
            source.codebook_id == 5 || source.codebook_id == 11)
        {
            return canonicalDeviceVnniFormat(source);
        }
        if (source.codebook_id == 8 ||
            hasCompactMultiScaleVnniPayload(source.codebook_id))
        {
            return {
                .codebook_id = source.codebook_id,
                .payload_bytes_per_block = static_cast<uint8_t>(source.payload_bytes),
                .is_asymmetric = true,
                .has_emins = source.has_emins,
            };
        }
        return {
            .codebook_id = source.is_asymmetric
                               ? kNativeQuantisedExpandedInt8MinCodebook
                               : static_cast<std::uint8_t>(19),
            .payload_bytes_per_block = 32,
            .is_asymmetric = source.is_asymmetric,
            .has_emins = false,
        };
    }

    /**
     * @brief Derive the allocation union for initial and migrated GPU bytes.
     * @param source Exact GGUF source-format metadata.
     * @return Per-block capacities that make a loader slot recyclable.
     */
    [[nodiscard]] inline constexpr NativeQuantisedReusableDeviceAllocationFormat
    reusableDeviceVnniAllocationFormat(
        const NativeQuantisedFormatInfo &source) noexcept
    {
        const auto migrated = migrationStableDeviceVnniFormat(source);
        return {
            .payload_bytes_per_block = static_cast<std::uint8_t>(
                source.payload_bytes > migrated.payload_bytes_per_block
                    ? source.payload_bytes
                    : migrated.payload_bytes_per_block),
            .has_mins = source.is_asymmetric || migrated.is_asymmetric,
            .has_emins = source.has_emins || migrated.has_emins,
        };
    }

    /**
     * @brief Test whether an execution representation preserves one source's
     *        value semantics.
     * @param source Original GGUF NativeQuantised format descriptor.
     * @param execution_codebook Codebook carried by prepared device bytes.
     * @return Whether the source can be represented by that execution format.
     *
     * The canonical compact representation is always compatible. A fully
     * expanded representation is also compatible when its correction kind
     * matches the source: codebook 19 stores signed INT8 plus scale, while
     * codebook 23 additionally stores one FP16 minimum. Source provenance is
     * retained separately and is never inferred from these normalized IDs.
     */
    [[nodiscard]] inline constexpr bool deviceVnniExecutionCompatibleWithSource(
        const NativeQuantisedFormatInfo &source,
        uint8_t execution_codebook) noexcept
    {
        // Retire only the lossy native alias. Explicit existing ExpandedInt8
        // preparation remains representable by physical19 below; no loader or
        // migration factory automatically selects it for native IQ4_XS.
        if (source.codebook_id == 4 && source.is_superblock && execution_codebook == 4)
            return false;
        if (execution_codebook ==
            canonicalDeviceVnniCodebookId(source))
        {
            return true;
        }
        if (execution_codebook == 19)
            return !source.is_asymmetric;
        return execution_codebook == kNativeQuantisedExpandedInt8MinCodebook &&
               source.is_asymmetric &&
               !hasCompactMultiScaleVnniPayload(source.codebook_id) &&
               source.codebook_id != 8;
    }

    /**
     * @brief Capture-stable NativeQuantised formats reachable by one descriptor table.
     *
     * Immutable grouped tables contain only the execution codebooks present at
     * graph setup. A runtime-placement table may later receive the canonical
     * compact GPU representation or the migration-stable representation of the
     * same source tensor. Both launch masks must therefore be certified before
     * capture; discovering a new codebook after publication would otherwise
     * leave a valid runtime descriptor with no owning grouped kernel.
     */
    struct NativeQuantisedExecutionFormatEnvelope
    {
        uint32_t execution_codebook_mask = 0u; ///< Physical decoders reachable at replay.
        uint32_t policy_codebook_mask = 0u; ///< Source arithmetic policies retained at replay.

        /** @return Whether at least one complete source contract was added. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return execution_codebook_mask != 0u &&
                   policy_codebook_mask != 0u;
        }
    };

    /** @return One mask bit for an ABI codebook, or zero outside the ABI. */
    [[nodiscard]] inline constexpr uint32_t nativeQuantisedCodebookMaskBit(
        uint8_t codebook_id) noexcept
    {
        return codebook_id < 32u
                   ? (uint32_t{1} << codebook_id)
                   : 0u;
    }

    /**
     * @brief Add one prepared projection to its capture-time format envelope.
     *
     * @param envelope Destination union for a gate, up, or down table.
     * @param execution_codebook Codebook used by the currently prepared bytes.
     * @param source Exact GGUF provenance carried by the prepared engine.
     * @param runtime_mutable Whether future placement publication may replace
     *        the descriptor without recapturing the graph.
     * @return True when the current representation and every reachable runtime
     *         representation have a complete, supported identity.
     *
     * Runtime-mutability requires source provenance. The canonical compact GPU
     * codebook and the CPU-round-trip migration codebook are both included even
     * when the current bytes use only one of them. This makes promotion,
     * demotion, and GPU-to-GPU movement format-total without widening immutable
     * static tables or consulting mutable host state during inference.
     */
    [[nodiscard]] inline constexpr bool addNativeQuantisedExecutionFormat(
        NativeQuantisedExecutionFormatEnvelope &envelope,
        uint8_t execution_codebook,
        NativeQuantisedSourceIdentity source,
        bool runtime_mutable) noexcept
    {
        const uint32_t execution_bit =
            nativeQuantisedCodebookMaskBit(execution_codebook);
        if (execution_bit == 0u)
            return false;

        if (!source.present)
        {
            if (runtime_mutable || execution_codebook == kNativeQuantisedIQ4XSSourceScaleCodebook)
                return false;
            envelope.execution_codebook_mask |= execution_bit;
            envelope.policy_codebook_mask |= execution_bit;
            return true;
        }

        const NativeQuantisedFormatInfo *const source_format =
            native_quantised_formats::forSourceIdentity(
                source.codebook_id,
                source.is_superblock);
        if (!source_format ||
            !deviceVnniExecutionCompatibleWithSource(
                *source_format,
                execution_codebook))
        {
            return false;
        }

        const uint32_t canonical_bit = nativeQuantisedCodebookMaskBit(
            canonicalDeviceVnniCodebookId(*source_format));
        const uint32_t policy_bit = nativeQuantisedCodebookMaskBit(
            canonicalDeviceVnniCodebookId(source.codebook_id));
        if (canonical_bit == 0u)
            return false;

        uint32_t reachable_execution = execution_bit;
        if (runtime_mutable)
        {
            const auto migration =
                migrationStableDeviceVnniFormat(*source_format);
            const uint32_t migration_bit =
                nativeQuantisedCodebookMaskBit(migration.codebook_id);
            if (migration_bit == 0u)
                return false;
            reachable_execution |= canonical_bit | migration_bit;
        }

        envelope.execution_codebook_mask |= reachable_execution;
        envelope.policy_codebook_mask |= policy_bit;
        return true;
    }

    /**
     * @brief Compute the exact logical pool regions for one packed matrix.
     *
     * NativeQuantised kernels require K to consist of complete 32-value execution
     * blocks. The loader enforces that shape contract before calling the pool;
     * this pure helper mirrors the resulting persistent allocation sizes.
     */
    [[nodiscard]] inline constexpr NativeQuantisedPackedRegionSizes
    nativeQuantisedPackedRegionSizes(
        size_t rows,
        size_t columns,
        const NativeQuantisedFormatInfo &format) noexcept
    {
        const auto physical = canonicalDeviceVnniFormat(format);
        const size_t block_count = rows * (columns / 32);
        return {
            .payload_bytes =
                block_count * static_cast<size_t>(format.payload_bytes),
            .scales_bytes = block_count * sizeof(uint16_t),
            .mins_bytes =
                physical.is_asymmetric ? block_count * sizeof(uint16_t) : 0,
            .emins_bytes =
                format.has_emins ? block_count * sizeof(uint32_t) : 0,
        };
    }

} // namespace llaminar2

#endif // LLAMINAR2_TENSORS_NATIVEQUANTISEDFORMATINFO_H
