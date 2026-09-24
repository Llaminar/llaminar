#ifndef LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H
#define LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H

/**
 * @file NativeVnniFormatInfo.h
 * @brief Intrinsic format metadata for native-VNNI quantization formats
 *
 * Describes encoding properties that are invariant for each quantization format:
 * payload size, symmetry, super-block layout, and magnitude bounds. These are
 * returned by IINT8Unpackable::vnniFormatInfo() on each tensor class, replacing
 * the external dispatch table.
 *
 * Codebook IDs match the NativeVNNIFormat enum in ROCmGemvKernel_native_VNNI.hip.
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
    inline constexpr std::uint8_t kNativeVnniExpandedInt8MinCodebook = 23;

    /**
     * @brief Identify compact payloads whose two native scales must survive CPU preparation.
     * @param codebook Canonical execution codebook, independent of source aliases.
     * @return Whether the common compact multi-scale CPU representation applies.
     *
     * Q6_K has its own already-lossless bitplane representation. The remaining
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
     * @brief Metadata descriptor for a native-VNNI quantization format
     *
     * Each native-VNNI format (≤6-bit) has fixed metadata that determines:
     * - How the GEMV/GEMM GPU kernel decodes the payload (codebook_id)
     * - How many bytes each block's payload occupies (payload_bytes)
     * - Whether min-value correction is needed (is_asymmetric)
     * - Whether 256→8×32 sub-block decomposition is needed (is_superblock)
     * - Conservative element magnitude for INT8 re-quantization (max_abs_factor)
     */
    struct NativeVnniFormatInfo
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
    struct NativeVnniSourceIdentity
    {
        uint8_t codebook_id = 0;
        bool is_superblock = false;
        bool present = false;

        /** @brief Compare all provenance fields, including explicit presence. */
        bool operator==(const NativeVnniSourceIdentity &) const = default;
    };

    /**
     * @brief One named source-format entry in the exhaustive NativeVNNI catalog.
     *
     * Source identity is intentionally separate from execution codebook
     * identity.  Several GGUF formats share the same 32-value accelerator
     * representation, while Q8_1 and Q8_K normalize to the Q8_0 execution
     * codebook during preparation.
     */
    struct NativeVnniSourceFormat
    {
        std::string_view quant_type;          ///< Canonical GGUF format name.
        const NativeVnniFormatInfo *metadata; ///< Immutable format descriptor.
    };

    /**
     * @brief Byte lengths of the four persistent native-VNNI storage regions.
     *
     * GPU preparation stores each logical 32-value execution block as one
     * compressed payload plus FP16 scale metadata. Asymmetric formats add an
     * FP16 minimum and Q2_K adds one packed effective-minimum word. Keeping
     * this calculation beside the format catalog makes the loader, memory
     * planner, and tests consume exactly the same storage contract.
     */
    struct NativeVnniPackedRegionSizes
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
     * @brief Canonical native-VNNI format catalog.
     *
     * These constants are the sole host-side declaration of payload geometry.
     * Tensor classes advertise one of these entries, the device loader uses
     * the advertised entry to allocate its pool, and preflight planning looks
     * up the same entry by GGUF format name.
     */
    namespace native_vnni_formats
    {
        inline constexpr NativeVnniFormatInfo IQ4_NL{
            4, 16, false, false, false, 127.0f};
        inline constexpr NativeVnniFormatInfo Q8_0{
            19, 32, false, false, false, 127.0f};
        inline constexpr NativeVnniFormatInfo Q8_1{
            20, 32, false, false, false, 127.0f};
        inline constexpr NativeVnniFormatInfo Q4_0{
            0, 16, false, false, false, 8.0f};
        inline constexpr NativeVnniFormatInfo Q4_1{
            5, 16, true, false, false, 15.0f};
        inline constexpr NativeVnniFormatInfo Q5_0{
            6, 20, false, false, false, 16.0f};
        inline constexpr NativeVnniFormatInfo Q5_1{
            7, 20, true, false, false, 31.0f};
        inline constexpr NativeVnniFormatInfo Q6_K{
            8, 24, true, true, false, 32.0f};
        inline constexpr NativeVnniFormatInfo Q2_K{
            10, 8, true, true, true, 3.0f};
        inline constexpr NativeVnniFormatInfo Q5_K{
            7, 20, true, true, false, 31.0f};
        inline constexpr NativeVnniFormatInfo Q3_K{
            9, 12, true, true, false, 4.0f};
        inline constexpr NativeVnniFormatInfo Q4_K{
            5, 16, true, true, false, 15.0f};
        inline constexpr NativeVnniFormatInfo Q8_K{
            21, 32, false, true, false, 127.0f};
        inline constexpr NativeVnniFormatInfo IQ4_XS{
            4, 16, false, true, false, 127.0f};
        inline constexpr NativeVnniFormatInfo IQ2_XXS{
            15, 8, false, true, false, 43.0f};
        inline constexpr NativeVnniFormatInfo IQ2_XS{
            14, 9, true, true, false, 43.0f};
        inline constexpr NativeVnniFormatInfo IQ3_XXS{
            12, 12, false, true, false, 62.0f};
        inline constexpr NativeVnniFormatInfo IQ2_S{
            13, 9, true, true, false, 43.0f};
        inline constexpr NativeVnniFormatInfo IQ3_S{
            11, 13, false, true, false, 15.0f};
        inline constexpr NativeVnniFormatInfo IQ1_S{
            16, 6, true, true, false, 1.125f};
        inline constexpr NativeVnniFormatInfo IQ1_M{
            17, 6, true, true, false, 1.125f};

        /**
         * @brief Exhaustive set of quantized source formats accepted by the
         * NativeVNNI preparation pipeline.
         *
         * Tests and migration campaigns iterate this catalog instead of
         * maintaining private format lists.  Adding a tensor format without
         * adding it here therefore fails totality gates rather than silently
         * omitting it from cross-tier correctness coverage.
         */
        inline constexpr std::array<NativeVnniSourceFormat, 21>
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
        [[nodiscard]] inline constexpr const NativeVnniFormatInfo *
        forSourceIdentity(
            uint8_t source_codebook_id,
            bool is_superblock) noexcept
        {
            const NativeVnniFormatInfo *match = nullptr;
            for (const NativeVnniSourceFormat &entry : kAllSourceFormats)
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
        [[nodiscard]] inline constexpr const NativeVnniFormatInfo *
        forQuantType(std::string_view quant_type) noexcept
        {
            // Exact source names are resolved from the same iterable catalog
            // used by all-format tests and migration campaign generation.
            for (const NativeVnniSourceFormat &entry : kAllSourceFormats)
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
    } // namespace native_vnni_formats

    /**
     * @brief Return the codebook consumed by accelerator kernels after repack.
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
     * CPU NativeVNNI preserves nibble and multi-scale native payloads, including
     * Q6_K bitplanes, Q2_K effective minima, and IQ1_M signed deltas. Single-scale
     * sources expand to one signed byte per value without changing quantization;
     * asymmetric sources retain a separate FP16 minimum. Source provenance is
     * deliberately not part of this structure because the prepared engine
     * carries it independently for arithmetic-policy selection.
     */
    struct NativeVnniMigrationStableDeviceFormat
    {
        std::uint8_t codebook_id = 0;
        std::uint8_t payload_bytes_per_block = 0;
        bool is_asymmetric = false;
        bool has_emins = false;
    };

    /**
     * @brief Physical capacity required by a recyclable GPU expert slot.
     *
     * A loader-owned slot initially contains the source's canonical compact
     * representation, but a later CPU-to-GPU arrival may contain the normalized
     * migration representation. The slot therefore owns the union of both
     * layouts while its current descriptor names only the live encoding.
     */
    struct NativeVnniReusableDeviceAllocationFormat
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
    [[nodiscard]] inline constexpr NativeVnniMigrationStableDeviceFormat
    migrationStableDeviceVnniFormat(
        const NativeVnniFormatInfo &source) noexcept
    {
        if (source.codebook_id == 0 || source.codebook_id == 4 ||
            source.codebook_id == 5)
        {
            return {
                .codebook_id =
                    canonicalDeviceVnniCodebookId(source.codebook_id),
                .payload_bytes_per_block = 16,
                .is_asymmetric = source.is_asymmetric,
                .has_emins = false,
            };
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
                               ? kNativeVnniExpandedInt8MinCodebook
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
    [[nodiscard]] inline constexpr NativeVnniReusableDeviceAllocationFormat
    reusableDeviceVnniAllocationFormat(
        const NativeVnniFormatInfo &source) noexcept
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
     * @param source Original GGUF NativeVNNI format descriptor.
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
        const NativeVnniFormatInfo &source,
        uint8_t execution_codebook) noexcept
    {
        if (execution_codebook ==
            canonicalDeviceVnniCodebookId(source.codebook_id))
        {
            return true;
        }
        if (execution_codebook == 19)
            return !source.is_asymmetric;
        return execution_codebook == kNativeVnniExpandedInt8MinCodebook &&
               source.is_asymmetric &&
               !hasCompactMultiScaleVnniPayload(source.codebook_id) &&
               source.codebook_id != 8;
    }

    /**
     * @brief Capture-stable NativeVNNI formats reachable by one descriptor table.
     *
     * Immutable grouped tables contain only the execution codebooks present at
     * graph setup. A runtime-placement table may later receive the canonical
     * compact GPU representation or the migration-stable representation of the
     * same source tensor. Both launch masks must therefore be certified before
     * capture; discovering a new codebook after publication would otherwise
     * leave a valid runtime descriptor with no owning grouped kernel.
     */
    struct NativeVnniExecutionFormatEnvelope
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
    [[nodiscard]] inline constexpr uint32_t nativeVnniCodebookMaskBit(
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
    [[nodiscard]] inline constexpr bool addNativeVnniExecutionFormat(
        NativeVnniExecutionFormatEnvelope &envelope,
        uint8_t execution_codebook,
        NativeVnniSourceIdentity source,
        bool runtime_mutable) noexcept
    {
        const uint32_t execution_bit =
            nativeVnniCodebookMaskBit(execution_codebook);
        if (execution_bit == 0u)
            return false;

        if (!source.present)
        {
            if (runtime_mutable)
                return false;
            envelope.execution_codebook_mask |= execution_bit;
            envelope.policy_codebook_mask |= execution_bit;
            return true;
        }

        const NativeVnniFormatInfo *const source_format =
            native_vnni_formats::forSourceIdentity(
                source.codebook_id,
                source.is_superblock);
        if (!source_format ||
            !deviceVnniExecutionCompatibleWithSource(
                *source_format,
                execution_codebook))
        {
            return false;
        }

        const uint8_t canonical_codebook =
            canonicalDeviceVnniCodebookId(source.codebook_id);
        const uint32_t canonical_bit =
            nativeVnniCodebookMaskBit(canonical_codebook);
        if (canonical_bit == 0u)
            return false;

        uint32_t reachable_execution = execution_bit;
        if (runtime_mutable)
        {
            const auto migration =
                migrationStableDeviceVnniFormat(*source_format);
            const uint32_t migration_bit =
                nativeVnniCodebookMaskBit(migration.codebook_id);
            if (migration_bit == 0u)
                return false;
            reachable_execution |= canonical_bit | migration_bit;
        }

        envelope.execution_codebook_mask |= reachable_execution;
        envelope.policy_codebook_mask |= canonical_bit;
        return true;
    }

    /**
     * @brief Compute the exact logical pool regions for one packed matrix.
     *
     * Native-VNNI kernels require K to consist of complete 32-value execution
     * blocks. The loader enforces that shape contract before calling the pool;
     * this pure helper mirrors the resulting persistent allocation sizes.
     */
    [[nodiscard]] inline constexpr NativeVnniPackedRegionSizes
    nativeVnniPackedRegionSizes(
        size_t rows,
        size_t columns,
        const NativeVnniFormatInfo &format) noexcept
    {
        const size_t block_count = rows * (columns / 32);
        return {
            .payload_bytes =
                block_count * static_cast<size_t>(format.payload_bytes),
            .scales_bytes = block_count * sizeof(uint16_t),
            .mins_bytes =
                format.is_asymmetric ? block_count * sizeof(uint16_t) : 0,
            .emins_bytes =
                format.has_emins ? block_count * sizeof(uint32_t) : 0,
        };
    }

} // namespace llaminar2

#endif // LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H
