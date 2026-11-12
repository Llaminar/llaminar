/**
 * @file CPUNativeQuantisedEncoding.h
 * @brief Allocation-free source-to-prepared encoding policy for CPU projections.
 *
 * The packer, prepared read view, cache/workspace contracts and ISA leaves
 * consume this one mapping. Source codebooks and physical encodings are
 * distinct: IQ4_XS and IQ3_S retain original source factors, while explicit
 * activation rotation owns its existing expanded-INT8 representation.
 */
#pragma once

#include "CPUNativeQuantisedPreparedFootprint.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/BlockStructures.h"
#include <cstdint>

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief Identify payloads decoded with a four-bit lookup/linear map.
     * @param codebook_id Prepared CPU payload codebook.
     * @return Whether its canonical nibble decoder applies.
     */
    inline constexpr bool is_nibble_lut_format(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0: // Q4_0:  nibble - 8 → [-8, +7]
        case 4: // IQ4_NL / IQ4_XS: kvalues_iq4nl LUT
        case 5: // Q4_1:  nibble → [0, 15]
            return true;
        default:
            return false;
        }
    }

    /**
     * @brief Select the physical CPU execution encoding for a source codebook.
     *
     * @param codebook_id NativeQuantised execution codebook identifier.
     * @param rotation_enabled Whether preparation applies activation rotation;
     *        rotated values are requantized to the expanded INT8 encoding.
     * @return Physical encoding consumed by CPU GEMV/GEMM kernels.
     */
    [[nodiscard]] inline CPUNativeQuantisedEncoding preparedEncodingForCodebook(
        uint8_t codebook_id,
        bool rotation_enabled = false) noexcept
    {
        if (rotation_enabled)
            return CPUNativeQuantisedEncoding::ExpandedInt8;
        if (codebook_id == 11)
            return CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale;
        if (is_nibble_lut_format(codebook_id))
            return CPUNativeQuantisedEncoding::NibbleLUT;
        if (codebook_id == 8)
            return CPUNativeQuantisedEncoding::Q6KNativeDualScale;
        if (hasCompactMultiScaleVnniPayload(codebook_id))
            return CPUNativeQuantisedEncoding::CompactMultiScale;
        return CPUNativeQuantisedEncoding::ExpandedInt8;
    }

    /** @brief Select CPU encoding using the original source tuple.
     * @param source Original GGUF descriptor, including superblock identity.
     * @param rotation_enabled Explicit separately selected rotated INT8 policy.
     * @return Lossless ordinary encoding or the unchanged explicit rotation encoding.
     */
    [[nodiscard]] inline CPUNativeQuantisedEncoding preparedEncodingForSource(
        const NativeQuantisedFormatInfo &source, bool rotation_enabled = false) noexcept
    {
        if (!rotation_enabled && source.codebook_id == 4 && source.is_superblock)
            return CPUNativeQuantisedEncoding::NibbleLUTSourceScale;
        return preparedEncodingForCodebook(source.codebook_id, rotation_enabled);
    }

    /**
     * @brief Build an exact canonical cache footprint without materializing
     * a matrix.
     *
     * This helper is used by planning/corpus tools that know format metadata
     * but intentionally do not allocate production-sized weights. It prices
     * the canonical layout only: source identity alone cannot select the
     * compact layout, whose payload width depends on the exact codebook.
     */
    [[nodiscard]] inline NativeQuantisedPreparedFootprint preparedFootprintForFormat(
        uint8_t codebook_id,
        bool is_asymmetric,
        bool rotation_enabled = false) noexcept
    {
        const CPUNativeQuantisedEncoding encoding =
            preparedEncodingForCodebook(codebook_id, rotation_enabled);
        const bool prepared_is_asymmetric =
            rotation_enabled ? false : is_asymmetric;
        return NativeQuantisedPreparedFootprint{
            .encoding = encoding,
            .is_asymmetric = prepared_is_asymmetric,
            .weight_bytes_per_n_chunk_k_block =
                static_cast<std::uint64_t>(preparedInterleavedBlockStride(
                    encoding, prepared_is_asymmetric)),
            .activation_bytes_per_row_k_block = sizeof(Q8_1Block),
            .output_bytes_per_row_n_chunk = 64u * sizeof(float),
        };
    }

    /** @brief Compute source-aware footprint without materializing weight data.
     * @param source Exact original source descriptor.
     * @param rotation_enabled Explicit CPU rotation policy.
     * @param layout Physical unit layout to price; defaults to this build's
     *   production layout. An unimplemented layout fails closed as an invalid
     *   footprint instead of substituting canonical geometry.
     * @return Authoritative physical per-unit weight and activation byte counts.
     */
    [[nodiscard]] inline NativeQuantisedPreparedFootprint preparedFootprintForFormat(
        const NativeQuantisedFormatInfo &source, bool rotation_enabled = false,
        CPUNativeQuantisedLayout layout = kHostPreparedLayout) noexcept
    {
        const CPUNativeQuantisedEncoding encoding =
            preparedEncodingForSource(source, rotation_enabled);
        const bool prepared_is_asymmetric =
            rotation_enabled ? false : source.is_asymmetric;
        const PreparedUnitSections sections =
            preparedUnitSections(layout, encoding, prepared_is_asymmetric, source);
        return NativeQuantisedPreparedFootprint{
            .layout = layout,
            .encoding = encoding,
            .is_asymmetric = prepared_is_asymmetric,
            .weight_bytes_per_n_chunk_k_block = sections.block_stride,
            .activation_bytes_per_row_k_block = sizeof(Q8_1Block),
            .output_bytes_per_row_n_chunk = 64u * sizeof(float),
        };
    }

} // namespace llaminar2::cpu::native_quantised
