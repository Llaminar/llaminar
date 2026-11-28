/**
 * @file CPUNativeQuantisedEncoding.h
 * @brief Allocation-free source-to-prepared encoding policy for CPU projections.
 *
 * The packer, prepared read view, cache/workspace contracts and ISA leaves
 * consume this one mapping. Source codebooks and physical encodings are
 * distinct: coded formats retain original source factors, while Q8 retains
 * signed bytes with one FP16 scale per block.
 */
#pragma once

#include "CPUNativeQuantisedPreparedFootprint.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/BlockStructures.h"
#include <cstdint>
#include <stdexcept>

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
        case kNativeQuantisedMXFP4Codebook: // MXFP4: doubled E2M1 LUT
        case 5: // Q4_1:  nibble → [0, 15]
            return true;
        default:
            return false;
        }
    }

    /** @brief Select the canonical CPU encoding using the original source tuple.
     * @param source Original GGUF descriptor, including superblock identity.
     * @return Physical encoding consumed by CPU GEMV/GEMM.
     * @throws std::invalid_argument For an uncataloged codebook.
     */
    [[nodiscard]] inline constexpr CPUNativeQuantisedEncoding preparedEncodingForSource(
        const NativeQuantisedFormatInfo &source)
    {
        const auto codebook = source.codebook_id;
        if (codebook == 5 && source.is_superblock)
            return CPUNativeQuantisedEncoding::Q4KSuperblockCoded;
        if (isQ5NativeSingleScaleSource(codebook, source.is_superblock))
            return CPUNativeQuantisedEncoding::Q5NativeSingleScale;
        if (codebook == 4 && source.is_superblock)
            return CPUNativeQuantisedEncoding::NibbleLUTSourceScale;
        if (is_nibble_lut_format(codebook))
            return CPUNativeQuantisedEncoding::NibbleLUT;
        if (codebook == 8)
            return CPUNativeQuantisedEncoding::Q6KNativeDualScale;
        if (codebook == 7 || codebook == 11 || hasCompactMultiScaleVnniPayload(codebook))
            return CPUNativeQuantisedEncoding::CompactMultiScale;
        if (isSignedInt8SingleScaleSource(codebook))
            return CPUNativeQuantisedEncoding::SignedInt8SingleScale;
        throw std::invalid_argument("Native CPU preparation requires a cataloged source codebook");
    }

    /** @return Whether migration must preserve this source's codes and superblock words, not staged products. */
    [[nodiscard]] inline constexpr bool migrationUsesSuperblockCodes(const NativeQuantisedFormatInfo &source)
    {
        return preparedSuperblockCoded(preparedEncodingForSource(source));
    }

    /** @brief Compute a source-aware cache footprint without materializing weight data.
     * @param source Exact original source descriptor.
     * @return Authoritative per-unit weight and activation byte counts for cache tiling.
     */
    [[nodiscard]] inline NativeQuantisedPreparedFootprint preparedFootprintForFormat(
        const NativeQuantisedFormatInfo &source)
    {
        const auto encoding = preparedEncodingForSource(source);
        auto result = NativeQuantisedPreparedFootprint{
            .encoding = encoding,
            .is_asymmetric = source.is_asymmetric,
            .weight_bytes_per_n_chunk_k_block = preparedInterleavedBlockStride(encoding, source.is_asymmetric, source.codebook_id),
            .activation_bytes_per_row_k_block = sizeof(Q8_1Block),
            .output_bytes_per_row_n_chunk = 64u * sizeof(float),
        };
        if (result.encoding == CPUNativeQuantisedEncoding::Q4KSuperblockCoded ||
            result.encoding == CPUNativeQuantisedEncoding::Q6KNativeDualScale ||
            result.encoding == CPUNativeQuantisedEncoding::NibbleLUTSourceScale ||
            (result.encoding == CPUNativeQuantisedEncoding::CompactMultiScale &&
             (source.codebook_id == 7 || source.codebook_id == 11)))
            result.weight_bytes_per_n_chunk_k_block += preparedSuperblockHeaderBytes(result.encoding, source.codebook_id) / 8u;
        return result;
    }

} // namespace llaminar2::cpu::native_quantised
