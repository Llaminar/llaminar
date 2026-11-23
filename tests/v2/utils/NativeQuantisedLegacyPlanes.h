/**
 * @file NativeQuantisedLegacyPlanes.h
 * @brief Read a prepared GPU NativeQuantised matrix as legacy per-block planes.
 *
 * Test oracles and byte-level inspections reason about the per-block layout
 * (one staged primary word, optional secondary word and Q2_K effective-minimum
 * word per 32-value block, plus the per-block payload). Immutable source-native
 * CUDA superblock matrices are prepared superblock-coded instead: bit-packed
 * codes (or codes embedded in the payload) plus one word per source
 * superblock. This helper downloads whichever representation the descriptor
 * names and returns the exact legacy view through the production codec
 * (native_quantised_superblock::decodeCodedMatrix), so an oracle is
 * independent of the representation and a coded matrix is never misread.
 *
 * Diagnostic/test use only: it copies device bytes to the host.
 */
#pragma once

#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "tensors/NativeQuantisedScalePlane.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    /** @brief Legacy per-block view of one prepared matrix, block-major (blk * n + row). */
    struct NativeQuantisedLegacyPlanes
    {
        std::vector<std::uint8_t> payload;   ///< Per-block payload bytes (code-in-payload restored).
        std::vector<std::uint16_t> primary;  ///< Staged primary word per block.
        std::vector<std::uint16_t> secondary;///< Staged secondary word per block (empty when absent).
        std::vector<std::uint32_t> emins;    ///< Q2_K effective-minimum word per block (empty when absent).
        bool coded = false;                  ///< Source representation was superblock-coded.
    };

    /**
     * @brief Download @p d as legacy per-block planes.
     * @param d Prepared descriptor (block-major planes).
     * @param payload_bytes Payload bytes per block of the live encoding.
     * @param download Callable (void *dst, const void *src, std::size_t bytes) copying device
     *        bytes to the host and completing before it returns.
     * @param primary_from_plane Callable (const std::vector<uint8_t>& plane, size_t blocks) ->
     *        std::vector<uint16_t> decoding a per-block primary plane (MXFP4's one-byte plane differs);
     *        used only for per-block matrices.
     */
    template <class Download, class PrimaryFromPlane>
    NativeQuantisedLegacyPlanes nativeQuantisedLegacyPlanes(const DeviceNativeQuantisedMatrixDesc &d,
                                                            std::uint32_t payload_bytes, Download &&download,
                                                            PrimaryFromPlane &&primary_from_plane)
    {
        namespace sb = native_quantised_superblock;
        const std::size_t rows = static_cast<std::size_t>(d.n);
        const std::size_t blocks = rows * d.blocks_per_row;
        NativeQuantisedLegacyPlanes out;
        out.payload.resize(blocks * payload_bytes);
        download(out.payload.data(), d.payload, out.payload.size());
        if (nativeQuantisedLayoutRepresentation(d.metadata_layout) !=
            NativeQuantisedMetadataRepresentation::SuperblockCoded)
        {
            std::vector<std::uint8_t> plane(nativeQuantisedScalePlaneLiveBytes(d.codebook_id, blocks));
            download(plane.data(), d.scales, plane.size());
            out.primary = primary_from_plane(plane, blocks);
            if (d.mins)
            {
                out.secondary.resize(blocks);
                download(out.secondary.data(), d.mins, blocks * sizeof(std::uint16_t));
            }
            if (d.emins)
            {
                out.emins.resize(blocks);
                download(out.emins.data(), d.emins, blocks * sizeof(std::uint32_t));
            }
            return out;
        }
        out.coded = true;
        const std::uint8_t codebook = d.codebook_id;
        const std::uint32_t phase = nativeQuantisedLayoutPhase(d.metadata_layout);
        const std::uint32_t shift = nativeQuantisedLayoutCodeBitShift(d.metadata_layout);
        std::vector<std::uint8_t> codes;
        if (!sb::superblockCodeInPayload(codebook))
        {
            const std::size_t bits = sb::superblockCodePlaneBits(codebook);
            codes.resize((shift + blocks * bits + 7u) / 8u);
            download(codes.data(), d.scales, codes.size());
            codes.resize(codes.size() + sb::kSuperblockCodePlanePadBytes, 0);
        }
        const std::size_t superblocks = rows * sb::superblocksPerRow(d.blocks_per_row, phase);
        std::vector<std::uint8_t> superblock_bytes(superblocks * sb::superblockFieldBytes(codebook));
        download(superblock_bytes.data(), d.mins, superblock_bytes.size());
        // Keep the word plane naturally aligned for the codec's typed reads.
        std::vector<std::uint32_t> superblock_words((superblock_bytes.size() + 3) / 4);
        std::memcpy(superblock_words.data(), superblock_bytes.data(), superblock_bytes.size());
        out.primary.resize(blocks);
        out.secondary.resize(blocks);
        std::vector<std::uint32_t> emins(blocks);
        sb::decodeCodedMatrix(codebook, rows, d.blocks_per_row, phase, shift, payload_bytes, out.payload.data(),
                              codes.data(), superblock_words.data(), out.primary.data(), out.secondary.data(),
                              emins.data());
        if (codebook == 10)
            out.emins = std::move(emins);
        return out;
    }
} // namespace llaminar2::test
