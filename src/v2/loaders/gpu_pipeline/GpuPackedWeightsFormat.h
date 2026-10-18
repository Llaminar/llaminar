#pragma once

/**
 * @file GpuPackedWeightsFormat.h
 * @brief GPU-native wire format for weight transfer between devices.
 *
 * Defines a self-describing serialization format for GPU separated VNNI
 * weights (payload + scales + mins + emins arrays).  This format can be:
 *
 *  - Uploaded directly to GPU and used immediately (no repack needed)
 *  - Sent via MPI for GPU→GPU weight transfers
 *  - Downloaded to host for CPU-side conversion
 *
 * ## Wire Layout (little-endian)
 *
 * ```
 * [GpuPackedWeightsHeader — 64 bytes]
 * [Payload section         — payload_size bytes]
 * [Scales section          — scales_size bytes]
 * [Mins section            — mins_size bytes, 0 if symmetric]
 * [Emins section           — emins_size bytes, 0 if not Q2_K]
 * ```
 *
 * The payload/scales/mins/emins sections are laid out identically to the
 * GPU VRAM representation (N-interleaved, linear index = b * N + n). IQ2_S
 * retains original d and an opaque coefficient byte rather than two half scales, so
 * a receiver with a GPU can upload the sections directly and point kernels
 * at them with zero conversion.
 */

#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace llaminar2 {

/// Magic bytes identifying a GPU packed weights blob: "GPWT"
static constexpr uint32_t GPU_PACKED_WEIGHTS_MAGIC = 0x54575047; // "GPWT" little-endian

/// Version4 adds mandatory IQ3_S original-base/raw-nibble tuples; older wires reject.
/// Version2 retains IQ2_S original d/coefficient metadata. Version1's same-sized
/// derived-half words must be rejected before any packed payload is adopted.
static constexpr uint32_t GPU_PACKED_WEIGHTS_VERSION = 4;

#pragma pack(push, 1)

/**
 * @brief Fixed 64-byte header for GPU-native packed weight serialization.
 *
 * All multi-byte fields are little-endian (x86-64 native).
 * Sections follow immediately after the header in declaration order.
 */
struct GpuPackedWeightsHeader
{
    uint32_t magic;              //  4B  "GPWT"
    uint32_t version;            //  4B  wire format version
    uint8_t  format;             //  1B  RepackFormat enum value
    uint8_t  codebook_id;        //  1B  NativeVnniFormatInfo codebook_id
    uint8_t  payload_bytes;      //  1B  bytes per payload block (16, 20, 24, 32, ...)
    uint8_t  is_asymmetric;      //  1B  secondary plane present (raw IQ4_XS coefficients for24)
    uint8_t  is_superblock;      //  1B  256-element super-blocks
    uint8_t  has_emins;          //  1B  Q2_K effective mins
    uint8_t  reserved_flags[2];  //  2B  zero-padded

    int32_t  N;                  //  4B  output features (rows)
    int32_t  K;                  //  4B  input features (columns)
    int32_t  blocks_per_row;     //  4B  K / 32

    uint64_t payload_size;       //  8B  bytes: blocks_per_row * N * payload_bytes
    uint64_t scales_size;        //  8B  bytes: blocks_per_row * N * sizeof(uint16_t)
    uint64_t mins_size;          //  8B  bytes: 0 if symmetric
    uint64_t emins_size;         //  8B  bytes: 0 if not Q2_K

    uint8_t  reserved[4];        //  4B  zero-padded
};

static_assert(sizeof(GpuPackedWeightsHeader) == 64,
    "GpuPackedWeightsHeader must be exactly 64 bytes");

#pragma pack(pop)

// ============================================================================
// Helpers
// ============================================================================

/// Compute total serialized size for a GPU packed weight blob.
inline size_t gpuPackedTotalSize(const GpuPackedWeightsHeader& hdr)
{
    return sizeof(GpuPackedWeightsHeader)
         + hdr.payload_size
         + hdr.scales_size
         + hdr.mins_size
         + hdr.emins_size;
}

/// Build a header from weight parameters.
inline GpuPackedWeightsHeader buildGpuPackedHeader(
    RepackFormat format,
    uint8_t codebook_id,
    uint8_t payload_bytes_per_block,
    bool is_asymmetric,
    bool is_superblock,
    bool has_emins,
    int N, int K)
{
    const int blocks_per_row = K / 32;
    const size_t total_blocks = static_cast<size_t>(blocks_per_row) * N;

    GpuPackedWeightsHeader hdr{};
    hdr.magic          = GPU_PACKED_WEIGHTS_MAGIC;
    hdr.version        = GPU_PACKED_WEIGHTS_VERSION;
    hdr.format         = static_cast<uint8_t>(format);
    hdr.codebook_id    = codebook_id;
    hdr.payload_bytes  = payload_bytes_per_block;
    hdr.is_asymmetric  = is_asymmetric ? 1 : 0;
    hdr.is_superblock  = is_superblock ? 1 : 0;
    hdr.has_emins      = has_emins ? 1 : 0;
    std::memset(hdr.reserved_flags, 0, sizeof(hdr.reserved_flags));
    hdr.N              = N;
    hdr.K              = K;
    hdr.blocks_per_row = blocks_per_row;
    hdr.payload_size   = total_blocks * payload_bytes_per_block;
    hdr.scales_size    = total_blocks * sizeof(uint16_t);
    hdr.mins_size      = is_asymmetric ? total_blocks * sizeof(uint16_t) : 0;
    hdr.emins_size     = has_emins ? total_blocks * sizeof(uint32_t) : 0;
    std::memset(hdr.reserved, 0, sizeof(hdr.reserved));
    return hdr;
}

/// Validate a header read from a byte buffer.  Returns true if all
/// invariants hold (magic, version, non-negative sizes).
inline bool validateGpuPackedHeader(const GpuPackedWeightsHeader& hdr, size_t buffer_size)
{
    if (hdr.magic != GPU_PACKED_WEIGHTS_MAGIC) return false;
    if (hdr.version != GPU_PACKED_WEIGHTS_VERSION) return false;
    if (hdr.N <= 0 || hdr.K <= 0 || hdr.blocks_per_row <= 0) return false;
    // IQ3_S physical11 owns original half base and mandatory raw nibble.
    if (hdr.codebook_id == 11 || hdr.format == static_cast<uint8_t>(RepackFormat::IQ3_S))
    {
        const uint64_t blocks = uint64_t(hdr.N) * uint64_t(hdr.blocks_per_row);
        if (hdr.codebook_id != 11 || hdr.format != static_cast<uint8_t>(RepackFormat::IQ3_S) ||
            hdr.payload_bytes != 13 || hdr.is_asymmetric != 1 || hdr.is_superblock != 1 || hdr.has_emins != 0 ||
            hdr.K % 32 != 0 || hdr.blocks_per_row != hdr.K / 32 || blocks > UINT64_MAX / 17u ||
            hdr.payload_size != blocks * 13u || hdr.scales_size != blocks * 2u ||
            hdr.mins_size != blocks * 2u || hdr.emins_size != 0u) return false;
    }
    // Source4+superblock is distinct from IQ4_NL despite sharing the LUT.
    // This is metadata-only rejection before a wire payload can be adopted.
    if (hdr.format == static_cast<uint8_t>(RepackFormat::IQ4_XS) &&
        (hdr.codebook_id != kNativeVnniIQ4XSSourceScaleCodebook ||
         hdr.payload_bytes != 16 || hdr.is_asymmetric != 1 ||
         hdr.is_superblock != 1 || hdr.has_emins != 0)) return false;
    if (hdr.codebook_id == kNativeVnniIQ4XSSourceScaleCodebook &&
        hdr.format != static_cast<uint8_t>(RepackFormat::IQ4_XS)) return false;
    if (hdr.codebook_id == kNativeVnniIQ4XSSourceScaleCodebook)
    {
        const uint64_t blocks = uint64_t(hdr.N) * uint64_t(hdr.blocks_per_row);
        if (hdr.K % 32 != 0 || hdr.blocks_per_row != hdr.K / 32 ||
            blocks > UINT64_MAX / 20u ||
            hdr.payload_size != blocks * 16u || hdr.scales_size != blocks * 2u ||
            hdr.mins_size != blocks * 2u || hdr.emins_size != 0u) return false;
    }
    if (buffer_size < gpuPackedTotalSize(hdr)) return false;
    return true;
}

/// Pointers into a GPU packed weight buffer (either host or device memory).
/// All pointers are into the buffer passed to parseGpuPackedBuffer().
struct GpuPackedWeightsView
{
    GpuPackedWeightsHeader header;
    const uint8_t*  payload = nullptr;   ///< [bpr * N * payload_bytes]
    const uint16_t* scales  = nullptr;   ///< [bpr * N]
    const uint16_t* mins    = nullptr;   ///< [bpr * N] or nullptr
    const uint32_t* emins   = nullptr;   ///< [bpr * N] or nullptr
};

/// Parse a serialized GPU packed weight buffer into a view.
/// The view points into the provided buffer (no copy).
/// Returns false if validation fails.
inline bool parseGpuPackedBuffer(const void* buffer, size_t size, GpuPackedWeightsView& out)
{
    if (!buffer || size < sizeof(GpuPackedWeightsHeader)) return false;

    std::memcpy(&out.header, buffer, sizeof(GpuPackedWeightsHeader));
    if (!validateGpuPackedHeader(out.header, size)) return false;

    const auto* base = static_cast<const uint8_t*>(buffer) + sizeof(GpuPackedWeightsHeader);
    size_t offset = 0;

    out.payload = base + offset;
    offset += out.header.payload_size;

    out.scales = reinterpret_cast<const uint16_t*>(base + offset);
    offset += out.header.scales_size;

    if (out.header.mins_size > 0) {
        out.mins = reinterpret_cast<const uint16_t*>(base + offset);
        offset += out.header.mins_size;
    }

    if (out.header.emins_size > 0) {
        out.emins = reinterpret_cast<const uint32_t*>(base + offset);
    }

    return true;
}

} // namespace llaminar2
