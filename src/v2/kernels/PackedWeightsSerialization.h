/**
 * @file PackedWeightsSerialization.h
 * @brief Serialization/deserialization for packed GEMM weights.
 *
 * Enables transferring pre-packed weights between MPI ranks instead of
 * repacking from the original quantized tensor on each rank.
 *
 * ## Wire Format (little-endian, x86-64)
 *
 * ```
 * [Header — 64 bytes]
 *   magic, version, format, metadata scalars, reserved
 * [Section table — 32 bytes]
 *   4 × uint64 sizes for data sections
 * [Data sections — contiguous]
 *   native_interleaved, payload, int8_flat, native_blocks
 * ```
 *
 * Total fixed overhead = 96 bytes, then variable data.
 */

#pragma once

#include "IPackedWeights.h"
#include "cpu/native_vnni/CPUPackedWeights.h"
#include "cpu/native_vnni/CPUNativeVNNIWeightPacker.h"
#include "utils/Logger.h"

#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2 {

/// Magic bytes identifying a packed weights blob: "LPWT"
static constexpr uint32_t PACKED_WEIGHTS_MAGIC = 0x5457504C; // "LPWT" in little-endian

/// Current wire format version.
static constexpr uint32_t PACKED_WEIGHTS_VERSION = 1;

#pragma pack(push, 1)

/// Fixed 64-byte header for serialized packed weights.
struct PackedWeightsHeader
{
    uint32_t magic;                    //  4B  "LPWT"
    uint32_t version;                  //  4B  wire format version
    uint32_t format;                   //  4B  PackedWeightsFormat enum
    int32_t  N;                        //  4B
    int32_t  K;                        //  4B
    int32_t  N_padded;                 //  4B
    int32_t  blocks_per_row;           //  4B
    uint8_t  codebook_id;              //  1B
    uint8_t  payload_bytes;            //  1B
    uint8_t  is_asymmetric;            //  1B
    uint8_t  is_nibble_lut;            //  1B
    uint8_t  is_superblock;            //  1B
    uint8_t  has_native_blocks;        //  1B
    int32_t  data_stride;              //  4B
    int32_t  interleaved_block_stride; //  4B
    int32_t  native_block_size;        //  4B
    uint8_t  reserved[18];            // 18B  zero-padded
};

static_assert(sizeof(PackedWeightsHeader) == 64,
    "PackedWeightsHeader must be exactly 64 bytes");

/// Section table following the header — sizes of each data section.
struct PackedWeightsSectionTable
{
    uint64_t interleaved_size;   // native_interleaved byte count
    uint64_t payload_size;       // payload byte count
    uint64_t int8_flat_size;     // int8_flat byte count
    uint64_t native_blocks_size; // native_blocks byte count
};

static_assert(sizeof(PackedWeightsSectionTable) == 32,
    "PackedWeightsSectionTable must be exactly 32 bytes");

#pragma pack(pop)

namespace packed_weights_serialization {

/// Serialize IPackedWeights to a self-describing byte buffer.
/// Supports CPUPackedWeights and CPUPackedWeightsWithNativeBlocks.
/// Returns empty vector if the format is not serializable.
inline std::vector<uint8_t> serialize(const IPackedWeights& weights)
{
    if (weights.format() != PackedWeightsFormat::CPU_NATIVE_VNNI)
    {
        LOG_WARN("[PackedWeightsSerialization] Cannot serialize format "
                 << static_cast<int>(weights.format()) << " (only CPU_NATIVE_VNNI supported)");
        return {};
    }

    // Extract underlying CPUNativeVNNIPackedWeights via CPUPackedWeights.
    const auto* cpu_pw = dynamic_cast<const cpu::native_vnni::CPUPackedWeights*>(&weights);
    if (!cpu_pw)
    {
        LOG_WARN("[PackedWeightsSerialization] dynamic_cast to CPUPackedWeights failed");
        return {};
    }

    const auto& packed = cpu_pw->packed();

    // Check for CPUPackedWeightsWithNativeBlocks subclass.
    const auto* with_nb = dynamic_cast<const cpu::native_vnni::CPUPackedWeightsWithNativeBlocks*>(&weights);

    // Build header.
    PackedWeightsHeader header{};
    header.magic                    = PACKED_WEIGHTS_MAGIC;
    header.version                  = PACKED_WEIGHTS_VERSION;
    header.format                   = static_cast<uint32_t>(PackedWeightsFormat::CPU_NATIVE_VNNI);
    header.N                        = packed.N;
    header.K                        = packed.K;
    header.N_padded                 = packed.N_padded;
    header.blocks_per_row           = packed.blocks_per_row;
    header.codebook_id              = packed.codebook_id;
    header.payload_bytes            = static_cast<uint8_t>(packed.payload_bytes);
    header.is_asymmetric            = packed.is_asymmetric ? 1 : 0;
    header.is_nibble_lut            = packed.is_nibble_lut ? 1 : 0;
    header.is_superblock            = packed.is_superblock ? 1 : 0;
    header.has_native_blocks        = with_nb ? 1 : 0;
    header.data_stride              = packed.data_stride;
    header.interleaved_block_stride = packed.interleaved_block_stride;
    header.native_block_size        = with_nb ? static_cast<int32_t>(with_nb->nativeBlockSize()) : 0;
    std::memset(header.reserved, 0, sizeof(header.reserved));

    // Build section table.
    PackedWeightsSectionTable sections{};
    sections.interleaved_size   = packed.native_interleaved.size();
    sections.payload_size       = packed.payload.size();
    sections.int8_flat_size     = packed.int8_flat.size();
    sections.native_blocks_size = with_nb ? with_nb->nativeBlocks().size() : 0;

    // Compute total size and allocate output buffer.
    const size_t total = sizeof(PackedWeightsHeader)
                       + sizeof(PackedWeightsSectionTable)
                       + sections.interleaved_size
                       + sections.payload_size
                       + sections.int8_flat_size
                       + sections.native_blocks_size;

    std::vector<uint8_t> buffer(total);
    uint8_t* dst = buffer.data();

    // Write header.
    std::memcpy(dst, &header, sizeof(header));
    dst += sizeof(header);

    // Write section table.
    std::memcpy(dst, &sections, sizeof(sections));
    dst += sizeof(sections);

    // Write data sections in section table order.
    if (sections.interleaved_size > 0)
    {
        std::memcpy(dst, packed.native_interleaved.data(), sections.interleaved_size);
        dst += sections.interleaved_size;
    }
    if (sections.payload_size > 0)
    {
        std::memcpy(dst, packed.payload.data(), sections.payload_size);
        dst += sections.payload_size;
    }
    if (sections.int8_flat_size > 0)
    {
        std::memcpy(dst, packed.int8_flat.data(), sections.int8_flat_size);
        dst += sections.int8_flat_size;
    }
    if (sections.native_blocks_size > 0)
    {
        std::memcpy(dst, with_nb->nativeBlocks().data(), sections.native_blocks_size);
        dst += sections.native_blocks_size;
    }

    return buffer;
}

/// Deserialize packed weights from a byte buffer.
/// Returns nullptr on validation failure (bad magic, version, truncated data, etc.).
inline std::unique_ptr<IPackedWeights> deserialize(const uint8_t* data, size_t size)
{
    constexpr size_t MIN_SIZE = sizeof(PackedWeightsHeader) + sizeof(PackedWeightsSectionTable);

    if (!data || size < MIN_SIZE)
    {
        LOG_WARN("[PackedWeightsSerialization] Buffer too small: " << size << " < " << MIN_SIZE);
        return nullptr;
    }

    // Read header.
    PackedWeightsHeader header;
    std::memcpy(&header, data, sizeof(header));

    if (header.magic != PACKED_WEIGHTS_MAGIC)
    {
        LOG_WARN("[PackedWeightsSerialization] Bad magic: 0x"
                 << std::hex << header.magic << " (expected 0x" << PACKED_WEIGHTS_MAGIC << ")");
        return nullptr;
    }

    if (header.version != PACKED_WEIGHTS_VERSION)
    {
        LOG_WARN("[PackedWeightsSerialization] Unsupported version: "
                 << header.version << " (expected " << PACKED_WEIGHTS_VERSION << ")");
        return nullptr;
    }

    if (header.format != static_cast<uint32_t>(PackedWeightsFormat::CPU_NATIVE_VNNI))
    {
        LOG_WARN("[PackedWeightsSerialization] Unsupported format: " << header.format);
        return nullptr;
    }

    // Read section table.
    PackedWeightsSectionTable sections;
    std::memcpy(&sections, data + sizeof(header), sizeof(sections));

    // Validate total size.
    const size_t expected_size = MIN_SIZE
                               + sections.interleaved_size
                               + sections.payload_size
                               + sections.int8_flat_size
                               + sections.native_blocks_size;

    if (size < expected_size)
    {
        LOG_WARN("[PackedWeightsSerialization] Buffer truncated: " << size
                 << " < expected " << expected_size);
        return nullptr;
    }

    // Reconstruct CPUNativeVNNIPackedWeights from header + data sections.
    cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
    packed.N                        = header.N;
    packed.K                        = header.K;
    packed.N_padded                 = header.N_padded;
    packed.blocks_per_row           = header.blocks_per_row;
    packed.codebook_id              = header.codebook_id;
    packed.payload_bytes            = header.payload_bytes;
    packed.is_asymmetric            = header.is_asymmetric != 0;
    packed.is_nibble_lut            = header.is_nibble_lut != 0;
    packed.is_superblock            = header.is_superblock != 0;
    packed.data_stride              = header.data_stride;
    packed.interleaved_block_stride = header.interleaved_block_stride;
    packed.workspace_data_          = nullptr;

    const uint8_t* src = data + MIN_SIZE;

    // Read native_interleaved (64-byte aligned).
    if (sections.interleaved_size > 0)
    {
        packed.native_interleaved.resize_uninitialized(sections.interleaved_size);
        std::memcpy(packed.native_interleaved.data(), src, sections.interleaved_size);
        src += sections.interleaved_size;
    }

    // Read payload.
    if (sections.payload_size > 0)
    {
        packed.payload.resize(sections.payload_size);
        std::memcpy(packed.payload.data(), src, sections.payload_size);
        src += sections.payload_size;
    }

    // Read int8_flat.
    if (sections.int8_flat_size > 0)
    {
        packed.int8_flat.resize(sections.int8_flat_size);
        std::memcpy(packed.int8_flat.data(), src, sections.int8_flat_size);
        src += sections.int8_flat_size;
    }

    // Read native_blocks if present.
    if (header.has_native_blocks && sections.native_blocks_size > 0)
    {
        std::vector<uint8_t> native_blocks(sections.native_blocks_size);
        std::memcpy(native_blocks.data(), src, sections.native_blocks_size);

        return std::make_unique<cpu::native_vnni::CPUPackedWeightsWithNativeBlocks>(
            std::move(packed),
            std::move(native_blocks),
            static_cast<size_t>(header.native_block_size));
    }

    return std::make_unique<cpu::native_vnni::CPUPackedWeights>(std::move(packed));
}

} // namespace packed_weights_serialization
} // namespace llaminar2
