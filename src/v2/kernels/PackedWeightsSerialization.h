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
#include "cpu/gemm/CPUPackedWeights.h"
#include "cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "utils/Logger.h"

#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

namespace llaminar2 {

/// Magic bytes identifying a packed weights blob: "LPWT"
static constexpr uint32_t PACKED_WEIGHTS_MAGIC = 0x5457504C; // "LPWT" in little-endian

/// Version 3 requires lossless native multi-scale preparation. Version-2
/// archives may contain lossy expanded experts with otherwise valid metadata;
/// reject them rather than accepting stale arithmetic from an older producer.
static constexpr uint32_t PACKED_WEIGHTS_VERSION = 3;

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
    uint8_t  prepared_encoding;        //  1B  CPUNativeVNNIEncoding
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

/// Number of independently owned byte sections in the packed-weight format.
inline constexpr size_t PACKED_WEIGHT_SECTION_COUNT = 4;

/**
 * @brief Decode and validate the prepared CPU encoding stored on the wire.
 *
 * The wire byte is not trusted merely because the surrounding codebook is
 * known: preparation may legitimately choose a different physical encoding
 * after activation rotation. Rejecting unknown ordinals here prevents a
 * transferred expert from reaching a kernel with guessed metadata offsets.
 */
inline cpu::native_vnni::CPUNativeVNNIEncoding decodePreparedEncoding(
    uint8_t value)
{
    using Encoding = cpu::native_vnni::CPUNativeVNNIEncoding;
    switch (value)
    {
    case static_cast<uint8_t>(Encoding::NibbleLUT):
        return Encoding::NibbleLUT;
    case static_cast<uint8_t>(Encoding::ExpandedInt8):
        return Encoding::ExpandedInt8;
    case static_cast<uint8_t>(Encoding::Q6KNativeDualScale):
        return Encoding::Q6KNativeDualScale;
    case static_cast<uint8_t>(Encoding::CompactMultiScale):
        return Encoding::CompactMultiScale;
    default:
        throw std::invalid_argument(
            "Packed-weight transfer received an unknown prepared encoding");
    }
}

/**
 * @brief Fixed metadata required to reconstruct one packed-weight object.
 *
 * This is also the metadata record used by direct MPI expert movement.  The
 * bulk sections remain in their native allocations and are transferred
 * separately, avoiding construction of a concatenated serialization blob.
 */
struct PackedWeightsTransferDescriptor
{
    PackedWeightsHeader header{};
    PackedWeightsSectionTable sections{};
};

static_assert(std::is_trivially_copyable_v<PackedWeightsTransferDescriptor>,
              "Packed-weight transfer metadata must be MPI-byte transferable");

/** @brief Read-only native sections owned by one prepared weight object. */
struct PackedWeightsConstSectionViews
{
    std::array<const uint8_t *, PACKED_WEIGHT_SECTION_COUNT> data{};
    std::array<size_t, PACKED_WEIGHT_SECTION_COUNT> sizes{};
};

/**
 * @brief Final receive allocation and its writable native section views.
 *
 * `weights` owns every address in `data`.  Moving the target or its weight
 * object preserves the section addresses because the underlying vectors move
 * their allocations rather than their bytes.
 */
struct PackedWeightsReceiveTarget
{
    std::unique_ptr<IPackedWeights> weights;
    std::array<uint8_t *, PACKED_WEIGHT_SECTION_COUNT> data{};
    std::array<size_t, PACKED_WEIGHT_SECTION_COUNT> sizes{};
};

/** @brief Return the host page size used by direct receive allocations. */
inline size_t directReceivePageSize()
{
#ifdef __linux__
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size > 0)
        return static_cast<size_t>(page_size);
#endif
    return 4096;
}

/**
 * @brief Allocate logical elements with a page-aligned external-policy range.
 *
 * `AlignedVector` page-aligns allocations at least one page in size. Reserving
 * one page even for a smaller logical section lets the direct MPI path install
 * an exact NUMA policy without rounding into an unrelated heap allocation.
 */
template <typename T>
inline void resizeDirectReceiveSection(
    AlignedVector<T> &buffer, size_t logical_elements)
{
    if (logical_elements == 0)
        return;
    const size_t page_elements =
        (directReceivePageSize() + sizeof(T) - 1) / sizeof(T);
    buffer.reserve(std::max(logical_elements, page_elements));
    buffer.resize_uninitialized(logical_elements);
}

/**
 * @brief Describe prepared weights without copying their bulk data.
 *
 * @param weights Prepared CPU NativeVNNI weights.
 * @param descriptor Self-describing scalar metadata and section lengths.
 * @param views Native read-only section addresses.
 * @return `true` when the concrete format is directly transferable.
 */
inline bool describeTransfer(
    const IPackedWeights &weights,
    PackedWeightsTransferDescriptor &descriptor,
    PackedWeightsConstSectionViews &views)
{
    descriptor = PackedWeightsTransferDescriptor{};
    views = PackedWeightsConstSectionViews{};

    if (weights.format() != PackedWeightsFormat::CPU_NATIVE_VNNI)
    {
        LOG_WARN("[PackedWeightsSerialization] Cannot describe format "
                 << static_cast<int>(weights.format())
                 << " (only CPU_NATIVE_VNNI supported)");
        return false;
    }

    const auto *cpu_pw =
        dynamic_cast<const cpu::native_vnni::CPUPackedWeights *>(&weights);
    if (!cpu_pw)
    {
        LOG_WARN("[PackedWeightsSerialization] dynamic_cast to CPUPackedWeights failed");
        return false;
    }

    const auto &packed = cpu_pw->packed();
    const auto *with_nb =
        dynamic_cast<const cpu::native_vnni::CPUPackedWeightsWithNativeBlocks *>(
            &weights);

    auto &header = descriptor.header;
    header.magic = PACKED_WEIGHTS_MAGIC;
    header.version = PACKED_WEIGHTS_VERSION;
    header.format =
        static_cast<uint32_t>(PackedWeightsFormat::CPU_NATIVE_VNNI);
    header.N = packed.N;
    header.K = packed.K;
    header.N_padded = packed.N_padded;
    header.blocks_per_row = packed.blocks_per_row;
    header.codebook_id = packed.codebook_id;
    header.payload_bytes = static_cast<uint8_t>(packed.payload_bytes);
    header.is_asymmetric = packed.is_asymmetric ? 1 : 0;
    header.prepared_encoding = static_cast<uint8_t>(packed.encoding);
    header.is_superblock = packed.is_superblock ? 1 : 0;
    header.has_native_blocks = with_nb ? 1 : 0;
    header.data_stride = packed.data_stride;
    header.interleaved_block_stride = packed.interleaved_block_stride;
    header.native_block_size =
        with_nb ? static_cast<int32_t>(with_nb->nativeBlockSize()) : 0;
    std::memset(header.reserved, 0, sizeof(header.reserved));

    auto &sections = descriptor.sections;
    sections.interleaved_size = packed.native_interleaved.size();
    sections.payload_size = packed.payload.size();
    sections.int8_flat_size = packed.int8_flat.size();
    sections.native_blocks_size = with_nb ? with_nb->nativeBlocks().size() : 0;

    views.data = {
        packed.native_interleaved.data(),
        packed.payload.data(),
        reinterpret_cast<const uint8_t *>(packed.int8_flat.data()),
        with_nb ? with_nb->nativeBlocks().data() : nullptr,
    };
    views.sizes = {
        static_cast<size_t>(sections.interleaved_size),
        static_cast<size_t>(sections.payload_size),
        static_cast<size_t>(sections.int8_flat_size),
        static_cast<size_t>(sections.native_blocks_size),
    };
    return true;
}

/**
 * @brief Allocate the final packed-weight object for direct section receives.
 *
 * The three eager CPU sections use uninitialized aligned storage because MPI
 * overwrites every advertised byte.  This establishes destination NUMA
 * placement on the receiving rank and avoids both zero-fill and a subsequent
 * deserialize copy.
 *
 * @throws std::invalid_argument for malformed or unsupported metadata.
 * @throws std::length_error when a section cannot be represented locally.
 */
inline PackedWeightsReceiveTarget allocateTransferTarget(
    const PackedWeightsTransferDescriptor &descriptor)
{
    const auto &header = descriptor.header;
    const auto &sections = descriptor.sections;
    if (header.magic != PACKED_WEIGHTS_MAGIC ||
        header.version != PACKED_WEIGHTS_VERSION ||
        header.format !=
            static_cast<uint32_t>(PackedWeightsFormat::CPU_NATIVE_VNNI))
    {
        throw std::invalid_argument(
            "Direct packed-weight transfer received invalid format metadata");
    }
    if (header.N <= 0 || header.K <= 0 || header.N_padded < header.N ||
        header.blocks_per_row <= 0 || header.data_stride <= 0 ||
        header.interleaved_block_stride <= 0)
    {
        throw std::invalid_argument(
            "Direct packed-weight transfer received invalid geometry metadata");
    }
    if ((header.has_native_blocks == 0 && sections.native_blocks_size != 0) ||
        (header.has_native_blocks != 0 && header.native_block_size <= 0))
    {
        throw std::invalid_argument(
            "Direct packed-weight transfer received inconsistent native-block metadata");
    }

    const std::array<uint64_t, PACKED_WEIGHT_SECTION_COUNT> wire_sizes{
        sections.interleaved_size,
        sections.payload_size,
        sections.int8_flat_size,
        sections.native_blocks_size,
    };
    for (uint64_t bytes : wire_sizes)
    {
        if (bytes > static_cast<uint64_t>(
                        std::numeric_limits<size_t>::max()))
        {
            throw std::length_error(
                "Direct packed-weight transfer section exceeds host size_t");
        }
    }

    cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
    packed.N = header.N;
    packed.K = header.K;
    packed.N_padded = header.N_padded;
    packed.blocks_per_row = header.blocks_per_row;
    packed.codebook_id = header.codebook_id;
    packed.payload_bytes = header.payload_bytes;
    packed.is_asymmetric = header.is_asymmetric != 0;
    packed.encoding = decodePreparedEncoding(header.prepared_encoding);
    packed.is_superblock = header.is_superblock != 0;
    packed.data_stride = header.data_stride;
    packed.interleaved_block_stride = header.interleaved_block_stride;
    packed.workspace_data_ = nullptr;

    resizeDirectReceiveSection(
        packed.native_interleaved,
        static_cast<size_t>(sections.interleaved_size));
    resizeDirectReceiveSection(
        packed.payload,
        static_cast<size_t>(sections.payload_size));
    resizeDirectReceiveSection(
        packed.int8_flat,
        static_cast<size_t>(sections.int8_flat_size));

    PackedWeightsReceiveTarget target;
    target.data[0] = packed.native_interleaved.data();
    target.data[1] = packed.payload.data();
    target.data[2] =
        reinterpret_cast<uint8_t *>(packed.int8_flat.data());
    target.sizes = {
        static_cast<size_t>(sections.interleaved_size),
        static_cast<size_t>(sections.payload_size),
        static_cast<size_t>(sections.int8_flat_size),
        static_cast<size_t>(sections.native_blocks_size),
    };

    if (header.has_native_blocks != 0)
    {
        std::vector<uint8_t> native_blocks(
            static_cast<size_t>(sections.native_blocks_size));
        target.data[3] = native_blocks.data();
        target.weights = std::make_unique<
            cpu::native_vnni::CPUPackedWeightsWithNativeBlocks>(
            std::move(packed),
            std::move(native_blocks),
            static_cast<size_t>(header.native_block_size));
    }
    else
    {
        target.weights =
            std::make_unique<cpu::native_vnni::CPUPackedWeights>(
                std::move(packed));
    }
    return target;
}

/**
 * @brief Serialize packed weights into caller-owned contiguous storage.
 *
 * Buffers that expose `resize_uninitialized()` avoid first-touching pages with
 * zeroes immediately before the serializer overwrites every byte.  Ordinary
 * `std::vector` callers retain their existing behavior.
 *
 * @tparam ByteBuffer Contiguous byte container with `data()`, `clear()`, and
 *         either `resize()` or `resize_uninitialized()`.
 * @param weights Prepared CPU NativeVNNI weights.
 * @param buffer Destination buffer, replaced with one complete wire record.
 * @return `true` when the format was serialized completely.
 */
template <typename ByteBuffer>
inline bool serializeInto(const IPackedWeights& weights, ByteBuffer& buffer)
{
    PackedWeightsTransferDescriptor descriptor;
    PackedWeightsConstSectionViews views;
    if (!describeTransfer(weights, descriptor, views))
    {
        buffer.clear();
        return false;
    }
    const auto &header = descriptor.header;
    const auto &sections = descriptor.sections;

    // Compute total size and allocate output buffer.
    const size_t total = sizeof(PackedWeightsHeader)
                       + sizeof(PackedWeightsSectionTable)
                       + sections.interleaved_size
                       + sections.payload_size
                       + sections.int8_flat_size
                       + sections.native_blocks_size;

    if constexpr (requires(ByteBuffer &candidate, size_t bytes)
                  { candidate.resize_uninitialized(bytes); })
    {
        buffer.resize_uninitialized(total);
    }
    else
    {
        buffer.resize(total);
    }
    uint8_t* dst = buffer.data();

    // Write header.
    std::memcpy(dst, &header, sizeof(header));
    dst += sizeof(header);

    // Write section table.
    std::memcpy(dst, &sections, sizeof(sections));
    dst += sizeof(sections);

    // Write data sections in section table order.
    for (size_t section = 0; section < PACKED_WEIGHT_SECTION_COUNT; ++section)
    {
        if (views.sizes[section] == 0)
            continue;
        std::memcpy(dst, views.data[section], views.sizes[section]);
        dst += views.sizes[section];
    }

    return true;
}

/// Serialize IPackedWeights to a self-describing standard byte vector.
inline std::vector<uint8_t> serialize(const IPackedWeights& weights)
{
    std::vector<uint8_t> buffer;
    if (!serializeInto(weights, buffer))
        return {};
    return buffer;
}

/**
 * @brief Deserialize a portable archive into final owned CPU execution storage.
 * @param data Beginning of the complete wire record.
 * @param size Readable wire bytes.
 * @param placement First-touch policy applied before copying execution bytes.
 * @return Owned weights, or nullptr for invalid wire metadata.
 * @throws std::runtime_error When requested final placement cannot be certified.
 */
inline std::unique_ptr<IPackedWeights> deserialize(
    const uint8_t* data, size_t size,
    CPUWeightStoragePlacement placement = CPUWeightStoragePlacement::local())
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
    try
    {
        packed.encoding = decodePreparedEncoding(header.prepared_encoding);
    }
    catch (const std::invalid_argument &error)
    {
        LOG_WARN("[PackedWeightsSerialization] " << error.what());
        return nullptr;
    }
    packed.is_superblock            = header.is_superblock != 0;
    packed.data_stride              = header.data_stride;
    packed.interleaved_block_stride = header.interleaved_block_stride;
    packed.workspace_data_          = nullptr;

    const uint8_t* src = data + MIN_SIZE;

    // Read native_interleaved (64-byte aligned).
    if (sections.interleaved_size > 0)
    {
        // Establish the final CPU destination before copying the eager wire
        // representation. Never migrate or repack it after engine publication.
        packed.native_interleaved = placement.allocate<uint8_t>(sections.interleaved_size);
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
