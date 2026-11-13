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
#include "cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#include "utils/Logger.h"

#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

namespace llaminar2 {

/// Magic bytes identifying a packed weights blob: "LPWT"
static constexpr uint32_t PACKED_WEIGHTS_MAGIC = 0x5457504C; // "LPWT" in little-endian

/// Version 5 additionally retains original IQ2_S bases and coefficient bytes.
/// Version-4 IQ2_S archives narrowed both derived factors to half. Their
/// same-sized words have different semantics and must be rejected before any
/// payload is adopted. Q6_K's version-4 lossless representation is unchanged.
/// Version-3 Q6 and version-2 expanded multi-scale archives also remain rejected.
/// Version6 preserves IQ4_XS original half bases and encoded signed coefficients.
/// Version5 and earlier archives are rejected before any section is adopted.
/// Version7 retains IQ3_S original half bases and raw nibbles in CPU encoding5.
/// Version8 carries the prepared CPU layout in the sealed header identity:
/// receivers decode `prepared_layout` and re-derive both strides from that
/// layout's section tables before adopting any payload. Exactly version 7
/// remains admissible as an explicitly canonical archive — its implied layout
/// is the canonical selector by statement, never a guess; version 6 and
/// earlier are rejected globally, not per-format.
static constexpr uint32_t PACKED_WEIGHTS_VERSION = 8;
static constexpr uint32_t PACKED_WEIGHTS_CANONICAL_VERSION = 7;

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
    uint8_t  prepared_encoding;        //  1B  CPUNativeQuantisedEncoding
    uint8_t  is_superblock;            //  1B
    uint8_t  has_native_blocks;        //  1B
    int32_t  data_stride;              //  4B
    int32_t  interleaved_block_stride; //  4B
    int32_t  native_block_size;        //  4B
    uint8_t  prepared_layout;          //  1B  v8: sealed CPU layout identity
    uint8_t  reserved[17];            // 17B  zero-padded
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
inline cpu::native_quantised::CPUNativeQuantisedEncoding decodePreparedEncoding(
    uint8_t value)
{
    using Encoding = cpu::native_quantised::CPUNativeQuantisedEncoding;
    switch (value)
    {
    case static_cast<uint8_t>(Encoding::IQ3SExpandedSourceScale):
        return Encoding::IQ3SExpandedSourceScale;
    case static_cast<uint8_t>(Encoding::NibbleLUTSourceScale):
        return Encoding::NibbleLUTSourceScale;
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
 * @brief Decode and validate the prepared CPU layout stored on the wire.
 *
 * Version-8 headers name the layout in their sealed identity; receivers
 * never trust the byte without confirming it names a known layout. Knowing
 * a value is not the same as the build implementing it — section tables
 * decide implementability and fail closed in the stride re-derivation.
 */
inline cpu::native_quantised::CPUNativeQuantisedLayout decodePreparedLayout(
    uint8_t value)
{
    using Layout = cpu::native_quantised::CPUNativeQuantisedLayout;
    switch (value)
    {
    case static_cast<uint8_t>(Layout::InterleavedCanonical):
        return Layout::InterleavedCanonical;
    case static_cast<uint8_t>(Layout::InterleavedCompactA64):
        return Layout::InterleavedCompactA64;
    default:
        throw std::invalid_argument(
            "Packed-weight transfer received an unknown prepared layout");
    }
}

/**
 * @brief Resolve the admitted layout for an archive or transfer header.
 *
 * Exactly version 8 decodes the sealed `prepared_layout` byte; exactly
 * version 7 is admissible as an explicitly canonical archive (the implied
 * layout is a versioned statement, never a guess about the sender); every
 * earlier version is rejected globally, not per-format.
 *
 * @throws std::invalid_argument for any version outside {7, 8} or an
 *         unknown version-8 layout byte.
 */
inline cpu::native_quantised::CPUNativeQuantisedLayout admittedPreparedLayout(
    const PackedWeightsHeader &header)
{
    if (header.version == PACKED_WEIGHTS_VERSION)
        return decodePreparedLayout(header.prepared_layout);
    if (header.version == PACKED_WEIGHTS_CANONICAL_VERSION)
        return cpu::native_quantised::CPUNativeQuantisedLayout::
            InterleavedCanonical;
    throw std::invalid_argument(
        "Packed-weight transfer received an unsupported archive version " +
        std::to_string(header.version));
}

/**
 * @brief Re-derive both strides from the admitted layout before adoption.
 *
 * The receiver never trusts the header's strides alone: the section tables
 * of the decoded layout (with the header's encoding, asymmetry and cataloged
 * source identity) must reproduce them exactly. An unimplemented layout
 * yields invalid sections and fails closed here, before any payload byte is
 * adopted.
 *
 * @throws std::invalid_argument when the layout has no implemented section
 *         tables, the source is uncataloged, or the derived strides
 *         disagree with the header.
 */
inline void requirePreparedLayoutStrides(
    const PackedWeightsHeader &header,
    cpu::native_quantised::CPUNativeQuantisedLayout layout)
{
    const auto *const source = native_quantised_formats::forSourceIdentity(
        header.codebook_id, header.is_superblock);
    if (!source)
        throw std::invalid_argument(
            "Packed-weight transfer received an uncataloged source identity");
    const auto sections = cpu::native_quantised::preparedUnitSections(
        layout,
        decodePreparedEncoding(header.prepared_encoding),
        header.is_asymmetric != 0,
        *source);
    if (!sections.isValid() ||
        header.data_stride !=
            static_cast<int32_t>(sections.payload_stride) ||
        header.interleaved_block_stride !=
            static_cast<int32_t>(sections.block_stride))
    {
        throw std::invalid_argument(
            "Packed-weight transfer strides disagree with the admitted "
            "prepared layout's section tables");
    }
}

/** @brief Validate IQ3_S/IQ4_XS source tuples before adopting CPU sections.
 * @param header Archive or direct-transfer scalar header.
 * @return Whether encoding4 identifies exactly the IQ4_XS source tuple, and
 *         ordinary IQ4_XS does not claim the retired derived-half nibble encoding.
 */
inline bool validNativeSourceScaleTupleHeader(const PackedWeightsHeader &header) noexcept
{
    using Encoding = cpu::native_quantised::CPUNativeQuantisedEncoding;
    if (header.prepared_encoding == static_cast<uint8_t>(Encoding::IQ3SExpandedSourceScale))
        return header.codebook_id == 11 && header.is_superblock == 1 &&
               header.is_asymmetric == 0 && header.payload_bytes == 13 &&
               header.data_stride == static_cast<int32_t>(cpu::native_quantised::preparedDataStride(Encoding::IQ3SExpandedSourceScale)) &&
               header.interleaved_block_stride == static_cast<int32_t>(cpu::native_quantised::preparedInterleavedBlockStride(Encoding::IQ3SExpandedSourceScale, false)) &&
               header.has_native_blocks == 0;
    if (header.prepared_encoding == static_cast<uint8_t>(Encoding::NibbleLUTSourceScale))
        return header.codebook_id == 4 && header.is_superblock == 1 &&
               header.is_asymmetric == 0 && header.payload_bytes == 16 &&
               header.data_stride == static_cast<int32_t>(cpu::native_quantised::preparedDataStride(
                   Encoding::NibbleLUTSourceScale)) &&
               header.interleaved_block_stride == static_cast<int32_t>(
                   cpu::native_quantised::preparedInterleavedBlockStride(Encoding::NibbleLUTSourceScale, false)) &&
               header.has_native_blocks == 0;
    return !(header.codebook_id == 4 && header.is_superblock != 0 &&
             header.prepared_encoding == static_cast<uint8_t>(Encoding::NibbleLUT));
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
 * @param weights Prepared CPU NativeQuantised weights.
 * @param descriptor Self-describing scalar metadata and section lengths.
 * @param views Native read-only section addresses.
 * @return `true` when the concrete format is directly transferable.
 */
inline bool describeTransfer(
    const cpu::native_quantised::CPUNativeQuantisedPackedWeights &packed,
    PackedWeightsTransferDescriptor &descriptor,
    PackedWeightsConstSectionViews &views,
    std::span<const uint8_t> native_blocks = {},
    size_t native_block_size = 0)
{
    descriptor = {};
    views = {};
    auto &header = descriptor.header;
    header.magic = PACKED_WEIGHTS_MAGIC;
    header.version = PACKED_WEIGHTS_VERSION;
    header.format =
        static_cast<uint32_t>(PackedWeightsFormat::CPU_NATIVE_QUANTISED);
    header.N = packed.N;
    header.K = packed.K;
    header.N_padded = packed.N_padded;
    header.blocks_per_row = packed.blocks_per_row;
    header.codebook_id = packed.codebook_id;
    header.payload_bytes = static_cast<uint8_t>(packed.payload_bytes);
    header.is_asymmetric = packed.is_asymmetric ? 1 : 0;
    header.prepared_encoding = static_cast<uint8_t>(packed.encoding);
    header.is_superblock = packed.is_superblock ? 1 : 0;
    header.has_native_blocks = !native_blocks.empty() ? 1 : 0;
    header.data_stride = packed.data_stride;
    header.interleaved_block_stride = packed.interleaved_block_stride;
    header.native_block_size =
        static_cast<int32_t>(native_block_size);
    // The sealed layout identity: receivers re-derive both strides from
    // this byte's section tables before adopting any payload section.
    header.prepared_layout = static_cast<uint8_t>(packed.layout);
    std::memset(header.reserved, 0, sizeof(header.reserved));

    auto &sections = descriptor.sections;
    sections.interleaved_size = packed.native_interleaved.size();
    sections.payload_size = packed.payload.size();
    sections.int8_flat_size = packed.int8_flat.size();
    sections.native_blocks_size = native_blocks.size();

    views.data = {
        packed.native_interleaved.data(),
        packed.payload.data(),
        reinterpret_cast<const uint8_t *>(packed.int8_flat.data()),
        native_blocks.data(),
    };
    views.sizes = {
        static_cast<size_t>(sections.interleaved_size),
        static_cast<size_t>(sections.payload_size),
        static_cast<size_t>(sections.int8_flat_size),
        static_cast<size_t>(sections.native_blocks_size),
    };
    return true;
}

/** @brief Describe an owning packed object while borrowing its immutable sections. */
inline bool describeTransfer(
    const IPackedWeights &weights,
    PackedWeightsTransferDescriptor &descriptor,
    PackedWeightsConstSectionViews &views)
{
    descriptor = {};
    views = {};
    if (weights.format() != PackedWeightsFormat::CPU_NATIVE_QUANTISED)
        return false;
    const auto *cpu = dynamic_cast<const cpu::native_quantised::CPUPackedWeights *>(&weights);
    if (!cpu)
        return false;
    const auto *native = dynamic_cast<const cpu::native_quantised::CPUPackedWeightsWithNativeBlocks *>(&weights);
    return describeTransfer(cpu->packed(), descriptor, views,
        native ? std::span<const uint8_t>(native->nativeBlocks().data(), native->nativeBlocks().size()) : std::span<const uint8_t>{},
        native ? native->nativeBlockSize() : 0);
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
    const PackedWeightsTransferDescriptor &descriptor,
    std::optional<CPUWeightStoragePlacement> placement = std::nullopt)
{
    const auto &header = descriptor.header;
    const auto &sections = descriptor.sections;
    if (header.magic != PACKED_WEIGHTS_MAGIC ||
        !validNativeSourceScaleTupleHeader(header) ||
        header.format !=
            static_cast<uint32_t>(PackedWeightsFormat::CPU_NATIVE_QUANTISED))
    {
        throw std::invalid_argument(
            "Direct packed-weight transfer received invalid format metadata");
    }
    // Versioned layout admission: v8 decodes the sealed byte, exactly v7 is
    // explicitly canonical, anything earlier rejects globally. The decoded
    // layout must reproduce the header's strides from its section tables
    // before any receive buffer is adopted.
    const auto admitted_layout = admittedPreparedLayout(header);
    requirePreparedLayoutStrides(header, admitted_layout);
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

    cpu::native_quantised::CPUNativeQuantisedPackedWeights packed;
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
    packed.layout = admitted_layout;
    packed.workspace_data_ = nullptr;

    if (placement)
    {
        packed.native_interleaved.replace(placement->allocate<uint8_t>(sections.interleaved_size));
        packed.payload = placement->allocate<uint8_t>(sections.payload_size);
        packed.int8_flat = placement->allocate<int8_t>(sections.int8_flat_size);
    }
    else
    {
        {
            AlignedVector<uint8_t> interleaved;
            resizeDirectReceiveSection(interleaved, sections.interleaved_size);
            packed.native_interleaved.replace(std::move(interleaved));
        }
        resizeDirectReceiveSection(packed.payload, sections.payload_size);
        resizeDirectReceiveSection(packed.int8_flat, sections.int8_flat_size);
    }

    PackedWeightsReceiveTarget target;
    // The receive writes this generation; attachment admits it after completion.
    target.data[0] = packed.native_interleaved.beginRewrite().data();
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
        auto native_blocks = placement.value_or(CPUWeightStoragePlacement::local()).allocate<uint8_t>(sections.native_blocks_size);
        target.data[3] = native_blocks.data();
        target.weights = std::make_unique<
            cpu::native_quantised::CPUPackedWeightsWithNativeBlocks>(
            std::move(packed),
            std::move(native_blocks),
            static_cast<size_t>(header.native_block_size));
    }
    else
    {
        target.weights =
            std::make_unique<cpu::native_quantised::CPUPackedWeights>(
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
 * @tparam WeightView Borrowed native packed storage or owning packed object.
 * @tparam ByteBuffer Contiguous byte container with `data()`, `clear()`, and
 *         either `resize()` or `resize_uninitialized()`.
 * @param weights Prepared CPU NativeQuantised weights.
 * @param buffer Destination buffer, replaced with one complete wire record.
 * @param placement Optional admitted final wire allocation; retained by aligned storage.
 * @return `true` when the format was serialized completely.
 */
template <typename WeightView, typename ByteBuffer>
inline bool serializeInto(const WeightView& weights, ByteBuffer& buffer,
    std::optional<CPUWeightStoragePlacement> placement = std::nullopt)
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
        if (placement)
            buffer = placement->template allocate<uint8_t>(total);
        else
            buffer.resize_uninitialized(total);
    }
    else
    {
        if (placement)
            throw std::invalid_argument("Accounted serialization requires allocation-owning aligned storage");
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

    cpu::native_quantised::CPUNativeQuantisedLayout admitted_layout =
        cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical;
    try
    {
        // Exactly v8 decodes the sealed layout byte; exactly v7 is admitted
        // as an explicitly canonical archive; earlier versions reject
        // globally, never per-format.
        admitted_layout = admittedPreparedLayout(header);
        requirePreparedLayoutStrides(header, admitted_layout);
    }
    catch (const std::invalid_argument &error)
    {
        LOG_WARN("[PackedWeightsSerialization] " << error.what());
        return nullptr;
    }

    if (header.format != static_cast<uint32_t>(PackedWeightsFormat::CPU_NATIVE_QUANTISED))
    {
        LOG_WARN("[PackedWeightsSerialization] Unsupported format: " << header.format);
        return nullptr;
    }

    if (!validNativeSourceScaleTupleHeader(header))
        return nullptr;

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

    // Reconstruct CPUNativeQuantisedPackedWeights from header + data sections.
    cpu::native_quantised::CPUNativeQuantisedPackedWeights packed;
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
    packed.layout                   = admitted_layout;
    packed.workspace_data_          = nullptr;

    const uint8_t* src = data + MIN_SIZE;

    // Read native_interleaved (64-byte aligned).
    if (sections.interleaved_size > 0)
    {
        // Establish the final CPU destination before copying the eager wire
        // representation. Never migrate or repack it after engine publication.
        packed.native_interleaved.replace(placement.allocate<uint8_t>(sections.interleaved_size));
        std::memcpy(packed.native_interleaved.beginRewrite().data(), src, sections.interleaved_size);
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
        auto native_blocks = placement.allocate<uint8_t>(sections.native_blocks_size);
        std::memcpy(native_blocks.data(), src, sections.native_blocks_size);

        return std::make_unique<cpu::native_quantised::CPUPackedWeightsWithNativeBlocks>(
            std::move(packed),
            std::move(native_blocks),
            static_cast<size_t>(header.native_block_size));
    }

    return std::make_unique<cpu::native_quantised::CPUPackedWeights>(std::move(packed));
}

} // namespace packed_weights_serialization
} // namespace llaminar2
