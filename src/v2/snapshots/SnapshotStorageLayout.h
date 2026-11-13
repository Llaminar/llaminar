/**
 * @file SnapshotStorageLayout.h
 * @brief Checked payload, authored metadata and absolute arena-slot geometry.
 *
 * Relative metadata alignment alone is insufficient: an odd legacy FP32 payload
 * may leave the next slot at 4 modulo 8. Authored observations therefore require
 * an eight-byte slot base as well as their eight-byte relative metadata offset.
 * This pure geometry owns no memory, live count, device pointer or reservation;
 * executor binding and typed BOM contributors can reuse the same byte contract.
 */
#pragma once
#include "kernels/common/DiagnosticSnapshotCopy.h"
#include <limits>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
/** @brief Immutable maximum source shape, without pointers or count contents. */
struct SnapshotStorageShape
{
    std::size_t native_payload_bytes;
    std::optional<DiagnosticSnapshotCopyGeometry> authored_geometry;
};
class SnapshotStorageLayout;
/** @brief Construct the checked immutable storage layout for one actual output descriptor. */
[[nodiscard]] inline SnapshotStorageLayout snapshotStorageLayout(SnapshotStorageShape shape);

/** @brief One checked output's physical geometry; aliases do not create additional outputs. */
class SnapshotStorageLayout final
{
public:
    const std::size_t payload_storage_bytes; ///< Native payload rounded to existing FP32 arena units.
    const std::size_t metadata_offset_bytes; ///< Relative aligned metadata offset; zero for legacy outputs.
    const std::size_t metadata_bytes; ///< Canonical diagnostic metadata capacity, not live transfer bytes.
    const std::size_t storage_bytes; ///< Complete payload, internal padding and metadata capacity.
    const std::size_t required_alignment; ///< Four-byte legacy or eight-byte authored slot base.
private:
    friend SnapshotStorageLayout snapshotStorageLayout(SnapshotStorageShape shape);
    /** @brief Seal only geometry already checked by the single public factory. */
    SnapshotStorageLayout(std::size_t payload, std::size_t offset, std::size_t metadata,
        std::size_t total, std::size_t alignment)
        : payload_storage_bytes(payload), metadata_offset_bytes(offset), metadata_bytes(metadata),
          storage_bytes(total), required_alignment(alignment) {}
};

namespace snapshot_storage_detail
{
/** @return Representable sum; invalid geometry cannot wrap into a smaller allocation. */
inline std::size_t add(std::size_t left, std::size_t right)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
        throw std::overflow_error("Snapshot storage extent overflows size_t");
    return left + right;
}
/** @return Checked power-of-two alignment without overflowing an already aligned maximum. */
inline std::size_t align(std::size_t bytes, std::size_t alignment)
{
    const auto remainder = bytes % alignment;
    return remainder ? add(bytes, alignment - remainder) : bytes;
}
}

/**
 * @brief Price one concrete output descriptor using its frozen authored geometry.
 * @param shape Positive native payload and optional exact native copy geometry.
 * @return Immutable relative layout and required absolute slot-base alignment.
 * @throws std::invalid_argument For empty/invalid geometry or an undersized native payload.
 * @throws std::overflow_error If payload rounding, metadata alignment or total extent overflows.
 *
 * Metadata size remains diagnosticSnapshotMetadataCapacity's authority. No live
 * value is read, no semantic filename is counted and no allocation is requested.
 */
inline SnapshotStorageLayout snapshotStorageLayout(SnapshotStorageShape shape)
{
    if (!shape.native_payload_bytes)
        throw std::invalid_argument("Snapshot storage requires a positive native payload capacity");
    const auto payload = snapshot_storage_detail::align(shape.native_payload_bytes, alignof(float));
    if (!shape.authored_geometry)
        return {payload, 0, 0, payload, alignof(float)};
    const auto geometry = *shape.authored_geometry;
    const auto metadata = diagnosticSnapshotMetadataCapacity(geometry);
    // The canonical metadata validator already proved this product positive
    // and representable. A raw pointer copy cannot discover an undersized slot.
    const auto native_capacity = static_cast<std::size_t>(geometry.rows) *
        static_cast<std::size_t>(geometry.columns) * static_cast<std::size_t>(geometry.elementBytes);
    if (shape.native_payload_bytes < native_capacity)
        throw std::invalid_argument("Snapshot storage payload is smaller than its authored native geometry");
    const auto offset = snapshot_storage_detail::align(payload, alignof(SnapshotObservedHeader));
    return {payload, offset, metadata, snapshot_storage_detail::add(offset, metadata),
        alignof(SnapshotObservedHeader)};
}

/** @brief One immutable output slot placed after preceding outputs in graph execution order. */
struct SnapshotStorageSlot
{
    std::size_t offset_bytes;
    std::size_t end_bytes;
};
/**
 * @brief Place a checked slot with its actual required absolute alignment.
 * @param previous_end Complete end of the preceding actual producer output.
 * @param layout Immutable checked layout, independent of alias names/tensor identity.
 * @return Aligned slot base and checked complete end for the next output.
 * @throws std::overflow_error If leading alignment or the complete arena end overflows.
 *
 * Every temporal producer remains a distinct call, even if it reuses a tensor.
 * Multiple semantic keys of one output descriptor are still a single call.
 */
[[nodiscard]] inline SnapshotStorageSlot placeSnapshotStorageSlot(
    std::size_t previous_end, const SnapshotStorageLayout &layout)
{
    const auto offset = snapshot_storage_detail::align(previous_end, layout.required_alignment);
    return {offset, snapshot_storage_detail::add(offset, layout.storage_bytes)};
}
} // namespace llaminar2
