/**
 * @file ImmutableRowTable.h
 * @brief Native quantized row storage with explicit RAM and bounded SSD mappings.
 *
 * This is a CPU storage service, not an inference-state owner. Callers supply
 * row IDs and destination storage; the table never sees tokens, sampler state,
 * or an accelerator stream. GPU graphs reach it only through an explicit
 * heterogeneous storage boundary. Quantized rows are copied verbatim. Physical
 * backing is admitted and leased through PhysicalMemoryAuthority; file extent
 * and virtual mappings are reported separately from that physical budget.
 * Exact-node RAM is certified on its explicitly selected NUMA node before source reads.
 * Task-policy RAM follows inherited Linux policy without an exact-node promise. Read-only
 * shared file mappings use Linux page-cache placement; their associated host
 * resource is not a guarantee of physical locality for shared cache pages.
 */
#pragma once

#include "ModelLoader.h"
#include "ImmutableRowTablePolicy.h"
#include "ImmutableRowTableHostPlacement.h"
#include "HostImmutableRowOwnerObservation.h"
#include "../planning/PhysicalMemoryAuthority.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace llaminar2
{
/** @brief Native tensor extent inside one immutable, already downloaded shard. */
struct ImmutableRowTableSource
{
    std::string path; ///< Durable source, opened once for the table lifetime.
    std::string tensor_name; ///< Exact directory identity, independent of its file label.
    std::uint64_t file_offset = 0; ///< Absolute first byte of the tensor.
    std::uint64_t rows = 0; ///< Number of independently addressable rows.
    std::uint64_t row_elements = 0; ///< Logical columns, including complete native blocks.
    GGUFTensorType format = GGUFTensorType::F32; ///< Actual GGUF tensor format.

    /** @return Exact native bytes per row; rejects incomplete blocks and overflow. */
    [[nodiscard]] std::size_t rowBytes() const;
    /** @return Exact file extent, with checked multiplication. */
    [[nodiscard]] std::size_t tensorBytes() const;
    bool operator==(const ImmutableRowTableSource &) const = default;
};

/** @brief File metadata identity; no table-wide checksum or payload hash is needed. */
struct ImmutableRowFileIdentity
{
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    std::uint64_t file_bytes = 0;
    std::int64_t modified_seconds = 0;
    std::int64_t modified_nanoseconds = 0;
    std::int64_t changed_seconds = 0;
    std::int64_t changed_nanoseconds = 0;
    bool operator==(const ImmutableRowFileIdentity &) const = default;
};

/**
 * @brief Window-resolution work of one successfully completed native gather.
 *
 * The table produces this value while holding its existing gather mutex. It
 * never includes another client's work. Hits/misses describe retained mapping
 * windows, not OS page-cache hits or physical SSD reads. A thrown partial gather
 * returns no successful observation; stats() still retains any cache work it
 * performed before failure. RAM and empty gathers report zero window work.
 */
struct ImmutableRowGatherObservation
{
    std::uint64_t windowHits = 0;
    std::uint64_t windowMisses = 0;
    std::uint64_t windowEvictions = 0;
};

/** @brief One selected source row and its position in a caller-owned native response; targets must be strictly ordered. */
struct ImmutableRowTableScatterRow final
{
    std::int64_t source_row;
    std::size_t response_row;
};

/** @brief Coherent storage-service counters and explicit memory/file extents. */
struct ImmutableRowTableStats
{
    ImmutableRowStorage storage = ImmutableRowStorage::RAM;
    std::size_t table_file_bytes = 0; ///< Durable tensor extent, not RAM demand.
    std::size_t physical_budget_bytes = 0; ///< Sole admitted resident/cache envelope.
    std::size_t mapped_virtual_bytes = 0; ///< Actual page-rounded virtual extent of retained file mappings; not resident/physical bytes.
    std::uint64_t requests = 0;
    std::uint64_t rows = 0;
    std::uint64_t copied_bytes = 0; ///< Requested native data, excluding page amplification.
    std::uint64_t source_metadata_checks = 0; ///< Completed fstat/stat authentication pairs, including detected identity changes.
    std::uint64_t window_hits = 0;
    std::uint64_t window_misses = 0;
    std::uint64_t window_evictions = 0;
};

/**
 * @brief Immutable native rows backed by RAM or a bounded LRU of SSD mappings.
 *
 * Source files must remain immutable for the lifetime of an open table. Every
 * gather authenticates the open descriptor and its path metadata before reading;
 * mutation, truncation and replacement fail explicitly. Independent clients are
 * serialized at this CPU service boundary so mappings cannot be evicted during
 * a copy. Neither a row lookup nor an SSD miss allocates accelerator memory.
 */
class ImmutableRowTable final
{
public:
    /**
     * @brief Return the exact RAM or mapping-cache demand for the canonical BOM.
     * @param source Native tensor coordinates.
     * @param policy Explicit storage choice and bounded mapping geometry.
     * @return Physical bytes to charge to ImmutableTableStorage on the CPU resource.
     */
    [[nodiscard]] static std::size_t physicalBytes(
        const ImmutableRowTableSource &source, const ImmutableRowTableStoragePolicy &policy);

    /** @brief Price an already validated exact native extent for metadata-only BOM assembly. */
    [[nodiscard]] static std::size_t physicalBytesForNativeExtent(
        std::size_t native_bytes, const ImmutableRowTableStoragePolicy &policy);

    /** @return Exact live CPU storage claim for retained-model attestation; never recomputed. */
    [[nodiscard]] std::size_t physicalBytes() const noexcept;
    /**
     * @brief Authenticate ready storage and observe its actual retained PMA lease.
     * @return Exact host/locality/PMA identity and live table bytes; staging is zero.
     * @throws std::exception For a stale source or missing/foreign live owner lease.
     * @note This is a point-in-time observation, not a storage pin or allocation permit.
     */
    [[nodiscard]] HostImmutableRowOwnerObservation retainedOwnerObservation() const;
    /** @brief Authenticate the retained source at an explicit setup/reuse boundary without reading payload. */
    void authenticateSource() const;


    /**
     * @brief Open and authenticate a source, then materialize its admitted storage.
     * @param source Native tensor coordinates; never requantized.
     * @param policy User-selected storage and cache geometry.
     * @param host_device Exact admitted CPU allocator; its ordinal is not a NUMA node.
     * @param host_placement Sealed RAM locality, independent of the PMA allocation identity.
     * @param memory Shared canonical authority with an admitted CPU owner line.
     * @throws std::invalid_argument For invalid geometry or missing authority.
     * @throws std::runtime_error For missing/changed files, rejected admission or I/O.
     */
    ImmutableRowTable(ImmutableRowTableSource source,
                      ImmutableRowTableStoragePolicy policy,
                      DeviceId host_device,
                      std::shared_ptr<PhysicalMemoryAuthority> memory,
                      ImmutableRowTableHostPlacement host_placement);
    /** @brief Retire all mappings and RAM before releasing their physical lease. */
    ~ImmutableRowTable();
    ImmutableRowTable(const ImmutableRowTable &) = delete;
    ImmutableRowTable &operator=(const ImmutableRowTable &) = delete;

    /**
     * @brief Copy exactly the requested native rows in caller order.
     * @param ids Execution-authenticated row IDs; duplicates remain duplicated.
     * @param destination Exactly ids.size()*rowBytes() caller-owned bytes, disjoint from IDs.
     * @return Only this successful call's mapping-window work, authored under
     *         the same mutex as its copies and lifetime table counters.
     * @throws std::out_of_range Before copying anything if an ID is invalid.
     * @throws std::invalid_argument For invalid byte/address extents or ID/response overlap,
     *         before output, source authentication or table observations change.
     * @throws std::runtime_error If source metadata changed or a mapping fails;
     *         partial cache/copy work remains visible only in table stats and
     *         must not be claimed as successful service work.
     */
    ImmutableRowGatherObservation gather(std::span<const std::int64_t> ids, std::span<std::byte> destination);
    /**
     * @brief Copy selected native rows directly to strictly increasing caller response positions.
     * @param rows Compact selected source/response rows, validated completely before any copy.
     * @param destination Complete caller response, with an exact native-row extent; unselected holes stay unchanged.
     * @return Mapping work of this successful batch only; bytes/counters count selected rows, never response holes.
     * @throws std::exception For invalid/overlapping/aliased metadata, changed source or native I/O failure.
     * @note One exact source transaction authenticates metadata before and after copies, independent of target count.
     */
    ImmutableRowGatherObservation gatherScattered(std::span<const ImmutableRowTableScatterRow> rows,
        std::span<std::byte> destination);

    /** @return Source descriptor retained for type-correct row decoding. */
    [[nodiscard]] const ImmutableRowTableSource &source() const noexcept;
    /** @return Actual immutable RAM/mmap choice admitted for this shared table. */
    [[nodiscard]] ImmutableRowStorage storage() const noexcept;
    /** @return Sealed RAM locality; mmap page-cache locality remains Linux-owned. */
    [[nodiscard]] ImmutableRowTableHostPlacement hostPlacement() const noexcept;
    /** @return Admitted CPU allocator; affinity is carried separately. */
    [[nodiscard]] DeviceId hostDevice() const noexcept;
    /** @return Immutable file identity authenticated when the table was opened. */
    [[nodiscard]] const ImmutableRowFileIdentity &identity() const noexcept;
    /** @return Coherent snapshot; does not walk mappings, fault pages, or scan the table. */
    [[nodiscard]] ImmutableRowTableStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace llaminar2
