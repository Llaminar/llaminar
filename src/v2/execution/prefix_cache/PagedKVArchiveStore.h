/**
 * @file PagedKVArchiveStore.h
 * @brief Durable finite-quota disk store for one immutable pooled KV archive section.
 *
 * Plan P3 ("Give disk extents one finite owner, including rounded quota,
 * metadata, pending/old replacement extents, durable completion and
 * namespace-safe cleanup"): a section is ONE extent with an exact identity
 * (namespace + generation + content digest + rounded bytes). The store
 * admits the extent in the route's finite quota BEFORE any byte reaches the
 * filesystem; a quota-full or impossible extent refuses without creating a
 * file. Publication is durable (write temp in the SAME directory -> fsync ->
 * rename -> directory fsync); a failed or cancelled write removes the temp
 * and releases the quota admission. Reads re-parse the section and verify the
 * provenance AND the content digest before serving a single payload byte
 * (partial/corrupt archives refuse). Retirement removes exactly the
 * advertised extent and releases its quota; replacement extents (old + new)
 * are BOTH admitted until the old one is retired, so the quota counts the
 * real on-disk set. Cleanup removes ONLY engine-named temp files; a foreign
 * file in the configured directory is never touched.
 *
 * The store never derives quota, never probes free disk space and never
 * invents a default directory: the route (validated by validatePagedKVDiskRoute)
 * owns quota and directory, and the store verifies both at construction.
 */
#pragma once

#include "execution/prefix_cache/DiskExtentQuota.h"
#include "execution/prefix_cache/PagedKVDiskRoute.h"
#include "kernels/cpu/attention/PagedKVArchive.h"

#include <cstdint>
#include <optional>
#include <string>

namespace llaminar2
{
/** @brief One typed disk extent identity (generation, namespace, digest, bytes). */
struct PagedKVArchiveExtentKey
{
    std::uint64_t generation = 0;
    std::uint64_t namespace_id = 0;
    std::uint64_t digest = 0;
    std::size_t bytes = 0;
    bool operator==(const PagedKVArchiveExtentKey &) const = default;
};

/**
 * @brief Durable section store bound to one validated route.
 *
 * Move-only; the store holds the route's quota ledger (the finite owner of
 * every extent it admits). Construction verifies the route, the quota and
 * the configured directory (must exist and be a directory; a wrong path is
 * refused, never substituted).
 */
class PagedKVArchiveStore final
{
public:
    /** @brief Open the store for a validated route. */
    explicit PagedKVArchiveStore(const PagedKVDiskRoute &route);
    PagedKVArchiveStore(PagedKVArchiveStore &&) noexcept;
    PagedKVArchiveStore &operator=(PagedKVArchiveStore &&) noexcept;
    PagedKVArchiveStore(const PagedKVArchiveStore &) = delete;
    PagedKVArchiveStore &operator=(const PagedKVArchiveStore &) = delete;
    ~PagedKVArchiveStore();

    /**
     * @brief Durably publish one section under an exact extent identity.
     * @param archive The immutable section (provenance must be FP32 attention).
     * @param generation Monotonic replacement generation for this namespace.
     * @param namespace_id Engine-chosen namespace (content partition).
     * @return The admitted extent key (digest + rounded bytes), or empty with
     * a precise refusal when the quota is too small, the directory is not
     * writable, the write fails, or the section is not storable. A refused
     * put NEVER leaves a file behind and NEVER leaves an outstanding quota
     * admission.
     */
    std::optional<PagedKVArchiveExtentKey> put(
        const PagedKVArchive &archive, std::uint64_t generation,
        std::uint64_t namespace_id, std::string *refusal = nullptr);

    /**
     * @brief Read one extent back into a verified archive.
     * @param key Exact extent identity (digest and bytes are verified).
     * @return The re-parsed archive (provenance checked before payload
     * access), or empty when the file is missing, truncated, corrupt,
     * digest-mismatched or the identity was never admitted (fail closed).
     */
    std::optional<PagedKVArchive> get(const PagedKVArchiveExtentKey &key,
        std::string *refusal = nullptr) const;

    /**
     * @brief Retire one exact admitted extent: unlink + quota release.
     * @return True when the extent existed and was removed; false when the
     * identity was not admitted or the file was already gone (no mutation).
     */
    bool retire(const PagedKVArchiveExtentKey &key);

    /**
     * @brief Engine-only cleanup of orphaned temp files.
     *
     * Removes ONLY files matching the engine's temp pattern in the configured
     * directory (crash-orphaned `.<strategy>.tmp-*` from failed puts). A
     * foreign file is never removed; a named extent is never removed here
     * (retirement owns extents). The configured directory itself is never
     * touched.
     * @return Number of temp files removed.
     */
    std::size_t cleanOrphanedTemps() const;

    /** @return The admitted finite quota ledger (exact ownership view). */
    const DiskExtentQuota &quota() const noexcept { return quota_; }

    /** @return The configured engine-owned directory (never defaulted). */
    const std::string &directory() const noexcept { return directory_; }

private:
    std::string pathFor(const PagedKVArchiveExtentKey &key) const;
    DiskExtentQuota quota_;
    std::string directory_;
};
} // namespace llaminar2
