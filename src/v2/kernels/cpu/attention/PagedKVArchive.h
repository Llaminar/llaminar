/**
 * @file PagedKVArchive.h
 * @brief Typed immutable archive section for pooled FP32 page sequences.
 *
 * The archive is a portable, neutral container of native payload: it carries
 * one immutable page sequence plus exact format/geometry provenance. It never
 * owns physical frames; import materializes into caller private pages and
 * verifies provenance before copying any payload byte. A truncated archive has
 * a shorter live row count while its payload pages keep their exact bytes.
 */
#pragma once
#include "kernels/cpu/attention/CPUFP32PagedKVView.h"
#include "kernels/kvcache/KVStorageLayout.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief Exact immutable page-sequence provenance; import rejects mismatches. */
struct PagedKVArchiveMeta
{
    KVCacheFamily family = KVCacheFamily::AttentionOnly;
    KVStorageFormat format = KVStorageFormat::FP32;
    int heads = 0;
    int dimension = 0;
    int page_tokens = 0;
    std::size_t rows = 0;
    std::uint64_t terminal_boundary = 0;   ///< Coherent terminal token boundary of the section.
    std::uint64_t trust_domain = 0;        ///< Authenticated isolation domain, never caller-chosen.
    std::uint64_t ancestry_id = 0;         ///< Exact ancestry proof identity (collision-resolving).
    bool operator==(const PagedKVArchiveMeta &) const = default;
};

/** @brief One typed immutable archive section; payload is copied only on import. */
class PagedKVArchive final
{
public:
    /** @brief Build from an exact view; payload is copied out of retained pages once. */
    static PagedKVArchive fromView(const CPUFP32PagedKVView &view);
    /** @brief Parse a previously serialized section; verifies geometry before copy. */
    static PagedKVArchive parse(std::span<const std::byte> bytes);
    /** @brief Serialize with a fixed layout header, provenance and exact native payload. */
    std::vector<std::byte> serialize() const;
    /** @return Authenticated meta; empty section meta has rows == 0. */
    const PagedKVArchiveMeta &meta() const noexcept { return meta_; }
    /** @return Exact native payload bytes (K/V interleaved, no padding). */
    std::span<const std::byte> payload() const noexcept { return payload_; }
    bool empty() const noexcept { return payload_.empty(); }

    /**
     * @brief Rewrite the authenticated provenance record of this section.
     *
     * The payload is unchanged (immutable bytes); the meta record is stamped
     * with the caller's authenticated trust domain and ancestry identity
     * (the ROUTE's values — never the archive's own). Used by the prefix
     * flow: the WRITE side stamps at store.put time so the RESTORE side can
     * pass the SAME values as the primitive's expectation parameters (the
     * table-level authenticated attach), making the trust/ancestry check
     * REAL end-to-end instead of tautological.
     * @throws std::invalid_argument For an empty section (no provenance to
     * stamp) or a zero trust domain (the authenticated domain is never 0).
     */
    /**
     * @brief Copy one exact row interval of this section (tail slice).
     *
     * Immutable payload, interval-clipped meta (rows + terminal boundary
     * adjusted to the interval; heads/dimension/page_tokens unchanged). The
     * interval must be row-aligned within [0, rows]; page-token alignment is
     * NOT required (a tail fragment may end mid-page). Used by the prefix
     * flow to demote a per-layer tail fragment as its own extent.
     * @throws std::invalid_argument For an empty section, a bad interval or a
     * terminal boundary outside the new extent.
     */
    PagedKVArchive slice(std::size_t rows_from, std::size_t rows_to) const
    {
        if (empty() || rows_from >= rows_to || rows_to > meta_.rows)
            throw std::invalid_argument("Paged archive slice requires a nonempty interval within the section");
        const auto layout = KVNativeStorageLayout::resolve(meta_.family,
            meta_.heads, meta_.dimension, KVStorageFormatName(meta_.format),
            DeviceId::cpu());
        const auto stride = layout.pagePayload(1).totalBytes();
        if (rows_to > std::numeric_limits<std::size_t>::max() / stride)
            throw std::invalid_argument("Paged archive slice geometry overflows");
        const auto offset = rows_from * stride;
        const auto length = (rows_to - rows_from) * stride;
        if (offset + length > payload_.size())
            throw std::invalid_argument("Paged archive slice payload interval is out of bounds");
        auto meta = meta_;
        meta.rows = rows_to - rows_from;
        meta.terminal_boundary = meta.terminal_boundary > rows_from
            ? std::min(meta.terminal_boundary, rows_to) - rows_from
            : 0;
        std::vector<std::byte> clipped(payload_.begin() + static_cast<std::ptrdiff_t>(offset),
            payload_.begin() + static_cast<std::ptrdiff_t>(offset + length));
        return PagedKVArchive(meta, std::move(clipped));
    }

    /**
     * @brief Concatenate two row intervals of the SAME section.
     *
     * `this` is the HEAD interval [0, first.rows), `tail` the continuation
     * ([first.rows, first.rows+tail.rows)) with IDENTICAL geometry and
     * provenance record (the primitive refuses incompatible
     * heads/dimension/page_tokens/family/format/trust/ancestry). The result
     * is one deterministic section of the merged rows (used by the prefix
     * flow's restore: resident head slice + demoted tail slice -> one
     * full-prefix section before import).
     */
    PagedKVArchive concat(const PagedKVArchive &tail) const
    {
        if (empty() || tail.empty())
            throw std::invalid_argument("Paged archive concat requires two nonempty sections");
        if (meta_.family != tail.meta_.family || meta_.format != tail.meta_.format ||
            meta_.heads != tail.meta_.heads || meta_.dimension != tail.meta_.dimension ||
            meta_.page_tokens != tail.meta_.page_tokens ||
            meta_.trust_domain != tail.meta_.trust_domain ||
            meta_.ancestry_id != tail.meta_.ancestry_id)
            throw std::invalid_argument("Paged archive concat requires identical geometry and provenance records");
        const auto layout = KVNativeStorageLayout::resolve(meta_.family, meta_.heads,
            meta_.dimension, KVStorageFormatName(meta_.format), DeviceId::cpu());
        const auto stride = layout.pagePayload(1).totalBytes();
        if (meta_.rows > std::numeric_limits<std::size_t>::max() - tail.meta_.rows ||
            (meta_.rows + tail.meta_.rows) > std::numeric_limits<std::size_t>::max() / stride)
            throw std::invalid_argument("Paged archive concat geometry overflows");
        auto meta = meta_;
        meta.rows = meta_.rows + tail.meta_.rows;
        meta.terminal_boundary = meta_.terminal_boundary >= meta_.rows
            ? meta_.terminal_boundary + tail.meta_.rows
            : meta_.terminal_boundary;
        std::vector<std::byte> merged;
        merged.reserve(payload_.size() + tail.payload_.size());
        merged.insert(merged.end(), payload_.begin(), payload_.end());
        merged.insert(merged.end(), tail.payload_.begin(), tail.payload_.end());
        return PagedKVArchive(meta, std::move(merged));
    }

    PagedKVArchive withProvenance(std::uint64_t trust_domain,
        std::uint64_t ancestry_id) const
    {
        if (empty() || trust_domain == 0)
            throw std::invalid_argument("Provenance stamping requires a nonempty section and a nonzero trust domain");
        auto meta = meta_;
        meta.trust_domain = trust_domain;
        meta.ancestry_id = ancestry_id;
        return PagedKVArchive(meta, payload_);
    }

    /**
     * @brief Materialize this archive into exact private pages under one grant.
     * @param grant Caller request with protected page growth for every page.
     * @param page_tokens Native logical block size for page slicing.
     * @return One committed immutable page per logical block, exact live rows.
     * @throws std::invalid_argument If the archive is empty, its provenance is
     * not FP32 attention, or the grant has insufficient protected growth.
     * @note This is private materialization; it never restores another request's
     * sampler/history/placement and does not adopt any shared referent.
     */
    std::vector<CPUFP32KVPage> materialize(PhysicalPagePool::Grant &grant, int page_tokens) const;
    /**
     * @brief Exact number of logical pages this archive occupies at @p page_tokens.
     * @throws std::invalid_argument for empty sections or nonpositive geometry.
     * @note Geometry/overflow checks match materialize exactly; callers use
     * this before any mutation so table cells and grant growth can be
     * validated atomically up front.
     */
    std::size_t pageCount(int page_tokens) const;
private:
    PagedKVArchive(PagedKVArchiveMeta meta, std::vector<std::byte> payload)
        : meta_(meta), payload_(std::move(payload)) {}
    PagedKVArchiveMeta meta_;
    std::vector<std::byte> payload_;
};
} // namespace llaminar2
