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
private:
    PagedKVArchive(PagedKVArchiveMeta meta, std::vector<std::byte> payload)
        : meta_(meta), payload_(std::move(payload)) {}
    PagedKVArchiveMeta meta_;
    std::vector<std::byte> payload_;
};
} // namespace llaminar2
