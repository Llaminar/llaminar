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
private:
    PagedKVArchive(PagedKVArchiveMeta meta, std::vector<std::byte> payload)
        : meta_(meta), payload_(std::move(payload)) {}
    PagedKVArchiveMeta meta_;
    std::vector<std::byte> payload_;
};
} // namespace llaminar2
