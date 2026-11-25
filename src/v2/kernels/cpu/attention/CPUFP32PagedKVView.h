/**
 * @file CPUFP32PagedKVView.h
 * @brief Direct FP32 page operands for the first native pooled-attention witness.
 *
 * Each immutable page retains its PMA byte publication. Logical token order is
 * given by the borrowed page table, never a physical frame stride. A token
 * stores its complete local K row followed by its local V row; page padding is
 * not readable payload. No contiguous history is allocated or reconstructed.
 * This private CPU operand seam is not a common GPU/quantized ABI or a serving
 * capability. Callers must retain the page table through the native operation.
 */
#pragma once
#include "planning/PhysicalPagePool.h"
#include "kernels/kvcache/KVStorageLayout.h"
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
/** @brief One immutable native FP32 K/V page and its exact live token extent. */
class CPUFP32KVPage final
{
public:
    /**
     * @brief Adopt completed bytes as native FP32 token-major K/V operands.
     * @param publication Exact retained PMA content/version, not an arbitrary pointer.
     * @param heads Actual local KV head count.
     * @param dimension Coordinates per head.
     * @throws std::invalid_argument For padding/non-token extents or misalignment.
     */
    CPUFP32KVPage(PhysicalPagePool::CommittedPage publication, int heads, int dimension)
        : publication_(std::move(publication)), heads_(heads), dimension_(dimension)
    {
        const auto layout = KVNativeStorageLayout::resolve(KVCacheFamily::AttentionOnly,
            heads, dimension, "fp32", DeviceId::cpu());
        const auto pair = layout.pagePayload(1);
        const auto bytes = publication_.read();
        const auto stride = pair.totalBytes();
        if (!bytes.size() || bytes.size() % stride ||
            reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(float) ||
            bytes.size() / stride > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Native FP32 page must expose exact aligned live K/V token rows");
        rows_ = static_cast<int>(bytes.size() / stride);
        // The retained immutable publication prevents mutation/reuse. Cache
        // these checked addresses once; a dot-product never reopens ownership.
        data_ = reinterpret_cast<const float *>(bytes.data());
        row_width_ = pair.k_bytes / sizeof(float);
    }
    /** @brief Transfer the publication and revoke every moved-from pointer/extent. */
    CPUFP32KVPage(CPUFP32KVPage &&other) noexcept
        : publication_(std::move(other.publication_)), data_(std::exchange(other.data_, nullptr)),
          row_width_(std::exchange(other.row_width_, 0)), heads_(std::exchange(other.heads_, 0)),
          dimension_(std::exchange(other.dimension_, 0)), rows_(std::exchange(other.rows_, 0)) {}
    /** @brief Retire the previous publication before adopting a new page owner. */
    CPUFP32KVPage &operator=(CPUFP32KVPage &&other) noexcept
    {
        if (this != &other)
        {
            publication_ = std::move(other.publication_);
            data_ = std::exchange(other.data_, nullptr);
            row_width_ = std::exchange(other.row_width_, 0);
            heads_ = std::exchange(other.heads_, 0);
            dimension_ = std::exchange(other.dimension_, 0);
            rows_ = std::exchange(other.rows_, 0);
        }
        return *this;
    }
    CPUFP32KVPage(const CPUFP32KVPage &) = delete;
    CPUFP32KVPage &operator=(const CPUFP32KVPage &) = delete;
    /** @return Original PMA issuer; table publication rejects foreign instance ownership. */
    const PhysicalMemoryAuthority *issuer() const noexcept { return publication_.issuer(); }
    /** @return Exact interleaved K/V byte bank for whole-invocation bounds/alias admission. */
    std::span<const std::byte> storage() const noexcept
    {
        return {reinterpret_cast<const std::byte *>(data_), static_cast<size_t>(rows_) * 2 * row_width_ * sizeof(float)};
    }
    /** @return Actual byte stride between native interleaved token rows. */
    size_t tokenStrideBytes() const noexcept { return 2 * row_width_ * sizeof(float); }
    /** @return Exact native geometry, independent of pool frame position. */
    int rows() const noexcept { return rows_; }
    /** @return Actual local KV heads. */
    int heads() const noexcept { return heads_; }
    /** @return Coordinates per native head. */
    int dimension() const noexcept { return dimension_; }
    /** @return A checked immutable key row inside this retained page. */
    const float *key(int row, int head) const
    {
        require(row, head);
        return data_ + static_cast<size_t>(row) * 2 * row_width_ + static_cast<size_t>(head) * dimension_;
    }
    /** @return A checked immutable value row inside this retained page. */
    const float *value(int row, int head) const
    {
        return key(row, head) + row_width_;
    }
private:
    friend class CPUFP32PagedKVTable;
    /** @brief Empty metadata cell, constructible only by the bounded page-table owner. */
    CPUFP32KVPage() noexcept = default;
    /** @brief Reject attempts to address another request, page padding or invalid head. */
    void require(int row, int head) const
    {
        if (!data_ || row < 0 || row >= rows_ || head < 0 || head >= heads_)
            throw std::out_of_range("Native FP32 page access exceeds its live token/head extent");
    }
    PhysicalPagePool::CommittedPage publication_;
    const float *data_ = nullptr;
    size_t row_width_ = 0;
    int heads_ = 0, dimension_ = 0, rows_ = 0;
};

/** @brief Operation-borrowed logical page order; no payload copy or table allocation. */
class CPUFP32PagedKVView final
{
public:
    /**
     * @brief Validate complete pages plus one optional partial tail before native execution.
     * @param pages Retained immutable page-table entries in logical order.
     * @param page_tokens Native logical block size, independent of request capacity.
     */
    CPUFP32PagedKVView(std::span<const CPUFP32KVPage> pages, int page_tokens)
        : pages_(pages), page_tokens_(page_tokens)
    {
        if (pages.empty() || page_tokens <= 0)
            throw std::invalid_argument("Native paged FP32 view requires nonempty live pages and positive block geometry");
        for (size_t page = 0; page < pages.size(); ++page)
        {
            const auto &entry = pages[page];
            if (entry.heads() != pages.front().heads() || entry.dimension() != pages.front().dimension() ||
                entry.rows() <= 0 || entry.rows() > page_tokens || (page + 1 < pages.size() && entry.rows() != page_tokens) ||
                entry.rows() > std::numeric_limits<int>::max() - rows_)
                throw std::invalid_argument("Native paged FP32 view has holes, mixed geometry or overflowing live rows");
            rows_ += entry.rows();
        }
    }
    /** @return Actual retained page banks, for per-bank admission rather than a fictitious contiguous extent. */
    std::span<const CPUFP32KVPage> pages() const noexcept { return pages_; }
    /** @return Total live tokens, excluding every physical frame's padding. */
    int rows() const noexcept { return rows_; }
    /** @return Participant-local KV heads. */
    int heads() const noexcept { return pages_.front().heads(); }
    /** @return Coordinates per head. */
    int dimension() const noexcept { return pages_.front().dimension(); }
    /** @return Original immutable native key address for a logical token/head. */
    const float *key(int logical_row, int head) const
    {
        return page(logical_row).key(logical_row % page_tokens_, head);
    }
    /** @return Original immutable native value address for a logical token/head. */
    const float *value(int logical_row, int head) const
    {
        return page(logical_row).value(logical_row % page_tokens_, head);
    }
private:
    /** @brief Locate a logical token without converting/reconstructing its retained history. */
    const CPUFP32KVPage &page(int logical_row) const
    {
        if (logical_row < 0 || logical_row >= rows_)
            throw std::out_of_range("Native paged FP32 access exceeds live history");
        return pages_[static_cast<size_t>(logical_row / page_tokens_)];
    }
    std::span<const CPUFP32KVPage> pages_;
    int page_tokens_ = 0;
    int rows_ = 0;
};
}
