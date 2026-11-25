/**
 * @file PagedKVArchive.cpp
 * @brief Fixed-layout immutable archive section implementation.
 *
 * Layout (little-endian): magic u64, family u32, format u32, heads i32,
 * dimension i32, page_tokens i32, rows u64, payload bytes. The payload is the
 * exact interleaved native K/V rows in page order, with no physical padding.
 * Parse rejects trailing bytes, unsupported magic, zero/inconsistent geometry
 * and payload/length mismatches before exposing any bytes.
 */
#include "PagedKVArchive.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
namespace
{
constexpr std::uint64_t kArchiveMagic = 0x4C4B564450474156ULL; // "LKVDPGAV" in memory order.
constexpr std::size_t kHeaderBytes =
    sizeof(std::uint64_t) + sizeof(std::uint32_t) + sizeof(std::uint32_t) +
    sizeof(std::int32_t) * 3 + sizeof(std::uint64_t);
constexpr std::size_t kMaxBytes = 1ull << 32; // Hard bound for one decodable section.

std::uint32_t familyCode(KVCacheFamily family)
{
    switch (family)
    {
        case KVCacheFamily::AttentionOnly: return 1;
        case KVCacheFamily::Hybrid: return 2;
    }
    throw std::logic_error("Unknown KV family code");
}
KVCacheFamily familyFromCode(std::uint32_t code)
{
    switch (code)
    {
        case 1: return KVCacheFamily::AttentionOnly;
        case 2: return KVCacheFamily::Hybrid;
    }
    throw std::invalid_argument("Unsupported archive KV family code");
}
std::uint32_t formatCode(KVStorageFormat format)
{
    return static_cast<std::uint32_t>(format);
}
KVStorageFormat formatFromCode(std::uint32_t code)
{
    switch (code)
    {
        case static_cast<std::uint32_t>(KVStorageFormat::FP32):
        case static_cast<std::uint32_t>(KVStorageFormat::BF16):
        case static_cast<std::uint32_t>(KVStorageFormat::FP16):
        case static_cast<std::uint32_t>(KVStorageFormat::Q8_1):
        case static_cast<std::uint32_t>(KVStorageFormat::Q16_1):
        case static_cast<std::uint32_t>(KVStorageFormat::TQ4):
        case static_cast<std::uint32_t>(KVStorageFormat::TQ8):
            return static_cast<KVStorageFormat>(code);
    }
    throw std::invalid_argument("Unsupported archive KV format code");
}
} // namespace

PagedKVArchive PagedKVArchive::fromView(const CPUFP32PagedKVView &view)
{
    PagedKVArchiveMeta meta{
        .family = KVCacheFamily::AttentionOnly,
        .format = KVStorageFormat::FP32,
        .heads = view.heads(),
        .dimension = view.dimension(),
        .page_tokens = view.pages().empty() ? 0 : view.rows(),
        .rows = static_cast<std::size_t>(view.rows()),
    };
    if (view.pages().empty())
        return PagedKVArchive(meta, {});
    meta.page_tokens = view.pages().front().rows();
    const auto layout = KVNativeStorageLayout::resolve(KVCacheFamily::AttentionOnly,
        meta.heads, meta.dimension, "fp32", DeviceId::cpu());
    const auto stride = layout.pagePayload(1).totalBytes();
    if (meta.rows > std::numeric_limits<std::size_t>::max() / stride)
        throw std::overflow_error("Paged archive payload size overflows");
    std::vector<std::byte> payload(meta.rows * stride);
    for (int row = 0; row < view.rows(); ++row)
    {
        std::memcpy(payload.data() + static_cast<std::size_t>(row) * stride,
            view.key(row, 0), stride);
    }
    return PagedKVArchive(meta, std::move(payload));
}

PagedKVArchive PagedKVArchive::parse(std::span<const std::byte> bytes)
{
    if (bytes.size() < kHeaderBytes)
        throw std::invalid_argument("Paged archive section is shorter than its fixed header");
    std::size_t offset = 0;
    const auto readU64 = [&]() {
        std::uint64_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value)); offset += sizeof(value);
        return value;
    };
    const auto readU32 = [&]() {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value)); offset += sizeof(value);
        return value;
    };
    const auto readI32 = [&]() {
        std::int32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value)); offset += sizeof(value);
        return value;
    };
    const auto magic = readU64();
    if (magic != kArchiveMagic)
        throw std::invalid_argument("Paged archive has an unrecognized magic");
    PagedKVArchiveMeta meta{
        .family = familyFromCode(readU32()),
        .format = formatFromCode(readU32()),
        .heads = readI32(),
        .dimension = readI32(),
        .page_tokens = readI32(),
        .rows = readU64(),
    };
    if (meta.heads <= 0 || meta.dimension <= 0 || meta.page_tokens <= 0 ||
        meta.rows > std::numeric_limits<std::size_t>::max() / sizeof(float) / 2)
        throw std::invalid_argument("Paged archive has invalid native geometry");
    const auto layout = KVNativeStorageLayout::resolve(meta.family, meta.heads,
        meta.dimension, KVStorageFormatName(meta.format), DeviceId::cpu());
    const auto stride = layout.pagePayload(1).totalBytes();
    if (meta.rows > std::numeric_limits<std::size_t>::max() / stride)
        throw std::invalid_argument("Paged archive payload length overflows");
    const std::size_t expected = meta.rows * stride;
    if (kMaxBytes < expected || bytes.size() != offset + expected)
        throw std::invalid_argument("Paged archive payload extent does not match its declared rows");
    std::vector<std::byte> payload(bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.end());
    return PagedKVArchive(meta, std::move(payload));
}

std::vector<CPUFP32KVPage> PagedKVArchive::materialize(PhysicalPagePool::Grant &grant, int page_tokens) const
{
    if (empty() || meta_.format != KVStorageFormat::FP32 ||
        meta_.family != KVCacheFamily::AttentionOnly)
        throw std::invalid_argument("Paged archive materialization requires a nonempty FP32 attention section");
    if (page_tokens <= 0)
        throw std::invalid_argument("Paged archive materialization requires positive page geometry");
    const auto layout = KVNativeStorageLayout::resolve(meta_.family, meta_.heads,
        meta_.dimension, KVStorageFormatName(meta_.format), DeviceId::cpu());
    const auto stride = layout.pagePayload(1).totalBytes();
    if (page_tokens > std::numeric_limits<std::size_t>::max() / stride)
        throw std::invalid_argument("Paged archive page geometry overflows");
    const std::size_t page_bytes = static_cast<std::size_t>(page_tokens) * stride;
    if (meta_.rows > std::numeric_limits<std::size_t>::max() / stride)
        throw std::invalid_argument("Paged archive row geometry overflows");
    const std::size_t total_bytes = meta_.rows * stride;
    if (payload_.size() != total_bytes)
        throw std::logic_error("Paged archive payload does not match its provenance");
    std::vector<CPUFP32KVPage> pages;
    for (std::size_t offset = 0; offset < total_bytes;)
    {
        const std::size_t count = std::min(page_bytes, total_bytes - offset);
        const std::size_t rows = count / stride;
        if (!rows)
            throw std::logic_error("Paged archive materialization produced an empty page");
        auto handle = grant.assignFrame(); // G -> O exactly once for this private page.
        auto plan = grant.prepareFreshWrite(handle, count);
        if (!plan)
            throw std::logic_error("Paged archive materialization destination is not writable");
        plan->copyFrom({payload_.data() + offset, count});
        pages.emplace_back(plan->commit(), meta_.heads, meta_.dimension);
        offset += count;
    }
    return pages;
}

std::vector<std::byte> PagedKVArchive::serialize() const
{
    const auto layout = KVNativeStorageLayout::resolve(meta_.family, meta_.heads,
        meta_.dimension, KVStorageFormatName(meta_.format), DeviceId::cpu());
    const auto stride = layout.pagePayload(1).totalBytes();
    if (payload_.size() != meta_.rows * stride)
        throw std::logic_error("Paged archive payload does not match its provenance");
    std::vector<std::byte> output;
    output.reserve(kHeaderBytes + payload_.size());
    auto appendRaw = [&](const void *data, std::size_t size) {
        const auto *bytes = static_cast<const std::byte *>(data);
        output.insert(output.end(), bytes, bytes + size);
    };
    const std::uint64_t magic = kArchiveMagic;
    const std::uint32_t family = familyCode(meta_.family);
    const std::uint32_t format = formatCode(meta_.format);
    const std::int32_t heads = meta_.heads;
    const std::int32_t dimension = meta_.dimension;
    const std::int32_t page_tokens = meta_.page_tokens;
    const std::uint64_t rows = meta_.rows;
    appendRaw(&magic, sizeof(magic));
    appendRaw(&family, sizeof(family));
    appendRaw(&format, sizeof(format));
    appendRaw(&heads, sizeof(heads));
    appendRaw(&dimension, sizeof(dimension));
    appendRaw(&page_tokens, sizeof(page_tokens));
    appendRaw(&rows, sizeof(rows));
    appendRaw(payload_.data(), payload_.size());
    return output;
}
} // namespace llaminar2
