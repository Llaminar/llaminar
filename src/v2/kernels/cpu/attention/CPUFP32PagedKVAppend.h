/**
 * @file CPUFP32PagedKVAppend.h
 * @brief Exact native FP32 page production without a retained-history gather.
 *
 * The caller supplies an original PMA private destination and positive native
 * page geometry. Separated token-major K/V rows are interleaved directly into
 * that frame. A retained partial tail, when supplied, is the only copied prior
 * payload; full-history materialization and capacity-padding copies are absent.
 * The byte authority publishes only after every granted byte is produced.
 * Logical table commit, speculative visibility and request admission remain
 * separate authorities; this CPU producer does not schedule or allocate frames.
 */
#pragma once
#include "CPUFP32PagedKVView.h"
#include <functional>
#include <optional>

namespace llaminar2
{
/** @brief Participant-local exact native inputs; optional source is one retained partial tail. */
struct CPUFP32KVAppendInput
{
    std::span<const float> keys;
    std::span<const float> values;
    int heads = 0;
    int dimension = 0;
    int page_tokens = 0;
    std::optional<std::reference_wrapper<const CPUFP32KVPage>> tail;
};

/**
 * @brief Produce one complete native page into an empty original private frame.
 * @param grant Original private growth owner; no allocation is performed here.
 * @param destination Exact original empty-frame incarnation.
 * @param input Live K/V rows and optional immutable partial tail, never padding.
 * @return Completed native page, or empty only while an original borrower pins the destination.
 * @throws std::invalid_argument For malformed geometry, provenance, alignment or aliasing.
 * @throws std::logic_error For stale ownership or a nonempty destination.
 *
 * All geometry/provenance and whole-invocation alias checks precede the first
 * copy. Prefix bytes remain pinned throughout production. No arithmetic is
 * applied to FP32 operands; original bit patterns are retained exactly.
 */
inline std::optional<CPUFP32KVPage> appendCPUFP32KVPage(PhysicalPagePool::Grant &grant,
    PhysicalPagePool::FrameHandle destination, const CPUFP32KVAppendInput &input)
{
    const auto layout = KVNativeStorageLayout::resolve(KVCacheFamily::AttentionOnly,
        input.heads, input.dimension, "fp32", DeviceId::cpu());
    const auto width = layout.pagePayload(1).k_bytes / sizeof(float);
    if (!grant.valid() || input.page_tokens <= 0 || input.keys.empty() || !input.keys.data() ||
        !input.values.data() || input.keys.size() != input.values.size() || input.keys.size() % width)
        throw std::invalid_argument("Native FP32 append requires exact live K/V rows and an original owner");
    const int prior_rows = input.tail ? input.tail->get().rows() : 0;
    if (input.tail && (prior_rows <= 0 || input.tail->get().issuer() != grant.issuer() ||
        input.tail->get().heads() != input.heads || input.tail->get().dimension() != input.dimension))
        throw std::invalid_argument("Native FP32 append tail has incompatible geometry or physical authority");
    const auto new_rows = input.keys.size() / width;
    if (prior_rows >= input.page_tokens || new_rows > static_cast<size_t>(input.page_tokens - prior_rows))
        throw std::invalid_argument("Native FP32 append exceeds one actual logical page");
    const auto bytes = layout.pagePayload(prior_rows + static_cast<int>(new_rows)).totalBytes();
    auto plan = grant.prepareFreshWrite(destination, bytes);
    if (!plan) return std::nullopt;
    const auto keys = std::as_bytes(input.keys), values = std::as_bytes(input.values);
    const auto prefix = input.tail ? input.tail->get().storage() : std::span<const std::byte>{};
    if (!plan->destinationAligned(alignof(float)) || plan->overlaps(keys) || plan->overlaps(values) ||
        plan->overlaps(prefix))
        throw std::invalid_argument("Native FP32 append destination has invalid alignment or aliases a source");
    if (!prefix.empty()) plan->copyNext(prefix);
    for (size_t row = 0; row < new_rows; ++row)
    {
        plan->copyNext(keys.subspan(row * width * sizeof(float), width * sizeof(float)));
        plan->copyNext(values.subspan(row * width * sizeof(float), width * sizeof(float)));
    }
    return CPUFP32KVPage(plan->commit(), input.heads, input.dimension);
}
}
