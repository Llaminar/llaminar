/**
 * @file Qwen4ExpFixedHeadLookupEvidence.h
 * @brief Require complete terminal native-lookup evidence for every actual owner.
 *
 * This model-test validator consumes executed counters, never requested policy
 * or allocation capacity. Native receipts describe useful contributed bytes;
 * they are not physical link traffic or the sum of receive-side replicas.
 */
#pragma once
#include "backends/DeviceId.h"
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string_view>

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Terminal fields emitted by actual fixed-head kernels/native collectives. */
inline constexpr std::array<std::string_view, 6> kFixedHeadLookupCounters{
    "fixed_head_lookup_calls", "fixed_head_live_ids", "fixed_head_native_bytes",
    "fixed_head_failures", "fixed_head_header_native_bytes", "fixed_head_payload_native_bytes"};

/**
 * @brief Authenticate every participant and exact native communication equations.
 * @param records Completed PerfStats records from the canonical model campaign.
 * @param participants Actual resolved source owners, independent of counter tags.
 * @param row_bytes Original immutable source's native row width.
 * @throws std::invalid_argument For missing, malformed, foreign or uneconomical evidence.
 *
 * Each successful cohort has equal head counts and identical query counts.
 * Native source rows may differ in value and table range length, but each
 * participant packs L*(16/P) rows. Header8B is required even for an empty lookup.
 * Zero failure counters must be present; absence is not equivalent to zero.
 */
inline void requireQwen4ExpFixedHeadLookupEvidence(
    std::span<const PerfStatRecord> records, std::span<const DeviceId> participants,
    std::size_t row_bytes)
{
    struct Totals { std::array<std::uint64_t, 6> fields{}; std::bitset<6> seen; };
    std::map<std::string, Totals> owners;
    if (!row_bytes || (participants.size() != 2 && participants.size() != 4 && participants.size() != 8))
        throw std::invalid_argument("Fixed-head evidence requires exact source width and TP2/4/8 owners");
    const auto local_heads = static_cast<std::size_t>(::llaminar2::qwen4exp::kNgramHeads) / participants.size();
    for (const auto device : participants)
        if (!device.is_valid() || !device.is_gpu() || device.type != participants.front().type ||
            !owners.emplace(device.toString(), Totals{}).second)
            throw std::invalid_argument("Fixed-head evidence has invalid or duplicate source owners");
    for (const auto &record : records)
    {
        if (record.domain != "immutable_lookup") continue;
        const auto name = std::find(kFixedHeadLookupCounters.begin(), kFixedHeadLookupCounters.end(), record.name);
        if (name == kFixedHeadLookupCounters.end()) continue;
        const auto storage = record.tags.find("storage"), transport = record.tags.find("transport");
        const auto owner = owners.find(record.device);
        if (record.kind != PerfStatRecord::Kind::Counter || !record.count || record.phase != "generation_terminal" ||
            storage == record.tags.end() || storage->second != "vram" ||
            transport == record.tags.end() || transport->second != "native_fixed_head" || owner == owners.end() ||
            !std::isfinite(record.value) || record.value < 0 || std::floor(record.value) != record.value ||
            record.value > 9007199254740991.0)
            throw std::invalid_argument("Fixed-head terminal counter has invalid identity, phase or exact integer value");
        const auto index = static_cast<std::size_t>(name - kFixedHeadLookupCounters.begin());
        auto &total = owner->second.fields[index];
        const auto value = static_cast<std::uint64_t>(record.value);
        if (value > std::numeric_limits<std::uint64_t>::max() - total)
            throw std::invalid_argument("Fixed-head terminal counter sum overflow");
        total += value;
        owner->second.seen.set(index);
    }
    const Totals *common = nullptr;
    for (const auto &[device, total] : owners)
    {
        const auto &[calls, ids, native, failures, header, payload] = total.fields;
        if (!total.seen.all() || !calls || !ids || ids % local_heads || failures ||
            calls > std::numeric_limits<std::uint64_t>::max() / 8 || header != calls * 8 ||
            ids > std::numeric_limits<std::uint64_t>::max() / row_bytes || native != ids * row_bytes ||
            payload != native)
            throw std::invalid_argument("Fixed-head owner lacks complete successful live-byte evidence: " + device);
        if (common && (calls != common->fields[0] || ids != common->fields[1]))
            throw std::invalid_argument("Fixed-head participant query windows disagree");
        common = &total;
    }
}
} // namespace llaminar2::test::parity::qwen4exp
