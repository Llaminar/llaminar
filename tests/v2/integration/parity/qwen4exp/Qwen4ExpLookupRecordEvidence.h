/**
 * @file Qwen4ExpLookupRecordEvidence.h
 * @brief Separate exact request-reset observations from native lookup evidence.
 *
 * The immutable_lookup domain contains two deliberately untagged lifecycle
 * counters as well as tagged storage work. Only the two named request-reset
 * producers have that contract. Missing tags or an unfamiliar name never make
 * another record a lifecycle counter, and lifecycle records cannot satisfy a
 * lookup-work obligation. The raw counter export precedes validation so a bad
 * record remains inspectable when the campaign rejects it.
 */
#pragma once
#include "Qwen4ExpFixedHeadLookupEvidence.h"
#include "loaders/HostImmutableRowMode.h"
#include <iomanip>
#include <optional>
#include <ostream>

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Closed evidence categories; request reset is never native lookup work. */
enum class Qwen4ExpLookupCounterRole { LookupWork, RequestResetLifecycle };

/**
 * @brief Authenticate one immutable-lookup counter against its actual producer class.
 * @param record One raw counter from the immutable_lookup domain.
 * @param consumers Actual resolved PLE execution participants, not expert-only devices.
 * @param hosted Actual host provider mode; absent only for native fixed-head lookup.
 * @return Exact lifecycle/work role, after validating identity and counter shape.
 * @throws std::invalid_argument For a foreign owner, unexpected name, wrong mode,
 *         transport, phase, missing tags or malformed numeric observation.
 * @note The complete native byte/row equations remain the campaign's separate
 *       mandatory evidence checks; classifying reset never supplies missing work.
 */
inline Qwen4ExpLookupCounterRole classifyQwen4ExpLookupCounter(
    const PerfStatRecord &record, std::span<const DeviceId> consumers,
    std::optional<HostImmutableRowMode> hosted)
{
    const auto owner = std::find_if(consumers.begin(), consumers.end(), [&](DeviceId device) {
        return device.is_valid() && device.toString() == record.device;
    });
    if (record.domain != "immutable_lookup" || record.kind != PerfStatRecord::Kind::Counter ||
        owner == consumers.end() || !record.count || !std::isfinite(record.value) || record.value < 0 ||
        std::floor(record.value) != record.value || record.value > 9007199254740991.0)
        throw std::invalid_argument("Flash Next lookup counter has an invalid producer or exact integer observation");

    if (record.name == "reset_terminal_closes" || record.name == "reset_terminal_waits")
    {
        // The real DGO reset producers add precisely1 per call, no storage or
        // transport tags, after submitting close or completing its exact event.
        if (!owner->is_gpu() || record.phase != "request_reset" || !record.tags.empty() ||
            record.count > 9007199254740991ULL || record.value != static_cast<double>(record.count))
            throw std::invalid_argument("Flash Next request-reset counter changed its exact lifecycle contract");
        return Qwen4ExpLookupCounterRole::RequestResetLifecycle;
    }

    const auto storage = record.tags.find("storage"), transport = record.tags.find("transport");
    const auto expectedStorage = hosted ? hostImmutableRowModeName(*hosted) : "vram";
    const auto expectedTransport = hosted ? (owner->is_cpu() ? "cpu_direct" : "mapped_channel") : "native_fixed_head";
    const auto expectedPhase = hosted && owner->is_cpu() ? "lookup_completion" : "generation_terminal";
    if ((!hosted && !owner->is_gpu()) || storage == record.tags.end() || transport == record.tags.end() ||
        storage->second != expectedStorage || transport->second != expectedTransport || record.phase != expectedPhase)
        throw std::invalid_argument("Flash Next lookup work changed its exact mode, transport or completion phase");

    const auto contains = [&](const auto &names) {
        return std::find(names.begin(), names.end(), record.name) != names.end();
    };
    if (!hosted)
    {
        if (!contains(kFixedHeadLookupCounters))
            throw std::invalid_argument("Flash Next native fixed-head counter name is unexpected");
    }
    else
    {
        constexpr std::array<std::string_view, 7> common{
            "requests", "ids", "native_bytes", "failures", "window_hits", "window_misses", "window_evictions"};
        constexpr std::array<std::string_view, 5> mapped{
            "intervals", "id_bytes", "metadata_bytes", "acknowledgement_bytes", "terminal_bytes"};
        constexpr std::array<std::string_view, 8> adaptive{
            "adaptive_batches", "ram_rows", "source_rows", "duplicate_rows", "cache_admissions",
            "cache_evictions", "cache_admission_rejections", "all_pinned_responses"};
        if (!contains(common) && !(owner->is_gpu() && contains(mapped)) &&
            !(*hosted == HostImmutableRowMode::AdaptiveRAMSSD && contains(adaptive)))
            throw std::invalid_argument("Flash Next hosted lookup counter name is unexpected for its actual provider");
    }
    return Qwen4ExpLookupCounterRole::LookupWork;
}

/**
 * @brief Export raw immutable-domain counters before any evidence assertion.
 * @param csv Open diagnostic output; the caller owns stream error reporting.
 * @param records Original PerfStats snapshot, unchanged by export or validation.
 * @note Untagged reset and malformed/foreign counters remain visible. No record
 *       can acquire storage/transport identity merely because the CSV has blanks.
 */
inline void writeQwen4ExpLookupCounterCSV(std::ostream &csv, std::span<const PerfStatRecord> records)
{
    csv << "storage,transport,phase,device,counter,value\n";
    for (const auto &record : records)
        if (record.kind == PerfStatRecord::Kind::Counter)
        {
            const auto mode = record.tags.find("storage"), wire = record.tags.find("transport");
            csv << (mode == record.tags.end() ? "" : mode->second) << ','
                << (wire == record.tags.end() ? "" : wire->second) << ','
                << record.phase << ',' << record.device << ',' << record.name << ','
                << std::setprecision(17) << record.value << '\n';
        }
}
} // namespace llaminar2::test::parity::qwen4exp
