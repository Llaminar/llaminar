/**
 * @file FixedHeadPipelineTerminalEvidence.h
 * @brief Authenticate actual fixed PLE owner receipts after native pipeline retirement.
 *
 * Every field originates in the existing device generation terminal, including
 * each endpoint's complete native receipts. The frozen placement supplies
 * only expected roles and byte equations. No transfer or live query occurs here.
 */
#pragma once

#include "kernels/qwen4exp/QwenFixedHeadPipelineCounters.h"
#include "loaders/ImmutableFixedHeadPipelineProjection.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace llaminar2::qwen4exp {

inline constexpr std::array<std::string_view, kFixedHeadPipelineMetricCount> kFixedHeadPipelineTerminalMetrics{
    "fixed_head_pipeline_calls", "fixed_head_pipeline_live_ids", "fixed_head_pipeline_native_bytes",
    "fixed_head_pipeline_failures", "fixed_head_pipeline_request_native_bytes",
    "fixed_head_pipeline_response_native_bytes", "fixed_head_pipeline_gather_native_bytes",
    "fixed_head_pipeline_completed_receipts"};

/** @brief Aggregate prefill and decode service observed at the original request terminal. */
inline constexpr std::string_view kFixedHeadPipelineTerminalPhase = "request_terminal";

using FixedHeadPipelineTerminalReport = std::map<std::string, FixedHeadPipelineTerminalTotals>;

/**
 * @brief Check exact compact native payload and endpoint receipt counts over actual completed work.
 * @param records Existing passive immutable_lookup terminal records.
 * @param projection Original resolved head owner and PLE consumer roles.
 * @param row_bytes Original encoded source row width.
 * @param expected_rows Independent exact input rows when the history owns that oracle.
 * @return Actual counters grouped by original GPU, after every byte law passes.
 * @throws std::invalid_argument For incomplete, foreign or failed terminal evidence.
 * @throws std::overflow_error For an unrepresentable sum or byte equation.
 *
 * All owners participate even at M=0. Across an observed useful interval
 * every owner must agree on calls and live rows. A remote range sends one
 * compact request and one compact response per call, including zero-payload
 * completions. A payload-bearing request is required for this model proof.
 * The gather equation concerns attributed logical native bytes. Uneven
 * contributions retain the exact cohort sum without making claims about
 * physical link traffic or attributing every fragment to its original root.
 */
inline FixedHeadPipelineTerminalReport requireFixedHeadPipelineTerminalEvidence(
    std::span<const PerfStatRecord> records, const ImmutableFixedHeadPipelineProjection& projection,
    std::size_t row_bytes, std::optional<std::uint64_t> expected_rows = {})
{
    using Metric=FixedHeadPipelineTerminalMetric;
    const auto index=[](Metric field) {return static_cast<std::size_t>(field);};
    const auto add=[](std::uint64_t a,std::uint64_t b) {
        if (b>std::numeric_limits<std::uint64_t>::max()-a)
            throw std::overflow_error("Fixed pipeline terminal counter sum overflows");
        return a+b;
    };
    const auto multiply=[](std::uint64_t a,std::uint64_t b) {
        if (a && b>std::numeric_limits<std::uint64_t>::max()/a)
            throw std::overflow_error("Fixed pipeline terminal byte extent overflows");
        return a*b;
    };
    const auto consumers=projection.consumers();
    if (!row_bytes || consumers.empty() || projection.owners().empty())
        throw std::invalid_argument("Fixed pipeline terminal proof requires exact original geometry");

    FixedHeadPipelineTerminalReport totals;
    std::map<std::string,std::bitset<kFixedHeadPipelineTerminalMetrics.size()>> seen;
    std::optional<std::uint64_t> common_observations;
    for (const auto& owner:projection.owners())
        totals.try_emplace(owner.allocator.device.to_string());
    for (const auto& record:records) {
        if (record.domain!="immutable_lookup") continue;
        const auto metric=std::find(kFixedHeadPipelineTerminalMetrics.begin(),
            kFixedHeadPipelineTerminalMetrics.end(),record.name);
        const auto authority=record.tags.find("authority");
        if (metric==kFixedHeadPipelineTerminalMetrics.end()) {
            if (record.phase=="generation_terminal" && authority!=record.tags.end() &&
                authority->second=="terminal_device_generation_control")
                throw std::invalid_argument("Fixed pipeline terminal contains a legacy immutable lookup layout");
            continue;
        }
        const auto owner=totals.find(record.device);
        const auto request=record.tags.find("request");
        if (record.kind!=PerfStatRecord::Kind::Counter || !record.count || record.phase!=kFixedHeadPipelineTerminalPhase ||
            record.sequence_word_count!=record.count || (common_observations && *common_observations!=record.count) ||
            owner==totals.end() || authority==record.tags.end() || authority->second!="pipeline_generation_terminal" ||
            request==record.tags.end() || request->second!="0" ||
            !std::isfinite(record.value) || record.value<0 || std::floor(record.value)!=record.value ||
            record.value>9007199254740991.0)
            throw std::invalid_argument("Fixed pipeline terminal counter has invalid identity, phase or integer value");
        const auto field=static_cast<std::size_t>(metric-kFixedHeadPipelineTerminalMetrics.begin());
        if (seen[record.device].test(field))
            throw std::invalid_argument("Fixed pipeline terminal repeats an original owner metric");
        owner->second[field]=static_cast<std::uint64_t>(record.value);
        seen[record.device].set(field);
        common_observations=record.count;
    }

    struct ReaderExtent {std::uint64_t heads=0, remote_heads=0, remote_ranges=0;};
    std::vector<ReaderExtent> readers(consumers.size());
    std::optional<std::uint64_t> common_calls, common_rows;
    std::uint64_t total_heads=0;
    for (const auto& owner:projection.owners()) {
        const auto device=owner.allocator.device.to_string();
        const auto& actual=totals.at(device);
        const auto calls=actual[index(Metric::Calls)], ids=actual[index(Metric::LiveIds)];
        const auto heads=static_cast<std::uint64_t>(owner.head_count);
        const auto bytes=actual[index(Metric::NativeBytes)];
        if (!seen[device].all() || !calls || !heads || !ids || ids%heads ||
            bytes!=multiply(ids,row_bytes) || actual[index(Metric::Failures)])
            throw std::invalid_argument("Fixed pipeline lacks complete successful native owner evidence: "+device);
        const auto rows=ids/heads;
        if ((common_calls && *common_calls!=calls) || (common_rows && *common_rows!=rows))
            throw std::invalid_argument("Fixed pipeline owners disagree on actual query calls or live rows");
        common_calls=calls;
        common_rows=rows;
        total_heads=add(total_heads,heads);
        auto& reader=readers.at(owner.consumer_index);
        reader.heads=add(reader.heads,heads);
        if (owner.stage_index==projection.consumerStage()) continue;
        reader.remote_heads=add(reader.remote_heads,heads);
        reader.remote_ranges=add(reader.remote_ranges,1);
        if (actual[index(Metric::RequestBytes)] || actual[index(Metric::GatherBytes)] ||
            actual[index(Metric::ResponseBytes)]!=bytes ||
            actual[index(Metric::CompletedReceipts)]!=multiply(calls,2))
            throw std::invalid_argument("Fixed pipeline storage-only owner changed its compact response or receipt extent: "+device);
    }

    if (expected_rows && *common_rows!=*expected_rows)
        throw std::invalid_argument("Fixed pipeline terminal live rows differ from the original input history");

    const bool uniform=std::all_of(readers.begin(),readers.end(),[&](const auto& reader) {
        return reader.heads==readers.front().heads;});
    std::uint64_t gather=0;
    for (std::size_t reader=0;reader<readers.size();++reader) {
        const auto device=consumers[reader].device.to_string();
        const auto& actual=totals.at(device);
        const auto& expected=readers[reader];
        const auto peer_bytes=actual[index(Metric::GatherBytes)];
        if (actual[index(Metric::RequestBytes)]!=multiply(multiply(*common_rows,expected.remote_heads),sizeof(std::int64_t)) ||
            actual[index(Metric::ResponseBytes)] ||
            actual[index(Metric::CompletedReceipts)]!=multiply(multiply(*common_calls,expected.remote_ranges),2) ||
            (uniform && peer_bytes!=multiply(multiply(multiply(*common_rows,expected.heads),row_bytes),readers.size()-1)))
            throw std::invalid_argument("Fixed pipeline PLE reader changed its compact request, gather or receipt extent: "+device);
        gather=add(gather,peer_bytes);
    }
    if (gather!=multiply(multiply(multiply(*common_rows,total_heads),row_bytes),readers.size()-1))
        throw std::invalid_argument("Fixed pipeline native gather bytes differ from useful reader contributions");
    return totals;
}

} // namespace llaminar2::qwen4exp
