/**
 * @file Qwen4ExpLocalHotTerminalEvidence.h
 * @brief Completed participant-local cache evidence for real QFN generation.
 *
 * Original retained storage supplies identity, native row size and capacity.
 * Observations cannot choose their own ownership or reinterpret a byte count.
 * This passive test projection neither reads devices nor changes cache state.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableLocalHotTerminal.h"
#include "utils/PerfStatsCollector.h"
#include "backends/DeviceId.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <map>
#include <span>
#include <string>

namespace llaminar2::test::parity::qwen4exp {
/** @brief Cold expected identity from the actual retained local cache owner. */
struct Qwen4ExpLocalHotTerminalOwner {
    DeviceId device;
    std::uint64_t storage_identity;
    std::uint64_t slots;
    std::size_t native_row_bytes;
    ::llaminar2::qwen4exp::local_hot::TerminalReceiveRole receive_role;
};
/**
 * @brief Authenticate every local participant's complete, bounded terminal work.
 * @param records Unmodified collected ngram counters, including other publishers.
 * @param owners Exact original local-cache roster, never inferred from records.
 * @throws std::invalid_argument For absent/malformed/foreign work or impossible bytes.
 * @details Completed-storage windows may contain multiple terminal publications.
 * Window groups must each have all fifteen fields with equal publication counts.
 * C0 cannot report hits or churn. Positive capacity need not imply a hit: only
 * real request locality can establish that separate performance obligation.
 */
inline void requireQwen4ExpLocalHotTerminalEvidence(std::span<const PerfStatRecord> records,
    std::span<const Qwen4ExpLocalHotTerminalOwner> owners)
{
    namespace h=::llaminar2::qwen4exp::local_hot;
    if(owners.empty())throw std::invalid_argument("Local terminal evidence requires original consumers");
    std::map<std::string,const Qwen4ExpLocalHotTerminalOwner*> roster;
    for(const auto& owner:owners) {
        if(!owner.device.is_gpu() || !owner.storage_identity || !owner.native_row_bytes ||
            (owner.receive_role!=h::TerminalReceiveRole::Coordinator && owner.receive_role!=h::TerminalReceiveRole::Reader) ||
            !roster.emplace(owner.device.toString(),&owner).second)
            throw std::invalid_argument("Local terminal evidence repeats or changes an original consumer");
    }
    struct Window {
        h::TerminalObservation totals;
        std::array<std::uint64_t,15> counts{};
        std::bitset<15> seen;
    };
    std::map<std::pair<std::string,std::string>,Window> windows;
    for(const auto& record:records) {
        if(record.domain!="ngram_lookup" || !record.name.starts_with("local_"))continue;
        const auto metric=std::find(h::TerminalMetricNames.begin(),h::TerminalMetricNames.end(),record.name);
        const auto owner=roster.find(record.device);
        if(metric==h::TerminalMetricNames.end() || owner==roster.end() ||
            record.kind!=PerfStatRecord::Kind::Counter || record.phase!="generation_terminal" ||
            !record.count || !std::isfinite(record.value) || record.value<0 ||
            std::floor(record.value)!=record.value || record.value>9007199254740991.0)
            throw std::invalid_argument("Local terminal evidence lost its original completed publisher");
        const auto index=static_cast<std::size_t>(metric-h::TerminalMetricNames.begin());
        const auto window=record.tags.find("window");
        if(window==record.tags.end() || (window->second!="since_original_storage_initialization" &&
            window->second!="between_completed_terminals") ||
            (window->second=="since_original_storage_initialization" && record.count!=1))
            throw std::invalid_argument("Local terminal evidence has an unknown or repeated initial storage window");
        const PerfStatsCollector::Tags expected{
            {"authority","completed_local_hot_terminal"},
            {"storage_identity",std::to_string(owner->second->storage_identity)},
            {"unit",index<4?"original_l1_ids":index<10?"native_rows":"bytes"},
            {"wire_attribution","receive_completed_only"},{"window",window->second}};
        if(record.tags!=expected || record.sequence_word_count!=record.count ||
            record.sequence_minimum_words.size()!=1 || record.sequence_maximum_words.size()!=1 ||
            record.sequence_minimum_words[0]>record.sequence_maximum_words[0])
            throw std::invalid_argument("Local terminal evidence changed storage identity, units or publication metadata");
        const auto average=static_cast<long double>(record.value)/record.count;
        if(average<record.sequence_minimum_words[0] || average>record.sequence_maximum_words[0])
            throw std::invalid_argument("Local terminal aggregate disagrees with its completed observations");
        auto& actual=windows[{record.device,window->second}];
        if(actual.seen.test(index))throw std::invalid_argument("Local terminal evidence duplicates a completed field");
        actual.seen.set(index);actual.counts[index]=record.count;
        actual.totals.values[index]=static_cast<std::uint64_t>(record.value);
    }
    std::map<std::string,std::uint64_t> reads;
    for(const auto& [identity,window]:windows) {
        const auto& owner=*roster.at(identity.first);
        h::TerminalObservation checked;
        if(!window.seen.all() ||
            std::any_of(window.counts.begin(),window.counts.end(),[&](auto count){return count!=window.counts.front();}) ||
            !h::terminalObservationDelta(window.totals,{},checked) ||
            !h::validTerminalWireDelta(checked,owner.receive_role,owner.native_row_bytes) ||
            (!owner.slots && (checked.values[1] || checked.values[8] || checked.values[9])))
            throw std::invalid_argument("Local terminal evidence has incomplete or impossible cache/native work");
        // A receiver gets at most one native row per original local miss;
        // deduplication can reduce this further. Capacity is never the extent.
        if(owner.receive_role==h::TerminalReceiveRole::Reader &&
            checked.values[11]/owner.native_row_bytes>checked.values[0]-checked.values[1])
            throw std::invalid_argument("Local terminal native delivery exceeds actual local misses");
        auto& total=reads[identity.first];
        if(!h::accumulateCompletedWireCounter(total,checked.values[0]))
            throw std::invalid_argument("Local terminal read total overflows");
    }
    for(const auto& [device,owner]:roster) {
        (void)owner;
        if(!reads.contains(device) || !reads.at(device))
            throw std::invalid_argument("Local terminal evidence omitted useful work on an original consumer");
    }
}
} // namespace llaminar2::test::parity::qwen4exp
