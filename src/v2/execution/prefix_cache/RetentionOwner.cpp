/**
 * @file RetentionOwner.cpp
 * @brief §8.4 retention ordering authority implementation.
 */
#include "execution/prefix_cache/RetentionOwner.h"

#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
namespace
{
std::optional<RetentionIntent> defaultIntent()
{
    // Code-owned default: `Default`, value-score governed (no strong promise).
    return RetentionIntent{RetentionClass::Default, 0};
}
} // namespace

std::optional<RetentionIntent> RetentionClassMap::forValue(std::string_view value,
    std::string *refusal)
{
    if (value.empty())
        return defaultIntent();
    // The ONLY accepted standard values (P0 baseline §1). A recognized field
    // with any other value is REFUSED with a precise diagnostic — never
    // silently remapped to another class.
    if (value == "in_memory")
        return RetentionIntent{RetentionClass::InMemory, 0};
    if (value == "24h")
        // 24 h measured in ordinals from the writing request start (where the
        // caller prices the ordinal horizon; the class carries the contract).
        return RetentionIntent{RetentionClass::TieredDay,
            24 * 60 * 60}; // Seconds; the caller owns the ordinal clock.
    if (refusal)
        *refusal = "unsupported prompt_cache_retention value; accepted values: "
                   "in_memory, 24h";
    return std::nullopt;
}

RetentionOwner::RetentionOwner(std::size_t metadata_capacity)
    : entries_(metadata_capacity)
{
    if (metadata_capacity == 0)
        throw std::invalid_argument(
            "retention owner requires a positive admitted metadata membership");
}

bool RetentionOwner::admit(RetentionEntryKey key, RetentionIntent intent,
    std::uint64_t write_ordinal, RetentionValueScore value, bool live)
{
    for (std::size_t i = 0; i < count_; ++i)
        if (entries_[i].key == key) return false; // Duplicate identity: no mutation.
    if (count_ == capacity()) return false;       // Bounded metadata full: no mutation.
    Entry entry;
    entry.key = key;
    entry.intent = intent;
    entry.write_ordinal = write_ordinal;
    entry.effective_use_seq = next_use_seq_++;
    entry.value = value;
    entry.live = live;
    entries_[count_++] = entry;
    return true;
}

bool RetentionOwner::refresh(RetentionEntryKey key, std::uint64_t use_ordinal)
{
    for (std::size_t i = 0; i < count_; ++i)
    {
        if (entries_[i].key != key) continue;
        // Effective use (§8.2 ordered receipt): the minimum lifetime restarts
        // from the use ordinal; the stable tie-break sequence is refreshed by
        // the SAME receipt so ordering stays deterministic.
        entries_[i].write_ordinal = use_ordinal;
        entries_[i].effective_use_seq = next_use_seq_++;
        return true;
    }
    return false;
}

std::optional<RetentionIntent> RetentionOwner::peek(RetentionEntryKey key,
    RetentionValueScore *score) const
{
    for (std::size_t i = 0; i < count_; ++i)
    {
        if (entries_[i].key != key) continue;
        // Provisional inspection: NO refresh, NO sequence advance.
        if (score) *score = entries_[i].value;
        return entries_[i].intent;
    }
    return std::nullopt;
}

bool RetentionOwner::expired(RetentionEntryKey key, std::uint64_t now) const
{
    for (std::size_t i = 0; i < count_; ++i)
    {
        if (entries_[i].key != key) continue;
        // Minimum lifetime is measured from the request start that wrote or
        // last effectively used the entry; zero = no strong promise (no
        // ordinal surplus — the value score alone governs once idle).
        const auto &e = entries_[i];
        const auto horizon = e.intent.min_lifetime_ordinals;
        if (horizon == 0) return true; // Value-score governed: no class after write.
        return now >= e.write_ordinal + horizon;
    }
    return false; // Unknown identity is not expired (absent, not expired).
}

bool RetentionOwner::setLive(RetentionEntryKey key, bool live)
{
    for (std::size_t i = 0; i < count_; ++i)
    {
        if (entries_[i].key == key)
        {
            entries_[i].live = live;
            return true;
        }
    }
    return false;
}

bool RetentionOwner::remove(RetentionEntryKey key)
{
    for (std::size_t i = 0; i < count_; ++i)
    {
        if (entries_[i].key != key) continue;
        entries_[i] = entries_[count_ - 1];
        --count_;
        return true;
    }
    return false;
}

std::vector<RetentionEntryKey> RetentionOwner::selectVictims(std::uint64_t now,
    std::size_t deficit) const
{
    std::vector<RetentionEntryKey> victims;
    if (deficit == 0) return victims;

    // Victim rank components; lower rank = selected first.
    // (1) live/protected never selected; (2) expired lose class (rank -1, all
    // expired before every unexpired class, value then decides); (3)
    // unexpired by class rank ascending; (4) value score; (5) older sequence.
    struct Ranked
    {
        RetentionEntryKey key;
        long class_rank; // -1 = expired.
        RetentionValueScore value;
        std::uint64_t seq;
    };
    std::vector<Ranked> idle;
    idle.reserve(count_);
    for (std::size_t i = 0; i < count_; ++i)
    {
        const auto &e = entries_[i];
        if (e.live) continue;
        Ranked r;
        r.key = e.key;
        const auto horizon = e.intent.min_lifetime_ordinals;
        const bool is_expired =
            horizon == 0 ? true : now >= e.write_ordinal + horizon;
        r.class_rank = is_expired ? -1 : static_cast<long>(e.intent.cls);
        r.value = e.value;
        r.seq = e.effective_use_seq;
        idle.push_back(r);
    }
    std::stable_sort(idle.begin(), idle.end(),
        [](const Ranked &a, const Ranked &b) {
            if (a.class_rank != b.class_rank)
                return a.class_rank < b.class_rank; // Expired first, then InMemory < Default < TieredDay.
            if (a.value < b.value)
                return true; // Lower value = victim first.
            if (b.value < a.value)
                return false;
            // Stable tie-break: older effective-use sequence first.
            return a.seq < b.seq;
        });
    const auto take = std::min(deficit, idle.size());
    for (std::size_t i = 0; i < take; ++i)
        victims.push_back(idle[i].key);
    return victims;
}

std::optional<RetentionClass> RetentionOwner::sharedClass(
    const std::vector<RetentionEntryKey> &referencing, std::uint64_t now) const
{
    // Shared pages take the highest UNEXPIRED class of any referencing entry
    // (an expired reference never raises the class); the page moves once in
    // the driver under this aggregate class.
    std::optional<RetentionClass> best;
    for (const auto &key : referencing)
    {
        for (std::size_t i = 0; i < count_; ++i)
        {
            if (entries_[i].key != key) continue;
            if (expired(key, now)) continue;
            if (!best.has_value() ||
                static_cast<int>(entries_[i].intent.cls) > static_cast<int>(*best))
                best = entries_[i].intent.cls;
            break;
        }
    }
    return best;
}
} // namespace llaminar2
