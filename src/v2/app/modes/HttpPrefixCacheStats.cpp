/**
 * @file HttpPrefixCacheStats.cpp
 * @brief Deduplicate shared disk capacity and retain truthful live cache gauges.
 *
 * RAM occupancy counts installed payload keys; aliases retained by restores or
 * asynchronous writers can outlive those keys. Disk occupancy counts committed
 * active payloads, excluding append history and obsolete compaction inodes.
 * Freshness names the last owner publication, not the time of the HTTP poll.
 */
#include "app/modes/HttpPrefixCacheStats.h"
#include <map>
#include <optional>
#include <set>
#include <utility>

namespace llaminar2
{
    namespace
    {
        using json = nlohmann::json;
        using Snapshot = PrefixCacheTierTelemetry::Snapshot;

        /** @brief Capacity is a payload envelope; disabled/uninitialized utilization is undefined. */
        json tier(bool enabled, uint64_t capacity, const std::optional<Snapshot> &value)
        {
            const bool initialized = enabled && value.has_value();
            json result{{"enabled", enabled}, {"initialized", initialized},
                {"capacity_bytes", enabled ? capacity : 0},
                {"used_bytes", initialized ? json(value->used_bytes) : enabled ? json(nullptr) : json(0)},
                {"entries", initialized ? json(value->entries) : enabled ? json(nullptr) : json(0)},
                {"utilization_percent", initialized && capacity ?
                    json(100.0 * static_cast<double>(value->used_bytes) / capacity) : json(nullptr)},
                {"revision", initialized ? json(value->revision) : json(nullptr)},
                {"observation_age_seconds", initialized ? json(std::chrono::duration<double>(
                    PrefixCacheTierTelemetry::Clock::now() - value->updated_at).count()) : json(nullptr)}};
            return result;
        }

        /** @brief Aggregate payload capacity by summing bytes, never averaging percentages. */
        json aggregate(json instances, const char *basis)
        {
            bool enabled = false, initialized = true;
            uint64_t capacity = 0, used = 0, entries = 0;
            for (const auto &instance : instances)
            {
                if (!instance.at("enabled").get<bool>()) continue;
                enabled = true;
                initialized &= instance.at("initialized").get<bool>();
                capacity += instance.at("capacity_bytes").get<uint64_t>();
                if (!instance.at("used_bytes").is_null()) used += instance.at("used_bytes").get<uint64_t>();
                if (!instance.at("entries").is_null()) entries += instance.at("entries").get<uint64_t>();
            }
            auto result = tier(enabled, capacity, initialized ?
                std::optional<Snapshot>{{.used_bytes = used, .entries = entries}} : std::nullopt);
            // Freshness belongs to individual publishers: a fresh sibling must
            // not disguise another archive's older process-local index frontier.
            result.erase("revision");
            result.erase("observation_age_seconds");
            result["occupancy_basis"] = basis;
            result["instances"] = std::move(instances);
            return result;
        }

        /** @return One completed event count and its exact payload extent. */
        json event(const PrefixTierActivity &activity, PrefixTierEvent kind)
        {
            const auto i = static_cast<size_t>(kind);
            return {{"operations", activity.operations[i]}, {"bytes", activity.bytes[i]}};
        }
    }

    HttpPrefixCacheStats::HttpPrefixCacheStats(PrefixCacheTelemetrySources sources)
        : sources_(std::move(sources))
    {
        for (const auto &source : sources_)
            if (!source.publisher || (source.disk_enabled && source.disk_namespace.empty()))
                throw std::invalid_argument("prefix stats source lacks its publication authority or archive namespace");
        reset();
    }

    HttpPrefixCacheStats::Reading HttpPrefixCacheStats::read() const
    {
        Reading reading;
        json ram = json::array(), disk = json::array();
        std::set<const PrefixCacheTelemetry *> participants;
        std::set<const PrefixCacheTierTelemetry *> ram_owners;
        struct Archive
        {
            uint64_t capacity = 0;
            std::shared_ptr<const PrefixCacheTierTelemetry> owner;
            json participants = json::array();
        };
        std::map<std::string, Archive> archives;
        for (const auto &source : sources_)
        {
            if (!participants.insert(source.publisher.get()).second) continue;
            const auto owners = source.publisher->sources();
            std::optional<Snapshot> value;
            if (source.ram_enabled && owners.ram)
            {
                if (!ram_owners.insert(owners.ram.get()).second)
                    throw std::logic_error("distinct prefix participants share a RAM publication authority");
                value = owners.ram->snapshot();
                if (value->capacity_bytes != source.ram_capacity_bytes)
                    throw std::logic_error("RAM telemetry disagrees with admitted prefix capacity");
                reading.ram += value->activity;
            }
            auto instance = tier(source.ram_enabled, source.ram_capacity_bytes, value);
            instance["participant"] = source.participant;
            ram.push_back(std::move(instance));
            if (!source.disk_enabled) continue;
            auto [found, inserted] = archives.try_emplace(source.disk_namespace);
            auto &archive = found->second;
            if (!inserted && archive.capacity != source.disk_capacity_bytes)
                throw std::logic_error("shared prefix archive has conflicting capacity");
            archive.capacity = source.disk_capacity_bytes;
            archive.participants.push_back(source.participant);
            if (owners.disk)
            {
                if (archive.owner && archive.owner != owners.disk)
                    throw std::logic_error("shared prefix archive has multiple publication authorities");
                archive.owner = owners.disk;
            }
        }
        std::map<const PrefixCacheTierTelemetry *, size_t> disk_owners;
        for (const auto &[identity, archive] : archives)
        {
            (void)identity;
            std::optional<Snapshot> value;
            if (archive.owner)
            {
                const auto [owner, inserted] = disk_owners.emplace(archive.owner.get(), disk.size());
                if (!inserted)
                {
                    auto &existing = disk.at(owner->second);
                    if (existing.at("capacity_bytes").get<uint64_t>() != archive.capacity)
                        throw std::logic_error("prefix archive aliases have conflicting capacity");
                    for (const auto &participant : archive.participants)
                        existing["participants"].push_back(participant);
                    continue; // The storage publisher, not a pathname spelling, is identity.
                }
                value = archive.owner->snapshot();
                if (value->capacity_bytes != archive.capacity)
                    throw std::logic_error("disk telemetry disagrees with configured prefix capacity");
                reading.disk += value->activity;
            }
            auto instance = tier(true, archive.capacity, value);
            instance["participants"] = archive.participants;
            disk.push_back(std::move(instance));
        }
        reading.tiers = {{"ram", aggregate(std::move(ram), "installed_payload_bytes")},
                         {"disk", aggregate(std::move(disk), "active_archive_payload_bytes")}};
        return reading;
    }

    void HttpPrefixCacheStats::reset()
    {
        const auto reading = read();
        ram_baseline_ = reading.ram;
        disk_baseline_ = reading.disk;
    }

    nlohmann::json HttpPrefixCacheStats::snapshot() const
    {
        auto reading = read();
        const auto ram = reading.ram.since(ram_baseline_);
        const auto disk = reading.disk.since(disk_baseline_);
        return {{"scope", "serving_rank"}, {"tiers", std::move(reading.tiers)},
            {"churn", {{"scope", "committed_operations_since_last_reset"},
                {"ram_to_disk", event(ram, PrefixTierEvent::Demotion)},
                {"disk_to_ram", event(disk, PrefixTierEvent::Read)},
                {"ram_evictions", event(ram, PrefixTierEvent::Eviction)},
                {"disk_evictions", event(disk, PrefixTierEvent::Eviction)},
                {"disk_payload_writes", event(disk, PrefixTierEvent::Write)},
                {"disk_backing_reuses", event(disk, PrefixTierEvent::BackingReuse)}}}};
    }
}
