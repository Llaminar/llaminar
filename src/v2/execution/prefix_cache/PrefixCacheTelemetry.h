/**
 * @file PrefixCacheTelemetry.h
 * @brief Passive cache occupancy and committed tier traffic, independent of admission.
 *
 * Storage owners publish small metadata snapshots at their existing mutation
 * boundaries. HTTP readers never enter cache policy, inspect payloads, query a
 * device, or acquire an archive I/O lock. These observations cannot authorize
 * allocation: PMA, the RAM arena and the disk index retain that authority.
 */
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Committed operations; touches carry no payload-write bytes. */
    enum class PrefixTierEvent : size_t { Write, Read, BackingReuse, Demotion, Eviction, Count };

    /** @brief Monotonic model-lifetime events, with exact live payload extents. */
    struct PrefixTierActivity
    {
        static constexpr size_t count = static_cast<size_t>(PrefixTierEvent::Count);
        std::array<uint64_t, count> operations{}, bytes{};

        /** @brief Combine distinct storage owners, never duplicate participant mirrors. */
        PrefixTierActivity &operator+=(const PrefixTierActivity &other)
        {
            for (size_t i = 0; i < count; ++i)
            {
                operations[i] += other.operations[i];
                bytes[i] += other.bytes[i];
            }
            return *this;
        }

        /** @return Activity since an earlier observation of the same model lifetime. */
        [[nodiscard]] PrefixTierActivity since(const PrefixTierActivity &earlier) const
        {
            PrefixTierActivity delta;
            for (size_t i = 0; i < count; ++i)
            {
                if (operations[i] < earlier.operations[i] || bytes[i] < earlier.bytes[i])
                    throw std::logic_error("prefix telemetry lifetime counters regressed");
                delta.operations[i] = operations[i] - earlier.operations[i];
                delta.bytes[i] = bytes[i] - earlier.bytes[i];
            }
            return delta;
        }
    };

    /** @brief Constant-size, host-only projection of one storage authority. */
    class PrefixCacheTierTelemetry final
    {
    public:
        using Clock = std::chrono::steady_clock;
        /** @brief Indexed payload occupancy; no allocation capacity or file-history bytes. */
        struct Snapshot
        {
            uint64_t capacity_bytes = 0, used_bytes = 0, entries = 0;
            uint64_t revision = 0;
            Clock::time_point updated_at = Clock::now();
            PrefixTierActivity activity;
        };

        /** @brief Bind the configured payload envelope, without reserving physical memory. */
        explicit PrefixCacheTierTelemetry(uint64_t capacity) { value_.capacity_bytes = capacity; }

        /** @brief Publish an owner's existing index totals without revisiting any payload. */
        void publishUsage(uint64_t used, uint64_t entries)
        {
            std::lock_guard lock(mutex_);
            value_.used_bytes = used;
            value_.entries = entries;
            changed();
        }

        /** @brief Count a completed operation and its actual payload extent exactly once. */
        void record(PrefixTierEvent event, uint64_t bytes)
        {
            const auto index = static_cast<size_t>(event);
            if (index >= PrefixTierActivity::count)
                throw std::invalid_argument("invalid prefix tier event");
            std::lock_guard lock(mutex_);
            ++value_.activity.operations[index];
            value_.activity.bytes[index] += bytes;
            // A completed demotion is also one RAM capacity eviction. Publish
            // both facts together so a concurrent reset cannot split one event.
            if (event == PrefixTierEvent::Demotion)
            {
                const auto eviction = static_cast<size_t>(PrefixTierEvent::Eviction);
                ++value_.activity.operations[eviction];
                value_.activity.bytes[eviction] += bytes;
            }
            changed();
        }

        /** @return A coherent metadata copy; no I/O, GPU query, maintenance or cache lock. */
        [[nodiscard]] Snapshot snapshot() const
        {
            std::lock_guard lock(mutex_);
            return value_;
        }

    private:
        /** @brief Advance publication freshness while the metadata lock is held. */
        void changed() { ++value_.revision; value_.updated_at = Clock::now(); }
        mutable std::mutex mutex_;
        Snapshot value_;
    };

    /** @brief Publish lazy storage initialization without exposing mutable runner pointers. */
    class PrefixCacheTelemetry final
    {
    public:
        /** @brief Shared disk publishers identify the same archive across local TP/PP children. */
        struct Sources
        {
            std::shared_ptr<const PrefixCacheTierTelemetry> ram, disk;
        };

        /** @brief Bind each model-lifetime storage authority once, before its first request. */
        void bind(std::shared_ptr<const PrefixCacheTierTelemetry> ram,
                  std::shared_ptr<const PrefixCacheTierTelemetry> disk)
        {
            if (!ram) throw std::invalid_argument("prefix telemetry requires its RAM owner");
            std::lock_guard lock(mutex_);
            if (sources_.ram || sources_.disk)
                throw std::logic_error("prefix telemetry storage authority already bound");
            sources_ = {std::move(ram), std::move(disk)};
        }

        /** @return Retained metadata publishers; their lifetime does not retain cache payloads. */
        [[nodiscard]] Sources sources() const
        {
            std::lock_guard lock(mutex_);
            return sources_;
        }

    private:
        mutable std::mutex mutex_;
        Sources sources_;
    };

    /** @brief Frozen startup geometry plus a passive source for lazy cache initialization. */
    struct PrefixCacheParticipantTelemetry
    {
        std::string participant;
        bool ram_enabled = false, disk_enabled = false;
        uint64_t ram_capacity_bytes = 0, disk_capacity_bytes = 0;
        /** Configured archive namespace; deduplicates capacity before lazy initialization. */
        std::string disk_namespace;
        std::shared_ptr<const PrefixCacheTelemetry> publisher;
    };
    using PrefixCacheTelemetrySources = std::vector<PrefixCacheParticipantTelemetry>;
}
