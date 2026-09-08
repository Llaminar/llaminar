/**
 * @file NativeVNNIDispatchCache.h
 * @brief Allocation-free host cache for immutable NativeVNNI dispatch policy.
 *
 * Generated NativeVNNI selectors are pure functions of backend-visible launch
 * geometry. Exact overlays commonly perform a binary search, while generic
 * policy may traverse several learned predicates. Graph replay repeatedly asks
 * for the same small set of decisions, so paying that search on every token is
 * unnecessary host overhead.
 *
 * This cache deliberately owns no policy logic. Callers retain a complete key,
 * invoke their generated selector on a miss, and store both successful and
 * unsuccessful results. Production wrappers instantiate the cache as
 * `thread_local`, which keeps lookup lock-free without putting mutable state in
 * packed weights or graph-owned kernel objects.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaminar2::native_vnni
{
    /**
     * @brief Fast deterministic hash for one integral dispatch-key word.
     *
     * The MRU path performs no hashing. This mixer is paid only when the key
     * changes, where its avalanche behavior prevents aligned N/K dimensions
     * from collapsing into one direct-mapped slot.
     */
    inline constexpr uint64_t mixDispatchCacheWord(uint64_t value) noexcept
    {
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31;
        return value;
    }

    /** @brief Hash an unsigned packed key. */
    struct DispatchCacheWordHasher
    {
        inline constexpr size_t operator()(uint64_t key) const noexcept
        {
            return static_cast<size_t>(mixDispatchCacheWord(key));
        }
    };

    /**
     * @brief Hash a complete multi-word geometry without lossy repacking.
     * @tparam WordCount Number of independent identity words.
     */
    template <size_t WordCount>
    struct DispatchCacheArrayHasher
    {
        inline constexpr size_t operator()(
            const std::array<uint64_t, WordCount> &key) const noexcept
        {
            /*
             * Fold every identity word first, then avalanche once. A cache
             * collision is harmless because lookup still compares the entire
             * key; the hash only chooses a direct-mapped slot. Performing a
             * full splitmix round for every word made rotating projection
             * dispatch needlessly expensive on the host.
             */
            uint64_t hash = key[0] + 0x9e3779b97f4a7c15ULL;
            for (size_t index = 0; index < WordCount; ++index)
            {
                hash ^= key[index] + 0x9e3779b97f4a7c15ULL +
                        (hash << 6) + (hash >> 2);
            }
            return static_cast<size_t>(mixDispatchCacheWord(hash));
        }
    };

    /**
     * @brief Tiny MRU plus direct-mapped cache for a pure dispatch selector.
     * @tparam Key Complete immutable selector identity.
     * @tparam Value Trivially copied selector result.
     * @tparam Capacity Power-of-two direct-mapped working-set capacity.
     * @tparam Hasher Stable hash functor for `Key`.
     *
     * `lookup()` returns whether the cache contained the key. Its
     * `selector_found` output separately preserves whether the underlying
     * generated selector accepted that key, allowing hard misses to be cached
     * without turning them into a policy fallback.
     */
    template <
        typename Key,
        typename Value,
        size_t Capacity,
        typename Hasher>
    class FixedDispatchCache final
    {
        static_assert(Capacity > 0, "dispatch cache cannot be empty");
        static_assert(
            (Capacity & (Capacity - 1)) == 0,
            "dispatch cache capacity must be a power of two");
        static_assert(
            std::is_trivially_copyable_v<Key>,
            "dispatch cache keys must be trivially copyable");
        static_assert(
            std::is_trivially_copyable_v<Value>,
            "dispatch cache values must be trivially copyable");

        struct Entry
        {
            Key key{};
            Value value{};
            bool selector_found = false;
            bool valid = false;

            inline bool matches(const Key &expected) const noexcept
            {
                return valid && key == expected;
            }
        };

    public:
        /**
         * @brief View one cached selector result without copying its value.
         * @return Stable cache-owned value until a later insertion overwrites
         * the same direct-mapped slot, or `nullptr` for a cache miss.
         *
         * Callers must consume the value immediately and must not retain the
         * pointer across another cache operation. This surface matters for
         * CUDA policy records, where copying the complete tuning object into a
         * temporary and then into launch locals doubled hot dispatch overhead.
         */
        inline const Value *lookupValue(
            const Key &key,
            bool &selector_found) noexcept
        {
            if (most_recent_ != nullptr && most_recent_->matches(key))
            {
                selector_found = most_recent_->selector_found;
                return &most_recent_->value;
            }

            Entry &entry = entries_[Hasher{}(key) & (Capacity - 1)];
            if (!entry.matches(key))
                return nullptr;

            most_recent_ = &entry;
            selector_found = entry.selector_found;
            return &entry.value;
        }

        /**
         * @brief Read a cached selector result without invoking policy code.
         * @return `true` for a cache hit, including a cached selector miss.
         */
        inline bool lookup(
            const Key &key,
            bool &selector_found,
            Value &value) noexcept
        {
            if (const Value *cached = lookupValue(key, selector_found))
            {
                value = *cached;
                return true;
            }
            return false;
        }

        /** @brief Publish one generated-selector result into both cache levels. */
        inline void insert(
            const Key &key,
            bool selector_found,
            const Value &value) noexcept
        {
            Entry entry{
                .key = key,
                .value = value,
                .selector_found = selector_found,
                .valid = true,
            };
            Entry &slot = entries_[Hasher{}(key) & (Capacity - 1)];
            slot = entry;
            most_recent_ = &slot;
        }

    private:
        std::array<Entry, Capacity> entries_{};
        Entry *most_recent_ = nullptr;
    };
}
