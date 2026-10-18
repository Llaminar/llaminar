/**
 * @file PrefixCacheShutdown.cpp
 * @brief Coordinate bounded terminal persistence without transferring KV bytes.
 *
 * Rank zero plans from compact live candidate metadata. Each source retains its
 * original RAM leases; only that rank can serialize them. Archive selections
 * protect all chosen incarnations, so writer order cannot discard a peer's
 * higher-priority RAM entry. Cold disk entries consume only the remaining space.
 */
#include "execution/prefix_cache/PrefixCacheShutdown.h"

#include "execution/prefix_cache/PrefixStateCache.h"
#include "interfaces/IMPIContext.h"
#include "utils/PerfStatsCollector.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace llaminar2
{
    namespace
    {
        using Json = nlohmann::json;
        constexpr int kMetadataLengthTag = 27120;
        constexpr int kMetadataBodyTag = 27121;
        constexpr int kSelectionLengthTag = 27122;
        constexpr int kSelectionBodyTag = 27123;

        /**
         * @brief Encode exact source metadata without pointers or payload content.
         * @param handle Original RAM owner supplying key, immutable identity and live bytes.
         * @return Compact serializable candidate metadata.
         * @throws std::exception If metadata allocation fails.
         */
        Json blockJson(const PrefixBlockHandle &handle)
        {
            const auto &key = handle.key;
            return {{"key", {key.fingerprint, key.parent_hash, key.token_hash,
                             key.block_index, key.token_start, key.token_count}},
                    {"identity", {handle.payload_identity.high, handle.payload_identity.low}},
                    {"bytes", handle.total_bytes}};
        }

        /**
         * @brief Parse typed, bounded metadata before it reaches the disk authority.
         * @param value Candidate key, incarnation and exact serialized extent.
         * @return Validated immutable selection metadata.
         * @throws std::invalid_argument For invalid geometry, identity or zero extent.
         * @throws Json::exception For missing fields or incompatible numeric encodings.
         */
        PrefixShutdownBlock parseBlock(const Json &value)
        {
            const auto &key = value.at("key");
            const auto &identity = value.at("identity");
            if (key.size() != 6 || identity.size() != 2)
                throw std::invalid_argument("invalid prefix shutdown metadata geometry");
            PrefixShutdownBlock result{
                .key = {.fingerprint = key.at(0).get<uint64_t>(), .parent_hash = key.at(1).get<uint64_t>(),
                        .token_hash = key.at(2).get<uint64_t>(), .block_index = key.at(3).get<int>(),
                        .token_start = key.at(4).get<int>(), .token_count = key.at(5).get<int>()},
                .payload_identity = {identity.at(0).get<uint64_t>(), identity.at(1).get<uint64_t>()},
                .bytes = value.at("bytes").get<uint64_t>()};
            if (!result.key.valid() || result.key.token_start < 0 ||
                !result.payload_identity.valid() || result.bytes == 0)
                throw std::invalid_argument("invalid prefix shutdown block identity or extent");
            return result;
        }

        /**
         * @brief Read stable shared-file identity independently of host path spelling.
         * @param value Two unsigned identity components from native archive initialization.
         * @return Nonempty identity grouping all writers of one physical archive.
         * @throws std::invalid_argument For malformed or empty identity metadata.
         * @throws Json::exception For incompatible identity component encodings.
         */
        PrefixArchiveIdentity parseArchive(const Json &value)
        {
            if (!value.is_array() || value.size() != 2)
                throw std::invalid_argument("invalid prefix shutdown archive identity");
            PrefixArchiveIdentity result{value.at(0).get<uint64_t>(), value.at(1).get<uint64_t>()};
            if (!result.valid()) throw std::invalid_argument("empty prefix shutdown archive identity");
            return result;
        }

        /**
         * @brief Send live metadata to its sole consumer using exact byte extents.
         * @param context Complete inference communicator, still live during terminal selection.
         * @param value Exact candidate inventory or relevant writer selection.
         * @param peer Sole rank consuming this metadata body.
         * @param length_tag Protocol tag for the immutable byte count.
         * @param body_tag Protocol tag for the immediately following compact body.
         * @throws std::overflow_error If serialized metadata exceeds the MPI count range.
         * @throws std::exception For serialization allocation or native send failure.
         */
        void sendJson(const IMPIContext &context, const Json &value, int peer, int length_tag, int body_tag)
        {
            const std::string bytes = value.dump();
            if (bytes.size() > INT_MAX)
                throw std::overflow_error("prefix shutdown metadata exceeds the MPI count range");
            const uint64_t size = bytes.size();
            context.send(&size, 1, MPI_UINT64_T, peer, length_tag);
            context.send(bytes.data(), bytes.size(), MPI_BYTE, peer, body_tag);
        }

        /**
         * @brief Receive only announced live metadata, rejecting invalid framing.
         * @param context Complete inference communicator, still live during terminal selection.
         * @param peer Sole metadata producer for this protocol edge.
         * @param length_tag Protocol tag for the immutable byte count.
         * @param body_tag Protocol tag for the compact metadata body.
         * @return Parsed metadata retaining precisely the announced live extent.
         * @throws std::overflow_error For empty framing or counts outside MPI's range.
         * @throws std::exception For native receive, allocation or JSON parse failure.
         */
        Json receiveJson(const IMPIContext &context, int peer, int length_tag, int body_tag)
        {
            uint64_t size = 0;
            context.recv(&size, 1, MPI_UINT64_T, peer, length_tag);
            if (size == 0 || size > INT_MAX)
                throw std::overflow_error("invalid prefix shutdown MPI metadata extent");
            std::string bytes(static_cast<size_t>(size), '\0');
            context.recv(bytes.data(), bytes.size(), MPI_BYTE, peer, body_tag);
            return Json::parse(bytes);
        }

        /** @brief One immutable source's local retention position and original owner. */
        struct Candidate
        {
            int rank;
            size_t source;
            size_t recency;
            PrefixShutdownBlock block;
            Json metadata;
        };

        /**
         * @brief Select once per shared archive before any participant starts writing.
         * @param manifests Actual candidate inventories indexed by communicator rank.
         * @return Per-rank plans containing only archives with a selected local writer.
         * @throws std::invalid_argument For invalid inventories or conflicting shared budgets.
         * @throws std::exception For malformed metadata or selection allocation failure.
         *
         * Local recency positions interleave because participant ticks are not
         * globally comparable. The disk authority remains the sole live ledger;
         * this terminal immutable selection only defines which owners it protects.
         */
        std::vector<Json> select(const std::vector<Json> &manifests)
        {
            struct Archive
            {
                uint64_t budget;
                std::vector<Candidate> candidates;
            };
            std::map<PrefixArchiveIdentity, Archive> archives;
            for (size_t rank = 0; rank < manifests.size(); ++rank)
            {
                const auto &sources = manifests[rank];
                if (!sources.is_array()) throw std::invalid_argument("invalid prefix shutdown source inventory");
                for (size_t source = 0; source < sources.size(); ++source)
                {
                    const auto &row = sources[source];
                    if (row.is_null()) continue; // This cache has no durable tier.
                    const auto identity = parseArchive(row.at("archive"));
                    const auto budget = row.at("budget").get<uint64_t>();
                    auto [iterator, inserted] = archives.try_emplace(identity, Archive{budget, {}});
                    if (iterator->second.budget != budget)
                        throw std::invalid_argument("prefix shutdown participants disagree on shared archive capacity");
                    const auto &entries = row.at("entries");
                    if (!entries.is_array()) throw std::invalid_argument("invalid prefix shutdown candidate inventory");
                    for (size_t recency = 0; recency < entries.size(); ++recency)
                        iterator->second.candidates.push_back({static_cast<int>(rank), source, recency,
                            parseBlock(entries[recency]), entries[recency]});
                }
            }
            std::vector<Json> plans(manifests.size(), Json::array());
            for (auto &[identity, archive] : archives)
            {
                auto &candidates = archive.candidates;
                std::stable_sort(candidates.begin(), candidates.end(), [](const auto &left, const auto &right) {
                    return std::tie(left.recency, left.rank, left.source) <
                           std::tie(right.recency, right.rank, right.source);
                });
                Json retained = Json::array();
                std::vector<Json> owned(manifests.size(), Json::array());
                std::unordered_set<PrefixCacheKey, PrefixCacheKeyHasher> seen;
                uint64_t selected_bytes = 0;
                for (const auto &candidate : candidates)
                {
                    if (!seen.insert(candidate.block.key).second) continue;
                    // Oversized entries and lower-priority entries which do
                    // not fit are not written just to be evicted immediately.
                    if (candidate.block.bytes > std::numeric_limits<uint64_t>::max() - selected_bytes ||
                        (archive.budget != 0 && candidate.block.bytes > archive.budget - selected_bytes))
                        continue;
                    selected_bytes += candidate.block.bytes;
                    retained.push_back(candidate.metadata);
                    owned.at(candidate.rank).push_back({candidate.source, candidate.recency});
                }
                for (size_t rank = 0; rank < plans.size(); ++rank)
                {
                    if (owned[rank].empty()) continue;
                    plans[rank].push_back({{"archive", {identity.high, identity.low}}, {"budget", archive.budget},
                                          {"retained", retained}, {"owned", owned[rank]}});
                }
            }
            return plans;
        }
    }

    /** @copydoc PrefixCacheShutdown::add */
    void PrefixCacheShutdown::add(std::shared_ptr<PrefixStateCache> cache)
    {
        if (phase_ != Phase::Collecting)
            throw std::logic_error("prefix shutdown source collection is closed");
        if (cache && std::find(caches_.begin(), caches_.end(), cache) == caches_.end())
            caches_.push_back(std::move(cache));
    }

    /** @copydoc PrefixCacheShutdown::flush */
    PrefixCacheShutdownResult PrefixCacheShutdown::flush(const IMPIContext *context)
    {
        if (phase_ != Phase::Collecting)
            throw std::logic_error("prefix shutdown flush can execute only once");
        phase_ = Phase::Flushing;
        PrefixCacheShutdownResult result;
        std::vector<PrefixCacheShutdownSource> sources;
        Json manifest = Json::array();
        for (const auto &cache : caches_)
        {
            sources.push_back(cache->sealForShutdown());
            const auto &source = sources.back();
            if (!source.archive) { manifest.push_back(nullptr); continue; }
            const auto identity = source.archive->shutdownIdentity();
            Json entries = Json::array();
            for (const auto &handle : source.most_recent_first) entries.push_back(blockJson(handle));
            result.candidates += source.most_recent_first.size();
            manifest.push_back({{"archive", {identity.high, identity.low}},
                                {"budget", source.archive->budgetBytes()}, {"entries", std::move(entries)}});
        }

        Json plan;
        const int rank = context ? context->rank() : 0;
        const int ranks = context ? context->world_size() : 1;
        if (ranks <= 0 || rank < 0 || rank >= ranks)
            throw std::invalid_argument("invalid prefix shutdown communicator");
        if (rank == 0)
        {
            std::vector<Json> manifests{std::move(manifest)};
            for (int peer = 1; peer < ranks; ++peer)
                manifests.push_back(receiveJson(*context, peer, kMetadataLengthTag, kMetadataBodyTag));
            const auto plans = select(manifests);
            for (int peer = 1; peer < ranks; ++peer)
                sendJson(*context, plans.at(peer), peer, kSelectionLengthTag, kSelectionBodyTag);
            plan = plans.front();
        }
        else
        {
            sendJson(*context, manifest, 0, kMetadataLengthTag, kMetadataBodyTag);
            plan = receiveJson(*context, 0, kSelectionLengthTag, kSelectionBodyTag);
        }
        for (const auto &row : plan)
        {
            PrefixShutdownArchivePlan selection{.archive = parseArchive(row.at("archive")),
                .budget_bytes = row.at("budget").get<uint64_t>()};
            for (const auto &block : row.at("retained")) selection.retained.push_back(parseBlock(block));
            std::shared_ptr<DiskPrefixStorageBackend> archive;
            std::vector<PrefixBlockHandle> selected;
            for (const auto &owner : row.at("owned"))
            {
                const auto &source = sources.at(owner.at(0).get<size_t>());
                if (!source.archive || source.archive->shutdownIdentity() != selection.archive)
                    throw std::invalid_argument("prefix shutdown planner selected a different archive owner");
                archive = source.archive;
                selected.push_back(source.most_recent_first.at(owner.at(1).get<size_t>()));
            }
            if (!archive || selected.empty())
                throw std::invalid_argument("prefix shutdown planner sent an empty writer selection");
            const auto completed = archive->flushShutdown(selected, selection);
            result.selected += completed.selected;
            result.written += completed.written;
            result.reused += completed.reused;
            result.payload_bytes_written += completed.payload_bytes_written;
        }
        PerfStatsCollector::addCounter("prefix_cache", "shutdown_ram_candidates", result.candidates, "shutdown");
        PerfStatsCollector::addCounter("prefix_cache", "shutdown_ram_selected", result.selected, "shutdown");
        PerfStatsCollector::addCounter("prefix_cache", "shutdown_payload_writes", result.written, "shutdown");
        PerfStatsCollector::addCounter("prefix_cache", "shutdown_backing_reuses", result.reused, "shutdown");
        PerfStatsCollector::addCounter("prefix_cache", "shutdown_payload_bytes_written", result.payload_bytes_written, "shutdown");
        phase_ = Phase::Complete;
        return result;
    }
}
