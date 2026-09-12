/**
 * @file MoEOverlayRankBatchTransport.cpp
 * @brief Allocation-free codec and direct MPI exchange for rank-batched MoE rows.
 *
 * The implementation deliberately uses an explicit field-by-field wire ABI.
 * C++ structure padding is never transmitted, and the receiver authenticates
 * the complete transaction key and canonical participant order before it
 * publishes a single live row.  This makes a stale graph, mismatched topology,
 * or out-of-order MTP transaction a hard protocol failure rather than a silent
 * numerical corruption.
 */

#include "MoEOverlayRankBatchTransport.h"
#include "MoEOverlayRankBatchTelemetry.h"

#include "collective/CollectiveTimeoutPolicy.h"
#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace llaminar2
{
    MoEOverlaySparseRows
    IMoEOverlayRankBatchTransport::sharedDispatchRows(int) const
    {
        throw std::logic_error(
            "rank-batch transport does not own node-local shared dispatch rows");
    }

    MoEOverlayReturnRows
    IMoEOverlayRankBatchTransport::sharedReturnRows(int) const
    {
        throw std::logic_error(
            "rank-batch transport does not own node-local shared return rows");
    }

    std::string makeMoEOverlayRankBatchChannelIdentity(
        int tier_index,
        int domain_ordinal,
        int source_world_rank,
        int target_world_rank,
        std::span<const int> participant_ids)
    {
        if (tier_index < 0 || domain_ordinal < 0 ||
            source_world_rank < 0 || target_world_rank < 0 ||
            source_world_rank == target_world_rank || participant_ids.empty())
        {
            throw std::invalid_argument(
                "MoE rank-batch channel identity requires complete remote-rank topology");
        }
        if (participant_ids.front() < 0 ||
            !std::is_sorted(participant_ids.begin(), participant_ids.end()) ||
            std::adjacent_find(
                participant_ids.begin(), participant_ids.end()) !=
                participant_ids.end())
        {
            throw std::invalid_argument(
                "MoE rank-batch channel participant ids must be strictly increasing and non-negative");
        }

        std::ostringstream identity;
        identity << "tier" << tier_index
                 << "#domain" << domain_ordinal
                 << "#rank" << source_world_rank << "to"
                 << target_world_rank << "#p";
        for (const int participant_id : participant_ids)
            identity << participant_id << ',';
        return identity.str();
    }

    void MoEOverlayRankBatchTransportRegistry::install(
        std::string channel_identity,
        std::shared_ptr<IMoEOverlayRankBatchTransport> transport)
    {
        if (channel_identity.empty() || !transport ||
            transport->sourceWorldRank() < 0 ||
            transport->targetWorldRank() < 0 ||
            transport->sourceWorldRank() == transport->targetWorldRank() ||
            transport->participantIds().empty())
        {
            throw std::invalid_argument(
                "MoE rank-batch registry requires a named, complete transport");
        }

        /*
         * Publication is one-way: a graph may retain the returned shared_ptr
         * immediately after setup. Replacing an entry later would leave two
         * live driver-registration authorities for the same shared mapping.
         */
        std::scoped_lock lock(mutex_);
        const auto [_, inserted] = transports_.emplace(
            std::move(channel_identity), std::move(transport));
        if (!inserted)
        {
            throw std::logic_error(
                "MoE rank-batch transport identity was installed more than once");
        }
    }

    std::shared_ptr<IMoEOverlayRankBatchTransport>
    MoEOverlayRankBatchTransportRegistry::require(
        const std::string &channel_identity,
        int source_world_rank,
        int target_world_rank,
        std::span<const int> participant_ids) const
    {
        std::scoped_lock lock(mutex_);
        const auto found = transports_.find(channel_identity);
        if (found == transports_.end() || !found->second)
        {
            throw std::logic_error(
                "MoE rank-batch graph requested a channel that preflight did not create: " +
                channel_identity);
        }
        const auto &transport = found->second;
        if (transport->sourceWorldRank() != source_world_rank ||
            transport->targetWorldRank() != target_world_rank ||
            !std::equal(
                participant_ids.begin(),
                participant_ids.end(),
                transport->participantIds().begin(),
                transport->participantIds().end()))
        {
            throw std::logic_error(
                "MoE rank-batch preflight channel topology diverged from graph construction: " +
                channel_identity);
        }
        return transport;
    }

    std::size_t MoEOverlayRankBatchTransportRegistry::size() const
    {
        std::scoped_lock lock(mutex_);
        return transports_.size();
    }

    namespace
    {
        constexpr uint32_t kBatchMagic = 0x42454f4dU; // "MOEB" in little-endian memory.
        constexpr uint32_t kBatchVersion = 3u; // Return records authenticate arithmetic/row identity.
        constexpr uint8_t kDispatchEnvelopeKind = 1u;
        constexpr uint8_t kReturnEnvelopeKind = 2u;

        /* MPI guarantees that MPI_TAG_UB is at least 32767. */
        constexpr int kDispatchTag = 32740;
        constexpr int kReturnTag = 32741;

        constexpr size_t kEnvelopeCapacityAllowance = 512u;
        constexpr size_t kDispatchSubrecordCapacityAllowance = 128u;
        constexpr size_t kReturnSubrecordCapacityAllowance = 96u;

        /** @brief Multiply setup-time capacities without wrapping size_t. */
        size_t checkedMultiply(size_t lhs, size_t rhs, const char *what)
        {
            if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
            {
                throw std::overflow_error(
                    std::string("MoE rank-batch wire capacity overflow: ") + what);
            }
            return lhs * rhs;
        }

        /** @brief Add setup-time capacities without wrapping size_t. */
        size_t checkedAdd(size_t lhs, size_t rhs, const char *what)
        {
            if (rhs > std::numeric_limits<size_t>::max() - lhs)
            {
                throw std::overflow_error(
                    std::string("MoE rank-batch wire capacity overflow: ") + what);
            }
            return lhs + rhs;
        }

        /** @brief Fixed-buffer writer used by the hot-path packet codec. */
        class FixedWriter final
        {
        public:
            /** @brief Bind one preallocated output buffer. */
            explicit FixedWriter(std::span<std::byte> storage)
                : storage_(storage)
            {
            }

            /** @brief Append one trivially-copyable scalar if capacity permits. */
            template <typename T>
            bool append(const T &value) noexcept
            {
                static_assert(std::is_trivially_copyable_v<T>);
                return appendBytes(&value, sizeof(T));
            }

            /** @brief Append raw bytes without allocating or changing the base address. */
            bool appendBytes(const void *source, size_t bytes) noexcept
            {
                if (bytes > remaining() || (bytes != 0 && !source))
                    return false;
                if (bytes != 0)
                {
                    std::memcpy(storage_.data() + offset_, source, bytes);
                    offset_ += bytes;
                }
                return true;
            }

            /** @return Number of bytes already written. */
            size_t size() const noexcept { return offset_; }

            /** @return Remaining capacity at the fixed address. */
            size_t remaining() const noexcept
            {
                return storage_.size() - offset_;
            }

        private:
            std::span<std::byte> storage_;
            size_t offset_ = 0;
        };

        /** @brief Bounds-checked reader used for both validation and publication passes. */
        class FixedReader final
        {
        public:
            /** @brief Bind the exact live prefix reported by MPI status. */
            explicit FixedReader(std::span<const std::byte> payload)
                : payload_(payload)
            {
            }

            /** @brief Read one trivially-copyable scalar. */
            template <typename T>
            bool read(T *value) noexcept
            {
                static_assert(std::is_trivially_copyable_v<T>);
                return readBytes(value, sizeof(T));
            }

            /** @brief Read raw bytes without advancing beyond the live prefix. */
            bool readBytes(void *destination, size_t bytes) noexcept
            {
                if (bytes > remaining() || (bytes != 0 && !destination))
                    return false;
                if (bytes != 0)
                {
                    std::memcpy(destination, payload_.data() + offset_, bytes);
                    offset_ += bytes;
                }
                return true;
            }

            /** @brief Skip bytes during the validation-only pass. */
            bool skip(size_t bytes) noexcept
            {
                if (bytes > remaining())
                    return false;
                offset_ += bytes;
                return true;
            }

            /** @return Bytes not yet consumed from the authenticated live prefix. */
            size_t remaining() const noexcept
            {
                return payload_.size() - offset_;
            }

        private:
            std::span<const std::byte> payload_;
            size_t offset_ = 0;
        };

        /** @brief Serialize the complete rank-batch key into a fixed writer. */
        bool writeKey(FixedWriter &writer, const MoEOverlayRankBatchKey &key) noexcept
        {
            const auto key_namespace = static_cast<uint8_t>(key.key_namespace);
            const auto histogram_source =
                static_cast<uint8_t>(key.histogram_source);
            const auto direction = static_cast<uint8_t>(key.direction);
            return writer.append(key_namespace) &&
                   writer.append(histogram_source) &&
                   writer.append(direction) &&
                   writer.append(key.generation_id) &&
                   writer.append(key.step_id) &&
                   writer.append(key.mtp_depth) &&
                   writer.append(key.layer_idx) &&
                   writer.append(key.tier_idx) &&
                   writer.append(key.domain_ordinal) &&
                   writer.append(key.source_world_rank) &&
                   writer.append(key.target_world_rank) &&
                   writer.append(key.sequence);
        }

        /** @brief Deserialize the complete rank-batch key from a fixed reader. */
        bool readKey(FixedReader &reader, MoEOverlayRankBatchKey *key) noexcept
        {
            if (!key)
                return false;
            uint8_t key_namespace = 0;
            uint8_t histogram_source = 0;
            uint8_t direction = 0;
            if (!reader.read(&key_namespace) ||
                !reader.read(&histogram_source) ||
                !reader.read(&direction) ||
                !reader.read(&key->generation_id) ||
                !reader.read(&key->step_id) ||
                !reader.read(&key->mtp_depth) ||
                !reader.read(&key->layer_idx) ||
                !reader.read(&key->tier_idx) ||
                !reader.read(&key->domain_ordinal) ||
                !reader.read(&key->source_world_rank) ||
                !reader.read(&key->target_world_rank) ||
                !reader.read(&key->sequence))
            {
                return false;
            }
            key->key_namespace =
                static_cast<MoEOverlayCollectiveNamespace>(key_namespace);
            key->histogram_source =
                static_cast<ExpertHistogramSource>(histogram_source);
            key->direction =
                static_cast<MoEOverlayCollectiveDirection>(direction);
            return true;
        }

        /** @brief Convert a batch identity into the participant-local key expected by local compute. */
        MoEOverlayCollectiveKey participantKey(
            const MoEOverlayRankBatchKey &batch,
            int participant_id)
        {
            MoEOverlayCollectiveKey result =
                batch.key_namespace == MoEOverlayCollectiveNamespace::MTP
                    ? makeMTPMoEOverlayCollectiveKey(
                          batch.generation_id,
                          batch.step_id,
                          batch.mtp_depth,
                          batch.layer_idx,
                          batch.tier_idx,
                          participant_id,
                          participant_id,
                          batch.direction)
                    : makeMoEOverlayCollectiveKey(
                          batch.generation_id,
                          batch.step_id,
                          batch.layer_idx,
                          batch.tier_idx,
                          participant_id,
                          participant_id,
                          batch.direction);
            result.histogram_source = batch.histogram_source;
            return result;
        }

        /** @brief Validate one sparse dispatch view without allocating diagnostic state. */
        bool validateDispatchRows(
            const MoEOverlaySparseRows &rows,
            int expected_participant,
            int d_model,
            int top_k,
            std::string *error)
        {
            if (rows.target_participant != expected_participant)
            {
                if (error)
                    *error = "dispatch subpacket target does not match canonical participant order";
                return false;
            }
            if (rows.source_participant < 0 || rows.d_model != d_model ||
                rows.top_k != top_k || !rows.row_ids_host ||
                !rows.entry_offsets_host || !rows.expert_ids_host ||
                !rows.route_weights_host ||
                !rows.original_route_slots_host ||
                !rows.compact_route_slots_host || !rows.hidden_rows_fp32)
            {
                if (error)
                    *error = "dispatch subpacket has invalid identity, geometry, or storage";
                return false;
            }
            if (rows.live_row_count > rows.row_capacity ||
                rows.live_entry_count > rows.entry_capacity)
            {
                if (error)
                    *error = "dispatch subpacket live counts exceed fixed capacity";
                return false;
            }
            if ((rows.live_row_count != 0 || rows.live_entry_count != 0) &&
                rows.residency_epoch == 0)
            {
                if (error)
                    *error = "non-empty dispatch subpacket has no residency epoch";
                return false;
            }
            if (rows.entry_offsets_host[0] != 0)
            {
                if (error)
                    *error = "dispatch subpacket entry offsets must begin at zero";
                return false;
            }
            for (size_t row = 0; row < rows.live_row_count; ++row)
            {
                const int32_t begin = rows.entry_offsets_host[row];
                const int32_t end = rows.entry_offsets_host[row + 1u];
                if (begin < 0 || end < begin ||
                    static_cast<size_t>(end) > rows.live_entry_count ||
                    end - begin > top_k)
                {
                    if (error)
                        *error = "dispatch subpacket has invalid entry offsets";
                    return false;
                }
            }
            if (static_cast<size_t>(rows.entry_offsets_host[rows.live_row_count]) !=
                rows.live_entry_count)
            {
                if (error)
                    *error = "dispatch subpacket terminal entry offset does not match live entries";
                return false;
            }
            return true;
        }

        /** @brief Validate one sparse return view without allocating diagnostic state. */
        bool validateReturnRows(
            const MoEOverlayReturnRows &rows,
            int expected_participant,
            int d_model,
            std::string *error)
        {
            if (rows.source_participant != expected_participant)
            {
                if (error)
                    *error = "return subpacket source does not match canonical participant order";
                return false;
            }
            if (rows.target_participant < 0 || rows.d_model != d_model ||
                !isValidMoEOverlayReturnLayout(rows.layout) ||
                !rows.row_ids_host || !rows.output_rows_fp32)
            {
                if (error)
                    *error = "return subpacket has invalid identity, geometry, or storage";
                return false;
            }
            if (rows.live_row_count > rows.row_capacity)
            {
                if (error)
                    *error = "return subpacket live count exceeds fixed capacity";
                return false;
            }
            if (rows.live_row_count != 0 && rows.residency_epoch == 0)
            {
                if (error)
                    *error = "non-empty return subpacket has no residency epoch";
                return false;
            }
            return true;
        }

        /** @brief RAII guard that releases one transport's fixed buffers after exchange. */
        class InFlightGuard final
        {
        public:
            /** @brief Attempt to acquire exclusive use of a transport workspace. */
            explicit InFlightGuard(std::atomic_flag &flag) noexcept
                : flag_(flag), acquired_(!flag_.test_and_set(std::memory_order_acquire))
            {
            }

            /** @brief Release the workspace only when this guard acquired it. */
            ~InFlightGuard()
            {
                if (acquired_)
                    flag_.clear(std::memory_order_release);
            }

            /** @return Whether this invocation owns the shared fixed buffers. */
            bool acquired() const noexcept { return acquired_; }

        private:
            std::atomic_flag &flag_;
            bool acquired_ = false;
        };

        /** @return Whether dispatch and return keys name the same sparse layer. */
        bool sameRoundTrip(
            const MoEOverlayRankBatchKey &dispatch,
            const MoEOverlayRankBatchKey &returned) noexcept
        {
            return dispatch.generation_id == returned.generation_id &&
                   dispatch.step_id == returned.step_id &&
                   dispatch.key_namespace == returned.key_namespace &&
                   dispatch.histogram_source == returned.histogram_source &&
                   dispatch.mtp_depth == returned.mtp_depth &&
                   dispatch.layer_idx == returned.layer_idx &&
                   dispatch.tier_idx == returned.tier_idx &&
                   dispatch.domain_ordinal == returned.domain_ordinal &&
                   dispatch.source_world_rank == returned.source_world_rank &&
                   dispatch.target_world_rank == returned.target_world_rank &&
                   dispatch.direction ==
                       MoEOverlayCollectiveDirection::Dispatch &&
                   returned.direction ==
                       MoEOverlayCollectiveDirection::ReturnReduce;
        }

    } // namespace

    bool MoEOverlayRankBatchKey::isValid() const noexcept
    {
        if (generation_id == 0 || layer_idx < 0 || tier_idx < 0 ||
            domain_ordinal < 0 || source_world_rank < 0 ||
            target_world_rank < 0 || source_world_rank == target_world_rank)
        {
            return false;
        }
        switch (key_namespace)
        {
        case MoEOverlayCollectiveNamespace::Main:
            return mtp_depth < 0 &&
                   (histogram_source == ExpertHistogramSource::DecodeToken ||
                    histogram_source == ExpertHistogramSource::PrefillChunk);
        case MoEOverlayCollectiveNamespace::MTP:
            return mtp_depth >= 0 &&
                   histogram_source == ExpertHistogramSource::GroupedVerifier;
        }
        return false;
    }

    std::string MoEOverlayRankBatchKey::toString() const
    {
        std::ostringstream stream;
        stream << "generation=" << generation_id
               << ",step=" << step_id
               << ",namespace=" << llaminar2::toString(key_namespace)
               << ",phase=" << static_cast<int>(histogram_source)
               << ",mtp_depth=" << mtp_depth
               << ",layer=" << layer_idx
               << ",tier=" << tier_idx
               << ",domain=" << domain_ordinal
               << ",source_rank=" << source_world_rank
               << ",target_rank=" << target_world_rank
               << ",direction=" << llaminar2::toString(direction)
               << ",sequence=" << sequence;
        return stream.str();
    }

    bool operator==(
        const MoEOverlayRankBatchKey &lhs,
        const MoEOverlayRankBatchKey &rhs) noexcept
    {
        return std::tie(
                   lhs.generation_id,
                   lhs.step_id,
                   lhs.key_namespace,
                   lhs.histogram_source,
                   lhs.mtp_depth,
                   lhs.layer_idx,
                   lhs.tier_idx,
                   lhs.domain_ordinal,
                   lhs.source_world_rank,
                   lhs.target_world_rank,
                   lhs.direction,
                   lhs.sequence) ==
               std::tie(
                   rhs.generation_id,
                   rhs.step_id,
                   rhs.key_namespace,
                   rhs.histogram_source,
                   rhs.mtp_depth,
                   rhs.layer_idx,
                   rhs.tier_idx,
                   rhs.domain_ordinal,
                   rhs.source_world_rank,
                   rhs.target_world_rank,
                   rhs.direction,
                   rhs.sequence);
    }

    bool operator!=(
        const MoEOverlayRankBatchKey &lhs,
        const MoEOverlayRankBatchKey &rhs) noexcept
    {
        return !(lhs == rhs);
    }

    namespace
    {
        /** @brief Derive a stable replay-ledger sequence without using an MPI tag. */
        uint64_t rankBatchSequence(
            MoEOverlayCollectiveNamespace key_namespace,
            int mtp_depth,
            int layer_idx,
            int tier_idx,
            int domain_ordinal,
            int target_world_rank,
            MoEOverlayCollectiveDirection direction) noexcept
        {
            /*
             * Full key authentication, not this compact value, is the wire
             * authority.  The sequence merely distributes graph keys over a
             * fixed stale-replay ledger and is deterministic on every rank.
             */
            uint64_t hash = 1469598103934665603ull;
            const auto mix = [&hash](uint64_t value)
            {
                hash ^= value;
                hash *= 1099511628211ull;
            };
            mix(static_cast<uint64_t>(key_namespace));
            mix(static_cast<uint64_t>(std::max(mtp_depth, -1) + 1));
            mix(static_cast<uint64_t>(std::max(layer_idx, 0)));
            mix(static_cast<uint64_t>(std::max(tier_idx, 0)));
            mix(static_cast<uint64_t>(std::max(domain_ordinal, 0)));
            mix(static_cast<uint64_t>(std::max(target_world_rank, 0)));
            mix(static_cast<uint64_t>(direction));
            return hash;
        }
    } // namespace

    MoEOverlayRankBatchKey makeMoEOverlayRankBatchKey(
        uint64_t generation_id,
        uint64_t step_id,
        ExpertHistogramSource histogram_source,
        int layer_idx,
        int tier_idx,
        int domain_ordinal,
        int source_world_rank,
        int target_world_rank,
        MoEOverlayCollectiveDirection direction)
    {
        MoEOverlayRankBatchKey key;
        key.generation_id = generation_id;
        key.step_id = step_id;
        key.key_namespace = MoEOverlayCollectiveNamespace::Main;
        key.histogram_source = histogram_source;
        key.mtp_depth = -1;
        key.layer_idx = layer_idx;
        key.tier_idx = tier_idx;
        key.domain_ordinal = domain_ordinal;
        key.source_world_rank = source_world_rank;
        key.target_world_rank = target_world_rank;
        key.direction = direction;
        key.sequence = rankBatchSequence(
            key.key_namespace,
            key.mtp_depth,
            layer_idx,
            tier_idx,
            domain_ordinal,
            target_world_rank,
            direction);
        return key;
    }

    MoEOverlayRankBatchKey makeMTPMoEOverlayRankBatchKey(
        uint64_t generation_id,
        uint64_t decode_step_id,
        int mtp_depth,
        int layer_idx,
        int tier_idx,
        int domain_ordinal,
        int source_world_rank,
        int target_world_rank,
        MoEOverlayCollectiveDirection direction)
    {
        MoEOverlayRankBatchKey key;
        key.generation_id = generation_id;
        key.step_id = decode_step_id;
        key.key_namespace = MoEOverlayCollectiveNamespace::MTP;
        key.histogram_source = ExpertHistogramSource::GroupedVerifier;
        key.mtp_depth = mtp_depth;
        key.layer_idx = layer_idx;
        key.tier_idx = tier_idx;
        key.domain_ordinal = domain_ordinal;
        key.source_world_rank = source_world_rank;
        key.target_world_rank = target_world_rank;
        key.direction = direction;
        key.sequence = rankBatchSequence(
            key.key_namespace,
            key.mtp_depth,
            layer_idx,
            tier_idx,
            domain_ordinal,
            target_world_rank,
            direction);
        return key;
    }

    MoEOverlayRankBatchWireWorkspace::MoEOverlayRankBatchWireWorkspace(
        Config config)
        : participant_ids_(std::move(config.participant_ids)),
          max_total_rows_(config.max_total_rows),
          max_total_return_rows_(config.return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes
                                     ? config.max_total_entries : config.max_total_rows),
          max_total_entries_(config.max_total_entries),
          d_model_(config.d_model),
          top_k_(config.top_k)
    {
        std::sort(participant_ids_.begin(), participant_ids_.end());
        if (participant_ids_.empty() || max_total_rows_ == 0 ||
            max_total_entries_ == 0 || d_model_ <= 0 || top_k_ <= 0 ||
            participant_ids_.front() < 0 || !isValidMoEOverlayReturnLayout(config.return_layout))
        {
            throw std::invalid_argument(
                "MoE rank-batch workspace requires participants and positive geometry");
        }
        if (std::adjacent_find(
                participant_ids_.begin(), participant_ids_.end()) !=
            participant_ids_.end())
        {
            throw std::invalid_argument(
                "MoE rank-batch workspace participant ids must be unique");
        }

        const size_t participant_count = participant_ids_.size();
        const size_t hidden_values = checkedMultiply(
            max_total_rows_, static_cast<size_t>(d_model_), "hidden values");
        const size_t hidden_bytes = checkedMultiply(
            hidden_values, sizeof(float), "hidden bytes");
        const size_t row_id_bytes = checkedMultiply(
            max_total_rows_, sizeof(int32_t), "row ids");
        const size_t offset_count = checkedAdd(
            max_total_rows_, participant_count, "entry offsets");
        const size_t offset_bytes = checkedMultiply(
            offset_count, sizeof(int32_t), "entry offset bytes");
        const size_t entry_bytes = checkedMultiply(
            max_total_entries_,
            3u * sizeof(int32_t) + sizeof(float),
            "route entries and slot identities");

        size_t dispatch_capacity = kEnvelopeCapacityAllowance;
        dispatch_capacity = checkedAdd(
            dispatch_capacity,
            checkedMultiply(
                participant_count,
                kDispatchSubrecordCapacityAllowance,
                "dispatch subrecord metadata"),
            "dispatch metadata");
        dispatch_capacity = checkedAdd(dispatch_capacity, row_id_bytes, "dispatch row ids");
        dispatch_capacity = checkedAdd(dispatch_capacity, offset_bytes, "dispatch offsets");
        dispatch_capacity = checkedAdd(dispatch_capacity, entry_bytes, "dispatch entries");
        dispatch_capacity = checkedAdd(dispatch_capacity, hidden_bytes, "dispatch hidden rows");

        size_t return_capacity = kEnvelopeCapacityAllowance;
        return_capacity = checkedAdd(
            return_capacity,
            checkedMultiply(
                participant_count,
                kReturnSubrecordCapacityAllowance,
                "return subrecord metadata"),
            "return metadata");
        return_capacity = checkedAdd(return_capacity,
            checkedMultiply(max_total_return_rows_, sizeof(int32_t), "return row ids"), "return row ids");
        return_capacity = checkedAdd(return_capacity,
            checkedMultiply(checkedMultiply(max_total_return_rows_, static_cast<size_t>(d_model_), "return elements"),
                            sizeof(float), "return rows"), "return rows");

        const size_t wire_capacity = std::max(dispatch_capacity, return_capacity);
        if (wire_capacity > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error(
                "MoE rank-batch wire capacity exceeds MPI's int byte-count limit");
        }
        send_buffer_.resize(wire_capacity);
        receive_buffer_.resize(wire_capacity);
    }

    bool MoEOverlayRankBatchWireWorkspace::encodeDispatch(
        const MoEOverlayRankBatchKey &key,
        std::span<const MoEOverlaySparseRows *const> rows,
        size_t *encoded_bytes,
        std::string *error)
    {
        return encodeDispatchInto(
            key, rows, send_buffer_, encoded_bytes, error);
    }

    bool MoEOverlayRankBatchWireWorkspace::encodeDispatchInto(
        const MoEOverlayRankBatchKey &key,
        std::span<const MoEOverlaySparseRows *const> rows,
        std::span<std::byte> destination,
        size_t *encoded_bytes,
        std::string *error)
    {
        if (error)
            error->clear();
        if (encoded_bytes)
            *encoded_bytes = 0;
        if (!encoded_bytes || !key.isValid() ||
            key.direction != MoEOverlayCollectiveDirection::Dispatch ||
            rows.size() != participant_ids_.size())
        {
            if (error)
                *error = "invalid dispatch batch key, output, or participant count";
            return false;
        }

        size_t total_rows = 0;
        size_t total_entries = 0;
        for (size_t index = 0; index < rows.size(); ++index)
        {
            if (!rows[index] ||
                !validateDispatchRows(
                    *rows[index], participant_ids_[index], d_model_, top_k_, error))
            {
                if (error && error->empty())
                    *error = "dispatch batch contains a null participant view";
                return false;
            }
            if (rows[index]->live_row_count > max_total_rows_ - total_rows ||
                rows[index]->live_entry_count > max_total_entries_ - total_entries)
            {
                if (error)
                    *error = "dispatch batch exceeds fixed aggregate row or entry capacity";
                return false;
            }
            total_rows += rows[index]->live_row_count;
            total_entries += rows[index]->live_entry_count;
        }

        FixedWriter writer(destination);
        const uint32_t participant_count =
            static_cast<uint32_t>(participant_ids_.size());
        if (!writer.append(kBatchMagic) || !writer.append(kBatchVersion) ||
            !writer.append(kDispatchEnvelopeKind) || !writeKey(writer, key) ||
            !writer.append(participant_count))
        {
            if (error)
                *error = "dispatch batch metadata exceeds fixed wire capacity";
            return false;
        }

        for (size_t index = 0; index < rows.size(); ++index)
        {
            const auto &packet = *rows[index];
            const uint64_t row_count = packet.live_row_count;
            const uint64_t entry_count = packet.live_entry_count;
            const size_t hidden_values =
                packet.live_row_count * static_cast<size_t>(d_model_);
            if (!writer.append(participant_ids_[index]) ||
                !writer.append(packet.residency_epoch) ||
                !writer.append(packet.source_participant) ||
                !writer.append(packet.target_participant) ||
                !writer.append(packet.d_model) ||
                !writer.append(packet.top_k) ||
                !writer.append(row_count) ||
                !writer.append(entry_count) ||
                !writer.appendBytes(
                    packet.row_ids_host,
                    packet.live_row_count * sizeof(int32_t)) ||
                !writer.appendBytes(
                    packet.entry_offsets_host,
                    (packet.live_row_count + 1u) * sizeof(int32_t)) ||
                !writer.appendBytes(
                    packet.expert_ids_host,
                    packet.live_entry_count * sizeof(int32_t)) ||
                !writer.appendBytes(
                    packet.route_weights_host,
                    packet.live_entry_count * sizeof(float)) ||
                !writer.appendBytes(
                    packet.original_route_slots_host,
                    packet.live_entry_count * sizeof(int32_t)) ||
                !writer.appendBytes(
                    packet.compact_route_slots_host,
                    packet.live_entry_count * sizeof(int32_t)) ||
                !writer.appendBytes(
                    packet.hidden_rows_fp32,
                    hidden_values * sizeof(float)))
            {
                if (error)
                    *error = "dispatch batch payload exceeds fixed wire capacity";
                return false;
            }
        }
        *encoded_bytes = writer.size();
        return true;
    }

    bool MoEOverlayRankBatchWireWorkspace::decodeDispatch(
        const MoEOverlayRankBatchKey &expected_key,
        std::span<const std::byte> payload,
        std::span<MoEOverlaySparseRows *const> rows,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!expected_key.isValid() ||
            expected_key.direction != MoEOverlayCollectiveDirection::Dispatch ||
            payload.empty() || payload.size() > receive_buffer_.size() ||
            rows.size() != participant_ids_.size())
        {
            if (error)
                *error = "invalid received dispatch batch geometry or key";
            return false;
        }

        const auto parse = [&](bool publish) -> bool
        {
            FixedReader reader(payload);
            uint32_t magic = 0;
            uint32_t version = 0;
            uint8_t kind = 0;
            uint32_t participant_count = 0;
            MoEOverlayRankBatchKey received_key;
            if (!reader.read(&magic) || !reader.read(&version) ||
                !reader.read(&kind) || !readKey(reader, &received_key) ||
                !reader.read(&participant_count))
            {
                if (error)
                    *error = "dispatch batch envelope header is truncated";
                return false;
            }
            if (magic != kBatchMagic || version != kBatchVersion ||
                kind != kDispatchEnvelopeKind || received_key != expected_key ||
                participant_count != participant_ids_.size())
            {
                if (error)
                    *error = "dispatch batch envelope identity or participant count mismatch";
                return false;
            }

            size_t aggregate_rows = 0;
            size_t aggregate_entries = 0;
            for (size_t index = 0; index < rows.size(); ++index)
            {
                int32_t participant_id = -1;
                uint64_t residency_epoch = 0;
                int32_t source_participant = -1;
                int32_t target_participant = -1;
                int32_t d_model = 0;
                int32_t top_k = 0;
                uint64_t row_count_wire = 0;
                uint64_t entry_count_wire = 0;
                if (!reader.read(&participant_id) ||
                    !reader.read(&residency_epoch) ||
                    !reader.read(&source_participant) ||
                    !reader.read(&target_participant) ||
                    !reader.read(&d_model) || !reader.read(&top_k) ||
                    !reader.read(&row_count_wire) ||
                    !reader.read(&entry_count_wire))
                {
                    if (error)
                        *error = "dispatch batch subrecord header is truncated";
                    return false;
                }
                if (row_count_wire > std::numeric_limits<size_t>::max() ||
                    entry_count_wire > std::numeric_limits<size_t>::max())
                {
                    if (error)
                        *error = "dispatch batch subrecord count is not representable";
                    return false;
                }
                const size_t row_count = static_cast<size_t>(row_count_wire);
                const size_t entry_count = static_cast<size_t>(entry_count_wire);
                auto *const destination = rows[index];
                if (!destination || participant_id != participant_ids_[index] ||
                    target_participant != participant_id ||
                    source_participant < 0 || d_model != d_model_ ||
                    top_k != top_k_ || row_count > destination->row_capacity ||
                    entry_count > destination->entry_capacity ||
                    !destination->row_ids_host ||
                    !destination->entry_offsets_host ||
                    !destination->expert_ids_host ||
                    !destination->route_weights_host ||
                    !destination->original_route_slots_host ||
                    !destination->compact_route_slots_host ||
                    !destination->hidden_rows_fp32 ||
                    ((row_count != 0 || entry_count != 0) &&
                     residency_epoch == 0) ||
                    row_count > max_total_rows_ - aggregate_rows ||
                    entry_count > max_total_entries_ - aggregate_entries)
                {
                    if (error)
                        *error = "dispatch batch subrecord violates graph-bound topology or capacity";
                    return false;
                }
                aggregate_rows += row_count;
                aggregate_entries += entry_count;

                const size_t row_bytes = row_count * sizeof(int32_t);
                const size_t offset_bytes = (row_count + 1u) * sizeof(int32_t);
                const size_t expert_bytes = entry_count * sizeof(int32_t);
                const size_t weight_bytes = entry_count * sizeof(float);
                const size_t route_slot_bytes =
                    entry_count * sizeof(int32_t);
                const size_t hidden_bytes =
                    row_count * static_cast<size_t>(d_model_) * sizeof(float);
                if (publish)
                {
                    if (!reader.readBytes(destination->row_ids_host, row_bytes) ||
                        !reader.readBytes(destination->entry_offsets_host, offset_bytes) ||
                        !reader.readBytes(destination->expert_ids_host, expert_bytes) ||
                        !reader.readBytes(destination->route_weights_host, weight_bytes) ||
                        !reader.readBytes(destination->original_route_slots_host, route_slot_bytes) ||
                        !reader.readBytes(destination->compact_route_slots_host, route_slot_bytes) ||
                        !reader.readBytes(destination->hidden_rows_fp32, hidden_bytes))
                    {
                        if (error)
                            *error = "dispatch batch payload is truncated";
                        return false;
                    }
                    destination->key = participantKey(expected_key, participant_id);
                    destination->residency_epoch = residency_epoch;
                    destination->source_participant = source_participant;
                    destination->target_participant = target_participant;
                    destination->d_model = d_model;
                    destination->top_k = top_k;
                    destination->live_row_count = row_count;
                    destination->live_entry_count = entry_count;
                }
                else if (!reader.skip(row_bytes) || !reader.skip(offset_bytes) ||
                         !reader.skip(expert_bytes) || !reader.skip(weight_bytes) ||
                         !reader.skip(route_slot_bytes) ||
                         !reader.skip(route_slot_bytes) ||
                         !reader.skip(hidden_bytes))
                {
                    if (error)
                        *error = "dispatch batch payload is truncated";
                    return false;
                }
            }
            if (reader.remaining() != 0)
            {
                if (error)
                    *error = "dispatch batch contains unauthenticated trailing bytes";
                return false;
            }
            return true;
        };

        /* Validate the complete envelope before publishing any participant view. */
        return parse(false) && parse(true);
    }

    bool MoEOverlayRankBatchWireWorkspace::encodeReturn(
        const MoEOverlayRankBatchKey &key,
        std::span<const MoEOverlayReturnRows *const> rows,
        size_t *encoded_bytes,
        std::string *error)
    {
        return encodeReturnInto(
            key, rows, send_buffer_, encoded_bytes, error);
    }

    bool MoEOverlayRankBatchWireWorkspace::encodeReturnInto(
        const MoEOverlayRankBatchKey &key,
        std::span<const MoEOverlayReturnRows *const> rows,
        std::span<std::byte> destination,
        size_t *encoded_bytes,
        std::string *error)
    {
        if (error)
            error->clear();
        if (encoded_bytes)
            *encoded_bytes = 0;
        if (!encoded_bytes || !key.isValid() ||
            key.direction != MoEOverlayCollectiveDirection::ReturnReduce ||
            rows.size() != participant_ids_.size())
        {
            if (error)
                *error = "invalid return batch key, output, or participant count";
            return false;
        }

        size_t total_rows = 0;
        for (size_t index = 0; index < rows.size(); ++index)
        {
            if (!rows[index] ||
                !validateReturnRows(
                    *rows[index], participant_ids_[index], d_model_, error))
            {
                if (error && error->empty())
                    *error = "return batch contains a null participant view";
                return false;
            }
            if (rows[index]->live_row_count > max_total_return_rows_ - total_rows)
            {
                if (error)
                    *error = "return batch exceeds fixed aggregate row capacity";
                return false;
            }
            total_rows += rows[index]->live_row_count;
        }

        FixedWriter writer(destination);
        const uint32_t participant_count =
            static_cast<uint32_t>(participant_ids_.size());
        if (!writer.append(kBatchMagic) || !writer.append(kBatchVersion) ||
            !writer.append(kReturnEnvelopeKind) || !writeKey(writer, key) ||
            !writer.append(participant_count))
        {
            if (error)
                *error = "return batch metadata exceeds fixed wire capacity";
            return false;
        }

        for (size_t index = 0; index < rows.size(); ++index)
        {
            const auto &packet = *rows[index];
            const uint64_t row_count = packet.live_row_count;
            const size_t output_values =
                packet.live_row_count * static_cast<size_t>(d_model_);
            if (!writer.append(participant_ids_[index]) ||
                !writer.append(packet.residency_epoch) ||
                !writer.append(packet.source_participant) ||
                !writer.append(packet.target_participant) ||
                !writer.append(packet.d_model) ||
                !writer.append(packet.layout) ||
                !writer.append(row_count) ||
                !writer.appendBytes(
                    packet.row_ids_host,
                    packet.live_row_count * sizeof(int32_t)) ||
                !writer.appendBytes(
                    packet.output_rows_fp32,
                    output_values * sizeof(float)))
            {
                if (error)
                    *error = "return batch payload exceeds fixed wire capacity";
                return false;
            }
        }
        *encoded_bytes = writer.size();
        return true;
    }

    bool MoEOverlayRankBatchWireWorkspace::decodeReturn(
        const MoEOverlayRankBatchKey &expected_key,
        std::span<const std::byte> payload,
        std::span<MoEOverlayReturnRows *const> rows,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!expected_key.isValid() ||
            expected_key.direction != MoEOverlayCollectiveDirection::ReturnReduce ||
            payload.empty() || payload.size() > receive_buffer_.size() ||
            rows.size() != participant_ids_.size())
        {
            if (error)
                *error = "invalid received return batch geometry or key";
            return false;
        }

        const auto parse = [&](bool publish) -> bool
        {
            FixedReader reader(payload);
            uint32_t magic = 0;
            uint32_t version = 0;
            uint8_t kind = 0;
            uint32_t participant_count = 0;
            MoEOverlayRankBatchKey received_key;
            if (!reader.read(&magic) || !reader.read(&version) ||
                !reader.read(&kind) || !readKey(reader, &received_key) ||
                !reader.read(&participant_count))
            {
                if (error)
                    *error = "return batch envelope header is truncated";
                return false;
            }
            if (magic != kBatchMagic || version != kBatchVersion ||
                kind != kReturnEnvelopeKind || received_key != expected_key ||
                participant_count != participant_ids_.size())
            {
                if (error)
                    *error = "return batch envelope identity or participant count mismatch";
                return false;
            }

            size_t aggregate_rows = 0;
            for (size_t index = 0; index < rows.size(); ++index)
            {
                int32_t participant_id = -1;
                uint64_t residency_epoch = 0;
                int32_t source_participant = -1;
                int32_t target_participant = -1;
                int32_t d_model = 0;
                uint64_t row_count_wire = 0;
                MoEOverlayReturnLayout layout{};
                if (!reader.read(&participant_id) ||
                    !reader.read(&residency_epoch) ||
                    !reader.read(&source_participant) ||
                    !reader.read(&target_participant) ||
                    !reader.read(&d_model) ||
                    !reader.read(&layout) ||
                    !reader.read(&row_count_wire))
                {
                    if (error)
                        *error = "return batch subrecord header is truncated";
                    return false;
                }
                if (row_count_wire > std::numeric_limits<size_t>::max())
                {
                    if (error)
                        *error = "return batch subrecord count is not representable";
                    return false;
                }
                const size_t row_count = static_cast<size_t>(row_count_wire);
                auto *const destination = rows[index];
                if (!destination || participant_id != participant_ids_[index] ||
                    source_participant != participant_id ||
                    target_participant < 0 || d_model != d_model_ ||
                    !isValidMoEOverlayReturnLayout(layout) ||
                    row_count > destination->row_capacity ||
                    !destination->row_ids_host ||
                    !destination->output_rows_fp32 ||
                    (row_count != 0 && residency_epoch == 0) ||
                    row_count > max_total_return_rows_ - aggregate_rows)
                {
                    if (error)
                        *error = "return batch subrecord violates graph-bound topology or capacity";
                    return false;
                }
                aggregate_rows += row_count;

                const size_t row_bytes = row_count * sizeof(int32_t);
                const size_t output_bytes =
                    row_count * static_cast<size_t>(d_model_) * sizeof(float);
                if (publish)
                {
                    if (!reader.readBytes(destination->row_ids_host, row_bytes) ||
                        !reader.readBytes(destination->output_rows_fp32, output_bytes))
                    {
                        if (error)
                            *error = "return batch payload is truncated";
                        return false;
                    }
                    destination->key = participantKey(expected_key, participant_id);
                    destination->residency_epoch = residency_epoch;
                    destination->source_participant = source_participant;
                    destination->target_participant = target_participant;
                    destination->d_model = d_model;
                    destination->live_row_count = row_count;
                    destination->layout = layout;
                }
                else if (!reader.skip(row_bytes) || !reader.skip(output_bytes))
                {
                    if (error)
                        *error = "return batch payload is truncated";
                    return false;
                }
            }
            if (reader.remaining() != 0)
            {
                if (error)
                    *error = "return batch contains unauthenticated trailing bytes";
                return false;
            }
            return true;
        };

        return parse(false) && parse(true);
    }

    std::span<const std::byte>
    MoEOverlayRankBatchWireWorkspace::encodedPayload(size_t encoded_bytes) const
    {
        if (encoded_bytes == 0 || encoded_bytes > send_buffer_.size())
        {
            throw std::out_of_range(
                "MoE rank-batch encoded payload exceeds fixed send buffer");
        }
        return std::span<const std::byte>(send_buffer_.data(), encoded_bytes);
    }

    MoEOverlayMPIRankBatchTransport::MoEOverlayMPIRankBatchTransport(
        Config config)
        : config_(std::move(config))
    {
        if (!config_.mpi_ctx || !config_.workspace ||
            config_.source_world_rank < 0 || config_.target_world_rank < 0 ||
            config_.source_world_rank == config_.target_world_rank ||
            config_.source_world_rank >= config_.mpi_ctx->world_size() ||
            config_.target_world_rank >= config_.mpi_ctx->world_size() ||
            (config_.mpi_ctx->rank() != config_.source_world_rank &&
             config_.mpi_ctx->rank() != config_.target_world_rank) ||
            config_.transaction_slot_count == 0 ||
            config_.asynchronous_send_slot_count == 0)
        {
            throw std::invalid_argument(
                "MoE MPI rank-batch transport requires one valid endpoint rank pair, workspace, and ledger");
        }
        dispatch_ledger_.resize(config_.transaction_slot_count);
        return_ledger_.resize(config_.transaction_slot_count);
        dispatch_send_slots_.resize(config_.asynchronous_send_slot_count);
        return_send_slots_.resize(config_.asynchronous_send_slot_count);
        for (auto &slot : dispatch_send_slots_)
            slot.storage.resize(config_.workspace->wireCapacityBytes());
        for (auto &slot : return_send_slots_)
            slot.storage.resize(config_.workspace->wireCapacityBytes());
    }

    MoEOverlayMPIRankBatchTransport::~MoEOverlayMPIRankBatchTransport()
    {
        drainSendSlotsNoexcept(dispatch_send_slots_, "dispatch");
        drainSendSlotsNoexcept(return_send_slots_, "return");
        if (pending_return_dispatch_key_ ||
            pending_return_receive_ != MPI_REQUEST_NULL)
        {
            LOG_ERROR(
                "MoE rank-batch transport destroyed with an unmatched preposted return receive");
            std::terminate();
        }
    }

    void MoEOverlayMPIRankBatchTransport::progressSendSlots(
        std::vector<AsynchronousSendSlot> &slots) const
    {
        for (auto &slot : slots)
        {
            if (!slot.in_flight)
                continue;
            if (config_.mpi_ctx->test(&slot.request))
            {
                slot.in_flight = false;
                slot.request = MPI_REQUEST_NULL;
            }
        }
    }

    bool MoEOverlayMPIRankBatchTransport::progressRequestToCompletion(
        MPI_Request *request,
        MPI_Status *status,
        const char *operation,
        std::string *error) const
    {
        if (!request || !operation)
        {
            if (error)
                *error = "rank-batch MPI progress request is invalid";
            return false;
        }
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(
                collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        while (!config_.mpi_ctx->test(request, status))
        {
            /*
             * MPI_Test is the progress engine on ordinary Open MPI builds. Do
             * not sleep or yield: doing so serializes segmented same-node
             * traffic behind scheduler quanta and materially increases decode
             * latency. The loop exists only where graph dependencies require
             * the received bytes before local compute may continue.
             */
            if (std::chrono::steady_clock::now() >= deadline)
            {
                if (error)
                {
                    *error = std::string(operation) +
                             " timed out after " +
                             std::to_string(
                                 collective_timeout_policy::
                                     kDefaultCollectiveTimeoutMs) +
                             "ms";
                }
                return false;
            }
        }
        return true;
    }

    MoEOverlayMPIRankBatchTransport::AsynchronousSendSlot *
    MoEOverlayMPIRankBatchTransport::acquireSendSlot(
        std::vector<AsynchronousSendSlot> &slots,
        size_t *cursor,
        const char *direction,
        std::string *error)
    {
        if (!cursor || !direction || slots.empty())
        {
            if (error)
                *error = "rank-batch asynchronous send ring is invalid";
            return nullptr;
        }

        const auto find_available = [&]() -> AsynchronousSendSlot *
        {
            progressSendSlots(slots);
            for (size_t offset = 0; offset < slots.size(); ++offset)
            {
                const size_t index = (*cursor + offset) % slots.size();
                if (!slots[index].in_flight)
                {
                    *cursor = (index + 1u) % slots.size();
                    return &slots[index];
                }
            }
            return nullptr;
        };

        if (auto *slot = find_available())
            return slot;

        const auto begin = std::chrono::steady_clock::now();
        const auto deadline =
            begin + std::chrono::milliseconds(
                        collective_timeout_policy::
                            kDefaultCollectiveTimeoutMs);
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (auto *slot = find_available())
            {
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_transport",
                    "rank_batch_send_ring_backpressure",
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<
                            std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - begin)
                            .count()),
                    "decode",
                    "mpi",
                    {{"direction", direction},
                     {"slots", std::to_string(slots.size())}});
                return slot;
            }
        }
        if (error)
        {
            *error = std::string("rank-batch ") + direction +
                     " send ring timed out waiting for a reusable fixed slot";
        }
        return nullptr;
    }

    void MoEOverlayMPIRankBatchTransport::drainSendSlotsNoexcept(
        std::vector<AsynchronousSendSlot> &slots,
        const char *direction) noexcept
    {
        try
        {
            for (auto &slot : slots)
            {
                if (!slot.in_flight)
                    continue;
                std::string error;
                if (!progressRequestToCompletion(
                        &slot.request,
                        nullptr,
                        direction,
                        &error))
                {
                    LOG_ERROR("MoE rank-batch teardown could not drain "
                              << direction << " send: " << error);
                    std::terminate();
                }
                slot.in_flight = false;
                slot.request = MPI_REQUEST_NULL;
            }
        }
        catch (const std::exception &exception)
        {
            LOG_ERROR("MoE rank-batch teardown MPI progress failed for "
                      << direction << ": " << exception.what());
            std::terminate();
        }
    }

    int MoEOverlayMPIRankBatchTransport::localWorldRank() const noexcept
    {
        return config_.mpi_ctx ? config_.mpi_ctx->rank() : -1;
    }

    bool MoEOverlayMPIRankBatchTransport::claimTransaction(
        const MoEOverlayRankBatchKey &key,
        std::vector<std::optional<MoEOverlayRankBatchKey>> &ledger,
        std::string *error)
    {
        if (!key.isValid() || ledger.empty())
        {
            if (error)
                *error = "rank-batch transaction key or ledger is invalid";
            return false;
        }
        /*
         * `sequence` is the immutable graph-stage identity. A model-lifetime
         * transport is intentionally shared by every serial transformer layer,
         * so unrelated sequences may hash to the same initial slot. Linear
         * probing binds each sequence to one stable slot without allocating a
         * hot-path map. Once bound, later request steps replace only that
         * sequence's prior key; replaying the exact live key remains fatal.
         */
        const size_t first = static_cast<size_t>(key.sequence % ledger.size());
        for (size_t probe = 0; probe < ledger.size(); ++probe)
        {
            auto &slot = ledger[(first + probe) % ledger.size()];
            if (!slot)
            {
                slot = key;
                return true;
            }
            if (slot->sequence != key.sequence)
                continue;
            if (*slot == key)
            {
                if (error)
                    *error = "stale rank-batch transaction key reuse rejected";
                return false;
            }
            slot = key;
            return true;
        }
        if (error)
        {
            *error =
                "rank-batch transaction ledger has no free graph-sequence slot";
        }
        return false;
    }

    MoEOverlayCollectiveResult
    MoEOverlayMPIRankBatchTransport::exchangeDispatch(
        const MoEOverlayRankBatchKey &key,
        std::span<const MoEOverlaySparseRows *const> outbound,
        std::span<MoEOverlaySparseRows *const> inbound)
    {
        MoEOverlayCollectiveResult result;
        InFlightGuard guard(in_flight_);
        if (!guard.acquired())
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "concurrent reuse of one rank-batch wire workspace is forbidden";
            return result;
        }
        if (key.direction != MoEOverlayCollectiveDirection::Dispatch ||
            key.source_world_rank != config_.source_world_rank ||
            key.target_world_rank != config_.target_world_rank)
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "dispatch key does not match immutable rank-pair topology";
            return result;
        }
        if (!claimTransaction(key, dispatch_ledger_, &result.error))
        {
            result.ok = false;
            result.error_code = 3;
            return result;
        }

        try
        {
            using Clock = std::chrono::steady_clock;
            const bool timing_enabled =
                PerfStatsCollector::isDomainEnabled(
                    "moe_overlay_transport");
            const auto total_begin = timing_enabled
                                         ? Clock::now()
                                         : Clock::time_point{};
            const int rank = config_.mpi_ctx->rank();
            const MoEOverlayRankBatchTelemetry telemetry(
                key, MoEOverlayRankBatchTransportKind::MPI,
                rank == config_.source_world_rank
                    ? MoEOverlayRankBatchEndpoint::Source
                    : MoEOverlayRankBatchEndpoint::Target);
            size_t payload_bytes = 0;
            uint64_t codec_ns = 0;
            uint64_t wait_ns = 0;
            if (rank == config_.source_world_rank)
            {
                if (!inbound.empty())
                {
                    result.ok = false;
                    result.error_code = 4;
                    result.error =
                        "dispatch source rank supplied inbound participant rows";
                    return result;
                }
                if (pending_return_dispatch_key_ ||
                    pending_return_receive_ != MPI_REQUEST_NULL)
                {
                    result.ok = false;
                    result.error_code = 4;
                    result.error =
                        "dispatch source already owns an unmatched preposted return receive";
                    return result;
                }

                auto *slot = acquireSendSlot(
                    dispatch_send_slots_,
                    &next_dispatch_send_slot_,
                    "dispatch",
                    &result.error);
                if (!slot)
                {
                    result.ok = false;
                    result.error_code = 4;
                    return result;
                }
                const auto codec_begin = timing_enabled
                                             ? Clock::now()
                                             : Clock::time_point{};
                if (!config_.workspace->encodeDispatchInto(
                        key,
                        outbound,
                        slot->storage,
                        &payload_bytes,
                        &result.error))
                {
                    result.ok = false;
                    result.error_code = 4;
                    return result;
                }
                const auto codec_end = timing_enabled
                                           ? Clock::now()
                                           : Clock::time_point{};

                /*
                 * The reverse receive is visible to MPI before dispatch is
                 * published. A fast endpoint can therefore return immediately
                 * without rendezvous setup landing on the critical path.
                 */
                auto return_capacity = config_.workspace->receiveCapacity();
                pending_return_receive_ = config_.mpi_ctx->irecv(
                    return_capacity.data(),
                    return_capacity.size(),
                    MPI_BYTE,
                    config_.target_world_rank,
                    kReturnTag);
                pending_return_dispatch_key_ = key;

                slot->request = config_.mpi_ctx->isend(
                    slot->storage.data(),
                    payload_bytes,
                    MPI_BYTE,
                    config_.target_world_rank,
                    kDispatchTag);
                slot->in_flight = slot->request != MPI_REQUEST_NULL;
                telemetry.recordAsyncSendSubmission(dispatch_send_slots_.size());
                if (timing_enabled)
                {
                    codec_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            codec_end - codec_begin)
                            .count());
                }
                telemetry.recordTransaction(
                    payload_bytes, config_.workspace->participantIds().size());
            }
            else
            {
                if (!outbound.empty())
                {
                    result.ok = false;
                    result.error_code = 5;
                    result.error = "dispatch target rank supplied outbound participant rows";
                    return result;
                }
                auto capacity = config_.workspace->receiveCapacity();
                MPI_Status status{};
                const auto wait_begin = timing_enabled
                                            ? Clock::now()
                                            : Clock::time_point{};
                MPI_Request request = config_.mpi_ctx->irecv(
                    capacity.data(),
                    capacity.size(),
                    MPI_BYTE,
                    config_.source_world_rank,
                    kDispatchTag);
                if (!progressRequestToCompletion(
                        &request,
                        &status,
                        "rank-batch dispatch receive",
                        &result.error))
                {
                    result.ok = false;
                    result.error_code = 6;
                    return result;
                }
                const auto wait_end = timing_enabled
                                          ? Clock::now()
                                          : Clock::time_point{};
                const int received = config_.mpi_ctx->getCount(status, MPI_BYTE);
                const auto codec_begin = timing_enabled
                                             ? Clock::now()
                                             : Clock::time_point{};
                if (received <= 0 ||
                    static_cast<size_t>(received) > capacity.size() ||
                    !config_.workspace->decodeDispatch(
                        key,
                        std::span<const std::byte>(
                            capacity.data(), static_cast<size_t>(received)),
                        inbound,
                        &result.error))
                {
                    result.ok = false;
                    result.error_code = 6;
                    if (result.error.empty())
                        result.error = "dispatch receive count exceeds fixed capacity";
                    return result;
                }
                const auto codec_end = timing_enabled
                                           ? Clock::now()
                                           : Clock::time_point{};
                payload_bytes = static_cast<size_t>(received);
                if (timing_enabled)
                {
                    codec_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            codec_end - codec_begin)
                            .count());
                    wait_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            wait_end - wait_begin)
                            .count());
                }
                telemetry.recordTransaction(
                    payload_bytes, config_.workspace->participantIds().size());
            }
            if (timing_enabled)
            {
                telemetry.recordTimings(
                    codec_ns,
                    wait_ns,
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            Clock::now() - total_begin)
                            .count()));
            }
        }
        catch (const std::exception &exception)
        {
            result.ok = false;
            result.error_code = 7;
            result.error = std::string("rank-batch dispatch MPI failure: ") +
                           exception.what();
            return result;
        }

        result.collective_complete = true;
        return result;
    }

    MoEOverlayCollectiveResult
    MoEOverlayMPIRankBatchTransport::exchangeReturn(
        const MoEOverlayRankBatchKey &key,
        std::span<const MoEOverlayReturnRows *const> outbound,
        std::span<MoEOverlayReturnRows *const> inbound)
    {
        MoEOverlayCollectiveResult result;
        InFlightGuard guard(in_flight_);
        if (!guard.acquired())
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "concurrent reuse of one rank-batch wire workspace is forbidden";
            return result;
        }
        if (key.direction != MoEOverlayCollectiveDirection::ReturnReduce ||
            key.source_world_rank != config_.source_world_rank ||
            key.target_world_rank != config_.target_world_rank)
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "return key does not match immutable rank-pair topology";
            return result;
        }
        if (!claimTransaction(key, return_ledger_, &result.error))
        {
            result.ok = false;
            result.error_code = 3;
            return result;
        }

        try
        {
            using Clock = std::chrono::steady_clock;
            const bool timing_enabled =
                PerfStatsCollector::isDomainEnabled(
                    "moe_overlay_transport");
            const auto total_begin = timing_enabled
                                         ? Clock::now()
                                         : Clock::time_point{};
            const int rank = config_.mpi_ctx->rank();
            const MoEOverlayRankBatchTelemetry telemetry(
                key, MoEOverlayRankBatchTransportKind::MPI,
                rank == config_.source_world_rank
                    ? MoEOverlayRankBatchEndpoint::Source
                    : MoEOverlayRankBatchEndpoint::Target);
            size_t payload_bytes = 0;
            uint64_t codec_ns = 0;
            uint64_t wait_ns = 0;
            if (rank == config_.target_world_rank)
            {
                if (!inbound.empty())
                {
                    result.ok = false;
                    result.error_code = 4;
                    result.error =
                        "return target rank supplied inbound participant rows";
                    return result;
                }
                auto *slot = acquireSendSlot(
                    return_send_slots_,
                    &next_return_send_slot_,
                    "return",
                    &result.error);
                if (!slot)
                {
                    result.ok = false;
                    result.error_code = 4;
                    return result;
                }
                const auto codec_begin = timing_enabled
                                             ? Clock::now()
                                             : Clock::time_point{};
                if (!config_.workspace->encodeReturnInto(
                        key,
                        outbound,
                        slot->storage,
                        &payload_bytes,
                        &result.error))
                {
                    result.ok = false;
                    result.error_code = 4;
                    return result;
                }
                const auto codec_end = timing_enabled
                                           ? Clock::now()
                                           : Clock::time_point{};
                slot->request = config_.mpi_ctx->isend(
                    slot->storage.data(),
                    payload_bytes,
                    MPI_BYTE,
                    config_.source_world_rank,
                    kReturnTag);
                slot->in_flight = slot->request != MPI_REQUEST_NULL;
                telemetry.recordAsyncSendSubmission(return_send_slots_.size());
                if (timing_enabled)
                {
                    codec_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            codec_end - codec_begin)
                            .count());
                }
                telemetry.recordTransaction(
                    payload_bytes, config_.workspace->participantIds().size());
            }
            else
            {
                if (!outbound.empty())
                {
                    result.ok = false;
                    result.error_code = 5;
                    result.error = "return source rank supplied outbound participant rows";
                    return result;
                }
                if (!pending_return_dispatch_key_ ||
                    !sameRoundTrip(*pending_return_dispatch_key_, key))
                {
                    result.ok = false;
                    result.error_code = 6;
                    result.error =
                        "return source has no matching preposted receive for this dispatch";
                    return result;
                }
                auto capacity = config_.workspace->receiveCapacity();
                MPI_Status status{};
                const auto wait_begin = timing_enabled
                                            ? Clock::now()
                                            : Clock::time_point{};
                if (!progressRequestToCompletion(
                        &pending_return_receive_,
                        &status,
                        "rank-batch return receive",
                        &result.error))
                {
                    result.ok = false;
                    result.error_code = 6;
                    return result;
                }
                const auto wait_end = timing_enabled
                                          ? Clock::now()
                                          : Clock::time_point{};
                pending_return_receive_ = MPI_REQUEST_NULL;
                pending_return_dispatch_key_.reset();
                const int received = config_.mpi_ctx->getCount(status, MPI_BYTE);
                const auto codec_begin = timing_enabled
                                             ? Clock::now()
                                             : Clock::time_point{};
                if (received <= 0 ||
                    static_cast<size_t>(received) > capacity.size() ||
                    !config_.workspace->decodeReturn(
                        key,
                        std::span<const std::byte>(
                            capacity.data(), static_cast<size_t>(received)),
                        inbound,
                        &result.error))
                {
                    result.ok = false;
                    result.error_code = 6;
                    if (result.error.empty())
                        result.error = "return receive count exceeds fixed capacity";
                    return result;
                }
                const auto codec_end = timing_enabled
                                           ? Clock::now()
                                           : Clock::time_point{};
                payload_bytes = static_cast<size_t>(received);
                if (timing_enabled)
                {
                    codec_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            codec_end - codec_begin)
                            .count());
                    wait_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            wait_end - wait_begin)
                            .count());
                }
                telemetry.recordTransaction(
                    payload_bytes, config_.workspace->participantIds().size());
            }
            if (timing_enabled)
            {
                telemetry.recordTimings(
                    codec_ns,
                    wait_ns,
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            Clock::now() - total_begin)
                            .count()));
            }
        }
        catch (const std::exception &exception)
        {
            result.ok = false;
            result.error_code = 7;
            result.error = std::string("rank-batch return MPI failure: ") +
                           exception.what();
            return result;
        }

        result.collective_complete = true;
        return result;
    }

} // namespace llaminar2
