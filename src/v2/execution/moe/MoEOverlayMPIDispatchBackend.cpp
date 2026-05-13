/**
 * @file MoEOverlayMPIDispatchBackend.cpp
 * @brief MPI-backed graph-native overlay dispatch backend implementation.
 */

#include "MoEOverlayMPIDispatchBackend.h"
#include "interfaces/IMPIContext.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        int32_t clampSizeToInt32(size_t value)
        {
            return static_cast<int32_t>(std::min<size_t>(
                value,
                static_cast<size_t>(std::numeric_limits<int32_t>::max())));
        }

        int32_t low32(uint64_t value)
        {
            return static_cast<int32_t>(value & 0xffffffffull);
        }

        int32_t high32(uint64_t value)
        {
            return static_cast<int32_t>((value >> 32) & 0xffffffffull);
        }

        uint64_t join64(int32_t low, int32_t high)
        {
            return (static_cast<uint64_t>(static_cast<uint32_t>(high)) << 32) |
                   static_cast<uint64_t>(static_cast<uint32_t>(low));
        }

        MoEOverlayDispatchMetrics metricsFor(const MoEOverlayDispatchRequest &request)
        {
            MoEOverlayDispatchMetrics metrics;
            switch (request.kind)
            {
            case MoEOverlayDispatchRequestKind::NoOp:
                metrics.no_op_count = 1;
                break;
            case MoEOverlayDispatchRequestKind::RoutedWork:
                metrics.routed_request_count = 1;
                metrics.selected_row_count = request.selected_row_count;
                metrics.routed_entry_count = request.routed_entry_count;
                metrics.transfer_bytes = request.transfer_bytes;
                break;
            case MoEOverlayDispatchRequestKind::Cancel:
                metrics.cancel_count = 1;
                break;
            case MoEOverlayDispatchRequestKind::Shutdown:
                break;
            }
            return metrics;
        }

        bool sameDispatchIdentity(
            const MoEOverlayDispatchGroup &lhs,
            const MoEOverlayDispatchGroup &rhs)
        {
            return lhs.domain_id == rhs.domain_id &&
                   lhs.layer_id == rhs.layer_id &&
                   lhs.dispatch_group_id == rhs.dispatch_group_id &&
                   lhs.stage_sequence == rhs.stage_sequence &&
                   lhs.microbatch_id == rhs.microbatch_id &&
                   lhs.decode_sequence == rhs.decode_sequence;
        }

        bool requestTakesPriority(
            MoEOverlayDispatchRequestKind candidate,
            MoEOverlayDispatchRequestKind current)
        {
            if (candidate == MoEOverlayDispatchRequestKind::Cancel)
                return true;
            if (current == MoEOverlayDispatchRequestKind::Cancel)
                return false;
            return candidate == MoEOverlayDispatchRequestKind::RoutedWork &&
                   current != MoEOverlayDispatchRequestKind::RoutedWork;
        }

        MoEOverlayMPIMessageKind messageKindFor(MoEOverlayDispatchRequestKind kind)
        {
            switch (kind)
            {
            case MoEOverlayDispatchRequestKind::RoutedWork:
                return MoEOverlayMPIMessageKind::RoutedWork;
            case MoEOverlayDispatchRequestKind::NoOp:
                return MoEOverlayMPIMessageKind::NoOp;
            case MoEOverlayDispatchRequestKind::Cancel:
                return MoEOverlayMPIMessageKind::Cancel;
            case MoEOverlayDispatchRequestKind::Shutdown:
                return MoEOverlayMPIMessageKind::Shutdown;
            }
            return MoEOverlayMPIMessageKind::Cancel;
        }
    } // namespace

    struct MoEOverlayMPIDispatchBackend::LocalRendezvousSlot
    {
        std::mutex mutex;
        std::condition_variable cv;
        bool active = false;
        bool complete = false;
        MoEOverlayDispatchGroup group;
        MoEOverlayDispatchRequest canonical_request;
        MoEOverlayDispatchResult shared_result;
        MoEOverlayDispatchMetrics metrics;
        int seen_count = 0;
        int remaining_readers = 0;
        std::vector<bool> seen;

        void start(const MoEOverlayDispatchGroup &new_group, int participant_count)
        {
            active = true;
            complete = false;
            group = new_group;
            canonical_request = MoEOverlayDispatchRequest::noOp(new_group, new_group.layer_id, -1);
            shared_result = {};
            metrics = {};
            seen_count = 0;
            remaining_readers = 0;
            seen.assign(static_cast<size_t>(participant_count), false);
        }

        void reset()
        {
            active = false;
            complete = false;
            group = {};
            canonical_request = {};
            shared_result = {};
            metrics = {};
            seen_count = 0;
            remaining_readers = 0;
            std::fill(seen.begin(), seen.end(), false);
        }
    };

    const char *toString(MoEOverlayMPIMessageKind kind)
    {
        switch (kind)
        {
        case MoEOverlayMPIMessageKind::RoutedWork:
            return "RoutedWork";
        case MoEOverlayMPIMessageKind::NoOp:
            return "NoOp";
        case MoEOverlayMPIMessageKind::Cancel:
            return "Cancel";
        case MoEOverlayMPIMessageKind::ForwardDone:
            return "ForwardDone";
        case MoEOverlayMPIMessageKind::Shutdown:
            return "Shutdown";
        }
        return "Unknown";
    }

    std::array<int32_t, MoEOverlayMPIDispatchHeader::kWordCount>
    MoEOverlayMPIDispatchHeader::toWords() const
    {
        return {
            version,
            static_cast<int32_t>(kind),
            domain_id,
            layer_id,
            tier_index,
            dispatch_group_id,
            participant_count,
            owner_participant_index,
            executor_participant_index,
            low32(stage_sequence),
            high32(stage_sequence),
            microbatch_id,
            low32(decode_sequence),
            high32(decode_sequence),
            selected_row_count,
            routed_entry_count,
            low32(transfer_bytes),
            high32(transfer_bytes),
            cancel_reason_code,
            0,
        };
    }

    bool MoEOverlayMPIDispatchHeader::fromWords(
        const int32_t *words,
        size_t word_count,
        MoEOverlayMPIDispatchHeader &out,
        std::string *error)
    {
        if (!words || word_count != kWordCount)
        {
            if (error)
                *error = "invalid overlay MPI dispatch header word count";
            return false;
        }

        if (words[0] != kVersion)
        {
            if (error)
            {
                std::ostringstream ss;
                ss << "unsupported overlay MPI dispatch header version " << words[0]
                   << " (expected " << kVersion << ")";
                *error = ss.str();
            }
            return false;
        }

        const auto kind = static_cast<MoEOverlayMPIMessageKind>(words[1]);
        switch (kind)
        {
        case MoEOverlayMPIMessageKind::RoutedWork:
        case MoEOverlayMPIMessageKind::NoOp:
        case MoEOverlayMPIMessageKind::Cancel:
        case MoEOverlayMPIMessageKind::ForwardDone:
        case MoEOverlayMPIMessageKind::Shutdown:
            break;
        default:
            if (error)
                *error = "invalid overlay MPI dispatch message kind";
            return false;
        }

        out.version = words[0];
        out.kind = kind;
        out.domain_id = words[2];
        out.layer_id = words[3];
        out.tier_index = words[4];
        out.dispatch_group_id = words[5];
        out.participant_count = words[6];
        out.owner_participant_index = words[7];
        out.executor_participant_index = words[8];
        out.stage_sequence = join64(words[9], words[10]);
        out.microbatch_id = words[11];
        out.decode_sequence = join64(words[12], words[13]);
        out.selected_row_count = words[14];
        out.routed_entry_count = words[15];
        out.transfer_bytes = join64(words[16], words[17]);
        out.cancel_reason_code = words[18];
        return true;
    }

    MoEOverlayMPIDispatchBackend::MoEOverlayMPIDispatchBackend(Config config)
        : config_(std::move(config))
    {
        if (config_.local_participant_count > 1)
        {
            const size_t slot_count = std::max<size_t>(1, config_.local_rendezvous_slots);
            local_rendezvous_slots_.reserve(slot_count);
            for (size_t index = 0; index < slot_count; ++index)
                local_rendezvous_slots_.push_back(std::make_unique<LocalRendezvousSlot>());
        }
    }

    MoEOverlayMPIDispatchBackend::~MoEOverlayMPIDispatchBackend() = default;

    void MoEOverlayMPIDispatchBackend::beginForward()
    {
        cancel_broadcast_since_forward_begin_.store(false, std::memory_order_release);
    }

    bool MoEOverlayMPIDispatchBackend::cancelBroadcastSinceForwardBegin() const
    {
        return cancel_broadcast_since_forward_begin_.load(std::memory_order_acquire);
    }

    MoEOverlayDispatchResult MoEOverlayMPIDispatchBackend::dispatch(
        const MoEOverlayDispatchGroup &group,
        const MoEOverlayDispatchRequest &request,
        IDeviceContext *)
    {
        if (config_.local_participant_count > 1)
            return dispatchWithLocalRendezvous(group, request, nullptr);
        return dispatchDirect(group, request, nullptr);
    }

    MoEOverlayDispatchResult MoEOverlayMPIDispatchBackend::dispatchDirect(
        const MoEOverlayDispatchGroup &group,
        const MoEOverlayDispatchRequest &request,
        IDeviceContext *)
    {
        MoEOverlayDispatchResult result;
        result.group = group;
        result.request_kind = request.kind;
        result.partial_output = request.output;
        result.metrics = metricsFor(request);

        if (!config_.mpi_ctx)
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "overlay MPI dispatch backend has no MPI context";
            return result;
        }

        if (!group.isValid())
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "overlay MPI dispatch backend received invalid dispatch group";
            return result;
        }

        if (config_.mpi_ctx->rank() != config_.root_rank)
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "overlay MPI dispatch backend dispatch() must be called on root rank";
            return result;
        }

        const auto header = headerFor(group, request, messageKindFor(request.kind));
        std::string error;
        if (!broadcastHeader(header, &error))
        {
            result.ok = false;
            result.error_code = 4;
            result.error = std::move(error);
            return result;
        }

        result.ok = request.kind != MoEOverlayDispatchRequestKind::Cancel;
        result.collective_complete = true;
        if (result.ok && request.kind == MoEOverlayDispatchRequestKind::RoutedWork &&
            config_.mpi_ctx->world_size() > 1)
        {
            result.metrics.remote_endpoint_work_count =
                static_cast<uint32_t>(std::max(0, config_.mpi_ctx->world_size() - 1));
        }
        if (!result.ok)
        {
            result.error_code = request.cancel_reason_code == 0 ? 5 : request.cancel_reason_code;
            result.error = "overlay MPI dispatch canceled";
        }
        return result;
    }

    MoEOverlayDispatchResult MoEOverlayMPIDispatchBackend::dispatchWithLocalRendezvous(
        const MoEOverlayDispatchGroup &group,
        const MoEOverlayDispatchRequest &request,
        IDeviceContext *ctx)
    {
        MoEOverlayDispatchResult result;
        result.group = group;
        result.request_kind = request.kind;
        result.partial_output = request.output;
        result.metrics = metricsFor(request);

        if (!group.isValid())
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "overlay MPI local rendezvous received invalid dispatch group";
            return result;
        }
        if (group.participant_count != config_.local_participant_count)
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "overlay MPI local rendezvous participant count mismatch";
            return result;
        }
        if (local_rendezvous_slots_.empty())
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "overlay MPI local rendezvous has no slots";
            return result;
        }

        auto &slot = *local_rendezvous_slots_[static_cast<size_t>(group.dispatch_group_id) % local_rendezvous_slots_.size()];
        std::unique_lock<std::mutex> lock(slot.mutex);

        const auto failSlot = [&](int error_code, const std::string &error)
        {
            slot.shared_result = result;
            slot.shared_result.ok = false;
            slot.shared_result.error_code = error_code;
            slot.shared_result.error = error;
            slot.shared_result.collective_complete = true;
            slot.complete = true;
            slot.remaining_readers = std::max(1, slot.seen_count);
            slot.cv.notify_all();
            return slot.shared_result;
        };

        if (!slot.active)
            slot.start(group, config_.local_participant_count);
        else if (!sameDispatchIdentity(slot.group, group))
            return failSlot(4, "overlay MPI local rendezvous slot is occupied by another dispatch group");

        const size_t participant_index = static_cast<size_t>(group.participant_index);
        if (participant_index >= slot.seen.size())
            return failSlot(5, "overlay MPI local rendezvous participant index out of range");
        if (slot.seen[participant_index])
            return failSlot(6, "duplicate overlay MPI local rendezvous participant publish");

        slot.seen[participant_index] = true;
        ++slot.seen_count;
        slot.metrics.merge(metricsFor(request));
        if (slot.seen_count == 1 || requestTakesPriority(request.kind, slot.canonical_request.kind))
        {
            slot.canonical_request = request;
            slot.canonical_request.group = group;
        }

        if (slot.seen_count == config_.local_participant_count)
        {
            auto canonical_group = slot.group;
            canonical_group.participant_index = slot.group.owner_participant_index >= 0
                                                   ? slot.group.owner_participant_index
                                                   : 0;
            slot.canonical_request.group = canonical_group;
            auto shared = dispatchDirect(canonical_group, slot.canonical_request, ctx);
            const auto remote_endpoint_work_count = shared.metrics.remote_endpoint_work_count;
            shared.metrics = slot.metrics;
            shared.metrics.remote_endpoint_work_count += remote_endpoint_work_count;
            shared.collective_complete = true;
            slot.shared_result = std::move(shared);
            slot.complete = true;
            slot.remaining_readers = config_.local_participant_count;
            slot.cv.notify_all();
        }
        else
        {
            const int timeout_ms = debugEnv().tp_collect_timeout_ms;
            const auto done = [&slot]() { return slot.complete; };
            if (timeout_ms > 0)
            {
                if (!slot.cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), done))
                    return failSlot(7, "overlay MPI local rendezvous timed out waiting for participants");
            }
            else
            {
                slot.cv.wait(lock, done);
            }
        }

        result = slot.shared_result;
        result.group = group;
        result.partial_output = request.output;
        if (slot.remaining_readers > 0)
            --slot.remaining_readers;
        if (slot.remaining_readers == 0)
            slot.reset();
        return result;
    }

    bool MoEOverlayMPIDispatchBackend::receiveHeader(
        MoEOverlayMPIDispatchHeader &header,
        std::string *error) const
    {
        if (!config_.mpi_ctx)
        {
            if (error)
                *error = "overlay MPI dispatch backend has no MPI context";
            return false;
        }

        std::array<int32_t, MoEOverlayMPIDispatchHeader::kWordCount> words{};
        try
        {
            config_.mpi_ctx->broadcast_int32(words.data(), words.size(), config_.root_rank);
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = std::string("overlay MPI dispatch header receive failed: ") + e.what();
            return false;
        }

        return MoEOverlayMPIDispatchHeader::fromWords(words.data(), words.size(), header, error);
    }

    bool MoEOverlayMPIDispatchBackend::sendForwardDone(
        const MoEOverlayDispatchGroup &group,
        int layer_idx,
        int tier_index)
    {
        MoEOverlayDispatchRequest request = MoEOverlayDispatchRequest::noOp(group, layer_idx, tier_index);
        return broadcastHeader(headerFor(group, request, MoEOverlayMPIMessageKind::ForwardDone));
    }

    bool MoEOverlayMPIDispatchBackend::sendShutdown()
    {
        MoEOverlayDispatchGroup group;
        group.domain_id = 0;
        group.layer_id = 0;
        group.dispatch_group_id = 0;
        group.participant_count = 1;
        group.participant_index = 0;
        group.owner_participant_index = 0;
        group.executor_participant_index = 0;

        MoEOverlayDispatchRequest request = MoEOverlayDispatchRequest::noOp(group, 0, 0);
        return broadcastHeader(headerFor(group, request, MoEOverlayMPIMessageKind::Shutdown));
    }

    bool MoEOverlayMPIDispatchBackend::sendCancel(
        const MoEOverlayDispatchGroup &group,
        int layer_idx,
        int tier_index,
        int reason_code)
    {
        MoEOverlayDispatchRequest request = MoEOverlayDispatchRequest::cancel(group, layer_idx, tier_index, reason_code);
        return broadcastHeader(headerFor(group, request, MoEOverlayMPIMessageKind::Cancel));
    }

    bool MoEOverlayMPIDispatchBackend::broadcastHeader(
        const MoEOverlayMPIDispatchHeader &header,
        std::string *error) const
    {
        if (!config_.mpi_ctx)
        {
            if (error)
                *error = "overlay MPI dispatch backend has no MPI context";
            return false;
        }

        auto words = header.toWords();
        try
        {
            if (config_.mpi_ctx->world_size() > 1)
                config_.mpi_ctx->broadcast_int32(words.data(), words.size(), config_.root_rank);
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = std::string("overlay MPI dispatch header broadcast failed: ") + e.what();
            return false;
        }
        catch (...)
        {
            if (error)
                *error = "overlay MPI dispatch header broadcast failed: unknown exception";
            return false;
        }
        if (header.kind == MoEOverlayMPIMessageKind::Cancel)
            cancel_broadcast_since_forward_begin_.store(true, std::memory_order_release);
        return true;
    }

    MoEOverlayMPIDispatchHeader MoEOverlayMPIDispatchBackend::headerFor(
        const MoEOverlayDispatchGroup &group,
        const MoEOverlayDispatchRequest &request,
        MoEOverlayMPIMessageKind kind) const
    {
        MoEOverlayMPIDispatchHeader header;
        header.kind = kind;
        header.domain_id = group.domain_id;
        header.layer_id = group.layer_id;
        header.tier_index = request.tier_index;
        header.dispatch_group_id = group.dispatch_group_id;
        header.participant_count = group.participant_count;
        header.owner_participant_index = group.owner_participant_index;
        header.executor_participant_index = group.executor_participant_index;
        header.stage_sequence = group.stage_sequence;
        header.microbatch_id = group.microbatch_id;
        header.decode_sequence = group.decode_sequence;
        header.selected_row_count = clampSizeToInt32(request.selected_row_count);
        header.routed_entry_count = clampSizeToInt32(request.routed_entry_count);
        header.transfer_bytes = request.transfer_bytes;
        header.cancel_reason_code = request.cancel_reason_code;
        return header;
    }

} // namespace llaminar2