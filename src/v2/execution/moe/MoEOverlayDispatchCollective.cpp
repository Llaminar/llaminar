/**
 * @file MoEOverlayDispatchCollective.cpp
 * @brief Graph-native sparse collective contract helpers for MoE overlay dispatch.
 */

#include "MoEOverlayDispatchCollective.h"

#include <atomic>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
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

        MoEOverlayDispatchMetrics metricsForRequest(const MoEOverlayDispatchRequest &request)
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

        struct SpinGuard
        {
            explicit SpinGuard(std::atomic_flag &flag)
                : flag_(flag)
            {
                while (flag_.test_and_set(std::memory_order_acquire))
                {
                }
            }

            ~SpinGuard()
            {
                flag_.clear(std::memory_order_release);
            }

            SpinGuard(const SpinGuard &) = delete;
            SpinGuard &operator=(const SpinGuard &) = delete;

        private:
            std::atomic_flag &flag_;
        };
    } // namespace

    struct MoEOverlayDispatchRendezvous::Slot
    {
        explicit Slot(int participant_count)
            : participants(static_cast<size_t>(participant_count))
        {
        }

        std::atomic_flag lock = ATOMIC_FLAG_INIT;
        bool active = false;
        MoEOverlayDispatchGroup group;
        int seen_count = 0;
        MoEOverlayDispatchMetrics metrics;
        std::vector<ParticipantState> participants;

        void reset()
        {
            active = false;
            group = {};
            seen_count = 0;
            metrics = {};
            for (auto &participant : participants)
                participant = {};
        }
    };

    const char *toString(MoEOverlayDispatchRequestKind kind)
    {
        switch (kind)
        {
        case MoEOverlayDispatchRequestKind::RoutedWork:
            return "RoutedWork";
        case MoEOverlayDispatchRequestKind::NoOp:
            return "NoOp";
        case MoEOverlayDispatchRequestKind::Cancel:
            return "Cancel";
        case MoEOverlayDispatchRequestKind::Shutdown:
            return "Shutdown";
        }
        return "Unknown";
    }

    bool MoEOverlayDispatchGroup::isValid() const
    {
        return domain_id >= 0 &&
               layer_id >= 0 &&
               dispatch_group_id >= 0 &&
               participant_count > 0 &&
               participant_index >= 0 &&
               participant_index < participant_count;
    }

    bool MoEOverlayDispatchGroup::isParticipant() const
    {
        return participant_index >= 0 && participant_index < participant_count;
    }

    bool MoEOverlayDispatchGroup::ownsExecution() const
    {
        return executor_participant_index >= 0 &&
               participant_index == executor_participant_index;
    }

    void MoEOverlayDispatchMetrics::merge(const MoEOverlayDispatchMetrics &other)
    {
        wait_ns += other.wait_ns;
        no_op_count += other.no_op_count;
        routed_request_count += other.routed_request_count;
        remote_endpoint_work_count += other.remote_endpoint_work_count;
        cancel_count += other.cancel_count;
        selected_row_count += other.selected_row_count;
        routed_entry_count += other.routed_entry_count;
        transfer_bytes += other.transfer_bytes;
    }

    bool MoEOverlayDispatchRequest::hasRoutedWork() const
    {
        return kind == MoEOverlayDispatchRequestKind::RoutedWork &&
               (routed_entry_count > 0 || selected_row_count > 0);
    }

    MoEOverlayDispatchRequest MoEOverlayDispatchRequest::noOp(
        const MoEOverlayDispatchGroup &group,
        int layer_idx,
        int tier_index)
    {
        MoEOverlayDispatchRequest request;
        request.kind = MoEOverlayDispatchRequestKind::NoOp;
        request.group = group;
        request.layer_idx = layer_idx;
        request.tier_index = tier_index;
        return request;
    }

    MoEOverlayDispatchRequest MoEOverlayDispatchRequest::routedWork(
        const MoEOverlayDispatchGroup &group,
        int layer_idx,
        int tier_index,
        const int *selected_rows,
        size_t selected_row_count,
        size_t routed_entry_count,
        size_t transfer_bytes,
        TensorBase *input,
        TensorBase *output)
    {
        MoEOverlayDispatchRequest request;
        request.kind = MoEOverlayDispatchRequestKind::RoutedWork;
        request.group = group;
        request.layer_idx = layer_idx;
        request.tier_index = tier_index;
        request.selected_rows = selected_rows;
        request.selected_row_count = selected_row_count;
        request.routed_entry_count = routed_entry_count;
        request.transfer_bytes = transfer_bytes;
        request.input = input;
        request.output = output;
        return request;
    }

    MoEOverlayDispatchRequest MoEOverlayDispatchRequest::cancel(
        const MoEOverlayDispatchGroup &group,
        int layer_idx,
        int tier_index,
        int reason_code)
    {
        MoEOverlayDispatchRequest request;
        request.kind = MoEOverlayDispatchRequestKind::Cancel;
        request.group = group;
        request.layer_idx = layer_idx;
        request.tier_index = tier_index;
        request.cancel_reason_code = reason_code;
        return request;
    }

    MoEOverlayDispatchRendezvous::MoEOverlayDispatchRendezvous(Config config)
        : participant_count_(config.participant_count)
    {
        if (participant_count_ <= 0)
            throw std::invalid_argument("MoEOverlayDispatchRendezvous requires participant_count > 0");
        if (config.slot_count == 0)
            throw std::invalid_argument("MoEOverlayDispatchRendezvous requires slot_count > 0");

        slots_.reserve(config.slot_count);
        for (size_t index = 0; index < config.slot_count; ++index)
            slots_.push_back(std::make_unique<Slot>(participant_count_));
    }

    MoEOverlayDispatchRendezvous::~MoEOverlayDispatchRendezvous() = default;

    MoEOverlayDispatchResult MoEOverlayDispatchRendezvous::publish(
        const MoEOverlayDispatchRequest &request)
    {
        MoEOverlayDispatchResult result;
        result.request_kind = request.kind;
        result.group = request.group;
        result.partial_output = request.output;
        result.metrics = metricsForRequest(request);

        if (!request.group.isValid())
        {
            result.ok = false;
            result.error_code = 1;
            result.error = "invalid overlay dispatch group";
            return result;
        }
        if (request.group.participant_count != participant_count_)
        {
            result.ok = false;
            result.error_code = 2;
            result.error = "dispatch group participant count does not match rendezvous";
            return result;
        }

        auto &slot = *slots_[static_cast<size_t>(request.group.dispatch_group_id) % slots_.size()];
        SpinGuard guard(slot.lock);

        if (!slot.active)
        {
            slot.active = true;
            slot.group = request.group;
        }
        else if (!sameDispatchIdentity(slot.group, request.group))
        {
            result.ok = false;
            result.error_code = 3;
            result.error = "overlay dispatch rendezvous slot is occupied by another dispatch group";
            return result;
        }

        const size_t participant_index = static_cast<size_t>(request.group.participant_index);
        auto &participant = slot.participants[participant_index];
        if (participant.seen)
        {
            result.ok = false;
            result.error_code = 4;
            result.error = "duplicate overlay dispatch participant publish";
            return result;
        }

        participant.seen = true;
        participant.kind = request.kind;
        participant.selected_row_count = request.selected_row_count;
        participant.routed_entry_count = request.routed_entry_count;
        participant.transfer_bytes = request.transfer_bytes;
        ++slot.seen_count;
        slot.metrics.merge(result.metrics);

        if (request.kind == MoEOverlayDispatchRequestKind::Cancel)
        {
            result.ok = false;
            result.collective_complete = true;
            result.error_code = request.cancel_reason_code == 0 ? 5 : request.cancel_reason_code;
            result.error = "overlay dispatch collective canceled";
            result.metrics = slot.metrics;
            slot.reset();
            return result;
        }

        if (slot.seen_count == participant_count_)
        {
            result.collective_complete = true;
            result.metrics = slot.metrics;
            slot.reset();
        }

        return result;
    }

    MoEOverlayLocalRendezvousBackend::MoEOverlayLocalRendezvousBackend(
        MoEOverlayDispatchRendezvous::Config config)
        : rendezvous_(std::move(config))
    {
    }

    MoEOverlayDispatchResult MoEOverlayLocalRendezvousBackend::dispatch(
        const MoEOverlayDispatchGroup &group,
        const MoEOverlayDispatchRequest &request,
        IDeviceContext *)
    {
        auto canonical_request = request;
        canonical_request.group = group;
        return rendezvous_.publish(canonical_request);
    }

} // namespace llaminar2