/**
 * @file MoEOverlayDispatchCollective.h
 * @brief Graph-native sparse collective contract for MoE overlay dispatch.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace llaminar2
{
    class IDeviceContext;
    class TensorBase;

    enum class MoEOverlayDispatchRequestKind : uint8_t
    {
        RoutedWork,
        NoOp,
        Cancel,
        Shutdown,
    };

    const char *toString(MoEOverlayDispatchRequestKind kind);

    struct MoEOverlayDispatchGroup
    {
        int domain_id = -1;
        int layer_id = -1;
        int dispatch_group_id = -1;
        int participant_count = 0;
        int participant_index = -1;
        int owner_participant_index = -1;
        int executor_participant_index = -1;
        uint64_t stage_sequence = 0;
        int microbatch_id = 0;
        uint64_t decode_sequence = 0;

        bool isValid() const;
        bool isParticipant() const;
        bool ownsExecution() const;
    };

    struct MoEOverlayDispatchMetrics
    {
        uint64_t wait_ns = 0;
        uint32_t no_op_count = 0;
        uint32_t routed_request_count = 0;
        uint32_t remote_endpoint_work_count = 0;
        uint32_t cancel_count = 0;
        size_t selected_row_count = 0;
        size_t routed_entry_count = 0;
        size_t transfer_bytes = 0;

        void merge(const MoEOverlayDispatchMetrics &other);
    };

    struct MoEOverlayDispatchRequest
    {
        MoEOverlayDispatchRequestKind kind = MoEOverlayDispatchRequestKind::NoOp;
        MoEOverlayDispatchGroup group;
        int layer_idx = -1;
        int tier_index = -1;
        const int *selected_rows = nullptr;
        size_t selected_row_count = 0;
        size_t routed_entry_count = 0;
        size_t transfer_bytes = 0;
        TensorBase *input = nullptr;
        TensorBase *output = nullptr;
        int cancel_reason_code = 0;

        bool hasRoutedWork() const;

        static MoEOverlayDispatchRequest noOp(const MoEOverlayDispatchGroup &group,
                                              int layer_idx,
                                              int tier_index);
        static MoEOverlayDispatchRequest routedWork(const MoEOverlayDispatchGroup &group,
                                                    int layer_idx,
                                                    int tier_index,
                                                    const int *selected_rows,
                                                    size_t selected_row_count,
                                                    size_t routed_entry_count,
                                                    size_t transfer_bytes,
                                                    TensorBase *input,
                                                    TensorBase *output);
        static MoEOverlayDispatchRequest cancel(const MoEOverlayDispatchGroup &group,
                                                int layer_idx,
                                                int tier_index,
                                                int reason_code);
    };

    struct MoEOverlayDispatchResult
    {
        bool ok = true;
        MoEOverlayDispatchRequestKind request_kind = MoEOverlayDispatchRequestKind::NoOp;
        MoEOverlayDispatchGroup group;
        TensorBase *partial_output = nullptr;
        MoEOverlayDispatchMetrics metrics;
        int error_code = 0;
        std::string error;
        bool collective_complete = false;
    };

    class IMoEOverlayDispatchBackend
    {
    public:
        virtual ~IMoEOverlayDispatchBackend() = default;

        virtual MoEOverlayDispatchResult dispatch(
            const MoEOverlayDispatchGroup &group,
            const MoEOverlayDispatchRequest &request,
            IDeviceContext *ctx) = 0;
    };

    class MoEOverlayDispatchRendezvous
    {
    public:
        struct Config
        {
            int participant_count = 0;
            size_t slot_count = 0;
        };

        explicit MoEOverlayDispatchRendezvous(Config config);
        ~MoEOverlayDispatchRendezvous();

        MoEOverlayDispatchRendezvous(const MoEOverlayDispatchRendezvous &) = delete;
        MoEOverlayDispatchRendezvous &operator=(const MoEOverlayDispatchRendezvous &) = delete;

        MoEOverlayDispatchResult publish(const MoEOverlayDispatchRequest &request);

        int participantCount() const { return participant_count_; }
        size_t slotCount() const { return slots_.size(); }

    private:
        struct ParticipantState
        {
            bool seen = false;
            MoEOverlayDispatchRequestKind kind = MoEOverlayDispatchRequestKind::NoOp;
            size_t selected_row_count = 0;
            size_t routed_entry_count = 0;
            size_t transfer_bytes = 0;
        };

        struct Slot;

        int participant_count_ = 0;
        std::vector<std::unique_ptr<Slot>> slots_;
    };

    class MoEOverlayLocalRendezvousBackend final : public IMoEOverlayDispatchBackend
    {
    public:
        explicit MoEOverlayLocalRendezvousBackend(MoEOverlayDispatchRendezvous::Config config);

        MoEOverlayDispatchResult dispatch(
            const MoEOverlayDispatchGroup &group,
            const MoEOverlayDispatchRequest &request,
            IDeviceContext *ctx) override;

        MoEOverlayDispatchRendezvous &rendezvous() { return rendezvous_; }
        const MoEOverlayDispatchRendezvous &rendezvous() const { return rendezvous_; }

    private:
        MoEOverlayDispatchRendezvous rendezvous_;
    };

} // namespace llaminar2