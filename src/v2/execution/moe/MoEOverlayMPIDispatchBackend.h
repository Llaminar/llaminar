/**
 * @file MoEOverlayMPIDispatchBackend.h
 * @brief MPI-backed graph-native overlay dispatch backend for remote domains.
 */

#pragma once

#include "MoEOverlayDispatchCollective.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;

    enum class MoEOverlayMPIMessageKind : int32_t
    {
        RoutedWork = 1,
        NoOp = 2,
        Cancel = 3,
        ForwardDone = 4,
        Shutdown = 5,
    };

    const char *toString(MoEOverlayMPIMessageKind kind);

    struct MoEOverlayMPIDispatchHeader
    {
        static constexpr int32_t kVersion = 1;
        static constexpr size_t kWordCount = 20;

        int32_t version = kVersion;
        MoEOverlayMPIMessageKind kind = MoEOverlayMPIMessageKind::NoOp;
        int32_t domain_id = -1;
        int32_t layer_id = -1;
        int32_t tier_index = -1;
        int32_t dispatch_group_id = -1;
        int32_t participant_count = 0;
        int32_t owner_participant_index = -1;
        int32_t executor_participant_index = -1;
        uint64_t stage_sequence = 0;
        int32_t microbatch_id = 0;
        uint64_t decode_sequence = 0;
        int32_t selected_row_count = 0;
        int32_t routed_entry_count = 0;
        uint64_t transfer_bytes = 0;
        int32_t cancel_reason_code = 0;

        std::array<int32_t, kWordCount> toWords() const;
        static bool fromWords(const int32_t *words,
                              size_t word_count,
                              MoEOverlayMPIDispatchHeader &out,
                              std::string *error = nullptr);
    };

    class MoEOverlayMPIDispatchBackend final : public IMoEOverlayDispatchBackend
    {
    public:
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_ctx;
            int root_rank = 0;
            int local_participant_count = 1;
            size_t local_rendezvous_slots = 64;
        };

        explicit MoEOverlayMPIDispatchBackend(Config config);
        ~MoEOverlayMPIDispatchBackend() override;

        int rootRank() const { return config_.root_rank; }
        MoEOverlayDispatchResult dispatch(
            const MoEOverlayDispatchGroup &group,
            const MoEOverlayDispatchRequest &request,
            IDeviceContext *ctx) override;

        void beginForward();
        bool cancelBroadcastSinceForwardBegin() const;
        bool receiveHeader(MoEOverlayMPIDispatchHeader &header,
                           std::string *error = nullptr) const;
        bool sendForwardDone(const MoEOverlayDispatchGroup &group,
                             int layer_idx,
                             int tier_index);
        bool sendShutdown();
        bool sendCancel(const MoEOverlayDispatchGroup &group,
                        int layer_idx,
                        int tier_index,
                        int reason_code);

    private:
        struct LocalRendezvousSlot;

        MoEOverlayDispatchResult dispatchDirect(
            const MoEOverlayDispatchGroup &group,
            const MoEOverlayDispatchRequest &request,
            IDeviceContext *ctx);
        MoEOverlayDispatchResult dispatchWithLocalRendezvous(
            const MoEOverlayDispatchGroup &group,
            const MoEOverlayDispatchRequest &request,
            IDeviceContext *ctx);
        bool broadcastHeader(const MoEOverlayMPIDispatchHeader &header,
                             std::string *error = nullptr) const;
        MoEOverlayMPIDispatchHeader headerFor(
            const MoEOverlayDispatchGroup &group,
            const MoEOverlayDispatchRequest &request,
            MoEOverlayMPIMessageKind kind) const;

        Config config_;
        mutable std::atomic<bool> cancel_broadcast_since_forward_begin_{false};
        std::vector<std::unique_ptr<LocalRendezvousSlot>> local_rendezvous_slots_;
    };

} // namespace llaminar2