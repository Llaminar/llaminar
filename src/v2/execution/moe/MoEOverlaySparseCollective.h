/**
 * @file MoEOverlaySparseCollective.h
 * @brief Compact sparse payload transport for graph-native MoE overlay collectives.
 */

#pragma once

#include "backends/DeviceId.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace llaminar2
{
    class IDeviceContext;
    class IMPIContext;

    enum class MoEOverlayCollectiveDirection : uint8_t
    {
        Dispatch = 0,
        ReturnReduce = 1,
    };

    const char *toString(MoEOverlayCollectiveDirection direction);

    struct MoEOverlayCollectiveKey
    {
        uint64_t generation_id = 0;
        uint64_t step_id = 0;
        int32_t layer_idx = -1;
        int32_t tier_idx = -1;
        int32_t domain_id = -1;
        MoEOverlayCollectiveDirection direction = MoEOverlayCollectiveDirection::Dispatch;
        uint64_t sequence = 0;

        bool isValid() const;
        std::string toString() const;
    };

    bool operator==(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs);
    bool operator!=(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs);
    bool operator<(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs);

    struct MoEOverlaySparseRows
    {
        MoEOverlayCollectiveKey key;
        int32_t source_participant = -1;
        int32_t target_participant = -1;
        int32_t d_model = 0;
        int32_t top_k = 0;
        size_t live_row_count = 0;
        size_t live_entry_count = 0;
        size_t row_capacity = 0;
        size_t entry_capacity = 0;

        int32_t *row_ids_host = nullptr;
        int32_t *entry_offsets_host = nullptr;
        int32_t *expert_ids_host = nullptr;
        float *route_weights_host = nullptr;
        float *hidden_rows_fp32 = nullptr;
    };

    struct MoEOverlayReturnRows
    {
        MoEOverlayCollectiveKey key;
        int32_t source_participant = -1;
        int32_t target_participant = -1;
        int32_t d_model = 0;
        size_t live_row_count = 0;
        size_t row_capacity = 0;

        int32_t *row_ids_host = nullptr;
        float *output_rows_fp32 = nullptr;
    };

    struct MoEOverlaySparseTransferCounters
    {
        size_t dense_dispatch_bytes = 0;
        size_t dense_return_bytes = 0;
        size_t compact_dispatch_bytes = 0;
        size_t compact_return_bytes = 0;
        size_t compact_row_count = 0;
        size_t compact_entry_count = 0;

        size_t denseTotalBytes() const { return dense_dispatch_bytes + dense_return_bytes; }
        size_t compactTotalBytes() const { return compact_dispatch_bytes + compact_return_bytes; }
        size_t denseBytesAvoided() const
        {
            const size_t dense = denseTotalBytes();
            const size_t compact = compactTotalBytes();
            return dense > compact ? dense - compact : 0;
        }
    };

    size_t denseMoEOverlayDispatchBytes(int seq_len, int top_k, int d_model);
    size_t denseMoEOverlayReturnBytes(int seq_len, int d_model);
    size_t compactMoEOverlayDispatchBytes(const MoEOverlaySparseRows &rows);
    size_t compactMoEOverlayReturnBytes(const MoEOverlayReturnRows &rows);
    MoEOverlaySparseTransferCounters measureMoEOverlaySparseTransferCounters(
        int seq_len,
        int top_k,
        int d_model,
        const MoEOverlaySparseRows *dispatch_rows,
        const MoEOverlayReturnRows *return_rows);

    class MoEOverlayCollectiveWorkspace
    {
    public:
        void ensureCapacity(size_t max_rows,
                            size_t max_entries,
                            int d_model,
                            int top_k,
                            DeviceId device);

        void resetForStep(uint64_t generation_id, uint64_t step_id);

        MoEOverlaySparseRows dispatchReceive(int layer_idx, int tier_idx);
        MoEOverlaySparseRows localExpertInput(int layer_idx, int tier_idx);
        MoEOverlayReturnRows localExpertOutput(int layer_idx, int tier_idx);
        MoEOverlayReturnRows returnReceive(int layer_idx, int tier_idx);

        size_t maxRows() const { return max_rows_; }
        size_t maxEntries() const { return max_entries_; }
        int dModel() const { return d_model_; }
        int topK() const { return top_k_; }

    private:
        struct SparseStorage
        {
            std::vector<int32_t> row_ids_host;
            std::vector<int32_t> entry_offsets_host;
            std::vector<int32_t> expert_ids_host;
            std::vector<float> route_weights_host;
            std::vector<float> hidden_rows_fp32;
        };

        struct ReturnStorage
        {
            std::vector<int32_t> row_ids_host;
            std::vector<float> output_rows_fp32;
        };

        struct LayerTierBuffers
        {
            SparseStorage dispatch_receive;
            SparseStorage local_expert_input;
            ReturnStorage local_expert_output;
            ReturnStorage return_receive;
        };

        LayerTierBuffers &buffersFor(int layer_idx, int tier_idx);
        void ensureSparseStorage(SparseStorage &storage);
        void ensureReturnStorage(ReturnStorage &storage);

        size_t max_rows_ = 0;
        size_t max_entries_ = 0;
        int d_model_ = 0;
        int top_k_ = 0;
        DeviceId device_ = DeviceId::cpu();
        uint64_t generation_id_ = 0;
        uint64_t step_id_ = 0;
        std::map<std::pair<int, int>, LayerTierBuffers> buffers_by_layer_tier_;
    };

    struct MoEOverlayCollectiveResult
    {
        bool ok = true;
        bool collective_complete = false;
        int error_code = 0;
        std::string error;
    };

    class IMoEOverlaySparseCollectiveContext
    {
    public:
        virtual ~IMoEOverlaySparseCollectiveContext() = default;

        virtual MoEOverlayCollectiveResult dispatch(const MoEOverlayCollectiveKey &key,
                                                    const MoEOverlaySparseRows &outbound,
                                                    MoEOverlaySparseRows *inbound,
                                                    IDeviceContext *ctx) = 0;

        virtual MoEOverlayCollectiveResult returnReduce(const MoEOverlayCollectiveKey &key,
                                                        const MoEOverlayReturnRows &outbound,
                                                        MoEOverlayReturnRows *inbound,
                                                        IDeviceContext *ctx) = 0;

        virtual void abort(const MoEOverlayCollectiveKey &key, int reason_code) = 0;
    };

    class MoEOverlayLocalSparseCollectiveContext final : public IMoEOverlaySparseCollectiveContext
    {
    public:
        struct Config
        {
            int participant_count = 0;
            size_t slot_count = 0;
        };

        explicit MoEOverlayLocalSparseCollectiveContext(Config config);
        ~MoEOverlayLocalSparseCollectiveContext() override;

        MoEOverlayCollectiveResult dispatch(const MoEOverlayCollectiveKey &key,
                                            const MoEOverlaySparseRows &outbound,
                                            MoEOverlaySparseRows *inbound,
                                            IDeviceContext *ctx) override;

        MoEOverlayCollectiveResult returnReduce(const MoEOverlayCollectiveKey &key,
                                                const MoEOverlayReturnRows &outbound,
                                                MoEOverlayReturnRows *inbound,
                                                IDeviceContext *ctx) override;

        void abort(const MoEOverlayCollectiveKey &key, int reason_code) override;

    private:
        struct DispatchPayload;
        struct ReturnPayload;
        struct Slot;

        MoEOverlayCollectiveResult publishDispatch(const MoEOverlayCollectiveKey &key,
                                                   const MoEOverlaySparseRows &outbound,
                                                   MoEOverlaySparseRows *inbound);

        MoEOverlayCollectiveResult publishReturn(const MoEOverlayCollectiveKey &key,
                                                 const MoEOverlayReturnRows &outbound,
                                                 MoEOverlayReturnRows *inbound);

        int participant_count_ = 0;
        std::vector<std::unique_ptr<Slot>> slots_;
        std::unordered_set<std::string> completed_keys_;
        std::map<std::string, int> aborted_keys_;
    };

    class MoEOverlayMPISparseCollectiveContext final : public IMoEOverlaySparseCollectiveContext
    {
    public:
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_ctx;
            int local_participant_id = -1;
        };

        explicit MoEOverlayMPISparseCollectiveContext(Config config);
        ~MoEOverlayMPISparseCollectiveContext() override;

        MoEOverlayCollectiveResult dispatch(const MoEOverlayCollectiveKey &key,
                                            const MoEOverlaySparseRows &outbound,
                                            MoEOverlaySparseRows *inbound,
                                            IDeviceContext *ctx) override;

        MoEOverlayCollectiveResult returnReduce(const MoEOverlayCollectiveKey &key,
                                                const MoEOverlayReturnRows &outbound,
                                                MoEOverlayReturnRows *inbound,
                                                IDeviceContext *ctx) override;

        void abort(const MoEOverlayCollectiveKey &key, int reason_code) override;

    private:
        struct DispatchPacket;
        struct ReturnPacket;

        int localParticipantId() const;

        MoEOverlayCollectiveResult dispatchHostStaged(const MoEOverlayCollectiveKey &key,
                                                      const MoEOverlaySparseRows &outbound,
                                                      MoEOverlaySparseRows *inbound);

        MoEOverlayCollectiveResult returnHostStaged(const MoEOverlayCollectiveKey &key,
                                                    const MoEOverlayReturnRows &outbound,
                                                    MoEOverlayReturnRows *inbound);

        Config config_;
        std::unordered_set<std::string> completed_keys_;
        std::map<std::string, int> aborted_keys_;
    };

} // namespace llaminar2
