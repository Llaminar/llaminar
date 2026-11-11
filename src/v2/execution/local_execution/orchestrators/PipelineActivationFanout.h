/**
 * @file PipelineActivationFanout.h
 * @brief PMA-owned original completed-byte fanout of one received pipeline activation.
 *
 * One receiving native TP member owns one persistent bank per adjacent boundary,
 * shared by all mutually exclusive retained row-bucket graphs. Header/extent
 * comes only from the original channel publication, never host observation.
 */
#pragma once
#include "transfer/PipelineActivationFanoutProtocol.h"
#include "transfer/CapturedTransferChannel.h"
#include "collective/ILocalTPContext.h"
#include <memory>
namespace llaminar2
{
/** @brief Actual native metadata owner; not a topology, byte ledger or inference controller. */
class PipelineActivationFanout final
{
public:
    /** @brief Materialize the exact admitted receiving-member metadata outside capture.
     * @param device Actual local TP member. @param context Original homogeneous native TP context.
     * @param participant Original ordered member. @param channel Exact incoming channel retained by its domain.
     * @param memory Existing admitted PMA with fanoutDeviceBytes()/fanoutHostBytes() staging contributions.
     * @param setup_stream This member's exact original setup stream, non-null.
     * @param transfer Exact transfer authority retained for allocation and replay; defaults to production singleton.
     * @return Shared fixed bank reused by every retained row bucket for this boundary.
     * @throws std::exception Before capture for foreign ownership, absent admission or initialization failure.
     */
    static std::shared_ptr<PipelineActivationFanout> create(DeviceId device, ILocalTPContext &context,
        int participant, std::shared_ptr<CapturedTransferChannel> channel,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, void *setup_stream,
        TransferEngine &transfer = TransferEngine::instance());
    /** @brief Retire after every original graph/terminal borrow; cold partial initialization drains its exact stream. */
    ~PipelineActivationFanout();
    /** @return Exact device bytes claimed per receiving native member; storage is independent of live rows. */
    static constexpr size_t deviceBytes() noexcept { return sizeof(PipelineActivationFanoutDeviceState); }
    /** @return TransferEngine's exact page-rounded immutable operand allocation. */
    static size_t hostBytes();
    /** @brief Authenticate one immutable row-count declaration before graph construction/submission.
     * @param device Original local member. @param context Exact original native context.
     * @param participant Original ordered member. @param channel Same incoming channel control block.
     * @throws std::invalid_argument For a foreign/replaced bank owner. */
    void requireOriginal(DeviceId device, ILocalTPContext *context, int participant,
        const std::shared_ptr<CapturedTransferChannel> &channel) const;
    /** @brief Enqueue fixed24 header, completed live payload and receipt/row validation on the original stream.
     * @param hidden Local persistent payload bank. @param rows Canonical retained local INT32 geometry input.
     * @param row_offset Original count offset. @param row_bytes Immutable bytes per live row.
     * @param message Same original maximum message/key as the leader channel binding.
     * @param leader_completion Original completed receive view only on member0; absent on TP siblings.
     * @param stream Exact participant stream after original receive, before model roots.
     * @return Whether every original native collective was accepted.
     * No host readback, hot allocation or per-request metadata rewriting occurs. */
    bool enqueue(TensorBase &hidden, TensorBase &rows, size_t row_offset, std::uint64_t row_bytes,
        CapturedTransferMessage message, CapturedTransferReceiveCompletion leader_completion, void *stream) const;
private:
    struct Storage;
    explicit PipelineActivationFanout(std::unique_ptr<Storage> storage);
    std::unique_ptr<Storage> storage_;
};
}
