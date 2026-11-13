/**
 * @file PipelineNativeByteChannel.h
 * @brief Original rank-owned native byte edges between real homogeneous pipeline domains.
 *
 * The source domain leader publishes only useful bytes to the actual next
 * domain consumers. Each endpoint retains its own completed native frame and
 * physical claim; payloads stay in their original arena/workspace allocation.
 * No mapped payload, second communicator mechanism or host row mirror exists.
 */
#pragma once
#include "collective/ILocalTPContext.h"
#include "collective/NativePipelineByteFrame.h"
#include "execution/mtp/DeviceGenerationCollectiveReceipt.h"
#include "transfer/TransferEngine.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include <map>
#include <optional>
#include <vector>
#include <span>
#include <variant>
namespace llaminar2 {
namespace test { struct PipelineNativeByteChannelReplayProbe; }
/** @brief Immutable original local native fanout edges; no mutable scheduling policy. */
class PipelineNativeByteChannel final {
public:
    /** @brief Materialize exact endpoint frames before capture under the final shared PMA.
     * @param context Existing rank-local native context over original pipeline participants.
     * @param source Original source leader coordinate. @param consumers Actual consuming native coordinates.
     * @param capacity Admitted maximum byte extent, never the runtime transfer extent.
     * @param memory Final canonical physical authority shared by every original stage.
     * @param streams Original participant setup streams in native membership order.
     * @return Original captured byte owner retaining the context's revocable borrow.
     */
    static std::shared_ptr<PipelineNativeByteChannel> create(ILocalTPContext &context, int source,
        std::span<const int> consumers, std::size_t capacity,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory, std::span<void *const> streams);
    /** @return Exact per-endpoint metadata allocation request, without payload storage. */
    static std::size_t endpointBytes() { return NativeCountedAllGatherLayout(1).bytes(); }
    /** @brief Authenticate the exact original native communicator and addressed member before port capture. */
    void requireOriginal(ILocalTPContext &context, int participant) const;
    /** @return Original native member's device; rejects nonmembers. */
    DeviceId device(int participant) const;
    /** @return Whether this actual member is the one producer. */
    bool isSource(int participant) const noexcept { return participant == source_; }
    /** @return Whether this actual member consumes this edge. */
    bool isConsumer(int participant) const noexcept;
    /** @return Exact admitted native capacity. */
    std::size_t capacity() const noexcept { return capacity_; }
private:
    friend class PipelineNativeByteBinding;
    friend struct test::PipelineNativeByteChannelReplayProbe;
    /** @brief Passive original device frame observation; only the native terminal fixture may inspect it. */
    std::pair<DeviceId, const void *> terminalObservation(int consumer, int participant) const;
    /** @brief Enqueue exact original native bytes and validate completion before dependent graph work.
     * @param participant Original native coordinate. @param payload Persistent local payload subrange.
     * @param message Capture-frozen semantic key and maximum bucket bytes.
     * @param rows Original INT32 live rows for a sender or an independently counted consumer.
     * @param row_bytes Exact semantic row size, zero only for a fixed extent.
     * @param received_rows Original consumer's output row field, derived from its fresh receipt.
     * @param stream This participant's original explicit stream.
     */
    void enqueue(ILocalTPContext &original, int participant, void *payload, CapturedTransferMessage message,
        const std::int32_t *rows, std::uint64_t row_bytes, std::int32_t *received_rows,
        NativePipelineByteObservation observation, void *stream) const;
    struct Endpoint;
    /** @brief Bind only the actual original immutable native ownership. */
    PipelineNativeByteChannel(ILocalTPContext &context, int source, std::vector<int> consumers,
        std::size_t capacity);
    NativeLocalTPContextBorrow original_;
    std::vector<GlobalDeviceAddress> members_;
    int source_;
    std::vector<int> consumers_;
    std::size_t capacity_;
    std::map<std::pair<int,int>, std::shared_ptr<Endpoint>> endpoints_;
};
/** @brief Original typed INT32 materializer row field, retained with its physical capture lease. */
struct PipelineNativeRowCount final {
    std::shared_ptr<TensorBase> owner;
    std::size_t offset = 0;
    std::uint64_t row_bytes = 0;
};
/** @brief Storage consumer's INT32 transport output, validated from the producer's actual completed bytes. */
struct PipelineNativeReceivedRows final {
    std::shared_ptr<TensorBase> owner;
    std::size_t offset = 0;
    std::uint64_t row_bytes = 0;
};
/** @brief One capture-frozen tensor/workspace subrange bound to its original native edge. */
class PipelineNativeByteBinding final {
public:
    /** @brief Retain an arena tensor and its captured physical storage before graph construction. */
    PipelineNativeByteBinding(std::shared_ptr<PipelineNativeByteChannel> channel, int participant,
        CapturedTransferMessage message, std::shared_ptr<TensorBase> owner, std::size_t offset = 0,
        std::optional<PipelineNativeRowCount> rows = {});
    /** @brief Retain a receive destination and its disjoint, receipt-owned row output. */
    PipelineNativeByteBinding(std::shared_ptr<PipelineNativeByteChannel> channel, int participant,
        CapturedTransferMessage message, std::shared_ptr<TensorBase> owner, std::size_t offset,
        PipelineNativeReceivedRows received);
    /** @brief Retain an already admitted workspace subrange with its original lifetime proof. */
    PipelineNativeByteBinding(std::shared_ptr<PipelineNativeByteChannel> channel, int participant,
        CapturedTransferMessage message, std::shared_ptr<const WorkspaceBufferLease> owner, std::size_t offset = 0);
    /** @brief Default copies retain the same original payload/capture/frame owners. */
    PipelineNativeByteBinding(const PipelineNativeByteBinding &) = default;
    /** @brief Transfer the complete ownership bundle. */
    PipelineNativeByteBinding(PipelineNativeByteBinding &&) noexcept = default;
    /** @brief Replace the ownership bundle without retiring a tensor ahead of its capture borrow. */
    PipelineNativeByteBinding &operator=(PipelineNativeByteBinding other) noexcept;
    /** @brief Move-bind actual terminal evidence for this one addressed native edge. */
    PipelineNativeByteBinding withTerminalReceipt(DeviceGenerationCollectiveReceipt receipt) &&;
    /** @brief Enqueue bytes with exact event-aware tensor input/output publication. */
    void enqueue(void *stream) const;
    /** @return Capture identity covering original owners, native membership and semantic geometry. */
    std::uint64_t captureIdentity() const noexcept;
private:
    using Rows = std::variant<std::monostate, PipelineNativeRowCount, PipelineNativeReceivedRows>;
    struct CheckedRows {}; ///< Select the common validating constructor only after typed public binding.
    /** @brief Validate the selected row authority together with its exact payload lifetime. */
    PipelineNativeByteBinding(std::shared_ptr<PipelineNativeByteChannel> channel, int participant,
        CapturedTransferMessage message, std::shared_ptr<TensorBase> owner, std::size_t offset, Rows rows, CheckedRows);
    /** @brief Reject changed or undersized original storage before enqueue. */
    void *data() const;
    std::shared_ptr<PipelineNativeByteChannel> channel_;
    int participant_;
    CapturedTransferMessage message_;
    std::variant<std::shared_ptr<TensorBase>,std::shared_ptr<const WorkspaceBufferLease>> owner_;
    std::size_t offset_;
    void *data_;
    Rows rows_;
    std::optional<DeviceGenerationCollectiveReceipt> receipt_;
    std::int32_t *row_address_ = nullptr; ///< Exact aligned device field embedded in the scalar capture nodes.
    std::shared_ptr<const CapturedDeviceStorageLease> row_capture_;
    std::shared_ptr<const CapturedDeviceStorageLease> capture_; ///< Retires before the original tensor owner.
};
} // namespace llaminar2
