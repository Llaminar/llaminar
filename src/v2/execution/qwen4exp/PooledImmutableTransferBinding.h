/**
 * @file PooledImmutableTransferBinding.h
 * @brief Closed pooled endpoint bindings to the original native collective or cross-vendor channel.
 *
 * Homogeneous inline continuation graphs borrow their original NCCL/RCCL
 * communicator. Mixed-vendor graphs retain the existing captured transfer owner.
 * Selection is cold and explicit; enqueue never chooses or retries a route.
 */
#pragma once
#include "PooledImmutableNativeTransfer.h"
#include "transfer/CapturedTransferChannel.h"
#include <variant>
#include <atomic>
#include "collective/LocalTPContext.h"

namespace llaminar2::qwen4exp {
class PooledImmutableDirectedTransport;
/**
 * @brief Same-PMA metadata for the original native transport population.
 *
 * Transport membership includes storage-only endpoints independently of model
 * reader membership. One exact CPU root holds the owner and a trailing endpoint
 * record for every actual member; no fixed GPU-count capacity or parallel ledger
 * exists. Graph bindings borrow scalar offsets and the original revocable context.
 */
class PooledImmutableNativeCompletionStorage final {
public:
    /** @brief Cold construction inputs; final records move into the admitted root. */
    using Participants=std::vector<std::optional<PreparedGPUContextUsePermit>>;
    /** @brief Pure readers supply their actual ReaderPhase frame when binding. */
    using Frames=std::vector<std::optional<PooledNativeCompletionFrame>>;
    /** @brief Price the exact owner, actual endpoint records and operand/receipt roots. */
    static PhysicalMemoryPlan memoryBOM(PhysicalMemoryResource host,
        std::span<const PhysicalMemoryResource> participants, PooledNativeCompletionLayout layout);
    /** @return Checked borrow of the actual native communicator, never a model-reader alias. */
    NativeLocalTPContextUse context() const {requireCurrent();return cohort_.acquire();}
    /** @return Original actual native coordinate, independent of logical reader order. */
    int coordinate(DeviceId device) const;
    /** @return Same prepared worker/PMA generation of one native participant. */
    const PreparedGPUContextUsePermit& original(DeviceId device) const;
    /** @return The original completion authority retained through native abort. */
    NativeCompletionAuthority authority(DeviceId device) const {return participants_[coordinate(device)].authority;}
    /** @brief Authenticate original context, worker generations and exact physical roots. */
    void requireCurrent() const;
    /** @brief Bind the declared counted edge to its own source or reader frame. */
    PooledImmutableNativeTransfer counted(PooledImmutableNativeTransfer binding,
        PooledNativeCountedMessage message, PooledNativeTransferSpan extent,
        std::optional<PooledNativeCompletionFrame> frame = {}) const;
    /** @brief Bind compact miss keys to the actual request lane and its error authority. */
    PooledImmutableNativeTransfer countedRequest(PooledImmutableNativeTransfer binding,
        PooledNativeTransferSpan extent, PooledNativeRequestCompletionFrame frame) const;
    /** @brief Retire endpoint aliases before native context and backing ownership. */
    ~PooledImmutableNativeCompletionStorage();
private:
    friend class PooledImmutableDirectedTransport;
    friend class PooledImmutableTransferChannel;
    friend class PooledImmutableTransferBinding;
    /** @brief One actual endpoint's immutable declaration and exact physical owners. */
    struct Participant final {
        std::optional<PreparedGPUContextUsePermit> original;
        std::optional<PooledNativeCompletionFrame> frame;
        NativeCompletionAuthority authority;
        std::size_t receipt_bytes=0;
        bool positive_owner=false,reader=false;
        PhysicalMemoryAllocationLease receipt_claim;
        std::shared_ptr<DeviceTransferBuffer> receipt;
    };
    /** @return Exact aligned trailing-record offset in the sole admitted CPU root. */
    static std::size_t participantOffset();
    /** @return Checked owner-plus-record payload extent, independent of GPU capacity. */
    static std::size_t objectPayloadBytes(std::size_t degree);
    /** @brief Reconstruct immutable declaration for cold binding only; replay never allocates. */
    PooledNativeCompletionLayout declaration() const;
    /** @brief Reject a counted endpoint from another native context or PMA generation. */
    int requireOriginalBinding(const PooledImmutableNativeTransfer& binding) const;
    /** @brief Revoke callers after original enclosing parents have retired. */
    void retireOriginalRuntime() const noexcept {retired_.store(true,std::memory_order_release);}
    /** @brief Materialize exact admitted roots on their original prepared workers. */
    static std::shared_ptr<const PooledImmutableNativeCompletionStorage> create(
        NativeLocalTPContextBorrow cohort,std::unique_ptr<LocalTPContext> ownedContext,
        Participants participants,Frames frames,PooledNativeCompletionLayout layout);
    /** @brief Attach the trailing span before construction of each endpoint record. */
    PooledImmutableNativeCompletionStorage(NativeLocalTPContextBorrow cohort,
        std::unique_ptr<LocalTPContext> ownedContext,std::span<Participant> participants,
        const PooledNativeCompletionLayout& layout)
        :owned_context_(std::move(ownedContext)),cohort_(std::move(cohort)),participants_(participants),
         coordinator_(layout.coordinator()),profile_(layout.readerProfile()),
         operand_bytes_(layout.operandBytes()),operand_allocation_bytes_(layout.operandAllocationBytes()) {}
    std::unique_ptr<LocalTPContext> owned_context_; ///< Absent only when borrowing the exact original compute communicator.
    NativeLocalTPContextBorrow cohort_;
    std::span<Participant> participants_; ///< In the same CPUStorageRoot as this object, never an independent allocation.
    std::size_t coordinator_,constructed_=0;
    PooledNativeReaderMessageProfile profile_;
    std::size_t operand_bytes_,operand_allocation_bytes_;
    mutable std::atomic<bool> retired_{false}; ///< Host lifetime only; no model state.
    PhysicalMemoryAllocationLease mapped_claim_;
    std::shared_ptr<MappedHostTransferRegion> operands_;
};

/** @brief Same original storage/stream/count binding under one capture-frozen transport law. */
class PooledImmutableTransferBinding final {
public:
    /** @brief Copy the same complete original owners without another admission. */
    PooledImmutableTransferBinding(const PooledImmutableTransferBinding&) = default;
    /** @brief Move the complete original owner set. */
    PooledImmutableTransferBinding(PooledImmutableTransferBinding&&) noexcept = default;
    /** @brief Replace ownership as one unit so former payload borrows retire before their bank/PMA claims. */
    PooledImmutableTransferBinding& operator=(PooledImmutableTransferBinding other) noexcept {
        using std::swap; swap(completion_, other.completion_); swap(binding_, other.binding_); return *this;
    }
    /** @brief Retain an already authenticated original cross-vendor binding. */
    PooledImmutableTransferBinding(CapturedTransferBinding original) : binding_(std::move(original)) {}
    /** @return Original fixed/count/source-slice capture identity, including explicit transport. */
    std::uint64_t captureIdentity() const;
    /** @return Original positive fixed length or admitted counted maximum. */
    CapturedTransferMessage message() const noexcept;
    /** @return Original device publication law; no live field is read. */
    CapturedTransferExtentSource extentSource() const noexcept;
    /** @return Whether this original producer uses a source-relative slice. */
    bool hasSourceSlice() const noexcept;
    /** @return Exact original participant. */
    DeviceId device() const noexcept;
    /** @return Capture-frozen producer/consumer role. */
    CapturedTransferEndpoint role() const noexcept;
    /** @return True only for the original native communicator binding. */
    bool nativeCollective() const noexcept;
    /** @return Revocable original native recording authority, absent only on the mapped contract. */
    std::optional<NativeLocalTPContextBorrow> originalNativeContext() const;
    /** @brief Bind only an original UINT64 publication before capture. */
    PooledImmutableTransferBinding withDeviceCounted(std::shared_ptr<TensorBase> root,
        std::size_t offset, PooledNativeCountedMessage message) const;
    /** @brief Bind a native original transfer-bank count and exact frame without pretending it is an arena tensor.
     * Cross-vendor bindings use their existing typed tensor/device-buffer APIs;
     * this overload rejects them instead of selecting another transport.
     */
    PooledImmutableTransferBinding withNativeCounted(PooledNativeTransferSpan extent,
        PooledNativeCountedMessage message, PooledNativeCompletionFrame frame) const;
    /**
     * @brief Bind native compact MissKeys using its original request-lane identity.
     * @param extent Exact original UInt64 byte publication.
     * @param frame Original request identity/error authority, not a response interval.
     * @return Same original native binding and completion owner; mapped bindings reject.
     */
    PooledImmutableTransferBinding withNativeRequestCounted(PooledNativeTransferSpan extent,
        PooledNativeRequestCompletionFrame frame) const;
    /** @brief Bind an original source descriptor and exact subspan, preserving its immutable offset origin. */
    PooledImmutableTransferBinding withDeviceSliced(std::shared_ptr<TensorBase> root,
        std::size_t offset, std::size_t sourceBytes, std::uint64_t sourceOrigin,
        PooledNativeCountedMessage message) const;
    /** @brief Publish immutable native operands on the original worker before capture; mapped preparation is unchanged. */
    void prepare(void* stream) const;
    /** @brief Submit only the selected original transport on the exact participant stream. */
    void enqueue(void* stream) const;
    /** @return Only the original mapped completion; native callers must use nativeCompletion(). */
    CapturedTransferReceiveCompletion mappedCompletion(void* stream) const;
    /** @return Only the original completed native receipt; mapped callers must use mappedCompletion(). */
    NativeReceiveCompletionView nativeCompletion() const;
    /** @brief Bind the original consumer invocation to its actual physical owner, with mapped/native authority kept distinct. */
    void bindOriginalInvocation(PooledImmutableOwnerPreparedStorage& owner) const;
    /** @return Original captured binding, for the existing mapped invocation cancellation owner. */
    const CapturedTransferBinding& captured() const;
private:
    friend class PooledImmutableTransferChannel;
    friend class PooledImmutableCapturedPinEdge;
    /** @brief Release the same physical owner's original invocation at its closed edge lifetime boundary. */
    void releaseOriginalInvocation(PooledImmutableOwnerPreparedStorage& owner) const;
    /** @brief Construct native bindings only through an authenticated original native metadata owner. */
    PooledImmutableTransferBinding(PooledImmutableNativeTransfer original,
        std::shared_ptr<const PooledImmutableNativeCompletionStorage> completion)
        : completion_(std::move(completion)), binding_(std::move(original)) {}
    // Destruction releases native physical borrows/buffer aliases before the
    // bank's final physical owners and PMA allocation claims.
    std::shared_ptr<const PooledImmutableNativeCompletionStorage> completion_;
    mutable std::variant<CapturedTransferBinding, PooledImmutableNativeTransfer> binding_;
};

/** @brief Copyable logical endpoint owner; its original physical owners are shared, never duplicated. */
class PooledImmutableTransferChannel final {
public:
    /** @brief Represent a missing cold endpoint until the actual factory succeeds. */
    PooledImmutableTransferChannel() = default;
    /** @brief Retain the original cross-vendor channel without a second physical wrapper allocation. */
    PooledImmutableTransferChannel(std::shared_ptr<CapturedTransferChannel> original)
        : captured_(std::move(original)) {}
    /** @brief Borrow one exact edge of the original native completion owner.
     * @param original Same admitted native operand/receipt owner.
     * @param producer Original producing device. @param consumer Actual consuming device.
     * @param maximumBytes Retained useful-message bound, never a live transfer extent.
     * @return Immutable addressed channel; no native communicator or alternate route is created. */
    static PooledImmutableTransferChannel native(
        std::shared_ptr<const PooledImmutableNativeCompletionStorage> original, DeviceId producer, DeviceId consumer,
        std::size_t maximumBytes);
    /** @return Whether the cold factory supplied an original physical owner. */
    explicit operator bool() const noexcept { return bool(captured_) || bool(native_); }
    /** @return Whether this declaration uses the original native library authority. */
    bool nativeCollective() const noexcept { return bool(native_); }
    /** @return Static admitted endpoint envelope; never the communication's live extent. */
    std::size_t maximumBytes() const noexcept { return captured_ ? captured_->capacityBytes() : maximum_; }
    /** @return Exact original producer or receiver device. */
    DeviceId endpointDevice(CapturedTransferEndpoint side) const;
    /** @return The original native metadata owner, absent for cross-vendor movement. */
    const std::shared_ptr<const PooledImmutableNativeCompletionStorage>& nativeStorage() const noexcept { return native_; }
    /** @return The original captured channel, absent for native movement. */
    const std::shared_ptr<CapturedTransferChannel>& captured() const noexcept { return captured_; }
    /** @brief Preserve an exact original bounded span/role while binding this native or captured endpoint. */
    PooledImmutableTransferBinding bind(CapturedTransferEndpoint,CapturedTransferMessage,PooledNativeTransferSpan)const;
    /** @return Original fixed endpoint, ready for an optional typed count/slice extension. */
    PooledImmutableTransferBinding bind(CapturedTransferEndpoint side, CapturedTransferMessage message,
        std::shared_ptr<TensorBase> root, std::size_t offset) const;
    /** @brief Bind an original protocol allocation; no invented tensor role or storage copy is introduced. */
    PooledImmutableTransferBinding bind(CapturedTransferEndpoint side, CapturedTransferMessage message,
        std::shared_ptr<DeviceTransferBuffer> root, std::size_t offset) const;
private:
    std::shared_ptr<CapturedTransferChannel> captured_;
    std::shared_ptr<const PooledImmutableNativeCompletionStorage> native_;
    int producer_ = -1, consumer_ = -1;
    std::size_t maximum_ = 0;
};
} // namespace llaminar2::qwen4exp
