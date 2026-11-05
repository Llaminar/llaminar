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

namespace llaminar2::qwen4exp {
class PooledImmutableDirectedTransport;
/** @brief Same-PMA static operands and native receipts for one original native cohort protocol, with no live admission ledger. */
class PooledImmutableNativeCompletionStorage final {
public:
    /** @brief Inline immutable original participants; the PMA-owned object prices this complete bounded container. */
    using Participants=std::array<std::optional<PreparedGPUContextUsePermit>,kMaximumPooledNativeParticipants>;
    /** @brief Optional original local frames; pure readers supply their real ReaderPhase frame at bind. */
    using Frames=std::array<std::optional<PooledNativeCompletionFrame>,kMaximumPooledNativeParticipants>;
    /** @brief Price only the original immutable operand/receipt backing.
     * @param host Original source host resource. @param participants Complete original native resources in coordinate order.
     * @param layout Exact original edge/message declaration.
     * @return Typed PMA contribution; LocalTP retains its own communicator resources. */
    static PhysicalMemoryPlan memoryBOM(PhysicalMemoryResource host,
        std::span<const PhysicalMemoryResource> participants, PooledNativeCompletionLayout layout);
    /** @return Original Rank-issued communicator/cohort, retained without a reverse strong model edge. */
    const std::shared_ptr<const Qwen4ExpReplicatedPLECohort>& cohort() const { requireCurrent(); return cohort_; }
    /** @return Original actual communicator coordinate, never physical allocator order. */
    int coordinate(DeviceId device) const;
    /** @return Same original prepared worker/PMA generation for a native participant. */
    const PreparedGPUContextUsePermit& original(DeviceId device) const;
    /** @return Immutable original borrow retained through abort; obtaining new native admission after abort is forbidden. */
    NativeCompletionAuthority authority(DeviceId device) const { return authorities_[coordinate(device)]; }
    /** @brief Authenticate original worker generations, native context and same PMA ownership. */
    void requireCurrent() const;
    /** @return Original storage-backed prefix completion, never a host snapshot of interval/error/count. */
    PooledImmutableNativeTransfer counted(PooledImmutableNativeTransfer binding,
        PooledNativeCountedMessage message, PooledNativeTransferSpan extent,
        std::optional<PooledNativeCompletionFrame> frame = {}) const;
private:
    friend class PooledImmutableDirectedTransport;
    /** @brief Revoke original caller bindings only after enclosing parent/cache retirement; storage releases with its final borrower. */
    void retireOriginalRuntime() const noexcept { retired_.store(true, std::memory_order_release); }
    /** @brief Materialize only exact admitted roots using original prepared workers before capture. */
    static std::shared_ptr<const PooledImmutableNativeCompletionStorage> create(
        std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort,
        Participants participants,
        Frames frames,
        PooledNativeCompletionLayout layout);
    /** @brief Retain the same original owners after all cold validation/allocation succeeds. */
    PooledImmutableNativeCompletionStorage(std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort,
        Participants participants,
        Frames frames, PooledNativeCompletionLayout layout,
        std::array<NativeCompletionAuthority, kMaximumPooledNativeParticipants> authorities,
        PhysicalMemoryAllocationLease mappedClaim, std::array<PhysicalMemoryAllocationLease, kMaximumPooledNativeParticipants> receiptClaims,
        std::shared_ptr<MappedHostTransferRegion> operands,
        std::array<std::shared_ptr<DeviceTransferBuffer>, kMaximumPooledNativeParticipants> receipts)
        : cohort_(std::move(cohort)), participants_(std::move(participants)), frames_(std::move(frames)), layout_(layout), authorities_(std::move(authorities)),
          mapped_claim_(std::move(mappedClaim)), receipt_claims_(std::move(receiptClaims)),
          operands_(std::move(operands)), receipts_(std::move(receipts)) {}
    std::shared_ptr<const Qwen4ExpReplicatedPLECohort> cohort_;
    Participants participants_;
    Frames frames_;
    PooledNativeCompletionLayout layout_;
    std::array<NativeCompletionAuthority, kMaximumPooledNativeParticipants> authorities_;
    mutable std::atomic<bool> retired_{false}; ///< Host lifetime only, never a model/protocol execution state.
    PhysicalMemoryAllocationLease mapped_claim_;
    std::array<PhysicalMemoryAllocationLease, kMaximumPooledNativeParticipants> receipt_claims_;
    std::shared_ptr<MappedHostTransferRegion> operands_;
    std::array<std::shared_ptr<DeviceTransferBuffer>, kMaximumPooledNativeParticipants> receipts_;
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
    /** @brief Bind only an original UINT64 publication before capture. */
    PooledImmutableTransferBinding withDeviceCounted(std::shared_ptr<TensorBase> root,
        std::size_t offset, PooledNativeCountedMessage message) const;
    /** @brief Bind a native original transfer-bank count and exact frame without pretending it is an arena tensor.
     * Cross-vendor bindings use their existing typed tensor/device-buffer APIs;
     * this overload rejects them instead of selecting another transport.
     */
    PooledImmutableTransferBinding withNativeCounted(PooledNativeTransferSpan extent,
        PooledNativeCountedMessage message, PooledNativeCompletionFrame frame) const;
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
