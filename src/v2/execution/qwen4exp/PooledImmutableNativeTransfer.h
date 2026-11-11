/**
 * @file PooledImmutableNativeTransfer.h
 * @brief Borrow the original TP communicator for ordered, participant-local pooled PLE transfers.
 *
 * This is a binding to the existing native collective authority, not another
 * transport engine or communicator registry. Original frame fields own live
 * extents and failure. Persistent completion metadata is admitted before capture;
 * neither a payload capacity nor a host observation becomes a wire extent.
 */
#pragma once

#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include "collective/NativeLocalTPContextBorrow.h"
#include "kernels/qwen4exp/PooledImmutableNativeOperands.h"
#include "collective/NativeCollectiveBytes.h"
#include "backends/NativeContextPreparation.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <optional>
#include <span>
#include <variant>
#include <vector>
#include <limits>

namespace llaminar2::qwen4exp {
/** @brief The counted messages in the existing source-owner protocol and final reader result. */
enum class PooledNativeCountedMessage : std::uint8_t {
    PinCommands, PinGrants, FreeDecisions, FreeGrants, VictimDecisions, VictimGrants,
    SourceFills, SourcePayload, ReadRequests, ReadRows, Releases, ReaderResult, LocalHotMissKeys, Count
};

/** @brief Optional reader semantics; shared-only message populations remain byte-identical. */
enum class PooledNativeReaderMessageProfile : std::uint8_t {ReplicatedFP32,CompactNative};

/**
 * @brief Exact immutable star-edge slots for original operands and local native receipts.
 * Source-private owner messages receive one slot per actual owner edge. Each
 * original reader has its own result slot, even at C0 or without physical cache.
 */
class PooledNativeCompletionLayout final {
public:
    /** @brief Declare two original endpoints through the same general star law.
     * @param positiveOwner Whether the one remote original owner has positive capacity. */
    explicit PooledNativeCompletionLayout(bool positiveOwner)
        :PooledNativeCompletionLayout(2,0,std::array<std::uint8_t,2>{0,std::uint8_t(positiveOwner)}) {}
    /** @brief Freeze original communicator order and actual positive remote-owner edges.
     * @param degree Exact complete native reader population. @param coordinator Original source coordinate.
     * @param positiveOwner One original-member flag per coordinate; source-self is not a remote edge. */
    PooledNativeCompletionLayout(std::size_t degree,std::size_t coordinator,std::span<const std::uint8_t> positiveOwner,
        PooledNativeReaderMessageProfile profile=PooledNativeReaderMessageProfile::ReplicatedFP32,
        std::span<const std::uint8_t> readers={})
        :degree_(degree),coordinator_(coordinator),profile_(profile) {
        if(profile!=PooledNativeReaderMessageProfile::ReplicatedFP32 && profile!=PooledNativeReaderMessageProfile::CompactNative)
            throw std::invalid_argument("Unknown native pooled reader message profile");
        if(degree<2 || degree>std::size_t(std::numeric_limits<int>::max()) || coordinator>=degree || positiveOwner.size()!=degree ||
            (!readers.empty() && readers.size()!=degree))
            throw std::invalid_argument("Native pooled layout requires exact admitted original star coordinates");
        positive_.resize(degree);readers_.assign(degree,1);
        for(std::size_t i=0;i<degree;++i){
            if(positiveOwner[i]>1 || (i==coordinator && positiveOwner[i]))
                throw std::invalid_argument("Native pooled layout cannot invent a positive source-self edge");
            positive_[i]=positiveOwner[i];
            if(!readers.empty()) {
                if(readers[i]>1)throw std::invalid_argument("Native pooled reader flag is not boolean");
                readers_[i]=readers[i];
            }
        }
        if(!readers_[coordinator_])throw std::invalid_argument("Native pooled coordinator must retain its original compute role");
        if(degree>(std::numeric_limits<std::size_t>::max()/sizeof(ncclLlaminarCompletionOperand))/26u)
            throw std::overflow_error("Native pooled completion metadata extent overflow");
        operand_allocation_bytes_=operandBytes()?TransferEngine::mappedHostRegionAllocationBytes(operandBytes()):0;
    }
    /** @return Exact original communicator population, independent of physical owners. */
    std::size_t degree()const noexcept{return degree_;}
    /** @return Declared optional wire semantics, never a live capability. */
    PooledNativeReaderMessageProfile readerProfile()const noexcept{return profile_;}
    /** @return Original source coordinate, independent of physical device ordinal. */
    std::size_t coordinator()const noexcept{return coordinator_;}
    /** @return Whether this actual endpoint has positive remote storage, independent of compute membership. */
    bool positiveOwner(std::size_t participant)const {requireParticipant(participant);return positive_[participant];}
    /** @return Whether this actual endpoint consumes query results; storage-only endpoints never do. */
    bool reader(std::size_t participant)const {requireParticipant(participant);return readers_[participant];}
    /** @return Counted slots in the original pair declaration. */
    std::size_t slots()const {requirePair();return edgeSlots(1-coordinator_);}
    /** @return Original nominal message coordinate in a pair. */
    std::size_t slot(PooledNativeCountedMessage message)const {requirePair();return messageSlot(message,1-coordinator_);}
    /** @return Only exact useful immutable operand bytes, excluding mapped page padding. */
    std::size_t operandBytes()const noexcept {
        std::size_t n=0;for(std::size_t p=0;p<degree_;++p)n+=participantSlots(p);
        return n*sizeof(ncclLlaminarCompletionOperand);
    }
    /** @return Exact TransferEngine mapped physical backing, never communication bytes. */
    std::size_t operandAllocationBytes()const noexcept{return operand_allocation_bytes_;}
    /** @brief Price only this participant's actual incident receipt slots.
     * @param participant Original communicator coordinate.
     * @return Exact local receipt bytes, independently from payload capacity. */
    std::size_t receiptBytesPerParticipant(std::size_t participant=0)const {
        requireParticipant(participant);return participantSlots(participant)*sizeof(ncclLlaminarByteCompletion);
    }
    /** @brief Authenticate each original physical root before binding.
     * @param operandAllocationBytes Actual mapped backing. @param receiptBytes Actual local receipt backing.
     * @param participant Exact original caller coordinate. @return Whether both roots match the typed BOM. */
    bool matchesStorageBytes(std::size_t operandAllocationBytes,std::size_t receiptBytes,std::size_t participant=0)const {
        return operandAllocationBytes==operand_allocation_bytes_ && receiptBytes==receiptBytesPerParticipant(participant);
    }
    /** @brief Resolve one immutable original endpoint/peer/message operand.
     * @param message Original protocol message class. @param participant Original local coordinate.
     * @param peer Actual opposite original coordinate.
     * @return Exact mapped operand offset, disjoint from all other edges and messages. */
    std::size_t operandOffset(PooledNativeCountedMessage message,int participant,int peer)const {
        const auto local=localSlot(message,participant,peer);std::size_t first=0;
        for(int p=0;p<participant;++p)first+=participantSlots(p);
        return (first+local)*sizeof(ncclLlaminarCompletionOperand);
    }
    /** @brief Resolve only this original local endpoint's receive evidence.
     * @param message Original protocol message class. @param participant Original local coordinate.
     * @param peer Actual opposite original coordinate.
     * @return Exact receipt offset, disjoint for every same-nominal edge. */
    std::size_t receiptOffset(PooledNativeCountedMessage message,int participant,int peer)const {
        return localSlot(message,participant,peer)*sizeof(ncclLlaminarByteCompletion);
    }
    /** @brief Resolve an explicitly two-endpoint declaration through the general edge law.
     * @param message Original protocol class. @param participant Original local coordinate.
     * @return Exact operand offset; plural stars require an explicit peer. */
    std::size_t operandOffset(PooledNativeCountedMessage message,int participant)const {
        requirePair();return operandOffset(message,participant,1-participant);
    }
    /** @brief Resolve an explicitly two-endpoint receipt through the general edge law.
     * @param message Original protocol message class.
     * @return Exact local receipt offset; plural stars require both original coordinates. */
    std::size_t receiptOffset(PooledNativeCountedMessage message)const {
        requirePair();return receiptOffset(message,0,1);
    }
private:
    /** @brief Validate an original coordinate before any offset arithmetic. */
    void requireParticipant(std::size_t participant)const {
        if(participant>=degree_)throw std::invalid_argument("Native pooled completion has a foreign original coordinate");
    }
    /** @brief Reject implicit endpoint inference when the original cohort contains multiple edges. */
    void requirePair()const {if(degree_!=2)throw std::invalid_argument("Plural native completion requires explicit original edge identity");}
    /** @return Exact static slots for one original source/reader pair. */
    std::size_t edgeSlots(std::size_t peer)const noexcept {
        return (positive_[peer]?11u:0u)+(readers_[peer]?(profile_==PooledNativeReaderMessageProfile::CompactNative?2u:1u):0u);
    }
    /** @return Exact total incident slots for one original endpoint. */
    std::size_t participantSlots(std::size_t p)const noexcept {
        if(p!=coordinator_)return edgeSlots(p);
        std::size_t n=0;for(std::size_t peer=0;peer<degree_;++peer)if(peer!=p)n+=edgeSlots(peer);return n;
    }
    /** @return Checked nominal slot, excluding absent optional owner messages at C0. */
    std::size_t messageSlot(PooledNativeCountedMessage message,std::size_t edge)const {
        const auto value=static_cast<std::size_t>(message);
        if(value>=static_cast<std::size_t>(PooledNativeCountedMessage::Count) ||
            ((message==PooledNativeCountedMessage::ReaderResult || message==PooledNativeCountedMessage::LocalHotMissKeys) && !readers_[edge]) ||
            (message==PooledNativeCountedMessage::LocalHotMissKeys && profile_!=PooledNativeReaderMessageProfile::CompactNative) ||
            (!positive_[edge] && message!=PooledNativeCountedMessage::ReaderResult && message!=PooledNativeCountedMessage::LocalHotMissKeys))
            throw std::invalid_argument("Native pooled completion has no original counted message on this edge");
        if(message==PooledNativeCountedMessage::LocalHotMissKeys)return positive_[edge]?12:1;
        return positive_[edge]?value:0;
    }
    /** @return Exact incident-edge index without allocating a second topology or runtime ledger. */
    std::size_t localSlot(PooledNativeCountedMessage message,int participant,int peer)const {
        requireParticipant(participant);requireParticipant(peer);
        if(participant==peer || (std::size_t(participant)!=coordinator_ && std::size_t(peer)!=coordinator_))
            throw std::invalid_argument("Native pooled completion edge is outside its original source star");
        const auto edge=std::size_t(participant)==coordinator_?std::size_t(peer):std::size_t(participant);
        std::size_t first=0;
        if(std::size_t(participant)==coordinator_)
            for(std::size_t prior=0;prior<edge;++prior)if(prior!=coordinator_)first+=edgeSlots(prior);
        return first+messageSlot(message,edge);
    }
    std::size_t degree_,coordinator_,operand_allocation_bytes_=0;
    PooledNativeReaderMessageProfile profile_;
    std::vector<std::uint8_t> positive_,readers_; ///< Cold declaration only; retained bindings project exact scalar offsets.
};

/** @brief Immutable scalar projection of one declared edge, without retaining a topology copy in every graph binding. */
struct PooledNativeCompletionBindingLayout final {
    std::size_t degree,coordinator,operand_offset,receipt_offset;
    PooledNativeReaderMessageProfile profile;
};

namespace detail {
/**
 * @brief Publish an original immutable operand once, or prove an exact reused binding.
 * @param target Same zero-initialized PMA-owned mapped slot, private to the original bank.
 * @param original Complete original pointer/message binding; no device field is read.
 * @throws std::logic_error Before overwriting a previous graph's different binding.
 * This is cold metadata construction, not a request/progress ledger. Reused
 * captured graphs cannot tolerate replacing their embedded operand contents.
 */
inline void sealPooledNativeOperand(ncclLlaminarCompletionOperand& target,
    const ncclLlaminarCompletionOperand& original)
{
    if (!original.table || !original.admission || !original.sequence || !original.firstError ||
        !original.receipt || !original.message || original.failureCode <= 0 || original.reserved)
        throw std::invalid_argument("Native pooled operand requires complete immutable original fields");
    const bool empty = !target.table && !target.admission && !target.sequence && !target.firstError &&
        !target.receipt && !target.outgoingPayloadBytes && !target.message && !target.failureCode && !target.reserved;
    if (empty) { target = original; return; }
    if (target.table != original.table || target.admission != original.admission || target.sequence != original.sequence ||
        target.firstError != original.firstError || target.receipt != original.receipt ||
        target.outgoingPayloadBytes != original.outgoingPayloadBytes || target.message != original.message ||
        target.failureCode != original.failureCode || target.reserved != original.reserved)
        throw std::logic_error("Native pooled captured operand changed its original immutable binding");
}
} // namespace detail

/** @brief One strong original physical subrange: canonical tensor or existing immutable transfer allocation. */
class PooledNativeTransferSpan final {
public:
    /** @brief Retain an original arena/native tensor role without materializing storage. */
    PooledNativeTransferSpan(PooledNativeTensorSpan original)
        : owner_(std::move(original.root)), role_(original.buffer), offset_(original.offset), bytes_(original.bytes) {}
    /** @brief Retain a canonical physical tensor when the enclosing graph, rather than this binding, owns its arena role. */
    PooledNativeTransferSpan(std::shared_ptr<TensorBase> original, std::size_t offset, std::size_t bytes)
        : owner_(std::move(original)), offset_(offset), bytes_(bytes) {}
    /** @brief Retain an original fixed transfer allocation, which has no invented arena tensor role. */
    PooledNativeTransferSpan(std::shared_ptr<DeviceTransferBuffer> original, std::size_t offset, std::size_t bytes)
        : owner_(std::move(original)), offset_(offset), bytes_(bytes) {}
    /** @brief Reject foreign, absent, aliased or out-of-range physical declarations. */
    void require(DeviceId device, std::size_t alignment = 1) const;
    /** @return Exact existing prepared address; never performs allocation, transfer or state publication. */
    void* pointer(DeviceId device, std::size_t alignment = 1) const;
    /** @return Canonical tensor owner or null for an existing transfer allocation. */
    std::shared_ptr<TensorBase> tensor() const noexcept {
        const auto* value = std::get_if<std::shared_ptr<TensorBase>>(&owner_); return value ? *value : nullptr;
    }
    /** @return Same original owner/offset with a checked cold source-bank declaration. */
    PooledNativeTransferSpan resized(std::size_t bytes) const;
    /**
     * @brief Derive a bounded field inside this same declared span, not the full root capacity.
     * @param relativeOffset Byte offset from the beginning of this original span.
     * @param bytes Positive exact field extent.
     * @return Same owner/control block and graph role with a strictly bounded nested range.
     * @throws std::exception For absent original ownership, overflow or an out-of-span field.
     */
    PooledNativeTransferSpan subspan(std::size_t relativeOffset, std::size_t bytes) const;
    /** @return Exact immutable physical-root control block identity. */
    bool sameOwner(const PooledNativeTransferSpan& other) const noexcept;
    /**
     * @brief Reject overlapping fields or aliased ownership control blocks.
     * @param other Another exact positive declaration in this same binding.
     * @throws std::invalid_argument For empty/overflowing ranges, overlapping bytes or foreign control blocks.
     * Both declarations retain original roots; different root types receive
     * their final native-address overlap check only when prepared storage seals.
     */
    void requireDisjoint(const PooledNativeTransferSpan& other) const;
    /** @return Complete immutable owner/type/role/subrange identity, excluding device values. */
    std::uint64_t identity() const noexcept;
    /** @return Bounded original source offset. */
    std::size_t offset() const noexcept { return offset_; }
    /** @return Bounded original storage subrange, never a live transfer count. */
    std::size_t bytes() const noexcept { return bytes_; }
private:
    friend class PooledImmutableTransferChannel;
    std::variant<std::shared_ptr<TensorBase>, std::shared_ptr<DeviceTransferBuffer>> owner_;
    std::optional<BufferId> role_;
    std::size_t offset_, bytes_;
};

/** @brief Original frame/error fields borrowed by the native library; none is copied into a shadow. */
struct PooledNativeCompletionFrame final {
    PooledNativeTransferSpan interval;
    PooledNativeTransferSpan error;
    std::int32_t failureCode;
    /**
     * @brief Check the response/pool purpose and exact disjoint original declarations.
     * @param nominal Existing response/owner message; MissKeys requires a request frame.
     * @param device Exact original GPU endpoint.
     * @param extent Original UInt64 live-byte field, never a host count.
     * @param payload Original useful payload bank, independent of metadata.
     */
    void require(PooledNativeCountedMessage nominal, DeviceId device,
        const PooledNativeTransferSpan& extent, const PooledNativeTransferSpan& payload) const;
};

/**
 * @brief Original request-lane identity/error fields, never a pool completion interval.
 *
 * The native ABI still borrows the same three UInt64 pointers. Their semantic
 * authority is the original immutable RequestLaneIdentity. A distinct type
 * prevents a local miss-key request from borrowing an unrelated pool interval.
 */
struct PooledNativeRequestCompletionFrame final {
    PooledNativeTransferSpan request_identity;
    PooledNativeTransferSpan error;
    std::int32_t failureCode;
    /**
     * @brief Check the compact request purpose and exact disjoint original fields.
     * @param profile Original immutable compact reader profile.
     * @param device Exact original GPU endpoint.
     * @param extent Original UInt64 live-byte field.
     * @param payload Original bounded compact miss-key bank.
     */
    void require(PooledNativeReaderMessageProfile profile, DeviceId device,
        const PooledNativeTransferSpan& extent, const PooledNativeTransferSpan& payload) const;
};

/**
 * @brief One immutable native endpoint inside an original continuation PLE graph.
 *
 * The caller supplies exact original admitted metadata storage and the Rank-issued
 * cohort. The original Rank retains that communicator through every native parent
 * join. A failed native enqueue aborts the same communicator, hence the complete
 * original TP request, never just a logical cache edge.
 */
class PooledImmutableNativeTransfer final {
public:
    /** @brief Copy only the same original physical borrowers and immutable capture declaration. */
    PooledImmutableNativeTransfer(const PooledImmutableNativeTransfer&) = default;
    /** @brief Move the complete physical owner set without another admission. */
    PooledImmutableNativeTransfer(PooledImmutableNativeTransfer&&) noexcept = default;
    /** @brief Replace a complete owner set before releasing old roots or allocation borrowers. */
    PooledImmutableNativeTransfer& operator=(PooledImmutableNativeTransfer other) noexcept;

    /**
     * @brief Bind an original counted prefix and its persistent operand/receipt storage.
     * @param nominal Original pool/response message; request MissKeys is rejected.
     * @param layout Original complete native completion-bank geometry.
     * @param operandBacking Same PMA-owned mapped metadata allocation, initialized before capture.
     * @param receipts Same original participant's admitted native receipt root.
     * @param extent Original UINT64 publication; never read on host.
     * @param frame Original local interval and sticky failure fields.
     */
    PooledImmutableNativeTransfer withPrefix(PooledNativeCountedMessage nominal,
        PooledNativeCompletionLayout layout, std::shared_ptr<MappedHostTransferRegion> operandBacking,
        std::shared_ptr<DeviceTransferBuffer> receipts, PooledNativeTransferSpan extent,
        PooledNativeCompletionFrame frame) const;
    /**
     * @brief Bind compact MissKeys to its actual request-lane identity before capture.
     * @param layout Original CompactNative completion bank declaration.
     * @param operandBacking Same PMA-owned mapped immutable operand storage.
     * @param receipts Original participant's admitted native receipt root.
     * @param extent Exact original UInt64 request byte publication.
     * @param frame Actual request identity/error authority, not a pool interval.
     * @return Same original native endpoint with a purpose-tagged counted binding.
     */
    PooledImmutableNativeTransfer withRequestPrefix(PooledNativeCompletionLayout layout,
        std::shared_ptr<MappedHostTransferRegion> operandBacking,
        std::shared_ptr<DeviceTransferBuffer> receipts, PooledNativeTransferSpan extent,
        PooledNativeRequestCompletionFrame frame) const;
    /**
     * @brief Bind a root-relative source slice; the receiver retains only its original acquired extent.
     * @param descriptor Original source descriptor; only the producer may supply it.
     * @param sourceBytes Actual admitted source bank extent, not a transfer length.
     */
    PooledImmutableNativeTransfer withSourceSlice(PooledNativeTransferSpan descriptor, std::size_t sourceBytes,
        std::uint64_t sourceOffsetOrigin = 0) const;
    /** @brief Declare the receiver half of the same source-slice operation, without inventing a source descriptor. */
    PooledImmutableNativeTransfer withReceivedSourceSlice() const;
    /** @brief Publish immutable operands on the original worker before parent capture; repeated preparation validates. */
    void prepare(void* stream);
    /** @brief Record one native fixed or completed counted Broadcast; first failure aborts original TP. */
    void enqueue(void* stream) const;
    /** @return Immutable actual-native receipt for a dependent reader ACK, absent for fixed records. */
    NativeReceiveCompletionView completion() const;
    /** @return Complete original communicator, physical storage and immutable operand identity. */
    std::uint64_t captureIdentity() const;
    /** @return Static useful record or counted maximum; no live value is downloaded. */
    CapturedTransferMessage message() const noexcept { return message_; }
    /** @return Original extent interpretation, compatible with the existing typed dispatch recipe. */
    CapturedTransferExtentSource extentSource() const noexcept {
        return extent_ ? (side_ == CapturedTransferEndpoint::Producer ? CapturedTransferExtentSource::ProducerDevice : CapturedTransferExtentSource::ProducerPublication) : CapturedTransferExtentSource::FixedMessage;
    }
    /** @return Whether this producer borrows the original source-relative slice descriptor. */
    bool hasSourceSlice() const noexcept { return slice_.has_value(); }
    /** @return Actual prepared worker/device of this participant-local endpoint. */
    DeviceId device() const noexcept { return original_.device(); }
    /** @return Original source/receiver operation role, independent of device ordinal. */
    CapturedTransferEndpoint role() const noexcept { return side_; }
private:
    friend class PooledImmutableTransferChannel;
    friend class PooledImmutableNativeCompletionStorage;
    /** @brief Fixed useful protocol record on the original communicator; no counted header is sent. */
    static PooledImmutableNativeTransfer fixed(
        NativeLocalTPContextBorrow cohort,
        PreparedGPUContextUsePermit original, CapturedTransferEndpoint side,
        int producerCoordinate, int consumerCoordinate, CapturedTransferMessage message, PooledNativeTransferSpan payload);
    /** @brief Construct only through checked fixed binding, then add typed count metadata. */
    PooledImmutableNativeTransfer(NativeLocalTPContextBorrow cohort,
        PreparedGPUContextUsePermit original, CapturedTransferEndpoint side,
        int producer, int consumer, int participant, CapturedTransferMessage message, PooledNativeTransferSpan payload);
    using CountedFrame = std::variant<PooledNativeCompletionFrame, PooledNativeRequestCompletionFrame>;
    /**
     * @brief Extend the same cold storage/operand path with one explicit frame purpose.
     * @param nominal Exact original counted message class.
     * @param layout Same original complete native completion-bank geometry.
     * @param operandBacking Same admitted mapped immutable operand bank.
     * @param receipts Same participant's actual admitted receipt root.
     * @param extent Original UInt64 live-byte publication.
     * @param frame Distinct pool or request identity/error authority.
     * @return Complete original endpoint retaining all original storage borrowers.
     */
    PooledImmutableNativeTransfer withCountedFrame(PooledNativeCountedMessage nominal,
        PooledNativeCompletionLayout layout, std::shared_ptr<MappedHostTransferRegion> operandBacking,
        std::shared_ptr<DeviceTransferBuffer> receipts, PooledNativeTransferSpan extent,
        CountedFrame frame) const;
    /** @return Original three-word pool or request identity span selected by its immutable purpose. */
    const PooledNativeTransferSpan& frameIdentity() const;
    /** @return Original sticky error field of the same purpose-tagged frame. */
    const PooledNativeTransferSpan& frameError() const;
    /** @return Original positive failure code embedded in the immutable native operand. */
    std::int32_t frameFailureCode() const;
    /** @brief Seal the complete original root set after each cold binding extension. */
    void sealStorage();
    /** @brief Validate exact native membership, strong physical roots and prepared worker generation. */
    void requireCurrent() const;
    /** @return Original prepared device subrange; rejects wrong root, extent, device or alignment. */
    void* pointer(const PooledNativeTransferSpan& span, std::size_t alignment = 1) const;
    NativeLocalTPContextBorrow cohort_;
    PreparedGPUContextUsePermit original_;
    CapturedTransferEndpoint side_;
    int producer_, consumer_, participant_;
    CapturedTransferMessage message_;
    NativeCompletionAuthority authority_;
    PooledNativeTransferSpan payload_;
    std::optional<PooledNativeTransferSpan> extent_, slice_;
    std::optional<CountedFrame> frame_;
    std::optional<PooledNativeCountedMessage> nominal_;
    std::optional<PooledNativeCompletionBindingLayout> layout_;
    std::shared_ptr<MappedHostTransferRegion> operands_;
    std::shared_ptr<DeviceTransferBuffer> receipts_;
    std::optional<Qwen4ExpPreparedStorage> prepared_;
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>, 5> borrowed_{};
    std::size_t source_bytes_ = 0;
    std::uint64_t source_origin_ = 0;
    bool slice_message_ = false;
    bool operand_published_ = false, sealed_ = false;
    std::array<void*, 5> addresses_{};
    IBackend* backend_ = nullptr;
    std::uint64_t runtime_generation_ = 0;
};
} // namespace llaminar2::qwen4exp
