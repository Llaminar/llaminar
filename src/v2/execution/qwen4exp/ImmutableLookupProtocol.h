/**
 * @file ImmutableLookupProtocol.h
 * @brief Exact live-row protocol for a retained GPU/CPU immutable-table boundary.
 *
 * One serialized inference lane owns one setup-created mapped slot. GPU-issued
 * sequence numbers remain monotonic across request resets; admission generations
 * authenticate request identity without exposing tokens or mutable model state.
 * The CPU sees only immutable IDs and native payload metadata. Separate release
 * words transfer request/completion ownership, while GPU acknowledgement protects
 * response reuse. Closing an interval is explicit graph metadata, never a host
 * inference-loop decision. No allocation capacity is a communication extent.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "loaders/CpuSubordinateImmutableRowRead.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QFN_LOOKUP_HD __host__ __device__
#else
#define QFN_LOOKUP_HD
#endif

namespace llaminar2::qwen4exp
{
/** @brief Fixed protocol identity, including layout version. */
inline constexpr std::uint64_t kImmutableLookupProtocol = 0x51464e4c4f4f4b01ULL;
/** @brief Explicit subordinate live-intent/invalidation-ACK ABI, never inferred from storage mode. */
inline constexpr std::uint64_t kImmutableLookupSubordinateProtocol = 0x51464e4c4f4f4b02ULL;
/** @brief CPU storage/protocol terminal status, independent of inference state. */
enum class ImmutableLookupStatus : std::uint64_t
{
    Success = 0,
    InvalidProtocol,
    StaleGeneration,
    InvalidSequence,
    WrongTable,
    InvalidExtent,
    InvalidRowId,
    StorageFailure,
    Cancelled,
    TimedOut,
    InvalidAcknowledgement,
};
/** @brief GPU-authored immutable request body; publication is stored last. */
struct alignas(64) ImmutableLookupRequest
{
    std::uint64_t publication = 0; ///< System-release sequence after all ID payload writes.
    std::uint64_t sequenceComplement = ~std::uint64_t{0};
    std::uint64_t generation = 0, generationComplement = ~std::uint64_t{0};
    std::uint64_t table = 0, tableComplement = ~std::uint64_t{0};
    std::uint64_t idCount = 0, idCountComplement = ~std::uint64_t{0};
    std::uint64_t idBytes = 0, idBytesComplement = ~std::uint64_t{0};
    std::uint64_t rowBytes = 0, rowBytesComplement = ~std::uint64_t{0};
    std::uint64_t protocol = kImmutableLookupProtocol;
    std::uint64_t protocolComplement = ~kImmutableLookupProtocol;
};
/** @brief CPU-authored immutable response; publication releases exact native rows. */
struct alignas(64) ImmutableLookupCompletion
{
    std::uint64_t publication = 0; ///< System-release matching request sequence.
    std::uint64_t sequenceComplement = ~std::uint64_t{0};
    std::uint64_t generation = 0, generationComplement = ~std::uint64_t{0};
    std::uint64_t table = 0, tableComplement = ~std::uint64_t{0};
    std::uint64_t idCount = 0, idCountComplement = ~std::uint64_t{0};
    std::uint64_t nativeBytes = 0, nativeBytesComplement = ~std::uint64_t{0};
    std::uint64_t status = 0, statusComplement = ~std::uint64_t{0};
    std::uint64_t protocol = kImmutableLookupProtocol;
    std::uint64_t protocolComplement = ~kImmutableLookupProtocol;
};
/** @brief GPU-authored response retirement and complete captured-interval close. */
struct alignas(64) ImmutableLookupAcknowledgement
{
    std::uint64_t consumedSequence = 0; ///< Release after every response-byte reader has completed.
    std::uint64_t closedGeneration = 0; ///< Release only at the complete generation graph's terminal.
    std::uint64_t reserved[6]{};
};
/** @brief Private GPU-owned slot state, retained unchanged across request resets.
 * @details issuedSequence never resets while the mapped channel is retained.
 * Request generation comes from the canonical admission/control tensor on GPU.
 */
struct alignas(64) ImmutableLookupDeviceState
{
    std::uint64_t issuedSequence = 0;
    std::uint64_t consumedSequence = 0;
    std::uint64_t generation = 0;
    std::uint64_t idCount = 0;
    std::uint64_t nativeBytes = 0;
    std::uint64_t status = 0;
    std::uint64_t admissionGeneration=0; ///< Immutable I/O interval identity, published once at admission.
    std::uint64_t reserved=0;
};

/** @brief Required wire metadata; aligned struct tail bytes are storage only. */
inline constexpr std::size_t kLookupRequestLiveBytes=offsetof(ImmutableLookupRequest,protocolComplement)+sizeof(std::uint64_t);
/** @brief Exact response metadata, excluding allocation-alignment tail bytes. */
inline constexpr std::size_t kLookupCompletionLiveBytes=offsetof(ImmutableLookupCompletion,protocolComplement)+sizeof(std::uint64_t);

/** @brief One checked byte layout shared by BOM admission, CPU service and GPU bindings. */
class ImmutableLookupLayout final
{
public:
    /** @brief Resolve exact storage, with no anonymous reserve or independent admission.
     * @param maximumIds Positive maximum lookup IDs in one captured transaction.
     * @param nativeRowBytes Positive exact source bytes per immutable table row.
     * @return Immutable, cache-line-separated metadata and payload coordinates.
     * @throws std::invalid_argument or std::overflow_error For invalid or unrepresentable geometry.
     */
    static ImmutableLookupLayout create(std::size_t maximumIds,std::size_t nativeRowBytes);
    /**
     * @brief Price one bounded GPU-subordinate source channel with live intent and ACK frames.
     * @param maximumIds Positive source-authored compact miss bound.
     * @param nativeRowBytes Exact unchanged original native row bytes.
     * @return Distinct protocol identity and disjoint bounded native/metadata storage.
     * @throws std::exception For invalid or overflowing geometry.
     * @note This CPU protocol prerequisite advertises no GPU producer/consumer kernel support.
     */
    static ImmutableLookupLayout createSubordinate(std::size_t maximumIds,std::size_t nativeRowBytes);
    /** @return Explicit channel ABI identity, never guessed from a provider or payload. */
    std::uint64_t protocol() const noexcept { return protocol_; }
    /** @return Whether acquired requests must carry subordinate intent/ACK metadata. */
    bool subordinate() const noexcept { return protocol_==kImmutableLookupSubordinateProtocol; }
    /** @return Intent frame offset; rejects an ordinary channel instead of aliasing its payload. */
    std::size_t intentsOffset() const;
    /** @return Exact invalidation ACK frame offset; rejects an ordinary channel. */
    std::size_t invalidationsOffset() const;
    /** @brief Maximum admitted IDs; not a communication count. */
    std::size_t maximumIds() const noexcept { return maximumIds_; }
    /** @brief Exact native bytes per source row. */
    std::size_t rowBytes() const noexcept { return rowBytes_; }
    /** @brief GPU request header byte offset. */
    std::size_t requestOffset() const noexcept { return 0; }
    /** @brief CPU completion header byte offset. */
    std::size_t completionOffset() const noexcept { return sizeof(ImmutableLookupRequest); }
    /** @brief GPU retirement/terminal metadata byte offset. */
    std::size_t acknowledgementOffset() const noexcept { return sizeof(ImmutableLookupRequest)+sizeof(ImmutableLookupCompletion); }
    /** @brief Compact INT64 ID payload byte offset. */
    std::size_t idsOffset() const noexcept { return idsOffset_; }
    /** @brief Native source-row response payload byte offset. */
    std::size_t payloadOffset() const noexcept { return payloadOffset_; }
    /** @brief Total requested mapped capacity before TransferEngine page rounding. */
    std::size_t bytes() const noexcept { return bytes_; }
private:
    /** @brief Construct only from overflow-checked geometry. */
    ImmutableLookupLayout(std::size_t maximumIds,std::size_t rowBytes,std::size_t ids,
        std::size_t payload,std::size_t bytes) noexcept
        :maximumIds_(maximumIds),rowBytes_(rowBytes),idsOffset_(ids),payloadOffset_(payload),bytes_(bytes) {}
    std::size_t maximumIds_,rowBytes_,idsOffset_,payloadOffset_,bytes_;
    std::uint64_t protocol_=kImmutableLookupProtocol;
    std::size_t intentsOffset_=0,invalidationsOffset_=0;
};

/** @brief Pure shared validation used before either side dereferences a payload. */
struct ImmutableLookupProtocol final
{
    /** @brief Validate an acquired request without reading any token/model state. */
    QFN_LOOKUP_HD static ImmutableLookupStatus validateRequest(const ImmutableLookupRequest &request,
        std::uint64_t acquiredSequence,std::uint64_t expectedGeneration,std::uint64_t expectedTable,
        std::uint64_t previousSequence,std::uint64_t maximumIds,std::uint64_t rowBytes,
        std::uint64_t expectedProtocol=kImmutableLookupProtocol) noexcept
    {
        if((expectedProtocol!=kImmutableLookupProtocol && expectedProtocol!=kImmutableLookupSubordinateProtocol)||
            request.publication!=acquiredSequence||request.protocol!=expectedProtocol||request.protocolComplement!=~expectedProtocol||
            request.sequenceComplement!=~acquiredSequence||request.generationComplement!=~request.generation||
            request.tableComplement!=~request.table||request.idCountComplement!=~request.idCount||
            request.idBytesComplement!=~request.idBytes||request.rowBytesComplement!=~request.rowBytes)
            return ImmutableLookupStatus::InvalidProtocol;
        if(request.generation!=expectedGeneration||expectedGeneration==0)return ImmutableLookupStatus::StaleGeneration;
        if(acquiredSequence==0||previousSequence==~std::uint64_t{0}||acquiredSequence!=previousSequence+1)
            return ImmutableLookupStatus::InvalidSequence;
        if(expectedTable==0||request.table!=expectedTable)return ImmutableLookupStatus::WrongTable;
        if(rowBytes==0||request.rowBytes!=rowBytes||request.idCount>maximumIds||
            request.idCount>~std::uint64_t{0}/sizeof(std::int64_t)||
            request.idBytes!=request.idCount*sizeof(std::int64_t)||request.idCount>~std::uint64_t{0}/rowBytes)
            return ImmutableLookupStatus::InvalidExtent;
        return ImmutableLookupStatus::Success;
    }
    /** @brief Validate acquired completion identity and exact native bytes before payload reads. */
    QFN_LOOKUP_HD static ImmutableLookupStatus validateCompletion(const ImmutableLookupCompletion &completion,
        std::uint64_t acquiredSequence,std::uint64_t expectedSequence,std::uint64_t generation,
        std::uint64_t table,std::uint64_t idCount,std::uint64_t rowBytes,
        std::uint64_t expectedProtocol=kImmutableLookupProtocol) noexcept
    {
        if((expectedProtocol!=kImmutableLookupProtocol && expectedProtocol!=kImmutableLookupSubordinateProtocol)||
            completion.publication!=acquiredSequence||completion.protocol!=expectedProtocol||completion.protocolComplement!=~expectedProtocol||
            completion.sequenceComplement!=~acquiredSequence||completion.generationComplement!=~completion.generation||
            completion.tableComplement!=~completion.table||completion.idCountComplement!=~completion.idCount||
            completion.nativeBytesComplement!=~completion.nativeBytes||completion.statusComplement!=~completion.status)
            return ImmutableLookupStatus::InvalidProtocol;
        if(acquiredSequence!=expectedSequence||expectedSequence==0)return ImmutableLookupStatus::InvalidSequence;
        if(completion.generation!=generation||generation==0)return ImmutableLookupStatus::StaleGeneration;
        if(completion.table!=table||table==0)return ImmutableLookupStatus::WrongTable;
        if(completion.idCount!=idCount||rowBytes==0||idCount>~std::uint64_t{0}/rowBytes||
            completion.status>static_cast<std::uint64_t>(ImmutableLookupStatus::InvalidAcknowledgement))
            return ImmutableLookupStatus::InvalidExtent;
        const auto status=static_cast<ImmutableLookupStatus>(completion.status);
        if(completion.nativeBytes!=(status==ImmutableLookupStatus::Success?idCount*rowBytes:0))
            return ImmutableLookupStatus::InvalidExtent;
        return status;
    }
};

static_assert(sizeof(ImmutableLookupRequest)==128&&sizeof(ImmutableLookupCompletion)==128);
static_assert(sizeof(ImmutableLookupAcknowledgement)==64&&sizeof(ImmutableLookupDeviceState)==64);
static_assert(std::is_trivially_copyable_v<ImmutableLookupRequest> && std::is_trivially_copyable_v<ImmutableLookupCompletion>);
} // namespace llaminar2::qwen4exp
#undef QFN_LOOKUP_HD
