/**
 * @file ImmutableFixedHeadLookupKernels.h
 * @brief Original-native fixed-head shard operations, separate from transport.
 *
 * The checked plan projects source metadata and an already selected local range;
 * it neither selects placement nor owns bytes. The graph must order original
 * query publication, local validation, mandatory fixed header allgather, cohort
 * validation, native packing, counted payload allgather and decode. Every peer
 * submits both collective nodes even on failure. The derived INT32 is solely a
 * transport extent and must never replace original query/controller authority.
 */
#pragma once
#include "QwenFixedHeadPipelineCounters.h"
#include "ImmutableResidentTableRangeView.h"
#include "ImmutableResidentLookupKernels.h"
#include "../common/DeviceRowRange.h"
#include "QwenFixedHeadTerminalCounters.h"
#include <memory>

namespace llaminar2::qwen4exp
{
/** @brief Exactly eight mandatory progress bytes per participant, including empty/error work. */
struct FixedHeadValidationHeader
{
    std::int32_t originalLiveRows; ///< Original scalar verbatim, including invalid publications.
    std::int32_t firstError; ///< Existing canonical error or local validation failure.
    bool operator==(const FixedHeadValidationHeader &) const = default;
};
static_assert(sizeof(FixedHeadValidationHeader)==8);
static_assert(alignof(FixedHeadValidationHeader)==alignof(std::int32_t));

/** @brief Immutable checked projection of existing source/head/range ownership. */
class ImmutableFixedHeadLookupPlan final
{
public:
    /**
     * @brief Authenticate equal head groups and the exact gap-preserving source range.
     * @param table Exact local range from the sole source/storage authority.
     * @param policy Complete original hash policy, including legal unused source gaps.
     * @param participants Actual ordered cohort size, initially two, four or eight.
     * @param participant This range owner's original ordered collective/head coordinate.
     * @return Checked immutable geometry; no allocation, placement or execution occurs.
     * @throws std::invalid_argument For unsupported degree, coordinate or range/head mismatch.
     * @details The resolved declaration assigns head group p to actual collective
     * participant p. Its physical-owner-sorted range vector is not that order:
     * composition must authenticate assignmentFor(owner).participant against the
     * real collective rank, never infer it from device ordinals or vector index.
     */
    static ImmutableFixedHeadLookupPlan bind(ImmutableResidentTableRangeView table,
        NgramHashPolicy policy,int participants,int participant);
    /** @return Original source allocation/range identity, not a full-table reinterpretation. */
    const ImmutableResidentTableRangeView &table() const noexcept { return table_; }
    /** @return Complete source hash metadata, unchanged by this projection. */
    const NgramHashPolicy &policy() const noexcept { return policy_; }
    /** @return Frozen rank-major result source count. */
    int participants() const noexcept { return participants_; }
    /** @return Frozen local source coordinate in the actual collective. */
    int participant() const noexcept { return participant_; }
    /** @return Exactly 16/participants selected native rows per live token. */
    int heads() const noexcept { return kNgramHeads/participants_; }
    /** @return First globally numbered hash head served by this source. */
    int firstHead() const noexcept { return participant_*heads(); }
    /** @return Whether complete source allocation, head metadata and cohort coordinate agree. */
    bool operator==(const ImmutableFixedHeadLookupPlan &other) const noexcept;
private:
    /** @brief Only bind() constructs validated, immutable derived geometry. */
    ImmutableFixedHeadLookupPlan(ImmutableResidentTableRangeView table,NgramHashPolicy policy,int p,int r)
        :table_(table),policy_(policy),participants_(p),participant_(r) {}
    ImmutableResidentTableRangeView table_;
    NgramHashPolicy policy_;
    int participants_,participant_;
};

/**
 * @brief Persistent borrowed operands for one graph-local lookup transaction.
 *
 * Original IDs/count remain immutable between validation and decode. The caller
 * retains every owner through terminal join and graph retirement. Header and
 * payload receive banks are written by explicit existing collective stages;
 * these kernels never communicate, allocate, synchronize or inspect host state.
 */
struct ImmutableFixedHeadLookupArgs
{
    DeviceRowRange tokenRows; ///< Complete original prefix descriptor, including M1/live0.
    const std::int64_t *ids=nullptr; ///< Original global IDs [M,16].
    std::size_t idCapacity=0; ///< INT64 physical elements, never a communication extent.
    FixedHeadValidationHeader *localHeader=nullptr; ///< Sole local validation writer, exactly8B.
    const FixedHeadValidationHeader *gatheredHeaders=nullptr; ///< Collective-owned [P] header bank.
    std::size_t headerCapacity=0; ///< Header elements, at least P.
    std::int32_t *transportRows=nullptr; ///< Local validation writes required header count1; consensus then writes original L/0 for payload.
    std::byte *localNative=nullptr; ///< Pack-owned native [M,H,B], suffix untouched.
    std::size_t localNativeBytes=0; ///< Exact admitted packet capacity.
    const std::byte *gatheredNative=nullptr; ///< Collective-owned [P,M,H,B], fixed rank stride.
    std::size_t gatheredNativeBytes=0; ///< Exact admitted rank-major result capacity.
    float *output=nullptr; ///< Decode-owned [M,16,width], suffix untouched.
    std::size_t outputElements=0; ///< Exact admitted FP32 result capacity.
    int *healthy=nullptr; ///< Existing canonical health; never initialized by lookup.
    int *error=nullptr; ///< Existing canonical first error; never cleared by lookup.
    QwenFixedHeadCounterRow counters{}; ///< Optional original request row; native operations never allocate or reset it.
};

/** @brief Complete immutable kernel-frame identity; excludes every mutable scalar value. */
struct ImmutableFixedHeadLookupCaptureIdentity
{
    ImmutableFixedHeadLookupPlan plan;
    DeviceRowRange::CaptureIdentity originalRows;
    const std::int64_t *ids;
    std::size_t idCapacity;
    FixedHeadValidationHeader *localHeader;
    const FixedHeadValidationHeader *gatheredHeaders;
    std::size_t headerCapacity;
    std::int32_t *transportRows;
    std::byte *localNative;
    std::size_t localNativeBytes;
    const std::byte *gatheredNative;
    std::size_t gatheredNativeBytes;
    float *output;
    std::size_t outputElements;
    int *healthy,*error;
    void *stream;
    std::array<const void *,6> codebooks{}; ///< Exact setup-published IQ grid pointers, empty in pure validation.
    QwenFixedHeadCounterRow counters{}; ///< Original optional terminal observation binding.
    bool operator==(const ImmutableFixedHeadLookupCaptureIdentity &) const = default;
};

namespace fixed_head_lookup_detail
{
/** @brief Checked extent arithmetic for source-native packet/output banks. */
std::size_t product(std::size_t a,std::size_t b);
/**
 * @brief Authenticate original whole-prefix identity and all borrowed storage before capture.
 * @param plan Checked immutable source/head projection.
 * @param args Persistent frame; no mutable device byte is read here.
 * @param stream Exact non-default stream retained by the graph executor.
 * @return Complete capture identity for the caller's stage/cache signature.
 * @throws std::invalid_argument For repeated/sliced rows, capacity, overlap or stream mismatch.
 * @throws std::overflow_error For an unrepresentable admitted extent.
 */
ImmutableFixedHeadLookupCaptureIdentity validate(const ImmutableFixedHeadLookupPlan &plan,
    const ImmutableFixedHeadLookupArgs &args,void *stream);
}

/** @brief Stateless CUDA/HIP operation bundle; no table, protocol or count ownership. */
class IImmutableFixedHeadLookupKernels
{
public:
    /** @brief Destroy setup bindings only after all captured consumers retire. */
    virtual ~IImmutableFixedHeadLookupKernels()=default;
    /** @return Exact checked source/head projection embedded in each launch. */
    virtual const ImmutableFixedHeadLookupPlan &plan() const noexcept=0;
    /**
     * @brief Authenticate complete frame and include the actual prepared decoder pointers.
     * @param args Persistent borrowed frame whose static geometry is sealed before capture.
     * @param stream Exact native stream embedded in the recording.
     * @return Complete immutable identity including original count and source/codebook allocation.
     */
    virtual ImmutableFixedHeadLookupCaptureIdentity captureIdentity(
        const ImmutableFixedHeadLookupArgs &args,void *stream) const=0;
    /**
     * @brief Publish an8B header and transport count1 even when canonical failure already exists.
     * @param args Original count/IDs and persistent frame.
     * @param stream Exact ordered native stream.
     * @details Checks only locally owned heads, against their own original vocabularies.
     * Invalid IDs/counts set canonical failure and header failure before exchange.
     */
    virtual void validateLocal(const ImmutableFixedHeadLookupArgs &args,void *stream) const=0;
    /**
     * @brief Validate all fixed gathered headers and overwrite transportRows on every replay.
     * @param args Gathered headers ordered after their actual native collective.
     * @param stream Exact ordered native stream.
     * @details Any invalid/disagreeing count/error yields zero transport payload on every peer.
     */
    virtual void validateCohort(const ImmutableFixedHeadLookupArgs &args,void *stream) const=0;
    /**
     * @brief Copy only validated live owned-head rows in their original source-native representation.
     * @param args Frame ordered after cohort validation, before native payload collective.
     * @param stream Exact ordered native stream.
     * @details Empty/error execution leaves every packet byte untouched, including old successes.
     */
    virtual void pack(const ImmutableFixedHeadLookupArgs &args,void *stream) const=0;
    /**
     * @brief Decode gathered rank-major native rows into original global-head order.
     * @param args Frame ordered after the actual counted payload collective.
     * @param stream Exact ordered native stream.
     * @details Uses the canonical original-format decoder; zero/error touches no payload/output.
     */
    virtual void decode(const ImmutableFixedHeadLookupArgs &args,void *stream) const=0;
};
/**
 * @brief Prepare only the explicitly selected compiled native backend.
 * @param device Original participant device; CPU and invalid identities reject.
 * @param plan Checked original source/head geometry, unchanged by selection.
 * @return The selected backend's actual immutable native decoder bundle.
 * @throws std::invalid_argument If the device is not a supported native GPU or
 * its backend was disabled at compilation. No alternate factory is attempted.
 * @details Implemented by the common kernel owner at cold preparation. Disabled
 * backends introduce no unresolved factory references in CPU-only binaries.
 */
std::unique_ptr<IImmutableFixedHeadLookupKernels> createImmutableFixedHeadLookupKernels(
    DeviceId device,ImmutableFixedHeadLookupPlan plan);
/** @brief Prepare CUDA codebook bindings outside capture.
 * @param plan Checked CUDA range and source geometry.
 * @return Stateless original-native operation bundle. */
std::unique_ptr<IImmutableFixedHeadLookupKernels> createCUDAImmutableFixedHeadLookupKernels(ImmutableFixedHeadLookupPlan plan);
/** @brief Prepare HIP codebook bindings outside capture.
 * @param plan Checked ROCm range and source geometry.
 * @return Stateless original-native operation bundle. */
std::unique_ptr<IImmutableFixedHeadLookupKernels> createROCmImmutableFixedHeadLookupKernels(ImmutableFixedHeadLookupPlan plan);

/** @brief A retained exact-width launch with its independently ordered device publication. */
struct FixedHeadPipelineRows {
    int exact; ///< Exact positive width admitted by the selected graph, or zero for metadata only.
    const std::int32_t *live; ///< Original device count. Null only for a declared fully active input.
};
/** @brief Canonical graph controller cells, borrowed without a new error owner. */
struct FixedHeadPipelineError {
    std::int32_t *healthy;
    std::int32_t *error;
    FixedHeadPipelineCounterRow counters{}; ///< Same original control row, optional for focused math replay.
};
/** @brief One immutable contiguous interval of original logical heads. */
struct FixedHeadPipelineHeads {
    int first;
    int count;
};
/** @brief Producer-side checked request header and collective-owned transport extent. */
struct FixedHeadPipelineConsensus {
    FixedHeadPipelineRows rows;
    FixedHeadPipelineError control;
    FixedHeadValidationHeader *local;
    const FixedHeadValidationHeader *gathered;
    std::int32_t *transport;
    int participants;
};
/** @brief Exact original ID subrange and compact destination for one addressed storage owner. */
struct FixedHeadPipelineRequest {
    FixedHeadPipelineRows rows;
    FixedHeadPipelineHeads heads;
    const std::int64_t *original;
    std::int64_t *compact;
};
/** @brief Pack one exact admitted range into either a local contribution or a compact reply. */
struct FixedHeadPipelinePack {
    FixedHeadPipelineRows rows;
    FixedHeadPipelineError control;
    const std::int64_t *ids;
    std::byte *destination;
    int id_stride; ///< INT64 coordinates per token in this original or compact input.
    int id_offset; ///< First owned head within each token's ID packet.
    int destination_heads; ///< Exact destination native rows per token.
    int destination_offset; ///< First owned head within each destination token.
};
/** @brief Merge one verified compact returned range into its actual PLE contribution. */
struct FixedHeadPipelineMerge {
    FixedHeadPipelineRows rows;
    FixedHeadPipelineError control;
    const std::byte *compact;
    std::byte *destination;
    int heads;
    int destination_heads;
    int destination_offset;
};
/** @brief An actual complete native packet and optional independently acquired extent. */
struct FixedHeadPipelineDecodePacket {
    const std::byte *data = nullptr;
    const std::uint64_t *receipt_bytes = nullptr; ///< Unequal native broadcast's completed useful extent.
    FixedHeadPipelineHeads heads{};
};
/** @brief Bounded exact model head topology, with no padded communication slots. */
struct FixedHeadPipelineDecode {
    FixedHeadPipelineRows rows;
    FixedHeadPipelineError control;
    FixedHeadPipelineDecodePacket packets[kNgramHeads];
    int participants;
    float *output;
};
/** @brief Shared native math over one actual original immutable table range. */
class IImmutableFixedHeadPipelineKernels {
public:
    virtual ~IImmutableFixedHeadPipelineKernels() = default;
    /** @brief Authenticate the selected width before publishing any usable request extent. */
    virtual void prepareConsumer(FixedHeadPipelineConsensus args, void *stream) const = 0;
    /** @brief Publish the current local status after every addressed exchange has drained. */
    virtual void sealConsumer(FixedHeadPipelineConsensus args, void *stream) const = 0;
    /** @brief Authenticate the original live count across the actual PLE consumer roster. */
    virtual void validateConsensus(FixedHeadPipelineConsensus args, void *stream) const = 0;
    /** @brief Pack only IDs the actual addressed range owner needs. */
    virtual void projectRequest(FixedHeadPipelineRequest args, void *stream) const = 0;
    /** @brief Gather exact native rows from this immutable range after per-head address validation. */
    virtual void pack(FixedHeadPipelinePack args, void *stream) const = 0;
    /** @brief Drain invalid storage work with a zero-row reply and preserve canonical first error. */
    virtual void sealResponse(std::int32_t *rows, FixedHeadPipelineError control, void *stream) const = 0;
    /** @brief Compare the completed producer extent to the original addressed request. */
    virtual void validateResponse(FixedHeadPipelineRows request, const std::int32_t *received,
        FixedHeadPipelineError control, void *stream) const = 0;
    /** @brief Place completed remote native rows in the original global-head contribution order. */
    virtual void merge(FixedHeadPipelineMerge args, void *stream) const = 0;
    /** @brief Publish exact variable local byte extent before the original native collective. */
    virtual void publishExtent(FixedHeadPipelineRows rows, int heads, std::uint64_t *extent, void *stream) const = 0;
    /** @brief Decode complete gathered contributions using the original source codec. */
    virtual void decode(FixedHeadPipelineDecode args, void *stream) const = 0;
};
/** @brief Bind actual original range and head metadata for this native backend. */
std::unique_ptr<IImmutableFixedHeadPipelineKernels> createImmutableFixedHeadPipelineKernels(
    DeviceId device, ImmutableResidentTableRangeView table, NgramHashPolicy policy,
    FixedHeadPipelineHeads heads);
/** @brief Shared checked bound for native range preparation; no GPU reads or allocation. */
void requireImmutableFixedHeadPipelineRange(ImmutableResidentTableRangeView table,
    const NgramHashPolicy &policy, FixedHeadPipelineHeads heads);
/** @brief Prepare CUDA codebooks and exact native range before capture. */
std::unique_ptr<IImmutableFixedHeadPipelineKernels> createCUDAImmutableFixedHeadPipelineKernels(
    ImmutableResidentTableRangeView table, NgramHashPolicy policy, FixedHeadPipelineHeads heads);
/** @brief Prepare HIP codebooks and exact native range before capture. */
std::unique_ptr<IImmutableFixedHeadPipelineKernels> createROCmImmutableFixedHeadPipelineKernels(
    ImmutableResidentTableRangeView table, NgramHashPolicy policy, FixedHeadPipelineHeads heads);
}
