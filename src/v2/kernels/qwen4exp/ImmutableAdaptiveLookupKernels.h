/**
 * @file ImmutableAdaptiveLookupKernels.h
 * @brief Native single-GPU adaptive component over the explicit subordinate source ABI.
 *
 * Factories retain the service-issued source binding. Existing canonical prepared
 * Tensor roots retain cache/frame/protocol bytes, and the actual generation
 * controller supplies health/error. No tier selector or serving mode is added.
 */
#pragma once
#include "AdaptiveImmutableRowCacheNativePlan.h"
#include "ImmutableSubordinateLookupBinding.h"
#include "../common/DeviceRowRange.h"
#include <memory>
namespace llaminar2
{
class DeviceGenerationControlBinding;
class TensorBase;
} // namespace llaminar2
namespace llaminar2::qwen4exp
{
class AdaptiveLookupPreparedStorage;
/** @brief Exact native storage after its existing root/runtime seal has been authenticated. */
struct AdaptiveLookupDeviceStorage
{
    std::byte *cache, *frame;
    ImmutableLookupDeviceState *protocol;
};
/** @brief Original complete counted prefix plus native IDs and output, borrowed through terminal retirement.
 */
struct AdaptiveLookupQuery
{
    DeviceRowRange token_rows;
    const std::int64_t *ids;
    std::size_t id_capacity;
    float *output;
    std::size_t output_elements;
};
/** @brief Private native launch operands; public callers cannot bypass checked enqueue. */
struct AdaptiveLookupDeviceArgs
{
    AdaptiveNativeParameters plan;
    AdaptiveLookupDeviceStorage storage;
    AdaptiveLookupQuery query;
    ImmutableSubordinateLookupDeviceView channel;
    int *healthy, *error;
    std::int32_t *control; ///< Same authenticated canonical request row, declared InOut by the actual stage.
};
/** @brief Stateless code/geometry bundle retaining its one actual source service/backing. */
class IImmutableAdaptiveLookupKernels
{
public:
    /** @brief Retire only after all borrowed native graphs/readers have retired. */
    virtual ~IImmutableAdaptiveLookupKernels() = default;
    /** @return Complete checked immutable source/policy geometry. */
    virtual const AdaptiveImmutableRowCacheNativePlan &plan() const noexcept = 0;
    /** @return Actual source owner/binding retained by this bundle. */
    virtual const ImmutableSubordinateLookupBinding &source() const noexcept = 0;
    /**
     * @brief Submit virgin model metadata exactly once; C0 still initializes the mandatory frame.
     * @param storage Source-issued shared setup owner of the original prepared roots.
     * @param exact_stream Actual non-default setup producer stream; its publication is retained.
     * @throws std::exception Before duplicate initialization, during capture, or on failed partial setup.
     * @note Completion/order follows TransferEngine publication. No host reads prove device completion.
     */
    void initialize(AdaptiveLookupPreparedStorage &storage, void *exact_stream) const;
    /**
     * @brief Publish immutable admission; every reset preserves warmed cache and I/O sequence.
     * @param storage Same initialized source-issued setup owner.
     * @param generation Positive API interval generation, independent of token/model decisions.
     * @param initialization InitializeOnce only on first admission; Preserve on every later admission.
     * @param exact_stream Actual ordered admission producer stream, outside capture.
     * @throws std::exception On wrong setup order, invalid generation, duplicate initialization or partial
     * publication.
     */
    void admit(AdaptiveLookupPreparedStorage &storage, std::uint64_t generation,
               ImmutableLookupAdmissionArgs::ProtocolInitialization initialization, void *exact_stream) const;
    /**
     * @brief Enqueue one complete native component frame, reusing original canonical controller ownership.
     * @param storage Original roots retained through all readers and terminal close. Runtime enqueue
     * requires admission; initialized-only storage is legal solely for the current executor stage
     * declaring the exact original protocol as a setup address-only external.
     * @param query Original complete prefix and native operands; no count is read on the host.
     * @param control Actual canonical request health/error authority, authenticated without mutable-value
     * reads.
     * @param exact_stream Exact native stream with enclosing graph producer/frontier dependencies already
     * joined.
     * @throws std::exception Before enqueue on wrong storage, geometry, stream, controller or operand
     * aliasing, or a cold call lacking that exact setup-only dependency ledger.
     * @note Cold recording neither launches work nor activates a source request; the executor must
     * acquire actual admission bytes before replay. This primitive does not certify IDs/count request
     * provenance. The checked stage must own
     * those producer edges, all root accesses/publications, complete query/control capture identity and
     * retirement.
     */
    void gather(AdaptiveLookupPreparedStorage &storage, const AdaptiveLookupQuery &query,
                const DeviceGenerationControlBinding &control, void *exact_stream) const;
    /** @return Complete source/geometry/prepared-root identity; the stage additionally binds its actual query
     * and controller. */
    std::uint64_t captureIdentity(const AdaptiveLookupPreparedStorage &storage) const;
    /**
     * @brief Unconditionally close only at an ordered terminal/reset boundary.
     * @param storage Actual admitted original frame/protocol owner.
     * @param exact_stream Exact terminal stream after every native reader and mapped acknowledgement.
     * @throws std::exception On absent initialization, foreign storage or invalid stream.
     * @note The native operation rejects an active frame or unretired response; it performs no host wait.
     */
    void close(AdaptiveLookupPreparedStorage &storage, void *exact_stream) const;
    /**
     * @brief Retained-ticket boundary: close only on canonical request failure or completion.
     * @param storage Actual admitted original frame/protocol owner.
     * @param control Actual request controller; the caller cannot substitute health/completion pointers.
     * @param exact_stream Exact retained transaction terminal stream ordered after all native readers.
     * @throws std::exception Before enqueue on invalid original controller/storage/stream.
     * @note A healthy incomplete HIP ticket leaves the storage interval open; failure or RequestComplete
     * closes it.
     */
    void closeWhenTerminal(AdaptiveLookupPreparedStorage &storage,
                           const DeviceGenerationControlBinding &control, void *exact_stream) const;

protected:
    /** @return Immutable native codebook, timeout and launch-policy identity, with no mutable values. */
    virtual std::uint64_t nativeCaptureIdentity() const noexcept = 0;
    /** @brief Backend implementation receives only already-authenticated native root coordinates. */
    virtual void initializeNative(AdaptiveLookupDeviceStorage storage, void *stream) const = 0;
    /** @brief Backend admission updates only original protocol words, never optional warmed policy. */
    virtual void admitNative(AdaptiveLookupDeviceStorage storage, std::uint64_t generation,
                             ImmutableLookupAdmissionArgs::ProtocolInitialization initialization,
                             void *stream) const = 0;
    /** @brief Record exact native metadata/source/copy/decode operations; no host readbacks or scratch
     * allocation. */
    virtual void gatherNative(const AdaptiveLookupDeviceArgs &args, void *stream) const = 0;
    /** @brief Record the existing explicit native service terminal, with no per-token wait. */
    virtual void closeNative(AdaptiveLookupDeviceStorage storage, const int *healthy, const int *complete,
                             void *stream) const = 0;
};
/**
 * @brief Bind CUDA original codebooks and the source-issued protocol2 owner before capture.
 * @param plan Checked immutable original source/policy/offset declaration.
 * @param source Real service-issued mapped source binding, retained by the returned bundle.
 * @return Stateless native code/geometry bundle; it owns neither query nor controller roots.
 * @throws std::exception For unsupported/foreign source, codebook, format or device setup.
 */
std::unique_ptr<IImmutableAdaptiveLookupKernels>
createCUDAImmutableAdaptiveLookupKernels(AdaptiveImmutableRowCacheNativePlan plan,
                                         ImmutableSubordinateLookupBinding source);
/**
 * @brief Bind HIP original codebooks and source owner under the same native/policy/byte contract.
 * @param plan Checked immutable original source/policy/offset declaration.
 * @param source Real service-issued mapped source binding retained by this bundle.
 * @return Native code/geometry bundle with the authenticated backend wall-clock deadline.
 * @throws std::exception For unsupported/foreign source, codebook, format, device or wall-clock setup.
 */
std::unique_ptr<IImmutableAdaptiveLookupKernels>
createROCmImmutableAdaptiveLookupKernels(AdaptiveImmutableRowCacheNativePlan plan,
                                         ImmutableSubordinateLookupBinding source);
} // namespace llaminar2::qwen4exp
