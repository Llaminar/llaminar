/**
 * @file ImmutableLookupKernels.h
 * @brief Capture-safe native GPU publication/gather interface for immutable row I/O.
 *
 * Setup binds TransferEngine-owned mapped pages and persistent device scratch.
 * Request publication exposes only live INT64 lookup IDs and protocol metadata.
 * Response acquisition uses one waiting CTA, so CPU-service progress never
 * depends on a GPU producer starved by a full-SM spin grid. Native bytes are
 * copied exactly once to private device scratch before ack and native decoding.
 */
#pragma once
#include "Qwen4ExpKernels.h"
#include "../../execution/qwen4exp/ImmutableLookupProtocol.h"
#include "../../backends/DeviceId.h"
#include "../common/NativeEmbeddingIQTables.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace llaminar2::qwen4exp
{
/** @brief Setup-authenticated device aliases; lifetime is retained by the service binding. */
struct ImmutableLookupDeviceView
{
    ImmutableLookupRequest *request = nullptr;
    ImmutableLookupCompletion *completion = nullptr;
    ImmutableLookupAcknowledgement *acknowledgement = nullptr;
    std::int64_t *ids = nullptr;
    std::byte *nativeRows = nullptr;
    std::uint64_t table = 0;
    std::uint64_t maximumIds = 0;
    std::uint64_t rowBytes = 0;
};
/** @brief Immutable I/O identity published once at request admission. */
struct ImmutableLookupAdmissionArgs
{
    /** @brief Storage materialization policy; request reset never reinitializes protocol history. */
    enum class ProtocolInitialization { Preserve,InitializeOnce };
    std::uint64_t generation=0; ///< API-owned session identity, independent of model decisions.
    std::uint64_t *deviceGeneration=nullptr; ///< Persistent GPU protocol identity word.
    ImmutableLookupDeviceState *state=nullptr; ///< Exact admitted protocol allocation.
    ProtocolInitialization initialization=ProtocolInitialization::Preserve;
};
/** @brief GPU request publication from raw hash output, with no host token access. */
struct ImmutableLookupPublishArgs
{
    RowExtent tokenRows; ///< Token rows, each contributing exactly sixteen lookup IDs.
    const std::int64_t *ids = nullptr; ///< [capacity,16] exact device hash output.
    const std::uint64_t *generation = nullptr; ///< Canonical device admission generation.
    ImmutableLookupDeviceState *state = nullptr; ///< Persistent GPU-owned monotonically sequenced state.
    ImmutableLookupDeviceView channel; ///< Setup-authenticated mapped addresses.
};
/** @brief GPU response consumer and exact native decode publication. */
struct ImmutableLookupConsumeArgs
{
    ImmutableLookupDeviceState *state = nullptr;
    ImmutableLookupDeviceView channel;
    std::byte *nativeScratch = nullptr; ///< [maximumIds*rowBytes], PMA-admitted device bytes.
    float *output = nullptr; ///< [maximumIds,rowElements] FP32 gathered embedding rows.
    int *healthy = nullptr; ///< Canonical generation-health word; cleared on a service failure.
    int *error = nullptr; ///< Canonical terminal-error word, storing the typed lookup status.
};
/** @brief Explicit terminal close of the complete captured generation service interval. */
struct ImmutableLookupCloseArgs
{
    const std::uint64_t *generation = nullptr; ///< Canonical GPU admission generation.
    const ImmutableLookupDeviceState *state = nullptr; ///< Final issued/consumed protocol positions.
    ImmutableLookupDeviceView channel;
    const int *healthy=nullptr; ///< Optional canonical terminal predicate, paired with complete.
    const int *complete=nullptr; ///< Publish only if unhealthy or complete; null pair is unconditional.
};
/** @brief Stateless native GPU bundle; protocol history is owned by persistent device state. */
class IImmutableLookupKernels
{
public:
    /** @brief Destroy setup-only codebook and geometry bindings. */
    virtual ~IImmutableLookupKernels() = default;
    /** @brief Publish immutable admission identity on the exact non-default stream.
     * @param args Positive session identity and persistent GPU destination.
     * @param stream Exact admission producer stream, outside capture.
     */
    virtual void admit(const ImmutableLookupAdmissionArgs &args,void *stream) const=0;
    /** @brief Publish one exact live ID transaction on the exact non-default stream.
     * @param args Complete immutable channel/source/state binding.
     * @param stream Exact producer stream, mandatory and non-default.
     */
    virtual void publish(const ImmutableLookupPublishArgs &args,void *stream) const = 0;
    /** @brief Acquire/authenticate native rows, copy exact bytes, ack and decode.
     * @param args Complete output, private scratch and canonical error bindings.
     * @param stream Exact consumer stream ordered after its request producer.
     */
    virtual void consume(const ImmutableLookupConsumeArgs &args,void *stream) const = 0;
    /** @brief Close the CPU-service interval at the complete parent graph terminal.
     * @param args Exact generation/channel and final GPU protocol state.
     * @param stream Exact terminal stream ordered after all lookup consumers.
     */
    virtual void close(const ImmutableLookupCloseArgs &args,void *stream) const = 0;
};
/** @brief Construct native CUDA bindings and immutable IQ codebooks before capture. */
std::unique_ptr<IImmutableLookupKernels> createCUDAImmutableLookupKernels(DeviceId device,TensorType format,int rowElements,std::size_t rowBytes);
/** @brief Construct native HIP bindings and immutable IQ codebooks before capture. */
std::unique_ptr<IImmutableLookupKernels> createROCmImmutableLookupKernels(DeviceId device,TensorType format,int rowElements,std::size_t rowBytes);
/** @brief Decode exact CPU-owned native rows without converting the table or allocating scratch.
 * @param format Authenticated original GGUF source type.
 * @param rowElements Exact logical elements per native row.
 * @param rowBytes Exact bytes per complete original row.
 * @param nativeRows Caller-owned contiguous live native rows.
 * @param count Exact live lookup row count; zero performs no payload work.
 * @param output Caller-owned FP32 output, count*rowElements elements.
 */
void decodeCPUImmutableLookupRows(TensorType format,int rowElements,std::size_t rowBytes,
    const std::byte *nativeRows,std::size_t count,float *output);
} // namespace llaminar2::qwen4exp
