/**
 * @file GPUPooledImmutableReplicatedReader.cuh
 * @brief Symmetric integer-only native handshake work around one pooled reader.
 *
 * Every launch is one ordered metadata writer. Payload stays in the original
 * optimized lookup and captured-transfer kernels. A failed handshake preserves
 * the original first error and publishes no successful payload/ACK.
 */
#pragma once
#include "PooledImmutableReplicatedReaderKernels.h"
#if defined(QFN_POOLED_NATIVE_CUDA)
#include <cuda/atomic>
#endif

namespace llaminar2::qwen4exp::QFN_NATIVE_NAMESPACE {
namespace rr = replicated_reader;
/** @brief Use the same system-acquire law as the original captured channel lowering. */
__device__ inline std::uint64_t readerSystemAcquire(const std::uint64_t* address)
{
#if defined(QFN_POOLED_NATIVE_CUDA)
    cuda::atomic_ref<std::uint64_t,cuda::thread_scope_system> word(*const_cast<std::uint64_t*>(address));
    return word.load(cuda::memory_order_acquire);
#else
    return __hip_atomic_load(address, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
/** @brief Reload descriptor fields after their original publication acquire, avoiding stale mapped cache lines. */
__device__ inline std::uint64_t readerPublishedWord(const std::uint64_t* address)
{
#if defined(QFN_POOLED_NATIVE_CUDA)
    return __ldcv(reinterpret_cast<const unsigned long long*>(address));
#else
    return __builtin_nontemporal_load(address);
#endif
}
/** @brief Fail the original controller without clearing an earlier model/storage error. */
__device__ inline void failReplicatedReader(rr::NativeArgs args)
{
    atomicCAS(args.first_error, static_cast<int>(sampling_math::DeviceGenerationError::None),
        static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure));
    *args.healthy = 0;
    if (args.published_bytes) *args.published_bytes = 0;
}
/** @brief Execute one immutable reader phase; no source ticket or count is minted here. */
template<rr::Action Action>
__global__ void replicatedReader(rr::NativeArgs args)
{
    bool accepted = false;
    if constexpr (Action == rr::Action::PrepareResultHeader) {
        accepted = rr::prepare(*args.source_frame, args.geometry, args.version,
            args.first_error, *args.publication);
        accepted = accepted && args.source_frame->interval.table == args.source_table;
        if (!accepted) args.publication->status = rr::Status::Cancelled;
    } else if constexpr (Action == rr::Action::AcceptResultHeader) {
        accepted = rr::accept(*args.received, *args.received_bytes, args.geometry,
            args.version, args.source_table, static_cast<std::uint32_t>(args.original_rows.activeRows()),
            args.first_error, *args.publication);
    } else if constexpr (Action == rr::Action::PrepareResult) {
        accepted = rr::valid(args.geometry) && rr::forwardValid(args.first_error) &&
            args.source_frame->phase == rr::p::LookupPhase::Ready &&
            args.source_frame->status == rr::Status::Success &&
            args.source_frame->original_ids <= args.geometry.maximum_original;
        *args.published_bytes = accepted ?
            rr::outputBytes(args.geometry, args.source_frame->original_ids) : 0;
    } else if constexpr (Action == rr::Action::AcknowledgeResult) {
        const auto* original = args.payload_completion.publication();
        const auto epoch = readerSystemAcquire(&original->epoch);
        const CapturedTransferMessage message{readerPublishedWord(&original->message.key),
            readerPublishedWord(&original->message.bytes)};
        const auto cancellation = readerSystemAcquire(args.payload_completion.cancellationAddress());
        const auto status = args.payload_completion.status(*args.received_bytes, cancellation, epoch, message);
        accepted = rr::acknowledge(*args.publication, *args.received_bytes,
            args.geometry, args.first_error, status);
    } else if constexpr (Action == rr::Action::ConfirmAck) {
        accepted = rr::confirm(*args.received, *args.received_bytes, *args.source_frame,
            args.geometry, args.version, args.first_error);
    }
    if (!accepted) failReplicatedReader(args);
}
} // namespace llaminar2::qwen4exp::QFN_NATIVE_NAMESPACE

namespace llaminar2::qwen4exp::replicated_reader {
#if defined(QFN_POOLED_NATIVE_CUDA)
#define QFN_READER_ENQUEUE enqueueCUDAPooledReplicatedReader
#else
#define QFN_READER_ENQUEUE enqueueROCmPooledReplicatedReader
#endif
void QFN_READER_ENQUEUE(Action action, NativeArgs args, void* exact_stream)
{
    QFN_NATIVE_NAMESPACE::require(args.device, exact_stream);
    const bool source = action == Action::PrepareResultHeader || action == Action::PrepareResult || action == Action::ConfirmAck;
    const bool receives = action == Action::AcceptResultHeader || action == Action::AcknowledgeResult || action == Action::ConfirmAck;
    const bool publishes = action == Action::PrepareResultHeader || action == Action::AcceptResultHeader ||
        action == Action::AcknowledgeResult;
    if (!valid(args.geometry) || !args.version || !args.source_table || !args.healthy || !args.first_error ||
        (source && !args.source_frame) || (receives && !args.received_bytes) ||
        ((action == Action::AcceptResultHeader || action == Action::ConfirmAck) && !args.received) ||
        (publishes && !args.publication) ||
        (action == Action::PrepareResult && !args.published_bytes) ||
        (action == Action::AcknowledgeResult && !args.payload_completion.valid()))
        throw std::invalid_argument("Replicated PLE reader requires its complete original operands");
    const auto stream = static_cast<QFN_NATIVE_STREAM>(exact_stream);
#define QFN_READER_CASE(Name) case Action::Name: \
    QFN_NATIVE_NAMESPACE::replicatedReader<Action::Name><<<1, 1, 0, stream>>>(args); break
    switch (action) {
        QFN_READER_CASE(PrepareResultHeader);
        QFN_READER_CASE(AcceptResultHeader);
        QFN_READER_CASE(PrepareResult);
        QFN_READER_CASE(AcknowledgeResult);
        QFN_READER_CASE(ConfirmAck);
        default: throw std::invalid_argument("Unknown replicated PLE reader phase");
    }
#undef QFN_READER_CASE
    QFN_NATIVE_NAMESPACE::checked();
}
#undef QFN_READER_ENQUEUE
} // namespace llaminar2::qwen4exp::replicated_reader
