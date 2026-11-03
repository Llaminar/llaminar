/**
 * @file PooledImmutableReplicatedReaderKernels.h
 * @brief Exact-stream metadata kernels for original replicated PLE reader edges.
 *
 * Graph-owned operand banks and original channel owners retain every address.
 * This stateless code interface owns no source, memory, sequence or submission.
 * The enclosing stage authenticates original arena/control/count identities.
 */
#pragma once
#include "PooledImmutableReplicatedReaderOps.h"
#include "kernels/common/DeviceRowRange.h"
#include "backends/DeviceId.h"

namespace llaminar2::qwen4exp::replicated_reader {
/** @brief Fixed metadata producers and validators surrounding explicit transfer nodes. */
enum class Action : std::uint32_t {
    PrepareResultHeader, AcceptResultHeader, PrepareResult, AcknowledgeResult, ConfirmAck
};
/** @brief Addresses sealed by the actual participant-local stage before recording. */
struct NativeArgs {
    DeviceId device;
    Geometry geometry;
    std::uint32_t version;
    std::uint64_t source_table;
    DeviceRowRange original_rows;
    const p::LookupFrame* source_frame;
    const Header* received;
    Header* publication;
    const std::uint64_t* received_bytes;
    std::uint64_t* published_bytes;
    int* healthy;
    int* first_error;
    CapturedTransferReceiveCompletion payload_completion;
};
/** @brief Submit only immutable metadata work on its original non-default CUDA stream. */
void enqueueCUDAPooledReplicatedReader(Action action, NativeArgs args, void* exact_stream);
/** @brief Submit the same metadata arithmetic on its original non-default HIP stream. */
void enqueueROCmPooledReplicatedReader(Action action, NativeArgs args, void* exact_stream);
} // namespace llaminar2::qwen4exp::replicated_reader
