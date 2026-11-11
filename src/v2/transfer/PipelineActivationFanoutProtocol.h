/**
 * @file PipelineActivationFanoutProtocol.h
 * @brief Original completed native-byte fanout metadata for a received pipeline activation.
 *
 * Only the original channel publication authors epoch/key/live bytes. A fixed
 * 24-byte native broadcast copies those necessary protocol words into each
 * participant's local metadata; the subsequent native prefix moves live payload.
 * No host mirror, scalar controller or independent request clock is introduced.
 */
#pragma once
#include "CapturedTransferChannelProtocol.h"
#include "collective/NativeCollectiveCompletion.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace llaminar2
{
/** @brief Exact useful publication words; native fanout never sends struct padding. */
struct PipelineActivationFanoutHeader final
{
    std::uint64_t epoch = 0;
    CapturedTransferMessage message;
};
static_assert(sizeof(PipelineActivationFanoutHeader) == 24 &&
    offsetof(PipelineActivationFanoutHeader, message) == offsetof(CapturedTransferPublication, message));
/** @brief Participant-private native outputs, initialized once and retained across graph families. */
struct alignas(64) PipelineActivationFanoutDeviceState final
{
    PipelineActivationFanoutHeader header;
    ncclLlaminarByteCompletion receipt{};
    std::int32_t first_error = 0; ///< Sticky native failure, never reset to conceal a failed collective.
};
static_assert(sizeof(PipelineActivationFanoutDeviceState) == 128);
/** @brief Write-once mapped original identity/operand, not a live host-side frame. */
struct PipelineActivationFanoutOperand final
{
    std::uint64_t allocation = 0; ///< Actual channel allocation nonce.
    std::uint64_t scope = 0; ///< Actual original allocation session, or nonce for a same-process channel.
    ncclLlaminarCompletionOperand native{};
};
/** @brief Exact same-stream validation inputs; all pointed state remains device-owned. */
struct PipelineActivationFanoutValidation final
{
    PipelineActivationFanoutDeviceState *state = nullptr;
    const std::int32_t *rows = nullptr;
    std::uint64_t row_bytes = 0, capacity = 0, expected_key = 0;
    std::uint64_t allocation = 0, scope = 0, native_message = 0;
    NativeReceiveCompletionView completion = NativeReceiveCompletionView::notRequired();
    CapturedTransferReceiveCompletion original_receive = CapturedTransferReceiveCompletion::notRequired();
    /** @return Whether the immutable original operands are complete before any enqueue. */
    bool valid() const noexcept
    { return state && rows && row_bytes && capacity && capacity % row_bytes == 0 &&
        expected_key && allocation && scope && native_message && completion.valid(); }
};
/** @brief Pure extent/frame law shared by native validation and device-free adversarial tests.
 * @param rows Actual freshly materialized local row count.
 * @param row_bytes Immutable original full residual row width in bytes.
 * @param capacity Admitted maximum for this captured bucket, never its live extent.
 * @param expected_key Original graph-frozen message identity.
 * @param header Original received publication, never a host reconstruction.
 * @return Whether source live extent agrees with the local invocation before model/KV execution. */
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
constexpr bool pipelineActivationFanoutHeaderMatches(std::int32_t rows, std::uint64_t row_bytes,
    std::uint64_t capacity, std::uint64_t expected_key, PipelineActivationFanoutHeader header) noexcept
{
    const auto extent = capturedTransferRowBytes(rows, row_bytes, capacity);
    return extent.valid && header.epoch && header.epoch != kCapturedTransferAbortEpoch &&
        header.message.key == expected_key && header.message.bytes == extent.bytes;
}
}
