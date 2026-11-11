/**
 * @file NativePipelineByteFrame.h
 * @brief Exact native pipeline extent projection over the original completed-byte frame.
 *
 * Each addressed endpoint owns one admitted frame. The existing native frame
 * remains the only invocation/error/receipt authority. Live signed prefill rows
 * come from the original materializer or the producer's completed native
 * receipt. Capacity only bounds storage and can never become live wire bytes.
 */
#pragma once
#include "NativeCountedAllGatherFrame.h"
#include "transfer/CapturedTransferChannelProtocol.h"
#include "kernels/common/NativeCounter64.h"
#include <limits>
namespace llaminar2 {
/** @brief Cold initialization and the two captured endpoint operations. */
enum class NativePipelineByteOperation : std::uint8_t { Initialize, Prepare, Validate };
/** @brief Optional original terminal-only send and completed receipt destinations. */
struct NativePipelineByteObservation {
    unsigned long long *outgoing = nullptr;
    NativeCounter64 completions;
};
/** @brief Capture-frozen owner and exact live-extent inputs for one native edge. */
struct NativePipelineByteInvocation final {
    void *memory = nullptr; ///< Exact NativeCountedAllGatherLayout(1) allocation.
    std::uint64_t key = 0, capacity = 0;
    const std::int32_t *rows = nullptr; ///< Existing source or independently expected consumer rows.
    std::uint64_t row_bytes = 0;
    ncclLlaminarCompletionAuthority authority{};
    std::int32_t *received_rows = nullptr; ///< Storage consumer's transport output, derived only from its completed receipt.
    NativePipelineByteObservation observation{};
    /** @return Whether setup retained a complete, non-overflowing native operand. */
    bool valid() const noexcept {
        if (!memory || !key || !capacity || !authority.abortFlag || !authority.deviceCommunicator ||
            (rows && received_rows) || (observation.outgoing &&
                (!observation.completions.valid() || reinterpret_cast<std::uintptr_t>(observation.outgoing) % alignof(unsigned long long))) ||
            (observation.completions.bound() && !observation.completions.valid())) return false;
        if (!rows && !received_rows) return row_bytes == 0;
        return row_bytes && capacity % row_bytes == 0 &&
            capacity / row_bytes <= std::uint64_t{std::numeric_limits<std::int32_t>::max()};
    }
};
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_PIPELINE_BYTE_HD __host__ __device__
#else
#define LLAMINAR_PIPELINE_BYTE_HD
#endif
/** @brief Project useful rows or arm a receive whose extent the original producer publishes. */
LLAMINAR_PIPELINE_BYTE_HD inline bool prepareNativePipelineByteInvocation(NativePipelineByteInvocation a) {
    auto *frame = static_cast<NativeCountedAllGatherFrame *>(a.memory);
    if (a.received_rows) *a.received_rows = -1;
    const auto extent = a.rows ? capturedTransferRowBytes(*a.rows, a.row_bytes, a.capacity)
                              : CapturedTransferRowBytes{true, a.received_rows ? 0 : a.capacity};
    if (!extent.valid) { if (!frame->first_error) frame->first_error = 1; return false; }
    if (!prepareNativeCountedAllGatherInvocation(a.memory, 1, a.key, a.capacity)) return false;
    auto *bytes = reinterpret_cast<std::uint64_t *>(frame + 1);
    *bytes = extent.bytes;
    if (a.observation.outgoing)
        reinterpret_cast<ncclLlaminarCompletionOperand *>(bytes + 1)->outgoingPayloadBytes = a.observation.outgoing;
    return true;
}
/** @brief Prove current native completion, then validate or publish the exact rows for this consumer. */
LLAMINAR_PIPELINE_BYTE_HD inline bool completeNativePipelineByteInvocation(
    NativePipelineByteInvocation a, std::uint32_t aborted) {
    auto *frame = static_cast<NativeCountedAllGatherFrame *>(a.memory);
    const auto received = *reinterpret_cast<const std::uint64_t *>(frame + 1);
    const auto observe = [&] {
        return !a.observation.completions.bound() || addNativeCounter64(a.observation.completions, 1);
    };
    if (completeNativeCountedAllGatherInvocation(a.memory, 1, a.authority, aborted)) {
        if (a.received_rows) {
            // The native fixed header owns this extent. The receive payload
            // and fresh receipt are both complete before downstream storage
            // work can consume any published row count.
            if (*a.received_rows == -1 && a.row_bytes && received <= a.capacity && received % a.row_bytes == 0 && observe()) {
                *a.received_rows = static_cast<std::int32_t>(received / a.row_bytes);
                return true;
            }
        } else {
            const auto expected = a.rows ? capturedTransferRowBytes(*a.rows, a.row_bytes, a.capacity)
                                         : CapturedTransferRowBytes{true, a.capacity};
            if (expected.valid && received == expected.bytes && observe()) return true;
        }
    }
    if (a.received_rows) *a.received_rows = -1;
    if (!frame->first_error) frame->first_error = 1;
    return false;
}
#undef LLAMINAR_PIPELINE_BYTE_HD
#if defined(__CUDACC__) || defined(__HIPCC__)
/** @brief One exact scalar node prepares original native metadata without allocation/readback. */
static __global__ void prepareNativePipelineByteFrame(NativePipelineByteInvocation a) {
    if (!prepareNativePipelineByteInvocation(a)) {
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }
}
/** @brief One exact scalar node rejects stale, aborted, padded or mismatched native payloads. */
static __global__ void validateNativePipelineByteFrame(NativePipelineByteInvocation a) {
    std::uint32_t aborted;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(aborted) : "l"(a.authority.abortFlag) : "memory");
#elif defined(__HIP_DEVICE_COMPILE__)
    aborted = __atomic_load_n(a.authority.abortFlag, __ATOMIC_ACQUIRE);
#else
    aborted = *a.authority.abortFlag;
#endif
    if (!completeNativePipelineByteInvocation(a, aborted)) {
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }
}
#endif
} // namespace llaminar2
