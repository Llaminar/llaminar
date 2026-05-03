/**
 * @file GPUExpertTransfer.cpp
 * @brief GPU↔GPU direct expert weight transfer implementation.
 *
 * Uses HIP peer-to-peer DMA (hipMemcpyPeerAsync) for ROCm↔ROCm transfers.
 * Falls back to host-staged copy when P2P is not available.
 */

#include "GPUExpertTransfer.h"
#include "../../utils/Logger.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2 {

#ifdef HAVE_ROCM

namespace {

/// Copy one array from src_device to dst_device using peer DMA or host-staged fallback.
/// Returns true on success.
bool copyArrayPeerAsync(
    void* dst, int dst_ordinal,
    const void* src, int src_ordinal,
    size_t bytes, hipStream_t stream,
    bool peer_available)
{
    if (bytes == 0 || dst == nullptr || src == nullptr)
        return true;  // nothing to transfer

    if (peer_available) {
        hipError_t err = hipMemcpyPeerAsync(dst, dst_ordinal, src, src_ordinal, bytes, stream);
        if (err != hipSuccess) {
            LOG_ERROR("[GPUExpertTransfer] hipMemcpyPeerAsync failed: " << hipGetErrorString(err)
                      << " (src=" << src_ordinal << " dst=" << dst_ordinal << " bytes=" << bytes << ")");
            return false;
        }
        return true;
    }

    // Fallback: hipMemcpyPeer handles host-staging internally and is much faster
    // than manual bounce buffer allocation per array.
    hipError_t err = hipMemcpyPeer(dst, dst_ordinal, src, src_ordinal, bytes);
    if (err != hipSuccess) {
        LOG_ERROR("[GPUExpertTransfer] hipMemcpyPeer fallback failed: " << hipGetErrorString(err)
                  << " (src=" << src_ordinal << " dst=" << dst_ordinal << " bytes=" << bytes << ")");
        return false;
    }
    return true;
}

} // anonymous namespace

bool GPUExpertTransfer::transferExpert(
    const GPUExpertPointers& src_ptrs,
    const GPUExpertPointers& dst_ptrs,
    const DeviceId& src_device,
    const DeviceId& dst_device,
    size_t vnni_bytes,
    size_t scales_bytes,
    size_t mins_bytes,
    size_t emins_bytes,
    void* stream)
{
    if (!src_device.is_rocm() || !dst_device.is_rocm()) {
        LOG_ERROR("[GPUExpertTransfer] Both devices must be ROCm, got "
                  << src_device.to_string() << " → " << dst_device.to_string());
        return false;
    }

    // Save caller's active device to restore on exit
    int original_device = -1;
    hipGetDevice(&original_device);

    const int src_ord = src_device.rocm_ordinal();
    const int dst_ord = dst_device.rocm_ordinal();
    const bool peer = canAccessPeer(src_ord, dst_ord);

    if (peer) {
        enablePeerAccess(dst_ord);
    }

    auto hip_stream = static_cast<hipStream_t>(stream);

    // Transfer all arrays
    bool success = true;
    if (!copyArrayPeerAsync(dst_ptrs.d_vnni, dst_ord,
                            src_ptrs.d_vnni, src_ord,
                            vnni_bytes, hip_stream, peer))
        success = false;

    if (success && !copyArrayPeerAsync(dst_ptrs.d_scales, dst_ord,
                            src_ptrs.d_scales, src_ord,
                            scales_bytes, hip_stream, peer))
        success = false;

    if (success && mins_bytes > 0) {
        if (!copyArrayPeerAsync(dst_ptrs.d_mins, dst_ord,
                                src_ptrs.d_mins, src_ord,
                                mins_bytes, hip_stream, peer))
            success = false;
    }

    if (success && emins_bytes > 0) {
        if (!copyArrayPeerAsync(dst_ptrs.d_emins, dst_ord,
                                src_ptrs.d_emins, src_ord,
                                emins_bytes, hip_stream, peer))
            success = false;
    }

    // Restore caller's original device context
    if (original_device >= 0) {
        hipSetDevice(original_device);
    }

    if (success) {
        LOG_DEBUG("[GPUExpertTransfer] Transferred expert ROCm:" << src_ord
                  << " → ROCm:" << dst_ord
                  << " vnni=" << vnni_bytes << "B scales=" << scales_bytes << "B"
                  << " mins=" << mins_bytes << "B emins=" << emins_bytes << "B"
                  << (peer ? " (P2P)" : " (host-staged)"));
    }
    return success;
}

bool GPUExpertTransfer::canAccessPeer(int src_ordinal, int dst_ordinal)
{
    if (src_ordinal == dst_ordinal)
        return true;

    int can_access = 0;
    hipError_t err = hipDeviceCanAccessPeer(&can_access, src_ordinal, dst_ordinal);
    return (err == hipSuccess && can_access != 0);
}

bool GPUExpertTransfer::enablePeerAccess(int peer_ordinal)
{
    hipError_t err = hipDeviceEnablePeerAccess(peer_ordinal, 0);
    // hipErrorPeerAccessAlreadyEnabled is fine — idempotent
    return (err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled);
}

#else // !HAVE_ROCM

bool GPUExpertTransfer::transferExpert(
    const GPUExpertPointers& /*src_ptrs*/,
    const GPUExpertPointers& /*dst_ptrs*/,
    const DeviceId& /*src_device*/,
    const DeviceId& /*dst_device*/,
    size_t /*vnni_bytes*/,
    size_t /*scales_bytes*/,
    size_t /*mins_bytes*/,
    size_t /*emins_bytes*/,
    void* /*stream*/)
{
    LOG_ERROR("[GPUExpertTransfer] ROCm not available");
    return false;
}

bool GPUExpertTransfer::canAccessPeer(int /*src_ordinal*/, int /*dst_ordinal*/)
{
    return false;
}

bool GPUExpertTransfer::enablePeerAccess(int /*peer_ordinal*/)
{
    return false;
}

#endif // HAVE_ROCM

} // namespace llaminar2
