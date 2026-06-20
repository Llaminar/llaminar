/**
 * @file GPUExpertTransfer.h
 * @brief GPU↔GPU direct expert weight transfer for MoE rebalancing.
 *
 * Provides peer-to-peer (P2P) or host-staged device-to-device transfer
 * of packed MoE expert weights between GPU devices. This is ~50x faster
 * than the serialize → MPI → deserialize → repack path for intra-node
 * GPU↔GPU transfers since both devices use identical packed weight format.
 *
 * ── Format Conversion Note (CPU ↔ GPU) ──────────────────────────────
 *
 * CPU (CPUNativeVNNIPackedWeights) and GPU (MoEBatchPackedWeightsROCm/CUDA)
 * use DIFFERENT memory layouts for the same underlying quantized bytes:
 *
 *   CPU interleaved:
 *     Per K-block across 64 rows (N-chunk): [1024B payload (4 groups × 4 ZMMs,
 *     transposed)] [128B comp_int16] [128B scales_fp16] [128B mins_fp16]
 *     - Payload bytes are transposed into VNNI register layout for AVX-512
 *     - stride = 1280B (symmetric) or 1408B (asymmetric)
 *
 *   GPU separated:
 *     Flat arrays: all_native_vnni[linear_block * payload_bytes], one scale per block,
 *     one min per block. Payload bytes are in per-row linear order.
 *
 * Metadata granularity is IDENTICAL (1 FP16 scale, 1 FP16 min per 32-element block).
 * The payload bytes are the SAME raw quantized bytes, but stored in different
 * transposition order (VNNI group/ZMM/lane layout vs. linear per-row). Conversion
 * is a pure byte-level scatter/gather (no mathematical transformation), but the
 * transposition across 64-row N-chunks adds non-trivial complexity.
 *
 * GPU↔GPU transfer does NOT require format conversion since both ROCm (and CUDA)
 * devices use the identical separated layout. CPU↔GPU transfer requires the
 * existing serialize → deserialize → repack path until deinterleave kernels are
 * implemented.
 */

#pragma once

#include "../../backends/DeviceId.h"

#include <cstddef>
#include <cstdint>

namespace llaminar2 {

/// Pointers to one expert's packed weight arrays on a GPU device.
struct GPUExpertPointers {
    uint8_t* d_vnni = nullptr;
    void* d_scales = nullptr;    // uint16_t* (FP16)
    void* d_mins = nullptr;      // uint16_t* (FP16), nullptr if symmetric
    void* d_emins = nullptr;     // uint32_t*, nullptr if not present
};

/// Transfer expert weights between GPU devices via peer DMA or host-staged copy.
///
/// Works for ROCm↔ROCm (hipMemcpyPeer). Both source and destination must use
/// the same packed weight format (MoEBatchPackedWeightsROCm or CUDA equivalent).
class GPUExpertTransfer {
public:
    /// Transfer one expert's packed weights from src to dst device.
    /// Both src and dst must be ROCm devices (or both CUDA).
    /// @param src_ptrs Source device pointers (on src_device)
    /// @param dst_ptrs Destination device pointers (pre-allocated on dst_device)
    /// @param src_device Source device
    /// @param dst_device Destination device
    /// @param vnni_bytes Size of vnni array for this expert
    /// @param scales_bytes Size of scales array for this expert (in bytes)
    /// @param mins_bytes Size of mins array in bytes (0 if symmetric)
    /// @param emins_bytes Size of emins array in bytes (0 if not present)
    /// @param stream Stream on dst_device for async transfer (nullptr = sync)
    /// @return true on success
    static bool transferExpert(
        const GPUExpertPointers& src_ptrs,
        const GPUExpertPointers& dst_ptrs,
        const DeviceId& src_device,
        const DeviceId& dst_device,
        size_t vnni_bytes,
        size_t scales_bytes,
        size_t mins_bytes,
        size_t emins_bytes,
        void* stream);

    /// Check if peer-to-peer access is available between two ROCm devices.
    static bool canAccessPeer(int src_ordinal, int dst_ordinal);

    /// Enable peer access from current device to peer device.
    /// Safe to call multiple times (handles already-enabled case).
    static bool enablePeerAccess(int peer_ordinal);
};

} // namespace llaminar2
